"""Minimal serving-loop simulation for the Forge architecture."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from forge.autotuner.candidate import TuningResult
from forge.autotuner.tuner import Autotuner
from forge.cache.kernel_cache import KernelCache
from forge.hotswap.swap_manager import SwapManager
from forge.kernels.decode_attention import compile_decode_kernel
from forge.kernels.kernel_configs import DEFAULT_KERNEL_CONFIG
from forge.profiler.profiler import DecodeStepMetrics, RuntimeProfiler
from forge.profiler.workload_spec import WorkloadSpec


@dataclass(frozen=True)
class DecodeStepResult:
    step: int
    kernel: str
    latency_ms: float
    tokens_per_sec: float
    swapped: bool


class ServingLoop:
    """Runs decode steps and performs profile-guided kernel swaps."""

    def __init__(
        self,
        profiler: RuntimeProfiler,
        autotuner: Autotuner,
        cache: KernelCache,
        tune_every_steps: int = 8,
    ) -> None:
        default_kernel = compile_decode_kernel(DEFAULT_KERNEL_CONFIG)
        self.profiler = profiler
        self.autotuner = autotuner
        self.cache = cache
        self.tune_every_steps = tune_every_steps
        self.swap_manager = SwapManager(active_kernel=default_kernel)
        self.step = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="forge-tuner")
        self._tuning_job: Future[TuningResult] | None = None
        self._tuning_spec: WorkloadSpec | None = None
        self.last_tuning_error: Exception | None = None

    def decode_step(self, batch_size: int, seq_len: int) -> DecodeStepResult:
        self.step += 1
        metrics = self.swap_manager.active_kernel.launch(batch_size, seq_len)
        self.profiler.record_step(
            DecodeStepMetrics(
                batch_size=batch_size,
                seq_len=seq_len,
                latency_ms=float(metrics["latency_ms"]),
            )
        )

        self._collect_tuning_result()
        if self.step % self.tune_every_steps == 0:
            self._maybe_start_tuning()

        swapped = self.swap_manager.swap_at_step_boundary()
        return DecodeStepResult(
            step=self.step,
            kernel=str(metrics["kernel"]),
            latency_ms=float(metrics["latency_ms"]),
            tokens_per_sec=float(metrics["tokens_per_sec"]),
            swapped=swapped,
        )

    @property
    def tuning_in_progress(self) -> bool:
        return self._tuning_job is not None

    def wait_for_tuning(self, timeout: float | None = None) -> bool:
        """Wait for an in-flight tuning job and stage its validated result."""

        if self._tuning_job is None:
            return False
        try:
            self._tuning_job.result(timeout=timeout)
        finally:
            self._collect_tuning_result()
        return self.swap_manager.staging_kernel is not None

    def close(self) -> None:
        """Release the background tuning worker."""

        self._executor.shutdown(wait=True)
        self._collect_tuning_result()

    def __enter__(self) -> ServingLoop:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _maybe_start_tuning(self) -> None:
        if self._tuning_job is not None:
            return

        spec = self.profiler.current_spec()
        if spec is None:
            return

        result = self.cache.get(spec)
        if result is not None:
            self._stage_result(spec, result)
            return

        self._tuning_spec = spec
        self._tuning_job = self._executor.submit(self._tune_and_cache, spec)

    def _tune_and_cache(self, spec: WorkloadSpec) -> TuningResult:
        result = self.autotuner.tune(spec)
        self.cache.put(spec, result)
        return result

    def _collect_tuning_result(self) -> None:
        if self._tuning_job is None or not self._tuning_job.done():
            return

        job = self._tuning_job
        spec = self._tuning_spec
        self._tuning_job = None
        self._tuning_spec = None
        try:
            result = job.result()
        except Exception as error:
            self.last_tuning_error = error
            return

        self.last_tuning_error = None
        if spec is not None:
            self._stage_result(spec, result)

    def _stage_result(self, spec: WorkloadSpec, result: TuningResult) -> None:
        if result.validated:
            kernel = compile_decode_kernel(result.config, spec)
            if kernel != self.swap_manager.active_kernel:
                self.swap_manager.stage(kernel)
