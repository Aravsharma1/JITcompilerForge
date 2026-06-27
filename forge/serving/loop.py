"""Minimal serving-loop simulation for the Forge architecture."""

from __future__ import annotations

from dataclasses import dataclass

from forge.autotuner.tuner import Autotuner
from forge.cache.kernel_cache import KernelCache
from forge.hotswap.swap_manager import SwapManager
from forge.kernels.decode_attention import compile_decode_kernel
from forge.kernels.kernel_configs import DEFAULT_KERNEL_CONFIG
from forge.profiler.profiler import DecodeStepMetrics, RuntimeProfiler


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

        if self.step % self.tune_every_steps == 0:
            self._maybe_stage_optimized_kernel()

        swapped = self.swap_manager.swap_at_step_boundary()
        return DecodeStepResult(
            step=self.step,
            kernel=str(metrics["kernel"]),
            latency_ms=float(metrics["latency_ms"]),
            tokens_per_sec=float(metrics["tokens_per_sec"]),
            swapped=swapped,
        )

    def _maybe_stage_optimized_kernel(self) -> None:
        spec = self.profiler.current_spec()
        if spec is None:
            return

        result = self.cache.get(spec)
        if result is None:
            result = self.autotuner.tune(spec)
            self.cache.put(spec, result)

        if result.validated:
            self.swap_manager.stage(compile_decode_kernel(result.config, spec))
