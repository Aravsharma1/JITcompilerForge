"""Candidate benchmarking helpers."""

from __future__ import annotations

from forge.autotuner.candidate import TuningResult
from forge.kernels.decode_attention import compile_decode_kernel
from forge.kernels.kernel_configs import KernelConfig
from forge.profiler.workload_spec import WorkloadSpec


def benchmark_candidate(
    config: KernelConfig,
    spec: WorkloadSpec,
    representative_batch_size: int,
    representative_seq_len: int,
) -> TuningResult:
    """Benchmark a candidate with deterministic simulated decode work."""

    kernel = compile_decode_kernel(config, spec)
    metrics = kernel.launch(representative_batch_size, representative_seq_len)
    return TuningResult(
        config=config,
        latency_ms=float(metrics["latency_ms"]),
        tokens_per_sec=float(metrics["tokens_per_sec"]),
        validated=True,
    )
