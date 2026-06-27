"""Decode attention kernel abstraction.

This first implementation is intentionally a CPU-side simulation. The class has
the same lifecycle as a real Triton kernel variant: it is created from compile
time parameters, launched for a decode step, and returns timing-like metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

from forge.kernels.kernel_configs import KernelConfig
from forge.profiler.workload_spec import WorkloadSpec


@dataclass(frozen=True)
class DecodeKernel:
    """Compiled decode kernel placeholder."""

    name: str
    config: KernelConfig

    def estimate_latency_ms(self, batch_size: int, seq_len: int) -> float:
        """Return deterministic simulated latency for this kernel."""

        memory_work = batch_size * seq_len
        tile_efficiency = self.config.block_n * self.config.num_warps
        pipeline_bonus = 1.0 + (0.05 * self.config.num_stages)
        overhead = 0.2 + (self.config.block_m / 256.0)
        return overhead + (memory_work / max(tile_efficiency * pipeline_bonus, 1)) / 1000.0

    def launch(self, batch_size: int, seq_len: int) -> dict[str, float | str]:
        latency_ms = self.estimate_latency_ms(batch_size, seq_len)
        tokens_per_sec = (batch_size * 1000.0) / latency_ms
        return {
            "kernel": self.name,
            "latency_ms": latency_ms,
            "tokens_per_sec": tokens_per_sec,
        }


def compile_decode_kernel(config: KernelConfig, spec: WorkloadSpec | None = None) -> DecodeKernel:
    """Compile a decode kernel variant.

    Real Triton integration will replace this with a @triton.jit template. For
    now, the function gives the rest of the system a concrete kernel object.
    """

    suffix = spec.cache_key if spec else "default"
    return DecodeKernel(name=f"decode_attention[{suffix}]", config=config)
