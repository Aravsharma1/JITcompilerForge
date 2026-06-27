"""Autotuner result models."""

from __future__ import annotations

from dataclasses import dataclass

from forge.kernels.kernel_configs import KernelConfig


@dataclass(frozen=True)
class TuningResult:
    """Best candidate selected for a workload."""

    config: KernelConfig
    latency_ms: float
    tokens_per_sec: float
    validated: bool = True
