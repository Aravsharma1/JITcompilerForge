"""Candidate generation for kernel autotuning."""

from __future__ import annotations

from forge.kernels.kernel_configs import SEARCH_SPACE, KernelConfig


def candidate_configs(limit: int = 8) -> tuple[KernelConfig, ...]:
    """Return a small deterministic search set."""

    if limit < 1:
        raise ValueError("limit must be positive")
    return SEARCH_SPACE[:limit]
