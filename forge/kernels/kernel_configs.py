"""Compile-time settings for decode kernel variants."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class KernelConfig:
    """A candidate configuration for a decode attention kernel."""

    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


DEFAULT_KERNEL_CONFIG = KernelConfig(
    block_m=16,
    block_n=64,
    block_k=64,
    num_warps=4,
    num_stages=3,
)


SEARCH_SPACE: tuple[KernelConfig, ...] = (
    KernelConfig(8, 32, 64, 4, 3),
    KernelConfig(8, 64, 64, 4, 3),
    KernelConfig(16, 64, 64, 4, 3),
    KernelConfig(16, 128, 64, 4, 3),
    KernelConfig(32, 64, 64, 8, 4),
    KernelConfig(32, 128, 64, 8, 4),
    KernelConfig(16, 64, 128, 4, 4),
    KernelConfig(32, 128, 128, 8, 4),
)
