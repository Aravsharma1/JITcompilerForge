"""Step-boundary kernel hot-swap manager."""

from __future__ import annotations

from dataclasses import dataclass

from forge.kernels.decode_attention import DecodeKernel


@dataclass
class SwapManager:
    """Keeps active and staging kernels separate until a safe boundary."""

    active_kernel: DecodeKernel
    staging_kernel: DecodeKernel | None = None

    def stage(self, kernel: DecodeKernel) -> None:
        self.staging_kernel = kernel

    def swap_at_step_boundary(self) -> bool:
        if self.staging_kernel is None:
            return False
        self.active_kernel = self.staging_kernel
        self.staging_kernel = None
        return True
