"""Workload descriptions used to specialize decode kernels."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkloadSpec:
    """A bucketed description of the current decode workload."""

    batch_size_bucket: str
    seq_len_bucket: str
    model_config_hash: str
    window_size: int

    @property
    def cache_key(self) -> str:
        return "|".join(
            [
                self.model_config_hash,
                self.batch_size_bucket,
                self.seq_len_bucket,
            ]
        )


def bucket_value(value: int, buckets: tuple[int, ...]) -> str:
    """Return a stable label for the bucket containing value."""

    if value < 1:
        raise ValueError("bucketed values must be positive")

    lower = 1
    for upper in buckets:
        if value <= upper:
            return f"{lower}-{upper}"
        lower = upper + 1

    return f"{lower}+"
