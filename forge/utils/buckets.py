"""Bucket parsing helpers."""

from __future__ import annotations


def bucket_midpoint(label: str) -> int:
    """Return a representative integer for a bucket label."""

    if label.endswith("+"):
        return int(label[:-1])
    lower, upper = label.split("-", maxsplit=1)
    return (int(lower) + int(upper)) // 2
