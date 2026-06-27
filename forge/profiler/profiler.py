"""Rolling-window profiler for decode-step metrics."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from statistics import median

from forge.profiler.workload_spec import WorkloadSpec, bucket_value


@dataclass(frozen=True)
class DecodeStepMetrics:
    """Metrics emitted by one decode step."""

    batch_size: int
    seq_len: int
    latency_ms: float


class RuntimeProfiler:
    """Summarizes recent decode steps into workload buckets."""

    def __init__(
        self,
        model_config_hash: str,
        window_size: int = 32,
        batch_buckets: tuple[int, ...] = (1, 4, 8, 16, 32, 64),
        seq_len_buckets: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096, 8192),
    ) -> None:
        self.model_config_hash = model_config_hash
        self.window_size = window_size
        self.batch_buckets = batch_buckets
        self.seq_len_buckets = seq_len_buckets
        self._steps: deque[DecodeStepMetrics] = deque(maxlen=window_size)

    def record_step(self, metrics: DecodeStepMetrics) -> None:
        if metrics.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if metrics.seq_len < 1:
            raise ValueError("seq_len must be positive")
        if metrics.latency_ms <= 0:
            raise ValueError("latency_ms must be positive")
        self._steps.append(metrics)

    def current_spec(self) -> WorkloadSpec | None:
        if not self._steps:
            return None

        batch_bucket = self._most_common_bucket(
            [step.batch_size for step in self._steps],
            self.batch_buckets,
        )
        seq_len_bucket = self._most_common_bucket(
            [step.seq_len for step in self._steps],
            self.seq_len_buckets,
        )
        return WorkloadSpec(
            batch_size_bucket=batch_bucket,
            seq_len_bucket=seq_len_bucket,
            model_config_hash=self.model_config_hash,
            window_size=len(self._steps),
        )

    def latency_p50_ms(self) -> float | None:
        if not self._steps:
            return None
        return float(median(step.latency_ms for step in self._steps))

    @staticmethod
    def _most_common_bucket(values: list[int], buckets: tuple[int, ...]) -> str:
        labels = [bucket_value(value, buckets) for value in values]
        return Counter(labels).most_common(1)[0][0]
