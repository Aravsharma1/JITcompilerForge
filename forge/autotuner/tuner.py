"""Profile-guided decode-kernel autotuner."""

from __future__ import annotations

from forge.autotuner.benchmark import benchmark_candidate
from forge.autotuner.candidate import TuningResult
from forge.autotuner.search import candidate_configs
from forge.profiler.workload_spec import WorkloadSpec
from forge.utils.buckets import bucket_midpoint


class Autotuner:
    """Searches for the fastest kernel config for a workload spec."""

    def __init__(self, candidate_limit: int = 8) -> None:
        self.candidate_limit = candidate_limit

    def tune(self, spec: WorkloadSpec) -> TuningResult:
        batch_size = bucket_midpoint(spec.batch_size_bucket)
        seq_len = bucket_midpoint(spec.seq_len_bucket)
        results = [
            benchmark_candidate(config, spec, batch_size, seq_len)
            for config in candidate_configs(self.candidate_limit)
        ]
        return max(results, key=lambda result: result.tokens_per_sec)
