"""JSON-backed cache for tuned kernel configurations."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from time import time

from forge.autotuner.candidate import TuningResult
from forge.kernels.kernel_configs import KernelConfig
from forge.profiler.workload_spec import WorkloadSpec


class KernelCache:
    """Stores best-known kernel configs for workload buckets."""

    def __init__(self, path: str | Path = "results/raw/kernel_cache.json") -> None:
        self.path = Path(path)
        self._entries: dict[str, dict] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._entries = {}
            return
        self._entries = json.loads(self.path.read_text())

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._entries, indent=2, sort_keys=True))

    def get(self, spec: WorkloadSpec) -> TuningResult | None:
        entry = self._entries.get(spec.cache_key)
        if not entry:
            return None
        config = KernelConfig(**entry["config"])
        return TuningResult(
            config=config,
            latency_ms=float(entry["latency_ms"]),
            tokens_per_sec=float(entry["tokens_per_sec"]),
            validated=bool(entry["validated"]),
        )

    def put(self, spec: WorkloadSpec, result: TuningResult) -> None:
        self._entries[spec.cache_key] = {
            "config": asdict(result.config),
            "latency_ms": result.latency_ms,
            "tokens_per_sec": result.tokens_per_sec,
            "validated": result.validated,
            "timestamp": time(),
        }
        self.save()
