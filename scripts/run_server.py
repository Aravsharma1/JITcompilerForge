"""Run a tiny Forge serving-loop simulation."""

from __future__ import annotations

import sys
from pathlib import Path
from time import sleep

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge.autotuner.tuner import Autotuner
from forge.cache.kernel_cache import KernelCache
from forge.profiler.profiler import RuntimeProfiler
from forge.serving.loop import ServingLoop


def main() -> None:
    profiler = RuntimeProfiler(model_config_hash="toy_llm_head128", window_size=16)
    cache = KernelCache()
    workload = [(8, 768)] * 10 + [(4, 4096)] * 10
    with ServingLoop(
        profiler=profiler,
        autotuner=Autotuner(candidate_limit=8),
        cache=cache,
        tune_every_steps=8,
    ) as loop:
        for batch_size, seq_len in workload:
            result = loop.decode_step(batch_size=batch_size, seq_len=seq_len)
            marker = " swapped" if result.swapped else ""
            print(
                f"step={result.step:02d} kernel={result.kernel} "
                f"latency_ms={result.latency_ms:.3f} "
                f"tokens_per_sec={result.tokens_per_sec:.1f}{marker}"
            )
            # Yield between simulated requests so the background tuner can run.
            sleep(0.001)


if __name__ == "__main__":
    main()
