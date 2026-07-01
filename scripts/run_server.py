"""Run a tiny Forge serving-loop simulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import sleep

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from forge.autotuner.tuner import Autotuner
from forge.cache.kernel_cache import KernelCache
from forge.config import load_config
from forge.profiler.profiler import RuntimeProfiler
from forge.serving.loop import ServingLoop


def main(config_path: str | Path | None = None) -> None:
    config = load_config(config_path or PROJECT_ROOT / "configs/default.yaml")
    profiler = RuntimeProfiler(
        model_config_hash=config.model_config_hash,
        window_size=config.profiler_window_size,
    )
    cache = KernelCache()
    workload = [(8, 768)] * 10 + [(4, 4096)] * 10
    with ServingLoop(
        profiler=profiler,
        autotuner=Autotuner(candidate_limit=config.autotuner_candidate_limit),
        cache=cache,
        tune_every_steps=config.tune_every_steps,
        minimum_speedup_percent=config.minimum_speedup_percent,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/default.yaml",
        help="path to a Forge YAML configuration",
    )
    args = parser.parse_args()
    main(args.config)
