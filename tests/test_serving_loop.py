from pathlib import Path

from forge.autotuner.tuner import Autotuner
from forge.cache.kernel_cache import KernelCache
from forge.profiler.profiler import RuntimeProfiler
from forge.serving.loop import ServingLoop


def test_serving_loop_stages_and_swaps_kernel(tmp_path: Path) -> None:
    loop = ServingLoop(
        profiler=RuntimeProfiler(model_config_hash="toy", window_size=4),
        autotuner=Autotuner(candidate_limit=4),
        cache=KernelCache(tmp_path / "cache.json"),
        tune_every_steps=4,
    )

    results = [loop.decode_step(batch_size=8, seq_len=900) for _ in range(4)]

    assert results[-1].swapped is True
    assert loop.swap_manager.active_kernel.name.startswith("decode_attention[toy|")
