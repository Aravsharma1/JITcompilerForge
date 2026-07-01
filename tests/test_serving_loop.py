from threading import Event
from pathlib import Path

from forge.autotuner.candidate import TuningResult
from forge.autotuner.tuner import Autotuner
from forge.cache.kernel_cache import KernelCache
from forge.kernels.kernel_configs import KernelConfig
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

    assert results[-1].swapped is False
    assert loop.wait_for_tuning(timeout=1.0) is True

    result = loop.decode_step(batch_size=8, seq_len=900)

    assert result.swapped is True
    assert loop.swap_manager.active_kernel.name.startswith("decode_attention[toy|")
    loop.close()


class BlockingAutotuner(Autotuner):
    def __init__(self, started: Event, release: Event) -> None:
        super().__init__(candidate_limit=1)
        self.started = started
        self.release = release

    def tune(self, spec):
        self.started.set()
        self.release.wait(timeout=1.0)
        return super().tune(spec)


def test_decode_continues_while_tuning_runs_in_background(tmp_path: Path) -> None:
    started = Event()
    release = Event()
    loop = ServingLoop(
        profiler=RuntimeProfiler(model_config_hash="toy", window_size=1),
        autotuner=BlockingAutotuner(started, release),
        cache=KernelCache(tmp_path / "cache.json"),
        tune_every_steps=1,
    )

    first = loop.decode_step(batch_size=1, seq_len=128)

    assert first.swapped is False
    assert started.wait(timeout=1.0)
    assert loop.tuning_in_progress

    second = loop.decode_step(batch_size=1, seq_len=128)

    assert second.step == 2
    assert second.swapped is False

    release.set()
    assert loop.wait_for_tuning(timeout=1.0)
    loop.close()


class RegressingAutotuner(Autotuner):
    def tune(self, spec):
        return TuningResult(
            config=KernelConfig(8, 32, 64, 4, 3),
            latency_ms=10.0,
            tokens_per_sec=100.0,
        )


def test_serving_loop_rejects_kernel_below_speedup_threshold(
    tmp_path: Path,
) -> None:
    loop = ServingLoop(
        profiler=RuntimeProfiler(model_config_hash="toy", window_size=1),
        autotuner=RegressingAutotuner(),
        cache=KernelCache(tmp_path / "cache.json"),
        tune_every_steps=1,
        minimum_speedup_percent=2.0,
    )

    result = loop.decode_step(batch_size=1, seq_len=128)

    assert result.swapped is False
    assert loop.wait_for_tuning(timeout=1.0) is False
    assert loop.last_candidate_speedup_percent is not None
    assert loop.last_candidate_speedup_percent < 2.0
    assert loop.swap_manager.staging_kernel is None
    loop.close()
