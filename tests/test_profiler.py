from forge.profiler.profiler import DecodeStepMetrics, RuntimeProfiler


def test_profiler_builds_workload_spec_from_recent_steps() -> None:
    profiler = RuntimeProfiler(model_config_hash="toy", window_size=4)

    for _ in range(4):
        profiler.record_step(DecodeStepMetrics(batch_size=8, seq_len=900, latency_ms=1.0))

    spec = profiler.current_spec()

    assert spec is not None
    assert spec.batch_size_bucket == "5-8"
    assert spec.seq_len_bucket == "513-1024"
    assert spec.model_config_hash == "toy"
