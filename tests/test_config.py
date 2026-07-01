from pathlib import Path

import pytest

from forge.config import load_config


def test_load_config_reads_runtime_settings(tmp_path: Path) -> None:
    path = tmp_path / "forge.yaml"
    path.write_text(
        """
model_config_hash: test_model
profiler:
  window_size: 12
autotuner:
  candidate_limit: 4
serving:
  tune_every_steps: 6
""".strip()
    )

    config = load_config(path)

    assert config.model_config_hash == "test_model"
    assert config.profiler_window_size == 12
    assert config.autotuner_candidate_limit == 4
    assert config.tune_every_steps == 6


def test_load_config_rejects_invalid_positive_integer(tmp_path: Path) -> None:
    path = tmp_path / "forge.yaml"
    path.write_text(
        """
model_config_hash: test_model
profiler:
  window_size: 0
autotuner:
  candidate_limit: 4
serving:
  tune_every_steps: 6
""".strip()
    )

    with pytest.raises(ValueError, match="window_size must be a positive integer"):
        load_config(path)
