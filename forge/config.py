"""Typed configuration loading for Forge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ForgeConfig:
    """Runtime settings shared by Forge entry points."""

    model_config_hash: str
    profiler_window_size: int
    autotuner_candidate_limit: int
    tune_every_steps: int


def load_config(path: str | Path) -> ForgeConfig:
    """Load and validate a Forge YAML configuration file."""

    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")

    model_config_hash = _required_string(raw, "model_config_hash")
    profiler = _required_mapping(raw, "profiler")
    autotuner = _required_mapping(raw, "autotuner")
    serving = _required_mapping(raw, "serving")

    return ForgeConfig(
        model_config_hash=model_config_hash,
        profiler_window_size=_positive_integer(profiler, "window_size"),
        autotuner_candidate_limit=_positive_integer(autotuner, "candidate_limit"),
        tune_every_steps=_positive_integer(serving, "tune_every_steps"),
    )


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _required_string(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _positive_integer(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value
