"""A run's config for the training tests, built as ``platoon train`` builds it, by one of three builders:

- :func:`make_config`, in memory: :data:`DEFAULTS` (a run of three steps of two groups each, with evaluation and the
  watchdog off) with a test's settings laid over them, for a test that hands a config to the trainer or its parts;
- :func:`trainer_config`, from a file: the same :data:`DEFAULTS` with ``key=value`` overrides as the command line gives
  them, the run directory under the test's ``tmp_path``, for a run that writes there;
- :func:`mixture_config`, from a file: a mixture of environments (``environments:``) over :data:`BASE` alone, the
  run's geometry (batch size, steps, rollouts at once) the test's or the config's own defaults.

Each builds the real config, with every check loading makes.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.utils.config import build_config, resolve_config

# What every run of the tests sets: a model and a renderer, which nothing loads here, and no evaluation or watchdog.
BASE: dict[str, Any] = {
    "train": {"model_name": "m", "renderer_name": "r"},
    "eval": {"every": 0},
    "watchdog": {"enabled": False},
}
# A run of three steps of two groups each, with rollouts for two groups of the default size (8) at once.
DEFAULTS: dict[str, Any] = {
    **BASE,
    "train": {**BASE["train"], "batch_size": 2, "max_training_steps": 3, "max_concurrent_rollouts": 16},
}
# The run directory's root of a config whose test writes nothing there: no test can create it.
UNWRITTEN_LOG_PATH = "/nonexistent/platoon-tests/logs"


def make_config(**overrides: Any) -> PlatoonTinkerRLTrainerConfig:
    """:data:`DEFAULTS` with ``overrides`` laid over them, section by section: ``train={"batch_size": 4}`` changes one
    setting of ``train:`` and keeps the others, and a value that is not a mapping replaces the default's.  A test
    whose run writes gives ``log_path``; without one, the config's run directory is one no test can write."""

    settings = _laid_over(DEFAULTS, {"log_path": UNWRITTEN_LOG_PATH, **overrides})
    return build_config(PlatoonTinkerRLTrainerConfig, settings)


def trainer_config(tmp_path: Path, *overrides: str) -> PlatoonTinkerRLTrainerConfig:
    """:data:`DEFAULTS`, with the run directory under ``tmp_path/logs``, loaded from a file with ``overrides``
    (``key=value`` strings), as ``platoon train`` loads its config."""

    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump({**DEFAULTS, "log_path": str(tmp_path / "logs")}))
    return resolve_config(path, overrides, PlatoonTinkerRLTrainerConfig).config


def mixture_config(
    tmp_path: Path, train: Mapping[str, Any], environments: list[dict[str, Any]] | None = None, **sections: Any
) -> PlatoonTinkerRLTrainerConfig:
    """A mixture's config, loaded from ``tmp_path/train.yaml`` as ``platoon train`` loads it: :data:`BASE` with
    ``train`` laid over its ``train:`` and ``sections`` (``checkpoint={...}``, ...) over the others, the run directory
    under ``tmp_path/logs``, and the entries ``environments`` (each with the rollout ``x:y`` unless it names one; no
    ``environments:`` when None)."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    settings = _laid_over(BASE, {"train": train, "log_path": str(tmp_path / "logs"), **sections})
    if environments is not None:
        settings["environments"] = [{"rollout": "x:y", **entry} for entry in environments]
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(settings))
    return resolve_config(path, [], PlatoonTinkerRLTrainerConfig).config


def _laid_over(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    laid = dict(base)
    for key, value in overlay.items():
        below = laid.get(key)
        laid[key] = _laid_over(below, value) if isinstance(below, Mapping) and isinstance(value, Mapping) else value
    return laid
