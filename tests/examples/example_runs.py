"""Run an example's config the way ``platoon rollout`` runs one task: resolve the config, import its packages,
load the task through its environment route and roll it out through this process's policy gateway."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


def load_config(path: str | Path, *overrides: str) -> Any:
    """A training config, validated like ``platoon config validate`` validates it."""

    from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
    from platoon.utils.config import resolve_config

    return resolve_config(Path(path), overrides, PlatoonTinkerRLTrainerConfig).config


async def rollout(config: Any, task_id: str, output_dir: Path, *, split: str = "train") -> Any:
    """One rollout of ``task_id`` with the config's rollout settings; returns the ``TrajectoryCollection``."""

    from platoon.train.auto import AutoEnvironment, AutoRollout, AutoTaskLoader

    phase = config.train if split == "train" else config.eval
    rollout_config = copy.deepcopy(phase.workflow_config.rollout_config)
    rollout_config.output_dir, rollout_config.verbose, rollout_config.return_dict = str(output_dir), False, False
    AutoEnvironment.load(config)
    task = AutoTaskLoader.from_config(config)(task_id)
    if rollout_config.max_steps is not None:
        task.max_steps = rollout_config.max_steps
    return await AutoRollout.from_config(config, split)(task, rollout_config)


def by_depth(collection: Any) -> list[Any]:
    """The collection's trajectories, the root first."""

    return sorted(collection.trajectories.values(), key=lambda trajectory: trajectory.parent_info is not None)
