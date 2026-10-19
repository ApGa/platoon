"""Dataset and task loaders for training on Harbor task directories through the ``environments`` config.

    environments:
      - dataset_loader: platoon.envs.harbor.loaders:load_dataset   # or "harbor" with package: platoon.envs.harbor
        task_loader: platoon.envs.harbor.loaders:load_task
        dataset_kwargs: {dataset_dir: /data/harbor/my-dataset, splits: splits.json}

Task ids are absolute task directory paths, so ``load_task`` needs no config and
a ``HarborEnvironmentService`` opens sessions for the loaded tasks directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from platoon.envs.harbor.dataset import HarborDataset, SplitManifest
from platoon.envs.harbor.task import AgentUserPolicy, HarborTask, VerifierUserPolicy
from platoon.registry import register_dataset_loader, register_task_loader
from platoon.task import Task


def load_dataset(
    config: Any,
    split: str,
    *,
    dataset_dir: str | Path | None = None,
    task_dirs: Sequence[str | Path] | None = None,
    splits: SplitManifest | None = None,
    split_name: str | None = None,
    limit: int | None = None,
    agent_user: AgentUserPolicy = "refuse",
    verifier_user: VerifierUserPolicy | None = None,
) -> list[str]:
    """Task ids of a split (``split_name`` renames it, e.g. eval -> test), in manifest or directory order."""

    dataset = HarborDataset(
        dataset_dir, task_dirs=task_dirs, splits=splits, agent_user=agent_user, verifier_user=verifier_user
    )
    task_ids = [str(dataset.get(name).task_dir) for name in dataset.names(split_name or split)]
    return task_ids if limit is None else task_ids[:limit]


def load_task(task_id: str) -> Task:
    return HarborTask.load(task_id).to_task(task_id)


register_dataset_loader("harbor", load_dataset)
register_task_loader("harbor", load_task)
