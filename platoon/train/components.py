"""Shared component protocols and resolver config for registered trainers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from platoon.envs.base import Task


@runtime_checkable
class DatasetLoader(Protocol):
    """Build a backend dataset or list of task ids for a train/eval split."""

    def __call__(self, config: Any, split: str, **kwargs: Any) -> Any: ...


@runtime_checkable
class TaskLoader(Protocol):
    """Resolve a task id into a Platoon task."""

    def __call__(self, task_id: str) -> Task: ...


@runtime_checkable
class RolloutFn(Protocol):
    """Run one rollout for a task and rollout config."""

    def __call__(self, task: Task, config: Any) -> Any: ...


RewardProcessor = Callable[[dict[str, Any]], tuple[float, dict[str, Any]]]
WorkflowFactory = Callable[..., Any]
TrainerConfigClass = type[Any]
LossFn = Callable[..., Any]


@dataclass
class PluginResolverConfig:
    """Registry/import-path references for plugin-specific training components."""

    package: str | None = None
    discover_entry_points: bool = False
    trainer_config: str | None = None
    dataset_loader: str | None = None
    eval_dataset_loader: str | None = None
    task_loader: str | None = None
    rollout: str | None = None
    eval_rollout: str | None = None
    reward_processor: str | None = None
    workflow: str = "group_rollout"
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    eval_dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    workflow_kwargs: dict[str, Any] = field(default_factory=dict)
    eval_workflow_kwargs: dict[str, Any] = field(default_factory=dict)



def task_ids_to_dataset(task_ids: Sequence[str]) -> Any:
    """Convert task ids to a Hugging Face Dataset lazily to keep core imports light."""

    from datasets import Dataset

    return Dataset.from_list([{"task_id": task_id} for task_id in task_ids])
