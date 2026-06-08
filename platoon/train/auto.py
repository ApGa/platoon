"""Auto factories for training components selected by environment config."""

from __future__ import annotations

from typing import Any

from platoon.registry import discover_entry_points, import_modules, resolve_component
from platoon.train.components import EnvironmentConfig, task_ids_to_dataset


class AutoEnvironment:
    """Resolve the environment selected by config."""

    @classmethod
    def from_config(cls, config: Any) -> EnvironmentConfig:
        environments = getattr(config, "environments", None)
        if environments is None:
            raise ValueError("Config must define `environments` as a list with one environment")
        if not isinstance(environments, (list, tuple)):
            raise TypeError("Config field `environments` must be a list")
        if len(environments) == 0:
            raise ValueError("Config must define at least one environment in `environments`")
        if len(environments) > 1:
            raise NotImplementedError(
                "Multiple environments are not yet supported; provide exactly one entry"
            )
        environment = environments[0]
        if not isinstance(environment, EnvironmentConfig):
            raise TypeError("Each `environments` entry must be an EnvironmentConfig")
        return environment

    @classmethod
    def load(cls, config: Any) -> None:
        environment = cls.from_config(config)
        if environment.discover_entry_points:
            discover_entry_points()
        if environment.package is None:
            return
        import_modules([environment.package])


def _resolve_required_component(kind: str, spec: str | None) -> Any:
    if spec is None:
        raise ValueError(f"Config must set environments[0].{kind}")
    return resolve_component(kind, spec)


class AutoDataset:
    """Build datasets from environment dataset loaders."""

    @classmethod
    def from_config(cls, config: Any, split: str) -> Any:
        environment = AutoEnvironment.from_config(config)
        loader_spec = (
            environment.dataset_loader
            if split == "train"
            else environment.eval_dataset_loader or environment.dataset_loader
        )
        loader = _resolve_required_component("dataset_loader", loader_spec)
        kwargs = environment.dataset_kwargs if split == "train" else environment.eval_dataset_kwargs
        dataset = loader(config, split, **kwargs)
        if isinstance(dataset, list):
            return task_ids_to_dataset(dataset)
        return dataset


class AutoTaskLoader:
    """Resolve a task loader from environment config."""

    @classmethod
    def from_config(cls, config: Any) -> Any:
        environment = AutoEnvironment.from_config(config)
        return _resolve_required_component("task_loader", environment.task_loader)


class AutoRollout:
    """Resolve train/eval rollout functions from environment config."""

    @classmethod
    def from_config(cls, config: Any, split: str = "train") -> Any:
        environment = AutoEnvironment.from_config(config)
        rollout_spec = (
            environment.rollout
            if split == "train"
            else environment.eval_rollout or environment.rollout
        )
        return _resolve_required_component("rollout", rollout_spec)


class AutoRewardProcessor:
    """Resolve a reward processor from environment config."""

    @classmethod
    def from_config(cls, config: Any) -> Any:
        environment = AutoEnvironment.from_config(config)
        if environment.reward_processor is None:
            return lambda traj: (traj["reward"], {})
        return resolve_component("reward_processor", environment.reward_processor)


class AutoWorkflow:
    """Resolve a workflow class from environment config."""

    @classmethod
    def from_config(cls, config: Any, default: type) -> type:
        environment = AutoEnvironment.from_config(config)
        if environment.workflow == "group_rollout":
            return default
        return resolve_component("workflow", environment.workflow)
