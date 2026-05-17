"""Helpers for trainer entrypoints driven by registered plugin components."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from platoon.registry import discover_entry_points, import_modules, resolve_component
from platoon.train.components import PluginResolverConfig, task_ids_to_dataset


def load_plugin_components(plugin: PluginResolverConfig) -> None:
    """Import plugin registration modules requested by config."""

    if plugin.discover_entry_points:
        discover_entry_points()
    if plugin.package is None:
        return
    import_modules([plugin.package])


def resolve_required_component(kind: str, spec: str | None) -> Any:
    if spec is None:
        raise ValueError(f"Config must set plugin.{kind}")
    return resolve_component(kind, spec)


def build_registered_dataset(config: Any, split: str) -> Any:
    """Build a train/eval dataset through the configured dataset loader."""

    plugin = config.plugin
    loader_spec = plugin.dataset_loader if split == "train" else plugin.eval_dataset_loader or plugin.dataset_loader
    loader = resolve_required_component("dataset_loader", loader_spec)
    kwargs = plugin.dataset_kwargs if split == "train" else plugin.eval_dataset_kwargs
    dataset = loader(config, split, **kwargs)
    if isinstance(dataset, list):
        return task_ids_to_dataset(dataset)
    return dataset


def resolve_registered_task_loader(config: Any) -> Callable:
    return resolve_required_component("task_loader", config.plugin.task_loader)


def resolve_registered_rollout(config: Any, split: str = "train") -> Callable:
    rollout_spec = config.plugin.rollout if split == "train" else config.plugin.eval_rollout or config.plugin.rollout
    return resolve_required_component("rollout", rollout_spec)


def resolve_registered_reward_processor(config: Any) -> Callable[[dict[str, Any]], tuple[float, dict[str, Any]]]:
    if config.plugin.reward_processor is None:
        return lambda traj: (traj["reward"], {})
    return resolve_component("reward_processor", config.plugin.reward_processor)


def resolve_registered_workflow(config: Any, default_workflow: type) -> type:
    if config.plugin.workflow == "group_rollout":
        return default_workflow
    return resolve_component("workflow", config.plugin.workflow)
