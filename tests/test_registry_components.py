from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from platoon.registry import Registry, import_from_string
from platoon.train.auto import AutoDataset, AutoEnvironment, AutoRollout, AutoWorkflow
from platoon.train.components import EnvironmentConfig


def double(value):
    return value * 2


def test_registry_registers_resolves_and_rejects_duplicates():
    registry = Registry("example")

    registry.register("double", double)

    assert registry.get("double")(3) == 6
    assert registry.resolve("double")(4) == 8
    assert registry.get_item("double").import_path == "test_registry_components.double"

    with pytest.raises(ValueError, match="already has an entry"):
        registry.register("double", double)


def test_registry_resolves_import_paths():
    sqrt = import_from_string("math.sqrt")
    assert sqrt(9) == 3


def test_auto_dataset_loader_receives_split_kwargs():
    dataset_registry = Registry("dataset_loader")
    seen = {}

    def loader(config, split, **kwargs):
        seen["split"] = split
        seen["kwargs"] = kwargs
        return ("dataset", split)

    dataset_registry.register("fake", loader)

    config = type(
        "Config",
        (),
        {
            "environments": [
                EnvironmentConfig(
                    dataset_loader="fake",
                    eval_dataset_loader="fake",
                    dataset_kwargs={"difficulty": "train"},
                    eval_dataset_kwargs={"difficulty": "eval"},
                )
            ]
        },
    )()

    from platoon import registry as registry_module

    original = registry_module._REGISTRIES.get("dataset_loader")
    registry_module._REGISTRIES["dataset_loader"] = dataset_registry
    try:
        assert AutoDataset.from_config(config, "eval") == ("dataset", "eval")
        assert seen == {"split": "eval", "kwargs": {"difficulty": "eval"}}
    finally:
        if original is None:
            registry_module._REGISTRIES.pop("dataset_loader", None)
        else:
            registry_module._REGISTRIES["dataset_loader"] = original


def test_auto_factories_import_packages_for_side_effects():
    rollout_registry = Registry("rollout")

    def rollout(task, config):
        return {"task": task, "config": config}

    rollout_registry.register("fake/rollout", rollout)

    from platoon import registry as registry_module

    original = registry_module._REGISTRIES.get("rollout")
    registry_module._REGISTRIES["rollout"] = rollout_registry
    try:
        config = type("Config", (), {"environments": [EnvironmentConfig(rollout="fake/rollout")]})()
        AutoEnvironment.load(config)
        assert AutoRollout.from_config(config) is rollout
    finally:
        if original is None:
            registry_module._REGISTRIES.pop("rollout", None)
        else:
            registry_module._REGISTRIES["rollout"] = original


def test_auto_workflow_uses_default_group_rollout_name():
    class DefaultWorkflow:
        pass

    config = type("Config", (), {"environments": [EnvironmentConfig(workflow="group_rollout")]})()
    assert AutoWorkflow.from_config(config, default=DefaultWorkflow) is DefaultWorkflow


def test_auto_environment_rejects_multiple_environments():
    config = type(
        "Config",
        (),
        {
            "environments": [
                EnvironmentConfig(rollout="fake/one"),
                EnvironmentConfig(rollout="fake/two"),
            ]
        },
    )()

    with pytest.raises(NotImplementedError, match="Multiple environments are not yet supported"):
        AutoEnvironment.from_config(config)
