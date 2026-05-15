"""Focused regression tests for AReaL loss registry and workflow naming."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_loss_functions_module():
    actor_mod = types.ModuleType("areal.trainer.ppo.actor")
    actor_mod.grpo_loss_fn = lambda logprobs, entropy, input_data, **kwargs: torch.tensor(1.0)
    sys.modules["areal.trainer.ppo.actor"] = actor_mod

    stats_mod = types.ModuleType("areal.trainer.ppo.stats")
    stats_mod.infer_token_denominator = lambda input_data, loss_mask: loss_mask
    sys.modules["areal.trainer.ppo.stats"] = stats_mod

    utils_mod = types.ModuleType("areal.utils")
    utils_mod.stats_tracker = types.SimpleNamespace(
        denominator=lambda **kwargs: None,
        stat=lambda **kwargs: None,
    )
    sys.modules["areal.utils"] = utils_mod

    return _load_module(
        "platoon_areal_loss_functions_test",
        REPO_ROOT / "platoon/train/areal/loss_functions.py",
    )


def _load_group_workflow_module():
    api_mod = types.ModuleType("areal.api")
    api_mod.InferenceEngine = type("InferenceEngine", (), {})
    api_mod.RolloutWorkflow = type("RolloutWorkflow", (), {})
    sys.modules["areal.api"] = api_mod

    infra_mod = types.ModuleType("areal.infra")
    infra_mod.workflow_context = types.SimpleNamespace(stat_scope=lambda: "test")
    sys.modules["areal.infra"] = infra_mod

    utils_mod = types.ModuleType("areal.utils")
    utils_mod.stats_tracker = types.SimpleNamespace(get=lambda scope: types.SimpleNamespace())
    sys.modules["areal.utils"] = utils_mod

    data_mod = types.ModuleType("areal.utils.data")
    data_mod.concat_padded_tensors = lambda items: items
    sys.modules["areal.utils.data"] = data_mod

    env_mod = types.ModuleType("platoon.envs.base")
    env_mod.Task = object
    sys.modules["platoon.envs.base"] = env_mod

    config_mod = types.ModuleType("platoon.train.areal.config_defs")
    config_mod.WorkflowConfig = object
    sys.modules["platoon.train.areal.config_defs"] = config_mod

    proxy_mod = types.ModuleType("platoon.train.areal.proxy")
    proxy_mod.ArealProxySession = object
    sys.modules["platoon.train.areal.proxy"] = proxy_mod

    processing_mod = types.ModuleType("platoon.utils.areal_data_processing")
    processing_mod.get_train_data_for_trajectory_collection = lambda *args, **kwargs: None
    sys.modules["platoon.utils.areal_data_processing"] = processing_mod

    return _load_module(
        "platoon_areal_group_workflow_test",
        REPO_ROOT / "platoon/train/areal/workflows/step_wise.py",
    )


def test_registered_loss_functions_include_builtin_names():
    loss_functions = _load_loss_functions_module()
    assert {"cispo", "grpo", "ppo"}.issubset(set(loss_functions.list_loss_fns()))


def test_build_loss_fn_filters_unknown_kwargs_for_plugin_losses():
    loss_functions = _load_loss_functions_module()

    @loss_functions.register_loss_fn("plugin_loss")
    def plugin_loss(logprobs, entropy, input_data, alpha=1.0):
        return logprobs.sum() * alpha

    bound = loss_functions.build_loss_fn("plugin_loss", alpha=3.0, ignored=99.0)
    result = bound(torch.ones(2), torch.zeros(2), {})
    assert torch.equal(result, torch.tensor(6.0))


def test_group_rollout_workflow_is_backward_compatible_alias():
    workflow_mod = _load_group_workflow_module()
    assert workflow_mod.GroupRolloutWorkflow is workflow_mod.StepWiseArealWorkflow
