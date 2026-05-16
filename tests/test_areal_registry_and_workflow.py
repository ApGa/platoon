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
    infra_mod.__path__ = []
    infra_mod.workflow_context = types.SimpleNamespace(stat_scope=lambda: "test")
    sys.modules["areal.infra"] = infra_mod

    remote_inf_engine_mod = types.ModuleType("areal.infra.remote_inf_engine")

    class RemoteInfEngine:
        def _resolve_workflow(self, workflow, workflow_kwargs, group_size=1, proxy_addr=None):
            return workflow

    remote_inf_engine_mod.RemoteInfEngine = RemoteInfEngine
    sys.modules["areal.infra.remote_inf_engine"] = remote_inf_engine_mod

    utils_mod = types.ModuleType("areal.utils")
    utils_mod.__path__ = []
    utils_mod.stats_tracker = types.SimpleNamespace(get=lambda scope: types.SimpleNamespace())
    sys.modules["areal.utils"] = utils_mod

    dynamic_import_mod = types.ModuleType("areal.utils.dynamic_import")
    dynamic_import_mod.import_from_string = lambda path: path
    sys.modules["areal.utils.dynamic_import"] = dynamic_import_mod

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

    serialization_mod = types.ModuleType("platoon.train.areal.workflow_serialization")

    class RemoteWorkflowSerializable:
        pass

    serialization_mod.RemoteWorkflowSerializable = RemoteWorkflowSerializable
    serialization_mod.callable_import_path = lambda fn: f"{fn.__module__}.{fn.__name__}"
    sys.modules["platoon.train.areal.workflow_serialization"] = serialization_mod

    processing_mod = types.ModuleType("platoon.utils.areal_data_processing")
    processing_mod.get_train_data_for_trajectory_collection = lambda *args, **kwargs: None
    sys.modules["platoon.utils.areal_data_processing"] = processing_mod

    return _load_module(
        "platoon_areal_group_workflow_test",
        REPO_ROOT / "platoon/train/areal/workflows/group_rollout_workflow.py",
    )


def test_registered_loss_functions_include_builtin_names():
    loss_functions = _load_loss_functions_module()
    assert {"cispo", "grpo", "ppo"}.issubset(set(loss_functions.list_loss_fns()))


def test_build_loss_fn_filters_unknown_kwargs_for_plugin_losses():
    loss_functions = _load_loss_functions_module()

    @loss_functions.register_loss_fn("plugin_loss", defaults={"alpha": 2.0})
    def plugin_loss(logprobs, entropy, input_data, alpha=1.0):
        return logprobs.sum() * alpha

    default_bound = loss_functions.build_loss_fn("plugin_loss", ignored=99.0)
    default_result = default_bound(torch.ones(2), torch.zeros(2), {})
    assert torch.equal(default_result, torch.tensor(4.0))

    bound = loss_functions.build_loss_fn(
        "plugin_loss",
        loss_fn_kwargs={"alpha": 2.0},
        common_kwargs={"alpha": 3.0, "ignored": 99.0},
    )
    result = bound(torch.ones(2), torch.zeros(2), {})
    assert torch.equal(result, torch.tensor(6.0))


def test_build_loss_fn_applies_registered_loss_defaults():
    loss_functions = _load_loss_functions_module()

    defaults = loss_functions.get_loss_fn_defaults("cispo")
    assert defaults["clip_low_threshold"] == 0.0
    assert defaults["clip_high_threshold"] == 5.0


def test_group_rollout_workflow_exports_primary_class():
    workflow_mod = _load_group_workflow_module()
    assert workflow_mod.GroupRolloutWorkflow.__name__ == "GroupRolloutWorkflow"
    assert not hasattr(workflow_mod, "StepWiseArealWorkflow")
