"""Focused regression tests for AReaL loss registry and workflow naming."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from concurrent import futures
from pathlib import Path

import pytest
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

    class RecordingTracker:
        def __init__(self):
            self.values: dict[str, list[float]] = {}

        def scalar(self, **kwargs):
            for key, value in kwargs.items():
                self.values.setdefault(key, []).append(float(value))

    recording_tracker = RecordingTracker()
    utils_mod = types.ModuleType("areal.utils")
    utils_mod.__path__ = []
    utils_mod.stats_tracker = types.SimpleNamespace(get=lambda scope: recording_tracker)
    sys.modules["areal.utils"] = utils_mod

    dynamic_import_mod = types.ModuleType("areal.utils.dynamic_import")
    dynamic_import_mod.import_from_string = lambda path: path
    sys.modules["areal.utils.dynamic_import"] = dynamic_import_mod

    data_mod = types.ModuleType("areal.utils.data")

    def concat_padded_tensors(items):
        if not items or not isinstance(items[0], dict):
            return items
        return {
            key: torch.cat([item[key] for item in items], dim=0)
            for key in items[0]
        }

    data_mod.concat_padded_tensors = concat_padded_tensors
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
    processing_mod.EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY = "exclude_from_policy_training"
    processing_mod.OPTIONAL_REWARD_METRIC_MASK_PREFIX = "_platoon_reward_metric_present/"
    processing_mod.POLICY_TRAINING_ELIGIBILITY_MASK_KEY = "_platoon_policy_training_eligible"
    processing_mod.RouterReplayConfig = object
    processing_mod.SUBAGENT_DATUM_DEPTH_KEY = "_platoon_subagent_datum_depth"
    processing_mod.SUBAGENT_DATUM_KEEP_MASK_KEY = "_platoon_subagent_datum_keep"
    processing_mod.get_train_data_for_trajectory_collection = lambda *args, **kwargs: None
    processing_mod.harmonize_optional_reward_metrics = lambda items: items
    processing_mod.reward_metric_presence_key = lambda key: f"_platoon_reward_metric_present/{key}"
    sys.modules["platoon.utils.areal_data_processing"] = processing_mod

    module = _load_module(
        "platoon_areal_group_workflow_test",
        REPO_ROOT / "platoon/train/areal/workflows/group_rollout_workflow.py",
    )
    module._test_recording_tracker = recording_tracker
    return module


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


def test_areal_rollout_workload_counts_raw_recursive_tree_and_unique_exports():
    workflow_mod = _load_group_workflow_module()

    class Completion:
        def __init__(self, loss_mask, attention_mask=None):
            self.loss_mask = torch.tensor([loss_mask])
            self.attention_mask = (
                torch.tensor([attention_mask]) if attention_mask is not None else None
            )

        def to_tensor_dict(self):
            result = {
                "input_ids": torch.arange(self.loss_mask.numel()).reshape_as(self.loss_mask),
                "loss_mask": self.loss_mask,
            }
            if self.attention_mask is not None:
                result["attention_mask"] = self.attention_mask
            return result

    class MalformedCompletion:
        def to_tensor_dict(self):
            raise RuntimeError("telemetry fixture failure")

    raw_tree = {
        "trajectories": {
            "root": {"steps": [{}, {}]},
            "child": {"steps": [{}]},
            "verifier": {"steps": [{}, {}, {}]},
            "cancelled-child": {"steps": [{"cancelled": True}]},
        }
    }
    workload = workflow_mod._rollout_workload(
        raw_tree,
        {
            # The final padded token is excluded by attention_mask.
            "root-call": Completion([0, 0, 0, 1, 1, 0], [1, 1, 1, 1, 1, 0]),
            "child-call": Completion([0, 1, 1]),
            # A bad telemetry record counts as an observed model call but must
            # not prevent the usable calls or raw tree from being reported.
            "bad-verifier-call": MalformedCompletion(),
        },
    )

    assert workload == workflow_mod.RolloutWorkload(
        environment_steps=7,
        model_calls=3,
        input_tokens=4,
        output_tokens=4,
        trajectories=4,
    )


def test_zero_centered_reward_candidates_are_measured_without_mutating_training_mask():
    workflow_mod = _load_group_workflow_module()
    trainable_datums = torch.tensor([True, True, False, True])
    train_data = {
        # Exact zero and negative zero are candidates, but an excluded zero is
        # not. Arbitrarily small real signal must remain distinguishable.
        "rewards": torch.tensor([0.0, 1e-12, 0.0, -0.0]),
        "trainable_datums": trainable_datums.clone(),
        "loss_mask": torch.tensor(
            [
                [False, True, True],
                [False, False, True],
                [True, True, True],
                [False, True, False],
            ]
        ),
        "attention_mask": torch.tensor(
            [
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, False],
            ]
        ),
    }

    metrics = workflow_mod._measure_zero_centered_reward_candidates(train_data)

    assert metrics == {
        "workflow_zero_reward_candidate_population_datums": 3.0,
        "workflow_zero_reward_candidate_population_action_tokens": 4.0,
        "workflow_zero_reward_candidate_population_attention_tokens": 8.0,
        "workflow_zero_reward_candidate_datums": 2.0,
        "workflow_zero_reward_candidate_action_tokens": 3.0,
        "workflow_zero_reward_candidate_attention_tokens": 5.0,
    }
    assert torch.equal(train_data["trainable_datums"], trainable_datums)


def test_areal_error_filter_uses_centered_reward_sign_and_preserves_mixed_datums():
    workflow_mod = _load_group_workflow_module()
    sidechannel = workflow_mod.ERROR_ACTION_MASK_KEY
    train_data = {
        "rewards": torch.tensor([1.0, -1.0, 0.0, 2.0]),
        "loss_mask": torch.tensor(
            [
                [False, True, True],   # positive, one error plus one clean
                [False, True, True],   # negative: retain both error tokens
                [False, True, False],  # zero: retain the error token
                [False, True, True],   # positive and all-error: drop datum
            ]
        ),
        sidechannel: torch.tensor(
            [
                [False, True, False],
                [False, True, True],
                [False, True, False],
                [False, True, True],
            ]
        ),
    }

    metrics = workflow_mod._filter_positive_centered_error_tokens(train_data)

    assert sidechannel not in train_data
    assert torch.equal(
        train_data["loss_mask"],
        torch.tensor(
            [
                [False, False, True],
                [False, True, True],
                [False, True, False],
                [False, False, False],
            ]
        ),
    )
    assert torch.equal(train_data["trainable_datums"], torch.tensor([True, True, True, False]))
    assert metrics == {
        "error_filter/detected_action_tokens": 6.0,
        "error_filter/suppressed_positive_action_tokens": 3.0,
        "error_filter/retained_nonpositive_action_tokens": 3.0,
        "error_filter/emptied_datums": 1.0,
    }


@pytest.mark.asyncio
async def test_areal_group_filters_errors_only_after_leave_one_out_centering():
    workflow_mod = _load_group_workflow_module()
    workflow = workflow_mod.GroupRolloutWorkflow.__new__(workflow_mod.GroupRolloutWorkflow)
    workflow.config = types.SimpleNamespace(
        group_size=2,
        use_subprocesses=False,
        min_successful_group_size=1,
        leave_one_out_baseline=True,
        filter_zero_variance_groups=False,
        filter_zero_advantage_datums=False,
    )
    sidechannel = workflow_mod.ERROR_ACTION_MASK_KEY
    results = [
        {
            "rewards": torch.tensor([2.0]),
            "task_reward": torch.tensor([2.0]),
            "task_reward_valid": torch.tensor([True]),
            "attention_mask": torch.ones(1, 3, dtype=torch.bool),
            "loss_mask": torch.tensor([[False, True, True]]),
            sidechannel: torch.tensor([[False, True, True]]),
        },
        {
            "rewards": torch.tensor([0.0]),
            "task_reward": torch.tensor([0.0]),
            "task_reward_valid": torch.tensor([True]),
            "attention_mask": torch.ones(1, 3, dtype=torch.bool),
            "loss_mask": torch.tensor([[False, True, True]]),
            sidechannel: torch.tensor([[False, True, True]]),
        },
    ]

    async def fake_episode(self, engine, data, rollout_number):
        _ = self, engine, data
        return results[rollout_number]

    workflow._arun_episode_single = types.MethodType(fake_episode, workflow)
    workflow._record_stats = lambda _train_data: None

    train_data = await workflow.arun_episode(object(), {"task_id": "task"})

    assert train_data is not None
    assert sidechannel not in train_data
    assert torch.equal(train_data["rewards"], torch.tensor([2.0, -2.0]))
    assert torch.equal(
        train_data["loss_mask"],
        torch.tensor([[False, False, False], [False, True, True]]),
    )
    assert torch.equal(train_data["trainable_datums"], torch.tensor([False, True]))


@pytest.mark.asyncio
async def test_areal_exports_interactions_even_when_raw_rollout_is_none():
    workflow_mod = _load_group_workflow_module()
    workflow = workflow_mod.GroupRolloutWorkflow.__new__(workflow_mod.GroupRolloutWorkflow)

    class Completion:
        def to_tensor_dict(self):
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "loss_mask": torch.tensor([[0, 1, 1]]),
            }

    class Session:
        exports = 0

        async def export_interactions(self):
            self.exports += 1
            return {"completed-before-timeout": Completion()}

    session = Session()
    processed = await workflow._process_trajectory_result(None, session, "task", 0)

    assert session.exports == 1
    assert processed.train_data is None
    assert processed.observed is False
    assert processed.workload == workflow_mod.RolloutWorkload(
        model_calls=1,
        input_tokens=1,
        output_tokens=2,
    )


@pytest.mark.asyncio
async def test_areal_telemetry_failure_does_not_reject_data_and_attaches_datum_funnel(monkeypatch):
    workflow_mod = _load_group_workflow_module()
    workflow = workflow_mod.GroupRolloutWorkflow.__new__(workflow_mod.GroupRolloutWorkflow)
    workflow.config = types.SimpleNamespace(
        depth_level_weighting=False,
        depth_level_discount_gamma=None,
    )
    workflow.subagent_datum_sampler = None
    workflow.router_replay_config = None
    workflow.filter_errors = False
    workflow.reward_processor = lambda trajectory: (trajectory.get("reward", 0.0), {})
    workflow.merge_prefixes = True

    train_data = {
        "rewards": torch.tensor([1.0, 0.5, 0.25]),
        workflow_mod.POLICY_TRAINING_ELIGIBILITY_MASK_KEY: torch.tensor(
            [True, False, True]
        ),
        workflow_mod.SUBAGENT_DATUM_KEEP_MASK_KEY: torch.tensor([True, True, False]),
    }
    monkeypatch.setattr(
        workflow_mod,
        "get_train_data_for_trajectory_collection",
        lambda *args, **kwargs: train_data,
    )

    class BadTelemetryCompletion:
        def to_tensor_dict(self):
            raise RuntimeError("metrics-only parsing failed")

    class Session:
        async def export_interactions(self):
            return {"bad-but-convertible-by-canonical-path": BadTelemetryCompletion()}

    raw_tree = {"trajectories": {"root": {"steps": [{}, {}]}}}
    processed = await workflow._process_trajectory_result(raw_tree, Session(), "task", 0)

    assert processed.train_data is train_data
    assert processed.workload == workflow_mod.RolloutWorkload(
        environment_steps=2,
        model_calls=1,
        trajectories=1,
        postmerge_datums=3,
        policy_eligible_datums=2,
        post_sampling_datums=1,
    )


def test_areal_task_workload_sidecar_preserves_no_train_data_work():
    workflow_mod = _load_group_workflow_module()
    workflow = workflow_mod.GroupRolloutWorkflow.__new__(workflow_mod.GroupRolloutWorkflow)
    train_data = {
        "rewards": torch.tensor([1.0]),
        # A processed rollout is not necessarily trainable after policy
        # eligibility/sampling/zero-signal masks are activated.
        "trainable_datums": torch.tensor([False]),
    }
    processed = [
        workflow_mod._ProcessedRolloutResult(
            train_data={"rewards": torch.tensor([1.0])},
            workload=workflow_mod.RolloutWorkload(
                environment_steps=3,
                model_calls=2,
                input_tokens=10,
                output_tokens=4,
                trajectories=2,
                postmerge_datums=1,
                policy_eligible_datums=1,
                post_sampling_datums=1,
            ),
            observed=True,
        ),
        workflow_mod._ProcessedRolloutResult(
            train_data=None,
            workload=workflow_mod.RolloutWorkload(
                environment_steps=5,
                model_calls=3,
                input_tokens=20,
                output_tokens=7,
                trajectories=3,
            ),
            observed=True,
        ),
        workflow_mod._ProcessedRolloutResult(
            train_data=None,
            workload=workflow_mod.RolloutWorkload(model_calls=1, output_tokens=2),
            observed=False,
        ),
    ]

    workflow._attach_task_workload_sidecar(train_data, processed)

    assert train_data[workflow_mod._WORKLOAD_SIDECAR_FIELDS["environment_steps"]].item() == 8
    assert train_data[workflow_mod._WORKLOAD_SIDECAR_FIELDS["model_calls"]].item() == 6
    assert train_data[workflow_mod._WORKLOAD_SIDECAR_FIELDS["input_tokens"]].item() == 30
    assert train_data[workflow_mod._WORKLOAD_SIDECAR_FIELDS["output_tokens"]].item() == 13
    assert train_data[workflow_mod._WORKLOAD_SIDECAR_FIELDS["trajectories"]].item() == 5
    assert train_data[workflow_mod._WORKLOAD_REQUESTED_ROLLOUTS_KEY].item() == 3
    assert train_data[workflow_mod._WORKLOAD_OBSERVED_ROLLOUTS_KEY].item() == 2
    assert train_data[workflow_mod._WORKLOAD_TRAINABLE_ROLLOUTS_KEY].item() == 0
    assert train_data[
        workflow_mod._WORKLOAD_DATUM_SIDECAR_FIELDS["postmerge_datums"]
    ].item() == 1
    assert train_data[
        workflow_mod._WORKLOAD_DATUM_SIDECAR_FIELDS["policy_eligible_datums"]
    ].item() == 1
    assert train_data[
        workflow_mod._WORKLOAD_DATUM_SIDECAR_FIELDS["post_sampling_datums"]
    ].item() == 1
    assert train_data[workflow_mod._WORKLOAD_TASK_RETAINED_DATUMS_KEY].item() == 0


def test_group_rollout_stats_ignore_synthetic_optional_reward_zeros():
    workflow_mod = _load_group_workflow_module()
    workflow = workflow_mod.GroupRolloutWorkflow.__new__(workflow_mod.GroupRolloutWorkflow)
    presence = workflow_mod.reward_metric_presence_key

    workflow._record_stats(
        {
            "task_reward": torch.tensor([1.0, 0.0]),
            "num_steps": torch.tensor([2.0, 3.0, 1.0]),
            "num_input_tokens": torch.tensor([10.0, 20.0, 5.0]),
            "num_output_tokens": torch.tensor([4.0, 8.0, 2.0]),
            "reward/success": torch.tensor([1.0, 0.75, 0.0]),
            "reward/subagent_judgment": torch.tensor([0.0, 0.75, 0.0]),
            presence("reward/subagent_judgment"): torch.tensor([False, True, False]),
            "root_reward/success": torch.tensor([1.0, 0.0]),
            "root_reward/optional_diagnostic": torch.tensor([2.0, 0.0]),
            presence("root_reward/optional_diagnostic"): torch.tensor([True, False]),
        }
    )

    values = workflow_mod._test_recording_tracker.values
    assert values["reward/subagent_judgment"] == [0.75]
    assert values["reward/success"] == [1.0, 0.75, 0.0]
    assert values["root_reward/optional_diagnostic"] == [2.0]
    assert values["root_reward/optional_diagnostic_at_k_mean"] == [2.0]


def test_group_sampling_mask_is_activated_after_full_metrics_and_preserves_alignment():
    workflow_mod = _load_group_workflow_module()
    workflow = workflow_mod.GroupRolloutWorkflow.__new__(workflow_mod.GroupRolloutWorkflow)
    workflow.config = types.SimpleNamespace(
        depth_level_weighting=True,
        depth_level_discount_gamma=None,
    )
    policy_key = workflow_mod.POLICY_TRAINING_ELIGIBILITY_MASK_KEY
    keep_key = workflow_mod.SUBAGENT_DATUM_KEEP_MASK_KEY
    depth_key = workflow_mod.SUBAGENT_DATUM_DEPTH_KEY
    routes = torch.arange(3 * 4 * 2 * 2).reshape(3, 4, 2, 2)
    train_data = {
        "rewards": torch.tensor([1.0, 0.5, -1.0]),
        "attention_mask": torch.tensor(
            [[True, True, False, False], [True, True, True, False], [True, False, False, False]]
        ),
        "loss_mask": torch.tensor(
            [[False, True, False, False], [False, True, True, False], [True, False, False, False]]
        ),
        "traj_depth": torch.tensor([0, 1, 1]),
        "traj_start": torch.tensor([1.0, 0.0, 1.0]),
        "routed_experts": routes.clone(),
        policy_key: torch.tensor([True, False, True]),
        keep_key: torch.tensor([True, True, False]),
        depth_key: torch.tensor([0, 1, 1]),
    }

    workflow._activate_subagent_datum_sampling(train_data)

    assert torch.equal(train_data["trainable_datums"], torch.tensor([True, False, False]))
    assert torch.equal(train_data["routed_experts"], routes)
    assert torch.equal(train_data["traj_start"], torch.tensor([1.0, 0.0, 1.0]))
    assert keep_key not in train_data
    assert depth_key not in train_data
    assert policy_key not in train_data

    values = workflow_mod._test_recording_tracker.values
    assert values["subagent_sampling/eligible_datums"] == [2.0]
    assert values["subagent_sampling/retained_datums"] == [1.0]
    assert values["subagent_sampling/eligible_attention_tokens"] == [3.0]
    assert values["subagent_sampling/retained_attention_tokens"] == [2.0]
    assert values["subagent_sampling/eligible_loss_tokens"] == [2.0]
    assert values["subagent_sampling/retained_loss_tokens"] == [1.0]
    assert values["subagent_sampling/depth_0/retained_datums"] == [1.0]
    assert values["subagent_sampling/depth_1/eligible_datums"] == [1.0]
    assert values["subagent_sampling/depth_1/retained_datums"] == [0.0]


def test_group_sampling_intersects_existing_trainable_mask_and_retains_trim_metadata():
    workflow_mod = _load_group_workflow_module()
    workflow = workflow_mod.GroupRolloutWorkflow.__new__(workflow_mod.GroupRolloutWorkflow)
    workflow.config = types.SimpleNamespace(
        depth_level_weighting=False,
        depth_level_discount_gamma=None,
    )
    train_data = {
        "rewards": torch.ones(3),
        "attention_mask": torch.ones(3, 2, dtype=torch.bool),
        "loss_mask": torch.ones(3, 2, dtype=torch.bool),
        "trainable_datums": torch.tensor([True, True, False]),
        "traj_depth": torch.tensor([0, 1, 1]),
        "traj_start": torch.tensor([1.0, 1.0, 0.0]),
        workflow_mod.SUBAGENT_DATUM_KEEP_MASK_KEY: torch.tensor([True, False, True]),
        workflow_mod.SUBAGENT_DATUM_DEPTH_KEY: torch.tensor([0, 1, 1]),
    }

    workflow._activate_subagent_datum_sampling(train_data)

    assert torch.equal(train_data["trainable_datums"], torch.tensor([True, False, False]))
    assert torch.equal(train_data["traj_depth"], torch.tensor([0, 1, 1]))
    assert torch.equal(train_data["traj_start"], torch.tensor([1.0, 1.0, 0.0]))


def test_policy_exclusion_activates_without_bernoulli_sampling():
    workflow_mod = _load_group_workflow_module()
    workflow = workflow_mod.GroupRolloutWorkflow.__new__(workflow_mod.GroupRolloutWorkflow)
    workflow.config = types.SimpleNamespace(
        depth_level_weighting=False,
        depth_level_discount_gamma=None,
    )
    policy_key = workflow_mod.POLICY_TRAINING_ELIGIBILITY_MASK_KEY
    train_data = {
        "rewards": torch.tensor([1.0, 0.5]),
        "traj_depth": torch.tensor([0, 1]),
        "traj_start": torch.tensor([1.0, 1.0]),
        policy_key: torch.tensor([True, False]),
    }

    workflow._activate_subagent_datum_sampling(train_data)

    assert torch.equal(train_data["trainable_datums"], torch.tensor([True, False]))
    assert policy_key not in train_data
    assert "traj_depth" in train_data
    assert "traj_start" in train_data
    assert not any(
        key.startswith("subagent_sampling/")
        for key in workflow_mod._test_recording_tracker.values
    )


def test_all_policy_eligible_without_sampling_is_an_exact_mask_noop():
    workflow_mod = _load_group_workflow_module()
    workflow = workflow_mod.GroupRolloutWorkflow.__new__(workflow_mod.GroupRolloutWorkflow)
    policy_key = workflow_mod.POLICY_TRAINING_ELIGIBILITY_MASK_KEY
    train_data = {
        "rewards": torch.tensor([1.0, 0.5]),
        policy_key: torch.tensor([True, True]),
    }

    workflow._activate_subagent_datum_sampling(train_data)

    assert policy_key not in train_data
    assert "trainable_datums" not in train_data


@pytest.mark.asyncio
async def test_group_sampling_activates_only_after_full_group_loo_and_stats():
    workflow_mod = _load_group_workflow_module()
    workflow = workflow_mod.GroupRolloutWorkflow.__new__(workflow_mod.GroupRolloutWorkflow)
    workflow.config = types.SimpleNamespace(
        group_size=2,
        use_subprocesses=False,
        min_successful_group_size=1,
        leave_one_out_baseline=True,
        filter_zero_variance_groups=False,
        depth_level_weighting=True,
        depth_level_discount_gamma=None,
    )
    policy_key = workflow_mod.POLICY_TRAINING_ELIGIBILITY_MASK_KEY
    keep_key = workflow_mod.SUBAGENT_DATUM_KEEP_MASK_KEY
    depth_key = workflow_mod.SUBAGENT_DATUM_DEPTH_KEY
    results = [
        {
            "rewards": torch.tensor([2.0, 1.0]),
            "task_reward": torch.tensor([2.0]),
            "task_reward_valid": torch.tensor([True]),
            "attention_mask": torch.ones(2, 2, dtype=torch.bool),
            "loss_mask": torch.ones(2, 2, dtype=torch.bool),
            "traj_depth": torch.tensor([0, 1]),
            "traj_start": torch.tensor([1.0, 1.0]),
            policy_key: torch.tensor([True, False]),
            keep_key: torch.tensor([True, True]),
            depth_key: torch.tensor([0, 1]),
        },
        {
            "rewards": torch.tensor([0.0]),
            "task_reward": torch.tensor([0.0]),
            "task_reward_valid": torch.tensor([True]),
            "attention_mask": torch.ones(1, 2, dtype=torch.bool),
            "loss_mask": torch.ones(1, 2, dtype=torch.bool),
            "traj_depth": torch.tensor([0]),
            "traj_start": torch.tensor([1.0]),
            policy_key: torch.tensor([True]),
            keep_key: torch.tensor([True]),
            depth_key: torch.tensor([0]),
        },
    ]

    async def fake_episode(self, engine, data, rollout_number):
        _ = self, engine, data
        return results[rollout_number]

    stats_snapshot = {}

    def capture_stats(train_data):
        stats_snapshot["rewards"] = train_data["rewards"].clone()
        stats_snapshot["has_trainable_mask"] = "trainable_datums" in train_data

    workflow._arun_episode_single = types.MethodType(fake_episode, workflow)
    workflow._record_stats = capture_stats

    train_data = await workflow.arun_episode(object(), {"task_id": "task"})

    assert train_data is not None
    # Both members and every datum participate in LOO before the child datum is
    # masked: rollout 0 subtracts 0; rollout 1 subtracts rollout 0's reward 2.
    assert torch.equal(stats_snapshot["rewards"], torch.tensor([2.0, 1.0, -2.0]))
    assert stats_snapshot["has_trainable_mask"] is False
    assert torch.equal(train_data["rewards"], torch.tensor([2.0, 1.0, -2.0]))
    assert torch.equal(train_data["trainable_datums"], torch.tensor([True, False, True]))


@pytest.mark.parametrize(
    ("leave_one_out", "expected_rewards"),
    [
        (True, [-2.0, 97.0, 2.0, 2.0]),
        (False, [-1.0, 97.0, 2.0, 1.0]),
    ],
)
@pytest.mark.asyncio
async def test_partial_root_uses_only_valid_roots_for_group_baseline(
    leave_one_out,
    expected_rewards,
):
    workflow_mod = _load_group_workflow_module()
    workflow = workflow_mod.GroupRolloutWorkflow.__new__(workflow_mod.GroupRolloutWorkflow)
    workflow.config = types.SimpleNamespace(
        group_size=3,
        use_subprocesses=False,
        min_successful_group_size=1,
        leave_one_out_baseline=leave_one_out,
        filter_zero_variance_groups=False,
    )
    policy_key = workflow_mod.POLICY_TRAINING_ELIGIBILITY_MASK_KEY
    results = [
        {
            "rewards": torch.tensor([2.0]),
            "task_reward": torch.tensor([2.0]),
            "task_reward_valid": torch.tensor([True]),
            policy_key: torch.tensor([True]),
        },
        {
            # Cancelled root plus a completed child. The invalid root's value
            # must not contaminate any member's baseline.
            "rewards": torch.tensor([100.0, 5.0]),
            "task_reward": torch.tensor([100.0]),
            "task_reward_valid": torch.tensor([False]),
            policy_key: torch.tensor([False, True]),
        },
        {
            "rewards": torch.tensor([4.0]),
            "task_reward": torch.tensor([4.0]),
            "task_reward_valid": torch.tensor([True]),
            policy_key: torch.tensor([True]),
        },
    ]

    async def fake_episode(self, engine, data, rollout_number):
        _ = self, engine, data
        return results[rollout_number]

    stats_snapshot = {}
    workflow._arun_episode_single = types.MethodType(fake_episode, workflow)
    workflow._record_stats = lambda train_data: stats_snapshot.setdefault(
        "rewards", train_data["rewards"].clone()
    )

    train_data = await workflow.arun_episode(object(), {"task_id": "task"})

    assert train_data is not None
    expected = torch.tensor(expected_rewards)
    assert torch.equal(stats_snapshot["rewards"], expected)
    assert torch.equal(train_data["rewards"], expected)
    assert torch.equal(
        train_data["trainable_datums"],
        torch.tensor([True, False, True, True]),
    )


@pytest.mark.asyncio
async def test_single_valid_root_loo_falls_back_to_its_own_reward():
    workflow_mod = _load_group_workflow_module()
    workflow = workflow_mod.GroupRolloutWorkflow.__new__(workflow_mod.GroupRolloutWorkflow)
    workflow.config = types.SimpleNamespace(
        group_size=2,
        use_subprocesses=False,
        min_successful_group_size=1,
        leave_one_out_baseline=True,
        filter_zero_variance_groups=False,
    )
    policy_key = workflow_mod.POLICY_TRAINING_ELIGIBILITY_MASK_KEY
    results = [
        {
            "rewards": torch.tensor([2.0]),
            "task_reward": torch.tensor([2.0]),
            "task_reward_valid": torch.tensor([True]),
            policy_key: torch.tensor([True]),
        },
        {
            "rewards": torch.tensor([100.0, 5.0]),
            "task_reward": torch.tensor([100.0]),
            "task_reward_valid": torch.tensor([False]),
            policy_key: torch.tensor([False, True]),
        },
    ]

    async def fake_episode(self, engine, data, rollout_number):
        _ = self, engine, data
        return results[rollout_number]

    workflow._arun_episode_single = types.MethodType(fake_episode, workflow)
    workflow._record_stats = lambda train_data: None

    train_data = await workflow.arun_episode(object(), {"task_id": "task"})

    assert train_data is not None
    assert torch.equal(train_data["rewards"], torch.tensor([0.0, 98.0, 3.0]))
    assert torch.equal(train_data["trainable_datums"], torch.tensor([True, False, True]))


@pytest.mark.asyncio
async def test_group_with_no_valid_root_records_full_stats_then_rejects():
    workflow_mod = _load_group_workflow_module()
    workflow = workflow_mod.GroupRolloutWorkflow.__new__(workflow_mod.GroupRolloutWorkflow)
    workflow.config = types.SimpleNamespace(
        group_size=2,
        use_subprocesses=False,
        min_successful_group_size=1,
        leave_one_out_baseline=True,
        filter_zero_variance_groups=False,
    )
    policy_key = workflow_mod.POLICY_TRAINING_ELIGIBILITY_MASK_KEY
    results = [
        {
            "rewards": torch.tensor([10.0, 1.0]),
            "task_reward": torch.tensor([10.0]),
            "task_reward_valid": torch.tensor([False]),
            policy_key: torch.tensor([False, True]),
        },
        {
            "rewards": torch.tensor([20.0, 2.0]),
            "task_reward": torch.tensor([20.0]),
            "task_reward_valid": torch.tensor([False]),
            policy_key: torch.tensor([False, True]),
        },
    ]

    async def fake_episode(self, engine, data, rollout_number):
        _ = self, engine, data
        return results[rollout_number]

    stats_snapshot = {}
    workflow._arun_episode_single = types.MethodType(fake_episode, workflow)
    workflow._record_stats = lambda train_data: stats_snapshot.setdefault(
        "rewards", train_data["rewards"].clone()
    )

    train_data = await workflow.arun_episode(object(), {"task_id": "task"})

    assert train_data is None
    assert torch.equal(stats_snapshot["rewards"], torch.tensor([10.0, 1.0, 20.0, 2.0]))
    assert workflow_mod._test_recording_tracker.values["no_valid_root_reward_group"] == [1.0]


@pytest.mark.asyncio
async def test_partial_members_do_not_satisfy_completed_root_quorum():
    workflow_mod = _load_group_workflow_module()
    workflow = workflow_mod.GroupRolloutWorkflow.__new__(workflow_mod.GroupRolloutWorkflow)
    workflow.config = types.SimpleNamespace(
        group_size=4,
        use_subprocesses=False,
        min_successful_group_size=3,
        leave_one_out_baseline=True,
        filter_zero_variance_groups=False,
    )
    policy_key = workflow_mod.POLICY_TRAINING_ELIGIBILITY_MASK_KEY
    results = [
        {
            "rewards": torch.tensor([float(index + 1)]),
            "task_reward": torch.tensor([float(index + 1)]),
            "task_reward_valid": torch.tensor([index < 2]),
            policy_key: torch.tensor([index < 2]),
        }
        for index in range(4)
    ]

    async def fake_episode(self, engine, data, rollout_number):
        _ = self, engine, data
        return results[rollout_number]

    stats_snapshot = {}
    workflow._arun_episode_single = types.MethodType(fake_episode, workflow)
    workflow._record_stats = lambda train_data: stats_snapshot.setdefault(
        "task_reward", train_data["task_reward"].clone()
    )

    assert await workflow.arun_episode(object(), {"task_id": "task"}) is None
    assert torch.equal(stats_snapshot["task_reward"], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    tracker_values = workflow_mod._test_recording_tracker.values
    assert tracker_values["group_size_completed_roots"] == [2.0]
    assert tracker_values["group_completed_root_quorum_rejected"] == [1.0]


@pytest.mark.asyncio
async def test_zero_variance_filter_ignores_policy_excluded_verifier_rewards():
    workflow_mod = _load_group_workflow_module()
    workflow = workflow_mod.GroupRolloutWorkflow.__new__(workflow_mod.GroupRolloutWorkflow)
    workflow.config = types.SimpleNamespace(
        group_size=2,
        use_subprocesses=False,
        min_successful_group_size=1,
        leave_one_out_baseline=True,
        filter_zero_variance_groups=True,
    )
    policy_key = workflow_mod.POLICY_TRAINING_ELIGIBILITY_MASK_KEY
    results = [
        {
            "rewards": torch.tensor([1.0, 2.0]),
            "task_reward": torch.tensor([1.0]),
            policy_key: torch.tensor([True, False]),
        },
        {
            "rewards": torch.tensor([1.0, 3.0]),
            "task_reward": torch.tensor([1.0]),
            policy_key: torch.tensor([True, False]),
        },
    ]

    async def fake_episode(self, engine, data, rollout_number):
        _ = self, engine, data
        return results[rollout_number]

    stats_snapshot = {}
    workflow._arun_episode_single = types.MethodType(fake_episode, workflow)
    workflow._record_stats = lambda train_data: stats_snapshot.setdefault(
        "rewards", train_data["rewards"].clone()
    )

    train_data = await workflow.arun_episode(object(), {"task_id": "task"})

    # Full LOO/stats still observe the verifier rewards, but only the two root
    # zeros are policy-trainable, so the group has no policy-gradient signal.
    assert torch.equal(stats_snapshot["rewards"], torch.tensor([0.0, 1.0, 0.0, 2.0]))
    assert train_data is None
    assert workflow_mod._test_recording_tracker.values["zero_variance_reward_group"] == [1.0]


@pytest.mark.asyncio
async def test_group_tail_counts_interrupted_members_before_killing_process_pool(monkeypatch):
    """Four usable plus three interrupted peers arm grace for the last member."""

    workflow_mod = _load_group_workflow_module()
    events: list[str] = []
    process_killed = asyncio.Event()

    class FakeExecutor:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs
            self._processes = {}

        def shutdown(self, wait=True, cancel_futures=False):
            events.append(f"shutdown:{wait}:{cancel_futures}")

    class FakeProxySession:
        def __init__(self, **kwargs):
            self.session_id = kwargs["task_id"]
            self.session_api_key = "session-key"

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            events.append("session_closed")

    async def get_http_session():
        return object()

    monkeypatch.setattr(futures, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(workflow_mod, "ArealProxySession", FakeProxySession)
    workflow_mod.workflow_context.get_aiohttp_session = get_http_session

    workflow = workflow_mod.GroupRolloutWorkflow.__new__(workflow_mod.GroupRolloutWorkflow)
    workflow.config = types.SimpleNamespace(
        group_size=8,
        straggler_timeout_seconds=0.01,
        straggler_quorum=6,
        subprocess_shutdown_grace_seconds=0.0,
    )
    workflow.proxy_base_url = "http://proxy"
    workflow.proxy_admin_api_key = "admin"

    async def fake_run(self, executor, engine, task_id, rollout_number, session):
        _ = self, executor, engine, task_id, session
        if rollout_number < 4:
            return workflow_mod._SubprocessRolloutOutcome(
                result={"rollout": rollout_number},
                elapsed_seconds=0.001,
            )
        if rollout_number < 7:
            return workflow_mod._SubprocessRolloutOutcome(
                result={
                    "trajectories": {
                        "root": {
                            "misc": {"trajectory_timed_out": True},
                            "error_message": "Episode timed out",
                        }
                    }
                },
                elapsed_seconds=0.002,
            )
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            events.append("wrapper_cancelled")
            # Deliberately suppress cancellation until the process-tree reap.
            await process_killed.wait()
            events.append("wrapper_released")
            return workflow_mod._SubprocessRolloutOutcome(
                result=None,
                elapsed_seconds=0.02,
            )

    async def fake_terminate(self, executor, grace_seconds):
        _ = self, executor, grace_seconds
        events.append("terminate")
        process_killed.set()

    async def fake_process(self, raw_result, session, task_id, rollout_number):
        _ = self, session, task_id, rollout_number
        return raw_result

    workflow._run_rollout_subprocess = types.MethodType(fake_run, workflow)
    workflow._terminate_executor_processes = types.MethodType(fake_terminate, workflow)
    workflow._process_trajectory_result = types.MethodType(fake_process, workflow)

    result = await asyncio.wait_for(
        workflow._arun_episode_with_subprocesses(object(), {"task_id": "task"}),
        timeout=0.5,
    )

    assert [result[index]["rollout"] for index in range(4)] == list(range(4))
    assert all(
        result[index]["trajectories"]["root"]["misc"]["trajectory_timed_out"]
        for index in range(4, 7)
    )
    assert result[7] is None
    assert events.index("terminate") < events.index("wrapper_released")
    assert "shutdown:False:True" in events
    assert workflow_mod._test_recording_tracker.values["group_tail_cancelled"] == [1.0]
