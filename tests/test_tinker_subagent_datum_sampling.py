"""Focused tests for Tinker's post-merge recursive-datum sampling path."""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from platoon.utils.subagent_sampling import DeterministicSubagentDatumSampler
from platoon.utils.trajectory_status import TRAJECTORY_CANCELLED_MISC_KEY

REPO_ROOT = Path(__file__).resolve().parents[1]


class DummyTensorData:
    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    def to_torch(self) -> torch.Tensor:
        return self._tensor

    @classmethod
    def from_torch(cls, tensor: torch.Tensor) -> "DummyTensorData":
        return cls(tensor)


class DummyModelInput:
    def __init__(self, length: int | None = None, chunks=None):
        self.chunks = list(chunks or [])
        self.length = (
            int(length)
            if length is not None
            else sum(chunk.length for chunk in self.chunks)
        )


class DummyDatum:
    def __init__(self, model_input: DummyModelInput, loss_fn_inputs: dict):
        self.model_input = model_input
        self.loss_fn_inputs = loss_fn_inputs


class DummyEncodedTextChunk:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.length = len(self.tokens)


def _load_tinker_data_processing(monkeypatch):
    tinker_module = types.ModuleType("tinker")
    tinker_module.Datum = DummyDatum
    tinker_module.TensorData = DummyTensorData
    tinker_module.ModelInput = DummyModelInput
    tinker_module.ModelInputChunk = object
    tinker_module.EncodedTextChunk = DummyEncodedTextChunk
    tinker_module.types = types.SimpleNamespace(EncodedTextChunk=DummyEncodedTextChunk)
    monkeypatch.setitem(sys.modules, "tinker", tinker_module)

    proxy_module = types.ModuleType("platoon.train.tinker.proxy")
    proxy_module.TinkerLLMInteraction = object
    monkeypatch.setitem(sys.modules, "platoon.train.tinker.proxy", proxy_module)

    subagent_module = types.ModuleType("platoon.agents.actions.subagent")
    subagent_module.EXCLUDE_FROM_TRAINING_MISC_KEY = "exclude_from_training"
    subagent_module.EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY = "exclude_from_policy_training"
    subagent_module.SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY = "subagent_reward_verifier_task"
    monkeypatch.setitem(sys.modules, "platoon.agents.actions.subagent", subagent_module)

    name = "platoon_tinker_subagent_sampling_data_test"
    spec = importlib.util.spec_from_file_location(
        name,
        REPO_ROOT / "platoon/utils/tinker_data_processing.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_tinker_batch_transforms():
    name = "platoon_tinker_subagent_sampling_transform_test"
    spec = importlib.util.spec_from_file_location(
        name,
        REPO_ROOT / "platoon/train/tinker/batch_transforms.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _RecordingTracker:
    def __init__(self):
        self.scalars: list[dict[str, float]] = []
        self.denominators: dict[str, torch.Tensor] = {}
        self.stats: dict[str, tuple[torch.Tensor, str]] = {}

    def scalar(self, **kwargs):
        self.scalars.append(kwargs)

    def denominator(self, **kwargs):
        self.denominators.update(kwargs)

    def stat(self, denominator, **kwargs):
        for key, value in kwargs.items():
            self.stats[key] = (value, denominator)


def _load_tinker_group_workflow(monkeypatch, processing, tracker):
    env_module = types.ModuleType("platoon.envs.base")
    env_module.Task = object
    monkeypatch.setitem(sys.modules, "platoon.envs.base", env_module)

    config_module = types.ModuleType("platoon.train.tinker.config_defs")
    config_module.RolloutConfig = object
    config_module.WorkflowConfig = object
    monkeypatch.setitem(sys.modules, "platoon.train.tinker.config_defs", config_module)

    proxy_module = types.ModuleType("platoon.train.tinker.proxy")
    proxy_module.ModelInfo = object
    proxy_module.TinkerLLMProxySession = object
    proxy_module.TinkerLLMInteraction = object
    monkeypatch.setitem(sys.modules, "platoon.train.tinker.proxy", proxy_module)

    tracker_module = types.ModuleType("platoon.utils.stats_tracker")
    tracker_module.get = lambda _name: tracker
    monkeypatch.setitem(sys.modules, "platoon.utils.stats_tracker", tracker_module)

    processing_module = types.ModuleType("platoon.utils.tinker_data_processing")
    processing_module.SubagentDatumSamplingStats = processing.SubagentDatumSamplingStats
    processing_module.TrajectoryCollectionResult = processing.TrajectoryCollectionResult
    processing_module.TrajectoryStats = processing.TrajectoryStats
    processing_module.get_train_data_for_trajectory_collection = processing.get_train_data_for_trajectory_collection
    monkeypatch.setitem(sys.modules, "platoon.utils.tinker_data_processing", processing_module)

    name = "platoon_tinker_subagent_sampling_workflow_test"
    spec = importlib.util.spec_from_file_location(
        name,
        REPO_ROOT / "platoon/train/tinker/workflows/group_rollout_workflow.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _datum(*, attention_tokens: int, loss_tokens: int, depth: int | None, start: bool) -> DummyDatum:
    loss_mask = torch.zeros(attention_tokens, dtype=torch.float32)
    loss_mask[:loss_tokens] = 1.0
    loss_fn_inputs = {
        "mask": DummyTensorData(loss_mask),
        "advantages": DummyTensorData(torch.ones(attention_tokens)),
    }
    if depth is not None:
        loss_fn_inputs["traj_depth"] = DummyTensorData(torch.tensor([depth], dtype=torch.long))
        loss_fn_inputs["traj_start"] = DummyTensorData(torch.tensor([1.0 if start else 0.0]))
    return DummyDatum(DummyModelInput(attention_tokens), loss_fn_inputs)


def _install_fake_trajectory_conversion(module, events):
    def fake_trajectory_to_data(*, trajectory, trajectory_id, traj_depth, **kwargs):
        del kwargs
        events.append(f"convert:{trajectory_id}")
        datums = [
            _datum(
                attention_tokens=attention_tokens,
                loss_tokens=loss_tokens,
                depth=traj_depth,
                start=index == 0,
            )
            for index, (attention_tokens, loss_tokens) in enumerate(trajectory["datum_specs"])
        ]
        return module.TrajectoryDataResult(
            datums=datums,
            num_steps=len(datums),
            num_input_tokens=sum(spec[0] for spec in trajectory["datum_specs"]),
            num_output_tokens=sum(spec[1] for spec in trajectory["datum_specs"]),
        )

    module.trajectory_to_data = fake_trajectory_to_data


@pytest.mark.parametrize(
    "error_event",
    [
        {"kind": "AgentErrorEvent", "error": "unknown tool"},
        {
            "kind": "UserRejectObservation",
            "rejection_source": "hook",
            "rejection_reason": "blocked by policy",
        },
        {
            "kind": "ObservationEvent",
            "observation": {
                "kind": "ProgrammaticToolCallingObservation",
                "is_error": True,
            },
        },
    ],
)
def test_tinker_marks_typed_openhands_errors_by_completion_id(monkeypatch, error_event):
    processing = _load_tinker_data_processing(monkeypatch)
    interactions = {
        "completion-a": types.SimpleNamespace(
            obs=types.SimpleNamespace(chunks=[DummyEncodedTextChunk([10, 11])]),
            action=types.SimpleNamespace(tokens=[12, 13], logprobs=[-0.1, -0.1]),
        ),
        "completion-b": types.SimpleNamespace(
            obs=types.SimpleNamespace(chunks=[DummyEncodedTextChunk([20, 21])]),
            action=types.SimpleNamespace(tokens=[22], logprobs=[-0.1]),
        ),
    }
    step_a = {"misc": {"action_misc": {"completion_id": "completion-a"}}}
    trajectory = {
        "steps": [
            step_a,
            {
                **step_a,
                "observation_events": {"observation_events": [[error_event]]},
            },
            {"misc": {"action_misc": {"completion_id": "completion-b"}}},
        ]
    }

    result = processing.trajectory_to_data(
        trajectory=trajectory,
        interactions=interactions,
        task_id="task",
        trajectory_id="trajectory",
        trajectory_advantage=-0.25,
        checkpoint_version=3,
        filter_errors=True,
        trajectory_reward=-0.25,
    )

    # The later error occurrence marks the earlier clean-looking occurrence of
    # the same parallel response, but reward-sign filtering is deferred.
    assert result.num_steps == 2
    assert result.num_input_tokens == 4
    assert result.num_output_tokens == 3
    assert len(result.datums) == 2
    sidechannel = "_platoon_error_action_mask"
    assert result.datums[0].model_input.length == 3
    assert torch.equal(
        result.datums[0].loss_fn_inputs[sidechannel].to_torch(),
        torch.tensor([False, True, True]),
    )
    assert not result.datums[1].loss_fn_inputs[sidechannel].to_torch().any()


def test_tinker_error_filter_uses_final_centered_advantage_sign(monkeypatch):
    processing = _load_tinker_data_processing(monkeypatch)
    tracker = _RecordingTracker()
    workflow_module = _load_tinker_group_workflow(monkeypatch, processing, tracker)
    sidechannel = workflow_module.ERROR_ACTION_MASK_KEY

    mixed = _datum(attention_tokens=4, loss_tokens=4, depth=None, start=False)
    mixed.loss_fn_inputs["advantages"] = DummyTensorData(
        torch.tensor([1.0, -1.0, 0.0, 2.0])
    )
    mixed.loss_fn_inputs[sidechannel] = DummyTensorData(torch.ones(4, dtype=torch.bool))

    keep, metrics = workflow_module._filter_positive_centered_error_tokens(mixed)

    assert keep is True
    assert sidechannel not in mixed.loss_fn_inputs
    assert torch.equal(
        mixed.loss_fn_inputs["mask"].to_torch(),
        torch.tensor([0.0, 1.0, 1.0, 0.0]),
    )
    assert metrics == {
        "detected_action_tokens": 4.0,
        "suppressed_positive_action_tokens": 2.0,
        "retained_nonpositive_action_tokens": 2.0,
        "emptied_datums": 0.0,
    }

    all_positive = _datum(attention_tokens=2, loss_tokens=2, depth=None, start=False)
    all_positive.loss_fn_inputs["advantages"] = DummyTensorData(torch.tensor([0.5, 1.5]))
    all_positive.loss_fn_inputs[sidechannel] = DummyTensorData(torch.ones(2, dtype=torch.bool))

    keep, metrics = workflow_module._filter_positive_centered_error_tokens(all_positive)

    assert keep is False
    assert not all_positive.loss_fn_inputs["mask"].to_torch().any()
    assert metrics["emptied_datums"] == 1.0


def test_tinker_group_filters_errors_only_after_leave_one_out_centering(monkeypatch):
    processing = _load_tinker_data_processing(monkeypatch)
    tracker = _RecordingTracker()
    workflow_module = _load_tinker_group_workflow(monkeypatch, processing, tracker)
    sidechannel = workflow_module.ERROR_ACTION_MASK_KEY

    def result(trajectory_id: str, reward: float):
        datum = _datum(attention_tokens=2, loss_tokens=2, depth=None, start=False)
        datum.loss_fn_inputs["advantages"] = DummyTensorData(torch.full((2,), reward))
        datum.loss_fn_inputs[sidechannel] = DummyTensorData(torch.ones(2, dtype=torch.bool))
        return processing.TrajectoryCollectionResult(
            datums=[datum],
            task_reward=reward,
            trajectory_stats=[
                processing.TrajectoryStats(
                    trajectory_id=trajectory_id,
                    reward=reward,
                    num_steps=1,
                    num_input_tokens=2,
                    num_output_tokens=2,
                    num_datums=1,
                    is_root=True,
                )
            ],
            root_rewards_dict={"reward/success": reward},
        )

    high = result("high", 2.0)
    low = result("low", 0.0)
    workflow = workflow_module.GroupRolloutWorkflow.__new__(workflow_module.GroupRolloutWorkflow)
    workflow.config = types.SimpleNamespace(group_size=2, leave_one_out_baseline=True)
    workflow.tracker = tracker
    workflow.subagent_datum_sampler = None

    async def fake_single(_data, rollout_number):
        return [high, low][rollout_number]

    workflow.arun_episode_single = fake_single
    output = asyncio.run(workflow.arun_episode({"task_id": "task"}))

    # LOO produces +2 for high and -2 for low. The same typed error is removed
    # from the positive member but retained as useful negative policy signal.
    assert output == [low.datums[0]]
    assert high.datums == []
    assert sidechannel not in low.datums[0].loss_fn_inputs
    assert torch.equal(
        low.datums[0].loss_fn_inputs["advantages"].to_torch(),
        torch.tensor([-2.0, -2.0]),
    )
    assert torch.equal(low.datums[0].loss_fn_inputs["mask"].to_torch(), torch.ones(2))
    assert tracker.scalars[-1] == {
        "error_filter/detected_action_tokens": 4.0,
        "error_filter/suppressed_positive_action_tokens": 2.0,
        "error_filter/retained_nonpositive_action_tokens": 2.0,
        "error_filter/emptied_datums": 1.0,
    }


def test_tinker_raw_workload_counts_full_recursive_tree_and_unique_proxy_calls(monkeypatch):
    processing = _load_tinker_data_processing(monkeypatch)
    tracker = _RecordingTracker()
    workflow_module = _load_tinker_group_workflow(monkeypatch, processing, tracker)
    collection = {
        "trajectories": {
            "root": {"steps": [{}, {}]},
            "completed-child": {"steps": [{}]},
            "interrupted-child": {"steps": [{}, {}, {}]},
            "excluded-verifier": {"steps": [{}, {}]},
        }
    }
    interactions = {
        "completion-a": types.SimpleNamespace(
            obs=types.SimpleNamespace(length=11),
            action=types.SimpleNamespace(tokens=[1, 2, 3]),
        ),
        "completion-b": types.SimpleNamespace(
            obs=types.SimpleNamespace(length=17),
            action=types.SimpleNamespace(tokens=[4, 5]),
        ),
    }

    workload = workflow_module._workload_from_raw_rollout(collection, interactions)

    assert workload.environment_steps == 8
    assert workload.model_calls == 2
    assert workload.input_tokens == 28
    assert workload.output_tokens == 5
    assert workload.total_tokens == 33
    assert workload.trajectories == 4


def test_tinker_empty_task_output_carries_exact_workload_and_task_distributions(monkeypatch):
    processing = _load_tinker_data_processing(monkeypatch)
    tracker = _RecordingTracker()
    workflow_module = _load_tinker_group_workflow(monkeypatch, processing, tracker)
    workflow = workflow_module.GroupRolloutWorkflow.__new__(workflow_module.GroupRolloutWorkflow)
    workflow.config = types.SimpleNamespace(group_size=3)
    workflow.tracker = tracker
    outcomes = [
        workflow_module._SingleRolloutOutcome(
            result=None,
            workload=workflow_module.RolloutWorkload(
                environment_steps=2,
                model_calls=1,
                input_tokens=10,
                output_tokens=3,
                trajectories=1,
                postmerge_datums=5,
                policy_eligible_datums=4,
                post_sampling_datums=2,
            ),
            observed=True,
        ),
        workflow_module._SingleRolloutOutcome(
            result=None,
            workload=workflow_module.RolloutWorkload(
                environment_steps=5,
                model_calls=2,
                input_tokens=20,
                output_tokens=7,
                trajectories=3,
                postmerge_datums=4,
                policy_eligible_datums=4,
                post_sampling_datums=3,
            ),
            observed=True,
        ),
        workflow_module._SingleRolloutOutcome(
            result=None,
            workload=workflow_module.RolloutWorkload(),
            observed=False,
        ),
    ]

    output = workflow._make_task_output([], outcomes=outcomes, trainable_rollouts=0)

    assert isinstance(output, list)
    assert output == []
    assert output.workload == workflow_module.RolloutWorkload(
        environment_steps=7,
        model_calls=3,
        input_tokens=30,
        output_tokens=10,
        trajectories=4,
        postmerge_datums=9,
        policy_eligible_datums=8,
        post_sampling_datums=5,
    )
    assert output.requested_rollouts == 3
    assert output.observed_rollouts == 2
    assert output.trainable_rollouts == 0
    rollout_trajectories, rollout_denominator = tracker.stats["workload/rollout/total_trajectories"]
    task_trajectories, task_denominator = tracker.stats["workload/task/total_trajectories"]
    assert torch.equal(rollout_trajectories, torch.tensor([1.0, 3.0, 0.0]))
    assert torch.equal(task_trajectories, torch.tensor([4.0]))
    assert rollout_denominator == "workload/rollout/count"
    assert task_denominator == "workload/task/count"
    task_retained, retained_denominator = tracker.stats["workload/task/total_task_retained_datums"]
    task_trainable, _ = tracker.stats[
        "workload/task/total_task_workflow_trainable_datums"
    ]
    task_non_trainable, _ = tracker.stats[
        "workload/task/total_task_workflow_non_trainable_datums"
    ]
    assert torch.equal(task_retained, torch.tensor([0.0]))
    assert torch.equal(task_trainable, torch.tensor([0.0]))
    assert torch.equal(task_non_trainable, torch.tensor([9.0]))
    assert retained_denominator == "workload/task/count"


def test_tinker_sampling_keeps_roots_allows_empty_children_and_preserves_full_tree_rewards(monkeypatch):
    processing = _load_tinker_data_processing(monkeypatch)
    events: list[str] = []
    _install_fake_trajectory_conversion(processing, events)
    collection = {
        "trajectories": {
            "root": {
                "name": "root",
                "reward": 1.0,
                "steps": [],
                "datum_specs": [(5, 2), (7, 3)],
            },
            "child-a": {
                "name": "child-a",
                "reward": 0.5,
                "parent_info": {"id": "root"},
                "steps": [],
                "datum_specs": [(11, 4), (13, 5)],
            },
            "child-b": {
                "name": "child-b",
                "reward": 0.25,
                "parent_info": {"id": "root"},
                "steps": [],
                "datum_specs": [(17, 6)],
            },
        }
    }

    def reward_processor(trajectory):
        events.append(f"reward:{trajectory['name']}")
        return trajectory["reward"], {"reward/success": trajectory["reward"]}

    result = processing.get_train_data_for_trajectory_collection(
        trajectory_collection=collection,
        interactions={},
        task_id="task",
        checkpoint_version=3,
        reward_processor=reward_processor,
        include_traj_depth=True,
        include_traj_start=True,
        subagent_datum_sampler=DeterministicSubagentDatumSampler(keep_probability=0.0, seed=7),
    )

    # All trajectory rewards are computed before datum conversion/sampling.
    assert events == [
        "reward:root",
        "reward:child-a",
        "reward:child-b",
        "convert:root",
        "convert:child-a",
        "convert:child-b",
    ]
    assert len(result.datums) == 2
    assert [int(d.loss_fn_inputs["traj_depth"].to_torch().item()) for d in result.datums] == [0, 0]
    assert [float(d.loss_fn_inputs["traj_start"].to_torch().item()) for d in result.datums] == [1.0, 0.0]
    assert result.task_reward == 1.0
    assert result.task_reward_valid is True
    assert [stats.num_datums for stats in result.trajectory_stats] == [2, 2, 1]
    assert [stats.reward for stats in result.trajectory_stats] == [1.0, 0.5, 0.25]

    metrics = result.subagent_sampling_stats.to_metrics()
    assert metrics["subagent_sampling/eligible_datums"] == 5.0
    assert metrics["subagent_sampling/retained_datums"] == 2.0
    assert metrics["subagent_sampling/eligible_attention_tokens"] == 53.0
    assert metrics["subagent_sampling/retained_attention_tokens"] == 12.0
    assert metrics["subagent_sampling/eligible_loss_tokens"] == 20.0
    assert metrics["subagent_sampling/retained_loss_tokens"] == 5.0
    assert metrics["subagent_sampling/depth_0/eligible_datums"] == 2.0
    assert metrics["subagent_sampling/depth_0/retained_datums"] == 2.0
    assert metrics["subagent_sampling/depth_1/eligible_datums"] == 3.0
    assert metrics["subagent_sampling/depth_1/retained_datums"] == 0.0


def test_tinker_sampling_repairs_first_retained_trajectory_boundary(monkeypatch):
    processing = _load_tinker_data_processing(monkeypatch)
    _install_fake_trajectory_conversion(processing, [])

    class FixedSampler:
        def sample_mask(self, *, task_id, trajectory_id, depth, num_datums):
            assert task_id == "task"
            if trajectory_id == "root":
                assert depth == 0 and num_datums == 1
                return [True]
            assert trajectory_id == "child"
            assert depth == 1 and num_datums == 2
            return [False, True]

    collection = {
        "trajectories": {
            "root": {"reward": 1.0, "steps": [], "datum_specs": [(5, 2)]},
            "child": {
                "reward": 0.5,
                "parent_info": {"id": "root"},
                "steps": [],
                "datum_specs": [(7, 3), (11, 4)],
            },
        }
    }
    result = processing.get_train_data_for_trajectory_collection(
        trajectory_collection=collection,
        interactions={},
        task_id="task",
        checkpoint_version=3,
        include_traj_depth=True,
        include_traj_start=True,
        subagent_datum_sampler=FixedSampler(),
    )

    assert len(result.datums) == 2
    retained_child = result.datums[1]
    assert int(retained_child.loss_fn_inputs["traj_depth"].to_torch().item()) == 1
    assert float(retained_child.loss_fn_inputs["traj_start"].to_torch().item()) == 1.0
    assert result.trajectory_stats[1].num_datums == 2
    metrics = result.subagent_sampling_stats.to_metrics()
    assert metrics["subagent_sampling/depth_1/eligible_attention_tokens"] == 18.0
    assert metrics["subagent_sampling/depth_1/retained_attention_tokens"] == 11.0
    assert metrics["subagent_sampling/depth_1/eligible_loss_tokens"] == 7.0
    assert metrics["subagent_sampling/depth_1/retained_loss_tokens"] == 4.0

    # The repaired marker makes depth normalization operate on the retained
    # batch. If the dropped first child datum still owned traj_start, depth 1
    # would have zero trajectories here and its advantages would be zeroed.
    batch_transforms = _load_tinker_batch_transforms()
    context = batch_transforms.BatchTransformContext(
        config=types.SimpleNamespace(train=types.SimpleNamespace(workflow_config=types.SimpleNamespace())),
        train_step=0,
        minibatch_num=0,
        microbatch_num=0,
    )
    batch_transforms.DepthLevelWeightingTransform()(result.datums, context)
    for datum in result.datums:
        assert torch.equal(
            datum.loss_fn_inputs["advantages"].to_torch(),
            torch.ones(datum.model_input.length),
        )


def test_tinker_policy_ineligible_child_keeps_stats_but_not_datums_or_sampling_metrics(monkeypatch):
    processing = _load_tinker_data_processing(monkeypatch)
    events: list[str] = []
    _install_fake_trajectory_conversion(processing, events)

    class RecordingSampler:
        def __init__(self):
            self.trajectory_ids: list[str] = []

        def sample_mask(self, *, task_id, trajectory_id, depth, num_datums):
            assert task_id == "task"
            self.trajectory_ids.append(trajectory_id)
            return [True] * num_datums

    sampler = RecordingSampler()
    collection = {
        "trajectories": {
            # The source marker is child-only, but a stray root marker must not
            # violate the mandatory-root invariant at the converter boundary.
            "root": {
                "name": "root",
                "reward": 1.0,
                "misc": {"exclude_from_policy_training": True},
                "steps": [],
                "datum_specs": [(5, 2)],
            },
            "valid-child": {
                "name": "valid-child",
                "reward": 0.5,
                "parent_info": {"id": "root"},
                "steps": [],
                "datum_specs": [(7, 3)],
            },
            "invalid-child": {
                "name": "invalid-child",
                "reward": 0.25,
                "parent_info": {"id": "root"},
                "misc": {"exclude_from_policy_training": True},
                "steps": [],
                "datum_specs": [(11, 4)],
            },
        }
    }

    def reward_processor(trajectory):
        events.append(f"reward:{trajectory['name']}")
        return trajectory["reward"], {"reward/success": trajectory["reward"]}

    result = processing.get_train_data_for_trajectory_collection(
        trajectory_collection=collection,
        interactions={},
        task_id="task",
        checkpoint_version=3,
        reward_processor=reward_processor,
        include_traj_depth=True,
        include_traj_start=True,
        subagent_datum_sampler=sampler,
    )

    assert sampler.trajectory_ids == ["root", "valid-child"]
    assert [datum.model_input.length for datum in result.datums] == [5, 7]
    assert result.num_policy_excluded_datums == 1
    assert [stats.trajectory_id for stats in result.trajectory_stats] == [
        "root",
        "valid-child",
        "invalid-child",
    ]
    assert [stats.num_datums for stats in result.trajectory_stats] == [1, 1, 1]
    assert events[:3] == ["reward:root", "reward:valid-child", "reward:invalid-child"]
    metrics = result.subagent_sampling_stats.to_metrics()
    assert metrics["subagent_sampling/eligible_datums"] == 2.0
    assert metrics["subagent_sampling/retained_datums"] == 2.0
    assert metrics["subagent_sampling/eligible_attention_tokens"] == 12.0
    assert "subagent_sampling/depth_1/eligible_datums" in metrics
    assert metrics["subagent_sampling/depth_1/eligible_datums"] == 1.0


def test_tinker_cancelled_root_keeps_completed_child_and_full_stats(monkeypatch):
    processing = _load_tinker_data_processing(monkeypatch)
    _install_fake_trajectory_conversion(processing, [])

    class RecordingSampler:
        def __init__(self):
            self.trajectory_ids = []

        def sample_mask(self, *, task_id, trajectory_id, depth, num_datums):
            _ = task_id, depth
            self.trajectory_ids.append(trajectory_id)
            return [True] * num_datums

    sampler = RecordingSampler()
    collection = {
        "trajectories": {
            "root": {
                "reward": 0.0,
                "error_message": "Episode cancelled at step 4\nCancelledError",
                "misc": {TRAJECTORY_CANCELLED_MISC_KEY: True},
                "steps": [],
                "datum_specs": [(5, 2)],
            },
            "completed-child": {
                "reward": 0.75,
                "parent_info": {"id": "root"},
                "steps": [],
                "datum_specs": [(7, 3)],
            },
        }
    }

    result = processing.get_train_data_for_trajectory_collection(
        trajectory_collection=collection,
        interactions={},
        task_id="task",
        checkpoint_version=3,
        include_traj_depth=True,
        include_traj_start=True,
        subagent_datum_sampler=sampler,
    )

    assert sampler.trajectory_ids == ["completed-child"]
    assert [datum.model_input.length for datum in result.datums] == [7]
    assert [stats.trajectory_id for stats in result.trajectory_stats] == [
        "root",
        "completed-child",
    ]
    assert result.num_policy_excluded_datums == 1
    assert result.task_reward_valid is False
    metrics = result.subagent_sampling_stats.to_metrics()
    assert metrics["subagent_sampling/eligible_datums"] == 1.0
    assert metrics["subagent_sampling/depth_1/eligible_datums"] == 1.0


def test_tinker_partial_root_reward_is_metric_only_and_completed_child_uses_valid_baseline(monkeypatch):
    processing = _load_tinker_data_processing(monkeypatch)
    _install_fake_trajectory_conversion(processing, [])
    partial_result = processing.get_train_data_for_trajectory_collection(
        trajectory_collection={
            "trajectories": {
                "interrupted-root": {
                    "reward": 9.0,
                    "misc": {TRAJECTORY_CANCELLED_MISC_KEY: True},
                    "steps": [],
                    "datum_specs": [(3, 2)],
                },
                "completed-child": {
                    "reward": 0.75,
                    "parent_info": {"id": "interrupted-root"},
                    "steps": [],
                    "datum_specs": [(3, 2)],
                },
            }
        },
        interactions={},
        task_id="task",
        checkpoint_version=3,
    )
    assert partial_result.task_reward_valid is False
    assert [stats.trajectory_id for stats in partial_result.trajectory_stats] == [
        "interrupted-root",
        "completed-child",
    ]
    assert len(partial_result.datums) == 1
    child_datum = partial_result.datums[0]
    child_datum.loss_fn_inputs["advantages"] = DummyTensorData(torch.tensor([0.75, 0.75, 0.0]))

    completed_root_datum = _datum(attention_tokens=3, loss_tokens=2, depth=None, start=False)
    completed_root_datum.loss_fn_inputs["advantages"] = DummyTensorData(torch.tensor([1.0, 1.0, 0.0]))
    completed_result = processing.TrajectoryCollectionResult(
        datums=[completed_root_datum],
        task_reward=1.0,
        trajectory_stats=[
            processing.TrajectoryStats(
                trajectory_id="completed-root",
                reward=1.0,
                num_steps=1,
                num_input_tokens=3,
                num_output_tokens=2,
                num_datums=1,
                is_root=True,
            )
        ],
        root_rewards_dict={},
    )

    tracker = _RecordingTracker()
    workflow_module = _load_tinker_group_workflow(monkeypatch, processing, tracker)
    workflow = workflow_module.GroupRolloutWorkflow.__new__(workflow_module.GroupRolloutWorkflow)
    workflow.config = types.SimpleNamespace(
        group_size=2,
        leave_one_out_baseline=True,
        filter_zero_advantage_datums=True,
    )
    workflow.tracker = tracker
    workflow.subagent_datum_sampler = None
    results = [partial_result, completed_result]

    async def fake_single(_data, rollout_number):
        return results[rollout_number]

    workflow.arun_episode_single = fake_single
    output = asyncio.run(workflow.arun_episode({"task_id": "task"}))

    assert output == [child_datum, completed_root_datum]
    # The interrupted reward 9 remains visible in metrics, but it is not a
    # baseline candidate. The partial member's completed child uses mean(valid)
    # = 1, and the singleton valid member falls back to its own reward.
    assert torch.equal(
        child_datum.loss_fn_inputs["advantages"].to_torch(),
        torch.tensor([-0.25, -0.25, 0.0]),
    )
    assert torch.equal(
        completed_root_datum.loss_fn_inputs["advantages"].to_torch(),
        torch.zeros(3),
    )
    transform_module = _load_tinker_batch_transforms()
    output = transform_module.filter_zero_advantage_datums(output)
    assert output == [child_datum]
    transform_module.set_loss_normalization_token_counts(output, represented_loss_tokens=4)
    normalization_key = transform_module.LOSS_NORMALIZATION_TOKENS_KEY
    # Two child action tokens remain, but the denominator still includes the
    # two zero-gradient root action tokens which were removed from model compute.
    assert float(child_datum.loss_fn_inputs[normalization_key].to_torch().item()) == 4.0
    task_reward_values, task_reward_denominator = tracker.stats["task_reward"]
    assert torch.equal(task_reward_values, torch.tensor([9.0, 1.0]))
    assert task_reward_denominator == "task_reward_mask"


def test_tinker_zero_advantage_datum_filter_can_be_disabled(monkeypatch):
    processing = _load_tinker_data_processing(monkeypatch)
    tracker = _RecordingTracker()
    workflow_module = _load_tinker_group_workflow(monkeypatch, processing, tracker)

    zero_datum = _datum(attention_tokens=2, loss_tokens=2, depth=None, start=False)
    zero_datum.loss_fn_inputs["advantages"] = DummyTensorData(torch.tensor([1.0, 1.0]))
    signal_datum = _datum(attention_tokens=2, loss_tokens=2, depth=None, start=False)
    signal_datum.loss_fn_inputs["advantages"] = DummyTensorData(torch.tensor([0.5, 0.5]))
    results = [
        processing.TrajectoryCollectionResult(
            datums=[zero_datum],
            task_reward=1.0,
            trajectory_stats=[
                processing.TrajectoryStats(
                    trajectory_id="root",
                    reward=1.0,
                    num_steps=1,
                    num_input_tokens=2,
                    num_output_tokens=2,
                    num_datums=1,
                    is_root=True,
                )
            ],
            root_rewards_dict={},
        ),
        # A partial member's completed child uses mean(valid roots)=1 as its
        # baseline, while its interrupted root reward remains metric-only.
        processing.TrajectoryCollectionResult(
            datums=[signal_datum],
            task_reward=9.0,
            trajectory_stats=[
                processing.TrajectoryStats(
                    trajectory_id="child",
                    reward=0.5,
                    num_steps=1,
                    num_input_tokens=2,
                    num_output_tokens=2,
                    num_datums=1,
                    is_root=False,
                )
            ],
            root_rewards_dict={},
            task_reward_valid=False,
        ),
    ]

    workflow = workflow_module.GroupRolloutWorkflow.__new__(workflow_module.GroupRolloutWorkflow)
    workflow.config = types.SimpleNamespace(
        group_size=2,
        leave_one_out_baseline=True,
        filter_zero_advantage_datums=False,
    )
    workflow.tracker = tracker
    workflow.subagent_datum_sampler = None

    async def fake_single(_data, rollout_number):
        return results[rollout_number]

    workflow.arun_episode_single = fake_single
    output = asyncio.run(workflow.arun_episode({"task_id": "task"}))

    assert output == [zero_datum, signal_datum]
    assert "_loss_normalization_tokens" not in zero_datum.loss_fn_inputs
    assert "_loss_normalization_tokens" not in signal_datum.loss_fn_inputs
    assert torch.equal(zero_datum.loss_fn_inputs["advantages"].to_torch(), torch.zeros(2))
    assert torch.equal(signal_datum.loss_fn_inputs["advantages"].to_torch(), torch.tensor([-0.5, -0.5]))


@pytest.mark.parametrize("filter_enabled", [False, True])
def test_tinker_workflow_retains_an_all_zero_group_until_batch_transforms(
    monkeypatch,
    filter_enabled,
):
    processing = _load_tinker_data_processing(monkeypatch)
    tracker = _RecordingTracker()
    workflow_module = _load_tinker_group_workflow(monkeypatch, processing, tracker)

    def make_result(trajectory_id):
        datum = _datum(attention_tokens=2, loss_tokens=2, depth=None, start=False)
        datum.loss_fn_inputs["advantages"] = DummyTensorData(torch.ones(2))
        return processing.TrajectoryCollectionResult(
            datums=[datum],
            task_reward=1.0,
            trajectory_stats=[
                processing.TrajectoryStats(
                    trajectory_id=trajectory_id,
                    reward=1.0,
                    num_steps=1,
                    num_input_tokens=2,
                    num_output_tokens=2,
                    num_datums=1,
                    is_root=True,
                )
            ],
            root_rewards_dict={},
        )

    results = [make_result("root-a"), make_result("root-b")]
    workflow = workflow_module.GroupRolloutWorkflow.__new__(workflow_module.GroupRolloutWorkflow)
    workflow.config = types.SimpleNamespace(
        group_size=2,
        leave_one_out_baseline=True,
        filter_zero_advantage_datums=filter_enabled,
    )
    workflow.tracker = tracker
    workflow.subagent_datum_sampler = None

    async def fake_single(_data, rollout_number):
        return results[rollout_number]

    workflow.arun_episode_single = fake_single
    output = asyncio.run(workflow.arun_episode({"task_id": "task"}))

    assert output == [results[0].datums[0], results[1].datums[0]]
    assert output.task_retained_datums == 2
    assert output.trainable_rollouts == 0
    assert all(torch.equal(datum.loss_fn_inputs["advantages"].to_torch(), torch.zeros(2)) for datum in output)
    assert all("_loss_normalization_tokens" not in datum.loss_fn_inputs for datum in output)


def test_tinker_zero_advantage_filter_runs_after_depth_weighting_and_preserves_denominator(monkeypatch):
    processing = _load_tinker_data_processing(monkeypatch)
    tracker = _RecordingTracker()
    workflow_module = _load_tinker_group_workflow(monkeypatch, processing, tracker)

    dropped = _datum(attention_tokens=2, loss_tokens=2, depth=1, start=True)
    dropped.loss_fn_inputs["advantages"] = DummyTensorData(torch.tensor([1.0, 1.0]))
    retained = _datum(attention_tokens=2, loss_tokens=2, depth=1, start=False)
    retained.loss_fn_inputs["advantages"] = DummyTensorData(torch.tensor([2.0, 2.0]))
    result = processing.TrajectoryCollectionResult(
        datums=[dropped, retained],
        task_reward=1.0,
        trajectory_stats=[
            processing.TrajectoryStats(
                trajectory_id="root",
                reward=1.0,
                num_steps=2,
                num_input_tokens=4,
                num_output_tokens=4,
                num_datums=2,
                is_root=True,
            )
        ],
        root_rewards_dict={},
    )

    workflow = workflow_module.GroupRolloutWorkflow.__new__(workflow_module.GroupRolloutWorkflow)
    workflow.config = types.SimpleNamespace(
        group_size=1,
        leave_one_out_baseline=True,
        filter_zero_advantage_datums=True,
    )
    workflow.tracker = tracker
    workflow.subagent_datum_sampler = None

    async def fake_single(_data, _rollout_number):
        return result

    workflow.arun_episode_single = fake_single
    output = asyncio.run(workflow.arun_episode({"task_id": "task"}))

    assert output == [dropped, retained]
    assert torch.equal(retained.loss_fn_inputs["advantages"].to_torch(), torch.ones(2))

    transform_module = _load_tinker_batch_transforms()
    context = transform_module.BatchTransformContext(
        config=types.SimpleNamespace(),
        train_step=0,
        minibatch_num=0,
        microbatch_num=0,
    )
    # Both datums participate in mathematical depth normalization; the
    # compute-only filter runs afterward.
    assert transform_module.DepthLevelWeightingTransform()(output, context) == [dropped, retained]
    output = transform_module.filter_zero_advantage_datums(output)
    assert output == [retained]
    assert float(retained.loss_fn_inputs["traj_start"].to_torch().item()) == 0.0
    transform_module.set_loss_normalization_token_counts(output, represented_loss_tokens=4)
    normalization_key = transform_module.LOSS_NORMALIZATION_TOKENS_KEY
    assert float(retained.loss_fn_inputs[normalization_key].to_torch().item()) == 4.0
    assert torch.equal(retained.loss_fn_inputs["advantages"].to_torch(), torch.ones(2))


def test_tinker_zero_advantage_filter_uses_action_mask_and_exact_zero(monkeypatch):
    _load_tinker_data_processing(monkeypatch)
    transform_module = _load_tinker_batch_transforms()
    datum = _datum(attention_tokens=3, loss_tokens=1, depth=None, start=False)

    # Prompt-token advantages do not matter, while even tiny action-token
    # advantages are legitimate signal.
    datum.loss_fn_inputs["advantages"] = DummyTensorData(torch.tensor([1e-12, 7.0, 0.0]))
    assert transform_module.has_zero_action_advantage(datum) is False

    datum.loss_fn_inputs["advantages"] = DummyTensorData(torch.tensor([0.0, 7.0, 0.0]))
    assert transform_module.has_zero_action_advantage(datum) is True

    datum.loss_fn_inputs["mask"] = DummyTensorData(torch.zeros(3))
    datum.loss_fn_inputs["advantages"] = DummyTensorData(torch.tensor([1.0, 2.0, 3.0]))
    assert transform_module.has_zero_action_advantage(datum) is True


def test_tinker_group_with_no_valid_root_reward_logs_full_stats_and_does_not_train(monkeypatch):
    processing = _load_tinker_data_processing(monkeypatch)
    tracker = _RecordingTracker()
    workflow_module = _load_tinker_group_workflow(monkeypatch, processing, tracker)
    child_datum = _datum(attention_tokens=2, loss_tokens=2, depth=None, start=False)
    child_datum.loss_fn_inputs["advantages"] = DummyTensorData(torch.tensor([0.75, 0.75]))
    partial_result = processing.TrajectoryCollectionResult(
        datums=[child_datum],
        task_reward=9.0,
        trajectory_stats=[
            processing.TrajectoryStats(
                trajectory_id="interrupted-root",
                reward=9.0,
                num_steps=1,
                num_input_tokens=2,
                num_output_tokens=2,
                num_datums=1,
                is_root=True,
            )
        ],
        root_rewards_dict={},
        task_reward_valid=False,
    )

    workflow = workflow_module.GroupRolloutWorkflow.__new__(workflow_module.GroupRolloutWorkflow)
    workflow.config = types.SimpleNamespace(group_size=1, leave_one_out_baseline=True)
    workflow.tracker = tracker
    workflow.subagent_datum_sampler = None

    async def fake_single(_data, _rollout_number):
        return partial_result

    workflow.arun_episode_single = fake_single
    output = asyncio.run(workflow.arun_episode({"task_id": "task"}))
    assert output == []
    assert output.requested_rollouts == 1
    assert output.observed_rollouts == 1
    assert output.trainable_rollouts == 0
    assert torch.equal(child_datum.loss_fn_inputs["advantages"].to_torch(), torch.tensor([0.75, 0.75]))
    task_reward_values, _ = tracker.stats["task_reward"]
    assert torch.equal(task_reward_values, torch.tensor([9.0]))


def test_tinker_sampling_does_not_keep_rollout_with_zero_pre_sampling_datums(monkeypatch, tmp_path):
    processing = _load_tinker_data_processing(monkeypatch)
    tracker = _RecordingTracker()
    workflow_module = _load_tinker_group_workflow(monkeypatch, processing, tracker)

    empty_result = processing.TrajectoryCollectionResult(
        datums=[],
        task_reward=1.0,
        trajectory_stats=[
            processing.TrajectoryStats(
                trajectory_id="root",
                reward=1.0,
                num_steps=0,
                num_input_tokens=0,
                num_output_tokens=0,
                num_datums=0,
                is_root=True,
            )
        ],
        root_rewards_dict={},
    )

    class FakeProxySession:
        interactions = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    workflow_module.TinkerLLMProxySession = FakeProxySession
    workflow_module.get_train_data_for_trajectory_collection = lambda **_kwargs: empty_result

    workflow = workflow_module.GroupRolloutWorkflow.__new__(workflow_module.GroupRolloutWorkflow)
    workflow.config = types.SimpleNamespace(depth_level_weighting=False)
    workflow.model_info = types.SimpleNamespace(llm=types.SimpleNamespace(version=7))
    workflow.filter_errors = False
    workflow.reward_processor = lambda trajectory: (trajectory.get("reward", 0.0), {})
    workflow.subagent_datum_sampler = object()
    workflow.get_task_fn = lambda _task_id: types.SimpleNamespace(max_steps=None)
    workflow._get_rollout_config = lambda: types.SimpleNamespace(
        max_steps=None,
        output_dir=str(tmp_path),
    )

    async def fake_rollout(_task, _config):
        return {"trajectories": {"root": {}}}

    workflow.rollout_fn = fake_rollout

    # Sampling must not make a corrupt/missing-interaction rollout count as a
    # reward-baseline member. There was nothing for Bernoulli sampling to drop.
    outcome = asyncio.run(workflow.arun_episode_single({"task_id": "task"}, 0))
    assert outcome.result is None
    assert outcome.observed is True
    assert outcome.workload.trajectories == 1


def test_tinker_partial_verifier_is_excluded_by_prelaunch_task_marker(monkeypatch):
    processing = _load_tinker_data_processing(monkeypatch)
    _install_fake_trajectory_conversion(processing, [])
    collection = {
        "trajectories": {
            "root": {
                "reward": 1.0,
                "steps": [],
                "datum_specs": [(5, 2)],
            },
            "partial-verifier": {
                "reward": 0.0,
                "parent_info": {"id": "root"},
                "task": {"misc": {"subagent_reward_verifier_task": True}},
                "steps": [],
                "datum_specs": [(7, 3)],
            },
        }
    }

    result = processing.get_train_data_for_trajectory_collection(
        trajectory_collection=collection,
        interactions={},
        task_id="task",
        checkpoint_version=3,
    )

    assert [datum.model_input.length for datum in result.datums] == [5]
    assert [stats.trajectory_id for stats in result.trajectory_stats] == ["root"]


@pytest.mark.parametrize(
    ("sampling_active", "num_policy_excluded_datums"),
    [(True, 0), (False, 1)],
)
def test_tinker_loo_and_optional_reward_stats_include_empty_valid_rollout(
    monkeypatch,
    sampling_active,
    num_policy_excluded_datums,
):
    processing = _load_tinker_data_processing(monkeypatch)
    tracker = _RecordingTracker()
    workflow_module = _load_tinker_group_workflow(monkeypatch, processing, tracker)
    retained_datum = _datum(attention_tokens=3, loss_tokens=2, depth=None, start=False)
    retained_datum.loss_fn_inputs["advantages"] = DummyTensorData(torch.tensor([1.0, 1.0, 0.0]))
    optional_reward = {"reward/subagent_judgment": 0.8}
    result_with_data = processing.TrajectoryCollectionResult(
        datums=[retained_datum],
        task_reward=1.0,
        trajectory_stats=[
            processing.TrajectoryStats(
                trajectory_id="root-a",
                reward=1.0,
                num_steps=1,
                num_input_tokens=3,
                num_output_tokens=2,
                num_datums=1,
                rewards_dict=optional_reward,
                is_root=True,
            )
        ],
        root_rewards_dict=optional_reward,
    )
    empty_sampled_result = processing.TrajectoryCollectionResult(
        datums=[],
        task_reward=0.0,
        trajectory_stats=[
            processing.TrajectoryStats(
                trajectory_id="root-b",
                reward=0.0,
                num_steps=1,
                num_input_tokens=3,
                num_output_tokens=2,
                num_datums=1,
                rewards_dict={},
                is_root=True,
            )
        ],
        root_rewards_dict={},
        num_policy_excluded_datums=num_policy_excluded_datums,
    )

    workflow = workflow_module.GroupRolloutWorkflow.__new__(workflow_module.GroupRolloutWorkflow)
    workflow.config = types.SimpleNamespace(group_size=2, leave_one_out_baseline=True)
    workflow.tracker = tracker
    workflow.subagent_datum_sampler = object() if sampling_active else None
    results = [result_with_data, empty_sampled_result]

    async def fake_single(_data, rollout_number):
        return results[rollout_number]

    workflow.arun_episode_single = fake_single
    output = asyncio.run(workflow.arun_episode({"task_id": "task"}))

    assert output == [retained_datum]
    # With leave-one-out over both rollout rewards, the retained reward-1 datum
    # subtracts the other rollout's reward 0. Dropping the empty result would
    # instead subtract 1 and incorrectly zero these action-token advantages.
    assert torch.equal(
        retained_datum.loss_fn_inputs["advantages"].to_torch(),
        torch.tensor([1.0, 1.0, 0.0]),
    )
    root_values, root_denominator = tracker.stats["root_reward/subagent_judgment"]
    child_values, child_denominator = tracker.stats["reward/subagent_judgment"]
    assert torch.equal(root_values, torch.tensor([0.8]))
    assert torch.equal(child_values, torch.tensor([0.8]))
    assert root_denominator == "root_reward/subagent_judgment_mask"
    assert child_denominator == "reward/subagent_judgment_mask"


def test_tinker_zero_signal_detection_ignores_policy_ineligible_child_reward(monkeypatch):
    processing = _load_tinker_data_processing(monkeypatch)
    tracker = _RecordingTracker()
    workflow_module = _load_tinker_group_workflow(monkeypatch, processing, tracker)

    def root_result(trajectory_id):
        datum = _datum(attention_tokens=2, loss_tokens=2, depth=None, start=False)
        datum.loss_fn_inputs["advantages"] = DummyTensorData(torch.tensor([1.0, 1.0]))
        return processing.TrajectoryCollectionResult(
            datums=[datum],
            task_reward=1.0,
            trajectory_stats=[
                processing.TrajectoryStats(
                    trajectory_id=trajectory_id,
                    reward=1.0,
                    num_steps=1,
                    num_input_tokens=2,
                    num_output_tokens=2,
                    num_datums=1,
                    is_root=True,
                )
            ],
            root_rewards_dict={},
        )

    results = [root_result("root-a"), root_result("root-b")]
    # This child remains in full reward/stat reporting, but its failed verifier
    # removed all of its policy datums. Its different reward must not defeat the
    # zero-signal check on the two retained reward-1 roots.
    results[0].trajectory_stats.append(
        processing.TrajectoryStats(
            trajectory_id="invalid-child",
            reward=0.0,
            num_steps=1,
            num_input_tokens=3,
            num_output_tokens=1,
            num_datums=1,
            is_root=False,
        )
    )
    results[0].num_policy_excluded_datums = 1

    workflow = workflow_module.GroupRolloutWorkflow.__new__(workflow_module.GroupRolloutWorkflow)
    workflow.config = types.SimpleNamespace(group_size=2, leave_one_out_baseline=True)
    workflow.tracker = tracker
    workflow.subagent_datum_sampler = None

    async def fake_single(_data, rollout_number):
        return results[rollout_number]

    workflow.arun_episode_single = fake_single

    output = asyncio.run(workflow.arun_episode({"task_id": "task"}))
    # The workflow correctly recognizes no trainable rollout, but retains the
    # candidates until cross-task depth weighting and denominator accounting;
    # the trainer-side filter removes them before model compute.
    assert output == [results[0].datums[0], results[1].datums[0]]
    assert output.requested_rollouts == 2
    assert output.observed_rollouts == 2
    assert output.trainable_rollouts == 0
    for result in results:
        assert torch.equal(
            result.datums[0].loss_fn_inputs["advantages"].to_torch(),
            torch.zeros(2),
        )


def test_tinker_workflow_sampling_config_defaults_off_and_validates_inputs():
    name = "platoon_tinker_sampling_config_test"
    spec = importlib.util.spec_from_file_location(
        name,
        REPO_ROOT / "platoon/train/tinker/config_defs.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    config = module.WorkflowConfig()
    assert config.subagent_datum_keep_probability == 1.0
    assert config.subagent_datum_sampling_seed == 0
    assert config.filter_zero_advantage_datums is True
    assert config.filter_errors is True
    assert module.EvalConfig().workflow_config.filter_zero_advantage_datums is False
    assert module.EvalConfig().workflow_config.filter_errors is False
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        module.WorkflowConfig(subagent_datum_keep_probability=-0.1)
    with pytest.raises(ValueError, match="integer"):
        module.WorkflowConfig(subagent_datum_sampling_seed=True)


def test_tinker_non_submitted_metric_requires_complete_workload_metadata():
    """Legacy plain-list workflows must not yield a negative exact count."""

    source = (REPO_ROOT / "platoon/train/tinker/rl.py").read_text()
    tree = ast.parse(source)
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "batch_total_tasks_with_workload_metadata == batch_total_tasks" in ast.unparse(node.test)
    ]
    assert len(guards) == 1
    guarded_source = "\n".join(ast.unparse(statement) for statement in guards[0].body)
    assert "batch_workload.postmerge_datums - batch_submitted_training_datums" in guarded_source
    assert "workload/training_batch/total_non_submitted_datums" in guarded_source


def test_tinker_zero_filter_remains_after_depth_transform_and_runs_once():
    source = (REPO_ROOT / "platoon/train/tinker/rl.py").read_text()
    assert source.count("task_rollout_results = filter_zero_advantage_datums(task_rollout_results)") == 1
    assert source.index("task_rollout_results = run_batch_transforms(") < source.index(
        "task_rollout_results = filter_zero_advantage_datums(task_rollout_results)"
    )
