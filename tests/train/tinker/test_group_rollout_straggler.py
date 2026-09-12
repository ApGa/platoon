"""Straggler quorum / timeout / minimum group size in the Tinker group workflow.

Rollouts run in-process. Once ``straggler_quorum`` members have settled, the
rest get ``straggler_timeout_seconds`` more before being cancelled; cancelled
members are unobserved outcomes and ``straggler_cancelled`` is emitted. A group
with fewer than ``min_successful_group_size`` usable members yields no datums.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import tinker
import torch
from tinker import TensorData

from platoon.train.tinker.config_defs import WorkflowConfig
from platoon.train.tinker.workflows import group_rollout_workflow as grw
from platoon.utils.rollout_workload import RolloutWorkload
from platoon.utils.stats_tracker import get as get_tracker
from platoon.utils.tinker_data_processing import TrajectoryCollectionResult, TrajectoryStats


def _datum(reward: float) -> tinker.Datum:
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor([2, 3, 4])),
            "logprobs": TensorData.from_torch(torch.tensor([0.0, -0.1, -0.2])),
            "advantages": TensorData.from_torch(torch.tensor([0.0, reward, reward])),
            "mask": TensorData.from_torch(torch.tensor([0.0, 1.0, 1.0])),
        },
    )


def _outcome(reward: float) -> grw._SingleRolloutOutcome:
    result = TrajectoryCollectionResult(
        datums=[_datum(reward)],
        task_reward=reward,
        trajectory_stats=[
            TrajectoryStats(
                trajectory_id="root", reward=reward, num_steps=1, num_input_tokens=3, num_output_tokens=2,
                num_datums=1, is_root=True,
            )
        ],
        root_rewards_dict={},
    )
    workload = RolloutWorkload(
        environment_steps=1, model_calls=1, input_tokens=3, output_tokens=2, trajectories=1,
        postmerge_datums=1, policy_eligible_datums=1, post_sampling_datums=1,
    )
    return grw._SingleRolloutOutcome(result=result, workload=workload, observed=True)


class _FakeWorkflow(grw.GroupRolloutWorkflow):
    """Replaces the rollout with fake fast/slow coroutines."""

    def __init__(self, config: WorkflowConfig, delays: list[float], rewards: list[float], scope: str):
        super().__init__(
            rollout_fn=None,
            get_task_fn=None,
            config=config,
            model_info=SimpleNamespace(model_name="m", base_url="u", api_key="k", llm=None),
            stats_scope=scope,
        )
        self.delays = delays
        self.rewards = rewards
        self.cancelled: list[int] = []

    async def arun_episode_single(self, data: dict, rollout_number: int = 0):
        try:
            await asyncio.sleep(self.delays[rollout_number])
        except asyncio.CancelledError:
            self.cancelled.append(rollout_number)
            raise
        return _outcome(self.rewards[rollout_number])


def _run(workflow: _FakeWorkflow):
    return asyncio.run(workflow.arun_episode({"task_id": "t"}))


def _scalars(scope: str) -> dict[str, float]:
    # Tracker keys are prefixed with the tracker's scope name.
    exported = get_tracker(scope).export(reset=True)
    return {key.removeprefix(f"{scope}/"): value for key, value in exported.items()}


def test_without_straggler_policy_every_member_is_awaited():
    scope = "test_straggler_off"
    config = WorkflowConfig(group_size=4, leave_one_out_baseline=True, filter_zero_advantage_datums=False)
    workflow = _FakeWorkflow(config, delays=[0.0, 0.0, 0.05, 0.1], rewards=[0.0, 1.0, 1.0, 0.0], scope=scope)
    output = _run(workflow)
    assert len(output) == 4
    assert output.observed_rollouts == 4
    assert workflow.cancelled == []
    assert "straggler_cancelled" not in _scalars(scope)


def test_workflow_attaches_raw_rewards_while_centering_training_advantages():
    scope = "test_batch_stats_before_centering"
    workflow = _FakeWorkflow(
        WorkflowConfig(group_size=2, leave_one_out_baseline=True),
        delays=[0.0, 0.0],
        rewards=[0.0, 1.0],
        scope=scope,
    )
    output = _run(workflow)
    assert output.rollout_stats.to_metrics(prefix="batch")["batch/task_reward/avg"] == 0.5
    assert output.rollout_stats.values["task_reward"] == [0.0, 1.0]
    assert output[0].loss_fn_inputs["advantages"].to_torch().tolist() == [0.0, -1.0, -1.0]
    assert output[1].loss_fn_inputs["advantages"].to_torch().tolist() == [0.0, 1.0, 1.0]
    assert _scalars(scope)["rollout_completion/task_reward/avg"] == 0.5


def test_quorum_then_timeout_cancels_stragglers():
    scope = "test_straggler_cancel"
    config = WorkflowConfig(
        group_size=4,
        leave_one_out_baseline=True,
        filter_zero_advantage_datums=False,
        straggler_timeout_seconds=0.1,
        straggler_quorum=2,
        min_successful_group_size=2,
    )
    # Two fast members settle the quorum; one finishes inside the grace,
    # one would take far longer and is cancelled.
    workflow = _FakeWorkflow(config, delays=[0.0, 0.01, 0.05, 30.0], rewards=[0.0, 1.0, 1.0, 1.0], scope=scope)
    output = _run(workflow)
    assert workflow.cancelled == [3]
    assert len(output) == 3
    assert output.requested_rollouts == 4
    assert output.observed_rollouts == 3
    scalars = _scalars(scope)
    assert scalars["straggler_cancelled"] == 1.0


def test_default_quorum_is_group_size_minus_one():
    scope = "test_straggler_default_quorum_timing"
    config = WorkflowConfig(
        group_size=3,
        leave_one_out_baseline=True,
        filter_zero_advantage_datums=False,
        straggler_timeout_seconds=0.05,
    )
    workflow = _FakeWorkflow(config, delays=[0.0, 0.02, 30.0], rewards=[1.0, 0.0, 0.0], scope=scope)

    async def _timed():
        loop = asyncio.get_running_loop()
        start = loop.time()
        output = await workflow.arun_episode({"task_id": "t"})
        return output, loop.time() - start

    output, elapsed = asyncio.run(_timed())
    assert workflow.cancelled == [2]
    assert elapsed < 5.0
    assert output.observed_rollouts == 2
    assert _scalars(scope)["straggler_cancelled"] == 1.0


def test_min_successful_group_size_rejects_small_groups():
    scope = "test_straggler_min_group"
    config = WorkflowConfig(
        group_size=4,
        leave_one_out_baseline=True,
        filter_zero_advantage_datums=False,
        straggler_timeout_seconds=0.05,
        straggler_quorum=2,
        min_successful_group_size=3,
    )
    workflow = _FakeWorkflow(config, delays=[0.0, 0.0, 30.0, 30.0], rewards=[0.0, 1.0, 1.0, 1.0], scope=scope)
    output = _run(workflow)
    assert sorted(workflow.cancelled) == [2, 3]
    assert list(output) == []
    assert output.observed_rollouts == 2
    assert output.trainable_rollouts == 0
    scalars = _scalars(scope)
    assert scalars["straggler_cancelled"] == 2.0
    assert scalars["group_size_rejected"] == 1.0


def test_outer_cancellation_cancels_every_member():
    scope = "test_straggler_outer_cancel"
    config = WorkflowConfig(group_size=3, straggler_timeout_seconds=1.0, straggler_quorum=2)
    workflow = _FakeWorkflow(config, delays=[30.0, 30.0, 30.0], rewards=[0.0, 0.0, 0.0], scope=scope)

    async def _cancel_outer():
        task = asyncio.ensure_future(workflow.arun_episode({"task_id": "t"}))
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_cancel_outer())
    assert sorted(workflow.cancelled) == [0, 1, 2]


def test_workflow_config_validation():
    with pytest.raises(ValueError):
        WorkflowConfig(group_size=4, straggler_quorum=2)  # requires the timeout
    with pytest.raises(ValueError):
        WorkflowConfig(group_size=4, straggler_timeout_seconds=0)
    with pytest.raises(ValueError):
        WorkflowConfig(group_size=4, straggler_timeout_seconds=1.0, straggler_quorum=5)
    with pytest.raises(ValueError):
        WorkflowConfig(group_size=4, min_successful_group_size=5)
    config = WorkflowConfig(
        group_size=8, straggler_timeout_seconds=900, straggler_quorum=6, min_successful_group_size=4
    )
    assert (config.straggler_timeout_seconds, config.straggler_quorum, config.min_successful_group_size) == (900, 6, 4)
