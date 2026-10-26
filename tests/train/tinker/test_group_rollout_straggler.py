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

from platoon.train.config_defs import StragglerPolicy, WorkflowConfig
from platoon.train.workflow import group as group_module
from platoon.train.workflow import output as output_module
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


def _outcome(reward: float) -> output_module._SingleRolloutOutcome:
    result = TrajectoryCollectionResult(
        datums=[_datum(reward)],
        task_reward=reward,
        trajectory_stats=[
            TrajectoryStats(
                trajectory_id="root",
                reward=reward,
                num_steps=1,
                num_input_tokens=3,
                num_output_tokens=2,
                num_datums=1,
                is_root=True,
            )
        ],
        root_rewards_dict={},
    )
    workload = RolloutWorkload(
        environment_steps=1,
        model_calls=1,
        input_tokens=3,
        output_tokens=2,
        trajectories=1,
        postmerge_datums=1,
        policy_eligible_datums=1,
        post_sampling_datums=1,
    )
    return output_module._SingleRolloutOutcome(result=result, workload=workload, observed=True)


class _FakeWorkflow(group_module.GroupRolloutWorkflow):
    """Replaces the rollout with fake fast/slow coroutines."""

    def __init__(
        self,
        config: WorkflowConfig,
        delays: list[float],
        rewards: list[float],
        scope: str,
        label_by: str = "_platoon_environment",
    ):
        super().__init__(
            rollout_fn=None,
            get_task_fn=None,
            config=config,
            model_info=SimpleNamespace(model_name="m", base_url="u", api_key="k", llm=None),
            stats_scope=scope,
            label_by=label_by,
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


def _scalars(scope: str, *outputs) -> dict[str, float]:
    # The trainer records a group's own statistics when it consumes the group; tracker keys are prefixed with the
    # tracker's scope name.
    tracker = get_tracker(scope)
    for output in outputs:
        output.group_stats.record(tracker)
    exported = tracker.export(reset=True)
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
    assert "rollout_completion/task_reward/avg" not in _scalars(scope)  # not when the group completed ...
    assert _scalars(scope, output)["rollout_completion/task_reward/avg"] == 0.5  # ... but when it is consumed


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
    scalars = _scalars(scope)  # straggler kills are recorded as they happen ...
    assert scalars["straggler_cancelled"] == 2.0 and "group_size_rejected" not in scalars
    assert _scalars(scope, output)["group_size_rejected"] == 1.0  # ... the group's rejection with the group


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


@pytest.mark.parametrize("failed_fast_members", [0, 2])
def test_environment_policy_cancels_only_algorithmic_tail_and_does_not_insert_zero_rewards(failed_fast_members):
    # Policies by label: a task's mixture entry by default (train.label_by).
    config = WorkflowConfig(
        group_size=8,
        leave_one_out_baseline=True,
        straggler_policies={"fcs_algorithmic": {"timeout_seconds": 0.02, "quorum": 6}},
    )
    assert isinstance(config.straggler_policies["fcs_algorithmic"], StragglerPolicy)
    rewards = [1.0] * 8

    class WorkflowWithFailures(_FakeWorkflow):
        async def arun_episode_single(self, data, rollout_number=0):
            if rollout_number < failed_fast_members:
                return output_module._SingleRolloutOutcome(
                    result=None, workload=RolloutWorkload(), observed=False, disposition="worker_crashed"
                )
            return await super().arun_episode_single(data, rollout_number)

    algorithmic = WorkflowWithFailures(config, [0.0] * 6 + [0.12, 0.12], rewards, "test_algo_policy")
    cpu = WorkflowWithFailures(config, [0.0] * 6 + [0.12, 0.12], rewards, "test_cpu_policy")

    async def run_both():
        return await asyncio.gather(
            algorithmic.arun_episode({"task_id": "a", "_platoon_environment": "fcs_algorithmic"}),
            cpu.arun_episode({"task_id": "b", "_platoon_environment": "fcs_2_0"}),
        )

    algo_output, cpu_output = asyncio.run(run_both())
    assert sorted(algorithmic.cancelled) == [6, 7]
    assert cpu.cancelled == []
    assert algo_output.requested_rollouts == cpu_output.requested_rollouts == 8
    assert algo_output.observed_rollouts == 6 - failed_fast_members
    assert cpu_output.observed_rollouts == 8 - failed_fast_members
    assert algo_output.rollout_stats.values["task_reward"] == [1.0] * (6 - failed_fast_members)
    # Failed members still settle the quorum. Cancelled members are unobserved,
    # not reward-zero peers: equal successes have zero centered advantage.
    assert algo_output.trainable_rollouts == 0
    assert all(torch.all(d.loss_fn_inputs["advantages"].to_torch() == 0) for d in algo_output)
    assert _scalars("test_algo_policy")["straggler_cancelled"] == 2
    assert "straggler_cancelled" not in _scalars("test_cpu_policy")


def test_explicit_disabled_policy_overrides_global_cutoff_and_unlisted_values_use_global():
    config = WorkflowConfig(
        group_size=2,
        straggler_timeout_seconds=0.02,
        straggler_quorum=1,
        straggler_policies={"slow": StragglerPolicy()},
    )

    async def run_both():
        slow = _FakeWorkflow(config, [0, 0.08], [0, 1], "test_disabled_override", label_by="suite")
        default = _FakeWorkflow(config, [0, 0.08], [0, 1], "test_global_fallback", label_by="suite")
        outputs = await asyncio.gather(
            slow.arun_episode({"task_id": "a", "suite": "slow"}),
            default.arun_episode({"task_id": "b", "suite": "other"}),
        )
        return slow, default, outputs

    slow, default, outputs = asyncio.run(run_both())
    assert slow.cancelled == []
    assert default.cancelled == [1]
    assert [output.observed_rollouts for output in outputs] == [2, 1]
    _scalars("test_disabled_override")
    _scalars("test_global_fallback")


def test_a_label_policy_applies_to_a_columns_values_as_text():
    # A level of 3 is labelled "3" (platoon.train.labels.label_text); a level of 4 has no policy of its own.
    by_label = WorkflowConfig(group_size=2, straggler_policies={"3": StragglerPolicy(timeout_seconds=0.02)})

    async def run_all():
        labelled = _FakeWorkflow(by_label, [0, 0.3], [0, 1], "test_label_text_policy", label_by="level")
        other = _FakeWorkflow(by_label, [0, 0.08], [0, 1], "test_label_text_other", label_by="level")
        await asyncio.gather(
            labelled.arun_episode({"task_id": "a", "level": 3}),
            other.arun_episode({"task_id": "b", "level": 4}),
        )
        return labelled, other

    labelled, other = asyncio.run(run_all())
    assert (labelled.cancelled, other.cancelled) == ([1], [])
    for scope in ("test_label_text_policy", "test_label_text_other"):
        _scalars(scope)


def test_straggler_policies_without_a_column_of_their_own_key_on_the_runs_labels():
    config = WorkflowConfig(group_size=8, straggler_policies={"a": {"timeout_seconds": 1, "quorum": 2}})
    assert config.straggler_policies["a"] == StragglerPolicy(timeout_seconds=1, quorum=2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"straggler_policies": {"": {"timeout_seconds": 1}}},
        {"straggler_policies": {"a": {"quorum": 1}}},
        {"straggler_policies": {"a": {"timeout_seconds": 0}}},
        {"straggler_policies": {"a": {"timeout_seconds": 1, "quorum": 9}}},
    ],
)
def test_per_label_policy_validation(kwargs):
    with pytest.raises(ValueError):
        WorkflowConfig(group_size=8, **kwargs)
