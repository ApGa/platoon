"""``min_successful_group_size`` counts trainable members only.

A member is trainable when its root completed with a scored outcome (``task_reward_valid``). An untrainable member
that recorded model calls (an ``error``, ``invalid``, ``timeout`` or ``cancelled`` root) counts no more than one that
returned nothing (a setup failure, a cancelled straggler, a worker that died): a group trains when at least the
minimum of its members are trainable, and the untrainable ones stay out of its baseline.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import tinker
import torch
from tinker import TensorData

from platoon.envs.service import Outcome
from platoon.gateway.core import AgentSession
from platoon.gateway.trace import build_trajectory_collection
from platoon.task import Task
from platoon.train.config_defs import WorkflowConfig
from platoon.train.workflow import group as group_module
from platoon.train.workflow import output as output_module
from platoon.utils.rollout_workload import RolloutWorkload
from platoon.utils.stats_tracker import get as get_tracker
from platoon.utils.tinker_data_processing import (
    TrajectoryCollectionResult,
    TrajectoryStats,
    get_train_data_for_trajectory_collection,
)

# How each member ends: "scored" (a completed root), "untrainable" (a root that recorded calls but ended
# untrainable), "crashed" (a worker that died: no result), "slow" (cancelled as a straggler).
COMPLETED, UNTRAINABLE, CRASHED, SLOW = "scored", "untrainable", "crashed", "slow"


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


def _member(kind: str, reward: float) -> output_module._SingleRolloutOutcome:
    if kind == CRASHED:
        return output_module._SingleRolloutOutcome(
            result=None, workload=RolloutWorkload(), observed=False, disposition="worker_crashed"
        )
    completed = kind == COMPLETED
    result = TrajectoryCollectionResult(
        # An untrainable root keeps no policy datums (the converter drops interrupted trajectories' tokens).
        datums=[_datum(reward)] if completed else [],
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
        num_policy_excluded_datums=0 if completed else 1,
        task_reward_valid=completed,
    )
    workload = RolloutWorkload(
        environment_steps=1,
        model_calls=1,
        input_tokens=3,
        output_tokens=2,
        trajectories=1,
        postmerge_datums=1,
        policy_eligible_datums=1 if completed else 0,
        post_sampling_datums=1 if completed else 0,
    )
    return output_module._SingleRolloutOutcome(result=result, workload=workload, observed=True)


class _Workflow(group_module.GroupRolloutWorkflow):
    def __init__(self, config: WorkflowConfig, kinds: list[str], rewards: list[float], scope: str) -> None:
        super().__init__(
            rollout_fn=None,
            get_task_fn=None,
            config=config,
            model_info=SimpleNamespace(model_name="m", base_url="u", api_key="k", llm=None),
            stats_scope=scope,
        )
        self.kinds = kinds
        self.rewards = rewards

    async def arun_episode_single(self, data: dict, rollout_number: int = 0):
        kind = self.kinds[rollout_number]
        if kind == SLOW:
            await asyncio.sleep(30)
        return _member(kind, self.rewards[rollout_number])


def _run(config: WorkflowConfig, kinds: list[str], rewards: list[float] | None = None, *, scope: str):
    workflow = _Workflow(config, kinds, rewards or [float(index % 2) for index in range(len(kinds))], scope)
    output = asyncio.run(workflow.arun_episode({"task_id": "t"}))
    tracker = get_tracker(scope)
    output.group_stats.record(tracker)
    scalars = {key.removeprefix(f"{scope}/"): value for key, value in tracker.export(reset=True).items()}
    return output, scalars


def _config(group_size: int, min_successful: int, **extra) -> WorkflowConfig:
    return WorkflowConfig(
        group_size=group_size,
        min_successful_group_size=min_successful,
        leave_one_out_baseline=True,
        filter_zero_advantage_datums=False,
        **extra,
    )


@pytest.mark.parametrize(
    "kinds",
    [
        [COMPLETED, COMPLETED, COMPLETED, UNTRAINABLE],  # an error/invalid/timeout root that recorded model calls
        [COMPLETED, COMPLETED, COMPLETED, CRASHED],  # a worker that died (hard timeout, out of memory, crash)
    ],
    ids=["untrainable-member", "dead-worker"],
)
def test_an_untrainable_or_missing_member_does_not_count_toward_the_minimum(kinds):
    output, scalars = _run(_config(4, 4), kinds, scope=f"test_trainable_drop_{kinds[-1]}")

    assert list(output) == [] and output.trainable_rollouts == 0
    assert output.requested_rollouts == 4
    assert scalars["group_size_rejected"] == 1.0


def test_a_cancelled_straggler_does_not_count_toward_the_minimum():
    config = _config(4, 4, straggler_timeout_seconds=0.05, straggler_quorum=3)
    output, scalars = _run(config, [COMPLETED, COMPLETED, COMPLETED, SLOW], scope="test_trainable_drop_straggler")

    assert list(output) == []
    assert output.observed_rollouts == 3
    assert scalars["group_size_rejected"] == 1.0


def test_a_group_whose_every_member_completed_trains():
    output, scalars = _run(_config(4, 4), [COMPLETED] * 4, scope="test_trainable_full")

    assert len(output) == 4
    assert "group_size_rejected" not in scalars


def test_a_partial_minimum_counts_completed_roots_not_returned_results():
    # Three of four members returned a result, but only two completed: below a minimum of three.
    kinds = [COMPLETED, COMPLETED, UNTRAINABLE, CRASHED]
    output, scalars = _run(_config(4, 3), kinds, scope="test_trainable_partial_reject")
    assert list(output) == [] and scalars["group_size_rejected"] == 1.0

    # A minimum of two is met by the two completed roots, which alone set the baseline.
    output, scalars = _run(_config(4, 2), kinds, rewards=[0.0, 1.0, 1.0, 0.0], scope="test_trainable_partial_keep")
    assert "group_size_rejected" not in scalars
    assert [datum.loss_fn_inputs["advantages"].to_torch().tolist() for datum in output] == [
        [0.0, -1.0, -1.0],
        [0.0, 1.0, 1.0],
    ]


def test_half_of_a_group_of_eight_trains_without_the_untrainable_members():
    """``group_size: 8`` with ``min_successful_group_size: 4``: four trainable members keep the group, and the
    baseline rests on them alone; three do not, however many other members returned a result."""

    kinds = [COMPLETED] * 4 + [UNTRAINABLE, UNTRAINABLE, CRASHED, SLOW]
    config = _config(8, 4, straggler_timeout_seconds=0.05, straggler_quorum=7)
    output, scalars = _run(config, kinds, rewards=[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0], scope="test_half_keep")
    assert "group_size_rejected" not in scalars
    assert [datum.loss_fn_inputs["advantages"].to_torch().tolist()[1] for datum in output] == [
        pytest.approx(2 / 3),
        pytest.approx(2 / 3),
        pytest.approx(-2 / 3),
        pytest.approx(-2 / 3),
    ]

    kinds = [COMPLETED] * 3 + [UNTRAINABLE] * 4 + [CRASHED]  # seven returned a result, three are trainable
    output, scalars = _run(_config(8, 4), kinds, scope="test_half_reject")
    assert list(output) == [] and scalars["group_size_rejected"] == 1.0


def test_the_default_minimum_trains_on_one_completed_root_and_never_counts_a_group_without_one():
    kinds = [COMPLETED, UNTRAINABLE, UNTRAINABLE, CRASHED]
    output, scalars = _run(_config(4, 1), kinds, scope="test_trainable_default_one")
    assert len(output) == 1 and "group_size_rejected" not in scalars

    output, scalars = _run(_config(4, 1), [UNTRAINABLE, UNTRAINABLE, CRASHED, CRASHED], scope="test_trainable_none")
    assert list(output) == [] and "group_size_rejected" not in scalars  # no completed root: as before


@pytest.mark.parametrize("status", ["error", "invalid", "timeout", "cancelled"])
def test_a_gateway_rollout_with_an_untrainable_outcome_is_not_a_trainable_member(status):
    """The member kind above as a gateway rollout builds it: calls recorded, the outcome untrainable."""

    from platoon.gateway.core import ChatResponse

    call = ChatResponse(
        call_id="call-1",
        model="m",
        prompt=tinker.ModelInput.from_ints([1, 2]),
        tokens=[3, 4],
        logprobs=[-0.1, -0.2],
        stop_reason="stop",
        parse_success=True,
        role="assistant",
        content="hi",
        reasoning_content=None,
        tool_calls=None,
        parsed_tool_calls=0,
        routing_id=None,
    )
    agent = AgentSession(agent_id="root", rollout_id="r", trajectory_id="traj-root", secret="s", interactions={})
    agent.calls.append(call)
    collection = build_trajectory_collection(
        [agent], task=Task(id="t", goal="g"), outcome=Outcome.infra("harness_crash", status=status)
    )
    result = get_train_data_for_trajectory_collection(
        trajectory_collection=collection.to_dict(),
        interactions={"call-1": call.interaction()},
        task_id="t",
        checkpoint_version=0,
    )

    assert result.task_reward_valid is False
    assert [stats.num_steps for stats in result.trajectory_stats] == [1]  # it recorded its call
