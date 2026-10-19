"""The members of a training group must have trained the same reward key, or the group does not train
(``reward_key_mismatch``); each group counts its rollouts by the key they trained (``reward_key/<key>``).

A reward contract may list the names a reward trains under (``reward: [reward, correctness]``), and each reading
reports exactly one of them.  Which one a task's reading reports is the task's: code of the policy's that the tests
run must not be able to choose it rollout by rollout, and so the scale a group's rewards are centered on.  A member
that trained no key (a score platoon gave itself, an outcome that is not scored) takes no part.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import tinker
import torch
from tinker import TensorData

from platoon.train.tinker.config_defs import WorkflowConfig
from platoon.train.tinker.workers.protocol import PROTOCOL_VERSION, RolloutWorkerResponse
from platoon.train.tinker.workflows import GroupRolloutWorkflow
from platoon.train.tinker.workflows.group_rollout_workflow import _SingleRolloutOutcome
from platoon.utils.rollout_workload import RolloutWorkload
from platoon.utils.stats_tracker import StatsTracker
from platoon.utils.tinker_data_processing import TrajectoryCollectionResult, TrajectoryStats


def _datum(reward: float) -> tinker.Datum:
    """One datum whose action tokens carry ``reward`` until the group centers it."""

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor([2, 3, 4])),
            "logprobs": TensorData.from_torch(torch.tensor([0.0, -0.1, -0.2])),
            "advantages": TensorData.from_torch(torch.tensor([0.0, reward, reward])),
            "mask": TensorData.from_torch(torch.tensor([0.0, 1.0, 1.0])),
        },
    )


def _member(reward: float, key: str | None) -> _SingleRolloutOutcome:
    """A member whose root completed with a scored outcome of ``reward``, trained under ``key``."""

    stats = TrajectoryStats("root", reward, num_steps=1, num_input_tokens=3, num_output_tokens=2, num_datums=1)
    result = TrajectoryCollectionResult(
        datums=[_datum(reward)], task_reward=reward, trajectory_stats=[stats], root_rewards_dict={}
    )
    workload = RolloutWorkload(postmerge_datums=1, policy_eligible_datums=1, post_sampling_datums=1)
    return _SingleRolloutOutcome(
        result=result, workload=workload, observed=True, outcome_status="scored", reward_key=key
    )


def _group(members: list[tuple[float, str | None]]):
    workflow = GroupRolloutWorkflow(
        rollout_fn=None,
        get_task_fn=None,
        config=WorkflowConfig(group_size=len(members), filter_zero_advantage_datums=False),
        model_info=SimpleNamespace(model_name="m", llm=SimpleNamespace(version=0)),
    )

    async def single(data, rollout_number):
        return _member(*members[rollout_number])

    workflow.arun_episode_single = single
    return asyncio.run(workflow.arun_episode({"task_id": "task"}))


def _completion(output) -> dict[str, float]:
    tracker = StatsTracker("train")
    output.group_stats.record(tracker)
    return tracker.export()


def test_a_group_whose_rollouts_trained_different_reward_keys_does_not_train(caplog):
    output = _group([(1.0, "reward"), (0.0, "correctness"), (1.0, "reward")])

    assert list(output) == [] and output.trainable_rollouts == 0  # no datum trains, though advantages would differ
    counts = output.rollout_stats.counts
    assert (counts["reward_key/reward"], counts["reward_key/correctness"], counts["reward_key_mismatch"]) == (2, 1, 1)
    assert output.rollout_stats.values["task_reward"] == [1.0, 0.0, 1.0]  # still measured
    assert _completion(output)["train/rollout_completion/reward_key_mismatch"] == 1
    assert "its rollouts trained different reward keys (correctness, reward), so none of them trains" in caplog.text


def test_a_group_that_trained_one_key_trains_and_platoons_own_zero_takes_no_part():
    output = _group([(1.0, "correctness"), (0.0, None), (0.0, "correctness")])

    assert len(output) == 3 and output.trainable_rollouts == 3  # every member's advantage is nonzero
    counts = output.rollout_stats.counts
    assert counts["reward_key/correctness"] == 2 and counts["reward_key_mismatch"] == 0
    assert not any(name.startswith("reward_key/") and name != "reward_key/correctness" for name in counts)


def test_the_key_travels_with_the_worker_response():
    workflow = GroupRolloutWorkflow(
        rollout_fn=None,
        get_task_fn=None,
        config=WorkflowConfig(group_size=1),
        model_info=SimpleNamespace(model_name="m", llm=SimpleNamespace(version=0)),
    )
    response = RolloutWorkerResponse(
        protocol_version=PROTOCOL_VERSION,
        request_id="r",
        status="conversion_empty",
        result=None,
        workload=RolloutWorkload(),
        observed=True,
        outcome_status="scored",
        reward_key="correctness",
    )

    member = workflow._outcome_from_response(response, "completed")

    assert (member.outcome_status, member.reward_key) == ("scored", "correctness")  # also without a result
