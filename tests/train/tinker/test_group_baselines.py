"""Mean and leave-one-out group baselines, including groups with interrupted roots."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import tinker
import torch
from tinker import TensorData

from platoon.train.tinker.config_defs import WorkflowConfig
from platoon.train.tinker.workflows import GroupRolloutWorkflow
from platoon.train.tinker.workflows.group_rollout_workflow import group_baselines
from platoon.utils.tinker_data_processing import TrajectoryCollectionResult, TrajectoryStats


@pytest.mark.parametrize(
    ("rewards", "valid", "leave_one_out", "expected"),
    [
        ([1.0, 0.0, 1.0, 0.0], [True] * 4, False, [0.5] * 4),
        ([1.0, 0.0, 1.0, 0.0], [True] * 4, True, [1 / 3, 2 / 3, 1 / 3, 2 / 3]),
        # The interrupted root's partial 0.9 never enters a baseline; it is
        # itself centered on the completed-root mean.
        ([1.0, 0.0, 0.9], [True, True, False], False, [0.5, 0.5, 0.5]),
        ([1.0, 0.0, 1.0, 0.9], [True, True, True, False], True, [0.5, 1.0, 0.5, 2 / 3]),
        # A singleton completed root has no leave-one-out peer.
        ([1.0, 0.4, 0.2], [True, False, False], True, [1.0, 1.0, 1.0]),
    ],
)
def test_group_baselines(rewards, valid, leave_one_out, expected):
    assert group_baselines(rewards, valid, leave_one_out=leave_one_out) == pytest.approx(expected)


@pytest.mark.parametrize("leave_one_out", [False, True])
def test_group_baselines_need_a_completed_root(leave_one_out):
    with pytest.raises(ValueError, match="completed root"):
        group_baselines([0.3, 0.6], [False, False], leave_one_out=leave_one_out)


def _datum(reward: float) -> tinker.Datum:
    """One observation token followed by two action tokens carrying ``reward``."""

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            "mask": TensorData.from_torch(torch.tensor([0.0, 1.0, 1.0])),
            "advantages": TensorData.from_torch(torch.tensor([0.0, reward, reward])),
        },
    )


def _rollout(root_reward: float, *child_rewards: float, root_valid: bool = True) -> TrajectoryCollectionResult:
    """A rollout as the converter returns it: an interrupted root keeps stats but no datums."""

    datums = ([_datum(root_reward)] if root_valid else []) + [_datum(reward) for reward in child_rewards]
    stats = [
        TrajectoryStats(
            trajectory_id=f"trajectory-{index}",
            reward=reward,
            num_steps=1,
            num_input_tokens=1,
            num_output_tokens=2,
            num_datums=1,
            is_root=index == 0,
        )
        for index, reward in enumerate((root_reward, *child_rewards))
    ]
    return TrajectoryCollectionResult(
        datums=datums,
        task_reward=root_reward,
        trajectory_stats=stats,
        root_rewards_dict={"reward/success": root_reward},
        num_policy_excluded_datums=0 if root_valid else 1,
        task_reward_valid=root_valid,
    )


def _action_advantages(datum: tinker.Datum) -> float:
    advantages = datum.loss_fn_inputs["advantages"].to_torch()
    assert advantages[0] == 0.0, "the observation token must keep a zero advantage"
    assert advantages[1] == advantages[2]
    return float(advantages[1])


def _run_group(results: list[TrajectoryCollectionResult], workflow_cls=GroupRolloutWorkflow, **config):
    async def rollout(task, rollout_config):
        raise AssertionError("not run")

    workflow = workflow_cls(
        rollout_fn=rollout,
        get_task_fn=lambda task_id: SimpleNamespace(id=task_id, max_steps=None),
        config=WorkflowConfig(group_size=len(results), **config),
        model_info=SimpleNamespace(model_name="m", base_url=None, api_key=None, llm=SimpleNamespace(version=0)),
    )

    async def single(data, rollout_number):
        return results[rollout_number]

    workflow.arun_episode_single = single
    return asyncio.run(workflow.arun_episode({"task_id": "task"}))


def test_mean_baseline_ignores_interrupted_roots_and_still_trains_their_completed_children():
    completed_win = _rollout(1.0, 1.0)
    completed_loss = _rollout(0.0)
    interrupted = _rollout(0.9, 1.0, root_valid=False)

    output = _run_group([completed_win, completed_loss, interrupted])

    # Completed-root mean is 0.5; the interrupted root's 0.9 is metric-only.
    assert [_action_advantages(datum) for datum in output] == pytest.approx([0.5, 0.5, -0.5, 0.5])
    assert output.trainable_rollouts == 3


def test_leave_one_out_baseline_centers_each_rollout_on_its_completed_peers():
    first = _rollout(1.0, 0.0)
    second = _rollout(0.0)
    third = _rollout(1.0)
    interrupted = _rollout(0.9, 1.0, root_valid=False)

    output = _run_group([first, second, third, interrupted], leave_one_out_baseline=True)

    # Peers of `first` are (0, 1) -> 0.5, of `second` (1, 1) -> 1, of `third`
    # (1, 0) -> 0.5. The interrupted rollout's completed child uses the mean, 2/3.
    first_root, first_child, second_root, third_root, interrupted_child = map(_action_advantages, output)
    assert (first_root, first_child) == pytest.approx((0.5, -0.5))
    assert second_root == pytest.approx(-1.0)
    assert third_root == pytest.approx(0.5)
    assert interrupted_child == pytest.approx(1 / 3)


def test_leave_one_out_singleton_completed_root_has_zero_advantage():
    completed = _rollout(1.0, 0.0)
    interrupted = _rollout(0.4, 1.0, root_valid=False)

    output = _run_group([completed, interrupted], leave_one_out_baseline=True)

    root, child, interrupted_child = map(_action_advantages, output)
    assert (root, child, interrupted_child) == pytest.approx((0.0, -1.0, 0.0))


def test_group_without_a_completed_root_does_not_train():
    output = _run_group([_rollout(0.5, 1.0, root_valid=False), _rollout(0.2, 1.0, root_valid=False)])

    assert output == []
    assert output.trainable_rollouts == 0


def test_subclass_can_swap_the_baseline_estimator():
    class ZeroBaselineWorkflow(GroupRolloutWorkflow):
        def compute_baselines(self, results):
            return [0.0] * len(results)

    output = _run_group([_rollout(1.0, 0.25), _rollout(0.0)], workflow_cls=ZeroBaselineWorkflow)

    assert [_action_advantages(datum) for datum in output] == [1.0, 0.25, 0.0]


@pytest.mark.parametrize("leave_one_out", [False, True])
def test_baselines_are_logged_and_only_leave_one_out_spreads_them(leave_one_out):
    from platoon.utils.stats_tracker import get as get_tracker

    get_tracker("train").export(reset=True)
    # Completed rewards 1, 0, 1 and an interrupted 0.9: mean baselines 2/3 each; leave-one-out 1/2, 1, 1/2.
    output = _run_group(
        [_rollout(1.0), _rollout(0.0), _rollout(1.0), _rollout(0.9, root_valid=False)],
        leave_one_out_baseline=leave_one_out,
    )
    # The baselines travel with the group, for the step that consumes it (not the one running when it completed).
    assert not any(key.startswith("train/credit/") for key in get_tracker("train").export(reset=True))
    output.group_stats.record(get_tracker("train"))
    stats = get_tracker("train").export(reset=True)

    if leave_one_out:
        assert stats["train/credit/baseline/avg"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)
        assert stats["train/credit/baseline/spread"] == pytest.approx(0.5)
        assert stats["train/credit/root_advantage/abs_avg"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    else:
        assert stats["train/credit/baseline/avg"] == pytest.approx(2 / 3)
        assert stats["train/credit/baseline/spread"] == 0.0
        assert stats["train/credit/root_advantage/abs_avg"] == pytest.approx((1 / 3 + 2 / 3 + 1 / 3) / 3)
