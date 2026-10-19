"""Recursive reward helpers: root-success propagation and the direct-child delegation bonus."""

from __future__ import annotations

import pytest

from platoon.config_defs import RolloutConfig
from platoon.trajectory import ParentInfo, Trajectory, TrajectoryCollection, TrajectoryStep
from platoon.utils.subagent_rewards import (
    SUBAGENT_DELEGATION_REWARD_MISC_KEY,
    add_direct_subagent_delegation_rewards,
    propagate_root_success,
    propogate_root_success,
)
from platoon.utils.trajectory_status import TRAJECTORY_FORCED_ZERO_REWARD_MISC_KEY


def _finish(trajectory: Trajectory, success: float, **reward_misc: float) -> None:
    trajectory.steps.append(TrajectoryStep(misc={"reward_misc": {"reward/success": success, **reward_misc}}))
    trajectory.reward = success


def _add(collection: TrajectoryCollection, parent: Trajectory | None = None) -> Trajectory:
    trajectory = Trajectory(id=f"t{len(collection.trajectories)}")
    if parent is not None:
        trajectory.parent_info = ParentInfo(id=parent.id, fork_step=len(parent.steps))
    collection.trajectories[trajectory.id] = trajectory
    return trajectory


def _tree(root_success: float) -> tuple[TrajectoryCollection, dict[str, Trajectory]]:
    """root -> (child -> grandchild, failed_child) as a recursive rollout leaves it."""

    collection = TrajectoryCollection()
    root = _add(collection)
    child = _add(collection, root)
    grandchild = _add(collection, child)
    failed_child = _add(collection, root)
    root.steps.append(TrajectoryStep(misc={"reward_misc": {"reward/success": 0.0, "reward/subagent_launched": 2.0}}))
    _finish(root, root_success)
    _finish(child, 0.0, **{"reward/subagent_launched": 1.0, "reward/subagent_succeeded": 1.0})
    _finish(grandchild, 1.0)
    _finish(failed_child, 0.0)
    failed_child.misc[TRAJECTORY_FORCED_ZERO_REWARD_MISC_KEY] = True
    return collection, {"root": root, "child": child, "grandchild": grandchild, "failed_child": failed_child}


def _rewards(collection: dict | TrajectoryCollection, names: dict[str, Trajectory]) -> dict[str, tuple]:
    """Each trajectory's (reward, last-step success) by name, for objects and serialized dicts alike."""

    trajectories = collection["trajectories"] if isinstance(collection, dict) else collection.trajectories
    result = {}
    for name, original in names.items():
        trajectory = trajectories[original.id]
        if isinstance(trajectory, dict):
            result[name] = (trajectory["reward"], trajectory["steps"][-1]["misc"]["reward_misc"]["reward/success"])
        else:
            result[name] = (trajectory.reward, trajectory.steps[-1].misc["reward_misc"]["reward/success"])
    return result


@pytest.mark.parametrize("serialized", [False, True])
def test_root_success_relabels_every_descendant_except_forced_zeros(serialized):
    collection, names = _tree(root_success=1.0)
    data = collection.to_dict() if serialized else collection

    assert propagate_root_success(data) is data

    assert _rewards(data, names) == {
        "root": (1.0, 1.0),
        "child": (1.0, 1.0),
        "grandchild": (1.0, 1.0),
        # An explicit terminal failure is never relabeled with the root's success.
        "failed_child": (0.0, 0.0),
    }


@pytest.mark.parametrize("serialized", [False, True])
def test_failed_root_zeroes_successful_children(serialized):
    collection, names = _tree(root_success=0.0)
    data = collection.to_dict() if serialized else collection

    propagate_root_success(data)

    assert {name: reward for name, (reward, _) in _rewards(data, names).items()} == dict.fromkeys(names, 0.0)


def test_root_success_rescales_delegation_counters_on_every_launching_step():
    collection, names = _tree(root_success=1.0)

    propagate_root_success(collection)

    launch_step, _ = names["root"].steps
    assert launch_step.misc["reward_misc"]["reward/subagent_succeeded"] == 2.0
    assert names["child"].steps[-1].misc["reward_misc"]["reward/subagent_succeeded"] == 1.0


def test_root_success_uses_base_success_not_shaped_root_reward():
    collection, names = _tree(root_success=1.0)
    names["root"].reward = 1.25  # e.g. success plus a delegation bonus

    propagate_root_success(collection)

    assert names["child"].reward == 1.0


def test_propagating_an_empty_collection_is_a_no_op():
    assert propagate_root_success({"trajectories": {}}) == {"trajectories": {}}


def test_legacy_propagation_name_is_the_same_function():
    assert propogate_root_success is propagate_root_success


def test_delegation_bonus_counts_direct_trainable_children_only():
    collection, names = _tree(root_success=1.0)

    add_direct_subagent_delegation_rewards(collection, coefficient=0.5)

    root_bonus = names["root"].misc[SUBAGENT_DELEGATION_REWARD_MISC_KEY]
    # Direct children: child (base success 0) and failed_child (forced 0); the
    # grandchild's success counts for child, not for the root.
    assert (root_bonus["launched"], root_bonus["succeeded"], root_bonus["bonus"]) == (2.0, 0.0, 0.0)
    child_bonus = names["child"].misc[SUBAGENT_DELEGATION_REWARD_MISC_KEY]
    assert (child_bonus["launched"], child_bonus["succeeded"], child_bonus["bonus"]) == (1.0, 1.0, 0.5)


def test_delegation_bonus_rejects_negative_coefficients():
    collection, _ = _tree(root_success=1.0)

    with pytest.raises(ValueError, match="non-negative"):
        add_direct_subagent_delegation_rewards(collection, coefficient=-0.1)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, False),
        ({"propagate_root_success": True}, True),
        ({"propogate_root_success": True}, True),
        ({"propagate_root_success": False, "propogate_root_success": False}, False),
    ],
)
def test_rollout_config_resolves_root_propagation(kwargs, expected):
    assert RolloutConfig(**kwargs).propagate_root_success is expected


def test_rollout_config_rejects_conflicting_root_propagation_spellings():
    with pytest.raises(ValueError, match="Conflicting"):
        RolloutConfig(propagate_root_success=True, propogate_root_success=False)
