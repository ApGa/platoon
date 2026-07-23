from __future__ import annotations

import math

import pytest

from platoon.utils.token_efficiency import (
    POLICY_SUBTREE_TOKEN_EFFICIENCY_MISC_KEY,
    TOKEN_EFFICIENCY_PENALTY_REWARD_KEY,
    annotate_policy_subtree_token_efficiency,
    trajectory_token_efficiency_metrics,
)


def _step(completion_id: str) -> dict:
    return {"misc": {"action_misc": {"completion_id": completion_id}}}


def _trajectory(
    *completion_ids: str,
    parent_id: str | None = None,
    task_misc: dict | None = None,
    misc: dict | None = None,
) -> dict:
    trajectory = {
        "steps": [_step(completion_id) for completion_id in completion_ids],
        "task": {"misc": dict(task_misc or {})},
        "misc": dict(misc or {}),
    }
    if parent_id is not None:
        trajectory["parent_info"] = {"id": parent_id}
    return trajectory


def test_policy_subtree_tokens_deduplicate_and_exclude_verifier_branch():
    collection = {
        "trajectories": {
            # Repeated completion IDs arise from parallel OpenHands tool events.
            "root": _trajectory("root-call", "root-call"),
            "child": _trajectory(
                "child-call",
                parent_id="root",
                # Policy-ineligible children still consumed deployable work.
                misc={"exclude_from_policy_training": True},
            ),
            "grandchild": _trajectory("grandchild-call", parent_id="child"),
            "verifier": _trajectory(
                "verifier-call",
                parent_id="child",
                task_misc={"subagent_reward_verifier_task": True},
            ),
            # The inherited branch classification is deliberate even if an
            # individual verifier descendant lacks its own marker.
            "verifier-descendant": _trajectory(
                "verifier-descendant-call",
                parent_id="verifier",
            ),
        }
    }
    token_counts = {
        "root-call": (100, 10),
        "child-call": (200, 20),
        "grandchild-call": (300, 30),
        "verifier-call": (10_000, 1_000),
        "verifier-descendant-call": (20_000, 2_000),
        "unattributed-call": (7, 3),
    }

    stats = annotate_policy_subtree_token_efficiency(
        collection,
        token_counts,
        coefficient=0.05,
        reference_tokens=100.0,
        max_penalty=0.2,
        input_token_weight=0.1,
        output_token_weight=1.0,
    )

    root = collection["trajectories"]["root"]["misc"][POLICY_SUBTREE_TOKEN_EFFICIENCY_MISC_KEY]
    child = collection["trajectories"]["child"]["misc"][POLICY_SUBTREE_TOKEN_EFFICIENCY_MISC_KEY]
    grandchild = collection["trajectories"]["grandchild"]["misc"][POLICY_SUBTREE_TOKEN_EFFICIENCY_MISC_KEY]
    assert (root["self_input_tokens"], root["self_output_tokens"]) == (100, 10)
    assert (root["subtree_input_tokens"], root["subtree_output_tokens"]) == (600, 60)
    assert (child["subtree_input_tokens"], child["subtree_output_tokens"]) == (500, 50)
    assert (grandchild["subtree_input_tokens"], grandchild["subtree_output_tokens"]) == (300, 30)
    assert root["subtree_policy_trajectories"] == 3
    assert child["subtree_policy_trajectories"] == 2
    assert POLICY_SUBTREE_TOKEN_EFFICIENCY_MISC_KEY not in collection["trajectories"]["verifier"]["misc"]
    assert POLICY_SUBTREE_TOKEN_EFFICIENCY_MISC_KEY not in collection["trajectories"]["verifier-descendant"]["misc"]

    expected_root_effective = 0.1 * 600 + 60
    expected_root_penalty = min(0.2, 0.05 * math.log2(1 + expected_root_effective / 100.0))
    assert root["subtree_effective_tokens"] == expected_root_effective
    assert root["penalty"] == pytest.approx(expected_root_penalty)
    assert stats.policy_trajectories == 3
    assert stats.verifier_trajectories == 2
    assert stats.attributed_completions == 3
    assert stats.verifier_completions == 2
    assert stats.unattributed_completions == 1
    assert stats.unattributed_input_tokens == 7
    assert stats.unattributed_output_tokens == 3


def test_cross_trajectory_completion_is_ambiguous_and_not_double_charged():
    collection = {
        "trajectories": {
            "root": _trajectory("shared"),
            "child": _trajectory("shared", parent_id="root"),
        }
    }
    stats = annotate_policy_subtree_token_efficiency(
        collection,
        {"shared": (50, 10)},
        coefficient=0.05,
        reference_tokens=100.0,
        max_penalty=0.2,
        input_token_weight=0.01,
        output_token_weight=1.0,
    )

    assert stats.ambiguous_completions == 1
    for trajectory in collection["trajectories"].values():
        metadata = trajectory["misc"][POLICY_SUBTREE_TOKEN_EFFICIENCY_MISC_KEY]
        assert metadata["self_effective_tokens"] == 0
        assert metadata["penalty"] == 0


def test_efficiency_metrics_are_positive_costs_and_annotation_is_idempotent():
    collection = {"trajectories": {"root": _trajectory("call")}}
    kwargs = dict(
        coefficient=0.05,
        reference_tokens=20_000.0,
        max_penalty=0.2,
        input_token_weight=0.01,
        output_token_weight=1.0,
    )
    annotate_policy_subtree_token_efficiency(collection, {"call": (400_000, 16_000)}, **kwargs)
    first = dict(collection["trajectories"]["root"]["misc"][POLICY_SUBTREE_TOKEN_EFFICIENCY_MISC_KEY])
    annotate_policy_subtree_token_efficiency(collection, {"call": (400_000, 16_000)}, **kwargs)
    second = collection["trajectories"]["root"]["misc"][POLICY_SUBTREE_TOKEN_EFFICIENCY_MISC_KEY]

    assert second == first
    metrics = trajectory_token_efficiency_metrics(collection["trajectories"]["root"])
    assert metrics[TOKEN_EFFICIENCY_PENALTY_REWARD_KEY] == pytest.approx(first["penalty"])
    assert metrics["efficiency/subtree_effective_tokens"] == 20_000.0
    assert first["penalty"] == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"coefficient": -0.1}, "non-negative"),
        ({"reference_tokens": 0.0}, "positive"),
        ({"max_penalty": float("inf")}, "finite"),
        ({"input_token_weight": 0.0, "output_token_weight": 0.0}, "at least one"),
    ],
)
def test_invalid_efficiency_settings_are_rejected(override, message):
    kwargs = dict(
        coefficient=0.05,
        reference_tokens=20_000.0,
        max_penalty=0.2,
        input_token_weight=0.01,
        output_token_weight=1.0,
    )
    kwargs.update(override)
    with pytest.raises(ValueError, match=message):
        annotate_policy_subtree_token_efficiency(
            {"trajectories": {"root": _trajectory("call")}},
            {"call": (1, 1)},
            **kwargs,
        )
