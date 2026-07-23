"""Inference-aligned token-cost attribution for recursive trajectory trees."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Mapping

POLICY_SUBTREE_TOKEN_EFFICIENCY_MISC_KEY = "policy_subtree_token_efficiency"
TOKEN_EFFICIENCY_COLLECTION_MISC_KEY = "token_efficiency_attribution"
TOKEN_EFFICIENCY_PENALTY_REWARD_KEY = "reward/efficiency_penalty"

# Keep this utility dependency-light: the workflow is imported in isolated
# subprocesses and in tests that stub the agent package. This value is the
# public marker defined in platoon.agents.actions.subagent.
_VERIFIER_TASK_MISC_KEY = "subagent_reward_verifier_task"


@dataclass(frozen=True)
class TokenEfficiencyAttributionStats:
    policy_trajectories: int
    verifier_trajectories: int
    attributed_completions: int
    verifier_completions: int
    ambiguous_completions: int
    unattributed_completions: int
    malformed_or_missing_completions: int
    unattributed_input_tokens: int
    unattributed_output_tokens: int


def _task_misc(trajectory: dict) -> dict:
    task = trajectory.get("task")
    if not isinstance(task, dict):
        return {}
    misc = task.get("misc")
    return misc if isinstance(misc, dict) else {}


def _trajectory_misc(trajectory: dict) -> dict:
    misc = trajectory.get("misc")
    if isinstance(misc, dict):
        return misc
    misc = {}
    trajectory["misc"] = misc
    return misc


def _is_marked_verifier(trajectory: dict) -> bool:
    return bool(
        _task_misc(trajectory).get(_VERIFIER_TASK_MISC_KEY) or _trajectory_misc(trajectory).get(_VERIFIER_TASK_MISC_KEY)
    )


def _completion_ids(trajectory: dict) -> set[str]:
    """Return unique exported model requests referenced by one trajectory."""

    completion_ids: set[str] = set()
    for step in trajectory.get("steps", []):
        if not isinstance(step, dict):
            continue
        misc = step.get("misc")
        if not isinstance(misc, dict):
            continue
        action_misc = misc.get("action_misc")
        if not isinstance(action_misc, dict):
            continue
        completion_id = action_misc.get("completion_id")
        if isinstance(completion_id, str) and completion_id:
            completion_ids.add(completion_id)
    return completion_ids


def _policy_and_verifier_ids(
    trajectories: Mapping[str, dict],
    parents: Mapping[str, str | None],
) -> tuple[set[str], set[str]]:
    """Classify verifier branches, including descendants missing the marker."""

    verifier_cache: dict[str, bool] = {}
    visiting: set[str] = set()

    def is_verifier(trajectory_id: str) -> bool:
        cached = verifier_cache.get(trajectory_id)
        if cached is not None:
            return cached
        if trajectory_id in visiting:
            raise ValueError(f"Cycle in trajectory parent links at {trajectory_id}")
        visiting.add(trajectory_id)
        trajectory = trajectories[trajectory_id]
        parent_id = parents.get(trajectory_id)
        result = _is_marked_verifier(trajectory) or bool(parent_id in trajectories and is_verifier(str(parent_id)))
        visiting.remove(trajectory_id)
        verifier_cache[trajectory_id] = result
        return result

    verifier_ids = {trajectory_id for trajectory_id in trajectories if is_verifier(trajectory_id)}
    return set(trajectories) - verifier_ids, verifier_ids


def _penalty(
    effective_tokens: float,
    *,
    coefficient: float,
    reference_tokens: float,
    max_penalty: float,
) -> float:
    if effective_tokens <= 0 or coefficient <= 0 or max_penalty <= 0:
        return 0.0
    normalized_cost = math.log2(1.0 + effective_tokens / reference_tokens)
    return min(max_penalty, coefficient * normalized_cost)


def annotate_policy_subtree_token_efficiency(
    trajectory_collection: dict,
    completion_token_counts: Mapping[str, tuple[int, int]],
    *,
    coefficient: float,
    reference_tokens: float,
    max_penalty: float,
    input_token_weight: float,
    output_token_weight: float,
) -> TokenEfficiencyAttributionStats:
    """Attach a bounded token penalty to each deployable policy trajectory.

    A policy trajectory pays for its own unique model requests and every
    non-verifier descendant. Synthetic reward-verifier branches are excluded
    entirely because they are absent at inference. Overlap between an agent and
    its ancestors is intentional: a child owns its local behavior, while each
    parent owns the decision to launch that subtree.
    """

    numeric_values = {
        "coefficient": coefficient,
        "reference_tokens": reference_tokens,
        "max_penalty": max_penalty,
        "input_token_weight": input_token_weight,
        "output_token_weight": output_token_weight,
    }
    if any(not math.isfinite(float(value)) for value in numeric_values.values()):
        raise ValueError("token-efficiency settings must be finite")
    if coefficient < 0 or max_penalty < 0:
        raise ValueError("token-efficiency coefficient and max_penalty must be non-negative")
    if reference_tokens <= 0:
        raise ValueError("token-efficiency reference_tokens must be positive")
    if input_token_weight < 0 or output_token_weight < 0:
        raise ValueError("token-efficiency token weights must be non-negative")
    if input_token_weight == 0 and output_token_weight == 0:
        raise ValueError("at least one token-efficiency token weight must be positive")

    trajectories = trajectory_collection.get("trajectories")
    if not isinstance(trajectories, dict) or not trajectories:
        stats = TokenEfficiencyAttributionStats(0, 0, 0, 0, 0, len(completion_token_counts), 0, 0, 0)
        trajectory_collection.setdefault("misc", {})[TOKEN_EFFICIENCY_COLLECTION_MISC_KEY] = asdict(stats)
        return stats
    if any(not isinstance(trajectory, dict) for trajectory in trajectories.values()):
        raise TypeError("trajectory collection values must be dictionaries")

    parents: dict[str, str | None] = {}
    for trajectory_id, trajectory in trajectories.items():
        parent_info = trajectory.get("parent_info")
        parent_id = parent_info.get("id") if isinstance(parent_info, dict) else None
        parents[trajectory_id] = parent_id if isinstance(parent_id, str) else None

    policy_ids, verifier_ids = _policy_and_verifier_ids(trajectories, parents)
    completion_owners: dict[str, set[str]] = defaultdict(set)
    for trajectory_id, trajectory in trajectories.items():
        for completion_id in _completion_ids(trajectory):
            completion_owners[completion_id].add(trajectory_id)

    self_input = {trajectory_id: 0 for trajectory_id in policy_ids}
    self_output = {trajectory_id: 0 for trajectory_id in policy_ids}
    attributed_completions = 0
    verifier_completions = 0
    ambiguous_completions = 0
    malformed_or_missing_completions = 0
    for completion_id, owners in completion_owners.items():
        token_counts = completion_token_counts.get(completion_id)
        if token_counts is None:
            malformed_or_missing_completions += 1
            continue
        if len(owners) != 1:
            ambiguous_completions += 1
            continue
        owner = next(iter(owners))
        if owner in verifier_ids:
            verifier_completions += 1
            continue
        input_tokens, output_tokens = token_counts
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError(f"negative token count for completion {completion_id}")
        self_input[owner] += int(input_tokens)
        self_output[owner] += int(output_tokens)
        attributed_completions += 1

    referenced_completion_ids = set(completion_owners)
    unattributed_ids = set(completion_token_counts) - referenced_completion_ids
    unattributed_input_tokens = sum(int(completion_token_counts[key][0]) for key in unattributed_ids)
    unattributed_output_tokens = sum(int(completion_token_counts[key][1]) for key in unattributed_ids)

    policy_children: dict[str, list[str]] = {trajectory_id: [] for trajectory_id in policy_ids}
    for child_id in policy_ids:
        parent_id = parents.get(child_id)
        if parent_id in policy_ids:
            policy_children[str(parent_id)].append(child_id)

    subtree_input: dict[str, int] = {}
    subtree_output: dict[str, int] = {}
    subtree_trajectories: dict[str, int] = {}
    visiting: set[str] = set()

    def aggregate(trajectory_id: str) -> tuple[int, int, int]:
        if trajectory_id in subtree_input:
            return (
                subtree_input[trajectory_id],
                subtree_output[trajectory_id],
                subtree_trajectories[trajectory_id],
            )
        if trajectory_id in visiting:
            raise ValueError(f"Cycle in policy trajectory tree at {trajectory_id}")
        visiting.add(trajectory_id)
        input_tokens = self_input[trajectory_id]
        output_tokens = self_output[trajectory_id]
        trajectory_count = 1
        for child_id in policy_children[trajectory_id]:
            child_input, child_output, child_count = aggregate(child_id)
            input_tokens += child_input
            output_tokens += child_output
            trajectory_count += child_count
        visiting.remove(trajectory_id)
        subtree_input[trajectory_id] = input_tokens
        subtree_output[trajectory_id] = output_tokens
        subtree_trajectories[trajectory_id] = trajectory_count
        return input_tokens, output_tokens, trajectory_count

    for trajectory_id in policy_ids:
        aggregate(trajectory_id)

    for trajectory_id in policy_ids:
        own_effective_tokens = (
            input_token_weight * self_input[trajectory_id] + output_token_weight * self_output[trajectory_id]
        )
        subtree_effective_tokens = (
            input_token_weight * subtree_input[trajectory_id] + output_token_weight * subtree_output[trajectory_id]
        )
        normalized_cost = math.log2(1.0 + subtree_effective_tokens / reference_tokens)
        metadata = {
            "self_input_tokens": self_input[trajectory_id],
            "self_output_tokens": self_output[trajectory_id],
            "self_effective_tokens": own_effective_tokens,
            "subtree_input_tokens": subtree_input[trajectory_id],
            "subtree_output_tokens": subtree_output[trajectory_id],
            "subtree_effective_tokens": subtree_effective_tokens,
            "subtree_policy_trajectories": subtree_trajectories[trajectory_id],
            "normalized_cost": normalized_cost,
            "coefficient": coefficient,
            "reference_tokens": reference_tokens,
            "max_penalty": max_penalty,
            "input_token_weight": input_token_weight,
            "output_token_weight": output_token_weight,
            "penalty": _penalty(
                subtree_effective_tokens,
                coefficient=coefficient,
                reference_tokens=reference_tokens,
                max_penalty=max_penalty,
            ),
        }
        _trajectory_misc(trajectories[trajectory_id])[POLICY_SUBTREE_TOKEN_EFFICIENCY_MISC_KEY] = metadata

    stats = TokenEfficiencyAttributionStats(
        policy_trajectories=len(policy_ids),
        verifier_trajectories=len(verifier_ids),
        attributed_completions=attributed_completions,
        verifier_completions=verifier_completions,
        ambiguous_completions=ambiguous_completions,
        unattributed_completions=len(unattributed_ids),
        malformed_or_missing_completions=malformed_or_missing_completions,
        unattributed_input_tokens=unattributed_input_tokens,
        unattributed_output_tokens=unattributed_output_tokens,
    )
    collection_misc = trajectory_collection.setdefault("misc", {})
    if not isinstance(collection_misc, dict):
        raise TypeError("trajectory collection misc must be a dictionary")
    collection_misc[TOKEN_EFFICIENCY_COLLECTION_MISC_KEY] = asdict(stats)
    return stats


def trajectory_token_efficiency_metrics(trajectory: dict) -> dict[str, float]:
    """Return reward/stat metrics from a precomputed policy-subtree cost."""

    misc = trajectory.get("misc")
    metadata = misc.get(POLICY_SUBTREE_TOKEN_EFFICIENCY_MISC_KEY) if isinstance(misc, dict) else None
    if not isinstance(metadata, dict):
        return {}
    fields = {
        TOKEN_EFFICIENCY_PENALTY_REWARD_KEY: "penalty",
        "efficiency/self_input_tokens": "self_input_tokens",
        "efficiency/self_output_tokens": "self_output_tokens",
        "efficiency/self_effective_tokens": "self_effective_tokens",
        "efficiency/subtree_input_tokens": "subtree_input_tokens",
        "efficiency/subtree_output_tokens": "subtree_output_tokens",
        "efficiency/subtree_effective_tokens": "subtree_effective_tokens",
        "efficiency/subtree_policy_trajectories": "subtree_policy_trajectories",
        "efficiency/normalized_cost": "normalized_cost",
    }
    metrics: dict[str, float] = {}
    for metric_key, metadata_key in fields.items():
        value = metadata.get(metadata_key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            metrics[metric_key] = float(value)
    return metrics
