"""Backend-neutral accounting for recursive rollout inference work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class RolloutWorkload:
    """Known work for one recursive rollout tree.

    Token counts are logical, unpadded request/output tokens. Input tokens count
    the full prompt sent for every recorded model call; they are not an estimate
    of cache-adjusted prefill FLOPs.
    """

    environment_steps: int = 0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    trajectories: int = 0
    postmerge_datums: int = 0
    policy_eligible_datums: int = 0
    post_sampling_datums: int = 0

    def __post_init__(self) -> None:
        for name in (
            "environment_steps",
            "model_calls",
            "input_tokens",
            "output_tokens",
            "trajectories",
            "postmerge_datums",
            "policy_eligible_datums",
            "post_sampling_datums",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not self.post_sampling_datums <= self.policy_eligible_datums <= self.postmerge_datums:
            raise ValueError(
                "datum counts must satisfy post_sampling_datums <= "
                "policy_eligible_datums <= postmerge_datums"
            )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def policy_excluded_datums(self) -> int:
        return self.postmerge_datums - self.policy_eligible_datums

    @property
    def sampling_dropped_datums(self) -> int:
        return self.policy_eligible_datums - self.post_sampling_datums

    @property
    def candidate_trainable_datums(self) -> int:
        return self.post_sampling_datums

    @property
    def candidate_non_trainable_datums(self) -> int:
        return self.postmerge_datums - self.post_sampling_datums

    def __add__(self, other: "RolloutWorkload") -> "RolloutWorkload":
        if not isinstance(other, RolloutWorkload):
            return NotImplemented
        return RolloutWorkload(
            environment_steps=self.environment_steps + other.environment_steps,
            model_calls=self.model_calls + other.model_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            trajectories=self.trajectories + other.trajectories,
            postmerge_datums=self.postmerge_datums + other.postmerge_datums,
            policy_eligible_datums=self.policy_eligible_datums + other.policy_eligible_datums,
            post_sampling_datums=self.post_sampling_datums + other.post_sampling_datums,
        )

    def to_metrics(self, prefix: str) -> dict[str, float]:
        prefix = prefix.rstrip("/")
        return {
            f"{prefix}/total_environment_steps": float(self.environment_steps),
            f"{prefix}/total_model_calls": float(self.model_calls),
            f"{prefix}/total_input_tokens": float(self.input_tokens),
            f"{prefix}/total_output_tokens": float(self.output_tokens),
            f"{prefix}/total_tokens": float(self.total_tokens),
            f"{prefix}/total_trajectories": float(self.trajectories),
            f"{prefix}/total_postmerge_datums": float(self.postmerge_datums),
            f"{prefix}/total_policy_eligible_datums": float(self.policy_eligible_datums),
            f"{prefix}/total_post_sampling_datums": float(self.post_sampling_datums),
            f"{prefix}/total_policy_excluded_datums": float(self.policy_excluded_datums),
            f"{prefix}/total_sampling_dropped_datums": float(self.sampling_dropped_datums),
            f"{prefix}/total_candidate_trainable_datums": float(self.candidate_trainable_datums),
            f"{prefix}/total_candidate_non_trainable_datums": float(self.candidate_non_trainable_datums),
        }


def sum_rollout_workloads(workloads: Iterable[RolloutWorkload]) -> RolloutWorkload:
    total = RolloutWorkload()
    for workload in workloads:
        total += workload
    return total


def trajectory_collection_shape(trajectory_collection: dict[str, Any] | None) -> tuple[int, int]:
    """Return ``(trajectory_count, environment_step_count)`` for a raw tree."""

    if not isinstance(trajectory_collection, dict):
        return 0, 0
    trajectories = trajectory_collection.get("trajectories")
    if not isinstance(trajectories, dict):
        return 0, 0

    environment_steps = 0
    for trajectory in trajectories.values():
        if not isinstance(trajectory, dict):
            continue
        steps = trajectory.get("steps")
        if isinstance(steps, list):
            environment_steps += len(steps)
    return len(trajectories), environment_steps


def record_workload_distribution(
    tracker: Any,
    *,
    prefix: str,
    workloads: Iterable[RolloutWorkload],
) -> None:
    """Record AVG/MIN/MAX totals for a set of rollout or task units."""

    values = list(workloads)
    if not values:
        return

    prefix = prefix.rstrip("/")
    denominator = f"{prefix}/count"
    tracker.denominator(**{denominator: torch.ones(len(values), dtype=torch.bool)})
    fields = {
        "total_environment_steps": [value.environment_steps for value in values],
        "total_model_calls": [value.model_calls for value in values],
        "total_input_tokens": [value.input_tokens for value in values],
        "total_output_tokens": [value.output_tokens for value in values],
        "total_tokens": [value.total_tokens for value in values],
        "total_trajectories": [value.trajectories for value in values],
        "total_postmerge_datums": [value.postmerge_datums for value in values],
        "total_policy_eligible_datums": [value.policy_eligible_datums for value in values],
        "total_post_sampling_datums": [value.post_sampling_datums for value in values],
        "total_policy_excluded_datums": [value.policy_excluded_datums for value in values],
        "total_sampling_dropped_datums": [value.sampling_dropped_datums for value in values],
        "total_candidate_trainable_datums": [value.candidate_trainable_datums for value in values],
        "total_candidate_non_trainable_datums": [value.candidate_non_trainable_datums for value in values],
    }
    tracker.stat(
        denominator=denominator,
        **{
            f"{prefix}/{name}": torch.tensor(field_values, dtype=torch.float32)
            for name, field_values in fields.items()
        },
    )
