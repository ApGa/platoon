"""Rollout statistics that travel with groups until a trainer consumes them."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from platoon.utils.stats_tracker import StatsTracker

# Per-trajectory tool-call parse counts (``TrajectoryStats``): parsed, parsed by a lenient rule, and not parsed.
TOOL_CALL_STATS = ("num_tool_calls", "num_recovered_tool_calls", "num_unparsed_tool_calls")
# What ``outcome/<status>`` counts each rollout of a group by: the status of the outcome its root recorded
# (``platoon.envs.service.Outcome.status``), and ``error`` for a rollout whose environment could not open its session
# (``EnvSetupError``), which records none; else ``none``: its rollout raised, its worker died, it was cancelled as a
# straggler, or a custom rollout recorded no outcome.
OUTCOME_STATUSES = ("scored", "invalid", "error", "timeout", "cancelled", "none")


# What ``reward_key/<key>`` counts each rollout by: the reward key its root's scored outcome trained, the name its
# grader reported the reward under (``platoon.utils.trajectory_status.trained_reward_key``); a rollout that trained
# none (an outcome that is not scored, a score platoon gave itself) counts under no key.
REWARD_KEY_PREFIX = "reward_key/"
# The groups whose rollouts trained different reward keys, which do not train (one count per group).
REWARD_KEY_MISMATCH = "reward_key_mismatch"


def reward_key_counts(keys: Iterable[str | None]) -> dict[str, int]:
    """``reward_key/<key>`` counts of rollouts that trained these reward keys (None: none), a ``/`` in a key written
    ``--``."""

    counts: dict[str, int] = {}
    for key in keys:
        if key:
            name = REWARD_KEY_PREFIX + key.replace("/", "--")
            counts[name] = counts.get(name, 0) + 1
    return counts


def outcome_counts(statuses: Iterable[str | None]) -> dict[str, int]:
    """``outcome/<status>`` counts of rollouts with these statuses (None: ``none``), every status of
    :data:`OUTCOME_STATUSES` included, at zero when no rollout had it."""

    counts = {f"outcome/{status}": 0 for status in OUTCOME_STATUSES}
    for status in statuses:
        key = f"outcome/{status or 'none'}"
        counts[key] = counts.get(key, 0) + 1
    return counts


@dataclass
class RolloutStats:
    """Small CPU-only observations, independent of tokens and training datums.

    Observations can be combined across groups and data-parallel ranks before
    finalizing a batch. Each metric has its own denominator: roots,
    trajectories, and optional reward components need not have the same count.
    ``counts`` are totals, such as the rollouts of each outcome status
    (:func:`outcome_counts`), which add up across groups.
    """

    groups: int = 0
    values: dict[str, list[float]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, values: Any) -> None:
        tensor = torch.as_tensor(values).detach().float().reshape(-1).cpu()
        if tensor.numel():
            self.values.setdefault(name, []).extend(tensor.tolist())

    def merge(self, other: RolloutStats) -> None:
        self.groups += other.groups
        for name, values in other.values.items():
            self.values.setdefault(name, []).extend(values)
        for name, count in other.counts.items():
            self.counts[name] = self.counts.get(name, 0) + count

    def subset(self, names: Iterable[str]) -> RolloutStats:
        """A copy with only the ``values`` named, and every count."""

        wanted = set(names)
        return RolloutStats(
            groups=self.groups,
            values={name: list(values) for name, values in self.values.items() if name in wanted},
            counts=dict(self.counts),
        )

    def record(self, tracker: Any, *, prefix: str) -> None:
        """Record once per batch, or in an explicitly named completion scope."""
        # A nonempty mask gives even an empty batch an explicit zero count.
        group_mask = torch.full((max(1, self.groups),), self.groups > 0, dtype=torch.bool)
        tracker.denominator(**{f"{prefix}/groups": group_mask})
        for name, values in self.values.items():
            key = f"{prefix}/{name}"
            denominator = f"{key}/count"
            tracker.denominator(**{denominator: torch.ones(len(values), dtype=torch.bool)})
            tracker.stat(**{key: torch.tensor(values, dtype=torch.float32)}, denominator=denominator)
        for name, count in self.counts.items():
            # Summed like the group count: an explicit zero stays a zero.
            tracker.denominator(**{f"{prefix}/{name}": torch.full((max(1, count),), count > 0, dtype=torch.bool)})

    def to_metrics(self, *, prefix: str) -> dict[str, float]:
        """Finalize one batch before passing it to a scalar-only exporter."""
        tracker = StatsTracker()
        self.record(tracker, prefix=prefix)
        return tracker.export()


def summarize_rollout_group(data: Mapping[str, Any], *, presence_mask_prefix: str = "") -> RolloutStats:
    """Read raw root/trajectory metrics from one group, before datum selection.

    Missing metrics stay missing. In particular, an absent rollout is never
    synthesized as a zero reward, and optional components exclude filled zeros.
    """
    stats = RolloutStats(groups=1)

    def present(name: str) -> torch.Tensor:
        values = torch.as_tensor(data[name]).detach().float().reshape(-1).cpu()
        mask = data.get(f"{presence_mask_prefix}{name}") if presence_mask_prefix else None
        if mask is not None:
            mask = torch.as_tensor(mask).detach().bool().reshape(-1).cpu()
            if mask.shape != values.shape:
                raise ValueError(f"Reward metric presence mask does not match {name}")
            values = values[mask]
        return values

    for name in ("task_reward", "task_reward_valid", "num_steps", "num_input_tokens", "num_output_tokens"):
        if name in data:
            stats.add(name, present(name))
    for name in TOOL_CALL_STATS:
        if name in data:
            stats.add(name, present(name))

    for name in data:
        if name.startswith(("root_", "reward/")):
            stats.add(name, present(name))

    for name in ("task_reward", *(key for key in data if key.startswith("root_"))):
        if name not in data:
            continue
        values = present(name)
        if values.numel():
            stats.add(f"{name}_at_k_mean", values.mean())
            stats.add(f"{name}_at_k_min", values.min())
            stats.add(f"{name}_at_k_max", values.max())

    if "num_steps" in data:
        steps = present("num_steps").clamp(min=1)
        for name in ("num_input_tokens", "num_output_tokens"):
            if name in data:
                stats.add(f"avg_{name.removeprefix('num_')}_per_step", present(name) / steps)
    return stats
