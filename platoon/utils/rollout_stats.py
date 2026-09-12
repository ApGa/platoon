"""Rollout statistics that travel with groups until a trainer consumes them."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from platoon.utils.stats_tracker import StatsTracker


@dataclass
class RolloutStats:
    """Small CPU-only observations, independent of tokens and training datums.

    Observations can be combined across groups and data-parallel ranks before
    finalizing a batch. Each metric has its own denominator: roots,
    trajectories, and optional reward components need not have the same count.
    """

    groups: int = 0
    values: dict[str, list[float]] = field(default_factory=dict)

    def add(self, name: str, values: Any) -> None:
        tensor = torch.as_tensor(values).detach().float().reshape(-1).cpu()
        if tensor.numel():
            self.values.setdefault(name, []).extend(tensor.tolist())

    def merge(self, other: RolloutStats) -> None:
        self.groups += other.groups
        for name, values in other.values.items():
            self.values.setdefault(name, []).extend(values)

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
