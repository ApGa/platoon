"""Stats tracking utilities for training metrics.

A simplified stats tracker inspired by areal's DistributedStatsTracker:
https://github.com/inclusionAI/AReaL/blob/main/areal/utils/stats_tracker.py

This implementation is single-process and does not require torch.distributed.
"""

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from threading import Lock
from typing import Any

import torch


class ReduceType(Enum):
    """Reduction types for aggregating statistics."""

    AVG_MIN_MAX = auto()  # Report average, min, and max
    AVG = auto()  # Report only average
    SUM = auto()  # Report sum
    MIN = auto()  # Report minimum
    MAX = auto()  # Report maximum
    SCALAR = auto()  # Single scalar value (averaged across calls)


@dataclass
class StatEntry:
    """A single statistic entry with value and optional denominator."""

    values: list[torch.Tensor | float] = field(default_factory=list)
    denominators: list[torch.Tensor] = field(default_factory=list)
    reduce_type: ReduceType = ReduceType.AVG


class StatsTracker:
    """Tracks and aggregates training statistics.

    Usage:
        tracker = StatsTracker("train")

        # Record a scalar
        tracker.scalar(loss=0.5)

        # Record tensor stats with denominator (mask)
        mask = torch.ones(batch_size, dtype=torch.bool)
        tracker.denominator(batch_mask=mask)
        tracker.stat(rewards=reward_tensor, denominator="batch_mask")

        # Export aggregated stats
        stats = tracker.export()
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.lock = Lock()
        self.scope_stack: list[str] = []
        if name:
            self.scope_stack.append(name.strip("/"))

        self.stats: dict[str, StatEntry] = defaultdict(StatEntry)
        self.denominators: dict[str, str] = {}  # stat_key -> denominator_key

    def _get_full_key(self, key: str) -> str:
        """Combine scope stack with current key."""
        if not self.scope_stack:
            return key
        return "/".join(self.scope_stack + [key])

    @contextmanager
    def scope(self, name: str):
        """Context manager for hierarchical scoping."""
        with self.lock:
            self.scope_stack.append(name.strip("/"))
        try:
            yield
        finally:
            with self.lock:
                self.scope_stack.pop()

    @contextmanager
    def record_timing(self, key: str):
        """Record timing for a code block."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            with self.lock:
                full_key = f"timing/{key}"
                entry = self.stats[full_key]
                entry.values.append(elapsed)
                entry.reduce_type = ReduceType.SCALAR

    def scalar(self, **kwargs: float):
        """Record scalar values."""
        with self.lock:
            for key, value in kwargs.items():
                full_key = self._get_full_key(key)
                entry = self.stats[full_key]
                entry.values.append(float(value))
                entry.reduce_type = ReduceType.SCALAR

    def denominator(self, **kwargs: torch.Tensor):
        """Register denominator masks for subsequent stat calls."""
        with self.lock:
            for key, value in kwargs.items():
                if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
                    raise ValueError(f"`{key}` must be a pytorch bool tensor")
                full_key = self._get_full_key(key)
                entry = self.stats[full_key]
                entry.values.append(value.detach().clone())
                entry.reduce_type = ReduceType.SUM

    def stat(
        self,
        denominator: str,
        reduce_type: ReduceType | None = None,
        **kwargs: torch.Tensor,
    ):
        """Record tensor statistics with a denominator mask."""
        with self.lock:
            denom_key = self._get_full_key(denominator)
            if denom_key not in self.stats:
                raise ValueError(f"Denominator `{denom_key}` not registered")

            for key, value in kwargs.items():
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"`{key}` must be a pytorch tensor")

                full_key = self._get_full_key(key)
                entry = self.stats[full_key]
                entry.values.append(value.detach().float().clone())
                entry.reduce_type = reduce_type or ReduceType.AVG_MIN_MAX
                self.denominators[full_key] = denom_key

    def export(self, reset: bool = True) -> dict[str, float]:
        """Export aggregated statistics."""
        with self.lock:
            results: dict[str, float] = {}

            for key, entry in list(self.stats.items()):
                if not entry.values:
                    continue

                aggregated = self._aggregate(key, entry)
                results.update(aggregated)

            if reset:
                self.stats.clear()
                self.denominators.clear()

            return results

    def _aggregate(self, key: str, entry: StatEntry) -> dict[str, float]:
        """Aggregate a single stat entry."""
        result: dict[str, float] = {}

        if entry.reduce_type == ReduceType.SCALAR:
            # Average of scalar values
            if entry.values:
                result[key] = sum(entry.values) / len(entry.values)
                result[f"{key}__count"] = len(entry.values)

        elif entry.reduce_type == ReduceType.SUM:
            # Sum of tensor values
            total = sum(v.sum().item() if isinstance(v, torch.Tensor) else v for v in entry.values)
            result[key] = total

        elif entry.reduce_type in (ReduceType.AVG, ReduceType.AVG_MIN_MAX):
            # Weighted average using denominator
            denom_key = self.denominators.get(key)
            if denom_key and denom_key in self.stats:
                denom_entry = self.stats[denom_key]

                numerator = 0.0
                denominator = 0.0
                mins = []
                maxs = []

                for v, d in zip(entry.values, denom_entry.values):
                    if isinstance(v, torch.Tensor) and isinstance(d, torch.Tensor):
                        masked = torch.where(d, v, torch.zeros_like(v))
                        numerator += masked.sum().item()
                        denominator += d.sum().item()

                        if entry.reduce_type == ReduceType.AVG_MIN_MAX:
                            valid = v[d]
                            if valid.numel() > 0:
                                mins.append(valid.min().item())
                                maxs.append(valid.max().item())

                if denominator > 0:
                    avg = numerator / denominator
                    if entry.reduce_type == ReduceType.AVG_MIN_MAX:
                        result[f"{key}/avg"] = avg
                        if mins:
                            result[f"{key}/min"] = min(mins)
                        if maxs:
                            result[f"{key}/max"] = max(maxs)
                    else:
                        result[key] = avg

        elif entry.reduce_type == ReduceType.MIN:
            mins = [v.min().item() if isinstance(v, torch.Tensor) else v for v in entry.values]
            if mins:
                result[key] = min(mins)

        elif entry.reduce_type == ReduceType.MAX:
            maxs = [v.max().item() if isinstance(v, torch.Tensor) else v for v in entry.values]
            if maxs:
                result[key] = max(maxs)

        return result


class DeferredStats:
    """Statistics recorded now and added to a tracker later.

    It takes the same ``scalar``, ``denominator`` and ``stat`` calls as :class:`StatsTracker`, checks them the same
    way, keeps them in order, and :meth:`record` makes them on a tracker, in its scope.  A rollout group records what
    it measured about itself here and the trainer records it in the training step that consumes the group: recorded
    when the group completed, it would land in whichever step was running then, which with prefetching is often an
    earlier one.
    """

    def __init__(self) -> None:
        self._calls: list[tuple[str, dict[str, Any]]] = []
        self._denominators: set[str] = set()

    def __len__(self) -> int:
        return len(self._calls)

    def scalar(self, **kwargs: float) -> None:
        self._calls.append(("scalar", {key: float(value) for key, value in kwargs.items()}))

    def denominator(self, **kwargs: torch.Tensor) -> None:
        for key, value in kwargs.items():
            if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
                raise ValueError(f"`{key}` must be a pytorch bool tensor")
        self._denominators.update(kwargs)
        self._calls.append(("denominator", {key: value.detach().clone() for key, value in kwargs.items()}))

    def stat(self, denominator: str, reduce_type: ReduceType | None = None, **kwargs: torch.Tensor) -> None:
        if denominator not in self._denominators:
            raise ValueError(f"Denominator `{denominator}` not registered")
        for key, value in kwargs.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"`{key}` must be a pytorch tensor")
        values = {key: value.detach().clone() for key, value in kwargs.items()}
        self._calls.append(("stat", {"denominator": denominator, "reduce_type": reduce_type, **values}))

    def record(self, tracker: Any) -> None:
        """Make every call, in order, on ``tracker`` (a :class:`StatsTracker` or anything with its methods)."""

        for method, kwargs in self._calls:
            getattr(tracker, method)(**kwargs)

    def to_json(self) -> list[Any]:
        """The calls, in order, as JSON data, which :meth:`from_json` reads back."""

        return [[method, {key: _value_json(value) for key, value in kwargs.items()}] for method, kwargs in self._calls]

    @classmethod
    def from_json(cls, data: list[Any]) -> "DeferredStats":
        """The statistics :meth:`to_json` wrote, each call made again, and so checked again."""

        stats = cls()
        for method, kwargs in data:
            if method not in ("scalar", "denominator", "stat"):
                raise ValueError(f"not a statistics call: {method!r}")
            getattr(stats, method)(**{str(key): _value_from_json(value) for key, value in kwargs.items()})
        return stats


def _value_json(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"tensor": str(value.dtype).removeprefix("torch."), "shape": list(value.shape), "values": value.tolist()}
    if isinstance(value, ReduceType):
        return {"reduce": value.name}
    return value


def _value_from_json(value: Any) -> Any:
    if isinstance(value, dict) and "tensor" in value:
        dtype = getattr(torch, str(value["tensor"]))
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"not a tensor type: {value['tensor']!r}")
        return torch.tensor(value["values"], dtype=dtype).reshape(value["shape"])
    if isinstance(value, dict) and "reduce" in value:
        return ReduceType[str(value["reduce"])]
    return value


# Global trackers registry
_TRACKERS: dict[str, StatsTracker] = {}
_LOCK = Lock()


def get(name: str = "") -> StatsTracker:
    """Get or create a named tracker."""
    global _TRACKERS
    with _LOCK:
        if name not in _TRACKERS:
            _TRACKERS[name] = StatsTracker(name)
        return _TRACKERS[name]


def export_all(reset: bool = True) -> dict[str, float]:
    """Export stats from all trackers."""
    global _TRACKERS
    results: dict[str, float] = {}
    with _LOCK:
        for tracker in _TRACKERS.values():
            results.update(tracker.export(reset=reset))
    return results
