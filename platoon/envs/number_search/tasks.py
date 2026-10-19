"""Number search's tasks, generated from a seed: the same task specs on every run.

:func:`generate` with its defaults (seed 42) gives the environment's 50,000 ``train`` and 1,000 ``val`` tasks, with
targets in 0-1024.  Each task spec is ``{"goal", "id", "max_steps", "misc": {"low", "high", "target"}}``: the goal
names a range around the target, and the triple ``(low, target, high)`` is unique across both splits.  A triple goes
to ``val`` when a hash of it falls below the val split's share of all tasks.  Standard library only.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

SEED = 42
TRAIN_SIZE = 50_000
VAL_SIZE = 1_000
MIN_BOUND = 0
MAX_BOUND = 1024


def _bounds_around(target: int, rng: random.Random, min_bound: int, max_bound: int) -> tuple[int, int]:
    """``(low, high)`` with ``low < target < high`` where the bounds allow it, each drawn uniformly."""

    low = min_bound if target <= min_bound else rng.randint(min_bound, max(min_bound, target - 1))
    high = max_bound if target >= max_bound else rng.randint(min(max_bound, target + 1), max_bound)
    if low >= target:
        low = max(min_bound, target - 1)
    if high <= target:
        high = min(max_bound, target + 1)
    if low >= high:  # the smallest window around the target, else the whole range
        low, high = max(min_bound, target - 1), min(max_bound, target + 1)
        if low >= high:
            low, high = min_bound, max_bound
    return low, high


def generate(
    seed: int = SEED,
    train_size: int = TRAIN_SIZE,
    val_size: int = VAL_SIZE,
    min_bound: int = MIN_BOUND,
    max_bound: int = MAX_BOUND,
) -> dict[str, list[dict[str, Any]]]:
    """The ``train`` and ``val`` task specs that ``seed`` gives, each split in the order its tasks were drawn."""

    if max_bound <= min_bound:
        raise ValueError("max_bound must be greater than min_bound")
    rng = random.Random(seed)
    val_share = val_size / max(1, train_size + val_size)

    def in_val(low: int, target: int, high: int) -> bool:
        digest = hashlib.sha256(f"{seed}:{low}:{target}:{high}".encode()).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF < val_share

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    sizes = {"train": train_size, "val": val_size}
    drawn: set[tuple[int, int, int]] = set()
    while len(splits["train"]) < train_size or len(splits["val"]) < val_size:
        target = rng.randint(min_bound, max_bound)
        low, high = _bounds_around(target, rng, min_bound, max_bound)
        if (low, target, high) in drawn:
            continue
        drawn.add((low, target, high))
        split = "val" if in_val(low, target, high) else "train"
        tasks = splits[split]
        if len(tasks) < sizes[split]:
            tasks.append(
                {
                    "goal": f"Guess the correct number between {low} and {high}.",
                    "id": f"number_search.{split}.{len(tasks)}",
                    "max_steps": 1,
                    "misc": {"low": low, "high": high, "target": target},
                }
            )
    return splits
