from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "platoon_tinker_dataset_order_test",
    REPO_ROOT / "platoon/train/tinker/dataset_order.py",
)
assert _spec is not None and _spec.loader is not None
_dataset_order = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dataset_order)
PRESERVE_DATASET_ORDER_COLUMN = _dataset_order.PRESERVE_DATASET_ORDER_COLUMN
epoch_order = _dataset_order.epoch_order


class FakeDataset:
    def __init__(self, rows):
        self.rows = [dict(row) for row in rows]
        self.shuffle_seeds = []

    @property
    def column_names(self):
        return list(self.rows[0]) if self.rows else []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, slice):  # a slice of rows, as columns
            return {name: [row.get(name) for row in self.rows[key]] for name in self.column_names}
        return [row.get(key) for row in self.rows]

    def shuffle(self, seed):
        self.shuffle_seeds.append(seed)
        rows = list(self.rows)
        random.Random(seed).shuffle(rows)
        return type(self)(rows)

    def remove_columns(self, key):
        return type(self)([{name: value for name, value in row.items() if name != key} for row in self.rows])

    def to_list(self):
        return [dict(row) for row in self.rows]


def test_preserve_order_marker_skips_shuffle_and_is_removed():
    dataset = FakeDataset([{"task_id": f"task-{index}", PRESERVE_DATASET_ORDER_COLUMN: True} for index in range(8)])

    order = epoch_order(dataset, shuffle_seed=42)

    assert [row["task_id"] for row in order.epoch(0)] == [f"task-{index}" for index in range(8)]
    assert all(PRESERVE_DATASET_ORDER_COLUMN not in row for row in order.epoch(0))
    assert order.epoch(3) == order.epoch(0) and order.epoch_size == 8
    assert dataset.shuffle_seeds == []


def test_unmarked_dataset_is_drawn_in_seeded_shuffled_passes_that_go_on_across_epochs():
    """Pass p is the dataset shuffled with seed + p * 101; an epoch of 6 of the 8 rows leaves the last 2 of its pass
    to start the next epoch, which any epoch reads directly."""

    dataset = FakeDataset([{"task_id": f"task-{index}"} for index in range(8)])

    order = epoch_order(dataset, shuffle_seed=42, epoch_size=6)

    def shuffled(seed):
        ids = [f"task-{index}" for index in range(8)]
        random.Random(seed).shuffle(ids)
        return ids

    passes = shuffled(42) + shuffled(42 + 101) + shuffled(42 + 202)
    assert [row["task_id"] for row in order.epoch(0)] == passes[:6]
    assert [row["task_id"] for row in order.epoch(2)] == passes[12:18]
    assert [row["task_id"] for row in order.epoch(1)] == passes[6:12]
    assert dataset.shuffle_seeds[:2] == [42, 143]
    assert [row["task_id"] for row in epoch_order(dataset, shuffle_seed=None).epoch(0)] == dataset["task_id"]


def test_preserve_order_marker_must_cover_every_record():
    dataset = FakeDataset(
        [
            {"task_id": "first", PRESERVE_DATASET_ORDER_COLUMN: True},
            {"task_id": "second", PRESERVE_DATASET_ORDER_COLUMN: False},
        ]
    )

    with pytest.raises(ValueError, match="must be true for every ordered record"):
        epoch_order(dataset, shuffle_seed=42)
