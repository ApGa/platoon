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
prepare_dataset_for_dataloader = _dataset_order.prepare_dataset_for_dataloader


class FakeDataset:
    def __init__(self, rows):
        self.rows = [dict(row) for row in rows]
        self.shuffle_seeds = []

    @property
    def column_names(self):
        return list(self.rows[0]) if self.rows else []

    def __getitem__(self, key):
        return [row.get(key) for row in self.rows]

    def shuffle(self, seed):
        self.shuffle_seeds.append(seed)
        rows = list(self.rows)
        random.Random(seed).shuffle(rows)
        return type(self)(rows)

    def remove_columns(self, key):
        return type(self)(
            [
                {name: value for name, value in row.items() if name != key}
                for row in self.rows
            ]
        )


def test_preserve_order_marker_skips_shuffle_and_is_removed():
    dataset = FakeDataset(
        [
            {"task_id": f"task-{index}", PRESERVE_DATASET_ORDER_COLUMN: True}
            for index in range(8)
        ]
    )

    prepared = prepare_dataset_for_dataloader(dataset, shuffle_seed=42)

    assert prepared["task_id"] == [f"task-{index}" for index in range(8)]
    assert PRESERVE_DATASET_ORDER_COLUMN not in prepared.column_names
    assert dataset.shuffle_seeds == []


def test_unmarked_dataset_preserves_legacy_shuffle():
    dataset = FakeDataset(
        [{"task_id": f"task-{index}"} for index in range(8)]
    )

    prepared = prepare_dataset_for_dataloader(dataset, shuffle_seed=42)

    assert dataset.shuffle_seeds == [42]
    assert prepared["task_id"] != dataset["task_id"]


def test_preserve_order_marker_must_cover_every_record():
    dataset = FakeDataset(
        [
            {"task_id": "first", PRESERVE_DATASET_ORDER_COLUMN: True},
            {"task_id": "second", PRESERVE_DATASET_ORDER_COLUMN: False},
        ]
    )

    with pytest.raises(ValueError, match="must be true for every ordered record"):
        prepare_dataset_for_dataloader(dataset, shuffle_seed=42)
