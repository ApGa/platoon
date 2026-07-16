"""Dataset ordering controls shared by Tinker environment integrations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

PRESERVE_DATASET_ORDER_COLUMN = "_platoon_preserve_dataset_order"


def prepare_dataset_for_dataloader(
    dataset: Dataset,
    *,
    shuffle_seed: int | None,
) -> Dataset:
    """Honor an explicit ordered-dataset marker, otherwise preserve legacy shuffle."""

    if PRESERVE_DATASET_ORDER_COLUMN not in dataset.column_names:
        return dataset.shuffle(seed=shuffle_seed) if shuffle_seed is not None else dataset

    marker_values = dataset[PRESERVE_DATASET_ORDER_COLUMN]
    if not marker_values or any(value is not True for value in marker_values):
        raise ValueError(
            f"{PRESERVE_DATASET_ORDER_COLUMN} must be true for every ordered record"
        )
    return dataset.remove_columns(PRESERVE_DATASET_ORDER_COLUMN)
