"""Task loading for the email-search benchmark from HuggingFace datasets."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from datasets import load_dataset

from platoon.envs.base import Task

from .data.query_iterators import BAD_QUERIES

DATASET_NAME = "corbt/enron_emails_sample_questions"
DATASET_SPLITS = ("train", "test")
DEFAULT_MAX_STEPS = 20

_DATA_CACHE: dict[str, list[dict[str, Any]]] = {}
_TASK_CACHE: dict[str, Task] = {}


def _get_split_data(split: Literal["train", "test"]) -> list[dict[str, Any]]:
    if split not in _DATA_CACHE:
        dataset = load_dataset(DATASET_NAME, split=split)
        _DATA_CACHE[split] = [dict(example) for example in dataset]
    return _DATA_CACHE[split]


def _example_to_task(example: dict[str, Any], split: Literal["train", "test"], idx: int) -> Task:
    task_id = f"email_search.{split}.{idx}"
    misc = dict(example)
    misc["ground_truth"] = str(example["answer"])
    misc["gold_message_ids"] = [str(message_id) for message_id in example["message_ids"]]
    misc["dataset_name"] = DATASET_NAME
    misc["dataset_split"] = split
    misc["dataset_index"] = idx
    return Task(
        goal=str(example["question"]),
        id=task_id,
        max_steps=DEFAULT_MAX_STEPS,
        misc=misc,
    )


def get_task_ids(
    split: Literal["train", "test"] = "train",
    max_messages: int | None = 1,
    exclude_known_bad_queries: bool = True,
) -> list[str]:
    task_ids: list[str] = []
    for idx, example in enumerate(_get_split_data(split)):
        if max_messages is not None and len(example["message_ids"]) > max_messages:
            continue
        if exclude_known_bad_queries and int(example["id"]) in BAD_QUERIES:
            continue
        task_ids.append(f"email_search.{split}.{idx}")
    return task_ids


def load_task_from_hf(task_id: str) -> Task:
    parts = task_id.split(".")
    if len(parts) != 3 or parts[0] != "email_search":
        raise ValueError(f"Invalid task ID format: {task_id}")

    split = parts[1]
    if split not in DATASET_SPLITS:
        raise ValueError(f"Invalid email-search split: {split}")

    idx = int(parts[2])
    data = _get_split_data(split)
    if idx >= len(data):
        raise IndexError(f"Task index {idx} out of range for split {split}")

    return _example_to_task(data[idx], split, idx)


def get_task(task_id: str) -> Task:
    if task_id not in _TASK_CACHE:
        _TASK_CACHE[task_id] = load_task_from_hf(task_id)
    return deepcopy(_TASK_CACHE[task_id])
