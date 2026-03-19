"""Task loading for DeepDive benchmark from HuggingFace datasets."""

from __future__ import annotations

from typing import Any, Literal

from platoon.envs.base import Task

DATASET_NAME = "zai-org/DeepDive"
DATASET_SPLITS = ("qa_rl", "qa_sft")
DEFAULT_MAX_STEPS = 50

_DATA_CACHE: dict[str, list[dict[str, Any]]] = {}
_TASK_CACHE: dict[str, Task] = {}


def _load_dataset_from_hf(split: Literal["qa_rl", "qa_sft"]) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets library is required. Install with: pip install datasets"
        ) from exc

    dataset = load_dataset(DATASET_NAME, split=split)
    return [dict(example) for example in dataset]


def _get_split_data(split: Literal["qa_rl", "qa_sft"]) -> list[dict[str, Any]]:
    if split not in _DATA_CACHE:
        _DATA_CACHE[split] = _load_dataset_from_hf(split)
    return _DATA_CACHE[split]


def _example_to_task(
    example: dict[str, Any],
    split: Literal["qa_rl", "qa_sft"],
    idx: int,
) -> Task:
    task_id = f"deepdive.{split}.{idx}"
    question = str(example["question"])
    answer = str(example["answer"])

    task_misc = dict(example)
    task_misc["ground_truth"] = answer
    task_misc["dataset_name"] = DATASET_NAME
    task_misc["dataset_split"] = split
    task_misc["dataset_index"] = idx

    return Task(
        goal=question,
        id=task_id,
        max_steps=DEFAULT_MAX_STEPS,
        misc=task_misc,
    )


def get_task_ids(split: Literal["qa_rl", "qa_sft"] = "qa_sft") -> list[str]:
    return [f"deepdive.{split}.{idx}" for idx, _ in enumerate(_get_split_data(split))]


def load_task_from_hf(task_id: str) -> Task:
    parts = task_id.split(".")
    if len(parts) != 3 or parts[0] != "deepdive":
        raise ValueError(f"Invalid task ID format: {task_id}")

    split = parts[1]
    if split not in DATASET_SPLITS:
        raise ValueError(f"Invalid DeepDive split: {split}")

    idx = int(parts[2])
    data = _get_split_data(split)
    if idx >= len(data):
        raise IndexError(f"Task index {idx} out of range for split {split}")

    return _example_to_task(data[idx], split, idx)


def get_task(task_id: str) -> Task:
    if task_id not in _TASK_CACHE:
        _TASK_CACHE[task_id] = load_task_from_hf(task_id)
    return _TASK_CACHE[task_id]