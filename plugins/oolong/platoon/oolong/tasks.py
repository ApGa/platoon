"""Task loading for Oolong benchmark from HuggingFace datasets."""
import argparse
import json
from copy import deepcopy
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional

from platoon.envs.base import Task


class TaskGroup(str, Enum):
    """Task groups in Oolong benchmark."""
    COUNTING = "counting"
    USER = "user"
    TIMELINE = "timeline"


class AnswerType(str, Enum):
    """Answer types in Oolong benchmark."""
    NUMERIC = "NUMERIC"
    LABEL = "LABEL"
    COMPARISON = "COMPARISON"
    USER = "USER"
    MONTH_YEAR = "MONTH_YEAR"


# Cache for loaded data
_SYNTH_VALIDATION_DATA: Optional[List[Dict]] = None
_SYNTH_TEST_DATA: Optional[List[Dict]] = None
_REAL_VALIDATION_DATA: Optional[List[Dict]] = None
_REAL_TEST_DATA: Optional[List[Dict]] = None
_TASKS: Dict[str, Task] = {}
_VALID_DATASETS = ("synth", "real")


def _normalize_datasets(
    dataset: Literal["synth", "real", "both"] | list[str],
) -> list[str]:
    """Normalize dataset selection into a deduplicated ordered list."""
    if isinstance(dataset, str):
        datasets = list(_VALID_DATASETS) if dataset == "both" else [dataset]
    else:
        datasets = list(dataset)

    invalid = [ds for ds in datasets if ds not in _VALID_DATASETS]
    if invalid:
        raise ValueError(
            f"Invalid dataset(s): {invalid}. Must be drawn from {_VALID_DATASETS}."
        )

    normalized: list[str] = []
    for ds in datasets:
        if ds not in normalized:
            normalized.append(ds)
    return normalized


def _get_example_context_len(example: Dict) -> Optional[int]:
    """Get context length directly from the raw context text."""
    context_text = example.get("context_window_text") or example.get("context")
    if context_text is None:
        return None

    return len(context_text)


def _matches_filters(
    example: Dict,
    task_group: Optional[TaskGroup] = None,
    answer_type: Optional[AnswerType] = None,
    min_context_len: Optional[int] = None,
    max_context_len: Optional[int] = None,
) -> bool:
    """Return whether a dataset example matches all requested filters."""
    if task_group and example.get("task_group") != task_group.value:
        return False
    if answer_type and example.get("answer_type") != answer_type.value:
        return False

    if min_context_len is None and max_context_len is None:
        return True

    context_len = _get_example_context_len(example)
    if context_len is None:
        return False
    if min_context_len is not None and context_len < min_context_len:
        return False
    if max_context_len is not None and context_len > max_context_len:
        return False

    return True


def _load_dataset_from_hf(dataset_name: str, split: str) -> List[Dict]:
    """Load dataset from HuggingFace.

    Args:
        dataset_name: Either "oolong-synth" or "oolong-real"
        split: Either "validation" or "test"

    Returns:
        List of data examples as dictionaries
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library is required. Install with: pip install datasets"
        )
    hf_name = f"oolongbench/{dataset_name}"
    if dataset_name == "oolong-real":
        dataset = load_dataset(hf_name, "dnd", split=split)
    else:
        dataset = load_dataset(hf_name, split=split)
    return [dict(example) for example in dataset]


def _get_synth_data(split: str) -> List[Dict]:
    """Get oolong-synth data for the given split."""
    global _SYNTH_VALIDATION_DATA, _SYNTH_TEST_DATA

    if split == "validation":
        if _SYNTH_VALIDATION_DATA is None:
            _SYNTH_VALIDATION_DATA = _load_dataset_from_hf("oolong-synth", "validation")
        return _SYNTH_VALIDATION_DATA
    elif split == "test":
        if _SYNTH_TEST_DATA is None:
            _SYNTH_TEST_DATA = _load_dataset_from_hf("oolong-synth", "test")
        return _SYNTH_TEST_DATA
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'validation' or 'test'")


def _get_real_data(split: str) -> List[Dict]:
    """Get oolong-real data for the given split."""
    global _REAL_VALIDATION_DATA, _REAL_TEST_DATA

    if split == "validation":
        if _REAL_VALIDATION_DATA is None:
            _REAL_VALIDATION_DATA = _load_dataset_from_hf("oolong-real", "validation")
        return _REAL_VALIDATION_DATA
    elif split == "test":
        if _REAL_TEST_DATA is None:
            _REAL_TEST_DATA = _load_dataset_from_hf("oolong-real", "test")
        return _REAL_TEST_DATA
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'validation' or 'test'")


def _example_to_task(example: Dict, dataset: str, split: str, idx: int) -> Task:
    """Convert a HuggingFace example to a Task object.

    Args:
        example: Dictionary with keys like context_window_text, question, answer, etc.
        dataset: "synth" or "real"
        split: "validation" or "test"
        idx: Index of the example

    Returns:
        Task object
    """
    # Build task ID
    task_id = f"oolong.{dataset}.{split}.{idx}"

    # Get context text
    example['context'] = example.pop("context_window_text")
    example.pop('context_window_text_with_labels', None)

    return Task(
        goal=example['question'],
        max_steps=50,
        id=task_id,
        misc=example,
    )


def get_synth_task_ids(
    split: Literal["validation", "test"] = "validation",
    task_group: Optional[TaskGroup] = None,
    answer_type: Optional[AnswerType] = None,
    min_context_len: Optional[int] = None,
    max_context_len: Optional[int] = None,
) -> List[str]:
    """Get task IDs for oolong-synth dataset.

    Args:
        split: "validation" or "test"
        task_group: Filter by task group (counting, user, timeline)
        answer_type: Filter by answer type (NUMERIC, LABEL, etc.)
        min_context_len: Filter to only include tasks with context >= this length
        max_context_len: Filter to only include tasks with context <= this length

    Returns:
        List of task IDs
    """
    data = _get_synth_data(split)
    task_ids = []

    for idx, example in enumerate(data):
        if not _matches_filters(
            example,
            task_group=task_group,
            answer_type=answer_type,
            min_context_len=min_context_len,
            max_context_len=max_context_len,
        ):
            continue

        task_ids.append(f"oolong.synth.{split}.{idx}")

    return task_ids


def get_real_task_ids(
    split: Literal["validation", "test"] = "validation",
    task_group: Optional[TaskGroup] = None,
    answer_type: Optional[AnswerType] = None,
    min_context_len: Optional[int] = None,
    max_context_len: Optional[int] = None,
) -> List[str]:
    """Get task IDs for oolong-real dataset.

    Args:
        split: "validation" or "test"
        task_group: Filter by task group
        answer_type: Filter by answer type
        min_context_len: Filter to only include tasks with context >= this length
        max_context_len: Filter to only include tasks with context <= this length

    Returns:
        List of task IDs
    """
    data = _get_real_data(split)
    task_ids = []

    for idx, example in enumerate(data):
        if not _matches_filters(
            example,
            task_group=task_group,
            answer_type=answer_type,
            min_context_len=min_context_len,
            max_context_len=max_context_len,
        ):
            continue

        task_ids.append(f"oolong.real.{split}.{idx}")

    return task_ids


def get_task_ids(
    dataset: Literal["synth", "real", "both"] | list[str] = "synth",
    split: Literal["validation", "test"] = "validation",
    **kwargs
) -> List[str]:
    """Get task IDs for oolong benchmark.

    Args:
        dataset: "synth", "real", "both", or a list containing dataset names
        split: "validation" or "test"
        **kwargs: Additional filters (task_group, answer_type, min_context_len, max_context_len)

    Returns:
        List of task IDs
    """
    task_ids: list[str] = []
    for dataset_name in _normalize_datasets(dataset):
        if dataset_name == "synth":
            task_ids.extend(get_synth_task_ids(split, **kwargs))
        elif dataset_name == "real":
            task_ids.extend(get_real_task_ids(split, **kwargs))
    return task_ids


def load_task_from_hf(task_id: str) -> Task:
    """Load a task from HuggingFace.

    Args:
        task_id: Task ID in format "oolong.{synth|real}.{validation|test}.{idx}"

    Returns:
        Task object
    """
    parts = task_id.split(".")
    if len(parts) != 4 or parts[0] != "oolong":
        raise ValueError(f"Invalid task ID format: {task_id}")

    dataset = parts[1]  # synth or real
    split = parts[2]    # validation or test
    idx = int(parts[3])

    if dataset == "synth":
        data = _get_synth_data(split)
    elif dataset == "real":
        data = _get_real_data(split)
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    if idx >= len(data):
        raise IndexError(f"Task index {idx} out of range for {dataset}/{split}")

    return _example_to_task(data[idx], dataset, split, idx)


def get_task(task_id: str) -> Task:
    """Get a task by ID (with caching).

    Args:
        task_id: Task ID

    Returns:
        Task object
    """
    global _TASKS
    if task_id in _TASKS:
        return deepcopy(_TASKS[task_id])

    task = load_task_from_hf(task_id)
    _TASKS[task_id] = task
    return deepcopy(task)


def create_oolong_datasets(
    dataset: Literal["synth", "real", "both"] | list[str] = "synth",
    max_samples: Optional[int] = None,
    **kwargs
) -> tuple[List[Task], List[Task]]:
    """Create validation and test datasets for Oolong.

    Args:
        dataset: "synth", "real", "both", or a list containing dataset names
        max_samples: Maximum number of samples per split (None for all)
        **kwargs: Additional filters

    Returns:
        Tuple of (validation_tasks, test_tasks)
    """
    val_ids = get_task_ids(dataset, "validation", **kwargs)
    test_ids = get_task_ids(dataset, "test", **kwargs)

    if max_samples:
        val_ids = val_ids[:max_samples]
        test_ids = test_ids[:max_samples]

    val_tasks = [get_task(tid) for tid in val_ids]
    test_tasks = [get_task(tid) for tid in test_ids]

    return val_tasks, test_tasks


def save_tasks_to_disk(
    tasks: List[Task],
    output_path: Path,
) -> None:
    """Save tasks to a JSONL file.

    Args:
        tasks: List of Task objects
        output_path: Path to output JSONL file
    """
    with open(output_path, 'w') as f:
        for task in tasks:
            json.dump(asdict(task), f)
            f.write('\n')


def load_task_from_disk(task_id: str, data_dir: Optional[Path] = None) -> Task:
    """Load a task from disk (cached JSONL files).

    Args:
        task_id: Task ID
        data_dir: Directory containing JSONL files (defaults to plugin directory)

    Returns:
        Task object
    """
    if data_dir is None:
        data_dir = Path(__file__).parent

    parts = task_id.split(".")
    if len(parts) != 4:
        raise ValueError(f"Invalid task ID: {task_id}")

    dataset = parts[1]  # synth or real
    split = parts[2]    # validation or test
    idx = int(parts[3])

    filename = f"oolong_{dataset}_{split}.jsonl"
    filepath = data_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Task file not found: {filepath}. "
            f"Generate with: python -m platoon.oolong.tasks --generate"
        )

    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i == idx:
                return Task.from_dict(json.loads(line))

    raise IndexError(f"Task index {idx} out of range")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Oolong task files")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate JSONL files from HuggingFace"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="synth",
        choices=["synth", "real", "both"],
        help="Which dataset to generate"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples per split"
    )
    args = parser.parse_args()

    if args.generate:
        output_dir = Path(__file__).parent

        datasets = ["synth", "real"] if args.dataset == "both" else [args.dataset]

        for ds in datasets:
            print(f"Loading {ds} dataset from HuggingFace...")
            val_tasks, test_tasks = create_oolong_datasets(ds, args.max_samples)

            val_file = output_dir / f"oolong_{ds}_validation.jsonl"
            test_file = output_dir / f"oolong_{ds}_test.jsonl"

            print(f"Saving {len(val_tasks)} validation tasks to {val_file}")
            save_tasks_to_disk(val_tasks, val_file)

            print(f"Saving {len(test_tasks)} test tasks to {test_file}")
            save_tasks_to_disk(test_tasks, test_file)

        print("Done!")
    else:
        # Print dataset info
        print("Oolong Task Loading Utility")
        print("=" * 40)
        print("\nTo generate local task files, run:")
        print("  python -m platoon.oolong.tasks --generate --dataset synth")
        print("\nDatasets available:")
        print("  - oolong-synth: Synthetic aggregation tasks")
        print("  - oolong-real: D&D transcript analysis tasks")
        print("\nTask groups:")
        print("  - counting: Count items by label")
        print("  - user: Analyze patterns by user")
        print("  - timeline: Temporal analysis")
