"""Harbor datasets: task directories indexed by id, with optional named splits."""

from __future__ import annotations

import json
import logging
import os
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from platoon.envs.harbor.holdout import HoldoutRow
from platoon.envs.harbor.inputs import input_files
from platoon.envs.harbor.task import CONFIG_FILENAME, AgentUserPolicy, HarborTask, HarborTaskError

logger = logging.getLogger(__name__)

SplitManifest = str | Path | Mapping[str, Sequence[str]]
# A dataset's own index files at its root: Harbor's registry (a list of datasets, each naming its tasks' paths) and
# a per-task metadata table (``{id: {"source_dataset": ..., ...}}``, as large Harbor datasets publish it).
REGISTRY_FILENAME = "registry.json"
TASK_INDEX_FILENAME = "task-index.json"


class HarborDataset:
    """Harbor tasks keyed by id.

    Tasks come from a dataset directory or an explicit list of task directories; directories platoon cannot run
    are skipped with a warning.  In a dataset directory with a ``registry.json`` (Harbor's registry format), the
    registry lists the tasks and their ids are its names; otherwise every directory below it that holds a
    ``task.toml`` is a task, whose id is its directory's name when it sits directly in the dataset directory and
    its relative path (``group/name``) when it sits deeper.  Tasks of a ``task_dirs`` list are named by their
    directories.  No two tasks may share an id, whether platoon can run them or not.  A ``task-index.json`` next
    to the registry gives each task its ``HarborTask.index`` row.

    Splits come from a manifest mapping split names to task ids: a mapping, or a JSON or TOML file (relative paths
    resolve against ``dataset_dir``).  Without a manifest every split holds every task.
    """

    def __init__(
        self,
        dataset_dir: str | Path | None = None,
        *,
        task_dirs: Sequence[str | Path] | None = None,
        splits: SplitManifest | None = None,
        agent_user: AgentUserPolicy = "refuse",
    ) -> None:
        if (dataset_dir is None) == (task_dirs is None):
            raise ValueError("Pass exactly one of dataset_dir or task_dirs")
        if dataset_dir is not None:
            candidates = list(dataset_tasks(dataset_dir))
            index = read_task_index(dataset_dir)
        else:
            candidates = [(Path(task_dir).name, Path(task_dir)) for task_dir in task_dirs or []]
            index = {}
        self._tasks: dict[str, HarborTask] = {}
        self._refused: dict[str, str] = {}  # task id -> why platoon cannot run it
        self._index = index
        self._dirs = {task_id: Path(task_dir) for task_id, task_dir in candidates}
        for task_id, task_dir in candidates:
            # Refused ids count too: get() and the manifest check answer a refused id with its refusal.
            if task_id in self._tasks or task_id in self._refused:
                raise ValueError(f"Two Harbor tasks are named {task_id!r}")
            try:
                task = HarborTask.load(task_dir, agent_user=agent_user, index=index.get(task_id))
            except HarborTaskError as exc:
                logger.warning("Skipping Harbor task %s: %s", task_dir, exc)
                self._refused[task_id] = str(exc)
                continue
            self._tasks[task_id] = task
        if not self._tasks:
            raise ValueError(f"No runnable Harbor tasks in {dataset_dir or task_dirs}{self._why(self._refused)}")
        self._splits = _load_splits(splits, dataset_dir) if splits is not None else None
        if self._splits is not None:
            named = {name for names in self._splits.values() for name in names}
            if refused := sorted(named & set(self._refused)):
                raise ValueError(
                    "The split manifest names Harbor tasks platoon cannot run"
                    f"{self._why({name: self._refused[name] for name in refused})}"
                )
            unknown = sorted(named - set(self._tasks))
            if unknown:
                raise ValueError(f"The split manifest names unknown Harbor tasks: {unknown}")

    def names(self, split: str) -> list[str]:
        if self._splits is None:
            return list(self._tasks)
        if split not in self._splits:
            raise KeyError(f"Unknown split {split!r}; the manifest has {sorted(self._splits)}")
        return list(self._splits[split])

    def split_names(self) -> list[str]:
        """The manifest's splits, in its order; none without a manifest, where every split holds every task."""

        return list(self._splits or {})

    def holdout_rows(self) -> list[HoldoutRow]:
        """Every task, refused ones included (``task_config``), with its ``task-index.json`` row and its input files
        (:func:`~platoon.envs.harbor.inputs.input_files`, hashed now): what a hold-out splits."""

        rows = [
            HoldoutRow(name, dict(task.index or {}), input_files=input_files(self._dirs[name]))
            for name, task in self._tasks.items()
        ]
        rows += [
            HoldoutRow(name, dict(self._index.get(name) or {}), f"task_config: {reason}", input_files(self._dirs[name]))
            for name, reason in self._refused.items()
        ]
        return rows

    def get(self, name: str) -> HarborTask:
        """The task named ``name``; a ``HarborTaskError`` (a ``ValueError``) says why one platoon refused cannot run."""

        if name in self._refused:
            raise HarborTaskError(self._refused[name])
        try:
            return self._tasks[name]
        except KeyError:
            raise KeyError(f"Unknown Harbor task {name!r}") from None

    @staticmethod
    def _why(refused: dict[str, str], limit: int = 3) -> str:
        """``: <reason>; <reason>`` for the first refused tasks, to say what a refusal was about."""

        if not refused:
            return ""
        reasons = [refused[name] for name in sorted(refused)[:limit]]
        more = f"; and {len(refused) - limit} more" if len(refused) > limit else ""
        return ": " + "; ".join(reasons) + more


def dataset_tasks(dataset_dir: str | Path) -> Iterator[tuple[str, Path]]:
    """``(id, task directory)`` of every task of a dataset directory, in id order (see :class:`HarborDataset`).

    With a ``registry.json``, only the tasks it lists whose directories exist here (a partial download of a large
    dataset has the registry of the whole one).
    """

    root = Path(dataset_dir)
    registry = read_registry(root)
    if registry is not None:
        missing = 0
        for task_id, relative in sorted(registry.items()):
            task_dir = root / relative
            if (task_dir / CONFIG_FILENAME).is_file():
                yield task_id, task_dir
            else:
                missing += 1
        if missing:
            logger.info("%s lists %d Harbor tasks that are not under %s", REGISTRY_FILENAME, missing, root)
        return
    yield from sorted((task_dir.relative_to(root).as_posix(), task_dir) for task_dir in _scan(root))


_UNREAD: Any = object()


def task_dir_of(dataset_dir: str | Path, task_id: str, *, registry: dict[str, str] | None = _UNREAD) -> Path | None:
    """The directory of task ``task_id`` of a dataset directory, found without loading any task: through its
    ``registry.json`` when it has one, else by the id as a relative path.  None when there is no such task.
    ``registry`` is the dataset's :func:`read_registry`, for a caller that has read it already."""

    root = Path(dataset_dir)
    if registry is _UNREAD:
        registry = read_registry(root)
    if registry is not None:
        relative = registry.get(task_id)
        return None if relative is None else root / relative
    parts = task_id.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return None  # only a task's own id names it: no absolute, empty, "." or ".." parts
    task_dir = root.joinpath(*parts)
    return task_dir if (task_dir / CONFIG_FILENAME).is_file() else None


def read_registry(dataset_dir: str | Path) -> dict[str, str] | None:
    """``{task name: relative path}`` from the dataset's ``registry.json``, or None when it has none.

    The registry is Harbor's: a list of datasets, each with ``tasks`` of ``{name, path}`` (and ``git_url`` for
    tasks fetched from elsewhere, which a local directory cannot hold and which are refused).
    """

    path = Path(dataset_dir) / REGISTRY_FILENAME
    if not path.is_file():
        return None
    try:
        registry = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path} is not a Harbor registry: {exc}") from exc
    datasets = [registry] if isinstance(registry, dict) else registry  # a dataset's own spec, or a registry
    if not isinstance(datasets, list) or not all(
        isinstance(dataset, dict) and isinstance(dataset.get("tasks"), list) for dataset in datasets
    ):
        raise ValueError(f"{path} is not a Harbor registry: it must list datasets, each with a list of tasks")
    tasks: dict[str, str] = {}
    for dataset in datasets:
        for entry in dataset["tasks"]:
            name, relative = (entry.get("name"), entry.get("path")) if isinstance(entry, dict) else (None, None)
            if not isinstance(name, str) or not isinstance(relative, str):
                raise ValueError(f"{path}: every task needs a name and a path, got {entry!r}")
            if entry.get("git_url"):
                raise ValueError(f"{path}: task {name!r} lives in {entry['git_url']}, not in this directory")
            parts = PurePosixPath(relative).parts
            if PurePosixPath(relative).is_absolute() or ".." in parts:
                raise ValueError(f"{path}: task {name!r} has a path outside the dataset: {relative!r}")
            if tasks.setdefault(name, relative) != relative:
                raise ValueError(f"Two Harbor tasks are named {name!r}")
    return tasks


def read_task_index(dataset_dir: str | Path) -> dict[str, dict[str, Any]]:
    """The dataset's ``task-index.json`` (``{task id: {field: value}}``), or an empty mapping without one."""

    path = Path(dataset_dir) / TASK_INDEX_FILENAME
    if not path.is_file():
        return {}
    try:
        index = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path} is not a task index: {exc}") from exc
    if not isinstance(index, dict) or not all(isinstance(row, dict) for row in index.values()):
        raise ValueError(f"{path} is not a task index: it must map task ids to tables")
    return index


def _scan(root: Path) -> Iterator[Path]:
    """Every directory below ``root`` that holds a ``task.toml``; a task's own subdirectories are not searched, and
    neither are hidden directories and ``__pycache__``."""

    for directory, subdirectories, files in os.walk(root):
        if CONFIG_FILENAME in files and Path(directory) != root:
            subdirectories.clear()
            yield Path(directory)
            continue
        subdirectories[:] = sorted(
            name for name in subdirectories if not name.startswith(".") and name != "__pycache__"
        )


def _load_splits(splits: SplitManifest, dataset_dir: str | Path | None) -> dict[str, list[str]]:
    if isinstance(splits, Mapping):
        manifest = splits
    else:
        path = Path(splits)
        if not path.is_absolute() and dataset_dir is not None:
            path = Path(dataset_dir) / path
        text = path.read_text()
        manifest = tomllib.loads(text) if path.suffix == ".toml" else json.loads(text)
    for split, names in manifest.items():
        if isinstance(names, str) or not isinstance(names, Sequence) or not all(isinstance(n, str) for n in names):
            raise ValueError(f"Split {split!r} must list task names, got {names!r}")
    return {split: list(names) for split, names in manifest.items()}
