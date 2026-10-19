"""The Harbor service's view of a fetched dataset: task ids from a lock, task directories from the cache, and the
images sessions run (the site's image map plus each task's deferred Dockerfile steps).

A service configured with ``dataset`` (a source to fetch) or ``lock`` (a lock to read) lists and finds its tasks
here instead of scanning a directory. The trainer, which lists the tasks, fetches a pinned ``dataset`` whose lock
does not exist yet; rollout workers only read the lock, and check a task's cache entry against its digest once per
node before its first session.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import Any

from platoon.envs.harbor.dockerfile import DockerfileError, plan_dockerfile
from platoon.envs.harbor.fetch import FetchConfig, ensure_lock, fetch_config
from platoon.envs.harbor.holdout import HoldoutRow
from platoon.envs.harbor.images import ImageMap, SessionImage, session_image
from platoon.envs.harbor.lock import DatasetLock, LockedDataset, LockError, default_cache, read_lock

logger = logging.getLogger(__name__)

HARBOR_MISC_KEY = "harbor"
_PLANS: dict[tuple[Path, str], dict[str, Any] | None] = {}  # (task dir, context) -> plan made at open
_PLANS_LOCK = threading.Lock()


class LockedTasks:
    """Lock-backed task lookup for :class:`~platoon.envs.harbor.service.HarborEnvironmentService`."""

    def __init__(
        self,
        *,
        dataset: str | None = None,
        lock: str | Path | None = None,
        cache: str | Path | None = None,
        tasks: str | Path | Sequence[str | Path] = (),
        select: Mapping[str, Any] | None = None,
        overlays: str | Path | Sequence[str | Path] = (),
        refuse: Mapping[str, Any] | None = None,
        index: Sequence[str | Mapping[str, Any]] = (),
        splits: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        if dataset is None and lock is None:
            raise ValueError("LockedTasks needs a dataset or a lock")
        self.fetch: FetchConfig | None = (
            fetch_config(
                dataset,
                cache=cache,
                lock=lock,
                tasks=tasks,
                select=select,
                overlays=overlays,
                refuse=refuse,
                index=index,
            )
            if dataset is not None
            else None
        )
        if self.fetch is None and (tasks or select or overlays or refuse or index):
            raise ValueError("tasks, select, overlays, refuse and index choose what a fetch of `dataset` locks")
        if self.fetch is not None:
            self.lock_path = self.fetch.lock_path()
        else:
            assert lock is not None  # checked above: a dataset or a lock
            self.lock_path = Path(lock).expanduser()
        self.cache = self.fetch.cache if self.fetch is not None else (Path(cache) if cache else default_cache())
        self._splits = splits
        self._dataset: LockedDataset | None = None
        self._rows_by_dir: dict[Path, Any] | None = None
        self._cache_root: Path | None = None

    def dataset(self, *, fetch_allowed: bool) -> LockedDataset:
        """The locked tasks; fetched first where allowed (the trainer) and needed (no lock answers the config)."""

        if self._dataset is None:
            if self.fetch is not None:
                lock = ensure_lock(self.fetch, fetch_allowed=fetch_allowed)
            else:
                lock = read_lock(self.lock_path)
            self._dataset = LockedDataset(lock, self.cache, self._splits)
        return self._dataset

    @property
    def lock(self) -> DatasetLock:
        return self.dataset(fetch_allowed=False).lock

    def names(self, split: str) -> list[str]:
        return self.dataset(fetch_allowed=True).names(split)

    def split_names(self) -> list[str]:
        return self.dataset(fetch_allowed=True).split_names()

    def holdout_rows(self) -> list[HoldoutRow]:
        """Every task of the lock, refused ones included, with its index fields and input files: what a hold-out
        splits.  A ``LockError`` for a lock written before platoon recorded input files: a hold-out keeps the tasks
        that share one on one side, and without them it would not know which do."""

        lock = self.dataset(fetch_allowed=True).lock
        if not lock.records_input_files:
            missing = sum(1 for task in lock.tasks if task.input_files is None)
            raise LockError(
                f"The lock {self.lock_path} records no input files for {missing:,} of its {len(lock.tasks):,} "
                "task(s), as locks written before platoon recorded them do, and a hold-out keeps the tasks that share "
                "an input file on one side: fetch the dataset again to record them (`platoon harbor fetch CONFIG`)"
            )
        return [HoldoutRow(task.name, dict(task.index), task.refused, task.input_files) for task in lock.tasks]

    def task_dir(self, task_id: str) -> Path:
        """The cache directory of ``task_id``, checked against its digest (once per node)."""

        return self.dataset(fetch_allowed=False).task_dir(task_id)

    def index(self, task_id: str) -> dict[str, Any]:
        """The task's index fields recorded in the lock (``source_dataset``, ...): ``HarborTask.index``, which
        reward-contract hooks read."""

        return dict(self.dataset(fetch_allowed=False).row(task_id).index)

    def task_misc(self, task_id: str) -> dict[str, Any]:
        """What a task's ``misc["harbor"]`` records about it from the lock: digest, index, image key."""

        row = self.dataset(fetch_allowed=False).row(task_id)
        misc: dict[str, Any] = {"digest": row.digest, "index": dict(row.index)}
        if row.image.get("key"):
            misc["image_key"] = row.image["key"]
        return misc

    def row_for(self, task_dir: str | Path):
        """The lock row of a cached task directory, or None for a directory outside the lock."""

        dataset = self.dataset(fetch_allowed=False)
        if self._rows_by_dir is None or self._cache_root is None:
            self._rows_by_dir = {Path(row.path): row for row in dataset.lock.tasks}
            self._cache_root = dataset.cache.resolve()
        directory = Path(task_dir).resolve()
        try:
            relative = directory.relative_to(self._cache_root)
        except ValueError:
            return None
        return self._rows_by_dir.get(relative)


class TaskImages:
    """Session images: the image map's entry for a task's build key, and its deferred steps.

    The build plan comes from the task's lock row when it has one, else from its ``environment/Dockerfile``,
    planned when its first session opens (only with an image map the entry names: without one, tasks outside a lock
    keep running on their ``docker_image`` or the service's ``default_image``).

    The map is the one the entry names (``image_map``), else ``default``, the default image map of a lock-backed
    entry (:func:`~platoon.envs.harbor.images.default_image_map`), when that file exists; a missing default file means
    no map.  It is read when the first session opens.
    """

    def __init__(
        self,
        image_map: str | Path | Mapping[str, Any] | None,
        locked: LockedTasks | None,
        *,
        default: Path | None = None,
    ) -> None:
        self._source = image_map
        self._locked = locked
        self._default = default if image_map is None and locked is not None else None

    @property
    def path(self) -> Path | None:
        """The image map file: the one the entry names, else the default map (which may not exist yet); None for an
        inline map, or none at all."""

        if isinstance(self._source, (str, Path)):
            return Path(self._source)
        return self._default if self._source is None else None

    @property
    def inline(self) -> Mapping[str, Any] | None:
        """The image map the entry gives inline, as a mapping; None for a map in a file, or none."""

        return self._source if isinstance(self._source, Mapping) else None

    @property
    def is_default(self) -> bool:
        """Whether the map is the entry's default image map, not one it names."""

        return self._default is not None

    def load(self) -> ImageMap | None:
        """The image map as it is now (``ImageMapError`` when the entry's own map cannot be read); None without one,
        or when the default map's file does not exist yet."""

        if self._source is not None:
            return ImageMap.load(self._source)
        if self._default is not None and self._default.is_file():
            return ImageMap.load(self._default)
        return None

    @cached_property
    def image_map(self) -> ImageMap | None:
        """The image map sessions run, read when the first one opens (:meth:`load`)."""

        return self.load()

    def for_task(self, task_dir: Path, *, docker_image: str | None, default_image: str | None) -> SessionImage | None:
        plan = None
        row = self._locked.row_for(task_dir) if self._locked is not None else None
        if row is not None:
            plan = dict(row.image)
        elif self._source is not None and not docker_image:
            plan = _plan_at_open(task_dir)
        if plan is None:
            return None
        return session_image(
            task_dir, plan=plan, image_map=self.image_map, docker_image=docker_image, default_image=default_image
        )

    def for_verifier(self, task_dir: Path, *, docker_image: str | None = None) -> SessionImage | None:
        """The image of a separate verifier built from ``tests/Dockerfile`` (the lock row's ``verifier_image``,
        else planned now), with its deferred steps read from ``tests/``; None for a verifier without one (it runs
        ``[verifier.environment].docker_image``, else the task's own image: :meth:`for_task`)."""

        plan = None
        row = self._locked.row_for(task_dir) if self._locked is not None else None
        if row is not None:
            plan = dict(row.verifier_image) if row.verifier_image else None
        elif not docker_image:
            plan = _plan_at_open(task_dir, "tests")
        if plan is None or not plan.get("key"):
            return None
        return session_image(task_dir, plan=plan, image_map=self.image_map, context="tests")


def _plan_at_open(task_dir: Path, context: str = "environment") -> dict[str, Any] | None:
    task_dir = Path(task_dir).resolve()
    with _PLANS_LOCK:
        if (task_dir, context) in _PLANS:
            return _PLANS[(task_dir, context)]
    dockerfile = task_dir / context / "Dockerfile"
    plan: dict[str, Any] | None = None
    if dockerfile.is_file():
        try:
            plan = {"kind": "per_task", **plan_dockerfile(dockerfile, task_dir / context).to_json()}
        except (DockerfileError, OSError) as exc:
            raise LockError(f"{dockerfile}: {exc}") from exc
    with _PLANS_LOCK:
        _PLANS[(task_dir, context)] = plan
    return plan
