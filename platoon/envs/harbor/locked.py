"""The Harbor service's view of a fetched dataset: task ids from a lock, task directories from the Harbor cache, and the
images sessions run (the dataset's default image map plus each task's deferred Dockerfile steps).

A service configured with ``dataset`` (a source to fetch), ``adapter`` (an adapter that generates the tasks) or
``lock`` (a lock to read) lists and finds its tasks here instead of scanning a directory. The trainer, which lists
the tasks, fetches a pinned ``dataset`` whose lock does not exist yet; rollout workers only read the lock, and check a
task's cache entry against its digest once per node before its first session.
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
from platoon.envs.harbor.images import ImageMap, ImageMapError, SessionImage, default_image_map, session_image
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
        adapter: Mapping[str, Any] | None = None,
        registry: str | Path | None = None,
        lock: str | Path | None = None,
        tasks: str | Path | Sequence[str | Path] = (),
        select: Mapping[str, Any] | None = None,
        overlays: str | Path | Sequence[str | Path] = (),
        refuse: Mapping[str, Any] | None = None,
        index: Sequence[str | Mapping[str, Any]] = (),
        splits: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        if dataset is None and adapter is None and lock is None:
            raise ValueError("LockedTasks needs a dataset, an adapter or a lock")
        self.fetch: FetchConfig | None = (
            fetch_config(
                dataset,
                adapter=adapter,
                registry=registry,
                lock=lock,
                tasks=tasks,
                select=select,
                overlays=overlays,
                refuse=refuse,
                index=index,
            )
            if dataset is not None or adapter is not None
            else None
        )
        if self.fetch is None and (tasks or select or overlays or refuse or index or registry):
            raise ValueError(
                "tasks, select, overlays, refuse, index and registry choose what a fetch of `dataset` locks"
            )
        if self.fetch is not None:
            self.lock_path = self.fetch.lock_path()
        else:
            assert lock is not None  # checked above: a dataset or a lock
            self.lock_path = Path(lock).expanduser()
        self.cache = self.fetch.cache if self.fetch is not None else default_cache()
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

    def compose(self, task_id: str) -> dict[str, Any] | None:
        """The task's stack plan as the lock records it (``LockedTask.compose``, overlays' edits applied), None for a
        task without a compose file: what its sessions run, never planned again from its files."""

        compose = self.dataset(fetch_allowed=False).row(task_id).compose
        return dict(compose) if compose else None

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

    The build plan comes from the task's lock row.  The map is the fetched dataset's default image map,
    ``images/<provider>.json`` in its Harbor cache (:func:`~platoon.envs.harbor.images.default_image_map`), when that
    file exists; a missing file means no map.  It is read when the first session opens.  A dataset directory
    (``locked`` None) has no image map: its tasks run their ``docker_image`` or the service's ``default_image``.
    """

    def __init__(self, locked: LockedTasks | None, provider: str) -> None:
        self._locked = locked
        self._path = default_image_map(locked.cache, provider) if locked is not None else None

    @property
    def path(self) -> Path | None:
        """The image map file, which may not exist yet; None for a dataset directory."""

        return self._path

    def load(self) -> ImageMap | None:
        """The image map as it is now (``ImageMapError`` when its file cannot be read); None for a dataset directory,
        or when the file does not exist yet."""

        if self._path is not None and self._path.is_file():
            return ImageMap.load(self._path)
        return None

    @cached_property
    def image_map(self) -> ImageMap | None:
        """The image map sessions run, read when the first one opens (:meth:`load`)."""

        return self.load()

    def for_task(self, task_dir: Path, *, docker_image: str | None, default_image: str | None) -> SessionImage | None:
        row = self._locked.row_for(task_dir) if self._locked is not None else None
        if row is None:
            return None
        return session_image(
            task_dir,
            plan=dict(row.image),
            image_map=self.image_map,
            docker_image=docker_image,
            default_image=default_image,
        )

    def for_service(
        self, task_dir: Path, service: str, build: Mapping[str, Any], *, task: str | None = None
    ) -> SessionImage:
        """The image of a stack's service that builds from a context of the task (its plan's ``build``): the image
        map's image for the context's whole build key, with what its Dockerfile sets over its base (``config``), and
        nothing deferred, since a service's command starts with its sandbox.  An ``ImageMapError`` when the map has
        none, as for a dataset directory's task, whose plan has no key; ``task`` is the task's name, which errors
        give (else its directory's)."""

        task_dir = Path(task_dir)
        name = task or task_dir.name
        context = f"{build.get('context', 'environment')}/{build.get('dockerfile', 'Dockerfile')}"
        key = build.get("key")
        if not key:
            raise ImageMapError(
                f"Harbor task {name!r}: the service {service} builds its image from {context}, and a dataset "
                "directory has no image map: serve the directory as a dataset to fetch (service_kwargs.dataset), "
                "whose image map can hold the image of its build key"
            )
        image_map = self.image_map
        entry = image_map.get(key) if image_map is not None else None
        if entry is None:
            missing = f"which the image map {self._path} lacks" if image_map is not None else "and no image map exists"
            raise ImageMapError(
                f"Harbor task {name!r}: the service {service} builds its image from {context} (build key {key}), "
                f"{missing}: platoon prepare builds it (or platoon harbor images build)"
            )
        config = build.get("config") or {}
        entrypoint = config.get("entrypoint", entry.entrypoint)
        cmd = config.get("cmd", entry.cmd)
        return SessionImage(
            image=entry.image,
            env=dict(entry.env),
            workdir=entry.workdir,
            user=config.get("user", entry.user),
            key=key,
            entrypoint=tuple(entrypoint) if entrypoint is not None else None,
            cmd=tuple(cmd) if cmd is not None else None,
            healthcheck=config.get("healthcheck", entry.healthcheck),
        )

    def for_verifier(self, task_dir: Path, *, docker_image: str | None = None) -> SessionImage | None:
        """The image of a separate verifier built from ``tests/Dockerfile`` (the lock row's ``verifier_image``,
        else planned now), with its deferred steps read from ``tests/``; None for a verifier without one (it runs
        ``[verifier.environment].docker_image``, else the task's own image: :meth:`for_task`).  An ``ImageMapError``
        for one that a dataset directory's task builds, which no image map has."""

        plan = None
        row = self._locked.row_for(task_dir) if self._locked is not None else None
        if row is not None:
            plan = dict(row.verifier_image) if row.verifier_image else None
        elif not docker_image:
            plan = _plan_at_open(task_dir, "tests")
        if plan is None or not plan.get("key"):
            return None
        if self._path is None:
            raise ImageMapError(
                f"Harbor task {Path(task_dir).name} builds from tests/Dockerfile (build key {plan['key']}) for its "
                "verifier, and a dataset directory has no image map: serve the directory as a dataset to fetch "
                "(service_kwargs.dataset), whose images platoon prepare builds into its image map"
            )
        return session_image(task_dir, plan=plan, image_map=self.image_map, context="tests")


def _plan_at_open(task_dir: Path, context: str) -> dict[str, Any] | None:
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
