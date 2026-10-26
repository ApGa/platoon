"""Dataset locks: the tasks a fetch resolved, pinned by content hash, which rollout workers read offline.

``platoon harbor fetch`` (or ``platoon train``/``eval`` for a pinned reference) writes a lock next to a
content-addressed cache. Each locked task records its id, its Harbor content digest, where its files are in the
cache, the dataset index's metadata, the flags platoon reads from ``task.toml``, its build plan, its compose file's
stack plan (:mod:`platoon.envs.harbor.compose`), why platoon refuses it, if it does, and the digest of each of its
input files (a text file's that of its text, line endings normalized), by which a hold-out keeps the tasks that share
data together. Workers look a task up by id in the lock,
never by scanning a directory, and check each cache entry's digest once per node before its first session.

Field names follow Harbor's ``TaskLock`` where they overlap (``name``, ``digest``).
"""

from __future__ import annotations

import contextlib
import fcntl
import functools
import json
import logging
import os
from collections.abc import Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from platoon.envs.harbor.compose import COMPOSE_RULE
from platoon.envs.harbor.hashing import HASH_RULE, content_hash
from platoon.site import cache_root
from platoon.utils.scratch import node_scratch

logger = logging.getLogger(__name__)

LOCK_VERSION = 1
TASKS_DIR = "tasks"


class LockError(ValueError):
    """A lock that cannot be read, a task it does not list, or a cache entry that does not match it."""


@dataclass(frozen=True)
class LockedTask:
    # The id: the registry's name, else the task index's key, else the task's path relative to the dataset (its
    # directory name directly below it), as dataset_dir names tasks.
    name: str
    digest: str  # "sha256:<Harbor content hash>" of the cached files
    path: str  # the task's directory, relative to the cache root
    source_path: str = ""  # where the dataset keeps it
    upstream_digest: str | None = None  # the digest before overlays, when overlays changed the task
    index: Mapping[str, Any] = field(default_factory=dict)  # metadata from the dataset's own index files
    flags: Mapping[str, Any] = field(default_factory=dict)  # verifier mode, stack, network, gpus, users, ...
    image: Mapping[str, Any] = field(default_factory=dict)  # build plan of environment/ (kind, key, deferred)
    verifier_image: Mapping[str, Any] | None = None  # build plan of tests/Dockerfile, for separate verifiers
    overlays: tuple[str, ...] = ()
    refused: str | None = None  # why platoon cannot run it, recorded instead of skipping silently
    # Its input files, by task-relative path, with "sha256:<hex>" of each one's digest (platoon.envs.harbor.inputs:
    # a text file's text, line endings normalized): what a hold-out joins groups by. None in a lock written before
    # platoon recorded them.
    input_files: Mapping[str, str] | None = None
    # The stack plan of its compose file (platoon.envs.harbor.compose.ComposePlan), with overlays' edits applied;
    # None without a compose file, for a file outside the subset (the task is refused), and in older locks.
    compose: Mapping[str, Any] | None = None

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> LockedTask:
        try:
            files = data.get("input_files")
            return cls(
                name=data["name"],
                digest=data["digest"],
                path=data["path"],
                source_path=data.get("source_path", ""),
                upstream_digest=data.get("upstream_digest"),
                index=dict(data.get("index") or {}),
                flags=dict(data.get("flags") or {}),
                image=dict(data.get("image") or {}),
                verifier_image=dict(data["verifier_image"]) if data.get("verifier_image") else None,
                overlays=tuple(data.get("overlays") or ()),
                refused=data.get("refused"),
                input_files=None if files is None else {str(path): str(digest) for path, digest in files.items()},
                compose=dict(data["compose"]) if data.get("compose") else None,
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise LockError(f"malformed locked task {data!r}: {exc}") from None

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {"name": self.name, "digest": self.digest, "path": self.path}
        if self.source_path:
            data["source_path"] = self.source_path
        if self.upstream_digest:
            data["upstream_digest"] = self.upstream_digest
        data["index"] = dict(self.index)
        data["flags"] = dict(self.flags)
        data["image"] = dict(self.image)
        if self.verifier_image:
            data["verifier_image"] = dict(self.verifier_image)
        if self.overlays:
            data["overlays"] = list(self.overlays)
        data["refused"] = self.refused
        if self.input_files is not None:
            data["input_files"] = dict(self.input_files)
        if self.compose:
            data["compose"] = dict(self.compose)
        return data

    def service_builds(self) -> list[Mapping[str, Any]]:
        """The build plans of its stack's services that build their images from a context of the task (``build``,
        with its whole build key: ``context``, ``dockerfile``, ``args``, ``key``, ``base``), in the plan's order."""

        return [
            service["build"]
            for service in self._services()
            if isinstance(service.get("build"), Mapping) and service["build"].get("key")
        ]

    def service_images(self) -> list[str]:
        """The registry images its stack's services run (a compose ``image`` reference, not ``main``'s), in the
        plan's order."""

        return [service["image"] for service in self._services() if isinstance(service.get("image"), str)]

    def _services(self) -> list[Mapping[str, Any]]:
        return list(((self.compose or {}).get("services") or {}).values())


@dataclass(frozen=True)
class DatasetLock:
    dataset: str  # as written in the config (an adapter: adapter:<repo>@<commit>#<path>)
    resolved: str  # commit, revision or digest (a local source: its absolute path)
    fetched_with: str  # e.g. "huggingface_hub==1.33.0", "harbor==0.24.0 (cli)", "git 2.47.3"
    tasks: tuple[LockedTask, ...]
    hash_rule: str = HASH_RULE
    build_key_rule: str = ""
    compose_rule: str = ""  # the Harbor compose layering the stack plans follow (compose.COMPOSE_RULE)
    overlay: str | None = None  # sha256 over the overlays applied, in order
    inputs: str | None = None  # sha256 over what was asked for (source, selection, overlays, refusals)
    selection: Mapping[str, Any] = field(default_factory=dict)  # how tasks were chosen, for the record
    modes: str = ""  # how file modes were restored after the download
    created_at: str = ""
    version: int = LOCK_VERSION
    # How a Harbor Hub, registry, git or adapter source resolved (platoon.envs.harbor.sources): the package and its
    # digest, the registry file's digest and the repositories' commits, an adapter's command and arguments, ...
    source: Mapping[str, Any] = field(default_factory=dict)

    @functools.cached_property
    def _by_name(self) -> dict[str, LockedTask]:
        return {task.name: task for task in self.tasks}

    def get(self, name: str) -> LockedTask:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"The lock lists no Harbor task {name!r}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def runnable(self) -> list[LockedTask]:
        return [task for task in self.tasks if task.refused is None]

    @property
    def records_input_files(self) -> bool:
        """Whether every task records its input files, as a fetch since platoon recorded them does."""

        return all(task.input_files is not None for task in self.tasks)

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "version": self.version,
            "dataset": self.dataset,
            "resolved": self.resolved,
            "fetched_with": self.fetched_with,
        }
        if self.source:  # only sources that say how they resolved: the locks of others read as they always have
            data["source"] = dict(self.source)
        data.update(
            hash_rule=self.hash_rule,
            build_key_rule=self.build_key_rule,
            compose_rule=self.compose_rule,
            overlay=self.overlay,
            inputs=self.inputs,
            selection=dict(self.selection),
            modes=self.modes,
            created_at=self.created_at,
            tasks=[task.to_json() for task in self.tasks],
        )
        return data


# path -> (file stamp, lock): the rollouts of one process read a lock once
_READ: dict[Path, tuple[tuple[int, int, int], DatasetLock]] = {}


def read_lock(path: str | Path) -> DatasetLock:
    path = Path(path)
    try:
        info = path.stat()
    except FileNotFoundError:
        raise LockError(f"Harbor lock {path} does not exist") from None
    stamp = (info.st_ino, info.st_mtime_ns, info.st_size)  # a rewrite replaces the file: a new inode
    cached = _READ.get(path)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    lock = _parse_lock(path)
    _READ[path] = (stamp, lock)
    return lock


def _parse_lock(path: Path) -> DatasetLock:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise LockError(f"Harbor lock {path} does not exist") from None
    except json.JSONDecodeError as exc:
        raise LockError(f"Harbor lock {path} is not JSON: {exc}") from None
    if not isinstance(data, Mapping) or data.get("version") != LOCK_VERSION:
        raise LockError(f"Harbor lock {path}: expected version {LOCK_VERSION}, got {data.get('version')!r}")
    if data.get("hash_rule") != HASH_RULE:
        raise LockError(f"Harbor lock {path} uses hash rule {data.get('hash_rule')!r}; this platoon knows {HASH_RULE}")
    tasks = tuple(LockedTask.from_json(task) for task in data.get("tasks") or ())
    compose_rule = data.get("compose_rule", "")
    if compose_rule != COMPOSE_RULE and any(task.compose for task in tasks):
        raise LockError(
            f"Harbor lock {path} plans its tasks' compose files by rule {compose_rule!r}; this platoon plans them by "
            f"{COMPOSE_RULE}: fetch the dataset again"
        )
    return DatasetLock(
        dataset=data["dataset"],
        resolved=data["resolved"],
        fetched_with=data.get("fetched_with", ""),
        tasks=tasks,
        hash_rule=data["hash_rule"],
        build_key_rule=data.get("build_key_rule", ""),
        compose_rule=compose_rule,
        overlay=data.get("overlay"),
        inputs=data.get("inputs"),
        selection=dict(data.get("selection") or {}),
        modes=data.get("modes", ""),
        created_at=data.get("created_at", ""),
        source=dict(data.get("source") or {}),
    )


def write_lock(lock: DatasetLock, path: str | Path) -> None:
    """Write ``lock`` atomically, one task per line, so a lock diffs well in version control."""

    data = lock.to_json()
    tasks = data.pop("tasks")
    head = json.dumps(data, indent=1)[:-2]  # without the closing brace
    lines = ",\n".join("  " + json.dumps(task, sort_keys=False, separators=(",", ":")) for task in tasks)
    text = f'{head},\n "tasks": [\n{lines}\n ]\n}}\n' if tasks else f'{head},\n "tasks": []\n}}\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def default_cache() -> Path:
    """The Harbor cache: ``harbor`` under the cache root (:func:`platoon.site.cache_root`: the site profile's
    ``cache``, else ``~/.cache/platoon``)."""

    return cache_root() / "harbor"


class LockedDataset:
    """The tasks of a lock, with optional named splits, looked up by id: the lock-backed counterpart of
    :class:`~platoon.envs.harbor.dataset.HarborDataset` that rollout workers use."""

    def __init__(self, lock: DatasetLock, cache: str | Path, splits: Mapping[str, Sequence[str]] | None = None):
        self.lock = lock
        self.cache = Path(cache)
        self._splits = {name: list(names) for name, names in splits.items()} if splits is not None else None
        if self._splits is not None:
            named = {name for names in self._splits.values() for name in names}
            if unknown := sorted(name for name in named if name not in lock):
                raise ValueError(f"The split manifest names Harbor tasks the lock does not list: {unknown[:5]}")
            refused = sorted(name for name in named if lock.get(name).refused is not None)
            if refused:
                reasons = "; ".join(f"{name}: {lock.get(name).refused}" for name in refused[:3])
                raise ValueError(f"The split manifest names Harbor tasks platoon refuses: {reasons}")
        if not lock.runnable():
            raise ValueError(f"The lock for {lock.dataset} lists no runnable Harbor task")

    def names(self, split: str) -> list[str]:
        if self._splits is None:
            return [task.name for task in self.lock.runnable()]
        if split not in self._splits:
            raise KeyError(f"Unknown split {split!r}; the manifest has {sorted(self._splits)}")
        return list(self._splits[split])

    def split_names(self) -> list[str]:
        """The manifest's splits, in its order; none without a manifest, where every split holds every task."""

        return list(self._splits or {})

    def row(self, name: str) -> LockedTask:
        task = self.lock.get(name)
        if task.refused is not None:
            raise ValueError(f"Harbor task {name!r} is refused: {task.refused}")
        return task

    def task_dir(self, name: str) -> Path:
        """The task's cache directory, its digest checked once per node (a marker under node-local scratch)."""

        row = self.row(name)
        directory = self.cache / row.path
        check_entry(directory, row.digest)
        return directory


def check_entry(directory: Path, digest: str) -> None:
    """Raise :class:`LockError` unless ``directory`` holds the files of ``digest``; a match is remembered per node.

    Rollouts of one task often start together (a group), so one of them hashes the entry while the others wait for
    its marker instead of reading the same files at once.
    """

    marker = _marker(digest)
    if marker is not None and _marked(marker, directory):
        return
    if not directory.is_dir():
        raise LockError(f"The Harbor cache has no entry {directory} (run `platoon harbor fetch` on this cache)")
    guard = contextlib.nullcontext()
    if marker is not None:
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            guard = _flock(marker.with_name(f".{marker.name}.lock"))
        except OSError:
            logger.debug("Cannot create the verification marker directory %s", marker.parent, exc_info=True)
    with guard:
        if marker is not None and _marked(marker, directory):
            return  # another rollout on this node checked it while this one waited
        actual = "sha256:" + content_hash(directory)
        if actual != digest:
            raise LockError(f"The Harbor cache entry {directory} has digest {actual}, the lock says {digest}")
        if marker is not None:
            try:
                temporary = marker.with_name(f".{marker.name}.{os.getpid()}")
                temporary.write_text(str(directory))
                os.replace(temporary, marker)
            except OSError:
                logger.debug("Cannot write the verification marker %s", marker, exc_info=True)


def _marked(marker: Path, directory: Path) -> bool:
    try:
        return marker.read_text() == str(directory)
    except OSError:
        return False


@contextlib.contextmanager
def _flock(path: Path) -> Generator[None, None, None]:
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def verify_lock(lock: DatasetLock, cache: str | Path, *, names: Sequence[str] | None = None) -> Iterator[str]:
    """Problems with the cache entries of ``lock`` (all tasks, or ``names``): missing entries, digest mismatches."""

    cache = Path(cache)
    for task in lock.tasks if names is None else [lock.get(name) for name in names]:
        directory = cache / task.path
        if not directory.is_dir():
            yield f"{task.name}: no cache entry {directory}"
            continue
        actual = "sha256:" + content_hash(directory)
        if actual != task.digest:
            yield f"{task.name}: {directory} has digest {actual}, the lock says {task.digest}"


def _marker(digest: str) -> Path | None:
    """Where a node records that it checked a cache entry against its digest: in node-local scratch
    (:func:`platoon.utils.scratch.node_scratch`)."""

    return node_scratch() / f"platoon-harbor-{os.getuid()}" / "verified" / digest.removeprefix("sha256:")
