"""Fetch Harbor datasets by name into a content-addressed cache, and pin them with a lock.

    dataset: hf://org/harbor-dataset@f216b86ca1a18d44433d6cb45f70bec81f47c66e   # or a local directory, harbor://, git+
    lock: locks/harbor-dataset-phase0.lock.json  # default: <cache>/locks/<hash of these settings>.lock.json
    tasks: [splits/phase0-smoldata.json]         # allowlists of task ids: only these are downloaded
    select: {source_dataset: [FineEnvs/SmolDataEnvs-harbor-train]}   # filters on the dataset index's fields
    overlays: [overlays/smoldata-run-before-copy]
    refuse: {sidecars: false, gpus: true, agent_user: true, privileges: true, per_task_build: false,
             uploads_over_mb: 100}
    index: [{path: data/tasks.jsonl, fields: [split_group]}, {metadata: [kaggle_dataset_name]}]

These are keys of a Harbor entry's ``service_kwargs`` (:func:`fetch_config`). The cache is the Harbor cache,
``harbor`` in the site profile's cache (:func:`platoon.envs.harbor.lock.default_cache`). The Harbor Hub
(``harbor://org/name@ref``), Harbor's legacy registry (``harbor://name@version``, with ``registry``), git repositories
(``git+<url>@<commit>#<path>``) and adapters (``adapter: {repo, commit, path, ...}`` instead of ``dataset``) are
:mod:`platoon.envs.harbor.sources`.

A fetch resolves the source at its pinned revision, reads the dataset's index (``registry.json``, else
``task-index.json``, else a scan for ``task.toml``), downloads only the selected tasks, applies overlays, and
writes each task to ``<cache>/tasks/<content hash>/``: built in a temporary directory, then renamed, so a
concurrent fetch never sees a partial task. An entry that is there already is hashed again before it is reused,
and replaced when it no longer matches, so fetching again repairs a damaged cache. The fetch then records each
task's flags, build plan and refusal in the lock: what the ``refuse`` policy refuses, and anything that keeps
:meth:`HarborTask.load` from loading the task (``task_config``), which would fail every rollout of it.  It also
records the digest of each of the task's input files (``input_files``, :mod:`platoon.envs.harbor.inputs`: a text
file's is that of its text, line endings normalized), by which a hold-out keeps the tasks that share data on one
side; a pinned dataset whose lock predates them is fetched again by ``platoon harbor fetch``, while the lock still
answers its config for everything else.

**Index fields** come from the dataset's index (``task-index.json``), from the extra index files ``index`` names
(JSON or JSONL rows keyed by task id), and, with ``{metadata: [field, ...]}``, from each selected task's ``task.toml``
``[metadata]`` table: those are read after selection (``select`` cannot name them), only scalars are recorded, and
a field the dataset's index gives the same task with another value fails the fetch.  The lock records them with
each task, where a hold-out (:mod:`platoon.envs.harbor.holdout`) and reward-contract hooks (``task.index``) read
them.

Task ids are the registry's names, else the keys of ``task-index.json``, else (a scan) each task's path relative to
the dataset: its directory name when it sits directly in the dataset, ``group/name`` deeper, as ``dataset_dir``
names them. A local directory that is itself one task is named by its ``[task].name``, else its directory name.
Index paths must stay inside the dataset: an absolute path or one with ``..`` fails the fetch.

**Hugging Face** (``hf://org/name@revision[#subdir]``, a dataset repository): the index files come from
``hf_hub_download``; the selected tasks' files from listings of only their directories (``list_repo_tree``) and
one ``hf_hub_download`` per distinct file content into a Hugging Face cache under ``<cache>/hf``. Only the files
Harbor's content hash covers are downloaded: each task's ``.gitignore`` is read first, and what it (else Harbor's
default ignores) leaves out is skipped, as for a local directory. Each download is checked against the listing (an
LFS file's SHA-256, else its git blob id); one that does not match is downloaded once more, bypassing the local
cache, and fails the fetch if it still does not. Files with the same content (inputs that many tasks share, common
graders) are downloaded once and hard-linked into each task.
The Hub keeps no file modes, so ``*.sh`` files and files that start with ``#!`` get their executable bit back.
Moving references (a branch, no revision) resolve only with ``update=True`` (``platoon harbor fetch --update``).

**Overlays** are directories with an ``overlay.toml`` that edit task files at fetch, for the tasks they match::

    description = "Install math-verify before copying the inputs, so all SmolData tasks share one image"
    [match]                            # index fields (one value or a list), and `name` globs; all must match
    source_dataset = "FineEnvs/SmolDataEnvs-harbor-train"
    [[edit]]                           # a regular-expression replacement (Python syntax, multiline mode)
    file = "environment/Dockerfile"
    pattern = '^(COPY input/ /data/input/\\n)(RUN pip install [^\\n]*\\n)'
    replace = '\\2\\1'
    count = 1                          # replacements expected (default 1); otherwise the task is refused
    [[copy]]                           # a file of the overlay replaces (or adds) a task file
    file = "tests/test.sh"
    source = "files/test.sh"

An overlay changes the cached task, so its digest is the overlaid content's; the lock keeps the upstream digest
as ``upstream_digest`` and the overlays' own hash as ``overlay``.

**Compose files.** A task's ``environment/docker-compose.yaml`` is planned at fetch
(:mod:`platoon.envs.harbor.compose`): the lock records its kind (``flags.stack``: ``single``, ``trivial``, ``main_only``
or ``sidecars``) and its stack plan (``compose``), and refuses a file outside the subset platoon runs (``compose``,
naming the service and key); with ``refuse: {privileges: false}`` it plans what only the docker provider grants
(``privileged``, a ``cap_add`` of ``NET_ADMIN``, ...) instead, as the plan's ``privileges``.  Compose's other names
(``compose.yaml``, ...) are plain files of the task, as in Harbor: the fetch warns of them (``flags.unread_compose``).
An overlay may edit a task's compose services, and list the host variables its file may name, which a session
resolves when it opens (they never enter the lock)::

    compose_host_env = ["SCICODE_TEST_DATA"]  # ${SCICODE_TEST_DATA} in the compose file names this host variable
    [[compose]]
    service = "main"
    unset = ["cap_add"]                    # keys to remove (dotted for nested ones: "deploy.resources.reservations")
    set = {environment = {MODE = "eval"}}  # merged into the service's mappings; other values replace
    [[compose]]
    service = "judge"
    replace = {command = ["judge", "--port", "8081"]}  # keys set whole
    host_network = {listen = {8081 = "JUDGE_PORT"}, token = "JUDGE_TOKEN", bind = "JUDGE_HOST"}  # on the fallback

A stack's plan also records the references to its services that the host-network fallback could not rewrite, from a
scan of the task's files (:func:`platoon.envs.harbor.host_network.with_unrewritable_references`): under ``stacks:
host_network`` a task that holds any is refused, unless an overlay edits them away.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import random
import re
import shutil
import stat
import threading
import time
import tomllib
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from platoon.envs.harbor.compose import (
    COMPOSE_FILE,
    COMPOSE_RULE,
    ComposeEdit,
    ComposeEditError,
    ComposeError,
    ComposePlan,
    compose_path,
    plan_compose,
    read_document,
    stack_kind,
    unread_compose_files,
)
from platoon.envs.harbor.dataset import dataset_tasks
from platoon.envs.harbor.dockerfile import BUILD_KEY_RULE, DockerfileError, copy_operands, plan_dockerfile
from platoon.envs.harbor.hashing import (
    DEFAULT_IGNORES,
    GITIGNORE,
    HASH_RULE,
    TASK_DIRS,
    TASK_FILES,
    IgnoreMatcher,
    file_sha256,
    task_files,
)
from platoon.envs.harbor.host_network import with_unrewritable_references
from platoon.envs.harbor.inputs import input_digest, input_files
from platoon.envs.harbor.lock import TASKS_DIR, DatasetLock, LockedTask, LockError, default_cache, read_lock, write_lock
from platoon.envs.harbor.references import (
    _COMMIT,
    _URL,
    REGISTRY,
    AdapterSpec,
    FetchError,
    check_source,
    check_url,
    is_legacy_registry,
    legacy_registry_pinned,
    registry_has_local_tasks,
    registry_inputs,
    source_pinned,
)
from platoon.envs.harbor.task import HarborTask, HarborTaskError, _environment_settings

logger = logging.getLogger(__name__)

TASK_INDEX = "task-index.json"
OVERLAY_FILE = "overlay.toml"
MODES_RULE = "executable bit restored on *.sh files and files starting with #!"
_HF = re.compile(
    r"^hf://(?:datasets/)?(?P<repo>[A-Za-z0-9][\w.-]*/[\w.-]+)(?:@(?P<revision>[^#]+))?(?:#(?P<subdir>.+))?$"
)
# What names a source to download rather than a local directory; ``adapter:`` is an adapter's name in the lock.
_REMOTE_PREFIXES = ("hf://", "harbor://", "git+", "adapter:")
_PER_TASK_LISTING = 64  # list each task's directory below this many selected tasks per parent, else the parent
_MAX_ATTEMPTS = 10
MIB = 1 << 20  # refuse.uploads_over_mb counts MiB
# How a refusal of users to run as says what fetches the task.
_USERS_FETCHED = "refuse: {agent_user: false} fetches it, for an environment that honors or ignores its users"


@dataclass(frozen=True)
class RefusePolicy:
    """What a fetch records as refused (``refuse:``), besides what :meth:`HarborTask.load` refuses, which a fetch
    always records (``task_config``: multi-step and non-Linux tasks, a bad ``network_mode``, a stack's sidecar whose
    command platoon does not know, ...)."""

    # Compose stacks with services beyond `main`, as `stack`; with `sidecars: false`, the default since platoon 0.5,
    # they are locked as stacks, which a session runs on a provider that runs stacks (the stacks capability) and a run
    # refuses elsewhere when it lists its tasks.
    sidecars: bool = False
    gpus: bool = True  # tasks that ask for GPUs
    agent_user: bool = True  # [agent].user / [verifier].user / a USER in a deferred suffix (users to run as)
    per_task_build: bool = False  # Dockerfiles whose build prefix no other task shares
    # Compose services that ask for privileged, or for a cap_add beyond what a rootless service holds (compose); with
    # false they are planned as the stack's privileges, which only the docker provider grants.
    privileges: bool = True
    # Tasks whose sessions would upload more than this many MiB into one of their sandboxes (flags.upload_bytes);
    # None: no limit.
    uploads_over_mb: float | None = None

    _SWITCHES = ("sidecars", "gpus", "agent_user", "per_task_build", "privileges")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> RefusePolicy:
        data = dict(data or {})
        if "stack" in data:  # the design's spelling: stack: sidecars
            data["sidecars"] = data.pop("stack") in ("sidecars", True)
        known = (*cls._SWITCHES, "uploads_over_mb")
        if unknown := sorted(set(data) - set(known)):
            raise FetchError(f"refuse: unknown keys {unknown}; known: {', '.join(known)}")
        for key in cls._SWITCHES:
            if key in data and not isinstance(data[key], bool):
                raise FetchError(f"refuse.{key} must be true or false, got {data[key]!r}")
        limit = data.get("uploads_over_mb")
        if limit is not None:
            if (
                isinstance(limit, bool)
                or not isinstance(limit, int | float)
                or not (math.isfinite(limit) and limit > 0)
            ):
                raise FetchError(f"refuse.uploads_over_mb must be a positive number of MiB, got {limit!r}")
            data["uploads_over_mb"] = float(limit)
        return cls(**data)

    def inputs(self) -> dict[str, Any]:
        """What the lock's inputs digest hashes: the switches, and ``privileges`` and the size limit only when they are
        not their defaults, so that a lock written before they existed still answers a config that sets neither."""

        payload: dict[str, Any] = {key: getattr(self, key) for key in self._SWITCHES if key != "privileges"}
        if not self.privileges:
            payload["privileges"] = False
        if self.uploads_over_mb is not None:
            payload["uploads_over_mb"] = self.uploads_over_mb
        return payload


@dataclass(frozen=True)
class IndexFile:
    """An extra index file of the dataset whose rows add metadata to the lock (``index:``): a JSON object keyed by
    task id, or JSON/JSONL rows naming the task in ``key``."""

    path: str
    key: str = "task_id"
    fields: tuple[str, ...] | None = None  # keep only these fields

    @classmethod
    def parse(cls, value: str | Mapping[str, Any]) -> IndexFile:
        if isinstance(value, str):
            return cls(path=value)
        fields = value.get("fields")
        return cls(path=value["path"], key=value.get("key", "task_id"), fields=tuple(fields) if fields else None)


@dataclass(frozen=True)
class MetadataFields:
    """An ``index`` entry ``{metadata: [field, ...]}``: these fields of each selected task's ``task.toml``
    ``[metadata]`` table, recorded among its index fields.  Only scalars (strings, finite numbers, booleans) are
    recorded; a task without a field records none for it."""

    metadata: tuple[str, ...]

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> MetadataFields:
        if unknown := sorted(set(value) - {"metadata"}):
            raise FetchError(f"index: an entry with metadata takes no other key, got {unknown}")
        names = [value["metadata"]] if isinstance(value["metadata"], str) else value["metadata"]
        if not isinstance(names, list | tuple) or not names or not all(isinstance(n, str) and n for n in names):
            raise FetchError(f"index: metadata must list field names, got {value['metadata']!r}")
        return cls(metadata=tuple(names))


def index_entry(value: str | Mapping[str, Any]) -> IndexFile | MetadataFields:
    """An ``index`` item: an index file (a path, or ``{path, key, fields}``) or ``{metadata: [field, ...]}``."""

    if isinstance(value, Mapping) and "metadata" in value:
        return MetadataFields.parse(value)
    return IndexFile.parse(value)


@dataclass(frozen=True)
class FetchConfig:
    dataset: str
    cache: Path
    lock: Path | None = None
    tasks: tuple[Path, ...] = ()
    select: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    overlays: tuple[Path, ...] = ()
    refuse: RefusePolicy = RefusePolicy()
    index: tuple[IndexFile | MetadataFields, ...] = ()
    workers: int = 8
    # The legacy registry of a harbor://name@version dataset: a registry.json file or URL (None: Harbor's own).
    registry: str | None = None
    # An adapter that generates the tasks, instead of a dataset to download; ``dataset`` is then its name for messages
    # and the lock (adapter:<repo>@<commit>#<path>).
    adapter: AdapterSpec | None = None

    @property
    def local(self) -> bool:
        """Whether the fetch copies its tasks from local directories, which it does again every time, since their files
        may change under the same paths: a local dataset directory, or a registry file's tasks that name no
        ``git_url`` (directories next to the file)."""

        if self.adapter is not None:
            return False
        if self.registry is not None and is_legacy_registry(self.dataset):
            return registry_has_local_tasks(self.dataset, self.registry)
        return is_local(self.dataset)

    @property
    def pinned(self) -> bool:
        """Whether the source names immutable content, which ``platoon train`` and ``platoon eval`` fetch by
        themselves (:func:`is_pinned`; an adapter at a full commit; a legacy registry's dataset by its version and
        its registry, a file or a URL with a commit).  The one place that decides it: a fetch refuses a source that
        is not pinned unless it updates."""

        if self.adapter is not None:
            return self.adapter.pinned
        if is_legacy_registry(self.dataset):
            return legacy_registry_pinned(self.dataset, self.registry)
        return is_pinned(self.dataset)

    @property
    def immutable(self) -> bool:
        """Whether the files the source names cannot change: pinned, and not copied from local directories."""

        return self.pinned and not self.local

    @property
    def metadata_fields(self) -> tuple[str, ...]:
        """The ``task.toml`` ``[metadata]`` fields the ``index`` entries name, in order, each once."""

        return tuple(
            dict.fromkeys(name for entry in self.index if isinstance(entry, MetadataFields) for name in entry.metadata)
        )

    def lock_path(self) -> Path:
        return self.lock if self.lock is not None else self.cache / "locks" / f"{self.inputs_digest()[:16]}.lock.json"

    def earlier_lock_path(self) -> Path | None:
        """Where the lock is that platoon wrote for this config before ``refuse.sidecars`` defaulted to false
        (:meth:`earlier_inputs_digest`): ``lock``, else the cache's lock named by that digest; None for a config that
        refuses stacks."""

        earlier = self.earlier_inputs_digest()
        if earlier is None:
            return None
        return self.lock if self.lock is not None else self.cache / "locks" / f"{earlier[:16]}.lock.json"

    def inputs_digest(self) -> str:
        """SHA-256 over what the fetch is asked for, so a lock can tell whether it answers this config."""

        return self._inputs_digest(self.refuse)

    def earlier_inputs_digest(self) -> str | None:
        """The inputs digest that this config, which refuses no stacks, had when it set no ``refuse.sidecars`` before
        that defaulted to false: its digest with ``sidecars: true``.  A lock that records it still answers the config
        where it refused no task as a stack (:func:`answers`).  None for a config that refuses stacks, whose digest has
        not changed."""

        return None if self.refuse.sidecars else self._inputs_digest(replace(self.refuse, sidecars=True))

    def _inputs_digest(self, refuse: RefusePolicy) -> str:
        payload = {
            "dataset": self.dataset,
            "tasks": [_sha256_file(path) for path in self.tasks],
            "select": {key: list(values) for key, values in sorted(self.select.items())},
            "overlays": [Overlay.load(path).digest for path in self.overlays],
            "refuse": refuse.inputs(),
            "index": [index.__dict__ for index in self.index],
        }
        # Only when set, so that a lock written before these keys still answers a config that sets neither.
        if self.registry is not None:
            payload["registry"] = registry_inputs(self.registry)
        if self.adapter is not None:
            payload["adapter"] = self.adapter.inputs()
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=list).encode()).hexdigest()


def fetch_config(
    dataset: str | None = None,
    *,
    adapter: Mapping[str, Any] | None = None,
    registry: str | Path | None = None,
    cache: str | Path | None = None,
    lock: str | Path | None = None,
    tasks: str | Path | Sequence[str | Path] = (),
    select: Mapping[str, Any] | None = None,
    overlays: str | Path | Sequence[str | Path] = (),
    refuse: Mapping[str, Any] | None = None,
    index: Sequence[str | Mapping[str, Any]] = (),
) -> FetchConfig:
    """A :class:`FetchConfig` from a Harbor entry's values, fetched into ``cache`` (default: the Harbor cache,
    :func:`~platoon.envs.harbor.lock.default_cache`); relative paths resolve against the working directory, and a
    relative local ``dataset`` too (a config's are absolute already).  ``adapter`` (an adapter that generates the
    tasks, :class:`~platoon.envs.harbor.references.AdapterSpec`) replaces ``dataset``; ``registry`` (a ``registry.json``
    file or URL) goes with a ``harbor://name@version`` dataset."""

    def resolve(value: str | Path) -> Path:
        path = Path(value).expanduser()
        return Path(os.path.normpath(path if path.is_absolute() else Path.cwd() / path))

    def paths(value: str | Path | Sequence[str | Path]) -> tuple[Path, ...]:
        return (resolve(value),) if isinstance(value, (str, Path)) else tuple(resolve(item) for item in value)

    spec = None
    if adapter is not None:
        if dataset is not None:
            raise FetchError("Name the dataset or the adapter that generates it, not both")
        spec = AdapterSpec.parse(adapter, resolve=lambda value: str(resolve(value)))
        dataset = spec.dataset
    elif not isinstance(dataset, str) or not dataset:
        raise FetchError(
            "dataset must name a source: hf://org/name@revision, harbor://org/name@ref, harbor://name@version, "
            "git+<url>@<commit>#<path> or a local directory (or name an adapter instead)"
        )
    elif not is_local(dataset):
        check_source(dataset)
    else:
        dataset = str(resolve(dataset))
    if registry is not None:
        if adapter is not None or not is_legacy_registry(dataset):
            raise FetchError(
                f"registry names the legacy registry of a harbor://name@version dataset, and {dataset} is none"
            )
        if _URL.match(str(registry)):
            check_url(str(registry), "registry")
            registry = str(registry)
        else:
            registry = str(resolve(registry))
    selection: dict[str, tuple[Any, ...]] = {}
    for key, values in (select or {}).items():
        name = key.removeprefix("index.")
        if name.startswith("flags."):
            raise FetchError(f"select: {key} filters on flags, which a fetch computes; select on index fields")
        selection[name] = tuple(values) if isinstance(values, (list, tuple)) else (values,)
    return FetchConfig(
        dataset=dataset,
        cache=resolve(cache) if cache is not None else default_cache(),
        lock=resolve(lock) if lock is not None else None,
        tasks=paths(tasks),
        select=selection,
        overlays=paths(overlays),
        refuse=RefusePolicy.from_mapping(refuse),
        index=tuple(index_entry(item) for item in index),
        registry=registry,
        adapter=spec,
    )


def is_local(dataset: str) -> bool:
    """Whether ``dataset`` names a local directory rather than a source to download (``hf://``, ``harbor://``,
    ``git+``, or an adapter's ``adapter:`` name)."""

    return not dataset.startswith(_REMOTE_PREFIXES)


def is_pinned(dataset: str) -> bool:
    """Whether ``dataset`` names immutable content: an hf:// commit, a Hub package's revision or digest, a git
    commit, an adapter at a commit, or a local directory.  A dataset of the legacy registry is pinned only by its
    registry (:attr:`FetchConfig.pinned`), so it counts as moving here."""

    if dataset.startswith("hf://"):
        match = _HF.match(dataset)
        return bool(match and match.group("revision") and _COMMIT.match(match.group("revision")))
    return is_local(dataset) or source_pinned(dataset)


def answers(lock: DatasetLock, config: FetchConfig) -> bool:
    """Whether ``lock`` was written for ``config``'s fetch settings: by its inputs digest, or, for a lock that platoon
    wrote before ``refuse.sidecars`` defaulted to false, by the digest it had then
    (:meth:`FetchConfig.earlier_inputs_digest`), where it refused no task as a stack (``stack``): the new default
    changes nothing in it."""

    if lock.inputs == config.inputs_digest():
        return True
    earlier = config.earlier_inputs_digest()
    return earlier is not None and lock.inputs == earlier and not any(_refused_as_stack(task) for task in lock.tasks)


def _refused_as_stack(task: LockedTask) -> bool:
    return task.refused is not None and task.refused.split(":")[0] == "stack"


def answering_lock(config: FetchConfig) -> DatasetLock | None:
    """The lock that answers ``config`` (:func:`answers`), else None: the one at ``config.lock_path()``, else one
    written before ``refuse.sidecars`` defaulted to false (:meth:`FetchConfig.earlier_lock_path`)."""

    located = _answering(config)
    return located[1] if located is not None else None


def answering_lock_path(config: FetchConfig) -> Path:
    """Where the lock that answers ``config`` is (:func:`answering_lock`), else where a fetch of it writes its lock
    (:meth:`FetchConfig.lock_path`)."""

    located = _answering(config)
    return located[0] if located is not None else config.lock_path()


def _answering(config: FetchConfig) -> tuple[Path, DatasetLock] | None:
    for path in dict.fromkeys(path for path in (config.lock_path(), config.earlier_lock_path()) if path is not None):
        existing = _read_if_present(path)
        if existing is not None and answers(existing, config):
            return path, existing
    return None


def current_lock(config: FetchConfig) -> DatasetLock | None:
    """The lock that answers ``config`` when it records its tasks' input files and the cache holds a directory for
    each of its tasks: a fetch would change nothing.  None when one is needed, as for a lock written before platoon
    recorded input files.  The entries are not hashed; ``platoon harbor verify`` does that."""

    existing = answering_lock(config)
    if existing is None or not existing.records_input_files:
        return None
    if any(not (config.cache / task.path).is_dir() for task in existing.tasks):
        return None
    return existing


def ensure_lock(config: FetchConfig, *, fetch_allowed: bool, log: Callable[[str], None] | None = None) -> DatasetLock:
    """The lock for ``config``: read when it answers this config, else fetched (pinned references only, and only
    where ``fetch_allowed``: the trainer, never a rollout worker)."""

    path = config.lock_path()
    located = _answering(config)
    if located is not None:
        found, existing = located
        missing = (
            [task.name for task in existing.tasks if not (config.cache / task.path).is_dir()] if fetch_allowed else []
        )
        if not missing or not config.pinned:
            return existing
        logger.warning(
            "The cache %s lacks %d task(s) of the lock %s; fetching them again", config.cache, len(missing), found
        )
    why = _why_not(config, path)
    if not fetch_allowed:
        raise LockError(
            f"The Harbor lock {path} {why}: run `platoon prepare` on the run's config (or `platoon harbor fetch`), or "
            "start `platoon train` or `platoon eval`, which fetch pinned references, before rollouts of "
            f"{config.dataset}"
        )
    if not config.pinned:
        raise LockError(
            f"{config.dataset} is a moving reference and the lock {path} {why}: pin it with "
            "`platoon harbor fetch CONFIG --update`, or name a commit (a revision or digest on the Harbor Hub)"
        )
    with _flock(fetch_lock_file(config)):
        existing = answering_lock(config)  # another process may have fetched while this one waited
        if existing is not None and all((config.cache / task.path).is_dir() for task in existing.tasks):
            return existing
        (log or logger.info)(f"Fetching {config.dataset} into {config.cache} (lock {path})")
        return fetch(config, log=log)


def _why_not(config: FetchConfig, path: Path) -> str:
    """Why no lock answers ``config``, the lock at ``path`` being where a fetch writes it; naming a lock written for it
    before ``refuse.sidecars`` defaulted to false that refused stacks, which a fetch now locks as runnable."""

    why = "does not exist" if _read_if_present(path) is None else "was written for other fetch settings"
    earlier = config.earlier_lock_path()
    old = _read_if_present(earlier) if earlier is not None else None
    stacks = sum(1 for task in old.tasks if _refused_as_stack(task)) if old is not None else 0
    if stacks and old is not None and old.inputs == config.earlier_inputs_digest():
        why += (
            f" (the lock {earlier}, which platoon wrote for these settings before refuse.sidecars defaulted to false, "
            f"refused {stacks} task(s) as stacks, which a fetch now locks as runnable; refuse: {{sidecars: true}} "
            "keeps them refused)"
        )
    return why


def fetch_lock_file(config: FetchConfig) -> Path:
    """The file lock on which the fetches of ``config``'s lock take turns, in its cache: processes that would write one
    lock, on one node or on several that share the cache's filesystem, fetch it one at a time."""

    digest = hashlib.sha256(str(config.lock_path()).encode()).hexdigest()[:16]
    return config.cache / ".locks" / f"fetch-{digest}.lock"


def fetch_if_needed(
    config: FetchConfig, *, update: bool = False, repair: bool = False, log: Callable[[str], None] | None = None
) -> tuple[DatasetLock, bool]:
    """The lock for ``config``, and whether this call fetched it.

    A dataset pinned to a commit, revision or digest whose lock is current (:func:`current_lock`: it answers the
    settings, records input files, and the cache holds its tasks) is left alone, unless ``update`` or ``repair``;
    anything else is fetched, a local directory every time, since its files may have changed under the same path.
    The fetch takes turns with the other fetches of the same lock (:func:`fetch_lock_file`), so of two at once, on
    one node or two, the second finds the first's lock current and fetches nothing.
    """

    def current() -> DatasetLock | None:
        return current_lock(config) if config.immutable and not (update or repair) else None

    if (lock := current()) is not None:
        return lock, False
    with _flock(fetch_lock_file(config)):
        if (lock := current()) is not None:  # fetched by another process while this one waited
            return lock, False
        return fetch(config, update=update, log=log), True


def fetch(config: FetchConfig, *, update: bool = False, log: Callable[[str], None] | None = None) -> DatasetLock:
    """Resolve, download, overlay, hash, plan and lock ``config``'s tasks; write and return the lock."""

    say = log or logger.info
    started = time.monotonic()
    overlays = [Overlay.load(path) for path in config.overlays]  # before the source, which may take long to open
    source = _open_source(config, update=update, log=say)
    try:
        candidates = source.candidates()
        for index_file in config.index:
            if isinstance(index_file, IndexFile):
                _merge_index(candidates, source.read_json_rows(index_file), index_file)
        selected = _select(candidates, config)
        say(f"{len(selected)} of {len(candidates)} tasks selected from {config.dataset}")
        hashes = _HashMemo()  # what checking the downloads hashed, caching the tasks does not hash again
        files = source.materialize(selected, hashes)
        tasks: list[LockedTask] = []
        (config.cache / TASKS_DIR).mkdir(parents=True, exist_ok=True)
        for count, candidate in enumerate(selected, 1):
            tasks.append(_cache_task(candidate, files[candidate.name], config, overlays, hashes))
            if count % 500 == 0:
                say(f"cached {count}/{len(selected)} tasks")
    finally:
        source.close()  # its downloads and checkouts: what the cache needs is in it now
    tasks = _assign_image_kinds(tasks, config.refuse)
    lock = DatasetLock(
        dataset=config.dataset,
        resolved=source.resolved,
        fetched_with=source.fetched_with,
        tasks=tuple(sorted(tasks, key=lambda task: task.name)),
        hash_rule=HASH_RULE,
        build_key_rule=BUILD_KEY_RULE,
        compose_rule=COMPOSE_RULE,
        overlay=_overlays_digest(overlays),
        inputs=config.inputs_digest(),
        selection={
            "tasks": {str(path): _sha256_file(path) for path in config.tasks},
            "select": {key: list(values) for key, values in config.select.items()},
            "selected": len(selected),
            "of": len(candidates),
        },
        modes=MODES_RULE,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source=dict(source.details),
    )
    write_lock(lock, config.lock_path())
    unread = [task for task in lock.tasks if task.flags.get("unread_compose")]
    if unread:
        names = sorted({name for task in unread for name in task.flags["unread_compose"]})
        logger.warning(
            "%d Harbor task(s) of %s hold %s, which Harbor does not read (it reads only environment/%s), such as %s: "
            "platoon treats them as plain files of the task and runs each task without them, as Harbor does",
            len(unread),
            config.dataset,
            " or ".join(names),
            COMPOSE_FILE,
            unread[0].name,
        )
    refused = Counter(task.refused.split(":")[0] for task in lock.tasks if task.refused)
    say(
        f"Locked {len(lock.tasks)} tasks ({len(lock.runnable())} runnable"
        + (f"; refused: {dict(refused)}" if refused else "")
        + f") in {config.lock_path()} after {time.monotonic() - started:.0f}s"
    )
    return lock


# --- candidates and selection ------------------------------------------------------------------------------


@dataclass
class Candidate:
    name: str
    path: str  # relative to the dataset root (where a registry's task lives: its repository, commit and path)
    index: dict[str, Any] = field(default_factory=dict)
    # The content digest the source gives the task ("sha256:<hex>", the Harbor Hub's), which its files must hash to.
    digest: str | None = None


def _select(candidates: Mapping[str, Candidate], config: FetchConfig) -> list[Candidate]:
    chosen = list(candidates.values())
    if config.tasks:
        allowed: set[str] = set()
        for path in config.tasks:
            allowed |= _allowlist(path)
        if unknown := sorted(allowed - set(candidates)):
            raise FetchError(f"tasks: {len(unknown)} ids are not in {config.dataset}, e.g. {unknown[:3]}")
        chosen = [candidate for candidate in chosen if candidate.name in allowed]
    for key, values in config.select.items():
        if key == "name":
            chosen = [candidate for candidate in chosen if any(fnmatchcase(candidate.name, str(v)) for v in values)]
            continue
        if not any(key in candidate.index for candidate in candidates.values()):
            hint = (
                "; fields from task.toml [metadata] are read after selection, so select cannot name them"
                if key in config.metadata_fields
                else ""
            )
            raise FetchError(f"select: no task of {config.dataset} has the index field {key!r}{hint}")
        chosen = [candidate for candidate in chosen if candidate.index.get(key) in values]
    if not chosen:
        raise FetchError(f"No task of {config.dataset} matches the selection")
    return sorted(chosen, key=lambda candidate: candidate.name)


def _allowlist(path: Path) -> set[str]:
    """Task ids from an allowlist file: a JSON list, a JSON split manifest (``{split: [ids]}``), or lines."""

    try:
        text = path.read_text()
    except OSError as exc:
        raise FetchError(f"tasks: cannot read {path}: {exc}") from exc
    if path.suffix == ".json":
        data = json.loads(text)
        if isinstance(data, Mapping):
            return {str(name) for names in data.values() for name in names}
        return {str(name) for name in data}
    if path.suffix == ".toml":
        return {str(name) for names in tomllib.loads(text).values() for name in names}
    return {line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")}


def _candidates_from_index(registry: Any, task_index: Any, label: str) -> dict[str, Candidate]:
    candidates: dict[str, Candidate] = {}
    if registry is not None:
        datasets = registry if isinstance(registry, list) else [registry]
        for dataset in datasets:
            for entry in dataset.get("tasks") or []:
                if entry.get("git_url"):
                    raise FetchError(f"{label}: registry task {entry.get('name')!r} lives in a git repository")
                name = entry["name"]
                path = _index_path(entry["path"], name, f"{label}: {REGISTRY}")
                if name in candidates:
                    raise FetchError(f"{label}: two registry tasks are named {name!r}")
                candidates[name] = Candidate(name=name, path=path)
    if isinstance(task_index, Mapping):
        for name, row in task_index.items():
            if not isinstance(row, Mapping):
                continue
            fields = {key: value for key, value in row.items() if key != "task_path"}
            if name in candidates:
                candidates[name].index.update(fields)
            elif registry is None and row.get("task_path"):
                path = _index_path(row["task_path"], name, f"{label}: {TASK_INDEX}")
                candidates[name] = Candidate(name=name, path=path, index=dict(fields))
    return candidates


def _index_path(path: Any, name: str, where: str) -> str:
    """A task's directory as a dataset index gives it, relative to the dataset root. An absolute path or one with
    ``..`` would leave the dataset: ``FetchError``, as ``HarborDataset`` refuses them (``read_registry``)."""

    posix = PurePosixPath(str(path))
    if posix.is_absolute() or ".." in posix.parts:
        raise FetchError(f"{where}: task {name!r} has a path outside the dataset: {path!r}")
    return str(path).strip("/")


def _merge_index(candidates: Mapping[str, Candidate], rows: Iterator[tuple[str, Mapping[str, Any]]], spec: IndexFile):
    for name, row in rows:
        candidate = candidates.get(name)
        if candidate is None:
            continue
        candidate.index.update({key: value for key, value in row.items() if spec.fields is None or key in spec.fields})


# --- sources ---------------------------------------------------------------------------------------------------


@dataclass
class _TaskFiles:
    """A task's files before caching: task-relative path -> a local file, and whether it may be hard-linked."""

    files: dict[str, Path]
    link: bool  # hard-link (content-addressed downloads) or copy (a user's directory)


class _Source:
    resolved: str
    fetched_with: str
    details: Mapping[str, Any] = MappingProxyType({})  # how the source resolved, for the lock (DatasetLock.source)

    def candidates(self) -> dict[str, Candidate]:
        raise NotImplementedError

    def read_json_rows(self, spec: IndexFile) -> Iterator[tuple[str, Mapping[str, Any]]]:
        raise NotImplementedError

    def materialize(self, selected: Sequence[Candidate], hashes: _HashMemo) -> dict[str, _TaskFiles]:
        """The selected tasks' files; ``hashes`` keeps what checking downloads hashed, for caching the tasks."""

        raise NotImplementedError

    def close(self) -> None:
        """Remove what the source downloaded or checked out for the fetch, once its tasks are cached."""


def _open_source(config: FetchConfig, *, update: bool, log: Callable[[str], None]) -> _Source:
    dataset = config.dataset
    if dataset.startswith("hf://"):
        return _HubSource(config, update=update, log=log)
    if config.adapter is not None or not is_local(dataset):
        from platoon.envs.harbor.sources import open_source  # sources builds on this module

        return open_source(config, update=update, log=log)
    return _LocalSource(Path(dataset))


def _json_rows(text: str, spec: IndexFile) -> Iterator[tuple[str, Mapping[str, Any]]]:
    if spec.path.endswith(".jsonl"):
        rows: Any = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = json.loads(text)
    if isinstance(rows, Mapping):
        for name, row in rows.items():
            if isinstance(row, Mapping):
                yield str(name), row
        return
    for row in rows:
        if isinstance(row, Mapping) and spec.key in row:
            yield str(row[spec.key]), row


class _LocalSource(_Source):
    """A directory of tasks: those its ``registry.json`` or ``task-index.json`` lists; else the directory itself when
    it holds a ``task.toml``; else every directory below it that holds one, with the ids ``dataset_dir`` gives
    (:func:`~platoon.envs.harbor.dataset.dataset_tasks`)."""

    def __init__(self, root: Path, *, link: bool = False) -> None:
        if not root.is_dir():
            raise FetchError(f"{root} is not a directory")
        self.root = root.resolve()
        self.resolved = str(self.root)
        self.fetched_with = "platoon (local directory)"
        self.link = link  # hard-link the files into the cache (a directory the fetch made), else copy them

    def candidates(self) -> dict[str, Candidate]:
        registry = _load_json(self.root / REGISTRY)
        task_index = _load_json(self.root / TASK_INDEX)
        if registry is not None or task_index is not None:
            return _candidates_from_index(registry, task_index, str(self.root))
        config = self.root / "task.toml"
        if config.is_file():  # one task, which dataset_dir does not take: named by [task].name or its directory
            name = _task_name(config) or self.root.name
            return {name: Candidate(name=name, path=".")}
        # A task directly below the directory keeps its directory's name, a deeper one its relative path.
        candidates = {name: Candidate(name=name, path=name) for name, _ in dataset_tasks(self.root)}
        if not candidates:
            raise FetchError(f"{self.root} holds no Harbor task (no task.toml)")
        return candidates

    def read_json_rows(self, spec: IndexFile) -> Iterator[tuple[str, Mapping[str, Any]]]:
        return _json_rows((self.root / spec.path).read_text(), spec)

    def materialize(self, selected: Sequence[Candidate], hashes: _HashMemo) -> dict[str, _TaskFiles]:
        out = {}
        for candidate in selected:
            task_dir = self.root / candidate.path
            if not (task_dir / "task.toml").is_file():
                raise FetchError(f"{task_dir} has no task.toml")
            files = {relative: task_dir / relative for relative in _hashed_paths(task_dir)}
            if self.link:  # a link is hard-linked as its target, which outlives the directory the fetch made
                files = {relative: linked_target(path) for relative, path in files.items()}
            out[candidate.name] = _TaskFiles(files=files, link=self.link)
        return out


def linked_target(path: Path) -> Path:
    """The file a symbolic link names, else ``path``: ``os.link`` would link the link itself (Python before 3.14, on
    Linux), and a copy of a link whose target is removed with its download or checkout would dangle."""

    return Path(os.path.realpath(path)) if os.path.islink(path) else path


class _HubSource(_Source):
    """A Hugging Face dataset repository at one commit (``dataset``, else the config's)."""

    def __init__(
        self, config: FetchConfig, *, update: bool, log: Callable[[str], None], dataset: str | None = None
    ) -> None:
        dataset = dataset or config.dataset
        match = _HF.match(dataset)
        if match is None:
            raise FetchError(f"{dataset}: expected hf://org/name@revision[#subdir]")
        from huggingface_hub import HfApi

        self.repo = match.group("repo")
        self.subdir = (match.group("subdir") or "").strip("/")
        revision = match.group("revision")
        self.api = HfApi()
        self.cache_dir = config.cache / "hf"
        self.workers = config.workers
        self.log = log
        if revision and _COMMIT.match(revision):
            self.resolved = revision
        elif not update:
            raise FetchError(
                f"{dataset} is a moving reference: pin a commit, or resolve it with `platoon harbor fetch --update`"
            )
        else:
            info = _retry(lambda: self.api.repo_info(self.repo, repo_type="dataset", revision=revision or "main"))
            self.resolved = info.sha
        self.fetched_with = f"huggingface_hub=={importlib.metadata.version('huggingface_hub')}"
        self._listing_dir = config.cache / "hf-listings" / self.repo.replace("/", "--") / self.resolved

    def _repo_path(self, relative: str) -> str:
        return f"{self.subdir}/{relative}" if self.subdir else relative

    def _download(self, repo_path: str, *, force: bool = False) -> Path:
        """The local file of ``repo_path``; ``force`` downloads it again even when the local cache has it."""

        from huggingface_hub import hf_hub_download

        def call() -> str:
            return hf_hub_download(
                self.repo,
                repo_path,
                repo_type="dataset",
                revision=self.resolved,
                cache_dir=self.cache_dir,
                force_download=force,
            )

        return Path(os.path.realpath(_retry(call)))

    def _download_checked(self, repo_path: str, content: str, hashes: _HashMemo) -> Path:
        """The local file of ``repo_path``, checked against its listed ``content``. One that does not match (a
        damaged copy in the local cache, a bad transfer) is downloaded once more, bypassing the local cache, and
        ``FetchError`` names it if it still does not match."""

        local = self._download(repo_path)
        if _has_content(local, content, hashes):
            return local
        where = f"hf://{self.repo}@{self.resolved}: {repo_path}"
        logger.warning("%s does not match the repository's listing (%s); downloading it again", where, content)
        try:
            local = self._download(repo_path, force=True)
        except Exception as exc:  # e.g. offline mode, which refuses forced downloads
            raise FetchError(
                f"{where} does not match the repository's listing ({content}), and downloading it again failed: {exc}"
            ) from exc
        if not _has_content(local, content, hashes):
            raise FetchError(f"{where} does not match the repository's listing ({content}), downloaded again too")
        return local

    def _optional_json(self, relative: str) -> Any:
        try:
            return json.loads(self._download(self._repo_path(relative)).read_text())
        except _remote_entry_not_found():
            return None

    def candidates(self) -> dict[str, Candidate]:
        registry = self._optional_json(REGISTRY)
        task_index = self._optional_json(TASK_INDEX)
        if registry is not None or task_index is not None:
            return _candidates_from_index(registry, task_index, f"hf://{self.repo}@{self.resolved}")
        candidates: dict[str, Candidate] = {}
        for entry in self._list(self.subdir):
            path = entry["path"]
            if not path.endswith("/task.toml"):
                continue
            relative = path.removeprefix(f"{self.subdir}/" if self.subdir else "").rsplit("/", 1)[0]
            if relative in candidates:
                continue
            candidates[relative] = Candidate(name=relative, path=relative)
        if not candidates:
            raise FetchError(f"hf://{self.repo}@{self.resolved} holds no Harbor task (no task.toml)")
        return candidates

    def read_json_rows(self, spec: IndexFile) -> Iterator[tuple[str, Mapping[str, Any]]]:
        return _json_rows(self._download(self._repo_path(spec.path)).read_text(), spec)

    def _list(self, repo_dir: str) -> list[dict[str, Any]]:
        """Files below ``repo_dir`` (recursively) with sizes and content ids, cached per commit."""

        from huggingface_hub.hf_api import RepoFile

        cached = self._listing_dir / (hashlib.sha256(repo_dir.encode()).hexdigest()[:24] + ".json")
        if cached.is_file():
            return json.loads(cached.read_text())

        def call() -> list[dict[str, Any]]:
            entries = self.api.list_repo_tree(
                self.repo, path_in_repo=repo_dir or None, recursive=True, revision=self.resolved, repo_type="dataset"
            )
            return [
                {
                    "path": entry.path,
                    "size": entry.size,
                    "content": f"lfs:{entry.lfs.sha256}" if entry.lfs is not None else f"git:{entry.blob_id}",
                    "sha256": entry.lfs.sha256 if entry.lfs is not None else None,
                }
                for entry in entries
                if isinstance(entry, RepoFile)
            ]

        listing = _retry(call)
        cached.parent.mkdir(parents=True, exist_ok=True)
        temporary = cached.with_name(f".{cached.name}.{uuid.uuid4().hex}")
        temporary.write_text(json.dumps(listing))
        os.replace(temporary, cached)
        return listing

    def materialize(self, selected: Sequence[Candidate], hashes: _HashMemo) -> dict[str, _TaskFiles]:
        roots = {self._repo_path(candidate.path): candidate for candidate in selected}
        by_parent: dict[str, list[str]] = defaultdict(list)
        for root in roots:
            by_parent[root.rsplit("/", 1)[0] if "/" in root else ""].append(root)
        listed: list[dict[str, Any]] = []
        started = time.monotonic()
        for parent, children in sorted(by_parent.items()):
            if len(children) < _PER_TASK_LISTING:
                for child in children:
                    listed += self._list(child)
            else:
                listed += self._list(parent)
        self.log(f"listed {len(listed)} files in {time.monotonic() - started:.0f}s")
        per_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for entry in listed:
            parts = entry["path"].split("/")
            for depth in range(len(parts) - 1, 0, -1):
                root = "/".join(parts[:depth])
                if root in roots:
                    relative = "/".join(parts[depth:])
                    if _is_hashed_path(relative):
                        per_task[root][relative] = entry
                    break
        missing = sorted(root for root in roots if "task.toml" not in per_task.get(root, {}))
        if missing:
            raise FetchError(f"{len(missing)} selected tasks have no task.toml in the repository, e.g. {missing[:3]}")
        # Harbor's hash leaves out what a task's .gitignore (else its default ignores) names, and a fetch of a local
        # directory never copies those files (hashing.task_files): read the .gitignore files first, so that only
        # files the hash covers are downloaded (and the .gitignore itself, which the hash of the cached task reads).
        gitignores = [entries[GITIGNORE] for entries in per_task.values() if GITIGNORE in entries]
        local = self._download_all(_one_path_per_content(gitignores), hashes) if gitignores else {}
        hashed: dict[str, dict[str, dict[str, Any]]] = {}
        for root, entries in per_task.items():
            gitignore = entries.get(GITIGNORE)
            matcher = IgnoreMatcher(
                local[gitignore["content"]].read_text().splitlines() if gitignore is not None else DEFAULT_IGNORES
            )
            hashed[root] = {
                relative: entry
                for relative, entry in entries.items()
                if relative == GITIGNORE or not matcher.ignored(relative)
            }
        remaining = [
            entry for entries in hashed.values() for entry in entries.values() if entry["content"] not in local
        ]
        local.update(self._download_all(_one_path_per_content(remaining), hashes))
        return {
            roots[root].name: _TaskFiles(
                files={relative: local[entry["content"]] for relative, entry in entries.items()}, link=True
            )
            for root, entries in hashed.items()
        }

    def _download_all(self, contents: Mapping[str, str], hashes: _HashMemo) -> dict[str, Path]:
        """Download one repository path per distinct content, ``workers`` at a time, each checked against its
        content id (:meth:`_download_checked`)."""

        local: dict[str, Path] = {}
        started = time.monotonic()
        total = len(contents)
        self.log(f"downloading {total} distinct files with {self.workers} workers")
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(self._download_checked, path, content, hashes): content
                for content, path in contents.items()
            }
            for done, future in enumerate(as_completed(futures), 1):
                local[futures[future]] = future.result()
                if done % 2000 == 0 or done == total:
                    elapsed = time.monotonic() - started
                    self.log(f"downloaded {done}/{total} files in {elapsed:.0f}s")
        return local


# --- caching one task ------------------------------------------------------------------------------------------


class _HashMemo:
    """File hashes by inode: hard-linked copies of one download are hashed once. The key holds the file's size and
    modification time too, so a file rewritten since it was hashed is hashed again.

    Removing a file's last link frees its inode, and a file created later may take it with the same size and
    modification time: a copy keeps its source's, and timestamps are coarse.  A remembered hash would then answer for
    that file's bytes, and a second fetch of a local directory, which removes each task's copies once it finds the
    task cached, gave tasks the digests of others.  So while a task is cached (:meth:`watch`), the memo notes where
    each file it hashes is, and :meth:`forget_moved` drops the hashes of those no longer there: after overlays
    replace files, and once the task's staging directory is gone."""

    def __init__(self) -> None:
        self._sha256: dict[tuple[int, int, int, int], str] = {}
        self._blob_ids: dict[tuple[int, int, int, int], str] = {}
        self._inputs: dict[tuple[int, int, int, int], str] = {}  # input digests (inputs.input_digest)
        self._lock = threading.Lock()
        self._watched: dict[tuple[int, int, int, int], Path] | None = None  # key -> the path it was hashed at

    def sha256(self, path: Path) -> str:
        key = _file_key(path)
        with self._lock:
            known = self._sha256.get(key)
        if known is None:
            known = file_sha256(path)
            with self._lock:
                self._sha256[key] = known
                if self._watched is not None:
                    self._watched[key] = path
        return known

    def watch(self) -> None:
        """Note where each file hashed from now on is, for :meth:`forget_moved`."""

        self._watched = {}

    def forget_moved(self, *, stop: bool = False) -> None:
        """Forget the hashes of the files hashed since :meth:`watch` that are no longer where they were hashed (removed,
        replaced or moved): their inodes may be another file's next.  ``stop`` ends the watch."""

        with self._lock:
            watched, self._watched = self._watched or {}, None if stop else {}
        for key, path in watched.items():
            try:
                kept = _file_key(path) == key
            except OSError:
                kept = False
            with self._lock:
                if not kept:
                    self._sha256.pop(key, None)
                    self._blob_ids.pop(key, None)
                    self._inputs.pop(key, None)
                elif not stop and self._watched is not None:
                    self._watched[key] = path

    def git_blob_id(self, path: Path) -> str:
        """The file's git blob id (SHA-1 over ``blob <size>\\0`` and its bytes), hashed in one read with its SHA-256."""

        key = _file_key(path)
        with self._lock:
            known = self._blob_ids.get(key)
        if known is None:
            sha256, known = _sha256_and_blob_id(path)
            with self._lock:
                self._sha256[key] = sha256
                self._blob_ids[key] = known
        return known

    def input_digest(self, path: Path) -> str:
        """The digest by which a hold-out compares the input file at ``path``
        (:func:`~platoon.envs.harbor.inputs.input_digest`): a binary file's is its SHA-256, which the task's digest
        hashed already; a text file is read once more, to normalize its text.  Remembered by the key of
        :meth:`sha256`, and forgotten with it."""

        self.sha256(path)  # remembered, and watched while a task is cached, as for the task's digest
        key = _file_key(path)
        with self._lock:
            known = self._inputs.get(key)
        if known is None:
            known = input_digest(path, sha256=self.sha256)
            with self._lock:
                self._inputs[key] = known
        return known

    def content_hash(self, task_dir: Path) -> str:
        outer = hashlib.sha256()
        for relative in task_files(task_dir):
            outer.update(f"{relative}\0{self.sha256(task_dir / relative)}\n".encode())
        return outer.hexdigest()


def _file_key(path: Path) -> tuple[int, int, int, int]:
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _sha256_and_blob_id(path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        blob = hashlib.sha1(b"blob %d\0" % os.fstat(handle.fileno()).st_size, usedforsecurity=False)
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(chunk)
            blob.update(chunk)
    return sha256.hexdigest(), blob.hexdigest()


def _has_content(path: Path, content: str, hashes: _HashMemo) -> bool:
    """Whether ``path`` holds a listing's ``content``: ``lfs:<SHA-256>`` or ``git:<blob id>``."""

    kind, _, expected = content.partition(":")
    return (hashes.sha256(path) if kind == "lfs" else hashes.git_blob_id(path)) == expected


def _one_path_per_content(entries: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Listing entries as content id -> the repository path to download for it (the first in sort order)."""

    contents: dict[str, str] = {}
    for entry in entries:
        current = contents.get(entry["content"])
        if current is None or entry["path"] < current:
            contents[entry["content"]] = entry["path"]
    return contents


def _cache_task(
    candidate: Candidate, files: _TaskFiles, config: FetchConfig, overlays: Sequence[Overlay], hashes: _HashMemo
) -> LockedTask:
    staging = config.cache / ".tmp" / uuid.uuid4().hex
    staging.mkdir(parents=True)
    hashes.watch()
    try:
        for relative, source in files.files.items():
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if files.link:
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
            else:
                shutil.copy2(source, target)
            _restore_executable(target)
        upstream = hashes.content_hash(staging)
        if candidate.digest is not None and f"sha256:{upstream}" != candidate.digest:
            raise FetchError(
                f"Harbor task {candidate.name!r} of {config.dataset}: its files hash to sha256:{upstream}, and the "
                f"source gives it {candidate.digest}: a damaged download, or a content hash that is not Harbor's"
            )
        applied: list[str] = []
        refused: str | None = None
        edits: list[ComposeEdit] = []
        host_env: list[str] = []
        for overlay in overlays:
            if overlay.matches(candidate.name, candidate.index):
                problem = overlay.apply(staging)
                applied.append(overlay.name)
                edits += overlay.compose
                host_env += overlay.compose_host_env
                if problem is not None:
                    refused = f"overlay: {problem}"
                    break
        if applied:
            hashes.forget_moved()  # the files the overlays replaced
        digest = hashes.content_hash(staging) if applied else upstream
        index = _with_metadata(candidate, staging, config.metadata_fields)
        inspected = _inspect(staging, candidate.name, config.refuse, edits=edits, host_env=host_env)
        flags, image, verifier_image, problem, compose = inspected
        # Binary files are hashed already, for the digest; the files main's compose binds are inputs too.
        files_in = input_files(staging, digest=hashes.input_digest, compose=compose)
        refused = refused or problem
        entry = config.cache / TASKS_DIR / digest
        with _flock(config.cache / ".locks" / "tasks.lock"):
            if _holds(entry, digest, hashes):
                shutil.rmtree(staging, ignore_errors=True)
            else:
                _install(staging, entry)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        hashes.forget_moved(stop=True)  # the staging copies, gone or moved into the cache
    return LockedTask(
        name=candidate.name,
        digest=f"sha256:{digest}",
        path=f"{TASKS_DIR}/{digest}",
        source_path=candidate.path,
        upstream_digest=f"sha256:{upstream}" if upstream != digest else None,
        index=index,
        flags=flags,
        image=image,
        verifier_image=verifier_image,
        overlays=tuple(applied),
        refused=refused,
        input_files=files_in,
        compose=compose.to_json() if compose is not None else None,
    )


def _with_metadata(candidate: Candidate, task_dir: Path, fields: Sequence[str]) -> dict[str, Any]:
    """The candidate's index fields, with ``fields`` of the ``[metadata]`` table of the ``task.toml`` in
    ``task_dir`` (the task as cached, overlays applied) added.  A task.toml that cannot be read records none, as the
    fetch refuses that task (``task_config``); a value that is not a scalar, or that differs from the one the
    dataset's index gives, fails the fetch."""

    index = dict(candidate.index)
    if not fields:
        return index
    try:
        config = tomllib.loads((task_dir / "task.toml").read_text())
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return index
    table = config.get("metadata")
    if not isinstance(table, dict):
        return index
    for name in fields:
        if name not in table:
            continue
        value = table[name]
        if not (isinstance(value, str | bool | int) or (isinstance(value, float) and math.isfinite(value))):
            raise FetchError(
                f"Harbor task {candidate.name!r}: task.toml [metadata] {name} is {value!r}; an index field must be a "
                "string, a finite number or a boolean"
            )
        if name in index and index[name] != value:
            raise FetchError(
                f"Harbor task {candidate.name!r}: the index field {name!r} is {index[name]!r} in the dataset's index "
                f"and {value!r} in task.toml [metadata]"
            )
        index[name] = value
    return index


def _holds(entry: Path, digest: str, hashes: _HashMemo) -> bool:
    """Whether the cache entry ``entry`` holds the files of ``digest``. It is hashed again, since its files may have
    changed after it was written; those hard-linked to this fetch's downloads were hashed already."""

    try:
        return entry.is_dir() and hashes.content_hash(entry) == digest
    except OSError:
        return False


def _install(staging: Path, entry: Path) -> None:
    """Rename ``staging`` to the cache entry ``entry``. A damaged entry there is renamed aside first and deleted
    after the new one is in place, so the entry's path never holds a mix of both."""

    if not os.path.lexists(entry):
        os.replace(staging, entry)
        return
    logger.warning("The Harbor cache entry %s does not match its digest; replacing it", entry)
    damaged = staging.with_name(f"damaged-{uuid.uuid4().hex}")
    os.replace(entry, damaged)
    os.replace(staging, entry)
    if damaged.is_dir() and not damaged.is_symlink():
        shutil.rmtree(damaged, ignore_errors=True)
    else:
        damaged.unlink(missing_ok=True)


def _inspect(
    task_dir: Path,
    name: str,
    refuse: RefusePolicy,
    *,
    edits: Sequence[ComposeEdit] = (),
    host_env: Sequence[str] = (),
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, str | None, ComposePlan | None]:
    """A task's flags, build plans, refusal and stack plan, from its ``task.toml`` and files in ``task_dir``, a
    temporary directory that messages never name (they name ``name``, the task's id, and paths relative to the task).
    ``edits`` and ``host_env`` are what the overlays that match the task give its compose file.

    The refusal is the first of: a compose file outside the subset platoon runs (``compose``); what ``refuse``
    refuses (``stack``, ``gpus``, ``agent_user``); anything that keeps :meth:`HarborTask.load` from loading the task
    (``task_config``), which every rollout of it would fail on; a Dockerfile that cannot be planned (``image``); and
    uploads over ``refuse.uploads_over_mb`` (``uploads_over_mb``).  ``flags["upload_bytes"]`` records what a session
    copies into one of the task's sandboxes (:func:`_upload_bytes`).
    """

    try:
        config = tomllib.loads((task_dir / "task.toml").read_text())
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        problem = f"task_config: task.toml: {_without_paths(str(exc), task_dir, name)}"
        return {}, {"kind": "none"}, None, problem, None
    environment = _table(config, "environment")
    verifier = _table(config, "verifier")
    agent = _table(config, "agent")
    verifier_environment = verifier.get("environment") if isinstance(verifier.get("environment"), dict) else None
    network = environment.get("network_mode") or (
        "no-network" if environment.get("allow_internet") is False else "public"
    )
    artifacts = config.get("artifacts") or []
    gpus = environment.get("gpus")
    compose, compose_problem = _plan_compose(
        task_dir, name, environment, edits, host_env, privileges=not refuse.privileges
    )
    flags: dict[str, Any] = {
        "verifier": "separate" if verifier.get("environment_mode") == "separate" or verifier_environment else "shared",
        "artifacts": len(artifacts) if isinstance(artifacts, list) else 0,
        "stack": compose.kind if compose is not None else _stack_kind(task_dir),
        "network": network,
        "gpus": gpus if isinstance(gpus, int) and not isinstance(gpus, bool) else 0,  # the loader refuses the others
        "agent_user": agent.get("user"),
        "verifier_user": verifier.get("user"),
        "os": str(environment.get("os", "linux")).lower(),
    }
    if verifier_environment is not None and verifier_environment.get("network_mode"):
        flags["verifier_network"] = verifier_environment["network_mode"]
    if unread := unread_compose_files(task_dir):
        flags["unread_compose"] = unread
    image = _image_plan(task_dir, environment, "environment")
    verifier_image = None
    if flags["verifier"] == "separate":
        if verifier_environment is not None and verifier_environment.get("docker_image"):
            verifier_image = {"kind": "prebuilt", "ref": verifier_environment["docker_image"]}
        elif (task_dir / "tests" / "Dockerfile").is_file():
            verifier_image = _image_plan(task_dir, {}, "tests")
    plans = [plan for plan in (image, verifier_image) if plan is not None]
    for plan in plans:
        if plan.get("error"):  # recorded in the lock, which must not name the temporary directory
            plan["error"] = _without_paths(plan["error"], task_dir, name)
    flags["upload_bytes"] = _upload_bytes(task_dir, image, verifier_image, separate=flags["verifier"] == "separate")
    suffix_users = [plan["suffix_user"] for plan in plans if plan.get("suffix_user")]
    errors = [plan["error"] for plan in plans if plan.get("error")]
    if compose_problem is not None:
        problem = compose_problem
    elif flags["stack"] == "sidecars" and refuse.sidecars:
        problem = "stack: the compose file adds sidecar services (refuse: {sidecars: false} fetches it as a stack)"
    elif flags["gpus"] > 0 and refuse.gpus:
        problem = f"gpus: the task asks for {flags['gpus']} GPU(s)"
    elif (flags["agent_user"] or flags["verifier_user"]) and refuse.agent_user:
        problem = f"agent_user: the task sets [agent].user or [verifier].user ({_USERS_FETCHED})"
    elif suffix_users and refuse.agent_user:
        problem = f"agent_user: the Dockerfile ends with USER {suffix_users[0]} ({_USERS_FETCHED})"
    else:
        problem = _load_problem(task_dir, name, compose) or (f"image: {errors[0]}" if errors else None)
    limit = refuse.uploads_over_mb
    if problem is None and limit is not None and flags["upload_bytes"] > limit * MIB:
        problem = (
            f"uploads_over_mb: a session uploads {flags['upload_bytes'] / MIB:,.1f} MiB into one of the task's "
            f"sandboxes, over the limit of {limit:g} MiB"
        )
    return flags, image, verifier_image, problem, compose


def _plan_compose(
    task_dir: Path,
    name: str,
    environment: Mapping[str, Any],
    edits: Sequence[ComposeEdit],
    host_env: Sequence[str],
    *,
    privileges: bool = False,
) -> tuple[ComposePlan | None, str | None]:
    """The task's stack plan (None without a compose file), and the refusal of a file outside the subset
    (``compose: <service>.<key>: why``) or that an overlay edits wrongly (``overlay: ...``); ``privileges`` plans what
    only the docker provider grants rather than refusing it."""

    if compose_path(task_dir) is None:
        if edits:
            return None, f"overlay: {edits[0].overlay}: edits compose services, and the task has no {COMPOSE_FILE}"
        return None, None
    settings: dict[str, Any] = {}
    try:
        settings = _environment_settings(dict(environment), "[environment]")
    except HarborTaskError:  # the task is refused for it (task_config): plan without its settings
        pass
    image = environment.get("docker_image")
    try:
        plan = plan_compose(
            task_dir,
            docker_image=image if isinstance(image, str) and image else None,
            env=settings.get("env") or {},
            cpus=settings.get("cpus"),
            memory_mb=settings.get("memory_mb"),
            edits=edits,
            host_env=host_env,
            privileges=privileges,
        )
    except ComposeEditError as exc:
        return None, f"overlay: {_without_paths(str(exc), task_dir, name)}"
    except ComposeError as exc:
        return None, f"compose: {_without_paths(str(exc), task_dir, name)}"
    return with_unrewritable_references(plan, task_dir), None


def _upload_bytes(
    task_dir: Path, image: Mapping[str, Any], verifier_image: Mapping[str, Any] | None, *, separate: bool
) -> int:
    """The bytes a session copies into one of the task's sandboxes when it opens: the sources of the deferred
    ``COPY`` steps of its image's plan (or a prebuilt-image task's ``environment/``), and for a separate verifier
    those of its own image's plan, or the task's again when it runs the task's image.  The larger of the agents'
    sandbox and the verifier's, since each sandbox gets its own copy."""

    agents = _plan_uploads(task_dir, "environment", image)
    verifier = 0
    if separate:
        if verifier_image is None:  # the verifier runs the task's own image, with its deferred steps
            verifier = agents
        else:
            verifier = _plan_uploads(task_dir, "tests", verifier_image)
    return max(agents, verifier)


def _plan_uploads(task_dir: Path, context: str, plan: Mapping[str, Any]) -> int:
    root = task_dir / context
    if plan.get("upload_environment"):
        return _tree_bytes(root)
    total = 0
    for step in plan.get("deferred") or ():
        if not step or step[0] != "COPY":
            continue
        for source in copy_operands(step)[1]:
            relative = os.path.normpath(source.rstrip("/") or ".")
            if os.path.isabs(relative) or relative == ".." or relative.startswith("../"):
                continue  # outside the build context: nothing a session would upload
            total += _tree_bytes(root / relative)
    return total


def _tree_bytes(path: Path) -> int:
    """The size of the regular file at ``path``, or of the regular files below it; symlinks are not followed."""

    try:
        info = path.lstat()
    except OSError:
        return 0
    if stat.S_ISREG(info.st_mode):
        return info.st_size
    if not stat.S_ISDIR(info.st_mode):
        return 0
    total = 0
    for directory, _, files in os.walk(path):
        for name in files:
            try:
                info = os.lstat(os.path.join(directory, name))
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
    return total


def _load_problem(task_dir: Path, name: str, compose: ComposePlan | None = None) -> str | None:
    """Why :meth:`HarborTask.load` refuses the task, as a refusal (``task_config: ...``), or None when it loads, with
    its stack plan ``compose`` as the lock records it.

    Users to run as are let through (``agent_user="ignore"``): the refuse policy decides about them.
    """

    try:
        HarborTask.load(task_dir, agent_user="ignore", compose=compose)
    except (OSError, ValueError) as exc:  # HarborTaskError, and files that cannot be read or decoded
        return f"task_config: {_without_paths(str(exc), task_dir, name)}"
    return None


def _without_paths(message: str, task_dir: Path, name: str) -> str:
    """``message`` without the temporary ``task_dir``: its files relative to the task, and itself as ``name``."""

    for root in sorted({str(task_dir), str(task_dir.resolve())}, key=len, reverse=True):
        message = message.replace(f"{root}/", "").replace(root, name)
    return message


def _image_plan(task_dir: Path, environment: Mapping[str, Any], context: str) -> dict[str, Any]:
    docker_image = environment.get("docker_image")
    dockerfile = task_dir / context / "Dockerfile"
    if docker_image:
        plan: dict[str, Any] = {"kind": "prebuilt", "ref": docker_image}
        directory = task_dir / context
        compose = (directory / COMPOSE_FILE).is_file()  # the one compose file Harbor reads
        if not dockerfile.is_file() and not compose and directory.is_dir() and any(directory.iterdir()):
            plan["upload_environment"] = True  # Harbor uploads environment/ into the workdir of prebuilt tasks
        return plan
    if not dockerfile.is_file():
        return {"kind": "none"}
    try:
        planned = plan_dockerfile(dockerfile, task_dir / context)
    except (DockerfileError, OSError, ValueError) as exc:
        return {"kind": "none", "error": f"{context}/Dockerfile: {exc}"}
    data = {"kind": "per_task", **planned.to_json()}
    if planned.notes:
        data["notes"] = list(planned.notes)
    return data


def _assign_image_kinds(tasks: Sequence[LockedTask], refuse: RefusePolicy) -> list[LockedTask]:
    """``shared`` for a build key two or more runnable tasks use, else ``per_task`` (refused if so configured)."""

    counts: Counter[str] = Counter()
    for task in tasks:
        if task.refused is None:
            for plan in (task.image, task.verifier_image):
                if plan and plan.get("key"):
                    counts[plan["key"]] += 1
    out = []
    for task in tasks:
        image = _with_kind(task.image, counts)
        verifier_image = _with_kind(task.verifier_image, counts) if task.verifier_image else None
        refused = task.refused
        if refused is None and refuse.per_task_build:
            if any(plan and plan.get("kind") == "per_task" for plan in (image, verifier_image)):
                refused = "per_task_build: no other task shares this task's image build"
        out.append(replace(task, image=image, verifier_image=verifier_image, refused=refused))
    return out


def _with_kind(plan: Mapping[str, Any], counts: Counter[str]) -> dict[str, Any]:
    plan = dict(plan)
    if plan.get("key"):
        plan["kind"] = "shared" if counts[plan["key"]] >= 2 else "per_task"
    return plan


def _stack_kind(task_dir: Path) -> str:
    """``single`` (no compose file), else the kind its services make as written (``trivial``, ``main_only``,
    ``sidecars``): for a task whose compose file has no plan, since it is refused."""

    path = compose_path(task_dir)
    if path is None:
        return "single"
    try:
        return stack_kind(read_document(path))
    except ComposeError:
        return "sidecars"


def _restore_executable(path: Path) -> None:
    """The Hub keeps no file modes: give ``*.sh`` and ``#!`` files their executable bit back."""

    try:
        if path.suffix != ".sh":
            with path.open("rb") as handle:
                if handle.read(2) != b"#!":
                    return
        mode = path.stat().st_mode
        wanted = mode | ((mode & 0o444) >> 2)
        if wanted != mode:
            path.chmod(stat.S_IMODE(wanted))
    except OSError:
        logger.debug("Cannot restore the executable bit of %s", path, exc_info=True)


def _is_hashed_path(relative: str) -> bool:
    """Whether Harbor's content hash can cover ``relative`` (a task-relative path), or it is the ``.gitignore``."""

    top = relative.split("/", 1)[0]
    return relative in TASK_FILES or relative == GITIGNORE or ("/" in relative and top in TASK_DIRS)


def _hashed_paths(task_dir: Path) -> list[str]:
    paths = set(task_files(task_dir))
    if (task_dir / GITIGNORE).is_file():
        paths.add(GITIGNORE)
    return sorted(paths)


# --- overlays --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Overlay:
    name: str
    root: Path
    match: Mapping[str, tuple[str, ...]]
    edits: tuple[Mapping[str, Any], ...]
    copies: tuple[Mapping[str, Any], ...]
    digest: str
    compose: tuple[ComposeEdit, ...] = ()  # edits of compose services ([[compose]])
    compose_host_env: tuple[str, ...] = ()  # host variables the compose file may name

    @classmethod
    def load(cls, root: str | Path) -> Overlay:
        root = Path(root)
        path = root / OVERLAY_FILE
        try:
            raw = path.read_bytes()
            data = tomllib.loads(raw.decode())
        except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
            raise FetchError(f"overlay {root}: cannot read {OVERLAY_FILE}: {exc}") from exc
        unknown = sorted(set(data) - {"name", "description", "match", "edit", "copy", "compose", "compose_host_env"})
        if unknown:
            raise FetchError(f"overlay {path}: unknown keys {unknown}")
        name = str(data.get("name") or root.name)
        if not isinstance(data.get("compose", []), list):
            raise FetchError(f"overlay {path}: [[compose]] must be an array of tables")
        try:
            compose = tuple(ComposeEdit.parse(name, entry) for entry in data.get("compose") or ())
        except ValueError as exc:
            raise FetchError(f"overlay {path}: {exc}") from exc
        host_env = data.get("compose_host_env") or []
        if not isinstance(host_env, list) or not all(
            isinstance(variable, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", variable) for variable in host_env
        ):
            raise FetchError(f"overlay {path}: compose_host_env must list variable names, got {host_env!r}")
        match = {
            str(key): tuple(str(v) for v in (value if isinstance(value, list) else [value]))
            for key, value in (data.get("match") or {}).items()
        }
        edits = tuple(data.get("edit") or ())
        copies = tuple(data.get("copy") or ())
        digest = hashlib.sha256(raw)
        for edit in edits:
            if not isinstance(edit.get("file"), str) or not isinstance(edit.get("pattern"), str):
                raise FetchError(f"overlay {path}: each [[edit]] needs file and pattern")
            try:
                re.compile(edit["pattern"], re.MULTILINE)
            except re.error as exc:
                raise FetchError(f"overlay {path}: bad pattern {edit['pattern']!r}: {exc}") from exc
        for copy in copies:
            if not isinstance(copy.get("file"), str) or not isinstance(copy.get("source"), str):
                raise FetchError(f"overlay {path}: each [[copy]] needs file and source")
            source = root / copy["source"]
            if not source.is_file():
                raise FetchError(f"overlay {path}: {source} does not exist")
            digest.update(f"\0{copy['source']}\0{file_sha256(source)}".encode())
        if not edits and not copies and not compose and not host_env:
            raise FetchError(f"overlay {path} has no [[edit]], [[copy]], [[compose]] or compose_host_env")
        return cls(
            name=name,
            root=root,
            match=match,
            edits=edits,
            copies=copies,
            digest=digest.hexdigest(),
            compose=compose,
            compose_host_env=tuple(host_env),
        )

    def matches(self, name: str, index: Mapping[str, Any]) -> bool:
        for key, values in self.match.items():
            if key == "name":
                if not any(fnmatchcase(name, value) for value in values):
                    return False
            elif str(index.get(key)) not in values:
                return False
        return True

    def apply(self, task_dir: Path) -> str | None:
        """Edit ``task_dir`` (new files, never written in place: they may be hard links); a problem, or None."""

        for edit in self.edits:
            target = task_dir / edit["file"]
            if not target.is_file():
                return f"{self.name}: {edit['file']} does not exist"
            text = target.read_text()
            updated, count = re.subn(edit["pattern"], edit.get("replace", ""), text, flags=re.MULTILINE)
            expected = int(edit.get("count", 1))
            if count != expected:
                return f"{self.name}: {edit['file']}: the pattern matched {count} time(s), expected {expected}"
            _replace_file(target, updated.encode(), executable=os.access(target, os.X_OK) or None)
        for copy in self.copies:
            source = self.root / copy["source"]
            target = task_dir / copy["file"]
            _replace_file(target, source.read_bytes(), executable=copy.get("executable"))
        return None


def _overlays_digest(overlays: Sequence[Overlay]) -> str | None:
    if not overlays:
        return None
    combined = hashlib.sha256("".join(f"{overlay.name}\0{overlay.digest}\n" for overlay in overlays).encode())
    return f"sha256:{combined.hexdigest()}"


def _replace_file(target: Path, data: bytes, *, executable: bool | None) -> None:
    """Write ``target`` as a new file (it may be a hard link to a shared download); executable when asked, else
    by the fetch's rule (``*.sh`` and ``#!`` files)."""

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}")
    temporary.write_bytes(data)
    if executable:
        mode = temporary.stat().st_mode
        temporary.chmod(stat.S_IMODE(mode | ((mode & 0o444) >> 2)))
    os.replace(temporary, target)
    if executable is None:
        _restore_executable(target)


# --- helpers ---------------------------------------------------------------------------------------------------


def _retry(call: Callable[[], Any], attempts: int = _MAX_ATTEMPTS) -> Any:
    """Call with exponential backoff on rate limits (429), server errors and transport failures; a missing
    repository, revision or file fails at once, and so does anything in offline mode."""

    from huggingface_hub import constants, errors

    permanent = (errors.RepositoryNotFoundError, errors.RevisionNotFoundError, _remote_entry_not_found())
    delay = 5.0
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            if isinstance(exc, permanent) or attempt == attempts or constants.HF_HUB_OFFLINE:
                raise
            wait = delay
            if isinstance(exc, errors.HfHubHTTPError):
                response = getattr(exc, "response", None)
                status = getattr(response, "status_code", None)
                if status is not None and status < 500 and status != 429:
                    raise
                retry_after = (getattr(response, "headers", None) or {}).get("Retry-After")
                if retry_after and str(retry_after).isdigit():
                    wait = float(retry_after)
            elif not isinstance(exc, OSError) and type(exc).__module__.split(".")[0] not in ("httpx", "httpcore"):
                raise
            logger.warning("Retrying after %s: %s", type(exc).__name__, exc)
        time.sleep(wait + random.uniform(0, wait / 4))
        delay = min(delay * 2, 300.0)
    raise AssertionError("unreachable")


def _remote_entry_not_found() -> type[Exception]:
    """The error for a file the repository does not have (not one that could not be reached)."""

    from huggingface_hub import errors

    return getattr(errors, "RemoteEntryNotFoundError", errors.EntryNotFoundError)


def _read_if_present(path: Path) -> DatasetLock | None:
    if not path.is_file():
        return None
    try:
        return read_lock(path)
    except LockError as exc:
        logger.warning("Ignoring the unreadable Harbor lock %s: %s", path, exc)
        return None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.is_file() else None


def _sha256_file(path: Path) -> str:
    try:
        return file_sha256(path)
    except OSError as exc:
        raise FetchError(f"cannot read {path}: {exc}") from exc


def _task_name(config: Path) -> str | None:
    try:
        name = tomllib.loads(config.read_text()).get("task", {}).get("name")
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return name if isinstance(name, str) and name else None


def _table(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    return value if isinstance(value, dict) else {}


@contextlib.contextmanager
def _flock(path: Path) -> Generator[None, None, None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
