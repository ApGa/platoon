"""The sandboxes a process holds, and reaping them by label when the process cannot stop them itself.

A rollout worker that is killed never runs the teardown of the sandboxes its rollout opened: hpc-sandbox instances
and Docker containers outlive it (the supervisor's SIGKILL, its own hard-timeout alarm, the RSS watchdog's
``os._exit``), and a stopping worker's deadline may end before a slow teardown does.  So the built-in providers
(:mod:`platoon.sandbox.process`, :mod:`platoon.sandbox.docker`, :mod:`platoon.sandbox.hpc`) do three things for every
sandbox they start:

* they add this process's labels (:func:`set_labels`) to the sandbox's own, so its sandboxes can be found by label
  from any process: a run's ``platoon.run: <run directory>`` (:func:`run_label`), which ``platoon train`` and
  ``platoon eval`` set and their rollout workers carry on, and a rollout worker's ``platoon.worker: <request id>``;
* they register the sandbox here before it starts (:func:`register`) and unregister it once it stopped, and
  :func:`mirror_to` keeps a 0600 file that says how to reap what is registered: the provider's class and options,
  and the labels;
* they refuse to start a sandbox once this process stopped using them (:func:`shut_down`).

Then a stopping worker stops what its rollout left registered (:func:`stop_live`), and its supervisor, once the
worker is gone however it ended, reaps the labels the file lists through providers built from it
(:func:`reap_records`): hpc-sandbox sweeps the sessions their dead owner left, Docker removes the labelled
containers, the process provider kills the labelled sandboxes' process groups and deletes their directories.
Importing this module imports no provider.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "RUN_LABEL",
    "WORKER_LABEL",
    "SandboxRecord",
    "SandboxesShutDown",
    "attach",
    "check_running",
    "labelled",
    "labels",
    "labels_match",
    "labels_to_reap",
    "live_sandboxes",
    "mirror_to",
    "provider_ref",
    "read_mirror",
    "reap_records",
    "register",
    "run_label",
    "set_labels",
    "shut_down",
    "stop_live",
    "unregister",
]

#: The label every sandbox of a ``platoon train`` or ``platoon eval`` run carries, in the run's own process and in
#: its rollout workers: the run directory, as an absolute path (:func:`run_label`).  It selects one run's sandboxes,
#: and no other run's, where several runs share a node.
RUN_LABEL = "platoon.run"
#: The label a rollout worker puts on every sandbox it starts: its request id.
WORKER_LABEL = "platoon.worker"


@dataclass(frozen=True)
class SandboxRecord:
    """How to reap a sandbox without the process that started it: build ``provider`` (a ``module:Class`` that
    ``platoon.sandbox.get_provider`` takes) with ``options`` and call its ``reap(labels)``."""

    provider: str
    options: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)


class SandboxesShutDown(RuntimeError):
    """This process stopped using sandboxes (:func:`shut_down`): no sandbox starts."""


@dataclass
class _Entry:
    record: SandboxRecord | None  # None: no labels to reap it by, so it is not mirrored
    sandbox: Any = None  # set once the sandbox runs


_lock = threading.Lock()
_live: dict[int, _Entry] = {}
_tokens = itertools.count(1)
_labels: dict[str, str] = {}
_mirror_path: str | None = None
_shut_down = False


def set_labels(labels: Mapping[str, str]) -> None:
    """Add ``labels`` to every sandbox the built-in providers start in this process from now on."""

    with _lock:
        _labels.clear()
        _labels.update({str(key): str(value) for key, value in labels.items()})


def labels() -> dict[str, str]:
    """The labels this process adds to its sandboxes (:func:`set_labels`)."""

    with _lock:
        return dict(_labels)


@contextmanager
def labelled(extra: Mapping[str, str]) -> Iterator[dict[str, str]]:
    """Add ``extra`` to the labels this process adds to its sandboxes (:func:`set_labels`) inside the block, and put
    back the labels it had after it; yields the labels in force."""

    previous = labels()
    set_labels({**previous, **extra})
    try:
        yield labels()
    finally:
        set_labels(previous)


def run_label(run_dir: str | os.PathLike[str]) -> dict[str, str]:
    """The label of the sandboxes of the run whose directory is ``run_dir``: ``{RUN_LABEL: <its absolute path>}``."""

    return {RUN_LABEL: os.path.abspath(os.fspath(run_dir))}


def check_running() -> None:
    """Raise :class:`SandboxesShutDown` once :func:`shut_down` was called."""

    if _shut_down:
        raise SandboxesShutDown("This process is shutting down: it starts no sandbox")


def shut_down() -> int:
    """Stop starting sandboxes for good; returns how many are still registered.  Idempotent.

    For a process that is exiting (a rollout worker told to stop): from now on a built-in provider's ``start`` and
    ``restore`` raise :class:`SandboxesShutDown` before they start anything; stopping still works.
    """

    global _shut_down
    with _lock:
        _shut_down = True
        return len(_live)


def register(*, provider: str, options: Mapping[str, Any], labels: Mapping[str, str]) -> int:
    """Note a sandbox that is about to start, before it starts; returns the token :func:`unregister` takes.

    ``labels`` are the sandbox's labels; it is reaped by this process's labels (:func:`set_labels`) when it has
    any, else by its own, and not mirrored when it has neither, since an empty label set would select every
    sandbox.  Raises :class:`SandboxesShutDown` after :func:`shut_down`.
    """

    check_running()
    selector = labels_to_reap(labels)
    record = SandboxRecord(provider=provider, options=_jsonable(options), labels=selector) if selector else None
    with _lock:
        if _shut_down:
            raise SandboxesShutDown("This process is shutting down: it starts no sandbox")
        token = next(_tokens)
        _live[token] = _Entry(record)
        _write_mirror()
    return token


def attach(token: int, sandbox: Any) -> None:
    """Note the running sandbox of a registration, which :func:`stop_live` stops."""

    with _lock:
        entry = _live.get(token)
        if entry is not None:
            entry.sandbox = sandbox


def unregister(token: int | None) -> None:
    """Forget a sandbox once it stopped, or failed to start (an unknown token is ignored)."""

    if token is None:
        return
    with _lock:
        if _live.pop(token, None) is not None:
            _write_mirror()


def _jsonable(options: Mapping[str, Any]) -> dict[str, Any]:
    """``options`` as JSON holds them (paths become strings), which is how the mirror records them."""

    return json.loads(json.dumps(dict(options), default=str))


def provider_ref(provider: Any) -> str:
    """The ``module:Class`` reference that rebuilds ``provider`` (``platoon.sandbox.get_provider`` takes it)."""

    cls = type(provider)
    return f"{cls.__module__}:{cls.__qualname__}"


def labels_match(labels: Any, wanted: Mapping[str, str]) -> bool:
    """Whether ``labels`` (a mapping, else nothing matches) include every ``wanted`` key with its value."""

    return isinstance(labels, Mapping) and all(labels.get(key) == value for key, value in wanted.items())


def labels_to_reap(sandbox_labels: Mapping[str, str]) -> dict[str, str]:
    """The labels that select a sandbox for reaping: this process's, else the sandbox's own (may be empty)."""

    own = labels()
    return own if own else {str(key): str(value) for key, value in sandbox_labels.items()}


def live_sandboxes() -> list[Any]:
    """The running sandboxes this process registered and has not stopped."""

    with _lock:
        return [entry.sandbox for entry in _live.values() if entry.sandbox is not None]


def mirror_to(path: str | None) -> None:
    """Keep ``path`` (written 0600) listing how to reap the registered sandboxes from now on; None stops."""

    global _mirror_path
    with _lock:
        _mirror_path = path
        _write_mirror()


def read_mirror(path: str) -> list[SandboxRecord]:
    """The distinct records a :func:`mirror_to` file lists; a missing or unreadable file lists none."""

    try:
        with open(path, encoding="utf-8") as handle:
            entries = json.load(handle)
        records = [SandboxRecord(**entry) for entry in entries]
    except (OSError, ValueError, TypeError):
        return []
    return [record for record in records if record.labels]


def _write_mirror() -> None:
    # Called with _lock held.
    if _mirror_path is None:
        return
    records: list[dict[str, Any]] = []
    for entry in _live.values():
        if entry.record is not None and (payload := asdict(entry.record)) not in records:
            records.append(payload)
    temporary = f"{_mirror_path}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(records, handle)
        os.replace(temporary, _mirror_path)
    except (OSError, TypeError, ValueError):
        logger.warning("Cannot record the live sandboxes in %s", _mirror_path, exc_info=True)


async def stop_live(timeout_s: float) -> int:
    """Stop every running sandbox still registered, concurrently and within ``timeout_s`` in all; returns how many
    stopped.  Never raises: a failure is logged, and what is left is the supervisor's to reap."""

    sandboxes = live_sandboxes()
    if not sandboxes:
        return 0

    async def stop(sandbox: Any) -> bool:
        try:
            await sandbox.stop()
        except Exception:
            logger.warning("Cannot stop sandbox %s", getattr(sandbox, "id", sandbox), exc_info=True)
            return False
        return True

    tasks = [asyncio.ensure_future(stop(sandbox)) for sandbox in sandboxes]
    done, pending = await asyncio.wait(tasks, timeout=max(0.0, timeout_s))
    for task in pending:
        task.cancel()
    stopped = sum(1 for task in done if not task.cancelled() and task.result())
    logger.info("Stopped %d of %d sandboxes this process left running", stopped, len(sandboxes))
    return stopped


async def reap_records(records: Sequence[SandboxRecord], timeout_s: float) -> int:
    """Reap ``records``' labels through their providers, built from the records, within ``timeout_s`` in all;
    returns how many sandboxes the providers stopped or removed.  Never raises (a failure is logged)."""

    from platoon.sandbox import get_provider
    from platoon.sandbox.base import SupportsReap

    unique: list[SandboxRecord] = []
    for record in records:
        if record.labels and record not in unique:
            unique.append(record)

    async def reap(record: SandboxRecord) -> int:
        try:
            provider = get_provider(record.provider, **record.options)
            if not isinstance(provider, SupportsReap):
                logger.warning("Sandbox provider %s cannot reap sandboxes by label", record.provider)
                return 0
            return await provider.reap(record.labels)
        except Exception:
            logger.warning("Cannot reap the sandboxes labelled %s (%s)", record.labels, record.provider, exc_info=True)
            return 0

    if not unique:
        return 0
    tasks = [asyncio.ensure_future(reap(record)) for record in unique]
    done, pending = await asyncio.wait(tasks, timeout=max(0.0, timeout_s))
    for task in pending:
        task.cancel()
    if pending:
        logger.warning("Gave up reaping sandboxes after %.0fs", timeout_s)
    return sum(task.result() for task in done if not task.cancelled())
