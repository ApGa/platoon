"""The node-local record of the rollout workers a trainer runs, so a dead trainer's workers can be killed.

A worker leads its own session (``start_new_session=True``), so a signal to the trainer's process group never
reaches it.  A worker stops on its own when its trainer dies -- PDEATHSIG and its bounded termination
(:mod:`platoon.train.workers.entrypoint`) -- and this module is the backstop for when that does not happen:
the supervisor writes one small file per live worker under ``<scratch>/workers/`` (its pid, process group and start
time, and the trainer's pid and start time) and deletes it once it has reaped the worker.  A file whose trainer is
gone therefore names a worker nobody owns; :func:`sweep_orphaned_workers` kills its process group.  Two callers run
the sweep before new workers start next to the old ones:

* ``platoon train`` (:mod:`platoon.train.reaping`), before it starts training, before each restart of a supervised
  run and when it exits, for every trainer that is gone;
* the supervisor of a new trainer, before it launches its first worker, for every trainer that is gone (a restart
  by any other means).

A process is identified by its pid *and* its start time (``/proc/<pid>/stat``), and signalled through a pidfd
opened before the identity check, so a recycled pid is never signalled.  A worker whose trainer still runs, this
trainer's or another one sharing the scratch directory, is never touched.  Stdlib only.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import select
import signal
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

__all__ = [
    "REGISTRY_DIRNAME",
    "WorkerRecord",
    "forget_worker",
    "process_start_time",
    "record_worker",
    "registry_dir",
    "sweep_orphaned_workers",
]

#: Subdirectory of the supervisor's scratch directory holding one file per live worker.
REGISTRY_DIRNAME = "workers"
#: How long a swept worker's process group gets between SIGTERM and SIGKILL.
DEFAULT_SWEEP_GRACE_SECONDS = 10.0


@dataclass(frozen=True)
class WorkerRecord:
    """A live rollout worker and the trainer that launched it, each identified by pid and start time."""

    pid: int
    pgid: int
    start_time: int
    trainer_pid: int
    trainer_start_time: int
    request_id: str = ""


def registry_dir(scratch_dir: str) -> str:
    return os.path.join(scratch_dir, REGISTRY_DIRNAME)


def process_start_time(pid: int) -> int | None:
    """When ``pid`` started (clock ticks since boot, ``/proc/<pid>/stat`` field 22); None for no such process."""

    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            raw = handle.read()
        return int(raw[raw.rindex(b")") + 2 :].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _running(pid: int, start_time: int) -> bool:
    """Whether the process ``pid`` that started at ``start_time`` still runs (a zombie has stopped running)."""

    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            raw = handle.read()
        fields = raw[raw.rindex(b")") + 2 :].split()
        return int(fields[19]) == start_time and fields[0] not in (b"Z", b"X", b"x")
    except (OSError, ValueError, IndexError):
        return False


def record_worker(directory: str, record: WorkerRecord) -> None:
    """Note a live worker (written 0600, atomically); a failure is logged, never raised."""

    try:
        os.makedirs(directory, mode=0o700, exist_ok=True)
        path = os.path.join(directory, str(record.pid))
        temporary = f"{path}.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(asdict(record), handle)
        os.replace(temporary, path)
    except OSError:
        logger.warning("Cannot record rollout worker %s in %s", record.pid, directory, exc_info=True)


def forget_worker(directory: str, pid: int) -> None:
    """Drop a reaped worker's file."""

    try:
        os.unlink(os.path.join(directory, str(pid)))
    except OSError as exc:
        if exc.errno != errno.ENOENT:
            logger.debug("Cannot drop rollout worker %s from %s: %s", pid, directory, exc)


def _records(directory: str) -> Iterator[tuple[str, WorkerRecord | None]]:
    try:
        names = os.listdir(directory)
    except OSError:
        return
    for name in names:
        if not name.isdigit():
            continue
        path = os.path.join(directory, name)
        try:
            with open(path, encoding="utf-8") as handle:
                yield path, WorkerRecord(**json.load(handle))
        except (OSError, ValueError, TypeError):
            yield path, None


def sweep_orphaned_workers(
    directory: str,
    *,
    trainer_pid: int | None = None,
    grace_seconds: float = DEFAULT_SWEEP_GRACE_SECONDS,
) -> list[int]:
    """Kill the process group of every recorded worker whose trainer is gone; returns the pids it stopped.

    ``trainer_pid`` limits the sweep to the workers of that trainer.  Each
    orphan's group gets SIGTERM, which a worker answers by deleting its env sessions and unwinding, then SIGKILL
    after ``grace_seconds``; orphans are stopped together, so the sweep takes at most about ``grace_seconds``.
    Files of workers that are gone, or whose pid now names another process, are dropped.
    """

    orphans: list[tuple[str, WorkerRecord, int | None]] = []
    for path, record in _records(directory):
        if record is None:
            _drop(path)
            continue
        if trainer_pid is not None and record.trainer_pid != trainer_pid:
            continue
        if _running(record.trainer_pid, record.trainer_start_time):
            continue  # its trainer still owns it
        if not _running(record.pid, record.start_time):
            _drop(path)
            continue
        orphans.append((path, record, _open_pidfd(record.pid)))
    stopped: list[int] = []
    try:
        live = [(path, record, pidfd) for path, record, pidfd in orphans if _still_ours(record, pidfd)]
        for _, record, pidfd in live:
            logger.warning(
                "Stopping rollout worker %s (request %s): its trainer %s is gone",
                record.pid,
                record.request_id or "?",
                record.trainer_pid,
            )
            _signal_worker(record, pidfd, signal.SIGTERM)
        deadline = time.monotonic() + max(0.0, grace_seconds)
        for _, record, pidfd in live:
            if not _wait_for_exit(record, pidfd, deadline - time.monotonic()):
                logger.warning("Rollout worker %s outlived its %.0fs grace; killing it", record.pid, grace_seconds)
                _signal_worker(record, pidfd, signal.SIGKILL)
                _wait_for_exit(record, pidfd, 5.0)
            stopped.append(record.pid)
        for path, _, _ in orphans:
            _drop(path)
    finally:
        for _, _, pidfd in orphans:
            if pidfd is not None:
                try:
                    os.close(pidfd)
                except OSError:
                    pass
    return stopped


def _drop(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _open_pidfd(pid: int) -> int | None:
    if not hasattr(os, "pidfd_open"):  # pragma: no cover - Linux >= 5.3
        return None
    try:
        return os.pidfd_open(pid)
    except OSError:
        return None


def _still_ours(record: WorkerRecord, pidfd: int | None) -> bool:
    # Checked after the pidfd was opened: the pidfd then names the very process whose start time matched.
    return (pidfd is not None or not hasattr(os, "pidfd_open")) and _running(record.pid, record.start_time)


def _signal_worker(record: WorkerRecord, pidfd: int | None, sig: int) -> None:
    """Signal the worker's process group, while the worker (its leader) still runs, and the worker itself."""

    if record.pgid == record.pid and _running(record.pid, record.start_time):
        try:
            os.killpg(record.pgid, sig)
        except OSError:
            pass
    try:
        if pidfd is not None:
            signal.pidfd_send_signal(pidfd, sig)
        elif _running(record.pid, record.start_time):  # pragma: no cover - no pidfd support
            os.kill(record.pid, sig)
    except OSError:
        pass


def _wait_for_exit(record: WorkerRecord, pidfd: int | None, timeout: float) -> bool:
    """Wait up to ``timeout`` seconds for the worker to stop running; True once it has."""

    timeout = max(0.0, timeout)
    if pidfd is not None:
        poller = select.poll()
        poller.register(pidfd, select.POLLIN)
        if poller.poll(timeout * 1000):
            return True
        return not _running(record.pid, record.start_time)
    deadline = time.monotonic() + timeout  # pragma: no cover - no pidfd support
    while _running(record.pid, record.start_time):  # pragma: no cover
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True  # pragma: no cover
