"""What a process still has to end when it stops, and the cleanups it lets finish.

**Process groups.**  A command started in a session of its own (``start_new_session=True``) -- a harness CLI on the
rollout host (:mod:`platoon.harness.runner`), a sandbox command (:mod:`platoon.sandbox._exec`) -- is out of reach of
a signal to the process group of the process that started it, so it outlives that group's SIGKILL, the way a stopping
rollout worker ends (:mod:`platoon.train.tinker.workers.entrypoint`).  The code that starts one registers its group
here (:func:`register_process_group`) until it has ended it, and a stopping process ends the groups still registered:
:func:`end_process_groups` (SIGTERM, a grace, SIGKILL), or :func:`kill_process_groups` (SIGKILL, also from a signal
handler).  :func:`register_descendant_groups` adds the groups of the sessions this process's descendants started
themselves (a harness's own commands), which nothing registered.  A group is identified by its id and its leader's
start time, so a recycled pid is never signalled; a group whose leader is gone is signalled while it still has
members, since Linux reuses no pid that is still a process group's id.

**Cleanups.**  A task that tears something down -- an env session's teardown, a command's SIGTERM, grace and SIGKILL --
is marked with :func:`let_finish`.  A stopping rollout worker cancels every other task of its rollout, again and again
for tasks that swallow the cancellation, but lets these run to their end within its deadline (:func:`cleanups`).
Each also joins the :class:`CleanupScope` of the rollout whose code started it, if one is open, as does a teardown
that waits, unbounded, for env code still running (:func:`in_cleanup_scope`): an in-process rollout's permit waits for
its scope's cleanups, a bounded time, before it comes back (``GroupRolloutWorkflow._run_group_member``).

Stdlib only.  The process-group functions take no lock, so a signal handler may call them.
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from contextvars import ContextVar, Token
from typing import Any, TypeVar

__all__ = [
    "CleanupScope",
    "cleanups",
    "end_process_groups",
    "in_cleanup_scope",
    "kill_process_groups",
    "let_finish",
    "process_start_time",
    "register_descendant_groups",
    "register_process_group",
    "registered_process_groups",
    "signal_process_group",
    "unregister_process_group",
]

T = TypeVar("T")

#: Registered process groups: pgid -> the start time of its leader (``/proc/<pid>/stat``), None when unknown.
_groups: dict[int, int | None] = {}
#: How often :func:`end_process_groups` looks whether the groups it signalled have exited.
_POLL_SECONDS = 0.02
_GONE = (b"Z", b"X", b"x")  # a zombie or a dead process runs no code


# ----------------------------------------------------------------------------- process groups


def register_process_group(pgid: int) -> None:
    """Note a process group this process started in a session of its own, until :func:`unregister_process_group`."""

    stat = _stat(pgid)
    _groups[pgid] = stat[3] if stat is not None else None


def unregister_process_group(pgid: int) -> None:
    """Forget a group once it has been ended (an unknown one is ignored)."""

    _groups.pop(pgid, None)


def registered_process_groups() -> list[int]:
    """The process groups registered and not yet ended."""

    return list(_groups)


def kill_process_groups() -> list[int]:
    """SIGKILL every registered process group and forget them; returns the groups that still had members.

    No lock, no wait: safe in a signal handler, such as a rollout worker's alarm.
    """

    killed = []
    for pgid, start_time in list(_groups.items()):
        if _signal_group(pgid, start_time, signal.SIGKILL):
            killed.append(pgid)
        _groups.pop(pgid, None)
    return killed


def end_process_groups(grace_s: float) -> list[int]:
    """SIGTERM every registered process group, give them ``grace_s`` to exit, then SIGKILL what is left of them.

    Synchronous, so nothing interrupts it between the two signals; returns the groups that still had members.
    The groups are forgotten.
    """

    groups = list(_groups.items())
    signalled = [(pgid, start_time) for pgid, start_time in groups if _signal_group(pgid, start_time, signal.SIGTERM)]
    deadline = time.monotonic() + max(0.0, grace_s)
    running = {pgid for pgid, _ in signalled}
    while running and time.monotonic() < deadline:
        time.sleep(_POLL_SECONDS)
        running &= _running_groups()
    for pgid, start_time in signalled:
        if pgid in running:
            _signal_group(pgid, start_time, signal.SIGKILL)
    for pgid, _ in groups:
        _groups.pop(pgid, None)
    return [pgid for pgid, _ in signalled]


def register_descendant_groups() -> list[int]:
    """Register the groups of this process's descendants that are in neither its own group nor a registered one.

    These are the sessions or groups that processes it started set up themselves -- a harness CLI's own commands,
    a sandbox runtime's -- which no code here registered.  Only descendants still attached to this process are
    found: a process whose parent exited belongs to init (or a subreaper) from then on.  Returns the groups added.
    """

    table = _processes()
    children: dict[int, list[int]] = {}
    for pid, (_, ppid, _, _) in table.items():
        children.setdefault(ppid, []).append(pid)
    own = os.getpgrp()
    added: list[int] = []
    pending, seen = list(children.get(os.getpid(), ())), set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        pending.extend(children.get(pid, ()))
        state, _, pgid, _ = table[pid]
        if pgid in (own, 0) or pgid in _groups or state in _GONE:
            continue
        leader = table.get(pgid)
        _groups[pgid] = leader[3] if leader is not None else None
        added.append(pgid)
    return added


def process_start_time(pid: int) -> int | None:
    """The start time of process ``pid`` (clock ticks since boot, ``/proc/<pid>/stat``); None for no such process.

    With the pid, it identifies a process even after the pid is reused: a group recorded with it can be signalled
    later, by another process too (:func:`signal_process_group`)."""

    stat = _stat(pid)
    return stat[3] if stat is not None else None


def signal_process_group(pgid: int, start_time: int | None, signum: int) -> bool:
    """Send ``signum`` to process group ``pgid`` unless its id now names another process's group: the leader's
    start time must be ``start_time`` while the leader exists.  Whether any member got it.  No lock: safe in a
    signal handler."""

    return _signal_group(pgid, start_time, signum)


def _signal_group(pgid: int, start_time: int | None, signum: int) -> bool:
    """Send ``signum`` to a registered group unless its id now names another process's; whether any member got it."""

    if pgid <= 1:
        return False
    if start_time is not None:
        leader = _stat(pgid)
        if leader is not None and leader[3] != start_time:
            return False  # the pid was free, so the group was gone, and another process has it now
    try:
        os.killpg(pgid, signum)
    except OSError:  # no member left, or none we may signal
        return False
    return True


def _stat(pid: int) -> tuple[bytes, int, int, int] | None:
    """State, parent pid, process group and start time of ``pid`` (``/proc/<pid>/stat``); None for no such process.

    Raw reads, no buffered file: also called from signal handlers.
    """

    try:
        descriptor = os.open(f"/proc/{pid}/stat", os.O_RDONLY)
    except OSError:
        return None
    try:
        raw = os.read(descriptor, 4096)
    except OSError:
        return None
    finally:
        os.close(descriptor)
    try:
        fields = raw[raw.rindex(b")") + 2 :].split()
        return fields[0], int(fields[1]), int(fields[2]), int(fields[19])
    except (ValueError, IndexError):
        return None


def _processes() -> dict[int, tuple[bytes, int, int, int]]:
    """Every process ``/proc`` lists: pid -> its :func:`_stat`."""

    try:
        names = os.listdir("/proc")
    except OSError:
        return {}
    table = {}
    for name in names:
        if name.isdigit():
            stat = _stat(int(name))
            if stat is not None:
                table[int(name)] = stat
    return table


def _running_groups() -> set[int]:
    """The process groups with a member that still runs; every registered one when ``/proc`` cannot be read."""

    table = _processes()
    if not table:
        return set(_groups)
    return {pgid for state, _, pgid, _ in table.values() if state not in _GONE}


# ----------------------------------------------------------------------------- cleanups

_cleanups: set[asyncio.Task[Any]] = set()


class CleanupScope:
    """The cleanups that one rollout's code starts and leaves running in the background (:func:`let_finish`,
    :func:`in_cleanup_scope`), so that what holds the rollout's resources can wait for them (:meth:`finish`).

    ``with scope:`` opens it for the code that runs in the block, and for the tasks that code creates, which copy
    the context, for as long as they run: a teardown that one of them starts after the block, as an env's that waits
    for the env's code to end does, joins it too.  Only tasks of the event loop it was created in join it.
    """

    def __init__(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._token: Token[CleanupScope | None] | None = None

    def __enter__(self) -> CleanupScope:
        self._token = _scope.set(self)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._token is not None:
            _scope.reset(self._token)
            self._token = None

    def add(self, task: asyncio.Task[Any]) -> None:
        if task.get_loop() is self._loop and not task.done():
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    def running(self) -> list[asyncio.Task[Any]]:
        """Its cleanups still running."""

        return [task for task in list(self._tasks) if not task.done()]

    async def finish(self, timeout_s: float) -> int:
        """Wait until its cleanups have ended, the ones they start meanwhile included, for at most ``timeout_s``;
        returns how many still run.  Cancelling the caller ends the wait and cancels no cleanup."""

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while (running := self.running()) and (left := deadline - loop.time()) > 0:
            await asyncio.wait(running, timeout=left)
        return len(self.running())


#: The cleanup scope of the rollout whose code runs in this context (:class:`CleanupScope`).
_scope: ContextVar[CleanupScope | None] = ContextVar("platoon_cleanup_scope", default=None)


def let_finish(task: asyncio.Task[T]) -> asyncio.Task[T]:
    """Mark ``task`` as a cleanup that a stopping process lets run to its end, and keep it referenced until then.  It
    joins the open cleanup scope too (:func:`in_cleanup_scope`).

    Its outcome is still its creator's to retrieve.  Returns ``task``.
    """

    _cleanups.add(task)
    task.add_done_callback(_cleanups.discard)
    return in_cleanup_scope(task)


def in_cleanup_scope(task: asyncio.Task[T]) -> asyncio.Task[T]:
    """Add ``task`` to the cleanup scope open in this context, if any (:class:`CleanupScope`), without marking it as
    a cleanup that a stopping process lets finish (:func:`let_finish`): for a teardown whose wait has no bound of its
    own, which a stopping rollout worker cancels.  Returns ``task``."""

    scope = _scope.get()
    if scope is not None:
        scope.add(task)
    return task


def cleanups(loop: asyncio.AbstractEventLoop | None = None) -> list[asyncio.Task[Any]]:
    """The cleanup tasks still running (:func:`let_finish`), of ``loop`` when one is given."""

    return [task for task in list(_cleanups) if not task.done() and (loop is None or task.get_loop() is loop)]
