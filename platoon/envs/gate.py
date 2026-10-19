"""Admission to one live env: the order in which tool calls, grading, forks and teardown reach it.

Agents that share a live env share its :class:`ToolGate`: the root, its inline children, verifiers and the
children a harness mints over HTTP all reach an ORS lease (with its ``shared`` views and ``platoon_fork``
children) or a Harbor sandbox lease through the same gate.  Each operation asks for a grant of one class:

* ``serial``, every tool unless the env says otherwise: one at a time per live env, since an env's tools are not
  assumed safe to overlap.
* ``concurrent``, tools an env declares safe to overlap: they run beside serial and other concurrent calls.
* ``exclusive``, declared terminal tools, grading that reads the env's state, copy snapshots, fork hooks and the
  first tool listing: they wait for everything in flight and run alone, so nothing changes the state they read.

**Order.**  Each operation takes a number when it arrives.  Each *caller* (one agent: the one the tool plane
names with :func:`on_behalf_of`, else the session view that makes the call) has at most one waiter in the shared
queue, and its next call joins the back when the previous one is granted.  So calls are FIFO within one agent
and take turns across agents: one agent's gather of many calls does not hold off a sibling's single call.  A
concurrent call may start ahead of a serial call that only waits for the serial slot.  An exclusive operation is
a barrier in arrival order, whichever caller's line it waits in: it starts once every operation that arrived
before it, from any caller, has ended or left the queue, and nothing that arrives after it starts before it
ends.  So a stream of calls cannot starve it, and it never overtakes a call that arrived first.

**Deadline.**  A waiter not granted within ``queue_timeout_s`` (default 240 s, below the 300 s per-call deadline
of OpenHands' MCP client) gets :class:`EnvBusy` and never runs.  MCP clients that time out do not tell the
server, so without the deadline a call the model was told had failed would still run when its turn came.

**Cancellation.**  A caller cancelled while it waits leaves the queue, and its call never runs.  Once granted,
:meth:`ToolGate.run` runs the operation as its own task, shielded from the caller, and releases the grant when
that task ends: a call running in a worker thread or on an env server finishes before the next one starts,
even when its client gave up.

**No deadlocks.**  A grant covers one env operation.  Platoon holds none across a model call, a spawn or another
tool call, and env code never calls back into platoon (a re-entrant acquire raises instead of waiting for
itself).  The one stall left is a tool that waits for another call's effect (a wait/set pair): it waits until
its own timeout unless both tools are declared concurrent.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import itertools
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Hashable
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, get_args

CallClass = Literal["serial", "concurrent", "exclusive"]
CALL_CLASSES: tuple[str, ...] = get_args(CallClass)
DEFAULT_QUEUE_TIMEOUT_S = 240.0

T = TypeVar("T")

_GRANTED, _BUSY, _CLOSED = "granted", "busy", "closed"
_DEFAULT: Any = object()  # timeout: the gate's queue_timeout_s
# The gates whose grant the running code holds, to refuse a re-entrant acquire that would wait for itself.
_HELD: contextvars.ContextVar[tuple[ToolGate, ...]] = contextvars.ContextVar("platoon_env_gates_held", default=())
# The agent on whose behalf the running code calls into envs (see on_behalf_of).
_CALLER: contextvars.ContextVar[Hashable | None] = contextvars.ContextVar("platoon_env_caller", default=None)


class EnvBusy(Exception):
    """The operation was not admitted within the queue deadline: it never ran."""

    def __init__(self, waited_s: float, timeout_s: float) -> None:
        super().__init__(f"not run: env busy (waited {waited_s:.1f} s for its turn, the limit is {timeout_s:g} s)")
        self.waited_s = waited_s
        self.timeout_s = timeout_s


class GateClosed(Exception):
    """The live env is closed: it admits nothing more."""


@dataclass(eq=False)
class _Waiter:
    call_class: CallClass
    view: Hashable | None
    future: asyncio.Future[str]
    enqueued_at: float
    number: int  # arrival order on its gate
    timer: asyncio.TimerHandle | None = None
    waited_s: float = 0.0


class Grant:
    """The right to run one operation on the live env; :meth:`release` gives it back (idempotent, synchronous, so
    a done-callback can call it)."""

    def __init__(self, gate: ToolGate, call_class: CallClass, waited_s: float) -> None:
        self.gate = gate
        self.call_class: CallClass = call_class
        self.waited_s = waited_s  # time spent queued
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self.gate._finish(self.call_class)
        self.gate._dispatch()


class ToolGate:
    """One live env's admission queue (see the module docstring).

    ``queue_timeout_s`` bounds how long an operation waits for its grant (None: no bound); ``timeout=`` on
    :meth:`acquire`, :meth:`hold` and :meth:`run` overrides it per operation.
    """

    def __init__(self, *, queue_timeout_s: float | None = DEFAULT_QUEUE_TIMEOUT_S) -> None:
        self.queue_timeout_s = check_queue_timeout(queue_timeout_s)
        self._queue: list[_Waiter] = []  # the shared queue, in the order waiters joined it; at most one per view
        self._lines: dict[Hashable, deque[_Waiter]] = {}  # per view: its waiters, the head is in _queue
        self._exclusive_waiters: list[_Waiter] = []  # in arrival order, wherever they wait: the barriers
        self._arrivals = itertools.count()
        self._serial = False
        self._concurrent = 0
        self._exclusive = False
        self._closed = False

    # ------------------------------------------------------------------ state

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def idle(self) -> bool:
        """Nothing runs and nothing waits."""
        return not (self._serial or self._concurrent or self._exclusive or self._queue)

    @property
    def waiting(self) -> int:
        return len(self._waiters())

    # ------------------------------------------------------------------ admission

    async def acquire(
        self, call_class: CallClass, *, view: Hashable | None = None, timeout: float | None = _DEFAULT
    ) -> Grant:
        """Wait for a grant of ``call_class``; ``view`` is the fairness key: one caller, normally an agent.

        Raises :class:`EnvBusy` when the deadline passes first and :class:`GateClosed` when the env closes;
        either way nothing was granted.  The caller must release the grant (or use :meth:`hold`/:meth:`run`).
        """
        if call_class not in CALL_CLASSES:
            raise ValueError(f"Unknown call class {call_class!r}; expected one of {CALL_CLASSES}")
        if self._closed:
            raise GateClosed("The env is closed")
        if self in _HELD.get():
            raise RuntimeError(
                "An operation on this env tried to start another one on the same env from inside itself; it would "
                "wait for itself forever (env code must not call back into its own session)"
            )
        timeout = self.queue_timeout_s if timeout is _DEFAULT else check_queue_timeout(timeout)
        loop = asyncio.get_running_loop()
        waiter = _Waiter(call_class, view, loop.create_future(), loop.time(), next(self._arrivals))
        self._enqueue(waiter)
        self._dispatch()
        if not waiter.future.done() and timeout is not None:
            waiter.timer = loop.call_later(timeout, self._expire, waiter)
        try:
            outcome = await waiter.future
        except asyncio.CancelledError:
            self._abandon(waiter)
            raise
        finally:
            if waiter.timer is not None:
                waiter.timer.cancel()
        if outcome == _BUSY:
            raise EnvBusy(waiter.waited_s, timeout if timeout is not None else 0.0)
        if outcome == _CLOSED:
            raise GateClosed("The env closed while the operation waited for its turn")
        return Grant(self, call_class, waiter.waited_s)

    @contextlib.asynccontextmanager
    async def hold(
        self, call_class: CallClass, *, view: Hashable | None = None, timeout: float | None = _DEFAULT
    ) -> AsyncGenerator[Grant, None]:
        """Hold a grant for the block.  The caller's cancellation reaches the block, so use it only for work
        that stops when cancelled (no worker threads); :meth:`run` covers the rest."""
        grant = await self.acquire(call_class, view=view, timeout=timeout)
        token = _HELD.set((*_HELD.get(), self))
        try:
            yield grant
        finally:
            _HELD.reset(token)
            grant.release()

    async def run(
        self,
        call_class: CallClass,
        operation: Callable[[Grant], Awaitable[T]],
        *,
        view: Hashable | None = None,
        timeout: float | None = _DEFAULT,
        inflight: set[asyncio.Task[Any]] | None = None,
    ) -> T:
        """Wait for a grant, then run ``operation(grant)`` as a task that holds it until the task ends.

        The task is shielded: a caller cancelled after the grant gets ``CancelledError`` while the operation runs
        to completion, and only then does the next operation start.  ``inflight`` collects the running tasks
        (for a teardown that drains them).
        """
        grant = await self.acquire(call_class, view=view, timeout=timeout)
        context = contextvars.copy_context()
        context.run(_HELD.set, (*_HELD.get(), self))
        try:
            task = asyncio.get_running_loop().create_task(_awaited(operation(grant)), context=context)
        except BaseException:
            grant.release()
            raise
        task.add_done_callback(functools.partial(_settle, grant))
        if inflight is not None:
            inflight.add(task)
            task.add_done_callback(inflight.discard)
        return await asyncio.shield(task)

    def close(self) -> None:
        """Admit nothing more: waiters get :class:`GateClosed`; grants already given run on.  Idempotent."""
        if self._closed:
            return
        self._closed = True
        waiters = self._waiters()
        self._queue.clear()
        self._lines.clear()
        self._exclusive_waiters.clear()
        for waiter in waiters:
            self._resolve(waiter, _CLOSED)

    # ------------------------------------------------------------------ queues

    def _waiters(self) -> list[_Waiter]:
        seen: dict[int, _Waiter] = {id(waiter): waiter for waiter in self._queue}
        for line in self._lines.values():
            seen.update((id(waiter), waiter) for waiter in line)
        return list(seen.values())

    def _enqueue(self, waiter: _Waiter) -> None:
        if waiter.call_class == "exclusive":
            self._exclusive_waiters.append(waiter)
        if waiter.view is None:
            self._queue.append(waiter)
            return
        line = self._lines.get(waiter.view)
        if line is None:
            self._lines[waiter.view] = deque([waiter])
            self._queue.append(waiter)
        else:
            line.append(waiter)  # behind its view's earlier calls; it joins the queue when they are granted

    def _leave(self, waiter: _Waiter) -> None:
        """Take a waiter out of the queues; when it headed its view's line, the view's next waiter joins the
        back of the shared queue."""
        if waiter in self._queue:
            self._queue.remove(waiter)
        if waiter in self._exclusive_waiters:
            self._exclusive_waiters.remove(waiter)
        if waiter.view is None:
            return
        line = self._lines.get(waiter.view)
        if line is None or waiter not in line:
            return
        was_head = line[0] is waiter
        line.remove(waiter)
        if not line:
            del self._lines[waiter.view]
        elif was_head:
            self._queue.append(line[0])

    def _dispatch(self) -> None:
        """Grant what the state allows, in queue order (see the module docstring for the rules)."""
        for waiter in self._waiters():
            if waiter.future.done():  # its caller was cancelled and has not resumed to leave the queue yet
                self._leave(waiter)
        # The first exclusive waiter: nothing that arrived after it may start before it ends.
        barrier = self._exclusive_waiters[0] if self._exclusive_waiters else None
        index = 0
        while index < len(self._queue) and not self._exclusive:
            waiter = self._queue[index]
            if barrier is not None and waiter.number > barrier.number:
                index += 1  # it arrived after an exclusive operation that still waits
            elif waiter is barrier:
                if self._serial or self._concurrent or min(other.number for other in self._queue) < waiter.number:
                    index += 1  # it waits for what runs, and for what arrived before it
                else:
                    self._grant(waiter)
            elif waiter.call_class == "serial" and self._serial:
                index += 1  # the serial slot is taken; concurrent calls behind it may still start
            else:
                self._grant(waiter)  # leaves the queue: the next waiter is now at ``index``

    def _grant(self, waiter: _Waiter) -> None:
        self._leave(waiter)
        if waiter.call_class == "serial":
            self._serial = True
        elif waiter.call_class == "concurrent":
            self._concurrent += 1
        else:
            self._exclusive = True
        self._resolve(waiter, _GRANTED)

    def _finish(self, call_class: CallClass) -> None:
        if call_class == "serial":
            self._serial = False
        elif call_class == "concurrent":
            self._concurrent -= 1
        else:
            self._exclusive = False

    def _resolve(self, waiter: _Waiter, outcome: str) -> None:
        if waiter.timer is not None:
            waiter.timer.cancel()
        if not waiter.future.done():
            waiter.waited_s = waiter.future.get_loop().time() - waiter.enqueued_at
            waiter.future.set_result(outcome)

    def _expire(self, waiter: _Waiter) -> None:
        if waiter.future.done():
            return
        self._leave(waiter)
        self._resolve(waiter, _BUSY)
        self._dispatch()  # an expired exclusive waiter no longer holds back the calls behind it

    def _abandon(self, waiter: _Waiter) -> None:
        """The caller was cancelled: leave the queue, or pass on a grant that raced the cancellation."""
        if waiter.future.cancelled():
            self._leave(waiter)
            self._dispatch()
        elif waiter.future.result() == _GRANTED:
            self._finish(waiter.call_class)
            self._dispatch()


@contextlib.contextmanager
def on_behalf_of(caller: Hashable) -> Generator[None, None, None]:
    """Queue the env operations the block starts as ``caller``'s (one agent): its fairness key on every gate.

    The tool plane names the agent of each call, so agents that share one session (children minted over HTTP
    share their parent's) still take turns; outside the block an operation queues as its session view's.
    """
    token = _CALLER.set(caller)
    try:
        yield
    finally:
        _CALLER.reset(token)


def caller_or(default: Hashable) -> Hashable:
    """The fairness key of an operation that starts now: the caller :func:`on_behalf_of` names, else ``default``."""
    caller = _CALLER.get()
    return default if caller is None else caller


def check_queue_timeout(value: float | None) -> float | None:
    """A queue deadline: a positive number of seconds, or None for none."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        raise ValueError(f"queue_timeout_s must be a positive number of seconds or None, got {value!r}")
    return float(value)


async def _awaited(awaitable: Awaitable[T]) -> T:
    return await awaitable


def _settle(grant: Grant, task: asyncio.Task[Any]) -> None:
    grant.release()
    if not task.cancelled():
        task.exception()  # retrieved: a caller that was cancelled no longer awaits the task


__all__ = [
    "CALL_CLASSES",
    "DEFAULT_QUEUE_TIMEOUT_S",
    "CallClass",
    "EnvBusy",
    "GateClosed",
    "Grant",
    "ToolGate",
    "caller_or",
    "check_queue_timeout",
    "on_behalf_of",
]
