"""A limit on the rollouts starting at once in a training run (``train.max_starting_rollouts``, experimental).

A ``GroupRolloutWorkflow`` member takes a start slot once it holds its rollout permit, before its request is built and
its worker launched, and gives it back at the first of three events: its rollout tree's first model call is issued,
the rollout ends, or the slot's timer fires (``worker_startup_budget_seconds`` plus :data:`START_SLOT_GRACE_SECONDS`).
So at most ``limit`` rollouts set up their workers, sandboxes and harnesses at once, training's and evaluation's
together, first come, first served.  While the member's block runs, its slot is :func:`held_slot`, which the
rollout's host gives back at the first call.  The trainer builds one :class:`StartLimit` per start of training and sets
it as the ``start_limit`` of its training and evaluation workflows.  A slot holder waits for nothing, so a start-up must
not wait for another rollout of the run.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import Counter, deque
from contextvars import ContextVar, Token
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["NO_START_SLOT", "START_RELEASE_CAUSES", "START_SLOT_GRACE_SECONDS", "StartLimit", "StartSlot", "held_slot"]

#: Seconds beyond ``worker_startup_budget_seconds`` that a slot is held without a model call before its timer gives it
#: back (600 s in all by default): longer than any start-up measured, so the timer only keeps the limit live when a
#: start-up hangs or its harness calls no model through platoon's gateway.
START_SLOT_GRACE_SECONDS = 300.0
#: Why a slot came back (``rollout_client/start_limit/releases/<cause>``): its rollout's first model call was issued,
#: the rollout ended, the slot's timer fired, or the rollout was cancelled while it held the slot (a straggler cut, a
#: stop).
START_RELEASE_CAUSES = ("first_call", "end", "timer", "cancelled")
#: Slots given back, cancelled rollouts' aside, before the run warns that most came back only at their rollout's end or
#: on the timer.
_RELEASES_BEFORE_WARNING = 32
# The slot of the member whose block runs in this context (:func:`held_slot`).
_current: ContextVar[StartSlot | None] = ContextVar("platoon_start_slot", default=None)


def held_slot() -> StartSlot | None:
    """The start slot of the member whose block runs in this context, given back or not; None outside one and without a
    start limit, whose null slot sets nothing."""

    return _current.get()


class StartLimit:
    """At most ``limit`` rollouts starting at once in this trainer, granted first come, first served (its own queue of
    waiters, oldest first).  ``held``: the slots held now; ``releases``: the slots given back so far, by cause
    (:data:`START_RELEASE_CAUSES`)."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.held = 0
        self.releases: Counter[str] = Counter()
        self._waiters: deque[asyncio.Future[None]] = deque()
        self._warned: set[str] = set()

    def slot(self, *, timeout_seconds: float, monitor: Any) -> StartSlot:
        """One rollout's slot, ``async with limit.slot(...) as start:``, given back on its timer after
        ``timeout_seconds`` without a model call; ``monitor``, the client monitor of the rollout's stats scope, records
        its wait and its release (``rollout_client/start_limit/...``)."""

        return StartSlot(self, timeout_seconds, monitor)

    def announce(self, *, largest_group: int, rollouts_at_once: int, gated: bool) -> None:
        """Say, as training starts, what the limit does to this run: groups of up to ``largest_group`` rollouts, up to
        ``rollouts_at_once`` rollouts running (the permits of training and of evaluation during training), and whether
        the training workflow takes start slots at all (``gated``)."""

        logger.info(
            "Rollout start-ups are limited to %d at once (train.max_starting_rollouts), training's and evaluation's "
            "together",
            self.limit,
        )
        if not gated:
            logger.warning(
                "train.max_starting_rollouts has no effect on training: only GroupRolloutWorkflow members take start "
                "slots, and the training workflow is not one"
            )
        if self.limit < largest_group:
            logger.warning(
                "train.max_starting_rollouts (%d) is below the largest group size (%d): a group's members start in "
                "waves; each keeps its full straggler grace; set it to at least %d unless start-up is the bottleneck",
                self.limit,
                largest_group,
                largest_group,
            )
        if self.limit >= rollouts_at_once:
            logger.info(
                "train.max_starting_rollouts (%d) can never bind: no more than %d rollouts run at once",
                self.limit,
                rollouts_at_once,
            )

    def _grant(self) -> None:
        """Hand the free slots to the waiters, oldest first; a cancelled waiter takes none."""

        while self._waiters and self.held < self.limit:
            waiter = self._waiters.popleft()
            if not waiter.done():
                self.held += 1
                waiter.set_result(None)

    def _released(self, cause: str, timeout_seconds: float) -> None:
        self.held -= 1
        self.releases[cause] += 1
        self._grant()
        if cause == "timer" and "timer" not in self._warned:
            self._warned.add("timer")
            logger.warning(
                "A rollout held its start slot for %.0fs without a model call, so the start limit released it on its "
                "timer: its harness may make no model call through platoon's gateway (platoon-oracle, platoon-noop, a "
                "custom rollout function), or its start-up takes longer than the timer, which "
                "workflow_config.worker_startup_budget_seconds lengthens (warned once per start)",
                timeout_seconds,
            )
        late = self.releases["end"] + self.releases["timer"]
        total = late + self.releases["first_call"]  # a cancelled rollout's slot says nothing of its start-up
        if total >= _RELEASES_BEFORE_WARNING and 2 * late > total and "end" not in self._warned:
            self._warned.add("end")
            logger.warning(
                "Most rollouts returned their start slot only at their end or on its timer (%d of %d): they made no "
                "model call through platoon's gateway, or failed before it, so train.max_starting_rollouts bounds "
                "whole rollouts for them; turn it off for this workload, or see the failures (warned once per start)",
                late,
                total,
            )


class StartSlot:
    """One rollout's slot of a :class:`StartLimit`.

    ``async with`` waits for it: at once while a slot is free and nobody waits, without yielding, and then ``waited``,
    the seconds it waited, is exactly 0.0.  It arms the slot's timer, is :func:`held_slot` while the block
    runs, and gives the slot back as the block exits (``end``, or ``cancelled`` for a block that a cancellation ends)
    unless :meth:`first_call` or the timer gave it back first; once, whichever comes first.  A wait that is cancelled
    takes nothing, and a slot granted at the instant of its cancellation goes to the next waiter.  The block exits in
    the task that entered it, as ``async with`` does.
    """

    def __init__(self, limit: StartLimit, timeout_seconds: float, monitor: Any) -> None:
        self.waited = 0.0
        self._limit, self._timeout, self._monitor = limit, timeout_seconds, monitor
        self._loop: asyncio.AbstractEventLoop | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._token: Token[StartSlot | None] | None = None
        self._held = False

    async def __aenter__(self) -> StartSlot:
        limit, monitor = self._limit, self._monitor
        loop = self._loop = asyncio.get_running_loop()
        if limit._waiters or limit.held >= limit.limit:
            waiter: asyncio.Future[None] = loop.create_future()
            limit._waiters.append(waiter)
            began = loop.time()
            with monitor.start_slot_wait():
                try:
                    await waiter
                except BaseException:
                    if waiter.done() and not waiter.cancelled():  # granted as it was cancelled: it passes on
                        limit.held -= 1
                        limit._grant()
                    else:
                        with contextlib.suppress(ValueError):
                            limit._waiters.remove(waiter)
                    raise
            self.waited = loop.time() - began
        else:
            limit.held += 1
        self._held = True
        self._token = _current.set(self)
        self._timer = loop.call_later(self._timeout, self._release, "timer")
        monitor.start_slot_taken(self.waited)
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._release("cancelled" if isinstance(exc, asyncio.CancelledError) else "end")
        if self._token is not None:
            _current.reset(self._token)

    def first_call(self) -> None:
        """The rollout's first model call was issued: give the slot back, on its event loop, from whichever thread the
        call came; nothing once the loop has closed."""

        loop = self._loop
        if loop is None:
            return
        try:
            on_loop = asyncio.get_running_loop() is loop
        except RuntimeError:  # no loop runs in this thread
            on_loop = False
        if on_loop:
            self._release("first_call")
        else:
            with contextlib.suppress(RuntimeError):  # the loop has closed, and the slot with it
                loop.call_soon_threadsafe(self._release, "first_call")

    def _release(self, cause: str) -> None:
        if not self._held:
            return
        self._held = False
        if self._timer is not None:
            self._timer.cancel()
        self._monitor.start_slot_released(cause)
        self._limit._released(cause, self._timeout)


class _NoStartSlot(StartSlot):
    """The slot when there is no start limit: it enters at once, without yielding, having waited exactly 0.0, holds
    nothing, is nobody's :func:`held_slot`, and its first call does nothing."""

    def __init__(self) -> None:
        self.waited = 0.0

    async def __aenter__(self) -> StartSlot:
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        pass

    def first_call(self) -> None:
        pass


# The slot of every rollout when there is no start limit: it has no state, so one serves them all.
NO_START_SLOT: StartSlot = _NoStartSlot()
