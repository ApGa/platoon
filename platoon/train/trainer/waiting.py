"""What a start waits for: a call, with heartbeats and a deadline; a rollout group, with watchdog heartbeats; the next
settled group, or a shutdown.

``_await_with_heartbeat`` takes the trainer first and is a method of every trainer still (on ``TrainerBase``): to
replace it, replace the method, on the class or an instance, not the function.  The other two take no trainer: a rollout
source (:mod:`platoon.train.dispatch.source`) calls them, and so do the rl trainer's loops, through its static methods.
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Protocol

from platoon.train.trainer.signals import TrainInterrupted, Watchdog

if TYPE_CHECKING:
    from platoon.train.trainer.api import TrainerBase

logger = logging.getLogger(__name__)


# Watchdog heartbeat cadence while a rollout group is in flight (rollouts can
# outlast the watchdog timeout: rollout timeout + straggler grace + judge tail).
ROLLOUT_HEARTBEAT_INTERVAL_S = 60.0


async def _await_with_heartbeat(
    trainer: "TrainerBase",
    coro,
    name: str,
    heartbeat_interval: float = 60.0,
    watchdog: Watchdog | None = None,
    total_timeout: float | None = None,
):
    """Await a coroutine with periodic heartbeat logging.

    Useful for long-running async operations where we want visibility
    that we're still waiting. Also sends watchdog heartbeats to prevent
    the process from being killed during long operations.

    Args:
        coro: The coroutine to await
        name: Name for logging purposes
        heartbeat_interval: Seconds between heartbeat logs
        watchdog: Optional watchdog to send heartbeats to
        total_timeout: Optional total timeout in seconds (raises TimeoutError if exceeded)
    """
    start_time = time.perf_counter()
    task = asyncio.ensure_future(coro)

    while not task.done():
        wait = heartbeat_interval
        # Check total timeout
        if total_timeout is not None:
            elapsed = time.perf_counter() - start_time
            if elapsed >= total_timeout:
                task.cancel()
                raise TimeoutError(f"{name} timed out after {elapsed:.0f}s (limit: {total_timeout}s)")
            wait = min(wait, total_timeout - elapsed)  # the deadline fires on time, not at the next heartbeat

        try:
            # Wait for either the task to complete or the heartbeat interval
            await asyncio.wait_for(asyncio.shield(task), timeout=wait)
        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - start_time
            logger.info(f"Still waiting for {name}... ({elapsed:.0f}s elapsed)")
            # Send watchdog heartbeat to prevent timeout during long operations
            if watchdog is not None:
                watchdog.heartbeat()

    return task.result()


async def _await_rollout_with_heartbeat(
    coro,
    watchdog: Watchdog | None,
    heartbeat_interval: float | None = None,
):
    """Await an in-flight rollout group while feeding the watchdog, a training group's (its rollout source runs it)
    or an evaluation's.

    A group is bounded by the per-member rollout timeout plus the straggler
    grace and the judge/reward tail, which can exceed the watchdog timeout;
    without heartbeats during collection the watchdog kills the client at
    startup and after every restart before any group completes.  Unlike
    :meth:`_await_with_heartbeat` this does not shield the rollout, so a
    cancellation (interrupt checkpoint) still propagates to the group and
    its straggler members.
    """
    if heartbeat_interval is None:
        heartbeat_interval = ROLLOUT_HEARTBEAT_INTERVAL_S
    task = asyncio.ensure_future(coro)
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=heartbeat_interval)
            if watchdog is not None:
                watchdog.heartbeat()
    except asyncio.CancelledError:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise
    return task.result()


class _Results(Protocol):
    """What a step dequeues settled groups from: a start's result queue, or a policy's share queue."""

    async def get(self) -> Any: ...
    def get_nowait(self) -> Any: ...
    def empty(self) -> bool: ...


async def _next_rollout(
    queue: _Results, shutdown_event: asyncio.Event | None, *, interruptible: bool, ready: bool = False
):
    """The next rollout result from ``queue``, where a start's training groups settle; with ``interruptible`` (no
    update of the current batch applied yet), ``TrainInterrupted`` once a shutdown is requested (``shutdown_event``;
    None: never) instead, since the feeders stop on it and the batch may never complete.  The batch is then dropped
    whole: a resume from the last completed step redoes it.
    With ``ready`` (a step under a group admission rule), a result already queued is taken without letting
    other tasks run first: a step takes the queued markers in one go, so that groups that end at once cannot start
    again in the slots those markers free and queue more markers between two of them.
    """

    if ready and not queue.empty():
        if interruptible and shutdown_event is not None and shutdown_event.is_set():
            raise TrainInterrupted()
        return queue.get_nowait()
    if not interruptible or shutdown_event is None:
        return await queue.get()
    if shutdown_event.is_set():
        raise TrainInterrupted()
    get = asyncio.ensure_future(queue.get())
    stop = asyncio.ensure_future(shutdown_event.wait())
    try:
        await asyncio.wait({get, stop}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        stop.cancel()
        if not get.done():
            get.cancel()
    if get.done() and not get.cancelled():
        return get.result()
    raise TrainInterrupted()
