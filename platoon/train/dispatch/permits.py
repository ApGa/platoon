"""The rollout permits a training run's draws share (RFC 0020).

``train.max_concurrent_rollouts`` is a :class:`PermitPool`.  The dispatcher reserves a draw's group size of permits
(:meth:`PermitPool.reserve`) and starts the draw once the reservation is :meth:`~DrawLease.ready`.  The pool grants
reservations first come, first served, and serves the next one only once the first has all of its permits, so at most
one draw is partly granted and a large draw waiting first is never overtaken: a ``rolling`` reservation (a workflow
that runs its rollouts on the lease, under ``train.permit_release: rollout``) is granted one permit at a time and can
start on the first; any other is granted whole.

The reservation is the draw's :class:`DrawLease`, the :class:`~platoon.train.leases.RolloutLease` its workflow runs
its rollouts on: when ``active`` (``permit_release: rollout``), each ``rollout(i)`` block takes a granted permit and
returns it as it exits.  Calling the lease is the dispatcher's ``release``.

The rows of a phase of a run (``train`` or ``eval``) and their budget (``_rows``, ``_Rows``): each row's entry, label
and group size, whose permits a group reserves, and how many groups of a phase run at once.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import operator
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any

from platoon.train import entry_settings
from platoon.train.components import row_entry
from platoon.train.labels import label_column, task_label

logger = logging.getLogger(__name__)

__all__ = ["DrawLease", "PermitPool"]


class PermitPool:
    """Rollout permits that draws of different sizes share, granted first come, first served (module docstring).
    ``on_held``, when set, is told the permits held after every change (``rollout_client/permits_held_mean``)."""

    def __init__(self, permits: int) -> None:
        self.capacity = permits
        self._free = permits
        self._queue: deque[DrawLease] = deque()  # waiting for grants, oldest first; only the first partly granted
        self.on_held: Callable[[int], None] | None = None

    @property
    def free(self) -> int:
        return self._free

    @property
    def held(self) -> int:
        return self.capacity - self._free

    def reserve(self, count: int, *, rolling: bool = False, active: bool = False) -> DrawLease:
        """Reserve ``count`` permits behind every reservation that waits, granting what can be granted now."""

        lease = DrawLease(self, count, rolling=rolling, active=active)
        self._queue.append(lease)
        self._grant()
        return lease

    def _put_back(self, count: int) -> None:
        self._free += count
        self._grant()

    def _grant(self) -> None:
        while self._queue:
            head = self._queue[0]
            need = head.size - head.granted
            grant = min(need, self._free)
            if grant > 0 and (head.rolling or grant == need):
                self._free -= grant
                head._granted(grant)
            if head.granted < head.size:
                break
            self._queue.popleft()
        if self.on_held is not None:
            with contextlib.suppress(Exception):  # a metric never breaks dispatch
                self.on_held(self.held)


class DrawLease:
    """A draw's reservation of ``size`` permits of a :class:`PermitPool`, and the lease its workflow runs its rollouts
    on.

    Calling it releases the draw, once its workflow has returned, raised or was cancelled: every permit it still holds
    comes back, the rest of its reservation is withdrawn, and ``on_release`` runs (the dispatcher frees the draw's label
    slot).  ``name`` names the draw in errors and warnings."""

    def __init__(self, pool: PermitPool, size: int, *, rolling: bool, active: bool) -> None:
        self._pool = pool
        self.size = size
        self.rolling = rolling
        self.active = active
        self.on_release: Callable[[], None] | None = None
        self.name: str | None = None
        self.granted = 0  # granted so far, the returned ones included
        self._returned = 0
        self._unused = 0  # granted and taken by no block
        self.closed = False  # withdrawn: it holds nothing any more
        self._ready: asyncio.Future[None] | None = None
        self._waiters: deque[asyncio.Future[bool]] = deque()  # blocks waiting for a grant, oldest first
        self._used: set[int] = set()  # indices whose block entered and did not give up its wait
        self._warned = False  # of a block that found the draw released

    @property
    def held(self) -> int:
        return self.granted - self._returned

    @property
    def startable(self) -> bool:
        """Whether its draw can start: all its permits granted, or with rolling grants the first of them."""

        return self.granted >= (1 if self.rolling else self.size)

    async def ready(self) -> None:
        """Wait until the draw can start; return without yielding when it can, or once it is withdrawn."""

        if self.startable or self.closed:
            return
        if self._ready is None or self._ready.done():
            self._ready = asyncio.get_running_loop().create_future()
        await self._ready

    def resize(self, size: int) -> None:
        """Reserve ``size`` permits instead, before the draw starts (an adaptive decision's group size): those granted
        above it go back, and a draw that needs more waits first in line, being the one the dispatcher starts."""

        excess = min(self._unused, max(0, self.granted - size))
        self.size, self.granted, self._unused = size, self.granted - excess, self._unused - excess
        self._pool._free += excess
        if self.granted < size and not self.closed and self not in self._pool._queue:
            self._pool._queue.appendleft(self)
        self._pool._grant()

    def withdraw(self) -> None:
        """Every permit it holds comes back, the rest of its reservation is withdrawn, and its waiting blocks get no
        permit; once.  Also for a draw that will not start."""

        if self.closed:
            return
        self.closed = True
        with contextlib.suppress(ValueError):
            self._pool._queue.remove(self)
        held, self._returned, self._unused = self.held, self.granted, 0
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.set_result(False)
        if self._ready is not None and not self._ready.done():
            self._ready.cancel()
        self._pool._put_back(held)

    def rollout(self, index: int) -> AbstractAsyncContextManager[float]:
        index = operator.index(index)
        if not 0 <= index < self.size:
            raise IndexError(
                f"rollout {index} of task {self.name} is outside its draw's lease, of size {self.size}: a draw runs "
                "each rollout index from 0 to its size less one"
            )
        return _Block(self, index)

    def __call__(self) -> None:
        if self.closed:
            return
        self.withdraw()
        if self.on_release is not None:
            self.on_release()

    def _granted(self, count: int) -> None:
        self.granted += count
        self._unused += count
        self._hand_out()
        if self._ready is not None and not self._ready.done() and self.startable:
            self._ready.set_result(None)

    def _hand_out(self) -> None:
        while self._unused and self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                self._unused -= 1
                waiter.set_result(True)

    def _take_now(self) -> bool:
        """Take a granted permit that no block uses, if there is one."""

        if self.closed or not self._unused:
            return False
        self._unused -= 1
        return True

    async def _wait_take(self) -> bool:
        """Wait for a grant behind the draw's earlier blocks; False once the draw is released.  A cancelled wait takes
        nothing: a permit granted to it as it was cancelled goes to the next block."""

        if self.closed:
            return False
        waiter: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        try:
            return await waiter
        except BaseException:
            if waiter.done() and not waiter.cancelled():
                if waiter.result() and not self.closed:  # granted as it was cancelled: the next block takes it
                    self._unused += 1
                    self._hand_out()
            else:
                with contextlib.suppress(ValueError):
                    self._waiters.remove(waiter)
            raise

    def _give_back(self) -> None:
        if not self.closed:
            self._returned += 1
            self._pool._put_back(1)

    def _warn_released(self, index: int, how: str) -> None:
        if not self._warned:
            self._warned = True
            logger.warning(
                "Rollout %d of task %s %s its draw was released: it runs on no permit, so more rollouts can run at "
                "once than train.max_concurrent_rollouts. A workflow awaits each of its rollouts before it returns "
                "(warned once per draw)",
                index,
                self.name,
                how,
            )


class _Block:
    """``async with lease.rollout(index):``.  On an active lease it waits for a granted permit no other block uses, runs
    on it and returns it as it exits; its value is the seconds it waited."""

    __slots__ = ("_holds", "_index", "_lease")

    def __init__(self, lease: DrawLease, index: int) -> None:
        self._lease, self._index, self._holds = lease, index, False

    @property
    def holds(self) -> bool:
        """Whether it runs on a permit, which it returns as it exits: inside the block of an active lease."""

        return self._holds

    async def __aenter__(self) -> float:
        lease, index = self._lease, self._index
        if index in lease._used:
            raise RuntimeError(
                f"rollout {index} of task {lease.name} entered a second lease block: a draw runs each rollout index "
                "once, and a rollout's retry stays inside its block"
            )
        if lease.closed:
            lease._warn_released(index, "entered its lease block after")
            return 0.0
        lease._used.add(index)
        if not lease.active:
            return 0.0
        if lease._take_now():
            self._holds = True
            return 0.0  # exactly: members granted together have no start delay
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            self._holds = await lease._wait_take()
        except BaseException:
            lease._used.discard(index)  # it never ran: its index is free again
            raise
        if not self._holds:
            lease._warn_released(index, "was waiting for its permit when")
        return loop.time() - started

    async def __aexit__(self, *exc: object) -> bool:
        if self._holds:
            self._holds = False
            self._lease._give_back()
        return False


@dataclass(frozen=True)
class _Rows:
    """What the dispatcher, and an evaluation's rollout workers, read from each row they run: the row's environment
    entry (one of ``entries``), its label (its ``column`` value) and its entry's group size (``group_sizes``, else
    ``group_size``), whose rollout permits (``permits``) the group reserves before it starts."""

    entries: tuple[str, ...]
    column: str
    group_sizes: Mapping[str, int]
    group_size: int
    permits: PermitPool

    def of(self, data: Mapping[str, Any]) -> tuple[str | None, str | None, int]:
        entry = row_entry(data, self.entries)
        return entry, task_label(data, self.column), self.group_sizes.get(entry, self.group_size)


def _group_sizes(config: Any, phase: str) -> tuple[dict[str, int], int]:
    """The group size of each environment entry of a phase (``train`` or ``eval``) of a run that sets its own, by
    label, and the run's."""

    settings = entry_settings.run_settings(config)
    sizes = {
        label: getattr(own, phase).group_size for label, own in settings.items() if getattr(own, phase) is not None
    }
    return sizes, int(getattr(config, phase).workflow_config.group_size)


def _groups_at_once(config: Any, phase: str, budget: int | None) -> tuple[int, int]:
    """The budget of rollouts of a phase of a run (``max_concurrent_rollouts``, else its group size) and how many of
    its groups run at once at most: as many as it holds groups of the smallest entry's size, each taking its group's
    size in permits, so the budget is never exceeded."""

    sizes, group_size = _group_sizes(config, phase)
    budget = budget if budget is not None else group_size
    return budget, max(1, budget // max(1, min([*sizes.values(), group_size])))


def _rows(config: Any, phase: str, budget: int | None) -> tuple[_Rows, int]:
    """The rows of a phase (``train`` or ``eval``) of a run, with the rollout permits of its budget, and how many of
    its groups run at once at most (:func:`_groups_at_once`): an evaluation runs as many rollout workers."""

    sizes, group_size = _group_sizes(config, phase)
    budget, groups = _groups_at_once(config, phase, budget)
    rows = _Rows(
        entries=tuple(entry_settings.run_settings(config)),
        column=label_column(config),
        group_sizes=sizes,
        group_size=group_size,
        permits=PermitPool(budget),
    )
    return rows, groups


def _rollouts_at_once(config: Any, phase: str, budget: int | None) -> float:
    """The most rollouts of a phase of a run that run at once, which the client metrics compare the rollouts in flight
    with: its budget, or as many groups of the largest entry's size as it holds groups of the smallest's, when that
    is fewer; the budget itself under ``train.permit_release: rollout``, whose rollouts return their permits as they
    end."""

    sizes, group_size = _group_sizes(config, phase)
    budget, groups = _groups_at_once(config, phase, budget)
    if phase == "train" and getattr(config.train, "permit_release", "group") == "rollout":
        return float(budget)
    return float(min(budget, groups * max([*sizes.values(), group_size])))
