"""The dispatcher with rollout permits that come back as each rollout ends (``train.permit_release: rollout``, RFC
0020), its groups running each member in a block of their lease as ``GroupRolloutWorkflow`` does: what the dispatcher
adds to the pool (``test_permit_pool.py``).  Groups of mixed sizes start in the order they were drawn with at most one
partly granted, a stop while a draw waits for permits behaves as before, an adaptive decision comes on the first grant
under rolling starts, a drawn group larger than its decision held takes its other permits after the decision, and the
blocked shares agree with the dispatcher's own waits."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_dispatcher_samplers import Chooser, _mixture, _SizedRows
from test_permit_pool import _quiet

from platoon.train.components import ENVIRONMENT_COLUMN
from platoon.train.labels import task_label
from platoon.train.leases import holding, rollout_lease
from platoon.train.sampler_runtime import SamplerSet
from platoon.train.tinker import dispatcher as dispatcher_module
from platoon.train.tinker.dispatcher import BoundDraw, Dispatcher
from platoon.train.tinker.permits import DrawLease, PermitPool


class _Rows:
    """What the dispatcher reads of a row: no entry, its ``env`` label, its ``size`` (default 1); and the permits."""

    def __init__(self, permits: int) -> None:
        self.permits = PermitPool(permits)

    def of(self, data: Mapping[str, Any]) -> tuple[str | None, str | None, int]:
        return None, task_label(data, "env"), int(data.get("size", 1))


class _Members:
    """A dispatcher under ``permit_release: rollout`` whose groups run each member in a block of the group's lease, as
    ``GroupRolloutWorkflow`` does, until the test ends it (:meth:`end`)."""

    def __init__(self, dispatcher: Dispatcher, *, rolling: bool) -> None:
        self.dispatcher = dispatcher
        dispatcher.rolling = rolling  # as the trainer sets it from the workflow's runs_rollouts_on_lease
        dispatcher.bind(asyncio.Queue())
        self.started: list[BoundDraw] = []
        self.leases: list[DrawLease] = []
        self.running: set[tuple[int | None, int]] = set()
        self.gates: dict[tuple[tuple[int, int], int], asyncio.Event] = {}
        self.tasks: list[asyncio.Task] = []
        self.loop: asyncio.Task | None = None

    @classmethod
    def over(cls, rows: list[dict[str, Any]], *, permits: int, rolling: bool, **kwargs: Any) -> _Members:
        kwargs.setdefault("batch_size", 4)
        kwargs.setdefault("max_staleness", None)
        kwargs.setdefault("steps", 100)
        order = lambda position: rows[position % len(rows)]  # noqa: E731
        return cls(Dispatcher(rows=_Rows(permits), order=order, permit_release="rollout", **kwargs), rolling=rolling)

    def run(self, stop: asyncio.Event | None = None) -> None:
        self.loop = asyncio.ensure_future(self.dispatcher.run(spawn=self._spawn, run_group=self._group, stop=stop))
        self.tasks.append(self.loop)

    def _spawn(self, coroutine) -> asyncio.Task:
        self.tasks.append(task := asyncio.ensure_future(coroutine))
        return task

    async def _group(self, draw: BoundDraw, lease: DrawLease) -> None:
        self.started.append(draw)
        self.leases.append(lease)
        try:
            with holding(lease):  # as the trainer's runner does
                members = [asyncio.ensure_future(self._member(draw, index)) for index in range(draw.group_size)]
                try:
                    await asyncio.gather(*members)
                finally:
                    for task in members:
                        task.cancel()
                    await asyncio.gather(*members, return_exceptions=True)
        finally:
            lease()
        self.dispatcher.settle(draw, draw)

    async def _member(self, draw: BoundDraw, index: int) -> None:
        async with rollout_lease().rollout(index):
            self.running.add((draw.position, index))
            try:
                await self.gates.setdefault((draw.key, index), asyncio.Event()).wait()
            finally:
                self.running.discard((draw.position, index))

    def end(self, draw: BoundDraw, *members: int) -> None:
        for index in members or range(draw.group_size):
            self.gates.setdefault((draw.key, index), asyncio.Event()).set()

    @property
    def pool(self) -> PermitPool:
        return cast(PermitPool, self.dispatcher._rows.permits)

    def partly_granted(self) -> int:
        return sum(1 for lease in self.leases if not lease.closed and 0 < lease.granted < lease.size)

    def members_of(self, position: int | None) -> list[int]:
        return sorted(index for where, index in self.running if where == position)

    async def close(self) -> None:
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


def _rows(*sizes: int) -> list[dict[str, Any]]:
    return [{"task_id": f"t{index}", "env": "x", "size": size} for index, size in enumerate(sizes)]


@pytest.mark.parametrize("rolling", [False, True], ids=["co-start", "rolling"])
def test_mixed_sizes_start_in_the_order_they_were_drawn_as_permits_trickle_back(rolling):
    """Groups of 3, 1, 2, 3 and 1 with 4 permits, whose members end one at a time: the groups start in the order they
    were drawn, a group of one never overtakes the group of three that waits first, at most one group is partly
    granted at a time (under co-start none is), and the budget holds."""

    async def run():
        harness = _Members.over(_rows(3, 1, 2, 3, 1), permits=4, rolling=rolling, steps=2)
        harness.run()
        await _quiet()
        assert [draw.position for draw in harness.started] == [0, 1]
        ends = [(0, 0), (1, 0), (0, 1), (0, 2), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0)]
        for position, member in ends:
            harness.end(next(draw for draw in harness.started if draw.position == position), member)
            await _quiet()
            assert harness.partly_granted() <= (1 if rolling else 0)
            assert len(harness.running) <= harness.pool.held <= harness.pool.capacity
        assert [draw.position for draw in harness.started] == [0, 1, 2, 3, 4, 5, 6]  # the eighth waits for 2
        await harness.close()

    asyncio.run(run())


@pytest.mark.parametrize("rolling", [False, True], ids=["co-start", "rolling"])
def test_a_stop_while_a_draw_waits_for_permits_starts_no_group_and_keeps_the_draw(rolling):
    async def run():
        harness = _Members.over(_rows(4), permits=4, rolling=rolling, batch_size=3, steps=1)
        stop = asyncio.Event()
        harness.run(stop)
        await _quiet()
        (first,) = harness.started
        stop.set()
        if rolling:
            harness.end(first, 0)  # its first permit comes after the request
        else:
            harness.end(first)
        await _quiet()
        assert harness.started == [first]  # it does not start
        assert harness.dispatcher.state()["waiting"] == {"x": [[0, 1]]}  # but waits, first in line, for a resume
        assert harness.loop is not None and harness.loop.done()
        assert harness.pool.free == (1 if rolling else 4)  # what it was granted came back
        await harness.close()

    asyncio.run(run())


def _adaptive(permits: int, *, decided: Mapping[str, int], drawn: Mapping[str, int]) -> tuple[Dispatcher, SamplerSet]:
    """A dispatcher under ``permit_release: rollout`` whose pool-wide adaptive sampler sees groups of ``decided`` sizes
    by entry, and whose rows run groups of ``drawn`` sizes."""

    sampling = _mixture(run=Chooser, sizes=decided)
    samplers = SamplerSet.build(sampling, batch_size=4, label_column=ENVIRONMENT_COLUMN)
    dispatcher = Dispatcher(
        rows=_SizedRows(permits, drawn),
        samplers=samplers,
        batch_size=4,
        max_staleness=None,
        steps=100,
        labels=["a", "b"],
        sampler=sampling.run.name,
        permit_release="rollout",
    )
    return dispatcher, samplers


@pytest.mark.parametrize("rolling", [False, True], ids=["co-start", "rolling"])
def test_an_adaptive_decision_comes_on_the_first_grant_under_rolling_starts_and_on_all_of_them_under_co_start(rolling):
    """A pool-wide adaptive sampler holds the permits of the largest group (4) for each decision, and its draws are
    groups of 2: under co-start the second decision waits for 4 free permits, as before; under rolling starts it comes
    as soon as the first is granted, and the reservation then takes the drawn group's size."""

    async def run():
        sizes = {"a": 2, "b": 4}
        dispatcher, samplers = _adaptive(4, decided=sizes, drawn=sizes)
        harness = _Members(dispatcher, rolling=rolling)
        harness.run()
        await _quiet()
        chooser = samplers.run.sampler
        if rolling:
            assert len(chooser.asked) == 2 and [draw.group_size for draw in harness.started] == [2, 2]
            assert harness.pool.free == 0  # the next decision waits for a grant
        else:
            assert len(chooser.asked) == 1 and harness.pool.free == 2  # the second waits for 4 free permits
        harness.end(harness.started[0], 0)
        await _quiet()
        assert len(chooser.asked) == (3 if rolling else 1)
        await harness.close()

    asyncio.run(run())


@pytest.mark.parametrize("rolling", [False, True], ids=["co-start", "rolling"])
def test_a_drawn_group_larger_than_its_decision_held_takes_its_other_permits_after_the_decision(rolling):
    """The samplers say groups of 2 while the rows run groups of 4, so each decision holds 2 of the 6 permits and its
    drawn group takes 2 more after it, first in line: under co-start the second group starts once its other 2 come
    back, under rolling starts at once, its other members starting as permits free; the budget holds throughout."""

    async def run():
        dispatcher, _samplers = _adaptive(6, decided={"a": 2, "b": 2}, drawn={"a": 4, "b": 4})
        harness = _Members(dispatcher, rolling=rolling)
        harness.run()
        await _quiet()
        first = harness.started[0]
        assert harness.members_of(first.position) == [0, 1, 2, 3] and harness.pool.held == 6
        assert len(harness.started) == (2 if rolling else 1)
        if rolling:
            assert harness.members_of(harness.started[1].position) == [0, 1]  # on its 2 permits, 2 to come
        for count, member in enumerate((0, 1), start=1):
            harness.end(first, member)
            await _quiet()
            second = harness.started[1] if len(harness.started) > 1 else None
            running = harness.members_of(second.position) if second is not None else []
            assert running == ([0, 1, 2, 3][: 2 + count] if rolling else [0, 1, 2, 3] if count == 2 else [])
            assert len(harness.running) <= harness.pool.held <= harness.pool.capacity
        await harness.close()

    asyncio.run(run())


def test_the_blocked_shares_agree_with_the_dispatcher_s_own_waits(monkeypatch):
    """On a clock the test moves: the dispatcher waits 5 s for a permit, 3 s on the horizon (k = 0), and once the
    run's last groups have started, on the end of its plan; each read covers the time since the last one."""

    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(dispatcher_module, "time", SimpleNamespace(monotonic=lambda: clock.now))

    def shares(**reasons: float) -> dict[str, float]:
        return {
            f"dispatch/blocked_share/{reason}": reasons.get(reason, 0.0) for reason in dispatcher_module.BLOCKED_REASONS
        }

    async def run():
        harness = _Members.over(_rows(1), permits=1, rolling=True, batch_size=2, max_staleness=0, steps=2)
        harness.run()
        await _quiet()
        first = harness.started[0]  # the second draw waits for the one permit
        clock.now += 5.0
        harness.end(first)
        await _quiet()
        second = harness.started[1]  # the horizon (k = 0, a batch of 2) holds the third back
        clock.now += 3.0
        assert harness.dispatcher.blocked_shares() == pytest.approx(shares(permits=5 / 8, horizon=3 / 8))
        harness.end(second)
        await _quiet()
        for _ in range(2):
            harness.dispatcher.dequeued()
        harness.dispatcher.step_started()
        await _quiet()
        third = harness.started[2]  # the fourth waits for its permit
        clock.now += 2.0
        harness.end(third)
        await _quiet()  # the fourth starts: the plan's last group
        clock.now += 6.0
        assert harness.dispatcher.blocked_shares() == pytest.approx(shares(permits=2 / 8, plan=6 / 8))
        await harness.close()

    asyncio.run(run())
