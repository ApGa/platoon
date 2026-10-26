"""The start limit alone (``train.max_starting_rollouts``, :mod:`platoon.train.workflow.start_limit`).

- At most ``limit`` slots are held (checked at every grant), granted first come, first served, training's and
  evaluation's takers alike.
- A slot taken at once waits exactly 0.0 and does not yield; a queued one waits.  A slot is its block's
  ``held_slot()``.
- Each slot comes back once, by the first of its first call, its end and its timer; the timer grants the next waiter;
  a cancelled holder's comes back as ``cancelled``, which no warning counts.
- A cancelled waiter takes nothing, a slot granted at the instant of its cancellation passes on, and a fuzz of
  cancellations, failures, timers and first calls ends with nothing held or leaked.
- A first call from another thread gives the slot back on its loop, and does nothing once the loop has closed.
- The two warnings, once each; the metrics of a hand-computed run; the null slot.

Times are a virtual clock's where they matter: the loop's clock jumps to its next timer instead of sleeping.
"""

from __future__ import annotations

import asyncio
import logging
import random
import selectors
import threading
from types import SimpleNamespace

import pytest

from platoon.train.tinker import rollout_metrics
from platoon.train.tinker.rollout_metrics import ClientThroughputMonitor
from platoon.train.workflow.group import GroupRolloutWorkflow
from platoon.train.workflow.start_limit import NO_START_SLOT, StartLimit, held_slot

LOGGER = "platoon.train.workflow.start_limit"


class _JumpingSelector(selectors.DefaultSelector):
    """Never sleeps: when its loop would wait for its next timer, the loop's clock jumps there instead."""

    loop: _VirtualClockLoop

    def select(self, timeout=None):
        events = super().select(0)
        if not events and timeout:
            self.loop.now += timeout
        return events


class _VirtualClockLoop(asyncio.SelectorEventLoop):
    def __init__(self) -> None:
        selector = _JumpingSelector()
        super().__init__(selector)
        selector.loop = self
        self.now = 0.0

    def time(self) -> float:
        return self.now


def _virtual(coro):
    return asyncio.run(coro, loop_factory=_VirtualClockLoop)


def _monitor(scope: str = "test_start_limit") -> ClientThroughputMonitor:
    return ClientThroughputMonitor(scope)


def test_at_most_limit_slots_are_held_granted_first_come_first_served_across_training_and_evaluation():
    """10 takers of 3 slots, alternately training's and evaluation's, each holding its slot a few loop turns."""

    async def scenario():
        limit, granted = StartLimit(3), []
        monitors = {"train": _monitor("test_sl_train"), "eval": _monitor("test_sl_eval")}

        async def taker(index: int) -> None:
            async with limit.slot(timeout_seconds=60, monitor=monitors["eval" if index % 2 else "train"]):
                assert limit.held <= 3  # at its grant
                granted.append(index)
                for _ in range(1 + index % 3):
                    await asyncio.sleep(0)

        await asyncio.gather(*(taker(index) for index in range(10)))
        return limit, granted, monitors

    limit, granted, monitors = asyncio.run(scenario())
    assert granted == list(range(10)) and limit.held == 0 and not limit._waiters
    assert limit.releases == {"end": 10}
    ends = {
        scope: monitor.start_limit_metrics()["rollout_client/start_limit/releases/end"]
        for scope, monitor in monitors.items()
    }
    assert ends == {"train": 5.0, "eval": 5.0}  # each scope counts its own rollouts


def test_a_slot_taken_at_once_waits_exactly_zero_without_yielding_and_a_queued_one_waits():
    async def scenario():
        limit, monitor, ran = StartLimit(1), _monitor(), []
        asyncio.get_running_loop().call_soon(ran.append, "the loop turned")

        async def queued() -> float:
            async with limit.slot(timeout_seconds=600, monitor=monitor) as second:
                return second.waited

        async with limit.slot(timeout_seconds=600, monitor=monitor) as first:
            assert first.waited == 0.0 and ran == []
            second = asyncio.ensure_future(queued())
            await asyncio.sleep(2.5)
        return await second

    assert _virtual(scenario()) == 2.5


def test_a_slot_is_its_block_s_held_slot_given_back_or_not_and_the_null_slot_is_nobody_s():
    async def scenario():
        async with StartLimit(1).slot(timeout_seconds=600, monitor=_monitor()) as start:
            start.first_call()
            seen = (held_slot(), await asyncio.ensure_future(asyncio.to_thread(held_slot)))
        async with NO_START_SLOT:
            assert held_slot() is None
        return start, seen, held_slot()

    start, seen, after = asyncio.run(scenario())
    assert seen == (start, start) and after is None  # the block's tasks and threads see it too


def test_a_slot_comes_back_once_at_the_first_of_its_first_call_and_its_end():
    async def scenario():
        limit, monitor = StartLimit(1), _monitor()
        async with limit.slot(timeout_seconds=600, monitor=monitor) as start:
            start.first_call()
            assert limit.held == 0
            start.first_call()
        async with limit.slot(timeout_seconds=600, monitor=monitor) as start:
            pass
        start.first_call()  # after its end: nothing
        return limit, monitor

    limit, monitor = asyncio.run(scenario())
    assert limit.held == 0 and limit.releases == {"first_call": 1, "end": 1}
    metrics = monitor.start_limit_metrics()
    assert metrics["rollout_client/start_limit/releases/first_call"] == 1.0
    assert metrics["rollout_client/start_limit/releases/end"] == 1.0


def test_the_timer_gives_a_slot_back_grants_the_next_waiter_and_warns_once(caplog):
    """Timers of 60 s on a virtual clock: three rollouts that make no model call and run 100 s, on one slot."""

    async def scenario():
        limit, monitor, entered = StartLimit(1), _monitor(), []

        async def rollout() -> None:
            async with limit.slot(timeout_seconds=60, monitor=monitor) as start:
                entered.append((asyncio.get_running_loop().time(), start.waited))
                await asyncio.sleep(100)

        await asyncio.gather(rollout(), rollout(), rollout())
        return limit, entered

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        limit, entered = _virtual(scenario())
    assert entered == [(0.0, 0.0), (60.0, 60.0), (120.0, 120.0)]
    assert limit.releases == {"timer": 3} and limit.held == 0
    warnings = [record.getMessage() for record in caplog.records if "on its timer" in record.getMessage()]
    assert len(warnings) == 1 and "60s without a model call" in warnings[0]


def test_most_slots_coming_back_only_at_their_end_warns_once_from_32_releases_on(caplog):
    async def scenario(first_calls: int, ends: int) -> None:
        limit, monitor = StartLimit(4), _monitor()
        for index in range(first_calls + ends):
            async with limit.slot(timeout_seconds=600, monitor=monitor) as start:
                if index < first_calls:
                    start.first_call()

    def warnings() -> list[str]:
        return [record.getMessage() for record in caplog.records if "only at their end" in record.getMessage()]

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        asyncio.run(scenario(first_calls=12, ends=40))  # at the 32nd release, 20 came at the end
        assert len(warnings()) == 1 and "(20 of 32)" in warnings()[0]
        caplog.clear()
        asyncio.run(scenario(first_calls=16, ends=16))  # half, not more
        asyncio.run(scenario(first_calls=0, ends=31))  # fewer than 32 releases
        assert warnings() == []


def test_a_cancelled_holder_s_slot_comes_back_as_cancelled_and_no_warning_counts_it(caplog):
    """A run whose first 40 rollouts called the model is stopped while 64 hold their slots: those 64 come back as
    ``cancelled``, which says nothing of a start-up, so the run does not warn that most came back at their end."""

    async def scenario():
        limit, monitor = StartLimit(64), _monitor()
        for _ in range(40):
            async with limit.slot(timeout_seconds=600, monitor=monitor) as start:
                start.first_call()

        async def starting() -> None:
            async with limit.slot(timeout_seconds=600, monitor=monitor):
                await asyncio.sleep(3600)

        tasks = [asyncio.ensure_future(starting()) for _ in range(64)]
        await asyncio.sleep(0)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return limit, monitor

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        limit, monitor = asyncio.run(scenario())
    assert limit.releases == {"first_call": 40, "cancelled": 64} and limit.held == 0
    assert monitor.start_limit_metrics()["rollout_client/start_limit/releases/cancelled"] == 64.0
    assert caplog.records == []


def test_a_cancelled_waiter_takes_nothing_and_a_grant_at_the_instant_of_cancellation_passes_on():
    async def scenario():
        limit, monitor, entered = StartLimit(1), _monitor(), []
        holder = await limit.slot(timeout_seconds=600, monitor=monitor).__aenter__()

        async def waiter(name: str) -> None:
            async with limit.slot(timeout_seconds=600, monitor=monitor):
                entered.append(name)

        first, second, third = (asyncio.ensure_future(waiter(name)) for name in ("first", "second", "third"))
        await asyncio.sleep(0)
        first.cancel()  # while it waits
        await asyncio.sleep(0)
        assert first.cancelled() and len(limit._waiters) == 2
        holder.first_call()  # grants "second" ...
        second.cancel()  # ... which is cancelled before it resumes: "third" takes the slot
        await asyncio.gather(second, third, return_exceptions=True)
        return limit, entered, second

    limit, entered, second = asyncio.run(scenario())
    assert entered == ["third"] and second.cancelled()
    assert limit.held == 0 and not limit._waiters and limit.releases == {"first_call": 1, "end": 1}


@pytest.mark.parametrize("seed", range(40))
def test_random_cancellations_failures_timers_and_first_calls_leak_nothing(seed):
    rng = random.Random(seed)
    size = rng.randint(1, 3)

    async def scenario():
        limit, monitor, peak, got = StartLimit(size), _monitor(), [0], [0]

        async def rollout() -> None:
            async with limit.slot(timeout_seconds=rng.choice((0.5, 600)), monitor=monitor) as start:
                got[0] += 1
                peak[0] = max(peak[0], limit.held)
                for _ in range(rng.randint(0, 4)):
                    await asyncio.sleep(rng.choice((0, 0, 0.1)))
                if rng.random() < 0.3:
                    start.first_call()
                if rng.random() < 0.2:
                    raise RuntimeError("a rollout that fails")
                await asyncio.sleep(rng.choice((0, 1.0)))

        tasks = [asyncio.ensure_future(rollout()) for _ in range(12)]
        for _ in range(rng.randint(0, 6)):
            await asyncio.sleep(rng.choice((0, 0.05, 0.3)))
            rng.choice(tasks).cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return limit, peak[0], got[0]

    limit, peak, got = _virtual(scenario())
    assert limit.held == 0 and not limit._waiters and peak <= size
    assert sum(limit.releases.values()) == got  # every slot taken came back, once


def test_a_first_call_from_another_thread_gives_the_slot_back_on_its_loop_and_nothing_once_it_closed():
    errors: list[BaseException] = []

    def call(start) -> None:
        try:
            start.first_call()
        except BaseException as exc:  # noqa: BLE001 - the test reports it
            errors.append(exc)

    def in_a_thread(start) -> None:
        thread = threading.Thread(target=call, args=(start,))
        thread.start()
        thread.join()

    async def scenario():
        limit = StartLimit(1)
        async with limit.slot(timeout_seconds=600, monitor=_monitor()) as start:
            in_a_thread(start)
            assert limit.held == 1  # scheduled on the loop, not run yet
            await asyncio.sleep(0)
            assert limit.held == 0 and limit.releases == {"first_call": 1}
        return start

    start = asyncio.run(scenario())
    start.first_call()  # its loop has closed
    in_a_thread(start)
    assert errors == []


def test_the_metrics_of_a_hand_computed_run(monkeypatch):
    """Two slots and four rollouts on a virtual clock: 0 and 1 enter at once and call at 1 s and 3 s, so 2 and 3 take
    their slots then (waits 1 and 3 s); 2 calls at 4 s; 3 makes no call and ends at 6 s; the others end at 5 s.  Over
    8 s: 10 slot-seconds held (mean 1.25), 4 waited (mean 0.5), waits [0, 0, 1, 3] s, releases 3 first calls and 1
    end.  A scope that took no slot exports none of these."""

    async def scenario():
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(rollout_metrics, "time", SimpleNamespace(monotonic=loop.time))
        limit, monitor, idle = StartLimit(2), _monitor(), _monitor("test_sl_idle")

        async def rollout(first_call_at: float | None, end_at: float) -> None:
            async with limit.slot(timeout_seconds=600, monitor=monitor) as start:
                if first_call_at is not None:
                    await asyncio.sleep(first_call_at - loop.time())
                    start.first_call()
                await asyncio.sleep(end_at - loop.time())

        await asyncio.gather(rollout(1, 5), rollout(3, 5), rollout(4, 5), rollout(None, 6))
        await asyncio.sleep(8 - loop.time())
        return monitor.export(), idle.export()

    metrics, idle = _virtual(scenario())
    name = "rollout_client/start_limit"
    assert {key: value for key, value in metrics.items() if key.startswith(name)} == {
        f"{name}/starting_mean": pytest.approx(1.25),
        f"{name}/waiting_mean": pytest.approx(0.5),
        f"{name}/wait_seconds_p50": pytest.approx(0.5),
        f"{name}/wait_seconds_p90": pytest.approx(2.4),
        f"{name}/wait_seconds_max": pytest.approx(3.0),
        f"{name}/releases/first_call": 3.0,
        f"{name}/releases/end": 1.0,
        f"{name}/releases/timer": 0.0,
        f"{name}/releases/cancelled": 0.0,
    }
    assert not [key for key in idle if "start_limit" in key]


def test_the_null_slot_enters_at_once_waits_exactly_zero_and_holds_nothing():
    async def scenario():
        ran = []
        asyncio.get_running_loop().call_soon(ran.append, "the loop turned")
        async with NO_START_SLOT as start:
            assert start.waited == 0.0 and ran == []
            start.first_call()
        assert ran == []  # nor did its exit yield

    asyncio.run(scenario())
    assert GroupRolloutWorkflow.start_limit is None
