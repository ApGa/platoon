"""The rollout permits a training run's draws share, and the lease each draw holds (``platoon.train.tinker.permits``,
RFC 0020): reservations granted first come first served, whole or one at a time; at most one draw partly granted; the
budget never exceeded; and a lease whose blocks take a permit each and return it as they exit, each rollout index once
and never more than its size, that holds nothing once released and loses nothing to a cancellation."""

from __future__ import annotations

import asyncio
import logging
import random

import pytest

from platoon.train import leases
from platoon.train.tinker.permits import DrawLease, PermitPool


async def _quiet() -> None:
    """Let every task run until nothing changes."""

    for _ in range(50):
        await asyncio.sleep(0)


class _Checked(PermitPool):
    """A pool that checks, whenever its grants change, that it holds no more than its budget and that no draw is
    granted anything while an older one still waits (so at most one is partly granted); ``problems`` lists what broke.
    """

    def __init__(self, permits: int) -> None:
        super().__init__(permits)
        self.leases: list[DrawLease] = []
        self.problems: list[str] = []
        self.on_held = self._check

    def reserve(self, count: int, *, rolling: bool = False, active: bool = False) -> DrawLease:
        lease = super().reserve(count, rolling=rolling, active=active)
        self.leases.append(lease)
        self._check(self.held)
        return lease

    def _check(self, held: int) -> None:
        if not 0 <= held <= self.capacity:
            self.problems.append(f"{held} held of {self.capacity}")
        waiting = [lease for lease in self.leases if not lease.closed and lease.granted < lease.size]
        if any(lease.granted for lease in waiting[1:]):
            self.problems.append(f"granted behind a waiting draw: {[(draw.granted, draw.size) for draw in waiting]}")


@pytest.fixture
def pool():
    """Pools that check their invariants (:class:`_Checked`), asserted once the test ends."""

    made: list[_Checked] = []

    def make(permits: int) -> _Checked:
        made.append(checked := _Checked(permits))
        return checked

    yield make
    assert not [problem for checked in made for problem in checked.problems]


# --- reservations ---------------------------------------------------------------------------------------------------


def test_a_whole_reservation_starts_only_once_its_whole_size_is_free(pool):
    """Co-start: with the pool's 8 permits held by a draw of 8, the next draw of 4 waits; one returned permit starts
    nothing, and the fourth starts it."""

    async def run():
        permits = pool(8)
        first = permits.reserve(8, active=True)
        assert first.startable and permits.free == 0
        waiting = permits.reserve(4)
        ready = asyncio.ensure_future(waiting.ready())
        blocks = [first.rollout(index) for index in range(4)]
        for block in blocks:
            await block.__aenter__()
        for returned, block in enumerate(blocks, start=1):
            await block.__aexit__(None, None, None)
            await _quiet()
            assert ready.done() == (returned == 4) and waiting.granted == (4 if returned == 4 else 0)
        first()
        waiting.withdraw()
        assert permits.free == 8

    asyncio.run(run())


def test_a_rolling_reservation_starts_on_one_free_permit_and_is_granted_the_rest_as_they_free(pool):
    async def run():
        permits = pool(8)
        first = permits.reserve(8, active=True)
        rolling = permits.reserve(4, rolling=True)
        ready = asyncio.ensure_future(rolling.ready())
        blocks = [first.rollout(index) for index in range(5)]
        for block in blocks:
            await block.__aenter__()
        await _quiet()
        assert not ready.done()
        for granted, block in enumerate(blocks, start=1):
            await block.__aexit__(None, None, None)
            await _quiet()
            assert ready.done() and rolling.granted == min(granted, 4)  # it starts on the first, never takes more
        assert permits.free == 1

    asyncio.run(run())


@pytest.mark.parametrize("rolling", [False, True], ids=["whole", "rolling"])
def test_reservations_are_granted_first_come_first_served_with_trickling_permits_and_mixed_sizes(pool, rolling):
    """Draws of 6, 2 and 4 wait behind a full pool of 8 whose permits come back one at a time: they are granted in
    the order they reserved, a small draw never overtakes a large one waiting first, and at most one is partly granted
    at a time (rolling grants) or none is (whole ones)."""

    async def run():
        permits = pool(8)
        first = permits.reserve(8, active=True)
        reservations = [permits.reserve(size, rolling=rolling) for size in (6, 2, 4)]
        started: list[int] = []
        for index, reservation in enumerate(reservations):
            asyncio.ensure_future(reservation.ready()).add_done_callback(
                lambda _done, index=index: started.append(index)
            )
        trace = []
        for index in range(8):
            block = first.rollout(index)
            await block.__aenter__()
            await block.__aexit__(None, None, None)
            await _quiet()
            trace.append(tuple(reservation.granted for reservation in reservations))
        if rolling:
            assert trace == [(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0), (6, 0, 0), (6, 1, 0), (6, 2, 0)]
        else:
            assert trace == [(0, 0, 0)] * 5 + [(6, 0, 0), (6, 0, 0), (6, 2, 0)]
        assert started == [0, 1]
        reservations[0].withdraw()  # its 6 permits back: the draw of 4 starts on them
        await _quiet()
        assert started == [0, 1, 2] and permits.free == 2

    asyncio.run(run())


def test_the_budget_is_never_exceeded_under_random_reservations_and_returns(pool):
    """Random draws of 1 to 8 permits, whole or rolling, reserve and return their permits in random order, granted
    blocks running and ending: every change keeps the pool within its budget with no draw granted while an older one
    waits (the fixture checks), and in the end every permit is back."""

    async def run():
        rng = random.Random(20261025)
        permits = pool(12)
        live: list[DrawLease] = []
        blocks: list[asyncio.Task] = []
        holds: list[asyncio.Event] = []

        async def rollout(lease: DrawLease, index: int, hold: asyncio.Event) -> None:
            async with lease.rollout(index):
                await hold.wait()

        for _ in range(400):
            action = rng.random()
            if action < 0.3:
                lease = permits.reserve(rng.randint(1, 8), rolling=rng.random() < 0.5, active=True)
                live.append(lease)
                for index in range(lease.size):
                    holds.append(hold := asyncio.Event())
                    blocks.append(asyncio.ensure_future(rollout(lease, index, hold)))
            elif action < 0.75 and holds:
                holds.pop(rng.randrange(len(holds))).set()
            elif live:
                live.pop(rng.randrange(len(live)))()  # the draw ends: what it still holds comes back
            await _quiet()
        for hold in holds:
            hold.set()
        for lease in live:
            lease()
        await asyncio.gather(*blocks, return_exceptions=True)
        assert permits.free == permits.capacity

    asyncio.run(run())


def test_a_reservation_withdrawn_while_it_waits_takes_nothing_and_one_granted_as_it_is_withdrawn_gives_back(pool):
    """An evaluation's group (reserve, wait until it can start, withdraw at the end) cancelled while it waits takes
    nothing, and one cancelled as its permits are granted gives them back."""

    async def group(permits: PermitPool, count: int) -> None:
        reservation = permits.reserve(count)
        try:
            await reservation.ready()
            await asyncio.sleep(10)
        finally:
            reservation.withdraw()

    async def run():
        permits = pool(4)
        first = permits.reserve(4)
        waiter = asyncio.ensure_future(group(permits, 2))
        await _quiet()
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        assert permits.free == 0
        granted = asyncio.ensure_future(group(permits, 2))
        await _quiet()
        first.withdraw()  # granted ...
        granted.cancel()  # ... as it was cancelled: the permits go back
        await asyncio.gather(granted, return_exceptions=True)
        assert permits.free == 4

    asyncio.run(run())


def test_the_pool_reports_the_permits_held():
    async def run():
        permits = PermitPool(6)
        held: list[int] = []
        permits.on_held = held.append
        lease = permits.reserve(4, rolling=True, active=True)
        async with lease.rollout(0):
            assert held[-1] == 4
        assert held[-1] == 3  # the block's permit came back
        lease()
        assert held[-1] == 0

    asyncio.run(run())


# --- the lease ------------------------------------------------------------------------------------------------------


def _lease(permits: PermitPool, size: int, *, rolling: bool = True, active: bool = True) -> DrawLease:
    lease = permits.reserve(size, rolling=rolling, active=active)
    lease.name = "t0"
    return lease


def test_a_block_runs_on_a_permit_and_returns_it_as_it_exits(pool):
    async def run():
        permits = pool(4)
        lease = _lease(permits, 4)
        async with lease.rollout(0) as waited:
            assert waited == 0.0 and lease.held == 4 and permits.free == 0
        assert lease.held == 3 and permits.free == 1
        lease()
        assert permits.free == 4 and lease.held == 0

    asyncio.run(run())


class _Index:
    """An index of another type that ``operator.index`` takes, as a numpy integer is."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value


@pytest.mark.parametrize("active", [True, False], ids=["rollout", "group"])
def test_a_lease_runs_each_index_once_within_its_size_and_fails_fast_otherwise(pool, active):
    """Under either setting, a block for an index outside the draw's size raises IndexError, and a second block for an
    index, while the first runs or once it ended, raises RuntimeError as it enters; any index ``operator.index`` takes
    serves, and a refused block takes no permit."""

    async def run():
        permits = pool(8)
        lease = _lease(permits, 2, active=active)
        for index in (2, -1):
            with pytest.raises(IndexError, match=f"rollout {index} of task t0 is outside"):
                lease.rollout(index)
        with pytest.raises(TypeError):
            lease.rollout("0")  # type: ignore[arg-type]
        async with lease.rollout(_Index(0)):  # type: ignore[arg-type]
            with pytest.raises(RuntimeError, match="rollout 0 of task t0 entered a second lease block"):
                async with lease.rollout(0):
                    pass
        with pytest.raises(RuntimeError, match="entered a second lease block"):
            async with lease.rollout(0):
                pass
        assert permits.free == (7 if active else 6)  # the draw's 2, one returned by the block that ran
        lease()
        assert permits.free == 8

    asyncio.run(run())


def test_blocks_that_find_their_draw_released_hold_nothing_and_warn_once_per_draw(pool, caplog):
    """A block still waiting for its permit when its draw is released, and one that enters after it, hold nothing and
    return nothing; each draw warns once, so a second draw warns again."""

    async def run():
        permits = pool(2)
        other = permits.reserve(2)
        lease = _lease(permits, 2)
        waiting = asyncio.ensure_future(lease.rollout(0).__aenter__())
        await _quiet()
        lease()  # the workflow returned with a block still waiting
        assert await waiting == pytest.approx(0.0, abs=1.0) and permits.free == 0
        async with lease.rollout(1):
            assert permits.free == 0
        other.withdraw()
        second = _lease(permits, 2)
        second()
        async with second.rollout(0):
            assert permits.free == 2
        second()  # releasing twice changes nothing
        assert permits.free == 2

    with caplog.at_level(logging.WARNING):
        asyncio.run(run())
    warnings = [record.getMessage() for record in caplog.records if "its draw was released" in record.getMessage()]
    assert len(warnings) == 2
    assert warnings[0].startswith("Rollout 0 of task t0 was waiting for its permit when its draw was released")
    assert warnings[1].startswith("Rollout 0 of task t0 entered its lease block after its draw was released")


def test_rolling_blocks_wait_for_grants_first_come_first_served(pool):
    """A draw of 3 granted one permit: its first block runs at once, the others wait, and each starts as a permit
    frees, in the order they entered; each reports how long it waited."""

    async def run():
        permits = pool(3)
        other = permits.reserve(2)  # another draw holds 2
        lease = _lease(permits, 3)
        assert lease.held == 1
        order: list[int] = []
        waited: dict[int, float] = {}
        gates = {index: asyncio.Event() for index in range(3)}

        async def member(index: int) -> None:
            async with lease.rollout(index) as seconds:
                order.append(index)
                waited[index] = seconds
                await gates[index].wait()

        tasks = [asyncio.ensure_future(member(index)) for index in (0, 2, 1)]
        await _quiet()
        assert order == [0]
        await asyncio.sleep(0.02)
        other.withdraw()  # its two permits come back to the first draw in line, ours
        await _quiet()
        assert order == [0, 2, 1] and waited[0] == 0.0 and min(waited[2], waited[1]) >= 0.015
        for gate in gates.values():
            gate.set()
        await asyncio.gather(*tasks)
        lease()
        assert permits.free == 3

    asyncio.run(run())


def test_a_block_cancelled_while_it_waits_for_a_grant_takes_nothing(pool):
    async def run():
        permits = pool(2)
        other = permits.reserve(2)
        lease = _lease(permits, 2)
        waiting = asyncio.ensure_future(lease.rollout(0).__aenter__())
        behind = asyncio.ensure_future(lease.rollout(1).__aenter__())
        await _quiet()
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        other.withdraw()  # both permits go to our draw: the cancelled block takes neither
        await _quiet()
        assert behind.done() and lease.held == 2
        async with lease.rollout(0) as waited:  # its index is free again, and a granted permit waits for it
            assert waited == 0.0
        lease()
        assert permits.free == 2

    asyncio.run(run())


def test_a_permit_granted_to_a_block_as_it_is_cancelled_goes_to_the_next_block(pool):
    async def run():
        permits = pool(1)
        other = permits.reserve(1)
        lease = _lease(permits, 2)
        first = asyncio.ensure_future(lease.rollout(0).__aenter__())
        second = asyncio.ensure_future(lease.rollout(1).__aenter__())
        await _quiet()
        other.withdraw()  # the permit is handed to the first block ...
        first.cancel()  # ... which is cancelled before it runs
        await asyncio.gather(first, return_exceptions=True)
        await _quiet()
        assert first.cancelled() and second.done() and lease.held == 1  # the second block runs on it
        lease()
        assert permits.free == 1

    asyncio.run(run())


@pytest.mark.parametrize("ending", ["raises", "cancelled"])
def test_a_workflow_that_fails_or_is_cancelled_returns_every_permit(pool, ending):
    """A workflow whose members run in blocks, one waiting for a grant, ends by raising or by cancellation: every
    block's permit comes back as it exits, the waiting one takes nothing, and the release returns the rest and
    withdraws what was never granted, so the next draw is granted at once."""

    async def run():
        permits = pool(3)
        other = permits.reserve(1)
        lease = _lease(permits, 3)
        assert lease.held == 2

        async def member(index: int) -> None:
            async with lease.rollout(index):
                await asyncio.sleep(10)

        async def workflow() -> None:
            members = [asyncio.ensure_future(member(index)) for index in range(3)]
            try:
                await _quiet()
                if ending == "raises":
                    raise RuntimeError("the group failed")
                await asyncio.gather(*members)
            finally:
                for task in members:
                    task.cancel()
                await asyncio.gather(*members, return_exceptions=True)

        task = asyncio.ensure_future(workflow())
        await _quiet()
        if ending == "cancelled":
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        lease()
        assert permits.free == 2 and permits.reserve(2).startable
        other.withdraw()

    asyncio.run(run())


def test_under_permit_release_group_a_lease_holds_nothing_of_its_own_and_its_permits_come_back_together(pool):
    async def run():
        permits = pool(4)
        lease = _lease(permits, 4, rolling=False, active=False)
        assert lease.size == 4
        for index in range(4):
            async with lease.rollout(index) as waited:
                assert waited == 0.0
        assert permits.free == 0  # its permits come back together, when its workflow returns
        lease()
        assert permits.free == 4

    asyncio.run(run())


def test_outside_a_training_draw_the_lease_is_a_no_op_of_size_0():
    async def run():
        lease = leases.rollout_lease()
        assert lease is leases.NO_LEASE and lease.size == 0
        async with lease.rollout(5) as waited:
            assert waited == 0.0

        async def read() -> leases.RolloutLease:
            return leases.rollout_lease()

        with leases.holding(_lease(PermitPool(2), 2)):
            assert leases.rollout_lease().size == 2
            assert (await asyncio.ensure_future(read())).size == 2  # a task the workflow creates copies it
            in_thread = await asyncio.get_running_loop().run_in_executor(None, leases.rollout_lease)
            assert in_thread is leases.NO_LEASE  # a thread does not see it, unless the context is copied
        assert leases.rollout_lease() is leases.NO_LEASE

    asyncio.run(run())
