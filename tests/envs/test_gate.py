"""The live-env admission gate (``platoon.envs.gate``): classes, order, fairness, deadlines and cancellation."""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

import pytest

from platoon.envs.gate import EnvBusy, GateClosed, ToolGate, caller_or, on_behalf_of

pytestmark = pytest.mark.asyncio


class Recorder:
    """Runs named operations through a gate and logs when each starts and ends."""

    def __init__(self, gate: ToolGate) -> None:
        self.gate = gate
        self.log: list[tuple[str, str]] = []  # ("start" | "end", name)
        self.running: set[str] = set()
        self.peak = 0

    def started(self) -> list[str]:
        return [name for event, name in self.log if event == "start"]

    async def op(
        self,
        name: str,
        call_class: str = "serial",
        seconds: float = 0.0,
        until: asyncio.Event | None = None,
        **kwargs: Any,
    ) -> str:
        """Run ``name`` for ``seconds``, or until ``until`` is set."""

        async def body(grant):
            self.log.append(("start", name))
            self.running.add(name)
            self.peak = max(self.peak, len(self.running))
            try:
                await (until.wait() if until is not None else asyncio.sleep(seconds))
            finally:
                self.running.discard(name)
                self.log.append(("end", name))
            return name

        return await self.gate.run(call_class, body, **kwargs)  # type: ignore[arg-type]


async def started_in_order(*coroutines) -> list[asyncio.Task]:
    """Start each coroutine and let it reach the gate (it runs up to its first wait) before the next one starts."""
    tasks = []
    for coroutine in coroutines:
        tasks.append(asyncio.ensure_future(coroutine))
        for _ in range(3):
            await asyncio.sleep(0)
    return tasks


async def test_serial_calls_run_one_at_a_time_in_arrival_order():
    recorder = Recorder(ToolGate())
    tasks = await started_in_order(*(recorder.op(f"s{index}", seconds=0.01) for index in range(5)))
    assert await asyncio.gather(*tasks) == [f"s{index}" for index in range(5)]
    assert recorder.peak == 1 and recorder.started() == [f"s{index}" for index in range(5)]
    assert recorder.gate.idle


async def test_concurrent_calls_overlap_serial_calls_and_each_other_but_never_an_exclusive_one():
    recorder, release = Recorder(ToolGate()), asyncio.Event()
    tasks = await started_in_order(
        recorder.op("serial", until=release),
        recorder.op("c1", "concurrent", until=release),
        recorder.op("c2", "concurrent", until=release),
        recorder.op("submit", "exclusive"),
        recorder.op("c3", "concurrent"),
    )
    assert recorder.log == [("start", "serial"), ("start", "c1"), ("start", "c2")]  # three at once; the rest wait
    release.set()
    await asyncio.gather(*tasks)
    assert {name for _, name in recorder.log[3:6]} == {"serial", "c1", "c2"}  # they end ...
    assert recorder.log[6:] == [("start", "submit"), ("end", "submit"), ("start", "c3"), ("end", "c3")]


async def test_an_exclusive_call_waits_for_calls_ahead_of_it_and_nothing_behind_it_overtakes():
    recorder, release = Recorder(ToolGate()), asyncio.Event()
    tasks = await started_in_order(
        recorder.op("s1", until=release),
        recorder.op("s2"),
        recorder.op("submit", "exclusive"),
        recorder.op("s3"),
        recorder.op("c1", "concurrent"),
    )
    release.set()
    await asyncio.gather(*tasks)
    # Strict arrival order around the exclusive call: s2 (queued before it) is not overtaken, and the calls queued
    # after it, concurrent ones included, wait until it ends.
    assert recorder.started() == ["s1", "s2", "submit", "s3", "c1"]
    assert recorder.log.index(("end", "submit")) < recorder.log.index(("start", "s3"))


async def test_an_exclusive_call_is_a_barrier_in_arrival_order_whatever_line_it_waits_in():
    a, b, c = object(), object(), object()
    # Queued behind its own agent's earlier call, it still starts before a call another agent sent after it.
    recorder, release = Recorder(ToolGate()), asyncio.Event()
    tasks = await started_in_order(
        recorder.op("x_c", until=release, view=c),
        recorder.op("s_a", view=a),
        recorder.op("submit_a", "exclusive", view=a),
        recorder.op("w_b", view=b),
        recorder.op("scan_c", "concurrent", view=c),
    )
    release.set()
    await asyncio.gather(*tasks)
    assert recorder.started() == ["x_c", "s_a", "submit_a", "w_b", "scan_c"]
    assert recorder.log.index(("end", "submit_a")) < recorder.log.index(("start", "w_b"))

    # It never overtakes a call that arrived first, also one still waiting in another agent's line.
    recorder, release = Recorder(ToolGate()), asyncio.Event()
    tasks = await started_in_order(
        recorder.op("s1_a", until=release, view=a),
        recorder.op("s2_a", view=a),
        recorder.op("s3_a", view=a),
        recorder.op("scan_c", "concurrent", view=c),  # arrived before the exclusive call: it may start at once
        recorder.op("submit_b", "exclusive", view=b),
        recorder.op("s_c", view=c),
    )
    assert recorder.started() == ["s1_a", "scan_c"]
    release.set()
    await asyncio.gather(*tasks)
    assert recorder.started() == ["s1_a", "scan_c", "s2_a", "s3_a", "submit_b", "s_c"]
    assert recorder.log.index(("end", "s3_a")) < recorder.log.index(("start", "submit_b"))


async def test_the_caller_on_whose_behalf_calls_run_is_their_fairness_key():
    view = object()
    assert caller_or(view) is view  # a call outside the tool plane queues as its session view's
    with on_behalf_of("agent-1"):
        assert caller_or(view) == "agent-1"
        with on_behalf_of("agent-2"):
            assert caller_or(view) == "agent-2"
        assert caller_or(view) == "agent-1"
    assert caller_or(view) is view


async def test_one_agents_gather_takes_turns_with_a_siblings_call():
    recorder, release = Recorder(ToolGate()), asyncio.Event()
    gather_view, sibling_view = object(), object()
    gathered = await started_in_order(
        recorder.op("a0", until=release, view=gather_view),
        *(recorder.op(f"a{i}", view=gather_view) for i in range(1, 5)),
    )
    (sibling,) = await started_in_order(recorder.op("b", view=sibling_view))
    release.set()
    await asyncio.gather(*gathered, sibling)
    started = recorder.started()
    assert [name for name in started if name.startswith("a")] == [f"a{i}" for i in range(5)]  # FIFO per view
    assert started.index("b") == 2  # after the running call and the one already queued, not after all of them
    assert recorder.peak == 1


async def test_a_waiter_cancelled_in_the_queue_never_runs():
    recorder, release = Recorder(ToolGate()), asyncio.Event()
    first, cancelled, last = await started_in_order(
        recorder.op("first", until=release), recorder.op("cancelled"), recorder.op("last")
    )
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    release.set()
    assert await first == "first" and await last == "last"
    assert recorder.started() == ["first", "last"] and recorder.gate.idle


async def test_a_cancelled_or_expired_exclusive_waiter_unblocks_the_calls_behind_it():
    for drop in ("cancel", "expire"):
        recorder, release = Recorder(ToolGate()), asyncio.Event()
        timeout = 0.03 if drop == "expire" else None
        holder, submit, behind = await started_in_order(
            recorder.op("holder", until=release),
            recorder.op("submit", "exclusive", timeout=timeout),
            recorder.op("behind", "concurrent"),
        )
        assert recorder.started() == ["holder"]  # "behind" waits behind the exclusive call
        if drop == "cancel":
            submit.cancel()
        with pytest.raises(asyncio.CancelledError if drop == "cancel" else EnvBusy):
            await submit
        assert await behind == "behind"  # ... until it is gone, while the holder still runs
        release.set()
        await holder
        assert recorder.started() == ["holder", "behind"] and recorder.gate.idle


async def test_a_grant_that_races_its_callers_cancellation_is_passed_on():
    gate = ToolGate()
    first = await gate.acquire("serial")
    second = asyncio.ensure_future(gate.acquire("serial"))
    third = asyncio.ensure_future(gate.acquire("serial"))
    await asyncio.sleep(0)
    first.release()  # grants the second waiter at once ...
    second.cancel()  # ... whose caller is cancelled before it resumes
    with pytest.raises(asyncio.CancelledError):
        await second
    grant = await asyncio.wait_for(third, 1)
    grant.release()
    assert gate.idle


async def test_the_queue_deadline_refuses_a_call_that_never_got_its_turn():
    recorder, release = Recorder(ToolGate(queue_timeout_s=0.05)), asyncio.Event()
    (holder,) = await started_in_order(recorder.op("holder", until=release))
    started = time.monotonic()
    with pytest.raises(EnvBusy, match="not run: env busy") as busy:
        await recorder.op("late")
    assert time.monotonic() - started >= 0.045
    assert busy.value.timeout_s == 0.05 and busy.value.waited_s >= 0.045
    # A per-operation timeout overrides the gate's; None waits as long as it takes.
    patient = asyncio.ensure_future(recorder.op("patient", timeout=None))
    await asyncio.sleep(0.1)
    release.set()
    assert await holder == "holder" and await patient == "patient"
    assert recorder.started() == ["holder", "patient"] and recorder.gate.idle


async def test_a_granted_operation_keeps_its_grant_until_it_ends_even_when_its_caller_gives_up():
    gate = ToolGate()
    events: list[str] = []

    def blocking_tool() -> str:  # a sync env tool: it runs in a worker thread that nothing can interrupt
        events.append("thread start")
        time.sleep(0.15)
        events.append("thread end")
        return "done"

    caller = asyncio.ensure_future(gate.run("serial", lambda grant: asyncio.to_thread(blocking_tool)))
    await asyncio.sleep(0.05)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    async def next_call(grant):
        events.append("next")

    await gate.run("serial", next_call)
    assert events == ["thread start", "thread end", "next"]  # the next call waited for the abandoned one


async def test_inflight_tracks_running_operations():
    gate, inflight = ToolGate(), set()
    release = asyncio.Event()

    async def body(grant):
        await release.wait()
        return grant.waited_s

    task = asyncio.ensure_future(gate.run("concurrent", body, inflight=inflight))
    await asyncio.sleep(0.01)
    assert len(inflight) == 1
    release.set()
    assert await task >= 0 and not inflight


async def test_release_is_idempotent_and_hold_releases_on_exit_and_cancellation():
    gate = ToolGate()
    grant = await gate.acquire("exclusive")
    grant.release()
    grant.release()
    assert grant.released and gate.idle

    async with gate.hold("serial") as held:
        assert not held.released and not gate.idle
    assert held.released and gate.idle

    async def holding():
        async with gate.hold("serial"):
            await asyncio.sleep(10)

    task = asyncio.ensure_future(holding())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert gate.idle


async def test_close_refuses_waiters_and_later_operations_while_grants_run_on():
    gate = ToolGate()
    grant = await gate.acquire("serial")
    waiting = asyncio.ensure_future(gate.acquire("serial"))
    await asyncio.sleep(0)
    gate.close()
    gate.close()
    with pytest.raises(GateClosed):
        await waiting
    with pytest.raises(GateClosed):
        await gate.acquire("concurrent")
    assert gate.closed and not gate.idle
    grant.release()
    assert gate.idle


async def test_a_reentrant_acquire_raises_instead_of_waiting_for_itself():
    gate, other = ToolGate(), ToolGate()

    async def calls_back(grant):
        await gate.acquire("concurrent")

    with pytest.raises(RuntimeError, match="wait for itself"):
        await gate.run("serial", calls_back)
    async with gate.hold("serial"):
        with pytest.raises(RuntimeError, match="wait for itself"):
            await gate.acquire("serial")
        (await other.acquire("serial")).release()  # another env is fine
    assert gate.idle and other.idle


async def test_arguments_are_checked():
    for bad in (0, -1, True, "5"):
        with pytest.raises(ValueError, match="queue_timeout_s"):
            ToolGate(queue_timeout_s=bad)  # type: ignore[arg-type]
    assert ToolGate(queue_timeout_s=None).queue_timeout_s is None
    with pytest.raises(ValueError, match="Unknown call class"):
        await ToolGate().acquire("read")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="queue_timeout_s"):
        await ToolGate().acquire("serial", timeout=0)


async def test_randomized_invariants_hold_under_cancellation_and_deadlines():
    """Random classes, views, durations, cancellations and deadlines: no forbidden overlap, FIFO per view, nothing
    that arrives after a waiting exclusive call starts first, no exclusive call that starts while an earlier call
    still waits (in any view's line), no deadlock, and an idle gate afterwards."""

    classes = ["serial", "serial", "concurrent", "exclusive"]
    for seed in range(40):
        rng = random.Random(seed)
        gate = ToolGate(queue_timeout_s=None)
        running: dict[int, str] = {}
        granted_order: dict[int, list[int]] = {}
        waiting: set[int] = set()  # calls that wait for their grant, in the shared queue or in their view's line
        waiting_exclusive: set[int] = set()
        violations: list[str] = []

        async def call(seq: int) -> None:
            call_class = rng.choice(classes)
            view = rng.choice([None, 0, 1, 2])
            timeout = rng.choice([None, None, 0.002, 0.01])
            waiting.add(seq)
            if call_class == "exclusive":
                waiting_exclusive.add(seq)

            async def body(grant):
                waiting.discard(seq)
                waiting_exclusive.discard(seq)
                if any(other == "exclusive" for other in running.values()):
                    violations.append(f"{seq} {call_class} beside an exclusive call")
                if call_class == "exclusive" and running:
                    violations.append(f"exclusive {seq} beside {running}")
                if call_class == "serial" and "serial" in running.values():
                    violations.append(f"serial {seq} beside another serial call")
                if any(other < seq for other in waiting_exclusive):
                    violations.append(f"{seq} overtook a waiting exclusive call")
                if call_class == "exclusive" and any(other < seq for other in waiting):
                    violations.append(f"exclusive {seq} overtook waiting calls {sorted(waiting)}")
                if view is not None:
                    granted_order.setdefault(view, []).append(seq)
                running[seq] = call_class
                try:
                    for _ in range(rng.randint(0, 6)):
                        await asyncio.sleep(0)
                    if rng.random() < 0.2:
                        await asyncio.sleep(0.001)
                finally:
                    del running[seq]

            try:
                await gate.run(call_class, body, view=view, timeout=timeout)  # type: ignore[arg-type]
            finally:
                waiting.discard(seq)
                waiting_exclusive.discard(seq)

        tasks = []
        for seq in range(40):
            tasks.append(asyncio.ensure_future(call(seq)))
            for _ in range(rng.randint(0, 3)):
                await asyncio.sleep(0)
            if rng.random() < 0.25:
                victim = rng.choice(tasks)
                asyncio.get_running_loop().call_later(rng.random() * 0.003, victim.cancel)
        done, pending = await asyncio.wait(tasks, timeout=10)
        assert not pending, f"seed {seed}: {len(pending)} calls never finished"
        for task in done:
            if not task.cancelled() and task.exception() is not None:
                assert isinstance(task.exception(), EnvBusy), task.exception()
        for _ in range(400):  # the shielded operations of cancelled callers finish on their own
            if gate.idle:
                break
            await asyncio.sleep(0.005)
        assert not violations, (seed, violations[:3])
        for view, seqs in granted_order.items():
            assert seqs == sorted(seqs), (seed, view, seqs)
        assert gate.idle and gate.waiting == 0, seed
