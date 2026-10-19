"""The rollout worker loop feeds the watchdog while a task group is in flight.

A group's wall time (rollout timeout + straggler grace + judge tail) can exceed
the watchdog timeout, so the heartbeat must not wait for ``arun_episode`` to
return. The await is unshielded: cancelling the worker cancels the group.

    HF_HUB_OFFLINE=1 python -m pytest tests/train/tinker -q
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

from platoon.train.tinker import rl as rl_module  # noqa: E402
from platoon.train.tinker.rl import PlatoonTinkerRLTrainer  # noqa: E402


class _StubWatchdog:
    def __init__(self) -> None:
        self.beats: list[float] = []

    def heartbeat(self) -> None:
        self.beats.append(asyncio.get_event_loop().time())


class _StubTracker:
    def __init__(self) -> None:
        self.scalars: list[dict] = []

    def scalar(self, **kwargs) -> None:
        self.scalars.append(kwargs)


class _SlowWorkflow:
    def __init__(self, delay: float, fail: bool = False) -> None:
        self.delay = delay
        self.fail = fail
        self.cancelled = False

    async def arun_episode(self, data: dict):
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if self.fail:
            raise RuntimeError("boom")
        return [f"rollout-{data['task_id']}"]


def _trainer() -> PlatoonTinkerRLTrainer:
    return object.__new__(PlatoonTinkerRLTrainer)


def _shared_state(watchdog):
    return SimpleNamespace(shutdown_event=asyncio.Event(), watchdog=watchdog, train_tracker=_StubTracker())


def test_worker_loop_heartbeats_before_group_finishes(monkeypatch):
    monkeypatch.setattr(rl_module, "ROLLOUT_HEARTBEAT_INTERVAL_S", 0.05)

    async def run():
        watchdog = _StubWatchdog()
        shared = _shared_state(watchdog)
        data_q: asyncio.Queue = asyncio.Queue()
        result_q: asyncio.Queue = asyncio.Queue()
        workflow = _SlowWorkflow(delay=0.3)
        started = asyncio.get_event_loop().time()
        data_q.put_nowait({"task_id": "t1"})
        worker = asyncio.ensure_future(
            _trainer()._rollout_workflow_worker_loop(shared, workflow, data_q, result_q, tracker=shared.train_tracker)
        )
        result = await asyncio.wait_for(result_q.get(), timeout=5)
        finished = asyncio.get_event_loop().time()
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        return watchdog, result, started, finished

    watchdog, result, started, finished = asyncio.run(run())
    assert result == ["rollout-t1"]
    # Several heartbeats landed while the group was still running (before its 0.3 s completion).
    in_flight = [t for t in watchdog.beats if t - started < 0.25]
    assert len(in_flight) >= 2, watchdog.beats
    # And the post-completion heartbeat still fires.
    assert watchdog.beats[-1] >= finished - 0.05


def test_await_rollout_with_heartbeat_default_interval_is_below_watchdog_timeouts():
    assert rl_module.ROLLOUT_HEARTBEAT_INTERVAL_S == 60.0
    import inspect

    # Resolved at call time from the module constant (so tests can shorten it); 60 s is far below any watchdog.
    sig = inspect.signature(PlatoonTinkerRLTrainer._await_rollout_with_heartbeat)
    assert sig.parameters["heartbeat_interval"].default is None


def test_await_rollout_with_heartbeat_propagates_exception_and_none_watchdog():
    async def run():
        trainer = _trainer()
        with_none = await trainer._await_rollout_with_heartbeat(
            _SlowWorkflow(0.0).arun_episode({"task_id": "a"}), watchdog=None, heartbeat_interval=0.01
        )
        try:
            await trainer._await_rollout_with_heartbeat(
                _SlowWorkflow(0.0, fail=True).arun_episode({"task_id": "b"}), watchdog=None, heartbeat_interval=0.01
            )
        except RuntimeError as e:
            return with_none, str(e)
        raise AssertionError("expected RuntimeError")

    assert asyncio.run(run()) == (["rollout-a"], "boom")


def test_worker_loop_counts_failed_group_and_keeps_heartbeating(monkeypatch):
    monkeypatch.setattr(rl_module, "ROLLOUT_HEARTBEAT_INTERVAL_S", 0.02)

    async def run():
        watchdog = _StubWatchdog()
        shared = _shared_state(watchdog)
        data_q: asyncio.Queue = asyncio.Queue()
        result_q: asyncio.Queue = asyncio.Queue()
        data_q.put_nowait({"task_id": "t2"})
        worker = asyncio.ensure_future(
            _trainer()._rollout_workflow_worker_loop(
                shared, _SlowWorkflow(0.1, fail=True), data_q, result_q, tracker=shared.train_tracker
            )
        )
        result = await asyncio.wait_for(result_q.get(), timeout=5)
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        return result, shared.train_tracker.scalars, len(watchdog.beats)

    result, scalars, beats = asyncio.run(run())
    assert result is None
    assert {"rollout_completion/failed_rollouts": 1.0} in scalars
    assert beats >= 2


def test_cancelling_worker_cancels_in_flight_group(monkeypatch):
    """Unshielded: an interrupt reaches the group (and its straggler members) instead of leaking it."""
    monkeypatch.setattr(rl_module, "ROLLOUT_HEARTBEAT_INTERVAL_S", 0.02)

    async def run():
        shared = _shared_state(_StubWatchdog())
        data_q: asyncio.Queue = asyncio.Queue()
        result_q: asyncio.Queue = asyncio.Queue()
        workflow = _SlowWorkflow(delay=10.0)
        data_q.put_nowait({"task_id": "t3"})
        worker = asyncio.ensure_future(
            _trainer()._rollout_workflow_worker_loop(shared, workflow, data_q, result_q, tracker=shared.train_tracker)
        )
        await asyncio.sleep(0.1)
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        return workflow.cancelled, worker.cancelled(), result_q.qsize()

    assert asyncio.run(run()) == (True, True, 0)


class _GroupWorkflow:
    """Returns what ``GroupRolloutWorkflow`` returns: a group output carrying its own statistics."""

    async def arun_episode(self, data: dict):
        from platoon.train.tinker.workflows.group_rollout_workflow import _TaskRolloutOutput
        from platoon.utils.rollout_workload import RolloutWorkload

        await asyncio.sleep(0.01)
        output = _TaskRolloutOutput(
            [],
            workload=RolloutWorkload(),
            requested_rollouts=1,
            observed_rollouts=0,
            trainable_rollouts=0,
            task_retained_datums=0,
        )
        output.group_stats.scalar(**{"credit/baseline/avg": 0.5})
        return output


def test_worker_loop_keeps_a_groups_rollout_time_with_the_group():
    """Recorded when the group completed, it would land in whichever step was running then."""

    async def run():
        shared = _shared_state(_StubWatchdog())
        data_q: asyncio.Queue = asyncio.Queue()
        result_q: asyncio.Queue = asyncio.Queue()
        data_q.put_nowait({"task_id": "t4"})
        worker = asyncio.ensure_future(
            _trainer()._rollout_workflow_worker_loop(
                shared, _GroupWorkflow(), data_q, result_q, tracker=shared.train_tracker
            )
        )
        result = await asyncio.wait_for(result_q.get(), timeout=5)
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        return result, shared.train_tracker.scalars

    result, scalars = asyncio.run(run())
    assert scalars == []
    consumer = _StubTracker()
    result.group_stats.record(consumer)
    assert consumer.scalars[0] == {"credit/baseline/avg": 0.5}
    assert consumer.scalars[1]["rollout_completion/rollout_time"] > 0


def test_eval_records_each_groups_statistics_as_it_collects_the_group():
    async def run():
        queue: asyncio.Queue = asyncio.Queue()
        queue.put_nowait(await _GroupWorkflow().arun_episode({}))
        queue.put_nowait(None)  # a group that failed carries nothing
        tracker = _StubTracker()
        try:
            await _trainer()._eval_completion_monitor(2, queue, tracker=tracker)
        except rl_module.TerminateEvalLoop:
            return tracker.scalars
        raise AssertionError("the monitor ends the eval loop once every group is in")

    assert asyncio.run(run()) == [{"credit/baseline/avg": 0.5}]
