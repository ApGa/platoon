import asyncio
from types import SimpleNamespace

import pytest

from platoon.train.tinker import rl as rl_module
from platoon.train.tinker.rollout_queue import RolloutTaskQueue
from platoon.utils.stats_tracker import StatsTracker


def test_capacity_does_not_block_other_environments_and_cpu_keeps_fifo():
    async def run():
        queue = RolloutTaskQueue(8, key="env", limits={"cpu": 1})
        rows = [{"env": env, "id": i} for i, env in enumerate(["cpu", "cpu", "gpu", "cpu", "gpu"])]
        for row in rows:
            await queue.put(row)
        first = await queue.get()
        assert first == rows[0]
        assert await queue.get() == rows[2]
        assert await queue.get() == rows[4]
        waiter = asyncio.create_task(queue.get())
        await asyncio.sleep(0)
        assert not waiter.done()
        await queue.release(first)
        second = await asyncio.wait_for(waiter, 1)
        assert second == rows[1]
        await queue.release(second)
        assert await queue.get() == rows[3]
    asyncio.run(run())


def test_pending_rows_are_bounded_and_waiter_cancellation_loses_nothing():
    async def run():
        queue = RolloutTaskQueue(1, key="env", limits={"cpu": 1})
        first, second, third = ({"env": "cpu", "id": i} for i in range(3))
        await queue.put(first)
        assert await queue.get() == first
        await queue.put(second)
        producer = asyncio.create_task(queue.put(third))
        blocked = asyncio.create_task(queue.get())
        await asyncio.sleep(0)
        assert not producer.done() and not blocked.done()
        blocked.cancel()
        with pytest.raises(asyncio.CancelledError):
            await blocked
        await queue.release(first)
        assert await queue.get() == second
        await asyncio.wait_for(producer, 1)
        await queue.release(second)
        assert await queue.get() == third
    asyncio.run(run())


@pytest.mark.parametrize("cancel", [False, True])
def test_worker_releases_capacity_after_failure_or_cancellation(cancel):
    async def run():
        trainer = object.__new__(rl_module.PlatoonTinkerRLTrainer)
        queue = RolloutTaskQueue(3, key="env", limits={"cpu": 1})
        row = {"env": "cpu", "task_id": "first"}
        next_row = {"env": "cpu", "task_id": "next"}
        await queue.put(row)
        await queue.put(next_row)
        started = asyncio.Event()
        shared = SimpleNamespace(shutdown_event=asyncio.Event(), watchdog=None, train_tracker=StatsTracker("train"))
        async def episode(data):
            started.set()
            if cancel:
                await asyncio.Event().wait()
            shared.shutdown_event.set()
            raise RuntimeError("simulated setup failure")
        worker = asyncio.create_task(trainer._rollout_workflow_worker_loop(
            shared, SimpleNamespace(arun_episode=episode), queue, asyncio.Queue(),
        ))
        await started.wait()
        if cancel:
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker
        else:
            await worker
        assert await asyncio.wait_for(queue.get(), 1) == next_row
    asyncio.run(run())


def test_waiting_task_binds_policy_only_after_admission():
    async def run():
        queue = RolloutTaskQueue(2, key="env", limits={"cpu": 1})
        await queue.put({"env": "cpu", "id": 1})
        await queue.put({"env": "cpu", "id": 2})
        first = await queue.get()
        policy = 3
        async def bind_after_get():
            task = await queue.get()
            return task, policy
        waiter = asyncio.create_task(bind_after_get())
        await asyncio.sleep(0)
        policy = 8
        await queue.release(first)
        task, bound_version = await asyncio.wait_for(waiter, 1)
        assert task["id"] == 2 and bound_version == 8
    asyncio.run(run())
