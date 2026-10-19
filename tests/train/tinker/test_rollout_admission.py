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


def test_pending_rows_are_bounded_but_parked_rows_do_not_count_and_waiter_cancellation_loses_nothing():
    """The bound counts the rows a worker could take: rows of a label at its limit wait beside it, parked, so a
    capped label's rows never hold back another label's."""

    async def run():
        queue = RolloutTaskQueue(1, key="env", limits={"cpu": 1})
        first, second, third = ({"env": "cpu", "id": i} for i in range(3))
        gpu = [{"env": "gpu", "id": 10 + i} for i in range(2)]
        await queue.put(first)
        assert await queue.get() == first  # cpu runs as many groups as its limit allows
        assert await queue.put(second) is True  # parked: they do not count toward the bound
        assert await queue.put(third) is True
        await queue.put(gpu[0])  # the one row a worker can take
        producer = asyncio.create_task(queue.put(gpu[1]))
        await asyncio.sleep(0)
        assert not producer.done()  # the bound holds gpu[0]
        assert await queue.get() == gpu[0]  # the rows of a label at its limit wait
        await asyncio.wait_for(producer, 1)
        assert await queue.get() == gpu[1]
        blocked = asyncio.create_task(queue.get())  # only parked rows are left
        await asyncio.sleep(0)
        assert not blocked.done()
        blocked.cancel()
        with pytest.raises(asyncio.CancelledError):
            await blocked
        await queue.release(first)
        assert await queue.get() == second
        await queue.release(second)
        assert await queue.get() == third
        assert not queue.skipped

    asyncio.run(run())


def test_a_capped_labels_full_backlog_skips_its_new_rows_and_counts_them(caplog):
    """Past its backlog, a capped label's new rows are skipped (put returns False), counted and warned of once, while
    another label's rows can start: the other labels keep the batches full."""

    async def run():
        queue = RolloutTaskQueue(1, key="env", limits={"slow": 1}, backlog_limit=2)
        running = {"env": "slow", "id": 0}
        await queue.put(running)
        assert await queue.get() == running
        assert [await queue.put({"env": "slow", "id": i}) for i in (1, 2)] == [True, True]  # its backlog, full
        await queue.put({"env": "fast", "id": 3})  # a row that can start
        assert await queue.put({"env": "slow", "id": 4}) is False
        assert await queue.put({"env": "slow", "id": 5}) is False
        assert queue.skipped == {"slow": 2}
        assert (await queue.get())["id"] == 3
        await queue.release(running)
        assert (await queue.get())["id"] == 1  # its waiting rows keep their order

    with caplog.at_level("WARNING"):
        asyncio.run(run())
    warnings = [record.getMessage() for record in caplog.records if "rows are skipped" in record.getMessage()]
    assert len(warnings) == 1 and "Label 'slow' has 2 rows waiting for its limit" in warnings[0]


@pytest.mark.parametrize("given", [True, False])
def test_a_full_backlog_skips_while_a_label_of_the_run_has_room_though_no_row_of_it_is_pending(given):
    """With the capped label's backlog full and only its rows pending (a worker took the last row of another label),
    its next row is skipped, not waited for: a label without a limit always has room, so the producer goes on to the
    rows behind it, which a worker can start at once.  The run's labels are given (the labels of its tasks), or known
    from the rows queued."""

    async def run():
        queue = RolloutTaskQueue(
            2, key="env", limits={"slow": 1}, backlog_limit=2, labels=("quick", "slow") if given else ()
        )
        running = {"env": "slow", "id": 0}
        await queue.put(running)
        assert await queue.get() == running
        assert [await queue.put({"env": "slow", "id": i}) for i in (1, 2)] == [True, True]  # its backlog, full
        quick = {"env": "quick", "id": 3}
        assert await queue.put(quick) is True
        assert await queue.get() == quick  # a worker took it: only parked rows are pending
        assert await asyncio.wait_for(queue.put({"env": "slow", "id": 4}), 1) is False
        assert queue.skipped == {"slow": 1}
        assert queue.parked() == 2 and queue.parked_by_label() == {"slow": 2}

    asyncio.run(run())


def test_a_full_backlog_waits_instead_of_skipping_while_nothing_can_start():
    """When every label of the run is at its limit, a row of a label whose backlog is full waits for a release rather
    than being skipped: a skip must leave a slot that a row of another label can use."""

    async def run():
        queue = RolloutTaskQueue(
            4, key="env", limits={"slow": 1, "other": 1}, backlog_limit=1, labels=("slow", "other")
        )
        running = [{"env": "slow", "id": 0}, {"env": "other", "id": 10}]
        for row in running:
            await queue.put(row)
            assert await queue.get() == row
        await queue.put({"env": "slow", "id": 1})  # parked: the backlog is full
        producer = asyncio.create_task(queue.put({"env": "slow", "id": 2}))
        await asyncio.sleep(0)
        assert not producer.done()  # every label of the run is at its limit: nothing can start, no skip
        await queue.release(running[0])
        assert await asyncio.wait_for(producer, 1) is True  # room again: queued, not skipped
        assert not queue.skipped

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

        worker = asyncio.create_task(
            trainer._rollout_workflow_worker_loop(
                shared,
                SimpleNamespace(arun_episode=episode),
                queue,
                asyncio.Queue(),
            )
        )
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


def test_rollout_groups_default_to_the_environment_column_mixtures_write():
    from platoon.train.components import ENVIRONMENT_COLUMN
    from platoon.train.tinker.config_defs import TrainConfig

    config = TrainConfig(model_name="Qwen/Qwen3-4B-Instruct-2507", renderer_name="qwen3_instruct")
    assert config.label_by == ENVIRONMENT_COLUMN == "_platoon_environment"
