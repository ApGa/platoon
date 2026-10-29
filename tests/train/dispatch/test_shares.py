"""The shares of a run's groups when its policies step independently (``platoon.train.dispatch.shares``), and the
dispatcher's horizon per policy (``platoon.train.dispatch.horizons``), on their own: a settled group splits into one
share per trained policy with rows in it (the leader's alone for a group that failed), each policy's steps take theirs
in order (the leader's every group, one without its rows too, so that the plan counts groups), each consumption records
the share's step, age and fate, a group's fate goes to the statistics once, a group retires once every share is in a
durable row, a resume puts back what each policy consumed after its row (with its recorded age, at which a trainer of
its own judges it), a policy's queue ends once the plan is drawn, and a trainer's take stops on a stop request; a group
starts only while every policy's horizon has room, and the metrics name the horizon that held the dispatcher back; once
a stop ended a policy's loop and its horizon is full, no share will come, and the other policies' queues end."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
import tinker
import torch
from fakes.dispatcher import LabelledRows, settle

from platoon.train.credit.api import TrainingSample
from platoon.train.dispatch.dispatcher import BoundDraw, Dispatcher
from platoon.train.dispatch.horizons import Horizons
from platoon.train.dispatch.settled import _Settled
from platoon.train.dispatch.shares import PLAN_END, Share, ShareQueue, Shares
from platoon.train.trainer.signals import TrainInterrupted
from platoon.train.workflow.output import PolicyGroupOutput
from platoon.train.workflow.protocol import OutputReport, report_of
from platoon.utils.rollout_workload import RolloutWorkload


class _Policies:
    """The run's trained policies as the ledger reads them: their ids, the leader and each one's version."""

    def __init__(self, *ids: str) -> None:
        self._policies = {policy: SimpleNamespace(id=policy, version=0) for policy in ids}
        self.leader = ids[0]

    def trainable(self) -> list[Any]:
        return list(self._policies.values())

    def __getitem__(self, policy: str) -> Any:
        return self._policies[policy]


def _row(version: int) -> tinker.Datum:
    inputs = {"target_tokens": [2], "advantages": [1.0], "mask": [1.0], "checkpoint_version": [version]}
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1]),
        loss_fn_inputs={key: tinker.TensorData.from_torch(torch.tensor(value)) for key, value in inputs.items()},
    )


def _group(key: tuple[int, int], rows: dict[str, int], *, record: Any = "record") -> _Settled:
    """A settled group of draw ``key`` with one row of each policy of ``rows`` at its version, whose report gives a
    workload."""

    samples = tuple(TrainingSample(_row(version), policy, 0) for policy, version in rows.items())
    report = OutputReport(
        workload=RolloutWorkload(),
        requested_rollouts=1,
        observed_rollouts=1,
        trainable_rollouts=1,
        task_retained_datums=len(samples),
        verdict="trains",
    )
    return _Settled("entry", "label", PolicyGroupOutput(samples, report), record, key)


class _Run:
    """A ledger over a dispatcher with a horizon per policy (main: 2 shares a step, bound 1; worker: 1, bound 3, unless
    the test says otherwise), whose groups settle when the test says (:meth:`settle`); the fates the ledger hands the
    statistics (``fates``)."""

    def __init__(self, *, steps: int = 10, worker_bound: int = 3, main_batch: int = 2) -> None:
        self.policies = _Policies("main", "worker")
        bounds, batches = {"main": 1, "worker": worker_bound}, {"main": main_batch, "worker": 1}
        horizons = Horizons(bounds, batches, leader="main", steps=steps)
        self.dispatcher = Dispatcher(
            rows=LabelledRows(),
            order=lambda position: {"task_id": str(position)},
            batch_size=2,
            max_staleness=1,
            steps=steps,
            horizons=horizons,
        )
        self.fates: list[tuple[tuple[int, int], str]] = []
        policies: Any = self.policies
        self.ledger = Shares(
            policies,
            dispatcher=self.dispatcher,
            record_fates=lambda keys, fates: self.fates.extend((key, fates[key]) for key in keys),
        )
        self.dispatcher.bind(self.ledger)
        self.draws: list[BoundDraw] = []

    def start(self, count: int) -> None:
        """Start ``count`` groups as the dispatcher would, each counted as running."""

        for _ in range(count):
            draw = self.dispatcher._new_draw()
            assert isinstance(draw, BoundDraw)
            self.dispatcher.pipeline += 1
            self.draws.append(draw)

    def settle(self, draw: BoundDraw, rows: dict[str, int]) -> None:
        self.dispatcher.settle(draw, _group(draw.key, rows))


def test_a_group_splits_into_a_share_per_trained_policy_and_the_leader_s_carries_its_report():
    async def run():
        harness = _Run()
        harness.start(2)
        harness.settle(harness.draws[0], {"main": 0, "worker": 0})
        harness.dispatcher.settle(harness.draws[1], _Settled("entry", "label", None, "record", harness.draws[1].key))
        main, failed = await harness.ledger.take("main", 2)  # a trainer's take: each share as its settled group
        worker = await _share(harness.ledger.queue("worker"))  # the rl trainer's: the share
        assert (main.key, failed.key, worker.group.key) == (
            harness.draws[0].key,
            harness.draws[1].key,
            harness.draws[0].key,
        )
        assert len(main.rollout) == 1 and len(worker.group.rollout) == 1
        assert main.record == "record" and worker.group.record is None
        assert report_of(main.rollout).workload is not None
        assert report_of(worker.group.rollout).workload is None and report_of(worker.group.rollout).verdict == "trains"
        assert failed.rollout is None  # a group that failed is the leader's share alone
        assert worker.peers == {"main": 0}
        assert harness.dispatcher.pipeline == 0

    asyncio.run(run())


def test_a_group_without_the_leader_s_rows_is_a_share_of_the_leader_s_with_none_and_counts_toward_the_plan():
    """The leader's steps take every group, as the plan counts them: a group with the worker's rows only is a share of
    main's too, which holds none of its rows and carries the group's report and record; its fate is the rows'."""

    async def run():
        harness = _Run(steps=2)  # main takes 2 shares a step: a plan of 4 groups
        harness.start(4)
        for index, draw in enumerate(harness.draws):
            harness.settle(draw, {"worker": 0} if index % 2 else {"main": 0, "worker": 0})
        key = harness.draws[1].key
        # The plan is drawn: 4 groups, whether or not they hold main's rows.
        assert not harness.ledger.horizons.plan_has_room(0)
        main = _queued(harness.ledger.queue("main"))
        worker = _queued(harness.ledger.queue("worker"))
        assert [share.group.key for share in main] == [draw.key for draw in harness.draws]
        empty, rows = main[1], worker[1]
        assert len(empty.group.rollout) == 0 and empty.group.record == "record" and empty.peers == {}
        assert report_of(empty.group.rollout).workload is not None
        assert rows.group.record is None and report_of(rows.group.rollout).workload is None and rows.peers == {}
        assert not harness.ledger.horizons.plan_has_room(0)  # taken, they still count
        # Its fate is its rows': stale when the worker's went stale, as a lockstep step judges the group.
        harness.ledger.consumed("main", [key], step=0)
        assert harness.fates == []
        harness.ledger.consumed("worker", [key], step=0, fates={key: "stale"})
        assert harness.fates == [(key, "stale")]

    asyncio.run(run())


def test_a_group_s_fate_goes_to_the_statistics_once_when_its_last_share_is_consumed():
    async def run():
        harness = _Run()
        harness.start(1)
        harness.settle(harness.draws[0], {"main": 0, "worker": 0})
        key = harness.draws[0].key
        harness.policies["worker"].version = 3
        harness.ledger.consumed("worker", [key], step=0, fates={key: "stale"})
        assert harness.fates == []
        harness.ledger.consumed("main", [key], step=0, fates={key: "consumed"})
        assert harness.fates == [(key, "consumed")]
        # Each share's step, age (its policy's version then, less the version on its rows) and fate.
        assert harness.ledger.snapshot()[key] == {
            "shares": {
                "main": {"step": 0, "age": 0, "fate": "consumed"},
                "worker": {"step": 0, "age": 3, "fate": "stale"},
            },
            "recorded": True,
        }

    asyncio.run(run())


def test_a_group_retires_once_every_share_is_in_a_durable_row():
    async def run():
        harness = _Run()
        harness.start(1)
        harness.settle(harness.draws[0], {"main": 0, "worker": 0})
        key = harness.draws[0].key
        harness.ledger.consumed("main", [key], step=0)
        harness.ledger.consumed("worker", [key], step=4)
        assert harness.ledger.durable("main", 1) == []  # the worker's share is not in a durable row yet
        assert harness.ledger.durable("worker", 4) == []  # a row of step 4 does not hold what step 4 consumed
        assert harness.dispatcher.holding(key) is not None
        assert harness.ledger.durable("worker", 5) == [key]
        assert harness.dispatcher.holding(key) is None and harness.ledger.retained() == 0

    asyncio.run(run())


def test_a_resume_puts_back_what_each_policy_consumed_after_its_row_with_its_recorded_age():
    async def run():
        harness = _Run()
        harness.start(2)
        for draw in harness.draws:
            harness.settle(draw, {"main": 0, "worker": 1})
        first, second = (draw.key for draw in harness.draws)
        harness.policies["worker"].version = 3
        harness.ledger.consumed("worker", [first], step=5)
        harness.ledger.consumed("worker", [second], step=7)
        saved = harness.ledger.snapshot()

        again = _Run()
        again.dispatcher.restore(
            kept=[(key, "task", None, _group(key, {"main": 0, "worker": 1})) for key in (first, second)],
            reissue=[],
            waiting=[],
            version=0,
        )
        again.ledger.restore(saved)
        # The worker resumes from its row of step 6: what it consumed at step 7 goes back, at the age it had then.
        start: Any = SimpleNamespace(
            step=0, policies={"main": SimpleNamespace(step=0), "worker": SimpleNamespace(step=6)}
        )
        again.ledger.revert(start)
        reverted = await _share(again.ledger.queue("worker"))
        assert reverted.group.key == second and reverted.age == 2 and reverted.judged_at() == 3
        assert again.ledger.queue("worker").empty()
        assert [group.key for group in await again.ledger.take("main", 2)] == [first, second]

    asyncio.run(run())


def test_a_trainer_judges_a_share_a_resume_put_back_at_the_version_its_recorded_age_gives():
    """A trainer of its own takes its shares as settled groups (``take``), which carry no age: ``judge_version`` gives
    the version to judge each at, the policy's version now for a fresh share, and for one a resume put back the version
    at which it has the age it had when first consumed, though the resume moved the policy's version on."""

    async def run():
        harness = _Run()
        harness.start(1)
        harness.settle(harness.draws[0], {"main": 0, "worker": 1})
        key = harness.draws[0].key
        harness.policies["worker"].version = 3
        [fresh] = await harness.ledger.take("worker", 1)
        assert harness.ledger.judge_version("worker", fresh) == 3  # a fresh share: the version now
        harness.ledger.consumed("worker", [key], step=7)  # at the age 2
        saved = harness.ledger.snapshot()

        again = _Run()
        again.dispatcher.restore(
            kept=[(key, "task", None, _group(key, {"main": 0, "worker": 1}))], reissue=[], waiting=[], version=0
        )
        again.ledger.restore(saved)
        again.policies["worker"].version = 9  # the version rule: one more than the commit records it published
        start: Any = SimpleNamespace(
            step=0, policies={"main": SimpleNamespace(step=0), "worker": SimpleNamespace(step=6)}
        )
        again.ledger.revert(start)
        [reverted] = await again.ledger.take("worker", 1)
        assert reverted.key == key and again.ledger.judge_version("worker", reverted) == 3
        again.ledger.consumed("worker", [reverted.key], step=6)
        assert again.ledger.judge_version("worker", reverted) == 9  # consumed: no longer one a resume put back

    asyncio.run(run())


def test_a_policy_s_queue_ends_once_the_plan_is_drawn_and_nothing_runs():
    async def run():
        harness = _Run(steps=1)
        harness.start(2)
        queue = harness.ledger.queue("worker")
        waiting = asyncio.ensure_future(_share(queue))
        await settle()
        assert not waiting.done()
        harness.settle(harness.draws[0], {"main": 0, "worker": 0})
        assert (await waiting).group.key == harness.draws[0].key
        harness.settle(harness.draws[1], {"main": 0})  # no share of the worker's
        await harness.ledger.take("main", 2)
        harness.ledger.step_started("main")  # the leader's last step: the plan is drawn
        assert await queue.get() is PLAN_END

    asyncio.run(run())


def test_a_trainer_s_take_stops_on_a_stop_request_and_leaves_the_shares_it_took_unconsumed():
    async def run():
        harness = _Run()
        harness.start(2)
        harness.settle(harness.draws[0], {"main": 0, "worker": 0})
        stop = asyncio.Event()
        taking = asyncio.ensure_future(harness.ledger.take("main", 2, stop=stop))  # one share there, one to come
        await settle()
        assert not taking.done()
        stop.set()
        with pytest.raises(TrainInterrupted):
            await taking
        with pytest.raises(TrainInterrupted):  # once stop is set, at once
            await harness.ledger.take("worker", 1, stop=stop)
        assert harness.ledger.snapshot()[harness.draws[0].key]["shares"]["main"] is None  # not consumed: kept

    asyncio.run(run())


def test_once_a_stopped_policy_s_full_horizon_holds_every_group_back_no_share_will_come_and_the_queue_ends():
    """main's open step (4 shares) took 2 when a stop request ended the worker's loop (its horizon: 2 shares).  While
    the worker's horizon has room a group may still start, and main's queue waits; once it is full, no group starts
    again, and main's queue gives the plan's end, so that its step closes short rather than wait for good."""

    async def run():
        harness = _Run(worker_bound=1, main_batch=4)
        harness.start(2)
        harness.settle(harness.draws[0], {"main": 0, "worker": 0})
        harness.settle(harness.draws[1], {"main": 0})  # no share of the worker's
        queue = harness.ledger.queue("main")
        await queue.get()
        await queue.get()
        harness.ledger.stopped("worker")
        assert not harness.ledger.done("main") and not harness.ledger.horizons.held_for_good()
        harness.start(1)
        harness.settle(harness.draws[2], {"main": 0, "worker": 0})  # the worker's horizon is full now
        assert harness.ledger.horizons.held_for_good()
        assert (await _share(queue)).group.key == harness.draws[2].key
        assert await queue.get() is PLAN_END  # nothing queued, nothing runs, and no group can start
        assert not harness.ledger.queue("worker").empty()  # the stopped policy's shares stay, for the checkpoint

    asyncio.run(run())


def test_every_policy_s_horizon_bounds_the_groups_that_start_and_the_metrics_name_the_one_that_binds():
    horizons = Horizons({"main": 1, "worker": 0}, {"main": 2, "worker": 1}, leader="main", steps=10)
    assert horizons.sizes() == {"main": 4, "worker": 1}
    assert horizons.binding(running=0) is None
    assert horizons.binding(running=1) == "worker"  # a running group counts against every horizon
    horizons.queued("main", 3)
    assert horizons.plan_has_room(0) and not Horizons(
        {"main": None}, {"main": 2}, leader="main", steps=1
    ).plan_has_room(2)

    async def run():
        harness = _Run()
        stop = asyncio.Event()
        task = asyncio.ensure_future(
            harness.dispatcher.run(spawn=asyncio.ensure_future, run_group=_never_ends, stop=stop)
        )
        await settle()
        # main's horizon (2 × 2 shares) binds before the worker's (4 × 1): four groups run.
        assert harness.dispatcher.pipeline == 4
        shares = harness.dispatcher.blocked_shares()
        assert (
            shares["dispatch/blocked_share/horizon:main"] > 0 and shares["dispatch/blocked_share/horizon:worker"] == 0
        )
        assert shares["dispatch/blocked_share/horizon"] == shares["dispatch/blocked_share/horizon:main"]
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def _queued(queue: ShareQueue) -> list[Share]:
    """Every share ``queue`` holds now, taken."""

    shares = []
    while not queue.empty():
        shares.append(queue.get_nowait())
    return shares


async def _share(queue: ShareQueue) -> Share:
    """The next share of ``queue``, which has one to come."""

    share = await queue.get()
    assert isinstance(share, Share)
    return share


async def _never_ends(draw: BoundDraw, lease: Any) -> None:
    await asyncio.Event().wait()
