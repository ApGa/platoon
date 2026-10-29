"""The dispatcher of a training run's rollout groups (``platoon.train.dispatch.dispatcher``), on its own: it starts a
group only when the horizon, the rollout permits and the label limits allow, keeps a capped label's draws in its
backlog, holds a slot of the horizon for a capped label's next group, skips past a full backlog only while another
label has room, never draws while every label is at its limit, and saves and takes up what a checkpoint saved."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import tinker
from fakes.dispatcher import LabelledRows, Recorder, settle

from platoon.train.checkpoints import dispatch_state as dispatch_state_module
from platoon.train.checkpoints.dispatch_state import plan_resume, read_state, save_state
from platoon.train.checkpoints.kept_groups import decode_group, read_kept
from platoon.train.dispatch.dispatcher import POSITION_COLUMN, SEQ_COLUMN, BoundDraw, Dispatcher


class _Harness:
    """A dispatcher over ``rows`` (the order repeats them), whose groups run until the test ends them (:meth:`end`)."""

    def __init__(self, rows: list[dict[str, Any]], *, permits: int = 1000, column: str = "env", **kwargs: Any) -> None:
        self.rows = rows
        self.recorder = Recorder()
        kwargs.setdefault("batch_size", 4)
        kwargs.setdefault("max_staleness", None)
        kwargs.setdefault("steps", 100)
        self.dispatcher = Dispatcher(
            rows=LabelledRows(permits, column),
            order=lambda position: rows[position % len(rows)],
            recorder=self.recorder,
            **kwargs,
        )
        self.results: asyncio.Queue[Any] = asyncio.Queue()
        self.dispatcher.bind(self.results)
        self.started: list[BoundDraw] = []
        self.gates: dict[tuple[int, int], asyncio.Event] = {}
        self.tasks: list[asyncio.Task] = []
        self.version = 0
        self.versions: dict[tuple[int, int], int] = {}
        self.fail: set[int] = set()

    def run(self, stop: asyncio.Event | None = None) -> None:
        self.tasks.append(
            asyncio.ensure_future(self.dispatcher.run(spawn=self._spawn, run_group=self._group, stop=stop))
        )

    def _spawn(self, coroutine) -> asyncio.Task:
        task = asyncio.ensure_future(coroutine)
        self.tasks.append(task)
        return task

    async def _group(self, draw: BoundDraw, release) -> None:
        self.started.append(draw)
        self.versions[draw.key] = self.version  # what the group binds when it starts
        try:
            await self.gates.setdefault(draw.key, asyncio.Event()).wait()
            if draw.position in self.fail:
                raise RuntimeError("the group failed")
        finally:
            release()
        self.dispatcher.settle(draw, draw)

    def end(self, *draws: BoundDraw) -> None:
        for draw in draws:
            self.gates.setdefault(draw.key, asyncio.Event()).set()

    @property
    def positions(self) -> list[int | None]:
        return [draw.position for draw in self.started]

    async def close(self) -> None:
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


def _labels(*names: str) -> list[dict[str, Any]]:
    return [{"task_id": f"{name}{index}", "env": name} for index, name in enumerate(names)]


def test_a_label_at_its_limit_never_holds_back_another_and_its_draws_keep_their_order():
    async def run():
        rows = _labels("cpu", "cpu", "gpu", "cpu", "gpu")
        harness = _Harness(rows, limits={"cpu": 1, "gpu": 2}, labels=("cpu", "gpu"))
        harness.run()
        await settle()
        # gpu's groups start past the cpu draws that wait; once both labels are at their limits, nothing is drawn.
        assert harness.positions == [0, 2, 4]
        assert harness.dispatcher.parked_by_label() == {"cpu": 2} and harness.dispatcher.next_position == 5
        harness.end(harness.started[0])
        await settle()
        assert harness.positions == [0, 2, 4, 1]
        harness.end(harness.started[3])
        await settle()
        assert harness.positions == [0, 2, 4, 1, 3]  # in their order
        await harness.close()

    asyncio.run(run())


def test_the_horizon_counts_the_groups_that_can_run_not_the_draws_that_wait():
    """With max_staleness 1 the pipeline holds two batches of groups that can run, less the slot held for the capped
    label's next group; the draws of a capped label that wait for its limit do not count, so the dispatcher draws past
    them, and a step that starts opens one batch more."""

    async def run():
        rows = [{"task_id": str(i), "env": "slow" if i % 2 else "quick"} for i in range(400)]
        harness = _Harness(rows, limits={"slow": 1}, max_staleness=1, batch_size=4)
        harness.run()
        await settle()
        # Six quick groups and the one slow group, and a slot held for the slow label's next group; four slow draws
        # wait for the limit: 11 draws, past two batches.
        assert len(harness.started) == 7 and sum(draw.label == "slow" for draw in harness.started) == 1
        assert harness.dispatcher.parked_by_label() == {"slow": 4} and harness.dispatcher.next_position == 11
        for _ in range(4):  # a step dequeues its four groups: they still count until it starts the next
            harness.dispatcher.dequeued()
        await settle()
        assert len(harness.started) == 7
        harness.dispatcher.step_started()
        await settle()
        assert len(harness.started) == 11 and harness.dispatcher.parked_by_label() == {"slow": 8}
        assert harness.dispatcher.peak == 7
        await harness.close()

    asyncio.run(run())


def test_a_capped_label_s_next_group_starts_as_soon_as_one_ends_not_at_the_next_step():
    """While the capped label has draws waiting, its running group holds a slot of the horizon that the other label's
    draws leave free: when the group ends in the middle of a step, the next one starts at once, rather than at the next
    step, which would let a limit of one start one group a step however short its groups."""

    async def run():
        rows = _labels(*["a", "b"] * 40)
        harness = _Harness(rows, limits={"a": 1}, labels=("a", "b"), max_staleness=1, batch_size=4)
        harness.run()
        await settle()
        assert harness.positions == [0, 1, 3, 5, 7, 9, 11]  # a0 and six b groups: a slot held for a's next group
        assert harness.dispatcher.parked_by_label() == {"a": 5}
        harness.end(harness.started[0])  # a0 ends, and no step starts
        await settle()
        assert harness.positions[7:] == [2]  # a's next group took the slot its group held
        assert harness.dispatcher.peak == 2 * 4  # the horizon is full: eight groups started, none consumed
        harness.end(harness.started[7])
        await settle()
        assert len(harness.started) == 8  # so no other group starts until a step consumes some
        for _ in range(4):
            harness.dispatcher.dequeued()
        harness.dispatcher.step_started()
        await settle()
        assert harness.positions[8:] == [4, 13, 15]  # a's oldest waiting draw first, then b's, a slot held for a's
        assert harness.dispatcher.peak == 2 * 4
        await harness.close()

    asyncio.run(run())


def test_a_capped_label_below_its_limit_starts_its_waiting_draw_in_the_slots_held_for_it():
    """A label capped at two groups, both running and its draws waiting: each holds a slot for the label's next group.
    When one ends, the label is below its limit, and its oldest waiting draw takes the slot held for it, though the
    other group's slot is still held."""

    async def run():
        rows = _labels(*["a", "b"] * 40)
        harness = _Harness(rows, limits={"a": 2}, labels=("a", "b"), max_staleness=1, batch_size=4)
        harness.run()
        await settle()
        assert harness.positions == [0, 1, 2, 3, 5, 7]  # a0, a2 and four b groups: two slots held for a's next groups
        harness.end(harness.started[0])
        await settle()
        assert harness.positions[6:] == [4]
        harness.end(harness.started[2])  # a2: a is below its limit again, a4 still running
        await settle()
        assert harness.positions[6:] == [4, 6] and harness.dispatcher.peak == 2 * 4
        await harness.close()

    asyncio.run(run())


def test_the_slots_held_for_capped_labels_leave_a_batch_of_the_horizon_to_the_others():
    """The capped labels' running groups and the slots held for their next ones fit in k batches: with k = 1 and a
    label at its limit of one batch, no slot is held, and the other label's groups take the rest of the horizon."""

    async def run():
        rows = _labels("a", "a", "a", "b", "b", "b", "b", "b")
        harness = _Harness(rows, limits={"a": 2}, labels=("a", "b"), max_staleness=1, batch_size=2)
        harness.run()
        await settle()
        assert harness.positions == [0, 1, 3, 4]  # a's two groups fill a batch; b's take the other
        await harness.close()

    asyncio.run(run())


def test_with_k_0_no_slot_is_held_and_a_capped_label_starts_its_limit_of_groups_a_step():
    async def run():
        rows = _labels(*["a", "b"] * 20)
        harness = _Harness(rows, limits={"a": 1}, labels=("a", "b"), max_staleness=0, batch_size=4)
        harness.run()
        await settle()
        assert harness.positions == [0, 1, 3, 5]
        harness.end(harness.started[0])
        await settle()
        assert harness.positions == [0, 1, 3, 5]  # a's next group waits for the next step
        for _ in range(4):
            harness.dispatcher.dequeued()
        harness.dispatcher.step_started()
        await settle()
        assert harness.positions[4:] == [2, 7, 9, 11]  # a's oldest waiting draw first, then b's
        await harness.close()

    asyncio.run(run())


def test_a_label_the_dispatcher_is_not_given_counts_as_having_room_only_once_it_is_drawn():
    """``labels`` must be complete: the dispatcher draws only while a label it knows has room, and learns a label only
    by drawing it, so with none given the quick draws behind a capped label at its limit wait for it to end."""

    async def positions(**kwargs: Any) -> list[int | None]:
        harness = _Harness(_labels("slow", "slow", "slow", "quick", "quick"), limits={"slow": 1}, **kwargs)
        harness.run()
        await settle()
        result = harness.positions
        await harness.close()
        return result

    assert asyncio.run(positions(labels=("quick", "slow"), batch_size=2, steps=10))[:3] == [0, 3, 4]
    assert asyncio.run(positions(batch_size=2, steps=10)) == [0]


@pytest.mark.parametrize("k", [0, 2])
def test_the_pipeline_holds_k_plus_one_batches_and_a_step_opens_one_more(k):
    async def run():
        harness = _Harness(_labels(*["a"] * 8), max_staleness=k, batch_size=2)
        harness.run()
        await settle()
        assert len(harness.started) == 2 * (k + 1)
        harness.end(*harness.started)
        await settle()
        assert len(harness.started) == 2 * (k + 1)  # settled, and not yet dequeued: they still count
        harness.dispatcher.dequeued()
        harness.dispatcher.dequeued()
        harness.dispatcher.step_started()
        await settle()
        assert len(harness.started) == 2 * (k + 2) and harness.dispatcher.peak == 2 * (k + 1)
        await harness.close()

    asyncio.run(run())


def test_it_never_starts_more_groups_than_the_run_s_steps_consume():
    async def run():
        harness = _Harness(_labels(*["a"] * 8), batch_size=2, steps=2)
        harness.run()
        await settle()
        harness.end(*harness.started)
        await settle()
        assert harness.positions == [0, 1, 2, 3]  # two steps of two, and no group past them
        await harness.close()

    asyncio.run(run())


def test_a_full_backlog_skips_new_draws_counts_them_and_warns_once(caplog):
    async def run():
        rows = _labels("slow", "slow", "slow", "fast", "slow", "slow", "fast")
        harness = _Harness(
            rows, limits={"slow": 1}, labels=("slow", "fast"), batch_size=2, max_staleness=1, backlog_limit=2
        )
        harness.run()
        await settle()
        # slow0 runs, slow1 and slow2 wait (its backlog is full), fast3 runs, slow4 and slow5 are skipped, fast6 runs,
        # and so on to the horizon's four groups, less the slot held for slow's next group.
        assert harness.positions == [0, 3, 6]
        assert harness.recorder.skips == [4, 5] and harness.dispatcher.skipped == {"slow": 2}
        assert [draw[0:2] for draw in harness.dispatcher.state()["outstanding"]] == [
            [0, 0],
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 6],
        ]
        harness.end(harness.started[0])
        await settle()
        assert harness.positions[3:] == [1]  # its waiting draws keep their order: slow1 takes the slot slow0 held
        await harness.close()

    with caplog.at_level(logging.WARNING):
        asyncio.run(run())
    warnings = [record.getMessage() for record in caplog.records if "rows are skipped" in record.getMessage()]
    assert len(warnings) == 1 and "Label 'slow' has 2 rows waiting for its limit" in warnings[0]


def test_a_full_backlog_skips_while_a_label_of_the_run_has_room_though_its_draws_are_rare():
    """A label without a limit always has room, so a draw of a capped label whose backlog is full is skipped, not
    waited for, even when the next draws of the other label are far down the order: the dispatcher goes on drawing."""

    async def run():
        rows = _labels(*["slow"] * 9, "quick")
        harness = _Harness(rows, limits={"slow": 1}, labels=("quick", "slow"), batch_size=2, steps=1, backlog_limit=2)
        harness.run()
        await settle()
        assert harness.positions == [0, 9]  # slow0, and quick9 past two waiting and six skipped slow draws
        assert harness.dispatcher.skipped == {"slow": 6} and harness.dispatcher.parked_by_label() == {"slow": 2}
        await harness.close()

    asyncio.run(run())


def test_no_draw_while_every_label_is_at_its_limit():
    """With every label of the run at its limit, the dispatcher makes no draw (none to wait, none to skip) until a
    group ends; then it draws again."""

    async def run():
        rows = _labels("slow", "other", "slow", "other", "slow", "other")
        harness = _Harness(rows, limits={"slow": 1, "other": 1}, labels=("slow", "other"), batch_size=4)
        harness.run()
        await settle()
        assert harness.positions == [0, 1]
        drawn = harness.dispatcher.next_position
        assert drawn == 2 and len(harness.recorder.draws) == 2 and not harness.recorder.skips
        for _ in range(3):
            await settle()
        assert harness.dispatcher.next_position == drawn  # nothing drawn while both are at their limits
        harness.end(harness.started[0])
        await settle()
        assert harness.positions == [0, 1, 2] and harness.dispatcher.next_position == 3
        await harness.close()

    asyncio.run(run())


def test_with_one_capped_label_it_draws_one_group_at_a_time_and_none_waits():
    async def run():
        harness = _Harness(_labels(*["slow"] * 6), limits={"slow": 1}, labels=("slow",), batch_size=4)
        harness.run()
        for expected in range(1, 5):
            await settle()
            assert len(harness.started) == expected and harness.dispatcher.next_position == expected
            assert not harness.dispatcher.parked_by_label() and not harness.recorder.skips
            harness.end(harness.started[-1])
        await harness.close()

    asyncio.run(run())


@pytest.mark.parametrize("cancel", [False, True])
def test_a_group_that_fails_or_is_cancelled_frees_its_label_and_permits(cancel):
    async def run():
        harness = _Harness(_labels("cpu", "cpu", "cpu"), limits={"cpu": 1}, permits=1, backlog_limit=10)
        harness.fail.add(0)
        harness.run()
        await settle()
        assert harness.positions == [0]
        first = harness.tasks[1]
        if cancel:
            first.cancel()
        else:
            harness.end(harness.started[0])
        await settle()
        assert harness.positions == [0, 1]  # its label slot and its permit were given back
        await harness.close()

    asyncio.run(run())


def test_groups_take_their_size_in_permits_first_come_first_served():
    """A group of 8 that waits first for permits is not overtaken by a group of 4 that the free permits would hold."""

    async def run():
        rows = [{"task_id": "a", "env": "x", "size": 8}, {"task_id": "b", "env": "x", "size": 8}]
        rows.append({"task_id": "c", "env": "x", "size": 4})
        harness = _Harness(rows, permits=12, batch_size=3, steps=1)
        harness.run()
        await settle()
        assert harness.positions == [0]  # b waits for 8 permits; c, behind it, for its turn
        harness.end(harness.started[0])
        await settle()
        assert harness.positions == [0, 1, 2]
        await harness.close()

    asyncio.run(run())


def test_a_group_binds_the_policy_when_it_starts_not_when_it_is_drawn():
    async def run():
        harness = _Harness(_labels("a", "a"), permits=1, batch_size=2, steps=1)
        harness.run()
        await settle()
        assert harness.dispatcher.next_position == 2  # the second draw waits for a permit
        harness.version = 8  # the policy moves on meanwhile
        harness.end(harness.started[0])
        await settle()
        assert harness.versions[harness.started[1].key] == 8
        await harness.close()

    asyncio.run(run())


@pytest.mark.parametrize("finish_batches", [False, True])
def test_after_a_stop_request_it_starts_groups_only_while_a_batch_with_several_updates_must_finish(finish_batches):
    async def run():
        harness = _Harness(_labels(*["a"] * 4), batch_size=4, steps=1, finish_batches=finish_batches)
        stop = asyncio.Event()
        stop.set()
        harness.run(stop)
        await settle()
        assert len(harness.started) == (4 if finish_batches else 0)
        await harness.close()

    asyncio.run(run())


def test_a_stop_request_while_a_draw_waits_for_permits_starts_no_group_and_keeps_the_draw():
    async def run():
        harness = _Harness(_labels("a", "a", "a"), permits=1, batch_size=3, steps=1)
        stop = asyncio.Event()
        harness.run(stop)
        await settle()
        assert harness.positions == [0]  # the second draw waits for the permit
        stop.set()
        harness.end(harness.started[0])
        await settle()
        assert harness.positions == [0]  # its permit came after the request: it does not start
        assert harness.dispatcher.state()["waiting"] == {"a": [[0, 1]]}  # but waits, first in line, for a resume
        assert harness.tasks[0].done()
        await harness.close()

    asyncio.run(run())


def test_draws_are_numbered_and_stamped_with_their_position():
    async def run():
        harness = _Harness(_labels("a", "a", "a"), batch_size=3, steps=1, start=7)
        harness.run()
        await settle()
        assert [(draw.start, draw.seq, draw.position) for draw in harness.started] == [(7, 0, 0), (7, 1, 1), (7, 2, 2)]
        assert [(draw.row[SEQ_COLUMN], draw.row[POSITION_COLUMN]) for draw in harness.started] == [
            (0, 0),
            (1, 1),
            (2, 2),
        ]
        assert [draw["seq"] for draw in harness.recorder.draws] == [0, 1, 2]
        await harness.close()

    asyncio.run(run())


def test_a_resume_trains_the_kept_groups_first_then_draws_again_then_goes_on_with_the_order():
    """What a checkpoint saved: the kept groups go to the result queue first and count in the pipeline; the draws to
    draw again start before any new draw, as new draws of this start at their old positions; a capped label's waiting
    draws go back to its backlog, in order; and the order goes on from where it stopped."""

    async def run():
        rows = _labels("quick", "slow", "quick", "slow", "quick", "quick", "quick", "quick")
        harness = _Harness(rows, limits={"slow": 1, "quick": 3}, labels=("quick", "slow"), batch_size=4, start=3)
        dispatcher = harness.dispatcher
        dispatcher.next_position = 6
        dispatcher.restore(
            kept=[((2, 5), "quick2", 2, "kept-a"), ((2, 9), "quick5", 5, "kept-b")],
            reissue=[((2, 4), rows[4], 4), ((2, 7), rows[1], 1)],
            waiting=[((2, 8), rows[3], 3)],
            version=11,
        )
        assert dispatcher.kept() == [((2, 5), "kept-a"), ((2, 9), "kept-b")] and dispatcher.pipeline == 2
        assert [draw["reissues"] for draw in harness.recorder.draws] == [(2, 4), (2, 7), (2, 8)]
        assert {draw["version"] for draw in harness.recorder.draws} == {11}
        state = dispatcher.state()
        assert state["next_position"] == 6 and state["waiting"] == {"quick": [[3, 0]], "slow": [[3, 1], [3, 2]]}
        # The draws no committed step consumed: the kept groups under their own draws, the others under their new ones.
        assert state["outstanding"] == [
            [2, 5, "quick2", 2],
            [2, 9, "quick5", 5],
            [3, 0, "quick4", 4],
            [3, 1, "slow1", 1],
            [3, 2, "slow3", 3],
        ]
        results: asyncio.Queue[Any] = asyncio.Queue()
        dispatcher.bind(results)
        assert [results.get_nowait(), results.get_nowait()] == ["kept-a", "kept-b"]
        harness.run()
        await settle()
        # The draws drawn again first (the quick one, then the slow one), as draws 0 and 1 of start 3; then new draws
        # from position 6 until both labels are at their limits; the other slow draw waits its turn.
        assert [(draw.start, draw.seq, draw.position) for draw in harness.started[:2]] == [(3, 0, 4), (3, 1, 1)]
        assert harness.positions == [4, 1, 6, 7]
        harness.end(harness.started[1])
        await settle()
        assert harness.started[4].position == 3 and harness.started[4].key == (3, 2)
        await harness.close()

    asyncio.run(run())


def test_the_outstanding_draws_are_those_no_committed_step_consumed_and_none_skipped():
    async def run():
        rows = _labels("slow", "slow", "slow", "quick")
        harness = _Harness(rows, limits={"slow": 1}, labels=("slow", "quick"), batch_size=2, steps=3, backlog_limit=1)
        harness.run()
        await settle()
        # slow0 runs, slow1 waits, slow2 is skipped, quick3 runs, and so on to the run's six groups.
        outstanding = [draw[:2] for draw in harness.dispatcher.state()["outstanding"]]
        assert [0, 2] not in outstanding and outstanding[:3] == [[0, 0], [0, 1], [0, 3]]
        harness.dispatcher.committed([(0, 0), None, (0, 3)])  # a step consumed two of them
        outstanding = [draw[:2] for draw in harness.dispatcher.state()["outstanding"]]
        assert [0, 0] not in outstanding and [0, 3] not in outstanding and [0, 1] in outstanding
        await harness.close()

    asyncio.run(run())


def test_a_resume_plan_trains_the_kept_groups_puts_back_the_waiting_draws_and_draws_the_others_again():
    outstanding = [
        [1, 0, "t0", 0],
        [1, 1, "t1", 1],
        [1, 2, "t2", 2],
        [1, 3, "gone", 3],
        [1, 4, "t4", 4],
        [2, 0, "t5", 5],
    ]
    plan = plan_resume(
        outstanding=outstanding,
        waiting={"slow": [[1, 4], [7, 7]]},
        kept=[(2, 0), (1, 1), (9, 9)],  # in the order they settled; a kept group that is not outstanding is passed over
        pool={"t0", "t1", "t2", "t4", "t5"},
    )

    assert plan.kept == [((2, 0), "t5", 5), ((1, 1), "t1", 1)]
    assert plan.waiting == [((1, 4), "t4", 4)]
    assert plan.reissue == [((1, 0), "t0", 0), ((1, 2), "t2", 2)]  # in (start, seq) order
    assert plan.lost == [((1, 3), "gone", 3)]  # its task left the run
    assert sorted(plan.keys()) == sorted((start, seq) for start, seq, _task, _position in outstanding)
    with pytest.raises(ValueError, match="cannot read the outstanding draws"):
        plan_resume(outstanding=[[1, "x", "t0", 0]], waiting={}, kept=[], pool={"t0"})


def _datum(position: int) -> tinker.Datum:
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([position, 1]),
        loss_fn_inputs={"advantages": tinker.TensorData(data=[0.0, 0.5], dtype="float32", shape=[2])},
    )


def test_a_checkpoint_saves_the_dispatcher_s_state_and_its_kept_groups_in_files_of_their_own(tmp_path):
    """The state names the kept groups' file with its SHA-256, and the record the checkpoint's row keeps names both:
    a failed group is kept as failed, and a group whose output cannot be kept is not."""

    kept = [
        ((3, 0), SimpleNamespace(entry="a", label="a", record=None, output=[_datum(4)])),
        ((3, 1), SimpleNamespace(entry="a", label="a", record=None, output=None)),  # its workflow failed
        ((3, 2), SimpleNamespace(entry="a", label="a", record=None, output={"not": "kept"})),
    ]
    state = {"start": 3, "next_position": 9, "outstanding": [[3, 0, "a0", 4]], "waiting": {}, "fingerprint": {}}

    record = save_state(tmp_path, "recovery-000002", state, kept)

    assert record is not None
    assert (record["file"], record["kept"], record["start"]) == (
        "tasks/dispatch-s3-recovery-000002.json",
        "tasks/kept-s3-recovery-000002.json.gz",
        3,
    )
    saved = read_state(tmp_path, record)
    assert {key: saved[key] for key in state} == state and saved["kept"]["groups"] == [[3, 0], [3, 1]]
    groups = [decode_group(group) for group in read_kept(tmp_path, saved["kept"])]
    assert [(key, output is None) for key, _entry, _label, _record, output in groups] == [
        ((3, 0), False),
        ((3, 1), True),
    ]
    assert groups[0][4][0].model_input.to_ints() == [4, 1]
    (tmp_path / record["file"]).write_text(json.dumps({**saved, "next_position": 10}))
    with pytest.raises(ValueError, match="does not have the SHA-256 its checkpoint recorded"):
        read_state(tmp_path, record)


def test_a_state_or_kept_groups_that_cannot_be_written_cost_their_checkpoint_alone(tmp_path, monkeypatch, caplog):
    """A disk that refuses the kept groups leaves the state without them, so a resume draws their tasks again; one that
    refuses the state leaves the checkpoint without one, so a resume from it draws from its batch.  Neither raises."""

    real = dispatch_state_module.write_tasks_file
    refused: set[str] = set()

    def flaky(run_dir, name, content):
        if name.startswith(tuple(refused)):
            raise OSError(28, "No space left on device")
        return real(run_dir, name, content)

    monkeypatch.setattr(dispatch_state_module, "write_tasks_file", flaky)
    monkeypatch.setattr("platoon.train.checkpoints.kept_groups.write_tasks_file", flaky)
    kept = [((1, 0), SimpleNamespace(entry=None, label=None, record=None, output=[_datum(0)]))]
    state = {"start": 1, "next_position": 2, "outstanding": [[1, 0, "t0", 0]], "waiting": {}, "fingerprint": {}}

    with caplog.at_level(logging.WARNING):
        refused.add("kept-")
        record = save_state(tmp_path, "000001", state, kept)
        assert record is not None and "kept" not in record and read_state(tmp_path, record)["kept"] is None
        refused.add("dispatch-")
        assert save_state(tmp_path, "000002", state, kept) is None

    messages = [entry.getMessage() for entry in caplog.records]
    assert any("Could not write the 1 groups that settled" in message for message in messages)
    assert any("Could not write where the run's draws stand with checkpoint 000002" in message for message in messages)
    assert not list(Path(tmp_path, "tasks").glob(".*partial"))
