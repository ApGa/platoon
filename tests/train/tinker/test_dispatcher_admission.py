"""The dispatcher under a group admission rule (RFC 0019, section 6), on its own, its groups gated: a rejected group
leaves the outstanding draws and frees its slot at once, so the next group starts while no step dequeues, but one that
failed or carried no datum keeps its slot until a step dequeues its marker; a step counts each group it dequeues and
each marker of a group rejected while it was current, an earlier step's marker passing over (:meth:`Dispatcher.took`);
at ``max_staleness: 0`` and on the run's last step no group starts that the current step could not count, and the
step's counts add up to what it started; and a marker carries a group's statistics, not its datums (``emptied``)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
import tinker
from test_dispatcher import _Recorder, _Rows, _settle

from platoon.train.tinker.dispatcher import BoundDraw, Dispatcher
from platoon.train.tinker.rl import _Settled, emptied


class _Harness:
    """A dispatcher under a rule over ``rows`` (the order repeats them), whose groups run until the test ends them,
    each admitted (settled) or rejected (its marker sent), as the test says (:meth:`end`)."""

    def __init__(self, rows: list[dict[str, Any]], *, cap: int, permits: int = 1000, **kwargs: Any) -> None:
        kwargs.setdefault("batch_size", 4)
        kwargs.setdefault("max_staleness", None)
        kwargs.setdefault("steps", 100)
        self.dispatcher = Dispatcher(
            rows=_Rows(permits),
            order=lambda position: rows[position % len(rows)],
            recorder=_Recorder(),
            groups_per_step=cap,
            **kwargs,
        )
        self.results: asyncio.Queue[Any] = asyncio.Queue()
        self.dispatcher.bind(self.results)
        self.started: list[BoundDraw] = []
        self.gates: dict[tuple[int, int], asyncio.Event] = {}
        self.admit: dict[tuple[int, int], bool] = {}
        self.hold: dict[tuple[int, int], bool] = {}
        self.tasks: list[asyncio.Task] = []

    def run(self) -> None:
        self.tasks.append(asyncio.ensure_future(self.dispatcher.run(spawn=self._spawn, run_group=self._group)))

    def _spawn(self, coroutine) -> asyncio.Task:
        task = asyncio.ensure_future(coroutine)
        self.tasks.append(task)
        return task

    async def _group(self, draw: BoundDraw, release) -> None:
        self.started.append(draw)
        try:
            await self.gates.setdefault(draw.key, asyncio.Event()).wait()
        finally:
            release()
        if self.admit.get(draw.key, True):
            self.dispatcher.settle(draw, _Settled(None, draw.label, [], None, draw.key))
        else:
            held = self.hold.get(draw.key, False)
            marker = _Settled(
                None,
                draw.label,
                [],
                None,
                None,
                rejected="no_results" if held else "uniform_pass",
                window=self.dispatcher.step,
                held=held,
            )
            self.dispatcher.reject(draw, marker, hold=held)

    def end(self, *draws: BoundDraw, admit: bool = True, hold: bool = False) -> None:
        """End the groups of ``draws``: admitted, or rejected (with ``hold``, as a group that carried no datum is)."""

        for draw in draws:
            self.admit[draw.key] = admit
            self.hold[draw.key] = hold
            self.gates.setdefault(draw.key, asyncio.Event()).set()

    def take(self) -> tuple[Any, bool]:
        """What a step dequeues next, and whether it counts toward its cap."""

        item = self.results.get_nowait()
        if item.rejected is not None:
            return item, self.dispatcher.took("marker", item.window, held=item.held)
        return item, self.dispatcher.took("admitted")

    def accounted(self) -> int:
        dispatcher = self.dispatcher
        return dispatcher.counted + dispatcher.pipeline + dispatcher._markers_now

    async def close(self) -> None:
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


def _rows(count: int = 400) -> list[dict[str, Any]]:
    return [{"task_id": f"t{index}", "env": "a"} for index in range(count)]


def test_a_rejection_frees_its_slot_at_once_and_its_draw_is_no_longer_outstanding():
    async def run():
        harness = _Harness(_rows(), cap=24, max_staleness=1, batch_size=2)
        harness.run()
        await _settle()
        assert len(harness.started) == 4  # the horizon: (k + 1) × batch_size
        first = harness.started[0]
        harness.end(first, admit=False)
        await _settle()
        dispatcher = harness.dispatcher
        assert len(harness.started) == 5  # the next group started while no step dequeued
        assert first.key not in dispatcher._outstanding and first.key not in dict(dispatcher.kept())
        assert dispatcher.pipeline == 4 and dispatcher._markers_now == 1
        harness.end(harness.started[1])  # an admitted one holds its slot until a step dequeues it
        await _settle()
        assert len(harness.started) == 5 and harness.started[1].key in dict(dispatcher.kept())
        state = dispatcher.state()
        assert first.key not in {(start, seq) for start, seq, _task, _position in state["outstanding"]}
        await harness.close()

    asyncio.run(run())


def test_a_group_that_ended_without_a_datum_keeps_its_slot_until_a_step_dequeues_its_marker():
    """A failed group, or one whose output carries no datum, can end at once: it keeps its slot, so that groups that
    end at once start no faster than steps dequeue their markers, while a rejected group with datums frees its slot at
    once.  Dequeued, the marker frees the slot and counts for the step it was rejected in."""

    async def run():
        harness = _Harness(_rows(), cap=24, max_staleness=1, batch_size=2)
        harness.run()
        await _settle()
        assert len(harness.started) == 4
        dispatcher = harness.dispatcher
        held, freed = harness.started[0], harness.started[1]
        harness.end(held, admit=False, hold=True)
        await _settle()
        assert len(harness.started) == 4  # no group started in its slot
        assert held.key not in dispatcher._outstanding and held.key not in dict(dispatcher.kept())
        assert (dispatcher.pipeline, dispatcher._markers_now) == (4, 0)
        harness.end(freed, admit=False)
        await _settle()
        assert len(harness.started) == 5 and (dispatcher.pipeline, dispatcher._markers_now) == (4, 1)
        marker, counts = harness.take()
        assert marker.held and counts and (dispatcher.pipeline, dispatcher.counted) == (3, 1)
        await _settle()
        assert len(harness.started) == 6  # its slot freed once a step dequeued it
        marker, counts = harness.take()
        assert not marker.held and counts and (dispatcher.pipeline, dispatcher._markers_now) == (4, 0)
        await harness.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("kind", "window", "held", "counts", "pipeline", "admitted", "markers"),
    [
        ("admitted", None, False, True, 2, 1, 1),
        ("stale", None, False, True, 2, 0, 1),
        ("failed", None, False, True, 2, 0, 1),
        ("marker", 5, False, True, 3, 0, 0),
        ("marker", 4, False, False, 3, 0, 1),
        ("marker", 5, True, True, 2, 0, 1),  # it kept its slot: it frees it now, and was no queued marker
        ("marker", 4, True, False, 2, 0, 1),  # an earlier step's that kept its slot frees it too
    ],
)
def test_took_counts_each_kind_as_its_table_says(kind, window, held, counts, pipeline, admitted, markers):
    async def run():
        dispatcher = Dispatcher(
            rows=_Rows(), order=lambda position: {}, batch_size=4, max_staleness=1, steps=10, groups_per_step=12, step=5
        )
        dispatcher.pipeline, dispatcher._markers_now = 3, 1
        dispatcher._changed.clear()
        assert dispatcher.took(kind, window, held=held) is counts
        assert (dispatcher.pipeline, dispatcher.admitted, dispatcher._markers_now) == (pipeline, admitted, markers)
        assert dispatcher.counted == int(counts)
        assert dispatcher._changed.is_set()  # every time: a stale or kept failed group frees a slot too

    asyncio.run(run())


def test_at_k_0_a_step_starts_at_most_its_cap_and_its_counts_add_up_to_what_it_started():
    """Batch 4, cap 8, one group in two rejected: the horizon refills each rejected slot, the item bound stops the
    step at 8 groups started, and after every event counted + pipeline + queued markers equals the groups started and
    never decreases."""

    async def run():
        harness = _Harness(_rows(), cap=8, max_staleness=0, batch_size=4)
        dispatcher = harness.dispatcher
        harness.run()
        await _settle()
        seen: list[int] = []

        def check() -> None:
            assert harness.accounted() == len(harness.started)
            seen.append(harness.accounted())
            assert seen == sorted(seen) and dispatcher._markers_now >= 0

        assert len(harness.started) == 4
        check()
        for index in range(8):
            draw = harness.started[index]
            harness.end(draw, admit=index % 2 == 1)
            await _settle()
            check()
            while not harness.results.empty():
                harness.take()
                check()
        assert len(harness.started) == 8  # four rejections refilled, and no more: the cap
        assert dispatcher.counted == 8 and dispatcher.admitted == 4 and dispatcher.pipeline == 0
        dispatcher.step_started()
        assert dispatcher.counted == dispatcher._markers_now == 0
        await _settle()
        assert len(harness.started) == 12  # the next step starts its batch
        await harness.close()

    asyncio.run(run())


def test_at_k_0_a_step_closed_by_its_cap_leaves_nothing_running():
    async def run():
        harness = _Harness(_rows(), cap=6, max_staleness=0, batch_size=4)
        harness.run()
        await _settle()
        for index in range(6):
            harness.end(harness.started[index], admit=False)
            await _settle()
        dispatcher = harness.dispatcher
        assert len(harness.started) == 6  # every one rejected: the item bound stops at the cap
        assert dispatcher.pipeline == 0 and dispatcher._markers_now == 6
        counted = [harness.take()[1] for _ in range(6)]
        assert all(counted) and dispatcher.counted == 6
        await harness.close()

    asyncio.run(run())


def test_an_earlier_step_s_markers_pass_over_without_counting():
    async def run():
        harness = _Harness(_rows(), cap=24, max_staleness=1, batch_size=2)
        dispatcher = harness.dispatcher
        harness.run()
        await _settle()
        harness.end(harness.started[0], admit=False)  # rejected while step 0 is current
        await _settle()
        dispatcher.step_started()  # its marker is an earlier step's now
        harness.end(harness.started[1], admit=False)  # rejected while step 1 is current
        await _settle()
        assert dispatcher._markers_now == 1
        first, counts_first = harness.take()
        second, counts_second = harness.take()
        assert (first.window, counts_first) == (0, False)
        assert (second.window, counts_second) == (1, True)
        assert dispatcher.counted == 1 and dispatcher._markers_now == 0
        await harness.close()

    asyncio.run(run())


@pytest.mark.parametrize("staleness", [3, None])
def test_on_the_last_step_no_group_starts_that_it_could_not_count(staleness):
    """The run's last step (one step left) at k = 3, and without a staleness bound: the end-of-plan bound lets four
    groups run at once, and the item bound stops the step at its cap of 6 started, however many are rejected."""

    async def run():
        harness = _Harness(_rows(), cap=6, max_staleness=staleness, batch_size=4, steps=1)
        harness.run()
        await _settle()
        assert len(harness.started) == 4
        for index in range(6):
            harness.end(harness.started[index], admit=False)
            await _settle()
            assert harness.accounted() == len(harness.started) <= 6
        assert len(harness.started) == 6
        await harness.close()

    asyncio.run(run())


def test_without_a_rule_nothing_counts_and_no_item_bound_applies():
    async def run():
        harness = _Harness(_rows(), cap=6, max_staleness=0, batch_size=4)
        harness.dispatcher.groups_per_step = None  # all mode
        harness.run()
        await _settle()
        assert len(harness.started) == 4 and harness.dispatcher.counted == 0
        await harness.close()

    asyncio.run(run())


# --- capped labels under a rule ---------------------------------------------------------------------------------------


def _labelled(*names: str) -> list[dict[str, Any]]:
    return [{"task_id": f"{name}{index}", "env": name} for index, name in enumerate(names)]


def test_under_a_rule_a_capped_label_s_next_group_starts_as_one_ends_and_a_rejection_frees_the_horizon():
    """M3's held slot under a rule: the capped label's running group holds a slot for its next group, which starts as
    the group ends, whether it was admitted or rejected; a rejected group of the other label frees its slot at once."""

    async def run():
        harness = _Harness(_labelled(*["a", "b"] * 40), cap=24, limits={"a": 1}, labels=("a", "b"), max_staleness=1)
        harness.run()
        await _settle()
        assert [draw.position for draw in harness.started] == [0, 1, 3, 5, 7, 9, 11]  # a slot held for a's next
        harness.end(harness.started[0], admit=False)  # a0 rejected: a's next group takes the held slot
        await _settle()
        assert [draw.position for draw in harness.started][7:] == [2]
        harness.end(harness.started[1], admit=False)  # b1 rejected: its slot frees, and b's next group starts
        await _settle()
        assert [draw.position for draw in harness.started][8:] == [13]
        harness.end(harness.started[3])  # b3 admitted: it holds its slot until a step dequeues it
        await _settle()
        assert len(harness.started) == 9
        await harness.close()

    asyncio.run(run())


def test_under_a_rule_no_draw_while_every_label_is_at_its_limit():
    async def run():
        harness = _Harness(_labelled(*["a", "b"] * 40), cap=24, limits={"a": 1, "b": 1}, labels=("a", "b"))
        draws: list[bool] = []
        new_draw = Dispatcher._new_draw

        def watched() -> Any:
            draws.append(any(harness.dispatcher._active[label] < limit for label, limit in {"a": 1, "b": 1}.items()))
            return new_draw(harness.dispatcher)

        harness.dispatcher._new_draw = watched  # type: ignore[method-assign]
        harness.run()
        await _settle()
        for _ in range(6):  # each rejection frees its label at once: the label's next draw starts
            harness.end(harness.started[-1], admit=False)
            await _settle()
        assert draws and all(draws) and len(harness.started) == 8
        await harness.close()

    asyncio.run(run())


# --- a marker's output ------------------------------------------------------------------------------------------------


def _datum() -> tinker.Datum:
    return tinker.Datum(model_input=tinker.ModelInput.from_ints([1, 2]), loss_fn_inputs={})


class _Output(list):
    pass


class _Frozen(list):
    def __delitem__(self, index):
        raise TypeError("frozen")


def test_a_marker_keeps_a_group_s_statistics_not_its_datums():
    output = _Output([_datum(), _datum()])
    output.rollout_stats = SimpleNamespace(groups=1)
    output.verdict = "uniform"
    marker = emptied(output)
    assert marker is output and len(marker) == 0  # emptied in place
    assert marker.rollout_stats.groups == 1 and marker.verdict == "uniform"
    assert emptied((_datum(),)) == [] and emptied(None) == []
    assert emptied(_Frozen([_datum()])) == []  # a list that refuses: replaced, never raised
