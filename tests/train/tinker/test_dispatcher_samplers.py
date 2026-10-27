"""The dispatcher draws through the run's task samplers (``platoon.train.sampler_runtime``), and what a full checkpoint
saves of them: its state of format 2 adds each outstanding draw's pool row and sampler and each sampler's count and
state to platoon 0.3.0's format 1, which a resume still reads."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from platoon.train.checkpoints.dispatch_state import STATE_FORMAT, read_state, save_state, saved_draws
from platoon.train.components import ENVIRONMENT_COLUMN
from platoon.train.dispatch.dispatcher import BoundDraw, Dispatcher
from platoon.train.dispatch.permits import PermitPool
from platoon.train.labels import task_label
from platoon.train.sampler_runtime import SamplerSet, TaskPool, TaskSampling
from platoon.train.samplers import SamplerChoice, UniformSampler, WeightedSampler
from platoon.train.task_stats import EntryInfo

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "resume-0.3.0"


class _Rows:
    def __init__(self, permits: int = 1000) -> None:
        self.permits = PermitPool(permits)

    def of(self, data: Mapping[str, Any]) -> tuple[str | None, str | None, int]:
        label = task_label(data, ENVIRONMENT_COLUMN)
        return label, label, 1


class _Recorder:
    def __init__(self) -> None:
        self.draws: list[dict[str, Any]] = []

    def draw(self, row, *, seq, position, version, reissues=None, sampler=None, reason=None, score=None) -> None:
        self.draws.append(
            {"task_id": row["task_id"], "position": position, "sampler": sampler, "reason": reason, "score": score}
        )

    def skipped(self, row) -> None:
        pass


def _sampling() -> TaskSampling:
    """A mixture of entries a (five tasks) and b (three), weighted two to one, as AutoDataset.training builds it."""

    rows = [{"task_id": f"a::{index}", ENVIRONMENT_COLUMN: "a"} for index in range(5)]
    rows += [{"task_id": f"b::{index}", ENVIRONMENT_COLUMN: "b"} for index in range(3)]
    pool = TaskPool.from_rows(rows, entries=["a", "b"], column=ENVIRONMENT_COLUMN)
    infos = {
        "a": EntryInfo(group_size=1, name="a", weight=2.0, seed=1, index=0, task_ids=pool.task_ids[:5]),
        "b": EntryInfo(group_size=1, name="b", weight=1.0, seed=2, index=1, task_ids=pool.task_ids[5:]),
    }
    uniform = SamplerChoice("uniform", UniformSampler)
    return TaskSampling(
        pool=pool,
        run=SamplerChoice("weighted", WeightedSampler),
        entries={"a": uniform, "b": uniform},
        infos=infos,
        epoch_size=8,
    )


def _dispatcher(**kwargs: Any) -> tuple[Dispatcher, _Recorder]:
    recorder = _Recorder()
    dispatcher = Dispatcher(
        rows=_Rows(),
        samplers=SamplerSet.build(_sampling(), batch_size=2, label_column=ENVIRONMENT_COLUMN),
        batch_size=2,
        max_staleness=None,
        steps=100,
        recorder=recorder,
        sampler="weighted",
        **kwargs,
    )
    return dispatcher, recorder


def _draw(dispatcher: Dispatcher, count: int) -> list[BoundDraw]:
    """``count`` new draws, as the dispatcher's loop makes them."""

    return [dispatcher._new_draw() for _ in range(count)]


def test_each_draw_names_its_pool_row_and_its_sampler_and_the_recorder_is_told_why():
    dispatcher, recorder = _dispatcher()
    draws = _draw(dispatcher, 6)

    pool = dispatcher.samplers.pool
    assert [pool.task_id(draw.index) for draw in draws] == [draw.task_id for draw in draws]
    assert [draw.sampler for draw in draws] == [f"uniform@{draw.entry}" for draw in draws]
    assert [draw["reason"] for draw in recorder.draws] == ["order"] * 6
    assert [draw["sampler"] for draw in recorder.draws] == [draw.sampler for draw in draws]
    state = dispatcher.state()
    assert state["outstanding"] == [[0, draw.seq, draw.task_id, draw.position] for draw in draws]
    assert state["drawn"] == [[0, draw.seq, draw.index, draw.sampler] for draw in draws]
    assert state["samplers"]["run"]["decisions"] == 6
    assert sum(record["decisions"] for record in state["samplers"]["entries"].values()) == 6


def test_a_checkpoint_saves_a_state_of_format_2_that_reads_back(tmp_path):
    dispatcher, _recorder = _dispatcher()
    _draw(dispatcher, 5)

    record = save_state(tmp_path, "000003", dispatcher.state(), [])
    assert record is not None and record["format"] == STATE_FORMAT == 2
    saved = read_state(tmp_path, record)
    assert saved["format"] == 2 and saved["drawn"] == dispatcher.state()["drawn"]
    assert saved["samplers"] == dispatcher.state()["samplers"]
    assert saved_draws(saved) == {(start, seq): (index, by) for start, seq, index, by in saved["drawn"]}


def test_a_dispatcher_restored_from_a_state_goes_on_with_the_same_draws():
    """A new dispatcher whose samplers take up the saved counts draws, from the saved position on, what the first
    goes on to draw."""

    first, _recorder = _dispatcher()
    _draw(first, 7)
    state = json.loads(json.dumps(first.state()))
    expected = [(draw.task_id, draw.index) for draw in _draw(first, 12)]

    second, _recorder = _dispatcher()
    second.next_position = state["next_position"]
    second.samplers.restore(state["samplers"], next_position=state["next_position"])
    assert [(draw.task_id, draw.index) for draw in _draw(second, 12)] == expected


@pytest.mark.parametrize("name", ["weighted-capped", "single", "uniform-mixture"])
def test_a_state_of_platoon_0_3_0_s_format_1_still_reads(tmp_path, name):
    run_dir = tmp_path / name
    shutil.copytree(FIXTURES / name, run_dir)
    row = json.loads((run_dir / "checkpoints.jsonl").read_text().splitlines()[-1])

    state = read_state(run_dir, row["dispatch"])
    assert state["format"] == 1 and "drawn" not in state and "samplers" not in state
    assert saved_draws(state) == {}


def test_a_state_of_another_format_does_not_read(tmp_path):
    dispatcher, _recorder = _dispatcher()
    record = save_state(tmp_path, "000001", dispatcher.state(), [])
    assert record is not None
    path = tmp_path / record["file"]
    data = json.loads(path.read_bytes())
    content = (json.dumps({**data, "format": 3}, separators=(",", ":")) + "\n").encode()
    path.write_bytes(content)
    with pytest.raises(ValueError, match="no dispatcher's state of format 1 or 2"):
        read_state(tmp_path, {**record, "sha256": hashlib.sha256(content).hexdigest()})


@pytest.mark.parametrize(
    "drawn",
    [[[1, 2, -1, "uniform@a"]], [[1, 2, True, None]], [[1, 2, "3", None]], [["x", 2, 3, None]], [[1, 2, 3]], 7],
)
def test_saved_rows_that_cannot_be_read_are_refused(drawn):
    with pytest.raises(ValueError, match="cannot read the outstanding draws' rows"):
        saved_draws({"drawn": drawn})


# --- the adaptive path: tokens, the token being started, permits first ------------------------------------------------

import asyncio  # noqa: E402

from platoon.experimental import (  # noqa: E402
    Draw,
    Sampler,
    Slot,  # noqa: E402
)
from platoon.train.checkpoints.dispatch_state import saved_fallback_slots, saved_token_positions  # noqa: E402
from platoon.train.dispatch.dispatcher import Token  # noqa: E402


class Chooser(Sampler):
    """An adaptive sampler: of its pool's eligible tasks, the one at its decision index, in turn; it counts the
    decisions it was asked and records whether its context's statistics were in view."""

    adaptive = True

    def __init__(self, setup, options=None) -> None:
        super().__init__(setup, options)
        self.asked: list[int] = []

    def next(self, ctx):
        assert ctx.stats is not None
        self.asked.append(ctx.decision)
        eligible = [task for task in ctx.task_ids if ctx.eligible(task)]
        if not eligible:
            return None
        return Draw(eligible[ctx.decision % len(eligible)], "chosen", 0.5)


class _SizedRows:
    """Rows by entry (their ENVIRONMENT_COLUMN), labelled by ``column``, each entry's group ``sizes``."""

    def __init__(self, permits: int, sizes: Mapping[str, int], column: str = ENVIRONMENT_COLUMN) -> None:
        self.permits = PermitPool(permits)
        self.sizes = dict(sizes)
        self.column = column

    def of(self, data: Mapping[str, Any]) -> tuple[str | None, str | None, int]:
        entry = data.get(ENVIRONMENT_COLUMN)
        return entry, task_label(data, self.column), self.sizes.get(entry, 1)


class _Events(_Recorder):
    def __init__(self) -> None:
        super().__init__()
        self.slots: list[dict[str, Any]] = []

    def slot_skipped(self, *, position, entry, why) -> None:
        self.slots.append({"position": position, "entry": entry, "why": why})


def _mixture(
    *,
    a: Any = Chooser,
    b: Any = UniformSampler,
    run: Any = WeightedSampler,
    sizes: Mapping[str, int] | None = None,
    kinds: bool = False,
    weights: Mapping[str, float] | None = None,
) -> TaskSampling:
    """Entries a and b, four tasks each, weighted one to one (or by ``weights``); a's sampler adaptive by default.  With
    ``kinds``, a's tasks carry two values of the column ``kind``."""

    sizes = dict(sizes or {"a": 1, "b": 1})
    weights = dict(weights or {"a": 1.0, "b": 1.0})
    rows = []
    for entry in ("a", "b"):
        for index in range(4):
            row = {"task_id": f"{entry}::{index}", ENVIRONMENT_COLUMN: entry}
            if kinds:
                row["kind"] = ("x" if index % 2 else "y") if entry == "a" else "z"
            rows.append(row)
    column = "kind" if kinds else ENVIRONMENT_COLUMN
    pool = TaskPool.from_rows(rows, entries=["a", "b"], column=column)
    infos = {
        entry: EntryInfo(
            group_size=sizes[entry],
            name=entry,
            weight=weights[entry],
            seed=7,
            index=index,
            task_ids=pool.task_ids[4 * index : 4 * index + 4],
        )
        for index, entry in enumerate(("a", "b"))
    }

    def choice(cls: Any) -> SamplerChoice:
        name = {UniformSampler: "uniform", WeightedSampler: "weighted"}.get(cls, cls.__name__.lower())
        return SamplerChoice(name, cls)

    return TaskSampling(
        pool=pool,
        run=choice(run),
        entries={"a": choice(a), "b": choice(b)} if getattr(run, "entry_slots", False) else {},
        infos=infos,
        epoch_size=8,
    )


class _Run:
    """A dispatcher over a sampling, whose groups run until the test ends them."""

    def __init__(
        self,
        sampling: TaskSampling,
        *,
        permits: int = 1000,
        sizes: Mapping[str, int] | None = None,
        column: str = ENVIRONMENT_COLUMN,
        **kwargs: Any,
    ) -> None:
        self.recorder = _Events()
        kwargs.setdefault("batch_size", 4)
        kwargs.setdefault("max_staleness", None)
        kwargs.setdefault("steps", 100)
        kwargs.setdefault("labels", sorted({sampling.pool.label(index) for index in range(len(sampling.pool))}))
        self.samplers = SamplerSet.build(sampling, batch_size=kwargs["batch_size"], label_column=column)
        self.dispatcher = Dispatcher(
            rows=_SizedRows(permits, sizes or {"a": 1, "b": 1}, column),
            samplers=self.samplers,
            recorder=self.recorder,
            sampler=sampling.run.name,
            **kwargs,
        )
        self.results: asyncio.Queue[Any] = asyncio.Queue()
        self.dispatcher.bind(self.results)
        self.started: list[BoundDraw] = []
        self.gates: dict[tuple[int, int], asyncio.Event] = {}
        self.tasks: list[asyncio.Task] = []
        self.loop: asyncio.Task | None = None

    def run(self, stop: asyncio.Event | None = None) -> None:
        self.loop = asyncio.ensure_future(self.dispatcher.run(spawn=self._spawn, run_group=self._group, stop=stop))
        self.tasks.append(self.loop)

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
        self.dispatcher.settle(draw, draw)

    def end(self, *draws: BoundDraw) -> None:
        for draw in draws:
            self.gates.setdefault(draw.key, asyncio.Event()).set()

    def chooser(self, entry: str | None = "a") -> Chooser:
        member = self.samplers.run if entry is None else self.samplers.entries[entry]
        return member.sampler

    async def close(self) -> None:
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


async def _quiet() -> None:
    for _ in range(50):
        await asyncio.sleep(0)


def test_a_full_save_while_a_token_waits_for_its_permits_lists_it_and_a_resume_decides_it_once():
    """A token taken from its queue waits for its rollout permits in no queue; a full save then lists it first among
    its entry's tokens, and a dispatcher restored from that save decides its position exactly once.
    The draws of both starts and the slots they skipped account for every position."""

    async def run():
        first = _Run(_mixture(b=Chooser), permits=1)
        first.run()
        await _quiet()
        assert [draw.position for draw in first.started] == [0]  # a's group at position 0 holds the only permit
        token = first.dispatcher._starting
        assert token is not None and token.entry == "b"  # the next slot's token waits for its permit
        waiting = token.position
        state = json.loads(json.dumps(first.dispatcher.state()))
        assert saved_token_positions(state)["b"][0] == waiting
        assert first.chooser("a").asked == [0] and first.chooser("b").asked == []  # no decision made for it
        await first.close()  # the crash: nothing after the save

        second = _Run(_mixture(b=Chooser), permits=1)
        second.dispatcher.next_position = state["next_position"]
        second.samplers.restore(state["samplers"], next_position=state["next_position"])
        reissue = [
            ((start, seq), first.started[0].row, position) for start, seq, _task, position in state["outstanding"]
        ]
        second.dispatcher.restore(kept=[], reissue=reissue, waiting=[], version=0)
        second.dispatcher.restore_tokens(saved_token_positions(state), version=0)
        second.run()
        for _ in range(6):
            await _quiet()
            if second.started:
                second.end(*second.started)
        decided = [draw.position for draw in second.started]
        assert decided.count(waiting) == 1 and decided[:2] == [0, waiting]
        assert second.chooser("b").asked[0] == state["samplers"]["entries"]["b"]["decisions"] == 0
        positions = {draw["position"] for draw in first.recorder.draws + second.recorder.draws}
        positions |= {slot["position"] for slot in second.recorder.slots}
        queued = {position for tokens in second.dispatcher.state()["tokens"].values() for position in tokens}
        assert positions | queued == set(range(second.dispatcher.next_position))
        await second.close()

    asyncio.run(run())


def test_a_save_while_no_token_waits_lists_none_as_being_started():
    async def run():
        harness = _Run(_mixture())
        harness.run()
        await _quiet()
        assert harness.dispatcher._starting is None
        assert all(not tokens for tokens in harness.dispatcher._tokens.values())
        assert harness.dispatcher.state()["tokens"] == {}
        await harness.close()

    asyncio.run(run())


def test_a_token_is_decided_only_with_its_permits_held():
    """A budget of one group: the second decision comes after the first group returns its permits, not before."""

    async def run():
        harness = _Run(_mixture(b=Chooser, sizes={"a": 2, "b": 2}), permits=2, sizes={"a": 2, "b": 2})
        harness.run()
        await _quiet()
        asked = harness.chooser("a").asked + harness.chooser("b").asked
        assert len(harness.started) == 1 and len(asked) == 1
        harness.end(harness.started[0])
        await _quiet()
        asked = harness.chooser("a").asked + harness.chooser("b").asked
        assert len(harness.started) == 2 and len(asked) == 2
        await harness.close()

    asyncio.run(run())


def test_queued_tokens_hold_no_slot_and_a_full_entry_is_not_asked():
    """While a's one group runs (its limit is 1), its slots wait as tokens, which hold no slot of the horizon and are
    not decided: its sampler is not asked until the group ends."""

    async def run():
        harness = _Run(_mixture(), limits={"a": 1}, max_staleness=1)
        harness.run()
        await _quiet()
        a_groups = [draw for draw in harness.started if draw.entry == "a"]
        assert len(a_groups) == 1 and len(harness.chooser().asked) == 1
        tokens = harness.dispatcher.tokens_by_entry()["a"]
        assert tokens >= 1
        assert harness.dispatcher.pipeline == len(harness.started)  # no slot for a queued token
        assert harness.dispatcher.parked_by_label()["a"] == tokens
        harness.end(a_groups[0])
        await _quiet()
        assert len(harness.chooser().asked) == 2  # the next token, decided once a's label had room
        await harness.close()

    asyncio.run(run())


def test_tokens_past_the_backlog_limit_are_skipped_and_their_positions_used_up():
    async def run():
        harness = _Run(_mixture(), limits={"a": 1}, backlog_limit=2, max_staleness=None, steps=5)
        harness.run()
        await _quiet()
        # a runs one group and b's run until the plan's twenty: a's tokens wait, and those past two are skipped,
        # since b's label has room and the order goes on.
        skipped = [slot for slot in harness.recorder.slots if slot["why"] == "backlog"]
        assert harness.dispatcher.skipped_slots == len(skipped) >= 1
        assert len(harness.dispatcher._tokens["a"]) == 2
        drawn = {draw["position"] for draw in harness.recorder.draws}
        queued = {token.position for token in harness.dispatcher._tokens["a"]}
        assert drawn | queued | {slot["position"] for slot in skipped} == set(range(harness.dispatcher.next_position))
        await harness.close()

    asyncio.run(run())


def test_a_capped_adaptive_entry_s_next_group_starts_as_one_ends_through_a_held_slot():
    async def run():
        harness = _Run(_mixture(), limits={"a": 1}, max_staleness=1, batch_size=4)
        harness.run()
        await _quiet()
        assert harness.dispatcher._held() == 1  # a's running group holds a slot for its next token
        started = len(harness.started)
        harness.end(next(draw for draw in harness.started if draw.entry == "a"))
        await _quiet()
        assert len(harness.started) == started + 1 and harness.started[-1].entry == "a"  # no step started
        await harness.close()

    asyncio.run(run())


def test_an_entry_of_several_labels_holds_no_slot():
    async def run():
        harness = _Run(_mixture(kinds=True), column="kind", limits={"x": 1, "y": 1}, max_staleness=1)
        harness.run()
        await _quiet()
        assert harness.dispatcher.tokens_by_entry().get("a", 0) >= 1
        assert harness.dispatcher._held() == 0
        await harness.close()

    asyncio.run(run())


def test_a_pool_wide_decision_holds_the_largest_group_s_permits_returns_the_rest_and_takes_its_position():
    async def run():
        sizes = {"a": 2, "b": 4}
        harness = _Run(_mixture(run=Chooser, sizes=sizes), permits=4, sizes=sizes)
        harness.run()
        await _quiet()
        assert len(harness.started) == 1
        first = harness.started[0]
        assert (first.task_id, first.position, first.group_size) == ("a::0", 0, 2)  # decision 0
        # It held 4 permits, the largest group's, and returned 2; the next decision waits for 4.
        assert harness.dispatcher._rows.permits._free == 2 and harness.dispatcher.next_position == 1
        harness.end(first)
        await _quiet()
        assert [(draw.task_id, draw.position) for draw in harness.started] == [("a::0", 0), ("a::1", 1)]
        await harness.close()

    asyncio.run(run())


def test_a_stop_request_while_a_token_waits_makes_no_decision_and_keeps_it():
    async def run():
        harness = _Run(_mixture(b=Chooser), permits=1)
        stop = asyncio.Event()
        harness.run(stop)
        await _quiet()
        token = harness.dispatcher._starting
        assert token is not None and harness.chooser("b").asked == []
        stop.set()
        harness.end(harness.started[0])  # its permit comes back while the stop request stands
        await _quiet()
        assert harness.chooser("b").asked == []
        assert harness.dispatcher._starting is None and harness.dispatcher._tokens[token.entry][0] is token
        assert harness.dispatcher.state()["tokens"][token.entry][0] == token.position
        await harness.close()

    asyncio.run(run())


def test_a_cancellation_while_a_token_waits_puts_it_back_first_in_its_queue():
    async def run():
        harness = _Run(_mixture(b=Chooser), permits=1)
        harness.run()
        await _quiet()
        token = harness.dispatcher._starting
        assert token is not None
        assert harness.loop is not None
        harness.loop.cancel()
        await asyncio.gather(harness.loop, return_exceptions=True)
        assert harness.dispatcher._starting is None and harness.dispatcher._tokens[token.entry][0] is token
        assert harness.dispatcher.pipeline == 1  # the running group's alone
        await harness.close()

    asyncio.run(run())


@pytest.mark.parametrize("staleness", [0, 1, 3])
def test_the_pipeline_never_exceeds_k_plus_one_batches_with_adaptive_entries(staleness):
    async def run():
        harness = _Run(_mixture(), limits={"a": 2}, max_staleness=staleness, batch_size=4, steps=6)
        harness.run()
        rng = __import__("random").Random(staleness)
        for _step in range(6):
            for _ in range(20):
                await _quiet()
                running = [
                    draw for draw in harness.started if not harness.gates.get(draw.key, asyncio.Event()).is_set()
                ]
                if running:
                    harness.end(rng.choice(running))
            for _ in range(4):
                harness.dispatcher.dequeued()
            harness.dispatcher.step_started()
        assert harness.dispatcher.peak <= (staleness + 1) * 4
        await harness.close()

    asyncio.run(run())


def test_a_state_with_tokens_round_trips_and_a_resume_binds_drops_or_queues_them(tmp_path):
    """Tokens saved with the state come back: an adaptive entry's to its queue, in order; an entry whose sampler is
    static now binds them, in order, as waiting draws; an entry that no longer gets slots, or left the run, drops them
    with a slot_skipped event each."""

    async def run():
        harness = _Run(_mixture(), limits={"a": 1}, max_staleness=None)
        harness.run()
        await _quiet()
        state = harness.dispatcher.state()
        await harness.close()
        record = save_state(tmp_path, "000001", state, [])
        assert record is not None
        tokens = saved_token_positions(read_state(tmp_path, record))
        assert tokens == state["tokens"] and len(tokens["a"]) >= 1
        return state, tokens

    state, tokens = asyncio.run(run())
    adaptive = _Run(_mixture())
    adaptive.samplers.restore(state["samplers"], next_position=state["next_position"])
    assert adaptive.dispatcher.restore_tokens(tokens, version=0) == {
        "restored": len(tokens["a"]),
        "bound": 0,
        "dropped": 0,
    }
    assert [token.position for token in adaptive.dispatcher._tokens["a"]] == tokens["a"]
    static = _Run(_mixture(a=UniformSampler))
    static.samplers.restore(state["samplers"], next_position=state["next_position"])
    assert static.dispatcher.restore_tokens(tokens, version=0)["bound"] == len(tokens["a"])
    waiting = static.dispatcher._waiting["a"]
    assert [draw.position for draw in waiting] == tokens["a"] and all(draw.sampler == "uniform@a" for draw in waiting)
    assert static.samplers.entries["a"].decisions == static.samplers._slot_counts(state["next_position"])["a"]
    gone = _Run(_mixture(run=Chooser))
    assert gone.dispatcher.restore_tokens({"a": tokens["a"], "c": [99]}, version=0)["dropped"] == len(tokens["a"]) + 1
    assert [slot["why"] for slot in gone.recorder.slots] == ["no_slots"] * len(tokens["a"]) + ["entry_left"]


def test_a_token_is_an_item_of_its_entry_s_queue():
    token = Token(position=3, entry="a", ticket=0)
    assert (token.position, token.entry, token.ticket) == (3, "a", 0)


# --- a sampler's state as a save took it, the run's fallback slots, and a pool-wide decision's permits ------------


class LiveState(Chooser):
    """A Chooser that keeps each decision it made in a dict, which its state_dict returns as it is."""

    def __init__(self, setup, options=None) -> None:
        super().__init__(setup, options)
        self.made: dict[str, str] = {}

    def next(self, ctx):
        answer = super().next(ctx)
        if answer is not None:
            self.made[str(ctx.decision)] = answer.task_id
        return answer

    def state_dict(self):
        return {"made": self.made}

    def load_state_dict(self, state):
        self.made = dict(state["made"]) if state else {}


def test_a_full_save_writes_a_sampler_s_state_as_it_stood_when_the_save_took_it():
    """A full save takes the dispatcher's state at one point and writes it in a thread while the dispatcher goes on
    deciding: the state it writes holds the decisions its count holds, not the ones made meanwhile."""

    async def run():
        harness = _Run(_mixture(a=LiveState), permits=1)
        harness.run()
        await _quiet()
        state = harness.dispatcher.state()
        for _ in range(6):  # the dispatcher goes on while the thread writes
            harness.end(*harness.started)
            await _quiet()
        written = json.loads(json.dumps(state))
        decided = harness.samplers.entries["a"].decisions
        await harness.close()
        return written["samplers"]["entries"]["a"], decided

    record, decided = asyncio.run(run())
    assert decided > record["decisions"] >= 1
    assert len(record["state"]["made"]) == record["decisions"]


class FailingSlots(Sampler):
    """A static run sampler that hands slots: it fails at decisions 1 and 3 (it raises, then hands a slot to no entry)
    and hands every other slot to b, so the slots of its fallback, by weight, go to a alone (b weighs nothing)."""

    entry_slots = True

    def next(self, ctx):
        if ctx.decision == 1:
            raise RuntimeError("a bug")
        if ctx.decision == 3:
            return Slot("nobody")
        return Slot("b")


def test_a_slot_the_run_sampler_s_fallback_owes_an_adaptive_entry_draws_as_the_run_s_fallback():
    """The fallback's slot to an adaptive entry waits as a token, and the draw that fills it is the run sampler's
    fallback, counted in draws/fallback, as a fallback's slot to an entry with a static sampler is."""

    async def run():
        harness = _Run(_mixture(run=FailingSlots, weights={"a": 1.0, "b": 0.0}), steps=1, batch_size=4)
        harness.run()
        await _quiet()
        await harness.close()
        return harness

    harness = asyncio.run(run())
    drawn = [(draw["position"], draw["sampler"], draw["reason"]) for draw in harness.recorder.draws]
    assert drawn == [
        (0, "uniform@b", "order"),
        (1, "failingslots", "fallback"),
        (2, "uniform@b", "order"),
        (3, "failingslots", "fallback"),
    ]
    assert harness.chooser("a").asked == [0, 1]  # a's sampler chose the tasks of both slots
    metrics = harness.samplers.step_metrics(0)
    assert metrics["sampler/draws/fallback"] == 2.0 and harness.samplers.fallbacks()["failingslots"] == 2


def test_the_fallback_s_slots_waiting_as_tokens_stay_the_run_s_fallback_across_a_resume():
    async def run():
        sampling = _mixture(run=FailingSlots, weights={"a": 1.0, "b": 0.0})
        harness = _Run(sampling, limits={"a": 1}, steps=3, batch_size=4)
        harness.run()
        await _quiet()
        state = json.loads(json.dumps(harness.dispatcher.state()))
        await harness.close()
        return state

    state = asyncio.run(run())
    assert state["tokens"] == {"a": [3]} and state["fallback_slots"] == [3]  # a's one group runs: position 3 waits
    tokens, fallbacks = saved_token_positions(state), saved_fallback_slots(state)
    adaptive = _Run(_mixture(run=FailingSlots, weights={"a": 1.0, "b": 0.0}))
    adaptive.samplers.restore(state["samplers"], next_position=state["next_position"])
    adaptive.dispatcher.restore_tokens(tokens, version=0, fallbacks=fallbacks)
    assert [(token.position, token.fallback) for token in adaptive.dispatcher._tokens["a"]] == [(3, True)]
    static = _Run(_mixture(run=FailingSlots, a=UniformSampler, weights={"a": 1.0, "b": 0.0}))
    static.samplers.restore(state["samplers"], next_position=state["next_position"])
    static.dispatcher.restore_tokens(tokens, version=0, fallbacks=fallbacks)
    [waiting] = static.dispatcher._waiting["a"]
    assert (waiting.position, waiting.sampler) == (3, "failingslots")
    assert static.recorder.draws[-1]["reason"] == "fallback"
    with pytest.raises(ValueError, match="fallback"):
        saved_fallback_slots({"fallback_slots": ["3"]})


def test_a_pool_wide_decision_holds_only_the_permits_of_the_entries_that_can_start():
    """Entries a (groups of 4, at most one at once) and b (groups of 1), five permits: while a's group runs, the next
    decision over the pool holds one permit, for b's tasks, the only ones eligible, rather than waiting for four."""

    async def run():
        sizes = {"a": 4, "b": 1}
        harness = _Run(_mixture(run=Chooser, sizes=sizes), permits=5, sizes=sizes, limits={"a": 1})
        harness.run()
        await _quiet()
        started = [(draw.task_id, draw.group_size) for draw in harness.started]
        await harness.close()
        return started

    started = asyncio.run(run())
    assert started[0] == ("a::0", 4)
    assert len(started) == 2 and started[1][0].startswith("b::") and started[1][1] == 1
