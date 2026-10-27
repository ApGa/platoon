"""The run's task samplers as the dispatcher asks them (``platoon.train.sampler_runtime``): the task pool, the checks of
each answer and the fallback draw that replaces a refused one, and the samplers' counts and states across a resume."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from datasets import Dataset

from platoon.train.components import ENVIRONMENT_COLUMN
from platoon.train.sampler_api import Draw, Sampler, SamplerContext, SamplerError, Slot, decision_key
from platoon.train.sampler_runtime import SamplerSet, TaskPool, TaskSampling, _Row
from platoon.train.samplers import SamplerChoice, UniformSampler, WeightedSampler
from platoon.train.task_stats import EntryInfo

# --- the pool ---------------------------------------------------------------------------------------------------------


def _rows() -> list[dict[str, Any]]:
    return [
        {"task_id": "a::0", ENVIRONMENT_COLUMN: "a", "kind": "hard"},
        {"task_id": "a::1", ENVIRONMENT_COLUMN: "a", "kind": 3},
        {"task_id": "b::0", ENVIRONMENT_COLUMN: "b", "kind": None},
        {"task_id": "a::0", ENVIRONMENT_COLUMN: "a", "kind": "easy"},
    ]


@pytest.mark.parametrize("form", ["rows", "dataset"])
def test_a_pool_knows_each_row_s_task_entry_and_label_and_draws_each_row_afresh(form):
    rows = _rows()
    if form == "rows":
        pool = TaskPool.from_rows(rows, entries=["a", "b"], column="kind")
    else:  # a dataset's column has one type
        kinds = [None if row["kind"] is None else str(row["kind"]) for row in rows]
        columns = {**{key: [row[key] for row in rows] for key in rows[0]}, "kind": kinds}
        pool = TaskPool.from_dataset(Dataset.from_dict(columns), entries=["a", "b"], column="kind")

    assert pool.task_ids == ("a::0", "a::1", "b::0", "a::0") and len(pool) == 4
    assert [pool.entry(index) for index in range(4)] == ["a", "a", "b", "a"]
    assert pool.entry_indices("a") == (0, 1, 3) and pool.entry_indices("b") == (2,) and pool.entry_indices("c") == ()
    if form == "rows":
        assert [pool.label(index) for index in range(4)] == ["hard", "3", None, "easy"]
    assert pool.first_index("a::0") == 0 and pool.first_index("x") is None
    assert "b::0" in pool and "x" not in pool
    first, again = pool.row(3), pool.row(3)
    assert first == again and first is not again and first["kind"] == "easy"


def test_a_pool_without_an_environment_column_puts_a_single_environment_s_rows_in_its_entry():
    pool = TaskPool.from_dataset(Dataset.from_dict({"task_id": ["t0", "t1"]}), entries=["only"], column="kind")
    assert [pool.entry(index) for index in range(2)] == ["only", "only"]
    two = TaskPool.from_rows([{"task_id": "t0"}], entries=["x", "y"], column=ENVIRONMENT_COLUMN)
    assert two.entry(0) is None and two.entry_indices("x") == ()


def test_a_pool_leaves_out_the_columns_it_drops():
    pool = TaskPool.from_rows(
        [{"task_id": "t0", "_platoon_preserve_dataset_order": True}],
        entries=["only"],
        column=ENVIRONMENT_COLUMN,
        drop_columns=("_platoon_preserve_dataset_order",),
    )
    assert pool.row(0) == {"task_id": "t0"}


def test_a_reissued_draw_takes_its_row_while_that_row_holds_its_task_and_else_the_first_row_of_its_task():
    pool = TaskPool.from_rows(_rows(), entries=["a", "b"], column="kind")

    assert pool.reissue_row("a::0", 3) == (pool.row(3), 3)  # its own row, not the first of its task
    assert pool.reissue_row("a::0", None) == (pool.row(0), 0)  # a state of format 1 saved no row
    assert pool.reissue_row("a::0", 2) == (pool.row(0), 0)  # row 2 holds another task now: the pool changed
    assert pool.reissue_row("a::0", 99) == (pool.row(0), 0)
    assert pool.reissue_row("gone", 1) is None


# --- answers, and the fallback ----------------------------------------------------------------------------------------


class _Answers(Sampler):
    """A static sampler that answers ``answers[decision]`` (raising an exception it holds)."""

    answers: list[Any] = []

    def next(self, ctx: SamplerContext) -> Any:
        answer = type(self).answers[ctx.decision]
        if isinstance(answer, BaseException):
            raise answer
        return answer


def _pool(count: int = 6) -> TaskPool:
    rows = [{"task_id": f"a::{index}", ENVIRONMENT_COLUMN: "a"} for index in range(count)]
    return TaskPool.from_rows(rows, entries=["a"], column=ENVIRONMENT_COLUMN)


def _single(sampler_class: type, pool: TaskPool) -> SamplerSet:
    infos = {"a": EntryInfo(group_size=1, name="a", weight=1.0, index=0, task_ids=pool.task_ids)}
    sampling = TaskSampling(
        pool=pool, run=SamplerChoice("custom", sampler_class), entries={}, infos=infos, epoch_size=6
    )
    return SamplerSet.build(sampling, batch_size=2, label_column=ENVIRONMENT_COLUMN)


def _fallback_index(name: str, decision: int, count: int) -> int:
    return decision_key(0, name, decision) % count


def test_a_draw_or_a_bare_task_id_binds_that_task_with_its_reason_and_score():
    class Good(_Answers):
        answers = [Draw("a::2", reason="explore", score=0.25), "a::4", Draw("a::1", score=3)]

    samplers = _single(Good, _pool())
    first, second, third = (samplers.static(position) for position in range(3))
    assert (first.row["task_id"], first.index, first.sampler, first.reason, first.score) == (
        "a::2",
        2,
        "custom",
        "explore",
        0.25,
    )
    assert (second.row["task_id"], second.reason, second.score) == ("a::4", "order", None)
    assert third.score == 3.0 and isinstance(third.score, float)
    assert samplers.run.decisions == 3 and samplers.fallbacks() == {"custom": 0}


@pytest.mark.parametrize(
    "answer",
    [
        None,
        "elsewhere::9",
        Draw("a::1", reason=""),
        Draw("a::1", reason="x" * 65),
        Draw("a::1", score=float("nan")),
        Draw("a::1", score=float("inf")),
        Draw("a::1", score=True),
        Slot("a"),  # a class without entry_slots
        _Row(1),  # core's answer, from a class that is not core's
        42,
        RuntimeError("a bug"),
    ],
    ids=[
        "none",
        "outside-pool",
        "empty-reason",
        "long-reason",
        "nan-score",
        "inf-score",
        "bool-score",
        "slot",
        "core-row",
        "int",
        "raises",
    ],
)
def test_a_refused_answer_gets_the_fallback_draw_which_uses_the_decision_up(answer, caplog):
    """The task at offset key mod size of its pool, reason ``fallback``, logged once and counted; the next decision
    is the one after it, and a resumed run draws the same fallback."""

    class Bad(_Answers):
        answers = [answer, answer, "a::5"]

    samplers = _single(Bad, _pool())
    with caplog.at_level(logging.ERROR):
        first, second, third = (samplers.static(position) for position in range(3))

    assert first.reason == second.reason == "fallback" and first.score is None
    assert first.index == _fallback_index("custom", 0, 6) and second.index == _fallback_index("custom", 1, 6)
    assert first.row["task_id"] == f"a::{first.index}" and first.sampler == "custom"
    assert third.row["task_id"] == "a::5" and third.reason == "order"
    assert samplers.fallbacks() == {"custom": 2} and samplers.run.decisions == 3
    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 1  # once per sampler and kind
    assert ("raised at decision 0" in errors[0].getMessage()) == isinstance(answer, Exception)
    resumed = _single(Bad, _pool())
    assert resumed.static(1).index == second.index  # a fallback keeps no state


def test_a_slot_handing_sampler_that_fails_falls_back_to_a_slot_by_weight_which_the_entry_s_sampler_fills():
    """The fallback's slot goes to an entry with probability proportional to its weight; the entry's sampler fills it,
    and the draw names the run's sampler, whose decision it replaced, with reason ``fallback``."""

    class Slots(_Answers):
        entry_slots = True
        answers = [RuntimeError("no slot")] * 400

    rows = [{"task_id": f"{label}::{index}", ENVIRONMENT_COLUMN: label} for label in "ab" for index in range(3)]
    pool = TaskPool.from_rows(rows, entries=["a", "b"], column=ENVIRONMENT_COLUMN)
    infos = {
        label: EntryInfo(group_size=1, name=label, weight=weight, index=index, task_ids=pool.task_ids[3 * index :][:3])
        for index, (label, weight) in enumerate((("a", 3.0), ("b", 1.0)))
    }
    uniform = SamplerChoice("uniform", UniformSampler)
    sampling = TaskSampling(
        pool=pool, run=SamplerChoice("slots", Slots), entries={"a": uniform, "b": uniform}, infos=infos, epoch_size=6
    )
    samplers = SamplerSet.build(sampling, batch_size=2, label_column=ENVIRONMENT_COLUMN)
    drawn = [samplers.static(position) for position in range(400)]

    by_entry = [row.row[ENVIRONMENT_COLUMN] for row in drawn]
    assert 0.65 < by_entry.count("a") / 400 < 0.85  # three to one, in expectation
    assert {(row.sampler, row.reason) for row in drawn} == {("slots", "fallback")}
    assert samplers.entries["a"].decisions == by_entry.count("a")  # the entry's sampler filled each of its slots
    assert samplers.fallbacks()["slots"] == 400


def test_an_entry_s_sampler_that_hands_a_slot_gets_the_fallback_task_of_its_entry(caplog):
    """Only a run's sampler hands slots: an entry's that answers one is refused, and its decision falls back to a task
    of its own entry's pool, named after it."""

    class Hands(_Answers):
        answers = [Slot("b")] * 20

    sampling = _mixture()
    sampling = TaskSampling(
        pool=sampling.pool,
        run=sampling.run,
        entries={**sampling.entries, "a": SamplerChoice("hands", Hands)},
        infos=sampling.infos,
        epoch_size=sampling.epoch_size,
    )
    samplers = SamplerSet.build(sampling, batch_size=2, label_column=ENVIRONMENT_COLUMN)
    with caplog.at_level(logging.ERROR):
        drawn = [samplers.static(position) for position in range(9)]

    from_a = [draw for draw in drawn if draw.row[ENVIRONMENT_COLUMN] == "a"]
    assert from_a and {(draw.sampler, draw.reason) for draw in from_a} == {("hands@a", "fallback")}
    assert all(draw.index in samplers.pool.entry_indices("a") for draw in from_a)
    assert samplers.fallbacks()["hands@a"] == len(from_a) == samplers.entries["a"].decisions
    assert sum("hands@a" in record.getMessage() for record in caplog.records) == 1


def test_a_decision_without_a_task_to_fall_back_to_is_an_error():
    class Raises(_Answers):
        answers = [RuntimeError("a bug")]

    samplers = _single(Raises, TaskPool((), (), ()))  # a sampler of its own over no task
    with pytest.raises(SamplerError, match="no task to fall back to"):
        samplers.static(0)


def test_core_s_static_samplers_fail_as_platoon_0_3_0_s_orders_did_with_no_fallback(monkeypatch, caplog):
    """An order handed in that raises (an epoch it cannot give), or that gives what is no row, stops the run with its
    own error, as platoon 0.3.0's dispatcher did when it read the order, rather than training rows of the fallback's in
    its place; so does a built-in that raises, which a valid setup never makes it do, also where a resume recomputes
    the counts.  Only a sampler of one's own gets the fallback draw."""

    def order(position: int) -> dict[str, Any]:
        if position >= 6:
            raise ValueError(f"the order has no epoch {position // 6}")
        return {"task_id": f"a::{position}"}

    samplers = SamplerSet.of_order(order, pool=_pool())
    assert [samplers.static(position).row["task_id"] for position in range(6)] == [f"a::{i}" for i in range(6)]
    with caplog.at_level(logging.ERROR), pytest.raises(ValueError, match="no epoch 1"):
        samplers.static(6)
    assert samplers.fallbacks() == {"auto": 0} and not caplog.records
    rows = SamplerSet.of_order(lambda position: ["no", "row"], pool=_pool())
    with pytest.raises(SamplerError, match="decision 0 answered .*'no', 'row'"):
        rows.static(0)

    def broken(self: Any, ctx: SamplerContext) -> Any:
        raise RuntimeError("a bug of the built-in")

    monkeypatch.setattr(WeightedSampler, "next", broken)
    weighted = SamplerSet.build(_mixture(), batch_size=2, label_column=ENVIRONMENT_COLUMN)
    with pytest.raises(RuntimeError, match="a bug of the built-in"):
        weighted.static(0)
    resumed = SamplerSet.build(_mixture(), batch_size=2, label_column=ENVIRONMENT_COLUMN)
    with pytest.raises(RuntimeError, match="a bug of the built-in"):
        resumed.restore(None, next_position=3)  # the counts recomputed from the position ask it again


def test_an_order_handed_in_draws_its_rows_as_they_are_from_any_position():
    rows = [{"task_id": f"t{index}", "n": index} for index in range(5)]
    samplers = SamplerSet.of_order(lambda position: rows[position % 5])

    drawn = samplers.static(7)
    assert drawn.row == rows[2] and drawn.row is not rows[2] and drawn.index is None and drawn.sampler == "auto"
    assert samplers.static(8).row == rows[3] and samplers.run.decisions == 9


# --- counts and states ------------------------------------------------------------------------------------------------


def _mixture() -> TaskSampling:
    """Entries a and b of four tasks each, weighted two to one."""

    weights = {"a": 2.0, "b": 1.0}
    rows = [{"task_id": f"{label}::{index}", ENVIRONMENT_COLUMN: label} for label in weights for index in range(4)]
    pool = TaskPool.from_rows(rows, entries=list(weights), column=ENVIRONMENT_COLUMN)
    infos = {
        label: EntryInfo(
            group_size=1, name=label, weight=weight, seed=index, index=index, task_ids=pool.task_ids[4 * index :][:4]
        )
        for index, (label, weight) in enumerate(weights.items())
    }
    uniform = SamplerChoice("uniform", UniformSampler)
    return TaskSampling(
        pool=pool,
        run=SamplerChoice("weighted", WeightedSampler),
        entries=dict.fromkeys(weights, uniform),
        infos=infos,
        epoch_size=8,
    )


def test_the_samplers_state_holds_each_one_s_name_class_count_and_state_as_json():
    samplers = SamplerSet.build(_mixture(), batch_size=2, label_column=ENVIRONMENT_COLUMN)
    for position in range(7):
        samplers.static(position)

    state = json.loads(json.dumps(samplers.state(), allow_nan=False))
    assert state == {
        "run": {"name": "weighted", "class": "platoon.train.samplers:WeightedSampler", "decisions": 7, "state": {}},
        "entries": {
            "a": {"name": "uniform@a", "class": "platoon.train.samplers:UniformSampler", "decisions": 5, "state": {}},
            "b": {"name": "uniform@b", "class": "platoon.train.samplers:UniformSampler", "decisions": 2, "state": {}},
        },
    }


@pytest.mark.parametrize("saved", ["state", "none", "renamed", "bad-state", "unwritable"])
def test_a_resume_takes_up_the_saved_counts_or_recomputes_them_from_the_position(saved, caplog):
    """Saved under the same names and classes, the entries' counts are taken as saved; without a state (format 1), or
    one that names other samplers or does not load, they are recomputed from the position: either way the draws go on
    as an uninterrupted run's."""

    sampling = _mixture()
    uninterrupted = SamplerSet.build(sampling, batch_size=2, label_column=ENVIRONMENT_COLUMN)
    expected = [uninterrupted.static(position).row["task_id"] for position in range(30)]
    first = SamplerSet.build(sampling, batch_size=2, label_column=ENVIRONMENT_COLUMN)
    for position in range(11):
        first.static(position)
    state: Any = first.state()
    if saved == "none":
        state = None
    elif saved == "renamed":
        state["entries"]["a"]["name"] = "pass_rate@a"
    elif saved == "bad-state":
        state["entries"]["a"]["state"] = {"unexpected": 1}
    elif saved == "unwritable":
        del state["entries"]["b"]["state"]
    else:
        state["entries"]["a"]["decisions"] = 7  # what it saved is what it takes up, not what a replay gives

    resumed = SamplerSet.build(sampling, batch_size=2, label_column=ENVIRONMENT_COLUMN)
    with caplog.at_level(logging.WARNING):
        resumed.restore(state, next_position=11)

    counts = {label: member.decisions for label, member in resumed.entries.items()}
    assert resumed.run.decisions == 11
    if saved == "state":
        assert counts == {"a": 7, "b": first.entries["b"].decisions}
    else:
        assert counts == {label: member.decisions for label, member in first.entries.items()}
        assert [resumed.static(position).row["task_id"] for position in range(11, 30)] == expected[11:]
    assert any("does not load" in record.getMessage() for record in caplog.records) == (saved == "bad-state")


def test_a_sampler_s_state_is_saved_as_it_stood_when_the_state_was_taken():
    """The state a full checkpoint takes is written later, in a thread, while the dispatcher goes on deciding: each
    sampler's state is copied whole when it is taken, though its state_dict() returns the containers it keeps
    changing."""

    class Keeps(_Answers):
        answers = ["a::1", "a::2", "a::3", "a::4"]

        def __init__(self, setup, options=None) -> None:
            super().__init__(setup, options)
            self.made: dict[str, list[int]] = {"decisions": []}

        def next(self, ctx: SamplerContext) -> Any:
            self.made["decisions"].append(ctx.decision)
            return super().next(ctx)

        def state_dict(self) -> dict[str, Any]:
            return {"made": self.made}  # its own containers, as nothing in the contract forbids

    samplers = _single(Keeps, _pool())
    samplers.static(0)
    state = samplers.state()
    samplers.static(1)
    samplers.static(2)
    assert state["run"] == {
        "name": "custom",
        "class": f"{Keeps.__module__}:{Keeps.__qualname__}",
        "decisions": 1,
        "state": {"made": {"decisions": [0]}},
    }


def test_a_sampler_s_metrics_leave_the_dispatcher_s_own_counts_as_they_are(caplog):
    """A sampler whose metrics() names a key the dispatcher writes itself under its prefix (its draws by reason, its
    decision time) has that key left out, with an error logged once: the fallbacks it counts are the ones logged."""

    class Masks(_Answers):
        answers = [RuntimeError("a bug")] * 4

        def metrics(self, stats: Any) -> dict[str, float]:
            return {"draws/fallback": 0.0, "draws": 0.0, "decision_ms": 0.0, "seen": 2.0}

    samplers = _single(Masks, _pool())
    for position in range(4):
        decided = samplers.static(position)
        samplers.count(decided.sampler, decided.reason)  # as the dispatcher counts each draw it records
    with caplog.at_level(logging.ERROR):
        metrics = samplers.step_metrics(0)
        samplers.step_metrics(0)
    assert metrics["sampler/draws/fallback"] == metrics["sampler/draws"] == 4.0
    assert metrics["sampler/seen"] == 2.0 and metrics["sampler/decision_ms"] >= 0.0
    reserved = [record for record in caplog.records if "the dispatcher writes itself" in record.getMessage()]
    assert len(reserved) == 1


def test_a_state_that_is_not_json_is_left_out_with_a_warning(caplog):
    class Unsaved(_Answers):
        answers = ["a::1"] * 3

        def state_dict(self) -> dict[str, Any]:
            return {"when": float("nan")}

    samplers = _single(Unsaved, _pool())
    samplers.static(0)
    with caplog.at_level(logging.ERROR):
        state = samplers.state()
    assert "state" not in state["run"] and state["run"]["decisions"] == 1
    assert any("Could not save the state of task sampler custom" in record.getMessage() for record in caplog.records)


# --- adaptive decisions ---------------------------------------------------------------------------------------------


class _Adaptive(Sampler):
    """An adaptive sampler that answers ``answers[decision]`` (raising an exception it holds), and records the groups
    it observes; with ``answers`` empty, the first eligible task."""

    adaptive = True
    answers: list[Any] = []

    def __init__(self, setup, options=None) -> None:
        super().__init__(setup, options)
        self.seen: list[tuple[int, int]] = []

    def next(self, ctx: SamplerContext) -> Any:
        answers = type(self).answers
        if not answers:
            return next((task for task in ctx.task_ids if ctx.eligible(task)), None)
        answer = answers[ctx.decision % len(answers)]
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def observe(self, group: Any) -> None:
        self.seen.append((group.start, group.seq))


def _adaptive_set(cls: type = _Adaptive, *, b: type = UniformSampler) -> SamplerSet:
    """Entries a (tasks a::0..a::3, label a) and b (b::0, b::1), weighted one to one, a's sampler ``cls``."""

    rows = [{"task_id": f"a::{index}", ENVIRONMENT_COLUMN: "a"} for index in range(4)]
    rows += [{"task_id": f"b::{index}", ENVIRONMENT_COLUMN: "b"} for index in range(2)]
    pool = TaskPool.from_rows(rows, entries=["a", "b"], column=ENVIRONMENT_COLUMN)
    infos = {
        "a": EntryInfo(group_size=2, name="a", weight=1.0, seed=3, index=0, task_ids=pool.task_ids[:4]),
        "b": EntryInfo(group_size=2, name="b", weight=1.0, seed=4, index=1, task_ids=pool.task_ids[4:]),
    }
    sampling = TaskSampling(
        pool=pool,
        run=SamplerChoice("weighted", WeightedSampler),
        entries={"a": SamplerChoice("custom", cls), "b": SamplerChoice("uniform", b)},
        infos=infos,
        epoch_size=6,
    )
    return SamplerSet.build(sampling, batch_size=2, label_column=ENVIRONMENT_COLUMN)


@pytest.mark.parametrize(
    "answer",
    [
        None,  # while a task is eligible
        RuntimeError("a bug"),
        "elsewhere::0",
        "b::0",  # another entry's task
        Slot("b"),
        Draw("a::1", "", None),
        Draw("a::1", "ok", float("nan")),
        _Row(0),  # core's row answer, from a class that is not core's
        object(),
    ],
)
def test_an_adaptive_decision_that_fails_gets_an_eligible_fallback_task_and_uses_the_decision_up(answer, caplog):
    _Adaptive.answers = [answer]
    try:
        samplers = _adaptive_set()
        member = samplers.entries["a"]
        room = lambda label: True  # noqa: E731
        with caplog.at_level(logging.ERROR):
            first = samplers.decide("a", room=room)
            second = samplers.decide("a", room=room)
        assert first is not None and second is not None
        assert {first.reason, second.reason} == {"fallback"} and first.sampler == "custom@a"
        assert first.row["task_id"].startswith("a::") and second.row["task_id"].startswith("a::")
        ctx = samplers.context(member, room=room)
        assert member.decisions == ctx.decision == 2 and member.fallbacks == 2
        assert len([record for record in caplog.records if record.levelno == logging.ERROR]) == 1  # logged once
    finally:
        _Adaptive.answers = []


def test_an_adaptive_decision_chooses_among_eligible_tasks_and_decides_nothing_when_none_is():
    _Adaptive.answers = ["a::0"]  # an ineligible task when a's label is full: refused, never drawn
    try:
        samplers = _adaptive_set()
        member = samplers.entries["a"]
        assert samplers.decide("a", room=lambda label: label != "a") is None  # nothing eligible: nothing decided
        assert member.decisions == 0 and member.fallbacks == 0
        decided = samplers.decide("a", room=lambda label: True)
        assert decided is not None and decided.row["task_id"] == "a::0" and member.decisions == 1
    finally:
        _Adaptive.answers = []


def test_the_keyed_fallback_is_the_same_after_a_resume():
    _Adaptive.answers = [RuntimeError("a bug")]
    try:
        first = _adaptive_set()
        made = [first.decide("a", room=lambda label: True).row["task_id"] for _ in range(6)]
        again = _adaptive_set()
        again.restore(json.loads(json.dumps(first.state())), next_position=0)
        assert again.entries["a"].decisions == 6
        resumed = _adaptive_set()
        for _ in range(3):
            resumed.decide("a", room=lambda label: True)
        state = json.loads(json.dumps(resumed.state()))
        later = _adaptive_set()
        later.restore(state, next_position=0)
        assert [later.decide("a", room=lambda label: True).row["task_id"] for _ in range(3)] == made[3:]
    finally:
        _Adaptive.answers = []


def test_settled_groups_reach_the_adaptive_samplers_of_their_pool_alone_and_static_ones_none():
    from platoon.train.task_stats import GroupRecord

    samplers = _adaptive_set(b=_Adaptive)

    class _Statistics:
        def __init__(self) -> None:
            self.listeners: list[Any] = []

        def listen(self, listener) -> None:
            self.listeners.append(listener)

    statistics = _Statistics()
    samplers.attach(statistics)
    assert len(statistics.listeners) == 1

    def record(seq: int, task: str) -> GroupRecord:
        return GroupRecord(
            start=1, seq=seq, task_id=task, entry=task.split("::")[0], labels={}, harness=None, group_size=2,
            sampler="x", reason="order", score=None, position=seq, drawn_at={"main": 0}, members=(), verdict="unknown",
        )  # fmt: skip

    for seq, task in enumerate(["a::0", "b::1", "a::2"]):
        statistics.listeners[0](record(seq, task))
    assert samplers.entries["a"].sampler.seen == [(1, 0), (1, 2)]
    assert samplers.entries["b"].sampler.seen == [(1, 1)]
    static = _adaptive_set()
    static_statistics = _Statistics()
    static.attach(static_statistics)  # a's sampler observes; b's uniform is static and gets none
    assert len(static_statistics.listeners) == 1


def test_without_statistics_an_adaptive_sampler_reads_an_empty_store_that_is_not_live():
    samplers = _adaptive_set()
    view = samplers.view(3)
    assert view.live is False and view.now == 3 and view.counts("a::0").groups == 0
    assert set(view.task_ids()) == {"a::0", "a::1", "a::2", "a::3", "b::0", "b::1"}
