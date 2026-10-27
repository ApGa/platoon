"""The run's per-task statistics take in settled groups in the order they were drawn (RFC 0019, section 3.3): a
group's record is made when it settles and held, and the store takes the held records in, sorted by ``(start, seq)``,
at the next flush, which every reader of the store runs first (a snapshot, a step's metrics, an adaptive sampler's
view, the ``store`` property).  So groups that settle out of draw order leave the statistics, their snapshot and the
step's metrics byte for byte as groups that settle in draw order do, although their decayed counts are float sums
whose order shows in the last bits; a fate does not wait for the flush; listeners (the adaptive samplers' ``observe``)
get each record in the same order; and a recorder that stops drops what it holds."""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fakes.config import make_config

from platoon.train.components import ENVIRONMENT_COLUMN
from platoon.train.dispatch.dispatcher import POSITION_COLUMN, SEQ_COLUMN
from platoon.train.sampler_runtime import SamplerSet, TaskPool
from platoon.train.task_records import TaskRecorder
from platoon.train.task_stats import MAIN, MemberResult, TaskStatsStore, snapshot_bytes

TASKS = [f"{entry}::{index}" for entry in ("quick", "slow") for index in range(3)]


def _config() -> Any:
    """A mixture of two entries, whose groups of 4 pass at 1.0 and at 0.5."""

    return make_config(
        environments=[{"name": "quick", "success_threshold": 1.0}, {"name": "slow", "success_threshold": 0.5}],
        train={"workflow_config": {"group_size": 4}},
    )


def _rows() -> list[dict[str, Any]]:
    return [{"task_id": task, ENVIRONMENT_COLUMN: task.split("::")[0]} for task in TASKS]


def _output(rng: random.Random, version: int) -> Any:
    """A group of four members at policy ``version``, whose rewards are fractions, some members without a result."""

    members = []
    for _ in range(4):
        valid = rng.random() > 0.2
        reward = rng.choice((0.0, 0.25, 0.5, 1.0)) if valid else None
        members.append(
            MemberResult(
                reward=reward,
                scored=reward,
                valid=valid,
                outcome="scored" if valid else "error",
                disposition="completed",
                reward_key=None,
                steps=rng.randrange(1, 9),
                output_tokens=rng.randrange(10, 99),
                seconds=rng.uniform(0.5, 9.5),
                versions={MAIN: (version, version)},
            )
        )
    passes = sum(member.passes(1.0) for member in members)
    verdict = "trained" if 0 < passes < sum(member.valid for member in members) else "uniform"
    return SimpleNamespace(verdict=verdict, members=tuple(members))


DRAWS = 18


def _recorded(tmp_path: Path, order: list[int], *, flush_every: int | None = None) -> TaskRecorder:
    """A recorder that drew ``DRAWS`` groups of the tasks in turn, at policy versions 0 to 5, and saw them settle in
    ``order`` (their draw numbers); with ``flush_every``, flushed after that many settles."""

    run_dir = tmp_path / "-".join(map(str, order))[:40]
    run_dir.mkdir(parents=True, exist_ok=True)
    recorder = TaskRecorder(run_dir=str(run_dir), config=_config(), tasks=_rows(), start=1)
    rows = []
    for seq in range(DRAWS):
        row = {"task_id": TASKS[seq % len(TASKS)], SEQ_COLUMN: seq, POSITION_COLUMN: seq}
        recorder.draw(row, seq=seq, position=seq, version=seq // 3)
        rows.append(row)
    outputs = {seq: _output(random.Random(seq), seq // 3) for seq in range(DRAWS)}
    for count, seq in enumerate(order, start=1):
        entry = rows[seq]["task_id"].split("::")[0]
        recorder.settle(rows[seq], entry=entry, label=entry, output=outputs[seq])
        if flush_every and count % flush_every == 0:
            recorder.flush()
    return recorder


def _state(recorder: TaskRecorder) -> tuple[bytes, dict[str, float]]:
    """The statistics' snapshot as bytes, and a step's metrics, each read as the run reads them (flushing first)."""

    metrics = recorder.step_metrics(now=6, prefixes={"quick": "environment/quick", "slow": "environment/slow"})
    return snapshot_bytes(recorder.store.snapshot(start=1, next_seq=DRAWS)), metrics


def _completion_orders(count: int) -> list[list[int]]:
    rng = random.Random(3)
    return [rng.sample(range(DRAWS), DRAWS) for _ in range(count)]


def test_groups_settled_out_of_draw_order_leave_the_statistics_as_in_draw_order_byte_for_byte(tmp_path):
    drawn = _state(_recorded(tmp_path, list(range(DRAWS))))
    for order in _completion_orders(12):
        assert _state(_recorded(tmp_path, order)) == drawn


def test_the_test_would_see_a_completion_order_in_the_last_bits(tmp_path):
    """Applied at settle, as platoon 0.3.0 did, some completion order leaves other bits in the decayed sums: the
    check above can tell."""

    def settled_at_once(order: list[int]) -> bytes:
        recorder = _recorded(tmp_path, list(range(DRAWS)))  # the records, as the recorder makes them
        records = sorted(recorder._held, key=lambda record: record.seq)
        store = TaskStatsStore(
            [(task, task.split("::")[0], task.split("::")[0]) for task in TASKS],
            recorder.entries,
            column=ENVIRONMENT_COLUMN,
            group_size=4,
        )
        for record in records:
            store.draw(record.start, record.seq, record.task_id, record.position)
        for seq in order:
            store.settle(records[seq])
        return snapshot_bytes(store.snapshot(start=1, next_seq=DRAWS))

    in_order = settled_at_once(list(range(DRAWS)))
    assert any(settled_at_once(order) != in_order for order in _completion_orders(400))


def test_flushes_between_settles_take_each_flush_s_records_in_draw_order(tmp_path):
    """A flush takes in what settled since the last one, sorted: two runs whose groups settle in other orders within
    each flush's window leave the same statistics."""

    first = list(itertools.chain.from_iterable(sorted(range(start, start + 6)) for start in range(0, DRAWS, 6)))
    second = list(
        itertools.chain.from_iterable(sorted(range(start, start + 6), reverse=True) for start in range(0, DRAWS, 6))
    )
    assert _state(_recorded(tmp_path, first, flush_every=6)) == _state(_recorded(tmp_path, second, flush_every=6))


def test_nothing_is_taken_in_before_a_flush_and_every_reader_flushes(tmp_path):
    recorder = _recorded(tmp_path, [2, 0, 1])
    assert sum(recorder._store.task(task).groups for task in TASKS) == 0  # held, not taken in
    assert recorder.store.task(TASKS[0]).groups == 1  # the store property flushes
    for read in (
        lambda r: r.step_metrics(now=1, prefixes={}),
        lambda r: r.snapshot("000001"),
        lambda r: _view(r),
    ):
        again = _recorded(tmp_path / "again", [5, 4, 3])
        assert sum(again._store.task(task).groups for task in TASKS) == 0
        read(again)
        assert sum(again._store.task(task).groups for task in TASKS) == 3


def _view(recorder: TaskRecorder) -> Any:
    """An adaptive decision's view of the statistics, which takes in every settled group first."""

    pool = TaskPool.from_rows(_rows(), entries=["quick", "slow"], column=ENVIRONMENT_COLUMN)
    samplers = SamplerSet.of_order(lambda position: _rows()[position % 6], pool=pool)
    samplers.attach(recorder)
    return samplers.view(0)


def test_a_fate_does_not_wait_for_the_flush(tmp_path):
    """A committed step gives its groups their fates whether the store took them in yet or not: the same statistics."""

    early = _recorded(tmp_path / "early", list(range(6)))
    late = _recorded(tmp_path / "late", list(range(6)))
    for record in sorted(early._held, key=lambda record: record.seq):
        early.fate(record, "consumed")  # before the store took the group in
    records = sorted(late._held, key=lambda record: record.seq)
    late.flush()
    for record in records:
        late.fate(record, "consumed")
    assert _state(early) == _state(late)
    task = early.store.task(TASKS[0])
    assert (task.groups, task.fates, task.outstanding) == (1, {"consumed": 1}, 2)


def test_listeners_get_each_record_in_draw_order_and_one_that_raises_does_not_stop_the_statistics(tmp_path, caplog):
    recorder = _recorded(tmp_path, [4, 1, 3, 0, 2])
    seen: list[int] = []
    recorder.listen(lambda record: seen.append(record.seq))

    def broken(record: Any) -> None:
        raise RuntimeError("a listener's own bug")

    recorder.listen(broken)
    recorder.flush()
    assert seen == [0, 1, 2, 3, 4] and not recorder.stopped
    recorder.flush()  # nothing held: nothing handed again
    assert seen == [0, 1, 2, 3, 4]
    assert "Could not hand a settled group" in caplog.text


def test_a_recorder_that_stops_drops_what_it_holds(tmp_path, monkeypatch):
    recorder = _recorded(tmp_path, [1, 0])

    def settle(self, record):
        raise RuntimeError("a bug in the statistics")

    monkeypatch.setattr(TaskStatsStore, "settle", settle)
    recorder.flush()
    assert recorder.stopped and recorder._held == []
    row = {"task_id": TASKS[2], SEQ_COLUMN: 2, POSITION_COLUMN: 2}
    assert recorder.settle(row, entry="quick", label="quick", output=_output(random.Random(2), 0)) is None
    assert recorder._held == []


@pytest.mark.parametrize("flush", [False, True])
def test_a_snapshot_read_back_holds_every_group_that_settled_before_it(tmp_path, flush):
    recorder = _recorded(tmp_path, [3, 1, 2, 0])
    if flush:
        recorder.flush()
    record = recorder.snapshot("000002")
    assert record is not None
    data = json.loads((Path(recorder.run_dir) / record["file"]).read_text())
    assert sum(task["groups"] for task in data["tasks"].values()) == 4
