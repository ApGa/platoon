"""The dispatcher in the trainer's own loop (Tinker faked): RFC 0019's exit checks for its milestone M3, and the groups
a full checkpoint keeps (its question 4).

- At ``max_staleness: 0`` every step trains exactly the rows of its batch of the order, as the producer drew them.
- At ``max_staleness: k >= 1`` the groups started and not yet consumed by a committed step never exceed
  ``(k + 1) × batch_size``.
- A capped slow label leaves the other label's step time within 10% of a run without it.
- A run interrupted twice consumes every position of the order exactly once, and rolls out each once.
- No draw is made while every label of the run is at its limit.
- A resume from a full checkpoint trains each group it kept once, without rolling it out again, a failed one as
  failed; from one that kept none (its file lost, or never written), it rolls them out again; pruning deletes them
  with their checkpoint.
- A resume is exact also when the run's per-task statistics stopped, or could not be written, before its checkpoint;
  a group that a step dropped as stale before the checkpoint was consumed, and the resume does not draw it again.
- A resume whose sampling inputs changed warns, records it, and goes on; one from a checkpoint that saved no state of
  its draws (one saved before platoon 0.3.0, or whose write of it failed) draws from its batch, as before.
- A group that fails, or whose workflow is cancelled from within, settles as failed and frees its slot; a step that an
  interrupt drops leaves the groups it dequeued as they settled, for the resume to train.

Each group returns one datum whose first token is its row's position in the order, so the datums a step trains say
which positions it consumed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import signal
import statistics
from pathlib import Path
from typing import Any

import pytest
import tinker
from datasets import Dataset
from fakes.config import mixture_config, trainer_config
from fakes.runs import (
    RECOVERY_RUN,
    FailingGroups,
    PositionGroups,
    RecordingClient,
    crash_after,
    dispatch_state,
    run_start,
    service_start,
    weighted_mixture,
)
from fakes.tinker_service import FakeTinkerService, make_service_trainer
from fakes.trainer import draws_of, interrupt_at, make_trainer

from platoon.train import dataloader as dataloader_module
from platoon.train import task_records
from platoon.train.checkpoints import kept_groups
from platoon.train.checkpoints.pruning import DELETED_CHECKPOINTS_NAME, read_checkpoint_records
from platoon.train.components import ENVIRONMENT_COLUMN, task_ids_to_dataset
from platoon.train.dispatch.dispatcher import POSITION_COLUMN, Dispatcher
from platoon.train.task_stats import TaskStatsStore, read_events

# --- k = 0: each step trains exactly the rows of its batch -----------------------------------------------------------


@pytest.mark.parametrize("mixture", [False, True], ids=["one-environment", "mixture"])
def test_at_k_0_every_step_trains_exactly_the_rows_of_its_batch(tmp_path, mixture):
    """Groups that take random times, two of a step's four at once: each step still trains the four positions of its
    batch, the rows the dataloader hands out for it, as the producer drew them at max_staleness 0."""

    overrides = ("train.batch_size=4", "train.max_training_steps=6", "train.max_staleness=0")
    if mixture:
        config = mixture_config(
            tmp_path,
            {"batch_size": 4, "max_training_steps": 6, "max_staleness": 0, "max_concurrent_rollouts": 16},
            [{"name": "a"}, {"name": "b"}],
        )
    else:
        config = trainer_config(tmp_path, *overrides)  # two groups of 8 at once
    client = RecordingClient()
    mixture_tasks = {}
    if mixture:
        mixture_tasks["train_dataset"], mixture_tasks["train_order"] = weighted_mixture(a=7, b=5)
    trainer, _saves, logged = make_trainer(config, tasks=10, training_client=client, **mixture_tasks)
    pace = random.Random(7)
    workflow = PositionGroups(trainer.model_info.llm, delay=lambda row: pace.uniform(0.0, 0.03))

    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=workflow), 60))

    assert logged == [1, 2, 3, 4, 5, 6]
    assert [sorted(step) for step in client.steps] == [list(range(4 * step, 4 * step + 4)) for step in range(6)]
    assert sorted(workflow.started) == list(range(24))  # every group a step consumed, and no other
    dataloader = dataloader_module.PlatoonTinkerDataloader(trainer.train_dataset, 4, order=trainer.train_order)
    batches = [[row["task_id"] for row in dataloader.get_batch(step)] for step in range(6)]
    assert [[workflow.tasks[position] for position in range(4 * step, 4 * step + 4)] for step in range(6)] == batches


# --- k >= 1: the pipeline never exceeds (k + 1) × batch_size ---------------------------------------------------------


@pytest.mark.parametrize("k", [1, 3])
def test_at_k_the_groups_started_and_not_consumed_never_exceed_k_plus_one_batches(tmp_path, k):
    """Rollouts outpace training (a step's training takes 20 ms, a group 0 to 20 ms) with rollout permits for every
    group of the run at once: the horizon alone bounds the groups started ahead, at (k + 1) batches, and reaches it.
    Counted from outside the dispatcher: the groups started, less those the steps logged so far consumed."""

    config = trainer_config(
        tmp_path,
        "train.batch_size=3",
        "train.max_training_steps=10",
        f"train.max_staleness={k}",
        "train.max_concurrent_rollouts=1024",
    )
    client = RecordingClient(delay=0.02)
    trainer, _saves, logged = make_trainer(config, tasks=7, training_client=client)
    pace = random.Random(k)
    ahead: list[int] = []
    workflow = PositionGroups(
        trainer.model_info.llm,
        delay=lambda row: pace.uniform(0.0, 0.02),
        on_start=lambda row, count: ahead.append(count - 3 * len(logged)),
    )

    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=workflow), 60))

    assert logged == list(range(1, 11))
    assert max(ahead) <= (k + 1) * 3
    assert draws_of(trainer).dispatcher.peak == (k + 1) * 3  # it is reached: the bound is the horizon, not the permits
    assert sorted(position for step in client.steps for position in step) == list(range(30))


# --- a capped slow label leaves the other label's step time as it was ------------------------------------------------


def _step_times(tmp_path: Path, *, slow: bool) -> list[float]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    environments = [{"name": "quick"}]
    train: dict[str, Any] = {"batch_size": 8, "max_training_steps": 30, "max_staleness": 3}
    if slow:
        environments.append({"name": "slow", "max_staleness": 48})
        train.update(
            rollout_group_limits={"slow": 1}, max_concurrent_rollouts=(8 + 1) * 8
        )  # 8 quick groups and the slow
    else:
        train.update(max_concurrent_rollouts=8 * 8)  # 8 quick groups at once
    config = mixture_config(tmp_path, train, environments)
    rows = [{"task_id": f"quick::q{index}", ENVIRONMENT_COLUMN: "quick"} for index in range(87)]
    if slow:
        rows += [{"task_id": f"slow::s{index}", ENVIRONMENT_COLUMN: "slow"} for index in range(13)]
    trainer, _saves, logged = make_trainer(config, train_dataset=Dataset.from_list(rows))
    workflow = PositionGroups(
        trainer.model_info.llm, delay=lambda row: 1.0 if row[ENVIRONMENT_COLUMN] == "slow" else 0.1
    )

    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=workflow), 120))
    times = [stats["timing/batch_total"] for stats in logged.stats]

    assert len(times) == 30
    return times


def test_a_capped_slow_label_leaves_the_other_label_s_step_time_within_ten_percent(tmp_path):
    """A slow label at 13% of the order, capped at one group at a time, whose groups take ten times a quick one's: the
    median step time stays within 10% of a run of the quick label alone, with the same rollouts at once for it.  (The
    run's last step waits for the slow group still running, which it consumes: the run starts no group past its
    plan.)"""

    alone = statistics.median(_step_times(tmp_path / "alone", slow=False)[2:])
    mixed = statistics.median(_step_times(tmp_path / "mixed", slow=True)[2:])

    assert abs(mixed - alone) <= 0.10 * alone, (mixed, alone)


# --- no draw while every label is at its limit --------------------------------------------------------------------


def test_no_draw_while_every_label_of_the_run_is_at_its_limit(tmp_path, monkeypatch):
    config = mixture_config(
        tmp_path,
        {"batch_size": 2, "max_training_steps": 6, "rollout_group_limits": {"a": 1, "b": 1}},
        [{"name": "a"}, {"name": "b"}],
    )
    dataset, order = weighted_mixture(a=4, b=4)
    trainer, _saves, logged = make_trainer(config, train_dataset=dataset, train_order=order)
    draws: list[bool] = []
    new_draw = Dispatcher._new_draw

    def watched(self):
        draws.append(any(self._active[label] < limit for label, limit in self._limits.items()))
        return new_draw(self)

    monkeypatch.setattr(Dispatcher, "_new_draw", watched)
    workflow = PositionGroups(trainer.model_info.llm, delay=lambda row: 0.02)

    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=workflow), 60))

    assert logged == [1, 2, 3, 4, 5, 6]
    assert draws and all(draws)  # each draw was made while some label had room
    dispatcher = draws_of(trainer).dispatcher
    assert not dispatcher.skipped and dispatcher.next_position == len(workflow.started) + sum(
        dispatcher.parked_by_label().values()
    )


# --- resumes ------------------------------------------------------------------------------------------------------


def _positions(run_dir: Path) -> tuple[dict[tuple[int, int], int], list[int], list[tuple[int, str]]]:
    """From the run's event log: each draw's position, by ``(start, seq)``; the positions of the groups that settled;
    and the position and fate of each fate."""

    events = list(read_events(run_dir))
    position = {(event["start"], event["seq"]): event["position"] for event in events if event["event"] == "draw"}
    settled = [position[(event["start"], event["seq"])] for event in events if event["event"] == "settle"]
    fates = [(position[(event["start"], event["seq"])], event["fate"]) for event in events if event["event"] == "fate"]
    return position, settled, fates


def test_a_run_interrupted_twice_consumes_every_position_exactly_once(tmp_path):
    """Interrupted at its seventh group twice, each time with groups that had settled and that no step had trained,
    and a group running: the next start trains the kept groups without rolling them out again, rolls out the running
    one again as a new draw, and goes on with the order.  Over the three starts every position of the order is rolled
    out to its end once and trained once."""

    service = FakeTinkerService()
    overrides = (
        "train.batch_size=2",
        "train.max_training_steps=10",
        "train.max_staleness=2",
        "train.max_concurrent_rollouts=48",
        "checkpoint={every: 100}",
    )
    trained: list[int] = []
    starts = []
    for interrupt in (7, 7, None):
        trainer, client, logged = service_start(tmp_path, service, *overrides)
        workflow = PositionGroups(trainer.model_info.llm, on_start=interrupt_at(interrupt) if interrupt else None)
        run_start(trainer, workflow)
        trained += [position for step in client.steps for position in step]
        starts.append((logged, workflow.started))

    run_dir = Path(trainer.run_log_path)
    rows = read_checkpoint_records(run_dir)
    interrupted = [row for row in rows if row["name"].endswith("_interrupted")]
    assert [row["name"] for row in interrupted] == ["000001_interrupted", "000004_interrupted"]
    for row in interrupted:  # each kept groups that had settled, for the next start to train
        state = json.loads((run_dir / row["dispatch"]["file"]).read_bytes())
        assert state["kept"]["groups"]
        assert (run_dir / row["dispatch"]["kept"]).is_file()
    assert sorted(trained) == list(range(20))  # every position trained exactly once
    _draws, settled, fates = _positions(run_dir)
    assert sorted(settled) == list(range(20))  # and rolled out to its end once: no kept group again
    assert sorted(position for position, fate in fates if fate == "consumed") == list(range(20))
    assert {fate for _position, fate in fates} == {"consumed"}
    # Rolled out again: the groups running at the interrupts, each a draw of the next start at its position.
    reissued = [event for event in read_events(run_dir) if event.get("reason") == "reissue"]
    rolled_out = [position for _logged, started in starts for position in started]
    assert reissued and len(rolled_out) == 20 + len(reissued)


def _published(llm, *versions: int) -> dict[int, asyncio.Event]:
    """An event for each of ``versions``, set once ``llm`` publishes that policy version.  A start publishes step
    ``v``'s policy only after it saves checkpoint ``recovery-<v>``, so a group that waits for it ends after that save,
    however long the save takes."""

    events = {version: asyncio.Event() for version in versions}
    publish = llm.update_sampling_client

    def update_sampling_client(client, increment_version=True):
        publish(client, increment_version)
        for version, event in events.items():
            if llm.version >= version:
                event.set()

    llm.update_sampling_client = update_sampling_client
    return events


@pytest.mark.parametrize("kept", ["kept", "file lost", "never written"])
def test_a_resume_trains_each_group_its_checkpoint_kept_once_and_without_it_rolls_them_out_again(
    tmp_path, monkeypatch, kept, caplog
):
    """At recovery-000001 the groups of positions 2 to 5 had settled and no step had trained them: the checkpoint keeps
    them, and a resume from it trains each once, rolling out only the positions after them.  Without them (their file
    lost, or never written), the resume rolls them out again, as before, and still trains each position once."""

    service = FakeTinkerService()
    real = kept_groups.write_kept
    if kept == "never written":

        def failing(run_dir, name, groups):
            if name.endswith("recovery-000001.json.gz"):
                raise OSError(28, "No space left on device")
            return real(run_dir, name, groups)

        monkeypatch.setattr(kept_groups, "write_kept", failing)
    first, _client, _logged = service_start(tmp_path, service, *RECOVERY_RUN)
    run_start(first, PositionGroups(first.model_info.llm))
    run_dir = Path(first.run_log_path)
    rows = crash_after(run_dir, "recovery-000001")
    state = dispatch_state(run_dir, rows[-1])
    outstanding = sorted(position for _start, _seq, _task, position in state["outstanding"])
    assert outstanding == [2, 3, 4, 5]
    snapshot = json.loads((run_dir / rows[-1]["tasks"]["file"]).read_bytes())
    assert sorted(position for *_key, position in snapshot["outstanding"]) == outstanding  # the statistics agree
    if kept == "never written":
        assert state["kept"] is None and "kept" not in rows[-1]["dispatch"]
    else:
        assert len(state["kept"]["groups"]) == 4
        if kept == "file lost":
            (run_dir / rows[-1]["dispatch"]["kept"]).unlink()
    monkeypatch.setattr(kept_groups, "write_kept", real)

    second, client, logged = service_start(tmp_path, service, *RECOVERY_RUN)
    workflow = PositionGroups(second.model_info.llm)
    with caplog.at_level(logging.WARNING):
        run_start(second, workflow)

    assert logged == [2, 3, 4]
    trained = [position for step in client.steps for position in step]
    assert sorted(trained) == [2, 3, 4, 5, 6, 7]  # each position after the checkpoint trained once
    if kept == "kept":
        assert sorted(workflow.started) == [6, 7]  # the kept groups were not rolled out again
        assert sorted(trained[:4]) == [2, 3, 4, 5]  # and the first steps trained them
        store = draws_of(second).recorder.store
        assert sum(store.task(task_id).groups for task_id in store.task_ids()) == 8  # each group settled once
    else:
        assert sorted(workflow.started) == [2, 3, 4, 5, 6, 7]  # rolled out again, as before
    if kept == "file lost":
        assert any("cannot be read, so their tasks are rolled out again" in r.getMessage() for r in caplog.records)


def test_pruning_deletes_the_groups_a_checkpoint_kept_with_it(tmp_path):
    service = FakeTinkerService()
    trainer, _client, _logged = service_start(
        tmp_path,
        service,
        "train.batch_size=2",
        "train.max_training_steps=6",
        "train.max_staleness=1",
        "train.max_concurrent_rollouts=32",
        "checkpoint={every: 100, recovery_every: 1, keep_recovery: 2}",
    )
    run_start(trainer, PositionGroups(trainer.model_info.llm))

    run_dir = Path(trainer.run_log_path)
    deleted = {json.loads(line)["path"] for line in (run_dir / DELETED_CHECKPOINTS_NAME).read_text().splitlines()}
    kept_files = {row["name"]: row["dispatch"].get("kept") for row in read_checkpoint_records(run_dir)}
    pruned = [row for row in read_checkpoint_records(run_dir) if row["state_path"] in deleted]
    assert pruned and any(row["dispatch"].get("kept") for row in pruned)
    for row in read_checkpoint_records(run_dir):
        for file in (row["tasks"]["file"], row["dispatch"]["file"], row["dispatch"].get("kept")):
            if file is not None:
                assert (run_dir / file).exists() == (row["state_path"] not in deleted), (row["name"], kept_files)
    assert not list((run_dir / "tasks").glob(".*partial"))


def test_a_resume_whose_sampling_inputs_changed_warns_records_it_and_goes_on(tmp_path, caplog):
    """Two of the run's eight tasks leave it between the checkpoint and the resume: the start warns, lock.json lists
    the change under sampling_changed, the outstanding draws of the tasks that left are lost and the others go on."""

    service = FakeTinkerService()
    first, _client, _logged = service_start(tmp_path, service, *RECOVERY_RUN)
    run_start(first, PositionGroups(first.model_info.llm))
    run_dir = Path(first.run_log_path)
    rows = crash_after(run_dir, "recovery-000001")
    state = dispatch_state(run_dir, rows[-1])
    (run_dir / "lock.json").write_text("{}\n")
    # The tasks of two of the outstanding draws leave the run.
    gone = {task for _start, _seq, task, position in state["outstanding"] if position in (2, 3)}

    kept = task_ids_to_dataset([f"train:{index}" for index in range(8) if f"train:{index}" not in gone])
    second, client, logged = service_start(tmp_path, service, *RECOVERY_RUN, train_dataset=kept)
    locks_at_warning = _LockAtWarning(run_dir / "lock.json")
    logging.getLogger().addHandler(locks_at_warning)
    try:
        with caplog.at_level(logging.WARNING):
            run_start(second, PositionGroups(second.model_info.llm))
    finally:
        logging.getLogger().removeHandler(locks_at_warning)

    [changed] = json.loads((run_dir / "lock.json").read_text())[task_records.CHANGED_KEY]
    assert changed == {
        "start": 2,
        "checkpoint": "recovery-000001",
        "changes": ["environments[default]: its training tasks, 8 -> 6"],
    }
    assert any("sampling inputs changed since checkpoint recovery-000001" in r.getMessage() for r in caplog.records)
    left = {(start, seq) for start, seq, task, _position in state["outstanding"] if task in gone}
    lost = {(event["start"], event["seq"]) for event in read_events(run_dir) if event.get("fate") == "lost"}
    assert len(gone) == 2 and lost == left
    assert logged == [2, 3, 4]
    assert locks_at_warning.seen == [{}]  # warned of first, then recorded


class _LockAtWarning(logging.Handler):
    """The run's lock as each warning of changed sampling inputs is emitted (``seen``)."""

    def __init__(self, lock_path) -> None:
        super().__init__(logging.WARNING)
        self.lock_path = lock_path
        self.seen: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        if "sampling inputs changed" in record.getMessage():
            self.seen.append(json.loads(self.lock_path.read_text()))


@pytest.mark.parametrize("state", ["none", "changed"])
def test_a_resume_from_a_checkpoint_without_the_dispatchers_state_draws_from_its_batch(tmp_path, caplog, state):
    """A checkpoint whose state of the dispatcher's draws could not be written names the statistics' snapshot but no
    such state, and one whose state was changed since does not read back: a resume from either counts the outstanding
    draws as lost and draws from the checkpoint's batch, as the producer did."""

    service = FakeTinkerService()
    first, _client, _logged = service_start(tmp_path, service, *RECOVERY_RUN)
    run_start(first, PositionGroups(first.model_info.llm))
    run_dir = Path(first.run_log_path)
    rows = crash_after(run_dir, "recovery-000001")
    snapshot = json.loads((run_dir / rows[-1]["tasks"]["file"]).read_bytes())
    if state == "none":
        for name in (rows[-1]["dispatch"]["file"], rows[-1]["dispatch"]["kept"]):
            (run_dir / name).unlink()
        del rows[-1]["dispatch"]
        (run_dir / "checkpoints.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        (run_dir / rows[-1]["dispatch"]["file"]).write_text(
            json.dumps({**dispatch_state(run_dir, rows[-1]), "waiting": {}})
        )

    second, client, logged = service_start(tmp_path, service, *RECOVERY_RUN)
    workflow = PositionGroups(second.model_info.llm)
    with caplog.at_level(logging.WARNING):
        run_start(second, workflow)

    assert sorted(workflow.started) == [2, 3, 4, 5, 6, 7]  # from position 1 × 2 on, the outstanding ones again
    lost = sorted(event["seq"] for event in read_events(run_dir) if event.get("fate") == "lost")
    assert len(lost) == len(snapshot["outstanding"]) == 4
    warned = any("saved cannot be read" in entry.getMessage() for entry in caplog.records)
    assert warned == (state == "changed")


# --- exact resume without the statistics ----------------------------------------------------------------------------

EXACT_RUN = (
    "train.batch_size=2",
    "train.max_training_steps=6",
    "train.max_staleness=2",
    "train.max_concurrent_rollouts=48",
    "checkpoint.recovery_every=1",
)


@pytest.mark.parametrize("statistics", ["stopped", "not written"])
def test_a_resume_is_exact_when_the_statistics_stopped_or_were_not_written(tmp_path, monkeypatch, statistics):
    """The state of the dispatcher's draws is a file of its own: whether the run's per-task statistics stopped after an
    error (no checkpoint names a snapshot after it) or the checkpoint's snapshot could not be written, a crash after
    recovery-000004 resumes exactly where it stood.  Every third group runs until the first start has saved
    recovery-000004, so the run's first four steps consume the order out of its order, and drawing again from the
    checkpoint's batch would train some positions twice and lose others.

    The groups wait for that save, not for a time: a group that took a fixed time could end before the steps on a
    slow machine, and a step would then dequeue it more than ``max_staleness`` policy versions after it sampled and
    drop it as stale.  That position is consumed exactly once all the same, as RFC 0019 has it, but not trained."""

    service = FakeTinkerService()
    calls: list[int] = []
    real_metrics, real_write = TaskStatsStore.step_metrics, task_records.write_snapshot

    def failing_metrics(self, **kwargs):
        calls.append(1)
        if len(calls) == 2:  # step 2's: the statistics stop, and recovery-000003 and on name no snapshot
            raise RuntimeError("a bug in the statistics")
        return real_metrics(self, **kwargs)

    def failing_write(run_dir, name, data):
        if name == "state-s1-recovery-000004.json":
            raise OSError(28, "No space left on device")
        return real_write(run_dir, name, data)

    if statistics == "stopped":
        monkeypatch.setattr(TaskStatsStore, "step_metrics", failing_metrics)
    else:
        monkeypatch.setattr(task_records, "write_snapshot", failing_write)
    first, first_client, _logged = service_start(tmp_path, service, *EXACT_RUN)
    llm = first.model_info.llm
    saved = _published(llm, 4)[4]  # step 4's policy: recovery-000004 is saved
    run_start(
        first,
        PositionGroups(llm, delay=lambda row: 0.0, hold=lambda row: saved if row[POSITION_COLUMN] % 3 == 0 else None),
    )
    run_dir = Path(first.run_log_path)
    rows = crash_after(run_dir, "recovery-000004")
    assert "tasks" not in rows[-1] and "dispatch" in rows[-1]
    state = dispatch_state(
        run_dir, rows[-1]
    )  # every third group still running, and no settled group waiting for a step
    assert sorted(position for *_key, position in state["outstanding"]) == [0, 3, 6, 9]
    assert state["kept"] is None and state["next_position"] == 12
    trained = [position for step in first_client.steps[:4] for position in step]  # the steps before the crash
    assert sorted(trained) == [1, 2, 4, 5, 7, 8, 10, 11]  # out of the order's order: its batches would not resume it
    monkeypatch.setattr(TaskStatsStore, "step_metrics", real_metrics)
    monkeypatch.setattr(task_records, "write_snapshot", real_write)

    second, client, logged = service_start(tmp_path, service, *EXACT_RUN)
    workflow = PositionGroups(second.model_info.llm)
    run_start(second, workflow)

    assert logged == [5, 6]
    assert sorted(workflow.started) == [0, 3, 6, 9]  # the draws that were running, drawn again, and no other
    trained += [position for step in client.steps for position in step]
    assert sorted(trained) == list(range(12))  # every position trained exactly once: none could be stale


@pytest.mark.parametrize("statistics", ["written", "not written"])
def test_a_group_a_step_dropped_as_stale_before_the_checkpoint_is_not_drawn_again(tmp_path, monkeypatch, statistics):
    """A slow machine's interleaving: position 3's group ends only once recovery-000003 is saved, three policy
    versions after it sampled, so step 4 dequeues it, drops it as stale (``max_staleness`` 2) and trains only position
    10 of its batch.  A position dropped as stale is consumed (RFC 0019, section 4): recovery-000004 holds it neither
    outstanding nor kept, and the resume draws again only the draws that were still running, so every other position
    is trained once and position 3 never, whether or not the checkpoint's per-task statistics were written."""

    service = FakeTinkerService()
    real_write = task_records.write_snapshot

    def failing_write(run_dir, name, data):
        if name == "state-s1-recovery-000004.json":
            raise OSError(28, "No space left on device")
        return real_write(run_dir, name, data)

    if statistics == "not written":
        monkeypatch.setattr(task_records, "write_snapshot", failing_write)
    first, first_client, first_logged = service_start(tmp_path, service, *EXACT_RUN)
    published = _published(first.model_info.llm, 3, 4)
    held = {3: published[3], 0: published[4], 6: published[4], 9: published[4], 11: published[4]}
    run_start(
        first,
        PositionGroups(first.model_info.llm, delay=lambda row: 0.0, hold=lambda row: held.get(row[POSITION_COLUMN])),
    )
    run_dir = Path(first.run_log_path)
    rows = crash_after(run_dir, "recovery-000004")
    state = dispatch_state(run_dir, rows[-1])
    assert sorted(position for *_key, position in state["outstanding"]) == [0, 6, 9, 11]  # not 3: step 4 consumed it
    assert state["kept"] is None and state["next_position"] == 12
    assert first_client.steps[3] == [10]  # step 4 trained position 10 alone
    assert next(stats for step, stats in first_logged.rows() if step == 4)["train/batch/stale_groups"] == 1
    if statistics == "written":
        snapshot = json.loads((run_dir / rows[-1]["tasks"]["file"]).read_bytes())
        assert sorted(position for *_key, position in snapshot["outstanding"]) == [0, 6, 9, 11]  # the statistics agree
    else:
        assert "tasks" not in rows[-1] and "dispatch" in rows[-1]
    trained = [position for step in first_client.steps[:4] for position in step]  # the steps before the crash
    monkeypatch.setattr(task_records, "write_snapshot", real_write)

    second, client, logged = service_start(tmp_path, service, *EXACT_RUN)
    workflow = PositionGroups(second.model_info.llm)
    run_start(second, workflow)

    assert logged == [5, 6]
    assert sorted(workflow.started) == [0, 6, 9, 11]  # the draws that were running, drawn again, and not position 3
    trained += [position for step in client.steps for position in step]
    assert sorted(trained) == [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]  # every other position trained exactly once
    if statistics == "written":
        _draws, _settled, fates = _positions(run_dir)
        assert [fate for position, fate in fates if position == 3] == ["stale"]  # consumed once, as stale


# --- groups that fail ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("error", ["raises", "cancelled from within"])
def test_a_group_that_fails_settles_as_failed_frees_its_slot_and_is_counted(tmp_path, error):
    """One group at a time (permits for one group of 8): the group at position 1 raises, or its workflow's await is
    cancelled while the group is not (a task it awaited was cancelled).  Either way it settles with no output: its
    permits come back, the run goes on, failed_rollouts is logged, and its draw's fate is failed."""

    service = FakeTinkerService()
    trainer, client, logged = service_start(
        tmp_path,
        service,
        "train.batch_size=2",
        "train.max_training_steps=3",
        "train.max_staleness=1",
        "train.max_concurrent_rollouts=8",
    )
    workflow = FailingGroups(trainer.model_info.llm, [1], asyncio.CancelledError() if error != "raises" else None)

    async def run() -> None:
        await asyncio.wait_for(trainer.train(train_workflow=workflow), 30)

    asyncio.run(run())

    assert logged == [1, 2, 3]
    assert any(value for stats in logged.stats for key, value in stats.items() if key.endswith("failed_rollouts"))
    assert 1 not in [position for step in client.steps for position in step]
    _draws, _settled, fates = _positions(Path(trainer.run_log_path))
    assert dict(fates)[1] == "failed"


def test_a_failed_group_a_checkpoint_kept_is_consumed_as_failed_after_the_resume(tmp_path):
    """At recovery-000001 the group of position 3 had failed and no step had consumed it: the checkpoint keeps it as
    failed, and the resume consumes it as failed, as the run would have without the restart, rather than rolling its
    task out again: its task counts one group."""

    service = FakeTinkerService()
    first, _client, _logged = service_start(tmp_path, service, *RECOVERY_RUN)
    run_start(first, FailingGroups(first.model_info.llm, [3]))
    run_dir = Path(first.run_log_path)
    rows = crash_after(run_dir, "recovery-000001")
    [task] = [
        task for _start, _seq, task, position in dispatch_state(run_dir, rows[-1])["outstanding"] if position == 3
    ]

    second, client, _logged = service_start(tmp_path, service, *RECOVERY_RUN)
    workflow = PositionGroups(second.model_info.llm)
    run_start(second, workflow)

    assert sorted(workflow.started) == [6, 7]  # position 3 was not rolled out again
    assert sorted(position for step in client.steps for position in step) == [2, 4, 5, 6, 7]
    record = draws_of(second).recorder.store.task(task)
    assert record.groups == 1 and record.fates == {"failed": 1}


# --- a step an interrupt drops leaves its groups as they settled ----------------------------------------------------


def test_a_step_an_interrupt_drops_leaves_the_groups_it_dequeued_as_they_settled(tmp_path):
    """A batch transform that doubles the advantages in place, two microbatches a step: the interrupt comes as step 2
    has sent its first microbatch, whose group the interrupt checkpoint keeps.  The resume transforms it once, as every
    other group: the step's datums are its own, not the group's."""

    service = FakeTinkerService()
    overrides = (
        "train.batch_size=2",
        "train.num_microbatches=2",
        "train.max_training_steps=3",
        "train.max_staleness=2",
        "train.max_concurrent_rollouts=48",
        "checkpoint={every: 100}",
    )
    sent: list[tuple[int, list[float]]] = []

    def double(datums, context):
        for datum in datums:
            advantages = datum.loss_fn_inputs["advantages"].to_torch()
            datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(advantages * 2)
        return datums

    def start(interrupt_at: int | None):
        trainer, logged = make_service_trainer(trainer_config(tmp_path, *overrides), service, batch_transforms=[double])
        calls: list[int] = []

        async def recording(datums):
            calls.append(1)
            for datum in datums:
                sent.append((datum.model_input.to_ints()[0], datum.loss_fn_inputs["advantages"].data))
            if len(calls) == interrupt_at:
                signal.raise_signal(signal.SIGINT)

        service.on_forward_backward = recording
        return trainer, logged

    first, _logged = start(interrupt_at=3)  # step 2's first microbatch
    run_start(first, PositionGroups(first.model_info.llm))
    run_dir = Path(first.run_log_path)
    assert [row["name"] for row in read_checkpoint_records(run_dir)][-1] == "000001_interrupted"
    dropped = sent[-1][0]  # the group of the microbatch the dropped step sent
    second, logged = start(interrupt_at=None)
    workflow = PositionGroups(second.model_info.llm)
    run_start(second, workflow)

    assert logged == [2, 3]
    assert dropped not in workflow.started  # kept, not rolled out again
    assert len({tuple(advantages) for _position, advantages in sent}) == 1  # each group transformed once
