"""The trainer keeps the run's per-task statistics, in its own loop (Tinker faked): every step logs each label's
``tasks/`` metrics and the kinds of the groups it trained on; each full checkpoint saves a snapshot that its row names
with its SHA-256; a resume loads it before its launch save, which saves it again under the new start's own name; and
pruning a checkpoint deletes the snapshot its own row names, never another start's."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from datasets import Dataset
from fakes.config import make_config, mixture_config, trainer_config
from fakes.runs import run_start
from fakes.tinker_service import FakeTinkerService, make_service_trainer
from fakes.trainer import TrainingClient, Workflow, datum, make_trainer

from platoon.task import Task
from platoon.train import entry_settings, task_records
from platoon.train.checkpoints.pruning import read_checkpoint_records
from platoon.train.components import ENVIRONMENT_COLUMN
from platoon.train.dataset_order import PRESERVE_DATASET_ORDER_COLUMN
from platoon.train.task_records import LOST_KEY
from platoon.train.task_stats import TaskStatsStore, read_events, snapshot_bytes
from platoon.train.workers.protocol import PROTOCOL_VERSION, RolloutWorkerResponse
from platoon.train.workflow import members as members_module
from platoon.train.workflow.group import GroupRolloutWorkflow
from platoon.train.workflow.output import _TaskRolloutOutput
from platoon.utils.rollout_workload import RolloutWorkload
from platoon.utils.tinker_data_processing import TrajectoryCollectionResult, TrajectoryStats


class _SlowTrainingClient(TrainingClient):
    """Each forward and backward takes 50 ms, so the groups running beside a step settle before it ends."""

    async def forward_backward_async(self, datums, loss_fn, loss_fn_config):
        await asyncio.sleep(0.05)
        return await super().forward_backward_async(datums, loss_fn, loss_fn_config)


def test_a_cpu_run_logs_each_label_s_pass_at_1_and_pass_at_G_every_step(tmp_path, monkeypatch):
    """Two entries, groups of 4 and of 2, whose rewards a shaping reward processor scaled by 0.98: each step logs
    each label's mean expected pass@1 and pass@G, the kinds of the groups it trained on partition them, and the pass
    counts are those of the rewards the environment scored."""

    config = mixture_config(
        tmp_path,
        {"batch_size": 2, "max_training_steps": 4, "max_concurrent_rollouts": 12, "workflow_config": {"group_size": 4}},
        [{"name": "quick"}, {"name": "slow", "workflow_config": {"group_size": 2}}],
    )
    settings = entry_settings.entry_settings(config)
    # quick and slow tasks in turn, as they are: each batch holds one of each.
    tasks = Dataset.from_list(
        [
            {"task_id": f"{name}::{name}{index}", ENVIRONMENT_COLUMN: name, PRESERVE_DATASET_ORDER_COLUMN: True}
            for index in range(3)
            for name in ("quick", "slow")
        ]
    )
    trainer, saves, logged = make_trainer(config, training_client=_SlowTrainingClient(), train_dataset=tasks)
    policy = trainer.model_info.llm

    async def run_rollout_request(request, llm, *, rollout_fn=None, reward_processor=None):
        await asyncio.sleep(0.001)
        # Every member of a slow task passes; every other member of a quick task does.
        success = 1.0 if request.task_id.startswith("slow::") or request.rollout_number % 2 == 0 else 0.0
        reward = 0.98 * success  # what the reward processor trains
        stats = TrajectoryStats(
            trajectory_id="root",
            reward=reward,
            num_steps=3,
            num_input_tokens=3,
            num_output_tokens=5,
            num_datums=1,
            is_root=True,
        )
        result = TrajectoryCollectionResult(
            datums=[datum(advantage=reward, version=policy.version)],
            task_reward=reward,
            trajectory_stats=[stats],
            root_rewards_dict={},
            task_reward_valid=True,
        )
        return RolloutWorkerResponse(
            protocol_version=PROTOCOL_VERSION,
            request_id=request.request_id,
            status="completed",
            result=result,
            workload=RolloutWorkload(
                trajectories=1, postmerge_datums=1, policy_eligible_datums=1, post_sampling_datums=1
            ),
            observed=True,
            outcome_status="scored",
            success_reward=success,
            rollout_seconds=0.5,
        )

    monkeypatch.setattr(members_module, "run_rollout_request", run_rollout_request)
    loop_states = saves.loop_states
    workflow = GroupRolloutWorkflow(
        rollout_fn=lambda task, rollout_config: None,
        get_task_fn=lambda task_id: Task(id=task_id),
        config=config.train.workflow_config,
        model_info=SimpleNamespace(model_name="m", llm=SimpleNamespace(version=0, model_name="m")),
    )
    workflow.entry_configs = {label: own.train for label, own in settings.items()}

    asyncio.run(trainer.train(train_workflow=workflow))

    assert len(logged) == 4
    for step in logged.stats:
        for label, size in (("quick", 4), ("slow", 2)):
            pass_1, pass_g = (
                step[f"train/tasks/environment/{label}/pass_at_1"],
                step[f"train/tasks/environment/{label}/pass_at_G"],
            )
            assert 0 < pass_1 < pass_g < 1, label
            assert step[f"train/tasks/environment/{label}/coverage"] > 0
        assert 0 < step["train/tasks/pass_at_1"] < step["train/tasks/pass_at_G"] < 1
        # Each step's two groups: a quick one, which trains, and a slow one, which every member passed.
        assert (
            step["train/batch/environment/quick/informative_groups"]
            + step["train/batch/environment/slow/all_pass_groups"]
            == 2
        )
        assert step["train/batch/informative_groups"] + step["train/batch/all_pass_groups"] == 2
        assert step["train/batch/all_fail_groups"] == step["train/batch/uniform_mixed_groups"] == 0
    # The reward processor's 0.98 changed no pass: half of each quick group's members passed, and every slow one.
    store = trainer._recorder.store
    for task_id in store.task_ids():
        task = store.task(task_id)
        assert task.passes == (task.attempts if task_id.startswith("slow::") else task.attempts // 2) and task.groups
    assert store.label("slow").all_pass_groups == store.label("slow").groups
    # The final checkpoint's row names its snapshot, which holds the same counts.
    final = loop_states["final"]["tasks"]
    content = (Path(trainer.run_log_path) / final["file"]).read_bytes()
    assert final["sha256"] == hashlib.sha256(content).hexdigest() and final["start"] == 1
    saved = TaskStatsStore.from_snapshot(json.loads(content))
    assert [saved.task(task_id).passes for task_id in saved.task_ids()] == [
        store.task(task_id).passes for task_id in store.task_ids()
    ]
    # The event log holds the start, each draw, each settle and each fate, joined on (start, seq).
    events = list(read_events(trainer.run_log_path))
    assert events[0]["event"] == "start" and events[0]["start"] == 1 and events[0]["state_lost"] is None
    draws = {(event["start"], event["seq"]) for event in events if event["event"] == "draw"}
    settles = {(event["start"], event["seq"]) for event in events if event["event"] == "settle"}
    consumed = [(event["start"], event["seq"]) for event in events if event.get("fate") == "consumed"]
    assert len(consumed) == 8 and set(consumed) <= settles <= draws


# --- resuming, and pruning ------------------------------------------------------------------------------------------


def _train(trainer, workflow: Workflow | None = None) -> None:
    """Run ``train()`` to its end (``fakes.runs.run_start``); one cut short by a timeout would save an interrupt
    checkpoint and return."""

    run_start(trainer, workflow or Workflow(delay=0.02))


def _states(run_dir: Path) -> list[str]:
    return sorted(path.name for path in (run_dir / "tasks").glob("state-*"))


def test_a_resume_saves_the_loaded_statistics_under_its_own_start_and_pruning_deletes_only_its_own(tmp_path):
    """A run crashes after recovery-000001 and resumes from it: the new start's launch save, recovery-000001 again,
    saves the loaded statistics as its own snapshot and leaves the older row's snapshot and SHA-256 as they were; and
    pruning the older recovery-000001 deletes the file its own row names, never the newer one."""

    service = FakeTinkerService()
    first, _logged = make_service_trainer(
        trainer_config(tmp_path, "train.max_training_steps=2", "train.max_staleness=1", "checkpoint.recovery_every=1"),
        service,
    )
    _train(first)
    run_dir = Path(first.run_log_path)
    rows = read_checkpoint_records(run_dir)
    assert [row["name"] for row in rows] == ["recovery-000000", "recovery-000001", "recovery-000002", "final"]
    for row in rows:
        assert row["tasks"]["file"] == f"tasks/state-s1-{row['name']}.json" and row["tasks"]["start"] == 1
        assert row["tasks"]["sha256"] == hashlib.sha256((run_dir / row["tasks"]["file"]).read_bytes()).hexdigest()

    # The crash: nothing after recovery-000001 was saved.
    (run_dir / "checkpoints.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows[:2]))
    for row in rows[2:]:
        for name in (row["tasks"]["file"], row["dispatch"]["file"], row["dispatch"].get("kept")):
            if name is not None:
                (run_dir / name).unlink()
    older = rows[1]["tasks"]
    older_bytes = (run_dir / older["file"]).read_bytes()
    saved = json.loads(older_bytes)
    assert saved["outstanding"]  # the draws of step 2, which no committed step consumed
    # A loaded snapshot serializes byte for byte as the one saved; the dispatcher's state is a file of its own.
    loaded = TaskStatsStore.from_snapshot(saved)
    assert snapshot_bytes(loaded.snapshot(start=saved["start"], next_seq=saved["next_seq"])) == older_bytes
    dispatch = json.loads((run_dir / rows[1]["dispatch"]["file"]).read_bytes())
    assert sorted(map(tuple, dispatch["outstanding"])) == sorted(map(tuple, saved["outstanding"]))

    seen: dict[str, Any] = {}

    def watch(path: str) -> None:
        """Look at the snapshots as pruning deletes the older recovery-000001: its state, then its sampler."""

        if path == "tinker://run-1/weights/recovery-000001":
            seen["before"], seen["older"] = _states(run_dir), (run_dir / older["file"]).read_bytes()
        elif path == "tinker://run-1/sampler_weights/recovery-000001":
            seen["after"] = _states(run_dir)

    service.failure = watch
    second, _logged = make_service_trainer(
        trainer_config(
            tmp_path,
            "train.max_training_steps=2",
            "train.max_staleness=1",
            "checkpoint={every: 100, recovery_every: 1, keep_recovery: 2}",
        ),
        service,
    )
    _train(second)

    rows = read_checkpoint_records(run_dir)
    launch = rows[2]
    assert (launch["name"], launch["state_path"]) == ("recovery-000001", "tinker://run-2/weights/recovery-000001")
    assert launch["tasks"]["file"] == "tasks/state-s2-recovery-000001.json" and launch["tasks"]["start"] == 2
    assert rows[1]["tasks"] == older  # the older row is as it was
    # Until pruning deleted the older checkpoint, its snapshot was as it saved it; then that file alone went.
    assert seen["older"] == older_bytes and hashlib.sha256(older_bytes).hexdigest() == older["sha256"]
    assert {"state-s1-recovery-000001.json", "state-s2-recovery-000001.json"} <= set(seen["before"])
    assert set(seen["before"]) - set(seen["after"]) == {"state-s1-recovery-000001.json"}
    assert "state-s2-recovery-000001.json" in _states(run_dir)
    # The launch save holds the loaded statistics, the draws no committed step consumed still outstanding: a group the
    # checkpoint kept under its own draw, the others drawn again as this start's draws, at their positions.
    newer = (run_dir / launch["tasks"]["file"]).read_bytes()
    assert hashlib.sha256(newer).hexdigest() == launch["tasks"]["sha256"]
    relaunched = json.loads(newer)
    kept = {tuple(key) for key in (dispatch["kept"] or {}).get("groups", [])}
    before = {(start, seq): (task, place) for start, seq, task, place in saved["outstanding"]}
    after = {(start, seq): (task, place) for start, seq, task, place in relaunched["outstanding"]}
    assert sorted(after.values()) == sorted(before.values())
    assert {key for key in after if key[0] == 1} == kept and {key[0] for key in set(after) - kept} == {2}
    assert relaunched["tasks"] == saved["tasks"]  # each task's draws and fates as they were: none lost
    events = list(read_events(run_dir))
    starts = [(event["start"], event["batch"], event["checkpoint"]) for event in events if event["event"] == "start"]
    assert starts == [(1, 0, None), (2, 1, "recovery-000001")]
    assert not [event for event in events if event.get("fate") == "lost"]
    reissued = [tuple(event["reissues"]) for event in events if event.get("reason") == "reissue"]
    assert sorted(reissued) == sorted(set(before) - kept)


@pytest.mark.parametrize("damage", ["no snapshot", "changed snapshot", "every snapshot"])
def test_a_resume_without_its_statistics_goes_on_from_an_older_snapshot_and_records_why(tmp_path, damage):
    """A resume whose checkpoint names no snapshot (one a platoon that kept none saved, or one whose write failed), or
    names one that is changed, goes on from the newest older checkpoint's snapshot that reads back (here the first
    start's launch save, before any group), else from empty, and lock.json lists why, by start."""

    service = FakeTinkerService()
    config = trainer_config(tmp_path, "train.max_training_steps=1", "checkpoint.recovery_every=1")
    first, _logged = make_service_trainer(config, service)
    _train(first)
    run_dir = Path(first.run_log_path)
    rows = read_checkpoint_records(run_dir)[:-1]  # resume from recovery-000001
    if damage == "no snapshot":
        rows[-1].pop("tasks")
    else:
        for row in rows if damage == "every snapshot" else rows[-1:]:
            path = run_dir / row["tasks"]["file"]
            path.write_bytes(path.read_bytes().replace(b'"start":1', b'"start":7'))
    (run_dir / "checkpoints.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (run_dir / "lock.json").write_text("{}\n")

    second, _logged = make_service_trainer(
        trainer_config(tmp_path, "train.max_training_steps=2", "checkpoint.recovery_every=1"), service
    )
    _train(second)

    [lost] = json.loads((run_dir / "lock.json").read_text())[LOST_KEY]
    assert (lost["start"], lost["checkpoint"]) == (2, "recovery-000001")
    if damage == "no snapshot":
        assert lost["problem"] == "checkpoint recovery-000001 names no snapshot of the per-task statistics"
    else:
        assert lost["problem"].endswith("does not have the SHA-256 its checkpoint recorded")
    assert lost["loaded"] == (None if damage == "every snapshot" else "recovery-000000")
    store = second._recorder.store
    assert sum(store.task(task_id).groups for task_id in store.task_ids()) == 2  # step 2's groups alone
    start = [event for event in read_events(run_dir) if event["event"] == "start"][-1]
    assert (start["state_lost"], start["loaded"]) == (lost["problem"], lost["loaded"])


def test_statistics_that_fail_stop_with_a_warning_and_training_goes_on(tmp_path, monkeypatch, caplog):
    def settle(self, record):
        raise RuntimeError("a bug in the statistics")

    monkeypatch.setattr(TaskStatsStore, "settle", settle)
    trainer, saves, logged = make_trainer(trainer_config(tmp_path, "checkpoint.recovery_every=1"))
    loop_states = saves.loop_states

    with caplog.at_level(logging.WARNING):
        asyncio.run(trainer.train(train_workflow=Workflow(delay=0.01)))

    assert logged == [1, 2, 3]
    assert "tasks" in loop_states["recovery-000000"]  # the launch save, before any group settled
    assert "tasks" not in loop_states["recovery-000001"] and "tasks" not in loop_states["final"]
    # Where the run's draws stand is saved all the same, so a resume from any of them is exact.
    assert all("dispatch" in loop_state for loop_state in loop_states.values())
    assert any(
        "per-task statistics stop for the rest of this start: recording a settled group failed" in entry.getMessage()
        for entry in caplog.records
    )


def test_a_snapshot_that_cannot_be_written_costs_its_checkpoint_alone(tmp_path, monkeypatch, caplog):
    """A full disk at one save: that checkpoint names no snapshot, the statistics go on, and the next save writes them.
    A resume from that very checkpoint goes on from the newest older snapshot, not from empty."""

    service = FakeTinkerService()
    real = task_records.write_snapshot
    failed: list[str] = []

    def flaky(run_dir, name, data):
        if name == "state-s1-recovery-000002.json" and not failed:
            failed.append(name)
            raise OSError(28, "No space left on device")
        return real(run_dir, name, data)

    monkeypatch.setattr(task_records, "write_snapshot", flaky)
    first, _logged = make_service_trainer(
        trainer_config(tmp_path, "train.max_training_steps=4", "checkpoint.recovery_every=1"), service
    )
    with caplog.at_level(logging.WARNING):
        _train(first)

    run_dir = Path(first.run_log_path)
    rows = read_checkpoint_records(run_dir)
    assert {row["name"]: "tasks" in row for row in rows} == {
        "recovery-000000": True,
        "recovery-000001": True,
        "recovery-000002": False,
        "recovery-000003": True,
        "recovery-000004": True,
        "final": True,
    }
    assert not first._recorder.stopped
    assert any(
        "Could not write the per-task statistics with checkpoint recovery-000002" in r.message for r in caplog.records
    )
    assert not list((run_dir / "tasks").glob(".*partial"))
    # A crash right after recovery-000002: the resume goes on from recovery-000001's statistics.
    (run_dir / "checkpoints.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows[:3]))
    older = TaskStatsStore.from_snapshot(json.loads((run_dir / rows[1]["tasks"]["file"]).read_bytes()))
    (run_dir / "lock.json").write_text("{}\n")

    second, _logged = make_service_trainer(
        trainer_config(tmp_path, "train.max_training_steps=4", "checkpoint.recovery_every=1"), service
    )
    workflow = Workflow(delay=0.02)
    _train(second, workflow)

    [lost] = json.loads((run_dir / "lock.json").read_text())[LOST_KEY]
    assert (lost["checkpoint"], lost["loaded"]) == ("recovery-000002", "recovery-000001")
    groups = sum(older.task(task_id).groups for task_id in older.task_ids())
    store = second._recorder.store
    # The older statistics, and each group the resume rolled out; the draws themselves resume exactly, from the
    # dispatcher's own state, which recovery-000002 saved.
    assert groups > 0 and sum(store.task(task_id).groups for task_id in store.task_ids()) == groups + workflow.started
    assert "dispatch" in rows[2] and workflow.started <= 4


def test_a_start_whose_launch_save_fails_logs_neither_its_start_nor_its_draws_drawn_again(tmp_path):
    """The next start takes the dead start's number, so the dead start's events would have doubled it and every draw
    it drew again: a start's own events wait for its launch save."""

    service = FakeTinkerService()
    config = ("train.max_training_steps=2", "train.max_staleness=1", "checkpoint.recovery_every=1")
    first, _logged = make_service_trainer(trainer_config(tmp_path, *config), service)
    _train(first, Workflow(delay=0.02))
    run_dir = Path(first.run_log_path)
    rows = read_checkpoint_records(run_dir)
    (run_dir / "checkpoints.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows[:2]))  # a crash
    calls: list[str] = []

    async def failing_once(name):
        calls.append(name)
        if len(calls) == 1:
            raise TimeoutError("save_checkpoint timed out")

    second, _logged = make_service_trainer(trainer_config(tmp_path, *config), service)
    service.on_save_state = failing_once
    with pytest.raises(TimeoutError):
        _train(second, Workflow(delay=0.02))
    third, _logged = make_service_trainer(trainer_config(tmp_path, *config), service)
    _train(third, Workflow(delay=0.02))

    events = list(read_events(run_dir))
    assert [event["start"] for event in events if event["event"] == "start"] == [1, 2]
    reissued = [(event["start"], event["seq"]) for event in events if event.get("reason") == "reissue"]
    assert reissued and len(reissued) == len(set(reissued)) and {start for start, _seq in reissued} == {2}


def test_once_the_statistics_stop_a_step_logs_no_group_kinds(tmp_path, monkeypatch):
    """Zeros would read as steps without an informative group; the kinds are left out instead, as the tasks/ metrics
    are."""

    def logged_keys(*, failing: bool) -> list[set[str]]:
        if failing:

            def settle(self, record):
                raise RuntimeError("a bug in the statistics")

            monkeypatch.setattr(TaskStatsStore, "settle", settle)
        (tmp_path / str(failing)).mkdir()
        trainer, _saves, logged = make_trainer(trainer_config(tmp_path / str(failing)))
        asyncio.run(trainer.train(train_workflow=Workflow(delay=0.01)))
        return [set(stats) for stats in logged.stats]

    assert all("train/batch/informative_groups" in keys for keys in logged_keys(failing=False))
    steps = logged_keys(failing=True)
    assert steps and not any(
        key.startswith(("train/batch/informative", "train/tasks/")) for keys in steps for key in keys
    )


class _OverClaiming(Workflow):
    """Each group trains a datum that its report says it never merged."""

    async def arun_episode(self, data):
        datums = await super().arun_episode(data)
        return _TaskRolloutOutput(
            datums,
            workload=RolloutWorkload(),  # no post-merge datum
            requested_rollouts=1,
            observed_rollouts=1,
            trainable_rollouts=1,
            task_retained_datums=1,
        )


def test_a_step_whose_workload_check_stops_the_run_commits_no_fate_to_the_statistics(tmp_path):
    """The check of the step's workload metrics stops the run before the step commits: its groups settled, and no fate
    of theirs was recorded."""

    trainer, _saves, _logged = make_trainer(trainer_config(tmp_path))

    with pytest.raises(BaseException) as raised:
        asyncio.run(trainer.train(train_workflow=_OverClaiming(delay=0.01)))
    error = raised.value
    while isinstance(error, BaseExceptionGroup):
        error = error.exceptions[0]
    assert isinstance(error, ValueError) and "exceed the consumed post-merge datum count" in str(error)

    events = [event["event"] for event in read_events(trainer.run_log_path)]
    assert "settle" in events and "fate" not in events


def test_a_custom_workflow_s_plain_datums_give_no_evidence(tmp_path):
    """A workflow class of a project's own returns datums alone: its groups count, but no task reads as seen or as
    without evidence, and no pass@k is logged from the prior alone."""

    trainer, _saves, logged = make_trainer(
        trainer_config(tmp_path, "train.max_training_steps=6", "checkpoint.recovery_every=100"), tasks=2
    )

    asyncio.run(trainer.train(train_workflow=Workflow(delay=0.01)))

    last = logged.stats[-1]
    assert (last["train/tasks/no_evidence"], last["train/tasks/coverage"]) == (0.0, 0.0)
    assert not [key for key in last if key.endswith(("pass_at_1", "pass_at_G"))]
    store = trainer._recorder.store
    assert all(store.task(task_id).groups and not store.task(task_id).reported_groups for task_id in store.task_ids())


def test_lock_json_lists_each_start_that_lost_the_statistics(tmp_path):
    (tmp_path / "lock.json").write_text("{}\n")
    first = {"start": 2, "checkpoint": "final", "problem": "p", "loaded": None}
    second = {"start": 3, "checkpoint": "recovery-000004", "problem": "q", "loaded": "recovery-000002"}

    task_records.record_lost_state(tmp_path, first)
    task_records.record_lost_state(tmp_path, second)

    assert json.loads((tmp_path / "lock.json").read_text())[LOST_KEY] == [first, second]
    broken = tmp_path / "broken"
    (broken / "lock.json").mkdir(parents=True)  # unreadable: never fails training
    task_records.record_lost_state(broken, first)


def test_a_kept_group_without_a_record_gets_its_fate_by_its_draw_and_older_outstanding_draws_are_lost(tmp_path):
    """A group that a start whose statistics had stopped kept has no record: its fate goes by its draw.  And a resume
    whose statistics came from an older snapshot counts as lost the outstanding draws the checkpoint no longer holds,
    but not those it does, which it takes up."""

    config = make_config(train={"workflow_config": {"group_size": 1}})
    tasks = [{"task_id": "t0"}, {"task_id": "t1"}, {"task_id": "t2"}]
    recorder = task_records.TaskRecorder(run_dir=str(tmp_path), config=config, tasks=tasks, start=2)
    for seq, task in enumerate(tasks):
        recorder.store.draw(1, seq, task["task_id"], seq)  # as an older snapshot holds them

    assert recorder.lose_outstanding(keep=[(1, 1), (1, 2)]) == 1  # (1, 0) was consumed since: the statistics lost it
    recorder.fate_of((1, 1), "consumed")

    assert [draw[:2] for draw in recorder.store.outstanding] == [(1, 2)]
    assert recorder.store.task("t0").fates == {"lost": 1} and recorder.store.task("t1").fates == {"consumed": 1}
    recorder.close()
