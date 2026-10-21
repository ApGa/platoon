"""``platoon tasks show`` and ``platoon tasks export`` read a training run's per-task statistics from the snapshot its
newest full checkpoint names (``platoon.train.task_stats``)."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from platoon import cli
from platoon.train.components import ENVIRONMENT_COLUMN
from platoon.train.task_stats import (
    TASK_COLUMNS,
    EntryInfo,
    GroupRecord,
    MemberResult,
    TaskStatsStore,
    snapshot_name,
    write_snapshot,
)

FIXTURE = Path(__file__).parent / "fixtures" / "task_stats"


def _member(scored: float | None, version: int) -> MemberResult:
    valid = scored is not None
    return MemberResult(
        reward=scored,
        scored=scored,
        valid=valid,
        outcome="scored" if valid else "timeout",
        disposition="completed",
        reward_key="reward" if valid else None,
        steps=5,
        output_tokens=100,
        seconds=30.0,
        versions={"main": (version, version)} if valid else {},
    )


def _record(task_id: str, seq: int, scores: list[float | None], verdict: str, version: int) -> GroupRecord:
    entry = task_id.partition("::")[0]
    return GroupRecord(
        start=1,
        seq=seq,
        task_id=task_id,
        entry=entry,
        labels={ENVIRONMENT_COLUMN: entry},
        harness=None,
        group_size=len(scores),
        sampler="auto",
        reason="order",
        score=None,
        position=seq,
        drawn_at={"main": version},
        members=tuple(_member(score, version) for score in scores),
        verdict=verdict,
    )


def make_run_dir(root: Path) -> Path:
    """A run directory whose newest full checkpoint, recovery-000004 at policy version 4, names a snapshot of
    statistics over four tasks of two entries: groups of 4 and of 2, the second passing at 0.5."""

    store = TaskStatsStore(
        [
            ("quick::easy", "quick", "quick"),
            ("quick::hard", "quick", "quick"),
            ("quick::unseen", "quick", "quick"),
            ("slow::partial", "slow", "slow"),
        ],
        {"quick": EntryInfo(group_size=4), "slow": EntryInfo(group_size=2, success_threshold=0.5)},
        column=ENVIRONMENT_COLUMN,
    )
    groups = [
        ("quick::easy", [1.0, 1.0, 1.0, 1.0], "uniform", 0, "consumed"),
        ("quick::hard", [1.0, 0.0, 0.0, 0.0], "trained", 0, "consumed"),
        ("slow::partial", [0.6, 0.2], "trained", 1, "stale"),
        ("quick::easy", [1.0, 0.0, 1.0, 1.0], "trained", 2, "consumed"),
        ("quick::hard", [0.0, 0.0, None, 0.0], "uniform", 3, "consumed"),
        ("slow::partial", [None, None], "no_results", 3, None),
    ]
    for seq, (task_id, _scores, _verdict, _version, _fate) in enumerate(groups):
        store.draw(1, seq, task_id, position=seq)
    for seq, (task_id, scores, verdict, version, fate) in enumerate(groups):
        store.settle(_record(task_id, seq, scores, verdict, version))
        if fate is not None:
            store.fate(1, seq, fate)
    run_dir = root / "run"
    record = write_snapshot(run_dir, snapshot_name(1, "recovery-000004"), store.snapshot(start=1, next_seq=len(groups)))
    rows = [
        {"name": "recovery-000000", "batch": 0, "state_path": "tinker://run-1/weights/recovery-000000"},
        {
            "name": "recovery-000004",
            "batch": 4,
            "policy_version": 4,
            "state_path": "tinker://run-1/weights/recovery-000004",
            "tasks": record,
        },
    ]
    (run_dir / "checkpoints.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return run_dir


def _show(run_dir: Path, capsys, *options: str) -> str:
    cli.main(["tasks", "show", str(run_dir), *options])
    return capsys.readouterr().out.replace(str(run_dir), "RUN")


def test_tasks_show_matches_the_golden_output(tmp_path, capsys):
    run_dir = make_run_dir(tmp_path)

    assert _show(run_dir, capsys) == (FIXTURE / "show.txt").read_text()


def test_tasks_show_sorts_filters_and_shortens(tmp_path, capsys):
    run_dir = make_run_dir(tmp_path)

    hardest = _show(run_dir, capsys, "--sort", "pass_rate", "--top", "1").splitlines()
    assert len(hardest) == 3 and hardest[2].startswith("quick::hard ")
    slow = _show(run_dir, capsys, "--label", "slow").splitlines()
    assert (
        slow[0] == "RUN: checkpoint recovery-000004, policy version 4: 1 of 1 tasks of label slow have a settled group"
    )
    assert [line.split()[0] for line in slow[1:]] == ["task", "slow::partial"]


def test_tasks_export_writes_every_task(tmp_path, capsys):
    run_dir = make_run_dir(tmp_path)

    cli.main(["tasks", "export", str(run_dir)])
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert list(rows[0]) == list(TASK_COLUMNS)
    assert [(row["task_id"], row["groups"], row["passes"]) for row in rows] == [
        ("quick::easy", "2", "7"),
        ("quick::hard", "2", "1"),
        ("quick::unseen", "0", "0"),
        ("slow::partial", "2", "1"),
    ]
    cli.main(["tasks", "export", str(run_dir), "--format", "jsonl"])
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [record["fate_stale"] for record in records] == [0, 0, 0, 1]
    assert records[3]["no_evidence_streak"] == 1 and records[3]["outstanding"] == 1
    # The sixth fate, rejected (a group admission rule's), comes after lost: the columns after it move one place.
    columns = list(TASK_COLUMNS)
    assert columns[columns.index("fate_lost") + 1 :][:2] == ["fate_rejected", "group_size"]
    assert [record["fate_rejected"] for record in records] == [0, 0, 0, 0]


def test_tasks_show_says_why_a_run_has_no_statistics(tmp_path, capsys):
    run_dir = make_run_dir(tmp_path)
    (run_dir / "tasks" / "state-s1-recovery-000004.json").write_text("{}")

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["tasks", "show", str(run_dir)])
    assert exit_info.value.code == 1
    assert "does not have the SHA-256 its checkpoint recorded" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        cli.main(["tasks", "show", str(tmp_path)])
    assert "has no checkpoint that saved per-task statistics" in capsys.readouterr().err


def test_tasks_show_falls_back_to_an_older_snapshot_and_says_why(tmp_path, capsys):
    """The newest checkpoint names no snapshot (its write failed), and the one before names a file pruning deleted:
    the newest snapshot that reads back, a permanent checkpoint's, is shown, and stderr says what was passed over."""

    run_dir = make_run_dir(tmp_path)
    rows = [json.loads(line) for line in (run_dir / "checkpoints.jsonl").read_text().splitlines()]
    (run_dir / rows[1]["tasks"]["file"]).rename(run_dir / "tasks" / "kept.json")
    permanent = {**rows[1], "name": "000004", "tasks": {**rows[1]["tasks"], "file": "tasks/kept.json"}}
    newest = {"name": "recovery-000005", "batch": 5, "policy_version": 5, "state_path": "tinker://run-1/weights/r5"}
    rows = [rows[0], permanent, rows[1], newest]
    (run_dir / "checkpoints.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    cli.main(["tasks", "show", str(run_dir)])

    captured = capsys.readouterr()
    assert captured.out.startswith(f"{run_dir}: checkpoint 000004, policy version 4: 3 of 4 tasks")
    assert "platoon: passed over checkpoint recovery-000004: " in captured.err and "cannot be read" in captured.err
    assert "recovery-000005" not in captured.err  # a row that names no snapshot is no snapshot to pass over


def test_tasks_show_prints_no_negative_zero_for_groups_of_one(tmp_path, capsys):
    store = TaskStatsStore([("one::a", "one", "one")], {"one": EntryInfo(group_size=1)}, column=ENVIRONMENT_COLUMN)
    printed = set()
    for seq in range(30):
        store.settle(_record("one::a", seq, [1.0 if seq % 3 else 0.0], "uniform", seq))
        record = write_snapshot(tmp_path, snapshot_name(1, f"recovery-{seq:06d}"), store.snapshot(start=1, next_seq=0))
        row = {"name": f"recovery-{seq:06d}", "policy_version": seq, "state_path": "tinker://r/w", "tasks": record}
        (tmp_path / "checkpoints.jsonl").write_text(json.dumps(row) + "\n")
        cli.main(["tasks", "show", str(tmp_path)])
        printed.add(capsys.readouterr().out.splitlines()[2].split()[5])
    assert printed == {"0.00"}


def test_tasks_export_names_the_group_kinds_as_the_step_metrics_do(tmp_path, capsys):
    run_dir = make_run_dir(tmp_path)

    cli.main(["tasks", "export", str(run_dir), "--format", "jsonl"])

    records = {record["task_id"]: record for record in map(json.loads, capsys.readouterr().out.splitlines())}
    easy = records["quick::easy"]
    assert (easy["informative_groups"], easy["all_pass_groups"], easy["all_fail_groups"]) == (1, 1, 0)
    assert records["quick::hard"]["all_fail_groups"] == 1
    assert [records[task]["reported_groups"] for task in records] == [2, 2, 0, 2]
