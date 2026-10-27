"""The commits of a run whose policies step independently (``platoon.train.checkpoints.commits``), on their own, over
rows written by hand: a resume takes the commit with the highest ``commit_seq`` in any policy's file whose named rows
hold (a row its policy's pruner deleted is missing, and so is a state path two rows of a file hold, one saved over
the other; a newer commit that does not hold is skipped and recorded), whatever order the commits were appended in; a
run whose rows hold no commit was saved in lockstep, and no start that steps independently resumes it; numbers go on
from the highest in any file; a policy resumes at its row's version, or at one more than a later version the commit
records it published; a commit's named rows stay pinned while it is in flight and once it is its file's newest; a
commit whose writing fails closes the writings after it, unrun, and pins nothing; and the share ledger takes the
commit lock that a commit waits on for the step hooks."""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from platoon.train.checkpoints.barrier import Counters
from platoon.train.checkpoints.commits import Commits, newest_commit, next_commit_seq, resumed_version
from platoon.train.checkpoints.resume import StartPoint
from platoon.utils.config import ConfigError

POLICIES = ("main", "worker")


def _append(directory: Path, row: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "checkpoints.jsonl").open("a") as file:
        file.write(json.dumps(row) + "\n")


def _row(
    policy: str, name: str, *, seq: int | None, batch: int, step: int, version: int, names: dict[str, Any]
) -> dict[str, Any]:
    """A row of ``policy``'s file named ``name``, a commit numbered ``seq`` (None: a barrier's plain row) whose
    ``policies`` names ``names`` (its own entry added)."""

    row: dict[str, Any] = {
        "name": name,
        "batch": batch,
        "policy_version": version,
        "optimizer_steps": version,
        "state_path": f"tinker://{policy}/weights/{name}",
        "sampler_path": f"tinker://{policy}/sampler_weights/{name}",
    }
    if seq is not None:
        own = {"name": name, "policy_version": version, "published_version": version, "optimizer_steps": version}
        row["commit_seq"] = seq
        row["policies"] = {**names, policy: {**own, "step": step}}
    return row


def _named(policy: str, name: str, *, step: int, version: int, published: int | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "state_path": f"tinker://{policy}/weights/{name}",
        "policy_version": version,
        "published_version": version if published is None else published,
        "optimizer_steps": version,
        "step": step,
    }


def _worker_dir(run: Path) -> Path:
    return run / "policies" / "worker"


def _launch(run: Path) -> None:
    """The launch barrier: the worker's plain row, then main's commit 0 naming it."""

    _append(_worker_dir(run), _row("worker", "recovery-000000", seq=None, batch=0, step=0, version=0, names={}))
    worker = _named("worker", "recovery-000000", step=0, version=0)
    _append(run, _row("main", "recovery-000000", seq=0, batch=0, step=0, version=0, names={"worker": worker}))


def test_a_resume_takes_the_highest_commit_seq_whatever_order_they_were_appended_in(tmp_path):
    _launch(tmp_path)
    main_row = _named("main", "recovery-000000", step=0, version=0)
    # The worker's commit 2 is appended before its commit 1 (captured first, its save slower): the order is the seq's.
    two = _row("worker", "recovery-000004", seq=2, batch=0, step=4, version=4, names={"main": main_row})
    one = _row("worker", "recovery-000002", seq=1, batch=0, step=2, version=2, names={"main": main_row})
    _append(_worker_dir(tmp_path), two)
    _append(_worker_dir(tmp_path), one)

    commit = newest_commit(str(tmp_path), POLICIES)
    assert commit is not None and (commit.seq, commit.policy, commit.record.name) == (2, "worker", "recovery-000004")
    assert commit.rows["main"].name == "recovery-000000"
    assert next_commit_seq(str(tmp_path), POLICIES) == 3

    start = StartPoint.read(str(tmp_path), 1, leader="main", others=("worker",), independent=True)
    assert (start.step, start.record.name if start.record else None) == (0, "recovery-000004")
    assert {policy: (begins.step, begins.version) for policy, begins in start.policies.items()} == {
        "main": (0, 0),
        "worker": (4, 4),
    }


def test_a_commit_that_names_a_deleted_row_is_skipped_and_recorded(tmp_path):
    (tmp_path / "lock.json").write_text("{}\n")  # what the command that started the run wrote
    _launch(tmp_path)
    old_worker = _named("worker", "recovery-000000", step=0, version=0)
    _append(tmp_path, _row("main", "recovery-000001", seq=1, batch=1, step=1, version=1, names={"worker": old_worker}))
    # The worker's pruner deleted the row commit 1 names.
    (tmp_path / "policies" / "worker" / "checkpoints_deleted.jsonl").write_text(
        json.dumps({"path": "tinker://worker/weights/recovery-000000"}) + "\n"
    )
    with pytest.raises(ConfigError, match="no longer has the row it names"):
        newest_commit(str(tmp_path), POLICIES)

    worker_row = _row("worker", "recovery-000002", seq=None, batch=1, step=2, version=2, names={})
    _append(_worker_dir(tmp_path), worker_row)
    named = _named("worker", "recovery-000002", step=2, version=2)
    _append(tmp_path, _row("main", "recovery-000002", seq=2, batch=2, step=2, version=2, names={"worker": named}))
    _append(tmp_path, _row("main", "recovery-000003", seq=3, batch=3, step=3, version=3, names={"worker": old_worker}))
    commit = newest_commit(str(tmp_path), POLICIES)
    assert commit is not None and commit.seq == 2
    lock = json.loads((tmp_path / "lock.json").read_text())
    [restore] = lock["policies_restore"]
    assert restore["skipped"][0].startswith("commit 3 (policy main's checkpoint recovery-000003): policy worker no")


def test_a_policy_resumes_after_the_versions_a_commit_records_it_published():
    assert resumed_version({"published_version": 214}, 210) == 215
    assert resumed_version({"published_version": 210}, 210) == 210
    assert resumed_version({}, 210) == 210


def test_a_fresh_run_has_no_commit_and_rows_without_one_cannot_resume(tmp_path):
    assert newest_commit(str(tmp_path), POLICIES) is None
    assert next_commit_seq(str(tmp_path), POLICIES) == 0
    _append(tmp_path, _row("main", "recovery-000000", seq=None, batch=0, step=0, version=0, names={}))
    with pytest.raises(ConfigError, match=r"stepping in lockstep \(no row holds a commit_seq\), and this start steps"):
        newest_commit(str(tmp_path), POLICIES)


def test_a_commit_that_names_a_state_path_two_rows_of_a_file_hold_is_skipped(tmp_path):
    """A later save under one name, in one training run, writes over an earlier row's state: no row says whose
    weights the path holds, so a commit that names it, or whose own state it is, does not hold."""

    _launch(tmp_path)
    main_row = _named("main", "recovery-000000", step=0, version=0)
    # The worker saves again under its launch row's name, as its own commit.
    again = _row("worker", "recovery-000000", seq=1, batch=0, step=3, version=3, names={"main": main_row})
    _append(_worker_dir(tmp_path), again)
    old_worker = _named("worker", "recovery-000000", step=0, version=0)
    _append(tmp_path, _row("main", "recovery-000001", seq=2, batch=1, step=1, version=1, names={"worker": old_worker}))

    with pytest.raises(ConfigError) as raised:
        newest_commit(str(tmp_path), POLICIES)
    assert "commit 2 (policy main's checkpoint recovery-000001): policy worker no longer has the row it names" in str(
        raised.value
    )
    assert "commit 1 (policy worker's checkpoint recovery-000000): its state" in str(raised.value)


class _Saving:
    """A policy whose save waits until the test lets it, then appends its row as tinker-cookbook does."""

    def __init__(self, policy: str, run: Path) -> None:
        self.id, self.run, self.version, self.optimizer_steps = policy, run, 0, 0
        self.go = asyncio.Event()

    async def save(self, name: str, *, log_path: str, loop_state: dict[str, Any], wait: Any) -> dict[str, str]:
        await self.go.wait()
        paths = {"state_path": f"tinker://{self.id}/weights/{name}", "sampler_path": f"tinker://{self.id}/s/{name}"}
        _append(Path(log_path), {"name": name, **loop_state, **paths})
        return paths


def test_a_commit_pins_the_rows_it_names_while_it_is_in_flight_and_once_it_is_its_file_s_newest(tmp_path):
    _launch(tmp_path)
    saving = {policy: _Saving(policy, tmp_path) for policy in POLICIES}
    policies: Any = _PolicySet(saving)
    commits = Commits(str(tmp_path), policies, participants=[], wandb_run_id=lambda: None)
    commits.newest = {
        "main": {**_named("main", "recovery-000000", step=0, version=0)},
        "worker": {**_named("worker", "recovery-000000", step=0, version=0)},
    }
    assert commits.next_seq == 1
    assert commits.pinned("worker") == {"tinker://worker/weights/recovery-000000"}  # main's newest commit names it

    async def run():
        commit = asyncio.ensure_future(
            commits.commit("worker", "recovery-000002", step=2, counters=Counters(2, 2), wait=_unwaited)
        )
        await asyncio.sleep(0)
        # In flight: it pins main's row it named; the worker's own row is not written yet.
        assert commits.pinned("main") == {"tinker://main/weights/recovery-000000"}
        saving["worker"].go.set()
        await commit
        assert commits.pinned("worker") == {
            "tinker://worker/weights/recovery-000000",  # main's newest commit still names it
            "tinker://worker/weights/recovery-000002",
        }
        assert commits.newest["worker"]["step"] == 2

    asyncio.run(run())
    [row] = [row for row in _rows(_worker_dir(tmp_path)) if row.get("commit_seq") == 1]
    assert row["batch"] == 0 and row["policies"]["main"]["state_path"] == "tinker://main/weights/recovery-000000"
    assert row["policies"]["worker"] == {
        "name": "recovery-000002",
        "policy_version": 2,
        "published_version": 2,
        "optimizer_steps": 2,
        "step": 2,
    }
    assert row["policies"]["main"]["published_version"] == 0


class _Writing:
    """A participant whose capture gives a writing (a coroutine) that fails (``fails``), or records that it ran."""

    keys = ("writing",)

    def __init__(self, *, fails: bool) -> None:
        self.fails, self.ran = fails, False
        self.writings: list[Any] = []

    def capture(self, name: str) -> Any:
        async def write() -> dict[str, Any]:
            if self.fails:
                raise OSError("no space left on the device")
            self.ran = True
            return {}

        self.writings.append(write())
        return self.writings[-1]


def test_a_commit_whose_writing_fails_closes_the_writings_after_it_unrun_and_unpins(tmp_path):
    _launch(tmp_path)
    saving = {policy: _Saving(policy, tmp_path) for policy in POLICIES}
    failing, after = _Writing(fails=True), _Writing(fails=False)
    policies: Any = _PolicySet(saving)
    commits = Commits(str(tmp_path), policies, participants=[failing, after], wandb_run_id=lambda: None)
    commits.newest = {policy: _named(policy, "recovery-000000", step=0, version=0) for policy in POLICIES}
    commits._newest_names = {}  # no newest commit pins a row: what a commit pins in flight shows alone

    async def run() -> None:
        with pytest.raises(OSError, match="no space left"):
            await commits.commit("worker", "recovery-000002", step=2, counters=Counters(2, 2), wait=_unwaited)

    asyncio.run(run())
    [writing] = after.writings
    assert not after.ran and inspect.getcoroutinestate(writing) == inspect.CORO_CLOSED  # never awaited, never warns
    assert commits.pinned("main") == set()  # the failed commit pins nothing any more
    assert not (_worker_dir(tmp_path) / "checkpoints.jsonl").read_text().count("recovery-000002")


def test_the_share_ledger_takes_the_commit_lock_that_a_commit_waits_on_for_the_hooks(tmp_path):
    """A step's end that ran its hooks before the run's draws opened made the condition a commit waits on, over the
    commit lock; the share ledger, attached once the draws open, takes that lock, so that its settle sections and the
    steps' consumption exclude a commit's capture."""

    policies: Any = _PolicySet({policy: _Saving(policy, tmp_path) for policy in POLICIES})
    commits = Commits(str(tmp_path), policies, participants=[], wandb_run_id=lambda: None)

    async def run() -> None:
        async with commits.hook_phase():
            pass
        ledger: Any = SimpleNamespace(lock=asyncio.Lock())
        commits.attach(ledger)
        async with ledger.lock:  # a group's settle section, or a step's consumption
            assert ledger.lock is commits.lock and commits._condition().locked()

    asyncio.run(run())


async def _unwaited(awaitable: Any, **kwargs: Any) -> Any:
    return await awaitable


class _PolicySet:
    def __init__(self, policies: dict[str, Any]) -> None:
        self._policies, self.leader = policies, "main"

    def trainable(self) -> list[Any]:
        return list(self._policies.values())

    def __getitem__(self, policy: str) -> Any:
        return self._policies[policy]


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (path / "checkpoints.jsonl").read_text().splitlines()]
