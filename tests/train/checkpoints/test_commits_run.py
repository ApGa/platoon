"""Commits of two policies that step independently, whole starts on CPU on two fake Tinker services
(``fakes.independent``), each killed where a test says and resumed: a kill at every point between a step's
consumption of its shares, its commit's capture, its writing, its save and its row's append leaves each share trained
once in each policy's weights, also of a worker slower than main, whose own commits never save over its launch row; a
commit that names another policy's row four of its steps old puts back every share that policy consumed since, which
trains again, none dropped as stale; a kill after main's last commit, with the worker behind, resumes to a final row
of both; a pruner that runs while a commit is in flight keeps the row it names, and a resume skips a commit whose
named row was deleted; commits appended out of their ``commit_seq`` order resume from the highest, and the next
commit takes the number after the highest; and a stateful credit estimator killed between a group's credit and its
settling counts each group once."""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fakes.independent import (
    Lineages,
    Services,
    TaggedGroups,
    final_lineage,
    resumed_from,
    rows,
    run_dir,
    shares_put_back,
    start,
)
from tinker_cookbook import checkpoint_utils

from platoon.testing.trainers import FakePolicyService
from platoon.train.checkpoints import commits as commits_module
from platoon.train.dispatch import runner
from platoon.train.dispatch.shares import Shares
from platoon.train.policy.policy import Policy
from platoon.train.policy.spec import MAIN


class Killed(BaseException):
    """The process died: nothing after it runs, no exit save."""


def _trained_once(lineages: Lineages, root: Path, put_back: dict[str, list[int]]) -> None:
    """Each policy's weights at the run's end trained each share once, and every share the resume put back among
    them, first."""

    run = run_dir(root)
    for policy, path in ((MAIN, run), ("worker", run / "policies" / "worker")):
        trained = final_lineage(lineages, path)
        assert all(count == 1 for count in Counter(trained).values()), path
        assert set(put_back[policy]) <= set(trained), path


def _kill_after(target: Any, name: str, monkeypatch: Any, *, when: Any, before: bool = False) -> None:
    """Make ``target.name`` kill the process once ``when(*args, **kwargs)`` holds: right after the original runs,
    or right before it (``before``)."""

    original = getattr(target, name)

    def killing(*args: Any, **kwargs: Any) -> Any:
        if before and when(*args, **kwargs):
            raise Killed()
        result = original(*args, **kwargs)
        if not before and when(*args, **kwargs):
            raise Killed()
        return result

    async def killing_async(*args: Any, **kwargs: Any) -> Any:
        if before and when(*args, **kwargs):
            raise Killed()
        result = await original(*args, **kwargs)
        if not before and when(*args, **kwargs):
            raise Killed()
        return result

    monkeypatch.setattr(target, name, killing_async if inspect.iscoroutinefunction(original) else killing)


def _crash(root: Path, lineages: Lineages, **sections: Any) -> None:
    """A start of 3 steps, which a kill ends."""

    with pytest.raises(BaseException) as raised:
        start(root, lineages, steps=3, **sections)
    assert _killed(raised.value), raised.value


def _killed(error: BaseException) -> bool:
    if isinstance(error, BaseExceptionGroup):
        return any(_killed(inner) for inner in error.exceptions)
    return isinstance(error, Killed)


# The points of a commit of ``policy`` (its step ``step``, counted from 0) a kill comes at.
POINTS = ["consume", "capture", "write", "save", "append"]


def _arm(point: str, policy: str, monkeypatch: Any, *, steps: dict[str, int] | None = None) -> None:
    """Kill the process at ``point`` of ``policy``'s second commit after the launch (main's after its step 1, the
    worker's after its step 3; ``steps``: the step whose consumption ``consume`` kills after, by policy).  Its writing
    and its save count the launch's save as the first."""

    seen: Counter[str] = Counter()

    def nth(of: str) -> bool:
        if of != policy:
            return False
        seen[of] += 1
        return seen[of] == 2

    if point == "consume":  # the step consumed its shares; its commit has not captured
        last = steps or {MAIN: 1, "worker": 3}

        def consumed(ledger: Any, of: str, keys: Any, *, step: int, fates: Any = None) -> bool:
            return of == policy and step == last[policy]

        _kill_after(Shares, "consumed", monkeypatch, when=consumed)
    elif point == "capture":  # captured under the lock; nothing written
        capture = commits_module.Commits._capture

        async def captured_then_die(self: Any, of: str, label: str) -> Any:
            found = await capture(self, of, label)
            if nth(of):
                for writing in found[1]:  # the writing the process never got to
                    if inspect.iscoroutine(writing):
                        writing.close()
                raise Killed()
            return found

        monkeypatch.setattr(commits_module.Commits, "_capture", captured_then_die)
    elif point == "write":  # the participants' files written; the Tinker save not begun
        _kill_after(Policy, "save", monkeypatch, when=lambda self, name, **kwargs: nth(self.id), before=True)
    elif point == "save":  # the Tinker save done; the row not appended
        original = checkpoint_utils.save_checkpoint_async

        async def save(*, training_client: Any, name: str, log_path: str, loop_state: Any, **kwargs: Any) -> Any:
            if training_client.run.startswith(policy) and nth(policy):
                await training_client.save_state_async(name)
                raise Killed()
            return await original(
                training_client=training_client, name=name, log_path=log_path, loop_state=loop_state, **kwargs
            )

        monkeypatch.setattr(checkpoint_utils, "save_checkpoint_async", save)
    else:  # the row appended; its pins and durable shares not yet updated
        _kill_after(
            commits_module.Commits, "_appended", monkeypatch, when=lambda self, of, own, others: nth(of), before=True
        )


@pytest.mark.parametrize("policy", [MAIN, "worker"])
@pytest.mark.parametrize("point", POINTS)
def test_a_kill_at_any_point_of_a_commit_leaves_each_share_trained_once_per_lineage(
    tmp_path, monkeypatch, point, policy
):
    lineages = Lineages()
    with monkeypatch.context() as armed:
        _arm(point, policy, armed)
        _crash(tmp_path, lineages)
    put_back = shares_put_back(tmp_path, resumed_from(tmp_path))
    start(tmp_path, lineages, steps=3)
    _trained_once(lineages, tmp_path, put_back)


# A worker slower than main: main steps on 1 share and commits at each of its steps, the worker on 4, committing at
# every third of its own steps.
SLOW_WORKER: dict[str, Any] = {
    "train": {"batch_size": 1, "max_staleness": 3},
    "policies": {"worker": {"batch_size": 4, "max_staleness": 7, "checkpoint": {"recovery_every": 3}}},
}


@pytest.mark.parametrize("point", POINTS)
def test_a_slow_worker_killed_at_any_point_of_a_commit_after_a_resume_trains_each_share_once(
    tmp_path, monkeypatch, point
):
    """The first start ends with main at step 4 and the worker at its step 1: the second start's launch saves the
    worker's row under main's step's name (recovery-000004), in the session the worker's own commits save in, after
    its steps 3 and 6.  Those are named with its id, so none saves over the launch row, whose state the commits of
    main name until then; the second start dies at a point of one of them, and the resume trains each share once."""

    lineages = Lineages()
    start(tmp_path, lineages, steps=4, **SLOW_WORKER)
    with monkeypatch.context() as armed:
        _arm(point, "worker", armed, steps={"worker": 3})
        with pytest.raises(BaseException) as raised:
            start(tmp_path, lineages, steps=28, **SLOW_WORKER)
        assert _killed(raised.value), raised.value
    put_back = shares_put_back(tmp_path, resumed_from(tmp_path))
    start(tmp_path, lineages, steps=28, **SLOW_WORKER)
    _trained_once(lineages, tmp_path, put_back)
    worker_rows = rows(run_dir(tmp_path) / "policies" / "worker")
    paths = Counter(row["state_path"] for row in worker_rows)
    assert all(count == 1 for count in paths.values()), paths  # no row of the worker saves over another


def test_a_commit_that_names_another_policy_s_row_four_steps_old_puts_back_every_share_and_drops_none(
    tmp_path, monkeypatch, caplog
):
    """Main commits at each of its steps; the worker saves no row past its launch's.  Main's commit after the worker
    has stepped four times past that row names it, and the process dies right after it: the worker resumes from the
    row, at one version more than it published, and every share the commit records it consumed goes back and trains
    again first, judged at the age it had (a bound of 2: judged against the resumed version, the oldest would be
    stale).  The shares it had not consumed by the commit are judged against its version as they come."""

    lineages = Lineages()
    # The worker saves its launch row alone; its horizon (3 shares) keeps it within 3 groups of the dispatch, so that
    # main's commit after its step 2 (8 groups settled) captures it at least 4 published versions past that row.
    worker = {"max_staleness": 2, "checkpoint": {"recovery_every": 100}}

    def four_past(self: Any, of: str, own: Any, others: Any) -> bool:  # as the commit captured the worker
        return of == MAIN and others["worker"]["published_version"] - others["worker"]["policy_version"] >= 4

    with monkeypatch.context() as armed:
        _kill_after(commits_module.Commits, "_appended", armed, when=four_past)
        _crash(tmp_path, lineages, policies={"worker": worker})
    commit = resumed_from(tmp_path)
    named = commit.entries["worker"]
    assert commit.policy == MAIN and named["published_version"] - named["policy_version"] >= 4
    put_back = shares_put_back(tmp_path, commit)
    assert len(put_back["worker"]) >= 4
    with caplog.at_level("INFO"):
        start(tmp_path, lineages, steps=3, policies={"worker": worker})
    _trained_once(lineages, tmp_path, put_back)
    # Every share the worker had consumed by the commit trains again first, in order, none dropped as stale.
    trained = final_lineage(lineages, run_dir(tmp_path) / "policies" / "worker")
    assert trained[: len(put_back["worker"])] == put_back["worker"]


def test_a_kill_after_main_s_last_commit_with_the_worker_behind_resumes_to_a_final_row_of_both(tmp_path, monkeypatch):
    """The worker's optimizer steps are slow: main commits after its last step while the worker still has shares to
    train, and the process dies right after that commit is appended.  The resume starts with main at the plan's end:
    the worker trains the shares the commit puts back and those it had not taken, and the start saves the final row of
    both (it would otherwise save nothing, so that every later start trained them again).  A start after it trains
    nothing."""

    lineages = Lineages()
    services = Services(lineages)
    services.worker.optim_delay = 0.05

    def last(self: Any, of: str, own: Any, others: Any) -> bool:  # main's commit after its step 3, the plan's last
        return of == MAIN and own["step"] == 3

    with monkeypatch.context() as armed:
        _kill_after(commits_module.Commits, "_appended", armed, when=last, before=True)
        with pytest.raises(BaseException) as raised:
            start(tmp_path, lineages, steps=3, services=services)
        assert _killed(raised.value), raised.value
    commit = resumed_from(tmp_path)
    assert commit.policy == MAIN and commit.entries[MAIN]["step"] == 3 and commit.entries["worker"]["step"] < 12
    put_back = shares_put_back(tmp_path, commit)

    start(tmp_path, lineages, steps=3)
    run = run_dir(tmp_path)
    final = rows(run)[-1]
    assert final["name"] == "final" and final["policies"]["worker"]["step"] == 12
    _trained_once(lineages, tmp_path, put_back)
    for path in (run, run / "policies" / "worker"):
        assert len(final_lineage(lineages, path)) == 12  # every share of the plan's 12 groups
    again, _ = start(tmp_path, lineages, steps=3)
    assert [client.optimizer_steps for client in again.worker.training_clients] == [0]


def _deleting(service: FakePolicyService) -> Any:
    async def delete(path: str) -> None:
        await asyncio.sleep(0)

    return SimpleNamespace(delete_checkpoint_from_tinker_path_async=delete)


def _deleted(path: Path) -> set[str]:
    file = path / "checkpoints_deleted.jsonl"
    return {json.loads(line)["path"] for line in file.read_text().splitlines()} if file.exists() else set()


def test_a_pruner_running_while_a_commit_is_in_flight_keeps_the_row_it_names_and_a_resume_skips_a_deleted_one(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(FakePolicyService, "create_rest_client", _deleting, raising=False)
    lineages = Lineages()
    services = Services(lineages)
    let_go = asyncio.Event()
    held: list[str] = []

    async def hold(name: str) -> None:  # main's commit after its step 1 waits for the worker's later commits
        if name == "recovery-000002":
            held.append(name)
            await let_go.wait()

    original = commits_module.Commits._appended
    worker_dir = run_dir(tmp_path) / "policies" / "worker"
    at_append: dict[str, Any] = {}

    def appended(self: Any, of: str, own: Any, others: Any) -> None:
        if of == MAIN and own["name"] == "recovery-000002":  # the held commit appends: what was deleted meanwhile
            at_append.update(named=others["worker"]["state_path"], deleted=_deleted(worker_dir))
        original(self, of, own, others)
        if of == "worker" and held and own["step"] >= 8:
            let_go.set()

    services.main.hold = hold
    monkeypatch.setattr(commits_module.Commits, "_appended", appended)
    sections: dict[str, Any] = {
        "policies": {"worker": {"max_staleness": 1, "checkpoint": {"recovery_every": 1}}},
        "checkpoint": {"recovery_every": 1, "keep_recovery": 2},
    }
    start(tmp_path, lineages, steps=3, services=services, **sections)

    run = run_dir(tmp_path)
    [held_commit] = [row for row in rows(run) if row["name"] == "recovery-000002"]
    assert held_commit["policies"]["worker"]["state_path"] == at_append["named"]
    # While main's commit was in flight, the worker's pruner deleted its older rows, never the one the commit named.
    assert at_append["deleted"] and at_append["named"] not in at_append["deleted"]

    # The worker's pruner deletes the row the newest commit names (as a crash between a pin and a delete could): a
    # resume skips that commit, records it, and goes on from an older one.
    newest = rows(run)[-1]
    with (worker_dir / "checkpoints_deleted.jsonl").open("a") as file:
        file.write(json.dumps({"path": newest["policies"]["worker"]["state_path"]}) + "\n")
    (run / "lock.json").write_text("{}\n")
    start(tmp_path, lineages, steps=4, **sections)
    restore = json.loads((run / "lock.json").read_text())["policies_restore"]
    assert restore and restore[0]["skipped"][0].startswith(f"commit {newest['commit_seq']} ")


def test_commits_appended_out_of_order_resume_from_the_highest_and_numbers_go_on_after_it(tmp_path, monkeypatch):
    """The worker's commit captures first and its save is slow; main's later commit appends first.  The process dies
    once both are appended; the resume takes the highest commit_seq, and its launch commit the next number."""

    lineages = Lineages()
    services = Services(lineages)
    let_go = asyncio.Event()

    async def hold(name: str) -> None:  # the worker's first commit, captured before any of main's
        if name == "recovery-worker-000002":
            async with asyncio.timeout(60):
                await let_go.wait()

    original, capture = commits_module.Commits._appended, commits_module.Commits._capture
    captured: dict[tuple[str, str], int] = {}

    async def numbered(self: Any, of: str, label: str) -> Any:
        found = await capture(self, of, label)
        captured[(of, label)] = found[0]
        return found

    def appended(self: Any, of: str, own: Any, others: Any) -> None:
        original(self, of, own, others)
        held = captured.get(("worker", "recovery-worker-000002"))
        if of == MAIN and held is not None and captured[(MAIN, own["name"])] > held:
            let_go.set()  # main's commit, captured after the worker's, appended first
        if of == "worker" and own["name"] == "recovery-worker-000002":
            raise Killed()

    services.worker.hold = hold
    monkeypatch.setattr(commits_module.Commits, "_capture", numbered)
    monkeypatch.setattr(commits_module.Commits, "_appended", appended)
    with pytest.raises(BaseException) as raised:
        start(tmp_path, lineages, steps=3, services=services)
    assert _killed(raised.value)

    run = run_dir(tmp_path)
    worker_rows = rows(run / "policies" / "worker")
    late = [row for row in worker_rows if row["name"] == "recovery-worker-000002"][0]
    earlier_main = [row for row in rows(run) if row.get("commit_seq", -1) > late["commit_seq"]]
    assert earlier_main, "main's commit captured after the worker's was appended before it"
    highest = max(row["commit_seq"] for row in rows(run) + worker_rows if "commit_seq" in row)

    monkeypatch.setattr(commits_module.Commits, "_appended", original)
    monkeypatch.setattr(commits_module.Commits, "_capture", capture)
    commit = resumed_from(tmp_path)
    assert commit.seq == highest
    put_back, before = shares_put_back(tmp_path, commit), len(rows(run))
    start(tmp_path, lineages, steps=3)
    launch = rows(run)[before]  # the resumed start's launch commit
    assert launch["commit_seq"] == highest + 1
    _trained_once(lineages, tmp_path, put_back)


class Counting:
    """A credit estimator with state: it counts the groups it settled (``state()``, ``load_state``)."""

    def __init__(self) -> None:
        self.settled = 0

    def settle(self, group: Any) -> None:
        self.settled += 1

    def state(self) -> dict[str, int]:
        return {"settled": self.settled}

    def load_state(self, state: dict[str, int]) -> None:
        self.settled = state["settled"]


class CreditedGroups(TaggedGroups):
    """The tagged groups, each credited by a stateful estimator, which the run settles as it files the group."""

    def __init__(self, policies: Any) -> None:
        super().__init__(policies)
        self.entry_credit = {"default": Counting()}

    async def arun_episode(self, data: dict) -> Any:
        from platoon.train.credit.runtime import Credit

        output = await super().arun_episode(data)
        group: Any = SimpleNamespace(task_id=data["task_id"], entry="default")
        return dataclasses.replace(output, credit=Credit(self.entry_credit["default"], group, ()))


def test_a_stateful_estimator_killed_between_a_group_s_credit_and_its_settling_counts_each_group_once(
    tmp_path, monkeypatch
):
    lineages = Lineages()
    settles = Counter[str]()
    original = runner.settle_credit

    def settle(output: Any) -> None:
        settles["n"] += 1
        if settles["n"] == 7:  # credited, not settled: the process dies in the section's place
            raise Killed()
        original(output)

    with monkeypatch.context() as armed:
        armed.setattr(runner, "settle_credit", settle)
        with pytest.raises(BaseException) as raised:
            start(tmp_path, lineages, steps=3, workflow=CreditedGroups)
        assert _killed(raised.value)
    put_back = shares_put_back(tmp_path, resumed_from(tmp_path))
    start(tmp_path, lineages, steps=3, workflow=CreditedGroups)
    final = rows(run_dir(tmp_path))[-1]
    # Each of the plan's 12 groups, which main's steps consumed, was settled once in the state the run's commits carry.
    assert final["credit"]["default"]["state"] == {"settled": 12}
    _trained_once(lineages, tmp_path, put_back)
