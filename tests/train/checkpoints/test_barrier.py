"""The checkpoint of a run that trains several policies in lockstep (``platoon.train.checkpoints.barrier``): every other
trained policy saves first, into its own directory, and the leader's row names their rows; a resume takes the newest
checkpoint whose rows every policy still has, and stops as a configuration error when none holds; a policy's pruner
keeps the rows that the leader rows a resume may take name; and the key of a leader row that names the other
policies' rows is the run's own, which no participant may hold and a resume claims."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pytest
from fakes.trainer import named_policy

from platoon.train.checkpoints import barrier
from platoon.train.checkpoints.barrier import Counters, named_state_paths, policy_dir
from platoon.train.checkpoints.participants import RunCheckpoints
from platoon.train.checkpoints.pruning import plan_deletions
from platoon.train.checkpoints.resume import StartPoint
from platoon.train.policy.policy import Policy
from platoon.train.policy.set import PolicySet
from platoon.train.policy.spec import MAIN, PolicySpec
from platoon.utils.config import ConfigError


def test_the_key_that_names_the_other_policies_rows_is_the_run_s_own_and_a_resume_claims_it(tmp_path):
    class Cursor:
        def __init__(self, *keys: str) -> None:
            self.keys = keys

        async def checkpoint_state(self, name: str) -> dict[str, Any]:
            return {}

        def resume(self, start: Any) -> None:
            pass

    checkpoints = RunCheckpoints(str(tmp_path), PolicySet([named_policy(MAIN)]), wandb_run_id=lambda: None)
    checkpoints.register(Cursor("tasks", "dispatch"))
    with pytest.raises(ValueError, match="loop_state key 'policies' is the run's own: '.*Cursor' may not hold it"):
        checkpoints.register(Cursor("policies"))
    assert len(checkpoints.participants) == 1
    # A leader row of several trained policies names their rows (loop_state.policies): the run claims the key, so a
    # resume from it starts.
    loop_state = {"batch": 1, "policy_version": 1, "optimizer_steps": 1, "tasks": {}, "policies": {"worker": {}}}
    start = StartPoint(record=None, step=1, policy_version=1, optimizer_steps=1, loop_state=loop_state)
    assert checkpoints.unclaimed(start) == []


def _saving(policy: Policy, saves: list[tuple[str, str, dict[str, Any]]]) -> None:
    async def save(name: str, *, log_path: str, loop_state: dict[str, Any], wait: Any) -> dict[str, str]:
        saves.append((policy.id, log_path, dict(loop_state)))
        run = f"run-{policy.id}"
        return {
            "state_path": f"tinker://{run}/weights/{name}",
            "sampler_path": f"tinker://{run}/sampler_weights/{name}",
        }

    setattr(policy, "save", save)


def test_a_checkpoint_saves_every_other_trained_policy_first_and_the_leader_s_row_names_their_rows(tmp_path):
    main, trained, frozen = named_policy(MAIN), named_policy("worker"), named_policy("judge")
    frozen.spec = PolicySpec("judge", model_name="m", renderer_name="r", trainable=False)
    trained.optimizer_steps = 4
    saves: list[tuple[str, str, dict[str, Any]]] = []
    for policy in (main, trained):
        _saving(policy, saves)
    checkpoints = RunCheckpoints(str(tmp_path), PolicySet([main, trained, frozen]), wandb_run_id=lambda: "w")
    unused: Any = None

    paths = asyncio.run(
        checkpoints.save(
            "000003", step=3, policy_version=3, optimizer_steps=6, wait=unused, others={"worker": Counters(2, 5)}
        )
    )
    assert [policy for policy, _, _ in saves] == ["worker", MAIN]
    assert saves[0][1:] == (
        policy_dir(str(tmp_path), "worker"),
        {"batch": 3, "policy_version": 2, "optimizer_steps": 5},
    )
    assert saves[1][2]["policies"] == {
        "worker": {
            "name": "000003",
            "state_path": "tinker://run-worker/weights/000003",
            "policy_version": 2,
            "optimizer_steps": 5,
        }
    }
    assert list(paths) == ["worker", MAIN]
    # Without counters, each other policy's row records its own now.
    saves.clear()
    asyncio.run(checkpoints.save("final", step=4, policy_version=4, optimizer_steps=7, wait=unused))
    assert saves[0][2] == {"batch": 4, "policy_version": 0, "optimizer_steps": 4}


def _rows(path: Path, *rows: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "checkpoints.jsonl").open("a") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def _leader_row(name: str, batch: int, worker_state: str) -> dict[str, Any]:
    named = {"name": name, "state_path": worker_state, "policy_version": batch + 10, "optimizer_steps": batch + 20}
    return {
        "name": name,
        "batch": batch,
        "policy_version": batch,
        "optimizer_steps": batch,
        "policies": {"worker": named},
        "state_path": f"tinker://main/weights/{name}",
        "sampler_path": f"tinker://main/sampler_weights/{name}",
    }


def _worker_row(name: str, batch: int) -> dict[str, Any]:
    state = f"tinker://worker/weights/{name}"
    return {
        "name": name,
        "batch": batch,
        "policy_version": batch + 10,
        "optimizer_steps": batch + 20,
        "state_path": state,
    }


def test_a_resume_takes_the_newest_checkpoint_whose_rows_every_policy_still_has(tmp_path, caplog):
    run = tmp_path / "run"
    workers = run / "policies" / "worker"
    _rows(run, _leader_row("000001", 1, "tinker://worker/weights/000001"))
    _rows(run, _leader_row("000002", 2, "tinker://worker/weights/000002"))
    _rows(run, _leader_row("000003", 3, "tinker://worker/weights/000003"))
    _rows(workers, _worker_row("000001", 1), _worker_row("000002", 2))  # a crash lost the third
    deleted = {"path": "tinker://worker/weights/000002", "name": "000002", "result": "deleted"}
    (workers / "checkpoints_deleted.jsonl").write_text(json.dumps(deleted) + "\n")
    (run / "lock.json").write_text("{}")

    with caplog.at_level(logging.WARNING):
        start = StartPoint.read(str(run), 1, leader=MAIN, others=["worker"])

    record, worker_record = start.record, start.policies["worker"].record
    assert record is not None and worker_record is not None
    assert (record.name, start.step, start.policy_version) == ("000001", 1, 1)
    assert worker_record.state_path == "tinker://worker/weights/000001"
    assert (start.policies["worker"].version, start.policies["worker"].optimizer_steps) == (11, 21)
    [record] = json.loads((run / "lock.json").read_text())[barrier.RESTORE_KEY]
    assert record["checkpoint"] == "000001" and len(record["skipped"]) == 2
    assert "policy worker no longer has the row it names (tinker://worker/weights/000003)" in record["skipped"][0]
    assert "Resuming from checkpoint 000001" in caplog.text


def test_a_run_without_checkpoints_starts_every_policy_fresh_and_one_whose_rows_never_hold_cannot_resume(tmp_path):
    start = StartPoint.read(str(tmp_path), 1, leader=MAIN, others=["worker"])
    assert start.record is None and {policy: begins.version for policy, begins in start.policies.items()} == {
        MAIN: 0,
        "worker": 0,
    }
    _rows(tmp_path, _leader_row("000001", 1, "tinker://worker/weights/000001"))
    # No restart could mend it: a configuration error (exit 78), which says what to do.
    with pytest.raises(ConfigError, match="names rows that every trained policy still has") as raised:
        StartPoint.read(str(tmp_path), 1, leader=MAIN, others=["worker"])
    assert str(raised.value).endswith(
        "Restore the rows each names (in <run>/policies/<id>/checkpoints.jsonl, with their checkpoints on the policy's "
        "service), or train a new trial in a new run directory"
    )


def test_a_policy_s_pruner_keeps_the_rows_that_the_leader_rows_a_resume_may_take_name(tmp_path):
    for batch in (1, 2, 3, 4):
        name = f"recovery-00000{batch}"
        _rows(tmp_path, _leader_row(name, batch, f"tinker://worker/weights/{name}"))
    _rows(tmp_path, _leader_row("000005", 5, "tinker://worker/weights/000005"))
    # The leader keeps its newest row and its newest keep_recovery recovery rows to resume from; the others, which it
    # may keep while its rollouts sample them, keep no row of the worker's.
    assert named_state_paths(str(tmp_path), "worker", keep=1) == {
        "tinker://worker/weights/000005",
        "tinker://worker/weights/recovery-000004",
    }
    assert len(named_state_paths(str(tmp_path), "worker", keep=3)) == 4
    (tmp_path / "checkpoints_deleted.jsonl").write_text(
        json.dumps({"path": "tinker://main/weights/recovery-000004"}) + "\n"
    )
    named = named_state_paths(str(tmp_path), "worker", keep=1)
    # The fourth leader row can no longer be resumed from: the newest the leader still has is the third.
    assert named == {"tinker://worker/weights/000005", "tinker://worker/weights/recovery-000003"}
    named = {"tinker://worker/weights/recovery-000001"}

    rows = [
        {**_worker_row(f"recovery-00000{batch}", batch), "sampler_path": f"tinker://worker/sampler_weights/r{batch}"}
        for batch in (1, 2, 3)
    ]
    plan = plan_deletions(
        rows, keep=1, deleted=set(), policy_version=13, max_staleness=0, this_start=set(), named=named
    )
    assert [path for _, path in plan] == [
        "tinker://worker/weights/recovery-000002",
        "tinker://worker/sampler_weights/r2",
    ]
