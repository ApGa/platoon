"""A rollout worker's sandboxes do not outlive it: a stopping worker stops them, and its supervisor reaps by the
worker's label what a killed worker left.

These launch the real worker entrypoint (slow: each worker imports the tinker/torch stack).  The rollout starts a
process-provider sandbox with a background command and hangs; the background command leads a session of its own,
so only the sandbox's own cleanup ends it once the worker is gone.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import time
from pathlib import Path

from fakes import WORKER_STUBS
from fakes.workers import worker_request

import platoon.train.tinker  # noqa: F401  (import order: package before data processing)
from platoon.sandbox.cleanup import RUN_LABEL, WORKER_LABEL
from platoon.train.workers.protocol import RolloutWorkerRequest
from platoon.train.workers.supervisor import RolloutWorkerSupervisor

STUB_DIR = str(WORKER_STUBS)


def _request(tmp_path: Path, request_id: str, sandbox_labels: dict[str, str] | None = None) -> RolloutWorkerRequest:
    scratch = tmp_path / "ipc"
    scratch.mkdir(parents=True, exist_ok=True)
    return worker_request(
        scratch,
        request_id,
        rollout="worker_stub_rollouts:hold_a_sandbox",
        output_dir=tmp_path / "out",
        startup_budget_seconds=600.0,
        rollout_budget_seconds=600.0,
        sandbox_labels=dict(sandbox_labels or {}),
    )


def _supervisor(tmp_path: Path) -> RolloutWorkerSupervisor:
    return RolloutWorkerSupervisor(
        scratch_dir=str(tmp_path / "ipc"),
        log_dir=str(tmp_path / "logs"),
        env_overrides={
            "PYTHONPATH": os.pathsep.join([STUB_DIR, os.environ.get("PYTHONPATH", "")]),
            "STUB_SANDBOX_ROOT": str(tmp_path / "sandboxes"),
            "STUB_FACTS_PATH": str(tmp_path / "facts.json"),
            "HF_HUB_OFFLINE": "1",
        },
        parent_grace_seconds=30.0,
    )


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state not in ("Z", "X")


async def _until(condition, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting")
        await asyncio.sleep(0.1)


def test_a_stopping_worker_stops_the_sandboxes_its_rollout_holds(tmp_path, monkeypatch):
    async def leave_them(self, request) -> None:  # only the worker itself may stop them here
        return None

    monkeypatch.setattr(RolloutWorkerSupervisor, "_reap_orphaned_sandboxes", leave_them)
    supervisor = _supervisor(tmp_path)
    facts_path = tmp_path / "facts.json"

    async def scenario() -> dict:
        running = asyncio.create_task(supervisor.run(_request(tmp_path, "straggler-0")))
        await _until(facts_path.exists, 300)  # the rollout holds its sandbox
        running.cancel()  # the supervisor's straggler kill: SIGTERM
        try:
            await running
        except asyncio.CancelledError:
            pass
        return json.loads(facts_path.read_text())

    facts = asyncio.run(scenario())

    assert facts["labels"] == {WORKER_LABEL: "straggler-0"}
    assert not Path(facts["sandbox_dir"]).exists()  # stopped: only the sandbox's own stop deletes it
    asyncio.run(_until(lambda: not _alive(facts["sleep_pid"]), 10))


def test_the_supervisor_reaps_the_sandboxes_of_a_killed_worker(tmp_path):
    """The worker's sandbox carries the trainer's run label beside its own, and is reaped by both."""

    supervisor = _supervisor(tmp_path)
    run = {RUN_LABEL: str(tmp_path / "run")}
    request = _request(tmp_path, "killed-0", sandbox_labels=run)
    facts_path = tmp_path / "facts.json"
    pids: list[int] = []

    async def scenario():
        running = asyncio.create_task(supervisor.run(request, on_launch=pids.append))
        await _until(facts_path.exists, 300)
        assert Path(request.sandboxes_path).exists()  # the worker listed how to reap its sandbox
        os.kill(pids[0], signal.SIGKILL)  # no cleanup of its own can run
        return await running

    outcome = asyncio.run(scenario())
    facts = json.loads(facts_path.read_text())

    assert outcome.response is None
    assert facts["labels"] == {**run, WORKER_LABEL: "killed-0"}
    assert not Path(facts["sandbox_dir"]).exists()  # reaped by the worker's label
    # Its process group, in a session of its own, survived the worker until the reap.
    asyncio.run(_until(lambda: not _alive(facts["sleep_pid"]), 10))
    assert not os.path.exists(request.sandboxes_path)
