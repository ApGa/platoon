"""The scenarios the worker shutdown tests share: a rollout worker under a stand-in trainer, which is SIGKILLed.

The stand-in trainer runs the real supervisor, and the worker the real entrypoint, with a rollout from
``worker_stub_rollouts``.  ``_kill_trainer_mid_rollout`` kills the trainer once the agents of a recursive rollout are
busy at depth 2, ``_kill_trainer_once_ready`` once the rollout has noted its facts; both wait for the worker to be
gone.  Slow: each process imports the tinker/torch stack.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pytest

import platoon.train.tinker  # noqa: F401  (import order: package before data processing)
from platoon.config_defs import RolloutConfig
from platoon.task import Task
from platoon.train.tinker.workers.entrypoint import TERMINATION_SECONDS
from platoon.train.tinker.workers.protocol import (
    PROTOCOL_VERSION,
    ConversionPolicy,
    ModelBinding,
    RolloutWorkerRequest,
    dump_request,
)

STUB_DIR = os.path.dirname(os.path.abspath(__file__))

#: How much later than its deadline a worker may be seen gone: the alarm, the SIGKILL and this test's polling.
EXIT_SLACK_SECONDS = 4.0

TRAINER = """
import asyncio, sys
import platoon.train.tinker  # noqa: F401
from platoon.train.tinker.workers.protocol import load_request
from platoon.train.tinker.workers.supervisor import RolloutWorkerSupervisor

request_path, scratch, logs, pid_path = sys.argv[1:5]
supervisor = RolloutWorkerSupervisor(scratch_dir=scratch, log_dir=logs, parent_grace_seconds=600.0)


def launched(pid):
    with open(pid_path, "w") as handle:
        handle.write(str(pid))


asyncio.run(supervisor.run(load_request(request_path), on_launch=launched))
"""


def _running(pid: int) -> bool:
    """Whether ``pid`` can still run code (a zombie, waiting for whoever inherited it, cannot)."""

    try:
        raw = Path(f"/proc/{pid}/stat").read_bytes()
    except OSError:
        return False
    return raw[raw.rindex(b")") + 2 : raw.rindex(b")") + 3] not in (b"Z", b"X", b"x")


def _request(tmp_path: Path, rollout: str) -> RolloutWorkerRequest:
    scratch = tmp_path / "ipc"
    scratch.mkdir(parents=True, exist_ok=True)
    request_id = "shutdown-r0-00000000"
    return RolloutWorkerRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id=request_id,
        task_id="t0",
        rollout_number=0,
        task=Task(goal="root", id="t0"),
        rollout_config=asdict(RolloutConfig(output_dir=str(tmp_path / "out"))),
        rollout_fn_ref=f"worker_stub_rollouts:{rollout}",
        model=ModelBinding(
            binding_path=str(scratch / "model_binding.pkl"),
            bootstrap_blob=b"",
            follow_poll_seconds=5.0,
            model_name="",  # the stub rollout sets its own gateway
            renderer_name="",
        ),
        conversion=ConversionPolicy(),
        response_path=str(scratch / f"{request_id}.res"),
        ready_path=str(scratch / f"{request_id}.ready"),
        startup_budget_seconds=600.0,
        rollout_budget_seconds=3600.0,
    )


def _calls(path: Path) -> list[tuple[float, str, str]]:
    calls = []
    for line in path.read_text().splitlines() if path.exists() else []:
        stamp, kind, *goal = line.split(" ", 2)
        calls.append((float(stamp), kind, goal[0] if goal else ""))
    return calls


def _wait(condition, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            pytest.fail(f"timed out waiting for {what}")
        time.sleep(0.1)


def _kill_trainer_mid_rollout(tmp_path: Path, rollout: str) -> dict:
    """Run ``rollout`` in a worker under a stand-in trainer, SIGKILL the trainer once agents at depth 2 are busy,
    and wait for the worker to be gone."""

    request = _request(tmp_path, rollout)
    request_path = tmp_path / "request.pkl"
    dump_request(request, str(request_path))
    calls_path, facts_path, pid_path = tmp_path / "calls.log", tmp_path / "facts.json", tmp_path / "worker.pid"
    logs = tmp_path / "logs"
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": os.pathsep.join([STUB_DIR, env.get("PYTHONPATH", "")]),
            "STUB_CALL_LOG": str(calls_path),
            "STUB_FACTS_PATH": str(facts_path),
            "HF_HUB_OFFLINE": "1",
            "OPENHANDS_SUPPRESS_BANNER": "1",
        }
    )
    trainer = subprocess.Popen(
        [sys.executable, "-c", TRAINER, str(request_path), str(tmp_path / "ipc"), str(logs), str(pid_path)],
        env=env,
        start_new_session=True,
    )
    worker_pid = None
    try:
        # The whole tree is busy: subagents at depth 2 call the model and the env.
        _wait(lambda: any(goal.startswith("short-d2") for _, _, goal in _calls(calls_path)), 300, "depth-2 agents")
        _wait(facts_path.exists, 30, "the worker's facts")
        worker_pid = int(pid_path.read_text())
        facts = json.loads(facts_path.read_text())
        assert facts["worker_pid"] == worker_pid
        assert facts["parent_death_signal"] == signal.SIGTERM
        time.sleep(1.0)

        killed_at = time.time()
        trainer.kill()  # SIGKILL: nothing in the trainer runs, the worker only gets its PDEATHSIG
        trainer.wait(timeout=30)
        _wait(lambda: not _running(worker_pid), TERMINATION_SECONDS + EXIT_SLACK_SECONDS + 30, "the worker to exit")
        gone_at = time.time()
    finally:
        if trainer.poll() is None:
            trainer.kill()
        if worker_pid is not None and _running(worker_pid):
            os.killpg(worker_pid, signal.SIGKILL)
    log = (logs / f"{request.request_id}.log").read_text(errors="replace")
    # The trainer's death reaches the worker twice, and either names its stop: the PDEATHSIG SIGTERM, and its monitor
    # thread's check of getppid(), for a lost signal.  The kernel reparents the worker and sends the signal in one
    # step, but the signal's handler runs only once the worker's main thread runs Python again, and the monitor's
    # check (every two seconds) can come in between.
    started = re.search(r"\[rollout-worker\] terminating at (\d+\.\d+) \((?:SIGTERM|its parent is gone)\)", log)
    assert started, log[-4000:]
    return {
        "killed_at": killed_at,
        "gone_at": gone_at,
        "terminating_at": float(started.group(1)),
        "calls": _calls(calls_path),
        "log": log,
        "sleeper_pid": facts["sleeper_pid"],
    }


def _assert_stopped_cleanly(run: dict) -> None:
    killed_at, terminating_at = run["killed_at"], run["terminating_at"]
    # The trainer's death reached the worker (its PDEATHSIG, or its monitor thread's check), and it acted at once.
    assert 0 <= terminating_at - killed_at < 2.0, (killed_at, terminating_at)
    # Its tree was busy right up to the signal ...
    assert sum(1 for stamp, _, _ in run["calls"] if killed_at - 1.0 <= stamp <= terminating_at) >= 10
    # ... and from the moment it began terminating: no model call, no tool call, no subagent, no env session.
    late = [call for call in run["calls"] if call[0] > terminating_at]
    assert late == [], late[:10]
    # Nothing it started outlives it: its last act SIGKILLs its process group, which the sleeper may still be leaving.
    assert _gone_soon(run["sleeper_pid"]), "the rollout's subprocess outlived its worker"
    assert "[rollout-worker] terminated:" in run["log"] and "killing the process group" in run["log"]


def _kill_trainer_once_ready(tmp_path: Path, rollout: str, **env: str) -> dict:
    """Run ``rollout`` in a worker under a stand-in trainer, SIGKILL the trainer once the rollout noted its facts,
    and wait for the worker to be gone."""

    request = _request(tmp_path, rollout)
    request_path = tmp_path / "request.pkl"
    dump_request(request, str(request_path))
    facts_path, pid_path, state = tmp_path / "facts.json", tmp_path / "worker.pid", tmp_path / "state"
    state.mkdir()
    logs = tmp_path / "logs"
    child_env = dict(os.environ)
    child_env.update(
        {
            "PYTHONPATH": os.pathsep.join([STUB_DIR, child_env.get("PYTHONPATH", "")]),
            "STUB_FACTS_PATH": str(facts_path),
            "STUB_STATE_DIR": str(state),
            "HF_HUB_OFFLINE": "1",
            **env,
        }
    )
    trainer = subprocess.Popen(
        [sys.executable, "-c", TRAINER, str(request_path), str(tmp_path / "ipc"), str(logs), str(pid_path)],
        env=child_env,
        start_new_session=True,
    )
    worker_pid = None
    try:
        _wait(facts_path.exists, 300, "the rollout to be ready")
        worker_pid = int(pid_path.read_text())
        facts = json.loads(facts_path.read_text())
        assert facts["worker_pid"] == worker_pid
        assert facts["parent_death_signal"] == signal.SIGTERM
        time.sleep(0.5)

        killed_at = time.time()
        trainer.kill()  # SIGKILL: nothing in the trainer runs, the worker only gets its PDEATHSIG
        trainer.wait(timeout=30)
        _wait(lambda: not _running(worker_pid), TERMINATION_SECONDS + EXIT_SLACK_SECONDS + 30, "the worker to exit")
        gone_at = time.time()
    finally:
        if trainer.poll() is None:
            trainer.kill()
        if worker_pid is not None and _running(worker_pid):
            os.killpg(worker_pid, signal.SIGKILL)
    return {
        "killed_at": killed_at,
        "gone_at": gone_at,
        "facts": facts,
        "state": state,
        "diag": request.diag_path,
        "log": (logs / f"{request.request_id}.log").read_text(errors="replace"),
    }


def _gone_soon(pid: int) -> bool:
    """Whether ``pid`` stops running within a moment (a SIGKILL sent before the worker ended takes effect)."""

    deadline = time.monotonic() + 1.0
    while _running(pid):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def _kill_leftover(pid: int) -> None:
    """Kill what a failed test left running: ``pid``, and its group unless that is this test's own."""

    if not _running(pid):
        return
    try:
        group = os.getpgid(pid)
        if group != os.getpgrp():
            os.killpg(group, signal.SIGKILL)
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
