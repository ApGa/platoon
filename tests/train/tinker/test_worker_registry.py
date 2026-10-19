"""A dead trainer's rollout workers are killed before the next trainer starts, and live trainers' never are.

Workers lead their own sessions, so nothing in a trainer's exit reaches them but PDEATHSIG.  These tests cover the
backstops for a worker that outlives its trainer anyway: the on-disk registry and its sweep, the new supervisor's
sweep before its first launch, and the SIGTERM the trainer sends its live workers on the exits that run no cleanup
(the watchdog's ``os._exit``).  ``platoon train``'s sweep before each restart (``watchdog.max_restarts``) is
``tests/train/test_supervise.py``'s.  The "workers" are stand-ins that
import nothing (``sleep``, or the dependency-free stub entrypoint) unless a test needs the real supervisor.
"""

from __future__ import annotations

import asyncio
import os
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
from platoon.train.tinker import rl as rl_module
from platoon.train.tinker.workers.protocol import (
    PROTOCOL_VERSION,
    ConversionPolicy,
    ModelBinding,
    RolloutWorkerRequest,
)
from platoon.train.tinker.workers.registry import (
    WorkerRecord,
    process_start_time,
    record_worker,
    registry_dir,
    sweep_orphaned_workers,
)
from platoon.train.tinker.workers.supervisor import RolloutWorkerSupervisor, signal_live_workers

STUB_DIR = os.path.dirname(os.path.abspath(__file__))


def _running(pid: int) -> bool:
    try:
        raw = Path(f"/proc/{pid}/stat").read_bytes()
    except OSError:
        return False
    return raw[raw.rindex(b")") + 2 : raw.rindex(b")") + 3] not in (b"Z", b"X", b"x")


def _wait_gone(pid: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while _running(pid):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def _dead_process() -> tuple[int, int]:
    """The pid and start time of a process that has exited (a trainer that is gone)."""

    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.3)"])
    start_time = process_start_time(process.pid)
    assert start_time is not None
    process.wait(timeout=30)
    return process.pid, start_time


def _worker(*command: str) -> subprocess.Popen:
    """A stand-in worker that leads its own session, as a real one does."""

    return subprocess.Popen(list(command) or ["sleep", "1000"], start_new_session=True)


def _record(directory: str, worker: subprocess.Popen, trainer: tuple[int, int]) -> None:
    start_time = process_start_time(worker.pid)
    assert start_time is not None
    record_worker(
        directory,
        WorkerRecord(
            pid=worker.pid,
            pgid=worker.pid,
            start_time=start_time,
            trainer_pid=trainer[0],
            trainer_start_time=trainer[1],
            request_id=f"r-{worker.pid}",
        ),
    )


@pytest.fixture
def workers():
    started: list[subprocess.Popen] = []
    yield started
    for worker in started:
        if worker.poll() is None:
            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait(timeout=30)


def test_the_sweep_stops_only_the_workers_whose_trainer_is_gone(tmp_path, workers):
    directory = str(tmp_path / "workers")
    this_trainer = (os.getpid(), process_start_time(os.getpid()))
    orphan, owned = _worker(), _worker()
    workers += [orphan, owned]
    _record(directory, orphan, _dead_process())
    _record(directory, owned, this_trainer)

    stopped = sweep_orphaned_workers(directory, grace_seconds=5.0)

    assert stopped == [orphan.pid]
    assert orphan.wait(timeout=10) == -signal.SIGTERM  # the polite SIGTERM was enough
    assert owned.poll() is None
    assert sorted(os.listdir(directory)) == [str(owned.pid)]  # the orphan's file is dropped, the live one's kept


def test_a_worker_that_ignores_sigterm_is_killed_after_the_grace(tmp_path, workers):
    directory = str(tmp_path / "workers")
    # bash ignores TERM, and so does the sleep it runs: an ignored signal stays ignored across exec.
    stubborn = _worker("bash", "-c", "trap '' TERM; sleep 1000; :")
    workers.append(stubborn)
    time.sleep(0.2)  # let bash set its trap
    _record(directory, stubborn, _dead_process())

    started = time.monotonic()
    assert sweep_orphaned_workers(directory, grace_seconds=0.5) == [stubborn.pid]

    assert stubborn.wait(timeout=10) == -signal.SIGKILL
    assert time.monotonic() - started < 5.0


def test_a_recycled_pid_is_never_signalled(tmp_path, workers):
    """A file whose pid now names another process (a different start time) is dropped, the process untouched."""

    directory = str(tmp_path / "workers")
    bystander = _worker()
    workers.append(bystander)
    start_time = process_start_time(bystander.pid)
    trainer = _dead_process()
    record_worker(
        directory,
        WorkerRecord(
            pid=bystander.pid,
            pgid=bystander.pid,
            start_time=start_time - 1,  # the recorded worker started earlier: not this process
            trainer_pid=trainer[0],
            trainer_start_time=trainer[1],
        ),
    )

    assert sweep_orphaned_workers(directory, grace_seconds=0.5) == []
    assert bystander.poll() is None and os.listdir(directory) == []


def test_a_new_supervisor_stops_a_dead_trainers_workers_before_its_first_launch(tmp_path, workers):
    scratch = tmp_path / "ipc"
    scratch.mkdir()
    orphan = _worker()
    workers.append(orphan)
    _record(registry_dir(str(scratch)), orphan, _dead_process())
    supervisor = RolloutWorkerSupervisor(
        module="worker_stub_entrypoint",
        scratch_dir=str(scratch),
        log_dir=str(tmp_path / "logs"),
        env_overrides={"STUB_MODE": "clean_no_response", "PYTHONPATH": STUB_DIR},
        termination_grace_seconds=1.0,
    )
    request = _request(scratch)

    asyncio.run(supervisor.run(request))

    assert orphan.wait(timeout=10) == -signal.SIGTERM
    # The supervisor's own worker came and went, and was dropped from the registry once reaped.
    assert os.listdir(registry_dir(str(scratch))) == []


def _request(scratch: Path, request_id: str = "task-r0-aaaaaaaa", **overrides) -> RolloutWorkerRequest:
    base = dict(
        protocol_version=PROTOCOL_VERSION,
        request_id=request_id,
        task_id="task",
        rollout_number=0,
        task=Task(goal="g", id="task"),
        rollout_config=asdict(RolloutConfig(output_dir=str(scratch))),
        rollout_fn_ref="worker_stub_entrypoint:main",
        model=ModelBinding(
            binding_path=str(scratch / "model_binding.pkl"),
            bootstrap_blob=b"",
            follow_poll_seconds=5.0,
            model_name="",
            renderer_name="",
        ),
        conversion=ConversionPolicy(),
        response_path=str(scratch / f"{request_id}.res"),
        ready_path=str(scratch / f"{request_id}.ready"),
        startup_budget_seconds=20.0,
        rollout_budget_seconds=60.0,
    )
    base.update(overrides)
    return RolloutWorkerRequest(**base)


def test_the_trainer_signals_its_live_workers(tmp_path):
    """``signal_live_workers`` is what the watchdog's and the second signal's ``os._exit`` call first."""

    scratch = tmp_path / "ipc"
    scratch.mkdir()
    request = _request(scratch)
    supervisor = RolloutWorkerSupervisor(
        module="worker_stub_entrypoint",
        scratch_dir=str(scratch),
        log_dir=str(tmp_path / "logs"),
        env_overrides={"STUB_MODE": "sleep", "STUB_READY_PATH": request.ready_path, "PYTHONPATH": STUB_DIR},
        ready_poll_seconds=0.05,
    )

    async def scenario() -> tuple[int, object]:
        running = asyncio.ensure_future(supervisor.run(request))
        while not os.path.exists(request.ready_path):
            await asyncio.sleep(0.05)
        signalled = signal_live_workers()
        return signalled, await asyncio.wait_for(running, 30)

    signalled, outcome = asyncio.run(scenario())

    assert signalled == 1
    assert outcome.returncode == -signal.SIGTERM and not outcome.killed_by_parent
    assert signal_live_workers() == 0  # reaped workers are forgotten


def test_the_watchdog_tells_the_workers_to_stop_before_it_exits(monkeypatch):
    events: list[object] = []
    watchdog = rl_module.Watchdog(timeout_seconds=0.2, exit_code=2)

    def fake_exit(code: int) -> None:
        events.append(("exit", code))
        watchdog._running = False  # end the monitor loop instead of the process

    monkeypatch.setattr(rl_module, "_stop_rollout_workers", lambda: events.append("stop workers"))
    monkeypatch.setattr(rl_module.os, "_exit", fake_exit)
    watchdog.start()
    deadline = time.monotonic() + 10
    while ("exit", 2) not in events and time.monotonic() < deadline:
        time.sleep(0.05)
    watchdog.stop()

    assert events[:2] == ["stop workers", ("exit", 2)]


def test_registry_files_are_private(tmp_path, workers):
    directory = str(tmp_path / "workers")
    worker = _worker()
    workers.append(worker)
    _record(directory, worker, (os.getpid(), process_start_time(os.getpid())))

    assert os.stat(directory).st_mode & 0o777 == 0o700
    assert os.stat(os.path.join(directory, str(worker.pid))).st_mode & 0o777 == 0o600
