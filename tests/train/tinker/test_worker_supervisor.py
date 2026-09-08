"""The parent side: deadlines, killing a process tree, and classification.

The properties under test are the ones that make process-per-rollout safe to
run at 24-96 concurrent rollouts for a week:

* one killed worker must not disturb its siblings (the property a
  ``ProcessPoolExecutor`` cannot give -- one killed child there fails every
  concurrent future with ``BrokenProcessPool``);
* a worker that wrote a valid response and was then signalled must still count,
  because a worker needs ~1.9 s of interpreter teardown after it writes and its
  group's tail grace can expire inside that window;
* a pre-exec failure must cost one member, never the group, and must be retried
  once before it does even that.
"""

from __future__ import annotations

import asyncio
import os
import pickle
import signal
from dataclasses import asdict

import pytest

import platoon.train.tinker  # noqa: F401  (import order: package before data processing)
from platoon.config_defs import RolloutConfig
from platoon.envs.base import Task
from platoon.train.tinker.workers.protocol import (
    PROTOCOL_VERSION,
    ConversionPolicy,
    ModelBinding,
    RolloutWorkerRequest,
    RolloutWorkerResponse,
)
from platoon.train.tinker.workers.supervisor import RolloutWorkerSupervisor
from platoon.utils.rollout_workload import RolloutWorkload

STUB_DIR = os.path.dirname(os.path.abspath(__file__))
STUB_MODULE = "worker_stub_entrypoint"


def _request(scratch, request_id="task-r0-aaaaaaaa", **overrides) -> RolloutWorkerRequest:
    base = dict(
        protocol_version=PROTOCOL_VERSION,
        request_id=request_id,
        task_id="task",
        rollout_number=0,
        task=Task(goal="g", id="task"),
        rollout_config=asdict(RolloutConfig(output_dir=str(scratch))),
        rollout_fn_ref="worker_stub_entrypoint:main",
        model=ModelBinding(
            binding_path=os.path.join(str(scratch), "model_binding.pkl"),
            bootstrap_blob=b"",
            follow_poll_seconds=5.0,
            model_name="",
            renderer_name="",
        ),
        conversion=ConversionPolicy(),
        response_path=os.path.join(str(scratch), f"{request_id}.res"),
        ready_path=os.path.join(str(scratch), f"{request_id}.ready"),
        startup_budget_seconds=20.0,
        rollout_budget_seconds=2.0,
    )
    base.update(overrides)
    return RolloutWorkerRequest(**base)


def _response_blob(tmp_path, request_id="task-r0-aaaaaaaa", datums=0) -> str:
    response = RolloutWorkerResponse(
        protocol_version=PROTOCOL_VERSION,
        request_id=request_id,
        status="completed",
        result=None,
        workload=RolloutWorkload(trajectories=1, model_calls=datums),
        observed=True,
        model_calls=datums,
        routing_id_present=datums,
        worker_pid=0,
    )
    path = str(tmp_path / f"blob-{request_id}.pkl")
    with open(path, "wb") as handle:
        pickle.dump(response, handle, protocol=5)
    return path


def _supervisor(tmp_path, mode, *, response_blob=None, request_id="task-r0-aaaaaaaa", **kwargs):
    scratch = tmp_path / "ipc"
    scratch.mkdir(parents=True, exist_ok=True)
    markers = tmp_path / "markers"
    markers.mkdir(parents=True, exist_ok=True)
    env = {
        "STUB_MODE": mode,
        "STUB_MARKER_DIR": str(markers),
        "STUB_READY_PATH": os.path.join(str(scratch), f"{request_id}.ready"),
        "PYTHONPATH": os.pathsep.join([STUB_DIR, os.environ.get("PYTHONPATH", "")]),
        "STUB_RESPONSE_DST": os.path.join(str(scratch), f"{request_id}.res"),
    }
    if response_blob:
        env["STUB_RESPONSE_SRC"] = response_blob
    kwargs.setdefault("termination_grace_seconds", 1.0)
    kwargs.setdefault("parent_grace_seconds", 2.0)
    kwargs.setdefault("ready_poll_seconds", 0.05)
    return (
        RolloutWorkerSupervisor(
            module=STUB_MODULE,
            scratch_dir=str(scratch),
            log_dir=str(tmp_path / "logs"),
            env_overrides=env,
            **kwargs,
        ),
        str(scratch),
        markers,
    )


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_normal_completion_and_cleanup(tmp_path):
    blob = _response_blob(tmp_path)
    supervisor, scratch, _markers = _supervisor(tmp_path, "ok", response_blob=blob)
    request = _request(scratch)

    outcome = asyncio.run(supervisor.run(request))

    assert outcome.disposition == "completed"
    assert outcome.response is not None and outcome.response.observed is True
    assert outcome.returncode == 0
    assert outcome.killed_by_parent is False
    assert outcome.retried is False
    # Both temp files are gone; the request especially, since it carries the key.
    assert not os.path.exists(request.response_path)
    assert not os.path.exists(os.path.join(scratch, f"{request.request_id}.req"))
    assert os.path.exists(outcome.log_path)
    assert outcome.launch_seconds > 0.0


def test_a_response_written_just_before_a_kill_still_counts(tmp_path):
    """RACE GATE: classification is response-first, never returncode-first.

    A worker needs ~1.9 s of interpreter teardown after ``os.replace``-ing its
    response, and during that window the parent's task is still not done and
    can be selected as a straggler.  Classifying on the return code would throw
    away a complete rollout's datums because the tail grace expired two seconds
    after it finished.
    """

    blob = _response_blob(tmp_path, datums=5)
    supervisor, scratch, _markers = _supervisor(tmp_path, "write_then_sleep", response_blob=blob)
    request = _request(scratch, rollout_budget_seconds=0.5)

    outcome = asyncio.run(supervisor.run(request))

    assert outcome.killed_by_parent is True
    assert outcome.returncode is not None and outcome.returncode < 0
    # The worker's own verdict wins regardless of who signalled it.
    assert outcome.disposition == "completed"
    assert outcome.response is not None
    assert outcome.response.model_calls == 5


def test_a_killed_worker_does_not_disturb_its_siblings(tmp_path):
    """The ProcessPoolExecutor property this design exists to avoid.

    In a pool, one child that is SIGKILLed fails every concurrent future with
    ``BrokenProcessPool``.  Here a spinner is killed at the parent deadline
    while two siblings complete normally.
    """

    async def scenario():
        good_blob = _response_blob(tmp_path, request_id="good-0")
        spinner, spin_scratch, _m = _supervisor(
            tmp_path / "spin", "spin", request_id="spin-0"
        )
        results = []

        async def sibling(index):
            blob = _response_blob(tmp_path, request_id=f"good-{index}")
            supervisor, scratch, _markers = _supervisor(
                tmp_path / f"good{index}", "ok", response_blob=blob, request_id=f"good-{index}"
            )
            outcome = await supervisor.run(_request(scratch, request_id=f"good-{index}"))
            results.append(outcome)

        assert good_blob
        spin_outcome, *_ = await asyncio.gather(
            spinner.run(_request(spin_scratch, request_id="spin-0", rollout_budget_seconds=0.5)),
            sibling(1),
            sibling(2),
        )
        return spin_outcome, results

    spin_outcome, siblings = asyncio.run(scenario())
    assert spin_outcome.disposition in ("hard_timeout", "worker_crashed")
    assert spin_outcome.killed_by_parent is True
    assert spin_outcome.response is None
    assert len(siblings) == 2
    assert all(outcome.disposition == "completed" for outcome in siblings)
    assert all(outcome.response is not None for outcome in siblings)


def test_segfault_is_contained_and_classified(tmp_path):
    supervisor, scratch, _markers = _supervisor(tmp_path, "segfault")
    outcome = asyncio.run(supervisor.run(_request(scratch)))
    assert outcome.disposition == "worker_crashed"
    assert outcome.returncode == -signal.SIGSEGV
    assert outcome.killed_by_parent is False
    assert outcome.response is None


def test_a_nonzero_exit_is_a_crash(tmp_path):
    supervisor, scratch, _markers = _supervisor(tmp_path, "crash")
    outcome = asyncio.run(supervisor.run(_request(scratch)))
    assert outcome.disposition == "worker_crashed"
    assert outcome.returncode == 3


def test_a_clean_exit_with_no_response_is_a_launch_failure(tmp_path):
    supervisor, scratch, _markers = _supervisor(tmp_path, "clean_no_response")
    outcome = asyncio.run(supervisor.run(_request(scratch)))
    assert outcome.disposition == "worker_launch_failed"
    assert outcome.returncode == 0


def test_a_truncated_response_is_never_partially_unpickled(tmp_path):
    blob = _response_blob(tmp_path)
    supervisor, scratch, _markers = _supervisor(tmp_path, "truncated", response_blob=blob)
    outcome = asyncio.run(supervisor.run(_request(scratch)))
    assert outcome.response is None
    assert outcome.disposition == "worker_launch_failed"


def test_a_worker_trapping_sigterm_is_sigkilled_after_the_grace(tmp_path):
    supervisor, scratch, markers = _supervisor(
        tmp_path, "trap_sigterm", termination_grace_seconds=0.5
    )
    outcome = asyncio.run(supervisor.run(_request(scratch, rollout_budget_seconds=0.5)))
    assert outcome.killed_by_parent is True
    assert outcome.returncode == -signal.SIGKILL
    assert any(name.startswith("trapped-") for name in os.listdir(markers))


def test_the_whole_process_group_is_reaped(tmp_path):
    """A rollout's own children die with it, not with the trainer."""

    supervisor, scratch, markers = _supervisor(tmp_path, "spawn_child_then_spin")
    outcome = asyncio.run(supervisor.run(_request(scratch, rollout_budget_seconds=0.5)))
    assert outcome.killed_by_parent is True

    grandchildren = [
        int(open(os.path.join(markers, name)).read())
        for name in os.listdir(markers)
        if name.startswith("grandchild-pid-")
    ]
    assert grandchildren, "the stub did not report its grandchild"
    for pid in grandchildren:
        for _ in range(100):
            if not _alive(pid):
                break
            import time as _time

            _time.sleep(0.05)
        assert not _alive(pid), f"grandchild {pid} outlived its worker"


def test_cancellation_reaps_the_group_and_re_raises(tmp_path):
    async def scenario():
        supervisor, scratch, _markers = _supervisor(tmp_path, "sleep")
        request = _request(scratch, rollout_budget_seconds=60.0)
        task = asyncio.ensure_future(supervisor.run(request))
        await asyncio.sleep(0.8)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    # Nothing is left behind for the next attempt to trip over.
    assert not any(name.endswith(".req") for name in os.listdir(tmp_path / "ipc"))


def test_exec_of_a_missing_module_is_retried_once_and_never_raises(tmp_path):
    scratch = tmp_path / "ipc"
    scratch.mkdir()
    supervisor = RolloutWorkerSupervisor(
        module="definitely_not_a_module_xyz",
        scratch_dir=str(scratch),
        log_dir=str(tmp_path / "logs"),
        parent_grace_seconds=2.0,
        ready_poll_seconds=0.05,
    )
    outcome = asyncio.run(supervisor.run(_request(str(scratch), rollout_budget_seconds=1.0)))
    # A missing module still execs successfully (python starts, then fails), so
    # the process exits non-zero with no response.
    assert outcome.disposition in ("worker_crashed", "worker_launch_failed")
    assert outcome.response is None


def test_a_pre_exec_failure_is_retried_once_then_reported(tmp_path, monkeypatch):
    from platoon.train.tinker.workers import supervisor as supervisor_module

    scratch = tmp_path / "ipc"
    scratch.mkdir()
    supervisor = RolloutWorkerSupervisor(
        scratch_dir=str(scratch), log_dir=str(tmp_path / "logs"), parent_grace_seconds=1.0
    )
    attempts = []

    def failing_dump(request, path):
        attempts.append(request.request_id)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(supervisor_module, "dump_request", failing_dump)
    outcome = asyncio.run(supervisor.run(_request(str(scratch))))

    assert outcome.disposition == "worker_launch_failed"
    assert outcome.retried is True
    assert outcome.response is None
    # Exactly one retry, with a fresh id so it cannot collide with the first.
    assert len(attempts) == 2
    assert attempts[0] != attempts[1]


def test_stdin_is_devnull_in_the_child(tmp_path):
    """PTC runs model-authored code on an embedded IPython shell, which can
    prompt; inheriting fd 0 would let workers race to read the trainer's stdin.
    """

    blob = _response_blob(tmp_path)
    supervisor, scratch, markers = _supervisor(tmp_path, "check_stdin", response_blob=blob)
    asyncio.run(supervisor.run(_request(scratch)))
    assert open(os.path.join(markers, "stdin")).read() == "/dev/null"


def test_diag_marker_separates_timeout_from_oom_from_crash(tmp_path):
    """Three failures with the identical observable and opposite responses."""

    supervisor, scratch, _m = _supervisor(tmp_path / "a", "diag_timeout", request_id="t-0")
    timeout_outcome = asyncio.run(supervisor.run(_request(scratch, request_id="t-0")))
    assert timeout_outcome.disposition == "hard_timeout"
    assert timeout_outcome.diag["phase"] == "rollout"
    # A SIGKILLed worker still contributes its cell-guard counters.
    assert timeout_outcome.diag["ptc_cells_stuck_at_exit"] == "1"

    supervisor, scratch, _m = _supervisor(tmp_path / "b", "diag_startup", request_id="s-0")
    startup_outcome = asyncio.run(supervisor.run(_request(scratch, request_id="s-0")))
    assert startup_outcome.disposition == "hard_timeout"
    assert startup_outcome.diag["phase"] == "startup"

    supervisor, scratch, _m = _supervisor(tmp_path / "c", "diag_rss", request_id="r-0")
    oom_outcome = asyncio.run(supervisor.run(_request(scratch, request_id="r-0")))
    assert oom_outcome.disposition == "rollout_worker_oom"
    assert oom_outcome.diag["phase"] == "rss"

    # The distinct exit code alone is enough, even without a marker.
    supervisor, scratch, _m = _supervisor(tmp_path / "d", "rss_exit_only", request_id="r-1")
    code_only = asyncio.run(supervisor.run(_request(scratch, request_id="r-1")))
    assert code_only.disposition == "rollout_worker_oom"

    supervisor, scratch, _m = _supervisor(tmp_path / "e", "segfault", request_id="x-0")
    crash = asyncio.run(supervisor.run(_request(scratch, request_id="x-0")))
    assert crash.disposition == "worker_crashed"
    assert crash.diag is None


def test_memory_floor_refuses_to_launch(tmp_path):
    scratch = tmp_path / "ipc"
    scratch.mkdir()
    supervisor = RolloutWorkerSupervisor(
        module=STUB_MODULE,
        scratch_dir=str(scratch),
        mem_available_floor_bytes=1 << 62,  # unreachable on any machine
        launch_retries=0,
    )
    outcome = asyncio.run(supervisor.run(_request(str(scratch))))
    assert outcome.disposition == "worker_launch_failed"
    assert outcome.worker_pid is None


def test_a_response_for_a_different_request_is_ignored(tmp_path):
    blob = _response_blob(tmp_path, request_id="somebody-else")
    supervisor, scratch, _markers = _supervisor(tmp_path, "ok", response_blob=blob)
    outcome = asyncio.run(supervisor.run(_request(scratch)))
    assert outcome.response is None
    assert outcome.disposition == "worker_launch_failed"


def test_ready_callback_fires_before_the_worker_finishes(tmp_path):
    blob = _response_blob(tmp_path)
    supervisor, scratch, _markers = _supervisor(tmp_path, "ok", response_blob=blob)
    fired = []
    launched = []
    reaped = []
    asyncio.run(
        supervisor.run(
            _request(scratch),
            on_ready=lambda: fired.append(True),
            on_launch=launched.append,
            on_reap=reaped.append,
        )
    )
    assert fired == [True]
    assert launched and launched == reaped


def test_workers_are_launched_with_pinned_threads_and_no_telemetry(tmp_path):
    supervisor = RolloutWorkerSupervisor(scratch_dir=str(tmp_path))
    env = supervisor._child_env()
    assert env["TINKER_TELEMETRY"] == "0"
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["PLATOON_TINKER_ROLLOUT_WORKER"] == "1"
    assert "TINKER_SUBPROCESS_SAMPLING" not in env
    # Inherit-plus-override: the curated production environment survives.
    if "PATH" in os.environ:
        assert env["PATH"] == os.environ["PATH"]


def test_stale_ipc_files_are_swept_at_construction(tmp_path):
    scratch = tmp_path / "ipc"
    scratch.mkdir()
    stale = scratch / "old.res"
    stale.write_bytes(b"x")
    fresh = scratch / "new.res"
    fresh.write_bytes(b"x")
    old_time = 1.0
    os.utime(stale, (old_time, old_time))

    RolloutWorkerSupervisor(scratch_dir=str(scratch))
    assert not stale.exists()
    assert fresh.exists()
