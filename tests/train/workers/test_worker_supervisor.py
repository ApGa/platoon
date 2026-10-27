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

Every test runs twice, with workers started both ways (``start_method`` exec and
forkserver): a forked worker must behave as an exec'd one in every case.
"""

from __future__ import annotations

import asyncio
import os
import pickle
import signal
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fakes import WORKER_STUBS
from fakes.workers import worker_request

import platoon.train.tinker  # noqa: F401  (import order: package before data processing)
from platoon.train.workers.protocol import PROTOCOL_VERSION, RolloutWorkerRequest, RolloutWorkerResponse
from platoon.train.workers.supervisor import RolloutWorkerSupervisor
from platoon.utils.rollout_workload import RolloutWorkload

STUB_DIR = str(WORKER_STUBS)
STUB_MODULE = "worker_stub_entrypoint"

# The start method of the running test (the autouse fixture below), and the supervisors it built, closed after it.
_STATE: dict = {"start_method": "exec", "supervisors": []}


@pytest.fixture(autouse=True, params=["exec", "forkserver"])
def start_method(request):
    _STATE["start_method"] = request.param
    _STATE["supervisors"] = []
    yield request.param
    for supervisor in _STATE["supervisors"]:
        supervisor.close()


def _new_supervisor(**kwargs) -> RolloutWorkerSupervisor:
    kwargs.setdefault("start_method", _STATE["start_method"])
    supervisor = RolloutWorkerSupervisor(**kwargs)
    _STATE["supervisors"].append(supervisor)
    return supervisor


def _request(scratch, request_id="task-r0-aaaaaaaa", **overrides) -> RolloutWorkerRequest:
    fields: dict[str, Any] = {"startup_budget_seconds": 20.0, "rollout_budget_seconds": 2.0, **overrides}
    return worker_request(scratch, request_id, **fields)


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
        "STUB_FIRST_CALL_PATH": os.path.join(str(scratch), f"{request_id}.res.first_call"),
        "STUB_GO_PATH": os.path.join(str(markers), "go"),
    }
    if response_blob:
        env["STUB_RESPONSE_SRC"] = response_blob
    kwargs.setdefault("termination_grace_seconds", 1.0)
    kwargs.setdefault("parent_grace_seconds", 2.0)
    kwargs.setdefault("ready_poll_seconds", 0.05)
    return (
        _new_supervisor(
            module=STUB_MODULE,
            scratch_dir=str(scratch),
            log_dir=str(tmp_path / "logs"),
            env_overrides=env,
            **kwargs,
        ),
        str(scratch),
        markers,
    )


def _running(pid: int) -> bool:
    """Whether a pid is a process that can still execute code.

    Deliberately not ``os.kill(pid, 0)``: an orphan whose parent we just killed
    is re-parented, and how quickly its exit status is collected depends on
    whoever inherited it -- so a *zombie* would read as alive and make this
    test a load-dependent flake.  A zombie has already been killed, which is
    the property under test.
    """

    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            raw = handle.read()
    except OSError:
        return False
    try:
        state = raw[raw.rindex(b")") + 2 : raw.rindex(b")") + 3]
    except ValueError:
        return False
    return state not in (b"Z", b"X", b"x")


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


def test_the_first_model_call_s_marker_fires_on_first_call_while_the_worker_runs_and_is_removed(tmp_path):
    """The start limit's signal under process isolation: a worker marks its rollout's first model call, then answers
    only once ``on_first_call`` has told it to. It completes, so ``on_first_call`` fired, once, while it ran; the marker
    is gone."""

    blob = _response_blob(tmp_path)
    supervisor, scratch, markers = _supervisor(tmp_path, "called_then_wait_for_go", response_blob=blob)
    request = _request(scratch, signal_first_call=True)
    fired: list[int] = []

    def on_first_call() -> None:
        fired.append(1)
        (markers / "go").touch()

    outcome = asyncio.run(supervisor.run(request, on_first_call=on_first_call))
    assert outcome.disposition == "completed" and outcome.killed_by_parent is False
    assert fired == [1] and not os.path.exists(request.first_call_path)


def test_a_cancelled_rollout_leaves_none_of_its_worker_s_files(tmp_path):
    """A member cancelled after its worker marked its first model call, as a straggler cut, its group's cancellation or
    a stop cancels it: once its worker is reaped, nothing of the worker's is left in the scratch directory, which
    outlives the run (its readiness file, its marker)."""

    supervisor, scratch, _markers = _supervisor(tmp_path, "called_then_wait_for_go")
    request = _request(scratch, signal_first_call=True, rollout_budget_seconds=30.0)
    fired: list[int] = []

    async def run() -> None:
        task = asyncio.ensure_future(supervisor.run(request, on_first_call=lambda: fired.append(1)))
        while not fired:
            await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(asyncio.wait_for(run(), 60))
    assert fired == [1] and [name for name in os.listdir(scratch) if name.startswith(request.request_id)] == []


def test_without_on_first_call_the_supervisor_never_looks_for_the_marker(tmp_path, monkeypatch):
    blob = _response_blob(tmp_path)
    supervisor, scratch, _markers = _supervisor(tmp_path, "called_then_ok", response_blob=blob)
    request = _request(scratch, signal_first_call=True)
    looked: list[str] = []
    exists = os.path.exists

    def counting(path) -> bool:
        if str(path).endswith(".first_call"):
            looked.append(str(path))
        return exists(path)

    monkeypatch.setattr(os.path, "exists", counting)
    outcome = asyncio.run(supervisor.run(request))
    monkeypatch.undo()
    assert outcome.disposition == "completed" and looked == []
    assert not os.path.exists(request.first_call_path)  # removed all the same: the request asked for it


def test_the_marker_watch_ends_with_the_worker_s_wait_and_counts_a_marker_left_as_it_exits(tmp_path):
    """``_watch_ready`` for a worker whose return code never comes (a forked worker whose fork server died): after
    readiness it watches for the marker until the worker's wait ends, and then looks once more."""

    supervisor = _new_supervisor(scratch_dir=str(tmp_path), ready_poll_seconds=0.01)
    request = _request(tmp_path, signal_first_call=True)
    Path(request.ready_path).touch()
    proc = SimpleNamespace(returncode=None)

    async def watch(marked_as_it_exits: bool) -> list[int]:
        fired: list[int] = []
        waiter = asyncio.get_running_loop().create_future()
        watcher = asyncio.ensure_future(
            supervisor._watch_ready(request, proc, time.monotonic(), {}, None, lambda: fired.append(1), waiter)
        )
        await asyncio.sleep(0.1)
        assert not watcher.done()  # still watching: no marker yet, and the worker runs
        if marked_as_it_exits:
            Path(request.first_call_path).touch()
        waiter.set_result(None)  # its wait ends, with no return code
        await asyncio.wait_for(watcher, 5)
        return fired

    assert asyncio.run(watch(marked_as_it_exits=False)) == []
    assert asyncio.run(watch(marked_as_it_exits=True)) == [1]


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
        spinner, spin_scratch, _m = _supervisor(tmp_path / "spin", "spin", request_id="spin-0")
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


def test_a_worker_killed_while_writing_events_leaves_no_temporary_file(tmp_path):
    supervisor, scratch, _markers = _supervisor(tmp_path, "events_then_killed")
    events = os.path.join(scratch, "events")  # the request's rollout output_dir is the scratch dir
    supervisor.env_overrides["STUB_EVENTS_DIR"] = events
    os.makedirs(events)
    others = [os.path.join(events, ".events_task_c2.jsonl.1.tmp"), os.path.join(events, "events_task_c0.jsonl")]
    for path in others:  # another process's temporary and a finished file
        with open(path, "w") as handle:
            handle.write("{}\n")

    outcome = asyncio.run(supervisor.run(_request(scratch)))

    assert outcome.response is None and outcome.returncode == -signal.SIGKILL
    assert sorted(os.listdir(events)) == sorted(os.path.basename(path) for path in others)


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
    supervisor, scratch, markers = _supervisor(tmp_path, "trap_sigterm", termination_grace_seconds=0.5)
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
    import time as _time

    for pid in grandchildren:
        deadline = _time.monotonic() + 30.0
        while _running(pid) and _time.monotonic() < deadline:
            _time.sleep(0.05)
        assert not _running(pid), f"grandchild {pid} outlived its worker"


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
    supervisor = _new_supervisor(
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
    from platoon.train.workers import supervisor as supervisor_module

    scratch = tmp_path / "ipc"
    scratch.mkdir()
    supervisor = _new_supervisor(scratch_dir=str(scratch), log_dir=str(tmp_path / "logs"), parent_grace_seconds=1.0)
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
    supervisor = _new_supervisor(
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
    # Inherit-plus-override: the trainer's environment survives.
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


# --- what an outcome says of its worker, and what the supervisor leaves behind ------------------------------------


def test_a_worker_its_parent_s_deadline_stops_is_a_hard_timeout_the_parent_killed(tmp_path):
    """No response and no marker of its own: the parent's deadline, its budget and the parent's grace, stopped it, and
    the outcome says how long it ran."""

    supervisor, scratch, _markers = _supervisor(tmp_path, "sleep", parent_grace_seconds=0.3)
    outcome = asyncio.run(supervisor.run(_request(scratch, rollout_budget_seconds=0.3)))

    assert (outcome.disposition, outcome.killed_by_parent) == ("hard_timeout", True)
    assert outcome.wall_seconds >= 0.6


def test_the_parent_gives_a_worker_its_grace_past_its_budget(tmp_path):
    """A worker that answers 0.4 s in, past its 0.2 s budget but within the parent's 2 s grace, completes."""

    blob = _response_blob(tmp_path)
    supervisor, scratch, markers = _supervisor(tmp_path, "called_then_wait_for_go", response_blob=blob)
    request = _request(scratch, signal_first_call=True, rollout_budget_seconds=0.2)

    def on_first_call() -> None:
        threading.Timer(0.4, (markers / "go").touch).start()

    outcome = asyncio.run(supervisor.run(request, on_first_call=on_first_call))

    assert (outcome.disposition, outcome.killed_by_parent) == ("completed", False)


def test_a_worker_launched_on_its_retry_says_it_was_retried(tmp_path, monkeypatch):
    from platoon.train.workers import supervisor as supervisor_module

    supervisor, scratch, _markers = _supervisor(tmp_path, "clean_no_response")
    dump = supervisor_module.dump_request
    attempts: list[str] = []

    def failing_once(request, path):
        attempts.append(request.request_id)
        if len(attempts) == 1:
            raise OSError(28, "No space left on device")
        return dump(request, path)

    monkeypatch.setattr(supervisor_module, "dump_request", failing_once)
    outcome = asyncio.run(supervisor.run(_request(scratch)))

    assert len(attempts) == 2 and outcome.returncode == 0  # the retry ran its worker
    assert outcome.retried is True


def _descriptors() -> int:
    return len(os.listdir("/proc/self/fd"))


def test_workers_that_ran_leave_no_descriptor_open_and_no_request_behind(tmp_path):
    """Each worker's log, and its pidfd where the supervisor holds one, are closed once it is reaped, and its request
    file is gone; the first worker starts what the supervisor keeps (its fork server)."""

    supervisor, scratch, _markers = _supervisor(tmp_path, "clean_no_response")

    async def run() -> tuple[int, int]:
        await supervisor.run(_request(scratch, request_id="task-r0-warmup00"))
        before = _descriptors()
        for index in range(3):
            await supervisor.run(_request(scratch, request_id=f"task-r{index}-cccccccc"))
        return before, _descriptors()

    before, after = asyncio.run(run())

    assert after == before
    assert not any(name.endswith(".req") for name in os.listdir(scratch))


def test_a_worker_that_never_read_its_request_leaves_no_request_behind(tmp_path):
    """The request carries the gateway's key: a worker that died before it read (and unlinked) it leaves it to the
    supervisor, which removes it as it reaps the worker."""

    scratch = tmp_path / "ipc"
    scratch.mkdir()
    supervisor = _new_supervisor(
        module="definitely_not_a_module_xyz",
        scratch_dir=str(scratch),
        log_dir=str(tmp_path / "logs"),
        parent_grace_seconds=2.0,
        ready_poll_seconds=0.05,
    )
    asyncio.run(supervisor.run(_request(str(scratch), rollout_budget_seconds=1.0)))

    assert not any(name.endswith(".req") for name in os.listdir(scratch))
