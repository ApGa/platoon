"""The real worker process: isolation, environment invariants, and its own kill.

These launch ``python -m platoon.train.workers.entrypoint`` for real, so
they are slow (each worker imports the tinker/torch stack).  They are also the
only place the ordering guarantees in ``entrypoint.main`` are actually
observed from inside a worker, which is the point: every one of them is a
silent failure if it regresses.

No model is bound (``ModelBinding.model_name == ""``), so no gateway is
contacted and no tokenizer is loaded.
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

import pytest

import platoon.train.tinker  # noqa: F401  (import order: package before data processing)
from platoon.config_defs import RolloutConfig
from platoon.task import Task
from platoon.train.workers.entrypoint import HEADER_TEMPLATE  # noqa: F401
from platoon.train.workers.protocol import (
    PROTOCOL_VERSION,
    ConversionPolicy,
    ModelBinding,
    RolloutWorkerRequest,
    dump_request,
    load_response,
    read_diag,
)

STUB_DIR = os.path.dirname(os.path.abspath(__file__))
ENTRYPOINT = "platoon.train.workers.entrypoint"
HEADER_RE = re.compile(
    r"^\[rollout-worker\] request=(?P<request>\S+) pid=(?P<pid>\d+) pgid=(?P<pgid>\d+) "
    r"extra_allow=(?P<extra_allow>\d) telemetry=(?P<telemetry>\S+) "
    r"torch_threads=(?P<torch_threads>\d+) startup=(?P<startup>[\d.]+)s$",
    re.MULTILINE,
)


def _request(tmp_path, rollout_fn_ref, request_id="entry-0", **overrides):
    base = dict(
        protocol_version=PROTOCOL_VERSION,
        request_id=request_id,
        task_id="task",
        rollout_number=0,
        task=Task(goal="g", id="task"),
        rollout_config=asdict(RolloutConfig(output_dir=str(tmp_path))),
        rollout_fn_ref=rollout_fn_ref,
        model=ModelBinding(
            binding_path=str(tmp_path / "model_binding.pkl"),
            bootstrap_blob=b"",
            follow_poll_seconds=5.0,
            model_name="",  # no model bound: no gateway, no tokenizer
            renderer_name="",
        ),
        conversion=ConversionPolicy(),
        response_path=str(tmp_path / f"{request_id}.res"),
        ready_path=str(tmp_path / f"{request_id}.ready"),
        startup_budget_seconds=600.0,
        rollout_budget_seconds=600.0,
    )
    base.update(overrides)
    return RolloutWorkerRequest(**base)


def _launch(tmp_path, request, *, env_extra=None, timeout=600):
    request_path = str(tmp_path / f"{request.request_id}.req")
    dump_request(request, request_path)
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": os.pathsep.join([STUB_DIR, env.get("PYTHONPATH", "")]),
            "STUB_FACTS_PATH": str(tmp_path / f"{request.request_id}.facts.json"),
            "STUB_REQUEST_PATH": request_path,
            "HF_HUB_OFFLINE": "1",
            "OPENHANDS_SUPPRESS_BANNER": "1",
            # Deliberately hostile: telemetry ON and no thread pin, as a trainer's
            # environment may set them.  The worker must fix both itself.
            "TINKER_TELEMETRY": "1",
            "TINKER_SUBPROCESS_SAMPLING": "1",
            # What the supervisor passes so the startup deadline exists before
            # the child can read its request.
            "PLATOON_ROLLOUT_WORKER_STARTUP_BUDGET": repr(float(request.startup_budget_seconds)),
            "PLATOON_ROLLOUT_WORKER_DIAG": request.diag_path,
        }
    )
    env.pop("OMP_NUM_THREADS", None)
    env.pop("MKL_NUM_THREADS", None)
    env.update(env_extra or {})
    completed = subprocess.run(
        [sys.executable, "-m", ENTRYPOINT, request_path],
        capture_output=True,
        text=True,
        env=env,
        start_new_session=True,
        timeout=timeout,
    )
    return completed, request_path


@pytest.fixture(scope="module")
def clean_worker(tmp_path_factory):
    """One successful worker run, shared by the assertions that inspect it."""

    tmp_path = tmp_path_factory.mktemp("clean-worker")
    request = _request(tmp_path, "worker_stub_rollouts:report_and_finish")
    completed, request_path = _launch(tmp_path, request)
    facts_path = tmp_path / f"{request.request_id}.facts.json"
    facts = json.loads(facts_path.read_text()) if facts_path.exists() else {}
    return {
        "completed": completed,
        "request": request,
        "request_path": request_path,
        "facts": facts,
        "tmp_path": tmp_path,
    }


def test_worker_completes_and_writes_a_readable_response(clean_worker):
    completed = clean_worker["completed"]
    assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-4000:]
    response = load_response(clean_worker["request"].response_path)
    assert response.request_id == clean_worker["request"].request_id
    assert response.status == "no_trajectories"
    assert response.observed is False
    assert response.startup_seconds > 0.0


def test_worker_unlinks_the_request_that_carries_the_api_key(clean_worker):
    assert not os.path.exists(clean_worker["request_path"])
    # ... and it did so before running the rollout, not merely at exit.
    assert clean_worker["facts"]["request_file_exists"] is False


def test_worker_is_its_own_session_and_process_group_leader(clean_worker):
    facts = clean_worker["facts"]
    assert facts["is_session_leader"] is True
    assert facts["pgid"] == facts["pid"]


def test_worker_fixes_a_hostile_parent_environment(clean_worker):
    """Every one of these is a run-killer or a node-killer if it regresses."""

    facts = clean_worker["facts"]
    # Telemetry: a shadow sampling client with telemetry on can emit
    # SESSION_END, which the gateway reads as the client exiting -- it then
    # reclaims the session's models and slots and the training state dies.
    assert facts["TINKER_TELEMETRY"] == "0"
    # Thread pins: torch otherwise takes os.cpu_count() intra- and inter-op
    # threads per worker, on the node that also hosts the gateway.
    assert facts["OMP_NUM_THREADS"] == "1"
    assert facts["MKL_NUM_THREADS"] == "1"
    assert facts["torch_threads"] == 1
    assert facts["torch_interop_threads"] == 1
    assert facts["TOKENIZERS_PARALLELISM"] == "false"
    # The sampling sidecar cannot carry the extra='allow' patch, so R3's
    # routing_id would be dropped silently.
    assert facts["subprocess_sampling_present"] is False
    assert facts["PLATOON_TINKER_ROLLOUT_WORKER"] == "1"
    # R3: the patch is applied and verified inside the worker.
    assert facts["extra_allow"] is True


def test_worker_header_line_is_machine_readable(clean_worker):
    match = HEADER_RE.search(clean_worker["completed"].stdout)
    assert match, clean_worker["completed"].stdout[-4000:]
    assert match.group("request") == clean_worker["request"].request_id
    assert match.group("extra_allow") == "1"
    assert match.group("telemetry") == "0"
    assert match.group("torch_threads") == "1"
    assert int(match.group("pid")) == clean_worker["facts"]["pid"]
    assert float(match.group("startup")) > 0.0


def test_worker_signals_readiness_before_the_rollout(clean_worker):
    assert os.path.exists(clean_worker["request"].ready_path)


def test_a_spinning_rollout_is_killed_by_the_worker_s_own_alarm(tmp_path):
    """A runaway rollout is killed by its worker's own alarm.

    A model-authored ``while True`` cannot be killed in-thread -- asyncio
    cancellation abandons the coroutine and the thread underneath keeps running
    forever.  A process can be killed, and it leaves evidence first.
    """

    request = _request(
        tmp_path,
        "worker_stub_rollouts:spin_forever",
        request_id="spin-0",
        rollout_budget_seconds=6.0,
    )
    completed, _path = _launch(tmp_path, request)

    assert completed.returncode == -signal.SIGKILL
    assert not os.path.exists(request.response_path)
    diag = read_diag(request.diag_path)
    assert diag is not None, "a killed worker must leave a diagnostic marker"
    assert diag["phase"] == "rollout"
    # Every marker field is as of the monitor thread's last refresh, so this
    # is a lower bound rather than the exact kill time.
    assert float(diag["elapsed"]) >= 2.0
    # The cumulative cell-guard counters survive the kill, which is the only
    # way they can ever be non-zero for a worker that hung.
    assert diag["ptc_cells_started"] == "1"
    assert diag["ptc_cells_stuck_at_exit"] == "1"
    assert diag["stuck_cells"] == "while True: pass"


def test_an_overrun_startup_is_reported_as_a_startup_overrun(tmp_path):
    """The alarm is split in two on purpose.

    A single alarm covering imports and the rollout would let a cold-start
    storm silently eat the rollout's budget and turn a slow start into a hard
    timeout with no data and no explanation.
    """

    request = _request(
        tmp_path,
        "worker_stub_rollouts:report_and_finish",
        request_id="startup-0",
        startup_budget_seconds=0.75,
        rollout_budget_seconds=600.0,
    )
    completed, _path = _launch(tmp_path, request)

    assert completed.returncode == -signal.SIGKILL
    diag = read_diag(request.diag_path)
    assert diag is not None
    assert diag["phase"] == "startup"
    assert not os.path.exists(request.ready_path)


def test_killing_the_parent_kills_the_worker(tmp_path):
    """PDEATHSIG is the only thing that stops a watchdog kill orphaning workers.

    The trainer's watchdog and its double-signal handler both use ``os._exit``,
    which runs no cleanup, and the job launcher's signals never reach workers at
    all because they live in their own session.
    """

    request = _request(
        tmp_path,
        "worker_stub_rollouts:sleep_forever",
        request_id="orphan-0",
        rollout_budget_seconds=600.0,
    )
    request_path = str(tmp_path / f"{request.request_id}.req")
    dump_request(request, request_path)

    helper = (
        "import os, subprocess, sys, time\n"
        f"env = dict(os.environ)\n"
        f"env['PYTHONPATH'] = {os.pathsep.join([STUB_DIR, os.environ.get('PYTHONPATH', '')])!r}\n"
        f"env['STUB_FACTS_PATH'] = {str(tmp_path / 'orphan.facts.json')!r}\n"
        "env['HF_HUB_OFFLINE'] = '1'\n"
        "env['OPENHANDS_SUPPRESS_BANNER'] = '1'\n"
        f"child = subprocess.Popen([sys.executable, '-m', {ENTRYPOINT!r}, {request_path!r}],\n"
        "    env=env, start_new_session=True,\n"
        "    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        f"open({str(tmp_path / 'worker.pid')!r}, 'w').write(str(child.pid))\n"
        "time.sleep(3600)\n"
    )
    parent = subprocess.Popen([sys.executable, "-c", helper], start_new_session=True)
    try:
        pid_file = tmp_path / "worker.pid"
        facts_file = tmp_path / "orphan.facts.json"
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline and not facts_file.exists():
            time.sleep(0.5)
        assert facts_file.exists(), "the worker never reached its rollout"
        worker_pid = int(pid_file.read_text())

        parent.kill()
        parent.wait(timeout=30)

        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                os.kill(worker_pid, 0)
            except OSError:
                break
            time.sleep(0.25)
        else:
            os.kill(worker_pid, signal.SIGKILL)
            pytest.fail("the worker outlived its parent: PDEATHSIG did not fire")
    finally:
        if parent.poll() is None:
            parent.kill()


def test_entrypoint_rejects_a_bad_invocation():
    completed = subprocess.run(
        [sys.executable, "-m", ENTRYPOINT],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 2
    assert "usage:" in completed.stderr


def test_response_pickle_is_protocol_5(clean_worker):
    with open(clean_worker["request"].response_path, "rb") as handle:
        header = handle.read(2)
    assert header[0:1] == b"\x80" and header[1] == 5


@pytest.mark.parametrize("packages", [("eval_pkg.harnesses", "eval_pkg.programs"), ()], ids=["packages", "none"])
def test_worker_imports_the_environments_packages_before_its_rollout(tmp_path, packages):
    """``environments[].package`` registers harnesses and programs by being imported, in the trainer; a worker
    (``rollout_isolation: process``) imports it too, or the rollout's harness and program are unknown there."""

    tests_dir = os.path.dirname(os.path.dirname(STUB_DIR))  # tests/, where eval_pkg lives
    request = _request(
        tmp_path, "worker_stub_rollouts:report_registrations", request_id=f"pkg-{len(packages)}", packages=packages
    )
    pythonpath = os.pathsep.join([STUB_DIR, tests_dir, os.environ.get("PYTHONPATH", "")])
    completed, _ = _launch(tmp_path, request, env_extra={"PYTHONPATH": pythonpath})

    assert completed.returncode == 0, completed.stderr[-3000:]
    known = json.loads((tmp_path / f"{request.request_id}.facts.json").read_text())
    expected = bool(packages)
    assert known == {"harness": expected, "program": expected}
