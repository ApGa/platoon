from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import socket
import stat
import subprocess
import textwrap
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "pbs-scripts/openreward-service-cleanup-receipt.py"
REQUEST_HELPER = REPO_ROOT / "pbs-scripts/openreward-service-shutdown-request.py"
COMMON = REPO_ROOT / "pbs-scripts/openreward-ta-swe-prealloc-common.sh"
NODE_SERVICES = REPO_ROOT / "pbs-scripts/openreward-node-services.sh"


def _load():
    spec = importlib.util.spec_from_file_location("service_cleanup_receipt_test", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_request_helper():
    spec = importlib.util.spec_from_file_location("service_shutdown_request_test", REQUEST_HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shutdown_request_is_create_only_private_exact_and_fsynced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helper = _load_request_helper()
    root = tmp_path / "requests"
    root.mkdir(mode=0o700)
    path = root / "node.json"
    host = socket.gethostname().split(".", 1)[0]
    monkeypatch.setenv("PBS_JOBID", "123.pbs1")
    helper.publish(path, job_id="123.pbs1", host=host)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert helper.verify(path, job_id="123.pbs1", host=host) == {
        "schema_version": 1,
        "action": "shutdown",
        "reason": "natural_success",
        "pbs_job_id": "123.pbs1",
        "host": host,
    }
    with pytest.raises(RuntimeError, match="stale"):
        helper.publish(path, job_id="123.pbs1", host=host)
    with pytest.raises(RuntimeError, match="PBS job"):
        helper.verify(path, job_id="wrong.pbs1", host=host)
    with pytest.raises(RuntimeError, match="host identity"):
        helper.verify(path, job_id="123.pbs1", host="wrong-host")


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ({"extra": True}, "content mismatch"),
        ({"pbs_job_id": "wrong.pbs1"}, "content mismatch"),
        ({"host": "wrong-host"}, "content mismatch"),
    ),
)
def test_shutdown_request_rejects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: dict,
    message: str,
) -> None:
    helper = _load_request_helper()
    root = tmp_path / "requests"
    root.mkdir(mode=0o700)
    path = root / "node.json"
    host = socket.gethostname().split(".", 1)[0]
    monkeypatch.setenv("PBS_JOBID", "123.pbs1")
    helper.publish(path, job_id="123.pbs1", host=host)
    value = json.loads(path.read_text())
    value.update(mutation)
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    with pytest.raises(RuntimeError, match=message):
        helper.verify(path, job_id="123.pbs1", host=host)


def test_shutdown_request_rejects_symlink_and_permissive_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helper = _load_request_helper()
    host = socket.gethostname().split(".", 1)[0]
    monkeypatch.setenv("PBS_JOBID", "123.pbs1")
    root = tmp_path / "requests"
    root.mkdir(mode=0o700)
    target = root / "target.json"
    target.write_text("{}")
    target.chmod(0o600)
    link = root / "node.json"
    link.symlink_to(target)
    with pytest.raises(RuntimeError, match="indirect"):
        helper.verify(link, job_id="123.pbs1", host=host)
    link.unlink()
    for mode in (0o750, 0o770):
        root.chmod(mode)
        with pytest.raises(RuntimeError, match="private and user-owned"):
            helper.publish(link, job_id="123.pbs1", host=host)


def test_shutdown_request_accepts_private_setgid_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _load_request_helper()
    host = socket.gethostname().split(".", 1)[0]
    monkeypatch.setenv("PBS_JOBID", "123.pbs1")
    root = tmp_path / "requests"
    root.mkdir(mode=0o700)
    root.chmod(0o2700)
    path = root / "node.json"
    helper.publish(path, job_id="123.pbs1", host=host)
    assert stat.S_IMODE(root.stat().st_mode) == 0o2700
    assert helper.verify(path, job_id="123.pbs1", host=host)["host"] == host


def test_shutdown_request_rejects_foreign_owner_malformed_json_and_publish_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helper = _load_request_helper()
    host = socket.gethostname().split(".", 1)[0]
    monkeypatch.setenv("PBS_JOBID", "123.pbs1")
    root = tmp_path / "requests"
    root.mkdir(mode=0o700)
    path = root / "node.json"
    actual_uid = os.getuid()
    monkeypatch.setattr(helper.os, "getuid", lambda: actual_uid + 1)
    with pytest.raises(RuntimeError, match="user-owned"):
        helper.publish(path, job_id="123.pbs1", host=host)
    monkeypatch.setattr(helper.os, "getuid", lambda: actual_uid)
    path.write_text("{")
    path.chmod(0o600)
    with pytest.raises(RuntimeError, match="malformed"):
        helper.verify(path, job_id="123.pbs1", host=host)
    path.unlink()
    original_link = helper.os.link

    def racing_link(source, target):
        Path(target).write_text("racer")
        Path(target).chmod(0o600)
        return original_link(source, target)

    monkeypatch.setattr(helper.os, "link", racing_link)
    with pytest.raises(RuntimeError, match="appeared during create-only"):
        helper.publish(path, job_id="123.pbs1", host=host)
    assert path.read_text() == "racer"


def test_request_driven_supervisor_reaps_both_live_children_and_publishes_status_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Process-level model of the node supervisor's request/cleanup contract."""

    request_helper = _load_request_helper()
    receipt_helper = _load()
    root = tmp_path / "handshake"
    root.mkdir(mode=0o700)
    request = root / "request.json"
    receipt = root / "receipt.json"
    events = root / "events"
    host = socket.gethostname().split(".", 1)[0]
    job_id = "123.pbs1"
    monkeypatch.setenv("PBS_JOBID", job_id)
    child = textwrap.dedent(
        """
        import signal, sys, time
        output, name = sys.argv[1:]
        def stop(_signal, _frame):
            with open(output, "a") as stream:
                stream.write(f"{name}:TERM\\n")
            raise SystemExit(0)
        signal.signal(signal.SIGTERM, stop)
        with open(output, "a") as stream:
            stream.write(f"{name}:READY\\n")
        while True:
            time.sleep(0.05)
        """
    )
    children = [
        subprocess.Popen([os.sys.executable, "-c", child, str(events), name]) for name in ("toolathlon", "swe-rebench")
    ]
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if events.exists() and events.read_text().count(":READY") == 2:
                break
            time.sleep(0.02)
        assert all(child_process.poll() is None for child_process in children)

        request_helper.publish(request, job_id=job_id, host=host)
        request_helper.verify(request, job_id=job_id, host=host)
        assert all(child_process.poll() is None for child_process in children)
        for child_process in children:
            child_process.terminate()
        exits = [child_process.wait(timeout=5) for child_process in children]
        assert exits == [0, 0]
        receipt_helper.publish(receipt, job_id=job_id, host=host, primary_status=0)
        value = receipt_helper.verify(receipt, job_id=job_id, host=host, primary_status=0)
        assert value["shutdown_request_verified"] is True
        assert value["forced_kill"] is False
        assert sorted(line for line in events.read_text().splitlines() if line.endswith(":TERM")) == [
            "swe-rebench:TERM",
            "toolathlon:TERM",
        ]
    finally:
        for child_process in children:
            if child_process.poll() is None:
                child_process.kill()
                child_process.wait(timeout=5)


def test_actual_node_supervisor_observes_request_with_both_fake_children_live(
    tmp_path: Path,
) -> None:
    try:
        probe = socket.socket()
    except PermissionError:
        pytest.skip("sandbox forbids TCP closure probes; run this test unrestricted")
    else:
        probe.close()
    root = tmp_path / "node-supervisor"
    root.mkdir(mode=0o700)
    fixture_scripts = root / "pbs-scripts"
    fixture_scripts.mkdir()
    fixture_supervisor = fixture_scripts / NODE_SERVICES.name
    shutil.copy2(NODE_SERVICES, fixture_supervisor)
    shutil.copy2(HELPER, fixture_scripts / HELPER.name)
    shutil.copy2(REQUEST_HELPER, fixture_scripts / REQUEST_HELPER.name)
    shutil.copy2(
        REPO_ROOT / "pbs-scripts/cleanup_swe_singularity_sessions.py",
        fixture_scripts / "cleanup_swe_singularity_sessions.py",
    )
    fake_repo = root / "repo"
    script_root = fake_repo / "plugins/openreward/scripts"
    script_root.mkdir(parents=True)
    entrypoint = script_root / "openreward-toolathlon-resilient-entrypoint.sh"
    stager = script_root / "openreward-toolathlon-stage-pgdata.sh"
    for script in (entrypoint, stager):
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)

    events = root / "events"
    runtime = root / "fake-runtime"
    runtime.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            set -eu
            command_name=${{1:-}}
            shift || true
            if [[ "${{command_name}}" == inspect ]]; then exit 0; fi
            if [[ " $* " == *" --fakeroot "* ]]; then exit 0; fi
            printf 'toolathlon:READY\\n' >>{events}
            trap 'printf "toolathlon:TERM\\n" >>{events}; exit 0' TERM INT
            while :; do sleep 0.05; done
            """
        )
    )
    runtime.chmod(0o755)
    mkfs = root / "fake-mkfs"
    mkfs.write_text("#!/bin/sh\nexit 0\n")
    mkfs.chmod(0o755)
    swe = fixture_scripts / "openreward-swe-singularity-server.sh"
    swe.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            set -eu
            printf 'swe-rebench:READY\\n' >>{events}
            trap 'printf "swe-rebench:TERM\\n" >>{events}; exit 0' TERM INT
            while :; do sleep 0.05; done
            """
        )
    )
    swe.chmod(0o755)
    image = root / "toolathlon.sif"
    image.write_bytes(b"fake-image-for-supervisor-test")
    image_sha = hashlib.sha256(image.read_bytes()).hexdigest()
    abci_env = root / "abci.env"
    abci_env.write_text("set +x\n")

    local_root = root / "local"
    local_root.mkdir(mode=0o700)
    request = root / "shutdown-request.json"
    receipt = root / "cleanup-receipt.json"
    output = root / "supervisor.log"
    host = socket.gethostname().split(".", 1)[0]
    job_id = "123.pbs1"

    # The fake children never bind these high ports. Avoid relying on socket
    # creation, which is intentionally blocked in some unit-test sandboxes.
    first_port = 65431
    second_port = 65432
    environment = dict(os.environ)
    for name in list(environment):
        if name.startswith("BASH_FUNC_") and "module" in name:
            environment.pop(name)
    environment.update(
        {
            "PBS_JOBID": job_id,
            "PBS_LOCALDIR": str(local_root),
            "PLATOON_ABCI_ENV_FILE": str(abci_env),
            "PLATOON_REPO_ROOT": str(fake_repo),
            "PLATOON_CONTAINER_RUNTIME": str(runtime),
            "PLATOON_MKFS_EXT3_BIN": str(mkfs),
            "PLATOON_NODE_LOCAL_ROOT": str(local_root),
            "PLATOON_NODE_SERVICES_CLEANUP_RECEIPT": str(receipt),
            "PLATOON_NODE_SERVICES_SHUTDOWN_REQUEST": str(request),
            "PLATOON_NODE_SERVICES_EXPECTED_HOST": host,
            "PLATOON_NODE_SERVICES_EXPECTED_JOB_ID": job_id,
            "OPENREWARD_TOOLATHLON_SIF": str(image),
            "OPENREWARD_TOOLATHLON_SIF_SHA256": image_sha,
            "OPENREWARD_TOOLATHLON_OVERLAY_SIZE_GB": "1",
            "PLATOON_STAGE_TOOLATHLON_SIF": "0",
            "PLATOON_STAGE_TRAINING_SIF": "0",
            "OPENREWARD_PORT": str(first_port),
            "OPENREWARD_SWE_REBENCH_PORT": str(second_port),
            "SINGULARITY_CACHEDIR": str(local_root / "singularity-cache"),
            "SINGULARITY_TMPDIR": str(local_root / "singularity-tmp"),
            "SWE_APPTAINER_SESSION_ROOT": str(local_root / "swe-sessions"),
        }
    )
    with output.open("w") as output_stream:
        supervisor = subprocess.Popen(
            ["/bin/bash", str(fixture_supervisor)],
            env=environment,
            stdout=output_stream,
            stderr=subprocess.STDOUT,
        )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            ready = events.read_text().count(":READY") if events.exists() else 0
            if ready == 2:
                break
            if supervisor.poll() is not None:
                raise AssertionError(output.read_text())
            time.sleep(0.05)
        assert ready == 2
        assert supervisor.poll() is None
        request_helper = _load_request_helper()
        old_job = os.environ.get("PBS_JOBID")
        os.environ["PBS_JOBID"] = job_id
        try:
            request_helper.publish(request, job_id=job_id, host=host)
        finally:
            if old_job is None:
                os.environ.pop("PBS_JOBID", None)
            else:
                os.environ["PBS_JOBID"] = old_job
        assert supervisor.wait(timeout=20) == 0, output.read_text()
        receipt_helper = _load()
        value = receipt_helper.verify(receipt, job_id=job_id, host=host, primary_status=0)
        assert value["shutdown_request_verified"] is True
        assert value["forced_kill"] is False
        lines = events.read_text().splitlines()
        assert lines.count("toolathlon:TERM") == 1
        assert lines.count("swe-rebench:TERM") == 1
        supervisor_log = output.read_text()
        assert "verified natural shutdown request" in supervisor_log
        assert "forced_KILL=0" in supervisor_log
        assert "forced_KILL=1" not in supervisor_log
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait(timeout=5)


def test_receipt_is_create_only_private_and_exact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _load()
    root = tmp_path / "receipts"
    root.mkdir(mode=0o700)
    path = root / "node.json"
    host = socket.gethostname().split(".", 1)[0]
    monkeypatch.setenv("PBS_JOBID", "123.pbs1")
    helper.publish(path, job_id="123.pbs1", host=host, primary_status=143)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    value = helper.verify(path, job_id="123.pbs1", host=host, primary_status=143)
    assert value["schema_version"] == 2
    assert value["forced_kill"] is False
    assert value["shutdown_request_verified"] is False
    with pytest.raises(RuntimeError, match="stale"):
        helper.publish(path, job_id="123.pbs1", host=host, primary_status=143)
    with pytest.raises(RuntimeError, match="content mismatch"):
        helper.verify(path, job_id="123.pbs1", host=host, primary_status=1)
    value = json.loads(path.read_text())
    value["host"] = "wrong"
    path.write_text(json.dumps(value))
    os.chmod(path, 0o600)
    with pytest.raises(RuntimeError, match="content mismatch"):
        helper.verify(path, job_id="123.pbs1", host=host, primary_status=143)


def test_receipt_publication_cannot_overwrite_racing_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _load()
    root = tmp_path / "receipts"
    root.mkdir(mode=0o700)
    path = root / "node.json"
    host = socket.gethostname().split(".", 1)[0]
    monkeypatch.setenv("PBS_JOBID", "123.pbs1")
    original_link = helper.os.link

    def racing_link(source, target):
        Path(target).write_text("racer")
        os.chmod(target, 0o600)
        return original_link(source, target)

    monkeypatch.setattr(helper.os, "link", racing_link)
    with pytest.raises(RuntimeError, match="appeared during create-only"):
        helper.publish(path, job_id="123.pbs1", host=host, primary_status=143)
    assert path.read_text() == "racer"


def test_tcp_closure_requires_three_stable_refusals(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _load()
    attempts = iter((ConnectionRefusedError(), object()))

    class _Connection:
        def close(self):
            pass

    def unstable(*_args, **_kwargs):
        value = next(attempts)
        if isinstance(value, BaseException):
            raise value
        return _Connection()

    monkeypatch.setattr(helper.socket, "create_connection", unstable)
    with pytest.raises(RuntimeError, match="still accepting"):
        helper.tcp_closed("host", [1234], 0.01, 0.01)
    monkeypatch.setattr(
        helper.socket,
        "create_connection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ConnectionRefusedError()),
    )
    helper.tcp_closed("host", [1234], 0.01, 1)
    monkeypatch.setattr(
        helper.socket,
        "create_connection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError()),
    )
    with pytest.raises(RuntimeError, match="probe failed"):
        helper.tcp_closed("host", [1234], 0.01, 1)


@pytest.mark.parametrize("kind", ("directory", "file", "symlink", "fifo"))
def test_swe_scan_rejects_every_matching_entry(tmp_path: Path, kind: str) -> None:
    helper = _load()
    root = tmp_path / "sessions"
    root.mkdir()
    helper.assert_empty_swe(root)
    survivor = root / "swe-rebench-survivor"
    if kind == "directory":
        survivor.mkdir()
    elif kind == "file":
        survivor.write_text("x")
    elif kind == "symlink":
        survivor.symlink_to("missing")
    else:
        os.mkfifo(survivor)
    with pytest.raises(RuntimeError, match="survived cleanup"):
        helper.assert_empty_swe(root)


def test_common_service_cleanup_is_fanout_fail_closed_and_receipt_bound() -> None:
    common = COMMON.read_text()
    request = common.index("request_all_service_shutdowns()")
    verify = common.index("verify_service_shutdown", request) if "verify_service_shutdown" in common else -1
    cleanup = common.index("cleanup()", request)
    cleanup_body = common[cleanup : common.index("handle_signal()", cleanup)]
    assert "request_all_service_shutdowns" in cleanup_body
    assert cleanup_body.index("request_all_service_shutdowns") < cleanup_body.index('wait "${pid}"')
    assert "publish_all_service_shutdown_requests" in cleanup_body
    assert cleanup_body.index("pbs_cleanup_barrier.py") < cleanup_body.index("publish_all_service_shutdown_requests")
    assert "expected_pbsdsh_status=0" in cleanup_body
    assert "expected_primary_status=0" in cleanup_body
    assert "expected_pbsdsh_status=1" in cleanup_body
    assert "expected_primary_status=143" in cleanup_body
    assert '[[ "${wait_status}" != "${expected_pbsdsh_status}" ]]' in cleanup_body
    assert "service_shutdown_deadline=$((SECONDS + 180))" in cleanup_body
    assert '--primary-status "${expected_primary_status}"' in cleanup_body
    assert "node-service task was already dead before requested cleanup" in common
    assert "local pbsdsh required forced KILL" in common
    assert "assert-tcp-closed" in common
    assert "assert-empty-swe" in common
    assert "node_services_remote_audit=passed" in common
    assert '"PLATOON_NODE_SERVICES_SHUTDOWN_REQUEST=${service_shutdown_request}"' in common
    assert verify == -1
    node = NODE_SERVICES.read_text()
    assert "was already dead before requested cleanup" in node
    assert "could not send TERM to service child" in node
    assert "assert-tcp-closed" in node and "assert-empty-swe" in node
    assert '--primary-status "${primary_status}"' in node
    assert "natural_shutdown_requested=1" in node
    assert '"${SHUTDOWN_REQUEST_HELPER}" verify' in node
    assert "cleanup-child name=${name}" in node
    assert "forced_KILL=${child_forced_kill}" in node


def test_host_only_cleanup_helpers_are_executable() -> None:
    controller = REPO_ROOT / "pbs-scripts/openreward-controller-runtime-receipt.py"
    shutdown_request = REPO_ROOT / "pbs-scripts/openreward-service-shutdown-request.py"
    assert stat.S_IMODE(HELPER.stat().st_mode) == 0o755
    assert stat.S_IMODE(controller.stat().st_mode) == 0o755
    assert stat.S_IMODE(shutdown_request.stat().st_mode) == 0o755
