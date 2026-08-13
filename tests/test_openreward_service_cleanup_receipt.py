from __future__ import annotations

import importlib.util
import json
import os
import socket
import stat
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "pbs-scripts/openreward-service-cleanup-receipt.py"
COMMON = REPO_ROOT / "pbs-scripts/openreward-ta-swe-prealloc-common.sh"
NODE_SERVICES = REPO_ROOT / "pbs-scripts/openreward-node-services.sh"


def _load():
    spec = importlib.util.spec_from_file_location("service_cleanup_receipt_test", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_receipt_is_create_only_private_and_exact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _load()
    root = tmp_path / "receipts"
    root.mkdir(mode=0o700)
    path = root / "node.json"
    host = socket.gethostname().split(".", 1)[0]
    monkeypatch.setenv("PBS_JOBID", "123.pbs1")
    helper.publish(path, job_id="123.pbs1", host=host, primary_status=143)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert helper.verify(path, job_id="123.pbs1", host=host, primary_status=143)["forced_kill"] is False
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
    assert '[[ "${wait_status}" -ne 1 ]]' in cleanup_body
    assert "service_shutdown_deadline=$((SECONDS + 180))" in cleanup_body
    assert "--primary-status 143" in cleanup_body
    assert "node-service task was already dead before requested cleanup" in common
    assert "local pbsdsh required forced KILL" in common
    assert "assert-tcp-closed" in common
    assert "assert-empty-swe" in common
    assert "node_services_remote_audit=passed" in common
    assert verify == -1
    node = NODE_SERVICES.read_text()
    assert "a service child was already dead before requested cleanup" in node
    assert "could not request service-child cleanup" in node
    assert "assert-tcp-closed" in node and "assert-empty-swe" in node
    assert '--primary-status "${primary_status}"' in node


def test_host_only_cleanup_helpers_are_executable() -> None:
    controller = REPO_ROOT / "pbs-scripts/openreward-controller-runtime-receipt.py"
    assert stat.S_IMODE(HELPER.stat().st_mode) == 0o755
    assert stat.S_IMODE(controller.stat().st_mode) == 0o755
