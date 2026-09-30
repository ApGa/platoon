import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import uuid

import pytest

_PATH = Path(__file__).resolve().parents[1] / "pbs-scripts/cleanup_frontier_native_services.py"
_SPEC = importlib.util.spec_from_file_location("frontier_native_cleanup", _PATH)
cleanup = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cleanup)


def marker(root, process, token):
    session = root / "judge-session"
    session.mkdir()
    path = root / "native.json"
    identity = cleanup.process_identity(process.pid)
    assert identity is not None
    path.write_text(json.dumps({**identity, "version": 1, "hostname": socket.gethostname(),
                              "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                              "service_instance_id": token, "session_path": str(session)}))
    return path


def test_cleanup_stops_only_the_recorded_service(tmp_path):
    token = uuid.uuid4().hex
    with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                          start_new_session=True,
                          env={**os.environ, "FRONTIER_NATIVE_SERVICE_ID": token}) as owned:
        with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                              start_new_session=True) as unrelated:
            try:
                path = marker(tmp_path, owned, token)
                cleanup.stop_record(path, root=tmp_path, grace_seconds=0.2)
                assert owned.wait(timeout=2) < 0
                assert unrelated.poll() is None
                assert not path.exists()
            finally:
                unrelated.terminate()
                unrelated.wait(timeout=2)


def test_reused_pid_is_never_signaled(tmp_path):
    with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                          start_new_session=True) as process:
        try:
            path = marker(tmp_path, process, "unused")
            data = json.loads(path.read_text())
            data["start_time"] = str(int(data["start_time"]) - 1)
            path.write_text(json.dumps(data))
            with pytest.raises(RuntimeError, match="reused"):
                cleanup.stop_record(path, root=tmp_path, grace_seconds=0.1)
            assert process.poll() is None
            assert path.exists()
        finally:
            process.terminate()
            process.wait(timeout=2)


def test_unproven_orphan_is_not_signaled(tmp_path, monkeypatch):
    identity = {"pid": 12345, "pgid": 12345, "uid": os.getuid(),
                "start_time": "1", "executable": "/test/judge"}
    session = tmp_path / "judge"
    session.mkdir()
    path = tmp_path / "native.json"
    path.write_text(json.dumps({**identity, "version": 1, "hostname": socket.gethostname(),
                              "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                              "service_instance_id": "secret", "session_path": str(session)}))
    monkeypatch.setattr(cleanup, "process_identity", lambda pid: None)
    monkeypatch.setattr(cleanup, "group_members", lambda pgid: [{**identity, "pid": 12346}])
    monkeypatch.setattr(cleanup, "has_service_token", lambda pid, token: False)
    monkeypatch.setattr(cleanup.os, "kill", lambda *args: pytest.fail("Must not signal an unproven process"))
    with pytest.raises(RuntimeError, match="Cannot prove ownership"):
        cleanup.stop_record(path, root=tmp_path, grace_seconds=0.1)
    assert path.exists()
