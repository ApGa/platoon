"""Tests for deterministic post-F v9 acceptance finalization."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "pbs-scripts/finalize_openreward_v9_acceptance.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("finalize_openreward_v9_acceptance", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qstat_capture_uses_exact_history_query_and_atomic_private_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helper = _load_helper()
    output = tmp_path / "private" / "qstat-final.json"
    output.parent.mkdir(mode=0o700)
    payload = json.dumps(
        {
            "Jobs": {
                "1234.pbs1": {
                    "job_state": "F",
                    "substate": 92,
                    "Exit_status": 0,
                    "run_count": 1,
                }
            }
        }
    ).encode()
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=payload, stderr=b"")

    monkeypatch.setattr(helper, "require_trusted_qstat", lambda: None)
    monkeypatch.setattr(helper.subprocess, "run", fake_run)

    helper.capture_qstat("1234.pbs1", output)

    assert calls == [["/opt/pbs/bin/qstat", "-x", "-f", "-F", "json", "1234.pbs1"]]
    assert output.read_bytes() == payload
    assert output.stat().st_mode & 0o777 == 0o400


def test_qstat_capture_rejects_nonterminal_or_failed_job_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helper = _load_helper()
    output = tmp_path / "private" / "qstat-final.json"
    output.parent.mkdir(mode=0o700)
    payload = json.dumps(
        {
            "Jobs": {
                "1234.pbs1": {
                    "job_state": "R",
                    "substate": 42,
                    "Exit_status": 0,
                    "run_count": 1,
                }
            }
        }
    ).encode()
    monkeypatch.setattr(helper, "require_trusted_qstat", lambda: None)
    monkeypatch.setattr(
        helper.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout=payload, stderr=b""),
    )

    with pytest.raises(RuntimeError, match="not one exact first-run PBS success"):
        helper.capture_qstat("1234.pbs1", output)
    assert not output.exists()


def test_helper_is_executable_has_no_submission_and_scrubs_finalizer_environment() -> None:
    source = HELPER.read_text()

    assert HELPER.stat().st_mode & 0o111
    assert (
        'RELEASE_ROOT = Path("/home/aci18912it/rasp/releases/platoon-v8r18-376d1b9-18e62c0")'
        in source
    )
    assert 'QSTAT = Path("/opt/pbs/bin/qstat")' in source
    assert '[str(QSTAT), "-x", "-f", "-F", "json", job_id]' in source
    assert '"GIT_CONFIG_GLOBAL": "/dev/null"' in source
    assert "**os.environ" not in source
    assert "qsub" not in source.lower()
    assert "kill(" not in source
