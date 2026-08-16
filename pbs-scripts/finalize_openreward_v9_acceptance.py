#!/usr/bin/env python3
"""Capture terminal PBS accounting and finalize one v9 acceptance gate.

The helper intentionally performs no submission or signaling.  It runs the
exact historical-accounting query only after the caller identifies a finished
job, validates F/substate 92/Exit 0/run_count 1 in memory, atomically publishes
the private raw JSON, then delegates to the gate-specific create-only finalizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

RELEASE_ROOT = Path("/home/aci18912it/rasp/releases/platoon-v8r13-376d1b9-18e62c0")
QSTAT = Path("/opt/pbs/bin/qstat")
QSTAT_SHA256 = "293e9861ccedf01b55e6cc46afea5fdd07630d3c8feceb1796ce2f2eb74a55ae"
QSTAT_VERSION = "pbs_version = 2022.1.6.20240517133149"
PROJECT_ROOT = Path("/groups/gai51740/aci18912it/rasp")
JOB_ID_RE = re.compile(r"[A-Za-z0-9_.-]+")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_trusted_qstat() -> None:
    value = QSTAT.lstat()
    mode = stat.S_IMODE(value.st_mode)
    if (
        not stat.S_ISREG(value.st_mode)
        or QSTAT.is_symlink()
        or value.st_uid not in {0, 65534}
        or mode & 0o022
        or not mode & 0o111
        or sha256(QSTAT) != QSTAT_SHA256
    ):
        raise RuntimeError("canonical PBS qstat identity mismatch")
    completed = subprocess.run(
        [str(QSTAT), "--version"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        env={"PATH": "/opt/pbs/bin:/usr/bin:/bin"},
    )
    if completed.returncode != 0 or completed.stdout.decode().strip() != QSTAT_VERSION:
        raise RuntimeError("canonical PBS qstat version mismatch")


def require_private_parent(path: Path) -> None:
    if not path.is_absolute() or path.parent.resolve() != path.parent:
        raise RuntimeError("acceptance evidence path is not direct and absolute")
    value = path.parent.lstat()
    if not stat.S_ISDIR(value.st_mode) or value.st_uid != os.getuid() or stat.S_IMODE(value.st_mode) & 0o077:
        raise RuntimeError("acceptance evidence parent is not private/user-owned")


def publish_bytes_create_only(path: Path, payload: bytes) -> None:
    require_private_parent(path)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing stale PBS accounting evidence: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o400)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RuntimeError(f"PBS accounting target appeared: {path}") from exc
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _job_record(document: dict[str, Any], job_id: str) -> dict[str, Any]:
    jobs = document.get("Jobs", document)
    if not isinstance(jobs, dict):
        raise RuntimeError("qstat JSON has no job map")
    matches = [item for key, item in jobs.items() if key == job_id or key.split(".", 1)[0] == job_id.split(".", 1)[0]]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise RuntimeError("qstat JSON does not uniquely contain the requested job")
    return matches[0]


def capture_qstat(job_id: str, output: Path) -> None:
    require_trusted_qstat()
    command = [str(QSTAT), "-x", "-f", "-F", "json", job_id]
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        env={"PATH": "/opt/pbs/bin:/usr/bin:/bin"},
    )
    if completed.returncode != 0:
        raise RuntimeError("terminal PBS accounting query failed")
    try:
        document = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("terminal PBS accounting query returned malformed JSON") from exc
    if not isinstance(document, dict):
        raise RuntimeError("terminal PBS accounting query returned a non-object")
    record = _job_record(document, job_id)
    actual = {
        "state": record.get("job_state"),
        "substate": int(record.get("substate", -1)),
        "exit_status": int(record.get("Exit_status", -1)),
        "run_count": int(record.get("run_count", 0)),
    }
    expected = {"state": "F", "substate": 92, "exit_status": 0, "run_count": 1}
    if actual != expected:
        raise RuntimeError(f"job is not one exact first-run PBS success: {actual}")
    publish_bytes_create_only(output, completed.stdout)


def run_finalizer(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=300,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": "/nonexistent",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(f"acceptance finalizer failed: {completed.stderr[-1000:]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gate", choices=("gate2", "all-mcp"))
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--gate2-result", type=Path)
    args = parser.parse_args()
    if JOB_ID_RE.fullmatch(args.job_id) is None:
        raise RuntimeError("PBS job id contains unsafe path characters")

    if args.gate == "gate2":
        root = PROJECT_ROOT / "cache/gate2-immutable-runtime-v9" / args.job_id
        publisher = RELEASE_ROOT / "pbs-scripts/publish_openreward_v9_gate2_result.py"
        extra: list[str] = []
    else:
        root = PROJECT_ROOT / "validation/toolathlon-all-mcp-v9" / args.job_id
        publisher = RELEASE_ROOT / "pbs-scripts/publish_toolathlon_all_mcp_health_result.py"
        cleanup = root / "node-services-cleanup.json"
        extra = ["--cleanup-receipt", str(cleanup)]
        if args.gate2_result is None:
            raise RuntimeError("all-MCP finalization requires the exact final Gate2 RESULT")
    candidate = root / "RESULT.candidate"
    accounting = root / "qstat-final.json"
    result = root / "RESULT"
    require_private_parent(candidate)
    if result.exists() or result.is_symlink() or accounting.exists() or accounting.is_symlink():
        raise RuntimeError("refusing stale acceptance finalization output")
    capture_qstat(args.job_id, accounting)
    run_finalizer(
        [
            "/usr/bin/python3",
            "-I",
            str(publisher),
            "finalize",
            "--candidate",
            str(candidate),
            "--result",
            str(result),
            "--qstat-json",
            str(accounting),
            *extra,
        ]
    )
    if args.gate == "gate2":
        verifier = RELEASE_ROOT / "pbs-scripts/verify_openreward_v9_gate2_result.py"
        verify_command = [
            "/usr/bin/python3",
            "-I",
            str(verifier),
            "--release-root",
            str(RELEASE_ROOT),
            "--result",
            str(result),
        ]
    else:
        verifier = RELEASE_ROOT / "pbs-scripts/verify_toolathlon_all_mcp_health_result.py"
        verify_command = [
            "/usr/bin/python3",
            "-I",
            str(verifier),
            "--release-root",
            str(RELEASE_ROOT),
            "--result",
            str(result),
            "--gate2-result",
            str(args.gate2_result.absolute()),
        ]
    run_finalizer(verify_command)
    print(f"openreward_v9_acceptance_finalized=PASS gate={args.gate} result={result}")


if __name__ == "__main__":
    main()
