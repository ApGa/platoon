#!/usr/bin/env python3
"""Publish v9 Gate2 evidence in-job, then authorize it after PBS Exit 0."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import time
from pathlib import Path
from typing import Any

_VERIFIER_PATH = Path(__file__).with_name("verify_openreward_v9_gate2_result.py")
_SPEC = importlib.util.spec_from_file_location("verify_openreward_v9_gate2_result", _VERIFIER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load Gate2 verifier: {_VERIFIER_PATH}")
_VERIFIER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_VERIFIER)
BASE_KEYS = _VERIFIER.BASE_KEYS
RELEASE_REF = _VERIFIER.RELEASE_REF
RELEASE_ROOT = _VERIFIER.RELEASE_ROOT
parse_result = _VERIFIER.parse_result
validate_bound_evidence = _VERIFIER.validate_bound_evidence


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_private_parent(path: Path) -> None:
    if not path.is_absolute() or path.parent.resolve() != path.parent:
        raise RuntimeError("Gate2 publication path is not direct and absolute")
    value = path.parent.lstat()
    if not stat.S_ISDIR(value.st_mode) or value.st_uid != os.getuid() or stat.S_IMODE(value.st_mode) & 0o077:
        raise RuntimeError("Gate2 publication parent is not private/user-owned")


def publish_create_only(path: Path, value: dict[str, str], *, mode: int) -> None:
    require_private_parent(path)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing stale Gate2 publication path: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            for key in sorted(value):
                stream.write(f"{key}={value[key]}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RuntimeError(f"Gate2 publication target appeared: {path}") from exc
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def candidate(args: argparse.Namespace) -> dict[str, str]:
    if os.environ.get("PBS_JOBID") != args.pbs_job_id:
        raise RuntimeError("Gate2 candidate PBS job identity differs from environment")
    value = {
        "gate2": "passed_pending_pbs_accounting",
        "gate2_schema_version": "2",
        "release_flavor": "v9",
        "release_ref": RELEASE_REF,
        "release_commit": args.release_commit,
        "pbs_job_id": args.pbs_job_id,
        "worker_environment_key": args.worker_environment_key,
        "controller_environment_key": args.controller_environment_key,
        "source_sha256": args.source_sha256,
        "training_sif": str(args.training_sif.absolute()),
        "training_sif_sha256": args.training_sif_sha256,
        "worker_venv": str(args.worker_venv.absolute()),
        "controller_venv": str(args.controller_venv.absolute()),
        "toolathlon_sif": str(args.toolathlon_sif.absolute()),
        "toolathlon_sif_sha256": args.toolathlon_sif_sha256,
        "toolathlon_build_receipt": str(args.toolathlon_build_receipt.absolute()),
        "toolathlon_build_receipt_sha256": args.toolathlon_build_receipt_sha256,
        "toolathlon_portability_patch_sha256": args.toolathlon_portability_patch_sha256,
        "completed_at_epoch": str(int(time.time())),
    }
    if set(value) != BASE_KEYS:
        raise RuntimeError("internal Gate2 candidate schema mismatch")
    validate_bound_evidence(RELEASE_ROOT, value, expected_gate=value["gate2"])
    publish_create_only(args.candidate.absolute(), value, mode=0o600)
    return value


def _qstat_record(path: Path, job_id: str) -> tuple[dict[str, str], str]:
    if not path.is_absolute() or path.resolve() != path:
        raise RuntimeError("PBS accounting path is not direct and absolute")
    value = path.lstat()
    if (
        not stat.S_ISREG(value.st_mode)
        or value.st_uid != os.getuid()
        or stat.S_IMODE(value.st_mode) not in {0o400, 0o600}
    ):
        raise RuntimeError("PBS accounting is not a private user-owned regular file")
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("PBS accounting JSON is unreadable or malformed") from exc
    jobs = document.get("Jobs", document) if isinstance(document, dict) else None
    if not isinstance(jobs, dict):
        raise RuntimeError("PBS accounting JSON has no job map")
    matches = [item for key, item in jobs.items() if key == job_id or key.split(".", 1)[0] == job_id.split(".", 1)[0]]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise RuntimeError("PBS accounting does not uniquely contain the Gate2 job")
    record: dict[str, Any] = matches[0]
    normalized = {
        "pbs_state": str(record.get("job_state", "")),
        "pbs_substate": str(int(record.get("substate", -1))),
        "pbs_exit_status": str(int(record.get("Exit_status", -1))),
        "pbs_run_count": str(int(record.get("run_count", 0))),
    }
    expected = {
        "pbs_state": "F",
        "pbs_substate": "92",
        "pbs_exit_status": "0",
        "pbs_run_count": "1",
    }
    if normalized != expected:
        raise RuntimeError(f"Gate2 PBS accounting mismatch: {normalized} != {expected}")
    return normalized, sha256(path)


def finalize(args: argparse.Namespace) -> dict[str, str]:
    candidate_path = args.candidate.absolute()
    candidate_value = parse_result(candidate_path, modes={0o600})
    if set(candidate_value) != BASE_KEYS:
        raise RuntimeError("Gate2 candidate field set mismatch")
    validate_bound_evidence(
        RELEASE_ROOT,
        candidate_value,
        expected_gate="passed_pending_pbs_accounting",
    )
    pbs, accounting_sha = _qstat_record(args.qstat_json.absolute(), candidate_value["pbs_job_id"])
    value = dict(candidate_value)
    value["gate2"] = "passed"
    value.update(
        {
            "candidate_path": str(candidate_path),
            "candidate_sha256": sha256(candidate_path),
            "pbs_accounting_path": str(args.qstat_json.absolute()),
            "pbs_accounting_sha256": accounting_sha,
            **pbs,
            "finalized_at_epoch": str(int(time.time())),
        }
    )
    if int(value["finalized_at_epoch"]) < int(value["completed_at_epoch"]):
        raise RuntimeError("Gate2 finalization predates its candidate")
    publish_create_only(args.result.absolute(), value, mode=0o400)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    candidate_parser = commands.add_parser("candidate")
    candidate_parser.add_argument("--candidate", type=Path, required=True)
    candidate_parser.add_argument("--release-commit", required=True)
    candidate_parser.add_argument("--pbs-job-id", required=True)
    candidate_parser.add_argument("--worker-environment-key", required=True)
    candidate_parser.add_argument("--controller-environment-key", required=True)
    candidate_parser.add_argument("--source-sha256", required=True)
    candidate_parser.add_argument("--training-sif", type=Path, required=True)
    candidate_parser.add_argument("--training-sif-sha256", required=True)
    candidate_parser.add_argument("--worker-venv", type=Path, required=True)
    candidate_parser.add_argument("--controller-venv", type=Path, required=True)
    candidate_parser.add_argument("--toolathlon-sif", type=Path, required=True)
    candidate_parser.add_argument("--toolathlon-sif-sha256", required=True)
    candidate_parser.add_argument("--toolathlon-build-receipt", type=Path, required=True)
    candidate_parser.add_argument("--toolathlon-build-receipt-sha256", required=True)
    candidate_parser.add_argument("--toolathlon-portability-patch-sha256", required=True)
    finalize_parser = commands.add_parser("finalize")
    finalize_parser.add_argument("--candidate", type=Path, required=True)
    finalize_parser.add_argument("--result", type=Path, required=True)
    finalize_parser.add_argument("--qstat-json", type=Path, required=True)
    args = parser.parse_args(argv)
    value = candidate(args) if args.command == "candidate" else finalize(args)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
