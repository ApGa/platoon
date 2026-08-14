#!/usr/bin/env python3
"""Verify finalized post-PBS v9 immutable-runtime Gate2 evidence."""

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

RELEASE_ROOT = Path("/home/aci18912it/rasp/releases/platoon-v8r4-376d1b9-18e62c0")
RELEASE_REF = "refs/tags/abci-v8r4-376d1b9-18e62c0"
ENV_ADAPTER_RELATIVE = Path("pbs-scripts/prepare_openreward_env.sh")
HEX64 = re.compile(r"[0-9a-f]{64}")
BASE_KEYS = {
    "gate2",
    "gate2_schema_version",
    "release_flavor",
    "release_ref",
    "release_commit",
    "pbs_job_id",
    "worker_environment_key",
    "controller_environment_key",
    "source_sha256",
    "training_sif",
    "training_sif_sha256",
    "worker_venv",
    "controller_venv",
    "toolathlon_sif",
    "toolathlon_sif_sha256",
    "toolathlon_build_receipt",
    "toolathlon_build_receipt_sha256",
    "toolathlon_portability_patch_sha256",
    "completed_at_epoch",
}
FINAL_KEYS = BASE_KEYS | {
    "candidate_path",
    "candidate_sha256",
    "pbs_accounting_path",
    "pbs_accounting_sha256",
    "pbs_state",
    "pbs_substate",
    "pbs_exit_status",
    "pbs_run_count",
    "finalized_at_epoch",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["/usr/bin/git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        env={"PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1"},
    ).stdout.strip()


def require_regular(path: Path, *, modes: set[int]) -> os.stat_result:
    if not path.is_absolute() or path.resolve() != path:
        raise RuntimeError(f"Gate2 evidence is not a direct absolute path: {path}")
    value = path.lstat()
    mode = stat.S_IMODE(value.st_mode)
    if not stat.S_ISREG(value.st_mode) or value.st_uid != os.getuid() or mode not in modes:
        raise RuntimeError(f"Gate2 evidence is not an owner-private regular file: {path}")
    return value


def parse_result(path: Path, *, modes: set[int] = {0o400}) -> dict[str, str]:
    require_regular(path, modes=modes)
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            raise RuntimeError("Gate2 RESULT is malformed or has duplicate fields")
        values[key] = value
    return values


def manifest_value(path: Path, key: str) -> str:
    matches = [line.removeprefix(f"{key}=") for line in path.read_text().splitlines() if line.startswith(f"{key}=")]
    if len(matches) != 1:
        raise RuntimeError(f"immutable environment manifest lacks exact {key}")
    return matches[0]


def require_immutable_environment(path: Path) -> None:
    if not path.is_absolute() or path.resolve() != path:
        raise RuntimeError(f"Gate2 environment is not a direct absolute path: {path}")
    value = path.lstat()
    if not stat.S_ISDIR(value.st_mode) or value.st_uid != os.getuid() or stat.S_IMODE(value.st_mode) & 0o222:
        raise RuntimeError(f"Gate2 environment root is not immutable/user-owned: {path}")


def _release_commit(root: Path) -> str:
    if root != RELEASE_ROOT or not root.is_absolute() or root.resolve() != root:
        raise RuntimeError("Gate2 verifier received another release root")
    release_commit = git(root, "rev-parse", "--verify", f"{RELEASE_REF}^{{commit}}")
    if git(root, "rev-parse", "--verify", "HEAD") != release_commit:
        raise RuntimeError("current release HEAD no longer matches its v9 tag")
    if git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("current v9 release checkout is not clean")
    return release_commit


def validate_bound_evidence(root: Path, value: dict[str, str], *, expected_gate: str) -> str:
    if set(value) != BASE_KEYS:
        raise RuntimeError("v9 Gate2 bound evidence field set mismatch")
    release_commit = _release_commit(root)
    expected_identity = {
        "gate2": expected_gate,
        "gate2_schema_version": "2",
        "release_flavor": "v9",
        "release_ref": RELEASE_REF,
        "release_commit": release_commit,
    }
    if any(value.get(key) != expected for key, expected in expected_identity.items()):
        raise RuntimeError("v9 Gate2 release/status identity mismatch")
    for key in (
        "worker_environment_key",
        "controller_environment_key",
        "source_sha256",
        "training_sif_sha256",
        "toolathlon_sif_sha256",
        "toolathlon_build_receipt_sha256",
        "toolathlon_portability_patch_sha256",
    ):
        if HEX64.fullmatch(value[key]) is None:
            raise RuntimeError(f"v9 Gate2 evidence has malformed {key}")
    if not value["pbs_job_id"]:
        raise RuntimeError("v9 Gate2 evidence lacks its PBS job identity")
    try:
        completed_at = int(value["completed_at_epoch"])
    except ValueError as exc:
        raise RuntimeError("v9 Gate2 completion time is malformed") from exc
    if completed_at <= 0:
        raise RuntimeError("v9 Gate2 completion time is invalid")

    adapter = root / ENV_ADAPTER_RELATIVE
    adapter_schema = f"pbs-singularity-1-{sha256(adapter)}"
    worker = Path(value["worker_venv"])
    controller = Path(value["controller_venv"])
    require_immutable_environment(worker)
    require_immutable_environment(controller)
    worker_manifest = worker / ".platoon-env-manifest"
    controller_manifest = controller / ".platoon-controller-manifest"
    for path in (worker_manifest, controller_manifest):
        require_regular(path, modes={0o400, 0o440, 0o444})
    if (
        worker.name != value["worker_environment_key"]
        or controller.name != value["controller_environment_key"]
        or sha256(worker_manifest) != value["worker_environment_key"]
        or sha256(controller_manifest) != value["controller_environment_key"]
        or manifest_value(worker_manifest, "schema") != adapter_schema
        or manifest_value(worker_manifest, "source_sha256") != value["source_sha256"]
        or manifest_value(controller_manifest, "source_sha256") != value["source_sha256"]
    ):
        raise RuntimeError("v9 Gate2 environment manifests do not match their evidence")

    training_sif = Path(value["training_sif"])
    require_regular(training_sif, modes={0o440, 0o444, 0o550, 0o555})
    if sha256(training_sif) != value["training_sif_sha256"]:
        raise RuntimeError("current training SIF differs from the v9 Gate2 evidence")
    selected_training = os.environ.get("PLATOON_TRAINING_SIF")
    if selected_training is not None and selected_training != str(training_sif):
        raise RuntimeError("current overlay selects another training SIF")

    image = Path(value["toolathlon_sif"])
    receipt = Path(value["toolathlon_build_receipt"])
    require_regular(image, modes={0o550})
    require_regular(receipt, modes={0o400})
    if sha256(image) != value["toolathlon_sif_sha256"] or sha256(receipt) != value["toolathlon_build_receipt_sha256"]:
        raise RuntimeError("current v9 Toolathlon artifact differs from Gate2 evidence")
    selected = {
        "OPENREWARD_TOOLATHLON_SIF": str(image),
        "OPENREWARD_TOOLATHLON_SIF_SHA256": value["toolathlon_sif_sha256"],
        "OPENREWARD_TOOLATHLON_BUILD_RECEIPT": str(receipt),
        "OPENREWARD_TOOLATHLON_PORTABILITY_PATCH_SHA256": value["toolathlon_portability_patch_sha256"],
    }
    for name, expected in selected.items():
        if name in os.environ and os.environ[name] != expected:
            raise RuntimeError(f"current v9 overlay {name} differs from Gate2 evidence")
    return release_commit


def _qstat_record(path: Path, job_id: str) -> dict[str, str]:
    require_regular(path, modes={0o400, 0o600})
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Gate2 PBS accounting JSON is unreadable or malformed") from exc
    jobs: Any = document.get("Jobs", document) if isinstance(document, dict) else None
    if not isinstance(jobs, dict):
        raise RuntimeError("Gate2 PBS accounting JSON has no job map")
    matches = [item for key, item in jobs.items() if key == job_id or key.split(".", 1)[0] == job_id.split(".", 1)[0]]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise RuntimeError("Gate2 PBS accounting does not uniquely contain its job")
    record = matches[0]
    return {
        "pbs_state": str(record.get("job_state", "")),
        "pbs_substate": str(int(record.get("substate", -1))),
        "pbs_exit_status": str(int(record.get("Exit_status", -1))),
        "pbs_run_count": str(int(record.get("run_count", 0))),
    }


def verify(root: Path, result_path: Path) -> dict[str, str]:
    value = parse_result(result_path)
    if set(value) != FINAL_KEYS:
        raise RuntimeError("v9 Gate2 final RESULT field set mismatch")
    base = {key: value[key] for key in BASE_KEYS}
    release_commit = validate_bound_evidence(root, base, expected_gate="passed")

    candidate_path = Path(value["candidate_path"])
    candidate = parse_result(candidate_path, modes={0o600})
    if value["candidate_sha256"] != sha256(candidate_path):
        raise RuntimeError("v9 Gate2 candidate digest mismatch")
    expected_candidate = dict(base)
    expected_candidate["gate2"] = "passed_pending_pbs_accounting"
    if candidate != expected_candidate:
        raise RuntimeError("v9 Gate2 final RESULT differs from its candidate")

    accounting_path = Path(value["pbs_accounting_path"])
    if value["pbs_accounting_sha256"] != sha256(accounting_path):
        raise RuntimeError("v9 Gate2 PBS accounting digest mismatch")
    pbs = _qstat_record(accounting_path, value["pbs_job_id"])
    expected_pbs = {
        "pbs_state": "F",
        "pbs_substate": "92",
        "pbs_exit_status": "0",
        "pbs_run_count": "1",
    }
    if pbs != expected_pbs or any(value[key] != expected for key, expected in pbs.items()):
        raise RuntimeError("v9 Gate2 final RESULT lacks exact first-run PBS Exit 0")
    try:
        finalized_at = int(value["finalized_at_epoch"])
        completed_at = int(value["completed_at_epoch"])
    except ValueError as exc:
        raise RuntimeError("v9 Gate2 finalization time is malformed") from exc
    if finalized_at < completed_at:
        raise RuntimeError("v9 Gate2 final RESULT predates its candidate")

    print(
        "openreward_v9_gate2_result=PASS "
        f"release_commit={release_commit} source_sha256={value['source_sha256']} "
        f"pbs_job_id={value['pbs_job_id']}"
    )
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    verify(args.release_root.absolute(), args.result.absolute())


if __name__ == "__main__":
    main()
