"""Independent checks for the production v9 Gate2 RESULT verifier."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER = REPO_ROOT / "pbs-scripts/verify_openreward_v9_gate2_result.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("verify_openreward_v9_gate2_result", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["/usr/bin/git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _evidence(tmp_path: Path, verifier, monkeypatch: pytest.MonkeyPatch):
    release = tmp_path / "release"
    scripts = release / "pbs-scripts"
    scripts.mkdir(parents=True)
    adapter = scripts / "prepare_openreward_env.sh"
    adapter.write_text("#!/bin/bash\n# synthetic adapter\n")

    source_sha = "a" * 64
    adapter_schema = f"pbs-singularity-1-{verifier.sha256(adapter)}"
    worker_manifest_text = f"schema={adapter_schema}\nsource_sha256={source_sha}\nrole=worker\n"
    controller_manifest_text = f"schema=controller\nsource_sha256={source_sha}\nrole=controller\n"
    worker_key = __import__("hashlib").sha256(worker_manifest_text.encode()).hexdigest()
    controller_key = __import__("hashlib").sha256(controller_manifest_text.encode()).hexdigest()
    worker = tmp_path / "worker-cache" / "envs" / worker_key
    controller = tmp_path / "controller-cache" / "envs" / controller_key
    worker.mkdir(parents=True)
    controller.mkdir(parents=True)
    worker_manifest = worker / ".platoon-env-manifest"
    controller_manifest = controller / ".platoon-controller-manifest"
    worker_manifest.write_text(worker_manifest_text)
    controller_manifest.write_text(controller_manifest_text)
    worker_manifest.chmod(0o400)
    controller_manifest.chmod(0o400)
    worker.chmod(0o500)
    controller.chmod(0o500)

    training = tmp_path / "training.sif"
    image = tmp_path / "toolathlon.sif"
    receipt = tmp_path / "build-receipt.json"
    training.write_bytes(b"training-sif\n")
    image.write_bytes(b"toolathlon-sif\n")
    receipt.write_text(json.dumps({"status": "passed"}) + "\n")
    training.chmod(0o550)
    image.chmod(0o550)
    receipt.chmod(0o400)
    patch_identity = "b" * 64
    monkeypatch.setenv("PLATOON_TRAINING_SIF", str(training))
    monkeypatch.setenv("OPENREWARD_TOOLATHLON_SIF", str(image))
    monkeypatch.setenv("OPENREWARD_TOOLATHLON_SIF_SHA256", verifier.sha256(image))
    monkeypatch.setenv("OPENREWARD_TOOLATHLON_BUILD_RECEIPT", str(receipt))
    monkeypatch.setenv("OPENREWARD_TOOLATHLON_PORTABILITY_PATCH_SHA256", patch_identity)

    _git(release, "init", "-q")
    _git(release, "config", "user.name", "Gate Test")
    _git(release, "config", "user.email", "gate@example.invalid")
    _git(release, "add", ".")
    _git(release, "commit", "-qm", "synthetic release")
    _git(release, "tag", "-a", "test-v9", "-m", "synthetic v9")
    commit = _git(release, "rev-parse", "HEAD")

    candidate = tmp_path / "RESULT.candidate"
    values = {
        "gate2": "passed_pending_pbs_accounting",
        "gate2_schema_version": "2",
        "release_flavor": "v9",
        "release_ref": "refs/tags/test-v9",
        "release_commit": commit,
        "pbs_job_id": "1234.pbs1",
        "worker_environment_key": worker_key,
        "controller_environment_key": controller_key,
        "source_sha256": source_sha,
        "training_sif": str(training),
        "training_sif_sha256": verifier.sha256(training),
        "worker_venv": str(worker),
        "controller_venv": str(controller),
        "toolathlon_sif": str(image),
        "toolathlon_sif_sha256": verifier.sha256(image),
        "toolathlon_build_receipt": str(receipt),
        "toolathlon_build_receipt_sha256": verifier.sha256(receipt),
        "toolathlon_portability_patch_sha256": patch_identity,
        "completed_at_epoch": "100",
    }
    candidate.write_text("".join(f"{key}={value}\n" for key, value in values.items()))
    candidate.chmod(0o600)
    accounting = tmp_path / "qstat.json"
    accounting.write_text(
        json.dumps(
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
        )
        + "\n"
    )
    accounting.chmod(0o600)
    result = tmp_path / "RESULT"
    values["gate2"] = "passed"
    values.update(
        {
            "candidate_path": str(candidate),
            "candidate_sha256": verifier.sha256(candidate),
            "pbs_accounting_path": str(accounting),
            "pbs_accounting_sha256": verifier.sha256(accounting),
            "pbs_state": "F",
            "pbs_substate": "92",
            "pbs_exit_status": "0",
            "pbs_run_count": "1",
            "finalized_at_epoch": "101",
        }
    )
    result.write_text("".join(f"{key}={value}\n" for key, value in values.items()))
    result.chmod(0o400)
    return release, result, training


def test_v9_gate2_result_rehashes_current_release_environments_and_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = _load_verifier()
    release, result, _ = _evidence(tmp_path, verifier, monkeypatch)
    monkeypatch.setattr(verifier, "RELEASE_ROOT", release)
    monkeypatch.setattr(verifier, "RELEASE_REF", "refs/tags/test-v9")

    value = verifier.verify(release, result)

    assert value["release_flavor"] == "v9"
    assert value["gate2"] == "passed"


def test_v9_gate2_result_rejects_current_training_sif_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    verifier = _load_verifier()
    release, result, training = _evidence(tmp_path, verifier, monkeypatch)
    monkeypatch.setattr(verifier, "RELEASE_ROOT", release)
    monkeypatch.setattr(verifier, "RELEASE_REF", "refs/tags/test-v9")
    training.chmod(0o750)
    training.write_bytes(b"changed\n")
    training.chmod(0o550)

    with pytest.raises(RuntimeError, match="training SIF differs"):
        verifier.verify(release, result)


def test_old_gate2_result_cannot_satisfy_v9_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    verifier = _load_verifier()
    release, _, _ = _evidence(tmp_path, verifier, monkeypatch)
    monkeypatch.setattr(verifier, "RELEASE_ROOT", release)
    monkeypatch.setattr(verifier, "RELEASE_REF", "refs/tags/test-v9")
    old = tmp_path / "old-v8-RESULT"
    old.write_text("gate2=passed\nsource_sha256=" + "a" * 64 + "\n")
    old.chmod(0o400)

    with pytest.raises(RuntimeError, match="field set mismatch"):
        verifier.verify(release, old)
