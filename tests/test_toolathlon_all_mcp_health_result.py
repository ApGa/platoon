"""Independent verification tests for the post-PBS all-MCP RESULT."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER = REPO_ROOT / "pbs-scripts/verify_toolathlon_all_mcp_health_result.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("verify_toolathlon_all_mcp_health_result", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: dict, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    path.chmod(mode)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _evidence(tmp_path: Path, verifier) -> tuple[Path, Path, Path, Path]:
    release = tmp_path / "release"
    scripts = release / "pbs-scripts"
    scripts.mkdir(parents=True)
    driver = scripts / "toolathlon_all_mcp_health.py"
    driver.write_text("print('health driver')\n")
    driver.chmod(0o640)

    image = tmp_path / "toolathlon.sif"
    image.write_bytes(b"synthetic-sif-content\n")
    image.chmod(0o550)
    image_sha = verifier.sha256(image)
    patch_identity = "a" * 64
    build_path = tmp_path / "build-receipt.json"
    image_stat = image.stat()
    build = {
        "schema_version": 1,
        "gate": "toolathlon_pbs_portable_build",
        "status": "passed",
        "authorization_scope": "image_artifact_only",
        "pbs_accounting_attested": False,
        "toolathlon_source_revision": verifier.EXPECTED_REVISION,
        "base_image": {
            "path": str(tmp_path / "base.sif"),
            "sha256": verifier.EXPECTED_BASE_SIF_SHA256,
            "oci_digest": verifier.EXPECTED_BASE_OCI,
        },
        "derived_image": {
            "path": str(image),
            "sha256": image_sha,
            "size": image_stat.st_size,
            "device": image_stat.st_dev,
            "inode": image_stat.st_ino,
            "mode": 0o550,
            "uid": os.getuid(),
        },
        "portability_patch_sha256": patch_identity,
        "embedded": {
            "base_server_sha256": "base-server-sha",
            "patched_server_sha256": verifier.EXPECTED_PATCHED_SERVER_SHA256,
            "base_tool_schemas_sha256": "base-catalog-sha",
            "repaired_tool_schemas_sha256": verifier.EXPECTED_REPAIRED_CATALOG_SHA256,
            "playwright": verifier.EXPECTED_PLAYWRIGHT,
            "verify": "passed",
        },
        "labels": {
            "org.opencontainers.image.source": "https://github.com/ApGa/toolathlon-gym",
            "org.opencontainers.image.revision": verifier.EXPECTED_REVISION,
            "org.openreward.toolathlon.base-sif-sha256": verifier.EXPECTED_BASE_SIF_SHA256,
            "org.openreward.toolathlon.base-oci-digest": verifier.EXPECTED_BASE_OCI,
            "org.openreward.toolathlon.pbs-portability": patch_identity,
            "org.openreward.toolathlon.playwright-browsers-sha256": (verifier.EXPECTED_PLAYWRIGHT_BROWSER_TREE_SHA256),
        },
    }
    _write_json(build_path, build, 0o400)

    release_ref = "refs/tags/test-v9-release"
    manifest = {
        "release_ref": release_ref,
        "release_root": str(release),
        "toolathlon_image": {"path": str(image), "sha256": image_sha},
        "toolathlon_image_derivation": {
            "status": "ready",
            "build_receipt_path": str(build_path),
            "base_path": str(tmp_path / "base.sif"),
            "base_sha256": verifier.EXPECTED_BASE_SIF_SHA256,
            "base_oci_amd64_digest": verifier.EXPECTED_BASE_OCI,
            "base_server_sha256": "base-server-sha",
            "patched_server_sha256": verifier.EXPECTED_PATCHED_SERVER_SHA256,
            "base_tool_schemas_sha256": "base-catalog-sha",
            "repaired_tool_schemas_sha256": verifier.EXPECTED_REPAIRED_CATALOG_SHA256,
            "playwright": verifier.EXPECTED_PLAYWRIGHT,
            "pbs_portability_patch_sha256": patch_identity,
        },
    }
    _write_json(scripts / "openreward-v9-release.json", manifest, 0o640)

    _git(release, "init", "-q")
    _git(release, "config", "user.name", "Verifier Test")
    _git(release, "config", "user.email", "verifier@example.invalid")
    _git(release, "add", ".")
    _git(release, "commit", "-qm", "synthetic release")
    _git(release, "tag", "-a", "test-v9-release", "-m", "synthetic tag")
    release_commit = _git(release, "rev-parse", "HEAD")

    # The independent all-MCP verifier must re-run and exactly cross-check one
    # finalized post-PBS v9 Gate2 evidence chain.
    gate2_path = tmp_path / "gate2-RESULT"
    gate2_values = {
        "gate2": "passed",
        "gate2_schema_version": "2",
        "release_flavor": "v9",
        "release_ref": release_ref,
        "release_commit": release_commit,
        "pbs_job_id": "11111.pbs1",
        "worker_environment_key": "1" * 64,
        "controller_environment_key": "2" * 64,
        "source_sha256": "3" * 64,
        "training_sif_sha256": "4" * 64,
        "toolathlon_sif_sha256": image_sha,
        "toolathlon_build_receipt_sha256": verifier.sha256(build_path),
        "toolathlon_portability_patch_sha256": patch_identity,
    }
    _write_json(gate2_path, gate2_values, 0o400)
    gate2_bound = {
        "path": str(gate2_path),
        "sha256": verifier.sha256(gate2_path),
        **{
            key: gate2_values[key]
            for key in (
                "pbs_job_id",
                "source_sha256",
                "worker_environment_key",
                "controller_environment_key",
                "training_sif_sha256",
                "toolathlon_sif_sha256",
                "toolathlon_build_receipt_sha256",
                "toolathlon_portability_patch_sha256",
                "release_ref",
                "release_commit",
            )
        },
    }

    tasks = sorted(verifier.EXPECTED_TASKS)
    sessions = [
        {
            "task_name": task,
            "sid": f"session-{index}",
            "expected_servers": list(verifier.EXPECTED_SESSION_COVER[task]),
            "catalog_tool_count": len(verifier.EXPECTED_SESSION_COVER[task]),
            "live_probes": ["mcp_health"],
            "concurrency_barrier": task in verifier.CONCURRENT_TASKS,
            "mcp_health_checks": [
                {
                    "status": "passed",
                    "server_count": len(verifier.EXPECTED_SESSION_COVER[task]),
                    "servers": {
                        server: {
                            "live_tool_count": 1,
                            "live_tool_names_sha256": "f" * 64,
                        }
                        for server in verifier.EXPECTED_SESSION_COVER[task]
                    },
                }
                for _ in range(2 if task in verifier.CONCURRENT_TASKS else 1)
            ],
        }
        for index, task in enumerate(tasks)
    ]
    tested_tools = sorted(verifier.EXPECTED_EMAIL_TOOLS | {"mcp_health", "playwright_with_chunk.browser_tab_list"})
    report = {
        "schema_version": 1,
        "gate": "toolathlon_all_mcp_health_driver",
        "status": "passed",
        "expected_server_count": 25,
        "tested_servers": list(verifier.EXPECTED_SERVERS),
        "failure_count": 0,
        "catalog_sha256": verifier.EXPECTED_REPAIRED_CATALOG_SHA256,
        "sessions": sessions,
        "concurrency": verifier.EXPECTED_CONCURRENCY,
        "playwright": verifier.EXPECTED_PLAYWRIGHT_HEALTH,
        "tested_tools": tested_tools,
        "catalog_representatives": [f"{server}.probe" for server in verifier.EXPECTED_SERVERS],
        "completed_at_epoch": 100,
    }
    report_path = tmp_path / "health-report.json"
    _write_json(report_path, report, 0o600)
    job_id = "12345.pbs1"
    candidate = {
        "schema_version": 1,
        "gate": "toolathlon_all_mcp_health_candidate",
        "status": "passed_pending_pbs_accounting",
        "release_ref": release_ref,
        "release_commit": release_commit,
        "release_root": str(release),
        "pbs_job_id": job_id,
        "toolathlon_image": {
            "path": str(image),
            "sha256": image_sha,
            "size": image_stat.st_size,
            "device": image_stat.st_dev,
            "inode": image_stat.st_ino,
            "mode": 0o550,
            "uid": os.getuid(),
        },
        "toolathlon_source_revision": verifier.EXPECTED_REVISION,
        "base_sif_sha256": verifier.EXPECTED_BASE_SIF_SHA256,
        "base_oci_digest": verifier.EXPECTED_BASE_OCI,
        "portability_patch_sha256": patch_identity,
        "patched_server_sha256": verifier.EXPECTED_PATCHED_SERVER_SHA256,
        "repaired_tool_schemas_sha256": verifier.EXPECTED_REPAIRED_CATALOG_SHA256,
        "toolathlon_build_receipt": {
            "path": str(build_path),
            "sha256": verifier.sha256(build_path),
        },
        "gate2_result": gate2_bound,
        "health_driver": {"path": str(driver), "sha256": verifier.sha256(driver)},
        "health_report": {
            "path": str(report_path),
            "sha256": verifier.sha256(report_path),
        },
        "expected_server_count": 25,
        "tested_servers": list(verifier.EXPECTED_SERVERS),
        "tested_tools": tested_tools,
        "sessions": sessions,
        "concurrency": verifier.EXPECTED_CONCURRENCY,
        "playwright": verifier.EXPECTED_PLAYWRIGHT_HEALTH,
        "failure_count": 0,
        "completed_at_epoch": 100,
    }
    candidate_path = tmp_path / "RESULT.candidate"
    _write_json(candidate_path, candidate, 0o600)
    accounting_path = tmp_path / "qstat.json"
    _write_json(
        accounting_path,
        {
            "Jobs": {
                job_id: {
                    "job_state": "F",
                    "substate": 92,
                    "Exit_status": 0,
                    "run_count": 1,
                }
            }
        },
        0o600,
    )
    cleanup_path = tmp_path / "cleanup.json"
    _write_json(
        cleanup_path,
        {
            "schema_version": 2,
            "status": "passed",
            "pbs_job_id": job_id,
            "host": "node-a",
            "primary_status": 0,
            "cleanup_completed": True,
            "forced_kill": False,
            "endpoints_stopped": True,
            "swe_sessions_clean": True,
            "shutdown_request_verified": True,
        },
        0o600,
    )
    result = dict(candidate)
    result.update(
        {
            "gate": "toolathlon_all_mcp_health",
            "status": "passed",
            "candidate": {
                "path": str(candidate_path),
                "sha256": verifier.sha256(candidate_path),
            },
            "pbs": {
                "job_id": job_id,
                "state": "F",
                "substate": 92,
                "exit_status": 0,
                "run_count": 1,
            },
            "pbs_accounting": {
                "path": str(accounting_path),
                "sha256": verifier.sha256(accounting_path),
            },
            "cleanup_receipt": {
                "path": str(cleanup_path),
                "sha256": verifier.sha256(cleanup_path),
                "schema_version": 2,
                "status": "passed",
                "primary_status": 0,
                "shutdown_request_verified": True,
            },
            "finalized_at_epoch": 101,
        }
    )
    result_path = tmp_path / "RESULT"
    _write_json(result_path, result, 0o400)
    return release, result_path, image, gate2_path


def test_final_result_rehashes_current_release_image_and_all_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = _load_verifier()
    release, result, _, gate2 = _evidence(tmp_path, verifier)
    monkeypatch.setattr(verifier, "RELEASE_ROOT", release)
    monkeypatch.setattr(verifier, "RELEASE_REF", "refs/tags/test-v9-release")

    monkeypatch.setattr(
        verifier,
        "validate_gate2_evidence",
        lambda _root, _result, path: {} if path == gate2 else None,
    )
    value = verifier.verify(release, result, gate2)

    assert value["gate"] == "toolathlon_all_mcp_health"
    assert value["pbs"]["exit_status"] == 0


def test_final_result_rejects_sif_changed_after_health(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    verifier = _load_verifier()
    release, result, image, gate2 = _evidence(tmp_path, verifier)
    monkeypatch.setattr(verifier, "RELEASE_ROOT", release)
    monkeypatch.setattr(verifier, "RELEASE_REF", "refs/tags/test-v9-release")
    monkeypatch.setattr(verifier, "validate_gate2_evidence", lambda *_args: {})
    image.chmod(0o750)
    with image.open("ab") as stream:
        stream.write(b"changed")
    image.chmod(0o550)

    with pytest.raises(RuntimeError, match="stream hash differs"):
        verifier.verify(release, result, gate2)


def test_old_node_smoke_result_cannot_satisfy_all_mcp_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    verifier = _load_verifier()
    release, _, _, gate2 = _evidence(tmp_path, verifier)
    monkeypatch.setattr(verifier, "RELEASE_ROOT", release)
    monkeypatch.setattr(verifier, "RELEASE_REF", "refs/tags/test-v9-release")
    old = tmp_path / "old-node-smoke-RESULT"
    _write_json(old, {"node_services_pbsdsh_smoke": "PASS"}, 0o400)

    with pytest.raises(RuntimeError, match="field set mismatch"):
        verifier.verify(release, old, gate2)
