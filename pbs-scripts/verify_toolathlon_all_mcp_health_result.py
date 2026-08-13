#!/usr/bin/env python3
"""Verify the post-PBS all-25-MCP RESULT before a production launch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

RELEASE_ROOT = Path("/home/aci18912it/rasp/releases/platoon-v8r3-376d1b9-18e62c0")
RELEASE_REF = "refs/tags/abci-v8r3-376d1b9-18e62c0"
MANIFEST_RELATIVE = Path("pbs-scripts/openreward-v9-release.json")
HEALTH_DRIVER_RELATIVE = Path("pbs-scripts/toolathlon_all_mcp_health.py")
EXPECTED_REVISION = "18e62c0d0419da88209af911528f2f8a0b44a5ed"
EXPECTED_BASE_SIF_SHA256 = "e730e11d1abc63f5cac9eb1788e36da79aa9c7027ed89d91ee42a5b1e311e267"
EXPECTED_BASE_OCI = "sha256:c1e60b3673c848b758bda1b32ec39898feb5d6ff4c7e55ead8661adb04b327bd"
EXPECTED_PATCHED_SERVER_SHA256 = "f38415553a88d16a9b7327ce241c20eea1ef172ca9ae4fb9da797b8e06c76485"
EXPECTED_REPAIRED_CATALOG_SHA256 = "a6bdfea0a5524ec8d198a0b2eb9b107aba89e308b50659a106abaff0002fd64d"
EXPECTED_PLAYWRIGHT_BROWSER_TREE_SHA256 = "84d979d1a301e24d7ee04009ff85b806cda195cce9d6bd1a8a686f06887805e5"
EXPECTED_PLAYWRIGHT_CHROMIUM_SHA256 = "b93030bc068a3850d4d668e69816fe4a5f04ec93db9b2f63f32f58a39f207db8"
EXPECTED_PLAYWRIGHT_CONFIG_SHA256 = "47841318775a9afc122097e769a3fcc02d3af7089d372b60fc11da1ea44f0de9"
EXPECTED_PLAYWRIGHT = {
    "browsers_path": "/opt/ms-playwright",
    "base_browser_tree_sha256": EXPECTED_PLAYWRIGHT_BROWSER_TREE_SHA256,
    "portable_browser_tree_sha256": EXPECTED_PLAYWRIGHT_BROWSER_TREE_SHA256,
    "browser_file_count": 156,
    "browser_total_bytes": 959511402,
    "browser_symlink_count": 0,
    "browser_directory_count": 13,
    "chromium_revision": "1182",
    "chromium_executable": "/opt/ms-playwright/chromium-1182/chrome-linux/chrome",
    "chromium_sha256": EXPECTED_PLAYWRIGHT_CHROMIUM_SHA256,
    "headless_shell_executable": "/opt/ms-playwright/chromium_headless_shell-1182/chrome-linux/headless_shell",
    "headless_shell_sha256": "71ebe8760eb72a99b265470d4030fef296f119e25b9aab5d3efbb6ac821483a5",
    "headless_shell_bytes": 305228784,
    "base_config_sha256": "66f5bc92c182bf365e499d8237308452a016d8ca9c5e878d1628393647c82d1a",
    "patched_config_sha256": EXPECTED_PLAYWRIGHT_CONFIG_SHA256,
    "chromium_sandbox": "disabled_for_unprivileged_singularity",
    "rootless_launch": "passed",
}
EXPECTED_PLAYWRIGHT_HEALTH = {
    "status": "passed",
    "browsers_path": "/opt/ms-playwright",
    "chromium_executable": "/opt/ms-playwright/chromium-1182/chrome-linux/chrome",
    "chromium_sha256": EXPECTED_PLAYWRIGHT_CHROMIUM_SHA256,
    "headless_shell_executable": "/opt/ms-playwright/chromium_headless_shell-1182/chrome-linux/headless_shell",
    "headless_shell_sha256": "71ebe8760eb72a99b265470d4030fef296f119e25b9aab5d3efbb6ac821483a5",
    "patched_config_sha256": EXPECTED_PLAYWRIGHT_CONFIG_SHA256,
    "chromium_sandbox": "disabled_for_unprivileged_singularity",
    "browser_tab_list": "passed",
}
EXPECTED_SERVERS = (
    "arxiv-latex",
    "arxiv_local",
    "canvas",
    "emails",
    "excel",
    "fetch",
    "filesystem",
    "google_calendar",
    "google_forms",
    "google_sheet",
    "howtocook",
    "memory",
    "notion",
    "pdf-tools",
    "playwright_with_chunk",
    "pptx",
    "rail_12306",
    "scholarly",
    "snowflake",
    "terminal",
    "woocommerce",
    "word",
    "yahoo-finance",
    "youtube",
    "youtube-transcript",
)
EXPECTED_EMAIL_TOOLS = {
    "emails.check_connection",
    "emails.send_email",
    "emails.get_emails",
}
EXPECTED_SESSION_COVER = {
    "financial-audit-cross-system-reconciliation": (
        "emails",
        "excel",
        "google_calendar",
        "pdf-tools",
        "snowflake",
        "terminal",
        "woocommerce",
        "word",
    ),
    "arxiv-lit-review-gsheet": (
        "arxiv-latex",
        "arxiv_local",
        "filesystem",
        "google_sheet",
        "memory",
        "scholarly",
    ),
    "terminal-canvas-howtocook-gsheet-notion-email": (
        "canvas",
        "emails",
        "filesystem",
        "google_sheet",
        "howtocook",
        "notion",
        "terminal",
    ),
    "pw-yf-risk-assessment-excel-ppt": (
        "excel",
        "filesystem",
        "playwright_with_chunk",
        "pptx",
        "terminal",
        "yahoo-finance",
    ),
    "fetch-sf-support-survey-gform-gcal-excel": (
        "excel",
        "fetch",
        "google_calendar",
        "google_forms",
        "pdf-tools",
        "snowflake",
        "terminal",
    ),
    "yt-train-conference-planning-notion-gcal-excel": (
        "emails",
        "excel",
        "filesystem",
        "google_calendar",
        "notion",
        "rail_12306",
        "youtube",
    ),
    "yt-transcript-word-gform": (
        "emails",
        "filesystem",
        "google_forms",
        "word",
        "youtube-transcript",
    ),
}
EXPECTED_TASKS = set(EXPECTED_SESSION_COVER)
CONCURRENT_TASKS = {
    "terminal-canvas-howtocook-gsheet-notion-email",
    "yt-transcript-word-gform",
}
EXPECTED_CONCURRENCY = {
    "status": "passed",
    "simultaneous_active_sessions_verified": 2,
    "tasks": [
        "terminal-canvas-howtocook-gsheet-notion-email",
        "yt-transcript-word-gform",
    ],
    "overlapping_servers": ["emails", "filesystem"],
    "both_called": ["emails.check_connection"],
}
CANDIDATE_KEYS = {
    "schema_version",
    "gate",
    "status",
    "release_ref",
    "release_commit",
    "release_root",
    "pbs_job_id",
    "toolathlon_image",
    "toolathlon_source_revision",
    "base_sif_sha256",
    "base_oci_digest",
    "portability_patch_sha256",
    "patched_server_sha256",
    "repaired_tool_schemas_sha256",
    "toolathlon_build_receipt",
    "gate2_result",
    "health_driver",
    "health_report",
    "expected_server_count",
    "tested_servers",
    "tested_tools",
    "sessions",
    "concurrency",
    "playwright",
    "failure_count",
    "completed_at_epoch",
}
FINAL_ONLY_KEYS = {
    "candidate",
    "pbs",
    "pbs_accounting",
    "cleanup_receipt",
    "finalized_at_epoch",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def require_file(path: Path, *, modes: set[int]) -> os.stat_result:
    if not path.is_absolute() or path.resolve() != path:
        raise RuntimeError(f"evidence path is not direct and absolute: {path}")
    value = path.lstat()
    mode = stat.S_IMODE(value.st_mode)
    if not stat.S_ISREG(value.st_mode) or value.st_uid != os.getuid() or mode not in modes:
        raise RuntimeError(f"evidence is not a user-owned regular file in modes {sorted(modes)}: {path}")
    return value


def read_json(path: Path, *, modes: set[int]) -> dict[str, Any]:
    require_file(path, modes=modes)
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"evidence JSON is unreadable or malformed: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"evidence JSON is not an object: {path}")
    return value


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    ).stdout.strip()


def validate_release(root: Path, result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if root != RELEASE_ROOT or not root.is_absolute() or root.resolve() != root:
        raise RuntimeError("all-MCP result verifier received another release root")
    head = git(root, "rev-parse", "--verify", "HEAD")
    release_commit = git(root, "rev-parse", "--verify", f"{RELEASE_REF}^{{commit}}")
    if head != release_commit:
        raise RuntimeError("current release HEAD no longer matches its tag")
    if (
        result.get("release_root") != str(root)
        or result.get("release_ref") != RELEASE_REF
        or result.get("release_commit") != release_commit
    ):
        raise RuntimeError("all-MCP result is bound to another release identity")
    manifest = json.loads((root / MANIFEST_RELATIVE).read_text())
    if manifest.get("release_ref") != RELEASE_REF or manifest.get("release_root") != str(root):
        raise RuntimeError("v9 manifest release identity mismatch")
    return release_commit, manifest


def validate_gate2_evidence(root: Path, result: dict[str, Any], gate2_result: Path) -> dict[str, str]:
    reference = result.get("gate2_result")
    if not isinstance(reference, dict):
        raise RuntimeError("all-MCP RESULT lacks its exact v9 Gate2 binding")
    gate2_path = gate2_result.absolute()
    require_file(gate2_path, modes={0o400})
    if reference.get("path") != str(gate2_path) or reference.get("sha256") != sha256(gate2_path):
        raise RuntimeError("all-MCP RESULT is bound to another v9 Gate2 RESULT")

    verifier_path = root / "pbs-scripts/verify_openreward_v9_gate2_result.py"
    spec = importlib.util.spec_from_file_location("verify_openreward_v9_gate2_result", verifier_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the v9 Gate2 verifier")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    gate2 = verifier.verify(root, gate2_path)
    bound_fields = (
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
    expected = {
        "path": str(gate2_path),
        "sha256": sha256(gate2_path),
        **{field: gate2[field] for field in bound_fields},
    }
    if reference != expected:
        raise RuntimeError("all-MCP RESULT Gate2 identities differ from final evidence")
    if (
        result.get("release_ref") != gate2["release_ref"]
        or result.get("release_commit") != gate2["release_commit"]
        or result.get("portability_patch_sha256") != gate2["toolathlon_portability_patch_sha256"]
    ):
        raise RuntimeError("all-MCP and v9 Gate2 release/image chains disagree")
    return gate2


def validate_image_and_build(result: dict[str, Any], manifest: dict[str, Any]) -> tuple[Path, str]:
    image_contract = manifest.get("toolathlon_image")
    derivation = manifest.get("toolathlon_image_derivation")
    if not isinstance(image_contract, dict) or not isinstance(derivation, dict):
        raise RuntimeError("v9 manifest lacks final Toolathlon image contracts")
    image = Path(str(image_contract.get("path", "")))
    image_stat = require_file(image, modes={0o550})
    image_sha = sha256(image)
    if image_sha != image_contract.get("sha256"):
        raise RuntimeError("current Toolathlon SIF stream hash differs from v9 manifest")
    if result.get("toolathlon_image") != {
        "path": str(image),
        "sha256": image_sha,
        "size": image_stat.st_size,
        "device": image_stat.st_dev,
        "inode": image_stat.st_ino,
        "mode": 0o550,
        "uid": os.getuid(),
    }:
        raise RuntimeError("all-MCP result Toolathlon image identity mismatch")
    expected_scalars = {
        "toolathlon_source_revision": EXPECTED_REVISION,
        "base_sif_sha256": EXPECTED_BASE_SIF_SHA256,
        "base_oci_digest": EXPECTED_BASE_OCI,
        "portability_patch_sha256": derivation.get("pbs_portability_patch_sha256"),
        "patched_server_sha256": EXPECTED_PATCHED_SERVER_SHA256,
        "repaired_tool_schemas_sha256": EXPECTED_REPAIRED_CATALOG_SHA256,
    }
    if any(result.get(key) != value for key, value in expected_scalars.items()):
        raise RuntimeError("all-MCP result image derivation identity mismatch")
    if derivation.get("playwright") != EXPECTED_PLAYWRIGHT:
        raise RuntimeError("v9 release Playwright provenance mismatch")
    patch_identity = result.get("portability_patch_sha256")
    if (
        not isinstance(patch_identity, str)
        or len(patch_identity) != 64
        or any(character not in "0123456789abcdef" for character in patch_identity)
    ):
        raise RuntimeError("all-MCP result portability patch identity is malformed")

    build_ref = result.get("toolathlon_build_receipt")
    if not isinstance(build_ref, dict):
        raise RuntimeError("all-MCP result lacks its build receipt reference")
    build_path = Path(str(build_ref.get("path", "")))
    build = read_json(build_path, modes={0o400})
    if (
        str(build_path) != derivation.get("build_receipt_path")
        or build_ref.get("sha256") != sha256(build_path)
        or build.get("schema_version") != 1
        or build.get("gate") != "toolathlon_pbs_portable_build"
        or build.get("status") != "passed"
        or build.get("authorization_scope") != "image_artifact_only"
        or build.get("pbs_accounting_attested") is not False
        or build.get("toolathlon_source_revision") != EXPECTED_REVISION
        or build.get("portability_patch_sha256") != derivation.get("pbs_portability_patch_sha256")
    ):
        raise RuntimeError("Toolathlon build receipt identity mismatch")
    expected_derived = {
        "path": str(image),
        "sha256": image_sha,
        "size": image_stat.st_size,
        "device": image_stat.st_dev,
        "inode": image_stat.st_ino,
        "mode": 0o550,
        "uid": os.getuid(),
    }
    if build.get("derived_image") != expected_derived:
        raise RuntimeError("build receipt does not identify the current SIF inode/content")
    base = build.get("base_image")
    if not isinstance(base, dict) or (
        base.get("path") != derivation.get("base_path")
        or base.get("sha256") != EXPECTED_BASE_SIF_SHA256
        or base.get("oci_digest") != EXPECTED_BASE_OCI
    ):
        raise RuntimeError("build receipt base image identity mismatch")
    embedded = build.get("embedded")
    if not isinstance(embedded, dict) or (
        embedded.get("base_server_sha256") != derivation.get("base_server_sha256")
        or embedded.get("patched_server_sha256") != EXPECTED_PATCHED_SERVER_SHA256
        or embedded.get("base_tool_schemas_sha256") != derivation.get("base_tool_schemas_sha256")
        or embedded.get("repaired_tool_schemas_sha256") != EXPECTED_REPAIRED_CATALOG_SHA256
        or embedded.get("playwright") != EXPECTED_PLAYWRIGHT
        or embedded.get("verify") != "passed"
    ):
        raise RuntimeError("build receipt embedded portability verification mismatch")
    labels = build.get("labels")
    required_labels = {
        "org.opencontainers.image.source": "https://github.com/ApGa/toolathlon-gym",
        "org.opencontainers.image.revision": EXPECTED_REVISION,
        "org.openreward.toolathlon.base-sif-sha256": EXPECTED_BASE_SIF_SHA256,
        "org.openreward.toolathlon.base-oci-digest": EXPECTED_BASE_OCI,
        "org.openreward.toolathlon.pbs-portability": derivation.get("pbs_portability_patch_sha256"),
        "org.openreward.toolathlon.playwright-browsers-sha256": EXPECTED_PLAYWRIGHT_BROWSER_TREE_SHA256,
    }
    if not isinstance(labels, dict) or any(labels.get(k) != v for k, v in required_labels.items()):
        raise RuntimeError("build receipt derived SIF labels mismatch")
    return image, image_sha


def validate_health_evidence(result: dict[str, Any], root: Path) -> None:
    driver_ref = result.get("health_driver")
    report_ref = result.get("health_report")
    if not isinstance(driver_ref, dict) or not isinstance(report_ref, dict):
        raise RuntimeError("all-MCP result lacks driver/report references")
    driver = root / HEALTH_DRIVER_RELATIVE
    require_file(driver, modes={0o440, 0o550, 0o640, 0o750})
    if driver_ref != {"path": str(driver), "sha256": sha256(driver)}:
        raise RuntimeError("all-MCP result health driver identity mismatch")
    report_path = Path(str(report_ref.get("path", "")))
    report = read_json(report_path, modes={0o600})
    if report_ref.get("sha256") != sha256(report_path):
        raise RuntimeError("all-MCP health report digest mismatch")
    expected_report = {
        "schema_version": 1,
        "gate": "toolathlon_all_mcp_health_driver",
        "status": "passed",
        "expected_server_count": 25,
        "tested_servers": list(EXPECTED_SERVERS),
        "failure_count": 0,
        "catalog_sha256": EXPECTED_REPAIRED_CATALOG_SHA256,
    }
    if any(report.get(key) != value for key, value in expected_report.items()):
        raise RuntimeError("all-MCP health report status/server identity mismatch")
    if (
        result.get("expected_server_count") != 25
        or result.get("tested_servers") != list(EXPECTED_SERVERS)
        or result.get("failure_count") != 0
        or result.get("sessions") != report.get("sessions")
        or result.get("concurrency") != report.get("concurrency")
        or result.get("playwright") != report.get("playwright")
        or result.get("tested_tools") != report.get("tested_tools")
        or result.get("completed_at_epoch") != report.get("completed_at_epoch")
    ):
        raise RuntimeError("final RESULT does not reproduce its health report evidence")
    sessions = report.get("sessions")
    tested_tools = report.get("tested_tools")
    representatives = report.get("catalog_representatives")
    if not isinstance(sessions, list) or len(sessions) != 7:
        raise RuntimeError("all-MCP health report lacks seven completed sessions")
    if {item.get("task_name") for item in sessions if isinstance(item, dict)} != EXPECTED_TASKS:
        raise RuntimeError("all-MCP health report seven-session identity mismatch")
    by_task = {
        item.get("task_name"): item
        for item in sessions
        if isinstance(item, dict) and isinstance(item.get("task_name"), str)
    }
    server_union: set[str] = set()
    for task_name, expected_server_tuple in EXPECTED_SESSION_COVER.items():
        session = by_task[task_name]
        expected_servers = list(expected_server_tuple)
        if (
            not isinstance(session.get("sid"), str)
            or not session["sid"]
            or session.get("expected_servers") != expected_servers
            or not isinstance(session.get("catalog_tool_count"), int)
            or session["catalog_tool_count"] <= 0
            or not isinstance(session.get("live_probes"), list)
            or "mcp_health" not in session["live_probes"]
            or session.get("concurrency_barrier") != (task_name in CONCURRENT_TASKS)
        ):
            raise RuntimeError(f"all-MCP session evidence mismatch for {task_name}")
        health_checks = session.get("mcp_health_checks")
        expected_check_count = 2 if task_name in CONCURRENT_TASKS else 1
        if not isinstance(health_checks, list) or len(health_checks) != expected_check_count:
            raise RuntimeError(f"all-MCP live MCP check count mismatch for {task_name}")
        for check in health_checks:
            servers = check.get("servers") if isinstance(check, dict) else None
            if (
                not isinstance(check, dict)
                or set(check) != {"status", "server_count", "servers"}
                or check.get("status") != "passed"
                or check.get("server_count") != len(expected_servers)
                or not isinstance(servers, dict)
                or set(servers) != set(expected_servers)
            ):
                raise RuntimeError(f"all-MCP live MCP identity mismatch for {task_name}")
            for evidence in servers.values():
                if (
                    not isinstance(evidence, dict)
                    or set(evidence) != {"live_tool_count", "live_tool_names_sha256"}
                    or not isinstance(evidence.get("live_tool_count"), int)
                    or evidence["live_tool_count"] <= 0
                    or not is_sha256(evidence.get("live_tool_names_sha256"))
                ):
                    raise RuntimeError(f"all-MCP live MCP evidence mismatch for {task_name}")
        server_union.update(expected_servers)
    if server_union != set(EXPECTED_SERVERS):
        raise RuntimeError("all-MCP seven-session union is not the exact 25-server set")
    if report.get("concurrency") != EXPECTED_CONCURRENCY:
        raise RuntimeError("all-MCP health report concurrency proof mismatch")
    if not isinstance(tested_tools, list) or not (
        EXPECTED_EMAIL_TOOLS | {"mcp_health", "playwright_with_chunk.browser_tab_list"}
    ).issubset(tested_tools):
        raise RuntimeError("all-MCP health report lacks email, Playwright, or live MCP probes")
    if report.get("playwright") != EXPECTED_PLAYWRIGHT_HEALTH:
        raise RuntimeError("all-MCP health report lacks exact rootless Playwright evidence")
    if not isinstance(representatives, list) or len(representatives) != 25:
        raise RuntimeError("all-MCP health report lacks 25 catalog representatives")
    if {name.split(".", 1)[0] for name in representatives} != set(EXPECTED_SERVERS):
        raise RuntimeError("all-MCP representative server set mismatch")


def validate_candidate(result: dict[str, Any]) -> None:
    candidate_ref = result.get("candidate")
    if not isinstance(candidate_ref, dict):
        raise RuntimeError("final RESULT lacks its in-job candidate reference")
    path = Path(str(candidate_ref.get("path", "")))
    candidate = read_json(path, modes={0o600})
    if candidate_ref.get("sha256") != sha256(path):
        raise RuntimeError("all-MCP candidate digest mismatch")
    expected = {key: result[key] for key in CANDIDATE_KEYS}
    expected["gate"] = "toolathlon_all_mcp_health_candidate"
    expected["status"] = "passed_pending_pbs_accounting"
    if set(candidate) != CANDIDATE_KEYS or candidate != expected:
        raise RuntimeError("final RESULT and in-job candidate content differ")


def _qstat_record(value: dict[str, Any], job_id: str) -> dict[str, Any]:
    jobs = value.get("Jobs", value)
    if not isinstance(jobs, dict):
        raise RuntimeError("PBS accounting JSON has no job mapping")
    candidates = [
        item for key, item in jobs.items() if key == job_id or key.split(".", 1)[0] == job_id.split(".", 1)[0]
    ]
    if len(candidates) != 1 or not isinstance(candidates[0], dict):
        raise RuntimeError("PBS accounting JSON does not uniquely contain the health job")
    return candidates[0]


def validate_post_job_evidence(result: dict[str, Any]) -> None:
    job_id = result.get("pbs_job_id")
    expected_pbs = {
        "job_id": job_id,
        "state": "F",
        "substate": 92,
        "exit_status": 0,
        "run_count": 1,
    }
    if not isinstance(job_id, str) or not job_id or result.get("pbs") != expected_pbs:
        raise RuntimeError("final RESULT lacks exact successful PBS accounting")
    accounting_ref = result.get("pbs_accounting")
    if not isinstance(accounting_ref, dict):
        raise RuntimeError("final RESULT lacks PBS accounting evidence")
    accounting_path = Path(str(accounting_ref.get("path", "")))
    accounting = read_json(accounting_path, modes={0o400, 0o600})
    if accounting_ref.get("sha256") != sha256(accounting_path):
        raise RuntimeError("PBS accounting evidence digest mismatch")
    record = _qstat_record(accounting, job_id)
    actual_pbs = {
        "job_id": job_id,
        "state": record.get("job_state"),
        "substate": int(record.get("substate", -1)),
        "exit_status": int(record.get("Exit_status", -1)),
        "run_count": int(record.get("run_count", 0)),
    }
    if actual_pbs != expected_pbs:
        raise RuntimeError("referenced PBS accounting does not prove a first-run Exit 0")

    cleanup_ref = result.get("cleanup_receipt")
    if not isinstance(cleanup_ref, dict):
        raise RuntimeError("final RESULT lacks normal cleanup evidence")
    cleanup_path = Path(str(cleanup_ref.get("path", "")))
    cleanup = read_json(cleanup_path, modes={0o600})
    if cleanup_ref != {
        "path": str(cleanup_path),
        "sha256": sha256(cleanup_path),
        "schema_version": 2,
        "status": "passed",
        "primary_status": 0,
        "shutdown_request_verified": True,
    }:
        raise RuntimeError("final RESULT cleanup receipt reference mismatch")
    expected_cleanup = {
        "schema_version": 2,
        "status": "passed",
        "pbs_job_id": job_id,
        "host": cleanup.get("host"),
        "primary_status": 0,
        "cleanup_completed": True,
        "forced_kill": False,
        "endpoints_stopped": True,
        "swe_sessions_clean": True,
        "shutdown_request_verified": True,
    }
    if not isinstance(cleanup.get("host"), str) or not cleanup["host"] or cleanup != expected_cleanup:
        raise RuntimeError("referenced cleanup receipt is not an exact normal shutdown")


def verify(root: Path, result_path: Path, gate2_result: Path | None = None) -> dict[str, Any]:
    result = read_json(result_path, modes={0o400})
    if set(result) != CANDIDATE_KEYS | FINAL_ONLY_KEYS:
        raise RuntimeError("final all-MCP RESULT field set mismatch")
    if (
        result.get("schema_version") != 1
        or result.get("gate") != "toolathlon_all_mcp_health"
        or result.get("status") != "passed"
        or not isinstance(result.get("finalized_at_epoch"), int)
    ):
        raise RuntimeError("final all-MCP RESULT identity/status mismatch")
    _, manifest = validate_release(root, result)
    if gate2_result is None:
        raise RuntimeError("explicit final v9 Gate2 RESULT is required")
    validate_gate2_evidence(root, result, gate2_result)
    _, image_sha = validate_image_and_build(result, manifest)
    validate_health_evidence(result, root)
    validate_candidate(result)
    validate_post_job_evidence(result)
    if result["finalized_at_epoch"] < result["completed_at_epoch"]:
        raise RuntimeError("final all-MCP RESULT predates its health report")
    print(
        f"toolathlon_all_mcp_health_result=PASS release_commit={result['release_commit']} "
        f"image_sha256={image_sha} pbs_job_id={result['pbs_job_id']}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--gate2-result", type=Path, required=True)
    args = parser.parse_args()
    verify(
        args.release_root.absolute(),
        args.result.absolute(),
        args.gate2_result.absolute(),
    )


if __name__ == "__main__":
    main()
