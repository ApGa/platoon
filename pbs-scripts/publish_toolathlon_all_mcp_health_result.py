#!/usr/bin/env python3
"""Create an in-job candidate, then finalize it only after PBS success.

The health job cannot truthfully attest its own final scheduler accounting.
``candidate`` binds the successful driver report, release, image and build
receipt while the job is running.  ``finalize`` runs after PBS reports
F/substate 92/Exit_status 0/run_count 1, verifies the normal-cleanup receipt,
then atomically publishes the mode-0400 production RESULT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import time
from pathlib import Path
from typing import Any

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
CONCURRENT_TASKS = (
    "terminal-canvas-howtocook-gsheet-notion-email",
    "yt-transcript-word-gform",
)
EXPECTED_REVISION = "18e62c0d0419da88209af911528f2f8a0b44a5ed"
EXPECTED_BASE_OCI = "sha256:c1e60b3673c848b758bda1b32ec39898feb5d6ff4c7e55ead8661adb04b327bd"
EXPECTED_BASE_SIF_SHA256 = "e730e11d1abc63f5cac9eb1788e36da79aa9c7027ed89d91ee42a5b1e311e267"
EXPECTED_BASE_SIF_PATH = "/groups/gai51740/aci18912it/rasp/images/openreward-toolathlon-18e62c0d-c1e60b36.sif"
EXPECTED_BASE_SERVER_SHA256 = "719ec82fb3efada36c01115df3f636f7311b40778ba7c52103d1e6950800152e"
EXPECTED_PATCHED_SERVER_SHA256 = "26f58c49d42975616c5bef19b654cc4483e665b2107533000861cecbcd74c8a5"
EXPECTED_BASE_CATALOG_SHA256 = "48c56aabcf85833c53a967b1927bdf8a3e96c600983e9aa4f830ed99ab48f057"
EXPECTED_REPAIRED_CATALOG_SHA256 = "a6bdfea0a5524ec8d198a0b2eb9b107aba89e308b50659a106abaff0002fd64d"
EXPECTED_PLAYWRIGHT_BROWSER_TREE_SHA256 = "84d979d1a301e24d7ee04009ff85b806cda195cce9d6bd1a8a686f06887805e5"
EXPECTED_PLAYWRIGHT_CHROMIUM_SHA256 = "b93030bc068a3850d4d668e69816fe4a5f04ec93db9b2f63f32f58a39f207db8"
EXPECTED_PLAYWRIGHT_CONFIG_SHA256 = "47841318775a9afc122097e769a3fcc02d3af7089d372b60fc11da1ea44f0de9"
EXPECTED_SINGULARITY_SHA256 = "085a757f172dcabda66b44d086913314d6aa574055a8fea90523be82b770cec6"
EXPECTED_SINGULARITY_VERSION = "4.4.1-1.el9"
PATCH_ASSETS = (
    "toolathlon-pbs-portable.def",
    "patch_toolathlon_pbs_portable.py",
    "repair_toolathlon_snowflake_catalog.py",
    "launch_toolathlon_python_mcp.py",
    "probe_toolathlon_python_mcp_launchers.py",
    "verify_toolathlon_pbs_portability.sh",
)
GATE2_FIELDS = {
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
GATE2_BOUND_FIELDS = (
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
CANDIDATE_FIELDS = {
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
GIT = Path("/usr/bin/git")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def require_private_regular(path: Path, *, modes: set[int], direct: bool = True) -> os.stat_result:
    if not path.is_absolute() or (direct and path.resolve() != path):
        raise RuntimeError(f"path is not direct and absolute: {path}")
    value = path.lstat()
    mode = stat.S_IMODE(value.st_mode)
    if not stat.S_ISREG(value.st_mode) or value.st_uid != os.getuid() or mode not in modes:
        raise RuntimeError(f"path is not a private user-owned regular file in modes {sorted(modes)}: {path}")
    return value


def require_image(path: Path) -> os.stat_result:
    if not path.is_absolute() or path.resolve() != path:
        raise RuntimeError(f"image path is not direct and absolute: {path}")
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or value.st_uid != os.getuid() or stat.S_IMODE(value.st_mode) != 0o550:
        raise RuntimeError(f"image is not user-owned regular mode 0550: {path}")
    return value


def read_json(path: Path, *, modes: set[int]) -> dict[str, Any]:
    require_private_regular(path, modes=modes)
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"JSON is unreadable or malformed: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON is not an object: {path}")
    return value


def require_private_parent(path: Path) -> None:
    if not path.is_absolute() or path.parent.resolve() != path.parent:
        raise RuntimeError(f"publication path is not direct and absolute: {path}")
    value = path.parent.lstat()
    if not stat.S_ISDIR(value.st_mode) or value.st_uid != os.getuid() or stat.S_IMODE(value.st_mode) & 0o077:
        raise RuntimeError(f"publication parent is not private/user-owned: {path.parent}")


def publish_create_only(path: Path, value: dict[str, Any], *, mode: int) -> None:
    require_private_parent(path)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing stale publication path: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RuntimeError(f"publication target appeared: {path}") from exc
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        temporary.unlink(missing_ok=True)
    require_private_regular(path, modes={mode})


def validate_health_report(report: dict[str, Any]) -> None:
    expected = {
        "schema_version": 1,
        "gate": "toolathlon_all_mcp_health_driver",
        "status": "passed",
        "expected_server_count": 25,
        "tested_servers": list(EXPECTED_SERVERS),
        "failure_count": 0,
        "catalog_sha256": EXPECTED_REPAIRED_CATALOG_SHA256,
    }
    mismatches = {key: (report.get(key), value) for key, value in expected.items() if report.get(key) != value}
    if mismatches:
        raise RuntimeError(f"health report identity/status mismatch: {mismatches}")
    if not isinstance(report.get("completed_at_epoch"), int) or report["completed_at_epoch"] <= 0:
        raise RuntimeError("health report completion time is invalid")
    sessions = report.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != 7:
        raise RuntimeError("health report did not complete seven sessions")
    by_task = {
        item.get("task_name"): item
        for item in sessions
        if isinstance(item, dict) and isinstance(item.get("task_name"), str)
    }
    if set(by_task) != set(EXPECTED_SESSION_COVER):
        raise RuntimeError("health report seven-session task identity mismatch")
    for task_name, expected_servers in EXPECTED_SESSION_COVER.items():
        session = by_task[task_name]
        if session.get("expected_servers") != list(expected_servers):
            raise RuntimeError(f"health report server cover mismatch for {task_name}")
        if (
            not isinstance(session.get("sid"), str)
            or not session["sid"]
            or not isinstance(session.get("catalog_tool_count"), int)
            or session["catalog_tool_count"] <= 0
            or not isinstance(session.get("live_probes"), list)
            or session.get("concurrency_barrier") != (task_name in set(CONCURRENT_TASKS))
        ):
            raise RuntimeError(f"health report session evidence mismatch for {task_name}")
        health_checks = session.get("mcp_health_checks")
        expected_check_count = 2 if task_name in CONCURRENT_TASKS else 1
        if not isinstance(health_checks, list) or len(health_checks) != expected_check_count:
            raise RuntimeError(f"health report live MCP check count mismatch for {task_name}")
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
                raise RuntimeError(f"health report live MCP identity mismatch for {task_name}")
            for evidence in servers.values():
                if (
                    not isinstance(evidence, dict)
                    or set(evidence) != {"live_tool_count", "live_tool_names_sha256"}
                    or not isinstance(evidence.get("live_tool_count"), int)
                    or evidence["live_tool_count"] <= 0
                    or not is_sha256(evidence.get("live_tool_names_sha256"))
                ):
                    raise RuntimeError(f"health report live MCP evidence mismatch for {task_name}")
    concurrency = report.get("concurrency")
    if not isinstance(concurrency, dict) or concurrency != {
        "status": "passed",
        "simultaneous_active_sessions_verified": 2,
        "tasks": [
            "terminal-canvas-howtocook-gsheet-notion-email",
            "yt-transcript-word-gform",
        ],
        "overlapping_servers": ["emails", "filesystem"],
        "both_called": ["emails.check_connection"],
    }:
        raise RuntimeError("health report concurrency proof mismatch")
    tested = report.get("tested_tools")
    if not isinstance(tested, list) or not (
        EXPECTED_EMAIL_TOOLS | {"mcp_health", "playwright_with_chunk.browser_tab_list"}
    ).issubset(tested):
        raise RuntimeError("health report lacks email, Playwright, or repeated live MCP probes")
    if report.get("playwright") != {
        "status": "passed",
        "browsers_path": "/opt/ms-playwright",
        "chromium_executable": "/opt/ms-playwright/chromium-1182/chrome-linux/chrome",
        "chromium_sha256": EXPECTED_PLAYWRIGHT_CHROMIUM_SHA256,
        "headless_shell_executable": "/opt/ms-playwright/chromium_headless_shell-1182/chrome-linux/headless_shell",
        "headless_shell_sha256": "71ebe8760eb72a99b265470d4030fef296f119e25b9aab5d3efbb6ac821483a5",
        "patched_config_sha256": EXPECTED_PLAYWRIGHT_CONFIG_SHA256,
        "chromium_sandbox": "disabled_for_unprivileged_singularity",
        "browser_tab_list": "passed",
    }:
        raise RuntimeError("health report lacks exact portable Playwright live-launch evidence")
    representatives = report.get("catalog_representatives")
    if not isinstance(representatives, list) or len(representatives) != 25:
        raise RuntimeError("health report lacks one catalog representative per server")
    representative_servers = {name.split(".", 1)[0] for name in representatives}
    if representative_servers != set(EXPECTED_SERVERS):
        raise RuntimeError("health report representative server set mismatch")


def validate_build_receipt(value: dict[str, Any], *, image: Path, image_sha256: str, receipt: Path) -> None:
    require_private_regular(receipt, modes={0o400})
    expected_top = {
        "schema_version": 1,
        "gate": "toolathlon_pbs_portable_build",
        "status": "passed",
        "authorization_scope": "image_artifact_only",
        "pbs_accounting_attested": False,
        "toolathlon_source_revision": EXPECTED_REVISION,
    }
    if any(value.get(key) != item for key, item in expected_top.items()):
        raise RuntimeError("Toolathlon build receipt top-level identity mismatch")
    derived = value.get("derived_image")
    if not isinstance(derived, dict) or derived != {
        "path": str(image),
        "sha256": image_sha256,
        "size": image.stat().st_size,
        "device": image.stat().st_dev,
        "inode": image.stat().st_ino,
        "mode": 0o550,
        "uid": os.getuid(),
    }:
        raise RuntimeError("Toolathlon build receipt derived image mismatch")
    base = value.get("base_image")
    if (
        not isinstance(base, dict)
        or base.get("path") != EXPECTED_BASE_SIF_PATH
        or base.get("sha256") != EXPECTED_BASE_SIF_SHA256
        or base.get("oci_digest") != EXPECTED_BASE_OCI
    ):
        raise RuntimeError("Toolathlon build receipt base image mismatch")
    embedded = value.get("embedded")
    expected_playwright = {
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
    if (
        not isinstance(embedded, dict)
        or embedded.get("base_server_sha256") != EXPECTED_BASE_SERVER_SHA256
        or embedded.get("patched_server_sha256") != EXPECTED_PATCHED_SERVER_SHA256
        or embedded.get("base_tool_schemas_sha256") != EXPECTED_BASE_CATALOG_SHA256
        or embedded.get("repaired_tool_schemas_sha256") != EXPECTED_REPAIRED_CATALOG_SHA256
        or embedded.get("playwright") != expected_playwright
        or embedded.get("verify") != "passed"
    ):
        raise RuntimeError("Toolathlon build receipt embedded verification mismatch")
    patch_identity = value.get("portability_patch_sha256")
    assets = value.get("patch_assets")
    if (
        not is_sha256(patch_identity)
        or not isinstance(assets, dict)
        or set(assets) != set(PATCH_ASSETS)
        or not all(is_sha256(digest) for digest in assets.values())
    ):
        raise RuntimeError("Toolathlon build receipt patch identity is invalid")
    identity_payload = "".join(f"{assets[name]}  {name}\n" for name in PATCH_ASSETS).encode()
    if hashlib.sha256(identity_payload).hexdigest() != patch_identity:
        raise RuntimeError("Toolathlon build receipt patch identity does not match assets")
    tooling = value.get("build_tooling")
    singularity = tooling.get("singularity") if isinstance(tooling, dict) else None
    if singularity != {
        "path": "/usr/bin/singularity",
        "sha256": EXPECTED_SINGULARITY_SHA256,
        "version": EXPECTED_SINGULARITY_VERSION,
    }:
        raise RuntimeError("Toolathlon build receipt runtime identity mismatch")
    labels = value.get("labels")
    if (
        not isinstance(labels, dict)
        or labels.get("org.openreward.toolathlon.playwright-browsers-sha256") != EXPECTED_PLAYWRIGHT_BROWSER_TREE_SHA256
    ):
        raise RuntimeError("Toolathlon build receipt Playwright label mismatch")


def validate_release(root: Path, ref: str, commit: str) -> None:
    if not root.is_absolute() or root.resolve() != root:
        raise RuntimeError("release root is not direct and absolute")
    value = root.lstat()
    if not stat.S_ISDIR(value.st_mode) or value.st_uid != os.getuid():
        raise RuntimeError("release root is not a user-owned directory")
    if (
        not ref.startswith("refs/tags/")
        or ref == "refs/tags/"
        or ".." in ref.split("/")
        or not commit
        or len(commit) != 40
        or any(c not in "0123456789abcdef" for c in commit)
    ):
        raise RuntimeError("release ref/commit is invalid")
    git_dir = root / ".git"
    if not git_dir.is_dir():
        raise RuntimeError("release root is not a full Git checkout")

    def git(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                [str(GIT), "-C", str(root), *arguments],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
                env={
                    "PATH": "/usr/bin:/bin",
                    "HOME": "/nonexistent",
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_CONFIG_GLOBAL": "/dev/null",
                },
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("release Git validation timed out") from exc
        if completed.returncode != 0:
            raise RuntimeError(f"release Git validation failed for {arguments!r}: {completed.stderr[-500:]}")
        return completed.stdout.strip()

    actual_commit = git("rev-parse", "--verify", "HEAD")
    tag_commit = git("rev-parse", "--verify", f"{ref}^{{commit}}")
    if actual_commit != commit:
        raise RuntimeError(f"release HEAD mismatch: {actual_commit} != {commit}")
    if tag_commit != commit:
        raise RuntimeError(f"release tag mismatch: {tag_commit} != {commit}")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("release checkout is dirty")


def validate_gate2_result(
    *,
    release_root: Path,
    result_path: Path,
    release_ref: str,
    release_commit: str,
    image: Path,
    image_sha256: str,
    build_receipt: Path,
    build_receipt_sha256: str,
    patch_identity: str,
) -> dict[str, str]:
    require_private_regular(result_path, modes={0o400})
    require_private_parent(result_path)
    verifier = release_root / "pbs-scripts/verify_openreward_v9_gate2_result.py"
    if verifier.resolve() != verifier or not verifier.is_file() or verifier.is_symlink():
        raise RuntimeError("v9 Gate2 verifier is missing or indirect")
    try:
        completed = subprocess.run(
            [
                "/usr/bin/python3",
                "-I",
                str(verifier),
                "--release-root",
                str(release_root),
                "--result",
                str(result_path),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
            env={**os.environ, "PATH": "/usr/bin:/bin"},
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("v9 Gate2 verification timed out") from exc
    if completed.returncode != 0 or "openreward_v9_gate2_result=PASS" not in completed.stdout:
        raise RuntimeError(f"v9 Gate2 verification failed: {(completed.stderr or completed.stdout)[-1000:]}")

    values: dict[str, str] = {}
    for line in result_path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            raise RuntimeError("v9 Gate2 RESULT is malformed or has duplicate fields")
        values[key] = value
    if set(values) != GATE2_FIELDS:
        raise RuntimeError("v9 Gate2 RESULT field set mismatch")
    expected = {
        "gate2": "passed",
        "gate2_schema_version": "2",
        "release_flavor": "v9",
        "release_ref": release_ref,
        "release_commit": release_commit,
        "toolathlon_sif": str(image),
        "toolathlon_sif_sha256": image_sha256,
        "toolathlon_build_receipt": str(build_receipt),
        "toolathlon_build_receipt_sha256": build_receipt_sha256,
        "toolathlon_portability_patch_sha256": patch_identity,
    }
    mismatches = {
        key: (values.get(key), expected_value)
        for key, expected_value in expected.items()
        if values.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"v9 Gate2 RESULT artifact/release mismatch: {mismatches}")
    return values


def bound_gate2(path: Path, values: dict[str, str]) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": sha256(path),
        **{field: values[field] for field in GATE2_BOUND_FIELDS},
    }


def candidate(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("PBS_JOBID") != args.pbs_job_id:
        raise RuntimeError("candidate PBS job identity does not match environment")
    image = args.toolathlon_image.absolute()
    image_stat = require_image(image)
    image_sha = sha256(image)
    report = read_json(args.health_report.absolute(), modes={0o600})
    validate_health_report(report)
    build_receipt_path = args.toolathlon_build_receipt.absolute()
    build_receipt = read_json(build_receipt_path, modes={0o400})
    validate_build_receipt(
        build_receipt,
        image=image,
        image_sha256=image_sha,
        receipt=build_receipt_path,
    )
    driver = args.health_driver.absolute()
    require_private_regular(driver, modes={0o440, 0o550, 0o640, 0o750}, direct=True)
    release_root = args.release_root.absolute()
    validate_release(release_root, args.release_ref, args.release_commit)
    if driver != release_root / "pbs-scripts/toolathlon_all_mcp_health.py":
        raise RuntimeError("health driver is not the release-pinned asset")
    gate2_path = args.gate2_result.absolute()
    build_receipt_sha = sha256(build_receipt_path)
    gate2_values = validate_gate2_result(
        release_root=release_root,
        result_path=gate2_path,
        release_ref=args.release_ref,
        release_commit=args.release_commit,
        image=image,
        image_sha256=image_sha,
        build_receipt=build_receipt_path,
        build_receipt_sha256=build_receipt_sha,
        patch_identity=build_receipt["portability_patch_sha256"],
    )

    value = {
        "schema_version": 1,
        "gate": "toolathlon_all_mcp_health_candidate",
        "status": "passed_pending_pbs_accounting",
        "release_ref": args.release_ref,
        "release_commit": args.release_commit,
        "release_root": str(args.release_root.absolute()),
        "pbs_job_id": args.pbs_job_id,
        "toolathlon_image": {
            "path": str(image),
            "sha256": image_sha,
            "size": image_stat.st_size,
            "device": image_stat.st_dev,
            "inode": image_stat.st_ino,
            "mode": stat.S_IMODE(image_stat.st_mode),
            "uid": image_stat.st_uid,
        },
        "toolathlon_source_revision": EXPECTED_REVISION,
        "base_sif_sha256": EXPECTED_BASE_SIF_SHA256,
        "base_oci_digest": EXPECTED_BASE_OCI,
        "portability_patch_sha256": build_receipt["portability_patch_sha256"],
        "patched_server_sha256": EXPECTED_PATCHED_SERVER_SHA256,
        "repaired_tool_schemas_sha256": EXPECTED_REPAIRED_CATALOG_SHA256,
        "toolathlon_build_receipt": {
            "path": str(build_receipt_path),
            "sha256": build_receipt_sha,
        },
        "gate2_result": bound_gate2(gate2_path, gate2_values),
        "health_driver": {"path": str(driver), "sha256": sha256(driver)},
        "health_report": {
            "path": str(args.health_report.absolute()),
            "sha256": sha256(args.health_report.absolute()),
        },
        "expected_server_count": 25,
        "tested_servers": list(EXPECTED_SERVERS),
        "tested_tools": report["tested_tools"],
        "sessions": report["sessions"],
        "concurrency": report["concurrency"],
        "playwright": report["playwright"],
        "failure_count": 0,
        "completed_at_epoch": report["completed_at_epoch"],
    }
    publish_create_only(args.candidate.absolute(), value, mode=0o600)
    return value


def parse_qstat(path: Path, job_id: str) -> dict[str, Any]:
    value = read_json(path, modes={0o600, 0o400})
    jobs: dict[str, Any]
    if "Jobs" in value and isinstance(value["Jobs"], dict):
        jobs = value["Jobs"]
    else:
        jobs = value
    if job_id in jobs:
        record = jobs[job_id]
    elif f"{job_id}.pbs1" in jobs:
        record = jobs[f"{job_id}.pbs1"]
    else:
        matches = [item for key, item in jobs.items() if key.split(".", 1)[0] == job_id.split(".", 1)[0]]
        if len(matches) != 1:
            raise RuntimeError(f"qstat record does not uniquely contain {job_id}")
        record = matches[0]
    if not isinstance(record, dict):
        raise RuntimeError("qstat job record is not an object")
    normalized = {
        "job_id": job_id,
        "state": record.get("job_state"),
        "substate": int(record.get("substate", -1)),
        "exit_status": int(record.get("Exit_status", -1)),
        "run_count": int(record.get("run_count", 0)),
    }
    expected = {
        "job_id": job_id,
        "state": "F",
        "substate": 92,
        "exit_status": 0,
        "run_count": 1,
    }
    if normalized != expected:
        raise RuntimeError(f"PBS accounting mismatch: {normalized} != {expected}")
    return normalized


def validate_cleanup(path: Path, job_id: str) -> tuple[dict[str, Any], str]:
    value = read_json(path, modes={0o600})
    expected = {
        "schema_version": 2,
        "status": "passed",
        "pbs_job_id": job_id,
        "primary_status": 0,
        "cleanup_completed": True,
        "forced_kill": False,
        "endpoints_stopped": True,
        "swe_sessions_clean": True,
        "shutdown_request_verified": True,
    }
    mismatches = {key: (value.get(key), item) for key, item in expected.items() if value.get(key) != item}
    if mismatches:
        raise RuntimeError(f"cleanup receipt mismatch: {mismatches}")
    return value, sha256(path)


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    candidate_path = args.candidate.absolute()
    candidate_value = read_json(candidate_path, modes={0o600})
    if (
        set(candidate_value) != CANDIDATE_FIELDS
        or candidate_value.get("schema_version") != 1
        or candidate_value.get("gate") != "toolathlon_all_mcp_health_candidate"
        or candidate_value.get("status") != "passed_pending_pbs_accounting"
    ):
        raise RuntimeError("health candidate identity/status mismatch")
    job_id = candidate_value.get("pbs_job_id")
    if not isinstance(job_id, str) or not job_id:
        raise RuntimeError("health candidate lacks PBS job identity")
    # Revalidate every mutable path immediately before authorization.  The
    # shared image directory is group-writable, so the build receipt alone is
    # never treated as proof that the current path still names tested bytes.
    image_info = candidate_value.get("toolathlon_image")
    if not isinstance(image_info, dict) or not isinstance(image_info.get("path"), str):
        raise RuntimeError("health candidate lacks Toolathlon image identity")
    image_path = Path(image_info["path"])
    image_stat = require_image(image_path)
    current_image = {
        "path": str(image_path),
        "sha256": sha256(image_path),
        "size": image_stat.st_size,
        "device": image_stat.st_dev,
        "inode": image_stat.st_ino,
        "mode": stat.S_IMODE(image_stat.st_mode),
        "uid": image_stat.st_uid,
    }
    if current_image != image_info:
        raise RuntimeError("Toolathlon image changed after the health candidate")

    build_info = candidate_value.get("toolathlon_build_receipt")
    if not isinstance(build_info, dict) or not isinstance(build_info.get("path"), str):
        raise RuntimeError("health candidate lacks build receipt identity")
    build_path = Path(build_info["path"])
    build_value = read_json(build_path, modes={0o400})
    if sha256(build_path) != build_info.get("sha256"):
        raise RuntimeError("Toolathlon build receipt changed after candidate")
    validate_build_receipt(
        build_value,
        image=image_path,
        image_sha256=current_image["sha256"],
        receipt=build_path,
    )

    for field, modes in (("health_driver", {0o440, 0o550, 0o640, 0o750}), ("health_report", {0o600})):
        info = candidate_value.get(field)
        if not isinstance(info, dict) or not isinstance(info.get("path"), str):
            raise RuntimeError(f"health candidate lacks {field} identity")
        bound_path = Path(info["path"])
        require_private_regular(bound_path, modes=modes)
        if sha256(bound_path) != info.get("sha256"):
            raise RuntimeError(f"{field} changed after candidate")
    validate_release(
        Path(candidate_value["release_root"]),
        candidate_value["release_ref"],
        candidate_value["release_commit"],
    )
    gate2_info = candidate_value.get("gate2_result")
    if not isinstance(gate2_info, dict) or not isinstance(gate2_info.get("path"), str):
        raise RuntimeError("health candidate lacks v9 Gate2 identity")
    gate2_path = Path(gate2_info["path"])
    gate2_values = validate_gate2_result(
        release_root=Path(candidate_value["release_root"]),
        result_path=gate2_path,
        release_ref=candidate_value["release_ref"],
        release_commit=candidate_value["release_commit"],
        image=image_path,
        image_sha256=current_image["sha256"],
        build_receipt=build_path,
        build_receipt_sha256=build_info["sha256"],
        patch_identity=candidate_value["portability_patch_sha256"],
    )
    if bound_gate2(gate2_path, gate2_values) != gate2_info:
        raise RuntimeError("v9 Gate2 RESULT changed after health candidate")

    pbs = parse_qstat(args.qstat_json.absolute(), job_id)
    cleanup_path = args.cleanup_receipt.absolute()
    _, cleanup_sha = validate_cleanup(cleanup_path, job_id)

    value = dict(candidate_value)
    value.update(
        {
            "gate": "toolathlon_all_mcp_health",
            "status": "passed",
            "candidate": {
                "path": str(candidate_path),
                "sha256": sha256(candidate_path),
            },
            "pbs": pbs,
            "pbs_accounting": {
                "path": str(args.qstat_json.absolute()),
                "sha256": sha256(args.qstat_json.absolute()),
            },
            "cleanup_receipt": {
                "path": str(cleanup_path),
                "sha256": cleanup_sha,
                "schema_version": 2,
                "status": "passed",
                "primary_status": 0,
                "shutdown_request_verified": True,
            },
            "finalized_at_epoch": int(time.time()),
        }
    )
    publish_create_only(args.result.absolute(), value, mode=0o400)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    candidate_parser = commands.add_parser("candidate")
    candidate_parser.add_argument("--candidate", type=Path, required=True)
    candidate_parser.add_argument("--health-report", type=Path, required=True)
    candidate_parser.add_argument("--release-root", type=Path, required=True)
    candidate_parser.add_argument("--release-ref", required=True)
    candidate_parser.add_argument("--release-commit", required=True)
    candidate_parser.add_argument("--toolathlon-image", type=Path, required=True)
    candidate_parser.add_argument("--toolathlon-build-receipt", type=Path, required=True)
    candidate_parser.add_argument("--health-driver", type=Path, required=True)
    candidate_parser.add_argument("--gate2-result", type=Path, required=True)
    candidate_parser.add_argument("--pbs-job-id", required=True)
    finalize_parser = commands.add_parser("finalize")
    finalize_parser.add_argument("--candidate", type=Path, required=True)
    finalize_parser.add_argument("--result", type=Path, required=True)
    finalize_parser.add_argument("--qstat-json", type=Path, required=True)
    finalize_parser.add_argument("--cleanup-receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    value = candidate(args) if args.command == "candidate" else finalize(args)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
