"""Structural and receipt tests for the exact all-MCP Toolathlon gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLATHLON = REPO_ROOT / "external/toolathlon-gym"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HEALTH = load_module(
    "test_toolathlon_all_mcp_health",
    REPO_ROOT / "pbs-scripts/toolathlon_all_mcp_health.py",
)
RESULT = load_module(
    "test_publish_toolathlon_all_mcp_health_result",
    REPO_ROOT / "pbs-scripts/publish_toolathlon_all_mcp_health_result.py",
)


def health_report() -> dict:
    return {
        "schema_version": 1,
        "gate": "toolathlon_all_mcp_health_driver",
        "status": "passed",
        "expected_server_count": 25,
        "tested_servers": list(RESULT.EXPECTED_SERVERS),
        "failure_count": 0,
        "catalog_sha256": RESULT.EXPECTED_REPAIRED_CATALOG_SHA256,
        "sessions": [
            {
                "task_name": task,
                "sid": f"health-{index}",
                "expected_servers": list(servers),
                "catalog_tool_count": len(servers),
                "live_probes": ["mcp_health"],
                "concurrency_barrier": task in HEALTH.CONCURRENT_TASKS,
                "mcp_health_checks": [
                    {
                        "status": "passed",
                        "server_count": len(servers),
                        "servers": {
                            server: {
                                "live_tool_count": 1,
                                "live_tool_names_sha256": "a" * 64,
                            }
                            for server in servers
                        },
                    }
                    for _ in range(2 if task in HEALTH.CONCURRENT_TASKS else 1)
                ],
            }
            for index, (task, servers) in enumerate(HEALTH.SESSION_COVER.items())
        ],
        "concurrency": {
            "status": "passed",
            "simultaneous_active_sessions_verified": 2,
            "tasks": list(HEALTH.CONCURRENT_TASKS),
            "overlapping_servers": list(HEALTH.CONCURRENT_OVERLAP),
            "both_called": ["emails.check_connection"],
        },
        "catalog_representatives": [f"{server}.representative" for server in RESULT.EXPECTED_SERVERS],
        "tested_tools": sorted(RESULT.EXPECTED_EMAIL_TOOLS | {"mcp_health", "playwright_with_chunk.browser_tab_list"}),
        "playwright": {
            "status": "passed",
            "browsers_path": "/opt/ms-playwright",
            "chromium_executable": "/opt/ms-playwright/chromium-1182/chrome-linux/chrome",
            "chromium_sha256": RESULT.EXPECTED_PLAYWRIGHT_CHROMIUM_SHA256,
            "headless_shell_executable": (
                "/opt/ms-playwright/chromium_headless_shell-1182/chrome-linux/headless_shell"
            ),
            "headless_shell_sha256": HEALTH.EXPECTED_PLAYWRIGHT_HEADLESS_SHELL_SHA256,
            "patched_config_sha256": RESULT.EXPECTED_PLAYWRIGHT_CONFIG_SHA256,
            "chromium_sandbox": "disabled_for_unprivileged_singularity",
            "browser_tab_list": "passed",
        },
        "completed_at_epoch": 1,
    }


def test_seven_task_cover_is_exact_and_concurrency_overlap_is_stable() -> None:
    HEALTH.validate_cover(TOOLATHLON / "tasks")
    union = set().union(*map(set, HEALTH.SESSION_COVER.values()))
    assert tuple(sorted(union)) == HEALTH.EXPECTED_SERVERS
    overlap = set(HEALTH.SESSION_COVER[HEALTH.CONCURRENT_TASKS[0]]) & set(
        HEALTH.SESSION_COVER[HEALTH.CONCURRENT_TASKS[1]]
    )
    assert tuple(sorted(overlap)) == HEALTH.CONCURRENT_OVERLAP
    assert overlap == {"emails", "filesystem"}


def test_health_report_requires_email_send_readback_and_concurrency() -> None:
    value = health_report()
    RESULT.validate_health_report(value)

    broken = json.loads(json.dumps(value))
    broken["tested_tools"].remove("emails.send_email")
    with pytest.raises(RuntimeError, match="email"):
        RESULT.validate_health_report(broken)

    broken = json.loads(json.dumps(value))
    broken["concurrency"]["simultaneous_active_sessions_verified"] = 1
    with pytest.raises(RuntimeError, match="concurrency"):
        RESULT.validate_health_report(broken)

    broken = json.loads(json.dumps(value))
    broken["playwright"]["browser_tab_list"] = "pending_live_probe"
    with pytest.raises(RuntimeError, match="Playwright"):
        RESULT.validate_health_report(broken)


def test_health_report_publication_is_private_and_create_only(tmp_path: Path) -> None:
    parent = tmp_path / "private"
    parent.mkdir(mode=0o700)
    report = parent / "health.json"
    value = health_report()
    HEALTH.write_report(report, value)
    assert report.stat().st_mode & 0o777 == 0o600
    assert json.loads(report.read_text()) == value
    with pytest.raises(RuntimeError, match="unsafe"):
        HEALTH.write_report(report, value)


def test_candidate_and_finalizer_accept_git_executable_health_driver(tmp_path: Path) -> None:
    driver = tmp_path / "toolathlon_all_mcp_health.py"
    driver.write_text("#!/usr/bin/env python3\n")
    driver.chmod(0o755)

    assert RESULT.HEALTH_DRIVER_MODES == {0o440, 0o550, 0o640, 0o750, 0o755}
    assert RESULT.require_private_regular(
        driver,
        modes=RESULT.HEALTH_DRIVER_MODES,
    ).st_mode & 0o777 == 0o755
    source = (REPO_ROOT / "pbs-scripts/publish_toolathlon_all_mcp_health_result.py").read_text()
    assert source.count("HEALTH_DRIVER_MODES") == 3


def test_health_driver_binds_portable_playwright_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    browsers = tmp_path / "ms-playwright"
    chromium = browsers / "chromium-1182/chrome-linux/chrome"
    chromium.parent.mkdir(parents=True)
    chromium.write_bytes(b"fixture chromium")
    chromium.chmod(0o755)
    headless = browsers / "chromium_headless_shell-1182/chrome-linux/headless_shell"
    headless.parent.mkdir(parents=True)
    headless.write_bytes(b"fixture headless shell")
    headless.chmod(0o755)
    config = tmp_path / "playwright_with_chunk.yaml"
    config.write_text('params:\n  args:\n    - "--no-sandbox"\n')
    config.chmod(0o644)
    monkeypatch.setattr(HEALTH, "PLAYWRIGHT_BROWSERS_PATH", browsers)
    monkeypatch.setattr(HEALTH, "PLAYWRIGHT_CHROMIUM_EXECUTABLE", chromium)
    monkeypatch.setattr(HEALTH, "PLAYWRIGHT_HEADLESS_SHELL_EXECUTABLE", headless)
    monkeypatch.setattr(HEALTH, "PLAYWRIGHT_CONFIG_PATH", config)
    monkeypatch.setattr(
        HEALTH,
        "EXPECTED_PLAYWRIGHT_CHROMIUM_SHA256",
        hashlib.sha256(chromium.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        HEALTH,
        "EXPECTED_PLAYWRIGHT_CONFIG_SHA256",
        hashlib.sha256(config.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        HEALTH,
        "EXPECTED_PLAYWRIGHT_HEADLESS_SHELL_SHA256",
        hashlib.sha256(headless.read_bytes()).hexdigest(),
    )
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(browsers))
    assert HEALTH.validate_playwright_runtime()["browser_tab_list"] == "pending_live_probe"

    chromium.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        HEALTH.validate_playwright_runtime()


@pytest.mark.parametrize("pack_refs", [False, True])
def test_release_validation_accepts_full_annotated_tag_ref(tmp_path: Path, pack_refs: bool) -> None:
    release = tmp_path / ("packed" if pack_refs else "loose")
    release.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=release, check=True)
    subprocess.run(
        ["git", "config", "user.email", "gate@example.invalid"],
        cwd=release,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Gate"], cwd=release, check=True)
    (release / "tracked").write_text("content\n")
    subprocess.run(["git", "add", "tracked"], cwd=release, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=release, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=release, text=True).strip()
    subprocess.run(
        ["git", "tag", "-a", "release-fixture", "-m", "fixture tag"],
        cwd=release,
        check=True,
    )
    subprocess.run(["git", "checkout", "-q", "--detach", commit], cwd=release, check=True)
    if pack_refs:
        subprocess.run(["git", "pack-refs", "--all"], cwd=release, check=True)

    RESULT.validate_release(release, "refs/tags/release-fixture", commit)
    with pytest.raises(RuntimeError, match="release ref"):
        RESULT.validate_release(release, "release-fixture", commit)


def test_qstat_accounting_requires_exact_single_success(tmp_path: Path) -> None:
    path = tmp_path / "qstat.json"
    path.write_text(
        json.dumps(
            {
                "Jobs": {
                    "123.pbs1": {
                        "job_state": "F",
                        "substate": 92,
                        "Exit_status": 0,
                        "run_count": 1,
                    }
                }
            }
        )
    )
    path.chmod(0o600)
    assert RESULT.parse_qstat(path, "123.pbs1") == {
        "job_id": "123.pbs1",
        "state": "F",
        "substate": 92,
        "exit_status": 0,
        "run_count": 1,
    }
    value = json.loads(path.read_text())
    value["Jobs"]["123.pbs1"]["run_count"] = 2
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    with pytest.raises(RuntimeError, match="PBS accounting mismatch"):
        RESULT.parse_qstat(path, "123.pbs1")
