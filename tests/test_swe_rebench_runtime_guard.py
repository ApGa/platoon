from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_GUARD = REPO_ROOT / "plugins/openreward/swe-rebench-runtime-guard.sh"
REVISION_PIN = REPO_ROOT / "plugins/openreward/swe-rebench-source-revision.txt"
MIXED_LAUNCHERS = (
    REPO_ROOT / "slurm-scripts/openreward-multienv-prealloc.sh",
    REPO_ROOT / "slurm-scripts/openreward-multienv-prealloc-16node-ptc-task-tracker-full.sh",
)
MIXED_SERVER = REPO_ROOT / "slurm-scripts/openreward-multienv-server.sh"
PREFLIGHT = REPO_ROOT / "slurm-scripts/openreward-swe-rebench-preflight.sh"
KNOWN_UNSAFE_REVISION = "10c49ab856fd0e62097815ba5909dfc4f31e7f93"


def _run_guard(
    repo_root: Path,
    source_root: Path,
    requested_revision: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "/bin/bash",
            "-c",
            'source "$GUARD"; '
            "swe_rebench_require_validated_source "
            '"$REPO_ROOT" "$SOURCE_ROOT" "$REQUESTED_REVISION" test; '
            'status=$?; printf "%s" "${SWE_REBENCH_VALIDATED_SOURCE_REVISION:-}"; '
            "exit $status",
        ],
        env={
            **os.environ,
            "GUARD": str(RUNTIME_GUARD),
            "REPO_ROOT": str(repo_root),
            "SOURCE_ROOT": str(source_root),
            "REQUESTED_REVISION": requested_revision,
        },
        capture_output=True,
        text=True,
    )


def _write_executable(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/bash\nset -euo pipefail\n{body}\n")
    path.chmod(0o755)


def _run_enroot_runtime_guard(
    tmp_path: Path,
    *,
    capabilities: str = "cap_sys_admin,cap_mknod=ep",
    helper_body: str = (
        'rm -f -- "$1/opaque/.wh..wh..opq" "$1/.wh.deleted"; '
        ': >"$1/deleted"'
    ),
    helper_executable: bool = True,
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    probe_root = tmp_path / "probe"
    bin_dir.mkdir(exist_ok=True)
    probe_root.mkdir(exist_ok=True)
    enroot = bin_dir / "enroot"
    helper = bin_dir / "enroot-aufs2ovlfs"
    getcap = bin_dir / "getcap"
    _write_executable(enroot, "exit 0")
    _write_executable(helper, helper_body)
    if not helper_executable:
        helper.chmod(0o644)
    _write_executable(getcap, f'printf "%s {capabilities}\\n" "$1"')

    return subprocess.run(
        [
            "/bin/bash",
            "-c",
            'source "$GUARD"; '
            'swe_rebench_require_enroot_runtime "$PROBE_ROOT" test',
        ],
        env={
            **os.environ,
            "GUARD": str(RUNTIME_GUARD),
            "PROBE_ROOT": str(probe_root),
            "SWE_REBENCH_ENROOT_BIN": str(enroot),
            "SWE_REBENCH_ENROOT_AUFS2OVLFS_BIN": str(helper),
            "SWE_REBENCH_GETCAP_BIN": str(getcap),
        },
        capture_output=True,
        text=True,
    )


def _make_clean_source_checkout(tmp_path: Path) -> tuple[Path, str]:
    source_root = tmp_path / "source"
    source_root.mkdir()
    subprocess.run(["git", "init", "-q", str(source_root)], check=True)
    (source_root / "tracked.txt").write_text("runtime\n")
    subprocess.run(["git", "-C", str(source_root), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "-c",
            "user.name=OpenReward test",
            "-c",
            "user.email=openreward-test@example.invalid",
            "commit",
            "-qm",
            "runtime fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    return source_root, revision


def _write_revision_pin(repo_root: Path, revision: str) -> None:
    pin_dir = repo_root / "plugins/openreward"
    pin_dir.mkdir(parents=True)
    (pin_dir / REVISION_PIN.name).write_text(f"{revision}\n")


def test_runtime_guard_rejects_known_unsafe_revision_before_checkout(tmp_path: Path):
    _write_revision_pin(tmp_path, KNOWN_UNSAFE_REVISION)

    result = _run_guard(tmp_path, tmp_path / "missing-source")

    assert result.returncode == 2
    assert "blocked by the sandbox-incident denylist" in result.stderr
    assert KNOWN_UNSAFE_REVISION in result.stderr


def test_runtime_guard_accepts_only_matching_clean_checkout(tmp_path: Path):
    source_root, revision = _make_clean_source_checkout(tmp_path)
    repo_root = tmp_path / "platoon"
    _write_revision_pin(repo_root, revision)

    accepted = _run_guard(repo_root, source_root, revision)
    assert accepted.returncode == 0, accepted.stderr
    assert accepted.stdout == revision

    wrong_revision = "f" * 40 if revision != "f" * 40 else "e" * 40
    mismatched = _run_guard(repo_root, source_root, wrong_revision)
    assert mismatched.returncode == 2
    assert "requested SWE-rebench revision" in mismatched.stderr

    (source_root / "tracked.txt").write_text("locally modified\n")
    dirty = _run_guard(repo_root, source_root, revision)
    assert dirty.returncode == 2
    assert "requires a clean SWE-rebench worktree" in dirty.stderr


def test_enroot_runtime_guard_accepts_capable_helper_and_runs_whiteout_probe(tmp_path: Path):
    result = _run_enroot_runtime_guard(tmp_path)

    assert result.returncode == 0, result.stderr
    assert not list((tmp_path / "probe").iterdir())


def test_enroot_runtime_guard_rejects_non_executable_helper(tmp_path: Path):
    result = _run_enroot_runtime_guard(tmp_path, helper_executable=False)

    assert result.returncode == 2
    assert "non-executable Enroot dependency" in result.stderr
    assert "Enroot +caps" in result.stderr


def test_enroot_runtime_guard_accepts_equivalent_packaging_without_reported_file_caps(tmp_path: Path):
    result = _run_enroot_runtime_guard(tmp_path, capabilities="cap_mknod=ep")

    assert result.returncode == 0, result.stderr


def test_enroot_runtime_guard_rejects_unusable_capabilities(tmp_path: Path):
    result = _run_enroot_runtime_guard(
        tmp_path,
        helper_body='echo "failed to create opaque ovlfs whiteout: Operation not permitted" >&2; exit 1',
    )

    assert result.returncode == 2
    assert "cannot use enroot-aufs2ovlfs" in result.stderr
    assert "host initial user namespace" in result.stderr
    assert "trusted overlay xattrs" in result.stderr


def test_enroot_runtime_guard_rejects_noop_whiteout_helper(tmp_path: Path):
    result = _run_enroot_runtime_guard(tmp_path, helper_body="exit 0")

    assert result.returncode == 2
    assert "did not convert both opaque and ordinary whiteouts" in result.stderr


def test_training_and_preflight_share_guard_and_allocation_local_session_root():
    subprocess.run(
        [
            "bash",
            "-n",
            str(RUNTIME_GUARD),
            *(str(path) for path in MIXED_LAUNCHERS),
            str(MIXED_SERVER),
            str(PREFLIGHT),
        ],
        check=True,
    )

    for launcher in MIXED_LAUNCHERS:
        contents = launcher.read_text()
        assert "swe-rebench-runtime-guard.sh" in contents
        assert "swe_rebench_require_validated_source" in contents
        assert ("export SWE_REBENCH_SOURCE_REVISION=${SWE_REBENCH_VALIDATED_SOURCE_REVISION}") in contents

    server = MIXED_SERVER.read_text()
    assert "swe-rebench-runtime-guard.sh" in server
    assert "swe_rebench_require_validated_source" in server
    assert "swe_rebench_require_enroot_runtime" in server
    assert "ENROOT_TEMP_PATH=${LOCAL_ROOT}/enroot-tmp" in server
    assert "SWE_ENROOT_SESSION_TMP_ROOT=${LOCAL_ROOT}/swe-enroot-sessions" in server
    assert 'mkdir -p -m 0700 "${SWE_ENROOT_SESSION_TMP_ROOT}"' in server
    assert 'chmod 0700 "${SWE_ENROOT_SESSION_TMP_ROOT}"' in server

    preflight = PREFLIGHT.read_text()
    assert "swe-rebench-runtime-guard.sh" in preflight
    assert "swe_rebench_require_validated_source" in preflight
    assert "swe_rebench_require_enroot_runtime" in preflight
    assert preflight.index('if [[ "${MODE}" == metadata ]]') < preflight.index(
        "swe_rebench_require_enroot_runtime"
    )
    assert 'chmod 0700 "${SWE_ENROOT_SESSION_TMP_ROOT}"' in preflight


def test_repository_pin_is_one_exact_commit_and_old_pin_remains_denied():
    assert re.fullmatch(r"[0-9a-f]{40}\n", REVISION_PIN.read_text())
    assert KNOWN_UNSAFE_REVISION in RUNTIME_GUARD.read_text()
