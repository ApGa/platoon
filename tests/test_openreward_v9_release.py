"""Fail-closed scaffolding checks for the PBS-portable v9 release wave."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PBS_ROOT = REPO_ROOT / "pbs-scripts"
MANIFEST_PATH = PBS_ROOT / "openreward-v9-release.json"
OVERLAY_PATH = PBS_ROOT / "openreward-v9-abci.env"
VERIFIER_PATH = PBS_ROOT / "verify_openreward_v9_release.py"


def test_v9_manifest_records_new_identity_full_horizon_and_ptc_matrix() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())

    assert manifest["schema_version"] == 1
    assert manifest["release_root"] == ("/home/aci18912it/rasp/releases/platoon-v8r3-376d1b9-18e62c0")
    assert manifest["release_ref"] == "refs/tags/abci-v8r3-376d1b9-18e62c0"
    assert manifest["components"]["toolathlon"] == ("18e62c0d0419da88209af911528f2f8a0b44a5ed")
    assert manifest["upstream_refresh"]["fetched_at_utc"] == "2026-08-13T16:38:39Z"
    assert {name: value["sha"] for name, value in manifest["upstream_refresh"].items() if name != "fetched_at_utc"} == {
        "platoon": "376d1b9885ea40464766fb3fea74815c6a698fbf",
        "openhands": "b1acf52ee362884f092b9eaf5e701a1bb5d1b335",
        "toolathlon": "18e62c0d0419da88209af911528f2f8a0b44a5ed",
        "swe_rebench": "25b14c06b9236c075a4ede25bff6979e5783bb09",
    }
    assert manifest["training_semantics"] == {
        "full_horizon": True,
        "total_train_epochs": 10,
        "total_train_steps": None,
        "fresh_16node_model_only_lineage": True,
        "nonrecursive_16node_ptc": False,
        "recursive_32node_ptc": True,
        "recursive_32node_ptc_mode": "orchestration_only",
        "recursive_32node_ptc_max_tool_calls_per_execution": 1024,
    }
    assert manifest["toolathlon_image_derivation"]["playwright"] == {
        "browsers_path": "/opt/ms-playwright",
        "base_browser_tree_sha256": "84d979d1a301e24d7ee04009ff85b806cda195cce9d6bd1a8a686f06887805e5",
        "portable_browser_tree_sha256": "84d979d1a301e24d7ee04009ff85b806cda195cce9d6bd1a8a686f06887805e5",
        "browser_file_count": 156,
        "browser_total_bytes": 959511402,
        "browser_symlink_count": 0,
        "browser_directory_count": 13,
        "chromium_revision": "1182",
        "chromium_executable": "/opt/ms-playwright/chromium-1182/chrome-linux/chrome",
        "chromium_sha256": "b93030bc068a3850d4d668e69816fe4a5f04ec93db9b2f63f32f58a39f207db8",
        "headless_shell_executable": "/opt/ms-playwright/chromium_headless_shell-1182/chrome-linux/headless_shell",
        "headless_shell_sha256": "71ebe8760eb72a99b265470d4030fef296f119e25b9aab5d3efbb6ac821483a5",
        "headless_shell_bytes": 305228784,
        "base_config_sha256": "66f5bc92c182bf365e499d8237308452a016d8ca9c5e878d1628393647c82d1a",
        "patched_config_sha256": "47841318775a9afc122097e769a3fcc02d3af7089d372b60fc11da1ea44f0de9",
        "chromium_sandbox": "disabled_for_unprivileged_singularity",
        "rootless_launch": "passed",
    }


def test_v9_scaffold_cannot_launch_before_repaired_image_identity_lands() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    overlay = OVERLAY_PATH.read_text()
    verifier = VERIFIER_PATH.read_text()

    image = manifest["toolathlon_image"]
    derivation = manifest["toolathlon_image_derivation"]
    assert "PENDING" in image["path"]
    assert image["sha256"] == "PENDING_TOOLATHLON_SIF_SHA256"
    assert derivation["status"] == "pending"
    assert any("PENDING" in str(value) for value in derivation.values())
    assert f"OPENREWARD_TOOLATHLON_SIF={image['path']}" in overlay
    assert f"OPENREWARD_TOOLATHLON_SIF_SHA256={image['sha256']}" in overlay
    assert "v9 Toolathlon image identity is pending and cannot launch" in verifier
    assert 'derivation.get("status") != "ready"' in verifier
    assert 'derivation.get("playwright") != PLAYWRIGHT_IDENTITY' in verifier


def test_v9_verifier_binds_exact_manifest_tag_externals_and_swe_contract() -> None:
    source = VERIFIER_PATH.read_text()
    digest = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    match = re.search(r'^MANIFEST_SHA256 = "([0-9a-f]{64})"$', source, re.MULTILINE)

    assert match is not None and match.group(1) == digest
    assert 'RELEASE_ROOT = Path("/home/aci18912it/rasp/releases/platoon-v8r3-376d1b9-18e62c0")' in source
    assert 'RELEASE_REF = "refs/tags/abci-v8r3-376d1b9-18e62c0"' in source
    assert '"status", "--porcelain=v1", "--untracked-files=all"' in source
    assert "release HEAD does not match its pinned tag" in source
    assert "materialized external revision mismatch" in source
    assert "v9 and retained v8 SWE runtime contracts differ" in source
    assert "verify_swe_runtime(root)" in source
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert "pbs-scripts/finalize_openreward_v9_acceptance.py" in manifest["host_acceptance_assets"]
    assert "HOST_ACCEPTANCE_ASSETS" in source
    assert "release host acceptance asset set mismatch" in source


def test_v9_overlay_and_verifier_have_valid_syntax_and_safe_current_tree_failure() -> None:
    subprocess.run(["bash", "-n", str(OVERLAY_PATH)], check=True)
    subprocess.run([sys.executable, "-m", "py_compile", str(VERIFIER_PATH)], check=True)
    assert VERIFIER_PATH.stat().st_mode & 0o111 == 0o111

    completed = subprocess.run(
        [sys.executable, str(VERIFIER_PATH), "--release-root", str(REPO_ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "release root is not the exact canonical path" in completed.stderr


def test_v9_launch_path_uses_derived_labels_without_weakening_v8_validation() -> None:
    overlay = OVERLAY_PATH.read_text()
    base_sha = "e730e11d1abc63f5cac9eb1788e36da79aa9c7027ed89d91ee42a5b1e311e267"

    assert f"OPENREWARD_TOOLATHLON_BASE_SIF_SHA256={base_sha}" in overlay
    assert "PLATOON_CONTAINER_RUNTIME=/usr/bin/singularity" in overlay
    assert "OPENREWARD_TOOLATHLON_PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright" in overlay
    assert (
        "OPENREWARD_TOOLATHLON_PORTABILITY_PATCH_SHA256="
        "80d78d63036972eab73efcdca93cb89c9f1bb8674d1bf22992c7ec6b262ed927"
    ) in overlay
    for relative in (
        "openreward-ta-swe-prealloc-common.sh",
        "openreward-node-services.sh",
    ):
        source = (PBS_ROOT / relative).read_text()
        assert "OPENREWARD_TOOLATHLON_BASE_SIF_SHA256" in source
        assert "OPENREWARD_TOOLATHLON_PORTABILITY_PATCH_SHA256" in source
        assert "org.openreward.toolathlon.base-sif-sha256:" in source
        assert "org.openreward.toolathlon.base-oci-digest:" in source
        assert "org.openreward.toolathlon.pbs-portability:" in source
        assert "Toolathlon derived SIF portability patch mismatch" in source
        # The legacy v8 OCI-origin check remains the explicit fallback when no
        # derived-image identity is supplied by its unchanged overlay.
        assert 'elif [[ -n "${TOOLATHLON_OCI_AMD64_DIGEST}" ]]; then' in source
        assert "org.label-schema.usage.singularity.deffile.from:" in source
    services = (PBS_ROOT / "openreward-node-services.sh").read_text()
    assert "toolathlon_portable_env=()" in services
    assert 'if [[ -n "${TOOLATHLON_PORTABILITY_PATCH_SHA256}" ]]; then' in services
    assert "UV_PYTHON_INSTALL_DIR=/opt/uv/python" in services
    assert "UV_PYTHON_PREFERENCE=only-managed" in services
    assert "UV_PYTHON_DOWNLOADS=never" in services
    assert '[[ "${TOOLATHLON_PLAYWRIGHT_BROWSERS_PATH}" == /opt/ms-playwright ]]' in services
    assert "PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright" in services
    assert "UV_CACHE_DIR=/mnt/state/cache/uv" in services
    assert '"${toolathlon_portable_env[@]}"' in services
    assert "PATH=/opt/venv/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin" in services
