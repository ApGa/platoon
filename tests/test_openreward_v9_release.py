"""Fail-closed scaffolding checks for the PBS-portable v9 release wave."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PBS_ROOT = REPO_ROOT / "pbs-scripts"
MANIFEST_PATH = PBS_ROOT / "openreward-v9-release.json"
OVERLAY_PATH = PBS_ROOT / "openreward-v9-abci.env"
VERIFIER_PATH = PBS_ROOT / "verify_openreward_v9_release.py"
EXPECTED_RELEASE_ROOT = "/home/aci18912it/rasp/releases/platoon-v8r9-376d1b9-18e62c0"
EXPECTED_RELEASE_REF = "refs/tags/abci-v8r9-376d1b9-18e62c0"
EXPECTED_BUILD_COMMIT = "69d04f210097870a984d26024016a705a278c1d2"
EXPECTED_PATCH_IDENTITY = "53541066e97beed06c4e0997969bf745a686e0542654dab763532a539317ba2b"
EXPECTED_IMAGE_PATH = "/groups/gai51740/aci18912it/rasp/images/openreward-toolathlon-18e62c0d-pbs-53541066e97b.sif"
EXPECTED_IMAGE_SHA256 = "11d158f7c1a76486f1338f8cf90ce8a0280962792c0e5c95559f19cc391cda14"
EXPECTED_RECEIPT_PATH = (
    "/groups/gai51740/aci18912it/rasp/images/build-receipts/openreward-toolathlon-18e62c0d-pbs-53541066e97b.sif.json"
)
EXPECTED_RECEIPT_SHA256 = "37286eeb005f83e3b9133ceb8204b12d93220ac02cc033118e98c5f314908598"


def test_v9_manifest_records_new_identity_full_horizon_and_ptc_matrix() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())

    assert manifest["schema_version"] == 1
    assert manifest["release_root"] == EXPECTED_RELEASE_ROOT
    assert manifest["release_ref"] == EXPECTED_RELEASE_REF
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
        "qwen35_gdn_cp_size": 1,
        "qwen35_gdn_packed": False,
        "qwen35_gdn_layout_selection": "model_architecture_and_context_parallel_topology",
        "qwen35_gdn_cp1_layout": "padded_bshd_upstream_gdn",
        "qwen35_gdn_cp_gt1_layout": "packed_thd_platoon_custom_gdn_cp",
        "r3_padded_routes_layout": "sequence_major",
        "dynamic_bs": True,
        "filter_zero_advantage_datums": True,
        "filter_zero_variance_groups": True,
        "rollout_max_head_offpolicyness": 3,
        "rollout_consumer_batch_size": 8,
        "max_concurrent_rollouts_16node": 12,
        "max_concurrent_rollouts_32node": 24,
        "balance_accepted_batches": False,
        "fresh_16node_model_only_lineage": True,
        "fresh_16node_lineage": "v12_wandb_safe",
        "fresh_32node_lineage": "v11_signal_filter",
        "nonrecursive_16node_ptc": False,
        "recursive_32node_ptc": True,
        "recursive_32node_ptc_mode": "orchestration_only",
        "recursive_32node_ptc_max_tool_calls_per_execution": 1024,
        "recursive_32node_subagent_datum_keep_probability": 0.25,
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


def test_v9_release_binds_ready_repaired_image_and_build_receipt() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    overlay = OVERLAY_PATH.read_text()
    verifier = VERIFIER_PATH.read_text()

    image = manifest["toolathlon_image"]
    derivation = manifest["toolathlon_image_derivation"]
    assert image["path"] == EXPECTED_IMAGE_PATH
    assert image["sha256"] == EXPECTED_IMAGE_SHA256
    assert derivation["status"] == "ready"
    assert derivation["build_receipt_path"] == EXPECTED_RECEIPT_PATH
    assert derivation["build_receipt_sha256"] == EXPECTED_RECEIPT_SHA256
    assert derivation["build_commit"] == EXPECTED_BUILD_COMMIT
    assert derivation["pbs_portability_patch_sha256"] == EXPECTED_PATCH_IDENTITY
    assert "PENDING" not in json.dumps(manifest)
    assert f"OPENREWARD_TOOLATHLON_SIF={image['path']}" in overlay
    assert f"OPENREWARD_TOOLATHLON_SIF_SHA256={image['sha256']}" in overlay
    assert f"OPENREWARD_TOOLATHLON_BUILD_RECEIPT={EXPECTED_RECEIPT_PATH}" in overlay
    assert f"OPENREWARD_TOOLATHLON_BUILD_RECEIPT_SHA256={EXPECTED_RECEIPT_SHA256}" in overlay
    assert "v9 Toolathlon image identity is not final and cannot launch" in verifier
    assert 'derivation.get("status") != "ready"' in verifier
    assert 'derivation.get("playwright") != PLAYWRIGHT_IDENTITY' in verifier

    provenance = derivation["build_receipt_provenance"]
    assert provenance["schema_version"] == 1
    assert provenance["gate"] == "toolathlon_pbs_portable_build"
    assert provenance["status"] == "passed"
    assert provenance["authorization_scope"] == "image_artifact_only"
    assert provenance["pbs_accounting_attested"] is False
    assert provenance["completed_at_epoch"] == 1786678639
    assert provenance["derived_image"] == {
        "path": EXPECTED_IMAGE_PATH,
        "sha256": EXPECTED_IMAGE_SHA256,
        "size": 1645539328,
        "device": 1214686570,
        "inode": 414331859103761933,
        "mode": 0o550,
        "uid": 18912,
    }
    assert provenance["build_tooling"] == {
        "builder": {
            "name": "build_toolathlon_pbs_portable_sif.sh",
            "sha256": "04839fcb4fcae72ad065de2b8863a0f494efc681cb71403e67b28e390780e70b",
        },
        "publisher": {
            "name": "publish_toolathlon_pbs_build_receipt.py",
            "sha256": "07a60a958948ae672aa708e4eb291a3f82746035463573216df2736aec9e0870",
        },
        "image_publisher": {
            "name": "publish_toolathlon_pbs_image.py",
            "sha256": "d5ae5d34b2192c3d7f232c0f535bb4833fc277ef9ae265b82070f0f3abaf457d",
        },
        "singularity": {
            "path": "/usr/bin/singularity",
            "sha256": "085a757f172dcabda66b44d086913314d6aa574055a8fea90523be82b770cec6",
            "version": "4.4.1-1.el9",
        },
    }
    assert provenance["embedded"] == {
        "manifest_path": "/usr/local/share/openreward/toolathlon-pbs-portable.json",
        "manifest_sha256": "a1cfac5ea288af21c42c3c13de3ab6e21eae7a46b25c9070347e3ddad5e39c2d",
        "assets": {
            "launch_toolathlon_python_mcp.py": ("409c978c81ad5f8e89c6454878a1075820f936ac4a9014b141b526fdeefbfd00"),
            "patch_toolathlon_pbs_portable.py": ("7c7caa3f7fa59807c111476bc61f1d2e40281e015d0bc326dab7cc3b92bc5421"),
            "probe_toolathlon_python_mcp_launchers.py": (
                "cf23c165adfc5d19c1d6a40117e0c257ecf519102140b053854af97f87534d65"
            ),
            "repair_toolathlon_snowflake_catalog.py": (
                "97be7bb955c04967bb5a54f4ffdf42bd5c00d5216314dd7938ad9ac3a5c6ab6a"
            ),
            "verify_toolathlon_pbs_portability.sh": (
                "818c9cd4ecd965295b0370ffe4a60695da4ffea678ea51a49cbf1e2b5048e14e"
            ),
        },
        "verify": "passed",
    }
    assert provenance["labels"]["org.label-schema.usage.singularity.deffile.from"].startswith("/local/2137418.pbs1/")
    assert provenance["labels"]["org.openreward.toolathlon.pbs-portability"] == EXPECTED_PATCH_IDENTITY
    python_mcp = derivation["python_mcp"]
    assert python_mcp["policy"] == "baked_venv_no_uv"
    assert python_mcp["audited_python_server_count"] == 13
    assert python_mcp["rewritten_uv_config_count"] == 12
    assert python_mcp["wrapper_sha256"] == provenance["embedded"]["assets"]["launch_toolathlon_python_mcp.py"]
    assert python_mcp["probe_sha256"] == provenance["embedded"]["assets"]["probe_toolathlon_python_mcp_launchers.py"]
    assert (
        python_mcp["unprivileged_exact_launch_initialize_live_name_set_persistence_clean_teardown_no_mutation"]
        == "passed"
    )
    snowflake = python_mcp["snowflake_mcp_runtime"]
    assert snowflake["installed_mcp_version"] == "1.26.0"
    assert snowflake["installed_closure_sha256"] == ("b618e4359067acb69893820bae6a970c4d8a3714d77550c972346f6120497bc5")
    assert "set(receipt) != RECEIPT_KEYS" in verifier
    assert "receipt_provenance != provenance" in verifier
    assert "v9 Toolathlon build tooling identity mismatch" in verifier
    assert "v9 Toolathlon derived SIF labels mismatch" in verifier


def test_v9_verifier_binds_exact_manifest_tag_externals_and_swe_contract() -> None:
    source = VERIFIER_PATH.read_text()
    digest = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    match = re.search(r'^MANIFEST_SHA256 = "([0-9a-f]{64})"$', source, re.MULTILINE)

    assert match is not None and match.group(1) == digest
    assert f'RELEASE_ROOT = Path("{EXPECTED_RELEASE_ROOT}")' in source
    assert f'RELEASE_REF = "{EXPECTED_RELEASE_REF}"' in source
    assert f'TOOLATHLON_BUILD_COMMIT = "{EXPECTED_BUILD_COMMIT}"' in source
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


def test_v9_verifier_rejects_hardlinked_release_artifacts(tmp_path: Path) -> None:
    module_name = "verify_openreward_v9_release_hardlink_test"
    spec = importlib.util.spec_from_file_location(module_name, VERIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(PBS_ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(PBS_ROOT))

    artifact = tmp_path / "artifact.sif"
    alias = tmp_path / "artifact.alias.sif"
    artifact.write_bytes(b"immutable-image\n")
    artifact.chmod(0o550)
    os.link(artifact, alias)
    with pytest.raises(RuntimeError, match="release artifact identity/mode mismatch"):
        module._require_owned_regular(artifact, mode=0o550)


def test_v9_launch_path_uses_derived_labels_without_weakening_v8_validation() -> None:
    overlay = OVERLAY_PATH.read_text()
    base_sha = "e730e11d1abc63f5cac9eb1788e36da79aa9c7027ed89d91ee42a5b1e311e267"

    assert f"OPENREWARD_TOOLATHLON_BASE_SIF_SHA256={base_sha}" in overlay
    assert "PLATOON_CONTAINER_RUNTIME=/usr/bin/singularity" in overlay
    assert "OPENREWARD_TOOLATHLON_PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright" in overlay
    assert f"OPENREWARD_TOOLATHLON_PORTABILITY_PATCH_SHA256={EXPECTED_PATCH_IDENTITY}" in overlay
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
