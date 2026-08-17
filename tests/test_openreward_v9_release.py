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
RECURSIVE_WRAPPER_PATH = PBS_ROOT / "openreward-ta-swe-32node-recursive-qwen38-resume-h200-v11-abci.pbs"
NONRECURSIVE_WRAPPER_PATH = PBS_ROOT / "openreward-ta-swe-16node-nonrecursive-qwen38-gs0-nooptim-h200-v10-abci.pbs"
EXPECTED_RELEASE_ROOT = "/home/aci18912it/rasp/releases/platoon-v8r22-e11d720-057108f"
EXPECTED_RELEASE_REF = "refs/tags/abci-v8r22-e11d720-057108f"
EXPECTED_PLATOON_UPSTREAM = "e11d7201ae8118a9350210615038ff2fe582945f"
EXPECTED_OPENHANDS_REVISION = "057108f1f1235940e5d04401b4c294e4777c3baf"
HISTORICAL_V8_OPENHANDS_REVISION = "b1acf52ee362884f092b9eaf5e701a1bb5d1b335"
EXPECTED_MODEL_REPOSITORY = "apurvaga/Qwen3.8-27B-preserve-thinking"
EXPECTED_MODEL_REVISION = "97c725615b8aa025b800da4ceed38811358b30f1"
EXPECTED_MODEL_SOURCE = "Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
EXPECTED_MODEL_SNAPSHOT = (
    "/groups/gai51740/aci18912it/rasp/cache/huggingface/hub/"
    "models--apurvaga--Qwen3.8-27B-preserve-thinking/snapshots/"
    "97c725615b8aa025b800da4ceed38811358b30f1"
)
EXPECTED_BUILD_COMMIT = "69d04f210097870a984d26024016a705a278c1d2"
EXPECTED_PATCH_IDENTITY = "53541066e97beed06c4e0997969bf745a686e0542654dab763532a539317ba2b"
EXPECTED_IMAGE_PATH = "/groups/gai51740/aci18912it/rasp/images/openreward-toolathlon-18e62c0d-pbs-53541066e97b.sif"
EXPECTED_IMAGE_SHA256 = "11d158f7c1a76486f1338f8cf90ce8a0280962792c0e5c95559f19cc391cda14"
EXPECTED_RECEIPT_PATH = (
    "/groups/gai51740/aci18912it/rasp/images/build-receipts/openreward-toolathlon-18e62c0d-pbs-53541066e97b.sif.json"
)
EXPECTED_RECEIPT_SHA256 = "37286eeb005f83e3b9133ceb8204b12d93220ac02cc033118e98c5f314908598"


def _load_release_verifier(module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, VERIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(PBS_ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(PBS_ROOT))
    return module


def test_v9_manifest_records_new_identity_full_horizon_and_ptc_matrix() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())

    assert manifest["schema_version"] == 1
    assert manifest["release_root"] == EXPECTED_RELEASE_ROOT
    assert manifest["release_ref"] == EXPECTED_RELEASE_REF
    assert manifest["components"]["toolathlon"] == ("18e62c0d0419da88209af911528f2f8a0b44a5ed")
    assert manifest["components"]["platoon_upstream"] == EXPECTED_PLATOON_UPSTREAM
    assert manifest["components"]["openhands"] == EXPECTED_OPENHANDS_REVISION
    assert manifest["upstream_refresh"]["fetched_at_utc"] == "2026-08-17T21:00:22Z"
    assert {name: value["sha"] for name, value in manifest["upstream_refresh"].items() if name != "fetched_at_utc"} == {
        "platoon": EXPECTED_PLATOON_UPSTREAM,
        "openhands": EXPECTED_OPENHANDS_REVISION,
        "toolathlon": "18e62c0d0419da88209af911528f2f8a0b44a5ed",
        "swe_rebench": "287a0dcb0f747dee110037107e4792d5c6511299",
    }
    assert manifest["upstream_refresh"]["openhands"]["ref"] == "refs/heads/main"
    assert manifest["upstream_refresh"]["swe_rebench"]["release_ahead"] == 1
    assert manifest["upstream_refresh"]["swe_rebench"]["release_behind"] == 2
    verifier_module = _load_release_verifier("verify_openreward_v9_release_semantics_test")
    assert manifest["model_artifacts"] == verifier_module.MODEL_ARTIFACTS
    for artifact in manifest["model_artifacts"].values():
        snapshot = Path(artifact["snapshot_path"])
        assert snapshot.name == artifact["revision"]
        index_path = snapshot / "model.safetensors.index.json"
        assert hashlib.sha256(index_path.read_bytes()).hexdigest() == artifact["safetensors_index_sha256"]
        index = json.loads(index_path.read_text())
        assert len(set(index["weight_map"].values())) == artifact["safetensors_shard_count"]
    base_semantics = dict(manifest["training_semantics"])
    successor_contracts = base_semantics.pop("v8r22_successor_contracts")
    memory_guard = base_semantics.pop("service_node_memory_guard")
    assert base_semantics == verifier_module.TRAINING_SEMANTICS
    assert set(successor_contracts) == verifier_module.V8R22_SUCCESSOR_CONTRACT_NAMES
    assert verifier_module._object_sha256(successor_contracts) == verifier_module.V8R22_SUCCESSOR_CONTRACTS_SHA256
    assert verifier_module._object_sha256(memory_guard) == verifier_module.SERVICE_NODE_MEMORY_GUARD_SHA256
    for contract_name in (
        "nonrecursive_16node_resume_contract",
        "fresh_recursive_32node_qwen36_cp2_contract",
        "recursive_32node_qwen38_v20_successor_contract",
    ):
        contract = manifest["training_semantics"][contract_name]
        for kind in ("config", "wrapper"):
            path = REPO_ROOT / contract[f"{kind}_relative"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == contract[f"{kind}_sha256"]
    for contract in successor_contracts.values():
        for kind in ("config", "wrapper"):
            path = REPO_ROOT / contract[f"{kind}_relative"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == contract[f"{kind}_sha256"]
    successor = manifest["training_semantics"]["recursive_32node_qwen38_v20_successor_contract"]
    assert successor["binding_status"] == "frozen_gs6_checkpoint"
    assert successor["launchable"] is True
    assert successor["recover_global_step"] == 6
    assert successor["source_pbs_job_id"] == "2156202.pbs1"
    assert successor["dcp_state_keys"] == 1472
    assert successor["dcp_storage_entries"] == 18297
    assert successor["lr_scheduler_num_steps"] == 7
    assert successor["dataloader_samples_yielded"] == 336
    assert successor["dataloader_iterator_yielded"] == 42
    preupdate = manifest["training_semantics"]["nonrecursive_16node_resume_contract"]
    assert preupdate["recover_global_step"] == 0
    assert preupdate["next_global_step"] == 1
    assert preupdate["optimizer_moment_keys"] == 0
    assert preupdate["recover_no_load_optim"] is True
    assert preupdate["recover_no_save_optim"] is False
    assert preupdate["one_shot_after_checkpoint_advance"] is True
    assert preupdate["first_step_lr_alignment"] == ("fresh_lr0_fast_forward_once_then_successful_gs1_update_saves_lr2")
    assert preupdate["first_step_checkpoint_promotion_guard"] == (
        "requires_datums_update_and_weight_sync_before_overwriting_gs0"
    )
    assert preupdate["future_resume_policy"] == (
        "gs1_lr2_with_moments_then_global_step_gte_1_requires_normal_optimizer_loading"
    )
    assert preupdate["future_resume_authorization"] == (
        "fresh_exact_successor_release_from_observed_gs1_candidate_and_checkpoint_digests"
    )
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
    assert "release model artifact identity mismatch" in source
    assert manifest["training_runtime_assets"] == {
        "platoon/train/areal/rl.py": hashlib.sha256((REPO_ROOT / "platoon/train/areal/rl.py").read_bytes()).hexdigest(),
        "plugins/openreward/platoon/openreward/config_defs.py": hashlib.sha256(
            (REPO_ROOT / "plugins/openreward/platoon/openreward/config_defs.py").read_bytes()
        ).hexdigest(),
        "plugins/openreward/platoon/openreward/rollout.py": hashlib.sha256(
            (REPO_ROOT / "plugins/openreward/platoon/openreward/rollout.py").read_bytes()
        ).hexdigest(),
        "pbs-scripts/openreward-ta-swe-prealloc-common.sh": hashlib.sha256(
            (PBS_ROOT / "openreward-ta-swe-prealloc-common.sh").read_bytes()
        ).hexdigest(),
        "pbs-scripts/openreward-node-services.sh": hashlib.sha256(
            (PBS_ROOT / "openreward-node-services.sh").read_bytes()
        ).hexdigest(),
        "pbs-scripts/openreward-node-memory-guard.py": hashlib.sha256(
            (PBS_ROOT / "openreward-node-memory-guard.py").read_bytes()
        ).hexdigest(),
    }
    assert "release training runtime asset content mismatch" in source
    assert "_require_model_artifacts()" in source
    assert "release model shard set does not match its pinned" in source


def test_v9_openhands_pin_advances_without_rewriting_historical_v8_artifacts() -> None:
    active_paths = (
        PBS_ROOT / "openreward-v9-release.json",
        PBS_ROOT / "verify_openreward_v9_release.py",
        REPO_ROOT / "plugins/openreward/pyproject.toml",
        REPO_ROOT / "plugins/openreward/uv.lock",
    )
    for path in active_paths:
        source = path.read_text()
        assert EXPECTED_OPENHANDS_REVISION in source
        assert HISTORICAL_V8_OPENHANDS_REVISION not in source

    historical_paths = (
        PBS_ROOT / "openreward-v8-release.json",
        PBS_ROOT / "verify_openreward_v8_release.py",
        PBS_ROOT / "openreward_v8_ptc_e2e.py",
        PBS_ROOT / "openreward-v8-ptc-e2e-smoke.pbs",
    )
    for path in historical_paths:
        source = path.read_text()
        assert HISTORICAL_V8_OPENHANDS_REVISION in source
        assert EXPECTED_OPENHANDS_REVISION not in source

    verifier = VERIFIER_PATH.read_text()
    assert "retained v8 artifacts remain immutable" in verifier
    assert "Only their SWE contract is consumed" in verifier


def test_v9_release_binds_exact_resume_dense_qwen38_recursive_wrapper() -> None:
    overlay = OVERLAY_PATH.read_text()
    wrapper = RECURSIVE_WRAPPER_PATH.read_text()

    subprocess.run(["bash", "-n", str(RECURSIVE_WRAPPER_PATH)], check=True)
    assert "#PBS -N rasp-h200-32q38-rsp6-v20" in wrapper
    assert "REPO_ROOT=/home/aci18912it/rasp/releases/platoon-v8r21-376d1b9-18e62c0" in wrapper
    assert "RELEASE_REF=refs/tags/abci-v8r21-376d1b9-18e62c0" in wrapper
    assert (
        "CONFIG=${REPO_ROOT}/plugins/openreward/platoon/openreward/configs/areal/"
        "abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_h200_"
        "qwen38_dense_c24_ta20_v20.yaml"
    ) in wrapper
    assert "EXPERIMENT=abci-or-ta-swe32-q38-v16" in wrapper
    assert "TRIAL=abci-ta-swe32-rec-q38d-c24-v16-t0" in wrapper
    assert "export OPENREWARD_REQUIRE_EXISTING_RECOVERY=1" in wrapper
    assert "export OPENREWARD_EXPECTED_RESUME_GLOBAL_STEP=${EXPECTED_GLOBAL_STEP}" in wrapper
    assert "resume-post-step6-v8r20.success.json" in wrapper
    assert "export OPENREWARD_EXPECTED_MODEL_SHARDS=18" in wrapper
    assert "export PLATOON_MODEL_SNAPSHOT=${OPENREWARD_QWEN38_MODEL_SNAPSHOT}" in wrapper
    assert "--expected-start-global-step 7" in wrapper
    assert "--successor-binding qwen38-32-v20-post-step6" in wrapper
    assert f"OPENREWARD_QWEN38_MODEL_REPOSITORY={EXPECTED_MODEL_REPOSITORY}" in overlay
    assert f"OPENREWARD_QWEN38_MODEL_REVISION={EXPECTED_MODEL_REVISION}" in overlay
    assert f"OPENREWARD_QWEN38_UPSTREAM_SOURCE={EXPECTED_MODEL_SOURCE}" in overlay
    assert f"OPENREWARD_QWEN38_MODEL_SNAPSHOT={EXPECTED_MODEL_SNAPSHOT}" in overlay
    assert "export OPENREWARD_MODEL_FAMILY=qwen38" in wrapper
    assert wrapper.index("export OPENREWARD_MODEL_FAMILY=qwen38") < wrapper.index('source "${ABCI_ENV}"')
    assert "export PLATOON_MODEL_SNAPSHOT=${OPENREWARD_QWEN36_MODEL_SNAPSHOT}" in overlay
    assert "export PLATOON_MODEL_SNAPSHOT=${OPENREWARD_QWEN38_MODEL_SNAPSHOT}" in overlay


def test_v9_release_binds_step_zero_resume_dense_qwen38_nonrecursive_wrapper() -> None:
    wrapper = NONRECURSIVE_WRAPPER_PATH.read_text()

    subprocess.run(["bash", "-n", str(NONRECURSIVE_WRAPPER_PATH)], check=True)
    assert "#PBS -N rasp-h200-16q38-rs0n-v20" in wrapper
    assert "#PBS -l walltime=72:00:00" in wrapper
    assert "REPO_ROOT=/home/aci18912it/rasp/releases/platoon-v8r21-376d1b9-18e62c0" in wrapper
    assert "RELEASE_REF=refs/tags/abci-v8r21-376d1b9-18e62c0" in wrapper
    assert (
        "CONFIG=${REPO_ROOT}/plugins/openreward/platoon/openreward/configs/areal/"
        "abci_toolathlon_swe_openhands_areal_prealloc_16node_nonrecursive_h200_"
        "qwen38_dense_c12_ta20_v20_gs0_no_load_optim.yaml"
    ) in wrapper
    assert "EXPERIMENT=abci-or-ta-swe16-q38-v18" in wrapper
    assert "TRIAL=abci-ta-swe16-nr-q38d-c12-v18-t0" in wrapper
    assert "export OPENREWARD_REQUIRE_EXISTING_RECOVERY=1" in wrapper
    assert "export OPENREWARD_ALLOW_PREUPDATE_RESUME=1" in wrapper
    assert "export OPENREWARD_EXPECTED_RESUME_GLOBAL_STEP=0" in wrapper
    assert "export OPENREWARD_EXPECTED_START_GLOBAL_STEP=1" in wrapper
    assert "export OPENREWARD_PREUPDATE_NO_LOAD_OPTIM=1" in wrapper
    assert "resume-gs0-nooptim-v8r20.success.json" in wrapper
    assert "export OPENREWARD_EXPECTED_TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600" in wrapper
    assert (
        "export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${OPENREWARD_EXPECTED_TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC}"
    ) in wrapper
    assert "export OPENREWARD_EXPECTED_MODEL_SHARDS=18" in wrapper
    assert "export OPENREWARD_MODEL_FAMILY=qwen38" in wrapper
    assert wrapper.index("export OPENREWARD_MODEL_FAMILY=qwen38") < wrapper.index('source "${ABCI_ENV}"')
    assert "export PLATOON_MODEL_SNAPSHOT=${OPENREWARD_QWEN38_MODEL_SNAPSHOT}" in wrapper
    for literal in (
        "no_load_optim: true",
        "no_save_optim: false",
        "backend: megatron:d10p2t4c1",
        "backend: sglang:d6p1t8",
        "enable_recursive_subagents: false",
        "enable_programmatic_tool_calling: false",
    ):
        assert literal in wrapper


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


def test_v9_model_selection_is_reentrant_and_unknown_fails_closed(tmp_path: Path) -> None:
    source = OVERLAY_PATH.read_text()
    begin_marker = "# BEGIN OPENREWARD_V9_MODEL_SELECTION\n"
    end_marker = "# END OPENREWARD_V9_MODEL_SELECTION"
    begin = source.index(begin_marker) + len(begin_marker)
    end = source.index(end_marker, begin)
    selection = tmp_path / "model-selection.env"
    selection.write_text(source[begin:end], encoding="utf-8")

    completed = subprocess.run(
        [
            "bash",
            "-c",
            """
set -euo pipefail
export OPENREWARD_MODEL_FAMILY=qwen36
source "$1"
printf '%s\n' "${PLATOON_MODEL_SNAPSHOT}"
source "$1"
printf '%s\n' "${PLATOON_MODEL_SNAPSHOT}"
export OPENREWARD_MODEL_FAMILY=qwen38
source "$1"
printf '%s\n' "${PLATOON_MODEL_SNAPSHOT}"
""",
            "model-selection-test",
            str(selection),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads(MANIFEST_PATH.read_text())
    q36 = manifest["model_artifacts"]["fresh_recursive_32node_qwen36_cp2"]
    q38 = manifest["model_artifacts"]["recursive_32node"]
    assert completed.stdout.splitlines() == [
        q36["snapshot_path"],
        q36["snapshot_path"],
        q38["snapshot_path"],
    ]

    for family in ("", "unknown"):
        completed = subprocess.run(
            [
                "bash",
                "-c",
                'set -uo pipefail; export OPENREWARD_MODEL_FAMILY="$2"; source "$1"',
                "model-selection-test",
                str(selection),
                family,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 2
        assert "OPENREWARD_MODEL_FAMILY must be qwen36 or qwen38" in completed.stderr


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
