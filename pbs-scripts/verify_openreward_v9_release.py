#!/usr/bin/env python3
"""Fail-closed verification for the immutable ABCI v8r9/v9 checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path

from verify_openreward_v8_swe_env import verify as verify_swe_runtime

RELEASE_ROOT = Path("/home/aci18912it/rasp/releases/platoon-v8r9-376d1b9-18e62c0")
RELEASE_REF = "refs/tags/abci-v8r9-376d1b9-18e62c0"
MANIFEST_RELATIVE = Path("pbs-scripts/openreward-v9-release.json")
MANIFEST_SHA256 = "0f68447140a4274b6d194a89b561972e6b5ac82160ac7e7fc3eab6a4287803d5"
UPSTREAM_COMMIT = "376d1b9885ea40464766fb3fea74815c6a698fbf"
TOOLATHLON_BUILD_COMMIT = "69d04f210097870a984d26024016a705a278c1d2"
TOOLATHLON_IMAGE_PATH = Path(
    "/groups/gai51740/aci18912it/rasp/images/openreward-toolathlon-18e62c0d-pbs-53541066e97b.sif"
)
TOOLATHLON_IMAGE_SHA256 = "11d158f7c1a76486f1338f8cf90ce8a0280962792c0e5c95559f19cc391cda14"
TOOLATHLON_RECEIPT_PATH = Path(
    "/groups/gai51740/aci18912it/rasp/images/build-receipts/openreward-toolathlon-18e62c0d-pbs-53541066e97b.sif.json"
)
TOOLATHLON_RECEIPT_SHA256 = "37286eeb005f83e3b9133ceb8204b12d93220ac02cc033118e98c5f314908598"
TOOLATHLON_BASE_PATH = Path("/groups/gai51740/aci18912it/rasp/images/openreward-toolathlon-18e62c0d-c1e60b36.sif")
TOOLATHLON_BASE_SHA256 = "e730e11d1abc63f5cac9eb1788e36da79aa9c7027ed89d91ee42a5b1e311e267"
TOOLATHLON_BASE_OCI = "sha256:c1e60b3673c848b758bda1b32ec39898feb5d6ff4c7e55ead8661adb04b327bd"
TOOLATHLON_PATCHED_SERVER_SHA256 = "26f58c49d42975616c5bef19b654cc4483e665b2107533000861cecbcd74c8a5"
PATCH_ASSETS = (
    "toolathlon-pbs-portable.def",
    "patch_toolathlon_pbs_portable.py",
    "repair_toolathlon_snowflake_catalog.py",
    "launch_toolathlon_python_mcp.py",
    "probe_toolathlon_python_mcp_launchers.py",
    "verify_toolathlon_pbs_portability.sh",
)
RECEIPT_KEYS = {
    "authorization_scope",
    "base_image",
    "build_tooling",
    "completed_at_epoch",
    "derived_image",
    "embedded",
    "gate",
    "labels",
    "patch_assets",
    "pbs_accounting_attested",
    "portability_patch_sha256",
    "schema_version",
    "status",
    "toolathlon_source_revision",
}
EMBEDDED_RECEIPT_KEYS = {
    "assets",
    "base_server_sha256",
    "base_tool_schemas_sha256",
    "manifest_path",
    "manifest_sha256",
    "patched_server_sha256",
    "playwright",
    "python_mcp",
    "repaired_tool_schemas_sha256",
    "verify",
}
COMPONENTS = {
    "platoon_upstream": UPSTREAM_COMMIT,
    "areal": "d99124ec15102ca2fcd4960cc8beaef3950c2672",
    "openhands": "b1acf52ee362884f092b9eaf5e701a1bb5d1b335",
    "swe_rebench": "538b5c5c9c529d3c78d5ef7fd3df7e54e7902ad7",
    "toolathlon": "18e62c0d0419da88209af911528f2f8a0b44a5ed",
}
EXTERNAL_PATHS = {
    "areal": "external/AReaL",
    "openhands": "external/openhands-agent-sdk",
    "swe_rebench": "external/swe-rebench-v2-openrewardenv",
    "toolathlon": "external/toolathlon-gym",
}
TRAINING_SEMANTICS = {
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
HOST_ACCEPTANCE_ASSETS = [
    "pbs-scripts/openreward-immutable-runtime-gate2-v9.pbs",
    "pbs-scripts/publish_openreward_v9_gate2_result.py",
    "pbs-scripts/verify_openreward_v9_gate2_result.py",
    "pbs-scripts/finalize_openreward_v9_acceptance.py",
    "pbs-scripts/openreward-toolathlon-all-mcp-health-v9.pbs",
    "pbs-scripts/toolathlon_all_mcp_health.py",
    "pbs-scripts/publish_toolathlon_all_mcp_health_result.py",
    "pbs-scripts/verify_toolathlon_all_mcp_health_result.py",
]
PLAYWRIGHT_IDENTITY = {
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


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_owned_regular(path: Path, *, mode: int) -> os.stat_result:
    if not path.is_absolute() or path.resolve() != path:
        raise RuntimeError(f"release artifact is not direct and canonical: {path}")
    value = path.lstat()
    if (
        not stat.S_ISREG(value.st_mode)
        or value.st_uid != os.getuid()
        or stat.S_IMODE(value.st_mode) != mode
        or value.st_nlink != 1
    ):
        raise RuntimeError(f"release artifact identity/mode mismatch: {path}")
    return value


def _require_ancestor(root: Path, older: str, newer: str) -> None:
    completed = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", older, newer],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"release ancestry check failed: {older} is not an ancestor of {newer}")


def _require_python_mcp_identity(value: object, assets: dict[str, object]) -> None:
    if not isinstance(value, dict):
        raise RuntimeError("v9 Toolathlon Python MCP provenance is missing")
    expected = {
        "audited_python_server_count": 13,
        "rewritten_uv_config_count": 12,
        "policy": "baked_venv_no_uv",
        "base_config_set_sha256": "ed8f303ce1bebf763f45bc838e58e68477bb4cded54bcb455471782705dc9511",
        "patched_config_set_sha256": "0be2dc5984be8280f5ba5e6d781dad56e765a62f72fb9ebd791f51dcc80b1d60",
        "launcher_mapping_sha256": "23497715f4e3388420e1842558e4017cad3aec62fdf8662e294d4480b19412cc",
        "wrapper_path": "/usr/local/libexec/openreward/launch_toolathlon_python_mcp.py",
        "wrapper_sha256": assets.get("launch_toolathlon_python_mcp.py"),
        "probe_path": "/usr/local/libexec/openreward/probe_toolathlon_python_mcp_launchers.py",
        "probe_sha256": assets.get("probe_toolathlon_python_mcp_launchers.py"),
        "rootless_probe_policy": (
            "cleanenv_no_home_exact_launch_initialize_live_name_set_persistence_clean_teardown_no_mutation"
        ),
        "unprivileged_exact_launch_initialize_live_name_set_persistence_clean_teardown_no_mutation": "passed",
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise RuntimeError("v9 Toolathlon Python MCP launch provenance mismatch")
    snowflake = value.get("snowflake_mcp_runtime")
    project = snowflake.get("project_metadata") if isinstance(snowflake, dict) else None
    if (
        not isinstance(snowflake, dict)
        or snowflake.get("policy") != "exact_in_place_record_repaired_ten_tree_replacement"
        or snowflake.get("installed_mcp_version") != "1.26.0"
        or snowflake.get("donor_closure_sha256") != "e7b1bf8895eec6e95d738b68a53cd4f0eaad9672284c9cdbe5ec3613399c2eec"
        or snowflake.get("installed_closure_sha256")
        != "b618e4359067acb69893820bae6a970c4d8a3714d77550c972346f6120497bc5"
        or not isinstance(project, dict)
        or project.get("patched_pyproject_sha256") != "8d8ed2e26ffe2621403a640fff0d598a35a78dbb06da2867e0bcf6ae80df2743"
        or project.get("patched_lock_sha256") != "2945a5f989fd610b8a31017cd29c41cd3a662e5267f1f17d4871afb7f1ace81c"
        or project.get("offline_uv_lock_check") != "passed"
    ):
        raise RuntimeError("v9 Toolathlon Snowflake MCP runtime provenance mismatch")


def _require_final_image_identity(root: Path, manifest: dict[str, object]) -> None:
    image = manifest.get("toolathlon_image")
    derivation = manifest.get("toolathlon_image_derivation")
    if not isinstance(image, dict) or not isinstance(derivation, dict):
        raise RuntimeError("release lacks the derived Toolathlon image contract")
    path = image.get("path")
    digest = image.get("sha256")
    if (
        not isinstance(path, str)
        or not path.startswith("/")
        or "PENDING" in path
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or derivation.get("status") != "ready"
        or any("PENDING" in str(value) for value in derivation.values())
    ):
        raise RuntimeError("v9 Toolathlon image identity is not final and cannot launch")
    if image != {
        "path": str(TOOLATHLON_IMAGE_PATH),
        "sha256": TOOLATHLON_IMAGE_SHA256,
        "oci_index_digest": "sha256:494af57e8a0f158d668982bf985b5937e73ba58eb468715f884de03550bd677c",
        "oci_amd64_digest": TOOLATHLON_BASE_OCI,
    }:
        raise RuntimeError("v9 Toolathlon image manifest identity mismatch")
    if derivation.get("playwright") != PLAYWRIGHT_IDENTITY:
        raise RuntimeError("v9 Toolathlon Playwright provenance mismatch")
    if derivation.get("build_commit") != TOOLATHLON_BUILD_COMMIT:
        raise RuntimeError("v9 Toolathlon build commit provenance mismatch")

    receipt_path = derivation.get("build_receipt_path")
    receipt_sha256 = derivation.get("build_receipt_sha256")
    provenance = derivation.get("build_receipt_provenance")
    if (
        receipt_path != str(TOOLATHLON_RECEIPT_PATH)
        or receipt_sha256 != TOOLATHLON_RECEIPT_SHA256
        or not isinstance(provenance, dict)
    ):
        raise RuntimeError("v9 Toolathlon build receipt reference mismatch")

    image_stat = _require_owned_regular(TOOLATHLON_IMAGE_PATH, mode=0o550)
    _require_owned_regular(TOOLATHLON_RECEIPT_PATH, mode=0o400)
    if _sha256(TOOLATHLON_IMAGE_PATH) != TOOLATHLON_IMAGE_SHA256:
        raise RuntimeError("v9 Toolathlon SIF stream digest mismatch")
    if _sha256(TOOLATHLON_RECEIPT_PATH) != TOOLATHLON_RECEIPT_SHA256:
        raise RuntimeError("v9 Toolathlon build receipt stream digest mismatch")
    try:
        receipt = json.loads(TOOLATHLON_RECEIPT_PATH.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("v9 Toolathlon build receipt is unreadable") from exc
    if not isinstance(receipt, dict) or set(receipt) != RECEIPT_KEYS:
        raise RuntimeError("v9 Toolathlon build receipt schema mismatch")
    embedded = receipt.get("embedded")
    if not isinstance(embedded, dict) or set(embedded) != EMBEDDED_RECEIPT_KEYS:
        raise RuntimeError("v9 Toolathlon embedded receipt schema mismatch")

    receipt_provenance = {
        "schema_version": receipt.get("schema_version"),
        "gate": receipt.get("gate"),
        "status": receipt.get("status"),
        "authorization_scope": receipt.get("authorization_scope"),
        "pbs_accounting_attested": receipt.get("pbs_accounting_attested"),
        "completed_at_epoch": receipt.get("completed_at_epoch"),
        "derived_image": receipt.get("derived_image"),
        "patch_assets": receipt.get("patch_assets"),
        "build_tooling": receipt.get("build_tooling"),
        "embedded": {
            "manifest_path": embedded.get("manifest_path"),
            "manifest_sha256": embedded.get("manifest_sha256"),
            "assets": embedded.get("assets"),
            "verify": embedded.get("verify"),
        },
        "labels": receipt.get("labels"),
    }
    if receipt_provenance != provenance:
        raise RuntimeError("v9 Toolathlon build receipt provenance mismatch")

    expected_derived = {
        "path": str(TOOLATHLON_IMAGE_PATH),
        "sha256": TOOLATHLON_IMAGE_SHA256,
        "size": image_stat.st_size,
        "device": image_stat.st_dev,
        "inode": image_stat.st_ino,
        "mode": 0o550,
        "uid": os.getuid(),
    }
    if receipt.get("derived_image") != expected_derived:
        raise RuntimeError("v9 Toolathlon build receipt SIF identity mismatch")
    if receipt.get("base_image") != {
        "path": str(TOOLATHLON_BASE_PATH),
        "sha256": TOOLATHLON_BASE_SHA256,
        "oci_digest": TOOLATHLON_BASE_OCI,
    }:
        raise RuntimeError("v9 Toolathlon build receipt base identity mismatch")
    if (
        receipt.get("toolathlon_source_revision") != COMPONENTS["toolathlon"]
        or receipt.get("portability_patch_sha256") != derivation.get("pbs_portability_patch_sha256")
        or embedded.get("base_server_sha256") != derivation.get("base_server_sha256")
        or embedded.get("patched_server_sha256") != TOOLATHLON_PATCHED_SERVER_SHA256
        or derivation.get("patched_server_sha256") != TOOLATHLON_PATCHED_SERVER_SHA256
        or embedded.get("base_tool_schemas_sha256") != derivation.get("base_tool_schemas_sha256")
        or embedded.get("repaired_tool_schemas_sha256") != derivation.get("repaired_tool_schemas_sha256")
        or embedded.get("playwright") != PLAYWRIGHT_IDENTITY
        or embedded.get("python_mcp") != derivation.get("python_mcp")
    ):
        raise RuntimeError("v9 Toolathlon build receipt derivation mismatch")

    assets = receipt.get("patch_assets")
    if not isinstance(assets, dict) or set(assets) != set(PATCH_ASSETS):
        raise RuntimeError("v9 Toolathlon patch asset set mismatch")
    patch_input = "".join(f"{assets[name]}  {name}\n" for name in PATCH_ASSETS).encode()
    if hashlib.sha256(patch_input).hexdigest() != derivation.get("pbs_portability_patch_sha256"):
        raise RuntimeError("v9 Toolathlon portability identity does not match its assets")
    embedded_assets = embedded.get("assets")
    if not isinstance(embedded_assets, dict) or embedded_assets != {
        name: assets[name] for name in PATCH_ASSETS if name != "toolathlon-pbs-portable.def"
    }:
        raise RuntimeError("v9 Toolathlon embedded asset identities mismatch")
    _require_python_mcp_identity(embedded.get("python_mcp"), assets)

    tooling = receipt.get("build_tooling")
    expected_tooling_paths = {
        "builder": root / "containers/build_toolathlon_pbs_portable_sif.sh",
        "publisher": root / "containers/publish_toolathlon_pbs_build_receipt.py",
        "image_publisher": root / "containers/publish_toolathlon_pbs_image.py",
    }
    if not isinstance(tooling, dict):
        raise RuntimeError("v9 Toolathlon build tooling contract is missing")
    for name, tooling_path in expected_tooling_paths.items():
        reference = tooling.get(name)
        if (
            not isinstance(reference, dict)
            or reference.get("name") != tooling_path.name
            or reference.get("sha256") != _sha256(tooling_path)
        ):
            raise RuntimeError(f"v9 Toolathlon build tooling identity mismatch: {name}")
    if tooling.get("singularity") != {
        "path": "/usr/bin/singularity",
        "sha256": "085a757f172dcabda66b44d086913314d6aa574055a8fea90523be82b770cec6",
        "version": "4.4.1-1.el9",
    }:
        raise RuntimeError("v9 Toolathlon build Singularity identity mismatch")

    labels = receipt.get("labels")
    required_labels = {
        "org.opencontainers.image.source": "https://github.com/ApGa/toolathlon-gym",
        "org.opencontainers.image.revision": COMPONENTS["toolathlon"],
        "org.openreward.toolathlon.base-sif-sha256": TOOLATHLON_BASE_SHA256,
        "org.openreward.toolathlon.base-oci-digest": TOOLATHLON_BASE_OCI,
        "org.openreward.toolathlon.pbs-portability": derivation.get("pbs_portability_patch_sha256"),
        "org.openreward.toolathlon.playwright-browsers-sha256": PLAYWRIGHT_IDENTITY["portable_browser_tree_sha256"],
    }
    if not isinstance(labels, dict) or any(labels.get(key) != value for key, value in required_labels.items()):
        raise RuntimeError("v9 Toolathlon derived SIF labels mismatch")


def verify(root: Path) -> None:
    if not root.is_absolute() or root.resolve() != root or root != RELEASE_ROOT:
        raise RuntimeError(f"release root is not the exact canonical path: {root}")
    manifest_path = root / MANIFEST_RELATIVE
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError("v9 release manifest is missing or indirect")
    if _sha256(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("v9 release manifest content digest mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise RuntimeError("release manifest schema is not 1")
    if manifest.get("release_root") != str(RELEASE_ROOT):
        raise RuntimeError("release manifest root mismatch")
    if manifest.get("release_ref") != RELEASE_REF:
        raise RuntimeError("release manifest ref mismatch")
    if manifest.get("components") != COMPONENTS:
        raise RuntimeError("release component revisions mismatch")
    if manifest.get("training_semantics") != TRAINING_SEMANTICS:
        raise RuntimeError("release training semantics mismatch")
    if manifest.get("host_acceptance_assets") != HOST_ACCEPTANCE_ASSETS:
        raise RuntimeError("release host acceptance asset set mismatch")
    for relative in HOST_ACCEPTANCE_ASSETS:
        asset = root / relative
        if not asset.is_file() or asset.is_symlink() or not asset.stat().st_mode & 0o111:
            raise RuntimeError(f"release host acceptance asset is unavailable: {relative}")
    _require_final_image_identity(root, manifest)

    head = _git(root, "rev-parse", "--verify", "HEAD")
    release_commit = _git(root, "rev-parse", "--verify", f"{RELEASE_REF}^{{commit}}")
    if head != release_commit:
        raise RuntimeError("release HEAD does not match its pinned tag")
    _require_ancestor(root, UPSTREAM_COMMIT, head)
    _require_ancestor(root, TOOLATHLON_BUILD_COMMIT, head)
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("release checkout has tracked or ordinary untracked changes")
    tracked = _git(root, "ls-files").splitlines()
    if any(
        path.endswith(".orig")
        or path.endswith("cleanup_openreward_2132698_generation.py")
        or path.endswith("test_offline_cleanup_2132698.py")
        for path in tracked
    ):
        raise RuntimeError("release tracks retired migration debris")

    for name, relative in EXTERNAL_PATHS.items():
        external = root / relative
        if external.resolve() != external or not (external / ".git").exists():
            raise RuntimeError(f"materialized external checkout is invalid: {name}")
        if _git(external, "rev-parse", "--verify", "HEAD") != COMPONENTS[name]:
            raise RuntimeError(f"materialized external revision mismatch: {name}")
        if _git(external, "status", "--porcelain=v1", "--untracked-files=all"):
            raise RuntimeError(f"materialized external checkout is dirty: {name}")

    # The SWE dependency closure did not change between v8r2 and v8r9.  The
    # existing runtime verifier reads the retained v8 manifest, so require both
    # manifests to carry byte-for-byte equal SWE contracts before delegating.
    v8_manifest = json.loads((root / "pbs-scripts/openreward-v8-release.json").read_text())
    if v8_manifest.get("swe_runtime") != manifest.get("swe_runtime"):
        raise RuntimeError("v9 and retained v8 SWE runtime contracts differ")
    verify_swe_runtime(root)

    print(f"openreward_v9_release=PASS release_commit={head}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    args = parser.parse_args()
    verify(args.release_root)


if __name__ == "__main__":
    main()
