#!/usr/bin/env python3
"""Fail-closed verification for the immutable ABCI v8r3/v9 checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from verify_openreward_v8_swe_env import verify as verify_swe_runtime

RELEASE_ROOT = Path("/home/aci18912it/rasp/releases/platoon-v8r3-376d1b9-18e62c0")
RELEASE_REF = "refs/tags/abci-v8r3-376d1b9-18e62c0"
MANIFEST_RELATIVE = Path("pbs-scripts/openreward-v9-release.json")
MANIFEST_SHA256 = "d5361eabcf9a8cb0091d92b79f8c76703a39bcec45eb1bfbc6dca70bed9dea94"
UPSTREAM_COMMIT = "376d1b9885ea40464766fb3fea74815c6a698fbf"
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
    "fresh_16node_model_only_lineage": True,
    "nonrecursive_16node_ptc": False,
    "recursive_32node_ptc": True,
    "recursive_32node_ptc_mode": "orchestration_only",
    "recursive_32node_ptc_max_tool_calls_per_execution": 1024,
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


def _require_final_image_identity(manifest: dict[str, object]) -> None:
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
        raise RuntimeError("v9 Toolathlon image identity is pending and cannot launch")
    if derivation.get("playwright") != PLAYWRIGHT_IDENTITY:
        raise RuntimeError("v9 Toolathlon Playwright provenance mismatch")


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
    _require_final_image_identity(manifest)

    head = _git(root, "rev-parse", "--verify", "HEAD")
    release_commit = _git(root, "rev-parse", "--verify", f"{RELEASE_REF}^{{commit}}")
    if head != release_commit:
        raise RuntimeError("release HEAD does not match its pinned tag")
    _require_ancestor(root, UPSTREAM_COMMIT, head)
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

    # The SWE dependency closure did not change between v8r2 and v8r3.  The
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
