#!/usr/bin/env python3
"""Fail-closed verification for the immutable ABCI v8 release checkout."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from verify_openreward_v8_swe_env import verify as verify_swe_runtime

RELEASE_ROOT = Path("/home/aci18912it/rasp/releases/platoon-v8-376d1b9-18e62c0")
RELEASE_REF = "refs/tags/abci-v8-376d1b9-18e62c0"
UPSTREAM_COMMIT = "376d1b9885ea40464766fb3fea74815c6a698fbf"
COMPONENTS = {
    "platoon_upstream": UPSTREAM_COMMIT,
    "areal": "d99124ec15102ca2fcd4960cc8beaef3950c2672",
    "openhands": "b1acf52ee362884f092b9eaf5e701a1bb5d1b335",
    "swe_rebench": "538b5c5c9c529d3c78d5ef7fd3df7e54e7902ad7",
    "toolathlon": "18e62c0d0419da88209af911528f2f8a0b44a5ed",
}
UPSTREAM_REFRESH = {
    "fetched_at_utc": "2026-08-13T07:54:55Z",
    "platoon": {
        "remote": "https://github.com/ApGa/platoon.git",
        "ref": "refs/heads/apga/update-areal-with-registry",
        "sha": COMPONENTS["platoon_upstream"],
        "release_ahead": 0,
        "release_behind": 0,
    },
    "openhands": {
        "remote": "https://github.com/apga/openhands-agent-sdk.git",
        "ref": "refs/heads/apga/ptc",
        "sha": COMPONENTS["openhands"],
        "release_ahead": 0,
        "release_behind": 0,
    },
    "toolathlon": {
        "remote": "https://github.com/ApGa/toolathlon-gym.git",
        "ref": "refs/heads/apga/session-db-shim",
        "sha": COMPONENTS["toolathlon"],
        "release_ahead": 0,
        "release_behind": 0,
    },
    "swe_rebench": {
        "remote": "https://github.com/ApGa/SWE-rebench-V2-OpenRewardEnv.git",
        "ref": "refs/heads/main",
        "sha": "25b14c06b9236c075a4ede25bff6979e5783bb09",
        "release_ahead": 1,
        "release_behind": 0,
    },
}
EXTERNAL_PATHS = {
    "areal": "external/AReaL",
    "openhands": "external/openhands-agent-sdk",
    "swe_rebench": "external/swe-rebench-v2-openrewardenv",
    "toolathlon": "external/toolathlon-gym",
}
IMAGE = {
    "path": "/groups/gai51740/aci18912it/rasp/images/openreward-toolathlon-18e62c0d-c1e60b36.sif",
    "sha256": "e730e11d1abc63f5cac9eb1788e36da79aa9c7027ed89d91ee42a5b1e311e267",
    "oci_index_digest": "sha256:494af57e8a0f158d668982bf985b5937e73ba58eb468715f884de03550bd677c",
    "oci_amd64_digest": "sha256:c1e60b3673c848b758bda1b32ec39898feb5d6ff4c7e55ead8661adb04b327bd",
}
SWE_RUNTIME = {
    "swe_revision": COMPONENTS["swe_rebench"],
    "requirements_sha256": "9d953d984c4d9095054d666492b51674797572b44f7bb02245a98611e38b6a7a",
    "lock_path": "containers/swe-rebench-538b5c5-py311.lock.txt",
    "lock_sha256": "dd33fc4a252140a3976c2b989c7769405781b863a21fdbb7a77ad5adc52de836",
    "python_real_executable": "/home/aci18912it/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11",
    "python_executable_sha256": "c0b74b270ea823c760917e64fbde1945e67ba739d97fa716a48a5b69e6505d62",
    "python_library_sha256": "46da5d861633bd1627474c0d870592d8ec7817bef573ddc89b2128ed21721857",
    "python_build_sha256": "aae4c341cd4a142780552b76d2f7e41c6e26b9317a414ac7c6252fa7848146e2",
    "python_version": "3.11.15",
    "uv_executable": "/home/aci18912it/.local/bin/uv",
    "uv_executable_sha256": "92face6b1f0462ad911857957bd168cd4ae45515e2a2cb3fcc3ecbda3d4d82b1",
    "uv_version": "0.12.1",
    "package_count": 62,
    "freeze_sha256": "96c72a454e95bb0da0df14861d23909ff00d9345d301c9277300948f67cf1c4c",
    "package_pins": {
        "openreward": "0.1.137",
        "pyarrow": "25.0.1",
        "pydantic": "2.13.4",
        "pyright": "1.1.411",
    },
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


def verify(root: Path) -> None:
    if not root.is_absolute() or root.resolve() != root or root != RELEASE_ROOT:
        raise RuntimeError(f"release root is not the exact canonical path: {root}")
    manifest_path = root / "pbs-scripts/openreward-v8-release.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise RuntimeError("release manifest schema is not 1")
    if manifest.get("release_root") != str(RELEASE_ROOT):
        raise RuntimeError("release manifest root mismatch")
    if manifest.get("release_ref") != RELEASE_REF:
        raise RuntimeError("release manifest ref mismatch")
    if manifest.get("components") != COMPONENTS:
        raise RuntimeError("release component revisions mismatch")
    if manifest.get("upstream_refresh") != UPSTREAM_REFRESH:
        raise RuntimeError("release upstream refresh evidence mismatch")
    if manifest.get("toolathlon_image") != IMAGE:
        raise RuntimeError("release Toolathlon image identity mismatch")
    if manifest.get("swe_runtime") != SWE_RUNTIME:
        raise RuntimeError("release SWE runtime identity mismatch")

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

    verify_swe_runtime(root)

    print(f"openreward_v8_release=PASS release_commit={head}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    args = parser.parse_args()
    verify(args.release_root)


if __name__ == "__main__":
    main()
