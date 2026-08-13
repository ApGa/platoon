#!/usr/bin/env python3
"""Verify the host Python environment for the pinned v8 SWE service."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

SWE_RELATIVE_ROOT = Path("external/swe-rebench-v2-openrewardenv")
READY_NAME = ".platoon-swe-runtime.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    ).stdout.strip()


def _load_runtime_contract(release_root: Path) -> dict[str, Any]:
    manifest_path = release_root / "pbs-scripts/openreward-v8-release.json"
    value = json.loads(manifest_path.read_text())
    runtime = value.get("swe_runtime")
    if not isinstance(runtime, dict):
        raise RuntimeError("release manifest lacks the SWE runtime contract")
    return runtime


def _runtime_probe(python: Path) -> dict[str, Any]:
    probe = r"""import hashlib, importlib.metadata, json, os, re, sys
rows = sorted({
    f"{re.sub(r'[-_.]+', '-', dist.metadata['Name']).lower()}=={dist.version}"
    for dist in importlib.metadata.distributions()
    if dist.metadata.get('Name')
})
payload = ("\n".join(rows) + "\n").encode()
packages = dict(row.split("==", 1) for row in rows)
print(json.dumps({
    "executable": sys.executable,
    "real_executable": os.path.realpath(sys.executable),
    "prefix": sys.prefix,
    "python_version": sys.version.split()[0],
    "package_count": len(rows),
    "freeze_sha256": hashlib.sha256(payload).hexdigest(),
    "packages": packages,
}, sort_keys=True))"""
    completed = subprocess.run(
        [str(python), "-I", "-c", probe],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("SWE runtime probe did not return JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError("SWE runtime probe returned a non-object")
    return value


def expected_ready_receipt(release_root: Path) -> dict[str, Any]:
    runtime = _load_runtime_contract(release_root)
    return {
        "schema_version": 1,
        "swe_revision": runtime["swe_revision"],
        "requirements_sha256": runtime["requirements_sha256"],
        "lock_sha256": runtime["lock_sha256"],
        "python_version": runtime["python_version"],
        "python_executable_sha256": runtime["python_executable_sha256"],
        "python_library_sha256": runtime["python_library_sha256"],
        "python_build_sha256": runtime["python_build_sha256"],
        "python_real_executable": runtime["python_real_executable"],
        "uv_version": runtime["uv_version"],
        "uv_executable_sha256": runtime["uv_executable_sha256"],
        "package_count": runtime["package_count"],
        "freeze_sha256": runtime["freeze_sha256"],
        "package_pins": runtime["package_pins"],
    }


def verify(release_root: Path) -> dict[str, Any]:
    if not release_root.is_absolute() or release_root.resolve() != release_root:
        raise RuntimeError(f"SWE release root is not an exact absolute path: {release_root}")

    runtime = _load_runtime_contract(release_root)
    swe_root = release_root / SWE_RELATIVE_ROOT
    if swe_root.resolve() != swe_root or not (swe_root / ".git").exists():
        raise RuntimeError("SWE source checkout is missing or indirect")
    if _git(swe_root, "rev-parse", "--verify", "HEAD") != runtime["swe_revision"]:
        raise RuntimeError("SWE source revision does not match the runtime contract")
    if _git(swe_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("SWE source checkout has non-ignored changes")

    requirements = swe_root / "requirements.txt"
    lock = release_root / runtime["lock_path"]
    if _sha256(requirements) != runtime["requirements_sha256"]:
        raise RuntimeError("SWE source requirements digest mismatch")
    if not lock.is_file() or lock.is_symlink() or _sha256(lock) != runtime["lock_sha256"]:
        raise RuntimeError("SWE release lock digest mismatch")
    python_base = Path(runtime["python_real_executable"]).parents[1]
    toolchain_files = {
        Path(runtime["python_real_executable"]): runtime["python_executable_sha256"],
        python_base / "lib/libpython3.11.so.1.0": runtime["python_library_sha256"],
        python_base / "BUILD": runtime["python_build_sha256"],
        Path(runtime["uv_executable"]): runtime["uv_executable_sha256"],
    }
    if any(not path.is_file() or _sha256(path) != digest for path, digest in toolchain_files.items()):
        raise RuntimeError("SWE runtime build toolchain digest mismatch")

    environment = swe_root / ".venv-openreward"
    python = environment / "bin/python"
    ready = environment / READY_NAME
    environment_stat = environment.lstat()
    if (
        not stat.S_ISDIR(environment_stat.st_mode)
        or environment_stat.st_uid != os.getuid()
        or stat.S_IMODE(environment_stat.st_mode) & 0o222
    ):
        raise RuntimeError("SWE runtime root is not immutable, direct, and user-owned")
    if not python.is_symlink() or os.path.realpath(python) != runtime["python_real_executable"]:
        raise RuntimeError("SWE runtime Python target mismatch")
    if not ready.is_file() or ready.is_symlink():
        raise RuntimeError("SWE runtime ready receipt is absent or indirect")
    ready_stat = ready.stat()
    if ready_stat.st_uid != os.getuid() or stat.S_IMODE(ready_stat.st_mode) & 0o222:
        raise RuntimeError("SWE runtime ready receipt is not immutable and user-owned")
    if json.loads(ready.read_text()) != expected_ready_receipt(release_root):
        raise RuntimeError("SWE runtime ready receipt does not match the release contract")
    for raw_root, directories, files in os.walk(environment):
        for name in [*directories, *files]:
            path = Path(raw_root) / name
            if path.is_symlink():
                continue
            path_stat = path.stat()
            if path_stat.st_uid != os.getuid() or stat.S_IMODE(path_stat.st_mode) & 0o222:
                raise RuntimeError("SWE runtime contains a writable or foreign-owned published entry")

    actual = _runtime_probe(python)
    expected = {
        "executable": str(python),
        "real_executable": runtime["python_real_executable"],
        "prefix": str(environment),
        "python_version": runtime["python_version"],
        "package_count": runtime["package_count"],
        "freeze_sha256": runtime["freeze_sha256"],
    }
    actual_identity = {name: actual[name] for name in expected}
    if actual_identity != expected:
        raise RuntimeError("SWE runtime package or interpreter attestation mismatch")
    packages = actual.get("packages")
    if not isinstance(packages, dict) or any(
        packages.get(name) != version for name, version in runtime["package_pins"].items()
    ):
        raise RuntimeError("SWE runtime key package pins mismatch")
    print(f"openreward_v8_swe_runtime=PASS revision={runtime['swe_revision']} freeze_sha256={runtime['freeze_sha256']}")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    args = parser.parse_args()
    verify(args.release_root)


if __name__ == "__main__":
    main()
