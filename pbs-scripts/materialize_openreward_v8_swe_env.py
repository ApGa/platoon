#!/usr/bin/env python3
"""Build the pinned, host-side SWE service venv for an exact v8 checkout."""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
from pathlib import Path

from verify_openreward_v8_swe_env import (
    READY_NAME,
    SWE_RELATIVE_ROOT,
    _git,
    _load_runtime_contract,
    _runtime_probe,
    _sha256,
    expected_ready_receipt,
    verify,
)


def _run(*args: str, timeout: int = 900) -> None:
    subprocess.run(
        list(args),
        check=True,
        timeout=timeout,
        env={
            "HOME": os.environ["HOME"],
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", "/groups/gai51740/aci18912it/rasp/cache/uv"),
        },
    )


def materialize(release_root: Path) -> None:
    if not release_root.is_absolute() or release_root.resolve() != release_root:
        raise RuntimeError(f"release root is not an exact absolute path: {release_root}")
    runtime = _load_runtime_contract(release_root)
    swe_root = release_root / SWE_RELATIVE_ROOT
    environment = swe_root / ".venv-openreward"
    requirements = swe_root / "requirements.txt"
    lock = release_root / runtime["lock_path"]
    uv = Path(runtime["uv_executable"])
    base_python = Path(runtime["python_real_executable"])

    if environment.exists() or environment.is_symlink():
        raise RuntimeError(f"refusing an existing SWE runtime: {environment}")
    if _git(swe_root, "rev-parse", "--verify", "HEAD") != runtime["swe_revision"]:
        raise RuntimeError("SWE checkout revision mismatch before environment build")
    if _git(swe_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("SWE checkout is dirty before environment build")
    if _sha256(requirements) != runtime["requirements_sha256"]:
        raise RuntimeError("SWE requirements digest mismatch before environment build")
    if _sha256(lock) != runtime["lock_sha256"]:
        raise RuntimeError("SWE lock digest mismatch before environment build")
    if not uv.is_file() or not os.access(uv, os.X_OK):
        raise RuntimeError(f"pinned uv executable is unavailable: {uv}")
    if not base_python.is_file() or not os.access(base_python, os.X_OK):
        raise RuntimeError(f"pinned Python executable is unavailable: {base_python}")
    python_base = base_python.parents[1]
    toolchain_files = {
        base_python: runtime["python_executable_sha256"],
        python_base / "lib/libpython3.11.so.1.0": runtime["python_library_sha256"],
        python_base / "BUILD": runtime["python_build_sha256"],
        uv: runtime["uv_executable_sha256"],
    }
    if any(not path.is_file() or _sha256(path) != digest for path, digest in toolchain_files.items()):
        raise RuntimeError("SWE runtime build toolchain digest mismatch")
    uv_version = subprocess.run(
        [str(uv), "--version"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
        timeout=30,
    ).stdout.strip()
    if uv_version != f"uv {runtime['uv_version']} (x86_64-unknown-linux-gnu)":
        raise RuntimeError("uv identity differs from the SWE runtime contract")
    python_version = subprocess.run(
        [str(base_python), "-I", "-c", "import sys; print(sys.version.split()[0])"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
        timeout=30,
    ).stdout.strip()
    if python_version != runtime["python_version"]:
        raise RuntimeError("base Python version differs from the SWE runtime contract")

    # Build at its final absolute location so every console-script shebang is
    # correct.  A failed build deliberately leaves an incomplete directory;
    # the next attempt refuses it instead of deleting or silently reusing it.
    _run(
        str(uv),
        "venv",
        "--python",
        str(base_python),
        "--no-python-downloads",
        "--no-project",
        str(environment),
    )
    _run(
        str(uv),
        "pip",
        "sync",
        "--python",
        str(environment / "bin/python"),
        "--require-hashes",
        "--strict",
        "--link-mode",
        "copy",
        str(lock),
    )
    actual = _runtime_probe(environment / "bin/python")
    expected = {
        "executable": str(environment / "bin/python"),
        "real_executable": runtime["python_real_executable"],
        "prefix": str(environment),
        "python_version": runtime["python_version"],
        "package_count": runtime["package_count"],
        "freeze_sha256": runtime["freeze_sha256"],
    }
    actual_identity = {name: actual[name] for name in expected}
    packages = actual.get("packages")
    if (
        actual_identity != expected
        or not isinstance(packages, dict)
        or any(packages.get(name) != version for name, version in runtime["package_pins"].items())
    ):
        raise RuntimeError("new SWE environment does not match its pinned package contract")

    ready = environment / READY_NAME
    descriptor = os.open(ready, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(expected_ready_receipt(release_root), stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(ready, 0o400)

    # The environment is a release artifact, not a cache. Remove all write
    # bits after publication; runtime caches live outside this prefix.
    for raw_root, directories, files in os.walk(environment):
        root = Path(raw_root)
        for name in files:
            path = root / name
            if path.is_symlink():
                continue
            mode = stat.S_IMODE(path.stat().st_mode)
            path.chmod(mode & ~0o222)
        for name in directories:
            path = root / name
            if path.is_symlink():
                continue
            mode = stat.S_IMODE(path.stat().st_mode)
            path.chmod(mode & ~0o222)
    environment.chmod(stat.S_IMODE(environment.stat().st_mode) & ~0o222)
    verify(release_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.release_root)


if __name__ == "__main__":
    main()
