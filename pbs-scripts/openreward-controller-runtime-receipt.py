#!/usr/bin/env python3
"""Publish/read the exact immutable controller runtime after clean teardown."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

HEX64 = re.compile(r"[0-9a-f]{64}")
IDENTITY = re.compile(r"[A-Za-z0-9_.-]+")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_owned(path: Path, *, nonwritable: bool = True) -> os.stat_result:
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or value.st_uid != os.getuid():
        raise RuntimeError(f"runtime file is not direct and user-owned: {path}")
    if nonwritable and stat.S_IMODE(value.st_mode) & 0o222:
        raise RuntimeError(f"runtime file is writable: {path}")
    return value


def _directory_owned(path: Path, *, nonwritable: bool) -> os.stat_result:
    value = path.lstat()
    if not stat.S_ISDIR(value.st_mode) or value.st_uid != os.getuid():
        raise RuntimeError(f"runtime directory is not direct and user-owned: {path}")
    if nonwritable and stat.S_IMODE(value.st_mode) & 0o222:
        raise RuntimeError(f"runtime directory is writable: {path}")
    return value


def _manifest_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            raise RuntimeError("controller manifest contains a malformed line")
        name, value = line.split("=", 1)
        if not name or name in values:
            raise RuntimeError("controller manifest contains a duplicate/empty key")
        values[name] = value
    return values


def _runtime_identity(cache_root: Path, key: str, source_sha: str) -> dict[str, str]:
    if (
        not cache_root.is_absolute()
        or cache_root.resolve() != cache_root
        or HEX64.fullmatch(key) is None
        or HEX64.fullmatch(source_sha) is None
    ):
        raise RuntimeError("controller cache/source/key identity is malformed")
    _directory_owned(cache_root, nonwritable=False)
    envs = cache_root / "envs"
    manifests = cache_root / "manifests"
    _directory_owned(envs, nonwritable=False)
    _directory_owned(manifests, nonwritable=False)
    venv = envs / key
    if venv.parent != envs or venv.name != key:
        raise RuntimeError("controller venv is outside its exact keyed cache path")
    _directory_owned(venv, nonwritable=True)
    _directory_owned(venv / "bin", nonwritable=True)
    for marker in (".platoon-controller-ready", ".build-complete"):
        _plain_owned(venv / marker)
    manifest = venv / ".platoon-controller-manifest"
    canonical_manifest = manifests / f"{key}.txt"
    _plain_owned(manifest)
    _plain_owned(canonical_manifest)
    if manifest.read_bytes() != canonical_manifest.read_bytes() or _sha256(manifest) != key:
        raise RuntimeError("controller manifest/cache key identity mismatch")
    values = _manifest_values(manifest)
    if values.get("source_sha256") != source_sha:
        raise RuntimeError("controller manifest source identity mismatch")
    python_root = Path(values.get("python_root", ""))
    if not python_root.is_absolute() or python_root.resolve() != python_root:
        raise RuntimeError("controller Python root is not direct and absolute")
    _directory_owned(python_root, nonwritable=False)
    _directory_owned(python_root / "bin", nonwritable=False)
    expected_python = python_root / "bin/python3.12"
    python_stat = _plain_owned(expected_python, nonwritable=False)
    if not python_stat.st_mode & 0o111:
        raise RuntimeError("controller Python target is not executable")
    python = venv / "bin/python"
    if (
        not python.is_symlink()
        or python.lstat().st_uid != os.getuid()
        or Path(os.path.realpath(python)) != expected_python
    ):
        raise RuntimeError("controller Python target mismatch")
    if _sha256(expected_python) != values.get("python_executable_sha256"):
        raise RuntimeError("controller Python executable digest mismatch")
    completed = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            (
                "import json,pathlib,sys; import platoon.training_success_receipt as m; "
                "print(json.dumps({'prefix':str(pathlib.Path(sys.prefix).resolve()),"
                "'module':str(pathlib.Path(m.__file__).resolve())},sort_keys=True))"
            ),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    probe = json.loads(completed.stdout)
    module = Path(probe["module"])
    if probe["prefix"] != str(venv) or not module.is_relative_to(venv) or module.name != "training_success_receipt.py":
        raise RuntimeError("controller Python prefix/module escaped the keyed venv")
    # uv intentionally leaves installed package files owner-writable because
    # they may be hardlinks into its cache.  Bind this exact module by digest
    # across publish/read instead of claiming the entire environment is
    # immutable merely because its directories are non-writable.
    _plain_owned(module, nonwritable=False)
    return {
        "controller_cache_root": str(cache_root),
        "controller_venv": str(venv),
        "controller_python": str(python),
        "controller_manifest_sha256": key,
        "training_success_receipt_module": str(module),
        "training_success_receipt_sha256": _sha256(module),
    }


def _safe_receipt(path: Path, exists: bool) -> None:
    if not path.is_absolute() or path.parent.resolve() != path.parent:
        raise RuntimeError("controller runtime receipt path is not exact")
    parent = path.parent.lstat()
    if not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) & 0o077:
        raise RuntimeError("controller runtime receipt parent is unsafe")
    actual = path.exists() or path.is_symlink()
    if actual != exists:
        raise RuntimeError("controller runtime receipt freshness mismatch")
    if actual:
        value = path.lstat()
        if not stat.S_ISREG(value.st_mode) or value.st_uid != os.getuid() or stat.S_IMODE(value.st_mode) != 0o600:
            raise RuntimeError("controller runtime receipt is unsafe")


def _base(args: argparse.Namespace) -> dict[str, Any]:
    if any(IDENTITY.fullmatch(value) is None for value in (args.pbs_job_id, args.experiment, args.trial)):
        raise RuntimeError("controller runtime job/experiment/trial identity is malformed")
    return {
        "schema_version": 1,
        "status": "passed",
        "pbs_job_id": args.pbs_job_id,
        "experiment": args.experiment,
        "trial": args.trial,
        "source_sha256": args.source_sha256,
        "controller_environment_key": args.controller_environment_key,
        "role_cleanup_passed": True,
        "service_cleanup_passed": True,
    }


def _publish(path: Path, value: dict[str, Any]) -> None:
    _safe_receipt(path, False)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RuntimeError("controller runtime receipt appeared during publication") from error
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        temporary.unlink(missing_ok=True)
    _safe_receipt(path, True)
    if json.loads(path.read_text()) != value:
        raise RuntimeError("controller runtime receipt changed during publication")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("publish", "read-controller-python"))
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--pbs-job-id", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--controller-cache-root", type=Path, required=True)
    parser.add_argument("--source-sha256")
    parser.add_argument("--controller-environment-key")
    args = parser.parse_args(argv)
    if args.command == "publish":
        if not args.source_sha256 or not args.controller_environment_key:
            parser.error("publish requires source SHA and controller environment key")
        value = {
            **_base(args),
            **_runtime_identity(
                args.controller_cache_root,
                args.controller_environment_key,
                args.source_sha256,
            ),
        }
        _publish(args.receipt, value)
        print("controller_runtime_receipt=published")
        return 0

    _safe_receipt(args.receipt, True)
    value = json.loads(args.receipt.read_text())
    identity = {
        "schema_version": 1,
        "status": "passed",
        "pbs_job_id": args.pbs_job_id,
        "experiment": args.experiment,
        "trial": args.trial,
        "role_cleanup_passed": True,
        "service_cleanup_passed": True,
    }
    if any(value.get(name) != expected for name, expected in identity.items()):
        raise RuntimeError("controller runtime receipt identity mismatch")
    key = value.get("controller_environment_key")
    source = value.get("source_sha256")
    if value.get("controller_cache_root") != str(args.controller_cache_root):
        raise RuntimeError("controller runtime cache-root mismatch")
    current = _runtime_identity(args.controller_cache_root, str(key), str(source))
    expected = {**identity, "source_sha256": source, "controller_environment_key": key, **current}
    if value != expected:
        raise RuntimeError("controller runtime receipt content/runtime mismatch")
    print(value["controller_python"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
