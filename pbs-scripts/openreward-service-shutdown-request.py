#!/usr/bin/env python3
"""Create and verify exact, private node-service shutdown requests."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import stat
from pathlib import Path
from typing import Any


def _expected(job_id: str, host: str) -> dict[str, Any]:
    if not job_id or not host:
        raise RuntimeError("shutdown request identity is invalid")
    return {
        "schema_version": 1,
        "action": "shutdown",
        "reason": "natural_success",
        "pbs_job_id": job_id,
        "host": host,
    }


def _safe_parent(path: Path) -> None:
    if not path.is_absolute() or path.parent.resolve() != path.parent:
        raise RuntimeError("shutdown request path is not direct and absolute")
    value = path.parent.lstat()
    if not stat.S_ISDIR(value.st_mode) or value.st_uid != os.getuid() or stat.S_IMODE(value.st_mode) & 0o077:
        raise RuntimeError("shutdown request parent is not private and user-owned")


def _assert_absent(path: Path) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise RuntimeError(f"shutdown request is stale or indirect: {path}")


def _read_private(path: Path) -> dict[str, Any]:
    _safe_parent(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise RuntimeError("shutdown request is missing or indirect") from error
    try:
        value = os.fstat(descriptor)
        if not stat.S_ISREG(value.st_mode) or value.st_uid != os.getuid() or stat.S_IMODE(value.st_mode) != 0o600:
            raise RuntimeError("shutdown request is not a mode-0600 user-owned regular file")
        with os.fdopen(descriptor) as stream:
            descriptor = -1
            try:
                payload = json.load(stream)
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError("shutdown request is unreadable or malformed") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(payload, dict):
        raise RuntimeError("shutdown request is not an object")
    return payload


def publish(path: Path, *, job_id: str, host: str) -> None:
    if os.environ.get("PBS_JOBID") != job_id:
        raise RuntimeError("shutdown request PBS job identity mismatch")
    payload = _expected(job_id, host)
    _safe_parent(path)
    _assert_absent(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            descriptor = -1
            json.dump(payload, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RuntimeError("shutdown request appeared during create-only publication") from error
        parent_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    if _read_private(path) != payload:
        raise RuntimeError("published shutdown request failed exact verification")


def verify(path: Path, *, job_id: str, host: str) -> dict[str, Any]:
    if os.environ.get("PBS_JOBID") != job_id:
        raise RuntimeError("shutdown request PBS job identity mismatch")
    actual_host = socket.gethostname().split(".", 1)[0]
    if actual_host != host.split(".", 1)[0]:
        raise RuntimeError("shutdown request host identity mismatch")
    payload = _read_private(path)
    if payload != _expected(job_id, host):
        raise RuntimeError("shutdown request content mismatch")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("publish", "verify"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--request", type=Path, required=True)
        command_parser.add_argument("--pbs-job-id", required=True)
        command_parser.add_argument("--host", required=True)
    args = parser.parse_args(argv)
    if args.command == "publish":
        publish(args.request, job_id=args.pbs_job_id, host=args.host)
        print("node_services_shutdown_request=published")
    else:
        verify(args.request, job_id=args.pbs_job_id, host=args.host)
        print("node_services_shutdown_request=verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
