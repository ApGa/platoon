#!/usr/bin/env python3
"""Publish and verify fail-closed node-service cleanup receipts."""

from __future__ import annotations

import argparse
import json
import os
import socket
import stat
import time
from pathlib import Path
from typing import Any


def _safe_receipt(path: Path, *, must_exist: bool) -> None:
    if not path.is_absolute() or path.parent.resolve() != path.parent:
        raise RuntimeError("cleanup receipt path is not direct and absolute")
    parent_stat = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent_stat.st_mode)
        or parent_stat.st_uid != os.getuid()
        or stat.S_IMODE(parent_stat.st_mode) & 0o077
    ):
        raise RuntimeError("cleanup receipt parent is not private and user-owned")
    exists = path.exists() or path.is_symlink()
    if exists != must_exist:
        state = "missing" if must_exist else "stale"
        raise RuntimeError(f"cleanup receipt is {state}: {path}")
    if exists:
        value = path.lstat()
        if not stat.S_ISREG(value.st_mode) or value.st_uid != os.getuid() or stat.S_IMODE(value.st_mode) != 0o600:
            raise RuntimeError("cleanup receipt is not a private user-owned regular file")


def _expected(job_id: str, host: str, primary_status: int) -> dict[str, Any]:
    if not job_id or not host or not 0 <= primary_status <= 255:
        raise RuntimeError("cleanup receipt identity or primary status is invalid")
    return {
        "schema_version": 1,
        "status": "passed",
        "pbs_job_id": job_id,
        "host": host,
        "primary_status": primary_status,
        "cleanup_completed": True,
        "forced_kill": False,
        "endpoints_stopped": True,
        "swe_sessions_clean": True,
    }


def publish(path: Path, *, job_id: str, host: str, primary_status: int) -> None:
    if os.environ.get("PBS_JOBID") != job_id:
        raise RuntimeError("cleanup receipt PBS job identity mismatch")
    actual_host = socket.gethostname().split(".", 1)[0]
    if actual_host != host.split(".", 1)[0]:
        raise RuntimeError("cleanup receipt host identity mismatch")
    _safe_receipt(path, must_exist=False)
    value = _expected(job_id, host, primary_status)
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
            raise RuntimeError("cleanup receipt appeared during create-only publication") from error
        parent_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    _safe_receipt(path, must_exist=True)


def verify(path: Path, *, job_id: str, host: str, primary_status: int) -> dict[str, Any]:
    _safe_receipt(path, must_exist=True)
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("cleanup receipt is unreadable or malformed") from error
    if not isinstance(value, dict):
        raise RuntimeError("cleanup receipt is not an object")
    if value != _expected(job_id, host, primary_status):
        raise RuntimeError("cleanup receipt content mismatch")
    return value


def tcp_closed(host: str, ports: list[int], timeout_seconds: float, deadline_seconds: float) -> None:
    if not host or timeout_seconds <= 0 or deadline_seconds <= 0 or any(not 1 <= port <= 65535 for port in ports):
        raise RuntimeError("TCP closure probe arguments are invalid")
    for port in ports:
        deadline = time.monotonic() + deadline_seconds
        refusals = 0
        while refusals < 3:
            try:
                connection = socket.create_connection((host, port), timeout=timeout_seconds)
            except ConnectionRefusedError:
                refusals += 1
                if refusals < 3:
                    time.sleep(0.2)
            except (TimeoutError, OSError) as error:
                raise RuntimeError(f"TCP endpoint closure probe failed: {host}:{port}") from error
            else:
                connection.close()
                refusals = 0
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"TCP endpoint is still accepting connections: {host}:{port}")
                time.sleep(0.2)


def assert_empty_swe(session_root: Path) -> None:
    value = session_root.lstat()
    if not stat.S_ISDIR(value.st_mode) or value.st_uid != os.getuid():
        raise RuntimeError("SWE session root is not a direct user-owned directory")
    try:
        survivors = sorted(entry.name for entry in os.scandir(session_root) if entry.name.startswith("swe-rebench-"))
    except OSError as error:
        raise RuntimeError("SWE session root scan failed") from error
    if survivors:
        raise RuntimeError(f"SWE session entries survived cleanup: {survivors!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish_parser = subparsers.add_parser("publish")
    verify_parser = subparsers.add_parser("verify")
    tcp_parser = subparsers.add_parser("assert-tcp-closed")
    swe_parser = subparsers.add_parser("assert-empty-swe")
    for command_parser in (publish_parser, verify_parser):
        command_parser.add_argument("--receipt", type=Path, required=True)
        command_parser.add_argument("--pbs-job-id", required=True)
        command_parser.add_argument("--host", required=True)
        command_parser.add_argument("--primary-status", type=int, required=True)
    tcp_parser.add_argument("--host", required=True)
    tcp_parser.add_argument("--port", type=int, action="append", required=True)
    tcp_parser.add_argument("--timeout-seconds", type=float, default=2)
    tcp_parser.add_argument("--deadline-seconds", type=float, default=20)
    swe_parser.add_argument("--session-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "publish":
        publish(
            args.receipt,
            job_id=args.pbs_job_id,
            host=args.host,
            primary_status=args.primary_status,
        )
        print("node_services_cleanup_receipt=published")
    elif args.command == "verify":
        value = verify(
            args.receipt,
            job_id=args.pbs_job_id,
            host=args.host,
            primary_status=args.primary_status,
        )
        print(f"node_services_cleanup_receipt=passed primary_status={value['primary_status']}")
    elif args.command == "assert-tcp-closed":
        tcp_closed(args.host, args.port, args.timeout_seconds, args.deadline_seconds)
        print("node_services_tcp_endpoints=closed")
    else:
        assert_empty_swe(args.session_root)
        print("node_services_swe_sessions=empty")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
