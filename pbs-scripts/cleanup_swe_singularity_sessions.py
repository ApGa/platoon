#!/usr/bin/env python3
"""Fail-closed cleanup for exact OpenReward SWE Singularity instances.

ABCI's Singularity CE 4.4.1 writes unprivileged instance metadata below the
real passwd-database home and ignores both CONFIGDIR aliases and an exported
HOME for this purpose. Each backend session therefore publishes a UUID name,
runtime ``--pid-file``, hostname, PID start time, and owner lease. This helper
uses all of that identity before stopping anything. It never uses wildcards,
``instance stop --all``, or a user-wide instance listing.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

OWNER_FILE = ".openreward-session-owner.json"
SESSION_LOCK_FILE = ".openreward-session.lock"
JANITOR_LOCK_FILE = ".openreward-session-janitor.lock"
INSTANCE_CONTROL_LOCK_FILE = ".openreward-instance-control.lock"
INSTANCE_PID_FILE = "instance.pid"


def _runtime_environment(runtime: Path) -> dict[str, str]:
    environment = dict(os.environ)
    for key in (
        "APPTAINER_BIND",
        "APPTAINER_BINDPATH",
        "SINGULARITY_BIND",
        "SINGULARITY_BINDPATH",
        "APPTAINER_CONFIGDIR",
        "SINGULARITY_CONFIGDIR",
    ):
        environment.pop(key, None)
    for key in list(environment):
        if key.startswith(("APPTAINERENV_", "SINGULARITYENV_")):
            environment.pop(key)
    return environment


def _run(runtime: Path, arguments: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(runtime), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        env=_runtime_environment(runtime),
    )


def _instance_record(runtime: Path, name: str) -> dict[str, Any] | None:
    result = _run(runtime, ["instance", "list", "--json", name], 30)
    if result.returncode != 0:
        raise RuntimeError(
            f"could not query exact instance {name!r}: {result.stdout.strip()}"
        )
    try:
        payload: Any = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"invalid instance-list JSON for {name!r}: {result.stdout!r}"
        ) from error
    instances = payload.get("instances") if isinstance(payload, dict) else None
    if not isinstance(instances, list):
        raise RuntimeError(f"instance-list response has no instances array: {payload!r}")
    matches = [
        item
        for item in instances
        if isinstance(item, dict)
        and item.get("instance", item.get("name")) == name
    ]
    if len(matches) > 1:
        raise RuntimeError(f"multiple runtime records matched exact instance {name!r}")
    return matches[0] if matches else None


def _instance_pid(record: dict[str, Any]) -> int:
    for key in ("pid", "PID", "Pid"):
        try:
            pid = int(record[key])
        except (KeyError, TypeError, ValueError):
            continue
        if pid > 0:
            return pid
    raise RuntimeError(f"instance runtime record has no valid PID: {record}")


def _session_instance_name(session_dir: Path) -> str:
    prefix = "swe-rebench-"
    suffix = session_dir.name.removeprefix(prefix)
    if (
        len(suffix) < 18
        or suffix[16] != "-"
        or any(character not in "0123456789abcdef" for character in suffix[:16])
    ):
        raise RuntimeError(f"refusing malformed SWE session name: {session_dir}")
    return prefix + suffix[:16]


def _process_state_and_start_time(pid: int) -> tuple[str, str] | None:
    try:
        contents = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    command_end = contents.rfind(")")
    if command_end < 0:
        return None
    fields = contents[command_end + 2 :].split()
    if len(fields) <= 19:
        return None
    return fields[0], fields[19]


def _process_identity_matches(pid: int, start_time: str) -> bool:
    identity = _process_state_and_start_time(pid)
    return identity is not None and identity[1] == start_time


def _instance_process_start_time(pid: int, name: str) -> str | None:
    if pid <= 0:
        return None
    try:
        process_stat = Path(f"/proc/{pid}").stat()
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        return None
    if process_stat.st_uid != os.getuid():
        return None
    decoded = command.decode(errors="replace")
    if name not in decoded or "Singularity instance:" not in decoded:
        return None
    identity = _process_state_and_start_time(pid)
    if identity is None or identity[0] == "Z":
        return None
    return identity[1]


def _owned_process_start_time(pid: int) -> str | None:
    if pid <= 0:
        return None
    try:
        process_stat = Path(f"/proc/{pid}").stat()
    except OSError:
        return None
    if process_stat.st_uid != os.getuid():
        return None
    identity = _process_state_and_start_time(pid)
    if identity is None or identity[0] == "Z":
        return None
    return identity[1]


def _owner_is_live(session_dir: Path) -> bool:
    owner_path = session_dir / OWNER_FILE
    try:
        owner_stat = os.stat(owner_path, follow_symlinks=False)
        if (
            not stat.S_ISREG(owner_stat.st_mode)
            or owner_stat.st_uid != os.getuid()
            or stat.S_IMODE(owner_stat.st_mode) & 0o077
        ):
            return False
        record = json.loads(owner_path.read_text())
        return _process_identity_matches(
            int(record["owner_pid"]),
            str(record["owner_start_time"]),
        )
    except (FileNotFoundError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _open_private_lock(path: Path, *, create: bool) -> Any:
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    if create:
        flags |= os.O_CREAT
    descriptor = os.open(path, flags, 0o600)
    try:
        lock_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(lock_stat.st_mode)
            or lock_stat.st_uid != os.getuid()
            or stat.S_IMODE(lock_stat.st_mode) & 0o077
        ):
            raise RuntimeError(f"refusing non-private or foreign lock: {path}")
        return os.fdopen(descriptor, "a+")
    except BaseException:
        os.close(descriptor)
        raise


def _read_pid_file(path: Path) -> int | None:
    try:
        file_stat = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_uid != os.getuid()
            or stat.S_IMODE(file_stat.st_mode) & 0o022
        ):
            raise RuntimeError(f"refusing unsafe instance PID file: {path}")
        pid = int(path.read_text().strip())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        raise RuntimeError(f"invalid instance PID file {path}: {error}") from error
    if pid <= 0:
        raise RuntimeError(f"invalid PID {pid} in {path}")
    return pid


def _validated_session(
    session_root: Path,
    session_dir: Path,
) -> tuple[str, Path, dict[str, Any] | None]:
    if session_dir.is_symlink() or not session_dir.is_dir():
        raise RuntimeError(f"refusing non-directory SWE session: {session_dir}")
    session_stat = session_dir.stat()
    if session_stat.st_uid != os.getuid() or stat.S_IMODE(session_stat.st_mode) & 0o077:
        raise RuntimeError(f"refusing non-private or foreign SWE session: {session_dir}")
    if session_dir.parent.resolve() != session_root.resolve():
        raise RuntimeError(f"SWE session escaped its configured root: {session_dir}")
    name = _session_instance_name(session_dir)
    pid_path = session_dir / INSTANCE_PID_FILE
    try:
        owner = json.loads((session_dir / OWNER_FILE).read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        owner = None
    return name, pid_path, owner


def _owner_binds_identity(
    owner: dict[str, Any] | None,
    *,
    name: str,
    session_dir: Path,
    pid_path: Path,
) -> bool:
    return bool(
        isinstance(owner, dict)
        and owner.get("schema_version") == 2
        and owner.get("instance_name") == name
        and owner.get("session_directory") == str(session_dir)
        and owner.get("hostname") == socket.gethostname()
        and owner.get("instance_pid_file") == str(pid_path)
    )


def _terminate_exact(pid: int, name: str, start_time: str, grace: float) -> None:
    if _instance_process_start_time(pid, name) != start_time:
        raise RuntimeError(f"refusing to signal {name!r}: PID identity changed")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + max(0.0, grace)
    while _process_identity_matches(pid, start_time) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _process_identity_matches(pid, start_time):
        if _instance_process_start_time(pid, name) != start_time:
            raise RuntimeError(f"refusing to kill {name!r}: PID identity changed")
        os.kill(pid, signal.SIGKILL)
        deadline = time.monotonic() + 5
        while _process_identity_matches(pid, start_time) and time.monotonic() < deadline:
            time.sleep(0.05)
    if _process_identity_matches(pid, start_time):
        raise RuntimeError(f"exact instance process {name!r} survived SIGKILL")


def cleanup_sessions(session_root: Path, runtime: Path, stop_grace_seconds: int) -> int:
    if not session_root.exists():
        return 0
    if session_root.is_symlink() or not session_root.is_dir():
        raise RuntimeError(f"refusing invalid SWE session root: {session_root}")
    if session_root.stat().st_uid != os.getuid():
        raise RuntimeError(f"refusing SWE session root not owned by uid {os.getuid()}: {session_root}")

    cleaned = 0
    errors: list[str] = []
    with _open_private_lock(session_root / JANITOR_LOCK_FILE, create=True) as janitor_lock:
        fcntl.flock(janitor_lock.fileno(), fcntl.LOCK_EX)
        for session_dir in sorted(session_root.glob("swe-rebench-*")):
            session_lock = None
            try:
                name, pid_path, owner = _validated_session(session_root, session_dir)
                lock_path = session_dir / SESSION_LOCK_FILE
                if not lock_path.exists():
                    minimum_age = float(
                        os.getenv("SWE_APPTAINER_LEGACY_SESSION_MIN_AGE_SECONDS", "300")
                    )
                    age = time.time() - session_dir.stat().st_mtime
                    if age < minimum_age:
                        raise RuntimeError(
                            f"legacy session has no lease and is only {age:.1f}s old "
                            f"(minimum {minimum_age:.1f}s)"
                        )
                session_lock = _open_private_lock(lock_path, create=True)
                try:
                    fcntl.flock(session_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise RuntimeError(
                        "session lease is still held by a live process"
                    ) from error
                if _owner_is_live(session_dir):
                    raise RuntimeError("session owner identity is still live")

                bound = _owner_binds_identity(
                    owner,
                    name=name,
                    session_dir=session_dir,
                    pid_path=pid_path,
                )
                pid_from_file = _read_pid_file(pid_path)
                with _open_private_lock(
                    session_root / INSTANCE_CONTROL_LOCK_FILE,
                    create=True,
                ) as control_lock:
                    fcntl.flock(control_lock.fileno(), fcntl.LOCK_EX)
                    runtime_record = _instance_record(runtime, name)
                    if runtime_record is not None:
                        if not bound:
                            raise RuntimeError(
                                f"instance {name!r} exists but its owner identity is unbound"
                            )
                        assert owner is not None
                        runtime_pid = _instance_pid(runtime_record)
                        try:
                            recorded_pid = int(owner["instance_pid"])
                            recorded_start = str(owner["instance_start_time"])
                        except (KeyError, TypeError, ValueError) as error:
                            raise RuntimeError(
                                f"instance {name!r} has incomplete PID identity"
                            ) from error
                        if (
                            runtime_pid != recorded_pid
                            or pid_from_file != recorded_pid
                            or _owned_process_start_time(runtime_pid)
                            != recorded_start
                        ):
                            raise RuntimeError(
                                f"instance {name!r} PID identity is ambiguous"
                            )
                        stopped = _run(
                            runtime,
                            [
                                "instance",
                                "stop",
                                "--timeout",
                                str(stop_grace_seconds),
                                name,
                            ],
                            max(60, stop_grace_seconds + 10),
                        )
                        if _instance_record(runtime, name) is not None:
                            raise RuntimeError(
                                f"instance {name!r} survived stop: {stopped.stdout.strip()}"
                            )
                        if _process_identity_matches(runtime_pid, recorded_start):
                            _terminate_exact(runtime_pid, name, recorded_start, 2)
                    elif bound:
                        assert owner is not None
                        try:
                            recorded_pid = int(owner["instance_pid"])
                            recorded_start = str(owner["instance_start_time"])
                        except (KeyError, TypeError, ValueError):
                            recorded_pid = -1
                            recorded_start = ""
                        if recorded_pid > 0 and _process_identity_matches(
                            recorded_pid,
                            recorded_start,
                        ):
                            if pid_from_file != recorded_pid:
                                raise RuntimeError(
                                    f"instance {name!r} PID file does not match"
                                )
                            _terminate_exact(recorded_pid, name, recorded_start, 2)
                    elif pid_from_file is not None and _instance_process_start_time(
                        pid_from_file,
                        name,
                    ) is not None:
                        raise RuntimeError(
                            f"unowned live instance process remains for {name!r}"
                        )
                shutil.rmtree(session_dir)
                cleaned += 1
            except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired, RuntimeError) as error:
                errors.append(f"{session_dir}: {error}")
            finally:
                if session_lock is not None:
                    session_lock.close()

    if errors:
        raise RuntimeError("SWE session cleanup did not complete:\n" + "\n".join(errors))
    return cleaned


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--stop-grace-seconds", type=int, default=10)
    args = parser.parse_args(argv)
    if args.stop_grace_seconds < 0:
        parser.error("--stop-grace-seconds must be non-negative")
    if not args.runtime.is_file() or not os.access(args.runtime, os.X_OK):
        parser.error(f"runtime is not executable: {args.runtime}")
    cleaned = cleanup_sessions(args.session_root, args.runtime, args.stop_grace_seconds)
    print(f"swe_singularity_sessions_cleaned={cleaned}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"[swe-session-cleanup] ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
