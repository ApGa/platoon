#!/usr/bin/env python3
"""Stop only native judge process groups whose recorded identity still matches."""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import time


def process_identity(pid):
    try:
        directory = Path("/proc") / str(pid)
        fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "uid": directory.stat().st_uid, "pgid": int(fields[2]),
                "start_time": fields[19], "executable": os.readlink(directory / "exe")}
    except (OSError, ValueError, IndexError):
        return None


def group_members(pgid):
    members = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            process = process_identity(int(path.name))
            if process and process["pgid"] == pgid:
                members.append(process)
    return members


def identity_matches(process, expected):
    return process is not None and all(process[key] == expected[key]
                                       for key in ("pid", "uid", "pgid", "start_time", "executable"))


def has_service_token(pid, token):
    try:
        entries = (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
        return ("FRONTIER_NATIVE_SERVICE_ID=" + token).encode() in entries
    except OSError:
        return False


def stop_record(path, *, root, grace_seconds):
    if path.is_symlink() or path.stat().st_uid != os.getuid():
        raise RuntimeError("Refusing an unowned native service marker")
    record = json.loads(path.read_text())
    if (record.get("version") != 1 or record.get("uid") != os.getuid()
            or record.get("hostname") != socket.gethostname()
            or record.get("boot_id") != Path("/proc/sys/kernel/random/boot_id").read_text().strip()):
        raise RuntimeError("Native service marker host/owner/boot does not match")
    session_path = Path(record["session_path"]).resolve()
    if root.resolve() not in session_path.parents:
        raise RuntimeError("Native service session is outside the configured root")
    pid, pgid = int(record["pid"]), int(record["pgid"])
    if pid <= 1 or pgid != pid:
        raise RuntimeError("Native service marker has no dedicated process group")
    leader = process_identity(pid)
    members = group_members(pgid)
    if leader is not None and not identity_matches(leader, record):
        raise RuntimeError("Native service PID was reused; refusing to signal it")
    # A live, verified group leader proves its group. If only descendants remain,
    # require the unique service token on every survivor before signaling.
    if any(member["uid"] != os.getuid() for member in members):
        raise RuntimeError("Native service group contains a foreign process")
    if leader is None and any(not has_service_token(member["pid"], record["service_instance_id"])
                              for member in members):
        raise RuntimeError("Cannot prove ownership of orphaned native judge helpers")
    snapshot = {member["pid"]: member for member in members}
    if snapshot:
        for number in (signal.SIGTERM, signal.SIGKILL):
            # Signal individually verified PIDs, avoiding a PGID reuse race.
            for process_pid, identity in snapshot.items():
                if identity_matches(process_identity(process_pid), identity):
                    try:
                        os.kill(process_pid, number)
                    except ProcessLookupError:
                        pass
            deadline = time.monotonic() + (grace_seconds if number == signal.SIGTERM else 5)
            while time.monotonic() < deadline:
                if all(not identity_matches(process_identity(process_pid), identity)
                       for process_pid, identity in snapshot.items()):
                    break
                time.sleep(0.1)
        if any(identity_matches(process_identity(process_pid), identity)
               for process_pid, identity in snapshot.items()):
            raise RuntimeError("Native judge process did not stop")
    path.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--stop-grace-seconds", type=float, default=10)
    args = parser.parse_args()
    markers = args.session_root / ".native-services"
    failures = []
    if markers.is_dir():
        for path in sorted(markers.glob("*.json")):
            try:
                stop_record(path, root=args.session_root, grace_seconds=args.stop_grace_seconds)
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                failures.append(f"{path.name}: {exc}")
    for failure in failures:
        print("[frontier-native-cleanup] " + failure, flush=True)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
