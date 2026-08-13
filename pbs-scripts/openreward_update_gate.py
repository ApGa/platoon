#!/usr/bin/env python3
"""Publish and verify the 16-node optimizer-update gate for ABCI.

The marker is created only after AReaL has atomically published recover_info for
an optimizer step and a non-empty recovery checkpoint.  The 32-node recursive
wrapper validates the marker and its current evidence before entering PBS
lifecycle setup.
"""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
GATE_NAME = "abci-16node-nonrecursive-optimizer-update"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_root(fileroot: Path, experiment: str, trial: str) -> Path:
    return fileroot / "checkpoints" / getpass.getuser() / experiment / trial


def _read_step_info(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
        global_step = int(value["global_step"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    value["global_step"] = global_step
    return value


def _checkpoint_has_payload(path: Path) -> bool:
    try:
        return path.is_dir() and any(item.is_file() and item.stat().st_size > 0 for item in path.rglob("*"))
    except OSError:
        return False


def _atomic_private_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _evidence(
    fileroot: Path,
    experiment: str,
    trial: str,
    min_global_step: int,
    not_before_epoch: float,
) -> tuple[dict[str, Any], Path, Path] | None:
    root = _checkpoint_root(fileroot, experiment, trial)
    step_path = root / "recover_info" / "step_info.json"
    checkpoint = root / "default" / "recover_checkpoint"
    step = _read_step_info(step_path)
    if step is None or step["global_step"] < min_global_step:
        return None
    try:
        # RecoverInfo rewrites step_info after the checkpoint completes.  This
        # mtime guard prevents a pre-existing update from satisfying a fresh
        # job's monitor before that job itself completes another update.
        if step_path.stat().st_mtime < not_before_epoch:
            return None
    except OSError:
        return None
    if not _checkpoint_has_payload(checkpoint):
        return None
    return step, step_path, checkpoint


def monitor(args: argparse.Namespace) -> int:
    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    deadline = time.monotonic() + args.timeout_seconds if args.timeout_seconds else None
    while not stop:
        evidence = _evidence(
            args.fileroot,
            args.experiment,
            args.trial,
            args.min_global_step,
            args.not_before_epoch,
        )
        if evidence is not None:
            step, step_path, checkpoint = evidence
            record = {
                "schema_version": SCHEMA_VERSION,
                "gate": GATE_NAME,
                "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "pbs_job_id": args.pbs_job_id,
                "hostname": os.uname().nodename,
                "node_count": 16,
                "recursive": False,
                "experiment_name": args.experiment,
                "trial_name": args.trial,
                "config_path": str(args.config.resolve()),
                "global_step": step["global_step"],
                "recorded_step_info": step,
                "step_info_path": str(step_path),
                "step_info_sha256": _sha256(step_path),
                "recover_checkpoint_path": str(checkpoint),
                "training_sif_path": str(args.training_sif.resolve()),
                "training_sif_sha256": args.training_sif_sha256,
            }
            _atomic_private_json(args.marker, record)
            print(
                f"published_update_gate={args.marker} global_step={step['global_step']}",
                flush=True,
            )
            return 0
        if args.controller_pid and not Path(f"/proc/{args.controller_pid}").exists():
            print("controller exited before the 16-node update gate was reached", file=sys.stderr)
            return 1
        if deadline is not None and time.monotonic() >= deadline:
            print("timed out waiting for the 16-node update gate", file=sys.stderr)
            return 1
        time.sleep(args.poll_seconds)
    return 143


def verify(args: argparse.Namespace) -> int:
    try:
        marker_stat = args.marker.stat()
        if marker_stat.st_uid != os.getuid() or marker_stat.st_mode & 0o077:
            raise ValueError("marker is not private and owned by the current user")
        record = json.loads(args.marker.read_text())
        if record.get("schema_version") != SCHEMA_VERSION or record.get("gate") != GATE_NAME:
            raise ValueError("marker schema/gate mismatch")
        if record.get("node_count") != 16 or record.get("recursive") is not False:
            raise ValueError("marker does not attest a 16-node nonrecursive run")
        if record.get("experiment_name") != args.experiment or record.get("trial_name") != args.trial:
            raise ValueError("marker experiment/trial mismatch")
        recorded_step = int(record.get("global_step", -1))
        if recorded_step < args.min_global_step:
            raise ValueError(f"marker global_step={recorded_step} is below {args.min_global_step}")
        if record.get("training_sif_sha256") != args.training_sif_sha256:
            raise ValueError("training SIF digest differs from the validated 16-node runtime")

        root = _checkpoint_root(args.fileroot, args.experiment, args.trial)
        step_path = root / "recover_info" / "step_info.json"
        checkpoint = root / "default" / "recover_checkpoint"
        current_step = _read_step_info(step_path)
        if current_step is None or current_step["global_step"] < recorded_step:
            raise ValueError("current recover_info no longer contains the attested update")
        if not _checkpoint_has_payload(checkpoint):
            raise ValueError("current recovery checkpoint is absent or empty")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"update gate verification failed: {error}", file=sys.stderr)
        return 1
    print(
        f"verified_update_gate={args.marker} global_step={recorded_step} "
        f"current_global_step={current_step['global_step']}"
    )
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--fileroot", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--min-global-step", type=int, default=0)
    parser.add_argument("--training-sif", type=Path, required=True)
    parser.add_argument("--training-sif-sha256", required=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    monitor_parser = subparsers.add_parser("monitor")
    _common(monitor_parser)
    monitor_parser.add_argument("--config", type=Path, required=True)
    monitor_parser.add_argument("--pbs-job-id", required=True)
    monitor_parser.add_argument("--not-before-epoch", type=float, required=True)
    monitor_parser.add_argument("--controller-pid", type=int)
    monitor_parser.add_argument("--poll-seconds", type=float, default=20)
    monitor_parser.add_argument("--timeout-seconds", type=float, default=0)
    monitor_parser.set_defaults(function=monitor)

    verify_parser = subparsers.add_parser("verify")
    _common(verify_parser)
    verify_parser.set_defaults(function=verify)
    args = parser.parse_args(argv)
    if args.min_global_step < 0:
        parser.error("--min-global-step must be non-negative")
    if not args.training_sif.is_file():
        parser.error(f"training SIF is missing: {args.training_sif}")
    if len(args.training_sif_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in args.training_sif_sha256
    ):
        parser.error("--training-sif-sha256 must be a lowercase SHA-256 digest")
    if args.command == "monitor" and (args.poll_seconds <= 0 or args.timeout_seconds < 0):
        parser.error("poll interval must be positive and timeout non-negative")
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
