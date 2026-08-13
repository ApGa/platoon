"""Keep a PBS allocation alive until exact role-launcher cleanup is durable.

The launcher, not this barrier, owns process signalling and record deletion.
This module only validates generation-specific cleanup receipts and verifies
that the exact control/discovery paths from any surviving manifest are gone.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
import time
from pathlib import Path
from typing import Any

_MANIFEST_VERSION = 1
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_GENERATION = re.compile(r"^[0-9a-f]{32}$")


class CleanupPending(RuntimeError):
    """The exact launcher generation has not finished cleanup yet."""


def _private_json(path: Path) -> dict[str, Any]:
    try:
        file_stat = path.lstat()
    except FileNotFoundError as error:
        raise CleanupPending(f"missing {path}") from error
    if (
        not stat.S_ISREG(file_stat.st_mode)
        or file_stat.st_uid != os.getuid()
        or stat.S_IMODE(file_stat.st_mode) & 0o077
    ):
        raise PermissionError(f"Refusing unsafe PBS cleanup artifact: {path}")
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"PBS cleanup artifact is not a JSON object: {path}")
    return value


def _lexists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _role_generation_from_name(path: Path, suffix: str) -> tuple[str, str]:
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected PBS cleanup artifact name: {path.name}")
    stem = path.name[: -len(suffix)]
    try:
        role, generation = stem.rsplit(".", 1)
    except ValueError as error:
        raise ValueError(f"Malformed PBS cleanup artifact name: {path.name}") from error
    if not role or not _SAFE_COMPONENT.fullmatch(role) or not _GENERATION.fullmatch(generation):
        raise ValueError(f"Malformed PBS role/generation: {path.name}")
    return role, generation


def _validate_status(
    path: Path,
    *,
    role: str,
    generation: str,
    host_count: int | None,
) -> None:
    status = _private_json(path)
    if (
        status.get("version") != _MANIFEST_VERSION
        or status.get("complete") is not True
        or status.get("role") != role
        or status.get("generation") != generation
    ):
        raise ValueError(f"Invalid PBS launcher cleanup status: {path}")
    failures = status.get("cleanup_failures")
    if not isinstance(failures, list):
        raise ValueError(f"PBS launcher status has invalid cleanup_failures: {path}")
    if failures:
        raise RuntimeError(
            f"PBS launcher reported cleanup failures for {role}.{generation}: "
            f"{json.dumps(failures, sort_keys=True)}"
        )
    if host_count is not None and status.get("cleanup_host_count") != host_count:
        raise ValueError(f"PBS launcher cleanup host count does not match manifest: {path}")


def _validate_manifest(
    path: Path,
    *,
    state_dir: Path,
    pbs_job_id: str,
    experiment: str,
    trial: str,
) -> tuple[str, str, Path, list[Path], list[Path], Path]:
    role, generation = _role_generation_from_name(path, ".manifest.json")
    manifest = _private_json(path)
    if (
        manifest.get("version") != _MANIFEST_VERSION
        or manifest.get("role") != role
        or manifest.get("generation") != generation
        or manifest.get("pbs_job_id") != pbs_job_id
    ):
        raise ValueError(f"PBS role manifest identity does not match its path: {path}")

    status_path = Path(str(manifest.get("launcher_status_path", "")))
    expected_status = state_dir / f"{role}.{generation}.launcher-status.json"
    if status_path != expected_status:
        raise ValueError(f"PBS launcher status escapes the exact state directory: {status_path}")

    hosts = manifest.get("hosts")
    if not isinstance(hosts, list) or not hosts:
        raise ValueError(f"PBS role manifest has no hosts: {path}")
    control_paths: list[Path] = []
    for host in hosts:
        if not isinstance(host, dict):
            raise ValueError(f"PBS role manifest has invalid host entry: {path}")
        control = Path(str(host.get("control_path", "")))
        if control.parent != state_dir or not control.name.startswith(f"{role}.{generation}."):
            raise ValueError(f"PBS control path escapes the exact generation: {control}")
        control_paths.append(control)

    discovery = manifest.get("discovery")
    if not isinstance(discovery, dict) or discovery.get("type") != "nfs":
        raise ValueError(f"PBS role manifest lacks NFS cleanup metadata: {path}")
    identity = discovery.get("lock_identity")
    if (
        not isinstance(identity, dict)
        or identity.get("experiment_name") != experiment
        or identity.get("trial_name") != trial
    ):
        raise ValueError(f"PBS discovery identity does not match this launch: {path}")
    record_root = Path(str(discovery.get("record_root", "")))
    role_root = Path(str(discovery.get("role_root", "")))
    registry_user = str(identity.get("registry_user", ""))
    expected_role_root = record_root / registry_user / experiment / trial / "workers" / role
    if not record_root.is_absolute() or role_root != expected_role_root:
        raise ValueError(f"PBS discovery role root is not canonical: {role_root}")
    raw_entries = discovery.get("entry_paths")
    replicas = int(manifest.get("replicas", -1))
    if not isinstance(raw_entries, list) or len(raw_entries) != replicas:
        raise ValueError(f"PBS discovery entry count does not match replicas: {path}")
    entries = [Path(str(value)) for value in raw_entries]
    expected_entries = [role_root / str(index) / "ENTRY" for index in range(replicas)]
    if entries != expected_entries:
        raise ValueError(f"PBS discovery entries are not exact worker paths: {path}")
    return role, generation, status_path, control_paths, entries, role_root


def assert_cleanup_complete(
    state_dir: Path,
    *,
    pbs_job_id: str,
    experiment: str,
    trial: str,
) -> int:
    """Validate every durable generation found in one exact PBS state dir."""

    if not state_dir.is_absolute():
        raise ValueError(f"PBS cleanup state directory must be absolute: {state_dir}")
    if state_dir.name != trial:
        raise ValueError(f"PBS cleanup state directory does not match trial {trial!r}: {state_dir}")
    try:
        directory_stat = state_dir.lstat()
    except FileNotFoundError:
        # A controller that failed before creating a scheduler role has no
        # cleanup work. Its exact job/trial directory legitimately never exists.
        return 0
    if (
        not stat.S_ISDIR(directory_stat.st_mode)
        or directory_stat.st_uid != os.getuid()
        or stat.S_IMODE(directory_stat.st_mode) & 0o077
    ):
        raise PermissionError(f"Refusing unsafe PBS cleanup state directory: {state_dir}")

    manifests = sorted(state_dir.glob("*.manifest.json"))
    statuses = sorted(state_dir.glob("*.launcher-status.json"))
    known: set[tuple[str, str]] = set()
    for manifest_path in manifests:
        role, generation, status_path, controls, entries, role_root = _validate_manifest(
            manifest_path,
            state_dir=state_dir,
            pbs_job_id=pbs_job_id,
            experiment=experiment,
            trial=trial,
        )
        known.add((role, generation))
        _validate_status(
            status_path,
            role=role,
            generation=generation,
            host_count=len(controls),
        )
        remaining = [str(path) for path in [*controls, *entries] if _lexists(path)]
        if _lexists(role_root):
            remaining.append(str(role_root))
        if remaining:
            raise CleanupPending(
                f"cleanup status exists but exact generation paths remain for "
                f"{role}.{generation}: {remaining[:8]}"
            )

    # A normal scheduler deletion unlinks its manifest only after validating
    # the status. Validate those job-local durable statuses as well.
    for status_path in statuses:
        role, generation = _role_generation_from_name(status_path, ".launcher-status.json")
        if (role, generation) in known:
            continue
        _validate_status(
            status_path,
            role=role,
            generation=generation,
            host_count=None,
        )
        prefix = f"{role}.{generation}."
        remaining_controls = [
            str(path)
            for path in state_dir.iterdir()
            if path.name.startswith(prefix) and path.name.endswith(".control.json")
        ]
        if remaining_controls:
            raise CleanupPending(
                f"manifest-free cleanup status still has controls for {role}.{generation}: "
                f"{remaining_controls[:8]}"
            )
        known.add((role, generation))
    return len(known)


def wait_for_cleanup(
    state_dir: Path,
    *,
    pbs_job_id: str,
    experiment: str,
    trial: str,
    timeout_seconds: float,
    poll_seconds: float = 0.2,
) -> int:
    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("PBS cleanup barrier timeouts must be positive")
    deadline = time.monotonic() + timeout_seconds
    last_pending: BaseException | None = None
    while True:
        try:
            return assert_cleanup_complete(
                state_dir,
                pbs_job_id=pbs_job_id,
                experiment=experiment,
                trial=trial,
            )
        except CleanupPending as error:
            last_pending = error
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out after {timeout_seconds:g}s waiting for PBS role cleanup: "
                f"{last_pending}"
            ) from last_pending
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--pbs-job-id", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args(argv)
    try:
        generations = wait_for_cleanup(
            args.state_dir,
            pbs_job_id=args.pbs_job_id,
            experiment=args.experiment,
            trial=args.trial,
            timeout_seconds=args.timeout_seconds,
        )
    except BaseException as error:
        print(f"PBS cleanup barrier failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"PBS cleanup barrier complete generations={generations}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
