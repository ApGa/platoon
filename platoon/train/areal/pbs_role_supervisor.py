"""Per-host process supervisor for preallocated PBS AReaL roles.

The controller starts one copy of this module on each selected physical host
through ``pbsdsh``.  A supervisor starts every worker assigned to that host,
sets explicit rank/GPU environment, and owns cleanup for those children.

Control records contain both PIDs and Linux process start times.  A later
launcher may therefore clean up a stale role without signalling an unrelated
process that happened to reuse a PID.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_MANIFEST_VERSION = 1
_GPU_UUID = re.compile(r"^GPU-[0-9A-Fa-f]+(?:-[0-9A-Fa-f]+)*$")
_RUNTIME_CONTROL_ENVIRONMENT = {
    "APPTAINER_BIND",
    "APPTAINER_BINDPATH",
    "SINGULARITY_BIND",
    "SINGULARITY_BINDPATH",
}


def _safe_runtime_environment_name(name: str) -> bool:
    return name not in _RUNTIME_CONTROL_ENVIRONMENT and not name.startswith(("APPTAINERENV_", "SINGULARITYENV_"))


def _process_details(pid: int) -> tuple[int, str] | None:
    try:
        contents = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    # Field two (comm) is parenthesized and may itself contain spaces or ')'.
    closing_paren = contents.rfind(")")
    if closing_paren < 0:
        return None
    fields_after_comm = contents[closing_paren + 1 :].split()
    # starttime is field 22; the first value above is field 3.
    if len(fields_after_comm) <= 19:
        return None
    try:
        return int(fields_after_comm[19]), fields_after_comm[0]
    except ValueError:
        return None


def process_start_time(pid: int) -> int | None:
    """Return Linux ``/proc`` start time ticks for *pid*, if it still exists."""

    details = _process_details(pid)
    return details[0] if details is not None else None


def _process_state(pid: int) -> str | None:
    details = _process_details(pid)
    return details[1] if details is not None else None


def process_identity(pid: int, *, process_group: bool = True) -> dict[str, Any]:
    start_time = process_start_time(pid)
    if start_time is None:
        raise RuntimeError(f"Could not determine start time for process {pid}")
    return {"pid": pid, "start_time": start_time, "process_group": process_group}


def identity_is_live(identity: dict[str, Any]) -> bool:
    try:
        pid = int(identity["pid"])
        expected = int(identity["start_time"])
    except (KeyError, TypeError, ValueError):
        return False
    actual_start = process_start_time(pid)
    state = _process_state(pid)
    if actual_start == expected and state != "Z":
        return True
    # A process-group leader can exit before TERM-ignoring descendants. Linux
    # keeps its PGID reserved while members remain, so it is safe to signal
    # that group only when the original PID itself has not been reused.
    if identity.get("process_group", True) and actual_start is None:
        return _process_group_has_live_members(pid)
    if identity.get("process_group", True) and actual_start == expected and state == "Z":
        return _process_group_has_live_members(pid, exclude_pid=pid)
    return False


def _process_group_has_live_members(pgid: int, *, exclude_pid: int | None = None) -> bool:
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            member_pid = int(stat_path.parent.name)
            contents = stat_path.read_text()
        except (OSError, ValueError):
            continue
        if member_pid == exclude_pid:
            continue
        closing_paren = contents.rfind(")")
        fields = contents[closing_paren + 1 :].split() if closing_paren >= 0 else []
        # Fields after comm start with state (3), ppid (4), pgrp (5).
        if len(fields) <= 2 or fields[0] == "Z":
            continue
        try:
            if int(fields[2]) == pgid:
                return True
        except ValueError:
            continue
    return False


def _same_host(expected: str, actual: str) -> bool:
    return expected == actual or expected.split(".", 1)[0] == actual.split(".", 1)[0]


def _read_private_json(path: Path) -> dict[str, Any]:
    file_stat = path.stat()
    if file_stat.st_uid != os.getuid():
        raise PermissionError(f"Refusing control file not owned by this user: {path}")
    if stat.S_IMODE(file_stat.st_mode) & 0o077:
        raise PermissionError(f"Refusing non-private control file: {path}")
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _atomic_private_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except PermissionError:
        pass
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _signal_identity(identity: dict[str, Any], sig: signal.Signals) -> bool:
    """Signal a process group only when its leader identity still matches."""

    if not identity_is_live(identity):
        return False
    pid = int(identity["pid"])
    try:
        if identity.get("process_group", True):
            os.killpg(pid, sig)
        else:
            os.kill(pid, sig)
    except ProcessLookupError:
        return False
    return True


def terminate_identities(
    identities: list[dict[str, Any]],
    grace_seconds: float,
    *,
    reverse_order: bool = False,
) -> None:
    if reverse_order:
        rank_zero = [item for item in identities if item.get("worker_index") == 0]
        others = [item for item in identities if item.get("worker_index") != 0]
        if others:
            terminate_identities(others, grace_seconds)
        if rank_zero:
            terminate_identities(rank_zero, grace_seconds)
        return
    live = [identity for identity in identities if identity_is_live(identity)]
    for identity in live:
        _signal_identity(identity, signal.SIGTERM)

    deadline = time.monotonic() + max(0.0, grace_seconds)
    while live and time.monotonic() < deadline:
        live = [identity for identity in live if identity_is_live(identity)]
        if live:
            time.sleep(0.05)

    for identity in live:
        _signal_identity(identity, signal.SIGKILL)


def _confirm_no_live_identities(
    identities: list[dict[str, Any]],
    timeout: float,
) -> None:
    """Fail closed unless every exact PID/start-time or process group is gone."""

    deadline = time.monotonic() + max(0.0, timeout)
    live = [identity for identity in identities if identity_is_live(identity)]
    while live and time.monotonic() < deadline:
        time.sleep(0.05)
        live = [identity for identity in identities if identity_is_live(identity)]
    if live:
        survivors = [
            {
                "pid": identity.get("pid"),
                "start_time": identity.get("start_time"),
                "worker_index": identity.get("worker_index"),
            }
            for identity in live
        ]
        raise RuntimeError(f"PBS cleanup left live process identities: {survivors}")


def cleanup_host(
    control_path: Path,
    grace_seconds: float,
    *,
    reverse_order: bool = False,
    expected_generation: str | None = None,
) -> bool:
    """Clean a stale supervisor record after validating host and PID identity."""

    try:
        record = _read_private_json(control_path)
    except FileNotFoundError:
        return False

    if expected_generation is not None and record.get("generation") != expected_generation:
        # A delayed cleanup from an older launcher must never signal or unlink
        # a replacement generation using the same role/control path. Report
        # failure so the older controller also keeps the node reserved.
        raise RuntimeError(
            f"Control record {control_path} belongs to replacement generation "
            f"{record.get('generation')!r}, not {expected_generation!r}"
        )

    expected_host = str(record.get("hostname", ""))
    actual_host = socket.gethostname()
    if not expected_host or not _same_host(expected_host, actual_host):
        raise RuntimeError(f"Control record {control_path} belongs to {expected_host!r}, not this host {actual_host!r}")

    # Ask the old supervisor to perform its own orderly cleanup, then target
    # any children that remain. Every signal is protected by start-time checks.
    supervisor = record.get("supervisor")
    if isinstance(supervisor, dict):
        terminate_identities([supervisor], grace_seconds)
    children = [item for item in record.get("children", []) if isinstance(item, dict)]
    terminate_identities(children, grace_seconds, reverse_order=reverse_order)
    identities = ([supervisor] if isinstance(supervisor, dict) else []) + children
    _confirm_no_live_identities(
        identities,
        timeout=max(1.0, min(5.0, grace_seconds + 1.0)),
    )

    # The stable role lifecycle lock prevents a replacement supervisor from
    # publishing here until this cleanup returns.  Re-read nonetheless so a
    # malformed/delayed cleanup fails closed instead of unlinking a replacement
    # generation's record.
    try:
        current = _read_private_json(control_path)
    except FileNotFoundError:
        return True
    if current.get("generation") != record.get("generation"):
        raise RuntimeError(f"Control record {control_path} was replaced during cleanup")
    control_path.unlink()
    return True


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_private_json(path)
    if manifest.get("version") != _MANIFEST_VERSION:
        raise ValueError(f"Unsupported PBS role manifest version {manifest.get('version')!r}")
    expected_job_id = str(manifest.get("pbs_job_id", ""))
    actual_job_id = os.environ.get("PBS_JOBID", "")
    if expected_job_id and actual_job_id and expected_job_id != actual_job_id:
        raise RuntimeError(f"PBS manifest belongs to job {expected_job_id!r}, not {actual_job_id!r}")
    return manifest


def _requested_reverse_order(manifest: dict[str, Any]) -> bool:
    request_value = manifest.get("termination_request_path")
    if not request_value:
        return bool(manifest.get("termination_reverse_order_default", False))
    try:
        request = _read_private_json(Path(str(request_value)))
    except FileNotFoundError:
        return bool(manifest.get("termination_reverse_order_default", False))
    if (
        request.get("version") != _MANIFEST_VERSION
        or request.get("role") != manifest.get("role")
        or request.get("generation") != manifest.get("generation")
    ):
        raise ValueError(f"Invalid PBS termination request: {request_value}")
    return bool(request.get("reverse_order", False))


def _gpu_visibility_config(manifest: dict[str, Any]) -> tuple[str, str]:
    value = manifest.get("gpu_visibility")
    if value is None:
        return "legacy", ""
    if not isinstance(value, dict):
        raise ValueError("gpu_visibility must be a JSON object")
    mode = str(value.get("mode", "legacy"))
    if mode == "legacy":
        return mode, ""
    if mode != "nvidia-container-cli":
        raise ValueError(f"Unsupported GPU visibility mode {mode!r}")
    binary = Path(str(value.get("nvidia_smi_binary", "")))
    if not binary.is_absolute():
        raise ValueError("nvidia-container-cli GPU visibility requires an absolute nvidia-smi path")
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"nvidia-smi is not an executable file: {binary}")
    return mode, str(binary)


def _nvidia_gpu_uuid_map(binary: str) -> dict[int, str]:
    """Resolve physical NVIDIA indexes to immutable UUIDs once per host."""

    try:
        completed = subprocess.run(
            [
                binary,
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )
    except subprocess.CalledProcessError as error:
        detail = re.sub(r"[^\x20-\x7e]+", " ", error.stderr or "").strip()[:500]
        raise RuntimeError(f"nvidia-smi GPU UUID query failed with status {error.returncode}: {detail}") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("nvidia-smi GPU UUID query timed out after 15 seconds") from error
    except OSError as error:
        detail = re.sub(r"[^\x20-\x7e]+", " ", str(error)).strip()[:500]
        raise RuntimeError(f"Could not execute nvidia-smi GPU UUID query: {detail}") from error

    lines = completed.stdout.splitlines()
    if not lines:
        raise RuntimeError("nvidia-smi GPU UUID query returned no devices")
    result: dict[int, str] = {}
    seen_uuids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise RuntimeError(f"Malformed nvidia-smi GPU mapping on line {line_number}")
        try:
            index = int(fields[0])
        except ValueError as error:
            raise RuntimeError(f"Invalid NVIDIA GPU index on line {line_number}: {fields[0]!r}") from error
        uuid = fields[1]
        if index < 0 or index in result:
            raise RuntimeError(f"Duplicate or invalid NVIDIA GPU index {index} on line {line_number}")
        if not _GPU_UUID.fullmatch(uuid) or uuid in seen_uuids:
            raise RuntimeError(f"Duplicate or invalid NVIDIA GPU UUID on line {line_number}: {uuid!r}")
        result[index] = uuid
        seen_uuids.add(uuid)
    return result


def _validated_assignments(host: dict[str, Any]) -> list[tuple[dict[str, Any], list[int]]]:
    assignments = host.get("workers", [])
    if not isinstance(assignments, list) or not assignments:
        raise ValueError(f"No worker assignments for host {host.get('hostname')}")
    validated: list[tuple[dict[str, Any], list[int]]] = []
    assigned: set[int] = set()
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise ValueError("PBS worker assignment must be a JSON object")
        gpu_ids = [int(value) for value in assignment.get("gpu_ids", [])]
        if not gpu_ids or any(value < 0 for value in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError(f"Invalid GPU assignment {gpu_ids!r}")
        overlap = assigned.intersection(gpu_ids)
        if overlap:
            raise ValueError(f"Overlapping physical GPU assignment: {sorted(overlap)}")
        assigned.update(gpu_ids)
        validated.append((assignment, gpu_ids))
    return validated


def _containerized_environment(
    base: dict[str, str],
    worker_environment: dict[str, Any],
    rank_environment: dict[str, str],
) -> dict[str, str]:
    result = {key: value for key, value in base.items() if _safe_runtime_environment_name(key)}
    explicit = {
        str(key): str(value) for key, value in worker_environment.items() if _safe_runtime_environment_name(str(key))
    }
    explicit.update(rank_environment)
    result.update(explicit)
    # Both names are populated intentionally: SingularityCE consumes the first,
    # while Apptainer consumes the second. This keeps --cleanenv explicit.
    for name, value in explicit.items():
        result[f"SINGULARITYENV_{name}"] = value
        result[f"APPTAINERENV_{name}"] = value
    return result


def _record_for(
    role: str,
    generation: str,
    hostname: str,
    supervisor: dict[str, Any],
    children: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "version": _MANIFEST_VERSION,
        "role": role,
        "generation": generation,
        "hostname": hostname,
        "supervisor": supervisor,
        "children": children,
    }


def supervise_host(manifest_path: Path, host_index: int) -> int:
    manifest = _load_manifest(manifest_path)
    hosts = manifest.get("hosts")
    if not isinstance(hosts, list) or not (0 <= host_index < len(hosts)):
        raise ValueError(f"Invalid host index {host_index} in {manifest_path}")
    host = hosts[host_index]
    generation = str(manifest.get("generation", ""))
    if not generation:
        raise ValueError("PBS role manifest is missing a generation nonce")
    expected_hostname = str(host["hostname"])
    actual_hostname = socket.gethostname()
    if not _same_host(expected_hostname, actual_hostname):
        raise RuntimeError(f"pbsdsh placed role on {actual_hostname!r}; expected {expected_hostname!r}")

    control_path = Path(str(host["control_path"]))
    grace_seconds = float(manifest.get("termination_grace_seconds", 10.0))
    # A previous controller may have died before reaping this same role.
    cleanup_host(control_path, grace_seconds)

    worker_command = manifest.get("worker_command")
    if not isinstance(worker_command, list) or not worker_command:
        raise ValueError("worker_command must be a non-empty argument vector")
    worker_command = [str(argument) for argument in worker_command]
    worker_environment = manifest.get("worker_environment", {})
    if not isinstance(worker_environment, dict):
        raise ValueError("worker_environment must be a JSON object")

    assignments = _validated_assignments(host)
    gpu_visibility_mode, nvidia_smi_binary = _gpu_visibility_config(manifest)
    gpu_uuid_map: dict[int, str] = {}
    if gpu_visibility_mode == "nvidia-container-cli":
        gpu_uuid_map = _nvidia_gpu_uuid_map(nvidia_smi_binary)
        requested_gpu_ids = {gpu_id for _assignment, gpu_ids in assignments for gpu_id in gpu_ids}
        missing_gpu_ids = sorted(requested_gpu_ids.difference(gpu_uuid_map))
        if missing_gpu_ids:
            raise RuntimeError(f"nvidia-smi did not report assigned physical GPU indexes: {missing_gpu_ids}")

    supervisor_identity = process_identity(os.getpid(), process_group=False)
    children: list[tuple[subprocess.Popen[Any], dict[str, Any]]] = []
    stopping = False
    worker_failed = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    _atomic_private_json(
        control_path,
        _record_for(
            str(manifest["role"]),
            generation,
            expected_hostname,
            supervisor_identity,
            [],
        ),
    )
    try:
        for assignment, gpu_ids in assignments:
            if stopping:
                break
            worker_index = int(assignment["worker_index"])
            local_rank = int(assignment["local_rank"])
            assigned_gpu_ids = ",".join(str(value) for value in gpu_ids)
            assigned_gpu_uuids = ""
            visible_devices = assigned_gpu_ids
            if gpu_visibility_mode == "nvidia-container-cli":
                assigned_gpu_uuids = ",".join(gpu_uuid_map[value] for value in gpu_ids)
                visible_devices = ",".join(str(value) for value in range(len(gpu_ids)))
            rank_environment = {
                "PLATOON_WORKER_INDEX": str(worker_index),
                "PLATOON_LOCAL_RANK": str(local_rank),
                "PLATOON_ROLE": str(manifest["role"]),
                "PLATOON_ROLE_GENERATION": generation,
                "PLATOON_CONTAINER_IMAGE": str(manifest.get("container_image", "")),
                "PLATOON_CONTAINER_IMAGE_DIGEST": str(manifest.get("container_image_digest", "")),
                "PLATOON_GPU_VISIBILITY_MODE": gpu_visibility_mode,
                "PLATOON_ASSIGNED_GPU_IDS": assigned_gpu_ids,
                "PLATOON_ASSIGNED_GPU_UUIDS": assigned_gpu_uuids,
                "CUDA_VISIBLE_DEVICES": visible_devices,
                "ASCEND_RT_VISIBLE_DEVICES": assigned_gpu_ids,
            }
            environment = _containerized_environment(
                os.environ,
                worker_environment,
                rank_environment,
            )
            if gpu_visibility_mode == "nvidia-container-cli":
                # SingularityCE consumes this unprefixed host variable while
                # constructing the contained /dev tree. It must not come from
                # SINGULARITYENV_/APPTAINERENV_ or an inherited controller
                # value, and it need not be passed through to the worker.
                environment["NVIDIA_VISIBLE_DEVICES"] = assigned_gpu_uuids
                environment.pop("SINGULARITYENV_NVIDIA_VISIBLE_DEVICES", None)
                environment.pop("APPTAINERENV_NVIDIA_VISIBLE_DEVICES", None)
            print(
                f"PBS GPU mapping role={manifest['role']} worker_index={worker_index} "
                f"host={expected_hostname} physical_ids={assigned_gpu_ids} "
                f"physical_uuids={assigned_gpu_uuids or '-'} inner_cuda={visible_devices} "
                f"mode={gpu_visibility_mode}",
                file=sys.stderr,
                flush=True,
            )
            process = subprocess.Popen(
                worker_command,
                env=environment,
                start_new_session=True,
            )
            identity: dict[str, Any] = process_identity(process.pid)
            identity.update(
                {
                    "worker_index": worker_index,
                    "local_rank": local_rank,
                    "gpu_ids": gpu_ids,
                    "gpu_uuids": [gpu_uuid_map[value] for value in gpu_ids]
                    if gpu_visibility_mode == "nvidia-container-cli"
                    else [],
                }
            )
            children.append((process, identity))
            _atomic_private_json(
                control_path,
                _record_for(
                    str(manifest["role"]),
                    generation,
                    expected_hostname,
                    supervisor_identity,
                    [item for _, item in children],
                ),
            )

        while not stopping:
            failed = next(
                ((process, identity) for process, identity in children if process.poll() is not None),
                None,
            )
            if failed is not None:
                process, identity = failed
                print(
                    f"PBS worker {identity['worker_index']} exited with status {process.returncode}",
                    file=sys.stderr,
                    flush=True,
                )
                stopping = True
                worker_failed = True
                break
            time.sleep(0.2)

        terminate_identities(
            [identity for _, identity in children],
            grace_seconds,
            reverse_order=_requested_reverse_order(manifest),
        )
        for process, _identity in children:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        # A role worker is intended to be long lived. Any spontaneous child
        # exit, including status zero, is a role failure; controller-requested
        # termination is successful.
        return 1 if worker_failed else 0
    finally:
        terminate_identities(
            [identity for _, identity in children],
            grace_seconds,
            reverse_order=_requested_reverse_order(manifest),
        )
        try:
            current = _read_private_json(control_path)
        except FileNotFoundError:
            current = None
        if current and current.get("generation") == generation and current.get("supervisor") == supervisor_identity:
            control_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--host-index", type=int)
    parser.add_argument("--cleanup-host", type=Path)
    parser.add_argument("--grace-seconds", type=float, default=10.0)
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument("--generation")
    args = parser.parse_args(argv)
    if args.cleanup_host is not None:
        if args.manifest is not None or args.host_index is not None:
            parser.error("--cleanup-host cannot be combined with supervisor arguments")
        cleanup_host(
            args.cleanup_host,
            args.grace_seconds,
            reverse_order=args.reverse_order,
            expected_generation=args.generation,
        )
        return 0
    if args.manifest is None or args.host_index is None:
        parser.error("--manifest and --host-index are required")
    return supervise_host(args.manifest, args.host_index)


if __name__ == "__main__":
    raise SystemExit(main())
