#!/usr/bin/env python3
"""Tiny AReaL-discoverable worker for the live PBS scheduler gate.

This intentionally does not import Platoon or AReaL.  It implements the narrow
wire/filesystem surface used during worker discovery: one NFS ``ENTRY`` record
and an HTTP ``/health`` endpoint.  Running it through the production PBS role
launcher still exercises Singularity, rank/GPU assignment, explicit worker
indexes, and the complete remote teardown path without loading a model.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def durable_directory_write_probe(directory: Path, name: str) -> bool:
    """Prove a worker-visible directory supports durable create and unlink."""

    if not directory.is_absolute() or "/" in name or name in {"", ".", ".."}:
        raise RuntimeError(f"unsafe Gate 4 write-probe target: {directory}/{name}")
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    created = False
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        created = True
        try:
            payload = f"gate4-worker-write-probe pid={os.getpid()}\n".encode()
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise RuntimeError(f"short Gate 4 write probe in {directory}")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise RuntimeError(f"unsafe Gate 4 write-probe artifact in {directory}")
        os.unlink(name, dir_fd=directory_fd)
        created = False
        os.fsync(directory_fd)
        return True
    finally:
        if created:
            try:
                os.unlink(name, dir_fd=directory_fd)
                os.fsync(directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Publish a private JSON receipt without exposing a partial write."""

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
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
        temporary.unlink(missing_ok=True)


def atomic_text(path: Path, value: str) -> None:
    """Publish one AReaL NFS name-resolution ``ENTRY`` atomically."""

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def name_record_path(
    root: Path,
    registry_user: str,
    experiment: str,
    trial: str,
    role: str,
    worker_index: int,
) -> Path:
    """Return the exact path used by AReaL's NFS name resolver."""

    return root / registry_user / experiment / trial / "workers" / role / str(worker_index) / "ENTRY"


def visible_gpu_ids(value: str) -> list[int]:
    try:
        result = [int(part) for part in value.split(",") if part != ""]
    except ValueError as error:
        raise RuntimeError(f"invalid CUDA_VISIBLE_DEVICES={value!r}") from error
    if not result or len(set(result)) != len(result):
        raise RuntimeError(f"invalid CUDA_VISIBLE_DEVICES={value!r}")
    return result


def outward_ipv4(probe_host: str = "8.8.8.8", probe_port: int = 80) -> str:
    """Match AReaL's route-based fallback when container DNS is isolated."""

    hostname = socket.gethostname()
    try:
        candidates = socket.getaddrinfo(
            hostname,
            None,
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )
        for candidate in candidates:
            address = candidate[4][0]
            if not address.startswith("127."):
                return address
    except socket.gaierror:
        # ``--no-mount home,cwd,bind-paths`` deliberately hides inherited bind
        # paths, which can include host resolver files.  AReaL's gethostip uses
        # this same UDP route-selection fallback; connect sends no packet.
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect((probe_host, probe_port))
            address = probe.getsockname()[0]
    except OSError as error:
        raise RuntimeError(f"could not determine an outward IPv4 address for {hostname}") from error
    if not address or address.startswith("127."):
        raise RuntimeError(f"route selection returned an invalid IPv4 address: {address!r}")
    return address


class HealthHandler(BaseHTTPRequestHandler):
    server_version = "PlatoonGate4/1"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urllib.parse.urlsplit(self.path).path
        if path != "/health":
            self.send_error(404)
            return
        payload = json.dumps(self.server.health_payload, sort_keys=True).encode()  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--trial-name", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--name-resolve-type", required=True)
    parser.add_argument("--nfs-record-root", type=Path, required=True)
    parser.add_argument("--etcd3-addr")
    parser.add_argument("--fileroot")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        raise ValueError(f"unexpected worker arguments: {unknown}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.name_resolve_type != "nfs":
        raise RuntimeError(f"Gate 4 requires NFS name resolution, got {args.name_resolve_type!r}")

    environment_index = int(os.environ["PLATOON_WORKER_INDEX"])
    local_rank = int(os.environ["PLATOON_LOCAL_RANK"])
    environment_role = os.environ["PLATOON_ROLE"]
    role_generation = os.environ["PLATOON_ROLE_GENERATION"]
    if environment_index != args.worker_index or environment_role != args.role:
        raise RuntimeError(
            "worker identity disagreement: "
            f"args={args.role}/{args.worker_index}, env={environment_role}/{environment_index}"
        )

    visible = visible_gpu_ids(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    expected_gpu_count = int(os.environ["GATE4_EXPECTED_GPU_COUNT"])
    if len(visible) != expected_gpu_count:
        raise RuntimeError(
            f"{args.role}/{args.worker_index}: visible GPU ids {visible} do not match "
            f"expected count {expected_gpu_count}"
        )
    if expected_gpu_count == 1 and visible != [0]:
        raise RuntimeError(
            f"{args.role}/{args.worker_index}: UUID-isolated GPU is not container ordinal zero: {visible}"
        )
    if expected_gpu_count == 8 and visible != list(range(8)):
        raise RuntimeError(f"{args.role}/{args.worker_index}: full-node GPU map is {visible}")

    assigned_gpu_ids = visible_gpu_ids(os.environ.get("PLATOON_ASSIGNED_GPU_IDS", ""))
    expected_assigned_gpu_ids = [local_rank] if expected_gpu_count == 1 else list(range(8))
    if assigned_gpu_ids != expected_assigned_gpu_ids:
        raise RuntimeError(
            f"{args.role}/{args.worker_index}: physical GPU assignment {assigned_gpu_ids} "
            f"does not match {expected_assigned_gpu_ids}"
        )
    assigned_gpu_uuids = os.environ.get("PLATOON_ASSIGNED_GPU_UUIDS", "").split(",")
    if (
        len(assigned_gpu_uuids) != expected_gpu_count
        or len(set(assigned_gpu_uuids)) != expected_gpu_count
        or any(not value.startswith("GPU-") for value in assigned_gpu_uuids)
    ):
        raise RuntimeError(
            f"{args.role}/{args.worker_index}: invalid physical GPU UUID assignment {assigned_gpu_uuids}"
        )
    observed_gpu_uuids = [
        line.strip()
        for line in subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        ).stdout.splitlines()
        if line.strip()
    ]
    if observed_gpu_uuids != assigned_gpu_uuids:
        raise RuntimeError(
            f"{args.role}/{args.worker_index}: mounted GPU UUIDs {observed_gpu_uuids} "
            f"do not match assignment {assigned_gpu_uuids}"
        )

    # A safe canary is set only in the host controller.  Neither it nor real
    # credential-shaped variables may cross the launcher/container boundary.
    forbidden = (
        "WANDB_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "OPENAI_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
    )
    leaked = [name for name in forbidden if name in os.environ]
    if leaked:
        raise RuntimeError(f"credential-shaped host variables leaked into worker: {leaked}")

    import torch

    torch_gpu_count = torch.cuda.device_count()
    if not torch.cuda.is_available() or torch_gpu_count != expected_gpu_count:
        raise RuntimeError(
            f"{args.role}/{args.worker_index}: torch sees {torch_gpu_count} GPUs; expected {expected_gpu_count}"
        )
    gpu_names = [torch.cuda.get_device_name(index) for index in range(torch_gpu_count)]
    if any("H200" not in name for name in gpu_names):
        raise RuntimeError(f"{args.role}/{args.worker_index}: unexpected GPU names {gpu_names}")
    probe = torch.ones(1, device="cuda")
    if float(probe.item()) != 1.0:
        raise RuntimeError("CUDA probe returned an unexpected value")
    torch.cuda.synchronize()

    image_digest = os.environ["PLATOON_CONTAINER_IMAGE_DIGEST"]
    expected_digest = os.environ["GATE4_EXPECTED_SIF_DIGEST"]
    if image_digest != expected_digest:
        raise RuntimeError(f"worker image digest {image_digest!r} != {expected_digest!r}")

    node_local_root_value = os.environ.get("GATE4_NODE_LOCAL_ROOT", "")
    if not node_local_root_value or not os.path.isabs(node_local_root_value):
        raise RuntimeError("GATE4_NODE_LOCAL_ROOT must be an absolute path")
    node_local_root = Path(node_local_root_value)
    worker_home = node_local_root / "worker-home"
    worker_tmp = node_local_root / "worker-tmp"
    if os.environ.get("HOME") != str(worker_home):
        raise RuntimeError(f"worker HOME is not the attested node-local path: {os.environ.get('HOME')!r}")
    if os.environ.get("TMPDIR") != "/tmp":
        raise RuntimeError(f"worker TMPDIR is not the contained /tmp bind: {os.environ.get('TMPDIR')!r}")
    if not node_local_root.is_dir() or not worker_home.is_dir() or not worker_tmp.is_dir():
        raise RuntimeError("one or more initialized worker bind/preamble directories is absent")
    root_mode = node_local_root.stat().st_mode & 0o7777
    home_mode = worker_home.stat().st_mode & 0o7777
    tmp_mode = worker_tmp.stat().st_mode & 0o7777
    if (root_mode, home_mode, tmp_mode) != (0o700, 0o700, 0o1777):
        raise RuntimeError(
            "worker bind/preamble directory modes differ from the initializer contract: "
            f"root={root_mode:o} home={home_mode:o} tmp={tmp_mode:o}"
        )
    if not role_generation or any(character not in "0123456789abcdef" for character in role_generation):
        raise RuntimeError(f"unsafe role generation for write probes: {role_generation!r}")
    if not args.role or any(
        not (character.isalnum() or character in "._-") for character in args.role
    ):
        raise RuntimeError(f"unsafe role name for write probes: {args.role!r}")
    probe_name = f".gate4-write-probe-{args.role}-{args.worker_index}-{role_generation}"
    worker_home_write_probe = durable_directory_write_probe(worker_home, probe_name)
    worker_tmp_write_probe = durable_directory_write_probe(Path("/tmp"), probe_name)

    host = socket.gethostname()
    resolver_addresses: list[str] = []
    if os.environ.get("GATE4_REQUIRE_RESOLVER") == "1":
        for resolver_path in (Path("/etc/hosts"), Path("/etc/resolv.conf")):
            if not resolver_path.is_file() or not os.access(resolver_path, os.R_OK):
                raise RuntimeError(f"explicit resolver bind is not readable: {resolver_path}")
        resolver_addresses = sorted(
            {
                str(candidate[4][0])
                for candidate in socket.getaddrinfo(
                    host,
                    None,
                    socket.AF_INET,
                    socket.SOCK_DGRAM,
                )
                if candidate[4][0] and not str(candidate[4][0]).startswith("127.")
            }
        )
        if not resolver_addresses:
            raise RuntimeError(f"explicit resolver binds did not resolve {host!r}")
    server = ThreadingHTTPServer(("0.0.0.0", 0), HealthHandler)
    address = f"{outward_ipv4()}:{server.server_port}"
    server.health_payload = {  # type: ignore[attr-defined]
        "status": "ok",
        "role": args.role,
        "role_generation": role_generation,
        "resolver_addresses": resolver_addresses,
        "resolver_files": ["/etc/hosts", "/etc/resolv.conf"],
        "worker_index": args.worker_index,
    }
    receipt_root = Path(os.environ["GATE4_RECEIPT_ROOT"])
    receipt_path = receipt_root / args.trial_name / args.role / f"{args.worker_index}.json"
    registry_user = os.environ["GATE4_REGISTRY_USER"]
    entry_path = name_record_path(
        args.nfs_record_root,
        registry_user,
        args.experiment_name,
        args.trial_name,
        args.role,
        args.worker_index,
    )
    receipt: dict[str, Any] = {
        "address": address,
        "assigned_gpu_ids": assigned_gpu_ids,
        "assigned_gpu_uuids": assigned_gpu_uuids,
        "container_image": os.environ.get("PLATOON_CONTAINER_IMAGE", ""),
        "container_image_digest": image_digest,
        "cuda_visible_devices": visible,
        "gpu_names": gpu_names,
        "host": host,
        "local_rank": local_rank,
        "node_local_init_generation": os.environ.get("GATE4_NODE_LOCAL_INIT_GENERATION", ""),
        "node_local_root": str(node_local_root),
        "node_local_root_mode": root_mode,
        "observed_gpu_uuids": observed_gpu_uuids,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "role": args.role,
        "role_generation": role_generation,
        "resolver_addresses": resolver_addresses,
        "resolver_files": ["/etc/hosts", "/etc/resolv.conf"],
        "state": "ready",
        "torch_gpu_count": torch_gpu_count,
        "worker_home": str(worker_home),
        "worker_home_mode": home_mode,
        "worker_home_write_probe": worker_home_write_probe,
        "worker_index": args.worker_index,
        "worker_tmp": str(worker_tmp),
        "worker_tmp_mode": tmp_mode,
        "worker_tmp_write_probe": worker_tmp_write_probe,
    }

    stop_requested = False
    stop_signal: int | None = None

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested, stop_signal
        # Signal handlers may run again while Python is unwinding imported
        # libraries.  Record the first request and let the bounded server loop
        # reach its normal finally block; never interrupt ENTRY/receipt
        # publication with a second asynchronous exception.
        if not stop_requested:
            stop_signal = signum
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    atomic_text(entry_path, address)
    atomic_json(receipt_path, receipt)
    print(
        "gate4_worker_ready "
        f"role={args.role} worker_index={args.worker_index} local_rank={local_rank} "
        f"generation={role_generation} "
        f"host={host} visible={visible} torch_gpus={torch_gpu_count} "
        f"digest={image_digest} address={address}",
        flush=True,
    )

    fail_index_value = os.environ.get("GATE4_FAIL_WORKER_INDEX", "")
    fail_index = int(fail_index_value) if fail_index_value else None
    fail_at = time.monotonic() + float(os.environ.get("GATE4_FAIL_DELAY_SECONDS", "4"))
    server.timeout = 0.2
    exit_status = 0
    try:
        while not stop_requested:
            server.handle_request()
            if stop_requested:
                break
            if fail_index == args.worker_index and time.monotonic() >= fail_at:
                print(
                    f"gate4_deliberate_worker_failure role={args.role} worker_index={args.worker_index}",
                    file=sys.stderr,
                    flush=True,
                )
                exit_status = 23
                break
    except (KeyboardInterrupt, SystemExit):
        exit_status = 0
    finally:
        server.server_close()
        entry_self_removed = False
        try:
            if entry_path.read_text().strip() == address:
                entry_path.unlink()
                entry_self_removed = True
        except FileNotFoundError:
            pass
        receipt.update(
            {
                "entry_self_removed": entry_self_removed,
                "exit_status": exit_status,
                "state": "stopped" if exit_status == 0 else "failed",
                "stop_signal": stop_signal,
            }
        )
        atomic_json(receipt_path, receipt)
        print(
            f"gate4_worker_stopped role={args.role} worker_index={args.worker_index} "
            f"exit_status={exit_status} signal={stop_signal}",
            flush=True,
        )
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
