#!/usr/bin/env python3
"""Initialize and attest production worker bind sources on every PBS node."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCHEMA_VERSION = 1
HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
JOB_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
GENERATION_RE = re.compile(r"^[0-9a-f]{32}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_CONTAINER_RUNTIME_PATH = Path("/usr/bin/singularity")
EXPECTED_CONTAINER_RUNTIME_SHA256 = "085a757f172dcabda66b44d086913314d6aa574055a8fea90523be82b770cec6"
EXPECTED_CONTAINER_RUNTIME_VERSION = "4.4.1-1.el9"
TRUSTED_CONTAINER_RUNTIME_UIDS = {0, 65534}
RUNTIME_RECEIPT_KEYS = {
    "container_runtime_path",
    "container_runtime_sha256",
    "container_runtime_version",
    "container_runtime_uid",
    "container_runtime_mode",
}


class InitError(RuntimeError):
    pass


def _short_host(value: str) -> str:
    return value.split(".", 1)[0]


def _require_token(name: str, value: str, pattern: re.Pattern[str]) -> str:
    if not pattern.fullmatch(value):
        raise InitError(f"{name} has an invalid format")
    return value


def _lstat_plain_directory(path: Path, *, mode: int, create: bool = True) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        if not create:
            raise InitError(f"required directory is absent: {path}") from None
        try:
            path.mkdir(mode=mode)
        except FileExistsError:
            pass
        metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise InitError(f"path is not a plain directory: {path}")
    if metadata.st_uid != os.getuid():
        raise InitError(f"directory is not owned by the job uid: {path}")
    os.chmod(path, mode, follow_symlinks=False)
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != mode:
        raise InitError(f"directory mode could not be secured: {path}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _attest_container_runtime(args: argparse.Namespace) -> dict[str, object]:
    runtime = Path(args.container_runtime)
    if (
        runtime != EXPECTED_CONTAINER_RUNTIME_PATH
        or args.container_runtime_sha256 != EXPECTED_CONTAINER_RUNTIME_SHA256
        or args.container_runtime_version != EXPECTED_CONTAINER_RUNTIME_VERSION
    ):
        raise InitError("container runtime request differs from the v9 pinned identity")
    if not runtime.is_absolute() or runtime.resolve() != runtime:
        raise InitError("container runtime is not one direct absolute path")
    metadata = runtime.lstat()
    mode = stat.S_IMODE(metadata.st_mode)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid not in TRUSTED_CONTAINER_RUNTIME_UIDS
        or mode & 0o022
        or not mode & 0o111
    ):
        raise InitError("container runtime ownership or mode is not trusted")
    if _sha256(runtime) != EXPECTED_CONTAINER_RUNTIME_SHA256:
        raise InitError("container runtime stream hash differs from the v9 pin")
    try:
        completed = subprocess.run(
            [str(runtime), "version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
    except subprocess.TimeoutExpired as error:
        raise InitError("container runtime version check timed out") from error
    if completed.returncode != 0 or completed.stdout.strip() != EXPECTED_CONTAINER_RUNTIME_VERSION:
        raise InitError("container runtime version differs from the v9 pin")
    return {
        "container_runtime_path": str(runtime),
        "container_runtime_sha256": EXPECTED_CONTAINER_RUNTIME_SHA256,
        "container_runtime_version": EXPECTED_CONTAINER_RUNTIME_VERSION,
        "container_runtime_uid": metadata.st_uid,
        "container_runtime_mode": mode,
    }


def _stage_training_sif(source: Path, destination: Path, expected_sha256: str) -> None:
    if not source.is_absolute() or source.is_symlink() or not source.is_file():
        raise InitError(f"training SIF is not a plain absolute file: {source}")

    if os.path.lexists(destination):
        metadata = destination.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
            raise InitError(f"node-local training SIF is not a uid-owned plain file: {destination}")
        if _sha256(destination) == expected_sha256:
            os.chmod(destination, 0o400, follow_symlinks=False)
            return

    temporary_fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.partial.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(temporary_fd, "wb") as output, source.open("rb") as input_stream:
            shutil.copyfileobj(input_stream, output, length=8 * 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        if _sha256(temporary) != expected_sha256:
            raise InitError("node-local training SIF changed while staging")
        os.chmod(temporary, 0o400, follow_symlinks=False)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_private_json(path: Path, value: dict[str, object]) -> None:
    _lstat_plain_directory(path.parent, mode=0o700, create=False)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _validated_local_paths(expected_root: str, pbs_job_id: str) -> tuple[Path, Path]:
    localdir_value = os.environ.get("PBS_LOCALDIR", "")
    if not localdir_value or not os.path.isabs(localdir_value):
        raise InitError("PBS_LOCALDIR is absent or is not absolute")
    localdir_text = os.path.normpath(localdir_value)
    if localdir_text != localdir_value.rstrip("/"):
        raise InitError("PBS_LOCALDIR is not normalized")
    localdir = Path(localdir_text)
    try:
        local_metadata = localdir.lstat()
    except FileNotFoundError:
        raise InitError("PBS_LOCALDIR does not exist") from None
    if not stat.S_ISDIR(local_metadata.st_mode):
        raise InitError("PBS_LOCALDIR is not a plain directory")

    if not expected_root or not os.path.isabs(expected_root):
        raise InitError("expected node-local root is not absolute")
    normalized_root = os.path.normpath(expected_root)
    if normalized_root != expected_root.rstrip("/"):
        raise InitError("expected node-local root is not normalized")
    derived_root = str(localdir / "platoon" / pbs_job_id)
    if normalized_root != derived_root:
        raise InitError("node-local root does not match PBS_LOCALDIR/platoon/PBS_JOBID")
    return localdir, Path(normalized_root)


def initialize(args: argparse.Namespace) -> int:
    expected_host = _require_token("expected host", args.expected_host, HOST_RE)
    pbs_job_id = _require_token("PBS job id", args.pbs_job_id, JOB_RE)
    generation = _require_token("generation", args.generation, GENERATION_RE)
    if os.environ.get("PBS_JOBID") != pbs_job_id:
        raise InitError("remote PBS_JOBID does not match the requested job")
    try:
        task_index = int(args.task_index)
    except ValueError:
        raise InitError("task index is not an integer") from None
    if task_index < 0:
        raise InitError("task index is negative")

    observed_host = _short_host(socket.gethostname())
    if observed_host != _short_host(expected_host):
        raise InitError(f"pbsdsh landed on host {observed_host!r}, expected {_short_host(expected_host)!r}")
    localdir, root = _validated_local_paths(args.expected_root, pbs_job_id)
    runtime_evidence = _attest_container_runtime(args) if args.attest_container_runtime else {}

    platoon_parent = root.parent
    worker_home = root / "worker-home"
    worker_tmp = root / "worker-tmp"
    _lstat_plain_directory(platoon_parent, mode=0o700)
    _lstat_plain_directory(root, mode=0o700)
    _lstat_plain_directory(worker_home, mode=0o700)
    _lstat_plain_directory(worker_tmp, mode=0o1777)

    stage_training_sif = bool(args.stage_training_sif)
    local_training_sif: Path | None = None
    if stage_training_sif:
        expected_digest = _require_token("training SIF sha256", args.training_sif_sha256, SHA256_RE)
        images = root / "images"
        _lstat_plain_directory(images, mode=0o700)
        local_training_sif = images / "platoon-training.sif"
        _stage_training_sif(Path(args.training_sif), local_training_sif, expected_digest)

    # Recheck every bind/preamble source after optional staging, immediately
    # before publishing the durable shared-filesystem attestation.
    _lstat_plain_directory(root, mode=0o700, create=False)
    _lstat_plain_directory(worker_home, mode=0o700, create=False)
    _lstat_plain_directory(worker_tmp, mode=0o1777, create=False)
    receipt = Path(args.receipt)
    if not receipt.is_absolute() or receipt.name != f"{expected_host}.json":
        raise InitError("receipt path is not the exact expected per-host JSON path")
    receipt_value: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "pbs_job_id": pbs_job_id,
        "generation": generation,
        "expected_host": expected_host,
        "observed_host": observed_host,
        "task_index": task_index,
        "pbs_localdir": str(localdir),
        "node_local_root": str(root),
        "node_local_root_mode": 0o700,
        "worker_home": str(worker_home),
        "worker_home_mode": 0o700,
        "worker_tmp": str(worker_tmp),
        "worker_tmp_mode": 0o1777,
        "training_sif_staged": stage_training_sif,
        "training_sif_path": str(local_training_sif) if local_training_sif else "",
        "training_sif_sha256": args.training_sif_sha256 if stage_training_sif else "",
        "published_time_ns": time.time_ns(),
    }
    receipt_value.update(runtime_evidence)
    _atomic_private_json(receipt, receipt_value)
    print(f"node_local_ready host={expected_host} task_index={task_index}")
    return 0


def _load_private_json(path: Path) -> dict[str, object]:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise InitError(f"receipt is not a uid-owned plain file: {path}")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise InitError(f"receipt mode is not 0600: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InitError(f"receipt is unreadable or malformed: {path}: {error}") from None
    if not isinstance(value, dict):
        raise InitError(f"receipt is not a JSON object: {path}")
    return value


def _node_map(path: Path, expected_nodes: int) -> list[tuple[str, int]]:
    records: list[tuple[str, int]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw_line.split("\t")
        if len(fields) != 2:
            raise InitError(f"malformed node-map row {line_number}")
        host = _require_token("node-map host", fields[0], HOST_RE)
        try:
            task_index = int(fields[1])
        except ValueError:
            raise InitError(f"invalid node-map task index on row {line_number}") from None
        if task_index < 0:
            raise InitError(f"negative node-map task index on row {line_number}")
        records.append((host, task_index))
    if len(records) != expected_nodes:
        raise InitError(f"node map has {len(records)} rows; expected {expected_nodes}")
    if len({host for host, _index in records}) != len(records):
        raise InitError("node map contains duplicate hosts")
    if len({index for _host, index in records}) != len(records):
        raise InitError("node map contains duplicate task indexes")
    return records


def verify(args: argparse.Namespace) -> int:
    pbs_job_id = _require_token("PBS job id", args.pbs_job_id, JOB_RE)
    generation = _require_token("generation", args.generation, GENERATION_RE)
    expected_root = os.path.normpath(args.expected_root)
    if not os.path.isabs(args.expected_root) or expected_root != args.expected_root.rstrip("/"):
        raise InitError("expected node-local root is not an absolute normalized path")
    expected_nodes = int(args.expected_nodes)
    if expected_nodes < 1:
        raise InitError("expected node count must be positive")
    records = _node_map(Path(args.node_map), expected_nodes)
    receipt_dir = Path(args.receipt_dir)
    _lstat_plain_directory(receipt_dir, mode=0o700, create=False)
    expected_names = {f"{host}.json" for host, _index in records}
    actual_names = {entry.name for entry in receipt_dir.iterdir()}
    if actual_names != expected_names:
        raise InitError(
            f"receipt cardinality/identity mismatch expected={len(expected_names)} actual={len(actual_names)}"
        )

    stage_training_sif = bool(args.stage_training_sif)
    expected_digest = args.training_sif_sha256 if stage_training_sif else ""
    if stage_training_sif:
        _require_token("training SIF sha256", expected_digest, SHA256_RE)
    for host, task_index in records:
        value = _load_private_json(receipt_dir / f"{host}.json")
        expected = {
            "schema_version": SCHEMA_VERSION,
            "pbs_job_id": pbs_job_id,
            "generation": generation,
            "expected_host": host,
            "observed_host": _short_host(host),
            "task_index": task_index,
            "node_local_root": expected_root,
            "node_local_root_mode": 0o700,
            "worker_home": f"{expected_root}/worker-home",
            "worker_home_mode": 0o700,
            "worker_tmp": f"{expected_root}/worker-tmp",
            "worker_tmp_mode": 0o1777,
            "training_sif_staged": stage_training_sif,
            "training_sif_path": (f"{expected_root}/images/platoon-training.sif" if stage_training_sif else ""),
            "training_sif_sha256": expected_digest,
        }
        for key, expected_value in expected.items():
            if value.get(key) != expected_value:
                raise InitError(f"receipt field mismatch host={host} field={key}")
        runtime_keys = set(value) & RUNTIME_RECEIPT_KEYS
        if args.attest_container_runtime:
            if runtime_keys != RUNTIME_RECEIPT_KEYS:
                raise InitError(f"receipt lacks exact runtime evidence for host={host}")
            if (
                value.get("container_runtime_path") != str(EXPECTED_CONTAINER_RUNTIME_PATH)
                or value.get("container_runtime_sha256") != EXPECTED_CONTAINER_RUNTIME_SHA256
                or value.get("container_runtime_version") != EXPECTED_CONTAINER_RUNTIME_VERSION
                or value.get("container_runtime_uid") not in TRUSTED_CONTAINER_RUNTIME_UIDS
                or not isinstance(value.get("container_runtime_mode"), int)
                or int(value["container_runtime_mode"]) & 0o022
                or not int(value["container_runtime_mode"]) & 0o111
            ):
                raise InitError(f"receipt runtime evidence mismatch for host={host}")
        elif runtime_keys:
            raise InitError(f"legacy receipt unexpectedly contains runtime evidence for host={host}")
        pbs_localdir = value.get("pbs_localdir")
        if not isinstance(pbs_localdir, str) or expected_root != str(Path(pbs_localdir) / "platoon" / pbs_job_id):
            raise InitError(f"receipt has inconsistent PBS_LOCALDIR for host={host}")
        published = value.get("published_time_ns")
        if not isinstance(published, int) or published <= 0:
            raise InitError(f"receipt has invalid publication time for host={host}")

    print(f"node_local_initialization=ok nodes={expected_nodes} generation={generation}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init_parser = commands.add_parser("init")
    init_parser.add_argument("--expected-host", required=True)
    init_parser.add_argument("--task-index", required=True)
    init_parser.add_argument("--expected-root", required=True)
    init_parser.add_argument("--pbs-job-id", required=True)
    init_parser.add_argument("--generation", required=True)
    init_parser.add_argument("--receipt", required=True)
    init_parser.add_argument("--stage-training-sif", type=int, choices=(0, 1), required=True)
    init_parser.add_argument("--training-sif", required=True)
    init_parser.add_argument("--training-sif-sha256", required=True)
    init_parser.add_argument("--attest-container-runtime", type=int, choices=(0, 1), default=0)
    init_parser.add_argument("--container-runtime", default="")
    init_parser.add_argument("--container-runtime-sha256", default="")
    init_parser.add_argument("--container-runtime-version", default="")
    init_parser.set_defaults(handler=initialize)

    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--node-map", required=True)
    verify_parser.add_argument("--receipt-dir", required=True)
    verify_parser.add_argument("--expected-nodes", required=True)
    verify_parser.add_argument("--expected-root", required=True)
    verify_parser.add_argument("--pbs-job-id", required=True)
    verify_parser.add_argument("--generation", required=True)
    verify_parser.add_argument("--stage-training-sif", type=int, choices=(0, 1), required=True)
    verify_parser.add_argument("--training-sif-sha256", required=True)
    verify_parser.add_argument("--attest-container-runtime", type=int, choices=(0, 1), default=0)
    verify_parser.add_argument("--container-runtime", default="")
    verify_parser.add_argument("--container-runtime-sha256", default="")
    verify_parser.add_argument("--container-runtime-version", default="")
    verify_parser.set_defaults(handler=verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (InitError, OSError, ValueError) as error:
        print(f"node-local-init ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
