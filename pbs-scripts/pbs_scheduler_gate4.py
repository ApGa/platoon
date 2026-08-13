#!/usr/bin/env python3
"""Live two-node gate for Platoon's preallocated PBS scheduler.

The immutable production AReaL environment is a separate migration gate.  This
script therefore supplies only the small upstream scheduler API surface needed
to load the real Platoon ``PreallocatedPBSScheduler`` source.  Workers implement
the exact AReaL NFS discovery record and ``/health`` protocol, but deliberately
do not import the full AReaL RPC server or load a model.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import types
import urllib.request
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclasses.dataclass
class GateWorker:
    id: str
    ip: str
    worker_ports: list[str] = dataclasses.field(default_factory=list)
    engine_ports: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class GateJob:
    role: str
    replicas: int
    tasks: list[Any]
    scheduling_strategy: Any


class GateSchedulingStrategyType(str, enum.Enum):
    separation = "separation"
    colocation = "colocation"


@dataclasses.dataclass
class GateSchedulingSpec:
    cpu: int = 8
    gpu: int = 0
    mem: int = 32
    port_count: int = 1
    image: str = ""
    task_type: str = "worker"
    env_vars: dict[str, str] = dataclasses.field(default_factory=dict)
    cmd: str | None = None
    srun_additional_args: str = "--unbuffered"
    additional_bash_cmds: list[str] | None = None
    container_type: str = "none"
    mount: str = ""
    nodelist: str | None = None
    exclude: str | None = None
    ray_placement_strategy: str = "shared"


@dataclasses.dataclass
class GateWorkerInfo:
    worker: GateWorker
    role: str
    slurm_job_id: int
    task_index: int
    discovered: bool = False
    spec: GateSchedulingSpec | None = None
    node: str | None = None


class GateWorkerError(Exception):
    pass


class GateWorkerCreationError(GateWorkerError):
    pass


class GateWorkerFailedError(GateWorkerError):
    pass


class GateWorkerNotFoundError(GateWorkerError):
    pass


class GateEngineCreationError(GateWorkerError):
    def __init__(self, worker_id: str, reason: str, status_code: Any = None):
        self.worker_id = worker_id
        self.reason = reason
        self.status_code = status_code
        super().__init__(reason)


def short_hostname(value: str) -> str:
    return value.split(".", 1)[0]


def name_record_path(
    root: Path,
    registry_user: str,
    experiment: str,
    trial: str,
    role: str,
    worker_index: int,
) -> Path:
    return root / registry_user / experiment / trial / "workers" / role / str(worker_index) / "ENTRY"


class GateSlurmScheduler:
    """Minimal upstream surface for loading the real Platoon PBS adapter."""

    def __init__(
        self,
        n_gpus_per_node: int = 8,
        experiment_name: str | None = None,
        trial_name: str | None = None,
        fileroot: str | None = None,
        name_resolve_type: str = "nfs",
        nfs_record_root: str = "/tmp/areal/name_resolve",
        etcd3_addr: str = "localhost:2379",
        startup_timeout: float = 120,
        health_check_interval: float = 0.2,
        exp_config: Any = None,
        **_kwargs: Any,
    ):
        if not experiment_name or not trial_name or not fileroot:
            raise ValueError("experiment_name, trial_name, and fileroot are required")
        self._n_gpus_per_node = n_gpus_per_node
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.fileroot = fileroot
        self.name_resolve_config = types.SimpleNamespace(
            type=name_resolve_type,
            nfs_record_root=nfs_record_root,
            etcd3_addr=etcd3_addr,
        )
        self.startup_timeout = startup_timeout
        self.health_check_interval = health_check_interval
        self.exp_config = exp_config
        self.enable_tms_offload = False
        self.cluster_name = "abci-gate4"
        self._workers: dict[str, list[GateWorkerInfo]] = {}
        self._jobs: dict[str, int] = {}
        self._job_status_cache: dict[Any, Any] = {}
        self._colocated_roles: dict[str, str] = {}
        trial_root = Path(nfs_record_root) / os.environ["GATE4_REGISTRY_USER"] / experiment_name / trial_name
        shutil.rmtree(trial_root, ignore_errors=True)

    @property
    def n_gpus_per_node(self) -> int:
        return self._n_gpus_per_node

    def _prepare_worker_specs(
        self,
        role: str,
        num_workers: int,
        schedulings: list[GateSchedulingSpec] | None,
    ) -> list[GateSchedulingSpec]:
        if not schedulings:
            raise ValueError(f"No scheduling specs provided for role {role!r}")
        if len(schedulings) == 1:
            return schedulings * num_workers
        if len(schedulings) == num_workers:
            return list(schedulings)
        raise ValueError(f"invalid scheduling count for {role}: {len(schedulings)}")

    def _log_root(self) -> Path:
        root = Path(self.fileroot) / "logs" / self.experiment_name / self.trial_name
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        return root

    def _log_path_of(self, role: str) -> str:
        return str(self._log_root() / f"{role}.log")

    def _merged_log_path(self) -> str:
        return str(self._log_root() / "merged.log")

    def _read_log_tail(self, role: str, lines: int = 80) -> str:
        try:
            return "".join(Path(self._log_path_of(role)).read_text().splitlines(keepends=True)[-lines:])
        except OSError as error:
            return f"could not read role log: {error}"

    async def create_engine(self, *_args: Any, **_kwargs: Any) -> Any:
        raise NotImplementedError("Gate 4 does not create engines")

    def _configure_worker(self, _worker: GateWorkerInfo, _rank: int) -> None:
        return

    def get_workers(self, role: str, timeout: float | None = None) -> list[GateWorker]:
        if role not in self._workers:
            raise GateWorkerNotFoundError(role)
        deadline = time.monotonic() + (timeout or self.startup_timeout)
        workers = self._workers[role]
        while time.monotonic() < deadline:
            self._check_job_status(role)
            for worker_info in workers:
                if worker_info.discovered:
                    continue
                entry = name_record_path(
                    Path(self.name_resolve_config.nfs_record_root),
                    os.environ["GATE4_REGISTRY_USER"],
                    self.experiment_name,
                    self.trial_name,
                    role,
                    worker_info.task_index,
                )
                try:
                    address = entry.read_text().strip()
                    host, port_text = address.rsplit(":", 1)
                    request = urllib.request.Request(f"http://{host}:{int(port_text)}/health")
                    with urllib.request.urlopen(request, timeout=2) as response:
                        payload = json.loads(response.read())
                    if response.status != 200:
                        continue
                    if payload.get("role") != role or payload.get("worker_index") != worker_info.task_index:
                        raise RuntimeError(f"health identity mismatch for {role}/{worker_info.task_index}: {payload}")
                except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
                    continue
                worker_info.worker.ip = host
                worker_info.worker.worker_ports = [port_text]
                worker_info.discovered = True
            if all(worker.discovered for worker in workers):
                return [worker.worker for worker in workers]
            time.sleep(self.health_check_interval)
        pending = [worker.worker.id for worker in workers if not worker.discovered]
        raise TimeoutError(f"worker discovery timed out for {role}: {pending}")


def _module(name: str, *, package_path: Path | None = None) -> types.ModuleType:
    value = types.ModuleType(name)
    if package_path is not None:
        value.__path__ = [str(package_path)]  # type: ignore[attr-defined]
    sys.modules[name] = value
    return value


def _load_source(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_real_pbs_scheduler(repo_root: Path) -> types.ModuleType:
    """Load the production PBS adapter while replacing only upstream imports."""

    areal = _module("areal")
    api = _module("areal.api")
    api.Job = GateJob  # type: ignore[attr-defined]
    api.Worker = GateWorker  # type: ignore[attr-defined]
    areal.api = api  # type: ignore[attr-defined]

    cli_args = _module("areal.api.cli_args")
    cli_args.SchedulingSpec = GateSchedulingSpec  # type: ignore[attr-defined]
    cli_args.SchedulingStrategyType = GateSchedulingStrategyType  # type: ignore[attr-defined]

    _module("areal.infra")
    _module("areal.infra.scheduler")
    exceptions = _module("areal.infra.scheduler.exceptions")
    exceptions.EngineCreationError = GateEngineCreationError  # type: ignore[attr-defined]
    exceptions.WorkerCreationError = GateWorkerCreationError  # type: ignore[attr-defined]
    exceptions.WorkerFailedError = GateWorkerFailedError  # type: ignore[attr-defined]
    exceptions.WorkerNotFoundError = GateWorkerNotFoundError  # type: ignore[attr-defined]
    slurm = _module("areal.infra.scheduler.slurm")
    slurm.SlurmScheduler = GateSlurmScheduler  # type: ignore[attr-defined]
    slurm.SlurmWorkerInfo = GateWorkerInfo  # type: ignore[attr-defined]
    _module("areal.infra.utils")
    proc = _module("areal.infra.utils.proc")
    proc.build_streaming_log_cmd = lambda command, *_args, **_kwargs: command  # type: ignore[attr-defined]
    utils = _module("areal.utils")
    utils.logging = logging  # type: ignore[attr-defined]

    platoon_root = repo_root / "platoon"
    _module("platoon", package_path=platoon_root)
    _module("platoon.train", package_path=platoon_root / "train")
    _module("platoon.train.areal", package_path=platoon_root / "train" / "areal")
    _load_source(
        "platoon.train.areal.preallocated_slurm",
        platoon_root / "train" / "areal" / "preallocated_slurm.py",
    )
    return _load_source(
        "platoon.train.areal.preallocated_pbs",
        platoon_root / "train" / "areal" / "preallocated_pbs.py",
    )


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def read_private_json(path: Path) -> dict[str, Any]:
    file_stat = path.stat()
    if file_stat.st_uid != os.getuid() or stat.S_IMODE(file_stat.st_mode) & 0o077:
        raise PermissionError(f"refusing non-private Gate 4 record: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def pbs_nodes(module: types.ModuleType) -> list[Any]:
    nodefile = Path(os.environ["PBS_NODEFILE"])
    nodes = module.parse_pbs_nodefile(nodefile)
    raw = [line.strip() for line in nodefile.read_text().splitlines() if line.strip()]
    counts = {node.hostname: raw.count(node.hostname) for node in nodes}
    if len(nodes) != 2 or any(count != 8 for count in counts.values()):
        raise RuntimeError(f"Gate 4 requires two ordered 8-slot nodes, got nodes={nodes}, counts={counts}")
    return nodes


def configure_runtime(repo_root: Path, state_root: Path, image: Path) -> None:
    state_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(state_root, 0o700)
    node_local_root_value = os.environ.get("GATE4_NODE_LOCAL_ROOT", "")
    if not node_local_root_value or not Path(node_local_root_value).is_absolute():
        raise RuntimeError("Gate 4 requires an absolute initialized node-local root")
    node_local_root = Path(node_local_root_value)
    singularity_binary = os.environ.get("GATE4_SINGULARITY_BIN") or shutil.which("singularity")
    if not singularity_binary or not Path(singularity_binary).is_absolute():
        raise RuntimeError("Gate 4 requires the absolute Singularity binary selected by the site module")
    os.environ.update(
        {
            "PLATOON_AREAL_PBS_CONTAINER_RUNTIME": singularity_binary,
            "PLATOON_AREAL_PBS_CONTAINER_IMAGE": str(image),
            "PLATOON_AREAL_PBS_CONTAINER_BINDS": (
                f"{repo_root}:{repo_root}:ro {state_root}:{state_root}:rw "
                f"{node_local_root}:{node_local_root}:rw "
                "/etc/hosts:/etc/hosts:ro /etc/resolv.conf:/etc/resolv.conf:ro"
            ),
            "PLATOON_AREAL_PBS_CONTAINER_ARGS": (
                "--containall --no-home --no-mount home,cwd,bind-paths "
                f"--bind {node_local_root}/worker-tmp:/tmp:rw"
            ),
            "PLATOON_AREAL_PBS_CONTAINER_WORKDIR": str(repo_root),
            "PLATOON_AREAL_PBS_GPU_VISIBILITY_MODE": "nvidia-container-cli",
            "PLATOON_AREAL_PBS_NVIDIA_SMI_BIN": "/usr/bin/nvidia-smi",
            "PLATOON_AREAL_PBS_HOST_PYTHON": "/usr/bin/python3",
            "PLATOON_AREAL_PBS_PBSDSH_BIN": "/opt/pbs/bin/pbsdsh",
            "PLATOON_AREAL_PBS_STATE_DIR": str(state_root / "scheduler-state"),
            "PLATOON_AREAL_PBS_TERMINATION_GRACE_SECONDS": "5",
            "PLATOON_AREAL_PBS_LAUNCHER_STOP_TIMEOUT": "40",
            "PLATOON_AREAL_PBS_VALIDATE_NODEFILE_SLOTS": "1",
            "PLATOON_AREAL_PBS_WORKER_PREAMBLE": (
                "set -euo pipefail\n"
                f"export HOME={node_local_root}/worker-home\n"
                "export TMPDIR=/tmp"
            ),
            # Safe, non-secret values verify that the host launcher denylist is
            # effective.  The worker fails if either value becomes visible.
            "WANDB_API_KEY": "gate4-non-secret-canary",
            "HF_TOKEN": "gate4-non-secret-canary",
        }
    )


def validate_node_local_init_receipts(nodes: list[Any]) -> list[dict[str, Any]]:
    receipt_root_value = os.environ.get("GATE4_NODE_LOCAL_RECEIPT_DIR", "")
    expected_root = os.environ.get("GATE4_NODE_LOCAL_ROOT", "")
    generation = os.environ.get("GATE4_NODE_LOCAL_INIT_GENERATION", "")
    if not receipt_root_value or not expected_root or not generation:
        raise AssertionError("Gate 4 node-local initialization environment is incomplete")
    receipt_root = Path(receipt_root_value)
    expected_names = {f"{node.hostname}.json" for node in nodes}
    actual_names = {path.name for path in receipt_root.iterdir()}
    if actual_names != expected_names:
        raise AssertionError(
            f"node-local receipt identity/cardinality mismatch: {actual_names} != {expected_names}"
        )
    receipts: list[dict[str, Any]] = []
    for node in nodes:
        receipt = read_private_json(receipt_root / f"{node.hostname}.json")
        expected = {
            "schema_version": 1,
            "pbs_job_id": os.environ["PBS_JOBID"],
            "generation": generation,
            "expected_host": node.hostname,
            "observed_host": short_hostname(node.hostname),
            "task_index": node.vnode_index,
            "node_local_root": expected_root,
            "node_local_root_mode": 0o700,
            "worker_home": f"{expected_root}/worker-home",
            "worker_home_mode": 0o700,
            "worker_tmp": f"{expected_root}/worker-tmp",
            "worker_tmp_mode": 0o1777,
            "training_sif_staged": False,
            "training_sif_path": "",
            "training_sif_sha256": "",
        }
        mismatches = {
            key: (receipt.get(key), value)
            for key, value in expected.items()
            if receipt.get(key) != value
        }
        if mismatches:
            raise AssertionError(
                f"invalid node-local initialization receipt for {node.hostname}: {mismatches}"
            )
        receipts.append(receipt)
    return receipts


def separated_job(role: str, replicas: int, spec: GateSchedulingSpec) -> GateJob:
    return GateJob(
        role=role,
        replicas=replicas,
        tasks=[spec],
        scheduling_strategy=types.SimpleNamespace(type="separation", target=None),
    )


def worker_spec(
    repo_root: Path,
    state_root: Path,
    expected_digest: str,
    token: str,
    gpu_count: int,
    *,
    fail_worker_index: int | None = None,
) -> GateSchedulingSpec:
    environment = {
        "GATE4_EXPECTED_GPU_COUNT": str(gpu_count),
        "GATE4_EXPECTED_SIF_DIGEST": expected_digest,
        "GATE4_NODE_LOCAL_INIT_GENERATION": os.environ["GATE4_NODE_LOCAL_INIT_GENERATION"],
        "GATE4_NODE_LOCAL_ROOT": os.environ["GATE4_NODE_LOCAL_ROOT"],
        "GATE4_REQUIRE_RESOLVER": "1",
        "GATE4_RECEIPT_ROOT": str(state_root / "receipts"),
        "GATE4_REGISTRY_USER": os.environ["GATE4_REGISTRY_USER"],
        "GATE4_TOKEN": token,
        "PYTHONUNBUFFERED": "1",
    }
    if fail_worker_index is not None:
        environment.update(
            {
                "GATE4_FAIL_WORKER_INDEX": str(fail_worker_index),
                "GATE4_FAIL_DELAY_SECONDS": "8",
            }
        )
    return GateSchedulingSpec(
        cpu=2 if gpu_count == 1 else 8,
        gpu=gpu_count,
        mem=4,
        port_count=1,
        env_vars=environment,
        cmd=f"python3 {repo_root / 'pbs-scripts' / 'pbs_scheduler_gate4_worker.py'}",
    )


def make_scheduler(
    module: types.ModuleType,
    state_root: Path,
    trial: str,
) -> Any:
    return module.PreallocatedPBSScheduler(
        n_gpus_per_node=8,
        experiment_name="pbs-scheduler-gate4",
        trial_name=trial,
        fileroot=str(state_root),
        name_resolve_type="nfs",
        nfs_record_root=str(state_root / "name-resolve"),
        startup_timeout=120,
        health_check_interval=0.2,
    )


def role_paths(scheduler: Any, role: str) -> dict[str, str]:
    return {
        "manifest": str(scheduler._role_manifests[role]),
        "status": str(scheduler._role_status_paths[role]),
        "termination": str(scheduler._role_termination_paths[role]),
    }


def validate_role_manifest(
    path: Path,
    *,
    role: str,
    expected_host: str,
    expected_vnode: int,
    expected_digest: str,
    expected_assignments: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest = read_private_json(path)
    if manifest.get("role") != role or manifest.get("container_image_digest") != expected_digest:
        raise AssertionError(f"manifest identity mismatch: {manifest}")
    hosts = manifest.get("hosts")
    if not isinstance(hosts, list) or len(hosts) != 1:
        raise AssertionError(f"{role} did not reserve exactly one node: {hosts}")
    host = hosts[0]
    if short_hostname(str(host.get("hostname"))) != short_hostname(expected_host):
        raise AssertionError(f"{role} host mismatch: {host}")
    if int(host.get("vnode_index")) != expected_vnode or host.get("workers") != expected_assignments:
        raise AssertionError(f"{role} assignment mismatch: {host}")
    command = [str(value) for value in manifest.get("worker_command", [])]
    required = {
        "--cleanenv",
        "--containall",
        "--no-home",
        "--no-mount",
        "home,cwd,bind-paths",
        "--nv",
    }
    if not required.issubset(command):
        raise AssertionError(f"{role} container command lacks isolation flags: {command}")
    if manifest.get("gpu_visibility") != {
        "mode": "nvidia-container-cli",
        "nvidia_smi_binary": "/usr/bin/nvidia-smi",
    }:
        raise AssertionError(f"{role} manifest lacks explicit nvidia-container-cli UUID isolation")
    if manifest.get("termination_reverse_order_default") is not True:
        raise AssertionError(f"{role} manifest does not default to rank-zero-last teardown")
    if not any("--worker-index" in value for value in command):
        raise AssertionError(f"{role} worker command lacks explicit worker index: {command}")
    for resolver_bind in (
        "/etc/hosts:/etc/hosts:ro",
        "/etc/resolv.conf:/etc/resolv.conf:ro",
    ):
        if resolver_bind not in command:
            raise AssertionError(f"{role} worker command lacks resolver bind {resolver_bind}: {command}")
    discovery = manifest.get("discovery")
    if not isinstance(discovery, dict):
        raise AssertionError(f"{role} manifest lacks discovery ownership: {manifest}")
    role_root = Path(str(discovery.get("role_root", "")))
    expected_entries = [str(role_root / str(index) / "ENTRY") for index in range(len(expected_assignments))]
    lock_identity = discovery.get("lock_identity")
    if not isinstance(lock_identity, dict):
        raise AssertionError(f"{role} manifest lacks canonical lock identity: {discovery}")
    lock_root = (
        Path(str(lock_identity.get("record_root", "")))
        / str(lock_identity.get("registry_user", ""))
        / ".pbs-prealloc-locks"
    )
    trial_digest = hashlib.sha256(json.dumps(lock_identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    role_identity = {**lock_identity, "role": role}
    role_digest = hashlib.sha256(json.dumps(role_identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    expected_constructor_lock = lock_root / f"constructor-{trial_digest}.lock"
    expected_trial_lock = lock_root / f"trial-{trial_digest}.lock"
    expected_lock = lock_root / f"role-{role_digest}.lock"
    if (
        role_root.name != role
        or discovery.get("entry_paths") != expected_entries
        or discovery.get("constructor_lifecycle_lock_path") != str(expected_constructor_lock)
        or discovery.get("lifecycle_lock_path") != str(expected_lock)
        or discovery.get("trial_lifecycle_lock_path") != str(expected_trial_lock)
    ):
        raise AssertionError(f"{role} discovery ownership is not exact: {discovery}")
    return manifest


def validate_lifecycle_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    discovery = manifest["discovery"]
    identity = discovery["lock_identity"]
    expected = {
        "constructor": {
            "version": 1,
            "kind": "pbs-constructor-lifecycle",
            "identity": identity,
        },
        "trial": {
            "version": 1,
            "kind": "pbs-trial-lifecycle",
            "identity": identity,
        },
        "role": {
            "version": 1,
            "kind": "pbs-role-lifecycle",
            "identity": identity,
            "role": manifest["role"],
            "generation": manifest["generation"],
        },
    }
    observed: dict[str, Any] = {}
    for label, field in (
        ("constructor", "constructor_lifecycle_lock_path"),
        ("trial", "trial_lifecycle_lock_path"),
        ("role", "lifecycle_lock_path"),
    ):
        path = Path(discovery[field])
        directory_stat = path.parent.lstat()
        file_stat = path.lstat()
        if (
            not stat.S_ISDIR(directory_stat.st_mode)
            or directory_stat.st_uid != os.getuid()
            or stat.S_IMODE(directory_stat.st_mode) & 0o077
        ):
            raise AssertionError(f"unsafe lifecycle lock directory: {path.parent}")
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_uid != os.getuid()
            or stat.S_IMODE(file_stat.st_mode) & 0o077
        ):
            raise AssertionError(f"unsafe lifecycle lock (including symlink): {path}")
        value = read_private_json(path)
        if value != expected[label]:
            raise AssertionError(f"{label} lifecycle metadata mismatch: {value}")
        observed[label] = {
            "path": str(path),
            "mode": oct(stat.S_IMODE(file_stat.st_mode)),
            "metadata": value,
        }
    return observed


def receipt_path(state_root: Path, trial: str, role: str, index: int) -> Path:
    return state_root / "receipts" / trial / role / f"{index}.json"


def validate_ready_receipts(
    state_root: Path,
    trial: str,
    role: str,
    workers: list[GateWorker],
    expected_host: str,
    expected_digest: str,
    gpu_count: int,
    expected_generation: str | None = None,
) -> list[dict[str, Any]]:
    receipts = []
    for index, worker in enumerate(workers):
        receipt = read_private_json(receipt_path(state_root, trial, role, index))
        expected_visible = [0] if gpu_count == 1 else list(range(8))
        expected_assigned_ids = [index] if gpu_count == 1 else list(range(8))
        assigned_uuids = receipt.get("assigned_gpu_uuids")
        if (
            receipt.get("state") != "ready"
            or receipt.get("role") != role
            or receipt.get("worker_index") != index
            or receipt.get("local_rank") != (index if gpu_count == 1 else 0)
            or receipt.get("cuda_visible_devices") != expected_visible
            or receipt.get("assigned_gpu_ids") != expected_assigned_ids
            or not isinstance(assigned_uuids, list)
            or len(assigned_uuids) != gpu_count
            or len(set(assigned_uuids)) != gpu_count
            or receipt.get("observed_gpu_uuids") != assigned_uuids
            or receipt.get("torch_gpu_count") != gpu_count
            or receipt.get("container_image_digest") != expected_digest
            or receipt.get("node_local_init_generation")
            != os.environ.get("GATE4_NODE_LOCAL_INIT_GENERATION")
            or receipt.get("node_local_root") != os.environ.get("GATE4_NODE_LOCAL_ROOT")
            or receipt.get("node_local_root_mode") != 0o700
            or receipt.get("worker_home")
            != f"{os.environ.get('GATE4_NODE_LOCAL_ROOT')}/worker-home"
            or receipt.get("worker_home_mode") != 0o700
            or receipt.get("worker_home_write_probe") is not True
            or receipt.get("worker_tmp")
            != f"{os.environ.get('GATE4_NODE_LOCAL_ROOT')}/worker-tmp"
            or receipt.get("worker_tmp_mode") != 0o1777
            or receipt.get("worker_tmp_write_probe") is not True
            or short_hostname(str(receipt.get("host"))) != short_hostname(expected_host)
            or receipt.get("resolver_files") != ["/etc/hosts", "/etc/resolv.conf"]
            or not receipt.get("resolver_addresses")
            or (expected_generation is not None and receipt.get("role_generation") != expected_generation)
        ):
            raise AssertionError(f"invalid ready receipt for {role}/{index}: {receipt}")
        if not all("H200" in str(name) for name in receipt.get("gpu_names", [])):
            raise AssertionError(f"non-H200 receipt for {role}/{index}: {receipt}")
        expected_address = f"{worker.ip}:{worker.worker_ports[0]}"
        if receipt.get("address") != expected_address:
            raise AssertionError(f"discovery address mismatch for {role}/{index}: {receipt}")
        receipts.append(receipt)
    all_uuids = [uuid for receipt in receipts for uuid in receipt["assigned_gpu_uuids"]]
    if len(set(all_uuids)) != len(all_uuids):
        raise AssertionError(f"{role} reused a physical H200 UUID: {all_uuids}")
    return receipts


def wait_for_receipts(
    state_root: Path,
    trial: str,
    role: str,
    count: int,
    timeout: float = 120,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            values = [read_private_json(receipt_path(state_root, trial, role, index)) for index in range(count)]
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.2)
            continue
        if len(values) == count:
            return values
    raise TimeoutError(f"timed out waiting for {count} {role} receipts")


def wait_for_launcher_status(path: Path, timeout: float = 90) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = read_private_json(path)
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.2)
            continue
        if value.get("complete"):
            return value
        time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for launcher status: {path}")


def assert_clean_status(
    value: dict[str, Any],
    *,
    expect_failure: bool = False,
    expected_discovery_records: int | None = None,
) -> None:
    if not value.get("complete") or value.get("cleanup_failures") != []:
        raise AssertionError(f"launcher did not confirm exact cleanup: {value}")
    primary = int(value.get("primary_status", 0))
    if expect_failure and primary == 0:
        raise AssertionError(f"deliberate failure was not observed: {value}")
    if not expect_failure and primary != 0:
        raise AssertionError(f"launcher reported unexpected worker failure: {value}")
    if value.get("stale_discovery_records_removed") != 0:
        raise AssertionError(f"launcher found unexpected stale discovery records: {value}")
    if expected_discovery_records is not None:
        if value.get("discovery_records_removed") != expected_discovery_records:
            raise AssertionError(f"launcher did not remove its exact discovery records: {value}")


def assert_cleanup_union(
    status: dict[str, Any],
    receipts: list[dict[str, Any]],
    *,
    expected_generation: str,
    expected_entry_paths: list[Path],
    expected_role_root: Path,
    expect_failure: bool = False,
) -> dict[str, Any]:
    """Prove exact cleanup while retaining best-effort remover attribution."""

    assert_clean_status(status, expect_failure=expect_failure)
    expected_records = len(expected_entry_paths)
    if expected_records <= 0 or len(set(expected_entry_paths)) != expected_records:
        raise AssertionError(f"invalid exact ENTRY path set: {expected_entry_paths}")

    remaining_entries: list[str] = []
    for path in expected_entry_paths:
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        remaining_entries.append(str(path))
    try:
        expected_role_root.lstat()
    except FileNotFoundError:
        role_root_remains = False
    else:
        role_root_remains = True
    if remaining_entries or role_root_remains:
        raise AssertionError(
            "exact discovery cleanup postcondition failed: "
            f"entries={remaining_entries}, role_root_remains={role_root_remains}, "
            f"role_root={expected_role_root}"
        )

    if len(receipts) != expected_records:
        raise AssertionError(
            f"cleanup receipt count {len(receipts)} does not match {expected_records}"
        )
    receipt_indices: list[int] = []
    receipt_attributed: list[int] = []
    for receipt in receipts:
        if receipt.get("role_generation") != expected_generation:
            raise AssertionError(f"cleanup receipt belongs to another generation: {receipt}")
        worker_index = receipt.get("worker_index")
        if (
            isinstance(worker_index, bool)
            or not isinstance(worker_index, int)
            or worker_index < 0
            or worker_index >= expected_records
        ):
            raise AssertionError(f"cleanup receipt worker index is out of bounds: {receipt}")
        receipt_indices.append(worker_index)
        if receipt.get("entry_self_removed") is True:
            receipt_attributed.append(worker_index)
    if len(set(receipt_indices)) != expected_records:
        raise AssertionError(f"cleanup receipt worker indices are not unique: {receipt_indices}")

    launcher_removed = status.get("discovery_records_removed")
    if (
        isinstance(launcher_removed, bool)
        or not isinstance(launcher_removed, int)
        or launcher_removed < 0
        or launcher_removed > expected_records
    ):
        raise AssertionError(f"launcher discovery removal count is out of bounds: {status}")
    receipt_attributed.sort()
    attributed_total = launcher_removed + len(receipt_attributed)
    if attributed_total > expected_records:
        raise AssertionError(
            "discovery cleanup attribution is double-counted: "
            f"launcher={launcher_removed}, receipt_workers={receipt_attributed}, "
            f"expected={expected_records}, status={status}"
        )
    return {
        "exact_entry_paths_absent": True,
        "exact_role_root_absent": True,
        "expected_records": expected_records,
        "launcher_removed_records": launcher_removed,
        "receipt_attributed_worker_indices": receipt_attributed,
        # A worker can unlink ENTRY and then be killed before replacing its
        # receipt. Exact absence under the lifecycle lock is authoritative;
        # this residual records that unavoidable attribution window.
        "unattributed_worker_removals": expected_records - attributed_total,
    }


def assert_worker_health(
    worker: GateWorker,
    *,
    role: str,
    worker_index: int,
    generation: str,
) -> dict[str, Any]:
    request = urllib.request.Request(f"http://{worker.ip}:{int(worker.worker_ports[0])}/health")
    with urllib.request.urlopen(request, timeout=3) as response:
        payload = json.loads(response.read())
    if (
        response.status != 200
        or payload.get("status") != "ok"
        or payload.get("role") != role
        or payload.get("worker_index") != worker_index
        or payload.get("role_generation") != generation
    ):
        raise AssertionError(f"worker health identity mismatch: {payload}")
    return payload


def find_survivors(
    token: str,
    marker: str,
    *,
    proc_root: Path = Path("/proc"),
    own_pid: int | None = None,
) -> list[dict[str, Any]]:
    """Find tokenized workers and marked launcher/supervisor processes."""

    own_pid = os.getpid() if own_pid is None else own_pid
    result: list[dict[str, Any]] = []
    token_entry = f"GATE4_TOKEN={token}".encode()
    lifecycle_names = (
        "pbs_role_launcher.py",
        "pbs_role_supervisor.py",
        "pbs_scheduler_gate4_worker.py",
    )
    for process_dir in proc_root.glob("[0-9]*"):
        try:
            pid = int(process_dir.name)
        except ValueError:
            continue
        if pid == own_pid:
            continue
        token_match = False
        command_match = False
        command = ""
        try:
            environment = (process_dir / "environ").read_bytes().split(b"\0")
            token_match = token_entry in environment
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            pass
        try:
            command = (process_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            command_match = marker in command and any(name in command for name in lifecycle_names)
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            pass
        if token_match or command_match:
            result.append(
                {
                    "pid": pid,
                    "token_match": token_match,
                    "command_match": command_match,
                    "command": command,
                }
            )
    return sorted(result, key=lambda value: int(value["pid"]))


def probe_node(
    vnode_index: int,
    token: str,
    marker: str,
    result_root: Path,
) -> dict[str, Any]:
    challenge = uuid4().hex
    result_path = result_root / marker / f"vnode-{vnode_index}-{challenge}.json"
    result_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    command = [
        "/opt/pbs/bin/pbsdsh",
        "-n",
        str(vnode_index),
        "--",
        "/usr/bin/python3",
        str(Path(__file__).resolve()),
        "--probe",
        "--token",
        token,
        "--marker",
        marker,
        "--challenge",
        challenge,
        "--probe-result",
        str(result_path),
    ]
    completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"remote survivor probe failed ({completed.returncode}): {completed.stdout}")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            value = read_private_json(result_path)
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.1)
            continue
        break
    else:
        raise RuntimeError(
            f"remote survivor probe returned no private result for vnode {vnode_index}: {completed.stdout}"
        )
    if value.get("challenge") != challenge:
        raise RuntimeError(f"remote survivor probe challenge mismatch: {value}")
    value["vnode_index"] = vnode_index
    return value


def assert_no_survivors(
    nodes: list[Any],
    state_root: Path,
    trial: str,
    token: str,
    timeout: float = 45,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    last_probes: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        last_probes = [
            probe_node(
                node.vnode_index,
                token,
                trial,
                state_root / "survivor-probes",
            )
            for node in nodes
        ]
        controls = list((state_root / "scheduler-state").rglob(f"{trial}/*.control.json"))
        entries = list(
            (state_root / "name-resolve" / os.environ["GATE4_REGISTRY_USER"] / "pbs-scheduler-gate4" / trial).rglob(
                "ENTRY"
            )
        )
        if all(not probe["survivors"] for probe in last_probes) and not controls and not entries:
            return last_probes
        time.sleep(0.5)
    raise AssertionError(
        f"Gate 4 survivors remain for {trial}: probes={last_probes}, controls={controls}, entries={entries}"
    )


def run_normal_phase(
    module: types.ModuleType,
    repo_root: Path,
    state_root: Path,
    nodes: list[Any],
    digest: str,
    trial: str,
) -> dict[str, Any]:
    scheduler = make_scheduler(module, state_root, trial)
    token = f"{trial}-token"
    actor_paths: dict[str, str] | None = None
    replacement_actor_paths: dict[str, str] | None = None
    rollout_paths: dict[str, str] | None = None
    first_actor_status: dict[str, Any] | None = None
    try:
        actor_ids = scheduler.create_workers(
            separated_job("actor_like", 8, worker_spec(repo_root, state_root, digest, token, 1))
        )
        actor_paths = role_paths(scheduler, "actor_like")
        rollout_ids = scheduler.create_workers(
            separated_job("rollout_like", 1, worker_spec(repo_root, state_root, digest, token, 8))
        )
        rollout_paths = role_paths(scheduler, "rollout_like")
        actor_manifest = validate_role_manifest(
            Path(actor_paths["manifest"]),
            role="actor_like",
            expected_host=nodes[0].hostname,
            expected_vnode=nodes[0].vnode_index,
            expected_digest=digest,
            expected_assignments=[
                {"gpu_ids": [index], "local_rank": index, "worker_index": index} for index in range(8)
            ],
        )
        rollout_manifest = validate_role_manifest(
            Path(rollout_paths["manifest"]),
            role="rollout_like",
            expected_host=nodes[1].hostname,
            expected_vnode=nodes[1].vnode_index,
            expected_digest=digest,
            expected_assignments=[{"gpu_ids": list(range(8)), "local_rank": 0, "worker_index": 0}],
        )
        actor_workers = scheduler.get_workers("actor_like", timeout=120)
        rollout_workers = scheduler.get_workers("rollout_like", timeout=120)
        actor_lifecycle = validate_lifecycle_metadata(actor_manifest)
        rollout_lifecycle = validate_lifecycle_metadata(rollout_manifest)
        actor_receipts = validate_ready_receipts(
            state_root,
            trial,
            "actor_like",
            actor_workers,
            nodes[0].hostname,
            digest,
            1,
            actor_manifest["generation"],
        )
        rollout_receipts = validate_ready_receipts(
            state_root,
            trial,
            "rollout_like",
            rollout_workers,
            nodes[1].hostname,
            digest,
            8,
            rollout_manifest["generation"],
        )
        if actor_ids != [f"actor_like/{index}" for index in range(8)] or rollout_ids != ["rollout_like/0"]:
            raise AssertionError(f"unexpected explicit worker ids: actor={actor_ids}, rollout={rollout_ids}")
        if set(scheduler._role_nodes["actor_like"]) & set(scheduler._role_nodes["rollout_like"]):
            raise AssertionError("actor-like and rollout-like roles share a physical node")

        # Delete and recreate the actor-like role while the rollout-like role
        # remains live.  This is the production race that stale AReaL ENTRY
        # records used to make unsafe: the old generation must finish exact
        # process and discovery cleanup before the replacement can register.
        scheduler.delete_workers("actor_like", reverse_order=True)
        first_actor_status = wait_for_launcher_status(Path(actor_paths["status"]))
        first_actor_final_receipts = wait_for_receipts(
            state_root,
            trial,
            "actor_like",
            8,
        )
        actor_role_root = Path(actor_manifest["discovery"]["role_root"])
        first_actor_cleanup = assert_cleanup_union(
            first_actor_status,
            first_actor_final_receipts,
            expected_generation=actor_manifest["generation"],
            expected_entry_paths=[
                Path(value) for value in actor_manifest["discovery"]["entry_paths"]
            ],
            expected_role_root=actor_role_root,
        )
        rollout_entries = [Path(value) for value in rollout_manifest["discovery"]["entry_paths"]]
        if not all(path.is_file() for path in rollout_entries):
            raise AssertionError("actor cleanup removed the live rollout discovery record")
        scheduler._check_job_status("rollout_like")
        rollout_health_while_actor_recreated = assert_worker_health(
            rollout_workers[0],
            role="rollout_like",
            worker_index=0,
            generation=rollout_manifest["generation"],
        )

        for index in range(8):
            receipt_path(state_root, trial, "actor_like", index).unlink(missing_ok=True)
        replacement_actor_ids = scheduler.create_workers(
            separated_job("actor_like", 8, worker_spec(repo_root, state_root, digest, token, 1))
        )
        replacement_actor_paths = role_paths(scheduler, "actor_like")
        replacement_actor_manifest = validate_role_manifest(
            Path(replacement_actor_paths["manifest"]),
            role="actor_like",
            expected_host=nodes[0].hostname,
            expected_vnode=nodes[0].vnode_index,
            expected_digest=digest,
            expected_assignments=[
                {"gpu_ids": [index], "local_rank": index, "worker_index": index} for index in range(8)
            ],
        )
        if replacement_actor_manifest["generation"] == actor_manifest["generation"]:
            raise AssertionError("recreated actor reused its old generation nonce")
        replacement_actor_workers = scheduler.get_workers("actor_like", timeout=120)
        replacement_actor_lifecycle = validate_lifecycle_metadata(replacement_actor_manifest)
        replacement_actor_receipts = validate_ready_receipts(
            state_root,
            trial,
            "actor_like",
            replacement_actor_workers,
            nodes[0].hostname,
            digest,
            1,
            replacement_actor_manifest["generation"],
        )
        if replacement_actor_ids != actor_ids:
            raise AssertionError(f"recreated actor ids changed: {replacement_actor_ids} != {actor_ids}")
    finally:
        # Every ordinary test error must still take the production scheduler's
        # confirmed remote-cleanup path before the outer PBS script exits.
        for active_role in reversed(list(scheduler._workers)):
            scheduler.delete_workers(active_role, reverse_order=True)

    assert actor_paths is not None and replacement_actor_paths is not None and rollout_paths is not None
    assert first_actor_status is not None
    replacement_actor_status = wait_for_launcher_status(Path(replacement_actor_paths["status"]))
    rollout_status = wait_for_launcher_status(Path(rollout_paths["status"]))
    replacement_actor_final_receipts = wait_for_receipts(
        state_root,
        trial,
        "actor_like",
        8,
    )
    rollout_final_receipts = wait_for_receipts(
        state_root,
        trial,
        "rollout_like",
        1,
    )
    replacement_actor_cleanup = assert_cleanup_union(
        replacement_actor_status,
        replacement_actor_final_receipts,
        expected_generation=replacement_actor_manifest["generation"],
        expected_entry_paths=[
            Path(value)
            for value in replacement_actor_manifest["discovery"]["entry_paths"]
        ],
        expected_role_root=Path(replacement_actor_manifest["discovery"]["role_root"]),
    )
    rollout_cleanup = assert_cleanup_union(
        rollout_status,
        rollout_final_receipts,
        expected_generation=rollout_manifest["generation"],
        expected_entry_paths=[
            Path(value) for value in rollout_manifest["discovery"]["entry_paths"]
        ],
        expected_role_root=Path(rollout_manifest["discovery"]["role_root"]),
    )
    probes = assert_no_survivors(nodes, state_root, trial, token)
    if scheduler._node_owners or scheduler._role_nodes:
        raise AssertionError(f"scheduler retained node ownership: {scheduler._node_owners}")
    return {
        "actor_ids": actor_ids,
        "actor_manifest_generation": actor_manifest["generation"],
        "actor_receipts": actor_receipts,
        "actor_final_receipts": first_actor_final_receipts,
        "actor_discovery_cleanup": first_actor_cleanup,
        "actor_lifecycle": actor_lifecycle,
        "actor_status": first_actor_status,
        "replacement_actor_ids": replacement_actor_ids,
        "replacement_actor_manifest_generation": replacement_actor_manifest["generation"],
        "replacement_actor_receipts": replacement_actor_receipts,
        "replacement_actor_final_receipts": replacement_actor_final_receipts,
        "replacement_actor_discovery_cleanup": replacement_actor_cleanup,
        "replacement_actor_lifecycle": replacement_actor_lifecycle,
        "replacement_actor_status": replacement_actor_status,
        "rollout_ids": rollout_ids,
        "rollout_manifest_generation": rollout_manifest["generation"],
        "rollout_receipts": rollout_receipts,
        "rollout_final_receipts": rollout_final_receipts,
        "rollout_discovery_cleanup": rollout_cleanup,
        "rollout_lifecycle": rollout_lifecycle,
        "rollout_status": rollout_status,
        "rollout_health_while_actor_recreated": rollout_health_while_actor_recreated,
        "survivor_probes": probes,
    }


def run_failure_phase(
    module: types.ModuleType,
    repo_root: Path,
    state_root: Path,
    nodes: list[Any],
    digest: str,
    trial: str,
) -> dict[str, Any]:
    scheduler = make_scheduler(module, state_root, trial)
    token = f"{trial}-token"
    paths: dict[str, str] | None = None
    try:
        scheduler.create_workers(
            separated_job(
                "failing_actor",
                8,
                worker_spec(repo_root, state_root, digest, token, 1, fail_worker_index=3),
            )
        )
        paths = role_paths(scheduler, "failing_actor")
        receipts = wait_for_receipts(state_root, trial, "failing_actor", 8)
        manifest = read_private_json(Path(paths["manifest"]))
        lifecycle = validate_lifecycle_metadata(manifest)
        deadline = time.monotonic() + 90
        observed: str | None = None
        while time.monotonic() < deadline:
            try:
                scheduler._check_job_status("failing_actor")
            except GateWorkerFailedError as error:
                observed = str(error)
                break
            time.sleep(0.2)
        if observed is None:
            raise TimeoutError("scheduler did not surface the deliberate worker failure")
    finally:
        for active_role in reversed(list(scheduler._workers)):
            scheduler.delete_workers(active_role, reverse_order=True)

    assert paths is not None and observed is not None
    status = wait_for_launcher_status(Path(paths["status"]))
    assert_clean_status(status, expect_failure=True)
    probes = assert_no_survivors(nodes, state_root, trial, token)
    if scheduler._node_owners or scheduler._role_nodes:
        raise AssertionError(f"failed role retained node ownership: {scheduler._node_owners}")
    final_receipts = wait_for_receipts(state_root, trial, "failing_actor", 8)
    if final_receipts[3].get("state") != "failed" or final_receipts[3].get("exit_status") != 23:
        raise AssertionError(f"deliberate failure receipt is wrong: {final_receipts[3]}")
    discovery_cleanup = assert_cleanup_union(
        status,
        final_receipts,
        expected_generation=manifest["generation"],
        expected_entry_paths=[
            Path(value) for value in manifest["discovery"]["entry_paths"]
        ],
        expected_role_root=Path(manifest["discovery"]["role_root"]),
        expect_failure=True,
    )
    return {
        "initial_receipts": receipts,
        "lifecycle": lifecycle,
        "final_receipts": final_receipts,
        "observed_scheduler_error": observed,
        "discovery_cleanup": discovery_cleanup,
        "status": status,
        "survivor_probes": probes,
    }


def run_constructor_race_phase(
    module: types.ModuleType,
    repo_root: Path,
    state_root: Path,
    nodes: list[Any],
    digest: str,
    trial: str,
) -> dict[str, Any]:
    """Exercise controller replacement against delayed old-launcher cleanup."""

    role_launcher = _load_source(
        f"pbs_scheduler_gate4_role_launcher_{uuid4().hex}",
        repo_root / "platoon" / "train" / "areal" / "pbs_role_launcher.py",
    )
    old_scheduler = make_scheduler(module, state_root, trial)
    token = f"{trial}-token"
    role = "constructor_race_actor"
    replacement_schedulers: list[Any] = []
    replacement_values: dict[str, Any] = {}
    replacement_errors: list[str] = []
    old_paths: dict[str, str] | None = None
    old_status: dict[str, Any] | None = None
    replacement_status: dict[str, Any] | None = None
    stale_role_view = None
    thread: threading.Thread | None = None
    blocked_started = 0.0
    constructor_blocked_seconds = 0.0

    def construct_and_launch_replacement() -> None:
        try:
            scheduler = make_scheduler(module, state_root, trial)
            replacement_schedulers.append(scheduler)
            replacement_values["constructed_monotonic"] = time.monotonic()
            ids = scheduler.create_workers(
                separated_job(
                    role,
                    8,
                    worker_spec(repo_root, state_root, digest, token, 1),
                )
            )
            paths = role_paths(scheduler, role)
            workers = scheduler.get_workers(role, timeout=120)
            replacement_values.update(
                {
                    "ids": ids,
                    "paths": paths,
                    "workers": workers,
                }
            )
        except BaseException as error:
            replacement_errors.append(f"{type(error).__name__}: {error}")

    try:
        old_ids = old_scheduler.create_workers(
            separated_job(
                role,
                8,
                worker_spec(repo_root, state_root, digest, token, 1),
            )
        )
        old_paths = role_paths(old_scheduler, role)
        old_manifest = validate_role_manifest(
            Path(old_paths["manifest"]),
            role=role,
            expected_host=nodes[0].hostname,
            expected_vnode=nodes[0].vnode_index,
            expected_digest=digest,
            expected_assignments=[
                {"gpu_ids": [index], "local_rank": index, "worker_index": index} for index in range(8)
            ],
        )
        old_workers = old_scheduler.get_workers(role, timeout=120)
        old_lifecycle = validate_lifecycle_metadata(old_manifest)
        old_receipts = validate_ready_receipts(
            state_root,
            trial,
            role,
            old_workers,
            nodes[0].hostname,
            digest,
            1,
            old_manifest["generation"],
        )
        old_entries = [Path(value) for value in old_manifest["discovery"]["entry_paths"]]
        old_entry_values = {str(path): path.read_text() for path in old_entries}
        stale_role_view = Path(old_manifest["discovery"]["lifecycle_lock_path"]).open()

        # Model abrupt controller death without killing this gate process.  The
        # old launcher retains shared trial ownership until its remote cleanup,
        # so a new constructor must not reach upstream's recursive clear.
        old_scheduler._release_trial_lifecycle_lock()
        blocked_started = time.monotonic()
        thread = threading.Thread(
            target=construct_and_launch_replacement,
            name="gate4-replacement-constructor",
            daemon=True,
        )
        thread.start()
        time.sleep(1.0)
        if not thread.is_alive() or replacement_schedulers or replacement_errors:
            raise AssertionError(
                "replacement constructor passed the trial lock before old launcher cleanup: "
                f"schedulers={len(replacement_schedulers)} errors={replacement_errors}"
            )
        if {str(path): path.read_text() for path in old_entries} != old_entry_values:
            raise AssertionError("replacement constructor cleared or changed old discovery entries")
        blocked_health = assert_worker_health(
            old_workers[0],
            role=role,
            worker_index=0,
            generation=old_manifest["generation"],
        )

        old_scheduler.delete_workers(role, reverse_order=True)
        old_status = wait_for_launcher_status(Path(old_paths["status"]))
        assert_clean_status(old_status)
        thread.join(timeout=180)
        if thread.is_alive():
            raise TimeoutError("replacement scheduler did not pass the trial lock after old cleanup")
        constructor_blocked_seconds = float(replacement_values["constructed_monotonic"]) - blocked_started
        if replacement_errors or len(replacement_schedulers) != 1:
            raise RuntimeError(f"replacement scheduler failed: {replacement_errors}")

        replacement_paths = replacement_values["paths"]
        replacement_manifest = validate_role_manifest(
            Path(replacement_paths["manifest"]),
            role=role,
            expected_host=nodes[0].hostname,
            expected_vnode=nodes[0].vnode_index,
            expected_digest=digest,
            expected_assignments=[
                {"gpu_ids": [index], "local_rank": index, "worker_index": index} for index in range(8)
            ],
        )
        if replacement_values["ids"] != old_ids:
            raise AssertionError(f"replacement role ids changed: {replacement_values['ids']} != {old_ids}")
        if replacement_manifest["generation"] == old_manifest["generation"]:
            raise AssertionError("replacement role reused the old generation nonce")
        if replacement_manifest["discovery"]["lifecycle_lock_path"] != old_manifest["discovery"]["lifecycle_lock_path"]:
            raise AssertionError("same role did not reuse its stable external lifecycle lock")
        replacement_lifecycle = validate_lifecycle_metadata(replacement_manifest)
        replacement_workers = replacement_values["workers"]
        deadline = time.monotonic() + 20
        while True:
            try:
                replacement_receipts = validate_ready_receipts(
                    state_root,
                    trial,
                    role,
                    replacement_workers,
                    nodes[0].hostname,
                    digest,
                    1,
                    replacement_manifest["generation"],
                )
                break
            except (AssertionError, FileNotFoundError, json.JSONDecodeError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)

        # A delayed callback retaining an old descriptor sees the replacement
        # generation metadata and must fail closed before touching any ENTRY.
        try:
            role_launcher._cleanup_discovery_records(
                old_manifest,
                stale_role_view,
            )
        except RuntimeError as error:
            stale_cleanup_error = str(error)
            if "generation changed" not in stale_cleanup_error:
                raise
        else:
            raise AssertionError("stale old-generation cleanup unexpectedly succeeded")
        replacement_entries = [Path(value) for value in replacement_manifest["discovery"]["entry_paths"]]
        if not all(path.is_file() for path in replacement_entries):
            raise AssertionError("stale cleanup removed a replacement ENTRY")
        replacement_health = assert_worker_health(
            replacement_workers[0],
            role=role,
            worker_index=0,
            generation=replacement_manifest["generation"],
        )
    finally:
        for active_role in reversed(list(old_scheduler._workers)):
            old_scheduler.delete_workers(active_role, reverse_order=True)
        old_scheduler._release_trial_lifecycle_lock()
        if thread is not None:
            thread.join(timeout=180)
        for scheduler in replacement_schedulers:
            for active_role in reversed(list(scheduler._workers)):
                scheduler.delete_workers(active_role, reverse_order=True)
            scheduler._release_trial_lifecycle_lock()
        if stale_role_view is not None:
            stale_role_view.close()

    assert old_paths is not None and old_status is not None
    replacement_paths = replacement_values["paths"]
    replacement_status = wait_for_launcher_status(Path(replacement_paths["status"]))
    assert_clean_status(replacement_status)
    probes = assert_no_survivors(nodes, state_root, trial, token)
    return {
        "constructor_blocked_seconds": constructor_blocked_seconds,
        "old_generation": old_manifest["generation"],
        "old_lifecycle": old_lifecycle,
        "old_receipts": old_receipts,
        "old_status": old_status,
        "blocked_old_worker_health": blocked_health,
        "replacement_generation": replacement_manifest["generation"],
        "replacement_lifecycle": replacement_lifecycle,
        "replacement_receipts": replacement_receipts,
        "replacement_status": replacement_status,
        "replacement_health_after_stale_cleanup": replacement_health,
        "stale_cleanup_error": stale_cleanup_error,
        "survivor_probes": probes,
    }


def run_controller_death_child(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    state_root = args.state_root.resolve()
    image = args.image.resolve()
    configure_runtime(repo_root, state_root, image)
    module = load_real_pbs_scheduler(repo_root)
    nodes = pbs_nodes(module)
    scheduler = make_scheduler(module, state_root, args.trial)
    token = f"{args.trial}-token"
    scheduler.create_workers(
        separated_job("death_actor", 8, worker_spec(repo_root, state_root, args.expected_digest, token, 1))
    )
    scheduler.create_workers(
        separated_job("death_rollout", 1, worker_spec(repo_root, state_root, args.expected_digest, token, 8))
    )
    actor_workers = scheduler.get_workers("death_actor", timeout=120)
    rollout_workers = scheduler.get_workers("death_rollout", timeout=120)
    validate_ready_receipts(
        state_root, args.trial, "death_actor", actor_workers, nodes[0].hostname, args.expected_digest, 1
    )
    validate_ready_receipts(
        state_root, args.trial, "death_rollout", rollout_workers, nodes[1].hostname, args.expected_digest, 8
    )
    actor_lifecycle = validate_lifecycle_metadata(read_private_json(Path(scheduler._role_manifests["death_actor"])))
    rollout_lifecycle = validate_lifecycle_metadata(read_private_json(Path(scheduler._role_manifests["death_rollout"])))
    atomic_json(
        args.ready_path,
        {
            "actor": role_paths(scheduler, "death_actor"),
            "actor_lifecycle": actor_lifecycle,
            "controller_pid": os.getpid(),
            "rollout": role_paths(scheduler, "death_rollout"),
            "rollout_lifecycle": rollout_lifecycle,
            "token": token,
        },
    )
    print(f"gate4_controller_death_ready pid={os.getpid()} trial={args.trial}", flush=True)
    os.kill(os.getpid(), signal.SIGKILL)
    raise AssertionError("SIGKILL returned")


def run_controller_death_phase(
    repo_root: Path,
    state_root: Path,
    nodes: list[Any],
    digest: str,
    trial: str,
) -> dict[str, Any]:
    ready_path = state_root / "controller-death-ready.json"
    ready_path.unlink(missing_ok=True)
    log_path = state_root / "controller-death.log"
    command = [
        "/usr/bin/python3",
        "-u",
        str(Path(__file__).resolve()),
        "--controller-death-child",
        "--repo-root",
        str(repo_root),
        "--state-root",
        str(state_root),
        "--image",
        str(Path(os.environ["PLATOON_AREAL_PBS_CONTAINER_IMAGE"])),
        "--expected-digest",
        digest,
        "--trial",
        trial,
        "--ready-path",
        str(ready_path),
    ]
    with log_path.open("w") as log_stream:
        process = subprocess.Popen(command, stdout=log_stream, stderr=subprocess.STDOUT, env=os.environ.copy())
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline and not ready_path.exists():
            if process.poll() is not None:
                raise RuntimeError(
                    f"controller-death child exited before readiness ({process.returncode}):\n{log_path.read_text()}"
                )
            time.sleep(0.2)
        if not ready_path.exists():
            process.terminate()
            raise TimeoutError("controller-death child did not become ready")
        returncode = process.wait(timeout=30)
    if returncode != -signal.SIGKILL:
        raise AssertionError(f"controller-death child returned {returncode}, expected {-signal.SIGKILL}")
    ready = read_private_json(ready_path)
    actor_status = wait_for_launcher_status(Path(ready["actor"]["status"]))
    rollout_status = wait_for_launcher_status(Path(ready["rollout"]["status"]))
    assert_clean_status(actor_status)
    assert_clean_status(rollout_status)
    probes = assert_no_survivors(nodes, state_root, trial, str(ready["token"]), timeout=60)
    actor_generation = ready["actor_lifecycle"]["role"]["metadata"]["generation"]
    rollout_generation = ready["rollout_lifecycle"]["role"]["metadata"]["generation"]
    actor_manifest = read_private_json(Path(ready["actor"]["manifest"]))
    rollout_manifest = read_private_json(Path(ready["rollout"]["manifest"]))
    actor_final_receipts = wait_for_receipts(
        state_root,
        trial,
        "death_actor",
        8,
    )
    rollout_final_receipts = wait_for_receipts(
        state_root,
        trial,
        "death_rollout",
        1,
    )
    actor_cleanup = assert_cleanup_union(
        actor_status,
        actor_final_receipts,
        expected_generation=actor_generation,
        expected_entry_paths=[
            Path(value) for value in actor_manifest["discovery"]["entry_paths"]
        ],
        expected_role_root=Path(actor_manifest["discovery"]["role_root"]),
    )
    rollout_cleanup = assert_cleanup_union(
        rollout_status,
        rollout_final_receipts,
        expected_generation=rollout_generation,
        expected_entry_paths=[
            Path(value) for value in rollout_manifest["discovery"]["entry_paths"]
        ],
        expected_role_root=Path(rollout_manifest["discovery"]["role_root"]),
    )
    return {
        "controller_pid": ready["controller_pid"],
        "controller_returncode": returncode,
        "actor_status": actor_status,
        "actor_final_receipts": actor_final_receipts,
        "actor_discovery_cleanup": actor_cleanup,
        "actor_lifecycle": ready["actor_lifecycle"],
        "rollout_status": rollout_status,
        "rollout_final_receipts": rollout_final_receipts,
        "rollout_discovery_cleanup": rollout_cleanup,
        "rollout_lifecycle": ready["rollout_lifecycle"],
        "survivor_probes": probes,
        "log": str(log_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--probe", action="store_true")
    mode.add_argument("--controller-death-child", action="store_true")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--expected-sha")
    parser.add_argument("--expected-digest")
    parser.add_argument("--trial")
    parser.add_argument("--ready-path", type=Path)
    parser.add_argument("--token")
    parser.add_argument("--marker")
    parser.add_argument("--challenge")
    parser.add_argument("--probe-result", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.probe:
        if not args.token or not args.marker or not args.challenge or args.probe_result is None:
            raise ValueError("--probe requires --token, --marker, --challenge, and --probe-result")
        value = {
            "challenge": args.challenge,
            "host": os.uname().nodename,
            "survivors": find_survivors(args.token, args.marker),
        }
        atomic_json(args.probe_result, value)
        print("gate4_probe_published", flush=True)
        return 0
    if args.controller_death_child:
        required = (args.repo_root, args.state_root, args.image, args.expected_digest, args.trial, args.ready_path)
        if any(value is None for value in required):
            raise ValueError("controller-death child arguments are incomplete")
        return run_controller_death_child(args)
    if args.repo_root is None or args.state_root is None or args.image is None or args.expected_sha is None:
        raise ValueError("main gate requires --repo-root, --state-root, --image, and --expected-sha")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    repo_root = args.repo_root.resolve()
    state_root = args.state_root.resolve()
    image = args.image.resolve()
    configure_runtime(repo_root, state_root, image)
    actual_sha = sha256(image)
    if actual_sha != args.expected_sha:
        raise AssertionError(f"SIF SHA-256 {actual_sha} != expected {args.expected_sha}")
    digest = f"sha256:{actual_sha}"
    module = load_real_pbs_scheduler(repo_root)
    nodes = pbs_nodes(module)
    node_local_init_receipts = validate_node_local_init_receipts(nodes)
    safe_job = str(os.environ["PBS_JOBID"]).split(".", 1)[0]

    normal_trial = f"gate4-{safe_job}-normal"
    constructor_race_trial = f"gate4-{safe_job}-constructor-race"
    failure_trial = f"gate4-{safe_job}-failure"
    death_trial = f"gate4-{safe_job}-controller-death"
    started = time.time()
    print(
        f"gate4_start job={os.environ['PBS_JOBID']} nodes={[node.hostname for node in nodes]} "
        f"vnode_indices={[node.vnode_index for node in nodes]} image_digest={digest}",
        flush=True,
    )
    normal = run_normal_phase(module, repo_root, state_root, nodes, digest, normal_trial)
    print("gate4_normal_phase=passed", flush=True)
    constructor_race = run_constructor_race_phase(
        module,
        repo_root,
        state_root,
        nodes,
        digest,
        constructor_race_trial,
    )
    print("gate4_constructor_race_phase=passed", flush=True)
    failure = run_failure_phase(module, repo_root, state_root, nodes, digest, failure_trial)
    print("gate4_deliberate_failure_phase=passed", flush=True)
    controller_death = run_controller_death_phase(repo_root, state_root, nodes, digest, death_trial)
    print("gate4_controller_death_phase=passed", flush=True)

    result = {
        "schema": 1,
        "state": "passed",
        "pbs_job_id": os.environ["PBS_JOBID"],
        "ordered_nodes": [dataclasses.asdict(node) if dataclasses.is_dataclass(node) else list(node) for node in nodes],
        "image": str(image),
        "image_sha256": actual_sha,
        "node_local_init_generation": os.environ["GATE4_NODE_LOCAL_INIT_GENERATION"],
        "node_local_init_receipts": node_local_init_receipts,
        "normal": normal,
        "constructor_race": constructor_race,
        "deliberate_failure": failure,
        "controller_death": controller_death,
        "elapsed_seconds": time.time() - started,
        "scope": (
            "real PreallocatedPBSScheduler/PBS launcher/supervisor and exact AReaL NFS discovery + /health; "
            "full AReaL rpc_server configure/engine creation remains gated on the immutable runtime"
        ),
    }
    result_path = state_root / "RESULT.json"
    atomic_json(result_path, result)
    print(f"gate4_pbs_scheduler=passed result={result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
