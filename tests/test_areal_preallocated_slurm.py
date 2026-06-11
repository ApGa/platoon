"""Focused tests for Platoon's preallocated Slurm scheduler."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class FakeWorker:
    id: str
    ip: str
    worker_ports: list[str]
    engine_ports: list[str]


@dataclass
class FakeJob:
    role: str
    replicas: int
    tasks: list
    scheduling_strategy: object


class FakeSchedulingStrategyType(str, Enum):
    separation = "separation"
    colocation = "colocation"


@dataclass
class FakeSchedulingSpec:
    cpu: int = 8
    gpu: int = 0
    mem: int = 32
    port_count: int = 2
    image: str = ""
    task_type: str = "worker"
    env_vars: dict[str, str] = field(default_factory=dict)
    cmd: str | None = None
    srun_additional_args: str = "--unbuffered"
    additional_bash_cmds: list[str] | None = None
    container_type: str = "none"
    mount: str = ""
    nodelist: str | None = None
    exclude: str | None = None
    ray_placement_strategy: str = "shared"


@dataclass
class FakeSlurmWorkerInfo:
    worker: FakeWorker
    role: str
    slurm_job_id: int
    task_index: int
    discovered: bool = False
    spec: FakeSchedulingSpec | None = None
    node: str | None = None


class FakeWorkerError(Exception):
    pass


class FakeWorkerCreationError(FakeWorkerError):
    pass


class FakeWorkerFailedError(FakeWorkerError):
    pass


class FakeWorkerNotFoundError(FakeWorkerError):
    pass


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class FakeSlurmScheduler:
    def __init__(self, *args, **kwargs):
        self.n_gpus_per_node = 8
        self.experiment_name = "exp"
        self.trial_name = "trial"
        self.fileroot = "/tmp/areal"
        self.name_resolve_config = types.SimpleNamespace(
            type="nfs",
            nfs_record_root="/tmp/areal/name_resolve",
            etcd3_addr="localhost:2379",
        )
        self._workers = {}
        self._jobs = {}
        self._job_status_cache = {}
        self._colocated_roles = {}
        self.startup_timeout = 1
        self.health_check_interval = 0.01
        self.exp_config = None

    def _prepare_worker_specs(self, role, num_workers, schedulings):
        if len(schedulings) == 1:
            return schedulings * num_workers
        return schedulings

    def _log_path_of(self, role):
        return f"/tmp/{role}.log"

    def _merged_log_path(self):
        return "/tmp/merged.log"

    def _read_log_tail(self, role):
        return "tail"


def _install_fake_areal(monkeypatch):
    areal_mod = types.ModuleType("areal")
    api_mod = types.ModuleType("areal.api")
    api_mod.Job = FakeJob
    api_mod.Worker = FakeWorker
    monkeypatch.setitem(sys.modules, "areal", areal_mod)
    monkeypatch.setitem(sys.modules, "areal.api", api_mod)

    cli_args_mod = types.ModuleType("areal.api.cli_args")
    cli_args_mod.SchedulingSpec = FakeSchedulingSpec
    cli_args_mod.SchedulingStrategyType = FakeSchedulingStrategyType
    monkeypatch.setitem(sys.modules, "areal.api.cli_args", cli_args_mod)

    exceptions_mod = types.ModuleType("areal.infra.scheduler.exceptions")
    exceptions_mod.WorkerCreationError = FakeWorkerCreationError
    exceptions_mod.WorkerFailedError = FakeWorkerFailedError
    exceptions_mod.WorkerNotFoundError = FakeWorkerNotFoundError
    monkeypatch.setitem(sys.modules, "areal.infra.scheduler.exceptions", exceptions_mod)

    slurm_mod = types.ModuleType("areal.infra.scheduler.slurm")
    slurm_mod.SlurmScheduler = FakeSlurmScheduler
    slurm_mod.SlurmWorkerInfo = FakeSlurmWorkerInfo
    monkeypatch.setitem(sys.modules, "areal.infra.scheduler.slurm", slurm_mod)

    proc_mod = types.ModuleType("areal.infra.utils.proc")
    proc_mod.build_streaming_log_cmd = (
        lambda cmd, role_log, merged_log, role: " ".join(cmd) if isinstance(cmd, list) else cmd
    )
    monkeypatch.setitem(sys.modules, "areal.infra.utils.proc", proc_mod)

    utils_mod = types.ModuleType("areal.utils")
    utils_mod.logging = types.SimpleNamespace(getLogger=lambda name: FakeLogger())
    monkeypatch.setitem(sys.modules, "areal.utils", utils_mod)


def _load_scheduler_module(monkeypatch):
    _install_fake_areal(monkeypatch)
    return _load_module(
        "platoon_preallocated_slurm_test",
        REPO_ROOT / "platoon/train/areal/preallocated_slurm.py",
    )


def test_preallocated_command_uses_current_srun_allocation(monkeypatch):
    module = _load_scheduler_module(monkeypatch)
    monkeypatch.setenv("PLATOON_AREAL_PREALLOC_CONTAINER_IMAGE", "/image.sqsh")
    monkeypatch.setenv("PLATOON_AREAL_PREALLOC_CONTAINER_MOUNTS", "/lustre:/lustre")
    monkeypatch.setenv("PLATOON_AREAL_PREALLOC_CONTAINER_WORKDIR", "/work")
    monkeypatch.setenv("PLATOON_AREAL_PREALLOC_WORKER_PREAMBLE", "export PATH=/venv/bin:$PATH")

    scheduler = module.PreallocatedSlurmScheduler()
    command = scheduler._build_role_srun_command(
        role="actor",
        replicas=8,
        nodes=1,
        total_gpus=8,
        cpus_per_task=2,
        mem_per_task=4096,
        schedulings=[FakeSchedulingSpec(cpu=2, gpu=1, mem=4)],
        nodelist=None,
        exclude=None,
    )

    assert command.startswith("srun ")
    assert "--exclusive" in command
    assert "--gpus-per-node=8" in command
    assert "--container-image=/image.sqsh" in command
    assert "CUDA_VISIBLE_DEVICES" in command
    assert "--experiment-name exp" in command
    assert "export PATH=/venv/bin:$PATH" in command
    assert "\n;\n" not in command


def test_preallocated_overlap_workers_do_not_use_exclusive(monkeypatch):
    module = _load_scheduler_module(monkeypatch)
    monkeypatch.setenv("PLATOON_AREAL_PREALLOC_SRUN_BIN", "/usr/bin/srun")
    monkeypatch.setenv("PLATOON_AREAL_PREALLOC_SRUN_ARGS", "--unbuffered --overlap")

    scheduler = module.PreallocatedSlurmScheduler()
    command = scheduler._build_role_srun_command(
        role="actor",
        replicas=8,
        nodes=1,
        total_gpus=8,
        cpus_per_task=2,
        mem_per_task=4096,
        schedulings=[FakeSchedulingSpec(cpu=2, gpu=1, mem=4)],
        nodelist=None,
        exclude=None,
    )

    assert command.startswith("/usr/bin/srun ")
    assert "--overlap" in command
    assert "--exclusive" not in command


def test_separated_roles_are_pinned_to_distinct_nodes(monkeypatch):
    module = _load_scheduler_module(monkeypatch)

    scheduler = module.PreallocatedSlurmScheduler()
    # Two-node allocation; spreading should pin actor -> node0, rollout -> node1.
    monkeypatch.setattr(scheduler, "_allocation_nodes", lambda: ["node0", "node1"])

    captured: dict[str, str] = {}

    class FakeProcess:
        pid = 999

        def poll(self):
            return None

    def fake_launch(role, command):
        captured[role] = command
        return FakeProcess()

    monkeypatch.setattr(scheduler, "_launch_role_process", fake_launch)

    for role in ("actor", "rollout"):
        job = FakeJob(
            role=role,
            replicas=8,
            tasks=[FakeSchedulingSpec(cpu=2, gpu=1, mem=4)],
            scheduling_strategy=types.SimpleNamespace(type="separation", target=None),
        )
        scheduler.create_workers(job)

    assert "--nodelist=node0" in captured["actor"]
    assert "--nodelist=node1" in captured["rollout"]


def test_node_spreading_can_be_disabled(monkeypatch):
    module = _load_scheduler_module(monkeypatch)
    monkeypatch.setenv("PLATOON_AREAL_PREALLOC_SPREAD_NODES", "0")

    scheduler = module.PreallocatedSlurmScheduler()
    monkeypatch.setattr(scheduler, "_allocation_nodes", lambda: ["node0", "node1"])

    captured: dict[str, str] = {}

    class FakeProcess:
        pid = 1000

        def poll(self):
            return None

    monkeypatch.setattr(
        scheduler,
        "_launch_role_process",
        lambda role, command: captured.setdefault(role, command) or FakeProcess(),
    )

    job = FakeJob(
        role="actor",
        replicas=8,
        tasks=[FakeSchedulingSpec(cpu=2, gpu=1, mem=4)],
        scheduling_strategy=types.SimpleNamespace(type="separation", target=None),
    )
    scheduler.create_workers(job)

    assert "--nodelist" not in captured["actor"]


def test_create_workers_tracks_background_srun_process(monkeypatch):
    module = _load_scheduler_module(monkeypatch)

    class FakeProcess:
        pid = 12345

        def __init__(self):
            self.status = None

        def poll(self):
            return self.status

    process = FakeProcess()
    scheduler = module.PreallocatedSlurmScheduler()
    monkeypatch.setattr(scheduler, "_launch_role_process", lambda role, command: process)

    job = FakeJob(
        role="actor",
        replicas=8,
        tasks=[FakeSchedulingSpec(cpu=2, gpu=1, mem=4)],
        scheduling_strategy=types.SimpleNamespace(type="separation", target=None),
    )

    worker_ids = scheduler.create_workers(job)

    assert worker_ids == [f"actor/{i}" for i in range(8)]
    assert scheduler._jobs["actor"] == process.pid
    assert scheduler._role_processes["actor"] is process
    scheduler._check_job_status("actor")

    process.status = 1
    with pytest.raises(FakeWorkerFailedError):
        scheduler._check_job_status("actor")
