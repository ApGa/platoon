"""Focused tests for Platoon's preallocated Slurm scheduler."""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
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

    def error(self, *args, **kwargs):
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

    def _configure_worker(self, worker_info, worker_rank):
        pass


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


def _worker_info(role: str, rank: int, host: str) -> FakeSlurmWorkerInfo:
    return FakeSlurmWorkerInfo(
        worker=FakeWorker(
            id=f"{role}/{rank}",
            ip=host,
            worker_ports=[str(10000 + rank)],
            engine_ports=[],
        ),
        role=role,
        slurm_job_id=123,
        task_index=rank,
        discovered=True,
        spec=FakeSchedulingSpec(),
        node=host,
    )


def test_worker_configuration_is_parallel_across_hosts_and_serial_per_host(monkeypatch):
    module = _load_scheduler_module(monkeypatch)
    scheduler = module.PreallocatedSlurmScheduler()
    workers = [
        _worker_info("actor", 0, "node-a"),
        _worker_info("actor", 1, "node-a"),
        _worker_info("actor", 2, "node-b"),
        _worker_info("actor", 3, "node-b"),
        _worker_info("actor", 4, "node-c"),
    ]
    # Preallocated discovery currently populates IP but not SlurmWorkerInfo.node.
    for worker in workers:
        worker.node = None
    scheduler._workers["actor"] = workers

    state_lock = threading.Lock()
    active_hosts: set[str] = set()
    max_active_hosts = 0
    calls: list[int] = []

    def configure(_self, worker_info, worker_rank):
        nonlocal max_active_hosts
        host = worker_info.worker.ip
        with state_lock:
            # Workers on one host must be configured by the same serial stream.
            assert host not in active_hosts
            active_hosts.add(host)
            max_active_hosts = max(max_active_hosts, len(active_hosts))
            calls.append(worker_rank)
        time.sleep(0.03)
        with state_lock:
            active_hosts.remove(host)

    monkeypatch.setattr(FakeSlurmScheduler, "_configure_worker", configure)

    # This first upstream-loop call configures the complete role.
    scheduler._configure_worker(workers[0], 0)
    assert sorted(calls) == list(range(len(workers)))
    assert max_active_hosts >= 2

    # The remainder of upstream's serial loop must be no-ops.
    for rank, worker in enumerate(workers[1:], start=1):
        scheduler._configure_worker(worker, rank)
    assert len(calls) == len(workers)


def test_worker_configuration_respects_concurrency_cap(monkeypatch):
    module = _load_scheduler_module(monkeypatch)
    monkeypatch.setenv("PLATOON_AREAL_PREALLOC_CONFIGURE_CONCURRENCY", "2")
    scheduler = module.PreallocatedSlurmScheduler()
    workers = [_worker_info("actor", rank, f"node-{rank}") for rank in range(6)]
    scheduler._workers["actor"] = workers

    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def configure(_self, _worker_info, _worker_rank):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with state_lock:
            active -= 1

    monkeypatch.setattr(FakeSlurmScheduler, "_configure_worker", configure)

    scheduler._configure_worker(workers[0], 0)

    assert max_active == 2


def test_worker_configuration_preserves_failure_and_allows_retry(monkeypatch):
    module = _load_scheduler_module(monkeypatch)
    scheduler = module.PreallocatedSlurmScheduler()
    workers = [
        _worker_info("actor", 0, "node-a"),
        _worker_info("actor", 1, "node-a"),
        _worker_info("actor", 2, "node-b"),
    ]
    scheduler._workers["actor"] = workers

    expected = FakeWorkerFailedError("rank zero failed")
    calls: list[int] = []

    def fail_first(_self, _worker_info, worker_rank):
        calls.append(worker_rank)
        if worker_rank == 0:
            raise expected

    monkeypatch.setattr(FakeSlurmScheduler, "_configure_worker", fail_first)

    with pytest.raises(FakeWorkerFailedError) as caught:
        scheduler._configure_worker(workers[0], 0)

    assert caught.value is expected
    assert 1 not in calls  # same-host stream stops after its first failure
    assert 2 in calls  # independent hosts are allowed to finish
    assert "actor" not in scheduler._configured_worker_generations

    retry_calls: list[int] = []
    monkeypatch.setattr(
        FakeSlurmScheduler,
        "_configure_worker",
        lambda _self, _worker_info, rank: retry_calls.append(rank),
    )
    scheduler._configure_worker(workers[0], 0)
    assert sorted(retry_calls) == [0, 1, 2]


def test_worker_configuration_cache_is_cleared_when_role_is_recreated(monkeypatch):
    module = _load_scheduler_module(monkeypatch)
    scheduler = module.PreallocatedSlurmScheduler()
    calls: list[str] = []
    monkeypatch.setattr(
        FakeSlurmScheduler,
        "_configure_worker",
        lambda _self, worker_info, _rank: calls.append(worker_info.worker.id),
    )

    original = [_worker_info("actor", 0, "node-a")]
    scheduler._workers["actor"] = original
    scheduler._configure_worker(original[0], 0)
    scheduler.delete_workers("actor")

    replacement = [_worker_info("actor", 0, "node-b")]
    scheduler._workers["actor"] = replacement
    scheduler._configure_worker(replacement[0], 0)

    assert calls == ["actor/0", "actor/0"]


@pytest.mark.parametrize("value", ["0", "-1", "not-an-integer"])
def test_worker_configuration_rejects_invalid_concurrency(monkeypatch, value):
    module = _load_scheduler_module(monkeypatch)
    monkeypatch.setenv("PLATOON_AREAL_PREALLOC_CONFIGURE_CONCURRENCY", value)

    with pytest.raises(ValueError, match="positive integer"):
        module.PreallocatedSlurmScheduler()


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
        lambda role, command: captured.__setitem__(role, command) or FakeProcess(),
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


def _configure_idle_guard(monkeypatch, tmp_path):
    monkeypatch.setenv("PLATOON_AREAL_ROLLOUT_IDLE_GUARD", "1")
    monkeypatch.setenv("PLATOON_AREAL_ROLLOUT_IDLE_GUARD_SCRIPT", "/repo/rollout_gpu_idle_guard.py")
    monkeypatch.setenv("PLATOON_AREAL_ROLLOUT_IDLE_GUARD_PYTHON", "/venv/bin/python")
    monkeypatch.setenv("PLATOON_AREAL_ROLLOUT_IDLE_GUARD_READY_DIR", str(tmp_path / "ready"))
    monkeypatch.setenv("PLATOON_AREAL_ROLLOUT_IDLE_GUARD_LOG_PREFIX", str(tmp_path / "guard"))


@pytest.mark.parametrize(
    ("role", "enabled"),
    [
        ("rollout", True),
        ("actor", True),
        ("eval-actor", False),
        ("actor-worker", False),
        ("eval-rollout", False),
        ("proxy-rollout", False),
        ("rollout-worker", False),
        ("trainer", False),
    ],
)
def test_idle_guard_matches_only_exact_role(monkeypatch, role, enabled):
    module = _load_scheduler_module(monkeypatch)
    monkeypatch.setenv("PLATOON_AREAL_ROLLOUT_IDLE_GUARD", "1")

    assert module.PreallocatedSlurmScheduler._idle_guard_enabled(role) is enabled


def test_idle_guard_commands_use_exact_role_topologies(monkeypatch, tmp_path):
    module = _load_scheduler_module(monkeypatch)
    _configure_idle_guard(monkeypatch, tmp_path)
    monkeypatch.setenv("PLATOON_AREAL_PREALLOC_CONTAINER_IMAGE", "/image.sqsh")
    scheduler = module.PreallocatedSlurmScheduler()
    cases = [
        ("actor", 10, ",".join(f"node{rank}" for rank in range(10)), 80),
        ("rollout", 6, ",".join(f"node{rank}" for rank in range(10, 16)), 48),
    ]

    for role, nodes, nodelist, task_count in cases:
        command = scheduler._build_idle_guard_command(role, nodes, nodelist)

        assert "--overlap" in command
        assert "--exact" in command
        assert f"--job-name={role}-idle-guard" in command
        assert f"--nodes={nodes}" in command
        assert f"--ntasks={task_count}" in command
        assert "--ntasks-per-node=8" in command
        assert "--cpu-bind=none" in command
        assert "--gpus-per-task=1" in command
        assert "--gpu-bind=single:1" in command
        assert "--kill-on-bad-exit=1" in command
        assert "--exclusive" not in command
        assert f"--nodelist={nodelist}" in command
        assert "/repo/rollout_gpu_idle_guard.py" in command
        assert str(tmp_path / "ready" / role) in command
        assert f"{tmp_path / 'guard'}-{role}-%N-%t.log" in command


def test_idle_guard_ready_dirs_are_role_isolated(monkeypatch, tmp_path):
    module = _load_scheduler_module(monkeypatch)
    _configure_idle_guard(monkeypatch, tmp_path)

    actor_dir = module.PreallocatedSlurmScheduler._prepare_idle_guard_ready_dir(
        "actor"
    )
    actor_marker = actor_dir / "0.ready"
    actor_marker.write_text("actor", encoding="utf-8")
    rollout_dir = module.PreallocatedSlurmScheduler._prepare_idle_guard_ready_dir(
        "rollout"
    )

    assert actor_dir == tmp_path / "ready" / "actor"
    assert rollout_dir == tmp_path / "ready" / "rollout"
    assert actor_marker.read_text(encoding="utf-8") == "actor"


def test_actor_and_rollout_guards_keep_independent_lifecycle_state(
    monkeypatch, tmp_path
):
    module = _load_scheduler_module(monkeypatch)
    _configure_idle_guard(monkeypatch, tmp_path)
    scheduler = module.PreallocatedSlurmScheduler()

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid
            self.status = None

        def poll(self):
            return self.status

        def wait(self, timeout):
            self.status = -15
            return self.status

    processes = {"actor": FakeProcess(100), "rollout": FakeProcess(200)}
    monkeypatch.setattr(
        scheduler,
        "_launch_idle_guard_process",
        lambda role, command: processes[role],
    )
    monkeypatch.setattr(scheduler, "_wait_for_idle_guard", lambda *args: None)

    scheduler._start_idle_guard(
        "actor", 10, ",".join(f"node{rank}" for rank in range(10))
    )
    scheduler._start_idle_guard(
        "rollout", 6, ",".join(f"node{rank}" for rank in range(10, 16))
    )

    assert scheduler._role_idle_guard_processes == processes
    assert set(scheduler._role_idle_guard_commands) == {"actor", "rollout"}
    assert "--job-name=actor-idle-guard" in scheduler._role_idle_guard_commands[
        "actor"
    ]
    assert "--job-name=rollout-idle-guard" in scheduler._role_idle_guard_commands[
        "rollout"
    ]

    signals = []
    monkeypatch.setattr(module.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    scheduler._terminate_idle_guard_process("actor")

    assert "actor" not in scheduler._role_idle_guard_processes
    assert "actor" not in scheduler._role_idle_guard_commands
    assert scheduler._role_idle_guard_processes["rollout"] is processes["rollout"]
    assert "rollout" in scheduler._role_idle_guard_commands

    scheduler._terminate_idle_guard_process("rollout")
    assert signals == [
        (100, module.signal.SIGTERM),
        (200, module.signal.SIGTERM),
    ]


def test_idle_guard_starts_before_each_guarded_role_on_same_nodes(monkeypatch, tmp_path):
    module = _load_scheduler_module(monkeypatch)
    _configure_idle_guard(monkeypatch, tmp_path)
    cases = [("actor", 80, 1, 10), ("rollout", 6, 8, 6)]

    class FakeProcess:
        pid = 4321

        def poll(self):
            return None

    for role, replicas, gpus_per_worker, expected_nodes in cases:
        scheduler = module.PreallocatedSlurmScheduler()
        monkeypatch.setattr(
            scheduler,
            "_allocation_nodes",
            lambda: [f"node{rank}" for rank in range(16)],
        )
        events = []
        monkeypatch.setattr(
            scheduler,
            "_start_idle_guard",
            lambda role, nodes, nodelist: events.append(
                ("guard", role, nodes, nodelist)
            ),
        )
        monkeypatch.setattr(
            scheduler,
            "_launch_role_process",
            lambda role, command: events.append(("role", role, command))
            or FakeProcess(),
        )

        scheduler.create_workers(
            FakeJob(
                role=role,
                replicas=replicas,
                tasks=[FakeSchedulingSpec(cpu=4, gpu=gpus_per_worker, mem=16)],
                scheduling_strategy=types.SimpleNamespace(
                    type="separation", target=None
                ),
            )
        )

        expected_nodelist = ",".join(
            f"node{rank}" for rank in range(expected_nodes)
        )
        assert events[0] == (
            "guard",
            role,
            expected_nodes,
            expected_nodelist,
        )
        assert events[1][0:2] == ("role", role)
        assert f"--nodelist={expected_nodelist}" in events[1][2]


def test_rollout_role_launch_failure_cleans_up_ready_guard(monkeypatch, tmp_path):
    module = _load_scheduler_module(monkeypatch)
    _configure_idle_guard(monkeypatch, tmp_path)
    scheduler = module.PreallocatedSlurmScheduler()
    monkeypatch.setattr(scheduler, "_allocation_nodes", lambda: ["node0"])
    events = []
    monkeypatch.setattr(
        scheduler,
        "_start_idle_guard",
        lambda role, nodes, nodelist: events.append(("start", role)),
    )
    monkeypatch.setattr(
        scheduler,
        "_launch_role_process",
        lambda role, command: (_ for _ in ()).throw(RuntimeError("launch failed")),
    )
    monkeypatch.setattr(
        scheduler,
        "_terminate_idle_guard_process",
        lambda role: events.append(("stop", role)),
    )

    with pytest.raises(RuntimeError, match="launch failed"):
        scheduler.create_workers(
            FakeJob(
                role="rollout",
                replicas=1,
                tasks=[FakeSchedulingSpec(cpu=4, gpu=8, mem=16)],
                scheduling_strategy=types.SimpleNamespace(type="separation", target=None),
            )
        )

    assert events == [("start", "rollout"), ("stop", "rollout")]


@pytest.mark.parametrize("guard_status", [0, 1])
def test_unexpected_idle_guard_exit_fails_and_stops_role(monkeypatch, guard_status):
    module = _load_scheduler_module(monkeypatch)
    scheduler = module.PreallocatedSlurmScheduler()

    class FakeProcess:
        def __init__(self, pid, status):
            self.pid = pid
            self.status = status

        def poll(self):
            return self.status

        def wait(self, timeout):
            self.status = -15
            return self.status

    role_process = FakeProcess(100, None)
    guard_process = FakeProcess(200, guard_status)
    scheduler._role_processes["rollout"] = role_process
    scheduler._role_idle_guard_processes["rollout"] = guard_process
    scheduler._role_idle_guard_commands["rollout"] = "guard"
    scheduler._workers["rollout"] = []
    signals = []
    monkeypatch.setattr(module.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    with pytest.raises(FakeWorkerFailedError, match="idle guard exited unexpectedly"):
        scheduler._check_job_status("rollout")

    assert signals == [(100, module.signal.SIGTERM)]
    assert "rollout" not in scheduler._role_processes
    assert "rollout" not in scheduler._role_idle_guard_processes


def test_delete_guarded_role_stops_guard_before_main(monkeypatch):
    module = _load_scheduler_module(monkeypatch)

    for role in ("actor", "rollout"):
        scheduler = module.PreallocatedSlurmScheduler()
        scheduler._workers[role] = []
        events = []
        monkeypatch.setattr(
            scheduler,
            "_terminate_idle_guard_process",
            lambda stopped_role: events.append(("guard", stopped_role)),
        )
        monkeypatch.setattr(
            scheduler,
            "_terminate_role_process",
            lambda stopped_role: events.append(("role", stopped_role)),
        )

        scheduler.delete_workers(role)

        assert events == [("guard", role), ("role", role)]


def test_rollout_registration_failure_cleans_up_guard_and_role(monkeypatch, tmp_path):
    module = _load_scheduler_module(monkeypatch)
    _configure_idle_guard(monkeypatch, tmp_path)
    scheduler = module.PreallocatedSlurmScheduler()
    monkeypatch.setattr(scheduler, "_allocation_nodes", lambda: ["node0"])
    events = []

    class FakeProcess:
        pid = 900

        def poll(self):
            return None

    monkeypatch.setattr(
        scheduler,
        "_start_idle_guard",
        lambda role, nodes, nodelist: events.append(("start", role)),
    )
    monkeypatch.setattr(
        scheduler,
        "_launch_role_process",
        lambda role, command: FakeProcess(),
    )
    monkeypatch.setattr(
        module,
        "Worker",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("registration failed")),
    )
    monkeypatch.setattr(
        scheduler,
        "_terminate_idle_guard_process",
        lambda role: events.append(("guard", role)),
    )
    monkeypatch.setattr(
        scheduler,
        "_terminate_role_process",
        lambda role: events.append(("role", role)),
    )

    with pytest.raises(RuntimeError, match="registration failed"):
        scheduler.create_workers(
            FakeJob(
                role="rollout",
                replicas=1,
                tasks=[FakeSchedulingSpec(cpu=4, gpu=8, mem=16)],
                scheduling_strategy=types.SimpleNamespace(
                    type="separation", target=None
                ),
            )
        )

    assert events == [
        ("start", "rollout"),
        ("guard", "rollout"),
        ("role", "rollout"),
    ]
