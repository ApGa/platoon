"""Focused tests for Platoon's preallocated PBS scheduler."""

from __future__ import annotations

import getpass
import json
import shutil
import stat
import sys
import threading
import time
import types
from pathlib import Path

import pytest
from test_areal_preallocated_slurm import (
    FakeJob,
    FakeSchedulingSpec,
    _install_fake_areal,
    _load_module,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeProcess:
    _next_pid = 4000

    def __init__(self):
        self.pid = FakeProcess._next_pid
        FakeProcess._next_pid += 1
        self.status = None

    def poll(self):
        return self.status

    def terminate(self):
        self.status = 0

    def wait(self, timeout=None):
        self.status = -15
        return self.status


def _load_scheduler_module(monkeypatch):
    _install_fake_areal(monkeypatch)

    # Load the dependency without executing platoon.train.areal.__init__, which
    # imports the full optional training stack unavailable in these unit tests.
    platoon_package = types.ModuleType("platoon")
    platoon_package.__path__ = [str(REPO_ROOT / "platoon")]
    train_package = types.ModuleType("platoon.train")
    train_package.__path__ = [str(REPO_ROOT / "platoon/train")]
    areal_package = types.ModuleType("platoon.train.areal")
    areal_package.__path__ = [str(REPO_ROOT / "platoon/train/areal")]
    monkeypatch.setitem(sys.modules, "platoon", platoon_package)
    monkeypatch.setitem(sys.modules, "platoon.train", train_package)
    monkeypatch.setitem(sys.modules, "platoon.train.areal", areal_package)
    _load_module(
        "platoon.train.areal.preallocated_slurm",
        REPO_ROOT / "platoon/train/areal/preallocated_slurm.py",
    )
    return _load_module(
        "platoon_preallocated_pbs_test",
        REPO_ROOT / "platoon/train/areal/preallocated_pbs.py",
    )


def _separated_job(role: str, replicas: int, spec: FakeSchedulingSpec) -> FakeJob:
    return FakeJob(
        role=role,
        replicas=replicas,
        tasks=[spec],
        scheduling_strategy=types.SimpleNamespace(type="separation", target=None),
    )


def _scheduler(monkeypatch, tmp_path, node_count=16, name_resolve_root=None):
    module = _load_scheduler_module(monkeypatch)
    monkeypatch.setenv("PLATOON_AREAL_PBS_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PLATOON_AREAL_PBS_CONTAINER_RUNTIME", "host")
    name_resolve_root = name_resolve_root or (tmp_path / "name-resolve")
    scheduler = module.PreallocatedPBSScheduler(
        experiment_name="exp",
        trial_name="trial",
        fileroot=str(tmp_path),
        name_resolve_type="nfs",
        nfs_record_root=str(name_resolve_root),
    )
    # FakeSlurmScheduler intentionally ignores constructor values. Restore the
    # values the real base class would retain so manifest identity is exercised.
    scheduler.experiment_name = "exp"
    scheduler.trial_name = "trial"
    scheduler.fileroot = str(tmp_path)
    scheduler.name_resolve_config.nfs_record_root = str(name_resolve_root)
    nodes = [module.PBSNode(f"node-{index:02d}", index * 8) for index in range(node_count)]
    monkeypatch.setattr(scheduler, "_pbs_allocation_nodes", lambda: nodes)
    processes: dict[str, FakeProcess] = {}

    def launch(role, _command):
        process = FakeProcess()
        processes[role] = process
        return process

    monkeypatch.setattr(scheduler, "_launch_role_process", launch)
    return module, scheduler, processes


def test_nodefile_parser_preserves_order_and_first_vnode_index(monkeypatch):
    module = _load_scheduler_module(monkeypatch)

    nodes = module.parse_pbs_nodefile_lines(["node-b\n"] * 8 + ["node-a\n"] * 8 + ["node-b\n"])

    assert nodes == [module.PBSNode("node-b", 0), module.PBSNode("node-a", 8)]


def test_pbs_scheduler_import_and_fork_route_remain_controller_safe(monkeypatch):
    before = set(sys.modules)
    module = _load_scheduler_module(monkeypatch)
    imported = set(sys.modules) - before

    assert module.PreallocatedPBSScheduler.__module__ == "platoon_preallocated_pbs_test"
    assert not {"torch", "megatron", "transformer_engine"} & imported


def test_exact_16_node_actor_and_rollout_topology(monkeypatch, tmp_path):
    _module, scheduler, _processes = _scheduler(monkeypatch, tmp_path, node_count=16)

    actor_ids = scheduler.create_workers(_separated_job("actor", 80, FakeSchedulingSpec(cpu=2, gpu=1, mem=16)))
    rollout_ids = scheduler.create_workers(_separated_job("rollout", 6, FakeSchedulingSpec(cpu=8, gpu=8, mem=64)))

    assert len(actor_ids) == 80
    assert len(rollout_ids) == 6
    assert [node.hostname for node in scheduler._role_nodes["actor"]] == [f"node-{index:02d}" for index in range(10)]
    assert [node.hostname for node in scheduler._role_nodes["rollout"]] == [
        f"node-{index:02d}" for index in range(10, 16)
    ]

    actor_manifest = json.loads(scheduler._role_manifests["actor"].read_text())
    rollout_manifest = json.loads(scheduler._role_manifests["rollout"].read_text())
    assert [host["vnode_index"] for host in actor_manifest["hosts"]] == list(range(0, 80, 8))
    assert [host["vnode_index"] for host in rollout_manifest["hosts"]] == list(range(80, 128, 8))
    first_actor_host = actor_manifest["hosts"][0]["workers"]
    assert [worker["worker_index"] for worker in first_actor_host] == list(range(8))
    assert [worker["gpu_ids"] for worker in first_actor_host] == [[index] for index in range(8)]
    assert rollout_manifest["hosts"][0]["workers"] == [{"gpu_ids": list(range(8)), "local_rank": 0, "worker_index": 0}]
    assert actor_manifest["gpu_visibility"] == {
        "mode": "legacy",
        "nvidia_smi_binary": "/usr/bin/nvidia-smi",
    }
    assert actor_manifest["termination_reverse_order_default"] is True
    assert "--worker-index" in actor_manifest["worker_command"][-1]
    assert "SLURM" not in actor_manifest["worker_command"][-1]


def test_allocation_exhaustion_does_not_mutate_existing_reservations(monkeypatch, tmp_path):
    _module, scheduler, _processes = _scheduler(monkeypatch, tmp_path, node_count=2)
    scheduler.create_workers(_separated_job("actor", 16, FakeSchedulingSpec(cpu=2, gpu=1, mem=8)))

    with pytest.raises(Exception, match="Insufficient PBS allocation"):
        scheduler.create_workers(_separated_job("rollout", 1, FakeSchedulingSpec(cpu=4, gpu=8, mem=8)))

    assert scheduler._node_owners == {"node-00": "actor", "node-01": "actor"}
    assert "rollout" not in scheduler._role_nodes


def test_worker_registration_failure_rolls_back_process_manifest_and_nodes(monkeypatch, tmp_path):
    module, scheduler, processes = _scheduler(monkeypatch, tmp_path, node_count=1)
    monkeypatch.setattr(
        module,
        "Worker",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("registration failed")),
    )
    terminated: list[str] = []

    def terminate(role):
        terminated.append(role)
        scheduler._role_processes.pop(role, None)

    monkeypatch.setattr(scheduler, "_terminate_role_process", terminate)

    with pytest.raises(RuntimeError, match="registration failed"):
        scheduler.create_workers(_separated_job("actor", 8, FakeSchedulingSpec(cpu=2, gpu=1, mem=8)))

    assert processes["actor"].poll() is None
    assert terminated == ["actor"]
    assert scheduler._node_owners == {}
    assert scheduler._role_nodes == {}
    assert scheduler._role_manifests == {}
    assert not list(tmp_path.rglob("*.manifest.json"))


def test_rejects_heterogeneous_specs_before_reserving_nodes(monkeypatch, tmp_path):
    _module, scheduler, _processes = _scheduler(monkeypatch, tmp_path, node_count=1)
    specs = [FakeSchedulingSpec(cpu=2, gpu=1, mem=8) for _ in range(8)]
    specs[-1] = FakeSchedulingSpec(cpu=4, gpu=1, mem=8)

    with pytest.raises(Exception, match="heterogeneous"):
        scheduler.create_workers(
            FakeJob(
                role="actor",
                replicas=8,
                tasks=specs,
                scheduling_strategy=types.SimpleNamespace(type="separation", target=None),
            )
        )

    assert scheduler._node_owners == {}


@pytest.mark.parametrize(
    ("replicas", "gpus", "message"),
    [(1, 3, "does not divide"), (7, 1, "does not fill whole")],
)
def test_rejects_nondivisible_or_partial_node_topology(monkeypatch, tmp_path, replicas, gpus, message):
    _module, scheduler, _processes = _scheduler(monkeypatch, tmp_path, node_count=1)

    with pytest.raises(Exception, match=message):
        scheduler.create_workers(_separated_job("actor", replicas, FakeSchedulingSpec(gpu=gpus)))

    assert scheduler._node_owners == {}


def test_manifest_forwards_environment_without_putting_values_on_command_line(monkeypatch, tmp_path):
    _module, scheduler, _processes = _scheduler(monkeypatch, tmp_path, node_count=1)
    monkeypatch.setenv("FORWARDED_SECRET", "controller secret with spaces;$")
    monkeypatch.setenv("PLATOON_AREAL_PBS_FORWARD_ENV", "FORWARDED_SECRET")
    spec = FakeSchedulingSpec(
        cpu=2,
        gpu=1,
        mem=8,
        env_vars={"SPEC_VALUE": "literal $(do-not-run)"},
    )

    scheduler.create_workers(_separated_job("actor", 8, spec))

    manifest_path = scheduler._role_manifests["actor"]
    manifest = json.loads(manifest_path.read_text())
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600
    assert manifest["worker_environment"] == {
        "FORWARDED_SECRET": "controller secret with spaces;$",
        "SPEC_VALUE": "literal $(do-not-run)",
    }
    launcher_command = scheduler._role_commands["actor"]
    assert "controller secret" not in launcher_command
    assert "do-not-run" not in launcher_command


def test_host_launcher_environment_excludes_credentials_and_runtime_injection(monkeypatch, tmp_path):
    _module, scheduler, _processes = _scheduler(monkeypatch, tmp_path, node_count=1)
    safe_values = {
        "PBS_JOBID": "123.pbs1",
        "PATH": "/site/bin:/usr/bin",
        "LD_LIBRARY_PATH": "/site/hpcx/lib",
        "CUSTOM_SAFE_FLAG": "enabled",
    }
    forbidden_values = {
        "WANDB_API_KEY": "wandb-secret",
        "OPENAI_API_KEY": "openai-secret",
        "HF_TOKEN": "hf-secret",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "SINGULARITY_BINDPATH": "/home:/home",
        "APPTAINERENV_INJECTED": "runtime-bypass",
        "LD_PRELOAD": "/tmp/inject.so",
        "PYTHONPATH": "/tmp/import-inject",
        "UNRELATED_INHERITED_VALUE": "not-needed",
        "CUSTOM_DATABASE_PASSWORD_FILE": "/secret/db-password",
    }
    for name, value in {**safe_values, **forbidden_values}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(
        "PLATOON_AREAL_PBS_LAUNCHER_FORWARD_ENV",
        "CUSTOM_SAFE_FLAG",
    )

    environment = scheduler._host_launcher_environment("actor")

    assert {name: environment[name] for name in safe_values} == safe_values
    assert not set(forbidden_values) & set(environment)


def test_host_launcher_refuses_explicit_credential_forward(monkeypatch, tmp_path):
    _module, scheduler, _processes = _scheduler(monkeypatch, tmp_path, node_count=1)
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv(
        "PLATOON_AREAL_PBS_LAUNCHER_FORWARD_ENV",
        "OPENAI_API_KEY",
    )

    with pytest.raises(Exception, match="Refusing to expose"):
        scheduler._host_launcher_environment("actor")


@pytest.mark.asyncio
async def test_fork_uses_captured_worker_python_without_mutating_controller_state(monkeypatch, tmp_path):
    worker_python = tmp_path / "worker-venv" / "bin" / "python"
    worker_python.parent.mkdir(parents=True)
    worker_python.symlink_to("/usr/bin/python3.12")
    monkeypatch.setenv("PLATOON_AREAL_PBS_WORKER_PYTHON", str(worker_python))
    monkeypatch.setenv("OPENAI_API_KEY", "controller-only-secret")
    module, scheduler, _processes = _scheduler(monkeypatch, tmp_path, node_count=1)
    controller_python = sys.executable
    later_python = tmp_path / "different-worker-venv" / "bin" / "python"
    later_python.parent.mkdir(parents=True)
    later_python.symlink_to("/usr/bin/python3.12")
    monkeypatch.setenv("PLATOON_AREAL_PBS_WORKER_PYTHON", str(later_python))
    posted: list[tuple[str, dict]] = []

    class Response:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class RecordingSession:
        def post(self, url, *args, **kwargs):
            del args
            posted.append((url, kwargs["json"]))
            return Response()

    async def upstream_fork(_self, session, role, idx, target_wi, target_role, command=None):
        del target_wi, target_role
        module_path = command or "areal.infra.rpc.rpc_server"
        async with session.post("http://worker:1234/alloc_ports", json={"count": 1}):
            pass
        async with session.post(
            "http://worker:1234/fork",
            json={
                "role": role,
                "worker_index": idx,
                "raw_cmd": [
                    sys.executable,
                    "-m",
                    module_path,
                    "--worker-index",
                    str(idx),
                ],
            },
        ):
            pass
        async with session.post(
            "http://worker:1234/kill_forked_worker",
            json={"role": role, "worker_index": idx, "raw_cmd": [sys.executable]},
        ):
            pass
        return "forked-worker"

    monkeypatch.setattr(
        module.SlurmScheduler,
        "_fork_single_worker",
        upstream_fork,
        raising=False,
    )
    result = await scheduler._fork_single_worker(
        RecordingSession(),
        "proxy-rollout",
        3,
        object(),
        "rollout",
        "platoon.areal_proxy_rollout",
    )

    assert result == "forked-worker"
    assert sys.executable == controller_python
    assert posted == [
        ("http://worker:1234/alloc_ports", {"count": 1}),
        (
            "http://worker:1234/fork",
            {
                "role": "proxy-rollout",
                "worker_index": 3,
                "raw_cmd": [
                    str(worker_python),
                    "-m",
                    "platoon.areal_proxy_rollout",
                    "--worker-index",
                    "3",
                ],
            },
        ),
        (
            "http://worker:1234/kill_forked_worker",
            {
                "role": "proxy-rollout",
                "worker_index": 3,
                "raw_cmd": [controller_python],
            },
        ),
    ]
    assert "controller-only-secret" not in repr(posted)


@pytest.mark.asyncio
async def test_fork_without_worker_python_fails_before_upstream_or_remote_call(monkeypatch, tmp_path):
    monkeypatch.delenv("PLATOON_AREAL_PBS_WORKER_PYTHON", raising=False)
    module, scheduler, _processes = _scheduler(monkeypatch, tmp_path, node_count=1)
    upstream_called = False

    async def upstream_fork(*_args, **_kwargs):
        nonlocal upstream_called
        upstream_called = True

    monkeypatch.setattr(
        module.SlurmScheduler,
        "_fork_single_worker",
        upstream_fork,
        raising=False,
    )

    with pytest.raises(Exception, match="interpreter is not configured"):
        await scheduler._fork_single_worker(object(), "proxy-rollout", 0, object(), "rollout")
    assert not upstream_called


@pytest.mark.asyncio
async def test_fork_rejects_changed_upstream_interpreter_before_post(monkeypatch, tmp_path):
    worker_python = tmp_path / "worker-python"
    worker_python.symlink_to("/usr/bin/python3.12")
    monkeypatch.setenv("PLATOON_AREAL_PBS_WORKER_PYTHON", str(worker_python))
    module, scheduler, _processes = _scheduler(monkeypatch, tmp_path, node_count=1)
    posted = False

    class RecordingSession:
        def post(self, *_args, **_kwargs):
            nonlocal posted
            posted = True

    async def changed_upstream(_self, session, role, idx, target_wi, target_role, command=None):
        del target_wi, target_role, command
        return session.post(
            "http://worker:1234/fork",
            json={
                "role": role,
                "worker_index": idx,
                "raw_cmd": ["/unexpected/python", "-m", "platoon.areal_proxy_rollout"],
            },
        )

    monkeypatch.setattr(
        module.SlurmScheduler,
        "_fork_single_worker",
        changed_upstream,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="interpreter changed unexpectedly"):
        await scheduler._fork_single_worker(RecordingSession(), "proxy-rollout", 0, object(), "rollout")
    assert not posted


@pytest.mark.parametrize(
    ("entry_kind", "value", "match"),
    [
        ("none", "relative/python", "absolute normalized"),
        ("none", "/missing/worker/python", "existing directory entry"),
        ("relative_symlink", "worker-python", "target must be absolute"),
        ("nonexec", "worker-python", "must be executable"),
        ("directory", "worker-python", "symlink or executable regular file"),
    ],
)
def test_invalid_fork_worker_python_is_rejected_before_scheduler_construction(
    monkeypatch, tmp_path, entry_kind, value, match
):
    path = tmp_path / value
    if entry_kind == "relative_symlink":
        path.symlink_to("python3.12")
        value = str(path)
    elif entry_kind == "nonexec":
        path.write_text("#!/bin/sh\n")
        path.chmod(0o600)
        value = str(path)
    elif entry_kind == "directory":
        path.mkdir()
        value = str(path)
    monkeypatch.setenv("PLATOON_AREAL_PBS_WORKER_PYTHON", value)

    module = _load_scheduler_module(monkeypatch)
    with pytest.raises(ValueError, match=match):
        module.PreallocatedPBSScheduler(
            experiment_name="exp",
            trial_name="trial",
            fileroot=str(tmp_path),
            name_resolve_type="nfs",
            nfs_record_root=str(tmp_path / "name-resolve"),
        )


def test_singularity_command_is_configurable(monkeypatch, tmp_path):
    _module, scheduler, _processes = _scheduler(monkeypatch, tmp_path, node_count=1)
    image = tmp_path / "train.sif"
    image.write_bytes(b"pinned training image")
    monkeypatch.setenv("PLATOON_AREAL_PBS_CONTAINER_RUNTIME", "/opt/singularity")
    monkeypatch.setenv("PLATOON_AREAL_PBS_CONTAINER_IMAGE", str(image))
    monkeypatch.setenv("PLATOON_AREAL_PBS_CONTAINER_BINDS", "/groups:/groups /local:/local")
    monkeypatch.setenv("PLATOON_AREAL_PBS_CONTAINER_WORKDIR", "/groups/work")
    monkeypatch.setenv("PLATOON_AREAL_PBS_GPU_VISIBILITY_MODE", "nvidia-container-cli")
    monkeypatch.setenv("PLATOON_AREAL_PBS_NVIDIA_SMI_BIN", "/usr/local/bin/nvidia-smi")

    scheduler.create_workers(_separated_job("actor", 8, FakeSchedulingSpec(cpu=2, gpu=1, mem=8)))

    manifest = json.loads(scheduler._role_manifests["actor"].read_text())
    command = manifest["worker_command"]
    assert command[:4] == ["/opt/singularity", "exec", "--nv", "--cleanenv"]
    assert command[4:7] == ["--no-home", "--no-mount", "home,cwd,bind-paths"]
    assert command.count("--bind") == 2
    assert str(image) in command
    assert manifest["container_image_digest"] == (
        "sha256:4c8501a1519322d21061f794099b4732f9e5bedb648dcec07ce1e93adace1b1a"
    )
    assert manifest["gpu_visibility"] == {
        "mode": "nvidia-container-cli",
        "nvidia_smi_binary": "/usr/local/bin/nvidia-smi",
    }
    assert "PLATOON_CONTAINER_IMAGE_DIGEST" in command[-1]
    assert "PLATOON_ASSIGNED_GPU_IDS" in command[-1]
    assert "PLATOON_ASSIGNED_GPU_UUIDS" in command[-1]
    assert command[-3:-1] == ["/bin/bash", "-c"]


@pytest.mark.parametrize(
    "mode,binary,match",
    [
        ("unknown", "/usr/bin/nvidia-smi", "Invalid GPU visibility mode"),
        ("nvidia-container-cli", "nvidia-smi", "must be an absolute path"),
    ],
)
def test_manifest_rejects_invalid_gpu_visibility_configuration(monkeypatch, tmp_path, mode, binary, match):
    _module, scheduler, _processes = _scheduler(monkeypatch, tmp_path, node_count=1)
    monkeypatch.setenv("PLATOON_AREAL_PBS_GPU_VISIBILITY_MODE", mode)
    monkeypatch.setenv("PLATOON_AREAL_PBS_NVIDIA_SMI_BIN", binary)

    with pytest.raises(Exception, match=match):
        scheduler.create_workers(_separated_job("actor", 8, FakeSchedulingSpec(cpu=2, gpu=1, mem=8)))


def test_delete_workers_releases_nodes_in_reverse_role_order(monkeypatch, tmp_path):
    _module, scheduler, _processes = _scheduler(monkeypatch, tmp_path, node_count=2)
    scheduler.create_workers(_separated_job("actor", 8, FakeSchedulingSpec(cpu=2, gpu=1, mem=8)))
    scheduler.create_workers(_separated_job("rollout", 1, FakeSchedulingSpec(cpu=8, gpu=8, mem=8)))
    terminated: list[tuple[str, bool]] = []

    def terminate(role, *, reverse_order=False):
        terminated.append((role, reverse_order))
        scheduler._role_processes.pop(role, None)

    monkeypatch.setattr(scheduler, "_terminate_role_process", terminate)

    scheduler.delete_workers(reverse_order=True)

    assert terminated == [("rollout", True), ("actor", True)]
    assert scheduler._node_owners == {}
    assert scheduler._role_nodes == {}


def test_delete_forked_alias_runs_lifecycle_before_metadata_drop(monkeypatch, tmp_path):
    module, scheduler, _processes = _scheduler(monkeypatch, tmp_path, node_count=1)
    owner = object()
    forked = object()
    scheduler._workers = {"rollout": [owner], "proxy": [forked]}
    scheduler._colocated_roles = {"proxy": "rollout"}
    observations = []

    def cleanup(instance, role, *, reverse_order=False):
        observations.append(
            (
                instance is scheduler,
                role,
                reverse_order,
                set(instance._workers),
                dict(instance._colocated_roles),
            )
        )
        return True

    monkeypatch.setattr(module, "cleanup_forked_alias_workers", cleanup)

    scheduler.delete_workers("proxy", reverse_order=True)

    assert observations == [(True, "proxy", True, {"rollout", "proxy"}, {"proxy": "rollout"})]
    assert scheduler._workers == {"rollout": [owner]}
    assert scheduler._colocated_roles == {}


@pytest.mark.parametrize(
    "failure",
    [
        "Owner Guard refused kill response",
        "fork endpoint did not attest process death",
        "Refusing to delete replaced fork discovery entries",
    ],
)
def test_delete_forked_alias_lifecycle_failure_preserves_metadata(
    monkeypatch,
    tmp_path,
    failure,
):
    module, scheduler, _processes = _scheduler(monkeypatch, tmp_path, node_count=1)
    owner = object()
    forked = object()
    scheduler._workers = {"rollout": [owner], "proxy": [forked]}
    scheduler._colocated_roles = {"proxy": "rollout"}
    monkeypatch.setattr(
        module,
        "cleanup_forked_alias_workers",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError(failure)),
    )

    with pytest.raises(RuntimeError, match=failure):
        scheduler.delete_workers("proxy")

    assert scheduler._workers == {"rollout": [owner], "proxy": [forked]}
    assert scheduler._colocated_roles == {"proxy": "rollout"}


def test_delete_plain_alias_remains_metadata_only(monkeypatch, tmp_path):
    module, scheduler, _processes = _scheduler(monkeypatch, tmp_path, node_count=1)
    owner = object()
    scheduler._workers = {"actor": [owner]}
    scheduler._colocated_roles = {"ref": "actor"}
    calls = []
    monkeypatch.setattr(
        module,
        "cleanup_forked_alias_workers",
        lambda *args, **kwargs: calls.append((args, kwargs)) or False,
    )

    scheduler.delete_workers("ref")

    assert len(calls) == 1
    assert scheduler._workers == {"actor": [owner]}
    assert scheduler._colocated_roles == {}


def test_delete_all_cleans_forked_alias_before_owner(monkeypatch, tmp_path):
    module, scheduler, _processes = _scheduler(monkeypatch, tmp_path, node_count=1)
    owner = object()
    forked = object()
    scheduler._workers = {"rollout": [owner], "proxy": [forked]}
    scheduler._colocated_roles = {"proxy": "rollout"}
    events = []

    def cleanup(instance, role, *, reverse_order=False):
        del reverse_order
        assert "rollout" in instance._workers
        events.append("cleanup:proxy")
        return True

    monkeypatch.setattr(module, "cleanup_forked_alias_workers", cleanup)
    monkeypatch.setattr(
        scheduler,
        "_terminate_role_process",
        lambda role: events.append(f"terminate:{role}"),
    )
    monkeypatch.setattr(scheduler, "_release_nodes", lambda role: None)

    scheduler.delete_workers()

    assert events == ["cleanup:proxy", "terminate:rollout"]
    assert scheduler._workers == {}
    assert scheduler._colocated_roles == {}


def test_direct_owner_delete_cleans_dependent_alias_first(monkeypatch, tmp_path):
    module, scheduler, _processes = _scheduler(monkeypatch, tmp_path, node_count=1)
    owner = object()
    alias = object()
    scheduler._workers = {"rollout": [owner], "proxy": [alias]}
    scheduler._colocated_roles = {"proxy": "rollout"}
    events = []

    def cleanup_dependents(instance, role, *, reverse_order=False):
        events.append(f"dependents:{role}:reverse={reverse_order}")
        if role == "rollout":
            instance.delete_workers("proxy", reverse_order=reverse_order)
            return ("proxy",)
        return ()

    monkeypatch.setattr(module, "cleanup_dependent_aliases_before_owner", cleanup_dependents)
    monkeypatch.setattr(
        module,
        "cleanup_forked_alias_workers",
        lambda instance, role, *, reverse_order=False: events.append(f"kill:{role}:reverse={reverse_order}") or True,
    )
    monkeypatch.setattr(
        scheduler,
        "_terminate_role_process",
        lambda role, **kwargs: events.append(f"terminate:{role}:{kwargs}"),
    )
    monkeypatch.setattr(scheduler, "_release_nodes", lambda role: None)

    scheduler.delete_workers("rollout", reverse_order=True)

    assert events == [
        "dependents:rollout:reverse=True",
        "dependents:proxy:reverse=True",
        "kill:proxy:reverse=True",
        "terminate:rollout:{'reverse_order': True}",
    ]
    assert scheduler._workers == {}
    assert scheduler._colocated_roles == {}


def test_direct_owner_delete_preserves_owner_when_alias_cleanup_fails(monkeypatch, tmp_path):
    module, scheduler, _processes = _scheduler(monkeypatch, tmp_path, node_count=1)
    owner = object()
    alias = object()
    scheduler._workers = {"rollout": [owner], "proxy": [alias]}
    scheduler._colocated_roles = {"proxy": "rollout"}
    terminated = []

    def cleanup_dependents(instance, role, *, reverse_order=False):
        if role == "rollout":
            instance.delete_workers("proxy", reverse_order=reverse_order)
        return ()

    monkeypatch.setattr(module, "cleanup_dependent_aliases_before_owner", cleanup_dependents)
    monkeypatch.setattr(
        module,
        "cleanup_forked_alias_workers",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("attestation failed")),
    )
    monkeypatch.setattr(
        scheduler,
        "_terminate_role_process",
        lambda role, **kwargs: terminated.append((role, kwargs)),
    )

    with pytest.raises(RuntimeError, match="attestation failed"):
        scheduler.delete_workers("rollout")

    assert terminated == []
    assert scheduler._workers == {"rollout": [owner], "proxy": [alias]}
    assert scheduler._colocated_roles == {"proxy": "rollout"}


def test_duplicate_role_does_not_consume_more_nodes(monkeypatch, tmp_path):
    _module, scheduler, _processes = _scheduler(monkeypatch, tmp_path, node_count=2)
    job = _separated_job("actor", 8, FakeSchedulingSpec(cpu=2, gpu=1, mem=8))
    scheduler.create_workers(job)

    with pytest.raises(Exception, match="already exists"):
        scheduler.create_workers(job)

    assert scheduler._node_owners == {"node-00": "actor"}


def test_delete_and_recreate_role_keeps_other_role_live(monkeypatch, tmp_path):
    _module, scheduler, _processes = _scheduler(monkeypatch, tmp_path, node_count=2)
    scheduler.name_resolve_config.nfs_record_root = str(tmp_path / "name-resolve")
    scheduler.create_workers(_separated_job("actor", 8, FakeSchedulingSpec(cpu=2, gpu=1, mem=8)))
    scheduler.create_workers(_separated_job("rollout", 1, FakeSchedulingSpec(cpu=8, gpu=8, mem=8)))
    first_generation = scheduler._role_generations["actor"]
    rollout_generation = scheduler._role_generations["rollout"]
    rollout_process = scheduler._role_processes["rollout"]

    def terminate(role, **_kwargs):
        scheduler._role_processes.pop(role, None)

    monkeypatch.setattr(scheduler, "_terminate_role_process", terminate)

    scheduler.delete_workers("actor", reverse_order=True)

    assert scheduler._node_owners == {"node-01": "rollout"}
    assert scheduler._role_processes["rollout"] is rollout_process
    assert scheduler._role_generations["rollout"] == rollout_generation

    scheduler.create_workers(_separated_job("actor", 8, FakeSchedulingSpec(cpu=2, gpu=1, mem=8)))

    assert scheduler._role_generations["actor"] != first_generation
    assert scheduler._node_owners == {"node-01": "rollout", "node-00": "actor"}
    assert scheduler._role_processes["rollout"] is rollout_process


def test_discovery_lifecycle_lock_is_stable_across_pbs_job_ids(monkeypatch, tmp_path):
    monkeypatch.setenv("PBS_JOBID", "first-job.pbs1")
    record_root = tmp_path / "name-resolve"
    _module, first, _processes = _scheduler(
        monkeypatch,
        tmp_path / "state-a",
        node_count=1,
        name_resolve_root=record_root,
    )
    first.create_workers(_separated_job("actor", 8, FakeSchedulingSpec(cpu=2, gpu=1, mem=8)))
    first_manifest = json.loads(first._role_manifests["actor"].read_text())
    first._release_trial_lifecycle_lock()

    monkeypatch.setenv("PBS_JOBID", "replacement-job.pbs1")
    _module, replacement, _processes = _scheduler(
        monkeypatch,
        tmp_path / "state-b",
        node_count=1,
        name_resolve_root=record_root,
    )
    replacement.create_workers(_separated_job("actor", 8, FakeSchedulingSpec(cpu=2, gpu=1, mem=8)))
    replacement_manifest = json.loads(replacement._role_manifests["actor"].read_text())

    first_lock = first_manifest["discovery"]["lifecycle_lock_path"]
    replacement_lock = replacement_manifest["discovery"]["lifecycle_lock_path"]
    assert first_lock == replacement_lock
    assert Path(first_lock).parent == record_root / getpass.getuser() / ".pbs-prealloc-locks"
    assert Path(first_lock).name.startswith("role-")
    assert Path(first_manifest["discovery"]["trial_lifecycle_lock_path"]).name.startswith("trial-")
    assert first_manifest["generation"] != replacement_manifest["generation"]


def test_private_lifecycle_lock_rejects_symlink_directory_and_file(monkeypatch, tmp_path):
    module = _load_scheduler_module(monkeypatch)

    real_lock_dir = tmp_path / "real-locks"
    real_lock_dir.mkdir(mode=0o700)
    lock_parent = tmp_path / "name-resolve" / getpass.getuser() / ".pbs-prealloc-locks"
    lock_parent.parent.mkdir(parents=True)
    lock_parent.symlink_to(real_lock_dir, target_is_directory=True)
    with pytest.raises(PermissionError, match="unsafe PBS lifecycle lock directory"):
        module._open_private_lock(lock_parent / "trial-dir-link.lock")

    lock_parent.unlink()
    lock_parent.mkdir(mode=0o700)
    target = tmp_path / "target.lock"
    target.write_text("must-not-change")
    target.chmod(0o600)
    linked_lock = lock_parent / "trial-file-link.lock"
    linked_lock.symlink_to(target)
    with pytest.raises(OSError):
        module._open_private_lock(linked_lock)
    assert target.read_text() == "must-not-change"


def test_private_lifecycle_lock_rejects_nonprivate_directory(monkeypatch, tmp_path):
    module = _load_scheduler_module(monkeypatch)
    lock_parent = tmp_path / "name-resolve" / getpass.getuser() / ".pbs-prealloc-locks"
    lock_parent.mkdir(mode=0o755, parents=True)
    lock_parent.chmod(0o755)

    with pytest.raises(PermissionError, match="unsafe PBS lifecycle lock directory"):
        module._open_private_lock(lock_parent / "trial.lock")


def test_constructor_mutex_closes_nonatomic_activity_lock_conversion(
    monkeypatch,
    tmp_path,
):
    module = _load_scheduler_module(monkeypatch)
    monkeypatch.setenv("PLATOON_AREAL_PBS_TRIAL_LOCK_TIMEOUT", "5")
    record_root = tmp_path / "name-resolve"
    trial_root = record_root / getpass.getuser() / "exp" / "trial"
    sentinel = trial_root / "constructor-owner"
    fake_base = module.PreallocatedPBSScheduler.__mro__[2]
    original_base_init = fake_base.__init__
    clear_count = 0
    conversion_started = threading.Event()
    allow_conversion = threading.Event()
    original_flock = module.fcntl.flock

    def clearing_base_init(instance, *args, **kwargs):
        nonlocal clear_count
        original_base_init(instance, *args, **kwargs)
        clear_count += 1
        shutil.rmtree(trial_root, ignore_errors=True)
        trial_root.mkdir(parents=True)
        sentinel.write_text(f"constructor-{clear_count}")

    def deliberately_nonatomic_flock(descriptor, operation):
        if threading.current_thread().name == "first-constructor" and operation == module.fcntl.LOCK_SH:
            # Model filesystems where EX -> SH conversion temporarily drops
            # ownership. The separate constructor mutex must still exclude a
            # replacement from the upstream recursive clear.
            original_flock(descriptor, module.fcntl.LOCK_UN)
            conversion_started.set()
            if not allow_conversion.wait(timeout=5):
                raise TimeoutError("test did not release simulated lock conversion")
        return original_flock(descriptor, operation)

    monkeypatch.setattr(fake_base, "__init__", clearing_base_init)
    monkeypatch.setattr(module.fcntl, "flock", deliberately_nonatomic_flock)

    def construct(results, errors):
        try:
            results.append(
                module.PreallocatedPBSScheduler(
                    experiment_name="exp",
                    trial_name="trial",
                    fileroot=str(tmp_path),
                    name_resolve_type="nfs",
                    nfs_record_root=str(record_root),
                )
            )
        except BaseException as error:
            errors.append(error)

    first_results: list = []
    first_errors: list[BaseException] = []
    first_thread = threading.Thread(
        target=construct,
        args=(first_results, first_errors),
        name="first-constructor",
    )
    first_thread.start()
    assert conversion_started.wait(timeout=5)
    assert sentinel.read_text() == "constructor-1"

    second_results: list = []
    second_errors: list[BaseException] = []
    second_thread = threading.Thread(
        target=construct,
        args=(second_results, second_errors),
        name="second-constructor",
    )
    second_thread.start()
    time.sleep(0.2)
    assert second_thread.is_alive()
    assert clear_count == 1
    assert sentinel.read_text() == "constructor-1"

    allow_conversion.set()
    first_thread.join(timeout=5)
    assert not first_thread.is_alive()
    assert first_errors == []
    assert len(first_results) == 1
    # The second constructor now owns the constructor mutex but must still
    # wait on the first controller's shared activity lock.
    time.sleep(0.2)
    assert second_thread.is_alive()
    assert clear_count == 1
    first_results[0]._release_trial_lifecycle_lock()
    second_thread.join(timeout=5)
    assert not second_thread.is_alive()
    assert second_errors == []
    assert len(second_results) == 1
    assert clear_count == 2
    assert sentinel.read_text() == "constructor-2"
    second_results[0]._release_trial_lifecycle_lock()


def test_replacement_constructor_waits_for_old_launcher_cleanup(monkeypatch, tmp_path):
    module = _load_scheduler_module(monkeypatch)
    role_launcher = _load_module(
        "platoon_pbs_role_launcher_constructor_race_test",
        REPO_ROOT / "platoon/train/areal/pbs_role_launcher.py",
    )
    monkeypatch.setenv("PLATOON_AREAL_PBS_CONTAINER_RUNTIME", "host")
    monkeypatch.setenv("PLATOON_AREAL_PBS_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("PLATOON_AREAL_PBS_TRIAL_LOCK_TIMEOUT", "5")
    record_root = tmp_path / "name-resolve"
    fake_base = module.PreallocatedPBSScheduler.__mro__[2]
    original_base_init = fake_base.__init__
    constructor_clear = threading.Event()

    def clearing_base_init(instance, *args, **kwargs):
        original_base_init(instance, *args, **kwargs)
        instance.experiment_name = str(kwargs["experiment_name"])
        instance.trial_name = str(kwargs["trial_name"])
        instance.fileroot = str(kwargs["fileroot"])
        instance.name_resolve_config.nfs_record_root = str(kwargs["nfs_record_root"])
        trial_root = (
            Path(kwargs["nfs_record_root"]) / getpass.getuser() / instance.experiment_name / instance.trial_name
        )
        shutil.rmtree(trial_root, ignore_errors=True)
        constructor_clear.set()

    monkeypatch.setattr(fake_base, "__init__", clearing_base_init)

    def construct():
        return module.PreallocatedPBSScheduler(
            experiment_name="exp",
            trial_name="trial",
            fileroot=str(tmp_path),
            name_resolve_type="nfs",
            nfs_record_root=str(record_root),
        )

    first = construct()
    old_manifest_path, *_rest = first._role_manifest(
        "actor",
        8,
        FakeSchedulingSpec(cpu=2, gpu=1, mem=8),
        [module.PBSNode("node-00", 0)],
        8,
    )
    old_manifest = json.loads(old_manifest_path.read_text())
    old_entry = Path(old_manifest["discovery"]["entry_paths"][0])
    old_entry.parent.mkdir(parents=True, exist_ok=True)
    old_entry.write_text("old-address")

    # Model a launcher that has spawned and is deliberately delayed in cleanup.
    old_trial_lock = role_launcher._acquire_trial_lifecycle_lock(old_manifest)
    old_role_lock = role_launcher._acquire_lifecycle_lock(old_manifest)
    stale_role_view = Path(old_manifest["discovery"]["lifecycle_lock_path"]).open("r")
    constructor_clear.clear()
    replacement: list = []
    replacement_errors: list[BaseException] = []

    def construct_replacement() -> None:
        try:
            replacement.append(construct())
        except BaseException as error:
            replacement_errors.append(error)

    thread = threading.Thread(target=construct_replacement)
    thread.start()
    try:
        time.sleep(0.2)
        assert thread.is_alive()
        assert not constructor_clear.is_set()
        assert old_entry.read_text() == "old-address"

        # Controller death releases its shared ownership, but the new
        # constructor remains blocked on the old launcher's shared ownership.
        first._release_trial_lifecycle_lock()
        time.sleep(0.2)
        assert thread.is_alive()
        assert not constructor_clear.is_set()
        assert old_entry.read_text() == "old-address"

        assert (
            role_launcher._cleanup_discovery_records(
                old_manifest,
                old_role_lock,
            )
            == 1
        )
        old_role_lock.close()
        old_trial_lock.close()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert replacement_errors == []
        assert constructor_clear.is_set()
        second = replacement[0]

        new_manifest_path, *_rest = second._role_manifest(
            "actor",
            8,
            FakeSchedulingSpec(cpu=2, gpu=1, mem=8),
            [module.PBSNode("node-00", 0)],
            8,
        )
        new_manifest = json.loads(new_manifest_path.read_text())
        new_trial_lock = role_launcher._acquire_trial_lifecycle_lock(new_manifest)
        new_role_lock = role_launcher._acquire_lifecycle_lock(new_manifest)
        new_entry = Path(new_manifest["discovery"]["entry_paths"][0])
        new_entry.parent.mkdir(parents=True, exist_ok=True)
        new_entry.write_text("new-address")
        try:
            with pytest.raises(RuntimeError, match="generation changed"):
                role_launcher._cleanup_discovery_records(
                    old_manifest,
                    stale_role_view,
                )
            assert new_entry.read_text() == "new-address"
        finally:
            new_role_lock.close()
            new_trial_lock.close()
            second._release_trial_lifecycle_lock()
    finally:
        if not old_role_lock.closed:
            old_role_lock.close()
        if not old_trial_lock.closed:
            old_trial_lock.close()
        stale_role_view.close()
        first._release_trial_lifecycle_lock()
        thread.join(timeout=5)


def test_delete_waits_for_launcher_cleanup_confirmation_before_releasing_nodes(monkeypatch, tmp_path):
    _module, scheduler, _processes = _scheduler(monkeypatch, tmp_path, node_count=1)
    scheduler.create_workers(_separated_job("actor", 8, FakeSchedulingSpec(cpu=2, gpu=1, mem=8)))
    status_path = scheduler._role_status_paths["actor"]
    generation = scheduler._role_generations["actor"]
    events: list[str] = []

    class CleanupAwareProcess:
        pid = 9876
        status = None

        def poll(self):
            return self.status

        def terminate(self):
            events.append("terminate-launcher")

        def wait(self, timeout=None):
            del timeout
            events.append("remote-cleanup-complete")
            status_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "generation": generation,
                        "complete": True,
                        "cleanup_failures": [],
                    }
                )
            )
            status_path.chmod(0o600)
            self.status = 0
            return 0

    scheduler._role_processes["actor"] = CleanupAwareProcess()
    original_release = scheduler._release_nodes

    def release(role):
        events.append("release-nodes")
        original_release(role)

    monkeypatch.setattr(scheduler, "_release_nodes", release)

    scheduler.delete_workers("actor")

    assert events == [
        "terminate-launcher",
        "remote-cleanup-complete",
        "release-nodes",
    ]
    assert scheduler._node_owners == {}


def test_cleanup_failure_keeps_role_and_nodes_reserved(monkeypatch, tmp_path):
    _module, scheduler, _processes = _scheduler(monkeypatch, tmp_path, node_count=1)
    scheduler.create_workers(_separated_job("actor", 8, FakeSchedulingSpec(cpu=2, gpu=1, mem=8)))
    status_path = scheduler._role_status_paths["actor"]
    generation = scheduler._role_generations["actor"]

    class FailedCleanupProcess:
        pid = 9877
        status = None

        def poll(self):
            return self.status

        def terminate(self):
            pass

        def wait(self, timeout=None):
            del timeout
            status_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "generation": generation,
                        "complete": True,
                        "cleanup_failures": [{"hostname": "node-00", "error": "pbsdsh exited 1"}],
                    }
                )
            )
            status_path.chmod(0o600)
            self.status = 70
            return 70

    scheduler._role_processes["actor"] = FailedCleanupProcess()

    with pytest.raises(RuntimeError, match="nodes remain reserved"):
        scheduler.delete_workers("actor")

    assert scheduler._node_owners == {"node-00": "actor"}
    assert "actor" in scheduler._workers
    assert "actor" in scheduler._role_manifests
    assert "actor" in scheduler._role_processes


def test_single_role_reverse_deletion_requests_rank_zero_last(monkeypatch, tmp_path):
    _module, scheduler, _processes = _scheduler(monkeypatch, tmp_path, node_count=1)
    scheduler.create_workers(_separated_job("actor", 8, FakeSchedulingSpec(cpu=2, gpu=1, mem=8)))
    status_path = scheduler._role_status_paths["actor"]
    termination_path = scheduler._role_termination_paths["actor"]
    generation = scheduler._role_generations["actor"]
    observed_request: dict = {}

    class ReverseAwareProcess:
        pid = 9878
        status = None

        def poll(self):
            return self.status

        def terminate(self):
            observed_request.update(json.loads(termination_path.read_text()))

        def wait(self, timeout=None):
            del timeout
            status_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "generation": generation,
                        "complete": True,
                        "cleanup_failures": [],
                    }
                )
            )
            status_path.chmod(0o600)
            self.status = 0
            return 0

    scheduler._role_processes["actor"] = ReverseAwareProcess()

    scheduler.delete_workers("actor", reverse_order=True)

    assert observed_request == {
        "version": 1,
        "role": "actor",
        "generation": generation,
        "reverse_order": True,
    }
    assert scheduler._node_owners == {}
