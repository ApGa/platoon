"""Tests for PBS host launcher and identity-safe supervisor cleanup."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pbs_role_launcher = _load_module(
    "platoon_pbs_role_launcher_test",
    "platoon/train/areal/pbs_role_launcher.py",
)
pbs_role_supervisor = _load_module(
    "platoon_pbs_role_supervisor_test",
    "platoon/train/areal/pbs_role_supervisor.py",
)


def _write_private_json(path: Path, value: dict) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream)


def _with_discovery(
    tmp_path: Path,
    manifest: dict,
    *,
    replicas: int = 1,
) -> dict:
    """Add the exact stable AReaL NFS paths required by the launcher."""

    role = str(manifest["role"])
    record_root = (tmp_path / "name-resolve").resolve()
    role_root = record_root / "user" / "exp" / "trial" / "workers" / role
    identity = {
        "version": 1,
        "record_root": str(record_root),
        "registry_user": "user",
        "experiment_name": "exp",
        "trial_name": "trial",
    }

    def digest(*, digest_role: str | None = None) -> str:
        value = dict(identity)
        if digest_role is not None:
            value["role"] = digest_role
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    lock_root = record_root / str(identity["registry_user"]) / ".pbs-prealloc-locks"
    lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_root.chmod(0o700)
    trial_lock = lock_root / f"trial-{digest()}.lock"
    constructor_lock = lock_root / f"constructor-{digest()}.lock"
    _write_private_json(
        constructor_lock,
        {
            "version": 1,
            "kind": "pbs-constructor-lifecycle",
            "identity": identity,
        },
    )
    _write_private_json(
        trial_lock,
        {
            "version": 1,
            "kind": "pbs-trial-lifecycle",
            "identity": identity,
        },
    )
    manifest.update(
        {
            "replicas": replicas,
            "discovery": {
                "type": "nfs",
                "record_root": str(record_root),
                "role_root": str(role_root),
                "entry_paths": [str(role_root / str(index) / "ENTRY") for index in range(replicas)],
                "lock_identity": identity,
                "constructor_lifecycle_lock_path": str(constructor_lock),
                "trial_lifecycle_lock_path": str(trial_lock),
                "lifecycle_lock_path": str(lock_root / f"role-{digest(digest_role=role)}.lock"),
            },
        }
    )
    return manifest


def test_launcher_private_lock_rejects_symlink_file(tmp_path):
    lock_root = tmp_path / "locks"
    lock_root.mkdir(mode=0o700)
    target = tmp_path / "target.lock"
    target.write_text("must-not-change")
    target.chmod(0o600)
    linked_lock = lock_root / "role.lock"
    linked_lock.symlink_to(target)

    with pytest.raises(OSError):
        pbs_role_launcher._open_private_lock(linked_lock, create=True)
    assert target.read_text() == "must-not-change"


def test_container_environment_forwards_explicit_values_and_rank():
    environment = pbs_role_supervisor._containerized_environment(
        {
            "PATH": "/usr/bin",
            "APPTAINER_BIND": "/home:/home",
            "SINGULARITY_BINDPATH": "/secret:/secret",
            "APPTAINERENV_OLD_SECRET": "leak",
            "SINGULARITYENV_CUDA_VISIBLE_DEVICES": "0",
        },
        {
            "MODEL_PATH": "/groups/model with spaces",
            "SINGULARITY_BIND": "/explicit-bypass:/mnt",
            "APPTAINERENV_INJECTED": "bypass",
        },
        {"CUDA_VISIBLE_DEVICES": "4,5", "PLATOON_WORKER_INDEX": "17"},
    )

    assert environment["MODEL_PATH"] == "/groups/model with spaces"
    assert environment["SINGULARITYENV_MODEL_PATH"] == "/groups/model with spaces"
    assert environment["APPTAINERENV_CUDA_VISIBLE_DEVICES"] == "4,5"
    assert environment["SINGULARITYENV_PLATOON_WORKER_INDEX"] == "17"
    assert "APPTAINER_BIND" not in environment
    assert "SINGULARITY_BINDPATH" not in environment
    assert "APPTAINERENV_OLD_SECRET" not in environment
    assert environment["SINGULARITYENV_CUDA_VISIBLE_DEVICES"] == "4,5"
    assert "APPTAINERENV_INJECTED" not in environment
    assert "SINGULARITYENV_SINGULARITY_BIND" not in environment


def _gpu_uuid(index: int) -> str:
    return f"GPU-{index + 1:08x}-1111-2222-3333-{index + 1:012x}"


def test_nvidia_gpu_uuid_map_uses_one_bounded_absolute_query(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return types.SimpleNamespace(
            stdout=f"7, {_gpu_uuid(7)}\n0, {_gpu_uuid(0)}\n",
            stderr="",
        )

    monkeypatch.setattr(pbs_role_supervisor.subprocess, "run", run)

    assert pbs_role_supervisor._nvidia_gpu_uuid_map("/usr/bin/nvidia-smi") == {
        0: _gpu_uuid(0),
        7: _gpu_uuid(7),
    }
    assert calls == [
        (
            [
                "/usr/bin/nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            {
                "check": True,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "timeout": 15,
            },
        )
    ]


@pytest.mark.parametrize(
    "output,match",
    [
        ("", "returned no devices"),
        ("0, not-a-uuid\n", "invalid NVIDIA GPU UUID"),
        (f"0, {_gpu_uuid(0)}\n0, {_gpu_uuid(1)}\n", "Duplicate or invalid NVIDIA GPU index"),
        (f"0, {_gpu_uuid(0)}\n1, {_gpu_uuid(0)}\n", "Duplicate or invalid NVIDIA GPU UUID"),
        (f"0, {_gpu_uuid(0)}, extra\n", "Malformed nvidia-smi GPU mapping"),
    ],
)
def test_nvidia_gpu_uuid_map_rejects_malformed_or_duplicate_inventory(monkeypatch, output, match):
    monkeypatch.setattr(
        pbs_role_supervisor.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(stdout=output, stderr=""),
    )

    with pytest.raises(RuntimeError, match=match):
        pbs_role_supervisor._nvidia_gpu_uuid_map("/usr/bin/nvidia-smi")


def test_nvidia_gpu_uuid_map_fails_closed_on_timeout(monkeypatch):
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", 15)

    monkeypatch.setattr(pbs_role_supervisor.subprocess, "run", timeout)

    with pytest.raises(RuntimeError, match="timed out after 15 seconds"):
        pbs_role_supervisor._nvidia_gpu_uuid_map("/usr/bin/nvidia-smi")


def test_guard_launch_does_not_preseed_distributed_engine_ranks():
    """AReaL supplies engine ranks only after each RPC guard is registered.

    In particular, an actor guard may have host-local placement rank seven but
    see only one CUDA device.  Exporting ``LOCAL_RANK=7`` before engine setup
    would make eager imports address a nonexistent visible device.
    """

    launcher_rank_environment = {
        "PLATOON_WORKER_INDEX": "17",
        "PLATOON_LOCAL_RANK": "7",
        "PLATOON_ROLE": "actor",
        "CUDA_VISIBLE_DEVICES": "7",
    }
    environment = pbs_role_supervisor._containerized_environment(
        {"PATH": "/usr/bin"},
        {},
        launcher_rank_environment,
    )

    assert environment["PLATOON_LOCAL_RANK"] == "7"
    assert environment["CUDA_VISIBLE_DEVICES"] == "7"
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        assert name not in environment
        assert f"SINGULARITYENV_{name}" not in environment


def test_cleanup_only_signals_matching_pid_start_times(monkeypatch, tmp_path):
    control_path = tmp_path / "actor.control.json"
    hostname = socket.gethostname()
    _write_private_json(
        control_path,
        {
            "version": 1,
            "role": "actor",
            "hostname": hostname,
            "supervisor": {"pid": 100, "start_time": 10, "process_group": False},
            "children": [
                {"pid": 200, "start_time": 20, "process_group": True},
                {"pid": 300, "start_time": 30, "process_group": True},
            ],
        },
    )
    starts = {100: 999, 200: 20, 300: 999}
    monkeypatch.setattr(
        pbs_role_supervisor,
        "process_start_time",
        lambda pid: starts.get(pid),
    )
    process_signals: list[tuple[int, signal.Signals]] = []
    group_signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        pbs_role_supervisor.os,
        "kill",
        lambda pid, sig: process_signals.append((pid, sig)),
    )

    def killpg(pid, sig):
        group_signals.append((pid, sig))
        if sig == signal.SIGKILL:
            starts[pid] = None

    monkeypatch.setattr(pbs_role_supervisor.os, "killpg", killpg)

    assert pbs_role_supervisor.cleanup_host(control_path, grace_seconds=0) is True

    assert process_signals == []  # stale/reused supervisor PID is untouched
    assert group_signals == [(200, signal.SIGTERM), (200, signal.SIGKILL)]
    assert not control_path.exists()


def test_cleanup_rejects_control_record_for_another_host(tmp_path):
    control_path = tmp_path / "actor.control.json"
    _write_private_json(
        control_path,
        {
            "version": 1,
            "role": "actor",
            "hostname": "definitely-a-different-host",
            "children": [],
        },
    )

    with pytest.raises(RuntimeError, match="belongs to"):
        pbs_role_supervisor.cleanup_host(control_path, grace_seconds=0)

    assert control_path.exists()


def test_delayed_cleanup_cannot_touch_replacement_generation(monkeypatch, tmp_path):
    control_path = tmp_path / "actor.control.json"
    _write_private_json(
        control_path,
        {
            "version": 1,
            "role": "actor",
            "generation": "new-generation",
            "hostname": socket.gethostname(),
            "supervisor": {"pid": 100, "start_time": 10},
            "children": [{"pid": 200, "start_time": 20}],
        },
    )
    monkeypatch.setattr(
        pbs_role_supervisor,
        "_signal_identity",
        lambda *_args, **_kwargs: pytest.fail("replacement process was signalled"),
    )

    with pytest.raises(RuntimeError, match="replacement generation"):
        pbs_role_supervisor.cleanup_host(
            control_path,
            grace_seconds=0,
            expected_generation="old-generation",
        )
    assert control_path.exists()


def test_cleanup_refuses_to_confirm_while_exact_identity_survives(monkeypatch, tmp_path):
    control_path = tmp_path / "actor.control.json"
    record = {
        "version": 1,
        "role": "actor",
        "generation": "generation-a",
        "hostname": socket.gethostname(),
        "children": [
            {
                "pid": 200,
                "start_time": 20,
                "process_group": True,
                "worker_index": 0,
            }
        ],
    }
    _write_private_json(control_path, record)
    monkeypatch.setattr(
        pbs_role_supervisor,
        "terminate_identities",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        pbs_role_supervisor,
        "identity_is_live",
        lambda identity: identity.get("pid") == 200,
    )

    with pytest.raises(RuntimeError, match="left live process identities"):
        pbs_role_supervisor.cleanup_host(
            control_path,
            grace_seconds=0,
            expected_generation="generation-a",
        )

    assert json.loads(control_path.read_text()) == record


def test_group_cleanup_reaches_descendants_after_leader_exit(monkeypatch):
    identity = {"pid": 2468, "start_time": 123, "process_group": True}
    monkeypatch.setattr(pbs_role_supervisor, "_process_details", lambda _pid: None)
    monkeypatch.setattr(
        pbs_role_supervisor,
        "_process_group_has_live_members",
        lambda _pgid, **_kwargs: True,
    )
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        pbs_role_supervisor.os,
        "killpg",
        lambda pgid, sig: signals.append((pgid, sig)),
    )

    pbs_role_supervisor.terminate_identities([identity], grace_seconds=0)

    assert signals == [
        (2468, signal.SIGTERM),
        (2468, signal.SIGKILL),
    ]


def test_reverse_identity_cleanup_signals_worker_zero_last(monkeypatch):
    identities = [{"pid": 100 + rank, "start_time": rank, "worker_index": rank} for rank in range(3)]
    monkeypatch.setattr(pbs_role_supervisor, "identity_is_live", lambda _item: True)
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        pbs_role_supervisor,
        "_signal_identity",
        lambda item, sig: signals.append((item["worker_index"], sig)) or True,
    )

    pbs_role_supervisor.terminate_identities(
        identities,
        grace_seconds=0,
        reverse_order=True,
    )

    zero_first_signal = next(index for index, value in enumerate(signals) if value[0] == 0)
    assert all(rank != 0 for rank, _sig in signals[:zero_first_signal])
    assert {rank for rank, _sig in signals[:zero_first_signal]} == {1, 2}


def test_supervisor_reverse_order_default_and_explicit_request(tmp_path):
    request_path = tmp_path / "termination.json"
    manifest = {
        "version": 1,
        "role": "actor",
        "generation": "generation-a",
        "termination_reverse_order_default": True,
        "termination_request_path": str(request_path),
    }

    assert pbs_role_supervisor._requested_reverse_order(manifest) is True
    _write_private_json(
        request_path,
        {
            "version": 1,
            "role": "actor",
            "generation": "generation-a",
            "reverse_order": False,
        },
    )
    assert pbs_role_supervisor._requested_reverse_order(manifest) is False
    manifest.pop("termination_reverse_order_default")
    request_path.unlink()
    assert pbs_role_supervisor._requested_reverse_order(manifest) is False


def test_supervisor_signal_stops_spawning_remaining_workers(monkeypatch, tmp_path):
    manifest_path = tmp_path / "role.json"
    control_path = tmp_path / "actor.control.json"
    manifest = {
        "version": 1,
        "generation": "generation-stop-during-spawn",
        "role": "actor",
        "pbs_job_id": "",
        "worker_command": ["worker"],
        "worker_environment": {},
        "termination_grace_seconds": 0,
        "hosts": [
            {
                "hostname": socket.gethostname(),
                "vnode_index": 0,
                "control_path": str(control_path),
                "workers": [
                    {"worker_index": 0, "local_rank": 0, "gpu_ids": [0]},
                    {"worker_index": 1, "local_rank": 1, "gpu_ids": [1]},
                ],
            }
        ],
    }
    _write_private_json(manifest_path, manifest)
    handlers: dict[signal.Signals, object] = {}
    monkeypatch.setattr(
        pbs_role_supervisor.signal,
        "signal",
        lambda sig, handler: handlers.__setitem__(sig, handler),
    )
    monkeypatch.setattr(pbs_role_supervisor, "cleanup_host", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        pbs_role_supervisor,
        "process_identity",
        lambda pid, **_kwargs: {
            "pid": pid,
            "start_time": pid,
            "process_group": True,
        },
    )
    monkeypatch.setattr(
        pbs_role_supervisor,
        "terminate_identities",
        lambda *_args, **_kwargs: None,
    )

    class Child:
        pid = 8000
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            del timeout
            self.returncode = 0
            return 0

    spawn_count = 0

    def popen(*_args, **_kwargs):
        nonlocal spawn_count
        spawn_count += 1
        if spawn_count == 1:
            handlers[signal.SIGTERM](signal.SIGTERM, None)
        return Child()

    monkeypatch.setattr(pbs_role_supervisor.subprocess, "Popen", popen)

    assert pbs_role_supervisor.supervise_host(manifest_path, 0) == 0
    assert spawn_count == 1


@pytest.mark.parametrize(
    "role,assignments",
    [
        (
            "actor",
            [{"worker_index": index, "local_rank": index, "gpu_ids": [index]} for index in range(8)],
        ),
        (
            "rollout",
            [{"worker_index": 0, "local_rank": 0, "gpu_ids": list(range(8))}],
        ),
    ],
)
def test_nvccli_supervisor_maps_physical_uuids_to_inner_ordinals(
    monkeypatch,
    tmp_path,
    role,
    assignments,
):
    manifest_path = tmp_path / f"{role}.json"
    control_path = tmp_path / f"{role}.control.json"
    manifest = {
        "version": 1,
        "generation": f"generation-{role}",
        "role": role,
        "pbs_job_id": "",
        "container_image": "/images/training.sif",
        "worker_command": ["singularity", "exec", "--nv", "--containall", "/images/training.sif", "true"],
        "worker_environment": {"NVIDIA_VISIBLE_DEVICES": "worker-bypass"},
        "gpu_visibility": {
            "mode": "nvidia-container-cli",
            "nvidia_smi_binary": sys.executable,
        },
        "termination_grace_seconds": 0,
        "hosts": [
            {
                "hostname": socket.gethostname(),
                "vnode_index": 0,
                "control_path": str(control_path),
                "workers": assignments,
            }
        ],
    }
    _write_private_json(manifest_path, manifest)
    handlers = {}
    monkeypatch.setattr(
        pbs_role_supervisor.signal,
        "signal",
        lambda sig, handler: handlers.__setitem__(sig, handler),
    )
    monkeypatch.setattr(pbs_role_supervisor, "cleanup_host", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        pbs_role_supervisor,
        "process_identity",
        lambda pid, process_group=True: {
            "pid": pid,
            "start_time": pid,
            "process_group": process_group,
        },
    )
    monkeypatch.setattr(pbs_role_supervisor, "terminate_identities", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "inherited-bypass")
    monkeypatch.setenv("SINGULARITYENV_NVIDIA_VISIBLE_DEVICES", "prefixed-bypass")

    query_calls = 0
    query_finished = False

    def run(command, **kwargs):
        nonlocal query_calls, query_finished
        assert command[0] == sys.executable
        assert kwargs["timeout"] == 15
        query_calls += 1
        query_finished = True
        return types.SimpleNamespace(
            stdout="".join(f"{index}, {_gpu_uuid(index)}\n" for index in range(8)),
            stderr="",
        )

    monkeypatch.setattr(pbs_role_supervisor.subprocess, "run", run)
    environments = []

    class Child:
        returncode = None

        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            del timeout
            self.returncode = 0
            return 0

    def popen(_command, *, env, start_new_session):
        assert query_finished
        assert start_new_session is True
        environments.append(env)
        child = Child(9000 + len(environments))
        if len(environments) == len(assignments):
            handlers[signal.SIGTERM](signal.SIGTERM, None)
        return child

    monkeypatch.setattr(pbs_role_supervisor.subprocess, "Popen", popen)

    assert pbs_role_supervisor.supervise_host(manifest_path, 0) == 0
    assert query_calls == 1
    assert len(environments) == len(assignments)
    for assignment, environment in zip(assignments, environments, strict=True):
        gpu_ids = assignment["gpu_ids"]
        expected_ids = ",".join(str(value) for value in gpu_ids)
        expected_uuids = ",".join(_gpu_uuid(value) for value in gpu_ids)
        expected_inner = ",".join(str(value) for value in range(len(gpu_ids)))
        assert environment["NVIDIA_VISIBLE_DEVICES"] == expected_uuids
        assert environment["CUDA_VISIBLE_DEVICES"] == expected_inner
        assert environment["SINGULARITYENV_CUDA_VISIBLE_DEVICES"] == expected_inner
        assert environment["APPTAINERENV_CUDA_VISIBLE_DEVICES"] == expected_inner
        assert environment["PLATOON_ASSIGNED_GPU_IDS"] == expected_ids
        assert environment["PLATOON_ASSIGNED_GPU_UUIDS"] == expected_uuids
        assert environment["SINGULARITYENV_PLATOON_ASSIGNED_GPU_IDS"] == expected_ids
        assert environment["SINGULARITYENV_PLATOON_ASSIGNED_GPU_UUIDS"] == expected_uuids
        assert "SINGULARITYENV_NVIDIA_VISIBLE_DEVICES" not in environment
        assert "APPTAINERENV_NVIDIA_VISIBLE_DEVICES" not in environment


def test_nvccli_missing_assigned_gpu_fails_before_control_or_worker(monkeypatch, tmp_path):
    manifest_path = tmp_path / "actor.json"
    control_path = tmp_path / "actor.control.json"
    _write_private_json(
        manifest_path,
        {
            "version": 1,
            "generation": "generation-missing-gpu",
            "role": "actor",
            "pbs_job_id": "",
            "container_image": "/images/training.sif",
            "worker_command": ["worker"],
            "worker_environment": {},
            "gpu_visibility": {
                "mode": "nvidia-container-cli",
                "nvidia_smi_binary": sys.executable,
            },
            "termination_grace_seconds": 0,
            "hosts": [
                {
                    "hostname": socket.gethostname(),
                    "vnode_index": 0,
                    "control_path": str(control_path),
                    "workers": [{"worker_index": 0, "local_rank": 0, "gpu_ids": [7]}],
                }
            ],
        },
    )
    monkeypatch.setattr(pbs_role_supervisor, "cleanup_host", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        pbs_role_supervisor.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(stdout=f"0, {_gpu_uuid(0)}\n", stderr=""),
    )
    spawn_count = 0

    def popen(*_args, **_kwargs):
        nonlocal spawn_count
        spawn_count += 1
        raise AssertionError("worker must not start with an incomplete GPU inventory")

    monkeypatch.setattr(pbs_role_supervisor.subprocess, "Popen", popen)

    with pytest.raises(RuntimeError, match="did not report assigned physical GPU indexes: \\[7\\]"):
        pbs_role_supervisor.supervise_host(manifest_path, 0)
    assert spawn_count == 0
    assert not control_path.exists()


def test_launcher_uses_one_pbsdsh_rank_per_physical_host(tmp_path):
    manifest_path = tmp_path / "role.json"
    manifest = {
        "version": 1,
        "pbsdsh_binary": "/opt/pbsdsh",
        "python_binary": "/venv/bin/python",
        "supervisor_script": "/source/pbs_role_supervisor.py",
        "hosts": [
            {"hostname": "node-a", "vnode_index": 0, "control_path": "/tmp/a"},
            {"hostname": "node-b", "vnode_index": 8, "control_path": "/tmp/b"},
        ],
    }

    first = pbs_role_launcher.supervisor_command(manifest_path, 0, manifest)
    second = pbs_role_launcher.supervisor_command(manifest_path, 1, manifest)

    assert first[:4] == ["/opt/pbsdsh", "-n", "0", "--"]
    assert second[:4] == ["/opt/pbsdsh", "-n", "8", "--"]
    assert "/source/pbs_role_supervisor.py" in second


def test_reverse_launcher_shutdown_waits_for_last_host_before_rank_zero_host():
    events: list[str] = []

    class Process:
        def __init__(self, name):
            self.name = name
            self.status = None

        def poll(self):
            return self.status

        def terminate(self):
            events.append(f"terminate-{self.name}")

        def wait(self, timeout=None):
            del timeout
            events.append(f"wait-{self.name}")
            self.status = 0
            return 0

        def kill(self):
            self.status = -9

    processes = [Process("rank-zero-host"), Process("last-host")]

    pbs_role_launcher._stop_processes(
        processes,
        grace_seconds=1,
        reverse_order=True,
    )

    assert events == [
        "terminate-last-host",
        "wait-last-host",
        "terminate-rank-zero-host",
        "wait-rank-zero-host",
    ]


def test_launcher_rolls_back_started_supervisors_when_later_spawn_fails(monkeypatch, tmp_path):
    manifest_path = tmp_path / "role.json"
    manifest = _with_discovery(
        tmp_path,
        {
            "version": 1,
            "generation": "generation-a",
            "role": "actor",
            "pbsdsh_binary": "pbsdsh",
            "python_binary": "/venv/bin/python",
            "supervisor_script": "/source/pbs_role_supervisor.py",
            "termination_grace_seconds": 0,
            "hosts": [
                {"hostname": "node-a", "vnode_index": 0, "control_path": "/tmp/a"},
                {"hostname": "node-b", "vnode_index": 8, "control_path": "/tmp/b"},
            ],
        },
        replicas=2,
    )
    _write_private_json(manifest_path, manifest)
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600

    class FirstProcess:
        def __init__(self):
            self.status = None
            self.terminated = False

        def poll(self):
            return self.status

        def terminate(self):
            self.terminated = True
            self.status = -15

        def wait(self, timeout=None):
            return self.status

        def kill(self):
            self.status = -9

    first = FirstProcess()
    calls = 0

    def popen(_command):
        nonlocal calls
        calls += 1
        if calls == 1:
            return first
        raise OSError("pbsdsh spawn failed")

    monkeypatch.setattr(pbs_role_launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(
        pbs_role_launcher.subprocess,
        "run",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(OSError, match="spawn failed"):
        pbs_role_launcher.launch_role(manifest_path)

    assert first.terminated is True


def test_launcher_records_and_propagates_redundant_cleanup_failure(monkeypatch, tmp_path):
    manifest_path = tmp_path / "role.json"
    status_path = tmp_path / "role.launcher-status.json"
    manifest = _with_discovery(
        tmp_path,
        {
            "version": 1,
            "generation": "generation-b",
            "role": "actor",
            "pbsdsh_binary": "pbsdsh",
            "python_binary": "/venv/bin/python",
            "supervisor_script": "/source/pbs_role_supervisor.py",
            "launcher_status_path": str(status_path),
            "termination_grace_seconds": 0,
            "hosts": [
                {"hostname": "node-a", "vnode_index": 0, "control_path": "/tmp/a"},
                {"hostname": "node-b", "vnode_index": 8, "control_path": "/tmp/b"},
            ],
        },
        replicas=2,
    )
    _write_private_json(manifest_path, manifest)

    class FailedSupervisor:
        returncode = 5

        def poll(self):
            return self.returncode

    monkeypatch.setattr(
        pbs_role_launcher.subprocess,
        "Popen",
        lambda _command: FailedSupervisor(),
    )

    def cleanup_run(command, **_kwargs):
        vnode_index = command[2]
        return types.SimpleNamespace(returncode=9 if vnode_index == "8" else 0)

    monkeypatch.setattr(pbs_role_launcher.subprocess, "run", cleanup_run)

    assert pbs_role_launcher.launch_role(manifest_path) == 70
    status = json.loads(status_path.read_text())
    assert stat.S_IMODE(status_path.stat().st_mode) == 0o600
    assert status == {
        "cleanup_failures": [
            {
                "error": "cleanup command exited with status 9",
                "hostname": "node-b",
            }
        ],
        "cleanup_host_count": 2,
        "complete": True,
        "generation": "generation-b",
        "launcher_exception": None,
        "primary_status": 5,
        "role": "actor",
        "stale_discovery_records_removed": 0,
        "discovery_records_removed": 0,
        "version": 1,
    }


def test_launcher_parent_race_skips_spawns_and_runs_cleanup(monkeypatch, tmp_path):
    manifest_path = tmp_path / "role.json"
    status_path = tmp_path / "role.launcher-status.json"
    manifest = _with_discovery(
        tmp_path,
        {
            "version": 1,
            "generation": "generation-parent-race",
            "role": "actor",
            "launcher_status_path": str(status_path),
            "termination_grace_seconds": 0,
            "hosts": [
                {"hostname": "node-a", "vnode_index": 0, "control_path": "/tmp/a"},
            ],
        },
    )
    _write_private_json(manifest_path, manifest)
    popen_calls: list[list[str]] = []
    monkeypatch.setattr(
        pbs_role_launcher,
        "_arm_parent_death_signal",
        lambda _parent: False,
    )
    monkeypatch.setattr(
        pbs_role_launcher.subprocess,
        "Popen",
        lambda command: popen_calls.append(command),
    )
    cleanup_calls: list[list[str]] = []
    monkeypatch.setattr(
        pbs_role_launcher.subprocess,
        "run",
        lambda command, **_kwargs: cleanup_calls.append(command) or types.SimpleNamespace(returncode=0),
    )

    assert (
        pbs_role_launcher.launch_role(
            manifest_path,
            expected_parent_pid=12345,
        )
        == 0
    )
    assert popen_calls == []
    assert len(cleanup_calls) == 1
    assert json.loads(status_path.read_text())["cleanup_failures"] == []


def test_discovery_cleanup_removes_only_exact_role_entries(tmp_path):
    actor = _with_discovery(
        tmp_path,
        {
            "version": 1,
            "generation": "actor-generation",
            "role": "actor",
        },
        replicas=2,
    )
    actor_entries = [Path(value) for value in actor["discovery"]["entry_paths"]]
    for index, entry in enumerate(actor_entries):
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_text(f"10.0.0.1:{1000 + index}")
    actor_extra = actor_entries[0].parents[1] / "999" / "ENTRY"
    actor_extra.parent.mkdir(parents=True)
    actor_extra.write_text("must-stay")
    rollout_entry = actor_entries[0].parents[2] / "rollout" / "0" / "ENTRY"
    rollout_entry.parent.mkdir(parents=True)
    rollout_entry.write_text("live-rollout")

    lifecycle_lock = pbs_role_launcher._acquire_lifecycle_lock(actor)
    try:
        assert pbs_role_launcher._cleanup_discovery_records(actor, lifecycle_lock) == 2
    finally:
        lifecycle_lock.close()

    assert not any(entry.exists() for entry in actor_entries)
    assert actor_extra.read_text() == "must-stay"
    assert rollout_entry.read_text() == "live-rollout"


def test_discovery_cleanup_removes_empty_self_cleaned_index_and_role_dirs(tmp_path):
    actor = _with_discovery(
        tmp_path,
        {
            "version": 1,
            "generation": "actor-self-clean-generation",
            "role": "actor",
        },
    )
    entry = Path(actor["discovery"]["entry_paths"][0])
    entry.parent.mkdir(parents=True)
    role_root = entry.parents[1]

    lifecycle_lock = pbs_role_launcher._acquire_lifecycle_lock(actor)
    try:
        assert pbs_role_launcher._cleanup_discovery_records(actor, lifecycle_lock) == 0
    finally:
        lifecycle_lock.close()

    assert not entry.parent.exists()
    assert not role_root.exists()


def test_discovery_cleanup_rejects_symlink_worker_index_directory(tmp_path):
    actor = _with_discovery(
        tmp_path,
        {
            "version": 1,
            "generation": "actor-symlink-generation",
            "role": "actor",
        },
    )
    entry = Path(actor["discovery"]["entry_paths"][0])
    entry.parents[1].mkdir(parents=True)
    outside = tmp_path / "outside-index"
    outside.mkdir()
    entry.parent.symlink_to(outside, target_is_directory=True)

    lifecycle_lock = pbs_role_launcher._acquire_lifecycle_lock(actor)
    try:
        with pytest.raises(PermissionError, match="unsafe PBS discovery directory"):
            pbs_role_launcher._cleanup_discovery_records(actor, lifecycle_lock)
    finally:
        lifecycle_lock.close()

    assert entry.parent.is_symlink()
    assert outside.is_dir()


def test_old_discovery_cleanup_cannot_delete_recreated_generation(tmp_path):
    base = {
        "version": 1,
        "role": "actor",
    }
    old = _with_discovery(
        tmp_path,
        {**base, "generation": "old-generation"},
    )
    new = _with_discovery(
        tmp_path,
        {**base, "generation": "new-generation"},
    )
    entry = Path(old["discovery"]["entry_paths"][0])
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("old-address")
    old_lock = pbs_role_launcher._acquire_lifecycle_lock(old)
    stale_view = Path(old["discovery"]["lifecycle_lock_path"]).open("r")
    replacement_acquired = threading.Event()
    release_replacement = threading.Event()
    errors: list[BaseException] = []

    def run_replacement() -> None:
        try:
            new_lock = pbs_role_launcher._acquire_lifecycle_lock(new)
            try:
                entry.parent.mkdir(parents=True, exist_ok=True)
                entry.write_text("new-address")
                replacement_acquired.set()
                assert release_replacement.wait(timeout=5)
            finally:
                new_lock.close()
        except BaseException as error:
            errors.append(error)
            replacement_acquired.set()

    replacement = threading.Thread(target=run_replacement)
    replacement.start()
    try:
        time.sleep(0.15)
        assert not replacement_acquired.is_set()
        assert pbs_role_launcher._cleanup_discovery_records(old, old_lock) == 1
        old_lock.close()
        assert replacement_acquired.wait(timeout=5)
        assert errors == []
        assert entry.read_text() == "new-address"
        with pytest.raises(RuntimeError, match="generation changed"):
            pbs_role_launcher._cleanup_discovery_records(old, stale_view)
        assert entry.read_text() == "new-address"
    finally:
        if not old_lock.closed:
            old_lock.close()
        stale_view.close()
        release_replacement.set()
        replacement.join(timeout=5)
    assert not replacement.is_alive()
    assert errors == []


def test_real_supervisor_terminates_worker_and_removes_control_record(tmp_path):
    manifest_path = tmp_path / "role.json"
    control_path = tmp_path / "actor.control.json"
    manifest = {
        "version": 1,
        "generation": "generation-c",
        "role": "actor",
        "replicas": 1,
        "pbs_job_id": "",
        "container_image": "",
        "termination_grace_seconds": 0.2,
        "worker_command": [sys.executable, "-c", "import time; time.sleep(60)"],
        "worker_environment": {"SAFE_VALUE": "literal value"},
        "hosts": [
            {
                "hostname": socket.gethostname(),
                "vnode_index": 0,
                "control_path": str(control_path),
                "workers": [{"worker_index": 0, "local_rank": 0, "gpu_ids": [0]}],
            }
        ],
    }
    _write_private_json(manifest_path, manifest)
    process = subprocess.Popen(
        [
            sys.executable,
            str(REPO_ROOT / "platoon/train/areal/pbs_role_supervisor.py"),
            "--manifest",
            str(manifest_path),
            "--host-index",
            "0",
        ]
    )
    try:
        deadline = time.monotonic() + 5
        child_pid = None
        while process.poll() is None and time.monotonic() < deadline:
            if control_path.exists():
                record = json.loads(control_path.read_text())
                if record["children"]:
                    child_pid = record["children"][0]["pid"]
                    break
            time.sleep(0.02)
        assert child_pid is not None
        assert stat.S_IMODE(control_path.stat().st_mode) == 0o600

        process.terminate()
        assert process.wait(timeout=5) == 0
        assert not control_path.exists()
        assert not Path(f"/proc/{child_pid}").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
