"""Static and focused unit checks for the live PBS scheduler Gate 4."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import socket
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = REPO_ROOT / "pbs-scripts" / "pbs_scheduler_gate4.py"
WORKER_PATH = REPO_ROOT / "pbs-scripts" / "pbs_scheduler_gate4_worker.py"
PBS_PATH = REPO_ROOT / "pbs-scripts" / "pbs-scheduler-gate4.pbs"


def _load_gate_module():
    spec = importlib.util.spec_from_file_location("pbs_scheduler_gate4_test", GATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_worker_module():
    spec = importlib.util.spec_from_file_location("pbs_scheduler_gate4_worker_test", WORKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_discovery_path_exactly_matches_areal_nfs_layout(tmp_path):
    gate = _load_gate_module()
    worker = _load_worker_module()
    expected = tmp_path / "alice" / "exp" / "trial" / "workers" / "actor" / "7" / "ENTRY"

    assert gate.name_record_path(tmp_path, "alice", "exp", "trial", "actor", 7) == expected
    assert worker.name_record_path(tmp_path, "alice", "exp", "trial", "actor", 7) == expected


def test_worker_ip_discovery_uses_areal_route_fallback_when_dns_is_hidden(monkeypatch):
    worker = _load_worker_module()

    monkeypatch.setattr(
        worker.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(socket.gaierror("isolated resolver")),
    )

    class ProbeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def connect(self, target):
            assert target == ("8.8.8.8", 80)

        def getsockname(self):
            return ("10.20.30.40", 54321)

    monkeypatch.setattr(worker.socket, "socket", lambda *_args, **_kwargs: ProbeSocket())

    assert worker.outward_ipv4() == "10.20.30.40"


def test_repeated_worker_stop_signals_preserve_orderly_entry_and_receipt_cleanup(
    monkeypatch,
    tmp_path,
):
    worker = _load_worker_module()
    registered_handlers = {}

    def register_handler(signum, handler):
        registered_handlers[signum] = handler

    class FakeServer:
        server_port = 12345
        timeout = None
        closed = False

        def handle_request(self):
            registered_handlers[worker.signal.SIGTERM](worker.signal.SIGTERM, None)
            registered_handlers[worker.signal.SIGINT](worker.signal.SIGINT, None)

        def server_close(self):
            self.closed = True

    server = FakeServer()
    fake_cuda = types.SimpleNamespace(
        device_count=lambda: 1,
        get_device_name=lambda _index: "NVIDIA H200",
        is_available=lambda: True,
        synchronize=lambda: None,
    )
    fake_torch = types.SimpleNamespace(
        cuda=fake_cuda,
        ones=lambda *_args, **_kwargs: types.SimpleNamespace(item=lambda: 1.0),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(worker.signal, "signal", register_handler)
    monkeypatch.setattr(worker, "ThreadingHTTPServer", lambda *_args: server)
    monkeypatch.setattr(worker, "outward_ipv4", lambda: "10.0.80.11")
    monkeypatch.setattr(worker.socket, "gethostname", lambda: "hnode011")
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(stdout="GPU-test\n"),
    )

    node_local_root = tmp_path / "node-local"
    worker_home = node_local_root / "worker-home"
    worker_tmp = node_local_root / "worker-tmp"
    worker_home.mkdir(parents=True, mode=0o700)
    worker_tmp.mkdir(mode=0o1777)
    node_local_root.chmod(0o700)
    worker_home.chmod(0o700)
    worker_tmp.chmod(0o1777)

    environment = {
        "CUDA_VISIBLE_DEVICES": "0",
        "GATE4_EXPECTED_GPU_COUNT": "1",
        "GATE4_EXPECTED_SIF_DIGEST": "sha256:test",
        "GATE4_NODE_LOCAL_INIT_GENERATION": "generation-init",
        "GATE4_NODE_LOCAL_ROOT": str(node_local_root),
        "GATE4_RECEIPT_ROOT": str(tmp_path / "receipts"),
        "GATE4_REGISTRY_USER": "alice",
        "HOME": str(worker_home),
        "PLATOON_ASSIGNED_GPU_IDS": "0",
        "PLATOON_ASSIGNED_GPU_UUIDS": "GPU-test",
        "PLATOON_CONTAINER_IMAGE_DIGEST": "sha256:test",
        "PLATOON_LOCAL_RANK": "0",
        "PLATOON_ROLE": "actor_like",
        "PLATOON_ROLE_GENERATION": "a" * 32,
        "PLATOON_WORKER_INDEX": "0",
        "TMPDIR": "/tmp",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    for name in (
        "AWS_SECRET_ACCESS_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "OPENAI_API_KEY",
        "WANDB_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    record_root = tmp_path / "name-resolve"
    assert (
        worker.main(
            [
                "--experiment-name",
                "exp",
                "--trial-name",
                "trial",
                "--role",
                "actor_like",
                "--worker-index",
                "0",
                "--name-resolve-type",
                "nfs",
                "--nfs-record-root",
                str(record_root),
            ]
        )
        == 0
    )

    entry = worker.name_record_path(record_root, "alice", "exp", "trial", "actor_like", 0)
    receipt_path = tmp_path / "receipts/trial/actor_like/0.json"
    receipt = json.loads(receipt_path.read_text())
    assert not entry.exists()
    assert server.closed is True
    assert receipt["entry_self_removed"] is True
    assert receipt["exit_status"] == 0
    assert receipt["state"] == "stopped"
    assert receipt["stop_signal"] == worker.signal.SIGTERM
    assert receipt["node_local_init_generation"] == "generation-init"
    assert receipt["node_local_root"] == str(node_local_root)
    assert receipt["node_local_root_mode"] == 0o700
    assert receipt["worker_home"] == str(worker_home)
    assert receipt["worker_home_mode"] == 0o700
    assert receipt["worker_home_write_probe"] is True
    assert receipt["worker_tmp"] == str(worker_tmp)
    assert receipt["worker_tmp_mode"] == 0o1777
    assert receipt["worker_tmp_write_probe"] is True
    probe_name = f".gate4-write-probe-actor_like-0-{'a' * 32}"
    assert not (worker_home / probe_name).exists()
    assert not (Path("/tmp") / probe_name).exists()


def test_survivor_probe_detects_worker_token_and_lifecycle_marker(tmp_path):
    gate = _load_gate_module()
    token_process = tmp_path / "101"
    token_process.mkdir()
    (token_process / "environ").write_bytes(b"A=B\0GATE4_TOKEN=unique-token\0")
    (token_process / "cmdline").write_bytes(b"python\0worker.py\0")
    launcher_process = tmp_path / "102"
    launcher_process.mkdir()
    (launcher_process / "environ").write_bytes(b"A=B\0")
    (launcher_process / "cmdline").write_bytes(
        b"python\0pbs_role_launcher.py\0/state/gate4-job-controller-death/manifest\0"
    )
    unrelated_process = tmp_path / "103"
    unrelated_process.mkdir()
    (unrelated_process / "environ").write_bytes(b"A=B\0")
    (unrelated_process / "cmdline").write_bytes(b"python\0unrelated.py\0gate4-job-controller-death\0")

    survivors = gate.find_survivors(
        "unique-token",
        "gate4-job-controller-death",
        proc_root=tmp_path,
        own_pid=999,
    )

    assert [item["pid"] for item in survivors] == [101, 102]
    assert survivors[0]["token_match"] is True
    assert survivors[1]["command_match"] is True


def test_remote_probe_publishes_private_nonce_bound_result(monkeypatch, tmp_path):
    gate = _load_gate_module()
    monkeypatch.setattr(
        gate,
        "find_survivors",
        lambda token, marker: [{"pid": 123, "token": token, "marker": marker}],
    )
    result = tmp_path / "probe.json"

    assert (
        gate.main(
            [
                "--probe",
                "--token",
                "token-a",
                "--marker",
                "trial-a",
                "--challenge",
                "nonce-a",
                "--probe-result",
                str(result),
            ]
        )
        == 0
    )

    value = json.loads(result.read_text())
    assert value["challenge"] == "nonce-a"
    assert value["survivors"] == [{"pid": 123, "token": "token-a", "marker": "trial-a"}]
    assert result.stat().st_mode & 0o077 == 0


def test_manifest_gate_requires_explicit_index_isolation_and_exact_topology(tmp_path):
    gate = _load_gate_module()
    manifest_path = tmp_path / "actor.manifest.json"
    assignments = [{"gpu_ids": [index], "local_rank": index, "worker_index": index} for index in range(8)]
    role_root = tmp_path / "name-resolve" / "alice" / "exp" / "trial" / "workers" / "actor_like"
    lock_identity = {
        "version": 1,
        "record_root": str((tmp_path / "name-resolve").resolve()),
        "registry_user": "alice",
        "experiment_name": "exp",
        "trial_name": "trial",
    }
    trial_digest = hashlib.sha256(json.dumps(lock_identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    role_digest = hashlib.sha256(
        json.dumps(
            {**lock_identity, "role": "actor_like"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    lock_root = tmp_path / "name-resolve" / "alice" / ".pbs-prealloc-locks"
    gate.atomic_json(
        manifest_path,
        {
            "role": "actor_like",
            "generation": "nonce",
            "container_image_digest": "sha256:" + "1" * 64,
            "termination_reverse_order_default": True,
            "gpu_visibility": {
                "mode": "nvidia-container-cli",
                "nvidia_smi_binary": "/usr/bin/nvidia-smi",
            },
            "discovery": {
                "type": "nfs",
                "record_root": str(tmp_path / "name-resolve"),
                "role_root": str(role_root),
                "entry_paths": [str(role_root / str(index) / "ENTRY") for index in range(8)],
                "lock_identity": lock_identity,
                "constructor_lifecycle_lock_path": str(lock_root / f"constructor-{trial_digest}.lock"),
                "trial_lifecycle_lock_path": str(lock_root / f"trial-{trial_digest}.lock"),
                "lifecycle_lock_path": str(lock_root / f"role-{role_digest}.lock"),
            },
            "worker_command": [
                "singularity",
                "exec",
                "--nv",
                "--cleanenv",
                "--containall",
                "--no-home",
                "--no-mount",
                "home,cwd,bind-paths",
                "--bind",
                "/etc/hosts:/etc/hosts:ro",
                "--bind",
                "/etc/resolv.conf:/etc/resolv.conf:ro",
                "/image.sif",
                "bash",
                "--worker-index ${PLATOON_WORKER_INDEX}",
            ],
            "hosts": [
                {
                    "hostname": "node-a",
                    "vnode_index": 0,
                    "workers": assignments,
                }
            ],
        },
    )

    value = gate.validate_role_manifest(
        manifest_path,
        role="actor_like",
        expected_host="node-a.example",
        expected_vnode=0,
        expected_digest="sha256:" + "1" * 64,
        expected_assignments=assignments,
    )

    assert value["generation"] == "nonce"
    assert manifest_path.stat().st_mode & 0o077 == 0


def test_ready_receipt_requires_live_resolver_evidence(monkeypatch, tmp_path):
    gate = _load_gate_module()
    node_local_root = tmp_path / "node-local"
    monkeypatch.setenv("GATE4_NODE_LOCAL_INIT_GENERATION", "generation-init")
    monkeypatch.setenv("GATE4_NODE_LOCAL_ROOT", str(node_local_root))
    trial = "trial"
    role = "actor_like"
    path = gate.receipt_path(tmp_path, trial, role, 0)
    gate.atomic_json(
        path,
        {
            "address": "10.0.80.84:12345",
            "assigned_gpu_ids": [0],
            "assigned_gpu_uuids": ["GPU-00000001-1111-2222-3333-000000000001"],
            "container_image_digest": "sha256:" + "1" * 64,
            "cuda_visible_devices": [0],
            "gpu_names": ["NVIDIA H200"],
            "host": "hnode084",
            "local_rank": 0,
            "node_local_init_generation": "generation-init",
            "node_local_root": str(node_local_root),
            "node_local_root_mode": 0o700,
            "observed_gpu_uuids": ["GPU-00000001-1111-2222-3333-000000000001"],
            "resolver_addresses": ["10.0.80.84"],
            "resolver_files": ["/etc/hosts", "/etc/resolv.conf"],
            "role": role,
            "state": "ready",
            "torch_gpu_count": 1,
            "worker_home": str(node_local_root / "worker-home"),
            "worker_home_mode": 0o700,
            "worker_home_write_probe": True,
            "worker_index": 0,
            "worker_tmp": str(node_local_root / "worker-tmp"),
            "worker_tmp_mode": 0o1777,
            "worker_tmp_write_probe": True,
        },
    )
    worker = gate.GateWorker(id=f"{role}/0", ip="10.0.80.84", worker_ports=["12345"])

    values = gate.validate_ready_receipts(
        tmp_path,
        trial,
        role,
        [worker],
        "hnode084",
        "sha256:" + "1" * 64,
        1,
    )

    assert values[0]["resolver_addresses"] == ["10.0.80.84"]


def test_lifecycle_metadata_gate_requires_private_regular_files(tmp_path):
    gate = _load_gate_module()
    identity = {
        "version": 1,
        "record_root": str((tmp_path / "name-resolve").resolve()),
        "registry_user": "alice",
        "experiment_name": "exp",
        "trial_name": "trial",
    }
    lock_root = tmp_path / "name-resolve" / "alice" / ".pbs-prealloc-locks"
    lock_root.mkdir(mode=0o700, parents=True)
    constructor_path = lock_root / "constructor.lock"
    trial_path = lock_root / "trial.lock"
    role_path = lock_root / "role.lock"
    gate.atomic_json(
        constructor_path,
        {
            "version": 1,
            "kind": "pbs-constructor-lifecycle",
            "identity": identity,
        },
    )
    gate.atomic_json(
        trial_path,
        {
            "version": 1,
            "kind": "pbs-trial-lifecycle",
            "identity": identity,
        },
    )
    gate.atomic_json(
        role_path,
        {
            "version": 1,
            "kind": "pbs-role-lifecycle",
            "identity": identity,
            "role": "actor",
            "generation": "generation-a",
        },
    )
    manifest = {
        "role": "actor",
        "generation": "generation-a",
        "discovery": {
            "lock_identity": identity,
            "constructor_lifecycle_lock_path": str(constructor_path),
            "trial_lifecycle_lock_path": str(trial_path),
            "lifecycle_lock_path": str(role_path),
        },
    }

    observed = gate.validate_lifecycle_metadata(manifest)
    assert observed["constructor"]["mode"] == "0o600"
    assert observed["trial"]["mode"] == "0o600"
    assert observed["role"]["mode"] == "0o600"

    target = tmp_path / "role-target.lock"
    role_path.replace(target)
    role_path.symlink_to(target)
    with pytest.raises(AssertionError, match="including symlink"):
        gate.validate_lifecycle_metadata(manifest)


def test_gate4_validates_exact_two_node_initializer_receipts(monkeypatch, tmp_path):
    gate = _load_gate_module()
    root = Path("/local/12345.pbs1/platoon/12345.pbs1")
    receipt_root = tmp_path / "node-local-ready" / ("a" * 32)
    receipt_root.mkdir(parents=True, mode=0o700)
    receipt_root.chmod(0o700)
    nodes = [
        types.SimpleNamespace(hostname="hnode001", vnode_index=0),
        types.SimpleNamespace(hostname="hnode002", vnode_index=8),
    ]
    monkeypatch.setenv("PBS_JOBID", "12345.pbs1")
    monkeypatch.setenv("GATE4_NODE_LOCAL_ROOT", str(root))
    monkeypatch.setenv("GATE4_NODE_LOCAL_RECEIPT_DIR", str(receipt_root))
    monkeypatch.setenv("GATE4_NODE_LOCAL_INIT_GENERATION", "a" * 32)
    for node in nodes:
        gate.atomic_json(
            receipt_root / f"{node.hostname}.json",
            {
                "schema_version": 1,
                "pbs_job_id": "12345.pbs1",
                "generation": "a" * 32,
                "expected_host": node.hostname,
                "observed_host": node.hostname,
                "task_index": node.vnode_index,
                "node_local_root": str(root),
                "node_local_root_mode": 0o700,
                "worker_home": str(root / "worker-home"),
                "worker_home_mode": 0o700,
                "worker_tmp": str(root / "worker-tmp"),
                "worker_tmp_mode": 0o1777,
                "training_sif_staged": False,
                "training_sif_path": "",
                "training_sif_sha256": "",
            },
        )

    values = gate.validate_node_local_init_receipts(nodes)

    assert [value["expected_host"] for value in values] == ["hnode001", "hnode002"]


def test_gate4_runtime_binds_initialized_root_and_worker_tmp(monkeypatch, tmp_path):
    gate = _load_gate_module()
    node_local_root = Path("/local/12345.pbs1/platoon/12345.pbs1")
    environment = dict(gate.os.environ)
    environment["GATE4_NODE_LOCAL_ROOT"] = str(node_local_root)
    environment["GATE4_SINGULARITY_BIN"] = "/usr/bin/singularity"
    monkeypatch.setattr(gate.os, "environ", environment)

    gate.configure_runtime(tmp_path / "repo", tmp_path / "state", tmp_path / "image.sif")

    assert f"{node_local_root}:{node_local_root}:rw" in environment[
        "PLATOON_AREAL_PBS_CONTAINER_BINDS"
    ]
    assert f"--bind {node_local_root}/worker-tmp:/tmp:rw" in environment[
        "PLATOON_AREAL_PBS_CONTAINER_ARGS"
    ]
    assert f"export HOME={node_local_root}/worker-home" in environment[
        "PLATOON_AREAL_PBS_WORKER_PREAMBLE"
    ]
    assert "export TMPDIR=/tmp" in environment["PLATOON_AREAL_PBS_WORKER_PREAMBLE"]


def test_cleanup_gate_accepts_unattributed_worker_unlink_when_exact_paths_are_absent(
    tmp_path,
):
    gate = _load_gate_module()
    role_root = tmp_path / "workers/actor"
    entries = [role_root / str(index) / "ENTRY" for index in range(8)]
    status = {
        "complete": True,
        "cleanup_failures": [],
        "primary_status": 0,
        "discovery_records_removed": 7,
        "stale_discovery_records_removed": 0,
    }
    receipts = [
        {
            "worker_index": index,
            "role_generation": "generation-a",
            "state": "ready",
        }
        for index in range(8)
    ]

    result = gate.assert_cleanup_union(
        status,
        receipts,
        expected_generation="generation-a",
        expected_entry_paths=entries,
        expected_role_root=role_root,
    )

    assert result["launcher_removed_records"] == 7
    assert result["receipt_attributed_worker_indices"] == []
    assert result["unattributed_worker_removals"] == 1


def test_cleanup_gate_rejects_any_remaining_exact_entry(tmp_path):
    gate = _load_gate_module()
    role_root = tmp_path / "workers/actor"
    entries = [role_root / str(index) / "ENTRY" for index in range(8)]
    entries[3].parent.mkdir(parents=True)
    entries[3].write_text("10.0.0.1:1234")
    status = {
        "complete": True,
        "cleanup_failures": [],
        "primary_status": 0,
        "discovery_records_removed": 7,
        "stale_discovery_records_removed": 0,
    }
    receipts = [
        {
            "worker_index": index,
            "role_generation": "generation-a",
            "state": "ready",
        }
        for index in range(8)
    ]

    with pytest.raises(AssertionError, match="exact discovery cleanup postcondition"):
        gate.assert_cleanup_union(
            status,
            receipts,
            expected_generation="generation-a",
            expected_entry_paths=entries,
            expected_role_root=role_root,
        )


def test_cleanup_gate_rejects_double_counted_attribution(tmp_path):
    gate = _load_gate_module()
    role_root = tmp_path / "workers/actor"
    entries = [role_root / str(index) / "ENTRY" for index in range(8)]
    status = {
        "complete": True,
        "cleanup_failures": [],
        "primary_status": 1,
        "discovery_records_removed": 8,
        "stale_discovery_records_removed": 0,
    }
    receipts = [
        {
            "entry_self_removed": index == 3,
            "worker_index": index,
            "role_generation": "generation-a",
            "state": "failed" if index == 3 else "ready",
        }
        for index in range(8)
    ]

    with pytest.raises(AssertionError, match="double-counted"):
        gate.assert_cleanup_union(
            status,
            receipts,
            expected_generation="generation-a",
            expected_entry_paths=entries,
            expected_role_root=role_root,
            expect_failure=True,
        )


def test_real_scheduler_source_loads_with_gate_upstream_surface(tmp_path):
    code = f"""
import importlib.util, os, pathlib, sys
path = pathlib.Path({str(GATE_PATH)!r})
spec = importlib.util.spec_from_file_location('gate4_subprocess', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
os.environ['GATE4_REGISTRY_USER'] = 'aci18912it'
loaded = module.load_real_pbs_scheduler(path.parents[1])
assert pathlib.Path(loaded.__file__).resolve() == path.parents[1] / 'platoon/train/areal/preallocated_pbs.py'
assert loaded.PreallocatedPBSScheduler.__mro__[1].__module__ == 'platoon.train.areal.preallocated_slurm'
"""

    subprocess.run([sys.executable, "-c", code], check=True, cwd=tmp_path)


def test_pbs_asset_is_two_node_short_and_never_sources_secrets():
    contents = PBS_PATH.read_text()

    assert "#PBS -P gai51740" in contents
    assert "#PBS -q R9920261300" in contents
    assert "#PBS -v RTYPE=rt_HF" in contents
    assert "#PBS -l select=2:mpiprocs=8" in contents
    assert "#PBS -l place=scatter" in contents
    assert "#PBS -l walltime=00:20:00" in contents
    assert "config/abci.env" not in contents.replace(
        "# Never source config/abci.env or config/secrets.env in this no-training gate.",
        "",
    )
    assert "source /home/" not in contents
    assert "WANDB_API_KEY=" not in contents
    assert "HF_TOKEN=" not in contents
    assert 'NODE_LOCAL_INIT=${REPO_ROOT}/pbs-scripts/openreward-node-local-init.py' in contents
    assert 'done <"${NODE_MAP}"' in contents
    assert '/opt/pbs/bin/pbsdsh -n "${vnode_index}" --' in contents
    assert '"${NODE_LOCAL_INIT_PYTHON}" "${NODE_LOCAL_INIT_SNAPSHOT}" init' in contents
    assert '--stage-training-sif 0' in contents
    assert '"${NODE_LOCAL_INIT_PYTHON}" -I "${NODE_LOCAL_INIT_SNAPSHOT}" verify' in contents
    assert '--expected-nodes 2' in contents
    assert 'export GATE4_NODE_LOCAL_RECEIPT_DIR=${NODE_LOCAL_RECEIPT_DIR}' in contents
