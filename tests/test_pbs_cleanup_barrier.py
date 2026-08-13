from __future__ import annotations

import getpass
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = REPO_ROOT / "platoon/train/areal/pbs_cleanup_barrier.py"
    spec = importlib.util.spec_from_file_location("pbs_cleanup_barrier_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_private(path: Path, value: dict) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    path.chmod(0o600)


def _generation(tmp_path: Path):
    role = "actor"
    generation = "a" * 32
    job_id = "123.pbs1"
    experiment = "exp"
    trial = "trial"
    state = tmp_path / job_id / trial
    state.mkdir(parents=True, mode=0o700)
    record_root = tmp_path / "name-resolve"
    registry_user = getpass.getuser()
    role_root = record_root / registry_user / experiment / trial / "workers" / role
    entry = role_root / "0" / "ENTRY"
    entry.parent.mkdir(parents=True)
    entry.write_text("127.0.0.1:1\n")
    control = state / f"{role}.{generation}.node.0.control.json"
    _write_private(control, {})
    status = state / f"{role}.{generation}.launcher-status.json"
    manifest = state / f"{role}.{generation}.manifest.json"
    _write_private(
        manifest,
        {
            "version": 1,
            "role": role,
            "generation": generation,
            "pbs_job_id": job_id,
            "replicas": 1,
            "launcher_status_path": str(status),
            "hosts": [{"control_path": str(control)}],
            "discovery": {
                "type": "nfs",
                "record_root": str(record_root),
                "role_root": str(role_root),
                "entry_paths": [str(entry)],
                "lock_identity": {
                    "registry_user": registry_user,
                    "experiment_name": experiment,
                    "trial_name": trial,
                },
            },
        },
    )
    return state, manifest, status, control, entry, role_root, job_id, experiment, trial


def test_barrier_waits_for_exact_generation_paths_to_disappear(tmp_path):
    module = _module()
    state, _manifest, status, control, entry, role_root, job_id, experiment, trial = _generation(tmp_path)
    _write_private(
        status,
        {
            "version": 1,
            "complete": True,
            "role": "actor",
            "generation": "a" * 32,
            "cleanup_host_count": 1,
            "cleanup_failures": [],
        },
    )

    with pytest.raises(module.CleanupPending, match="paths remain"):
        module.assert_cleanup_complete(state, pbs_job_id=job_id, experiment=experiment, trial=trial)

    control.unlink()
    entry.unlink()
    entry.parent.rmdir()
    role_root.rmdir()
    assert module.assert_cleanup_complete(state, pbs_job_id=job_id, experiment=experiment, trial=trial) == 1


def test_barrier_rejects_failed_or_wrong_generation_status(tmp_path):
    module = _module()
    state, _manifest, status, _control, _entry, _role_root, job_id, experiment, trial = _generation(tmp_path)
    _write_private(
        status,
        {
            "version": 1,
            "complete": True,
            "role": "actor",
            "generation": "b" * 32,
            "cleanup_host_count": 1,
            "cleanup_failures": [],
        },
    )
    with pytest.raises(ValueError, match="Invalid PBS launcher cleanup status"):
        module.assert_cleanup_complete(state, pbs_job_id=job_id, experiment=experiment, trial=trial)


def test_missing_exact_trial_state_is_a_no_role_success(tmp_path):
    module = _module()
    state = tmp_path / "123.pbs1" / "trial"
    assert module.assert_cleanup_complete(state, pbs_job_id="123.pbs1", experiment="exp", trial="trial") == 0


def test_common_wrapper_runs_bounded_barrier_before_node_service_teardown():
    common = (REPO_ROOT / "pbs-scripts/openreward-ta-swe-prealloc-common.sh").read_text()
    cleanup = common[common.index("cleanup() {") : common.index("handle_signal() {")]
    barrier = cleanup.index("pbs_cleanup_barrier.py")
    controller = cleanup.index('terminate_pid "${controller_pid}" 75')
    services = cleanup.index("request_all_service_shutdowns")
    assert controller < barrier < services
    assert "service_shutdown_deadline=$((SECONDS + 180))" in cleanup
    publication = cleanup.index("publish_all_service_shutdown_requests")
    assert barrier < publication
    assert "expected_primary_status=0" in cleanup
    assert "expected_primary_status=143" in cleanup
    assert '--primary-status "${expected_primary_status}"' in cleanup
    assert "PLATOON_AREAL_PBS_CLEANUP_BARRIER_TIMEOUT" in common
    assert '--state-dir "${JOB_STATE_DIR}/pbs-prealloc/${JOB_ID_SAFE}/${EXPECTED_TRIAL}"' in cleanup
