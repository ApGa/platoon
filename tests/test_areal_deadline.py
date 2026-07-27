from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "platoon/train/areal/deadline.py"
LAUNCHER = REPO_ROOT / "slurm-scripts/openreward-toolathlon-prealloc-base.sh"

_SPEC = importlib.util.spec_from_file_location("platoon_areal_deadline_test", MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
StepDeadlineGuard = _MODULE.StepDeadlineGuard


def test_deadline_guard_is_disabled_without_launcher_deadline():
    assert StepDeadlineGuard.from_environment({}) is None


def test_deadline_guard_uses_conservative_recent_step_estimate(tmp_path):
    guard = StepDeadlineGuard(
        deadline_epoch=1000.0,
        drain_file=tmp_path / "drain.json",
        initial_step_seconds=100.0,
        safety_seconds=20.0,
        history_size=2,
        history_multiplier=1.5,
    )

    first = guard.decide(now_epoch=870.0)
    assert first.should_drain is False
    assert first.required_seconds == 120.0

    guard.record_completed_step(120.0)
    learned = guard.decide(now_epoch=810.0)
    assert learned.estimated_step_seconds == 180.0
    assert learned.required_seconds == 200.0
    assert learned.should_drain is True


def test_deadline_guard_writes_atomic_explanatory_marker(tmp_path):
    marker = tmp_path / "job-state" / "deadline-drain.json"
    guard = StepDeadlineGuard(
        deadline_epoch=1000.0,
        drain_file=marker,
        initial_step_seconds=300.0,
        safety_seconds=60.0,
    )
    decision = guard.decide(now_epoch=700.0)
    assert decision.should_drain is True

    guard.write_drain_marker(decision, global_step=42)

    payload = json.loads(marker.read_text())
    assert payload["reason"] == "insufficient_time_for_complete_training_step"
    assert payload["global_step"] == 42
    assert payload["remaining_seconds"] == 300.0
    assert payload["required_seconds"] == 360.0
    assert list(marker.parent.glob(".*.tmp")) == []


def test_deadline_guard_requires_marker_path_when_enabled():
    with pytest.raises(ValueError, match="PLATOON_TRAINING_DRAIN_FILE"):
        StepDeadlineGuard.from_environment({"PLATOON_TRAINING_DEADLINE_EPOCH": "1000"})


def test_preallocated_launcher_resubmits_only_after_clean_boundary_drain():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    launcher = LAUNCHER.read_text()

    assert "resolve_slurm_deadline_epoch" in launcher
    assert "PLATOON_TRAINING_DRAIN_FILE" in launcher
    assert '[[ -f "${PLATOON_TRAINING_DRAIN_FILE}" && "${status}" -eq 0 ]]' in launcher
    assert 'restart_reason="step-boundary deadline drain"' in launcher
    assert "successor_infrastructure_restart=0" in launcher
    assert 'submit_successor "${restart_reason}" "${successor_infrastructure_restart}"' in launcher
