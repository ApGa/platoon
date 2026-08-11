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
RECURSIVE_LAUNCHER = (
    REPO_ROOT
    / "slurm-scripts/openreward-multienv-prealloc-32node-ptc-recursive-bs8-efficiency.sh"
)

_SPEC = importlib.util.spec_from_file_location("platoon_areal_deadline_test", MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
StepDeadlineGuard = _MODULE.StepDeadlineGuard


def test_deadline_guard_is_disabled_without_launcher_deadline():
    assert StepDeadlineGuard.from_environment({}) is None


def test_deadline_guard_excludes_first_step_then_uses_recent_estimate(tmp_path):
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

    guard.record_completed_step(900.0)
    warmup = guard.decide(now_epoch=870.0)
    assert guard.completed_steps == 1
    assert warmup.estimated_step_seconds == 100.0
    assert warmup.required_seconds == 120.0
    assert warmup.should_drain is False

    guard.record_completed_step(120.0)
    learned = guard.decide(now_epoch=810.0)
    assert guard.completed_steps == 2
    assert learned.estimated_step_seconds == 180.0
    assert learned.required_seconds == 200.0
    assert learned.should_drain is True


def test_deadline_guard_never_leaks_first_step_into_bounded_history(tmp_path):
    guard = StepDeadlineGuard(
        deadline_epoch=1000.0,
        drain_file=tmp_path / "drain.json",
        initial_step_seconds=10.0,
        safety_seconds=0.0,
        history_size=2,
        history_multiplier=1.0,
    )

    guard.record_completed_step(1000.0)
    guard.record_completed_step(20.0)
    guard.record_completed_step(30.0)

    assert guard.estimated_step_seconds == 30.0


def test_deadline_guard_drains_only_below_exact_required_boundary(tmp_path):
    guard = StepDeadlineGuard(
        deadline_epoch=1000.0,
        drain_file=tmp_path / "drain.json",
        initial_step_seconds=100.0,
        safety_seconds=20.0,
    )

    assert guard.decide(now_epoch=880.0).should_drain is False
    assert guard.decide(now_epoch=880.001).should_drain is True


def test_deadline_guard_validates_excluded_first_duration(tmp_path):
    guard = StepDeadlineGuard(
        deadline_epoch=1000.0,
        drain_file=tmp_path / "drain.json",
    )

    with pytest.raises(ValueError, match="finite and non-negative"):
        guard.record_completed_step(float("nan"))
    assert guard.completed_steps == 0


def test_deadline_guard_writes_atomic_explanatory_marker(tmp_path):
    marker = tmp_path / "job-state" / "deadline-drain.json"
    guard = StepDeadlineGuard(
        deadline_epoch=1000.0,
        drain_file=marker,
        initial_step_seconds=300.0,
        safety_seconds=60.0,
    )
    guard.record_completed_step(999.0)
    decision = guard.decide(now_epoch=700.0)
    assert decision.should_drain is True

    guard.write_drain_marker(decision, global_step=42)

    payload = json.loads(marker.read_text())
    assert payload["reason"] == "insufficient_time_for_complete_training_step"
    assert payload["global_step"] == 42
    assert payload["remaining_seconds"] == 300.0
    assert payload["estimated_step_seconds"] == 300.0
    assert payload["required_seconds"] == 360.0
    assert payload["completed_steps_in_allocation"] == 1
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


def test_recursive_launcher_uses_30_minute_bootstrap_floor():
    subprocess.run(["bash", "-n", str(RECURSIVE_LAUNCHER)], check=True)
    launcher = RECURSIVE_LAUNCHER.read_text()

    assert "OPENREWARD_DEADLINE_INITIAL_STEP_SECONDS:-1800" in launcher
    assert "OPENREWARD_DEADLINE_SAFETY_SECONDS:-600" in launcher
