"""Focused tests for end-to-end AReaL inter-update wall-clock telemetry."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "platoon/train/areal/inter_update_timing.py"
TRAINER_PATH = REPO_ROOT / "platoon/train/areal/rl.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "platoon_inter_update_timing_test",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_first_completion_is_explicit_without_a_bogus_zero_duration():
    timing = _load_module()
    clock = timing.InterAppliedUpdateWallClock()

    skipped = clock.metrics_for_step(None)
    assert skipped == {
        timing.APPLIED_UPDATE_COMPLETED: 0.0,
        timing.END_TO_END_UPDATE_INTERVAL_AVAILABLE: 0.0,
        timing.END_TO_END_UPDATE_INTERVAL_FIRST_COMPLETION: 0.0,
    }

    first = clock.metrics_for_step(100.0)
    assert first == {
        timing.APPLIED_UPDATE_COMPLETED: 1.0,
        timing.END_TO_END_UPDATE_INTERVAL_AVAILABLE: 0.0,
        timing.END_TO_END_UPDATE_INTERVAL_FIRST_COMPLETION: 1.0,
    }
    assert timing.END_TO_END_UPDATE_INTERVAL not in first


def test_skipped_steps_do_not_advance_the_applied_update_boundary():
    timing = _load_module()
    clock = timing.InterAppliedUpdateWallClock()

    clock.metrics_for_step(10.0)
    assert clock.metrics_for_step(None)[timing.APPLIED_UPDATE_COMPLETED] == 0.0
    second = clock.metrics_for_step(37.5)

    assert second[timing.END_TO_END_UPDATE_INTERVAL] == pytest.approx(27.5)
    assert second[timing.END_TO_END_UPDATE_INTERVAL_AVAILABLE] == 1.0
    assert second[timing.END_TO_END_UPDATE_INTERVAL_FIRST_COMPLETION] == 0.0
    assert second[timing.APPLIED_UPDATE_COMPLETED] == 1.0


@pytest.mark.parametrize("invalid", [math.inf, -math.inf, math.nan])
def test_completion_timestamp_must_be_finite(invalid):
    timing = _load_module()
    clock = timing.InterAppliedUpdateWallClock()

    with pytest.raises(ValueError, match="must be finite"):
        clock.metrics_for_step(invalid)
    assert clock.previous_completion is None


def test_regressing_timestamp_fails_without_corrupting_the_boundary():
    timing = _load_module()
    clock = timing.InterAppliedUpdateWallClock()
    clock.metrics_for_step(20.0)

    with pytest.raises(ValueError, match="must be monotonic"):
        clock.metrics_for_step(19.0)

    recovered = clock.metrics_for_step(23.0)
    assert recovered[timing.END_TO_END_UPDATE_INTERVAL] == pytest.approx(3.0)


def test_trainer_marks_completion_after_weight_sync_and_before_post_update_phases():
    source = TRAINER_PATH.read_text()
    train_start = source.index("    def train(")
    train_end = source.index("\n    def _evaluate_fn", train_start)
    train = source[train_start:train_end]

    clock = train.index("inter_update_clock = InterAppliedUpdateWallClock()")
    loop = train.index("for global_step in range(")
    reset = train.index("applied_update_completed_at = None", loop)
    optimizer = train.index("self.actor.ppo_update(optimizer_batch)", reset)
    weight_sync = train.index("self.actor.update_weights(versioned_meta)", optimizer)
    version_advance = train.index("self._advance_logical_versions(new_version)", weight_sync)
    sync_success = train.index("weight_sync_succeeded = True", version_advance)
    completion = train.index("applied_update_completed_at = time.monotonic()", sync_success)
    metric = train.index("inter_update_clock.metrics_for_step(", completion)
    save = train.index("self._save_hf(", metric)
    recover = train.index("self._save_recover_checkpoint(", save)
    evaluate = train.index("self._evaluate(", recover)
    clear = train.index("self.actor.clear_batches(", evaluate)
    commit = train.index("self._export_and_commit_stats(", clear)
    resume = train.index("self.rollout.resume()", commit)

    assert (
        clock
        < loop
        < reset
        < optimizer
        < weight_sync
        < version_advance
        < sync_success
        < completion
        < metric
        < save
        < recover
        < evaluate
        < clear
        < commit
        < resume
    )
