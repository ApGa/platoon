"""``inference.workflow.success_threshold`` decides the success of a traced (harness) rollout, whose root records its
outcome and whose last step carries the scored reward as ``reward/success``; a legacy collection without a recorded
outcome keeps its own success flag, which counts at 1."""

from __future__ import annotations

import pytest

from platoon.eval.config_defs import EvalWorkflowConfig
from platoon.eval.workflow import DefaultEvalGroupWorkflow


def _traced(grade: float | None, status: str = "scored") -> dict:
    """A rollout as the gateway trace records it (``platoon.gateway.trace``): the outcome on the root's misc, and a
    scored reward on the last step as ``reward/success``."""

    reward_misc = {"reward/success": grade} if status == "scored" else {}
    rewards = {"reward": grade} if status == "scored" else {}
    return {
        "id": "c",
        "trajectories": {
            "root": {
                "id": "root",
                "task": {"id": "t"},
                "reward": grade if status == "scored" else 0.0,
                "steps": [{"misc": {}}, {"misc": {"reward_misc": reward_misc}}],
                "misc": {"outcome": {"status": status, "rewards": rewards, "reason": None}},
            }
        },
    }


def _legacy(flag: float, reward: float) -> dict:
    return {
        "trajectories": {
            "root": {"id": "root", "reward": reward, "steps": [{"misc": {"reward_misc": {"success": flag}}}]}
        }
    }


def _success(collection: dict, threshold: float, *, reward_processor=None, **config) -> bool:
    workflow = DefaultEvalGroupWorkflow(
        rollout_fn=None,
        get_task_fn=None,
        model_name="m",
        config=EvalWorkflowConfig(success_threshold=threshold, **config),
        **({"reward_processor": reward_processor} if reward_processor else {}),
    )
    return workflow.record_from_collection("t", 0, collection).success


@pytest.mark.parametrize(
    ("threshold", "grade", "success"),
    [(0.5, 0.6, True), (0.5, 0.5, True), (0.5, 0.4, False), (1.0, 1.0, True), (1.0, 0.99, False), (1.0, 0.0, False)],
)
def test_the_threshold_decides_a_traced_rollouts_success(threshold, grade, success):
    assert _success(_traced(grade), threshold) is success


def test_a_legacy_success_flag_still_decides_without_a_recorded_outcome():
    assert _success(_legacy(flag=1.0, reward=0.2), 5.0) is True
    assert _success(_legacy(flag=0.0, reward=2.0), 1.0) is False


def test_a_traced_rollout_is_judged_on_its_scored_reward_not_the_processed_one():
    def shaped(trajectory):  # a reward processor that shapes the root's reward
        return trajectory["reward"] - 0.3, {}

    assert _success(_traced(1.0), 1.0, reward_processor=shaped) is True
    assert _success(_traced(0.6), 0.5, reward_processor=shaped) is True


def test_an_unscored_traced_rollout_never_succeeds():
    # count_timeout_as_failure makes a timeout a valid failure: never a success, whatever the threshold.
    assert _success(_traced(None, status="timeout"), 0.0, count_timeout_as_failure=True) is False


def test_an_outcome_without_a_status_is_judged_as_a_legacy_rollout():
    # As outcome_status reads it: only an outcome with a status was recorded by the trace.
    legacy = _legacy(flag=1.0, reward=0.2)
    legacy["trajectories"]["root"]["misc"] = {"outcome": {"note": "written by a custom rollout"}}

    assert _success(legacy, 5.0) is True
