"""Tool-call parse counts travel from each model call's generation record to the rollout metrics.

The gateway records, per call, how many tool-call blocks parsed, how many only by a lenient rule and how many did not
parse (``ChatResponse.generation``, kept under ``MODEL_GENERATIONS_KEY`` in the trajectory misc).  Each trajectory's
stats sum them, and a group's rollout statistics report them per trajectory, as ``num_tool_calls``,
``num_recovered_tool_calls`` and ``num_unparsed_tool_calls``.
"""

from __future__ import annotations

import pytest

from platoon.train.workflow import output as output_module
from platoon.utils.generation import MODEL_GENERATIONS_KEY
from platoon.utils.rollout_workload import RolloutWorkload
from platoon.utils.stats_tracker import StatsTracker
from platoon.utils.tinker_data_processing import TrajectoryCollectionResult, TrajectoryStats, tool_call_counts


def _generation(parsed: int, recovered: int = 0, unparsed: int = 0) -> dict:
    return {
        "stop_reason": "stop",
        "parsed_tool_calls": parsed,
        "emitted_tool_calls": parsed,
        "recovered_tool_calls": recovered,
        "unparsed_tool_calls": unparsed,
        "finish_reason": "tool_calls" if parsed else "stop",
    }


def test_a_trajectory_sums_its_calls_counts():
    trajectory = {"misc": {MODEL_GENERATIONS_KEY: {"c0": _generation(2, 1), "c1": _generation(0, 0, 1)}}}
    assert tool_call_counts(trajectory) == {
        "num_tool_calls": 2,
        "num_recovered_tool_calls": 1,
        "num_unparsed_tool_calls": 1,
    }
    # Records from before the counts existed, and trajectories without calls, count nothing.
    old = {"misc": {MODEL_GENERATIONS_KEY: {"c0": {"parsed_tool_calls": 1}}}}
    assert tool_call_counts(old) == {"num_tool_calls": 1, "num_recovered_tool_calls": 0, "num_unparsed_tool_calls": 0}
    assert tool_call_counts({"steps": []}) == {
        "num_tool_calls": 0,
        "num_recovered_tool_calls": 0,
        "num_unparsed_tool_calls": 0,
    }


def _outcome(tool_calls: int, unparsed: int) -> output_module._SingleRolloutOutcome:
    stats = TrajectoryStats(
        trajectory_id="root",
        reward=1.0,
        num_steps=1,
        num_input_tokens=3,
        num_output_tokens=2,
        num_datums=1,
        is_root=True,
        num_tool_calls=tool_calls,
        num_unparsed_tool_calls=unparsed,
    )
    result = TrajectoryCollectionResult(datums=[], task_reward=1.0, trajectory_stats=[stats], root_rewards_dict={})
    return output_module._SingleRolloutOutcome(result=result, workload=RolloutWorkload(), observed=True)


def test_rollout_metrics_report_the_counts_per_trajectory():
    stats = output_module._summarize_outcomes([_outcome(3, 0), _outcome(1, 2)])
    tracker = StatsTracker("train")
    stats.record(tracker, prefix="rollout_completion")
    metrics = tracker.export()
    assert metrics["train/rollout_completion/num_tool_calls/avg"] == pytest.approx(2.0)
    assert metrics["train/rollout_completion/num_unparsed_tool_calls/avg"] == pytest.approx(1.0)
    assert metrics["train/rollout_completion/num_unparsed_tool_calls/max"] == 2
    assert metrics["train/rollout_completion/num_recovered_tool_calls/avg"] == 0
