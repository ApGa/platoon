from types import SimpleNamespace

import pytest

from platoon.utils.rollout_workload import (
    RolloutWorkload,
    record_workload_distribution,
    sum_rollout_workloads,
    trajectory_collection_shape,
)
from platoon.utils.stats_tracker import StatsTracker


def test_rollout_workload_sums_and_exports_combined_tokens():
    first = RolloutWorkload(
        environment_steps=3,
        model_calls=2,
        input_tokens=100,
        output_tokens=20,
        trajectories=1,
        postmerge_datums=8,
        policy_eligible_datums=6,
        post_sampling_datums=4,
    )
    second = RolloutWorkload(
        environment_steps=7,
        model_calls=4,
        input_tokens=300,
        output_tokens=50,
        trajectories=3,
        postmerge_datums=12,
        policy_eligible_datums=10,
        post_sampling_datums=5,
    )

    total = sum_rollout_workloads([first, second])

    assert total == RolloutWorkload(
        environment_steps=10,
        model_calls=6,
        input_tokens=400,
        output_tokens=70,
        trajectories=4,
        postmerge_datums=20,
        policy_eligible_datums=16,
        post_sampling_datums=9,
    )
    assert total.to_metrics("workload/batch") == {
        "workload/batch/total_environment_steps": 10.0,
        "workload/batch/total_model_calls": 6.0,
        "workload/batch/total_input_tokens": 400.0,
        "workload/batch/total_output_tokens": 70.0,
        "workload/batch/total_tokens": 470.0,
        "workload/batch/total_trajectories": 4.0,
        "workload/batch/total_postmerge_datums": 20.0,
        "workload/batch/total_policy_eligible_datums": 16.0,
        "workload/batch/total_post_sampling_datums": 9.0,
        "workload/batch/total_policy_excluded_datums": 4.0,
        "workload/batch/total_sampling_dropped_datums": 7.0,
        "workload/batch/total_candidate_trainable_datums": 9.0,
        "workload/batch/total_candidate_non_trainable_datums": 11.0,
    }


def test_rollout_workload_distribution_reports_unit_statistics():
    tracker = StatsTracker()
    record_workload_distribution(
        tracker,
        prefix="workload/rollout",
        workloads=[
            RolloutWorkload(environment_steps=2, input_tokens=10, output_tokens=3),
            RolloutWorkload(
                environment_steps=6,
                input_tokens=30,
                output_tokens=7,
                postmerge_datums=8,
                policy_eligible_datums=6,
                post_sampling_datums=3,
            ),
        ],
    )

    exported = tracker.export()

    assert exported["workload/rollout/count"] == 2.0
    assert exported["workload/rollout/total_environment_steps/avg"] == 4.0
    assert exported["workload/rollout/total_environment_steps/min"] == 2.0
    assert exported["workload/rollout/total_environment_steps/max"] == 6.0
    assert exported["workload/rollout/total_tokens/avg"] == 25.0
    assert exported["workload/rollout/total_postmerge_datums/avg"] == 4.0
    assert exported["workload/rollout/total_policy_excluded_datums/max"] == 2.0
    assert exported["workload/rollout/total_sampling_dropped_datums/max"] == 3.0
    assert exported["workload/rollout/total_candidate_trainable_datums/max"] == 3.0
    assert exported["workload/rollout/total_candidate_non_trainable_datums/max"] == 5.0


def test_rollout_workload_distribution_exports_every_field_and_accepts_generator():
    tracker = StatsTracker()
    workloads = (
        workload
        for workload in [
            RolloutWorkload(
                environment_steps=1,
                model_calls=2,
                input_tokens=3,
                output_tokens=4,
                trajectories=5,
            ),
            RolloutWorkload(
                environment_steps=10,
                model_calls=20,
                input_tokens=30,
                output_tokens=40,
                trajectories=50,
            ),
        ]
    )

    record_workload_distribution(
        tracker,
        prefix="workload/task/",
        workloads=workloads,
    )

    exported = tracker.export()
    expected = {
        "total_environment_steps": (5.5, 1.0, 10.0),
        "total_model_calls": (11.0, 2.0, 20.0),
        "total_input_tokens": (16.5, 3.0, 30.0),
        "total_output_tokens": (22.0, 4.0, 40.0),
        "total_tokens": (38.5, 7.0, 70.0),
        "total_trajectories": (27.5, 5.0, 50.0),
    }
    assert exported["workload/task/count"] == 2.0
    for field, (average, minimum, maximum) in expected.items():
        assert exported[f"workload/task/{field}/avg"] == average
        assert exported[f"workload/task/{field}/min"] == minimum
        assert exported[f"workload/task/{field}/max"] == maximum


def test_rollout_workload_distribution_does_not_emit_phantom_zero_unit():
    tracker = StatsTracker()

    record_workload_distribution(tracker, prefix="workload/task", workloads=[])

    assert tracker.export() == {}


def test_trajectory_collection_shape_counts_full_recursive_tree():
    collection = {
        "trajectories": {
            "root": {"steps": [{}, {}]},
            "child": {"steps": [{}]},
            "verifier": {"steps": [{}, {}, {}]},
            "malformed": SimpleNamespace(),
        }
    }

    assert trajectory_collection_shape(collection) == (4, 6)


@pytest.mark.parametrize(
    "collection",
    [
        None,
        {},
        {"trajectories": None},
        {"trajectories": []},
        SimpleNamespace(trajectories={}),
    ],
)
def test_trajectory_collection_shape_tolerates_missing_or_malformed_collection(collection):
    assert trajectory_collection_shape(collection) == (0, 0)


@pytest.mark.parametrize(
    "field",
    [
        "environment_steps",
        "model_calls",
        "input_tokens",
        "output_tokens",
        "trajectories",
        "postmerge_datums",
        "policy_eligible_datums",
        "post_sampling_datums",
    ],
)
def test_rollout_workload_rejects_invalid_counts(field):
    with pytest.raises(ValueError, match=field):
        RolloutWorkload(**{field: -1})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"postmerge_datums": 1, "policy_eligible_datums": 2},
        {"postmerge_datums": 2, "policy_eligible_datums": 1, "post_sampling_datums": 2},
    ],
)
def test_rollout_workload_rejects_out_of_order_datum_funnel(kwargs):
    with pytest.raises(ValueError, match="post_sampling_datums"):
        RolloutWorkload(**kwargs)


@pytest.mark.parametrize("value", [True, 1.5, "1", None])
def test_rollout_workload_rejects_non_integer_counts(value):
    with pytest.raises(ValueError, match="environment_steps"):
        RolloutWorkload(environment_steps=value)
