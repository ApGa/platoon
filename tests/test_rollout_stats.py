import pytest
import torch

from platoon.utils.rollout_stats import OUTCOME_STATUSES, RolloutStats, outcome_counts, summarize_rollout_group
from platoon.utils.stats_tracker import StatsTracker


def export(stats):
    tracker = StatsTracker("train")
    stats.record(tracker, prefix="batch")
    return tracker.export()


def test_empty_batch_reports_zero_groups_without_fabricating_rewards():
    assert export(RolloutStats()) == {"train/batch/groups": 0}


def test_unequal_groups_use_rollout_counts_but_at_k_metrics_count_groups():
    batch = summarize_rollout_group({"task_reward": [0.0] * 8})
    batch.merge(summarize_rollout_group({"task_reward": [1.0] * 2}))
    metrics = export(batch)
    assert metrics["train/batch/groups"] == 2
    assert metrics["train/batch/task_reward/count"] == 10
    assert metrics["train/batch/task_reward/avg"] == pytest.approx(0.2)
    assert metrics["train/batch/task_reward/min"] == 0
    assert metrics["train/batch/task_reward/max"] == 1
    assert metrics["train/batch/task_reward_at_k_mean/avg"] == 0.5


def test_root_and_trajectory_metrics_have_independent_denominators():
    stats = summarize_rollout_group(
        {
            "task_reward": [1.0, 0.0],
            "task_reward_valid": [True, False],
            "num_steps": [2, 0, 4],
            "num_input_tokens": [10, 0, 20],
            "num_output_tokens": [4, 0, 8],
            "reward/success": [1.0, 0.75, 0.0],
        }
    )
    metrics = export(stats)
    assert metrics["train/batch/task_reward/count"] == 2
    assert metrics["train/batch/task_reward_valid/avg"] == 0.5
    assert metrics["train/batch/num_steps/count"] == 3
    assert metrics["train/batch/num_steps/avg"] == 2
    assert metrics["train/batch/avg_input_tokens_per_step/avg"] == pytest.approx(10 / 3)
    assert metrics["train/batch/reward/success/count"] == 3


def test_missing_optional_components_are_not_counted_as_zero():
    prefix = "_present/"
    batch = summarize_rollout_group(
        {
            "task_reward": [1.0, 0.0],
            "root_reward/judge": [0.8, 0.0],
            prefix + "root_reward/judge": [True, False],
            "reward/judge": [0.0, 0.8, 0.0],
            prefix + "reward/judge": [False, True, False],
        },
        presence_mask_prefix=prefix,
    )
    batch.merge(summarize_rollout_group({"task_reward": [0.0]}))
    metrics = export(batch)
    assert metrics["train/batch/root_reward/judge/count"] == 1
    assert metrics["train/batch/root_reward/judge/avg"] == pytest.approx(0.8)
    assert metrics["train/batch/root_reward/judge_at_k_mean/count"] == 1
    assert metrics["train/batch/reward/judge/count"] == 1


def test_no_result_does_not_become_a_zero_reward_and_zero_groups_remain_visible():
    failed = summarize_rollout_group({"task_reward": []})
    assert export(failed) == {"train/batch/groups": 1}
    failed.merge(summarize_rollout_group({"task_reward": [0.0] * 8}))
    metrics = export(failed)
    assert metrics["train/batch/groups"] == 2
    assert metrics["train/batch/task_reward/count"] == 8
    assert metrics["train/batch/task_reward/avg"] == 0


def test_queued_stats_do_not_alias_source_tensors_or_other_batches():
    rewards = torch.tensor([0.0, 1.0])
    group = summarize_rollout_group({"task_reward": rewards})
    batch = RolloutStats()
    batch.merge(group)
    rewards.fill_(99)
    group.add("task_reward", [99])
    assert export(batch)["train/batch/task_reward/avg"] == 0.5
    assert all(isinstance(value, float) for values in batch.values.values() for value in values)


def test_outcome_counts_name_every_status_and_count_a_missing_one_as_none():
    counts = outcome_counts(["scored", "scored", "error", None, "timeout"])
    assert counts == {
        "outcome/scored": 2,
        "outcome/invalid": 0,
        "outcome/error": 1,
        "outcome/timeout": 1,
        "outcome/cancelled": 0,
        "outcome/none": 1,
    }
    assert outcome_counts(["surprise"])["outcome/surprise"] == 1  # a status of a custom rollout is kept, not dropped


def test_outcome_statuses_are_the_environment_outcome_statuses_and_none():
    from typing import get_args

    from platoon.envs.service import OutcomeStatus

    assert OUTCOME_STATUSES == (*get_args(OutcomeStatus), "none")


def test_counts_add_up_across_groups_and_record_as_totals_with_explicit_zeros():
    first = summarize_rollout_group({"task_reward": [1.0, 0.0]})
    first.counts.update(outcome_counts(["scored", "scored", "error"]))
    second = summarize_rollout_group({"task_reward": [1.0]})
    second.counts.update(outcome_counts(["scored", None]))
    batch = RolloutStats()
    batch.merge(first)
    batch.merge(second)

    metrics = export(batch)

    assert metrics["train/batch/groups"] == 2
    assert metrics["train/batch/outcome/scored"] == 3
    assert metrics["train/batch/outcome/error"] == 1
    assert metrics["train/batch/outcome/none"] == 1
    assert metrics["train/batch/outcome/invalid"] == metrics["train/batch/outcome/timeout"] == 0
    assert metrics["train/batch/outcome/cancelled"] == 0
    assert first.counts["outcome/scored"] == 2  # merging copies, it never aliases a group's counts


def test_a_subset_keeps_only_the_named_values_but_every_count():
    group = summarize_rollout_group({"task_reward": [1.0, 0.0], "num_steps": [3, 4]})
    group.counts.update(outcome_counts(["scored", "invalid"]))

    metrics = export(group.subset(["task_reward"]))

    assert metrics["train/batch/task_reward/avg"] == 0.5
    assert metrics["train/batch/groups"] == 1
    assert metrics["train/batch/outcome/scored"] == metrics["train/batch/outcome/invalid"] == 1
    assert not any(key.startswith("train/batch/num_steps") for key in metrics)
    assert "train/batch/task_reward_at_k_mean/avg" not in metrics
    assert export(group)["train/batch/num_steps/avg"] == 3.5  # the group itself is unchanged


def test_mismatched_optional_mask_is_rejected():
    with pytest.raises(ValueError, match="presence mask"):
        summarize_rollout_group({"root_reward/x": [1, 2], "mask/root_reward/x": [True]}, presence_mask_prefix="mask/")


def test_deferred_stats_record_exactly_what_recording_directly_would():
    from platoon.utils.rollout_workload import RolloutWorkload, record_workload_distribution
    from platoon.utils.stats_tracker import DeferredStats, StatsTracker

    def measure(tracker) -> None:
        summarize_rollout_group({"task_reward": [0.0, 1.0, 1.0]}).record(tracker, prefix="rollout_completion")
        record_workload_distribution(tracker, prefix="workload/task", workloads=[RolloutWorkload(trajectories=3)])
        tracker.scalar(**{"credit/baseline/avg": 2 / 3})

    direct, later = StatsTracker("train"), StatsTracker("train")
    for _ in range(2):  # two groups
        measure(direct)
    groups = [DeferredStats(), DeferredStats()]
    for group in groups:
        measure(group)

    assert len(groups[0]) > 0 and later.export() == {}  # nothing is recorded until a consumer records the groups
    for group in groups:
        group.record(later)
    assert later.export() == direct.export()


def test_deferred_stats_refuse_what_a_tracker_refuses():
    from platoon.utils.stats_tracker import DeferredStats

    stats = DeferredStats()
    with pytest.raises(ValueError, match="not registered"):
        stats.stat(denominator="missing", values=torch.ones(2))
    with pytest.raises(ValueError, match="bool tensor"):
        stats.denominator(mask=torch.ones(2))
    stats.denominator(mask=torch.ones(2, dtype=torch.bool))
    with pytest.raises(ValueError, match="pytorch tensor"):
        stats.stat(denominator="mask", values=[1.0, 2.0])
