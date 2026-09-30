"""Batch rewards follow consumed groups even when producers finish ahead."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import tinker
import torch

from platoon.train.tinker import rl as rl_module
from platoon.train.tinker.workflows.group_rollout_workflow import _TaskRolloutOutput
from platoon.utils.rollout_stats import summarize_rollout_group
from platoon.utils.rollout_workload import RolloutWorkload
from platoon.utils.stats_tracker import StatsTracker


def _group(rewards, *, empty=False, checkpoint_version=0):
    # Uniform-reward groups have zero centered advantages but real outcomes.
    datums = (
        []
        if empty
        else [
            tinker.Datum(
                model_input=tinker.ModelInput.from_ints([1, 2, 3]),
                loss_fn_inputs={
                    name: tinker.TensorData.from_torch(torch.tensor(values))
                    for name, values in {
                        "target_tokens": [2, 3, 4],
                        "logprobs": [0.0, -0.1, -0.2],
                        "advantages": [0.0, 0.0, 0.0],
                        "mask": [0.0, 1.0, 1.0],
                        "checkpoint_version": [checkpoint_version],
                    }.items()
                },
            )
            for _ in rewards
        ]
    )
    return _TaskRolloutOutput(
        datums,
        workload=RolloutWorkload(
            postmerge_datums=len(rewards),
            policy_eligible_datums=len(rewards),
            post_sampling_datums=len(rewards),
        ),
        requested_rollouts=len(rewards),
        observed_rollouts=len(rewards),
        trainable_rollouts=len(datums),
        task_retained_datums=len(datums),
        rollout_stats=summarize_rollout_group({"task_reward": rewards}),
    )


def _run_batches(monkeypatch, outputs, *, start_batch=0, max_staleness=None):
    trainer = object.__new__(rl_module.PlatoonTinkerRLTrainer)
    trainer.config = SimpleNamespace(
        train=SimpleNamespace(
            batch_size=1,
            num_minibatches=1,
            num_microbatches=1,
            num_concurrent_rollout_workflow_workers=1,
            workflow_config=SimpleNamespace(group_size=8, filter_zero_advantage_datums=True),
            max_staleness=max_staleness,
            optimizer=SimpleNamespace(learning_rate=1e-5),
        ),
        checkpoint=SimpleNamespace(recovery_every=None),
    )
    trainer.batch_transforms = []
    trainer._train_dataloader_loop = AsyncMock()
    trainer._save_checkpoint_and_get_sampling_client = AsyncMock(return_value=object())
    trainer.shutdown_loops = AsyncMock()
    monkeypatch.setattr(rl_module, "client_monitor", lambda scope: Mock())
    monkeypatch.setattr(rl_module, "client_step_metrics", lambda scope: {})
    monkeypatch.setattr(rl_module, "_ptc_cell_guard_stats", lambda scope: {})
    logged = []

    async def run():
        shared = SimpleNamespace(
            train_step=start_batch,
            sampling_client_step=start_batch,
            train_tracker=StatsTracker("train"),
            sampling_client=object(),
            training_client=object(),
            model_info=SimpleNamespace(llm=SimpleNamespace(version=start_batch, update_sampling_client=Mock())),
            watchdog=None,
            save_every=10,
            sampling_client_updated_event=asyncio.Event(),
            num_train_batches_per_epoch=10,
            stats_logger=SimpleNamespace(log=lambda *, step, stats: logged.append(stats)),
        )

        async def finish_all_groups(shared_state, workflow, task_queue, result_queue):
            # No yield: all groups finish before the first batch is trained.
            for output in outputs:
                stats = getattr(output, "rollout_stats", None)
                if stats is not None:
                    stats.record(shared_state.train_tracker, prefix="rollout_completion")
                result_queue.put_nowait(output)

        trainer._rollout_workflow_worker_loop = finish_all_groups
        async with asyncio.TaskGroup() as task_group:
            await trainer._train_loop(task_group, shared, start_batch, start_batch + len(outputs), object())

    asyncio.run(run())
    return logged


def test_prefetched_groups_are_logged_only_when_consumed_even_with_zero_advantages(monkeypatch):
    first, second = _run_batches(monkeypatch, [_group([0.0] * 8), _group([1.0] * 2)])

    assert first["train/rollout_completion/task_reward/avg"] == pytest.approx(0.2)
    assert first["train/batch/task_reward/avg"] == 0
    assert first["train/batch/task_reward/count"] == 8
    assert second["train/batch/task_reward/avg"] == 1
    assert second["train/batch/task_reward/count"] == 2
    assert "train/rollout_completion/task_reward/avg" not in second
    for metrics in (first, second):
        assert metrics["train/batch/groups"] == 1
        assert metrics["train/batch/groups_without_stats"] == 0
        assert metrics["train/workload/training_batch/total_submitted_datums"] == 0


def test_stale_groups_are_excluded_but_empty_group_outcomes_still_count(monkeypatch):
    stale, empty, failed = _run_batches(
        monkeypatch,
        [_group([1.0] * 8), _group([0.0] * 8, empty=True), None],
        start_batch=2,
        max_staleness=0,
    )
    assert stale["train/batch/stale_groups"] == 1
    assert stale["train/batch/groups"] == 0
    assert "train/batch/task_reward/avg" not in stale
    assert empty["train/batch/task_reward/avg"] == 0
    assert empty["train/batch/task_reward/count"] == 8
    assert empty["train/batch/groups"] == 1
    assert failed["train/batch/groups"] == 0
    assert failed["train/batch/groups_without_stats"] == 1
    assert "train/batch/task_reward/avg" not in failed
