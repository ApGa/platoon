"""Zero-signal consumption must not update Adam or age slower useful rollouts."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
import tinker
import torch

from platoon.train.tinker import rl as rl_module
from platoon.train.tinker.batch_transforms import has_zero_objective
from platoon.utils.stats_tracker import StatsTracker


def datum(advantage, version=0):
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={
            key: tinker.TensorData.from_torch(torch.tensor(values))
            for key, values in {
                "advantages": [advantage, advantage],
                "mask": [1.0, 1.0],
                "target_tokens": [2, 3],
                "logprobs": [-0.1, -0.2],
                "checkpoint_version": [version],
            }.items()
        },
    )


@pytest.mark.parametrize("loss", ["cispo", "ppo", "importance_sampling"])
def test_exact_zero_only_for_supported_objectives(loss):
    assert has_zero_objective([datum(0.0)], loss, {})
    assert not has_zero_objective([datum(1e-20)], loss, {})
    assert not has_zero_objective([datum(float("nan"))], loss, {})
    assert not has_zero_objective([datum(0.0)], loss, {"kl_coef": 0.1})
    assert not has_zero_objective([datum(0.0)], "custom_loss", {})


def run_batches(
    monkeypatch,
    outputs,
    *,
    policy_version=0,
    start_batch=0,
    max_staleness=1,
    filter_zero=False,
    loss_fn_config=None,
):
    trainer: Any = object.__new__(rl_module.PlatoonTinkerRLTrainer)  # its seams are replaced below
    trainer.config = SimpleNamespace(
        train=SimpleNamespace(
            batch_size=1,
            num_minibatches=1,
            num_microbatches=1,
            max_concurrent_rollouts=1,
            workflow_config=SimpleNamespace(group_size=1, filter_zero_advantage_datums=filter_zero),
            max_staleness=max_staleness,
            loss_fn="cispo",
            loss_fn_config=loss_fn_config or {},
            optimizer=SimpleNamespace(learning_rate=1e-6),
            forward_backward_timeout_seconds=10,
            optim_step_timeout_seconds=10,
        ),
        checkpoint=SimpleNamespace(recovery_every=1),
    )
    trainer.batch_transforms = []
    trainer._save_checkpoint_and_get_sampling_client = AsyncMock(return_value=object())
    trainer.shutdown_loops = AsyncMock()
    monkeypatch.setattr(rl_module, "client_monitor", lambda scope: Mock())
    monkeypatch.setattr(rl_module, "client_step_metrics", lambda scope: {})
    monkeypatch.setattr(rl_module, "_ptc_cell_guard_stats", lambda scope: {})
    monkeypatch.setattr(rl_module, "compute_training_metrics", lambda **kwargs: {})
    fb_future = SimpleNamespace(result_async=AsyncMock(return_value=object()))
    optim_future = SimpleNamespace(result_async=AsyncMock(return_value=SimpleNamespace(metrics={})))
    client = SimpleNamespace(
        forward_backward_async=AsyncMock(return_value=fb_future),
        optim_step_async=AsyncMock(return_value=optim_future),
    )
    llm = SimpleNamespace(version=policy_version)

    def update(_client):
        llm.version += 1

    llm.update_sampling_client = Mock(side_effect=update)
    logged = []
    shared = SimpleNamespace(
        train_step=start_batch,
        sampling_client_step=policy_version,
        optimizer_steps=policy_version,
        train_tracker=StatsTracker("train"),
        sampling_client=object(),
        training_client=client,
        model_info=SimpleNamespace(llm=llm),
        optimizer_params=object(),
        watchdog=None,
        save_every=10,
        sampling_client_updated_event=SimpleNamespace(set=Mock()),
        eval_loop_running=False,
        num_train_batches_per_epoch=10,
        stats_logger=SimpleNamespace(log=lambda *, step, stats: logged.append(stats)),
    )

    def produce(task_group, _shared, _start, _end, _workflow, results):  # in place of the dispatcher's groups
        for output in outputs:
            results.put_nowait(output)

    trainer._dispatch = produce

    async def run():
        async with asyncio.TaskGroup() as group:
            await trainer._train_loop(group, shared, start_batch, start_batch + len(outputs), object())

    asyncio.run(run())
    return trainer, shared, logged


@pytest.mark.parametrize("filter_zero", [False, True])
def test_zero_nonzero_zero_updates_only_once(monkeypatch, filter_zero):
    trainer, shared, logs = run_batches(
        monkeypatch,
        [[datum(0.0, 4)], [datum(1.0, 4)], [datum(0.0, 4)]],
        start_batch=20,
        policy_version=4,
        filter_zero=filter_zero,
    )
    assert shared.train_step == 23  # dataset consumption is independent
    assert shared.model_info.llm.version == 5
    assert shared.sampling_client_step == 5
    assert shared.sampling_client_updated_event.set.call_count == 3
    assert shared.optimizer_steps == 5
    assert shared.training_client.forward_backward_async.await_count == 1
    assert shared.training_client.optim_step_async.await_count == 1
    assert trainer._save_checkpoint_and_get_sampling_client.await_count == 1
    save = trainer._save_checkpoint_and_get_sampling_client.await_args.kwargs
    assert (save["i_batch"], save["policy_version"], save["optimizer_steps"]) == (21, 5, 5)
    assert [r["train/batch/optimizer_updates"] for r in logs] == [0, 1, 0]
    assert [r["train/policy/version"] for r in logs] == [4, 5, 5]
    assert [r["train/batch/stale_groups"] for r in logs] == [0, 0, 0]


def test_staleness_counts_published_changes_not_consumed_batches(monkeypatch):
    outputs = [[datum(0.0, 7)] for _ in range(5)] + [
        [datum(1.0, 7)],
        [datum(1.0, 7)],
        [datum(1.0, 7)],
    ]
    _, shared, logs = run_batches(monkeypatch, outputs, start_batch=100, policy_version=7)
    assert shared.train_step == 108
    assert shared.model_info.llm.version == 9
    assert shared.training_client.optim_step_async.await_count == 2
    assert [r["train/batch/stale_groups"] for r in logs] == [0] * 7 + [1]


def test_unknown_auxiliary_objective_is_not_skipped_or_filtered(monkeypatch):
    _, shared, _ = run_batches(
        monkeypatch,
        [[datum(0.0)]],
        filter_zero=True,
        loss_fn_config={"kl_coef": 0.1},
    )
    assert shared.training_client.forward_backward_async.await_count == 1
    assert shared.training_client.optim_step_async.await_count == 1


@pytest.mark.parametrize("versions", [(5, 1), (1, 5), (5, 3), (3, 5)])
def test_group_staleness_uses_oldest_datum_regardless_of_order(monkeypatch, versions):
    _, shared, logs = run_batches(
        monkeypatch,
        [[datum(1.0, version) for version in versions]],
        policy_version=5,
        max_staleness=3,
    )
    stale = min(versions) < 2
    assert logs[0]["train/batch/stale_groups"] == int(stale)
    assert shared.training_client.optim_step_async.await_count == int(not stale)


def test_resume_policy_counters_are_independent_with_legacy_fallback():
    record = SimpleNamespace(batch=21, extra={"policy_version": 3, "optimizer_steps": 5})
    assert rl_module._resume_policy_counters(record, 2) == (3, 5)
    assert rl_module._resume_policy_counters(SimpleNamespace(batch=21, extra={}), 2) == (21, 42)
    assert rl_module._resume_policy_counters(None, 2) == (0, 0)
    with pytest.raises(ValueError):
        rl_module._resume_policy_counters(SimpleNamespace(batch=21, extra={"policy_version": -1}), 2)


def test_recovery_checkpoint_persists_actual_counters(tmp_path):
    trainer = object.__new__(rl_module.PlatoonTinkerRLTrainer)
    trainer.config = SimpleNamespace(log_path=str(tmp_path), stats=SimpleNamespace(experiment_name="x", trial_name="y"))
    trainer._stats_logger = SimpleNamespace(wandb_run_id="run")
    Path(trainer.run_log_path).mkdir(parents=True)
    state_path = "tinker://m/weights/recovery-000021"
    sampler_path = "tinker://m/sampler_weights/recovery-000021"
    client = SimpleNamespace(
        save_state_async=AsyncMock(
            return_value=SimpleNamespace(
                result_async=AsyncMock(return_value=SimpleNamespace(path=state_path)),
            )
        ),
        save_weights_for_sampler_async=AsyncMock(
            return_value=SimpleNamespace(
                result_async=AsyncMock(return_value=SimpleNamespace(path=sampler_path)),
            )
        ),
        create_sampling_client=Mock(return_value=object()),
    )
    asyncio.run(
        trainer._save_checkpoint_and_get_sampling_client(
            client,
            i_batch=20,
            save_every=100,
            recovery_every=1,
            policy_version=3,
            optimizer_steps=5,
        )
    )
    # Exercise the real cookbook JSONL serialization and lookup, not a mock
    # record: custom counters must survive in CheckpointRecord.extra.
    saved = rl_module.checkpoint_utils.get_last_checkpoint(trainer.run_log_path)
    assert saved.batch == 21
    assert saved.state_path == state_path
    assert saved.extra == {"wandb_run_id": "run", "policy_version": 3, "optimizer_steps": 5}
    assert rl_module._resume_policy_counters(saved, 1) == (3, 5)
