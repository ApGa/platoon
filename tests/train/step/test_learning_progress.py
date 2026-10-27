"""Zero-signal consumption must not update Adam or age slower useful rollouts."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import tinker
import torch
from fakes.config import make_config
from fakes.loop import loop_state, loop_trainer, run_train_loop
from fakes.trainer import TrainingClient, make_trainer
from tinker_cookbook import checkpoint_utils

from platoon.train.checkpoints import resume as resume_module
from platoon.train.step.batch_transforms import has_zero_objective


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
    outputs,
    *,
    policy_version=0,
    start_batch=0,
    max_staleness=1,
    filter_zero=False,
    loss_fn_config=None,
    training_client=None,
):
    """Run the train loop over ``outputs``, one group a step (``fakes.loop``): ``(trainer, state, each step's logged
    stats)``; the trainer's training client is ``training_client`` (``trainer.policies["main"].training_client``)."""

    config = make_config(
        train={
            "batch_size": 1,
            "num_minibatches": 1,
            "num_microbatches": 1,
            "max_concurrent_rollouts": 1,
            "workflow_config": {"group_size": 1, "filter_zero_advantage_datums": filter_zero},
            "max_staleness": max_staleness,
            "loss_fn": "cispo",
            "loss_fn_config": loss_fn_config or {},
            "optimizer": {"learning_rate": 1e-6},
            "forward_backward_timeout_seconds": 10,
            "optim_step_timeout_seconds": 10,
        },
        checkpoint={"recovery_every": 1},
    )
    trainer = loop_trainer(config, training_client=training_client)
    state = loop_state(trainer, start_batch=start_batch, policy_version=policy_version)
    stats = run_train_loop(trainer, state, outputs, steps=len(outputs))
    return trainer, state, stats


@pytest.mark.parametrize("filter_zero", [False, True])
def test_zero_nonzero_zero_updates_only_once(filter_zero):
    trainer, shared, logs = run_batches(
        [[datum(0.0, 4)], [datum(1.0, 4)], [datum(0.0, 4)]],
        start_batch=20,
        policy_version=4,
        filter_zero=filter_zero,
    )
    assert shared.train_step == 23  # dataset consumption is independent
    assert shared.model_info.llm.version == 5
    assert shared.sampling_client_step == 5
    assert shared.clock.updated.sets == 3
    assert shared.optimizer_steps == 5
    assert len(shared.training_client.forward_backwards) == 1
    assert shared.training_client.optim_steps == 1
    assert trainer._save_checkpoint_and_get_sampling_client.await_count == 1
    save = trainer._save_checkpoint_and_get_sampling_client.await_args.kwargs
    assert (save["i_batch"], save["policy_version"], save["optimizer_steps"]) == (21, 5, 5)
    assert [r["train/batch/optimizer_updates"] for r in logs] == [0, 1, 0]
    assert [r["train/policy/version"] for r in logs] == [4, 5, 5]
    assert [r["train/batch/stale_groups"] for r in logs] == [0, 0, 0]


def test_a_step_that_updates_hands_the_loops_the_sampling_client_it_saved():
    """The evaluation loop and the next step's binding sample the policy the step published."""

    trainer, shared, _logs = run_batches([[datum(1.0)]])

    assert shared.sampling_client is trainer._save_checkpoint_and_get_sampling_client.return_value


class _SlowClient(TrainingClient):
    """A training client whose forward_backward takes 50 ms."""

    async def forward_backward_async(self, datums, loss_fn, loss_fn_config):
        await asyncio.sleep(0.05)
        return await super().forward_backward_async(datums, loss_fn, loss_fn_config)


def test_a_step_s_time_covers_its_update():
    _, _, logs = run_batches([[datum(1.0)]], training_client=_SlowClient())

    assert logs[0]["timing/batch_total"] >= 0.05


def test_staleness_counts_published_changes_not_consumed_batches():
    outputs = [[datum(0.0, 7)] for _ in range(5)] + [
        [datum(1.0, 7)],
        [datum(1.0, 7)],
        [datum(1.0, 7)],
    ]
    _, shared, logs = run_batches(outputs, start_batch=100, policy_version=7)
    assert shared.train_step == 108
    assert shared.model_info.llm.version == 9
    assert shared.training_client.optim_steps == 2
    assert [r["train/batch/stale_groups"] for r in logs] == [0] * 7 + [1]


def test_unknown_auxiliary_objective_is_not_skipped_or_filtered():
    _, shared, _ = run_batches(
        [[datum(0.0)]],
        filter_zero=True,
        loss_fn_config={"kl_coef": 0.1},
    )
    assert len(shared.training_client.forward_backwards) == 1
    assert shared.training_client.optim_steps == 1


@pytest.mark.parametrize("versions", [(5, 1), (1, 5), (5, 3), (3, 5)])
def test_group_staleness_uses_oldest_datum_regardless_of_order(versions):
    _, shared, logs = run_batches(
        [[datum(1.0, version) for version in versions]],
        policy_version=5,
        max_staleness=3,
    )
    stale = min(versions) < 2
    assert logs[0]["train/batch/stale_groups"] == int(stale)
    assert shared.training_client.optim_steps == int(not stale)


def test_resume_policy_counters_are_independent_with_legacy_fallback():
    record = SimpleNamespace(batch=21, extra={"policy_version": 3, "optimizer_steps": 5})
    assert resume_module._resume_policy_counters(record, 2) == (3, 5)
    assert resume_module._resume_policy_counters(SimpleNamespace(batch=21, extra={}), 2) == (21, 42)
    assert resume_module._resume_policy_counters(None, 2) == (0, 0)
    with pytest.raises(ValueError):
        resume_module._resume_policy_counters(SimpleNamespace(batch=21, extra={"policy_version": -1}), 2)


def test_recovery_checkpoint_persists_actual_counters(tmp_path):
    trainer, _saves, logged = make_trainer(
        make_config(log_path=str(tmp_path), stats={"experiment_name": "x", "trial_name": "y"})
    )
    logged.wandb_run_id = "run"
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
    saved = checkpoint_utils.get_last_checkpoint(trainer.run_log_path)
    assert saved.batch == 21
    assert saved.state_path == state_path
    assert saved.extra == {"wandb_run_id": "run", "policy_version": 3, "optimizer_steps": 5}
    assert resume_module._resume_policy_counters(saved, 1) == (3, 5)
