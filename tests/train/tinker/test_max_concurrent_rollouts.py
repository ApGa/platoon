"""``max_concurrent_rollouts`` counts the rollouts that run at once, in ``train`` and in ``eval``: a group's rollouts
start together, so as many whole groups run at once as it holds.  Training runs every rollout of a batch at once by
default, an evaluation 256.

It replaced ``num_concurrent_rollout_workflow_workers``, which counted groups and which platoon 0.3.0 removed
(``tests/test_removed_forms.py``)."""

from __future__ import annotations

import asyncio
import signal

import pytest
from trainer_fakes import datum, make_trainer, trainer_config

from platoon.train.tinker.config_defs import EvalConfig, TrainConfig, WorkflowConfig, concurrent_groups
from platoon.utils.config import ConfigError


@pytest.fixture(autouse=True)
def _restore_signal_handlers():
    """``train()`` installs SIGINT and SIGTERM handlers; put the test process's back."""

    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def _train_config(**values) -> TrainConfig:
    return TrainConfig(model_name="m", renderer_name="r", **values)


def test_training_runs_every_rollout_of_a_batch_at_once_by_default():
    config = _train_config(batch_size=4, workflow_config=WorkflowConfig(group_size=8))

    assert config.max_concurrent_rollouts == 32
    assert concurrent_groups(config) == 4


def test_an_evaluation_runs_256_rollouts_at_once_by_default():
    assert EvalConfig().max_concurrent_rollouts == 256
    assert EvalConfig(workflow_config=WorkflowConfig(group_size=4)).max_concurrent_rollouts == 256


@pytest.mark.parametrize(("rollouts", "groups"), [(8, 1), (20, 2), (24, 3)])
def test_as_many_whole_groups_run_as_the_count_holds(rollouts, groups):
    config = _train_config(max_concurrent_rollouts=rollouts, workflow_config=WorkflowConfig(group_size=8))

    assert concurrent_groups(config) == groups


@pytest.mark.parametrize("section", ["train", "eval"])
def test_a_count_that_holds_no_whole_group_is_a_config_error(tmp_path, section):
    with pytest.raises(
        ConfigError, match=rf"{section}\.max_concurrent_rollouts must hold one group's rollouts at least"
    ):
        trainer_config(tmp_path, f"{section}.workflow_config.group_size=4", f"{section}.max_concurrent_rollouts=3")


class _Peak:
    """A rollout workflow whose groups each take a moment; records the most that ran at once, and what it saw."""

    def __init__(self) -> None:
        self.running = self.peak = self.started = 0

    async def arun_episode(self, data):
        self.started += 1
        self.running += 1
        self.peak = max(self.peak, self.running)
        await asyncio.sleep(0.05)
        self.running -= 1
        return [datum()]


@pytest.mark.parametrize(
    ("overrides", "train_groups", "eval_groups"),
    [
        (("train.max_concurrent_rollouts=16", "eval.max_concurrent_rollouts=2"), 2, 2),
        (("train.max_concurrent_rollouts=31", "eval.max_concurrent_rollouts=3"), 3, 3),  # whole groups of 8 only
    ],
    ids=["two-groups", "rounded-down"],
)
def test_the_trainer_runs_as_many_groups_at_once_as_the_counts_hold(
    tmp_path, monkeypatch, overrides, train_groups, eval_groups
):
    config = trainer_config(tmp_path, "train.batch_size=4", "train.max_training_steps=1", "eval.every=1", *overrides)
    trainer, _saves, _logged = make_trainer(monkeypatch, config, tasks=8, eval_tasks=6)
    training, evaluation = _Peak(), _Peak()

    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=training, eval_workflow=evaluation), timeout=60))

    assert training.started == 4  # one step of 4 tasks
    assert training.peak == train_groups
    assert evaluation.started == 6  # step 1's evaluation of six tasks (no eval.at_start: not the starting policy's)
    assert evaluation.peak == eval_groups
