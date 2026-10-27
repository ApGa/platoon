"""Run the rl trainer's train loop over groups a test scripts, on CPU, as a start runs it but without the start's
dispatcher or saves.

:func:`loop_trainer` builds the trainer with its constructor, over a fake policy (``fakes.trainer``) whose training
client is the test's; the rollout client's metrics start afresh.  :func:`loop_state` is what a start's loops share (a
``TrainLoopSharedState`` over the trainer's policies and a step clock whose event counts its sets), at the step and the
policy version the test starts from.  :func:`run_train_loop` runs ``_train_loop`` over ``outputs``, which settle into
the loop's queue in order before its first step, or one at a time, late, and stops with the last step.

The trainer's seams they replace:

- ``_save_checkpoint_and_get_sampling_client`` (:func:`loop_trainer`): each step's save, an ``AsyncMock`` that returns a
  sampling client, so a test reads what each step saved from its calls;
- ``_dispatch`` (:func:`run_train_loop`): the start's dispatcher, which would start the groups; the test's outputs
  settle into the queue instead;
- ``shutdown_loops`` (:func:`run_train_loop`): an ``AsyncMock``, as no rollout or evaluation loop runs beside.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any
from unittest.mock import AsyncMock

from fakes.trainer import FakePolicy, OneClientService, StepLog, TrainingClient
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.policy.policy import adam_params
from platoon.train.policy.set import PolicySet
from platoon.train.tinker.rollout_metrics import reset_monitors_for_tests
from platoon.train.trainer.rl import PlatoonTinkerRLTrainer
from platoon.train.trainer.run import StepClock
from platoon.train.trainer.state import TrainLoopSharedState
from platoon.utils.stats_tracker import StatsTracker


class CountingEvent(asyncio.Event):
    """An event that counts how often it was set (``sets``)."""

    def __init__(self) -> None:
        super().__init__()
        self.sets = 0

    def set(self) -> None:
        self.sets += 1
        super().set()


def loop_trainer(
    config: PlatoonTinkerRLTrainerConfig,
    *,
    train_dataset: Any = (),
    eval_dataset: Any = None,
    training_client: Any = None,
) -> PlatoonTinkerRLTrainer:
    """The trainer of ``config`` over ``train_dataset`` (the rows its steps' metrics are labelled by) and
    ``eval_dataset``, on a fake policy whose training client is ``training_client`` (a ``fakes.trainer.TrainingClient``
    when None), whose steps' saves are an ``AsyncMock``."""

    service = OneClientService(training_client if training_client is not None else TrainingClient())
    trainer = PlatoonTinkerRLTrainer(
        config,
        list(train_dataset),
        eval_dataset,
        policies=PolicySet([FakePolicy(config, service.service_client)]),
    )
    trainer.policies[trainer.policies.leader].training_client = service.training_client
    trainer._save_checkpoint_and_get_sampling_client = AsyncMock(return_value=object())
    reset_monitors_for_tests()
    return trainer


def loop_state(
    trainer: PlatoonTinkerRLTrainer,
    *,
    start_batch: int = 0,
    policy_version: int = 0,
    optimizer_steps: int | None = None,
    batches_per_epoch: int = 10,
    watchdog: Any = None,
    train_tracker: Any = None,
) -> TrainLoopSharedState:
    """What the loops of a start of ``trainer`` share, at ``start_batch`` and ``policy_version`` (with as many optimizer
    steps, unless ``optimizer_steps`` says), with a training dataloader of ``batches_per_epoch`` batches, ``watchdog``,
    ``train_tracker`` (a new ``StatsTracker`` when None) and a stats logger that records each step's stats
    (:class:`~fakes.trainer.StepLog`)."""

    policy = trainer.policies[trainer.policies.leader]
    assert policy.spec.optimizer is not None
    policy.optimizer_params = adam_params(policy.spec.optimizer)
    policy.optimizer_steps = policy_version if optimizer_steps is None else optimizer_steps
    policy.sampling_client = object()
    policy.info.llm.version = policy_version
    return TrainLoopSharedState(
        config=trainer.config,
        shutdown_event=asyncio.Event(),
        train_dataloader=[None] * batches_per_epoch,  # the loops read its length alone: the batches of an epoch
        eval_dataloader=None,
        policies=trainer.policies,
        clock=StepClock(step=start_batch, updated=CountingEvent()),
        sampling_client_step=policy_version,
        stats_logger=StepLog(),
        train_tracker=train_tracker if train_tracker is not None else StatsTracker("train"),
        watchdog=watchdog,
    )


def run_train_loop(
    trainer: PlatoonTinkerRLTrainer,
    state: TrainLoopSharedState,
    outputs: Sequence[Any],
    *,
    steps: int,
    settle_after: float | None = None,
) -> list[dict[str, Any]]:
    """Run ``trainer``'s train loop from the state's step for ``steps`` steps over ``outputs``, which settle in order
    before the first step, or each ``settle_after`` seconds after the one before; each step's logged stats."""

    async def settle_late(results):
        for output in outputs:
            await asyncio.sleep(settle_after)
            results.put_nowait(output)

    def settle_all(task_group, shared_state, start, end, workflow, results):
        if settle_after is not None:
            task_group.create_task(settle_late(results))
            return
        for output in outputs:
            results.put_nowait(output)

    trainer._dispatch = settle_all
    trainer.shutdown_loops = AsyncMock()
    start = state.train_step

    async def run() -> None:
        async with asyncio.TaskGroup() as task_group:
            await trainer._train_loop(task_group, state, start, start + steps, object())

    asyncio.run(run())
    return state.stats_logger.stats
