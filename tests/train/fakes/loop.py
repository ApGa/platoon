"""Run the rl trainer's train loop over groups a test scripts, on CPU, as a start runs it but without the start's
dispatcher.

:func:`loop_trainer` builds the trainer with its constructor, over a fake policy (``fakes.trainer``) whose training
client is the test's and whose full saves are recorded, not written (:func:`saves_of`), in the start its steps save
through; the rollout client's metrics start afresh.  :func:`loop_state` is what a start's loops share (a
``TrainLoopSharedState`` over the trainer's policies and a step clock whose event counts its sets), at the step and the
policy version the test starts from.  :func:`run_train_loop` runs the train loop
(:func:`~platoon.train.trainer.loops._train_loop`) over ``outputs``, which settle into the loop's queue in order before
its first step, or one at a time, late, and stops the loops after the last step, as a start does.

The trainer's ``_dispatch``, which would start the dispatcher of the start's groups, is replaced
(:func:`run_train_loop`): the test's outputs settle into the queue instead, and the loop tells no dispatcher of its
steps.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from fakes.trainer import FakePolicy, OneClientService, SaveLog, StepLog, TrainingClient
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.policy.policy import adam_params
from platoon.train.policy.set import PolicySet
from platoon.train.tinker.rollout_metrics import reset_monitors_for_tests
from platoon.train.trainer.loops import _train_loop
from platoon.train.trainer.rl import PlatoonTinkerRLTrainer
from platoon.train.trainer.run import StepClock, TrainerRun
from platoon.train.trainer.signals import TerminateTrainLoop
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
    when None), whose steps save through a start of the trainer, as ``train()`` records it: its full saves are
    recorded (:func:`saves_of`), and a weights-only save asks the training client."""

    service = OneClientService(training_client if training_client is not None else TrainingClient())
    policy = FakePolicy(config, service.service_client, saves=SaveLog())
    trainer = PlatoonTinkerRLTrainer(config, list(train_dataset), eval_dataset, policies=PolicySet([policy]))
    policy.training_client = service.training_client
    trainer._run = TrainerRun(trainer)
    reset_monitors_for_tests()
    return trainer


def saves_of(trainer: PlatoonTinkerRLTrainer) -> SaveLog:
    """The full checkpoints the steps of a :func:`loop_trainer` saved, in order, each with its ``loop_state``."""

    policy = trainer.policies[trainer.policies.leader]
    assert isinstance(policy, FakePolicy) and policy.saves is not None
    return policy.saves


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
    before the first step, or each ``settle_after`` seconds after the one before, until the loops stop; each step's
    logged stats."""

    async def settle_late(results):
        for output in outputs:
            await asyncio.sleep(settle_after)
            results.put_nowait(output)

    def settle_all(task_group, shared_state, results):
        if settle_after is not None:
            task_group.create_task(settle_late(results))
            return
        for output in outputs:
            results.put_nowait(output)

    trainer._dispatch = settle_all
    start = state.train_step

    async def run() -> None:
        try:
            async with asyncio.TaskGroup() as task_group:
                await _train_loop(trainer, task_group, state, start, start + steps, object())
        except* TerminateTrainLoop:  # the loops stop after the last step
            pass

    asyncio.run(run())
    return state.stats_logger.stats
