"""The rl trainer's training loop of one start (:func:`_train_loop`), and stopping the loops (:func:`shutdown_loops`).

The loop starts the start's draws, whose groups settle into the queue the steps dequeue from, publishes the first
sampler of each policy the agents sample to the rollout workers, has each trained one hand its later samplers to them,
and sets up what its steps share (:class:`~platoon.train.step.step.StepContext`).  Then it runs one
:class:`~platoon.train.step.step.TrainStep` per step to the end batch, lets the evaluation loop handle the last step,
and stops the loops.  When the run's trained policies step independently, each runs a loop of its own
(:mod:`platoon.train.step.independent`).
"""

import asyncio
import logging
from typing import TYPE_CHECKING

from platoon.train.dispatch.permits import _rollouts_at_once
from platoon.train.dispatch.settled import SettledGroup
from platoon.train.dispatch.source import training_tasks
from platoon.train.evaluation.loop import _wait_for_last_step_evaluation
from platoon.train.labels import label_column
from platoon.train.policy.binding import _publish_worker_binding, publish_other_bindings
from platoon.train.step.metrics import RunTotals
from platoon.train.step.publish import subscribed_bindings
from platoon.train.step.step import StepContext, TrainStep
from platoon.train.tinker.rollout_metrics import monitor as client_monitor
from platoon.train.trainer.signals import TerminateTrainLoop
from platoon.train.trainer.state import TrainLoopSharedState
from platoon.train.workflow.protocol import RolloutWorkflow

if TYPE_CHECKING:
    from platoon.train.trainer.rl import PlatoonTinkerRLTrainer

logger = logging.getLogger(__name__)


async def _train_loop(
    trainer: "PlatoonTinkerRLTrainer",
    task_group: asyncio.TaskGroup,
    shared_state: TrainLoopSharedState,
    start_batch: int,
    end_batch: int,
    train_workflow: RolloutWorkflow,
):
    """Train this start's steps, ``start_batch`` to ``end_batch``: start the dispatcher of its training groups in
    ``task_group``, publish the first sampler of each policy the agents sample, then run one :class:`TrainStep` per
    step, each trained policy handing the samplers it publishes to the rollout workers (``subscribed_bindings``); then
    let the evaluation loop handle the last step, and stop the loops (:func:`shutdown_loops`, which ends this one with
    ``TerminateTrainLoop``).  When the run's trained policies step independently, each runs a loop of its own."""

    if shared_state.clocks is not None:  # each trained policy on its own clock
        return await _policy_loops(trainer, task_group, shared_state, end_batch, train_workflow)
    column = label_column(trainer.config)
    # Every task the run draws from, a weighted order's tasks that its first epoch leaves out included.
    tasks = training_tasks(trainer._task_sources())
    group_limits = trainer.config.train.rollout_group_limits
    # The groups settle into this queue, which the steps dequeue from: the dispatcher starts each group when the
    # horizon, the rollout permits and the label limits allow (platoon.train.dispatch.dispatcher).
    settled_queue = asyncio.Queue[SettledGroup]()
    dispatcher = trainer._dispatch(task_group, shared_state, settled_queue)

    # Publish the initial sampling-client binding before any group can
    # start, then declare the concurrency ceiling the metrics compare
    # against: max_concurrent_rollouts is a budget of rollouts, of which
    # a group takes its size in permits, and the monitor warns once when
    # more rollouts run at once.
    _publish_first_bindings(shared_state, train_workflow)
    budget = trainer.config.train.max_concurrent_rollouts
    client_monitor("train").set_configured_concurrency(
        _rollouts_at_once(trainer.config, "train", budget), budget=budget
    )

    logger.info(f"Starting training from batch {start_batch} to {end_batch}")
    context = StepContext.build(
        trainer,
        shared_state,
        train_workflow,
        start_batch,
        end_batch,
        queue=settled_queue,
        dispatcher=dispatcher,
        tasks=tasks,
        column=column,
        group_limits=group_limits,
    )
    totals = RunTotals()
    with subscribed_bindings(shared_state.policies, train_workflow):
        while shared_state.train_step < end_batch:
            await TrainStep(context, totals).run()

        await _wait_for_last_step_evaluation(shared_state)
        await shutdown_loops(shared_state)


async def _policy_loops(
    trainer: "PlatoonTinkerRLTrainer",
    task_group: asyncio.TaskGroup,
    shared_state: TrainLoopSharedState,
    end_batch: int,
    train_workflow: RolloutWorkflow,
) -> None:
    """The train loop of a start whose trained policies step independently: each policy's own loop
    (:func:`~platoon.train.step.independent.train_policies`)."""

    from platoon.train.step.independent import train_policies  # it builds on the lockstep step

    await train_policies(trainer, task_group, shared_state, end_batch, train_workflow)


def _publish_first_bindings(shared_state: TrainLoopSharedState, train_workflow: RolloutWorkflow) -> None:
    """Before any group starts, give the rollout workers each sampled policy's sampler: the agents' default policy's
    (``agents.policy``; the policy the loops train when the agents name none) as their default binding, then each
    other one's (:func:`~platoon.train.policy.binding.publish_other_bindings`)."""

    policies = shared_state.policies
    default = policies[policies.agents.default]
    _publish_worker_binding(train_workflow, default.sampling_client, default.version, required=True)
    publish_other_bindings(policies, train_workflow, _publish_worker_binding, required=True)


async def shutdown_loops(shared_state: TrainLoopSharedState):
    """Stop the loops: no group starts and the evaluation loop wakes to see the shutdown; then end the train loop's task
    group with ``TerminateTrainLoop``, which cancels the evaluation loop."""

    shared_state.shutdown_event.set()
    shared_state.step_ended.set()
    raise TerminateTrainLoop()
