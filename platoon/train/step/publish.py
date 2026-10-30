"""Publishing the policies a step updated, the fifth stage of a step, after its updates and its commit
(:func:`publish_policies`): the save (weights for the samplers, or a checkpoint when one is due), then each updated
policy's new sampling client, which the policy samples from and hands to whoever subscribed; the subscription that
hands it to the rollout workers (:func:`subscribed_bindings`); and why a step took no update
(:func:`_no_update_reason`).

The step itself is :class:`~platoon.train.step.step.TrainStep`, which publishes at its end
(:func:`~platoon.train.trainer.hooks.finish_step`) each policy that it or one of the run's step hooks updated, and logs
why one that neither did keeps its sampler and version.
"""

import logging
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from platoon.train.checkpoints.barrier import Counters
from platoon.train.checkpoints.saving import _save_checkpoint_and_get_sampling_client, save_after_step
from platoon.train.policy.binding import _publish_worker_binding
from platoon.train.policy.policy import Subscriber

if TYPE_CHECKING:
    from platoon.train.policy.set import PolicySet
    from platoon.train.step.lanes import PolicyLane
    from platoon.train.step.step import StepContext

logger = logging.getLogger(__name__)


async def publish_policies(context: "StepContext", updated: "Sequence[PolicyLane]") -> None:
    """Publish the policies whose optimizer a step stepped (``updated``, their lanes): save them, a checkpoint of every
    trained policy when the cadence says one is due, then give each its new sampling client
    (:meth:`~platoon.train.policy.policy.Policy.publish`), which counts one more version of it, and which its
    subscribers hand to the rollout workers.  A run of one trained policy saves through
    :func:`~platoon.train.checkpoints.saving._save_checkpoint_and_get_sampling_client`, as it always did; a run of
    several through :func:`~platoon.train.checkpoints.saving.save_after_step`, which that function calls for one."""

    trainer, state = context.trainer, context.state
    logger.info(f"Calling _save_checkpoint_and_get_sampling_client for batch {state.train_step}")
    if len(context.lanes) == 1:
        (lane,) = updated
        sampling_clients = {
            lane.policy_id: await _save_checkpoint_and_get_sampling_client(
                trainer,
                training_client=state.training_client,
                i_batch=state.train_step,
                save_every=state.save_every,
                start_batch=context.start_batch,
                watchdog=state.watchdog,
                recovery_every=trainer.config.checkpoint.recovery_every,
                policy_version=context.version() + 1,
                optimizer_steps=state.optimizer_steps,
            )
        }
    else:
        sampling_clients = await save_after_step(
            trainer,
            trainer._checkpoints(),
            state.train_step,
            updated={lane.policy_id: Counters(lane.version() + 1, lane.policy.optimizer_steps) for lane in updated},
            save_every=state.save_every,
            start_batch=context.start_batch,
            watchdog=state.watchdog,
            recovery_every=trainer.config.checkpoint.recovery_every,
        )
    for lane in updated:
        lane.policy.publish(sampling_clients[lane.policy_id])


@contextmanager
def subscribed_bindings(policies: "PolicySet", workflow: Any) -> Generator[None]:
    """While the steps run: each trained policy that the agents sample hands each sampler it publishes to the rollout
    workers of the training ``workflow`` (:func:`~platoon.train.policy.binding._publish_worker_binding`): the agents'
    default policy as their default binding, each other one as its own."""

    default = policies.agents.default
    ends = [
        policy.subscribe(_binding_writer(workflow, None if policy.id == default else policy.id))
        for policy in policies.sampled()
        if policy.trainable
    ]
    try:
        yield
    finally:
        for end in ends:
            end()


def _binding_writer(workflow: Any, policy: str | None) -> Subscriber:
    """What hands a policy's new sampler to ``workflow``'s rollout workers: the agents' default binding (``policy``
    None), else ``policy``'s own."""

    def write(sampling_client: Any, version: int) -> None:
        if policy is None:
            _publish_worker_binding(workflow, sampling_client, version)
        else:
            _publish_worker_binding(workflow, sampling_client, version, policy=policy)

    return write


def _no_update_reason(zero_objective_microbatches: int, empty_microbatches: int, *, counted: int | None = None) -> str:
    """Why a step took no optimizer update, for its log line; ``counted``, of a step under a group admission rule that
    admitted no group, the groups it counted."""

    if counted is not None:
        return f"it admitted no group (every one of the {counted} groups it counted was rejected, stale or failed)"
    if zero_objective_microbatches and not empty_microbatches:
        # The usual case: every group scored alike, so every advantage is zero.  No setting forces a step then.
        return (
            "its objective is zero (every advantage is exactly zero: say every group's rollouts scored alike), and "
            "a step whose objective is zero never steps the optimizer, whatever filter_zero_advantage_datums says"
        )
    if empty_microbatches and not zero_objective_microbatches:
        return "it had no datums to train on"
    return f"{zero_objective_microbatches} microbatch(es) had a zero objective and {empty_microbatches} had no datums"
