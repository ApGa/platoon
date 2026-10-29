"""Publishing the policies a step updated, the fifth stage of a step, after its updates and its commit
(:func:`publish_policies`): the save (weights for the samplers, or a checkpoint when one is due), then each updated
policy's new sampling client, which the policy samples from and hands to whoever subscribed; the subscription that
hands it to the rollout workers (:func:`subscribed_bindings`, through :func:`_publish_worker_binding`); and why a step
took no update (:func:`_no_update_reason`).

The step itself, ``TrainStep`` in :mod:`platoon.train.trainer.loops`, publishes the policies whose optimizer it
stepped, and logs why one it did not keeps its sampler and version.  ``_publish_worker_binding`` is a static method of
:class:`~platoon.train.trainer.rl.PlatoonTinkerRLTrainer` still, which keeps it under its name: to replace it, replace
the method, not the function.
"""

import logging
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from platoon.train.checkpoints.barrier import Counters
from platoon.train.checkpoints.saving import save_after_step
from platoon.train.policy.policy import Policy, Subscriber

if TYPE_CHECKING:
    from platoon.train.policy.set import PolicySet
    from platoon.train.step.lanes import PolicyLane
    from platoon.train.trainer.loops import StepContext

logger = logging.getLogger(__name__)


async def publish_policies(context: "StepContext", updated: "Sequence[PolicyLane]") -> None:
    """Publish the policies whose optimizer a step stepped (``updated``, their lanes): save them, a checkpoint of every
    trained policy when the cadence says one is due, then give each its new sampling client (:func:`_publish`), which
    counts one more version of it.  A run of one trained policy saves through the trainer's seam
    (``_save_checkpoint_and_get_sampling_client``), as it always did; a run of several through
    :func:`~platoon.train.checkpoints.saving.save_after_step`, which that seam calls for one."""

    trainer, state = context.trainer, context.state
    logger.info(f"Calling _save_checkpoint_and_get_sampling_client for batch {state.train_step}")
    if len(context.lanes) == 1:
        (lane,) = updated
        sampling_clients = {
            lane.policy_id: await trainer._save_checkpoint_and_get_sampling_client(
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
            trainer._checkpoints(state.training_client),
            state.train_step,
            updated={lane.policy_id: Counters(lane.version() + 1, lane.policy.optimizer_steps) for lane in updated},
            save_every=state.save_every,
            start_batch=context.start_batch,
            watchdog=state.watchdog,
            recovery_every=trainer.config.checkpoint.recovery_every,
        )
    for lane in updated:
        _publish(context, lane, sampling_clients[lane.policy_id])


def _publish(context: "StepContext", lane: "PolicyLane", sampling_client: Any) -> None:
    """Give ``sampling_client`` to the lane's policy (:meth:`~platoon.train.policy.policy.Policy.publish`): its model
    gateway samples it at one version more, and its subscribers hand it to the rollout workers.  The leader's
    publications are counted on the loops' shared state too."""

    policy = lane.policy
    if isinstance(policy, Policy):
        policy.publish(sampling_client)
    else:  # a state a test made as platoon 0.4.0 did, which holds the policy's fields itself and has no subscriber
        policy.model_info.llm.update_sampling_client(sampling_client)
        context.trainer._publish_worker_binding(context.workflow, sampling_client, context.version())
        policy.sampling_client = sampling_client
    if lane.leads:
        context.state.sampling_client_step += 1


@contextmanager
def subscribed_bindings(trainer: Any, policies: "PolicySet | None", workflow: Any) -> Generator[None]:
    """While the steps run: each trained policy that the agents sample hands each sampler it publishes to the rollout
    workers of the training ``workflow``, through the trainer's ``_publish_worker_binding``, looked up at each
    publication: the agents' default policy as their default binding, each other one as its own.  Nothing for a state a
    test made as platoon 0.4.0 did, without the run's policies (``policies`` None)."""

    if policies is None:
        yield
        return
    default = policies.agents.default
    ends = [
        policy.subscribe(_binding_writer(trainer, workflow, None if policy.id == default else policy.id))
        for policy in policies.sampled()
        if policy.trainable
    ]
    try:
        yield
    finally:
        for end in ends:
            end()


def _binding_writer(trainer: Any, workflow: Any, policy: str | None) -> Subscriber:
    """What hands a policy's new sampler to ``workflow``'s rollout workers: the agents' default binding (``policy``
    None), else ``policy``'s own."""

    def write(sampling_client: Any, version: int) -> None:
        if policy is None:
            trainer._publish_worker_binding(workflow, sampling_client, version)
        else:
            trainer._publish_worker_binding(workflow, sampling_client, version, policy=policy)

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


def _publish_worker_binding(
    workflow, sampling_client, version: int, *, required: bool = False, policy: str | None = None
) -> None:
    """Republish the sampling client for rollout workers: the agents' default policy's, or ``policy``'s, another
    policy they sample, to a workflow that binds several (``binds_policies``).

    A no-op for ``inprocess`` mode and for custom workflows that do not
    run workers.  A *re*-publication failure is logged and swallowed: every
    worker keeps the client it already holds, and under the full-finetune
    gateway that client samples the newest weights anyway.

    The FIRST publication is different, and only under
    ``rollout_isolation: process``.  Each worker builds its sampling client
    by reading this file at startup and dies with ``FileNotFoundError``
    when it is missing, so swallowing that failure would leave every
    rollout of the run crashing and the trainer producing no training data
    at all -- for as long as the job is allowed to run.  That one is fatal,
    and fatal here, before the first rollout is dispatched.
    """

    publish = getattr(workflow, "publish_model_binding", None)
    if publish is None:
        return
    try:
        if policy is None:
            publish(sampling_client, version)
        else:
            publish(sampling_client, version, policy=policy)
    except Exception:
        logger.exception("could not publish the rollout worker sampling-client binding")
        if required and getattr(workflow, "rollout_isolation", "inprocess") == "process":
            raise
