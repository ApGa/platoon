"""Publishing a step's policy, the fifth stage of a step, after its updates and its commit (:func:`publish_policy`):
the save (weights for the sampler, or a checkpoint when one is due), the new sampling client, which the policy samples
from and the rollout workers are given (:func:`_publish_worker_binding`); and why a step took no update
(:func:`_no_update_reason`).

The step itself, ``TrainStep`` in :mod:`platoon.train.trainer.loops`, publishes after a step that stepped the optimizer,
and logs why a step that did not keeps its policy.  ``_publish_worker_binding`` is a static method of
:class:`~platoon.train.trainer.rl.PlatoonTinkerRLTrainer` still, which keeps it under its name: to replace it, replace
the method, not the function.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from platoon.train.trainer.loops import StepContext

logger = logging.getLogger(__name__)


async def publish_policy(context: "StepContext") -> None:
    """Publish the policy that a step's updates made, through the trainer's seams: save it, and a checkpoint when one
    is due (``_save_checkpoint_and_get_sampling_client``), then give its sampling client to the policy, which counts
    one more version, and to the rollout workers (``_publish_worker_binding``)."""

    trainer, state = context.trainer, context.state
    logger.info(f"Calling _save_checkpoint_and_get_sampling_client for batch {state.train_step}")
    sampling_client = await trainer._save_checkpoint_and_get_sampling_client(
        training_client=state.training_client,
        i_batch=state.train_step,
        save_every=state.save_every,
        start_batch=context.start_batch,
        watchdog=state.watchdog,
        recovery_every=trainer.config.checkpoint.recovery_every,
        policy_version=context.version() + 1,
        optimizer_steps=state.optimizer_steps,
    )
    state.model_info.llm.update_sampling_client(sampling_client)
    trainer._publish_worker_binding(context.workflow, sampling_client, context.version())
    state.sampling_client = sampling_client
    state.sampling_client_step += 1


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
