"""After a step's update: the sampling client published to the rollout workers, and why a step took no update.

``_publish_worker_binding`` is a static method of :class:`~platoon.train.trainer.rl.PlatoonTinkerRLTrainer` still,
which keeps it under its name: to replace it, replace the method, not the function.
"""

import logging

logger = logging.getLogger(__name__)


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


def _publish_worker_binding(workflow, sampling_client, version: int, *, required: bool = False) -> None:
    """Republish the sampling client for rollout workers.

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
        publish(sampling_client, version)
    except Exception:
        logger.exception("could not publish the rollout worker sampling-client binding")
        if required and getattr(workflow, "rollout_isolation", "inprocess") == "process":
            raise
