"""Saving after a step: the cadence (a permanent or a recovery checkpoint, else the weights only), and a full
checkpoint's ``loop_state``, with the per-task statistics' snapshot and the dispatcher's state.

The functions here that take the trainer first are its methods still: ``PlatoonTinkerRLTrainer`` keeps each under its
name, so to replace one, replace the method, on the class or an instance, not the function.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import tinker
from tinker_cookbook import checkpoint_utils

from platoon.train.checkpoints.dispatch_state import save_state
from platoon.train.checkpoints.pruning import RECOVERY_CHECKPOINT_PREFIX
from platoon.train.task_stats import DISPATCH_KEY
from platoon.train.trainer.signals import Watchdog

if TYPE_CHECKING:
    from platoon.train.trainer.rl import PlatoonTinkerRLTrainer

logger = logging.getLogger(__name__)


# The key of a full checkpoint row's loop_state that, under a group admission rule, records where the run's order
# stood (the dispatcher's next position), so that a resume that cannot read the dispatcher's state goes on from there:
# a step under a rule draws up to its cap's worth of positions, so the checkpoint's batch no longer says where.
ORDER_POSITION_KEY = "next_position"


# Upper bound for one save_checkpoint_async (full-parameter save_state of a
# 35B model is ~490 GB and takes minutes at 1-2 GB/s).
SAVE_CHECKPOINT_TIMEOUT_S = 7200.0


async def _save_checkpoint_and_get_sampling_client(
    trainer: "PlatoonTinkerRLTrainer",
    training_client: tinker.TrainingClient,
    i_batch: int,
    save_every: int | None,
    start_batch: int = 0,
    watchdog: Watchdog | None = None,
    recovery_every: int | None = None,
    policy_version: int | None = None,
    optimizer_steps: int | None = None,
) -> tinker.SamplingClient:
    """Save a checkpoint with wandb_run_id and return the new sampling client.

    Three cadences, in precedence order (``steps_completed`` counts steps
    since ``start_batch``; the name/batch are ``i_batch + 1`` because the
    weights are the ones *after* training on ``i_batch``, so a resume starts
    from ``i_batch + 1``):

    1. ``steps_completed % save_every == 0`` - a *permanent* full checkpoint
       named ``<batch>``, kept forever. ``save_every`` is ``None`` for
       ``checkpoint.every: 0``, which saves no permanent checkpoints.
    2. ``recovery_every`` set and ``steps_completed % recovery_every == 0``
       - an AReaL-style *recovery* full checkpoint named
       ``recovery-<batch>``. Same weights+optimizer payload, but
       ``checkpoint.keep_recovery`` deletes all but the newest few once the
       save is recorded (:mod:`platoon.train.checkpoints.pruning`),
       so the recovery cadence can run every step without growing the
       checkpoint store.
    3. otherwise - a weights-only update (no disk write in the gateway's
       full mode), which cannot be resumed from.

    Unlike tinker_cookbook's version this puts the wandb_run_id in the
    checkpoint loop_state so WandB run resumption works.
    """
    # Count steps completed since start_batch (1-indexed for human intuition)
    steps_completed = i_batch - start_batch + 1
    next_batch = i_batch + 1
    if save_every is not None and steps_completed % save_every == 0:
        checkpoint_kind, checkpoint_name = "permanent", f"{next_batch:06d}"
    elif recovery_every is not None and steps_completed % recovery_every == 0:
        checkpoint_kind = "recovery"
        checkpoint_name = f"{RECOVERY_CHECKPOINT_PREFIX}{next_batch:06d}"
    else:
        checkpoint_kind, checkpoint_name = None, None

    if checkpoint_name is None:
        logger.info(
            f"Saving weights only after completing batch {i_batch} "
            f"(steps_completed={steps_completed}, save_every={save_every}, "
            f"recovery_every={recovery_every})"
        )
        try:
            sampling_client = await training_client.save_weights_and_get_sampling_client_async()
        except Exception as e:
            logger.exception(f"save_weights_and_get_sampling_client_async failed: {e}")
            raise
        logger.debug("Weights saved and sampling client created")
        return sampling_client

    # Save a full (weights + optimizer) checkpoint with loop state.
    logger.info(
        f"Saving {checkpoint_kind} checkpoint {checkpoint_name} after completing batch {i_batch} "
        f"(steps_completed={steps_completed}, save_every={save_every}, "
        f"recovery_every={recovery_every})"
    )
    try:
        # A full-parameter save_state takes minutes; keep the watchdog fed.
        path_dict = await trainer._await_with_heartbeat(
            checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=checkpoint_name,
                log_path=trainer.run_log_path,
                loop_state=await trainer._checkpoint_state(
                    checkpoint_name,
                    {
                        "batch": next_batch,
                        "wandb_run_id": trainer._stats_logger.wandb_run_id,
                        "policy_version": next_batch if policy_version is None else policy_version,
                        "optimizer_steps": next_batch if optimizer_steps is None else optimizer_steps,
                    },
                ),
                kind="both",
            ),
            name="save_checkpoint",
            watchdog=watchdog,
            total_timeout=SAVE_CHECKPOINT_TIMEOUT_S,
        )
    except Exception as e:
        logger.exception(f"save_checkpoint_async failed: {e}")
        raise
    if checkpoint_kind == "recovery":
        trainer._recovery_saved(path_dict, next_batch if policy_version is None else policy_version)
    logger.info(f"Checkpoint async completed. Creating sampling client from {path_dict['sampler_path']}")
    try:
        sampling_client = training_client.create_sampling_client(path_dict["sampler_path"])
    except Exception as e:
        logger.exception(f"create_sampling_client failed: {e}")
        raise
    logger.info("Sampling client created successfully")
    return sampling_client


async def _checkpoint_state(
    trainer: "PlatoonTinkerRLTrainer", checkpoint: str, loop_state: dict[str, Any]
) -> dict[str, Any]:
    """``loop_state`` of full checkpoint ``checkpoint``, with the records of what it saves first: the snapshot of
    the run's per-task statistics (``tasks``), and where the run's draws stand, with the groups that settled and no
    committed step consumed (``dispatch``, :func:`~platoon.train.checkpoints.dispatch_state.save_state`).  Both are
    taken at the same point, before anything else runs; the dispatcher's files are written in a thread, since the
    groups they keep can take seconds to encode.  Each record is left out when it was not written: the statistics are
    not kept or stopped, or a file could not be written.  Under a group admission rule, the row also records where the
    order stands (``next_position``), which a resume that cannot read the dispatcher's state goes on from."""

    loop_state = dict(loop_state)
    recorder = getattr(trainer, "_recorder", None)
    if recorder is not None and (record := recorder.snapshot(checkpoint)) is not None:
        loop_state["tasks"] = record
    dispatcher = getattr(trainer, "_dispatcher", None)
    if dispatcher is not None:
        state, kept = dispatcher.state(), dispatcher.kept()
        if getattr(trainer, "_admissions", None) is not None:  # under a rule: where the order stands, in the row too
            loop_state[ORDER_POSITION_KEY] = state["next_position"]
        if (record := await asyncio.to_thread(save_state, trainer.run_log_path, checkpoint, state, kept)) is not None:
            loop_state[DISPATCH_KEY] = record
    return loop_state
