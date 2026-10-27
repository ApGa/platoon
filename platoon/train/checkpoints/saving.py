"""Saving: the cadence after a step (a permanent or a recovery checkpoint, else the weights only), the one full
checkpoint of the run that every save of a start writes (:func:`save_run_checkpoint`: at launch, by the cadence, when
training is interrupted and at its end), which is its ``loop_state`` and then the policy's own save
(:func:`~platoon.train.policy.checkpoints.save_full_checkpoint`), and a full checkpoint's ``loop_state``, with the
per-task statistics' snapshot and the dispatcher's state.

``_save_checkpoint_and_get_sampling_client`` and ``_checkpoint_state`` take the trainer first and are its methods still:
``PlatoonTinkerRLTrainer`` keeps each under its name, so to replace one, replace the method, on the class or an
instance, not the function.  The other functions here call the trainer's methods through the trainer.
"""

import asyncio
import logging
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any

import tinker

from platoon.train.checkpoints.dispatch_state import save_state
from platoon.train.checkpoints.pruning import RECOVERY_CHECKPOINT_PREFIX
from platoon.train.policy.checkpoints import Waiter, sampler_from, save_full_checkpoint, save_weights_only
from platoon.train.task_stats import DISPATCH_KEY
from platoon.train.trainer.signals import Watchdog

if TYPE_CHECKING:
    from platoon.train.trainer.rl import PlatoonTinkerRLTrainer

logger = logging.getLogger(__name__)


# The key of a full checkpoint row's loop_state that, under a group admission rule, records where the run's order
# stood (the dispatcher's next position), so that a resume that cannot read the dispatcher's state goes on from there:
# a step under a rule draws up to its cap's worth of positions, so the checkpoint's batch no longer says where.
ORDER_POSITION_KEY = "next_position"


async def save_run_checkpoint(
    trainer: "PlatoonTinkerRLTrainer",
    training_client: tinker.TrainingClient,
    name: str,
    *,
    batch: int,
    policy_version: int,
    optimizer_steps: int,
    wait: Waiter,
) -> dict[str, str]:
    """Save full checkpoint ``name`` of the trainer's run (:func:`save_full_checkpoint`, into its ``run_log_path``),
    whose ``loop_state`` holds ``batch``, the step a resume from it starts at; the W&B run, so that a resume goes on
    with it (tinker_cookbook's own save leaves it out); ``policy_version`` and ``optimizer_steps``; in that order, then
    what ``_checkpoint_state`` adds, which runs before the save starts.  The paths the save wrote."""

    log_path = trainer.run_log_path
    loop_state = await trainer._checkpoint_state(
        name,
        {
            "batch": batch,
            "wandb_run_id": trainer._stats_logger.wandb_run_id,
            "policy_version": policy_version,
            "optimizer_steps": optimizer_steps,
        },
    )
    return await save_full_checkpoint(training_client, name, log_path=log_path, loop_state=loop_state, wait=wait)


def _feeding(trainer: "PlatoonTinkerRLTrainer", watchdog: Watchdog | None) -> Waiter:
    """The waiter of a save after a step: the trainer's heartbeat, which feeds ``watchdog`` while the save runs."""

    async def wait(awaitable: Awaitable[Any], *, name: str, total_timeout: float | None = None) -> Any:
        return await trainer._await_with_heartbeat(awaitable, name=name, watchdog=watchdog, total_timeout=total_timeout)

    return wait


def checkpoint_kind_for_step(steps_completed: int, save_every: int | None, recovery_every: int | None) -> str | None:
    """The full checkpoint that the cadence saves after a start's ``steps_completed``-th step (counted from 1), in
    precedence order:

    1. ``steps_completed % save_every == 0`` - ``"permanent"``, a full checkpoint
       named ``<batch>``, kept forever. ``save_every`` is ``None`` for
       ``checkpoint.every: 0``, which saves no permanent checkpoints.
    2. ``recovery_every`` set and ``steps_completed % recovery_every == 0``
       - ``"recovery"``, an AReaL-style recovery full checkpoint named
       ``recovery-<batch>``. Same weights+optimizer payload, but
       ``checkpoint.keep_recovery`` deletes all but the newest few once the
       save is recorded (:mod:`platoon.train.checkpoints.pruning`),
       so the recovery cadence can run every step without growing the
       checkpoint store.
    3. otherwise - None: a weights-only update (no disk write in the gateway's
       full mode), which cannot be resumed from.
    """

    if save_every is not None and steps_completed % save_every == 0:
        return "permanent"
    if recovery_every is not None and steps_completed % recovery_every == 0:
        return "recovery"
    return None


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
    """Save after step ``i_batch`` as the cadence says (:func:`checkpoint_kind_for_step`; ``steps_completed`` counts
    steps since ``start_batch``), and return the new sampling client: a full checkpoint's
    (:func:`save_run_checkpoint`, then :func:`sampler_from`), whose name and batch are ``i_batch + 1`` because the
    weights are the ones *after* training on ``i_batch``, so a resume starts from ``i_batch + 1``, and whose
    ``policy_version`` and ``optimizer_steps`` are those given, else that batch; or a weights-only save's
    (:func:`save_weights_only`).  A recovery checkpoint goes to the pruner once ``checkpoints.jsonl`` records it."""

    # Count steps completed since start_batch (1-indexed for human intuition)
    steps_completed = i_batch - start_batch + 1
    next_batch = i_batch + 1
    checkpoint_kind = checkpoint_kind_for_step(steps_completed, save_every, recovery_every)
    if checkpoint_kind is None:
        logger.info(
            f"Saving weights only after completing batch {i_batch} "
            f"(steps_completed={steps_completed}, save_every={save_every}, "
            f"recovery_every={recovery_every})"
        )
        return await save_weights_only(training_client)

    # Save a full (weights + optimizer) checkpoint with loop state.
    checkpoint_name = (
        f"{next_batch:06d}" if checkpoint_kind == "permanent" else f"{RECOVERY_CHECKPOINT_PREFIX}{next_batch:06d}"
    )
    logger.info(
        f"Saving {checkpoint_kind} checkpoint {checkpoint_name} after completing batch {i_batch} "
        f"(steps_completed={steps_completed}, save_every={save_every}, "
        f"recovery_every={recovery_every})"
    )
    policy_version = next_batch if policy_version is None else policy_version
    optimizer_steps = next_batch if optimizer_steps is None else optimizer_steps
    try:
        # A full-parameter save_state takes minutes; keep the watchdog fed.
        path_dict = await save_run_checkpoint(
            trainer,
            training_client,
            checkpoint_name,
            batch=next_batch,
            policy_version=policy_version,
            optimizer_steps=optimizer_steps,
            wait=_feeding(trainer, watchdog),
        )
    except Exception as e:
        logger.exception(f"save_checkpoint_async failed: {e}")
        raise
    if checkpoint_kind == "recovery":
        trainer._recovery_saved(path_dict, policy_version)
    return sampler_from(training_client, path_dict)


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
