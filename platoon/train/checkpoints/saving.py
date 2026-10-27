"""Saving: the cadence after a step (a permanent or a recovery checkpoint, else the weights only), and what a full
checkpoint of the rl trainer's run holds besides the run's counters (``_checkpoint_state``: the per-task statistics'
snapshot and the dispatcher's state).  The save itself is the run's (:class:`~platoon.train.checkpoints.participants.
RunCheckpoints`), which the trainer's draws take part in, and the policy's
(:meth:`~platoon.train.policy.policy.Policy.save`).

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
from platoon.train.policy.checkpoints import Waiter, sampler_from
from platoon.train.task_stats import DISPATCH_KEY
from platoon.train.trainer.signals import Watchdog

if TYPE_CHECKING:
    from platoon.train.checkpoints.resume import StartPoint
    from platoon.train.trainer.rl import PlatoonTinkerRLTrainer

logger = logging.getLogger(__name__)


# The key of a full checkpoint row's loop_state that, under a group admission rule, records where the run's order
# stood (the dispatcher's next position), so that a resume that cannot read the dispatcher's state goes on from there:
# a step under a rule draws up to its cap's worth of positions, so the checkpoint's batch no longer says where.
ORDER_POSITION_KEY = "next_position"


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
    steps since ``start_batch``), and return the new sampling client: a full checkpoint's (the run's save, then
    :func:`sampler_from`), named and at batch ``i_batch + 1``, since its weights are the ones *after* training on
    ``i_batch``, with the ``policy_version`` and ``optimizer_steps`` given, else that batch; or a weights-only save's.
    The leader of the trainer's checkpoints (``_checkpoints``), which trains with ``training_client``, saves.  A
    recovery checkpoint goes to the pruner once ``checkpoints.jsonl`` records it."""

    # Count steps completed since start_batch (1-indexed for human intuition)
    steps_completed = i_batch - start_batch + 1
    next_batch = i_batch + 1
    checkpoints = trainer._checkpoints(training_client)
    policy = checkpoints.policies[checkpoints.policies.leader]
    checkpoint_kind = checkpoint_kind_for_step(steps_completed, save_every, recovery_every)
    if checkpoint_kind is None:
        logger.info(
            f"Saving weights only after completing batch {i_batch} "
            f"(steps_completed={steps_completed}, save_every={save_every}, "
            f"recovery_every={recovery_every})"
        )
        return await policy.save_for_sampler()

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
        saved = await checkpoints.save(
            checkpoint_name,
            step=next_batch,
            policy_version=policy_version,
            optimizer_steps=optimizer_steps,
            wait=_feeding(trainer, watchdog),
        )
    except Exception as e:
        logger.exception(f"save_checkpoint_async failed: {e}")
        raise
    path_dict = saved[policy.id]
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
    order stands (``next_position``), which a resume that cannot read the dispatcher's state goes on from.  The run's
    checkpoints call it with an empty ``loop_state``: they hold the counters themselves
    (:class:`~platoon.train.checkpoints.participants.RunCheckpoints`)."""

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


class Draws:
    """The rl trainer's draws as a participant of the run's full checkpoints: the per-task statistics' snapshot
    (``tasks``), where the order stands under a group admission rule (``next_position``) and the dispatcher's state
    (``dispatch``), which ``_checkpoint_state`` records; and, as a start resumes, the draws the checkpoint left
    outstanding, which the dispatcher takes up (``_restore_dispatch``)."""

    keys: tuple[str, ...] = ("tasks", ORDER_POSITION_KEY, DISPATCH_KEY)

    def __init__(self, trainer: "PlatoonTinkerRLTrainer") -> None:
        self.trainer = trainer

    async def checkpoint_state(self, name: str) -> dict[str, Any]:
        return await self.trainer._checkpoint_state(name, {})

    def resume(self, start: "StartPoint") -> None:
        dispatcher = self.trainer._dispatcher
        assert dispatcher is not None  # the trainer's prepare() starts it before its draws take part
        if start.record is not None:
            self.trainer._restore_dispatch(dispatcher, start.record, start.policy_version)
