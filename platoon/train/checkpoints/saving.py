"""Saving: the cadence after a step (a permanent or a recovery checkpoint, else the weights only, of each policy the
step updated: :func:`save_after_step`), and what a full checkpoint holds of the run's draws besides the run's counters
(``_checkpoint_state``: the per-task statistics' snapshot and the dispatcher's state).  The save itself is the run's
(:class:`~platoon.train.checkpoints.participants.RunCheckpoints`, a barrier of every trained policy's), which the draws
take part in, and each policy's (:meth:`~platoon.train.policy.policy.Policy.save`).

The step's save takes the trainer whose steps save (what it calls of it is :class:`SavingTrainer`); the checkpoint
state takes the start's rollout source (:class:`~platoon.train.dispatch.source.RolloutSource`), which calls it as a
participant of the run's checkpoints.
"""

import asyncio
import logging
from collections.abc import Awaitable, Collection, Coroutine, Mapping
from typing import TYPE_CHECKING, Any, Protocol

import tinker

from platoon.train.checkpoints.barrier import Counters
from platoon.train.checkpoints.dispatch_state import save_state
from platoon.train.checkpoints.pruning import RECOVERY_CHECKPOINT_PREFIX
from platoon.train.credit.runtime import CREDIT_KEY, credit_states
from platoon.train.policy.checkpoints import Waiter, sampler_from
from platoon.train.policy.spec import PolicyId
from platoon.train.task_stats import DISPATCH_KEY
from platoon.train.trainer.signals import Watchdog

if TYPE_CHECKING:
    from platoon.train.checkpoints.participants import RunCheckpoints
    from platoon.train.dispatch.source import RolloutSource

logger = logging.getLogger(__name__)


# The key of a full checkpoint row's loop_state that, under a group admission rule, records where the run's order
# stood (the dispatcher's next position), so that a resume that cannot read the dispatcher's state goes on from there:
# a step under a rule draws up to its cap's worth of positions, so the checkpoint's batch no longer says where.
ORDER_POSITION_KEY = "next_position"


class SavingTrainer(Protocol):
    """What a step's save calls on the trainer whose steps save: the checkpoints its steps save through, the hand-off of
    a recovery save to its pruner, and its heartbeat."""

    def _checkpoints(self) -> "RunCheckpoints": ...
    def _recovery_saved(
        self, paths: dict[str, str], policy_version: int, policy: PolicyId | None = None, *, published: bool = True
    ) -> None: ...
    async def _await_with_heartbeat(
        self,
        coro: Awaitable[Any],
        name: str,
        heartbeat_interval: float = 60.0,
        watchdog: Watchdog | None = None,
        total_timeout: float | None = None,
    ) -> Any: ...


def _feeding(trainer: SavingTrainer, watchdog: Watchdog | None) -> Waiter:
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
    trainer: SavingTrainer,
    training_client: tinker.TrainingClient,
    i_batch: int,
    save_every: int | None,
    start_batch: int = 0,
    watchdog: Watchdog | None = None,
    recovery_every: int | None = None,
    policy_version: int | None = None,
    optimizer_steps: int | None = None,
) -> tinker.SamplingClient:
    """Save after step ``i_batch`` of a run of one trained policy, the leader of the trainer's checkpoints
    (``_checkpoints``), which trains with ``training_client``, and return its new sampling client
    (:func:`save_after_step`): its full checkpoint, when one is due, records the ``policy_version`` and
    ``optimizer_steps`` given, else ``i_batch + 1``, since its weights are the ones *after* training on ``i_batch``.
    The step's line before it names this function (:func:`~platoon.train.step.publish.publish_policies`)."""

    checkpoints = trainer._checkpoints()
    leader, next_batch = checkpoints.policies.leader, i_batch + 1
    counters = Counters(
        next_batch if policy_version is None else policy_version,
        next_batch if optimizer_steps is None else optimizer_steps,
    )
    sampling_clients = await save_after_step(
        trainer,
        checkpoints,
        i_batch,
        updated={leader: counters},
        save_every=save_every,
        start_batch=start_batch,
        watchdog=watchdog,
        recovery_every=recovery_every,
    )
    return sampling_clients[leader]


async def save_after_step(
    trainer: SavingTrainer,
    checkpoints: "RunCheckpoints",
    i_batch: int,
    *,
    updated: Mapping[PolicyId, Counters],
    save_every: int | None,
    start_batch: int = 0,
    watchdog: Watchdog | None = None,
    recovery_every: int | None = None,
) -> dict[PolicyId, tinker.SamplingClient]:
    """Save after step ``i_batch`` as the cadence says (:func:`checkpoint_kind_for_step`; ``steps_completed`` counts
    steps since ``start_batch``), and return the new sampling client of each policy the step updated (``updated``,
    with the counters its row of a checkpoint records): a full checkpoint's, named and at batch ``i_batch + 1``, of
    every trained policy (:meth:`RunCheckpoints.save`, whose policies the step did not update record their counters
    now), then :func:`sampler_from`; or a weights-only save's, of each updated policy.  Each recovery checkpoint goes to
    its policy's pruner once its ``checkpoints.jsonl`` records it."""

    # Count steps completed since start_batch (1-indexed for human intuition)
    steps_completed = i_batch - start_batch + 1
    next_batch = i_batch + 1
    policies = checkpoints.policies
    checkpoint_kind = checkpoint_kind_for_step(steps_completed, save_every, recovery_every)
    if checkpoint_kind is None:
        logger.info(
            f"Saving weights only after completing batch {i_batch} "
            f"(steps_completed={steps_completed}, save_every={save_every}, "
            f"recovery_every={recovery_every})"
        )
        return await policies.each(
            {policy: policies[policy].save_for_sampler() for policy in updated}, doing="saving its weights"
        )

    # Save a full (weights + optimizer) checkpoint with loop state.
    checkpoint_name = (
        f"{next_batch:06d}" if checkpoint_kind == "permanent" else f"{RECOVERY_CHECKPOINT_PREFIX}{next_batch:06d}"
    )
    logger.info(
        f"Saving {checkpoint_kind} checkpoint {checkpoint_name} after completing batch {i_batch} "
        f"(steps_completed={steps_completed}, save_every={save_every}, "
        f"recovery_every={recovery_every})"
    )
    saved = await _save_full(trainer, checkpoints, checkpoint_name, next_batch, updated=updated, watchdog=watchdog)
    if checkpoint_kind == "recovery":
        versions = {policy: _counted(policies[policy], updated).version for policy in saved}
        _hand_to_pruners(trainer, saved, versions, leader=policies.leader, published=updated)
    return {policy: sampler_from(policies[policy].training_client, saved[policy]) for policy in updated}


async def _save_full(
    trainer: SavingTrainer,
    checkpoints: "RunCheckpoints",
    name: str,
    step: int,
    *,
    updated: Mapping[PolicyId, Counters],
    watchdog: Watchdog | None,
) -> dict[PolicyId, dict[str, str]]:
    """Full checkpoint ``name`` of every trained policy at batch ``step`` (:meth:`RunCheckpoints.save`): each row
    records its policy's counters, those in ``updated`` for a policy the step updated, else its own now.  The paths
    each policy's save wrote; a save that fails is logged and raised."""

    policies = checkpoints.policies
    counted = _counted(policies[policies.leader], updated)
    others = {policy: counters for policy, counters in updated.items() if policy != policies.leader}
    try:
        # A full-parameter save_state takes minutes; keep the watchdog fed.
        return await checkpoints.save(
            name,
            step=step,
            policy_version=counted.version,
            optimizer_steps=counted.optimizer_steps,
            wait=_feeding(trainer, watchdog),
            **({"others": others} if others else {}),
        )
    except Exception as e:
        logger.exception(f"save_checkpoint_async failed: {e}")
        raise


def _counted(policy: Any, updated: Mapping[PolicyId, Counters]) -> Counters:
    """What ``policy``'s row of a checkpoint records: its counters in ``updated`` when the step updated it, else its
    own now."""

    return updated[policy.id] if policy.id in updated else Counters(policy.version, policy.optimizer_steps)


def _hand_to_pruners(
    trainer: SavingTrainer,
    saved: Mapping[PolicyId, dict[str, str]],
    versions: Mapping[PolicyId, int],
    *,
    leader: str,
    published: Collection[PolicyId],
) -> None:
    """Hand each policy's recovery save (``saved``) to its pruner, at the version its row records (``versions``): the
    leader's first, as a run of one policy always did, then each other policy's.  The save of a policy the step did not
    update (not in ``published``) holds weights saved before, whose sampler no rollout samples: its pruner is told so,
    and keeps it no longer than ``checkpoint.keep_recovery`` asks."""

    # The leader's published save makes the call a run of one policy always made, argument for argument: a subclass's
    # _recovery_saved, a seam of the rl trainer, may take those two only.
    if leader in published:
        trainer._recovery_saved(saved[leader], versions[leader])
    else:
        trainer._recovery_saved(saved[leader], versions[leader], published=False)
    for policy, paths in saved.items():
        if policy != leader:
            trainer._recovery_saved(paths, versions[policy], policy=policy, published=policy in published)


async def _checkpoint_state(source: "RolloutSource", checkpoint: str, loop_state: dict[str, Any]) -> dict[str, Any]:
    """``loop_state`` of full checkpoint ``checkpoint``, with the records of what it saves first: the snapshot of
    the run's per-task statistics (``tasks``), and where the source's draws stand, with the groups that settled and no
    committed step consumed (``dispatch``, :func:`~platoon.train.checkpoints.dispatch_state.save_state`).  Both are
    taken at the same point, before anything else runs; the dispatcher's files are written in a thread, since the
    groups they keep can take seconds to encode.  Each record is left out when it was not written: the statistics are
    not kept or stopped, or a file could not be written.  Under a group admission rule, the row also records where the
    order stands (``next_position``), which a resume that cannot read the dispatcher's state goes on from.  When the
    workflow's credit estimators have state, the row records it too (``credit``), taken with the groups the dispatcher
    keeps (:func:`~platoon.train.credit.runtime.credit_states`), but only beside a dispatcher's record that kept them
    (their file, when there are any): a resume that rolls them out again starts the estimators afresh, rather than
    settle their groups twice.  The run's checkpoints call it with an empty ``loop_state``: they hold the counters
    themselves (:class:`~platoon.train.checkpoints.participants.RunCheckpoints`)."""

    loop_state = dict(loop_state)
    recorder = source.recorder
    if recorder is not None and (record := recorder.snapshot(checkpoint)) is not None:
        loop_state["tasks"] = record
    dispatcher = source.dispatcher
    if dispatcher is not None:
        state, kept, credit = dispatcher.state(), dispatcher.kept(), credit_states(source.workflow)
        if source.admissions is not None:  # under a rule: where the order stands, in the row too
            loop_state[ORDER_POSITION_KEY] = state["next_position"]
        if (record := await asyncio.to_thread(save_state, source.run_dir, checkpoint, state, kept)) is not None:
            loop_state[DISPATCH_KEY] = record
            if credit is not None and (not kept or "kept" in record):  # the groups the state settled are kept
                loop_state[CREDIT_KEY] = credit
    return loop_state


def capture_state(source: "RolloutSource", checkpoint: str) -> Coroutine[Any, Any, dict[str, Any]]:
    """When the run's policies step independently, what commit ``checkpoint`` saves of the source's draws, taken at once
    under the commit lock (the source's ``capture``, :mod:`platoon.train.checkpoints.commits`): the snapshot of the
    per-task statistics, written now; the dispatcher's state, the groups it keeps and what each policy did with its
    share of each (the share ledger's snapshot), and the credit estimators' state, as :func:`_checkpoint_state` takes
    them; and the writing still to do, which writes the dispatcher's files in a thread and gives the entries."""

    loop_state: dict[str, Any] = {}
    recorder, dispatcher, ledger = source.recorder, source.dispatcher, source.ledger
    assert dispatcher is not None and ledger is not None  # a source whose policies step independently is open
    if recorder is not None and (record := recorder.snapshot(checkpoint)) is not None:
        loop_state["tasks"] = record
    state, kept, credit, shares = (
        dispatcher.state(),
        dispatcher.kept(),
        credit_states(source.workflow),
        ledger.snapshot(),
    )

    async def write() -> dict[str, Any]:
        record = await asyncio.to_thread(save_state, source.run_dir, checkpoint, state, kept, shares)
        if record is not None:
            loop_state[DISPATCH_KEY] = record
            if credit is not None and (not kept or "kept" in record):  # the groups the state settled are kept
                loop_state[CREDIT_KEY] = credit
        return loop_state

    return write()
