"""The run's full checkpoints (:class:`RunCheckpoints`), and the parts of the run's state that each of them holds
(:class:`CheckpointParticipant`).

Every full checkpoint of a start (at launch, by the cadence, when training is interrupted, and at its end) is one row
of the run's ``checkpoints.jsonl``, the leader policy's save, whose ``loop_state`` holds the run's counters first
(:data:`COUNTER_KEYS`), then each participant's entries, in the order the participants registered.  A trainer registers
its participants with the run (``run.checkpoints``); the start's draws, which the run opens for a trainer that rolls out
(``run.rollouts``, the rl trainer's), are one: the per-task statistics and the dispatcher's state.  The sft trainer's is
the cursor of its conversations.  In a run of several trained policies a checkpoint is a barrier of their saves, which
the leader's row, written last, commits (:mod:`platoon.train.checkpoints.barrier`).  When they step independently,
every full save of any policy is a commit of its own (:meth:`RunCheckpoints.commit`,
:mod:`platoon.train.checkpoints.commits`), and the start's launch and exit barriers are commits too.

The participants' rules (RFC 0021, section 8): each key is its participant's own (:meth:`RunCheckpoints.register`
refuses one that the counters, the keys kept for several policies, the row itself (:func:`row_fields`) or another
participant hold, and a step hook's that is one of the draws'); a participant registered before the start point is read
takes it up then, before any training client opens, and one registered later as it registers; and a key of the row a
start resumes from that no part of the run claims stops the start (:meth:`RunCheckpoints.unclaimed`), so that nothing
saved is dropped silently.

Imports no Tinker, so that ``platoon.experimental`` re-exports :class:`CheckpointParticipant`.
"""

import dataclasses
import functools
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol

from platoon.train.policy.spec import failure_note

if TYPE_CHECKING:
    from platoon.train.checkpoints.barrier import Counters
    from platoon.train.checkpoints.commits import Commits
    from platoon.train.checkpoints.resume import StartPoint
    from platoon.train.policy.checkpoints import Waiter
    from platoon.train.policy.set import PolicySet
    from platoon.train.policy.spec import PolicyId

# The keys of a row's loop_state that hold the run's own counters, in the row's order: the step a resume from the row
# starts at, the W&B run it goes on with, and the policy's version and optimizer steps.
COUNTER_KEYS = ("batch", "wandb_run_id", "policy_version", "optimizer_steps")
# The key of a leader row's loop_state that names each other trained policy's row of the same checkpoint
# (platoon.train.checkpoints.barrier), after the participants' entries.
POLICIES_KEY = "policies"
# The keys kept for runs of several policies, which no participant may hold: the other trained policies' rows of a
# checkpoint (POLICIES_KEY), and commit_seq.
RESERVED_KEYS = (POLICIES_KEY, "commit_seq")
# The keys of a row that the run claims whoever takes part: its counters, the other trained policies' rows, a commit's
# number, and its draws' (the rollout source's, its credit estimators' state included), the only other keys a platoon
# release writes, so that every run directory a release wrote resumes.
CORE_KEYS = (*COUNTER_KEYS, POLICIES_KEY, "commit_seq", "tasks", "next_position", "dispatch", "credit")


@functools.cache
def row_fields() -> frozenset[str]:
    """The keys that a row of ``checkpoints.jsonl`` holds itself, beside its ``loop_state``: the fields of
    tinker_cookbook's ``CheckpointRecord`` (the checkpoint's ``name``, ``batch``, ``epoch``, ``final``, and the
    ``state_path`` and ``sampler_path`` its save wrote), read from the class so that they follow it, but ``extra``,
    which holds the other entries.  The row is ``{"name": name, **loop_state, **paths}``: an entry of one of these
    names would replace the row's own value (its name, which the pruners read), or be replaced by the save's paths."""

    from tinker_cookbook.checkpoint_utils import CheckpointRecord

    return frozenset(field.name for field in dataclasses.fields(CheckpointRecord) if field.name != "extra")


class CheckpointParticipant(Protocol):
    """Part of the run's state that every full checkpoint saves, and that a start which resumes from one takes up."""

    # Its keys of a row's loop_state.
    keys: tuple[str, ...]

    async def checkpoint_state(self, name: str) -> Mapping[str, Any]:
        """Its ``loop_state`` entries of full checkpoint ``name``, as JSON; it may write files of its own beside the
        row, which its entries name.  An entry it could not write is left out."""
        ...

    def resume(self, start: "StartPoint") -> None:
        """Take up what the checkpoint row that the start resumes from saved: its entries are in ``start.loop_state``
        (empty on a fresh run, when ``start.record`` is None), and the start begins at ``start.step``."""
        ...

    # Optional: ``async def aclose(self) -> None``, which the run calls as the start ends, also after a failure.
    # Optional: ``capture(name) -> Awaitable[Mapping[str, Any]]``, a plain method that takes what checkpoint ``name``
    # saves at once (no await: immutable values, or references to them) and returns the writing still to do, whose
    # result is its entries.  When the run's policies step independently, a commit calls it under the commit lock and
    # awaits the writing after releasing it; a participant without it has its ``checkpoint_state`` awaited under the
    # lock (:mod:`platoon.train.checkpoints.commits`).


class RunCheckpoints:
    """The full checkpoints of one start of a run in ``run_dir``: who takes part in them (:meth:`register`), and the
    save itself (:meth:`save`), the leader policy's, into the run's ``checkpoints.jsonl``.  ``wandb_run_id`` reads
    the W&B run as each save is taken, so that a resume goes on with it."""

    def __init__(
        self,
        run_dir: str,
        policies: "PolicySet",
        *,
        wandb_run_id: Callable[[], str | None],
        joined: Callable[[Any], None] | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.policies = policies
        self._wandb_run_id = wandb_run_id
        self.participants: list[CheckpointParticipant] = []
        # Hears of each participant as it registers: the run, which closes it (aclose).
        self._joined = joined
        # Where the start begins, once it is read (resume); None before.
        self.start: "StartPoint | None" = None
        # The commits of a run whose trained policies step independently (None: they step in lockstep).
        self.commits: "Commits | None" = None
        # The name of the newest full checkpoint this start saved; None before its launch save.
        self.last_saved: str | None = None

    def register(self, participant: CheckpointParticipant, *, hook: bool = False) -> None:
        """``participant`` takes part in every full checkpoint from now on, after those that registered before it;
        registering it again does nothing.  Registered once the start's point is read, it takes that up at once
        (:meth:`CheckpointParticipant.resume`).  Its keys are its own: a key of the run's counters
        (:data:`COUNTER_KEYS`), one kept for several policies (:data:`RESERVED_KEYS`), one the row holds itself
        (:func:`row_fields`), one the run claims for its draws (:data:`CORE_KEYS`) when it is a step hook (``hook``,
        which ``run.hooks.register`` passes), or one another participant holds is refused (``ConfigError``, exit 78),
        naming the participant and the key."""

        if any(other is participant for other in self.participants):
            return
        for key in participant.keys:
            if (refusal := self._refusal(key, participant, hook=hook)) is not None:
                from platoon.utils.config import ConfigError

                raise ConfigError(refusal)
        self.participants.append(participant)
        if self._joined is not None:
            self._joined(participant)
        if self.start is not None:
            participant.resume(self.start)

    def _refusal(self, key: str, participant: CheckpointParticipant, *, hook: bool) -> str | None:
        """Why ``participant`` may not hold ``key`` (:meth:`register`), or None when it may."""

        named = _named(participant)
        if key in COUNTER_KEYS or key in RESERVED_KEYS:
            return f"loop_state key {key!r} is the run's own: {named} may not hold it"
        if key in row_fields():
            return f"loop_state key {key!r} is a field of the checkpoint row itself: {named} may not hold it"
        if hook and key in CORE_KEYS:
            return f"loop_state key {key!r} is the draws' (the rollout source's): step hook {named} may not hold it"
        if (owner := next((other for other in self.participants if key in other.keys), None)) is not None:
            return f"loop_state key {key!r} is {_named(owner)}'s: {named} may not hold it too"
        return None

    def unclaimed(self, start: "StartPoint") -> list[str]:
        """The keys of the row that ``start`` resumes from that neither the run (:data:`CORE_KEYS`) nor a registered
        participant claims, in the row's order."""

        claimed = {*CORE_KEYS, *(key for participant in self.participants for key in participant.keys)}
        return [key for key in start.loop_state if key not in claimed]

    def resume(self, start: "StartPoint") -> None:
        """The start begins at ``start``: each participant registered so far takes it up, in their order; those that
        register later take it up as they register."""

        self.start = start
        for participant in self.participants:
            participant.resume(start)

    async def save(
        self,
        name: str,
        *,
        step: int,
        policy_version: int,
        optimizer_steps: int,
        wait: "Waiter",
        others: "Mapping[PolicyId, Counters] | None" = None,
    ) -> "dict[PolicyId, dict[str, str]]":
        """Save full checkpoint ``name``: its row's ``loop_state`` holds ``batch`` (``step``, the step a resume from it
        starts at), the W&B run (tinker_cookbook's own save leaves it out), the policy's ``policy_version`` and
        ``optimizer_steps`` as a resume takes them up, then each participant's entries, all taken before the save
        starts; then every other trained policy saves, at the same time, each recording its counters (``others``,
        else its own now), and the leader's row names their rows (``policies``,
        :func:`~platoon.train.checkpoints.barrier.save_others`); then the leader policy saves
        (:meth:`~platoon.train.policy.policy.Policy.save`, under ``wait``), and ``checkpoints.jsonl`` records the row
        once its save succeeded: the row that commits the checkpoint.  When the policies step independently, the
        barrier is a commit too: its row's ``commit_seq`` follows the counters, and ``policies`` names every policy's
        row, the leader's own included, with each one's published version and step (each other's in ``others``).  The
        paths the save wrote, by policy."""

        from platoon.train.checkpoints.barrier import Counters, save_others  # Tinker loads with the saves

        loop_state: dict[str, Any] = {
            "batch": step,
            "wandb_run_id": self._wandb_run_id(),
            "policy_version": policy_version,
            "optimizer_steps": optimizer_steps,
        }
        if (commits := self.commits) is not None:
            loop_state["commit_seq"] = commits.take_seq()
        for participant in self.participants:
            if commits is not None and callable(capture := getattr(participant, "capture", None)):
                loop_state.update(await capture(name))  # a commit's entries: the shares of the kept groups too
            else:
                loop_state.update(await participant.checkpoint_state(name))
        saved, named = await save_others(self.policies, self.run_dir, name, step=step, counters=others or {}, wait=wait)
        if named and commits is not None:  # independent stepping: every policy's entry, as a commit names them
            named = commits.barrier_entries(
                name, named, leader=Counters(policy_version, optimizer_steps), step=step, others=others or {}
            )
        if named:  # several trained policies: the leader's row names each other one's
            loop_state[POLICIES_KEY] = named
        leader = self.policies[self.policies.leader]
        try:
            saved[leader.id] = await leader.save(name, log_path=self.run_dir, loop_state=loop_state, wait=wait)
        except Exception as error:
            if named:  # among several policies, the error says whose save failed
                error.add_note(failure_note(leader.spec, f"saving checkpoint {name}"))
            raise
        if named and commits is not None:
            commits.barrier_saved(named, saved[leader.id])
        self.last_saved = name
        return saved

    async def commit(self, policy: "PolicyId", name: str, *, step: int, counters: "Counters", wait: "Waiter") -> Any:
        """Stepping independently, ``policy``'s full checkpoint ``name`` at its own step ``step``, a commit
        (:meth:`~platoon.train.checkpoints.commits.Commits.commit`): the paths its save wrote.  A failure names the
        policy and its service."""

        assert self.commits is not None  # only a run that steps independently commits one policy at a time
        try:
            saved = await self.commits.commit(policy, name, step=step, counters=counters, wait=wait)
        except Exception as error:
            error.add_note(failure_note(self.policies[policy].spec, f"saving checkpoint {name}"))
            raise
        if policy == self.policies.leader:
            self.last_saved = name
        return saved


def _named(participant: Any) -> str:
    """A participant as an error names it: its name when it has one (a step hook's), else its class."""

    return repr(getattr(participant, "name", None) or type(participant).__qualname__)
