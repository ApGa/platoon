"""The run's full checkpoints (:class:`RunCheckpoints`), and the parts of the run's state that each of them holds
(:class:`CheckpointParticipant`).

Every full checkpoint of a start (at launch, by the cadence, when training is interrupted, and at its end) is one row
of the run's ``checkpoints.jsonl``, the leader policy's save, whose ``loop_state`` holds the run's counters first
(:data:`COUNTER_KEYS`), then each participant's entries, in the order the participants registered.  A trainer registers
its participants with the run (``run.checkpoints``); the rl trainer's is its draws: the per-task statistics and the
dispatcher's state.

The participants' rules (RFC 0021, section 8): each key is its participant's own (:meth:`RunCheckpoints.register`
refuses one that the counters, the keys kept for several policies or another participant hold); a participant
registered before the start point is read takes it up then, before any training client opens, and one registered later
as it registers; and a key of the row a start resumes from that no part of the run claims stops the start
(:meth:`RunCheckpoints.unclaimed`), so that nothing saved is dropped silently.

Imports no Tinker, so that ``platoon.experimental`` re-exports :class:`CheckpointParticipant`.
"""

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from platoon.train.checkpoints.resume import StartPoint
    from platoon.train.policy.checkpoints import Waiter
    from platoon.train.policy.set import PolicySet
    from platoon.train.policy.spec import PolicyId

# The keys of a row's loop_state that hold the run's own counters, in the row's order: the step a resume from the row
# starts at, the W&B run it goes on with, and the policy's version and optimizer steps.
COUNTER_KEYS = ("batch", "wandb_run_id", "policy_version", "optimizer_steps")
# The keys kept for runs of several policies, which no participant may hold.
RESERVED_KEYS = ("policies", "commit_seq")
# The keys of a row that the run claims whoever takes part: its counters, and its draws' (the rollout source's), the
# only other keys a platoon release ever wrote, so that every run directory a release wrote resumes.
CORE_KEYS = (*COUNTER_KEYS, "tasks", "next_position", "dispatch")


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
        # The name of the newest full checkpoint this start saved; None before its launch save.
        self.last_saved: str | None = None

    def register(self, participant: CheckpointParticipant) -> None:
        """``participant`` takes part in every full checkpoint from now on, after those that registered before it;
        registering it again does nothing.  Registered once the start's point is read, it takes that up at once
        (:meth:`CheckpointParticipant.resume`).  Its keys are its own: a key of the run's counters
        (:data:`COUNTER_KEYS`), one kept for several policies (:data:`RESERVED_KEYS`) or one another participant holds
        is refused (``ValueError``), naming both owners."""

        if any(other is participant for other in self.participants):
            return
        for key in participant.keys:
            if key in COUNTER_KEYS or key in RESERVED_KEYS:
                raise ValueError(f"loop_state key {key!r} is the run's own: {_named(participant)} may not hold it")
            if (owner := next((other for other in self.participants if key in other.keys), None)) is not None:
                raise ValueError(
                    f"loop_state key {key!r} is {_named(owner)}'s: {_named(participant)} may not hold it too"
                )
        self.participants.append(participant)
        if self._joined is not None:
            self._joined(participant)
        if self.start is not None:
            participant.resume(self.start)

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
        self, name: str, *, step: int, policy_version: int, optimizer_steps: int, wait: "Waiter"
    ) -> "dict[PolicyId, dict[str, str]]":
        """Save full checkpoint ``name``: its row's ``loop_state`` holds ``batch`` (``step``, the step a resume from it
        starts at), the W&B run (tinker_cookbook's own save leaves it out), the policy's ``policy_version`` and
        ``optimizer_steps`` as a resume takes them up, then each participant's entries, all taken before the save
        starts; then the leader policy saves (:meth:`~platoon.train.policy.policy.Policy.save`, under ``wait``), and
        ``checkpoints.jsonl`` records the row once its save succeeded.  The paths the save wrote, by policy."""

        loop_state: dict[str, Any] = {
            "batch": step,
            "wandb_run_id": self._wandb_run_id(),
            "policy_version": policy_version,
            "optimizer_steps": optimizer_steps,
        }
        for participant in self.participants:
            loop_state.update(await participant.checkpoint_state(name))
        leader = self.policies[self.policies.leader]
        saved = {leader.id: await leader.save(name, log_path=self.run_dir, loop_state=loop_state, wait=wait)}
        self.last_saved = name
        return saved


def _named(participant: Any) -> str:
    """A participant as an error names it: its name when it has one (a step hook's), else its class."""

    return repr(getattr(participant, "name", None) or type(participant).__qualname__)
