"""Each trained policy's own clock, when a run's trained policies step independently (RFC 0021, section 2.11;
:attr:`~platoon.train.trainer.run.TrainerRun.independent`): :class:`OwnClocks`.

The leader's clock is the run's (``run.clock``), which counts the plan and the evaluation cadence; each other trained
policy's begins this start at its own step, the one the row it resumes from records.  Each policy saves a full
checkpoint, a commit, on its own cadence, counted in its own steps (:meth:`OwnClocks.checkpoint_after`): its
``policies.<id>.checkpoint``, else the run's ``checkpoint.every`` (in steps) and ``checkpoint.recovery_every``.  A
policy other than the leader names its commits with its id (``recovery-<id>-<step>``, ``<id>-<step>``): the rows that
the start's launch and exit barriers write into its file carry the leader's step instead, and one Tinker session saves
both, so a name of its own step alone could save over a barrier's row.  The exit save records each other policy at its
own step (:meth:`OwnClocks.exit_counters`).
"""

import asyncio
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

from platoon.train.checkpoints.barrier import Counters
from platoon.train.checkpoints.pruning import RECOVERY_CHECKPOINT_PREFIX
from platoon.train.checkpoints.saving import checkpoint_kind_for_step
from platoon.train.config_defs import event_interval
from platoon.train.policy.spec import PolicyId

if TYPE_CHECKING:
    from platoon.train.checkpoints.resume import StartPoint
    from platoon.train.policy.set import PolicySet
    from platoon.train.trainer.run import StepClock


class OwnClocks(Mapping[PolicyId, "StepClock"]):
    """Each trained policy's clock, by policy, the leader's first: ``clock``, the run's, and a clock of each other
    policy at the step ``start`` gives it, with the run's watchdog and an event of its own, each naming its policy.
    ``config`` and ``steps_per_epoch`` (the plan's) give each policy's checkpoint cadence."""

    def __init__(
        self,
        clock: "StepClock",
        *,
        config: Any,
        policies: "PolicySet",
        start: "StartPoint",
        steps_per_epoch: int | None,
    ) -> None:
        from platoon.train.trainer.run import StepClock  # the run imports this module

        self.config, self.policies, self.leader = config, policies, policies.leader
        self._steps_per_epoch = steps_per_epoch or 1
        # The step each policy began this start at, which its cadence counts from: its own, else the run's.
        self.began: dict[PolicyId, int] = {}
        for policy in policies.trainable():
            own = start.policies[policy.id].step
            self.began[policy.id] = start.step if own is None else own
        clock.policy = self.leader
        self._clocks = {self.leader: clock}
        for policy, step in self.began.items():
            if policy != self.leader:
                self._clocks[policy] = StepClock(
                    step=step, updated=asyncio.Event(), watchdog=clock.watchdog, policy=policy
                )

    def __getitem__(self, policy: PolicyId) -> "StepClock":
        return self._clocks[policy]

    def __iter__(self) -> Iterator[PolicyId]:
        return iter(self._clocks)

    def __len__(self) -> int:
        return len(self._clocks)

    def checkpoint_after(self, policy: PolicyId) -> tuple[str | None, str]:
        """The full checkpoint ``policy``'s own cadence saves after its clock's step (``permanent``, ``recovery``, or
        None: a weights-only save), and its name, both counted in its own steps from the step it began this start at:
        ``policies.<id>.checkpoint``'s ``every`` (0: never) and ``recovery_every``, each left out the run's
        (``checkpoint.every``, in steps, and ``recovery_every``); the leader's the run's.  The leader's name is the
        step's, ``<step + 1>`` or ``recovery-<step + 1>``; another policy's carries its id, ``<id>-<step + 1>`` or
        ``recovery-<id>-<step + 1>``."""

        checkpoint = self.config.checkpoint
        every, recovery = event_interval(checkpoint, self._steps_per_epoch), checkpoint.recovery_every
        if policy != self.leader and (own := self.policies[policy].spec.checkpoint) is not None:
            if own.every is not None:
                every = own.every or None  # 0: no permanent checkpoint
            if own.recovery_every is not None:
                recovery = own.recovery_every
        step = self._clocks[policy].step
        kind = checkpoint_kind_for_step(step - self.began[policy] + 1, every, recovery)
        name = f"{step + 1:06d}" if policy == self.leader else f"{policy}-{step + 1:06d}"
        return kind, f"{RECOVERY_CHECKPOINT_PREFIX}{name}" if kind == "recovery" else name

    def exit_counters(self) -> dict[PolicyId, Counters]:
        """What each other trained policy's row of the exit save records: its version, its optimizer steps and its own
        step."""

        return {
            policy: Counters(self.policies[policy].version, self.policies[policy].optimizer_steps, clock.step)
            for policy, clock in self._clocks.items()
            if policy != self.leader
        }

    def others_stepped(self) -> bool:
        """Whether a policy other than the leader stepped in this start, so that a start that the leader did not step
        in, interrupted or resumed with the leader at the plan's end, still saves what they trained."""

        return any(clock.step > self.began[policy] for policy, clock in self._clocks.items() if policy != self.leader)
