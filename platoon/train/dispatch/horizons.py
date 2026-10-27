"""The dispatcher's horizons when the run's trained policies step independently (RFC 0021, section 2.11): one per
trained policy that receives shares, private to the dispatcher (:mod:`platoon.train.dispatch.dispatcher`).

Stepping in lockstep, the dispatcher counts one horizon, ``(k + 1) × batch_size`` groups, with counters of its own.
Stepping independently, each trained policy ``p`` that receives shares (its rows of a group,
:mod:`platoon.train.dispatch.shares`) has its own horizon of ``(k_p + 1) × batch_p`` shares, ``batch_p`` its batch and
``k_p`` its staleness bound (none: no horizon of its own).  What each one counts:

- every group running: which policies a rollout samples is known only once it settles, so a running group counts
  against every horizon;
- the policy's shares that settled and that its steps have not taken (``queued``);
- the shares its current step took (``taken``), until its next step starts.

A group starts only when every horizon has room for it.  Each horizon applies the dispatcher's rule of the slots held
for capped labels with its own ``k`` and batch.  The leader's horizon also carries the plan: the run's remaining
leader steps take ``batch_size`` shares each, and no group starts beyond them, as the one horizon's plan does in
lockstep.  Every group gives the leader a share, one without its rows when it has none
(:meth:`~platoon.train.dispatch.shares.Shares.put_nowait`), so the plan counts groups, as in lockstep.

Once a stop request ended some policy's loop (:meth:`Horizons.stopped`), that policy takes no more shares: when its
horizon is full, no group starts again (:meth:`Horizons.held_for_good`), and a step that applied an update and waits
for more shares closes short with those it has (:meth:`~platoon.train.dispatch.shares.Shares.done`), as at the plan's
end, rather than wait for groups that never start.  Until then the dispatcher starts groups as before, within every
horizon, so that no share it gives a stopped policy is older than its bound allows.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from platoon.train.policy.spec import PolicyId

if TYPE_CHECKING:
    from platoon.train.policy.set import PolicySet


@dataclass
class _Horizon:
    """One policy's horizon: its bound (``k``; None: unbounded) and batch, and what it counts besides the running
    groups."""

    bound: int | None
    batch: int
    queued: int = 0
    taken: int = 0

    def full(self, running: int, capped: int, own: int) -> bool:
        """Whether a group may not start: ``running`` groups run, ``capped`` of them of capped labels with draws
        waiting (each holds a slot for the draw that follows it, as far as they fit in ``k`` batches), ``own`` of them
        of the label of the draw that would take one of those slots."""

        if self.bound is None:
            return False
        held = min(capped, max(0, self.bound * self.batch - capped))
        held -= min(own, held)
        return running + self.queued + self.taken + held >= (self.bound + 1) * self.batch


class Horizons:
    """Each trained policy's horizon, by policy, in the run's order, the leader's first, and the leader's remaining
    steps (``steps``), which bound the plan.  ``changed`` hears of each change, so that the dispatcher wakes."""

    def __init__(
        self,
        bounds: Mapping[PolicyId, int | None],
        batches: Mapping[PolicyId, int],
        *,
        leader: PolicyId,
        steps: int,
    ) -> None:
        self._horizons = {policy: _Horizon(bounds[policy], batches[policy]) for policy in bounds}
        self.leader = leader
        self.steps_left = steps
        self.changed: Callable[[], None] = lambda: None
        # The policies whose loop a stop request ended (stopped).
        self._stopped: set[PolicyId] = set()

    @classmethod
    def of_run(cls, config: Any, policies: "PolicySet", *, steps: int) -> "Horizons":
        """The horizons of the run's trained ``policies`` over the leader's remaining ``steps``: each policy's bound its
        own ``max_staleness``, else ``train.max_staleness`` (the leader's always), and its batch its own
        ``batch_size``, else ``train.batch_size`` (the leader's always)."""

        train, leader = config.train, policies.leader
        trained = policies.trainable()
        bounds = {
            p.id: train.max_staleness if p.id == leader or p.spec.max_staleness is None else p.spec.max_staleness
            for p in trained
        }
        batches = {
            p.id: train.batch_size if p.id == leader or p.spec.batch_size is None else p.spec.batch_size
            for p in trained
        }
        return cls(bounds, batches, leader=leader, steps=steps)

    @property
    def policies(self) -> list[PolicyId]:
        return list(self._horizons)

    def queued(self, policy: PolicyId, count: int = 1) -> None:
        """``count`` shares of ``policy`` settled (or a resume put them back), for its steps to take."""

        self._horizons[policy].queued += count
        self.changed()

    def taken(self, policy: PolicyId) -> None:
        """A step of ``policy`` took one of its shares: it counts toward that step until the next one starts."""

        horizon = self._horizons[policy]
        horizon.queued = max(0, horizon.queued - 1)
        horizon.taken += 1
        self.changed()

    def step_started(self, policy: PolicyId) -> None:
        """The next step of ``policy`` started: the shares the last one took no longer count; the leader's counts one
        step of the plan."""

        self._horizons[policy].taken = 0
        if policy == self.leader:
            self.steps_left = max(0, self.steps_left - 1)
        self.changed()

    def stopped(self, policy: PolicyId) -> None:
        """A stop request ended ``policy``'s loop: it takes no more shares (:meth:`held_for_good`)."""

        self._stopped.add(policy)
        self.changed()

    def held_for_good(self) -> bool:
        """Whether, with no group running, the horizon of a policy whose loop a stop request ended has no room for
        another group: it never frees, so no group starts again."""

        return any(self._horizons[policy].full(0, 0, 0) for policy in self._stopped)

    def plan_has_room(self, running: int) -> bool:
        """Whether the leader's remaining steps take more shares than ``running`` groups and its shares queued and
        taken may give it (every group gives it one): else no group starts."""

        leader = self._horizons[self.leader]
        return running + leader.queued + leader.taken < self.steps_left * leader.batch

    def binding(self, running: int, capped: int = 0, own: int = 0) -> PolicyId | None:
        """The first policy whose horizon has no room for another group (:meth:`_Horizon.full`), in the run's order;
        None when every one has room."""

        return next((policy for policy, horizon in self._horizons.items() if horizon.full(running, capped, own)), None)

    def sizes(self) -> dict[PolicyId, int | None]:
        """Each policy's horizon in shares, ``(k_p + 1) × batch_p`` (None: unbounded), which loading logs."""

        return {
            policy: None if horizon.bound is None else (horizon.bound + 1) * horizon.batch
            for policy, horizon in self._horizons.items()
        }
