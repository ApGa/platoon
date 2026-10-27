"""Each trained policy's lane through a step (:class:`PolicyLane`), and the lanes of a start's steps
(:func:`policy_lanes`).

A run that trains several policies in lockstep trains them all in each step, on the same groups.  The step splits each
group's rows by the policy they train (:func:`~platoon.train.step.datums.policy_rows`), and each policy's rows go
their own way: its stale filter judges them against its own version and bound, the batch transforms and its objective
prepare them, its learner sends them to its own Tinker service and steps its own optimizer, and the policy publishes
its own sampler.  A lane holds what that way needs, and where its metrics go: under ``policies/<id>/`` in front of
each name (:class:`~platoon.train.policy.learner.PrefixedTracker`), except the leader's (``main``), whose metrics and
counts keep their names.

A step hook that updates a policy at the step's end does so through that policy's lane: its learner, under the hook's
objective (the step's end hands it out), whose metrics go under the lane's prefix, and the lane counts the hook's
updates beside the step's own; a policy that either updated publishes once.  A run of one trained policy has one lane,
the leader's, whose counts are the step's own: it steps, logs and publishes call for call as before.  The step itself,
``TrainStep`` in :mod:`platoon.train.trainer.loops`, takes the lanes this module builds once per start, and walks each
group through them.
"""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import tinker

from platoon.train.policy.learner import Learner, OpenedSteps, PrefixedTracker, UpdateTimeouts
from platoon.train.policy.spec import PolicyId, PolicySpec
from platoon.train.step.microbatch import MicrobatchFill

if TYPE_CHECKING:
    from platoon.train.objectives.api import Objective
    from platoon.train.policy.checkpoints import Waiter
    from platoon.train.step.collect import StepTally
    from platoon.train.step.staleness import StaleFilter


@dataclass
class LaneTally:
    """What a step counts of a lane other than the leader's, whose counts are the step's own
    (:class:`~platoon.train.step.collect.StepTally`): the optimizer steps its policy took, and its microbatches that
    sent nothing because they had no rows or a zero objective."""

    optimizer_steps: int = 0
    zero_objective_microbatches: int = 0
    empty_microbatches: int = 0


@dataclass(frozen=True)
class PolicyLane:
    """One trained policy's lane through a step: its learner (the policy, its objective, its training client), its
    stale filter, its version as the step reads it, and where its metrics go: ``tracker``, the step's own for the
    leader, else a :class:`PrefixedTracker` that puts ``prefix`` (``policies/<id>/``) in front of each name.
    ``named``: whether its log lines name its policy, as every lane's do in a run of several trained policies, where its
    learner is named too (an error of its service names the policy and the service,
    :func:`~platoon.train.policy.spec.failure_note`)."""

    learner: Learner
    stale_filter: "StaleFilter"
    version: Callable[[], int]
    tracker: Any
    prefix: str = ""
    named: bool = False

    @property
    def policy(self) -> Any:
        """The policy the lane trains (its learner's)."""

        return self.learner.policy

    @property
    def policy_id(self) -> PolicyId:
        """The id of the policy the lane trains."""

        return self.policy.spec.id

    @property
    def leads(self) -> bool:
        """Whether it is the leader's lane, whose metrics keep their names and whose counts are the step's own."""

        return not self.prefix

    @property
    def policy_suffix(self) -> str:
        """What the step's log lines add to say whose rows they mean: `` of policy <id>``, nothing in a run of one."""

        return f" of policy {self.policy_id}" if self.named else ""

    def rows_in(self, fill: MicrobatchFill) -> list[tinker.Datum]:
        """The lane's datums of a microbatch: the leader's are its ``datums``, each other policy's its own list."""

        return fill.datums if self.leads else fill.others.get(self.policy_id, [])

    def add_rows(self, fill: MicrobatchFill, datums: Sequence[tinker.Datum]) -> None:
        """Add ``datums`` to the lane's datums of a microbatch (:meth:`rows_in`)."""

        (fill.datums if self.leads else fill.others.setdefault(self.policy_id, [])).extend(datums)

    def record_counts(self, tally: "LaneTally | StepTally", *, hook_updates: int | None = None) -> None:
        """What the step did with a lane other than the leader's (whose counts are among the step's), under its
        prefix: the step's own optimizer updates of its policy, whether it took none, and, in a run with step hooks,
        the hooks' updates of it (``hook_updates``; a policy that either updated publishes once); its version and its
        optimizer steps over the run, the hooks' included."""

        counts = {
            "batch/optimizer_updates": float(tally.optimizer_steps),
            "batch/skipped_optimizer": float(tally.optimizer_steps == 0),
            "policy/version": float(self.version()),
            "optim/updates_total": float(self.policy.optimizer_steps),
        }
        if hook_updates is not None:
            counts["batch/hook_optimizer_updates"] = float(hook_updates)
        self.tracker.scalar(**counts)


def policy_lanes(
    leader: Any,
    others: Sequence[Any],
    *,
    objective: "Callable[[PolicySpec], Objective]",
    timeouts: UpdateTimeouts,
    wait: "Waiter",
    tracker: Any,
    stale_filter: "StaleFilter",
    leader_version: Callable[[], int],
    fail_closed: bool = False,
    before_update: Callable[[], Awaitable[None]] | None = None,
    clock: "OpenedSteps | None" = None,
) -> tuple[PolicyLane, ...]:
    """Each trained policy's lane through a start's steps: the leader's (``leader``, the policy the steps train first)
    first, then each of ``others`` in the run's order.  Each lane's learner updates its policy once per minibatch, under
    the ``objective`` its spec names, each result awaited with ``wait`` under its deadline in ``timeouts``; every one
    fails closed when ``fail_closed`` says so, awaits the same barrier before each optimizer step (``before_update``)
    and opens the step and its policy's update on the same ``clock`` (:class:`~platoon.train.policy.learner.Learner`).
    The leader's lane judges rows with the run's ``stale_filter`` at ``leader_version`` and records into the step's
    ``tracker``; each other one with its own bound, if it has one, at its own version, under ``policies/<id>/``.  Every
    lane, and its learner, names its policy when there are ``others``: the step fails, in lockstep, as it does for one
    policy, when a service fails, and says whose did (a step hook's update through the lane's learner too)."""

    def learner(policy: Any, records: Any) -> Learner:
        return Learner(
            policy,
            objective=objective(policy.spec),
            timeouts=timeouts,
            wait=wait,
            tracker=records,
            fail_closed=fail_closed,
            before_update=before_update,
            clock=clock,
            named=bool(others),
        )

    lanes = [PolicyLane(learner(leader, tracker), stale_filter, leader_version, tracker, named=bool(others))]
    for policy in others:
        prefixed = PrefixedTracker(tracker, f"policies/{policy.id}/")
        lanes.append(
            PolicyLane(
                learner(policy, prefixed),
                stale_filter.for_policy(policy.spec.max_staleness),
                _version_of(policy),
                prefixed,
                prefixed.prefix,
                named=True,
            )
        )
    return tuple(lanes)


def _version_of(policy: Any) -> Callable[[], int]:
    """How a lane other than the leader's reads its policy's version: from the policy, as each publication counts it."""

    return lambda: policy.version
