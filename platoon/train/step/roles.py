"""A step's metrics by agent role (``roles/<role>/*``), in a run with an entry whose estimator credits each role of its
trees apart (``credit: roles``, :func:`~platoon.train.credit.runtime.steps_log_by_role`): for each trained policy's
lane, each role's share of the advantage mass the step sends, over the rows of every entry (``roles/<role>/mass_share``:
an environment generator's share of the step, say); how far its importance ratios ``r``, a token's probability as the
step trained it over its probability as it was sampled, strayed from 1: their mean ``|log r|`` over its action tokens
(``roles/<role>/log_ratio/abs_avg``) and the share of those whose ``r`` is outside ``[0.5, 2]``
(``roles/<role>/ratio_outside_frac``); and the sampler's entropy over them, the mean negative log-probability its
tokens were sampled at (``roles/<role>/entropy``, as ``optim/entropy`` is the lane's).  A lane other than the leader's
logs them under ``policies/<id>/``, as its other metrics.

A row's role is its agent's, as the tags of a group's output give it
(:class:`~platoon.train.workflow.output.PolicyGroupOutput`, under an estimator that reads agents); a row without one,
of a group whose estimator reads none, counts in its lane's mass under no role.  A row's mass is the sum of ``|A|``
over its action tokens as the step sends them, each microbatch's advantages scaled by the action tokens it represents.
A run without such an entry logs none of these, and steps as it did.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import tinker

from platoon.train.policy.spec import PolicyId

# A token whose |log r| is above this has its ratio r outside [0.5, 2].
_OUTSIDE = math.log(2.0)


@dataclass
class _Tokens:
    """What a step counted of one role's action tokens as their results came back: how many, the sum of their
    ``|log r|``, how many have ``r`` outside ``[0.5, 2]``, and the sum of the negative log-probabilities they were
    sampled at."""

    count: float = 0.0
    abs_log_ratio: float = 0.0
    outside: float = 0.0
    surprise: float = 0.0


@dataclass
class _LaneRoles:
    """What a step counted of one lane's rows by role: the roles of the rows it kept, in the order it met them, each
    role's mass sent and the lane's whole (``total``), and each role's action tokens (:class:`_Tokens`)."""

    seen: dict[str, None] = field(default_factory=dict)
    mass: dict[str, float] = field(default_factory=dict)
    total: float = 0.0
    tokens: dict[str, _Tokens] = field(default_factory=dict)


class RoleMetrics:
    """What one step counts by role (the module's docstring): the role of each row it keeps, by the identity of the
    step's own copy of the row, which the batch transforms change in place and its learner's results come back with.
    It holds each copy it was given a role for until the step ends, so that no other copy takes the address of one the
    stale filter dropped or a finished minibatch let go, and answers a role only for the very copy it was given."""

    def __init__(self) -> None:
        self.roles: dict[int, tuple[tinker.Datum, str]] = {}
        self.lanes: dict[PolicyId, _LaneRoles] = {}

    def _lane(self, policy: PolicyId) -> _LaneRoles:
        return self.lanes.setdefault(policy, _LaneRoles())

    def role(self, datum: tinker.Datum) -> str | None:
        """The role of the agent of ``datum``, a copy the step kept (:meth:`keep`); None for a copy of a row without
        one, or any other datum."""

        held = self.roles.get(id(datum))
        return held[1] if held is not None and held[0] is datum else None

    def keep(self, policy: PolicyId, datums: Sequence[tinker.Datum], roles: Sequence[str | None]) -> None:
        """The step keeps ``datums``, its copies of a group's rows of ``policy``, whose agents have ``roles``."""

        lane = self._lane(policy)
        for datum, role in zip(datums, roles, strict=True):
            if role is not None:
                self.roles[id(datum)] = (datum, role)
                lane.seen.setdefault(role)

    def sent(self, policy: PolicyId, datums: Sequence[tinker.Datum], scale: float) -> None:
        """A microbatch of ``policy``'s lane sends ``datums``, their advantages multiplied by ``scale``."""

        lane = self._lane(policy)
        for datum in datums:
            inputs = datum.loss_fn_inputs
            mass = float(inputs["advantages"].to_torch()[inputs["mask"].to_torch() > 0].abs().sum()) * scale
            lane.total += mass
            if (role := self.role(datum)) is not None:
                lane.mass[role] = lane.mass.get(role, 0.0) + mass

    def observer(self, policy: PolicyId) -> Callable[[list[tinker.Datum], tinker.ForwardBackwardOutput], None]:
        """What reads each result of ``policy``'s microbatches (``Learner.update(observe=)``): the ratios of each
        role's action tokens, from the logprobs the step trained them at and those they were sampled at."""

        lane = self._lane(policy)

        def observe(originals: list[tinker.Datum], result: tinker.ForwardBackwardOutput) -> None:
            for datum, output in zip(originals, result.loss_fn_outputs, strict=True):
                if (role := self.role(datum)) is None:
                    continue
                inputs = datum.loss_fn_inputs
                actions = inputs["mask"].to_torch() > 0
                sampled = inputs["logprobs"].to_torch()[actions]
                log_ratios = (output["logprobs"].to_torch()[actions] - sampled).abs()
                counts = lane.tokens.setdefault(role, _Tokens())
                counts.count += float(log_ratios.numel())
                counts.abs_log_ratio += float(log_ratios.sum())
                counts.outside += float((log_ratios > _OUTSIDE).sum())
                counts.surprise -= float(sampled.sum())

        return observe

    def record(self, policy: PolicyId, tracker: Any) -> None:
        """Record ``policy``'s lane's metrics by role in its ``tracker`` (the step's, or under ``policies/<id>/``): a
        role whose rows the step kept has its mass share, 0 when it sent none of them, once its lane sent any mass;
        its ratios and its entropy, once it sent an action token."""

        lane = self.lanes.get(policy)
        if lane is None:
            return
        metrics = {}
        for role in lane.seen:
            if lane.total > 0:
                metrics[f"roles/{role}/mass_share"] = lane.mass.get(role, 0.0) / lane.total
            counts = lane.tokens.get(role)
            if counts is not None and counts.count:
                metrics[f"roles/{role}/log_ratio/abs_avg"] = counts.abs_log_ratio / counts.count
                metrics[f"roles/{role}/ratio_outside_frac"] = counts.outside / counts.count
                metrics[f"roles/{role}/entropy"] = counts.surprise / counts.count
        if metrics:
            tracker.scalar(**metrics)
