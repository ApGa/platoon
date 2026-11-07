"""``credit: roles``: each role of an entry's agent trees credited by an estimator of its own, in the one credit call
of each group, with a share of the entry's advantage mass held for the roles that set one.

An entry whose trees hold agents of several roles may need other credit for each, such as an environment generator's
(a paper-to-task pipeline, say), whose root writes a task once per draw and whose children solve it: the root against
running statistics across groups, the children leave-one-out among each other.  ``by_role`` names each role's
estimator and its options, as an entry names its own, and what the composite adds to the role's rows::

    credit: roles
    credit_options:
      by_role:
        policy: {credit: running, options: {strata: [kind]}, share: 0.1}       # the root: the generator
        solver: {credit: keyed, options: {key: [role], leave_one_out: true}}  # the children it mints

- **A view per role.** Each role's estimator credits the group's agents of its role alone
  (:meth:`~platoon.train.credit.api.CreditGroup.only`), a view over the group's very rows, and records its metrics under
  ``credit/roles/<role>/`` in place of ``credit/``.  It sees only the groups that hold an agent of its role, each in a
  view it settles too, and its state is part of the composite's.  An agent of a role ``by_role`` does not name trains
  nothing: its rows get advantages of 0 (warned of once per role).
- **A KL term per role** (``kl: {to, coef}``): after the role's estimator, each scored agent's rows of the role add
  ``-coef * (logprob - to's logprob)`` on every action token, the reverse KL to the frozen policy ``to``, scored as
  ``reverse_kl`` scores its teacher's (:func:`~platoon.train.credit.reverse_kl.reverse_kl_terms`); a scoring that fails
  fails the group.
- **A share per role** (``share``): the role's share of the advantage mass that its policy's rows of the entry carry,
  held by a weight on its rows, ``w = share / (1 - shares) * M_unshared / M_role``.  A role's mass is the decayed sum,
  over the groups that settled, of ``|A|`` over its rows' action tokens as its estimator and its KL term left them;
  ``M_unshared`` is that of the named roles of its policy without a share, and ``shares`` the sum of the shares of the
  roles whose mass is known.  The masses move only as groups settle, each policy's halving every
  :data:`MASS_HALF_LIFE` of its versions, and a group's weight reads them as they were when its agents of the role
  began sampling (the oldest version of their rows, as ``running`` reads its statistics), kept as far back as the
  policy's rows may be old and still train: positive, and blind to the group's outcome and to how long the group took
  to settle, it keeps its estimator's signs and expectation.  A group whose agents of the role began sampling before
  its masses were known reads them as they are when its credit begins, and a role whose mass, or whose unshared roles'
  mass, is not known yet trains at weight 1.  When a role with a share first carries mass (as ``running``'s warm-up
  ends, say), its policy's masses start over from that group, so that the role's and the others' cover the same
  groups.

Each group's metrics: ``credit/role_mass/<role>``, its rows' advantage mass as they train, summed over the step's
groups (``credit/role_groups`` counts those); ``credit/role_weight/<role>``, the weight the rows of a role with a share
trained at, and ``credit/role_kl/<role>``, the mean reverse KL per action token of a role with a KL term.  A run with an
entry under ``roles`` (its class's ``logs_by_role``) also logs, at each step, every role's share of the advantage mass
the step sends, over every entry's rows, how far its importance ratios strayed and its sampler's entropy
(``roles/<role>/*``, :mod:`platoon.train.step.roles`).
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from platoon.train.credit.api import (
    DEFAULT_ASSIGN_TIMEOUT_SECONDS,
    AgentRun,
    CreditCheckContext,
    CreditEstimatorBase,
    CreditGroup,
    CreditSetup,
    ScoringFailed,
    TrainingSample,
)
from platoon.train.credit.registry import resolve_credit_estimator
from platoon.train.credit.reverse_kl import frozen_scorer_problem, loaded_tokenizer_differs, reverse_kl_terms
from platoon.train.credit.runtime import class_ref
from platoon.train.policy.spec import ROOT_ROLE, PolicyId, agent_bindings
from platoon.utils.config import ConfigError

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# How many versions of a policy the masses of its roles take to halve.
MASS_HALF_LIFE = 4.0
# How many versions back a read of a policy's masses reaches when the setup gives its rows no staleness bound (a setup
# built outside a run, or an entry without a bound), as running's statistics do.
DEFAULT_WINDOW = 64
# The attribute of a group that holds what its credit kept for its settle: the views its roles' estimators credited,
# and its rows' masses before their weights (rule 4 of the credit API: in the group, never in the estimator).
_CREDITED = "_roles_credited"


@dataclass(frozen=True)
class RoleKL:
    """``by_role.<role>.kl`` under ``credit: roles``: a reverse KL term on the role's rows."""

    # The frozen policy the role's sampled tokens are compared with: the id of a policies: entry with trainable: false,
    # of the student's tokenizer.
    to: str
    # The weight of the term, above 0: each action token's advantage less coef times the reverse KL at it.
    coef: float


@dataclass(frozen=True)
class RoleCredit:
    """``by_role.<role>`` under ``credit: roles``: how the agents of one role are credited."""

    # The role's estimator, as an entry names its own: one of core's (group, keyed, reverse_kl, running), a name an
    # entry's package registers, or module:Class; not roles.
    credit: str
    # Its options, as an entry's credit_options would be, checked against its options when the config loads.
    options: dict[str, Any] = field(default_factory=dict)
    # The role's share of the advantage mass its policy's rows of the entry carry, above 0 and below 1, held by a
    # weight on its rows. None: its rows keep the mass its estimator gives them.
    share: float | None = None
    # A reverse KL term on the role's rows (to, coef). None: none.
    kl: RoleKL | None = None


@dataclass(frozen=True)
class RolesOptions:
    """``credit_options`` under ``credit: roles``."""

    # Each role's credit, by role: a role of the entry's agents (a root's is policy unless its program says otherwise,
    # a child's the role it was minted with). The agents of a role it does not name train nothing.
    by_role: dict[str, RoleCredit]

    def __post_init__(self) -> None:
        """Resolve each role's estimator and parse its options against its schema, check its share and its KL term,
        every problem under its path; and refuse shares that leave no mass to the roles without one."""

        from platoon.extensions import ExtensionError
        from platoon.options import schema_options

        if not isinstance(self.by_role, dict) or not self.by_role:
            raise ValueError(f"by_role: one or more roles, each with its credit, got {self.by_role!r}")
        problems: list[str] = []
        estimators: dict[str, tuple[type, Any]] = {}
        for role, credit in self.by_role.items():
            where = f"by_role.{role}"
            if not isinstance(role, str) or not role:
                problems.append(f"by_role: a role is a non-empty name, got {role!r}")
                continue
            problems += _role_problems(where, credit)
            try:
                cls = resolve_credit_estimator(credit.credit)
            except ExtensionError as exc:
                problems.append(f"{where}.credit: {exc}")
                continue
            if issubclass(cls, Roles):
                problems.append(
                    f"{where}.credit: {credit.credit} credits the roles of a tree, and a role's estimator its "
                    "agents alone: name another"
                )
                continue
            schema = schema_options(getattr(cls, "options_schema", None), name=credit.credit)
            parsed, found = schema.check(credit.options, f"{where}.options")
            problems += found
            if not found:
                estimators[role] = (cls, parsed)
                object.__setattr__(credit, "options", schema.as_json(parsed, credit.options))  # as lock.json records
        shares = [credit.share for credit in self.by_role.values() if credit.share is not None]
        if shares and all(_share(share) for share in shares) and sum(shares) >= 1:
            problems.append(
                f"by_role: the shares add up to {sum(shares):g}: each is a role's share of the mass, beside the mass "
                "of the roles without one, so together they come to less than 1"
            )
        if shares and len(shares) == len(self.by_role):
            problems.append(
                "by_role: every role has a share, and a share is held against the mass of the roles without one: "
                "leave out the share of one role, whose rows keep the mass their estimator gives them"
            )
        if problems:
            raise ConfigError(problems)
        object.__setattr__(self, "_estimators", estimators)

    def estimators(self) -> dict[str, tuple[type, Any]]:
        """Each role's estimator class and its options parsed, by role, in ``by_role``'s order."""

        return dict(getattr(self, "_estimators"))


def _role_problems(where: str, credit: RoleCredit) -> list[str]:
    """What is wrong with a role's share and its KL term, each under its path (``by_role.<role>.share``, ...)."""

    problems = []
    if credit.share is not None and not _share(credit.share):
        problems.append(f"{where}.share: a number above 0 and below 1, got {credit.share!r}")
    if credit.kl is not None:
        if not isinstance(credit.kl.to, str) or not credit.kl.to:
            problems.append(f"{where}.kl.to: the id of a frozen policy of policies:, got {credit.kl.to!r}")
        if not _positive(credit.kl.coef):
            problems.append(f"{where}.kl.coef: a finite number above 0, got {credit.kl.coef!r}")
    return problems


def _positive(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value) and value > 0


def _share(value: Any) -> bool:
    return _positive(value) and value < 1


def _unheld_shares(options: RolesOptions, config: Any) -> list[str]:
    """Each role with a share whose agents sample a policy that no role ``by_role`` names without a share samples:
    the masses are kept per policy, and a share is held against the mass of its own policy's rows of the roles without
    one, so it could never hold (its rows would train at weight 1).  The root agent's role samples the agents' default
    policy, and any other its ``agents.roles`` binding's, else the default."""

    bindings = agent_bindings(config)

    def policy_of(role: str) -> PolicyId:
        return bindings.default if role == ROOT_ROLE else bindings.roles.get(role, bindings.default)

    unshared = {policy_of(role) for role, credit in options.by_role.items() if credit.share is None}
    return [
        f"by_role.{role}.share: the agents of role {role} sample the policy {policy_of(role)}, and no role by_role "
        f"names without a share samples it: a share is held against the mass of its own policy's rows of the roles "
        f"without one, so this one could never hold; leave it out, or name a role of {policy_of(role)} without one"
        for role, credit in options.by_role.items()
        if credit.share is not None and policy_of(role) not in unshared
    ]


@dataclass
class _Masses:
    """The decayed advantage masses of each role of one policy's rows, by version: as they were after the settles of
    each version a read may still need (``after``, in order, each aged to its version), from the newest of those a read
    no longer reaches, which the reads after it start from, to the newest, which takes the settles of its version.  A
    read reaches ``window`` versions back from the newest."""

    window: int
    after: dict[int, dict[str, float]] = field(default_factory=dict)

    def live(self) -> dict[str, float]:
        """The masses with every settle so far (none before any)."""

        return self.after[next(reversed(self.after))] if self.after else {}

    def at(self, version: int) -> dict[str, float] | None:
        """The masses as they were when the policy reached ``version``: after the settles of the newest version below
        it, unaged (aging scales every role's alike, and a weight reads their ratios); None when nothing settled
        before it, or the window no longer reaches it."""

        if not self.after or version <= next(reversed(self.after)) - self.window:
            return None
        earlier = [at for at in self.after if at < version]
        return self.after[earlier[-1]] if earlier else None

    def settle(self, version: int, masses: dict[str, float], shared: Collection[str]) -> None:
        """Add a group's ``masses`` by role at policy version ``version``: at a version past the newest, the masses
        start from the newest's, aged to it, and those no read in the window needs any more leave; at an older one,
        which a resume took back, no time passed: it is the newest's.  When a role among ``shared`` (those with a
        share) first carries mass, the masses start over from the group's."""

        newest = next(reversed(self.after)) if self.after else None
        if newest is None or version > newest:
            kept = 1.0 if newest is None else 0.5 ** ((version - newest) / MASS_HALF_LIFE)
            self.after[version] = {role: mass * kept for role, mass in self.live().items()}
            for at in [at for at in self.after if at <= version - self.window][:-1]:
                del self.after[at]
            newest = version
        held = self.after[newest]
        if any(mass > 0 and not held.get(role) for role, mass in masses.items() if role in shared):
            held.clear()  # the role's mass and the others' cover the same groups from here on
        for role, mass in masses.items():
            held[role] = held.get(role, 0.0) + mass


@dataclass(frozen=True)
class _Credited:
    """What a group's credit kept for its settle: the view each role's estimator credited, and the group's rows'
    advantage masses before their weights, by policy and role."""

    views: dict[str, CreditGroup]
    masses: dict[tuple[PolicyId, str], float]


class Roles(CreditEstimatorBase):
    """Each role of an entry's trees credited by its own estimator, with shares of the advantage mass (the module's
    docstring)."""

    needs_agents = True
    credits_single_members = True
    # A run with an entry it credits logs its steps' metrics by agent role (platoon.train.step.roles).
    logs_by_role: ClassVar[bool] = True
    options_schema = RolesOptions
    options: RolesOptions
    # Each role's estimator awaits under its own assign_timeout_seconds, and a KL term's scoring under its policy's
    # sample_timeout_seconds.
    assign_timeout_seconds = None
    # Its roles' estimators' metrics go under this, then the role, in place of credit/.
    PREFIX: ClassVar[str] = "credit/roles/"

    def __init__(self, setup: CreditSetup, options: RolesOptions) -> None:
        """Build each role's estimator with the entry's setup, under the role's own name and options; a
        :class:`~platoon.utils.config.ConfigError` (exit 78) for a KL term's policy whose tokenizer, as the run loaded
        it, differs from the student's."""

        super().__init__(setup, options)
        self._estimators: dict[str, Any] = {
            role: cls(_named(setup, options.by_role[role].credit), parsed)
            for role, (cls, parsed) in options.estimators().items()
        }
        for role, credit in options.by_role.items():
            if credit.kl is not None and loaded_tokenizer_differs(setup, credit.kl.to):
                raise ConfigError(
                    f"{setup.entry}: credit_options.by_role.{role}.kl.to: policies.{credit.kl.to}'s tokenizer, as the "
                    "run loaded it, differs from the student's, so it cannot score the student's token ids: name a "
                    "policy of the student's model family"
                )
        self._shares = {role: credit.share for role, credit in options.by_role.items() if credit.share is not None}
        self._masses: dict[PolicyId, _Masses] = {}
        self._warned: set[str] = set()  # the roles by_role does not name whose rows were zeroed, warned of once

    @staticmethod
    def role_estimators(options: RolesOptions) -> dict[str, tuple[str, type, dict[str, Any]]]:
        """Each role's estimator, by role, as ``lock.json`` records it beside the composite's: its name as ``by_role``
        gives it, its class, and its options with their defaults."""

        return {
            role: (options.by_role[role].credit, cls, dict(options.by_role[role].options))
            for role, (cls, _) in options.estimators().items()
        }

    @staticmethod
    def check(options: RolesOptions, ctx: CreditCheckContext) -> None:
        """Each role's estimator's own check; a KL term's policy is a frozen policy the config declares, of the
        student's tokenizer (:func:`~platoon.train.credit.reverse_kl.frozen_scorer_problem`); and each role with a
        share has a role of its policy without one beside it (:func:`_unheld_shares`)."""

        problems = []
        for role, (cls, parsed) in options.estimators().items():
            method = getattr(cls, "check", None)
            if callable(method):
                try:
                    method(parsed, ctx)
                except Exception as exc:
                    name = options.by_role[role].credit
                    problems.append(f"by_role.{role}: {name}: {str(exc) or type(exc).__name__}")
        for role, credit in options.by_role.items():
            what = "the policy a kl term scores with"
            if credit.kl is not None and (problem := frozen_scorer_problem(credit.kl.to, ctx.config, what=what)):
                problems.append(f"by_role.{role}.kl.to: {problem}")
        problems += _unheld_shares(options, ctx.config)
        if problems:
            raise ValueError("; ".join(problems))

    async def assign(self, group: CreditGroup) -> None:
        weights = self._weights(group)  # what settled before the group's roles began sampling: before anything awaits
        self._zero_unnamed(group)
        views = {}
        for role, estimator in self._estimators.items():
            views[role] = view = _view(group, role)
            if view.agents():
                await _credit_role(role, self.options.by_role[role], estimator, view)
        divergences = {}
        for role, credit in self.options.by_role.items():
            scored = [agent for agent in views[role].agents() if agent.reward is not None and agent.samples]
            if credit.kl is not None and scored:
                divergences[role] = await self._add_kl(role, credit.kl, scored)
        masses = _masses(group, self._estimators)
        applied = {}
        for (policy, role), weight in weights.items():
            applied[(policy, role)] = weight
            for sample in _rows(group, policy, role):
                sample.set_advantages(sample.advantages * weight)
        setattr(group, _CREDITED, _Credited(views, masses))
        self._record(group, applied, divergences)

    def settle(self, group: CreditGroup) -> None:
        """Settle each role's view of the group into its estimator, and move each policy's masses by the group's rows,
        as its credit left them before their weights; a group this estimator did not credit moves no mass."""

        credited = getattr(group, _CREDITED, None)
        if credited is not None:
            delattr(group, _CREDITED)
        views = credited.views if credited is not None else {role: _view(group, role) for role in self._estimators}
        for role, estimator in self._estimators.items():
            settle, view = getattr(estimator, "settle", None), views[role]
            if callable(settle) and view.agents():
                try:
                    settle(view)
                except Exception as exc:
                    raise RuntimeError(f"by_role.{role}: {_raised('settle', self.options.by_role[role], exc)}") from exc
        if credited is not None:
            self._move(group, credited.masses)

    def state(self) -> dict[str, Any]:
        return {
            "estimators": {
                role: {"class": class_ref(estimator), "state": estimator.state()}
                for role, estimator in self._estimators.items()
                if callable(getattr(estimator, "state", None))
            },
            "masses": [
                [
                    policy,
                    [[version, [[role, held[role]] for role in sorted(held)]] for version, held in kept.after.items()],
                ]
                for policy, kept in sorted(self._masses.items())
            ],
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Take up each role's estimator's state, and the masses; a role whose estimator is of another class than the
        one that saved its state (its ``credit`` changed) starts afresh, with a warning."""

        for role, saved in state["estimators"].items():
            estimator = self._estimators.get(role)
            if estimator is None or not callable(getattr(estimator, "load_state", None)):
                continue
            if saved["class"] != class_ref(estimator):
                logger.warning(
                    "environments[%s].credit_options.by_role.%s starts afresh: the checkpoint holds the state of "
                    "another credit estimator, %s",
                    self.setup.entry,
                    role,
                    saved["class"],
                )
                continue
            estimator.load_state(saved["state"])
        self._masses = {}
        for policy, kept in state["masses"]:
            masses = self._masses[str(policy)] = _Masses(self._window(str(policy)))
            for version, sums in kept:
                masses.after[int(version)] = {str(role): float(mass) for role, mass in sums}

    def _weights(self, group: CreditGroup) -> dict[tuple[PolicyId, str], float]:
        """The weight of the group's rows of each role with a share, by policy and role (the module's docstring): from
        the masses as they were when its agents of the role began sampling, the oldest version of their rows; when
        they were not known then, as they are now; none (weight 1) while they are not known."""

        weights = {}
        for policy, masses in sorted(self._masses.items()):
            for role in sorted(self._shares):
                if not (rows := _rows(group, policy, role)):
                    continue
                now = group.versions.get(policy, 0)
                sampled = min(now if row.version is None else row.version for row in rows)
                weight = self._weight(role, masses.at(sampled))
                if weight is None:
                    weight = self._weight(role, masses.live())
                if weight is not None:
                    weights[(policy, role)] = weight
        return weights

    def _weight(self, role: str, masses: dict[str, float] | None) -> float | None:
        """``share / (1 - shares) * M_unshared / M_role`` from one policy's ``masses``, ``shares`` the sum of the
        shares of the roles whose mass is known, each sum in the order of the roles' names (so that a resume, which
        rebuilds the masses in that order, reads the weight alike to the last bit); None when the role's mass or the
        unshared roles' is not known."""

        if masses is None:
            return None
        names = sorted(masses)
        unshared = sum(masses[name] for name in names if name not in self._shares)
        mass = masses.get(role, 0.0)
        if unshared <= 0 or mass <= 0:
            return None
        rest = 1.0 - sum(self._shares[name] for name in names if name in self._shares and masses[name] > 0)
        return self._shares[role] / rest * unshared / mass

    def _move(self, group: CreditGroup, masses: dict[tuple[PolicyId, str], float]) -> None:
        """Add the group's masses to each policy's, at its version now (:meth:`settle`)."""

        for policy in sorted({policy for policy, _ in masses}):
            held = self._masses.get(policy)
            if held is None:
                held = self._masses[policy] = _Masses(self._window(policy))
            ours = {role: mass for (of, role), mass in masses.items() if of == policy}
            held.settle(_version_now(group, policy), ours, self._shares)

    def _window(self, policy: PolicyId) -> int:
        """How many versions back a read of ``policy``'s masses reaches: as many as its rows of the entry may be old
        and still train, and the newest (:data:`DEFAULT_WINDOW` when the setup gives no bound)."""

        bound = self.setup.max_staleness.get(policy)
        return DEFAULT_WINDOW if bound is None else bound + 1

    def _zero_unnamed(self, group: CreditGroup) -> None:
        """Set 0 on every action token of the rows of the agents whose role ``by_role`` does not name: they train
        nothing (warned of once per role)."""

        for agent in group.agents():
            if agent.role in self._estimators:
                continue
            for sample in agent.samples:
                sample.set_advantage(0.0)
            if agent.samples and agent.role not in self._warned:
                self._warned.add(agent.role)
                logger.warning(
                    "environments[%s].credit: roles: the agents of role %s train nothing (their rows get advantages "
                    "of 0): credit_options.by_role names %s, not %s (warned once per role)",
                    group.entry,
                    agent.role,
                    ", ".join(self._estimators),
                    agent.role,
                )

    async def _add_kl(self, role: str, kl: RoleKL, agents: list[AgentRun]) -> float | None:
        """Add the KL term to the rows of the role's scored ``agents``: ``-coef`` times the reverse KL to ``kl.to`` at
        each action token (:func:`~platoon.train.credit.reverse_kl.reverse_kl_terms`, whose ``ScoringFailed`` fails
        the group); the mean reverse KL per action token, None for rows without one."""

        view = self.setup.policy_views.get(kl.to)
        if view is None:
            raise RuntimeError(f"by_role.{role}.kl.to: policy {kl.to} is not among the run's policies here")
        samples = [sample for agent in agents for sample in agent.samples]
        terms = await reverse_kl_terms(view, samples)
        total, tokens = 0.0, 0
        for sample, term in zip(samples, terms, strict=True):
            sample.set_advantages(sample.advantages - kl.coef * term)
            actions = sample.mask > 0
            total += float(term[actions].sum())
            tokens += int(actions.sum())
        return total / tokens if tokens else None

    def _record(
        self, group: CreditGroup, weights: dict[tuple[PolicyId, str], float], divergences: dict[str, float | None]
    ) -> None:
        """The group's ``credit/role_*`` metrics (the module's docstring), with the ``weights`` its rows trained at and
        the mean reverse KL of each role with a KL term (``divergences``)."""

        import torch

        from platoon.utils.stats_tracker import ReduceType

        final: dict[str, float] = dict.fromkeys(self._estimators, 0.0)
        for (_, role), mass in _masses(group, self._estimators).items():
            final[role] += mass
        group.stats.denominator(**{"credit/role_groups": torch.ones(1, dtype=torch.bool)})
        group.stats.stat(
            "credit/role_groups",
            ReduceType.SUM,
            **{f"credit/role_mass/{role}": torch.tensor([mass]) for role, mass in final.items()},
        )
        scalars = {f"credit/role_weight/{role}": weight for (_, role), weight in weights.items()}
        scalars |= {f"credit/role_kl/{role}": kl for role, kl in divergences.items() if kl is not None}
        if scalars:
            group.stats.scalar(**scalars)


class _RoleStats:
    """A group's statistics as one role's estimator records into them: every name under ``credit/roles/<role>/``, in
    place of the ``credit/`` it begins with, so that two roles' estimators of one kind keep their metrics apart."""

    def __init__(self, stats: Any, role: str) -> None:
        self.stats = stats
        self.prefix = f"{Roles.PREFIX}{role}/"

    def _name(self, key: str) -> str:
        return self.prefix + key.removeprefix("credit/")

    def scalar(self, **values: float) -> None:
        self.stats.scalar(**{self._name(key): value for key, value in values.items()})

    def denominator(self, **values: torch.Tensor) -> None:
        self.stats.denominator(**{self._name(key): value for key, value in values.items()})

    def stat(self, denominator: str, reduce_type: Any = None, **values: torch.Tensor) -> None:
        self.stats.stat(
            self._name(denominator), reduce_type, **{self._name(key): value for key, value in values.items()}
        )


def _view(group: CreditGroup, role: str) -> CreditGroup:
    """The group as ``role``'s estimator sees it: its agents of the role and their rows, its metrics under the role's
    prefix (:class:`_RoleStats`)."""

    view = group.only(roles=[role])
    view.stats = _RoleStats(group.stats, role)  # type: ignore[assignment]  # what a role's metrics go through
    return view


async def _credit_role(role: str, credit: RoleCredit, estimator: Any, view: CreditGroup) -> None:
    """Await a role's estimator's ``assign`` over its view, within its own ``assign_timeout_seconds``; what it raises
    names the role, but a policy's failed scoring (``ScoringFailed``), which fails the group alone."""

    seconds = getattr(estimator, "assign_timeout_seconds", DEFAULT_ASSIGN_TIMEOUT_SECONDS)
    limit = asyncio.timeout(seconds)
    try:
        async with limit:
            await estimator.assign(view)
    except ScoringFailed:
        raise
    except Exception as exc:
        if limit.expired():
            raise RuntimeError(
                f"by_role.{role}: {credit.credit}'s assign(group) did not return within {seconds:g} s"
            ) from exc
        raise RuntimeError(f"by_role.{role}: {_raised('assign', credit, exc)}") from exc


def _raised(method: str, credit: RoleCredit, exc: BaseException) -> str:
    return f"{credit.credit}'s {method}(group) raised {type(exc).__name__}: {exc}"


def _named(setup: CreditSetup, name: str) -> CreditSetup:
    """The entry's setup, under the name of one role's estimator."""

    import dataclasses

    return dataclasses.replace(setup, name=name)


def _rows(group: CreditGroup, policy: PolicyId, role: str) -> list[TrainingSample]:
    """The rows of the group's agents of ``role`` that train ``policy``."""

    return [
        sample for agent in group.agents() if agent.role == role for sample in agent.samples if sample.policy == policy
    ]


def _masses(group: CreditGroup, roles: Any) -> dict[tuple[PolicyId, str], float]:
    """The advantage mass of the group's rows of each of ``roles`` as they are now, by the policy each trains and its
    role: the sum of ``|A|`` over their action tokens."""

    masses: dict[tuple[PolicyId, str], float] = {}
    for agent in group.agents():
        if agent.role not in roles:
            continue
        for sample in agent.samples:
            key = (sample.policy, agent.role)
            masses[key] = masses.get(key, 0.0) + float(sample.advantages[sample.mask > 0].abs().sum())
    return masses


def _version_now(group: CreditGroup, policy: PolicyId) -> int:
    """A policy's version as the group settles: its version now, and at least its version when ``assign`` began."""

    view = group.policies.get(policy)
    return max(view.version if view is not None else 0, group.versions.get(policy, 0))
