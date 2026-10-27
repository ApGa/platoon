"""The credit API (RFC 0021, section 6; experimental): how a group's rewards become the advantages its rows train with.

A credit estimator (:class:`CreditEstimator`, or a subclass of :class:`CreditEstimatorBase`) is chosen per environment
entry (``environments[].credit``, with its ``credit_options``) and built once per start, ``Class(setup, options)``
(:class:`CreditSetup`).  Its ``assign`` runs once per training group, in the trainer process, after every member has
settled: it sees the group (:class:`CreditGroup`) with every member (:class:`MemberRollout`), and, when it asks for
them (``needs_agents``), every agent of every member (:class:`AgentRun`: its role, its policy, its own reward and its
rows), and it sets the advantages of the rows (:class:`TrainingSample`), on their action tokens.  The error-token
filter, the group's verdict and its output follow, as without an estimator.  An evaluation never runs one.

The ``assign`` calls of different groups run concurrently, each switching to the others at its awaits: what an
estimator keeps of one group stays in the call's locals, or in the group.  An estimator with state to resume (a running
baseline, say) changes it in ``settle(group)`` and only there, and gives it as ``state()`` and takes it back with
``load_state(state)``: the run settles each group as it files the group with the groups a full checkpoint keeps, and
takes the estimators' state at the point it takes the kept groups, so that a checkpoint never holds an estimator's
state for a group it does not hold, nor the reverse (:mod:`platoon.train.credit.runtime`, which also says what stops
a run: an estimator that raises, or an ``assign`` that does not return within ``assign_timeout_seconds``).

This module imports neither Tinker nor torch, so that ``platoon.experimental`` re-exports it; the rows' methods import
torch when they are called.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from platoon.train.policy.spec import PolicyId

if TYPE_CHECKING:
    import tinker
    import torch

    from platoon.train.config_defs import WorkflowConfig
    from platoon.utils.stats_tracker import DeferredStats, StatsTracker
    from platoon.utils.tinker_data_processing import TrajectoryCollectionResult

# How long an estimator's ``assign`` may take on one group, in seconds, unless it says otherwise
# (``assign_timeout_seconds``): a call that does not return within it stops the run.
DEFAULT_ASSIGN_TIMEOUT_SECONDS = 600.0


@dataclass(eq=False)
class TrainingSample:
    """One row to train (``datum``), with who produced it: the policy that sampled it and trains on it, the member of
    the group whose rollout it is (its place among the members that returned a result), and, when the estimator asked
    for agents (``needs_agents``), the agent, its role, its trajectory and its depth in the agent tree.  It lives in
    the trainer process only: what reaches the training service is the datum."""

    datum: tinker.Datum
    policy: PolicyId
    member: int
    agent_id: str | None = None
    role: str | None = None
    trajectory_id: str | None = None
    depth: int | None = None

    @property
    def advantages(self) -> torch.Tensor:
        """The row's advantages, one per target token, as they are now: the converter loads every action token with its
        trajectory's processed reward, and the other positions with 0."""

        return self.datum.loss_fn_inputs["advantages"].to_torch()

    @property
    def mask(self) -> torch.Tensor:
        """Which of the row's target tokens are actions, which train (``mask > 0``), one per target token."""

        return self.datum.loss_fn_inputs["mask"].to_torch()

    def set_advantage(self, value: float) -> None:
        """Set ``value`` as the advantage of every action token of the row; its other positions keep theirs."""

        import torch

        advantages = self.advantages
        self._set(torch.where(self.mask > 0, torch.full_like(advantages, float(value)), advantages))

    def set_advantages(self, values: torch.Tensor | Sequence[float]) -> None:
        """Set one advantage per target token (``values``, as many as the row has target tokens) on the row's action
        tokens; its other positions keep theirs.  A ``ValueError`` for values of another length."""

        import torch

        advantages = self.advantages
        values = torch.as_tensor(values, dtype=advantages.dtype)
        if values.shape != advantages.shape:
            raise ValueError(
                f"set_advantages: {tuple(values.shape)} values for a row of {tuple(advantages.shape)} target tokens"
            )
        self._set(torch.where(self.mask > 0, values, advantages))

    def _set(self, advantages: torch.Tensor) -> None:
        from tinker import TensorData

        self.datum.loss_fn_inputs["advantages"] = TensorData.from_torch(advantages)


@dataclass
class AgentRun:
    """One agent of one member's rollout: its trajectory, its parent's trajectory (None for the root), its role
    (``policy`` for the root unless its program says otherwise; a child's is the role it was minted with), the policy it
    samples, its depth in the tree, its reward and its rows.  ``reward`` is its trajectory's processed reward, as the
    converter loaded it onto its action tokens; None when it was not scored: the agent was interrupted, or it is a
    child whose verifier gave no verdict (a judge error, an unparseable or a pending verdict), whose 0 is no score.  An
    estimator never trains an unscored agent's reward as 0, nor counts it in a baseline.  An agent whose rows the
    converter left out has no ``samples``: an unscored one, and a scored child that was sampled out or that its parent
    did not use, whose reward still counts."""

    agent_id: str
    trajectory_id: str
    parent_id: str | None
    role: str
    policy: PolicyId
    depth: int
    reward: float | None
    samples: list[TrainingSample] = field(default_factory=list)


@dataclass
class MemberRollout:
    """One member of the group that returned a result: its place among them (``index``), every row of its rollout
    (``samples``), its agents (filled when the estimator asked for them, else empty), its root's processed reward and
    whether that is a completed root's scored outcome (``task_reward_valid``: an interrupted root's is not, and enters
    no baseline), and what the converter returned (``result``), to read: its trajectories' statistics and its root's
    reward components.  Its ``datums`` are the member's rows; an estimator sets their advantages through ``samples``."""

    index: int
    samples: list[TrainingSample]
    agents: list[AgentRun]
    task_reward: float
    task_reward_valid: bool
    result: TrajectoryCollectionResult


@dataclass(frozen=True)
class PolicyView:
    """What an estimator may read of a policy: its id, and the renderer that turns its token ids into text."""

    id: PolicyId
    renderer: Any = None


@dataclass
class CreditGroup:
    """One training group, as its estimator sees it: its task, its label (its value of the run's label column) and its
    environment entry, the entry's training settings (``settings``), the members that returned a result, in the group's
    order, the run's policies, each policy's version when ``assign`` began (``versions``; it does not move during the
    call, although the policies may publish meanwhile), and where the group's own statistics go (``stats``: in a
    training group the ``DeferredStats`` that the step that consumes the group records, outside one the stats tracker;
    an estimator's metrics belong under ``credit/``)."""

    task_id: str
    label: str | None
    entry: str | None
    settings: WorkflowConfig
    members: list[MemberRollout]
    policies: Mapping[PolicyId, PolicyView]
    versions: Mapping[PolicyId, int]
    stats: DeferredStats | StatsTracker

    def samples(self) -> list[TrainingSample]:
        """Every row of the group, member by member, each member's in its order."""

        return [sample for member in self.members for sample in member.samples]

    def agents(self) -> list[AgentRun]:
        """Every agent of the group, member by member (none when the estimator did not ask for them)."""

        return [agent for member in self.members for agent in member.agents]


@dataclass(frozen=True)
class CreditSetup:
    """What an estimator is built with, once per start of the process and environment entry: ``Class(setup,
    options)``."""

    # Its name as the config gives it ("keyed", "acme.credit:ByRole").
    name: str
    # The label of the environment entry whose groups it credits.
    entry: str
    # The entry's training settings (train.workflow_config with the entry's own laid over it).
    settings: WorkflowConfig
    # The run's policies, by id: the one the run trains, main, in this release.
    policies: tuple[PolicyId, ...]


@dataclass(frozen=True)
class CreditCheckContext:
    """What an estimator's load-time ``check`` sees, beside its options."""

    # The label of the environment entry that names it.
    entry: str
    # The run's policies, by id.
    policies: tuple[PolicyId, ...]
    # The whole loaded config, read-only, to cross-check the settings the estimator relies on.
    config: Any


class CreditEstimator(Protocol):
    """What the trainer calls of a credit estimator: ``assign(group)``, once per training group, awaited.

    ``needs_agents``: True when it reads the group's agents (``MemberRollout.agents``) and the rows' tags; the rollout
    workers then tag each row with its agent (worker protocol 3).  Optional: ``settle(group)``, synchronous, the one
    place where an estimator with state changes it; ``state()`` and ``load_state(state)``, which give and take that
    state as JSON data; and ``assign_timeout_seconds``, how long ``assign`` may take on one group
    (:data:`DEFAULT_ASSIGN_TIMEOUT_SECONDS` without it; None: no limit).  An estimator that raises in any of them, or
    whose ``assign`` runs out of time, stops the run with an error that names it, its entry and the group."""

    needs_agents: ClassVar[bool]

    async def assign(self, group: CreditGroup) -> None:
        """Set the advantages of the group's rows (:meth:`TrainingSample.set_advantage`,
        :meth:`TrainingSample.set_advantages`).  A row it leaves alone keeps what the converter loaded: its trajectory's
        processed reward on every action token.  It may await (a service's call, say), and assume that every member
        has settled; a group is credited once, and a resume that trains a group a checkpoint kept trains it with the
        advantages this call set.  Other groups' calls run while it awaits."""
        ...


class CreditEstimatorBase:
    """An optional base class of credit estimators: the protocol's shape, typed options and a load-time check."""

    # Whether it reads the group's agents and the rows' tags (CreditEstimator.needs_agents).
    needs_agents: ClassVar[bool] = False
    # Its options' schema: a dataclass, a pydantic model, or a function of the options' mapping that returns them parsed
    # and raises ValueError; None: it takes no options.  ``self.options`` holds them parsed.
    options_schema: ClassVar[Any] = None
    # How long ``assign`` may take on one group, in seconds, before the run stops with an error that names the
    # estimator; None: no limit.  An estimator that calls a service may set it, from its options say, in ``__init__``.
    assign_timeout_seconds: float | None = DEFAULT_ASSIGN_TIMEOUT_SECONDS

    def __init__(self, setup: CreditSetup, options: Any = None) -> None:
        """Keep what the estimator is built with: ``setup`` (:class:`CreditSetup`) and ``options``, parsed against
        :attr:`options_schema` (None when it takes none)."""

        self.setup = setup
        self.options = options

    async def assign(self, group: CreditGroup) -> None:
        """Set the advantages of the group's rows (:meth:`CreditEstimator.assign`): a subclass implements it."""

        raise NotImplementedError(f"{type(self).__name__} does not implement assign(group)")

    @staticmethod
    def check(options: Any, ctx: CreditCheckContext) -> None:
        """Refuse a config it cannot serve, with a ``ValueError`` that says why: it accepts every config."""
