"""Step budgets for agent trees, keyed by agent id.

A *step* is one ``policy``-kind model call recorded on an agent (``aux`` and ``judge`` calls are free).  Each
agent's allocation is its ``max_steps`` (``None`` or 0: unbounded).  Two trackers, ported with the same
semantics and numbers from the legacy trajectory-keyed classes:

* :class:`StepBudgetTracker` (``budget: linear``): a subagent draws on its parent's budget.  An agent's used
  budget counts its descendants' steps, and ``spawn`` reserves the child's ``max_steps + 1`` on the parent
  until the child returns (the extra step is the parent's turn to read the result).
* :class:`DepthAwareStepBudgetTracker` (``budget: depth_aware``): every agent is bounded only by its own
  ``max_steps``; reservations always succeed, and ``max_depth`` caps the tree.

Both refuse a policy child below ``max_depth`` when one is set (legacy: only the depth-aware tracker had a
depth limit, so the linear default stays unlimited and ignores verifier scopes).  Reservations are atomic (a
lock, and no awaits), so concurrent spawns are safe.

**``max_depth`` bounds the policy tree, not its verifiers.**  This is by design, ported unchanged from the legacy
subagent launcher (``_subagent_depth_scope``).  Synthetic reward verifiers (``agents.child_grading: verifier``) sit
outside the policy recursion tree: a verifier root and one level of verifier helpers are admissible at any depth,
and a helper may not delegate again.  A verifier judges one policy child, so it has to be admissible under a child
at ``max_depth``: counting it against ``max_depth`` would refuse the verifier of every deepest child, which would
then never be trainable.  The physical tree therefore reaches ``max_depth + 2`` (the judged child's verifier one level
down, its helpers one more), and those levels add wall time and model calls that ``max_depth`` does not bound.
What bounds a verifier subtree is ``agents.verifier.max_steps`` (the verifier's own steps; with ``linear``
budgets its helpers draw on them too), each helper's ``max_steps`` (``depth_aware``), ``agents.child_timeout_s``
for every agent, and the rollout's timeout.
"""

from __future__ import annotations

import math
import threading
from collections import defaultdict
from collections.abc import Iterable
from typing import Literal, Protocol, TypeAlias, runtime_checkable

SubagentDepthScope: TypeAlias = Literal["policy", "verifier_root", "verifier_helper", "verifier_descendant"]

BUDGET_EXCEEDED_GUIDANCE = (
    "Note: launch_subagent will automatically reserve max_steps + 1 steps "
    "since you will need one or more steps to process the result of the "
    "subagent and complete the task. "
    "You could try requesting a smaller budget or perform the task yourself."
)


class BudgetExceededError(ValueError):
    """Raised when a budget limit would be exceeded.

    Attributes:
        reason: A short tag identifying the kind of budget that was exceeded
                (``"step_budget"``, ``"depth"`` or ``"verifier_depth"``).
        guidance: An actionable suggestion for the caller describing how to
                  work around the limitation.
    """

    def __init__(self, message: str, *, reason: str = "step_budget", guidance: str = ""):
        super().__init__(message)
        self.reason = reason
        self.guidance = guidance


class BudgetedAgent(Protocol):
    """What a tracker reads of an agent (``platoon.gateway.AgentSession``)."""

    agent_id: str
    parent_agent_id: str | None
    max_steps: int | None

    @property
    def policy_calls(self) -> int: ...


@runtime_checkable
class BudgetTracker(Protocol):
    def add_agent(self, agent: BudgetedAgent) -> None: ...

    def reserve_budget(
        self,
        agent_id: str,
        requested_budget: float,
        raise_on_failure: bool = False,
        *,
        child_depth_scope: SubagentDepthScope = "policy",
    ) -> bool: ...

    def release_budget(self, agent_id: str, amount_to_release: float) -> None: ...

    def remaining_budget_for(self, agent_id: str) -> float: ...

    def used_budget_for(self, agent_id: str) -> float: ...


class _AgentTree:
    """The registered agents and their child links."""

    def __init__(self, max_depth: int | None = None) -> None:
        if max_depth is not None and (isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError(f"max_depth must be a non-negative integer or None, got {max_depth!r}")
        self.max_depth = max_depth
        self._agents: dict[str, BudgetedAgent] = {}
        self._children: defaultdict[str, list[str]] = defaultdict(list)
        self._lock = threading.RLock()

    def add_agent(self, agent: BudgetedAgent) -> None:
        with self._lock:
            if agent.agent_id in self._agents:
                return
            self._agents[agent.agent_id] = agent
            if agent.parent_agent_id is not None:
                self._children[agent.parent_agent_id].append(agent.agent_id)

    def _agent(self, agent_id: str) -> BudgetedAgent:
        try:
            return self._agents[agent_id]
        except KeyError:
            raise KeyError(f"Agent {agent_id!r} is not registered with the budget tracker") from None

    def _allocated_budget(self, agent_id: str) -> float:
        return self._agent(agent_id).max_steps or math.inf

    def _iter_descendant_ids(self, agent_id: str) -> Iterable[str]:
        stack = [agent_id]
        while stack:
            for child_id in self._children.get(stack.pop(), ()):
                yield child_id
                stack.append(child_id)

    def depth(self, agent_id: str) -> int:
        """The depth of ``agent_id`` among the registered agents (root = 0)."""
        depth, agent = 0, self._agent(agent_id)
        while agent.parent_agent_id is not None and agent.parent_agent_id in self._agents:
            depth += 1
            agent = self._agents[agent.parent_agent_id]
        return depth

    def _check_depth(self, agent_id: str, raise_on_failure: bool, child_depth_scope: SubagentDepthScope) -> bool:
        """Policy and synthetic-verifier depth constraints for a child of ``agent_id``.

        Synthetic reward verifiers are outside the policy recursion tree, so a verifier root and one verifier
        helper level remain admissible even when their physical depth exceeds ``max_depth`` (by design: see the
        module docstring).  A helper may not recursively delegate again; this keeps the exception narrow and
        prevents an unbounded verifier tree.
        """
        if child_depth_scope in {"verifier_root", "verifier_helper"}:
            return True
        if child_depth_scope == "verifier_descendant":
            if raise_on_failure:
                raise BudgetExceededError(
                    "Verifier helper agents may not launch further subagents.",
                    reason="verifier_depth",
                    guidance=(
                        "The synthetic verifier tree allows one helper level. "
                        "Inspect the evidence directly and return it to the verifier."
                    ),
                )
            return False
        if self.max_depth is not None:
            current_depth = self.depth(agent_id)
            if current_depth + 1 > self.max_depth:
                if raise_on_failure:
                    raise BudgetExceededError(
                        f"Launching a subagent from depth {current_depth} would exceed "
                        f"the maximum allowed depth of {self.max_depth}.",
                        reason="depth",
                        guidance=(
                            "The maximum depth for hierarchical delegation has been reached. "
                            "You should perform the task yourself instead of delegating."
                        ),
                    )
                return False
        return True


class StepBudgetTracker(_AgentTree):
    """Subagent steps draw on the parent's budget (``budget: linear``)."""

    def __init__(self, max_depth: int | None = None) -> None:
        super().__init__(max_depth)
        self.reserved_agent_budgets: defaultdict[str, float] = defaultdict(float)

    def _used_budget(self, agent_id: str, recursive: bool = False) -> float:
        used_steps = self._agent(agent_id).policy_calls
        if recursive:
            for descendant_id in self._iter_descendant_ids(agent_id):
                used_steps += self._agents[descendant_id].policy_calls
        return used_steps

    def used_budget_for(self, agent_id: str) -> float:
        with self._lock:
            return self._used_budget(agent_id, recursive=True)

    def remaining_budget_for(self, agent_id: str) -> float:
        with self._lock:
            return (
                self._allocated_budget(agent_id)
                - self._used_budget(agent_id, recursive=True)
                - self.reserved_agent_budgets[agent_id]
            )

    def reserve_budget(
        self,
        agent_id: str,
        requested_budget: float,
        raise_on_failure: bool = False,
        *,
        child_depth_scope: SubagentDepthScope = "policy",
    ) -> bool:
        with self._lock:
            # Legacy linear budgets had no depth limit and ignored the scope; a max_depth opts into both.
            if self.max_depth is not None and not self._check_depth(agent_id, raise_on_failure, child_depth_scope):
                return False
            remaining = self.remaining_budget_for(agent_id)
            if remaining < requested_budget:
                if raise_on_failure:
                    raise BudgetExceededError(
                        f"Requested step budget {requested_budget} exceeds remaining budget {remaining}.",
                        reason="step_budget",
                        guidance=BUDGET_EXCEEDED_GUIDANCE,
                    )
                return False
            self.reserved_agent_budgets[agent_id] += requested_budget
            return True

    def release_budget(self, agent_id: str, amount_to_release: float) -> None:
        with self._lock:
            self.reserved_agent_budgets[agent_id] -= amount_to_release
            if self.reserved_agent_budgets[agent_id] < 0:
                self.reserved_agent_budgets[agent_id] = 0


class DepthAwareStepBudgetTracker(_AgentTree):
    """Each agent is bounded by its own ``max_steps``; ``max_depth`` caps the tree (``budget: depth_aware``)."""

    def used_budget_for(self, agent_id: str) -> float:
        """Only the agent's own steps count towards its budget."""
        with self._lock:
            return self._agent(agent_id).policy_calls

    def remaining_budget_for(self, agent_id: str) -> float:
        with self._lock:
            return self._allocated_budget(agent_id) - self._agent(agent_id).policy_calls

    def reserve_budget(
        self,
        agent_id: str,
        requested_budget: float,
        raise_on_failure: bool = False,
        *,
        child_depth_scope: SubagentDepthScope = "policy",
    ) -> bool:
        """Check the policy and synthetic-verifier depth constraints; steps are never reserved."""
        _ = requested_budget
        with self._lock:
            self._agent(agent_id)
            return self._check_depth(agent_id, raise_on_failure, child_depth_scope)

    def release_budget(self, agent_id: str, amount_to_release: float) -> None:
        """No-op: subagent steps do not consume parent budget."""


BudgetMode = Literal["linear", "depth_aware"]


def make_tracker(mode: BudgetMode, *, max_depth: int | None = None) -> StepBudgetTracker | DepthAwareStepBudgetTracker:
    """The tracker for an environment's ``agents.budget`` mode."""

    if mode == "linear":
        return StepBudgetTracker(max_depth=max_depth)
    if mode == "depth_aware":
        return DepthAwareStepBudgetTracker(max_depth=max_depth)
    raise ValueError(f"Unknown budget mode {mode!r}; expected 'linear' or 'depth_aware'")


class AgentBudget:
    """``ctx.budget``: one agent's view of the rollout's tracker."""

    def __init__(self, tracker: BudgetTracker, agent_id: str) -> None:
        self._tracker = tracker
        self.agent_id = agent_id

    def remaining(self) -> float:
        return self._tracker.remaining_budget_for(self.agent_id)

    def used(self) -> float:
        return self._tracker.used_budget_for(self.agent_id)

    def __repr__(self) -> str:
        return f"AgentBudget(agent_id={self.agent_id!r}, used={self.used()}, remaining={self.remaining()})"


__all__ = [
    "BudgetExceededError",
    "BudgetMode",
    "BudgetedAgent",
    "BudgetTracker",
    "DepthAwareStepBudgetTracker",
    "AgentBudget",
    "StepBudgetTracker",
    "SubagentDepthScope",
    "make_tracker",
]
