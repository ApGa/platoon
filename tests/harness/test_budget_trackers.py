"""Agent-keyed step budgets (platoon.harness.budget), ported from the legacy trajectory-keyed tracker tests.

A step is one ``policy`` call recorded on an agent; these fake agents count them directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from platoon.harness.budget import (
    AgentBudget,
    BudgetExceededError,
    DepthAwareStepBudgetTracker,
    StepBudgetTracker,
    make_tracker,
)


@dataclass
class Agent:
    agent_id: str
    parent_agent_id: str | None = None
    max_steps: int | None = None
    policy_calls: int = 0


def tree(tracker, *agents: Agent):
    for agent in agents:
        tracker.add_agent(agent)
    return {agent.agent_id: agent for agent in agents}


# --------------------------------------------------------------------------- linear (StepBudgetTracker)


def test_linear_budget_caps_a_agent_by_its_own_max_steps():
    tracker = StepBudgetTracker()
    agents = tree(tracker, Agent("root", max_steps=5))
    agents["root"].policy_calls = 3
    assert tracker.used_budget_for("root") == 3 and tracker.remaining_budget_for("root") == 2
    agents["root"].policy_calls = 5
    assert tracker.remaining_budget_for("root") == 0


def test_linear_children_draw_on_the_parent_budget_recursively():
    tracker = StepBudgetTracker()
    agents = tree(
        tracker,
        Agent("root", max_steps=20),
        Agent("child", "root", max_steps=6),
        Agent("grandchild", "child", max_steps=3),
    )
    agents["root"].policy_calls, agents["child"].policy_calls, agents["grandchild"].policy_calls = 2, 3, 3
    assert tracker.used_budget_for("root") == 8
    assert tracker.used_budget_for("child") == 6 and tracker.remaining_budget_for("child") == 0
    assert tracker.remaining_budget_for("root") == 12


def test_reserve_and_release_adjust_the_remaining_budget():
    tracker = StepBudgetTracker()
    tree(tracker, Agent("root", max_steps=10))
    assert tracker.reserve_budget("root", 6)
    assert tracker.remaining_budget_for("root") == 4
    tracker.release_budget("root", 6)
    assert tracker.remaining_budget_for("root") == 10
    tracker.release_budget("root", 6)  # never negative
    assert tracker.reserved_agent_budgets["root"] == 0


def test_a_failed_reservation_consumes_nothing_and_explains_itself():
    tracker = StepBudgetTracker()
    tree(tracker, Agent("root", max_steps=5, policy_calls=1))
    assert not tracker.reserve_budget("root", 6)
    with pytest.raises(BudgetExceededError) as error:
        tracker.reserve_budget("root", 6, raise_on_failure=True)
    assert error.value.reason == "step_budget" and "launch_subagent will automatically reserve" in error.value.guidance
    assert tracker.remaining_budget_for("root") == 4


def test_oversubscribed_children_are_blocked_by_earlier_reservations():
    tracker = StepBudgetTracker()
    tree(tracker, Agent("root", max_steps=14, policy_calls=1))
    assert tracker.reserve_budget("root", 6) and tracker.reserve_budget("root", 6)
    assert not tracker.reserve_budget("root", 6)
    assert tracker.remaining_budget_for("root") == 1


def test_unbounded_agents_and_zero_max_steps_are_infinite_as_before():
    tracker = StepBudgetTracker()
    tree(tracker, Agent("a"), Agent("b", max_steps=0))
    assert tracker.remaining_budget_for("a") == math.inf and tracker.remaining_budget_for("b") == math.inf


def test_linear_budgets_ignore_verifier_scopes_without_a_depth_limit():
    tracker = StepBudgetTracker()
    tree(tracker, Agent("root", max_steps=50))
    assert tracker.reserve_budget("root", 1, child_depth_scope="verifier_descendant")


def test_a_linear_depth_limit_is_opt_in():
    tracker = StepBudgetTracker(max_depth=1)
    tree(tracker, Agent("root", max_steps=50), Agent("child", "root", max_steps=10))
    assert tracker.reserve_budget("root", 1)
    with pytest.raises(BudgetExceededError) as error:
        tracker.reserve_budget("child", 1, raise_on_failure=True)
    assert error.value.reason == "depth"


def test_agent_budget_is_the_agent_view():
    tracker = StepBudgetTracker()
    tree(tracker, Agent("root", max_steps=4, policy_calls=1))
    budget = AgentBudget(tracker, "root")
    assert budget.used() == 1 and budget.remaining() == 3


# --------------------------------------------------------------------------- depth aware


def test_depth_aware_children_do_not_consume_the_parent_budget():
    tracker = DepthAwareStepBudgetTracker()
    agents = tree(tracker, Agent("root", max_steps=5), Agent("child", "root", max_steps=5))
    agents["root"].policy_calls, agents["child"].policy_calls = 2, 5
    assert tracker.used_budget_for("root") == 2 and tracker.remaining_budget_for("root") == 3
    assert tracker.remaining_budget_for("child") == 0


def test_depth_aware_reservations_always_succeed_and_release_is_a_noop():
    tracker = DepthAwareStepBudgetTracker()
    tree(tracker, Agent("root", max_steps=2, policy_calls=2))
    assert tracker.reserve_budget("root", 100)
    tracker.release_budget("root", 100)
    assert tracker.remaining_budget_for("root") == 0


@pytest.mark.parametrize(
    ("max_depth", "allowed"), [(None, [True, True, True]), (1, [True, False, False]), (2, [True, True, False])]
)
def test_depth_limits(max_depth, allowed):
    tracker = DepthAwareStepBudgetTracker(max_depth=max_depth)
    tree(tracker, Agent("d0"), Agent("d1", "d0"), Agent("d2", "d1"))
    assert [tracker.reserve_budget(agent, 1) for agent in ("d0", "d1", "d2")] == allowed


def test_depth_violation_raises_with_the_depth_reason():
    tracker = DepthAwareStepBudgetTracker(max_depth=0)
    tree(tracker, Agent("root"))
    with pytest.raises(BudgetExceededError) as error:
        tracker.reserve_budget("root", 1, raise_on_failure=True)
    assert error.value.reason == "depth" and "perform the task yourself" in error.value.guidance


def test_verifier_root_and_one_helper_are_independent_of_the_policy_depth_limit():
    tracker = DepthAwareStepBudgetTracker(max_depth=1)
    tree(tracker, Agent("root"), Agent("child", "root"), Agent("verifier", "child"), Agent("helper", "verifier"))
    assert not tracker.reserve_budget("child", 1)  # policy depth 2 is over the limit
    assert tracker.reserve_budget("child", 1, child_depth_scope="verifier_root")
    assert tracker.reserve_budget("verifier", 1, child_depth_scope="verifier_helper")
    with pytest.raises(BudgetExceededError) as error:
        tracker.reserve_budget("helper", 1, raise_on_failure=True, child_depth_scope="verifier_descendant")
    assert error.value.reason == "verifier_depth"


def test_make_tracker_and_unknown_agents():
    assert isinstance(make_tracker("linear"), StepBudgetTracker)
    assert isinstance(make_tracker("depth_aware", max_depth=2), DepthAwareStepBudgetTracker)
    with pytest.raises(ValueError):
        make_tracker("flat")  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="not registered"):
        StepBudgetTracker().remaining_budget_for("ghost")
