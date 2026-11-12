"""A function harness that has the gateway enforce its step budget (``@pt.harness(budget_enforcement="gateway")``,
RFC 0009, option A): the gateway answers each of its agents' ``policy`` calls with the budget stop once
``out_of_steps`` holds for the agent, with either half (its ``max_steps``, or a linear ``ctx.budget`` that its spawned
children drew on), nothing past it is sampled, and a run that completes after a stop is graded ``limit`` with reason
``budget``, a stop of an agent it minted, whose calls its cap bounds, too; ``aux`` calls are never counted, a rollout
without a step budget is refused, and a harness that declares nothing is unchanged.  CPU only, on number search and a
scripted policy."""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import pytest

import platoon as pt
from platoon.gateway.core import BUDGET_STOP_PREFIX
from platoon.harness.agent import CALL_KIND_HEADER
from platoon.testing import ScriptedPolicy, run_episode

NUMBER_SEARCH = "python://platoon.envs.number_search.env:NumberSearchEnvironment"
# More calls than any budget here: a harness the gateway does not stop makes this many.
MAX_CALLS = 8


async def _call(ctx: pt.AgentContext, http: aiohttp.ClientSession, kind: str) -> str:
    """One call of ``kind``, over the agent's HTTP endpoint or through ``ctx.chat``, as its options say; its id."""

    messages = [{"role": "user", "content": ctx.instruction}]
    if ctx.options.get("via") == "chat":
        return (await ctx.chat(messages, kind=kind)).call_id
    endpoint = ctx.endpoint
    headers = {"Authorization": f"Bearer {endpoint.api_key}", CALL_KIND_HEADER: kind}
    body = {"model": endpoint.model, "messages": messages}
    async with http.post(f"{endpoint.base_url}/chat/completions", json=body, headers=headers) as response:
        response.raise_for_status()
        return (await response.json())["id"]


async def _call_until_stopped(ctx: pt.AgentContext) -> pt.AgentResult:
    """Calls the policy until it is told to stop, never reading its own budget: an ``aux`` call first when its options
    ask, and after ``spawn_after`` calls a child of two steps that does the same."""

    async with aiohttp.ClientSession() as http:
        if ctx.options.get("aux"):
            await _call(ctx, http, "aux")
        for calls in range(MAX_CALLS):
            if calls == ctx.options.get("spawn_after"):
                await ctx.agents.spawn("Help.", max_steps=2, options={"via": ctx.options.get("via")})
            if (await _call(ctx, http, "policy")).startswith(BUDGET_STOP_PREFIX):
                break
    return pt.AgentResult("completed")


@pt.harness("budgettest-enforced", budget_enforcement="gateway")
async def enforced(ctx: pt.AgentContext) -> pt.AgentResult:
    return await _call_until_stopped(ctx)


@pt.harness("budgettest-cooperative")
async def cooperative(ctx: pt.AgentContext) -> pt.AgentResult:
    return await _call_until_stopped(ctx)


@pt.harness("budgettest-minting", budget_enforcement="gateway")
async def minting(ctx: pt.AgentContext) -> pt.AgentResult:
    """Makes no call of its own: mints an agent over HTTP, which calls the policy until it is told to stop."""

    endpoint = ctx.endpoint
    async with aiohttp.ClientSession() as http:
        headers = {"Authorization": f"Bearer {endpoint.api_key}"}
        async with http.post(f"{endpoint.base_url}/platoon/agents", json={"role": "helper"}, headers=headers) as made:
            made.raise_for_status()
            minted = await made.json()
        headers = {"Authorization": f"Bearer {minted['api_key']}"}
        body = {"model": endpoint.model, "messages": [{"role": "user", "content": ctx.instruction}]}
        for _ in range(MAX_CALLS):
            async with http.post(f"{minted['base_url']}/chat/completions", json=body, headers=headers) as response:
                response.raise_for_status()
                if (await response.json())["id"].startswith(BUDGET_STOP_PREFIX):
                    break
    return pt.AgentResult("completed")


def episode(harness: str, *, max_steps: int | None, agents: dict[str, Any] | None = None, **options: Any):
    policy = ScriptedPolicy(lambda prompt: "Thinking.")
    run = asyncio.run(
        run_episode(
            "train:0",
            service=NUMBER_SEARCH,
            harness=harness,
            policy=policy,
            harness_options=options,
            agents=agents,
            max_steps=max_steps,
        )
    )
    return run, policy


def policy_steps(trajectory) -> int:
    return sum(1 for step in trajectory.steps if step.misc["action_misc"].get("kind", "policy") == "policy")


@pytest.mark.parametrize("via", ["http", "chat"])
def test_a_declaring_harness_is_stopped_at_out_of_steps(via):
    run, policy = episode("budgettest-enforced", max_steps=2, via=via, aux=True)

    (step_aux, *steps) = run.root.steps
    assert step_aux.misc["action_misc"]["kind"] == "aux"  # never counted
    assert (len(steps), len(policy.calls)) == (2, 3)  # nothing past the cap was sampled
    assert run.root.misc["budget"] == {"enforce": "gateway", "max_steps": 2, "stops": 1}
    assert run.root.misc["agent_result"]["status"] == "limit"
    assert (run.outcome["status"], run.outcome["reason"]) == ("scored", "budget")


def test_a_linear_budget_its_children_drew_on_stops_a_declaring_harness():
    """The other half of ``out_of_steps``: a parent of five steps spends one, spawns a child of two, which reserves
    three of its steps until it returns, and then has two left, its child's two steps being its own under
    ``agents.budget: linear``; its ``max_steps`` alone would have left it four."""

    run, policy = episode(
        "budgettest-enforced", max_steps=5, agents={"budget": "linear", "max_depth": 1}, spawn_after=1
    )

    (child,) = run.children
    assert (policy_steps(run.root), policy_steps(child)) == (3, 2)
    assert len(policy.calls) == 5
    assert child.misc["budget"] == {"enforce": "gateway", "max_steps": 2, "stops": 1}
    assert [trajectory.misc["agent_result"]["status"] for trajectory in (run.root, child)] == ["limit", "limit"]
    assert run.outcome["reason"] == "budget"


def test_a_stop_of_an_agent_it_minted_grades_a_declaring_harness_limit():
    """The agents a declaring harness's agent mints share its cap: once the gateway stops one of them, the agent's run,
    which made no call of its own, is a ``limit`` run, graded with reason ``budget``."""

    run, policy = episode("budgettest-minting", max_steps=2)

    (minted,) = run.children
    assert (policy_steps(run.root), policy_steps(minted), len(policy.calls)) == (0, 2, 2)
    assert minted.misc["budget"]["stops"] == 1 and run.root.misc["budget"]["stops"] == 0
    assert run.root.misc["agent_result"]["status"] == "limit"
    assert (run.outcome["status"], run.outcome["reason"]) == ("scored", "budget")


def test_a_declaring_harness_without_a_step_budget_is_refused():
    """Neither the rollout nor the task sets a step budget: the gateway would have nothing to hold the harness to, so
    the rollout is a configuration error, as a spec's is, and nothing is sampled."""

    run, policy = episode("budgettest-enforced", max_steps=None)

    assert (run.outcome["status"], run.outcome["failure_kind"]) == ("error", "harness_config")
    assert "has the gateway enforce its step budget, but the run sets no step budget" in run.outcome["reason"]
    assert policy.calls == []


def test_a_cooperative_harness_is_unchanged():
    run, policy = episode("budgettest-cooperative", max_steps=2, aux=True)

    assert (policy_steps(run.root), len(policy.calls)) == (MAX_CALLS, MAX_CALLS + 1)  # nothing stopped it
    assert "budget" not in run.root.misc
    assert run.root.misc["agent_result"]["status"] == "completed"
    assert run.outcome.get("reason") != "budget"


def test_harness_show_says_who_enforces_the_step_budget(capsys):
    from platoon import cli

    cli.main(["harness", "show", "budgettest-enforced"])

    assert "step budget: enforced by the gateway" in capsys.readouterr().out.splitlines()
