"""Scripted agent trees on a live ORS session, for the tests of agents that share an env (``test_ors_concurrency.py``,
and the TextCraft extension's ``test_textcraft_concurrency.py``).

A root agent runs a plan from its goal through the inline runtime: ``call`` a tool, ``gather`` steps in the agent
(as a PTC gather does), ``spawn`` children with plans of their own, ``sleep``.  The agents make no model calls.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, cast

from platoon.envs.ors import OrsSession
from platoon.gateway.core import ModelGateway
from platoon.harness.agent import AgentResult
from platoon.harness.function import FunctionHarness
from platoon.harness.inline import InlineAgentContext, InlineRuntime, run_agent


def env_of(session: OrsSession) -> Any:
    """The env object a session's calls reach (``platoon_fork`` children share their parent's)."""
    return session._lease.backend.env  # type: ignore[union-attr]


class StubRenderer:
    def build_generation_prompt(self, messages, **kwargs):
        raise AssertionError("these agents make no model calls")

    def parse_response(self, tokens, **kwargs):
        raise AssertionError("these agents make no model calls")

    def get_stop_sequences(self):
        return []


def make_gateway() -> ModelGateway:
    return ModelGateway(
        sampler=cast(Any, None), renderer=cast(Any, StubRenderer()), tokenizer=cast(Any, None), model_name="policy"
    )


async def _run_steps(ctx: InlineAgentContext, steps: list[dict[str, Any]]) -> None:
    for step in steps:
        await _run_step(ctx, step)


async def _run_step(ctx: InlineAgentContext, step: dict[str, Any]) -> None:
    if "call" in step:
        result = await ctx.call_tool(step["call"], step.get("args", {}))
        ctx.runtime.results.append((step["call"], result))  # type: ignore[attr-defined]
    elif "gather" in step:
        await asyncio.gather(*(_run_step(ctx, sub) for sub in step["gather"]))
    elif "spawn" in step:
        goals = ["PLAN " + json.dumps(plan) for plan in step["spawn"]]
        runs = await asyncio.gather(*(ctx.agents.spawn(goal, fork=step.get("fork")) for goal in goals))
        ctx.runtime.spawned.extend(runs)  # type: ignore[attr-defined]
    elif "sleep" in step:
        await asyncio.sleep(step["sleep"])


async def _scripted(ctx: InlineAgentContext) -> AgentResult:
    """Runs the plan in its goal: ``call``, ``gather`` (in this agent, like a PTC gather), ``spawn``, ``sleep``."""
    await _run_steps(ctx, json.loads(ctx.instruction.removeprefix("PLAN ")))
    return AgentResult("completed", final_message="done")


SCRIPTED = FunctionHarness(name="scripted", version="1", runtime="inline", executes_model_code=False, fn=_scripted)


def call(tool: str, **args: Any) -> dict[str, Any]:
    return {"call": tool, "args": args}


async def run_tree(
    session: OrsSession,
    plan: Any,
    *,
    harness: FunctionHarness = SCRIPTED,
    prefix: str = "PLAN ",
    base_url: str | None = None,
    gateway: ModelGateway | None = None,
    timeout: float = 30.0,
    **agents: Any,
) -> SimpleNamespace:
    """Run ``plan`` on a root agent of ``session`` through the inline runtime; children spawn as the plan says."""
    gateway = gateway or make_gateway()
    settings = {"budget": "depth_aware", "max_depth": 4, "child_max_steps": 5, **agents}
    runtime = InlineRuntime(gateway=gateway, base_url=base_url, agents=settings)
    runtime.results, runtime.spawned = [], []  # type: ignore[attr-defined]
    root = gateway.open_agent(rollout_id=session.rollout_id, trajectory_id="root", interactions={}, max_steps=1000)
    ctx = InlineAgentContext(
        runtime, root, task=session.task, session=session, harness=harness, instruction=prefix + json.dumps(plan)
    )
    result = await asyncio.wait_for(run_agent(ctx), timeout)
    return SimpleNamespace(result=result, results=runtime.results, spawned=runtime.spawned, gateway=gateway)  # type: ignore[attr-defined]
