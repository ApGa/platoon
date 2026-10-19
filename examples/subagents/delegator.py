"""``example-delegator``: a tool-calling agent that can hand parts of its task to subagents.

The model sees the environment's tools plus ``delegate(goal, max_steps)``. A ``delegate`` call runs
``ctx.agents.spawn``: a child agent (by default this same harness) works on the goal in a fork of the
environment, and its final message comes back as the tool result. The calls of one reply run concurrently, so
several ``delegate`` calls in one reply run their subagents in parallel.

Import this module (``package: delegator``) to register the harness.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

import platoon as pt
from platoon.harness.agent import LIMIT_MESSAGE, out_of_steps
from platoon.harness.builtin.react import openai_tool

DELEGATE_TOOL = {
    "type": "function",
    "function": {
        "name": "delegate",
        "description": (
            "Hand a self-contained subtask to a subagent. It works on the goal on its own, with its own step "
            "budget, and you get its final answer back. Several delegate calls in one reply run in parallel."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "What the subagent should do."},
                "max_steps": {"type": "integer", "description": "The subagent's step budget (optional)."},
            },
            "required": ["goal"],
        },
    },
}


@pt.harness("example-delegator")
async def delegator(ctx: pt.AgentContext) -> pt.AgentResult:
    """Call tools until a tool ends the episode or the model answers in plain text; ``delegate`` spawns."""

    tools = [openai_tool(tool) for tool in await ctx.list_tools()] + [DELEGATE_TOOL]
    messages: list[dict[str, Any]] = [{"role": "user", "content": ctx.instruction}]
    steps = 0
    while not out_of_steps(ctx, steps):
        reply = await ctx.chat(messages, tools=tools)
        steps += 1
        messages.append({key: value for key, value in reply.message.items() if value is not None})
        if not reply.tool_calls:
            return pt.AgentResult("completed", final_message=reply.text)
        results = await asyncio.gather(*(run_call(ctx, call) for call in reply.tool_calls))
        for call, (text, _) in zip(reply.tool_calls, results):
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": text})
        finished = next((text for text, done in results if done), None)
        if finished is not None:
            return pt.AgentResult("completed", final_message=finished)
    return pt.AgentResult("limit", error=LIMIT_MESSAGE)


async def run_call(ctx: pt.AgentContext, call: Mapping[str, Any]) -> tuple[str, bool]:
    """Run one tool call: ``(text for the model, whether it ended the episode)``."""

    function = call.get("function") or {}
    try:
        arguments = json.loads(function.get("arguments") or "{}")
        if not isinstance(arguments, dict):
            raise ValueError("the arguments must be a JSON object")
    except ValueError as exc:
        return f"Invalid arguments: {exc}", False
    if function.get("name") == "delegate":
        child = await ctx.agents.spawn(str(arguments.get("goal", "")), max_steps=arguments.get("max_steps"))
        # A child never raises into its parent: a rejected launch or a failed child comes back as a message.
        if child.ok:
            return child.message, False
        return f"The subagent did not finish ({child.result.status}): {child.message}", False
    result = await ctx.call_tool(str(function.get("name", "")), arguments)
    return result.as_text(), result.finished
