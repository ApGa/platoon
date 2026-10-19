"""``example-packing-team``: a tool-calling agent that can hand parts of its task to subagents.

Importing this module registers the harness (configs import it with ``package: packing_team``).  The root agent
gets the env's tools plus a ``delegate(goal)`` tool, which runs ``goal`` in a child agent through
``ctx.agents.spawn``; children get the env's tools only.  Each child runs on a fork of the root's env session:
with ``OrderPackingEnv``, ``platoon_fork`` gives it its own order on the shared shelf and box, so its own
``done`` grades its part.  A final answer without a tool call goes to the env's terminal tool, as in the
built-in ``platoon-react`` harness.
"""

from __future__ import annotations

import json
from typing import Any

import platoon as pt
from platoon.harness.agent import terminal_arguments

DELEGATE = {
    "type": "function",
    "function": {
        "name": "delegate",
        "description": "Hand part of the task to a helper agent, e.g. 'Pack 2 apple'. Returns the helper's report.",
        "parameters": {"type": "object", "properties": {"goal": {"type": "string"}}, "required": ["goal"]},
    },
}


def as_openai_tool(spec: pt.ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": spec.name, "description": spec.description, "parameters": spec.input_schema},
    }


@pt.harness("example-packing-team")
async def packing_team(ctx: pt.AgentContext) -> pt.AgentResult:
    env_tools = await ctx.list_tools()  # filtered by this agent's tool access
    finish = next((spec for spec in env_tools if spec.terminal), None)
    tools = [as_openai_tool(spec) for spec in env_tools if spec is not finish]
    if ctx.depth == 0:
        tools.append(DELEGATE)  # only the root delegates
    messages: list[dict[str, Any]] = [{"role": "user", "content": ctx.instruction}]
    while ctx.budget.remaining() > 0:
        reply = await ctx.chat(messages, tools=tools)
        messages.append({key: value for key, value in reply.message.items() if value is not None})
        if not reply.tool_calls:  # the final answer ends the agent; the env's terminal tool grades it
            if finish is not None:
                await ctx.call_tool(finish.name, terminal_arguments(finish, reply.text))
            return pt.AgentResult(final_message=reply.text)
        for call in reply.tool_calls:
            name, arguments = call["function"]["name"], json.loads(call["function"]["arguments"] or "{}")
            if name == "delegate":
                child = await ctx.agents.spawn(str(arguments.get("goal", "")))
                text = child.message
            else:
                result = await ctx.call_tool(name, arguments)
                text = result.as_text()
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": text})
    return pt.AgentResult("limit")
