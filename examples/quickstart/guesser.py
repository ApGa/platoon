"""A function harness: a minimal tool-calling loop over the environment's tools.

Importing this module registers the harness as ``example-guesser``; the configs import it through ``package: guesser``.
"""

from __future__ import annotations

import json

import platoon as pt


@pt.harness("example-guesser")
async def guesser(ctx: pt.AgentContext) -> pt.AgentResult:
    """Call the model with the env's tools until a tool result ends the episode or the step budget runs out."""

    tools = [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
        for t in await ctx.list_tools()
    ]
    messages: list[dict] = [{"role": "user", "content": ctx.instruction}]
    while ctx.budget.remaining() > 0:
        # One policy call: sampled through the model gateway and recorded, token for token, for training.
        reply = await ctx.chat(messages, tools=tools)
        messages.append({key: value for key, value in reply.message.items() if value is not None})
        if not reply.tool_calls:
            return pt.AgentResult("completed", final_message=reply.text)
        for call in reply.tool_calls:
            arguments = json.loads(call["function"]["arguments"] or "{}")
            result = await ctx.call_tool(call["function"]["name"], arguments)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result.as_text()})
            if result.finished:  # the environment ended and graded the episode
                return pt.AgentResult("completed", final_message=result.as_text())
    return pt.AgentResult("limit")
