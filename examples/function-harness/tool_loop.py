"""``example-tool-loop``: a minimal function harness that plays any environment through its tools.

The model sees the environment's tools as OpenAI function tools. Each turn is one model call; every tool call
in the reply is answered through the environment, and the agent ends when a tool result says the episode is
over, when the model answers without calling a tool, or when the step budget runs out.

Options (``harness_options`` in the environment config):

* ``system_prompt``: a system message to put before the instruction.
* ``sampling``: Chat Completions parameters (``temperature``, ``max_completion_tokens``, ...) for every call;
  they override the rollout's ``inference_params``.
"""

from __future__ import annotations

import json
from typing import Any

import platoon as pt
from platoon.harness.agent import LIMIT_MESSAGE


@pt.harness("example-tool-loop")
async def tool_loop(ctx: pt.AgentContext) -> pt.AgentResult:
    """Call the environment's tools until it ends the episode, the model answers, or the budget runs out."""

    # The env's tools, already filtered by this agent's tool access.
    specs = await ctx.list_tools()
    if not specs:
        return pt.AgentResult("error", error="example-tool-loop needs an environment with tools")
    tools = [
        {
            "type": "function",
            "function": {"name": spec.name, "description": spec.description, "parameters": spec.input_schema},
        }
        for spec in specs
    ]

    messages: list[dict[str, Any]] = []
    if ctx.options.get("system_prompt"):
        messages.append({"role": "system", "content": ctx.options["system_prompt"]})
    messages.append({"role": "user", "content": ctx.instruction})
    sampling = dict(ctx.options.get("sampling") or {})

    while True:
        # One step is one policy call. Check the budget before each one: platoon does not stop the loop for you.
        if ctx.budget.remaining() <= 0:
            return pt.AgentResult("limit", error=LIMIT_MESSAGE)
        reply = await ctx.chat(messages, tools=tools, **sampling)
        messages.append({key: value for key, value in reply.message.items() if value is not None})

        if not reply.tool_calls:  # an answer without a tool call ends the agent
            ctx.annotate_step(observations=[])
            return pt.AgentResult("completed", final_message=reply.text)

        observations = []
        for call in reply.tool_calls:
            name = call["function"]["name"]
            try:
                arguments = json.loads(call["function"]["arguments"] or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be a JSON object")
            except ValueError as exc:  # json.JSONDecodeError is a ValueError too
                text, result = f"Invalid arguments: {exc}", None
            else:
                result = await ctx.call_tool(name, arguments)
                text = result.as_text()
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": text})
            observations.append({"tool": name, "output": text})
            if result is not None and result.finished:  # the environment ended the episode
                ctx.annotate_step(observations=observations)
                return pt.AgentResult("completed", final_message=text)
        # Annotations land on the step of the last model call: this turn's reply.
        ctx.annotate_step(observations=observations)
