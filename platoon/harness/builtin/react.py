"""The ``platoon-react`` function harness: a minimal tool-calling loop over ``ctx.chat``.

The env's tools go to the model as OpenAI tools, and every tool call it makes is answered
through ``ctx.call_tool`` (:func:`answer_tool_calls`). The agent ends when a tool reports the
episode finished, or when the model answers without calling a tool: that final answer goes
to the env's terminal tool when it has one (which is kept out of the model's tool list, the
way the openreward SDK treats ``@terminal`` tools). It ends with ``limit`` and the
"Exhausted budget" warning (``LIMIT_MESSAGE``) when its budget or ``ctx.max_steps`` runs out.

Options (``harness_options`` in the environment config; unknown keys are an error):
``system_prompt`` (a system message before the instruction) and ``sampling`` (Chat
Completions parameters such as ``max_completion_tokens``, which override the rollout's
default sampling, its ``inference_params``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

from platoon.harness.agent import (
    LIMIT_MESSAGE,
    AgentContext,
    AgentResult,
    ChatReply,
    out_of_steps,
    terminal_arguments,
    terminal_tool,
)
from platoon.harness.function import harness

if TYPE_CHECKING:
    from platoon.envs.service import ToolResult, ToolSpec


@dataclass(frozen=True)
class ReactOptions:
    """The ``platoon-react`` harness options (see the module docstring); unknown keys are an error."""

    system_prompt: str | None = None
    sampling: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any] | None) -> ReactOptions:
        options = dict(options or {})
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(options) - known)
        if unknown:
            raise ValueError(f"Unknown platoon-react harness options {unknown}; known: {sorted(known)}")
        options["sampling"] = dict(options.get("sampling") or {})
        return cls(**options)


def openai_tool(tool: ToolSpec) -> dict[str, Any]:
    """An env tool as an OpenAI Chat Completions function tool."""

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema or {"type": "object", "properties": {}},
        },
    }


async def answer_tool_calls(ctx: AgentContext, reply: ChatReply) -> tuple[list[dict[str, Any]], ToolResult | None]:
    """Answer ``reply``'s tool calls in order through ``ctx.call_tool``.

    Returns the ``tool`` messages that answer them and the result that reported the episode finished, if one did;
    the calls after that one are not made. A call whose arguments are not a JSON object is answered with
    "Invalid arguments: ..." and never reaches the env.
    """

    messages: list[dict[str, Any]] = []
    for call in reply.tool_calls:
        function = call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
            if not isinstance(arguments, Mapping):
                raise ValueError("tool arguments must be a JSON object")
        except ValueError as exc:
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": f"Invalid arguments: {exc}"})
            continue
        result = await ctx.call_tool(str(function.get("name", "")), dict(arguments))
        messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": result.as_text()})
        if result.finished:
            return messages, result
    return messages, None


@harness("platoon-react")
async def react(ctx: AgentContext) -> AgentResult:
    """One tool-calling agent: the env's tools go to the model as OpenAI tools, and each tool call is answered with
    the tool's result.

    A reply without tool calls is the final answer, which goes to the env's terminal tool when it has one (that tool
    is kept out of the model's tool list).  A tool that reports the episode finished also ends the agent, and the
    step budget (``ctx.max_steps``) ends it with status ``limit``.

    Options (``harness_options``; unknown keys are an error):

    - ``system_prompt``: a system message before the instruction.
    - ``sampling``: Chat Completions parameters such as ``max_completion_tokens``, which override the rollout's
      default sampling (its ``inference_params``).
    """

    options = ReactOptions.from_mapping(ctx.options)
    tools = await ctx.list_tools()
    terminal = terminal_tool(tools)
    openai_tools = [openai_tool(tool) for tool in tools if tool is not terminal]
    messages: list[dict[str, Any]] = []
    if options.system_prompt:
        messages.append({"role": "system", "content": str(options.system_prompt)})
    messages.append({"role": "user", "content": ctx.instruction})

    steps = 0
    while True:
        if out_of_steps(ctx, steps):
            return AgentResult("limit", error=LIMIT_MESSAGE)
        reply = await ctx.chat(messages, tools=openai_tools or None, **options.sampling)
        steps += 1
        messages.append({key: value for key, value in reply.message.items() if value is not None})
        if not reply.tool_calls:  # the final answer
            if terminal is not None:
                await ctx.call_tool(terminal.name, terminal_arguments(terminal, reply.text))
            return AgentResult("completed", final_message=reply.text)
        answers, finished = await answer_tool_calls(ctx, reply)
        messages.extend(answers)
        if finished is not None:
            return AgentResult("completed", final_message=finished.as_text())


__all__ = ["ReactOptions", "answer_tool_calls", "openai_tool", "react"]
