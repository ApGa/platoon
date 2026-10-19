"""``parity-probe``: the mode-neutral function harness of the parity tests.

It builds the messages ``gym.loop`` builds: the instruction as a user message, then per turn the model's reply and
the observation as a user message.  The model never sees a tool listing, so its prompts carry nothing a mode chooses
for it (an ORS env's own tools, a Harbor task's generic ones, a gym env's ``act``).  It acts through the adapter that
``harness_options`` names, the one thing that changes between modes:

* ``ors``: the number-search environment's own ``guess(number)`` tool;
* ``act``: the ``act`` tool of a gym step environment (``pt.gym.as_environment``);
* ``bash``: a Harbor task's ``bash`` tool, running the task's ``guess`` script.

The protocol (``parity_policy`` speaks it): ``GUESS <n>`` calls the adapter and ends the agent when the result says
the episode finished or the guess was right; ``DELEGATE <goal>`` spawns a subagent running this harness, forked as
the environment's ``agents`` settings say, and shows its answer; ``REPORT <text>`` ends a subagent with that answer;
``GIVE UP`` ends the agent.  The observation of a tool call is its text, stripped.  The probe also records the tools
it was given in its trajectory's misc (``probe.listing``), so a listing that differs between modes shows up in the
comparison even though the model never sees it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import platoon as pt
from platoon.harness.agent import LIMIT_MESSAGE, out_of_steps

PROBE = "parity-probe"

# adapter -> (guess -> (tool name, arguments))
ADAPTERS: dict[str, Callable[[int], tuple[str, dict[str, Any]]]] = {
    "ors": lambda number: ("guess", {"number": number}),
    "act": lambda number: ("act", {"action": f"GUESS {number}"}),
    "bash": lambda number: ("bash", {"command": f"guess {number}"}),
}


@pt.harness(PROBE)
async def parity_probe(ctx: pt.AgentContext) -> pt.AgentResult:
    """Play number search through the adapter ``harness_options`` names (see the module docstring)."""

    adapter = ADAPTERS[ctx.options["adapter"]]
    ctx.annotate(
        probe={
            "listing": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                    "terminal": tool.terminal,
                    "read_only": tool.read_only,
                    "meta": dict(tool.meta or {}),
                }
                for tool in await ctx.list_tools()
            ]
        }
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": ctx.instruction}]
    steps = 0
    while True:
        if out_of_steps(ctx, steps):
            return pt.AgentResult("limit", error=LIMIT_MESSAGE)
        reply = await ctx.chat(messages)
        steps += 1
        messages.append({key: value for key, value in reply.message.items() if value is not None})
        verb, _, argument = reply.text.strip().partition(" ")
        if verb == "GUESS":
            result = await ctx.call_tool(*adapter(int(argument)))
            observation = result.as_text().strip()
            if result.finished or "correctly" in observation:
                return pt.AgentResult("completed", final_message=observation)
        elif verb == "DELEGATE":
            child = await ctx.agents.spawn(argument)
            observation = f"Subagent ({child.result.status}): {child.message}"
        elif verb == "REPORT":
            return pt.AgentResult("completed", final_message=argument)
        elif reply.text.strip() == "GIVE UP":
            return pt.AgentResult("completed", final_message="GIVE UP")
        else:
            observation = f"Unknown action: {reply.text.strip()}"
        messages.append({"role": "user", "content": observation})
