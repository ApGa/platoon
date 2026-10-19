"""``example-self-check``: a function harness that answers, then checks its answer before giving it.

It suits tasks without tools, such as the rows of a task table: the first model call answers the instruction,
each check asks the model to look over its answer, and the last reply is the agent's final answer. Every model
call is one step of the agent's budget.

Options (``harness_options`` in the environment config):

* ``checks``: how many times the model checks its answer (default 1; 0 answers once).
* ``check_prompt``: the message that asks for a check.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import platoon as pt
from platoon.harness.agent import LIMIT_MESSAGE

CHECK_PROMPT = "Check your answer step by step. Then reply with the final answer only."


@dataclass(frozen=True)
class SelfCheckOptions:
    checks: int = 1
    check_prompt: str = CHECK_PROMPT

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any]) -> SelfCheckOptions:
        """The options of ``harness_options``; an unknown key or a bad value is a ``ValueError``."""

        unknown = sorted(set(options) - {field.name for field in fields(cls)})
        if unknown:
            raise ValueError(f"example-self-check has no options {unknown}")
        parsed = cls(**options)
        if isinstance(parsed.checks, bool) or not isinstance(parsed.checks, int) or parsed.checks < 0:
            raise ValueError(f"checks must be a non-negative integer, got {parsed.checks!r}")
        return parsed


@pt.harness("example-self-check")
async def self_check(ctx: pt.AgentContext) -> pt.AgentResult:
    """Answer the instruction, check the answer `checks` times, and reply with the final answer."""

    options = SelfCheckOptions.from_mapping(ctx.options)  # raising here makes the rollout an error, never a score
    messages: list[dict[str, Any]] = [{"role": "user", "content": ctx.instruction}]
    for turn in range(options.checks + 1):
        if turn:
            messages.append({"role": "user", "content": options.check_prompt})
        if ctx.budget.remaining() <= 0:  # one step per model call; platoon does not stop the loop for you
            return pt.AgentResult("limit", error=LIMIT_MESSAGE)
        reply = await ctx.chat(messages)
        messages.append({"role": "assistant", "content": reply.text})
        ctx.annotate_step(turn="answer" if turn == 0 else "check")  # shown by the trajectory viewer
    return pt.AgentResult("completed", final_message=reply.text)
