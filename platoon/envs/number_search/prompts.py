"""The prompts of platoon's earlier number-search agent, for the ``platoon-codeact`` harness.

Select them with ``prompt_builder: platoon.envs.number_search.prompts:NumberSearchPromptBuilder`` (the named
environment ``platoon-number-search@1`` gives ``platoon-codeact`` these by default).

The system prompt asks for ``guess(number)`` in a ``<python>`` cell, and the action
space stays empty as it was, so prompts are token-for-token those of the earlier agent.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from platoon.harness.builtin.codeact import CellFunction, CodeActPromptBuilder, CodeActState


class NumberSearchPromptBuilder(CodeActPromptBuilder):
    """Number-search system prompt, with or without ``<thought>`` (``include_reasoning``)."""

    def build_system_prompt(self, state: CodeActState, **context: Any) -> str:
        include_reasoning = context.get("include_reasoning", self.include_reasoning)
        if include_reasoning:
            return """Solve step by step. Put thoughts in <thought> </thought> and code in <python> </python>.
Your answer must call guess(number: int) with the guessed number as an integer.

Example:
<thought>
thought process here
</thought>
<python>
guess(42)
</python>
"""
        else:
            return """Solve the problem step by step. Write your action in <python> </python> tags.
Your answer must call guess(number: int) with the guessed number as an integer.

Example:
<python>
guess(42)
</python>
"""

    def describe_action_space(self, functions: Sequence[CellFunction]) -> str:
        return ""
