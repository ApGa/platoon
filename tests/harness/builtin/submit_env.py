"""A tiny ORS environment for the CodeAct golden scenarios (``python://submit_env:SubmitEnvironment``).

Its terminal ``submit`` replies "Submitted." whatever the answer, so a scenario can tell ``finish``'s message from the
tool's text, and ``echo`` has an optional parameter, so a scenario can leave one out.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import openreward.environments as ors
from openreward.environments import Environment, Split, TextBlock, ToolOutput, tool
from pydantic import BaseModel, Field


def _terminal(fn: Any) -> Any:
    """The SDK's ``@terminal`` where it exists (0.1.157+); older SDKs get the same marker by hand."""

    marker = getattr(ors, "terminal", None)
    if marker is not None:
        return marker(fn)
    fn._env_tool_terminal = True
    return fn


class EchoParams(BaseModel):
    text: str = Field(description="What to echo.")
    times: int = Field(default=1, description="How often.")


class SubmitParams(BaseModel):
    answer: str | None = Field(description="The answer.")


class SubmitEnvironment(Environment):
    """One question; ``submit("42")`` is right."""

    def __init__(self, task_spec: dict[str, Any] | None = None, secrets: dict[str, str] | None = None) -> None:
        task_spec = dict(task_spec or {})
        super().__init__(task_spec, secrets or {})
        self.goal = str(task_spec.get("goal") or "Answer the question.")

    @classmethod
    def list_splits(cls) -> Sequence[Split | str]:
        return [Split(name="val", type="validation")]

    @classmethod
    def list_tasks(cls, split: str) -> Sequence[dict[str, Any]]:
        return [{"id": f"t{index}", "goal": f"Answer question {index}."} for index in range(3)]

    def get_prompt(self) -> list[TextBlock]:
        return [TextBlock(text=self.goal)]

    @tool
    async def echo(self, params: EchoParams) -> ToolOutput:
        """Echo the text."""
        return ToolOutput(blocks=[TextBlock(text=params.text * params.times)])

    @_terminal
    @tool
    async def submit(self, params: SubmitParams) -> ToolOutput:
        """Submit your answer."""
        reward = 1.0 if params.answer == "42" else 0.0
        return ToolOutput(blocks=[TextBlock(text="Submitted.")], reward=reward, finished=True)
