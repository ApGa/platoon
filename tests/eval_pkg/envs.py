"""An ORS environment that rewards a submit."""

from __future__ import annotations

from openreward.environments import Environment, TextBlock, ToolOutput, tool


class OneShotEnv(Environment):
    """Two tasks, each solved by calling ``submit``."""

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[dict]:
        return [{"name": "one"}, {"name": "two"}]

    def get_prompt(self) -> list[TextBlock]:
        return [TextBlock(text="Submit.")]

    @tool
    async def submit(self) -> ToolOutput:
        """Submit; this ends the episode."""
        return ToolOutput(blocks=[TextBlock(text="submitted")], reward=1.0, finished=True)
