"""A number-guessing environment, written as an OpenReward (ORS) ``Environment`` class.

Platoon serves it in process (``service: python://guess:GuessTheNumber``): every rollout gets its own instance,
the ``@tool`` methods are the agent's tools, and a tool result with ``finished=True`` ends the episode with its
``reward``. The class never imports platoon.
"""

from __future__ import annotations

from openreward.environments import Environment, TextBlock, ToolOutput, tool
from pydantic import BaseModel, Field


class Guess(BaseModel):
    number: int = Field(description="Your guess, from 1 to 100.")


class GuessTheNumber(Environment):
    """Find a secret number from 1 to 100; each wrong guess is answered with "higher" or "lower"."""

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train", "test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[dict]:
        # 100 tasks per split. Platoon names them "<split>:<index>", so "train:41" is the task whose secret is 42.
        return [{"id": f"{split}-{n}", "secret": n} for n in range(1, 101)]

    def get_prompt(self) -> list[TextBlock]:
        return [
            TextBlock(
                text="Find the secret number from 1 to 100 with as few guesses as possible. "
                "Call the guess tool with one number at a time."
            )
        ]

    @tool
    async def guess(self, args: Guess) -> ToolOutput:
        """Guess the number; the answer says whether the secret is higher or lower."""

        secret = self.task_spec["secret"]
        if args.number == secret:
            return ToolOutput(blocks=[TextBlock(text="Correct!")], reward=1.0, finished=True)
        return ToolOutput(blocks=[TextBlock(text="higher" if args.number < secret else "lower")])
