"""Number search as an ORS environment: guess a hidden integer, told "too low" or "too high" after each guess.

Served in process with ``service: python://platoon.envs.number_search.env:NumberSearchEnvironment``, or named as
``env: platoon-number-search@1``.  Its tasks are generated from seed 42 the first time a process lists them
(``tasks.py``): task specs with ``goal``, ``id``, ``max_steps`` and ``misc.target``.  The ``guess`` tool ends the
episode with reward 1 on the right number; an episode that never gets there is graded 0 (``no_submit``).

Written against openreward 0.1.125's API (``@tool`` with one pydantic parameter model,
``ToolOutput(finished=True)``), which 0.1.163 still serves.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any

from openreward.environments import Environment, Split, TextBlock, ToolOutput, tool
from pydantic import BaseModel, Field

from platoon.envs.number_search.tasks import generate

SPLITS = ("train", "val")


@functools.cache
def _generated() -> dict[str, tuple[dict[str, Any], ...]]:
    return {split: tuple(specs) for split, specs in generate().items()}


def task_specs(split: str) -> tuple[dict[str, Any], ...]:
    """The task specs of ``split`` (``train`` or ``val``), generated once per process."""

    if split not in SPLITS:
        raise ValueError(f"Unknown number-search split {split!r}; known: {list(SPLITS)}")
    return _generated()[split]


class GuessParams(BaseModel):
    number: int = Field(description="The number to guess.")


class NumberSearchEnvironment(Environment):
    """One number-search task: the target is in ``task_spec["misc"]["target"]``."""

    def __init__(self, task_spec: dict[str, Any] | None = None, secrets: dict[str, str] | None = None) -> None:
        task_spec = dict(task_spec or {})
        super().__init__(task_spec, secrets or {})
        misc = task_spec.get("misc") or {}
        self.target = int(misc.get("target", task_spec.get("target", 0)))
        self.goal = str(task_spec.get("goal") or "Guess the correct number.")

    @classmethod
    def list_splits(cls) -> Sequence[Split | str]:
        return [Split(name="train", type="train"), Split(name="val", type="validation")]

    @classmethod
    def list_tasks(cls, split: str) -> Sequence[dict[str, Any]]:
        return list(task_specs(split))

    def get_prompt(self) -> list[TextBlock]:
        return [TextBlock(text=self.goal)]

    @tool
    async def guess(self, params: GuessParams) -> ToolOutput:
        """Guess the hidden number; the reply says whether it is too low, too high or correct."""

        if params.number == self.target:
            text = f"You guessed the number {self.target} correctly!"
            return ToolOutput(blocks=[TextBlock(text=text)], reward=1.0, finished=True)
        text = "Too low, try again." if params.number < self.target else "Too high, try again."
        return ToolOutput(blocks=[TextBlock(text=text)])
