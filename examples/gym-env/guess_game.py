"""Hi-lo, a step environment, served to platoon two ways (``pt.gym``):

* ``example-hilo-loop``: ``pt.gym.loop`` makes the game a function harness.  Each model reply is an action, each
  observation the next user message, and the ``gym`` program scores the episode with the return.  Tasks come
  from a task table (``service: table``).
* ``HiLoEnvironment``: ``pt.gym.as_environment`` wraps the game as an ORS environment with one ``act`` tool,
  served with ``python://guess_game:HiLoEnvironment``, so any harness plays it through a tool call.

Importing this module registers the harness (configs import it with ``package: guess_game``).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import platoon as pt

TASKS = Path(__file__).with_name("tasks.jsonl")
MAX_GUESSES = 7  # enough for a binary search of 1..100


class HiLoGame:
    """Guess the task's ``target`` from 1 to 100; the reply says whether it is higher or lower."""

    async def reset(self, task: pt.Task) -> str:
        self.target = int(task.misc["target"])
        return (
            f"I am thinking of a number from 1 to 100. Guess it in at most {MAX_GUESSES} tries; "
            "reply with just the number."
        )

    async def step(self, action: str) -> tuple[str, float, bool]:
        """One guess: ``(observation, reward, done)``."""

        try:
            guess = int(action.strip())
        except ValueError:
            return "Reply with a single whole number.", 0.0, False
        if guess == self.target:
            return f"Correct! The number was {self.target}.", 1.0, True
        return ("Too low." if guess < self.target else "Too high."), 0.0, False

    async def close(self) -> None:
        """Called when the episode ends (optional)."""


def tasks_by_split(path: Path = TASKS) -> dict[str, list[dict[str, Any]]]:
    """The task table's rows by split: the task specs ``as_environment`` serves."""

    splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            splits[row.get("split", "train")].append(row)
    return dict(splits)


# 1. The game as a function harness, scored by the `gym` program.
pt.harness("example-hilo-loop")(
    pt.gym.loop(HiLoGame, max_turns=MAX_GUESSES, system="You are playing a guessing game. Reply with one number.")
)

# 2. The game as an ORS environment with one `act` tool, for any harness.
HiLoEnvironment = pt.gym.as_environment(HiLoGame, name="HiLoEnvironment", tasks=tasks_by_split())
