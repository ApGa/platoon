"""Vault: add up the numbers stored under some keys. An ORS environment made for delegation.

Every task stores a number under each of twelve keys and asks for the sum under four to eight of them.
``read(key)`` returns one value; ``answer(total)`` ends the episode, with reward 1 when the total is right.

``platoon_fork`` is what makes the tasks split: a subagent forked in ``shared`` mode gets an environment of its
own on the same vault, for the keys its goal names, so its own ``answer`` grades its part of the work
(``agents.child_grading: env``).

Served in process: ``service: python://vault_env:VaultEnvironment`` (with this directory on ``PYTHONPATH``).
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence
from typing import Any

from openreward.environments import Environment, Split, TextBlock, ToolOutput, tool
from pydantic import BaseModel, Field

KEYS = ("amber", "basil", "cedar", "delta", "ember", "fjord", "grove", "haven", "ivory", "jade", "kelp", "lumen")
SPLIT_SIZES = {"train": 200, "test": 50}
GOAL = "Report the sum of the values stored under the keys {keys}."
_GOAL_KEYS = re.compile(r"under the keys ([a-z, ]+)")


def goal_for(keys: Sequence[str]) -> str:
    """The goal text of a (sub)task over ``keys``; subagent goals use the same wording."""

    return GOAL.format(keys=", ".join(keys))


def parse_keys(goal: str | None) -> list[str]:
    """The keys a goal names (``[]`` when it names none)."""

    match = _GOAL_KEYS.search(goal or "")
    return [key.strip() for key in match.group(1).split(",") if key.strip()] if match else []


class ReadParams(BaseModel):
    key: str = Field(description="The key to read.")


class AnswerParams(BaseModel):
    total: int = Field(description="The sum of the values under your keys.")


class VaultEnvironment(Environment):
    """One Vault task: ``task_spec`` holds the ``vault`` ({key: value}) and the ``keys`` to add up."""

    # Hints platoon reads from the class: ``answer`` ends the episode (agents with ``no_terminal`` access do not
    # see it), and ``read`` changes nothing (agents with ``read_only`` access, such as verifiers, may call it).
    platoon_terminal_tools = ("answer",)
    platoon_read_only_tools = ("read",)

    def __init__(self, task_spec: Mapping[str, Any] | None = None, secrets: Mapping[str, str] | None = None) -> None:
        task_spec = dict(task_spec or {})
        super().__init__(task_spec, dict(secrets or {}))
        self.vault = {str(key): int(value) for key, value in dict(task_spec.get("vault") or {}).items()}
        self.keys = [str(key) for key in task_spec.get("keys") or parse_keys(task_spec.get("goal"))]

    @classmethod
    def list_splits(cls) -> Sequence[Split | str]:
        return [Split(name="train", type="train"), Split(name="test", type="test")]

    @classmethod
    def list_tasks(cls, split: str) -> Sequence[dict[str, Any]]:
        if split not in SPLIT_SIZES:
            raise ValueError(f"Unknown Vault split {split!r}; known: {sorted(SPLIT_SIZES)}")
        tasks = []
        for index in range(SPLIT_SIZES[split]):
            rng = random.Random(f"{split}-{index}")
            vault = {key: rng.randint(1, 99) for key in KEYS}
            keys = rng.sample(KEYS, rng.randint(4, 8))
            tasks.append({"id": f"{split}-{index}", "vault": vault, "keys": keys})
        return tasks

    def get_prompt(self) -> list[TextBlock]:
        return [TextBlock(text=goal_for(self.keys))]

    def platoon_fork(self, task_spec: Mapping[str, Any]) -> VaultEnvironment:
        """A subagent's environment: the same vault, and the keys its goal names.

        ``task_spec`` is the parent's task spec with the subagent's ``goal``, so the keys come from the goal and
        never from ``task_spec["keys"]``, which still lists the parent's.
        """

        return type(self)({"vault": self.vault, "keys": parse_keys(task_spec.get("goal"))})

    @tool
    async def read(self, params: ReadParams) -> ToolOutput:
        """Read the value stored under a key."""

        if params.key not in self.vault:
            return _text(f"There is no key {params.key!r}. The keys are: {', '.join(sorted(self.vault))}.")
        return _text(str(self.vault[params.key]))

    @tool
    async def answer(self, params: AnswerParams) -> ToolOutput:
        """Submit the sum of the values under your keys. This ends your task."""

        # A goal that names no key, or a key the vault does not have, cannot be answered right.
        known = bool(self.keys) and all(key in self.vault for key in self.keys)
        correct = known and params.total == sum(self.vault[key] for key in self.keys)
        return ToolOutput(
            blocks=[TextBlock(text=f"Answered {params.total}.")], reward=1.0 if correct else 0.0, finished=True
        )


def _text(text: str) -> ToolOutput:
    return ToolOutput(blocks=[TextBlock(text=text)])
