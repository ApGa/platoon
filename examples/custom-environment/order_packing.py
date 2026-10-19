"""Order packing: an ORS (OpenReward) environment that platoon serves with ``python://order_packing:OrderPackingEnv``.

The agent moves items from a shelf into a box until the box holds exactly the order.  It shows the parts of an
environment platoon uses: splits and task specs, a prompt, tools with pydantic parameters, a read-only tool, a
terminal tool that grades the episode, and ``platoon_fork``, which gives a subagent its own goal on the shared
shelf and box.  It needs only the openreward SDK, and runs on 0.1.125 and later.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from openreward.environments import Environment, Split, TextBlock, ToolOutput, tool
from pydantic import BaseModel, Field

SHELF = {"apple": 6, "bread": 2, "cheese": 3, "egg": 12, "milk": 2}
ORDERS = {
    "train": [
        {"apple": 2, "bread": 1},
        {"egg": 6, "milk": 1},
        {"cheese": 1, "apple": 3},
        {"bread": 2, "egg": 2, "milk": 1},
    ],
    "test": [
        {"apple": 1, "cheese": 2},
        {"egg": 4, "bread": 1, "apple": 2},
    ],
}


def describe(items: Mapping[str, int]) -> str:
    """``{"apple": 2, "bread": 1}`` as ``2 apple, 1 bread``."""

    return ", ".join(f"{count} {item}" for item, count in items.items() if count) or "nothing"


def parse_items(text: str) -> dict[str, int]:
    """The ``<count> <item>`` pairs of a goal, e.g. ``Pack 2 apple and 1 bread``."""

    return {item: int(count) for count, item in re.findall(r"(\d+)\s+([a-z]+)", text.lower())}


class PackParams(BaseModel):
    item: str = Field(description="The item to move from the shelf into the box, e.g. 'apple'.")
    quantity: int = Field(default=1, ge=1, le=12, description="How many to move.")


class DoneParams(BaseModel):
    summary: str = Field(default="", description="One line on what you packed.")


class OrderPackingEnv(Environment):
    """Pack an order from a shelf into a box (see the module docstring)."""

    # Hints platoon reads from the class (in process only; an env server sends tool names and schemas).
    platoon_terminal_tools = ("done",)  # ends the episode: runs alone, hidden from no_terminal agents
    platoon_read_only_tools = ("look",)  # never changes state: the only kind of tool read_only agents see

    def __init__(
        self,
        task_spec: Mapping[str, Any] | None = None,
        secrets: Mapping[str, str] | None = None,
        *,
        shelf: dict[str, int] | None = None,
        box: dict[str, int] | None = None,
    ) -> None:
        super().__init__(dict(task_spec or {}), dict(secrets or {}))
        self.order: dict[str, int] = dict(self.task_spec.get("order") or {})
        self.subtask = shelf is not None  # a platoon_fork child, packing into its parent's box
        self.shelf = shelf if shelf is not None else dict(SHELF)
        self.box = box if box is not None else {}
        self.start = dict(self.box)  # what the box held when this agent started

    # ------------------------------------------------------------------ tasks

    @classmethod
    def list_splits(cls) -> Sequence[Split | str]:
        return [Split(name="train", type="train"), Split(name="test", type="test")]

    @classmethod
    def list_tasks(cls, split: str) -> Sequence[dict[str, Any]]:
        if split not in ORDERS:
            raise ValueError(f"Unknown split {split!r}; known: {sorted(ORDERS)}")
        return [{"id": f"{split}-{index}", "order": order} for index, order in enumerate(ORDERS[split])]

    def get_prompt(self) -> list[TextBlock]:
        return [
            TextBlock(
                text=(
                    f"Pack this order into the box: {describe(self.order)}.\n"
                    f"The shelf holds {describe(self.shelf)}. Move items with the pack tool. "
                    "When the box holds exactly the order, reply with a one-line summary to finish."
                )
            )
        ]

    # ------------------------------------------------------------------ subagents

    def platoon_fork(self, task_spec: Mapping[str, Any]) -> OrderPackingEnv:
        """A subagent's env: the goal's items as its order, on this env's shelf and box (shared, not copied).

        Platoon passes this env's task spec with the subagent's ``goal``.  Raising refuses the subgoal: the
        parent's spawn call then returns a failed launch with this message.
        """

        goal = str(task_spec.get("goal") or "")
        order = parse_items(goal)
        if not order:
            raise ValueError(f"a subgoal names the items to pack, like 'Pack 2 apple'; got {goal!r}")
        return OrderPackingEnv({"id": task_spec.get("id"), "order": order}, shelf=self.shelf, box=self.box)

    # ------------------------------------------------------------------ tools

    @tool
    async def pack(self, params: PackParams) -> ToolOutput:
        """Move items from the shelf into the box."""

        item, quantity = params.item.strip().lower(), params.quantity
        left = self.shelf.get(item, 0)
        if left < quantity:
            # A mistake the policy made: say so in the result instead of raising (see the tutorial).
            return ToolOutput(blocks=[TextBlock(text=f"Cannot pack {quantity} {item}: the shelf has {left}.")])
        self.shelf[item] = left - quantity
        self.box[item] = self.box.get(item, 0) + quantity
        return ToolOutput(blocks=[TextBlock(text=f"Packed {quantity} {item}. The box holds {describe(self.box)}.")])

    @tool
    async def look(self) -> ToolOutput:
        """Show what is left on the shelf and what is in the box."""

        return ToolOutput(blocks=[TextBlock(text=f"Shelf: {describe(self.shelf)}.\nBox: {describe(self.box)}.")])

    @tool
    async def done(self, params: DoneParams) -> ToolOutput:
        """Finish packing: the box is checked against the order, and the episode ends."""

        packed = {item: self.box.get(item, 0) - self.start.get(item, 0) for item in self.box}
        missing = {item: n - packed.get(item, 0) for item, n in self.order.items() if packed.get(item, 0) < n}
        extra = {item: n - self.order.get(item, 0) for item, n in packed.items() if n > self.order.get(item, 0)}
        if self.subtask:  # a subagent answers only for its own items: its siblings pack into the same box
            extra = {}
        reward = 0.0 if missing or extra else 1.0
        problems = [f"missing {describe(missing)}"] if missing else []
        problems += [f"not in the order: {describe(extra)}"] if extra else []
        verdict = "The order is packed." if reward else f"Wrong box: {'; '.join(problems)}."
        return ToolOutput(
            blocks=[TextBlock(text=verdict)],
            reward=reward,
            finished=True,
            metadata={"missing": missing, "extra": extra},
        )
