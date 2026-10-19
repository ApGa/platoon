"""A TextCraft variant for the shared-inventory test (``test_textcraft_concurrency.py``).

``SyncCraftTextCraft`` runs the real crafting logic as a sync tool, which the SDK runs in a worker thread, over an
inventory whose lookups take a millisecond: what TextCraft's shared inventory meets if a craft ever runs off the
event loop.  Unordered, two crafts both pass the availability check and overdraw the inventory.
"""

from __future__ import annotations

import time
from typing import Any

from openreward.environments import ToolOutput, tool

from platoon_textcraft.env import CraftParams, TextCraftEnvironment, _text


class SlowInventory(dict):
    """An inventory whose lookups answer a millisecond late, which widens a craft's check-then-act window."""

    def get(self, key: Any, default: Any = None) -> Any:
        value = dict.get(self, key, default)
        time.sleep(0.001)
        return value


class SyncCraftTextCraft(TextCraftEnvironment):
    def __init__(self, task_spec: Any = None, secrets: Any = None) -> None:
        super().__init__(task_spec, secrets)
        self.inventory = SlowInventory(self.inventory)

    @tool
    def craft(self, params: CraftParams) -> ToolOutput:
        """Craft items from your inventory (a sync tool: it runs in a worker thread)."""
        return _text(self._craft(params.ingredients, params.target))
