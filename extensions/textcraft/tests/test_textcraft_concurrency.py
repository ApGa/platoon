"""TextCraft's ``platoon_fork`` children take turns on their parent's inventory: core's shared-env gate
(``platoon.envs.gate``) orders their crafts, as ``tests/envs/test_ors_concurrency.py`` checks on toy envs.

``textcraft_toys.SyncCraftTextCraft`` runs the crafts in worker threads over a slow inventory, where unordered crafts
would overdraw it.
"""

from __future__ import annotations

import asyncio
import random
import uuid
from collections import Counter
from typing import Any

import pytest
from platoon.envs.ors import OrsEnvironmentService
from platoon.envs.ors.session import LocalBackend
from platoon.gateway.mcp import tool_call_records
from scripted_trees import call, env_of, run_tree
from textcraft_toys import SyncCraftTextCraft

from platoon_textcraft.env import TextCraftEnvironment

pytestmark = pytest.mark.asyncio


TEXTCRAFT_RECIPES = [
    ({"sugar_cane": 3}, ("paper", 3)),
    ({"rabbit_hide": 4}, ("leather", 1)),
    ({"cobblestone": 2, "quartz": 2}, ("diorite", 2)),
    ({"diorite": 1, "cobblestone": 1}, ("andesite", 2)),
    ({"diorite": 1, "quartz": 1}, ("granite", 1)),
    ({"paper": 3, "leather": 1}, ("book", 1)),
]


@pytest.mark.parametrize("variant", ["TextCraftEnvironment", "SyncCraftTextCraft"])
async def test_textcraft_children_keep_their_shared_inventory_consistent_and_finish_runs_alone(variant, monkeypatch):
    """TextCraft's platoon_fork children craft at once from the root's inventory.  Every successful craft lands
    exactly once, nothing is overdrawn, and each child's terminal ``finish`` runs alone.
    ``SyncCraftTextCraft`` crafts in worker threads over a slow inventory, where unordered crafts would overdraw."""

    environment = SyncCraftTextCraft if variant == "SyncCraftTextCraft" else TextCraftEnvironment
    active: list[str] = []
    overlaps: list[tuple[str, list[str]]] = []
    peak = 0
    original = LocalBackend.call

    async def traced(self: LocalBackend, name: str, arguments: dict[str, Any]) -> Any:
        nonlocal peak
        if (name == "finish" and active) or "finish" in active:
            overlaps.append((name, list(active)))
        active.append(name)
        peak = max(peak, len(active))
        try:
            await asyncio.sleep(0.001)  # a pause inside the call, where an unordered call would start
            return await original(self, name, arguments)
        finally:
            active.remove(name)

    monkeypatch.setattr(LocalBackend, "call", traced)
    svc = OrsEnvironmentService(environment)
    session = await svc.open(await svc.get_task("train:0"), rollout_id=f"textcraft-{uuid.uuid4().hex[:6]}")
    initial = Counter(environment.list_tasks("train")[0]["misc"]["initial_inventory"])
    rng = random.Random(11)

    def child_plan() -> list[dict[str, Any]]:
        crafts = [rng.choice(TEXTCRAFT_RECIPES) for _ in range(8)]
        steps = [call("craft", ingredients=dict(ingredients), target=list(target)) for ingredients, target in crafts]
        return [*steps, call("finish", message="crafted")]

    try:
        tree = await run_tree(session, [{"spawn": [child_plan() for _ in range(6)]}])
        assert tree.result.status == "completed" and [run.result.status for run in tree.spawned] == ["completed"] * 6
        records = [record for agent in tree.gateway.agents(session.rollout_id) for record in tool_call_records(agent)]
        craft_records = [record for record in records if record["name"] == "craft"]
        succeeded = [record for record in craft_records if record["text"].startswith("Successfully crafted")]
        assert len(craft_records) == 48 and 0 < len(succeeded) < 48  # the children ran out of ingredients
        expected = Counter(initial)
        for record in succeeded:
            expected.subtract(record["arguments"]["ingredients"])
            item, count = record["arguments"]["target"]
            expected[item] += count
        inventory = dict(env_of(session).inventory)
        assert inventory == {item: count for item, count in expected.items() if count}
        assert all(count > 0 for count in inventory.values())
        finishes = [record for record in records if record["name"] == "finish"]
        assert len(finishes) == 6 and all(record["finished"] for record in finishes)
        assert overlaps == [] and peak == 1
    finally:
        await session.close()
