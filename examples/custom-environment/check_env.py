"""Drive the order-packing environment through platoon's environment service, without a model.

    cd examples/custom-environment
    PYTHONPATH=. uv run python check_env.py

It lists tasks, opens the session a rollout would open, shows the tools with platoon's hints, makes the calls an
agent would make (one from a subagent's forked session) and prints both grades.
"""

from __future__ import annotations

import asyncio

import platoon as pt
from platoon.envs import get_service


def show(outcome: pt.Outcome) -> str:
    return f"{outcome.status} reward={outcome.reward} reason={outcome.reason}"


async def main() -> None:
    service = get_service("python://order_packing:OrderPackingEnv")
    print("splits:", await service.list_splits())
    print("test tasks:", await service.list_tasks("test"))

    session = await service.open(await service.get_task("train:0"), rollout_id="check-1")
    try:
        print("instruction:", session.instruction.splitlines()[0])
        for spec in await session.tools():
            hints = [hint for hint in ("terminal", "read_only") if getattr(spec, hint)]
            print(f"tool {spec.name}({', '.join(spec.input_schema['properties'])}): {', '.join(hints) or '-'}")

        bad = await session.call_tool("pack", {"item": "apple", "quantity": 0})  # quantity must be >= 1
        print("pack 0 apple:", "error" if bad.is_error else "ok", bad.metadata.get("error_reason"))

        # A subagent's session: platoon_fork gives it its own goal on the same shelf and box.
        child = await session.fork("shared", task=pt.Task(id="child-1", goal="Pack 2 apple"))
        try:
            print("child:", (await child.call_tool("pack", {"item": "apple", "quantity": 2})).as_text())
            await child.call_tool("done", {"summary": "2 apple"})
            print("child grade:", show(await child.grade()))
        finally:
            await child.close()

        print("root:", (await session.call_tool("pack", {"item": "bread", "quantity": 1})).as_text())
        print("root:", (await session.call_tool("look", {})).as_text().replace("\n", " "))
        await session.call_tool("done", {"summary": "packed"})
        print("root grade:", show(await session.grade()))
    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(main())
