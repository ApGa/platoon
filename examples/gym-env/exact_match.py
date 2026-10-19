"""``example-exact-match``: an episode program that grades a task-table row by its ``answer`` column.

A task table has no grader (its sessions grade ``invalid``, reason ``self_grade``), so the program sets the
outcome itself.  Importing this module registers the program (configs import it with ``package: exact_match``).
"""

from __future__ import annotations

import platoon as pt


def normalize(text: str | None) -> str:
    return " ".join((text or "").lower().strip().rstrip(".!").split())


@pt.program("example-exact-match")
async def exact_match(ep: pt.EpisodeContext) -> pt.Outcome:
    result = await ep.run()  # the root agent, with the environment's harness
    if result.status not in ("completed", "limit"):
        return await ep.grade()  # a crash, timeout or cancellation keeps its untrainable outcome
    expected = normalize(str(ep.task.misc["answer"]))
    correct = normalize(result.final_message) == expected
    return pt.Outcome(status="scored", rewards={"reward": float(correct)}, metadata={"expected": expected})
