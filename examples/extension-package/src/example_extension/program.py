"""``example-exact-answer``: an episode program that grades the root agent's final answer.

A task table has no grader of its own (its sessions grade ``invalid``, reason ``self_grade``), so this program sets
the outcome: the task's ``answer`` field (a column of the table) is the expected answer. The outcome's primary
reward is 1.0 for an exact match, after normalizing case, spaces and a final period, and 0.0 otherwise. A second
reward, ``close``, says how near a numeric answer came to the expected number, for a reward processor that wants
it (``example-partial-credit``).
"""

from __future__ import annotations

import re

import platoon as pt

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def normalize(text: str) -> str:
    """Lower case, single spaces, no final period or exclamation mark."""

    return " ".join(text.lower().strip().rstrip(".!").split())


def closeness(given: str, expected: str) -> float:
    """1.0 for the expected number, falling linearly to 0.0 as the answer's last number moves away from it."""

    numbers = _NUMBER.findall(given)
    try:
        value, target = float(numbers[-1]), float(expected)
    except (IndexError, ValueError):
        return 0.0
    return max(0.0, 1.0 - abs(value - target) / max(1.0, abs(target)))


@pt.program("example-exact-answer")
async def exact_answer(ep: pt.EpisodeContext) -> pt.Outcome:
    """Run the root agent, then compare its final answer with the task's ``answer``."""

    result = await ep.run()  # the root agent, with the environment's harness and options
    if result.status not in ("completed", "limit"):
        return await ep.grade()  # a crash, timeout or cancellation keeps its untrainable outcome
    expected = ep.task.misc.get("answer")
    if expected is None:  # a row without an answer cannot be measured: never train on it as reward 0
        return pt.Outcome.infra("no_answer", status="invalid", reason="the task has no answer to compare with")
    given = normalize(result.final_message or "")
    rewards = {"reward": float(given == normalize(str(expected))), "close": closeness(given, str(expected))}
    return pt.Outcome(status="scored", rewards=rewards, metadata={"expected": str(expected), "given": given})
