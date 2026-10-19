"""The helpers the agent contract gives every harness loop: the step limit and the env's terminal tool."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from platoon.envs.service import ToolSpec
from platoon.harness.agent import LIMIT_MESSAGE, out_of_steps, terminal_arguments, terminal_tool


def _ctx(*, remaining: float, max_steps: int | None) -> SimpleNamespace:
    return SimpleNamespace(budget=SimpleNamespace(remaining=lambda: remaining), max_steps=max_steps)


@pytest.mark.parametrize(
    ("remaining", "max_steps", "steps_taken", "expected"),
    [
        (math.inf, None, 1000, False),  # unbounded
        (math.inf, 3, 2, False),
        (math.inf, 3, 3, True),  # ctx.max_steps reached, whatever the budget says
        (0, None, 0, True),  # the budget is spent (children's reservations count)
        (-1, 10, 0, True),
        (1, 10, 9, False),
    ],
)
def test_out_of_steps_checks_the_budget_and_max_steps(remaining, max_steps, steps_taken, expected):
    assert out_of_steps(_ctx(remaining=remaining, max_steps=max_steps), steps_taken) is expected


def test_the_limit_message_keeps_its_wording():
    # Behavior judges may recognize a limit by this prefix.
    assert LIMIT_MESSAGE == "WARNING: Exhausted budget when running episode. Halting episode; task may be incomplete."


def test_terminal_tool_and_its_arguments():
    guess = ToolSpec("guess", "Guess.", {"type": "object", "properties": {"number": {"type": "integer"}}})
    submit = ToolSpec(
        "submit",
        "Submit.",
        {"type": "object", "properties": {"answer": {"type": "string"}, "note": {"type": "string"}}},
        terminal=True,
    )
    done = ToolSpec("done", "Done.", {}, terminal=True)

    assert terminal_tool([guess, submit, done]) is submit
    assert terminal_tool([guess]) is None
    assert terminal_arguments(submit, "42") == {"answer": "42"}  # the message goes in the first property
    assert terminal_arguments(done, "42") == {}
