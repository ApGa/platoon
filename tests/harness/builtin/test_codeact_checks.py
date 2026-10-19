"""CodeAct's checks: the original agent's refusals read off a cell's syntax tree, and its stuck-in-loop rule.

No IPython.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from platoon.harness.builtin.codeact import Checks, CodeActOptions, CodeActStep
from platoon.harness.builtin.codeact.checks import check_cell, describe_loop

ALL_CHECKS = Checks(unawaited_async_calls=True, while_loops=True, interactive_input=True)
UNAWAITED = (
    "ERROR: launch_subagent() requires 'await' to run. "
    "Use: `await launch_subagent(...)` or `await asyncio.gather(launch_subagent(...), ...)`"
)


def test_syntax_errors_and_empty_cells_are_refused_first():
    assert check_cell("guess(", ALL_CHECKS, ()) == (
        "Execution failed. Traceback:\nSyntax error in line:\nguess(\nMessage: '(' was never closed"
    )
    # A syntax error wins over every detector.
    assert check_cell("while True: input(", ALL_CHECKS, ()).startswith("Execution failed. Traceback:\n")
    assert check_cell("", ALL_CHECKS, ()) == "No code available to execute."
    assert check_cell("print(1)", ALL_CHECKS, ()) is None


@pytest.mark.parametrize(
    ("code", "refused"),
    [
        ("launch_subagent('x')", True),
        ("x = launch_subagent('x')", True),
        ("agents.launch_subagent('x')", True),  # attribute calls count too
        ("await launch_subagent('x')", False),
        ("print(await launch_subagent('x'))", False),
        ("await asyncio.gather(launch_subagent('a'), launch_subagent('b'))", False),
        ("task = asyncio.create_task(launch_subagent('a'))", False),
        ("await asyncio.wait_for(launch_subagent('a'), 5)", False),
        ("guess(5)", False),
    ],
)
def test_async_functions_must_be_awaited_or_gathered(code, refused):
    assert check_cell(code, Checks(), {"launch_subagent"}) == (UNAWAITED if refused else None)


def test_each_unawaited_call_is_reported():
    assert check_cell("launch_subagent('a')\nlaunch_subagent('b')", Checks(), {"launch_subagent"}) == (
        f"{UNAWAITED}\n{UNAWAITED}"
    )


def test_while_loops_and_input_are_refused_only_when_checked():
    loop, ask = "while x < 3:\n    x += 1", "name = input()"

    assert check_cell(loop, Checks(), ()) is None and check_cell(ask, Checks(), ()) is None
    assert check_cell(loop, ALL_CHECKS, ()) == (
        "ERROR: `while` loops are not allowed in this environment because they can hang the shared Python "
        "executor. Consider rewriting this as a bounded `for` loop (for example, `for i in range(...)`) or iterate "
        "over a concrete collection instead. Detected condition: `x < 3`"
    )
    assert check_cell(ask, ALL_CHECKS, ()) == (
        "ERROR: Interactive input functions like `input()` are not allowed in this environment because they block "
        "the shared Python executor waiting for terminal input that will never arrive."
    )


def test_detectors_run_in_the_original_order():
    # Unawaited calls, then while loops, then input(): the first detector that objects is the refusal.
    code = "while True:\n    launch_subagent(input())"
    assert check_cell(code, ALL_CHECKS, {"launch_subagent"}) == UNAWAITED
    assert check_cell(code, ALL_CHECKS, ()).startswith("ERROR: `while` loops")
    only_input = Checks(unawaited_async_calls=False, interactive_input=True)
    assert check_cell(code, only_input, {"launch_subagent"}).startswith("ERROR: Interactive input functions")
    assert check_cell(code, Checks(unawaited_async_calls=False), {"launch_subagent"}) is None


def test_checks_are_a_nested_harness_option():
    options = CodeActOptions.from_mapping({"checks": {"while_loops": True, "stuck_in_loop_threshold": 6}})

    assert options.checks == Checks(while_loops=True, stuck_in_loop_threshold=6)
    assert CodeActOptions.from_mapping({}).checks == Checks()
    with pytest.raises(ValueError, match=r"Unknown platoon-codeact checks \['while'\]"):
        CodeActOptions.from_mapping({"checks": {"while": True}})
    # The earlier flat keys are gone: they live under `checks` now.
    with pytest.raises(ValueError, match=r"Unknown platoon-codeact harness options \['detect_while_loops'\]"):
        CodeActOptions.from_mapping({"detect_while_loops": True})


def test_a_window_of_0_switches_the_loop_check_off_and_bad_numbers_are_refused():
    off = CodeActOptions.from_mapping({"checks": {"stuck_in_loop_window": 0}}).checks
    assert describe_loop([CodeActStep(code="guess(1)")] * 8, off) is None
    with pytest.raises(ValueError, match="stuck_in_loop_threshold must be at least 1, got 0"):
        CodeActOptions.from_mapping({"checks": {"stuck_in_loop_threshold": 0}})
    with pytest.raises(ValueError, match=r"stuck_in_loop_window must be 0 \(off\) or more, got -1"):
        CodeActOptions.from_mapping({"checks": {"stuck_in_loop_window": -1}})


# ---------------------------------------------------------------------------- the stuck-in-loop rule


def original_stuck_in_loop(history, stuck_in_loop_threshold: int, stuck_in_loop_window: int) -> bool:
    """The original ``CodeActAgent._stuck_in_loop`` (platoon/agents/codeact/agent.py before the harness
    restructure), verbatim."""
    obs = SimpleNamespace(history=history)
    self = SimpleNamespace(stuck_in_loop_threshold=stuck_in_loop_threshold, stuck_in_loop_window=stuck_in_loop_window)

    if len(obs.history) < self.stuck_in_loop_threshold:
        return False
    codes = [(step.code or "").strip() for step in obs.history]
    n = len(codes)
    max_period = min(self.stuck_in_loop_window, n // self.stuck_in_loop_threshold)
    if max_period <= 0:
        return False
    for period in range(1, max_period + 1):
        segment_len = period * self.stuck_in_loop_threshold
        segment = codes[-segment_len:]
        pattern = segment[:period]
        matches_pattern = True
        for i in range(segment_len):
            if segment[i] != pattern[i % period]:
                matches_pattern = False
                break
        if matches_pattern:
            return True
    return False


def test_the_loop_check_matches_the_original_on_random_histories():
    rng = random.Random(20261014)
    codes = ["guess(1)", " guess(1)\n", "guess(2)", "craft()", "", None]
    stuck = 0
    for _ in range(20000):
        history = [CodeActStep(code=rng.choice(codes)) for _ in range(rng.randint(0, 16))]
        threshold, window = rng.randint(1, 6), rng.randint(-1, 5)
        expected = original_stuck_in_loop(history, threshold, window)
        actual = describe_loop(history, Checks(stuck_in_loop_threshold=threshold, stuck_in_loop_window=window))
        assert (actual is not None) == expected, (history, threshold, window)
        if expected:
            stuck += 1
            assert actual == f"Detected a repeating pattern (length ≤ {window}) repeated at least {threshold} times."
    assert 1000 < stuck < 19000  # both outcomes are well represented


def test_the_loop_check_needs_threshold_repeats_of_a_short_pattern():
    def history(*codes):
        return [CodeActStep(code=code) for code in codes]

    assert describe_loop(history("a", "a", "a"), Checks()) is None
    assert describe_loop(history("a", "a", "a", "a"), Checks()) is not None
    assert describe_loop(history("a", "b") * 4, Checks()) is not None
    assert describe_loop(history("a", "b", "c", "d") * 4, Checks()) is None  # longer than the window of 3
    assert describe_loop(history("x", "a", "a", "a", "a"), Checks()) is not None  # only the tail counts
