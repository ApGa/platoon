"""What cells may not do: the refusals of the original ``CodeActAgent`` (the agent class this harness replaced),
read off a cell's syntax tree before it runs, and its loop check.

Pure ``ast``, no IPython. :func:`check_cell` returns the original refusal text for a cell that
must not run, checking in the original order: a syntax error, an empty cell, then the enabled
detectors (an async function called without ``await``, a ``while`` loop, ``input()``). The
model sees the refusal as the cell's error. :func:`describe_loop` reports when the last cells
repeat a short pattern; the harness then ends the agent with a synthetic ``finish`` cell, as
the original agent did.

The ``checks`` harness option (:class:`Checks`) switches each of the three detectors on or
off and sets the loop check's threshold and window (a window of 0 switches it off). Syntax
errors and empty cells are always refused.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from platoon.harness.builtin.codeact.prompt_builder import CodeActStep


@dataclass(frozen=True)
class Checks:
    """What CodeAct refuses to run and when it decides an agent is stuck (``harness_options.checks``).

    Attributes:
        unawaited_async_calls: Refuse cells that call one of the agent's async functions (``launch_subagent``)
            without awaiting it.
        while_loops: Refuse cells with ``while`` loops.
        interactive_input: Refuse cells that call ``input()``.
        stuck_in_loop_threshold: How often the last cells must repeat a pattern to count as stuck (at least 1).
        stuck_in_loop_window: The longest pattern (in cells) the loop check looks for; 0 switches the check off.
    """

    unawaited_async_calls: bool = True
    while_loops: bool = False
    interactive_input: bool = False
    stuck_in_loop_threshold: int = 4
    stuck_in_loop_window: int = 3

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any] | None) -> Checks:
        """The ``checks`` option: unknown keys, a threshold below 1 and a negative window are an error."""

        options = dict(options or {})
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(options) - known)
        if unknown:
            raise ValueError(f"Unknown platoon-codeact checks {unknown}; known: {sorted(known)}")
        checks = cls(**options)
        if checks.stuck_in_loop_threshold < 1:
            raise ValueError(f"stuck_in_loop_threshold must be at least 1, got {checks.stuck_in_loop_threshold}")
        if checks.stuck_in_loop_window < 0:
            raise ValueError(f"stuck_in_loop_window must be 0 (off) or more, got {checks.stuck_in_loop_window}")
        return checks


class UnawaitedAsyncCallDetector(ast.NodeVisitor):
    """Finds calls to async functions that are neither awaited nor passed to ``asyncio.gather`` and friends.

    Catches ``launch_subagent(...)`` and ``x = launch_subagent(...)``; allows
    ``await launch_subagent(...)`` and ``asyncio.gather(launch_subagent(...), ...)``.
    """

    def __init__(self, async_functions: Iterable[str]) -> None:
        self.async_functions = frozenset(async_functions)
        self.errors: list[str] = []
        self._in_await = False
        self._in_gather_or_similar = False

    def visit_Await(self, node: ast.Await) -> None:
        old_in_await = self._in_await
        self._in_await = True
        self.generic_visit(node)
        self._in_await = old_in_await

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in (
            "gather",
            "wait",
            "wait_for",
            "create_task",
            "as_completed",
        ):
            # Inside gather-like calls, coroutines are allowed without direct await
            old_in_gather = self._in_gather_or_similar
            self._in_gather_or_similar = True
            self.generic_visit(node)
            self._in_gather_or_similar = old_in_gather
            return
        func_name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if func_name in self.async_functions and not self._in_await and not self._in_gather_or_similar:
            self.errors.append(
                f"ERROR: {func_name}() requires 'await' to run. "
                f"Use: `await {func_name}(...)` or `await asyncio.gather({func_name}(...), ...)`"
            )
        self.generic_visit(node)


class WhileLoopDetector(ast.NodeVisitor):
    """Reports ``while`` loops, which models easily turn into cells that never finish."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_While(self, node: ast.While) -> None:
        condition_src = ast.unparse(node.test)
        self.errors.append(
            "ERROR: `while` loops are not allowed in this environment because they can hang the shared "
            "Python executor. Consider rewriting this as a bounded `for` loop (for example, `for i in range(...)`) "
            f"or iterate over a concrete collection instead. Detected condition: `{condition_src}`"
        )
        self.generic_visit(node)


class InteractiveInputDetector(ast.NodeVisitor):
    """Reports ``input()`` calls, which wait for terminal input that never arrives."""

    INTERACTIVE_CALLS = {"input"}

    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        func_name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if func_name in self.INTERACTIVE_CALLS:
            self.errors.append(
                "ERROR: Interactive input functions like `input()` are not allowed in this environment because "
                "they block the shared Python executor waiting for terminal input that will never arrive."
            )
        self.generic_visit(node)


def check_cell(code: str, checks: Checks, async_names: Iterable[str]) -> str | None:
    """Why ``code`` (already stripped) must not run, in the original agent's words, or None when it may.

    ``async_names`` are the functions the unawaited-call check covers.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return (
            "Execution failed. Traceback:\n"
            + "Syntax error in line:\n"
            + (e.text or "").rstrip()
            + "\n"
            + "Message: "
            + e.msg
        )
    if not code:
        return "No code available to execute."
    detectors: list[Any] = []
    if checks.unawaited_async_calls:
        detectors.append(UnawaitedAsyncCallDetector(async_names))
    if checks.while_loops:
        detectors.append(WhileLoopDetector())
    if checks.interactive_input:
        detectors.append(InteractiveInputDetector())
    for detector in detectors:
        detector.visit(tree)
        if detector.errors:
            return "\n".join(detector.errors)
    return None


def describe_loop(history: Sequence[CodeActStep], checks: Checks) -> str | None:
    """The original loop check: its description of the loop when the last cells repeat a pattern of at most
    ``window`` cells ``threshold`` times, else None."""

    threshold, window = checks.stuck_in_loop_threshold, checks.stuck_in_loop_window
    if len(history) < threshold:
        return None
    codes = [(step.code or "").strip() for step in history]
    for period in range(1, min(window, len(codes) // threshold) + 1):
        segment = codes[-period * threshold :]
        if all(segment[index] == segment[index % period] for index in range(len(segment))):
            return f"Detected a repeating pattern (length ≤ {window}) repeated at least {threshold} times."
    return None


__all__ = ["Checks", "check_cell", "describe_loop"]
