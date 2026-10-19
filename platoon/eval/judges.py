"""``inference.judges``: which environments an evaluation with its judges off refuses, and what ``lock.json``
records about the switch.

With judges off, a child that its environment grades by verifier (``agents.child_grading: verifier``) is not
judged: it shares the episode's outcome, as with ``child_grading: none``, and records why
(:mod:`platoon.utils.judges`).  The root's success, which evaluation measures, never depends on that.  Code of the
environment's own may, though: a ``reward_processor`` or an episode program other than core's can read child
verdicts, which platoon cannot check, so such an entry is refused until it states the fallback itself
(``agents.child_grading: none``) or the evaluation keeps its judges on.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from platoon.utils.judges import JUDGES_OFF


def _verifier_graded(environment: Any) -> bool:
    """Whether an entry may spawn children and grades them by verifier."""

    agents = environment.agents
    return agents.child_grading == "verifier" and agents.max_depth != 0


def _core_program(name: str) -> bool:
    from platoon import extensions
    from platoon.programs import resolve_program

    try:
        record = resolve_program(name)
    except Exception:  # a program that does not resolve is refused where it is resolved; here it is not core's
        return False
    return extensions.is_core_module(record.fn.__module__)


def judges_off_problems(environments: Sequence[Any]) -> list[str]:
    """The entries an evaluation with its judges off refuses (see the module docstring), one line each."""

    problems = []
    for environment in environments:
        if not _verifier_graded(environment):
            continue
        readers = []
        if environment.reward_processor:
            readers.append(f"reward_processor {environment.reward_processor!r}")
        if not _core_program(environment.program):
            readers.append(f"program {environment.program!r}")
        if readers:
            problems.append(
                f"environments[{environment.label}]: inference.judges is false, so no verifier grades this entry's "
                f"children (agents.child_grading: verifier) and they share the episode's outcome, but its "
                f"{' and '.join(readers)} may read their verdicts, which platoon cannot check; set "
                "agents.child_grading: none for this evaluation to accept the fallback, or keep inference.judges on"
            )
    return problems


def judges_lock_record(environments: Sequence[Any], *, enabled: bool) -> dict[str, Any]:
    """``lock.json``'s ``judges``: whether they run and, when they do not, each entry whose children's grading
    falls back from ``verifier`` to ``none``."""

    record: dict[str, Any] = {"enabled": enabled}
    if not enabled:
        record["child_grading"] = {
            environment.label: {"configured": "verifier", "applied": "none", "reason": JUDGES_OFF}
            for environment in environments
            if _verifier_graded(environment)
        }
    return record


__all__ = ["judges_lock_record", "judges_off_problems"]
