"""The conformance check of a group admission rule (RFC 0019, section 6): :func:`check_group_admission`.

It is made of named checks, each a function of this module (:data:`CHECKS`), which :func:`check_group_admission` runs
one after the other, at each level the rule serves; its ``AssertionError`` lists each check that failed, by name, with
what it saw.  Each answer is judged by the trainer's own rule (:func:`~platoon.train.admission_api.checked_admission`),
so that what it checks is what runs.

The fixture is ``check_task_sampler``'s (:mod:`platoon.testing.samplers`): a pool of 40 tasks in two entries, ``a``
(group size 4) and ``b`` (group size 8), and the run's per-task statistics over it, fed synthetic settled groups, with
some draws outstanding, which the rule reads through a view; a second pool (a fifth of each entry's tasks replaced,
other statistics) gives a second view.  It is asked about :data:`RECORDS` settled groups: the fixture's own under every
verdict a workflow gives, without valid members, without members at all (a workflow's plain list of datums), without
labels (a single environment's groups) and without an entry either (a trainer given an order); at the entry level,
those of entry ``a``.  The timing check uses a pool of :data:`TIMED_TASKS` tasks.

Above all it looks for state that a rule keeps between calls (``pure``): exact resume rests on a rule being a pure
function of what it is given, since a resume decides the groups that settle after it with a new instance.  No check can
prove that: a rule whose state changes its answers only after more calls than the check makes, or only on groups or
statistics the fixture lacks, passes it.

The experimental admission API (``platoon.experimental``) can change in a minor release, and this check changes with it.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

# The groups each check asks about, and the pool and the decisions the timing check times.
RECORDS = 400
TASKS = 40
TIMED_TASKS = 4096
TIMED_DECISIONS = 101
# How many times ``pure`` asks one instance about the groups before it compares that instance with new ones.
PASSES = 10
# The checks, by name, in the order they run.
CHECKS = ("protocol", "options", "answers", "pure", "fast")
# The levels a rule serves: the run's rule, and an entry's own.
LEVELS: tuple[Literal["run", "entry"], ...] = ("run", "entry")
# A key no options schema takes.
_UNKNOWN_OPTION = "conformance_check_unknown_option"
# The policy version the fixture's views read at.
_NOW = 4
# The entry whose own rule the entry level builds, and the shapes of group asked about beside each verdict.
_ENTRY = "a"
_SHAPES = ("no valid member", "no members", "no labels", "no labels or entry")


def check_group_admission(
    rule: type | str,
    *,
    options: Mapping[str, Any] | None = None,
    levels: Sequence[Literal["run", "entry"]] = ("run", "entry"),
    budget_ms: float | None = 2.0,
) -> None:
    """Check a group admission rule against RFC 0019's contract; raise ``AssertionError`` naming every check it fails.

    ``rule`` is a class, a name a package registers with ``@group_admission``, or ``module:Class``; ``options`` are
    checked against its options schema as a config's would be, and it is built with them; ``levels`` are the levels it
    serves (``run``, the run's rule, and ``entry``, an entry's own: a rule only an entry may set serves ``entry``
    alone); ``budget_ms`` is the most a decision may take, the median over decisions on a pool of 4,096 tasks (None:
    not timed).  Given by reference (a registered name of an importable package, or ``module:Class``), its answers are
    also replayed in two processes with other ``PYTHONHASHSEED``s, which catches iteration over a set of strings; a
    class given directly skips that replay."""

    if isinstance(levels, str) or not levels or any(level not in LEVELS for level in levels):
        raise ValueError(f"levels: one or both of {', '.join(LEVELS)}, got {levels!r}")
    target = _target(rule)
    failures = run_checks(target, options=options, levels=levels, budget_ms=budget_ms)
    if failures:
        lines = [f"{name}: {what}" for name, found in failures.items() for what in found]
        raise AssertionError(
            f"group admission rule {target.name} fails {len(failures)} check(s):\n  " + "\n  ".join(lines)
        )


# --- the target ------------------------------------------------------------------------------------------------------


@dataclass
class _Target:
    """The rule under check: its name, its class (None when it does not resolve), the reference a subprocess resolves
    (None: not replayed in subprocesses) with the module to import first, and the problem that kept it from
    resolving."""

    name: str
    cls: type | None
    ref: str | None = None
    module: str | None = None
    problem: str | None = None


def _target(rule: type | str) -> _Target:
    from platoon.extensions import ExtensionError
    from platoon.train.admission_api import resolve_group_admission

    if isinstance(rule, str):
        try:
            cls = resolve_group_admission(rule)
        except ExtensionError as exc:
            return _Target(name=rule, cls=None, problem=str(exc))
        return _Target(name=rule, cls=cls, ref=rule, module=cls.__module__)
    name = getattr(rule, "__qualname__", repr(rule))
    return _Target(name=name, cls=rule if isinstance(rule, type) else None)


def run_checks(
    target: _Target,
    *,
    options: Mapping[str, Any] | None = None,
    levels: Sequence[str] = LEVELS,
    budget_ms: float | None = 2.0,
) -> dict[str, list[str]]:
    """Run every check of :data:`CHECKS` on ``target``, at each of ``levels``: the failures by check, each with what it
    saw."""

    failures: dict[str, list[str]] = {}

    def fail(name: str, found: Sequence[str]) -> None:
        if found:
            failures.setdefault(name, []).extend(found)

    problems = check_protocol(target, options, levels)
    fail("protocol", problems)
    if target.cls is None or problems:
        return failures
    parsed, found = check_options(target, options)
    fail("options", found)
    if found:
        return failures
    for index, level in enumerate(levels):
        where = f"at the {level} level, " if len(levels) > 1 else ""
        fail("answers", [where + what for what in check_answers(target, parsed, level)])
        fail("pure", [where + what for what in check_pure(target, parsed, options or {}, level, replay=index == 0)])
    if budget_ms is not None:
        fail("fast", check_fast(target, parsed, budget_ms, levels[0]))
    return failures


# --- the fixture -----------------------------------------------------------------------------------------------------


def _setup(target: _Target, infos: Mapping[str, Any], level: str = "run") -> Any:
    from platoon.testing.samplers import _COLUMN
    from platoon.train.admission_api import AdmissionSetup

    name = target.ref or target.name
    return AdmissionSetup(
        name=name if level == "run" else f"{name}@{_ENTRY}",
        level=level,  # type: ignore[arg-type]
        entry=None if level == "run" else _ENTRY,
        entries=infos,
        batch_size=8,
        label_column=_COLUMN,
    )


def _instance(target: _Target, parsed: Any, level: str = "run", tasks: int = TASKS, *, entries: bool = True) -> Any:
    """A new instance of the rule over the fixture's pool, at ``level``; without ``entries``, built as a trainer given
    an order without sampling builds it (no entries)."""

    from platoon.testing.samplers import _fixture

    _rows, infos, _records, _store = _fixture(tasks)
    assert target.cls is not None
    return target.cls(_setup(target, infos if entries else {}, level), parsed)


def _view(tasks: int = TASKS, *, changed: bool = False) -> Any:
    """A view of the fixture's statistics; with ``changed``, of the second pool's."""

    from platoon.testing.samplers import _fixture
    from platoon.train.sampler_api import TaskStatsView

    _rows, _infos, _records, store = _fixture(tasks, changed=changed)
    return TaskStatsView(store, now=_NOW)


def _check_context(level: str) -> Any:
    """A context of the rule's load-time check as a config of the fixture's two entries gives it, at ``level``."""

    from platoon.testing.samplers import _COLUMN, _ENTRIES
    from platoon.train.admission_api import AdmissionCheckContext

    entries = {
        label: {
            "weight": weight,
            "seed": seed,
            "group_size": size,
            "success_threshold": 1.0,
            "task_sampler": "uniform",
            "task_sampler_options": {},
        }
        for label, size, _task_labels, weight, seed in _ENTRIES
    }
    return AdmissionCheckContext(
        level=level,  # type: ignore[arg-type]
        entry=None if level == "run" else _ENTRY,
        entries=entries,
        batch_size=8,
        label_column=_COLUMN,
        max_groups_per_step=3.0,
        workflow_gives_verdicts=True,
    )


_GROUPS: dict[tuple[int, str], list[Any]] = {}


def groups(tasks: int = TASKS, level: str = "run") -> list[Any]:
    """:data:`RECORDS` settled groups of the fixture's pool (at the entry level, entry ``a``'s): its own records, each
    under every verdict a workflow gives, without valid members, without members (a plain list of datums), without
    labels (a single environment's group) and, at the run level, without an entry either (a trainer given an order),
    the shapes taking turns so that each meets many records, in a fixed order."""

    key = (tasks, level)
    if key in _GROUPS:
        return _GROUPS[key]
    from platoon.testing.samplers import _fixture
    from platoon.train.task_stats import VERDICTS

    _rows, _infos, records, _store = _fixture(tasks)
    if level != "run":
        records = [record for record in records if record.entry == _ENTRY]
    shapes = [*VERDICTS, *(_SHAPES if level == "run" else _SHAPES[:-1])]
    found: list[Any] = []
    for index in range(RECORDS):
        base = records[index % len(records)]
        shape = shapes[(index + index // len(records)) % len(shapes)]
        if shape in VERDICTS:
            found.append(dataclasses.replace(base, verdict=shape))
        elif shape == "no valid member":
            invalid = tuple(dataclasses.replace(member, valid=False, scored=None) for member in base.members)
            found.append(dataclasses.replace(base, members=invalid, verdict="no_completed_root"))
        elif shape == "no members":
            found.append(dataclasses.replace(base, members=(), verdict="unknown"))
        elif shape == "no labels":
            found.append(dataclasses.replace(base, labels={}))
        else:
            found.append(dataclasses.replace(base, labels={}, entry=None))
    _GROUPS[key] = found
    return found


def _answer(rule_name: str, answer: Any) -> Any:
    """An answer as JSON data, to compare and to print; ``{"invalid": ...}`` for one outside the contract."""

    from platoon.train.admission_api import AdmissionError, checked_admission

    if inspect.isawaitable(answer):
        close = getattr(answer, "close", None)
        if callable(close):
            close()
        return {"invalid": "an awaitable: a decision is one synchronous call"}
    try:
        decision = checked_admission(answer, rule_name)
    except AdmissionError as exc:
        return {"invalid": str(exc)}
    return {"admit": decision.admit, "reason": decision.reason}


def _ask(instance: Any, view: Any, records: Sequence[Any], name: str) -> list[Any]:
    """The rule's answers to ``records``, in their order, as JSON data; ``{"raised": ...}`` for a call that raised."""

    answers = []
    for record in records:
        try:
            answers.append(_answer(name, instance.admit(record, view)))
        except Exception as exc:
            answers.append({"raised": f"{type(exc).__name__}: {exc}"})
    return answers


def _first(problems: Sequence[str], most: int = 3) -> list[str]:
    shown = list(dict.fromkeys(problems))
    return shown[:most] + ([f"... and {len(shown) - most} more"] if len(shown) > most else [])


# --- the checks ------------------------------------------------------------------------------------------------------


def check_protocol(
    target: _Target, options: Mapping[str, Any] | None = None, levels: Sequence[str] = LEVELS
) -> list[str]:
    """``protocol``: it resolves; it has a callable ``admit(group, stats)`` that is no coroutine function;
    ``Class(setup, options)`` builds it at each level, with the fixture's entries and with none (a trainer given an
    order without sampling); and its load-time ``check(options, ctx)``, called as config loading calls it, returns or
    refuses the config with a ``ValueError`` at each level."""

    from platoon.extensions import ExtensionError
    from platoon.train.admission_api import admission_problem

    if target.problem is not None:
        return [target.problem]
    if target.cls is None:
        return [f"{target.name} is no class"]
    try:
        admission_problem(target.cls, target.name)
    except ExtensionError as exc:
        return [str(exc)]
    parsed, problems = check_options(target, options)
    if problems:
        return []  # options reports them
    found = []
    for level in levels:
        for entries, how in ((True, ""), (False, " with no entries (a trainer given an order without sampling)")):
            try:
                _instance(target, parsed, level, entries=entries)
            except Exception as exc:
                found.append(f"Class(setup, options) at the {level} level{how} raised {type(exc).__name__}: {exc}")
    method = getattr(target.cls, "check", None)
    if callable(method):
        for level in levels:
            try:
                method(parsed, _check_context(level))
            except ValueError:
                pass  # a refusal, which says why
            except Exception as exc:
                found.append(
                    f"check(options, ctx) at the {level} level raised {type(exc).__name__}: {exc} (a check is a static "
                    "method that returns, or refuses the config with a ValueError)"
                )
    return found


def check_options(target: _Target, options: Mapping[str, Any] | None) -> tuple[Any, list[str]]:
    """``options``: its schema takes no options (``{}``) and the options given; a dataclass's or a pydantic model's
    refuses an unknown key.  Returns the options given, parsed."""

    from platoon.options import schema_options
    from platoon.train.admission_api import rule_options_schema
    from platoon.utils.config import _is_pydantic

    schema = rule_options_schema(target.cls) if target.cls is not None else None
    checker = schema_options(schema, name=target.name)
    problems = [f"the default options: {problem}" for problem in checker.option_problems({}, "options")]
    parsed, found = checker.check(dict(options or {}), "options")
    problems += [f"the options given: {problem}" for problem in found]
    typed = schema is not None and (
        (isinstance(schema, type) and dataclasses.is_dataclass(schema)) or _is_pydantic(schema)
    )
    if typed and not checker.option_problems({**dict(options or {}), _UNKNOWN_OPTION: 1}, "options"):
        problems.append(f"its options schema takes an unknown key ({_UNKNOWN_OPTION}); it should refuse it")
    return parsed, problems


def check_answers(target: _Target, parsed: Any, level: str = "run") -> list[str]:
    """``answers``: over :data:`RECORDS` settled groups of every verdict and shape (with and without valid members,
    members, labels, an entry), each answer is an ``Admission`` (whose reason is a non-empty string of at most 64
    characters) or a ``bool``, and no call raises."""

    instance = _instance(target, parsed, level)
    problems = []
    for index, answer in enumerate(_ask(instance, _view(), groups(level=level), target.name)):
        if "raised" in answer:
            problems.append(f"admit() raised on group {index}: {answer['raised']}")
        elif "invalid" in answer:
            problems.append(f"group {index}: {answer['invalid']}")
    return _first(problems)


def check_pure(
    target: _Target, parsed: Any, written: Mapping[str, Any], level: str = "run", *, replay: bool = True
) -> list[str]:
    """``pure``: the rule answers as a new instance would, whatever it was asked before.  Over the fixture's view, an
    instance asked about the groups in order and then again, and another asked in reverse order, answer every group
    alike; asked :data:`PASSES` times over, the first answers each group as an instance built for that group alone
    does (a count kept in the rule fails here); and over the second view it answers as a new instance does (a view
    kept from an earlier call fails here).  Given by reference (and ``replay``), so do two processes with other hash
    seeds.  A rule that keeps state fails it where a resume, whose new instance starts without that state, would
    decide otherwise; one whose state shows only after more calls, or on other inputs, can pass."""

    name = target.name
    records = groups(level=level)
    view = _view()
    first = _instance(target, parsed, level)
    once = _ask(first, view, records, name)
    again = _ask(first, view, records, name)
    reverse = list(reversed(_ask(_instance(target, parsed, level), view, list(reversed(records)), name)))
    alike = []
    for index, (a, b, c) in enumerate(zip(once, again, reverse)):
        if a != b:
            alike.append(f"group {index} asked twice answered {a!r}, then {b!r}")
        elif a != c:
            alike.append(f"group {index} answered {a!r} in order and {c!r} by an instance asked in reverse order")
    for _ in range(PASSES - 2):
        _ask(first, view, records, name)
    late = _ask(first, view, records, name)
    alone = [_ask(_instance(target, parsed, level), view, [record], name)[0] for record in records]
    history = [
        f"group {index} answered {a!r} after {PASSES} passes over the groups, and {b!r} asked of a new instance"
        for index, (a, b) in enumerate(zip(late, alone))
        if a != b
    ]
    second = _view(changed=True)
    kept = [
        f"over a second view of the statistics, group {index} answered {a!r}, and {b!r} asked of a new instance"
        for index, (a, b) in enumerate(
            zip(_ask(first, second, records, name), _ask(_instance(target, parsed, level), second, records, name))
        )
        if a != b
    ]
    problems = _first(alike) + _first(history) + _first(kept)
    if target.ref is None or not replay:
        return problems
    for seed in ("0", "1"):
        try:
            elsewhere = _replay_elsewhere(target, written, seed, level)
        except RuntimeError as exc:
            problems.append(str(exc))
            continue
        if elsewhere != once:
            index = next((i for i, (a, b) in enumerate(zip(once, elsewhere)) if a != b), min(len(once), len(elsewhere)))
            problems.append(
                f"a process with PYTHONHASHSEED={seed} answered group {index} otherwise: "
                f"{elsewhere[index] if index < len(elsewhere) else None!r} against "
                f"{once[index] if index < len(once) else None!r}"
            )
    return problems


def _replay_elsewhere(target: _Target, written: Mapping[str, Any], seed: str, level: str) -> list[Any]:
    request = {"ref": target.ref, "module": target.module, "options": dict(written), "level": level}
    environment = {**os.environ, "PYTHONHASHSEED": seed}
    done = subprocess.run(
        [sys.executable, "-m", "platoon.testing.admissions", json.dumps(request)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
    )
    if done.returncode != 0:
        raise RuntimeError(
            f"replaying its answers in a process with PYTHONHASHSEED={seed} failed: "
            f"{(done.stderr or done.stdout).strip()[-400:]}"
        )
    return json.loads(done.stdout)


def check_fast(target: _Target, parsed: Any, budget_ms: float, level: str = "run") -> list[str]:
    """``fast``: ``admit`` is synchronous, and the median of :data:`TIMED_DECISIONS` decisions on the statistics of a
    pool of :data:`TIMED_TASKS` tasks stays within ``budget_ms``: a decision runs in the group's coroutine, beside
    every other group's."""

    if inspect.iscoroutinefunction(getattr(target.cls, "admit", None)):
        return ["its admit(group, stats) is a coroutine function"]
    try:
        instance, view = _instance(target, parsed, level, TIMED_TASKS), _view(TIMED_TASKS)
    except Exception as exc:
        return [f"building it on {TIMED_TASKS} tasks raised {type(exc).__name__}: {exc}"]
    records = groups(TIMED_TASKS, level)
    times = []
    for index in range(TIMED_DECISIONS):
        record = records[index % len(records)]
        started = time.perf_counter()
        try:
            instance.admit(record, view)
        except Exception as exc:
            return [f"admit() raised on {TIMED_TASKS} tasks: {type(exc).__name__}: {exc}"]
        times.append(1000.0 * (time.perf_counter() - started))
    median = statistics.median(times)
    if median > budget_ms:
        return [
            f"a decision on the statistics of {TIMED_TASKS} tasks takes {median:.2f} ms (the median of "
            f"{TIMED_DECISIONS}), more than the budget of {budget_ms:g} ms"
        ]
    return []


# --- the replay in another process -----------------------------------------------------------------------------------


def _main(argv: Sequence[str]) -> int:
    """``python -m platoon.testing.admissions '<request>'``: print, as JSON, a rule's answers to the fixture's groups
    at the level the request names (its replay in a process with another hash seed)."""

    import importlib

    from platoon.options import schema_options
    from platoon.train.admission_api import rule_options_schema

    request = json.loads(argv[0])
    if request.get("module"):
        importlib.import_module(request["module"])
    target = _target(request["ref"])
    if target.cls is None:
        print(target.problem, file=sys.stderr)
        return 1
    level = request.get("level", "run")
    parsed = schema_options(rule_options_schema(target.cls), name=target.name).parse(request["options"], "options")
    print(json.dumps(_ask(_instance(target, parsed, level), _view(), groups(level=level), target.name)))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
