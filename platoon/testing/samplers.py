"""The conformance check of a task sampler (RFC 0019, section 5): :func:`check_task_sampler`.

It is made of named checks, each a function of this module (:data:`CHECKS`), which :func:`check_task_sampler` runs
one after the other for each level a sampler serves; its ``AssertionError`` lists each check that failed, by name,
with what it saw.  Each check builds its decisions' contexts with the dispatcher's own factory
(:meth:`~platoon.train.sampler_runtime.SamplerSet.context`) and judges each answer by the dispatcher's own rules
(:meth:`~platoon.train.sampler_runtime.SamplerSet.accepts`), so that what it checks is what runs.

The fixture: a pool of :data:`TASKS` tasks in two entries, ``a`` (group size 4, its tasks labelled ``x`` and ``w`` in
turn) and ``b`` (group size 8, label ``y``), and the run's per-task statistics over it, fed synthetic settled groups
(pass counts drawn from fixed per-task rates, some members without a valid result, some groups failed), with some draws
outstanding.  An adaptive sampler is asked under three eligibility patterns: every task eligible, label ``x`` at its
limit (which leaves entry ``a`` its ``w`` tasks), and nothing eligible; and one instance under the three in turn, so
that a sampler that keeps a decision's eligibility shows.  The timing check uses a pool of :data:`TIMED_TASKS` tasks
(at the entry level, each entry's).

The experimental sampler API (``platoon.experimental``) can change in a minor release, and this check changes with it.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

# The fixture's pool, the decisions asked under each eligibility pattern, and the pool and the decisions timed.
TASKS = 40
DECISIONS = 200
TIMED_TASKS = 4096
TIMED_DECISIONS = 41
# The decisions the determinism check replays, in this process and in two others with other hash seeds.
REPLAYED = 60
# The checks, by name, in the order they run.
CHECKS = (
    "protocol",
    "options",
    "pool-task-or-nothing",
    "fast",
    "deterministic",
    "static",
    "state",
    "observe",
    "metrics",
)
# The fixture's label column and its entries: (label, group size, its tasks' labels in turn, weight, seed).
_COLUMN = "label"
_ENTRIES = (("a", 4, ("x", "w"), 1.0, 3), ("b", 8, ("y",), 2.0, 5))
# The decisions of each eligibility pattern in the series that switches them in one instance.
_SWITCH_EVERY = 10
# A key no options schema takes.
_UNKNOWN_OPTION = "conformance_check_unknown_option"


def check_task_sampler(
    sampler: type | str,
    *,
    options: Mapping[str, Any] | None = None,
    levels: Sequence[Literal["run", "entry"]] = ("entry", "run"),
    budget_ms: float | None = 20.0,
) -> None:
    """Check a task sampler against RFC 0019's contract; raise ``AssertionError`` naming every check it fails.

    ``sampler`` is a class, a name a package registers with ``@task_sampler``, or ``module:Class``; ``options`` are
    checked against its options schema as a config's would be, and it is built with them; ``levels`` are the levels it
    serves (a sampler that hands slots to the entries serves the run's only); ``budget_ms`` is the most a decision may
    take, the median over decisions on a pool of 4,096 tasks (None: not timed).  Given by reference (a registered name
    of an importable package, or ``module:Class``), its decisions are also replayed in two processes with other
    ``PYTHONHASHSEED``s, which catches iteration over a set of strings; a class given directly skips that replay."""

    target = _target(sampler)
    failures = run_checks(target, options=options, levels=levels, budget_ms=budget_ms)
    if failures:
        lines = [f"{name}: {what}" for name, found in failures.items() for what in found]
        raise AssertionError(f"task sampler {target.name} fails {len(failures)} check(s):\n  " + "\n  ".join(lines))


# --- the target ------------------------------------------------------------------------------------------------------


@dataclass
class _Target:
    """The sampler under check: its name, its class (None when it does not resolve), how to build it, the reference a
    subprocess resolves (None: not replayed in subprocesses) with the module to import first, and the problem that
    kept it from resolving."""

    name: str
    cls: type | None
    factory: Callable[[Any, Any], Any] | None = None
    ref: str | None = None
    module: str | None = None
    problem: str | None = None

    def build(self, setup: Any, options: Any) -> Any:
        factory = self.factory or self.cls
        assert factory is not None
        return factory(setup, options)


def _target(sampler: type | str) -> _Target:
    from platoon.extensions import ExtensionError
    from platoon.train.sampler_api import resolve_task_sampler

    if isinstance(sampler, str):
        try:
            cls = resolve_task_sampler(sampler)
        except ExtensionError as exc:
            return _Target(name=sampler, cls=None, problem=str(exc))
        return _Target(name=sampler, cls=cls, ref=sampler, module=cls.__module__)
    name = getattr(sampler, "__qualname__", repr(sampler))
    return _Target(name=name, cls=sampler if isinstance(sampler, type) else None)


def run_checks(
    target: _Target,
    *,
    options: Mapping[str, Any] | None = None,
    levels: Sequence[str] = ("entry", "run"),
    budget_ms: float | None = 20.0,
    internal: bool = False,
) -> dict[str, list[str]]:
    """Run every check of :data:`CHECKS` on ``target`` at each of ``levels``: the failures by check, each with what it
    saw.  ``internal`` admits a target that is not a class of the protocol, such as core's internal orders."""

    failures: dict[str, list[str]] = {}

    def fail(name: str, found: Sequence[str]) -> None:
        if found:
            failures.setdefault(name, []).extend(found)

    problems = _protocol(target, levels, internal=internal)
    fail("protocol", problems)
    if target.cls is None or (problems and not internal):
        return failures
    parsed, found = _options(target, options)
    fail("options", found)
    if found:
        return failures
    for level in levels:
        cases = list(_cases(target, parsed, level, options or {}))
        if not cases:
            continue
        for check, name in (
            (check_pool_task_or_nothing, "pool-task-or-nothing"),
            (check_static, "static"),
            (check_state, "state"),
            (check_observe, "observe"),
            (check_metrics, "metrics"),
        ):
            for case in cases:
                fail(name, [f"{case.where}: {what}" for what in check(case)])
        fail("deterministic", [f"{cases[0].where}: {what}" for what in check_deterministic(cases[0])])
        if budget_ms is not None:
            fail("fast", [f"{cases[0].where}: {what}" for what in check_fast(cases[0], budget_ms)])
    return failures


# --- the fixture -----------------------------------------------------------------------------------------------------


@dataclass
class Case:
    """One sampler of the fixture, at one level: how to build the sampler set around it (``build``, a fresh one each
    call, over a pool of ``tasks`` tasks a side, or another pool), which member of it is under check, its target, its
    options as given, and its name for the failures."""

    target: _Target
    options: Any
    written: Mapping[str, Any]
    level: str
    entry: str | None
    where: str

    def build(self, tasks: int = TASKS, *, changed: bool = False) -> tuple[Any, Any]:
        """A fresh sampler set over the fixture's pool (``tasks`` tasks, half of them each entry's, the first fifth of
        each entry's renamed when ``changed``), with its statistics, and the member under check."""

        samplers = _sampler_set(self, tasks, changed=changed)
        member = samplers.run if self.entry is None else samplers.entries[self.entry]
        return samplers, member

    @property
    def adaptive(self) -> bool:
        return getattr(self.target.cls, "adaptive", False) is True


def _cases(target: _Target, parsed: Any, level: str, written: Mapping[str, Any]) -> Iterator[Case]:
    if level == "run":
        yield Case(target, parsed, written, "run", None, f"{target.name} at the run level")
    else:
        for label, *_ in _ENTRIES:
            yield Case(target, parsed, written, "entry", label, f"{target.name} as entry {label}'s sampler")


def _rows(tasks: int, *, changed: bool = False) -> list[dict[str, Any]]:
    """The fixture's pool: ``tasks`` tasks, half of them in each entry; with ``changed``, the first fifth of each
    entry's gone and as many new ones."""

    from platoon.train.components import ENVIRONMENT_COLUMN

    rows = []
    half = tasks // 2
    for label, _size, task_labels, _weight, _seed in _ENTRIES:
        start = half // 5 if changed else 0
        names = [f"{label}::t{index}" for index in range(start, half)]
        if changed:
            names += [f"{label}::new{index}" for index in range(start)]
        rows += [
            {"task_id": name, ENVIRONMENT_COLUMN: label, _COLUMN: task_labels[index % len(task_labels)]}
            for index, name in enumerate(names)
        ]
    return rows


def _infos(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from platoon.train.components import ENVIRONMENT_COLUMN
    from platoon.train.task_stats import EntryInfo

    infos = {}
    for index, (label, size, _task_labels, weight, seed) in enumerate(_ENTRIES):
        ids = tuple(str(row["task_id"]) for row in rows if row[ENVIRONMENT_COLUMN] == label)
        infos[label] = EntryInfo(
            group_size=size, success_threshold=1.0, name=label, weight=weight, seed=seed, index=index, task_ids=ids
        )
    return infos


def _records(rows: Sequence[Mapping[str, Any]], infos: Mapping[str, Any]) -> list[Any]:
    """Synthetic settled groups of the pool's tasks: each task's members pass at a rate of its own, some members
    have no valid result, some groups failed; a few tasks have none yet."""

    from platoon.train.components import ENVIRONMENT_COLUMN
    from platoon.train.task_stats import MAIN, GroupRecord, MemberResult

    rng = random.Random(20261021)
    records = []
    seq = 0
    for row in rows:
        task, entry = str(row["task_id"]), str(row[ENVIRONMENT_COLUMN])
        rate = rng.choice((0.0, 0.05, 0.3, 0.5, 0.7, 0.95, 1.0))
        size = infos[entry].group_size
        for group in range(rng.randrange(0, 4)):
            failed = rng.random() < 0.1
            members = []
            for _ in range(size):
                valid = not failed and rng.random() > 0.15
                scored = (1.0 if rng.random() < rate else 0.0) if valid else None
                members.append(
                    MemberResult(
                        reward=scored,
                        scored=scored,
                        valid=valid,
                        outcome="scored" if valid else "error",
                        disposition="completed",
                        reward_key=None,
                        steps=3,
                        output_tokens=10,
                        seconds=1.0,
                        versions={MAIN: (group, group)},
                    )
                )
            passes = sum(member.passes(1.0) for member in members)
            valid_count = sum(member.valid for member in members)
            verdict = "failed" if failed else "uniform" if passes in (0, valid_count) else "trained"
            records.append(
                GroupRecord(
                    start=1,
                    seq=seq,
                    task_id=task,
                    entry=entry,
                    labels={_COLUMN: str(row[_COLUMN])},
                    harness=None,
                    group_size=size,
                    sampler="conformance",
                    reason="order",
                    score=None,
                    position=seq,
                    drawn_at={MAIN: group},
                    members=tuple(members),
                    verdict=verdict,
                )
            )
            seq += 1
    return records


class _Statistics:
    """What a sampler set reads of a run's statistics (a TaskRecorder's ``store``, never stopped)."""

    def __init__(self, store: Any) -> None:
        self.store = store
        self.stopped = False

    def listen(self, listener: Callable[[Any], None]) -> None:  # the check hands groups to observe itself
        pass


def _store(rows: Sequence[Mapping[str, Any]], infos: Mapping[str, Any], records: Sequence[Any]) -> Any:
    from platoon.train.components import ENVIRONMENT_COLUMN
    from platoon.train.task_stats import TaskStatsStore

    store = TaskStatsStore(
        [(str(row["task_id"]), str(row[ENVIRONMENT_COLUMN]), str(row[_COLUMN])) for row in rows],
        infos,
        column=_COLUMN,
        group_size=4,
    )
    for record in records:
        store.draw(record.start, record.seq, record.task_id, record.position)
        store.settle(record)
        store.fate(record.start, record.seq, "consumed")
    # Some draws outstanding: one of every seventh task, after its groups.
    for index, row in enumerate(rows):
        if index % 7 == 3:
            store.draw(2, index, str(row["task_id"]), None)
    return store


_FIXTURES: dict[tuple[int, bool], tuple[list[dict[str, Any]], dict[str, Any], list[Any], Any]] = {}


def _fixture(tasks: int, *, changed: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any], list[Any], Any]:
    """The fixture's rows, entries, settled groups and statistics; built once per size (a sampler never changes the
    store: it reads it through a read-only view)."""

    key = (tasks, changed)
    if key not in _FIXTURES:
        rows = _rows(tasks, changed=changed)
        infos = _infos(rows)
        records = _records(rows, infos)
        _FIXTURES[key] = (rows, infos, records, _store(rows, infos, records))
    return _FIXTURES[key]


def _sampler_set(case: Case, tasks: int, *, changed: bool = False) -> Any:
    from platoon.train.sampler_api import SamplerSetup
    from platoon.train.sampler_runtime import SamplerSet, TaskPool, _Member
    from platoon.train.samplers import UniformSampler, WeightedSampler

    rows, infos, _records_, store = _fixture(tasks, changed=changed)
    pool = TaskPool.from_rows(rows, entries=[label for label, *_ in _ENTRIES], column=_COLUMN)

    def member(entry: str | None, build: Callable[[Any, Any], Any], options: Any, name: str) -> Any:
        indices = range(len(pool)) if entry is None else pool.entry_indices(entry)
        info = infos.get(entry) if entry is not None else None
        seed = info.seed if info is not None else 0
        extra = getattr(options, "seed", None)
        if isinstance(extra, int) and not isinstance(extra, bool):
            seed += extra
        setup = SamplerSetup(
            name=name if entry is None else f"{name}@{entry}",
            level="run" if entry is None else "entry",
            entry=entry,
            task_ids=tuple(pool.task_id(index) for index in indices),
            entries=infos,
            batch_size=8,
            label_column=_COLUMN,
            seed=seed,
        )
        return _Member(
            sampler=build(setup, options),
            setup=setup,
            options=options,
            indices=indices,
            seed=seed,
            rebuild=lambda: build(setup, options),
        )

    target = case.target
    name = target.ref or "conformance"
    if case.entry is None:
        run = member(None, target.build, case.options, name)
        entries = {}
        if getattr(type(run.sampler), "entry_slots", False) is True:
            entries = {label: member(label, UniformSampler, None, "uniform") for label, *_ in _ENTRIES}
    else:
        run = member(None, WeightedSampler, None, "weighted")
        entries = {label: member(label, target.build, case.options, name) for label, *_ in _ENTRIES}
    samplers = SamplerSet(pool, run, entries, infos, label_column=_COLUMN)
    samplers.attach(_Statistics(store))
    return samplers


# --- the patterns of eligibility, and asking ---------------------------------------------------------------------


def _patterns(adaptive: bool) -> list[tuple[str, Callable[[Any], bool] | None]]:
    """(name, room) for each eligibility pattern: room None asks as a static sampler is asked."""

    if not adaptive:
        return [("every task eligible", None)]
    return [
        ("every task eligible", lambda label: True),
        ("label x at its limit, which leaves entry a its w tasks", lambda label: label != "x"),
        ("nothing eligible", lambda label: False),
    ]


def _answer(answer: Any) -> Any:
    """An answer as JSON data, to compare and to print."""

    from platoon.train.sampler_api import Draw, Slot
    from platoon.train.sampler_runtime import _Row

    if answer is None:
        return None
    if isinstance(answer, str):
        return {"task": answer}
    if isinstance(answer, Draw):
        return {"task": answer.task_id, "reason": answer.reason, "score": answer.score}
    if isinstance(answer, Slot):
        return {"slot": answer.entry}
    if isinstance(answer, _Row):
        if answer.index is None:
            return {"row": json.dumps(dict(answer.row or {}), sort_keys=True, default=repr), "reason": answer.reason}
        return {"row": answer.index, "reason": answer.reason}
    return {"other": repr(answer)[:200]}


@dataclass
class _Asked:
    """What a series of decisions answered (as JSON data), and what went wrong."""

    answers: list[Any]
    problems: list[str]


def _ask(
    samplers: Any,
    member: Any,
    room: Callable[[Any], bool] | None,
    count: int,
    *,
    observe: Sequence[Any] = (),
    every: int = 0,
) -> _Asked:
    """Ask ``member`` ``count`` decisions as the dispatcher would (its count advances when its answer binds), each
    answer judged by the dispatcher's rules; with ``observe``, hand it those settled groups, ``every`` of them between
    two decisions, as the recorder does."""

    answers: list[Any] = []
    problems: list[str] = []
    adaptive = member.adaptive
    pending = list(observe)
    for _ in range(count):
        if pending and every:
            if member.observes:  # as the dispatcher, which hands groups only to a sampler that defines observe
                try:
                    for record in pending[:every]:
                        member.sampler.observe(record)
                except Exception as exc:
                    problems.append(f"observe() raised {type(exc).__name__}: {exc}")
                    break
            del pending[:every]
        ctx = samplers.context(member, room=room if adaptive else None)
        try:
            answer = member.sampler.next(ctx)
        except Exception as exc:
            problems.append(f"next() raised at decision {ctx.decision}: {type(exc).__name__}: {exc}")
            break
        if inspect.isawaitable(answer):
            close = getattr(answer, "close", None)
            if callable(close):
                close()
            problems.append("next() returned an awaitable: a decision is one synchronous call")
            break
        answers.append(_answer(answer))
        eligible = [task for task in member.setup.task_ids if ctx.eligible(task)] if adaptive else member.setup.task_ids
        if answer is None:
            if not adaptive:
                problems.append(
                    f"a static sampler answered None at decision {ctx.decision}: its every task is eligible"
                )
            elif eligible:
                problems.append(f"answered None at decision {ctx.decision} while {len(eligible)} tasks were eligible")
            continue
        if adaptive and not eligible:
            problems.append(f"answered {_brief(answer)} at decision {ctx.decision} while nothing was eligible")
            continue
        if not samplers.accepts(member, answer, ctx):
            problems.append(
                f"answered {_brief(answer)} at decision {ctx.decision}, which the dispatcher refuses (a task outside "
                "its pool or not eligible, a slot it may not give, a reason that is not a short non-empty string, a "
                "score that is neither None nor finite)"
            )
            continue
        member.decisions += 1
    return _Asked(answers, problems)


def _brief(answer: Any) -> str:
    text = repr(answer)
    return text if len(text) <= 120 else text[:117] + "..."


def _first(problems: Sequence[str], most: int = 3) -> list[str]:
    """The first few problems of a series, and how many more."""

    shown = list(dict.fromkeys(problems))
    return shown[:most] + ([f"... and {len(shown) - most} more"] if len(shown) > most else [])


# --- the checks ------------------------------------------------------------------------------------------------------


def _protocol(target: _Target, levels: Sequence[str], *, internal: bool = False) -> list[str]:
    """``protocol``: it resolves; it has ``next``, ``state_dict``, ``load_state_dict`` and a bool ``adaptive``; it
    hands slots (``entry_slots``) only as a static run sampler; ``next`` is no coroutine function."""

    from platoon.extensions import ExtensionError
    from platoon.train.sampler_api import hands_slots, sampler_problem

    if target.problem is not None:
        return [target.problem]
    cls = target.cls
    if cls is None:
        return [f"{target.name} is no class"]
    problems = []
    if not internal:
        try:
            sampler_problem(cls, target.name)
        except ExtensionError as exc:
            problems.append(str(exc))
    if inspect.iscoroutinefunction(getattr(cls, "next", None)):
        problems.append("its next(ctx) is a coroutine function: a decision is one synchronous call")
    if hands_slots(cls) and "entry" in levels:
        problems.append(
            "it hands slots to entries (entry_slots), which only a run's sampler does: check it at the run level only"
        )
    for level in levels:
        if level not in ("run", "entry"):
            problems.append(f"levels: {level!r} is no level (run, entry)")
    return problems


def _options(target: _Target, options: Mapping[str, Any] | None) -> tuple[Any, list[str]]:
    """``options``: its schema takes no options (``{}``) and the options given; a dataclass's or a pydantic model's
    refuses an unknown key.  Returns the options given, parsed."""

    from platoon.options import schema_options
    from platoon.train.sampler_api import options_schema
    from platoon.utils.config import _is_pydantic

    schema = options_schema(target.cls) if target.cls is not None else None
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


def check_pool_task_or_nothing(case: Case) -> list[str]:
    """``pool-task-or-nothing`` (rule 1): under each eligibility pattern, :data:`DECISIONS` decisions, each a task of
    its pool (eligible, for an adaptive sampler) or, from a static run sampler that hands slots, a slot; None only when
    nothing is eligible, and nothing else then."""

    problems = []
    patterns = _patterns(case.adaptive)
    for pattern, room in patterns:
        try:
            samplers, member = case.build()
        except Exception as exc:
            return [f"building it raised {type(exc).__name__}: {exc}"]
        asked = _ask(samplers, member, room, DECISIONS)
        problems += [f"{pattern}: {problem}" for problem in _first(asked.problems)]
    if case.adaptive and not problems:
        # One instance under every pattern in turn: a decision's eligibility is its own, never an earlier one's.
        samplers, member = case.build()
        switched: list[str] = []
        for turn in range(DECISIONS // _SWITCH_EVERY):
            pattern, room = patterns[turn % len(patterns)]
            asked = _ask(samplers, member, room, _SWITCH_EVERY)
            switched += [f"{pattern}, after decisions under other patterns: {problem}" for problem in asked.problems]
        problems += [f"one instance under the patterns in turn: {problem}" for problem in _first(switched)]
    return problems


def check_fast(case: Case, budget_ms: float) -> list[str]:
    """``fast`` (rule 2): ``next`` is synchronous, and the median of :data:`TIMED_DECISIONS` decisions on a pool of
    :data:`TIMED_TASKS` tasks (the run's, or at the entry level the entry's), every one eligible, stays within
    ``budget_ms``."""

    if inspect.iscoroutinefunction(getattr(case.target.cls, "next", None)):
        return ["its next(ctx) is a coroutine function"]
    try:
        # The fixture gives each entry half of its tasks: an entry's sampler gets a pool of TIMED_TASKS of its own.
        samplers, member = case.build(TIMED_TASKS if case.entry is None else 2 * TIMED_TASKS)
    except Exception as exc:
        return [f"building it on {TIMED_TASKS} tasks raised {type(exc).__name__}: {exc}"]
    room = (lambda label: True) if member.adaptive else None
    times = []
    for _ in range(TIMED_DECISIONS):
        ctx = samplers.context(member, room=room)
        started = time.perf_counter()
        try:
            answer = member.sampler.next(ctx)
        except Exception as exc:
            return [f"next() raised on {TIMED_TASKS} tasks: {type(exc).__name__}: {exc}"]
        times.append(1000.0 * (time.perf_counter() - started))
        if answer is not None:
            member.decisions += 1
    median = statistics.median(times)
    if median > budget_ms:
        return [
            f"a decision on {TIMED_TASKS} tasks takes {median:.1f} ms (the median of {TIMED_DECISIONS}), more than "
            f"the budget of {budget_ms:g} ms: read counts(), not task(), for every task"
        ]
    return []


def check_deterministic(case: Case) -> list[str]:
    """``deterministic`` (rule 3): two fresh instances fed the same contexts answer alike; given by reference, so do
    two processes with other hash seeds."""

    first = _replay(case)
    second = _replay(case)
    problems = []
    if first != second:
        problems.append(f"two fresh instances answered differently: {_difference(first, second)}")
    if case.target.ref is None:
        return problems
    for seed in ("0", "1"):
        try:
            other = _replay_elsewhere(case, seed)
        except RuntimeError as exc:
            problems.append(str(exc))
            continue
        if other != first:
            problems.append(f"a process with PYTHONHASHSEED={seed} answered differently: {_difference(first, other)}")
    return problems


def _replay(case: Case) -> list[Any]:
    """The answers to :data:`REPLAYED` decisions under each eligibility pattern but the last, from a fresh instance."""

    answers: list[Any] = []
    for _pattern, room in _patterns(case.adaptive)[:2]:
        samplers, member = case.build()
        answers += _ask(samplers, member, room, REPLAYED).answers
    return answers


def _replay_elsewhere(case: Case, seed: str) -> list[Any]:
    request = {
        "ref": case.target.ref,
        "module": case.target.module,
        "options": dict(case.written),
        "level": case.level,
        "entry": case.entry,
    }
    environment = {**os.environ, "PYTHONHASHSEED": seed}
    done = subprocess.run(
        [sys.executable, "-m", "platoon.testing.samplers", json.dumps(request)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
    )
    if done.returncode != 0:
        raise RuntimeError(
            f"replaying its decisions in a process with PYTHONHASHSEED={seed} failed: "
            f"{(done.stderr or done.stdout).strip()[-400:]}"
        )
    return json.loads(done.stdout)


def _difference(first: Sequence[Any], second: Sequence[Any]) -> str:
    for index, (one, other) in enumerate(zip(first, second)):
        if one != other:
            return f"decision {index}: {one!r} against {other!r}"
    return f"{len(first)} answers against {len(second)}"


def check_static(case: Case) -> list[str]:
    """``static`` (rule 4): a static sampler's contexts have no statistics, and a fresh instance asked at decision
    ``n`` alone, at another training step and policy version than the series, answers what a series answered at
    ``n``: it depends on nothing but its decision index and its state (a resume recomputes its counts at step 0)."""

    if case.adaptive:
        return []
    samplers, member = case.build()
    try:
        member.sampler.next(samplers.context(member))
    except Exception as exc:
        return [
            f"asked as a static sampler is asked (its context's statistics are None) it raised {type(exc).__name__}: "
            f"{exc}"
        ]
    samplers, member = case.build()
    asked = _ask(samplers, member, None, DECISIONS)
    if asked.problems:
        return []  # pool-task-or-nothing reports them
    problems = []
    from platoon.train.task_stats import MAIN

    for decision in random.Random(4).sample(range(DECISIONS), 12):
        fresh_samplers, fresh = case.build()
        fresh.decisions = decision
        ctx = fresh_samplers.context(fresh, step=decision + 3, versions={MAIN: decision + 5})
        try:
            answer = _answer(fresh.sampler.next(ctx))
        except Exception as exc:
            problems.append(f"a fresh instance asked at decision {decision} raised {type(exc).__name__}: {exc}")
            continue
        if answer != asked.answers[decision]:
            problems.append(
                f"a fresh instance asked at decision {decision} alone answered {answer!r}, the series "
                f"{asked.answers[decision]!r}"
            )
    return _first(problems)


def _string_keys(value: Any, where: str = "state") -> list[str]:
    if isinstance(value, Mapping):
        found = [f"{where} has a key that is no string: {key!r}" for key in value if not isinstance(key, str)]
        for key, item in value.items():
            found += _string_keys(item, f"{where}[{key!r}]")
        return found
    if isinstance(value, (list, tuple)):
        return [problem for index, item in enumerate(value) for problem in _string_keys(item, f"{where}[{index}]")]
    return []


def _malformed(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """States no sampler should take: an unknown key in place of its own, and its own keys with values of the wrong
    kind."""

    found: list[dict[str, Any]] = [{"conformance_check": "not a state"}]
    if state:
        found.append({key: "not a value of this state" for key in state})
    return found


def check_state(case: Case) -> list[str]:
    """``state`` (rule 5): after some decisions (an adaptive sampler handed the fixture's settled groups between them,
    as it observes them in a run) its ``state_dict()`` is JSON with string keys; a fresh instance that loads it answers
    the next decisions alike, so what it learned from observing is in its state; one over a changed pool loads it and
    answers within its pool; and a malformed state raises ``ValueError``, nothing else."""

    samplers, member = case.build()
    room = (lambda label: True) if member.adaptive else None
    _rows_, _infos_, records, _store_ = _fixture(TASKS)
    observed = sorted((record for record in records if member.holds(record.task_id)), key=_draw_order)
    if _ask(samplers, member, room, 37, observe=observed if member.adaptive else (), every=4).problems:
        return []  # pool-task-or-nothing reports them
    try:
        state = member.sampler.state_dict()
    except Exception as exc:
        return [f"state_dict() raised {type(exc).__name__}: {exc}"]
    if not isinstance(state, Mapping):
        return [f"state_dict() returned a {type(state).__name__}, not a mapping"]
    try:
        saved = json.loads(json.dumps(state, allow_nan=False))
    except (TypeError, ValueError) as exc:
        return [f"state_dict() is not JSON data: {exc}"]
    problems = _string_keys(state)
    fresh_samplers, fresh = case.build()
    fresh.decisions = member.decisions
    try:
        fresh.sampler.load_state_dict(saved)
    except Exception as exc:
        problems.append(f"a fresh instance could not load its state: {type(exc).__name__}: {exc}")
    else:
        ahead = _ask(samplers, member, room, 20).answers
        again = _ask(fresh_samplers, fresh, room, 20).answers
        if ahead != again:
            problems.append(f"a fresh instance that loaded its state answered differently: {_difference(ahead, again)}")
    changed_samplers, changed = case.build(changed=True)
    changed.decisions = member.decisions
    try:
        changed.sampler.load_state_dict(saved)
    except Exception as exc:
        problems.append(f"loading its state over a changed pool raised {type(exc).__name__}: {exc}")
    else:
        problems += [
            f"over a changed pool: {problem}" for problem in _first(_ask(changed_samplers, changed, room, 20).problems)
        ]
    for malformed in _malformed(state):
        broken_samplers, broken = case.build()
        try:
            broken.sampler.load_state_dict(malformed)
        except ValueError:
            continue
        except Exception as exc:
            problems.append(
                f"loading the malformed state {malformed!r} raised {type(exc).__name__}, not ValueError: a state that "
                "does not fit raises ValueError, so that a resume starts the sampler afresh"
            )
            continue
        problems.append(f"loading the malformed state {malformed!r} raised nothing: it should raise ValueError")
    return problems


def check_observe(case: Case) -> list[str]:
    """``observe`` (rule 6): an adaptive sampler is handed the fixture's settled groups in ``(start, seq)`` order
    between decisions, and runs as well without them."""

    if not case.adaptive:
        return []
    _rows_, _infos_, records, _store_ = _fixture(TASKS)
    ordered = sorted(records, key=_draw_order)
    problems = []
    samplers, member = case.build()
    holds = [record for record in ordered if member.holds(record.task_id)]
    try:
        asked = _ask(samplers, member, lambda label: True, 30, observe=holds, every=4)
        problems += [f"with observe: {problem}" for problem in _first(asked.problems)]
    except Exception as exc:
        problems.append(f"observe() raised {type(exc).__name__}: {exc}")
    bare_samplers, bare = case.build()
    problems += [
        f"without observe: {problem}" for problem in _first(_ask(bare_samplers, bare, lambda label: True, 30).problems)
    ]
    return problems


def _draw_order(record: Any) -> tuple[int, int]:
    return (record.start, record.seq)


def check_metrics(case: Case) -> list[str]:
    """``metrics``: ``metrics(stats)`` with the statistics' view and with None returns a mapping of finite numbers,
    none of them under a key the dispatcher writes itself under the sampler's prefix (its draws, ``draws/...``, its
    decision time, an entry's waiting slots, ``predicted_informative``, ``informative_share``)."""

    from platoon.train.sampler_api import RESERVED_METRICS, reserved_metric

    samplers, member = case.build()
    _ask(samplers, member, (lambda label: True) if member.adaptive else None, 20)
    problems = []
    for name, stats in (("the statistics' view", samplers.view(0)), ("None", None)):
        try:
            values = member.sampler.metrics(stats)
        except Exception as exc:
            problems.append(f"metrics({name}) raised {type(exc).__name__}: {exc}")
            continue
        if not isinstance(values, Mapping):
            problems.append(f"metrics({name}) returned a {type(values).__name__}, not a mapping")
            continue
        for key, value in values.items():
            if (
                not isinstance(key, str)
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                problems.append(f"metrics({name}) gave {key!r}: {value!r}, not a finite number")
            elif reserved_metric(key):
                problems.append(
                    f"metrics({name}) gave {key!r}, a key the dispatcher writes itself under the sampler's prefix "
                    f"({', '.join(sorted(RESERVED_METRICS))}, draws/<reason>): name it otherwise, or the run leaves it "
                    "out"
                )
    return problems


# --- the replay in another process -----------------------------------------------------------------------------------


def _main(argv: Sequence[str]) -> int:
    """``python -m platoon.testing.samplers '<request>'``: print, as JSON, the answers a sampler gives to the
    determinism check's decisions (its replay in a process with another hash seed)."""

    import importlib

    from platoon.options import schema_options
    from platoon.train.sampler_api import options_schema

    request = json.loads(argv[0])
    if request.get("module"):
        importlib.import_module(request["module"])
    target = _target(request["ref"])
    if target.cls is None:
        print(target.problem, file=sys.stderr)
        return 1
    parsed = schema_options(options_schema(target.cls), name=target.name).parse(request["options"], "options")
    case = Case(target, parsed, request["options"], request["level"], request["entry"], target.name)
    print(json.dumps(_replay(case)))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
