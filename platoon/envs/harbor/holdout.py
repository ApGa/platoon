"""Hold out whole groups of a dataset's tasks for evaluation: ``service_kwargs.holdout``.

    holdout: {fraction: 0.05, group_by: [kaggle_dataset_name, split_group], stratify_by: source_dataset, seed: 0}

The rule runs over a dataset's selected tasks, refused ones included, and reads their index fields and input files:

- **Strata.** Tasks fall into strata by their ``stratify_by`` field (one stratum without it).
- **Groups.** Tasks fall into groups by the first ``group_by`` field they have, so a dataset whose sources record
  their groups under different names is one rule.  Then groups whose tasks share an input file (byte for byte, or as
  text that differs only in line endings, a byte-order mark or blanks) are joined into one, also through other groups
  (a group that shares a file with each of two others joins all three): tasks that hold the same data always land on
  the same side, so no held-out task measures data that training has seen, whatever ``group_by`` names.  A joined
  group takes the name of its member with the most tasks.
- **The walk.** Within each stratum, groups are ranked by the SHA-256 of ``"<seed>\\0<group>"``.  Walking that order,
  a group is held out while the stratum's held-out tasks, with the group's, number at most
  ⌈``fraction`` × the stratum's selected tasks⌉.  A group that does not fit is passed over, and no group is split:
  one whose tasks fall in several strata must fit each of their shares, and is held out in all of them or in none.
- **The lists.** The ``eval`` split lists the held-out tasks that are runnable, and ``train`` the other runnable
  ones, each in the order of the SHA-256 of ``"<seed>\\0<task id>"``, so that a ``limit`` takes a spread-out sample.

**Input files** (:mod:`platoon.envs.harbor.inputs`) are the files a task's agents' sandbox takes from the task: the
build-context files its Dockerfile copies into the image, whether a build bakes them in or a session uploads them,
or a prebuilt-image task's ``environment/``, never the Dockerfile itself, the tests or the solution.  An empty file
joins nothing, and neither does a file ``ignore_inputs`` names: files that every task holds by design rather than
as data, such as a game's code and rules.  A fetch records each task's input files in the lock; a dataset
directory's are hashed when its tasks are listed.  A task whose input files are not known (a lock written before
platoon recorded them) is an error, never a task without inputs.

Refused tasks take part in the walk, their input files included, and then leave both lists, so changing a refusal
never moves a group.  A runnable task without the stratum field, or without any of the group fields, is an error; a
refused one sits out.  The same rows and rule give the same split anywhere: rollout workers receive task ids and
compute nothing.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from fractions import Fraction
from typing import Any

from platoon.envs.harbor.inputs import EMPTY_FILE

TRAIN = "train"
EVAL = "eval"
SPLITS = (TRAIN, EVAL)
_KEYS = ("fraction", "group_by", "stratify_by", "seed", "ignore_inputs")
_SHOWN = 3  # ids an error names


class HoldoutError(ValueError):
    """A hold-out that cannot be computed: a malformed rule, runnable tasks without the fields it names, or tasks
    whose input files are not known."""


@dataclass(frozen=True)
class HoldoutRule:
    """``holdout``: the share of each stratum to hold out, the fields that group and stratify tasks, the seed, and
    the input files that join no groups."""

    # The share of each stratum's selected tasks to hold out, at most (0 < fraction < 1).
    fraction: float
    # The index fields that name a task's group: the first one a task has.
    group_by: tuple[str, ...]
    # The index field whose values each get their own share (None: the whole dataset is one stratum).
    stratify_by: str | None = None
    # Ranks groups and orders the lists.
    seed: int = 0
    # Input files that join no groups, as patterns of task-relative paths (``environment/rules.json``; fnmatch, so
    # ``*`` matches ``/`` too): files that every task holds by design rather than as data, such as a game's code.
    ignore_inputs: tuple[str, ...] = ()

    @classmethod
    def coerce(cls, value: HoldoutRule | Mapping[str, Any]) -> HoldoutRule:
        """A rule from config: ``{fraction, group_by, stratify_by, seed, ignore_inputs}``."""

        if isinstance(value, HoldoutRule):
            return value
        if not isinstance(value, Mapping):
            raise HoldoutError(f"holdout must be a mapping of {', '.join(_KEYS)}, got {value!r}")
        if unknown := sorted(set(value) - set(_KEYS)):
            raise HoldoutError(f"holdout has unknown keys {unknown}; it takes {', '.join(_KEYS)}")
        fraction = value.get("fraction")
        if isinstance(fraction, bool) or not isinstance(fraction, int | float) or not 0 < fraction < 1:
            raise HoldoutError(f"holdout.fraction must be a number between 0 and 1, got {fraction!r}")
        group_by = value.get("group_by")
        names = [group_by] if isinstance(group_by, str) else group_by
        if not isinstance(names, list | tuple) or not names or not all(isinstance(n, str) and n for n in names):
            raise HoldoutError(f"holdout.group_by must name an index field, or list them, got {group_by!r}")
        stratify_by = value.get("stratify_by")
        if stratify_by is not None and (not isinstance(stratify_by, str) or not stratify_by):
            raise HoldoutError(f"holdout.stratify_by must name an index field, got {stratify_by!r}")
        seed = value.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise HoldoutError(f"holdout.seed must be an integer, got {seed!r}")
        ignore = value.get("ignore_inputs", ())
        patterns = [ignore] if isinstance(ignore, str) else ignore
        if not isinstance(patterns, list | tuple) or not all(isinstance(p, str) and p for p in patterns):
            raise HoldoutError(f"holdout.ignore_inputs must list task-relative path patterns, got {ignore!r}")
        return cls(
            fraction=float(fraction),
            group_by=tuple(names),
            stratify_by=stratify_by,
            seed=seed,
            ignore_inputs=tuple(dict.fromkeys(patterns)),
        )

    def to_json(self) -> dict[str, Any]:
        """The rule as a run records it; ``ignore_inputs`` only when set."""

        data: dict[str, Any] = {
            "fraction": self.fraction,
            "group_by": list(self.group_by),
            "stratify_by": self.stratify_by,
            "seed": self.seed,
        }
        if self.ignore_inputs:
            data["ignore_inputs"] = list(self.ignore_inputs)
        return data

    def describe(self) -> str:
        """``5% of each source_dataset, grouped by kaggle_dataset_name, else split_group, joined by shared input
        files, seed 0``."""

        where = f"each {self.stratify_by}" if self.stratify_by else "the dataset"
        joined = "joined by shared input files"
        if self.ignore_inputs:
            joined += f" other than {', '.join(self.ignore_inputs)}"
        fields = ", else ".join(self.group_by)
        return f"{self.fraction * 100:.6g}% of {where}, grouped by {fields}, {joined}, seed {self.seed}"

    def target(self, selected: int) -> int:
        """⌈fraction × selected⌉, computed on the decimal the config wrote (0.07 × 100 is 7, not 8)."""

        return math.ceil(Fraction(str(self.fraction)) * selected)


@dataclass(frozen=True)
class HoldoutRow:
    """A selected task: its id, its index fields, why it cannot run (None: runnable), and its input files."""

    task_id: str
    fields: Mapping[str, Any]
    refused: str | None = None
    # Task-relative path -> "sha256:<hex>" (platoon.envs.harbor.inputs.input_files); None when they are not known,
    # which a hold-out refuses.
    input_files: Mapping[str, str] | None = None


@dataclass(frozen=True)
class StratumReport:
    """What the hold-out did in one stratum."""

    # The stratify_by value (None without stratify_by).
    stratum: str | None
    # The stratum's tasks in the walk, refused ones included.
    selected: int
    runnable: int
    # Held-out tasks, refused ones included; ``eval`` of them are runnable.
    held_out: int
    eval: int
    # The groups of the evaluation's tasks, and the largest of them with its number of tasks.
    eval_groups: int
    largest_group: str | None
    largest_group_tasks: int
    train: int
    # Refused tasks by the reason's kind (``uploads_over_mb``, ``task_config``, ...).
    refused: Mapping[str, int]

    @property
    def largest_share(self) -> float:
        return self.largest_group_tasks / self.eval if self.eval else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "stratum": self.stratum,
            "selected": self.selected,
            "runnable": self.runnable,
            "held_out": self.held_out,
            "eval": self.eval,
            "eval_groups": self.eval_groups,
            "largest_group": self.largest_group,
            "largest_group_tasks": self.largest_group_tasks,
            "train": self.train,
            "refused": dict(self.refused),
        }


@dataclass(frozen=True)
class GroupsReport:
    """How the walk's groups came about: by the ``group_by`` fields, then joined where their tasks share input
    files."""

    # Groups by the group_by fields, how many of them shared input files joined, into how many groups, and the groups
    # of the walk after joining.
    by_fields: int
    joined_groups: int
    joined_into: int
    groups: int
    # Distinct input files that may join groups (neither empty nor ignored), those that the tasks of two or more
    # group_by groups share, and those ignore_inputs matched.
    distinct_files: int
    shared_files: int
    ignored_files: int
    # The largest group of the walk: its name, tasks (refused ones included) and how many group_by groups it joined.
    largest: str | None = None
    largest_tasks: int = 0
    largest_members: int = 0
    # For a joined largest group: how many shared files joined it, and the one most of its tasks hold, with their
    # number.
    largest_shared_files: int = 0
    largest_example: str | None = None
    largest_example_tasks: int = 0

    def line(self, rule: HoldoutRule) -> str:
        """``groups: 563 by kaggle_dataset_name, else split_group; 43 shared input file(s) joined 90 of them into 30,
        leaving 503; largest gregorut/videogamesales: 346 task(s)``, with what joined the largest group when it is a
        joined one."""

        line = f"  groups: {self.by_fields:,} by {', else '.join(rule.group_by)}; "
        if self.joined_groups:
            line += (
                f"{self.shared_files:,} shared input file(s) joined {self.joined_groups:,} of them into "
                f"{self.joined_into:,}, leaving {self.groups:,}"
            )
        else:
            line += "no two of them share an input file"
        if self.ignored_files:
            line += f" ({self.ignored_files:,} input file(s) ignored)"
        if self.largest is not None:
            line += f"; largest {self.largest}: {self.largest_tasks:,} task(s)"
            if self.largest_members > 1:
                line += (
                    f", {self.largest_members:,} groups joined by {self.largest_shared_files:,} file(s) such as "
                    f"{self.largest_example} (in {self.largest_example_tasks:,} task(s))"
                )
        return line

    def to_json(self) -> dict[str, Any]:
        return {
            "by_fields": self.by_fields,
            "joined_groups": self.joined_groups,
            "joined_into": self.joined_into,
            "groups": self.groups,
            "distinct_files": self.distinct_files,
            "shared_files": self.shared_files,
            "ignored_files": self.ignored_files,
            "largest": self.largest,
            "largest_tasks": self.largest_tasks,
            "largest_members": self.largest_members,
            "largest_shared_files": self.largest_shared_files,
            "largest_example": self.largest_example,
            "largest_example_tasks": self.largest_example_tasks,
        }


@dataclass(frozen=True)
class HoldoutSplit:
    """The ``train`` and ``eval`` lists of a hold-out, and the report of how they were drawn."""

    rule: HoldoutRule
    train: tuple[str, ...]
    eval: tuple[str, ...]
    strata: tuple[StratumReport, ...]
    # Refused tasks without the rule's fields, which sat out of the walk.
    sat_out: int = 0
    # Every held-out task, refused ones included: what a run records, so that a refusal that changes leaves its
    # held-out tasks where they were.
    held_out: tuple[str, ...] = ()
    # How the groups came about (None: no task in the walk).
    groups: GroupsReport | None = None

    def tasks(self, split: str) -> tuple[str, ...]:
        """The runnable tasks of ``split`` (``train`` or ``eval``); ``KeyError`` for another split."""

        if split == TRAIN:
            return self.train
        if split == EVAL:
            return self.eval
        raise KeyError(f"Unknown split {split!r}: a hold-out lists the splits {TRAIN} and {EVAL}")

    @property
    def eval_digest(self) -> str:
        """SHA-256 over the evaluation's task ids, sorted and joined by newlines."""

        return hashlib.sha256("\n".join(sorted(self.eval)).encode()).hexdigest()

    def report(self) -> list[str]:
        """The makeup of each stratum's splits, one line each, then how the groups came about, then the evaluation's
        size and digest."""

        lines = [f"hold-out: {self.rule.describe()}"]
        for stratum in self.strata:
            name = f"{self.rule.stratify_by} {stratum.stratum}" if self.rule.stratify_by else "all tasks"
            line = (
                f"  {name}: {stratum.selected:,} selected, {stratum.runnable:,} runnable; held out "
                f"{stratum.held_out:,}: eval {stratum.eval:,} in {stratum.eval_groups:,} group(s)"
            )
            if stratum.largest_group is not None:
                line += (
                    f" (largest {stratum.largest_group}: {stratum.largest_group_tasks:,}, {stratum.largest_share:.0%})"
                )
            line += f"; train {stratum.train:,}"
            if stratum.refused:
                line += "; refused: " + ", ".join(
                    f"{kind} {count:,}" for kind, count in sorted(stratum.refused.items())
                )
            lines.append(line)
        if self.groups is not None:
            lines.append(self.groups.line(self.rule))
        tail = f"  eval {len(self.eval):,} task(s), train {len(self.train):,}; eval digest {self.eval_digest[:12]}"
        if self.sat_out:
            tail += f"; {self.sat_out:,} refused task(s) without the rule's fields sat out"
        lines.append(tail)
        return lines

    def to_json(self) -> dict[str, Any]:
        return {
            "rule": self.rule.to_json(),
            "eval": len(self.eval),
            "train": len(self.train),
            "eval_digest": self.eval_digest,
            "sat_out": self.sat_out,
            "strata": [stratum.to_json() for stratum in self.strata],
            "groups": self.groups.to_json() if self.groups is not None else None,
        }


def hold_out(rows: Iterable[HoldoutRow], rule: HoldoutRule) -> HoldoutSplit:
    """Split ``rows``, a dataset's selected tasks, by ``rule`` (see the module docstring).

    Raises :class:`HoldoutError` for runnable tasks without the stratum field or without any group field, for tasks
    in the walk whose input files are not known, for an ``ignore_inputs`` pattern that matches no input file, and for
    two rows with one id.
    """

    walk: list[tuple[str | None, str, HoldoutRow]] = []  # (stratum, group by the fields, row)
    no_stratum: list[str] = []
    no_group: list[str] = []
    seen: set[str] = set()
    sat_out = 0
    for row in rows:
        if row.task_id in seen:
            raise HoldoutError(f"two tasks are named {row.task_id!r}")
        seen.add(row.task_id)
        stratum: str | None = None
        if rule.stratify_by is not None:
            value = row.fields.get(rule.stratify_by)
            if value is None:
                if row.refused is None:
                    no_stratum.append(row.task_id)
                else:
                    sat_out += 1
                continue
            stratum = str(value)
        group = next((str(row.fields[name]) for name in rule.group_by if row.fields.get(name) is not None), None)
        if group is None:
            if row.refused is None:
                no_group.append(row.task_id)
            else:
                sat_out += 1
            continue
        walk.append((stratum, group, row))
    if no_stratum:
        raise HoldoutError(
            f"{len(no_stratum)} runnable task(s) have no index field {rule.stratify_by!r} to stratify by, such as "
            f"{sorted(no_stratum)[:_SHOWN]}"
        )
    if no_group:
        raise HoldoutError(
            f"{len(no_group)} runnable task(s) have none of the index fields {list(rule.group_by)} to group by, such "
            f"as {sorted(no_group)[:_SHOWN]}"
        )
    if unknown := sorted(row.task_id for _, _, row in walk if row.input_files is None):
        raise HoldoutError(
            f"{len(unknown)} task(s) have no record of their input files, such as {unknown[:_SHOWN]}; a hold-out keeps "
            "the tasks that share an input file on one side, so it needs them"
        )
    joined, groups_report = _join(walk, rule)
    strata: dict[str | None, dict[str, list[HoldoutRow]]] = {}
    for stratum, group, row in walk:
        strata.setdefault(stratum, {}).setdefault(joined[group], []).append(row)

    # The walk, over every group in the order of its rank: for a group whose tasks sit in one stratum it is that
    # stratum's own walk; a group whose tasks fall in several strata must fit each of their shares, and is held out in
    # all of them or in none, so that no group is split between the lists.
    targets = {stratum: rule.target(sum(map(len, groups.values()))) for stratum, groups in strata.items()}
    counts = dict.fromkeys(strata, 0)
    parts: dict[str, list[tuple[str | None, list[HoldoutRow]]]] = {}
    for stratum, groups in strata.items():
        for group, members in groups.items():
            parts.setdefault(group, []).append((stratum, members))
    held: set[str] = set()
    for group in sorted(parts, key=lambda name: (_rank(rule.seed, name), name)):
        if all(counts[stratum] + len(members) <= targets[stratum] for stratum, members in parts[group]):
            for stratum, members in parts[group]:
                held.update(row.task_id for row in members)
                counts[stratum] += len(members)

    train: list[str] = []
    evaluate: list[str] = []
    reports: list[StratumReport] = []
    for stratum in sorted(strata, key=lambda name: (name is not None, name or "")):
        groups = strata[stratum]
        members = [row for rows_of_group in groups.values() for row in rows_of_group]
        runnable = [row for row in members if row.refused is None]
        stratum_eval = [row.task_id for row in runnable if row.task_id in held]
        stratum_train = [row.task_id for row in runnable if row.task_id not in held]
        eval_groups = Counter(
            group
            for group, rows_of_group in groups.items()
            for row in rows_of_group
            if row.task_id in held and row.refused is None
        )
        largest = min(eval_groups.items(), key=lambda item: (-item[1], item[0])) if eval_groups else None
        reports.append(
            StratumReport(
                stratum=stratum,
                selected=len(members),
                runnable=len(runnable),
                held_out=sum(1 for row in members if row.task_id in held),
                eval=len(stratum_eval),
                eval_groups=len(eval_groups),
                largest_group=largest[0] if largest else None,
                largest_group_tasks=largest[1] if largest else 0,
                train=len(stratum_train),
                refused=dict(Counter(row.refused.split(":", 1)[0] for row in members if row.refused is not None)),
            )
        )
        evaluate += stratum_eval
        train += stratum_train
    return HoldoutSplit(
        rule=rule,
        train=tuple(sorted(train, key=lambda task_id: (_rank(rule.seed, task_id), task_id))),
        eval=tuple(sorted(evaluate, key=lambda task_id: (_rank(rule.seed, task_id), task_id))),
        strata=tuple(reports),
        sat_out=sat_out,
        held_out=tuple(sorted(held, key=lambda task_id: (_rank(rule.seed, task_id), task_id))),
        groups=groups_report if walk else None,
    )


def _join(walk: Iterable[tuple[str | None, str, HoldoutRow]], rule: HoldoutRule) -> tuple[dict[str, str], GroupsReport]:
    """The group each ``group_by`` group joins (its own name when it shares no input file with another), and the
    report of how.  Groups whose tasks share an input file, directly or through other groups, are one; it takes the
    name of its member with the most tasks (on a tie, the first name in sorted order), so that a group that joins no
    other keeps its name and its rank."""

    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(node: tuple[str, str]) -> tuple[str, str]:
        root = parent.setdefault(node, node)
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:  # compress the path
            parent[node], node = root, parent[node]
        return root

    tasks: Counter[str] = Counter()  # tasks per group, refused ones included
    holders: dict[str, set[str]] = defaultdict(set)  # file -> the groups whose tasks hold it
    holding: Counter[str] = Counter()  # file -> the tasks that hold it
    paths: dict[str, str] = {}  # file -> the first of its paths in sorted order
    ignored: set[str] = set()
    matched: set[str] = set()
    for _, group, row in walk:
        tasks[group] += 1
        root = find(("group", group))  # stays the root: each file's root is joined to it
        for path, digest in (row.input_files or {}).items():
            if digest == EMPTY_FILE:
                continue
            if hits := [pattern for pattern in rule.ignore_inputs if fnmatchcase(path, pattern)]:
                matched.update(hits)
                ignored.add(digest)
                continue
            if (other := find(("file", digest))) != root:
                parent[other] = root
            holders[digest].add(group)
            holding[digest] += 1
            paths[digest] = min(path, paths.get(digest, path))
    if unmatched := [pattern for pattern in rule.ignore_inputs if pattern not in matched]:
        raise HoldoutError(f"holdout.ignore_inputs: {unmatched} match no input file of the dataset's selected tasks")

    members: dict[tuple[str, str], list[str]] = defaultdict(list)
    for group in sorted(tasks):
        members[find(("group", group))].append(group)
    joined: dict[str, str] = {}
    sizes: dict[str, int] = {}
    member_count: dict[str, int] = {}
    for groups in members.values():
        name = min(groups, key=lambda group: (-tasks[group], group))
        joined.update(dict.fromkeys(groups, name))
        sizes[name] = sum(tasks[group] for group in groups)
        member_count[name] = len(groups)
    shared = {digest: groups for digest, groups in holders.items() if len(groups) > 1}
    report: dict[str, Any] = {}
    if sizes:
        largest = min(sizes, key=lambda name: (-sizes[name], name))
        report = {"largest": largest, "largest_tasks": sizes[largest], "largest_members": member_count[largest]}
        if member_count[largest] > 1:
            own = [digest for digest, groups in shared.items() if joined[next(iter(groups))] == largest]
            example = min(own, key=lambda digest: (-holding[digest], paths[digest]))
            report.update(
                largest_shared_files=len(own),
                largest_example=paths[example],
                largest_example_tasks=holding[example],
            )
    return joined, GroupsReport(
        by_fields=len(tasks),
        joined_groups=sum(count for count in member_count.values() if count > 1),
        joined_into=sum(1 for count in member_count.values() if count > 1),
        groups=len(sizes),
        distinct_files=len(holders),
        shared_files=len(shared),
        ignored_files=len(ignored),
        **report,
    )


def _rank(seed: int, name: str) -> str:
    return hashlib.sha256(f"{seed}\0{name}".encode()).hexdigest()
