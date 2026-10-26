"""The built-in group admission rules (RFC 0019, section 6), what a config chooses, and the run's decisions at settle.

- :class:`AdmitAll`, ``all`` (the default): it admits every group.  A run in which every entry's rule is ``all`` builds
  no :class:`AdmissionSet` and trains as it did before admission existed; ``AdmitAll`` runs only for an entry at
  ``all`` beside an entry with another rule.
- :class:`Informative`, ``informative`` (experimental): it rejects the groups whose workflow's verdict says they train
  nothing (``uniform``, by success all-pass, all-fail or mixed, and the verdicts of groups that carry no datums), and,
  with a ``band`` (:class:`InformativeOptions`), the groups whose pass rate is outside it; it admits the rest, a group
  whose output carries no verdict included (reason ``unknown``), so that a workflow that gives none never empties a
  step.

What a config chooses (:class:`AdmissionChoice`, :func:`choose_admissions`): ``train.group_admission`` with its options,
and each entry's own ``group_admission`` (an entry that sets none inherits the run's rule and options), each resolved,
its options parsed and its ``check`` run; and ``train.refill.max_groups_per_step``, the cap.  "Rule mode" is a run where
the rule of some entry is not ``all``: its steps train the first ``batch_size`` admitted groups they dequeue that are
not stale, within the cap (:mod:`platoon.train.trainer`, :mod:`platoon.train.dispatch.dispatcher`).  It is decided by
the names the config gives (:func:`is_all`), so that loading a config, which resolves no rule, and its start agree: a
rule named by reference is a rule, even one that resolves to :class:`AdmitAll`.

:class:`AdmissionSet` holds a rule-mode run's instances by entry and makes each decision at settle
(:meth:`AdmissionSet.at_settle`): a failed group is decided ``failed`` without asking the rule; otherwise the rule
decides on the group's record, read against a view of the per-task statistics that never flushes them
(:meth:`~platoon.train.sampler_runtime.SamplerSet.view`), and anything that raises on that path admits the group.  It
also counts what each step dequeued, for the step's ``admission/...`` metrics, and logs the warnings of a short step.

The checks a config's keys get when it loads (the functions the config classes' ``VALUE_PROBLEMS`` call, and
:func:`config_admission_problems`) are here too, with what ``lock.json``, the sampling plan and ``platoon config show``
say of the rules.
"""

from __future__ import annotations

import inspect
import logging
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from platoon.train.admission_api import (
    BUILTIN_GROUP_ADMISSIONS,
    DEFAULT_MAX_GROUPS_PER_STEP,
    Admission,
    AdmissionCheckContext,
    AdmissionRule,
    AdmissionSetup,
    checked_admission,
    groups_per_step,
    max_groups_problem,
    resolve_group_admission,
    rule_options_schema,
)
from platoon.train.sampler_api import TaskStatsView
from platoon.train.task_stats import GroupRecord, TaskStatsStore

if TYPE_CHECKING:
    from platoon.train.sampler_runtime import SamplerSet

logger = logging.getLogger(__name__)

# Core's rules, as a config names them.
RULES = tuple(BUILTIN_GROUP_ADMISSIONS)
# The verdicts of a group that carries no datums, which informative rejects with the verdict as its reason.
NO_DATUM_VERDICTS = ("reward_key_mismatch", "no_results", "below_min", "no_completed_root", "no_datums", "failed")
# informative's reasons for a uniform group, by its kind (GroupRecord.kind).
_UNIFORM_REASONS = {"all_pass": "uniform_pass", "all_fail": "uniform_fail", "uniform_mixed": "uniform_mixed"}
# Each step that keeps waiting warns again as its count passes another multiple of this many batches.
WAITING_BATCHES = 10


def is_all(name: Any) -> bool:
    """Whether a rule, as a config names it, is ``all``, which admits every group: rule mode is a run where some
    entry's rule is not, by name (a reference that resolves to :class:`AdmitAll` is a rule like any other)."""

    return name == "all"


# --- the built-in rules -----------------------------------------------------------------------------------------------


class AdmitAll(AdmissionRule):
    """``all``: every group is admitted."""

    def admit(self, group: GroupRecord, stats: TaskStatsView) -> Admission:
        return Admission(True)


@dataclass(frozen=True)
class InformativeOptions:
    """``informative``'s options (``group_admission_options``)."""

    # Also reject a group whose pass rate, its passes over its valid members (passes as the per-task statistics judge
    # them: the scored reward against the entry's success_threshold), is below lo (reason below_band) or above hi
    # (above_band): two numbers 0 <= lo < hi <= 1, such as [0.2, 0.8]; null: no band.
    band: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        band = self.band
        if band is None:
            return
        if (
            not isinstance(band, (list, tuple))
            or len(band) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in band)
            or not all(math.isfinite(value) for value in band)
            or not 0 <= band[0] < band[1] <= 1
        ):
            shown = list(band) if isinstance(band, (list, tuple)) else band
            raise ValueError(f"band: must be two numbers lo < hi between 0 and 1, got {shown!r}")
        object.__setattr__(self, "band", (float(band[0]), float(band[1])))


class Informative(AdmissionRule):
    """``informative`` (experimental): the groups that carry signal.  In the order it checks:

    1. a group whose verdict is ``uniform`` (every retained advantage zero) is rejected: ``uniform_pass``,
       ``uniform_fail`` or ``uniform_mixed``, by whether every valid member passed, none did, or they were mixed;
    2. a group whose verdict says it carries no datums (``reward_key_mismatch``, ``no_results``, ``below_min``,
       ``no_completed_root``, ``no_datums``, ``failed``) is rejected with the verdict as its reason;
    3. with a ``band``, a group with valid members whose pass rate is below the band or above it is rejected
       (``below_band``, ``above_band``);
    4. a group whose verdict is ``trained`` is admitted; one whose output carries no verdict (``unknown``, or a string
       platoon does not know) is admitted with reason ``unknown``: an admitted group that trains nothing costs a slot,
       while rejecting every group of a workflow that gives no verdicts would train nothing at all.

    A ``trained`` group can be all-pass by success (rewards 1.0 and 0.98 under a threshold of 0.9): only a band
    rejects it."""

    options_schema = InformativeOptions

    def __init__(self, setup: AdmissionSetup, options: Any = None) -> None:
        super().__init__(setup, options if options is not None else InformativeOptions())

    def admit(self, group: GroupRecord, stats: TaskStatsView) -> Admission:
        verdict = group.verdict
        threshold = stats.threshold(group.entry)
        if verdict == "uniform":  # its kind is all_pass, all_fail or uniform_mixed
            return Admission(False, _UNIFORM_REASONS[group.kind(threshold) or "uniform_mixed"])
        if verdict in NO_DATUM_VERDICTS:
            return Admission(False, verdict)
        band = self.options.band
        valid = group.valid
        if band is not None and valid:
            rate = group.passes(threshold) / valid
            if rate < band[0]:
                return Admission(False, "below_band")
            if rate > band[1]:
                return Admission(False, "above_band")
        return Admission(True) if verdict == "trained" else Admission(True, "unknown")


# --- what a config chooses --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AdmissionChoice:
    """What a config chooses at one level, resolved: the rule's name as the config gives it (``all``,
    ``informative``, a registered name, ``module:Class``), its class, its options parsed (None: it takes none), and the
    options as the config wrote them."""

    name: str
    cls: type
    options: Any = None
    written: Mapping[str, Any] = field(default_factory=dict)

    @property
    def admits_all(self) -> bool:
        """Whether it is ``all``, by name (:func:`is_all`)."""

        return is_all(self.name)

    def options_json(self) -> dict[str, Any]:
        """Its options as ``lock.json`` and the sampling fingerprint record them: parsed, defaults included."""

        from platoon.options import schema_options

        return schema_options(rule_options_schema(self.cls), name=self.name).as_json(self.options, self.written)


@dataclass(frozen=True)
class AdmissionChoices:
    """The rules a config chooses (:func:`choose_admissions`): the run's, each entry's effective one by label in config
    order (the run's for an entry that sets none), the entries that set their own, and the cap."""

    run: AdmissionChoice
    entries: Mapping[str, AdmissionChoice]
    own: frozenset[str] = frozenset()
    max_groups_per_step: float = DEFAULT_MAX_GROUPS_PER_STEP

    @property
    def active(self) -> bool:
        """Whether this is rule mode: the rule of some entry (the run's, without entries) is not ``all``, by name, as
        :func:`rule_mode` decides from the config."""

        effective = list(self.entries.values()) or [self.run]
        return any(not choice.admits_all for choice in effective)

    def qualified(self, label: str | None) -> str:
        """The name of the rule that decides ``label``'s groups: the run's as the config names it, an entry's own with
        the entry's label (``informative@hard``)."""

        if label is not None and label in self.own:
            return f"{self.entries[label].name}@{label}"
        return self.run.name


def choose_admissions(
    config: Any, *, workflow: str | None = None, gives_verdicts: bool | None = None, check: bool = True
) -> AdmissionChoices:
    """The group admission rules ``config`` chooses, resolved (core's names first, then names the entries' packages
    register, then ``module:Class``), each class checked, its options parsed against its schema and its ``check`` run
    (unless ``check`` is False: the trainer's start, whose config ``platoon train`` resolved with the run's workflow):
    the run's (``train.group_admission``), and each entry's own (``environments[].group_admission``), an entry that
    sets none taking the run's.  With ``gives_verdicts`` False (the run's workflow, ``workflow``, gives its groups no
    verdict), ``informative`` is refused, since it would admit every group.  A
    :class:`~platoon.utils.config.ConfigError` with every problem, each with its path.  The entries' packages must be
    imported first, as a run imports them."""

    from platoon.extensions import ExtensionError
    from platoon.options import schema_options
    from platoon.train.samplers import _check_entries, _label_column
    from platoon.utils.config import ConfigError

    train = getattr(config, "train", None)
    environments = list(getattr(config, "environments", None) or [])
    problems: list[str] = []
    contexts = _check_entries(config, environments)
    batch_size = int(getattr(train, "batch_size", 0) or 0)
    column = _label_column(config)
    refill = getattr(train, "refill", None)
    cap = getattr(refill, "max_groups_per_step", DEFAULT_MAX_GROUPS_PER_STEP)
    if (problem := max_groups_problem(cap)) is not None:
        problems.append(f"train.refill.max_groups_per_step: {problem}")
        cap = DEFAULT_MAX_GROUPS_PER_STEP
    entry_index = {environment.label: index for index, environment in enumerate(environments)}

    def choose(
        name: Any, written: Mapping[str, Any], level: str, entry: str | None, where: str
    ) -> AdmissionChoice | None:
        if not isinstance(name, str) or not name:
            problems.append(f"{where}: a group admission rule is named by a non-empty string, got {name!r}")
            return None
        try:
            cls = resolve_group_admission(name)
        except ExtensionError as exc:
            problems.append(f"{where}: {exc}")
            return None
        options_where = (
            f"{where}_options" if level == "run" else f"environments.{entry_index[entry]}.group_admission_options"
        )
        before = len(problems)
        parsed, found = schema_options(rule_options_schema(cls), name=name).check(dict(written), options_where)
        problems.extend(found)
        choice = AdmissionChoice(name, cls, parsed, dict(written))
        method = getattr(cls, "check", None)
        if check and len(problems) == before and callable(method):
            ctx = AdmissionCheckContext(
                level=level,  # type: ignore[arg-type]
                entry=entry,
                entries=contexts,
                batch_size=batch_size,
                label_column=column,
                max_groups_per_step=float(cap),
                workflow_gives_verdicts=gives_verdicts is not False,
            )
            try:
                method(choice.options, ctx)
            except Exception as exc:
                message = str(exc) or type(exc).__name__
                problems.append(f"{where}: {message if message.startswith(name) else f'{name}: {message}'}")
        return choice

    run_name = getattr(train, "group_admission", None) or "all"
    run_written = dict(getattr(train, "group_admission_options", None) or {})
    run = choose(run_name, run_written, "run", None, "train.group_admission")
    entries: dict[str, AdmissionChoice] = {}
    own: set[str] = set()
    for environment in environments:
        label = environment.label
        name = getattr(environment, "group_admission", None)
        written = dict(getattr(environment, "group_admission_options", None) or {})
        if name is None:
            if written:
                problems.append(_options_without_rule(f"environments.{entry_index[label]}"))
            if run is not None:
                entries[label] = run
            continue
        choice = choose(name, written, "entry", label, f"environments[{label}].group_admission")
        if choice is not None:
            entries[label] = choice
            own.add(label)
    if gives_verdicts is False and run is not None:
        inherited = not environments or any(label not in own for label in entries)  # some entry takes the run's
        for label, choice in [(None, run), *entries.items()]:
            if issubclass(choice.cls, Informative) and (inherited if label is None else label in own):
                where = "train.group_admission" if label is None else f"environments[{label}].group_admission"
                problems.append(
                    f"{where}: {choice.name} rejects the groups whose workflow's verdict says they train nothing; the "
                    f"run's workflow, {workflow}, gives no verdicts (it is no GroupRolloutWorkflow, and does not "
                    f"declare gives_verdicts = True), so {choice.name} would admit every group and change nothing but "
                    "the step's accounting"
                )
    if problems or run is None:
        raise ConfigError(problems)
    return AdmissionChoices(run=run, entries=entries, own=frozenset(own), max_groups_per_step=float(cap))


def admission_lock_records(choices: AdmissionChoices) -> list[dict[str, Any]]:
    """What ``lock.json`` records of the run's rules other than ``all`` (``kind: group_admission``), one record per
    rule: the reference as the config gives it, what it resolved to, the distribution and version that provide it,
    its level, the entries whose groups it decides, and its options with their defaults (the run's under ``train``, an
    entry's own under its label).  None in a run without one."""

    from platoon.extensions import origin_of_module

    if not choices.active:
        return []
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for label, choice in [*choices.entries.items()] or [(None, choices.run)]:
        if choice.admits_all:
            continue
        level = "entry" if label is not None and label in choices.own else "run"
        source = f"{choice.cls.__module__}:{choice.cls.__qualname__}"
        key = (choice.name, source, level)
        if key not in records:
            origin = origin_of_module(choice.cls.__module__)
            records[key] = {
                "kind": "group_admission",
                "ref": choice.name,
                "resolved": source,
                "source": source,
                "distribution": origin.distribution,
                "distribution_version": origin.version,
                "entry_point": None,
                "level": level,
                "environments": [],
                "options": {"train": choice.options_json()} if level == "run" else {},
            }
        record = records[key]
        if label is not None:
            record["environments"].append(label)
            if level == "entry":
                record["options"][label] = choice.options_json()
    return list(records.values())


# --- the decisions at settle ------------------------------------------------------------------------------------------


@dataclass
class _Rule:
    """One rule instance of the set: its qualified name and the instance."""

    name: str
    instance: Any


class AdmissionSet:
    """A rule-mode run's rules, by entry (see the module docstring): built once per start, beside the dispatcher, over
    its task samplers (``samplers``), whose view of the per-task statistics each decision reads, and whose entries the
    rules' setups carry.  An entry that inherits the run's rule shares the run's instance."""

    def __init__(
        self, choices: AdmissionChoices, *, samplers: SamplerSet | None, batch_size: int, label_column: str
    ) -> None:
        self.choices = choices
        self.samplers = samplers
        self.batch_size = batch_size
        self.label_column = label_column
        self.cap = groups_per_step(choices.max_groups_per_step, batch_size)
        infos = dict(samplers.infos) if samplers is not None else {}
        self._infos = infos

        def build(choice: AdmissionChoice, name: str, level: str, entry: str | None) -> _Rule:
            setup = AdmissionSetup(
                name=name,
                level=level,  # type: ignore[arg-type]
                entry=entry,
                entries=infos,
                batch_size=batch_size,
                label_column=label_column,
            )
            return _Rule(name, choice.cls(setup, choice.options))

        self._run = build(choices.run, choices.run.name, "run", None)
        self._rules: dict[str, _Rule] = {}
        for label, choice in choices.entries.items():
            own = label in choices.own
            self._rules[label] = build(choice, choices.qualified(label), "entry", label) if own else self._run
        # Decisions that failed open, since the last step's metrics; the rules whose failure was logged.
        self.errors = 0
        self._logged: set[str] = set()
        self._unknown: set[str] = set()
        # The reasons each rule gave that are metric keys (the first REASON_KEYS of them).
        self._reason_keys: dict[str, set[str]] = {}
        # What the current step dequeued, for its metrics; and the warnings given once per run.
        self.step = StepAdmissions()
        self._warned: set[str] = set()
        # The rule's calls, which the tests read (evaluation never calls a rule).
        self.calls = 0

    @property
    def active(self) -> bool:
        return self.choices.active

    def rule(self, entry: str | None) -> _Rule:
        """The rule that decides ``entry``'s groups: its own, else the run's."""

        return self._rules.get(entry, self._run) if entry is not None else self._run

    def view(self, now: int) -> TaskStatsView:
        """The run's per-task statistics as a rule reads them, at policy version ``now``: as of the last flush, never
        flushing them (the settled groups' records wait for the next one, in draw order)."""

        if self.samplers is not None:
            return self.samplers.view(now, flush=False)
        return TaskStatsView(TaskStatsStore([], {}, column=self.label_column), now=now, live=False)

    def at_settle(self, draw: Any, rollout: Any, record: GroupRecord | None, *, now: int) -> tuple[Admission, str]:
        """The decision on the group of ``draw`` (a :class:`~platoon.train.dispatch.dispatcher.BoundDraw`) that settled
        with ``rollout`` (None: its workflow failed), whose record in the statistics is ``record`` (None when they
        stopped or never started), and the qualified name of the rule that decides it.  A failed group is rejected
        with reason ``failed``, its rule not asked.  Otherwise the rule decides on the record, or on one built from the
        draw and the output; anything on that path that raises, or an answer outside the contract, admits the group,
        logged once per rule and counted (``errors``).  Never raises."""

        rule = self.rule(draw.entry)
        if rollout is None:
            return Admission(False, "failed"), rule.name
        try:
            group = record if record is not None else self._record(draw, rollout)
            self.calls += 1
            answer = rule.instance.admit(group, self.view(now))
            if inspect.isawaitable(answer):
                close = getattr(answer, "close", None)
                if callable(close):
                    close()
            decision = checked_admission(answer, rule.name)
        except Exception as exc:
            self.errors += 1
            if rule.name not in self._logged:
                self._logged.add(rule.name)
                logger.error(
                    "Group admission rule %s failed on the group of task %s (%s: %s): the group is admitted, and "
                    "training goes on; later failures of this rule are counted (train/admission/errors), not logged",
                    rule.name,
                    getattr(draw, "task_id", None),
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
            return Admission(True), rule.name
        if decision.admit and decision.reason == "unknown" and isinstance(rule.instance, Informative):
            if rule.name not in self._unknown:
                self._unknown.add(rule.name)
                logger.warning(
                    "Group admission rule %s admits the groups of environments[%s] whose output carries no verdict "
                    "(reason unknown): its workflow returned a plain list, or a verdict platoon does not know, so it "
                    "cannot tell whether the group trains; such a group counts toward train.batch_size whether it "
                    "trains or not (counted under train/admission/admitted/unknown; logged once)",
                    rule.name,
                    draw.entry,
                )
        return decision, rule.name

    def _record(self, draw: Any, rollout: Any) -> GroupRecord:
        """A record of the group of ``draw`` when the statistics have none: built from the draw and its output."""

        from platoon.train.task_records import group_record

        info = self._infos.get(draw.entry) if draw.entry is not None else None
        return group_record(
            start=draw.start,
            seq=draw.seq,
            task_id=draw.task_id,
            entry=draw.entry,
            labels={self.label_column: draw.label} if draw.label is not None else {},
            group_size=info.group_size if info is not None else None,
            sampler=draw.sampler or "",
            reason="order",
            score=None,
            position=draw.position,
            drawn_at={},
            output=rollout,
        )

    # --- what a step counts -------------------------------------------------------------------------------------------

    def dequeued(self, label: str | None, kind: str, item: Any) -> None:
        """Count an item the current step dequeued, of ``label``: an ``admitted`` group (with the reason it was admitted
        with), a ``stale`` one, a kept ``failed`` one, or a ``marker`` of a group rejected or failed when it settled
        (``item``, the step's :class:`~platoon.train.dispatch.settled._Settled`); and, of those with an output,
        whether it was informative (``admission/.../informative_rate``)."""

        step = self.step
        output = getattr(item, "rollout", None)
        record = getattr(item, "record", None)
        verdict = record.verdict if record is not None else getattr(output, "verdict", "unknown")
        entry = getattr(item, "entry", None)
        if kind == "admitted":
            reason = getattr(item, "admitted", None) or "admitted"
            step.admitted(label, None if reason == "admitted" else self.reason_key(entry, reason))
        elif kind == "stale":
            step.stale(label)
        elif kind == "failed" or output is None:
            step.failed(label)
            return
        else:
            step.rejected(label, self.reason_key(entry, getattr(item, "rejected", None) or "rejected"))
        step.judged(label, verdict)

    def reason_key(self, entry: str | None, reason: str) -> str:
        """``reason`` as a metric key of the rule that decides ``entry``'s groups: a short word, the first REASON_KEYS
        of a rule's own; ``other`` beyond them."""

        from platoon.train.sampler_runtime import REASON_KEY, REASON_KEYS

        keys = self._reason_keys.setdefault(self.rule(entry).name, set())
        if reason in keys:
            return reason
        if REASON_KEY.fullmatch(reason) and len(keys) < REASON_KEYS:
            keys.add(reason)
            return reason
        return "other"

    def step_metrics(
        self, *, prefixes: Mapping[str, str], counted: int, admitted: int, draws: float | None
    ) -> dict[str, float]:
        """The current step's ``admission/...`` metrics (keys below ``train/``), then a new step's counts: per label
        (``admission/<prefix>/...``, each label of ``prefixes`` at zero when it had none), and run-wide."""

        step, self.step = self.step, StepAdmissions()
        errors, self.errors = self.errors, 0
        found: dict[str, float] = {}

        def add(where: str, counts: _Counts) -> None:
            found[f"{where}admitted"] = float(counts.admitted)
            for reason, count in counts.admitted_reasons.items():
                found[f"{where}admitted/{reason}"] = float(count)
            found[f"{where}rejected"] = float(counts.rejected)
            for reason, count in counts.rejected_reasons.items():
                found[f"{where}rejected/{reason}"] = float(count)
            found[f"{where}stale"] = float(counts.stale)
            found[f"{where}failed"] = float(counts.failed)
            if counts.judged:
                found[f"{where}informative_rate"] = counts.informative / counts.judged

        for label, prefix in prefixes.items():
            add(f"admission/{prefix}/", step.labels.get(label, _Counts()))
        total = step.total
        add("admission/", total)
        del found["admission/admitted"]  # the run's count of admitted groups is groups_admitted
        found["admission/groups_admitted"] = float(admitted)
        found["admission/counted"] = float(counted)
        found["admission/short"] = float(admitted < self.batch_size)
        found["admission/errors"] = float(errors)
        if admitted and draws is not None:
            found["admission/draws_per_admitted"] = float(draws) / admitted
        return found

    # --- warnings -----------------------------------------------------------------------------------------------------

    def waiting(self, step: int, *, counted: int, admitted: int) -> None:
        """A step that keeps waiting: each time its count reaches another multiple of :data:`WAITING_BATCHES` batches
        while it has admitted fewer than a batch, a warning, since its metrics are logged only when it closes."""

        every = WAITING_BATCHES * self.batch_size
        if counted and counted % every == 0 and admitted < self.batch_size and counted < self.cap:
            logger.warning(
                "Batch %s has counted %d groups and admitted %d of train.batch_size (%d); it waits for more, up to %d "
                "groups (train.refill.max_groups_per_step %g): a pool the policy has solved never fills a step",
                step,
                counted,
                admitted,
                self.batch_size,
                self.cap,
                self.choices.max_groups_per_step,
            )

    def step_closed(self, step: int, *, counted: int, admitted: int) -> None:
        """A step that closed short, or with nothing admitted: a warning the first time each happens in the run, with
        what the step dequeued that it did not train (its rejected, stale and failed groups, from the current step's
        counts, before :meth:`step_metrics` takes them) and the fix that fits: where failed groups outnumber the
        rejected ones, the environment's, not the rule's."""

        total = self.step.total
        dequeued = f"{total.rejected} were rejected, {total.stale} stale and {total.failed} failed"
        failing = total.failed > total.rejected
        if admitted == 0 and "empty" not in self._warned:
            self._warned.add("empty")
            logger.warning(
                "Batch %s admitted no group: of the groups it dequeued, %s, and it counted %d toward its cap, so it "
                "takes no optimizer update and the policy keeps its version.  %s (logged once)",
                step,
                dequeued,
                counted,
                "Most of them failed: see the failed groups' errors, which no rule decides"
                if failing
                else "The policy may have solved the pool: look at train/admission/rejected/<reason>",
            )
        elif 0 < admitted < self.batch_size and "short" not in self._warned:
            self._warned.add("short")
            logger.warning(
                "Batch %s trains %d admitted groups, fewer than train.batch_size (%d): it counted %d groups, its cap "
                "(train.refill.max_groups_per_step %g), and of the groups it dequeued, %s; a short step trains the "
                "groups it admitted.  %s (logged once)",
                step,
                admitted,
                self.batch_size,
                counted,
                self.choices.max_groups_per_step,
                dequeued,
                "Most of them failed: see the failed groups' errors, which no rule decides"
                if failing
                else "At this pass rate, raise train.refill.max_groups_per_step, or choose tasks with task_sampler: "
                "pass_rate",
            )


@dataclass
class _Counts:
    """What a step dequeued of one label (or of the whole run)."""

    admitted: int = 0
    admitted_reasons: Counter[str] = field(default_factory=Counter)
    rejected: int = 0
    rejected_reasons: Counter[str] = field(default_factory=Counter)
    stale: int = 0
    failed: int = 0
    # Of the groups with an output (admitted, stale, rejected): how many it judged, and how many were informative.
    judged: int = 0
    informative: int = 0


@dataclass
class StepAdmissions:
    """What the current step dequeued under a rule, by label, for its ``admission/...`` metrics."""

    labels: dict[str | None, _Counts] = field(default_factory=dict)
    total: _Counts = field(default_factory=_Counts)

    def _all(self, label: str | None) -> tuple[_Counts, _Counts]:
        return self.labels.setdefault(label, _Counts()), self.total

    def admitted(self, label: str | None, reason_key: str | None) -> None:
        for counts in self._all(label):
            counts.admitted += 1
            if reason_key is not None and reason_key != "admitted":
                counts.admitted_reasons[reason_key] += 1

    def rejected(self, label: str | None, reason_key: str) -> None:
        for counts in self._all(label):
            counts.rejected += 1
            counts.rejected_reasons[reason_key] += 1

    def stale(self, label: str | None) -> None:
        for counts in self._all(label):
            counts.stale += 1

    def failed(self, label: str | None) -> None:
        for counts in self._all(label):
            counts.failed += 1

    def judged(self, label: str | None, verdict: Any) -> None:
        for counts in self._all(label):
            counts.judged += 1
            counts.informative += verdict == "trained"


# --- the checks a config's keys get when it loads ---------------------------------------------------------------------


def _packages(root: Any) -> bool:
    """Whether some environment entry of the config as given names a ``package``, which may register rules."""

    environments = root.get("environments") if isinstance(root, Mapping) else None
    return isinstance(environments, list) and any(
        isinstance(entry, Mapping) and entry.get("package") for entry in environments
    )


def _unknown_name(name: str, packages: bool) -> bool:
    """Whether a rule name names nothing that could resolve: empty, or not core's, not ``module:Class``, while no entry
    names a package."""

    return not name or (name not in RULES and ":" not in name and not packages)


def _not_one_of(path: str, name: str) -> str:
    return (
        f"{path}: {name!r} is not one of {', '.join(map(repr, RULES))}, a name an environment's package registers, or "
        "module:Class"
    )


def _options_without_rule(where: str) -> str:
    return (
        f"{where}.group_admission_options: an entry's options go with the rule it sets: set {where}.group_admission "
        "beside them (the entry inherits train.group_admission and its options otherwise)"
    )


def _options_problems(name: Any, values: Any, path: str) -> list[str]:
    """The problems of the options ``values`` of core's rule ``name`` (none for another rule's, which are checked where
    it resolves)."""

    from platoon.options import schema_options
    from platoon.registry import import_from_string

    if not isinstance(name, str) or name not in BUILTIN_GROUP_ADMISSIONS:  # a name of another type is refused itself
        return []
    schema = rule_options_schema(import_from_string(BUILTIN_GROUP_ADMISSIONS[name]))
    return schema_options(schema, name=name).option_problems(values, path)


def run_rule_problems(value: Any, train: Mapping[str, Any], root: Any, path: str) -> list[str]:
    """``train.group_admission``'s problems when the config loads (``TrainConfig.VALUE_PROBLEMS``): a name that nothing
    could resolve, while no entry names a package that could register it."""

    if not isinstance(value, str) or not _unknown_name(value, _packages(root)):
        return []
    return [_not_one_of(path, value)]


def run_options_problems(value: Any, train: Mapping[str, Any], root: Any, path: str) -> list[str]:
    """``train.group_admission_options``' problems when the config loads: options for ``all``, which takes none, and
    ``informative``'s checked against :class:`InformativeOptions`."""

    if not isinstance(value, Mapping):
        return []
    name = train.get("group_admission") if isinstance(train, Mapping) else None
    return _options_problems("all" if name is None else name, value, path)


def entry_rule_problems(value: Any, entry: Mapping[str, Any], root: Any, path: str) -> list[str]:
    """An entry's ``group_admission``'s problems when the config loads (``EnvironmentConfig.VALUE_PROBLEMS``): a name
    that nothing could resolve, while no entry names a package."""

    if not isinstance(value, str) or not _unknown_name(value, _packages(root)):
        return []
    return [_not_one_of(path, value)]


def entry_options_problems(value: Any, entry: Mapping[str, Any], root: Any, path: str) -> list[str]:
    """An entry's ``group_admission_options``' problems when the config loads: options without the entry's own rule
    (the options go with the rule they configure), and core's rules' options checked against their schemas."""

    if not isinstance(value, Mapping):
        return []
    name = entry.get("group_admission") if isinstance(entry, Mapping) else None
    if name is None:
        return [_options_without_rule(path.rsplit(".", 1)[0])] if value else []
    return _options_problems(name, value, path)


def max_groups_problems(value: Any, refill: Mapping[str, Any], root: Any, path: str) -> list[str]:
    """``train.refill.max_groups_per_step``'s problems when the config loads (``RefillConfig.VALUE_PROBLEMS``)."""

    problem = max_groups_problem(value)
    return [f"{path}: {problem}"] if problem is not None else []


def effective_rules(config: Any) -> dict[str, str]:
    """Each entry's rule as the config names it, by label (the run's for an entry that sets none); the run's under
    ``""`` for a config without entries."""

    train = getattr(config, "train", None)
    run = str(getattr(train, "group_admission", None) or "all")
    environments = list(getattr(config, "environments", None) or [])
    if not environments:
        return {"": run}
    return {
        environment.label: str(getattr(environment, "group_admission", None) or run) for environment in environments
    }


def rule_mode(config: Any) -> bool:
    """Whether a config is in rule mode, by the names it gives (:func:`is_all`): the rule of some entry is not
    ``all``."""

    return any(not is_all(name) for name in effective_rules(config).values())


def config_admission_problems(config: Any) -> list[str]:
    """What a loaded (or built) config's group admission keys refuse, one line each: ``num_minibatches`` above 1 beside
    a rule other than ``all`` (a short step's last minibatch would take a full optimizer step on a few groups, and a
    step could not be interrupted after its first update until it had counted its groups); and the checks the loader
    makes of the names, the built-ins' options, an entry's options without its rule and the cap, again for a config
    built in code, which never passes the loader."""

    train = getattr(config, "train", None)
    if train is None:
        return []
    environments = list(getattr(config, "environments", None) or [])
    packages = any(getattr(environment, "package", None) for environment in environments)
    problems: list[str] = []
    run = getattr(train, "group_admission", None) or "all"
    if isinstance(run, str) and _unknown_name(run, packages):
        problems.append(_not_one_of("train.group_admission", run))
    else:
        problems += _options_problems(
            run, getattr(train, "group_admission_options", None) or {}, "train.group_admission_options"
        )
    for index, environment in enumerate(environments):
        name = getattr(environment, "group_admission", None)
        options = getattr(environment, "group_admission_options", None) or {}
        where = f"environments.{index}"
        if name is None:
            if options:
                problems.append(_options_without_rule(where))
        elif isinstance(name, str) and _unknown_name(name, packages):
            problems.append(_not_one_of(f"{where}.group_admission", name))
        else:
            problems += _options_problems(name, options, f"{where}.group_admission_options")
    cap = getattr(getattr(train, "refill", None), "max_groups_per_step", DEFAULT_MAX_GROUPS_PER_STEP)
    if (problem := max_groups_problem(cap)) is not None:
        problems.append(f"train.refill.max_groups_per_step: {problem}")
    minibatches = getattr(train, "num_minibatches", 1)
    if isinstance(minibatches, int) and minibatches > 1 and rule_mode(config):
        rules = effective_rules(config)
        if not is_all(run):
            which = f"train.group_admission: {run}"
        else:
            label = next(label for label, name in rules.items() if not is_all(name))
            which = f"environments[{label}].group_admission: {rules[label]}"
        problems.append(
            f"train.num_minibatches: {minibatches} takes {minibatches} optimizer steps a batch, and a group admission "
            f"rule ({which}) makes short steps: the last minibatch of a short step would take a full optimizer step on "
            "a few groups, and after its first update a step cannot be interrupted until it has counted its groups; "
            "set num_minibatches: 1 (num_microbatches may stay)"
        )
    return problems


def loss_warnings(config: Any) -> list[str]:
    """A rule other than ``all`` under a loss whose objective is not zero at zero advantages (one with
    ``loss_fn_config`` keys but the clip thresholds, or a loss the trainer does not know): a uniform group's datums
    train its advantage-independent terms (a KL or an entropy term), and rejecting such groups removes their share of
    those terms from the objective, not only from its scale.  One line, or none."""

    from platoon.train.step.batch_transforms import objective_is_zero_for_zero_advantages

    train = getattr(config, "train", None)
    if train is None or not rule_mode(config):
        return []
    loss = getattr(train, "loss_fn", "cispo")
    loss_config = dict(getattr(train, "loss_fn_config", None) or {})
    if objective_is_zero_for_zero_advantages(loss, loss_config):
        return []
    rules = effective_rules(config)
    entries = ", ".join(
        f"environments[{label}] ({name})" if label else name for label, name in rules.items() if not is_all(name)
    )
    keys = ", ".join(sorted(loss_config)) or "none"
    return [
        f"train.loss_fn: {loss} with loss_fn_config keys {keys} may train terms that do not depend on the advantage (a "
        "KL or entropy term), and a uniform group's datums train them; the group admission rule of "
        f"{entries} rejects such groups, which removes their share of those terms from the objective, not only from "
        "its scale"
    ]


# --- what the plan and platoon config show say of the rules ----------------------------------------------------------


def plan_line(train: Any, environments: Sequence[Any]) -> str | None:
    """The sampling plan's line of a run in rule mode: each entry's rule, and what a step trains; None in all mode."""

    run = str(getattr(train, "group_admission", None) or "all")
    rules = {
        environment.label: str(getattr(environment, "group_admission", None) or run) for environment in environments
    }
    if all(is_all(name) for name in (rules.values() if rules else [run])):
        return None
    batch_size = int(getattr(train, "batch_size", 0) or 0)
    cap = getattr(getattr(train, "refill", None), "max_groups_per_step", DEFAULT_MAX_GROUPS_PER_STEP)
    named = [label for label in rules if getattr(_by_label(environments, label), "name", None)]
    if len(set(rules.values())) == 1 and not named:
        which = next(iter(rules.values()), run)
    else:
        which = ", ".join(f"{name} for {label}" for label, name in rules.items())
    if batch_size:
        trains = f"a step trains the first {batch_size} admitted groups of at most {groups_per_step(cap, batch_size)}"
    else:
        trains = "a step trains the first train.batch_size admitted groups"
    return f"group admission (experimental): {which}; {trains} (refill.max_groups_per_step {cap})"


def _by_label(environments: Sequence[Any], label: str) -> Any:
    return next((environment for environment in environments if environment.label == label), None)


def describe_line(config: Any, label: str) -> tuple[str, str] | None:
    """What ``platoon config show`` says of an entry's rule: ``(setting, layer)``, the setting its rule with its
    options (``informative {band: [0.2, 0.8]}``), the layer ``entry``, ``train.group_admission`` or ``default``; None
    in all mode for an entry that sets none."""

    from platoon.utils.config import given_mapping

    train = config.train
    entry = _by_label(config.environments, label)
    own = getattr(entry, "group_admission", None)
    if own is None and not rule_mode(config):
        return None
    if own is not None:
        name, options, layer = own, dict(getattr(entry, "group_admission_options", None) or {}), "entry"
    else:
        name = str(getattr(train, "group_admission", None) or "all")
        options = dict(getattr(train, "group_admission_options", None) or {})
        given = given_mapping(train)
        layer = "train.group_admission" if given is None or "group_admission" in given else "default"
    text = f"group_admission: {name}"
    if options:
        text += " {" + ", ".join(f"{key}: {_json_text(value)}" for key, value in options.items()) + "}"
    return text, layer


def own_rule_text(config: Any, label: str) -> str | None:
    """What ``platoon config validate`` says of an entry's own rule, when it sets one that is not the run's (another
    rule, or other options): ``group_admission informative {band: [0.2, 0.8]}``; None otherwise."""

    entry = _by_label(getattr(config, "environments", None) or [], label)
    own = getattr(entry, "group_admission", None)
    if own is None:
        return None
    options = dict(getattr(entry, "group_admission_options", None) or {})
    train = getattr(config, "train", None)
    run = str(getattr(train, "group_admission", None) or "all")
    if own == run and options == dict(getattr(train, "group_admission_options", None) or {}):
        return None
    text = f"group_admission {own}"
    if options:
        text += " {" + ", ".join(f"{key}: {_json_text(value)}" for key, value in options.items()) + "}"
    return text


def _json_text(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return "null" if value is None else ("true" if value else "false")
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_json_text(item) for item in value) + "]"
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{key}: {_json_text(item)}" for key, item in value.items()) + "}"
    return str(value)


def fingerprint(choices: AdmissionChoices | None) -> dict[str, Any] | None:
    """The sampling fingerprint's ``admission`` key of a run in rule mode: the cap, and each entry's rule and options
    as parsed, defaults included; None in all mode, so that a run without a rule keeps its fingerprint."""

    if choices is None or not choices.active:
        return None
    return {
        "max_groups_per_step": choices.max_groups_per_step,
        "entries": {
            label: {"rule": choice.name, "options": choice.options_json()} for label, choice in choices.entries.items()
        }
        or {"": {"rule": choices.run.name, "options": choices.run.options_json()}},
    }
