"""Group admission: whether a settled training group counts toward the batch (RFC 0019, section 6).

A group admission rule is a small class: :meth:`AdmissionRule.admit` answers, for one settled group (its
:class:`~platoon.train.task_stats.GroupRecord`) and the run's per-task statistics (a read-only
:class:`~platoon.train.sampler_api.TaskStatsView`), whether the group is admitted to the step that dequeues it
(:class:`Admission`, or a bare ``bool``).  It is built once per start of the process, ``Class(setup, options)``
(:class:`AdmissionSetup`), and a load-time ``check`` may refuse a config (:class:`AdmissionCheckContext`).

The trainer asks the rule in the group's own coroutine, right after the per-task statistics record the group and before
any step dequeues it.  A rejected group's fate, ``rejected``, is final there: its slot of the horizon is free at once
(unless it carried no datum, which keeps it until a step dequeues the group's marker), so the dispatcher starts another
group while the trainer trains, and the step that dequeues the group's marker records its statistics.  A rule is a pure
function of ``(group, stats, options)``: it keeps no state between calls, so that a resume decides as the run would have
(``platoon.testing.check_group_admission`` checks it).  The view reads the settled groups' evidence as of the last flush
of the statistics (each step's metrics, each full checkpoint, each adaptive decision) and the draws and fates as they
are now; ``rejected`` (and, under a rule, ``failed``) is known when a group settles, ``consumed`` and ``stale`` only
when the step that dequeued the group commits.

A rule that raises, or answers outside the contract (anything but an :class:`Admission` or a ``bool``, or a reason that
is not a non-empty string of at most 64 characters), admits the group: an admitted group that trains nothing costs a
slot of one step, while a wrong rejection could empty every step.  The failure is logged once per rule and counted.

Rules are registered by name with :func:`group_admission` from a module an environment entry's ``package`` names, or
named as ``module:Class``; core's names (:data:`BUILTIN_GROUP_ADMISSIONS`: ``all`` and ``informative``) are reserved
(:func:`resolve_group_admission`).  The API is experimental (``platoon.experimental``): a minor release may change it.
"""

from __future__ import annotations

import difflib
import inspect
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, TypeVar, runtime_checkable

from platoon.train.sampler_api import MAX_REASON_LENGTH, Level, TaskStatsView
from platoon.train.task_stats import EntryInfo, GroupRecord

R = TypeVar("R", bound=type)

# The registry kind of group admission rules (platoon.registry), which @group_admission registers in.
REGISTRY_KIND = "group_admission"
# Core's rules, by name: resolved from here before the registry is read, so that nothing installed or registered can
# shadow them, whatever the import order.
BUILTIN_GROUP_ADMISSIONS: dict[str, str] = {
    "all": "platoon.train.admissions:AdmitAll",
    "informative": "platoon.train.admissions:Informative",
}
RESERVED_GROUP_ADMISSION_NAMES = frozenset(BUILTIN_GROUP_ADMISSIONS)
# What a registered name may be: no spaces, and neither ``:`` (module:Class) nor ``@`` (an entry's qualified name).
_NAME = re.compile(r"[^\s:@]+")


def _plain(name: str) -> str:
    """A rule's name as it reads, whatever its case and its ``-`` or ``_``: ``Informative`` reads as ``informative``."""

    return name.casefold().replace("-", "_")


# ``train.refill.max_groups_per_step``: its default, and what it may be, here alone, so that another default or a
# null (no cap) is a small change.  A step under a rule counts at most ``ceil(value × batch_size)`` groups.
DEFAULT_MAX_GROUPS_PER_STEP = 3.0


def max_groups_problem(value: Any) -> str | None:
    """Why ``value`` is no ``refill.max_groups_per_step`` (a finite number of at least 1), or None when it is one."""

    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 1:
        return (
            "must be a finite number of at least 1 (a step under a group admission rule counts at most this many "
            f"times train.batch_size groups), got {value!r}"
        )
    return None


def groups_per_step(max_groups_per_step: float, batch_size: int) -> int:
    """The most groups a step under a rule counts: ``ceil(max_groups_per_step × batch_size)``, the product rounded to
    nine decimals first, so that 1.1 × 10 counts 11 groups, not 12."""

    return max(batch_size, math.ceil(round(max_groups_per_step * batch_size, 9)))


@dataclass(frozen=True)
class Admission:
    """A rule's decision on one settled group: ``admit`` it to the step that dequeues it, or reject it, with
    ``reason``, a non-empty string of at most 64 characters, which goes into a rejection's fate event and the step's
    metrics.  Left out, the reason is ``"admitted"`` or ``"rejected"``, by ``admit``."""

    admit: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.reason is None:
            object.__setattr__(self, "reason", "admitted" if self.admit else "rejected")


@runtime_checkable
class GroupAdmission(Protocol):
    """What the trainer calls of a rule: ``admit(group, stats)``, one synchronous call per settled group."""

    def admit(self, group: GroupRecord, stats: TaskStatsView) -> Admission | bool:
        """Whether ``group``, a settled training group, counts toward the batch, read against ``stats``, the run's
        per-task statistics as of their last flush: an :class:`Admission`, or ``True`` (admitted) or ``False``
        (rejected, reason ``rejected``).  A pure function of the group, the view and the rule's options."""
        ...


@dataclass(frozen=True)
class AdmissionSetup:
    """What a rule is built with, once per start of the process: ``Class(setup, options)``."""

    # Its qualified name: the run's rule as the config names it ("informative", "acme.rules:Quota"), an entry's with the
    # entry's label ("informative@hard").
    name: str
    # "run" for the run's rule (train.group_admission), which decides the groups of every entry that sets none;
    # "entry" for an entry's own.
    level: Level
    # The entry, at the entry level.
    entry: str | None
    # The run's entries with training tasks, by label in config order (what the run's task samplers know of them,
    # their task ids included); empty for a trainer given an order without sampling, so a rule must not rely on it.
    entries: Mapping[str, EntryInfo]
    # train.batch_size: the admitted groups a step trains.
    batch_size: int
    # The column of the training tasks whose value labels a group (train.label_by; the entry by default), the key of
    # a group's labels (GroupRecord.labels, empty for a single environment's groups).
    label_column: str


@dataclass(frozen=True)
class AdmissionCheckContext:
    """What a rule's load-time ``check`` sees: the parts of the config a rule's rules need, as plain data."""

    # "run" for the run's rule (train.group_admission), "entry" for an entry's own.
    level: Level
    # The entry, at the entry level.
    entry: str | None
    # Every entry by label, in config order, each exactly {"weight", "seed", "group_size", "success_threshold",
    # "task_sampler", "task_sampler_options"}, as a task sampler's check sees them.
    entries: Mapping[str, Mapping[str, Any]]
    # train.batch_size: the admitted groups a step trains.
    batch_size: int
    # The column of the training tasks whose value labels a group (train.label_by; the entry by default).
    label_column: str
    # train.refill.max_groups_per_step.
    max_groups_per_step: float
    # Whether the run's workflow gives each group a verdict (a GroupRolloutWorkflow, or a class that declares
    # gives_verdicts = True).
    workflow_gives_verdicts: bool


class AdmissionRule:
    """An optional base class of group admission rules: the protocol's shape, typed options and a load-time check.  A
    rule must be a pure function of ``(group, stats, options)``: it keeps no state between calls, so that a resume
    decides as the run would have (``platoon.testing.check_group_admission`` checks it)."""

    # Its options' schema: a dataclass, a pydantic model, or a function of the options' mapping that returns them parsed
    # and raises ValueError; None: it takes no options.  ``self.options`` holds them parsed.
    options_schema: ClassVar[Any] = None

    def __init__(self, setup: AdmissionSetup, options: Any = None) -> None:
        """Keep what the rule is built with: ``setup`` (:class:`AdmissionSetup`) and ``options``, parsed against
        :attr:`options_schema` (None when it takes none)."""

        self.setup = setup
        self.options = options

    def admit(self, group: GroupRecord, stats: TaskStatsView) -> Admission | bool:
        """Whether ``group`` counts toward the batch, read against ``stats`` (:meth:`GroupAdmission.admit`): a subclass
        implements it."""

        raise NotImplementedError(f"{type(self).__name__} does not implement admit(group, stats)")

    @staticmethod
    def check(options: Any, ctx: AdmissionCheckContext) -> None:
        """Refuse a config it cannot serve, with a ``ValueError`` that says why: it accepts every config."""


class AdmissionError(RuntimeError):
    """A rule's answer the trainer refused, or a decision that raised: it names the rule and what went wrong."""

    def __init__(self, rule: str, message: str) -> None:
        super().__init__(f"group admission rule {rule}: {message}")
        self.rule = rule


def checked_admission(answer: Any, rule: str) -> Admission:
    """``answer`` as an :class:`Admission`: an ``Admission`` whose reason is a non-empty string of at most 64
    characters, or a bare ``bool`` (``True`` admitted, ``False`` rejected with reason ``rejected``); an
    :class:`AdmissionError` for anything else."""

    if isinstance(answer, bool):
        return Admission(answer)
    if not isinstance(answer, Admission) or not isinstance(answer.admit, bool):
        raise AdmissionError(rule, f"admit() returned {_brief(answer)}, not an Admission or a bool")
    if not isinstance(answer.reason, str) or not 0 < len(answer.reason) <= MAX_REASON_LENGTH:
        raise AdmissionError(
            rule,
            f"admit() returned a reason that is not a non-empty string of at most {MAX_REASON_LENGTH} characters: "
            f"{_brief(answer.reason)}",
        )
    return answer


def _brief(answer: Any) -> str:
    text = repr(answer)
    return text if len(text) <= 200 else text[:197] + "..."


# --- registration and resolution -------------------------------------------------------------------------------------


def group_admission(
    name: str, *, options: Any = None, check: Callable[[Any, AdmissionCheckContext], None] | None = None
) -> Callable[[R], R]:
    """Register a group admission rule class under ``name``, from a module that an environment entry's ``package``
    names, so that configs can name it (``group_admission: <name>``); ``options`` sets its ``options_schema`` and
    ``check`` its load-time check.  Core's names (``all``, ``informative``) are reserved whatever was imported first,
    and so is any name that reads as one of them whatever its case and its ``-`` or ``_`` (``Informative``, which a
    config that misspells core's name would otherwise pick up); a name registered twice is an error."""

    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError(f"a group admission rule's name is a word without spaces, ':' or '@', got {name!r}")
    if _plain(name) in RESERVED_GROUP_ADMISSION_NAMES:
        reads = "is" if name in RESERVED_GROUP_ADMISSION_NAMES else f"reads as {_plain(name)!r}, which is"
        raise ValueError(
            f"group admission rule name {name!r} {reads} core's ({', '.join(sorted(RESERVED_GROUP_ADMISSION_NAMES))} "
            "are reserved): register yours under another name"
        )

    def decorator(cls: R) -> R:
        from platoon.registry import register_component

        if not isinstance(cls, type):
            raise TypeError(f"@group_admission({name!r}) registers a class, got {cls!r}")
        if options is not None:
            cls.options_schema = options  # type: ignore[attr-defined]
        if check is not None:
            cls.check = staticmethod(check)  # type: ignore[attr-defined]
        register_component(REGISTRY_KIND, name, cls)
        return cls

    return decorator


def resolve_group_admission(ref: str) -> type:
    """The rule class ``ref`` names: core's (:data:`BUILTIN_GROUP_ADMISSIONS`), then a name a package registered
    (:func:`group_admission`), then ``module:Class``.  An :class:`~platoon.extensions.ExtensionError` that says why for
    anything else (with the nearest name), and for a class that is no rule (a callable ``admit`` that is no
    coroutine function)."""

    from platoon.extensions import ExtensionError
    from platoon.registry import get_registry, import_from_string

    if not isinstance(ref, str) or not ref:
        raise ExtensionError(f"a group admission rule is named by a non-empty string, got {ref!r}")
    if ref in BUILTIN_GROUP_ADMISSIONS:
        return import_from_string(BUILTIN_GROUP_ADMISSIONS[ref])
    registry = get_registry(REGISTRY_KIND)
    if ":" not in ref:
        if ref not in registry.names():
            known = [*BUILTIN_GROUP_ADMISSIONS, *registry.names()]
            close = difflib.get_close_matches(ref, known, n=1, cutoff=0.6)
            raise ExtensionError(
                f"unknown group admission rule {ref!r}"
                + (f"; did you mean {close[0]}?" if close else ";")
                + " Core's: all and informative; a package an entry names registers others with @group_admission, "
                "or name one as module:Class"
            )
        cls = registry.get(ref)
    else:
        try:
            cls = import_from_string(ref)
        except Exception as exc:
            raise ExtensionError(f"{ref} does not import ({type(exc).__name__}: {exc})") from exc
    admission_problem(cls, ref)
    return cls


def admission_problem(cls: Any, ref: str) -> None:
    """Raise :class:`~platoon.extensions.ExtensionError` when ``cls`` (what ``ref`` resolved to) is no group admission
    rule class: one with a callable ``admit(group, stats)`` that is no coroutine function."""

    from platoon.extensions import ExtensionError

    if not isinstance(cls, type) or not callable(getattr(cls, "admit", None)):
        raise ExtensionError(f"{ref} is not a group admission rule: a class with admit(group, stats)")
    if inspect.iscoroutinefunction(getattr(cls, "admit")):
        raise ExtensionError(
            f"{ref} is not a group admission rule: its admit(group, stats) is a coroutine, and a decision is one "
            "synchronous call"
        )


def rule_options_schema(cls: type) -> Any:
    """A rule class's options schema (``options_schema``), None for one that takes no options."""

    return getattr(cls, "options_schema", None)
