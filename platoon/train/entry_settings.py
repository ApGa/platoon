"""Per-environment training settings: each environment entry's own settings, laid over the run's.

An environment entry may set the training settings that act on its own groups, ``environments[].workflow_config``
(the keys of :class:`~platoon.train.components.EntryWorkflowSettings`: ``group_size``, the baseline, the minimum
group size, the straggler defaults, the error filters, subagent datum sampling and its rollout settings), what its
evaluation rollouts change, ``eval_overrides.workflow_config``, and the staleness bound of its training groups,
``max_staleness``.  When a training config loads, each entry's effective settings are built from these layers, lowest
first, each laid over the one below it by the config loader's rules (:func:`platoon.utils.config.lay_over`: a nested
section merges key by key, any other value replaces, ``null`` included, and a free-form mapping replaces whole):

- **its training groups:** ``WorkflowConfig``'s defaults, ``train.workflow_config``, a named environment's
  ``workflow_config.rollout_config``, then the entry's ``workflow_config``;
- **its evaluation rollouts during** ``platoon train``: evaluation's defaults, the entry's training settings for the
  keys evaluation takes from training (the rollout and worker settings), the keys ``eval.workflow_config`` sets, then
  the entry's ``eval_overrides.workflow_config``;
- **its rollouts in** ``platoon eval``: ``RolloutConfig``'s defaults, the entry's ``workflow_config.rollout_config``,
  the keys ``inference.workflow.rollout_config`` sets, then the entry's
  ``eval_overrides.workflow_config.rollout_config``.

So the phase's own section wins over an entry's settings, and the entry's ``eval_overrides`` win over both.  An entry
that sets nothing gets the run's settings themselves.

``platoon train`` hands each workflow its entries' settings (``GroupRolloutWorkflow`` reads them for each group), the
trainer's stale filter reads each entry's bound, and recovery pruning keeps the checkpoints within the largest bound
of the run and its entries.  ``platoon config show`` prints each entry's effective settings with the layer each came
from (:func:`describe`), ``platoon config validate`` one line of what differs from the run (:func:`summary`), and
``lock.json`` records the differences (:func:`lock_record`).
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from platoon.config_defs import InferenceParams
from platoon.train.components import ENTRY_ROLLOUT_KEYS, ENTRY_WORKFLOW_KEYS, ENVIRONMENT_COLUMN, row_entry
from platoon.train.labels import task_label

logger = logging.getLogger(__name__)

# The per-group keys, as dotted paths into a workflow config, in the order an entry's settings list them: the keys of
# EntryWorkflowSettings, its rollout_config's, and inference_params' one by one.
GROUP_PATHS: tuple[str, ...] = (
    *(name for name in ENTRY_WORKFLOW_KEYS if name != "rollout_config"),
    *(
        path
        for name in ENTRY_ROLLOUT_KEYS
        for path in (
            [f"rollout_config.inference_params.{item.name}" for item in dataclasses.fields(InferenceParams)]
            if name == "inference_params"
            else [f"rollout_config.{name}"]
        )
    ),
)
ROLLOUT_PATHS: tuple[str, ...] = tuple(path for path in GROUP_PATHS if path.startswith("rollout_config."))
# The keys whose values a laid section's error is read against, when the entry leaves them out.
_RELATED_KEYS = ("group_size", "min_successful_group_size", "straggler_timeout_seconds", "straggler_quorum")
_MISSING = object()


@dataclass(frozen=True)
class EntrySettings:
    """One environment entry's effective training settings."""

    # The entry's label (``EnvironmentConfig.label``).
    label: str
    # Its training groups' workflow config: ``train.workflow_config`` with the entry's ``workflow_config`` laid over
    # it, of the run's own class; the run's own object when the entry sets none.
    train: Any
    # Its evaluation rollouts' workflow config during training (the run's ``eval.workflow_config`` when the entry sets
    # neither ``workflow_config`` nor ``eval_overrides.workflow_config``).
    eval: Any
    # The stale filter's bound for its training groups: its ``max_staleness``, else ``train.max_staleness``.
    max_staleness: int | None


# --- building ---------------------------------------------------------------------------------------------------------


def lay_entry_settings(base: Any, overlay: Mapping[str, Any]) -> Any:
    """``overlay``, an entry's settings as given, laid over ``base``, a workflow config; ``base`` itself for an empty
    overlay.  ``base``'s straggler policies by label stay as they are: each applies to the groups of its label, whose
    group sizes may differ, and is checked against them (:func:`problems`)."""

    if not overlay:
        return base
    from platoon.utils.config import lay_over

    policies = getattr(base, "straggler_policies", None)
    if not policies:
        return lay_over(base, overlay)
    laid = lay_over(dataclasses.replace(base, straggler_policies={}), overlay)
    laid.straggler_policies = dict(policies)
    return laid


def _eval_default(config: Any) -> Any:
    """What a training config's ``eval.workflow_config`` is laid over before anything: its field's default
    (``group_size: 1``, the filters off)."""

    section = config.eval
    for item in dataclasses.fields(section):
        if item.name == "workflow_config":
            if item.default_factory is not dataclasses.MISSING:
                return item.default_factory()
            return item.default
    raise ValueError("the eval section has no workflow_config")


def _eval_settings(config: Any, train: Any, overlay: Mapping[str, Any]) -> Any:
    """An entry's evaluation settings during training: evaluation's defaults, the rollout and worker settings of
    ``train`` (the entry's training settings), the keys ``eval.workflow_config`` sets, then ``overlay`` (its
    ``eval_overrides.workflow_config``).  An ``eval.workflow_config`` built in Python, rather than loaded, is taken as
    given, whole."""

    from platoon.utils.config import given_mapping, lay_over

    run = config.eval.workflow_config
    given = given_mapping(run)
    if given is None:
        laid = run
    else:
        from platoon.train.tinker.config_defs import eval_workflow_base

        base = eval_workflow_base({"train": SimpleNamespace(workflow_config=train)}, _eval_default(config))
        laid = lay_over(base, given) if given else base
    return lay_entry_settings(laid, overlay)


def build(config: Any) -> tuple[dict[str, EntrySettings], list[str]]:
    """Each entry's settings of a training config, by label, and every problem they have, one line each with the key
    that has it: an entry's laid settings that their class refuses (with the values they keep from the run's), an
    entry staleness bound below the run's, a concurrency that holds no group of an entry, a label's straggler quorum
    above the group size of the entry it labels, and ``filter_errors`` set in two places."""

    environments = list(getattr(config, "environments", None) or [])
    train = config.train
    run = train.workflow_config
    run_eval = getattr(getattr(config, "eval", None), "workflow_config", None)
    bound = train.max_staleness
    settings: dict[str, EntrySettings] = {}
    problems: list[str] = []
    for environment in environments:
        label = environment.label
        where = f"environments[{label}]"
        overlay = environment.workflow_config
        eval_overlay = environment.eval_overrides.workflow_config
        try:
            entry_train = lay_entry_settings(run, overlay)
        except (TypeError, ValueError) as exc:
            problems.append(f"{where}.workflow_config: {exc}{_kept(run, overlay, 'train.workflow_config')}")
            continue
        entry_eval = run_eval
        if run_eval is not None and (overlay or eval_overlay):
            try:
                entry_eval = _eval_settings(config, entry_train, eval_overlay)
            except (TypeError, ValueError) as exc:
                note = _kept(run_eval, eval_overlay, "the entry's evaluation settings")
                problems.append(f"{where}.eval_overrides.workflow_config: {exc}{note}")
                continue
        own = environment.max_staleness
        if own is not None and bound is None:
            problems.append(
                f"{where}.max_staleness: {own} bounds the entry's groups, but train.max_staleness is null (no bound), "
                "and an entry's bound is at least the run's, which also bounds how far rollouts run ahead of "
                "training; set train.max_staleness to the smallest bound, and each slower entry's to its own"
            )
        elif own is not None and own < bound:
            problems.append(
                f"{where}.max_staleness: {own} is below train.max_staleness: {bound}, which bounds how far every "
                "entry's rollouts run ahead of training, so the entry's groups would be drawn as far ahead as the "
                f"others' and then dropped as stale; set train.max_staleness to {own} and raise the other entries' "
                "bounds instead"
            )
        settings[label] = EntrySettings(
            label=label, train=entry_train, eval=entry_eval, max_staleness=own if own is not None else bound
        )
    if settings:
        problems += _concurrency_problems(config, settings)
        problems += _straggler_problems(config, settings)
    problems += _filter_errors_problems(environments)
    return settings, problems


def entry_settings(config: Any) -> dict[str, EntrySettings]:
    """Each entry's settings of a training config, by label (:func:`build`); a
    :class:`~platoon.utils.config.ConfigError` with every problem they have."""

    from platoon.utils.config import ConfigError

    settings, problems = build(config)
    if problems:
        raise ConfigError(problems)
    return settings


def sets_anything(entry: Any) -> bool:
    """Whether an environment entry sets settings of its own: a ``workflow_config``, an
    ``eval_overrides.workflow_config``, a ``max_staleness`` or a ``group_admission``."""

    overrides = getattr(entry, "eval_overrides", None)
    return bool(
        getattr(entry, "workflow_config", None)
        or getattr(overrides, "workflow_config", None)
        or getattr(entry, "max_staleness", None) is not None
        or getattr(entry, "group_admission", None) is not None
    )


def run_settings(config: Any) -> dict[str, EntrySettings]:
    """Each entry's settings of a training config, by label, as the trainer applies them: :func:`entry_settings`, or,
    when no entry sets any, the run's settings for every entry, which also serves a config that is not a loaded
    training config, such as a stand-in with entries of labels alone."""

    environments = list(getattr(config, "environments", None) or [])
    if any(sets_anything(entry) for entry in environments):
        return entry_settings(config)
    train = getattr(config, "train", None)
    run = getattr(train, "workflow_config", None)
    run_eval = getattr(getattr(config, "eval", None), "workflow_config", None)
    bound = getattr(train, "max_staleness", None)
    return {
        entry.label: EntrySettings(label=entry.label, train=run, eval=run_eval, max_staleness=bound)
        for entry in environments
    }


def problems(config: Any) -> list[str]:
    """The problems of a training config's entry settings (:func:`build`), one line each."""

    return build(config)[1]


def _kept(base: Any, overlay: Mapping[str, Any], where: str) -> str:
    """The note that names the values a laid section keeps from ``base`` for the keys a refused value is read
    against, which ``overlay`` leaves out."""

    kept = [
        f"{name}: {_text(getattr(base, name))}" for name in _RELATED_KEYS if name not in overlay and hasattr(base, name)
    ]
    return f" (the keys it leaves out keep {where}'s values: {', '.join(kept)})" if kept else ""


def _concurrency_problems(config: Any, settings: Mapping[str, EntrySettings]) -> list[str]:
    """A ``max_concurrent_rollouts`` of training or evaluation that holds no whole group of an entry."""

    found = []
    for section, phase in (("train", "train"), ("eval", "eval")):
        budget = getattr(getattr(config, section, None), "max_concurrent_rollouts", None)
        if budget is None:
            continue
        sizes = {label: getattr(entry, phase).group_size for label, entry in settings.items() if getattr(entry, phase)}
        if not sizes:
            continue
        largest = max(sizes, key=lambda label: sizes[label])
        if budget < sizes[largest]:
            key = "workflow_config" if phase == "train" else "eval_overrides.workflow_config"
            found.append(
                f"{section}.max_concurrent_rollouts: {budget} holds no group of environments[{largest}], whose "
                f"{sizes[largest]} rollouts start together (its {key}.group_size, laid over the run's); raise it to "
                f"{sizes[largest]} at least"
            )
    return found


def _straggler_problems(config: Any, settings: Mapping[str, EntrySettings]) -> list[str]:
    """A label's straggler quorum above the group size of an entry it labels, for the default label column, each
    task's entry, whose labels are the entries' names; another column's labels are known only once the tasks are
    listed (:func:`label_straggler_problems`)."""

    if label_column_of(config) != ENVIRONMENT_COLUMN:
        return []
    return label_straggler_problems(config, settings, {label: {label} for label in settings})


def label_straggler_problems(
    config: Any, settings: Mapping[str, EntrySettings], entries_by_label: Mapping[str, set[str]]
) -> list[str]:
    """A label's straggler quorum (``straggler_policies`` of ``train.workflow_config`` and ``eval.workflow_config``)
    above the smallest group size among the entries whose tasks carry that label (``entries_by_label``)."""

    found = []
    for section, phase in (("train", "train"), ("eval", "eval")):
        workflow = getattr(getattr(config, section, None), "workflow_config", None)
        for label, policy in (getattr(workflow, "straggler_policies", None) or {}).items():
            quorum = getattr(policy, "quorum", None)
            sizes = {
                entry: getattr(settings[entry], phase).group_size
                for entry in sorted(entries_by_label.get(label, ()))
                if entry in settings and getattr(settings[entry], phase) is not None
            }
            if quorum is None or not sizes:
                continue
            smallest = min(sizes, key=lambda entry: sizes[entry])
            if quorum > sizes[smallest]:
                found.append(
                    f"{section}.workflow_config.straggler_policies.{label}.quorum: {quorum} is more than the "
                    f"{sizes[smallest]} rollouts of a group of environments[{smallest}], whose tasks carry the label"
                )
    return found


def listed_straggler_problems(
    config: Any, settings: Mapping[str, EntrySettings], train_tasks: Any, eval_tasks: Any = None
) -> list[str]:
    """:func:`label_straggler_problems` for a label column other than each task's entry, whose labels the listed
    tasks give: which entries' tasks carry each label, read from the training tasks (``train_tasks``) and the
    evaluation tasks (``eval_tasks``, when the run evaluates).  ``platoon train`` checks it once it has listed the
    tasks, and ``platoon config validate --tasks``; the default column's labels are the entries' names, which loading
    checks."""

    column = label_column_of(config)
    if column == ENVIRONMENT_COLUMN:
        return []
    labels = tuple(settings)
    entries_by_label: dict[str, set[str]] = {}
    for tasks in (train_tasks, eval_tasks):
        for row in tasks if tasks is not None else ():
            label, entry = task_label(row, column), row_entry(row, labels)
            if label is not None and entry is not None:
                entries_by_label.setdefault(label, set()).add(entry)
    return label_straggler_problems(config, settings, entries_by_label)


def _filter_errors_problems(environments: Sequence[Any]) -> list[str]:
    """``filter_errors`` set both in a run-level ``workflow_kwargs`` (or ``eval_workflow_kwargs``), which ``platoon
    train`` gives the whole workflow, and in an entry's ``workflow_config`` (or its ``eval_overrides``): one would win
    silently."""

    found = []
    for kwargs_field, path, settings_of in (
        ("workflow_kwargs", "workflow_config", lambda entry: entry.workflow_config),
        ("eval_workflow_kwargs", "eval_overrides.workflow_config", lambda entry: entry.eval_overrides.workflow_config),
    ):
        setters = [entry.label for entry in environments if "filter_errors" in getattr(entry, kwargs_field)]
        if not setters:
            continue
        for entry in environments:
            if "filter_errors" in settings_of(entry):
                found.append(
                    f"environments[{entry.label}].{path}.filter_errors: environments[{setters[0]}].{kwargs_field} "
                    "sets filter_errors for every entry's groups; set it in one place"
                )
    return found


def label_column_of(config: Any) -> str:
    """The run's label column (``train.label_by``), else each task's entry."""

    return getattr(getattr(config, "train", None), "label_by", None) or ENVIRONMENT_COLUMN


# --- the bounds -------------------------------------------------------------------------------------------------------


def staleness_bounds(config: Any) -> dict[str, int | None]:
    """The stale filter's bound for each entry's training groups, by label: its ``max_staleness``, else the run's."""

    run = getattr(config.train, "max_staleness", None)
    return {
        entry.label: entry.max_staleness if getattr(entry, "max_staleness", None) is not None else run
        for entry in getattr(config, "environments", None) or []
    }


def largest_staleness(config: Any) -> tuple[int | None, str]:
    """The largest staleness bound of the run and its entries, and the key that sets it: ``train.max_staleness``, or
    ``environments[<label>].max_staleness``.  None (no bound) when the run sets none, which an entry's bound may not
    be below.  Recovery pruning keeps every checkpoint within it: a slow entry's group may still sample from one."""

    run = getattr(config.train, "max_staleness", None)
    if run is None:
        return None, "train.max_staleness"
    largest, where = run, "train.max_staleness"
    for entry in getattr(config, "environments", None) or []:
        own = getattr(entry, "max_staleness", None)
        if own is not None and own > largest:
            largest, where = own, f"environments[{entry.label}].max_staleness"
    return largest, where


# --- warnings ---------------------------------------------------------------------------------------------------------


def warnings(config: Any, settings: Mapping[str, EntrySettings]) -> list[str]:
    """What a training config's entry settings may get wrong without being wrong, one line each: an entry whose groups
    have one rollout, which trains nothing under the group baselines; an entry that raises its staleness bound (the
    sign of a slow entry) without a limit on its label; an entry's rollout setting that evaluation's section
    overrides for its evaluation rollouts; for the default label column, a ``train.max_concurrent_rollouts`` that
    the capped labels' groups can fill (:func:`permit_warnings`; another column's, once the tasks are listed,
    :func:`listed_permit_warnings`); and a group admission rule under a loss whose objective is not zero at zero
    advantages (:func:`platoon.train.admissions.loss_warnings`)."""

    from platoon.utils.config import given_mapping

    found = []
    limits = getattr(config.train, "rollout_group_limits", None) or {}
    by_entry = label_column_of(config) == ENVIRONMENT_COLUMN
    run_bound = config.train.max_staleness
    eval_given = given_mapping(getattr(getattr(config, "eval", None), "workflow_config", None)) or {}
    for entry in getattr(config, "environments", None) or []:
        label = entry.label
        own = settings.get(label)
        if own is None:
            continue
        if "group_size" in entry.workflow_config and own.train.group_size == 1:
            found.append(
                f"environments[{label}].workflow_config.group_size is 1: a group of one rollout has a zero advantage "
                "under the group baselines and trains nothing, unless a workflow subclass computes its own"
            )
        raised = entry.max_staleness is not None and run_bound is not None and entry.max_staleness > run_bound
        if by_entry and raised and label not in limits:
            found.append(
                f"environments[{label}].max_staleness: {entry.max_staleness} raises the run's bound "
                f"({run_bound}), as a slow entry's does, but train.rollout_group_limits has no limit for it: its "
                "groups count toward how far rollouts run ahead of training for as long as they run, and enough of "
                f"them slow every step to their pace; set train.rollout_group_limits: {{{label}: N}}"
            )
        entry_paths = _flat(entry.workflow_config)
        eval_paths = _flat(eval_given)
        override_paths = _flat(entry.eval_overrides.workflow_config)
        for path in ROLLOUT_PATHS:
            if path in entry_paths and path in eval_paths and path not in override_paths:
                found.append(
                    f"environments[{label}].workflow_config.{path}: {_text(entry_paths[path])} is overridden in its "
                    f"evaluation rollouts by eval.workflow_config.{path}: {_text(eval_paths[path])}; set "
                    f"environments[{label}].eval_overrides.workflow_config.{path} to keep the entry's"
                )
    if by_entry:
        # A mixture's tasks carry their entry as the label; a single environment's carry none.
        by_label = {label: {label} for label in settings} if len(settings) > 1 else {None: set(settings)}
        found += permit_warnings(config, settings, by_label)
    from platoon.train.admissions import loss_warnings

    found += loss_warnings(config)
    return found


def permit_warnings(
    config: Any, settings: Mapping[str, EntrySettings], entries_by_label: Mapping[str | None, set[str]]
) -> list[str]:
    """A ``train.max_concurrent_rollouts`` that the groups of the labels with a limit (``train.rollout_group_limits``)
    can fill so that a group of another entry cannot start, one line: its rollout permits go to groups first come first
    served, so while the capped labels run as many groups as their limits allow, each holding its entry's group size of
    them, a larger group waits for one of those to end, and every group after it waits too.  The budget must hold, for
    each label with a limit, that many groups of the largest entry whose tasks carry it (``entries_by_label``: label,
    None for a task without one, to the entries whose tasks carry it), and a group of the largest other entry."""

    train = config.train
    limits = getattr(train, "rollout_group_limits", None) or {}
    budget = getattr(train, "max_concurrent_rollouts", None)
    if not limits or budget is None:
        return []

    def size(entry: str) -> int:
        return int(settings[entry].train.group_size)

    held: list[tuple[str, int, str, int]] = []  # a capped label, its limit, its largest entry and that one's group
    others: set[str] = set()
    for label, entries in entries_by_label.items():
        known = sorted(entry for entry in entries if entry in settings)
        if not known:
            continue
        if label in limits:
            largest = max(known, key=size)
            held.append((label, limits[label], largest, size(largest)))
        else:
            others.update(known)
    if not held:
        return []
    capped = sum(limit * group for _, limit, _, group in held)
    other = max(sorted(others), key=size) if others else None
    needed = capped + (size(other) if other is not None else 0)
    if budget >= needed:
        return []
    groups = ", ".join(f"{label}: {limit} × {group}" for label, limit, _, group in sorted(held))
    waiting = (
        f"a group of environments[{other}] ({size(other)} rollouts) can then wait for one of theirs to end"
        if other is not None
        else "a group of theirs can then wait for another of theirs to end"
    )
    return [
        f"train.max_concurrent_rollouts: {budget} rollouts, of which the groups of the labels with a limit "
        f"(train.rollout_group_limits) hold up to {capped} ({groups}): {waiting}, and every group after it with it, "
        f"so those labels set the run's pace; raise it to {needed} at least"
    ]


def listed_permit_warnings(config: Any, settings: Mapping[str, EntrySettings], train_tasks: Any) -> list[str]:
    """:func:`permit_warnings` for a label column other than each task's entry, whose labels the listed training tasks
    give (``train_tasks``); ``platoon train`` logs it once it has listed them, and ``platoon config validate --tasks``
    prints it.  The default column's labels are the entries' names, which :func:`warnings` checks at load."""

    column = label_column_of(config)
    if column == ENVIRONMENT_COLUMN:
        return []
    labels = tuple(settings)
    entries_by_label: dict[str | None, set[str]] = {}
    for row in train_tasks if train_tasks is not None else ():
        entry = row_entry(row, labels)
        if entry is not None:
            entries_by_label.setdefault(task_label(row, column), set()).add(entry)
    return permit_warnings(config, settings, entries_by_label)


def log_warnings(config: Any, settings: Mapping[str, EntrySettings]) -> None:
    """Log :func:`warnings`, and one line per entry with settings of its own (:func:`summary`)."""

    for line in summary(config, settings):
        logger.info("%s", line)
    for line in warnings(config, settings):
        logger.warning("%s", line)


# --- the workflow -----------------------------------------------------------------------------------------------------


def sets_settings(environments: Sequence[Any]) -> list[str]:
    """The labels of the entries that set training settings of their own (``workflow_config`` or
    ``eval_overrides.workflow_config``)."""

    return [
        entry.label
        for entry in environments
        if getattr(entry, "workflow_config", None) or getattr(entry.eval_overrides, "workflow_config", None)
    ]


def workflow_problems(environments: Sequence[Any], workflow_cls: Any, reference: str) -> list[str]:
    """An entry's training settings beside a workflow class that does not apply them: one that is not a
    ``GroupRolloutWorkflow`` and does not declare ``applies_entry_settings = True`` never reads them, so they would be
    ignored without a word."""

    labels = sets_settings(environments)
    if not labels or getattr(workflow_cls, "applies_entry_settings", False):
        return []
    return [
        f"environments[{label}].workflow_config: the run's workflow, {reference}, does not apply an entry's settings "
        "(it is no GroupRolloutWorkflow, and does not declare applies_entry_settings = True), so they would be "
        "ignored; set them in train.workflow_config, or use a workflow that applies them"
        for label in labels
    ]


# --- what platoon eval applies ----------------------------------------------------------------------------------------


def _evaluation_phase(config: Any, split: str | None) -> str:
    """The phase of an evaluation config's rollouts of ``split`` (its own ``split`` when None): ``train`` for the
    train split, whose rollouts take no ``eval_overrides``, else ``eval``."""

    from platoon.options import phase_of

    return phase_of(split if split is not None else str(getattr(config, "split", "eval")))


def evaluation_rollout_configs(config: Any, split: str | None = None) -> dict[str, Any]:
    """Each entry's rollout config in ``platoon eval``, by label: ``RolloutConfig``'s defaults, the entry's
    ``workflow_config.rollout_config``, the keys ``inference.workflow.rollout_config`` sets, then the entry's
    ``eval_overrides.workflow_config.rollout_config``, which apply to every split but ``train``, as an entry's other
    ``eval_overrides`` do (``split``: the config's own when None).  An entry that sets none of them gets
    ``inference.workflow.rollout_config`` itself, and so does every entry of a config built in Python, whose
    ``inference.workflow`` is taken as given, whole, below each entry's ``eval_overrides``.  ``judges`` is the run's,
    which ``platoon eval`` sets from ``inference.judges``."""

    from platoon.utils.config import given_mapping, lay_over

    workflow = config.inference.workflow
    run = workflow.rollout_config
    given = given_mapping(workflow)
    overrides_apply = _evaluation_phase(config, split) != "train"
    configs: dict[str, Any] = {}
    for entry in getattr(config, "environments", None) or []:
        own = entry.workflow_config.get("rollout_config") or {}
        override = (entry.eval_overrides.workflow_config.get("rollout_config") or {}) if overrides_apply else {}
        if not own and not override:
            configs[entry.label] = run
            continue
        if given is None:
            base = run
        else:
            base = lay_over(type(run)(), own) if own else type(run)()
            phase = given.get("rollout_config") or {}
            base = lay_over(base, phase) if phase else base
        laid = lay_over(base, override) if override else base
        laid.judges = run.judges
        configs[entry.label] = laid
    return configs


def evaluation_unapplied(config: Any, split: str | None = None) -> list[str]:
    """The entry keys ``platoon eval`` does not apply, one line each: it applies an entry's rollout settings only, so
    the rest of its ``workflow_config`` and ``eval_overrides.workflow_config``, and its ``max_staleness``, are
    training's; and the train split (``split``, else the config's own) takes no ``eval_overrides`` at all."""

    found = []
    train_split = _evaluation_phase(config, split) == "train"
    for entry in getattr(config, "environments", None) or []:
        where = f"environments[{entry.label}]"
        if train_split and entry.eval_overrides.workflow_config.get("rollout_config"):
            found.append(
                f"{where}.eval_overrides.workflow_config.rollout_config: the train split's rollouts take no "
                "eval_overrides, which apply to every other split"
            )
        for path, settings in (
            ("workflow_config", entry.workflow_config),
            ("eval_overrides.workflow_config", entry.eval_overrides.workflow_config),
        ):
            for key in settings:
                if key != "rollout_config":
                    found.append(
                        f"{where}.{path}.{key}: platoon eval applies an entry's rollout settings only "
                        "(rollout_config); inference.workflow sets how its rollouts run, such as "
                        "num_rollouts_per_task"
                    )
        if entry.max_staleness is not None:
            found.append(f"{where}.max_staleness: a training setting, which platoon eval does not apply")
    return found


# --- what an entry runs with, and where each value came from ----------------------------------------------------------


def _flat(mapping: Any, prefix: str = "") -> dict[str, Any]:
    """A settings mapping as ``{dotted path: value}``, nested sections (``rollout_config``, ``inference_params``)
    walked, ``extra`` and other free-form mappings kept whole."""

    if not isinstance(mapping, Mapping):
        return {}
    flat: dict[str, Any] = {}
    for key, value in mapping.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping) and key in ("rollout_config", "inference_params"):
            flat.update(_flat(value, f"{path}."))
        else:
            flat[path] = value
    return flat


def _value(section: Any, path: str) -> Any:
    """The value at dotted ``path`` of a built section; :data:`_MISSING` when it has none."""

    value = section
    for part in path.split("."):
        value = getattr(value, part, _MISSING)
        if value is _MISSING:
            return _MISSING
    return value


def _text(value: Any) -> str:
    """A value as a config writes it: ``null``, ``true``, ``8``, ``{temperature: 1.0}``."""

    if value is None or isinstance(value, bool):
        return json.dumps(value)
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{key}: {_text(item)}" for key, item in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_text(item) for item in value) + "]"
    return str(value)


@dataclass(frozen=True)
class _Layer:
    """One layer of an entry's settings: its name, and the values it sets, by dotted path."""

    name: str
    values: Mapping[str, Any]


def _named_settings(entry: Any) -> dict[str, Any]:
    """The settings the entry's named environment (``env``) ships, by dotted path; none without one, or for one that
    does not resolve here."""

    ref = getattr(entry, "env", None)
    if not ref:
        return {}
    try:
        from platoon.envs.named import get_environment

        return _flat(get_environment(ref).entry().get("workflow_config") or {})
    except Exception:  # a config show without the extension installed: the entry's layer holds those values
        return {}


def _lines(
    effective: Any, layers: Sequence[_Layer], paths: Sequence[str], base: Any, base_sources: Mapping[str, str]
) -> list[str]:
    """One line per path: its effective value, the highest of ``layers`` that sets it (else where ``base``, the
    section the layers are laid over, has it from: ``base_sources``, else ``default``), and the value that layer
    replaced, when it differs."""

    lines = []
    for path in paths:
        value = _value(effective, path)
        if value is _MISSING:
            continue
        setters = [layer for layer in layers if path in layer.values]
        source = setters[-1].name if setters else base_sources.get(path, "default")
        note = ""
        if setters:
            if len(setters) > 1:
                below, below_name = setters[-2].values[path], setters[-2].name
            else:
                below, below_name = _value(base, path), base_sources.get(path, "default")
            if below is not _MISSING and _text(below) != _text(value):
                note = f"({below_name}: {_text(below)})"
        lines.append(_line(f"{path}: {_text(value)}", source, note))
    return lines


def _line(setting: str, source: str, note: str = "") -> str:
    """One line of an entry's block: a setting, the layer it came from and a note, in columns."""

    return f"  {setting:<60} {source:<24} {note}".rstrip()


def _defaults(section: Any) -> Any:
    """A section of ``section``'s class as its defaults build it; None for a class that needs arguments."""

    try:
        return type(section)()
    except TypeError:
        return None


def _training_layers(config: Any, entry: Any) -> tuple[list[_Layer], dict[str, str]]:
    """The layers of an entry's training settings above the defaults, and the layer each path's value comes from."""

    from platoon.utils.config import given_mapping

    named = _named_settings(entry)
    own = {path: value for path, value in _flat(entry.workflow_config).items() if named.get(path, _MISSING) != value}
    layers = [
        _Layer("train.workflow_config", _flat(given_mapping(config.train.workflow_config) or {})),
        _Layer(
            "named environment", {path: value for path, value in named.items() if path in _flat(entry.workflow_config)}
        ),
        _Layer("entry", own),
    ]
    sources = {}
    for path in GROUP_PATHS:
        setters = [layer.name for layer in layers if path in layer.values]
        sources[path] = setters[-1] if setters else "default"
    return layers, sources


def describe(config: Any, label: str, settings: Mapping[str, EntrySettings] | None = None) -> list[str]:
    """What an entry of a training or an evaluation config runs with, in one block per phase the config runs:
    its training groups and its evaluation rollouts during training, or its rollouts in ``platoon eval``.  Each line
    is a per-group key with its effective value, the layer it came from (``default``, ``train.workflow_config``, the
    ``named environment``, the ``entry``, ``eval.workflow_config`` or ``inference.workflow``, ``eval_overrides``) and
    the value that layer replaced, then the entry's staleness bound and the straggler policy of its label."""

    from platoon.utils.config import given_mapping

    entry = next(entry for entry in config.environments if entry.label == label)
    blocks: list[str] = []
    if getattr(config, "train", None) is not None:
        from platoon.train.tinker.config_defs import eval_workflow_base, evaluates

        settings = settings if settings is not None else entry_settings(config)
        own = settings[label]
        layers, sources = _training_layers(config, entry)
        blocks.append(f"environments[{label}], training groups:")
        blocks += _lines(own.train, layers, GROUP_PATHS, _defaults(config.train.workflow_config), {})
        bound = config.train.max_staleness
        if entry.max_staleness is not None:
            note = f"(train.max_staleness: {_text(bound)}, which bounds how far rollouts run ahead)"
            blocks.append(_line(f"max_staleness: {entry.max_staleness}", "entry", note))
        else:
            blocks.append(_line(f"max_staleness: {_text(bound)}", "train.max_staleness"))
        from platoon.train.admissions import describe_line

        if (admission := describe_line(config, label)) is not None:  # under a group admission rule, or set here
            blocks.append(_line(*admission))
        blocks += _policy_lines(config, label, "train")
        if own.eval is not None and evaluates(config):
            eval_given = given_mapping(config.eval.workflow_config)
            eval_layers = [
                # A section built in Python is taken as given, whole.
                _Layer("eval.workflow_config", _flat(eval_given) if eval_given is not None else {}),
                _Layer("eval_overrides", _flat(entry.eval_overrides.workflow_config)),
            ]
            base = eval_workflow_base({"train": SimpleNamespace(workflow_config=own.train)}, _eval_default(config))
            inherited = {path: f"{sources[path]}, inherited" for path in ROLLOUT_PATHS if sources[path] != "default"}
            blocks.append(f"environments[{label}], evaluation during training:")
            blocks += _lines(own.eval, eval_layers, GROUP_PATHS, base, inherited)
            blocks += _policy_lines(config, label, "eval")
    elif getattr(config, "inference", None) is not None:
        rollout = evaluation_rollout_configs(config)[label]
        named = _named_settings(entry)
        own_rollout = _flat({"rollout_config": entry.workflow_config.get("rollout_config") or {}})
        train_split = _evaluation_phase(config, None) == "train"
        layers = [
            _Layer("named environment", {path: value for path, value in named.items() if path in own_rollout}),
            _Layer("entry", {path: value for path, value in own_rollout.items() if named.get(path, _MISSING) != value}),
            _Layer("inference.workflow", _flat(given_mapping(config.inference.workflow) or {})),
            # The train split's rollouts take no eval_overrides.
            _Layer("eval_overrides", {} if train_split else _flat(entry.eval_overrides.workflow_config)),
        ]
        blocks.append(f"environments[{label}], platoon eval rollouts{' of the train split' if train_split else ''}:")
        defaults = SimpleNamespace(rollout_config=_defaults(rollout))
        blocks += _lines(SimpleNamespace(rollout_config=rollout), layers, ROLLOUT_PATHS, defaults, {})
        blocks += [f"  not applied: {line}" for line in evaluation_unapplied(config) if f"[{label}]" in line]
    return blocks


def _policy_lines(config: Any, label: str, section: str) -> list[str]:
    """The line that says which straggler policy by label an entry's groups get in ``section`` (``train`` or
    ``eval``), when the run has any."""

    policies = getattr(getattr(getattr(config, section, None), "workflow_config", None), "straggler_policies", None)
    if not policies:
        return []
    column = label_column_of(config)
    key = f"{section}.workflow_config"
    if column != ENVIRONMENT_COLUMN:
        return [
            f"  straggler_policies: by train.label_by ({column}); a label's policy wins over straggler_timeout_seconds "
            "and straggler_quorum for the groups it labels"
        ]
    policy = policies.get(label)
    if policy is None:
        return [_line("straggler_policies: none for its label", key)]
    text = f"{{timeout_seconds: {_text(policy.timeout_seconds)}, quorum: {_text(policy.quorum)}}}"
    return [
        _line(f"straggler_policies.{label}: {text}", key, "(wins over straggler_timeout_seconds and straggler_quorum)")
    ]


def differences(config: Any, settings: Mapping[str, EntrySettings]) -> dict[str, dict[str, Any]]:
    """What differs from the run's in each entry's settings, by label: ``train`` and ``eval``, each ``{dotted path:
    value}`` of the per-group keys whose value differs from the run's section, and ``max_staleness`` when it differs
    from ``train.max_staleness``.  Entries whose settings are the run's are left out."""

    record: dict[str, dict[str, Any]] = {}
    run_eval = getattr(getattr(config, "eval", None), "workflow_config", None)
    for label, own in settings.items():
        entry: dict[str, Any] = {}
        for phase, run in (("train", config.train.workflow_config), ("eval", run_eval)):
            ours = getattr(own, phase)
            if run is None or ours is None or ours is run:
                continue
            changed = {
                path: _plain(value)
                for path in GROUP_PATHS
                if (value := _value(ours, path)) is not _MISSING and _plain(value) != _plain(_value(run, path))
            }
            if changed:
                entry[phase] = changed
        if own.max_staleness != config.train.max_staleness:
            entry["max_staleness"] = own.max_staleness
        if entry:
            record[label] = entry
    return record


def lock_record(config: Any, settings: Mapping[str, EntrySettings]) -> dict[str, dict[str, Any]]:
    """What ``lock.json`` records of a training run's entry settings (``entry_settings``): :func:`differences`."""

    return differences(config, settings)


def summary(config: Any, settings: Mapping[str, EntrySettings]) -> list[str]:
    """One line per entry whose settings differ from the run's: what differs (:func:`differences`), and its own group
    admission rule when that is not the run's (:func:`platoon.train.admissions.own_rule_text`)."""

    from platoon.train.admissions import own_rule_text

    found = differences(config, settings)
    lines = []
    for label in settings:
        entry = found.get(label, {})
        parts = [f"{path} {_text(value)}" for path, value in entry.get("train", {}).items()]
        if "max_staleness" in entry:
            parts.append(f"max_staleness {_text(entry['max_staleness'])}")
        if (rule := own_rule_text(config, label)) is not None:
            parts.append(rule)
        evaluation = [f"{path} {_text(value)}" for path, value in entry.get("eval", {}).items()]
        if evaluation:
            parts.append("in evaluation: " + ", ".join(evaluation))
        if parts:
            lines.append(f"environments[{label}]: {', '.join(parts)}")
    return lines


def _plain(value: Any) -> Any:
    """A setting's value as JSON data."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {item.name: _plain(getattr(value, item.name)) for item in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value
