"""How a run draws its training tasks: each environment's share, and how often each of its tasks is drawn.

:class:`SamplingPlan` describes the order a run draws its training tasks in (``AutoDataset.training`` builds it as it
lists them): for each environment entry its training tasks and its share of the slots, an epoch's size, and the run's
planned training steps, counted as the trainer counts them (:func:`planned_steps`).  ``platoon train`` logs it once it
has listed the tasks, and ``platoon config validate --tasks`` prints it.  Both warn of a mixture whose weights:

- **leave tasks out of the run**: an entry's share of the run's slots (its steps times the batch size) is smaller
  than its training tasks, in a run long enough to draw every task once with ``weight: tasks``, so its weight and
  not the run's length leaves the rest of its tasks out; or
- **draw some tasks far more often than others**: one entry's tasks are each drawn more than
  :data:`REPEAT_RATIO_LIMIT` times as often as another entry's.

Each warning says the numbers and the fix: ``weight: tasks`` on every entry draws every task equally often.

The plan names each task sampler other than the defaults (:mod:`platoon.train.samplers`): an entry's own sampler
under ``weighted`` (``pass_rate``, a custom one), which fills that entry's slots, and a run's sampler that chooses
among every task of the run; and, in a run under a group admission rule, each entry's rule and what a step trains
(:mod:`platoon.train.admissions`).

:func:`sampling_fingerprint` sums up what the order follows from (the sampler, the batch size, the label column, each
entry's weight, seed and training tasks, and the samplers platoon 0.3.0 did not have, with their options), which each
full checkpoint saves; a start that resumes compares its own with it, and :func:`fingerprint_changes` says what
changed, so that the run warns of a change and goes on with it.  A run under a group admission rule adds its rules and
its cap (``admission``); a run without one has no such key, so its fingerprint is what it was before admission
existed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# How many times as often one entry's tasks may each be drawn as another entry's before a mixture is warned of.
REPEAT_RATIO_LIMIT = 4.0
_FIX = "weight: tasks on every entry draws every task of the mixture equally often"


@dataclass(frozen=True)
class EntryShare:
    """One environment entry of the plan."""

    label: str
    # Its training tasks, once its selection, exclude_tasks and a hold-out have taken theirs.
    tasks: int
    # Its weight as the config sets it: a number, or "tasks".
    weight: float | str
    # Its share of the slots, 0 to 1.
    share: float


@dataclass(frozen=True)
class SamplingPlan:
    """How a run draws its training tasks: the entries' shares, the epochs and the planned steps."""

    entries: tuple[EntryShare, ...]
    # Tasks per training batch (None: not a training config's order).
    batch_size: int | None
    # Tasks per epoch: a mixture's as many as it holds, filled to whole batches; a single environment's (without
    # task_sampler: weighted) the whole batches its tasks fill.
    epoch_size: int
    # Training steps the run plans (num_epochs and max_training_steps, as the trainer counts them); None without either.
    steps: int | None
    # The task samplers other than the defaults, as (name, adaptive): an entry's own by its label, and the run's under
    # None when it chooses among every task of the run (pass_rate, a custom one).
    samplers: Mapping[str | None, tuple[str, bool]] = field(default_factory=dict)
    # The line that names a run's group admission rules and what a step trains; None without a rule.
    admission: str | None = None

    @property
    def tasks(self) -> int:
        return sum(entry.tasks for entry in self.entries)

    @property
    def slots(self) -> int | None:
        """The task slots of the whole run, ``steps`` batches; None when either is unknown."""

        return self.steps * self.batch_size if self.steps is not None and self.batch_size else None

    def draws(self, entry: EntryShare) -> float:
        """How many times each of ``entry``'s tasks is drawn over the run, or in one epoch when the run's length is
        unknown."""

        slots = self.slots if self.slots is not None else self.epoch_size
        return entry.share * slots / entry.tasks if entry.tasks else 0.0

    def describe(self) -> list[str]:
        """The plan, one line for the order and one per entry; each sampler other than the defaults is named, an
        adaptive one marked experimental; and the group admission rules, when the run has one."""

        lines = self._describe()
        return [*lines, self.admission] if self.admission is not None else lines

    def _describe(self) -> list[str]:
        span = f"over the run's {self.steps} steps" if self.slots is not None else "per epoch"
        batches = f" ({self.epoch_size // self.batch_size} batches of {self.batch_size})" if self.batch_size else ""
        steps = f", {self.steps} steps planned" if self.steps is not None else ""
        run = self.samplers.get(None)
        if run is not None:
            lines = [
                f"training tasks: {self.tasks}, drawn by {_sampler(run)} over every entry's tasks, in epochs of "
                f"{self.epoch_size}{batches}{steps}"
            ]
            for entry in self.entries:
                lines.append(
                    f"environments[{entry.label}]: {entry.tasks} tasks, weight {_number(entry.weight)}: each task "
                    f"drawn about {_number(self.draws(entry))} times {span} on average"
                )
            return lines
        lines = [f"training tasks: {self.tasks}, drawn in epochs of {self.epoch_size}{batches}{steps}"]
        for entry in self.entries:
            own = self.samplers.get(entry.label)
            if own is None:
                lines.append(
                    f"environments[{entry.label}]: {entry.tasks} tasks, weight {_number(entry.weight)}, "
                    f"{entry.share:.1%} of the slots: each task drawn about {_number(self.draws(entry))} times {span}"
                )
                continue
            when = " when its group can start" if own[1] else ""
            lines.append(
                f"environments[{entry.label}]: {entry.tasks} tasks, weight {_number(entry.weight)}, "
                f"{entry.share:.1%} of the slots, each filled by {_sampler(own)}{when}: its tasks drawn about "
                f"{_number(self.draws(entry))} times each {span} on average"
            )
        return lines

    def warnings(self) -> list[str]:
        """What the plan warns of: a mixture's weights that leave tasks out of the run, or that draw one entry's tasks
        more than :data:`REPEAT_RATIO_LIMIT` times as often as another's.  None for a single entry."""

        entries = [entry for entry in self.entries if entry.tasks]
        if len(entries) < 2:
            return []
        found = []
        slots = self.slots
        if slots is not None and slots >= self.tasks:  # long enough for every task once with weight: tasks
            for entry in entries:
                reached = entry.share * slots
                if reached < entry.tasks:
                    found.append(
                        f"environments[{entry.label}]: its weight ({_number(entry.weight)}, {entry.share:.1%} of the "
                        f"slots) gives it about {reached:.0f} of the run's {slots} task slots ({self.steps} steps of "
                        f"{self.batch_size}), fewer than its {entry.tasks} training tasks, so about "
                        f"{entry.tasks - reached:.0f} of them never train; {_FIX}"
                    )
        most = max(entries, key=lambda entry: entry.share / entry.tasks)
        least = min(entries, key=lambda entry: entry.share / entry.tasks)
        ratio = (most.share / most.tasks) / (least.share / least.tasks)
        if ratio > REPEAT_RATIO_LIMIT:
            span = f"over the run's {self.steps} steps" if slots is not None else "per epoch"
            found.append(
                f"environments[{most.label}]'s {most.tasks} training tasks are each drawn about {ratio:.1f} times as "
                f"often as environments[{least.label}]'s {least.tasks} (about {_number(self.draws(most))} and "
                f"{_number(self.draws(least))} times each {span}, at weights {_number(most.weight)} and "
                f"{_number(least.weight)}); {_FIX}"
            )
        return found


def sampling_plan(
    environments: Sequence[Any],
    tasks: Mapping[str, int],
    weights: Mapping[str, float],
    *,
    train: Any,
    epoch_size: int,
    samplers: Mapping[str | None, tuple[str, bool]] | None = None,
) -> SamplingPlan:
    """The plan of a run whose ``environments`` list ``tasks`` training tasks each (by label) and share the slots by
    ``weights`` (numbers, by label), in epochs of ``epoch_size`` tasks; ``train`` is the config's ``train`` section
    (None for a config without one); ``samplers`` names the task samplers other than the defaults
    (:attr:`SamplingPlan.samplers`)."""

    total = sum(weights[environment.label] for environment in environments if tasks.get(environment.label))
    entries = tuple(
        EntryShare(
            label=environment.label,
            tasks=tasks.get(environment.label, 0),
            weight=environment.weight,
            share=weights[environment.label] / total if total and tasks.get(environment.label) else 0.0,
        )
        for environment in environments
    )
    batch_size = getattr(train, "batch_size", None)
    batches = epoch_size // batch_size if batch_size else None
    steps = planned_steps(train, batches) if batches else None
    admission = None
    if train is not None:
        from platoon.train.admissions import plan_line

        admission = plan_line(train, environments)
    return SamplingPlan(entries, batch_size, epoch_size, steps, dict(samplers or {}), admission)


def planned_steps(train: Any, batches_per_epoch: int) -> int | None:
    """The training steps a run plans with ``batches_per_epoch`` batches an epoch: ``train.num_epochs`` epochs or
    ``train.max_training_steps`` steps, the shorter of the two when both are set; None when neither is."""

    epochs = getattr(train, "num_epochs", None)
    max_steps = getattr(train, "max_training_steps", None)
    epoch_steps = batches_per_epoch * epochs if epochs is not None else None
    if epoch_steps is not None and max_steps is not None:
        return min(epoch_steps, max_steps)
    return epoch_steps if epoch_steps is not None else max_steps


def log_sampling_plan(plan: SamplingPlan | None) -> None:
    """Log a run's plan, and warn of what :meth:`SamplingPlan.warnings` finds."""

    if plan is None:
        return
    for line in plan.describe():
        logger.info("%s", line)
    for warning in plan.warnings():
        logger.warning("%s", warning)


def _sampler(sampler: tuple[str, bool]) -> str:
    name, adaptive = sampler
    return f"{name} (experimental)" if adaptive else name


def _number(value: float | str) -> str:
    """A weight or a count as a config would write it: ``tasks``, ``2``, ``0.5``, ``116.4``."""

    if isinstance(value, str):
        return value
    if math.isclose(value, round(value), rel_tol=0, abs_tol=1e-9):
        return str(round(value))
    return f"{value:.1f}" if value >= 1 else f"{value:.2g}"


def sampling_fingerprint(
    config: Any, tasks: Iterable[Mapping[str, Any]], sampling: Any = None, *, admission: Any = None
) -> dict[str, Any]:
    """What the order a training run draws its tasks in follows from: ``train.task_sampler``, the batch size, the label
    column (which ``rollout_group_limits`` key on), the environment entries in the order the config lists them (a
    weighted order seeds and breaks ties by it, and ``uniform`` lists the pool in it), and for each entry its weight,
    its seed when a sampler reads it, and its training tasks (``tasks``, the run's pool: their number and a digest of
    their ids, in listing order).  With the samplers the run resolved (``sampling``, a
    :class:`~platoon.train.sampler_runtime.TaskSampling`), also those platoon 0.3.0 did not have, with their options
    parsed, defaults included: the run's options (``task_sampler_options``) when its sampler is neither ``auto``,
    ``weighted`` nor ``uniform``, and each entry's sampler and options when the run's sampler hands it slots and it is
    not ``uniform``.  Each such key is written only for such a sampler, so a config platoon 0.3.0 could load keeps
    0.3.0's fingerprint.  With the group admission rules the run resolved (``admission``, an
    :class:`~platoon.train.admissions.AdmissionChoices`) in rule mode, also ``admission``: the cap, and each entry's
    rule and options, parsed, defaults included (none in a run whose every rule is ``all``)."""

    from platoon.train.components import row_entry
    from platoon.train.labels import label_column
    from platoon.train.samplers import RUN_SAMPLERS, STATIC_RUN_SAMPLERS

    train = getattr(config, "train", None)
    environments = list(getattr(config, "environments", None) or [])
    sampler = str(getattr(train, "task_sampler", None) or "auto")
    labels = tuple(environment.label for environment in environments)
    ids: dict[str | None, list[str]] = {}
    for row in tasks:
        ids.setdefault(row_entry(row, labels), []).append(str(row.get("task_id")))
    run = getattr(sampling, "run", None)
    chosen: Mapping[str, Any] = getattr(sampling, "entries", None) or {}
    own = {label: choice for label, choice in chosen.items() if choice.name != "uniform"}
    custom_run = run is not None and sampler not in RUN_SAMPLERS
    # The entries' seeds shuffle a weighted order's entries, and key the entries' own samplers; a single environment's
    # draw (auto), uniform and pass_rate over every task ignore them, and a custom run sampler may read them.
    seeded = sampler == "weighted" or (sampler == "auto" and len(environments) > 1) or bool(own) or custom_run
    entries: dict[str, Any] = {}
    for environment in environments:
        listed = ids.get(environment.label, [])
        entry: dict[str, Any] = {"weight": environment.weight, "tasks": len(listed), "pool": _digest(listed)}
        if seeded:
            entry["seed"] = environment.seed
        if environment.label in own:
            entry["task_sampler"] = own[environment.label].name
            entry["task_sampler_options"] = own[environment.label].options_json()
        entries[environment.label] = entry
    fingerprint: dict[str, Any] = {
        "task_sampler": sampler,
        "batch_size": getattr(train, "batch_size", None),
        "label_by": label_column(config),
        "order": list(labels),
        "entries": entries,
    }
    if run is not None and sampler not in STATIC_RUN_SAMPLERS:
        fingerprint["task_sampler_options"] = run.options_json()
    from platoon.train.admissions import fingerprint as admission_fingerprint

    if (rules := admission_fingerprint(admission)) is not None:
        fingerprint["admission"] = rules
    return fingerprint


def fingerprint_changes(saved: Mapping[str, Any] | None, now: Mapping[str, Any]) -> list[str]:
    """What changed from the inputs a checkpoint drew with (``saved``, its :func:`sampling_fingerprint`) to a resume's
    (``now``), one line each; none when nothing did, or when ``saved`` says nothing (a checkpoint that saved none)."""

    if not isinstance(saved, Mapping) or not saved:
        return []
    changes = []
    for key, name in (
        ("task_sampler", "train.task_sampler"),
        ("batch_size", "train.batch_size"),
        ("label_by", "train.label_by"),
    ):
        if saved.get(key) != now.get(key):
            changes.append(f"{name}: {saved.get(key)} -> {now.get(key)}")
    if saved.get("task_sampler_options", {}) != now.get("task_sampler_options", {}):
        changes.append(
            f"train.task_sampler_options: {_json(saved.get('task_sampler_options', {}))} -> "
            f"{_json(now.get('task_sampler_options', {}))}"
        )
    before, after = saved.get("entries") or {}, now.get("entries") or {}
    old_order, new_order = saved.get("order"), now.get("order")
    if isinstance(old_order, list) and isinstance(new_order, list):
        # The entries both list, in the order each lists them: a joined or a departed entry is a change of its own.
        if [label for label in old_order if label in new_order] != [label for label in new_order if label in old_order]:
            changes.append(f"environments: order {', '.join(map(str, old_order))} -> {', '.join(map(str, new_order))}")
    for label in [*before, *(label for label in after if label not in before)]:
        if label not in after:
            changes.append(f"environments[{label}] left the run")
            continue
        if label not in before:
            changes.append(f"environments[{label}] joined the run")
            continue
        old, new = before[label], after[label]
        for key in ("weight", "seed"):
            if key in old and key in new and old[key] != new[key]:
                changes.append(f"environments[{label}].{key}: {old[key]} -> {new[key]}")
        # A missing sampler reads as the default, uniform without options.
        if old.get("task_sampler", "uniform") != new.get("task_sampler", "uniform"):
            changes.append(
                f"environments[{label}].task_sampler: {old.get('task_sampler', 'uniform')} -> "
                f"{new.get('task_sampler', 'uniform')}"
            )
        if old.get("task_sampler_options", {}) != new.get("task_sampler_options", {}):
            changes.append(
                f"environments[{label}].task_sampler_options: {_json(old.get('task_sampler_options', {}))} -> "
                f"{_json(new.get('task_sampler_options', {}))}"
            )
        if old.get("pool") != new.get("pool"):
            if old.get("tasks") != new.get("tasks"):
                changes.append(f"environments[{label}]: its training tasks, {old.get('tasks')} -> {new.get('tasks')}")
            else:
                changes.append(f"environments[{label}]: its training tasks (as many, {new.get('tasks')}, but others)")
    return changes + _admission_changes(saved, now)


def _admission_changes(saved: Mapping[str, Any], now: Mapping[str, Any]) -> list[str]:
    """What changed of the group admission rules (the fingerprints' ``admission``): a missing key reads as every entry
    at ``all``.  Of the entries both fingerprints list: a rule that changed, or the options of one that did not; and the
    cap when both are under a rule (a fingerprint without the key holds no cap: a run without a rule has none)."""

    before, after = saved.get("admission") or {}, now.get("admission") or {}
    if not before and not after:
        return []
    changes = []
    old_cap, new_cap = before.get("max_groups_per_step"), after.get("max_groups_per_step")
    if before and after and old_cap != new_cap:
        changes.append(f"train.refill.max_groups_per_step: {old_cap} -> {new_cap}")
    listed = [label for label in (saved.get("entries") or {}) if label in (now.get("entries") or {})]
    old_rules, new_rules = before.get("entries") or {}, after.get("entries") or {}
    for label in listed or [""]:
        old = old_rules.get(label) or {"rule": "all", "options": {}}
        new = new_rules.get(label) or {"rule": "all", "options": {}}
        where = f"environments[{label}]" if label else "train"
        if old.get("rule") != new.get("rule"):
            changes.append(f"{where}.group_admission: {old.get('rule')} -> {new.get('rule')}")
        elif old.get("options", {}) != new.get("options", {}):
            changes.append(
                f"{where}.group_admission_options: {_json(old.get('options', {}))} -> {_json(new.get('options', {}))}"
            )
    return changes


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _digest(task_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(task_ids).encode()).hexdigest()[:16]
