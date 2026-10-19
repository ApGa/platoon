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
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    # task_order: weighted) the whole batches its tasks fill.
    epoch_size: int
    # Training steps the run plans (num_epochs and max_training_steps, as the trainer counts them); None without either.
    steps: int | None

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
        """The plan, one line for the order and one per entry."""

        span = f"over the run's {self.steps} steps" if self.slots is not None else "per epoch"
        batches = f" ({self.epoch_size // self.batch_size} batches of {self.batch_size})" if self.batch_size else ""
        steps = f", {self.steps} steps planned" if self.steps is not None else ""
        lines = [f"training tasks: {self.tasks}, drawn in epochs of {self.epoch_size}{batches}{steps}"]
        for entry in self.entries:
            lines.append(
                f"environments[{entry.label}]: {entry.tasks} tasks, weight {_number(entry.weight)}, "
                f"{entry.share:.1%} of the slots: each task drawn about {_number(self.draws(entry))} times {span}"
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
) -> SamplingPlan:
    """The plan of a run whose ``environments`` list ``tasks`` training tasks each (by label) and share the slots by
    ``weights`` (numbers, by label), in epochs of ``epoch_size`` tasks; ``train`` is the config's ``train`` section
    (None for a config without one)."""

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
    return SamplingPlan(entries, batch_size, epoch_size, planned_steps(train, batches) if batches else None)


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


def _number(value: float | str) -> str:
    """A weight or a count as a config would write it: ``tasks``, ``2``, ``0.5``, ``116.4``."""

    if isinstance(value, str):
        return value
    if math.isclose(value, round(value), rel_tol=0, abs_tol=1e-9):
        return str(round(value))
    return f"{value:.1f}" if value >= 1 else f"{value:.2g}"
