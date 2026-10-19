"""The column that labels a run's tasks: ``train.label_by``.

Each task of a run is a dataset row: its ``task_id`` and the columns its environment gives it, such as the mixture
entry it belongs to (``_platoon_environment``, :data:`~platoon.train.components.ENVIRONMENT_COLUMN`).  A run labels
its tasks by one column, ``train.label_by``, by default that entry.  A task's label is the column's value as text
(:func:`label_text`); a task without the column, or with an empty value, has none.

An environment service may give its tasks columns of their own (``task_columns``): a Harbor dataset gives each task
its index fields.  A listing keeps them as text, and leaves out a column named like one of platoon's own
(``task_id``, ``_platoon*``), which is an error when one of the run's keys that read columns names it
(:func:`column_keys`): the key would read platoon's column, not the task's.

Labels key ``train.rollout_group_limits`` and the workflow configs' ``straggler_policies``, and a run's metrics break
down by them: each training step logs every label's groups, reward and outcome counts under ``batch/<column>/<label>``
(:func:`metric_prefix`), and each evaluation during training under ``<column>/<label>`` in its ``eval/`` metrics.  A
column with more than :data:`MAX_METRIC_LABELS` labels logs only that many most frequent (:func:`metric_labels`); limits
and straggler policies apply to every label.  :func:`check_labels` checks the column against the tasks a run listed,
before anything runs.
"""

from __future__ import annotations

import difflib
import logging
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from platoon.train.components import ENVIRONMENT_COLUMN

logger = logging.getLogger(__name__)

# The most labels whose metrics a run logs: a column with more, such as a task id, logs only its most frequent ones.
MAX_METRIC_LABELS = 64


def label_text(value: Any) -> str | None:
    """A column's value as a label: a string as it is, ``true`` or ``false`` for a boolean, a number as Python
    writes it; None for a missing or empty value, and for anything but a scalar."""

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value or None
    if isinstance(value, int | float):
        return str(value)
    return None


def task_label(row: Any, column: str) -> str | None:
    """The label of the task a dataset row holds: its ``column`` value as text (:func:`label_text`)."""

    return label_text(row.get(column)) if isinstance(row, Mapping) else None


def dataset_columns(dataset: Any) -> list[str]:
    """The columns of a dataset's rows: a Hugging Face dataset's, or every key of a list's rows."""

    names = getattr(dataset, "column_names", None)
    if names is not None:
        return list(names)
    return list(dict.fromkeys(key for row in dataset if isinstance(row, Mapping) for key in row))


def dataset_labels(dataset: Any, column: str) -> list[str | None]:
    """Each task's label, in the dataset's order (None for a task without one); none for a dataset whose rows cannot
    be read, such as a custom one that is not iterable."""

    names = getattr(dataset, "column_names", None)
    if names is not None:
        if column not in names:
            return [None] * len(dataset)
        return [label_text(value) for value in dataset[column]]
    try:
        return [task_label(row, column) for row in dataset]
    except TypeError:
        return []


def label_column(config: Any) -> str:
    """The run's label column: ``train.label_by``, else each task's mixture entry."""

    return getattr(getattr(config, "train", None), "label_by", None) or ENVIRONMENT_COLUMN


def metric_labels(dataset: Any, column: str, *, split: str = "training") -> list[str]:
    """The labels whose metrics a run logs for the tasks of ``dataset``, the most frequent first (ties in label order):
    every label its tasks have, or the :data:`MAX_METRIC_LABELS` most frequent, with a warning, when they have more.
    Empty without a dataset."""

    if dataset is None:
        return []
    counts = Counter(label for label in dataset_labels(dataset, column) if label is not None)
    ranked = sorted(counts, key=lambda label: (-counts[label], label))
    if len(ranked) > MAX_METRIC_LABELS:
        logged = ranked[:MAX_METRIC_LABELS]
        logger.warning(
            "train.label_by: the %s tasks have %d %r values; their metrics break down by the %d most frequent, which "
            "label %d of the %d labelled tasks (group limits and straggler policies apply to every value)",
            split,
            len(ranked),
            column,
            MAX_METRIC_LABELS,
            sum(counts[label] for label in logged),
            counts.total(),
        )
        return logged
    return ranked


def metric_prefix(column: str, label: str) -> str:
    """Where a label's metrics go, below the step's ``batch/`` or an evaluation's ``eval/``: ``<column>/<label>``, with
    ``environment`` for the default column (each task's mixture entry), and ``/`` in either written as ``--``."""

    name = "environment" if column == ENVIRONMENT_COLUMN else column
    return f"{name.replace('/', '--')}/{label.replace('/', '--')}"


def column_keys(config: Any) -> dict[str, str]:
    """The columns a run's keys read, each with the key that names it: ``train.label_by``, unless it names its
    default, the mixture entry, which is platoon's own column.  Empty for a config without a ``train`` section, such as
    an evaluation's."""

    label_by = getattr(getattr(config, "train", None), "label_by", None)
    return {label_by: "train.label_by"} if isinstance(label_by, str) and label_by != ENVIRONMENT_COLUMN else {}


def check_labels(config: Any, train_dataset: Any, eval_dataset: Any = None) -> None:
    """Check the run's label column (``train.label_by``) against the tasks it listed, before anything runs.

    A :class:`~platoon.utils.config.ConfigError` when the column is one that no training task has, naming the
    columns they have: a misspelled column would label nothing, so no limit or straggler policy would ever apply.
    The default column, each task's mixture entry, is exempt: a single environment's tasks have none.  A warning
    when only some training tasks have a label, and for each limit or straggler policy for a label that no task of
    its split has.
    """

    from platoon.utils.config import ConfigError

    train = getattr(config, "train", None)
    column = label_column(config)
    labels = dataset_labels(train_dataset, column)
    labelled = sum(label is not None for label in labels)
    if not labelled and column != ENVIRONMENT_COLUMN:
        columns = dataset_columns(train_dataset)
        matches = difflib.get_close_matches(column, columns, n=1, cutoff=0.6)
        hint = f"; did you mean {matches[0]}?" if matches else ""
        raise ConfigError(
            f"train.label_by: no training task has a {column!r} value to label it by; their columns are "
            f"{', '.join(columns)}{hint}"
        )
    if labelled and labelled < len(labels):
        logger.warning(
            "train.label_by: %d of the %d training tasks have no %r value, so no group limit or straggler policy "
            "applies to them",
            len(labels) - labelled,
            len(labels),
            column,
        )
    _warn_of_unknown("train.rollout_group_limits", getattr(train, "rollout_group_limits", None), train_dataset, column)
    workflow = getattr(train, "workflow_config", None)
    _warn_of_unknown(
        "train.workflow_config.straggler_policies", getattr(workflow, "straggler_policies", None), train_dataset, column
    )
    if eval_dataset is not None:
        workflow = getattr(getattr(config, "eval", None), "workflow_config", None)
        _warn_of_unknown(
            "eval.workflow_config.straggler_policies",
            getattr(workflow, "straggler_policies", None),
            eval_dataset,
            column,
            split="evaluation",
        )


def _warn_of_unknown(
    key: str, settings: Iterable[str] | None, dataset: Any, column: str, *, split: str = "training"
) -> None:
    """Warn of the labels ``settings`` names (a mapping's keys) that no task of ``dataset`` has."""

    if not settings:
        return
    present = Counter(dataset_labels(dataset, column))
    if unknown := sorted(label for label in settings if label not in present):
        logger.warning(
            "%s: no %s task has the %r value %s, so the setting applies to no task",
            key,
            split,
            column,
            " or ".join(unknown),
        )
