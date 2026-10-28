"""What the run calls of the environment entries' credit estimators (RFC 0021, section 6), beside the API an estimator
is written against (:mod:`platoon.train.credit.api`): its ``assign`` over a group, under its time limit
(:func:`assign_credit`); its ``settle`` as the run files the group (:func:`settle_credit`); and its state in a full
checkpoint (:func:`credit_states`, :func:`load_credit_states`).  An estimator that raises in any of them, or whose
``assign`` does not return in time, stops the run with a :class:`CreditEstimatorError` that names it, its entry and
the group: a group it could not credit is never trained otherwise, nor counted as a failed rollout.  The one exception
is a policy's scoring that failed (:class:`~platoon.train.credit.api.ScoringFailed`: a teacher's service that raised
or did not answer in time), which :func:`assign_credit` lets through: the group fails, and the run goes on.

This module is part of the training code's contract (``tests/train/test_layers.py``): the rollout source, the runner
and the checkpoint saves, which are generic code, call the estimators through it.  It imports neither Tinker nor
torch.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from platoon.train.credit.api import DEFAULT_ASSIGN_TIMEOUT_SECONDS, CreditGroup, ScoringFailed, TrainingSample

logger = logging.getLogger(__name__)

# The loop_state key of a full checkpoint row that holds the state of the run's estimators that have state, by entry.
CREDIT_KEY = "credit"


class CreditEstimatorError(RuntimeError):
    """A credit estimator that raised, or whose ``assign`` did not return in time: the run stops with it."""


@dataclass(frozen=True)
class Credit:
    """An estimator's credit of one group: the estimator, the group it credited, and the group's rows as they were
    handed to it (``samples``, member by member), which stay in that order whatever the estimator does to the
    group's lists.  A group's output carries it (its ``credit``) until the run settles it, when the estimator has a
    ``settle``."""

    estimator: Any
    group: CreditGroup
    samples: tuple[TrainingSample, ...]


def class_ref(estimator: Any) -> str:
    """The ``module:Class`` of ``estimator``'s class, which names it in errors and in the checkpoint's state."""

    cls = type(estimator)
    return f"{cls.__module__}:{cls.__qualname__}"


async def assign_credit(estimator: Any, group: CreditGroup) -> Credit:
    """Await ``estimator.assign(group)``, within its ``assign_timeout_seconds`` (600 by default; None: no limit), and
    return its credit of the group.  A :class:`CreditEstimatorError` when it raises or runs out of time, the call
    cancelled; a policy's failed scoring (:class:`~platoon.train.credit.api.ScoringFailed`), which fails the group
    alone, and a cancellation of the group itself go on."""

    samples = tuple(group.samples())
    seconds = getattr(estimator, "assign_timeout_seconds", DEFAULT_ASSIGN_TIMEOUT_SECONDS)
    limit = asyncio.timeout(seconds)
    try:
        async with limit:
            await estimator.assign(group)
    except ScoringFailed:
        raise
    except Exception as exc:
        what = f"did not return within {seconds:g} s" if limit.expired() else f"raised {type(exc).__name__}: {exc}"
        what = f"assign(group) {what}, on task {group.task_id}"
        raise CreditEstimatorError(f"{_named(estimator, group.entry)}: {what}") from exc
    return Credit(estimator, group, samples)


def settle_credit(output: Any) -> None:
    """Settle the credit of a group as the run files it, once: the ``settle(group)`` of the estimator that credited it
    (the output's ``credit``, which an estimator without ``settle`` leaves None); nothing for any other output.  The
    run calls it in the section, with no await, that hands the group to the step that consumes it, where a full
    checkpoint takes the groups it keeps.  A :class:`CreditEstimatorError` when ``settle`` raises."""

    credit = getattr(output, "credit", None)
    if not isinstance(credit, Credit):
        return
    output.credit = None
    try:
        credit.estimator.settle(credit.group)
    except Exception as exc:
        what = f"settle(group) raised {type(exc).__name__}: {exc}, on task {credit.group.task_id}"
        raise CreditEstimatorError(f"{_named(credit.estimator, credit.group.entry)}: {what}") from exc


def credit_states(workflow: Any) -> dict[str, Any] | None:
    """The state of each estimator of ``workflow``'s entries that has state (``state()``), by entry, with its class
    (``{"class": "module:Class", "state": ...}``), as JSON data taken now (a copy, which the estimator's later settles
    leave alone); None when none has state.  The rollout source takes it where it takes the groups a checkpoint keeps,
    with no await between.  A :class:`CreditEstimatorError` for a ``state()`` that raises or is not JSON data."""

    states = {}
    for entry, estimator in _estimators(workflow).items():
        state = getattr(estimator, "state", None)
        if callable(state):
            try:
                states[entry] = {"class": class_ref(estimator), "state": json.loads(json.dumps(state()))}
            except Exception as exc:
                what = f"state() gave no JSON data ({type(exc).__name__}: {exc})"
                raise CreditEstimatorError(f"{_named(estimator, entry)}: {what}") from exc
    return states or None


def load_credit_states(workflow: Any, saved: Mapping[str, Any] | None) -> None:
    """Give each estimator of ``workflow``'s entries that has state the state a checkpoint saved for its entry
    (``saved``, the row's ``credit``).  An estimator whose entry it does not name starts afresh, and so does one of
    another class than the one that saved it (the entry's ``credit`` changed), with a warning.  A
    :class:`CreditEstimatorError` when ``load_state`` raises."""

    for entry, estimator in _estimators(workflow).items():
        load = getattr(estimator, "load_state", None)
        record = saved.get(entry) if isinstance(saved, Mapping) else None
        if not callable(load) or not isinstance(record, Mapping):
            continue
        if record.get("class") != class_ref(estimator):
            logger.warning(
                "%s starts afresh: the checkpoint holds the state of another credit estimator, %s",
                _named(estimator, entry),
                record.get("class"),
            )
            continue
        try:
            load(record["state"])
        except Exception as exc:
            what = f"load_state(state) raised {type(exc).__name__}: {exc}"
            raise CreditEstimatorError(f"{_named(estimator, entry)}: {what}") from exc


def _named(estimator: Any, entry: str | None) -> str:
    """The estimator as an error names it: as the config named it (its setup's ``name``), or its class, and its
    entry."""

    name = getattr(getattr(estimator, "setup", None), "name", None) or class_ref(estimator)
    return f"the credit estimator {name} of environments[{entry}]"


def _estimators(workflow: Any) -> Mapping[str, Any]:
    """The estimators of ``workflow``'s entries, by entry: those ``platoon train`` gave it (``entry_credit``, which a
    workflow that applies credit keeps)."""

    return getattr(workflow, "entry_credit", None) or {}
