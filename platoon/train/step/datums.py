"""A group's datums as a step trains them, by their loss inputs.

- The first stage of a step: the rows of a group's output that the step trains (:func:`trained_rows`), and the copies
  it collects, whose loss inputs its batch transforms set in place (:func:`_step_datums`).
- The second stage: the scale of a microbatch's advantages by the action tokens its datums represent, as the step
  prepares it to send (:func:`advantage_scale`, :func:`scale_advantages`).
"""

import logging
from collections.abc import Sequence
from typing import Any

import tinker
from tinker import TensorData

from platoon.train.step.batch_transforms import get_loss_normalization_token_count
from platoon.train.task_stats import MAIN
from platoon.train.workflow.protocol import rows_by_policy

logger = logging.getLogger(__name__)


def trained_rows(output: Any) -> Sequence[tinker.Datum]:
    """The rows of a group's output that the step trains: the run's one trained policy's (``main``,
    :func:`~platoon.train.workflow.protocol.rows_by_policy`), the output itself for a list of datums; none when it
    gives none (an empty group).  The rows of any other policy are left out with a warning: this trainer trains one
    policy."""

    by_policy = rows_by_policy(output, MAIN)
    others = [policy for policy in by_policy if policy != MAIN]
    if others:
        logger.warning(
            "A group's output gives rows of %s, which this run does not train (it trains one policy, %s): they are "
            "left out",
            ", ".join(map(str, others)),
            MAIN,
        )
    return by_policy.get(MAIN, ())


def _step_datums(rollout: Sequence[tinker.Datum]) -> list[tinker.Datum]:
    """A group's datums as a step trains them: each its own datum, with its own loss inputs, so that the step's batch
    transforms, which set loss inputs in place, leave the group as it settled; a checkpoint that keeps a group a dropped
    step had dequeued then keeps it as it settled, and the step after the resume transforms it once."""

    return [tinker.Datum(model_input=datum.model_input, loss_fn_inputs=dict(datum.loss_fn_inputs)) for datum in rollout]


def advantage_scale(datums: list[tinker.Datum], weight: float | None = None) -> float:
    """What a microbatch's advantages are multiplied by: one over the action tokens its datums represent (their
    normalization tokens, else their mask), times ``weight``, a short last microbatch's; so that Tinker's sum-reduced
    objective behaves like a mean reduction and grad_norm is not sensitive to batch size."""

    normalization_token_count = get_loss_normalization_token_count(datums)
    if weight is not None:
        normalization_token_count *= weight
    return 1.0 / (normalization_token_count + 1e-8)


def scale_advantages(datums: list[tinker.Datum], scale_factor: float) -> None:
    """Multiply the advantages of ``datums``, the copies a microbatch sends, by ``scale_factor``, in place."""

    for datum in datums:
        datum.loss_fn_inputs["advantages"] = TensorData.from_torch(
            datum.loss_fn_inputs["advantages"].to_torch() * scale_factor
        )
