"""A step's microbatches, the unit of its first two stages: how a step splits its groups into minibatches of
microbatches (:class:`StepGeometry`), a microbatch as the step collects it (:class:`MicrobatchFill`, the first stage)
and as it prepares it to send (:class:`Prepared`, the second), and what the zero-advantage filter dropped of one
(:func:`zero_advantage_filter_metrics`).

The step itself, ``TrainStep`` in :mod:`platoon.train.trainer.loops`, fills each microbatch, runs the batch transforms
and the zero-advantage filter on it, and sends it.
"""

from dataclasses import dataclass, field
from typing import Any, NamedTuple

import tinker


@dataclass(frozen=True)
class StepGeometry:
    """How a step splits its ``batch_size`` groups: ``num_minibatches`` optimizer updates, each of
    ``num_microbatches`` microbatches of ``tasks_per_microbatch`` groups."""

    num_minibatches: int
    num_microbatches: int
    tasks_per_microbatch: int

    @classmethod
    def of(cls, config: Any, batch_size: int | None = None) -> "StepGeometry":
        """The split of a step under ``config``, of ``batch_size`` groups (default: ``train.batch_size``; a policy's own
        batch of shares when the policies step independently); an ``AssertionError`` when a count does not divide the
        one above it."""

        batch_size = config.train.batch_size if batch_size is None else batch_size
        assert batch_size % config.train.num_minibatches == 0, (
            f"batch_size {batch_size} must be divisible by num_minibatches {config.train.num_minibatches}"
        )
        tasks_per_minibatch = batch_size // config.train.num_minibatches

        assert tasks_per_minibatch % config.train.num_microbatches == 0, (
            f"tasks_per_minibatch {tasks_per_minibatch} must be divisible by "
            f"num_microbatches {config.train.num_microbatches}"
        )

        tasks_per_microbatch = tasks_per_minibatch // config.train.num_microbatches
        return cls(
            num_minibatches=config.train.num_minibatches,
            num_microbatches=config.train.num_microbatches,
            tasks_per_microbatch=tasks_per_microbatch,
        )


@dataclass
class MicrobatchFill:
    """A microbatch as a step fills it: the datums of its groups, as the step trains them
    (:func:`~platoon.train.step.datums._step_datums`), the leader's (``datums``: every datum, in a run of one trained
    policy) and each other trained policy's (``others``, by policy); and the places of the microbatch taken so far: each
    group the step dequeued, or, under a group admission rule, each admitted group, which counts toward the batch."""

    datums: list[tinker.Datum] = field(default_factory=list)
    places: int = 0
    others: dict[str, list[tinker.Datum]] = field(default_factory=dict)

    def empty(self) -> bool:
        """Whether it holds no datum of any policy."""

        return not self.datums and not any(self.others.values())


class Prepared(NamedTuple):
    """A microbatch ready to send: its datums as they train, with the loss inputs that the training metrics read
    (``originals``), and the copies of them that forward_backward takes (``datums``)."""

    originals: list[tinker.Datum]
    datums: list[tinker.Datum]


def zero_advantage_filter_metrics(eligible: tuple[int, int, int], retained: tuple[int, int, int]) -> dict[str, float]:
    """The ``zero_advantage_filter/*`` metrics of a microbatch, from the datums, attention tokens and loss tokens
    (:func:`~platoon.train.step.batch_transforms.get_datum_counts`) it had before the filter (``eligible``) and kept
    (``retained``)."""

    eligible_datums, eligible_attention_tokens, eligible_loss_tokens = eligible
    retained_datums, retained_attention_tokens, retained_loss_tokens = retained
    return {
        "zero_advantage_filter/eligible_datums": float(eligible_datums),
        "zero_advantage_filter/retained_datums": float(retained_datums),
        "zero_advantage_filter/filtered_datums": float(eligible_datums - retained_datums),
        "zero_advantage_filter/eligible_attention_tokens": float(eligible_attention_tokens),
        "zero_advantage_filter/retained_attention_tokens": float(retained_attention_tokens),
        "zero_advantage_filter/filtered_attention_tokens": float(eligible_attention_tokens - retained_attention_tokens),
        "zero_advantage_filter/eligible_loss_tokens": float(eligible_loss_tokens),
        "zero_advantage_filter/retained_loss_tokens": float(retained_loss_tokens),
        "zero_advantage_filter/filtered_loss_tokens": float(eligible_loss_tokens - retained_loss_tokens),
    }
