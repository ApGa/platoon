"""The objective supervised rows train under (:class:`CrossEntropy`): the Tinker service's ``cross_entropy`` loss, the
sum over a datum's target tokens of each one's weight times its negative log-probability, and its metric, the negative
log-likelihood per trained token.

The loss is a sum, so a step that wants the mean over its trained tokens, as LLaMA-Factory's loss is, scales the
weights of its datums first (:func:`mean_per_token`): the step's gradient then does not grow with its batch, and a
learning rate means the same whatever the batch size.
"""

from typing import Any

import tinker

# A microbatch's metrics: its negative log-likelihood summed over its trained tokens, each times its weight, and
# their weight.  The learner averages each metric over a step's microbatches into its tracker, which names them in its
# scope (the run's train tracker: train/nll_sum); step_nll divides the two.
NLL_SUM = "nll_sum"
NLL_WEIGHT = "nll_weight"


class CrossEntropy:
    """Rows with ``target_tokens`` and ``weights`` train under ``cross_entropy``, which takes no keyword arguments; a
    microbatch's metrics are its weighted negative log-likelihood and its weight, from which a step's NLL per trained
    token comes (:func:`step_nll`)."""

    loss_fn = "cross_entropy"
    # The datums carry only what the loss reads.
    side_channels: frozenset[str] = frozenset()

    def __init__(self) -> None:
        self.loss_fn_config: dict[str, Any] = {}

    def metrics(self, originals: list[tinker.Datum], result: tinker.ForwardBackwardOutput) -> dict[str, float]:
        nll = weight = 0.0
        for datum, output in zip(originals, result.loss_fn_outputs, strict=True):
            weights = datum.loss_fn_inputs["weights"].to_torch()
            nll -= float(output["logprobs"].to_torch().dot(weights))
            weight += float(weights.sum())
        return {NLL_SUM: nll, NLL_WEIGHT: weight}


def step_nll(stats: dict[str, Any], *, scope: str = "train") -> float | None:
    """A step's negative log-likelihood per trained token, from the metrics a tracker of ``scope`` (the run's train
    tracker's) exported for it, out of which it takes the two it divides and their counts (None for a step that trained
    nothing).  The learner averages each metric over the step's microbatches, and two averages over the same
    microbatches divide as their sums do."""

    keys = [f"{scope}/{name}" if scope else name for name in (NLL_SUM, NLL_WEIGHT)]
    total, weight = (stats.pop(key, None) for key in keys)
    for key in keys:
        stats.pop(f"{key}__count", None)
    return total / weight if total is not None and weight else None


def trained_tokens(datum: tinker.Datum) -> float:
    """The weight a datum trains with: how many of its tokens train, when each weighs 1."""

    return float(datum.loss_fn_inputs["weights"].to_torch().sum())


def mean_per_token(datums: list[tinker.Datum]) -> list[tinker.Datum]:
    """``datums`` with their weights divided by the weight of them all, so that ``cross_entropy``'s sum over them is
    the mean NLL of their trained tokens.  Their weight must not be 0."""

    total = sum(trained_tokens(datum) for datum in datums)
    return [
        tinker.Datum(
            model_input=datum.model_input,
            loss_fn_inputs={
                **datum.loss_fn_inputs,
                "weights": tinker.TensorData.from_torch(datum.loss_fn_inputs["weights"].to_torch() / total),
            },
        )
        for datum in datums
    ]
