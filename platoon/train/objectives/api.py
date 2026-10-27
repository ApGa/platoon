"""What a policy's rows train under (:class:`Objective`).  Imports no Tinker.

A learner (:mod:`platoon.train.policy.learner`) sends each microbatch to ``forward_backward`` under the objective's
loss, and records the objective's metrics of each; the step that prepares the microbatch leaves out of the datums it
sends the loss inputs that only the trainer reads (``side_channels``), and keeps the originals for the metrics.  One of
those inputs is the action tokens a datum represents (``LOSS_NORMALIZATION_TOKENS_KEY``), which the batch transforms
(:mod:`platoon.train.step.batch_transforms`) set and read under the name defined here.  Whether a loss's objective is
zero for zero advantages (:func:`objective_is_zero_for_zero_advantages`) is read by a step's zero-advantage filter and
by the group admission rules' check of the run's loss.
"""

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import tinker

# The loss input in which a datum carries the action tokens it represents, Tinker's original loss denominator, once the
# zero-advantage filter removed zero-gradient datums from model compute; read by the trainer only.
LOSS_NORMALIZATION_TOKENS_KEY = "_loss_normalization_tokens"


def objective_is_zero_for_zero_advantages(loss_fn: str, loss_fn_config: dict) -> bool:
    """Whether zero advantages guarantee a zero objective for a known RL loss.

    Unknown options may add an advantage-independent KL/entropy/other term.
    Be conservative until their semantics are explicitly supported here.
    """

    return loss_fn in {"cispo", "ppo", "importance_sampling"} and not (
        set(loss_fn_config) - {"clip_low_threshold", "clip_high_threshold"}
    )


class Objective(Protocol):
    """What a policy's rows train under: one of the Tinker service's built-in losses (``loss_fn``, such as ``cispo``)
    with its keyword arguments, the loss inputs the datums carry for the trainer only, and the training metrics of
    each microbatch."""

    loss_fn: str
    loss_fn_config: dict[str, Any]
    # The loss inputs that only the trainer reads, which the datums sent to forward_backward leave out.
    side_channels: frozenset[str]

    def metrics(self, originals: list["tinker.Datum"], result: "tinker.ForwardBackwardOutput") -> dict[str, float]:
        """A microbatch's training metrics, from its datums as they trained (with their side channels) and the result
        of their forward_backward."""
        ...
