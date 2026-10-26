"""A group's datums as a step trains them: copies whose loss inputs the step's batch transforms set in place."""

from collections.abc import Sequence

import tinker


def _step_datums(rollout: Sequence[tinker.Datum]) -> list[tinker.Datum]:
    """A group's datums as a step trains them: each its own datum, with its own loss inputs, so that the step's batch
    transforms, which set loss inputs in place, leave the group as it settled; a checkpoint that keeps a group a dropped
    step had dequeued then keeps it as it settled, and the step after the resume transforms it once."""

    return [tinker.Datum(model_input=datum.model_input, loss_fn_inputs=dict(datum.loss_fn_inputs)) for datum in rollout]
