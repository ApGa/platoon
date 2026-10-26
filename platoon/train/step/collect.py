"""What a step reads of each group it dequeues beside its datums: the workload its output carries, which a custom
workflow need not give.
"""

import logging
from dataclasses import dataclass
from typing import Any

from platoon.utils.rollout_workload import RolloutWorkload

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TaskWorkloadPayload:
    """Validated workload sidechannel attached to a list-compatible result."""

    workload: RolloutWorkload
    requested_rollouts: int
    observed_rollouts: int
    trainable_rollouts: int
    task_retained_datums: int


def _get_task_workload_payload(rollout: Any) -> _TaskWorkloadPayload | None:
    """Read optional workload metadata without constraining custom workflows."""

    workload = getattr(rollout, "workload", None)
    if not isinstance(workload, RolloutWorkload):
        return None
    count_names = (
        "requested_rollouts",
        "observed_rollouts",
        "trainable_rollouts",
        "task_retained_datums",
    )
    counts = [getattr(rollout, name, None) for name in count_names]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
        logger.warning("Ignoring malformed task workload rollout counts: %s", dict(zip(count_names, counts)))
        return None
    return _TaskWorkloadPayload(workload, *counts)
