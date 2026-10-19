"""Process-per-rollout support for the Tinker training path.

This package hosts the seam between the trainer and a single rollout:

``protocol``
    The request/response/diagnostic contract that crosses the process
    boundary (pickle protocol 5 via node-local files).
``workload``
    The two raw-accounting helpers that must run wherever conversion runs.
``run``
    One implementation of a single rollout's body, used by both the
    in-process path and the worker process.
``binding``
    Publishing and following the trainer's ``SamplingClient``.
``entrypoint``
    ``python -m platoon.train.tinker.workers.entrypoint <request>``.
``supervisor``
    The parent side: launch, deadline, kill-the-tree, classify.

Nothing in this package may import
``platoon.train.tinker.workflows.group_rollout_workflow``; the dependency runs
the other way.
"""

from platoon.train.tinker.workers.protocol import (
    PROTOCOL_VERSION,
    ConversionPolicy,
    ModelBinding,
    RolloutWorkerRequest,
    RolloutWorkerResponse,
    callable_ref,
    resolve_ref,
    validated_ref,
)
from platoon.train.tinker.workers.workload import add_datum_funnel, workload_from_raw_rollout

__all__ = [
    "PROTOCOL_VERSION",
    "ConversionPolicy",
    "ModelBinding",
    "RolloutWorkerRequest",
    "RolloutWorkerResponse",
    "add_datum_funnel",
    "callable_ref",
    "resolve_ref",
    "validated_ref",
    "workload_from_raw_rollout",
]
