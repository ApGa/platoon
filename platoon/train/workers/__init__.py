"""Process-per-rollout support for the Tinker training path.

This package hosts the seam between the trainer and a single rollout:

``env``
    The invariants a worker sets before anything loads, which this package's
    init applies first.
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
    ``python -m platoon.train.workers.entrypoint <request>``.
``supervisor``
    The parent side: launch, deadline, kill-the-tree, classify.
``registry``
    The node-local record of the workers a trainer runs, so that a dead
    trainer's workers can be killed.
``telemetry``
    The programmatic-tool-calling cell guard's counters, added up across the
    worker processes.

Nothing in this package may import
``platoon.train.workflow``; the dependency runs
the other way.  The one exception is the start limit's module,
``platoon.train.workflow.start_limit`` (stdlib only, behind a package init
that imports nothing): the client monitor that ``protocol`` imports,
``platoon.train.tinker.rollout_metrics``, names its release causes.
"""

# A rollout worker must fix its environment before this package pulls in the
# tinker/torch stack, and `python -m platoon.train.workers.entrypoint` runs this
# file first.  A no-op in the trainer; see platoon.train.workers.env for what each
# invariant prevents.
from platoon.train.workers.env import apply_rollout_worker_env_invariants

apply_rollout_worker_env_invariants()

# ruff: isort: off  -- the invariants above must precede every import below.
from platoon.train.workers.protocol import (  # noqa: E402, I001
    PROTOCOL_VERSION,
    ConversionPolicy,
    ModelBinding,
    RolloutWorkerRequest,
    RolloutWorkerResponse,
    callable_ref,
    resolve_ref,
    validated_ref,
)
from platoon.train.workers.workload import add_datum_funnel, workload_from_raw_rollout  # noqa: E402, I001

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
