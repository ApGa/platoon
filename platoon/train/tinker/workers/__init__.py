"""Moved: platoon 0.5.0 moved process-per-rollout to :mod:`platoon.train.workers`.  The names this package exported, and
its modules, still import from here, each with a ``PlatoonDeprecationWarning``, until platoon 0.6.0 removes them."""

from platoon.train.tinker._moved import moved as _moved

__getattr__, __dir__ = _moved(
    globals(),
    {
        "PROTOCOL_VERSION": "platoon.train.workers",
        "ConversionPolicy": "platoon.train.workers",
        "ModelBinding": "platoon.train.workers",
        "RolloutWorkerRequest": "platoon.train.workers",
        "RolloutWorkerResponse": "platoon.train.workers",
        "add_datum_funnel": "platoon.train.workers",
        "callable_ref": "platoon.train.workers",
        "resolve_ref": "platoon.train.workers",
        "validated_ref": "platoon.train.workers",
        "workload_from_raw_rollout": "platoon.train.workers",
    },
    submodules=("binding", "protocol", "registry", "run", "supervisor", "telemetry", "workload"),
)

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
