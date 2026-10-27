"""The policy client, its renderers and the rollout timing; and the shims of the module paths that platoon 0.5.0
moved out of this package.

Before platoon 0.5.0 this package held the trainer, its configuration and the rollout workflows; they moved into the
stage packages of :mod:`platoon.train` (``trainer``, ``step``, ``dispatch``, ``workflow``, ``credit``, ``workers``,
``checkpoints``, ``evaluation``).  The names this package exported still import from here, each with a
``PlatoonDeprecationWarning`` that names its new module, and so do the old modules, until platoon 0.6.0 removes them;
importing an old module imports the trainer, as importing this package did.  ``proxy``, ``key_refusal``,
``rollout_metrics`` and ``renderers`` live here, and the new code imports them, so this init does not."""

# A rollout worker must fix its environment before anything imports the tinker/torch stack, and a worker started as
# `python -m platoon.train.tinker.workers.entrypoint` (the entrypoint's module before platoon 0.5.0) runs this file
# first.  A no-op in the trainer; see platoon.train.workers.env.
from platoon.train.workers.env import apply_rollout_worker_env_invariants

apply_rollout_worker_env_invariants()

from platoon.train.tinker._moved import moved as _moved  # noqa: E402

__getattr__, __dir__ = _moved(
    globals(),
    {
        "PlatoonTinkerRLTrainer": "platoon.train",
        "Watchdog": "platoon.train",
        "BatchTransform": "platoon.train",
        "BatchTransformContext": "platoon.train",
        "DepthLevelWeightingTransform": "platoon.train",
        "build_default_batch_transforms": "platoon.train",
        "PlatoonTinkerRLTrainerConfig": "platoon.train",
        "TrainConfig": "platoon.train",
        "EvalConfig": "platoon.train",
        "CheckpointConfig": "platoon.train",
        "StatsConfig": "platoon.train",
        "WatchdogConfig": "platoon.train",
        "AdamParams": "platoon.train",
        "WorkflowConfig": "platoon.train",
    },
    submodules=(
        "batch_transforms",
        "censoring",
        "checkpoint_pruning",
        "config_defs",
        "dataset_order",
        "dispatcher",
        "evaluation_record",
        "kept_groups",
        "key_refusal",
        "permits",
        "proxy",
        "renderers",
        "rl",
        "rollout_metrics",
        "staleness",
        "start_limit",
        "task_records",
        "worker_env",
        "workers",
        "workflows",
    ),
)

__all__ = [
    "PlatoonTinkerRLTrainer",
    "Watchdog",
    "BatchTransform",
    "BatchTransformContext",
    "DepthLevelWeightingTransform",
    "build_default_batch_transforms",
    "PlatoonTinkerRLTrainerConfig",
    "TrainConfig",
    "EvalConfig",
    "CheckpointConfig",
    "StatsConfig",
    "WatchdogConfig",
    "AdamParams",
    "WorkflowConfig",
]
