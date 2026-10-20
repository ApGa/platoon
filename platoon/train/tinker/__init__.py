# A rollout worker must fix its environment before this package pulls in the
# tinker/torch stack, and `python -m platoon.train.tinker.workers.entrypoint`
# runs this file first.  A no-op in the trainer; see the module docstring for
# what each invariant prevents.
from platoon.train.tinker.worker_env import apply_rollout_worker_env_invariants

apply_rollout_worker_env_invariants()

# ruff: isort: off  -- the invariants above must precede every import below.
from platoon.train.tinker.config_defs import (  # noqa: E402, I001
    AdamParams,
    CheckpointConfig,
    EvalConfig,
    PlatoonTinkerRLTrainerConfig,
    StatsConfig,
    TrainConfig,
    WatchdogConfig,
    WorkflowConfig,
)
from platoon.train.tinker.batch_transforms import (  # noqa: E402, I001
    BatchTransform,
    BatchTransformContext,
    DepthLevelWeightingTransform,
    build_default_batch_transforms,
)
from platoon.train.tinker.rl import PlatoonTinkerRLTrainer, Watchdog  # noqa: E402

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
