"""Training for Platoon, against a Tinker-API service: hosted Tinker, or a self-hosted server that implements the Tinker
API, such as miles-tinker.

``platoon train`` starts training through :mod:`platoon.train.supervise` and :mod:`platoon.train.launch`, and one
start of training is the trainer (:mod:`platoon.train.trainer`).  Each stage of a run has a package of its own:

- ``trainer``: one start of training: the training client, the train and evaluation loops, stopping;
- ``step``: one training step: what it dequeues, the microbatch transforms, the update's metrics, what it logs;
- ``dispatch``: what starts the training groups (RFC 0019, section 4), and what they settle into;
- ``workflow``: one group of rollouts (``GroupRolloutWorkflow`` and its parts);
- ``credit``: the advantages a group's rollouts train with: its baselines and error-token filter, and the credit
  estimators an environment entry may choose instead (``environments[].credit``);
- ``workers``: process-per-rollout: the worker entrypoint, its supervisor and the protocol between them;
- ``checkpoints``: what a full checkpoint saves, and how a start resumes from it;
- ``evaluation``: evaluation during training.

Its modules, by what they are for:

- starting a run: ``launch`` (``platoon train``'s start), ``supervise`` (in its own process, or restarted after
  failures), ``startup`` (what a run readies before its first rollout) and ``reaping`` (what a run's dead processes
  left behind);
- setting up a run: ``config_defs`` (the config's schema), ``components``, ``entry_settings`` and ``auto`` (each
  environment's components and settings), ``labels`` (``train.label_by``), ``exclusion`` (``exclude_tasks``) and
  ``identity`` (one run per run directory);
- the training tasks: ``dataloader`` (the batches, epoch after epoch) and ``dataset_order`` (their order);
- task selection (RFC 0019): ``sampler_api``, ``samplers``, ``sampler_runtime`` and ``sampling`` (which task each group
  runs), ``admission_api`` and ``admissions`` (which groups count), ``task_stats`` and ``task_records`` (the per-task
  statistics), and ``leases`` (each rollout's permit, RFC 0020).

Words the training code uses throughout:

- a **start** is one launch of training in a run directory: its first, or a resume;
- a **group** is the ``group_size`` rollouts of one task, whose rewards set each other's baseline;
- a **draw** is the task the run's order gives a group.  It is numbered, so that a resume can draw it again, and it is
  outstanding until a committed step consumes it;
- a group **settles** when it ends: what a step dequeues of it goes to the queue the steps dequeue from;
- the **horizon** bounds how far rollouts run ahead of training: at most ``(train.max_staleness + 1) × batch_size``
  groups dispatched and not yet consumed;
- an **entry** is one environment of ``environments:``; a **label** is what a task's metrics and limits are keyed by
  (``train.label_by``; by default, its entry).

The names below import from here without loading tinker or torch until one is used.  ``platoon.train.tinker`` holds the
policy client (``proxy``), its renderers and the rollout timing, and the shims of the module paths that platoon 0.5.0
moved out of it, which platoon 0.6.0 removes.
"""

from typing import TYPE_CHECKING, Any

# Re-export common configurations for convenience
from platoon.config_defs import RolloutConfig

# Public name -> the module that defines it.
_EXPORTS = {
    "PlatoonTinkerRLTrainer": "platoon.train.trainer.rl",
    "Watchdog": "platoon.train.trainer.signals",
    "PlatoonTinkerRLTrainerConfig": "platoon.train.config_defs",
    "TrainConfig": "platoon.train.config_defs",
    "EvalConfig": "platoon.train.config_defs",
    "CheckpointConfig": "platoon.train.config_defs",
    "StatsConfig": "platoon.train.config_defs",
    "WatchdogConfig": "platoon.train.config_defs",
    "AdamParams": "platoon.train.config_defs",
    "WorkflowConfig": "platoon.train.config_defs",
    "BatchTransform": "platoon.train.step.batch_transforms",
    "BatchTransformContext": "platoon.train.step.batch_transforms",
    "DepthLevelWeightingTransform": "platoon.train.step.batch_transforms",
    "build_default_batch_transforms": "platoon.train.step.batch_transforms",
    "GroupRolloutWorkflow": "platoon.train.workflow.group",
    "current_entry": "platoon.train.workflow.context",
}

__all__ = [
    "AdamParams",
    "BatchTransform",
    "BatchTransformContext",
    "CheckpointConfig",
    "DepthLevelWeightingTransform",
    "EvalConfig",
    "GroupRolloutWorkflow",
    "PlatoonTinkerRLTrainer",
    "PlatoonTinkerRLTrainerConfig",
    "RolloutConfig",
    "StatsConfig",
    "TrainConfig",
    "Watchdog",
    "WatchdogConfig",
    "WorkflowConfig",
    "build_default_batch_transforms",
    "current_entry",
]


def __getattr__(name: str) -> Any:
    import importlib

    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


if TYPE_CHECKING:
    from platoon.train.config_defs import (
        AdamParams,
        CheckpointConfig,
        EvalConfig,
        PlatoonTinkerRLTrainerConfig,
        StatsConfig,
        TrainConfig,
        WatchdogConfig,
        WorkflowConfig,
    )
    from platoon.train.step.batch_transforms import (
        BatchTransform,
        BatchTransformContext,
        DepthLevelWeightingTransform,
        build_default_batch_transforms,
    )
    from platoon.train.trainer.rl import PlatoonTinkerRLTrainer
    from platoon.train.trainer.signals import Watchdog
    from platoon.train.workflow.context import current_entry
    from platoon.train.workflow.group import GroupRolloutWorkflow
