"""AReaL training backend for Platoon.

This module provides the AReaL-based RL trainer for distributed training.
"""

# Apply only controller-safe patches before importing AReaL-dependent modules.
# Megatron patches are installed lazily in actor.py immediately before a real
# Megatron engine is resolved inside its TE-capable worker environment.
from platoon.train.areal.patches import apply_controller_safe_patches

apply_controller_safe_patches()

from platoon.config_defs import RolloutConfig  # noqa: E402
from platoon.train.areal.batch_transforms import (  # noqa: E402
    BatchTransform,
    BatchTransformContext,
    DepthLevelWeightingTransform,
    build_default_batch_transforms,
)
from platoon.train.areal.config_defs import (  # noqa: E402
    LossFnConfig,
    PlatoonArealRLTrainerConfig,
    PlatoonPPOActorConfig,
    WorkflowConfig,
)
from platoon.train.areal.loss_functions import (  # noqa: E402
    LossFnSpec,
    build_loss_fn,
    cispo_loss_fn,
    get_loss_fn,
    get_loss_fn_defaults,
    grpo_loss_fn,
    list_loss_fns,
    ppo_loss_fn,
    register_loss_fn,
)
from platoon.train.areal.preallocated_pbs import PreallocatedPBSScheduler  # noqa: E402
from platoon.train.areal.preallocated_slurm import PreallocatedSlurmScheduler  # noqa: E402
from platoon.train.areal.proxy import ArealProxySession  # noqa: E402
from platoon.train.areal.rl import PlatoonArealRLTrainer  # noqa: E402
from platoon.train.areal.workflows import GroupRolloutWorkflow  # noqa: E402


def __getattr__(name: str):
    # Actor engine implementations are worker-only. Even AReaL's concrete
    # FSDP engine imports its tree-attention Megatron adapter, so keep all actor
    # exports lazy and let a worker resolve the requested class/function.
    if name in {"PlatoonPPOActor", "PlatoonMegatronPPOActor", "create_actor"}:
        from platoon.train.areal import actor

        return getattr(actor, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Config
    "LossFnConfig",
    "PlatoonArealRLTrainerConfig",
    "PlatoonPPOActorConfig",
    "RolloutConfig",
    "WorkflowConfig",
    "PreallocatedSlurmScheduler",
    "PreallocatedPBSScheduler",
    # Batch transforms
    "BatchTransform",
    "BatchTransformContext",
    "DepthLevelWeightingTransform",
    "build_default_batch_transforms",
    # Trainer
    "PlatoonArealRLTrainer",
    # Workflows
    "GroupRolloutWorkflow",
    # Actor
    "PlatoonPPOActor",
    "PlatoonMegatronPPOActor",
    "create_actor",
    # Proxy
    "ArealProxySession",
    # Loss functions
    "LossFnSpec",
    "build_loss_fn",
    "cispo_loss_fn",
    "get_loss_fn",
    "get_loss_fn_defaults",
    "grpo_loss_fn",
    "ppo_loss_fn",
    "register_loss_fn",
    "list_loss_fns",
]
