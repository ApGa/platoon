"""AReaL training backend for Platoon.

This module provides the AReaL-based RL trainer for distributed training.
"""

# Apply areal patches before importing areal-dependent modules
from platoon.train.areal.patches import apply_all_patches

apply_all_patches()

from platoon.train.areal.actor import (  # noqa: E402
    PlatoonPPOActor,
    create_actor,
)
from platoon.train.areal.batch_transforms import (  # noqa: E402
    BatchTransform,
    BatchTransformContext,
    DepthLevelWeightingTransform,
    build_default_batch_transforms,
)
from platoon.config_defs import RolloutConfig  # noqa: E402
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
from platoon.train.areal.proxy import ArealProxySession  # noqa: E402
from platoon.train.areal.rl import PlatoonArealRLTrainer  # noqa: E402
from platoon.train.areal.workflows import GroupRolloutWorkflow  # noqa: E402

__all__ = [
    # Config
    "LossFnConfig",
    "PlatoonArealRLTrainerConfig",
    "PlatoonPPOActorConfig",
    "RolloutConfig",
    "WorkflowConfig",
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
