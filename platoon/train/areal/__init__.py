"""AReaL training backend for Platoon.

This module provides the AReaL-based RL trainer for distributed training.
"""

# Apply areal patches before importing areal-dependent modules
from platoon.train.areal.patches import apply_all_patches
apply_all_patches()

from platoon.train.areal.config_defs import (
    LossFnConfig,
    PlatoonArealRLTrainerConfig,
    RolloutConfig,
    WorkflowConfig,
)
from platoon.train.areal.loss_functions import (
    cispo_loss_fn,
    grpo_loss_fn,
    sapo_loss_fn_wrapper,
    get_loss_fn,
    register_loss_fn,
    list_loss_fns,
)
from platoon.train.areal.actor import (
    PlatoonPPOActor,
    create_actor,
)
from platoon.train.areal.proxy import ArealProxySession
from platoon.train.areal.rl import PlatoonArealRLTrainer

__all__ = [
    # Config
    "LossFnConfig",
    "PlatoonArealRLTrainerConfig",
    "RolloutConfig",
    "WorkflowConfig",
    # Trainer
    "PlatoonArealRLTrainer",
    # Actor
    "PlatoonPPOActor",
    "create_actor",
    # Proxy
    "ArealProxySession",
    # Loss functions
    "cispo_loss_fn",
    "grpo_loss_fn", 
    "sapo_loss_fn_wrapper",
    "get_loss_fn",
    "register_loss_fn",
    "list_loss_fns",
]

