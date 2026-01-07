"""Configuration definitions for AReaL RL training."""

from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig
from areal.engine.ppo.actor import PPOActorConfig

from platoon.config_defs import RolloutConfig
from platoon.utils.train import VariableBatchInferenceEngineConfig
    

@dataclass
class WorkflowConfig:
    """Configuration for the rollout workflow."""
    group_size: int = 1
    rollout_config: RolloutConfig = field(default_factory=RolloutConfig)


@dataclass
class LossFnConfig:
    """Configuration for the loss function.
    
    This allows switching between different policy optimization loss functions
    (GRPO/PPO, CISPO) while maintaining consistent training infrastructure.
    
    Loss functions available:
    - "grpo" / "ppo": Standard PPO with clipped objective
    - "cispo": Clipped Importance Sampling Policy Optimization
    
    Example usage:
        # Use CISPO with custom clipping thresholds
        loss_fn_config = LossFnConfig(
            loss_fn="cispo",
            clip_low_threshold=0.0,
            clip_high_threshold=5.0,
        )
    """
    # Loss function selection (valid values: "grpo", "ppo", "cispo")
    # Note: Using str instead of Literal for OmegaConf compatibility
    loss_fn: str = "grpo"
    
    # CISPO-specific parameters
    # CISPO clips the importance sampling ratio directly, then uses detach(clip(ρ)) * A * log π
    # This ensures gradients always flow through log π, maintaining signal to all tokens
    clip_low_threshold: float = 0.0   # Lower bound for importance ratio (default: no lower bound)
    clip_high_threshold: float = 5.0  # Upper bound for importance ratio (default: 5)


@dataclass
class PlatoonArealRLTrainerConfig(GRPOConfig):
    """Main configuration for the AReaL RL trainer."""
    workflow_config: WorkflowConfig = field(default_factory=WorkflowConfig)
    rollout: VariableBatchInferenceEngineConfig = field(default_factory=VariableBatchInferenceEngineConfig)
    actor: PPOActorConfig = field(default_factory=PPOActorConfig)
    loss_fn_config: LossFnConfig = field(default_factory=LossFnConfig)

