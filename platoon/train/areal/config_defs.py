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
class PlatoonArealRLTrainerConfig(GRPOConfig):
    """Main configuration for the AReaL RL trainer."""
    workflow_config: WorkflowConfig = field(default_factory=WorkflowConfig)
    rollout: VariableBatchInferenceEngineConfig = field(default_factory=VariableBatchInferenceEngineConfig)
    actor: PPOActorConfig = field(default_factory=PPOActorConfig)

