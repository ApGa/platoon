from dataclasses import dataclass, field
from typing import Literal

from platoon.config_defs import RolloutConfig
from platoon.utils.stats_logger import StatsLoggerConfig, WandBConfig


@dataclass
class AdamParams:
    learning_rate: float = 3e-5
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8


@dataclass
class WorkflowConfig:
    """Configuration for the rollout workflow."""
    group_size: int = 8
    rollout_config: RolloutConfig = field(default_factory=RolloutConfig)


# TODO:
# - timeout
@dataclass
class TrainConfig:
    model_name: str  # HuggingFace model identifier
    renderer_name: str  # Renderer type for prompt formatting
    batch_size: int = 32
    num_epochs: int = 1
    # Number of minibatches per batch. Batch must be divisible by num_minibatches.
    # Results in num_minibatches weight updates per batch.
    num_minibatches: int = 1
    # Number of microbatches per minibatch. Minibatch must be divisible by num_microbatches.
    # Used for gradient accumulation --> A single weight update per microbatch.
    # With tinker, gradient accumulation is less important, but this may still be useful for streaming
    # to overlap rollout sampling with forward backward computation within the same batch.
    num_microbatches: int = 1
    max_staleness: int | None = None  # Max staleness for off-policy rollouts
    optimizer: AdamParams = field(default_factory=AdamParams)
    loss_fn: str = 'cispo'
    loss_fn_config: dict = field(default_factory=lambda: {'clip_low_threshold': 0., 'clip_high_threshold': 5.})
    lora_rank: int = 32
    workflow_config: WorkflowConfig = field(default_factory=WorkflowConfig)
    num_concurrent_rollout_workflow_workers: int | None = None

    def __post_init__(self):
        if self.num_concurrent_rollout_workflow_workers is None:
            self.num_concurrent_rollout_workflow_workers = self.batch_size

@dataclass
class TrainEventTriggerConfig:
    strategy: Literal['epoch', 'step', 'none'] = "epoch"
    every: int = 1

@dataclass
class CheckpointConfig(TrainEventTriggerConfig):
    load_checkpoint_path: str | None = None

@dataclass
class EvalConfig(TrainEventTriggerConfig):
    num_concurrent_rollout_workflow_workers: int = 256
    workflow_config: WorkflowConfig = field(default_factory=lambda: WorkflowConfig(group_size=1))


@dataclass
class StatsConfig:
    """Configuration for stats tracking and logging."""
    experiment_name: str = "platoon_tinker"
    trial_name: str = "run"
    wandb: WandBConfig = field(default_factory=WandBConfig)
    
    def to_stats_logger_config(self, log_dir: str) -> StatsLoggerConfig:
        """Create a StatsLoggerConfig from this stats config."""
        return StatsLoggerConfig(
            experiment_name=self.experiment_name,
            trial_name=self.trial_name,
            log_dir=log_dir,
            wandb=self.wandb,
        )


@dataclass
class PlatoonTinkerRLTrainerConfig:
    train: TrainConfig
    eval: EvalConfig
    log_path: str
    tinker_base_url: str | None = None  # Tinker service URL
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
