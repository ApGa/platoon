"""Configuration definitions for AReaL RL training."""

from dataclasses import dataclass, field
from typing import Any

from areal.api.cli_args import GRPOConfig, PPOActorConfig

from platoon.config_defs import RolloutConfig
from platoon.utils.train import VariableBatchInferenceEngineConfig


@dataclass
class WorkflowConfig:
    """Configuration for the rollout workflow."""

    group_size: int = 1
    rollout_config: RolloutConfig = field(default_factory=RolloutConfig)
    use_subprocesses: bool = False  # Enable subprocess-based rollouts for isolation
    leave_one_out_baseline: bool = False  # Use leave-one-out baseline for advantage centering
    depth_level_weighting: bool = False  # Trainer-side full-batch inverse-frequency weighting by depth level
    depth_level_discount_gamma: float | None = None  # Trainer-side full-batch reward discounting by gamma^d
    filter_zero_variance_groups: bool = True  # Preserve old behavior by rejecting zero-variance groups

    def __post_init__(self) -> None:
        if isinstance(self.rollout_config, dict):
            self.rollout_config = RolloutConfig(**self.rollout_config)


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
            loss_fn_kwargs={
                "clip_low_threshold": 0.0,
                "clip_high_threshold": 5.0,
            },
        )
    """

    # Loss function selection (valid values: "grpo", "ppo", "cispo")
    # Note: Using str instead of Literal for OmegaConf compatibility
    loss_fn: str = "grpo"

    # Loss-specific kwargs. Registered loss defaults are applied first, and
    # values here override them.
    loss_fn_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlatoonPPOActorConfig(PPOActorConfig):
    """Actor config with Platoon-internal loss fields injected at runtime."""

    def __post_init__(self):
        super().__post_init__()
        self.loss_fn = "grpo"
        self.loss_fn_kwargs: dict[str, Any] = {}


@dataclass
class PlatoonGenerationConfig:
    """Minimal generation config still required by upstream AReaL internals."""

    lora_name: str = "default_lora"
    n_samples: int = 1

    def new(self, **kwargs):
        args = {"lora_name": self.lora_name, "n_samples": self.n_samples}
        args.update(kwargs)
        return type(self)(**args)


@dataclass
class PlatoonTrainDatasetConfig:
    """Minimal dataset config surface used by Platoon's AReaL path."""

    batch_size: int = 1
    shuffle: bool = True
    num_workers: int = 0
    drop_last: bool = True


@dataclass
class PlatoonValidDatasetConfig:
    """Validation dataloader config for Platoon's AReaL path."""

    batch_size: int = 1
    shuffle: bool = False
    num_workers: int = 0
    drop_last: bool = False


def _ensure_expandable_segments_env(specs) -> None:
    """Inject the CUDA allocator setting into worker launch envs.

    In single-controller AReaL, the trainer object is not the process doing the
    heavy GPU work. The actual actor/ref/critic/teacher and rollout engines run
    in scheduler-launched worker processes, so the allocator setting must be in
    their inherited environment before CUDA is initialized there.
    """

    if specs is None:
        return
    for spec in specs:
        current = spec.env_vars.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments:" in current:
            continue
        if current:
            spec.env_vars["PYTORCH_CUDA_ALLOC_CONF"] = f"{current},expandable_segments:True"
        else:
            spec.env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


@dataclass
class PlatoonArealRLTrainerConfig(GRPOConfig):
    """Main configuration for the AReaL RL trainer."""

    gconfig: PlatoonGenerationConfig = field(default_factory=PlatoonGenerationConfig)
    eval_gconfig: PlatoonGenerationConfig | None = None
    train_dataset: PlatoonTrainDatasetConfig = field(default_factory=PlatoonTrainDatasetConfig)
    valid_dataset: PlatoonValidDatasetConfig | None = field(default_factory=PlatoonValidDatasetConfig)
    workflow_config: WorkflowConfig = field(default_factory=WorkflowConfig)
    rollout: VariableBatchInferenceEngineConfig = field(default_factory=VariableBatchInferenceEngineConfig)
    actor: PlatoonPPOActorConfig = field(default_factory=PlatoonPPOActorConfig)
    ref: PlatoonPPOActorConfig | None = None
    loss_fn_config: LossFnConfig = field(default_factory=LossFnConfig)

    def __post_init__(self):
        if isinstance(self.gconfig, dict):
            self.gconfig = PlatoonGenerationConfig(**self.gconfig)
        if isinstance(self.eval_gconfig, dict):
            self.eval_gconfig = PlatoonGenerationConfig(**self.eval_gconfig)
        if isinstance(self.train_dataset, dict):
            self.train_dataset = PlatoonTrainDatasetConfig(**self.train_dataset)
        if isinstance(self.valid_dataset, dict):
            self.valid_dataset = PlatoonValidDatasetConfig(**self.valid_dataset)
        if isinstance(self.loss_fn_config, dict):
            self.loss_fn_config = LossFnConfig(**self.loss_fn_config)

        if self.scheduler.type is None:
            # Platoon's updated AReaL path relies on the single-controller scheduler.
            self.scheduler.type = "local"

        if self.eval_gconfig is None:
            self.eval_gconfig = self.gconfig.new()

        super().__post_init__()

        self.actor.loss_fn = self.loss_fn_config.loss_fn
        merged_loss_fn_kwargs = dict(getattr(self.actor, "loss_fn_kwargs", {}))
        merged_loss_fn_kwargs.update(self.loss_fn_config.loss_fn_kwargs)
        self.actor.loss_fn_kwargs = merged_loss_fn_kwargs

        if not self.rollout.backend:
            raise ValueError("rollout.backend must be set explicitly")
        if not self.actor.backend:
            raise ValueError("actor.backend must be set explicitly")
        if self.ref is not None and not self.ref.backend:
            self.ref.backend = self.actor.backend

        _ensure_expandable_segments_env(getattr(self.rollout, "scheduling_spec", None))
        _ensure_expandable_segments_env(getattr(self.actor, "scheduling_spec", None))
        if self.ref is not None:
            _ensure_expandable_segments_env(getattr(self.ref, "scheduling_spec", None))
        if self.critic is not None:
            _ensure_expandable_segments_env(getattr(self.critic, "scheduling_spec", None))
        if self.teacher is not None:
            _ensure_expandable_segments_env(getattr(self.teacher, "scheduling_spec", None))
