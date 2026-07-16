"""Configuration definitions for AReaL RL training."""

from dataclasses import dataclass, field
from typing import Any

from areal.api.cli_args import GRPOConfig, PPOActorConfig

from platoon.config_defs import RolloutConfig
from platoon.train.components import EnvironmentConfig, normalize_environment_configs
from platoon.utils.train import VariableBatchInferenceEngineConfig


@dataclass
class WorkflowConfig:
    """Configuration for the rollout workflow."""

    group_size: int = 1
    rollout_config: RolloutConfig = field(default_factory=RolloutConfig)
    use_subprocesses: bool = False  # Enable subprocess-based rollouts for isolation
    # Once every member but a tail straggler has finished, wait at most this
    # many additional seconds before reaping the dedicated group process pool.
    # None preserves the full absolute rollout timeout for every member.
    straggler_timeout_seconds: float | None = None
    # Number of completed root rollouts that starts the tail-grace clock.
    # Interrupted partial roots do not count. None uses group_size - 1 (the
    # classic single-straggler policy).
    straggler_quorum: int | None = None
    subprocess_shutdown_grace_seconds: float = 5.0
    # Reject/replenish groups that return too few valid members for a meaningful
    # within-task baseline.  Recursive runs use 4 for an intended group size 8.
    min_successful_group_size: int = 1
    leave_one_out_baseline: bool = False  # Use leave-one-out baseline for advantage centering
    depth_level_weighting: bool = False  # Trainer-side full-batch inverse-frequency weighting by depth level
    depth_level_discount_gamma: float | None = None  # Trainer-side full-batch reward discounting by gamma^d
    # Retain every root datum and independently sample each post-merge subagent
    # datum.  A value of one preserves the historical training batch exactly.
    subagent_datum_keep_probability: float = 1.0
    subagent_datum_sampling_seed: int = 0
    # Reward-only throughput fast path: identify exact-zero centered scalar
    # rewards after group centering and policy/Bernoulli masks, retain them
    # through global DP selection and multiplicative depth normalization, then
    # omit all but the minimum dispatch padding before model-side computation.
    # IMPORTANT: disable this when KL != 0, reward_bias != 0, reward/advantage
    # normalization is active, overlong_reward_penalty is enabled, a critic or
    # teacher/distillation objective or independent MoE/router auxiliary loss
    # is present, or a custom transform adds to rewards. In those modes zero
    # scalar reward need not imply zero final policy advantage (or zero total
    # objective). A trainer startup warning
    # repeats these constraints because the remote workflow cannot validate the
    # complete actor/objective configuration by itself.
    filter_zero_advantage_datums: bool = True
    filter_zero_variance_groups: bool = True  # Preserve old behavior by rejecting zero-variance groups
    # Router replay (R3) is opt-in. The actor config is the public source of
    # truth; PlatoonArealRLTrainerConfig copies its model-derived dimensions
    # here so remote rollout workers can reshape SGLang's flattened routes.
    enable_router_replay: bool = False
    router_replay_num_layers: int | None = None
    router_replay_topk: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.rollout_config, dict):
            self.rollout_config = RolloutConfig(**self.rollout_config)
        if self.group_size < 1:
            raise ValueError("workflow group_size must be positive")
        if self.straggler_timeout_seconds is not None and self.straggler_timeout_seconds <= 0:
            raise ValueError("straggler_timeout_seconds must be positive or null")
        if self.straggler_quorum is not None and not 1 <= self.straggler_quorum <= self.group_size:
            raise ValueError("straggler_quorum must be in [1, group_size] or null")
        if self.straggler_timeout_seconds is None and self.straggler_quorum is not None:
            raise ValueError("straggler_quorum requires straggler_timeout_seconds")
        if self.subprocess_shutdown_grace_seconds < 0:
            raise ValueError("subprocess_shutdown_grace_seconds must be non-negative")
        if not 1 <= self.min_successful_group_size <= self.group_size:
            raise ValueError("min_successful_group_size must be in [1, group_size]")
        if not 0.0 <= self.subagent_datum_keep_probability <= 1.0:
            raise ValueError("subagent_datum_keep_probability must be in [0, 1]")
        if isinstance(self.subagent_datum_sampling_seed, bool) or not isinstance(
            self.subagent_datum_sampling_seed, int
        ):
            raise ValueError("subagent_datum_sampling_seed must be an integer")


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

    loss_fn: str = "grpo"
    loss_fn_kwargs: dict[str, Any] = field(default_factory=dict)
    enable_router_replay: bool = False
    router_replay_num_layers: int | None = None
    router_replay_topk: int | None = None
    router_replay_num_experts: int | None = None

    def __post_init__(self):
        super().__post_init__()
        # These are runtime-only defaults for the custom actor implementation.
        # User configs should choose losses via top-level `loss_fn_config`;
        # PlatoonArealRLTrainerConfig copies those values onto `actor` below.
        if not getattr(self, "loss_fn", None):
            self.loss_fn = "grpo"
        if getattr(self, "loss_fn_kwargs", None) is None:
            self.loss_fn_kwargs = {}
        if self.enable_router_replay:
            if not isinstance(self.router_replay_num_layers, int) or self.router_replay_num_layers <= 0:
                raise ValueError(
                    "actor.router_replay_num_layers must be a positive integer when router replay is enabled"
                )
            if not isinstance(self.router_replay_topk, int) or self.router_replay_topk <= 0:
                raise ValueError("actor.router_replay_topk must be a positive integer when router replay is enabled")
            if self.router_replay_num_experts is not None and self.router_replay_num_experts <= 0:
                raise ValueError("actor.router_replay_num_experts must be positive or null")


@dataclass
class PlatoonGenerationConfig:
    """Minimal generation config still required by upstream AReaL internals."""

    lora_name: str = "default_lora"

    def new(self, **kwargs):
        args = {"lora_name": self.lora_name}
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
    environments: list[EnvironmentConfig] = field(default_factory=lambda: [EnvironmentConfig()])

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
        self.environments = normalize_environment_configs(self.environments)
        if len(self.environments) > 1:
            raise NotImplementedError("Multiple environments are not yet supported; provide exactly one entry")

        if self.scheduler.type is None:
            # Platoon's updated AReaL path relies on the single-controller scheduler.
            self.scheduler.type = "local"

        if self.eval_gconfig is None:
            self.eval_gconfig = self.gconfig.new()

        super().__post_init__()

        # Keep loss selection in one public config location (`loss_fn_config`)
        # while attaching it to the actor object consumed by PlatoonActorImpl.
        self.actor.loss_fn = self.loss_fn_config.loss_fn
        merged_loss_fn_kwargs = dict(getattr(self.actor, "loss_fn_kwargs", {}))
        merged_loss_fn_kwargs.update(self.loss_fn_config.loss_fn_kwargs)
        self.actor.loss_fn_kwargs = merged_loss_fn_kwargs

        # Keep one public R3 gate on the actor while giving remote workflows
        # the dimensions required to reshape SGLang's flattened routing data.
        self.workflow_config.enable_router_replay = self.actor.enable_router_replay
        self.workflow_config.router_replay_num_layers = self.actor.router_replay_num_layers
        self.workflow_config.router_replay_topk = self.actor.router_replay_topk

        if not self.rollout.backend:
            raise ValueError("rollout.backend must be set explicitly")
        if not self.actor.backend:
            raise ValueError("actor.backend must be set explicitly")
        if self.ref is not None and not self.ref.backend:
            self.ref.backend = self.actor.backend
        if self.actor.enable_router_replay:
            actor_backend = self.actor.backend.split(":", 1)[0]
            rollout_backend = self.rollout.backend.split(":", 1)[0]
            if actor_backend != "megatron":
                raise ValueError("actor.enable_router_replay requires the Megatron actor backend")
            if rollout_backend != "sglang":
                raise ValueError("actor.enable_router_replay requires the SGLang rollout backend")
            if not bool(getattr(self.rollout, "return_routed_experts", False)):
                raise ValueError("actor.enable_router_replay requires rollout.return_routed_experts=true")
            if bool(getattr(self.actor.megatron, "enable_mtp", False)):
                raise ValueError(
                    "actor.enable_router_replay requires actor.megatron.enable_mtp=false; "
                    "rollout routes do not include MTP layers"
                )
            if self.actor.should_compute_prox_logp():
                raise ValueError(
                    "actor.enable_router_replay currently requires proximal log-probability "
                    "recomputation to be disabled; forward-only replay is not implemented"
                )
            if self.actor.gradient_checkpointing and (
                self.actor.megatron.recompute_granularity != "full" or self.actor.megatron.recompute_method != "uniform"
            ):
                raise ValueError(
                    "actor.enable_router_replay with gradient checkpointing requires "
                    "actor.megatron.recompute_granularity=full and recompute_method=uniform"
                )

        _ensure_expandable_segments_env(getattr(self.rollout, "scheduling_spec", None))
        _ensure_expandable_segments_env(getattr(self.actor, "scheduling_spec", None))
        if self.ref is not None:
            _ensure_expandable_segments_env(getattr(self.ref, "scheduling_spec", None))
        if self.critic is not None:
            _ensure_expandable_segments_env(getattr(self.critic, "scheduling_spec", None))
        if self.teacher is not None:
            _ensure_expandable_segments_env(getattr(self.teacher, "scheduling_spec", None))
