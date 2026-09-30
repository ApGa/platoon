from dataclasses import dataclass, field
from typing import Literal

from platoon.config_defs import RolloutConfig
from platoon.train.components import EnvironmentConfig, normalize_environment_configs
from platoon.utils.stats_logger import StatsLoggerConfig, WandBConfig
from platoon.utils.token_efficiency_config import TokenEfficiencyRewardConfig


@dataclass
class AdamParams:
    learning_rate: float = 3e-5
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    # Set to a large value (e.g., 1e12) to enable grad_norm logging without clipping.
    # When > 0, tinker returns grad_norm in OptimStepResponse.metrics.
    grad_clip_norm: float = 0.0


@dataclass
class WorkflowConfig:
    """Configuration for the rollout workflow."""

    group_size: int = 8
    token_efficiency_reward: TokenEfficiencyRewardConfig = field(default_factory=TokenEfficiencyRewardConfig)
    rollout_config: RolloutConfig = field(default_factory=RolloutConfig)
    leave_one_out_baseline: bool = False  # Use leave-one-out baseline for advantage centering
    depth_level_weighting: bool = False  # Weight trajectories inversely by depth-level frequency
    # Retain every root datum and independently sample each post-merge subagent
    # datum. A value of one preserves the historical training batch exactly.
    subagent_datum_keep_probability: float = 1.0
    subagent_datum_sampling_seed: int = 0
    # Defer typed action errors until group centering, then suppress only error
    # tokens that would otherwise receive positive policy credit.
    filter_errors: bool = True
    # Retain agent messages as context, with no policy loss for either sign.
    filter_message_actions: bool = False
    # Zero-advantage datums have no policy-gradient contribution but still incur
    # a model forward/backward. Baselines and rollout metrics are computed before
    # this filter is applied.
    filter_zero_advantage_datums: bool = True
    # Straggler policy (ported from the AReaL workflow; rollouts stay in-process).
    # Once ``straggler_quorum`` group members have settled (completed, failed,
    # or been interrupted), wait at most this many additional seconds for the
    # rest, then cancel them. None keeps every member until its own rollout
    # timeout.
    straggler_timeout_seconds: float | None = None
    # Number of settled members that starts the tail-grace clock. None means
    # ``group_size - 1`` (the classic single-straggler policy). Requires
    # ``straggler_timeout_seconds``.
    straggler_quorum: int | None = None
    # A group that returns fewer usable rollouts than this produces no training
    # datums (its rollout workload is still accounted for).
    min_successful_group_size: int = 1

    # --- rollout process isolation ---------------------------------------
    # "inprocess": rollouts are asyncio tasks in the trainer (historical).
    # "process":   one exec'd worker process per rollout, killable as a group.
    # A process is the only unit of execution Python can actually kill, and it
    # is the only way two rollouts can stop sharing the interpreter's globals
    # (freezegun's freeze stack, IPython's singleton shell, sys.stdout).
    rollout_isolation: Literal["inprocess", "process"] = "inprocess"
    # Import/registration budget for a worker.  Measured ~26 s of imports;
    # sized for a concurrent cold-start storm on a shared filesystem.
    worker_startup_budget_seconds: float = 300.0
    # Extra seconds beyond rollout_config.timeout for datum conversion and
    # writing the (1-6 MB) response file.
    worker_finalize_grace_seconds: float = 120.0
    # Parent wall deadline = the child's budget + this, so the child's own
    # alarm always owns its tree first and can leave a diagnostic marker.
    worker_parent_grace_seconds: float = 60.0
    # SIGTERM -> grace -> SIGKILL.  Must stay well under the launcher's 30 s
    # budget for the whole client.  0.0 means SIGKILL immediately.
    worker_termination_grace_seconds: float = 10.0
    # Groups -- not rollouts -- allowed to be starting up at once.  A group is
    # admitted atomically so its members never skew against the straggler
    # quorum clock, which has no notion of when each member began.
    worker_launch_concurrent_groups: int = 2
    # Per-worker RSS soft limit.  Above it the worker writes a diagnostic
    # marker and exits with a distinct code, so an over-allocating rollout is
    # classified as an OOM rather than inferred from an anonymous rc=-9.
    # Not RLIMIT_AS: the bare import stack is ~12.7 GB of address space against
    # ~0.7 GB resident, so any address-space limit loose enough to import torch
    # bounds nothing and a tight one kills every worker at import.
    worker_rss_soft_limit_bytes: int = 24 * 1024**3
    # Refuse to launch a worker below this MemAvailable; 0 disables.
    worker_mem_available_floor_bytes: int = 0
    # How often a worker re-checks the published sampling-client binding.
    worker_binding_poll_seconds: float = 5.0
    # Child budget used when rollout_config.timeout is null.
    worker_unbounded_rollout_budget_seconds: float = 24 * 3600.0
    # Wrap the rollout in isolated_rollout_python_environment().
    worker_python_env_isolation: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.token_efficiency_reward, dict):
            self.token_efficiency_reward = TokenEfficiencyRewardConfig(**self.token_efficiency_reward)
        if not 0.0 <= self.subagent_datum_keep_probability <= 1.0:
            raise ValueError("subagent_datum_keep_probability must be in [0, 1]")
        if isinstance(self.subagent_datum_sampling_seed, bool) or not isinstance(
            self.subagent_datum_sampling_seed, int
        ):
            raise ValueError("subagent_datum_sampling_seed must be an integer")
        if self.group_size < 1:
            raise ValueError("workflow group_size must be positive")
        if self.straggler_timeout_seconds is not None and self.straggler_timeout_seconds <= 0:
            raise ValueError("straggler_timeout_seconds must be positive or null")
        if self.straggler_quorum is not None and not 1 <= self.straggler_quorum <= self.group_size:
            raise ValueError("straggler_quorum must be in [1, group_size] or null")
        if self.straggler_timeout_seconds is None and self.straggler_quorum is not None:
            raise ValueError("straggler_quorum requires straggler_timeout_seconds")
        if not 1 <= self.min_successful_group_size <= self.group_size:
            raise ValueError("min_successful_group_size must be in [1, group_size]")
        if self.rollout_isolation not in ("inprocess", "process"):
            raise ValueError("rollout_isolation must be 'inprocess' or 'process'")
        for name in (
            "worker_startup_budget_seconds",
            "worker_finalize_grace_seconds",
            "worker_parent_grace_seconds",
            "worker_termination_grace_seconds",
            "worker_binding_poll_seconds",
            "worker_unbounded_rollout_budget_seconds",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.worker_launch_concurrent_groups < 1:
            raise ValueError("worker_launch_concurrent_groups must be >= 1")
        if self.worker_rss_soft_limit_bytes < 0:
            raise ValueError("worker_rss_soft_limit_bytes must be non-negative")
        if self.worker_mem_available_floor_bytes < 0:
            raise ValueError("worker_mem_available_floor_bytes must be non-negative")
        if self.worker_python_env_isolation and self.rollout_isolation != "process":
            # The overlay mutates sys.path, sys.executable and os.environ
            # process-globally and restores them per call, so with concurrent
            # in-process rollouts the first to finish would tear it down under
            # the others -- breaking their imports and every in-flight MCP
            # bridge launch, which uses sys.executable.
            raise ValueError(
                "worker_python_env_isolation requires rollout_isolation='process'"
            )


# TODO:
# - timeout
@dataclass
class TrainConfig:
    model_name: str  # HuggingFace model identifier
    renderer_name: str  # Renderer type for prompt formatting
    renderer_kwargs: dict = field(default_factory=dict)  # Optional renderer attribute overrides
    context_window_length: int | None = None
    batch_size: int = 32
    # Training duration: specify num_epochs, max_training_steps, or both (takes max)
    num_epochs: int | None = None
    max_training_steps: int | None = None
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
    loss_fn: str = "cispo"
    loss_fn_config: dict = field(default_factory=lambda: {"clip_low_threshold": 0.0, "clip_high_threshold": 5.0})
    lora_rank: int = 32
    workflow_config: WorkflowConfig = field(default_factory=WorkflowConfig)
    num_concurrent_rollout_workflow_workers: int | None = None
    # Optional whole-workflow capacities keyed by a dataset column. Admission
    # happens before worker creation/model binding; waiting rows remain bounded
    # in the input queue and do not occupy rollout workers.
    rollout_group_key: str = "_openreward_environment"
    rollout_group_limits: dict[str, int] = field(default_factory=dict)

    # Client-side deadline for a single ``forward_backward`` result.  The client submits
    # every microbatch up front and the Tinker service drains them in order, so the await on
    # microbatch 0 covers the WHOLE accumulated window, not one microbatch: size this against
    # a full optimizer step's training compute, not a single unit.  A 15-node Qwen3.6-35B-A3B
    # toolathlon step measured 275-728 s non-recursive; the recursive arm keeps every subagent
    # trajectory, needs ~2x the units, and died against the previous hardcoded 900 s
    # (run ta15-rec-q36-r3-v3, 2026-09-08: 14 units / 840 s all ok=true, killed mid-unit-15).
    forward_backward_timeout_seconds: float = 5400.0
    # Deadline for the optimizer step itself, which runs after the window is finalized and is
    # fast (0.6 s measured); this only needs to cover a pathological stall.
    optim_step_timeout_seconds: float = 300.0

    def __post_init__(self):
        if self.num_concurrent_rollout_workflow_workers is None:
            self.num_concurrent_rollout_workflow_workers = self.batch_size
        if not self.rollout_group_key or any(
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for key, value in self.rollout_group_limits.items()
        ):
            raise ValueError("rollout_group_limits requires nonempty labels and positive integer capacities")


@dataclass
class TrainEventTriggerConfig:
    strategy: Literal["epoch", "step", "none"] = "epoch"
    every: int = 1


@dataclass
class CheckpointConfig(TrainEventTriggerConfig):
    load_checkpoint_path: str | None = None
    # AReaL-style recovery cadence. ``strategy``/``every`` select the *permanent*
    # checkpoints (named ``<batch>``, kept forever). When ``recovery_every`` is
    # set, every ``recovery_every`` steps that is not already a permanent save
    # writes a full weights+optimizer checkpoint named ``recovery-<batch>``
    # instead of the weights-only update, so a crash loses at most that many
    # steps. Recovery checkpoints are disposable: the client-side pruner keeps
    # only the newest few of them. ``None`` disables the recovery cadence.
    recovery_every: int | None = None

    def __post_init__(self) -> None:
        if self.recovery_every is not None and self.recovery_every < 1:
            raise ValueError("checkpoint recovery_every must be >= 1 or null")


@dataclass
class EvalConfig(TrainEventTriggerConfig):
    num_concurrent_rollout_workflow_workers: int = 256
    workflow_config: WorkflowConfig = field(
        default_factory=lambda: WorkflowConfig(
            group_size=1,
            filter_errors=False,
            filter_zero_advantage_datums=False,
        )
    )


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
class WatchdogConfig:
    """Configuration for the watchdog that monitors for hangs."""

    enabled: bool = True
    timeout_seconds: float = 600  # 10 minutes default
    exit_code: int = 2  # Exit code when watchdog kills process


@dataclass
class PlatoonTinkerRLTrainerConfig:
    train: TrainConfig
    eval: EvalConfig
    log_path: str
    tinker_base_url: str | None = None  # Tinker service URL
    environments: list[EnvironmentConfig] = field(default_factory=lambda: [EnvironmentConfig()])
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)

    def __post_init__(self):
        self.environments = normalize_environment_configs(self.environments)
        if len(self.environments) > 1:
            raise NotImplementedError(
                "Multiple environments are not yet supported; provide exactly one entry"
            )
