from dataclasses import dataclass, field

from platoon.config_defs import RolloutConfig


@dataclass
class InferenceWorkflowConfig:
    """Configuration for running inference rollouts over a dataset."""

    num_rollouts_per_task: int = 1
    num_concurrent_workers: int = 32
    success_threshold: float = 1.0
    fail_fast: bool = False
    use_subprocesses: bool = False
    subprocess_max_workers: int | None = None
    rollout_config: RolloutConfig = field(default_factory=RolloutConfig)


@dataclass
class InferenceBenchmarkConfig:
    """Top-level configuration for inference benchmarking."""

    model_name: str
    model_endpoint: str | None = None
    model_api_key: str | None = None
    output_dir: str = "inference_results"
    workflow: InferenceWorkflowConfig = field(default_factory=InferenceWorkflowConfig)
    resume: bool = True
