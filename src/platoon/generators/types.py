from dataclasses import dataclass
from platoon.utils.llm_client import LLMClient

@dataclass # TODO: We should consider just passing these params as inputs to the generate_rollouts_batch method.
class RolloutGeneratorConfig:
    max_parallel_processes: int
    model_name: str
    max_steps_per_rollout: int
    output_dir: str
    verbose: bool = True
    per_rollout_timeout_seconds: int = 600
    model_endpoint: str | None = None
    llm_client: LLMClient | None = None