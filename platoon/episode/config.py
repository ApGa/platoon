from dataclasses import dataclass

# TODO: We should make this customizable with a factory.
@dataclass
class RolloutConfig:
    model_name: str | None = None
    model_endpoint: str | None = None
    model_api_key: str | None = None
    train: bool = False
    max_steps: int | None = None
    output_dir: str = 'rollout_results'
    verbose: bool = True
    timeout: int | None = None
    return_dict: bool = False