"""Token-efficiency settings shared by AReaL and Tinker."""

import math
from dataclasses import dataclass


@dataclass
class TokenEfficiencyRewardConfig:
    """Bounded token cost on a policy subtree's sequential critical path."""

    enabled: bool = False
    # penalty = min(max_penalty, coefficient * log2(1 + effective/reference))
    coefficient: float = 0.05
    reference_tokens: float = 20_000.0
    max_penalty: float = 0.20
    # Exported prompts resend the full logical context even when the
    # inference server can reuse a cached prefix, so discount them separately.
    input_token_weight: float = 0.01
    output_token_weight: float = 1.0
    attribution: str = "policy_subtree"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("token_efficiency_reward.enabled must be a boolean")
        values = {
            "coefficient": self.coefficient,
            "reference_tokens": self.reference_tokens,
            "max_penalty": self.max_penalty,
            "input_token_weight": self.input_token_weight,
            "output_token_weight": self.output_token_weight,
        }
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("token_efficiency_reward numeric settings must be finite")
        if self.coefficient < 0 or self.max_penalty < 0:
            raise ValueError("token_efficiency_reward coefficient and max_penalty must be non-negative")
        if self.reference_tokens <= 0:
            raise ValueError("token_efficiency_reward reference_tokens must be positive")
        if self.input_token_weight < 0 or self.output_token_weight < 0:
            raise ValueError("token_efficiency_reward token weights must be non-negative")
        if self.enabled and self.input_token_weight == 0 and self.output_token_weight == 0:
            raise ValueError("enabled token_efficiency_reward requires at least one positive token weight")
        if self.attribution != "policy_subtree":
            raise ValueError("token_efficiency_reward attribution must be 'policy_subtree'")
