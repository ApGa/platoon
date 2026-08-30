"""Shared configuration definitions for Platoon.

This module contains configuration classes that are used across different
parts of the codebase (rollouts, training, inference, etc.).
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InferenceParams:
    """Inference parameters for rollout-time model calls.

    Defaults preserve current behavior to keep existing configs backwards compatible.
    """

    temperature: float | None = 1.0
    top_p: float | None = 1.0
    max_completion_tokens: int = 512


@dataclass
class RolloutConfig:
    """Configuration for rollout execution.

    This configuration is used for running agent rollouts, whether for
    training, evaluation, or standalone inference.
    """

    model_name: str | None = None
    model_endpoint: str | None = None
    model_api_key: str | None = None
    train: bool = False
    max_steps: int | None = None
    output_dir: str = "rollout_results"
    verbose: bool = True
    timeout: int | None = None  # Trajectory timeout (entire rollout)
    step_timeout: int = 300  # Per-step timeout (agent.act + env.step)
    return_dict: bool = False
    # ``None`` is used only while OmegaConf composes the canonical and legacy
    # keys. ``__post_init__`` always resolves this to a concrete boolean.
    propagate_root_success: bool | None = None
    # Deprecated compatibility key for configs created before the spelling
    # correction. New configs and code must use ``propagate_root_success``.
    propogate_root_success: bool | None = None
    skip_subagent_reward_computation: bool = False
    inference_params: InferenceParams = field(default_factory=InferenceParams)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        canonical = self.propagate_root_success
        legacy = self.propogate_root_success
        if canonical is not None and legacy is not None and canonical != legacy:
            raise ValueError(
                "Conflicting rollout propagation settings: "
                "propagate_root_success and deprecated propogate_root_success"
            )
        self.propagate_root_success = bool(
            legacy if canonical is None and legacy is not None else canonical
        )
        # Support loading from plain dicts from config loaders and subprocess paths.
        if isinstance(self.inference_params, dict):
            self.inference_params = InferenceParams(**self.inference_params)
