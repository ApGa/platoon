from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, cast

from platoon.inference import InferenceBenchmarkConfig


@dataclass
class OpenRewardConfig:
    env_name: str = "toolathlongym"
    split: str = "train"
    eval_split: str | None = None
    session_url: str = field(default_factory=lambda: os.getenv("OPENREWARD_SESSION_URL", "http://localhost:8080"))
    api_url: str | None = field(default_factory=lambda: os.getenv("OPENREWARD_API_URL"))
    api_key: str = field(default_factory=lambda: os.getenv("OPENREWARD_API_KEY", "local"))
    train_task_limit: int | None = None
    eval_task_limit: int | None = 50
    task_names: list[str] | None = None
    max_tool_calls: int = 0
    enable_programmatic_tool_calling: bool = False
    enable_recursive_subagents: bool = False
    subagent_default_max_steps: int = 50
    subagent_max_depth: int | None = None
    enable_subagent_reward_judging: bool = False
    subagent_reward_judge_max_steps: int = 20
    subagent_delegation_reward_coefficient: float = 0.0
    openhands_system_prompt_suffix: str | None = None

    def __post_init__(self) -> None:
        if self.subagent_delegation_reward_coefficient < 0:
            raise ValueError("subagent_delegation_reward_coefficient must be non-negative")

    @classmethod
    def from_mapping(cls, value: OpenRewardConfig | dict[str, Any] | None) -> OpenRewardConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            data = dict(cast(dict[str, Any], value))
            legacy_enabled = data.pop("enable_subagent_judging", None)
            if legacy_enabled is not None and "enable_subagent_reward_judging" not in data:
                data["enable_subagent_reward_judging"] = legacy_enabled
            legacy_max_steps = data.pop("subagent_judge_max_steps", None)
            if legacy_max_steps is not None and "subagent_reward_judge_max_steps" not in data:
                data["subagent_reward_judge_max_steps"] = legacy_max_steps
            return cls(**data)
        raise TypeError(f"Unsupported OpenRewardConfig value: {type(value).__name__}")


@dataclass
class OpenRewardInferenceConfig:
    inference: InferenceBenchmarkConfig
    openreward: OpenRewardConfig = field(default_factory=OpenRewardConfig)
    task_id: str | None = None
    stage: str = "full"
    shuffle_tasks: bool = False
    seed: int = 42

    def __post_init__(self):
        openreward = self.openreward
        if isinstance(openreward, dict):
            self.openreward = OpenRewardConfig(**cast(dict[str, Any], openreward))
