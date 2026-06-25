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
    openhands_system_prompt_suffix: str | None = None

    @classmethod
    def from_mapping(cls, value: OpenRewardConfig | dict[str, Any] | None) -> OpenRewardConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**cast(dict[str, Any], value))
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
