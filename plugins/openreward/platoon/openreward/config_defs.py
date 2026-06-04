from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

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

    @classmethod
    def from_mapping(cls, value: "OpenRewardConfig | dict[str, Any] | None") -> "OpenRewardConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        return cls(**value)


@dataclass
class OpenRewardInferenceConfig:
    inference: InferenceBenchmarkConfig
    openreward: OpenRewardConfig = field(default_factory=OpenRewardConfig)
    task_id: str | None = None
    stage: str = "full"
    shuffle_tasks: bool = False
    seed: int = 42

    def __post_init__(self):
        if isinstance(self.openreward, dict):
            self.openreward = OpenRewardConfig(**self.openreward)
