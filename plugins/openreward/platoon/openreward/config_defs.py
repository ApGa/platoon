from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, cast

from platoon.inference import InferenceBenchmarkConfig

SUBAGENT_ENVIRONMENT_ACCESS_MODES = frozenset({"shared", "read_only"})


@dataclass
class OpenRewardEnvironmentConfig:
    """Connection, task-selection, and sampling settings for one environment."""

    env_name: str = "toolathlongym"
    label: str | None = None
    split: str = "train"
    eval_split: str | None = None
    session_url: str = field(default_factory=lambda: os.getenv("OPENREWARD_SESSION_URL", "http://localhost:8080"))
    # Static URL pools are convenient outside Slurm. A launcher can instead
    # export OPENREWARD_SESSION_URLS_<LABEL>, where LABEL is upper-cased and
    # non-alphanumeric characters are replaced by underscores.
    session_urls: list[str] | None = None
    session_urls_env_var: str | None = None
    api_url: str | None = field(default_factory=lambda: os.getenv("OPENREWARD_API_URL"))
    api_key: str = field(default_factory=lambda: os.getenv("OPENREWARD_API_KEY", "local"))
    train_task_limit: int | None = None
    eval_task_limit: int | None = 50
    task_names: list[str] | None = None
    task_indices: list[int] | None = None
    max_tool_calls: int = 0
    sampling_weight: float = 1.0
    # Override the rollout-wide child access policy for this environment.
    subagent_environment_access: str | None = None

    def __post_init__(self) -> None:
        if not self.env_name.strip():
            raise ValueError("OpenReward environment env_name must not be empty")
        if self.label is not None and not self.label.strip():
            raise ValueError("OpenReward environment label must not be empty")
        if not math.isfinite(self.sampling_weight) or self.sampling_weight <= 0:
            raise ValueError("OpenReward environment sampling_weight must be finite and positive")
        if (
            self.subagent_environment_access is not None
            and self.subagent_environment_access not in SUBAGENT_ENVIRONMENT_ACCESS_MODES
        ):
            raise ValueError("OpenReward environment subagent_environment_access must be 'shared' or 'read_only'")
        for field_name, limit in (
            ("train_task_limit", self.train_task_limit),
            ("eval_task_limit", self.eval_task_limit),
        ):
            if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
                raise ValueError(f"OpenReward environment {field_name} must be a positive integer when set")
        if self.max_tool_calls < 0:
            raise ValueError("OpenReward environment max_tool_calls must be non-negative")
        if self.task_names is not None and self.task_indices is not None:
            raise ValueError("Configure task_names or task_indices for an environment, not both")
        if self.task_indices is not None:
            invalid = any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in self.task_indices
            )
            if invalid:
                raise ValueError("OpenReward environment task_indices must contain non-negative integers")
            if len(set(self.task_indices)) != len(self.task_indices):
                raise ValueError("OpenReward environment task_indices must not contain duplicates")
        if self.session_urls is not None:
            self.session_urls = [url.strip() for url in self.session_urls if url.strip()]
            if not self.session_urls:
                raise ValueError("OpenReward environment session_urls must contain at least one URL")
        if self.session_urls_env_var is not None:
            self.session_urls_env_var = self.session_urls_env_var.strip()
            if not self.session_urls_env_var:
                raise ValueError("OpenReward environment session_urls_env_var must not be empty")

    @property
    def resolved_label(self) -> str:
        return (self.label or self.env_name).strip()

    @property
    def resolved_session_urls_env_var(self) -> str:
        if self.session_urls_env_var is not None:
            return self.session_urls_env_var
        suffix = "".join(char.upper() if char.isalnum() else "_" for char in self.resolved_label)
        return f"OPENREWARD_SESSION_URLS_{suffix}"

    @classmethod
    def from_mapping(
        cls,
        value: OpenRewardEnvironmentConfig | dict[str, Any],
    ) -> OpenRewardEnvironmentConfig:
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**dict(cast(dict[str, Any], value)))
        raise TypeError(f"Unsupported OpenReward environment value: {type(value).__name__}")


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
    task_indices: list[int] | None = None
    max_tool_calls: int = 0
    # When present, these entries replace the legacy single-environment fields.
    # Equal sampling weights are the default auto-balanced mixture.
    environments: list[OpenRewardEnvironmentConfig] | None = None
    # AReaL completion order can otherwise let a faster environment crowd a
    # slower one out of an optimizer step. Strict balance admits exactly the
    # sampler-derived weighted quota and retries only missing labels.
    balance_accepted_batches: bool = True
    accepted_batch_max_replacement_rounds: int = 8
    enable_programmatic_tool_calling: bool = False
    enable_task_tracker: bool = False
    enable_recursive_subagents: bool = False
    # Forked agents share the root OpenReward session. ``read_only`` narrows
    # only child tool schemas; the root retains the full environment tool set.
    subagent_environment_access: str = "shared"
    subagent_default_max_steps: int = 50
    subagent_max_depth: int | None = None
    enable_subagent_reward_judging: bool = False
    subagent_reward_judge_max_steps: int = 20
    subagent_delegation_reward_coefficient: float = 0.0
    openhands_system_prompt_suffix: str | None = None
    # Context condensation is a separate state-maintenance completion. It
    # should not place private reasoning back into the agent's context. Keep
    # the normal rollout output budget: reasoning-capable local models often
    # need more than 2k tokens before producing the public summary.
    condenser_disable_thinking: bool = True
    condenser_max_completion_tokens: int = 4_096

    def __post_init__(self) -> None:
        if not isinstance(self.balance_accepted_batches, bool):
            raise ValueError("balance_accepted_batches must be a boolean")
        if not isinstance(self.condenser_disable_thinking, bool):
            raise ValueError("condenser_disable_thinking must be a boolean")
        if (
            isinstance(self.condenser_max_completion_tokens, bool)
            or not isinstance(self.condenser_max_completion_tokens, int)
            or self.condenser_max_completion_tokens <= 0
        ):
            raise ValueError("condenser_max_completion_tokens must be a positive integer")
        if (
            isinstance(self.accepted_batch_max_replacement_rounds, bool)
            or not isinstance(self.accepted_batch_max_replacement_rounds, int)
            or self.accepted_batch_max_replacement_rounds < 0
        ):
            raise ValueError("accepted_batch_max_replacement_rounds must be a non-negative integer")
        if self.subagent_delegation_reward_coefficient < 0:
            raise ValueError("subagent_delegation_reward_coefficient must be non-negative")
        if self.subagent_environment_access not in SUBAGENT_ENVIRONMENT_ACCESS_MODES:
            raise ValueError("OpenReward subagent_environment_access must be 'shared' or 'read_only'")
        if self.environments is not None:
            self.environments = [OpenRewardEnvironmentConfig.from_mapping(value) for value in self.environments]
            if not self.environments:
                raise ValueError("openreward.environments must contain at least one environment")
            labels = [environment.resolved_label for environment in self.environments]
            if len(set(labels)) != len(labels):
                raise ValueError("OpenReward environment labels must be unique")
            session_pool_env_vars = [environment.resolved_session_urls_env_var for environment in self.environments]
            if len(set(session_pool_env_vars)) != len(session_pool_env_vars):
                raise ValueError("OpenReward environment session URL pool env-var names must be unique")
        # Validate legacy environment fields through the same typed contract.
        if self.environments is None:
            self._legacy_environment()

    def _legacy_environment(self) -> OpenRewardEnvironmentConfig:
        return OpenRewardEnvironmentConfig(
            env_name=self.env_name,
            label=self.env_name,
            split=self.split,
            eval_split=self.eval_split,
            session_url=self.session_url,
            api_url=self.api_url,
            api_key=self.api_key,
            train_task_limit=self.train_task_limit,
            eval_task_limit=self.eval_task_limit,
            task_names=self.task_names,
            task_indices=self.task_indices,
            max_tool_calls=self.max_tool_calls,
        )

    def resolved_environments(self) -> list[OpenRewardEnvironmentConfig]:
        if self.environments is None:
            return [self._legacy_environment()]
        return list(self.environments)

    @property
    def is_mixture(self) -> bool:
        return self.environments is not None

    def environment(self, label: str | None = None) -> OpenRewardEnvironmentConfig:
        environments = self.resolved_environments()
        if label is None:
            if len(environments) != 1:
                raise ValueError("A mixed OpenReward config requires an environment label on every task")
            return environments[0]
        for environment in environments:
            if environment.resolved_label == label:
                return environment
        available = [environment.resolved_label for environment in environments]
        raise ValueError(f"Unknown OpenReward environment label {label!r}; available: {available}")

    def subagent_environment_access_for(
        self,
        environment: OpenRewardEnvironmentConfig,
    ) -> str:
        """Resolve an environment override against the rollout-wide policy."""

        return environment.subagent_environment_access or self.subagent_environment_access

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
