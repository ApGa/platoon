"""The two 15-node miles-tinker Toolathlon yamls load into the Tinker config.

``load_config`` drops unknown keys silently, so a strictness walk asserts that
every yaml key maps to a dataclass field, and the recursive/non-recursive pair
differs in exactly two openreward flags plus the stats identity.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from pathlib import Path

import pytest
import yaml

from platoon.openreward.tinker_config import OpenRewardTinkerTrainerConfig
from platoon.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "plugins/openreward/platoon/openreward/configs/tinker"
RECURSIVE = CONFIG_DIR / "abci_toolathlon_15node_recursive_qwen36_miles_tinker_r3_v1.yaml"
NONRECURSIVE = CONFIG_DIR / "abci_toolathlon_15node_nonrecursive_qwen36_miles_tinker_r3_v1.yaml"


def _load(path: Path) -> OpenRewardTinkerTrainerConfig:
    config, _raw = load_config(["--config", str(path)], OpenRewardTinkerTrainerConfig)
    return config


def _dataclass_in(type_hint) -> type | None:
    """Return the dataclass type wrapped by ``type_hint`` (X, X | None, list[X])."""

    if dataclasses.is_dataclass(type_hint):
        return type_hint
    origin = typing.get_origin(type_hint)
    if origin in (typing.Union, types.UnionType, list):
        for arg in typing.get_args(type_hint):
            found = _dataclass_in(arg)
            if found is not None:
                return found
    return None


def _unknown_keys(cls: type, data: dict, prefix: str = "") -> list[str]:
    hints = typing.get_type_hints(cls)
    known = {f.name: hints.get(f.name, f.type) for f in dataclasses.fields(cls)}
    unknown: list[str] = []
    for key, value in data.items():
        path = f"{prefix}{key}"
        if key not in known:
            unknown.append(path)
            continue
        nested = _dataclass_in(known[key])
        if nested is None:
            continue
        if isinstance(value, dict):
            unknown.extend(_unknown_keys(nested, value, f"{path}."))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    unknown.extend(_unknown_keys(nested, item, f"{path}[{index}]."))
    return unknown


@pytest.mark.parametrize("path", [RECURSIVE, NONRECURSIVE], ids=["recursive", "nonrecursive"])
def test_every_yaml_key_maps_to_a_dataclass_field(path: Path):
    raw = yaml.safe_load(path.read_text())
    assert _unknown_keys(OpenRewardTinkerTrainerConfig, raw) == []


@pytest.mark.parametrize("path", [RECURSIVE, NONRECURSIVE], ids=["recursive", "nonrecursive"])
def test_yaml_values(path: Path):
    config = _load(path)
    train = config.train
    assert train.model_name == "apurvaga/Qwen3.6-35B-A3B-preserve-thinking"
    assert train.renderer_name == "qwen3_6_preserve_thinking"
    assert train.renderer_kwargs == {"preserve_thinking": True, "enable_thinking": True}
    assert train.context_window_length == 65536
    assert (train.batch_size, train.num_epochs, train.max_training_steps) == (8, 10, None)
    assert (train.num_minibatches, train.num_microbatches, train.max_staleness) == (1, 1, 3)
    assert train.loss_fn == "cispo"
    assert train.loss_fn_config == {"clip_low_threshold": 0.0, "clip_high_threshold": 5.0}
    optimizer = train.optimizer
    assert (optimizer.learning_rate, optimizer.beta1, optimizer.beta2) == (3.0e-6, 0.9, 0.95)
    assert (optimizer.eps, optimizer.weight_decay, optimizer.grad_clip_norm) == (1.0e-8, 0.017, 1.0)
    assert train.num_concurrent_rollout_workflow_workers == 3

    workflow = train.workflow_config
    assert workflow.group_size == 8
    assert workflow.leave_one_out_baseline is True
    assert workflow.depth_level_weighting is True
    assert workflow.subagent_datum_keep_probability == 1.0
    assert workflow.subagent_datum_sampling_seed == 343
    assert workflow.filter_errors is True
    assert workflow.filter_zero_advantage_datums is False
    assert workflow.straggler_timeout_seconds == 900
    assert workflow.straggler_quorum == 6
    assert workflow.min_successful_group_size == 4
    rollout = workflow.rollout_config
    assert (rollout.max_steps, rollout.timeout, rollout.step_timeout) == (200, 3600, 2700)
    assert rollout.propagate_root_success is False
    assert rollout.verbose is True
    params = rollout.inference_params
    assert (params.temperature, params.top_p, params.max_completion_tokens) == (1.0, 1.0, 4096)

    assert config.eval.strategy == "none"
    assert config.eval.every == 0
    assert config.eval.num_concurrent_rollout_workflow_workers == 1
    assert config.eval.workflow_config.group_size == 1
    assert config.eval.workflow_config.filter_errors is False
    assert config.eval.workflow_config.rollout_config.max_steps == 200

    assert config.checkpoint.strategy == "step"
    assert config.checkpoint.every == 20
    assert config.checkpoint.recovery_every == 1
    assert config.checkpoint.load_checkpoint_path is None
    assert config.tinker_base_url is None
    assert config.log_path == "/groups/gai51740/aci18912it/miles-tinker/runs/default/logs"
    assert (config.watchdog.enabled, config.watchdog.timeout_seconds, config.watchdog.exit_code) == (True, 7200, 2)
    assert config.stats.experiment_name == "toolathlon-openhands-miles-tinker"
    assert config.stats.wandb.mode == "online"
    assert config.stats.wandb.project == "toolathlon-openhands"
    assert config.seed == 343

    openreward = config.openreward
    assert openreward.env_name == "toolathlongym"
    assert openreward.session_url == "http://localhost:8082"
    assert openreward.eval_task_limit is None
    assert openreward.balance_accepted_batches is False
    assert openreward.programmatic_tool_calling_mode == "orchestration_only"
    assert openreward.enable_task_tracker is True
    assert openreward.subagent_max_depth == 2
    assert openreward.subagent_default_max_steps == 100
    assert openreward.enable_subagent_reward_judging is True
    assert openreward.enable_subagent_behavior_judging is True
    assert openreward.enable_subagent_parent_usage_eligibility_gate is True
    assert openreward.condenser_max_tokens == 52428
    assert openreward.condenser_max_completion_tokens == 26214
    environments = openreward.resolved_environments()
    assert len(environments) == 1
    assert environments[0].label == "toolathlon"
    assert environments[0].session_urls_env_var == "OPENREWARD_SESSION_URLS_TOOLATHLON"
    assert environments[0].sampling_weight == 1.0


def test_recursive_yaml_is_the_recursive_variant():
    config = _load(RECURSIVE)
    assert config.openreward.enable_recursive_subagents is True
    assert config.openreward.enable_programmatic_tool_calling is True
    assert "recursive" in config.stats.wandb.tags
    assert "-rec-" in config.stats.trial_name


def _flatten(value, prefix: str = "") -> dict[str, object]:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        out: dict[str, object] = {}
        for key, item in value.items():
            out.update(_flatten(item, f"{prefix}{key}."))
        return out
    if isinstance(value, list):
        out = {}
        for index, item in enumerate(value):
            out.update(_flatten(item, f"{prefix}{index}."))
        return out
    return {prefix.rstrip("."): value}


def test_nonrecursive_differs_only_in_recursion_and_stats_identity():
    recursive = _flatten(_load(RECURSIVE))
    nonrecursive = _flatten(_load(NONRECURSIVE))
    assert recursive.keys() == nonrecursive.keys()
    diff = {key for key in recursive if recursive[key] != nonrecursive[key]}
    # Programmatic tool calling stays ON in both arms so the only difference is recursion.
    assert diff == {
        "openreward.enable_recursive_subagents",
        "stats.trial_name",
        "stats.wandb.group",
        "stats.wandb.tags.7",
    }
    assert nonrecursive["openreward.enable_recursive_subagents"] is False
    assert nonrecursive["openreward.enable_programmatic_tool_calling"] is True
    assert nonrecursive["stats.wandb.tags.7"] == "nonrecursive"
    assert "-nonrec-" in nonrecursive["stats.trial_name"]
    assert "nonrec" in nonrecursive["stats.wandb.group"]
    # Judging flags stay on in both (they never fire without launch_subagent).
    assert nonrecursive["openreward.enable_subagent_behavior_judging"] is True


def test_strictness_walk_detects_unknown_keys():
    raw = yaml.safe_load(RECURSIVE.read_text())
    raw["train"]["workflow_config"]["use_subprocesses"] = True
    raw["openreward"]["environments"][0]["bogus"] = 1
    raw["rollout"] = {"max_concurrent_rollouts": 24}
    assert sorted(_unknown_keys(OpenRewardTinkerTrainerConfig, raw)) == [
        "openreward.environments[0].bogus",
        "rollout",
        "train.workflow_config.use_subprocesses",
    ]
