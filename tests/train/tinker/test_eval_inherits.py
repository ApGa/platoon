"""A training config's ``eval.workflow_config`` is laid over ``train.workflow_config`` (RFC 0018, section 2, point 2):
the keys it leaves out keep training's rollout settings (``rollout_config``) and rollout-worker settings
(``rollout_isolation``, ``worker_start_method``, the ``worker_*`` keys), and evaluation's own defaults for the keys
that shape groups, advantages and filters.  Sections merge key by key (``inference_params`` among them), any other
value the eval section gives replaces training's, ``null`` included, and a free-form ``extra`` replaces it whole.

``test_eval_inherits_run.py`` runs the trainer on such a config and checks what its evaluation's rollouts ran with.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import yaml

from platoon import cli
from platoon.deprecations import PlatoonDeprecationWarning
from platoon.train.tinker.config_defs import (
    EVAL_INHERITED_KEYS,
    EVAL_OWN_KEYS,
    EvalConfig,
    PlatoonTinkerRLTrainerConfig,
    WorkflowConfig,
)
from platoon.utils.config import ConfigError, resolve_config

# train.workflow_config of a run whose evaluation should roll out as its training does: every inherited key, and
# every group, advantage and filter key, away from its default.
TRAIN_WORKFLOW: dict[str, Any] = {
    "group_size": 8,
    "leave_one_out_baseline": True,
    "depth_level_weighting": True,
    "subagent_datum_keep_probability": 0.5,
    "subagent_datum_sampling_seed": 7,
    "filter_errors": True,
    "filter_message_actions": True,
    "filter_zero_advantage_datums": True,
    "straggler_timeout_seconds": 600.0,
    "straggler_quorum": 6,
    "straggler_policies": {"slow": {"timeout_seconds": 60.0, "quorum": 4}},
    "min_successful_group_size": 4,
    "rollout_isolation": "process",
    "worker_start_method": "exec",
    "worker_startup_budget_seconds": 123.0,
    "worker_finalize_grace_seconds": 11.0,
    "worker_parent_grace_seconds": 7.0,
    "worker_termination_grace_seconds": 3.0,
    "worker_launch_concurrent_groups": 5,
    "worker_rss_soft_limit_bytes": 12345678901,
    "worker_mem_available_floor_bytes": 1024,
    "worker_binding_poll_seconds": 2.5,
    "worker_unbounded_rollout_budget_seconds": 9999.0,
    "worker_python_env_isolation": True,
    "rollout_config": {
        "max_steps": 200,
        "timeout": 3600,
        "step_timeout": 900,
        "verbose": False,
        "propagate_root_success": True,
        "skip_subagent_reward_computation": True,
        "enable_subagent_verifiers": False,
        "sampling_precedence": "config",
        "extra": {"token_efficiency_reward": {"enabled": False}, "process_env": {"A": "1"}},
        "inference_params": {"temperature": 1.0, "top_p": 0.95, "top_k": 40, "max_completion_tokens": 49152},
    },
}


def _write(tmp_path: Path, eval_section: dict[str, Any], workflow: dict[str, Any] | None = None) -> Path:
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "train": {
                    "model_name": "m",
                    "renderer_name": "r",
                    "workflow_config": TRAIN_WORKFLOW if workflow is None else workflow,
                },
                "eval": eval_section,
                "log_path": str(tmp_path / "logs"),
            },
            sort_keys=False,
        )
    )
    return path


def _load(tmp_path: Path, eval_section: dict[str, Any], *overrides: str, workflow: dict[str, Any] | None = None):
    return resolve_config(_write(tmp_path, eval_section, workflow), list(overrides), PlatoonTinkerRLTrainerConfig)


def test_every_workflow_key_is_inherited_or_evaluations_own():
    names = [item.name for item in dataclasses.fields(WorkflowConfig)]

    assert sorted(EVAL_INHERITED_KEYS + EVAL_OWN_KEYS) == sorted(names)
    assert not set(EVAL_INHERITED_KEYS) & set(EVAL_OWN_KEYS)


def test_an_eval_section_that_names_only_every_rolls_out_with_trainings_rollout_and_worker_settings(tmp_path):
    config = _load(tmp_path, {"every": 25}).config
    train, evaluation = config.train.workflow_config, config.eval.workflow_config
    defaults = EvalConfig().workflow_config  # evaluation's own: one rollout per task, no filters

    assert config.eval.every == 25
    for name in EVAL_INHERITED_KEYS:
        assert getattr(evaluation, name) == getattr(train, name), name
    for name in EVAL_OWN_KEYS:
        assert getattr(evaluation, name) == getattr(defaults, name), name
    assert evaluation.rollout_config is not train.rollout_config  # a copy, which nothing shares
    assert evaluation.rollout_config.inference_params.max_completion_tokens == 49152
    assert (evaluation.group_size, evaluation.filter_errors, evaluation.straggler_policies) == (1, False, {})


def test_without_training_settings_an_evaluation_rolls_out_as_before(tmp_path):
    config = _load(tmp_path, {"every": 25}, workflow={}).config

    assert config.eval.workflow_config == EvalConfig().workflow_config


def test_inference_params_merge_key_by_key(tmp_path):
    # An evaluation that sets only its temperature keeps training's completion budget, not the gateway's 4,096.
    rollout = {"inference_params": {"temperature": 0.6}}
    params = _load(tmp_path, {"workflow_config": {"rollout_config": rollout}}).config.eval.workflow_config

    assert dataclasses.asdict(params.rollout_config.inference_params) == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 40,
        "max_completion_tokens": 49152,
    }
    assert params.rollout_config.max_steps == 200  # the rest of rollout_config stays training's


def test_null_unsets_a_training_value(tmp_path):
    rollout = {"timeout": None, "max_steps": None, "inference_params": {"top_p": None, "max_completion_tokens": None}}
    evaluation = _load(tmp_path, {"workflow_config": {"rollout_config": rollout}}).config.eval.workflow_config

    assert (evaluation.rollout_config.timeout, evaluation.rollout_config.max_steps) == (None, None)
    params = evaluation.rollout_config.inference_params
    assert (params.temperature, params.top_p, params.top_k, params.max_completion_tokens) == (1.0, None, 40, None)


@pytest.mark.parametrize("extra", [{"other": 2}, {}])
def test_an_extra_replaces_trainings_whole(tmp_path, extra):
    evaluation = _load(tmp_path, {"workflow_config": {"rollout_config": {"extra": extra}}}).config.eval.workflow_config

    assert evaluation.rollout_config.extra == extra


def test_the_keys_the_eval_section_sets_win(tmp_path):
    workflow = {
        "group_size": 2,
        "filter_errors": True,
        "rollout_isolation": "inprocess",
        "worker_python_env_isolation": False,  # which inprocess requires
        "worker_start_method": "forkserver",
        "rollout_config": {"max_steps": 30, "sampling_precedence": "request"},
    }
    evaluation = _load(tmp_path, {"workflow_config": workflow}).config.eval.workflow_config

    assert (evaluation.group_size, evaluation.filter_errors) == (2, True)
    assert (evaluation.rollout_isolation, evaluation.worker_start_method) == ("inprocess", "forkserver")
    assert (evaluation.rollout_config.max_steps, evaluation.rollout_config.sampling_precedence) == (30, "request")
    assert evaluation.rollout_config.timeout == 3600 and evaluation.worker_startup_budget_seconds == 123.0


def test_overrides_reach_what_the_evaluation_inherits(tmp_path):
    config = _load(
        tmp_path,
        {"every": 5},
        "train.workflow_config.rollout_config.max_steps=7",
        "eval.workflow_config.rollout_config.timeout=60",
    ).config

    assert (
        config.train.workflow_config.rollout_config.max_steps,
        config.eval.workflow_config.rollout_config.max_steps,
    ) == (7, 7)
    assert (
        config.train.workflow_config.rollout_config.timeout,
        config.eval.workflow_config.rollout_config.timeout,
    ) == (3600, 60)


def test_a_deprecated_rollout_key_training_sets_is_inherited_and_the_evaluation_may_replace_it(tmp_path):
    workflow = {"rollout_config": {"propogate_root_success": True}}

    with pytest.warns(PlatoonDeprecationWarning) as warned:
        inherited = _load(tmp_path, {"every": 1}, workflow=workflow)
    with pytest.warns(PlatoonDeprecationWarning):
        replaced = _load(
            tmp_path, {"workflow_config": {"rollout_config": {"propagate_root_success": False}}}, workflow=workflow
        )

    assert [str(item.message).partition(":")[0] for item in warned if item.category is PlatoonDeprecationWarning] == [
        "train.workflow_config.rollout_config.propogate_root_success"  # training's only: evaluation sets nothing
    ]
    assert inherited.config.eval.workflow_config.rollout_config.propagate_root_success is True
    assert replaced.config.eval.workflow_config.rollout_config.propagate_root_success is False
    assert replaced.config.train.workflow_config.rollout_config.propagate_root_success is True


def test_the_deprecated_name_of_a_rollout_key_training_sets_replaces_it_in_the_evaluation(tmp_path):
    # Training sets propagate_root_success; the evaluation sets its deprecated name, which is the same key.
    workflow = {"rollout_config": {"propagate_root_success": True}}
    eval_section = {"workflow_config": {"rollout_config": {"propogate_root_success": False}}}

    with pytest.warns(
        PlatoonDeprecationWarning, match=r"eval\.workflow_config\.rollout_config\.propogate_root_success"
    ):
        config = _load(tmp_path, eval_section, workflow=workflow).config

    assert config.eval.workflow_config.rollout_config.propagate_root_success is False
    assert config.train.workflow_config.rollout_config.propagate_root_success is True


def test_a_refused_eval_section_names_the_training_settings_it_kept(tmp_path):
    with pytest.raises(ConfigError) as raised:
        _load(tmp_path, {"workflow_config": {"rollout_isolation": "inprocess"}})

    message = str(raised.value)
    assert "eval.workflow_config: worker_python_env_isolation requires rollout_isolation='process'" in message
    assert "worker_python_env_isolation: true" in message and "group_size: 1" in message


def test_the_resolved_record_loads_to_the_same_evaluation(tmp_path):
    first = _load(tmp_path, {"every": 3, "workflow_config": {"rollout_config": {"inference_params": {"top_k": 5}}}})
    snapshot = tmp_path / "config.resolved.yaml"
    snapshot.write_text(yaml.safe_dump(first.data, sort_keys=False))  # what platoon train writes and then loads

    second = resolve_config(snapshot, [], PlatoonTinkerRLTrainerConfig)

    assert second.config.eval == first.config.eval
    assert "rollout_isolation" not in second.data["eval"].get("workflow_config", {})  # the record keeps the config's


def test_config_show_defaults_prints_what_the_evaluation_runs_with(tmp_path, capsys):
    cli.main(["config", "show", str(_write(tmp_path, {"every": 3})), "--defaults"])

    shown = yaml.safe_load(capsys.readouterr().out)["eval"]["workflow_config"]
    assert (shown["group_size"], shown["rollout_isolation"]) == (1, "process")
    assert shown["rollout_config"]["max_steps"] == 200
    assert shown["rollout_config"]["inference_params"]["max_completion_tokens"] == 49152
    assert "propogate_root_success" not in shown["rollout_config"]


def test_platoon_rollout_of_the_eval_split_rolls_out_as_the_evaluation_does(tmp_path):
    config = _load(tmp_path, {"every": 3}).config

    rollout_config = cli._rollout_config(config, "eval", tmp_path / "run")

    assert (rollout_config.max_steps, rollout_config.timeout) == (200, 3600)
    assert rollout_config.inference_params.max_completion_tokens == 49152
