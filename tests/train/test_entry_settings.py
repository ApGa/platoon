"""Per-environment training settings: what an entry may set, how its settings are laid over the run's in each phase,
and what loading refuses (``platoon.train.entry_settings``)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from fakes.config import make_config

from platoon.config_defs import RolloutConfig
from platoon.eval.run import PlatoonEvalConfig
from platoon.train import entry_settings
from platoon.train.components import (
    ENTRY_ROLLOUT_KEYS,
    ENTRY_WORKFLOW_KEYS,
    ENVIRONMENT_COLUMN,
    EntryRolloutSettings,
    EntryWorkflowSettings,
    EnvironmentConfig,
)
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig, WorkflowConfig, training_problems
from platoon.utils.config import ConfigError, given_mapping, resolve_config

# A mix of a quick environment with a calibrated step budget and a slow one whose groups take many steps.
MIX: dict[str, Any] = {
    "train": {
        "model_name": "m",
        "renderer_name": "r",
        "batch_size": 4,
        "max_staleness": 3,
        "rollout_group_limits": {"slow": 1},
        "workflow_config": {
            "group_size": 8,
            "min_successful_group_size": 4,
            "rollout_config": {
                "max_steps": 200,
                "timeout": 3600,
                "inference_params": {"temperature": 1.0, "max_completion_tokens": 49152},
                "extra": {"a": 1, "b": 2},
            },
        },
    },
    "checkpoint": {"recovery_every": 10, "keep_recovery": 5},
    "eval": {"every": 10, "workflow_config": {"group_size": 4}},
    "log_path": "runs",
    "environments": [
        {"name": "quick", "rollout": "x:y", "workflow_config": {"rollout_config": {"max_steps": 30}}},
        {
            "name": "slow",
            "rollout": "x:y",
            "max_staleness": 48,
            "workflow_config": {
                "group_size": 4,
                "min_successful_group_size": 3,
                "leave_one_out_baseline": True,
                "straggler_timeout_seconds": None,
                "rollout_config": {
                    "max_steps": None,
                    "timeout": None,
                    "inference_params": {"temperature": 0.7},
                    "extra": {"c": 3},
                },
            },
            "eval_overrides": {"workflow_config": {"group_size": 1}},
        },
    ],
}


def _write(tmp_path: Path, data: dict[str, Any], name: str = "train.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _load(tmp_path: Path, data: dict[str, Any] = MIX, *overrides: str) -> PlatoonTinkerRLTrainerConfig:
    return resolve_config(_write(tmp_path, data), overrides, PlatoonTinkerRLTrainerConfig).config


def _problems(tmp_path: Path, data: dict[str, Any], *overrides: str) -> list[str]:
    with pytest.raises(ConfigError) as error:
        _load(tmp_path, data, *overrides)
    return error.value.problems


def _with(**entry: Any) -> dict[str, Any]:
    """MIX with the slow entry's keys replaced by ``entry``'s (None removes one)."""

    data = yaml.safe_load(yaml.safe_dump(MIX))
    slow = data["environments"][1]
    for key, value in entry.items():
        if value is None:
            slow.pop(key, None)
        else:
            slow[key] = value
    return data


# --- what an entry may set ------------------------------------------------------------------------------------------


def test_the_entry_schema_takes_the_per_group_keys_with_the_types_of_the_run_s():
    """An entry sets the keys that act on one group or one rollout, each typed as the workflow config's own."""

    import typing

    run = typing.get_type_hints(WorkflowConfig)
    rollout = typing.get_type_hints(RolloutConfig)
    entry = typing.get_type_hints(EntryWorkflowSettings)
    entry_rollout = typing.get_type_hints(EntryRolloutSettings)
    assert set(ENTRY_WORKFLOW_KEYS) == {
        "group_size",
        "leave_one_out_baseline",
        "min_successful_group_size",
        "straggler_timeout_seconds",
        "straggler_quorum",
        "filter_errors",
        "filter_message_actions",
        "subagent_datum_keep_probability",
        "subagent_datum_sampling_seed",
        "rollout_config",
    }
    assert all(entry[name] == run[name] for name in ENTRY_WORKFLOW_KEYS if name != "rollout_config")
    assert set(ENTRY_ROLLOUT_KEYS) == {
        "max_steps",
        "timeout",
        "step_timeout",
        "inference_params",
        "sampling_precedence",
        "extra",
        "propagate_root_success",
        "skip_subagent_reward_computation",
        "enable_subagent_verifiers",
    }
    assert all(entry_rollout[name] == rollout[name] for name in ENTRY_ROLLOUT_KEYS)


def test_a_key_an_entry_may_not_set_says_why_and_where_it_goes_with_every_problem_at_once(tmp_path):
    data = _with(
        workflow_config={
            "filter_zero_advantage_datums": False,
            "rollout_isolation": "process",
            "group_size": "four",
            "rollout_config": {"judges": False, "max_step": 3, "inference_params": {"temprature": 0.5}},
        }
    )

    problems = _problems(tmp_path, data)

    where = "environments.1.workflow_config"

    def starts(prefix: str) -> bool:
        return any(problem.startswith(prefix) for problem in problems)

    assert starts(
        f"{where}.filter_zero_advantage_datums: unknown key; an entry does not set it: it filters microbatches"
    )
    assert starts(f"{where}.rollout_isolation: unknown key; an entry does not set it: one pool of rollout workers")
    assert f"{where}.group_size: 'four' is not a int" in problems
    assert starts(f"{where}.rollout_config.judges: unknown key; an entry does not set it: training runs its judges")
    assert f"{where}.rollout_config.max_step: unknown key; did you mean {where}.rollout_config.max_steps?" in problems
    assert (
        f"{where}.rollout_config.inference_params.temprature: unknown key; did you mean "
        f"{where}.rollout_config.inference_params.temperature?"
    ) in problems


def test_an_entry_s_evaluation_settings_are_checked_as_its_own_are(tmp_path):
    data = _with(eval_overrides={"workflow_config": {"depth_level_weighting": True}, "max_steps": 30})

    problems = _problems(tmp_path, data)

    assert any(
        problem.startswith("environments.1.eval_overrides.workflow_config.depth_level_weighting: unknown key; an entry")
        for problem in problems
    )
    assert any(
        problem.startswith("environments.1.eval_overrides.max_steps: unknown key; an entry's step limit for its")
        for problem in problems
    )


def test_an_override_reaches_an_entry_s_settings_typed_and_checked(tmp_path):
    config = _load(tmp_path, MIX, "environments.1.workflow_config.rollout_config.max_steps=12")
    assert config.environments[1].workflow_config["rollout_config"]["max_steps"] == 12
    assert entry_settings.entry_settings(config)["slow"].train.rollout_config.max_steps == 12
    with pytest.raises(ConfigError, match="environments.1.workflow_config.worker_start_method: unknown key; an entry"):
        _load(tmp_path, MIX, "environments.1.workflow_config.worker_start_method=exec")


def test_an_entry_built_in_python_is_checked_too():
    with pytest.raises(ValueError, match=r"environment 'x': workflow_config.depth_level_weighting: unknown key"):
        EnvironmentConfig(name="x", workflow_config={"depth_level_weighting": True})
    with pytest.raises(ValueError, match=r"max_staleness must be a whole number of policy versions"):
        EnvironmentConfig(name="x", max_staleness=-1)
    entry = EnvironmentConfig(name="x", workflow_config={"rollout_config": {"max_steps": None}})
    assert entry.workflow_config == {"rollout_config": {"max_steps": None}}


# --- how the settings are laid over the run's --------------------------------------------------------------------


def test_training_groups_lay_the_entry_over_train_workflow_config(tmp_path):
    """Precedence for an entry's training groups: the defaults, train.workflow_config, then the entry; a section merges
    key by key, null unsets the run's value, and a free-form mapping replaces the run's whole."""

    config = _load(tmp_path)
    settings = entry_settings.entry_settings(config)

    quick, slow = settings["quick"].train, settings["slow"].train
    assert (quick.group_size, quick.min_successful_group_size, quick.leave_one_out_baseline) == (8, 4, False)
    assert (quick.rollout_config.max_steps, quick.rollout_config.timeout) == (30, 3600)
    assert (slow.group_size, slow.min_successful_group_size, slow.leave_one_out_baseline) == (4, 3, True)
    assert (slow.rollout_config.max_steps, slow.rollout_config.timeout, slow.straggler_timeout_seconds) == (
        None,
        None,
        None,
    )
    assert slow.rollout_config.inference_params.temperature == 0.7
    assert slow.rollout_config.inference_params.max_completion_tokens == 49152  # merged key by key
    assert slow.rollout_config.extra == {"c": 3}  # replaced whole
    assert type(slow) is type(config.train.workflow_config)
    # An entry that sets nothing runs with the run's settings themselves.
    plain = _load(tmp_path, {**MIX, "environments": [{"name": "a", "rollout": "x:y"}, {"name": "b", "rollout": "x:y"}]})
    assert entry_settings.entry_settings(plain)["a"].train is plain.train.workflow_config


def test_evaluation_during_training_inherits_the_entry_s_rollout_settings_below_eval_s_own(tmp_path):
    """Precedence for an entry's evaluation rollouts during training: evaluation's defaults, the entry's training
    settings for the keys evaluation takes from training, the keys eval.workflow_config sets, then the entry's
    eval_overrides.  Group keys never flow from training into evaluation."""

    config = _load(tmp_path)
    settings = entry_settings.entry_settings(config)

    quick, slow = settings["quick"].eval, settings["slow"].eval
    assert quick.rollout_config.max_steps == 30  # the entry's, inherited
    assert quick.group_size == 4 and quick.min_successful_group_size == 1  # eval.workflow_config's, evaluation's
    assert slow.group_size == 1  # its eval_overrides win
    assert (slow.rollout_config.max_steps, slow.rollout_config.timeout) == (None, None)
    assert slow.leave_one_out_baseline is False  # a group key of training's does not flow into evaluation

    # The phase's own section wins over the entry: a run-wide evaluation step limit overrides the entry's 30.
    data = yaml.safe_load(yaml.safe_dump(MIX))
    data["eval"]["workflow_config"]["rollout_config"] = {"max_steps": 100}
    overridden = _load(tmp_path, data)
    assert entry_settings.entry_settings(overridden)["quick"].eval.rollout_config.max_steps == 100
    lines = entry_settings.warnings(overridden, entry_settings.entry_settings(overridden))
    assert any(
        "environments[quick].workflow_config.rollout_config.max_steps: 30 is overridden in its evaluation rollouts "
        "by eval.workflow_config.rollout_config.max_steps: 100" in line
        for line in lines
    )
    # ... unless the entry's own eval_overrides keep it.
    data["environments"][0]["eval_overrides"] = {"workflow_config": {"rollout_config": {"max_steps": 30}}}
    kept = _load(tmp_path, data)
    assert entry_settings.entry_settings(kept)["quick"].eval.rollout_config.max_steps == 30


def test_the_loader_keeps_the_mapping_a_config_gave_for_each_section(tmp_path):
    config = _load(tmp_path)

    assert given_mapping(config.eval.workflow_config) == {"group_size": 4}
    assert given_mapping(config.train.workflow_config)["rollout_config"]["max_steps"] == 200
    assert given_mapping(config.checkpoint) == {"recovery_every": 10, "keep_recovery": 5}
    assert given_mapping(config.stats) == {}  # left out
    assert given_mapping(WorkflowConfig()) is None  # built in Python


def test_platoon_eval_applies_an_entry_s_rollout_settings_below_inference_workflow_s(tmp_path):
    """Precedence for an entry's rollouts in platoon eval: the defaults, the entry's workflow_config.rollout_config,
    the keys inference.workflow.rollout_config sets, then its eval_overrides; the rest of an entry's settings are
    training's, which platoon eval lists as not applied."""

    data = {
        "inference": {
            "model_name": "m",
            "renderer_name": "r",
            "workflow": {"rollout_config": {"timeout": 900, "inference_params": {"temperature": 0.6}}},
        },
        "environments": [
            {"name": "plain", "rollout": "x:y"},
            {
                "name": "calibrated",
                "rollout": "x:y",
                "max_staleness": 5,
                "workflow_config": {"group_size": 4, "rollout_config": {"max_steps": 30, "timeout": 60}},
                "eval_overrides": {"workflow_config": {"rollout_config": {"inference_params": {"top_p": 0.9}}}},
            },
        ],
    }
    config = resolve_config(_write(tmp_path, data, "eval.yaml"), [], PlatoonEvalConfig).config

    configs = entry_settings.evaluation_rollout_configs(config)

    assert configs["plain"] is config.inference.workflow.rollout_config
    calibrated = configs["calibrated"]
    assert calibrated.max_steps == 30  # the entry's
    assert calibrated.timeout == 900  # inference.workflow's wins over the entry's 60
    assert (calibrated.inference_params.temperature, calibrated.inference_params.top_p) == (0.6, 0.9)
    assert entry_settings.evaluation_unapplied(config) == [
        "environments[calibrated].workflow_config.group_size: platoon eval applies an entry's rollout settings only "
        "(rollout_config); inference.workflow sets how its rollouts run, such as num_rollouts_per_task",
        "environments[calibrated].max_staleness: a training setting, which platoon eval does not apply",
    ]


def test_platoon_eval_of_the_train_split_applies_no_eval_overrides(tmp_path):
    """An entry's eval_overrides apply to every split but train, its eval_overrides.workflow_config.rollout_config as
    its other evaluation overrides: platoon eval (and platoon rollout) of the train split runs the entry's
    workflow_config.rollout_config beneath inference.workflow's, and says that the overrides do not apply."""

    from platoon import cli

    data = {
        "inference": {"model_name": "m", "renderer_name": "r"},
        "environments": [
            {"name": "plain", "rollout": "x:y"},
            {
                "name": "calibrated",
                "rollout": "x:y",
                "workflow_config": {"rollout_config": {"max_steps": 30}},
                "eval_overrides": {"workflow_config": {"rollout_config": {"max_steps": 5}}},
            },
        ],
    }
    path = _write(tmp_path, data, "eval.yaml")
    evaluation = resolve_config(path, [], PlatoonEvalConfig).config
    train_split = resolve_config(path, ["split=train"], PlatoonEvalConfig).config

    assert entry_settings.evaluation_rollout_configs(evaluation)["calibrated"].max_steps == 5
    assert entry_settings.evaluation_rollout_configs(train_split)["calibrated"].max_steps == 30
    assert entry_settings.evaluation_rollout_configs(evaluation, "train")["calibrated"].max_steps == 30
    assert cli._entry_rollout_config(evaluation, "train", "calibrated::a").max_steps == 30  # platoon rollout
    assert cli._entry_rollout_config(evaluation, "test", "calibrated::a").max_steps == 5
    assert entry_settings.evaluation_unapplied(evaluation) == []
    assert entry_settings.evaluation_unapplied(train_split) == [
        "environments[calibrated].eval_overrides.workflow_config.rollout_config: the train split's rollouts take no "
        "eval_overrides, which apply to every other split"
    ]
    lines = entry_settings.describe(train_split, "calibrated")
    assert lines[0] == "environments[calibrated], platoon eval rollouts of the train split:"
    assert any(line.startswith("  rollout_config.max_steps: 30") and " entry " in line for line in lines)


# --- what loading refuses -------------------------------------------------------------------------------------------


def test_laid_settings_that_their_class_refuses_name_the_entry_and_the_values_kept(tmp_path):
    problems = _problems(tmp_path, _with(workflow_config={"min_successful_group_size": 9}))

    assert problems == [
        "environments[slow].workflow_config: min_successful_group_size must be in [1, group_size] (the keys it leaves "
        "out keep train.workflow_config's values: group_size: 8, straggler_timeout_seconds: null, "
        "straggler_quorum: null)"
    ]


def test_an_entry_s_staleness_bound_is_at_least_the_run_s(tmp_path):
    problems = _problems(tmp_path, _with(max_staleness=2))
    assert problems[0].startswith(
        "environments[slow].max_staleness: 2 is below train.max_staleness: 3, which bounds how far every entry's "
        "rollouts run ahead of training"
    )
    data = _with()
    data["train"]["max_staleness"] = None
    data.pop("checkpoint")
    problems = _problems(tmp_path, data)
    assert problems[0].startswith(
        "environments[slow].max_staleness: 48 bounds the entry's groups, but train.max_staleness is null"
    )
    with pytest.raises(ConfigError, match=r"environments.1.max_staleness: 'many' is not a int"):
        _load(tmp_path, _with(max_staleness="many"))


def test_recovery_pruning_keeps_what_the_largest_bound_may_sample_and_names_the_entry_that_sets_it(tmp_path):
    config = _load(tmp_path)
    assert entry_settings.largest_staleness(config) == (48, "environments[slow].max_staleness")
    assert entry_settings.staleness_bounds(config) == {"quick": 3, "slow": 48}

    data = yaml.safe_load(yaml.safe_dump(MIX))
    data["checkpoint"]["keep_recovery"] = 4  # 4 × 10 = 40, within the slow entry's 48

    problems = _problems(tmp_path, data)

    assert problems == [
        "checkpoint.keep_recovery: 4 × checkpoint.recovery_every: 10 = 40 must exceed "
        "environments[slow].max_staleness: 48, the run's largest staleness bound: a recovery checkpoint within 48 "
        "policy versions may still serve rollouts and is never deleted, so more would stay than keep_recovery keeps; "
        "raise checkpoint.keep_recovery to 5 or checkpoint.recovery_every to 13, or lower "
        "environments[slow].max_staleness"
    ]


def test_concurrency_holds_a_group_of_every_entry(tmp_path):
    data = _with(workflow_config={"group_size": 16, "min_successful_group_size": 3})
    data["train"]["max_concurrent_rollouts"] = 12

    problems = _problems(tmp_path, data)

    assert problems == [
        "train.max_concurrent_rollouts: 12 holds no group of environments[slow], whose 16 rollouts start together "
        "(its workflow_config.group_size, laid over the run's); raise it to 16 at least"
    ]


def test_the_rollout_permits_hold_the_capped_labels_at_their_limits_and_a_group_of_every_other_entry(tmp_path):
    """While the capped labels run as many groups as their limits allow, the budget of rollouts must still hold a
    group of the largest other entry: a group that does not fit waits, first in line, for a capped group to end, and
    every group after it with it.  Loading warns; the run goes on, at the capped labels' pace."""

    def permit_warnings(budget: int | None) -> list[str]:
        data = _with()
        if budget is not None:
            data["train"]["max_concurrent_rollouts"] = budget
        config = _load(tmp_path, data)
        lines = entry_settings.warnings(config, entry_settings.entry_settings(config))
        return [line for line in lines if line.startswith("train.max_concurrent_rollouts")]

    # slow: one group of 4 at a time; quick's groups are the run's 8.
    assert permit_warnings(11) == [
        "train.max_concurrent_rollouts: 11 rollouts, of which the groups of the labels with a limit "
        "(train.rollout_group_limits) hold up to 4 (slow: 1 × 4): a group of environments[quick] (8 rollouts) can "
        "then wait for one of theirs to end, and every group after it with it, so those labels set the run's pace; "
        "raise it to 12 at least"
    ]
    assert permit_warnings(12) == []
    assert permit_warnings(None) == []  # the default, the batch's rollouts: 4 × 8

    data = _with()
    data["train"]["rollout_group_limits"] = {"slow": 2, "quick": 1}  # every label capped: they must all fit
    data["train"]["max_concurrent_rollouts"] = 15
    config = _load(tmp_path, data)
    lines = entry_settings.warnings(config, entry_settings.entry_settings(config))
    assert any(line.endswith("raise it to 16 at least") and "(quick: 1 × 8, slow: 2 × 4)" in line for line in lines)


def test_the_rollout_permits_are_checked_by_the_labels_the_listed_tasks_carry():
    """With another label column, which entries' tasks carry a capped label is known once the tasks are listed."""

    config = make_config(
        train={"label_by": "difficulty", "rollout_group_limits": {"hard": 2}, "max_concurrent_rollouts": 15},
        environments=[{"name": "quick"}, {"name": "slow"}],
    )
    sizes = {"quick": 8, "slow": 4}
    settings = {
        label: entry_settings.EntrySettings(label, SimpleNamespace(group_size=size), None, None)
        for label, size in sizes.items()
    }
    rows = [
        {"task_id": "quick::a", ENVIRONMENT_COLUMN: "quick", "difficulty": "easy"},
        {"task_id": "slow::b", ENVIRONMENT_COLUMN: "slow", "difficulty": "hard"},
        {"task_id": "slow::c", ENVIRONMENT_COLUMN: "slow"},  # no label: no limit applies
    ]

    lines = entry_settings.listed_permit_warnings(config, settings, rows)

    assert len(lines) == 1 and "hold up to 8 (hard: 2 × 4)" in lines[0] and lines[0].endswith("raise it to 16 at least")
    rows[0]["difficulty"] = "hard"  # quick's tasks carry it too: its groups of 8 count
    assert "hold up to 16 (hard: 2 × 8)" in entry_settings.listed_permit_warnings(config, settings, rows)[0]
    config.train.max_concurrent_rollouts = 24
    assert entry_settings.listed_permit_warnings(config, settings, rows) == []


def test_a_label_s_straggler_quorum_fits_the_group_size_of_the_entry_it_labels(tmp_path):
    data = _with()
    data["train"]["workflow_config"]["straggler_policies"] = {
        "slow": {"timeout_seconds": 600, "quorum": 6},  # above slow's group of 4
        "quick": {"timeout_seconds": 600, "quorum": 6},  # within quick's group of 8
    }

    problems = _problems(tmp_path, data)

    assert problems == [
        "train.workflow_config.straggler_policies.slow.quorum: 6 is more than the 4 rollouts of a group of "
        "environments[slow], whose tasks carry the label"
    ]


def test_filter_errors_is_set_in_one_place(tmp_path):
    data = _with(workflow_config={"filter_errors": False})
    data["environments"][0]["workflow_kwargs"] = {"filter_errors": True}

    problems = _problems(tmp_path, data)

    assert problems == [
        "environments[slow].workflow_config.filter_errors: environments[quick].workflow_kwargs sets filter_errors "
        "for every entry's groups; set it in one place"
    ]


def test_a_config_class_that_skips_the_checks_is_refused_before_training(tmp_path):
    loaded = _load(tmp_path)

    class _ConfigWithoutTheChecks(PlatoonTinkerRLTrainerConfig):
        def __post_init__(self) -> None:
            pass

    config = _ConfigWithoutTheChecks(
        train=loaded.train,
        eval=loaded.eval,
        log_path="runs",
        environments=[EnvironmentConfig(name="a", rollout="x:y", max_staleness=1), EnvironmentConfig(name="b")],
    )
    config.train.max_staleness = 3

    assert any(problem.startswith("environments[a].max_staleness: 1 is below") for problem in training_problems(config))


# --- named environments ---------------------------------------------------------------------------------------------


def test_a_named_environment_ships_rollout_settings_only_as_its_entry_s_lowest_layer(tmp_path, monkeypatch):
    from platoon.envs import named
    from platoon.train.entry_settings import describe

    calibrated = named.EnvironmentEntry(
        "acme-calibrated",
        rollout="x:y",
        workflow_config={"rollout_config": {"max_steps": 30, "timeout": 600}},
    )
    monkeypatch.setattr(named, "_entries", {})
    named.register_environment(calibrated, source="test:calibrated")
    for keys, refused in (
        ({"workflow_config": {"group_size": 4}}, "workflow_config ['group_size']"),
        ({"eval_overrides": {"workflow_config": {"min_successful_group_size": 2}}}, "eval_overrides.workflow_config"),
        # Its evaluation rollout settings would win over the run's eval.workflow_config and inference.workflow.
        (
            {"eval_overrides": {"workflow_config": {"rollout_config": {"max_steps": 30}}}},
            "eval_overrides.workflow_config",
        ),
        ({"max_staleness": 9}, "['max_staleness']"),
    ):
        with pytest.raises(ValueError, match=r"acme-other@1 sets .*" + refused.replace("[", r"\[").replace("]", r"\]")):
            named.EnvironmentEntry("acme-other", rollout="x:y", **keys)

    data = {
        **{key: value for key, value in MIX.items() if key != "environments"},
        "environments": [
            {"name": "quick", "env": "acme-calibrated@1", "workflow_config": {"rollout_config": {"timeout": 60}}},
            {"name": "other", "rollout": "x:y"},
        ],
    }
    config = _load(tmp_path, data)

    quick = entry_settings.entry_settings(config)["quick"].train
    assert (quick.rollout_config.max_steps, quick.rollout_config.timeout) == (30, 60)  # the entry's wins
    lines = describe(config, "quick")
    assert any(line.startswith("  rollout_config.max_steps: 30") and "named environment" in line for line in lines)
    assert any(line.startswith("  rollout_config.timeout: 60") and " entry " in line for line in lines)


def test_a_named_environment_s_rollout_settings_rank_below_each_phase_s_section(tmp_path, monkeypatch):
    """A named environment's rollout settings are its entry's lowest layer: its training groups run them, and its
    evaluations inherit them beneath eval.workflow_config in platoon train and beneath inference.workflow in platoon
    eval, which decide for every entry that does not override them itself."""

    from platoon.envs import named

    monkeypatch.setattr(named, "_entries", {})
    named.register_environment(
        named.EnvironmentEntry("acme-calibrated", rollout="x:y", workflow_config={"rollout_config": {"max_steps": 40}}),
        source="test:calibrated",
    )
    data = {
        **{key: value for key, value in MIX.items() if key != "environments"},
        "environments": [{"name": "quick", "env": "acme-calibrated@1"}, {"name": "other", "rollout": "x:y"}],
    }
    data["eval"] = {"every": 10, "workflow_config": {"group_size": 4, "rollout_config": {"max_steps": 100}}}
    config = _load(tmp_path, data)
    quick = entry_settings.entry_settings(config)["quick"]
    assert (quick.train.rollout_config.max_steps, quick.eval.rollout_config.max_steps) == (40, 100)

    evaluation = {
        "inference": {"model_name": "m", "renderer_name": "r", "workflow": {"rollout_config": {"max_steps": 100}}},
        "environments": [{"name": "quick", "env": "acme-calibrated@1"}, {"name": "other", "rollout": "x:y"}],
    }
    loaded = resolve_config(_write(tmp_path, evaluation, "eval.yaml"), [], PlatoonEvalConfig).config
    assert entry_settings.evaluation_rollout_configs(loaded)["quick"].max_steps == 100
    evaluation["inference"]["workflow"] = {}
    loaded = resolve_config(_write(tmp_path, evaluation, "eval.yaml"), [], PlatoonEvalConfig).config
    assert entry_settings.evaluation_rollout_configs(loaded)["quick"].max_steps == 40


# --- what an entry runs with ---------------------------------------------------------------------------------------


def test_the_differences_from_the_run_are_recorded_and_summarized(tmp_path):
    config = _load(tmp_path)
    settings = entry_settings.entry_settings(config)

    record = entry_settings.lock_record(config, settings)

    assert record["quick"] == {"train": {"rollout_config.max_steps": 30}, "eval": {"rollout_config.max_steps": 30}}
    assert record["slow"]["max_staleness"] == 48
    assert record["slow"]["train"]["group_size"] == 4
    assert record["slow"]["train"]["rollout_config.extra"] == {"c": 3}
    assert record["slow"]["eval"]["group_size"] == 1
    assert entry_settings.summary(config, settings)[0] == (
        "environments[quick]: rollout_config.max_steps 30, in evaluation: rollout_config.max_steps 30"
    )


def test_warnings_name_a_group_of_one_and_a_raised_bound_without_a_limit(tmp_path):
    data = _with(workflow_config={"group_size": 1, "min_successful_group_size": 1})
    data["train"]["rollout_group_limits"] = {}

    config = _load(tmp_path, data)
    lines = entry_settings.warnings(config, entry_settings.entry_settings(config))

    assert lines[0].startswith("environments[slow].workflow_config.group_size is 1: a group of one rollout")
    assert lines[1].startswith(
        "environments[slow].max_staleness: 48 raises the run's bound (3), as a slow entry's does, but "
        "train.rollout_group_limits has no limit for it"
    )
