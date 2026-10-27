"""The forms that a release deprecated and the next minor release removed (RFC 0016, section 10): a config, an
override or a command that still uses one fails as a configuration error (exit 78), with a message that names what
replaces it and says which release removed it.

Platoon 0.4.0 removed the form that 0.3.0 deprecated:

- ``train.task_order``, now ``train.task_sampler``, which takes its values.

Platoon 0.3.0 removed the forms that 0.2.0 deprecated:

- ``rollout_config.propogate_root_success``, the misspelling of ``propagate_root_success``;
- ``train.rollout_group_key``, now ``train.label_by``;
- ``workflow_config.straggler_policy_key``: straggler policies key on ``train.label_by``'s labels;
- ``num_concurrent_rollout_workflow_workers`` of ``train`` and ``eval``, which counted groups:
  ``max_concurrent_rollouts`` counts rollouts;
- the ``strategy: none`` of ``eval`` and ``checkpoint``: ``every: 0`` (``strategy: step`` and ``epoch`` stay,
  ``tests/train/evaluation/test_event_strategy.py``);
- a Harbor entry's ``cache``, ``image_map`` and ``toolchain_root``, and ``$PLATOON_HARBOR_CACHE`` and
  ``$PLATOON_TOOLCHAINS``: the site profile's ``cache``;
- fetch config files of ``platoon harbor``, with ``platoon harbor fetch``'s ``--lock`` and ``--cache``: a training
  or evaluation config's Harbor entry;
- ``python -m platoon.train.tinker.restart_wrapper`` and ``run_with_restart``: ``watchdog.max_restarts``.
"""

from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from platoon import cli
from platoon.train.config_defs import EvalConfig, PlatoonTinkerRLTrainerConfig, TrainConfig
from platoon.utils.config import ConfigError, resolve_config

TASK_ORDER = "train.task_order: unknown key; platoon 0.4.0 removed it; use train.task_sampler"
PROPAGATION = "platoon 0.3.0 removed it; use propagate_root_success"
LABELS = "platoon 0.3.0 removed it; use train.label_by, whose labels straggler_policies key on"
TRAIN_ROLLOUTS = (
    "platoon 0.3.0 removed it; use train.max_concurrent_rollouts, which counts rollouts, not groups: this value × "
    "train.workflow_config.group_size"
)
EVAL_ROLLOUTS = (
    "platoon 0.3.0 removed it; use eval.max_concurrent_rollouts, which counts rollouts, not groups: this value × "
    "eval.workflow_config.group_size, 1 unless set"
)
HARBOR = {
    "cache": "platoon 0.3.0 removed it; use the site profile's cache, whose harbor directory is the Harbor cache",
    "image_map": (
        "platoon 0.3.0 removed it; use the default image map that platoon prepare builds, images/<provider>.json in "
        "the Harbor cache"
    ),
    "toolchain_root": (
        "platoon 0.3.0 removed it; use the site profile's cache, whose toolchains directory holds the harness "
        "toolchains"
    ),
}


def _write(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _training(tmp_path: Path, **sections: Any) -> Path:
    """A training config whose sections are merged with ``sections`` (a list, such as ``environments``, replaces)."""

    data: dict[str, Any] = {
        "train": {"model_name": "m", "renderer_name": "r"},
        "eval": {"every": 0},
        "log_path": str(tmp_path / "runs"),
    }
    for name, section in sections.items():
        data[name] = {**data[name], **section} if isinstance(data.get(name), dict) else section
    return _write(tmp_path / "configs" / "train.yaml", data)


def _exit_code(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    code = exit_info.value.code
    assert isinstance(code, int)
    return code


# --- config keys and values ------------------------------------------------------------------------------------------

REMOVED_KEYS = [
    pytest.param({"train": {"task_order": "weighted"}}, TASK_ORDER, id="train.task_order"),
    # Platoon 0.3.0 ignored task_order: auto, 0.2.0's default, beside task_sampler; 0.4.0 refuses it there too.
    pytest.param(
        {"train": {"task_order": "auto", "task_sampler": "weighted"}}, TASK_ORDER, id="train.task_order-beside-sampler"
    ),
    pytest.param(
        {"train": {"workflow_config": {"rollout_config": {"propogate_root_success": True}}}},
        f"train.workflow_config.rollout_config.propogate_root_success: unknown key; {PROPAGATION}",
        id="train.workflow_config.rollout_config.propogate_root_success",
    ),
    pytest.param(
        {"eval": {"every": 0, "workflow_config": {"rollout_config": {"propogate_root_success": False}}}},
        f"eval.workflow_config.rollout_config.propogate_root_success: unknown key; {PROPAGATION}",
        id="eval.workflow_config.rollout_config.propogate_root_success",
    ),
    pytest.param(
        {"environments": [{"name": "a", "workflow_config": {"rollout_config": {"propogate_root_success": True}}}]},
        f"environments.0.workflow_config.rollout_config.propogate_root_success: unknown key; {PROPAGATION}",
        id="environments.workflow_config.rollout_config.propogate_root_success",
    ),
    pytest.param(
        {"train": {"rollout_group_key": "suite"}},
        "train.rollout_group_key: unknown key; platoon 0.3.0 removed it; use train.label_by",
        id="train.rollout_group_key",
    ),
    pytest.param(
        {"train": {"workflow_config": {"straggler_policy_key": "suite"}}},
        f"train.workflow_config.straggler_policy_key: unknown key; {LABELS}",
        id="train.workflow_config.straggler_policy_key",
    ),
    pytest.param(
        {"eval": {"every": 0, "workflow_config": {"straggler_policy_key": "suite"}}},
        f"eval.workflow_config.straggler_policy_key: unknown key; {LABELS}",
        id="eval.workflow_config.straggler_policy_key",
    ),
    pytest.param(
        {"environments": [{"name": "a", "workflow_config": {"straggler_policy_key": "suite"}}]},
        f"environments.0.workflow_config.straggler_policy_key: unknown key; {LABELS}",
        id="environments.workflow_config.straggler_policy_key",
    ),
    pytest.param(
        {"train": {"num_concurrent_rollout_workflow_workers": 4}},
        f"train.num_concurrent_rollout_workflow_workers: unknown key; {TRAIN_ROLLOUTS}",
        id="train.num_concurrent_rollout_workflow_workers",
    ),
    pytest.param(
        {"eval": {"every": 0, "num_concurrent_rollout_workflow_workers": 4}},
        f"eval.num_concurrent_rollout_workflow_workers: unknown key; {EVAL_ROLLOUTS}",
        id="eval.num_concurrent_rollout_workflow_workers",
    ),
    pytest.param(
        {"eval": {"every": 5, "strategy": "none"}},
        "eval.strategy: 'none' is not one of 'step', 'epoch'; platoon 0.3.0 removed it; use eval.every: 0, which "
        "never evaluates",
        id="eval.strategy-none",
    ),
    pytest.param(
        {"checkpoint": {"every": 5, "strategy": "none"}},
        "checkpoint.strategy: 'none' is not one of 'step', 'epoch'; platoon 0.3.0 removed it; use checkpoint.every: "
        "0, which saves no permanent checkpoint",
        id="checkpoint.strategy-none",
    ),
]


@pytest.mark.parametrize(("sections", "problem"), REMOVED_KEYS)
def test_a_config_that_sets_a_removed_key_or_value_does_not_load_and_says_what_replaces_it(
    tmp_path, capsys, sections, problem
):
    config = _training(tmp_path, **sections)

    with pytest.raises(ConfigError) as refused:
        resolve_config(config, [], PlatoonTinkerRLTrainerConfig, warn=False)

    assert problem in refused.value.problems
    assert _exit_code(["config", "validate", "--no-resolve", str(config)]) == cli.EXIT_CONFIG
    out = capsys.readouterr().out
    assert out.startswith(f"FAIL {config}\n") and f"  {problem}\n" in out


def test_platoon_train_refuses_a_removed_key_before_it_makes_anything(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("platoon.train.launch.run_tinker_training", lambda args: pytest.fail("it trained"))
    config = _training(tmp_path, train={"rollout_group_key": "suite"})

    assert _exit_code(["train", str(config)]) == cli.EXIT_CONFIG

    err = capsys.readouterr().err
    assert f"platoon: invalid config {config}:\n" in err
    assert "  train.rollout_group_key: unknown key; platoon 0.3.0 removed it; use train.label_by\n" in err
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    ("override", "problem"),
    [
        ("train.task_order=weighted", TASK_ORDER),
        ("train.rollout_group_key=suite", "train.rollout_group_key: unknown key; platoon 0.3.0 removed it"),
        ("train.workflow_config.straggler_policy_key=suite", f"straggler_policy_key: unknown key; {LABELS}"),
        ("eval.num_concurrent_rollout_workflow_workers=2", f"unknown key; {EVAL_ROLLOUTS}"),
        ("train.workflow_config.rollout_config.propogate_root_success=true", f"unknown key; {PROPAGATION}"),
        ("eval={every: 3, strategy: none}", "'none' is not one of 'step', 'epoch'; platoon 0.3.0 removed it; use"),
        # As an override writes it, none is the removed value, not a null (eval.strategy=null leaves it unset).
        (
            "eval.strategy=none",
            "override eval.strategy=none: eval.strategy: 'none' is not one of 'step', 'epoch'; platoon 0.3.0 removed "
            "it; use eval.every: 0, which never evaluates",
        ),
        (
            "checkpoint.strategy=none",
            "override checkpoint.strategy=none: checkpoint.strategy: 'none' is not one of 'step', 'epoch'; platoon "
            "0.3.0 removed it; use checkpoint.every: 0, which saves no permanent checkpoint",
        ),
    ],
)
def test_an_override_of_a_removed_key_or_value_is_refused_alike(tmp_path, override, problem):
    with pytest.raises(ConfigError) as refused:
        resolve_config(_training(tmp_path), [override], PlatoonTinkerRLTrainerConfig, warn=False)

    assert problem in str(refused.value)


def test_an_evaluation_config_refuses_the_removed_key_of_its_rollouts(tmp_path):
    from platoon.eval.run import PlatoonEvalConfig

    workflow = {"rollout_config": {"propogate_root_success": True}}
    config = _write(tmp_path / "eval.yaml", {"inference": {"model_name": "m", "workflow": workflow}})

    with pytest.raises(ConfigError) as refused:
        resolve_config(config, [], PlatoonEvalConfig, warn=False)

    assert refused.value.problems == [
        f"inference.workflow.rollout_config.propogate_root_success: unknown key; {PROPAGATION}"
    ]


def test_the_config_classes_take_no_removed_key_from_python_either():
    """And ``strategy: none`` built in Python is refused too (``tests/train/evaluation/test_event_strategy.py``)."""

    with pytest.raises(TypeError, match="task_order"):
        TrainConfig(model_name="m", renderer_name="r", task_order="weighted")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="rollout_group_key"):
        TrainConfig(model_name="m", renderer_name="r", rollout_group_key="suite")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="num_concurrent_rollout_workflow_workers"):
        EvalConfig(num_concurrent_rollout_workflow_workers=4)  # type: ignore[call-arg]


# --- a Harbor entry's options, and the cache variables ---------------------------------------------------------------


def _harbor(tmp_path: Path, **service_kwargs: Any) -> Path:
    """A training config with one Harbor entry on a fetched local dataset of no task, with ``service_kwargs``."""

    (tmp_path / "dataset").mkdir(exist_ok=True)
    entry = {
        "service": "harbor",
        "service_kwargs": {"dataset": "../dataset", **service_kwargs},
        "harness": "platoon-bash",
    }
    return _training(tmp_path, environments=[entry])


@pytest.mark.parametrize(
    ("option", "value"), [("cache", "cache"), ("image_map", "images.json"), ("toolchain_root", "toolchains")]
)
def test_a_harbor_entry_that_sets_a_removed_option_is_refused_by_each_command_that_reads_it(
    tmp_path, monkeypatch, capsys, option, value
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("platoon.train.launch.run_tinker_training", lambda args: pytest.fail("it trained"))
    config = _harbor(tmp_path, **{option: value})
    problem = f"environments.0.service_kwargs.{option}: unknown key; {HARBOR[option]}"

    with pytest.raises(ConfigError) as refused:  # the config does not load, before any component resolves
        resolve_config(config, [], PlatoonTinkerRLTrainerConfig, warn=False)
    assert refused.value.problems == [problem]
    for validate in (["config", "validate"], ["config", "validate", "--no-resolve"]):
        assert _exit_code([*validate, str(config)]) == cli.EXIT_CONFIG, validate
        assert f"  {problem}\n" in capsys.readouterr().out, validate
    for command in (
        ["config", "show"],
        ["train"],
        ["harbor", "fetch"],
        ["harbor", "verify"],
        ["harbor", "images", "plan"],
        ["prepare"],
    ):
        assert _exit_code([*command, str(config)]) == cli.EXIT_CONFIG, command
        assert problem in capsys.readouterr().err, command
    assert not (tmp_path / "cache").exists() and not (tmp_path / "toolchains").exists()


def test_an_override_that_gives_a_harbor_entry_a_removed_option_is_refused_alike(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigError) as refused:
        resolve_config(
            _harbor(tmp_path), ["environments.0.service_kwargs.image_map=images.json"], PlatoonTinkerRLTrainerConfig
        )

    assert refused.value.problems == [f"environments.0.service_kwargs.image_map: unknown key; {HARBOR['image_map']}"]


@pytest.mark.parametrize(
    ("variable", "directory"), [("PLATOON_HARBOR_CACHE", "harbor"), ("PLATOON_TOOLCHAINS", "toolchains")]
)
def test_a_cache_variable_is_refused_by_each_command_that_takes_the_cache_from_the_site(
    tmp_path, monkeypatch, capsys, variable, directory
):
    monkeypatch.setenv(variable, str(tmp_path / "old"))
    config = _training(tmp_path)
    problem = (
        f"${variable}: platoon 0.3.0 removed it; use the site profile's cache, whose {directory} directory it named "
        f"(~/.config/platoon/site.yaml, or the file $PLATOON_SITE names), and unset ${variable}"
    )

    with pytest.raises(ConfigError) as refused:
        resolve_config(config, [], PlatoonTinkerRLTrainerConfig, warn=False)
    assert refused.value.problems == [problem]

    assert _exit_code(["config", "validate", str(config)]) == cli.EXIT_CONFIG
    assert f"  {problem}\n" in capsys.readouterr().out
    assert _exit_code(["harbor", "verify", str(tmp_path / "dataset.lock.json")]) == cli.EXIT_CONFIG  # the site's cache
    assert problem in capsys.readouterr().err
    assert _exit_code(["toolchain", "build", "openhands"]) == cli.EXIT_CONFIG  # the site's toolchain root
    assert problem in capsys.readouterr().err
    assert _exit_code(["harbor", "verify", str(tmp_path / "dataset.lock.json"), "--cache", str(tmp_path)]) == 78
    assert "does not exist" in capsys.readouterr().err and not (tmp_path / "old").exists()  # its own cache: no check


# --- fetch config files ------------------------------------------------------------------------------------------


def test_a_fetch_config_file_is_refused_with_the_entry_that_replaces_it(tmp_path, capsys):
    fetch_config = _write(tmp_path / "configs" / "fetch.yaml", {"dataset": "../dataset", "lock": "../dataset.lock"})
    message = (
        f"{fetch_config}: a fetch config file; platoon 0.3.0 removed it; use a training or evaluation config, whose "
        "Harbor entry names the dataset and its fetch settings in service_kwargs (dataset, lock, tasks, select, "
        "overlays, refuse, index), and run this command on it"
    )

    for command in (
        ["harbor", "fetch"],
        ["harbor", "verify"],
        ["harbor", "images", "plan"],
        ["harbor", "images", "build"],
        ["harbor", "images", "verify"],
    ):
        assert _exit_code([*command, str(fetch_config)]) == cli.EXIT_CONFIG, command
        assert f"platoon: invalid config:\n  {message}\n" in capsys.readouterr().err, command
    assert _exit_code(["harbor", "images", "alias", str(fetch_config), "--image", "local"]) == cli.EXIT_CONFIG
    assert message in capsys.readouterr().err
    assert not (tmp_path / "dataset.lock").exists()


@pytest.mark.parametrize(
    ("flag", "replacement"),
    [
        ("--lock", "use service_kwargs.lock in a Harbor entry of a training or evaluation config"),
        ("--cache", "use the site profile's cache, whose harbor directory is the Harbor cache"),
    ],
)
def test_the_fetch_options_of_fetch_config_files_are_refused_with_what_replaces_them(
    tmp_path, capsys, flag, replacement
):
    message = (
        f"platoon harbor fetch {flag}, which applied to fetch config files alone: platoon 0.3.0 removed it; "
        f"{replacement}"
    )
    fetch_config = _write(tmp_path / "configs" / "fetch.yaml", {"dataset": "../dataset"})

    for config in (_harbor(tmp_path), fetch_config):
        assert _exit_code(["harbor", "fetch", str(config), flag, str(tmp_path / "given")]) == cli.EXIT_CONFIG
        assert f"platoon: invalid config:\n  {message}\n" in capsys.readouterr().err
    assert not (tmp_path / "given").exists()

    cli.main(["harbor", "fetch", "--help"])  # and the help does not offer them
    help_text = capsys.readouterr().out
    assert "--workers" in help_text and flag not in help_text


# --- the restart wrapper -----------------------------------------------------------------------------------------

RESTART_WRAPPER = (
    "python -m platoon.train.tinker.restart_wrapper (run_with_restart): platoon 0.3.0 removed it; use "
    "watchdog.max_restarts in the training config, with which platoon train restarts training after any failure"
)


def test_the_restart_wrapper_exits_78_saying_what_replaces_it(tmp_path):
    command = [sys.executable, "-m", "platoon.train.tinker.restart_wrapper", "--max-restarts", "5", "--", "true"]
    environment = {**os.environ, "HF_HUB_OFFLINE": "1"}

    completed = subprocess.run(command, capture_output=True, text=True, timeout=300, cwd=tmp_path, env=environment)

    assert completed.returncode == 78, completed.stderr
    assert f"platoon: {RESTART_WRAPPER}\n" in completed.stderr


def test_importing_the_restart_wrapper_says_what_replaces_it():
    with pytest.raises(ImportError, match=re.escape(RESTART_WRAPPER)):
        importlib.import_module("platoon.train.tinker.restart_wrapper")
    with pytest.raises(ImportError, match="run_with_restart"):
        from platoon.train.tinker import run_with_restart  # noqa: F401
