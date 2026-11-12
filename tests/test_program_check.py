"""An episode program's load-time check (``@pt.program(check=...)``, RFC 0016, section 3): it runs when a config
loads, for each entry that names the program and each phase the command runs, sees the entry as that phase runs it,
and refuses an entry before anything runs, naming the entry and the phase."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

import platoon as pt
from platoon import cli
from platoon.eval.run import PlatoonEvalConfig
from platoon.extensions import run_components
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.utils.config import ConfigError, load_config

TRAIN = PlatoonTinkerRLTrainerConfig
EVAL = PlatoonEvalConfig
ROWS = {"rows": [{"id": "t1", "goal": "say hi"}], "splits": {"train": ["t1"], "eval": ["t1"]}}
REFUSAL = "it hands the task to a subagent, so it needs agents.max_depth of at least 1"
CHECKED: list[tuple[Mapping[str, Any], pt.CheckContext]] = []


def deep_enough(options: Mapping[str, Any], ctx: pt.CheckContext) -> None:
    CHECKED.append((options, ctx))
    if ctx.environment.agents.max_depth == 0:
        raise ValueError(REFUSAL)


@pt.program("checktest-delegating", check=deep_enough)
async def delegating(ep: pt.EpisodeContext) -> pt.Outcome:
    await ep.run()
    return await ep.grade()


def undecided(options: Mapping[str, Any], ctx: pt.CheckContext) -> None:
    raise KeyError("rubric")


@pt.program("checktest-undecided", check=undecided)
async def undecided_program(ep: pt.EpisodeContext) -> pt.Outcome:
    return await ep.grade()


def meddling(options: Mapping[str, Any], ctx: pt.CheckContext) -> None:
    assert ctx.rollout is not None
    ctx.rollout.max_steps = 99
    ctx.environment.harness_options["tries"] = 99


@pt.program("checktest-meddling", check=meddling)
async def meddling_program(ep: pt.EpisodeContext) -> pt.Outcome:
    return await ep.grade()


@pytest.fixture(autouse=True)
def fresh_checks(monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")  # train serves a policy: past its key check
    CHECKED.clear()


def train_config(tmp_path: Path, *, evaluates: bool = False, **entry: Any) -> Path:
    path = tmp_path / "train.yaml"
    environment = {"service": "table", "service_kwargs": ROWS, "harness": "platoon-noop", **entry}
    config = {
        "train": {"model_name": "Qwen/Qwen3-4B", "renderer_name": "qwen3", "batch_size": 1, "max_training_steps": 1},
        "eval": {"every": 1 if evaluates else 0},
        "log_path": str(tmp_path / "logs"),
        "stats": {"experiment_name": "exp", "trial_name": "trial"},
        "environments": [environment],
    }
    path.write_text(yaml.safe_dump(config))
    return path


def eval_config(tmp_path: Path, **entry: Any) -> Path:
    path = tmp_path / "eval.yaml"
    environment = {"service": "table", "service_kwargs": ROWS, "harness": "platoon-noop", **entry}
    inference = {"model_name": "Qwen/Qwen3-4B", "renderer_name": "qwen3", "output_dir": str(tmp_path / "out")}
    path.write_text(yaml.safe_dump({"inference": inference, "environments": [environment]}))
    return path


def _exit_code(argv: list[str]) -> int | str | None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    return exit_info.value.code


@pytest.mark.parametrize("command", ["train", "config validate"])
def test_a_program_check_refuses_an_entry_at_load(tmp_path, capsys, monkeypatch, command):
    from platoon.train import supervise

    def never(*args: Any, **kwargs: Any) -> int:
        raise AssertionError("training started")

    monkeypatch.setattr(supervise, "run_training", never)
    path = train_config(tmp_path, program="checktest-delegating", agents={"max_depth": 0})

    assert _exit_code([*command.split(), str(path)]) == cli.EXIT_CONFIG

    captured = capsys.readouterr()
    assert f"environments.0.program (train): {REFUSAL}" in captured.out + captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "logs").exists()  # nothing of a run was written


def test_a_check_sees_the_entry_as_each_phase_runs_it(tmp_path):
    path = train_config(
        tmp_path,
        evaluates=True,
        program="checktest-delegating",
        agents={"max_depth": 1},
        harness_options={"tries": 1},
        workflow_config={"rollout_config": {"max_steps": 7}},
        eval_overrides={"harness_options": {"tries": 2}, "workflow_config": {"rollout_config": {"max_steps": 9}}},
    )
    config, _ = load_config(["--config", str(path)], TRAIN)

    run_components(config)

    seen = [
        (dict(options), ctx.phase, ctx.environment.harness_options, ctx.rollout.max_steps) for options, ctx in CHECKED
    ]
    assert seen == [({}, "train", {"tries": 1}, 7), ({}, "eval", {"tries": 2}, 9)]
    assert config.environments[0].harness_options == {"tries": 1}  # each check saw a copy


def test_a_check_that_cannot_decide_refuses(tmp_path):
    config, _ = load_config(["--config", str(train_config(tmp_path, program="checktest-undecided"))], TRAIN)

    with pytest.raises(ConfigError) as error:
        run_components(config)

    assert error.value.problems == ["environments.0.program (train): its check raised KeyError: 'rubric'"]


@pytest.mark.parametrize("command", [["eval"], ["rollout", "--task", "t1"]], ids=["eval", "rollout"])
def test_eval_and_rollout_refuse_an_entry_a_program_check_refuses(tmp_path, capsys, monkeypatch, command):
    monkeypatch.chdir(tmp_path)  # where platoon rollout would write its run
    path = eval_config(tmp_path, program="checktest-delegating", agents={"max_depth": 0})

    assert _exit_code([command[0], str(path), *command[1:]]) == cli.EXIT_CONFIG

    captured = capsys.readouterr()
    assert f"environments.0.program (eval): {REFUSAL}" in captured.out + captured.err
    assert "Traceback" not in captured.err
    assert sorted(tmp_path.iterdir()) == [path]  # nothing of a run was written


@pytest.mark.parametrize("kind", ["train", "eval"])
def test_a_check_cannot_change_the_config(tmp_path, kind):
    entry: dict[str, Any] = {"program": "checktest-meddling", "harness_options": {"tries": 1}}
    if kind == "train":
        config, _ = load_config(["--config", str(train_config(tmp_path, **entry))], TRAIN)
        rollout = config.train.workflow_config.rollout_config
    else:
        config, _ = load_config(["--config", str(eval_config(tmp_path, **entry))], EVAL)
        rollout = config.inference.workflow.rollout_config
    max_steps = rollout.max_steps

    run_components(config)

    assert rollout.max_steps == max_steps and max_steps != 99
    assert config.environments[0].harness_options == {"tries": 1}
