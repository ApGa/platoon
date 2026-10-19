"""The ``platoon`` console script: argument parsing and output of every subcommand, with the run entry points
monkeypatched (no GPU, no Tinker)."""

from __future__ import annotations

import json
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

from platoon import cli
from platoon.deprecations import PlatoonDeprecationWarning
from platoon.harness import registry
from platoon.task import Task

ROLLOUT_CONFIGS: list = []


@dataclass(frozen=True)
class FakePolicy:
    """Stands in for the evaluation package's ``EvalPolicy``, whose ``install`` would start a Tinker sampler."""

    model_name: str
    renderer_name: str
    model_path: str | None = None
    renderer_kwargs: dict = field(default_factory=dict)
    context_window_length: int | None = None

    def install(self) -> None:
        INSTALLED_POLICIES.append(self)


INSTALLED_POLICIES: list[FakePolicy] = []


@pytest.fixture(autouse=True)
def tinker_key(monkeypatch):
    """``platoon train`` checks its Tinker key before anything else; the runs here are monkeypatched past it."""

    monkeypatch.setenv("TINKER_API_KEY", "tml-test")


@pytest.fixture
def policies(monkeypatch) -> list[FakePolicy]:
    INSTALLED_POLICIES.clear()
    ROLLOUT_CONFIGS.clear()
    monkeypatch.setattr(sys.modules[cli._evaluation().__package__], "EvalPolicy", FakePolicy, raising=False)
    return INSTALLED_POLICIES


def load_tasks(config, split, **kwargs) -> list[str]:
    return ["solved", "unsolved"]


def load_task(task_id: str) -> Task:
    return Task(goal=f"solve {task_id}", id=task_id)


async def answer(task: Task, config) -> dict:
    return {
        "trajectories": {
            "root": {
                "id": "root",
                "task": {"id": task.id},
                "parent_info": None,
                "reward": float(task.id == "solved"),
                "steps": [{}],
            }
        }
    }


async def tree(task: Task, config) -> dict:
    """A root that delegated once, as the gateway trace serializes it."""

    ROLLOUT_CONFIGS.append(config)
    return {
        "id": "rollout-1",
        "trajectories": {
            "root-trajectory": {
                "id": "root-trajectory",
                "task": {"id": task.id, "goal": task.goal},
                "parent_info": None,
                "reward": 1.0,
                "steps": [{}, {}, {}],
                "misc": {
                    "agent": {"agent_id": "n0", "role": "root"},
                    "outcome": {"status": "scored", "rewards": {"reward": 1.0}, "reason": None},
                },
            },
            "child-trajectory": {
                "id": "child-trajectory",
                "task": {"id": task.id, "goal": "craft the planks"},
                "parent_info": {"id": "root-trajectory", "fork_step": 1},
                "reward": 0.5,
                "steps": [{}],
                "misc": {"agent": {"agent_id": "n1", "role": "subagent"}},
            },
        },
    }


class MyWorkflow:
    """A training workflow of the config's own, which writes its rollouts where it says."""


def _environment(rollout: str = "answer") -> dict:
    return {
        "dataset_loader": f"{__name__}:load_tasks",
        "task_loader": f"{__name__}:load_task",
        "rollout": f"{__name__}:{rollout}",
    }


def _eval_config(tmp_path: Path, rollout: str = "answer", **inference) -> Path:
    path = tmp_path / "eval.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "inference": {"model_name": "fake-model", "output_dir": str(tmp_path / "results"), **inference},
                "environments": [_environment(rollout)],
            }
        )
    )
    return path


def _train_config(tmp_path: Path, **extra) -> Path:
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "train": {"model_name": "Qwen/Qwen3-4B", "renderer_name": "qwen3", "batch_size": 16},
                "eval": {"every": 0},
                "log_path": str(tmp_path / "logs"),
                "stats": {"experiment_name": "exp", "trial_name": "trial"},
                "environments": [_environment("tree")],
                **extra,
            }
        )
    )
    return path


def _exit_code(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    return exit_info.value.code


# --- train and eval ----------------------------------------------------------------------------------------------


def test_train_snapshots_the_resolved_config_into_the_run_dir_and_trains_from_it(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("platoon.train.tinker.train.run_tinker_training", calls.append)
    config = _train_config(tmp_path)

    cli.main(["train", str(config), "train.batch_size=64", "stats.trial_name=7"])

    run_dir = tmp_path / "logs/exp/7"
    assert calls == [["--config", str(run_dir / "config.resolved.yaml")]]
    snapshot = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    assert snapshot["train"]["batch_size"] == 64
    assert snapshot["stats"]["trial_name"] == "7"
    lock = json.loads((run_dir / "lock.json").read_text())
    assert lock["argv"] == ["platoon", "train", str(config), "train.batch_size=64", "stats.trial_name=7"]


def test_a_config_error_exits_78_with_a_suggestion_before_training(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr("platoon.train.tinker.train.run_tinker_training", calls.append)

    assert _exit_code(["train", str(_train_config(tmp_path)), "train.bach_size=64"]) == cli.EXIT_CONFIG

    assert "train.bach_size: unknown key; did you mean train.batch_size?" in capsys.readouterr().err
    assert calls == []


def test_a_trainer_config_that_does_not_resolve_exits_78_before_training(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr("platoon.train.tinker.train.run_tinker_training", calls.append)
    entry = {**_environment("tree"), "trainer_config": "no_such_plugin.config:TrainerConfig"}
    config = _train_config(tmp_path, environments=[entry])
    problem = (
        "environments.0.trainer_config: 'no_such_plugin.config:TrainerConfig' does not resolve (ModuleNotFoundError: "
        "No module named 'no_such_plugin')"
    )

    assert _exit_code(["train", str(config)]) == cli.EXIT_CONFIG
    assert problem in capsys.readouterr().err
    assert calls == []
    assert _exit_code(["config", "validate", str(config)]) == cli.EXIT_CONFIG
    assert f"FAIL {config}\n  {problem}" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["eval", "run.yaml"], ["train", "missing.yaml", "a=1"]])
def test_a_missing_config_file_is_a_config_error(argv, capsys):
    assert _exit_code(argv) == cli.EXIT_CONFIG
    assert "no such config file" in capsys.readouterr().err


def test_an_override_without_a_value_is_a_config_error(tmp_path, capsys):
    assert _exit_code(["eval", str(_eval_config(tmp_path)), "inference.model_name"]) == cli.EXIT_CONFIG
    assert "not of the form key=value" in capsys.readouterr().err


def test_eval_collects_rollouts_and_prints_the_report_summary(tmp_path, capsys):
    cli.main(["eval", str(_eval_config(tmp_path)), "inference.workflow.num_rollouts_per_task=2"])

    summary = json.loads(capsys.readouterr().out)
    assert summary["total_tasks"] == 2
    assert summary["total_rollouts"] == 4
    assert summary["success_rate"] == 0.5
    results = tmp_path / "results"
    assert (results / "reports").is_dir()
    snapshot = yaml.safe_load((results / "config.resolved.yaml").read_text())
    assert snapshot["inference"]["workflow"]["num_rollouts_per_task"] == 2
    assert json.loads((results / "lock.json").read_text())["argv"][:2] == ["platoon", "eval"]


IGNORED = "{} is ignored: platoon {} writes rollouts under its run directory"
TRAIN_OUTPUT_DIR = "train.workflow_config.rollout_config.output_dir"
EVAL_OUTPUT_DIR = "eval.workflow_config.rollout_config.output_dir"


def test_train_warns_that_it_replaces_a_configured_rollout_output_dir(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("platoon.train.tinker.train.run_tinker_training", lambda args: None)
    config = _train_config(tmp_path)

    cli.main(["train", str(config), f"{TRAIN_OUTPUT_DIR}=/elsewhere", f"{EVAL_OUTPUT_DIR}=/elsewhere"])

    err = capsys.readouterr().err
    assert IGNORED.format(TRAIN_OUTPUT_DIR, "train") in err and IGNORED.format(EVAL_OUTPUT_DIR, "train") in err
    cli.main(["train", str(config)])
    assert "is ignored" not in capsys.readouterr().err


def test_train_with_its_own_workflow_class_leaves_the_rollout_output_dir_to_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("platoon.train.tinker.train.run_tinker_training", lambda args: None)
    environment = {**_environment("tree"), "workflow": f"{__name__}:MyWorkflow"}
    config = _train_config(tmp_path, environments=[environment])

    cli.main(["train", str(config), f"{TRAIN_OUTPUT_DIR}=/elsewhere"])

    assert "is ignored" not in capsys.readouterr().err


def test_eval_warns_that_it_replaces_a_configured_rollout_output_dir(tmp_path, capsys):
    key = "inference.workflow.rollout_config.output_dir"

    cli.main(["eval", str(_eval_config(tmp_path)), f"{key}=/elsewhere"])

    assert IGNORED.format(key, "eval") in capsys.readouterr().err
    assert not Path("/elsewhere").exists()


def test_infer_is_not_a_command():
    assert _exit_code(["infer", "run.yaml"]) == cli.EXIT_USAGE


def test_eval_help_says_what_it_does(capsys):
    cli.main(["eval", "--help"])

    out = " ".join(capsys.readouterr().out.split())
    assert (
        "Evaluate a policy on an environment split: run every task, grade it, write trajectories and a report." in out
    )


@pytest.mark.parametrize("command", ["train", "eval", "rollout"])
def test_the_runs_take_prepare_and_their_help_says_where_to_use_it(command, capsys):
    cli.main([command, "--help"])

    out = " ".join(re.sub(r"[│╭╮╰╯─]", " ", capsys.readouterr().out).split())
    assert "--prepare" in out and "First run platoon prepare on the config and overrides, with its lines on" in out
    assert "Use it on a laptop or a single node. On a cluster, prepare once on a CPU node instead: in a GPU" in out


# --- rollout -----------------------------------------------------------------------------------------------------


def test_rollout_runs_one_task_and_prints_its_tree(tmp_path, capsys, policies):
    run_dir = tmp_path / "debug"
    config = _eval_config(tmp_path, "tree")

    cli.main(["rollout", str(config), "--task", "solved", "--output-dir", str(run_dir), "--renderer", "qwen3"])

    out = capsys.readouterr().out
    assert "rollout rollout-1  task solved" in out
    assert "agents 2  steps 4  max depth 1  reward 1.0  outcome scored reward=1.0" in out
    assert "└─ root root-tra  steps 3  reward 1.0  outcome scored reward=1.0" in out
    assert "   └─ subagent child-tr  steps 1  reward 0.5  goal 'craft the planks'" in out
    (rollout_config,) = ROLLOUT_CONFIGS
    assert (rollout_config.model_name, rollout_config.train, rollout_config.output_dir) == (
        "fake-model",
        False,
        str(run_dir / "rollouts"),
    )
    assert policies == [FakePolicy(model_name="fake-model", renderer_name="qwen3")]
    for name in ("config.resolved.yaml", "lock.json", "trajectory_collection.json"):
        assert (run_dir / name).is_file()


def test_rollout_writes_the_event_file_the_trajectory_viewer_reads(tmp_path, capsys, policies):
    run_dir = tmp_path / "debug"

    cli.main(["rollout", str(_eval_config(tmp_path, "tree")), "--task", "solved", "--output-dir", str(run_dir)])

    events = run_dir / "rollouts" / "events" / "events_solved_rollout-1.jsonl"
    records = [json.loads(line) for line in events.read_text().splitlines()]
    assert records and {record["process_id"] for record in records} == {os.getpid()}


def test_rollout_warns_that_it_replaces_a_configured_rollout_output_dir(tmp_path, capsys, policies):
    run_dir = tmp_path / "debug"
    config = _train_config(tmp_path)

    cli.main(
        ["rollout", str(config), f"{TRAIN_OUTPUT_DIR}=/elsewhere", "--task", "solved", "--output-dir", str(run_dir)]
    )

    assert IGNORED.format(TRAIN_OUTPUT_DIR, "rollout") in capsys.readouterr().err
    (rollout_config,) = ROLLOUT_CONFIGS
    assert rollout_config.output_dir == str(run_dir / "rollouts")


def test_rollout_serves_no_policy_when_no_renderer_is_configured(tmp_path, capsys, policies):
    cli.main(["rollout", str(_eval_config(tmp_path, "tree")), "--task", "solved", "--output-dir", str(tmp_path)])

    assert policies == []
    assert len(ROLLOUT_CONFIGS) == 1


def test_rollout_gives_an_evaluation_configs_policy_settings_the_renderer_flag(tmp_path, capsys, policies):
    config = _eval_config(tmp_path, "tree", context_window_length=4096)

    cli.main(["rollout", str(config), "--task", "solved", "--output-dir", str(tmp_path / "a"), "--renderer", "qwen3"])

    assert policies == [FakePolicy(model_name="fake-model", renderer_name="qwen3", context_window_length=4096)]
    # Without the flag the setting would be dropped: refused before anything runs.
    unrendered = ["rollout", str(config), "--task", "solved", "--output-dir", str(tmp_path / "b")]
    assert _exit_code(unrendered) == cli.EXIT_CONFIG
    assert "context_window_length configure the policy" in capsys.readouterr().err
    assert len(ROLLOUT_CONFIGS) == 1 and not (tmp_path / "b").exists()


def test_rollout_of_a_training_config_serves_the_trained_model_or_the_flags(tmp_path, capsys, policies):
    config = _train_config(tmp_path)

    cli.main(
        [
            "rollout",
            str(config),
            "train.workflow_config.rollout_config.max_steps=5",
            "--task",
            "unsolved",
            "--model",
            "Qwen/Qwen3-8B",
            "--model-path",
            "tinker://run-1/sampler_weights/000010",
            "--json-output",
            "--output-dir",
            str(tmp_path / "debug"),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert [(agent["role"], agent["depth"], agent["steps"]) for agent in summary["agents"]] == [
        ("root", 0, 3),
        ("subagent", 1, 1),
    ]
    assert summary["task"] == "unsolved"
    (rollout_config,) = ROLLOUT_CONFIGS
    assert (rollout_config.model_name, rollout_config.max_steps) == ("Qwen/Qwen3-8B", 5)
    assert policies == [
        FakePolicy(
            model_name="Qwen/Qwen3-8B", renderer_name="qwen3", model_path="tinker://run-1/sampler_weights/000010"
        )
    ]


def test_rollout_needs_a_task(tmp_path):
    assert _exit_code(["rollout", str(_eval_config(tmp_path))]) == cli.EXIT_USAGE


def test_rollout_needs_a_training_or_evaluation_config(tmp_path, capsys):
    config = tmp_path / "bare.yaml"
    config.write_text(yaml.safe_dump({"environments": [_environment("tree")]}))

    assert _exit_code(["rollout", str(config), "--task", "solved"]) == cli.EXIT_CONFIG
    assert "rollout needs a training config (train:) or an evaluation config (inference:)" in capsys.readouterr().err


# --- config ------------------------------------------------------------------------------------------------------


def test_config_show_prints_the_merged_config_after_overrides(tmp_path, capsys):
    base = _train_config(tmp_path)
    run = tmp_path / "run.yaml"
    run.write_text(yaml.safe_dump({"extends": base.name, "train": {"batch_size": 32}}))

    cli.main(["config", "show", str(run), "train.num_epochs=3"])

    shown = yaml.safe_load(capsys.readouterr().out)
    assert shown["train"] == {
        "model_name": "Qwen/Qwen3-4B",
        "renderer_name": "qwen3",
        "batch_size": 32,
        "num_epochs": 3,
    }
    assert "extends" not in shown


def test_config_show_with_defaults_prints_every_field(tmp_path, capsys):
    cli.main(["config", "show", str(_train_config(tmp_path)), "--defaults"])

    shown = yaml.safe_load(capsys.readouterr().out)
    assert shown["train"]["workflow_config"]["group_size"] == 8
    assert shown["checkpoint"]["every"] == 0
    # But a deprecated key left unset: a config copied from the output would set a key that is going away.
    rollout_config = shown["train"]["workflow_config"]["rollout_config"]
    assert rollout_config["propagate_root_success"] is False and "propogate_root_success" not in rollout_config
    assert "strategy" not in shown["checkpoint"] and "strategy" not in shown["eval"]


def test_config_validate_reports_each_file_and_exits_78_when_any_fails(tmp_path, capsys):
    good = _train_config(tmp_path)
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump({"extends": good.name, "train": {"task_sampler": "random"}, "stats": {"trail_name": "x"}})
    )

    assert _exit_code(["config", "validate", str(good), str(bad)]) == cli.EXIT_CONFIG

    out = capsys.readouterr().out
    assert f"ok   {good} (PlatoonTinkerRLTrainerConfig)" in out
    assert f"FAIL {bad}" in out
    assert "  train.task_sampler: 'random' is not one of 'auto', 'weighted'" in out
    assert "  stats.trail_name: unknown key; did you mean stats.trial_name?" in out


def test_config_validate_takes_an_explicit_schema(tmp_path, capsys):
    path = tmp_path / "sampling.yaml"
    path.write_text("temperature: 0.5\n")

    cli.main(["config", "validate", str(path), "--schema", "platoon.config_defs:InferenceParams"])
    assert capsys.readouterr().out == f"ok   {path} (InferenceParams)\n"
    assert _exit_code(["config", "validate", str(path), "--schema", "platoon.config_defs:RolloutConfig"]) == 78
    assert "  temperature: unknown key" in capsys.readouterr().out


def test_config_validate_needs_a_file():
    assert _exit_code(["config", "validate"]) == cli.EXIT_USAGE
    assert _exit_code(["config", "validate", "log_path=runs"]) == cli.EXIT_USAGE  # overrides alone name no file


def test_config_validate_takes_overrides_as_the_run_does(tmp_path, capsys):
    """A config that leaves log_path (and, say, a provider) to its launcher is checked as it runs: with the overrides,
    which apply to each file named."""

    config = _train_config(tmp_path)
    data = yaml.safe_load(config.read_text())
    del data["log_path"]
    config.write_text(yaml.safe_dump(data))

    assert _exit_code(["config", "validate", str(config)]) == cli.EXIT_CONFIG
    assert "  log_path: required key is missing" in capsys.readouterr().out
    cli.main(["config", "validate", str(config), f"log_path={tmp_path / 'runs'}", "train.batch_size=8"])
    assert capsys.readouterr().out == f"ok   {config} (PlatoonTinkerRLTrainerConfig)\n"
    assert _exit_code(["config", "validate", str(config), "log_path=runs", "train.batch_size=many"]) == 78
    assert "train.batch_size" in capsys.readouterr().out


def test_config_validate_checks_a_file_of_environments_against_the_entry_schema(tmp_path, capsys):
    """A file that holds only environments, which training and evaluation configs extend, is checked against the
    entry schema rather than passed as having no schema."""

    good = tmp_path / "environments.yaml"
    good.write_text(yaml.safe_dump({"environments": [_environment()]}))
    bad = tmp_path / "typo.yaml"
    bad.write_text(yaml.safe_dump({"environments": [{**_environment(), "harnes": "platoon-bash"}]}))

    assert _exit_code(["config", "validate", str(good), str(bad)]) == cli.EXIT_CONFIG

    out = capsys.readouterr().out
    assert f"ok   {good} (EnvironmentsConfig)" in out
    assert f"FAIL {bad}" in out and "  environments.0.harnes: unknown key; did you mean environments.0.harness?" in out


# --- deprecated keys ---------------------------------------------------------------------------------------------

LEGACY = "train.workflow_config.rollout_config.propogate_root_success"
LEGACY_LINE = (
    f"{LEGACY}: deprecated; use train.workflow_config.rollout_config.propagate_root_success instead "
    "(platoon 0.3.0 removes it)"
)


def _with_legacy_key(path: Path, section: tuple[str, ...]) -> Path:
    """The config at ``path`` with ``propogate_root_success: true`` in the rollout config under ``section``."""

    data = yaml.safe_load(path.read_text())
    node = data
    for key in section:
        node = node.setdefault(key, {})
    node["rollout_config"] = {"propogate_root_success": True}
    path.write_text(yaml.safe_dump(data))
    return path


def _platoon_deprecations(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(warning.message) for warning in caught if issubclass(warning.category, PlatoonDeprecationWarning)]


def test_config_validate_lists_a_deprecated_key_once_and_fails_on_it_with_strict(tmp_path, capsys):
    config = _with_legacy_key(_train_config(tmp_path), ("train", "workflow_config"))

    cli.main(["config", "validate", str(config)])

    captured = capsys.readouterr()
    assert captured.out == f"ok   {config} (PlatoonTinkerRLTrainerConfig)\n  {LEGACY_LINE}\n"
    assert captured.err == ""  # listed under the config, not warned of again
    assert _exit_code(["config", "validate", "--strict", str(config)]) == cli.EXIT_CONFIG
    assert capsys.readouterr().out == f"FAIL {config}\n  {LEGACY_LINE}\n"
    current = tmp_path / "current"
    current.mkdir()
    cli.main(["config", "validate", "--strict", str(_train_config(current))])  # nothing deprecated: passes
    assert capsys.readouterr().out.startswith("ok   ")


def test_train_names_a_deprecated_key_once_and_records_it_in_the_lock(tmp_path, monkeypatch, capsys):
    from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
    from platoon.utils.config import load_config

    loaded = []

    def run_tinker_training(args):  # training loads its snapshot of the config again
        loaded.append(load_config(args, PlatoonTinkerRLTrainerConfig)[0])

    monkeypatch.setattr("platoon.train.tinker.train.run_tinker_training", run_tinker_training)
    config = _with_legacy_key(_train_config(tmp_path), ("train", "workflow_config"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cli.main(["train", str(config)])

    err = capsys.readouterr().err
    assert f"platoon: {LEGACY_LINE}\n" in err and err.count("propogate_root_success") == 1
    assert _platoon_deprecations(caught) == []  # the reload of the snapshot does not warn again
    assert loaded[0].train.workflow_config.rollout_config.propagate_root_success is True  # the key still works
    lock = json.loads((tmp_path / "logs/exp/trial/lock.json").read_text())
    replacement = "train.workflow_config.rollout_config.propagate_root_success"
    assert lock["deprecations"] == [{"path": LEGACY, "replacement": replacement, "removed_in": "0.3.0"}]


def test_task_order_is_listed_flagged_by_strict_and_recorded_as_task_samplers_former_name(
    tmp_path, monkeypatch, capsys
):
    """``train.task_order``, ``train.task_sampler``'s name before platoon 0.3.0, still loads: ``config validate`` lists
    it under the config and ``--strict`` fails on it, and ``platoon train`` names it once and records it in
    ``lock.json``, with the sampler it names in force."""

    from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
    from platoon.utils.config import load_config

    config = _train_config(tmp_path)
    data = yaml.safe_load(config.read_text())
    data["train"]["task_order"] = "weighted"
    config.write_text(yaml.safe_dump(data))
    line = "train.task_order: deprecated; use train.task_sampler instead (platoon 0.4.0 removes it)"

    cli.main(["config", "validate", str(config)])
    assert capsys.readouterr().out == f"ok   {config} (PlatoonTinkerRLTrainerConfig)\n  {line}\n"
    assert _exit_code(["config", "validate", "--strict", str(config)]) == cli.EXIT_CONFIG
    assert capsys.readouterr().out == f"FAIL {config}\n  {line}\n"

    loaded = []
    monkeypatch.setattr(
        "platoon.train.tinker.train.run_tinker_training",
        lambda args: loaded.append(load_config(args, PlatoonTinkerRLTrainerConfig)[0]),
    )
    cli.main(["train", str(config)])
    assert capsys.readouterr().err.count(f"platoon: {line}\n") == 1
    assert (loaded[0].train.task_sampler, loaded[0].train.task_order) == ("weighted", None)
    lock = json.loads((tmp_path / "logs/exp/trial/lock.json").read_text())
    assert lock["deprecations"] == [
        {"path": "train.task_order", "replacement": "train.task_sampler", "removed_in": "0.4.0"}
    ]


# What platoon 0.2.0's `config show --defaults` printed for a training config on this file's loaders: every key at its
# 0.2.0 default, `train.task_order: auto` among them.
DEFAULTS_0_2_0 = Path(__file__).parent / "fixtures" / "config_show_defaults_0.2.0.yaml"
TASK_ORDER_LINE = "train.task_order: deprecated; use train.task_sampler instead (platoon 0.4.0 removes it)"


def _defaults_0_2_0(directory: Path) -> Path:
    """The 0.2.0 dump in ``directory``, its loaders named by this module's name."""

    directory.mkdir(parents=True, exist_ok=True)
    dump = directory / "defaults.yaml"
    dump.write_text(DEFAULTS_0_2_0.read_text().replace("test_cli:", f"{__name__}:"))
    assert yaml.safe_load(dump.read_text())["train"]["task_order"] == "auto"
    return dump


def test_a_0_2_0_defaults_dump_beside_task_sampler_loads_and_says_that_task_order_is_ignored(
    tmp_path, monkeypatch, capsys
):
    """A config copied from 0.2.0's ``config show --defaults`` spells ``train.task_order: auto``, its default then.
    Beside it ``train.task_sampler`` wins, and the deprecation says that ``task_order`` was ignored: ``config
    validate`` lists it and ``--strict`` fails on it, and ``platoon train`` names it once, trains with
    ``task_sampler``'s sampler and records the deprecation in ``lock.json``, for a config that extends the dump and
    for an override of the dump alike."""

    from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
    from platoon.utils.config import load_config

    dump = _defaults_0_2_0(tmp_path / "dump")
    child = tmp_path / "dump" / "child.yaml"
    child.write_text(yaml.safe_dump({"extends": dump.name, "train": {"task_sampler": "weighted"}}))
    line = f"{TASK_ORDER_LINE}; its value 'auto' is ignored, since train.task_sampler is 'weighted'"

    cli.main(["config", "validate", str(child)])
    assert capsys.readouterr().out == f"ok   {child} (PlatoonTinkerRLTrainerConfig)\n  {line}\n"
    assert _exit_code(["config", "validate", "--strict", str(child)]) == cli.EXIT_CONFIG
    assert capsys.readouterr().out == f"FAIL {child}\n  {line}\n"

    loaded = []
    monkeypatch.setattr(
        "platoon.train.tinker.train.run_tinker_training",
        lambda args: loaded.append(load_config(args, PlatoonTinkerRLTrainerConfig)[0]),
    )
    override = [str(_defaults_0_2_0(tmp_path / "override")), "train.task_sampler=weighted"]
    for argv in ([str(child)], override):
        directory = Path(argv[0]).parent
        monkeypatch.chdir(directory)  # the dump's log_path, logs, is relative to the working directory
        cli.main(["train", *argv])
        assert capsys.readouterr().err.count(f"platoon: {line}\n") == 1
        lock = json.loads((directory / "logs/exp/trial/lock.json").read_text())
        assert lock["deprecations"] == [
            {"path": "train.task_order", "replacement": "train.task_sampler", "removed_in": "0.4.0"}
        ]
    assert [(config.train.task_sampler, config.train.task_order) for config in loaded] == [("weighted", None)] * 2


def test_a_0_2_0_defaults_dump_loads_with_task_sampler_alone_unless_task_order_weighted_contradicts_it(
    tmp_path, monkeypatch, capsys
):
    """The 0.2.0 dump alone loads with ``task_order``'s ``auto``, and beside the same ``task_sampler`` too, each with
    the plain deprecation.  ``task_order: weighted`` beside another ``task_sampler`` is a conflict that neither
    ``config validate`` nor ``platoon train`` loads (exit 78)."""

    monkeypatch.chdir(tmp_path)
    dump = _defaults_0_2_0(tmp_path)

    for overrides in ([], ["train.task_sampler=auto"], ["train.task_order=weighted", "train.task_sampler=weighted"]):
        cli.main(["config", "validate", str(dump), *overrides])
        assert capsys.readouterr().out == f"ok   {dump} (PlatoonTinkerRLTrainerConfig)\n  {TASK_ORDER_LINE}\n"
    conflict = ["train.task_order=weighted", "train.task_sampler=auto"]
    problem = (
        "train.task_sampler is 'auto' and train.task_order, its deprecated name, is 'weighted'; set task_sampler alone"
    )
    assert _exit_code(["config", "validate", str(dump), *conflict]) == cli.EXIT_CONFIG
    out = capsys.readouterr().out
    assert out.startswith(f"FAIL {dump}\n") and problem in out
    assert _exit_code(["train", str(dump), *conflict]) == cli.EXIT_CONFIG
    assert problem in capsys.readouterr().err
    assert not (tmp_path / "logs").exists()  # refused before the run directory is made


def test_eval_names_a_deprecated_key_once_and_records_it_in_the_lock(tmp_path, capsys):
    config = _with_legacy_key(_eval_config(tmp_path), ("inference", "workflow"))
    key = "inference.workflow.rollout_config.propogate_root_success"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cli.main(["eval", str(config)])

    assert capsys.readouterr().err.count(f"platoon: {key}: deprecated; use ") == 1
    assert _platoon_deprecations(caught) == []  # run_eval loaded the snapshot again, silently
    lock = json.loads((tmp_path / "results/lock.json").read_text())
    assert [deprecation["path"] for deprecation in lock["deprecations"]] == [key]
    snapshot = yaml.safe_load((tmp_path / "results/config.resolved.yaml").read_text())
    assert snapshot["inference"]["workflow"]["rollout_config"] == {"propogate_root_success": True}  # as given


def test_a_config_without_deprecated_keys_records_none(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("platoon.train.tinker.train.run_tinker_training", lambda args: None)

    cli.main(["train", str(_train_config(tmp_path))])

    assert json.loads((tmp_path / "logs/exp/trial/lock.json").read_text())["deprecations"] == []
    assert "deprecated" not in capsys.readouterr().err


def test_an_error_warning_filter_makes_a_deprecated_key_a_config_error(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr("platoon.train.tinker.train.run_tinker_training", calls.append)
    config = _with_legacy_key(_train_config(tmp_path), ("train", "workflow_config"))

    with warnings.catch_warnings():
        warnings.simplefilter("error", PlatoonDeprecationWarning)
        assert _exit_code(["train", str(config)]) == cli.EXIT_CONFIG

    assert f"platoon: {LEGACY_LINE}; the warning filters make deprecations errors" in capsys.readouterr().err
    assert calls == [] and not (tmp_path / "logs/exp/trial/lock.json").exists()


def test_a_command_s_deprecation_filters_end_with_it(tmp_path, monkeypatch):
    monkeypatch.setattr("platoon.train.tinker.train.run_tinker_training", lambda args: None)
    before = list(warnings.filters)

    cli.main(["train", str(_with_legacy_key(_train_config(tmp_path), ("train", "workflow_config")))])

    assert warnings.filters == before


# --- harness and env ---------------------------------------------------------------------------------------------


def test_harness_list_shows_the_toml_specs(capsys):
    cli.main(["harness", "list"])

    out = capsys.readouterr().out
    assert out.splitlines()[0].split() == ["harness", "kind", "distribution", "source"]
    assert any(line.split()[:3] == ["platoon-bash@1", "spec", "platoon"] for line in out.splitlines())


def test_harness_list_and_show_cover_function_harnesses(monkeypatch, capsys):
    async def scripted(ctx):
        """Answer from a script."""

    class FakeFunctionHarness:
        ref, runtime, executes_model_code, fn = "scripted@1", "inline", False, scripted
        source = "tests:scripted"

    def get_harness(ref):
        return FakeFunctionHarness() if ref.startswith("scripted") else registry.get_harness_spec(ref)

    monkeypatch.setattr(registry, "available_harnesses", lambda: ["platoon-bash@1", "scripted@1"], raising=False)
    monkeypatch.setattr(registry, "get_harness", get_harness, raising=False)

    cli.main(["harness", "list"])
    cli.main(["harness", "show", "scripted"])

    out = capsys.readouterr().out
    assert any(line.split() == ["scripted@1", "function", "-", "tests:scripted"] for line in out.splitlines())
    assert "scripted@1  (function)\nsource: tests:scripted\nruntime: inline" in out
    assert "Answer from a script." in out


def test_harness_show_prints_a_spec(capsys):
    cli.main(["harness", "show", "platoon-bash"])

    out = capsys.readouterr().out
    assert out.startswith("platoon-bash@1  (spec)\n")
    assert "built-in tools: ['bash']" in out
    assert "launch: python3 {state_dir}/platoon_bash.py" in out


def test_an_unknown_harness_is_a_config_error(capsys):
    assert _exit_code(["harness", "show", "no-such-harness"]) == cli.EXIT_CONFIG
    assert "Unknown harness 'no-such-harness'" in capsys.readouterr().err


def test_env_list_shows_the_service_schemes_and_a_configs_environments(tmp_path, capsys):
    config = tmp_path / "mixture.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "environments": [
                    {"name": "swe", "weight": 2, "service": "harbor", "harness": "mini-swe-agent"},
                    {"name": "craft", "service": "python://textcraft.env:TextCraft", "harness": {"platoon-codeact": 1}},
                ]
            }
        )
    )

    cli.main(["env", "list", str(config)])

    out = capsys.readouterr().out
    for scheme in ("harbor", "table", "python://pkg.module:EnvClass", "ors+http://HOST:PORT[,...]/EnvName"):
        assert scheme in out
    rows = [line.split() for line in out.split(f"Environments of {config}:")[1].strip().splitlines()]
    assert rows == [
        ["name", "weight", "service", "harness", "program", "rollout"],
        ["swe", "2", "harbor", "mini-swe-agent", "-", "harness_rollout"],
        ["craft", "tasks", "python://textcraft.env:TextCraft", "platoon-codeact:1", "-", "harness_rollout"],
    ]


# --- viz, help and usage -----------------------------------------------------------------------------------------


def test_viz_hands_its_arguments_to_the_trajectory_viewer(monkeypatch):
    from platoon.visualization import cli as visualization_cli

    seen = []
    monkeypatch.setattr(visualization_cli, "main", lambda argv=None: seen.append(argv))

    cli.main(["viz", "tail", "--rdir", "runs/x", "--mode", "openhands"])

    assert seen == [["tail", "--rdir", "runs/x", "--mode", "openhands"]]


def test_viz_also_drives_a_viewer_that_reads_sys_argv(monkeypatch):
    from platoon.visualization import cli as visualization_cli

    seen = []
    monkeypatch.setattr(visualization_cli, "main", lambda: seen.append(list(sys.argv)))

    cli.main(["viz", "replay", "events.jsonl"])

    assert seen == [["platoon viz", "replay", "events.jsonl"]]


@pytest.mark.parametrize(
    "command",
    [
        [],
        ["train"],
        ["eval"],
        ["rollout"],
        ["config"],
        ["config", "show"],
        ["config", "validate"],
        ["harness"],
        ["harness", "list"],
        ["harness", "show"],
        ["env"],
        ["env", "list"],
    ],
)
def test_every_command_has_help(command, capsys):
    cli.main([*command, "--help"])

    out = capsys.readouterr().out
    assert out.startswith("Usage: platoon")
    assert len(out.splitlines()) > 3


def test_top_level_help_lists_every_command(capsys):
    cli.main(["--help"])

    out = capsys.readouterr().out
    for command in ("train", "eval", "rollout", "config", "harness", "env", "viz"):
        assert f"│ {command} " in out


def test_an_unknown_command_is_a_usage_error():
    assert _exit_code(["fly"]) == cli.EXIT_USAGE
