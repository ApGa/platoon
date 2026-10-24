"""``platoon train``, ``eval`` and ``rollout`` choose the policy's tool-call parser before they run and record the
choice in ``lock.json`` (``tool_call_parsing``); a chat template that no parser reads back is a configuration error
(exit 78) before anything runs.  The run entry points are monkeypatched, as in ``test_cli.py``."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import yaml
from huggingface_hub.constants import HF_HUB_CACHE

from platoon import cli
from platoon.task import Task

QWEN3_INSTRUCT = Path(os.environ.get("PLATOON_TEST_HF_HUB") or HF_HUB_CACHE) / "models--Qwen--Qwen3-4B-Instruct-2507"


def load_tasks(config, split, **kwargs) -> list[str]:
    return ["solved"]


def load_task(task_id: str) -> Task:
    return Task(goal=f"solve {task_id}", id=task_id)


async def answer(task: Task, config) -> dict:
    root = {
        "id": "root",
        "task": {"id": task.id, "goal": task.goal},
        "parent_info": None,
        "reward": 1.0,
        "steps": [{}],
        "misc": {"agent": {"agent_id": "a0", "role": "root"}, "outcome": {"status": "scored", "rewards": {}}},
    }
    return {"id": "rollout-1", "trajectories": {"root": root}}


ENVIRONMENT = {
    "dataset_loader": f"{__name__}:load_tasks",
    "task_loader": f"{__name__}:load_task",
    "rollout": f"{__name__}:answer",
}


class Policy:
    """Stands in for ``EvalPolicy``, whose ``install`` would start a Tinker sampler."""

    def __init__(self, **settings) -> None:
        self.__dict__.update(settings)

    def install(self) -> None:
        pass


@pytest.fixture
def no_tinker(monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")  # eval serves a policy: past its key check
    monkeypatch.setattr("platoon.train.tinker.train.run_tinker_training", lambda args: None)
    monkeypatch.setattr(sys.modules[cli._evaluation().__package__], "EvalPolicy", Policy, raising=False)
    monkeypatch.setattr("platoon.eval.config_defs.EvalPolicy.install", lambda self: None)


@pytest.fixture
def choices(monkeypatch) -> list[tuple]:
    """Answer ``tool_call_parsing_record`` without loading a model, and keep what it was asked."""

    from platoon.train.tinker import proxy

    asked: list[tuple] = []

    def record(model_name, renderer_name, renderer_kwargs=None):
        asked.append((model_name, renderer_name, dict(renderer_kwargs or {})))
        return {"model_name": model_name, "renderer_name": renderer_name, "parser": "hermes", "verified": True}

    monkeypatch.setattr(proxy, "tool_call_parsing_record", record)
    return asked


def _train_config(tmp_path: Path, **train) -> Path:
    path = tmp_path / "train.yaml"
    section = {"model_name": "Qwen/Qwen3-4B", "renderer_name": "qwen3", "batch_size": 16, **train}
    data = {
        "train": section,
        "eval": {"every": 0},
        "log_path": str(tmp_path / "logs"),
        "stats": {"experiment_name": "exp", "trial_name": "trial"},
        "environments": [ENVIRONMENT],
    }
    path.write_text(yaml.safe_dump(data))
    return path


def _lock(path: Path) -> dict:
    return json.loads(path.read_text())["tool_call_parsing"]


def _exit_code(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    return exit_info.value.code


def test_a_cookbook_renderer_is_recorded_as_parsing_its_own_tool_calls(tmp_path, no_tinker):
    cli.main(["train", str(_train_config(tmp_path))])

    assert _lock(tmp_path / "logs/exp/trial/lock.json") == {
        "model_name": "Qwen/Qwen3-4B",
        "renderer_name": "qwen3",
        "parser": None,
        "reason": "the renderer parses tool calls itself",
    }


def test_train_eval_and_rollout_record_the_policys_parser(tmp_path, no_tinker, choices):
    kwargs = {"tool_call_parser": "auto", "preserve_thinking": True}
    config = _train_config(tmp_path, renderer_name="chat_template", renderer_kwargs=kwargs)
    cli.main(["train", str(config)])
    assert _lock(tmp_path / "logs/exp/trial/lock.json")["parser"] == "hermes"

    inference = {"model_name": "m", "renderer_name": "chat_template", "output_dir": str(tmp_path / "out")}
    evaluation = tmp_path / "eval.yaml"
    evaluation.write_text(yaml.safe_dump({"inference": inference, "environments": [ENVIRONMENT]}))
    cli.main(["eval", str(evaluation)])
    assert _lock(tmp_path / "out/lock.json")["model_name"] == "m"

    cli.main(["rollout", str(evaluation), "--task", "solved", "--output-dir", str(tmp_path / "debug")])
    assert _lock(tmp_path / "debug/lock.json")["renderer_name"] == "chat_template"
    assert choices == [
        ("Qwen/Qwen3-4B", "chat_template", kwargs),
        ("m", "chat_template", {}),
        ("m", "chat_template", {}),
    ]


def test_the_parser_joins_the_components_in_each_lock(tmp_path, no_tinker, choices):
    """``lock.json`` gets the parser after the run record lists the resolved components; neither replaces the other."""

    train = _train_config(tmp_path, renderer_name="chat_template")
    inference = {"model_name": "m", "renderer_name": "chat_template", "output_dir": str(tmp_path / "out")}
    evaluation = tmp_path / "eval.yaml"
    evaluation.write_text(yaml.safe_dump({"inference": inference, "environments": [ENVIRONMENT]}))

    cli.main(["train", str(train)])
    cli.main(["eval", str(evaluation)])
    cli.main(["rollout", str(evaluation), "--task", "solved", "--output-dir", str(tmp_path / "debug")])

    components = {(kind, ENVIRONMENT[kind]) for kind in ("dataset_loader", "task_loader", "rollout")}
    # A training run also records the task sampler it draws its tasks with.
    trains = {**dict.fromkeys(("out", "debug"), components), "logs/exp/trial": components | {("task_sampler", "auto")}}
    for where, expected in trains.items():
        lock_path = tmp_path / where / "lock.json"
        lock = json.loads(lock_path.read_text())
        assert lock["tool_call_parsing"]["parser"] == "hermes", lock_path
        assert {(component["kind"], component["ref"]) for component in lock["components"]} == expected, lock_path
        assert "extensions" in lock, lock_path


def test_nothing_is_recorded_without_a_policy(tmp_path, no_tinker, choices):
    inference = {"model_name": "m", "output_dir": str(tmp_path / "out")}
    evaluation = tmp_path / "eval.yaml"
    evaluation.write_text(yaml.safe_dump({"inference": inference, "environments": [ENVIRONMENT]}))
    cli.main(["eval", str(evaluation)])
    assert "tool_call_parsing" not in json.loads((tmp_path / "out/lock.json").read_text()) and choices == []


def test_a_template_no_parser_reads_is_a_configuration_error_before_training(tmp_path, monkeypatch, capsys):
    from platoon.train.tinker import proxy
    from platoon.train.tinker.renderers import ToolCallParserError

    trained = []
    monkeypatch.setattr("platoon.train.tinker.train.run_tinker_training", trained.append)
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")  # train checks its key before anything else: past it

    def refuse(model_name, renderer_name, renderer_kwargs=None):
        raise ToolCallParserError("No tool-call parser reads the chat template's own tool calls back for m.")

    monkeypatch.setattr(proxy, "tool_call_parsing_record", refuse)
    config = _train_config(tmp_path, renderer_name="chat_template")

    assert _exit_code(["train", str(config)]) == cli.EXIT_CONFIG
    assert "renderer_kwargs.tool_call_parser: No tool-call parser reads" in capsys.readouterr().err
    assert trained == []


def test_an_unknown_parser_name_is_a_configuration_error(tmp_path, no_tinker, capsys):
    pytest.importorskip("transformers")
    if not QWEN3_INSTRUCT.exists():
        pytest.skip(f"no Qwen3-4B-Instruct-2507 snapshot under {QWEN3_INSTRUCT.parent}")
    config = _train_config(
        tmp_path,
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        renderer_name="chat_template",
        renderer_kwargs={"tool_call_parser": "qwen3"},
    )
    assert _exit_code(["train", str(config)]) == cli.EXIT_CONFIG
    assert "Unknown tool_call_parser 'qwen3'" in capsys.readouterr().err


def test_the_real_choice_for_a_cached_model(tmp_path, no_tinker):
    if not QWEN3_INSTRUCT.exists():
        pytest.skip(f"no Qwen3-4B-Instruct-2507 snapshot under {QWEN3_INSTRUCT.parent}")
    config = _train_config(tmp_path, model_name="Qwen/Qwen3-4B-Instruct-2507", renderer_name="chat_template")

    cli.main(["train", str(config)])

    assert _lock(tmp_path / "logs/exp/trial/lock.json") == {
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "renderer_name": "chat_template",
        "setting": "auto",
        "parser": "hermes",
        "verified": True,
        "reason": "round trip (one turn with every call)",
    }
