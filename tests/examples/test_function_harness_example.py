"""examples/function-harness: the example-tool-loop harness plays number search through ``platoon rollout``, with a
scripted policy behind the real gateway (CPU only; the test model's tokenizer is ``platoon.testing``'s toy
tokenizer)."""

from __future__ import annotations

import json
import tomllib
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import pytest
import yaml
from harness_example_fakes import (
    EXAMPLES,
    GUESS,
    VAL_0_GUESSES,
    bisect,
    isolate_example,
    load_tokenizer,
    serve_scripted_policy,
)

from platoon import cli
from platoon.harness.agent import LIMIT_MESSAGE
from platoon.harness.registry import HarnessRegistry, get_harness

EXAMPLE = EXAMPLES / "function-harness"
CONFIG = EXAMPLE / "config.yaml"


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


@pytest.fixture(autouse=True)
def example(monkeypatch):
    isolate_example(monkeypatch, EXAMPLE, ("tool_loop",))


def rollout(tmp_path: Path, *overrides: str) -> dict[str, Any]:
    """``platoon rollout config.yaml --task val:0``; returns the root trajectory it wrote."""

    run_dir = tmp_path / "run"
    cli.main(["rollout", str(CONFIG), *overrides, "--task", "val:0", "--output-dir", str(run_dir)])
    collection = json.loads((run_dir / "trajectory_collection.json").read_text())
    (trajectory,) = collection["trajectories"].values()
    return trajectory


def test_tool_loop_finds_the_number(tmp_path, monkeypatch, tokenizer, capsys):
    sampler = serve_scripted_policy(monkeypatch, tokenizer, bisect())

    trajectory = rollout(tmp_path)

    out = capsys.readouterr().out
    assert "task val:0\nagents 1  steps 9  max depth 0  reward 1.0  outcome scored reward=1.0\n" in out
    assert "└─ policy " in out
    assert trajectory["misc"]["harness"]["spec"] == "example-tool-loop@1"
    assert trajectory["misc"]["agent_result"] == {"status": "completed", "error": None}
    assert trajectory["finish_message"] == "You guessed the number 447 correctly!"
    assert [call["arguments"]["number"] for call in trajectory["misc"]["tool_calls"]] == VAL_0_GUESSES
    # annotate_step put each turn's tool outputs on that turn's step.
    steps = trajectory["steps"]
    assert steps[0]["misc"]["observations"] == [{"tool": "guess", "output": "Too high, try again."}]
    assert steps[-1]["misc"]["observations"][0]["output"] == "You guessed the number 447 correctly!"
    assert sampler.calls[0] == "Guess the correct number between 119 and 965."


def test_tool_loop_stops_at_its_step_budget_and_is_still_graded(tmp_path, monkeypatch, tokenizer, capsys):
    serve_scripted_policy(monkeypatch, tokenizer, lambda message: GUESS.format(1))

    trajectory = rollout(tmp_path, "inference.workflow.rollout_config.max_steps=3")

    assert "agents 1  steps 3  max depth 0  reward 0.0  outcome scored reward=0.0" in capsys.readouterr().out
    assert trajectory["misc"]["agent_result"] == {"status": "limit", "error": LIMIT_MESSAGE}


def test_an_answer_without_a_tool_call_ends_the_agent(tmp_path, monkeypatch, tokenizer, capsys):
    serve_scripted_policy(monkeypatch, tokenizer, lambda message: "I give up.")

    trajectory = rollout(tmp_path)

    assert "agents 1  steps 1  max depth 0  reward 0.0" in capsys.readouterr().out
    assert trajectory["finish_message"] == "I give up."
    assert trajectory["misc"]["agent_result"] == {"status": "completed", "error": None}


def test_the_config_names_the_harness_by_name_or_by_module_and_function():
    environment = yaml.safe_load(CONFIG.read_text())["environments"][0]

    __import__(environment["package"])  # what `package:` does in every rollout process
    by_name = get_harness(environment["harness"])

    assert get_harness("tool_loop:tool_loop") == by_name  # module:function needs no `package:`
    assert (by_name.ref, by_name.source, by_name.executes_model_code) == (
        "example-tool-loop@1",
        "tool_loop:tool_loop",
        False,
    )


def test_the_packaged_entry_point_registers_tool_loop():
    group = "platoon.harnesses"
    project = tomllib.loads((EXAMPLE / "pyproject.toml").read_text())
    registry = HarnessRegistry()

    for name, value in project["project"]["entry-points"][group].items():
        registry.add_entry_point(EntryPoint(name=name, value=value, group=group))

    assert registry.entry_point_failures == {}
    assert get_harness("example-tool-loop").source == "tool_loop:tool_loop"


def test_harness_show_describes_tool_loop(capsys):
    cli.main(["harness", "show", "tool_loop:tool_loop"])

    out = capsys.readouterr().out
    assert out.startswith("example-tool-loop@1  (function)\nsource: tool_loop:tool_loop\nruntime: inline\n")
    assert "executes_model_code: False" in out
    assert "Call the environment's tools until it ends the episode" in out


def test_the_example_configs_validate(capsys):
    assert cli.config_validate(CONFIG, EXAMPLE / "train.yaml") == 0

    out = capsys.readouterr().out
    assert "(PlatoonEvalConfig)" in out and "(PlatoonTinkerRLTrainerConfig)" in out
