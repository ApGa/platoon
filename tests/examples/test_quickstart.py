"""The quickstart example (``examples/quickstart``) runs as its README says, on CPU, with a scripted policy.

``platoon rollout`` and ``platoon eval`` serve the policy through Tinker. Here ``platoon.testing.serve_policy`` serves
a scripted policy in its place, behind a real model gateway with the toy tokenizer and the config's renderer, so the
example's environment, harness and configs run end to end through the CLI: the ORS environment in process, the
``example-guesser`` harness inline, and every model call rendered, sampled, parsed and recorded by the gateway. The
scripted policy plays the game by bisection, reading the env's answers from its prompt.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OPENREWARD_DISABLE_UPDATE_CHECK", "1")
pytest.importorskip("openreward")

from platoon import cli  # noqa: E402
from platoon.testing import ScriptedPolicy, serve_policy, tool_call  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "quickstart"


def bisect(prompt: str) -> str:
    """The next guess: replay the guesses and the env's answers so far, then halve the range that is left."""

    low, high = 1, 100
    guesses = [int(number) for number in re.findall(r'"number": (\d+)\}', prompt)]
    answers = re.findall(r"<tool_response>\s*(higher|lower)\s*</tool_response>", prompt)
    for guess, answer in zip(guesses, answers, strict=False):
        low, high = (guess + 1, high) if answer == "higher" else (low, guess - 1)
    return tool_call("guess", number=(low + high) // 2)


@pytest.fixture
def policy(monkeypatch) -> Iterator[ScriptedPolicy]:
    """Serve the bisection policy wherever the CLI would serve the Tinker policy; yields it."""

    monkeypatch.syspath_prepend(str(EXAMPLE))  # what PYTHONPATH=. does in the README
    monkeypatch.chdir(EXAMPLE)
    with serve_policy(bisect) as scripted:
        yield scripted


def test_the_example_configs_validate(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(EXAMPLE))  # what PYTHONPATH=. does in the README: the components import

    cli.main(["config", "validate", str(EXAMPLE / "train.yaml"), str(EXAMPLE / "eval.yaml")])

    assert capsys.readouterr().out.count("ok") == 2


def test_platoon_rollout_solves_one_task(policy, tmp_path, capsys):
    run_dir = tmp_path / "rollout"

    cli.main(["rollout", "train.yaml", "--task", "train:41", "--output-dir", str(run_dir), "--json-output"])

    summary: dict[str, Any] = json.loads(capsys.readouterr().out)
    (root,) = summary["agents"]
    # Secret 42 by bisection: 50, 25, 37, 43, 40, 41, 42.
    assert root["steps"] == 7 and root["reward"] == 1.0
    assert root["outcome"]["status"] == "scored"
    assert policy.calls[-1].completion.text == tool_call("guess", number=42)
    for name in ("config.resolved.yaml", "lock.json", "trajectory_collection.json"):
        assert (run_dir / name).is_file()


def test_platoon_eval_reports_every_task_solved(policy, tmp_path, capsys):
    output_dir = tmp_path / "eval"
    overrides = [
        f"inference.output_dir={output_dir}",
        "inference.workflow.num_rollouts_per_task=1",
        "environments.0.eval_dataset_kwargs.limit=3",
    ]

    cli.main(["eval", "eval.yaml", *overrides])

    summary = json.loads(capsys.readouterr().out)
    assert summary["success_rate"] == 1.0
    assert (output_dir / "config.resolved.yaml").is_file()
