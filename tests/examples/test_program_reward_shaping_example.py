"""examples/program-reward-shaping: ``platoon rollout`` runs the example-efficient-search program with a scripted
policy.

The policy plays number search by bisection after one wasted guess, so the finalizer has an extra guess to
penalize, and answers the judge's question with a fixed sentence.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import pytest
from efficient_search import binary_search_guesses, reward_parts

CONFIG = "examples/program-reward-shaping/efficient_search.yaml"
CRITIQUE = "Start in the middle of the range."
_GUESSES = re.compile(r'"name": "guess", "arguments": \{"number": (-?\d+)\}')
_FEEDBACK = re.compile(r"<tool_response>\n(.*?)\n</tool_response>", re.S)


def wasteful_bisection(prompt: str) -> str:
    """A guess at the bottom of the range, then a binary search; the judge gets a fixed critique."""

    if "how could this search have used fewer guesses" in prompt:
        return CRITIQUE
    low, high = (int(value) for value in re.search(r"between (-?\d+) and (-?\d+)", prompt).groups())
    guesses = [int(number) for number in _GUESSES.findall(prompt)]
    for number, feedback in zip(guesses, _FEEDBACK.findall(prompt)):
        if feedback.startswith("Too low"):
            low = max(low, number + 1)
        elif feedback.startswith("Too high"):
            high = min(high, number - 1)
    number = low if not guesses else (low + high) // 2
    return f"<tool_call>\n{json.dumps({'name': 'guess', 'arguments': {'number': number}})}\n</tool_call>"


def test_the_finalizer_shapes_the_root_reward_and_records_the_judge(serve_policy, platoon_cli, tmp_path, capsys):
    serve_policy(wasteful_bisection)
    platoon_cli("rollout", CONFIG, "--task", "train:0", "--output-dir", str(tmp_path))
    out = capsys.readouterr().out
    collection = json.loads((tmp_path / "trajectory_collection.json").read_text())

    (root,) = collection["trajectories"].values()
    goal = "Guess the correct number between 6 and 988."
    guesses = [call for call in root["misc"]["tool_calls"] if call["name"] == "guess"]
    assert guesses[-1]["finished"] and guesses[-1]["reward"] == 1.0  # the search succeeded
    extra = max(0, len(guesses) - binary_search_guesses(goal))
    assert binary_search_guesses(goal) == math.ceil(math.log2(988 - 6 + 2)) == 10
    assert len(guesses) == 11 and extra == 1
    reward = 0.9
    assert root["reward"] == pytest.approx(reward)
    reward_misc = root["steps"][-1]["misc"]["reward_misc"]
    assert reward_misc["reward/success"] == 1.0  # the success evaluation counts is untouched
    assert reward_misc["reward/extra_guesses"] == extra and reward_misc["reward/shaped"] == pytest.approx(reward)
    assert root["misc"]["outcome"] == {
        "status": "scored",
        "rewards": {"reward": 1.0},
        "reason": None,
        "reward_key": "reward",
    }
    # The judge's call is kept for audit, never as a training step.
    assert root["misc"]["judge_critique"] == CRITIQUE
    (judged,) = root["misc"]["judge_calls"]
    assert judged["role"] == "judge" and judged["message"]["content"] == CRITIQUE
    assert len(root["steps"]) == len(guesses)
    assert root["misc"]["harness"]["program"] == "example-efficient-search"
    assert f"reward {root['reward']}" in out
    # The environment's reward processor trains on the shaped reward and reports its parts.
    assert reward_parts(root) == (
        pytest.approx(reward),
        {"reward/success": 1.0, "reward/extra_guesses": 1.0, "reward/shaped": pytest.approx(reward)},
    )


def test_the_penalty_comes_from_the_rollout_config(serve_policy, platoon_cli, tmp_path, capsys):
    serve_policy(wasteful_bisection)
    platoon_cli(
        "rollout",
        CONFIG,
        "train.workflow_config.rollout_config.extra.guess_penalty=0",
        "--task",
        "train:0",
        "--output-dir",
        str(tmp_path),
    )
    collection: dict[str, Any] = json.loads((tmp_path / "trajectory_collection.json").read_text())
    (root,) = collection["trajectories"].values()
    assert root["reward"] == 1.0
