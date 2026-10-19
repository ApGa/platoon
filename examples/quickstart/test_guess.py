"""Tests of the quickstart's environment, harness and config, on CPU: no GPU, Tinker server or model download.

A scripted policy from platoon.testing writes what the model would; the rest is platoon's real code: the model
gateway, the harness, the environment, the episode program and the trajectories. From this directory:

    PYTHONPATH=. uv run pytest -p platoon.testing test_guess.py
"""

import asyncio
import json
import re
from pathlib import Path

import guesser  # noqa: F401  (importing it registers the example-guesser harness)

import platoon as pt
from platoon.cli import main as platoon_cli
from platoon.testing import ScriptedPolicy, run_episode, tool_call

GUESS_ENV = "python://guess:GuessTheNumber"
TRAIN_CONFIG = Path(__file__).with_name("train.yaml")


def bisect(prompt: str) -> str:
    """A policy: the middle of the range that the environment's answers so far leave."""

    low, high = 1, 100
    guesses = [int(number) for number in re.findall(r'"number": (\d+)\}', prompt)]
    answers = re.findall(r"<tool_response>\s*(higher|lower)\s*</tool_response>", prompt)
    for guess, answer in zip(guesses, answers):
        low, high = (guess + 1, high) if answer == "higher" else (low, guess - 1)
    return tool_call("guess", number=(low + high) // 2)


def test_the_environment_grades_a_correct_guess():
    # Task train:41 hides 42. The built-in platoon-react harness makes the tool calls the script writes, in order.
    policy = ScriptedPolicy([tool_call("guess", number=50), tool_call("guess", number=42)])
    run = asyncio.run(run_episode("train:41", service=GUESS_ENV, harness="platoon-react", policy=policy))

    assert run.reward == 1.0 and run.outcome["status"] == "scored"
    assert "lower" in policy.calls[1].last_user_message  # what the environment answered to 50


def test_the_guesser_finds_the_secret_by_bisection():
    run = asyncio.run(run_episode("train:41", service=GUESS_ENV, harness="example-guesser", policy=bisect))

    guesses = [call["arguments"]["number"] for call in run.root.misc["tool_calls"]]
    assert guesses == [50, 25, 37, 43, 40, 41, 42] and run.reward == 1.0


def test_the_guesser_stops_at_its_step_budget():
    policy = ScriptedPolicy(lambda prompt: tool_call("guess", number=1))
    run = asyncio.run(run_episode("train:41", service=GUESS_ENV, harness="example-guesser", policy=policy, max_steps=3))

    assert len(policy.calls) == 3 and run.reward == 0.0


@pt.program("example-count-guesses")
async def count_guesses(ep: pt.EpisodeContext) -> pt.Outcome:
    """An episode program: run the root agent, grade it, and record its number of guesses on its trajectory."""

    def record(collection: pt.TrajectoryCollection) -> None:
        root = next(iter(collection.trajectories.values()))
        root.misc["guesses"] = len(root.misc.get("tool_calls", []))

    await ep.run()
    ep.add_finalizer(record)
    return await ep.grade()


def test_the_program_records_the_number_of_guesses():
    run = asyncio.run(
        run_episode(
            "train:41", service=GUESS_ENV, harness="example-guesser", program="example-count-guesses", policy=bisect
        )
    )

    assert run.root.misc["guesses"] == 7 and run.reward == 1.0


def test_platoon_rollout_runs_the_training_config(serve_policy, tmp_path, capsys):
    # The serve_policy fixture (pytest -p platoon.testing) serves the policy in place of Tinker.
    serve_policy(bisect)
    platoon_cli(["rollout", str(TRAIN_CONFIG), "--task", "train:41", "--output-dir", str(tmp_path), "--json-output"])

    summary = json.loads(capsys.readouterr().out)
    assert summary["reward"] == 1.0 and summary["total_steps"] == 7
