"""CPU tests of the example extension: its task table, whole episodes with a scripted policy, and the named
environment through ``platoon rollout``.

    uv pip install -e examples/extension-package
    uv run pytest examples/extension-package/tests -q

``platoon.testing.run_episode`` runs one task the way training and evaluation do: the environment session, the
episode program, the real model gateway and the trajectories, with a scripted policy in place of the model. Those
tests name the components by import string, so they run whether or not the package is installed. The last test
serves a scripted policy to ``platoon rollout`` with ``platoon.testing.serve_policy`` and runs ``named.yaml``, whose
``env: example-arithmetic@1`` finds every component by the name its entry point gives: it needs the package
installed.
"""

from __future__ import annotations

import asyncio
import json
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import pytest
from platoon.cli import main as platoon_cli
from platoon.testing import ScriptedPolicy, run_episode, serve_policy

from example_extension.program import closeness
from example_extension.rewards import partial_credit
from example_extension.tasks import arithmetic

NAMED = Path(__file__).parents[1] / "named.yaml"
INSTALLED = bool(entry_points(group="platoon.environments", name="example-arithmetic"))


def episode(task_id: str, *replies: str, **harness_options: Any) -> tuple[Any, ScriptedPolicy]:
    """One episode of the arithmetic task ``task_id``, in which the model writes ``replies`` in order."""

    policy = ScriptedPolicy(list(replies))
    run = asyncio.run(
        run_episode(
            task_id,
            service="example_extension.tasks:arithmetic",
            harness="example_extension.harness:self_check",
            program="example_extension.program:exact_answer",
            harness_options=harness_options,
            policy=policy,
        )
    )
    return run, policy


def test_the_tasks_ship_with_the_package():
    async def read() -> tuple[list[str], list[str], Any]:
        table = arithmetic()
        return await table.list_tasks("train"), await table.list_tasks("test"), await table.get_task("test-0")

    train, test, task = asyncio.run(read())

    assert (len(train), len(test)) == (24, 8)
    assert (task.goal, task.misc["answer"]) == ("What is 22 + 80? Reply with the number only.", "102")


def test_the_answer_after_the_check_is_graded():
    run, policy = episode("test-0", "100", "102")  # a wrong first answer, corrected by the check

    assert run.outcome["status"] == "scored" and run.reward == 1.0
    assert run.outcome["rewards"]["close"] == 1.0
    assert len(policy.calls) == 2
    assert policy.calls[1].last_user_message.startswith("Check your answer step by step.")


def test_a_near_miss_earns_partial_credit_in_training_only():
    run, _ = episode("test-0", "100", "101")

    assert run.reward == 0.0  # the grade: 101 is not the answer
    trajectory = run.collection.to_dict()["trajectories"][run.root.id]
    assert partial_credit(trajectory) == (
        0.5 * closeness("101", "102"),
        {"reward/exact": 0.0, "reward/close": closeness("101", "102")},
    )


def test_checks_sets_how_many_times_the_model_is_asked():
    run, policy = episode("test-2", "36", checks=0)

    assert run.reward == 1.0 and len(policy.calls) == 1


def test_an_unknown_option_is_an_error_not_a_score():
    run, policy = episode("test-2", chekcs=1)

    assert run.outcome["status"] == "error" and not policy.calls


@pytest.mark.skipif(not INSTALLED, reason="needs the package installed: uv pip install -e examples/extension-package")
def test_the_named_environment_finds_every_component_by_name(tmp_path, capsys):
    with serve_policy(["100", "102"]) as policy:  # platoon rollout samples this policy instead of a Tinker model
        platoon_cli(["rollout", str(NAMED), "--task", "test-0", "--output-dir", str(tmp_path), "--json-output"])

    summary = json.loads(capsys.readouterr().out)
    assert (summary["reward"], summary["total_steps"]) == (1.0, 2) and len(policy.calls) == 2
    lock = json.loads((tmp_path / "lock.json").read_text())
    assert {component["kind"]: component["resolved"] for component in lock["components"]} == {
        "environment": "example-arithmetic@1",
        "harness": "example-self-check@1",
        "program": "example-exact-answer@1",
        "service": "example_extension.tasks:arithmetic",
        "reward_processor": "example-partial-credit",
    }
