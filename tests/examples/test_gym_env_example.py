"""``examples/gym-env``: a step environment through ``pt.gym.loop`` (task table + ``gym`` program) and
``pt.gym.as_environment`` (ORS ``act`` tool + ``platoon-react``), and a task table graded by an episode program."""

from __future__ import annotations

import asyncio
import json

import pytest
from example_runs import load_config, rollout

import platoon as pt
from platoon.envs import get_service, task_table

NAME = "gym-env"


def hilo(message: str) -> str:
    """A policy that finds 37: 50 first, then 37."""
    if message.startswith("I am thinking"):
        return "50"
    return "37" if message.startswith("Too high") else "1"


@pytest.mark.parametrize("config", ["train_loop.yaml", "train_tool.yaml", "train_qa.yaml"])
def test_the_configs_validate(example, config):
    example(NAME)
    assert len(load_config(config).environments) == 1


def test_the_table_lists_rows_by_split(example):
    example(NAME)
    service = get_service("table", path="tasks.jsonl")
    assert asyncio.run(service.list_tasks("test")) == ["test-0", "test-1", "test-2", "test-3"]
    task = asyncio.run(service.get_task("train-0"))
    assert (task.goal, task.misc) == ("Guess the number from 1 to 100.", {"target": 37})


def test_gym_loop_plays_the_game_and_the_gym_program_scores_the_return(example, policy_gateway, tmp_path):
    example(NAME)
    sampler = policy_gateway(hilo)
    collection = asyncio.run(rollout(load_config("train_loop.yaml"), "train-0", tmp_path))

    (root,) = collection.trajectories.values()
    assert (root.reward, root.misc["gym_return"], root.misc["outcome"]["status"]) == (1.0, 1.0, "scored")
    assert [step.misc["gym"]["observation"] for step in root.steps] == ["Too high.", "Correct! The number was 37."]
    assert sampler.calls[0].startswith("I am thinking of a number")


def test_gym_loop_scores_a_missed_game_zero(example, policy_gateway, tmp_path):
    example(NAME)
    policy_gateway(lambda message: "1")
    collection = asyncio.run(rollout(load_config("train_loop.yaml"), "train-0", tmp_path))
    (root,) = collection.trajectories.values()
    assert len(root.steps) == 7 and root.misc["agent_result"]["status"] == "limit"  # max_turns
    assert (root.reward, root.misc["outcome"]["status"]) == (0.0, "scored")


def test_as_environment_serves_the_game_through_an_act_tool(example, policy_gateway, tmp_path):
    example(NAME)

    def policy(message: str) -> str:
        guess = hilo(message.removeprefix("<tool_response>\n"))
        return "<tool_call>\n" + json.dumps({"name": "act", "arguments": {"action": guess}}) + "\n</tool_call>"

    policy_gateway(policy)
    collection = asyncio.run(rollout(load_config("train_tool.yaml"), "train:0", tmp_path))

    (root,) = collection.trajectories.values()
    assert (root.reward, root.misc["outcome"]["status"]) == (1.0, "scored")
    assert [call["arguments"] for call in root.misc["tool_calls"]] == [{"action": "50"}, {"action": "37"}]


@pytest.mark.parametrize(("answer", "reward"), [("Paris.", 1.0), ("Lyon", 0.0)])
def test_exact_match_grades_a_table_row_by_its_answer(example, policy_gateway, tmp_path, answer, reward):
    example(NAME)
    policy_gateway(lambda message: answer)
    collection = asyncio.run(rollout(load_config("train_qa.yaml"), "q0", tmp_path))
    (root,) = collection.trajectories.values()
    assert (root.reward, root.misc["outcome"]["status"]) == (reward, "scored")


def test_task_table_in_python():
    table = task_table(
        [
            {"id": "add", "goal": "What is 2 + 3?", "answer": "5"},
            {"id": "mul", "goal": "What is 2 * 3?", "answer": "6"},
            "Name a prime number.",  # a bare goal: id task-2, split train
        ],
        splits={"train": ["add", "task-2"], "test": ["mul"]},
    )

    async def check():
        task = await table.get_task("add")
        session = await table.open(task, rollout_id="r")
        return await table.list_tasks("train"), task, await session.tools(), await session.grade()

    ids, task, tools, outcome = asyncio.run(check())
    assert ids == ["add", "task-2"] and task.misc == {"answer": "5"} and tools == []
    assert (outcome.status, outcome.reason, outcome.trainable) == ("invalid", "self_grade", False)
    assert isinstance(task, pt.Task)
