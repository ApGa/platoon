"""``examples/custom-environment``: the order-packing ORS environment, its smoke script, its configs, and rollouts
with ``platoon-react`` and with the ``example-packing-team`` harness, whose children run on ``platoon_fork``
sessions."""

from __future__ import annotations

import asyncio
import json
import runpy

import pytest
from example_runs import by_depth, load_config, rollout

NAME = "custom-environment"


def tool_call(name: str, **arguments) -> str:
    """A Qwen3 tool call, as the model writes it."""
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"


def test_check_env_prints_the_tutorials_output(example, capsys):
    example(NAME)
    runpy.run_path("check_env.py", run_name="__main__")
    lines = [line for line in capsys.readouterr().out.splitlines() if "tool_input_validation_error" not in line]
    assert lines == [
        "splits: ['train', 'test']",
        "test tasks: ['test:0', 'test:1']",
        "instruction: Pack this order into the box: 2 apple, 1 bread.",
        "tool done(summary): terminal",
        "tool look(): read_only",
        "tool pack(item, quantity): -",
        lines[6],  # the SDK's error reason: input_validation (0.1.163) or invalid_call (0.1.125)
        "child: Packed 2 apple. The box holds 2 apple.",
        "child grade: scored reward=1.0 reason=None",
        "root: Packed 1 bread. The box holds 2 apple, 1 bread.",
        "root: Shelf: 4 apple, 1 bread, 3 cheese, 12 egg, 2 milk. Box: 2 apple, 1 bread.",
        "root grade: scored reward=1.0 reason=None",
    ]
    assert lines[6] in ("pack 0 apple: error input_validation", "pack 0 apple: error invalid_call")


@pytest.mark.parametrize("config", ["train.yaml", "train_team.yaml", "remote.yaml"])
def test_the_configs_validate(example, config):
    example(NAME)
    assert load_config(config).environments[0].name == "packing"


def test_platoon_react_packs_an_order_and_its_final_answer_grades_the_box(example, policy_gateway, tmp_path):
    example(NAME)

    def policy(message: str) -> str:
        if message.startswith("Pack this order"):
            return tool_call("pack", item="apple", quantity=2)
        if "Packed 2 apple" in message:
            return tool_call("pack", item="bread", quantity=1)
        return "Packed 2 apple and 1 bread."

    policy_gateway(policy)
    collection = asyncio.run(rollout(load_config("train.yaml"), "train:0", tmp_path))

    (root,) = collection.trajectories.values()
    assert (root.reward, root.misc["outcome"]["status"]) == (1.0, "scored")
    assert [call["name"] for call in root.misc["tool_calls"]] == ["pack", "pack", "done"]
    assert root.misc["tool_calls"][-1]["arguments"] == {"summary": "Packed 2 apple and 1 bread."}


def test_a_wrong_box_scores_zero(example, policy_gateway, tmp_path):
    example(NAME)
    policy_gateway(lambda message: tool_call("pack", item="egg") if message.startswith("Pack") else "Done.")
    collection = asyncio.run(rollout(load_config("train.yaml"), "train:0", tmp_path))
    (root,) = collection.trajectories.values()
    assert (root.reward, root.misc["outcome"]["status"]) == (0.0, "scored")


def test_packing_team_children_pack_on_forked_sessions_and_are_graded_by_them(example, policy_gateway, tmp_path):
    example(NAME)

    def policy(message: str) -> str:
        if message.startswith("Pack this order"):  # the root
            return tool_call("delegate", goal="Pack 2 apple") + "\n" + tool_call("delegate", goal="Pack 1 bread")
        if message == "Pack 2 apple":  # the children
            return tool_call("pack", item="apple", quantity=2)
        if message == "Pack 1 bread":
            return tool_call("pack", item="bread", quantity=1)
        if "Packed 1 bread" in message or "Packed 2 apple" in message:
            return "My part is packed."
        return "The helpers packed the order."  # the root, after its helpers reported

    sampler = policy_gateway(policy)
    collection = asyncio.run(rollout(load_config("train_team.yaml"), "train:0", tmp_path))

    root, *children = by_depth(collection)
    assert (root.reward, root.misc["outcome"]["status"]) == (1.0, "scored")
    assert [call["name"] for call in root.misc["tool_calls"]] == ["done"]  # the children did the packing
    assert sorted(child.task.goal for child in children) == ["Pack 1 bread", "Pack 2 apple"]
    assert [child.reward for child in children] == [1.0, 1.0]  # each graded by its own done()
    assert all(child.misc["outcome"]["status"] == "scored" for child in children)
    assert sampler.calls[0].startswith("Pack this order")
