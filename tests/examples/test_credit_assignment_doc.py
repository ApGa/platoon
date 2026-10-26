"""The numbers docs/tutorials/credit-assignment.md shows: per-agent rewards of the subagents example with and
without ``propagate_root_success``, and the group baselines of its worked example."""

from __future__ import annotations

import json
import re

import pytest
from test_subagents_example import CONFIG, TASK, tool_call, vault_policy
from vault_env import parse_keys

from platoon.train.credit.group import group_baselines


def careless_root(prompt: str) -> str:
    """The subagents' policy, except that the root adds one to the sum of its subagents' answers."""

    reply = vault_policy(prompt)
    goal = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt, re.S).group(1)
    if len(parse_keys(goal)) >= 4 and '"name": "answer"' in reply:
        total = json.loads(reply.split("\n")[1])["arguments"]["total"]
        return tool_call("answer", total=total + 1)
    return reply


@pytest.mark.parametrize(("propagate", "child_rewards"), [(False, [1.0, 1.0]), (True, [0.0, 0.0])])
def test_children_keep_their_own_grade_unless_root_success_propagates(
    serve_policy, platoon_cli, tmp_path, capsys, propagate, child_rewards
):
    serve_policy(careless_root)
    override = f"train.workflow_config.rollout_config.propagate_root_success={str(propagate).lower()}"
    platoon_cli("rollout", CONFIG, override, "--task", TASK, "--output-dir", str(tmp_path))
    out = capsys.readouterr().out
    root, *children = json.loads((tmp_path / "trajectory_collection.json").read_text())["trajectories"].values()

    assert root["reward"] == 0.0 and [child["reward"] for child in children] == child_rewards
    # A child's own grade stays in its outcome; propagation changes the reward it trains on.
    assert all(child["misc"]["outcome"]["rewards"] == {"reward": 1.0} for child in children)
    assert out.count("subagent") == 2


def test_the_worked_example_baselines():
    rewards, completed = [1.0, 0.0, 1.0, 1.0], [True] * 4
    assert group_baselines(rewards, completed, leave_one_out=False) == [0.75] * 4
    assert group_baselines(rewards, completed, leave_one_out=True) == pytest.approx([2 / 3, 1.0, 2 / 3, 2 / 3])
    # An interrupted root (the fourth) is left out of every baseline and takes the completed roots' mean.
    interrupted = [True, True, True, False]
    assert group_baselines(rewards, interrupted, leave_one_out=False) == pytest.approx([2 / 3] * 4)
    assert group_baselines(rewards, interrupted, leave_one_out=True) == pytest.approx([0.5, 1.0, 0.5, 2 / 3])
