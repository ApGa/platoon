"""examples/subagents: ``platoon rollout`` runs the example's config with a scripted policy that delegates.

The policy plays the Vault tasks the way the tutorial describes (docs/tutorials/subagents.md): the root splits its
keys between two subagents in one reply, each subagent reads its keys and answers, and the root answers the sum of
their answers.  Overrides switch the tree settings the tutorial covers.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from vault_env import VaultEnvironment, goal_for, parse_keys

from platoon.harness.verify import SUBAGENT_REWARD_JUDGMENT_MISC_KEY

CONFIG = "examples/subagents/subagents.yaml"
TASK = "train:0"
_FIRST_USER = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)
_TOOL_RESPONSES = re.compile(r"<tool_response>\n(.*?)\n</tool_response>", re.S)


def tool_call(name: str, **arguments: Any) -> str:
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': arguments})}\n</tool_call>"


def vault_policy(prompt: str) -> str:
    """A policy that delegates halves of a large task, and reads and adds up a small one."""

    goal = _FIRST_USER.search(prompt).group(1)
    keys = parse_keys(goal)
    responses = _TOOL_RESPONSES.findall(prompt)
    if goal.startswith("You are a verifier agent"):
        if not responses:
            return "\n".join(tool_call("read", key=key) for key in keys)
        claimed = int(re.search(r"Child Final Message:\nAnswered (-?\d+)", goal).group(1))
        verified = claimed == sum(int(value) for value in responses)
        verdict = {"status": "verified" if verified else "failed", "score": 1.0 if verified else 0.0}
        return json.dumps({**verdict, "summary": f"read {len(keys)} values"})
    can_answer = '"name": "answer"' in prompt
    if len(keys) >= 4 and not responses:  # a large task: delegate two halves at once
        half = len(keys) // 2
        return "\n".join(tool_call("delegate", goal=goal_for(part)) for part in (keys[:half], keys[half:]))
    if len(keys) >= 4 and all("did not finish" not in response for response in responses[:2]):
        total = sum(int(re.findall(r"-?\d+", response)[-1]) for response in responses[:2])
        return tool_call("answer", total=total)
    reads = [response for response in responses if re.fullmatch(r"\d+", response)]
    if len(reads) < len(keys):  # a small task, or the subagents failed: read the keys
        return "\n".join(tool_call("read", key=key) for key in keys)
    total = sum(int(value) for value in reads)
    return tool_call("answer", total=total) if can_answer else f"The sum is {total}."


def expected_sums() -> tuple[int, int, int]:
    """The root's sum and its two halves' sums for TASK."""

    (spec,) = [spec for spec in VaultEnvironment.list_tasks("train") if spec["id"] == "train-0"]
    keys, vault = spec["keys"], spec["vault"]
    half = len(keys) // 2
    return (
        sum(vault[key] for key in keys),
        sum(vault[key] for key in keys[:half]),
        sum(vault[key] for key in keys[half:]),
    )


def rollout(platoon_cli, tmp_path, capsys, *overrides: str) -> tuple[str, dict[str, Any]]:
    platoon_cli("rollout", CONFIG, *overrides, "--task", TASK, "--output-dir", str(tmp_path))
    collection = json.loads((tmp_path / "trajectory_collection.json").read_text())
    return capsys.readouterr().out, collection


def trajectories(collection: dict[str, Any]) -> list[dict[str, Any]]:
    return list(collection["trajectories"].values())


def test_the_root_delegates_and_every_subagent_is_graded_by_its_own_answer(serve_policy, platoon_cli, tmp_path, capsys):
    sampler = serve_policy(vault_policy)
    out, collection = rollout(platoon_cli, tmp_path, capsys)

    total, first, second = expected_sums()
    root, *children = trajectories(collection)
    assert root["reward"] == 1.0 and root["misc"]["outcome"]["status"] == "scored"
    assert [child["misc"]["agent"]["role"] for child in children] == ["subagent", "subagent"]
    assert sorted(child["finish_message"] for child in children) == sorted(
        [f"Answered {first}.", f"Answered {second}."]
    )
    assert all(child["reward"] == 1.0 and child["misc"]["outcome"]["status"] == "scored" for child in children)
    assert root["steps"][0]["misc"]["reward_misc"] == {"reward/subagent_launched": 2, "reward/subagent_succeeded": 2.0}
    assert root["finish_message"] == f"Answered {total}."
    # The tree as `platoon rollout` prints it.
    assert "agents 3  steps 6  max depth 1  reward 1.0  outcome scored reward=1.0" in out
    assert out.count("└─ subagent") + out.count("├─ subagent") == 2
    assert len(sampler.prompts) == 6  # the root's two turns and each subagent's two


def test_children_that_share_the_episode_outcome(serve_policy, platoon_cli, tmp_path, capsys):
    serve_policy(vault_policy)
    overrides = ("environments.0.agents.child_grading=none", "environments.0.agents.child_access=no_terminal")
    _, collection = rollout(platoon_cli, tmp_path, capsys, *overrides)

    root, *children = trajectories(collection)
    assert root["reward"] == 1.0 and len(children) == 2
    # No answer tool, so they report in text; ungraded, they take the root's reward.
    assert all(child["finish_message"].startswith("The sum is") for child in children)
    assert all(child["reward"] == 1.0 and "outcome" not in child["misc"] for child in children)


def test_verifier_agents_judge_each_subagent(serve_policy, platoon_cli, tmp_path, capsys):
    sampler = serve_policy(vault_policy)
    # With linear budgets a verifier's max_steps + 1 is reserved on the parent too: keep it inside the root's 20.
    overrides = ("environments.0.agents.child_grading=verifier", "environments.0.agents.verifier.max_steps=4")
    _, collection = rollout(platoon_cli, tmp_path, capsys, *overrides)

    by_role: dict[str, list[dict[str, Any]]] = {}
    for trajectory in trajectories(collection)[1:]:
        by_role.setdefault(trajectory["misc"]["agent"]["role"], []).append(trajectory)
    assert len(by_role["subagent"]) == 2 and len(by_role["verifier"]) == 2
    for child in by_role["subagent"]:
        judgment = child["misc"][SUBAGENT_REWARD_JUDGMENT_MISC_KEY]
        assert judgment["status"] == "verified" and judgment["training_eligible"] is True
        assert child["reward"] == 1.0
    # Verifiers read (read_only access: no answer tool) and never train.
    assert all(verifier["misc"]["exclude_from_training"] is True for verifier in by_role["verifier"])
    assert all(
        {call["name"] for call in verifier["misc"]["tool_calls"]} == {"read"} for verifier in by_role["verifier"]
    )
    verifier_prompts = [prompt for prompt in sampler.prompts if "You are a verifier agent" in prompt]
    assert verifier_prompts and all('"name": "answer"' not in prompt for prompt in verifier_prompts)


def test_a_rejected_launch_comes_back_as_a_tool_result(serve_policy, platoon_cli, tmp_path, capsys):
    serve_policy(vault_policy)
    _, collection = rollout(platoon_cli, tmp_path, capsys, "environments.0.agents.max_depth=0")

    (root,) = trajectories(collection)  # no subagent ran
    assert root["reward"] == 1.0 and root["finish_message"].startswith("Answered")  # the root read the keys itself
    assert "reward_misc" not in root["steps"][0]["misc"]  # a rejected launch is not a launched subagent


@pytest.mark.parametrize("fork", ["fresh", "copy"])
def test_other_fork_modes_grade_children_on_the_whole_task(serve_policy, platoon_cli, tmp_path, capsys, fork):
    serve_policy(vault_policy)
    _, collection = rollout(platoon_cli, tmp_path, capsys, f"environments.0.agents.fork={fork}")

    root, *children = trajectories(collection)
    assert root["reward"] == 1.0 and len(children) == 2
    # A fresh or copied environment is the whole task, not the subgoal: a half-sum answer scores 0 there.
    assert all(child["reward"] == 0.0 and child["misc"]["outcome"]["status"] == "scored" for child in children)
