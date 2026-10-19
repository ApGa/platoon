"""platoon.testing.run_episode: an environment, a harness and an episode program end to end, with a scripted policy
(the toys are in ``testing_toys.py``)."""

from __future__ import annotations

import asyncio

import pytest
from testing_toys import LOCKBOX

from platoon.testing import ScriptedPolicy, make_gateway, rules, run_episode, tool_call
from platoon.train.tinker.proxy import proxy_interactions


def run(task: str = "test:1", **kwargs):
    return asyncio.run(run_episode(task, service=LOCKBOX, **kwargs))


def test_an_environment_runs_with_a_built_in_harness():
    policy = ScriptedPolicy([tool_call("unlock", code="7"), tool_call("unlock", code="1")])

    episode = run(harness="platoon-react", policy=policy)

    assert episode.reward == 1.0 and episode.outcome["status"] == "scored"
    assert episode.root.finish_message == "unlocked" and episode.children == []
    assert policy.calls[1].last_user_message == "<tool_response>\nwrong code\n</tool_response>"
    assert episode.policy is policy
    # The recorded calls are the exact prompts and sampled tokens, which is what trains.
    recorded = list(episode.interactions.values())
    assert [interaction.action.tokens for interaction in recorded] == [call.tokens for call in policy.calls]
    assert [interaction.obs.to_ints() for interaction in recorded] == [call.prompt.to_ints() for call in policy.calls]


def test_a_function_harness_spawns_a_subagent_that_is_graded_by_its_session():
    policy = ScriptedPolicy(rules({"Open the box": "DELEGATE", "Find the code": "1"}))

    episode = run(harness="testing-delegator", policy=policy, agents={"child_grading": "env", "max_depth": 1})

    (child,) = episode.children
    assert child.parent_info.id == episode.root.id and child.task.goal == "Find the code and unlock the box."
    assert (child.reward, episode.reward) == (1.0, 1.0)
    assert episode.root.finish_message == "unlocked"


def test_a_program_and_its_finalizers_run():
    episode = run(harness="testing-delegator", program="testing-tagged", policy=["1"])

    assert episode.reward == 1.0 and episode.root.misc["tagged"] is True
    assert episode.root.misc["harness"]["program"] == "testing-tagged"


def test_the_step_budget_ends_the_agent():
    policy = ScriptedPolicy(lambda prompt: tool_call("unlock", code="9"))

    episode = run(harness="platoon-react", policy=policy, max_steps=2)

    assert len(policy.calls) == 2 and episode.reward == 0.0
    assert episode.root.misc["agent_result"]["status"] == "limit"


def test_a_gateway_and_rollout_settings_are_used():
    policy = ScriptedPolicy([tool_call("unlock", code="1")])
    gateway = make_gateway(policy, temperature=0.3)

    episode = run(harness="platoon-react", policy=gateway, rollout={"inference_params": {"max_completion_tokens": 32}})

    assert episode.gateway is gateway and episode.reward == 1.0
    (call,) = policy.calls
    assert (call.sampling_params.temperature, call.sampling_params.max_tokens) == (0.3, 32)


def test_the_episode_writes_where_it_is_told_and_leaves_no_interaction_context(tmp_path):
    episode = run(harness="platoon-react", policy=[tool_call("unlock", code="1")], output_dir=tmp_path / "out")

    assert episode.output_dir == tmp_path / "out"
    with pytest.raises(LookupError):
        proxy_interactions.get()


def test_a_harness_that_crashes_is_an_untrainable_outcome():
    episode = run(harness="platoon-react", policy=[])  # the policy has nothing to say: its first call fails

    assert episode.outcome["status"] == "error"
