"""Gateway traces: agent records -> TrajectoryCollection -> the unchanged prefix-merging datum builder."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import pytest
from gateway_fakes import TOOLS, WEATHER_CALL, Completion

from platoon.gateway.core import ChatRequest
from platoon.gateway.trace import build_trajectory_collection
from platoon.task import SubTask, Task
from platoon.trajectory import ParentInfo
from platoon.utils.critical_path import MODEL_CALL_SPANS_KEY
from platoon.utils.generation import MODEL_GENERATIONS_KEY
from platoon.utils.tinker_data_processing import AgentTag, get_train_data_for_trajectory_collection
from platoon.utils.trajectory_status import trajectory_was_interrupted

SYSTEM = {"role": "system", "content": "You are a helpful assistant."}
USER = {"role": "user", "content": "What's the weather in Paris?"}
TASK = Task(goal="What's the weather in Paris?", id="weather-1")


@dataclass
class Outcome:
    """Mirrors ``platoon.envs.service.Outcome``."""

    status: str
    rewards: dict[str, float] = field(default_factory=dict)
    primary: str = "reward"
    reason: str | None = None

    @property
    def reward(self) -> float | None:
        return self.rewards.get(self.primary) if self.status == "scored" else None


def chat(gateway, agent, messages, **params):
    return asyncio.run(gateway.chat(agent, ChatRequest.from_params(messages, params)))


def record_root_and_helper(gateway, sampler):
    """A root agent that calls a tool, a helper it spawns, and two more root turns."""
    sampler.script = [
        Completion(WEATHER_CALL),
        Completion("Forecast: sunny."),
        Completion("Sunny in Paris."),
        Completion("You're welcome!"),
    ]
    interactions: dict = {}
    root = gateway.open_agent(rollout_id="rollout-7", trajectory_id="root-trajectory", interactions=interactions)
    messages = [SYSTEM, USER]
    calls = [chat(gateway, root, messages, tools=TOOLS)]
    tool_call_id = calls[0].tool_calls[0]["id"]
    helper = gateway.open_subagent(root, role="researcher", goal="Check the forecast")
    helper_call = chat(gateway, helper, [{"role": "user", "content": "Check the forecast"}])
    messages += [calls[0].message(), {"role": "tool", "tool_call_id": tool_call_id, "content": "Sunny, 25C"}]
    calls.append(chat(gateway, root, messages, tools=TOOLS))
    messages += [calls[1].message(), {"role": "user", "content": "Thanks!"}]
    calls.append(chat(gateway, root, messages, tools=TOOLS))
    return root, helper, interactions, calls, helper_call


def tokens(datum, key):
    return datum.loss_fn_inputs[key].to_torch().tolist()


def test_multi_turn_trace_becomes_prefix_merged_datums(gateway, sampler):
    root, helper, interactions, calls, helper_call = record_root_and_helper(gateway, sampler)
    collection = build_trajectory_collection(
        gateway.agents("rollout-7"), task=TASK, outcome=Outcome("scored", {"reward": 1.0})
    )
    result = get_train_data_for_trajectory_collection(
        trajectory_collection=collection.to_dict(),
        interactions=interactions,
        task_id=TASK.id,
        checkpoint_version=3,
        include_traj_depth=True,
    )

    prompts = [call.prompt.to_ints() for call in calls]
    # The cookbook renderer re-renders a history tool call with a leading newline, so the second prompt does not
    # extend the first call (a new datum), while the plain-text third turn extends the second (merged).
    assert prompts[1][: len(prompts[0]) + len(calls[0].tokens)] != prompts[0] + calls[0].tokens
    assert prompts[2][: len(prompts[1]) + len(calls[1].tokens)] == prompts[1] + calls[1].tokens

    first, merged, helper_datum = result.datums
    sequence = prompts[0] + calls[0].tokens
    assert first.model_input.to_ints() == sequence[:-1]
    assert tokens(first, "target_tokens") == sequence[1:]
    assert tokens(first, "mask") == [0.0] * (len(prompts[0]) - 1) + [1.0] * len(calls[0].tokens)

    sequence = prompts[2] + calls[2].tokens
    assert merged.model_input.to_ints() == sequence[:-1]
    expected_mask = (
        [0.0] * len(prompts[1])
        + [1.0] * len(calls[1].tokens)
        + [0.0] * (len(prompts[2]) - len(prompts[1]) - len(calls[1].tokens))
        + [1.0] * len(calls[2].tokens)
    )
    assert tokens(merged, "mask") == expected_mask[1:]
    assert tokens(merged, "logprobs") == [-0.25 if m else 0.0 for m in expected_mask[1:]]
    assert tokens(merged, "advantages") == expected_mask[1:]  # reward 1.0 before group centering
    assert tokens(merged, "checkpoint_version") == [3]

    sequence = helper_call.prompt.to_ints() + helper_call.tokens
    assert helper_datum.model_input.to_ints() == sequence[:-1]
    assert [tokens(d, "traj_depth") for d in result.datums] == [[0], [0], [1]]
    assert result.task_reward == 1.0 and result.task_reward_valid
    assert [(s.num_steps, s.is_root, s.reward) for s in result.trajectory_stats] == [(3, True, 1.0), (1, False, 1.0)]


def test_a_step_annotated_exclude_from_training_contributes_no_datum(gateway, sampler):
    root, helper, interactions, calls, helper_call = record_root_and_helper(gateway, sampler)
    # e.g. a harness rejecting a condensation summary: the call stays a step, but nothing trains on it
    root.step_misc[calls[0].call_id] = {"exclude_from_training": True}
    helper.step_misc[helper_call.call_id] = {"exclude_from_training": True}
    collection = build_trajectory_collection(
        gateway.agents("rollout-7"), task=TASK, outcome=Outcome("scored", {"reward": 1.0})
    )
    trajectories = list(collection.trajectories.values())
    assert [len(t.steps) for t in trajectories] == [3, 1]
    assert trajectories[0].steps[0].misc["exclude_from_training"] is True

    result = get_train_data_for_trajectory_collection(
        trajectory_collection=collection.to_dict(),
        interactions=interactions,
        task_id=TASK.id,
        checkpoint_version=3,
        include_traj_depth=True,
    )
    (merged,) = result.datums  # the first root call's datum and the helper's datum are gone
    sequence = calls[2].prompt.to_ints() + calls[2].tokens
    assert merged.model_input.to_ints() == sequence[:-1]
    assert [(s.num_steps, s.is_root) for s in result.trajectory_stats] == [(2, True), (0, False)]


def test_tagged_datums_name_the_agent_of_each_and_every_agent_of_the_tree(gateway, sampler):
    """Under ``tag_datums`` (a credit estimator that reads agents), each datum names its trajectory, and each
    trajectory its agent, role, parent, depth and reward; without it, the result carries no tags."""

    root, helper, interactions, calls, helper_call = record_root_and_helper(gateway, sampler)
    collection = build_trajectory_collection(
        gateway.agents("rollout-7"), task=TASK, outcome=Outcome("scored", {"reward": 1.0})
    ).to_dict()

    def converted(**options):
        return get_train_data_for_trajectory_collection(
            trajectory_collection=collection,
            interactions=interactions,
            task_id="weather-1",
            checkpoint_version=3,
            **options,
        )

    tagged = converted(tag_datums=True)
    assert tagged.datum_trajectories == ("root-trajectory", "root-trajectory", helper.trajectory_id)
    assert tagged.agents == (
        AgentTag("root-trajectory", root.agent_id, "policy", None, 0, 1.0),
        AgentTag(helper.trajectory_id, helper.agent_id, "researcher", "root-trajectory", 1, 1.0),
    )
    assert "traj_depth" not in tagged.datums[0].loss_fn_inputs  # tags never reach the datums
    plain = converted()
    assert plain.agents is None and plain.datum_trajectories is None
    assert [datum.loss_fn_inputs.keys() for datum in plain.datums] == [
        datum.loss_fn_inputs.keys() for datum in tagged.datums
    ]


def test_trace_links_agents_calls_and_outcome(gateway, sampler):
    root, helper, interactions, calls, helper_call = record_root_and_helper(gateway, sampler)
    agents = gateway.close_rollout("rollout-7")
    collection = build_trajectory_collection(
        reversed(agents), task=TASK, outcome=Outcome("scored", {"reward": 0.5, "efficiency": 0.1})
    )

    assert collection.id == "rollout-7"
    assert list(collection.trajectories) == ["root-trajectory", helper.trajectory_id]
    root_trajectory = collection.trajectories["root-trajectory"]
    helper_trajectory = collection.trajectories[helper.trajectory_id]
    assert root_trajectory.task is TASK and root_trajectory.parent_info is None
    assert helper_trajectory.parent_info == ParentInfo(id="root-trajectory", fork_step=1)
    assert isinstance(helper_trajectory.task, SubTask) and helper_trajectory.task.goal == "Check the forecast"
    assert helper_trajectory.task.parent_tasks == [TASK]
    assert [step.misc["action_misc"]["completion_id"] for step in root_trajectory.steps] == root.call_ids
    assert root_trajectory.steps[0].misc["message"]["tool_calls"] == calls[0].tool_calls
    assert root_trajectory.steps[0].misc["action_misc"]["usage"] == calls[0].usage()
    assert root_trajectory.reward == helper_trajectory.reward == 0.5
    assert root_trajectory.misc["outcome"] == {
        "status": "scored",
        "rewards": {"reward": 0.5, "efficiency": 0.1},
        "reason": None,
        "reward_key": "reward",
    }
    assert "outcome" not in helper_trajectory.misc
    assert helper_trajectory.misc["agent"] == {"agent_id": helper.agent_id, "role": "researcher"}
    generations = root_trajectory.misc[MODEL_GENERATIONS_KEY]
    assert list(generations) == root.call_ids
    assert generations[root.call_ids[0]]["finish_reason"] == "tool_calls"
    for start, end in root_trajectory.misc[MODEL_CALL_SPANS_KEY].values():
        assert start <= end
    json.dumps(collection.to_dict())  # serializable as a rollout result


@pytest.mark.parametrize("status", ["invalid", "error", "timeout", "cancelled"])
def test_untrainable_outcomes_never_train_as_reward_zero(gateway, sampler, status):
    _, _, interactions, _, _ = record_root_and_helper(gateway, sampler)
    collection = build_trajectory_collection(
        gateway.agents("rollout-7"), task=TASK, outcome=Outcome(status, reason="judge_unavailable")
    )
    trajectories = collection.to_dict()["trajectories"]
    assert all(trajectory_was_interrupted(trajectory) for trajectory in trajectories.values())
    assert all(trajectory["reward"] == 0.0 for trajectory in trajectories.values())
    assert trajectories["root-trajectory"]["error_message"] == f"Outcome {status}: judge_unavailable"

    result = get_train_data_for_trajectory_collection(
        trajectory_collection=collection.to_dict(), interactions=interactions, task_id=TASK.id, checkpoint_version=0
    )
    assert result.datums == [] and result.num_policy_excluded_datums == 3
    assert not result.task_reward_valid


def test_trace_rejects_malformed_trees_and_outcomes(gateway, sampler):
    root, helper, _, _, _ = record_root_and_helper(gateway, sampler)
    scored = Outcome("scored", {"reward": 1.0})
    with pytest.raises(ValueError, match="exactly one root"):
        build_trajectory_collection([helper], task=TASK, outcome=scored)
    with pytest.raises(ValueError, match="exactly one root"):
        build_trajectory_collection([root, root], task=TASK, outcome=scored)
    stranger = gateway.open_subagent(helper, role="grandchild")
    with pytest.raises(ValueError, match="parent is not in the trace"):
        build_trajectory_collection([root, stranger], task=TASK, outcome=scored)
    with pytest.raises(ValueError, match="Unknown outcome status"):
        build_trajectory_collection([root], task=TASK, outcome=Outcome("graded"))
    with pytest.raises(ValueError, match="must carry a reward"):
        build_trajectory_collection([root], task=TASK, outcome=Outcome("scored"))


def test_a_trajectory_that_records_no_agent_is_its_own_agent_and_an_interrupted_one_is_unscored():
    """Under ``tag_datums``, a trajectory a rollout wrote outside the gateway (no agent in its misc) is its own agent,
    of the root's default role; an interrupted one has no reward, which its rows never train as 0."""

    from platoon.utils.tinker_data_processing import _agent_tag
    from platoon.utils.trajectory_status import TRAJECTORY_CANCELLED_MISC_KEY

    assert _agent_tag("t1", {"misc": {}, "parent_info": {"id": "t0"}}, 1, 0.5) == AgentTag(
        "t1", "t1", "policy", "t0", 1, 0.5
    )
    cancelled = {"misc": {TRAJECTORY_CANCELLED_MISC_KEY: True, "agent": {"agent_id": "a", "role": "helper"}}}
    assert _agent_tag("t2", cancelled, 1, 0.3) == AgentTag("t2", "a", "helper", None, 1, None)


@pytest.mark.parametrize("verdict", ["judge_error", "unparseable", "pending", "verified", "parent_did_not_use"])
def test_a_child_whose_verifier_gave_no_verdict_is_unscored_and_one_its_parent_did_not_use_keeps_its_reward(verdict):
    """Under ``tag_datums``, a judged child whose verifier gave no verdict (a judge error, an unparseable or a pending
    one) has no reward: its 0 is no score, which a credit estimator would count in a baseline.  A child its parent did
    not use is left out of policy training too, but its verdict, and so its reward, stands."""

    from platoon.harness import verify
    from platoon.utils.tinker_data_processing import _agent_tag
    from platoon.utils.trajectory_status import EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY

    judgments = {
        "judge_error": verify.judge_error_judgment("the verifier could not start"),
        "unparseable": verify.normalize_judgment(
            "no JSON here", verifier_trajectory_id="v", verifier_error_message=None, verifier_finished=True
        ),
        "pending": dict(verify.PENDING_JUDGMENT),
        "verified": verify.normalize_judgment(
            '{"status": "verified", "score": 1.0}',
            verifier_trajectory_id="v",
            verifier_error_message=None,
            verifier_finished=True,
        ),
    }
    judgment = judgments.get(verdict, judgments["verified"])
    misc = {verify.SUBAGENT_REWARD_JUDGMENT_MISC_KEY: judgment, "agent": {"agent_id": "c", "role": "subagent"}}
    if verdict != "verified":  # what record_judgment and the parent-usage judge mark
        misc[EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY] = True
    reward = 1.0 if judgment["score"] else 0.0
    tag = _agent_tag("t1", {"misc": misc, "parent_info": {"id": "t0"}}, 1, reward)
    scored = verdict in ("verified", "parent_did_not_use")
    assert tag == AgentTag("t1", "c", "subagent", "t0", 1, 1.0 if scored else None)
