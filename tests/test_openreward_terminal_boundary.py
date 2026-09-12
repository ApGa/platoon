"""Exercise the real SDK event classes with a producer ahead of the consumer."""

import asyncio
import json
from types import SimpleNamespace

import pytest


def _modules(monkeypatch):
    monkeypatch.setenv("OPENHANDS_SUPPRESS_BANNER", "1")
    events = pytest.importorskip("openhands.sdk.event")
    llm = pytest.importorskip("openhands.sdk.llm")
    env = pytest.importorskip("platoon.openreward.env")
    types = pytest.importorskip("platoon.openhands.types")
    utils = pytest.importorskip("platoon.utils.openhands_utils")
    return events, llm, env, types, utils


def _action(events, name, response):
    return events.ActionEvent.model_construct(
        id=name,
        source="agent",
        action=None,
        tool_name=name,
        tool_call_id=f"call-{name}",
        llm_response_id=response,
    )


def _result(events, llm, action, reward=None):
    payload = {"finished": True, "reward": reward} if reward is not None else {"finished": False}
    return events.ObservationEvent.model_construct(
        id=f"result-{action.id}",
        source="environment",
        action_id=action.id,
        tool_call_id=action.tool_call_id,
        tool_name=action.tool_name,
        observation=SimpleNamespace(tool_name=action.tool_name, content=[llm.TextContent(text=json.dumps(payload))]),
    )


def _setup(env_mod, types):
    from platoon.envs.base import Task
    from platoon.episode.trajectory import TrajectoryCollection

    task = Task(id="root", goal="Complete the task", max_steps=200)
    env = env_mod.OpenRewardOpenHandsEnv(task=task, agent=object(), workspace=".")
    state = SimpleNamespace(events=[], execution_status="running")
    env._conversation = SimpleNamespace(state=state)
    env._state = types.OpenHandsObservation(task=task, conversation_state=state)
    collection = TrajectoryCollection()
    trajectory = collection.create_trajectory()
    return env, state, collection, trajectory


def _drain(env, utils, types, collection, trajectory):
    from platoon.episode.context import current_trajectory, current_trajectory_collection, finish_message

    async def run():
        tokens = [
            (current_trajectory, current_trajectory.set(trajectory)),
            (current_trajectory_collection, current_trajectory_collection.set(collection)),
            (finish_message, finish_message.set(None)),
        ]
        try:
            for _ in range(10):
                if env._state.finished:
                    break
                actions = utils.get_actions_for_last_obs(env._state)
                assert actions, "the buffered trajectory should drain without another model call"
                await env.step(
                    types.OpenHandsAction(
                        action_events=actions,
                        misc={"completion_id": actions[-1].llm_response_id},
                    )
                )
            assert env._state.finished
        finally:
            for variable, token in reversed(tokens):
                variable.reset(token)

    asyncio.run(run())


@pytest.mark.parametrize("text", ["Task complete.", ""])
def test_message_is_terminal_zero_and_later_success_cannot_replace_it(monkeypatch, text):
    events, llm, env_mod, types, utils = _modules(monkeypatch)
    env, state, collection, trajectory = _setup(env_mod, types)
    tool = _action(events, "python_execute", "response-tool")
    result = _result(events, llm, tool)
    message = events.MessageEvent(
        source="agent",
        llm_message=llm.Message(role="assistant", content=[llm.TextContent(text=text)]),
        llm_response_id="response-message",
    )
    later = _action(events, "claim_done", "response-too-late")
    later_result = _result(events, llm, later, reward=1.0)
    for event in [tool, result, message, later, later_result]:
        env._stop_on_openreward_finished(event)
        state.events.append(event)  # SDK persists after invoking user callbacks.
    _drain(env, utils, types, collection, trajectory)

    assert [s.misc["action_misc"]["completion_id"] for s in trajectory.steps] == ["response-tool", "response-message"]
    assert trajectory.reward == 0.0
    assert trajectory.steps[-1].misc["reward_misc"]["openreward/message_terminated"] == 1.0
    from platoon.utils.trajectory_status import trajectory_has_forced_zero_reward, trajectory_was_interrupted

    assert trajectory_has_forced_zero_reward(trajectory)
    assert not trajectory_was_interrupted(trajectory)
    assert env._terminal_event_id == message.id


def test_tool_completion_keeps_reward_on_terminal_step_and_bounds_condensations(monkeypatch):
    events, llm, env_mod, types, utils = _modules(monkeypatch)
    env, state, collection, trajectory = _setup(env_mod, types)
    first = _action(events, "python_execute", "response-first")
    first_result = _result(events, llm, first)
    done = _action(events, "claim_done", "response-done")
    done_result = _result(events, llm, done, reward=0.75)
    summary = (
        "USER_CONTEXT: Complete the task.\nCOMPLETED: Created the output.\n"
        "PENDING: Submit it.\nCURRENT_STATE: The output exists."
    )
    before = SimpleNamespace(
        kind="Condensation", id="summary-before", llm_response_id="response-summary", summary=summary
    )
    after = SimpleNamespace(
        kind="Condensation", id="summary-after", llm_response_id="response-too-late", summary=summary
    )
    later_message = events.MessageEvent(
        source="agent",
        llm_message=llm.Message(role="assistant", content=[llm.TextContent(text="Done")]),
        llm_response_id="response-late-message",
    )
    for event in [first, first_result, before, done, done_result, after, later_message]:
        env._stop_on_openreward_finished(event)
        state.events.append(event)
    _drain(env, utils, types, collection, trajectory)

    assert [s.misc["action_misc"]["completion_id"] for s in trajectory.steps] == [
        "response-first",
        "response-summary",
        "response-done",
    ]
    assert [s.reward for s in trajectory.steps] == [0.0, 0.0, 0.75]
    assert trajectory.reward == 0.75
    assert trajectory.steps[-1].misc["reward_misc"]["openreward/message_terminated"] == 0.0
    assert not trajectory.misc.get("trajectory_forced_zero_reward")


def test_child_finish_remains_eligible_for_root_reward_after_late_message(monkeypatch):
    events, llm, env_mod, types, utils = _modules(monkeypatch)
    from openhands.sdk.tool.builtins.finish import FinishAction, FinishObservation

    from platoon.envs.base import SubTask
    from platoon.episode.trajectory import Trajectory, TrajectoryCollection
    from platoon.utils.subagent_rewards import propagate_root_success

    env, state, _, _ = _setup(env_mod, types)
    env._task = SubTask(id="child", goal="Complete the delegated work")
    root = Trajectory(id="root", reward=1.0)
    collection = TrajectoryCollection(trajectories={root.id: root})
    child = collection.create_trajectory(parent_traj=root)
    action = _action(events, "finish", "response-finish").model_copy(
        update={"action": FinishAction(message="Prepared the requested output.")},
    )
    result = events.ObservationEvent.model_construct(
        id="finish-result",
        source="environment",
        tool_name="finish",
        tool_call_id=action.tool_call_id,
        action_id=action.id,
        observation=FinishObservation.from_text(text=action.action.message),
    )
    late = events.MessageEvent(
        source="agent",
        llm_message=llm.Message(role="assistant", content=[llm.TextContent(text="Done")]),
        llm_response_id="response-too-late",
    )
    for event in [action, result, late]:
        env._stop_on_openreward_finished(event)
        state.events.append(event)
    _drain(env, utils, types, collection, child)
    propagate_root_success(collection)
    assert env._terminal_event_id == result.id
    assert len(child.steps) == 1
    assert child.reward == 1.0
    assert not child.misc.get("trajectory_forced_zero_reward")


@pytest.mark.parametrize("text,reasoning", [("Done.", None), ("", None), ("", "Reasoning without a tool call.")])
def test_sdk_producer_stops_after_one_message_without_retry(monkeypatch, tmp_path, text, reasoning):
    events, llm, env_mod, types, utils = _modules(monkeypatch)
    rollout = pytest.importorskip("platoon.openreward.rollout")
    agent_mod = pytest.importorskip("openhands.sdk.agent.agent")
    conversation_mod = pytest.importorskip("openhands.sdk.conversation.conversation")
    calls = []

    async def completion(*args, **kwargs):
        calls.append(1)
        assert len(calls) == 1, "a message response must not trigger another completion"
        return SimpleNamespace(
            id="response-one",
            message=llm.Message(
                role="assistant",
                content=[llm.TextContent(text=text)] if text else [],
                reasoning_content=reasoning,
            ),
        )

    monkeypatch.setattr(agent_mod, "amake_llm_completion", completion)
    agent = rollout.OpenRewardAgent(
        llm=llm.LLM(model="openai/test-model", api_key="test-key"),
        tools=[],
        include_default_tools=[],
    )
    env, _, collection, trajectory = _setup(env_mod, types)
    conversation = conversation_mod.Conversation(
        agent=agent,
        workspace=str(tmp_path),
        callbacks=[env._stop_on_openreward_finished],
        visualizer=None,
        max_iteration_per_run=10,
    )
    env._conversation = conversation
    env._state.conversation_state = conversation.state

    async def run():
        try:
            conversation.send_message("Use the task tools to complete the task.")
            await asyncio.wait_for(conversation.arun(), timeout=15)
        finally:
            conversation.close()

    asyncio.run(run())
    assert len(calls) == 1
    messages = [e for e in conversation.state.events if isinstance(e, events.MessageEvent)]
    assert len([e for e in messages if e.source == "agent"]) == 1
    assert len([e for e in messages if e.source == "user"]) == 1
    env._state.last_step_observation_id = next(e.id for e in messages if e.source == "user")
    _drain(env, utils, types, collection, trajectory)
    assert trajectory.reward == 0.0
    assert len(trajectory.steps) == 1
