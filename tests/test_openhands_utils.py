from types import SimpleNamespace

import pytest


def _parser_modules(monkeypatch):
    monkeypatch.setenv("OPENHANDS_SUPPRESS_BANNER", "1")
    events = pytest.importorskip("openhands.sdk.event")
    openhands_types = pytest.importorskip("platoon.openhands.types")
    openhands_utils = pytest.importorskip("platoon.utils.openhands_utils")
    return events, openhands_types, openhands_utils


def _action_event(events, event_id: str, tool_call_id: str, response_id: str):
    return events.ActionEvent.model_construct(
        id=event_id,
        source="agent",
        thought=[],
        action=None,
        tool_name="test_tool",
        tool_call_id=tool_call_id,
        tool_call=None,
        llm_response_id=response_id,
    )


def _agent_error_event(events, event_id: str, tool_call_id: str):
    return events.AgentErrorEvent.model_construct(
        id=event_id,
        source="agent",
        tool_name="test_tool",
        tool_call_id=tool_call_id,
        error="invalid tool call",
    )


def _observation_event(events, event_id: str, action_id: str, tool_call_id: str):
    return events.ObservationEvent.model_construct(
        id=event_id,
        source="environment",
        tool_name="test_tool",
        tool_call_id=tool_call_id,
        action_id=action_id,
        observation=None,
    )


def test_pending_action_polling_does_not_write_stdout(capsys, monkeypatch):
    monkeypatch.setenv("OPENHANDS_SUPPRESS_BANNER", "1")
    litellm = pytest.importorskip("litellm")
    utils = pytest.importorskip("litellm.types.utils")
    events = pytest.importorskip("openhands.sdk.event")
    llm = pytest.importorskip("openhands.sdk.llm")
    openhands_types = pytest.importorskip("platoon.openhands.types")
    openhands_utils = pytest.importorskip("platoon.utils.openhands_utils")

    tool_call = litellm.ChatCompletionMessageToolCall(
        id="call_1",
        type="function",
        function=utils.Function(name="test_tool", arguments="{}"),
    )
    action = events.ActionEvent(
        source="agent",
        thought=[llm.TextContent(text="pending")],
        action=None,
        tool_name="test_tool",
        tool_call_id="call_1",
        tool_call=llm.MessageToolCall.from_chat_tool_call(tool_call),
        llm_response_id="response_1",
    )
    capsys.readouterr()

    observation = openhands_types.OpenHandsObservation(
        conversation_state=SimpleNamespace(events=[action], execution_status=None),
    )

    assert openhands_utils.get_actions_for_last_obs(observation) == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_interleaved_agent_error_does_not_split_parallel_tool_batch(monkeypatch):
    events, openhands_types, openhands_utils = _parser_modules(monkeypatch)

    initial = events.MessageEvent.model_construct(id="initial", source="user")
    action_1 = _action_event(events, "action-1", "call-1", "response-1")
    action_2 = _action_event(events, "action-2", "call-2", "response-1")
    action_3 = _action_event(events, "action-3", "call-3", "response-1")
    error_2 = _agent_error_event(events, "error-2", "call-2")
    observation_1 = _observation_event(events, "observation-1", "action-1", "call-1")
    observation_3 = _observation_event(events, "observation-3", "action-3", "call-3")
    next_action = _action_event(events, "action-4", "call-4", "response-2")

    observation = openhands_types.OpenHandsObservation(
        conversation_state=SimpleNamespace(
            events=[
                initial,
                action_1,
                action_2,
                error_2,
                action_3,
                observation_1,
                observation_3,
                next_action,
            ],
            execution_status=None,
        ),
        last_step_observation_id=initial.id,
    )

    step_actions = openhands_utils.get_actions_for_last_obs(observation)
    assert [event.id for event in step_actions] == ["action-1", "action-2", "action-3"]

    observation.last_step_action_id = step_actions[-1].id
    step_observations = openhands_utils.get_obs_for_last_action(observation, step_actions)
    assert [event.id for event in step_observations] == ["observation-1", "error-2", "observation-3"]


def test_condensation_between_action_batches_is_observed_exactly_once(monkeypatch):
    events, openhands_types, openhands_utils = _parser_modules(monkeypatch)
    condenser_events = pytest.importorskip("openhands.sdk.event.condenser")

    initial = events.MessageEvent.model_construct(id="initial", source="user")
    action_1 = _action_event(events, "action-1", "call-1", "response-1")
    observation_1 = _observation_event(events, "observation-1", "action-1", "call-1")
    condensation = condenser_events.Condensation.model_construct(
        id="condensation-1",
        source="environment",
        forgotten_event_ids=set(),
        summary=(
            "USER_CONTEXT: Fix the parser.\n"
            "COMPLETED: Located the implementation.\n"
            "PENDING: Apply and test the patch.\n"
            "CURRENT_STATE: No files changed yet."
        ),
        summary_offset=0,
        llm_response_id="response-condensation",
    )
    action_2 = _action_event(events, "action-2", "call-2", "response-2")
    observation_2 = _observation_event(events, "observation-2", "action-2", "call-2")
    action_3 = _action_event(events, "action-3", "call-3", "response-3")

    event_stream = [
        initial,
        action_1,
        observation_1,
        condensation,
        action_2,
        observation_2,
        action_3,
    ]
    observation = openhands_types.OpenHandsObservation(
        conversation_state=SimpleNamespace(events=event_stream, execution_status=None),
        last_step_observation_id=initial.id,
    )

    first_actions = openhands_utils.get_actions_for_last_obs(observation)
    assert [event.id for event in first_actions] == [action_1.id]
    observation.last_step_action_id = first_actions[-1].id

    first_observations = openhands_utils.get_obs_for_last_action(
        observation,
        first_actions,
    )
    assert [event.id for event in first_observations] == [
        observation_1.id,
        condensation.id,
    ]

    first_observation_ids = {event.id for event in first_observations}
    observation.last_step_observation_id = next(
        event.id
        for event in reversed(event_stream)
        if event.id in first_observation_ids
    )
    second_actions = openhands_utils.get_actions_for_last_obs(observation)
    assert [event.id for event in second_actions] == [action_2.id]
    observation.last_step_action_id = second_actions[-1].id

    second_observations = openhands_utils.get_obs_for_last_action(
        observation,
        second_actions,
    )
    assert [event.id for event in second_observations] == [observation_2.id]
    assert sum(
        event.id == condensation.id
        for event in [*first_observations, *second_observations]
    ) == 1
    assert condensation.summary_event.id not in {
        event.id for event in observation.conversation_state.events
    }


def test_parallel_tool_batch_waits_for_every_action_result(monkeypatch):
    events, openhands_types, openhands_utils = _parser_modules(monkeypatch)

    initial = events.MessageEvent.model_construct(id="initial", source="user")
    action_1 = _action_event(events, "action-1", "call-1", "response-1")
    action_2 = _action_event(events, "action-2", "call-2", "response-1")
    action_3 = _action_event(events, "action-3", "call-3", "response-1")
    error_2 = _agent_error_event(events, "error-2", "call-2")
    observation_1 = _observation_event(events, "observation-1", "action-1", "call-1")

    observation = openhands_types.OpenHandsObservation(
        conversation_state=SimpleNamespace(
            events=[initial, action_1, action_2, error_2, action_3, observation_1],
            execution_status=None,
        ),
        last_step_observation_id=initial.id,
    )

    assert openhands_utils.get_actions_for_last_obs(observation) == []


@pytest.mark.parametrize("error_index", [0, 2])
def test_inline_agent_error_at_parallel_batch_edges(monkeypatch, error_index):
    events, openhands_types, openhands_utils = _parser_modules(monkeypatch)

    initial = events.MessageEvent.model_construct(id="initial", source="user")
    actions = [_action_event(events, f"action-{index + 1}", f"call-{index + 1}", "response-1") for index in range(3)]
    error = _agent_error_event(events, f"error-{error_index + 1}", f"call-{error_index + 1}")
    results = [
        (
            error
            if index == error_index
            else _observation_event(
                events,
                f"observation-{index + 1}",
                f"action-{index + 1}",
                f"call-{index + 1}",
            )
        )
        for index in range(3)
    ]
    next_action = _action_event(events, "action-4", "call-4", "response-2")
    next_observation = _observation_event(events, "observation-4", "action-4", "call-4")
    final_action = _action_event(events, "action-5", "call-5", "response-3")

    event_stream = [initial]
    for index, action in enumerate(actions):
        event_stream.append(action)
        if index == error_index:
            event_stream.append(error)
    event_stream.extend(result for index, result in enumerate(results) if index != error_index)
    event_stream.extend([next_action, next_observation, final_action])

    observation = openhands_types.OpenHandsObservation(
        conversation_state=SimpleNamespace(events=event_stream, execution_status=None),
        last_step_observation_id=initial.id,
    )

    step_actions = openhands_utils.get_actions_for_last_obs(observation)
    assert [event.id for event in step_actions] == [action.id for action in actions]
    observation.last_step_action_id = step_actions[-1].id

    step_observations = openhands_utils.get_obs_for_last_action(observation, step_actions)
    assert [event.id for event in step_observations] == [result.id for result in results]

    result_ids = {result.id for result in step_observations}
    observation.last_step_observation_id = next(event.id for event in reversed(event_stream) if event.id in result_ids)
    next_step_actions = openhands_utils.get_actions_for_last_obs(observation)
    assert [event.id for event in next_step_actions] == [next_action.id]


def test_agent_error_with_unrelated_tool_call_does_not_settle_action(monkeypatch):
    events, openhands_types, openhands_utils = _parser_modules(monkeypatch)

    initial = events.MessageEvent.model_construct(id="initial", source="user")
    action_1 = _action_event(events, "action-1", "call-1", "response-1")
    action_2 = _action_event(events, "action-2", "call-2", "response-1")
    unrelated_error = _agent_error_event(events, "error-other", "call-other")
    observation_1 = _observation_event(events, "observation-1", "action-1", "call-1")
    next_action = _action_event(events, "action-3", "call-3", "response-2")

    observation = openhands_types.OpenHandsObservation(
        conversation_state=SimpleNamespace(
            events=[initial, action_1, action_2, unrelated_error, observation_1, next_action],
            execution_status=None,
        ),
        last_step_observation_id=initial.id,
    )

    assert openhands_utils.get_actions_for_last_obs(observation) == []


def test_hook_event_with_action_id_does_not_settle_action(monkeypatch):
    events, openhands_types, openhands_utils = _parser_modules(monkeypatch)

    initial = events.MessageEvent.model_construct(id="initial", source="user")
    action_1 = _action_event(events, "action-1", "call-1", "response-1")
    action_2 = _action_event(events, "action-2", "call-2", "response-1")
    observation_1 = _observation_event(events, "observation-1", "action-1", "call-1")
    hook_event = events.HookExecutionEvent.model_construct(
        id="hook-2",
        source="hook",
        action_id="action-2",
        hook_event_type="PostToolUse",
        hook_command="true",
        success=True,
        exit_code=0,
    )
    next_action = _action_event(events, "action-3", "call-3", "response-2")

    observation = openhands_types.OpenHandsObservation(
        conversation_state=SimpleNamespace(
            events=[initial, action_1, action_2, observation_1, hook_event, next_action],
            execution_status=None,
        ),
        last_step_observation_id=initial.id,
    )

    assert openhands_utils.get_actions_for_last_obs(observation) == []
