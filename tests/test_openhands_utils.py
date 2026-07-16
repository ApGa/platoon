from types import SimpleNamespace

import pytest


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
