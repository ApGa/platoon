"""The ``platoon-react`` harness: its options, its tool-call answers, its limit and the text it ends with, the last
two through the real gateway and env service on number search (as ``test_codeact_rollouts.py``).  Its handoff to a
terminal tool runs on TextCraft-Synth, in the TextCraft extension's tests (``test_textcraft_react.py``)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from inline_runs import open_service_task, run_tree
from test_codeact_rollouts import NUMBER_SEARCH

from platoon.envs.service import ToolResult
from platoon.harness.agent import LIMIT_MESSAGE, AgentResult, ChatReply
from platoon.harness.builtin.react import ReactOptions, answer_tool_calls

WRONG_GUESS = '<tool_call>\n{"name": "guess", "arguments": {"number": -1}}\n</tool_call>'


class FakeTools:
    """The ``call_tool`` part of an agent context: a canned result per tool name, and the calls made."""

    def __init__(self, **results: ToolResult) -> None:
        self.results = results
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append((name, arguments))
        return self.results[name]


def _reply(*calls: tuple[str, str]) -> ChatReply:
    """A reply that calls ``(name, raw JSON arguments)`` in order, with ids ``c0``, ``c1``, ..."""

    tool_calls = [
        {"id": f"c{index}", "type": "function", "function": {"name": name, "arguments": arguments}}
        for index, (name, arguments) in enumerate(calls)
    ]
    return ChatReply("call", {"role": "assistant", "content": None, "tool_calls": tool_calls}, "tool_calls")


# ---------------------------------------------------------------------------- options


def test_react_options_are_typed_and_default_to_none_set():
    assert ReactOptions.from_mapping(None) == ReactOptions() == ReactOptions(system_prompt=None, sampling={})
    options = ReactOptions.from_mapping({"system_prompt": "Be brief.", "sampling": {"temperature": 0.5}})
    assert (options.system_prompt, options.sampling) == ("Be brief.", {"temperature": 0.5})
    assert ReactOptions.from_mapping({"sampling": None}).sampling == {}


def test_react_options_reject_unknown_keys_such_as_the_dropped_max_turns():
    # ctx.max_steps is the one step budget; max_turns was a second one.
    with pytest.raises(ValueError, match=r"Unknown platoon-react harness options \['max_turns'\]"):
        ReactOptions.from_mapping({"max_turns": 3})


@pytest.mark.asyncio
async def test_a_config_that_still_sets_max_turns_fails_the_agent_with_a_clear_error(make_gateway):
    gateway, sampler = make_gateway(lambda prompt: WRONG_GUESS)
    task, session = await open_service_task(NUMBER_SEARCH, "val:6")

    run = await run_tree(gateway, "platoon-react", session, task, max_steps=12, options={"max_turns": 3})

    assert run.result.status == "error" and "Unknown platoon-react harness options ['max_turns']" in run.result.error
    assert sampler.calls == []


# ---------------------------------------------------------------------------- answering tool calls


@pytest.mark.asyncio
async def test_tool_calls_are_answered_in_order_with_each_results_text():
    image = ToolResult([{"type": "text", "text": "a chart:"}, {"type": "image", "mimeType": "image/png", "data": ""}])
    tools = FakeTools(guess=ToolResult.text("Too low"), plot=image)

    messages, finished = await answer_tool_calls(tools, _reply(("guess", '{"number": 3}'), ("plot", "")))

    assert tools.calls == [("guess", {"number": 3}), ("plot", {})]  # no arguments is an empty object
    assert messages == [
        {"role": "tool", "tool_call_id": "c0", "content": "Too low"},
        {"role": "tool", "tool_call_id": "c1", "content": "a chart:\n[image/png]"},  # ToolResult.as_text
    ]
    assert finished is None


@pytest.mark.asyncio
async def test_a_call_with_bad_arguments_is_answered_without_reaching_the_env():
    tools = FakeTools(guess=ToolResult.text("Too high"))

    messages, finished = await answer_tool_calls(
        tools, _reply(("guess", "{not json"), ("guess", "[1]"), ("guess", json.dumps({"number": 9})))
    )

    assert tools.calls == [("guess", {"number": 9})]
    assert [message["content"].split(":")[0] for message in messages[:2]] == ["Invalid arguments"] * 2
    assert messages[1]["content"] == "Invalid arguments: tool arguments must be a JSON object"
    assert messages[2] == {"role": "tool", "tool_call_id": "c2", "content": "Too high"} and finished is None


@pytest.mark.asyncio
async def test_a_result_that_finishes_the_episode_ends_the_answers():
    done = ToolResult([{"type": "text", "text": "Correct!"}], finished=True)
    tools = FakeTools(guess=done, other=ToolResult.text("never asked"))

    messages, finished = await answer_tool_calls(tools, _reply(("guess", '{"number": 5}'), ("other", "{}")))

    assert finished is done and tools.calls == [("guess", {"number": 5})]  # the call after it is not made
    assert messages == [{"role": "tool", "tool_call_id": "c0", "content": "Correct!"}]


# ---------------------------------------------------------------------------- rollouts


@pytest.mark.asyncio
@pytest.mark.parametrize(("max_steps", "calls"), [(2, 2), (3, 3)])
async def test_react_ends_with_the_shared_limit_message(make_gateway, max_steps, calls):
    gateway, sampler = make_gateway(lambda prompt: WRONG_GUESS)
    task, session = await open_service_task(NUMBER_SEARCH, "val:6")

    run = await run_tree(gateway, "platoon-react", session, task, max_steps=max_steps)

    assert run.result == AgentResult("limit", error=LIMIT_MESSAGE)
    assert len(sampler.calls) == calls and run.root.error_message == LIMIT_MESSAGE


@pytest.mark.asyncio
async def test_react_ends_with_the_text_of_the_result_that_finished_the_episode(make_gateway):
    from platoon.envs.number_search.env import task_specs

    target = task_specs("val")[4]["misc"]["target"]
    gateway, sampler = make_gateway(
        lambda prompt: f'<tool_call>\n{{"name": "guess", "arguments": {{"number": {target}}}}}\n</tool_call>'
    )
    task, session = await open_service_task(NUMBER_SEARCH, "val:4")

    run = await run_tree(gateway, "platoon-react", session, task, max_steps=3)

    assert run.result == AgentResult("completed", final_message=f"You guessed the number {target} correctly!")
    assert len(sampler.calls) == 1
