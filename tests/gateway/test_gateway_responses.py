"""Experimental OpenAI Responses dialect (Codex): input-item mapping, output items, streaming events."""

from __future__ import annotations

import json
from typing import Any

import openai
import pytest
import pytest_asyncio
from gateway_fakes import TOOLS, WEATHER_CALL, Completion

from platoon.gateway.dialects.openai_chat import OpenAIChat
from platoon.gateway.dialects.responses import OpenAIResponses, input_messages, seal_reasoning
from platoon.gateway.server import GatewayServer

FUNCTION_TOOLS: list[Any] = [
    {"type": "function", **TOOLS[0]["function"], "strict": False},
    {"type": "web_search"},  # not a function tool: never rendered
]

# What Codex sends on its second turn: its instructions, the user turn, then the first response's output items
# (reasoning keeps only its encrypted_content) and the tool's output.
CODEX_INPUT: list[Any] = [
    {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Sandbox: workspace-write"}]},
    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What's the weather in Paris?"}]},
    {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": seal_reasoning("Use the tool.")},
    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Checking."}]},
    {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": '{"city": "Paris"}'},
    {"type": "function_call_output", "call_id": "call_1", "output": "Sunny, 25C"},
]


@pytest_asyncio.fixture
async def server(gateway):
    server = GatewayServer(gateway, keepalive_seconds=0.05)
    await server.start(host="127.0.0.1")
    yield server
    await server.stop()


def test_input_items_map_onto_chat_messages():
    assert input_messages(CODEX_INPUT) == [
        {"role": "system", "content": "Sandbox: workspace-write"},
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "role": "assistant",
            "content": "Checking.",
            "reasoning_content": "Use the tool.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 25C"},
    ]
    with pytest.raises(ValueError, match="input_image"):
        input_messages([{"role": "user", "content": [{"type": "input_image", "image_url": "data:"}]}])
    with pytest.raises(ValueError, match="local_shell_call"):
        input_messages([{"type": "local_shell_call", "call_id": "x"}])


def test_request_options_map_onto_the_core_request(gateway):
    dialect = OpenAIResponses(OpenAIChat(gateway))
    request = dialect.chat_request(
        {
            "instructions": "You are Codex.",
            "input": CODEX_INPUT,
            "tools": FUNCTION_TOOLS,
            "tool_choice": "auto",
            "max_output_tokens": 512,
            "reasoning": {"effort": "none"},
        }
    )
    assert request.messages[0] == {"role": "system", "content": "You are Codex."}
    assert request.render.tools == [{"type": "function", "function": TOOLS[0]["function"]}]
    assert request.render.render_kwargs == {"enable_thinking": False}
    assert request.sampling.max_tokens == 512
    with pytest.raises(ValueError, match="tool_choice"):
        dialect.chat_request({"input": "hi", "tools": FUNCTION_TOOLS, "tool_choice": "required"})
    with pytest.raises(ValueError, match="'input'"):
        dialect.chat_request({"input": 3})


@pytest.mark.asyncio
async def test_responses_round_trip_output_items_and_reasoning(gateway, sampler, server):
    sampler.script = [Completion("<think>Use the tool.</think>Checking.\n" + WEATHER_CALL), Completion("Sunny.")]
    agent = gateway.open_agent(rollout_id="r", trajectory_id="t", interactions={})
    async with openai.AsyncOpenAI(base_url=f"{server.base_url}/v1", api_key=agent.api_key, max_retries=0) as client:
        first = await client.responses.create(
            model="policy", instructions="You are Codex.", input="What's the weather in Paris?", tools=FUNCTION_TOOLS
        )
        reasoning, message, function_call = first.output
        assert first.id == agent.call_ids[0] and first.status == "completed"
        assert reasoning.type == "reasoning" and reasoning.content[0].text == "Use the tool."
        assert message.type == "message" and message.content[0].text == "Checking.\n"
        assert (function_call.type, function_call.name) == ("function_call", "get_weather")
        assert json.loads(function_call.arguments) == {"city": "Paris"}
        assert first.usage.input_tokens == agent.calls[0].prompt_tokens

        # Codex sends the output items back, dropping everything but encrypted_content from reasoning.
        replay: list[Any] = [
            {"type": "message", "role": "user", "content": "What's the weather in Paris?"},
            {"type": "reasoning", "summary": [], "encrypted_content": reasoning.encrypted_content},
            message.model_dump(exclude_none=True),
            function_call.model_dump(exclude_none=True),
            {"type": "function_call_output", "call_id": function_call.call_id, "output": "Sunny, 25C"},
        ]
        second = await client.responses.create(model="policy", input=replay, tools=FUNCTION_TOOLS)
    assert second.output_text == "Sunny." and agent.call_ids == [first.id, second.id]
    request = OpenAIResponses(OpenAIChat(gateway)).chat_request({"input": replay})
    assert dict(request.messages[1])["reasoning_content"] == "Use the tool."  # exactly what was sampled


@pytest.mark.asyncio
async def test_responses_stream_events(gateway, sampler, server):
    sampler.script = [Completion("Checking.\n" + WEATHER_CALL)]
    agent = gateway.open_agent(rollout_id="r", trajectory_id="t", interactions={})
    async with openai.AsyncOpenAI(base_url=f"{server.base_url}/v1", api_key=agent.api_key, max_retries=0) as client:
        stream = await client.responses.create(model="policy", input="Weather?", tools=FUNCTION_TOOLS, stream=True)
        events = [event async for event in stream]

    assert [event.type for event in events] == [
        "response.created",
        "response.output_item.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.output_item.done",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert [event.sequence_number for event in events] == list(range(len(events)))
    assert events[2].delta == "Checking.\n"
    assert events[8].item.name == "get_weather"
    completed = events[-1].response
    assert completed.id == agent.call_ids[0] and completed.usage.output_tokens == len(agent.calls[0].tokens)


@pytest.mark.asyncio
async def test_truncated_and_failed_responses(gateway, sampler, server):
    sampler.script = [Completion("cut off mid", stop_reason="length")]
    agent = gateway.open_agent(rollout_id="r", trajectory_id="t", interactions={})
    async with openai.AsyncOpenAI(base_url=f"{server.base_url}/v1", api_key=agent.api_key, max_retries=0) as client:
        truncated = await client.responses.create(model="policy", input="Go")
        assert truncated.status == "incomplete" and truncated.incomplete_details.reason == "max_output_tokens"

        async def broken(**kwargs):
            raise RuntimeError("backend lost")

        sampler.sample_async = broken
        stream = await client.responses.create(model="policy", input="Go", stream=True)
        (failed,) = [event async for event in stream]
    assert failed.type == "response.failed" and "backend lost" in failed.response.error.message
