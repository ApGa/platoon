"""Experimental Anthropic Messages dialect (Claude Code), driven by the Anthropic SDK."""

from __future__ import annotations

from typing import Any

import anthropic
import pytest
import pytest_asyncio
from gateway_fakes import TOOLS, WEATHER_CALL, Completion

from platoon.gateway.dialects.anthropic import AnthropicMessages
from platoon.gateway.server import GatewayServer

ANTHROPIC_TOOLS: list[Any] = [
    {
        "name": TOOLS[0]["function"]["name"],
        "description": TOOLS[0]["function"]["description"],
        "input_schema": TOOLS[0]["function"]["parameters"],
    }
]
USER: Any = {"role": "user", "content": "What's the weather in Paris?"}

# A Claude Code-shaped second turn: cached system blocks, the first answer's blocks, then the tool result.
CLAUDE_CODE_TURN = {
    "model": "claude-sonnet",
    "max_tokens": 1024,
    "system": [{"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}}],
    "messages": [
        USER,
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Use the tool.", "signature": "platoon:call-1"},
                {"type": "text", "text": "Checking."},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "Sunny"}]},
                {"type": "text", "text": "Anything else?"},
            ],
        },
    ],
    "tools": ANTHROPIC_TOOLS,
    "tool_choice": {"type": "auto"},
}


@pytest_asyncio.fixture
async def server(gateway):
    server = GatewayServer(gateway, keepalive_seconds=0.05)
    await server.start(host="127.0.0.1")
    yield server
    await server.stop()


def client(base_url: str, api_key: str) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(base_url=base_url, api_key=api_key, max_retries=0)


def test_messages_map_onto_chat_messages():
    request = AnthropicMessages().chat_request(CLAUDE_CODE_TURN)
    assert request.messages == [
        {"role": "system", "content": "You are Claude Code."},
        USER,
        {
            "role": "assistant",
            "content": "Checking.",
            "reasoning_content": "Use the tool.",
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_1", "content": "Sunny"},
        {"role": "user", "content": "Anything else?"},
    ]
    assert request.render.tools == TOOLS
    assert request.sampling.max_tokens == 1024


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"thinking": {"type": "disabled"}}, None),
        ({"tool_choice": {"type": "any"}}, "tool_choice"),
        ({"stop_sequences": ["END"]}, "stop_sequences"),
        ({"messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]}, "image"),
    ],
)
def test_request_options(change, match):
    body = {**CLAUDE_CODE_TURN, **change}
    if match is None:
        assert AnthropicMessages().chat_request(body).render.render_kwargs == {"enable_thinking": False}
    else:
        with pytest.raises(ValueError, match=match):
            AnthropicMessages().chat_request(body)


@pytest.mark.asyncio
async def test_messages_answer_with_thinking_text_and_tool_use(gateway, sampler, server):
    sampler.script = [Completion("<think>Use the tool.</think>Checking.\n" + WEATHER_CALL)]
    agent = gateway.open_agent(rollout_id="r", trajectory_id="t", interactions={})
    async with client(server.base_url, agent.api_key) as anthropic_client:
        message = await anthropic_client.messages.create(
            model="claude-sonnet", max_tokens=256, messages=[USER], tools=ANTHROPIC_TOOLS
        )
    thinking, text, tool_use = message.content
    assert message.id == agent.call_ids[0] and message.stop_reason == "tool_use"
    assert (thinking.type, thinking.thinking) == ("thinking", "Use the tool.") and thinking.signature
    assert (text.type, text.text) == ("text", "Checking.\n")
    assert (tool_use.type, tool_use.name, tool_use.input) == ("tool_use", "get_weather", {"city": "Paris"})
    assert tool_use.id == agent.calls[0].tool_calls[0]["id"]
    assert message.usage.input_tokens == agent.calls[0].prompt_tokens
    assert sampler.calls[0]["sampling_params"].max_tokens == 256


@pytest.mark.asyncio
async def test_messages_stream_and_count_tokens(gateway, sampler, server):
    sampler.script = [Completion("<think>Plan.</think>Sunny.")]
    agent = gateway.open_agent(rollout_id="r", trajectory_id="t", interactions={})
    async with client(f"{server.base_url}/s/{agent.secret}", "unused") as anthropic_client:
        async with anthropic_client.messages.stream(model="claude-sonnet", max_tokens=64, messages=[USER]) as stream:
            events = [event.type async for event in stream]
            message = await stream.get_final_message()
        counted = await anthropic_client.messages.count_tokens(model="claude-sonnet", messages=[USER])
    assert [block.type for block in message.content] == ["thinking", "text"]
    assert message.content[1].text == "Sunny." and message.stop_reason == "end_turn"
    assert events[0] == "message_start" and events[-1] == "message_stop"
    assert message.usage.output_tokens == len(agent.calls[0].tokens)
    assert counted.input_tokens == agent.calls[0].prompt_tokens


@pytest.mark.asyncio
async def test_anthropic_errors(make_gateway, sampler):
    gateway = make_gateway(context_window_length=64)  # USER fits; the 100 words do not
    server = GatewayServer(gateway)
    base_url = await server.start(host="127.0.0.1")
    agent = gateway.open_agent(rollout_id="r", trajectory_id="t", interactions={})
    try:
        async with client(base_url, agent.api_key) as anthropic_client:
            with pytest.raises(anthropic.BadRequestError, match="prompt is too long"):
                await anthropic_client.messages.create(
                    model="claude-sonnet", max_tokens=8, messages=[{"role": "user", "content": "word " * 100}]
                )

            async def broken(**kwargs):
                raise RuntimeError("backend lost")

            sampler.sample_async = broken
            stream = await anthropic_client.messages.create(
                model="claude-sonnet", max_tokens=8, messages=[USER], stream=True
            )
            with pytest.raises(anthropic.APIStatusError, match="backend lost"):
                async for _ in stream:
                    pass
        async with client(base_url, "not-an-agent-key") as anthropic_client:
            with pytest.raises(anthropic.AuthenticationError):
                await anthropic_client.messages.create(model="claude-sonnet", max_tokens=8, messages=[USER])
    finally:
        await server.stop()
    assert agent.calls == []
