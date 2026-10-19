"""GatewayServer over real HTTP, driven by the OpenAI SDK and litellm the way harnesses drive it."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp
import openai
import pytest
import pytest_asyncio
from gateway_fakes import MODEL_NAME, TOOLS, WEATHER_CALL, Completion

from platoon.config_defs import InferenceParams
from platoon.gateway.core import SamplingOverrides
from platoon.gateway.server import CALL_ID_HEADER, GatewayServer

SYSTEM: Any = {"role": "system", "content": "You are a helpful assistant."}
USER: Any = {"role": "user", "content": "What's the weather in Paris?"}


@pytest_asyncio.fixture
async def server(gateway):
    server = GatewayServer(gateway, keepalive_seconds=0.05)
    await server.start(host="127.0.0.1")
    yield server
    await server.stop()


def open_root(gateway, rollout_id="rollout-1", **kwargs):
    return gateway.open_agent(rollout_id=rollout_id, trajectory_id=f"{rollout_id}-root", interactions={}, **kwargs)


def client(base_url: str, api_key: str) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0)


async def post(url: str, payload, *, api_key: str | None = None, headers: dict | None = None):
    headers = dict(headers or {})
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    data = payload if isinstance(payload, str) else json.dumps(payload)
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data, headers={"Content-Type": "application/json", **headers}) as response:
            return response.status, response.headers, await response.text()


# --------------------------------------------------------------------------- Chat Completions


@pytest.mark.asyncio
async def test_chat_completion_records_the_call_on_its_agent(gateway, sampler, server):
    sampler.script = [Completion("Hello there.", routing_id=9)]
    agent = open_root(gateway)
    async with client(f"{server.base_url}/v1", agent.api_key) as openai_client:
        raw = await openai_client.chat.completions.with_raw_response.create(model="policy", messages=[SYSTEM, USER])
    completion = raw.parse()

    assert raw.headers[CALL_ID_HEADER] == completion.id == agent.call_ids[0]
    assert completion.model == MODEL_NAME and completion.object == "chat.completion"
    choice = completion.choices[0]
    assert (choice.message.role, choice.message.content, choice.finish_reason) == ("assistant", "Hello there.", "stop")
    call = agent.calls[0]
    assert (completion.usage.prompt_tokens, completion.usage.completion_tokens) == (
        call.prompt_tokens,
        call.completion_tokens,
    )
    interaction = agent.interactions[completion.id]
    assert interaction.obs.to_ints() == sampler.calls[0]["prompt"].to_ints()
    assert interaction.action.tokens == call.tokens and interaction.routing_id == 9


@pytest.mark.asyncio
async def test_tool_calls_and_results_round_trip_over_turns(gateway, sampler, tokenizer, server):
    sampler.script = [Completion(WEATHER_CALL), Completion("Sunny in Paris.")]
    agent = open_root(gateway)
    async with client(f"{server.base_url}/v1", agent.api_key) as openai_client:
        first = await openai_client.chat.completions.create(
            model="policy", messages=[SYSTEM, USER], tools=TOOLS, tool_choice="auto"
        )
        message = first.choices[0].message
        assert first.choices[0].finish_reason == "tool_calls" and message.content is None
        assert message.tool_calls is not None
        (tool_call,) = message.tool_calls
        assert tool_call.function.name == "get_weather"
        assert json.loads(tool_call.function.arguments) == {"city": "Paris"}
        # Harnesses append the SDK message as is, None fields included.
        history: list[Any] = [
            SYSTEM,
            USER,
            message.model_dump(),
            {"role": "tool", "tool_call_id": tool_call.id, "content": "Sunny, 25C"},
        ]
        second = await openai_client.chat.completions.create(model="policy", messages=history, tools=TOOLS)

    assert second.choices[0].message.content == "Sunny in Paris."
    assert agent.call_ids == [first.id, second.id]
    prompt = tokenizer.decode(sampler.calls[1]["prompt"].to_ints())
    assert '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>' in prompt
    assert "<tool_response>\nSunny, 25C\n</tool_response>" in prompt


@pytest.mark.asyncio
async def test_streaming_answers_with_synthetic_chunks(gateway, sampler, server):
    sampler.script = [Completion("Let me check.\n" + WEATHER_CALL)]
    agent = open_root(gateway)
    async with client(f"{server.base_url}/v1", agent.api_key) as openai_client:
        stream = await openai_client.chat.completions.create(
            model="policy",
            messages=[SYSTEM, USER],
            tools=TOOLS,
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks = [chunk async for chunk in stream]

    (call_id,) = agent.call_ids
    assert {chunk.id for chunk in chunks} == {call_id}
    assert chunks[0].choices[0].delta.role == "assistant"
    content = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
    assert content == "Let me check.\n"
    tool_deltas = [delta for chunk in chunks if chunk.choices for delta in chunk.choices[0].delta.tool_calls or []]
    assert [(delta.index, delta.function.name) for delta in tool_deltas] == [(0, "get_weather")]
    assert tool_deltas[0].id == agent.calls[0].tool_calls[0]["id"]
    assert json.loads(tool_deltas[0].function.arguments) == {"city": "Paris"}
    assert chunks[-2].choices[0].finish_reason == "tool_calls"
    assert chunks[-1].choices == [] and chunks[-1].usage.total_tokens == agent.calls[0].usage()["total_tokens"]


@pytest.mark.asyncio
async def test_streaming_keeps_the_connection_alive_while_sampling(gateway, sampler, server):
    sampler.script = [Completion("Done.", delay=0.3)]
    agent = open_root(gateway)
    payload = {"model": "policy", "messages": [USER], "stream": True}
    status, headers, body = await post(f"{server.base_url}/v1/chat/completions", payload, api_key=agent.api_key)

    assert status == 200 and headers["Content-Type"].startswith("text/event-stream")
    assert body.startswith(": keepalive\n\n") and body.count(": keepalive\n\n") >= 2
    events = [line.removeprefix("data: ") for line in body.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    assert json.loads(events[1])["choices"][0]["delta"] == {"content": "Done."}


@pytest.mark.asyncio
async def test_reasoning_logprobs_and_developer_messages(gateway, sampler, tokenizer, server):
    sampler.script = [Completion("<think>Paris is in France.</think>Sunny.")]
    agent = open_root(gateway)
    developer: Any = {"role": "developer", "content": "Be brief."}
    async with client(f"{server.base_url}/v1", agent.api_key) as openai_client:
        completion = await openai_client.chat.completions.create(
            model="policy", messages=[developer, USER], logprobs=True
        )
    message = completion.choices[0].message
    assert message.content == "Sunny." and message.model_extra["reasoning_content"] == "Paris is in France."
    logprobs = completion.choices[0].logprobs.content
    assert logprobs is not None
    assert [entry.logprob for entry in logprobs] == [-0.25] * len(agent.calls[0].tokens)
    assert "".join(entry.token for entry in logprobs) == tokenizer.decode(agent.calls[0].tokens)
    assert tokenizer.decode(sampler.calls[0]["prompt"].to_ints()).startswith("<|im_start|>system\nBe brief.")


# --------------------------------------------------------------------------- identity and the agent API


@pytest.mark.asyncio
async def test_every_secret_presentation_reaches_its_agent(gateway, sampler, server):
    sampler.script = [Completion("bearer"), Completion("x-api-key"), Completion("path")]
    agent = open_root(gateway)
    async with client(f"{server.base_url}/v1", agent.api_key) as openai_client:
        await openai_client.chat.completions.create(model="policy", messages=[USER])
    status, _, _ = await post(
        f"{server.base_url}/v1/chat/completions",
        {"model": "policy", "messages": [USER]},
        headers={"x-api-key": agent.api_key},
    )
    assert status == 200
    async with client(f"{server.base_url}/s/{agent.secret}/v1", "unused") as openai_client:
        await openai_client.chat.completions.create(model="policy", messages=[USER])
    assert [call.content for call in agent.calls] == ["bearer", "x-api-key", "path"]


@pytest.mark.asyncio
async def test_unknown_and_closed_secrets_are_refused(gateway, sampler, server):
    agent = open_root(gateway)
    async with client(f"{server.base_url}/v1", "not-an-agent-key") as openai_client:
        with pytest.raises(openai.AuthenticationError):
            await openai_client.chat.completions.create(model="policy", messages=[USER])
    status, _, body = await post(f"{server.base_url}/v1/chat/completions", {"model": "p", "messages": [USER]})
    assert status == 401 and json.loads(body)["error"]["code"] == "invalid_api_key"

    gateway.close_agent(agent)
    async with client(f"{server.base_url}/v1", agent.api_key) as openai_client:
        with pytest.raises(openai.APIStatusError) as info:
            await openai_client.chat.completions.create(model="policy", messages=[USER])
    assert info.value.status_code == 410 and info.value.body["code"] == "agent_closed"
    assert sampler.calls == []


@pytest.mark.asyncio
async def test_concurrent_agents_get_exact_attribution(gateway, sampler, server):
    sampler.script = [Completion("first", delay=0.1), Completion("second")]
    agents = [open_root(gateway, "rollout-a"), open_root(gateway, "rollout-b")]

    async def ask(agent, content):
        async with client(f"{server.base_url}/v1", agent.api_key) as openai_client:
            return await openai_client.chat.completions.create(
                model="policy", messages=[{"role": "user", "content": content}]
            )

    responses = await asyncio.gather(ask(agents[0], "a"), ask(agents[1], "b"))
    for agent, response in zip(agents, responses, strict=True):
        assert agent.call_ids == [response.id] and list(agent.interactions) == [response.id]
    assert agents[0].interactions is not agents[1].interactions


@pytest.mark.asyncio
async def test_agent_api_mints_children_with_their_own_secret(gateway, sampler, server):
    sampler.script = [Completion("child answer")]
    root = open_root(gateway)
    status, _, body = await post(
        f"{server.base_url}/v1/platoon/agents", {"role": "researcher", "goal": "Check"}, api_key=root.api_key
    )
    assert status == 201
    minted = json.loads(body)
    assert minted["base_url"] == f"{server.base_url}/v1" and minted["api_key"] != root.api_key
    async with client(minted["base_url"], minted["api_key"]) as openai_client:
        completion = await openai_client.chat.completions.create(model="policy", messages=[USER])

    child = gateway.authenticate(minted["api_key"])
    assert (child.agent_id, child.parent_agent_id, child.role, child.goal) == (
        minted["agent_id"],
        root.agent_id,
        "researcher",
        "Check",
    )
    assert child.call_ids == [completion.id] and root.calls == []
    assert completion.id in root.interactions  # one rollout, one interaction map
    assert gateway.agents("rollout-1") == [root, child]

    # A parent that addresses the gateway by path gets a path-form base URL for its child.
    status, _, body = await post(f"{server.base_url}/s/{root.secret}/v1/platoon/agents", {"role": "verifier"})
    minted = json.loads(body)
    assert status == 201 and minted["base_url"] == f"{server.base_url}/s/{minted['api_key']}/v1"
    status, _, body = await post(f"{server.base_url}/v1/platoon/agents", {"goal": "no role"}, api_key=root.api_key)
    assert status == 400 and "role" in json.loads(body)["error"]["message"]
    status, _, _ = await post(f"{server.base_url}/v1/platoon/agents", {"role": "x"}, api_key="unknown")
    assert status == 401


@pytest.mark.asyncio
async def test_http_calls_that_omit_sampling_get_the_agents_defaults(gateway, sampler, server):
    sampler.script = [Completion("default"), Completion("own")]
    root = open_root(gateway, sampling_defaults=SamplingOverrides(max_tokens=21, temperature=0.3, top_p=0.6))
    async with client(f"{server.base_url}/v1", root.api_key) as openai_client:
        await openai_client.chat.completions.create(model="policy", messages=[USER])
        await openai_client.chat.completions.create(model="policy", messages=[USER], temperature=1.5, max_tokens=9)
    default, own = (call["sampling_params"] for call in sampler.calls)
    assert (default.max_tokens, default.temperature, default.top_p) == (21, 0.3, 0.6)
    assert (own.max_tokens, own.temperature, own.top_p) == (9, 1.5, 0.6)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("inference_params", "budget"),
    [({}, 4096), ({"max_completion_tokens": 33}, 33)],
    ids=["unset-gateway-default", "rollout-budget"],
)
async def test_http_calls_get_only_the_sampling_the_rollout_sets(make_gateway, sampler, inference_params, budget):
    # The gateway's temperature and top_p are not 1 here, so an InferenceParams default that leaked would show.
    gateway = make_gateway(temperature=0.7, top_p=0.8)
    sampler.script = [Completion("without max_tokens"), Completion("with max_tokens")]
    defaults = SamplingOverrides.from_inference_params(InferenceParams(**inference_params))
    server = GatewayServer(gateway, keepalive_seconds=0.05)
    await server.start(host="127.0.0.1")
    try:
        root = open_root(gateway, sampling_defaults=defaults)
        async with client(f"{server.base_url}/v1", root.api_key) as openai_client:
            await openai_client.chat.completions.create(model="policy", messages=[USER])
            await openai_client.chat.completions.create(model="policy", messages=[USER], max_tokens=9)
    finally:
        await server.stop()
    without, own = (call["sampling_params"] for call in sampler.calls)
    assert gateway.max_tokens == 4096
    assert (without.max_tokens, without.temperature, without.top_p) == (budget, 0.7, 0.8)
    assert (own.max_tokens, own.temperature, own.top_p) == (9, 0.7, 0.8)


# --------------------------------------------------------------------------- errors


@pytest.mark.asyncio
async def test_context_overflow_is_openai_context_length_exceeded(make_gateway, sampler):
    import litellm

    gateway = make_gateway(context_window_length=32)
    server = GatewayServer(gateway)
    base_url = await server.start(host="127.0.0.1")
    agent = open_root(gateway)
    messages = [{"role": "user", "content": "word " * 100}]
    try:
        async with client(f"{base_url}/v1", agent.api_key) as openai_client:
            with pytest.raises(openai.BadRequestError) as info:
                await openai_client.chat.completions.create(model="policy", messages=messages)
        assert info.value.body["code"] == "context_length_exceeded"
        # litellm-based harnesses (OpenHands, mini-swe-agent) see their native context-window error.
        with pytest.raises(litellm.ContextWindowExceededError):
            await litellm.acompletion(
                model="openai/policy", api_base=f"{base_url}/v1", api_key=agent.api_key, messages=messages
            )
    finally:
        await server.stop()
    assert sampler.calls == []
    assert agent.context_overflows == 2  # one refusal per request, so a harness crash can be attributed to it


@pytest.mark.asyncio
async def test_sampling_failures_map_to_server_errors(gateway, sampler, server):
    agent = open_root(gateway)

    async def broken(**kwargs):
        raise RuntimeError("backend lost")

    sampler.sample_async = broken
    async with client(f"{server.base_url}/v1", agent.api_key) as openai_client:
        with pytest.raises(openai.InternalServerError, match="backend lost"):
            await openai_client.chat.completions.create(model="policy", messages=[USER])
        stream = await openai_client.chat.completions.create(model="policy", messages=[USER], stream=True)
        with pytest.raises(openai.APIError, match="backend lost"):
            async for _ in stream:
                pass
    assert agent.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"model": "p", "messages": [USER], "n": 2}, "n=1"),
        ({"model": "p", "messages": [USER], "tools": TOOLS, "tool_choice": "required"}, "tool_choice"),
        ({"model": "p", "messages": []}, "messages"),
        ({"model": "p", "messages": ["hi"]}, "role"),
        ({"model": "p", "messages": [USER], "temperature": 9}, "temperature"),
        ("{not json", "not valid JSON"),
    ],
)
async def test_bad_requests_are_rejected_before_sampling(gateway, sampler, server, payload, message):
    agent = open_root(gateway)
    status, _, body = await post(f"{server.base_url}/v1/chat/completions", payload, api_key=agent.api_key)
    assert status == 400 and message in json.loads(body)["error"]["message"]
    assert sampler.calls == []


# --------------------------------------------------------------------------- litellm, models, lifecycle


@pytest.mark.asyncio
async def test_litellm_harness_round_trip(gateway, sampler, server):
    import litellm

    sampler.script = [Completion(WEATHER_CALL)]
    agent = open_root(gateway)
    response = await litellm.acompletion(
        model="openai/policy",
        api_base=f"{server.base_url}/v1",
        api_key=agent.api_key,
        messages=[SYSTEM, USER],
        tools=TOOLS,
        num_retries=0,
    )
    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls is not None
    (tool_call,) = tool_calls
    assert response.id == agent.call_ids[0] and response.choices[0].finish_reason == "tool_calls"
    assert tool_call.function.name == "get_weather"


@pytest.mark.asyncio
async def test_models_health_and_lifecycle(gateway):
    server = GatewayServer(gateway)
    base_url = await server.start()
    assert base_url.startswith("http://127.0.0.1:") and server.base_url == base_url
    with pytest.raises(RuntimeError, match="already started"):
        await server.start()
    agent = open_root(gateway)
    async with client(f"{base_url}/v1", agent.api_key) as openai_client:
        models = await openai_client.models.list()
    assert [model.id for model in models.data] == [MODEL_NAME]
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base_url}/healthz") as response:
            assert response.status == 200 and (await response.json())["status"] == "ok"
        async with session.get(f"{base_url}/v1/models") as response:
            assert response.status == 401
    await server.stop()
    await server.stop()
    with pytest.raises(RuntimeError, match="not started"):
        server.base_url
