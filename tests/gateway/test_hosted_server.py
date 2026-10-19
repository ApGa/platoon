"""The gateway's HTTP front on a hosted gateway, driven by the real OpenAI and Anthropic SDKs as harnesses use them:
each dialect's request goes to the same path under the base URL, the provider's reply (JSON, stream or error) comes
back as it was sent with the key swapped and redacted, a stream's status waits for its first keepalive, and every
call is recorded on its agent."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from aiohttp import ClientSession
from hosted_fakes import (
    CHAT,
    FAKE_KEY,
    MESSAGES,
    MODEL,
    RESPONSES,
    SERVED_MODEL,
    FakeProvider,
    Fault,
    anthropic_error,
    function_call,
    openai_error,
    output_message,
    tool_call,
    tool_use,
)

from platoon.eval.config_defs import HostedConfig
from platoon.gateway import GatewayServer
from platoon.gateway import hosted as hosted_module
from platoon.gateway.hosted import HostedGateway
from platoon.gateway.server import CALL_ID_HEADER
from platoon.gateway.trace import build_agent_trajectory

openai = pytest.importorskip("openai")
anthropic = pytest.importorskip("anthropic")

KEY_ENV = "PLATOON_TEST_PROVIDER_KEY"
TOOLS = [{"type": "function", "function": {"name": "guess", "parameters": {"type": "object", "properties": {}}}}]


@pytest.fixture(autouse=True)
def _environment(monkeypatch):
    monkeypatch.setenv(KEY_ENV, FAKE_KEY)
    for name in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setattr(hosted_module, "RETRY_BACKOFF_S", (0.01, 0.02))


@pytest.fixture
def provider():
    with FakeProvider() as fake:
        yield fake


def hosted(provider: FakeProvider, **settings: Any) -> HostedGateway:
    options: dict[str, Any] = {"max_retries": 1, **settings}
    return HostedGateway(HostedConfig(base_url=provider.base_url, api_key_env=KEY_ENV, **options), MODEL)


async def serve(gateway: HostedGateway, use, *, keepalive_seconds: float = 10.0) -> Any:
    """Start the gateway's server, open an agent, run ``use(base_url, agent)``, stop."""

    server = GatewayServer(gateway, keepalive_seconds=keepalive_seconds)
    base_url = await server.start(host="127.0.0.1")
    agent = gateway.open_agent(rollout_id="r", trajectory_id="t", interactions={})
    try:
        return agent, await use(base_url, agent)
    finally:
        await server.stop()
        await gateway.aclose()


# --------------------------------------------------------------------------- Chat Completions


def test_a_chat_completions_harness_gets_the_provider_s_reply_unchanged(provider):
    reply = {"content": "", "reasoning_content": "hm", "tool_calls": [tool_call("guess", {}, call_id="call_x")]}
    provider.replies[CHAT] = lambda body: {**reply, "provider_extra": {"kept": True}}

    async def use(base_url, agent):
        client = openai.AsyncOpenAI(base_url=f"{base_url}/v1", api_key=agent.api_key, max_retries=0)
        completion = await client.chat.completions.create(
            model="policy",
            messages=[{"role": "user", "content": "go"}],
            tools=TOOLS,
            max_tokens=40,
            logprobs=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            extra_headers={"X-Platoon-Call-Kind": "aux"},
        )
        await client.close()
        return completion

    agent, completion = asyncio.run(serve(hosted(provider), use))

    (sent,) = provider.calls(CHAT)
    assert sent["body"] == {
        "model": MODEL,
        "messages": [{"role": "user", "content": "go"}],
        "tools": TOOLS,
        "max_tokens": 40,
        "logprobs": True,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    assert sent["headers"]["authorization"] == f"Bearer {FAKE_KEY}"  # the provider's key, never the agent's
    assert "x-platoon-call-kind" not in sent["headers"] and agent.api_key not in json.dumps(sent)
    # The harness gets the provider's completion: its id, served model and fields, unchanged.
    assert completion.id.startswith("chatcmpl-") and completion.model == SERVED_MODEL
    message = completion.choices[0].message
    assert message.tool_calls[0].id == "call_x" and message.model_extra["provider_extra"] == {"kept": True}
    assert message.model_extra["reasoning_content"] == "hm"
    assert completion.usage.prompt_tokens_details.cached_tokens == 64
    # Recorded on the agent as an aux call, with the provider's record.
    (call,) = agent.calls
    assert agent.call_kind(call.call_id) == "aux" and call.tool_calls[0]["id"] == "call_x"
    step = build_agent_trajectory(agent, task=None).steps[0]
    assert step.misc["action_misc"]["kind"] == "aux" and step.misc["action_misc"]["dialect"] == "openai_chat"
    assert step.misc["action_misc"]["served_model"] == SERVED_MODEL


def test_a_streaming_chat_harness_gets_the_provider_s_chunks(provider):
    provider.replies[CHAT] = lambda body: {"content": "streamed answer", "reasoning_content": "why"}

    async def use(base_url, agent):
        client = openai.AsyncOpenAI(base_url=f"{base_url}/v1", api_key=agent.api_key, max_retries=0)
        stream = await client.chat.completions.create(
            model="policy",
            messages=[{"role": "user", "content": "go"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks = [chunk async for chunk in stream]
        await client.close()
        return chunks

    agent, chunks = asyncio.run(serve(hosted(provider), use))

    text = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
    reasoning = "".join(
        (chunk.choices[0].delta.model_extra or {}).get("reasoning_content") or "" for chunk in chunks if chunk.choices
    )
    assert text == "streamed answer" and reasoning == "why"
    assert chunks[-1].usage.completion_tokens == 7 and chunks[0].model == SERVED_MODEL
    assert agent.calls[0].content == "streamed answer" and agent.calls[0].reasoning_content == "why"
    assert agent.calls[0].provider.usage_reported


def test_provider_errors_reach_the_harness_with_their_status_and_body(provider):
    weird = openai_error("messages[0].content: something the provider dislikes", code="odd_field")
    provider.faults = [
        Fault(400, weird),
        Fault(429, openai_error("slow down", kind="rate_limit_error"), {"retry-after": "0"}),
        Fault(429, openai_error("slow down", kind="rate_limit_error"), {"retry-after": "0"}),
    ]

    async def use(base_url, agent):
        client = openai.AsyncOpenAI(base_url=f"{base_url}/v1", api_key=agent.api_key, max_retries=0)
        errors = []
        for _ in range(2):
            with pytest.raises(openai.APIStatusError) as raised:
                await client.chat.completions.create(model="policy", messages=[{"role": "user", "content": "go"}])
            errors.append(raised.value)
        await client.close()
        return errors

    agent, (invalid, limited) = asyncio.run(serve(hosted(provider), use))

    assert invalid.status_code == 400 and invalid.body == weird["error"] and invalid.code == "odd_field"
    assert limited.status_code == 429 and limited.response.headers["retry-after"] == "0"
    assert agent.calls == [] and agent.misc["backend_failures"]["count"] == 1  # the 429, after its retry


def test_a_key_the_provider_echoes_never_reaches_the_harness(provider):
    provider.faults = [Fault(401, openai_error("key revoked: {key}", code="invalid_api_key"))]

    async def use(base_url, agent):
        async with ClientSession() as session:
            body = {"model": "policy", "messages": [{"role": "user", "content": "go"}]}
            headers = {"Authorization": f"Bearer {agent.api_key}"}
            async with session.post(f"{base_url}/v1/chat/completions", json=body, headers=headers) as reply:
                return reply.status, await reply.read()

    _, (status, body) = asyncio.run(serve(hosted(provider), use))

    assert status == 401 and FAKE_KEY.encode() not in body and b"key revoked: [redacted]" in body


# --------------------------------------------------------------------------- Messages


def test_a_messages_harness_gets_the_provider_s_messages_with_their_thinking_signatures(provider):
    signatures: list[str] = []

    def reply(body: dict) -> dict:
        signature = f"sig-{len(signatures)}"
        signatures.append(signature)
        return {
            "content": [
                {"type": "thinking", "thinking": "bisect", "signature": signature},
                tool_use("guess", {"number": 50}, call_id=f"toolu_{len(signatures)}"),
            ]
        }

    provider.replies[MESSAGES] = reply
    thinking = {"type": "enabled", "budget_tokens": 1024}

    async def use(base_url, agent):
        client = anthropic.AsyncAnthropic(base_url=base_url, api_key=agent.api_key, max_retries=0)
        tools = [{"name": "guess", "input_schema": {"type": "object", "properties": {}}}]
        turns: list[dict[str, Any]] = [{"role": "user", "content": "find it"}]
        first = await client.messages.create(
            model="policy", max_tokens=2048, messages=turns, tools=tools, thinking=thinking
        )
        turns += [
            {"role": "assistant", "content": [block.model_dump(exclude_none=True) for block in first.content]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "Too low"}]},
        ]
        async with client.messages.stream(
            model="policy", max_tokens=2048, messages=turns, tools=tools, thinking=thinking
        ) as stream:
            second = await stream.get_final_message()
        await client.close()
        return first, second

    agent, (first, second) = asyncio.run(serve(hosted(provider), use))

    assert first.model == SERVED_MODEL and first.content[0].signature == "sig-0"
    assert second.content[0].signature == "sig-1" and second.content[1].input == {"number": 50}
    replayed = provider.calls(MESSAGES)[1]["body"]["messages"][1]["content"][0]
    assert replayed == {"type": "thinking", "thinking": "bisect", "signature": "sig-0"}  # sent back unchanged
    for sent in provider.calls(MESSAGES):
        assert sent["headers"]["x-api-key"] == FAKE_KEY and sent["headers"]["anthropic-version"]
        assert sent["body"]["thinking"] == thinking and sent["body"]["model"] == MODEL
    assert [call.reasoning_content for call in agent.calls] == ["bisect", "bisect"]
    assert agent.calls[1].usage()["cache_write_tokens"] == 10


def test_count_tokens_is_the_provider_s_and_is_not_recorded(provider):
    async def use(base_url, agent):
        client = anthropic.AsyncAnthropic(base_url=base_url, api_key=agent.api_key, max_retries=0)
        counted = await client.messages.count_tokens(model="policy", messages=[{"role": "user", "content": "hi"}])
        await client.close()
        return counted

    agent, counted = asyncio.run(serve(hosted(provider), use))

    assert counted.input_tokens > 1000 and agent.calls == []
    (sent,) = provider.calls("/messages/count_tokens")
    assert sent["body"] == {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}


# --------------------------------------------------------------------------- Responses


def test_a_responses_harness_gets_the_provider_s_response(provider):
    sealed = "gAAAAB-provider-encrypted-reasoning"

    def reply(body: dict) -> dict:
        reasoning = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": sealed}
        return {"output": [reasoning, function_call("guess", {"number": 50}, call_id="call_1")]}

    provider.replies[RESPONSES] = reply
    tools = [{"type": "function", "name": "guess", "parameters": {"type": "object", "properties": {}}}]

    async def use(base_url, agent):
        client = openai.AsyncOpenAI(base_url=f"{base_url}/v1", api_key=agent.api_key, max_retries=0)
        first = await client.responses.create(
            model="policy", input="find it", tools=tools, include=["reasoning.encrypted_content"], store=False
        )
        history = [
            {"role": "user", "content": "find it"},
            *[item.model_dump(exclude_none=True) for item in first.output],
            {"type": "function_call_output", "call_id": "call_1", "output": "Too low"},
        ]
        stream = await client.responses.create(model="policy", input=history, tools=tools, stream=True)
        events = [event async for event in stream]
        await client.close()
        return first, events

    agent, (first, events) = asyncio.run(serve(hosted(provider), use))

    assert first.model == SERVED_MODEL and first.output[1].name == "guess"
    assert events[-1].type == "response.completed" and events[-1].response.output[1].call_id == "call_1"
    second = provider.calls(RESPONSES)[1]["body"]
    assert second["input"][1]["encrypted_content"] == sealed and second["stream"] is True
    assert second["tools"] == [{**tools[0], "strict": False}]
    assert [call.tool_calls[0]["id"] for call in agent.calls] == ["call_1", "call_1"]


# --------------------------------------------------------------------------- streams and keepalives


async def _raw(base_url: str, agent, path: str, body: dict[str, Any]) -> tuple[int, str, bytes]:
    headers = {"Authorization": f"Bearer {agent.api_key}", "x-api-key": agent.api_key}
    async with ClientSession() as session:
        async with session.post(f"{base_url}/v1{path}", json=body, headers=headers) as reply:
            return reply.status, reply.headers.get("Content-Type", ""), await reply.read()


def test_a_slow_stream_gets_keepalives_then_the_provider_s_events(provider):
    provider.delay = 0.4
    body = {"model": "policy", "messages": [{"role": "user", "content": "go"}], "stream": True}

    async def use(base_url, agent):
        return await _raw(base_url, agent, CHAT, body)

    agent, (status, content_type, raw) = asyncio.run(serve(hosted(provider), use, keepalive_seconds=0.05))

    assert status == 200 and content_type.startswith("text/event-stream")
    assert raw.startswith(b": keepalive\n\n") and raw.count(b": keepalive") >= 2
    assert raw.rstrip().endswith(b"data: [DONE]") and len(agent.calls) == 1


def test_an_early_stream_error_keeps_its_status_and_a_late_one_is_an_error_event(provider):
    body = {"model": "policy", "max_tokens": 64, "messages": [{"role": "user", "content": "go"}], "stream": True}
    overloaded = anthropic_error("Overloaded", kind="overloaded_error")

    async def use(base_url, agent):
        provider.faults = [Fault(400, anthropic_error("messages: bad"))]
        early = await _raw(base_url, agent, MESSAGES, body)
        provider.faults = [Fault(529, overloaded, delay=0.3), Fault(529, overloaded, delay=0.3)]
        late = await _raw(base_url, agent, MESSAGES, body)
        return early, late

    agent, (early, late) = asyncio.run(serve(hosted(provider), use, keepalive_seconds=0.1))

    assert early[0] == 400 and early[1].startswith("application/json")
    assert json.loads(early[2])["error"]["message"] == "messages: bad"
    status, content_type, raw = late
    assert status == 200 and content_type.startswith("text/event-stream") and raw.startswith(b": keepalive")
    event = raw.split(b"event: error\n", 1)[1]
    assert json.loads(event.split(b"data: ", 1)[1]) == overloaded  # the provider's own error, in the stream
    assert agent.calls == [] and agent.misc["backend_failures"]["count"] == 1


@pytest.mark.parametrize(
    ("path", "body", "error", "event"),
    [
        (
            CHAT,
            {"messages": [{"role": "user", "content": "go"}]},
            openai_error("overloaded", kind="server_error"),
            None,
        ),
        (RESPONSES, {"input": "go"}, openai_error("overloaded", kind="server_error"), "response.failed"),
        (
            MESSAGES,
            {"max_tokens": 64, "messages": [{"role": "user", "content": "go"}]},
            anthropic_error("overloaded"),
            "error",
        ),
    ],
    ids=["chat", "responses", "messages"],
)
def test_a_failure_after_the_stream_started_is_the_dialect_s_error_event(provider, path, body, error, event):
    provider.faults = [Fault(503, error, delay=0.3), Fault(503, error, delay=0.3)]

    async def use(base_url, agent):
        return await _raw(base_url, agent, path, {"model": "policy", **body, "stream": True})

    agent, (status, _, raw) = asyncio.run(serve(hosted(provider), use, keepalive_seconds=0.1))

    assert status == 200 and raw.startswith(b": keepalive")
    name, data = hosted_module.sse_events(raw)[-1]
    sent = json.loads(data)
    failure = sent["response"] if event == "response.failed" else sent  # a Responses failure wraps its error
    assert name == event and failure["error"]["message"] == "overloaded"
    assert agent.calls == [] and agent.misc["backend_failures"]["count"] == 1


def test_a_call_without_a_reply_is_the_dialect_s_timeout(provider):
    provider.delay = 1.0
    body = {"model": "policy", "messages": [{"role": "user", "content": "go"}]}

    async def use(base_url, agent):
        return await _raw(base_url, agent, CHAT, body)

    _, (status, _, raw) = asyncio.run(serve(hosted(provider, timeout_s=0.2, max_retries=0), use))

    assert status == 504 and json.loads(raw)["error"]["type"] == "timeout"


# --------------------------------------------------------------------------- what the gateway answers itself


def test_models_unknown_routes_and_refused_judges_are_the_gateway_s(provider):
    async def use(base_url, agent):
        headers = {"Authorization": f"Bearer {agent.api_key}"}
        async with ClientSession() as session:
            async with session.get(f"{base_url}/v1/models", headers=headers) as reply:
                models = await reply.json()
            async with session.post(f"{base_url}/v1/responses/compact", json={}, headers=headers) as reply:
                compact = reply.status
            agent.judges = False
            body = {"model": "policy", "messages": [{"role": "user", "content": "go"}]}
            judge = {**headers, "X-Platoon-Call-Kind": "judge"}
            async with session.post(f"{base_url}/v1/chat/completions", json=body, headers=judge) as reply:
                return models, compact, reply.status, await reply.json()

    agent, (models, compact, status, refused) = asyncio.run(serve(hosted(provider), use))

    assert [model["id"] for model in models["data"]] == [MODEL]
    assert compact == 404 and status == 403 and refused["error"]["type"] == "permission_error"
    assert provider.requests == [] and agent.misc["judges_refused"]["count"] == 1


def test_a_successful_reply_carries_the_call_id(provider):
    provider.replies[RESPONSES] = lambda body: {"output": [output_message("hi")]}

    async def use(base_url, agent):
        headers = {"Authorization": f"Bearer {agent.api_key}"}
        async with ClientSession() as session:
            async with session.post(
                f"{base_url}/v1/responses", json={"model": "p", "input": "go"}, headers=headers
            ) as r:
                return r.headers.get(CALL_ID_HEADER), await r.json()

    agent, (call_id, reply) = asyncio.run(serve(hosted(provider), use))

    assert call_id == agent.calls[0].call_id and reply["output"][0]["content"][0]["text"] == "hi"


# --------------------------------------------------------------------------- step budgets and call kinds

DIALECT_ROUTES = [
    ("/v1/chat/completions", CHAT, {"model": "policy", "messages": [{"role": "user", "content": "go"}]}),
    ("/v1/responses", RESPONSES, {"model": "policy", "input": "go"}),
    ("/v1/messages", MESSAGES, {"model": "policy", "max_tokens": 64, "messages": [{"role": "user", "content": "go"}]}),
]


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
@pytest.mark.parametrize(("route", "path", "body"), DIALECT_ROUTES, ids=["chat", "responses", "messages"])
def test_a_hosted_policy_call_past_the_step_budget_gets_the_budget_stop_and_is_not_forwarded(
    provider, route, path, body, stream
):
    """The gateway step budget holds for a hosted model as for Tinker: a policy call past the agent's call cap gets
    the budget stop in the dialect's own response or stream, and nothing reaches the provider."""

    async def use(base_url, agent):
        agent.call_cap = 1
        headers = {"x-api-key": agent.api_key, "anthropic-version": "2023-06-01"}
        replies = []
        async with ClientSession() as session:
            for _ in range(2):
                async with session.post(f"{base_url}{route}", json={**body, "stream": stream}, headers=headers) as r:
                    replies.append((r.status, await r.text()))
        return replies

    agent, replies = asyncio.run(serve(hosted(provider), use))

    assert [status for status, _ in replies] == [200, 200]
    assert "step budget" not in replies[0][1] and "The step budget of 1 model calls is used up" in replies[1][1]
    assert len(provider.calls(path)) == 1  # the stop is the gateway's own answer
    assert (agent.policy_calls, agent.budget_stops, agent.call_tally.recorded, agent.call_tally.in_flight) == (
        1,
        1,
        1,
        0,
    )
    assert agent.misc["budget"] == {"enforce": "gateway", "max_steps": 1, "stops": 1}


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
@pytest.mark.parametrize("minted", [False, True], ids=["one-agent", "minted-agents"])
def test_hosted_policy_calls_that_arrive_together_never_pass_the_step_budget(provider, stream, minted):
    """Six policy calls at once against a cap of one, from one agent or from six agents it minted: one reaches the
    provider and the others get the budget stop, as with Tinker."""

    provider.delay = 0.3
    gateway = hosted(provider)

    async def use(base_url, agent):
        agent.call_cap = 1
        keys = [agent.api_key] * 6
        if minted:  # as POST /v1/platoon/agents mints them
            keys = [gateway.mint_subagent(agent, role="helper").api_key for _ in keys]
        body = {"model": "policy", "messages": [{"role": "user", "content": "go"}], "stream": stream}
        async with ClientSession() as session:

            async def call(key):
                headers = {"Authorization": f"Bearer {key}"}
                async with session.post(f"{base_url}/v1/chat/completions", json=body, headers=headers) as reply:
                    return reply.status, await reply.text()

            return await asyncio.gather(*(call(key) for key in keys))

    agent, replies = asyncio.run(serve(gateway, use))

    assert [status for status, _ in replies] == [200] * 6
    assert len(provider.calls(CHAT)) == 1
    assert sum("The step budget of 1 model calls is used up" in text for _, text in replies) == 5
    assert (agent.call_tally.recorded, agent.call_tally.in_flight) == (1, 0)


@pytest.mark.parametrize(("kind", "call_cap"), [("aux", 5), ("judge", 5), ("judge", None)])
def test_a_policy_only_agent_s_hosted_aux_and_judge_calls_are_refused_unforwarded(provider, kind, call_cap):
    """A judge call always, and an aux call under a call cap, which counts only policy calls."""

    async def use(base_url, agent):
        agent.policy_only, agent.call_cap = True, call_cap
        headers = {"Authorization": f"Bearer {agent.api_key}", "X-Platoon-Call-Kind": kind}
        body = {"model": "policy", "messages": [{"role": "user", "content": "go"}]}
        async with ClientSession() as session:
            async with session.post(f"{base_url}/v1/chat/completions", json=body, headers=headers) as reply:
                return reply.status, await reply.json()

    agent, (status, refused) = asyncio.run(serve(hosted(provider), use))

    assert status == 403 and refused["error"]["type"] == "permission_error"
    assert f"its {kind} calls are refused" in refused["error"]["message"]
    assert provider.requests == [] and agent.calls == [] and "judges_refused" not in agent.misc


def test_a_policy_only_agent_without_a_cap_has_its_hosted_aux_call_forwarded(provider):
    async def use(base_url, agent):
        agent.policy_only = True
        headers = {"Authorization": f"Bearer {agent.api_key}", "X-Platoon-Call-Kind": "aux"}
        body = {"model": "policy", "messages": [{"role": "user", "content": "go"}]}
        async with ClientSession() as session:
            async with session.post(f"{base_url}/v1/chat/completions", json=body, headers=headers) as reply:
                return reply.status, reply.headers.get(CALL_ID_HEADER)

    agent, (status, call_id) = asyncio.run(serve(hosted(provider), use))

    assert status == 200 and len(provider.calls(CHAT)) == 1 and agent.call_ids == [call_id]
    assert agent.call_kind(call_id) == "aux" and agent.policy_calls == 0
