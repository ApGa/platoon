"""How a hosted gateway forwards calls to its base URL (``platoon.gateway.hosted``), against a fake provider over real
HTTP: retries with one budget and ``retry-after``, per-attempt timeouts, streams cut off or failed midway, what is
counted on the agent, what is relayed, the key's redaction, what is sent (headers, query, body), the request limit,
sessions per event loop, the in-process path, ``count_tokens`` and the probe."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import pytest
from aiohttp import ClientSession
from hosted_fakes import (
    CHAT,
    COUNT_TOKENS,
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

from platoon.eval.config_defs import EvalConfig, HostedConfig
from platoon.gateway import GatewayServer, HostedCallError
from platoon.gateway import hosted as hosted_module
from platoon.gateway.core import (
    CallKindRefusedError,
    ChatRequest,
    ContextWindowExceededError,
    HostedRefusedError,
    HostedRequestError,
    JudgesOffError,
    SamplingOverrides,
)
from platoon.gateway.hosted import (
    BACKEND_FAILURES_MISC_KEY,
    HostedGateway,
    ProbeError,
    backoff_seconds,
    context_length_error,
    env_proxy,
    error_message,
    redact,
    redact_bytes,
)
from platoon.gateway.trace import build_agent_trajectory
from platoon.utils.config import ConfigError

KEY_ENV = "PLATOON_TEST_PROVIDER_KEY"
PROXY_VARIABLES = ("http_proxy", "https_proxy", "all_proxy", "no_proxy")
CHAT_BODY = {"model": "anything", "messages": [{"role": "user", "content": "go"}]}
MESSAGES_BODY = {"model": "anything", "max_tokens": 64, "messages": [{"role": "user", "content": "go"}]}
RESPONSES_BODY = {"model": "anything", "input": "go"}
GUESS = {"type": "function", "function": {"name": "guess", "parameters": {"type": "object", "properties": {}}}}


@pytest.fixture(autouse=True)
def _environment(monkeypatch):
    monkeypatch.setenv(KEY_ENV, FAKE_KEY)
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setattr(hosted_module, "RETRY_BACKOFF_S", (0.01, 0.02))


@pytest.fixture
def provider():
    with FakeProvider() as fake:
        yield fake


def make_gateway(provider: FakeProvider, **settings: Any) -> HostedGateway:
    options: dict[str, Any] = {"max_retries": 2, **settings}
    return HostedGateway(HostedConfig(base_url=provider.base_url, api_key_env=KEY_ENV, **options), MODEL)


def open_agent(gateway: HostedGateway, **kwargs: Any):
    return gateway.open_agent(rollout_id="r", trajectory_id="t", interactions={}, **kwargs)


def forward(gateway: HostedGateway, agent, dialect: str, body: dict[str, Any], headers=None, kind: str = "policy"):
    path = {"openai_chat": CHAT, "anthropic_messages": MESSAGES, "openai_responses": RESPONSES}[dialect]

    async def go():
        try:
            return await gateway.forward(agent, dialect, path, "", body, headers or {}, kind)
        finally:
            await gateway.aclose()

    return asyncio.run(go())


def chat_in_process(gateway: HostedGateway, agent, messages, **params):
    async def go():
        try:
            return await gateway.chat(agent, ChatRequest.from_params(messages, params))
        finally:
            await gateway.aclose()

    return asyncio.run(go())


def failures(agent) -> dict[str, Any]:
    return agent.misc.get(BACKEND_FAILURES_MISC_KEY, {})


# --------------------------------------------------------------------------- retries, timeouts, broken streams


def test_retries_honour_retry_after_and_record_their_attempts(provider):
    provider.faults = [
        Fault(429, openai_error("slow down", kind="rate_limit_error"), {"retry-after-ms": "5"}),
        Fault(503, openai_error("overloaded", kind="server_error"), {"retry-after": "0"}),
    ]
    gateway = make_gateway(provider)
    agent = open_agent(gateway)

    relay = forward(gateway, agent, "openai_chat", CHAT_BODY)

    assert relay.status == 200 and len(provider.calls(CHAT)) == 3
    record = agent.calls[0].provider
    assert (record.attempts, record.rate_limited, record.request_id) == (3, 1, "req_fake")
    assert failures(agent) == {}


def test_every_failure_shares_one_budget_and_the_last_reply_is_relayed(provider):
    provider.faults = [
        Fault(503, openai_error("overloaded", kind="server_error")),
        Fault(429, openai_error("slow down", kind="rate_limit_error"), {"retry-after": "0"}),
        Fault(500, openai_error("never sent", kind="server_error")),
    ]
    gateway = make_gateway(provider, max_retries=1)
    agent = open_agent(gateway)

    relay = forward(gateway, agent, "openai_chat", CHAT_BODY)

    assert len(provider.calls(CHAT)) == 2  # max_retries + 1 attempts in all
    assert relay.status == 429 and relay.headers == {"retry-after": "0"} and relay.response is None
    assert relay.json()["error"]["message"] == "slow down"
    assert agent.calls == [] and failures(agent) == {"count": 1, "kinds": {"rate_limit": 1}, "rejected": 0}


@pytest.mark.parametrize(
    ("headers", "attempt", "expected"),
    [
        ({"retry-after-ms": "1500"}, 1, 1.5),
        ({"retry-after": "2"}, 1, 2.0),
        ({"retry-after": "120"}, 1, 60.0),
        ({"retry-after-ms": "250", "retry-after": "9"}, 1, 0.25),
    ],
)
def test_a_retry_waits_as_long_as_the_reply_asks_up_to_a_minute(headers, attempt, expected):
    assert backoff_seconds(attempt, headers) == expected


def test_without_retry_after_the_backoff_doubles_with_jitter(monkeypatch):
    monkeypatch.setattr(hosted_module, "RETRY_BACKOFF_S", (0.5, 8.0))
    assert 0.5 <= backoff_seconds(1) <= 0.625 and 1.0 <= backoff_seconds(2) <= 1.25
    assert 8.0 <= backoff_seconds(10) <= 10.0
    soon = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(time.time() + 30))
    assert 25 <= backoff_seconds(1, {"retry-after": soon}) <= 30


def test_each_attempt_has_timeout_s_and_a_call_without_a_reply_fails_as_a_timeout(provider):
    provider.delay = 1.0
    gateway = make_gateway(provider, timeout_s=0.2, max_retries=1)
    agent = open_agent(gateway)
    started = time.perf_counter()

    with pytest.raises(HostedCallError) as raised:
        forward(gateway, agent, "openai_chat", CHAT_BODY)

    assert time.perf_counter() - started < 1.5  # two attempts of 0.2 s, not the provider's 1 s each
    assert raised.value.kind == "timeout" and raised.value.attempts == 2
    assert len(provider.calls(CHAT)) == 2 and failures(agent)["kinds"] == {"timeout": 1}


@pytest.mark.parametrize(
    ("dialect", "path", "body"),
    [
        ("openai_chat", CHAT, {**CHAT_BODY, "stream": True}),
        ("anthropic_messages", MESSAGES, {**MESSAGES_BODY, "stream": True}),
        ("openai_responses", RESPONSES, {**RESPONSES_BODY, "stream": True}),
    ],
)
def test_a_stream_cut_off_midway_is_sent_again(provider, dialect, path, body):
    provider.cut_streams = 1
    gateway = make_gateway(provider)
    agent = open_agent(gateway)

    relay = forward(gateway, agent, dialect, body)

    assert relay.status == 200 and relay.content_type.startswith("text/event-stream")
    assert len(provider.calls(path)) == 2 and agent.calls[0].provider.attempts == 2
    assert agent.calls[0].content == "OK"


def test_a_stream_cut_off_at_every_attempt_fails_without_a_reply(provider):
    provider.cut_streams = 5
    gateway = make_gateway(provider, max_retries=1)
    agent = open_agent(gateway)

    with pytest.raises(HostedCallError) as raised:
        forward(gateway, agent, "anthropic_messages", {**MESSAGES_BODY, "stream": True})

    assert raised.value.kind == "server" and failures(agent)["kinds"] == {"server": 1}


@pytest.mark.parametrize(
    ("dialect", "path", "body", "marker"),
    [
        ("openai_chat", CHAT, {**CHAT_BODY, "stream": True}, b'"upstream failed"'),
        ("anthropic_messages", MESSAGES, {**MESSAGES_BODY, "stream": True}, b"event: error"),
        ("openai_responses", RESPONSES, {**RESPONSES_BODY, "stream": True}, b"response.failed"),
    ],
)
def test_an_error_event_in_a_stream_is_retried_then_relayed_as_it_came(provider, dialect, path, body, marker):
    provider.error_streams = 1
    gateway = make_gateway(provider)
    agent = open_agent(gateway)
    assert forward(gateway, agent, dialect, body).status == 200 and len(agent.calls) == 1

    provider.error_streams = 5
    agent = open_agent(gateway)
    relay = forward(gateway, agent, dialect, body)

    assert relay.status == 200 and marker in relay.body and relay.response is None
    assert agent.calls == [] and failures(agent)["count"] == 1


# --------------------------------------------------------------------------- what is counted and relayed


def test_a_refused_key_is_relayed_and_marks_the_run_fatal(provider):
    provider.faults = [Fault(401, openai_error("Incorrect API key provided.", code="invalid_api_key"))]
    gateway = make_gateway(provider)
    agent = open_agent(gateway)

    relay = forward(gateway, agent, "openai_chat", CHAT_BODY)

    assert relay.status == 401 and len(provider.calls(CHAT)) == 1  # never retried
    record = failures(agent)
    assert record["count"] == 1 and record["kinds"] == {"refused": 1}
    assert record["fatal"] == f"{provider.origin} refused an openai_chat call: 401 Incorrect API key provided."


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_a_redirect_is_never_followed_so_the_key_reaches_no_other_origin(provider, status):
    with FakeProvider(api_key=None) as elsewhere:
        provider.faults = [
            Fault(status, {"moved": True}, {"Location": f"{elsewhere.base_url}{path}"}) for path in (MESSAGES, CHAT)
        ]
        gateway = make_gateway(provider)
        agent = open_agent(gateway)

        relays = [
            forward(gateway, agent, "anthropic_messages", MESSAGES_BODY, {"anthropic-version": "2023-06-01"}),
            forward(gateway, agent, "openai_chat", CHAT_BODY),
        ]

        # Neither the request nor the key went to the other origin: aiohttp would have kept Messages' x-api-key
        # across origins, and sent the body (or, after a 301 or 302, a GET with the key) there.
        assert elsewhere.requests == [] and elsewhere.others == []
    assert [relay.status for relay in relays] == [status, status] and agent.calls == []
    assert all("location" not in relay.headers for relay in relays)  # nothing to follow reaches the harness
    assert len(provider.requests) == 2  # a redirect refuses the run, so it is not retried
    record = failures(agent)
    assert record["count"] == 2 and record["kinds"] == {"refused": 2}
    assert record["fatal"] == (
        f"{provider.origin} refused an anthropic_messages call: {status} the base URL redirects to "
        f"{elsewhere.origin}, and hosted calls follow no redirect: set inference.hosted.base_url to the URL that "
        "answers"
    )

    provider.faults = [Fault(status, {}, {"Location": "/elsewhere/v1/chat/completions"})]
    with pytest.raises(ProbeError) as raised:
        asyncio.run(make_gateway(provider).probe(["openai_chat"]))
    assert raised.value.status == status
    assert raised.value.detail.startswith(f"{status} the base URL redirects to {provider.origin},")


def test_an_invalid_request_is_the_harness_s_and_is_relayed_unchanged(provider, caplog):
    body = openai_error("messages[0].content: something the provider dislikes", code="odd_field")
    provider.faults = [Fault(400, body)]
    gateway = make_gateway(provider)
    agent = open_agent(gateway)
    caplog.set_level(logging.WARNING, logger="platoon.gateway.hosted")

    relay = forward(gateway, agent, "openai_chat", CHAT_BODY)

    assert (relay.status, relay.json()) == (400, body) and relay.content_type.startswith("application/json")
    assert failures(agent) == {"count": 0, "kinds": {}, "rejected": 1} and agent.context_overflows == 0
    assert "error type invalid_request_error" in caplog.text and "dislikes" not in caplog.text


# Context-length refusals as each server words them: (server, status, body, prompt tokens, context window).
CONTEXT_LENGTH_REPLIES = [
    (
        "openai",
        400,
        openai_error(
            "This model's maximum context length is 128000 tokens. However, your messages resulted in 130523 tokens. "
            "Please reduce the length of the messages.",
            code="context_length_exceeded",
        ),
        130523,
        128000,
    ),
    (
        "openai, newer",
        400,
        openai_error(
            "Input tokens exceed the configured limit of 272000 tokens. Your messages resulted in 280000 tokens. "
            "Please reduce the length of the messages.",
            code="context_length_exceeded",
        ),
        280000,
        272000,
    ),
    (
        "openai responses",
        400,
        openai_error(
            "Your input exceeds the context window of this model. Please adjust your input and try again.",
            code="context_length_exceeded",
        ),
        0,
        0,
    ),
    ("anthropic", 400, anthropic_error("prompt is too long: 208000 tokens > 200000 maximum"), 208000, 200000),
    (
        "anthropic, with max_tokens",
        400,
        anthropic_error("input length and `max_tokens` exceed context limit: 198000 + 4096 > 200000, decrease input"),
        198000,
        200000,
    ),
    (
        "vllm",
        400,
        {
            "object": "error",
            "message": "This model's maximum context length is 32768 tokens. However, you requested 33000 tokens "
            "(32000 in the messages, 1000 in the completion). Please reduce the length of the messages or completion.",
            "type": "BadRequestError",
            "param": None,
            "code": 400,
        },
        33000,
        32768,
    ),
    (
        "vllm, newer",
        400,
        {
            "error": {
                "message": "This model's maximum context length is 32768 tokens. However, you requested 1000 output "
                "tokens and your prompt contains at least 32001 input tokens, for a total of at least 33001 tokens. "
                "Please reduce the length of the input prompt or the number of requested output tokens.",
                "type": "BadRequestError",
                "param": None,
                "code": 400,
            }
        },
        32001,
        32768,
    ),
    (
        "sglang",
        400,
        {
            "object": "error",
            "message": "The input (9000 tokens) is longer than the model's context length (8192 tokens).",
            "type": "BadRequestError",
            "param": None,
            "code": 400,
        },
        9000,
        8192,
    ),
    (
        "litellm proxy",
        400,
        {
            "error": {
                "message": "litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: "
                "OpenAIException - Input is too long for the requested model.",
                "type": None,
                "param": None,
                "code": "400",
            }
        },
        0,
        0,
    ),
    (
        "openrouter",
        400,
        {
            "error": {
                "message": "This endpoint's maximum context length is 131072 tokens. However, you requested about "
                "150000 tokens (150000 of text input). Please reduce the length of either one, or use the "
                '"middle-out" transform to compress your prompt automatically.',
                "code": 400,
            }
        },
        150000,
        131072,
    ),
]


@pytest.mark.parametrize(
    ("server", "status", "body", "prompt", "window"), CONTEXT_LENGTH_REPLIES, ids=[c[0] for c in CONTEXT_LENGTH_REPLIES]
)
def test_context_length_refusals_of_each_server_are_recognized(server, status, body, prompt, window):
    error = context_length_error(status, error_message(json.dumps(body).encode()))
    assert isinstance(error, ContextWindowExceededError)
    assert (error.prompt_tokens, error.context_window) == (prompt, window)


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (429, openai_error("Rate limit reached for requests: too many tokens per minute")),
        (400, openai_error("messages.0.content: Field required")),
        (413, anthropic_error("request_too_large: the request exceeds the context window", kind="request_too_large")),
        (400, anthropic_error("max_tokens: 64000 > 32000, which is the maximum allowed number of output tokens")),
    ],
)
def test_other_refusals_are_not_context_length_errors(status, body):
    assert context_length_error(status, error_message(json.dumps(body).encode())) is None


def test_a_context_length_refusal_is_relayed_and_counts_as_a_context_overflow(provider):
    provider.faults = [Fault(400, anthropic_error("prompt is too long: 208000 tokens > 200000 maximum"))]
    gateway = make_gateway(provider)
    agent = open_agent(gateway)

    relay = forward(gateway, agent, "anthropic_messages", MESSAGES_BODY)

    assert relay.status == 400 and relay.json()["error"]["message"].startswith("prompt is too long")
    assert agent.context_overflows == 1 and failures(agent) == {}


# --------------------------------------------------------------------------- the key


def test_a_key_the_provider_echoes_is_redacted_everywhere(provider, caplog):
    caplog.set_level(logging.DEBUG)
    provider.faults = [Fault(401, openai_error("key revoked: {key}", code="invalid_api_key"))]
    gateway = make_gateway(provider)
    agent = open_agent(gateway)

    relay = forward(gateway, agent, "openai_chat", CHAT_BODY)

    assert relay.json()["error"]["message"] == "key revoked: [redacted]"
    assert FAKE_KEY not in failures(agent)["fatal"] and "[redacted]" in failures(agent)["fatal"]
    provider.replies[CHAT] = lambda body: {"content": f"your key is {FAKE_KEY}"}
    agent = open_agent(gateway)
    relay = forward(gateway, agent, "openai_chat", CHAT_BODY)
    assert FAKE_KEY.encode() not in relay.body and agent.calls[0].content == "your key is [redacted]"
    assert FAKE_KEY not in caplog.text


@pytest.mark.parametrize(
    "form",
    [
        "sk-abc/def+ghi=jkl0123",  # as it is
        "sk-abc\\/def+ghi=jkl0123",  # JSON with "/" escaped, as some encoders write it
        "sk-abc%2Fdef%2Bghi%3Djkl0123",  # percent-encoded, say in a URL an error repeats
        "sk-abc%2fdef%2bghi%3djkl0123",
    ],
)
def test_a_key_is_redacted_in_each_form_a_reply_may_echo_it_in(form):
    key = "sk-abc/def+ghi=jkl0123"
    assert redact(f"bad key {form}!", [key]) == "bad key [redacted]!"
    assert redact_bytes(f'{{"error": "bad key {form}"}}'.encode(), [key]) == b'{"error": "bad key [redacted]"}'


def test_a_key_is_read_without_the_whitespace_around_it(provider, monkeypatch):
    monkeypatch.setenv(KEY_ENV, f" {FAKE_KEY}\r\n")  # as a variable read from a file may hold it
    gateway = make_gateway(provider)
    agent = open_agent(gateway)

    assert forward(gateway, agent, "anthropic_messages", MESSAGES_BODY).status == 200
    assert forward(gateway, agent, "openai_chat", CHAT_BODY).status == 200
    assert provider.requests[0]["headers"]["x-api-key"] == FAKE_KEY  # a header cannot carry the newline
    assert provider.requests[1]["headers"]["authorization"] == f"Bearer {FAKE_KEY}"

    monkeypatch.setenv(KEY_ENV, " \n")
    hosted = HostedConfig(base_url=provider.base_url, api_key_env=KEY_ENV)
    policy = EvalConfig(model_name=MODEL, hosted=hosted).policy()
    assert policy is not None
    with pytest.raises(ConfigError, match=f"inference.hosted: set {KEY_ENV} in the environment"):
        policy.check_environment()


def test_sdk_environment_variables_and_netrc_never_reach_the_provider(provider, monkeypatch, tmp_path):
    netrc = tmp_path / "netrc"
    netrc.write_text("machine 127.0.0.1 login someone password netrc-password\n")
    netrc.chmod(0o600)
    leaks = {
        "OPENAI_ORG_ID": "org-leak",
        "OPENAI_PROJECT_ID": "proj-leak",
        "OPENAI_CUSTOM_HEADERS": "X-Leak: openai",
        "ANTHROPIC_CUSTOM_HEADERS": "X-Leak: anthropic",
        "ANTHROPIC_AUTH_TOKEN": "auth-token-leak",
        "OPENAI_BASE_URL": "http://127.0.0.1:9/v1",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:9",
        "NETRC": str(netrc),
    }
    for name, value in leaks.items():
        monkeypatch.setenv(name, value)
    gateway = make_gateway(provider)
    agent = open_agent(gateway)

    for dialect, body in (
        ("openai_chat", CHAT_BODY),
        ("anthropic_messages", MESSAGES_BODY),
        ("openai_responses", RESPONSES_BODY),
    ):
        assert forward(gateway, agent, dialect, body).status == 200

    for request in provider.requests:
        sent = json.dumps(request["headers"])
        assert not any(value in sent for value in ("org-leak", "proj-leak", "Leak", "auth-token-leak", "Basic"))
        assert {"openai-organization", "openai-project", "x-leak"}.isdisjoint(request["headers"])
    chat_headers, messages_headers = provider.requests[0]["headers"], provider.requests[1]["headers"]
    assert chat_headers["authorization"] == f"Bearer {FAKE_KEY}" and "x-api-key" not in chat_headers
    assert messages_headers["x-api-key"] == FAKE_KEY and "authorization" not in messages_headers


def test_a_server_without_auth_gets_no_key(provider):
    provider.api_key = None
    config = HostedConfig(base_url=provider.base_url)
    gateway = HostedGateway(config, MODEL)
    agent = open_agent(gateway)

    assert forward(gateway, agent, "openai_chat", CHAT_BODY).status == 200
    assert {"authorization", "x-api-key"}.isdisjoint(provider.requests[0]["headers"])


# --------------------------------------------------------------------------- what is sent


def test_only_the_dialect_s_own_client_headers_are_forwarded(provider):
    client = {
        "Authorization": "Bearer agent-key",
        "x-api-key": "agent-key",
        "X-Platoon-Call-Kind": "aux",
        "OpenAI-Organization": "org-client",
        "OpenAI-Project": "proj-client",
        "User-Agent": "claude-cli/9",
        "Connection": "keep-alive",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "interleaved-thinking-2025-05-14",
        "OpenAI-Beta": "responses=experimental",
    }
    gateway = make_gateway(provider)
    agent = open_agent(gateway)

    for dialect, body in (("openai_chat", CHAT_BODY), ("anthropic_messages", MESSAGES_BODY)):
        forward(gateway, agent, dialect, body, client)

    chat_headers, messages_headers = provider.requests[0]["headers"], provider.requests[1]["headers"]
    assert chat_headers["openai-beta"] == "responses=experimental" and "anthropic-beta" not in chat_headers
    assert messages_headers["anthropic-version"] == "2023-06-01" and "openai-beta" not in messages_headers
    assert messages_headers["anthropic-beta"] == "interleaved-thinking-2025-05-14"
    for headers in (chat_headers, messages_headers):
        sent = json.dumps(headers)
        assert "agent-key" not in sent and "org-client" not in sent and "claude-cli" not in sent
        assert {"x-platoon-call-kind", "openai-organization", "openai-project"}.isdisjoint(headers)


def test_the_body_is_forwarded_unchanged_but_for_the_listed_fields(provider):
    gateway = make_gateway(provider)
    agent = open_agent(gateway, sampling_defaults=SamplingOverrides(max_tokens=300, temperature=0.3, top_p=0.9))
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0K"}}
    body = {
        "model": "anything",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "be brief"}]},
            {"role": "user", "content": [{"type": "text", "text": "what is this"}, image]},
            {"role": "assistant", "content": "", "reasoning_content": "hm", "tool_calls": [tool_call("guess", {})]},
            {"role": "tool", "tool_call_id": "call_1", "content": "Too low"},
        ],
        "temperature": 0.7,
        "stop": ["\n\n"],
        "response_format": {"type": "json_object"},
        "logprobs": True,
        "reasoning_effort": "high",
        "metadata": {"run": "x"},
        "store": False,
        "seed": 3,
        "tools": [GUESS],
        "tool_choice": "auto",
        "stream_options": {"include_usage": True},
    }

    forward(gateway, agent, "openai_chat", body)

    sent = provider.calls(CHAT)[0]["body"]
    expected = {key: value for key, value in body.items() if key != "tool_choice"}
    assert sent == {**expected, "model": MODEL, "top_p": 0.9, "max_completion_tokens": 300}


@pytest.mark.parametrize(
    ("dialect", "path", "body", "expected"),
    [
        ("openai_chat", CHAT, {**CHAT_BODY, "max_tokens": 50}, {"max_tokens": 50, "temperature": 0.3, "top_p": 0.9}),
        ("openai_chat", CHAT, CHAT_BODY, {"max_completion_tokens": 300, "temperature": 0.3, "top_p": 0.9}),
        (
            "anthropic_messages",
            MESSAGES,
            {**MESSAGES_BODY, "temperature": 1},
            {"max_tokens": 64, "temperature": 1, "top_p": 0.9},
        ),
        ("openai_responses", RESPONSES, RESPONSES_BODY, {"max_output_tokens": 300, "temperature": 0.3, "top_p": 0.9}),
    ],
)
def test_sampling_parameters_a_request_leaves_unset_come_from_the_rollout(provider, dialect, path, body, expected):
    gateway = make_gateway(provider)
    agent = open_agent(gateway, sampling_defaults=SamplingOverrides(max_tokens=300, temperature=0.3, top_p=0.9))

    forward(gateway, agent, dialect, body)

    sent = provider.calls(path)[0]["body"]
    assert {key: sent[key] for key in expected} == expected
    assert not ({"max_tokens", "max_completion_tokens", "max_output_tokens"} - set(expected)) & set(sent)


def test_without_defaults_nothing_is_added_and_stop_sequences_and_images_pass(provider):
    gateway = make_gateway(provider)
    agent = open_agent(gateway)
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0K"}}
    body = {
        **MESSAGES_BODY,
        "system": [{"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "what is this"}, image]}],
        "stop_sequences": ["END"],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "metadata": {"user_id": "u"},
    }

    forward(gateway, agent, "anthropic_messages", body)

    assert provider.calls(MESSAGES)[0]["body"] == {**body, "model": MODEL}


def test_the_reply_s_model_is_recorded_beside_the_configured_one(provider):
    gateway = make_gateway(provider, pricing={"input": 1.0, "output": 4.0, "cached_input": 0.5})
    agent = open_agent(gateway)

    relay = forward(gateway, agent, "openai_chat", {**CHAT_BODY, "tools": [GUESS]}, kind="aux")

    assert relay.json()["model"] == SERVED_MODEL and relay.response is agent.calls[0]
    (call,) = agent.calls
    assert call.model == MODEL and call.provider.served_model == SERVED_MODEL and not call.trainable
    assert agent.interactions == {} and agent.call_kind(call.call_id) == "aux" and agent.policy_calls == 0
    usage = call.usage()
    assert usage["cached_prompt_tokens"] == 64 and call.provider.cost_usd == pytest.approx(
        ((usage["prompt_tokens"] - 64) * 1.0 + 64 * 0.5 + 7 * 4.0) / 1e6
    )
    action = build_agent_trajectory(agent, task=None).steps[0].misc["action_misc"]
    assert action["dialect"] == "openai_chat" and action["kind"] == "aux" and action["attempts"] == 1
    assert action["served_model"] == SERVED_MODEL and action["request_id"] == "req_fake"


@pytest.mark.parametrize(
    ("dialect", "body", "reply"),
    [
        (
            "anthropic_messages",
            MESSAGES_BODY,
            {
                "content": [
                    {"type": "thinking", "thinking": "think", "signature": "sig-1"},
                    {"type": "text", "text": "hi"},
                    tool_use("guess", {"number": 3}, call_id="toolu_1"),
                ],
                "usage": {"input_tokens": 5, "output_tokens": 9, "output_tokens_details": {"thinking_tokens": 4}},
            },
        ),
        (
            "openai_responses",
            RESPONSES_BODY,
            {
                "output": [
                    {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "think"}]},
                    output_message("hi"),
                    function_call("guess", {"number": 3}, call_id="toolu_1"),
                ]
            },
        ),
    ],
)
def test_each_dialect_s_reply_is_recorded_as_a_message(provider, dialect, body, reply):
    provider.replies[MESSAGES if dialect == "anthropic_messages" else RESPONSES] = lambda _: reply
    gateway = make_gateway(provider)
    agent = open_agent(gateway)

    for stream in (False, True):
        forward(gateway, agent, dialect, {**body, "stream": stream})

    first, second = agent.calls
    assert first.message() == second.message()
    assert first.message() == {
        "role": "assistant",
        "content": "hi",
        "reasoning_content": "think",
        "tool_calls": [
            {"id": "toolu_1", "type": "function", "function": {"name": "guess", "arguments": '{"number": 3}'}}
        ],
    }
    assert first.usage()["reasoning_tokens"] == 4


def test_a_chat_stream_without_include_usage_is_recorded_without_usage(provider):
    gateway = make_gateway(provider, pricing={"input": 1.0, "output": 1.0})
    agent = open_agent(gateway)

    forward(gateway, agent, "openai_chat", {**CHAT_BODY, "stream": True})
    forward(gateway, agent, "openai_chat", {**CHAT_BODY, "stream": True, "stream_options": {"include_usage": True}})

    unreported, reported = (call.provider for call in agent.calls)
    assert not unreported.usage_reported and unreported.cost_usd is None and unreported.usage["prompt_tokens"] == 0
    assert reported.usage_reported and reported.cost_usd > 0
    assert "stream_options" not in provider.calls(CHAT)[0]["body"]  # nothing is added to a harness's request


# --------------------------------------------------------------------------- the request limit and event loops


def test_the_request_limit_bounds_calls_in_flight(provider):
    provider.delay = 0.2
    gateway = make_gateway(provider, max_concurrent_requests=2)
    agent = open_agent(gateway)

    async def go():
        try:
            calls = [gateway.forward(agent, "openai_chat", CHAT, "", CHAT_BODY, {}) for _ in range(6)]
            return await asyncio.gather(*calls)
        finally:
            await gateway.aclose()

    assert all(relay.status == 200 for relay in asyncio.run(go()))
    assert provider.max_in_flight == 2


def test_each_event_loop_gets_its_own_session_closed_with_the_loop(provider):
    gateway = make_gateway(provider)
    agent = open_agent(gateway)

    for _ in range(2):  # a subprocess worker runs each rollout under an asyncio.run of its own
        asyncio.run(gateway.forward(agent, "openai_chat", CHAT, "", CHAT_BODY, {}))
        assert gateway._states._states == {}  # the loop's shutdown closed its session

    assert len(agent.calls) == 2


# --------------------------------------------------------------------------- in process


def test_ctx_chat_is_an_unstreamed_chat_completions_call(provider):
    gateway = make_gateway(provider)
    agent = open_agent(gateway, sampling_defaults=SamplingOverrides(max_tokens=300))
    provider.replies[CHAT] = lambda body: {
        "content": "",
        "reasoning_content": "hm",
        "tool_calls": [tool_call("guess", {})],
    }

    response = chat_in_process(
        gateway,
        agent,
        [{"role": "user", "content": "go"}],
        tools=[GUESS],
        temperature=0.5,
        stream=True,
        extra_body={"reasoning_effort": "low", "top_p": 0.8},
    )

    sent = provider.calls(CHAT)[0]["body"]
    assert sent == {
        "model": MODEL,
        "messages": [{"role": "user", "content": "go"}],
        "tools": [GUESS],
        "temperature": 0.5,
        "reasoning_effort": "low",
        "top_p": 0.8,
        "max_completion_tokens": 300,
    }
    assert response.tool_calls[0]["function"]["name"] == "guess" and response.reasoning_content == "hm"
    assert response.finish_reason == "tool_calls" and agent.calls == [response]


def test_ctx_chat_s_reply_cut_off_at_its_limit_runs_no_tools(provider):
    provider.replies[CHAT] = lambda body: {
        "content": "",
        "tool_calls": [tool_call("guess", {})],
        "finish_reason": "length",
    }
    gateway = make_gateway(provider)
    agent = open_agent(gateway)

    response = chat_in_process(gateway, agent, [{"role": "user", "content": "go"}], tools=[GUESS])

    assert response.tool_calls is None and response.parsed_tool_calls == 1 and response.finish_reason == "length"
    over_http = forward(gateway, agent, "openai_chat", {**CHAT_BODY, "tools": [GUESS]}).response
    assert over_http.tool_calls and over_http.stop_reason == "length"  # the harness got them: the record keeps them


@pytest.mark.parametrize(
    ("fault", "error"),
    [
        (Fault(400, openai_error("bad field")), HostedRequestError),
        (
            Fault(400, openai_error("maximum context length is 10 tokens. However, you requested 20 tokens")),
            ContextWindowExceededError,
        ),
        (Fault(401, openai_error("no", code="invalid_api_key")), HostedRefusedError),
        (Fault(503, openai_error("overloaded", kind="server_error")), HostedCallError),
    ],
)
def test_ctx_chat_raises_what_a_tinker_call_raises(provider, fault, error):
    provider.faults = [fault] * 3
    gateway = make_gateway(provider, max_retries=0)
    agent = open_agent(gateway)

    with pytest.raises(error):
        chat_in_process(gateway, agent, [{"role": "user", "content": "go"}])

    assert agent.calls == []
    assert agent.context_overflows == (1 if error is ContextWindowExceededError else 0)
    if error is HostedRequestError:
        assert isinstance(HostedRequestError("x"), ValueError)


def test_a_judge_call_is_refused_while_judges_are_off_and_sends_nothing(provider):
    gateway = make_gateway(provider)
    agent = open_agent(gateway)
    agent.judges = False

    with pytest.raises(JudgesOffError):
        forward(gateway, agent, "openai_chat", CHAT_BODY, kind="judge")
    with pytest.raises(JudgesOffError):
        asyncio.run(gateway.chat(agent, ChatRequest.from_params(CHAT_BODY["messages"], {}), kind="judge"))

    assert provider.requests == [] and agent.misc["judges_refused"]["count"] == 2


def test_a_capped_policy_only_agent_s_aux_and_judge_calls_are_refused_and_send_nothing(provider):
    gateway = make_gateway(provider)
    agent = open_agent(gateway)
    agent.policy_only = True  # a harness in a task sandbox, where the policy's code can read the agent's key
    agent.call_cap = 5  # under a gateway step budget, which counts only its policy calls
    agent.judges = False

    for kind in ("aux", "judge"):
        with pytest.raises(CallKindRefusedError, match=f"its {kind} calls are refused"):
            forward(gateway, agent, "openai_chat", CHAT_BODY, kind=kind)
        with pytest.raises(CallKindRefusedError):
            asyncio.run(gateway.chat(agent, ChatRequest.from_params(CHAT_BODY["messages"], {}), kind=kind))

    assert provider.requests == [] and agent.calls == []
    assert "judges_refused" not in agent.misc  # a kind the agent may never make is no judge the switch refused


def test_a_policy_only_agent_without_a_cap_forwards_its_aux_calls_and_none_of_its_judge_calls(provider):
    gateway = make_gateway(provider)
    agent = open_agent(gateway)
    agent.policy_only = True  # no call cap: its harness keeps its own step limit
    agent.judges = False

    with pytest.raises(CallKindRefusedError, match="its judge calls are refused"):
        forward(gateway, agent, "openai_chat", CHAT_BODY, kind="judge")
    relay = forward(gateway, agent, "openai_chat", CHAT_BODY, kind="aux")

    assert relay.status == 200 and len(provider.calls(CHAT)) == 1
    (call,) = agent.calls
    assert agent.call_kind(call.call_id) == "aux" and agent.policy_calls == 0
    assert "judges_refused" not in agent.misc


def test_a_hosted_policy_call_counts_against_the_call_cap_while_it_is_in_flight(provider):
    provider.delay = 0.3
    gateway = make_gateway(provider)
    agent = open_agent(gateway)
    agent.call_cap = 1

    async def go():
        try:
            call = asyncio.ensure_future(gateway.forward(agent, "openai_chat", CHAT, "", CHAT_BODY, {}))
            while provider.in_flight == 0:
                await asyncio.sleep(0.01)
            during = (agent.call_tally.in_flight, agent.policy_calls_in_flight, agent.over_call_cap())
            aux = await gateway.forward(agent, "openai_chat", CHAT, "", CHAT_BODY, {}, "aux")
            return during, aux, await call
        finally:
            await gateway.aclose()

    during, aux, relay = asyncio.run(go())

    assert during == (1, 1, True)  # a second policy call would get the budget stop now
    assert aux.status == relay.status == 200  # an aux call never counts against the cap
    assert (agent.call_tally.recorded, agent.call_tally.in_flight, agent.policy_calls_in_flight) == (1, 0, 0)
    assert agent.over_call_cap() and agent.policy_calls == 1


@pytest.mark.parametrize(
    ("fault", "error", "gateway_errors", "context_overflows"),
    [
        (Fault(503, openai_error("overloaded", kind="server_error")), HostedCallError, 1, 0),
        (Fault(401, openai_error("no", code="invalid_api_key")), HostedRefusedError, 1, 0),
        (Fault(400, openai_error("bad field")), HostedRequestError, 0, 0),
        (
            Fault(400, openai_error("maximum context length is 10 tokens. However, you requested 20 tokens")),
            ContextWindowExceededError,
            0,
            1,
        ),
    ],
    ids=["server", "refused", "invalid", "context-length"],
)
def test_a_call_the_backend_failed_counts_as_a_gateway_error(provider, fault, error, gateway_errors, context_overflows):
    """on_harness_exit's grade_if_gateway_clean grades a crashed harness only when the gateway failed none of its
    calls: a hosted call that failed after its retries, or was refused, is such a failure; a request the provider
    refused as invalid, or as too long, is the harness's."""

    provider.faults = [fault] * 4  # enough for two calls of two attempts each
    gateway = make_gateway(provider, max_retries=1)
    agent = open_agent(gateway)

    forward(gateway, agent, "openai_chat", CHAT_BODY)
    with pytest.raises(error):
        chat_in_process(gateway, agent, [{"role": "user", "content": "go"}])

    assert (agent.gateway_errors, agent.context_overflows) == (2 * gateway_errors, 2 * context_overflows)


@pytest.mark.parametrize(
    ("dialect", "path", "body", "budget"),
    [
        ("openai_chat", CHAT, {**CHAT_BODY, "max_tokens": 50}, "max_completion_tokens"),
        ("openai_chat", CHAT, {**CHAT_BODY, "max_completion_tokens": 50}, "max_completion_tokens"),
        ("anthropic_messages", MESSAGES, MESSAGES_BODY, "max_tokens"),
        ("openai_responses", RESPONSES, {**RESPONSES_BODY, "max_output_tokens": 50}, "max_output_tokens"),
    ],
)
def test_sampling_precedence_config_decides_a_hosted_policy_call_s_sampling(provider, dialect, path, body, budget):
    """With sampling_precedence: config, a policy call samples as the rollout says, never as its harness asks: the
    rollout's temperature and completion budget take the place of the call's own, and its top_p and top_k go (the
    rollout sets neither, so the server's defaults stand).  An aux call keeps its own, but one of an agent whose key
    code the policy runs can read (``policy_only``) is sent as its policy calls are."""

    gateway = make_gateway(provider)
    defaults = SamplingOverrides(max_tokens=300, temperature=0.3, precedence="config")
    agent = open_agent(gateway, sampling_defaults=defaults)
    own = {"temperature": 0.9, "top_p": 0.5, "top_k": 40}

    forward(gateway, agent, dialect, {**body, **own})
    forward(gateway, agent, dialect, {**body, **own}, kind="aux")
    agent.policy_only = True  # a harness in a task sandbox, without a call cap
    forward(gateway, agent, dialect, {**body, **own}, kind="aux")

    policy, aux, sandboxed_aux = (call["body"] for call in provider.calls(path))
    assert sandboxed_aux == policy
    budgets = {"max_tokens", "max_completion_tokens", "max_output_tokens"}
    assert (policy["temperature"], policy[budget]) == (0.3, 300)
    assert not ({"top_p", "top_k"} | (budgets - {budget})) & set(policy)
    assert {key: aux[key] for key in own} == own
    assert {key: value for key, value in aux.items() if key in budgets} == {
        key: value for key, value in body.items() if key in budgets
    }


# --------------------------------------------------------------------------- count_tokens


def test_count_tokens_is_forwarded_normalized_and_never_recorded(provider):
    gateway = make_gateway(provider)
    agent = open_agent(gateway, sampling_defaults=SamplingOverrides(max_tokens=300, temperature=0.3))
    body = {
        "model": "anything",
        "messages": [{"role": "user", "content": "go"}],
        "tools": [
            {"type": "web_search_20250305", "name": "web_search"},
            {"type": "mcp_toolset", "mcp_server_name": "x"},
        ],
        "tool_choice": {"type": "auto"},
    }

    async def go():
        try:
            return await gateway.count_tokens(agent, body, "beta=true", {"anthropic-version": "2023-06-01"})
        finally:
            await gateway.aclose()

    relay = asyncio.run(go())

    assert relay.status == 200 and relay.json()["input_tokens"] > 1000
    (sent,) = provider.calls(COUNT_TOKENS)
    assert sent["query"] == "beta=true" and sent["headers"]["anthropic-version"] == "2023-06-01"
    assert sent["body"] == {
        "model": MODEL,
        "messages": [{"role": "user", "content": "go"}],
        "tools": [{"name": "web_search", "input_schema": {"type": "object", "properties": {}}}],
    }
    assert agent.calls == [] and agent.policy_calls == 0
    provider.faults = [Fault(404, anthropic_error("Not found", kind="not_found_error"))]
    assert asyncio.run(go()).status == 404 and failures(agent) == {}  # never a failed call


# --------------------------------------------------------------------------- the probe


def test_the_probe_sends_one_minimal_request_in_each_dialect(provider):
    gateway = make_gateway(provider)

    async def go(dialects):
        try:
            return await gateway.probe(dialects)
        finally:
            await gateway.aclose()

    served = asyncio.run(go(["openai_chat", "anthropic_messages", "openai_responses"]))

    assert served == {dialect: {"served_model": SERVED_MODEL} for dialect in served} and len(served) == 3
    chat, messages, responses = (request["body"] for request in provider.requests)
    assert chat["max_completion_tokens"] == 16 and messages["max_tokens"] == 16
    assert responses["max_output_tokens"] == 16 and responses["store"] is False
    assert provider.requests[1]["headers"]["anthropic-version"] == "2023-06-01"


@pytest.mark.parametrize(
    ("setup", "status", "fragment"),
    [
        (lambda provider: setattr(provider, "models", ("another",)), 404, "does not exist"),
        (lambda provider: setattr(provider, "api_key", "sk-another-key-0000"), 401, "Incorrect API key"),
    ],
)
def test_a_probe_that_fails_names_the_dialect_and_the_reply(provider, setup, status, fragment):
    setup(provider)
    gateway = make_gateway(provider)

    with pytest.raises(ProbeError) as raised:
        asyncio.run(gateway.probe(["openai_chat"]))

    assert raised.value.dialect == "openai_chat" and raised.value.status == status
    assert raised.value.detail.startswith(f"{status} ") and fragment in raised.value.detail
    assert len(provider.requests) == 1  # a refusal is not retried


def test_a_probe_that_reaches_nothing_says_so():
    config = HostedConfig(base_url="http://127.0.0.1:9/v1", api_key_env=KEY_ENV, max_retries=0)
    with pytest.raises(ProbeError) as raised:
        asyncio.run(HostedGateway(config, MODEL).probe(["anthropic_messages"]))
    assert raised.value.status is None and "ClientConnectorError" in raised.value.detail


# --------------------------------------------------------------------------- proxies


def test_the_proxy_variables_are_read_and_nothing_else(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://user:p%40ss@proxy.example:3128")
    monkeypatch.setenv("NO_PROXY", "localhost,.internal.example")
    proxy, headers = env_proxy("https://api.example.com/v1/chat/completions")
    assert proxy == "http://proxy.example:3128" and headers is not None
    assert headers["Proxy-Authorization"].startswith("Basic ")
    assert env_proxy("https://models.internal.example/v1") == (None, None)
    assert env_proxy("http://api.example.com/v1") == (None, None)  # no HTTP_PROXY
    for no_proxy in ("*", "api.example.com:443", "[::1],.example.com"):
        monkeypatch.setenv("NO_PROXY", no_proxy)
        assert env_proxy("https://api.example.com/v1") == (None, None), no_proxy
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.example:3128")
    assert env_proxy("https://api.example.com/v1") == (None, None)  # aiohttp speaks to http proxies only


def test_a_call_goes_through_the_proxy_unless_no_proxy_covers_the_host(provider, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")  # nothing listens there
    gateway = make_gateway(provider, max_retries=0)
    agent = open_agent(gateway)
    with pytest.raises(HostedCallError):
        forward(gateway, agent, "openai_chat", CHAT_BODY)
    assert provider.requests == []

    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    assert forward(gateway, agent, "openai_chat", CHAT_BODY).status == 200


# --------------------------------------------------------------------------- over HTTP: path form and query


def test_the_path_form_and_the_query_string_are_forwarded(provider):
    gateway = make_gateway(provider)

    async def go():
        server = GatewayServer(gateway)
        base_url = await server.start(host="127.0.0.1")
        agent = gateway.open_agent(rollout_id="r", trajectory_id="t", interactions={})
        try:
            async with ClientSession() as session:
                url = f"{base_url}/s/{agent.api_key}/v1/messages?beta=true"
                async with session.post(url, json=MESSAGES_BODY, headers={"anthropic-version": "2023-06-01"}) as reply:
                    return reply.status, reply.headers.get("x-platoon-call-id"), agent
        finally:
            await server.stop()
            await gateway.aclose()

    status, call_id, agent = asyncio.run(go())

    assert status == 200 and call_id == agent.calls[0].call_id
    (sent,) = provider.calls(MESSAGES)
    assert sent["path"] == "/v1/messages" and sent["query"] == "beta=true"
    assert agent.api_key not in json.dumps(sent)
