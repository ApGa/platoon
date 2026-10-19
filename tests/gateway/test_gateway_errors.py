"""The per-agent count of calls the gateway failed for its own reasons (``AgentSession.gateway_errors``), in process
and over HTTP: what ``on_harness_exit: {crash: grade_if_gateway_clean}`` reads to keep infrastructure failures out of
training."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp
import openai
import pytest
from gateway_fakes import Completion

from platoon.gateway.core import ChatRequest
from platoon.gateway.server import GatewayServer

USER: Any = {"role": "user", "content": "What's the weather in Paris?"}


def open_root(gateway, **kwargs):
    return gateway.open_agent(rollout_id="rollout-1", trajectory_id="root", interactions={}, **kwargs)


def chat(gateway, agent, *, kind: str = "policy", **params):
    return asyncio.run(gateway.chat(agent, ChatRequest.from_params([USER], params), kind=kind))


def test_sampling_failures_and_timeouts_count_as_the_agents_gateway_errors(make_gateway, sampler):
    gateway = make_gateway(sample_timeout_seconds=0.05)
    root = open_root(gateway)
    child = gateway.open_subagent(root, role="helper")

    async def broken(**kwargs):
        raise RuntimeError("backend lost")

    original = sampler.sample_async
    sampler.sample_async = broken
    with pytest.raises(RuntimeError, match="backend lost"):
        chat(gateway, root)
    sampler.sample_async = original
    sampler.script = [Completion("too late", delay=1.0)]
    with pytest.raises(TimeoutError):
        chat(gateway, root)

    assert (root.gateway_errors, child.gateway_errors) == (2, 0)  # counted per agent
    assert root.calls == []


def test_a_bad_request_or_an_overflowing_prompt_is_not_a_gateway_error(make_gateway, sampler):
    gateway = make_gateway(context_window_length=32)
    root = open_root(gateway)

    with pytest.raises(ValueError, match="temperature"):
        chat(gateway, root, temperature=9)  # the request's fault
    with pytest.raises(Exception, match="context"):
        asyncio.run(gateway.chat(root, ChatRequest.from_params([{"role": "user", "content": "word " * 100}], {})))

    assert (root.gateway_errors, root.context_overflows) == (0, 1)
    assert sampler.calls == []


@pytest.mark.asyncio
async def test_http_server_errors_count_and_client_errors_do_not(gateway, sampler):
    server = GatewayServer(gateway, keepalive_seconds=0.05)
    await server.start(host="127.0.0.1")
    root = open_root(gateway)

    async def broken(**kwargs):
        raise RuntimeError("backend lost")

    sampler.sample_async = broken
    try:
        async with openai.AsyncOpenAI(base_url=f"{server.base_url}/v1", api_key=root.api_key, max_retries=0) as api:
            with pytest.raises(openai.InternalServerError):
                await api.chat.completions.create(model="p", messages=[USER])
            stream = await api.chat.completions.create(model="p", messages=[USER], stream=True)
            with pytest.raises(openai.APIError):
                async for _ in stream:
                    pass
        async with aiohttp.ClientSession() as http:
            async with http.post(
                f"{server.base_url}/v1/chat/completions",
                data=json.dumps({"model": "p", "messages": [USER], "temperature": 9}),
                headers={"Authorization": f"Bearer {root.api_key}", "Content-Type": "application/json"},
            ) as response:
                assert response.status == 400
    finally:
        await server.stop()
    assert root.gateway_errors == 2  # the non-streamed and the streamed 5xx; the 400 was the request's fault


@pytest.mark.asyncio
async def test_http_server_errors_while_preparing_a_call_count_and_client_errors_do_not(gateway, sampler, monkeypatch):
    server = GatewayServer(gateway, keepalive_seconds=0.05)
    await server.start(host="127.0.0.1")
    root = open_root(gateway)
    headers = {"Authorization": f"Bearer {root.api_key}", "Content-Type": "application/json"}

    def broken_render(messages, request=None):
        raise RuntimeError("the chat template failed")

    try:
        async with aiohttp.ClientSession() as http:

            async def status_of(body: dict) -> int:
                async with http.post(
                    f"{server.base_url}/v1/chat/completions", data=json.dumps(body), headers=headers
                ) as response:
                    return response.status

            # The request's fault, found while preparing the call: a 400, never the gateway's.
            assert await status_of({"model": "p", "messages": [USER], "temperature": 9}) == 400
            assert await status_of({"model": "p", "messages": [USER], "tool_choice": "required"}) == 400
            monkeypatch.setattr(gateway, "render", broken_render)
            assert await status_of({"model": "p", "messages": [USER]}) == 500
            assert await status_of({"model": "p", "messages": [USER], "stream": True}) == 500  # before any stream
    finally:
        await server.stop()
    assert (root.gateway_errors, root.context_overflows) == (2, 0)
    assert sampler.calls == [] and root.calls == []
