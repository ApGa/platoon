"""The gateway step budget and agent limits: an agent with a call cap gets a final message instead of a sample past
it (nothing recorded, so nothing trains), a harness that ignores the stop is refused, a rollout mints a bounded
number of agents over HTTP, whose policy calls all count against the minting agent's cap, and an agent limited to
policy calls (a harness in a task sandbox) is refused the aux and judge calls the cap does not count."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp
import openai
import pytest
import pytest_asyncio
from gateway_fakes import WEATHER_CALL, Completion

from platoon.gateway.core import (
    BUDGET_MISC_KEY,
    DEFAULT_MAX_MINTED_AGENTS,
    MAX_BUDGET_STOPS,
    MAX_MINTED_AGENTS_ENV,
    AgentLimitError,
    CallKindRefusedError,
    ChatRequest,
    ModelGateway,
)
from platoon.gateway.server import CALL_ID_HEADER, GatewayServer
from platoon.harness.agent import CALL_KIND_HEADER

USER: Any = {"role": "user", "content": "What's the weather in Paris?"}


@pytest_asyncio.fixture
async def server(gateway):
    server = GatewayServer(gateway, keepalive_seconds=0.05)
    await server.start(host="127.0.0.1")
    yield server
    await server.stop()


def open_root(gateway: ModelGateway, rollout_id: str = "rollout-1", *, call_cap: int | None = None):
    agent = gateway.open_agent(rollout_id=rollout_id, trajectory_id=f"{rollout_id}-root", interactions={})
    agent.call_cap = call_cap
    return agent


def client(base_url: str, api_key: str) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0)


async def post(url: str, payload: dict, api_key: str) -> tuple[int, dict]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=json.dumps(payload), headers=headers) as response:
            return response.status, json.loads(await response.text())


@pytest.mark.asyncio
async def test_calls_past_the_cap_get_a_final_message_that_is_never_recorded(gateway, sampler, server):
    sampler.script = [Completion(WEATHER_CALL), Completion(WEATHER_CALL)]
    agent = open_root(gateway, call_cap=2)
    async with client(f"{server.base_url}/v1", agent.api_key) as openai_client:
        sampled = [await openai_client.chat.completions.create(model="policy", messages=[USER]) for _ in range(2)]
        raw = await openai_client.chat.completions.with_raw_response.create(model="policy", messages=[USER])
    stop = raw.parse()

    assert [completion.id for completion in sampled] == agent.call_ids and len(sampler.calls) == 2
    choice = stop.choices[0]
    assert (choice.message.role, choice.finish_reason, choice.message.tool_calls) == ("assistant", "stop", None)
    assert "step budget of 2 model calls is used up" in choice.message.content
    assert raw.headers[CALL_ID_HEADER] == stop.id and stop.id not in agent.interactions
    assert (agent.policy_calls, agent.budget_stops, agent.budget_exhausted) == (2, 1, True)
    assert agent.misc[BUDGET_MISC_KEY] == {"enforce": "gateway", "max_steps": 2, "stops": 1}


@pytest.mark.asyncio
async def test_the_stop_streams_like_a_sample(gateway, sampler, server):
    agent = open_root(gateway, call_cap=0)
    async with client(f"{server.base_url}/v1", agent.api_key) as openai_client:
        stream = await openai_client.chat.completions.create(model="policy", messages=[USER], stream=True)
        chunks = [chunk async for chunk in stream]

    text = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
    assert "step budget of 0 model calls is used up" in text
    assert chunks[-1].choices[0].finish_reason == "stop" and sampler.calls == [] and agent.calls == []


@pytest.mark.asyncio
async def test_aux_and_judge_calls_are_outside_the_budget(gateway, sampler, server):
    sampler.script = [Completion("summary"), Completion("verdict")]
    agent = open_root(gateway, call_cap=0)
    async with client(f"{server.base_url}/v1", agent.api_key) as openai_client:
        for kind in ("aux", "judge"):
            completion = await openai_client.chat.completions.create(
                model="policy", messages=[USER], extra_headers={"X-Platoon-Call-Kind": kind}
            )
            assert completion.id in agent.call_ids

    assert (agent.policy_calls, agent.budget_stops) == (0, 0)


@pytest.mark.asyncio
async def test_calls_in_flight_count_against_the_cap(gateway, sampler, server):
    sampler.script = [Completion("slow", delay=0.3)]
    agent = open_root(gateway, call_cap=1)
    async with client(f"{server.base_url}/v1", agent.api_key) as openai_client:
        first = asyncio.ensure_future(openai_client.chat.completions.create(model="policy", messages=[USER]))
        while agent.policy_calls_in_flight == 0:
            await asyncio.sleep(0.01)
        second = await openai_client.chat.completions.create(model="policy", messages=[USER])
        sampled = await first

    assert "step budget" in second.choices[0].message.content and sampled.choices[0].message.content == "slow"
    assert (agent.policy_calls, agent.budget_stops, agent.policy_calls_in_flight) == (1, 1, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
@pytest.mark.parametrize("minted", [False, True], ids=["one-agent", "minted-agents"])
async def test_calls_that_arrive_together_never_pass_the_cap(gateway, sampler, server, stream, minted):
    """Six policy calls at once against a cap of one, from one agent or from six agents it minted: one samples and
    the others get the stop.  A streamed call that counted only once the event loop started it let every call
    checked before then through."""

    sampler.script = [Completion("slow", delay=0.3) for _ in range(6)]
    root = open_root(gateway, call_cap=1)
    url = f"{server.base_url}/v1"
    keys = [root.api_key] * 6
    if minted:
        keys = [(await post(f"{url}/platoon/agents", {"role": "helper"}, root.api_key))[1]["api_key"] for _ in keys]
    body = {"model": "policy", "messages": [USER], "stream": stream}

    async with aiohttp.ClientSession() as session:

        async def call(key: str) -> str:
            headers = {"Authorization": f"Bearer {key}"}
            async with session.post(f"{url}/chat/completions", json=body, headers=headers) as reply:
                assert reply.status == 200
                return await reply.text()

        replies = await asyncio.gather(*(call(key) for key in keys))

    assert len(sampler.calls) == 1 and sum("step budget of 1 model calls is used up" in reply for reply in replies) == 5
    assert (root.call_tally.recorded, root.call_tally.in_flight) == (1, 0)


@pytest.mark.asyncio
async def test_a_harness_that_ignores_the_stop_is_refused(gateway, server):
    agent = open_root(gateway, call_cap=0)
    async with client(f"{server.base_url}/v1", agent.api_key) as openai_client:
        for _ in range(MAX_BUDGET_STOPS):
            await openai_client.chat.completions.create(model="policy", messages=[USER])
        with pytest.raises(openai.BadRequestError, match="was told to stop 8 times"):
            await openai_client.chat.completions.create(model="policy", messages=[USER])

    assert agent.budget_stops == MAX_BUDGET_STOPS and agent.calls == []


def test_an_agent_without_a_cap_is_never_stopped(gateway):
    agent = open_root(gateway)
    agent.policy_calls = 10_000
    assert not agent.over_call_cap() and not agent.budget_exhausted


@pytest.mark.asyncio
async def test_minted_children_inherit_the_cap_and_a_rollout_mints_a_bounded_number(make_gateway, sampler):
    gateway = make_gateway(max_minted_agents=2)
    server = GatewayServer(gateway, keepalive_seconds=0.05)
    await server.start(host="127.0.0.1")
    try:
        root = open_root(gateway, call_cap=3)
        url = f"{server.base_url}/v1/platoon/agents"
        statuses = [await post(url, {"role": "helper"}, root.api_key) for _ in range(3)]
        other = open_root(gateway, "rollout-2")  # another rollout has its own allowance
        other_status, _ = await post(url, {"role": "helper"}, other.api_key)
    finally:
        await server.stop()

    assert [status for status, _ in statuses] == [201, 201, 400] and other_status == 201
    assert "minted its limit of 2 agents" in statuses[2][1]["error"]["message"]
    children = [gateway.authenticate(body["api_key"]) for status, body in statuses if status == 201]
    assert [child.call_cap for child in children] == [3, 3]
    gateway.close_rollout("rollout-1")
    assert "rollout-1" not in gateway._minted


@pytest.mark.asyncio
async def test_the_cap_bounds_the_policy_calls_of_every_agent_minted_under_the_capped_one(gateway, sampler, server):
    sampler.script = [Completion("root"), Completion("child"), Completion("grandchild")]
    root = open_root(gateway, call_cap=3)
    url = f"{server.base_url}/v1/platoon/agents"
    _, child = await post(url, {"role": "helper"}, root.api_key)
    _, grandchild = await post(url, {"role": "helper"}, child["api_key"])
    keys = [root.api_key, child["api_key"], grandchild["api_key"]]

    replies = []
    for _ in range(2):  # each agent calls twice: three calls fit the cap of the whole tree, three are stopped
        for key in keys:
            async with client(f"{server.base_url}/v1", key) as openai_client:
                completion = await openai_client.chat.completions.create(model="policy", messages=[USER])
                replies.append(completion.choices[0].message.content)

    assert replies[:3] == ["root", "child", "grandchild"] and len(sampler.calls) == 3
    assert all("step budget of 3 model calls is used up" in reply for reply in replies[3:])
    agents = [gateway.authenticate(key) for key in keys]
    assert [agent.policy_calls for agent in agents] == [1, 1, 1]  # each agent's own count stays its own
    assert [agent.budget_stops for agent in agents] == [1, 1, 1]
    assert root.call_tally.recorded == 3 and all(agent.call_tally is root.call_tally for agent in agents)


@pytest.mark.asyncio
async def test_a_minted_agents_call_in_flight_counts_against_its_parents_cap(gateway, sampler, server):
    sampler.script = [Completion("slow", delay=0.3)]
    root = open_root(gateway, call_cap=1)
    _, minted = await post(f"{server.base_url}/v1/platoon/agents", {"role": "helper"}, root.api_key)
    child = gateway.authenticate(minted["api_key"])
    async with (
        client(f"{server.base_url}/v1", minted["api_key"]) as child_client,
        client(f"{server.base_url}/v1", root.api_key) as root_client,
    ):
        first = asyncio.ensure_future(child_client.chat.completions.create(model="policy", messages=[USER]))
        while child.policy_calls_in_flight == 0:
            await asyncio.sleep(0.01)
        second = await root_client.chat.completions.create(model="policy", messages=[USER])
        sampled = await first

    assert "step budget" in second.choices[0].message.content and sampled.choices[0].message.content == "slow"
    assert (root.policy_calls, root.budget_stops, child.policy_calls, child.budget_stops) == (0, 1, 1, 0)
    assert (root.call_tally.recorded, root.call_tally.in_flight) == (1, 0)


@pytest.mark.asyncio
async def test_a_policy_only_agent_and_the_agents_it_mints_are_refused_aux_and_judge_calls(gateway, sampler, server):
    sampler.script = [Completion("policy")]
    root = open_root(gateway, call_cap=5)
    root.policy_only = True
    _, minted = await post(f"{server.base_url}/v1/platoon/agents", {"role": "helper"}, root.api_key)

    for key in (root.api_key, minted["api_key"]):
        async with client(f"{server.base_url}/v1", key) as openai_client:
            for kind in ("aux", "judge"):
                with pytest.raises(openai.PermissionDeniedError, match=f"only policy calls.*its {kind} calls"):
                    await openai_client.chat.completions.create(
                        model="policy", messages=[USER], extra_headers={CALL_KIND_HEADER: kind}
                    )
    async with client(f"{server.base_url}/v1", root.api_key) as openai_client:
        completion = await openai_client.chat.completions.create(model="policy", messages=[USER])

    assert len(sampler.calls) == 1 and root.call_ids == [completion.id] and root.call_kinds == {completion.id: "policy"}
    child = gateway.authenticate(minted["api_key"])
    assert child.policy_only and child.calls == [] and (root.gateway_errors, child.gateway_errors) == (0, 0)
    with pytest.raises(CallKindRefusedError):  # in process too
        await gateway.chat(child, ChatRequest.from_params([USER], {}), kind="aux")
    assert len(sampler.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/v1/chat/completions", {"model": "policy", "messages": [USER], "stream": True}),
        ("/v1/responses", {"model": "policy", "input": "Summarize.", "stream": True}),
        ("/v1/messages", {"model": "policy", "max_tokens": 8, "messages": [USER], "stream": True}),
    ],
    ids=["chat", "responses", "messages"],
)
async def test_the_refusal_is_the_dialects_permission_error_before_any_stream(gateway, sampler, server, path, body):
    agent = open_root(gateway)
    agent.policy_only = True

    headers = {"x-api-key": agent.api_key, "Content-Type": "application/json", CALL_KIND_HEADER: "judge"}
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{server.base_url}{path}", data=json.dumps(body), headers=headers) as response:
            status, content_type, payload = response.status, response.content_type, await response.json()

    assert (status, content_type) == (403, "application/json")
    assert payload["error"]["type"] == "permission_error" and "only policy calls" in payload["error"]["message"]
    if path == "/v1/messages":
        assert payload["type"] == "error"  # Anthropic's error envelope
    assert sampler.calls == [] and agent.calls == []


@pytest.mark.asyncio
async def test_a_policy_only_agent_s_judge_call_is_refused_for_its_kind_not_by_the_judges_switch(gateway, sampler):
    """A kind of call the agent may never make is refused as such, before the judges switch: it is not counted as a
    judge refused while judges are off, which would stop an evaluation (inference.judges: false)."""

    agent = open_root(gateway)
    agent.policy_only, agent.judges = True, False

    with pytest.raises(CallKindRefusedError, match="its judge calls are refused"):
        await gateway.chat(agent, ChatRequest.from_params([USER], {}), kind="judge")

    assert sampler.calls == [] and agent.calls == [] and "judges_refused" not in agent.misc


def test_an_agent_not_limited_to_policy_calls_keeps_aux_and_judge(gateway):
    agent = open_root(gateway)
    for kind in ("policy", "aux", "judge"):
        agent.check_kind(kind)
    agent.policy_only = True
    agent.check_kind("policy")
    with pytest.raises(CallKindRefusedError, match="its aux calls are refused"):
        agent.check_kind("aux")


def test_the_mint_limit_defaults_from_the_environment(make_gateway, monkeypatch):
    assert make_gateway().max_minted_agents == DEFAULT_MAX_MINTED_AGENTS
    monkeypatch.setenv(MAX_MINTED_AGENTS_ENV, "5")
    assert make_gateway().max_minted_agents == 5
    monkeypatch.setenv(MAX_MINTED_AGENTS_ENV, "-1")
    with pytest.raises(ValueError, match="non-negative"):
        make_gateway()
    gateway = make_gateway(max_minted_agents=0)
    with pytest.raises(AgentLimitError):
        gateway.mint_subagent(open_root(gateway), role="helper")
