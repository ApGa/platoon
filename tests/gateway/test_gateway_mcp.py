"""The MCP tool plane over real HTTP, driven by the official ``mcp`` client and by raw JSON-RPC posts."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import aiohttp
import httpx
import pytest
import pytest_asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client, streamablehttp_client
from mcp.shared.exceptions import McpError

from platoon.envs.service import ToolResult, ToolSpec
from platoon.gateway.core import AgentClosedError, ChatRequest, GatewayShutDownError, ModelGateway, UnknownAgentError
from platoon.gateway.mcp import (
    AGENT_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    LATEST_PROTOCOL_VERSION,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    UnknownToolError,
    bind_agent_tools,
    call_agent_tool,
    list_agent_tools,
    mcp_url,
    to_call_tool_result,
    tool_call_records,
)
from platoon.gateway.server import GatewayServer
from platoon.task import Task

ROUTING_KEY = "openhands.dev/tool-routing"


def _client_venv_python() -> Path | None:
    """A Python with the fastmcp and OpenHands SDK clients: ``$PLATOON_CLIENT_VENV/bin/python``, else this
    interpreter when it has both, else None."""

    if venv := os.environ.get("PLATOON_CLIENT_VENV"):
        return Path(venv) / "bin" / "python"
    try:
        found = all(importlib.util.find_spec(name) is not None for name in ("fastmcp", "openhands.sdk"))
    except ImportError:  # find_spec imports a dotted name's parent package
        found = False
    return Path(sys.executable) if found else None


CLIENT_VENV_PYTHON = _client_venv_python()
NO_CLIENT_VENV = (
    "needs the fastmcp and OpenHands SDK clients: run in an environment that has them, or set PLATOON_CLIENT_VENV "
    "to one"
)


class StubRenderer:
    """The tool plane renders nothing; the gateway only inspects a renderer's signatures."""

    def build_generation_prompt(self, messages, **kwargs):
        raise AssertionError("the MCP tests make no model calls")

    def parse_response(self, tokens, **kwargs):
        raise AssertionError("the MCP tests make no model calls")

    def get_stop_sequences(self):
        return []


@dataclass
class HintedSpec(ToolSpec):
    """``ToolSpec`` with the env hints that ORS sessions add."""

    terminal: bool = False
    read_only: bool = False
    meta: dict[str, Any] | None = None


@dataclass
class MetaResult(ToolResult):
    metadata: dict[str, Any] = field(default_factory=dict)


def schema(**properties: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {name: {"type": kind} for name, kind in properties.items()},
        "required": list(properties),
    }


@dataclass
class ToySession:
    """An env session with a read-only lookup, a plain note tool, a terminal submit and a tool that raises."""

    label: str = "root"
    task: Task = field(default_factory=lambda: Task(id="toy-0", goal="Find the code and submit it."))
    instruction: str = "Find the code and submit it."
    sandbox: None = None
    delay: float = 0.0
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    async def tools(self) -> list[ToolSpec]:
        return [
            HintedSpec("lookup", "Look up a key.", schema(key="string"), read_only=True),
            HintedSpec("note", "Write a note.", {**schema(text="string"), "x-platoon-reward-kind": "level"}),
            HintedSpec("submit", "Submit the answer; ends the episode.", schema(answer="string"), terminal=True),
            HintedSpec("crash", "Always fails.", schema()),
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append((name, arguments))
        if self.delay:
            await asyncio.sleep(self.delay)
        if name == "crash":
            raise RuntimeError("the env tool blew up")
        if name == "submit":
            correct = arguments.get("answer") == "1234"
            return MetaResult(
                content=[{"type": "text", "text": "correct" if correct else "wrong"}],
                finished=True,
                reward=1.0 if correct else 0.0,
                metadata={"graded_by": self.label},
            )
        return ToolResult.text(f"{self.label}:{name}:{json.dumps(arguments, sort_keys=True)}")


def make_gateway() -> ModelGateway:
    return ModelGateway(
        sampler=cast(Any, None), renderer=cast(Any, StubRenderer()), tokenizer=cast(Any, None), model_name="policy"
    )


@pytest.fixture
def gateway() -> ModelGateway:
    return make_gateway()


@pytest_asyncio.fixture
async def server(gateway):
    server = GatewayServer(gateway, keepalive_seconds=0.05)
    await server.start(host="127.0.0.1")
    yield server
    await server.stop()


def open_root(gateway: ModelGateway, rollout_id: str = "rollout-1"):
    return gateway.open_agent(rollout_id=rollout_id, trajectory_id=f"{rollout_id}-root", interactions={})


async def rpc(url: str, payload: Any, *, api_key: str | None = None, headers: dict[str, str] | None = None):
    """POST one JSON-RPC payload; returns (status, headers, parsed JSON or raw text)."""
    headers = {"Content-Type": "application/json", "Accept": "application/json", **(headers or {})}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    data = payload if isinstance(payload, str) else json.dumps(payload)
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data, headers=headers) as response:
            text = await response.text()
            try:
                body: Any = json.loads(text)
            except ValueError:
                body = text
            return response.status, response.headers, body


def request(method: str, params: dict[str, Any] | None = None, request_id: int = 1) -> dict[str, Any]:
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


# --------------------------------------------------------------------------- the official client


@pytest.mark.asyncio
async def test_official_client_initializes_lists_and_calls_tools(gateway, server):
    agent = open_root(gateway)
    session = ToySession()
    bind_agent_tools(gateway, agent, session)
    agent.calls.append(cast(Any, SimpleNamespace(call_id="call-7")))  # the model turn that asked for the tools

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        transport = streamablehttp_client(mcp_url(server.base_url, agent.secret))
    async with transport as (read, write, get_session_id):
        async with ClientSession(read, write) as client:
            init = await client.initialize()
            listed = (await client.list_tools()).tools
            lookup = await client.call_tool("lookup", {"key": "door"})
            submit = await client.call_tool("submit", {"answer": "1234"})
            await client.send_ping()
        assert get_session_id() is None  # stateless: the agent key is the session

    assert init.protocolVersion == LATEST_PROTOCOL_VERSION and init.serverInfo.name == "platoon-gateway"
    assert init.capabilities.tools is not None
    tools = {tool.name: tool for tool in listed}
    assert list(tools) == ["lookup", "note", "submit", "crash"]
    assert tools["lookup"].inputSchema == schema(key="string")
    assert tools["lookup"].annotations is not None and tools["lookup"].annotations.readOnlyHint is True
    assert "x-platoon-reward-kind" not in tools["note"].inputSchema  # platoon's own schema keys are stripped

    assert [block.text for block in lookup.content] == ['root:lookup:{"key": "door"}']
    assert lookup.isError is False
    assert lookup.meta is not None and lookup.meta["platoon"]["finished"] is False
    assert [block.text for block in submit.content] == ["correct"]
    signals = submit.meta["platoon"]
    assert signals["finished"] is True and signals["reward"] == 1.0 and signals["metadata"] == {"graded_by": "root"}

    assert session.calls == [("lookup", {"key": "door"}), ("submit", {"answer": "1234"})]
    records = tool_call_records(agent)
    assert json.loads(json.dumps(records)) == records
    assert [(r["name"], r["arguments"], r["text"]) for r in records] == [
        ("lookup", {"key": "door"}, 'root:lookup:{"key": "door"}'),
        ("submit", {"answer": "1234"}, "correct"),
    ]
    assert all(r["after_call_id"] == "call-7" and r["started_at"] <= r["ended_at"] for r in records)
    assert records[1]["finished"] is True and records[1]["reward"] == 1.0
    assert signals["event_id"] == records[1]["event_id"]


@pytest.mark.asyncio
async def test_bearer_form_with_the_current_client_api(gateway, server):
    agent = open_root(gateway)
    bind_agent_tools(gateway, agent, ToySession(), access="read_only")
    headers = {"Authorization": f"Bearer {agent.secret}"}

    async with httpx.AsyncClient(headers=headers) as http:
        async with streamable_http_client(mcp_url(server.base_url), http_client=http) as (read, write, _):
            async with ClientSession(read, write) as client:
                await client.initialize()
                listed = (await client.list_tools()).tools
                result = await client.call_tool("lookup", {"key": "k"})

    assert [tool.name for tool in listed] == ["lookup"]
    assert result.content[0].text == 'root:lookup:{"key": "k"}'


@pytest.mark.asyncio
async def test_unknown_tool_is_a_jsonrpc_error_and_a_raising_tool_is_is_error(gateway, server):
    agent = open_root(gateway)
    session = ToySession()
    bind_agent_tools(gateway, agent, session)

    async with streamable_http_client(mcp_url(server.base_url, agent.secret)) as (read, write, _):
        async with ClientSession(read, write) as client:
            await client.initialize()
            with pytest.raises(McpError, match="Unknown tool: teleport") as unknown:
                await client.call_tool("teleport", {})
            crashed = await client.call_tool("crash", {})

    assert unknown.value.error.code == INVALID_PARAMS
    assert crashed.isError is True
    assert crashed.content[0].text == "Tool 'crash' failed: RuntimeError: the env tool blew up"
    assert [call[0] for call in session.calls] == ["crash"]  # the unknown tool never reached the env
    (record,) = tool_call_records(agent)
    assert record["name"] == "crash" and record["is_error"] is True and record["after_call_id"] is None


# --------------------------------------------------------------------------- access, transforms and children


@pytest.mark.asyncio
async def test_access_levels_filter_listings_and_calls(gateway, server):
    expected = {
        "full": ["lookup", "note", "submit", "crash"],
        "no_terminal": ["lookup", "note", "crash"],
        "read_only": ["lookup"],
    }
    for access, names in expected.items():
        agent = open_root(gateway, rollout_id=f"rollout-{access}")
        session = ToySession()
        bind_agent_tools(gateway, agent, session, access=access)
        url = mcp_url(server.base_url, agent.secret)

        status, _, body = await rpc(url, request("tools/list"))
        assert status == 200 and [tool["name"] for tool in body["result"]["tools"]] == names
        status, _, body = await rpc(url, request("tools/call", {"name": "submit", "arguments": {"answer": "1"}}))
        if "submit" in names:
            assert body["result"]["content"][0]["text"] == "wrong"
        else:
            assert status == 200 and body["error"]["code"] == INVALID_PARAMS and session.calls == []


@pytest.mark.asyncio
async def test_transform_rewrites_listings_without_touching_the_session(gateway, server):
    def route_and_hide(tools: list[ToolSpec]) -> list[ToolSpec]:
        for tool in tools:
            cast(HintedSpec, tool).meta = {ROUTING_KEY: {"version": 1, "tool": tool.name}}
            tool.description = f"[routed] {tool.description}"
        return [tool for tool in tools if tool.name != "crash"]

    agent = open_root(gateway)
    session = ToySession()
    bind_agent_tools(gateway, agent, session, transform=route_and_hide)
    url = mcp_url(server.base_url, agent.secret)

    _, _, body = await rpc(url, request("tools/list"))
    tools = body["result"]["tools"]
    assert [tool["name"] for tool in tools] == ["lookup", "note", "submit"]
    assert tools[0]["_meta"] == {ROUTING_KEY: {"version": 1, "tool": "lookup"}}
    assert tools[0]["description"] == "[routed] Look up a key."
    _, _, body = await rpc(url, request("tools/call", {"name": "crash"}))
    assert body["error"]["code"] == INVALID_PARAMS  # a tool the transform drops cannot be called
    # The transform worked on copies: the session's listing is untouched.
    assert [getattr(tool, "meta", None) for tool in await session.tools()] == [None] * 4
    _, _, body = await rpc(url, request("tools/list"))
    assert body["result"]["tools"][0]["description"] == "[routed] Look up a key."


@pytest.mark.asyncio
async def test_children_inherit_the_parents_tools_until_rebound(gateway, server):
    def routed(tools: list[ToolSpec]) -> list[ToolSpec]:
        for tool in tools:
            cast(HintedSpec, tool).meta = {ROUTING_KEY: {"tool": tool.name}}
        return tools

    parent = open_root(gateway)
    parent_session = ToySession(label="parent")
    provider = bind_agent_tools(gateway, parent, parent_session, transform=routed)

    inherited = gateway.open_subagent(parent, role="subagent")
    assert inherited.tools is provider and inherited.tool_access == "full"
    _, _, body = await rpc(mcp_url(server.base_url, inherited.secret), request("tools/call", {"name": "note"}))
    assert body["result"]["content"][0]["text"] == "parent:note:{}"

    # Same session, narrower access: the provider (its cache and its call lock) stays shared.
    narrowed = gateway.open_subagent(parent, role="verifier")
    assert bind_agent_tools(gateway, narrowed, parent_session, access="read_only") is provider
    assert [tool.name for tool in await list_agent_tools(narrowed)] == ["lookup"]
    # open_subagent's access= narrows a child that keeps the parent's provider, without a rebind.
    verifier = gateway.open_subagent(parent, role="verifier", access="read_only")
    assert verifier.tools is provider and verifier.tool_access == "read_only"
    assert [tool.name for tool in await list_agent_tools(verifier)] == ["lookup"]

    # A forked session: a new provider that keeps the inherited transform.
    forked = gateway.open_subagent(parent, role="subagent")
    fork_session = ToySession(label="fork")
    bind_agent_tools(gateway, forked, fork_session, access="no_terminal")
    listed = await list_agent_tools(forked)
    assert [tool.name for tool in listed] == ["lookup", "note", "crash"]
    assert cast(HintedSpec, listed[0]).meta == {ROUTING_KEY: {"tool": "lookup"}}
    _, _, body = await rpc(mcp_url(server.base_url, forked.secret), request("tools/call", {"name": "note"}))
    assert body["result"]["content"][0]["text"] == "fork:note:{}"
    grandchild = gateway.open_subagent(forked, role="subagent")
    assert grandchild.tools is forked.tools and grandchild.tool_access == "no_terminal"

    assert parent.tool_access == "full" and parent.tools is provider
    assert [name for name, _ in parent_session.calls] == ["note"] and parent.tool_calls == []
    assert [r["name"] for r in tool_call_records(inherited)] == ["note"]
    assert [r["name"] for r in tool_call_records(forked)] == ["note"]
    assert fork_session.calls == [("note", {})]


@pytest.mark.asyncio
async def test_inline_calls_are_recorded_after_the_last_model_call(gateway):
    agent = open_root(gateway)
    assert await list_agent_tools(agent) == []
    with pytest.raises(UnknownToolError):
        await call_agent_tool(agent, "lookup", {"key": "k"})

    bind_agent_tools(gateway, agent, ToySession())
    first = await call_agent_tool(agent, "lookup", {"key": "a"})
    agent.calls.append(cast(Any, SimpleNamespace(call_id="call-1")))
    crashed = await call_agent_tool(agent, "crash", {})

    assert first.content == [{"type": "text", "text": 'root:lookup:{"key": "a"}'}]
    assert crashed.is_error
    assert [(r["name"], r["after_call_id"], r["is_error"]) for r in tool_call_records(agent)] == [
        ("lookup", None, False),
        ("crash", "call-1", True),
    ]
    gateway.close_agent(agent)
    with pytest.raises(AgentClosedError):
        await call_agent_tool(agent, "lookup", {"key": "a"})


def count_overlap(session: ToySession) -> dict[str, int]:
    """Wrap ``session.call_tool`` to count how many calls run at once (``peak``)."""
    counts = {"active": 0, "peak": 0}
    original = session.call_tool

    async def counting(name: str, arguments: dict[str, Any]) -> ToolResult:
        counts["active"] += 1
        counts["peak"] = max(counts["peak"], counts["active"])
        try:
            return await original(name, arguments)
        finally:
            counts["active"] -= 1

    session.call_tool = counting  # type: ignore[method-assign]
    return counts


@pytest.mark.asyncio
async def test_calls_into_an_unmanaged_session_are_serialized_per_provider(gateway):
    agent = open_root(gateway)
    session = ToySession(delay=0.05)
    bind_agent_tools(gateway, agent, session)
    child = gateway.open_subagent(agent, role="subagent")
    counts = count_overlap(session)

    await asyncio.gather(*(call_agent_tool(target, "note", {}) for target in (agent, child, agent, child)))

    assert counts["peak"] == 1 and len(session.calls) == 4


@pytest.mark.asyncio
async def test_a_session_that_orders_its_own_calls_gets_no_provider_lock(gateway):
    @dataclass
    class ManagedSession(ToySession):
        manages_concurrency: bool = True  # e.g. ORS and Harbor sessions: their live env's gate orders the calls

    agent = open_root(gateway)
    session = ManagedSession(delay=0.05)
    bind_agent_tools(gateway, agent, session)
    child = gateway.open_subagent(agent, role="subagent")  # shares the provider
    counts = count_overlap(session)

    await asyncio.gather(*(call_agent_tool(target, "note", {}) for target in (agent, child)))

    assert counts["peak"] == 2 and len(session.calls) == 2


@dataclass
class BlockingSession(ToySession):
    """A session whose ``wait`` tool runs until ``release`` is set, recording calls that start and get cancelled;
    it takes concurrent calls (no provider lock)."""

    manages_concurrency: bool = True
    release: asyncio.Event = field(default_factory=asyncio.Event)
    started: list[str] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name != "wait":
            return await super().call_tool(name, arguments)
        self.started.append(str(arguments.get("tag")))
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.append(str(arguments.get("tag")))
            raise
        return ToolResult.text(f"waited {arguments.get('tag')}")

    async def tools(self) -> list[ToolSpec]:
        return [*await super().tools(), HintedSpec("wait", "Wait for the test.", schema(tag="string"))]


SSE = {"Accept": "application/json, text/event-stream"}


def cancellation(request_id: Any) -> dict[str, Any]:
    params = {"requestId": request_id, "reason": "the client timed out"}
    return {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": params}


async def until(condition, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, "timed out waiting for the condition"
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_notifications_cancelled_cancels_the_call_with_that_request_id(gateway, server):
    agent = open_root(gateway)
    session = BlockingSession()
    bind_agent_tools(gateway, agent, session)
    url = mcp_url(server.base_url, agent.secret)

    # Over SSE: the stream ends without a response for the cancelled request.
    streamed = asyncio.ensure_future(
        rpc(url, request("tools/call", {"name": "wait", "arguments": {"tag": "sse"}}, 7), headers=SSE)
    )
    await until(lambda: session.started == ["sse"])
    status, _, body = await rpc(url, cancellation(7))
    assert status == 202 and body == ""
    status, headers, text = await asyncio.wait_for(streamed, 5)
    assert status == 200 and headers["Content-Type"].startswith("text/event-stream")
    assert "event: message" not in text

    # As JSON: the request is answered with 202 and no body.
    plain = asyncio.ensure_future(
        rpc(url, request("tools/call", {"name": "wait", "arguments": {"tag": "json"}}, "r-1"))
    )
    await until(lambda: session.started == ["sse", "json"])
    await rpc(url, cancellation("r-1"))
    status, _, body = await asyncio.wait_for(plain, 5)
    assert status == 202 and body == ""

    assert session.cancelled == ["sse", "json"]
    assert [(r["text"], r["is_error"]) for r in tool_call_records(agent)] == [("cancelled", True), ("cancelled", True)]


@pytest.mark.asyncio
async def test_a_cancellation_that_matches_no_single_call_of_the_agent_cancels_nothing(gateway, server):
    agent, other = open_root(gateway), open_root(gateway, rollout_id="rollout-2")
    session = BlockingSession()
    bind_agent_tools(gateway, agent, session)
    bind_agent_tools(gateway, other, ToySession())
    url = mcp_url(server.base_url, agent.secret)

    # Two clients of one agent may use the same request id at once (the endpoint has no MCP sessions).
    calls = [
        asyncio.ensure_future(rpc(url, request("tools/call", {"name": "wait", "arguments": {"tag": tag}}, 1)))
        for tag in ("a", "b")
    ]
    await until(lambda: len(session.started) == 2)
    await rpc(url, cancellation(1))  # ambiguous
    await rpc(mcp_url(server.base_url, other.secret), cancellation(1))  # another agent's request 1
    await rpc(url, cancellation("1"))  # a string id is another id
    await asyncio.sleep(0.05)
    session.release.set()
    replies = [body for _, _, body in await asyncio.gather(*calls)]

    assert session.cancelled == []
    assert sorted(reply["result"]["content"][0]["text"] for reply in replies) == ["waited a", "waited b"]


def test_bind_checks_the_agent_and_the_access_level(gateway):
    agent = open_root(gateway)
    bind_agent_tools(gateway, agent, ToySession(), "read_only")  # access may be passed positionally
    assert agent.tool_access == "read_only"
    with pytest.raises(ValueError, match="Unknown tool access"):
        bind_agent_tools(gateway, agent, ToySession(), access=cast(Any, "admin"))
    with pytest.raises(UnknownAgentError):
        bind_agent_tools(make_gateway(), agent, ToySession())
    gateway.close_agent(agent)
    with pytest.raises(AgentClosedError):
        bind_agent_tools(gateway, agent, ToySession())


# --------------------------------------------------------------------------- transport details


@pytest.mark.asyncio
async def test_bad_or_revoked_secrets_are_refused(gateway, server):
    agent = open_root(gateway)
    bind_agent_tools(gateway, agent, ToySession())
    ping = request("ping")

    for url, api_key in (
        (mcp_url(server.base_url, "not-a-secret"), None),
        (mcp_url(server.base_url), "not-a-secret"),
        (mcp_url(server.base_url), None),
    ):
        status, _, body = await rpc(url, ping, api_key=api_key)
        assert status == 401
        assert body["error"]["code"] == AGENT_ERROR and body["id"] is None
    assert (await rpc(mcp_url(server.base_url), ping, api_key=agent.secret))[0] == 200

    gateway.close_agent(agent)
    status, _, body = await rpc(mcp_url(server.base_url, agent.secret), ping)
    assert status == 410 and body["error"]["code"] == AGENT_ERROR


@pytest.mark.asyncio
async def test_a_shut_down_gateway_refuses_every_model_tool_and_agent_call(gateway, server):
    """What a stopping rollout worker relies on (``ModelGateway.shut_down``): a harness that ignores cancellation,
    inline or over HTTP, can neither sample nor reach its env again, and no subagent or verifier opens."""

    session = ToySession()
    root = open_root(gateway)
    bind_agent_tools(gateway, root, session)
    child = gateway.open_subagent(root, role="subagent")
    gateway.shut_down()
    gateway.shut_down()  # idempotent

    assert gateway.is_shut_down and root.closed and child.closed and gateway.agents("rollout-1") == []
    with pytest.raises(GatewayShutDownError):  # refused before rendering or sampling (this gateway has neither)
        await gateway.chat(root, ChatRequest.from_params([{"role": "user", "content": "go on"}], {}))
    with pytest.raises(AgentClosedError):
        await call_agent_tool(child, "lookup", {"key": "a"})
    with pytest.raises(GatewayShutDownError):
        open_root(gateway, "rollout-2")
    with pytest.raises(GatewayShutDownError):
        gateway.open_subagent(root, role="verifier")
    call = request("tools/call", {"name": "lookup", "arguments": {"key": "a"}})
    status, _, body = await rpc(mcp_url(server.base_url, root.secret), call)
    assert status == 410 and body["error"]["code"] == AGENT_ERROR
    chat = {"model": "policy", "messages": [{"role": "user", "content": "go on"}]}
    status, _, body = await rpc(f"{server.base_url}/v1/chat/completions", chat, api_key=child.secret)
    assert status == 410 and body["error"]["code"] == "agent_closed"
    assert session.calls == []


@pytest.mark.asyncio
async def test_jsonrpc_protocol_errors_notifications_and_batches(gateway, server):
    agent = open_root(gateway)
    bind_agent_tools(gateway, agent, ToySession())
    url = mcp_url(server.base_url, agent.secret)

    status, _, body = await rpc(url, "{not json")
    assert status == 400 and body["error"]["code"] == PARSE_ERROR
    status, _, body = await rpc(url, {"id": 1, "method": "ping"})
    assert status == 400 and body["error"]["code"] == INVALID_REQUEST
    status, _, body = await rpc(url, request("resources/list", request_id=5))
    assert status == 200 and body == {
        "jsonrpc": "2.0",
        "id": 5,
        "error": {"code": METHOD_NOT_FOUND, "message": "Method not found: resources/list"},
    }
    status, _, body = await rpc(url, request("tools/call", {"arguments": {}}))
    assert body["error"]["code"] == INVALID_PARAMS
    status, _, body = await rpc(url, {"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert status == 202 and body == ""

    batch = [request("ping", request_id=1), {"jsonrpc": "2.0", "method": "notifications/initialized"}]
    batch.append(request("tools/call", {"name": "lookup", "arguments": {"key": "b"}}, request_id=2))
    status, _, body = await rpc(url, batch)
    assert status == 200 and [reply["id"] for reply in body] == [1, 2]
    assert body[1]["result"]["content"][0]["text"] == 'root:lookup:{"key": "b"}'


@pytest.mark.asyncio
async def test_protocol_version_negotiation_and_header(gateway, server):
    agent = open_root(gateway)
    url = mcp_url(server.base_url, agent.secret)

    for requested, negotiated in (("2025-03-26", "2025-03-26"), ("1999-01-01", LATEST_PROTOCOL_VERSION)):
        params = {"protocolVersion": requested, "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}
        _, _, body = await rpc(url, request("initialize", params))
        assert body["result"]["protocolVersion"] == negotiated
    status, _, body = await rpc(url, request("ping"), headers={"MCP-Protocol-Version": "2025-06-18"})
    assert status == 200 and body["result"] == {}
    status, _, body = await rpc(url, request("ping"), headers={"MCP-Protocol-Version": "1999-01-01"})
    assert status == 400 and body["error"]["code"] == INVALID_REQUEST


@pytest.mark.asyncio
async def test_get_and_delete_are_not_allowed(gateway, server):
    agent = open_root(gateway)
    async with aiohttp.ClientSession() as http:
        for method in ("GET", "DELETE"):
            async with http.request(method, mcp_url(server.base_url, agent.secret)) as response:
                assert response.status == 405 and response.headers["Allow"] == "POST"


@pytest.mark.asyncio
async def test_long_tool_calls_stream_keepalives_then_the_reply(gateway, server):
    agent = open_root(gateway)
    bind_agent_tools(gateway, agent, ToySession(delay=0.3))
    url = mcp_url(server.base_url, agent.secret)
    sse = {"Accept": "application/json, text/event-stream"}

    status, headers, text = await rpc(url, request("tools/call", {"name": "note"}), headers=sse)
    assert status == 200 and headers["Content-Type"].startswith("text/event-stream")
    assert text.count(": keepalive") >= 2
    *_, last = [event for event in text.split("\n\n") if event.startswith("event: message")]
    assert json.loads(last.split("data: ", 1)[1])["result"]["content"][0]["text"] == "root:note:{}"

    params = {"name": "note", "_meta": {"progressToken": "p-1"}}
    _, _, text = await rpc(url, request("tools/call", params, request_id=9), headers=sse)
    messages = [json.loads(event.split("data: ", 1)[1]) for event in text.split("\n\n") if "data: " in event]
    progress = [m["params"] for m in messages if m.get("method") == "notifications/progress"]
    assert len(progress) >= 2 and all(p["progressToken"] == "p-1" for p in progress)
    assert [p["progress"] for p in progress] == sorted(p["progress"] for p in progress)
    assert messages[-1]["id"] == 9 and "result" in messages[-1]


def test_results_map_to_mcp_content_blocks():
    result = ToolResult(
        content=[
            {"type": "text", "text": "see"},
            {"type": "image", "data": "aGk=", "mime_type": "image/jpeg"},
            {"type": "resource_link", "uri": "file:///a", "name": "a"},
            {"type": "chart", "series": [1, 2]},
            "plain",
        ],
        is_error=True,
    )

    mapped = to_call_tool_result(result, event_id="tool-1")

    assert mapped["content"] == [
        {"type": "text", "text": "see"},
        {"type": "image", "data": "aGk=", "mimeType": "image/jpeg"},
        {"type": "resource_link", "uri": "file:///a", "name": "a"},
        {"type": "text", "text": '{"type": "chart", "series": [1, 2]}'},
        {"type": "text", "text": "plain"},
    ]
    assert mapped["isError"] is True
    assert mapped["_meta"] == {"platoon": {"finished": False, "reward": None, "event_id": "tool-1"}}


# --------------------------------------------------------------------------- the fastmcp and OpenHands clients

_CLIENT_VENV_SCRIPT = """
import asyncio, json, sys
url, secret = sys.argv[1:3]

async def with_fastmcp():
    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport
    transport = StreamableHttpTransport(url, headers={"Authorization": f"Bearer {secret}"})
    async with Client(transport) as client:
        tools = await client.list_tools()
        result = await client.call_tool_mcp("lookup", {"key": "fm"})
        crashed = await client.call_tool_mcp("crash", {})
    return {"tools": [t.name for t in tools], "text": result.content[0].text, "crash_is_error": crashed.isError}

def with_openhands():
    from openhands.sdk.mcp.utils import create_mcp_tools
    config = {"mcpServers": {"env": {"url": url, "headers": {"Authorization": f"Bearer {secret}"}}}}
    client = create_mcp_tools(config, timeout=60)
    try:
        tool = next(t for t in client.tools if t.name == "lookup")
        observation = tool(tool.action_from_arguments({"key": "oh"}))
        return {"tools": [t.name for t in client.tools], "text": [c.text for c in observation.content]}
    finally:
        client.sync_close()

print("RESULT " + json.dumps({"fastmcp": asyncio.run(with_fastmcp()), "openhands": with_openhands()}))
"""


@pytest.mark.asyncio
@pytest.mark.skipif(CLIENT_VENV_PYTHON is None or not CLIENT_VENV_PYTHON.exists(), reason=NO_CLIENT_VENV)
async def test_fastmcp_and_openhands_clients_from_the_client_venv(gateway, server, tmp_path):
    agent = open_root(gateway)
    bind_agent_tools(gateway, agent, ToySession())
    script = tmp_path / "client.py"
    script.write_text(_CLIENT_VENV_SCRIPT)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENHANDS_SUPPRESS_BANNER": "1"}
    env.pop("PYTHONPATH", None)
    process = await asyncio.create_subprocess_exec(
        str(CLIENT_VENV_PYTHON), "-I", str(script), mcp_url(server.base_url), agent.secret,
        cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )  # fmt: skip
    output, _ = await asyncio.wait_for(process.communicate(), timeout=300)
    text = output.decode()
    lines = [line for line in text.splitlines() if line.startswith("RESULT ")]
    assert process.returncode == 0 and lines, text[-4000:]

    result = json.loads(lines[-1].removeprefix("RESULT "))
    assert result["fastmcp"] == {
        "tools": ["lookup", "note", "submit", "crash"],
        "text": 'root:lookup:{"key": "fm"}',
        "crash_is_error": True,
    }
    assert result["openhands"]["tools"] == ["lookup", "note", "submit", "crash"]
    assert result["openhands"]["text"] == ["[Tool 'lookup' executed.]", 'root:lookup:{"key": "oh"}']
    assert [r["name"] for r in tool_call_records(agent)] == ["lookup", "crash", "lookup"]
