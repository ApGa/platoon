"""A tool allowlist per agent role (``agents.roles.<role>.tools``, ``ModelGateway.limit_tools``): an agent of a limited
role, and every agent under it, sees only the tools its allowlists allow, in every listing, and a call to any other
tool is refused where the gateway runs it, for MCP clients and inline harnesses alike, with an error result the agent
sees, which the env never gets and the agent's misc counts.  A gateway that limits no role lists and calls as before."""

from __future__ import annotations

from typing import Any, cast

import pytest
import pytest_asyncio
from test_gateway_mcp import ToySession, make_gateway, open_root, request, rpc

from platoon.envs.service import ToolSpec
from platoon.gateway.core import ModelGateway
from platoon.gateway.mcp import UnknownToolError, bind_agent_tools, call_agent_tool, list_agent_tools, mcp_url
from platoon.gateway.server import GatewayServer
from platoon.utils.role_tools import TOOLS_REFUSED_MISC_KEY, Allowlist, allowlists_of, unmatched

EVERY_TOOL = ["lookup", "note", "submit", "crash"]  # ToySession's


@pytest.fixture
def gateway() -> ModelGateway:
    return make_gateway()


@pytest_asyncio.fixture
async def server(gateway):
    server = GatewayServer(gateway, keepalive_seconds=0.05)
    await server.start(host="127.0.0.1")
    yield server
    await server.stop()


def bound_root(gateway: ModelGateway, session: ToySession, rollout_id: str = "rollout-1"):
    root = open_root(gateway, rollout_id)
    bind_agent_tools(gateway, root, cast(Any, session))
    return root


async def listed(server: GatewayServer, agent) -> list[str]:
    _, _, body = await rpc(mcp_url(server.base_url, agent.secret), request("tools/list"))
    return [tool["name"] for tool in body["result"]["tools"]]


async def called(server: GatewayServer, agent, name: str, arguments: dict | None = None) -> dict:
    payload = request("tools/call", {"name": name, "arguments": arguments or {}})
    status, _, body = await rpc(mcp_url(server.base_url, agent.secret), payload)
    assert status == 200 and "result" in body, body  # a refusal is a tool result, never a protocol error
    return body["result"]


@pytest.mark.asyncio
async def test_a_gateway_that_limits_no_role_lists_and_calls_every_tool(gateway, server):
    session = ToySession()
    root = bound_root(gateway, session)
    child = gateway.mint_subagent(root, role="reviewer")

    assert root.tool_allowlists == () and child.tool_allowlists == ()
    assert await listed(server, child) == EVERY_TOOL
    assert (await called(server, child, "submit", {"answer": "1"}))["content"][0]["text"] == "wrong"
    assert TOOLS_REFUSED_MISC_KEY not in child.misc


@pytest.mark.asyncio
async def test_a_limited_role_lists_only_its_tools_and_a_call_to_another_is_refused_unrun(gateway, server):
    gateway.limit_tools({"reviewer": ["look*", "note"]})
    session = ToySession()
    root = bound_root(gateway, session)
    reviewer = gateway.mint_subagent(root, role="reviewer")  # POST /v1/platoon/agents {"role": "reviewer"}

    assert await listed(server, root) == EVERY_TOOL  # the root's role, policy, is not limited
    assert await listed(server, reviewer) == ["lookup", "note"]
    assert [tool.name for tool in await list_agent_tools(reviewer)] == ["lookup", "note"]

    result = await called(server, reviewer, "submit", {"answer": "1234"})
    assert result["isError"] is True
    assert result["content"][0]["text"] == (
        "Tool 'submit' is not allowed: agents of role 'reviewer' may call only look*, note. The call was not run."
    )
    inline = await call_agent_tool(reviewer, "crash", {})  # an inline harness's call is refused alike
    assert inline.is_error and inline.content[0]["text"].startswith("Tool 'crash' is not allowed")
    assert session.calls == []  # the env never saw either call
    assert reviewer.tool_calls == []  # nor were they recorded as calls
    assert reviewer.misc[TOOLS_REFUSED_MISC_KEY] == {"count": 2, "first": "submit"}

    # The allowed tools run as before.
    assert (await called(server, reviewer, "lookup", {"key": "k"}))["content"][0]["text"] == 'root:lookup:{"key": "k"}'
    assert session.calls == [("lookup", {"key": "k"})]
    assert [record["name"] for record in reviewer.tool_calls] == ["lookup"]


@pytest.mark.asyncio
async def test_a_tool_the_env_does_not_have_stays_an_unknown_tool(gateway, server):
    gateway.limit_tools({"reviewer": ["*"]})
    reviewer = gateway.mint_subagent(bound_root(gateway, ToySession()), role="reviewer")

    _, _, body = await rpc(
        mcp_url(server.base_url, reviewer.secret), request("tools/call", {"name": "nope", "arguments": {}})
    )
    assert body["error"]["message"] == "Unknown tool: nope"
    with pytest.raises(UnknownToolError):
        await call_agent_tool(reviewer, "nope", {})
    assert TOOLS_REFUSED_MISC_KEY not in reviewer.misc


@pytest.mark.asyncio
async def test_an_empty_list_allows_no_tool(gateway, server):
    gateway.limit_tools({"judge": []})
    judge = gateway.mint_subagent(bound_root(gateway, ToySession()), role="judge")

    assert await listed(server, judge) == []
    result = await called(server, judge, "lookup", {"key": "k"})
    assert result["content"][0]["text"] == (
        "Tool 'lookup' is not allowed: agents of role 'judge' may call no tools. The call was not run."
    )


@pytest.mark.asyncio
async def test_the_root_agent_is_limited_by_its_role_policy(gateway, server):
    gateway.limit_tools({"policy": ["lookup", "submit"]})
    session = ToySession()
    root = bound_root(gateway, session)

    assert await listed(server, root) == ["lookup", "submit"]
    assert (await called(server, root, "note", {"text": "x"}))["isError"] is True
    assert session.calls == [] and root.misc[TOOLS_REFUSED_MISC_KEY] == {"count": 1, "first": "note"}


@pytest.mark.asyncio
async def test_an_agent_under_a_limited_one_never_sees_more_than_it(gateway, server):
    gateway.limit_tools({"reviewer": ["lookup", "note"], "scribe": ["note", "submit"]})
    reviewer = gateway.mint_subagent(bound_root(gateway, ToySession()), role="reviewer")
    helper = gateway.mint_subagent(reviewer, role="helper")  # a role without a list: its parent's
    scribe = gateway.mint_subagent(reviewer, role="scribe")  # its parent's and its own
    inline = gateway.open_subagent(reviewer, role="subagent")  # ctx.agents.spawn's children inherit alike

    assert await listed(server, helper) == ["lookup", "note"]
    assert await listed(server, inline) == ["lookup", "note"]
    assert await listed(server, scribe) == ["note"]
    assert scribe.tool_allowlists == (
        Allowlist("reviewer", ("lookup", "note")),
        Allowlist("scribe", ("note", "submit")),
    )
    result = await called(server, scribe, "submit", {"answer": "1"})
    assert result["content"][0]["text"] == (
        "Tool 'submit' is not allowed: agents under an agent of role 'reviewer' may call only lookup, note. The call "
        "was not run."
    )


@pytest.mark.asyncio
async def test_access_levels_apply_first_then_the_role_s_list(gateway, server):
    gateway.limit_tools({"verifier": ["lookup", "note"]})
    verifier = gateway.open_subagent(bound_root(gateway, ToySession()), role="verifier", access="read_only")

    assert await listed(server, verifier) == ["lookup"]  # read_only hides note, the list hides the rest
    _, _, body = await rpc(
        mcp_url(server.base_url, verifier.secret), request("tools/call", {"name": "note", "arguments": {}})
    )
    assert body["error"]["message"] == "Unknown tool: note"  # hidden by its access, as before


@pytest.mark.asyncio
async def test_task_mcp_tools_are_named_as_the_agents_see_them(gateway, server):
    class TaskToolSession(ToySession):
        async def tools(self) -> list[ToolSpec]:
            schema = {"type": "object", "properties": {}}
            names = ["bash", "mcp_server__get_secret", "mcp_server__set_secret", "other__get_secret"]
            return [ToolSpec(name, "", schema) for name in names]

    gateway.limit_tools({"reader": ["mcp_server__get_*", "bash"]})
    reader = gateway.mint_subagent(bound_root(gateway, TaskToolSession()), role="reader")

    assert await listed(server, reader) == ["bash", "mcp_server__get_secret"]


def test_matching_is_fnmatch_and_case_sensitive():
    allowlist = Allowlist("reader", ("read_*", "view", "file_[ab]"))

    assert [name for name in ("read_file", "view", "View", "file_a", "file_c", "read") if allowlist.allows(name)] == [
        "read_file",
        "view",
        "file_a",
    ]
    assert allowlists_of("other", {"reader": ["view"]}) == ()
    assert unmatched(["read_*", "write", "*"], ["read_file", "view"]) == ["write"]
