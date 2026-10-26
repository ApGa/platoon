"""What the tests of a task's MCP servers share: a function harness, ``test-mcp-solver`` (:data:`SOLVER`, registered
on import), that solves a task whose MCP server ``mcp-server`` offers ``get_secret`` through the tool plane, as an
agent of a function harness would (Harbor's hello-mcp, and the stack fixtures ``mcp-http`` and ``mcp-sse``), the check
that a tool an agent's access level hides cannot be called, and a CLI harness that solves those tasks on the direct
path instead, from the task's sandbox at the URL that ``{task_mcp_servers}`` gives (:data:`DIRECT`, a spec's path)::

    collection = await run_harness_rollout(route.attach(task), config, gateway=no_model_gateway())  # harness: SOLVER
    await hidden_from_read_only(session, ROTATE_TOOL)

The function harness calls ``mcp_server__get_secret``, the env tool that the session serves the server's tool as, and
writes what it answers to ``/app/secret.txt`` with the generic ``write`` tool; its final message lists the tools it
saw.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from no_model import no_model_gateway

import platoon as pt
from platoon.envs.service import EnvSession
from platoon.gateway.mcp import UnknownToolError, bind_agent_tools, call_agent_tool, list_agent_tools

SOLVER = "test-mcp-solver"
DIRECT = str(Path(__file__).parent / "envs" / "fixtures" / "harbor-stacks" / "harnesses" / "mcp-direct.toml")
SECRET = "harbor-mcp-secret-12345"
SECRET_TOOL = "mcp_server__get_secret"  # mcp-server's get_secret, as the session serves it
ROTATE_TOOL = "mcp_server__rotate_secret"  # the stack fixtures' mcp-server's rotate_secret


@pt.harness(SOLVER)
async def solve_through_the_plane(ctx: pt.AgentContext) -> pt.AgentResult:
    names = [tool.name for tool in await ctx.list_tools()]
    secret = await ctx.call_tool(SECRET_TOOL, {})
    if secret.is_error:
        return pt.AgentResult("error", final_message=secret.as_text())
    await ctx.call_tool("write", {"path": "/app/secret.txt", "content": secret.as_text()})
    return pt.AgentResult("completed", final_message=json.dumps(names))


def solved(root: dict[str, Any]) -> bool:
    """Whether a rollout of :data:`SOLVER` (its root trajectory, as a dict) scored 1.0 through the plane, its tool
    calls recorded as the trajectory's: the server's tool, then the write."""

    assert (root["misc"]["outcome"]["status"], root["reward"]) == ("scored", 1.0), root["misc"]
    calls = root["misc"]["tool_calls"]
    assert [(call["name"], call["is_error"]) for call in calls] == [(SECRET_TOOL, False), ("write", False)], calls
    assert calls[0]["text"] == SECRET
    return True


def solved_directly(root: dict[str, Any]) -> bool:
    """Whether a rollout of :data:`DIRECT` (its root trajectory, as a dict) scored 1.0 without the tool plane."""

    assert (root["misc"]["outcome"]["status"], root["reward"]) == ("scored", 1.0), root["misc"]
    assert root["misc"].get("tool_calls", []) == []
    return True


async def hidden_from_read_only(session: EnvSession, tool: str) -> bool:
    """Whether an agent with ``read_only`` access neither sees ``tool``, a tool of the session's task MCP server, nor
    can call it: the call is refused before it reaches the server, and is not recorded."""

    gateway = no_model_gateway().model_gateway
    reader = gateway.open_agent(rollout_id="access", trajectory_id="reader", interactions={})
    try:
        bind_agent_tools(gateway, reader, session, access="read_only")
        assert tool not in [listed.name for listed in await list_agent_tools(reader)]
        with pytest.raises(UnknownToolError):
            await call_agent_tool(reader, tool, {})
        assert reader.tool_calls == []
    finally:
        gateway.close_rollout("access")
    return True
