"""A Harbor task's own MCP servers, served through the tool plane (platoon.envs.harbor.session.TaskTools), on the
process provider: the stack fixtures ``mcp-http`` and ``mcp-sse`` (an MCP sidecar over streamable HTTP, and over the
legacy SSE transport) list their tools as env tools as the session opens, which a function harness calls through the
plane to solve the task, filtered by the agent's access level and recorded with the agent's tool calls; a server that
does not list its tools fails the session's start; harnesses in ``main`` get the servers themselves as
``{task_mcp_servers}``, and solve the tasks that way too.  The names, results and failures of the env tools that serve
a server's tools, against small aiohttp servers."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import socket
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, cast

import pytest
from aiohttp import web
from mcp_solver import DIRECT, ROTATE_TOOL, SECRET, SECRET_TOOL, SOLVER, hidden_from_read_only, solved, solved_directly
from no_model import no_model_gateway

from platoon.config_defs import RolloutConfig
from platoon.envs.harbor import HarborEnvironmentService, HarborSession
from platoon.envs.harbor.session import TaskTools, _in_main, task_tool_name
from platoon.envs.harbor.task import HarborTask, McpServer
from platoon.envs.harbor.tools import DEFAULT_MAX_OUTPUT_CHARS
from platoon.envs.service import EnvSetupError
from platoon.gateway.mcp import bind_agent_tools, call_agent_tool, list_agent_tools
from platoon.harness.builtin.codeact.namespace import Ending, tool_function
from platoon.harness.rollout import run_harness_rollout
from platoon.harness.runner import HarnessRunner
from platoon.sandbox.process import ProcessProvider
from platoon.sandbox.stacks import DrivenStack, Stack
from platoon.task import Task
from platoon.train.components import EnvironmentRoute, HarnessExitPolicy

STACKS = Path(__file__).parent / "fixtures" / "harbor-stacks" / "tasks"
TASKS = {"mcp-http": "streamable-http", "mcp-sse": "sse"}  # each fixture, and its server's transport
GENERIC = ["bash", "view", "write", "edit"]

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PLATOON_NODE_SCRATCH", str(tmp_path / "scratch"))
    return tmp_path / "scratch"


@pytest.fixture
def provider(tmp_path: Path) -> ProcessProvider:
    return ProcessProvider(root=tmp_path / "sandboxes")


async def _open(provider: ProcessProvider, name: str, root: Path = STACKS) -> HarborSession:
    service = HarborEnvironmentService(root, provider=provider)
    return await service.open(await service.get_task(name), rollout_id=f"rollout-{name}")


def _copy(name: str, root: Path, *, url: str | None = None, servers: str = "") -> Path:
    """A copy of a stack fixture in ``root``, its server's URL replaced by ``url``, with ``servers`` appended to its
    task.toml (more ``[[environment.mcp_servers]]``)."""

    task = root / name
    shutil.copytree(STACKS / name, task)
    toml = task / "task.toml"
    text = toml.read_text()
    if url is not None:
        text = text.replace("http://mcp-server:8000/mcp", url)
    toml.write_text(text + servers)
    return task


def _nothing_left(provider: ProcessProvider, scratch: Path) -> bool:
    leftovers = [
        *(provider.root.iterdir() if provider.root.exists() else ()),
        *(scratch.iterdir() if scratch.exists() else ()),
    ]
    assert leftovers == [], leftovers
    return True


@pytest.mark.parametrize("name", list(TASKS))
async def test_the_tasks_mcp_tools_are_env_tools_that_reach_its_server(
    provider: ProcessProvider, scratch: Path, name: str
):
    session = await _open(provider, name)
    try:
        tools = {tool.name: tool for tool in await session.tools()}
        assert list(tools) == [*GENERIC, SECRET_TOOL, ROTATE_TOOL]
        secret, rotate = tools[SECRET_TOOL], tools[ROTATE_TOOL]
        assert (secret.read_only, secret.terminal, secret.description) == (True, False, "Returns a secret value.")
        assert (rotate.read_only, rotate.terminal) == (False, False)
        assert secret.input_schema == {"type": "object", "properties": {}}

        result = await session.call_tool(SECRET_TOOL, {})
        assert (result.as_text(), result.is_error) == (SECRET, False)
        unknown = await session.call_tool("mcp_server__nope", {})
        assert unknown.is_error and "Unknown tool 'mcp_server__nope'" in unknown.as_text()
        # The generic tools are as they were; a shared fork shares the stack and its servers' sessions.
        fork = await session.fork("shared")
        assert [tool.name for tool in await fork.tools()] == list(tools)
        assert (await fork.call_tool(SECRET_TOOL, {})).as_text() == SECRET
        await fork.close()
    finally:
        await session.close()
    assert [tool.name for tool in await session.tools()] == GENERIC  # a closed session lists the generic tools alone
    assert _nothing_left(provider, scratch)


@pytest.mark.parametrize("name", list(TASKS))
async def test_a_harness_in_main_gets_the_servers_as_main_reaches_them(provider: ProcessProvider, name: str):
    """``{task_mcp_servers}`` lists the servers with their URLs as a harness in ``main`` reaches them: on the process
    provider, at the port of this host that the stack holds for the service, as main's own environment has it."""

    session = await _open(provider, name)
    try:
        assert session.sandbox is not None
        (server,) = session.task_mcp_servers
        reached = (await session.sandbox.exec(["sh", "-c", "echo $MCP_URL"])).stdout.strip()
        assert server == {"name": "mcp-server", "transport": TASKS[name], "url": reached}
        assert reached.startswith("http://127.0.0.1:") and reached != "http://127.0.0.1:8000"
        # The oracle reaches the server that way too.
        assert (await session.run_solution()).exit_code == 0
        assert (await session.grade()).reward == 1.0
    finally:
        await session.close()


# --- through the plane -----------------------------------------------------------------------------------------------


async def _rollout(provider: ProcessProvider, harness: str, task: str, tmp_path: Path) -> dict[str, Any]:
    settings = {"dataset_dir": str(STACKS), "provider": "process", "provider_options": {"root": str(provider.root)}}
    route = EnvironmentRoute(
        name="stacks", service="harbor", service_kwargs=settings, harness=harness, resolve_task=True
    )
    config = RolloutConfig(output_dir=str(tmp_path / "out"), return_dict=True, verbose=False)
    runner = HarnessRunner(logs_dir=tmp_path / "logs", state_root=tmp_path / "state", kill_grace_s=0.2)
    collection = await run_harness_rollout(
        route.attach(Task(id=task)), config, gateway=no_model_gateway(), runner=runner
    )
    assert isinstance(collection, dict)
    (root,) = collection["trajectories"].values()
    return root


@pytest.mark.parametrize("name", list(TASKS))
async def test_a_function_harness_solves_the_task_through_the_plane(
    provider: ProcessProvider, scratch: Path, tmp_path: Path, name: str
):
    assert solved(await _rollout(provider, SOLVER, name, tmp_path / "solved"))  # with its tool calls recorded
    noop = await _rollout(provider, "platoon-noop", name, tmp_path / "noop")
    assert (noop["misc"]["outcome"]["status"], noop["reward"]) == ("scored", 0.0), noop["misc"]
    assert _nothing_left(provider, scratch)


@pytest.mark.parametrize("name", list(TASKS))
async def test_a_harness_in_main_solves_the_task_at_the_url_it_is_given(
    provider: ProcessProvider, scratch: Path, tmp_path: Path, name: str
):
    """The direct path: a CLI harness in the task's sandbox reaches the server at the URL ``{task_mcp_servers}`` gives
    it (on the process provider, the port of this host that the stack holds), over the server's transport."""

    assert solved_directly(await _rollout(provider, DIRECT, name, tmp_path))
    assert _nothing_left(provider, scratch)


async def test_a_tool_the_agents_access_hides_cannot_be_called(provider: ProcessProvider):
    """``read_only`` access lists the server's read-only tool alone, and a call of the other one never reaches the
    server: the secret is still the one the server started with."""

    session = await _open(provider, "mcp-http")
    gateway = no_model_gateway().model_gateway
    try:
        assert await hidden_from_read_only(session, ROTATE_TOOL)
        reader = gateway.open_agent(rollout_id="r", trajectory_id="reader", interactions={})
        bind_agent_tools(gateway, reader, session, access="read_only")
        assert [tool.name for tool in await list_agent_tools(reader)] == [SECRET_TOOL]
        assert (await call_agent_tool(reader, SECRET_TOOL, {})).as_text() == SECRET
        assert [record["name"] for record in reader.tool_calls] == [SECRET_TOOL]

        writer = gateway.open_agent(rollout_id="r", trajectory_id="writer", interactions={})
        bind_agent_tools(gateway, writer, session, access="full")
        assert (await call_agent_tool(writer, ROTATE_TOOL, {})).as_text() == "rotated"
        assert (await call_agent_tool(reader, SECRET_TOOL, {})).as_text() != SECRET
    finally:
        await session.close()
        gateway.close_rollout("r")


# --- servers that fail ------------------------------------------------------------------------------------------------

# A URL whose port the service does not declare, and one whose path the server does not serve.
UNLISTED = {
    "http://mcp-server:8001/mcp": r"did not list its tools: ValueError: The service 'mcp-server' declares no port 8001",
    "http://mcp-server:8000/elsewhere": r"did not list its tools: McpError: the server answered initialize with HTTP 4",
}


@pytest.mark.parametrize("url", list(UNLISTED))
async def test_a_server_that_does_not_list_its_tools_fails_the_sessions_start(
    provider: ProcessProvider, scratch: Path, tmp_path: Path, url: str
):
    """The tools are listed as the session opens: a server that does not list them fails the start, naming it, and the
    stack is stopped, so no rollout runs without the tools of its task."""

    _copy("mcp-http", tmp_path / "tasks", url=url)
    with pytest.raises(EnvSetupError, match=r"The task's MCP server mcp-server \(port 800\d of the service ") as raised:
        await _open(provider, "mcp-http", tmp_path / "tasks")
    assert raised.value.failure_kind == "sidecar_start"
    assert re.search(UNLISTED[url], str(raised.value)), raised.value  # and why
    assert _nothing_left(provider, scratch)


@pytest.mark.parametrize("policy", ["exclude", "grade"])
async def test_a_rollout_whose_server_does_not_list_its_tools_is_never_graded(
    provider: ProcessProvider, tmp_path: Path, policy: str
):
    """Whatever the route does with a harness that exits (``on_harness_exit``), a rollout whose task's server lists no
    tools does not start, so it is an ``error`` that never trains, not a 0 that would."""

    _copy("mcp-http", tmp_path / "tasks", url="http://mcp-server:8001/mcp")
    settings = {"dataset_dir": str(tmp_path / "tasks"), "provider": "process"}
    settings["provider_options"] = {"root": str(provider.root)}
    route = EnvironmentRoute(
        name="stacks",
        service="harbor",
        service_kwargs=settings,
        harness=SOLVER,
        resolve_task=True,
        on_harness_exit=HarnessExitPolicy.coerce(policy),
    )
    config = RolloutConfig(output_dir=str(tmp_path / "out"), return_dict=True, verbose=False)
    with pytest.raises(EnvSetupError) as raised:
        await run_harness_rollout(route.attach(Task(id="mcp-http")), config, gateway=no_model_gateway())
    assert raised.value.failure_kind == "sidecar_start"


async def test_a_server_gone_after_the_start_fails_its_own_tools_alone(provider: ProcessProvider):
    """Through the plane, as every harness calls the tools: once the server is gone, the generic tools still answer,
    and a call of the server's tool is an error result, recorded as the agent's call."""

    session = await _open(provider, "mcp-http")
    gateway = no_model_gateway().model_gateway
    try:
        assert session.stack is not None
        stack = cast(DrivenStack, session.stack)
        handle = cast(Any, stack._handles["mcp-server"])
        shell = int(handle._entry.split(":")[0])  # the shell that runs the server's command, and waits for it
        [server] = Path(f"/proc/{shell}/task/{shell}/children").read_text().split()
        os.kill(int(server), signal.SIGKILL)
        while (await handle.status())[0]:
            await asyncio.sleep(0.02)
        agent = gateway.open_agent(rollout_id="r", trajectory_id="agent", interactions={})
        bind_agent_tools(gateway, agent, session, access="full")
        assert [tool.name for tool in await list_agent_tools(agent)] == [*GENERIC, SECRET_TOOL, ROTATE_TOOL]
        assert (await call_agent_tool(agent, "bash", {"command": "echo hi"})).as_text().startswith("hi")
        failed = await call_agent_tool(agent, SECRET_TOOL, {})
        assert failed.is_error and "The task's MCP server mcp-server" in failed.as_text()
        assert [(call["name"], call["is_error"]) for call in agent.tool_calls] == [("bash", False), (SECRET_TOOL, True)]
    finally:
        await session.close()
        gateway.close_rollout("r")


# --- which servers the plane serves, and what harnesses get ---------------------------------------------------------


async def test_the_plane_serves_the_http_servers_that_name_a_service_of_the_stack(tmp_path: Path):
    task = tmp_path / "task"
    shutil.copytree(STACKS / "mcp-http", task)
    toml = task / "task.toml"
    toml.write_text(
        toml.read_text()
        + '\n[[environment.mcp_servers]]\nname = "by-alias"\ntransport = "sse"\nurl = "http://tools.internal/sse"\n'
        + '\n[[environment.mcp_servers]]\nname = "elsewhere"\ntransport = "sse"\nurl = "http://example.org:9/sse"\n'
        + '\n[[environment.mcp_servers]]\nname = "local"\ntransport = "stdio"\ncommand = "npx"\nargs = ["server"]\n'
        + '\n[[environment.mcp_servers]]\nname = "tls"\ntransport = "sse"\nurl = "https://mcp-server/sse"\n'
    )
    compose = task / "environment" / "docker-compose.yaml"
    compose.write_text(compose.read_text().replace("    expose:\n", "    hostname: tools.internal\n    expose:\n", 1))
    harbor_task = HarborTask.load(task)
    served = {server.name: harbor_task.mcp_service(server) for server in harbor_task.mcp_servers}
    assert served == {
        "mcp-server": ("mcp-server", 8000),
        "by-alias": ("mcp-server", 80),  # the URL names no port: http's
        "elsewhere": None,
        "local": None,
        "tls": None,
    }
    assert McpServer("local", "stdio", command="npx", args=("server",)).described() == {
        "name": "local",
        "transport": "stdio",
        "command": "npx",
        "args": ["server"],
    }


async def test_tools_that_would_be_served_under_one_name_fail_the_start(
    provider: ProcessProvider, scratch: Path, tmp_path: Path
):
    """Two servers whose names serve alike (``mcp-server``, ``mcp_server``) and which both list ``get_secret``: the
    second tool would be served as the first, which fails the start, naming both.  A task that is no stack loads with
    two servers of one name, as before: nothing serves them."""

    alike = (
        '\n[[environment.mcp_servers]]\nname = "mcp_server"\ntransport = "streamable-http"\n'
        'url = "http://mcp-server:8000/mcp"\n'
    )
    task = _copy("mcp-http", tmp_path / "tasks", servers=alike)
    match = r"server mcp_server lists the tool 'get_secret', .* as the tool 'get_secret' of the server mcp-server is"
    with pytest.raises(EnvSetupError, match=match):
        await _open(provider, "mcp-http", tmp_path / "tasks")
    assert _nothing_left(provider, scratch)
    (task / "environment" / "docker-compose.yaml").unlink()
    twice = '\n[[environment.mcp_servers]]\nname = "mcp-server"\nurl = "http://mcp-server:8000/sse"\n'
    (task / "task.toml").write_text((task / "task.toml").read_text() + twice)
    assert [server.name for server in HarborTask.load(task).mcp_servers] == ["mcp-server", "mcp_server", "mcp-server"]


async def test_a_single_sandbox_task_serves_none_of_its_servers(tmp_path: Path):
    """mcp-http without its compose file: one sandbox, whose server's URL names no service, so the plane serves none."""

    copy = _copy("mcp-http", tmp_path)
    (copy / "environment" / "docker-compose.yaml").unlink()
    task = HarborTask.load(copy)
    (server,) = task.mcp_servers
    assert not task.is_stack and server.http_address == ("mcp-server", 8000)
    assert task.mcp_service(server) is None
    assert await TaskTools.open(cast(Stack, object()), task) is None


async def test_a_url_without_a_port_reaches_main_at_port_80():
    """``{task_mcp_servers}`` on the process provider: a URL that names no port is rewritten as one that names 80."""

    class Rewriting:
        def rewrite(self, value: str, *, service: str | None = None) -> str:
            return value.replace("http://mcp-server:80", "http://127.0.0.1:5000")

    class Named:
        def rewrite(self, value: str, *, service: str | None = None) -> str:
            return value

    assert _in_main(cast(Stack, Rewriting()), "http://mcp-server/mcp") == "http://127.0.0.1:5000/mcp"
    assert _in_main(cast(Stack, Rewriting()), "http://mcp-server:80/mcp") == "http://127.0.0.1:5000/mcp"
    assert _in_main(cast(Stack, Named()), "http://mcp-server/mcp") == "http://mcp-server/mcp"


# --- the env tools that serve a server's tools ------------------------------------------------------------------------


class _Reaching:
    """A stack whose connector reaches every service at one port of this host's loopback."""

    def __init__(self, port: int) -> None:
        self.port = port

    async def connect(self, service: str, port: int) -> str:
        return f"tcp://127.0.0.1:{self.port}"


@contextlib.asynccontextmanager
async def _serving(
    tools: list[dict[str, Any]], result: dict[str, Any] | None = None, *, listing_s: float = 0.0
) -> AsyncGenerator[Stack]:
    """A streamable HTTP server that lists ``tools`` (after ``listing_s``) and answers every call with ``result``, and
    a stack that reaches it as mcp-http's ``mcp-server``."""

    async def post(request: web.Request) -> web.Response:
        body = await request.json()
        if "id" not in body:
            return web.Response(status=202)
        if body["method"] == "tools/list":
            await asyncio.sleep(listing_s)
        answers = {
            "initialize": {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "t"}},
            "tools/list": {"tools": tools},
            "tools/call": result or {"content": []},
        }
        return web.json_response({"jsonrpc": "2.0", "id": body["id"], "result": answers[body["method"]]})

    app = web.Application()
    app.router.add_post("/mcp", post)
    runner = web.AppRunner(app, shutdown_timeout=0.5)
    await runner.setup()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    await web.TCPSite(runner, "127.0.0.1", port).start()
    try:
        yield cast(Stack, _Reaching(port))
    finally:
        await runner.cleanup()


async def test_an_env_tools_name_is_the_servers_and_the_tools_made_fit_for_model_providers():
    assert task_tool_name("mcp-server", "get_secret") == "mcp_server__get_secret"
    assert task_tool_name("my.server", "get secret/v2") == "my_server__get_secret_v2"
    assert task_tool_name("2fa", "get_code") == "_2fa__get_code"  # a function's name starts with no digit
    assert task_tool_name("été", "x") == f"_t__{hashlib.sha256('été'.encode()).hexdigest()[:8]}__x"
    names = {task_tool_name("s", tool) for tool in ("查询", "搜索", "__")}  # names in other scripts stay apart
    assert len(names) == 3 and all(name.isascii() and name.isidentifier() for name in names)


async def test_the_served_names_are_functions_that_codeact_can_call():
    tools = [{"name": "get-secret", "annotations": {"readOnlyHint": True}}, {"name": "rotate.secret"}]
    async with _serving(tools) as stack:
        served = await TaskTools.open(stack, HarborTask.load(STACKS / "mcp-http"))
        assert served is not None
        try:
            assert [tool.name for tool in served.specs] == ["mcp_server__get_secret", "mcp_server__rotate_secret"]
            assert "mcp_server__get_secret" in served and "mcp-server__get-secret" not in served
            function = tool_function(cast(Any, None), served.specs[0], Ending())
            assert function.describe().splitlines()[0] == "def mcp_server__get_secret() -> str"
        finally:
            await served.close()


async def test_a_listing_that_takes_too_long_fails_saying_so():
    slow = r"server mcp-server \(port 8000 of the service mcp-server\) did not list its tools within 0.5 s$"
    async with _serving([{"name": "t"}], listing_s=30) as stack:
        with pytest.raises(RuntimeError, match=slow):
            await TaskTools.open(stack, HarborTask.load(STACKS / "mcp-http"), timeout_s=0.5)


async def test_a_tool_whose_name_would_be_too_long_fails_the_listing():
    async with _serving([{"name": "ok"}, {"name": "x" * 60}]) as stack:
        with pytest.raises(RuntimeError, match=r"served as mcp_server__x{60}, a name longer than 64 characters"):
            await TaskTools.open(stack, HarborTask.load(STACKS / "mcp-http"))


async def test_a_long_result_is_cut_as_the_generic_tools_output_is():
    """Text blocks longer together than the generic tools' output become one, cut, in the first one's place; other
    blocks keep theirs."""

    image = {"type": "image", "data": "AAAA", "mimeType": "image/png"}
    blocks = [{"type": "text", "text": "a" * 40_000}, image, {"type": "text", "text": "b" * 40_000}]
    async with _serving([{"name": "flood"}], {"content": blocks}) as stack:
        served = await TaskTools.open(stack, HarborTask.load(STACKS / "mcp-http"))
        assert served is not None
        try:
            result = await served.call("mcp_server__flood", {}, timeout_s=5)
        finally:
            await served.close()
    text, kept = result.content
    assert kept == image and text["type"] == "text"
    assert text["text"].startswith("a" * 100) and text["text"].endswith("b" * 100)
    assert len(text["text"]) < DEFAULT_MAX_OUTPUT_CHARS + 100 and "characters truncated" in text["text"]


@pytest.mark.parametrize(
    "block",
    [
        {"type": "resource", "resource": {"uri": "file:///big.txt", "mimeType": "text/plain", "text": "r" * 40_000}},
        {"text": "t" * 40_000},
        {"type": "custom", "data": "c" * 40_000},
    ],
    ids=["resource", "no-type", "unknown-type"],
)
async def test_a_block_that_is_no_text_image_or_audio_is_text_that_the_cut_bounds(block: dict[str, Any]):
    async with _serving([{"name": "flood"}], {"content": [block]}) as stack:
        served = await TaskTools.open(stack, HarborTask.load(STACKS / "mcp-http"))
        assert served is not None
        try:
            result = await served.call("mcp_server__flood", {}, timeout_s=5)
        finally:
            await served.close()
    (text,) = result.content
    assert text["type"] == "text" and text["text"].startswith(json.dumps(block)[:100])
    assert len(text["text"]) < DEFAULT_MAX_OUTPUT_CHARS + 100 and "characters truncated" in text["text"]
