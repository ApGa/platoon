"""The MCP client through which the tool plane serves a Harbor task's own MCP servers (platoon.envs.harbor.mcp_client):
both transports against the stack fixtures' standard-library server, over TCP and over a Unix socket that platoon's
forwarder serves (as the docker relay does), and the protocol's edges against small aiohttp servers."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import AsyncGenerator, Callable, Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from aiohttp import web

from platoon.envs.harbor import mcp_client
from platoon.envs.harbor.mcp_client import PROTOCOL_VERSIONS, McpClient, McpError, Transport
from platoon.sandbox.routes import FORWARDER

SERVER = (
    Path(__file__).parent / "fixtures" / "harbor-stacks" / "tasks" / "mcp-http" / "environment" / "mcp" / "server.py"
)
SECRET = "harbor-mcp-secret-12345"
PATHS = {"streamable-http": "/mcp", "sse": "/sse"}

pytestmark = pytest.mark.asyncio


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _wait_for(ready: Callable[[], bool], what: str, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not ready():
        assert time.monotonic() < deadline, f"{what} never came up"
        time.sleep(0.05)


def _listens(port: int) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
    except OSError:
        return False
    return True


@pytest.fixture(params=["streamable-http", "sse"])
def fixture_server(request: pytest.FixtureRequest) -> Iterator[tuple[str, int]]:
    """The stack fixtures' MCP server on a port of this host's loopback, over each transport."""

    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, "-I", str(SERVER), "--transport", request.param], env={"PLATOON_PORT_8000": str(port)}
    )
    try:
        _wait_for(lambda: _listens(port), "the fixture MCP server")
        yield request.param, port
    finally:
        server.kill()
        server.wait()


@pytest.fixture
def short_tmp() -> Iterator[Path]:
    """A directory with a path short enough for a Unix socket's address."""

    directory = Path(tempfile.mkdtemp(prefix="mcp-", dir="/tmp"))
    try:
        yield directory
    finally:
        for path in directory.iterdir():
            path.unlink()
        directory.rmdir()


async def test_both_transports_list_and_call_through_a_tcp_endpoint(fixture_server: tuple[str, int]):
    transport, port = fixture_server
    url = f"http://mcp-server:8000{PATHS[transport]}"  # as main names it; this host reaches it at the port
    client = await McpClient.connect(url, transport=cast(Transport, transport), endpoint=f"tcp://127.0.0.1:{port}")
    try:
        assert client.protocol_version in PROTOCOL_VERSIONS
        tools = await client.list_tools()
        assert [tool["name"] for tool in tools] == ["get_secret", "rotate_secret"]
        assert tools[0]["annotations"] == {"readOnlyHint": True}
        result = await client.call_tool("get_secret", {}, timeout_s=10)
        assert result == {"content": [{"type": "text", "text": SECRET}], "isError": False}
        with pytest.raises(McpError, match="Unknown tool: nope") as error:
            await client.call_tool("nope", {}, timeout_s=10)
        assert error.value.code == -32602
        # Calls at once each get their own answer, on either transport.
        answers = await asyncio.gather(*(client.call_tool("get_secret", {}, timeout_s=10) for _ in range(8)))
        assert {answer["content"][0]["text"] for answer in answers} == {SECRET}
    finally:
        await client.close()
        await client.close()  # idempotent


async def test_both_transports_reach_a_server_through_the_forwarders_unix_socket(
    fixture_server: tuple[str, int], short_tmp: Path
):
    """As the docker provider's relay serves a stack's port: platoon's forwarder listens on a Unix socket, which
    appears only once it accepts connections, open to whoever reaches its directory (0666)."""

    transport, port = fixture_server
    path = short_tmp / "8000.sock"
    relay = subprocess.Popen(
        [sys.executable, "-I", str(FORWARDER), "--serve-socket", str(path), "--port", str(port)],
        stderr=subprocess.PIPE,
    )
    try:
        _wait_for(path.exists, "the relay's socket")
        assert path.stat().st_mode & 0o777 == 0o666
        url = f"http://mcp-server:8000{PATHS[transport]}"
        client = await McpClient.connect(url, transport=cast(Transport, transport), endpoint=f"unix://{path}")
        try:
            result = await client.call_tool("get_secret", {}, timeout_s=10)
            assert result["content"] == [{"type": "text", "text": SECRET}]
        finally:
            await client.close()
    finally:
        relay.kill()
        relay.wait()


# --- the protocol's edges, against aiohttp servers ------------------------------------------------------------------


class Recorded:
    """What a test server was sent: each request's method, path, headers and JSON body."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict[str, str], Any]] = []

    def methods(self) -> list[Any]:
        return [body.get("method") if isinstance(body, dict) else body for _, _, _, body in self.requests]


@contextlib.asynccontextmanager
async def _serving(app: web.Application) -> AsyncGenerator[int]:
    """Serve ``app`` on a port of this host's loopback, which it gives; a stream still open at the end is cut."""

    runner = web.AppRunner(app, shutdown_timeout=0.5)
    await runner.setup()
    port = _free_port()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    try:
        yield port
    finally:
        await runner.cleanup()


def _result(body: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": body["id"], "result": result}


INITIALIZED = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "t"}}


def _streamable(
    recorded: Recorded, answer: Callable[[dict[str, Any], web.Request], Any], *, sessions: Iterator[str] | None = None
) -> web.Application:
    """A streamable HTTP server at /mcp: ``answer`` gives each request's result (or a web.Response of its own), and
    each ``initialize`` the next of ``sessions`` (default: ``s1``) as its session id."""

    async def post(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        recorded.requests.append(("POST", request.path, dict(request.headers), body))
        if "id" not in body:
            return web.Response(status=202)
        if body["method"] == "initialize":
            session = next(sessions) if sessions is not None else "s1"
            return web.json_response(_result(body, INITIALIZED), headers={"Mcp-Session-Id": session})
        answered = await answer(body, request)
        if isinstance(answered, web.StreamResponse):
            return answered
        return web.json_response(_result(body, answered))

    async def delete(request: web.Request) -> web.Response:
        recorded.requests.append(("DELETE", request.path, dict(request.headers), None))
        return web.Response()

    app = web.Application()
    app.router.add_post("/mcp", post)
    app.router.add_delete("/mcp", delete)
    return app


async def _connect(port: int, *, transport: str = "streamable-http", path: str = "/mcp") -> McpClient:
    url = f"http://mcp-server:8000{path}"
    return await McpClient.connect(url, transport=cast(Transport, transport), endpoint=f"tcp://127.0.0.1:{port}")


async def test_streamable_http_keeps_the_session_and_the_servers_host_and_ends_it_at_close():
    recorded = Recorded()

    async def answer(body: dict[str, Any], request: web.Request) -> Any:
        return {"tools": []}

    async with _serving(_streamable(recorded, answer)) as port:
        client = await _connect(port)
        assert await client.list_tools() == []
        await client.close()

    (method, _, first, _), *rest = recorded.requests
    assert first["Host"] == "mcp-server:8000" and "Mcp-Session-Id" not in first  # the URL's own host, from main
    assert "application/json" in first["Accept"] and "text/event-stream" in first["Accept"]
    assert recorded.methods() == ["initialize", "notifications/initialized", "tools/list", None]
    for kind, _, headers, _ in rest:
        assert headers["Mcp-Session-Id"] == "s1" and headers["MCP-Protocol-Version"] == "2025-06-18", kind
    assert rest[-1][0] == "DELETE"  # the close ends the session


async def test_a_server_that_forgot_the_session_gets_a_new_one_once():
    recorded = Recorded()

    async def answer(body: dict[str, Any], request: web.Request) -> Any:
        if request.headers.get("Mcp-Session-Id") == "old":
            return web.Response(status=404)
        return {"content": [{"type": "text", "text": "ok"}]}

    async with _serving(_streamable(recorded, answer, sessions=iter(["old", "new"]))) as port:
        client = await _connect(port)
        try:
            assert (await client.call_tool("t", {}, timeout_s=5))["content"][0]["text"] == "ok"
        finally:
            await client.close()
    assert recorded.methods().count("initialize") == 2


async def test_a_server_answering_with_another_protocol_version_is_refused():
    async def post(request: web.Request) -> web.Response:
        body = await request.json()
        return web.json_response(_result(body, {**INITIALIZED, "protocolVersion": "1999-01-01"}))

    app = web.Application()
    app.router.add_post("/mcp", post)
    async with _serving(app) as port:
        with pytest.raises(McpError, match="the server speaks MCP '1999-01-01'"):
            await _connect(port)


async def test_an_answer_on_an_event_stream_after_the_servers_own_requests():
    """The server asks the client things on the stream before it answers: ``ping`` gets an empty result, anything else
    an error (the client offers no capability), and the answer to the call comes last."""

    recorded = Recorded()

    async def answer(body: dict[str, Any], request: web.Request) -> Any:
        if body["method"] != "tools/call":
            return {"tools": []}
        stream = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await stream.prepare(request)
        for message in (
            {"jsonrpc": "2.0", "id": "p", "method": "ping"},
            {"jsonrpc": "2.0", "id": "s", "method": "sampling/createMessage", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}},
            _result(body, {"content": [{"type": "text", "text": "x" * 200_000}]}),  # long, in many chunks
        ):
            await stream.write(b": a comment\n" + f"event: message\ndata: {json.dumps(message)}\r\n\r\n".encode())
        return stream

    async with _serving(_streamable(recorded, answer)) as port:
        client = await _connect(port)
        try:
            result = await client.call_tool("t", {}, timeout_s=5)
            await asyncio.sleep(0.2)
        finally:
            await client.close()
    assert len(result["content"][0]["text"]) == 200_000
    replies = {
        body["id"]: body for _, _, _, body in recorded.requests if isinstance(body, dict) and "method" not in body
    }
    assert replies["p"]["result"] == {} and replies["s"]["error"]["code"] == -32601


async def test_a_listing_goes_through_every_page():
    recorded = Recorded()

    async def answer(body: dict[str, Any], request: web.Request) -> Any:
        cursor = (body.get("params") or {}).get("cursor")
        if cursor is None:
            return {"tools": [{"name": "a"}], "nextCursor": "2"}
        return {"tools": [{"name": "b"}, {"no": "name"}]}  # a tool without a name is no tool

    async with _serving(_streamable(recorded, answer)) as port:
        client = await _connect(port)
        try:
            assert [tool["name"] for tool in await client.list_tools()] == ["a", "b"]
        finally:
            await client.close()


@pytest.mark.parametrize("paging", ["again", "long"])
async def test_a_listing_that_never_ends_or_is_too_long_fails(monkeypatch: pytest.MonkeyPatch, paging: str):
    """A listing whose cursor names a page it gave before fails at once, rather than list until its timeout, and one
    longer than MAX_MESSAGE_BYTES in all fails however short its pages are, rather than fill the worker's memory."""

    monkeypatch.setattr(mcp_client, "MAX_MESSAGE_BYTES", 10_000)
    pages = itertools.count()

    async def answer(body: dict[str, Any], request: web.Request) -> Any:
        page = next(pages)
        cursor = "again" if paging == "again" else str(page)
        return {"tools": [{"name": f"t{page}", "description": "d" * 1000}], "nextCursor": cursor}

    async with _serving(_streamable(Recorded(), answer)) as port:
        client = await _connect(port)
        try:
            match = 'the cursor "again", which names no new page' if paging == "again" else "longer than 10000 bytes"
            with pytest.raises(McpError, match=match):
                await client.list_tools()
        finally:
            await client.close()
    assert next(pages) == (2 if paging == "again" else 10)


async def test_a_call_with_no_answer_times_out():
    async def answer(body: dict[str, Any], request: web.Request) -> Any:
        if body["method"] == "tools/call":
            await asyncio.sleep(30)
        return {}

    async with _serving(_streamable(Recorded(), answer)) as port:
        client = await _connect(port)
        try:
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                await client.call_tool("slow", {}, timeout_s=0.5)
            assert time.monotonic() - started < 5
        finally:
            await client.close()


async def test_http_errors_and_answers_that_are_no_answers_fail():
    async def answer(body: dict[str, Any], request: web.Request) -> Any:
        if body["params"]["name"] == "500":
            return web.Response(status=500, text="boom")
        return web.Response(text="<html>", content_type="text/html")

    async with _serving(_streamable(Recorded(), answer)) as port:
        client = await _connect(port)
        try:
            with pytest.raises(McpError, match="HTTP 500: boom"):
                await client.call_tool("500", {}, timeout_s=5)
            with pytest.raises(McpError, match="answered tools/call with text/html"):
                await client.call_tool("html", {}, timeout_s=5)
        finally:
            await client.close()


@pytest.mark.parametrize("framing", ["sized", "chunked", "event"])
async def test_an_answer_longer_than_a_message_may_be_fails_the_request(monkeypatch: pytest.MonkeyPatch, framing: str):
    """However the server frames it, an answer longer than MAX_MESSAGE_BYTES is never read whole: it fails the call,
    rather than fill the worker's memory.  An event's length is in bytes, not characters."""

    monkeypatch.setattr(mcp_client, "MAX_MESSAGE_BYTES", 1000)

    async def answer(body: dict[str, Any], request: web.Request) -> Any:
        message = _result(body, {"content": [{"type": "text", "text": "\u20ac" * 600}]})  # 1800 bytes, 600 characters
        if framing == "sized":
            return web.json_response(message)
        content_type = "text/event-stream" if framing == "event" else "application/json"
        stream = web.StreamResponse(headers={"Content-Type": content_type})
        await stream.prepare(request)
        text = json.dumps(message, ensure_ascii=False)
        await stream.write((f"event: message\ndata: {text}\n\n" if framing == "event" else text).encode())
        return stream

    async with _serving(_streamable(Recorded(), answer)) as port:
        client = await _connect(port)
        try:
            with pytest.raises(McpError, match="more than 1000 bytes"):
                await client.call_tool("t", {}, timeout_s=5)
        finally:
            await client.close()


@pytest.mark.parametrize("location", ["/mcp", "http://mcp-server:8000/mcp"])
async def test_a_redirect_within_the_servers_origin_is_followed_through_the_endpoint(location: str):
    """A URL the server redirects (one with a trailing slash, which servers redirect to their own name): the client
    follows it itself, through the stack's endpoint, where aiohttp would look the server's name up."""

    recorded = Recorded()

    async def answer(body: dict[str, Any], request: web.Request) -> Any:
        return {"tools": [{"name": "t"}]}

    async def redirect(request: web.Request) -> web.Response:
        raise web.HTTPTemporaryRedirect(location)

    app = _streamable(recorded, answer)
    app.router.add_route("*", "/mcp/", redirect)
    async with _serving(app) as port:
        client = await _connect(port, path="/mcp/")
        try:
            assert [tool["name"] for tool in await client.list_tools()] == ["t"]
        finally:
            await client.close()
    assert recorded.methods() == ["initialize", "notifications/initialized", "tools/list", None]
    assert {headers["Host"] for _, _, headers, _ in recorded.requests} == {"mcp-server:8000"}


async def test_a_redirect_elsewhere_or_that_changes_the_method_fails():
    """A 307 to another origin is refused before anything goes there, and a 302 (which turns a POST into a GET) is the
    server's answer, an error."""

    hits: list[str] = []

    async def elsewhere(request: web.Request) -> web.Response:
        hits.append(request.path)
        return web.json_response({})

    other = web.Application()
    other.router.add_route("*", "/{tail:.*}", elsewhere)
    async with _serving(other) as other_port:

        async def away(request: web.Request) -> web.Response:
            raise web.HTTPTemporaryRedirect(f"http://127.0.0.1:{other_port}/stolen")

        async def found(request: web.Request) -> web.Response:
            raise web.HTTPFound("/mcp")

        app = web.Application()
        app.router.add_post("/away", away)
        app.router.add_post("/found", found)
        async with _serving(app) as port:
            with pytest.raises(McpError, match=f"sent the client to 'http://127.0.0.1:{other_port}/stolen'"):
                await _connect(port, path="/away")
            with pytest.raises(McpError, match="answered initialize with HTTP 302"):
                await _connect(port, path="/found")
    assert hits == []


def _legacy(
    recorded: Recorded, *, endpoint: str = "/messages?session=1", end_after: int | None = None
) -> web.Application:
    """A legacy SSE server: GET /sse names ``endpoint``, and each POST's answer comes on the stream (the stream ends
    after ``end_after`` answers, when given)."""

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def stream(request: web.Request) -> web.StreamResponse:
        recorded.requests.append(("GET", request.path, dict(request.headers), None))
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(f"event: endpoint\ndata: {endpoint}\n\n".encode())
        sent = 0
        while (message := await queue.get()) is not None:
            await response.write(f"event: message\ndata: {json.dumps(message)}\n\n".encode())
            sent += 1
            if end_after is not None and sent >= end_after:
                break
        return response

    async def post(request: web.Request) -> web.Response:
        body = await request.json()
        recorded.requests.append(("POST", request.path_qs, dict(request.headers), body))
        if "id" in body and "method" in body:
            if body["method"] == "initialize":
                await queue.put(_result(body, {**INITIALIZED, "protocolVersion": "2024-11-05"}))
            elif body["method"] == "tools/list":
                await queue.put(_result(body, {"tools": [{"name": "t"}]}))
            # tools/call: never answered
        return web.Response(status=202)

    app = web.Application()
    app.router.add_get("/sse", stream)
    app.router.add_post("/messages", post)
    return app


async def test_the_legacy_transport_posts_where_the_stream_says_with_the_servers_host():
    recorded = Recorded()
    async with _serving(_legacy(recorded)) as port:
        client = await _connect(port, transport="sse", path="/sse")
        try:
            assert client.protocol_version == "2024-11-05"
            assert [tool["name"] for tool in await client.list_tools()] == ["t"]
        finally:
            await client.close()
    assert [(kind, path) for kind, path, _, _ in recorded.requests] == [
        ("GET", "/sse"),
        ("POST", "/messages?session=1"),
        ("POST", "/messages?session=1"),
        ("POST", "/messages?session=1"),
    ]
    assert {headers["Host"] for _, _, headers, _ in recorded.requests} == {"mcp-server:8000"}


async def test_a_legacy_stream_that_ends_fails_the_requests_that_wait():
    recorded = Recorded()
    async with _serving(_legacy(recorded, end_after=2)) as port:  # initialize's and tools/list's answers, then the end
        client = await _connect(port, transport="sse", path="/sse")
        try:
            await client.list_tools()
            started = time.monotonic()
            with pytest.raises(McpError, match="event stream"):
                await client.call_tool("t", {}, timeout_s=30)
            assert time.monotonic() - started < 10
        finally:
            await client.close()


async def test_the_legacy_transport_answers_so_many_of_the_servers_requests_at_once():
    """A server that floods its stream with pings and takes its time with the replies: the client answers as many as
    _MAX_ANSWERING at once and leaves the others, rather than pile them up, and goes on reading the stream, where the
    answer to its own request comes after them."""

    replies: list[Any] = []
    released = asyncio.Event()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def stream(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b"event: endpoint\ndata: /messages\n\n")
        while True:
            await response.write(f"event: message\ndata: {json.dumps(await queue.get())}\n\n".encode())

    async def post(request: web.Request) -> web.Response:
        body = await request.json()
        if "method" not in body:  # a reply to a ping, which the server takes until the test ends
            replies.append(body["id"])
            await released.wait()
        elif body["method"] == "initialize":
            await queue.put(_result(body, {**INITIALIZED, "protocolVersion": "2024-11-05"}))
        elif body["method"] == "tools/list":
            for ping in range(200):
                await queue.put({"jsonrpc": "2.0", "id": f"p{ping}", "method": "ping"})
            await queue.put(_result(body, {"tools": [{"name": "t"}]}))
        return web.Response(status=202)

    app = web.Application()
    app.router.add_get("/sse", stream)
    app.router.add_post("/messages", post)
    async with _serving(app) as port:
        client = await _connect(port, transport="sse", path="/sse")
        try:
            assert [tool["name"] for tool in await client.list_tools()] == ["t"]
            assert len(cast(Any, client)._transport._answers) == mcp_client._MAX_ANSWERING
            for _ in range(100):  # the replies on their way
                if len(replies) >= mcp_client._MAX_ANSWERING:
                    break
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.2)
        finally:
            released.set()
            await client.close()
    assert sorted(replies) == sorted(f"p{ping}" for ping in range(mcp_client._MAX_ANSWERING))


async def test_a_legacy_endpoint_elsewhere_is_refused():
    async with _serving(_legacy(Recorded(), endpoint="http://elsewhere:9/messages")) as port:
        with pytest.raises(McpError, match="sent the client to 'http://elsewhere:9/messages'"):
            await _connect(port, transport="sse", path="/sse")


class _KeepingAlive:
    """An HTTP/1.1 MCP server over ``transport`` that keeps each connection for a later request, as uvicorn does, and
    ends one that idled longer than ``keep_alive_s`` as the next request arrives on it, unanswered: what a client sees
    when the server's keep-alive ends as its request is on its way.  It lists the tool ``t``, whose calls it answers
    ``ok`` and counts."""

    def __init__(self, transport: str, keep_alive_s: float) -> None:
        self.transport = transport
        self.keep_alive_s = keep_alive_s
        self.port = 0
        self.calls = 0
        self.handlers: set[asyncio.Task[Any]] = set()
        self._events: asyncio.Queue[bytes] = asyncio.Queue()  # what the legacy transport's stream sends next

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        handler = asyncio.current_task()
        assert handler is not None
        self.handlers.add(handler)
        answered_at: float | None = None  # when the connection answered its last request
        try:
            while True:
                head = await reader.readuntil(b"\r\n\r\n")
                if answered_at is not None and time.monotonic() - answered_at > self.keep_alive_s:
                    return  # its keep-alive has ended: the connection closes, the request unanswered
                start, *lines = head.decode().split("\r\n")
                method = start.split(" ")[0]
                headers = {name.lower(): value.strip() for name, _, value in (line.partition(":") for line in lines)}
                body = await reader.readexactly(int(headers.get("content-length", "0")))
                if method == "GET":  # the legacy transport's stream, which stays open until the client goes
                    writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
                    writer.write(b"event: endpoint\ndata: /messages\n\n")
                    while True:
                        await writer.drain()
                        writer.write(await self._events.get())
                status, answer = self._answer(json.loads(body))
                headings = f"HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {len(answer)}\r\n"
                writer.write(f"{headings}\r\n".encode() + answer)
                await writer.drain()
                if headers.get("connection", "").lower() == "close":
                    return
                answered_at = time.monotonic()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass  # the client closed the connection
        finally:
            self.handlers.discard(handler)
            writer.close()

    def _answer(self, message: dict[str, Any]) -> tuple[str, bytes]:
        """The status and body of a POST's answer; the legacy transport sends the JSON-RPC answer on its stream."""

        if "id" not in message:
            return "202 Accepted", b""
        if message["method"] == "initialize":
            result = {**INITIALIZED, "protocolVersion": "2024-11-05" if self.transport == "sse" else "2025-06-18"}
        elif message["method"] == "tools/list":
            result = {"tools": [{"name": "t"}]}
        else:
            self.calls += 1
            result = {"content": [{"type": "text", "text": "ok"}]}
        reply = json.dumps(_result(message, result))
        if self.transport == "sse":
            self._events.put_nowait(f"event: message\ndata: {reply}\n\n".encode())
            return "202 Accepted", b""
        return "200 OK", reply.encode()


@contextlib.asynccontextmanager
async def _keeping_alive(transport: str, *, keep_alive_s: float) -> AsyncGenerator[_KeepingAlive]:
    server = _KeepingAlive(transport, keep_alive_s)
    listening = await asyncio.start_server(server.handle, "127.0.0.1", 0)
    server.port = listening.sockets[0].getsockname()[1]
    try:
        yield server
    finally:
        listening.close()
        handlers = list(server.handlers)
        for handler in handlers:
            handler.cancel()
        await asyncio.gather(*handlers, return_exceptions=True)
        await listening.wait_closed()


@pytest.mark.parametrize("transport", ["streamable-http", "sse"])
async def test_a_call_after_the_servers_keep_alive_ended_is_answered(transport: str):
    """A server keeps a connection for a later request as long as its keep-alive (uvicorn's, which Harbor's hello-mcp
    runs on, 5 s), and ends one that idled longer as the next request arrives on it: an agent's call that came a while
    after the listing, on the connection that carried the listing, got no answer (aiohttp's "Server disconnected").
    The client sends each request on a connection of its own, so the call is answered, and made once."""

    async with _keeping_alive(transport, keep_alive_s=0.2) as server:
        client = await _connect(server.port, transport=transport, path=PATHS[transport])
        try:
            assert [tool["name"] for tool in await client.list_tools()] == ["t"]
            await asyncio.sleep(0.5)  # longer than the server's keep-alive, as an agent's first call comes later
            result = await client.call_tool("t", {}, timeout_s=5)
        finally:
            await client.close()
    assert result["content"] == [{"type": "text", "text": "ok"}]
    assert server.calls == 1


async def test_only_http_servers_are_reached():
    with pytest.raises(McpError, match="only an http:// MCP server"):
        await McpClient.connect("https://mcp-server/mcp", transport="sse", endpoint="tcp://127.0.0.1:1")
    with pytest.raises(ValueError, match="unix:// or tcp://"):
        await McpClient.connect("http://mcp-server/mcp", transport="sse", endpoint="mcp-server:80")


async def test_an_unreachable_server_fails_to_connect(short_tmp: Path):
    with pytest.raises(OSError):
        await McpClient.connect(
            "http://mcp-server:8000/mcp", transport="streamable-http", endpoint=f"unix://{short_tmp}/none.sock"
        )


async def test_the_client_offers_the_protocol_versions_the_gateway_serves():
    mcp = pytest.importorskip("platoon.gateway.mcp")
    assert PROTOCOL_VERSIONS == mcp.SUPPORTED_PROTOCOL_VERSIONS
    assert os.path.exists(FORWARDER)
