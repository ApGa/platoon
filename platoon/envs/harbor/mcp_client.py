"""A small MCP client, through which the tool plane serves the MCP servers of a Harbor task's stack (RFC 0010, §6).

A task's ``[[environment.mcp_servers]]`` that run in its stack (``http://mcp-server:8000/mcp``) listen where only the
stack's services reach them.  This host reaches them through the stack's connector
(:meth:`platoon.sandbox.stacks.Stack.connect`): a Unix socket (a pod's, or the docker provider's relay), or a TCP port
of this host (the ``process`` provider's).  :class:`McpClient` makes the requests the plane needs over it:
``initialize`` (then ``notifications/initialized``), ``tools/list`` and ``tools/call``, on either transport a task
names, which are two code paths:

- **streamable HTTP** (``streamable-http``): each message is a POST to the server's URL, answered with JSON or with an
  event stream that ends with the answer.  The ``Mcp-Session-Id`` the server gives with its answer to ``initialize``
  goes with every later request, a server that no longer knows it (404) gets a new session, once, and the client's
  close ends it (DELETE);
- **the legacy SSE transport** (``sse``, MCP 2024-11-05, Harbor's default for task servers): a GET opens an event
  stream whose first event (``endpoint``) says where to POST each message, and the answers come on that stream.

Every request goes to the server's own URL, with its own ``Host``: the server sees what ``main`` would send it, on a
connection of its own, which closes with the answer (:meth:`_Route.session`).  The client follows the server's
redirects itself, within its origin (:meth:`_Jsonrpc.send`).  It offers the protocol versions platoon's gateway accepts
(:data:`PROTOCOL_VERSIONS`) and refuses a server that answers with another; it offers no capability, so a request the
server sends it gets an error, but ``ping``, which gets an empty result.  It reads no message, and no listing of tools,
longer than :data:`MAX_MESSAGE_BYTES`, and answers no more than 16 of the server's requests at once, so that no server
fills the rollout worker's memory.  It needs aiohttp and nothing else: no ``mcp`` package enters the rollout workers.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
import urllib.parse
from collections.abc import AsyncIterator, Mapping
from typing import Any, Literal

import aiohttp

logger = logging.getLogger(__name__)

Transport = Literal["sse", "streamable-http"]
#: The protocol versions the client offers and accepts, newest last: the ones platoon's gateway serves.
PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25")
CLIENT_INFO = {"name": "platoon", "version": "1"}
#: How long the client waits for a server's answer to ``initialize`` or ``tools/list``.
REQUEST_TIMEOUT_S = 60.0
#: The longest message a server may send, a JSON answer or an event of a stream, and the longest listing of its tools,
#: in JSON over all its pages: a longer one fails the request, rather than fill the rollout worker's memory.
MAX_MESSAGE_BYTES = 16 * 1024 * 1024
_MAX_ANSWERING = 16  # of the server's own requests, which the legacy transport answers at once; it leaves the others
_MAX_REDIRECTS = 5  # of one request, which the client follows
_ERROR_BODY_BYTES = 1000  # how much of an HTTP error's body the client reads, for its message
_ACCEPT = "application/json, text/event-stream"
_SESSION_HEADER = "Mcp-Session-Id"
_VERSION_HEADER = "MCP-Protocol-Version"
_METHOD_NOT_FOUND = -32601


class McpError(RuntimeError):
    """A server could not be reached, broke the protocol, or answered a request with an error (``code``, JSON-RPC's,
    for an error answer; None otherwise)."""

    def __init__(self, message: str, *, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


class _SessionExpired(Exception):
    """A streamable HTTP server answered 404 to a request of a session it gave: the client starts a new one."""


class McpClient:
    """One session with one MCP server, open from :meth:`connect` until :meth:`close`."""

    def __init__(self, transport: _StreamableHttp | _LegacySse) -> None:
        self._transport = transport
        self.protocol_version = ""

    @classmethod
    async def connect(
        cls, url: str, *, transport: Transport, endpoint: str, timeout_s: float = REQUEST_TIMEOUT_S
    ) -> McpClient:
        """Open a session with the server at ``url``, reached through ``endpoint`` (``unix:///<socket>`` or
        ``tcp://<host>:<port>``): the transport's connection, then ``initialize``.  :class:`McpError` says why it
        could not."""

        route = _Route(url, endpoint)
        if transport == "streamable-http":
            client = cls(_StreamableHttp(route))
        elif transport == "sse":
            client = cls(_LegacySse(route))
        else:
            raise ValueError(f"An MCP server's transport is sse or streamable-http, not {transport!r}")
        try:
            await client._transport.open(timeout_s)
            await client._initialize(timeout_s)
        except BaseException:
            await client.close()
            raise
        return client

    async def list_tools(self, *, timeout_s: float = REQUEST_TIMEOUT_S) -> list[dict[str, Any]]:
        """Every tool the server lists, as MCP's ``Tool`` objects, through every page of its listing.  A listing longer
        than :data:`MAX_MESSAGE_BYTES` in all, or whose cursor names a page it gave before, fails, rather than fill the
        rollout worker's memory or never end."""

        tools: list[dict[str, Any]] = []
        cursors: set[str] = set()  # the pages the listing has named
        cursor, size = None, 0  # size: the bytes of the listing's JSON so far
        while True:
            result = await self.request("tools/list", {"cursor": cursor} if cursor else {}, timeout_s=timeout_s)
            listed = result.get("tools")
            if not isinstance(listed, list):
                raise McpError(f"tools/list answered without a list of tools: {_short(result)}")
            size += len(json.dumps(result))
            if size > MAX_MESSAGE_BYTES:
                raise McpError(f"the server's listing of its tools is longer than {MAX_MESSAGE_BYTES} bytes")
            tools.extend(tool for tool in listed if isinstance(tool, dict) and isinstance(tool.get("name"), str))
            cursor = result.get("nextCursor")
            if not cursor:
                return tools
            if not isinstance(cursor, str) or cursor in cursors:
                named = _short(json.dumps(cursor))
                raise McpError(f"tools/list answered with the cursor {named}, which names no new page")
            cursors.add(cursor)

    async def call_tool(self, name: str, arguments: Mapping[str, Any], *, timeout_s: float) -> dict[str, Any]:
        """The server's ``CallToolResult`` for one call of ``name``; :class:`McpError` for an error answer (an unknown
        tool, say) or none within ``timeout_s`` (``TimeoutError``)."""

        return await self.request("tools/call", {"name": name, "arguments": dict(arguments)}, timeout_s=timeout_s)

    async def request(self, method: str, params: Mapping[str, Any], *, timeout_s: float) -> dict[str, Any]:
        """The result of one request, in a new session if the server has forgotten this one."""

        try:
            return await self._transport.request(method, params, timeout_s=timeout_s)
        except _SessionExpired:
            await self._transport.reset()
            await self._initialize(timeout_s)
            return await self._transport.request(method, params, timeout_s=timeout_s)

    async def close(self) -> None:
        """End the session; idempotent, and it never raises."""

        await self._transport.close()

    async def _initialize(self, timeout_s: float) -> None:
        params = {"protocolVersion": PROTOCOL_VERSIONS[-1], "capabilities": {}, "clientInfo": CLIENT_INFO}
        result = await self._transport.request("initialize", params, timeout_s=timeout_s)
        version = result.get("protocolVersion")
        if version not in PROTOCOL_VERSIONS:
            raise McpError(f"the server speaks MCP {version!r}, and platoon's client {', '.join(PROTOCOL_VERSIONS)}")
        self.protocol_version = self._transport.version = version
        await self._transport.notify("notifications/initialized", {})


class _Route:
    """Where the requests to a server's URLs go: the URL as it is through a Unix socket, or to a TCP endpoint's address
    with the URL's own ``Host``."""

    def __init__(self, url: str, endpoint: str) -> None:
        self.url = url
        parts = urllib.parse.urlsplit(url)
        if parts.scheme != "http" or not parts.netloc:
            raise McpError(f"platoon reaches only an http:// MCP server through a stack, not {url!r}")
        self._origin = (parts.scheme, parts.netloc)
        self._socket: str | None = None
        self._address: str | None = None
        if endpoint.startswith("unix://"):
            self._socket = endpoint.removeprefix("unix://")
        elif endpoint.startswith("tcp://"):
            self._address = endpoint.removeprefix("tcp://")
        else:
            raise ValueError(f"A stack's endpoint is unix:// or tcp://, not {endpoint!r}")

    def session(self) -> aiohttp.ClientSession:
        """A client session whose connections go where the endpoint says; no total timeout, since a legacy SSE stream
        stays open for the session's life (each request has its own).  It keeps no connection for a later request
        (``force_close``, which also asks the server to close each): a server ends a connection that idles longer than
        its keep-alive (uvicorn's, which Harbor's hello-mcp runs on, 5 s), and a request that reaches one as it ends
        gets no answer, which the client cannot tell from a call the server made, so it cannot send the request again.
        An agent's call that came a while after the listing would fail so."""

        if self._socket:
            connector: aiohttp.BaseConnector = aiohttp.UnixConnector(path=self._socket, force_close=True)
        else:
            connector = aiohttp.TCPConnector(force_close=True)
        return aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=None))

    def absolute(self, url: str) -> str:
        """``url``, one of the server's, or relative to its URL, as an absolute URL."""

        return urllib.parse.urljoin(self.url, url)

    def target(self, url: str) -> tuple[str, dict[str, str]]:
        """The URL to request for ``url`` (one of the server's, or relative to its URL), and the headers that keep its
        ``Host``."""

        absolute = self.absolute(url)
        parts = urllib.parse.urlsplit(absolute)
        if (parts.scheme, parts.netloc) != self._origin:
            raise McpError(f"the server sent the client to {absolute!r}, which is not where it is ({self.url!r})")
        if self._address is None:
            return absolute, {}
        return parts._replace(netloc=self._address).geturl(), {"Host": parts.netloc}


class _Jsonrpc:
    """What both transports share: request ids, the session's protocol version, and answering the server's requests."""

    def __init__(self, route: _Route) -> None:
        self.route = route
        self.version: str | None = None  # the session's protocol version, once initialized
        self._ids = itertools.count(1)
        self._http: aiohttp.ClientSession | None = None

    def message(self, method: str, params: Mapping[str, Any], *, request: bool = True) -> dict[str, Any]:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": dict(params)}
        if request:
            message["id"] = next(self._ids)
        return message

    @property
    def http(self) -> aiohttp.ClientSession:
        if self._http is None:
            self._http = self.route.session()
        return self._http

    async def close_http(self) -> None:
        http, self._http = self._http, None
        if http is not None:
            with contextlib.suppress(Exception):
                await http.close()

    async def send(
        self, method: str, url: str, headers: Mapping[str, str], data: str | None = None
    ) -> aiohttp.ClientResponse:
        """The server's response to a request to ``url`` (one of its URLs, or one relative to its URL), sent where the
        route says, with the URL's own ``Host``.  The client follows the server's 307 and 308 redirects itself, as
        many as :data:`_MAX_REDIRECTS`, and only within the server's origin (:meth:`_Route.target`): aiohttp's own would
        leave the route, for another origin, or for the server's name, which only the stack resolves.  The response to
        a request redirected otherwise is the redirect."""

        for _ in range(_MAX_REDIRECTS + 1):
            target, host = self.route.target(url)
            response = await self.http.request(
                method, target, headers={**host, **headers}, data=data, allow_redirects=False
            )
            location = response.headers.get("Location")
            if response.status not in (307, 308) or location is None:
                return response
            response.release()
            url = urllib.parse.urljoin(self.route.absolute(url), location)
        raise McpError(f"the server redirected the client more than {_MAX_REDIRECTS} times")

    async def answer_server(self, message: Mapping[str, Any]) -> None:
        """Answer a request the server sent: ``ping`` with an empty result, any other with an error, since the client
        offers no capability.  A notification gets nothing."""

        if "id" not in message or not isinstance(message.get("method"), str):
            return
        if message["method"] == "ping":
            reply: dict[str, Any] = {"jsonrpc": "2.0", "id": message["id"], "result": {}}
        else:
            error = {"code": _METHOD_NOT_FOUND, "message": f"platoon's client does not take {message['method']}"}
            reply = {"jsonrpc": "2.0", "id": message["id"], "error": error}
        with contextlib.suppress(Exception):  # the server's own business: the client's request goes on without it
            await self.post_reply(reply)

    async def post_reply(self, reply: Mapping[str, Any]) -> None:
        """Send the server an answer to a request of its own."""

        raise NotImplementedError


class _StreamableHttp(_Jsonrpc):
    """The streamable HTTP transport: a POST per message, answered with JSON or an event stream."""

    def __init__(self, route: _Route) -> None:
        super().__init__(route)
        self._session_id: str | None = None

    async def open(self, timeout_s: float) -> None:
        """Nothing to open: each message is a request of its own."""

    async def request(self, method: str, params: Mapping[str, Any], *, timeout_s: float) -> dict[str, Any]:
        message = self.message(method, params)
        async with asyncio.timeout(timeout_s):
            async with await self._post(message) as response:
                if method == "initialize":
                    self._session_id = response.headers.get(_SESSION_HEADER)
                await self._check(response, method)
                content_type = response.headers.get("Content-Type", "")
                if content_type.startswith("application/json"):
                    return _result(_reply_in(_json(await _body(response)), message["id"]), method)
                if content_type.startswith("text/event-stream"):
                    async for event, data in _events(response.content):
                        reply = _parsed(data) if event == "message" else None
                        if isinstance(reply, dict) and reply.get("id") == message["id"] and "method" not in reply:
                            return _result(reply, method)
                        if isinstance(reply, dict):
                            await self.answer_server(reply)
                    raise McpError(f"the server's stream ended before it answered {method}")
                raise McpError(f"the server answered {method} with {content_type or 'no content type'}")

    async def notify(self, method: str, params: Mapping[str, Any]) -> None:
        async with asyncio.timeout(REQUEST_TIMEOUT_S):
            async with await self._post(self.message(method, params, request=False)) as response:
                await self._check(response, method)

    async def post_reply(self, reply: Mapping[str, Any]) -> None:
        async with asyncio.timeout(REQUEST_TIMEOUT_S):
            async with await self._post(reply):
                pass

    async def reset(self) -> None:
        self._session_id = self.version = None

    async def close(self) -> None:
        if self._session_id is not None and self._http is not None:
            with contextlib.suppress(Exception):  # ending the session is a courtesy: the stack goes anyway
                async with asyncio.timeout(5.0):
                    async with await self.send("DELETE", self.route.url, self._headers()):
                        pass
        self._session_id = None
        await self.close_http()

    async def _post(self, message: Mapping[str, Any]) -> aiohttp.ClientResponse:
        headers = {**self._headers(), "Accept": _ACCEPT, "Content-Type": "application/json"}
        return await self.send("POST", self.route.url, headers, json.dumps(message))

    def _headers(self) -> dict[str, str]:
        headers = {}
        if self._session_id is not None:
            headers[_SESSION_HEADER] = self._session_id
        if self.version is not None:
            headers[_VERSION_HEADER] = self.version
        return headers

    async def _check(self, response: aiohttp.ClientResponse, method: str) -> None:
        if response.status == 404 and self._session_id is not None and method != "initialize":
            raise _SessionExpired
        if response.status >= 300:  # an error, or a redirect that the client does not follow
            raise McpError(f"the server answered {method} with HTTP {response.status}: {await _error_text(response)}")


class _LegacySse(_Jsonrpc):
    """The legacy SSE transport: one event stream for the session, on which the answers to the POSTs come."""

    def __init__(self, route: _Route) -> None:
        super().__init__(route)
        self._endpoint: str | None = None  # where each message is posted, as the stream's first event says
        self._waiting: dict[Any, asyncio.Future[dict[str, Any]]] = {}
        self._reader: asyncio.Task[None] | None = None
        self._stream: aiohttp.ClientResponse | None = None
        self._answers: set[asyncio.Task[None]] = set()  # answers to the server's own requests, being sent

    async def open(self, timeout_s: float) -> None:
        endpoint: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        async with asyncio.timeout(timeout_s):
            self._stream = await self.send("GET", self.route.url, {"Accept": "text/event-stream"})
            if self._stream.status != 200:
                raise McpError(f"the server answered its event stream's GET with HTTP {self._stream.status}")
            self._reader = asyncio.ensure_future(self._read(self._stream, endpoint))
            self._endpoint = await asyncio.shield(endpoint)

    async def request(self, method: str, params: Mapping[str, Any], *, timeout_s: float) -> dict[str, Any]:
        message = self.message(method, params)
        answer: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._waiting[message["id"]] = answer
        try:
            async with asyncio.timeout(timeout_s):
                await self._post(message, method)
                return _result(await answer, method)
        finally:
            self._waiting.pop(message["id"], None)

    async def notify(self, method: str, params: Mapping[str, Any]) -> None:
        async with asyncio.timeout(REQUEST_TIMEOUT_S):
            await self._post(self.message(method, params, request=False), method)

    async def post_reply(self, reply: Mapping[str, Any]) -> None:
        async with asyncio.timeout(REQUEST_TIMEOUT_S):
            await self._post(reply, "a reply")

    async def reset(self) -> None:
        """Nothing: this transport's session is its event stream, which no server forgets while it is open."""

    async def close(self) -> None:
        for answering in list(self._answers):
            answering.cancel()
        if self._reader is not None:
            self._reader.cancel()
            with contextlib.suppress(BaseException):
                await self._reader
        if self._stream is not None:
            self._stream.close()
        await self.close_http()
        self._fail(McpError("the client closed"))

    async def _post(self, message: Mapping[str, Any], what: str) -> None:
        if self._endpoint is None or self._reader is None or self._reader.done():
            raise McpError("the server's event stream is closed")
        headers = {"Content-Type": "application/json"}
        async with await self.send("POST", self._endpoint, headers, json.dumps(message)) as response:
            if response.status >= 300:
                raise McpError(f"the server answered {what} with HTTP {response.status}: {await _error_text(response)}")

    async def _read(self, stream: aiohttp.ClientResponse, endpoint: asyncio.Future[str]) -> None:
        """Read the event stream: its ``endpoint``, then the answers, each to the request that waits for it."""

        error: Exception = McpError("the server's event stream ended")
        try:
            async for event, data in _events(stream.content):
                if event == "endpoint" and not endpoint.done():
                    endpoint.set_result(data.strip())
                    continue
                message = _parsed(data) if event == "message" else None
                if not isinstance(message, dict):
                    continue
                if "method" in message:  # answered aside as the stream goes on, as many as _MAX_ANSWERING at once
                    if len(self._answers) < _MAX_ANSWERING:
                        answering = asyncio.ensure_future(self.answer_server(message))
                        self._answers.add(answering)
                        answering.add_done_callback(self._answers.discard)
                elif (waiting := self._waiting.get(message.get("id"))) is not None and not waiting.done():
                    waiting.set_result(message)
        except (aiohttp.ClientError, McpError) as exc:
            error = McpError(f"the server's event stream failed: {exc}")
        finally:
            if not endpoint.done():
                endpoint.set_exception(McpError("the server's event stream ended before it named its endpoint"))
            self._fail(error)

    def _fail(self, error: Exception) -> None:
        for waiting in self._waiting.values():
            if not waiting.done():
                waiting.set_exception(error)


async def _events(stream: aiohttp.StreamReader) -> AsyncIterator[tuple[str, str]]:
    """The events of a ``text/event-stream`` (each one's type, ``message`` by default, and data), as the HTML standard
    parses them: comments skipped, data lines joined with newlines, an event dispatched at a blank line."""

    too_long = McpError(f"the server sent an event of more than {MAX_MESSAGE_BYTES} bytes")
    pending = bytearray()  # what came after the last full line
    event, data, size = "", [], 0  # size: the bytes of the event's lines so far
    async for chunk in stream.iter_any():
        pending += chunk
        while (end := pending.find(b"\n")) >= 0:
            raw = pending[:end].removesuffix(b"\r")
            del pending[: end + 1]
            if not raw:
                if data:
                    yield event or "message", "\n".join(data)
                event, data, size = "", [], 0
                continue
            size += len(raw)
            if size > MAX_MESSAGE_BYTES:
                raise too_long
            field, _, value = raw.decode("utf-8", errors="replace").partition(":")
            value = value.removeprefix(" ")
            if field == "event":
                event = value
            elif field == "data":
                data.append(value)
        if size + len(pending) > MAX_MESSAGE_BYTES:
            raise too_long


async def _body(response: aiohttp.ClientResponse) -> bytes:
    """An answer's body, read only as far as :data:`MAX_MESSAGE_BYTES`: a longer one is :class:`McpError`."""

    too_long = McpError(f"the server sent an answer of more than {MAX_MESSAGE_BYTES} bytes")
    if (response.content_length or 0) > MAX_MESSAGE_BYTES:
        raise too_long
    body = bytearray()
    async for chunk in response.content.iter_any():
        body += chunk
        if len(body) > MAX_MESSAGE_BYTES:
            raise too_long
    return bytes(body)


async def _error_text(response: aiohttp.ClientResponse) -> str:
    """The start of an HTTP error's body, for the error's message."""

    with contextlib.suppress(aiohttp.ClientError):  # the error is the status's, whatever its body
        return _short((await response.content.read(_ERROR_BODY_BYTES)).decode("utf-8", errors="replace"))
    return ""


def _json(body: bytes) -> Any:
    try:
        return json.loads(body)
    except ValueError as exc:
        raise McpError(f"the server's answer is not JSON: {exc}") from None


def _parsed(data: str) -> Any:
    try:
        return json.loads(data)
    except ValueError:
        return None


def _reply_in(body: Any, request_id: int) -> Any:
    """The answer to ``request_id`` in a JSON body: the body, or the one in a batch that answers it."""

    if isinstance(body, list):
        return next((item for item in body if isinstance(item, dict) and item.get("id") == request_id), None)
    return body


def _result(reply: Any, method: str) -> dict[str, Any]:
    """The result of a JSON-RPC answer, or :class:`McpError` for an error answer, or none."""

    if not isinstance(reply, dict):
        raise McpError(f"the server answered {method} with no JSON-RPC answer: {_short(reply)}")
    if isinstance(error := reply.get("error"), dict):
        code = error.get("code")
        raise McpError(
            f"{method} failed: {error.get('message') or error}", code=code if isinstance(code, int) else None
        )
    result = reply.get("result")
    if not isinstance(result, dict):
        raise McpError(f"the server answered {method} with no result: {_short(reply)}")
    return result


def _short(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=repr)
    return text.strip()[:300]


__all__ = ["CLIENT_INFO", "MAX_MESSAGE_BYTES", "PROTOCOL_VERSIONS", "McpClient", "McpError", "Transport"]
