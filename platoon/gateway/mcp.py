"""MCP tool plane: an agent's env tools, served over streamable-HTTP MCP at ``/s/<secret>/mcp``.

A harness reaches its environment's tools the way it reaches the model: through the gateway, as the agent its
key names.  :func:`bind_agent_tools` attaches an :class:`AgentTools` provider over an ``EnvSession`` to an agent,
with an access level and an optional listing transform.  Subagents opened with ``ModelGateway.open_subagent``
inherit their parent's provider, and its access unless they are given one, until they are rebound, e.g. to a
forked session or a narrower access level.  Every tool call, whether it arrives over MCP or from an inline
harness through :func:`call_agent_tool`, is recorded on the agent (``agent.tool_calls``);
:func:`tool_call_records` turns the records into trajectory misc.

Access levels filter listings on the env's tool hints: ``full`` lists every tool, ``no_terminal`` hides tools
with ``terminal=True`` (submit tools), and ``read_only`` lists only tools with ``read_only=True``.  A tool the
agent cannot see cannot be called.  ``ToolSpec.meta`` becomes the MCP tool's ``_meta``; a tool result's
``finished``, ``reward`` and ``metadata`` travel in ``_meta["platoon"]``, which MCP clients do not show the
model.

The endpoint is a small JSON-RPC handler for the MCP streamable-HTTP transport (protocol versions 2024-11-05
to 2025-11-25), not the ``mcp`` SDK's ASGI transport: the gateway is an aiohttp app, the agent key already
identifies the session, and the gateway does not depend on ``mcp``.  It is stateless: it issues no
``Mcp-Session-Id``, so GET (a server-initiated stream) and DELETE answer 405 as the transport allows.
``tools/call`` answers with SSE when the client accepts it, sending keepalive comments (or progress
notifications, when the client asked for progress) while the tool runs, so a long env tool never trips a
client's read timeout.  ``notifications/cancelled`` cancels the agent's call with that request id: a call still
waiting for its turn on the env never runs, a running one finishes in the env, and no response is sent.  A
client that disconnects cancels its call the same way.

Calls into one live env are ordered by the env, not here: sessions that say ``manages_concurrency`` (ORS and
Harbor) queue every agent's calls on their live env's gate (:mod:`platoon.envs.gate`), so the root, its
children, verifiers and agents minted over HTTP take turns on the state they share.  The plane names the agent of
each call (:func:`~platoon.envs.gate.on_behalf_of`), whose turn it is, also when agents share one session.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypedDict, get_args

from aiohttp import web

from platoon.envs.gate import on_behalf_of
from platoon.envs.service import ToolResult, ToolSpec
from platoon.gateway.core import AgentClosedError, AgentSession, GatewayError, ToolAccess, UnknownAgentError

if TYPE_CHECKING:
    from platoon.envs.service import EnvSession
    from platoon.gateway.core import ModelGateway

logger = logging.getLogger(__name__)

ToolTransform = Callable[[list[ToolSpec]], list[ToolSpec]]

MCP_PATH = "/mcp"
# Trajectory misc key under which an agent's tool calls are reported (see tool_call_records).
TOOL_CALLS_MISC_KEY = "tool_calls"
# _meta key of a tool result carrying platoon's signals (finished, reward, metadata, event_id).
RESULT_META_KEY = "platoon"
RESULT_SUMMARY_CHARS = 2000

SUPPORTED_PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25")
LATEST_PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[-1]
SERVER_INFO = {"name": "platoon-gateway", "version": "1"}
KEEPALIVE_SECONDS = 10.0

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
AGENT_ERROR = -32001  # implementation-defined: the agent key is missing, unknown or revoked
CANCELLED_NOTIFICATION = "notifications/cancelled"

_PROTOCOL_VERSION_HEADER = "MCP-Protocol-Version"
_SSE_HEADERS = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
_PASSTHROUGH_BLOCKS = ("audio", "resource", "resource_link")
_STRIPPED_SCHEMA_PREFIX = "x-platoon-"


class UnknownToolError(LookupError):
    """The agent sees no tool by this name: none is bound, its access hides it, or the env has none."""


# --------------------------------------------------------------------------- providers and records


@dataclass(eq=False)
class AgentTools:
    """An agent's tool provider: an env session's tools, with listings rewritten by ``transform``.

    An agent and the children that inherit it share one provider, and the session's listing is fetched once.
    A session that orders its own calls (``manages_concurrency``: ORS and Harbor sessions queue them on their
    live env's gate, :mod:`platoon.envs.gate`) is called directly, so every agent on a live env waits in the
    same queue whatever its provider.  Any other session is called one call at a time per provider: env
    sessions are not assumed to be safe for concurrent calls.
    """

    session: EnvSession
    transform: ToolTransform | None = None
    _specs: list[ToolSpec] | None = field(default=None, init=False, repr=False)
    _listing_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _call_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def specs(self) -> list[ToolSpec]:
        """Every tool of the session, unfiltered and untransformed."""
        if self._specs is None:
            async with self._listing_lock:
                if self._specs is None:
                    self._specs = list(await _session_tools(self.session))
        return self._specs

    async def visible(self, access: ToolAccess) -> list[ToolSpec]:
        """The tools an agent with ``access`` sees, after the transform (which gets copies it may change)."""
        tools = filter_tools(await self.specs(), access)
        if self.transform is not None:
            tools = list(self.transform([_copy(tool) for tool in tools]))
        return tools

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if getattr(self.session, "manages_concurrency", False):
            return await self.session.call_tool(name, arguments)
        async with self._call_lock:
            return await self.session.call_tool(name, arguments)


class ToolCallRecord(TypedDict):
    """One tool call of an agent, JSON-ready, as trajectory misc reports it under ``"tool_calls"``.

    ``after_call_id`` is the agent's last model call when the tool call started: the turn that asked for it,
    when the harness calls tools between model calls (parallel calls of one turn share it).
    """

    event_id: str
    name: str
    arguments: dict[str, Any]
    text: str  # the result's text, cut to RESULT_SUMMARY_CHARS
    is_error: bool
    finished: bool
    reward: float | None
    after_call_id: str | None
    started_at: float  # time.time()
    ended_at: float


def bind_agent_tools(
    gateway: ModelGateway,
    agent: AgentSession,
    session: EnvSession,
    access: ToolAccess = "full",
    *,
    transform: ToolTransform | None = None,
) -> AgentTools:
    """Serve ``session``'s tools to ``agent``, and to the children it opens afterwards, with ``access``.

    Rebinding replaces what the agent inherited.  ``transform`` rewrites listings and must keep tool names (it
    may drop tools, which then cannot be called); None keeps the transform the agent already has, so a child
    rebound to a forked session keeps its parent's.  Rebinding to the same session with the same transform
    only changes the access level and keeps the shared provider.
    """
    if access not in get_args(ToolAccess):
        raise ValueError(f"Unknown tool access {access!r}; expected one of {get_args(ToolAccess)}")
    if gateway.authenticate(agent.secret) is not agent:
        raise UnknownAgentError(f"Agent {agent.agent_id} does not belong to this gateway")
    current = agent.tools
    if transform is None and current is not None:
        transform = current.transform
    if current is None or current.session is not session or current.transform is not transform:
        current = AgentTools(session=session, transform=transform)
    agent.tools, agent.tool_access = current, access
    return current


def filter_tools(tools: Sequence[ToolSpec], access: ToolAccess) -> list[ToolSpec]:
    """The tools ``access`` allows, by the env's ``terminal`` and ``read_only`` hints (absent: False)."""
    if access == "full":
        return list(tools)
    if access == "no_terminal":
        return [tool for tool in tools if not getattr(tool, "terminal", False)]
    if access == "read_only":
        return [tool for tool in tools if getattr(tool, "read_only", False)]
    raise ValueError(f"Unknown tool access {access!r}; expected one of {get_args(ToolAccess)}")


async def list_agent_tools(agent: AgentSession) -> list[ToolSpec]:
    """The tools ``agent`` sees: none until bound, then filtered by its access and transformed."""
    if agent.tools is None:
        return []
    return await agent.tools.visible(agent.tool_access)


async def call_agent_tool(agent: AgentSession, name: str, arguments: Mapping[str, Any]) -> ToolResult:
    """Call one of the agent's visible tools and record the call on the agent.

    A tool that raises becomes an ``is_error`` result, as MCP reports tool execution errors.  Raises
    :class:`UnknownToolError` for a tool the agent does not see and ``AgentClosedError`` for a closed agent.
    """
    result, _ = await _call_and_record(agent, name, arguments)
    return result


def tool_call_records(agent: AgentSession) -> list[dict[str, Any]]:
    """Copies of the agent's tool call records, in call order (for trajectory misc under ``"tool_calls"``)."""
    return [dict(record) for record in getattr(agent, "tool_calls", ())]


def mcp_url(base_url: str, secret: str | None = None) -> str:
    """The MCP endpoint of a gateway served at ``base_url`` (``http://host:port``).

    Without ``secret`` clients authenticate with ``Authorization: Bearer <agent key>``, which keeps the key
    out of process listings and config files; with it the key is in the path, for clients that cannot set
    headers.
    """
    prefix = f"/s/{secret}" if secret else ""
    return f"{base_url.rstrip('/')}{prefix}{MCP_PATH}"


def _copy(tool: ToolSpec) -> ToolSpec:
    try:
        return copy.deepcopy(tool)
    except Exception:  # e.g. a mappingproxy in meta: a shallow copy still protects the cached spec's fields
        return copy.copy(tool)


async def _session_tools(session: EnvSession) -> list[ToolSpec]:
    tools = getattr(session, "tools", None)
    if not callable(tools):
        tools = session.list_tools  # type: ignore[attr-defined]
    return list(await tools())


async def _call_and_record(
    agent: AgentSession, name: str, arguments: Mapping[str, Any]
) -> tuple[ToolResult, ToolCallRecord | None]:
    if agent.closed:
        raise AgentClosedError(f"Agent {agent.agent_id} is closed")
    provider = agent.tools
    if provider is None or name not in {tool.name for tool in await provider.visible(agent.tool_access)}:
        raise UnknownToolError(f"Unknown tool: {name}")
    arguments = dict(arguments)
    after_call_id = agent.calls[-1].call_id if agent.calls else None
    started_at = time.time()
    try:
        with on_behalf_of(agent.agent_id):  # the agent takes its own turn on a live env, also on a shared session
            result = await provider.call(name, arguments)
    except asyncio.CancelledError:
        _record(agent, name, arguments, ToolResult.text("cancelled", is_error=True), started_at, after_call_id)
        raise
    except Exception as exc:
        logger.warning("Tool %r of agent %s raised", name, agent.agent_id, exc_info=True)
        result = ToolResult.text(f"Tool {name!r} failed: {type(exc).__name__}: {exc}", is_error=True)
    return result, _record(agent, name, arguments, result, started_at, after_call_id)


def _record(
    agent: AgentSession,
    name: str,
    arguments: dict[str, Any],
    result: ToolResult,
    started_at: float,
    after_call_id: str | None,
) -> ToolCallRecord | None:
    """Append the call to the agent's record; an agent closed meanwhile records nothing, as for model calls."""
    if agent.closed:
        return None
    record = ToolCallRecord(
        event_id=f"tool-{uuid.uuid4().hex}",
        name=name,
        arguments=_json_ready(arguments),
        text=result_text(result)[:RESULT_SUMMARY_CHARS],
        is_error=bool(result.is_error),
        finished=bool(result.finished),
        reward=result.reward,
        after_call_id=after_call_id,
        started_at=started_at,
        ended_at=time.time(),
    )
    agent.tool_calls.append(record)
    return record


# --------------------------------------------------------------------------- MCP shapes


def result_text(result: ToolResult) -> str:
    """The result's text blocks joined by newlines; other blocks appear as ``[<type>]`` markers."""
    parts: list[str] = []
    for block in result.content:
        if isinstance(block, Mapping) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
        elif isinstance(block, Mapping):
            parts.append(f"[{block.get('type', 'content')}]")
        else:
            parts.append(str(block))
    return "\n".join(parts)


def to_mcp_tool(tool: ToolSpec) -> dict[str, Any]:
    """A ``ToolSpec`` as an MCP ``Tool``: ``meta`` becomes ``_meta`` and ``read_only`` the ``readOnlyHint``.

    ``x-platoon-*`` schema keys are platoon's own annotations and are stripped, so the model never sees them.
    """
    schema = {
        key: value for key, value in (tool.input_schema or {}).items() if not key.startswith(_STRIPPED_SCHEMA_PREFIX)
    }
    schema.setdefault("type", "object")
    listed: dict[str, Any] = {"name": tool.name, "description": tool.description or "", "inputSchema": schema}
    if getattr(tool, "read_only", False):
        listed["annotations"] = {"readOnlyHint": True}
    meta = getattr(tool, "meta", None)
    if meta:
        listed["_meta"] = _json_ready(meta)
    return listed


def to_call_tool_result(result: ToolResult, *, event_id: str | None = None) -> dict[str, Any]:
    """A ``ToolResult`` as an MCP ``CallToolResult``; platoon's signals go in ``_meta["platoon"]``."""
    signals: dict[str, Any] = {"finished": bool(result.finished), "reward": result.reward}
    metadata = getattr(result, "metadata", None)
    if metadata:
        signals["metadata"] = _json_ready(metadata)
    if event_id is not None:
        signals["event_id"] = event_id
    return {
        "content": [_content_block(block) for block in result.content],
        "isError": bool(result.is_error),
        "_meta": {RESULT_META_KEY: signals},
    }


def _content_block(block: Any) -> dict[str, Any]:
    """An env content block as an MCP content block; anything MCP has no block for becomes JSON text."""
    if not isinstance(block, Mapping):
        return {"type": "text", "text": str(block)}
    kind = block.get("type")
    if kind == "text":
        return {"type": "text", "text": str(block.get("text", ""))}
    if kind == "image" and isinstance(block.get("data"), str):
        mime_type = block.get("mimeType") or block.get("mime_type") or "image/png"
        return {"type": "image", "data": block["data"], "mimeType": mime_type}
    if kind in _PASSTHROUGH_BLOCKS:
        return _json_ready(block)
    return {"type": "text", "text": json.dumps(_json_ready(block))}


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value, default=repr))


# --------------------------------------------------------------------------- the endpoint


class JsonRpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class _CancelledByClient(Exception):
    """The client sent ``notifications/cancelled`` for the request: it gets no response."""


@dataclass(eq=False)
class _RunningCall:
    task: asyncio.Task[dict[str, Any]]
    cancelled: bool = False  # by the client's notifications/cancelled


class McpHandler:
    """The ``/mcp`` route of a ``GatewayServer``: MCP streamable HTTP over the authenticated agent's tools."""

    def __init__(
        self, authenticate: Callable[[web.Request], AgentSession], *, keepalive_seconds: float = KEEPALIVE_SECONDS
    ) -> None:
        self._authenticate = authenticate
        self.keepalive_seconds = keepalive_seconds
        # Running tools/call requests by (agent id, request id), for notifications/cancelled.  Request ids are
        # unique per client session only, and the endpoint has no sessions: two clients of one agent may use one
        # id at once, so a cancellation that matches more than one call cancels none.
        self._calls: dict[tuple[str, str], list[_RunningCall]] = {}

    async def handle(self, request: web.Request) -> web.StreamResponse:
        if request.method != "POST":
            # No session ids are issued, so there is no server-initiated stream (GET) and no session to end.
            return web.Response(status=405, headers={"Allow": "POST"})
        try:
            agent = self._authenticate(request)
        except GatewayError as exc:
            status = 410 if isinstance(exc, AgentClosedError) else 401
            return web.json_response(_error_message(None, AGENT_ERROR, str(exc)), status=status)
        version = request.headers.get(_PROTOCOL_VERSION_HEADER)
        if version is not None and version not in SUPPORTED_PROTOCOL_VERSIONS:
            message = f"Unsupported MCP protocol version {version!r}; supported: {list(SUPPORTED_PROTOCOL_VERSIONS)}"
            return web.json_response(_error_message(None, INVALID_REQUEST, message), status=400)
        try:
            body = json.loads(await request.read())
        except ValueError as exc:
            return web.json_response(_error_message(None, PARSE_ERROR, f"Parse error: {exc}"), status=400)
        if isinstance(body, list):  # a JSON-RPC batch (protocol 2025-03-26)
            if not body:
                return web.json_response(_error_message(None, INVALID_REQUEST, "Empty batch"), status=400)
            replies = [reply for reply in [await self._reply(agent, message) for message in body] if reply is not None]
            return web.json_response(replies) if replies else web.Response(status=202)
        if _is_request(body) and body.get("method") == "tools/call" and _accepts(request, "text/event-stream"):
            return await self._stream(request, agent, body)
        reply = await self._reply(agent, body)
        if reply is None:
            return web.Response(status=202)  # notifications and client responses get no reply
        status = 400 if reply.get("id") is None and "error" in reply else 200
        return web.json_response(reply, status=status)

    async def _reply(self, agent: AgentSession, message: Any) -> dict[str, Any] | None:
        """The JSON-RPC reply to one message; None for a notification or a client response."""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _error_message(None, INVALID_REQUEST, "Invalid Request: expected a JSON-RPC 2.0 message")
        if "method" not in message:
            if "result" in message or "error" in message:
                return None  # a response to a server request; this server sends none
            return _error_message(message.get("id"), INVALID_REQUEST, "Invalid Request: no method")
        if not _is_request(message):
            if message.get("method") == CANCELLED_NOTIFICATION:
                self._cancel(agent, message.get("params"))
            return None  # notifications/initialized, notifications/cancelled, ...
        request_id = message["id"]
        try:
            if message.get("method") == "tools/call":
                result = await self._cancellable(agent, request_id, message)
            else:
                result = await self._dispatch(agent, message)
        except _CancelledByClient:
            return None  # MCP: the receiver of a cancellation sends no response for the cancelled request
        except JsonRpcError as exc:
            return _error_message(request_id, exc.code, exc.message)
        except GatewayError as exc:
            return _error_message(request_id, AGENT_ERROR, str(exc))
        except Exception as exc:
            logger.exception("MCP %s failed for agent %s", message.get("method"), agent.agent_id)
            return _error_message(request_id, INTERNAL_ERROR, f"Internal error: {type(exc).__name__}: {exc}")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    async def _cancellable(self, agent: AgentSession, request_id: Any, message: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a ``tools/call`` as a task that ``notifications/cancelled`` can cancel."""
        key = (agent.agent_id, _request_key(request_id))
        call = _RunningCall(asyncio.ensure_future(self._dispatch(agent, message)))
        self._calls.setdefault(key, []).append(call)
        try:
            return await call.task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if call.cancelled and not (current is not None and current.cancelling()):
                raise _CancelledByClient from None
            raise
        finally:
            calls = self._calls.get(key, [])
            if call in calls:
                calls.remove(call)
            if not calls:
                self._calls.pop(key, None)

    def _cancel(self, agent: AgentSession, params: Any) -> None:
        """Cancel the agent's running ``tools/call`` with ``params.requestId``: queued for the env, it never runs;
        running, it finishes in the env (``platoon.envs.gate``) but its response is dropped."""
        if not isinstance(params, dict) or "requestId" not in params:
            return
        calls = self._calls.get((agent.agent_id, _request_key(params["requestId"])), [])
        if len(calls) != 1 or calls[0].task.done():
            return  # finished, unknown, or ambiguous between two clients of the agent
        call = calls[0]
        call.cancelled = True
        call.task.cancel()
        logger.info(
            "MCP client cancelled tools/call %s of agent %s: %s",
            params["requestId"],
            agent.agent_id,
            params.get("reason") or "no reason given",
        )

    async def _dispatch(self, agent: AgentSession, message: dict[str, Any]) -> dict[str, Any]:
        method, params = message["method"], message.get("params")
        if params is None:
            params = {}
        if not isinstance(method, str) or not isinstance(params, dict):
            raise JsonRpcError(INVALID_REQUEST, "Invalid Request: method must be a string and params an object")
        if method == "initialize":
            requested = params.get("protocolVersion")
            version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else LATEST_PROTOCOL_VERSION
            return {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
            }
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": [to_mcp_tool(tool) for tool in await list_agent_tools(agent)]}
        if method == "tools/call":
            name, arguments = params.get("name"), params.get("arguments")
            if arguments is None:
                arguments = {}
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise JsonRpcError(INVALID_PARAMS, "tools/call needs a tool name and an arguments object")
            try:
                result, record = await _call_and_record(agent, name, arguments)
            except UnknownToolError as exc:
                raise JsonRpcError(INVALID_PARAMS, str(exc)) from None
            return to_call_tool_result(result, event_id=record["event_id"] if record else None)
        raise JsonRpcError(METHOD_NOT_FOUND, f"Method not found: {method}")

    async def _stream(self, request: web.Request, agent: AgentSession, message: dict[str, Any]) -> web.StreamResponse:
        """Answer one ``tools/call`` over SSE: keepalives while the tool runs, then the reply."""
        params = message.get("params")
        meta = params.get("_meta") if isinstance(params, dict) else None
        progress_token = meta.get("progressToken") if isinstance(meta, dict) else None
        stream = web.StreamResponse(headers=_SSE_HEADERS)
        task = asyncio.ensure_future(self._reply(agent, message))
        try:
            await stream.prepare(request)
            started = time.monotonic()
            while not task.done():
                finished, _ = await asyncio.wait({task}, timeout=self.keepalive_seconds)
                if finished:
                    break
                if progress_token is None:
                    await stream.write(b": keepalive\n\n")
                else:
                    progress = {"progressToken": progress_token, "progress": round(time.monotonic() - started, 3)}
                    notification = {"jsonrpc": "2.0", "method": "notifications/progress", "params": progress}
                    await stream.write(_sse(notification))
            reply = task.result()
            if reply is not None:  # None: the client cancelled the request, which gets no response
                await stream.write(_sse(reply))
            await stream.write_eof()
        except ConnectionResetError:
            logger.debug("MCP client disconnected during a tool call: agent=%s", agent.agent_id)
        finally:
            task.cancel()
        return stream


def _is_request(message: Any) -> bool:
    return isinstance(message, dict) and "method" in message and message.get("id") is not None


def _request_key(request_id: Any) -> str:
    """A JSON-RPC id as a key that keeps 1 and "1" apart."""
    try:
        return json.dumps(request_id, sort_keys=True)
    except (TypeError, ValueError):
        return repr(request_id)


def _accepts(request: web.Request, media_type: str) -> bool:
    return media_type in request.headers.get("Accept", "")


def _error_message(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _sse(message: Mapping[str, Any]) -> bytes:
    return f"event: message\ndata: {json.dumps(message)}\n\n".encode()
