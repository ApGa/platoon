"""HTTP front of the model gateway: wire dialects plus the minimal agent API.

``GatewayServer(gateway).start()`` serves, on the event loop of the process that holds the gateway (the
rollout worker):

* ``POST /v1/chat/completions``: OpenAI Chat Completions (:mod:`platoon.gateway.dialects.openai_chat`).
* ``POST /v1/responses``: OpenAI Responses for Codex (:mod:`platoon.gateway.dialects.responses`, experimental).
* ``POST /v1/messages`` and ``/v1/messages/count_tokens``: Anthropic Messages for Claude Code
  (:mod:`platoon.gateway.dialects.anthropic`, experimental; Anthropic clients take the base URL without ``/v1``).
* ``stream: true`` is answered with synthetic SSE once the whole completion is sampled, with ``: keepalive``
  comments while sampling so client idle timeouts do not fire during long generations or weight pushes.
* On a hosted gateway (:mod:`platoon.gateway.hosted`), a model call and ``count_tokens`` are forwarded to the base
  URL in their own dialect, and the provider's reply goes back with its status, content type and body (a stream's
  too, once read whole).  A stream's status is sent when its first keepalive is due, so a provider error that comes
  sooner keeps its own status.
* ``GET /v1/models`` and ``GET /healthz``.
* The ``X-Platoon-Call-Kind`` header marks a call ``policy`` (default), ``aux`` or ``judge`` (see
  :mod:`platoon.harness.agent`).  An agent limited to policy calls (``AgentSession.policy_only``, a harness in a task
  sandbox) gets the dialect's permission error (403) for the others, and nothing is sampled.
* ``POST /v1/platoon/agents``: mint a subagent (``{role, goal?}`` -> ``{agent_id, api_key, base_url}``), so a
  cooperative harness gets exact per-agent attribution.  A rollout mints at most ``ModelGateway.max_minted_agents``
  (400 past it), and a child's policy calls count against its parent's call cap.
* ``/mcp``: the agent's env tools over streamable-HTTP MCP (:mod:`platoon.gateway.mcp`, the tool plane).

An agent authenticates with its key as ``Authorization: Bearer``, as ``x-api-key``, or in the path
(``/s/<secret>/v1/...``) for clients that cannot set headers.  Unknown keys get 401, closed agents 410, and a
prompt that overflows the context window gets the dialect's native context-length error, so the harness
condenses instead of retrying.

Besides its loopback TCP port, the server can listen on a Unix socket (:meth:`GatewayServer.start_unix`) in a
private directory (:func:`socket_directory`), which a sandbox without a network reaches through a route
(:mod:`platoon.sandbox.routes`).  A socket served for one rollout takes only that rollout's agent keys: a key of
another rollout gets 401 there.  A server that serves rollouts (``rollout_ids``, :meth:`GatewayServer.serve_rollout`),
as a rollout's ``platoon.harness.rollout.EmbeddedGateway`` serves only its own, takes only those rollouts' agent keys
on every transport, its TCP port included.

An agent with a call cap (``AgentSession.call_cap``: the gateway step budget of a spec with ``[budget] enforce =
"gateway"``) gets, for each ``policy`` call past it, a final assistant message without tool calls
(``ModelGateway.budget_stop``), unsampled and unrecorded, in the dialect's normal response or stream.  The cap
counts the policy calls of the agent and of every agent minted under it, together, those in flight included, also
when many arrive at once, streamed or not.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from pathlib import Path
from typing import Any, TypeVar

from aiohttp import web

from platoon.gateway.core import (
    AgentSession,
    CallKindRefusedError,
    ContextWindowExceededError,
    GatewayError,
    ModelGateway,
    PreparedCall,
    UnknownAgentError,
    check_call_kind,
)
from platoon.gateway.dialects import ERROR_STATUS, Dialect, error_kind, sse
from platoon.gateway.dialects.anthropic import AnthropicMessages
from platoon.gateway.dialects.openai_chat import OpenAIChat
from platoon.gateway.dialects.responses import OpenAIResponses, failed_event
from platoon.gateway.hosted import PATHS, HostedCallError, HostedGateway, Relay, error_message
from platoon.gateway.mcp import MCP_PATH, McpHandler
from platoon.harness.agent import CALL_KIND_HEADER
from platoon.utils.scratch import node_scratch

logger = logging.getLogger(__name__)
_T = TypeVar("_T")

CALL_ID_HEADER = "x-platoon-call-id"
KEEPALIVE_SECONDS = 10.0
# Agent histories with long tool outputs exceed aiohttp's 1 MiB default body limit.
MAX_REQUEST_BYTES = 256 * 1024**2
# How long stop() lets in-flight requests finish before cancelling them.
SHUTDOWN_GRACE_SECONDS = 1.0

_SSE_HEADERS = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
_WILDCARD_HOSTS = ("", "0.0.0.0", "::")
#: The file name of the gateway's socket in its directory (:func:`socket_directory`).
SOCKET_NAME = "gw.sock"
#: What an ``AF_UNIX`` address holds on Linux, its terminating NUL included.
_SUN_PATH_BYTES = 108
#: The status and error type of a call its agent may not make (``CallKindRefusedError``): a permission error.
_PERMISSION_STATUS = 403
_PERMISSION_ERROR_TYPE = "permission_error"


def socket_directory(base: str | os.PathLike[str] | None = None) -> Path:
    """Create a private (0700) directory for one gateway socket, and return it; the caller removes it.

    It is ``<base>/pgw-<uid>/<12 hex>``, under node-local scratch by default
    (:func:`platoon.utils.scratch.node_scratch`): short, so that the socket path stays within an ``AF_UNIX`` address,
    and opaque.  When ``base`` would make the path too long, ``/tmp`` is used instead.  A sandbox binds the directory
    rather than the socket, so the socket may be created after it starts.
    """

    for root in (Path(base or node_scratch()), Path("/tmp")):
        parent = root / f"pgw-{os.getuid()}"
        directory = parent / secrets.token_hex(6)
        if len(os.fsencode(directory / SOCKET_NAME)) < _SUN_PATH_BYTES:
            break
    else:
        raise OSError(f"No directory short enough for a Unix socket: {directory / SOCKET_NAME}")
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = parent.lstat()
    if parent.is_symlink() or not parent.is_dir() or info.st_uid != os.getuid():
        raise PermissionError(f"{parent} must be a directory of this user's")
    if info.st_mode & 0o077:
        parent.chmod(0o700)
    directory.mkdir(mode=0o700)
    return directory


class GatewayServer:
    """An aiohttp server for one :class:`ModelGateway`, embedded in the process that samples.

    By default it takes the key of any open agent of the gateway.  Given ``rollout_ids``, or once told a rollout
    with :meth:`serve_rollout`, it takes only the keys of those rollouts' agents, on every transport, and answers
    any other key with 401; ``rollout_ids=()`` takes no key until the first rollout is served.
    """

    def __init__(
        self,
        gateway: ModelGateway,
        *,
        keepalive_seconds: float = KEEPALIVE_SECONDS,
        rollout_ids: Iterable[str] | None = None,
    ) -> None:
        self.gateway = gateway
        self.keepalive_seconds = keepalive_seconds
        self._chat = OpenAIChat(gateway)
        self._responses = OpenAIResponses(self._chat)
        self._messages = AnthropicMessages()
        self._mcp = McpHandler(self._agent, keepalive_seconds=keepalive_seconds)
        self._runner: web.AppRunner | None = None
        self._base_url: str | None = None
        self._socket_path: str | None = None
        self._socket_rollout: str | None = None
        # The rollouts whose agents' keys the server takes; None: any rollout's.
        self._rollouts: set[str] | None = None if rollout_ids is None else set(rollout_ids)

    def serve_rollout(self, rollout_id: str) -> None:
        """Take the keys of ``rollout_id``'s agents.  From the first call on (or from the start, given
        ``rollout_ids``), the server takes only the keys of the rollouts it serves, on its TCP port and its Unix
        socket alike, so a key of another rollout gets 401 here."""
        if self._rollouts is None:
            self._rollouts = set()
        self._rollouts.add(rollout_id)

    @property
    def base_url(self) -> str:
        """``http://host:port``; OpenAI clients use ``base_url + "/v1"``."""
        if self._base_url is None:
            raise RuntimeError("GatewayServer is not started")
        return self._base_url

    async def start(self, host: str = "0.0.0.0", port: int = 0, *, advertise_host: str | None = None) -> str:
        """Serve on ``host:port`` (port 0: any free port) and return :attr:`base_url`.

        For a wildcard ``host`` the URL names ``advertise_host``, by default the loopback address, which reaches
        the server from this host and from sandboxes that share its network namespace.
        """
        if self._runner is not None:
            raise RuntimeError("GatewayServer is already started")
        runner = web.AppRunner(
            self._make_app(),
            access_log=None,
            handler_cancellation=True,  # a client that disconnects abandons its call: nothing is recorded
            shutdown_timeout=SHUTDOWN_GRACE_SECONDS,
        )
        await runner.setup()
        try:
            await web.TCPSite(runner, host, port).start()
        except BaseException:
            await runner.cleanup()
            raise
        self._runner = runner
        if advertise_host is None:
            advertise_host = "127.0.0.1" if host in _WILDCARD_HOSTS else host
        if ":" in advertise_host:
            advertise_host = f"[{advertise_host}]"
        self._base_url = f"http://{advertise_host}:{runner.addresses[0][1]}"
        return self._base_url

    @property
    def socket_path(self) -> str | None:
        """The Unix socket the server listens on (:meth:`start_unix`), or None."""
        return self._socket_path

    async def start_unix(self, path: str | os.PathLike[str], *, rollout_id: str | None = None) -> str:
        """Also serve on the Unix socket ``path``, after :meth:`start`; returns the path.

        Put the socket in a private directory (:func:`socket_directory`): the directory's mode is what keeps other
        users out.  With ``rollout_id``, a request on the socket must carry the key of an agent of that rollout,
        else it gets 401, so a sandbox of one rollout cannot spend another rollout's keys there.  :meth:`stop`
        removes the socket file.
        """
        if self._runner is None:
            raise RuntimeError("Start the gateway server before its Unix socket")
        if self._socket_path is not None:
            raise RuntimeError(f"The gateway server already listens on {self._socket_path}")
        path = os.fspath(path)
        await web.UnixSite(self._runner, path).start()
        self._socket_path, self._socket_rollout = path, rollout_id
        return path

    async def stop(self) -> None:
        """Stop serving; in-flight requests get a short grace period.  Idempotent."""
        runner, self._runner, self._base_url = self._runner, None, None
        socket_path, self._socket_path, self._socket_rollout = self._socket_path, None, None
        if runner is not None:
            await runner.cleanup()
        if socket_path is not None:
            try:
                os.unlink(socket_path)
            except OSError:
                pass

    def _make_app(self) -> web.Application:
        app = web.Application(client_max_size=MAX_REQUEST_BYTES)
        app.router.add_get("/healthz", self._healthz)
        for prefix in ("", "/s/{secret}"):
            app.router.add_post(f"{prefix}/v1/chat/completions", self._dialect_handler(self._chat))
            app.router.add_post(f"{prefix}/v1/responses", self._dialect_handler(self._responses))
            app.router.add_post(f"{prefix}/v1/messages", self._dialect_handler(self._messages))
            app.router.add_post(f"{prefix}/v1/messages/count_tokens", self._count_tokens)
            app.router.add_get(f"{prefix}/v1/models", self._models)
            app.router.add_post(f"{prefix}/v1/platoon/agents", self._create_agent)
            app.router.add_route("*", f"{prefix}{MCP_PATH}", self._mcp.handle)
        return app

    def _agent(self, request: web.Request) -> AgentSession:
        secret = request.match_info.get("secret") or _presented_key(request)
        if not secret:
            raise UnknownAgentError("Missing API key: send the agent key as a Bearer token")
        agent = self.gateway.authenticate(secret)
        if self._rollouts is not None and agent.rollout_id not in self._rollouts:
            raise UnknownAgentError("This key belongs to another rollout than this gateway server's")
        if self._socket_rollout is not None and agent.rollout_id != self._socket_rollout and _on_unix_socket(request):
            raise UnknownAgentError("This key belongs to another rollout than this gateway socket's")
        return agent

    # ------------------------------------------------------------------ routes

    async def _healthz(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "model": self.gateway.model_name})

    async def _models(self, request: web.Request) -> web.Response:
        try:
            self._agent(request)
        except GatewayError as exc:
            return _error_response(self._chat, exc)
        model = {"id": self.gateway.model_name, "object": "model", "created": 0, "owned_by": "platoon"}
        return web.json_response({"object": "list", "data": [model]})

    async def _create_agent(self, request: web.Request) -> web.Response:
        try:
            parent = self._agent(request)
            body = await _json_body(request)
            role, goal = body.get("role"), body.get("goal")
            if not isinstance(role, str) or not role:
                raise ValueError("'role' must be a non-empty string")
            if goal is not None and not isinstance(goal, str):
                raise ValueError("'goal' must be a string")
            child = self.gateway.mint_subagent(parent, role=role, goal=goal)
        except (GatewayError, ValueError) as exc:
            return _error_response(self._chat, exc)
        # Address the child the way the parent addressed us: same origin, and the path form if it used it.
        prefix = f"/s/{child.secret}" if "secret" in request.match_info else ""
        base_url = f"{request.url.origin()}{prefix}/v1"
        payload = {"agent_id": child.agent_id, "api_key": child.api_key, "base_url": base_url}
        return web.json_response(payload, status=201)

    async def _count_tokens(self, request: web.Request) -> web.Response:
        try:
            agent = self._agent(request)
            body = await _json_body(request)
            if isinstance(self.gateway, HostedGateway):
                relay = await self.gateway.count_tokens(agent, body, request.query_string, request.headers)
                return _relay_response(relay)
            chat_request = self._messages.chat_request(body)
            input_tokens = self.gateway.render(chat_request.messages, chat_request.render).length
        except web.HTTPException:
            raise
        except Exception as exc:
            return _error_response(self._messages, exc)
        return web.json_response({"input_tokens": input_tokens})

    def _dialect_handler(self, dialect: Dialect) -> Callable[[web.Request], Awaitable[web.StreamResponse]]:
        async def handle(request: web.Request) -> web.StreamResponse:
            return await self._call(request, dialect)

        return handle

    async def _call(self, request: web.Request, dialect: Dialect) -> web.StreamResponse:
        """Authenticate, prepare the dialect's request, sample it for the agent and answer in the dialect.

        A call of a kind the agent may not make (``AgentSession.policy_only``) is refused before anything else.
        Rendering errors (an overflowing prompt, an unsupported parameter) are answered before any stream starts.
        """
        try:
            agent = self._agent(request)
            kind = check_call_kind(request.headers.get(CALL_KIND_HEADER) or "policy")
            agent.check_kind(kind)
            body = await _json_body(request)
        except web.HTTPException:
            raise
        except Exception as exc:
            return _error_response(dialect, exc)
        # The step budget holds for a hosted model's calls too: past the cap, the budget stop answers, unforwarded.
        if kind == "policy" and agent.over_call_cap():
            return await self._budget_stop(request, agent, dialect, body)
        if isinstance(self.gateway, HostedGateway):
            return await self._forward(request, agent, dialect, body, kind=kind)
        try:
            call = self.gateway.prepare(dialect.chat_request(body), defaults=agent.sampling_defaults, kind=kind)
        except ContextWindowExceededError as exc:
            agent.context_overflows += 1
            return _error_response(dialect, exc)
        except Exception as exc:
            if error_kind(exc) == "server":
                agent.gateway_errors += 1  # a 5xx: the gateway failed, not the request
            return _error_response(dialect, exc)
        if body.get("stream"):
            return await self._stream(request, agent, call, dialect, body, kind=kind)
        try:
            response = await self.gateway.chat(agent, call, kind=kind)
        except Exception as exc:
            return _error_response(dialect, exc)
        return web.json_response(dialect.response(response, body), headers={CALL_ID_HEADER: response.call_id})

    async def _budget_stop(
        self, request: web.Request, agent: AgentSession, dialect: Dialect, body: dict[str, Any]
    ) -> web.StreamResponse:
        """Answer a policy call past the agent's call cap with the budget stop, streamed if the call asked so."""
        try:
            response = self.gateway.budget_stop(agent)
        except Exception as exc:
            return _error_response(dialect, exc)
        if not body.get("stream"):
            return web.json_response(dialect.response(response, body), headers={CALL_ID_HEADER: response.call_id})
        stream = web.StreamResponse(headers=_SSE_HEADERS)
        try:
            await stream.prepare(request)
            for event in dialect.events(response, body):
                await stream.write(event)
            await stream.write_eof()
        except ConnectionResetError:
            logger.debug("Client disconnected during a budget stop: path=%s", request.path)
        return stream

    async def _stream(
        self,
        request: web.Request,
        agent: AgentSession,
        call: PreparedCall,
        dialect: Dialect,
        body: dict[str, Any],
        *,
        kind: str = "policy",
    ) -> web.StreamResponse:
        """Answer with SSE: keepalive comments while the call samples, then the dialect's events.

        A call that fails once the stream has started is reported as the dialect's error event.
        """
        stream = web.StreamResponse(headers=_SSE_HEADERS)
        task = _start_call(self.gateway.chat(agent, call, kind=kind))
        try:
            await stream.prepare(request)
            while not task.done():
                finished, _ = await asyncio.wait({task}, timeout=self.keepalive_seconds)
                if not finished:
                    await stream.write(b": keepalive\n\n")
            try:
                response = task.result()
            except Exception as exc:
                _log_server_error(exc)
                await stream.write(dialect.error_event(exc))
            else:
                for event in dialect.events(response, body):
                    await stream.write(event)
            await stream.write_eof()
        except ConnectionResetError:
            logger.debug("Client disconnected during an SSE response: path=%s", request.path)
        finally:
            task.cancel()
        return stream

    async def _forward(
        self, request: web.Request, agent: AgentSession, dialect: Dialect, body: dict[str, Any], *, kind: str
    ) -> web.StreamResponse:
        """Forward the request to a hosted gateway's base URL and relay the provider's reply.

        A request that does not stream gets the reply as it came.  A streamed one waits for the whole reply too, as a
        Tinker one waits for the whole completion: a reply that comes before the first keepalive is due is answered
        with its own status, so an early provider error keeps it.  After that the status is 200, a keepalive comment
        follows every ``keepalive_seconds``, and then the provider's stream, or the dialect's error event for a call
        that failed.
        """

        gateway = self.gateway
        assert isinstance(gateway, HostedGateway)
        path = PATHS[dialect.name]
        forward = gateway.forward(agent, dialect.name, path, request.query_string, body, request.headers, kind)
        if not body.get("stream"):
            try:
                return _relay_response(await forward)
            except Exception as exc:
                return _error_response(dialect, exc)
        task = _start_call(forward)
        try:
            await asyncio.wait({task}, timeout=self.keepalive_seconds)
            if task.done():
                try:
                    return _relay_response(task.result())
                except Exception as exc:
                    return _error_response(dialect, exc)
            stream = web.StreamResponse(headers=_SSE_HEADERS)
            try:
                await stream.prepare(request)
                await stream.write(b": keepalive\n\n")
                while not task.done():
                    finished, _ = await asyncio.wait({task}, timeout=self.keepalive_seconds)
                    if not finished:
                        await stream.write(b": keepalive\n\n")
                try:
                    relay = task.result()
                except Exception as exc:
                    _log_server_error(exc)
                    await stream.write(dialect.error_event(exc))
                else:
                    await stream.write(relay.body if relay.ok else _relay_error_event(dialect, relay))
                await stream.write_eof()
            except ConnectionResetError:
                logger.debug("Client disconnected during an SSE response: path=%s", request.path)
            return stream
        finally:
            task.cancel()


def _start_call(call: Coroutine[Any, Any, _T]) -> asyncio.Task[_T]:
    """A task for a streamed call that runs it at once, up to its first suspension, as an awaited call runs: a
    ``policy`` call then counts against its agent's call cap (``AgentSession.call_tally``) before this request's
    handler yields, so the budget check of a call that arrives meanwhile (:meth:`GatewayServer._call`) sees it, however
    many arrive together.  A task started the usual way would count only once the event loop got to it, and every
    request checked before then would pass the cap."""

    return asyncio.eager_task_factory(asyncio.get_running_loop(), call)


def _relay_response(relay: Relay) -> web.Response:
    """A hosted call's reply as the harness gets it: the provider's status, content type, body and ``retry-after``,
    and the recorded call's id."""

    headers = {"Content-Type": relay.content_type, **relay.headers}
    if relay.response is not None:
        headers[CALL_ID_HEADER] = relay.response.call_id
    return web.Response(status=relay.status, body=relay.body, headers=headers)


def _relay_error_event(dialect: Dialect, relay: Relay) -> bytes:
    """A provider's error reply as the dialect's error event, for a stream whose status was already sent: the
    provider's own error when it has the dialect's shape, else the dialect's server error naming the status."""

    body = relay.json()
    error = body.get("error") if isinstance(body, dict) else None
    if dialect.name == "openai_chat" and isinstance(error, dict):
        return sse(body)
    if dialect.name == "anthropic_messages" and isinstance(body, dict) and body.get("type") == "error":
        return sse(body, event="error")
    if dialect.name == "openai_responses" and isinstance(error, dict):
        return failed_event(error)
    return dialect.error_event(HostedCallError(f"{relay.status} {error_message(relay.body)}"))


def _on_unix_socket(request: web.Request) -> bool:
    """Whether the request came in on a Unix socket (its socket name is a path, a TCP one's an address tuple)."""

    transport = request.transport
    return transport is not None and isinstance(transport.get_extra_info("sockname"), (str, bytes))


def _presented_key(request: web.Request) -> str | None:
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    return request.headers.get("x-api-key") or None


async def _json_body(request: web.Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError as exc:
        raise ValueError(f"The request body is not valid JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise ValueError("The request body must be a JSON object")
    return body


def _log_server_error(exc: BaseException) -> None:
    if error_kind(exc) == "server":
        logger.error("Model call failed", exc_info=exc)


def _error_response(dialect: Dialect, exc: BaseException) -> web.Response:
    _log_server_error(exc)
    body = dialect.error(exc)
    if isinstance(exc, CallKindRefusedError):
        # The dialects word it as an invalid request; it is a permission error (403), the type Anthropic gives one.
        error = body.get("error")
        if isinstance(error, dict):
            error["type"] = _PERMISSION_ERROR_TYPE
        return web.json_response(body, status=_PERMISSION_STATUS)
    return web.json_response(body, status=ERROR_STATUS[error_kind(exc)])
