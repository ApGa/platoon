"""A node service's front: the only way into the service, one Unix socket per rollout's **scope**.

A rollout reaches a node service through its scope's socket alone (a pod relays it to the service's port; a stack on
this host's network relays a port of the host to it), never through the service's own socket.  For each request the
front:

- finds the scope by the socket it came through, and answers 404 to a request whose method and path the declaration's
  ``front.routes`` do not list (``POST /evaluate``), whatever else it holds: no other route of the service, no admin
  route, is reachable from a rollout;
- answers 413 to a body larger than ``front.max_request_bytes``, and 503 once the scope is **down**: its rollout can
  no longer trust what the service answers it (below), so that it learns that its node service failed;
- lets ``front.concurrency`` requests of a scope through at a time (the others wait), so one rollout cannot hold more
  of the service than that;
- waits, while the service starts again after a failure, for the start that replaces it (at most
  :data:`REQUEST_WAIT_S`), and forwards the request there: a rollout that did not use the start that failed goes on
  with the next one;
- forwards the request to the service's socket with the path alone (no query string), its body, its ``Content-Type``,
  the token the service checks (``Authorization: Bearer``), and the scope's values as JSON in ``X-Platoon-Scope``
  (``{"problem_id": "frontiersmith_81"}``): what the rollout may act on, which the service enforces.  Any
  ``Authorization`` or ``X-Platoon-*`` header the rollout sends is dropped, so the rollout can neither name another
  scope nor reach the service's other routes with the token, which it never sees;
- answers what the service answers (its status, ``Content-Type`` and body, at most 64 MiB), or 502 when the service
  cannot be reached.  A request that never reached the service because its socket's queue of connections was full
  (more rollouts asking at once than the service accepts) is tried again a few times first.

**A scope that is down** stays down: one that the front could not serve a request of (the service could not be
reached, or answered too much, or failed for good), and one that used a start of the service that then failed: a
request of it was in flight when the start failed, or came after the start last passed its health check
(:meth:`Front.start_failed`).  Its rollout is ``invalid``; a scope that did neither is not, and its next request goes
to the next start.  A scope also notes the starts that answered it (``served``).

A scope's values come from the task (an overlay's ``scope`` of the service it serves, ``${NAME}`` read from the task's
variables), so a rollout's scope names only its own task.
"""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import secrets
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
from aiohttp import web

from platoon.envs.harbor.node_services.spec import NodeServiceSpec

logger = logging.getLogger(__name__)

#: The header that carries a scope's values to the service.
SCOPE_HEADER = "X-Platoon-Scope"
#: The directory of the scopes' sockets, in the service's directory on the node.
SCOPES_DIR = "scopes"
#: How long a request waits, at most, for a start of the service that serves, while the service starts again.
REQUEST_WAIT_S = 600.0
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_CONNECT_TIMEOUT_S = 10.0
#: How often the front tries a request that the service's socket did not take (its queue full), and how long it waits
#: before the first try again (then twice as long each time).
_FORWARD_ATTEMPTS = 6
_FORWARD_BACKOFF_S = 0.05


@dataclass
class Scope:
    """One rollout's way into the service: its socket, its values, and the start of the service it was opened on;
    the requests it has in flight (``active``), when it last began or ended one (``last_active``, monotonic), why it is
    down (None while it is not: the module docstring says when it is), and the starts that answered it (``served``)."""

    id: str
    socket: Path
    values: Mapping[str, str]
    generation: int
    site: web.UnixSite
    gate: asyncio.Semaphore = field(repr=False)
    active: int = 0
    last_active: float = float("-inf")
    down: str | None = None
    served: set[int] = field(default_factory=set)


class Front:
    """The scopes of ``spec``'s service (the module docstring says what each request gets), in ``directory`` (the
    service's directory on the node), forwarded to its ``backend`` socket with ``token``; ``serving(wait_s)`` gives the
    start of the service that serves, once one does (waiting up to ``wait_s``), or None when none will."""

    def __init__(
        self,
        spec: NodeServiceSpec,
        directory: Path,
        *,
        backend: Path,
        token: str,
        serving: Callable[[float], Awaitable[int | None]],
        wait_s: float = REQUEST_WAIT_S,
    ) -> None:
        self.spec = spec
        self.directory = directory / SCOPES_DIR
        self._backend = backend
        self._token = token
        self._serving = serving
        self._wait_s = wait_s
        self._scopes: dict[str, Scope] = {}  # by socket path
        self._runner: web.AppRunner | None = None
        self._http: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        self.directory.mkdir(mode=0o700, exist_ok=True)
        app = web.Application(client_max_size=self.spec.front.max_request_bytes)
        app.router.add_route("*", "/{tail:.*}", self._handle)
        # A request whose rollout went away is dropped, and its place among the scope's requests with it.
        self._runner = web.AppRunner(app, access_log=None, handler_cancellation=True)
        await self._runner.setup()
        connector = aiohttp.UnixConnector(path=str(self._backend), limit=0)
        self._http = aiohttp.ClientSession(connector=connector)

    async def open(self, values: Mapping[str, str], *, generation: int) -> Scope:
        """A new scope with ``values``, on the service's start ``generation``: its socket listens once this returns."""

        assert self._runner is not None, "the front has not started"
        scope_id = secrets.token_hex(8)
        socket = self.directory / f"{scope_id}.sock"
        site = web.UnixSite(self._runner, str(socket))
        await site.start()
        scope = Scope(
            id=scope_id,
            socket=socket,
            values=dict(values),
            generation=generation,
            site=site,
            gate=asyncio.Semaphore(self.spec.front.concurrency),
        )
        self._scopes[str(socket)] = scope
        return scope

    async def close_scope(self, scope: Scope) -> None:
        """Stop serving ``scope``: its socket goes, and so do the requests it is serving."""

        if self._scopes.pop(str(scope.socket), None) is None:
            return
        try:
            await scope.site.stop()
        finally:
            scope.socket.unlink(missing_ok=True)

    def start_failed(self, detail: str, *, since: float) -> int:
        """A start of the service failed (``detail`` says how): every scope with a request in flight, or one that began
        or ended at ``since`` (when the start last passed its health check, monotonic) or after, is down for good; the
        others go on with the next start.  The number of scopes it made down."""

        down = 0
        for scope in self._scopes.values():
            if scope.down is None and (scope.active or scope.last_active >= since):
                scope.down = detail
                down += 1
        return down

    async def close(self) -> None:
        """Stop serving every scope, once the requests in flight are answered."""

        for scope in list(self._scopes.values()):
            await self.close_scope(scope)
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        if self._http is not None:
            await self._http.close()
            self._http = None

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        transport = request.transport
        scope = self._scopes.get(str(transport.get_extra_info("sockname"))) if transport is not None else None
        if scope is None or not self.spec.allows(request.method, request.path):
            return web.json_response({"error": "not found"}, status=404)
        if request.content_length is not None and request.content_length > self.spec.front.max_request_bytes:
            return web.json_response({"error": "request too large"}, status=413)
        try:
            body = await request.read()
        except web.HTTPRequestEntityTooLarge:
            return web.json_response({"error": "request too large"}, status=413)
        async with scope.gate:
            generation = await self._serving(self._wait_s) if scope.down is None else None
            if scope.down is None and generation is None:
                scope.down = f"the node service {self.spec.name} serves no more"
            if scope.down is not None or generation is None:
                return web.json_response({"error": scope.down}, status=503)
            scope.active += 1
            scope.last_active = time.monotonic()
            scope.served.add(generation)
            try:
                answer = await self._forward(request, scope, body)
            finally:
                scope.active -= 1
                scope.last_active = time.monotonic()
            if answer is None:  # the front could not serve it: the scope is down
                return web.json_response({"error": scope.down}, status=502)
            if scope.down is not None:  # the start failed while it answered: the answer cannot be trusted
                return web.json_response({"error": scope.down}, status=503)
            return answer

    async def _forward(self, request: web.Request, scope: Scope, body: bytes) -> web.StreamResponse | None:
        """The service's answer to ``request``; None, and the scope down, when there is none."""

        assert self._http is not None
        headers = {
            "Authorization": f"Bearer {self._token}",
            SCOPE_HEADER: json.dumps(dict(scope.values), separators=(",", ":")),
        }
        if content_type := request.headers.get("Content-Type"):
            headers["Content-Type"] = content_type
        for attempt in range(_FORWARD_ATTEMPTS):
            try:
                async with self._http.request(
                    request.method,
                    f"http://node-service{request.path}",
                    data=body or None,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=None, sock_connect=_CONNECT_TIMEOUT_S),
                ) as answer:
                    content = await answer.content.read(_MAX_RESPONSE_BYTES + 1)
                    if len(content) > _MAX_RESPONSE_BYTES:
                        scope.down = f"the node service {self.spec.name} answered more than {_MAX_RESPONSE_BYTES} bytes"
                        return None
                    return web.Response(
                        status=answer.status,
                        body=content,
                        headers={"Content-Type": answer.headers.get("Content-Type", "application/octet-stream")},
                    )
            except (aiohttp.ClientError, OSError) as exc:
                if _queue_full(exc) and attempt + 1 < _FORWARD_ATTEMPTS:
                    await asyncio.sleep(_FORWARD_BACKOFF_S * 2**attempt)
                    continue
                logger.warning("The node service %s cannot be reached: %s", self.spec.name, exc)
                break
        scope.down = f"the node service {self.spec.name} could not be reached"
        return None


def _queue_full(exc: BaseException | None) -> bool:
    """Whether ``exc`` says that the service's socket did not take the connection, its queue full, so the request never
    reached the service: a Unix socket refuses it (``EAGAIN``), which asyncio may only report as the request is written
    to a socket it took for connected (``ENOTCONN``)."""

    while exc is not None:
        error = getattr(exc, "os_error", exc)
        if isinstance(error, OSError) and error.errno in (errno.EAGAIN, errno.ENOTCONN):
            return True
        exc = exc.__cause__
    return False


__all__ = ["REQUEST_WAIT_S", "SCOPES_DIR", "SCOPE_HEADER", "Front", "Scope"]
