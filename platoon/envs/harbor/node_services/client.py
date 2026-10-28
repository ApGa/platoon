"""How a session reaches a node service: a scope of it for its rollout (:func:`open_scope`, :class:`NodeScope`), and
the service's state on this node (:func:`ping`), through its supervisor's control socket
(:mod:`platoon.envs.harbor.node_services.protocol`), found in the service's directory on the node: every process that
declares the same service finds the same supervisor.

A scope holds its connection to the supervisor for its life: the supervisor closes the scope when the connection
closes, so a rollout that dies takes its scope along.  :meth:`NodeScope.status` asks the supervisor whether the
rollout may trust what the service answered it.  An open that finds no supervisor listening while one is on the node
(it starts, or waits to take the service over from another run's that stops), or a run that started one (it starts
another), tries again until it is up, within its wait; a service that no supervisor serves on this node, or that
refused, raises :class:`NodeServiceUnavailable`.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from platoon.envs.harbor.node_services.protocol import (
    CONTROL_SOCKET,
    MAX_LINE_BYTES,
    PRESENCE,
    ProtocolError,
    receive,
    send,
)
from platoon.envs.harbor.node_services.spec import NodeServiceSpec, node_dir

#: How long a probe of a scope waits for the supervisor's answer, at most (the supervisor's own checks are bounded by
#: the service's health check).
STATUS_TIMEOUT_S = 300.0
#: How long closing a scope waits for the supervisor to say it is gone.
CLOSE_TIMEOUT_S = 10.0
#: How often an open tries again while a supervisor of the service on the node is not listening yet.
_RETRY_S = 0.2


class NodeServiceUnavailable(RuntimeError):
    """A node service that no supervisor serves on this node, or whose supervisor refused a scope."""


class _TryAgain(Exception):
    """An open that a supervisor on the node may answer later."""


class NodeScope:
    """A rollout's scope of a node service: ``socket``, the front's socket that reaches the service, opened on the
    start ``generation`` of it; open until :meth:`close`."""

    def __init__(
        self,
        spec: NodeServiceSpec,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        answer: Mapping[str, Any],
    ) -> None:
        self.spec = spec
        self.socket = str(answer["socket"])
        self.generation = int(answer["generation"])
        self.served: list[int] = []  # the starts that answered it, as its last status said
        self._reader, self._writer = reader, writer
        self._lock = asyncio.Lock()
        self._closed = False

    async def status(self) -> tuple[bool, str]:
        """Whether the scope's rollout may trust what the service answered it (no start that it used failed, and the
        front served each of its requests), and what failed when it may not; a supervisor that is gone, or does not
        answer within :data:`STATUS_TIMEOUT_S`, is down.  It notes the starts that answered the scope (``served``)."""

        if self._closed:
            return False, "the scope is closed"
        async with self._lock:
            try:
                async with asyncio.timeout(STATUS_TIMEOUT_S):
                    await send(self._writer, {"op": "status"})
                    answer = await receive(self._reader)
            except TimeoutError:
                return False, f"the supervisor of {self.spec.name} did not answer within {STATUS_TIMEOUT_S:g} s"
            except (ProtocolError, OSError) as exc:
                return False, f"the supervisor of {self.spec.name} is gone: {exc}"
        if not answer.get("ok"):
            return False, str(answer.get("error") or answer)
        served = answer.get("served")
        if isinstance(served, list):
            self.served = [int(generation) for generation in served if isinstance(generation, int)]
        return bool(answer.get("up")), str(answer.get("detail") or "")

    async def close(self) -> None:
        """Close the scope, and wait (a little) until the supervisor has: its socket is gone then; idempotent."""

        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(OSError, ProtocolError, TimeoutError):
            async with self._lock, asyncio.timeout(CLOSE_TIMEOUT_S):
                await send(self._writer, {"op": "close"})
                await receive(self._reader)
        self._writer.close()
        with contextlib.suppress(OSError, ConnectionError):
            await self._writer.wait_closed()


async def open_scope(spec: NodeServiceSpec, values: Mapping[str, str], *, wait_s: float = 600.0) -> NodeScope:
    """A new scope of ``spec``'s service with ``values``, once the service is up (waiting up to ``wait_s``, also while
    a supervisor on the node starts listening: the module docstring); :class:`NodeServiceUnavailable` when no
    supervisor serves it here, or it refuses."""

    deadline = time.monotonic() + wait_s
    while True:
        try:
            return await _open(spec, values, max(0.0, deadline - time.monotonic()))
        except _TryAgain as exc:
            if time.monotonic() >= deadline or not supervisor_present(spec):
                raise NodeServiceUnavailable(str(exc)) from exc.__cause__
        await asyncio.sleep(_RETRY_S)


async def _open(spec: NodeServiceSpec, values: Mapping[str, str], wait_s: float) -> NodeScope:
    try:
        reader, writer = await _connect(spec)
    except NodeServiceUnavailable as exc:
        raise _TryAgain(str(exc)) from exc
    try:
        await send(writer, {"op": "open", "scope": dict(values), "wait_s": wait_s})
        answer = await receive(reader)
    except (ProtocolError, OSError) as exc:
        writer.close()
        raise _TryAgain(f"the supervisor of the node service {spec.name} went away: {exc}") from exc
    except BaseException:
        writer.close()
        raise
    if not answer.get("ok"):
        writer.close()
        if answer.get("retry"):
            raise _TryAgain(f"the node service {spec.name} opened no scope: {answer.get('error')}")
        raise NodeServiceUnavailable(f"the node service {spec.name} opened no scope: {answer.get('error')}")
    return NodeScope(spec, reader, writer, answer)


def supervisor_present(spec: NodeServiceSpec) -> bool:
    """Whether a supervisor of ``spec``'s service runs on this node, owning the service or waiting to, or a run that
    started one does: one holds its presence lock (:data:`~platoon.envs.harbor.node_services.protocol.PRESENCE`)."""

    try:
        with open(node_dir(spec) / PRESENCE, "rb") as presence:
            try:
                fcntl.flock(presence, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            return False
    except OSError:
        return False


async def ping(spec: NodeServiceSpec) -> dict[str, Any]:
    """The service's state on this node, as its supervisor tells it (``state``, ``generation``, ``detail``);
    :class:`NodeServiceUnavailable` when no supervisor serves it here."""

    reader, writer = await _connect(spec)
    try:
        await send(writer, {"op": "ping"})
        return await receive(reader)
    except (ProtocolError, OSError) as exc:
        raise NodeServiceUnavailable(f"the supervisor of the node service {spec.name} went away: {exc}") from exc
    finally:
        writer.close()


def control_socket(spec: NodeServiceSpec) -> Path:
    """Where ``spec``'s supervisor listens on this node."""

    return node_dir(spec) / CONTROL_SOCKET


async def _connect(spec: NodeServiceSpec) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    path = control_socket(spec)
    try:
        return await asyncio.open_unix_connection(str(path), limit=MAX_LINE_BYTES)
    except (FileNotFoundError, ConnectionRefusedError) as exc:
        raise NodeServiceUnavailable(
            f"the node service {spec.name} is not running on this node (no supervisor listens at {path}): platoon "
            "train and platoon eval start the node services their environments declare before any rollout"
        ) from exc
    except OSError as exc:
        raise NodeServiceUnavailable(f"the node service {spec.name} cannot be reached at {path}: {exc}") from exc


__all__ = [
    "CLOSE_TIMEOUT_S",
    "STATUS_TIMEOUT_S",
    "NodeScope",
    "NodeServiceUnavailable",
    "control_socket",
    "open_scope",
    "ping",
    "supervisor_present",
]
