"""The control protocol between a node service's supervisor and the processes that use it: one JSON object per line,
over the supervisor's control socket (:data:`CONTROL_SOCKET` in the service's directory on the node,
:func:`~platoon.envs.harbor.node_services.spec.node_dir`, which only your uid may enter).

- ``{"op": "ping"}`` answers the service's state: ``{"ok": true, "state": "starting" | "up" | "down" | "failed" |
  "stopping", "generation": <n>, "detail": "..."}``, where a generation counts the service's starts on this node.
- ``{"op": "open", "scope": {...}, "wait_s": <s>}`` opens a **scope** of the service for one rollout, once it is up
  (waiting up to ``wait_s``): ``{"ok": true, "socket": "<path>", "generation": <n>}``, a Unix socket of the front that
  lets only the declared routes through, with the scope's values in the ``X-Platoon-Scope`` header.  The scope lasts
  as long as the connection that opened it (one scope per connection), so a rollout that dies takes its scope along.
  A supervisor that is stopping answers ``{"ok": false, "retry": true, ...}``: the next owner of the service may open
  it.
- ``{"op": "status"}``, on that connection, answers whether the scope's rollout may trust what the service answered
  it: ``{"ok": true, "up": <bool>, "detail": "...", "served": [<generation>, ...]}``, down once a start that it used
  failed, or the front could not serve one of its requests (:mod:`platoon.envs.harbor.node_services.front`); ``served``
  names the starts that answered it.
- ``{"op": "close"}`` closes the scope, and answers ``{"ok": true}`` once its socket is gone.

Any request that fails answers ``{"ok": false, "error": "..."}``.  Every supervisor of the service on the node, the
owner and those that wait to take over, holds :data:`PRESENCE` in its directory with a shared lock for its life, and so
does every run that started one, until it stops or its node services failed: a process that finds no control socket
there, while one does, waits for the socket rather than give up.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

#: The supervisor's control socket, in the service's directory on the node.
CONTROL_SOCKET = "control.sock"
#: The lock each supervisor of the service on the node, and each run that started one, holds shared (the module
#: docstring).
PRESENCE = "present.lock"
#: The longest line either side reads.
MAX_LINE_BYTES = 1 << 20


class ProtocolError(ConnectionError):
    """The other side closed the connection, or sent something that is not a message."""


async def send(writer: asyncio.StreamWriter, message: Mapping[str, Any]) -> None:
    """Write one message."""

    writer.write(json.dumps(dict(message), separators=(",", ":")).encode() + b"\n")
    await writer.drain()


async def receive(reader: asyncio.StreamReader) -> dict[str, Any]:
    """Read one message; :class:`ProtocolError` at the end of the stream or for a line that is not a JSON object."""

    try:
        line = await reader.readline()
    except (ValueError, asyncio.LimitOverrunError) as exc:  # a line longer than the reader's limit
        raise ProtocolError(f"a message longer than {MAX_LINE_BYTES} bytes") from exc
    if not line:
        raise ProtocolError("the connection closed")
    try:
        message = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"not a message: {line[:200]!r}") from exc
    if not isinstance(message, dict):
        raise ProtocolError(f"not a message: {line[:200]!r}")
    return message


__all__ = ["CONTROL_SOCKET", "MAX_LINE_BYTES", "PRESENCE", "ProtocolError", "receive", "send"]
