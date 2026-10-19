"""Remote ORS sessions a process holds, and deleting them when the process cannot close them itself.

An ``ors+http`` session lives on its env server until the client deletes it
(``POST /delete``) or the server's reaper finds it idle, which takes 900 s in
the SDK. A rollout worker that is killed (the supervisor's SIGTERM and SIGKILL,
its own hard-timeout alarm, the RSS watchdog's ``os._exit``) never runs its
sessions' cooperative close, so each killed Toolathlon rollout would hold its
env-server session (database, MCP servers) for 15 minutes.

:class:`~platoon.envs.ors.session.RemoteBackend` registers every session it
opens here and unregisters it once closed. A process can then delete what it
still holds (:func:`delete_orphaned_sessions`, as the worker's SIGTERM handler
does), and :func:`mirror_to` keeps a 0600 file listing them, which the worker
supervisor reads after a worker died without cleaning up. Importing this module
imports neither the SDK nor aiohttp.

:func:`shut_down` is the switch a stopping process throws: it returns the
sessions still open, for :func:`delete_orphaned_sessions`, and from then on
:class:`~platoon.envs.ors.session.RemoteBackend` refuses to open a session or
call a tool (:func:`check_running` raises :class:`RemoteSessionsShutDown`), so
a rollout that ignores cancellation cannot reach an env server again.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

__all__ = [
    "RemoteSessionRecord",
    "RemoteSessionsShutDown",
    "check_running",
    "delete_orphaned_sessions",
    "live_sessions",
    "mirror_to",
    "read_mirror",
    "register",
    "shut_down",
    "unregister",
]


@dataclass(frozen=True)
class RemoteSessionRecord:
    """Enough to delete a remote session without its client: the server, the session id and the credentials."""

    url: str
    sid: str
    api_key: str | None = None
    deployment: str | None = None


class RemoteSessionsShutDown(RuntimeError):
    """This process stopped using env servers (:func:`shut_down`): no session opens and no tool call goes out."""


_lock = threading.Lock()
_live: dict[int, RemoteSessionRecord] = {}
_tokens = itertools.count(1)
_mirror_path: str | None = None
_shut_down = False


def register(record: RemoteSessionRecord) -> int:
    """Note a session this process opened; returns the token :func:`unregister` takes."""

    with _lock:
        token = next(_tokens)
        _live[token] = record
        _write_mirror()
    return token


def unregister(token: int) -> None:
    """Forget a session once it is closed (an unknown token is ignored)."""

    with _lock:
        if _live.pop(token, None) is not None:
            _write_mirror()


def live_sessions() -> list[RemoteSessionRecord]:
    """The sessions this process opened and has not closed."""

    with _lock:
        return list(_live.values())


def shut_down() -> list[RemoteSessionRecord]:
    """Stop using env servers for good, and return the sessions still open (to delete).  Idempotent.

    For a process that is exiting (a rollout worker told to stop).  From now on opening a remote session or
    calling one of its tools raises :class:`RemoteSessionsShutDown` before anything is sent; closing one still
    works.
    """

    global _shut_down
    with _lock:
        _shut_down = True
        return list(_live.values())


def check_running() -> None:
    """Raise :class:`RemoteSessionsShutDown` once :func:`shut_down` was called."""

    if _shut_down:
        raise RemoteSessionsShutDown("This process is shutting down: it opens no env session and calls no tool")


def mirror_to(path: str | None) -> None:
    """Keep ``path`` (written 0600: it holds API keys) listing the live sessions from now on; None stops."""

    global _mirror_path
    with _lock:
        _mirror_path = path
        _write_mirror()


def read_mirror(path: str) -> list[RemoteSessionRecord]:
    """The sessions a :func:`mirror_to` file lists; a missing or unreadable file lists none."""

    try:
        with open(path, encoding="utf-8") as handle:
            entries = json.load(handle)
        return [RemoteSessionRecord(**entry) for entry in entries]
    except (OSError, ValueError, TypeError):
        return []


def _write_mirror() -> None:
    # Called with _lock held.
    if _mirror_path is None:
        return
    temporary = f"{_mirror_path}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump([asdict(record) for record in _live.values()], handle)
        os.replace(temporary, _mirror_path)
    except OSError:
        logger.warning("Cannot record the live ORS sessions in %s", _mirror_path, exc_info=True)


async def delete_orphaned_sessions(records: Sequence[RemoteSessionRecord], timeout_s: float) -> int:
    """Delete ``records`` on their env servers, concurrently and within ``timeout_s`` in all, as the SDK's own
    close does (``POST /delete`` with the session id and API key).  Returns how many deletes the servers
    accepted; never raises (a failure is logged)."""

    if not records:
        return 0
    import aiohttp

    async def delete(client: aiohttp.ClientSession, record: RemoteSessionRecord) -> bool:
        headers = {"X-Session-ID": record.sid}
        if record.api_key is not None:
            headers["X-API-Key"] = record.api_key
        if record.deployment:
            headers["X-Deployment"] = record.deployment
        try:
            async with client.post(f"{record.url.rstrip('/')}/delete", headers=headers) as response:
                return response.status < 300
        except (aiohttp.ClientError, OSError, asyncio.TimeoutError):
            logger.warning("Cannot delete ORS session %s on %s", record.sid, record.url, exc_info=True)
            return False

    timeout = aiohttp.ClientTimeout(total=timeout_s)
    try:
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as client:
            results = await asyncio.wait_for(asyncio.gather(*(delete(client, record) for record in records)), timeout_s)
    except (asyncio.TimeoutError, aiohttp.ClientError, OSError):
        logger.warning("Gave up deleting %d ORS sessions after %.0fs", len(records), timeout_s, exc_info=True)
        return 0
    deleted = sum(results)
    logger.info("Deleted %d of %d ORS sessions this process left open", deleted, len(records))
    return deleted
