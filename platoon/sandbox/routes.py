"""Routes: how a process in a sandbox reaches a service on the host (``SandboxSpec.upstreams`` -> ``Sandbox.routes``).

A spec's ``upstreams`` map a name to a host endpoint, and the built-in providers turn each one into a route, the URL
a process inside the sandbox calls:

* ``unix:///abs/dir/name.sock``, a Unix socket on the host.  In a sandbox without a network (``network: none``), the
  provider binds the socket's directory read-only at ``/run/platoon/<name>`` and starts a forwarder inside that
  relays a loopback TCP port to the socket, so the route is ``http://127.0.0.1:<port>``: port 18080 for the first
  such upstream (18081 for the second, in name order) in a sandbox with a network namespace of its own, a free port
  in one that shares the host's (the ``process`` provider).  Binding the directory, not the socket, lets the service
  create its socket before or after the sandbox starts; connecting to a socket needs no write access to the mount,
  so the bind is read-only and code in the sandbox cannot create or fill files in the host directory (the
  ``process`` provider, which enforces no mount, links the directory instead).  A sandbox on the host network gets
  no route to a Unix socket: its processes reach that service on the host's loopback as they are.
* ``http://host:port`` (or ``https``), a TCP endpoint: its own route in a sandbox on the host network.  A sandbox
  without a network cannot reach it, so a spec that asks for one is refused.

The forwarder (``platoon/sandbox/forwarder.py``, standard library only) runs with the first Python 3 it finds in the
sandbox: ``bin/python3`` of a read-only mount (a harness toolchain), then ``python3`` and ``python`` on the image's
``PATH``.  Without one, the sandbox does not start (:class:`RouteError`).  The forwarder carries ``PLATOON_INFRA=route``
in its environment and no agent's marker, so the cleanup of a harness, which kills the processes that carry its
agent's marker, spares it; it dies with the sandbox.  A policy can still kill it (``pkill python``):
``Sandbox.ensure_routes()``, which the harness runner calls before each launch, starts it again on the same port.
The route's own commands run in ``/``, as the runner's do, so they still work once the agent has removed or moved its
working directory, where no command starts.
"""

from __future__ import annotations

import asyncio
import posixpath
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from platoon.sandbox.base import Mount, SandboxSpec

if TYPE_CHECKING:
    from platoon.sandbox.base import Sandbox

#: The upstream through which a harness in a sandbox reaches the model gateway.
GATEWAY = "gateway"
#: Where a Unix-socket upstream's directory is bound in the sandbox: ``<ROUTE_ROOT>/<name>``.
ROUTE_ROOT = "/run/platoon"
#: Where the forwarder keeps its script, process id, port and log in the sandbox.
STATE_DIR = "/tmp/platoon-route"
#: The port of the first forwarded upstream in a sandbox with a network namespace of its own.
FIRST_PORT = 18080
LOOPBACK = "127.0.0.1"
#: The environment marker of platoon's infrastructure processes in a sandbox; the forwarder's value is ``route``.
INFRA_ENV = "PLATOON_INFRA"
ROUTE_MARKER = f"{INFRA_ENV}=route"
FORWARDER = Path(__file__).with_name("forwarder.py")
START_TIMEOUT_S = 60.0

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SCHEMES = ("unix://", "http://", "https://")

# Starts the forwarder in the background and prints its port.  $1 state dir, $2 upstream name, $3 socket path in the
# sandbox, $4 port (0: any), then the Python candidates.  Its output goes to a log file, so the exec returns at once.
_START_SCRIPT = r"""
set -u
dir="$1"; name="$2"; socket="$3"; port="$4"; shift 4
mkdir -p "$dir" || exit 1
python=""
for candidate in "$@"; do
  case "$candidate" in
    /*) [ -x "$candidate" ] || continue ;;
    *) command -v "$candidate" >/dev/null 2>&1 || continue ;;
  esac
  if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 6) else 1)' >/dev/null 2>&1; then
    python="$candidate"
    break
  fi
done
if [ -z "$python" ]; then
  echo "no Python 3 to run platoon's route forwarder; looked for: $*" >&2
  exit 127
fi
rm -f "$dir/$name.port"
PLATOON_INFRA=route "$python" -I "$dir/forwarder.py" --socket "$socket" --port "$port" \
  --port-file "$dir/$name.port" </dev/null >"$dir/$name.log" 2>&1 &
pid=$!
echo "$pid" > "$dir/$name.pid"
tries=0
while [ ! -s "$dir/$name.port" ]; do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "platoon's route forwarder exited at start:" >&2
    cat "$dir/$name.log" >&2
    exit 1
  fi
  tries=$((tries + 1))
  if [ "$tries" -gt 300 ]; then
    echo "platoon's route forwarder did not listen within 30 s:" >&2
    cat "$dir/$name.log" >&2
    kill "$pid" 2>/dev/null
    exit 1
  fi
  sleep 0.1 2>/dev/null || sleep 1
done
cat "$dir/$name.port"
"""

# Prints the port of a live forwarder, else fails.  $1 state dir, $2 upstream name.  A pid is the forwarder's only
# while its environment carries the marker: a dead forwarder's pid may have been reused.
_HEALTH_SCRIPT = r"""
pid=$(cat "$1/$2.pid" 2>/dev/null) || exit 1
[ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || exit 1
tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -qx 'PLATOON_INFRA=route' || exit 1
cat "$1/$2.port" 2>/dev/null
"""


class RouteError(RuntimeError):
    """A sandbox's route could not be served: no Python 3 for the forwarder, or the forwarder did not start."""


@dataclass(frozen=True)
class Upstream:
    """One entry of ``SandboxSpec.upstreams``."""

    name: str
    url: str

    @property
    def socket(self) -> str | None:
        """The host path of a ``unix://`` upstream; None for a TCP one."""

        return self.url.removeprefix("unix://") if self.url.startswith("unix://") else None

    @property
    def target(self) -> str:
        """Where a ``unix://`` upstream's directory is bound in the sandbox."""

        return posixpath.join(ROUTE_ROOT, self.name)


def parse_upstreams(upstreams: Mapping[str, str]) -> list[Upstream]:
    """Validate ``SandboxSpec.upstreams``, in name order; a bad name or URL raises ``ValueError``."""

    parsed = []
    for name, url in sorted(upstreams.items()):
        if not isinstance(name, str) or not _NAME.match(name):
            raise ValueError(f"Upstream name {name!r} must match {_NAME.pattern}")
        if not isinstance(url, str) or not url.startswith(_SCHEMES):
            raise ValueError(f"Upstream {name!r} must be a unix://, http:// or https:// URL, not {url!r}")
        upstream = Upstream(name, url)
        socket = upstream.socket
        if socket is not None and (not posixpath.isabs(socket) or posixpath.basename(socket) in ("", ".", "..")):
            raise ValueError(f"Upstream {name!r} must name a socket by its absolute path: unix:///dir/name.sock")
        if socket is None and not url.split("://", 1)[1].split("/", 1)[0]:
            raise ValueError(f"Upstream {name!r} has no host: {url!r}")
        parsed.append(upstream)
    return parsed


def python_candidates(mounts: Sequence[Mount]) -> list[str]:
    """The interpreters the forwarder tries, in order: ``bin/python3`` of each read-only mount, then the image's."""

    candidates = [posixpath.join(mount.target, "bin", "python3") for mount in mounts if mount.read_only]
    return list(dict.fromkeys([*candidates, "python3", "python"]))


class SandboxRoutes:
    """The routes of one sandbox: the binds its upstreams need, and the forwarders that serve them.

    A provider builds one per sandbox from the caller's spec, before it starts anything (a spec it cannot route
    raises ``ValueError``), adds :meth:`mounts` to the sandbox's mounts, and calls :meth:`start` once the sandbox
    runs.  ``own_network`` says whether a sandbox without a network gets a network namespace of its own, where the
    fixed ports cannot collide with another sandbox's.
    """

    def __init__(self, spec: SandboxSpec, *, own_network: bool) -> None:
        upstreams = parse_upstreams(spec.upstreams)
        direct = [upstream for upstream in upstreams if upstream.socket is None]
        isolated = spec.network == "none"
        if isolated and direct:
            names = ", ".join(upstream.name for upstream in direct)
            raise ValueError(
                f"A sandbox without a network cannot reach the TCP upstream(s) {names}; give them as unix:// "
                "sockets, which it reaches through a forwarder"
            )
        self._direct = {upstream.name: upstream.url for upstream in direct}
        self._forwarded = [upstream for upstream in upstreams if upstream.socket is not None] if isolated else []
        self._own_network = own_network
        self._pythons = python_candidates(spec.mounts)
        self._ports: dict[str, int] = {}
        self._lock = asyncio.Lock()

    @property
    def forwarded(self) -> tuple[str, ...]:
        """The names of the upstreams a forwarder serves."""

        return tuple(upstream.name for upstream in self._forwarded)

    def mounts(self) -> list[Mount]:
        """The read-only binds of the forwarded upstreams' socket directories."""

        return [
            Mount(source=posixpath.dirname(upstream.socket), target=upstream.target, read_only=True)
            for upstream in self._forwarded
            if upstream.socket is not None
        ]

    @property
    def routes(self) -> dict[str, str]:
        """Upstream name -> the URL a process in the sandbox calls; the forwarded ones once they have started."""

        routes = dict(self._direct)
        routes.update({name: f"http://{LOOPBACK}:{port}" for name, port in self._ports.items()})
        return routes

    async def start(self, sandbox: Sandbox) -> dict[str, str]:
        """Start a forwarder for each forwarded upstream; returns :attr:`routes`."""

        async with self._lock:
            for index, upstream in enumerate(self._forwarded):
                await self._start(sandbox, upstream, self._first_port(index))
        return self.routes

    async def ensure(self, sandbox: Sandbox) -> dict[str, str]:
        """Start again each forwarder that is gone, on its port; returns :attr:`routes`.

        In a sandbox that shares the host's network, another process may have taken the port meanwhile: the
        forwarder then gets a free one, and the route changes.
        """

        async with self._lock:
            for index, upstream in enumerate(self._forwarded):
                port = await self._live_port(sandbox, upstream)
                if port is not None:
                    self._ports[upstream.name] = port
                    continue
                previous = self._ports.get(upstream.name, self._first_port(index))
                try:
                    await self._start(sandbox, upstream, previous)
                except RouteError:
                    if self._own_network or previous == 0:
                        raise
                    await self._start(sandbox, upstream, 0)
        return self.routes

    def _first_port(self, index: int) -> int:
        return FIRST_PORT + index if self._own_network else 0

    async def _start(self, sandbox: Sandbox, upstream: Upstream, port: int) -> None:
        assert upstream.socket is not None
        script_path = posixpath.join(STATE_DIR, "forwarder.py")
        try:
            await sandbox.upload(FORWARDER, script_path)
            result = await sandbox.exec(
                [
                    "sh",
                    "-c",
                    _START_SCRIPT,
                    "platoon-route",
                    STATE_DIR,
                    upstream.name,
                    posixpath.join(upstream.target, posixpath.basename(upstream.socket)),
                    str(port),
                    *self._pythons,
                ],
                cwd="/",
                timeout_s=START_TIMEOUT_S,
            )
        except RouteError:
            raise
        except Exception as exc:
            raise RouteError(f"Cannot start the route to {upstream.name!r}: {type(exc).__name__}: {exc}") from exc
        output = result.stdout.strip().splitlines()
        if result.exit_code != 0 or result.timed_out or not output or not output[-1].strip().isdigit():
            detail = (result.stderr or result.stdout).strip()[-2000:]
            raise RouteError(f"Cannot start the route to {upstream.name!r} (exit {result.exit_code}): {detail}")
        self._ports[upstream.name] = int(output[-1])

    async def _live_port(self, sandbox: Sandbox, upstream: Upstream) -> int | None:
        try:
            result = await sandbox.exec(
                ["sh", "-c", _HEALTH_SCRIPT, "platoon-route", STATE_DIR, upstream.name],
                cwd="/",
                timeout_s=START_TIMEOUT_S,
            )
        except Exception:
            return None
        output = result.stdout.strip()
        return int(output) if result.exit_code == 0 and output.isdigit() else None


__all__ = [
    "FIRST_PORT",
    "GATEWAY",
    "INFRA_ENV",
    "ROUTE_MARKER",
    "ROUTE_ROOT",
    "RouteError",
    "SandboxRoutes",
    "Upstream",
    "parse_upstreams",
    "python_candidates",
]
