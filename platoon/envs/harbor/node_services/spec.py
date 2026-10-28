"""A node service's declaration (:class:`NodeServiceSpec`), and where its files are on a node (:func:`node_dir`).

An environment declares its node services in ``service_kwargs.node_services``, by name; a name given ``site`` takes
the site profile's declaration of it (``node_services`` there).  A declaration::

    frontier-cs-judge:
      image: /srv/frontiersmith/images/frontiercs-judge.sif   # the sandbox provider's image
      command: [sh, /node/start-judge.sh]                        # runs in the sandbox, in the background
      port: 8081                         # where a stack's services reach it, by the name of the service it serves
      cpus: 16                           # 16 of the run's CPUs, whole cores, kept from its rollouts; or "0-15"
      memory_mb: 65536                   # the sandbox's limit, where the provider enforces one
      env: {JUDGE_WORKERS: "4"}
      binds:                             # host paths into the sandbox, read-only unless read_only is false
        - {source: /srv/frontiersmith/problems, target: /app/problems, stage: true}   # copied once per node first
        - {source: /srv/frontiersmith/node-service, target: /node}
      health: {path: /health, interval: 10, timeout: 10, start_period: 600, retries: 3}
      front: {routes: ["POST /evaluate", "GET /health"], max_request_bytes: 16777216, concurrency: 1}
      network: none                      # the sandbox's network (none, or host)
      provider_options: {allow_network_none: true}   # laid over the environment's

The service runs in one sandbox of the environment's sandbox provider (its ``provider`` and ``provider_options``,
with the declaration's options laid over them), and listens on a Unix socket, ``$PLATOON_NODE_SERVICE_SOCKET``, in a
directory of this node that platoon binds into the sandbox (:data:`SERVICE_DIR`); every request reaches it through
platoon's front, with ``Authorization: Bearer $PLATOON_NODE_SERVICE_TOKEN``, which it must check.  The declaration's
canonical JSON names the service on a node (:attr:`NodeServiceSpec.key`): two runs that declare the same service share
it, and a change of any field makes another.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from platoon.utils.scratch import node_scratch

#: Where the node service's directory of this node, which holds the socket it listens on, is bound in its sandbox.
SERVICE_DIR = "/run/platoon-node"
#: The socket the service listens on, in that directory.
SERVICE_SOCKET = "service.sock"
#: The variables the service's command gets: its socket (a path in the sandbox), the token every request carries, and
#: the number of CPUs it is pinned to.
SOCKET_ENV = "PLATOON_NODE_SERVICE_SOCKET"
TOKEN_ENV = "PLATOON_NODE_SERVICE_TOKEN"
CPUS_ENV = "PLATOON_NODE_SERVICE_CPUS"
#: What a name given to an environment's node service in place of a declaration means: the site profile's.
FROM_SITE = "site"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,39}$")
_ROUTE = re.compile(r"^(GET|POST|PUT|DELETE|PATCH|HEAD) (/[A-Za-z0-9._~/-]*)$")
_CPU_LIST = re.compile(r"^\d+(-\d+)?(,\d+(-\d+)?)*$")
_KEYS = frozenset(
    {
        "image",
        "command",
        "port",
        "env",
        "binds",
        "cpus",
        "memory_mb",
        "network",
        "health",
        "front",
        "provider_options",
        "workdir",
        "restarts",
    }
)
_SOCKET_ADDRESS_BYTES = 108  # a Unix socket's path (sun_path), its terminating NUL included
_LONGEST_IN_DIR = len("/scopes/") + 16 + len(".sock")  # the longest socket path below a node service's directory


class NodeServiceError(ValueError):
    """A node service's declaration that cannot run: it names its service and what is wrong."""


@dataclass(frozen=True)
class NodeBind:
    """A host path bound into the node service's sandbox: read-only unless ``read_only`` is false, and with ``stage``,
    a copy of it in node-local scratch, made once per node (:mod:`platoon.envs.harbor.node_services.staging`)."""

    source: str
    target: str
    read_only: bool = True
    stage: bool = False


@dataclass(frozen=True)
class NodeHealth:
    """How the supervisor checks the service: ``GET path`` over its socket answers 2xx within ``timeout_s``, every
    ``interval_s``; failures in its first ``start_period_s`` do not count, and ``retries`` failures in a row make it
    down."""

    path: str = "/health"
    interval_s: float = 10.0
    timeout_s: float = 10.0
    start_period_s: float = 600.0
    retries: int = 3


@dataclass(frozen=True)
class NodeFront:
    """What a rollout's scope lets through to the service: only ``routes`` (``"POST /evaluate"``), with a body of at
    most ``max_request_bytes``, ``concurrency`` requests of one scope at a time (the others wait)."""

    routes: tuple[str, ...] = ("POST /evaluate",)
    max_request_bytes: int = 16 * 1024 * 1024
    concurrency: int = 1


@dataclass(frozen=True)
class NodeServiceSpec:
    """One node service, as declared (the module docstring says what each field means)."""

    name: str
    image: str
    command: tuple[str, ...]
    port: int
    provider: str
    provider_options: Mapping[str, Any] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)
    binds: tuple[NodeBind, ...] = ()
    cpus: int | str | None = None
    memory_mb: int | None = None
    network: Literal["none", "host"] = "none"
    workdir: str = "/"
    health: NodeHealth = NodeHealth()
    front: NodeFront = NodeFront()
    restarts: int = 3  # failed starts in a row (a start that failed soon after it came up counts) before giving up

    @classmethod
    def parse(
        cls, name: str, raw: Any, *, provider: str, provider_options: Mapping[str, Any] | None = None
    ) -> NodeServiceSpec:
        """A declaration from config (``raw``), its service run on ``provider`` with ``provider_options`` under the
        declaration's own; :class:`NodeServiceError` names what is wrong."""

        where = f"node_services.{name}"
        if not isinstance(name, str) or not _NAME.match(name):
            raise NodeServiceError(f"{where}: a node service's name is 1-40 letters, digits, '.', '_' or '-'")
        if not isinstance(raw, Mapping):
            raise NodeServiceError(f"{where} must be a mapping (or {FROM_SITE!r}, the site profile's), got {raw!r}")
        if unknown := sorted(set(raw) - _KEYS):
            raise NodeServiceError(f"{where}: unknown keys {unknown}; known: {', '.join(sorted(_KEYS))}")
        image = raw.get("image")
        if not isinstance(image, str) or not image:
            raise NodeServiceError(f"{where}.image must name the service's image")
        command = raw.get("command")
        if isinstance(command, str):
            command = shlex.split(command)
        if not isinstance(command, list | tuple) or not command or not all(isinstance(arg, str) for arg in command):
            raise NodeServiceError(f"{where}.command must be the service's command, an argv list or a string")
        port = raw.get("port")
        if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
            raise NodeServiceError(f"{where}.port must be the TCP port stacks reach it at, got {port!r}")
        env = raw.get("env") or {}
        if not isinstance(env, Mapping) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise NodeServiceError(f"{where}.env must map names to strings")
        network = raw.get("network", "none")
        if network not in ("none", "host"):
            raise NodeServiceError(f"{where}.network is none or host, not {network!r}")
        options = raw.get("provider_options") or {}
        if not isinstance(options, Mapping):
            raise NodeServiceError(f"{where}.provider_options must be a mapping")
        workdir = raw.get("workdir", "/")
        if not isinstance(workdir, str) or not workdir.startswith("/"):
            raise NodeServiceError(f"{where}.workdir must be an absolute path")
        return cls(
            name=name,
            image=image,
            command=tuple(command),
            port=port,
            provider=provider,
            provider_options={**(provider_options or {}), **options},
            env=dict(env),
            binds=_binds(raw.get("binds") or [], where),
            cpus=_cpus(raw.get("cpus"), where),
            memory_mb=_positive_int(raw.get("memory_mb"), f"{where}.memory_mb"),
            network=network,
            workdir=workdir,
            health=_health(raw.get("health") or {}, where),
            front=_front(raw.get("front") or {}, where),
            restarts=_positive_int(raw.get("restarts", 3), f"{where}.restarts") or 3,
        )

    def to_json(self) -> dict[str, Any]:
        """The declaration as JSON holds it (what the supervisor reads, and what :attr:`key` hashes)."""

        return {
            "name": self.name,
            "image": self.image,
            "command": list(self.command),
            "port": self.port,
            "provider": self.provider,
            "provider_options": json.loads(json.dumps(dict(self.provider_options), default=str)),
            "env": dict(self.env),
            "binds": [vars(bind) for bind in self.binds],
            "cpus": self.cpus,
            "memory_mb": self.memory_mb,
            "network": self.network,
            "workdir": self.workdir,
            "health": vars(self.health),
            "front": {**vars(self.front), "routes": list(self.front.routes)},
            "restarts": self.restarts,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> NodeServiceSpec:
        """The declaration that :meth:`to_json` wrote."""

        return cls(
            name=data["name"],
            image=data["image"],
            command=tuple(data["command"]),
            port=int(data["port"]),
            provider=data["provider"],
            provider_options=dict(data.get("provider_options") or {}),
            env=dict(data.get("env") or {}),
            binds=tuple(NodeBind(**bind) for bind in data.get("binds") or ()),
            cpus=data.get("cpus"),
            memory_mb=data.get("memory_mb"),
            network=data.get("network", "none"),
            workdir=data.get("workdir", "/"),
            health=NodeHealth(**data.get("health") or {}),
            front=_front_from_json(data.get("front") or {}),
            restarts=int(data.get("restarts", 3)),
        )

    @property
    def key(self) -> str:
        """What names the service on a node: a digest of its canonical declaration."""

        canonical = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def allows(self, method: str, path: str) -> bool:
        """Whether the front lets ``method path`` through to the service."""

        return f"{method.upper()} {path}" in self.front.routes


def _front_from_json(data: Mapping[str, Any]) -> NodeFront:
    default = NodeFront()
    return NodeFront(
        routes=tuple(data.get("routes", default.routes)),
        max_request_bytes=int(data.get("max_request_bytes", default.max_request_bytes)),
        concurrency=int(data.get("concurrency", default.concurrency)),
    )


def node_dir(spec: NodeServiceSpec) -> Path:
    """The directory of ``spec``'s service on this node: ``platoon-node-services-<uid>/<name>-<key>`` in node scratch,
    or in ``/tmp`` where node scratch's path leaves no room for a socket's address below it.  Every process of the node
    that declares the same service finds the same one (its supervisor's lock, control socket and scopes are there)."""

    leaf = f"platoon-node-services-{os.getuid()}/{spec.name}-{spec.key}"
    for root in (node_scratch(), Path("/tmp")):
        path = root / leaf
        if len(os.fsencode(path)) + _LONGEST_IN_DIR < _SOCKET_ADDRESS_BYTES:
            return path
    raise NodeServiceError(f"Neither node scratch nor /tmp has a path short enough for the sockets of {spec.name}")


def staging_root() -> Path:
    """Where staged binds are copied on this node, shared by every node service of this user's."""

    return node_scratch() / f"platoon-node-services-{os.getuid()}" / "staged"


def _binds(raw: Any, where: str) -> tuple[NodeBind, ...]:
    if not isinstance(raw, list):
        raise NodeServiceError(f"{where}.binds must be a list of {{source, target}} mappings")
    binds = []
    for number, entry in enumerate(raw):
        at = f"{where}.binds[{number}]"
        if not isinstance(entry, Mapping) or set(entry) - {"source", "target", "read_only", "stage"}:
            raise NodeServiceError(f"{at} must be a mapping of source, target, read_only and stage")
        source, target = entry.get("source"), entry.get("target")
        if not isinstance(source, str) or not os.path.isabs(source):
            raise NodeServiceError(f"{at}.source must be an absolute host path")
        if not isinstance(target, str) or not target.startswith("/") or target.rstrip("/") == SERVICE_DIR:
            raise NodeServiceError(f"{at}.target must be an absolute path in the sandbox, other than {SERVICE_DIR}")
        read_only, stage = entry.get("read_only", True), entry.get("stage", False)
        if not isinstance(read_only, bool) or not isinstance(stage, bool):
            raise NodeServiceError(f"{at}: read_only and stage are booleans")
        if stage and not read_only:
            raise NodeServiceError(f"{at}: a staged bind is shared by the node's services, so it is read-only")
        binds.append(NodeBind(source=source, target=target, read_only=read_only, stage=stage))
    return tuple(binds)


def _cpus(raw: Any, where: str) -> int | str | None:
    if raw is None:
        return None
    if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
        return raw
    if isinstance(raw, str) and _CPU_LIST.match(raw.replace(" ", "")):
        return raw.replace(" ", "")
    raise NodeServiceError(f"{where}.cpus is a number of CPUs or a list of them (0-15,32), not {raw!r}")


def _positive_int(raw: Any, where: str) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise NodeServiceError(f"{where} must be a positive integer, got {raw!r}")
    return raw


def _seconds(raw: Any, where: str, default: float) -> float:
    if raw is None:
        return default
    if isinstance(raw, bool) or not isinstance(raw, int | float) or raw < 0:
        raise NodeServiceError(f"{where} must be a number of seconds, got {raw!r}")
    return float(raw)


def _health(raw: Any, where: str) -> NodeHealth:
    if not isinstance(raw, Mapping) or set(raw) - {"path", "interval", "timeout", "start_period", "retries"}:
        raise NodeServiceError(f"{where}.health takes path, interval, timeout, start_period and retries")
    path = raw.get("path", "/health")
    if not isinstance(path, str) or not path.startswith("/"):
        raise NodeServiceError(f"{where}.health.path must be a path, such as /health")
    default = NodeHealth()
    return NodeHealth(
        path=path,
        interval_s=_seconds(raw.get("interval"), f"{where}.health.interval", default.interval_s) or 1.0,
        timeout_s=_seconds(raw.get("timeout"), f"{where}.health.timeout", default.timeout_s) or 1.0,
        start_period_s=_seconds(raw.get("start_period"), f"{where}.health.start_period", default.start_period_s),
        retries=_positive_int(raw.get("retries", default.retries), f"{where}.health.retries") or default.retries,
    )


def _front(raw: Any, where: str) -> NodeFront:
    if not isinstance(raw, Mapping) or set(raw) - {"routes", "max_request_bytes", "concurrency"}:
        raise NodeServiceError(f"{where}.front takes routes, max_request_bytes and concurrency")
    default = NodeFront()
    routes = raw.get("routes", list(default.routes))
    if not isinstance(routes, list) or not routes or not all(isinstance(r, str) and _ROUTE.match(r) for r in routes):
        raise NodeServiceError(f'{where}.front.routes must list routes such as "POST /evaluate"')
    return NodeFront(
        routes=tuple(routes),
        max_request_bytes=_positive_int(raw.get("max_request_bytes"), f"{where}.front.max_request_bytes")
        or default.max_request_bytes,
        concurrency=_positive_int(raw.get("concurrency"), f"{where}.front.concurrency") or default.concurrency,
    )


__all__ = [
    "CPUS_ENV",
    "FROM_SITE",
    "SERVICE_DIR",
    "SERVICE_SOCKET",
    "SOCKET_ENV",
    "TOKEN_ENV",
    "NodeBind",
    "NodeFront",
    "NodeHealth",
    "NodeServiceError",
    "NodeServiceSpec",
    "node_dir",
    "staging_root",
]
