"""Harbor compose files: the subset platoon runs, Harbor's layering and interpolation, and the stack plan.

A Harbor task may ship ``environment/docker-compose.yaml``, the only compose file Harbor reads (harbor
``environments/docker/docker.py``); another name (``docker-compose.yml``, ``compose.yaml``, ``compose.yml``) is a plain
file of the task, as in Harbor, which a fetch warns of.  Harbor layers the task's file over a base that defines
``main``, the agents' container (built from ``environment/``, or the prebuilt ``docker_image``, running ``sh -c "sleep
infinity"``), and interpolates it with its own variables over the task's ``[environment].env``.  :func:`plan_compose`
reads a task's file the same way into a **stack plan** (:class:`ComposePlan`), which a fetch records in the lock, and
refuses (:class:`ComposeError`, naming the service and key) a file that platoon could not run as written.  The rule
is Harbor's at a pinned commit, :data:`COMPOSE_RULE`, which the lock records: a change of Harbor's is a visible rule
bump.

**Kinds** (``flags.stack`` in a lock): ``single`` (no compose file), ``trivial`` (only ``main``, with nothing
platoon applies, as some converters write for every task), ``main_only`` (only ``main``, with settings: an
environment, binds, a command) and ``sidecars`` (services besides ``main``).  ``main_only`` tasks run on every
provider as one sandbox (:class:`MainService`); ``sidecars`` run as stacks on providers that run them
(:mod:`platoon.envs.harbor.sidecars`), and are refused elsewhere.

**Interpolation** follows compose: ``$NAME``, ``${NAME}``, ``${NAME:-default}``, ``${NAME-default}``,
``${NAME:?error}``, ``${NAME?error}``, ``${NAME:+value}``, ``${NAME+value}`` (defaults may nest), and ``$$`` for a
literal ``$``; an unset variable without a default is blank, as compose makes it, and the plan notes it.  The
variables are Harbor's, with platoon's meaning: ``CONTEXT_DIR`` is the task's ``environment/`` (only binds and build
contexts may name it), ``HOST_{VERIFIER,AGENT}_LOGS_PATH`` and ``HOST_ARTIFACTS_PATH`` the session's log
directories, ``ENV_*`` their paths in ``main`` (``/logs/verifier``, ...), ``MAIN_IMAGE_NAME`` ``main``'s image,
``PREBUILT_IMAGE_NAME`` the task's ``docker_image``, ``CPUS`` and ``MEMORY`` its ``[environment]`` resources; then the
task's ``[environment].env``; then the host variables an overlay lists (``compose_host_env``).  Nothing else of the
host's environment: a file that names an API key gets a blank one unless an overlay lists it.  A task variable that
is a host template (``${TOKEN}``) and a listed host variable stay unresolved in the plan, which records no host value:
a session resolves them when it opens, under the form the file gives them (``${NAME:-default}``, ...; one named
without a default must be set then).  Environment values may name them, and a read-only bind a listed host
variable's path.  Harbor's egress-control sidecar is never emulated: platoon has network phases of its own.

**Keys** (:data:`SERVICE_KEYS`): a short list is accepted (``image``, ``build``, ``command``, ``entrypoint``,
``environment``, ``env_file``, ``working_dir``, ``volumes``, ``tmpfs``, ``depends_on``, ``healthcheck``, ``expose``,
``ports``, ``networks``, ``network_mode`` of another service, ``hostname``, ``shm_size``, ``ulimits``, ``cpus``,
``mem_limit``, ``deploy.resources``, ``cap_add`` of capabilities a service's root holds over its own namespaces,
``stop_signal``, ``profiles``), a shorter one is ignored because it cannot change what a task measures
(:data:`IGNORED_KEYS`: ``container_name``, ``restart``, ``init``, ``stdin_open``, ``tty``, ``labels``, ``logging``,
``stop_grace_period``, ``pull_policy``, and ``platform`` when it is this host's), and every other key is refused,
security-relevant ones (``privileged``, ``devices``, ``pid``, ``ipc``, ``security_opt``, ``sysctls``, ``user``, ...)
included.  A fetch with ``refuse: {privileges: false}`` plans ``privileged`` and a ``cap_add`` of other capabilities
(``NET_ADMIN``, ...) instead, as the stack's **privileges** (:attr:`ComposePlan.privileges`), which only a sandbox
provider with the ``privileged`` capability (``docker``) grants: the task loader refuses such a task elsewhere.
Published ``ports`` are kept as ``expose``: nothing is ever published on the host.  A service with
``profiles`` is not started (Harbor starts none), and is left out.  A dataset's overlay may set, replace or unset a
service's keys (``[[compose]]``, :class:`ComposeEdit`), which the plan records.

**Volumes.** Binds of the task's own files (``./data:/data``, ``${CONTEXT_DIR}/..:/opt/task-src:ro``) bind the
cached task (a missing source, which Docker would create, binds an empty directory); anonymous and named volumes
below a bind **mask** it (:attr:`Bind.masks`), so that the masked files never enter a sandbox; binds of
``${HOST_*_PATH}`` bind the session's log directories; any other host path is refused, but for the one a listed host
variable names (or a path below it), read-only.
"""

from __future__ import annotations

import platform as host_platform
import posixpath
import re
import shlex
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml

from platoon.envs.harbor.dockerfile import DockerfileError, plan_dockerfile
from platoon.sandbox.stacks import dependency_cycle

COMPOSE_FILE = "docker-compose.yaml"
# Names compose itself reads, which Harbor does not: plain files of a Harbor task (a fetch warns of them).
UNREAD_COMPOSE_FILES = ("docker-compose.yml", "compose.yaml", "compose.yml")
COMPOSE_RULE = "harbor-c178c20"
MAIN = "main"
ENVIRONMENT_DIR = "environment"
StackKind = Literal["single", "trivial", "main_only", "sidecars"]
STACK_KINDS: tuple[str, ...] = ("single", "trivial", "main_only", "sidecars")

# Harbor's base gives main this command (docker-compose-build.yaml, docker-compose-prebuilt.yaml).
BASE_COMMAND = ("sh", "-c", "sleep infinity")
LOG_DIRS = {"verifier": "/logs/verifier", "agent": "/logs/agent", "artifacts": "/logs/artifacts"}
# Paths a bind may not cover in main: what platoon itself puts there.
RESERVED_TARGETS = ("/tests", "/solution", "/logs", "/run/platoon")

# Harbor's infrastructure variables, as platoon defines them while it plans: the task's directory and the session's
# log directories as placeholders that only binds and build contexts may name, resolved when a session opens.
_TASK_ROOT = "/.platoon-task"
_LOGS_ROOT = "/.platoon-logs"
_MAIN_IMAGE = "platoon.main-image"
# A variable the plan leaves for the session to resolve: a listed host variable, or a task variable that is a host
# template.  Private-use characters around "host.NAME" or "task.NAME", with the operator and word of a
# ${NAME:-word} form (one more private-use character each, never a colon, which separates a volume's source from its
# target), which the session applies.  A compose file, an env file or a task's settings that hold any of these
# characters is refused: written there, they would name host variables that no overlay lists.
_DEFER_OPEN, _DEFER_CLOSE = "\ue000", "\ue001"
_OPERATORS = {":-": "\ue002", "-": "\ue003", ":+": "\ue004", "+": "\ue005", ":?": "\ue006", "?": "\ue007"}
_OPERATOR_OF = {mark: operator for operator, mark in _OPERATORS.items()}
_RESERVED = re.compile("[\ue000-\ue007]")
_DEFERRED = re.compile(
    f"{_DEFER_OPEN}(host|task)[.]([A-Za-z_][A-Za-z0-9_]*)(?:([\ue002-\ue007])([^\ue000-\ue007]*))?{_DEFER_CLOSE}"
)
_BARE_DEFERRED = re.compile(f"{_DEFER_OPEN}(host|task)[.]([A-Za-z_][A-Za-z0-9_]*){_DEFER_CLOSE}")
# A compose file's values as its aliases expand them, at most: YAML's anchors let a small file stand for a tree too big
# to plan ("billion laughs").  Compose files hold hundreds.
MAX_DOCUMENT_VALUES = 100_000
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_HOST_TEMPLATE = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")  # a task's env value that names a host variable (task.py)
_SERVICE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_VOLUME_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

TOP_LEVEL_KEYS = frozenset({"services", "networks", "volumes", "version", "name"})
SERVICE_KEYS = frozenset(
    {
        "image",
        "build",
        "command",
        "entrypoint",
        "environment",
        "env_file",
        "working_dir",
        "volumes",
        "tmpfs",
        "depends_on",
        "healthcheck",
        "expose",
        "ports",
        "networks",
        "network_mode",
        "hostname",
        "shm_size",
        "ulimits",
        "cpus",
        "mem_limit",
        "deploy",
        "cap_add",
        "stop_signal",
        "profiles",
    }
)
# Keys that cannot change what a task measures: recorded in the plan, never applied.
IGNORED_KEYS = frozenset(
    {
        "container_name",
        "restart",
        "init",
        "stdin_open",
        "tty",
        "labels",
        "logging",
        "stop_grace_period",
        "pull_policy",
        "platform",
    }
)
# Capabilities a service's root holds over its own namespaces, which compose may add: those Docker grants by default,
# and SYS_PTRACE, over the service's own processes.  NET_ADMIN and NET_RAW over the stack's network, and everything
# else, are refused unless an overlay drops them, DAC_READ_SEARCH among them: where a service's root is the host's (a
# rootful Docker), it reads any file of the host by its handle (open_by_handle_at).
HELD_CAPABILITIES = frozenset(
    {
        "AUDIT_WRITE",
        "CHOWN",
        "DAC_OVERRIDE",
        "FOWNER",
        "FSETID",
        "KILL",
        "MKNOD",
        "NET_BIND_SERVICE",
        "SETFCAP",
        "SETGID",
        "SETPCAP",
        "SETUID",
        "SYS_CHROOT",
        "SYS_PTRACE",
    }
)
# Why a key that matters to a task's security, or to what it measures, is refused; every key that is neither accepted
# nor ignored is refused too, as outside the subset.
REFUSED_KEYS = {
    "privileged": (
        "gives the service every privilege of the host, which platoon grants only on the docker provider, to a "
        "dataset fetched with refuse: {privileges: false}"
    ),
    "devices": "gives the service host devices, which platoon never grants",
    "gpus": "asks for GPUs; a task asks for them with [environment].gpus, and a stack's services get none",
    "pid": "shares a PID namespace other than the service's own",
    "ipc": "shares an IPC namespace other than the service's own",
    "uts": "shares a UTS namespace other than the service's own",
    "userns_mode": "changes the service's user namespace",
    "cgroup": "changes the service's cgroup namespace",
    "cgroup_parent": "places the service in another cgroup",
    "security_opt": "changes the service's security profile",
    "sysctls": "sets kernel parameters, which a rootless sandbox does not",
    "cap_drop": "drops capabilities, which platoon cannot drop in a rootless sandbox",
    "user": "runs the service as another user, which platoon cannot honor yet (RFC 0010, section 7)",
    "group_add": "adds groups to the service's user, which platoon cannot honor",
    "read_only": "makes the service's root filesystem read-only, which platoon does not",
    "runtime": "picks a container runtime",
    "extra_hosts": "adds host names, which a stack's one network of its own does not take",
    "dns": "sets name servers, which a stack's one network of its own does not take",
    "dns_search": "sets name servers' search domains, which a stack's network does not take",
    "dns_opt": "sets resolver options, which a stack's network does not take",
    "domainname": "sets a domain name, which a stack's network does not take",
    "mac_address": "sets a MAC address, which a stack's one network of its own does not take",
    "links": "links services by legacy names; every service name of a stack resolves already: drop it with an overlay",
    "external_links": "links containers outside the stack",
    "volumes_from": "mounts another container's volumes",
    "extends": "extends a service of another file",
    "pids_limit": "limits the service's processes, which platoon does not",
}
CONDITIONS = ("service_started", "service_healthy", "service_completed_successfully")
_NETWORK_KEYS = frozenset({"internal", "driver", "labels", "attachable", "name"})
_VOLUME_KEYS = frozenset({"driver", "labels", "name"})
_BUILD_KEYS = frozenset({"context", "dockerfile", "args"})
_HEALTHCHECK_KEYS = frozenset({"test", "interval", "timeout", "retries", "start_period", "start_interval", "disable"})
_DEPLOY_KEYS = frozenset({"resources", "restart_policy", "labels", "replicas"})
_DURATION = re.compile(r"(\d+(?:\.\d*)?|\.\d+)(ns|us|\u00b5s|ms|s|m|h)")
# Docker's healthcheck defaults, which a value of 0 also means.
_DEFAULT_INTERVAL_S, _DEFAULT_TIMEOUT_S, _DEFAULT_START_INTERVAL_S, _DEFAULT_RETRIES = 30.0, 30.0, 5.0, 3
_DURATION_UNITS = {"ns": 1e-9, "us": 1e-6, "\u00b5s": 1e-6, "ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0}
_SIZE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgtp]?)b?\s*$", re.IGNORECASE)
_SIZE_UNITS = {"": 1, "k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "t": 1 << 40, "p": 1 << 50}
_IDLE = (("sleep", "infinity"), ("sleep", "inf"), ("tail", "-f", "/dev/null"))
_SHELLS = ("sh", "/bin/sh", "bash", "/bin/bash", "/usr/bin/bash", "ash", "/bin/ash", "dash", "/bin/dash")


class ComposeError(ValueError):
    """A compose file outside the subset platoon runs; ``where`` names the service and key (``main.cap_add``)."""

    def __init__(self, where: str, message: str) -> None:
        super().__init__(f"{where}: {message}" if where else message)
        self.where = where


class ComposeEditError(ComposeError):
    """A dataset overlay's compose edit that does not fit the task's compose file (a service it lacks)."""


@dataclass(frozen=True)
class ComposeEdit:
    """A dataset overlay's edit of one compose service (``[[compose]]`` in ``overlay.toml``), applied before the
    file is interpolated: ``unset`` removes keys (dotted for nested ones: ``deploy.resources.reservations``),
    ``replace`` sets keys whole, ``set`` merges mappings into the service's (other values replace), and ``external``
    marks a service that a node service serves, which platoon does not start."""

    overlay: str
    service: str
    set: Mapping[str, Any] = field(default_factory=dict)
    replace: Mapping[str, Any] = field(default_factory=dict)
    unset: tuple[str, ...] = ()
    external: str | None = None

    @classmethod
    def parse(cls, overlay: str, entry: Any) -> ComposeEdit:
        where = f"overlay {overlay}: [[compose]]"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{where} entries must be tables, got {entry!r}")
        if unknown := sorted(set(entry) - {"service", "set", "replace", "unset", "external"}):
            raise ValueError(f"{where}: unknown keys {unknown}")
        service = entry.get("service")
        if not isinstance(service, str) or not _SERVICE_NAME.match(service):
            raise ValueError(f"{where} needs a service name, got {service!r}")
        for key in ("set", "replace"):
            if not isinstance(entry.get(key, {}), Mapping):
                raise ValueError(f"{where} {service}: {key} must be a table")
        unset = entry.get("unset", [])
        if not isinstance(unset, list) or not all(isinstance(key, str) and key for key in unset):
            raise ValueError(f"{where} {service}: unset must list keys")
        external = entry.get("external")
        if external is not None and (not isinstance(external, str) or not external):
            raise ValueError(f"{where} {service}: external must name a node service")
        if not (entry.get("set") or entry.get("replace") or unset or external):
            raise ValueError(f"{where} {service}: give set, replace, unset or external")
        return cls(
            overlay=overlay,
            service=service,
            set=dict(entry.get("set") or {}),
            replace=dict(entry.get("replace") or {}),
            unset=tuple(unset),
            external=external,
        )

    def describe(self) -> str:
        """What the edit does, as the plan records it: ``overlay: service: set a, b; unset c``."""

        parts = [
            f"{verb} {', '.join(sorted(keys))}" for verb, keys in (("set", self.set), ("replace", self.replace)) if keys
        ]
        if self.unset:
            parts.append(f"unset {', '.join(self.unset)}")
        if self.external:
            parts.append(f"external {self.external}")
        return f"{self.overlay}: {self.service}: {'; '.join(parts)}"


@dataclass(frozen=True)
class Bind:
    """A bind of ``main`` (a main-only task's): the task's own files (``task``, relative to the task directory: ``.``
    for all of it), a session log directory (``logs``: ``agent``, ``verifier`` or ``artifacts``) or the host path that
    a listed host variable names, or one below it (``host``, a template the session resolves).  ``masks`` are the
    paths below the source that volumes hide, relative to it; ``missing``: the task has no such file, so an empty
    directory is bound, as Docker creates a missing source."""

    target: str
    read_only: bool
    task: str | None = None
    logs: str | None = None
    host: str | None = None
    masks: tuple[str, ...] = ()
    missing: bool = False


@dataclass(frozen=True)
class ServiceHealthcheck:
    """A compose healthcheck (or an image's ``HEALTHCHECK``): ``test`` as compose writes it (``["CMD", ...]``,
    ``["CMD-SHELL", text]``), with Docker's defaults, which a value of 0 also means (but for ``start_period``, whose
    default is 0)."""

    test: tuple[str, ...]
    interval_s: float = 30.0
    timeout_s: float = 30.0
    start_period_s: float = 0.0
    start_interval_s: float = 5.0
    retries: int = 3

    @property
    def command(self) -> list[str]:
        """The command it runs: a ``CMD`` test as given, a ``CMD-SHELL`` one under ``/bin/sh -c``."""

        kind, *rest = self.test
        return list(rest) if kind == "CMD" else ["/bin/sh", "-c", " ".join(rest)]

    @property
    def worst_case_s(self) -> float:
        """How long waiting for it to pass may take at most: its start period, then each retry's timeout and
        interval."""

        return self.start_period_s + (self.retries + 1) * (self.timeout_s + self.interval_s)

    def to_json(self) -> dict[str, Any]:
        return {
            "test": list(self.test),
            "interval_s": self.interval_s,
            "timeout_s": self.timeout_s,
            "start_period_s": self.start_period_s,
            "start_interval_s": self.start_interval_s,
            "retries": self.retries,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ServiceHealthcheck:
        return cls(
            test=tuple(data["test"]),
            interval_s=float(data.get("interval_s") or _DEFAULT_INTERVAL_S),
            timeout_s=float(data.get("timeout_s") or _DEFAULT_TIMEOUT_S),
            start_period_s=float(data.get("start_period_s", 0.0)),
            start_interval_s=float(data.get("start_interval_s") or _DEFAULT_START_INTERVAL_S),
            retries=int(data.get("retries") or _DEFAULT_RETRIES),
        )


@dataclass(frozen=True)
class MainService:
    """What a compose file gives ``main``: the one sandbox of a main-only task, or the agents' sandbox of a stack.

    ``command`` is compose's (Harbor's base ``sh -c "sleep infinity"`` when the file sets none), ``entrypoint``
    compose's (None: the image's; ``()``: cleared); :meth:`argv` combines them with the image's ``ENTRYPOINT`` as
    compose does.  ``healthcheck`` is compose's (None: the image's, if it has one, unless ``healthcheck_disabled``).
    ``unsupported`` lists the keys that platoon cannot apply to ``main`` yet, which the task loader refuses."""

    command: tuple[str, ...] = BASE_COMMAND
    entrypoint: tuple[str, ...] | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    working_dir: str | None = None
    healthcheck: ServiceHealthcheck | None = None
    healthcheck_disabled: bool = False
    binds: tuple[Bind, ...] = ()
    volumes: tuple[str, ...] = ()  # targets of volumes that mask no bind: directories of main's own
    cpus: float | None = None
    memory_mb: int | None = None
    unsupported: tuple[str, ...] = ()

    def argv(self, image_entrypoint: Sequence[str] | None) -> list[str]:
        """Main's command, as compose runs it (:func:`resolve_argv`): compose's ``entrypoint`` (else the image's) and
        ``command``, which main always has (Harbor's base gives it one), so the image's ``CMD`` never counts."""

        return resolve_argv(
            image_entrypoint=image_entrypoint, image_cmd=None, entrypoint=self.entrypoint, command=self.command
        )

    def init(self, image_entrypoint: Sequence[str] | None) -> list[str] | None:
        """The command a session starts in ``main`` once it runs, its init; None when ``main`` only idles, as with
        Harbor's base command (``sleep infinity``, ``tail -f /dev/null``, under a shell or not)."""

        argv = self.argv(image_entrypoint)
        return None if is_idle(argv) else argv


@dataclass(frozen=True)
class ComposePlan:
    """A task's compose file as platoon runs it, in the form a lock records (``compose`` of a locked task)."""

    data: Mapping[str, Any]

    @property
    def kind(self) -> StackKind:
        return self.data["kind"]

    @property
    def services(self) -> Mapping[str, Mapping[str, Any]]:
        return self.data.get("services") or {}

    @property
    def deferred(self) -> tuple[str, ...]:
        """The variables a session resolves when it opens, as ``host:NAME`` or ``task:NAME``."""

        return tuple(self.data.get("deferred") or ())

    def to_json(self) -> dict[str, Any]:
        return dict(self.data)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ComposePlan:
        if not isinstance(data, Mapping) or data.get("kind") not in STACK_KINDS:
            raise ValueError(f"not a compose plan: {data!r}")
        return cls(dict(data))

    def main(self) -> MainService | None:
        """``main``'s settings, for a main-only plan or a stack's (``sidecars``); None for the other kinds.  In a stack,
        what one sandbox cannot apply but a stack provider does (``cap_add``, ``shm_size``, ``ulimits``, ``tmpfs``, a
        shared network, internal networks) goes to the stack, not to :attr:`MainService.unsupported`."""

        if self.kind not in ("main_only", "sidecars"):
            return None
        return _main_service(
            self.services.get(MAIN) or {}, egress=self.data.get("egress", True), stack=self.kind == "sidecars"
        )

    @property
    def egress(self) -> bool:
        """Whether the stack's networks reach beyond it: False when every network its services join is
        ``internal``."""

        return bool(self.data.get("egress", True))

    @property
    def privileges(self) -> tuple[str, ...]:
        """What its services ask for that only the docker provider grants (``judge.privileged``, ``main.cap_add
        NET_ADMIN``): planned so by a fetch with ``refuse: {privileges: false}``, refused by every other."""

        return tuple(self.data.get("privileges") or ())

    @property
    def external(self) -> dict[str, str]:
        """The services an overlay marks as served by a node service (``external``), with that node service."""

        return {name: str(service["external"]) for name, service in self.services.items() if service.get("external")}

    def dependency_cycle(self) -> list[str] | None:
        """Services that depend on one another in a cycle, as the names along it (the first again at its end), which
        compose refuses to start; None when there is none."""

        return dependency_cycle({name: set(service.get("depends_on") or {}) for name, service in self.services.items()})


# --- the compose file ------------------------------------------------------------------------------------------


def compose_path(task_dir: str | Path) -> Path | None:
    """The task's compose file, ``environment/docker-compose.yaml``, when it has one."""

    path = Path(task_dir) / ENVIRONMENT_DIR / COMPOSE_FILE
    return path if path.is_file() else None


def unread_compose_files(task_dir: str | Path) -> list[str]:
    """The files of ``environment/`` that compose would read but Harbor does not (``compose.yaml``, ...)."""

    directory = Path(task_dir) / ENVIRONMENT_DIR
    return [f"{ENVIRONMENT_DIR}/{name}" for name in UNREAD_COMPOSE_FILES if (directory / name).is_file()]


class _ComposeLoader(yaml.SafeLoader):
    """YAML as compose reads it, with YAML 1.2's scalars: ``no``, ``on`` and ``22:22`` are strings (YAML 1.1 makes them
    booleans and a base-60 number), and so are dates."""


_ComposeLoader.yaml_implicit_resolvers = {
    first: [(tag, pattern) for tag, pattern in resolvers if not tag.endswith((":bool", ":int", ":float", ":timestamp"))]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_ComposeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)
_ComposeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:int", re.compile(r"^(?:[-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)$"), list("-+0123456789")
)
_ComposeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"^(?:[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"
    ),
    list("-+0123456789."),
)


def _construct_int(loader: yaml.SafeLoader, node: yaml.ScalarNode) -> int:
    text = loader.construct_scalar(node)
    base = {"0o": 8, "0x": 16}.get(text[:2], 10)
    return int(text[2:] if base != 10 else text, base)


_ComposeLoader.add_constructor("tag:yaml.org,2002:int", _construct_int)


def read_document(path: Path) -> dict[str, Any]:
    """A compose file's YAML document (anchors and merge keys resolved, YAML 1.2's scalars), or
    :class:`ComposeError`, also for a file whose aliases expand it beyond :data:`MAX_DOCUMENT_VALUES` values or into
    itself, and for one nested too deeply to read."""

    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        raise ComposeError("", f"the file cannot be read: {exc}") from exc
    loader = _ComposeLoader(text)  # a SafeLoader
    try:
        node = loader.get_single_node()
        if node is None:
            return {}
        _check_expansion(node)
        document = loader.construct_document(node)
    except yaml.YAMLError as exc:
        raise ComposeError("", f"the file is not YAML: {' '.join(str(exc).split())}") from exc
    except RecursionError:
        raise ComposeError("", "the file nests its values too deeply to read") from None
    finally:
        loader.dispose()
    if document is None:
        return {}
    if not isinstance(document, dict):
        raise ComposeError("", "the file must be a mapping")
    return document


def _check_expansion(root: yaml.Node) -> None:
    """Refuse a YAML document whose aliases expand it beyond :data:`MAX_DOCUMENT_VALUES` values, or whose anchor holds
    an alias of itself: its values are counted as planning walks them, each alias as what it stands for."""

    sizes: dict[int, int] = {}
    open_nodes: set[int] = set()

    def size(node: yaml.Node) -> int:
        key = id(node)
        if key in sizes:
            return sizes[key]
        if key in open_nodes:
            raise ComposeError("", "the file has an anchor that holds an alias of itself")
        open_nodes.add(key)
        total = 1
        if isinstance(node, yaml.SequenceNode):
            total += sum(size(item) for item in node.value)
        elif isinstance(node, yaml.MappingNode):
            total += sum(size(name) + size(value) for name, value in node.value)
        open_nodes.discard(key)
        if total > MAX_DOCUMENT_VALUES:
            raise ComposeError("", f"the file's aliases expand it to more than {MAX_DOCUMENT_VALUES:,} values")
        sizes[key] = total
        return total

    size(root)


def stack_kind(document: Mapping[str, Any]) -> StackKind:
    """``trivial``, ``main_only`` or ``sidecars``, from a compose document's services as written, before any check
    (a plan's kind counts only the services that start): services besides ``main`` make it ``sidecars``, a ``main``
    with keys ``main_only``.  A file whose services cannot be read counts as ``sidecars``."""

    services = document.get("services") or {}
    if not isinstance(services, Mapping):
        return "sidecars"
    if any(name != MAIN for name in services):
        return "sidecars"
    return "main_only" if services.get(MAIN) else "trivial"


def plan_compose(
    task_dir: str | Path,
    *,
    docker_image: str | None = None,
    env: Mapping[str, str] | None = None,
    cpus: float | None = None,
    memory_mb: int | None = None,
    edits: Sequence[ComposeEdit] = (),
    host_env: Iterable[str] = (),
    with_keys: bool = True,
    privileges: bool = False,
) -> ComposePlan | None:
    """The stack plan of the task at ``task_dir`` (None when it has no compose file), from its compose file as Harbor
    layers it: over a ``main`` built from ``environment/`` (prebuilt from ``docker_image`` when given), interpolated
    with Harbor's variables, the task's ``env`` and the listed ``host_env``, after the overlays' ``edits``.  ``cpus``
    and ``memory_mb`` are the task's ``[environment]`` resources (``CPUS``, ``MEMORY``).  ``with_keys`` plans the
    build keys of the services' build contexts (a fetch records them; a session needs none).  ``privileges`` plans
    ``privileged``, and a ``cap_add`` of a capability that a rootless service's root does not hold, as the plan's
    :attr:`~ComposePlan.privileges`, where both are refused otherwise.

    Raises :class:`ComposeError` for a file outside the subset, naming the service and key."""

    task_dir = Path(task_dir)
    path = compose_path(task_dir)
    if path is None:
        return None
    document = read_document(path)
    _check_top_level(document)
    services = document.get("services")
    if services is None:
        services = {}
    if not isinstance(services, Mapping):
        raise ComposeError("services", "must map service names to their settings")
    services = {str(name): ({} if value is None else value) for name, value in services.items()}
    for name, value in services.items():
        if not _SERVICE_NAME.match(name):
            raise ComposeError(name, "is not a valid service name")
        if not isinstance(value, Mapping):
            raise ComposeError(name, "must be a mapping of settings")
    document = {**document, "services": {name: dict(value) for name, value in services.items()}}
    applied, external = _apply_edits(document, edits)
    for key, value in document.items():
        if key == "services":
            for name, service in value.items():
                _check_reserved(service, name)
        else:
            _check_reserved(value, key)
    planner = _Planner(
        task_dir,
        docker_image=docker_image,
        env=dict(env or {}),
        cpus=cpus,
        memory_mb=memory_mb,
        host_env=frozenset(host_env),
        with_keys=with_keys,
        privileges=privileges,
    )
    return planner.plan(document, external=external, edits=applied)


def _check_reserved(value: Any, where: str) -> None:
    """Refuse the private-use characters U+E000 to U+E007 in what a task or an overlay writes (``where`` names it):
    platoon marks the variables a session resolves with them."""

    if isinstance(value, str):
        if _RESERVED.search(value):
            raise ComposeError(where, "holds a private-use character (U+E000 to U+E007), which platoon reserves")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _check_reserved(key, where)
            _check_reserved(item, f"{where}.{key}" if where else str(key))
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_reserved(item, where)


def _check_top_level(document: Mapping[str, Any]) -> None:
    for key in document:
        if not isinstance(key, str):
            raise ComposeError("", f"top-level key {key!r} is not a name")
        if key.startswith("x-") or key in TOP_LEVEL_KEYS:
            continue
        raise ComposeError(key, "is a top-level key outside the compose subset platoon runs")


def _apply_edits(document: dict[str, Any], edits: Sequence[ComposeEdit]) -> tuple[list[str], dict[str, str]]:
    """Apply overlays' service edits to ``document`` in place; what was applied, and the services marked external."""

    services: dict[str, dict[str, Any]] = document["services"]
    applied: list[str] = []
    external: dict[str, str] = {}
    for edit in edits:
        if edit.service not in services:
            if edit.service != MAIN:
                raise ComposeEditError(edit.service, f"overlay {edit.overlay} edits a service the compose file lacks")
            services[MAIN] = {}
        service = services[edit.service]
        for dotted in edit.unset:
            _unset(service, dotted.split("."))
        for key, value in edit.replace.items():
            service[key] = _plain(value)
        _merge(service, _plain(edit.set))
        if edit.external:
            if edit.service == MAIN:
                raise ComposeEditError(MAIN, f"overlay {edit.overlay} marks main external; main is the agents' sandbox")
            external[edit.service] = edit.external
        applied.append(edit.describe())
    return applied, external


def _unset(mapping: dict[str, Any], path: Sequence[str]) -> None:
    head, *rest = path
    if not rest:
        mapping.pop(head, None)
    elif isinstance(mapping.get(head), dict):
        _unset(mapping[head], rest)


def _merge(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    """Merge ``source`` into ``target`` as compose merges an override: mappings key by key (a list of ``NAME=value``,
    such as an ``environment``, as the mapping it means), other values replaced."""

    for key, value in source.items():
        current = target.get(key)
        if isinstance(value, Mapping) and isinstance(current, list) and all(isinstance(item, str) for item in current):
            pairs = (item.partition("=") for item in current)
            current = target[key] = {name: rest if sep else None for name, sep, rest in pairs}
        if isinstance(value, Mapping) and isinstance(current, dict):
            _merge(current, value)
        else:
            target[key] = value


def _plain(value: Any) -> Any:
    """A TOML value as YAML would load it (tables as dicts, arrays as lists)."""

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


# --- interpolation ----------------------------------------------------------------------------------------------


class _Unset:
    pass


UNSET = _Unset()


def interpolate(
    text: str, variables: Mapping[str, str | _Unset], *, where: str = "", notes: list[str] | None = None
) -> str:
    """``text`` with compose's interpolation applied: ``variables`` maps names to values (:data:`UNSET` and absent
    names are unset).  An unset variable without a default is blank, and ``notes`` (when given) records it; a ``$``
    that starts no name stays as it is, as in compose; an invalid ``${...}``, or a ``${NAME?error}`` of an unset
    variable, raises :class:`ComposeError`."""

    out, _ = _interpolate(text, 0, variables, where, notes, closing=False)
    return out


def _interpolate(
    text: str,
    index: int,
    variables: Mapping[str, str | _Unset],
    where: str,
    notes: list[str] | None,
    *,
    closing: bool,
) -> tuple[str, int]:
    """Interpolate from ``index`` to the end, or with ``closing`` to the ``}`` that closes a default, whose index is
    returned."""

    out: list[str] = []
    while index < len(text):
        char = text[index]
        if closing and char == "}":
            return "".join(out), index
        if char != "$":
            out.append(char)
            index += 1
            continue
        following = text[index + 1 : index + 2]
        if following == "$":
            out.append("$")
            index += 2
            continue
        if following == "{":
            value, index = _braced(text, index + 2, variables, where, notes)
            out.append(value)
            continue
        match = _NAME.match(text, index + 1)
        if match is None:  # compose leaves a $ that starts no name as it is ($?, $1, $(...))
            out.append(char)
            index += 1
            continue
        out.append(_lookup(match.group(0), variables, where, notes))
        index = match.end()
    if closing:
        raise ComposeError(where, f"unterminated ${{...}} in {text!r}")
    return "".join(out), index


def _braced(
    text: str, index: int, variables: Mapping[str, str | _Unset], where: str, notes: list[str] | None
) -> tuple[str, int]:
    """``${...}`` whose name starts at ``index``; the value and the index after the closing ``}``."""

    match = _NAME.match(text, index)
    if match is None:
        raise ComposeError(where, f"invalid interpolation format in {text!r}")
    name, index = match.group(0), match.end()
    if index < len(text) and text[index] == "}":
        return _lookup(name, variables, where, notes), index + 1
    operator = text[index : index + 2] if text[index : index + 2] in (":-", ":?", ":+") else text[index : index + 1]
    if operator not in (":-", ":?", ":+", "-", "?", "+"):
        raise ComposeError(where, f"invalid interpolation format in {text!r}")
    word, index = _interpolate(text, index + len(operator), variables, where, notes, closing=True)
    value = variables.get(name, UNSET)
    if isinstance(value, str) and _BARE_DEFERRED.fullmatch(value):  # only the session knows it: it applies the form
        if _RESERVED.search(word):
            raise ComposeError(
                where, f"${{{name}{operator}...}} names in its word a variable that a session resolves too"
            )
        return f"{value[:-1]}{_OPERATORS[operator]}{word}{_DEFER_CLOSE}", index + 1
    result = _operator_value(operator, None if isinstance(value, _Unset) else value, word)
    if result is None:
        raise ComposeError(where, f"required variable {name} is missing a value: {word}")
    return result, index + 1


def _operator_value(operator: str, value: str | None, word: str) -> str | None:
    """``${NAME<operator>word}`` as compose evaluates it, for a variable whose value is ``value`` (None: unset); None
    for a required one (``:?``, ``?``) that is missing.  The forms with a colon also count an empty value as
    missing."""

    missing = (value is None or value == "") if operator.startswith(":") else value is None
    if operator in (":-", "-"):
        return word if missing else value
    if operator in (":+", "+"):
        return "" if missing else word
    return None if missing else value


def _lookup(name: str, variables: Mapping[str, str | _Unset], where: str, notes: list[str] | None) -> str:
    value = variables.get(name, UNSET)
    if isinstance(value, _Unset):
        if notes is not None:
            note = f"{name} is not set: blank"
            if note not in notes:
                notes.append(note)
        return ""
    return value


def resolve_deferred(value: str, task_env: Mapping[str, str], host_env: Mapping[str, str]) -> str:
    """``value`` with the variables a plan left for the session (a listed host variable, a task variable that is a
    host template) resolved, under the form the compose file gave each (``${NAME:-default}``, ``${NAME:+value}``,
    ...): a host variable from ``host_env``, a task variable from the task's ``env`` resolved against it, as a task's
    ``[environment].env`` resolves (``${NAME}``, ``${NAME:-default}``).  ``KeyError`` names a variable that is unset
    where the file gives it no default, or that a ``${NAME:?error}`` requires."""

    def replace(match: re.Match[str]) -> str:
        kind, name, mark, word = match.groups()
        current = host_env.get(name) if kind == "host" else _task_value(task_env.get(name, ""), host_env)
        if mark is None:
            if current is None:
                raise KeyError(name)
            return current
        result = _operator_value(_OPERATOR_OF[mark], current, word)
        if result is None:
            raise KeyError(f"{name}: {word}" if word else name)
        return result

    return _DEFERRED.sub(replace, value)


def _task_value(text: str, host_env: Mapping[str, str]) -> str:
    """A task's ``[environment].env`` value resolved against ``host_env``, as a task's environment resolves:
    ``${NAME}`` and ``${NAME:-default}`` from the host, anything else as written; ``KeyError`` names an unset
    variable without a default."""

    template = _HOST_TEMPLATE.fullmatch(text)
    if template is None:
        return text
    if template.group(1) in host_env:
        return host_env[template.group(1)]
    if template.group(2) is not None:
        return template.group(2)
    raise KeyError(template.group(1))


def has_deferred(value: Any) -> bool:
    return isinstance(value, str) and _DEFERRED.search(value) is not None


# --- planning ---------------------------------------------------------------------------------------------------


class _Planner:
    def __init__(
        self,
        task_dir: Path,
        *,
        docker_image: str | None,
        env: dict[str, str],
        cpus: float | None,
        memory_mb: int | None,
        host_env: frozenset[str],
        with_keys: bool,
        privileges: bool,
    ) -> None:
        self.task_dir = task_dir
        self.docker_image = docker_image
        self.with_keys = with_keys
        self.privileges = privileges  # plan privileged and the other capabilities, for the docker provider
        self.granted: list[str] = []  # those the services ask for, as "<service>.privileged", "<service>.cap_add X"
        self.notes: list[str] = []
        self.ignored: list[str] = []
        self.deferred: set[str] = set()
        _check_reserved(env, "[environment].env")
        _check_reserved(docker_image, "[environment].docker_image")
        variables: dict[str, str | _Unset] = {}
        for name in host_env:
            variables[name] = f"{_DEFER_OPEN}host.{name}{_DEFER_CLOSE}"
        for name, value in env.items():  # the task's [environment].env; a host template resolves at session open
            variables[name] = f"{_DEFER_OPEN}task.{name}{_DEFER_CLOSE}" if "${" in value else value
        # What a plan may leave for a session: what the variables above defer, and nothing else.
        self.may_defer = {
            f"{match.group(1)}:{match.group(2)}"
            for value in variables.values()
            if isinstance(value, str) and (match := _BARE_DEFERRED.fullmatch(value))
        }
        variables.update(
            {
                "CONTEXT_DIR": f"{_TASK_ROOT}/{ENVIRONMENT_DIR}",
                "MAIN_IMAGE_NAME": _MAIN_IMAGE,
                **({"PREBUILT_IMAGE_NAME": docker_image} if docker_image else {}),
                **({"CPUS": _number(cpus)} if cpus is not None else {}),
                **({"MEMORY": f"{memory_mb}M"} if memory_mb is not None else {}),
                **{f"HOST_{_LOG_VARIABLE[name]}_PATH": f"{_LOGS_ROOT}/{name}" for name in LOG_DIRS},
                **{f"ENV_{_LOG_VARIABLE[name]}_PATH": path for name, path in LOG_DIRS.items()},
            }
        )
        self.variables = variables

    # The plan --------------------------------------------------------------------------------------------------

    def plan(self, document: dict[str, Any], *, external: Mapping[str, str], edits: list[str]) -> ComposePlan:
        raw_services: dict[str, dict[str, Any]] = document["services"]
        raw_services.setdefault(MAIN, {})
        if raw_services[MAIN].get("profiles"):
            raise ComposeError("main.profiles", "would keep main from starting; Harbor starts no profile")
        started = {name: raw for name, raw in raw_services.items() if not raw.get("profiles")}
        for name in sorted(set(raw_services) - set(started)):
            self.notes.append(f"{name} has profiles: not started, as Harbor starts no profile")
        networks = self._networks(document.get("networks"))
        volumes = self._volumes(document.get("volumes"))
        services = {}
        for name, raw in started.items():
            if name in external:
                services[name] = {"external": external[name]}
                continue
            services[name] = self._service(name, raw, networks=networks, volumes=volumes, started=started)
        egress = self._check_networks(services, networks)
        _check_ports(services)
        if len(services) > 1:
            kind: StackKind = "sidecars"
        elif services[MAIN] != {"command": list(BASE_COMMAND)}:
            kind = "main_only"
        else:
            kind = "trivial"  # main restates Harbor's base, or holds only keys that platoon ignores
        data: dict[str, Any] = {"rule": COMPOSE_RULE, "kind": kind, "services": services}
        if networks:
            data["networks"] = networks
        if volumes:
            data["volumes"] = volumes
        if not egress:
            data["egress"] = False
        for key, values in (("ignored", self.ignored), ("notes", self.notes), ("edits", edits)):
            if values:
                data[key] = list(values)
        if self.granted:
            data["privileges"] = list(self.granted)
        if self.deferred:
            data["deferred"] = sorted(self.deferred)
        return ComposePlan(data)

    def _service(
        self,
        name: str,
        raw: Mapping[str, Any],
        *,
        networks: Mapping[str, Any],
        volumes: Mapping[str, Any],
        started: Mapping[str, Any],
    ) -> dict[str, Any]:
        where = name
        service: dict[str, Any] = {}
        for key in raw:
            if not isinstance(key, str):
                raise ComposeError(where, f"key {key!r} is not a name")
            if key.startswith("x-") or key in SERVICE_KEYS or (key == "privileged" and self.privileges):
                continue
            if key in IGNORED_KEYS:
                self._ignored(name, key, raw[key])
                continue
            raise ComposeError(f"{name}.{key}", REFUSED_KEYS.get(key, "is outside the compose subset platoon runs"))
        values = {key: self._interpolated(raw[key], f"{name}.{key}") for key in raw if key in SERVICE_KEYS}
        if name == MAIN:
            service.update(self._main_image(values))
            service["command"] = list(BASE_COMMAND)
        else:
            service.update(self._sidecar_image(name, values))
        if "command" in values:
            service["command"] = _argv(values["command"], f"{name}.command")
        if "entrypoint" in values:
            service["entrypoint"] = _argv(values["entrypoint"], f"{name}.entrypoint")
        environment = self._environment(name, values)
        if environment:
            service["environment"] = environment
        if values.get("working_dir"):
            service["working_dir"] = _absolute(
                _string(values["working_dir"], f"{name}.working_dir"), f"{name}.working_dir"
            )
        mounts = self._mounts(name, values, volumes)
        if mounts:
            service["volumes"] = mounts
        if "healthcheck" in values:
            service["healthcheck"] = _healthcheck(values["healthcheck"], f"{name}.healthcheck")
        expose = _ports(values.get("expose"), f"{name}.expose", published=False) + _ports(
            values.get("ports"), f"{name}.ports", published=True
        )
        if expose:
            service["expose"] = list(dict.fromkeys(expose))
        if "ports" in values:
            self.notes.append(f"{name}.ports: kept as expose; platoon publishes no port on the host")
        if "depends_on" in values:
            dependencies = self._depends_on(name, values["depends_on"], started)
            if dependencies:
                service["depends_on"] = dependencies
        self._network_settings(name, values, networks, started, service)
        if "hostname" in values:
            service["hostname"] = _string(values["hostname"], f"{name}.hostname")
        resources = _resources(name, values)
        if resources:
            service["resources"] = resources
        if "shm_size" in values:
            service["shm_mb"] = _size_mb(values["shm_size"], f"{name}.shm_size")
        if "ulimits" in values:
            service["ulimits"] = _ulimits(values["ulimits"], f"{name}.ulimits")
        if "cap_add" in values:
            service["cap_add"] = _capabilities(values["cap_add"], f"{name}.cap_add", held_only=not self.privileges)
            self.granted += [f"{name}.cap_add {cap}" for cap in service["cap_add"] if cap not in HELD_CAPABILITIES]
        if "privileged" in raw:  # planned with privileges only: the keys' check refused it otherwise
            privileged = self._interpolated(raw["privileged"], f"{name}.privileged")
            if not isinstance(privileged, bool):
                raise ComposeError(f"{name}.privileged", "must be true or false")
            if privileged:
                service["privileged"] = True
                self.granted.append(f"{name}.privileged")
        if "stop_signal" in values:
            service["stop_signal"] = _string(values["stop_signal"], f"{name}.stop_signal")
        if "tmpfs" in values:
            tmpfs = values["tmpfs"]
            entries = [tmpfs] if isinstance(tmpfs, str) else tmpfs
            if not isinstance(entries, list) or not all(isinstance(entry, str) for entry in entries):
                raise ComposeError(f"{name}.tmpfs", "must be a path or a list of paths")
            service.setdefault("volumes", []).extend(
                {"type": "tmpfs", "target": _absolute(entry.split(":", 1)[0], f"{name}.tmpfs")} for entry in entries
            )
        self._check_deferred(name, service)
        return service

    def _ignored(self, service: str, key: str, value: Any) -> None:
        if key == "platform":
            value = self._interpolated(value, f"{service}.platform")
            if not isinstance(value, str) or value.lower() not in _host_platforms():
                raise ComposeError(f"{service}.platform", f"asks for {value!r}, which is not this host's platform")
        self.ignored.append(f"{service}.{key}")

    def _interpolated(self, value: Any, where: str) -> Any:
        if isinstance(value, str):
            return interpolate(value, self.variables, where=where, notes=self.notes)
        if isinstance(value, list):
            return [self._interpolated(item, where) for item in value]
        if isinstance(value, dict):
            return {key: self._interpolated(item, f"{where}.{key}") for key, item in value.items()}
        return value

    # Images and builds ------------------------------------------------------------------------------------------

    def _main_image(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Main's image is the task's (its build plan, or its docker_image): compose may only restate it."""

        image = values.get("image")
        build = values.get("build")
        if self.docker_image:
            if build is not None:
                raise ComposeError(
                    "main.build", "builds main, whose image is the task's docker_image; platoon runs that image"
                )
            if image is not None and image not in (self.docker_image, _MAIN_IMAGE):
                raise ComposeError(
                    "main.image", f"names {image!r}, which is not the task's docker_image ({self.docker_image})"
                )
            return {}
        if build is not None:
            context, dockerfile, args = _build(build, "main.build")
            if self._context(context, "main.build.context") != ENVIRONMENT_DIR or dockerfile != "Dockerfile" or args:
                raise ComposeError(
                    "main.build",
                    "builds main from another context, Dockerfile or arguments than environment/Dockerfile",
                )
        elif image is not None and image != _MAIN_IMAGE:
            # Harbor's base builds main from environment/: compose builds it and tags it with this name.
            if not (self.task_dir / ENVIRONMENT_DIR / "Dockerfile").is_file():
                raise ComposeError(
                    "main.image",
                    f"names {image!r}, but Harbor builds main from environment/, which has no Dockerfile (a task "
                    "names a prebuilt image with [environment].docker_image)",
                )
            self.notes.append(f"main.image: {image!r} names the image built from environment/")
        return {}

    def _sidecar_image(self, name: str, values: Mapping[str, Any]) -> dict[str, Any]:
        image = values.get("image")
        build = values.get("build")
        if image is not None and not isinstance(image, str):
            raise ComposeError(f"{name}.image", "must be an image reference")
        if build is None:
            if not image:
                raise ComposeError(name, "has neither an image nor a build")
            if image == _MAIN_IMAGE:
                return {"image": {"main": True}}
            if _MAIN_IMAGE in image:
                raise ComposeError(f"{name}.image", "names an image that only a build of it would make")
            return {"image": image}
        context, dockerfile, args = _build(build, f"{name}.build")
        relative = self._context(context, f"{name}.build.context")
        plan: dict[str, Any] = {"context": relative, "dockerfile": dockerfile}
        if args:
            plan["args"] = args
        if self.with_keys:
            directory = self.task_dir / relative
            try:
                planned = plan_dockerfile(directory / dockerfile, directory, whole=True, build_args=args)
            except (DockerfileError, OSError) as exc:
                raise ComposeError(f"{name}.build", f"{relative}/{dockerfile}: {exc}") from exc
            plan["key"] = planned.key
            plan["base"] = list(planned.base_images)
            if planned.config:
                plan["config"] = dict(planned.config)
        return {"build": plan}

    def _context(self, context: str, where: str) -> str:
        """A build context as a path relative to the task (``environment/mcp-server``); it must be a directory of the
        task's ``environment/``, which Harbor's compose reads relative to."""

        relative = self._task_path(context, where, base=ENVIRONMENT_DIR)
        if relative is None:
            raise ComposeError(where, f"{context!r} is not a directory of the task")
        if not (relative == ENVIRONMENT_DIR or relative.startswith(f"{ENVIRONMENT_DIR}/")):
            raise ComposeError(where, f"{relative}/ is outside environment/, which holds a task's build contexts")
        if not (self.task_dir / relative).is_dir():
            raise ComposeError(where, f"{relative}/ does not exist")
        return relative

    def _task_path(self, path: str, where: str, *, base: str) -> str | None:
        """``path`` (absolute through ``CONTEXT_DIR``, or relative to the compose file's directory) relative to the
        task, ``.`` for the task itself; None for a path outside it.  A path that climbs out of it is refused."""

        if has_deferred(path):
            return None
        if path.startswith(f"{_TASK_ROOT}/") or path == _TASK_ROOT:
            absolute = posixpath.normpath(path)
        elif path.startswith("/") or path.startswith("~"):
            return None
        else:
            absolute = posixpath.normpath(posixpath.join(_TASK_ROOT, base, path))
        if absolute == _TASK_ROOT:
            return "."
        if not absolute.startswith(f"{_TASK_ROOT}/"):
            raise ComposeError(where, f"{path!r} climbs out of the task's directory")
        return absolute.removeprefix(f"{_TASK_ROOT}/")

    # Environment ------------------------------------------------------------------------------------------------

    def _environment(self, name: str, values: Mapping[str, Any]) -> dict[str, str]:
        environment: dict[str, str] = {}
        for file in _list_of(values.get("env_file"), f"{name}.env_file"):
            environment.update(self._env_file(name, file, environment))
        raw = values.get("environment")
        if raw is None:
            return environment
        where = f"{name}.environment"
        items: list[tuple[str, Any]]
        if isinstance(raw, Mapping):
            items = [(str(key), value) for key, value in raw.items()]
        elif isinstance(raw, list):
            items = []
            for entry in raw:
                if not isinstance(entry, str):
                    raise ComposeError(where, f"entries must be NAME=value, got {entry!r}")
                key, sep, value = entry.partition("=")
                items.append((key, value if sep else None))
        else:
            raise ComposeError(where, "must be a mapping or a list of NAME=value")
        for key, value in items:
            if value is None:  # passed through from compose's own environment, which holds only these variables
                known = self.variables.get(key, UNSET)
                if isinstance(known, _Unset):
                    self.notes.append(f"{where}.{key} passes a host variable that no overlay lists: not set")
                else:
                    environment[key] = known
                continue
            if isinstance(value, bool):
                value = "true" if value else "false"
            if not isinstance(value, (str, int, float)):
                raise ComposeError(f"{where}.{key}", f"must be a string or a number, got {value!r}")
            environment[key] = value if isinstance(value, str) else _number(value)
        return environment

    def _env_file(self, name: str, file: Any, earlier: Mapping[str, str]) -> dict[str, str]:
        """An ``env_file``'s variables, as compose reads them (:func:`_dotenv`): a value interpolated (but a
        single-quoted one) with compose's variables, the variables of ``earlier`` files, then this file's own above it;
        a name alone takes compose's variable of that name."""

        where = f"{name}.env_file"
        required = True
        if isinstance(file, Mapping):
            required = bool(file.get("required", True))
            file = file.get("path")
        if not isinstance(file, str) or not file:
            raise ComposeError(where, "must name files")
        relative = self._task_path(file, where, base=ENVIRONMENT_DIR)
        path = self.task_dir / relative if relative is not None else None
        if path is None or not path.is_file():
            if required:
                raise ComposeError(where, f"{file!r} is not a file of the task")
            return {}
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError) as exc:
            raise ComposeError(where, f"{relative} cannot be read: {exc}") from exc
        _check_reserved(text, f"{where}: {relative}")
        environment: dict[str, str] = {}
        for key, value, quote in _dotenv(text, f"{where}: {relative}"):
            if value is None:
                known = earlier[key] if key in earlier else self.variables.get(key, UNSET)
                if isinstance(known, str):
                    environment[key] = known
                continue
            if quote != "'":
                value = interpolate(value, {**environment, **self.variables, **earlier}, where=where, notes=self.notes)
            environment[key] = value
        return environment

    # Volumes ----------------------------------------------------------------------------------------------------

    def _volumes(self, raw: Any) -> dict[str, Any]:
        if raw is None:
            return {}
        if not isinstance(raw, Mapping):
            raise ComposeError("volumes", "must map volume names to their settings")
        volumes: dict[str, Any] = {}
        for name, settings in raw.items():
            where = f"volumes.{name}"
            settings = self._interpolated(settings or {}, where)
            if not isinstance(settings, Mapping):
                raise ComposeError(where, "must be a mapping")
            if settings.get("external"):
                raise ComposeError(f"{where}.external", "names a volume outside the stack, which platoon cannot share")
            if unknown := sorted(set(settings) - _VOLUME_KEYS - {"external"}):
                raise ComposeError(f"{where}.{unknown[0]}", "is outside the compose subset platoon runs")
            if settings.get("driver") not in (None, "local"):
                raise ComposeError(f"{where}.driver", f"{settings['driver']!r} is not the local driver")
            volumes[str(name)] = {}
        return volumes

    def _mounts(self, name: str, values: Mapping[str, Any], volumes: Mapping[str, Any]) -> list[dict[str, Any]]:
        raw = values.get("volumes")
        if raw is None:
            return []
        if not isinstance(raw, list):
            raise ComposeError(f"{name}.volumes", "must be a list")
        mounts = [self._mount(name, entry, volumes) for entry in raw]
        targets = [mount["target"] for mount in mounts]
        if duplicate := next((target for target in targets if targets.count(target) > 1), None):
            raise ComposeError(f"{name}.volumes", f"mounts two volumes at {duplicate}")
        binds = [mount for mount in mounts if mount["type"] == "bind"]
        for mount in mounts:  # volumes below a bind mask what it binds there
            if mount["type"] == "bind":
                continue
            outer = [bind for bind in binds if _below(mount["target"], bind["target"])]
            if outer:
                bind = max(outer, key=lambda each: len(each["target"]))
                bind.setdefault("masks", []).append(posixpath.relpath(mount["target"], bind["target"]))
                mount["masks"] = bind["target"]
        return mounts

    def _mount(self, service: str, entry: Any, volumes: Mapping[str, Any]) -> dict[str, Any]:
        where = f"{service}.volumes"
        if isinstance(entry, str):
            kind, source, target, read_only, short = self._short_volume(entry, where)
        elif isinstance(entry, Mapping):
            kind, source, target, read_only, short = self._long_volume(entry, where)
        else:
            raise ComposeError(where, f"entries must be strings or mappings, got {entry!r}")
        if has_deferred(target):
            raise ComposeError(where, "mounts at a path that a host variable names, which only a session knows")
        target = _absolute(target, where)
        if kind == "volume":
            if source and source not in volumes:
                raise ComposeError(where, f"names the volume {source!r}, which the file does not declare")
            return {"type": "volume", "target": target, **({"name": source} if source else {})}
        if kind == "tmpfs":
            return {"type": "tmpfs", "target": target}
        return self._bind(service, source or "", target, read_only, short=short)

    def _short_volume(self, entry: str, where: str) -> tuple[str, str | None, str, bool, bool]:
        parts = entry.split(":")
        if len(parts) == 1:
            return "volume", None, parts[0], False, True
        if len(parts) > 3:
            raise ComposeError(where, f"cannot read the volume {entry!r}")
        source, target = parts[0], parts[1]
        modes = set(parts[2].split(",")) if len(parts) == 3 else set()
        if unknown := sorted(modes - {"ro", "rw", "z", "Z", "cached", "delegated", "consistent", "nocopy"}):
            raise ComposeError(where, f"{entry!r}: mode {unknown[0]!r} is outside the compose subset")
        read_only = "ro" in modes
        if not source:
            raise ComposeError(where, f"{entry!r} has no source (an unset variable?)")
        if _VOLUME_NAME.match(source) and not source.startswith("."):
            return "volume", source, target, read_only, True
        return "bind", source, target, read_only, True

    def _long_volume(self, entry: Mapping[str, Any], where: str) -> tuple[str, str | None, str, bool, bool]:
        if unknown := sorted(
            set(entry) - {"type", "source", "target", "read_only", "bind", "volume", "tmpfs", "consistency"}
        ):
            raise ComposeError(f"{where}.{unknown[0]}", "is outside the compose subset platoon runs")
        kind = entry.get("type", "volume")
        if kind not in ("bind", "volume", "tmpfs"):
            raise ComposeError(where, f"type {kind!r} is outside the compose subset")
        if (entry.get("volume") or {}).get("subpath"):
            raise ComposeError(where, "volume.subpath is outside the compose subset")
        target = entry.get("target")
        if not isinstance(target, str) or not target:
            raise ComposeError(where, "needs a target")
        source = entry.get("source")
        if source is not None and not isinstance(source, str):
            raise ComposeError(where, f"source must be a string, got {source!r}")
        creates = bool((entry.get("bind") or {}).get("create_host_path"))  # what the short syntax implies
        return str(kind), source, target, bool(entry.get("read_only")), creates

    def _bind(self, service: str, source: str, target: str, read_only: bool, *, short: bool) -> dict[str, Any]:
        where = f"{service}.volumes"
        bind: dict[str, Any] = {"type": "bind", "target": target, "read_only": read_only}
        normalized = posixpath.normpath(source) if source.startswith("/") else source
        if normalized.startswith(f"{_LOGS_ROOT}/") and normalized.removeprefix(f"{_LOGS_ROOT}/") in LOG_DIRS:
            bind["logs"] = normalized.removeprefix(f"{_LOGS_ROOT}/")
            return bind
        if service == MAIN and any(
            target == path or _below(target, path) or _below(path, target) for path in RESERVED_TARGETS
        ):
            raise ComposeError(where, f"binds {target}, where platoon puts the tests, the solution or the logs")
        if has_deferred(source):
            # Only a listed host variable names a host path, and only the path it names or one below it: never a task
            # variable (a host template the task chose), a default the file gives, or a path that climbs out.
            match = _BARE_DEFERRED.match(source)
            if match is not None and match.group(1) == "task":
                raise ComposeError(
                    where,
                    f"binds a host path through the task variable {match.group(2)}; only a host variable that an "
                    "overlay lists (compose_host_env) may name one",
                )
            below = source[match.end() :] if match is not None else ""
            if match is None or has_deferred(below) or ".." in below.split("/") or (below and below[0] != "/"):
                raise ComposeError(
                    where,
                    "binds a host path other than the one a listed host variable names, or a path below it (with no "
                    "default)",
                )
            if not read_only:
                raise ComposeError(where, "binds a host path writable; only read-only binds of host paths run")
            bind["host"] = source
            return bind
        relative = self._task_path(source, where, base=ENVIRONMENT_DIR)
        if relative is None:
            raise ComposeError(
                where,
                f"binds the host path {source!r}; platoon binds the task's own files, its log directories and the "
                "host paths an overlay's compose_host_env names",
            )
        bind["task"] = relative
        if not (self.task_dir / relative).exists():
            if not short:  # the long syntax creates no missing source, unless bind.create_host_path says so
                raise ComposeError(where, f"binds {relative}, which the task does not have")
            bind["missing"] = True
            self.notes.append(
                f"{where}: {relative} is not in the task: an empty directory is bound, as Docker makes one"
            )
        return bind

    # Health, dependencies, networks -----------------------------------------------------------------------------

    def _depends_on(self, name: str, raw: Any, started: Mapping[str, Any]) -> dict[str, Any]:
        where = f"{name}.depends_on"
        if isinstance(raw, list):
            raw = {str(service): {"condition": "service_started"} for service in raw}
        if not isinstance(raw, Mapping):
            raise ComposeError(where, "must be a list of services or a mapping")
        dependencies: dict[str, Any] = {}
        for service, settings in raw.items():
            settings = settings or {}
            if not isinstance(settings, Mapping):
                raise ComposeError(f"{where}.{service}", "must be a mapping")
            if unknown := sorted(set(settings) - {"condition", "required", "restart"}):
                raise ComposeError(f"{where}.{service}.{unknown[0]}", "is outside the compose subset platoon runs")
            condition = settings.get("condition", "service_started")
            if condition not in CONDITIONS:
                raise ComposeError(f"{where}.{service}", f"condition {condition!r} is not one of {CONDITIONS}")
            required = settings.get("required", True)
            if service not in started:
                if required:
                    raise ComposeError(where, f"depends on {service!r}, which is not started")
                self.notes.append(f"{where}: {service} is not started, and not required")
                continue
            if service == name:
                raise ComposeError(where, "depends on itself")
            dependencies[str(service)] = {"condition": condition, **({"required": False} if not required else {})}
        return dependencies

    def _networks(self, raw: Any) -> dict[str, Any]:
        if raw is None:
            return {}
        if not isinstance(raw, Mapping):
            raise ComposeError("networks", "must map network names to their settings")
        networks: dict[str, Any] = {}
        for name, settings in raw.items():
            where = f"networks.{name}"
            settings = self._interpolated(settings or {}, where)
            if not isinstance(settings, Mapping):
                raise ComposeError(where, "must be a mapping")
            if settings.get("external"):
                raise ComposeError(f"{where}.external", "names a network outside the stack")
            if unknown := sorted(set(settings) - _NETWORK_KEYS - {"external"}):
                raise ComposeError(
                    f"{where}.{unknown[0]}", "is outside the compose subset platoon runs (static addresses, drivers)"
                )
            if settings.get("driver") not in (None, "bridge"):
                raise ComposeError(f"{where}.driver", f"{settings['driver']!r} is not the bridge driver")
            networks[str(name)] = {"internal": True} if settings.get("internal") else {}
        return networks

    def _network_settings(
        self,
        name: str,
        values: Mapping[str, Any],
        networks: Mapping[str, Any],
        started: Mapping[str, Any],
        service: dict[str, Any],
    ) -> None:
        mode = values.get("network_mode")
        if mode is not None:
            where = f"{name}.network_mode"
            if not isinstance(mode, str):
                raise ComposeError(where, "must be a string")
            if mode.startswith("service:"):
                other = mode.removeprefix("service:")
                if other not in started or other == name:
                    raise ComposeError(where, f"shares the network of {other!r}, which is not another started service")
                service["network_mode"] = mode
            elif mode not in ("bridge", "default"):
                raise ComposeError(where, f"{mode!r}: every service of a stack shares one network of its own")
            if "networks" in values:
                raise ComposeError(where, "conflicts with networks")
            return
        raw = values.get("networks")
        if raw is None:
            return
        where = f"{name}.networks"
        if isinstance(raw, list):
            raw = {str(network): None for network in raw}
        if not isinstance(raw, Mapping):
            raise ComposeError(where, "must be a list or a mapping")
        attached: dict[str, Any] = {}
        for network, settings in raw.items():
            if network != "default" and network not in networks:
                raise ComposeError(where, f"names the network {network!r}, which the file does not declare")
            settings = settings or {}
            if not isinstance(settings, Mapping):
                raise ComposeError(f"{where}.{network}", "must be a mapping")
            if unknown := sorted(set(settings) - {"aliases", "priority"}):
                raise ComposeError(
                    f"{where}.{network}.{unknown[0]}", "is outside the compose subset (a static address?)"
                )
            aliases = _list_of(settings.get("aliases"), f"{where}.{network}.aliases")
            attached[str(network)] = {"aliases": [str(alias) for alias in aliases]} if aliases else {}
        service["networks"] = attached

    def _check_networks(self, services: Mapping[str, Mapping[str, Any]], networks: Mapping[str, Any]) -> bool:
        """Refuse networks that keep a service from ``main`` (one network namespace would let ``main`` reach it), and
        a mix of internal and other networks (egress is the whole stack's or none); whether the stack has egress."""

        def joined(name: str, seen: frozenset[str] = frozenset()) -> set[str]:
            service = services.get(name) or {}
            mode = service.get("network_mode", "")
            if mode.startswith("service:") and mode not in seen:
                return joined(mode.removeprefix("service:"), seen | {mode})
            return set(service.get("networks") or {"default": {}})

        main = joined(MAIN)
        for name, service in services.items():
            if name == MAIN or service.get("external"):
                continue
            if not joined(name) & main:
                raise ComposeError(
                    f"{name}.networks",
                    "keeps the service from main, which one network namespace for the stack would let main reach",
                )
        used = {network for name in services for network in joined(name)}
        internal = {network for network in used if (networks.get(network) or {}).get("internal")}
        if internal and internal != used:
            raise ComposeError(
                "networks", "mixes internal and other networks; a stack's egress is all of its services' or none"
            )
        return not internal

    def _check_deferred(self, name: str, service: Mapping[str, Any]) -> None:
        """Only environment values and host binds may hold a variable the session resolves; the placeholders of the
        task's directory and log directories only binds and build contexts."""

        for key, value in service.items():
            if key in ("environment", "volumes"):
                for item in _strings(value):
                    for match in _DEFERRED.finditer(item):
                        deferred = f"{match.group(1)}:{match.group(2)}"
                        if deferred not in self.may_defer:  # never one the planner did not defer itself
                            raise ComposeError(f"{name}.{key}", f"names {deferred}, which no overlay lists")
                        self.deferred.add(deferred)
                if key == "environment" and any(_TASK_ROOT in item or _LOGS_ROOT in item for item in _strings(value)):
                    raise ComposeError(f"{name}.environment", "names a host path of the task or its logs")
                continue
            for item in _strings(value):
                if has_deferred(item):
                    raise ComposeError(
                        f"{name}.{key}", "names a host variable, which only environment values and binds may"
                    )
                if _TASK_ROOT in item or _LOGS_ROOT in item:
                    raise ComposeError(
                        f"{name}.{key}", "names a host path of the task or its logs (CONTEXT_DIR, HOST_*)"
                    )


_LOG_VARIABLE = {"verifier": "VERIFIER_LOGS", "agent": "AGENT_LOGS", "artifacts": "ARTIFACTS"}


def _check_ports(services: Mapping[str, Mapping[str, Any]]) -> None:
    """Refuse two services that declare one port (``expose``, ``ports``): a stack's services share one loopback where
    platoon runs them as a pod, so the second could not listen."""

    owners: dict[str, str] = {}
    for name, service in services.items():
        for port in service.get("expose") or ():
            other = owners.setdefault(port, name)
            if other != name:
                raise ComposeError(
                    f"{name}.expose",
                    f"declares {port}, which {other} declares too; a stack's services share one loopback, as in a pod",
                )


def service_binds(data: Mapping[str, Any]) -> tuple[Bind, ...]:
    """The binds of a planned service (a plan's ``services[name]``), in the file's order."""

    return tuple(
        Bind(
            target=mount["target"],
            read_only=bool(mount.get("read_only")),
            task=mount.get("task"),
            logs=mount.get("logs"),
            host=mount.get("host"),
            masks=tuple(mount.get("masks") or ()),
            missing=bool(mount.get("missing")),
        )
        for mount in data.get("volumes") or ()
        if mount["type"] == "bind"
    )


def _main_service(data: Mapping[str, Any], *, egress: bool = True, stack: bool = False) -> MainService:
    binds = list(service_binds(data))
    volumes = [
        mount["target"]
        for mount in data.get("volumes") or ()
        if mount["type"] == "volume" and not mount.get("masks")  # one that masks a bind is part of its copy
    ]
    unsupported = []
    if not stack:  # a stack provider applies these, where its runtime can
        planned = {
            "cap_add": "cap_add",
            "privileged": "privileged",
            "shm_size": "shm_mb",
            "ulimits": "ulimits",
            "network_mode": "network_mode",
        }
        unsupported += [key for key, field in planned.items() if data.get(field)]  # as the compose file names them
        if any(mount["type"] == "tmpfs" for mount in data.get("volumes") or ()):
            unsupported.append("tmpfs")
    if any(bind.logs and bind.target != LOG_DIRS[bind.logs] for bind in binds):
        unsupported.append("volumes (a log directory bound at another path)")
    if any(_below(inner.target, outer.target) for inner in binds for outer in binds):
        unsupported.append("volumes (a bind inside another)")
    if not egress and not stack:
        unsupported.append("networks (internal only)")
    healthcheck = data.get("healthcheck")
    resources = data.get("resources") or {}
    return MainService(
        command=tuple(data.get("command", BASE_COMMAND)),
        entrypoint=tuple(data["entrypoint"]) if "entrypoint" in data else None,
        environment=dict(data.get("environment") or {}),
        working_dir=data.get("working_dir"),
        healthcheck=ServiceHealthcheck.from_json(healthcheck) if healthcheck and "test" in healthcheck else None,
        healthcheck_disabled=bool(healthcheck and healthcheck.get("disable")),
        binds=tuple(binds),
        volumes=tuple(volumes),
        cpus=resources.get("cpus"),
        memory_mb=resources.get("memory_mb"),
        unsupported=tuple(unsupported),
    )


# --- values -----------------------------------------------------------------------------------------------------


def _argv(value: Any, where: str) -> list[str]:
    """A ``command`` or ``entrypoint``: a list as written, a string split as compose splits it (shell words, no
    shell); ``[]`` and ``""`` give an empty one."""

    if isinstance(value, str):
        try:
            return shlex.split(value)
        except ValueError as exc:
            raise ComposeError(where, f"cannot split {value!r}: {exc}") from exc
    if value is None:
        return []
    if isinstance(value, list) and all(isinstance(item, (str, int, float)) for item in value):
        return [item if isinstance(item, str) else _number(item) for item in value]
    raise ComposeError(where, "must be a string or a list of strings")


def resolve_argv(
    *,
    image_entrypoint: Sequence[str] | None,
    image_cmd: Sequence[str] | None,
    entrypoint: Sequence[str] | None = None,
    command: Sequence[str] | None = None,
) -> list[str]:
    """A service's command as compose and Docker run it: compose's ``entrypoint`` replaces the image's
    ``ENTRYPOINT`` (an empty one clears it) and then also drops the image's ``CMD``; compose's ``command`` replaces the
    image's ``CMD``.  The image's are its own with what its Dockerfile's plan sets over them
    (:attr:`~platoon.envs.harbor.images.SessionImage.entrypoint`)."""

    final_entrypoint = list(image_entrypoint or ()) if entrypoint is None else list(entrypoint)
    if command is not None:
        final_cmd = list(command)
    elif entrypoint is not None:
        final_cmd = []
    else:
        final_cmd = list(image_cmd or ())
    return [*final_entrypoint, *final_cmd]


def is_idle(argv: Sequence[str]) -> bool:
    """Whether ``argv`` only keeps a container alive: ``sleep infinity``, ``tail -f /dev/null``, alone or under ``sh
    -c`` (with ``exec`` or not)."""

    words = list(argv)
    if len(words) == 3 and words[0] in _SHELLS and words[1] == "-c":
        try:
            words = shlex.split(words[2])
        except ValueError:
            return False
        if words[:1] == ["exec"]:
            words = words[1:]
    return tuple(words) in _IDLE


def _dotenv(text: str, where: str) -> list[tuple[str, str | None, str]]:
    """An env file's entries as compose reads them (compose-go's dotenv): ``NAME=value`` (or ``NAME: value``, after an
    optional ``export``) with the value unquoted (to the line's end, less a comment that `` #`` starts and the blanks
    that end it), in single quotes (as written) or in double quotes (with the escapes ``\\n``, ``\\t``, ``\\"``,
    ``\\\\`` and ``\\$``, a literal ``$``), quoted values spanning lines; a line that starts with ``#`` is a comment,
    and a ``NAME`` alone on its line takes compose's variable.  Each entry is a name, its value (None for a name alone)
    and its quote (``'``, ``"`` or none)."""

    entries: list[tuple[str, str | None, str]] = []
    text = text.removeprefix("\ufeff").replace("\r\n", "\n")
    index = 0
    while True:
        while index < len(text) and text[index] in _DOTENV_BLANKS + "\n":
            index += 1
        if index >= len(text):
            return entries
        end = _line_end(text, index)
        if text[index] == "#":
            index = end
            continue
        match = _DOTENV_NAME.match(text, index)
        if match is None or (match.group(2) is None and text[match.end() : end].strip()):
            raise ComposeError(where, f"cannot read the line {text[index:end]!r}")
        name, index = match.group(1), match.end()
        if match.group(2) is None:  # a name alone
            entries.append((name, None, ""))
            index = end
            continue
        while index < len(text) and text[index] in _DOTENV_BLANKS:
            index += 1
        quote = text[index] if index < len(text) and text[index] in "'\"" else ""
        if not quote:
            entries.append((name, text[index:end].split(" #", 1)[0].rstrip(), ""))
            index = end
            continue
        chars: list[str] = []
        index += 1
        while index < len(text) and text[index] != quote:
            char = text[index]
            following = text[index + 1 : index + 2]
            if char == "\\" and following:
                if following == quote:
                    chars.append(quote)
                elif quote == '"' and following in _DOTENV_ESCAPES:
                    chars.append(_DOTENV_ESCAPES[following])
                else:
                    chars.append(char + following)
                index += 2
            else:
                chars.append(char)
                index += 1
        if index >= len(text):
            raise ComposeError(where, f"{name}: its quoted value has no closing {quote}")
        entries.append((name, "".join(chars), quote))
        index += 1  # what follows on the line: blanks, a comment, or another entry, as compose-go reads it


_DOTENV_NAME = re.compile(r"(?:export[ \t]+)?([\w.\-\[\]]+)[ \t]*([=:])?")
_DOTENV_BLANKS = " \t\v\f\r"
# Double quotes' escapes, as compose-go expands them; \$ is compose's $$, a $ that interpolation keeps.
_DOTENV_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "a": "\a", "b": "\b", "f": "\f", "v": "\v", "\\": "\\", "$": "$$"}


def _line_end(text: str, index: int) -> int:
    end = text.find("\n", index)
    return len(text) if end < 0 else end


def _healthcheck(raw: Any, where: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ComposeError(where, "must be a mapping")
    if unknown := sorted(set(raw) - _HEALTHCHECK_KEYS):
        raise ComposeError(f"{where}.{unknown[0]}", "is outside the compose subset platoon runs")
    if raw.get("disable"):
        return {"disable": True}
    test = raw.get("test")
    if isinstance(test, str):
        test = ["CMD-SHELL", test]
    if not isinstance(test, list) or not test or not all(isinstance(item, str) for item in test):
        raise ComposeError(f"{where}.test", "must be a command")
    if test[0] == "NONE":
        return {"disable": True}
    if test[0] not in ("CMD", "CMD-SHELL") or len(test) < 2:
        raise ComposeError(f"{where}.test", f"must start with CMD or CMD-SHELL, got {test[0]!r}")
    if test[0] == "CMD-SHELL":
        test = ["CMD-SHELL", " ".join(test[1:])]
    retries = raw.get("retries", 3)
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ComposeError(f"{where}.retries", f"must be a number of retries, got {retries!r}")
    # Docker reads 0 as its default, for the durations as for the retries.
    return ServiceHealthcheck(
        test=tuple(test),
        interval_s=_duration(raw.get("interval", "30s"), f"{where}.interval") or _DEFAULT_INTERVAL_S,
        timeout_s=_duration(raw.get("timeout", "30s"), f"{where}.timeout") or _DEFAULT_TIMEOUT_S,
        start_period_s=_duration(raw.get("start_period", "0s"), f"{where}.start_period"),
        start_interval_s=(
            _duration(raw.get("start_interval", "5s"), f"{where}.start_interval") or _DEFAULT_START_INTERVAL_S
        ),
        retries=retries or _DEFAULT_RETRIES,
    ).to_json()


def image_healthcheck(raw: Mapping[str, Any] | None) -> ServiceHealthcheck | None:
    """An image's ``HEALTHCHECK`` (a plan's ``config["healthcheck"]``, or an image map entry's), None for none or
    ``NONE``."""

    if not raw:
        return None
    parsed = _healthcheck(raw, "HEALTHCHECK")
    return None if parsed.get("disable") else ServiceHealthcheck.from_json(parsed)


def _duration(value: Any, where: str) -> float:
    """A compose duration (``1m30s``, ``500ms``, ``2s``), or a number of seconds, in seconds."""

    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    if not isinstance(value, str) or not value.strip():
        raise ComposeError(where, f"must be a duration such as 30s, got {value!r}")
    text = value.strip()
    position, total = 0, 0.0
    while position < len(text):
        match = _DURATION.match(text, position)
        if match is None:
            raise ComposeError(where, f"must be a duration such as 1m30s, got {value!r}")
        total += float(match.group(1)) * _DURATION_UNITS[match.group(2)]
        position = match.end()
    return total


def _size_mb(value: Any, where: str) -> int:
    """A compose size (``4g``, ``1gb``, ``512m``, or bytes) in MiB, at least 1."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        size = float(value)
    else:
        match = _SIZE.match(str(value))
        if match is None:
            raise ComposeError(where, f"must be a size such as 512m or 4g, got {value!r}")
        size = float(match.group(1)) * _SIZE_UNITS[match.group(2).lower()]
    return max(1, int(size // (1 << 20)))


def _resources(name: str, values: Mapping[str, Any]) -> dict[str, Any]:
    resources: dict[str, Any] = {}
    deploy = values.get("deploy")
    if deploy is not None:
        where = f"{name}.deploy"
        if not isinstance(deploy, Mapping):
            raise ComposeError(where, "must be a mapping")
        if unknown := sorted(set(deploy) - _DEPLOY_KEYS):
            raise ComposeError(f"{where}.{unknown[0]}", "is outside the compose subset platoon runs")
        if deploy.get("replicas", 1) not in (1, "1"):
            raise ComposeError(f"{where}.replicas", "runs several copies of a service, which platoon does not")
        limits = (deploy.get("resources") or {}).get("limits") or {}
        reservations = (deploy.get("resources") or {}).get("reservations") or {}
        if unknown := sorted(set(deploy.get("resources") or {}) - {"limits", "reservations"}):
            raise ComposeError(f"{where}.resources.{unknown[0]}", "is outside the compose subset platoon runs")
        if reservations.get("devices") or reservations.get("generic_resources"):
            raise ComposeError(
                f"{where}.resources.reservations.devices",
                "asks for GPUs or devices; a task asks for GPUs with [environment].gpus; a stack's services get none",
            )
        if unknown := sorted(set(limits) - {"cpus", "memory", "pids"}):
            raise ComposeError(f"{where}.resources.limits.{unknown[0]}", "is outside the compose subset platoon runs")
        if limits.get("pids") is not None:
            raise ComposeError(f"{where}.resources.limits.pids", "is outside the compose subset platoon runs")
        if limits.get("cpus") is not None:
            resources["cpus"] = _cpus(limits["cpus"], f"{where}.resources.limits.cpus")
        if limits.get("memory") is not None:
            resources["memory_mb"] = _size_mb(limits["memory"], f"{where}.resources.limits.memory")
    if values.get("cpus") is not None:
        resources["cpus"] = _cpus(values["cpus"], f"{name}.cpus")
    if values.get("mem_limit") is not None:
        resources["memory_mb"] = _size_mb(values["mem_limit"], f"{name}.mem_limit")
    return resources


def _cpus(value: Any, where: str) -> float:
    try:
        cpus = float(value)
    except (TypeError, ValueError):
        raise ComposeError(where, f"must be a number of CPUs, got {value!r}") from None
    if not cpus > 0:
        raise ComposeError(where, f"must be a positive number of CPUs, got {value!r}")
    return cpus


def _ulimits(raw: Any, where: str) -> dict[str, list[int]]:
    if not isinstance(raw, Mapping):
        raise ComposeError(where, "must map limits to values")
    limits: dict[str, list[int]] = {}
    for name, value in raw.items():
        if isinstance(value, Mapping):
            soft, hard = value.get("soft"), value.get("hard")
        else:
            soft = hard = value
        if not all(isinstance(each, int) and not isinstance(each, bool) for each in (soft, hard)):
            raise ComposeError(f"{where}.{name}", f"must be a number or soft and hard numbers, got {value!r}")
        limits[str(name)] = [soft, hard]
    return limits


def _capabilities(raw: Any, where: str, *, held_only: bool = True) -> list[str]:
    """The capabilities a ``cap_add`` names; with ``held_only``, a ``ComposeError`` for one that a service's root does
    not hold over the stack."""

    names = [raw] if isinstance(raw, str) else raw
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ComposeError(where, "must list capabilities")
    capabilities = []
    for name in names:
        capability = name.upper().removeprefix("CAP_")
        if held_only and capability not in HELD_CAPABILITIES:
            raise ComposeError(
                where,
                f"adds {capability}, which a service's root does not hold over the stack (only capabilities over its "
                "own processes and files: SYS_PTRACE, SYS_CHROOT, ...); an overlay may drop it, and a fetch with "
                "refuse: {privileges: false} keeps it for the docker provider",
            )
        capabilities.append(capability)
    return capabilities


def _ports(raw: Any, where: str, *, published: bool) -> list[str]:
    """``expose`` or ``ports`` entries as the container ports they name (``8000/tcp``); ranges in full."""

    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ComposeError(where, "must be a list")
    ports: list[str] = []
    for entry in raw:
        if isinstance(entry, Mapping):
            if not published:
                raise ComposeError(where, f"entries must be ports, got {entry!r}")
            target, protocol = entry.get("target"), str(entry.get("protocol") or "tcp")
            text = f"{target}/{protocol}"
        elif isinstance(entry, (int, str)) and not isinstance(entry, bool):
            text = str(entry)
            if published:  # [[ip:]host:]container[/protocol]: the container's port
                port, slash, protocol = text.partition("/")
                text = port.rsplit(":", 1)[-1] + slash + protocol
        else:
            raise ComposeError(where, f"entries must be ports, got {entry!r}")
        port, _, protocol = text.partition("/")
        protocol = protocol or "tcp"
        low, dash, high = port.partition("-")
        if not low.isdigit() or (dash and not high.isdigit()):
            raise ComposeError(where, f"{entry!r} is not a port")
        first, last = int(low), int(high) if dash else int(low)
        if not 0 < first <= last <= 65535 or last - first > 1024:
            raise ComposeError(where, f"{entry!r} is not a port or a short range")
        ports += [f"{number}/{protocol}" for number in range(first, last + 1)]
    return ports


def _build(raw: Any, where: str) -> tuple[str, str, dict[str, str]]:
    if isinstance(raw, str):
        return raw, "Dockerfile", {}
    if not isinstance(raw, Mapping):
        raise ComposeError(where, "must be a context or a mapping")
    if unknown := sorted(set(raw) - _BUILD_KEYS):
        raise ComposeError(f"{where}.{unknown[0]}", "is outside the compose subset platoon runs")
    context = raw.get("context", ".")
    dockerfile = raw.get("dockerfile", "Dockerfile")
    if not isinstance(context, str) or not context:
        raise ComposeError(f"{where}.context", f"must name a directory, got {context!r}")
    if not isinstance(dockerfile, str) or not dockerfile or dockerfile.startswith("/") or ".." in dockerfile.split("/"):
        raise ComposeError(f"{where}.dockerfile", f"must name a file in the context, got {dockerfile!r}")
    args = raw.get("args") or {}
    if isinstance(args, list):
        args = dict(item.partition("=")[::2] for item in args if isinstance(item, str))
    if not isinstance(args, Mapping):
        raise ComposeError(f"{where}.args", "must map names to values")
    return context, dockerfile, {str(key): "" if value is None else str(value) for key, value in args.items()}


def _list_of(value: Any, where: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, Mapping)):
        return [value]
    if isinstance(value, list):
        return value
    raise ComposeError(where, "must be a value or a list")


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ComposeError(where, f"must be a string, got {value!r}")
    return value


def _absolute(path: str, where: str) -> str:
    if not isinstance(path, str) or not path.startswith("/"):
        raise ComposeError(where, f"{path!r} must be an absolute path in the container")
    normalized = posixpath.normpath(path)
    return "/" + normalized.lstrip("/") if normalized != "/" else "/"


def _below(path: str, directory: str) -> bool:
    return path != directory and PurePosixPath(path).is_relative_to(directory)


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _host_platforms() -> set[str]:
    machine = host_platform.machine().lower()
    arch = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(machine, machine)
    return {f"linux/{arch}", "linux"}
