"""Harbor task directories: ``task.toml``, ``instruction.md``, ``environment/``, ``tests/test.sh`` and ``solution/``.

Only the parts of Harbor's task schema that a platoon session uses are read, with Harbor 0.24's meaning.  A task is
graded in its own sandbox (``shared``) or in a separate verifier sandbox that receives the paths the task declares
in ``artifacts`` (``[verifier] environment_mode = "separate"``, or a ``[verifier.environment]`` table).

Refused: multi-step and non-Linux tasks; multi-container environments (a compose file with sidecar services; one
that only names ``main`` is a single container); verifier collect hooks and artifacts of other services; a
compose file in ``tests/``; and ``[agent].user`` or ``[verifier].user`` unless the caller ignores them, since
rootless sandboxes run every process as one user.
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, get_args

import yaml

from platoon.task import Task

CONFIG_FILENAME = "task.toml"
INSTRUCTION_FILENAME = "instruction.md"
TEST_SCRIPT = Path("tests") / "test.sh"
ENVIRONMENT_DIR = "environment"
TESTS_DIR = "tests"
# A compose file makes the environment a stack: the task's services (a judge, a database, an MCP server) layered over
# Harbor's base, which defines ``main`` (harbor: src/harbor/environments/docker/docker.py).
COMPOSE_FILENAMES = ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml")
# The files in tests/ that give a separate verifier its own image (harbor 0.24: models/task/verifier_mode.py).
VERIFIER_BUILD_FILES = ("Dockerfile", "docker-compose.yaml")
# The directory every agent may publish files in for a separate verifier, collected besides ``artifacts``.
ARTIFACTS_DIR = "/logs/artifacts"
HARBOR_MISC_KEY = "harbor"  # Task.misc entry that locates the task directory
DEFAULT_VERIFIER_TIMEOUT_S = 600.0
McpTransport = Literal["stdio", "sse", "streamable-http"]
VerifierMode = Literal["shared", "separate"]
# Where a separate verifier's image comes from, in Harbor's order of precedence: ``[verifier.environment]
# .docker_image``, an image built from ``tests/`` (its Dockerfile), else the task's own image.
VerifierImageSource = Literal["verifier_image", "tests", "task"]
# What a task's ``[agent].user`` or ``[verifier].user`` does: ``refuse`` the task, or ``ignore`` the user and run
# as the sandbox's single uid (rootless providers cannot switch users).
AgentUserPolicy = Literal["refuse", "ignore"]
AGENT_USER_POLICIES: tuple[str, ...] = get_args(AgentUserPolicy)
Network = Literal["host", "none"]

# Leading canary comments are data-provenance markers that agents should not see.
_CANARY_LINE = re.compile(r"^(<!--.*canary.*-->|#.*canary.*)$", re.IGNORECASE)
_SIZE_UNITS_MB = {"K": 1 / 1024, "M": 1, "G": 1024}
_ENV_TEMPLATE = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")
_MCP_TRANSPORTS: dict[str, McpTransport] = {
    "stdio": "stdio",
    "sse": "sse",
    "streamable-http": "streamable-http",
    "http": "streamable-http",
}


class HarborTaskError(ValueError):
    """A directory is not a Harbor task that platoon can run."""


@dataclass(frozen=True)
class McpServer:
    """An MCP server the task offers to agents (``[[environment.mcp_servers]]``)."""

    name: str
    transport: McpTransport = "sse"
    url: str | None = None  # sse and streamable-http
    command: str | None = None  # stdio
    args: tuple[str, ...] = ()


@dataclass(frozen=True)
class Healthcheck:
    """A readiness probe run after the sandbox starts (``[environment.healthcheck]``), like Docker's HEALTHCHECK."""

    command: str
    interval_s: float = 5.0
    timeout_s: float = 30.0
    start_period_s: float = 0.0
    start_interval_s: float = 5.0
    retries: int = 3


@dataclass(frozen=True)
class Artifact:
    """A path a separate verifier receives from the agent's sandbox (``artifacts`` in ``task.toml``).

    It is collected from ``main`` and put back at the same path in the verifier's sandbox; a relative path starts
    at the sandbox's workdir.  ``exclude`` patterns leave files out of a directory (``tar --exclude``).
    """

    source: str
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerifierEnvironment:
    """The sandbox a separate verifier runs in, resolved as Harbor 0.24 does (``models/task/verifier_mode.py``).

    Its settings come from ``[verifier.environment]``, else from a copy of ``[environment]``; ``[verifier]
    network_mode`` overrides its network.  Its image comes from ``image_source``: ``verifier_image`` (that table's
    ``docker_image``), ``tests`` (built from ``tests/Dockerfile``), or ``task`` (the task's own image).  As in
    Harbor, no healthcheck runs in it.
    """

    image_source: VerifierImageSource
    # The verifier_image's reference, or for image_source "task" the task's own docker_image (None: built).
    docker_image: str | None = None
    workdir: str | None = None
    env: dict[str, str] = field(default_factory=dict)  # ${VAR} and ${VAR:-default} resolve on the host
    network: Network = "host"
    cpus: float | None = None
    memory_mb: int | None = None
    gpus: int = 0

    @property
    def bundled_tests(self) -> bool:
        """The image holds ``/tests`` itself, so the task's ``tests/`` is not uploaded (Harbor skips it too)."""

        return self.image_source != "task"


@dataclass(frozen=True)
class HarborTask:
    task_dir: Path
    name: str
    instruction: str
    docker_image: str | None = None
    workdir: str | None = None
    env: dict[str, str] = field(default_factory=dict)  # ${VAR} and ${VAR:-default} resolve on the host
    network: Network = "host"
    cpus: float | None = None
    memory_mb: int | None = None
    gpus: int = 0
    agent_timeout_s: float | None = None
    verifier_timeout_s: float = DEFAULT_VERIFIER_TIMEOUT_S
    verifier_env: dict[str, str] = field(default_factory=dict)
    # shared: graded in the agents' sandbox; separate: in verifier_environment, from the collected artifacts.
    verifier_mode: VerifierMode = "shared"
    verifier_environment: VerifierEnvironment | None = None
    artifacts: tuple[Artifact, ...] = ()
    mcp_servers: tuple[McpServer, ...] = ()
    healthcheck: Healthcheck | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # The [agent].user and [verifier].user that a loader with agent_user="ignore" let through, by phase.
    ignored_users: dict[str, str] = field(default_factory=dict)
    # The task's row in its dataset's own index files (task-index.json, a lock), e.g. {"source_dataset": ...}.
    index: dict[str, Any] = field(default_factory=dict)

    @property
    def tests_dir(self) -> Path:
        return self.task_dir / TESTS_DIR

    @property
    def solution_dir(self) -> Path:
        return self.task_dir / "solution"

    @classmethod
    def load(
        cls, task_dir: str | Path, *, agent_user: AgentUserPolicy = "refuse", index: Mapping[str, Any] | None = None
    ) -> HarborTask:
        """Read a task directory; ``HarborTaskError`` says why platoon cannot run it.

        ``agent_user="ignore"`` accepts a task that names a user to run as, and records it in ``ignored_users``.
        ``index`` is the task's row in its dataset's index, kept as ``HarborTask.index``.
        """

        if agent_user not in AGENT_USER_POLICIES:
            raise ValueError(f"agent_user must be one of {AGENT_USER_POLICIES}, got {agent_user!r}")
        task_dir = Path(task_dir).resolve()
        config_path = task_dir / CONFIG_FILENAME
        try:
            config = tomllib.loads(config_path.read_text())
        except FileNotFoundError:
            raise HarborTaskError(f"{task_dir} has no {CONFIG_FILENAME}") from None
        except tomllib.TOMLDecodeError as exc:
            raise HarborTaskError(f"{config_path}: {exc}") from exc
        if config.get("steps"):
            raise HarborTaskError(f"{config_path}: multi-step tasks are not supported")
        environment = _value(config, "environment", dict, "task.toml") or {}
        verifier = _value(config, "verifier", dict, "task.toml") or {}
        agent = _value(config, "agent", dict, "task.toml") or {}
        _check_linux(environment, f"{config_path}")
        compose = compose_file(task_dir)
        if compose is not None and not is_trivial_compose(compose):
            # Fail closed (RFC 0010, P0): started from its image (docker_image or the service's default_image), the
            # main container would run alone, without its services or the files and variables compose gives it, and
            # its verifier would write 0 for a policy that may have done the task: a silent reward 0 in training.
            raise HarborTaskError(
                f"{task_dir}: {compose.relative_to(task_dir)} makes the environment a multi-container stack (sidecar "
                "services), which platoon cannot run yet (RFC 0010); the task is refused, never started without them"
            )
        if _value(verifier, "collect", list, "[verifier]"):
            raise HarborTaskError(
                f"{config_path}: [[verifier.collect]] hooks run commands in compose services before artifacts are "
                "collected, which platoon does not support (RFC 0010)"
            )
        users = {
            phase: str(user)
            for phase, table in (("agent", agent), ("verifier", verifier))
            if (user := _value(table, "user", (str, int), f"[{phase}]")) is not None
        }
        if users and agent_user == "refuse":
            named = ", ".join(f"[{phase}].user = {user!r}" for phase, user in users.items())
            raise HarborTaskError(
                f"{config_path}: {named}, but rootless sandboxes run every process as one user; the task is refused "
                "unless the environment sets agent_user: ignore"
            )
        verifier_environment = _verifier_environment(task_dir, environment, verifier)
        artifacts = _artifacts(config)
        instruction_path = task_dir / INSTRUCTION_FILENAME
        if not instruction_path.is_file():
            raise HarborTaskError(f"{task_dir} has no {INSTRUCTION_FILENAME}")
        bundled_tests = verifier_environment is not None and verifier_environment.bundled_tests
        if not bundled_tests and not (task_dir / TEST_SCRIPT).is_file():
            raise HarborTaskError(f"{task_dir} has no {TEST_SCRIPT}")

        verifier_timeout_s = _value(verifier, "timeout_sec", (int, float), "[verifier]")
        settings = _environment_settings(environment, "[environment]")
        healthcheck = _value(environment, "healthcheck", dict, "[environment]")
        return cls(
            task_dir=task_dir,
            name=_value(_value(config, "task", dict, "task.toml") or {}, "name", str, "[task]") or task_dir.name,
            instruction=strip_canary(instruction_path.read_text()),
            docker_image=_value(environment, "docker_image", str, "[environment]"),
            workdir=settings["workdir"],
            env=settings["env"],
            network=_phase_network(agent, "[agent]") or settings["network"],
            cpus=settings["cpus"],
            memory_mb=settings["memory_mb"],
            gpus=settings["gpus"],
            agent_timeout_s=_value(agent, "timeout_sec", (int, float), "[agent]"),
            verifier_timeout_s=DEFAULT_VERIFIER_TIMEOUT_S if verifier_timeout_s is None else float(verifier_timeout_s),
            verifier_env=_string_map(verifier, "env", "[verifier]"),
            verifier_mode="shared" if verifier_environment is None else "separate",
            verifier_environment=verifier_environment,
            artifacts=artifacts,
            mcp_servers=tuple(
                _mcp_server(entry) for entry in _value(environment, "mcp_servers", list, "[environment]") or []
            ),
            healthcheck=_healthcheck(healthcheck) if healthcheck is not None else None,
            metadata=_value(config, "metadata", dict, "task.toml") or {},
            ignored_users=users,
            index=dict(index or {}),
        )

    def to_task(self, task_id: str) -> Task:
        """The platoon task: the instruction is the goal and ``misc`` locates this directory."""

        misc: dict[str, Any] = {
            "task_dir": str(self.task_dir),
            "name": self.name,
            "agent_timeout_s": self.agent_timeout_s,
        }
        if self.index:
            misc["index"] = dict(self.index)
        return Task(id=task_id, goal=self.instruction, misc={HARBOR_MISC_KEY: misc})


def compose_file(task_dir: Path) -> Path | None:
    """The task's compose file (``environment/docker-compose.yaml`` and its aliases), or None."""

    return next((path for name in COMPOSE_FILENAMES if (path := task_dir / ENVIRONMENT_DIR / name).is_file()), None)


def is_trivial_compose(path: Path) -> bool:
    """A compose file that only names ``main``, with no keys (``services: {main: {}}``), as some converters write
    for every task: the environment is one container.  An unreadable file is not trivial."""

    try:
        document = yaml.safe_load(path.read_text())
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return False
    if not isinstance(document, dict) or set(document) - {"services", "version"}:
        return False
    services = document.get("services")
    return isinstance(services, dict) and list(services) == ["main"] and not services["main"]


def strip_canary(text: str) -> str:
    """Drop leading canary comment lines and the blank lines after them."""

    lines = text.split("\n")
    start = 0
    while start < len(lines) and _CANARY_LINE.match(lines[start].strip()):
        start += 1
    while start < len(lines) and not lines[start].strip():
        start += 1
    return "\n".join(lines[start:])


def resolve_env(env: Mapping[str, str]) -> dict[str, str]:
    """Resolve ``${VAR}`` and ``${VAR:-default}`` values from the host environment; other values are literal.

    Raises ``KeyError`` naming a variable that is unset and has no default.
    """

    resolved: dict[str, str] = {}
    for key, value in env.items():
        match = _ENV_TEMPLATE.fullmatch(value)
        if match is None:
            resolved[key] = value
        elif match.group(1) in os.environ:
            resolved[key] = os.environ[match.group(1)]
        elif match.group(2) is not None:
            resolved[key] = match.group(2)
        else:
            raise KeyError(match.group(1))
    return resolved


def _value(table: dict[str, Any], key: str, kind: type | tuple[type, ...], where: str) -> Any:
    value = table.get(key)
    kinds = kind if isinstance(kind, tuple) else (kind,)
    if value is None:
        return None
    if not isinstance(value, kinds) or (isinstance(value, bool) and bool not in kinds):
        expected = " or ".join(k.__name__ for k in kinds)
        raise HarborTaskError(f"{where}.{key} must be {expected}, got {value!r}")
    return value


def _string_map(table: dict[str, Any], key: str, where: str) -> dict[str, str]:
    mapping = _value(table, key, dict, where) or {}
    for name, value in mapping.items():
        if not isinstance(value, str):
            raise HarborTaskError(f"{where}.{key}.{name} must be str, got {value!r}")
    return dict(mapping)


def _check_linux(environment: dict[str, Any], where: str) -> None:
    if str(environment.get("os", "linux")).lower() != "linux":
        raise HarborTaskError(f"{where}: only Linux tasks are supported")


def _environment_settings(table: dict[str, Any], where: str) -> dict[str, Any]:
    """The sandbox settings of an ``[environment]`` table (or ``[verifier.environment]``, which has its schema)."""

    return {
        "workdir": _value(table, "workdir", str, where),
        "env": _string_map(table, "env", where),
        "network": _network(table, where),
        "cpus": _value(table, "cpus", (int, float), where),
        "memory_mb": _memory_mb(table, where),
        "gpus": _value(table, "gpus", int, where) or 0,
    }


def _verifier_environment(
    task_dir: Path, environment: dict[str, Any], verifier: dict[str, Any]
) -> VerifierEnvironment | None:
    """A separate verifier's sandbox, or None when the task grades in its own (Harbor's ``shared`` mode)."""

    mode = _value(verifier, "environment_mode", str, "[verifier]")
    table = _value(verifier, "environment", dict, "[verifier]")
    if mode not in (None, "shared", "separate"):
        raise HarborTaskError(f"[verifier].environment_mode must be shared or separate, got {mode!r}")
    if mode == "shared" and table is not None:
        raise HarborTaskError("[verifier].environment_mode = 'shared' conflicts with a [verifier.environment] table")
    if mode is None:
        mode = "separate" if table is not None else "shared"
    if mode == "shared":
        return None
    effective = environment if table is None else table
    if table is not None:
        _check_linux(table, "[verifier.environment]")
    settings = _environment_settings(effective, "[verifier.environment]" if table is not None else "[environment]")
    settings["network"] = _phase_network(verifier, "[verifier]") or settings["network"]
    if table is not None and (image := _value(table, "docker_image", str, "[verifier.environment]")):
        return VerifierEnvironment(image_source="verifier_image", docker_image=image, **settings)
    built = [name for name in VERIFIER_BUILD_FILES if (task_dir / TESTS_DIR / name).is_file()]
    if "Dockerfile" in built:
        return VerifierEnvironment(image_source="tests", **settings)
    if built:
        raise HarborTaskError(
            f"{task_dir}: tests/{built[0]} makes the separate verifier a multi-container stack, which platoon cannot "
            "run yet (RFC 0010)"
        )
    return VerifierEnvironment(
        image_source="task", docker_image=_value(environment, "docker_image", str, "[environment]"), **settings
    )


def _artifacts(config: dict[str, Any]) -> tuple[Artifact, ...]:
    """The ``artifacts`` list: paths as strings, or ``{source, destination, exclude, service}`` tables."""

    artifacts = []
    for entry in _value(config, "artifacts", list, "task.toml") or []:
        if isinstance(entry, str):
            entry = {"source": entry}
        if not isinstance(entry, dict) or not isinstance(entry.get("source"), str) or not entry["source"]:
            raise HarborTaskError(f"artifacts entries must be paths or tables with a source, got {entry!r}")
        source = entry["source"]
        if ".." in PurePosixPath(source).parts:
            raise HarborTaskError(f"artifact source {source!r} must not contain '..'")
        service = _value(entry, "service", str, "artifacts")
        if service not in (None, "main"):
            raise HarborTaskError(
                f"artifact {source!r} comes from the compose service {service!r}, which platoon cannot run yet "
                "(RFC 0010)"
            )
        exclude = _value(entry, "exclude", list, "artifacts") or []
        if not all(isinstance(pattern, str) for pattern in exclude):
            raise HarborTaskError(f"artifacts.exclude must list patterns, got {exclude!r}")
        artifacts.append(Artifact(source=source, exclude=tuple(exclude)))  # destination only places host copies
    return tuple(artifacts)


def _network(table: dict[str, Any], where: str = "[environment]") -> Network:
    mode = _value(table, "network_mode", str, where)
    if mode is None:
        return "none" if _value(table, "allow_internet", bool, where) is False else "host"
    return _network_mode(mode, where)


def _phase_network(table: dict[str, Any], where: str) -> Network | None:
    """``[agent]`` and ``[verifier]`` may override their phase's network (``network_mode``); None when unset."""

    mode = _value(table, "network_mode", str, where)
    return None if mode is None else _network_mode(mode, where)


def _network_mode(mode: str, where: str) -> Network:
    if mode not in ("public", "allowlist", "no-network"):
        raise HarborTaskError(f"{where}.network_mode must be public, allowlist or no-network, got {mode!r}")
    return "none" if mode == "no-network" else "host"  # allowlists are not enforced


def _memory_mb(environment: dict[str, Any], where: str = "[environment]") -> int | None:
    memory_mb = _value(environment, "memory_mb", int, where)
    legacy = _value(environment, "memory", str, where)  # deprecated, e.g. "2G"
    if memory_mb is not None or legacy is None:
        return memory_mb
    size, unit = legacy.strip()[:-1], legacy.strip()[-1:].upper()
    try:
        return int(float(size) * _SIZE_UNITS_MB[unit])
    except (KeyError, ValueError):
        raise HarborTaskError(f"{where}.memory must look like '2G' or '512M', got {legacy!r}") from None


def _mcp_server(entry: Any) -> McpServer:
    where = "[[environment.mcp_servers]]"
    if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
        raise HarborTaskError(f"{where} entries need a name, got {entry!r}")
    name = _value(entry, "transport", str, where) or "sse"
    transport = _MCP_TRANSPORTS.get(name)
    url = _value(entry, "url", str, where)
    command = _value(entry, "command", str, where)
    args = _value(entry, "args", list, where) or []
    if transport is None:
        raise HarborTaskError(f"{where}.transport must be stdio, sse or streamable-http, got {name!r}")
    if (transport == "stdio" and not command) or (transport != "stdio" and not url):
        raise HarborTaskError(f"{where} {entry['name']!r}: stdio needs a command, other transports a url")
    if not all(isinstance(arg, str) for arg in args):
        raise HarborTaskError(f"{where}.args must be strings, got {args!r}")
    return McpServer(name=entry["name"], transport=transport, url=url, command=command, args=tuple(args))


def _healthcheck(table: dict[str, Any]) -> Healthcheck:
    where = "[environment.healthcheck]"
    command = _value(table, "command", str, where)
    if not command:
        raise HarborTaskError(f"{where}.command is required")
    defaults = Healthcheck(command=command)

    def setting(key: str, kind: type | tuple[type, ...], default: Any) -> Any:
        value = _value(table, key, kind, where)
        return default if value is None else value

    number = (int, float)
    return Healthcheck(
        command=command,
        interval_s=setting("interval_sec", number, defaults.interval_s),
        timeout_s=setting("timeout_sec", number, defaults.timeout_s),
        start_period_s=setting("start_period_sec", number, defaults.start_period_s),
        start_interval_s=setting("start_interval_sec", number, defaults.start_interval_s),
        retries=setting("retries", int, defaults.retries),
    )
