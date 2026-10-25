"""Harbor task directories: ``task.toml``, ``instruction.md``, ``environment/``, ``tests/test.sh`` and ``solution/``.

Only the parts of Harbor's task schema that a platoon session uses are read, with Harbor 0.24's meaning.  A task is
graded in its own sandbox (``shared``) or in a separate verifier sandbox that receives the paths the task declares
in ``artifacts`` (``[verifier] environment_mode = "separate"``, or a ``[verifier.environment]`` table).

A compose file (``environment/docker-compose.yaml``, the only one Harbor reads) is read into a stack plan
(:mod:`platoon.envs.harbor.compose`): one that names only ``main`` runs as the task's one sandbox, with what it gives
``main`` (an environment, binds, a working directory, resources, a command, a healthcheck: :attr:`HarborTask.main`),
and one with services besides ``main`` runs as a multi-container stack (:mod:`platoon.envs.harbor.sidecars`), with
``[[verifier.collect]]`` hooks run in its services and ``artifacts`` collected from them (``service``).

Refused: multi-step and non-Linux tasks; stacks where the caller says none can run (``stack_refusal``), and stacks
with a service a node service serves, or whose services depend on one another in a cycle; compose files outside the
subset platoon runs, or that give ``main`` what platoon cannot apply to it yet; collect hooks and artifacts of a
service the task does not start; a compose file in ``tests/``; and ``[agent].user`` or ``[verifier].user`` (or a
collect hook's ``user``) unless the caller ignores them, since rootless sandboxes run every process as one user.  A
caller may instead honor a ``[verifier].user`` of root (``verifier_user="honor"``): the task's tests then run as root
in its separate verifier's sandbox, where they may run the agent's code as another user
(:attr:`HarborTask.verifier_root`).  It honors no other user, and no root verifier of a task graded in its agents' own
sandbox, whose agents would run as root too: those tasks are refused.
"""

from __future__ import annotations

import logging
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, get_args

from platoon.envs.harbor.compose import (
    COMPOSE_FILE,
    MAIN,
    ComposeError,
    ComposePlan,
    MainService,
    plan_compose,
    service_binds,
    unread_compose_files,
)
from platoon.task import Task

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "task.toml"
INSTRUCTION_FILENAME = "instruction.md"
TEST_SCRIPT = Path("tests") / "test.sh"
ENVIRONMENT_DIR = "environment"
TESTS_DIR = "tests"
# The compose file Harbor reads, which may make the environment a stack: the task's services (a judge, a database, an
# MCP server) layered over Harbor's base, which defines ``main`` (harbor: src/harbor/environments/docker/docker.py).
# Compose's other names (compose.yaml, ...) are plain files of a Harbor task.
COMPOSE_FILENAMES = (COMPOSE_FILE,)
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
# What a task's ``[verifier].user`` does, when it is not what ``[agent].user`` gets: ``refuse`` or ``ignore`` it as
# above, or ``honor`` it: a separate verifier whose user is root runs as root, in a sandbox that gives it root and
# other users to run the agent's code as; any other user is refused.
VerifierUserPolicy = Literal["refuse", "ignore", "honor"]
VERIFIER_USER_POLICIES: tuple[str, ...] = get_args(VerifierUserPolicy)
# How a task names root: Docker's ``--user`` forms, a name or a uid, optionally with a group.
_ROOT_NAMES = ("root", "0")
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


_PLANNED_HERE: Any = object()  # HarborTask.load plans the task's compose file itself


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

    It is collected from ``main``, or from the stack's ``service`` (an absolute path of that service's own files, which
    the agents cannot write), and put back at the same path in the verifier's sandbox; a relative path starts at the
    sandbox's workdir.  ``exclude`` patterns leave files out of a directory (``tar --exclude``).
    """

    source: str
    exclude: tuple[str, ...] = ()
    service: str | None = None  # None: main


@dataclass(frozen=True)
class CollectHook:
    """A command run in one of the task's services once its agents are done, before the artifacts are collected
    (``[[verifier.collect]]``, Harbor's): best effort, its failure is logged and the grading goes on.  Harbor runs
    ``main``'s hooks first, then the sidecars'."""

    command: str
    service: str = "main"
    timeout_s: float = 60.0


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
    # The [verifier].user is root and the loader honors it (verifier_user="honor"): the tests run as root in the
    # separate verifier's sandbox.
    verifier_root: bool = False
    # The task's row in its dataset's own index files (task-index.json, a lock), e.g. {"source_dataset": ...}.
    index: dict[str, Any] = field(default_factory=dict)
    # Its compose file's stack plan, None without one (platoon.envs.harbor.compose).
    compose: ComposePlan | None = None
    # [[verifier.collect]] hooks, in the task's order.
    collect: tuple[CollectHook, ...] = ()
    # The network of a stack while it is graded in its agents' own sandbox: [verifier].network_mode, else the task's.
    verifier_network: Network = "host"

    @property
    def main(self) -> MainService | None:
        """What a compose file gives the task's sandbox, ``main`` (a main-only file's, or a stack's); None for a task
        without one."""

        return self.compose.main() if self.compose is not None else None

    @property
    def is_stack(self) -> bool:
        """Whether the task's compose file names services besides ``main``: it runs as a multi-container stack."""

        return self.compose is not None and self.compose.kind == "sidecars"

    @property
    def tests_dir(self) -> Path:
        return self.task_dir / TESTS_DIR

    @property
    def solution_dir(self) -> Path:
        return self.task_dir / "solution"

    @classmethod
    def load(
        cls,
        task_dir: str | Path,
        *,
        agent_user: AgentUserPolicy = "refuse",
        verifier_user: VerifierUserPolicy | None = None,
        index: Mapping[str, Any] | None = None,
        compose: ComposePlan | Mapping[str, Any] | None | object = _PLANNED_HERE,
        stack_refusal: str | None = None,
    ) -> HarborTask:
        """Read a task directory; ``HarborTaskError`` says why platoon cannot run it.

        ``agent_user="ignore"`` accepts a task that names a user to run as, and records it in ``ignored_users``.
        ``verifier_user`` decides for ``[verifier].user`` instead, when given: ``honor`` accepts a separate verifier
        whose user is root and sets ``verifier_root``, and refuses any other user, and a root verifier of a task
        graded in its agents' sandbox.  ``index`` is the task's row in its dataset's index, kept as
        ``HarborTask.index``.  ``compose`` is the task's stack plan as its lock records it (None: the task has no
        compose file), which a dataset's overlays may have edited; without it, the task's compose file is planned
        here.  ``stack_refusal`` says why no stack could run (the caller's sandbox provider starts none), which
        refuses a task whose compose file makes it one.
        """

        if agent_user not in AGENT_USER_POLICIES:
            raise ValueError(f"agent_user must be one of {AGENT_USER_POLICIES}, got {agent_user!r}")
        if verifier_user is not None and verifier_user not in VERIFIER_USER_POLICIES:
            raise ValueError(f"verifier_user must be one of {VERIFIER_USER_POLICIES}, got {verifier_user!r}")
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
        settings = _environment_settings(environment, "[environment]")
        plan = _compose_plan(task_dir, environment, settings, compose, stack_refusal)
        users = {
            phase: str(user)
            for phase, table in (("agent", agent), ("verifier", verifier))
            if (user := _value(table, "user", (str, int), f"[{phase}]")) is not None
        }
        policies = {"agent": agent_user, "verifier": agent_user if verifier_user is None else verifier_user}
        if refused := {phase: user for phase, user in users.items() if policies[phase] == "refuse"}:
            named = ", ".join(f"[{phase}].user = {user!r}" for phase, user in refused.items())
            root = "verifier" in refused and is_root(refused["verifier"])
            if "agent" in refused or verifier_user is None:
                fix = "agent_user: ignore"
                if root and "agent" not in refused:
                    fix += ", or verifier_user: honor to run its tests as root"
            else:
                fix = "verifier_user: ignore" + (", or honor to run its tests as root" if root else "")
            raise HarborTaskError(
                f"{config_path}: {named}, but rootless sandboxes run every process as one user; the task is refused "
                f"unless the environment sets {fix}"
            )
        hooks, hook_users = _collect_hooks(verifier, plan, f"{config_path}")
        if hook_users and policies["verifier"] != "ignore":
            policy = "verifier_user" if verifier_user is not None else "agent_user"
            raise HarborTaskError(
                f"{config_path}: [[verifier.collect]] runs a hook as user {hook_users[0]!r}, but rootless sandboxes "
                f"run every process as one user; the task is refused unless the environment sets {policy}: ignore"
            )
        if hook_users:
            users = {**users, "collect": hook_users[0]}
            policies = {**policies, "collect": "ignore"}
        verifier_environment = _verifier_environment(task_dir, environment, verifier)
        verifier_root = policies["verifier"] == "honor" and "verifier" in users
        if verifier_root and not is_root(user := users["verifier"]):
            raise HarborTaskError(
                f"{config_path}: [verifier].user = {user!r}, but platoon runs a verifier as root or as its sandbox's "
                "own user, so verifier_user: honor honors only root; the task is refused unless the environment sets "
                "verifier_user: ignore"
            )
        if verifier_root and verifier_environment is None:
            raise HarborTaskError(
                f"{config_path}: [verifier].user = {users['verifier']!r}, but the task is graded in its agents' "
                "sandbox (Harbor's shared verifier mode), and platoon runs a verifier as root only in a separate "
                "verifier's sandbox: its agents would run as root too; the task is refused unless the environment "
                "sets verifier_user: ignore"
            )
        artifacts = _artifacts(config, plan)
        instruction_path = task_dir / INSTRUCTION_FILENAME
        if not instruction_path.is_file():
            raise HarborTaskError(f"{task_dir} has no {INSTRUCTION_FILENAME}")
        bundled_tests = verifier_environment is not None and verifier_environment.bundled_tests
        if not bundled_tests and not (task_dir / TEST_SCRIPT).is_file():
            raise HarborTaskError(f"{task_dir} has no {TEST_SCRIPT}")

        verifier_timeout_s = _value(verifier, "timeout_sec", (int, float), "[verifier]")
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
            ignored_users={phase: user for phase, user in users.items() if policies[phase] == "ignore"},
            verifier_root=verifier_root,
            index=dict(index or {}),
            compose=plan,
            collect=hooks,
            verifier_network=_phase_network(verifier, "[verifier]") or settings["network"],
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


def is_root(user: str) -> bool:
    """Whether a task's ``user`` (``[agent].user``, ``[verifier].user``) is root, in any of the forms Docker's
    ``--user`` takes: ``root`` or ``0``, optionally with the group ``root`` or ``0`` (``0:0``, ``root:root``)."""

    name, _, group = user.partition(":")
    return name in _ROOT_NAMES and group in ("", *_ROOT_NAMES)


def compose_file(task_dir: Path) -> Path | None:
    """The task's compose file, ``environment/docker-compose.yaml``, or None: Harbor reads no other name."""

    path = Path(task_dir) / ENVIRONMENT_DIR / COMPOSE_FILE
    return path if path.is_file() else None


_WARNED_UNREAD: set[str] = set()  # the compose file names that Harbor does not read, warned about by this process


def _compose_plan(
    task_dir: Path, environment: dict[str, Any], settings: dict[str, Any], compose: Any, stack_refusal: str | None
) -> ComposePlan | None:
    """The task's stack plan: ``compose`` when the caller has it (a lock's), else planned from the task's files; a
    plan that platoon cannot run (yet, or where no stack runs: ``stack_refusal``) refuses the task."""

    if compose is _PLANNED_HERE:
        for name in unread_compose_files(task_dir):
            if name not in _WARNED_UNREAD:
                _WARNED_UNREAD.add(name)
                logger.warning(
                    "Harbor task %s has %s, which Harbor does not read (it reads only %s/%s): platoon treats it as a "
                    "plain file of the task, as Harbor does",
                    task_dir.name,
                    name,
                    ENVIRONMENT_DIR,
                    COMPOSE_FILE,
                )
        try:
            plan = plan_compose(
                task_dir,
                docker_image=_value(environment, "docker_image", str, "[environment]"),
                env=settings["env"],
                cpus=settings["cpus"],
                memory_mb=settings["memory_mb"],
                with_keys=False,
            )
        except ComposeError as exc:
            raise HarborTaskError(
                f"{task_dir}: {ENVIRONMENT_DIR}/{COMPOSE_FILE}: {exc}, which platoon cannot run (RFC 0010); the task "
                "is refused"
            ) from exc
    elif compose is None or isinstance(compose, ComposePlan):
        plan = compose
    else:
        plan = ComposePlan.from_json(compose)  # type: ignore[arg-type]
    if plan is None:
        return None
    stack = (
        f"{task_dir}: {ENVIRONMENT_DIR}/{COMPOSE_FILE} makes the environment a multi-container stack (sidecar services)"
    )
    if plan.kind == "sidecars":
        # Fail closed (RFC 0010, P0): started from its image (docker_image or the service's default_image), the
        # main container would run alone, without its services or the files and variables compose gives it, and
        # its verifier would write 0 for a policy that may have done the task: a silent reward 0 in training.
        if stack_refusal is not None:
            raise HarborTaskError(f"{stack}, and {stack_refusal}; the task is refused, never started without them")
        if external := plan.external:
            name, served_by = next(iter(sorted(external.items())))
            raise HarborTaskError(
                f"{stack} whose service {name} a node service serves ({served_by}, an overlay's external), which "
                "platoon does not run; the task is refused"
            )
        if cycle := plan.dependency_cycle():
            raise HarborTaskError(f"{stack} whose services depend on one another in a cycle ({' -> '.join(cycle)})")
        if (problem := _sidecars_problem(plan, planned_here=compose is _PLANNED_HERE)) is not None:
            raise HarborTaskError(f"{stack} {problem}; the task is refused")
    main = plan.main()
    if main is not None and main.unsupported:
        sandbox = "main in a stack" if plan.kind == "sidecars" else "the one sandbox of a main-only task"
        raise HarborTaskError(
            f"{task_dir}: {ENVIRONMENT_DIR}/{COMPOSE_FILE} gives main {', '.join(main.unsupported)}, which platoon "
            f"does not apply to {sandbox} yet (RFC 0010); the task is refused"
        )
    return plan


def _sidecars_problem(plan: ComposePlan, *, planned_here: bool) -> str | None:
    """Why no stack could run a stack plan's sidecars as compose would; None when nothing says so.

    Always: a sidecar's read-only bind of a session log directory, which would show it what ``main`` writes there in
    Docker and here an empty directory of its own; and binds of one path of the task's files that another service
    writes with other masks, or one inside the other, which compose shares and platoon would copy apart.  For a plan
    ``planned_here``, a dataset directory's: a sidecar of a registry image whose compose file gives it no command
    (platoon does not record a pulled image's configuration yet), and one that builds its image from a context, which
    no image map holds for a dataset directory.  A fetched dataset's lock keeps such tasks, which a run refuses when it
    lists them, and a later platoon may run."""

    for name, service in plan.services.items():
        if name == MAIN or service.get("external"):
            continue
        image = service.get("image")
        commandless = service.get("command") is None and service.get("entrypoint") is None
        if planned_here and isinstance(image, str) and commandless:
            return (
                f"whose service {name} runs the command of its registry image {image}, which platoon does not know "
                "yet (a pulled image's configuration is not recorded): give it a command (an overlay's [[compose]] "
                "replace)"
            )
        build = service.get("build")
        if planned_here and isinstance(build, Mapping) and not build.get("key"):
            return (
                f"whose service {name} builds its image from {build.get('context', 'environment')}/"
                f"{build.get('dockerfile', 'Dockerfile')}, and a dataset directory's task has no image map that could "
                "hold it (nothing builds a sidecar's image in this release)"
            )
        for bind in service_binds(service):
            if bind.logs is not None and bind.read_only:
                return (
                    f"whose service {name} binds the {bind.logs} log directory read-only, to read what main writes "
                    "there, which a sidecar does not see (it has a directory of its own, which grading copies into "
                    "main's)"
                )
    binds = [
        (name, bind)
        for name, service in plan.services.items()
        if not service.get("external")
        for bind in service_binds(service)
        if bind.task is not None
    ]
    for index, (name, bind) in enumerate(binds):
        for other_name, other in binds[index + 1 :]:
            path, other_path = os.path.normpath(bind.task or "."), os.path.normpath(other.task or ".")
            same = path == other_path and sorted(bind.masks) == sorted(other.masks)
            writes = not bind.read_only or not other.read_only or bind.missing or other.missing
            if writes and not same and _within(path, other_path):
                return (
                    f"whose services {name} and {other_name} bind the task's {path} and {other_path}, one inside the "
                    "other or with other masks, and one writes there: compose shares one directory, platoon would "
                    "copy them apart"
                )
    return None


def _within(path: str, other: str) -> bool:
    """Whether one of two paths of the task (``.`` for all of it) is the other or below it."""

    if "." in (path, other):
        return True
    return path == other or path.startswith(f"{other}/") or other.startswith(f"{path}/")


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


def _collect_hooks(
    verifier: dict[str, Any], plan: ComposePlan | None, where: str
) -> tuple[tuple[CollectHook, ...], list[str]]:
    """The ``[[verifier.collect]]`` hooks, ``main``'s first and then the sidecars', each in the task's order (as Harbor
    runs them), and the users they name, which the caller's policy decides about.  A hook of a service the task does
    not start refuses the task."""

    hooks: list[CollectHook] = []
    users: list[str] = []
    services = _started_services(plan)
    for entry in _value(verifier, "collect", list, "[verifier]") or []:
        table = "[[verifier.collect]]"
        if not isinstance(entry, dict):
            raise HarborTaskError(f"{where}: {table} entries must be tables, got {entry!r}")
        if unknown := sorted(set(entry) - {"command", "service", "timeout_sec", "user"}):
            raise HarborTaskError(f"{where}: {table} has unknown keys {unknown}")
        command = _value(entry, "command", str, table)
        if not command:
            raise HarborTaskError(f"{where}: {table} needs a command")
        service = _value(entry, "service", str, table) or "main"
        if service not in services:
            raise HarborTaskError(
                f"{where}: {table} runs a hook in the compose service {service!r}, which the task does not start"
            )
        timeout = _value(entry, "timeout_sec", (int, float), table)
        if (user := _value(entry, "user", (str, int), table)) is not None:
            users.append(str(user))
        hooks.append(
            CollectHook(command=command, service=service, timeout_s=60.0 if timeout is None else float(timeout))
        )
    ordered = [hook for hook in hooks if hook.service == "main"] + [hook for hook in hooks if hook.service != "main"]
    return tuple(ordered), users


def _started_services(plan: ComposePlan | None) -> set[str]:
    """The services a stack plan starts (``main`` and its sidecars); ``main`` alone for a task without one."""

    if plan is None or plan.kind != "sidecars":
        return {"main"}
    return {name for name, service in plan.services.items() if not service.get("external")} | {"main"}


def _artifacts(config: dict[str, Any], plan: ComposePlan | None = None) -> tuple[Artifact, ...]:
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
        if service == "main":
            service = None
        if service is not None and service not in _started_services(plan):
            raise HarborTaskError(
                f"artifact {source!r} comes from the compose service {service!r}, which the task does not start"
            )
        if service is not None and not source.startswith("/"):
            raise HarborTaskError(
                f"artifact {source!r} of the compose service {service!r} must be an absolute path in that service"
            )
        exclude = _value(entry, "exclude", list, "artifacts") or []
        if not all(isinstance(pattern, str) for pattern in exclude):
            raise HarborTaskError(f"artifacts.exclude must list patterns, got {exclude!r}")
        artifacts.append(Artifact(source=source, exclude=tuple(exclude), service=service))  # destination: host copies
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
