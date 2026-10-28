"""Environment services: per-rollout sessions over tasks, with an explicit outcome taxonomy.

An :class:`EnvironmentService` lists and loads tasks and opens one :class:`EnvSession`
per rollout. A session carries the task instruction, an optional harness-capable
sandbox, env tools for harnesses that do not act natively, grading and forking.

``Outcome.status`` says whether a measurement is valid, not whether the policy
succeeded: a failed attempt is ``scored`` (with a low reward), while an
infrastructure failure is ``error`` (or ``invalid`` when the env declares the
measurement invalid) and never becomes a reward of 0.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

from platoon.registry import import_from_string
from platoon.sandbox.base import Mount
from platoon.task import Task

if TYPE_CHECKING:
    from platoon.deprecations import DeprecatedKey
    from platoon.sandbox.base import Sandbox, SandboxSpec

logger = logging.getLogger(__name__)

OutcomeStatus = Literal["scored", "invalid", "error", "timeout", "cancelled"]
# Outcome.metadata key for the grader's own reason when the rollout reports another (``context_overflow``).
GRADE_REASON_KEY = "grade_reason"
# Outcome.metadata key that marks a score platoon gave itself rather than read from a grader (True): a Harbor attempt
# whose agents broke what grading reads scores 0 under its reward contract's first name, which no grader reported.
# Such an outcome trained no reward key (``platoon.utils.trajectory_status.trained_reward_key``).
PLATOON_ZERO_KEY = "platoon_zero"
# shared: the same live state; copy: a snapshot restored into a new session; fresh: a new session of the task.
ForkMode = Literal["shared", "copy", "fresh"]


@dataclass
class Outcome:
    """The graded result of a session; only ``scored`` outcomes train."""

    # ``scored`` (a valid measurement), ``invalid``, ``error``, ``timeout`` or ``cancelled``.
    status: OutcomeStatus
    # Named reward components.
    rewards: dict[str, float] = field(default_factory=dict)
    # The component that is the reward; a scored outcome must have it.
    primary: str = "reward"
    # Why the outcome is what it is, e.g. "no_submit", "judge_unavailable".
    reason: str | None = None
    # The infrastructure failure class of an ``error`` or ``invalid`` outcome.
    failure_kind: str | None = None
    # The grader's structured detail.
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status == "scored" and self.primary not in self.rewards:
            raise ValueError(f"A scored outcome needs its primary reward {self.primary!r}; got {sorted(self.rewards)}")

    @property
    def reward(self) -> float | None:
        """The primary reward of a scored outcome; None otherwise."""
        return self.rewards[self.primary] if self.trainable else None

    @property
    def trainable(self) -> bool:
        """Whether the outcome is ``scored``, the only status that trains."""
        return self.status == "scored"

    @classmethod
    def infra(
        cls,
        failure_kind: str,
        *,
        status: OutcomeStatus = "error",
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Outcome:
        """An untrainable outcome for a failure that says nothing about the policy."""

        if status == "scored":
            raise ValueError("Infrastructure failures cannot be scored")
        return cls(status=status, reason=reason, failure_kind=failure_kind, metadata=metadata or {})


@dataclass
class ToolSpec:
    """An env tool for harnesses that do not act natively; ``input_schema`` is a JSON schema.

    The hints drive tool access (filtering happens in the harness layer and the
    MCP plane): a ``terminal`` tool ends the episode (submit, claim_done), so
    agents with ``no_terminal`` access do not see it; a ``read_only`` tool never
    changes env state, and only such tools reach ``read_only`` agents. ``meta``
    passes provider metadata through as MCP ``_meta`` (for example
    ``openhands.dev/tool-routing``).
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    # The tool ends the episode (submit, finish).
    terminal: bool = False
    # The tool never changes env state.
    read_only: bool = False
    # Provider metadata, passed through as MCP ``_meta``.
    meta: Mapping[str, Any] | None = None


@dataclass
class ToolResult:
    """An env tool's reply as MCP-style content blocks, plus the episode signals some envs send.

    ``finished`` says the env ended the episode, ``reward`` is the reward the
    env attached to this call, and ``metadata`` is the env's structured detail
    (``{"invalid": true}`` marks a grade the env declares invalid).
    """

    # MCP content blocks, such as ``{"type": "text", "text": ...}``.
    content: list[dict[str, Any]]
    # The call failed; the content says why.
    is_error: bool = False
    finished: bool = False
    reward: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def text(cls, text: str, *, is_error: bool = False) -> ToolResult:
        """A result with one text block."""
        return cls(content=[{"type": "text", "text": text}], is_error=is_error)

    def as_text(self) -> str:
        """The content as one string: text blocks joined by newlines, other blocks as placeholders."""

        parts = []
        for block in self.content:
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(f"[{block.get('mimeType') or block.get('type', 'block')}]")
        return "\n".join(parts)


@dataclass(frozen=True)
class SessionNeeds:
    """What the harness drawn for a rollout needs from the session's sandbox; the rollout computes it before the
    sandbox starts, and ``EnvironmentService.open`` applies it to the sandbox it opens (:meth:`apply`).

    ``upstreams`` are the host services the harness's processes must reach (``{"gateway": "unix:///…/gw.sock"}``:
    ``SandboxSpec.upstreams``, answered by ``Sandbox.routes``); ``mounts`` are binds such as the harness's toolchain,
    read-only at its build path; ``extra_memory_mb`` is the memory of the harness's own processes (its spec's
    ``[resources] memory_mb``), added to the sandbox's memory limit when it has one.
    """

    # Upstream name -> a host endpoint the harness's processes must reach (SandboxSpec.upstreams).
    upstreams: Mapping[str, str] = field(default_factory=dict)
    # Binds the harness needs, such as its toolchain, read-only at its build path.
    mounts: tuple[Mount, ...] = ()
    # The memory of the harness's own processes, added to the sandbox's memory limit when it has one.
    extra_memory_mb: int = 0

    def apply(self, spec: SandboxSpec) -> SandboxSpec:
        """``spec`` with these needs: their upstreams and mounts added (a mount whose target the spec already binds
        keeps the spec's), and the memory limit grown."""

        targets = {mount.target for mount in spec.mounts}
        memory_mb = spec.memory_mb
        if memory_mb is not None and self.extra_memory_mb > 0:
            memory_mb += self.extra_memory_mb
        return dataclasses.replace(
            spec,
            mounts=[*spec.mounts, *(mount for mount in self.mounts if mount.target not in targets)],
            upstreams={**spec.upstreams, **self.upstreams},
            memory_mb=memory_mb,
        )


class EnvSetupError(RuntimeError):
    """``EnvironmentService.open`` could not set up a session; ``failure_kind`` classifies why."""

    def __init__(self, failure_kind: str, message: str) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind


class EnvNotPrepared(EnvSetupError):
    """What a run needs and ``platoon prepare`` makes, such as a task's image, is missing: a configuration error
    (``env_config``), which a run reports with the ``platoon prepare`` command for its config."""

    def __init__(self, message: str) -> None:
        super().__init__("env_config", message)


@runtime_checkable
class EnvSession(Protocol):
    """One rollout's live view of a task.

    A session may set ``manages_concurrency = True`` (an optional attribute, so it is not part of the protocol's
    runtime check): its ``call_tool`` then takes calls from any number of agents at once, on the session and its
    shared forks, and orders them on the live env itself.  ORS and Harbor sessions do, through their live env's
    gate (:mod:`platoon.envs.gate`).  The tool plane (:mod:`platoon.gateway.mcp`) calls such a session directly,
    naming the calling agent (:func:`platoon.envs.gate.on_behalf_of`), and any other one call at a time per tool
    provider.

    A session with a sandbox may also set ``network_isolated = True`` (optional too): its sandbox has no network,
    so a harness placed in it could not reach the model gateway on loopback.  The harness runner then reaches the
    gateway through the sandbox's route (``session.sandbox.routes["gateway"]``, which the session's
    :class:`SessionNeeds` asked for), and refuses to start a harness there without one.  A session may likewise set
    ``host_loopback`` (optional): whether its sandbox shares this host's network, where the gateway listens on
    loopback.  The runner refuses to place a harness in a sandbox that does not (``host_loopback = False``) and has
    no route; a session that does not say keeps the earlier rule, that a sandbox with a network shares the host's.
    A session may also give ``agent_timeout_s`` (optional): the task's own time limit for its agent in seconds, or
    None, which harness rollouts cap the root's harness with, as they do with the rollout's ``timeout``.  And
    ``task_mcp_servers`` (optional): the MCP servers the task itself offers, as a harness in its sandbox reaches them
    (each a mapping with ``name``, ``transport``, and ``url`` or ``command`` and ``args``), which harness specs render
    as ``{task_mcp_servers}``.
    """

    # The task of the session.
    task: Task
    # The instruction the root agent starts from.
    instruction: str
    # A task sandbox that harnesses can run in with their own (native) tools, or None.
    sandbox: Sandbox | None

    async def tools(self) -> list[ToolSpec]:
        """Env tools for harnesses that do not act natively."""
        ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Call one env tool."""
        ...

    async def grade(self) -> Outcome:
        """Grade the current state; infra failures return error/invalid outcomes instead of raising."""
        ...

    async def fork(self, mode: ForkMode = "shared", *, task: Task | None = None) -> EnvSession:
        """A session for a child agent: ``shared`` (the same live state), ``copy`` (a copy of it) or ``fresh`` (a new
        session), on ``task`` when given."""
        ...

    async def close(self) -> None:
        """Release the session's resources; idempotent."""
        ...


@runtime_checkable
class EnvironmentService(Protocol):
    """A source of tasks that opens sessions for rollouts.

    A service may also have two async methods, which are optional and so not part of the protocol's runtime
    check: ``list_splits()``, its split names, and ``list_task_names(split)``, each task's name by task id, for a
    service whose task names are not its task ids.  An environment entry's ``exclude_tasks`` matches task names
    through ``list_task_names``, and looks for a task that the listed splits do not hold in the other splits that
    ``list_splits`` names.

    It may also have ``check_tasks(split, task_ids)``, async and optional too: it raises an ``env_config``
    :class:`EnvSetupError` for tasks among ``task_ids`` that it would refuse to run, which a run calls once
    ``exclude_tasks`` left those of ``split`` it keeps, so that the run stops at its start.  And two synchronous
    ones for deadlines on whole rollouts: ``rollout_overhead_s(task_id) -> float``, the seconds a rollout of the task
    may take beyond its harness's time limit, such as starting sandboxes and running a verifier, which those
    deadlines add (:func:`platoon.train.components.rollout_overhead_s`); and
    ``agent_timeout_s(task_id) -> float | None``, the task's own time limit for its agent (its sessions'
    ``agent_timeout_s``), which they count when the rollout sets no ``timeout``
    (:func:`platoon.train.components.harness_time_limit_s`), as long as the service has ``rollout_overhead_s``.
    """

    # The service's name.
    name: str

    async def list_tasks(self, split: str) -> list[str]:
        """The ids of the tasks in ``split``."""
        ...

    async def get_task(self, task_id: str) -> Task:
        """Load one task by id."""
        ...

    async def open(self, task: Task, *, rollout_id: str) -> EnvSession:
        """Open a session on ``task`` for one rollout; a setup failure raises ``EnvSetupError``.

        A service whose sessions have a sandbox may also take ``needs``: what the rollout's harness needs from that
        sandbox (:class:`SessionNeeds`).  :func:`open_session` passes them to a service that takes the keyword, and
        opens the sessions of one that does not, such as a service that predates ``needs``, without them.
        """
        ...


class _OpensWithNeeds(Protocol):
    """A service whose ``open`` takes ``needs`` (:func:`takes_needs`)."""

    async def open(self, task: Task, *, rollout_id: str, needs: SessionNeeds | None = None) -> EnvSession: ...


_TAKES_NEEDS: dict[type, bool] = {}


def takes_needs(service: EnvironmentService) -> bool:
    """Whether ``service.open`` takes the ``needs`` keyword (or any keyword)."""

    kind = type(service)
    if kind not in _TAKES_NEEDS:
        try:
            parameters = inspect.signature(service.open).parameters
        except (TypeError, ValueError):
            parameters = {}
        _TAKES_NEEDS[kind] = "needs" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        if not _TAKES_NEEDS[kind]:
            logger.info(
                "Environment service %s does not take session needs: its sandboxes get no routes, toolchain "
                "mounts or harness memory",
                kind.__qualname__,
            )
    return _TAKES_NEEDS[kind]


async def open_session(
    service: EnvironmentService, task: Task, *, rollout_id: str, needs: SessionNeeds | None = None
) -> EnvSession:
    """``service.open`` with ``needs``, or without them for a service that does not take them."""

    if needs is not None and takes_needs(service):
        return await cast(_OpensWithNeeds, service).open(task, rollout_id=rollout_id, needs=needs)
    return await service.open(task, rollout_id=rollout_id)


_BUILTIN_SERVICES = {
    "harbor": "platoon.envs.harbor:HarborEnvironmentService",
    "table": "platoon.envs.table:TaskTableService",
}
# ORS environments: an in-process Environment class, or env servers over HTTP (one URL or a pool).
_ORS_SCHEMES = ("python://", "ors+http://", "ors+https://")


def service_target(name: str) -> str | None:
    """The ``module:attr`` a service reference builds from, without building it: a core service's class for a
    built-in name, the ORS ``Environment`` class of ``python://``, the class of ``module:Class``; None for ORS env
    servers over HTTP, whose code is the servers' own."""

    if name.startswith("python://"):
        return name.removeprefix("python://")
    if name.startswith(_ORS_SCHEMES):
        return None
    return _BUILTIN_SERVICES.get(name, name)


@dataclass(frozen=True)
class PathOption:
    """How a service option that names files platoon reads resolves when a config file gives it a relative path
    (RFC 0018, section 5).

    A service class lists such options in a class attribute, ``path_options``: a tuple of option names, or a mapping
    of option names to ``PathOption``.  A name may reach into each mapping of a list, as ``mounts[].source``, into a
    mapping, as ``adapter.repo``, or into each value of a mapping, with ``*``, as ``node_services.*.binds[].source``.
    The config loader (:mod:`platoon.utils.config`) makes a relative path absolute against the directory of the config
    file that gives it, before the files merge, and refuses one that names nothing there; an override's path is
    relative to the working directory.  A value that is not a local path is left alone: a mapping (an inline split
    manifest), a URL (``hf://...``), ``git+...`` or a git address (``git@host:org/repo``).
    """

    # Another option of the service whose directory the path is relative to when the options give that one (a split
    # manifest inside a dataset directory): the path is then left as written.
    relative_to: str | None = None
    # The run may write the file when it does not exist yet (a lock that a fetch writes), so a path that names nothing
    # is refused only when it names a file relative to the working directory, as it did before.
    written: bool = False


def service_path_options(name: str) -> dict[str, PathOption]:
    """The options of the service reference ``name`` that name files platoon reads (its class's ``path_options``),
    read without building the service; none for a factory that does not import, and for ORS environments, whose
    options name no files."""

    if name.startswith(_ORS_SCHEMES):
        return {}
    target = service_target(name)
    try:
        factory = import_from_string(target) if target is not None else None
    except Exception:  # a reference that does not import is reported where the service resolves
        return {}
    declared = getattr(factory, "path_options", None) or ()
    if isinstance(declared, Mapping):
        return {
            str(option): rule if isinstance(rule, PathOption) else PathOption() for option, rule in declared.items()
        }
    return {str(option): PathOption() for option in declared}


def service_deprecated_options(name: str) -> dict[str, DeprecatedKey]:
    """The options of the service reference ``name`` that are going away (its class's ``DEPRECATED_KEYS``,
    :mod:`platoon.deprecations`), read without building the service; none for a factory that does not import."""

    from platoon.deprecations import deprecated_keys

    if name.startswith(_ORS_SCHEMES):
        return {}
    target = service_target(name)
    try:
        factory = import_from_string(target) if target is not None else None
    except Exception:  # a reference that does not import is reported where the service resolves
        return {}
    return dict(deprecated_keys(factory))


def get_service(name: str, /, **options: Any) -> EnvironmentService:
    """Build an environment service from a name, a URI or a ``module:Class`` path.

    * a built-in name: ``"harbor"``, ``"table"``;
    * ``python://pkg.mod:EnvClass``: an ORS ``Environment`` class run in process;
    * ``ors+http://<host:port>[,<host:port>...]/<EnvName>``: ORS env servers (``ors+https`` too);
    * ``module:Class``: any other service class, built with ``options``.
    """

    if name.startswith(_ORS_SCHEMES):
        from platoon.envs.ors import OrsEnvironmentService

        return OrsEnvironmentService.from_uri(name, **options)
    factory = import_from_string(_BUILTIN_SERVICES.get(name, name))
    return factory(**options)


# The keys a service's task listing takes from an entry's dataset_kwargs and eval_dataset_kwargs, when the entry names
# no dataset loader of its own (``platoon.train.auto._service_task_ids``).
LISTING_KEYS = ("split_name", "shuffle_seed", "limit")


def service_option_problems(name: str, options: Mapping[str, Any], *, where: str) -> list[str]:
    """What is wrong with ``options``, an entry's ``service_kwargs``, as the keyword arguments of the factory the
    service reference ``name`` builds from, bound to its signature without building it: a key it does not take, with
    the hint its class gives for it (``UNKNOWN_KEY_HINTS``, such as an option a release removed) or a did-you-mean; a
    key the reference itself gives (an ORS URI's environment); a required argument that is missing.  ``where`` is the
    options' dotted path.  The values are the service's to check when it is built.  None for a factory that does not
    import (the reference's own problem) or that takes any keyword."""

    from platoon.utils.config import UNKNOWN_KEY_HINTS, unknown_key

    found = _factory_signature(name)
    if found is None:
        return []
    factory, signature, given = found
    parameters = signature.parameters.values()
    takes_any = any(parameter.kind is parameter.VAR_KEYWORD for parameter in parameters)
    keywords = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    known = [parameter.name for parameter in parameters if parameter.kind in keywords and parameter.name not in given]
    hints = getattr(factory, UNKNOWN_KEY_HINTS, None) or {}
    problems = []
    for key in options:
        if key in given:
            problems.append(f"{where}.{key}: the service reference {name!r} gives it; leave it out of service_kwargs")
        elif not takes_any and key not in known:
            takes = f"the {name} service takes {', '.join(known)}"
            problems.append(unknown_key(key, known, where, hint=hints.get(key), otherwise=takes))
    if problems:
        return problems
    try:
        signature.bind(**given, **options)
    except TypeError as exc:  # a required argument that neither the reference nor the options give
        problems.append(f"{where}: {exc}")
    return problems


def service_option_hints(name: str) -> dict[str, str]:
    """The options that the factory the service reference ``name`` builds from does not take, but that its class says
    what to do instead of (``UNKNOWN_KEY_HINTS``, such as an option a release removed), each with that hint; read
    without building the service.  None for ORS environments, whose service names none, and for a factory that does
    not import or that takes any keyword.  The config loader refuses each that an entry's ``service_kwargs`` give
    (:class:`platoon.train.components.EnvironmentConfig`), as :func:`service_option_problems` does."""

    from platoon.utils.config import UNKNOWN_KEY_HINTS

    if name.startswith(_ORS_SCHEMES):
        return {}
    found = _factory_signature(name)
    if found is None:
        return {}
    factory, signature, _ = found
    hints = getattr(factory, UNKNOWN_KEY_HINTS, None) or {}
    parameters = signature.parameters.values()
    if not hints or any(parameter.kind is parameter.VAR_KEYWORD for parameter in parameters):
        return {}
    return {str(key): hint for key, hint in hints.items() if key not in signature.parameters}


def _factory_signature(name: str) -> tuple[Any, inspect.Signature, dict[str, Any]] | None:
    """The factory the service reference ``name`` builds from, its signature, and the arguments the reference gives it
    itself (an ORS URI's environment); None for a factory that does not import, which is reported where the service
    resolves, or whose signature Python cannot read."""

    given: dict[str, Any] = {}
    if name.startswith(_ORS_SCHEMES):
        from platoon.envs.ors import OrsEnvironmentService

        factory: Any = OrsEnvironmentService
        if name.startswith("python://"):
            given["environment"] = name.removeprefix("python://")
        else:
            given["env_name"] = name
    else:
        try:
            factory = import_from_string(_BUILTIN_SERVICES.get(name, name))
        except Exception:
            return None
    try:
        return factory, inspect.signature(factory), given
    except (TypeError, ValueError):
        return None


def listing_option_problems(options: Mapping[str, Any], *, where: str) -> list[str]:
    """The keys of an entry's ``dataset_kwargs`` (or ``eval_dataset_kwargs``) that its service's task listing does
    not take (:data:`LISTING_KEYS`), each with a did-you-mean; ``where`` is the options' dotted path."""

    from platoon.utils.config import unknown_key

    takes = ", ".join(LISTING_KEYS)
    return [
        unknown_key(key, LISTING_KEYS, where, otherwise=f"a service's task listing takes {takes}")
        for key in options
        if key not in LISTING_KEYS
    ]
