"""``OrsEnvironmentService``: ORS (OpenReward) environments as platoon environment services.

Two backends serve the same sessions:

* ``python://pkg.mod:EnvClass`` runs an ``openreward.environments.Environment``
  subclass in process, one instance per session, driven like the SDK's server
  drives it (constructor, ``setup``, ``get_prompt``, tools, ``teardown``).
* ``ors+http://<host:port>[,<host:port>...]/<EnvName>`` opens sessions on ORS
  env servers through the openreward client. A pool is sharded by rollout id
  (:mod:`platoon.envs.ors.pool`); with no hosts (``ors+http:///<EnvName>``) the
  pool comes from ``OPENREWARD_SESSION_URLS_<LABEL>`` and its fallbacks.
  ``<EnvName>`` names the deployment: a server answers requests that name no
  ``variant`` with its default (first) environment, so ``?variant=<Env>``
  selects one of a multi-environment server's, and a name that is one of its
  other environments is refused (:meth:`OrsEnvironmentService.check_server_environment`).

Task ids are ``<split>:<index>``, so a worker resolves its task without listing
the catalog. ``get_task`` also takes a task name (the first of ``task_name``,
``id``, ``task_id`` and ``name`` that the task spec sets) and an OpenReward
task reference (``openreward:v1:`` and an encoded environment, split and index,
see :meth:`TaskRef.from_reference`).

An env class can steer platoon without importing it, through optional
attributes: ``platoon_terminal_tools``, ``platoon_read_only_tools`` and
``platoon_concurrent_tools`` (tool names), ``platoon_tool_meta``
(``{tool: _meta}``), ``platoon_fork(task_spec)`` (a ``shared`` fork with its
own goal on shared state), ``platoon_copy()`` (the env a ``copy`` fork gets)
and ``platoon_grade()`` (a grade from state when nothing was submitted). They
work in process only, as do the SDK's ``@tool(exclusive=True)`` and
``@tool(concurrent=True)`` and a tool's ``_meta`` (``readOnlyHint`` among
them): an env server sends tool names, schemas and ``@terminal`` (its name and
argument), nothing else (openreward up to 0.1.163 at least). So a remote env
names its terminal, read-only and concurrent tools with the
``terminal_tools``/``read_only_tools``/``concurrent_tools`` options and its
tools' meta with ``tool_meta``, or names the class its env server serves
(``env_class``): the service then reads the hints and the tools' descriptors
from that class as it does in process, and checks that the server lists the
class's tools (:meth:`OrsEnvironmentService.check_served_class`). The hooks
need the live env, so they still run in process only.

Agents that share a live env take turns on it (:mod:`platoon.envs.gate`): its
tools run one at a time unless the env declares them safe to overlap
(:meth:`OrsEnvironmentService.call_class`).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import importlib
import inspect
import json
import logging
import os
import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

from platoon.envs.gate import DEFAULT_QUEUE_TIMEOUT_S, CallClass, EnvBusy, GateClosed, ToolGate, check_queue_timeout
from platoon.envs.ors import sdk
from platoon.envs.ors.pool import pool_env_var, resolve_session_pool, select_session_url
from platoon.envs.ors.session import EventLog, Lease, LocalBackend, OrsSession, RemoteBackend, ResultFormat, bounded
from platoon.envs.service import EnvSetupError, ToolSpec
from platoon.registry import import_from_string
from platoon.task import Task

logger = logging.getLogger(__name__)

ORS_MISC_KEY = "ors"  # Task.misc entry: {"split", "index", "name"}
OPENREWARD_TASK_REFERENCE_PREFIX = "openreward:v1:"
_INDEX_ID = re.compile(r"^(?P<split>.+):(?P<index>\d+)$")
_REFERENCE_FIELDS = frozenset({"environment", "index", "split"})
_URI_QUERY_KEYS = frozenset({"variant", "split", "label"})
_RESULT_FORMATS = ("blocks", "bridge_json")
# (env_name, variant, urls) of the remote envs already reported as declaring no terminal tool.
_WARNED_NO_TERMINAL_TOOL: set[tuple[str | None, str | None, tuple[str, ...]]] = set()
# The env classes already reported as defining hooks that an env server cannot run (check_served_class).
_WARNED_SERVED_HOOKS: set[type] = set()
# What check_served_class found per (server, env name, variant, class): None when the server lists the class's tools,
# else why not.  One check per server and process; the sessions that open while it runs wait for it.
_SERVED_CLASS_CHECKS: dict[tuple[str, str, str | None, type], str | None] = {}
_SERVED_CLASS_CHECKS_IN_FLIGHT: dict[tuple[str, str, str | None, type], asyncio.Task[str | None]] = {}
# The environments each env server listed (OrsEnvironmentService.check_server_environment), lowercased, its default
# first; empty for a server that lists none or could not be asked.  One request per server and process.
_SERVER_ENVIRONMENTS: dict[str, tuple[str, ...]] = {}
# The listing request under way for each server: the sessions that open while it is asked wait for its answer.
_LISTINGS_IN_FLIGHT: dict[str, asyncio.Task[tuple[str, ...]]] = {}
_LIST_ENVIRONMENTS_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class TaskRef:
    """Where a task lives in its environment: a split and an index (plus its display name when known)."""

    split: str
    index: int
    name: str | None = None

    @property
    def task_id(self) -> str:
        return f"{self.split}:{self.index}"

    def to_misc(self) -> dict[str, Any]:
        return {"split": self.split, "index": self.index, "name": self.name}

    @classmethod
    def from_misc(cls, value: Any) -> TaskRef | None:
        if not isinstance(value, Mapping) or not isinstance(value.get("split"), str):
            return None
        index = value.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            return None
        name = value.get("name")
        return cls(value["split"], index, name if isinstance(name, str) else None)

    @classmethod
    def parse(cls, task_id: str, *, label: str | None = None) -> TaskRef | None:
        """A ``<split>:<index>`` id or an OpenReward task reference; None for anything else (a name).

        A reference names its environment's label; with ``label`` given, one for another environment is refused.
        """

        if task_id.startswith(OPENREWARD_TASK_REFERENCE_PREFIX):
            return cls.from_reference(task_id, label=label)
        match = _INDEX_ID.match(task_id)
        return cls(match["split"], int(match["index"])) if match else None

    @classmethod
    def from_reference(cls, reference: str, *, label: str | None = None) -> TaskRef:
        """An OpenReward task reference: ``openreward:v1:`` and the URL-safe base64 (padding optional) of a JSON
        object with ``environment``, ``index`` and ``split`` and an optional ``name``, every string non-empty and the
        index a non-negative int.  Raises ``ValueError`` for anything else, and for a reference to an environment
        other than ``label``."""

        token = reference.removeprefix(OPENREWARD_TASK_REFERENCE_PREFIX)
        try:
            if not token:
                raise ValueError("the payload is empty")
            payload = json.loads(base64.b64decode(token + "=" * (-len(token) % 4), altchars=b"-_", validate=True))
        except ValueError as exc:
            raise ValueError(f"Malformed OpenReward task reference {reference!r}: {exc}") from exc
        fields = set(payload) if isinstance(payload, dict) else set()
        if not _REFERENCE_FIELDS <= fields <= _REFERENCE_FIELDS | {"name"}:
            raise ValueError(
                f"Malformed OpenReward task reference {reference!r}: expected the fields environment, index and split "
                "and an optional name"
            )
        environment, split, index, name = (
            payload["environment"],
            payload["split"],
            payload["index"],
            payload.get("name"),
        )
        if not all(isinstance(value, str) and value for value in (environment, split)) or (
            "name" in payload and not (isinstance(name, str) and name)
        ):
            raise ValueError(f"Malformed OpenReward task reference {reference!r}: its strings must be non-empty")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(f"Malformed OpenReward task reference {reference!r}: index must be a non-negative integer")
        if label is not None and environment != label:
            raise ValueError(
                f"OpenReward task reference {reference!r} is for environment {environment!r}, not {label!r}"
            )
        return cls(split, index, name)


class OrsEnvironmentService:
    """Serve an ORS environment, in process or from env servers (see the module docstring).

    Remote settings: ``env_name`` and ``variant`` (the deployment and the
    environment it serves), ``api_key`` (default ``$OPENREWARD_API_KEY`` or
    ``local``), ``api_url``, and the URL pool (``urls``, else ``label``'s
    variable, see :mod:`~platoon.envs.ors.pool`; set
    ``legacy_global_pool=False`` for mixture entries). For every backend:
    ``split`` is the default split for names; ``task_names`` or
    ``task_indices`` narrow ``list_tasks``; ``max_tool_calls`` caps calls per
    session (0: no cap); ``terminal_tools``/``read_only_tools`` and
    ``tool_meta``/``tool_routing_overrides`` add tool hints;
    ``concurrent_tools`` names tools that may run while other calls run (see
    :meth:`call_class`); ``queue_timeout_s`` (default 240 s, under OpenHands'
    300 s MCP deadline; None: no limit) is how long a call or a grade waits for
    its turn on a busy env before it is refused without running;
    ``result_format="bridge_json"`` renders each result as a stdio MCP bridge to
    ORS env servers shows it to the model (see :mod:`~platoon.envs.ors.session`),
    and ``bridge_tools=True`` adds that bridge's ``get_task``/``get_status``
    tools; ``log_dir`` writes the bridge's per-call JSONL under
    ``<log_dir>/<rollout id>/``; ``close_timeout_s`` bounds teardown.
    ``prompt_filter`` (a callable or a ``module:attr`` reference) rewrites the
    env's prompt text, which becomes the instruction and ``get_task``'s
    prompt: say, to drop the stale completion line of old Toolathlon images.
    ``env_class`` (remote only; a class or a ``module:Class`` reference) names
    the class the env servers serve: its hints and tool descriptors then apply
    as in process (see the module docstring).
    """

    def __init__(
        self,
        environment: type | str | None = None,
        *,
        env_name: str | None = None,
        urls: Sequence[str] | str | None = None,
        split: str = "train",
        variant: str | None = None,
        api_key: str | None = None,
        api_url: str | None = None,
        label: str | None = None,
        session_urls_env_var: str | None = None,
        legacy_global_pool: bool = True,
        max_tool_calls: int = 0,
        task_names: Sequence[str] | None = None,
        task_indices: Sequence[int] | None = None,
        terminal_tools: Sequence[str] = (),
        read_only_tools: Sequence[str] = (),
        concurrent_tools: Sequence[str] = (),
        queue_timeout_s: float | None = DEFAULT_QUEUE_TIMEOUT_S,
        tool_meta: Mapping[str, Mapping[str, Any]] | None = None,
        tool_routing_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        result_format: ResultFormat = "blocks",
        bridge_tools: bool = False,
        log_dir: str | Path | None = None,
        close_timeout_s: float = 30.0,
        secrets: Mapping[str, str] | None = None,
        name: str | None = None,
        prompt_filter: str | Callable[[str], str] | None = None,
        env_class: type | str | None = None,
    ) -> None:
        if (environment is None) == (env_name is None):
            raise ValueError("An ORS service needs an Environment class (in process) or an env_name (remote)")
        if environment is not None and (urls or variant):
            raise ValueError("urls and variant select env servers; an in-process Environment takes neither")
        if environment is not None and env_class is not None:
            raise ValueError("env_class names the class an env server serves; an in-process Environment is its own")
        if isinstance(max_tool_calls, bool) or not isinstance(max_tool_calls, int) or max_tool_calls < 0:
            raise ValueError(f"max_tool_calls must be a non-negative integer, got {max_tool_calls!r}")
        if task_names is not None and task_indices is not None:
            raise ValueError("Select tasks by task_names or task_indices, not both")
        if task_indices is not None and any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in task_indices
        ):
            raise ValueError("task_indices must be non-negative integers")
        if result_format not in _RESULT_FORMATS:
            raise ValueError(f"result_format must be one of {_RESULT_FORMATS}, got {result_format!r}")
        if not close_timeout_s > 0:
            raise ValueError("close_timeout_s must be positive")
        self.queue_timeout_s = check_queue_timeout(queue_timeout_s)
        if isinstance(prompt_filter, str):
            prompt_filter = import_from_string(prompt_filter)
        if prompt_filter is not None and not callable(prompt_filter):
            raise ValueError(f"prompt_filter must be a callable or a module:attr reference, got {prompt_filter!r}")
        self._environment = environment
        self.env_name = env_name
        self.urls = [urls] if isinstance(urls, str) else list(urls or [])
        self.split = split
        self.variant = variant
        self.api_key = api_key if api_key is not None else os.environ.get("OPENREWARD_API_KEY", "local")
        self.api_url = api_url if api_url is not None else os.environ.get("OPENREWARD_API_URL")
        self.label = (label or env_name or "").strip() or None
        self.session_urls_env_var = session_urls_env_var
        self.legacy_global_pool = legacy_global_pool
        self.max_tool_calls = max_tool_calls
        self.task_names = list(task_names) if task_names is not None else None
        self.task_indices = list(task_indices) if task_indices is not None else None
        self.terminal_tools = frozenset(terminal_tools)
        self.read_only_tools = frozenset(read_only_tools)
        self.concurrent_tools = _names(concurrent_tools)
        self.tool_meta = {tool: dict(meta) for tool, meta in (tool_meta or {}).items()}
        self.tool_routing_overrides = {tool: dict(route) for tool, route in (tool_routing_overrides or {}).items()}
        self.result_format: ResultFormat = result_format
        self.bridge_tools = bridge_tools
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.close_timeout_s = float(close_timeout_s)
        self.secrets = dict(secrets or {})
        # Tools seen ending an episode: the terminal-tool fallback for envs that declare none (see tool_specs).
        self.learned_terminal: set[str] = set()
        self.name = name or self.label or self._class_name()
        self.prompt_filter: Callable[[str], str] | None = prompt_filter
        self._env_class = env_class

    @property
    def reported_name(self) -> str:
        """The env name the bridge reports (``get_status``, the event log): ``env_name``, else :attr:`name`."""

        return self.env_name or self.name

    @classmethod
    def from_uri(cls, uri: str, **options: Any) -> OrsEnvironmentService:
        """``python://pkg.mod:EnvClass`` or ``ors+http(s)://<host:port>[,...]/<EnvName>[?variant=&split=&label=]``."""

        if uri.startswith("python://"):
            return cls(uri.removeprefix("python://"), **options)
        parts = urlsplit(uri)
        if parts.scheme not in ("ors+http", "ors+https"):
            raise ValueError(f"Unknown ORS URI scheme in {uri!r}; expected python://, ors+http:// or ors+https://")
        env_name = unquote(parts.path.strip("/"))
        if not env_name:
            raise ValueError(f"{uri!r} names no environment; expected ors+http://<host:port>/<EnvName>")
        query = dict(parse_qsl(parts.query))
        if unknown := sorted(set(query) - _URI_QUERY_KEYS):
            raise ValueError(f"Unknown query parameters {unknown} in {uri!r}; known: {sorted(_URI_QUERY_KEYS)}")
        scheme = parts.scheme.removeprefix("ors+")
        hosts = [f"{scheme}://{host.strip()}" for host in parts.netloc.split(",") if host.strip()]
        options.setdefault("urls", hosts or None)
        return cls(env_name=env_name, **{**query, **options})

    @property
    def is_remote(self) -> bool:
        return self.env_name is not None

    @cached_property
    def environment_class(self) -> type:
        """The in-process Environment class (imported on first use).

        Loading it quiets the SDK's INFO logs in this process (:func:`~platoon.envs.ors.sdk.quiet_in_process_logs`):
        the env's tool calls run here, and their ``tool_call_completed`` lines would reach harness output.
        """

        if self._environment is None:
            raise TypeError(f"ORS service {self.name!r} serves a remote environment")
        sdk.import_sdk()  # also routes the SDK's own sync env calls to platoon's threads
        value: Any = self._environment
        if isinstance(value, str):
            module, _, attr = value.replace(":", ".").rpartition(".")
            if not module or not attr:
                raise ValueError(f"Expected python://package.module:EnvClass, got {self._environment!r}")
            value = importlib.import_module(module)
            for part in attr.split("."):
                value = getattr(value, part)
        sdk.quiet_in_process_logs()
        return _own_task_cache(value)

    @cached_property
    def served_class(self) -> type | None:
        """The class the env servers serve (``env_class``), imported on first use; None when it is not named."""

        if self._env_class is None:
            return None
        if isinstance(self._env_class, str):
            return import_from_string(self._env_class)
        return self._env_class

    def hint(self, lease: Lease, name: str) -> Any:
        """A hint attribute of the env (``platoon_terminal_tools``, ``platoon_read_only_tools``,
        ``platoon_concurrent_tools``, ``platoon_tool_meta``): the live env's in process, the served class's on an env
        server (``env_class``), else None."""

        if lease.backend.url is None:
            return lease.backend.env_attr(name)
        cls = self.served_class
        return None if cls is None else getattr(cls, name, None)

    async def check_served_class(self, url: str) -> None:
        """Refuse an ``env_class`` that the env server at ``url`` does not serve as this process has it.

        The server's listing of the class (``GET /tools``: its tools, and 0.1.163's ``@terminal`` descriptor) must
        name the same tools, with the same descriptions and schemas, as ``env_class.list_tools()`` here
        (:func:`sdk.class_listing_differences`); otherwise the hints and descriptors read from the class could
        describe tools that the server does not serve, so this raises :class:`EnvSetupError` (``env_config``), as it
        does for a class that does not import.  A server that cannot be asked passes (best effort).  One check per
        server and process.  A class that defines hooks, which run only in process, is reported once.
        """

        if self._env_class is None or not self.env_name:
            return
        try:
            cls = self.served_class
        except Exception as exc:
            raise EnvSetupError(
                "env_config", f"env_class {self._env_class!r} does not import: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(cls, type) or not callable(getattr(cls, "list_tools", None)):
            raise EnvSetupError("env_config", f"env_class {self._env_class!r} is not an openreward Environment class")
        _warn_served_hooks(cls, self.env_name)
        key = (url, self.env_name, self.variant, cls)
        problem = _SERVED_CLASS_CHECKS.get(key)
        if key not in _SERVED_CLASS_CHECKS:
            loop = asyncio.get_running_loop()
            pending = _SERVED_CLASS_CHECKS_IN_FLIGHT.get(key)
            if pending is None or pending.done() or pending.get_loop() is not loop:
                pending = _SERVED_CLASS_CHECKS_IN_FLIGHT[key] = loop.create_task(self._compare_served_class(url, cls))
                pending.add_done_callback(lambda task: _check_done(_SERVED_CLASS_CHECKS_IN_FLIGHT, key, task))
            problem = await asyncio.shield(pending)  # a session open that is cancelled leaves the check to the others
        if problem is not None:
            raise EnvSetupError("env_config", problem)

    async def _compare_served_class(self, url: str, cls: type) -> str | None:
        key = (url, self.env_name or "", self.variant, cls)
        client = sdk.import_client()
        api = client.AsyncEnvironmentsAPI(base_url=url, api_key=self.api_key, api_base_url=self.api_url or url)
        try:
            environment = api.get(self.env_name or "", variant=self.variant)
            tools = await asyncio.wait_for(environment.list_tools(), _LIST_ENVIRONMENTS_TIMEOUT_S)
            terminal = None
            if callable(getattr(environment, "terminal_tool", None)):  # 0.1.163; 0.1.125 has no @terminal
                terminal = await asyncio.wait_for(environment.terminal_tool(), _LIST_ENVIRONMENTS_TIMEOUT_S)
        except Exception:
            logger.debug("Cannot list the tools %s serves to compare them with %s", url, cls, exc_info=True)
            _SERVED_CLASS_CHECKS[key] = None
            return None
        finally:
            await bounded(api.aclose(), self.close_timeout_s, f"close the {self.name} tool listing client")
        try:
            differences = sdk.class_listing_differences(cls, tools, terminal)
        except Exception as exc:
            differences = [f"the class cannot list its tools ({type(exc).__name__}: {exc})"]
        problem = None
        if differences:
            problem = (
                f"The env server {url} serves ORS env {self.env_name!r} with other tools than env_class "
                f"{cls.__module__}.{cls.__qualname__} has here: {'; '.join(differences)}. Serve the class's version "
                "that this process imports, or drop env_class (and name the hints with terminal_tools, "
                "read_only_tools, concurrent_tools and tool_meta)"
            )
        _SERVED_CLASS_CHECKS[key] = problem
        return problem

    def session_urls(self) -> list[str]:
        """The env-server pool, resolved now (launchers may export it after config load)."""

        return self.session_pool()[0]

    def session_pool(self) -> tuple[list[str], str | None]:
        """:meth:`session_urls`, and the process-wide source it fell back to (``OPENREWARD_SESSION_URLS``,
        ``OPENREWARD_SESSION_URL`` or the default URL); None when the pool is this environment's own, its URLs or
        :meth:`own_pool_env_var`'s (:func:`~platoon.envs.ors.pool.resolve_session_pool`)."""

        return resolve_session_pool(
            urls=self.urls,
            label=self.label or "",
            env_var=self.session_urls_env_var,
            legacy_global_pool=self.legacy_global_pool,
        )

    def own_pool_env_var(self) -> str:
        """The variable this environment's own pool comes from: ``session_urls_env_var``, else
        ``OPENREWARD_SESSION_URLS_<LABEL>``."""

        return self.session_urls_env_var or pool_env_var(self.label or "")

    async def check_server_environment(self, url: str) -> None:
        """Refuse an env name that the env server at ``url`` would answer with another environment.

        An env server answers every request that names no ``variant`` with its default (first) environment: the
        name in ``ors+http://.../<EnvName>`` travels only as the deployment and selects nothing.  When the server
        lists several environments (``GET /list_environments``) and the env name is one of them but not the
        default, every session would run the default one under this name, so this raises
        :class:`EnvSetupError` (``env_config``) saying to select it with ``?variant=``.  A name the server does not
        list (a deployment's own name) passes, and so does a server that lists nothing or cannot be asked (a
        hosted deployment): best effort, one listing per server and process.
        """

        if self.variant is not None or not self.env_name:
            return
        names = await _server_environments(url)
        name = self.env_name.lower()
        if len(names) > 1 and name in names and names[0] != name:
            raise EnvSetupError(
                "env_config",
                f"The env server {url} serves [{', '.join(names)}] and answers requests without a variant with "
                f"{names[0]!r}, its default, so ORS env {self.env_name!r} would run {names[0]!r} instead; select it "
                f"with ?variant={self.env_name} on the service URI (or variant in service_kwargs)",
            )

    async def list_splits(self) -> list[str]:
        if self.is_remote:
            async with self._remote_environment() as environment:
                return list(await environment.list_splits())
        return sdk.split_names(self.environment_class.list_splits())

    async def list_tasks(self, split: str) -> list[str]:
        """Task ids of ``split``, narrowed by ``task_names`` or ``task_indices``."""

        if self.task_names is not None:
            allowed = set(self.task_names)
            return [TaskRef(split, index).task_id for index, name in await self._names(split) if name in allowed]
        count = await self._count(split)
        indices = self.task_indices if self.task_indices is not None else range(count)
        if invalid := [index for index in indices if index >= count]:
            raise ValueError(f"Task indices {invalid[:5]} are outside {self.name} split {split!r} (0..{count - 1})")
        return [TaskRef(split, index).task_id for index in indices]

    async def list_task_names(self, split: str) -> dict[str, str]:
        """The name of every task of ``split`` by task id, whatever ``task_names`` or ``task_indices`` narrow
        ``list_tasks`` to; lists the split's catalog.  ``exclude_tasks`` matches task names through it."""

        return {TaskRef(split, index).task_id: name for index, name in await self._names(split)}

    def warm_task_lists(self, splits: Sequence[str]) -> None:
        """List each of ``splits`` of the in-process environment class once, into the class's task cache, which the
        task lookups of this process's sessions, and of the processes it forks, then read instead of listing them.

        A rollout worker's fork server calls it before it forks the workers
        (:mod:`platoon.train.startup`).  It runs ``list_tasks`` in the calling thread (awaiting it in an
        event loop of its own when it returns an awaitable, which is closed again), and stores what it returns where
        the SDK's ``get_task`` and ``num_tasks`` look first (the class's ``_tasks_cache``, which
        :func:`_own_task_cache` gives each class of its own); a cache that already holds a split keeps it.  A split
        the class does not list is skipped.  A remote environment's tasks are the env servers' to list: a no-op."""

        if self.is_remote:
            return
        cls = self.environment_class
        known = set(sdk.split_names(cls.list_splits()))
        cache = cls.__dict__.get("_tasks_cache")
        for split in splits:
            if split not in known or (isinstance(cache, dict) and split in cache):
                continue
            tasks = cls.list_tasks(split)
            if inspect.isawaitable(tasks):
                tasks = asyncio.run(_awaited(tasks))
            if isinstance(cache, dict):
                cache[split] = tasks

    async def get_task(self, task_id: str) -> Task:
        """The task for a ``<split>:<index>`` id, a task name (in the default split) or an OpenReward task
        reference, which must name this service's ``label`` when it has one."""

        ref = TaskRef.parse(task_id, label=self.label)
        if ref is None:
            matches = [index for index, name in await self._names(self.split) if name == task_id]
            if not matches:
                raise ValueError(f"No task named {task_id!r} in {self.name} split {self.split!r}")
            ref = TaskRef(self.split, matches[0], task_id)
        elif not self.is_remote and ref.name is None:
            try:
                spec = await self._local_spec(ref)
            except IndexError:
                raise ValueError(f"{self.name} split {ref.split!r} has no task {ref.index}") from None
            ref = dataclasses.replace(ref, name=sdk.task_name(spec, ref.index))
        return Task(id=task_id, misc={ORS_MISC_KEY: ref.to_misc()})

    async def open(self, task: Task, *, rollout_id: str) -> OrsSession:
        """Open an env session for ``task``; its instruction is the task's goal, else the env's prompt."""

        ref = TaskRef.from_misc(task.misc.get(ORS_MISC_KEY))
        if ref is None:
            if task.id is None:
                raise EnvSetupError("env_config", "An ORS task needs an id or misc['ors'] with its split and index")
            try:
                ref = TaskRef.from_misc((await self.get_task(task.id)).misc[ORS_MISC_KEY])
            except (KeyError, ValueError) as exc:
                raise EnvSetupError("env_config", str(exc)) from exc
            assert ref is not None
        return await self._open(task, ref, rollout_id=rollout_id, url=None)

    async def open_fork(
        self, task: Task, *, ref: TaskRef, rollout_id: str, url: str | None, log: EventLog | None = None
    ) -> OrsSession:
        """A ``fresh`` fork: a new env session for the same task (on the same server), with ``task``'s goal.

        ``log`` is the parent's event log, which the fork records into under its own session id.
        """

        return await self._open(task, ref, rollout_id=rollout_id, url=url, log=log)

    def tool_specs(self, raw: Sequence[sdk.RawTool], lease: Lease) -> list[ToolSpec]:
        """A session's listed tools with platoon's hints (:meth:`tool_spec`).

        The learned fallback (:attr:`learned_terminal`) applies only to an env that declares no terminal tool: no
        SDK ``@terminal`` or exclusive tool, no ``platoon_terminal_tools`` and no ``terminal_tools`` option.  An env
        that declares them keeps its declaration, so a tool that ends an episode without being declared terminal
        stays listed for ``no_terminal`` agents.
        """

        declared = self.declares_terminal(raw, lease)
        if not declared and lease.backend.url is not None:
            self._warn_no_remote_terminal_tool()
        learned = frozenset() if declared else frozenset(self.learned_terminal)
        return [self.tool_spec(tool, lease, learned=learned) for tool in raw]

    def declares_terminal(self, raw: Sequence[sdk.RawTool], lease: Lease) -> bool:
        """Whether the env declares its terminal tools: an SDK ``@terminal`` or exclusive tool among ``raw`` (its
        listing), ``platoon_terminal_tools`` or the ``terminal_tools`` option."""

        return (
            bool(self.terminal_tools)
            or bool(_names(self.hint(lease, "platoon_terminal_tools")))
            or any(tool.sdk_terminal for tool in raw)
        )

    async def call_class(self, name: str, lease: Lease) -> CallClass:
        """How a call to ``name`` takes its turn on the live env (:mod:`platoon.envs.gate`).

        * ``exclusive`` for a declared terminal tool: SDK ``@terminal`` or ``@tool(exclusive=True)``, the
          ``terminal_tools`` option or ``platoon_terminal_tools``.  It waits for the calls in flight and runs
          alone.  A tool only learned to end episodes (:attr:`learned_terminal`) is not declared.
        * ``concurrent`` for a tool declared safe to overlap: ``@tool(concurrent=True)`` (0.1.163; the flag is not
          on the wire, so remotely only through ``env_class``), ``platoon_concurrent_tools`` or the
          ``concurrent_tools`` option.
        * ``serial`` for everything else, read-only tools included (``read_only`` decides what an agent may
          see, not what may overlap): one at a time per live env.
        """

        if name in self.terminal_tools or name in _names(self.hint(lease, "platoon_terminal_tools")):
            return "exclusive"
        try:
            tool = next((tool for tool in await lease.raw_tools() if tool.name == name), None)
        except (EnvBusy, GateClosed):
            raise
        except Exception:  # the call itself will report what is wrong with the env; meanwhile assume the safe class
            logger.warning(
                "Cannot list the tools of %s to classify %r; it runs as a serial call", self.name, name, exc_info=True
            )
            return "serial"
        if tool is not None and tool.sdk_terminal:
            return "exclusive"
        if (
            (tool is not None and tool.sdk_concurrent)
            or name in self.concurrent_tools
            or name in _names(self.hint(lease, "platoon_concurrent_tools"))
        ):
            return "concurrent"
        return "serial"

    def _warn_no_remote_terminal_tool(self) -> None:
        """Say once per process that a remote env names no terminal tool, which agents without one then see."""

        key = (self.env_name, self.variant, tuple(self.urls))
        if key in _WARNED_NO_TERMINAL_TOOL:
            return
        _WARNED_NO_TERMINAL_TOOL.add(key)
        logger.warning(
            "ORS env %s (variant %s, servers %s) declares no terminal tool: env servers send only @terminal, so an "
            "exclusive submit tool and the class's platoon_terminal_tools are invisible over ors+http, and agents "
            "with no_terminal access (children, verifiers) can submit. Name the submit tools in "
            "service_kwargs.terminal_tools, or the class the env server serves in service_kwargs.env_class.",
            self.env_name,
            self.variant,
            ", ".join(self.urls) or "from the environment",
        )

    def tool_spec(self, tool: sdk.RawTool, lease: Lease, *, learned: frozenset[str] = frozenset()) -> ToolSpec:
        """A listed tool with platoon's hints: terminal and read-only flags, and ``_meta``.  ``learned`` names
        tools to mark terminal because they ended an episode (see :meth:`tool_specs`)."""

        schema = sdk.normalize_schema(tool.input_schema)
        extra = {
            **_mapping(self.hint(lease, "platoon_tool_meta")).get(tool.name, {}),
            **self.tool_meta.get(tool.name, {}),
        }
        if tool.routes_message:
            extra[sdk.TERMINAL_TOOL_META_KEY] = {"arg": tool.terminal_arg}
        meta = sdk.tool_meta(
            tool.meta, schema, extra=extra, routing_override=self.tool_routing_overrides.get(tool.name)
        )
        terminal = (
            tool.sdk_terminal
            or tool.name in self.terminal_tools
            or tool.name in _names(self.hint(lease, "platoon_terminal_tools"))
            or tool.name in learned
        )
        read_only = not terminal and (
            tool.name in self.read_only_tools
            or tool.name in _names(self.hint(lease, "platoon_read_only_tools"))
            or sdk.meta_says_read_only(meta)
        )
        return ToolSpec(tool.name, tool.description, schema, terminal=terminal, read_only=read_only, meta=meta)

    async def _open(
        self, task: Task, ref: TaskRef, *, rollout_id: str, url: str | None, log: EventLog | None = None
    ) -> OrsSession:
        lease, blocks, ref = await self._start(ref, rollout_id=rollout_id, url=url, log=log)
        prompt = sdk.prompt_text(blocks)
        if self.prompt_filter is not None:
            prompt = self.prompt_filter(prompt)
        goal = task.goal or prompt
        task = dataclasses.replace(task, goal=goal, misc={**task.misc, ORS_MISC_KEY: ref.to_misc()})
        return OrsSession(
            self,
            task,
            lease,
            ref=ref,
            instruction=goal,
            prompt=prompt,
            prompt_blocks=sdk.content_blocks(blocks),
            rollout_id=rollout_id,
        )

    async def _start(
        self, ref: TaskRef, *, rollout_id: str, url: str | None, log: EventLog | None = None
    ) -> tuple[Lease, list[Any], TaskRef]:
        """Start the env session: the lease, the prompt's blocks, and ``ref`` with the task's display name."""

        spec: Mapping[str, Any] | None = None
        try:
            if self.is_remote:
                url = url or select_session_url(self.session_urls(), rollout_id)
                await self.check_server_environment(url)
                await self.check_served_class(url)
                backend = await RemoteBackend.start(
                    url=url,
                    env_name=self.env_name or "",
                    variant=self.variant,
                    api_key=self.api_key,
                    api_url=self.api_url,
                    split=ref.split,
                    index=ref.index,
                    secrets=self.secrets,
                    timeout_s=self.close_timeout_s,
                    described_by=self.served_class,
                )
            else:
                spec = await self._local_spec(ref)
                backend = await LocalBackend.start(self.environment_class, spec, self.secrets, self.close_timeout_s)
        except EnvSetupError:
            raise
        except (IndexError, KeyError, ValueError) as exc:
            raise EnvSetupError("env_config", f"{self.name} task {ref.task_id}: {type(exc).__name__}: {exc}") from exc
        except Exception as exc:
            kind = "env_connect" if self.is_remote else "env_setup"
            raise EnvSetupError(kind, f"{self.name} task {ref.task_id}: {type(exc).__name__}: {exc}") from exc
        try:
            blocks = await backend.prompt()
            if ref.name is None:
                # The display name from the task spec (sdk.task_name); a remote task's needs one more request.
                name = (
                    sdk.task_name(spec, ref.index)
                    if spec is not None
                    else await backend.task_name(ref.split, ref.index, self.close_timeout_s)
                )
                ref = dataclasses.replace(ref, name=name)
        except BaseException as exc:
            await bounded(backend.close(), self.close_timeout_s, f"close {self.name} after a failed prompt")
            if not isinstance(exc, Exception):
                raise
            raise EnvSetupError("env_setup", f"{self.name} get_prompt failed: {type(exc).__name__}: {exc}") from exc
        if log is None and self.log_dir is not None:
            log = EventLog(self.log_dir / _slug(rollout_id), env=self.reported_name, ref=ref)
        lease = Lease(backend, log=log, gate=ToolGate(queue_timeout_s=self.queue_timeout_s))
        lease.record(
            "session_started",
            {
                "env": self.reported_name,
                "variant": self.variant,
                "split": ref.split,
                "task_index": ref.index,
                "task_name": ref.name,
                "session_url": backend.url,
            },
        )
        return lease, blocks, ref

    async def _local_spec(self, ref: TaskRef) -> dict[str, Any]:
        cls = self.environment_class
        splits = sdk.split_names(cls.list_splits())
        if ref.split not in splits:
            raise ValueError(f"{cls.__name__} has no split {ref.split!r}; its splits are {splits}")
        return dict(await sdk.run_user_callable(cls.get_task, ref.split, ref.index))

    async def _count(self, split: str) -> int:
        if self.is_remote:
            async with self._remote_environment() as environment:
                return int(await environment.num_tasks(split))
        cls = self.environment_class
        if split not in (splits := sdk.split_names(cls.list_splits())):
            raise ValueError(f"{cls.__name__} has no split {split!r}; its splits are {splits}")
        return int(await sdk.run_user_callable(cls.num_tasks, split))

    async def _names(self, split: str) -> list[tuple[int, str]]:
        """``(index, name)`` of every task in ``split`` (lists the whole catalog)."""

        if self.is_remote:
            async with self._remote_environment() as environment:
                specs = [task.task_spec for task in await environment.list_tasks(split)]
        else:
            cls = self.environment_class
            if split not in (splits := sdk.split_names(cls.list_splits())):
                raise ValueError(f"{cls.__name__} has no split {split!r}; its splits are {splits}")
            specs = list(await sdk.run_user_callable(cls.list_tasks, split))
        return [(index, sdk.task_name(spec, index)) for index, spec in enumerate(specs)]

    @contextlib.asynccontextmanager
    async def _remote_environment(self) -> AsyncIterator[Any]:
        client = sdk.import_client()
        url = self.session_urls()[0]
        await self.check_server_environment(url)
        await self.check_served_class(url)
        api = client.AsyncEnvironmentsAPI(base_url=url, api_key=self.api_key, api_base_url=self.api_url or url)
        try:
            yield api.get(self.env_name or "", variant=self.variant)
        finally:
            await bounded(api.aclose(), self.close_timeout_s, f"close the {self.name} catalog client")

    def _class_name(self) -> str:
        environment = self._environment
        if isinstance(environment, str):
            return environment.replace(":", ".").rpartition(".")[2]
        return getattr(environment, "__name__", "ors")


async def _server_environments(url: str) -> tuple[str, ...]:
    """The environments the env server at ``url`` lists, lowercased, its default first (``GET /list_environments``,
    which both supported SDK servers serve); empty when it lists none or cannot be asked.  Asked once per process:
    any answer is kept, so a server that never answers costs one bounded wait, and the sessions that open while it
    is asked (a training step opens one per rollout, all at once) wait for that one request."""

    listed = _SERVER_ENVIRONMENTS.get(url)
    if listed is not None:
        return listed
    loop = asyncio.get_running_loop()
    pending = _LISTINGS_IN_FLIGHT.get(url)
    if pending is None or pending.done() or pending.get_loop() is not loop:
        pending = _LISTINGS_IN_FLIGHT[url] = loop.create_task(_list_server_environments(url))
        pending.add_done_callback(lambda task: _listing_done(url, task))
    return await asyncio.shield(pending)  # a session open that is cancelled leaves the request to the others


def _listing_done(url: str, task: asyncio.Task[tuple[str, ...]]) -> None:
    _check_done(_LISTINGS_IN_FLIGHT, url, task)


def _check_done(in_flight: dict[Any, asyncio.Task[Any]], key: Any, task: asyncio.Task[Any]) -> None:
    if in_flight.get(key) is task:
        del in_flight[key]
    if not task.cancelled():
        task.exception()  # retrieved, so an error no session waited for is not logged as never retrieved


def _warn_served_hooks(cls: type, env_name: str) -> None:
    """Say once per class that the hooks it defines run only in process, so its env server's sessions skip them."""

    hooks = [name for name in sdk.HOOK_ATTRIBUTES if callable(getattr(cls, name, None))]
    if not hooks or cls in _WARNED_SERVED_HOOKS:
        return
    _WARNED_SERVED_HOOKS.add(cls)
    logger.warning(
        "ORS env %s serves %s, whose hooks (%s) run only in process: its env server's sessions run none of them "
        "(without platoon_grade an episode that never submits scores 0, no_submit; without platoon_fork a shared "
        "child is graded by the episode's first submission; without platoon_copy a copy fork is refused)",
        env_name,
        cls.__qualname__,
        ", ".join(hooks),
    )


async def _list_server_environments(url: str) -> tuple[str, ...]:
    import aiohttp

    body: Any = None
    try:
        timeout = aiohttp.ClientTimeout(total=_LIST_ENVIRONMENTS_TIMEOUT_S)
        # trust_env as the openreward client has it, so the listing takes the route the sessions take.
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as http:
            async with http.get(url.rstrip("/") + "/list_environments") as response:
                if response.status == 200:
                    body = await response.json(content_type=None)
    except (aiohttp.ClientError, OSError, TimeoutError, ValueError):
        logger.debug("Cannot list the environments of the env server %s", url, exc_info=True)
    listed = tuple(str(name).lower() for name in body) if isinstance(body, list) else ()
    _SERVER_ENVIRONMENTS[url] = listed
    return listed


async def _awaited(value: Any) -> Any:
    return await value


def _own_task_cache(cls: Any) -> Any:
    """``cls``, with a task cache of its own.

    The SDK caches each class's task list per split on the class (``Environment._list_tasks_cached``, behind the
    default ``num_tasks`` and ``get_task``), but finds the cache with ``hasattr``, which follows the MRO: once a
    parent class has listed a split, a subclass with its own ``list_tasks`` reads the parent's tasks for it.
    """

    if isinstance(cls, type):
        for name in ("_tasks_cache", "_tasks_cache_locks"):
            if name not in cls.__dict__:
                setattr(cls, name, {})
    return cls


def _names(value: Any) -> frozenset[str]:
    if isinstance(value, str):
        return frozenset({value})
    return frozenset(value or ())


def _mapping(value: Any) -> Mapping[str, Mapping[str, Any]]:
    return value if isinstance(value, Mapping) else {}


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")[:120] or "rollout"
