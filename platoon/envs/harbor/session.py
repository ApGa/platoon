"""Harbor sessions: one task sandbox per rollout, graded by the task's ``tests/test.sh``.

Agents that share a sandbox take turns on it (:mod:`platoon.envs.gate`): their tool calls run one at a time
unless the service lists the tool in ``concurrent_tools``, and grading and copy snapshots run alone.  A call not
admitted within ``queue_timeout_s`` is not run; an admitted call runs to the end even when its caller gives up,
and the last close stops the sandbox only after the calls in flight end (at most ``close_timeout_s``).

A task with a separate verifier is graded in a sandbox of its own: grading kills every process the agents left in
their sandbox (``main``), collects the task's artifacts from it as directories and regular files only
(:mod:`platoon.envs.harbor.artifacts`), starts the verifier's sandbox, hands it the artifacts and runs the tests
there, and stops it on every path.  The agents never touch the grader's files, and no symlink of theirs reaches
it.  When nothing else holds ``main`` (no shared fork, no call in flight), grading releases it as soon as the
artifacts are collected, as Harbor stops it before verifying: the session then takes no more tool calls or forks,
and grading it again reuses the collected artifacts.

**Users.**  In a sandbox that honors the users a task names (``agent_user: honor``, ``verifier_user: honor``), checked
when it starts to give root and other users (:mod:`platoon.envs.harbor.users`), the session's ``sandbox`` is the
agents' view of it, which the harness, the env tools and the oracle use, while grading, the kills and platoon's own
commands keep root's, and the tests run as theirs (:mod:`platoon.envs.harbor.views`): everything there runs with
Docker's umask, and a separate verifier's artifacts are readable by every user, since the tests may run the agent's
code as another user, as they would in Docker.  Tests that run as a user other than root get ``/tests`` as theirs and
``/logs/verifier`` open to every user (Harbor's 777); tests that run as root get both as root's, ``/tests`` with the
host's modes, as Harbor copies it, so that the agent's code they run as another user can neither read a test file the
host keeps from it nor write a reward.  A copy fork of such a sandbox is refused where its provider loses the files'
owners, which a snapshot would not keep.

Grading takes the sandbox the tests run in away from the agents.  In the agents' own sandbox (Harbor's ``shared``
verifier mode) it first kills every process they left running (``Sandbox.kill_processes``) and makes ``/tests`` and
``/logs/verifier`` afresh.  A main-only compose file's init (``main``'s command) and what it started are not the
agents': the kill spares them (``Sandbox.keep_processes``), and an init that is gone by then, which in Docker would
have ended the container, makes the outcome an ``error`` (``main_exited``) without the tests.  A copy fork of such a
task is refused, since a restore brings back its files, not its init.  In either sandbox grading kills again what
the tests left running (the tests may run the agent's code) before it reads their output, file by file and never
through a symlink (:mod:`platoon.envs.harbor.verifier`).

A task that is a multi-container stack (:mod:`platoon.envs.harbor.sidecars`) holds its whole stack in a
:class:`StackLease`: ``main`` is the agents' sandbox, shared forks share the stack, and a ``copy`` fork is refused,
since a restore would bring back ``main``'s files, not the services.  Grading runs the task's ``[[verifier.collect]]``
hooks in their services (``main``'s first), and probes the sidecars before and after the tests, or before and after
the artifacts are collected for a separate verifier, which also receives the artifacts of the sidecars that the task
names (``artifacts[].service``): a sidecar that is down makes the outcome ``invalid`` (``sidecar_unhealthy``), unless
the service grades anyway (``on_sidecar_down: grade``).  What a sidecar writes in its own log directories reaches the
sandbox the tests run in once grading made its log directories afresh.  While the tests run in ``main``, the stack has
the verifier's network.  A stack that reaches node services holds a scope of each
(:mod:`platoon.envs.harbor.node_services`), which the same probes check: one that a start of its node service that
failed may have answered wrongly, or whose request the node service could not answer, makes the outcome ``invalid``
(``node_service_down``), whatever ``on_sidecar_down`` says.

**Task MCP servers.**  The MCP servers a stack's task offers its agents (``[[environment.mcp_servers]]``) reach the
agents two ways.  The tool plane serves each ``sse`` or ``streamable-http`` server whose ``http://`` URL names a
service of the stack (:class:`TaskTools`): the session lists its tools as it opens, among its env tools
(:func:`task_tool_name`: ``mcp_server__get_secret``), read-only where the tool says so (MCP's ``readOnlyHint``), and
proxies each call to the server through the stack's connector (:meth:`~platoon.sandbox.stacks.Stack.connect`), in its
turn on the gate as every tool call, so that function harnesses and harnesses that take the env's tools over MCP get
them, filtered by their access level and recorded with their calls.  A server that does not list its tools fails the
session's start (``sidecar_start``).  A harness in ``main`` may instead reach the servers itself, as Harbor's agents
do, from the list that :attr:`HarborSession.task_mcp_servers` gives (``{task_mcp_servers}`` in a harness spec): every
server, ``stdio`` ones too, which run in the harness's own sandbox and which nothing proxies, since that would take a
streaming exec that no provider has.

The agents' workdir is where the tests run (``shared``) and where the artifacts are (``separate``), so an attempt
whose agents removed or moved it, left something else in its place or locked it scores 0 (``workdir_missing``) in
either mode, without the tests: no command starts in a working directory that cannot be entered (docker and
hpc-sandbox exit 126, the ``process`` provider raises).  Should the tests' command in the agents' sandbox exit 126,
grading checks the workdir again, with a command run in ``/``, before it reads any reward file: a test may also exit
126 by itself, and with the workdir in place its reward counts.  Where the sandbox says that the tests' command never
entered its working directory (hpc-sandbox's line, :func:`platoon.sandbox.base.cwd_error`), the outcome records it
(``verifier_cwd_error``), in a separate verifier's sandbox too.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import re
import shutil
import time
import urllib.parse
import uuid
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp

from platoon.envs.gate import DEFAULT_QUEUE_TIMEOUT_S, EnvBusy, GateClosed, ToolGate, caller_or
from platoon.envs.harbor import sidecars
from platoon.envs.harbor.artifacts import ArtifactError, CollectedArtifact, collect_artifacts, materialize_artifacts
from platoon.envs.harbor.mcp_client import McpClient, McpError
from platoon.envs.harbor.node_services import NodeScope
from platoon.envs.harbor.sidecars import SidecarLogs
from platoon.envs.harbor.task import CollectHook, HarborTask, McpServer, resolve_env
from platoon.envs.harbor.tools import DEFAULT_MAX_OUTPUT_CHARS, TOOL_SPECS, SandboxTools, truncate_output
from platoon.envs.harbor.verifier import TEST_STDOUT, collect, output_tail, suspect_outcome, write_local
from platoon.envs.harbor.views import SessionUsers, unviewed
from platoon.envs.service import EnvSetupError, ForkMode, Outcome, ToolResult, ToolSpec
from platoon.sandbox import loses_file_owners
from platoon.sandbox._processes import alive_processes
from platoon.sandbox.base import CWD_ERROR_EXIT_CODE, ExecResult, Sandbox, SandboxSpec, cwd_error
from platoon.sandbox.stacks import MAIN, ServiceState, Stack
from platoon.task import Task
from platoon.utils import scratch
from platoon.utils.stopping import let_finish

if TYPE_CHECKING:
    from platoon.envs.harbor.service import HarborEnvironmentService

logger = logging.getLogger(__name__)

TESTS_DIR = "/tests"
SOLUTION_DIR = "/solution"
VERIFIER_DIR = "/logs/verifier"
LOG_DIRS = ("/logs/agent", VERIFIER_DIR, "/logs/artifacts")
# Where the agents publish what they leave, which every user may write where users are honored (Harbor makes them 777).
PUBLISHED_DIRS = ("/logs/agent", "/logs/artifacts")
DEFAULT_CLOSE_TIMEOUT_S = 30.0
# Makes /tests and /logs/verifier afresh, whatever the agents left there or in their place (a symlink to elsewhere
# included: rm removes a symlink, not what it points to), after removing a /logs that is not a directory.  A
# directory an agent took the permissions from (chmod 000), which rm could not empty, gets them back first; chmod
# never follows a symlink there (the top directories are checked, and chmod -R skips the links it finds).
_RESET_SCRIPT = (
    "if [ -L /logs ] || { [ -e /logs ] && [ ! -d /logs ]; }; then rm -f /logs; fi; "
    "[ -d /logs ] && chmod u+rwx /logs 2>/dev/null; "
    f"for dir in {TESTS_DIR} {VERIFIER_DIR}; do "
    '[ -d "$dir" ] && [ ! -L "$dir" ] && chmod -R u+rwX "$dir" 2>/dev/null; done; '
    f"rm -rf {TESTS_DIR} {VERIFIER_DIR} && mkdir -p {TESTS_DIR} {VERIFIER_DIR}"
)
# Prints "ok" when the workdir ("$1") is a directory a command can start in (as hpc-sandbox, docker and the process
# provider enter it: with cd, or chdir), else what is there instead: "missing" (removed or moved, a symlink to
# nothing included), "not_a_directory", or "not_enterable" (its permissions were taken).
_WORKDIR_SCRIPT = (
    'if cd -- "$1" >/dev/null 2>&1; then echo ok; elif [ ! -e "$1" ]; then echo missing; '
    'elif [ ! -d "$1" ]; then echo not_a_directory; else echo not_enterable; fi'
)
_WORKDIR_STATES = frozenset({"ok", "missing", "not_a_directory", "not_enterable"})
_WORKDIR_TIMEOUT_S = 60.0
_MAX_SKIPPED_REPORTED = 20  # of the artifact paths left out, the ones an outcome names

#: What joins a task MCP server's name and its tool's in the env tool that serves it (:func:`task_tool_name`).
TASK_TOOL_SEPARATOR = "__"
#: The longest name of an env tool that serves a task MCP server's tool: the longest tool name model providers take.
MAX_TASK_TOOL_NAME_CHARS = 64
#: How long the listing of one task MCP server may take as a session opens: its connector, its session, its tools.
TASK_TOOLS_LISTING_TIMEOUT_S = 60.0
_NOT_IN_TOOL_NAMES = re.compile(r"[^A-Za-z0-9_]")

_BACKGROUND: set[asyncio.Task[Any]] = set()  # teardowns whose callers stopped waiting
_UNKILLABLE: set[type] = set()  # sandbox types without kill_processes, warned about once


class SandboxLease:
    """A sandbox shared by a session and its shared forks; the last release stops it.

    ``gate`` orders the operations on the sandbox (:mod:`platoon.envs.gate`); ``inflight`` holds the tool calls
    still running, which the last release waits for (at most ``close_timeout_s``) before it stops the sandbox.
    ``init`` is the main-only compose file's init running in it ("<pid>:<start time>"), ``scratch`` the host
    directories its binds mount (copies of the task's files), deleted once it stops, and ``users`` who runs what in
    it (``sandbox`` is their ``root``).
    """

    def __init__(
        self,
        sandbox: Sandbox,
        spec: SandboxSpec,
        *,
        queue_timeout_s: float | None = DEFAULT_QUEUE_TIMEOUT_S,
        close_timeout_s: float = DEFAULT_CLOSE_TIMEOUT_S,
        init: str | None = None,
        scratch: list[Path] | None = None,
        users: SessionUsers | None = None,
    ) -> None:
        self.sandbox = sandbox
        self.spec = spec
        self.init = init
        self.users = users if users is not None else SessionUsers.alone(sandbox)
        self.scratch = list(scratch or ())
        self.gate = ToolGate(queue_timeout_s=queue_timeout_s)
        self.close_timeout_s = close_timeout_s
        self.inflight: set[asyncio.Task[Any]] = set()
        self.closed = False  # the last holder released it: nothing new starts
        self._holders = 1
        self._closing: asyncio.Task[Any] | None = None

    @property
    def holders(self) -> int:
        """The sessions holding the sandbox: a session and its shared forks."""

        return self._holders

    def acquire(self) -> SandboxLease:
        self._holders += 1
        return self

    async def release(self) -> None:
        """Drop a holder.  The last one closes the lease at once and stops the sandbox in a background task the
        caller waits for, so a caller that stops waiting (a close under a timeout) does not leak the sandbox."""

        if self._holders > 0:
            self._holders -= 1
        if self._holders > 0:
            return
        if self._closing is None:
            self.closed = True
            self._closing = _in_background(self._teardown())
        await asyncio.shield(self._closing)

    async def _teardown(self) -> None:
        if self.inflight:
            _, running = await asyncio.wait(set(self.inflight), timeout=self.close_timeout_s)
            if running:
                logger.warning(
                    "Stopping sandbox %s with %d call(s) still running after %.0fs",
                    self.sandbox.id,
                    len(running),
                    self.close_timeout_s,
                )
        self.gate.close()
        try:
            await self._stop()
        finally:
            for directory in self.scratch:
                shutil.rmtree(directory, ignore_errors=True)

    async def _stop(self) -> None:
        await self.sandbox.stop()


class StackLease(SandboxLease):
    """A multi-container stack shared by a session and its shared forks: ``sandbox`` is its ``main``, and the last
    release stops the whole stack.  ``logs`` are the session's log directories that its sidecars bind, each a volume
    of the stack; ``task_tools`` the tools of the task's MCP servers in the stack, which the plane serves; ``scopes``
    the stack's scopes of node services, by the service each serves, which close once the stack stopped."""

    def __init__(
        self,
        stack: Stack,
        spec: SandboxSpec,
        *,
        logs: list[SidecarLogs] | None = None,
        task_tools: TaskTools | None = None,
        scopes: Mapping[str, NodeScope] | None = None,
        **options: Any,
    ) -> None:
        super().__init__(stack.main, spec, **options)
        self.stack = stack
        self.logs = list(logs or ())
        self.task_tools = task_tools
        self.scopes = dict(scopes or {})

    async def _stop(self) -> None:
        try:
            if self.task_tools is not None:
                await self.task_tools.close()
        finally:
            try:
                await self.stack.stop()
            finally:
                await asyncio.gather(*(scope.close() for scope in self.scopes.values()), return_exceptions=True)


class TaskTools:
    """The tools of a task's MCP servers in its stack, as the tool plane serves them: those of each ``sse`` or
    ``streamable-http`` server whose ``http://`` URL names a service of the stack, listed as the session opens
    (:meth:`open`) through one session with each server (:class:`~platoon.envs.harbor.mcp_client.McpClient`), which
    the stack's connector reaches and which closes with the stack.  ``specs`` are the env tools that serve them, each
    named by :func:`task_tool_name`, read-only where the tool's ``readOnlyHint`` says so; none ends the episode, since
    MCP has no hint for that."""

    def __init__(self) -> None:
        self.specs: list[ToolSpec] = []
        self._tools: dict[str, tuple[McpServer, McpClient, str]] = {}  # each env tool's server, session, name there
        self._clients: list[McpClient] = []

    @classmethod
    async def open(
        cls, stack: Stack, harbor_task: HarborTask, *, timeout_s: float = TASK_TOOLS_LISTING_TIMEOUT_S
    ) -> TaskTools | None:
        """The tools of the task's servers that the plane serves in ``stack``; None when it serves none.  A server that
        cannot be reached, or does not list its tools within ``timeout_s``, fails it, and so does a tool whose env
        tool's name would be too long or another's: a ``RuntimeError`` that names them, once the sessions it opened
        are closed."""

        served = [
            (server, address)
            for server in harbor_task.mcp_servers
            if (address := harbor_task.mcp_service(server)) is not None
        ]
        if not served:
            return None
        tools = cls()
        try:
            for server, (service, port) in served:
                await tools._add(stack, server, service, port, timeout_s=timeout_s)
        except BaseException:
            await tools.close()
            raise
        return tools

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    async def call(self, name: str, arguments: Mapping[str, Any], *, timeout_s: float) -> ToolResult:
        """Call the env tool ``name``, one of :attr:`specs`, on its server, within ``timeout_s``: the server's result,
        or an error result that says why there is none (the server's error, no answer)."""

        server, client, tool = self._tools[name]
        try:
            result = await client.call_tool(tool, arguments, timeout_s=timeout_s)
        except TimeoutError:
            message = f"The task's MCP server {server.name} did not answer the call of {tool} within {timeout_s:g} s"
            return ToolResult([{"type": "text", "text": message}], is_error=True, metadata={"error_reason": "timeout"})
        except McpError as exc:
            if exc.code is None:
                logger.warning("The call of %s on the task's MCP server %s failed: %s", tool, server.name, exc)
            return ToolResult.text(f"The task's MCP server {server.name}: {exc}", is_error=True)
        except (aiohttp.ClientError, OSError) as exc:
            logger.warning("The task's MCP server %s cannot be reached: %s", server.name, exc)
            return ToolResult.text(f"The task's MCP server {server.name} cannot be reached: {exc}", is_error=True)
        return _task_tool_result(result)

    async def close(self) -> None:
        """End every session with the servers; idempotent."""

        clients, self._clients = self._clients, []
        await asyncio.gather(*(client.close() for client in clients))

    async def _add(self, stack: Stack, server: McpServer, service: str, port: int, *, timeout_s: float) -> None:
        """Serve the tools of ``server``, which ``port`` of ``service`` serves."""

        assert server.url is not None and server.transport != "stdio"
        try:
            async with asyncio.timeout(timeout_s):
                endpoint = await stack.connect(service, port)
                client = await McpClient.connect(server.url, transport=server.transport, endpoint=endpoint)
                self._clients.append(client)
                listed = await client.list_tools()
        except Exception as exc:  # unreachable, a broken protocol, a port the service does not declare, too slow
            failure = f" within {timeout_s:g} s" if isinstance(exc, TimeoutError) else f": {type(exc).__name__}: {exc}"
            raise RuntimeError(
                f"The task's MCP server {server.name} (port {port} of the service {service}) did not list its tools"
                f"{failure}"
            ) from exc
        for tool in listed:
            name = task_tool_name(server.name, tool["name"])
            if (served := self._tools.get(name)) is not None:
                why = f"as the tool {served[2]!r} of the server {served[0].name} is"
            elif len(name) > MAX_TASK_TOOL_NAME_CHARS:
                why = f"a name longer than {MAX_TASK_TOOL_NAME_CHARS} characters, which model providers refuse"
            else:
                self._tools[name] = (server, client, tool["name"])
                self.specs.append(_task_tool_spec(name, tool))
                continue
            raise RuntimeError(
                f"The task's MCP server {server.name} lists the tool {tool['name']!r}, which would be served as "
                f"{name}, {why}"
            )


def task_tool_name(server: str, tool: str) -> str:
    """The env tool that serves the tool ``tool`` of a task's MCP server ``server``: their names joined by ``__``, each
    with an underscore in place of every character but an ASCII letter, a digit and an underscore, as model providers'
    tool names and CodeAct's functions need: hello-mcp's ``mcp-server`` and ``get_secret`` make
    ``mcp_server__get_secret``.  A name that is not all ASCII also ends with the start of its SHA-256, which keeps
    names in other scripts apart, and a server's that starts with a digit gets an underscore before it, as a Python
    function's name needs (``2fa`` makes ``_2fa__get_code``)."""

    server = _fit_for_tool_names(server)
    return f"{'_' if server[:1].isdigit() else ''}{server}{TASK_TOOL_SEPARATOR}{_fit_for_tool_names(tool)}"


def _fit_for_tool_names(name: str) -> str:
    fit = _NOT_IN_TOOL_NAMES.sub("_", name)
    return fit if name.isascii() else f"{fit}_{hashlib.sha256(name.encode()).hexdigest()[:8]}"


@dataclass
class _KeptArtifacts:
    """What a separate verifier's grading keeps once it released ``main``, for grading again: the directory of the
    copies, the artifacts, what was left out of them, and the copies of the sidecars' log directories."""

    directory: Path
    collected: list[CollectedArtifact]
    skipped: dict[str, str]
    logs: dict[str, Path]


class HarborSession:
    """A rollout's view of a Harbor task: instruction, task sandbox, generic tools, grading and forks.

    ``sandbox`` is what the agents act on: the task's sandbox, or its view as the agents' user where the sandbox
    honors users (:mod:`platoon.envs.harbor.views`).  ``mcp_servers`` lists the task's own MCP servers for harnesses
    that can attach them, at the URLs that reach them from ``main`` (a stack's rewritten for its addressing).
    ``network_isolated`` says the sandbox has no network: the task's ``network_mode`` is
    ``no-network`` and the provider enforces it (its ``network_none`` capability), so a harness placed in the sandbox
    could not reach the model gateway.  ``host_loopback`` says the sandbox shares this host's network, so a harness
    placed in it reaches the gateway on loopback: it has a network, its provider declares the ``host_loopback``
    capability, and it is not the ``main`` of a stack whose services have a network of their own (a pod's or a compose
    project's, whose ``main`` reaches the gateway through its route).
    """

    # call_tool takes calls from every agent at once and orders them on the sandbox's gate (platoon.envs.gate).
    manages_concurrency = True

    def __init__(
        self,
        service: HarborEnvironmentService,
        task: Task,
        harbor_task: HarborTask,
        lease: SandboxLease,
        *,
        rollout_id: str,
    ) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.task = task
        self.instruction = task.goal or harbor_task.instruction
        self.harbor_task = harbor_task
        self.rollout_id = rollout_id
        # What the agents act on: the sandbox, or its agents' view where it honors users (lease.sandbox is root's).
        self.sandbox: Sandbox | None = lease.users.agents
        # The task's multi-container stack, None for a single sandbox; sandbox is its main.
        self.stack: Stack | None = lease.stack if isinstance(lease, StackLease) else None
        self.mcp_servers: tuple[McpServer, ...] = tuple(
            replace(server, url=_in_main(self.stack, server.url)) if self.stack is not None and server.url else server
            for server in harbor_task.mcp_servers
        )
        self._task_tools = lease.task_tools if isinstance(lease, StackLease) else None
        capabilities = service.provider.capabilities
        self.network_isolated = lease.spec.network == "none" and "network_none" in capabilities
        self.host_loopback = (
            not self.network_isolated
            and "host_loopback" in capabilities
            and getattr(self.stack, "host_loopback", True)  # a stack's services may share a network of their own
        )
        self._service = service
        self._lease = lease
        self._tools = SandboxTools(
            lease.users.agents, workdir=lease.spec.workdir, command_timeout_s=service.tool_timeout_s
        )
        # Separate verifier: the artifacts kept once main was released after collecting them, and that release.
        self._kept_artifacts: _KeptArtifacts | None = None
        self._main_release: asyncio.Task[Any] | None = None
        self._notes: dict[str, Any] = {}  # what the grade in progress records besides its outcome (collect hooks, ...)
        self._states: dict[str, ServiceState] = {}  # the sidecars' states at this grade's last probe, if it ran one

    @property
    def routes(self) -> dict[str, str]:
        """The task sandbox's routes: upstream name -> the URL a process in it calls (``Sandbox.routes``)."""

        return dict(getattr(self._lease.sandbox, "routes", None) or {})

    @property
    def agent_timeout_s(self) -> float | None:
        """The task's wall clock for its agent: ``[agent].timeout_sec`` times the service's
        ``agent_timeout_multiplier`` (None when the task sets none).  Harness rollouts cap the harness with it."""

        if self.harbor_task.agent_timeout_s is None:
            return None
        return self.harbor_task.agent_timeout_s * self._service.agent_timeout_multiplier

    @property
    def task_mcp_servers(self) -> list[dict[str, Any]]:
        """The task's MCP servers as a harness in the task's sandbox reaches them (``{task_mcp_servers}``): each one's
        name and transport, and its URL as :attr:`mcp_servers` has it, rewritten for the stack's addressing, or its
        command and arguments (``stdio``)."""

        return [server.described() for server in self.mcp_servers]

    async def tools(self) -> list[ToolSpec]:
        """The generic tools, then those of the task's MCP servers that the plane serves (:class:`TaskTools`)."""

        if self._task_tools is None or self._lease.closed:
            return list(TOOL_SPECS)
        return [*TOOL_SPECS, *self._task_tools.specs]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Run a tool in its turn on the sandbox: refused unrun when the session is closed or the call is not
        admitted within ``queue_timeout_s``; once admitted it runs to the end, even when its caller gives up."""

        lease = self._lease
        if self.sandbox is None or lease.closed:
            return _closed()
        call_class = "concurrent" if name in self._service.concurrent_tools else "serial"
        try:
            return await lease.gate.run(
                call_class,
                lambda grant: self._run_tool(name, arguments),
                view=caller_or(self),  # the calling agent takes its turn, this view's when the plane names none
                inflight=lease.inflight,
            )
        except EnvBusy as busy:
            logger.warning(
                "Harbor tool %s of session %s not run: env busy (waited %.1fs, the limit is %gs)",
                name,
                self.id,
                busy.waited_s,
                busy.timeout_s,
            )
            message = (
                f"Not run: env busy. The call waited {busy.waited_s:.0f} s for other calls on this environment to "
                f"finish (the limit is {busy.timeout_s:g} s); it was not executed. Try again."
            )
            return ToolResult([{"type": "text", "text": message}], is_error=True, metadata={"error_reason": "env_busy"})
        except GateClosed:
            return _closed()

    async def _run_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if self.sandbox is None or self._lease.closed:  # closed while the call waited for its turn
            return _closed()
        if self._task_tools is not None and name in self._task_tools:
            return await self._task_tools.call(name, arguments, timeout_s=self._service.tool_timeout_s)
        return await self._tools.call(name, arguments)

    async def grade(self) -> Outcome:
        """Run the task's tests against the sandbox's current state, in the sandbox itself or, for a task with a
        separate verifier, in a verifier sandbox given the task's artifacts.

        The tests are uploaded only now (so agents never see them) and removed
        afterwards. Infrastructure failures come back as error/invalid outcomes.
        Grading runs alone on the sandbox: it waits for the calls in flight, and
        no agent's call runs while the tests are in the sandbox (or while the
        artifacts are collected). Nothing the agents started runs alongside the
        tests either, nor anything the tests left running while their output is
        read (see the module docstring); a grade that cannot make sure of that
        is an infra ``error`` (``verifier_isolation``). An attempt that left the
        agents' workdir unusable scores 0 (``workdir_missing``) and the tests do
        not run. A grade that cannot start within ``queue_timeout_s`` is an
        infra ``error`` (``env_busy``); a cancelled grade stops the tests and
        removes them before the next call.
        """

        separate = self.harbor_task.verifier_mode == "separate"
        if self.sandbox is None and not (separate and self._kept_artifacts is not None):
            return Outcome.infra("session_closed")
        self._notes, self._states = {}, {}
        try:
            if separate:
                outcome = await self._grade_separately()
            else:
                self._open_sandbox()
                async with self._lease.gate.hold("exclusive"):
                    outcome = await self._verify(self._lease.users)
        except EnvBusy as busy:
            logger.warning(
                "Grading Harbor session %s waited %.1fs for the calls in flight; gave up", self.id, busy.waited_s
            )
            return Outcome.infra("env_busy", reason=f"grading waited {busy.waited_s:.1f}s for the calls in flight")
        except GateClosed:
            return Outcome.infra("session_closed")
        except Exception as exc:
            logger.exception("Grading Harbor task %s failed", self.harbor_task.name)
            return Outcome.infra(f"unexpected:{type(exc).__name__}", metadata={"error": str(exc)})
        outcome.metadata.update(self._notes)
        outcome.metadata["verifier_mode"] = self.harbor_task.verifier_mode
        if self.harbor_task.compose is not None:
            outcome.metadata["stack"] = self.harbor_task.compose.kind
        if self.stack is not None:
            outcome.metadata["services"] = sidecars.services_metadata(self.stack, self._states)
        if isinstance(self._lease, StackLease) and self._lease.scopes:  # which starts of each node service answered it
            outcome.metadata["node_services"] = {
                name: {"node_service": scope.spec.name, "generation": scope.generation, "served_by": list(scope.served)}
                for name, scope in self._lease.scopes.items()
            }
        if self.harbor_task.ignored_users:
            outcome.metadata["ignored_users"] = dict(self.harbor_task.ignored_users)  # ran as the sandbox's uid
        if self.harbor_task.verifier_root:
            outcome.metadata["verifier_user"] = "root"
        if self.harbor_task.honored_users:
            outcome.metadata["users"] = dict(self.harbor_task.honored_users)  # None: the image's user, else root
        return outcome

    async def run_solution(self) -> ExecResult:
        """Run the task's reference ``solution/solve.sh`` (Harbor's oracle), e.g. to validate a dataset."""

        sandbox = self._open_sandbox()
        await sandbox.upload(self.harbor_task.solution_dir, SOLUTION_DIR)
        return await sandbox.exec(f"bash {SOLUTION_DIR}/solve.sh", timeout_s=self.agent_timeout_s)

    async def fork(self, mode: ForkMode = "shared", *, task: Task | None = None) -> HarborSession:
        """``shared`` shares this sandbox, ``copy`` restores a snapshot of it, ``fresh`` opens a new sandbox.

        A session whose ``main`` its separate verifier released forks no more: fork before grading."""

        self._open_sandbox()
        sandbox = self._lease.sandbox
        task = task or self.task
        if mode == "fresh":
            return await self._service.open(task, rollout_id=self.rollout_id)
        harbor_task = self._service.harbor_task(task)
        if mode == "shared":
            return HarborSession(self._service, task, harbor_task, self._lease.acquire(), rollout_id=self.rollout_id)
        if mode == "copy":
            if self.stack is not None:
                raise NotImplementedError(
                    f"Harbor task {self.harbor_task.name!r} is a multi-container stack, and a copy fork restores "
                    "main's files, not its services: fork it shared or fresh"
                )
            if self._lease.init is not None:
                raise NotImplementedError(
                    f"Harbor task {self.harbor_task.name!r} runs main's command from its compose file, and a copy fork "
                    "restores files, not processes: fork it shared or fresh"
                )
            provider = self._service.provider
            if "snapshot" not in provider.capabilities:
                raise NotImplementedError(f"The {provider.name!r} sandbox provider cannot snapshot for copy forks")
            if self._lease.users.honors and loses_file_owners(provider):
                raise NotImplementedError(
                    f"Harbor task {self.harbor_task.name!r} honors users, and a snapshot of a {provider.name!r} "
                    "sandbox keeps no file owners, which a restore would give to root: fork it shared or fresh"
                )
            # The snapshot runs alone on the sandbox, so it never copies a call half-applied.
            snapshot = await self._lease.gate.run(
                "exclusive", lambda grant: sandbox.snapshot(), inflight=self._lease.inflight
            )
            restored = await provider.restore(snapshot, self._lease.spec)
            try:
                users = await self._service.restored_users(restored, harbor_task, self._lease.spec)
            except BaseException:
                await asyncio.shield(restored.stop())
                raise
            lease = SandboxLease(
                users.root,
                self._lease.spec,
                queue_timeout_s=self._lease.gate.queue_timeout_s,
                close_timeout_s=self._lease.close_timeout_s,
                users=users,
            )
            return HarborSession(self._service, task, harbor_task, lease, rollout_id=self.rollout_id)
        raise ValueError(f"Unknown fork mode {mode!r}")

    async def close(self) -> None:
        kept, self._kept_artifacts = self._kept_artifacts, None
        if kept is not None:
            shutil.rmtree(kept.directory, ignore_errors=True)
        if self._main_release is not None:
            await asyncio.shield(self._main_release)  # main stops in the background once its artifacts are out
            return
        if self.sandbox is None:
            return
        self.sandbox = None
        await self._lease.release()

    def _open_sandbox(self) -> Sandbox:
        if self.sandbox is None:
            if self._main_release is not None:
                raise RuntimeError(
                    f"Harbor session {self.id} released its sandbox once the separate verifier collected the "
                    "artifacts; fork or act before grading"
                )
            raise RuntimeError(f"Harbor session {self.id} is closed")
        return self.sandbox

    async def _verify(self, users: SessionUsers) -> Outcome:
        """Grade in the agents' own sandbox (``shared`` mode): platoon's commands as its root, the tests as theirs, with
        ``[verifier].env``, which a stack rewrites for its addressing, as ``main``'s own environment."""

        sandbox = users.root
        try:
            env = resolve_env(self.harbor_task.verifier_env)
        except KeyError as exc:
            return Outcome.infra("verifier_setup", reason=f"host variable {exc} is not set")
        if self.stack is not None:
            env = {name: self.stack.rewrite(value, service=MAIN) for name, value in env.items()}
        if (exited := await self._check_init(sandbox)) is not None:
            return exited
        if (unsafe := await self._kill_processes(sandbox, "before the tests")) is not None:
            return unsafe
        # Grading's own commands run in "/", which an agent cannot remove, unlike the workdir, where the tests run.
        if (unusable := await self._check_workdir(sandbox)) is not None:
            return unusable
        reset = await sandbox.exec(["/bin/sh", "-c", _RESET_SCRIPT, "platoon-reset"], cwd="/")
        if reset.exit_code == 0 and users.tests_as_user:  # the tests write there as their user
            reset = await sandbox.exec(["chmod", "777", VERIFIER_DIR], cwd="/")
        if reset.exit_code != 0:
            return Outcome.infra("verifier_setup", reason=reset.stderr.strip())
        await self._run_hooks(users, self.harbor_task.collect)
        stack = self.stack
        agents_network = stack.network if stack is not None else None
        try:
            if stack is not None:
                await stack.set_network(
                    self._service.stack_network(self.harbor_task, self.harbor_task.verifier_network)
                )
                if (down := await self._probe("before the tests")) is not None:
                    return down
                if (unread := await self._put_sidecar_logs(sandbox)) is not None:
                    return unread
            try:
                await self._upload_tests(users)
            except OSError as exc:
                return Outcome.infra("verifier_setup", reason=f"cannot upload the tests: {exc}")
            outcome = await self._run_tests(users, env, agents_sandbox=True)
            if stack is not None and (down := await self._probe("after the tests")) is not None:
                down.metadata.update(
                    {key: value for key, value in outcome.metadata.items() if key not in down.metadata}
                )
                return down
            return outcome
        except Exception:
            # Where a provider raises for a working directory it cannot enter (the process provider's OSError, and
            # docker's upload, whose mkdir runs there), a process the kill missed may have removed the workdir since
            # the check: the upload or the tests' command failed for the agents' doing.  Every other failure stands,
            # also when the sandbox cannot be asked.
            with contextlib.suppress(Exception):
                if (unusable := await self._check_workdir(sandbox)) is not None and unusable.trainable:
                    return unusable
            raise
        finally:
            await self._remove_tests(sandbox)
            if stack is not None and agents_network is not None:  # the agents' network again, for shared forks
                with contextlib.suppress(Exception):
                    await stack.set_network(agents_network)

    async def _grade_separately(self) -> Outcome:
        """Collect the artifacts from main (unless main was released and they were kept), then grade them in a
        verifier sandbox."""

        if (kept := self._kept_artifacts) is not None:
            return await self._verify_separately(kept.collected, kept.skipped, kept.logs)
        self._open_sandbox()
        sandbox = self._lease.sandbox
        into = Path(scratch.mkdtemp("platoon-harbor-artifacts-"))
        skipped: dict[str, str] = {}
        collected: list[CollectedArtifact] = []
        logs: dict[str, Path] = {}
        try:
            async with self._lease.gate.hold("exclusive"):
                # Nothing the agents left running may change the artifacts while they are collected from the
                # workdir, and an unusable workdir scores 0 before anything is collected.
                early = await self._check_init(sandbox)
                if early is None:
                    early = await self._kill_processes(sandbox, "before collecting the artifacts")
                if early is None:
                    early = await self._check_workdir(sandbox)
                if early is None:
                    early = await self._probe("before the artifacts were collected")
                if early is None:
                    collected, logs = await self._collect(sandbox, into, skipped)
                    early = await self._probe("after the artifacts were collected")
        except ArtifactError as exc:
            shutil.rmtree(into, ignore_errors=True)
            return Outcome.infra("artifact_collect", reason=str(exc))
        except BaseException:
            shutil.rmtree(into, ignore_errors=True)
            raise
        if early is not None:  # unsafe to collect, or nothing to collect from
            shutil.rmtree(into, ignore_errors=True)
            return early
        if skipped:
            logger.info(
                "Harbor task %s: left out of the artifacts, as not directories or regular files: %s",
                self.harbor_task.name,
                skipped,
            )
        if self._lease.holders == 1 and not self._lease.inflight and self.sandbox is not None:
            # Nothing else needs main: let it go while the verifier starts, and keep the artifacts for regrading.
            self._kept_artifacts = _KeptArtifacts(into, collected, skipped, logs)
            self.sandbox = None
            self._main_release = _in_background(self._lease.release())
            return await self._verify_separately(collected, skipped, logs)
        try:
            return await self._verify_separately(collected, skipped, logs)
        finally:
            shutil.rmtree(into, ignore_errors=True)

    async def _collect(
        self, sandbox: Sandbox, into: Path, skipped: dict[str, str]
    ) -> tuple[list[CollectedArtifact], dict[str, Path]]:
        """Collect what a separate verifier gets, into ``into``: as Harbor does, ``main``'s collect hooks, ``main``'s
        artifacts, then the sidecars' hooks and their artifacts, each from its service's own files (a path already
        collected from another service is left out, the first kept), and copies of the sidecars' log directories."""

        hooks = self.harbor_task.collect
        await self._run_hooks(self._lease.users, [hook for hook in hooks if hook.service == "main"])
        collected = await collect_artifacts(
            sandbox,
            [artifact for artifact in self.harbor_task.artifacts if artifact.service is None],
            workdir=self._lease.spec.workdir,
            into=into / "main",
            skipped=skipped,
            max_bytes=self._service.max_artifact_bytes,
        )
        await self._run_hooks(self._lease.users, [hook for hook in hooks if hook.service != "main"])
        stack = self.stack
        if stack is None:
            return collected, {}
        services = dict.fromkeys(artifact.service for artifact in self.harbor_task.artifacts if artifact.service)
        for service in services:
            theirs: dict[str, str] = {}
            gathered = await collect_artifacts(
                stack.services[service],
                [artifact for artifact in self.harbor_task.artifacts if artifact.service == service],
                workdir="/",
                into=into / f"service-{service}",
                skipped=theirs,
                max_bytes=self._service.max_artifact_bytes,
                convention=False,
            )
            skipped.update({f"{service}:{path}": why for path, why in theirs.items()})
            workdir = self._lease.spec.workdir
            for artifact in gathered:
                target = artifact.target(workdir)
                if any(_overlaps(target, earlier.target(workdir)) for earlier in collected):
                    skipped[f"{service}:{target}"] = "collision"  # as Harbor keeps the first claimant
                    continue
                collected.append(artifact)
        return collected, await self._copy_sidecar_logs(into / "sidecar-logs", owner=into)

    async def _run_hooks(self, users: SessionUsers, hooks: Sequence[CollectHook]) -> None:
        """Run collect ``hooks`` (:func:`~platoon.envs.harbor.sidecars.run_collect_hooks`), ``main``'s as the image's
        user or the one a hook names, noting the ones that failed in the outcome (``collect_failed``)."""

        if not hooks:
            return
        failed = await sidecars.run_collect_hooks(
            hooks, main=users.default, stack=self.stack, task_name=self.harbor_task.name, users=users.hooks
        )
        if failed:
            self._notes.setdefault("collect_failed", []).extend(failed)

    async def _probe(self, when: str) -> Outcome | None:
        """Probe the stack's node services (:meth:`_probe_node_services`), then its sidecars
        (:func:`~platoon.envs.harbor.sidecars.probe`): the ``invalid`` outcome when one is down, unless a sidecar is and
        the service grades anyway (``on_sidecar_down: grade``), which notes it (``sidecars_down``).  None for a single
        sandbox."""

        stack = self.stack
        if stack is None or not isinstance(self._lease, StackLease):
            return None
        if (down := await self._probe_node_services(self._lease, when)) is not None:
            return down
        try:
            down, states = await sidecars.probe(stack, when=when)
        except Exception as exc:
            return Outcome.infra("sidecar_probe", reason=f"cannot probe the stack's sidecars {when}: {exc}")
        self._states = states
        if down is None:
            return None
        if self._service.on_sidecar_down == "grade":
            self._notes.setdefault("sidecars_down", []).append(down.reason)
            return None
        return down

    async def _probe_node_services(self, lease: StackLease, when: str) -> Outcome | None:
        """The ``invalid`` outcome (``node_service_down``) when the stack's rollout may not trust what a node service
        it reaches answered it, checked now (:meth:`NodeScope.status
        <platoon.envs.harbor.node_services.client.NodeScope.status>`); None while it may trust each one."""

        for name, scope in lease.scopes.items():
            up, detail = await scope.status()
            if up:
                continue
            reason = f"the node service {scope.spec.name}, which serves {name}, is down {when}: {detail}"
            logger.info("Harbor stack %s: %s", lease.stack.id, reason)
            return Outcome.infra(
                "node_service_down",
                status="invalid",
                reason=reason,
                metadata={"node_service": scope.spec.name, "service": name, "node_service_detail": detail},
            )
        return None

    async def _put_sidecar_logs(self, sandbox: Sandbox) -> Outcome | None:
        """Put what the sidecars wrote in their log directories into those of ``sandbox``, made afresh by now; an
        infra outcome (``verifier_setup``) when that cannot be done."""

        if not (isinstance(self._lease, StackLease) and self._lease.logs):
            return None
        directory = Path(scratch.mkdtemp("platoon-harbor-sidecar-logs-"))
        try:
            await sidecars.put_logs(sandbox, await self._copy_sidecar_logs(directory, owner=directory))
        except (ArtifactError, OSError) as exc:
            return Outcome.infra("verifier_setup", reason=str(exc))
        finally:
            shutil.rmtree(directory, ignore_errors=True)
        return None

    async def _copy_sidecar_logs(self, into: Path, *, owner: Path) -> dict[str, Path]:
        """Copy the sidecars' log directories into ``into`` (:func:`~platoon.envs.harbor.sidecars.snapshot_logs`, in a
        thread), each from where :func:`~platoon.envs.harbor.sidecars.log_source` finds it, which downloads into
        ``owner``, the scratch directory that holds ``into``: the copies by log directory, or :class:`ArtifactError`
        for what cannot be read.  A grade cancelled meanwhile leaves the thread running, and ``owner`` is deleted once
        it is done."""

        stack = self.stack
        if stack is None or not isinstance(self._lease, StackLease) or not self._lease.logs:
            return {}
        downloads = owner / "sidecar-downloads"
        directories = {entry.logs: await sidecars.log_source(stack, entry, downloads) for entry in self._lease.logs}
        try:
            return await sidecars.in_thread(
                lambda: sidecars.snapshot_logs(directories, into),
                undo=lambda _: shutil.rmtree(owner, ignore_errors=True),
            )
        except OSError as exc:
            raise ArtifactError(f"cannot copy what the sidecars wrote in their log directories: {exc}") from exc

    async def _verify_separately(
        self, collected: list[CollectedArtifact], skipped: dict[str, str], logs: dict[str, Path] | None = None
    ) -> Outcome:
        try:
            env = resolve_env(self.harbor_task.verifier_env)
        except KeyError as exc:
            return Outcome.infra("verifier_setup", reason=f"host variable {exc} is not set")
        try:
            users, spec = await self._service.open_verifier(self.harbor_task, rollout_id=self.rollout_id)
        except EnvSetupError as exc:
            return Outcome.infra("verifier_start", reason=str(exc), metadata={"setup_failure_kind": exc.failure_kind})
        verifier = users.root
        try:
            bundled = (
                self.harbor_task.verifier_environment is not None
                and self.harbor_task.verifier_environment.bundled_tests
            )
            directories = [VERIFIER_DIR] if bundled else [TESTS_DIR, VERIFIER_DIR]
            reset = f"rm -rf {' '.join(directories)} && mkdir -p {' '.join(directories)}"
            if users.tests_as_user:  # the tests write there as their user
                reset += f" && chmod 777 {VERIFIER_DIR}"
            result = await verifier.exec(["/bin/sh", "-c", reset], cwd="/")
            if result.exit_code != 0:
                return Outcome.infra("verifier_setup", reason=result.stderr.strip())
            try:
                await materialize_artifacts(verifier, collected, workdir=spec.workdir, readable=users.honors)
                await sidecars.put_logs(verifier, logs or {})
            except (ArtifactError, OSError) as exc:
                return Outcome.infra("verifier_setup", reason=str(exc))
            if not bundled:  # a verifier image built for the task holds its own /tests
                try:
                    await self._upload_tests(users)
                except OSError as exc:
                    return Outcome.infra("verifier_setup", reason=f"cannot upload the tests: {exc}")
            outcome = await self._run_tests(users, env, agents_sandbox=False)
            outcome.metadata["artifacts"] = sorted(artifact.target(spec.workdir) for artifact in collected)
            if skipped:
                outcome.metadata["artifacts_skipped"] = dict(sorted(skipped.items())[:_MAX_SKIPPED_REPORTED])
                outcome.metadata["artifacts_skipped_count"] = len(skipped)
            return outcome
        finally:
            await asyncio.shield(_in_background(verifier.stop()))

    async def _upload_tests(self, users: SessionUsers) -> None:
        """Upload the task's tests to ``/tests`` of the sandbox whose ``users`` these are: the tests' own where they run
        as a user other than root, else root's with the host's modes, as Harbor's copy keeps them."""

        uploader = users.tests if users.tests_as_user else unviewed(users.root)
        await uploader.upload(self.harbor_task.tests_dir, TESTS_DIR)

    async def _run_tests(self, users: SessionUsers, env: dict[str, str], *, agents_sandbox: bool) -> Outcome:
        """Run the tests, as their user, in the agents' own sandbox (``shared``) or a separate verifier's (the sandbox
        whose ``users`` these are), and read their verdict."""

        sandbox = users.root
        timeout_s = self.harbor_task.verifier_timeout_s * self._service.verifier_timeout_multiplier
        started = time.monotonic()
        result = await users.tests.exec(
            f"bash {TESTS_DIR}/test.sh > {VERIFIER_DIR}/{TEST_STDOUT} 2>&1", timeout_s=timeout_s, env=env
        )
        metadata: dict[str, Any] = {
            "verifier_exit_code": result.exit_code,
            "verifier_seconds": round(time.monotonic() - started, 3),
        }
        if (line := cwd_error(result)) is not None:  # the sandbox says the tests' command never entered the workdir
            metadata["verifier_cwd_error"] = line
        # The tests may have run the agent's code: nothing left of it may change their output while it is read.
        if (unsafe := await self._kill_processes(sandbox, "after the tests")) is not None:
            unsafe.metadata.update(metadata)
            return unsafe
        # A command that could not start in its working directory exits 126, and so may a command by itself (whose
        # output, the tests' here, could say anything): a check run in "/" confirms it.  When a process the kill missed
        # removed the agents' workdir since it was checked, the tests never ran and what /logs/verifier holds is not
        # their output.  A separate verifier's sandbox never held the agents' processes, and its workdir is the
        # verifier's own: its tests' exit status stands, and the reward files say what happened.
        if agents_sandbox and result.exit_code == CWD_ERROR_EXIT_CODE and not result.timed_out:
            if (unusable := await self._check_workdir(sandbox)) is not None:
                unusable.metadata.update(metadata)
                return unusable
        try:
            files = await collect(sandbox, VERIFIER_DIR)
        except RuntimeError as exc:
            return Outcome.infra("verifier_output", reason=str(exc), metadata=metadata)
        if result.timed_out:
            outcome = Outcome.infra("verifier_timeout", status="invalid", metadata={"timeout_s": timeout_s})
        elif (suspect := suspect_outcome(files, primary=self._service.reward.names[0])) is not None:
            outcome = suspect
        else:
            with scratch.temporary_directory() as local_dir:
                # A directory named verifier, as reward-contract hooks expect (platoon.envs.harbor.grading).
                local = write_local(files, Path(local_dir) / "verifier")
                outcome = self._service.verdict(local, self.harbor_task)
        if (output := output_tail(files)) is not None:
            metadata["test_output"] = output
        outcome.metadata.update(metadata)
        return outcome

    async def _kill_processes(self, sandbox: Sandbox, when: str) -> Outcome | None:
        """Kill every process left in the sandbox, so that nothing the agents started races the tests or the reading
        of their output; an infra outcome when that fails, since the grade could not be trusted."""

        kill = getattr(sandbox, "kill_processes", None)
        if kill is None:
            if type(sandbox) not in _UNKILLABLE:
                _UNKILLABLE.add(type(sandbox))
                logger.warning(
                    "%s has no kill_processes: Harbor grading cannot stop what agents leave running in its sandboxes",
                    type(sandbox).__name__,
                )
            return None
        try:
            await kill()
        except Exception as exc:
            logger.warning("Could not kill the processes left in sandbox %s %s: %s", sandbox.id, when, exc)
            return await self._lost_main(when) or Outcome.infra(
                "verifier_isolation", reason=f"could not kill the processes left in the sandbox {when}: {exc}"
            )
        return None

    async def _lost_main(self, when: str) -> Outcome | None:
        """Why a stack's ``main`` cannot be graded, as a probe of it finds: the stack cannot be probed (its pod is gone,
        say: ``sidecar_probe``), or ``main`` exited (``main_exited``, as Docker would have ended its container); None
        for a single sandbox, or a ``main`` that is up."""

        stack = self.stack
        if stack is None:
            return None
        try:
            state = (await stack.health([MAIN]))[MAIN]
        except Exception as exc:
            return Outcome.infra("sidecar_probe", reason=f"cannot probe the stack's services {when}: {exc}")
        if state.status != "exited":
            return None
        return Outcome.infra("main_exited", reason=f"main {state.describe()} {when}")

    async def _check_init(self, sandbox: Sandbox) -> Outcome | None:
        """None unless the main-only compose file's init (main's command) is gone: Docker would have ended the
        container with it, so the attempt is an infra ``error`` (``main_exited``), never graded."""

        init = self._lease.init
        if init is None:
            return None
        try:
            alive = await alive_processes(sandbox, {init})
        except Exception as exc:
            return Outcome.infra("verifier_setup", reason=f"cannot tell whether main's command runs: {exc}")
        if alive:
            return None
        logger.info("Harbor task %s: main's command (%s) exited before grading", self.harbor_task.name, init)
        return Outcome.infra(
            "main_exited",
            reason="main's command, which the task's compose file gives it, exited before grading",
            metadata={"main_init": init},
        )

    async def _check_workdir(self, sandbox: Sandbox) -> Outcome | None:
        """None when the agents' workdir is a directory a command can start in.  Else the agents made it unusable
        (grading never touches it): they removed or moved it, left something else in its place, or took its
        permissions, so the attempt scores 0 under the reward contract's first name (``workdir_missing``), an outcome
        whose ``metadata`` names the workdir and what is there instead.  An infra ``verifier_setup`` outcome when the
        sandbox's shell cannot tell, and the sandbox's own errors propagate: neither says anything about the
        policy."""

        workdir = self._lease.spec.workdir
        result = await sandbox.exec(
            ["/bin/sh", "-c", _WORKDIR_SCRIPT, "platoon-workdir", workdir], cwd="/", timeout_s=_WORKDIR_TIMEOUT_S
        )
        state = result.stdout.strip()
        if result.timed_out or result.exit_code != 0 or state not in _WORKDIR_STATES:
            detail = result.stderr.strip() or ("timed out" if result.timed_out else f"exit {result.exit_code}")
            return Outcome.infra("verifier_setup", reason=f"cannot check the workdir {workdir}: {detail}")
        if state == "ok":
            return None
        logger.info(
            "Harbor task %s: the workdir %s is %s, so the attempt scores 0 without its tests (workdir_missing)",
            self.harbor_task.name,
            workdir,
            state.replace("_", " "),
        )
        return self._service.reward.zero(
            reason="workdir_missing", metadata={"workdir": workdir, "workdir_state": state}
        )

    async def _remove_tests(self, sandbox: Sandbox) -> None:
        """Later agents sharing this sandbox must not see the tests, also after a failed or cancelled grade."""

        try:
            removed = (await sandbox.exec(["rm", "-rf", TESTS_DIR], cwd="/")).exit_code == 0
        except Exception:
            removed = False
        if not removed:
            logger.warning("Could not remove %s from sandbox %s after grading", TESTS_DIR, sandbox.id)


def _closed() -> ToolResult:
    return ToolResult.text("The session is closed", is_error=True)


def _task_tool_spec(name: str, tool: Mapping[str, Any]) -> ToolSpec:
    """A task MCP server's tool (MCP's ``Tool``) as the env tool ``name``."""

    annotations = tool.get("annotations")
    meta = tool.get("_meta")
    schema = tool.get("inputSchema")
    return ToolSpec(
        name=name,
        description=str(tool.get("description") or ""),
        input_schema=dict(schema) if isinstance(schema, Mapping) else {"type": "object"},
        read_only=isinstance(annotations, Mapping) and annotations.get("readOnlyHint") is True,
        meta=dict(meta) if isinstance(meta, Mapping) else None,
    )


def _task_tool_result(result: Mapping[str, Any]) -> ToolResult:
    """A task MCP server's ``CallToolResult`` as an env tool's result: its content blocks (its structured content as
    JSON text where it gives no block), and whether it is an error.  Images and audio stay as they are, and every other
    block is text (:func:`_task_tool_block`), cut as the generic tools' output is
    (:func:`~platoon.envs.harbor.tools.truncate_output`): text blocks longer together become one, cut, in the first
    one's place."""

    content = [_task_tool_block(block) for block in result.get("content") or () if isinstance(block, Mapping)]
    if not content and "structuredContent" in result:
        content = [{"type": "text", "text": json.dumps(result["structuredContent"])}]
    texts = [index for index, block in enumerate(content) if block["type"] == "text"]
    text = "\n".join(content[index]["text"] for index in texts)
    if len(text) > DEFAULT_MAX_OUTPUT_CHARS:
        content = [
            {"type": "text", "text": truncate_output(text)} if index == texts[0] else block
            for index, block in enumerate(content)
            if index == texts[0] or index not in texts
        ]
    return ToolResult(content=content, is_error=result.get("isError") is True)


def _task_tool_block(block: Mapping[str, Any]) -> dict[str, Any]:
    """A content block of a task MCP server's result: an image or audio as it is, a text block's text, and any other
    block (an embedded resource, a link, a block of no type or of one MCP does not have) as its JSON text, so that the
    result's cut bounds what it holds."""

    if block.get("type") in ("image", "audio") and isinstance(block.get("data"), str):
        return dict(block)
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    return {"type": "text", "text": json.dumps(block)}


def _in_main(stack: Stack, url: str) -> str:
    """A task MCP server's ``url`` as ``main`` reaches it in ``stack``: rewritten for the stack's addressing (the
    ``process`` provider's ports), which finds a service's port where the URL names it, so http's port 80 is named for
    the rewrite where the URL names none."""

    parts = urllib.parse.urlsplit(url)
    try:
        names_a_port = parts.port is not None
    except ValueError:  # it names one, if not as a number
        names_a_port = True
    if parts.scheme != "http" or not parts.hostname or names_a_port:
        return stack.rewrite(url, service=MAIN)
    named = parts._replace(netloc=f"{parts.netloc}:80").geturl()
    rewritten = stack.rewrite(named, service=MAIN)
    return url if rewritten == named else rewritten


def _overlaps(path: str, other: str) -> bool:
    """Whether one of two sandbox paths is the other or below it."""

    return path == other or path.startswith(other.rstrip("/") + "/") or other.startswith(path.rstrip("/") + "/")


def _in_background(coroutine: Awaitable[Any]) -> asyncio.Task[Any]:
    """Run a teardown as a task that lives on when its caller stops waiting (asyncio keeps only weak references).

    A stopping rollout worker lets it finish within its deadline (:func:`platoon.utils.stopping.let_finish`).
    """

    task = let_finish(asyncio.ensure_future(coroutine))
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)
    task.add_done_callback(_log_failure)
    return task


def _log_failure(task: asyncio.Task[Any]) -> None:
    if not task.cancelled() and task.exception() is not None:
        logger.warning("Stopping a Harbor sandbox failed", exc_info=task.exception())
