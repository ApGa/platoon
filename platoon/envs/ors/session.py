"""ORS sessions: one rollout's view of an ORS environment, in process or on an env server.

A session wraps a *lease* on one live env: an ``Environment`` instance (in
process) or a remote session (``ors+http``). ``shared`` forks hold the same
lease, so they see the same state and share its call accounting: a per-session
``max_tool_calls`` cap, a finished latch after the first ``finished`` result,
and ``bridge_error`` payloads for calls refused after either.

The *bridge* names (``bridge_error``, ``result_format="bridge_json"``, the
``get_task`` and ``get_status`` bridge tools, ``bridge_events.jsonl``) come from
a stdio MCP bridge that serves an ORS env's tools to an MCP client such as
OpenHands: a session can show the model what such a bridge shows it, for
policies trained behind one.

Every operation on the live env goes through its gate (:mod:`platoon.envs.gate`),
which ``platoon_fork`` children share with their parent. Tool calls of every
agent run one at a time unless the env declares a tool concurrent; declared
terminal tools run alone, and so do ``platoon_grade``, copy snapshots,
``platoon_fork`` hooks and the first tool listing. A call not admitted within
the service's ``queue_timeout_s`` is not run (an ``env_busy`` error result, not
counted); an admitted call runs to the end even when its caller gives up. The
last release closes the lease at once and tears the env down in the background:
in-flight calls drain, then the env closes, each bounded by a timeout, on every
path. An in-process env is never torn down under env code that still runs (a
sync call outlives any timeout): its teardown then waits for that code to end.

Grading fails closed (see :meth:`OrsSession.grade`): only the env's own reward
is ever scored, and an env that broke without grading is an infra error.

A tool that raised is classified the same way in process and on an env server,
whatever the openreward client's version (:func:`_failure`):

* It is infrastructure when the session or its transport failed (remotely), when
  a terminal tool raised (a submit that crashed must never score 0), or when the
  tool raised one of the SDK's own session or transport errors: the env's own
  backing session or service failed (:func:`~platoon.envs.ors.sdk.infra_error`).
* Otherwise it is the policy's call gone wrong: the model sees the error, and an
  episode that never submits scores 0 (``no_submit``).  An openreward client
  from 0.1.163 on ends a remote session whose tool raised (``ToolFailed``), so
  the episode ends there, with the same grade; 0.1.125 and in-process sessions
  live on.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import re
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from platoon.envs.gate import CallClass, EnvBusy, GateClosed, Grant, ToolGate, caller_or
from platoon.envs.ors import sdk
from platoon.envs.ors.cleanup import RemoteSessionRecord, check_running, register, unregister
from platoon.envs.service import ForkMode, Outcome, ToolResult, ToolSpec
from platoon.task import Task
from platoon.utils.stopping import in_cleanup_scope, let_finish

if TYPE_CHECKING:
    from platoon.envs.ors.service import OrsEnvironmentService, TaskRef

logger = logging.getLogger(__name__)

ResultFormat = Literal["blocks", "bridge_json"]
BRIDGE_TOOLS = ("get_task", "get_status")


class InvalidCall(Exception):
    """The env refused a call (unknown tool, bad arguments, episode over); the session lives on."""

    def __init__(self, message: str, reason: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason


class EnvBusyError(EnvBusy):
    """A ``copy`` fork did not get its turn on the env within ``queue_timeout_s``, so it took no copy."""

    def __init__(self, env: str, busy: EnvBusy) -> None:
        super().__init__(busy.waited_s, busy.timeout_s)
        self.args = (
            f"{env} was still busy after {busy.waited_s:.1f}s (the calls ahead of the copy did not end within "
            f"queue_timeout_s={busy.timeout_s:g}s), so it could not be copied",
        )


class Backend(Protocol):
    """One live env behind a lease."""

    url: str | None  # the env server, None in process
    task_spec: Mapping[str, Any] | None

    async def prompt(self) -> list[Any]: ...

    async def tools(self) -> list[sdk.RawTool]: ...

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        """Run a tool and return its ``ToolOutput``; raise :class:`InvalidCall` when the env refuses it."""
        ...

    def env_attr(self, name: str) -> Any:
        """An attribute of the env instance (platoon hooks and hints), or None remotely."""
        ...

    def copy(self) -> Backend: ...

    async def close(self) -> None: ...


class LocalBackend:
    """An ``Environment`` instance in this process, driven like the SDK's server drives it.

    Env code runs through :meth:`run`, which keeps it in :attr:`busy` until it really ends: a sync call runs
    on in its thread after its caller gives up (cancellation, a timeout), so teardown waits for it (:meth:`close`)
    instead of racing it, unless the event loop ends first.
    """

    url = None

    def __init__(
        self, env: Any, *, owned: bool = True, busy: set[asyncio.Future[Any]] | None = None, timeout_s: float = 30.0
    ) -> None:
        self.env = env
        self.owned = owned  # False for platoon_fork views: their state belongs to the parent session
        # The env code running on this env, until it ends; platoon_fork views share their parent's set.
        self.busy: set[asyncio.Future[Any]] = set() if busy is None else busy
        self.timeout_s = timeout_s  # bounds a teardown

    @property
    def task_spec(self) -> Mapping[str, Any] | None:
        return getattr(self.env, "task_spec", None)

    @classmethod
    async def start(
        cls, env_class: type, task_spec: Mapping[str, Any], secrets: Mapping[str, str], timeout_s: float
    ) -> LocalBackend:
        env = await sdk.run_user_callable(env_class, task_spec=task_spec, secrets=dict(secrets))
        backend = cls(env, timeout_s=timeout_s)
        try:
            await backend.run(env.setup)
        except BaseException:
            await bounded(backend.close(), timeout_s, f"tear down {env_class.__name__} after a failed setup")
            raise
        return backend

    async def run(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run env code ``fn(*args)`` the SDK's way (:func:`sdk.run_user_callable`), tracked in :attr:`busy`."""

        return await self._tracked(
            sdk.run_user_callable(fn, *args), sync=not inspect.iscoroutinefunction(fn), what=sdk.callable_name(fn)
        )

    async def _tracked(self, work: Awaitable[Any], *, sync: bool, what: str) -> Any:
        """Await env code, in :attr:`busy` until it ends.  Async code is cancelled with its caller; sync code
        cannot be stopped, so a caller that gives up only abandons it, and it stays busy until its thread returns."""

        task = asyncio.ensure_future(work)
        self.busy.add(task)
        task.add_done_callback(self.busy.discard)
        if not sync:
            return await task
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            sdk.abandon(task, what)
            raise

    def running(self) -> list[asyncio.Future[Any]]:
        """The env code still running on this env (and its ``platoon_fork`` family)."""

        return [task for task in self.busy if not task.done()]

    async def prompt(self) -> list[Any]:
        return list(await self.run(self.env.get_prompt))

    async def tools(self) -> list[sdk.RawTool]:
        # Task tools may be listed by sync env code, so the listing is shielded like a sync call.
        return await self._tracked(sdk.local_tools(self.env), sync=True, what="list_task_tools")

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        fn = sdk.tool_function(self.env, name)
        sync = fn is not None and not inspect.iscoroutinefunction(fn)
        output = await self._tracked(self._dispatch(name, arguments), sync=sync, what=f"tool {name}")
        reply = getattr(output, "root", output)
        if getattr(reply, "ok", False) is not True:
            raise InvalidCall(str(getattr(reply, "error", reply)), getattr(reply, "reason", None))
        return reply.output

    async def _dispatch(self, name: str, arguments: dict[str, Any]) -> Any:
        try:
            from openreward.environments.session import call_session_tool
        except ImportError:  # pragma: no cover - both supported SDKs have it
            return await self.env._call_tool(name, arguments)
        return await call_session_tool(self.env, None, name, arguments)

    def env_attr(self, name: str) -> Any:
        return getattr(self.env, name, None)

    def copy(self) -> LocalBackend:
        """A copy of the env for a ``copy`` fork, which takes it while it runs alone on the env (its gate's
        exclusive turn), so no env code runs on the env while it is copied.

        An env that defines ``platoon_copy()`` (or ``__deepcopy__``) makes its own copy, which owns whatever it
        holds, so it is torn down like the original.  Any other env is deep-copied, which shares the external
        resources the env holds (a workdir, a sandbox): one that defines ``teardown`` is refused, since the
        copy's writes would change what the original is graded on and either teardown would free the other's
        resources.  A deep copy is torn down only if the original is (never a ``platoon_fork`` view's).
        """

        name = type(self.env).__name__
        hook = getattr(self.env, "platoon_copy", None)
        if callable(hook):
            clone = hook()
            if clone is None or not hasattr(clone, "get_prompt"):
                raise TypeError(f"{name}.platoon_copy must return an Environment, got {clone!r}")
            return LocalBackend(clone, timeout_s=self.timeout_s)
        if hasattr(type(self.env), "__deepcopy__"):
            return LocalBackend(sdk.deepcopy_env(self.env), timeout_s=self.timeout_s)
        if sdk.overrides_teardown(self.env):
            raise NotImplementedError(
                f"{name} has a teardown, so a deep copy would share the resources it releases; "
                "define platoon_copy() to copy it"
            )
        return LocalBackend(sdk.deepcopy_env(self.env), owned=self.owned, timeout_s=self.timeout_s)

    async def close(self) -> None:
        """Tear the env down (an owned one): now, or, while env code still runs, in the background once it ends.

        The wait has no time limit, since nothing bounds a sync call that never returns, so it never runs inside
        a caller's bounded close; the teardown itself is bounded by :attr:`timeout_s`.  A loop that ends during
        the wait still tears the env down as it ends, while that code may run on.  The background teardown joins
        the rollout's cleanup scope, so that an in-process rollout's permit waits for it, a bounded time, but is no
        cleanup that a stopping rollout worker lets finish (:func:`platoon.utils.stopping.in_cleanup_scope`).
        """

        if not self.owned:
            return
        if self.running():
            in_cleanup_scope(defer(self._teardown_when_idle(), f"tear down {type(self.env).__name__}"))
            return
        await self.run(self.env.teardown)

    async def _teardown_when_idle(self) -> None:
        try:
            while running := self.running():
                await asyncio.wait(running)
        finally:
            # Also when the wait is cancelled: asyncio.run cancels every task as its loop ends (a rollout worker's or
            # an evaluation slot's ends with its rollout) and runs the loop until they finish.  The code may still
            # run in its thread then, but a teardown that races it beats none.
            await bounded(self.run(self.env.teardown), self.timeout_s, f"tear down {type(self.env).__name__}")


class RemoteBackend:
    """A session on an ORS env server, through the openreward client.

    ``described_by`` is the class the server serves, when the service names it (``env_class``): its tools then take
    the descriptors the class gives them in process (:func:`sdk.with_class_descriptors`).
    """

    task_spec = None

    def __init__(
        self, api: Any, session: Any, url: str, environment: Any = None, *, described_by: type | None = None
    ) -> None:
        self.api = api
        self.session = session
        self.url = url
        self.environment = environment
        self.described_by = described_by
        # Listed until closed, so a process that dies before closing it can still have it deleted
        # (platoon.envs.ors.cleanup).
        self._registered = register(
            RemoteSessionRecord(
                url=url,
                sid=str(getattr(session, "sid", "")),
                api_key=getattr(session, "api_key", None),
                deployment=getattr(session, "deployment", None),
            )
        )

    @classmethod
    async def start(
        cls,
        *,
        url: str,
        env_name: str,
        variant: str | None,
        api_key: str,
        api_url: str | None,
        split: str,
        index: int,
        secrets: Mapping[str, str],
        timeout_s: float,
        described_by: type | None = None,
    ) -> RemoteBackend:
        check_running()  # a stopping process opens no session (platoon.envs.ors.cleanup.shut_down)
        client = sdk.import_client()
        # The environments API directly, not OpenReward(): that client reads URLs from process-wide variables.
        api = client.AsyncEnvironmentsAPI(base_url=url, api_key=api_key, api_base_url=api_url or url)
        session = None
        try:
            environment = api.get(env_name, variant=variant)
            session = environment.session(secrets=dict(secrets) or None, split=split, index=index)
            await session.__aenter__()
        except BaseException:
            await bounded(_close_remote(api, session), timeout_s, f"clean up a failed session on {url}")
            raise
        return cls(api, session, url, environment, described_by=described_by)

    async def task_name(self, split: str, index: int, timeout_s: float) -> str | None:
        """The task's display name, read from the task spec the server lists (best effort: None on failure)."""

        try:
            task = await asyncio.wait_for(self.environment.get_task(split, index), timeout_s)
        except Exception:
            logger.debug("Cannot read the name of task %s:%s on %s", split, index, self.url, exc_info=True)
            return None
        return sdk.task_name(getattr(task, "task_spec", None), index)

    async def prompt(self) -> list[Any]:
        check_running()
        return list(await self.session.get_prompt())

    async def tools(self) -> list[sdk.RawTool]:
        check_running()
        tools = await sdk.remote_tools(self.session)
        return tools if self.described_by is None else sdk.with_class_descriptors(tools, self.described_by)

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        check_running()  # a stopping process calls no tool, whatever its rollout still tries
        try:
            return await self.session.call_tool(name, arguments)
        except Exception as exc:
            if sdk.error_kind(exc) == "invalid_call":
                raise InvalidCall(str(getattr(exc, "detail", None) or exc), getattr(exc, "reason", None)) from exc
            raise

    def env_attr(self, name: str) -> Any:
        return None

    def copy(self) -> Backend:
        raise NotImplementedError(
            "A copy fork snapshots an in-process environment; an ors+http session cannot be copied "
            "(fork it shared or fresh instead)"
        )

    async def close(self) -> None:
        await _close_remote(self.api, self.session)
        unregister(self._registered)


async def _close_remote(api: Any, session: Any) -> None:
    try:
        if session is not None:
            await session.__aexit__(None, None, None)
    finally:
        await api.aclose()


_background: set[asyncio.Task[Any]] = set()


def defer(work: Awaitable[Any], what: str) -> asyncio.Task[Any]:
    """Run cleanup ``work`` in the background, strongly referenced until it ends; a failure is logged."""

    task = asyncio.ensure_future(work)
    _background.add(task)

    def finished(task: asyncio.Task[Any]) -> None:
        _background.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.warning("Failed to %s", what, exc_info=task.exception())

    task.add_done_callback(finished)
    return task


async def bounded(cleanup: Awaitable[Any], timeout_s: float, what: str) -> None:
    """Await ``cleanup`` for at most ``timeout_s``; a caller's cancellation lets it finish in the background.

    Cleanup never raises: a failure or a timeout is logged.  A stopping rollout worker lets it finish within its
    deadline (:func:`platoon.utils.stopping.let_finish`).
    """

    task = let_finish(asyncio.ensure_future(cleanup))
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout_s)
    except TimeoutError:
        task.cancel()
        logger.warning("Gave up after %.0fs trying to %s", timeout_s, what)
    except asyncio.CancelledError:
        asyncio.get_running_loop().call_later(timeout_s, task.cancel)
        raise
    except Exception:
        logger.warning("Failed to %s", what, exc_info=True)


class EventLog:
    """The bridge's records: ``bridge_events.jsonl`` (one line per event) and a ``bridge_state.json`` per env."""

    def __init__(self, directory: Path, *, env: str, ref: TaskRef) -> None:
        self.directory = directory
        self.events_path = directory / "bridge_events.jsonl"
        self._header = {"env": env, "split": ref.split, "task_index": ref.index, "task_name": ref.name}
        self._root: str | None = None

    def record(self, lease: Lease, event_type: str, payload: Mapping[str, Any]) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            record = {"type": event_type, "time": time.time(), "session": lease.id, **payload}
            with self.events_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, default=_json_default, sort_keys=True) + "\n")
            self._root = self._root or lease.id
            name = "bridge_state.json" if lease.id == self._root else f"bridge_state.{lease.id}.json"
            state = {
                **self._header,
                "turn": lease.turn,
                "finished": lease.finished,
                "last_reward": lease.last_reward,
                "updated_at": time.time(),
            }
            tmp = self.directory / f"{name}.tmp"
            tmp.write_text(json.dumps(state, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self.directory / name)
        except OSError:
            logger.warning("Cannot write ORS session events to %s", self.directory, exc_info=True)


class Lease:
    """One live env shared by a session and its ``shared`` forks, with the call accounting they share.

    ``gate`` orders every operation on the live env (:mod:`platoon.envs.gate`): a session's lease makes one, a
    ``platoon_fork`` child's lease shares its parent's (the child's state is the parent's), and a copy gets its
    own.  ``inflight`` holds the operations still running, which the last release drains; ``running`` counts the
    tool calls among them by class, until each one's result is on the lease.
    """

    def __init__(
        self,
        backend: Backend,
        *,
        log: EventLog | None = None,
        parent: Lease | None = None,
        gate: ToolGate | None = None,
    ) -> None:
        self.id = uuid.uuid4().hex[:8]
        self.backend = backend
        self.log = log
        self.turn = 0
        self.finished = False
        self.last_reward: float | None = None
        self.terminal: ToolResult | None = None  # the first finished result
        self.terminal_tool: str | None = None
        self.cap_hit = False
        self.errors: list[str] = []  # every call that raised: tools that raised, transport errors, a dead session
        # The failures that make an unsubmitted episode an infra error rather than no_submit: the session died, a
        # transport error, a terminal tool raised (a submit that crashed must never score 0), or a tool raised one of
        # the SDK's own session or transport errors (see _failure).
        self.infra_errors: list[str] = []
        self.dead: str | None = None  # "<ExceptionType>: <message>" of the failure that ended the session
        self.dead_text: str | None = None  # the same failure's plain message, which bridge_json results show
        # Whether that failure was infrastructure: a client from openreward 0.1.163 on also ends the session when the
        # policy's call made a tool raise, which grades as the policy's.
        self.dead_infra = False
        self.gate = gate if gate is not None else parent.gate if parent is not None else ToolGate()
        self.inflight: set[asyncio.Task[Any]] = set()
        self.running: Counter[CallClass] = Counter()
        self.closed = False  # the last holder released it: nothing new starts
        self._tools: list[sdk.RawTool] | None = None
        self._holders = 1
        self._parent = parent  # a platoon_fork view keeps its parent's env alive until it closes
        self._closing: asyncio.Task[Any] | None = None

    def acquire(self) -> Lease:
        if self._holders <= 0:
            raise RuntimeError(f"ORS env session {self.id} is closed")
        self._holders += 1
        return self

    async def release(self, timeout_s: float) -> None:
        """Drop a holder.  The last one closes the lease at once, so nothing new starts, and tears the env down in
        a background task the caller waits for: in-flight calls drain (at most ``timeout_s``), the backend closes
        (at most ``timeout_s``), then a ``platoon_fork`` child releases its parent.  A caller that stops waiting
        (a close under a shorter timeout, a cancelled close) leaves the teardown running instead of cutting it
        short, so a child's parent is released on every path."""

        closing = self.release_nowait(timeout_s)
        if closing is not None:
            await asyncio.shield(closing)

    def release_nowait(self, timeout_s: float) -> asyncio.Task[Any] | None:
        """Drop a holder without waiting: the last one starts the teardown (:meth:`release`) and returns it."""

        if self._holders > 0:
            self._holders -= 1
        if self._holders > 0:
            return None
        if self._closing is None:
            self.closed = True
            # Each of its steps is bounded, so a stopping rollout worker lets it finish (platoon.utils.stopping).
            self._closing = let_finish(defer(self._teardown(timeout_s), f"tear down ORS env session {self.id}"))
        return self._closing

    async def _teardown(self, timeout_s: float) -> None:
        try:
            if self.inflight:
                _, running = await asyncio.wait(set(self.inflight), timeout=timeout_s)
                for operation in running:  # it runs on with nobody waiting for it; an in-process env waits below
                    sdk.abandon(operation, f"an operation on ORS env session {self.id} after {timeout_s:.0f}s")
            if self._parent is None:
                self.gate.close()  # its own gate: every view of it is closed
            self.record("session_closing", {"finished": self.finished, "last_reward": self.last_reward})
            # A platoon_fork child's env belongs to its parent: nothing to close, so the parent is released at once.
            if getattr(self.backend, "owned", True):
                await bounded(self.backend.close(), timeout_s, f"close ORS env session {self.id}")
        finally:
            if self._parent is not None:
                await self._parent.release(timeout_s)

    async def raw_tools(self) -> list[sdk.RawTool]:
        """The env's tools, fetched once.  The fetch runs alone on the env: a listing may run env code."""

        if self._tools is None:
            await self.gate.run("exclusive", self._fetch_tools, inflight=self.inflight)
        assert self._tools is not None
        return self._tools

    async def _fetch_tools(self, grant: Grant) -> None:
        if self._tools is None:  # another fetch may have run while this one waited for its turn
            self._tools = await self.backend.tools()

    def snapshot(self, backend: Backend) -> Lease:
        """A new lease, with its own gate, on a copied env that starts from this one's accounting."""

        clone = Lease(backend, log=self.log, gate=ToolGate(queue_timeout_s=self.gate.queue_timeout_s))
        for name in ("turn", "finished", "last_reward", "terminal", "terminal_tool", "cap_hit", "dead_infra"):
            setattr(clone, name, getattr(self, name))
        clone.errors = list(self.errors)
        clone.infra_errors = list(self.infra_errors)
        clone._tools = self._tools
        return clone

    def record(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if self.log is not None:
            self.log.record(self, event_type, payload)


class OrsSession:
    """A rollout's view of an ORS environment: instruction, env tools, grading and forks.

    ``instruction`` is the task's goal: the env's prompt (``get_prompt``) for a
    task the service resolved, the given goal for a forked child. ``prompt`` and
    ``prompt_blocks`` keep the env's own prompt (text, and MCP content blocks
    with any images).
    """

    # call_tool takes calls from every agent at once and orders them on the live env's gate (platoon.envs.gate).
    manages_concurrency = True

    def __init__(
        self,
        service: OrsEnvironmentService,
        task: Task,
        lease: Lease,
        *,
        ref: TaskRef,
        instruction: str,
        prompt: str,
        prompt_blocks: list[dict[str, Any]],
        rollout_id: str,
    ) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.task = task
        self.instruction = instruction
        self.prompt = prompt
        self.prompt_blocks = prompt_blocks
        self.rollout_id = rollout_id
        self.sandbox = None
        self.ref = ref
        self._service = service
        self._lease: Lease | None = lease

    @property
    def closed(self) -> bool:
        return self._lease is None

    @property
    def url(self) -> str | None:
        """The env server this session runs on (None in process)."""

        return None if self._lease is None else self._lease.backend.url

    async def tools(self) -> list[ToolSpec]:
        lease = self._require_lease()
        raw = await lease.raw_tools()
        specs = self._service.tool_specs(raw, lease)
        if self._service.bridge_tools:
            names = {tool.name for tool in raw}
            specs[:0] = [spec for spec in _bridge_tool_specs() if spec.name not in names]
        return specs

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        """Call an env tool in its turn among every operation on the live env (:mod:`platoon.envs.gate`).

        A call refused before it runs (episode over, session dead or closed, the cap hit, or not admitted within
        ``queue_timeout_s``) never reaches the env and is not counted.  An admitted call runs to the end and is
        recorded on the lease, even when its caller gives up.
        """

        lease = self._lease
        if lease is None:
            return self._closed_result()
        if not isinstance(arguments, Mapping) or not all(isinstance(key, str) for key in arguments):
            return ToolResult.text("Tool arguments must be a JSON object", is_error=True)
        arguments = dict(arguments)
        try:
            if self._service.bridge_tools and name in BRIDGE_TOOLS and name not in await self._env_tool_names(lease):
                return await self._bridge_tool(lease, name)
            refused = self._refusal(lease, name)
            if refused is not None:
                return refused  # no need to wait for a turn
            call_class = await self._service.call_class(name, lease)
            return await lease.gate.run(
                call_class,
                lambda grant: self._run_call(lease, name, arguments, grant),
                view=caller_or(self),  # the calling agent takes its turn, this view's when the plane names none
                inflight=lease.inflight,
            )
        except EnvBusy as busy:
            return self._not_run(lease, name, busy)
        except GateClosed:
            return self._closed_result()

    async def _run_call(self, lease: Lease, name: str, arguments: dict[str, Any], grant: Grant) -> ToolResult:
        """Run an admitted call.  The refusals are checked again: the calls ahead of it may have ended the episode
        or reached the cap, and its own session may have closed while it waited."""

        if self._lease is not lease or lease.closed:
            return self._closed_result()
        refused = self._refusal(lease, name)
        if refused is not None:
            return refused
        lease.running[grant.call_class] += 1  # until its result is on the lease (see grade)
        try:
            return await self._execute(lease, name, arguments, grant)
        finally:
            lease.running[grant.call_class] -= 1

    async def _execute(self, lease: Lease, name: str, arguments: dict[str, Any], grant: Grant) -> ToolResult:
        lease.turn += 1
        turn = lease.turn  # this call's own: other calls may start and count while it runs
        call_id = f"call_{turn:04d}_{_slug(name)}"
        lock_wait_ms = round(grant.waited_s * 1000, 1)
        lease.record(
            "tool_call",
            {"turn": turn, "call_id": call_id, "tool_name": name, "arguments": arguments, "lock_wait_ms": lock_wait_ms},
        )
        try:
            output = await lease.backend.call(name, dict(arguments))
        except InvalidCall as exc:
            lease.record("tool_error", {"turn": turn, "call_id": call_id, "tool_name": name, "error": str(exc)})
            return self._error_result(name, str(exc), {"error_reason": exc.reason or "invalid_call"})
        except Exception as exc:
            kind, infra, kills = _failure(exc, remote=lease.backend.url is not None)
            infra = infra or self._is_terminal_tool(lease, name, grant)
            message = f"{type(exc).__name__}: {exc}"
            lease.errors.append(f"{name}: {message}")
            if kills and lease.dead is None:
                lease.dead, lease.dead_text, lease.dead_infra = message, str(exc), infra
            if infra:
                lease.infra_errors.append(f"{name}: {message}")
            logger.warning("ORS tool %s failed (%s): %s", name, kind, message)
            lease.record("tool_error", {"turn": turn, "call_id": call_id, "tool_name": name, "error": message})
            return self._error_result(
                name, message, {"error_reason": kind, "error_type": type(exc).__name__}, bridge_text=str(exc)
            )

        result, payload = self._result(name, output)
        if result.finished:
            lease.finished = True
            # The fallback for envs that declare no terminal tool: one that ended an episode is listed as terminal
            # from now on (OrsEnvironmentService.tool_specs ignores it for an env that declares its terminal tools).
            self._service.learned_terminal.add(name)
            if lease.terminal is None:
                lease.terminal, lease.terminal_tool = result, name
        if _is_number(result.reward):
            lease.last_reward = float(result.reward)  # type: ignore[arg-type]
        lease.record("tool_result", {"turn": turn, "call_id": call_id, "tool_name": name, "result": payload})
        return result

    def _refusal(self, lease: Lease, name: str) -> ToolResult | None:
        """The bridge's reply to a call it refuses without running it, or None to run it."""

        if lease.finished:
            return self._bridge_error(lease, "environment_already_finished")
        if lease.dead is not None:
            text = lease.dead_text if self._service.result_format == "bridge_json" else None
            return self._error_result(
                name, f"The env session is gone: {lease.dead}", {"error_reason": "dead"}, bridge_text=text
            )
        cap = self._service.max_tool_calls
        if cap > 0 and lease.turn >= cap:
            lease.finished = lease.cap_hit = True
            lease.record("max_tool_calls_exceeded", {"max_tool_calls": cap, "tool_name": name})
            return self._bridge_error(lease, "max_tool_calls_exceeded")
        return None

    def _not_run(self, lease: Lease, name: str, busy: EnvBusy) -> ToolResult:
        """A call that never got its turn within the queue deadline: not run, not counted, logged."""

        waited_s = round(busy.waited_s, 3)
        logger.warning(
            "ORS tool %s of session %s not run: env busy (waited %.1fs, the limit is %gs)",
            name,
            self.id,
            busy.waited_s,
            busy.timeout_s,
        )
        lease.record(
            "tool_not_run",
            {"tool_name": name, "reason": "env_busy", "waited_s": waited_s, "queue_timeout_s": busy.timeout_s},
        )
        message = (
            f"Not run: env busy. The call waited {busy.waited_s:.0f} s for other calls on this environment to "
            f"finish (the limit is {busy.timeout_s:g} s); it was not executed and does not count. Try again."
        )
        return self._error_result(name, message, {"error_reason": "env_busy", "waited_s": waited_s})

    def _closed_result(self) -> ToolResult:
        return ToolResult.text(f"ORS session {self.id} is closed", is_error=True)

    async def grade(self) -> Outcome:
        """Grade the session, failing closed.

        * The first ``finished`` result decides: its reward is ``scored``, metadata
          ``{"invalid": true}`` is ``invalid``, and no numeric reward is ``invalid``.
        * Without one: hitting ``max_tool_calls`` scores the last reward (else 0)
          with ``reason="max_tool_calls"``; an in-process env's ``platoon_grade()``
          grades from state; an infrastructure failure (a session that died for
          one, a transport error, a terminal tool that raised, or a tool that
          raised one of the SDK's session or transport errors) is an infra
          ``error`` (``env_error``); otherwise the policy never submitted: scored
          0, ``reason="no_submit"``, with the errors of any tools that raised in
          its metadata (a session that 0.1.163's client ended because the
          policy's call made a tool raise included).

        A grade never reads a half-applied call.  ``platoon_grade()`` reads the
        env's state, so it runs alone on the live env, after the calls in flight;
        if it cannot start within ``queue_timeout_s`` the grade is an infra
        ``error`` (``env_busy``).  Every other grade reads only this lease, which
        a call updates when it ends: it waits for the lease's calls in flight (at
        most ``queue_timeout_s``) and is ``env_busy`` only when a call that may
        still end the episode runs on (:meth:`_may_end_episode`).
        """

        lease = self._lease
        if lease is None:
            return Outcome.infra("session_closed")
        if lease.terminal is not None:  # the first finished result decides, and nothing can change it
            return _terminal_outcome(lease.terminal, lease.terminal_tool or "", {"tool_calls": lease.turn})
        if callable(lease.backend.env_attr("platoon_grade")):  # it reads the env's state: grade alone on the env
            try:
                return await lease.gate.run("exclusive", lambda grant: self._grade(lease), inflight=lease.inflight)
            except EnvBusy as busy:
                return self._grade_busy(lease, busy.waited_s, "for its turn on the env")
            except GateClosed:
                return Outcome.infra("session_closed")
        # Any other grade reads only the lease, which a call updates when it ends: wait for the calls in flight.
        loop = asyncio.get_running_loop()
        started = loop.time()
        if lease.inflight:
            await asyncio.wait(set(lease.inflight), timeout=lease.gate.queue_timeout_s)
        if self._lease is lease and self._may_end_episode(lease):
            waited = "for the calls in flight on the env, and one that may end the episode still runs"
            return self._grade_busy(lease, loop.time() - started, waited)
        return await self._grade(lease)

    def _may_end_episode(self, lease: Lease) -> bool:
        """Whether a tool call still running on the lease may end the episode: a declared terminal call, or any call
        when the env declares no terminal tool (any tool may then end it)."""

        if lease.running["exclusive"]:
            return True
        return any(lease.running.values()) and not self._service.declares_terminal(lease._tools or (), lease)

    def _grade_busy(self, lease: Lease, waited_s: float, what: str) -> Outcome:
        logger.warning("Grading ORS session %s waited %.1fs %s; gave up", self.id, waited_s, what)
        reason = f"grading waited {waited_s:.1f}s {what}"
        return Outcome.infra("env_busy", reason=reason, metadata={"tool_calls": lease.turn})

    async def _grade(self, lease: Lease) -> Outcome:
        if self._lease is not lease:  # closed while the grade waited
            return Outcome.infra("session_closed")
        metadata: dict[str, Any] = {"tool_calls": lease.turn}
        if lease.terminal is not None:
            return _terminal_outcome(lease.terminal, lease.terminal_tool or "", metadata)
        if lease.cap_hit:
            reward = lease.last_reward if lease.last_reward is not None else 0.0
            return Outcome(status="scored", rewards={"reward": reward}, reason="max_tool_calls", metadata=metadata)
        grader = lease.backend.env_attr("platoon_grade")
        if callable(grader):
            try:
                return _graded_outcome(await _run_env_code(lease.backend, grader), metadata)
            except Exception as exc:
                logger.exception("platoon_grade raised")
                return Outcome.infra(f"unexpected:{type(exc).__name__}", reason=f"platoon_grade raised: {exc}")
        if lease.infra_errors:
            reason = lease.dead if lease.dead is not None and lease.dead_infra else lease.infra_errors[-1]
            return Outcome.infra("env_error", reason=reason, metadata={**metadata, "errors": lease.errors[-5:]})
        if lease.errors:
            metadata["errors"] = lease.errors[-5:]
        return Outcome(status="scored", rewards={"reward": 0.0}, reason="no_submit", metadata=metadata)

    async def fork(self, mode: ForkMode = "shared", *, task: Task | None = None) -> OrsSession:
        """``shared``: the same live env (an env's ``platoon_fork(task_spec)`` hook gives a child its own goal on
        shared state); ``fresh``: a new env for the same task; ``copy``: a copy of an in-process env.

        A ``platoon_fork`` hook and a copy run alone on the env (its gate's exclusive turn); one that cannot
        start within ``queue_timeout_s`` raises :class:`~platoon.envs.gate.EnvBusy` (:class:`EnvBusyError` for a
        copy), and nothing was forked.
        """

        lease = self._require_lease()
        child_task = task or self.task
        if mode == "shared":
            hook = lease.backend.env_attr("platoon_fork") if task is not None else None
            if not callable(hook):
                return self._view(child_task, lease.acquire())
            spec = {**(lease.backend.task_spec or {}), "goal": child_task.goal}
            # Hold the parent's env from before the hook runs: closing the parent meanwhile must not tear it down,
            # and the child takes this hold over.
            parent = lease.acquire()
            try:
                # The hook reads the shared state, so it runs alone on the env.
                env = await lease.gate.run(
                    "exclusive", lambda grant: _run_env_code(lease.backend, hook, spec), inflight=lease.inflight
                )
                if env is None or not hasattr(env, "get_prompt"):
                    raise TypeError(f"platoon_fork must return an Environment, got {env!r}")
            except BaseException:
                # No child: let go of the parent now.  A hook that runs on (its caller was cancelled) is an operation
                # in flight on the env, which the parent's teardown waits for.
                parent.release_nowait(self._service.close_timeout_s)
                raise
            # The child's env shares the parent's state, so its env code counts as the family's (LocalBackend.busy).
            backend = LocalBackend(
                env, owned=False, busy=getattr(lease.backend, "busy", None), timeout_s=self._service.close_timeout_s
            )
            return self._view(child_task, Lease(backend, log=lease.log, parent=parent))
        if mode == "fresh":
            return await self._service.open_fork(
                child_task, ref=self.ref, rollout_id=self.rollout_id, url=self.url, log=lease.log
            )
        if mode == "copy":
            # The copy runs alone on the env, so it never copies a call half-applied, a lock a call holds or state a
            # sync tool's thread is changing.  It is synchronous, so no cancellation lands between the copy and the
            # view that owns (and closes) it.
            try:
                async with lease.gate.hold("exclusive"):
                    clone = lease.snapshot(lease.backend.copy())
            except EnvBusy as busy:
                raise EnvBusyError(self._service.name, busy) from busy
            return self._view(child_task, clone)
        raise ValueError(f"Unknown fork mode {mode!r}")

    async def close(self) -> None:
        lease, self._lease = self._lease, None
        if lease is not None:
            await lease.release(self._service.close_timeout_s)

    def _view(self, task: Task, lease: Lease) -> OrsSession:
        return OrsSession(
            self._service,
            task,
            lease,
            ref=self.ref,
            instruction=task.goal or self.instruction,
            prompt=self.prompt,
            prompt_blocks=self.prompt_blocks,
            rollout_id=self.rollout_id,
        )

    def _require_lease(self) -> Lease:
        if self._lease is None:
            raise RuntimeError(f"ORS session {self.id} is closed")
        return self._lease

    async def _env_tool_names(self, lease: Lease) -> set[str]:
        return {tool.name for tool in await lease.raw_tools()}

    def _is_terminal_tool(self, lease: Lease, name: str, grant: Grant) -> bool:
        """Whether ``name`` is listed as terminal (:meth:`OrsEnvironmentService.tool_specs`' rule).

        An exclusive call is a declared terminal tool.  Otherwise the listing that classified the call decides,
        which also counts a tool learned to end episodes for an env that declares none; the call holds its turn
        on the env, so it reads the listing as it is and never fetches one.
        """

        if grant.call_class == "exclusive":
            return True
        if lease._tools is None:  # the listing failed: the call ran as serial, so no declared terminal tool
            return False
        return any(spec.terminal for spec in self._service.tool_specs(lease._tools, lease) if spec.name == name)

    def _result(self, name: str, output: Any) -> tuple[ToolResult, dict[str, Any]]:
        payload = bridge_payload(output)
        error = tool_error_message(payload)
        metadata = dict(getattr(output, "metadata", None) or {})
        reward = getattr(output, "reward", None)
        finished = bool(getattr(output, "finished", False))
        if self._service.result_format == "bridge_json":
            text = f"Error executing tool {name}: {error}" if error else json.dumps(payload, default=_json_default)
            content = [{"type": "text", "text": text}]
        else:
            content = sdk.content_blocks(getattr(output, "blocks", None) or [])
        result = ToolResult(content, is_error=error is not None, finished=finished, reward=reward, metadata=metadata)
        return result, payload

    def _error_result(
        self, name: str, message: str, metadata: dict[str, Any], *, bridge_text: str | None = None
    ) -> ToolResult:
        """An error result: ``message`` in blocks mode; in bridge_json mode the bridge's text,
        ``Error executing tool <name>: <bridge_text or message>`` (the bridge shows an exception's plain text)."""

        if self._service.result_format == "bridge_json":
            text = f"Error executing tool {name}: {message if bridge_text is None else bridge_text}"
        else:
            text = message
        return ToolResult(content=[{"type": "text", "text": text}], is_error=True, metadata=metadata)

    def _bridge_error(self, lease: Lease, error: str) -> ToolResult:
        """The bridge's reply to a call it refuses: a normal result in bridge_json mode, an error result otherwise."""

        payload = {"bridge_error": error, "finished": True, "reward": lease.last_reward}
        return ToolResult(
            content=[{"type": "text", "text": json.dumps(payload)}],
            is_error=self._service.result_format != "bridge_json",
            finished=True,
            reward=lease.last_reward,
            metadata={"bridge_error": error},
        )

    async def _bridge_tool(self, lease: Lease, name: str) -> ToolResult:
        if name == "get_status":
            payload = {
                "env": self._service.reported_name,
                "split": self.ref.split,
                "task_index": self.ref.index,
                "task_name": self.ref.name,
                "turn": lease.turn,
                "finished": lease.finished,
                "last_reward": lease.last_reward,
            }
            return ToolResult.text(json.dumps(payload, sort_keys=True))
        # The bridge's catalog: the env's tools as list_tools(format="openai") gives them.
        tools = sdk.openai_tools(await lease.raw_tools())
        lease.record("task_requested", {"prompt_chars": len(self.prompt), "tool_count": len(tools)})
        payload = {
            "task_name": self.ref.name,
            "task_index": self.ref.index,
            "prompt": self.prompt,
            "environment_tools": tools,
            "policy": completion_policy(tools),
        }
        return ToolResult.text(json.dumps(payload, default=_json_default))


async def _run_env_code(backend: Backend, fn: Callable[..., Any], *args: Any) -> Any:
    """Run an in-process env's hook (``platoon_grade``, ``platoon_fork``) as its backend runs env code."""

    run = getattr(backend, "run", None)
    return await (run(fn, *args) if run is not None else sdk.run_user_callable(fn, *args))


def _terminal_outcome(result: ToolResult, tool: str, metadata: dict[str, Any]) -> Outcome:
    metadata = {**metadata, "terminal_tool": tool}
    if result.metadata:
        metadata["env_metadata"] = result.metadata
    if result.metadata.get("invalid") is True:
        reason = result.metadata.get("reason")
        return Outcome.infra(
            "env_invalid",
            status="invalid",
            reason=reason if isinstance(reason, str) else f"{tool} declared the grade invalid",
            metadata=metadata,
        )
    if _is_number(result.reward) and math.isfinite(result.reward):  # type: ignore[arg-type]
        return Outcome(status="scored", rewards={"reward": float(result.reward)}, metadata=metadata)  # type: ignore[arg-type]
    return Outcome.infra(
        "no_reward", status="invalid", reason=f"{tool} ended the episode without a reward", metadata=metadata
    )


def _graded_outcome(value: Any, metadata: dict[str, Any]) -> Outcome:
    """Map a ``platoon_grade()`` return: an Outcome, a number, a ToolOutput, or ``{"reward", "invalid", ...}``."""

    if isinstance(value, Outcome):
        return value
    metadata = {**metadata, "graded_by": "platoon_grade"}
    if _is_number(value):
        return Outcome(status="scored", rewards={"reward": float(value)}, metadata=metadata)
    if isinstance(value, Mapping) or hasattr(value, "reward"):
        get = value.get if isinstance(value, Mapping) else lambda key, default=None: getattr(value, key, default)
        env_metadata = dict(get("metadata", None) or {})
        if get("invalid", False) is True:
            env_metadata["invalid"] = True
        result = ToolResult(content=[], finished=True, reward=get("reward", None), metadata=env_metadata)
        return _terminal_outcome(result, "platoon_grade", metadata)
    raise TypeError(f"platoon_grade returned {type(value).__name__}; expected a number, a mapping or an Outcome")


def bridge_payload(output: Any) -> dict[str, Any]:
    """A ``ToolOutput`` as the bridge serializes it for the model."""

    texts = []
    for block in getattr(output, "blocks", None) or []:
        text = getattr(block, "text", None)
        texts.append(text if isinstance(text, str) else str(block))
    data = getattr(output, "data", None)
    metadata = getattr(output, "metadata", None)
    payload: dict[str, Any] = {"finished": bool(getattr(output, "finished", False)), "reward": output.reward}
    if texts:
        payload["text"] = "\n".join(texts)
        payload["blocks"] = texts
    if data is not None:
        payload["data"] = data
    if metadata is not None:
        payload["metadata"] = metadata
    if not texts and data is None:
        payload["raw"] = repr(output)
    return payload


def tool_error_message(payload: Mapping[str, Any]) -> str | None:
    """An env-declared tool error (metadata ``openhands.dev/tool-error``), never parsed from prose."""

    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    marker = metadata.get(sdk.TOOL_ERROR_META_KEY)
    if marker is not True and not isinstance(marker, Mapping):
        return None
    if isinstance(marker, Mapping) and isinstance(marker.get("message"), str) and marker["message"].strip():
        return marker["message"].strip()
    text = payload.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return "The environment tool reported an error."


def completion_policy(tools: list[dict[str, Any]]) -> str:
    """The bridge's completion policy (``get_task``'s ``policy``), naming the env's terminal tool."""

    names = {str(tool.get("name")) for tool in tools if tool.get("name")}
    if "submit_answer" in names:
        terminal_instruction = "call `submit_answer`"
    elif "claim_done" in names:
        terminal_instruction = "call `claim_done`"
    else:
        terminal_instruction = "call the terminal environment tool identified by the task prompt"
    return (
        "Use the listed environment tools directly by name. If this environment "
        "exposes a catalog/meta tool such as `call_tool`, follow the environment "
        "prompt for when to use it. Completion requires an environment tool call: "
        f"when the task is complete, {terminal_instruction}. Do not end with a "
        "normal assistant message; a text response does not submit the environment "
        "and ends the rollout with zero reward."
    )


def _bridge_tool_specs() -> list[ToolSpec]:
    empty = {"type": "object", "properties": {}}
    return [
        ToolSpec("get_task", "Return the task prompt and environment tool catalog.", dict(empty), read_only=True),
        ToolSpec(
            "get_status",
            "Return bridge state including turn count, reward, and finished flag.",
            dict(empty),
            read_only=True,
        ),
    ]


def _failure(exc: BaseException, *, remote: bool) -> tuple[str, bool, bool]:
    """``(kind, infrastructure, ends the session)`` for a call that raised (the module docstring has the rule).

    The kind is the result's ``error_reason``.  Remotely :func:`sdk.error_kind` classifies the failure: ``dead``
    and ``transport`` are infrastructure.  A tool that raised (in process, or ``tool_raised`` remotely) is
    infrastructure, kind ``env_infra``, when it raised one of the SDK's own session or transport errors
    (:func:`sdk.infra_error`), and the policy's ``tool_raised`` otherwise; remotely, 0.1.163's ``ToolFailed`` also
    ends the session either way.
    """

    kind = sdk.error_kind(exc) if remote else "tool_raised"
    if kind == "tool_raised" and sdk.infra_error(exc, remote=remote):
        kind = "env_infra"
    infra = kind in ("dead", "transport", "env_infra")
    return kind, infra, remote and (kind == "dead" or _kills_session(exc))


def _kills_session(exc: BaseException) -> bool:
    # 0.1.163 marks a session dead when its tool raised (ToolFailed); 0.1.125 keeps it alive.
    return "ToolFailed" in {cls.__name__ for cls in type(exc).__mro__}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")[:80] or "value"


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    return repr(value)
