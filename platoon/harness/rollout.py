"""``harness_rollout``: the ``RolloutFn`` that runs a harness against an environment session.

One rollout opens the env session for the task, serves this process's policy
gateway over HTTP, opens the root agent, runs the harness, grades, and builds the
trajectory collection from the agent records. All of it is closed again on every
path, including cancellation. A rollout whose outcome is not scored (an invalid
or failed measurement, a harness crash or timeout) yields trajectories marked
untrainable, which the workflow keeps out of training and out of the group
baseline instead of training them as reward 0.

The task's ``EnvironmentRoute`` (attached by the Auto task loader) names the
service and the harness drawn for the task:

* a **function harness** (``@harness``) runs inline through the route's episode
  program (``route.program``, default ``single``): the program runs the root
  agent and grades, the collection is built with each agent's own outcome, and
  the program's finalizers run on it before it is returned. A program that
  raises is an ``error`` outcome, never a score.
* a **TOML spec** runs through ``HarnessRunner`` where the env wants it, with
  the placement and built-in tool settings.  Before the session opens, the
  rollout works out what the spec needs from the task sandbox
  (``SessionNeeds``): the gateway's Unix socket, which a sandbox without a
  network reaches through a route; the spec's named toolchain, staged on this
  node and mounted read-only at its build path; and the spec's
  ``[resources] memory_mb``.  For a spec with ``[budget] enforce = "gateway"``,
  the gateway caps the root's policy calls at the step budget
  (``AgentSession.call_cap``), and a run that the cap ended and that then exited
  cleanly is graded with reason ``budget``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import shutil
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from platoon.envs.service import Outcome, SessionNeeds, get_service, open_session, takes_needs
from platoon.harness.function import FunctionHarness
from platoon.harness.registry import get_harness
from platoon.harness.runner import (
    AgentEndpoint,
    HarnessResult,
    HarnessRunner,
    HarnessSetupError,
    Placement,
    resolve_placement,
)
from platoon.harness.spec import HarnessSpec, HarnessSpecError
from platoon.harness.toolchains import PreparedToolchain, ToolchainError, is_named, toolchain_problem
from platoon.harness.toolchains import prepare as prepare_toolchain
from platoon.sandbox.routes import GATEWAY
from platoon.task import Task
from platoon.train.components import EnvironmentRoute

if TYPE_CHECKING:
    from platoon.config_defs import RolloutConfig
    from platoon.envs.service import EnvironmentService, EnvSession
    from platoon.gateway import AgentSession, ModelGateway
    from platoon.gateway.core import SamplingOverrides
    from platoon.trajectory import TrajectoryCollection

logger = logging.getLogger(__name__)

# Harnesses reach the gateway on loopback: locally, and from task sandboxes that share the host network.
GATEWAY_HOST = "127.0.0.1"
HARNESS_MISC_KEY = "harness"


AgentT = TypeVar("AgentT", bound=AgentEndpoint)


class RolloutGateway(Protocol[AgentT]):
    """What one harness rollout needs from the model gateway.

    A gateway may also have ``async serve_unix(path, *, rollout_id)``: listen on the Unix socket ``path`` too,
    for that rollout's agents only.  The rollout then gives task sandboxes the socket as their gateway upstream,
    which a sandbox without a network reaches through a route (:mod:`platoon.sandbox.routes`).
    """

    async def start(self) -> str:
        """Start serving and return the base URL (``http://host:port``) harnesses use."""
        ...

    def open_agent(self, *, rollout_id: str, trajectory_id: str, sampling: SamplingOverrides | None = None) -> AgentT:
        """Open the root agent; ``sampling`` is the rollout's default sampling for the calls of its whole tree."""
        ...

    def serve_tools(self, agent: AgentT, session: EnvSession) -> str | None:
        """Serve ``session``'s tools to ``agent`` over MCP and return the endpoint (``{mcp_url}``); None without
        a tool plane."""
        ...

    def collect(self, root: AgentT, *, task: Task, outcome: Outcome) -> TrajectoryCollection:
        """Close the rollout's agents (the root and any children it minted) and build its trajectories."""
        ...

    async def stop(self) -> None:
        """Stop serving and close any agent the rollout left open; idempotent."""
        ...


@runtime_checkable
class _PolicyOnlyAgent(Protocol):
    """A gateway agent whose call kinds the gateway can limit, for a key that code the policy runs can read
    (``AgentSession.policy_only``)."""

    policy_only: bool


@runtime_checkable
class _CallCappedAgent(Protocol):
    """A gateway agent whose policy calls the gateway caps (``AgentSession.call_cap``)."""

    call_cap: int | None


class EmbeddedGateway:
    """A :class:`RolloutGateway` over a ``platoon.gateway.ModelGateway``, served for one rollout.

    Its server takes only the keys of the agents of the rollouts it opened agents for (``GatewayServer.serve_rollout``),
    on its loopback TCP port as on its Unix socket, and no key before then: a key of another rollout, which the
    shared ``ModelGateway`` knows too, gets 401 here.
    """

    def __init__(self, gateway: ModelGateway) -> None:
        from platoon.gateway import GatewayServer

        self._gateway = gateway
        self._server = GatewayServer(gateway, rollout_ids=())
        self._rollout_ids: set[str] = set()

    @classmethod
    def for_this_process(cls) -> EmbeddedGateway:
        """This process's default policy gateway (``TinkerLLM.set_as_default``)."""

        from platoon.gateway import default_gateway

        return cls(default_gateway())

    async def start(self) -> str:
        return await self._server.start(host=GATEWAY_HOST)

    async def serve_unix(self, path: str, *, rollout_id: str) -> None:
        """Also serve on the Unix socket ``path``, for the agents of ``rollout_id`` only."""

        await self._server.start_unix(path, rollout_id=rollout_id)

    def open_agent(
        self, *, rollout_id: str, trajectory_id: str, sampling: SamplingOverrides | None = None
    ) -> AgentSession:
        # Record into the rollout's interaction map and timeline, which the rollout worker converts and reports.
        from platoon.train.tinker.proxy import proxy_interactions
        from platoon.train.tinker.rollout_metrics import current_timeline

        self._rollout_ids.add(rollout_id)
        self._server.serve_rollout(rollout_id)
        return self._gateway.open_agent(
            rollout_id=rollout_id,
            trajectory_id=trajectory_id,
            interactions=proxy_interactions.get({}),
            timeline=current_timeline(),
            sampling_defaults=sampling,
        )

    def serve_tools(self, agent: AgentSession, session: EnvSession) -> str:
        from platoon.gateway.mcp import bind_agent_tools, mcp_url

        bind_agent_tools(self._gateway, agent, session)
        return mcp_url(self._server.base_url)

    def collect(self, root: AgentSession, *, task: Task, outcome: Outcome) -> TrajectoryCollection:
        from platoon.gateway import build_trajectory_collection

        return build_trajectory_collection(self._gateway.close_rollout(root.rollout_id), task=task, outcome=outcome)

    @property
    def model_gateway(self) -> ModelGateway:
        """The in-process gateway function harnesses chat through."""
        return self._gateway

    async def stop(self) -> None:
        await self._server.stop()
        for rollout_id in self._rollout_ids:
            self._gateway.close_rollout(rollout_id)


def rollout_sampling(config: RolloutConfig) -> SamplingOverrides:
    """The rollout's default sampling: the ``config.inference_params`` the config sets, which every agent's calls
    get unless they set their own (``ctx.chat(**sampling)``, or the parameters of an HTTP request); with
    ``config.sampling_precedence: config`` they win over a policy call's own instead.  A parameter the config
    leaves unset keeps the gateway's default."""

    from platoon.gateway.core import SamplingOverrides

    return SamplingOverrides.from_inference_params(
        getattr(config, "inference_params", None), precedence=getattr(config, "sampling_precedence", "request")
    )


async def harness_rollout(task: Task, config: RolloutConfig) -> dict[str, Any] | TrajectoryCollection:
    """Run the task's harness through this process's policy gateway (see the module docstring)."""

    return await run_harness_rollout(task, config, gateway=EmbeddedGateway.for_this_process())


async def run_harness_rollout(
    task: Task,
    config: RolloutConfig,
    *,
    gateway: RolloutGateway[AgentT],
    runner: HarnessRunner | None = None,
) -> dict[str, Any] | TrajectoryCollection:
    """``harness_rollout`` with its gateway (and optionally its runner) supplied."""

    route = EnvironmentRoute.of_task(task)
    if route is None or route.service is None or route.harness is None:
        raise ValueError(
            f"Task {task.id!r} has no environment route with a service and a harness; "
            "configure the environment's `service` and `harness` and load tasks with AutoTaskLoader"
        )
    harness = get_harness(route.harness)
    service = get_service(route.service, **route.service_kwargs)
    if route.resolve_task:
        if task.id is None:
            raise ValueError("A task the service resolves needs its task id")
        resolved = await service.get_task(task.id)
        if task.max_steps is not None:
            resolved.max_steps = task.max_steps
        task = route.attach(resolved)
    rollout_id = f"{re.sub(r'[^A-Za-z0-9._-]', '-', str(task.id))[-80:]}-{uuid.uuid4().hex[:12]}"
    if isinstance(harness, FunctionHarness):
        return await _run_function_rollout(
            task, config, route=route, harness=harness, service=service, gateway=gateway, rollout_id=rollout_id
        )
    spec = harness
    runner = runner or HarnessRunner(logs_dir=Path(config.output_dir) / "harness")

    async with contextlib.AsyncExitStack() as cleanup:
        setup = await HarnessSetup.prepare(spec, route=route, service=service, gateway=gateway)
        cleanup.callback(setup.close)
        session = await open_session(service, task, rollout_id=rollout_id, needs=setup.needs)
        cleanup.push_async_callback(_quietly, session.close, f"close env session of {rollout_id}")
        cleanup.push_async_callback(_quietly, gateway.stop, f"stop gateway of {rollout_id}")
        base_url = await gateway.start()
        if setup.socket_path is not None:  # only for a gateway that serves Unix sockets (HarnessSetup.prepare)
            serve_unix = getattr(gateway, "serve_unix")
            await serve_unix(setup.socket_path, rollout_id=rollout_id)
        root = gateway.open_agent(
            rollout_id=rollout_id, trajectory_id=str(uuid.uuid4()), sampling=rollout_sampling(config)
        )
        set_judges(root, config)
        # MCP tool plane: a spec with [mcp] enabled gets the env session's tools at the root's MCP endpoint.
        mcp_url = gateway.serve_tools(root, session) if spec.mcp.enabled else None
        outcome, result, placement = await _run_and_grade(
            runner, spec, session, root, base_url, route, config, mcp_url=mcp_url, setup=setup
        )
        collection = gateway.collect(root, task=_recorded_task(task, session), outcome=outcome)

    root_trajectory = next(iter(collection.trajectories.values()))
    root_trajectory.misc[HARNESS_MISC_KEY] = {
        "spec": spec.ref,
        "placement": placement,
        "exit_code": result.exit_code if result else None,
        "timed_out": result.timed_out if result else False,
        "logs_path": result.logs_path if result else None,
    }
    if result is not None and result.final_message is not None:
        root_trajectory.finish_message = result.final_message
    if config.verbose:
        logger.info("harness rollout %s (%s, %s): %s", rollout_id, spec.ref, placement, outcome.status)
    return collection.to_dict() if config.return_dict else collection


async def _run_function_rollout(
    task: Task,
    config: RolloutConfig,
    *,
    route: EnvironmentRoute,
    harness: FunctionHarness,
    service: EnvironmentService,
    gateway: RolloutGateway[Any],
    rollout_id: str,
) -> dict[str, Any] | TrajectoryCollection:
    """Run a function harness inline through the route's episode program."""

    from platoon.gateway.trace import attach_judge_audit, build_trajectory_collection
    from platoon.harness.inline import InlineRuntime, recorded_task
    from platoon.programs import Episode, get_program

    model_gateway = getattr(gateway, "model_gateway", None)
    if model_gateway is None:
        raise TypeError(f"Function harness {harness.ref} needs an in-process model gateway (EmbeddedGateway)")
    async with contextlib.AsyncExitStack() as cleanup:
        session = await service.open(task, rollout_id=rollout_id)
        cleanup.push_async_callback(_quietly, session.close, f"close env session of {rollout_id}")
        cleanup.push_async_callback(_quietly, gateway.stop, f"stop gateway of {rollout_id}")
        # The agents work on the task with the goal its session runs: an ORS task gets its goal, the env's prompt, only
        # when its session opens, and a subagent's task (a fork of its parent's) records its parent's goal.
        task = recorded_task(task, session)
        base_url = await gateway.start()
        root = gateway.open_agent(
            rollout_id=rollout_id, trajectory_id=str(uuid.uuid4()), sampling=rollout_sampling(config)
        )
        root.max_steps = task.max_steps if task.max_steps is not None else config.max_steps
        set_judges(root, config)
        runtime = InlineRuntime(
            gateway=model_gateway,
            base_url=base_url,
            agents=route.agents,
            verifiers_enabled=bool(config.enable_subagent_verifiers),
            on_context_overflow=route.on_context_overflow,
            on_harness_exit=route.on_harness_exit,
        )
        episode = Episode(task=task, session=session, config=config, route=route, runtime=runtime, root=root)
        try:
            outcome = await get_program(route.program)(episode)
            if not isinstance(outcome, Outcome):
                raise TypeError(f"Program {route.program!r} returned {type(outcome).__name__}, not an Outcome")
        except Exception as exc:
            logger.exception("episode program %s failed", route.program)
            outcome = Outcome.infra(f"unexpected:{type(exc).__name__}", reason=f"program {route.program}: {exc}")
        root.outcome = outcome
        agents = model_gateway.agents(rollout_id)
        traced_task = recorded_task(task, session)  # after the run: a goal the harness set on ctx.task stays
        collection = build_trajectory_collection(agents, task=traced_task, outcome=outcome)
        try:
            await episode.finalize(collection)
        except Exception as exc:
            logger.exception("a finalizer of program %s failed", route.program)
            outcome = Outcome.infra(f"finalizer:{type(exc).__name__}", reason=str(exc))
            collection = build_trajectory_collection(agents, task=traced_task, outcome=outcome)
        attach_judge_audit(collection, model_gateway.agents(rollout_id))
        model_gateway.close_rollout(rollout_id)

    root_trajectory = collection.trajectories[root.trajectory_id]
    root_trajectory.misc[HARNESS_MISC_KEY] = {
        "spec": harness.ref,
        "kind": "function",
        "runtime": harness.runtime,
        "executes_model_code": harness.executes_model_code,
        "program": route.program,
    }
    if config.verbose:
        logger.info("harness rollout %s (%s, %s): %s", rollout_id, harness.ref, route.program, outcome.status)
    return collection.to_dict() if config.return_dict else collection


@dataclass
class HarnessSetup:
    """What a spec's rollout prepares before its session opens: the session's needs, the gateway's socket
    directory, and the spec's named toolchain, staged (or why it could not be)."""

    needs: SessionNeeds | None = None
    socket_dir: Path | None = None
    toolchain: PreparedToolchain | None = None
    error: str | None = None  # a named toolchain that could not be prepared: the rollout is a harness_setup error

    @property
    def socket_path(self) -> str | None:
        """The gateway's Unix socket for the rollout, or None when the sandbox gets no gateway upstream."""

        from platoon.gateway.server import SOCKET_NAME

        return None if self.socket_dir is None else str(self.socket_dir / SOCKET_NAME)

    @classmethod
    async def prepare(
        cls, spec: HarnessSpec, *, route: EnvironmentRoute, service: EnvironmentService, gateway: Any
    ) -> HarnessSetup:
        """Stage a named toolchain (from the service's ``toolchain_root``, else the toolchain root that
        :func:`platoon.harness.toolchains.toolchain_root` finds) and, unless the harness runs on the rollout host,
        work out the session's needs: the gateway's Unix socket as an upstream (when the gateway can serve one and
        the service takes needs), the toolchain's mount and the spec's memory."""

        setup = cls()
        name = spec.install.toolchain
        if name is not None and is_named(name):
            root = getattr(service, "toolchain_root", None)
            try:
                setup.toolchain = await asyncio.to_thread(prepare_toolchain, name, root=root)
            except ToolchainError as exc:  # with the command that builds it, when it is not built
                setup.error = f"harness {spec.ref}: {toolchain_problem(spec, root=root) or exc}"
        if route.placement == "local":
            return setup
        upstreams = {}
        if callable(getattr(gateway, "serve_unix", None)) and takes_needs(service):
            from platoon.gateway.server import socket_directory

            setup.socket_dir = socket_directory()
            upstreams[GATEWAY] = f"unix://{setup.socket_path}"
        setup.needs = SessionNeeds(
            upstreams=upstreams,
            mounts=(setup.toolchain.mount(),) if setup.toolchain is not None else (),
            extra_memory_mb=spec.resources.memory_mb or 0,
        )
        return setup

    def toolchain_path(self, placement: Placement) -> str | None:
        """Where a launch at ``placement`` finds the named toolchain; None for a spec without one."""

        if self.toolchain is None:
            return None
        return self.toolchain.sandbox_path if placement == "task_sandbox" else self.toolchain.host_path

    def close(self) -> None:
        if self.socket_dir is not None:
            shutil.rmtree(self.socket_dir, ignore_errors=True)


async def _run_and_grade(
    runner: HarnessRunner,
    spec: HarnessSpec,
    session: EnvSession,
    root: AgentEndpoint,
    base_url: str,
    route: EnvironmentRoute,
    config: RolloutConfig,
    *,
    mcp_url: str | None = None,
    setup: HarnessSetup | None = None,
) -> tuple[Outcome, HarnessResult | None, str]:
    """Run the harness and grade the session, failing closed: nothing unclassified is ever scored.

    The harness runs for at most its spec's ``timeout_s``, the rollout's ``timeout`` and the task's own agent
    timeout (``session.agent_timeout_s``, where the env has one).  A timeout or crash then follows the route's
    ``on_harness_exit`` (:func:`platoon.harness.inline.failure_outcome`).  A harness in the task sandbox, where code
    the policy runs can read the agent's key, is limited in the kinds of call it makes (``AgentSession.policy_only``,
    on a gateway whose agents have it): the gateway refuses its ``judge`` calls, which never train, and under a
    gateway step budget its ``aux`` calls too, which the budget does not count.
    """

    from platoon.harness.inline import agent_timeout, crash_outcome

    setup = setup or HarnessSetup()
    placement = route.placement
    max_steps = session.task.max_steps if session.task.max_steps is not None else config.max_steps
    try:
        placement = resolve_placement(route.placement, session)
        if setup.error is not None:
            raise HarnessSetupError(setup.error)
        if placement == "task_sandbox" and isinstance(root, _PolicyOnlyAgent):
            root.policy_only = True
        _cap_calls(spec, root, max_steps)
        result = await runner.run(
            spec,
            session=session,
            agent=root,
            base_url=base_url,
            placement=placement,
            builtins=route.builtins,
            timeout_s=agent_timeout(config, session),
            max_steps=max_steps,
            mcp_url=mcp_url,
            toolchain=setup.toolchain_path(placement),
            options=route.harness_options,
        )
    except HarnessSetupError as exc:
        return Outcome.infra("harness_setup", reason=str(exc)), None, placement
    except HarnessSpecError as exc:  # the spec cannot launch with this run's settings: a config error, not a crash
        logger.error("harness %s cannot launch: %s", spec.ref, exc)
        return Outcome.infra("harness_config", reason=str(exc)), None, placement
    except Exception as exc:
        logger.exception("harness %s failed to run", spec.ref)
        return Outcome.infra(f"unexpected:{type(exc).__name__}", reason=str(exc)), None, placement
    if result.timed_out:
        if route.on_harness_exit.timeout == "grade":
            return await _grade(session, reason="agent_timeout"), result, placement
        return Outcome(status="timeout", reason="agent_timeout"), result, placement
    if spec.exit_status(result.exit_code) == "completed" and getattr(root, "budget_exhausted", False):
        # The gateway ended the run at the step budget: a graded policy outcome, like a harness's own step limit.
        return await _grade(session, reason="budget"), result, placement
    if spec.exit_status(result.exit_code) == "error":
        reason = f"{spec.ref} exited with code {result.exit_code}"
        if getattr(root, "context_overflows", 0):
            # The harness died because the policy filled its context window: a policy failure, not an infra one.
            if route.on_context_overflow == "exclude":
                return Outcome.infra("context_overflow", reason=reason), result, placement
            return await _grade(session, reason="context_overflow"), result, placement
        failed, grade_reason = crash_outcome(
            route.on_harness_exit, reason=reason, gateway_errors=getattr(root, "gateway_errors", 0)
        )
        if failed is not None:
            return failed, result, placement
        return await _grade(session, reason=grade_reason), result, placement
    return await _grade(session), result, placement


def _cap_calls(spec: HarnessSpec, root: AgentEndpoint, max_steps: int | None) -> None:
    """For a spec with ``[budget] enforce = "gateway"``, cap the root's policy calls at the gateway to the step budget:
    the run's ``max_steps``, else the spec's.  Neither is a ``HarnessSpecError`` (a ``harness_config`` outcome); a
    gateway without call caps is a setup error, never an unbounded run."""

    if spec.budget.enforce != "gateway":
        return
    steps = max_steps if max_steps is not None else spec.max_steps
    if steps is None:
        raise HarnessSpecError(
            f"harness {spec.ref} has the gateway enforce its step budget, but the run sets no step budget "
            "(rollout_config.max_steps or the task's max_steps) and the spec no max_steps default"
        )
    if not isinstance(root, _CallCappedAgent):
        raise HarnessSetupError(f"harness {spec.ref} needs a gateway that enforces step budgets; this one does not")
    root.call_cap = steps
    misc = getattr(root, "misc", None)
    if isinstance(misc, dict):  # the trajectory records the budget, and the gateway's stops when it makes any
        from platoon.gateway.core import BUDGET_MISC_KEY

        misc[BUDGET_MISC_KEY] = {"enforce": "gateway", "max_steps": steps, "stops": 0}


def _recorded_task(task: Task, session: EnvSession) -> Task:
    """``platoon.harness.inline.recorded_task``, imported only when a rollout needs it: the inline module pulls in
    the gateway and its tokenizer stack."""

    from platoon.harness.inline import recorded_task

    return recorded_task(task, session)


async def _grade(session: EnvSession, *, reason: str | None = None) -> Outcome:
    """Grade the session, failing closed: a grader that raises is an infra error, never a score.  ``reason`` is
    the outcome's reason when it is scored (``platoon.harness.inline.grade_session``)."""

    from platoon.harness.inline import grade_session

    return await grade_session(session, reason=reason)


async def _quietly(close: Callable[[], Awaitable[None]], what: str) -> None:
    """Cleanup must not discard a finished rollout: a failure to release is logged, not raised."""

    try:
        await close()
    except Exception:
        logger.warning("failed to %s", what, exc_info=True)


def set_judges(root: Any, config: RolloutConfig) -> None:
    """Give the root agent the rollout's ``judges`` setting, which every agent of its tree inherits
    (``AgentSession.judges``: :mod:`platoon.utils.judges`)."""

    if hasattr(root, "judges"):
        root.judges = bool(getattr(config, "judges", True))
