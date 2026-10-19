"""Inline function harnesses: the :class:`~platoon.harness.agent.AgentContext` of one agent, run in process.

A function harness runs in the rollout worker's event loop, next to the gateway.  :class:`InlineRuntime` is
what every agent of one rollout shares (the gateway, the embedded HTTP server's URL, the step-budget tracker,
the ``agents`` settings), and :class:`InlineAgentContext` is one agent's view of it:

* ``chat`` goes through the same render -> sample -> parse -> record path as HTTP calls
  (``ModelGateway.chat``), so a prompt that overflows the context window raises
  ``ContextWindowExceededError`` and counts in ``AgentSession.context_overflows`` either way.
* ``endpoint`` is the embedded server's URL scoped to the agent (``/s/<secret>/v1``), for harnesses with their
  own HTTP client; ``mcp`` is the agent's tool plane endpoint (:mod:`platoon.gateway.mcp`).
* ``list_tools``/``call_tool`` go through the tool plane (``list_agent_tools``/``call_agent_tool``), filtered
  by the agent's tool access: ``no_terminal`` hides tools whose ``ToolSpec.terminal`` is set, ``read_only``
  shows only tools whose ``ToolSpec.read_only`` is set.  Inline and MCP calls therefore share the listing
  transform, the env session's call order (:mod:`platoon.envs.gate`) and one record format
  (``AgentSession.tool_calls``).
* ``annotate``/``annotate_step`` land in the agent's trajectory misc and the last call's step misc.

:func:`run_agent` runs a harness on a context and never raises for a harness failure: it returns a
``AgentResult`` with status ``error``, ``timeout`` or ``limit``, and re-raises only cancellation.  An agent's run
includes the children it spawned: when its harness ends with children still running (say an un-awaited
``asyncio.gather`` of spawns), the agent waits for them to finish and be graded, within its own timeout.  Those
children are cancelled only when the agent itself is cancelled or runs out of time, which includes a harness that
reports a timeout of its own (``AgentResult("timeout")``, say CodeAct's cell timeout).  (While the harness runs,
cancelling a spawn call cancels its child, unless the harness called ``agents.keep_children()``:
:mod:`platoon.harness.agents`.)
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from platoon.gateway import mcp as tool_plane
from platoon.gateway.core import ChatRequest
from platoon.harness.agent import AgentResult, ChatReply, McpEndpoint, ModelEndpoint
from platoon.harness.budget import AgentBudget, make_tracker
from platoon.utils.critical_path import EXECUTION_SPAN_KEY

if TYPE_CHECKING:
    from platoon.envs.service import EnvSession, Outcome, ToolResult, ToolSpec
    from platoon.gateway.core import AgentSession, ModelGateway
    from platoon.harness.function import FunctionHarness
    from platoon.harness.verify import SubagentBehaviorJudge
    from platoon.task import Task
    from platoon.train.components import AgentTreeConfig, HarnessExitPolicy

logger = logging.getLogger(__name__)

# How long an agent waits for the children it cancels (when it is cancelled or runs out of time) to unwind.
CHILD_CANCEL_GRACE_SECONDS = 10.0

ToolTransform = Callable[..., Any]


class InlineRuntime:
    """What every agent of one rollout shares.

    Attributes:
        gateway: The policy gateway every agent records into.
        base_url: The embedded HTTP server (``http://host:port``) that serves ``ctx.endpoint`` and ``ctx.mcp``.
        agents: The environment's ``agents`` settings (subagents, budgets, child grading).
        tracker: The step-budget tracker for ``agents.budget``, keyed by agent id.
        verifiers_enabled: False switches verifier children off (eval, inference).
        on_context_overflow: How an agent that crashed after a context overflow is graded (``grade``/``exclude``).
        on_harness_exit: How an agent whose harness ran out of time or crashed is graded
            (``platoon.train.components.HarnessExitPolicy``).
        behavior_judge: The optional behavior judge of verifier grading (``verify.SubagentBehaviorJudge``).
        tool_transform: Passed to ``platoon.gateway.mcp.bind_agent_tools`` for every agent.
    """

    def __init__(
        self,
        *,
        gateway: ModelGateway,
        base_url: str | None,
        agents: AgentTreeConfig | None = None,
        verifiers_enabled: bool = True,
        on_context_overflow: str = "grade",
        on_harness_exit: HarnessExitPolicy | None = None,
        behavior_judge: SubagentBehaviorJudge | None = None,
        tool_transform: ToolTransform | None = None,
    ) -> None:
        from platoon.train.components import AgentTreeConfig, HarnessExitPolicy

        self.gateway = gateway
        self.base_url = base_url
        self.agents = AgentTreeConfig.coerce(agents)
        self.tracker = make_tracker(self.agents.budget, max_depth=self.agents.max_depth)
        self.verifiers_enabled = verifiers_enabled
        self.on_context_overflow = on_context_overflow
        self.on_harness_exit = HarnessExitPolicy.coerce(on_harness_exit)
        self.behavior_judge = behavior_judge
        self.tool_transform = tool_transform

    def endpoint(self, agent: AgentSession) -> ModelEndpoint:
        if self.base_url is None:
            raise RuntimeError("This rollout serves no HTTP endpoint (the gateway server was not started)")
        return ModelEndpoint(
            base_url=f"{self.base_url}/s/{agent.secret}/v1", api_key=agent.secret, model=self.gateway.model_name
        )

    def bind_tools(self, agent: AgentSession, session: EnvSession) -> McpEndpoint | None:
        """Bind the agent's tools on the gateway's tool plane (``platoon.gateway.mcp``), and return the agent's MCP
        endpoint when the gateway is served over HTTP."""
        tool_plane.bind_agent_tools(self.gateway, agent, session, agent.tool_access, transform=self.tool_transform)
        if self.base_url is None:
            return None
        return McpEndpoint(
            url=tool_plane.mcp_url(self.base_url, agent.secret), headers={"Authorization": f"Bearer {agent.secret}"}
        )


class InlineAgentContext:
    """The :class:`~platoon.harness.agent.AgentContext` of one agent run inline."""

    def __init__(
        self,
        runtime: InlineRuntime,
        agent: AgentSession,
        *,
        task: Task,
        session: EnvSession,
        harness: FunctionHarness,
        instruction: str,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        from platoon.harness.agents import Subagents

        self.runtime = runtime
        self.agent = agent
        self.harness = harness
        self.agent_id = agent.agent_id
        self.trajectory_id = agent.trajectory_id
        self.parent_agent_id = agent.parent_agent_id
        self.depth = agent.depth
        self.role = agent.role
        self.task = task
        self.session = session
        self.instruction = instruction
        self.max_steps = agent.max_steps
        self.access = agent.tool_access
        self.options: Mapping[str, Any] = MappingProxyType(dict(options or {}))
        agent.task = task
        runtime.tracker.add_agent(agent)
        self.budget = AgentBudget(runtime.tracker, agent.agent_id)
        self.mcp = runtime.bind_tools(agent, session)
        self.agents = Subagents(self)
        self.keeps_children = False  # set by agents.keep_children(): cancelling a spawn call keeps its child
        self.finished = False  # set when run_agent is done with the agent's children: spawns are refused from then
        self._children: set[asyncio.Task[Any]] = set()

    @property
    def endpoint(self) -> ModelEndpoint:
        return self.runtime.endpoint(self.agent)

    # ------------------------------------------------------------------ model

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        kind: str = "policy",
        **sampling: Any,
    ) -> ChatReply:
        params = dict(sampling)
        if tools:
            params["tools"] = [dict(tool) for tool in tools]
        request = ChatRequest.from_params([dict(message) for message in messages], params)
        response = await self.runtime.gateway.chat(self.agent, request, kind=kind)
        return ChatReply(call_id=response.call_id, message=response.message(), finish_reason=response.finish_reason)

    # ------------------------------------------------------------------ tools

    async def list_tools(self) -> list[ToolSpec]:
        return await tool_plane.list_agent_tools(self.agent)

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        """Call a visible tool and record it on the agent; a tool the agent does not see is an error result that
        neither reaches the env nor is recorded."""
        from platoon.envs.service import ToolResult

        refused = ToolResult.text(
            f"Tool {name!r} is not available to this agent (tool access: {self.access}).", is_error=True
        )
        try:
            return await tool_plane.call_agent_tool(self.agent, name, dict(arguments))
        except tool_plane.UnknownToolError:
            return refused

    # ------------------------------------------------------------------ annotations

    def annotate(self, **misc: Any) -> None:
        self.agent.misc.update(misc)

    def annotate_step(self, **misc: Any) -> None:
        if not self.agent.calls:
            raise RuntimeError("annotate_step needs a recorded model call on this agent")
        self.agent.step_misc.setdefault(self.agent.calls[-1].call_id, {}).update(misc)

    # ------------------------------------------------------------------ children

    def start_child(self, coroutine: Any) -> asyncio.Task[Any]:
        """Run a child's lifecycle (launch, run, grading) as a task that this agent owns.

        The spawn call awaits the task, and cancelling the call cancels it (unless :attr:`keeps_children` is set).
        A call still running when this agent's harness ends keeps it: :func:`run_agent` waits for the task, and
        cancels it only when this agent is cancelled or runs out of time (a timeout its harness reports included).
        Its exception is retrieved (and logged) when it ends, so a child that nobody awaits any more never leaves
        "exception was never retrieved" behind.
        """
        task = asyncio.get_running_loop().create_task(coroutine)
        self._children.add(task)
        task.add_done_callback(self._child_done)
        return task

    def _child_done(self, task: asyncio.Task[Any]) -> None:
        self._children.discard(task)
        if not task.cancelled() and (exc := task.exception()) is not None:
            logger.warning("a subagent of %s failed", self.agent_id, exc_info=exc)

    async def wait_children(self, deadline: float | None = None) -> None:
        """Wait for the children still running, and for any spawned meanwhile, until all have finished or the
        event-loop time ``deadline`` passes; children still running then are left to :meth:`cancel_children`.

        Each check first yields to the loop once, so that spawn calls already scheduled (an un-awaited
        ``asyncio.gather`` of spawns made just before the harness returned) start, and register their children.
        """

        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(0)
            pending = [task for task in self._children if not task.done()]
            if not pending:
                return
            timeout = None if deadline is None else deadline - loop.time()
            if timeout is not None and timeout <= 0:
                logger.warning(
                    "agent %s ran out of time with %d subagent(s) still running; cancelling them",
                    self.agent_id,
                    len(pending),
                )
                return
            await asyncio.wait(pending, timeout=timeout)

    async def cancel_children(self) -> None:
        pending = [task for task in self._children if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            _, still_running = await asyncio.wait(pending, timeout=CHILD_CANCEL_GRACE_SECONDS)
            if still_running:
                logger.warning(
                    "%d child agent(s) of %s did not unwind after cancellation", len(still_running), self.agent_id
                )


def failure_outcome(
    result: AgentResult,
    *,
    context_overflows: int,
    on_context_overflow: str,
    harness_ref: str,
    on_harness_exit: HarnessExitPolicy | None = None,
    gateway_errors: int = 0,
) -> tuple[Outcome | None, str | None]:
    """The failure taxonomy of a finished run: ``(outcome instead of grading, reason to add to a grade)``.

    A cancellation is untrainable.  A timeout is untrainable too, unless the environment's ``on_harness_exit``
    grades timeouts (``reason="agent_timeout"``).  A harness error after the policy filled its context window is a
    policy failure, graded with ``reason="context_overflow"`` (whatever reason the grader gives, see
    :func:`with_reason`), or excluded from training when the environment sets ``on_context_overflow: exclude``.
    Any other harness error follows ``on_harness_exit.crash``: an infrastructure failure (``harness_crash``,
    the default), graded (``reason="harness_crash"``), or graded unless the gateway failed one of the agent's
    calls (``gateway_errors``; then ``harness_crash_gateway``).  ``completed`` and ``limit`` runs are graded.
    """

    from platoon.envs.service import Outcome
    from platoon.train.components import HarnessExitPolicy

    policy = HarnessExitPolicy.coerce(on_harness_exit)
    if result.status == "timeout":
        if policy.timeout == "grade":
            return None, "agent_timeout"
        return Outcome(status="timeout", reason="agent_timeout"), None
    if result.status == "cancelled":
        return Outcome(status="cancelled", reason="agent_cancelled"), None
    if result.status == "error":
        reason = f"{harness_ref} failed: {result.error}"
        if context_overflows:
            if on_context_overflow == "exclude":
                return Outcome.infra("context_overflow", reason=reason), None
            return None, "context_overflow"
        return crash_outcome(policy, reason=reason, gateway_errors=gateway_errors)
    return None, None


def agent_timeout(config: Any, session: EnvSession) -> float | None:
    """The root agent's time limit: the rollout's ``timeout``, capped by the task's own limit when the env session
    has one (``session.agent_timeout_s``, such as a Harbor task's ``[agent].timeout_sec``); None for no limit."""

    limits = [getattr(config, "timeout", None), getattr(session, "agent_timeout_s", None)]
    return min((limit for limit in limits if limit is not None), default=None)


def crash_outcome(policy: HarnessExitPolicy, *, reason: str, gateway_errors: int) -> tuple[Outcome | None, str | None]:
    """A crashed harness under ``on_harness_exit.crash``: ``(outcome instead of grading, reason of the grade)``."""

    from platoon.envs.service import Outcome

    if policy.crash == "grade_if_gateway_clean" and gateway_errors:
        reason = f"{reason}, after the model gateway failed {gateway_errors} of the agent's calls"
        return Outcome.infra("harness_crash_gateway", reason=reason), None
    if policy.crash == "exclude":
        return Outcome.infra("harness_crash", reason=reason), None
    return None, "harness_crash"


async def grade_session(session: EnvSession, *, reason: str | None = None) -> Outcome:
    """Grade ``session``, failing closed: a grader that raises is an infra error, never a score.

    ``reason`` (e.g. ``context_overflow``) is the reason of a scored outcome (:func:`with_reason`).
    """

    from platoon.envs.service import Outcome

    try:
        outcome = await session.grade()
    except Exception as exc:
        logger.exception("grading raised instead of returning an outcome")
        return Outcome.infra(f"unexpected:{type(exc).__name__}", reason=f"grading raised: {exc}")
    return with_reason(outcome, reason)


def recorded_task(task: Task, session: EnvSession) -> Task:
    """``task`` as its rollout's trajectories record it: with the goal its session ran when it has none of its own.

    An ORS task gets its goal, the env's prompt, only when its session opens (``session.instruction``); an empty goal
    counts as none, as it does for the session.  A goal the task carries, or that its harness set on ``ctx.task``,
    stays.
    """

    if task.goal or not session.instruction:
        return task
    return dataclasses.replace(task, goal=session.instruction)


def with_reason(outcome: Outcome, reason: str | None) -> Outcome:
    """``outcome`` reporting ``reason`` (e.g. ``context_overflow``) when it is scored, even if its grader (an env or
    a verifier) gave its own reason (say ``no_submit``).

    The grader's reason is kept in the outcome's metadata under ``grade_reason``
    (``platoon.envs.service.GRADE_REASON_KEY``), which the trajectory's ``misc["outcome"]`` shows.
    """

    from platoon.envs.service import GRADE_REASON_KEY

    if reason is None or outcome.status != "scored" or outcome.reason == reason:
        return outcome
    metadata = dict(outcome.metadata)
    if outcome.reason is not None:
        metadata[GRADE_REASON_KEY] = outcome.reason
    return dataclasses.replace(outcome, reason=reason, metadata=metadata)


async def run_agent(ctx: InlineAgentContext, *, timeout: float | None = None) -> AgentResult:
    """Run ``ctx``'s harness to an :class:`AgentResult`; a failure is a result, only cancellation raises.

    When the harness returns or fails with children still running, the agent waits for them: each finishes within
    its own budget and timeout and is graded as usual.  ``timeout`` bounds the harness and that wait together.
    Children still running when it passes, or when the agent is cancelled, are cancelled; the harness's own result
    stands when only the wait ran out of time.  A harness that reports a timeout itself (``AgentResult("timeout")``,
    say CodeAct's cell timeout or a harness's own step timeout) has run out of time too: its children are cancelled
    without a wait.  Once the wait is over, a spawn on the agent (say from a task its harness left behind) is
    refused as a failed launch: nobody would wait for that child.
    """

    agent = ctx.agent
    span = agent.misc.setdefault(EXECUTION_SPAN_KEY, [time.perf_counter(), None])
    deadline = None if timeout is None else asyncio.get_running_loop().time() + timeout
    try:
        try:
            result = await asyncio.wait_for(ctx.harness.fn(ctx), timeout)
        except TimeoutError as exc:
            result = AgentResult("timeout", error=f"Agent timed out{f' after {timeout}s' if timeout else ''}: {exc}")
        except asyncio.CancelledError:
            agent.result = AgentResult("cancelled", error="The agent was cancelled")
            raise
        except Exception as exc:
            logger.warning("harness %s failed on agent %s", ctx.harness.ref, agent.agent_id, exc_info=True)
            result = AgentResult("error", error=f"{type(exc).__name__}: {exc}")
        else:
            if not isinstance(result, AgentResult):
                result = AgentResult(
                    "error", error=f"Harness {ctx.harness.ref} returned {type(result).__name__}, not an AgentResult"
                )
        if result.status != "timeout":  # the agent's deadline, or a timeout its harness reports, ends its children
            try:
                await ctx.wait_children(deadline)
            except asyncio.CancelledError:
                agent.result = dataclasses.replace(
                    result, status="cancelled", error="The agent was cancelled while its subagents ran"
                )
                raise
    finally:
        ctx.finished = True  # a spawn from here on is refused: nobody would wait for its child
        await ctx.cancel_children()  # nothing is left unless the agent was cancelled or ran out of time
        span[1] = time.perf_counter()
    agent.result = result
    return result


__all__ = [
    "InlineAgentContext",
    "InlineRuntime",
    "failure_outcome",
    "grade_session",
    "recorded_task",
    "run_agent",
    "with_reason",
]
