"""The agent contract: what a function harness sees of platoon while it runs one agent.

An *agent* is one running harness in a rollout's agent tree: the root runs the task, and every subagent it
spawns is a child agent with its own goal, step budget and tool access.  Platoon owns recording, identity,
budgets and outcomes; the harness owns only its loop.  A harness is an async function
``fn(ctx: AgentContext) -> AgentResult`` (:data:`AgentFn`), registered with ``@pt.harness(name)``
(:mod:`platoon.harness.function`).

Every model call an agent makes goes through the gateway and is recorded on that agent, whether the harness calls
``ctx.chat`` in process or sends HTTP to ``ctx.endpoint`` from its own client.  Each call carries a *kind*:

* ``policy`` (default): the agent's turns; one step of the agent's budget and one trajectory step.
* ``aux``: model calls the harness makes for itself (e.g. OpenHands condensation); a trainable trajectory
  step that does not count against the budget.
* ``judge``: judges; never trained, kept in the agent's misc for audit.

Over HTTP the kind is the ``X-Platoon-Call-Kind`` header (:meth:`ModelEndpoint.headers`).

This module is stdlib-only, so harness code can import the types without pulling in the trainer.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from platoon.envs.service import EnvSession, ForkMode, Outcome, ToolResult, ToolSpec
    from platoon.task import Task

# How a harness run ended (``AgentResult.status``).
AgentStatus = Literal["completed", "limit", "timeout", "error", "cancelled"]
# What an agent may do with the env's tools: everything, everything but the terminal (submit/finish) tool, or
# only tools that declare themselves read-only.
ToolAccess = Literal["full", "no_terminal", "read_only"]
# What a model call is for: ``policy`` (the agent's turns: trained, and one step of its budget), ``aux`` (calls the
# harness makes for itself: trained, outside the budget) or ``judge`` (recorded for audit, never trained).
CallKind = Literal["policy", "aux", "judge"]

CALL_KINDS: tuple[CallKind, ...] = ("policy", "aux", "judge")
CALL_KIND_HEADER = "X-Platoon-Call-Kind"
TOOL_ACCESS_LEVELS: tuple[ToolAccess, ...] = ("full", "no_terminal", "read_only")


@dataclass(frozen=True)
class AgentResult:
    """How a harness run ended.

    ``completed``: the harness finished (the env grades the outcome, not this status); ``limit``: it ran out of
    steps; ``timeout``/``cancelled``: it was interrupted; ``error``: it failed, with ``error`` saying why.
    """

    status: AgentStatus = "completed"
    # The agent's final answer, if it gave one.
    final_message: str | None = None
    # Why the run failed or stopped early.
    error: str | None = None


@dataclass(frozen=True)
class ChatReply:
    """One recorded model call: the assistant message (``content``, ``tool_calls``, ``reasoning_content``)."""

    # The id of the recorded call.
    call_id: str
    # The assistant message, in Chat Completions form.
    message: Mapping[str, Any]
    # Why generation stopped, for example ``stop``, ``length`` or ``tool_calls``.
    finish_reason: str

    @property
    def text(self) -> str:
        """The visible text of the reply ("" when the model only called tools)."""
        content = self.message.get("content")
        return content if isinstance(content, str) else ""

    @property
    def reasoning(self) -> str | None:
        """The model's reasoning text, when the renderer separates it from the reply."""
        return self.message.get("reasoning_content")

    @property
    def tool_calls(self) -> Sequence[Mapping[str, Any]]:
        """OpenAI tool calls (``{"id", "type": "function", "function": {"name", "arguments"}}``), possibly empty."""
        return self.message.get("tool_calls") or ()


@dataclass(frozen=True)
class ModelEndpoint:
    """The agent's model endpoint, for harnesses with their own HTTP client (OpenAI SDK, litellm, OpenHands).

    ``base_url`` is OpenAI-compatible and ends in ``/v1``; it is already scoped to this agent (the secret is in
    the path), so calls through it are recorded on the agent exactly like ``ctx.chat``.  ``api_key`` is the agent
    key, for clients that insist on sending one.
    """

    base_url: str
    api_key: str = field(repr=False)
    # The model name to send in requests.
    model: str

    def headers(self, kind: str = "policy") -> dict[str, str]:
        """Extra request headers that mark calls with ``kind`` (``policy``, ``aux`` or ``judge``)."""
        return {CALL_KIND_HEADER: kind}


@dataclass(frozen=True)
class McpEndpoint:
    """The agent's env tools over MCP (the gateway's tool plane, streamable HTTP), filtered by its access."""

    # The MCP server URL.
    url: str
    # Headers to send with every MCP request.
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SubagentResult:
    """What a parent gets back from ``ctx.agents.spawn``.

    A child never raises into its parent: a rejected launch (depth or budget) or a failed child comes back with
    a non-``completed`` ``result`` and a failure text in ``message``.  ``outcome`` is the child's own grade (its
    forked env session's grade, or its verifier's verdict), or None when the child was not graded.  A rejected
    launch opened no agent, so its ``agent_id`` and ``trajectory_id`` are empty.
    """

    # The child agent's id ("" for a rejected launch).
    agent_id: str
    # The id of the child's trajectory ("" for a rejected launch).
    trajectory_id: str
    # The goal the child was given.
    goal: str
    # How the child's harness run ended.
    result: AgentResult
    outcome: Outcome | None
    # The child's final message, or the failure text.
    message: str

    @property
    def launched(self) -> bool:
        """Whether a child agent ran (False for launches rejected by the depth limit or the budget)."""
        return bool(self.agent_id)

    @property
    def ok(self) -> bool:
        """Whether the child ran and its harness completed (its grade is in ``outcome``)."""
        return self.launched and self.result.status == "completed"


class Budget(Protocol):
    """The agent's step budget: one step is one ``policy`` call recorded on the agent."""

    def remaining(self) -> float:
        """Steps this agent may still take (``inf`` when unbounded); reserved child budgets are subtracted."""
        ...

    def used(self) -> float:
        """Steps used, counting descendants when children draw on the parent's budget."""
        ...


class AgentAPI(Protocol):
    """Subagents: ``ctx.agents.spawn`` runs a child agent to completion and returns its :class:`SubagentResult`."""

    async def spawn(
        self,
        goal: str,
        *,
        harness: str | None = None,
        max_steps: int | None = None,
        fork: ForkMode | None = None,
        access: ToolAccess | None = None,
        role: str = "subagent",
        options: Mapping[str, Any] | None = None,
        task_misc: Mapping[str, Any] | None = None,
    ) -> SubagentResult:
        """Run ``goal`` in a child agent.

        Unset arguments come from the environment's ``agents`` config and the parent: the same harness and
        options, the env's fork mode, the configured child access and step budget.  Concurrent spawns are
        allowed (``asyncio.gather``); each reserves its budget atomically.

        Cancelling the call cancels the child and gives its budget back, unless :meth:`keep_children` was called.
        A call still running when the parent's harness ends (say an un-awaited ``asyncio.gather`` of spawns) keeps
        its child: the parent waits for it to finish and be graded, and cancels it only when the parent itself is
        cancelled or times out, which includes a harness that returns ``AgentResult("timeout")``.
        """
        ...

    def keep_children(self) -> None:
        """Keep the children of spawn calls cancelled from now on: they run on, and the parent waits for them to
        finish and be graded before it ends.

        For a harness that cancels its pending calls on its way out but wants the children they launched to
        finish, as CodeAct does before it closes its cell loop (an un-awaited ``asyncio.gather`` of launches).
        """
        ...


class AgentContext(Protocol):
    """What a function harness gets for one agent."""

    # This agent's id.
    agent_id: str
    # The id of the trajectory this agent's calls are recorded on.
    trajectory_id: str
    # The parent agent's id (None for the root).
    parent_agent_id: str | None
    # Depth in the agent tree; the root is 0.
    depth: int
    # The agent's role: ``policy`` for the root, the ``role`` its spawn named for a child (default ``subagent``).
    role: str
    # The task the agent works on.
    task: Task
    # The environment session the agent acts on.
    session: EnvSession
    # The goal the agent works on.
    instruction: str
    # The agent's step limit (None: only the budget limits it).
    max_steps: int | None
    # The agent's tool access.
    access: ToolAccess
    # Harness options from the environment config (or the spawn call).
    options: Mapping[str, Any]
    # The agent's model endpoint over HTTP; calls through it are recorded on this agent exactly like ``chat``.
    endpoint: ModelEndpoint
    # The agent's tools over MCP, set when the gateway serves them.
    mcp: McpEndpoint | None
    # The agent's step budget.
    budget: Budget
    # Subagents: ``ctx.agents.spawn``.
    agents: AgentAPI

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        kind: str = "policy",
        **sampling: Any,
    ) -> ChatReply:
        """One model call, recorded on this agent; ``sampling`` takes Chat Completions parameters."""
        ...

    async def list_tools(self) -> list[ToolSpec]:
        """The env's tools, filtered by this agent's access."""
        ...

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        """Call an env tool; a tool outside this agent's access returns an error result."""
        ...

    def annotate(self, **misc: Any) -> None:
        """Merge ``misc`` into this agent's trajectory misc."""
        ...

    def annotate_step(self, **misc: Any) -> None:
        """Merge ``misc`` into the step misc of the last call recorded on this agent.

        ``exclude_from_training=True`` keeps that call out of the training data (the step stays for viewers).
        """
        ...


# A function harness: ``async def fn(ctx: AgentContext) -> AgentResult``; options arrive as ``ctx.options``.
AgentFn = Callable[..., Awaitable[AgentResult]]

# The error of a run that ends with ``limit``. platoon/gateway/trace.py and platoon/harness/verify.py give the same
# text to a ``limit`` run that reports no error, and behavior judges may recognize a limit by its prefix, so the
# wording stays fixed.
LIMIT_MESSAGE = "WARNING: Exhausted budget when running episode. Halting episode; task may be incomplete."


def out_of_steps(ctx: AgentContext, steps_taken: int) -> bool:
    """Whether the agent must stop: its budget is spent, or it took ``ctx.max_steps`` steps.

    Harness loops check this before each model call and end with ``AgentResult("limit", error=LIMIT_MESSAGE)``.
    """
    return ctx.budget.remaining() <= 0 or (ctx.max_steps is not None and steps_taken >= ctx.max_steps)


def terminal_tool(tools: Sequence[ToolSpec]) -> ToolSpec | None:
    """The env's terminal tool (``ToolSpec.terminal``: the submit/finish tool that ends the episode), if any."""
    return next((tool for tool in tools if getattr(tool, "terminal", False)), None)


def terminal_arguments(tool: ToolSpec, message: str) -> dict[str, Any]:
    """The arguments that hand ``message`` to a terminal tool: its first schema property, or none."""
    properties = list((tool.input_schema or {}).get("properties") or {})
    return {properties[0]: message} if properties else {}


__all__ = [
    "CALL_KINDS",
    "CALL_KIND_HEADER",
    "LIMIT_MESSAGE",
    "TOOL_ACCESS_LEVELS",
    "AgentAPI",
    "AgentContext",
    "AgentFn",
    "AgentResult",
    "AgentStatus",
    "Budget",
    "CallKind",
    "ChatReply",
    "McpEndpoint",
    "ModelEndpoint",
    "SubagentResult",
    "ToolAccess",
    "out_of_steps",
    "terminal_arguments",
    "terminal_tool",
]
