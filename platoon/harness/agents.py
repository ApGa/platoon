"""Subagents: ``ctx.agents.spawn`` runs a child agent inline and returns its :class:`SubagentResult`.

The semantics are the legacy ``launch_subagent``'s, on the agent tree:

1. **Limits.**  The child's step budget (``max_steps``, default ``agents.child_max_steps``) plus one step is
   reserved on the spawning agent (``linear`` budgets; ``depth_aware`` budgets only check ``max_depth``, which
   bounds policy children: a verifier and its helpers run below it, see :mod:`platoon.harness.budget`).  A
   rejected reservation returns a :class:`SubagentLaunchFailure` message and runs nothing, as does a spawn on an
   agent whose run is over (``platoon.harness.inline.run_agent`` no longer waits for children then), and a spawn
   or a verifier launch once the rollout's gateway is shut down (``ModelGateway.shut_down``: the rollout worker is
   stopping), which forks no env session.
2. **Isolation.**  The child's task is ``Task.fork(goal, max_steps, task_misc)`` and its session a fork of the
   parent's (``EnvSession.fork``, mode from the spawn call, then ``agents.fork``, then the session's default).
   Its tool access is the spawn's ``access`` or ``agents.child_access``, never more than the parent's.  A
   ``shared`` fork (a ``platoon_fork`` child included) shares the parent's live env and its gate, so their tool
   calls take turns on it (:mod:`platoon.envs.gate`); a spawn holds no turn, so a parent waiting for its child
   never blocks the child's calls.
3. **The child.**  A child agent (``gateway.open_subagent``: role, goal, depth, max_steps, access) runs the parent's
   harness and options unless overridden, under ``agents.child_timeout_s``.  Its launch, run, grading and session
   close are one task that the parent agent owns (``InlineAgentContext.start_child``) and the spawn call awaits.
   Cancelling the call cancels the child and gives its budget back, as CodeAct's original ``launch_subagent`` did
   (say a harness that bounds a spawn with ``asyncio.wait_for``), unless the harness called ``keep_children()``
   first, as CodeAct does before it closes its cell loop and so cancels the launches its cells left pending.  A call
   still running when the parent's harness ends (say an un-awaited ``asyncio.gather`` of spawns), or cancelled
   after ``keep_children()``, keeps its child: the parent waits for it to finish and be graded
   (``platoon.harness.inline.run_agent``), and cancels it only when the parent itself is cancelled or times
   out, a timeout its harness reports included (``AgentResult("timeout")``, say CodeAct's cell timeout).  A
   failing child never raises into its parent: it comes back with an error status and the legacy failure text.
4. **Grading** (``agents.child_grading``): ``env`` grades the child's forked session; ``verifier`` runs a
   read-only verifier agent (:mod:`platoon.harness.verify`); ``none`` leaves the child to share the episode
   outcome.  A child whose verifier does not run (the run's judges are off, or verifiers are switched off) shares
   the episode outcome too, and its trajectory says why (``child_grading``, :mod:`platoon.utils.judges`).  A child
   whose harness crashed, timed out or was cancelled gets the matching untrainable outcome
   without grading, as the root does; one that crashed after filling its context window is still graded, by its
   env or its verifier, with reason ``context_overflow`` (``platoon.harness.inline.with_reason``).  A child cut
   off while it is graded (by its parent's deadline or a cancelled spawn call) is ``cancelled`` (reason
   ``grading_cancelled``), never one that shares the episode outcome.
5. **Stats.**  The parent's step that spawned the child (its last recorded call) gets
   ``reward/subagent_launched`` and ``reward/subagent_succeeded`` in its ``reward_misc``, as the legacy
   recursive envs recorded them; harnesses must not record them again.

Children whose harness is a TOML spec are not supported: ``spawn`` raises ``TypeError`` for them.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from platoon.harness.agent import TOOL_ACCESS_LEVELS, AgentResult, SubagentResult
from platoon.harness.budget import BudgetExceededError, SubagentDepthScope
from platoon.harness.inline import (
    InlineAgentContext,
    InlineRuntime,
    failure_outcome,
    grade_session,
    run_agent,
    with_reason,
)
from platoon.harness.verify import (
    PENDING_JUDGMENT,
    SUBAGENT_BEHAVIOR_JUDGMENT_MISC_KEY,
    SUBAGENT_OUTCOME_JUDGMENT_MISC_KEY,
    SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY,
    SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY,
    SUBAGENT_REWARD_VERIFIES_TRAJECTORY_ID_MISC_KEY,
    VERIFIER_ROLE,
    combine_outcome_and_behavior_judgments,
    error_message,
    format_verifier_goal,
    is_verifier_task_misc,
    judge_error_judgment,
    mark_verifier_agent,
    propagate_verifier_task_misc,
    record_judgment,
    return_message,
    run_behavior_judge,
    verifier_judgment,
    verifier_task_misc,
)
from platoon.utils.judges import CHILD_GRADING_MISC_KEY, JUDGES_OFF, grading_fallback
from platoon.utils.trajectory_status import trajectory_has_forced_zero_reward

if TYPE_CHECKING:
    from platoon.envs.service import EnvSession, ForkMode, Outcome
    from platoon.gateway.core import AgentSession
    from platoon.harness.function import FunctionHarness
    from platoon.task import Task

logger = logging.getLogger(__name__)

# How long closing a child's forked session may take before it is abandoned.
SESSION_CLOSE_TIMEOUT_SECONDS = 10.0
# The (environment, tool access) pairs whose verifiers were reported to see no tools (see _warn_if_toolless).
_WARNED_TOOLLESS_VERIFIERS: set[tuple[str, str]] = set()

_ACCESS_RANK = {access: rank for rank, access in enumerate(reversed(TOOL_ACCESS_LEVELS))}  # read_only < ... < full


class SubagentLaunchFailure(str):
    """A rejected launch, preserving the string result expected by CodeAct.

    Tool adapters can report a typed error without guessing from the message
    text or mistaking a successful child's prose for a launch failure.
    """


class Subagents:
    """``ctx.agents`` of an inline agent (:class:`~platoon.harness.agent.AgentAPI`)."""

    def __init__(self, ctx: InlineAgentContext) -> None:
        self._ctx = ctx

    async def spawn(
        self,
        goal: str,
        *,
        harness: str | None = None,
        max_steps: int | None = None,
        fork: ForkMode | None = None,
        access: str | None = None,
        role: str = "subagent",
        options: Mapping[str, Any] | None = None,
        task_misc: Mapping[str, Any] | None = None,
    ) -> SubagentResult:
        return await spawn(
            self._ctx,
            goal,
            harness=harness,
            max_steps=max_steps,
            fork=fork,
            access=access,
            role=role,
            options=options,
            task_misc=task_misc,
        )

    def keep_children(self) -> None:
        self._ctx.keeps_children = True


SHUTTING_DOWN = "the rollout is shutting down."


def shutting_down(runtime: InlineRuntime) -> bool:
    """Whether the rollout's gateway was shut down (``ModelGateway.shut_down``): its process is stopping, so no
    subagent or verifier may start, and none may fork an env session."""

    return bool(getattr(runtime.gateway, "is_shut_down", False))


def clamp_access(requested: str, parent: str) -> str:
    """The more restrictive of two tool access levels."""

    for access in (requested, parent):
        if access not in _ACCESS_RANK:
            raise ValueError(f"Unknown tool access {access!r}; expected one of {TOOL_ACCESS_LEVELS}")
    return requested if _ACCESS_RANK[requested] <= _ACCESS_RANK[parent] else parent


def depth_scope(
    *, parent_task_misc: Mapping[str, Any], child_task_misc: Mapping[str, Any], verifies_trajectory_id: str | None
) -> SubagentDepthScope:
    """Where a child sits relative to the synthetic verifier tree (legacy ``_subagent_depth_scope``)."""

    if not child_task_misc.get(SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY):
        return "policy"
    if (
        verifies_trajectory_id is not None
        and child_task_misc.get(SUBAGENT_REWARD_VERIFIES_TRAJECTORY_ID_MISC_KEY) == verifies_trajectory_id
    ):
        return "verifier_root"
    if parent_task_misc.get(SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY) and parent_task_misc.get(
        SUBAGENT_REWARD_VERIFIES_TRAJECTORY_ID_MISC_KEY
    ):
        return "verifier_helper"
    return "verifier_descendant"


async def spawn(
    parent: InlineAgentContext,
    goal: str,
    *,
    harness: str | None = None,
    max_steps: int | None = None,
    fork: ForkMode | None = None,
    access: str | None = None,
    role: str = "subagent",
    options: Mapping[str, Any] | None = None,
    task_misc: Mapping[str, Any] | None = None,
) -> SubagentResult:
    """Run ``goal`` in a child of ``parent`` (see the module docstring)."""

    if parent.finished:
        return _rejected(goal, f"Cannot launch a subagent for goal {goal}: its parent agent has finished.")
    if shutting_down(parent.runtime):
        return _rejected(goal, f"Cannot launch a subagent for goal {goal}: {SHUTTING_DOWN}")
    runtime = parent.runtime
    settings = runtime.agents
    child_harness = _resolve_harness(harness, parent)
    if options is None:
        options = parent.options if child_harness is parent.harness else {}
    max_steps = settings.child_max_steps if max_steps is None else max_steps
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        return _rejected(goal, f"Cannot launch a subagent for goal {goal}: max_steps must be a positive integer.")
    child_access = clamp_access(access or settings.child_access, parent.access)
    # The parent's step that launched this child: it gets the child's stats.
    anchor_call_id = parent.agent.calls[-1].call_id if parent.agent.calls else None

    parent_task_misc = dict(parent.task.misc)
    child_task = parent.task.fork(
        goal,
        max_steps,
        task_misc=propagate_verifier_task_misc(parent_task_misc, dict(task_misc) if task_misc is not None else None),
    )
    scope = depth_scope(
        parent_task_misc=parent_task_misc, child_task_misc=dict(child_task.misc), verifies_trajectory_id=None
    )
    child = parent.start_child(
        _run_child(
            parent,
            child_task,
            goal=goal,
            harness=child_harness,
            max_steps=max_steps,
            fork=fork,
            access=child_access,
            role=role,
            options=options,
            scope=scope,
            anchor_call_id=anchor_call_id,
        )
    )
    # The parent owns the child, so run_agent waits for it if the harness ends with this call still running.
    try:
        return await asyncio.shield(child)
    except asyncio.CancelledError:
        # Cancelling the call cancels the child, and returns once it has unwound (its budget released, its session
        # closed), unless the parent keeps the children of cancelled calls (see the module docstring).
        if not parent.keeps_children:
            child.cancel()
            await asyncio.wait([child])
        raise


async def _run_child(
    parent: InlineAgentContext,
    child_task: Task,
    *,
    goal: str,
    harness: FunctionHarness,
    max_steps: int,
    fork: ForkMode | None,
    access: str,
    role: str,
    options: Mapping[str, Any],
    scope: SubagentDepthScope,
    anchor_call_id: str | None,
) -> SubagentResult:
    """One child's lifecycle: launch and run it, grade it, close its session, and credit the spawning step."""

    launched = await _launch(
        parent,
        child_task,
        goal=goal,
        harness=harness,
        max_steps=max_steps,
        fork=fork,
        access=access,
        role=role,
        options=options,
        scope=scope,
        tree_parent=parent.agent,
        base_session=parent.session,
    )
    if isinstance(launched, SubagentResult):
        return launched
    child, child_session, result = launched
    from platoon.envs.service import Outcome

    # Fail closed until the grade is in: a child cut off while it is graded (its parent's deadline, a cancelled
    # spawn call) is untrainable, never a child that shares the episode outcome.  A grade it got stands even if
    # closing its session is cut off.
    child.agent.outcome = Outcome(status="cancelled", reason="grading_cancelled")
    try:
        outcome = await _grade_child(parent, child, child_session, result, goal=goal)
        child.agent.outcome = outcome
    finally:
        await close_session(child_session, keep=parent.session)
    if anchor_call_id is not None and not is_verifier_task_misc(child_task.misc):
        reward_misc = parent.agent.step_misc.setdefault(anchor_call_id, {}).setdefault("reward_misc", {})
        success = float(outcome.reward) if outcome is not None and outcome.reward is not None else 0.0
        reward_misc["reward/subagent_launched"] = reward_misc.get("reward/subagent_launched", 0) + 1
        reward_misc["reward/subagent_succeeded"] = reward_misc.get("reward/subagent_succeeded", 0.0) + success
    return SubagentResult(
        agent_id=child.agent_id,
        trajectory_id=child.trajectory_id,
        goal=goal,
        result=result,
        outcome=outcome,
        message=return_message(result),
    )


def _resolve_harness(harness: str | None, parent: InlineAgentContext) -> FunctionHarness:
    from platoon.harness.function import FunctionHarness
    from platoon.harness.registry import get_harness

    if harness is None:
        return parent.harness
    resolved = get_harness(harness)
    if not isinstance(resolved, FunctionHarness):
        raise TypeError(
            f"spawn(harness={harness!r}): {resolved.ref} is a TOML harness spec; subagents run function harnesses "
            "only (@harness)"
        )
    return resolved


def _rejected(goal: str, message: str) -> SubagentResult:
    return SubagentResult(
        agent_id="",
        trajectory_id="",
        goal=goal,
        result=AgentResult("error", error=message),
        outcome=None,
        message=SubagentLaunchFailure(message),
    )


async def _launch(
    spawner: InlineAgentContext,
    child_task: Task,
    *,
    goal: str,
    harness: FunctionHarness,
    max_steps: int,
    fork: ForkMode | None,
    access: str,
    role: str,
    options: Mapping[str, Any],
    scope: SubagentDepthScope,
    tree_parent: AgentSession,
    base_session: EnvSession,
) -> SubagentResult | tuple[InlineAgentContext, EnvSession, AgentResult]:
    """Reserve, fork, open and run one child agent; a launch that fails comes back as a rejected SubagentResult.

    The reservation is the spawner's and is released as soon as the child returns.  On success the child's
    session is still open (the caller grades and closes it).  The child runs in a task of its own, so what its
    harness does to context variables stays there; cancelling the caller cancels it.
    """

    runtime = spawner.runtime
    if shutting_down(runtime):  # checked again here, for verifiers, and before anything reaches the env
        return _rejected(goal, f"Failed to launch subagent for goal {goal}: {SHUTTING_DOWN}")
    reserved = max_steps + 1
    try:
        runtime.tracker.reserve_budget(spawner.agent_id, reserved, raise_on_failure=True, child_depth_scope=scope)
    except (BudgetExceededError, ValueError) as exc:
        guidance = getattr(exc, "guidance", "")
        return _rejected(
            goal, f"Not enough budget to launch subagent for goal {goal}. {exc}" + (f" {guidance}" if guidance else "")
        )
    try:
        try:
            child_session = await fork_session(base_session, fork or runtime.agents.fork, child_task)
        except Exception as exc:
            logger.warning("forking the session for a subagent of %s failed", spawner.agent_id, exc_info=True)
            return _rejected(
                goal, f"Failed to launch subagent for goal {goal}: the environment could not be forked ({exc})."
            )
        try:
            agent = runtime.gateway.open_subagent(tree_parent, role=role, goal=goal, max_steps=max_steps, access=access)
            child = InlineAgentContext(
                runtime,
                agent,
                task=child_task,
                session=child_session,
                harness=harness,
                instruction=goal,
                options=options,
            )
            if role == VERIFIER_ROLE:
                mark_verifier_agent(agent, str(child_task.misc.get(SUBAGENT_REWARD_VERIFIES_TRAJECTORY_ID_MISC_KEY)))
            task = asyncio.get_running_loop().create_task(run_agent(child, timeout=runtime.agents.child_timeout_s))
        except Exception as exc:
            await close_session(child_session, keep=base_session)
            logger.warning("opening a subagent of %s failed", spawner.agent_id, exc_info=True)
            return _rejected(goal, f"Failed to launch subagent for goal {goal}: {exc}")
        try:
            result = await task
        except BaseException:
            await close_session(child_session, keep=base_session)
            raise
        return child, child_session, result
    finally:
        # Release synchronously, before any cleanup await can be cancelled.
        runtime.tracker.release_budget(spawner.agent_id, reserved)


async def _warn_if_toolless(verifier: InlineAgentContext) -> None:
    """Say once per environment and tool access when a verifier sees no tools: it then judges from the transcript
    alone.  ``read_only`` (the default) lists only tools that declare ``read_only``: on an env server, a tool says so
    only through the ``read_only_tools`` or ``tool_meta`` options, and Harbor's generic tools never do."""

    try:
        tools = await verifier.list_tools()
    except Exception:  # the verifier's own run reports what is wrong with the listing
        return
    if tools:
        return
    service = getattr(verifier.session, "_service", None)
    environment = str(getattr(service, "name", None) or type(verifier.session).__name__)
    key = (environment, verifier.access)
    if key in _WARNED_TOOLLESS_VERIFIERS:
        return
    _WARNED_TOOLLESS_VERIFIERS.add(key)
    logger.warning(
        "Verifier agents on %s see no tools with %s access, so they judge children from the transcript alone. "
        "Mark the tools a verifier may use read-only (the read_only_tools option, platoon_read_only_tools, or a "
        "readOnlyHint in tool_meta), or give verifiers another access (agents.verifier.access)",
        environment,
        verifier.access,
    )


async def fork_session(session: EnvSession, mode: ForkMode | None, task: Task) -> EnvSession:
    """``session.fork`` with ``mode``, or with the session's default mode when ``mode`` is None."""

    if mode is None:
        return await session.fork(task=task)
    return await session.fork(mode, task=task)


async def close_session(session: EnvSession, *, keep: EnvSession | None = None) -> None:
    """Close a child's session (never ``keep``, the session it was forked from); failures are logged."""

    if session is keep:
        return
    try:
        await asyncio.wait_for(session.close(), SESSION_CLOSE_TIMEOUT_SECONDS)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("closing a subagent session failed", exc_info=True)


async def _grade_child(
    spawner: InlineAgentContext,
    child: InlineAgentContext,
    child_session: EnvSession,
    result: AgentResult,
    *,
    goal: str,
) -> Outcome | None:
    runtime = spawner.runtime
    failed, overflow_reason = failure_outcome(
        result,
        context_overflows=child.agent.context_overflows,
        on_context_overflow=runtime.on_context_overflow,
        harness_ref=child.harness.ref,
        on_harness_exit=runtime.on_harness_exit,
        gateway_errors=child.agent.gateway_errors,
    )
    if failed is not None:
        return failed
    grading = runtime.agents.child_grading
    if is_verifier_task_misc(child.task.misc):
        return None  # verifier helpers are part of their verifier's verdict, never graded themselves
    if grading == "env":
        return await grade_session(child_session, reason=overflow_reason)
    if grading == "verifier":
        verdict = await _verify_child(spawner, child, child_session, result, goal=goal)
        return None if verdict is None else with_reason(verdict, overflow_reason)
    return None


def _verifier_fallback(agent: AgentSession, runtime: InlineRuntime) -> str | None:
    """Why no verifier judges ``agent``, a child of an environment that grades children by verifier, or None when
    one does (the reasons are :data:`platoon.utils.judges.GRADING_FALLBACK_REASONS`)."""

    settings = runtime.agents.verifier
    if not agent.judges:
        return JUDGES_OFF
    if not runtime.verifiers_enabled:
        return "verifiers_off"
    if not settings.enabled:
        return "verifier_disabled"
    if settings.max_steps <= 0:
        return "verifier_max_steps_0"
    return None


async def _verify_child(
    spawner: InlineAgentContext,
    child: InlineAgentContext,
    child_session: EnvSession,
    result: AgentResult,
    *,
    goal: str,
) -> Outcome | None:
    """Judge a finished policy child with a verifier agent (and the behavior judge, when one is set)."""

    from platoon.envs.service import Outcome
    from platoon.gateway.trace import build_agent_trajectory

    runtime = spawner.runtime
    settings = runtime.agents.verifier
    agent = child.agent
    fallback = _verifier_fallback(agent, runtime)
    if fallback is not None:
        # The child shares the episode's outcome, as with child_grading: none, and says why.
        agent.misc[CHILD_GRADING_MISC_KEY] = grading_fallback(fallback)
        return None
    if trajectory_has_forced_zero_reward(agent):
        # The environment already established a valid failed outcome; no
        # verifier is needed and its verdict must not relabel that failure.
        return Outcome(status="scored", rewards={"reward": 0.0}, reason="forced_zero")

    # Fail closed while verification is in flight: if the tree is cancelled now, the child stays ineligible.
    agent.outcome = record_judgment(agent, dict(PENDING_JUDGMENT))
    verifier_goal = format_verifier_goal(
        child_goal=goal,
        child_trajectory_id=agent.trajectory_id,
        child_message=return_message(result),
        child_error_message=error_message(result),
        child_task_goal=child.task.goal,
    )
    behavior_judge = runtime.behavior_judge
    # Audit behavior for every completed child, including verifier-zero
    # outcomes: those are precisely where task-tracker loops and stuck endings
    # would otherwise evade the behavior gate.  Run the one-shot audit beside
    # the environment verifier so the two reward-model requests overlap.
    behavior_task = (
        asyncio.create_task(
            run_behavior_judge(behavior_judge, goal=goal, trajectory=build_agent_trajectory(agent, task=child.task))
        )
        if behavior_judge is not None
        else None
    )
    try:
        verifier_task = child.task.fork(
            verifier_goal,
            settings.max_steps,
            task_misc=verifier_task_misc(dict(child.task.misc), agent.trajectory_id),
        )
        launched = await _launch(
            spawner,
            verifier_task,
            goal=verifier_goal,
            harness=child.harness,
            max_steps=settings.max_steps,
            fork=None,
            access=settings.access,
            role=VERIFIER_ROLE,
            options=child.options,
            scope="verifier_root",
            tree_parent=agent,
            base_session=child_session,
        )
        if isinstance(launched, SubagentResult):
            outcome_judgment = judge_error_judgment(launched.message)
        else:
            verifier, verifier_session, verifier_result = launched
            try:
                await _warn_if_toolless(verifier)  # its listing is cached by its run: no env call
            finally:
                await close_session(verifier_session, keep=child_session)
            outcome_judgment = verifier_judgment(verifier.agent, verifier_result)

        if behavior_task is None:
            judgment = outcome_judgment
        else:
            # Keep the source verdicts independently queryable; the effective
            # verdict stays under the historical key consumed by trainers.
            agent.misc[SUBAGENT_OUTCOME_JUDGMENT_MISC_KEY] = outcome_judgment
            behavior_judgment = await behavior_task
            agent.misc[SUBAGENT_BEHAVIOR_JUDGMENT_MISC_KEY] = behavior_judgment
            if bool(outcome_judgment.get(SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY)):
                judgment = combine_outcome_and_behavior_judgments(outcome_judgment, behavior_judgment)
            else:
                # A malformed outcome verdict remains the authoritative reason
                # this child is ineligible, even if the behavior audit passed.
                judgment = outcome_judgment
    finally:
        if behavior_task is not None:
            if not behavior_task.done():
                behavior_task.cancel()
            await asyncio.gather(behavior_task, return_exceptions=True)
    return record_judgment(agent, judgment)


__all__ = ["SubagentLaunchFailure", "Subagents", "clamp_access", "depth_scope", "spawn"]
