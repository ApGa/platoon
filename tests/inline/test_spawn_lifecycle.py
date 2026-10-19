"""Subagent launch lifecycle (ported from the legacy fork-lifecycle tests): budget admission before any fork,
budget release on every path, each forked session closed exactly once, children cancelled with the spawn call
that awaits them, and children of calls the parent's harness left running waited for."""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import pytest
from tinker import ModelInput

from platoon.envs.service import Outcome
from platoon.gateway.core import ModelGateway
from platoon.gateway.trace import build_trajectory_collection
from platoon.harness import agents as agents_module
from platoon.harness.agent import AgentResult
from platoon.harness.agents import SubagentLaunchFailure
from platoon.harness.function import FunctionHarness
from platoon.harness.inline import InlineAgentContext, InlineRuntime, run_agent
from platoon.harness.verify import SUBAGENT_REWARD_JUDGMENT_MISC_KEY, VERIFIER_ROLE
from platoon.task import Task
from platoon.train.components import AgentTreeConfig
from platoon.utils.trajectory_status import TRAJECTORY_FORCED_ZERO_REWARD_MISC_KEY, trajectory_was_cancelled


class Renderer:
    """Enough of a renderer for a gateway that never samples."""

    def build_generation_prompt(self, messages: Any) -> ModelInput:
        return ModelInput.from_ints([1, 2, 3])

    def parse_response(self, tokens: Any) -> Any:
        return {"role": "assistant", "content": ""}, True

    def get_stop_sequences(self) -> list[Any]:
        return []


@dataclass
class Session:
    task: Task
    events: list[str]
    fork_gate: asyncio.Event | None = None
    fork_error: Exception | None = None
    closed: int = 0
    sandbox: Any = None
    instruction: str = ""
    children: list[Session] = field(default_factory=list)

    async def tools(self) -> list[Any]:
        return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        raise AssertionError("no tools")

    async def grade(self) -> Any:
        from platoon.envs.service import Outcome

        return Outcome(status="scored", rewards={"reward": 1.0})

    async def fork(self, mode: str = "shared", *, task: Task | None = None) -> Session:
        self.events.append("fork")
        if self.fork_gate is not None:
            await self.fork_gate.wait()
        if self.fork_error is not None:
            raise self.fork_error
        child = Session(task=task or self.task, events=self.events)
        self.children.append(child)
        return child

    async def close(self) -> None:
        self.closed += 1
        self.events.append("close")


async def finish(ctx: Any) -> AgentResult:
    ctx.runtime.seen.append(ctx.agent_id)
    return AgentResult("completed", final_message=f"done {ctx.instruction}")


async def spawn_one(ctx: Any) -> AgentResult:
    """The root spawns one child and awaits it."""
    if ctx.depth == 0:
        await ctx.agents.spawn("child", max_steps=3)
        return AgentResult("completed", final_message="root done")
    return await finish(ctx)


async def leave_children_running(ctx: Any) -> AgentResult:
    """The root spawns two children and returns without awaiting them, like a CodeAct cell's un-awaited
    ``asyncio.gather`` of ``launch_subagent``; each child takes ``runtime.child_seconds``."""
    if ctx.depth == 0:
        ctx.runtime.pending = asyncio.gather(*(ctx.agents.spawn(goal, max_steps=2) for goal in ("a", "b")))
        return AgentResult("completed", final_message="root left early")
    await asyncio.sleep(ctx.runtime.child_seconds)
    return await finish(ctx)


async def cancel_spawn_calls_keeping_children(ctx: Any) -> AgentResult:
    """The root starts two spawns, then cancels the calls on its way out after ``keep_children()``, as CodeAct
    does with the launches its cells left pending when it closes its cell loop."""
    if ctx.depth == 0:
        calls = [asyncio.ensure_future(ctx.agents.spawn(goal, max_steps=2)) for goal in ("a", "b")]
        await asyncio.sleep(0.05)
        ctx.agents.keep_children()
        for call in calls:
            call.cancel()
        await asyncio.gather(*calls, return_exceptions=True)
        return AgentResult("completed", final_message="root left early")
    await asyncio.sleep(ctx.runtime.child_seconds)
    return await finish(ctx)


async def give_up_on_a_child(ctx: Any) -> AgentResult:
    """The root bounds a slow child with ``asyncio.wait_for``, then spawns another child."""
    if ctx.depth == 0:
        try:
            await asyncio.wait_for(ctx.agents.spawn("slow", max_steps=6), 0.1)
        except TimeoutError:
            ctx.runtime.after_giving_up = ctx.budget.remaining()
        second = await ctx.agents.spawn("second", max_steps=6)
        return AgentResult("completed", final_message=second.message)
    if ctx.instruction == "slow":
        await asyncio.sleep(60)
    return await finish(ctx)


async def time_out_with_children_running(ctx: Any) -> AgentResult:
    """The root's harness reports a timeout of its own (CodeAct's cell timeout, say, or a step timeout) while two
    children it spawned are still running."""
    if ctx.depth == 0:
        ctx.runtime.pending = asyncio.gather(*(ctx.agents.spawn(goal, max_steps=2) for goal in ("a", "b")))
        await asyncio.sleep(0.05)
        return AgentResult("timeout", error="Cell did not finish within 1 seconds")
    await asyncio.sleep(ctx.runtime.child_seconds)
    return await finish(ctx)


async def spawn_after_returning(ctx: Any) -> AgentResult:
    """The root leaves behind a task that yields to the loop before it spawns, so the spawn comes after the root
    stopped waiting for children."""
    if ctx.depth == 0:

        async def helper() -> Any:
            await asyncio.sleep(0)
            return await ctx.agents.spawn("late", max_steps=2)

        ctx.runtime.pending = asyncio.ensure_future(helper())
        return AgentResult("completed", final_message="left a helper")
    return await finish(ctx)


async def overflowing_child(ctx: Any) -> AgentResult:
    """The root spawns a child that crashes after filling its context window; its verifier (the same harness)
    says it succeeded.  With ``runtime.forced_zero`` the env has forced the child's reward to 0 first."""
    if ctx.depth == 0:
        await ctx.agents.spawn("child", max_steps=3)
        return AgentResult("completed", final_message="root done")
    if ctx.role == VERIFIER_ROLE:
        return AgentResult("completed", final_message=json.dumps({"status": "verified", "score": 1.0}))
    if ctx.runtime.forced_zero:
        ctx.annotate(**{TRAJECTORY_FORCED_ZERO_REWARD_MISC_KEY: True})
    ctx.agent.context_overflows += 1  # what the gateway counts when it refuses an overflowing prompt
    raise RuntimeError("ContextWindowExceededError: the prompt does not fit")


def root_context(*, max_steps: int = 10, fn: Any = finish, **agents: Any) -> tuple[InlineAgentContext, Session]:
    gateway = ModelGateway(sampler=None, renderer=Renderer(), tokenizer=None, model_name="toy")  # type: ignore[arg-type]
    runtime = InlineRuntime(gateway=gateway, base_url=None, agents=AgentTreeConfig(**agents))
    runtime.seen = []  # type: ignore[attr-defined]
    root = gateway.open_agent(rollout_id="r", trajectory_id="root", interactions={}, max_steps=max_steps)
    session = Session(task=Task(goal="root"), events=[])
    harness = FunctionHarness(name="lifecycle", version="1", runtime="inline", executes_model_code=False, fn=fn)
    ctx = InlineAgentContext(runtime, root, task=session.task, session=session, harness=harness, instruction="root")
    return ctx, session


def children(ctx: InlineAgentContext) -> list[Any]:
    return ctx.runtime.gateway.agents("r")[1:]


def test_budget_admission_precedes_forks_and_a_denial_allocates_nothing():
    ctx, session = root_context(max_steps=5)
    run = asyncio.run(ctx.agents.spawn("child", max_steps=10))
    assert isinstance(run.message, SubagentLaunchFailure) and not run.launched
    assert "Not enough budget to launch subagent for goal child" in run.message
    assert session.events == [] and ctx.budget.remaining() == 5


def test_a_fork_failure_releases_the_budget_and_comes_back_as_a_failed_launch():
    ctx, session = root_context()
    session.fork_error = RuntimeError("no capacity")
    run = asyncio.run(ctx.agents.spawn("child", max_steps=3))
    assert run.result.status == "error" and isinstance(run.message, SubagentLaunchFailure)
    assert "no capacity" in run.message and ctx.budget.remaining() == 10
    assert ctx.runtime.gateway.agents("r") == [ctx.agent]  # no child agent was opened


def test_cancelling_the_parent_during_a_fork_releases_the_budget():
    ctx, session = root_context(fn=spawn_one)
    session.fork_gate = asyncio.Event()

    async def scenario() -> None:
        run = asyncio.create_task(run_agent(ctx))
        await asyncio.sleep(0.01)
        assert ctx.budget.remaining() == 6  # 3 + 1 reserved while forking
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

    asyncio.run(scenario())
    assert ctx.budget.remaining() == 10 and session.closed == 0 and not session.children
    assert ctx.agent.result.status == "cancelled" and children(ctx) == []


def test_cancelling_a_spawn_call_during_its_fork_releases_the_budget():
    ctx, session = root_context()
    session.fork_gate = asyncio.Event()

    async def scenario() -> None:
        call = asyncio.create_task(ctx.agents.spawn("child", max_steps=3))
        await asyncio.sleep(0.01)
        assert ctx.budget.remaining() == 6  # 3 + 1 reserved while forking
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        assert ctx.budget.remaining() == 10  # released by the time the cancelled call returns

    asyncio.run(scenario())
    assert session.closed == 0 and not session.children and children(ctx) == []


def test_cancelling_a_spawn_call_cancels_its_running_child():
    ctx, session = root_context(fn=leave_children_running, child_grading="env")
    ctx.runtime.child_seconds = 60

    async def scenario() -> None:
        call = asyncio.create_task(ctx.agents.spawn("child", max_steps=3))
        await asyncio.sleep(0.05)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        # The call returns once the child has unwound: its budget is back and its session closed.
        assert ctx.budget.remaining() == 10 and [child.closed for child in session.children] == [1]

    asyncio.run(scenario())
    (child,) = children(ctx)
    assert child.result.status == "cancelled" and child.outcome is None


def test_a_harness_that_gives_up_on_a_child_gets_its_budget_back():
    # CodeAct's original launch_subagent semantics: asyncio.wait_for cancels the child it gives up on, which releases
    # the child's reservation (6 + 1 of the root's 10 steps), so the next spawn fits.
    ctx, session = root_context(fn=give_up_on_a_child, child_grading="env")

    result = asyncio.run(run_agent(ctx, timeout=30))

    assert ctx.runtime.after_giving_up == 10
    assert result == AgentResult("completed", final_message="done second")
    assert [(child.goal, child.result.status) for child in children(ctx)] == [
        ("slow", "cancelled"),
        ("second", "completed"),
    ]
    assert [child.closed for child in session.children] == [1, 1] and ctx.budget.remaining() == 10


def test_a_spawn_call_cancelled_after_keep_children_leaves_its_child_to_the_parent():
    ctx, session = root_context(child_grading="env")
    session.fork_gate = asyncio.Event()

    async def scenario() -> None:
        call = asyncio.create_task(ctx.agents.spawn("child", max_steps=3))
        await asyncio.sleep(0.01)
        ctx.agents.keep_children()
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        assert ctx.budget.remaining() == 6  # the launch goes on, still holding its reservation
        session.fork_gate.set()
        await ctx.wait_children()

    asyncio.run(scenario())
    (child_session,) = session.children
    (child,) = children(ctx)
    assert child.result.status == "completed" and child.outcome.reward == 1.0
    assert child_session.closed == 1 and ctx.budget.remaining() == 10


@pytest.mark.parametrize(
    "harness", [leave_children_running, cancel_spawn_calls_keeping_children], ids=["unawaited", "kept-children"]
)
def test_an_agent_waits_for_the_children_its_harness_left_running_and_grades_them(harness):
    ctx, session = root_context(fn=harness, child_grading="env")
    ctx.runtime.child_seconds = 0.2

    result = asyncio.run(run_agent(ctx, timeout=30))

    assert result == AgentResult("completed", final_message="root left early")
    assert [(child.result.status, child.outcome.reward) for child in children(ctx)] == [("completed", 1.0)] * 2
    assert sorted(ctx.runtime.seen) == sorted(child.agent_id for child in children(ctx))
    assert [child.closed for child in session.children] == [1, 1] and ctx.budget.remaining() == 10
    if harness is leave_children_running:
        assert [run.message for run in ctx.runtime.pending.result()] == ["done a", "done b"]


def test_children_still_running_when_their_parent_runs_out_of_time_are_cancelled():
    ctx, session = root_context(fn=leave_children_running, child_grading="env")
    ctx.runtime.child_seconds = 60
    started = time.monotonic()

    result = asyncio.run(run_agent(ctx, timeout=0.5))

    assert time.monotonic() - started < 10
    assert result == AgentResult("completed", final_message="root left early")  # the harness's result stands
    assert [(child.result.status, child.outcome) for child in children(ctx)] == [("cancelled", None)] * 2
    assert [child.closed for child in session.children] == [1, 1] and ctx.budget.remaining() == 10
    assert isinstance(ctx.runtime.pending.exception(), asyncio.CancelledError)  # the harness's own future


def test_a_harness_that_reports_a_timeout_has_its_children_cancelled_without_a_wait():
    # A timeout the harness reports is the agent's: a harness's step timeout ends its children at once.
    ctx, session = root_context(fn=time_out_with_children_running, child_grading="env")
    ctx.runtime.child_seconds = 60
    started = time.monotonic()

    result = asyncio.run(run_agent(ctx, timeout=30))

    assert time.monotonic() - started < 10  # long before the agent's own deadline
    assert result == AgentResult("timeout", error="Cell did not finish within 1 seconds")
    assert [(child.result.status, child.outcome) for child in children(ctx)] == [("cancelled", None)] * 2
    assert [child.closed for child in session.children] == [1, 1] and ctx.budget.remaining() == 10
    assert isinstance(ctx.runtime.pending.exception(), asyncio.CancelledError)  # the harness's own future


def test_an_agent_cancelled_while_its_children_run_cancels_them():
    ctx, session = root_context(fn=leave_children_running)
    ctx.runtime.child_seconds = 60

    async def scenario() -> None:
        run = asyncio.create_task(run_agent(ctx))
        await asyncio.sleep(0.2)  # the harness has returned; the agent waits for its children
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

    asyncio.run(scenario())
    assert ctx.agent.result == AgentResult(
        "cancelled", final_message="root left early", error="The agent was cancelled while its subagents ran"
    )
    assert [child.result.status for child in children(ctx)] == ["cancelled", "cancelled"]
    assert [child.closed for child in session.children] == [1, 1] and ctx.budget.remaining() == 10
    assert isinstance(ctx.runtime.pending.exception(), asyncio.CancelledError)


def test_a_spawn_on_an_agent_whose_run_is_over_is_refused():
    # Nobody would wait for that child: it would run after the rollout built its collection.
    ctx, session = root_context(fn=spawn_after_returning, child_grading="env")

    async def scenario() -> Any:
        assert (await run_agent(ctx, timeout=30)).status == "completed"
        return await ctx.runtime.pending

    late = asyncio.run(scenario())

    assert not late.launched and isinstance(late.message, SubagentLaunchFailure)
    assert late.message == "Cannot launch a subagent for goal late: its parent agent has finished."
    assert children(ctx) == [] and session.events == [] and ctx.budget.remaining() == 10


def test_a_spawn_after_the_gateway_shut_down_is_refused_and_forks_nothing():
    # A stopping rollout worker shuts its gateway down: an agent that ignores the cancellation must not start more.
    ctx, session = root_context(child_grading="env")
    ctx.runtime.gateway.shut_down()

    run = asyncio.run(ctx.agents.spawn("child", max_steps=3))

    assert not run.launched and isinstance(run.message, SubagentLaunchFailure)
    assert run.message == "Cannot launch a subagent for goal child: the rollout is shutting down."
    assert session.events == [] and ctx.budget.remaining() == 10


async def shut_down_as_the_child_finishes(ctx: Any) -> AgentResult:
    """The worker starts stopping (its gateway shuts down) just as the root's child finishes, before its verifier."""
    if ctx.depth == 0:
        ctx.runtime.child_run = await ctx.agents.spawn("child", max_steps=3)
        return AgentResult("completed", final_message="root done")
    ctx.runtime.gateway.shut_down()
    return AgentResult("completed", final_message="child done")


def test_no_verifier_starts_once_the_gateway_shut_down():
    # Enough budget for the verifier (its 20 steps + 1): only the shutdown can refuse it.
    ctx, session = root_context(max_steps=50, fn=shut_down_as_the_child_finishes, child_grading="verifier")

    assert asyncio.run(run_agent(ctx, timeout=30)).status == "completed"

    assert session.events == ["fork", "close"]  # the child's session only: the verifier forked none
    child = ctx.runtime.child_run
    assert child.launched and child.outcome.reward == 0.0
    judgment = child.outcome.metadata[SUBAGENT_REWARD_JUDGMENT_MISC_KEY]
    assert judgment["status"] == "judge_error", judgment
    assert judgment["summary"].endswith("the rollout is shutting down."), judgment


@pytest.mark.parametrize("slow", ["grade", "close"])
def test_a_child_cut_off_after_its_run_keeps_only_a_grade_it_got(monkeypatch, slow):
    # The children's runs finish fast, but grading them (say a Harbor verifier running the task's tests) or
    # closing their sessions does not, and the root runs out of time meanwhile.
    async def slow_grade(self: Session) -> Any:
        await asyncio.sleep(60)

    async def slow_close(self: Session) -> None:
        await asyncio.sleep(60)

    monkeypatch.setattr(Session, slow, slow_grade if slow == "grade" else slow_close)
    ctx, session = root_context(fn=leave_children_running, child_grading="env")
    ctx.runtime.child_seconds = 0.05

    result = asyncio.run(run_agent(ctx, timeout=0.5))

    assert result.status == "completed"
    episode = Outcome(status="scored", rewards={"reward": 1.0})  # say the root solved its task
    collection = build_trajectory_collection(ctx.runtime.gateway.agents("r"), task=session.task, outcome=episode)
    for child in children(ctx):
        trajectory = collection.trajectories[child.trajectory_id]
        assert child.result.status == "completed"
        if slow == "grade":
            # Cut off mid-grading: untrainable, never the root's reward.
            assert (child.outcome.status, child.outcome.reason) == ("cancelled", "grading_cancelled")
            assert trajectory.reward == 0.0 and trajectory_was_cancelled(trajectory)
        else:
            # Graded before the cut: the grade stands.
            assert child.outcome.reward == 1.0 and not trajectory_was_cancelled(trajectory)
    assert ctx.budget.remaining() == 10
    assert isinstance(ctx.runtime.pending.exception(), asyncio.CancelledError)  # the harness's own future


def test_a_kept_child_that_fails_after_its_call_was_cancelled_is_logged_not_left_unretrieved(monkeypatch, caplog):
    async def broken_grading(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("grading bug")

    monkeypatch.setattr(agents_module, "_grade_child", broken_grading)
    ctx, session = root_context(fn=cancel_spawn_calls_keeping_children, child_grading="env")
    ctx.runtime.child_seconds = 0.1

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(run_agent(ctx, timeout=30))
        gc.collect()

    assert result.status == "completed" and [child.closed for child in session.children] == [1, 1]
    assert caplog.text.count(f"a subagent of {ctx.agent_id} failed") == 2 and "grading bug" in caplog.text
    assert "never retrieved" not in caplog.text


@pytest.mark.parametrize("forced_zero", [False, True], ids=["verified", "forced-zero"])
def test_a_verifier_graded_child_that_overflowed_reports_context_overflow(forced_zero):
    ctx, session = root_context(max_steps=50, fn=overflowing_child, child_grading="verifier")
    ctx.runtime.forced_zero = forced_zero

    assert asyncio.run(run_agent(ctx, timeout=30)).status == "completed"

    (child,) = [agent for agent in children(ctx) if agent.role == "subagent"]
    assert child.result.status == "error" and child.outcome.status == "scored"
    assert child.outcome.reward == (0.0 if forced_zero else 1.0)
    # context_overflow is the reason; the verifier's own (forced_zero) stays in the metadata, as for env grades.
    summary = {
        "status": "scored",
        "rewards": {"reward": child.outcome.reward},
        "reason": "context_overflow",
        "reward_key": "reward",
    }
    if forced_zero:
        summary["grade_reason"] = "forced_zero"
    episode = Outcome(status="scored", rewards={"reward": 1.0})
    collection = build_trajectory_collection(ctx.runtime.gateway.agents("r"), task=session.task, outcome=episode)
    assert collection.trajectories[child.trajectory_id].misc["outcome"] == summary


def test_a_successful_child_closes_its_own_session_once_and_never_the_parents():
    ctx, session = root_context(child_grading="env")
    run = asyncio.run(ctx.agents.spawn("child", max_steps=3))
    assert run.ok and run.message == "done child" and run.outcome.reward == 1.0
    (child_session,) = session.children
    assert child_session.closed == 1 and session.closed == 0
    assert session.events == ["fork", "close"]
    assert ctx.budget.remaining() == 10  # released once the child returned
    child_agent = ctx.runtime.gateway.agents("r")[1]
    assert (child_agent.depth, child_agent.max_steps, child_agent.role, child_agent.tool_access) == (
        1,
        3,
        "subagent",
        "full",
    )
    assert child_agent.task.goal == "child" and child_agent.task.parent_tasks == [session.task]


def test_child_access_is_never_wider_than_the_parents():
    ctx, _ = root_context(child_access="no_terminal")
    asyncio.run(ctx.agents.spawn("a", max_steps=2))
    asyncio.run(ctx.agents.spawn("b", max_steps=2, access="read_only"))
    asyncio.run(ctx.agents.spawn("c", max_steps=2, access="full"))
    accesses = [agent.tool_access for agent in ctx.runtime.gateway.agents("r")[1:]]
    assert accesses == ["no_terminal", "read_only", "full"]

    ctx.agent.tool_access = ctx.access = "read_only"
    asyncio.run(ctx.agents.spawn("d", max_steps=2, access="full"))
    assert ctx.runtime.gateway.agents("r")[-1].tool_access == "read_only"
