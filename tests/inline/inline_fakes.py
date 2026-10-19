"""Toys for inline function-harness tests: a prompt-driven sampler, a forkable env service with a terminal
``submit`` tool and a read-only ``look`` tool, and ``toy-agent``, a harness that follows a tiny text protocol:

* ``SPAWN g1|g2``: spawn one child per goal (concurrently) and read their messages;
* ``LOOK`` / ``SUBMIT x``: call the env tools;
* ``AUX`` / ``JUDGE``: make an ``aux`` or ``judge`` call first;
* ``FINISH x``: finish with final message ``x``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import tinker
from gateway_fakes import sample_response

import platoon as pt
from platoon.envs.service import Outcome, ToolResult
from platoon.task import Task
from platoon.testing import last_user_message
from platoon.train.components import EnvironmentRoute

SERVICE = "inline_fakes:ToyService"


@dataclass
class PolicySampler:
    """Answers each call with ``policy(last user message)``; ``delays`` maps a message prefix to a sleep."""

    tokenizer: Any
    policy: Callable[[str], str]
    calls: list[str] = field(default_factory=list)
    delays: dict[str, float] = field(default_factory=dict)
    sampling: list[Any] = field(default_factory=list)  # each call's tinker SamplingParams

    async def sample_async(self, *, prompt: tinker.ModelInput, sampling_params: Any, num_samples: int) -> Any:
        message = last_user_message(self.tokenizer.decode(prompt.to_ints()))
        self.calls.append(message)
        self.sampling.append(sampling_params)
        for prefix, delay in self.delays.items():
            if message.startswith(prefix):
                await asyncio.sleep(delay)
        text = self.policy(message)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        tokens += self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        return sample_response(tokens)

    def sample(self, **kwargs: Any) -> Any:
        raise AssertionError("the gateway samples asynchronously")


@dataclass
class ToyTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    terminal: bool = False
    read_only: bool = False


TOOLS = [
    ToyTool("submit", "Submit the answer.", {"type": "object", "properties": {"answer": {"type": "string"}}}, True),
    ToyTool("look", "Look at the state.", {"type": "object", "properties": {}}, read_only=True),
]


@dataclass
class ToyState:
    """State shared by a session and its shared forks."""

    submissions: dict[str, str] = field(default_factory=dict)  # goal -> answer
    tool_calls: list[tuple[str, str]] = field(default_factory=list)  # (goal, tool)


@dataclass
class ToySession:
    task: Task
    state: ToyState
    service: ToyService
    sandbox: Any = None
    closed: bool = False
    forks: int = 0

    @property
    def instruction(self) -> str:
        return self.task.goal or ""

    async def tools(self) -> list[ToyTool]:
        return list(TOOLS)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        self.state.tool_calls.append((self.task.goal or "", name))
        if name == "submit":
            self.state.submissions[self.task.goal or ""] = str(arguments.get("answer", ""))
            result = ToolResult.text("submitted")
            result.finished = True
            return result
        if name == "look":
            return ToolResult.text(f"submissions: {sorted(self.state.submissions)}")
        return ToolResult.text(f"unknown tool {name}", is_error=True)

    async def grade(self) -> Outcome:
        if self.service.grade_raises:
            raise RuntimeError("grader crashed")
        answer = self.state.submissions.get(self.task.goal or "")
        reward = 1.0 if answer and "ok" in answer else 0.0
        return Outcome(status="scored", rewards={"reward": reward}, reason=self.service.grade_reason)

    async def fork(self, mode: str = "shared", *, task: Task | None = None) -> ToySession:
        if self.service.fork_raises:
            raise RuntimeError("fork failed")
        self.forks += 1
        state = self.state if mode == "shared" else ToyState()
        child = ToySession(task=task or self.task, state=state, service=self.service)
        self.service.sessions.append(child)
        return child

    async def close(self) -> None:
        self.closed = True


class ToyService:
    name = "toy"
    sessions: ClassVar[list[ToySession]] = []
    grade_raises: ClassVar[bool] = False
    grade_reason: ClassVar[str | None] = None
    fork_raises: ClassVar[bool] = False

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    async def list_tasks(self, split: str) -> list[str]:
        return ["t0"]

    async def get_task(self, task_id: str) -> Task:
        return Task(id=task_id, goal=f"goal:{task_id}")

    async def open(self, task: Task, *, rollout_id: str) -> ToySession:
        session = ToySession(task=task, state=ToyState(), service=self)  # type: ignore[arg-type]
        ToyService.sessions.append(session)
        return session


@pt.harness("toy-agent")
async def toy_agent(ctx: pt.AgentContext) -> pt.AgentResult:
    messages: list[dict[str, Any]] = [{"role": "user", "content": ctx.instruction}]
    while True:
        if ctx.budget.remaining() <= 0:
            return pt.AgentResult("limit")
        reply = await ctx.chat(messages)
        text = reply.text.strip()
        messages.append(dict(reply.message))
        ctx.annotate_step(action=text.split(" ", 1)[0])
        if text.startswith("FINISH"):
            return pt.AgentResult("completed", final_message=text.removeprefix("FINISH").strip())
        if text.startswith("CRASH"):
            raise RuntimeError("toy harness crashed")
        if text.startswith("HANG"):
            await asyncio.sleep(3600)
        if text.startswith("SPAWN"):
            goals = [goal.strip() for goal in text.removeprefix("SPAWN").split("|") if goal.strip()]
            steps = ctx.options.get("child_steps")
            runs = await asyncio.gather(*(ctx.agents.spawn(goal, max_steps=steps) for goal in goals))
            observation = "obs: " + " ; ".join(f"[{run.result.status}] {run.message}" for run in runs)
        elif text.startswith("SUBMIT"):
            result = await ctx.call_tool("submit", {"answer": text.removeprefix("SUBMIT").strip()})
            observation = "obs: " + str(result.content[0]["text"]) + (" (error)" if result.is_error else "")
        elif text.startswith("LOOK"):
            result = await ctx.call_tool("look", {})
            observation = "obs: " + str(result.content[0]["text"])
        elif text.startswith("AUX"):
            aux = await ctx.chat([{"role": "user", "content": "aux: summarize"}], kind="aux")
            observation = "obs: aux said " + aux.text
        elif text.startswith("JUDGE"):
            verdict = await ctx.chat([{"role": "user", "content": "judge: rate"}], kind="judge")
            observation = "obs: judge said " + verdict.text
        else:
            observation = "obs: unknown action"
        messages.append({"role": "user", "content": observation})


def routed_task(*, goal: str = "goal:root", max_steps: int | None = 50, program: str = "single", **route: Any) -> Task:
    route.setdefault("harness", "toy-agent")
    return EnvironmentRoute(name="toy", service=SERVICE, program=program, **route).attach(
        Task(id="t0", goal=goal, max_steps=max_steps)
    )
