"""Gym adapters: step environments as a function harness, or as an ORS environment with one ``act`` tool.

A *step env* is the classic reset/step loop over text::

    class StepEnv(Protocol):
        async def reset(self, task: Task) -> str: ...                        # the first observation
        async def step(self, action: str) -> tuple[str, float, bool]: ...    # observation, reward, done

* :func:`loop` turns one into a function harness that alternates model calls and ``env.step``: the model's
  reply text is the action, the observation is the next user message.  The step env lives inside the harness,
  so it grades itself: the return (the sum of the step rewards, the gym convention) is recorded in the agent's
  misc under ``gym_return``, and the ``gym`` program scores the episode with it.
* :func:`as_environment` wraps one as an openreward ``Environment`` subclass whose single ``act`` tool steps
  it, so any harness plays it through a tool and the ORS grading path scores it: every result carries the return
  so far as its reward, so the call that ends the episode grades it with the return, and so does the
  ``max_tool_calls`` cap.  It works with openreward 0.1.125 (``tool(fn=None, *, shared=True)``) and later.

When the agent's budget runs out before the episode ends, :func:`loop` is scored with the return so far, and so
is :func:`as_environment` at its session's ``max_tool_calls``; an ``act`` episode that the agent leaves unfinished
otherwise (its own step budget ran out first) is graded as any ORS episode without a finished result: 0,
``no_submit``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from platoon.harness.agent import LIMIT_MESSAGE, AgentFn, AgentResult, out_of_steps
from platoon.programs import program
from platoon.task import Task

if TYPE_CHECKING:
    from platoon.envs.service import Outcome
    from platoon.harness.agent import AgentContext
    from platoon.programs import EpisodeContext

# The agent misc key under which ``loop`` records the episode's return.
GYM_RETURN_MISC_KEY = "gym_return"
GYM_STEP_MISC_KEY = "gym"


class StepEnv(Protocol):
    """A reset/step environment over text.  An optional ``close`` (sync or async) is called when it is done."""

    async def reset(self, task: Task) -> str:
        """Start an episode of ``task``; return the first observation."""
        ...

    async def step(self, action: str) -> tuple[str, float, bool]:
        """Apply ``action``; return the observation, the step reward and whether the episode is done."""
        ...


async def _close(env: Any) -> None:
    close = getattr(env, "close", None)
    if close is not None:
        result = close()
        if inspect.isawaitable(result):
            await result


def loop(env_factory: Callable[[], StepEnv], *, max_turns: int | None = None, system: str | None = None) -> AgentFn:
    """A function harness that plays ``env_factory()`` turn by turn (register it with ``@pt.harness``).

    The loop ends when the env says ``done`` (``completed``), or with ``limit`` and the out-of-budget warning
    (``LIMIT_MESSAGE``) after ``max_turns`` turns or when the agent's step budget or ``ctx.max_steps`` runs out.
    ``system`` is an optional system message.  Each step is annotated with its action, observation and reward.
    """

    async def gym_loop(ctx: AgentContext) -> AgentResult:
        env = env_factory()
        try:
            observation = await env.reset(ctx.task)
            messages: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
            messages.append({"role": "user", "content": observation})
            total, turns = 0.0, 0
            ctx.annotate(**{GYM_RETURN_MISC_KEY: total})
            while not out_of_steps(ctx, turns) and (max_turns is None or turns < max_turns):
                reply = await ctx.chat(messages)
                messages.append(dict(reply.message))
                observation, reward, done = await env.step(reply.text)
                total += float(reward)
                turns += 1
                ctx.annotate_step(
                    **{GYM_STEP_MISC_KEY: {"action": reply.text, "observation": observation, "reward": reward}}
                )
                ctx.annotate(**{GYM_RETURN_MISC_KEY: total})
                if done:
                    return AgentResult("completed", final_message=reply.text)
                messages.append({"role": "user", "content": observation})
            return AgentResult("limit", error=LIMIT_MESSAGE)
        finally:
            await _close(env)

    return gym_loop


@program("gym")
async def gym_program(ep: EpisodeContext) -> Outcome:
    """Run the root agent and score it with the return its ``gym.loop`` harness recorded.

    A root that failed (error, timeout, cancellation) keeps the usual untrainable outcome; a harness that
    recorded no return is graded by the env session.
    """

    from platoon.envs.service import Outcome

    result = await ep.run()
    root = getattr(ep, "root", None)
    returned = None if root is None else root.misc.get(GYM_RETURN_MISC_KEY)
    if result.status not in ("completed", "limit") or returned is None:
        return await ep.grade()
    return Outcome(status="scored", rewards={"reward": float(returned)})


def as_environment(
    env_factory: Callable[[], StepEnv],
    *,
    name: str,
    tasks: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> type:
    """An openreward ``Environment`` subclass named ``name`` that serves ``env_factory()`` through ``act``.

    ``tasks`` maps each split to its task specs (default: one ``train`` task ``{"id": "0"}``); a task spec's
    ``goal`` and ``id`` become the ``Task`` the step env is reset with, and the whole spec its ``misc``.  Each
    ``act`` result carries the return so far as its reward, and the step's own reward and the return in its
    metadata (``step_reward``, ``return``).
    """

    from openreward.environments import Environment, tool
    from openreward.environments.types import TextBlock, ToolOutput
    from pydantic import BaseModel

    task_table = {split: [dict(spec) for spec in specs] for split, specs in (tasks or {"train": [{"id": "0"}]}).items()}

    class Act(BaseModel):
        action: str

    class GymEnvironment(Environment):
        def __init__(self, task_spec: Mapping[str, Any] | None = None, secrets: Mapping[str, str] | None = None):
            super().__init__(dict(task_spec or {}), dict(secrets or {}))
            self._env = env_factory()
            self._observation: str | None = None
            self._return = 0.0
            self._done = False

        def _task(self) -> Task:
            spec = dict(self.task_spec)
            return Task(goal=spec.get("goal"), id=None if spec.get("id") is None else str(spec["id"]), misc=spec)

        async def _reset(self) -> str:
            if self._observation is None:
                self._observation = await self._env.reset(self._task())
            return self._observation

        async def get_prompt(self) -> list[TextBlock]:
            return [TextBlock(text=await self._reset())]

        @tool
        async def act(self, params: Act) -> ToolOutput:
            """Take one action in the environment; the reply is the next observation."""
            await self._reset()
            if self._done:
                return ToolOutput(blocks=[TextBlock(text="The episode is over.")], reward=self._return, finished=True)
            observation, reward, done = await self._env.step(params.action)
            self._return += float(reward)
            self._done = bool(done)
            # The return so far, as gym.loop scores an unfinished episode: the session's grade at max_tool_calls.
            return ToolOutput(
                blocks=[TextBlock(text=observation)],
                reward=self._return,
                finished=self._done,
                metadata={"step_reward": float(reward), "return": self._return},
            )

        async def teardown(self) -> None:
            await _close(self._env)

        def platoon_copy(self) -> Environment:
            """A ``copy`` fork's env: a deep copy, step env included, which the copy closes on its own teardown."""

            from platoon.envs.ors.sdk import deepcopy_env

            return deepcopy_env(self)

        @classmethod
        def list_tasks(cls, split: str) -> list[dict[str, Any]]:
            return [dict(spec) for spec in task_table.get(split, [])]

        @classmethod
        def list_splits(cls) -> list[str]:
            return list(task_table)

    # This module's annotations are strings (``from __future__``) naming local classes; openreward resolves the
    # tool's parameter model with ``get_type_hints``, so give it the classes themselves.
    GymEnvironment.act.__annotations__ = {"params": Act, "return": ToolOutput}
    GymEnvironment.__name__ = GymEnvironment.__qualname__ = name
    return GymEnvironment


__all__ = ["GYM_RETURN_MISC_KEY", "StepEnv", "as_environment", "gym_program", "loop"]
