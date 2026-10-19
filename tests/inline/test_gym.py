"""Gym adapters: a step env as a function harness (with the gym program) and as an ORS environment."""

from __future__ import annotations

import asyncio

import pytest
from inline_fakes import ToyService, routed_task
from test_function_harness import rollout, rules

import platoon as pt
from platoon import gym
from platoon.harness.agent import LIMIT_MESSAGE
from platoon.task import Task


class GuessEnv:
    """Guess the task's number; one point for a hit."""

    closed = 0

    async def reset(self, task: Task) -> str:
        self.target = int(task.misc.get("target", 3))
        return "guess:start"

    async def step(self, action: str) -> tuple[str, float, bool]:
        guess = int(action.strip()) if action.strip().isdigit() else -1
        if guess == self.target:
            return "guess:correct", 1.0, True
        return ("guess:higher" if guess < self.target else "guess:lower"), 0.0, False

    async def close(self) -> None:
        GuessEnv.closed += 1


pt.harness("toy-guess")(gym.loop(GuessEnv, max_turns=4))


class DenseEnv:
    """A quarter point per ``+``, and the game never ends: an episode always runs out of budget."""

    async def reset(self, task: Task) -> str:
        self.turn = 0
        return "dense:0"

    async def step(self, action: str) -> tuple[str, float, bool]:
        self.turn += 1
        return f"dense:{self.turn}", 0.25 if action.strip() == "+" else 0.0, False


pt.harness("toy-dense")(gym.loop(DenseEnv, max_turns=4))
DENSE_ACTIONS = ["+", "+", "+", "-"]  # a return of 0.75, and a last step reward of 0


@pytest.fixture(autouse=True)
def _reset():
    ToyService.sessions.clear()
    GuessEnv.closed = 0


def test_gym_loop_plays_a_step_env_and_scores_its_return(make_gateway):
    gateway, sampler = make_gateway(rules({"guess:start": "1", "guess:higher": "3"}))
    collection, interactions = asyncio.run(rollout(gateway, routed_task(harness="toy-guess", program="gym")))

    (root,) = collection.trajectories.values()
    assert root.reward == 1.0 and root.misc["gym_return"] == 1.0
    assert [step.misc["gym"]["observation"] for step in root.steps] == ["guess:higher", "guess:correct"]
    assert root.finish_message == "3" and GuessEnv.closed == 1


def test_gym_loop_stops_at_max_turns(make_gateway):
    gateway, _ = make_gateway(rules({"guess:": "0"}))
    collection, _ = asyncio.run(rollout(gateway, routed_task(harness="toy-guess", program="gym")))
    (root,) = collection.trajectories.values()
    assert len(root.steps) == 4 and root.misc["agent_result"] == {"status": "limit", "error": LIMIT_MESSAGE}
    assert root.reward == 0.0 and root.misc["outcome"]["status"] == "scored"


def test_gym_loop_stops_at_the_agents_max_steps(make_gateway):
    gateway, _ = make_gateway(rules({"guess:": "0"}))
    task = routed_task(harness="toy-guess", program="gym", max_steps=2)
    collection, _ = asyncio.run(rollout(gateway, task))
    (root,) = collection.trajectories.values()
    assert len(root.steps) == 2 and root.misc["agent_result"]["status"] == "limit"
    assert root.error_message == LIMIT_MESSAGE


def test_as_environment_serves_the_step_env_through_one_act_tool():
    from openreward.environments.types import ToolOutput

    env_class = gym.as_environment(GuessEnv, name="GuessEnv", tasks={"train": [{"id": "a", "target": 2}]})
    assert env_class.__name__ == "GuessEnv"
    assert env_class.list_splits() == ["train"] and env_class.list_tasks("train") == [{"id": "a", "target": 2}]
    assert getattr(env_class.act, "_env_tool", False)

    async def play():
        env = env_class(task_spec={"id": "a", "target": 2})
        prompt = await env.get_prompt()
        miss = await env.act(_act(env_class, "1"))
        hit = await env.act(_act(env_class, "2"))
        await env.teardown()
        return prompt, miss, hit

    prompt, miss, hit = asyncio.run(play())
    assert prompt[0].text == "guess:start"
    assert isinstance(miss, ToolOutput) and not miss.finished and miss.reward == 0.0
    assert hit.finished and hit.reward == 1.0 and hit.blocks[0].text == "guess:correct"
    assert GuessEnv.closed == 1


def _act(env_class, action: str):
    import typing

    params_type = typing.get_type_hints(env_class.act)["params"]
    return params_type(action=action)


def test_an_ors_gym_environment_can_be_copied_and_each_copy_closes_its_own_step_env():
    from platoon.envs.ors import OrsEnvironmentService

    env_class = gym.as_environment(GuessEnv, name="GuessEnv", tasks={"train": [{"id": "a", "target": 2}]})

    async def play():
        service = OrsEnvironmentService(env_class)
        parent = await service.open(await service.get_task("train:0"), rollout_id="r")
        try:
            await parent.call_tool("act", {"action": "1"})
            copied = await parent.fork("copy")
            try:
                hit = await copied.call_tool("act", {"action": "2"})
                assert hit.finished and hit.reward == 1.0
                assert copied._lease.backend.env._env is not parent._lease.backend.env._env
            finally:
                await copied.close()
            assert GuessEnv.closed == 1
            assert not (await parent.call_tool("act", {"action": "3"})).finished  # the parent plays on
        finally:
            await parent.close()

    asyncio.run(play())
    assert GuessEnv.closed == 2


def test_both_adapters_score_an_episode_that_ran_out_of_budget_with_its_return_so_far(make_gateway):
    """gym.loop at its turn limit, and gym.as_environment at its session's max_tool_calls, score the return so far:
    each act result reports it as its reward, which is what the max_tool_calls grade takes."""

    from platoon.envs.ors import OrsEnvironmentService

    gateway, _ = make_gateway(rules({f"dense:{turn}": action for turn, action in enumerate(DENSE_ACTIONS)}))
    collection, _ = asyncio.run(rollout(gateway, routed_task(harness="toy-dense", program="gym")))
    (root,) = collection.trajectories.values()
    assert root.misc["agent_result"]["status"] == "limit" and root.reward == 0.75

    async def play():
        service = OrsEnvironmentService(gym.as_environment(DenseEnv, name="DenseEnv"), max_tool_calls=4)
        session = await service.open(await service.get_task("train:0"), rollout_id="r")
        try:
            results = [await session.call_tool("act", {"action": action}) for action in DENSE_ACTIONS]
            capped = await session.call_tool("act", {"action": "+"})
            return results, capped, await session.grade()
        finally:
            await session.close()

    results, capped, outcome = asyncio.run(play())
    assert [result.reward for result in results] == [0.25, 0.5, 0.75, 0.75]
    assert results[-1].metadata == {"step_reward": 0.0, "return": 0.75}
    assert capped.finished and capped.metadata == {"bridge_error": "max_tool_calls_exceeded"}
    assert (outcome.status, outcome.reward, outcome.reason) == ("scored", 0.75, "max_tool_calls")
