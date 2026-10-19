"""The judges switch (``RolloutConfig.judges``): with judges off no verifier agent, behavior judge, judge agent or
judge call runs; children graded by verifier share the episode's outcome and say why; with judges on, rollouts are
what they always were."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from inline_fakes import ToyService, routed_task

import platoon as pt
from platoon.config_defs import RolloutConfig
from platoon.harness.rollout import EmbeddedGateway, run_harness_rollout
from platoon.harness.verify import (
    PARENT_USAGE_SKIPPED,
    SUBAGENT_BEHAVIOR_JUDGMENT_MISC_KEY,
    SUBAGENT_PARENT_USAGE_JUDGMENT_MISC_KEY,
    SUBAGENT_REWARD_JUDGMENT_MISC_KEY,
    apply_subagent_parent_usage_gates,
)
from platoon.trajectory import ParentInfo, Trajectory, TrajectoryCollection, TrajectoryStep
from platoon.utils.judges import CHILD_GRADING_MISC_KEY, JUDGES_MISC_KEY, JUDGES_REFUSED_MISC_KEY, judges_enabled


@pytest.fixture(autouse=True)
def _reset_service():
    ToyService.sessions.clear()
    yield
    ToyService.sessions.clear()


def rules(table: dict[str, str]):
    def policy(message: str) -> str:
        for prefix, reply in table.items():
            if message.startswith(prefix):
                return reply
        return "FINISH no rule"

    return policy


async def rollout(gateway, task, **config: Any):
    from platoon.train.tinker.proxy import proxy_interactions

    proxy_interactions.set({})
    config.setdefault("verbose", False)
    return await run_harness_rollout(task, RolloutConfig(**config), gateway=EmbeddedGateway(gateway))


def by_goal(collection) -> dict[str, Any]:
    return {trajectory.task.goal: trajectory for trajectory in collection.trajectories.values()}


VERIFIED = json.dumps({"status": "verified", "score": 1.0, "summary": "checked"})
VERIFIER_RULES = {
    "goal:root": "SPAWN goal:a",
    "goal:a": "SUBMIT ok-a",
    "You are a verifier agent": "LOOK",
    "obs: submissions": f"FINISH {VERIFIED}",
    "obs: [": "SUBMIT ok",
    "obs: submitted": "FINISH done",
}


def test_judges_off_runs_no_verifier_and_the_child_shares_the_episode_outcome(make_gateway):
    gateway, sampler = make_gateway(rules(VERIFIER_RULES))
    task = routed_task(agents={"child_grading": "verifier"})
    collection = asyncio.run(rollout(gateway, task, judges=False))

    root, child = collection.trajectories.values()
    assert child.task.goal == "goal:a"
    assert child.misc[CHILD_GRADING_MISC_KEY] == {"configured": "verifier", "applied": "none", "reason": "judges_off"}
    assert SUBAGENT_REWARD_JUDGMENT_MISC_KEY not in child.misc and "outcome" not in child.misc
    assert child.reward == root.reward == 1.0  # the episode's outcome
    assert root.misc[JUDGES_MISC_KEY] == {"enabled": False} and not judges_enabled(collection)
    assert JUDGES_MISC_KEY not in child.misc
    assert not any(message.startswith("You are a verifier") for message in sampler.calls)


def test_judges_on_leave_the_rollout_as_it_was(make_gateway):
    gateway, sampler = make_gateway(rules(VERIFIER_RULES))
    task = routed_task(agents={"child_grading": "verifier", "verifier": {"max_steps": 4}})
    collection = asyncio.run(rollout(gateway, task))

    root, child, verifier = collection.trajectories.values()
    assert JUDGES_MISC_KEY not in root.misc and judges_enabled(collection)
    assert CHILD_GRADING_MISC_KEY not in child.misc
    assert child.misc[SUBAGENT_REWARD_JUDGMENT_MISC_KEY]["status"] == "verified"
    assert verifier.misc["agent"]["role"] == "verifier"
    assert any(message.startswith("You are a verifier") for message in sampler.calls)


@pytest.mark.parametrize(
    ("config", "agents", "reason"),
    [
        ({"enable_subagent_verifiers": False}, {}, "verifiers_off"),
        ({}, {"verifier": {"enabled": False}}, "verifier_disabled"),
        ({}, {"verifier": {"max_steps": 0}}, "verifier_max_steps_0"),
    ],
)
def test_the_finer_verifier_switches_record_their_fallback_too(make_gateway, config, agents, reason):
    gateway, sampler = make_gateway(rules(VERIFIER_RULES))
    task = routed_task(agents={"child_grading": "verifier", **agents})
    collection = asyncio.run(rollout(gateway, task, **config))

    root, child = collection.trajectories.values()
    assert child.misc[CHILD_GRADING_MISC_KEY] == {"configured": "verifier", "applied": "none", "reason": reason}
    assert JUDGES_MISC_KEY not in root.misc  # the judges themselves were on
    assert not any(message.startswith("You are a verifier") for message in sampler.calls)


def test_a_judge_call_is_refused_and_recorded_while_judges_are_off(make_gateway):
    gateway, sampler = make_gateway(rules({"goal:root": "JUDGE", "judge: rate": "FINISH 10/10"}))
    collection = asyncio.run(rollout(gateway, routed_task(), judges=False))

    (root,) = collection.trajectories.values()
    assert root.misc[JUDGES_REFUSED_MISC_KEY] == {"count": 1, "first": "A judge call"}
    assert root.misc["agent_result"]["status"] == "error"
    assert "JudgesOffError" in root.misc["agent_result"]["error"]
    assert root.misc["outcome"]["status"] == "error"  # the harness crashed on the refusal: never scored
    assert "judge: rate" not in sampler.calls


@pt.program("toy-judging-program")
async def judging_program(ep: pt.EpisodeContext) -> pt.Outcome:
    JUDGES_SEEN.append(ep.judges)
    ep.judge_endpoint()  # refused while judges are off
    await ep.run()
    return await ep.grade()


@pt.program("toy-judges-aware-program")
async def judges_aware_program(ep: pt.EpisodeContext) -> pt.Outcome:
    JUDGES_SEEN.append(ep.judges)
    if ep.judges:
        ep.judge_endpoint()
    await ep.run()
    return await ep.grade()


JUDGES_SEEN: list[bool] = []


def test_a_program_judge_is_refused_while_judges_are_off(make_gateway):
    JUDGES_SEEN.clear()
    gateway, _ = make_gateway(rules({"goal:root": "SUBMIT ok", "obs: submitted": "FINISH done"}))
    collection = asyncio.run(rollout(gateway, routed_task(program="toy-judging-program"), judges=False))

    (root,) = collection.trajectories.values()
    assert JUDGES_SEEN == [False]
    assert root.misc[JUDGES_REFUSED_MISC_KEY] == {"count": 1, "first": "A judge agent"}
    assert root.misc["outcome"]["status"] == "error"
    assert root.misc["outcome"]["failure_kind"] == "unexpected:JudgesOffError"


def test_a_program_that_reads_ep_judges_does_without_them(make_gateway):
    JUDGES_SEEN.clear()
    gateway, _ = make_gateway(rules({"goal:root": "SUBMIT ok", "obs: submitted": "FINISH done"}))
    off = asyncio.run(rollout(gateway, routed_task(program="toy-judges-aware-program"), judges=False))
    on = asyncio.run(rollout(gateway, routed_task(program="toy-judges-aware-program")))

    assert JUDGES_SEEN == [False, True]
    for collection in (off, on):
        (root,) = collection.trajectories.values()
        assert root.reward == 1.0 and JUDGES_REFUSED_MISC_KEY not in root.misc


class RecordingJudge:
    def __init__(self) -> None:
        self.goals: list[str] = []

    async def judge(self, *, goal: str, trajectory: Any) -> dict[str, Any]:
        self.goals.append(goal)
        return {"status": "pass", "passed": True, "reason": "fine"}


BEHAVIOR_JUDGE = RecordingJudge()


@pt.program("toy-behavior-judged")
async def behavior_judged(ep: pt.EpisodeContext) -> pt.Outcome:
    ep.behavior_judge = BEHAVIOR_JUDGE
    await ep.run()
    return await ep.grade()


def test_behavior_judges_do_not_run_while_judges_are_off(make_gateway):
    BEHAVIOR_JUDGE.goals.clear()
    gateway, _ = make_gateway(rules(VERIFIER_RULES))
    task = routed_task(program="toy-behavior-judged", agents={"child_grading": "verifier"})
    collection = asyncio.run(rollout(gateway, task, judges=False))

    child = by_goal(collection)["goal:a"]
    assert BEHAVIOR_JUDGE.goals == []
    assert SUBAGENT_BEHAVIOR_JUDGMENT_MISC_KEY not in child.misc
    assert child.misc[CHILD_GRADING_MISC_KEY]["reason"] == "judges_off"


def _verified_child_collection(*, judges: bool) -> TrajectoryCollection:
    root = Trajectory(id="root", steps=[TrajectoryStep(misc={})], reward=1.0)
    if not judges:
        root.misc[JUDGES_MISC_KEY] = {"enabled": False}
    child = Trajectory(id="child", parent_info=ParentInfo(id="root", fork_step=0), steps=[TrajectoryStep(misc={})])
    child.misc[SUBAGENT_REWARD_JUDGMENT_MISC_KEY] = {"status": "verified", "score": 1.0, "training_eligible": True}
    return TrajectoryCollection(id="c", trajectories={"root": root, "child": child})


class FailingUsageJudge:
    async def judge_parent_usage(self, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("a parent-usage judge ran while judges were off")


class UsedJudge:
    async def judge_parent_usage(self, **kwargs: Any) -> dict[str, Any]:
        return {"status": "used", "used": True, "reason": "it read the answer"}


def test_parent_usage_judges_are_skipped_while_judges_are_off():
    collection = _verified_child_collection(judges=False)
    asyncio.run(apply_subagent_parent_usage_gates(collection, FailingUsageJudge()))

    child = collection.trajectories["child"]
    assert child.misc[SUBAGENT_PARENT_USAGE_JUDGMENT_MISC_KEY] == PARENT_USAGE_SKIPPED
    assert "exclude_from_policy_training" not in child.misc
    assert "reward_misc" not in child.steps[-1].misc  # nothing gated

    collection = _verified_child_collection(judges=True)
    asyncio.run(apply_subagent_parent_usage_gates(collection, UsedJudge()))
    assert collection.trajectories["child"].misc[SUBAGENT_PARENT_USAGE_JUDGMENT_MISC_KEY]["status"] == "used"


def test_judge_calls_and_judge_agents_are_refused_over_http(make_gateway):
    import aiohttp

    from platoon.gateway import GatewayServer

    gateway, sampler = make_gateway(rules({"judge": "FINISH 10/10"}))

    async def run() -> tuple[int, dict, int, dict]:
        server = GatewayServer(gateway)
        base_url = await server.start(host="127.0.0.1")
        root = gateway.open_agent(rollout_id="r", trajectory_id="t", interactions={})
        root.judges = False
        headers = {"Authorization": f"Bearer {root.api_key}", "X-Platoon-Call-Kind": "judge"}
        body = {"model": "policy", "messages": [{"role": "user", "content": "judge this"}]}
        try:
            async with aiohttp.ClientSession() as http:
                async with http.post(f"{base_url}/v1/chat/completions", json=body, headers=headers) as response:
                    call = (response.status, await response.json())
                headers.pop("X-Platoon-Call-Kind")
                async with http.post(
                    f"{base_url}/v1/platoon/agents", json={"role": "judge"}, headers=headers
                ) as response:
                    mint = (response.status, await response.json())
        finally:
            await server.stop()
        return (*call, *mint)

    call_status, call_error, mint_status, mint_error = asyncio.run(run())
    assert call_status == 403 and call_error["error"]["type"] == "permission_error"
    assert "inference.judges: false" in call_error["error"]["message"]
    assert mint_status == 403
    assert sampler.calls == []
