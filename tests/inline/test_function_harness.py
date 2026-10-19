"""Function harnesses end to end: harness_rollout -> episode program -> inline agents -> trace -> datums."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp
import pytest
from inline_fakes import ToyService, routed_task

import platoon as pt
from platoon.config_defs import InferenceParams, RolloutConfig
from platoon.harness.rollout import EmbeddedGateway, run_harness_rollout
from platoon.harness.verify import (
    EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY,
    EXCLUDE_FROM_TRAINING_MISC_KEY,
    SUBAGENT_BEHAVIOR_JUDGMENT_MISC_KEY,
    SUBAGENT_REWARD_JUDGMENT_MISC_KEY,
    SUBAGENT_REWARD_VERIFIES_TRAJECTORY_ID_MISC_KEY,
)
from platoon.programs import get_program
from platoon.utils.tinker_data_processing import get_train_data_for_trajectory_collection
from platoon.utils.trajectory_status import trajectory_was_cancelled, trajectory_was_interrupted


@pytest.fixture(autouse=True)
def _reset_service():
    ToyService.sessions.clear()
    ToyService.grade_raises = False
    ToyService.grade_reason = None
    ToyService.fork_raises = False
    yield
    ToyService.sessions.clear()
    ToyService.grade_reason = None


def rules(table: dict[str, str]):
    """A policy: the reply of the first prefix of ``table`` the last user message starts with."""

    def policy(message: str) -> str:
        for prefix, reply in table.items():
            if message.startswith(prefix):
                return reply
        return "FINISH no rule"

    return policy


async def rollout(gateway, task, **config: Any):
    from platoon.train.tinker.proxy import proxy_interactions

    interactions: dict = {}
    proxy_interactions.set(interactions)
    config.setdefault("verbose", False)
    collection = await run_harness_rollout(task, RolloutConfig(**config), gateway=EmbeddedGateway(gateway))
    return collection, interactions


def by_goal(collection) -> dict[str, Any]:
    return {trajectory.task.goal: trajectory for trajectory in collection.trajectories.values()}


def datums(collection, interactions, **kwargs):
    return get_train_data_for_trajectory_collection(
        trajectory_collection=collection.to_dict(),
        interactions=interactions,
        task_id="t0",
        checkpoint_version=0,
        include_traj_depth=True,
        **kwargs,
    )


def test_a_function_harness_runs_end_to_end(make_gateway):
    gateway, sampler = make_gateway(
        rules({"goal:root": "LOOK", "obs: submissions": "SUBMIT ok-root", "obs: submitted": "FINISH root done"})
    )
    collection, interactions = asyncio.run(rollout(gateway, routed_task()))

    (root,) = collection.trajectories.values()
    assert root.reward == 1.0 and root.finish_message == "root done"
    assert root.misc["outcome"] == {
        "status": "scored",
        "rewards": {"reward": 1.0},
        "reason": None,
        "reward_key": "reward",
    }
    assert root.misc["harness"] == {
        "spec": "toy-agent@1",
        "kind": "function",
        "runtime": "inline",
        "executes_model_code": False,
        "program": "single",
    }
    assert [step.misc["action"] for step in root.steps] == ["LOOK", "SUBMIT", "FINISH"]
    assert root.steps[-1].misc["reward_misc"] == {"reward/success": 1.0}
    assert [call["name"] for call in root.misc["tool_calls"]] == ["look", "submit"]
    assert root.misc["agent_result"] == {"status": "completed", "error": None}
    (session,) = ToyService.sessions
    assert session.closed
    json.dumps(collection.to_dict())

    result = datums(collection, interactions)
    assert len(result.datums) == 1  # three turns, prefix-merged
    assert result.task_reward == 1.0 and result.task_reward_valid


def test_a_recursive_tree_grades_each_child_with_its_forked_session(make_gateway):
    gateway, _ = make_gateway(
        rules(
            {
                "goal:root": "SPAWN goal:a|goal:b",
                "goal:a": "SUBMIT ok-a",
                "goal:b1": "SUBMIT bad-b1",
                "goal:b": "SPAWN goal:b1",
                "obs: [": "SUBMIT ok",
                "obs: submitted": "FINISH done",
            }
        )
    )
    task = routed_task(agents={"child_grading": "env", "max_depth": 2, "budget": "depth_aware"})
    collection, interactions = asyncio.run(rollout(gateway, task))

    trajectories = by_goal(collection)
    assert list(collection.trajectories.values())[0].task.goal == "goal:root"
    assert {goal: trajectory.reward for goal, trajectory in trajectories.items()} == {
        "goal:root": 1.0,
        "goal:a": 1.0,
        "goal:b": 1.0,
        "goal:b1": 0.0,
    }
    assert trajectories["goal:b1"].misc["outcome"]["status"] == "scored"
    assert trajectories["goal:b1"].parent_info.id == trajectories["goal:b"].id
    assert trajectories["goal:a"].parent_info.id == trajectories["goal:root"].id
    spawn_step = trajectories["goal:root"].steps[0].misc["reward_misc"]
    assert spawn_step == {"reward/subagent_launched": 2, "reward/subagent_succeeded": 2.0}
    assert trajectories["goal:b"].steps[0].misc["reward_misc"] == {
        "reward/subagent_launched": 1,
        "reward/subagent_succeeded": 0.0,
    }
    assert all(session.closed for session in ToyService.sessions)

    result = datums(collection, interactions)
    depths = sorted(datum.loss_fn_inputs["traj_depth"].to_torch().item() for datum in result.datums)
    assert depths == [0, 1, 1, 2]
    rewards = {stats.trajectory_id: stats.reward for stats in result.trajectory_stats}
    assert rewards[trajectories["goal:b1"].id] == 0.0 and rewards[trajectories["goal:a"].id] == 1.0


class FailingBehaviorJudge:
    def __init__(self) -> None:
        self.goals: list[str] = []

    async def judge(self, *, goal: str, trajectory: Any) -> dict[str, Any]:
        self.goals.append(goal)
        assert trajectory.steps, "the judge sees the child's steps"
        return {"status": "fail", "passed": False, "reason": "it looped"}


JUDGE = FailingBehaviorJudge()


@pt.program("toy-judged")
async def judged_program(ep: pt.EpisodeContext) -> pt.Outcome:
    ep.behavior_judge = JUDGE
    await ep.run()
    return await ep.grade()


VERIFIED = json.dumps({"status": "verified", "score": 1.0, "summary": "checked"})
VERIFIER_RULES = {
    "goal:root": "SPAWN goal:a",
    "goal:a": "SUBMIT ok-a",
    "You are a verifier agent": "SUBMIT forged",
    "obs: Tool 'submit' is not available": "LOOK",
    "obs: submissions": f"FINISH {VERIFIED}",
    "obs: [": "SUBMIT ok",
    "obs: submitted": "FINISH done",
}


def test_verifier_children_judge_a_child_read_only(make_gateway):
    gateway, _ = make_gateway(rules(VERIFIER_RULES))
    task = routed_task(agents={"child_grading": "verifier", "verifier": {"max_steps": 4}})
    collection, interactions = asyncio.run(rollout(gateway, task))

    trajectories = list(collection.trajectories.values())
    root, child, verifier = trajectories
    assert child.task.goal == "goal:a" and verifier.parent_info.id == child.id
    judgment = child.misc[SUBAGENT_REWARD_JUDGMENT_MISC_KEY]
    assert judgment["status"] == "verified" and judgment["training_eligible"] is True
    assert child.reward == 1.0 and EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY not in child.misc
    assert child.steps[-1].misc["reward_misc"]["reward/subagent_judgment"] == 1.0
    assert verifier.misc[EXCLUDE_FROM_TRAINING_MISC_KEY] is True
    assert verifier.misc[SUBAGENT_REWARD_VERIFIES_TRAJECTORY_ID_MISC_KEY] == child.id
    assert verifier.misc["agent"]["role"] == "verifier"
    # Read-only access: the verifier's submit was refused (and not recorded); its look went through.
    assert [call["name"] for call in verifier.misc["tool_calls"]] == ["look"]
    assert ToyService.sessions[0].state.submissions.get(verifier.task.goal) is None

    result = datums(collection, interactions)
    trained = {stats.trajectory_id for stats in result.trajectory_stats}
    assert trained == {root.id, child.id}


def test_a_behavior_judge_gates_the_verified_score(make_gateway):
    gateway, _ = make_gateway(rules(VERIFIER_RULES))
    task = routed_task(program="toy-judged", agents={"child_grading": "verifier"})
    collection, _ = asyncio.run(rollout(gateway, task))

    child = by_goal(collection)["goal:a"]
    assert JUDGE.goals == ["goal:a"]
    assert child.misc[SUBAGENT_BEHAVIOR_JUDGMENT_MISC_KEY]["status"] == "fail"
    judgment = child.misc[SUBAGENT_REWARD_JUDGMENT_MISC_KEY]
    assert judgment["status"] == "behavior_rejected" and judgment["score"] == 0.0
    assert child.reward == 0.0


SNAPSHOTS: list[tuple[Any, str | None]] = []


class SnapshottingJudge:
    """Passes every child, keeping the episode's snapshot as it was when the judge was called."""

    def __init__(self, ep: pt.EpisodeContext) -> None:
        self.ep = ep

    async def judge(self, *, goal: str, trajectory: Any) -> dict[str, Any]:
        SNAPSHOTS.append((self.ep.snapshot(), trajectory.id))
        return {"status": "pass", "passed": True, "reason": "on task"}


@pt.program("toy-snapshot")
async def snapshot_program(ep: pt.EpisodeContext) -> pt.Outcome:
    SNAPSHOTS.append((ep.snapshot(), None))
    ep.behavior_judge = SnapshottingJudge(ep)
    await ep.run()
    return await ep.grade()


def test_a_snapshot_holds_the_tree_so_far_and_leaves_the_result_alone(make_gateway):
    SNAPSHOTS.clear()
    gateway, _ = make_gateway(rules(VERIFIER_RULES))
    task = routed_task(program="toy-snapshot", agents={"child_grading": "verifier"})
    collection, _ = asyncio.run(rollout(gateway, task))

    (before, _), (during, judged_id) = SNAPSHOTS
    # Before the run: the root alone, with no calls yet.
    [root_before] = before.trajectories.values()
    assert root_before.task.goal == "goal:root" and not root_before.steps
    # Mid-episode, while the verifier grading judges the child: the root's spawn call and the finished child.
    root_during = next(iter(during.trajectories.values()))
    child_during = by_goal(during)["goal:a"]
    assert child_during.id == judged_id and child_during.parent_info.id == root_during.id
    assert [step.misc["action"] for step in root_during.steps] == ["SPAWN"]
    assert child_during.finish_message == "done" and child_during.steps
    assert root_during.misc["outcome"] == {
        "status": "scored",
        "rewards": {"reward": 0.0},
        "reason": "snapshot",
        "reward_key": "reward",
    }

    # The rollout went on after both snapshots, and its result carries the real outcomes.
    root, child, verifier = collection.trajectories.values()
    assert root.id == root_during.id and child.id == child_during.id
    assert [step.misc["action"] for step in root.steps] == ["SPAWN", "SUBMIT", "FINISH"]
    assert root.misc["outcome"] == {
        "status": "scored",
        "rewards": {"reward": 1.0},
        "reason": None,
        "reward_key": "reward",
    }
    assert root.reward == 1.0 and root.steps[-1].misc["reward_misc"] == {"reward/success": 1.0}
    assert child.misc[SUBAGENT_BEHAVIOR_JUDGMENT_MISC_KEY]["status"] == "pass"
    assert child.misc[SUBAGENT_REWARD_JUDGMENT_MISC_KEY]["status"] == "verified" and child.reward == 1.0
    assert verifier.misc["agent"]["role"] == "verifier"
    assert not gateway.agents(collection.id) and not gateway._agents_by_id  # closed once the rollout ended


def test_disabled_verifiers_leave_children_ungraded(make_gateway):
    gateway, sampler = make_gateway(rules(VERIFIER_RULES))
    task = routed_task(agents={"child_grading": "verifier"})
    collection, _ = asyncio.run(rollout(gateway, task, enable_subagent_verifiers=False))

    assert len(collection.trajectories) == 2
    child = by_goal(collection)["goal:a"]
    assert SUBAGENT_REWARD_JUDGMENT_MISC_KEY not in child.misc and "outcome" not in child.misc
    assert child.reward == 1.0  # shares the episode outcome
    assert not any(message.startswith("You are a verifier") for message in sampler.calls)


def test_linear_budgets_reject_a_child_the_parent_cannot_afford(make_gateway):
    gateway, _ = make_gateway(
        rules({"goal:root": "SPAWN goal:a", "obs: [error] Not enough budget": "FINISH gave up", "goal:a": "FINISH a"})
    )
    task = routed_task(max_steps=3, harness_options={"child_steps": 5})
    collection, _ = asyncio.run(rollout(gateway, task))

    (root,) = collection.trajectories.values()
    assert root.finish_message == "gave up"
    assert "reward_misc" not in root.steps[0].misc  # a rejected launch is not a launched child


def test_depth_aware_budgets_cap_the_depth(make_gateway):
    gateway, _ = make_gateway(
        rules(
            {
                "goal:root": "SPAWN goal:a",
                "goal:a": "SPAWN goal:a1",
                "obs: [error] Not enough budget": "FINISH capped",
                "obs: [completed] capped": "FINISH root done",
            }
        )
    )
    task = routed_task(agents={"budget": "depth_aware", "max_depth": 1})
    collection, _ = asyncio.run(rollout(gateway, task))

    trajectories = by_goal(collection)
    assert set(trajectories) == {"goal:root", "goal:a"}
    assert trajectories["goal:a"].finish_message == "capped"
    assert trajectories["goal:root"].finish_message == "root done"


def test_a_child_that_runs_out_of_steps_reports_the_legacy_message(make_gateway):
    gateway, _ = make_gateway(
        rules(
            {
                "goal:root": "SPAWN goal:a",
                "goal:a": "LOOK",
                "obs: submissions": "LOOK",
                "obs: [limit] Subagent did not finish before its step budget was exhausted.": "FINISH ok",
            }
        )
    )
    task = routed_task(max_steps=20, harness_options={"child_steps": 2})
    collection, _ = asyncio.run(rollout(gateway, task))

    child = by_goal(collection)["goal:a"]
    assert len(child.steps) == 2 and child.misc["agent_result"]["status"] == "limit"
    assert child.error_message.startswith("WARNING: Exhausted budget")
    assert by_goal(collection)["goal:root"].finish_message == "ok"


def test_child_timeouts_and_crashes_come_back_as_failed_children(make_gateway):
    gateway, _ = make_gateway(
        rules(
            {
                "goal:root": "SPAWN goal:hang|goal:crash",
                "goal:hang": "HANG",
                "goal:crash": "CRASH",
                "obs: [timeout] Subagent failed before finishing. ; [error] Subagent failed before finishing.": (
                    "FINISH both failed"
                ),
            }
        )
    )
    task = routed_task(agents={"child_grading": "env", "child_timeout_s": 0.5})
    collection, interactions = asyncio.run(rollout(gateway, task))

    trajectories = by_goal(collection)
    assert trajectories["goal:root"].finish_message == "both failed" and trajectories["goal:root"].reward == 0.0
    assert trajectories["goal:hang"].misc["outcome"]["status"] == "timeout"
    assert trajectories["goal:crash"].misc["outcome"]["status"] == "error"
    assert all(trajectory_was_interrupted(trajectories[goal]) for goal in ("goal:hang", "goal:crash"))
    result = datums(collection, interactions)
    assert result.num_policy_excluded_datums == 2 and result.task_reward_valid


def test_a_root_timeout_cancels_its_running_children(make_gateway):
    gateway, _ = make_gateway(rules({"goal:root": "SPAWN goal:hang", "goal:hang": "HANG"}))
    collection, _ = asyncio.run(rollout(gateway, routed_task(), timeout=1))

    trajectories = by_goal(collection)
    root, child = trajectories["goal:root"], trajectories["goal:hang"]
    assert root.misc["outcome"]["status"] == "timeout" and trajectory_was_interrupted(root)
    assert child.misc["agent_result"]["status"] == "cancelled" and trajectory_was_cancelled(child)


def test_a_root_timeout_keeps_the_grades_of_children_that_finished(make_gateway):
    # The root times out after its child finished and was graded; the child keeps its grade.
    gateway, _ = make_gateway(
        rules(
            {
                "goal:root": "SPAWN goal:a",
                "goal:a": "SUBMIT ok-a",
                "obs: submitted": "FINISH a done",
                "obs: [completed] a done": "HANG",
            }
        )
    )
    collection, _ = asyncio.run(rollout(gateway, routed_task(agents={"child_grading": "env"}), timeout=2))

    trajectories = by_goal(collection)
    root, child = trajectories["goal:root"], trajectories["goal:a"]
    assert root.misc["outcome"]["status"] == "timeout" and trajectory_was_interrupted(root)
    assert child.misc["outcome"] == {
        "status": "scored",
        "rewards": {"reward": 1.0},
        "reason": None,
        "reward_key": "reward",
    }
    assert child.reward == 1.0 and child.finish_message == "a done"
    assert not trajectory_was_interrupted(child) and not trajectory_was_cancelled(child)


def test_concurrent_spawns_reserve_budget_atomically(make_gateway):
    gateway, sampler = make_gateway(
        rules(
            {
                "goal:root": "SPAWN goal:a|goal:b|goal:c",
                "goal:": "FINISH child done",
                "obs: [": "FINISH root done",
            }
        )
    )
    sampler.delays = {"goal:a": 0.2, "goal:b": 0.2}
    task = routed_task(max_steps=14, harness_options={"child_steps": 5})
    collection, _ = asyncio.run(rollout(gateway, task))

    trajectories = by_goal(collection)
    assert set(trajectories) == {"goal:root", "goal:a", "goal:b"}  # 13 left: two reservations of 6 fit
    spans = [trajectories[goal].misc["execution_span"] for goal in ("goal:a", "goal:b")]
    assert spans[0][0] < spans[1][1] and spans[1][0] < spans[0][1]  # they ran at the same time
    observation = sampler.calls[-1]
    assert observation.count("[completed] child done") == 2 and "[error] Not enough budget" in observation


def test_tool_access_filters_listing_and_calls(make_gateway):
    gateway, _ = make_gateway(rules({}))
    seen: dict[str, Any] = {}

    @pt.harness("toy-access-probe")
    async def probe(ctx: pt.AgentContext) -> pt.AgentResult:
        seen[ctx.access] = [tool.name for tool in await ctx.list_tools()]
        if ctx.depth == 0:
            for access in ("no_terminal", "read_only"):
                await ctx.agents.spawn(f"probe {access}", access=access)
        return pt.AgentResult("completed", final_message="probed")

    asyncio.run(rollout(gateway, routed_task(harness="toy-access-probe")))
    assert seen == {"full": ["submit", "look"], "no_terminal": ["look"], "read_only": ["look"]}


async def _http_judge(endpoint: pt.ModelEndpoint) -> str:
    async with aiohttp.ClientSession() as http:
        async with http.post(
            f"{endpoint.base_url}/chat/completions",
            json={"model": endpoint.model, "messages": [{"role": "user", "content": "judge: final"}]},
            headers=endpoint.headers("judge"),
        ) as response:
            assert response.status == 200, await response.text()
            return (await response.json())["choices"][0]["message"]["content"]


FINALIZED: list[str] = []


@pt.program("toy-finalized")
async def finalized_program(ep: pt.EpisodeContext) -> pt.Outcome:
    await ep.run()
    endpoint = ep.judge_endpoint()

    async def finalize(collection: pt.TrajectoryCollection) -> None:
        verdict = await _http_judge(endpoint)
        root = next(iter(collection.trajectories.values()))
        root.misc["finalized"] = verdict
        FINALIZED.append(verdict)

    ep.add_finalizer(finalize)
    ep.add_finalizer(lambda collection: FINALIZED.append("sync"))
    return await ep.grade()


def test_programs_finalize_the_collection_and_judges_never_train(make_gateway):
    gateway, _ = make_gateway(
        rules(
            {
                "goal:root": "AUX",
                "aux: summarize": "aux summary",
                "obs: aux said": "JUDGE",
                "judge: rate": "seven",
                "obs: judge said": "SUBMIT ok",
                "obs: submitted": "FINISH done",
                "judge: final": "fine",
            }
        )
    )
    collection, interactions = asyncio.run(rollout(gateway, routed_task(program="toy-finalized", max_steps=4)))

    (root,) = collection.trajectories.values()  # the judge agent has no trajectory
    assert FINALIZED[-2:] == ["fine", "sync"] and root.misc["finalized"] == "fine"
    kinds = [step.misc["message"]["content"] for step in root.steps]
    assert "aux summary" in kinds and "seven" not in kinds  # aux calls train, judge calls do not
    judged = [call["message"]["content"] for call in root.misc["judge_calls"]]
    assert sorted(judged) == ["fine", "seven"]
    # Four policy turns fit max_steps=4 because the aux and judge calls are free.
    assert root.finish_message == "done"
    result = datums(collection, interactions)
    assert result.task_reward == 1.0
    # Five steps (four policy turns and the aux call) prefix-merge into datums; the judge's tokens are in none.
    assert [stats.num_steps for stats in result.trajectory_stats] == [5]
    judge_ids = {call["call_id"] for call in root.misc["judge_calls"]}
    assert judge_ids and judge_ids.isdisjoint(step.misc["action_misc"]["completion_id"] for step in root.steps)


RECURSIVE_RULES = {
    "goal:root": "SPAWN goal:a|goal:b",
    "goal:a": "SUBMIT ok-a",
    "goal:b": "SUBMIT bad-b",
    "obs: [": "SUBMIT ok",
    "obs: submitted": "FINISH done",
}


def test_single_propagates_root_success_when_the_rollout_config_asks(make_gateway):
    def tree(**config: Any) -> dict[str, Any]:
        gateway, _ = make_gateway(rules(RECURSIVE_RULES))
        task = routed_task(agents={"child_grading": "env", "max_depth": 1})
        return by_goal(asyncio.run(rollout(gateway, task, **config))[0])

    graded = tree()
    assert {goal: t.reward for goal, t in graded.items()} == {"goal:root": 1.0, "goal:a": 1.0, "goal:b": 0.0}
    assert graded["goal:root"].steps[0].misc["reward_misc"]["reward/subagent_succeeded"] == 1.0

    propagated = tree(propagate_root_success=True)
    assert {goal: t.reward for goal, t in propagated.items()} == {"goal:root": 1.0, "goal:a": 1.0, "goal:b": 1.0}
    assert propagated["goal:b"].steps[-1].misc["reward_misc"]["reward/success"] == 1.0
    assert propagated["goal:root"].steps[0].misc["reward_misc"]["reward/subagent_succeeded"] == 2.0


def test_the_rollouts_inference_params_are_every_agents_default_sampling(make_gateway):
    gateway, sampler = make_gateway(rules(RECURSIVE_RULES))
    task = routed_task(agents={"child_grading": "env", "max_depth": 1})
    params = InferenceParams(temperature=0.25, top_p=0.75, max_completion_tokens=40)
    asyncio.run(rollout(gateway, task, inference_params=params))

    assert {"goal:root", "goal:a", "goal:b"} <= set(sampler.calls)  # the root's turns and each child's
    assert {(call.temperature, call.top_p, call.max_tokens) for call in sampler.sampling} == {(0.25, 0.75, 40)}


@pt.harness("toy-sampling-probe")
async def sampling_probe(ctx: pt.AgentContext) -> pt.AgentResult:
    await ctx.chat([{"role": "user", "content": "probe: without max_tokens"}])
    await ctx.chat([{"role": "user", "content": "probe: with max_tokens"}], max_tokens=11)
    return pt.AgentResult("completed", final_message="probed")


@pytest.mark.parametrize(
    ("inference_params", "budget"),
    [({}, 300), ({"max_completion_tokens": 40}, 40)],
    ids=["unset-gateway-default", "rollout-budget"],
)
def test_inline_calls_get_only_the_sampling_the_rollout_sets(make_gateway, inference_params, budget):
    # The gateway's own defaults are not 1 here, so an InferenceParams default that leaked would show.
    gateway, sampler = make_gateway(rules({}), max_tokens=300, temperature=0.7, top_p=0.8)
    task = routed_task(harness="toy-sampling-probe")
    asyncio.run(rollout(gateway, task, inference_params=InferenceParams(**inference_params)))

    without, own = ((call.max_tokens, call.temperature, call.top_p) for call in sampler.sampling)
    assert without == (budget, 0.7, 0.8)
    assert own == (11, 0.7, 0.8)


PROPAGATIONS: list[str] = []


def counting_propagation(collection: pt.TrajectoryCollection) -> None:
    PROPAGATIONS.append(collection.id)


@pt.program("toy-wraps-single")
async def wraps_single(ep: pt.EpisodeContext) -> pt.Outcome:
    ep.add_finalizer(counting_propagation)
    outcome = await get_program("single")(ep)
    ep.add_finalizer(counting_propagation)
    return outcome


def test_a_finalizer_added_twice_runs_once(make_gateway):
    PROPAGATIONS.clear()
    gateway, _ = make_gateway(rules({"goal:root": "SUBMIT ok", "obs: submitted": "FINISH done"}))
    collection, _ = asyncio.run(rollout(gateway, routed_task(program="toy-wraps-single")))
    assert PROPAGATIONS == [collection.id]


@pt.program("toy-broken")
async def broken_program(ep: pt.EpisodeContext) -> pt.Outcome:
    await ep.run()
    raise RuntimeError("program bug")


def test_a_failing_program_is_an_error_outcome(make_gateway):
    gateway, _ = make_gateway(rules({"goal:root": "FINISH done"}))
    collection, _ = asyncio.run(rollout(gateway, routed_task(program="toy-broken")))
    (root,) = collection.trajectories.values()
    assert root.misc["outcome"]["status"] == "error" and trajectory_was_interrupted(root)


def test_a_context_overflow_is_the_reason_even_when_the_grader_gives_its_own(make_gateway):
    # The policy filled its context window, so the harness crashed: graded as a policy failure whose reason is
    # context_overflow, while the grader's own reason (no_submit) stays in the trajectory's outcome misc.
    ToyService.grade_reason = "no_submit"
    gateway, sampler = make_gateway(rules({"goal:root": "FINISH never"}), context_window_length=4)
    collection, _ = asyncio.run(rollout(gateway, routed_task()))

    (root,) = collection.trajectories.values()
    assert (
        root.misc["agent_result"]["status"] == "error" and "ContextWindowExceeded" in root.misc["agent_result"]["error"]
    )
    assert root.misc["outcome"] == {
        "status": "scored",
        "rewards": {"reward": 0.0},
        "reason": "context_overflow",
        "grade_reason": "no_submit",
        "reward_key": "reward",
    }
    assert not sampler.calls  # the prompt never reached the sampler


def test_a_grader_that_raises_fails_closed(make_gateway):
    ToyService.grade_raises = True
    gateway, _ = make_gateway(rules({"goal:root": "FINISH done"}))
    collection, _ = asyncio.run(rollout(gateway, routed_task()))
    (root,) = collection.trajectories.values()
    assert root.misc["outcome"]["status"] == "error" and root.reward == 0.0


def test_a_fork_failure_is_a_failed_launch_not_a_crash(make_gateway):
    ToyService.fork_raises = True
    gateway, _ = make_gateway(rules({"goal:root": "SPAWN goal:a", "obs: [error] Failed to launch": "FINISH ok"}))
    collection, _ = asyncio.run(rollout(gateway, routed_task()))
    (root,) = collection.trajectories.values()
    assert root.finish_message == "ok"


def test_spawning_a_toml_harness_is_a_clear_error(make_gateway):
    gateway, _ = make_gateway(rules({}))
    errors: list[str] = []

    @pt.harness("toy-toml-spawner")
    async def spawner(ctx: pt.AgentContext) -> pt.AgentResult:
        try:
            await ctx.agents.spawn("x", harness="platoon-bash")
        except TypeError as exc:
            errors.append(str(exc))
        return pt.AgentResult("completed")

    asyncio.run(rollout(gateway, routed_task(harness="toy-toml-spawner")))
    assert errors and "TOML harness spec" in errors[0]


def test_verifier_access_follows_the_config(make_gateway):
    rules_no_terminal = dict(VERIFIER_RULES)
    rules_no_terminal["You are a verifier agent"] = "LOOK"
    gateway, _ = make_gateway(rules(rules_no_terminal))
    task = routed_task(agents={"child_grading": "verifier", "verifier": {"access": "no_terminal"}})
    collection, _ = asyncio.run(rollout(gateway, task))
    verifier = list(collection.trajectories.values())[2]
    assert verifier.misc["agent"]["role"] == "verifier"
    assert [call["name"] for call in verifier.misc["tool_calls"]] == ["look"]
