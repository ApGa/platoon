"""RAO parity, by value, on CPU: a serialized single-start tree through the bridge trains as its native counterpart, a
function harness that makes the same model calls with ``ctx.chat`` and spawns the same goals, under the RAO recipe's
settings (``recipes/rao``); both reproduce the group credit's golden table in both modes.  A mesh is centred on its
zero-call root; a serialized tree spends its steps where a native one does, under a linear budget too; and a compaction
is ``aux``, outside the budget, and a row of its own.

Serialized means the parent makes no model call between a spawn and its child's report, as only then do the positional
stamps of a spawn (``fork_step``) agree.  The two harnesses render different prompts, so prompt tokens are not
compared, and a concurrent tree has no native counterpart."""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from platoon_harness import Session
from platoon_harness.testing import AskModel, Call, Compact, Say
from teams_toys import LEDGER, SERVED, UNLOCK, named, prove, rollout, scripted, session_name, team, wayward
from testing_toys import LOCKBOX

import platoon as pt
from platoon.gateway.core import BUDGET_STOP_PREFIX
from platoon.harness.builtin.platoon_harness import bridge
from platoon.harness.builtin.platoon_harness.authority import PlatoonAuthority
from platoon.harness.builtin.platoon_harness.finalize import Mismatch, finalize
from platoon.rewards import delegation_capped
from platoon.testing import ScriptedPolicy, run_episode, tool_call
from platoon.train.config_defs import WorkflowConfig
from platoon.train.credit.group import group_baselines
from platoon.train.step.batch_transforms import DepthLevelWeightingTransform
from platoon.train.workflow.group import GroupRolloutWorkflow
from platoon.train.workflow.output import GroupOutput
from platoon.utils.stats_tracker import get as get_tracker
from platoon.utils.tinker_data_processing import get_train_data_for_trajectory_collection
from platoon.utils.trajectory_status import (
    TRAJECTORY_CANCELLED_MISC_KEY,
    TRAJECTORY_INVALID_MISC_KEY,
    TRAJECTORY_TIMED_OUT_MISC_KEY,
)

NATIVE = "teamsparity-native"
PART = "Score your part of the work."
SPAWN = Call("agent_spawn", {"name": "w1", "child_role": "worker", "goal": PART})
WORKER = scripted([AskModel(), Call("agent_finish", {"gist": "scored"})])
# The recipe's tree: 25 steps for every agent, at most 6 levels deep, each child graded on its own part.
AGENTS = {"max_depth": 6, "budget": "depth_aware", "child_max_steps": 25, "child_grading": "env"}
MARKERS = (TRAJECTORY_INVALID_MISC_KEY, TRAJECTORY_TIMED_OUT_MISC_KEY, TRAJECTORY_CANCELLED_MISC_KEY)
SCORE = f"{SERVED}score"


@pt.harness(NATIVE)
async def native(ctx: pt.AgentContext) -> pt.AgentResult:
    """A scripted tree's native counterpart: it asks the model, runs each tool the model calls and asks again, until the
    model only talks; ``DELEGATE`` spawns a worker on its part and hands the worker's report back.  It names the
    environment's tools to the model as the bridge's ScriptedDriver does, so the model says the same to both.  It stops
    at ``limit`` once its budget or its steps are spent, as platoon's own harnesses do."""

    listed = await ctx.list_tools()
    tools = [{"type": "function", "function": {"name": SERVED + t.name, "parameters": t.input_schema}} for t in listed]
    messages: list[dict[str, Any]] = [{"role": "user", "content": ctx.instruction}]
    steps = 0
    while True:
        if ctx.budget.remaining() <= 0 or (ctx.max_steps is not None and steps >= ctx.max_steps):
            return pt.AgentResult("limit")
        reply = await ctx.chat(messages, tools=tools)
        steps += 1
        messages.append(dict(reply.message))
        for call in reply.tool_calls:
            name = call["function"]["name"].removeprefix(SERVED)
            result = await ctx.call_tool(name, json.loads(call["function"]["arguments"]))
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result.as_text()})
        if reply.tool_calls:
            continue
        if reply.text != "DELEGATE":
            return pt.AgentResult("completed", final_message=reply.text)
        child = await ctx.agents.spawn(PART, harness=NATIVE, role="worker")
        messages.append({"role": "user", "content": child.message or ""})


@dataclass(frozen=True)
class Member:
    """One rollout of a scenario: the root's score, whether its environment declares that grade invalid, and the score
    of the one worker it delegates its part to, if it does."""

    root: float
    child: float | None = None
    invalid: bool = False

    def script(self) -> list[str]:
        """What the model says, in order: the root delegates, the worker scores its part, and the root scores."""

        scored = [tool_call(SCORE, value=self.root, invalid=self.invalid), "done"]
        return scored if self.child is None else ["DELEGATE", tool_call(SCORE, value=self.child), "done", *scored]


ALONE = named(team(lead={"harness": scripted([AskModel()]), "may_spawn": ["worker"]}, worker={"harness": WORKER}))
DELEGATING = named(
    team(
        lead={"harness": scripted([AskModel(), SPAWN], [AskModel()]), "may_spawn": ["worker"]},
        worker={"harness": WORKER},
    )
)


@pytest.fixture
def proven(_sdk_home: Path) -> Path:
    prove(_sdk_home)
    return _sdk_home


def both(member: Member, **rollout_settings: Any) -> tuple[Any, Any]:
    """``member`` rolled out through the bridge and through its native counterpart, on the ledger, with the recipe's
    agents and ``rollout_settings``."""

    settings: dict[str, Any] = {
        "task": "test:0",
        "service": LEDGER,
        "agents": AGENTS,
        "max_steps": 25,
        "rollout": rollout_settings,
    }
    tree = ALONE if member.child is None else DELEGATING
    bridged = rollout(tree, ScriptedPolicy(member.script()), **settings)
    settings["policy"] = ScriptedPolicy(member.script())
    nativ = run_episode(settings.pop("task"), harness=NATIVE, **settings)
    return asyncio.run(bridged), asyncio.run(nativ)


def converted(run: Any) -> Any:
    """A rollout's rows as a recipe's worker converts them: under ``platoon-delegation``, with no error filter, each
    datum with its depth and whether it starts its trajectory, and tagged with its agent."""

    return get_train_data_for_trajectory_collection(
        run.collection.to_dict(),
        run.interactions,
        "test:0",
        0,
        reward_processor=delegation_capped,
        filter_errors=False,
        include_traj_depth=True,
        include_traj_start=True,
        tag_datums=True,
    )


def shape(run: Any) -> dict[str, Any]:
    """What parity compares of one rollout, by position and with no prompt token: each trajectory's parent, fork step,
    reward, markers and tag; each datum's action tokens, depth and start, its reward before credit, and the rows of each
    trajectory."""

    trajectories = run.trajectories
    at = {trajectory.id: index for index, trajectory in enumerate(trajectories)}
    result = converted(run)
    return {
        "trajectories": [
            (
                None if t.parent_info is None else (at[t.parent_info.id], t.parent_info.fork_step),
                t.reward,
                sorted(marker for marker in MARKERS if t.misc.get(marker)),
            )
            for t in trajectories
        ],
        "tags": [
            (tag.role, None if tag.parent_id is None else at[tag.parent_id], tag.depth, tag.reward)
            for tag in result.agents
        ],
        "stats": [
            (s.reward, s.num_steps, s.num_output_tokens, s.num_datums, s.is_root) for s in result.trajectory_stats
        ],
        "task": (result.task_reward, result.task_reward_valid, result.root_rewards_dict),
        "datums": [
            (
                at[trajectory],
                int(datum.loss_fn_inputs["mask"].to_torch().sum()),
                int(datum.loss_fn_inputs["traj_depth"].to_torch().item()),
                float(datum.loss_fn_inputs["traj_start"].to_torch().item()),
                _actions(datum),
            )
            for trajectory, datum in zip(result.datum_trajectories, result.datums, strict=True)
        ],
    }


def _actions(datum: Any) -> list[float]:
    """The advantage of each of a datum's action tokens, rounded to float32's tolerance."""

    mask = datum.loss_fn_inputs["mask"].to_torch().bool()
    return [round(value, 5) for value in datum.loss_fn_inputs["advantages"].to_torch()[mask].tolist()]


def credited(results: list[Any], **settings: Any) -> dict[str, Any]:
    """The group credit of ``results``, a group's members as their workers converted them, as the recipe's workflow
    gives it: each trajectory's advantage, each datum's advantages, the depth weights its batch would get, the verdict
    and the ``credit/*`` metrics."""

    workflow = GroupRolloutWorkflow(
        rollout_fn=None,
        get_task_fn=lambda task_id: SimpleNamespace(id=task_id, max_steps=None),
        config=WorkflowConfig(group_size=len(results), depth_level_weighting=True, **settings),
        model_info=SimpleNamespace(model_name="m", base_url=None, api_key=None, llm=SimpleNamespace(version=0)),
        stats_scope="test_teams_parity",
    )

    async def member(data: dict, rollout_number: int = 0) -> Any:
        return copy.deepcopy(results[rollout_number])

    workflow.arun_episode_single = member  # type: ignore[method-assign]
    output = asyncio.run(workflow.arun_episode({"task_id": "test:0"}))
    assert isinstance(output, GroupOutput)
    weighted = DepthLevelWeightingTransform()(copy.deepcopy(list(output)), None) or []  # type: ignore[arg-type]
    tracker = get_tracker("test_teams_parity")
    tracker.export(reset=True)
    output.group_stats.record(tracker)
    advantages = [_actions(datum) for datum in output]
    owners = [(member, trajectory) for member, result in enumerate(results) for trajectory in result.datum_trajectories]
    by_trajectory: dict[tuple[int, str], set[float]] = {}
    for owner, found in zip(owners if output else [], advantages, strict=True):
        by_trajectory.setdefault(owner, set()).update(found)
    return {
        "trajectories": [values.pop() if len(values) == 1 else sorted(values) for values in by_trajectory.values()],
        "advantages": advantages,
        "weights": [_weight(before, after) for before, after in zip(output, weighted, strict=True)],
        "verdict": output.verdict,
        "credit": {key: value for key, value in tracker.export(reset=True).items() if "/credit/" in key},
    }


def _weight(before: Any, after: Any) -> float | None:
    """The depth weight a datum's advantages were scaled by, None for a datum with no advantage to scale."""

    scaled = before.loss_fn_inputs["advantages"].to_torch()
    nonzero = scaled != 0
    if not bool(nonzero.any()):
        return None
    return round(float((after.loss_fn_inputs["advantages"].to_torch()[nonzero] / scaled[nonzero]).mean()), 5)


# The golden table: each scenario's members and other workflow settings, and in each mode, leave-one-out or the
# group's mean, the baselines, the advantage of each trajectory that has rows, and the verdict.  An invalid root's score
# never enters a baseline, its rows are dropped, and its child is centred on the mean of the completed roots.  A
# scenario with at most one completed root, or none, has one answer in both modes.
MODES = {"loo": True, "mean": False}
GROUP = (Member(1.0, child=0.0), Member(0.0), Member(1.0), Member(0.9, child=1.0, invalid=True))


def alike(golden: tuple[Any, ...]) -> dict[str, tuple[Any, ...]]:
    return dict.fromkeys(MODES, golden)


GOLDEN = {
    "group": (
        GROUP,
        {},
        {
            "loo": ([0.5, 1.0, 0.5, 2 / 3], [0.5, -0.5, -1.0, 0.5, 1 / 3], "trained"),
            "mean": ([2 / 3] * 4, [1 / 3, -2 / 3, -2 / 3, 1 / 3, 1 / 3], "trained"),
        },
    ),
    "g1": ((Member(0.7, child=0.2),), {}, alike(([0.7], [0.0, -0.5], "trained"))),
    "singleton": (
        (Member(1.0, child=0.0), Member(0.4, child=1.0, invalid=True)),
        {},
        alike(([1.0, 1.0], [0.0, -1.0, 0.0], "trained")),
    ),
    "no-valid-root": (
        (Member(0.9, invalid=True), Member(0.4, invalid=True)),
        {},
        alike((None, [], "no_completed_root")),
    ),
    "below-min": (
        (Member(1.0), Member(0.9, invalid=True)),
        {"min_successful_group_size": 2},
        alike((None, [], "below_min")),
    ),
    "uniform": ((Member(1.0), Member(1.0)), {}, alike(([1.0, 1.0], [0.0, 0.0], "uniform"))),
}
# The flag cases: the group in leave-one-out under each of the settings single's finalizers read, which the golden table
# does not cover.
FLAGS = ("propagate_root_success", "skip_subagent_reward_computation")


@pytest.mark.parametrize("case", [*(f"{scenario}-{mode}" for scenario in GOLDEN for mode in MODES), *FLAGS])
def test_a_serialized_bridge_tree_trains_as_its_native_counterpart(proven, case):
    scenario, _, mode = ("group", "-", "loo") if case in FLAGS else case.rpartition("-")
    members, settings, golden = GOLDEN[scenario]
    baselines, advantages, verdict = golden[mode]
    flags = {case: True} if case in FLAGS else {}

    runs = [both(member, **flags) for member in members]

    results = {side: [converted(pair[index]) for pair in runs] for index, side in enumerate(("bridge", "native"))}
    assert [shape(bridged) for bridged, _ in runs] == [shape(nativ) for _, nativ in runs]
    leave_one_out = MODES[mode]
    group = {side: credited(found, leave_one_out_baseline=leave_one_out, **settings) for side, found in results.items()}
    assert group["bridge"] == group["native"]
    if case in FLAGS:
        return
    assert group["bridge"]["verdict"] == verdict
    assert group["bridge"]["trajectories"] == pytest.approx(advantages, abs=1e-5)
    if baselines is not None:
        rewards = [(result.task_reward, result.task_reward_valid) for result in results["bridge"]]
        found = group_baselines(*map(list, zip(*rewards, strict=True)), leave_one_out=leave_one_out)
        assert found == pytest.approx(baselines)


MESH = named(team(alpha={"harness": scripted([AskModel()])}, beta={"harness": scripted([Say("standing by")])}))


def test_a_mesh_is_centred_on_its_zero_call_root(proven):
    """Two meshes of minted peers, one scoring 1 and one 0: each zero-call root is a valid root with no rows, and the
    peer that has rows is centred on the root baseline, whose reward the minted peers share.  It pins how depth
    weighting and ``traj_start`` treat a root with no rows: every row is at depth 1, ``traj_start`` marks each peer's
    first, and depth 1 alone weighs every row by 1.  Each peer is tagged with its own role, as ``credit: keyed``
    reads it."""

    scripts = [[tool_call(SCORE, value=value), "done"] for value in (1.0, 0.0)]
    runs = [asyncio.run(rollout(MESH, ScriptedPolicy(script), task="test:0", service=LEDGER)) for script in scripts]

    results = [converted(run) for run in runs]
    group = credited(results)
    for result, reward in zip(results, (1.0, 0.0), strict=True):
        root = result.trajectory_stats[0]
        assert (root.is_root, root.num_steps, root.num_datums) == (True, 0, 0)
        assert (result.task_reward, result.task_reward_valid) == (reward, True)
        assert [stats.reward for stats in result.trajectory_stats] == [reward] * 3
    assert [(depth, start) for run in runs for _, _, depth, start, _ in shape(run)["datums"]] == [
        (1, 1.0),
        (1, 0.0),
    ] * 2
    assert group["trajectories"] == [0.5, -0.5]
    assert group["weights"] == [1.0] * 4
    # What `credit: keyed` keys each agent by and centres: its role, depth and reward.
    tags = [[(tag.role, tag.depth, tag.reward) for tag in result.agents] for result in results]
    assert tags == [[("policy", 0, reward), ("alpha", 1, reward), ("beta", 1, reward)] for reward in (1.0, 0.0)]


LOOPING = named(
    team(
        lead={"harness": scripted([AskModel(), SPAWN], [AskModel()]), "may_spawn": ["worker"]},
        worker={"harness": scripted()},
    )
)
WRONG = tool_call(UNLOCK, code="9")


def looping(script: list[str], **settings: Any) -> tuple[Any, Any]:
    """A tree on the lockbox whose worker tries a wrong code until it is stopped, through the bridge and natively, with
    the recipe's agents and ``settings``."""

    settings = {"agents": {**AGENTS, **settings.pop("agents", {})}, **settings}
    bridged = asyncio.run(rollout(LOOPING, ScriptedPolicy(script), **settings))
    nativ = asyncio.run(
        run_episode("test:1", service=LOCKBOX, harness=NATIVE, policy=ScriptedPolicy(script), **settings)
    )
    return bridged, nativ


def stopped_at(run: Any) -> list[int | None]:
    """Where each agent of a bridge rollout was first answered with a budget stop: the number of its calls before it."""

    async def stops() -> list[int | None]:
        async with Session.attach(session_name(run)) as ended:
            calls: dict[str | None, list[str]] = {}
            async for event in ended.events(match="turn/*"):
                if event.type == "model.called":
                    calls.setdefault(event.agent, []).append(event.data["call_id"])
        return [
            next((n for n, call in enumerate(made) if call.startswith(BUDGET_STOP_PREFIX)), None)
            for made in calls.values()
        ]

    return asyncio.run(stops())


def test_a_serialized_tree_spends_its_budget_where_a_native_one_does(proven):
    """A worker of two steps (``agents.child_max_steps``) runs past them: the bridge's worker gets its first budget
    stop at the step where the native worker's ``out_of_steps`` returned ``limit``, nothing past it is sampled, and the
    bridge retires it, so its run grades ``limit`` with reason ``budget``."""

    bridged, nativ = looping(["DELEGATE", WRONG, WRONG, "done"], agents={"child_max_steps": 2})

    steps = [[len(trajectory.steps) for trajectory in run.trajectories] for run in (bridged, nativ)]
    worker = bridged.trajectories[1]
    assert steps[0] == steps[1] == [2, 2]
    assert stopped_at(bridged) == [None, 2] and len(bridged.policy.calls) == len(nativ.policy.calls) == 4
    assert [run.trajectories[1].misc["agent_result"]["status"] for run in (bridged, nativ)] == ["limit", "limit"]
    assert (worker.misc["outcome"]["reason"], worker.misc["budget"]["stops"]) == ("budget", 1)


def test_a_serialized_tree_spends_a_linear_budget_where_a_native_one_does(proven, monkeypatch):
    """Under ``agents.budget: linear`` a child's reservation, its steps and one more, is its parent's until platoon's
    spawn of it returns, and here that spawn returns as late as it can: once the lead has called again, unless the
    authority's ``finish`` of the worker waits on it.  The session tells the lead nothing of its worker's end until that
    ``finish`` has returned, which is once platoon's spawn has, so the lead spends the two steps its worker left, as the
    native lead does, and gets its first budget stop where the native lead's budget returned ``limit``; nothing past a
    stop is sampled."""

    finish, member = PlatoonAuthority.finish, bridge._member
    finishing: set[str] = set()
    early: list[str] = []  # each agent whose finish returned before platoon's spawn of it

    async def told(self: PlatoonAuthority, agent: Any, outcome: Any) -> None:
        finishing.add(agent.agent)
        try:
            await finish(self, agent, outcome)
        finally:
            finishing.discard(agent.agent)
        if (spawned := self._spawns.get(agent.agent)) is not None and not spawned.done():
            early.append(agent.agent)

    async def late(ctx: Any, given: Any, agent: str) -> Any:
        head = await given.session.head()
        result = await member(ctx, given, agent)
        if agent not in finishing:  # nothing holds its end from the lead: platoon's spawn returns once it has called
            async for event in given.session.events(head, streams=["turn/lead"], follow=True):
                if event.type == "model.called":
                    break
        return result

    monkeypatch.setattr(PlatoonAuthority, "finish", told)
    monkeypatch.setattr(bridge, "_member", late)
    bridged, nativ = looping(
        ["DELEGATE", WRONG, WRONG, WRONG, WRONG], agents={"budget": "linear", "child_max_steps": 2}, max_steps=5
    )

    lead, worker = bridged.trajectories
    assert early == []
    assert [len(trajectory.steps) for trajectory in bridged.trajectories] == [3, 2]
    assert [len(trajectory.steps) for trajectory in nativ.trajectories] == [3, 2]
    assert stopped_at(bridged) == [3, 2] and len(bridged.policy.calls) == len(nativ.policy.calls) == 5
    assert [run.root.misc["agent_result"]["status"] for run in (bridged, nativ)] == ["limit", "limit"]
    for trajectory in (lead, worker):
        assert (trajectory.misc["agent_result"]["status"], trajectory.misc["outcome"]["reason"]) == ("limit", "budget")


@pytest.fixture
def mismatches(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Mismatch, ...]]:
    """What the finalizer found in each of the test's rollouts."""

    found: list[tuple[Mismatch, ...]] = []

    async def finding(collection: Any, name: str, producers: Any) -> tuple[Mismatch, ...]:
        found.append(await finalize(collection, name, producers))
        return found[-1]

    monkeypatch.setattr(bridge, "finalize", finding)
    return found


COMPACTING = [AskModel(), Compact(), AskModel(), Call(SCORE, {"value": 1.0})]


def noting(prompt: str) -> str:
    """A model that notes the work, and summarizes it the same way."""

    return "noted"


def test_a_compaction_is_aux_outside_the_budget_and_splits_its_rows(proven, mismatches):
    """A scripted agent of two steps notes the work, compacts and notes it again: its compaction is ``aux``, joined to
    the call it reported, and outside its steps; it trains on the trajectory's reward as a row of its own, and the next
    call starts another.  Sent without the headers, the compaction is counted, so the agent's next call is a budget
    stop, and named a mismatch; reported with no id, it is excluded, and no row holds its tokens."""

    runs = {
        harness: asyncio.run(
            rollout(named(team(solver={"harness": harness})), noting, task="test:0", service=LEDGER, max_steps=2)
        )
        for harness in (
            scripted(COMPACTING),
            wayward(COMPACTING, headers=False),
            wayward(COMPACTING, rewrite=_unreported),
        )
    }
    (sent, headless, unreported), (found, counted, unjoined) = runs.values(), mismatches

    calls = completions(sent.root)
    rows = converted(sent)
    assert [step.misc["action_misc"].get("kind", "policy") for step in sent.root.steps] == ["policy", "aux", "policy"]
    assert (sent.root.misc["budget"]["stops"], found) == (0, ())
    assert [[call for call, _, _ in merged] for merged in rows.datum_calls] == [[call] for call in calls]
    assert _actions(rows.datums[1]) == [1.0] * len(_actions(rows.datums[1]))
    assert headless.root.misc["budget"]["stops"] == 1 and counted == (
        Mismatch("solver", (), (), (completions(headless.root)[1],), ()),
    )
    held = completions(unreported.root)[1]
    assert unjoined == (Mismatch("solver", (), (held,), (), ()),)
    assert held not in {call for merged in converted(unreported).datum_calls for call, _, _ in merged}


def _unreported(call: Any) -> Any:
    """A compaction reported with no id, as a harness that keeps none reports it."""

    return call.model_copy(update={"call_id": None}) if call.purpose == "compaction" else call


def completions(trajectory: Any) -> list[str]:
    return [step.misc["action_misc"]["completion_id"] for step in trajectory.steps]
