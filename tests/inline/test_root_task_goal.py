"""The root trajectory records the goal its session ran.  An ORS task gets its goal (the env's prompt) only when its
session opens, and the trace used to keep the unopened task's None: the viewer, ``platoon rollout``, finalizers and
the parent-usage judge saw no goal.  A goal the task or its harness set stays, and a subagent's task records the
goal among its parent tasks."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import inline_fakes  # noqa: F401  (registers toy-agent)
from test_function_harness import rollout, rules

import platoon as pt
from platoon.config_defs import RolloutConfig
from platoon.harness.rollout import EmbeddedGateway, run_harness_rollout
from platoon.harness.runner import HarnessRunner
from platoon.task import Task
from platoon.train.components import EnvironmentRoute

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "envs"))  # ors_toys, imported by the service

COUNTER = "python://ors_toys:CounterEnv"  # task train:0's prompt is "Reach 2."
SNAPSHOTS: list = []


def ors_task(harness: str, **route) -> Task:
    route = EnvironmentRoute(name="counter", service=COUNTER, harness=harness, resolve_task=True, **route)
    return route.attach(Task(id="train:0", max_steps=5))


def test_an_ors_tasks_goal_reaches_the_root_trajectory(make_gateway):
    gateway, sampler = make_gateway(rules({"Reach": "FINISH done"}))

    collection, _ = asyncio.run(rollout(gateway, ors_task("toy-agent")))

    (root,) = collection.trajectories.values()
    assert sampler.calls == ["Reach 2."]  # the instruction the harness got: the env's prompt
    assert (root.task.id, root.task.goal) == ("train:0", "Reach 2.")
    assert root.task.misc["ors"]["split"] == "train"


def test_a_subagents_task_records_the_goal_its_parents_session_ran(make_gateway):
    """A subagent's task is a fork of its parent's, made while the parent runs: its parent tasks name the goal the
    root's session ran, as a Harbor or table task's do, not the unopened ORS task's None."""

    policy = rules({"Reach": "SPAWN count to two", "count": "FINISH counted", "obs": "FINISH done"})
    gateway, _ = make_gateway(policy)

    collection, _ = asyncio.run(rollout(gateway, ors_task("toy-agent", harness_options={"child_steps": 2})))

    root, child = collection.trajectories.values()
    assert child.task.goal == "count to two"
    assert [task.goal for task in child.task.parent_tasks] == ["Reach 2."] == [root.task.goal]


@pt.program("root-goal-snapshot")
async def snapshot_after_the_run(ep: pt.EpisodeContext) -> pt.Outcome:
    await ep.run()
    SNAPSHOTS.append(ep.snapshot())
    return await ep.grade()


def test_a_snapshot_records_the_root_goal_as_the_result_does(make_gateway):
    SNAPSHOTS.clear()
    gateway, _ = make_gateway(rules({"Reach": "FINISH done"}))

    collection, _ = asyncio.run(rollout(gateway, ors_task("toy-agent", program="root-goal-snapshot")))

    (snapshot,) = SNAPSHOTS
    assert next(iter(snapshot.trajectories.values())).task.goal == "Reach 2."
    assert next(iter(collection.trajectories.values())).task.goal == "Reach 2."


def test_a_spec_harness_rollout_records_the_goal_too(make_gateway, tmp_path):
    spec = tmp_path / "noop.toml"
    spec.write_text('name = "noop"\nversion = "1"\n[launch]\nargv = ["true"]\n')
    gateway, _ = make_gateway(rules({}))
    runner = HarnessRunner(logs_dir=tmp_path / "logs", state_root=tmp_path / "state", kill_grace_s=0.2)
    config = RolloutConfig(output_dir=str(tmp_path / "out"), verbose=False)

    collection = asyncio.run(
        run_harness_rollout(ors_task(str(spec)), config, gateway=EmbeddedGateway(gateway), runner=runner)
    )

    (root,) = collection.trajectories.values()
    assert (root.task.id, root.task.goal) == ("train:0", "Reach 2.")


@pt.harness("root-goal-setter")
async def goal_setter(ctx: pt.AgentContext) -> pt.AgentResult:
    ctx.task.goal = "the harness's own goal"  # as a harness that composes its own goal records it
    return pt.AgentResult("completed", final_message="ok")


def test_a_goal_the_harness_set_on_its_task_stays(make_gateway):
    gateway, _ = make_gateway(rules({}))

    collection, _ = asyncio.run(rollout(gateway, ors_task("root-goal-setter")))

    (root,) = collection.trajectories.values()
    assert root.task.goal == "the harness's own goal"


def test_a_tasks_own_goal_stays(make_gateway):
    gateway, sampler = make_gateway(rules({"Count": "FINISH done"}))
    route = EnvironmentRoute(name="counter", service=COUNTER, harness="toy-agent", resolve_task=False)

    collection, _ = asyncio.run(rollout(gateway, route.attach(Task(id="train:0", goal="Count to two.", max_steps=5))))

    (root,) = collection.trajectories.values()
    assert sampler.calls == ["Count to two."] and root.task.goal == "Count to two."


def test_an_empty_goal_records_the_goal_its_session_ran(make_gateway):
    # The ORS session treats an empty goal as none (it runs the env's prompt), and so does the record.
    gateway, sampler = make_gateway(rules({"Reach": "FINISH done"}))
    route = EnvironmentRoute(name="counter", service=COUNTER, harness="toy-agent", resolve_task=False)

    collection, _ = asyncio.run(rollout(gateway, route.attach(Task(id="train:0", goal="", max_steps=5))))

    (root,) = collection.trajectories.values()
    assert sampler.calls == ["Reach 2."] and root.task.goal == "Reach 2."
