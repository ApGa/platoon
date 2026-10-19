"""Scripted CodeAct rollouts of recursive agent trees on TextCraft-Synth, through the real gateway, the TextCraft
environment and core's inline runtime, as ``tests/harness/builtin/test_codeact_rollouts.py`` runs number search.

A scripted policy answers each model call from its decoded prompt.  Children share their parent's inventory
(``platoon_fork``) and are graded on their own targets (``child_grading: env``).  The last test runs the recursive
arm of ``recipes/rao`` and its Sparse ablation through ``harness_rollout``.
"""

from __future__ import annotations

import asyncio
import re

import pytest
from inline_runs import open_service_task, run_tree, train_datums
from platoon.rewards import zero_subagent_rewards
from platoon.utils.subagent_rewards import propagate_root_success
from platoon.utils.trajectory_status import trajectory_was_cancelled

TEXTCRAFT_SYNTH = "python://platoon_textcraft.env:TextCraftSynthEnvironment"
TEXTCRAFT_RECURSIVE_OPTIONS = {
    "prompt_builder": "platoon_textcraft.prompts:TextCraftPromptBuilder",
    "prompt_builder_options": {"variant": "recursive", "synth": True},
}


def _gold_cell(spec) -> str:
    crafts = [
        f"print(craft({step['ingredients']!r}, ({step['target'][0]!r}, {step['result_count']})))"
        for step in spec["misc"]["gold_trajectory"]
    ]
    return "\n".join(crafts)


def _textcraft_policy(spec):
    ((target, count),) = spec["misc"]["target_items"].items()

    def policy(prompt: str) -> str:
        goal = re.findall(r"Your Goal: (.*?)\nBudget: You have a total budget of (\d+) steps", prompt, re.S)[-1]
        done = "[Cell 0 Output]" in prompt
        if goal == (f"Craft the following items: {count}x {target}", "200"):
            if not done:
                return (
                    "<thought>Delegate the target and a hopeless item in parallel.</thought>\n<python>"
                    f"results = await asyncio.gather(launch_subagent({{{target!r}: {count}}}, 30), "
                    "launch_subagent({'zz_nope': 1}, 5, 'this one cannot work'))\nprint(results)</python>"
                )
            return f"<thought>Both returned.</thought>\n<python>finish('crafted {count} {target}')</python>"
        if goal[0].startswith("Craft the following items: 1x zz_nope"):
            return "<thought>Impossible.</thought>\n<python>finish('cannot craft zz_nope')</python>"
        if not done:
            return f"<thought>Replay the recipe chain.</thought>\n<python>{_gold_cell(spec)}</python>"
        return "<thought>Done.</thought>\n<python>finish('done')</python>"

    return policy, target, count


@pytest.mark.asyncio
async def test_recursive_textcraft_synth_tree_trains_every_agent(make_gateway):
    from platoon_textcraft.env import TextCraftSynthEnvironment

    spec = TextCraftSynthEnvironment.list_tasks("train-medium")[0]
    policy, target, count = _textcraft_policy(spec)
    gateway, sampler = make_gateway(policy)
    task, session = await open_service_task(TEXTCRAFT_SYNTH, "train-medium:0")
    agents = {"budget": "linear", "child_grading": "env"}

    run = await run_tree(
        gateway, "platoon-codeact", session, task, max_steps=200, options=TEXTCRAFT_RECURSIVE_OPTIONS, agents=agents
    )

    assert run.result.status == "completed" and run.outcome.reward == 1.0
    root, children = run.root, run.children
    assert len(children) == 2 and all(child.parent_info.id == root.id for child in children)
    # Children fork after the root's first call (the gateway counts the call that launched them).
    assert {child.parent_info.fork_step for child in children} == {1}
    rewards = {child.task.goal.splitlines()[0]: child.reward for child in children}
    assert rewards == {
        f"Craft the following items: {count}x {target}": 1.0,
        "Craft the following items: 1x zz_nope": 0.0,
    }
    # The root's first step launched both children; the step records the one that succeeded.
    first, last = root.steps[0].misc, root.steps[-1].misc
    assert first["reward_misc"]["reward/subagent_launched"] == 2
    assert first["reward_misc"]["reward/subagent_succeeded"] == 1.0
    assert "['done', 'cannot craft zz_nope']" in first["output"]
    assert last["reward_misc"]["reward/success"] == 1.0
    # The original recursive prompts: the action space offers launch_subagent with a step budget.
    assert "async def launch_subagent(targets: dict, num_steps: int" in sampler.calls[0][0]
    assert "DELEGATION STRATEGY" in sampler.calls[0][0]

    data = train_datums(run, task.id)
    depths = sorted(int(datum.loss_fn_inputs["traj_depth"].to_torch()[0]) for datum in data.datums)
    assert depths[0] == 0 and depths[-1] == 1
    assert data.task_reward == 1.0 and data.root_rewards_dict["reward/subagent_launched"] == 2.0

    # The single program's finalizers, in its order: the original skip, then root-success propagation.
    zero_subagent_rewards(run.collection)
    assert [trajectory.reward for trajectory in run.collection.trajectories.values()] == [1.0, 0.0, 0.0]
    assert run.root.steps[0].misc["reward_misc"]["reward/subagent_succeeded"] == 0.0
    propagate_root_success(run.collection)
    assert [trajectory.reward for trajectory in run.collection.trajectories.values()] == [1.0, 1.0, 1.0]
    assert run.root.steps[0].misc["reward_misc"]["reward/subagent_succeeded"] == 2.0


def _goal(prompt: str) -> tuple[str, str]:
    """The (goal, budget) of the agent a TextCraft prompt is for."""
    return re.findall(r"Your Goal: (.*?)\nBudget: You have a total budget of (\d+) steps", prompt, re.S)[-1]


class _SlowChildren:
    """A sampler that answers every agent but the root after a delay, so the root can finish first."""

    def __init__(self, inner, root_goal: tuple[str, str], delay_s: float) -> None:
        self.inner, self.root_goal, self.delay_s = inner, root_goal, delay_s

    async def sample_async(self, *, prompt, sampling_params, num_samples):
        if _goal(self.inner.tokenizer.decode(prompt.to_ints())) != self.root_goal:
            await asyncio.sleep(self.delay_s)
        return await self.inner.sample_async(prompt=prompt, sampling_params=sampling_params, num_samples=num_samples)

    def sample(self, **kwargs):
        raise AssertionError("the gateway samples asynchronously")


@pytest.mark.asyncio
async def test_children_of_an_unawaited_gather_finish_and_are_graded_after_the_root_ends(make_gateway):
    # A case from training runs: a cell launches subagents in an asyncio.gather it never awaits, and the root
    # finishes on its next turn. Closing the root's cell loop cancels the pending launch_subagent coroutines, but
    # not their spawns: the root waits for those children to finish and be graded, as the original agent did.
    from platoon_textcraft.env import TextCraftSynthEnvironment

    spec = TextCraftSynthEnvironment.list_tasks("train-medium")[0]
    child_policy, target, count = _textcraft_policy(spec)
    root_goal = (f"Craft the following items: {count}x {target}", "200")

    def policy(prompt: str) -> str:
        if _goal(prompt) != root_goal:
            return child_policy(prompt)
        if "[Cell 0 Output]" not in prompt:
            return (
                "<thought>Delegate and move on.</thought>\n<python>"
                f"pending = asyncio.gather(launch_subagent({{{target!r}: {count}}}, 30), "
                "launch_subagent({'zz_nope': 1}, 5, 'this one cannot work'))</python>"
            )
        return "<thought>They will finish on their own.</thought>\n<python>finish('delegated')</python>"

    gateway, sampler = make_gateway(policy)
    gateway.sampler = _SlowChildren(sampler, root_goal, delay_s=0.3)
    task, session = await open_service_task(TEXTCRAFT_SYNTH, "train-medium:0")
    agents = {"budget": "linear", "child_grading": "env"}

    run = await run_tree(
        gateway, "platoon-codeact", session, task, max_steps=200, options=TEXTCRAFT_RECURSIVE_OPTIONS, agents=agents
    )

    assert run.result.status == "completed" and run.result.final_message == "delegated"
    root, children = run.root, run.children
    assert len(root.steps) == 2 and len(children) == 2
    assert [child.misc["agent_result"]["status"] for child in children] == ["completed", "completed"]
    assert [child.misc["outcome"]["status"] for child in children] == ["scored", "scored"]
    rewards = {child.task.goal.splitlines()[0]: child.reward for child in children}
    assert rewards == {
        f"Craft the following items: {count}x {target}": 1.0,
        "Craft the following items: 1x zz_nope": 0.0,
    }
    first = root.steps[0].misc["reward_misc"]
    assert (first["reward/subagent_launched"], first["reward/subagent_succeeded"]) == (2, 1.0)
    # The root's harness had made its last call before either child finished.
    last_root_call = max(end for _, end in root.misc["model_call_spans"].values())
    assert all(max(end for _, end in child.misc["model_call_spans"].values()) > last_root_call for child in children)


class _SlowGoal:
    """A sampler that answers the agent working on ``goal`` after a delay."""

    def __init__(self, inner, goal: str, delay_s: float) -> None:
        self.inner, self.goal, self.delay_s = inner, goal, delay_s

    async def sample_async(self, *, prompt, sampling_params, num_samples):
        if _goal(self.inner.tokenizer.decode(prompt.to_ints()))[0] == self.goal:
            await asyncio.sleep(self.delay_s)
        return await self.inner.sample_async(prompt=prompt, sampling_params=sampling_params, num_samples=num_samples)

    def sample(self, **kwargs):
        raise AssertionError("the gateway samples asynchronously")


@pytest.mark.asyncio
async def test_a_cell_that_gives_up_on_a_launch_cancels_the_child_and_gets_its_budget_back(make_gateway):
    # A cell may bound a launch with asyncio.wait_for. As before, giving up cancels the child, and its reservation
    # (31 of the root's 40 steps) comes back, so the next launch fits.
    from platoon_textcraft.env import TextCraftSynthEnvironment

    spec = TextCraftSynthEnvironment.list_tasks("train-medium")[0]
    child_policy, target, count = _textcraft_policy(spec)
    root_goal = (f"Craft the following items: {count}x {target}", "40")

    def policy(prompt: str) -> str:
        if _goal(prompt) != root_goal:
            return child_policy(prompt)
        if "[Cell 0 Output]" not in prompt:
            return (
                "<thought>Bound the slow launch, then delegate the target.</thought>\n<python>"
                "try:\n"
                "    first = await asyncio.wait_for(launch_subagent({'zz_slow': 1}, 30), 0.5)\n"
                "except asyncio.TimeoutError:\n"
                "    first = 'gave up'\n"
                # The child unwinds on the rollout loop, a moment after the cell stops waiting for it.
                "await asyncio.sleep(0.5)\n"
                f"second = await launch_subagent({{{target!r}: {count}}}, 30)\n"
                "print(first, '|', second)</python>"
            )
        return "<thought>Done.</thought>\n<python>finish('done')</python>"

    gateway, sampler = make_gateway(policy)
    gateway.sampler = _SlowGoal(sampler, "Craft the following items: 1x zz_slow", delay_s=60)
    task, session = await open_service_task(TEXTCRAFT_SYNTH, "train-medium:0")
    agents = {"budget": "linear", "child_grading": "env"}

    run = await run_tree(
        gateway, "platoon-codeact", session, task, max_steps=40, options=TEXTCRAFT_RECURSIVE_OPTIONS, agents=agents
    )

    assert run.result.status == "completed"
    assert run.root.steps[0].misc["output"] == "gave up | done\n"
    slow, second = run.children
    assert slow.task.goal == "Craft the following items: 1x zz_slow"
    assert slow.misc["agent_result"]["status"] == "cancelled" and trajectory_was_cancelled(slow)
    assert second.misc["agent_result"]["status"] == "completed" and second.reward == 1.0


@pytest.mark.asyncio
async def test_depth_aware_children_get_a_fixed_budget_and_the_depth_cap(make_gateway):
    def policy(prompt: str) -> str:
        if "[Cell 0 Output]" in prompt:
            return "<python>finish('stop')</python>"
        if "Context provided from parent agent" in prompt:
            return "<python>print(await launch_subagent({'raw_a1': 1}))</python>"
        return "<python>print(await launch_subagent({'a9_i1': 1}, 'context for the child'))</python>"

    gateway, sampler = make_gateway(policy)
    task, session = await open_service_task(TEXTCRAFT_SYNTH, "train-medium:0")
    options = {
        "prompt_builder": "platoon_textcraft.prompts:TextCraftPromptBuilder",
        "prompt_builder_options": {"variant": "depth_aware", "synth": True},
        "subagent_budget": "fixed",
    }
    agents = {"budget": "depth_aware", "max_depth": 1, "child_max_steps": 4, "child_grading": "env"}

    run = await run_tree(gateway, "platoon-codeact", session, task, max_steps=25, options=options, agents=agents)

    (child,) = run.children
    child_prompt = next(prompt for prompt, _ in sampler.calls if "Context provided from parent agent" in prompt)
    assert "total budget of 4 steps" in child_prompt
    assert "async def launch_subagent(targets: dict, context: str" in child_prompt
    assert "maximum allowed depth of 1" in child.steps[0].misc["output"]


def _depth_aware_policy(spec):
    """The scripted policy of the RAO recipe's recursive arms (``variant: depth_aware``): the root delegates its target
    and a hopeless item, with a context for each, and finishes; a child replays the recipe chain, or gives up."""

    ((target, count),) = spec["misc"]["target_items"].items()

    def policy(prompt: str) -> str:
        goal, _ = _goal(prompt)
        done = "[Cell 0 Output]" in prompt
        if "Context provided from parent agent" not in goal:  # the root
            if not done:
                return (
                    "<thought>Delegate the target and a hopeless item in parallel.</thought>\n<python>"
                    f"results = await asyncio.gather(launch_subagent({{{target!r}: {count}}}, 'craft the target'), "
                    "launch_subagent({'zz_nope': 1}, 'this one cannot work'))\nprint(results)</python>"
                )
            return f"<thought>Both returned.</thought>\n<python>finish('crafted {count} {target}')</python>"
        if goal.startswith("Craft the following items: 1x zz_nope"):
            return "<thought>Impossible.</thought>\n<python>finish('cannot craft zz_nope')</python>"
        if not done:
            return f"<thought>Replay the recipe chain.</thought>\n<python>{_gold_cell(spec)}</python>"
        return "<thought>Done.</thought>\n<python>finish('done')</python>"

    return policy, target, count


@pytest.mark.asyncio
@pytest.mark.parametrize(("arm", "hopeless_reward"), [("recursive", 0.0), ("ablation_sparse", 1.0)])
async def test_the_rao_recipes_recursive_arms_run_through_harness_rollout(arm, hopeless_reward, make_gateway, tmp_path):
    # The whole path: config -> routed task -> harness_rollout -> single program -> finalizers, with the arm's own
    # rollout config. The recursive arm keeps each subagent's own grade (the paper's Dense rewards), so the subagent
    # that crafts nothing scores 0; its Sparse ablation gives every subagent the root's success.
    from dataclasses import replace
    from pathlib import Path

    from platoon.harness.rollout import EmbeddedGateway, run_harness_rollout
    from platoon.train.auto import AutoEnvironment, AutoTaskLoader
    from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
    from platoon.utils.config import load_config

    from platoon_textcraft.env import TextCraftSynthEnvironment

    path = Path(__file__).resolve().parents[3] / f"recipes/rao/textcraft-synth/{arm}.yaml"
    config, _ = load_config(["--config", str(path)], PlatoonTinkerRLTrainerConfig)
    AutoEnvironment.load(config)
    task = AutoTaskLoader.from_config(config)("train-medium:0")
    task.max_steps = config.train.workflow_config.rollout_config.max_steps
    rollout_config = replace(config.train.workflow_config.rollout_config, output_dir=str(tmp_path), verbose=False)
    policy, target, count = _depth_aware_policy(TextCraftSynthEnvironment.list_tasks("train-medium")[0])
    gateway, sampler = make_gateway(policy)

    collection = await run_harness_rollout(task, rollout_config, gateway=EmbeddedGateway(gateway))

    root, *children = collection.trajectories.values()
    assert root.misc["outcome"]["status"] == "scored" and root.reward == 1.0
    assert root.misc["harness"]["program"] == "single"
    # The paper's recursive agent: launch_subagent(targets, context), and 25 steps for the root and each subagent.
    assert "async def launch_subagent(targets: dict, context: str" in sampler.calls[0][0]
    assert all("total budget of 25 steps" in prompt for prompt, _ in sampler.calls)
    assert len(children) == 2
    rewards = {child.task.goal.splitlines()[0]: child.reward for child in children}
    assert rewards == {
        f"Craft the following items: {count}x {target}": 1.0,
        "Craft the following items: 1x zz_nope": hopeless_reward,
    }
