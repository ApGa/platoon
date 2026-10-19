"""The TextCraft-Synth scenarios of the CodeAct golden tests (``tests/harness/builtin/test_codeact_golden.py``):
recursive agent trees on TextCraft-Synth, replayed and compared in full with fixtures recorded on the CodeAct code that
predates its restructure (``golden/``, moved here unchanged with the rollouts they record).

A difference fails its scenario unless an entry of :data:`ALLOWED_DIFFERENCES` explains it: core's allow-lists
(``golden_scenarios``), which name these scenarios too, and :data:`MOVED_COMPONENTS`.  The fixtures hold the
Qwen3-4B-Instruct-2507 tokenizer's token ids, so the replays skip where its snapshot is not on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import codeact_golden as golden
import golden_scenarios
import pytest
from golden_scenarios import (
    TEXTCRAFT_SCENARIOS,
    GoldenRun,
    MakeGateway,
    Scenario,
    allow,
    cells,
    goal_line,
    run_scenario,
    script,
)
from test_textcraft_rollouts import TEXTCRAFT_RECURSIVE_OPTIONS, TEXTCRAFT_SYNTH, _textcraft_policy

from platoon_textcraft.env import TextCraftSynthEnvironment

GOLDEN_DIR = Path(__file__).parent / "golden"
CHILD_GOAL = "Craft the following items: 1x a9_i1"  # the goal of launch_subagent({'a9_i1': 1}, ...)
DEPTH_AWARE_OPTIONS = {
    "prompt_builder": "platoon_textcraft.prompts:TextCraftPromptBuilder",
    "prompt_builder_options": {"variant": "depth_aware", "synth": True},
    "subagent_budget": "fixed",
}
# The fixtures hold what Qwen3-4B-Instruct-2507's own tokenizer made of every call, so these replace the toy
# tokenizer and renderer of builtin_fixtures here (golden_scenarios; they skip where its tokenizer is not on disk).
model_name, tokenizer, renderer = golden_scenarios.model_name, golden_scenarios.tokenizer, golden_scenarios.renderer

# TextCraft left plugins/ in steps: its RAO program folded into single, its reward processor moved into core, and
# TextCraft became the extension platoon_textcraft.  Only the config-driven scenario records component names: its
# tasks' environment route and its trajectory's harness record.  [textcraft_synth_recursive_rao_config]
MOVED_COMPONENTS = (
    "TextCraft's RAO program (platoon.textcraft.programs:rao) folded into single, which adds the same finalizers in "
    "the same order; its reward processor moved into core as platoon-delegation (platoon.rewards:delegation_capped); "
    "and its environment and prompts moved from the platoon.textcraft plugin to the platoon_textcraft extension.  The "
    "route and the harness record name what the scenario ran, which changed names but not what it does."
)
MOVED_NAMES = {  # recorded -> now
    "platoon.textcraft.programs:rao": "single",
    "platoon.textcraft.rewards:delegation_capped": "platoon.rewards:delegation_capped",
    "python://platoon.textcraft.env:TextCraftSynthEnvironment": TEXTCRAFT_SYNTH,
    "platoon.textcraft.prompts:TextCraftPromptBuilder": "platoon_textcraft.prompts:TextCraftPromptBuilder",
}
# The RAO recipe's arms name their environment (env: textcraft-synth@1) since the recipe was written down, and so does
# the frozen copy of the arm this scenario recorded: the named environment's prompt-builder option (synth) comes
# first, then the arm's own (variant).  [textcraft_synth_recursive_rao_config]
NAMED_ENVIRONMENT = (
    "The recursive arm names its environment, env: textcraft-synth@1, whose prompt-builder options (synth) come first "
    "and the arm's own (variant) after them: the same options in another order."
)
# The recursive arm the scenario recorded, which the recipe no longer carries: since recipes/rao's recursive.yaml
# became the paper's depth-aware arm, this frozen copy of the earlier arm keeps the recording's settings.
LINEAR_BUDGET_ARM = Path(__file__).parent / "fixtures" / "rao_linear_budget_recursive.yaml"
ALLOWED_DIFFERENCES: tuple[golden.AllowedDifference, ...] = (
    *golden_scenarios.ALLOWED_DIFFERENCES,
    *allow(
        MOVED_COMPONENTS,
        "textcraft_synth_recursive_rao_config",
        *(
            f"/trajectories/*/task/misc/platoon_environment/{key}"
            for key in ("program", "reward_processor", "service", "harness_options/prompt_builder")
        ),
        "/trajectories/*/misc/harness/program",
        accept=lambda expected, actual: MOVED_NAMES.get(expected) == actual,
    ),
    *allow(
        NAMED_ENVIRONMENT,
        "textcraft_synth_recursive_rao_config",
        "/trajectories/*/task/misc/platoon_environment/harness_options/prompt_builder_options@keys",
        accept=lambda expected, actual: sorted(expected) == sorted(actual),
    ),
)


# --------------------------------------------------------------------------- scenarios


async def textcraft_synth_recursive_gather(make_gateway: MakeGateway, tmp_path: Any) -> GoldenRun:
    policy, _, _ = _textcraft_policy(TextCraftSynthEnvironment.list_tasks("train-medium")[0])
    return await run_scenario(
        make_gateway(policy),
        TEXTCRAFT_SYNTH,
        "train-medium:0",
        max_steps=200,
        options=TEXTCRAFT_RECURSIVE_OPTIONS,
        agents={"budget": "linear", "child_grading": "env"},
    )


async def textcraft_synth_depth_aware(make_gateway: MakeGateway, tmp_path: Any) -> GoldenRun:
    def policy(prompt: str) -> str:
        if "[Cell 0 Output]" in prompt:
            return "<python>finish('stop')</python>"
        if "Context provided from parent agent" in prompt:
            return "<python>print(await launch_subagent({'raw_a1': 1}))</python>"
        return "<python>print(await launch_subagent({'a9_i1': 1}, 'context for the child'))</python>"

    agents = {"budget": "depth_aware", "max_depth": 1, "child_max_steps": 4, "child_grading": "env"}
    return await run_scenario(
        make_gateway(policy),
        TEXTCRAFT_SYNTH,
        "train-medium:0",
        max_steps=25,
        options=DEPTH_AWARE_OPTIONS,
        agents=agents,
    )


async def textcraft_synth_child_runs_out_of_steps(make_gateway: MakeGateway, tmp_path: Any) -> GoldenRun:
    # A depth-aware child that never finishes uses up child_max_steps; the root gets the original budget text back.
    def policy(prompt: str) -> str:
        if goal_line(prompt) == CHILD_GOAL:
            return f"<python>print({cells(prompt)})</python>"
        if cells(prompt) == 0:
            return "<python>print(await launch_subagent({'a9_i1': 1}, 'context for the child'))</python>"
        return "<python>finish('done')</python>"

    agents = {"budget": "depth_aware", "max_depth": 1, "child_max_steps": 2, "child_grading": "env"}
    return await run_scenario(
        make_gateway(policy),
        TEXTCRAFT_SYNTH,
        "train-medium:0",
        max_steps=25,
        options=DEPTH_AWARE_OPTIONS,
        agents=agents,
    )


async def textcraft_synth_timeout_during_launch(make_gateway: MakeGateway, tmp_path: Any) -> GoldenRun:
    # The root's cell times out while it awaits a child; the child, in its second cell, is cancelled with it.
    child = [
        "<python>await asyncio.sleep(1.0)\nprint('first cell')</python>",
        "<python>await asyncio.sleep(30)</python>",
    ]

    def policy(prompt: str) -> str:
        if goal_line(prompt) == CHILD_GOAL:
            return child[min(cells(prompt), 1)]
        return "<python>print(await launch_subagent({'a9_i1': 1}, 5))</python>"

    return await run_scenario(
        make_gateway(policy),
        TEXTCRAFT_SYNTH,
        "train-medium:0",
        max_steps=40,
        options={**TEXTCRAFT_RECURSIVE_OPTIONS, "cell_timeout_s": 2.0},
        agents={"budget": "linear", "child_grading": "env"},
    )


async def textcraft_synth_keyword_call(make_gateway: MakeGateway, tmp_path: Any) -> GoldenRun:
    # An env tool called by keyword, out of its parameters' order.
    replies = [
        "<python>print(craft(target=('a9_i1', 12), ingredients={'raw_a1': 8}))</python>",
        "<python>finish('done')</python>",
    ]
    options = {
        "prompt_builder": "platoon_textcraft.prompts:TextCraftPromptBuilder",
        "prompt_builder_options": {"variant": "linear", "synth": True},
        "subagents": False,
    }
    return await run_scenario(
        make_gateway(script(replies)), TEXTCRAFT_SYNTH, "train-medium:0", max_steps=10, options=options
    )


async def textcraft_synth_recursive_rao_config(make_gateway: MakeGateway, tmp_path: Any) -> GoldenRun:
    # The whole path: config -> routed task -> harness_rollout -> single program -> finalizers.
    from platoon.config_defs import RolloutConfig
    from platoon.harness.rollout import EmbeddedGateway, run_harness_rollout
    from platoon.train.auto import AutoEnvironment, AutoTaskLoader
    from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
    from platoon.train.tinker.proxy import proxy_interactions
    from platoon.utils.config import load_config

    config, _ = load_config(["--config", str(LINEAR_BUDGET_ARM)], PlatoonTinkerRLTrainerConfig)
    AutoEnvironment.load(config)
    task = AutoTaskLoader.from_config(config)("train-medium:0")
    task.max_steps = config.train.workflow_config.rollout_config.max_steps
    # The config's sampling: its harness_options.sampling when the fixture was recorded, its rollout config's
    # inference_params since the sampling moved there (CONFIG_SAMPLING).
    inference_params = config.train.workflow_config.rollout_config.inference_params
    rollout_config = RolloutConfig(
        output_dir=str(tmp_path), propagate_root_success=True, verbose=False, inference_params=inference_params
    )
    policy, _, _ = _textcraft_policy(TextCraftSynthEnvironment.list_tasks("train-medium")[0])
    gateway = make_gateway(policy)
    interactions: dict[str, Any] = {}
    token = proxy_interactions.set(interactions)  # the rollout's interaction map, as the rollout worker binds it
    try:
        collection = await run_harness_rollout(task, rollout_config, gateway=EmbeddedGateway(gateway))
    finally:
        proxy_interactions.reset(token)
    return GoldenRun(gateway, collection, interactions, task.id)


SCENARIOS: dict[str, Scenario] = {
    scenario.__name__: scenario
    for scenario in (
        textcraft_synth_recursive_gather,
        textcraft_synth_depth_aware,
        textcraft_synth_recursive_rao_config,
        textcraft_synth_child_runs_out_of_steps,
        textcraft_synth_timeout_during_launch,
        textcraft_synth_keyword_call,
    )
}


# --------------------------------------------------------------------------- the comparison


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", list(SCENARIOS))
async def test_rollout_matches_its_golden_fixture(scenario, make_gateway, monkeypatch, tmp_path):
    await golden_scenarios.replay(
        scenario,
        SCENARIOS[scenario],
        make_gateway,
        monkeypatch,
        tmp_path,
        directory=GOLDEN_DIR,
        allowed=ALLOWED_DIFFERENCES,
    )


def test_every_golden_fixture_has_a_scenario():
    assert sorted(path.stem for path in GOLDEN_DIR.glob("*.json")) == sorted(SCENARIOS) == sorted(TEXTCRAFT_SCENARIOS)


def test_the_moves_explain_only_the_config_driven_scenario():
    own = ALLOWED_DIFFERENCES[len(golden_scenarios.ALLOWED_DIFFERENCES) :]

    assert {entry.reason for entry in own} == {MOVED_COMPONENTS, NAMED_ENVIRONMENT}
    assert {entry.scenarios for entry in own} == {("textcraft_synth_recursive_rao_config",)}
