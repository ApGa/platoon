"""The TextCraft prompt builder: what its ``platoon-codeact`` agents see and hand their subagents."""

from __future__ import annotations

from platoon import Task
from platoon.harness.builtin.codeact import CodeActOptions

from platoon_textcraft.prompts import TextCraftPromptBuilder


def test_textcraft_subagent_goals_keep_the_earlier_targets_form():
    builder = TextCraftPromptBuilder(variant="recursive")

    assert builder.subagent_goal({"stick": 2, "oak_planks": 1}, "use the oak") == (
        "Craft the following items: 2x stick, 1x oak_planks\n\nContext provided from parent agent: use the oak"
    )
    assert builder.subagent_goal({"m2_i2": 4}) == "Craft the following items: 4x m2_i2"
    assert builder.subagent_goal("Summarize the file") == "Summarize the file"
    assert builder.subagent_task_misc({"stick": 2}, Task(misc={"a": 1})) == {"a": 1, "target_items": {"stick": 2}}
    assert builder.subagent_task_misc("Summarize the file", Task(misc={"a": 1})) is None


def test_codeact_options_name_the_builder():
    textcraft = CodeActOptions.from_mapping(
        {
            "prompt_builder": "platoon_textcraft.prompts:TextCraftPromptBuilder",
            "prompt_builder_options": {"variant": "depth_aware", "synth": False},
        }
    ).make_prompt_builder()

    assert isinstance(textcraft, TextCraftPromptBuilder) and textcraft.variant == "depth_aware"
