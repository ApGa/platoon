"""What CodeAct cells can call: env tools with signatures from their JSON schemas, ``finish`` and
``launch_subagent``. Plain async tests against a fake agent context: no IPython, no threads."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from platoon.envs.service import ToolResult, ToolSpec
from platoon.harness.builtin.codeact import CellFunction, CodeActPromptBuilder
from platoon.harness.builtin.codeact.namespace import (
    Ending,
    cell_functions,
    finish_function,
    launch_subagent_function,
    result_value,
    schema_parameters,
    schema_type,
    tool_function,
)
from platoon.task import Task

CRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "ingredients": {"type": "object", "additionalProperties": {"type": "integer"}},
        "target": {"type": "array", "prefixItems": [{"type": "string"}, {"type": "integer"}]},
        "note": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
    },
    "required": ["ingredients", "target"],
}
GUESS = ToolSpec(
    "guess",
    "Guess a number.",
    {"type": "object", "properties": {"number": {"type": "integer"}}, "required": ["number"]},
)
SUBMIT = ToolSpec(
    "submit",
    "Submit your answer.",
    {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
    terminal=True,
)
FINISHED = ToolResult(content=[{"type": "text", "text": "Submitted."}], finished=True)


class FakeContext:
    """The part of ``AgentContext`` cell functions use: the task, ``call_tool`` and ``agents.spawn``."""

    def __init__(self, results: dict[str, ToolResult] | None = None) -> None:
        self.task = Task(goal="root goal", misc={"seed": 1})
        self.results = results or {}
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.spawned: list[dict[str, Any]] = []
        self.agents = self

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        self.tool_calls.append((name, dict(arguments)))
        return self.results.get(name, ToolResult.text(f"{name} done"))

    async def spawn(self, goal: str, *, max_steps: int | None = None, task_misc: Any = None, **_: Any) -> Any:
        self.spawned.append({"goal": goal, "max_steps": max_steps, "task_misc": task_misc})
        await asyncio.sleep(0)
        return SimpleNamespace(message=f"finished: {goal}")


async def _never(arguments: dict[str, Any]) -> None:
    raise AssertionError("not invoked")


# ---------------------------------------------------------------------------- signatures


def test_schema_types_become_python_annotations():
    assert schema_type({"type": "array", "items": {"type": "string"}}) == "list[str]"
    assert schema_type(CRAFT_SCHEMA["properties"]["target"]) == "tuple[str, int]"
    assert schema_type(CRAFT_SCHEMA["properties"]["ingredients"]) == "dict[str, int]"
    assert schema_type(CRAFT_SCHEMA["properties"]["note"]) == "str | None"
    assert schema_type({}) is None


def test_cell_function_describes_its_tool_signature():
    function = CellFunction("craft", schema_parameters(CRAFT_SCHEMA), "Craft items.\n\nUse your inventory.", _never)

    assert (
        function.signature()
        == "def craft(ingredients: dict[str, int], target: tuple[str, int], note: str | None = None) -> str"
    )
    assert function.describe() == function.signature() + "\n   Craft items.\n\n   Use your inventory."


def test_tool_results_return_structured_content_or_text():
    assert result_value(ToolResult.text("Too low, try again.")) == "Too low, try again."
    structured = ToolResult(content=[{"type": "text", "text": "{}"}], metadata={"structured_content": {"a": 1}})
    assert result_value(structured) == {"a": 1}


# ---------------------------------------------------------------------------- env tools


@pytest.mark.asyncio
async def test_a_tool_calls_the_env_with_the_arguments_passed_and_returns_its_value():
    info = ToolResult(content=[{"type": "text", "text": "[...]"}], metadata={"structured_content": [{"item": "a"}]})
    ctx, ending = FakeContext({"get_info": info}), Ending()
    craft = tool_function(ctx, ToolSpec("craft", "Craft items.", CRAFT_SCHEMA), ending)
    get_info = tool_function(ctx, ToolSpec("get_info", "Recipes.", {"type": "object", "properties": {}}), ending)

    assert await craft.invoke({"ingredients": {"a": 1}, "target": ["b", 1]}) == "craft done"
    assert await get_info.invoke({}) == [{"item": "a"}]
    # Arguments the call left out stay out: the env applies its own defaults.
    assert ctx.tool_calls == [("craft", {"ingredients": {"a": 1}, "target": ["b", 1]}), ("get_info", {})]
    assert (craft.name, craft.description, craft.is_async) == ("craft", "Craft items.", False)
    assert not ending.ended


def test_builtins_win_over_env_tools_of_the_same_name():
    env_finish = ToolSpec(
        "finish", "Finish the task.", {"type": "object", "properties": {"message": {}}}, terminal=True
    )
    functions = cell_functions(
        FakeContext(), [env_finish, GUESS], Ending(), CodeActPromptBuilder(), subagents=True, fixed_budget=False
    )

    assert [function.name for function in functions] == ["guess", "finish", "launch_subagent"]
    assert functions[1].description == "Finish the task."  # the terminal tool's description
    assert [function.is_async for function in functions] == [False, False, True]
    linear = cell_functions(
        FakeContext(), [GUESS], Ending(), CodeActPromptBuilder(), subagents=False, fixed_budget=False
    )
    assert [function.name for function in linear] == ["guess", "finish"]


# ---------------------------------------------------------------------------- the ending and finish


def test_the_ending_follows_the_original_halt_rule():
    ending = Ending()
    assert not ending.ended
    ending.observe(ToolResult.text("Too low"))
    assert not ending.ended
    ending.observe(ToolResult(content=[{"type": "text", "text": "Correct!"}], finished=True))
    assert ending.ended and ending.final_message == "Correct!"
    ending.observe(ToolResult(content=[{"type": "text", "text": "Again"}], finished=True))
    assert ending.final_message == "Correct!"  # the first message stays
    assert Ending(final_message="").ended  # an empty message still ends the agent


@pytest.mark.asyncio
async def test_finish_without_a_terminal_tool_ends_the_agent_with_its_message():
    ctx, ending, empty = FakeContext(), Ending(), Ending()
    finish = finish_function(ctx, None, ending)

    assert finish.signature() == "def finish(message: str = '') -> str"
    assert finish.description.startswith("End the agent trajectory and provide a message to the user.")
    assert await finish.invoke({"message": "done"}) == "done"
    assert ending.final_message == "done" and ctx.tool_calls == []
    assert await finish_function(ctx, None, empty).invoke({}) == "" and empty.ended


@pytest.mark.asyncio
async def test_finish_hands_its_message_to_the_terminal_tool():
    ctx, ending = FakeContext({"submit": FINISHED}), Ending()
    finish = finish_function(ctx, SUBMIT, ending)

    assert finish.description == "Submit your answer."
    await finish.invoke({"message": "42"})
    assert ctx.tool_calls == [("submit", {"answer": "42"})]
    assert ending.final_message == "42"  # finish's message wins over the tool's text


@pytest.mark.asyncio
async def test_finish_none_does_not_end_the_agent_unless_the_env_says_so():
    # The original halt rule: an agent ends when it has a final message, and finish(None) gives none.
    ending = Ending()
    await finish_function(FakeContext(), None, ending).invoke({"message": None})
    assert not ending.ended

    ending = Ending()
    await finish_function(FakeContext({"submit": FINISHED}), SUBMIT, ending).invoke({"message": None})
    assert ending.final_message == "Submitted."


@pytest.mark.asyncio
async def test_a_finished_tool_result_ends_the_agent_with_its_text():
    correct = ToolResult(content=[{"type": "text", "text": "Correct!"}], finished=True)
    ending = Ending()

    assert await tool_function(FakeContext({"guess": correct}), GUESS, ending).invoke({"number": 7}) == "Correct!"
    assert ending.final_message == "Correct!"


# ---------------------------------------------------------------------------- launch_subagent


class TargetsBuilder(CodeActPromptBuilder):
    """A builder whose ``{item: count}`` targets become a goal and task misc, as TextCraft's do."""

    def subagent_goal(self, targets: Any, context: str = "") -> str:
        if isinstance(targets, dict):
            targets = "Make " + ", ".join(f"{count}x {item}" for item, count in targets.items())
        return super().subagent_goal(targets, context)

    def subagent_task_misc(self, targets: Any, parent_task: Task) -> dict[str, Any] | None:
        return {**parent_task.misc, "targets": targets} if isinstance(targets, dict) else None


@pytest.mark.asyncio
async def test_launch_subagent_spawns_a_child_with_the_builders_goal_and_misc():
    ctx = FakeContext()
    launch = launch_subagent_function(ctx, TargetsBuilder(), fixed_budget=False)

    assert launch.is_async and launch.signature() == (
        "async def launch_subagent(targets: dict | str, num_steps: int, context: str = '') -> str"
    )
    assert await launch.invoke({"targets": {"stick": 2}, "num_steps": 5, "context": "use oak"}) == (
        "finished: Make 2x stick\n\nContext provided from parent agent: use oak"
    )
    await launch.invoke({"targets": "Summarize the file", "num_steps": 3})
    assert ctx.spawned == [
        {
            "goal": "Make 2x stick\n\nContext provided from parent agent: use oak",
            "max_steps": 5,
            "task_misc": {"seed": 1, "targets": {"stick": 2}},
        },
        {"goal": "Summarize the file", "max_steps": 3, "task_misc": None},
    ]


@pytest.mark.asyncio
async def test_a_fixed_budget_launch_has_no_num_steps():
    ctx = FakeContext()
    launch = launch_subagent_function(ctx, CodeActPromptBuilder(), fixed_budget=True)

    assert [parameter.name for parameter in launch.parameters] == ["targets", "context"]
    assert "num_steps" not in launch.description
    await launch.invoke({"targets": "Summarize the file"})
    assert ctx.spawned == [{"goal": "Summarize the file", "max_steps": None, "task_misc": None}]  # child_max_steps


@pytest.mark.asyncio
@pytest.mark.parametrize("num_steps", ["5", 2.5, True])
async def test_num_steps_must_be_an_int(num_steps):
    ctx = FakeContext()
    launch = launch_subagent_function(ctx, CodeActPromptBuilder(), fixed_budget=False)

    with pytest.raises(TypeError, match=f"num_steps must be an int, got {type(num_steps).__name__}"):
        await launch.invoke({"targets": "Summarize the file", "num_steps": num_steps})
    assert ctx.spawned == []
