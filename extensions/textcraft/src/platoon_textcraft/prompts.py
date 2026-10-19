"""The TextCraft prompts of platoon's earlier TextCraft agents, for the ``platoon-codeact`` harness.

``TextCraftPromptBuilder`` renders an earlier agent's TextCraft system prompt and action
space for one arm, so prompts stay token-for-token those of the earlier agents:

=============  ================================  =====================================
``variant``    earlier agent                     ``launch_subagent``
=============  ================================  =====================================
linear         ``TextCraftAgent``                not offered
recursive      ``TextCraftRecursiveAgent``       ``(targets, num_steps, context="")``
depth_aware    ``TextCraftDepthAwareAgent``      ``(targets, context="")``, fixed budget
=============  ================================  =====================================

``synth`` picks the TextCraft-Synth examples (item names like ``m2_i2``) over the
Minecraft ones. As before, ``launch_subagent({"stick": 2}, ...)`` delegates with the goal
``Craft the following items: 2x stick`` and gives the child ``target_items``. Select the
builder in the environment config::

    harness: platoon-codeact
    harness_options:
      prompt_builder: platoon_textcraft.prompts:TextCraftPromptBuilder
      prompt_builder_options: {variant: recursive, synth: true}
      subagent_budget: argument      # fixed for depth_aware
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from platoon.harness.builtin.codeact import CellFunction, CodeActPromptBuilder, CodeActState, PromptMode

if TYPE_CHECKING:
    from platoon import Task

Variant = Literal["linear", "recursive", "depth_aware"]
VARIANTS: tuple[Variant, ...] = ("linear", "recursive", "depth_aware")

LINEAR_INSTRUCTIONS = (
    "You are an agent in a crafting game. "
    "Your goal is to craft items by combining ingredients.\n"
    "You have access to an inventory of existing ingredients, "
    "which are sufficient to craft the target items; "
    "though, you may need to craft intermediate ingredients first.\n"
    "\n"
    "Note: If you already have one of the target items in your inventory, "
    "you should craft the requested number of the target on top of what you already have.\n"
    "For example, if you already have 2 wooden_pickaxes but your goal is to craft 3, "
    "your inventory should end up with 5 wooden_pickaxes.\n"
    "\n"
    "<TIPS>\n"
    "CRAFTING STRATEGY:\n"
    "- Recipes produce fixed quantities per execution - you cannot craft arbitrary amounts\n"
    "  Example: If a recipe produces 2 items, "
    "you can only craft in multiples of 2 (2, 4, 6...)\n"
    "- Recipe ingredients scale with the number of times you execute it\n"
    '  Example: Recipe "2 ore → 2 items" means 2 ore for 1 execution, 4 ore for 2 executions\n'
    "- Always verify what you have before claiming something is impossible\n"
    "- Check your inventory and recipe information to confirm ingredient availability\n"
    "- Calculate carefully: if a recipe uses 2 ingredients to make 2 items, "
    "you need exactly 2 ingredients for 2 items\n"
    "</TIPS>"
)
RECURSIVE_INSTRUCTIONS = (
    "You are an agent in a crafting game. "
    "Your goal is to craft items by combining ingredients.\n"
    "You have access to an inventory of existing ingredients, "
    "which are sufficient to craft the target items; "
    "though, you may need to craft intermediate ingredients first.\n"
    "\n"
    "Note: If you already have one of the target items in your inventory, "
    "you should craft the requested number of the target on top of what you already have.\n"
    "For example, if you already have 2 wooden_pickaxes but your goal is to craft 3, "
    "your inventory should end up with 5 wooden_pickaxes.\n"
    "\n"
    "<TIPS>\n"
    "CRAFTING STRATEGY:\n"
    "- Recipes produce fixed quantities per execution - you cannot craft arbitrary amounts\n"
    "  Example: If a recipe produces 2 items, "
    "you can only craft in multiples of 2 (2, 4, 6...)\n"
    "- Recipe ingredients scale with the number of times you execute it\n"
    '  Example: Recipe "2 ore → 2 items" means 2 ore for 1 execution, 4 ore for 2 executions\n'
    "- Always verify what you have before claiming something is impossible\n"
    "- Check your inventory and recipe information to confirm ingredient availability\n"
    "- Calculate carefully: if a recipe uses 2 ingredients to make 2 items, "
    "you need exactly 2 ingredients for 2 items\n"
    "\n"
    "DELEGATION STRATEGY:\n"
    "- It is **highly recommended** to delegate crafting of intermediate ingredients\n"
    "- Break complex tasks into INDEPENDENT subtasks that can be solved separately\n"
    "- For tasks that are sufficiently complex, it is recommended to recursively delegate; "
    "i.e., subagents can further delegate to other subagents.\n"
    "- Delegate one group of related items at a time, not everything at once\n"
    "- Use crafting depth from get_info() to estimate budget requirements:\n"
    "  * crafting_depth indicates complexity (0=base item, 1=direct craft, 2+=needs intermediates)\n"
    "  * Budget heuristic: depth × 8-10 steps "
    "(depth=4 needs ~32-40 steps, depth=8 needs ~64-80 steps)\n"
    "  * Always check crafting_depth before delegating to avoid under-budgeting\n"
    "- Items can be delegated in parallel if they don't depend on each other\n"
    "- Reserve budget for yourself to do final assembly after subtasks complete\n"
    "- Delegated tasks share your inventory - results are immediately available\n"
    "- **IMPORTANT**: `launch_subagent` is an async function, you MUST use `await`:\n"
    '  * CORRECT: `await launch_subagent({"item": 1}, 20)`\n'
    "  * CORRECT: `await asyncio.gather(launch_subagent(...), launch_subagent(...))`\n"
    '  * WRONG: `launch_subagent({"item": 1}, 20)` -- missing await, will error\n'
    "</TIPS>"
)
DEPTH_AWARE_INSTRUCTIONS = (
    "You are an agent in a crafting game. "
    "Your goal is to craft items by combining ingredients.\n"
    "You have access to an inventory of existing ingredients, "
    "which are sufficient to craft the target items; "
    "though, you may need to craft intermediate ingredients first.\n"
    "\n"
    "Note: If you already have one of the target items in your inventory, "
    "you should craft the requested number of the target on top of what you already have.\n"
    "For example, if you already have 2 wooden_pickaxes but your goal is to craft 3, "
    "your inventory should end up with 5 wooden_pickaxes.\n"
    "\n"
    "<TIPS>\n"
    "CRAFTING STRATEGY:\n"
    "- Recipes produce fixed quantities per execution - you cannot craft arbitrary amounts\n"
    "  Example: If a recipe produces 2 items, "
    "you can only craft in multiples of 2 (2, 4, 6...)\n"
    "- Recipe ingredients scale with the number of times you execute it\n"
    '  Example: Recipe "2 ore → 2 items" means 2 ore for 1 execution, 4 ore for 2 executions\n'
    "- Always verify what you have before claiming something is impossible\n"
    "- Check your inventory and recipe information to confirm ingredient availability\n"
    "- Calculate carefully: if a recipe uses 2 ingredients to make 2 items, "
    "you need exactly 2 ingredients for 2 items\n"
    "\n"
    "DELEGATION STRATEGY:\n"
    "- It is **highly recommended** to delegate crafting of intermediate ingredients\n"
    "- Break complex tasks into INDEPENDENT subtasks that can be solved separately\n"
    "- For tasks that are sufficiently complex, it is recommended to recursively delegate; "
    "i.e., subagents can further delegate to other subagents.\n"
    "- Delegate one group of related items at a time, not everything at once\n"
    "- Items can be delegated in parallel if they don't depend on each other\n"
    "- Delegated tasks share your inventory - results are immediately available\n"
    "- **IMPORTANT**: `launch_subagent` is an async function, you MUST use `await`:\n"
    '  * CORRECT: `await launch_subagent({"item": 1})`\n'
    "  * CORRECT: `await asyncio.gather(launch_subagent(...), launch_subagent(...))`\n"
    '  * WRONG: `launch_subagent({"item": 1})` -- missing await, will error\n'
    "</TIPS>"
)
INSTRUCTIONS: dict[Variant, str] = {
    "linear": LINEAR_INSTRUCTIONS,
    "recursive": RECURSIVE_INSTRUCTIONS,
    "depth_aware": DEPTH_AWARE_INSTRUCTIONS,
}

SYNTH_ACTIONS = """Available Actions (python functions):

1. def craft(ingredients: dict, target: tuple[str, int]) -> str
   Craft items using ingredients from your inventory.
   - ingredients: Dict of {item_name: count} to consume
   - target: (item_name, total_count) where total_count must be divisible by recipe result_count
   - Example: craft({"m0_i1": 2, "m1_i1": 1}, ("m2_i2", 2))

2. def get_info(items: list) -> list[dict]
   Get recipe information for items.
   - Returns: List with {"item": str, "can_craft": bool, "is_base": bool,
                         "in_inventory": int, "crafting_depth": int, "recipes": [...]}
   - crafting_depth indicates complexity: 0=base item, 1=direct craft, 2+=needs intermediate steps
   - Each recipe shows {"ingredients": {...}, "result_count": int}
   - Example: get_info(["m2_i2", "raw_m0"])

3. def view_inventory() -> dict
   View your current inventory.
   - Returns: Dict of {item_name: count}
   - Example: inv = view_inventory()

4. def finish(message: str) -> str
   Complete the task.
   - Example: finish("Successfully crafted all required items")
"""

SYNTH_SUBAGENT_ACTIONS = """
5. async def launch_subagent(targets: dict, num_steps: int, context: str = "") -> str
   Launch a subagent to craft specific targets (shares your inventory).
   - targets: Dict of {item_name: count} to craft
   - num_steps: Budget for subagent
     * Use crafting_depth from get_info() to estimate: depth × 8-10 steps
     * Example: depth=4 needs ~32-40 steps, depth=8 needs ~64-80 steps
   - context: Optional context string for the subagent
   - Sequential: await launch_subagent({"m0_i2": 4}, 20)
   - Parallel: results = await asyncio.gather(
                 launch_subagent({"m0_i2": 4}, 20),
                 launch_subagent({"c1_i2": 3}, 15)
               )
   - Returns: Subagent's finish message (or list if using gather)

Note: `asyncio` is already imported. Use `await asyncio.gather(...)` to run subtasks in parallel
or `await launch_subagent()` for a single subtask. **Do not forget to await** the results.
"""

SYNTH_DEPTH_AWARE_SUBAGENT_ACTIONS = """
5. async def launch_subagent(targets: dict, context: str = "") -> str
   Launch a subagent to craft specific targets (shares your inventory).
   - targets: Dict of {item_name: count} to craft
   - context: Optional context string for the subagent
   - Sequential: await launch_subagent({"m0_i2": 4})
   - Parallel: results = await asyncio.gather(
                 launch_subagent({"m0_i2": 4}),
                 launch_subagent({"c1_i2": 3})
               )
   - Returns: Subagent's finish message (or list if using gather)

Note: `asyncio` is already imported. Use `await asyncio.gather(...)` to run subtasks in parallel
or `await launch_subagent()` for a single subtask. **Do not forget to await** the results.
"""

MINECRAFT_ACTIONS = """Available Actions (python functions):

1. def craft(ingredients: dict, target: tuple[str, int]) -> str
   Craft items using ingredients from your inventory.
   - ingredients: Dict of {item_name: count} to consume
   - target: (item_name, total_count) where total_count must be divisible by recipe result_count
   - For tag-based ingredients (e.g., "tag:planks"), provide a CONCRETE item from that category
   - Example: craft({"stick": 2, "oak_planks": 3}, ("wooden_pickaxe", 1))
   - Example: craft({"oak_log": 4}, ("oak_planks", 16))

2. def get_info(items: list) -> list[dict]
   Get recipe information for items.
   - Returns: List with {"item": str, "can_craft": bool, "is_base": bool,
                         "in_inventory": int, "crafting_depth": int, "recipes": [...]}
   - crafting_depth indicates complexity: 0=base item, 1=direct craft, 2+=needs intermediate steps
   - Each recipe shows {"ingredients": {...}, "result_count": int}
   - Tag-based ingredients show {"tag:X": {"count": N, "use_one_of": [concrete items]}}
   - Example: get_info(["yellow_dye", "yellow_terracotta"])

3. def view_inventory() -> dict
   View your current inventory.
   - Returns: Dict of {item_name: count}
   - Example: inv = view_inventory()

4. def finish(message: str) -> str
   Complete the task.
   - Example: finish("Successfully crafted all required items")
"""

MINECRAFT_SUBAGENT_ACTIONS = """
5. async def launch_subagent(targets: dict, num_steps: int, context: str = "") -> str
   Launch a subagent to craft specific targets (shares your inventory).
   - targets: Dict of {item_name: count} to craft
   - num_steps: Budget for subagent
     * Use crafting_depth from get_info() to estimate: depth × 8-10 steps
     * Example: depth=4 needs ~32-40 steps, depth=8 needs ~64-80 steps
   - context: Optional context string for the subagent
   - Sequential: await launch_subagent({"yellow_dye": 2}, 20)
   - Parallel: results = await asyncio.gather(
                 launch_subagent({"yellow_dye": 2}, 15),
                 launch_subagent({"stick": 4}, 10)
               )
   - Returns: Subagent's finish message (or list if using gather)

Note: `asyncio` is already imported. Use `await asyncio.gather(...)` to run subtasks in parallel
or `await launch_subagent()` for a single subtask. **Do not forget to await** the results.
"""

MINECRAFT_DEPTH_AWARE_SUBAGENT_ACTIONS = """
5. async def launch_subagent(targets: dict, context: str = "") -> str
   Launch a subagent to craft specific targets (shares your inventory).
   - targets: Dict of {item_name: count} to craft
   - context: Optional context string for the subagent
   - Sequential: await launch_subagent({"yellow_dye": 2})
   - Parallel: results = await asyncio.gather(
                 launch_subagent({"yellow_dye": 2}),
                 launch_subagent({"stick": 4})
               )
   - Returns: Subagent's finish message (or list if using gather)

Note: `asyncio` is already imported. Use `await asyncio.gather(...)` to run subtasks in parallel
or `await launch_subagent()` for a single subtask. **Do not forget to await** the results.
"""


def action_space(variant: Variant, *, synth: bool) -> str:
    """The earlier action-space text for an arm (``describe_action_space`` of its code executor)."""

    actions = SYNTH_ACTIONS if synth else MINECRAFT_ACTIONS
    if variant == "recursive":
        actions += SYNTH_SUBAGENT_ACTIONS if synth else MINECRAFT_SUBAGENT_ACTIONS
    elif variant == "depth_aware":
        actions += SYNTH_DEPTH_AWARE_SUBAGENT_ACTIONS if synth else MINECRAFT_DEPTH_AWARE_SUBAGENT_ACTIONS
    return actions


class TextCraftPromptBuilder(CodeActPromptBuilder):
    """The earlier agents' TextCraft prompts for one arm (see the module docstring)."""

    def __init__(
        self,
        prompts_dir: Any = None,
        prompt_mode: PromptMode = "sequence_extension",
        include_reasoning: bool = True,
        *,
        variant: Variant = "linear",
        synth: bool = True,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        super().__init__(prompts_dir, prompt_mode=prompt_mode, include_reasoning=include_reasoning)
        self.variant: Variant = variant
        self.synth = synth

    def build_system_prompt(self, state: CodeActState, **context: Any) -> str:
        include_reasoning = context.get("include_reasoning", self.include_reasoning)
        base_instructions = INSTRUCTIONS[self.variant]
        if include_reasoning:
            return (
                base_instructions + "\n\n"
                "You can perform an action by writing a block of code. "
                "You will get multiple steps to complete the task.\n"
                "For your current step, first briefly reason (~1-3 sentences) about your next step "
                "in the <thought> </thought> tags and then output your code action "
                "in <python> </python> tags.\n"
                "Your code cell will be executed inside a jupyter notebook "
                "and the output will be shown to you."
            )
        else:
            return (
                base_instructions + "\n\n"
                "You can perform an action by writing a block of code. "
                "You will get multiple steps to complete the task.\n"
                "Output your code action in <python> </python> tags."
            )

    def describe_action_space(self, functions: Sequence[CellFunction]) -> str:
        return action_space(self.variant, synth=self.synth)

    def subagent_goal(self, targets: Any, context: str = "") -> str:
        """The goal for ``{item: count}`` targets: ``Craft the following items: 2x stick, 1x oak_planks``."""

        if isinstance(targets, Mapping):
            targets = "Craft the following items: " + ", ".join(f"{count}x {item}" for item, count in targets.items())
        return super().subagent_goal(targets, context)

    def subagent_task_misc(self, targets: Any, parent_task: Task) -> dict[str, Any] | None:
        """The parent's task misc plus ``target_items``, the ``{item: count}`` targets the child crafts."""

        if isinstance(targets, Mapping):
            return {**parent_task.misc, "target_items": dict(targets)}
        return super().subagent_task_misc(targets, parent_task)
