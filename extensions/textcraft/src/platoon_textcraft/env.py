"""TextCraft and TextCraft-Synth as ORS environments: craft target items from an inventory by recipe.

``TextCraftEnvironment`` serves the Minecraft recipes and ``TextCraftSynthEnvironment`` the
synthetic ones (deeper hierarchies, difficulty tags); both run in process with
``service: python://platoon_textcraft.env:<Class>``, or named as ``env: textcraft@1`` and
``env: textcraft-synth@1``. Tasks come from the generated ``data/*.jsonl`` files (``tasks.py``,
``synth_tasks.py``): ``train`` and ``val``, and for TextCraft-Synth also ``<split>-<difficulty>``
(``train-medium`` is the RAO training set). This module imports neither platoon nor IPython.

Tools: ``craft``, ``get_info``, ``view_inventory`` and the terminal ``finish(message)``,
which grades the episode as platoon's earlier TextCraft environments did: success (reward 1)
when every target was crafted on top of the starting inventory and the message is not
empty. ``finish`` attaches their ``reward_misc`` (``reward/success``, ``target_items``,
inventories, ``missing_items``) to its result metadata; ``get_info`` and ``view_inventory``
send their values as ``structured_content``, so CodeAct cells get lists and dicts back as
they did.

Recursion keeps the earlier shared inventory: ``platoon_fork(task_spec)`` returns a child
environment for a subgoal that works on the *same* inventory dict, snapshots it as the
child's starting inventory, and grades the child's own targets when the child calls
``finish``. Tools are coroutines, so concurrent subagents never interleave inside a craft.

Written against openreward 0.1.125's API (``@tool`` with one pydantic parameter model,
``ToolOutput(finished=True)``); ``finish`` is also marked the SDK's terminal tool.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import openreward.environments as ors
from openreward.environments import Environment, Split, TextBlock, ToolOutput, tool
from pydantic import BaseModel, Field

from platoon_textcraft.recipe_loader import RecipeDatabase

DATA_DIR = Path(__file__).parent / "data"
STRUCTURED_CONTENT_KEY = "structured_content"
CRAFT_GOAL_PREFIX = "Craft the following items:"


def _terminal(fn: Any) -> Any:
    """The SDK's ``@terminal`` where it exists (0.1.157+); older SDKs get the same marker by hand."""

    marker = getattr(ors, "terminal", None)
    if marker is not None:
        return marker(fn)
    fn._env_tool_terminal = True
    return fn


@functools.cache
def minecraft_recipes() -> RecipeDatabase:
    return RecipeDatabase(DATA_DIR / "recipes")


@functools.cache
def synth_recipes(items_per_domain_tier: int = 25) -> Any:
    from platoon_textcraft.synth_recipe_loader import SynthRecipeLoader

    return SynthRecipeLoader(items_per_domain_tier=items_per_domain_tier)


@functools.cache
def _read_specs(filename: str) -> tuple[dict[str, Any], ...]:
    lines = (DATA_DIR / filename).read_text().splitlines()
    return tuple(json.loads(line) for line in lines if line.strip())


def parse_craft_targets(goal: str | None) -> dict[str, int] | None:
    """``{item: count}`` of a ``Craft the following items: 2x stick, 1x oak_planks`` goal, or None."""

    if not goal or CRAFT_GOAL_PREFIX not in goal:
        return None
    match = re.search(r"Craft the following items:\s*(.+)", goal)
    if not match:
        return None
    targets = {item: int(count) for count, item in re.findall(r"(\d+)x\s+([a-z0-9_]+)", match.group(1))}
    return targets or None


class CraftParams(BaseModel):
    ingredients: Any = Field(description="Dict of {item_name: count} to consume.")
    target: Any = Field(description="(item_name, total_count); total_count must be divisible by the recipe's yield.")


class GetInfoParams(BaseModel):
    items: Any = Field(description="List of item names.")


class FinishParams(BaseModel):
    message: str = Field(default="", description="A message saying what was achieved.")


class TextCraftEnvironment(Environment):
    """A TextCraft task on the Minecraft recipes (see the module docstring)."""

    split_files: ClassVar[Mapping[str, str]] = {"train": "textcraft_train.jsonl", "val": "textcraft_val.jsonl"}

    def __init__(self, task_spec: Mapping[str, Any] | None = None, secrets: Mapping[str, str] | None = None) -> None:
        task_spec = dict(task_spec or {})
        super().__init__(task_spec, dict(secrets or {}))
        misc = dict(task_spec.get("misc") or {})
        self.goal = str(task_spec.get("goal") or "")
        self.target_items: dict[str, int] = dict(misc.get("target_items") or parse_craft_targets(self.goal) or {})
        self.inventory: dict[str, int] = dict(misc.get("initial_inventory") or {})
        self.initial_inventory: dict[str, int] = dict(self.inventory)
        self.finished = False

    # ------------------------------------------------------------------ tasks

    @classmethod
    def recipe_db(cls) -> Any:
        return minecraft_recipes()

    @classmethod
    def list_splits(cls) -> Sequence[Split | str]:
        return [
            Split(name=name, type="train" if name.startswith("train") else "validation") for name in cls.split_names()
        ]

    @classmethod
    def split_names(cls) -> list[str]:
        return list(cls.split_files)

    @classmethod
    def list_tasks(cls, split: str) -> Sequence[dict[str, Any]]:
        if split not in cls.split_files:
            raise ValueError(f"Unknown {cls.__name__} split {split!r}; known: {cls.split_names()}")
        return list(_read_specs(cls.split_files[split]))

    def get_prompt(self) -> list[TextBlock]:
        return [TextBlock(text=self.goal)]

    # ------------------------------------------------------------------ recursion

    def platoon_fork(self, task_spec: Mapping[str, Any]) -> TextCraftEnvironment:
        """A child environment for a subgoal: the same inventory dict, its own targets and grade.

        The service passes the parent's task spec with the child's ``goal``, so the targets come from the
        goal text (``Craft the following items: 2x stick, ...``), or from a top-level ``target_items``;
        never from ``misc``, which still describes the parent's task.
        """

        task_spec = dict(task_spec or {})
        goal = str(task_spec.get("goal") or "")
        targets = task_spec.get("target_items") or parse_craft_targets(goal) or {}
        child = type(self)({"goal": goal}, {})
        child.target_items = {str(item): int(count) for item, count in dict(targets).items()}
        child.inventory = self.inventory  # shared: what a child crafts is the parent's
        child.initial_inventory = dict(self.inventory)
        return child

    # ------------------------------------------------------------------ tools

    @tool
    async def craft(self, params: CraftParams) -> ToolOutput:
        """Craft items from your inventory: ingredients is {item_name: count} to consume, target is
        (item_name, total_count) with total_count divisible by the recipe's result count. Returns a
        success or error message."""

        return _text(self._craft(params.ingredients, params.target))

    @tool
    async def get_info(self, params: GetInfoParams) -> ToolOutput:
        """Recipe information for items: a list of {"item", "can_craft", "is_base", "in_inventory",
        "crafting_depth", "recipes": [{"ingredients": {...}, "result_count": int}]}."""

        info = self._get_info(params.items)
        return ToolOutput(blocks=[TextBlock(text=json.dumps(info))], metadata={STRUCTURED_CONTENT_KEY: info})

    @tool
    async def view_inventory(self) -> ToolOutput:
        """Your current inventory as {item_name: count}."""

        inventory = dict(self.inventory)
        return ToolOutput(blocks=[TextBlock(text=json.dumps(inventory))], metadata={STRUCTURED_CONTENT_KEY: inventory})

    @_terminal
    @tool
    async def finish(self, params: FinishParams) -> ToolOutput:
        """Complete the task with a message saying what was achieved."""

        score, reward_misc = self.evaluate(params.message)
        self.finished = True
        return ToolOutput(
            blocks=[TextBlock(text=params.message)],
            reward=score,
            finished=True,
            metadata={"reward_misc": reward_misc},
        )

    # ------------------------------------------------------------------ game logic, as the earlier environments had it

    def evaluate(self, finish_message: str) -> tuple[float, dict[str, Any]]:
        """The grade of a finished episode: did the inventory gain every target count?"""

        reward_misc: dict[str, Any] = {}
        score = 0.0
        if finish_message:
            if not self.target_items:
                return 0.0, {"success": False, "error": "No target items defined in task", "reward/success": 0.0}
            missing = {
                item: required - (self.inventory.get(item, 0) - self.initial_inventory.get(item, 0))
                for item, required in self.target_items.items()
                if self.inventory.get(item, 0) - self.initial_inventory.get(item, 0) < required
            }
            reward_misc["success"] = not missing
            if missing:
                reward_misc["missing_items"] = missing
            else:
                score = 1.0
            reward_misc["target_items"] = dict(self.target_items)
            reward_misc["initial_inventory"] = dict(self.initial_inventory)
            reward_misc["final_inventory"] = dict(self.inventory)
        else:
            reward_misc["success"] = False
            reward_misc["error"] = "Episode ended without calling finish() - likely timeout or max steps reached"
        reward_misc["reward/success"] = score
        return score, reward_misc

    def _craft(self, ingredients: Any, target: Any) -> str:
        if not isinstance(ingredients, dict):
            return f"Error: ingredients must be a dict, got {type(ingredients).__name__}"
        # JSON carries tuples as lists, so a two-item list is accepted where the earlier code wanted a tuple.
        if not isinstance(target, (tuple, list)) or len(target) != 2:
            return "Error: target must be a tuple of (item_name, count)"
        target_item, target_count = target
        if not isinstance(target_item, str):
            return f"Error: target item must be a string, got {type(target_item).__name__}"
        if not isinstance(target_count, int):
            return f"Error: target count must be an int, got {type(target_count).__name__}"
        if target_count <= 0:
            return f"Error: target count must be positive, got {target_count}"
        for ing_item, ing_count in ingredients.items():
            if not isinstance(ing_item, str):
                return f"Error: ingredient name must be a string, got {type(ing_item).__name__}"
            if not isinstance(ing_count, int):
                return f"Error: ingredient count for {ing_item} must be an int, got {type(ing_count).__name__}"
            if ing_count <= 0:
                return f"Error: ingredient count for {ing_item} must be positive, got {ing_count}"

        recipe_db = self.recipe_db()
        recipes = recipe_db.get_recipes_for_item(target_item)
        if not recipes:
            return f"Error: No recipe found for {target_item}"

        errors = []
        for recipe_idx, recipe in enumerate(recipes):
            if target_count % recipe.result_count != 0:
                errors.append(
                    f"Recipe {recipe_idx + 1}: Target count {target_count} is not divisible "
                    f"by recipe result count {recipe.result_count}"
                )
                continue
            num_crafts = target_count // recipe.result_count

            ingredient_mapping: dict[str, str] = {}  # recipe ingredient (or tag) -> provided item
            recipe_error = None
            for recipe_ing, recipe_count in recipe.ingredients.items():
                total_required = recipe_count * num_crafts
                if recipe_ing.startswith("tag:"):
                    satisfying_items = recipe_db.get_items_for_tag(recipe_ing.replace("tag:", ""))
                    provided = next(
                        (
                            item
                            for item, count in ingredients.items()
                            if item in satisfying_items and count == total_required
                        ),
                        None,
                    )
                    if provided is None:
                        recipe_error = f"No ingredient provided for {recipe_ing}. Expected one of: {satisfying_items}"
                        break
                    ingredient_mapping[recipe_ing] = provided
                else:
                    if recipe_ing not in ingredients:
                        recipe_error = f"Missing required ingredient {recipe_ing}"
                        break
                    if ingredients[recipe_ing] != total_required:
                        recipe_error = (
                            f"Wrong amount of {recipe_ing}. Need {total_required}, provided {ingredients[recipe_ing]}"
                        )
                        break
                    ingredient_mapping[recipe_ing] = recipe_ing
            if recipe_error:
                errors.append(f"Recipe {recipe_idx + 1}: {recipe_error}")
                continue

            extra_ingredients = [item for item in ingredients if item not in ingredient_mapping.values()]
            if extra_ingredients:
                errors.append(
                    f"Recipe {recipe_idx + 1}: Extra ingredients not required: {', '.join(extra_ingredients)}"
                )
                continue

            missing = []
            for recipe_ing, provided_ing in ingredient_mapping.items():
                required = recipe.ingredients[recipe_ing] * num_crafts
                available = self.inventory.get(provided_ing, 0)
                if available < required:
                    missing.append(f"{provided_ing}: need {required}, have {available}")
            if missing:
                errors.append(f"Recipe {recipe_idx + 1}: Insufficient ingredients in inventory: {', '.join(missing)}")
                continue

            for recipe_ing, provided_ing in ingredient_mapping.items():
                self.inventory[provided_ing] -= recipe.ingredients[recipe_ing] * num_crafts
                if self.inventory[provided_ing] <= 0:
                    del self.inventory[provided_ing]
            crafted_count = recipe.result_count * num_crafts
            self.inventory[target_item] = self.inventory.get(target_item, 0) + crafted_count
            return f"Successfully crafted {crafted_count} {target_item}(s)"

        return f"Error: All {len(recipes)} recipe(s) failed for {target_item}. Errors:\n" + "\n".join(errors)

    def _get_info(self, items: Any) -> list[dict[str, Any]]:
        recipe_db = self.recipe_db()
        results = []
        for item in items:
            info: dict[str, Any] = {
                "item": item,
                "can_craft": recipe_db.can_craft(item),
                "is_base": recipe_db.is_base_item(item),
                "in_inventory": self.inventory.get(item, 0),
                "crafting_depth": recipe_db.get_crafting_depth(item),
                "recipes": [],
            }
            if info["can_craft"]:
                for recipe in recipe_db.get_recipes_for_item(item):
                    expanded: dict[str, Any] = {}
                    for ing, count in recipe.ingredients.items():
                        if ing.startswith("tag:"):
                            expanded[ing] = {"count": count, "use_one_of": recipe_db.get_items_for_tag(ing[4:])}
                        else:
                            expanded[ing] = count
                    info["recipes"].append({"ingredients": expanded, "result_count": recipe.result_count})
            results.append(info)
        return results


class TextCraftSynthEnvironment(TextCraftEnvironment):
    """A TextCraft-Synth task: synthetic recipes, with ``<split>-<difficulty>`` splits for each difficulty."""

    split_files: ClassVar[Mapping[str, str]] = {
        "train": "textcraft_synth_train.jsonl",
        "val": "textcraft_synth_val.jsonl",
    }
    difficulties: ClassVar[tuple[str, ...]] = ("easy", "medium", "hard", "extreme")

    @classmethod
    def recipe_db(cls) -> Any:
        return synth_recipes()

    @classmethod
    def split_names(cls) -> list[str]:
        return [*cls.split_files, *(f"{split}-{level}" for split in cls.split_files for level in cls.difficulties)]

    @classmethod
    def list_tasks(cls, split: str) -> Sequence[dict[str, Any]]:
        base, _, difficulty = split.partition("-")
        if base not in cls.split_files or (difficulty and difficulty not in cls.difficulties):
            raise ValueError(f"Unknown {cls.__name__} split {split!r}; known: {cls.split_names()}")
        specs = _read_specs(cls.split_files[base])
        if not difficulty:
            return list(specs)
        return [spec for spec in specs if (spec.get("misc") or {}).get("difficulty") == difficulty]


def _text(text: str) -> ToolOutput:
    return ToolOutput(blocks=[TextBlock(text=text)])
