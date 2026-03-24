"""TextCraft environment for recursive agent spawning in crafting tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from platoon.agents.actions.common import finish
from platoon.agents.actions.subagent import launch_subagent as _launch_subagent
from platoon.envs.base import Task
from platoon.envs.codeact import (
    CodeActEnv,
    CodeActObservation,
    ForkableCodeExecutor,
    IPythonCodeExecutor,
    safe_asyncio,
)
from platoon.episode.context import current_trajectory, finish_message

from .recipe_loader import RecipeDatabase


class TextCraftCodeExecutor(IPythonCodeExecutor):
    """Code executor for TextCraft with crafting actions.

    Supports both original Minecraft recipes and synthetic recipes.
    """

    def __init__(
        self,
        task: Task,
        recipes_dir: Optional[Path] = None,
        recipe_db: Optional[Any] = None,
        inventory: Optional[Dict[str, int]] = None,
        _share_inventory: bool = False,
        use_synth: bool = False,
    ):
        """Initialize the TextCraft code executor.

        Args:
            task: The task to execute
            recipes_dir: Path to recipes directory (for original Minecraft recipes)
            recipe_db: Pre-initialized recipe database (for synthetic recipes or custom DBs)
            inventory: Initial inventory
            _share_inventory: If True, share inventory reference with subagents
            use_synth: If True, use synthetic recipe examples in action space
        """
        # Initialize recipe database from either recipe_db or recipes_dir
        if recipe_db is not None:
            self.recipe_db = recipe_db
            self.recipes_dir = None
        elif recipes_dir is not None:
            self.recipes_dir = Path(recipes_dir)
            self.recipe_db = RecipeDatabase(self.recipes_dir)
        else:
            # Default to original Minecraft recipes
            self.recipes_dir = Path(__file__).parent / "recipes"
            self.recipe_db = RecipeDatabase(self.recipes_dir)

        self.use_synth = use_synth

        # When _share_inventory=True, use the inventory dict directly (for subagent propagation)
        # When False (default), make a copy to avoid unintended mutations
        if _share_inventory and inventory is not None:
            self.inventory = inventory
        else:
            self.inventory = inventory.copy() if inventory else {}
        self.subagent_results: Dict[str, Dict[str, int]] = {}  # subagent_id -> {item: count}
        super().__init__(
            task,
            actions=(
                finish,
                self.craft,
                self.get_info,
                self.view_inventory,
                self.launch_subagent,
                safe_asyncio,
            ),
        )

    def craft(self, ingredients: Dict[str, int], target: Tuple[str, int]) -> str:
        """
        Craft an item using ingredients.

        Args:
            ingredients: Dictionary mapping item names to counts to consume
            target: Tuple of (item_name, total_count) where total_count is the total number of items to produce

        Returns:
            Success message or error message
        """

        # Validate input types and values
        if not isinstance(ingredients, dict):
            return f"Error: ingredients must be a dict, got {type(ingredients).__name__}"

        if not isinstance(target, tuple) or len(target) != 2:
            return "Error: target must be a tuple of (item_name, count)"

        target_item, target_count = target

        if not isinstance(target_item, str):
            return f"Error: target item must be a string, got {type(target_item).__name__}"

        if not isinstance(target_count, int):
            return f"Error: target count must be an int, got {type(target_count).__name__}"

        if target_count <= 0:
            return f"Error: target count must be positive, got {target_count}"

        # Validate ingredient counts
        for ing_item, ing_count in ingredients.items():
            if not isinstance(ing_item, str):
                return f"Error: ingredient name must be a string, got {type(ing_item).__name__}"

            if not isinstance(ing_count, int):
                return f"Error: ingredient count for {ing_item} must be an int, got {type(ing_count).__name__}"

            if ing_count <= 0:
                return f"Error: ingredient count for {ing_item} must be positive, got {ing_count}"

        # Check if recipe exists
        recipes = self.recipe_db.get_recipes_for_item(target_item)
        if not recipes:
            return f"Error: No recipe found for {target_item}"

        # Try each recipe until one succeeds
        errors = []
        for recipe_idx, recipe in enumerate(recipes):
            # Calculate how many times to execute this recipe
            # target_count is the total items to produce
            # recipe.result_count is how many items one recipe execution produces
            # target_count must be evenly divisible by result_count
            if target_count % recipe.result_count != 0:
                errors.append(
                    f"Recipe {recipe_idx + 1}: Target count {target_count} is not divisible "
                    f"by recipe result count {recipe.result_count}"
                )
                continue

            num_crafts = target_count // recipe.result_count

            # Validate ingredients match recipe requirements
            # Build a mapping from recipe ingredients (including tags) to provided ingredients
            ingredient_mapping = {}  # recipe_ing -> provided_ing
            recipe_error = None

            for recipe_ing, recipe_count in recipe.ingredients.items():
                required_per_craft = recipe_count
                total_required = required_per_craft * num_crafts

                if recipe_ing.startswith("tag:"):
                    # Tag-based ingredient: find which concrete item the agent provided
                    tag_name = recipe_ing.replace("tag:", "")
                    satisfying_items = self.recipe_db.get_items_for_tag(tag_name)

                    # Find which satisfying item is in the provided ingredients
                    found = False
                    for provided_ing, provided_count in ingredients.items():
                        if provided_ing in satisfying_items:
                            if provided_count == total_required:
                                ingredient_mapping[recipe_ing] = provided_ing
                                found = True
                                break

                    if not found:
                        recipe_error = f"No ingredient provided for {recipe_ing}. Expected one of: {satisfying_items}"
                        break
                else:
                    # Concrete ingredient: must be in provided ingredients
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

            # Check for extra ingredients not in recipe
            extra_ingredients = []
            for provided_ing in ingredients.keys():
                if provided_ing not in ingredient_mapping.values():
                    extra_ingredients.append(provided_ing)

            if extra_ingredients:
                errors.append(
                    f"Recipe {recipe_idx + 1}: Extra ingredients not required: {', '.join(extra_ingredients)}"
                )
                continue

            # Check we have enough of each ingredient in inventory
            missing = []
            for recipe_ing, provided_ing in ingredient_mapping.items():
                required = recipe.ingredients[recipe_ing] * num_crafts
                available = self.inventory.get(provided_ing, 0)

                if available < required:
                    missing.append(f"{provided_ing}: need {required}, have {available}")

            if missing:
                errors.append(f"Recipe {recipe_idx + 1}: Insufficient ingredients in inventory: {', '.join(missing)}")
                continue

            # This recipe works! Execute it
            # Consume ingredients from inventory
            for recipe_ing, provided_ing in ingredient_mapping.items():
                amount = recipe.ingredients[recipe_ing] * num_crafts
                self.inventory[provided_ing] -= amount
                if self.inventory[provided_ing] <= 0:
                    del self.inventory[provided_ing]

            # Add crafted items
            crafted_count = recipe.result_count * num_crafts
            self.inventory[target_item] = self.inventory.get(target_item, 0) + crafted_count

            return f"Successfully crafted {crafted_count} {target_item}(s)"

        # All recipes failed
        return f"Error: All {len(recipes)} recipe(s) failed for {target_item}. Errors:\n" + "\n".join(errors)

    def get_info(self, items: List[str]) -> List[Dict[str, Any]]:
        """
        Get information about items (recipes, whether they can be crafted, etc.)

        Args:
            items: List of item names to get info about

        Returns:
            List of dictionaries with item information
        """
        results = []
        for item in items:
            info = {
                "item": item,
                "can_craft": self.recipe_db.can_craft(item),
                "is_base": self.recipe_db.is_base_item(item),
                "in_inventory": self.inventory.get(item, 0),
                "crafting_depth": self.recipe_db.get_crafting_depth(item),
                "recipes": [],
            }

            if info["can_craft"]:
                for recipe in self.recipe_db.get_recipes_for_item(item):
                    # Expand tag-based ingredients to show concrete item options
                    expanded_ingredients = {}
                    for ing, count in recipe.ingredients.items():
                        if ing.startswith("tag:"):
                            tag_name = ing.replace("tag:", "")
                            concrete_items = self.recipe_db.get_items_for_tag(tag_name)
                            # Show both the tag and its concrete options
                            expanded_ingredients[ing] = {
                                "count": count,
                                "use_one_of": concrete_items,
                            }
                        else:
                            expanded_ingredients[ing] = count

                    info["recipes"].append({"ingredients": expanded_ingredients, "result_count": recipe.result_count})

            results.append(info)

        return results

    def view_inventory(self) -> Dict[str, int]:
        """
        View current inventory.

        Returns:
            Dictionary mapping item names to counts
        """
        return self.inventory.copy()

    async def launch_subagent(self, targets: Dict[str, int], num_steps: int, context: str = "") -> str:
        """
        Launch a subagent to craft the specified targets.

        Args:
            targets: Dictionary mapping item names to target counts
            num_steps: Maximum number of steps for the subagent
            context: Context string to pass to the subagent

        Returns:
            Message from the subagent indicating success or failure
        """
        # Convert targets dict to a goal string
        target_str = ", ".join([f"{count}x {item}" for item, count in targets.items()])
        goal = f"Craft the following items: {target_str}"

        if context:
            goal += f"\n\nContext provided from parent agent: {context}"

        # Use the general launch_subagent function
        # Inventory is shared by reference, so changes propagate automatically
        result = await _launch_subagent(goal=goal, max_steps=num_steps)

        return result

    def _get_action_space_parts(self, include_subagent: bool = False) -> str:
        """Build action space description with appropriate examples for synth vs regular mode."""
        if self.use_synth:
            actions = """Available Actions (python functions):

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
            if include_subagent:
                actions += """
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
        else:
            actions = """Available Actions (python functions):

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
            if include_subagent:
                actions += """
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
        return actions

    async def describe_action_space(self) -> str:
        return self._get_action_space_parts(include_subagent=False)

    async def reset(self) -> "TextCraftCodeExecutor":
        # Reset inventory to initial state if needed
        return self

    async def fork(self, task: Task) -> "TextCraftCodeExecutor":
        """Fork the executor for a subagent."""
        return TextCraftCodeExecutor(
            task=task,
            recipes_dir=self.recipes_dir,
            recipe_db=self.recipe_db,
            inventory=self.inventory,
            _share_inventory=True,  # Share inventory by reference for subagent propagation
            use_synth=self.use_synth,
        )


class TextCraftRecursiveCodeExecutor(TextCraftCodeExecutor):
    def __init__(
        self,
        task: Task,
        recipes_dir: Optional[Path] = None,
        recipe_db: Optional[Any] = None,
        inventory: Optional[Dict[str, int]] = None,
        _share_inventory: bool = False,
        use_synth: bool = False,
    ):
        super().__init__(
            task,
            recipes_dir=recipes_dir,
            recipe_db=recipe_db,
            inventory=inventory,
            _share_inventory=_share_inventory,
            use_synth=use_synth,
        )
        self._launched_subagent_ids_this_step: set[str] = set()
        self._subagent_success_by_child_this_step: dict[str, float] = {}

    def reset_subagent_stats(self) -> None:
        """Reset subagent tracking for a new step."""
        self._launched_subagent_ids_this_step.clear()
        self._subagent_success_by_child_this_step.clear()

    def get_subagent_stats(self) -> tuple[int, float]:
        """Get (unique launched children, summed child success score) for current step."""
        return len(self._launched_subagent_ids_this_step), sum(self._subagent_success_by_child_this_step.values())

    async def launch_subagent(self, targets: Dict[str, int], num_steps: int, context: str = "") -> str:
        """Launch a subagent and track whether it succeeded."""
        from platoon.episode.context import current_trajectory, current_trajectory_collection

        # Get current trajectory to identify child
        traj_collection = current_trajectory_collection.get()
        current_traj = current_trajectory.get()

        # Track trajectories before launch to find the new one
        traj_ids_before = set(traj_collection.trajectories.keys())

        # Call parent's launch_subagent
        result = await super().launch_subagent(targets, num_steps, context)

        for traj_id, traj in traj_collection.trajectories.items():
            if traj_id in traj_ids_before or traj_id in self._launched_subagent_ids_this_step:
                continue
            if not traj.parent_info or traj.parent_info.id != current_traj.id:
                continue

            self._launched_subagent_ids_this_step.add(traj_id)
            success_reward = 0.0
            if traj.steps:
                final_step = traj.steps[-1]
                reward_misc = final_step.misc.get("reward_misc", {})
                success_reward = float(reward_misc.get("reward/success", 0.0))
            self._subagent_success_by_child_this_step[traj_id] = success_reward
        return result

    async def describe_action_space(self) -> str:
        return self._get_action_space_parts(include_subagent=True)

    async def fork(self, task: Task) -> "TextCraftRecursiveCodeExecutor":
        """Fork the executor for a subagent."""
        return TextCraftRecursiveCodeExecutor(
            task=task,
            recipes_dir=self.recipes_dir,
            recipe_db=self.recipe_db,
            inventory=self.inventory,
            _share_inventory=True,
            use_synth=self.use_synth,
        )


class TextCraftEnv(CodeActEnv):
    """Environment for TextCraft crafting tasks with recursive agent spawning.

    Supports both original Minecraft recipes and synthetic recipes.
    """

    def __init__(
        self,
        task: Task,
        recipes_dir: Optional[Path] = None,
        recipe_db: Optional[Any] = None,
        initial_inventory: Optional[Dict[str, int]] = None,
        _share_inventory: bool = False,
        use_synth: bool = False,
        skip_subagent_reward_computation: bool = False,
    ):
        """Initialize the TextCraft environment.

        Args:
            task: The task to execute
            recipes_dir: Path to recipes directory (for original Minecraft recipes)
            recipe_db: Pre-initialized recipe database (for synthetic recipes)
            initial_inventory: Initial inventory
            _share_inventory: If True, share inventory reference with subagents
            use_synth: If True, use synthetic recipe examples in action space
        """
        # Default to original recipes if neither is provided
        if recipe_db is None and recipes_dir is None:
            recipes_dir = Path(__file__).parent / "recipes"

        # Get initial inventory from task if not provided
        if initial_inventory is None and task.misc.get("initial_inventory"):
            initial_inventory = task.misc["initial_inventory"]

        task.fork_strategy = "task"

        code_executor = TextCraftCodeExecutor(
            task,
            recipes_dir=recipes_dir,
            recipe_db=recipe_db,
            inventory=initial_inventory,
            _share_inventory=_share_inventory,
            use_synth=use_synth,
        )
        super().__init__(task, code_executor)
        self._recipes_dir = recipes_dir
        self._recipe_db = recipe_db
        self._use_synth = use_synth
        self._skip_subagent_reward_computation = skip_subagent_reward_computation
        # Always copy initial inventory for bookkeeping (to compare against for evaluation)
        # This is separate from whether the working inventory is shared with parent
        self._initial_inventory = initial_inventory.copy() if initial_inventory else {}

    async def evaluate(self) -> Tuple[float, dict]:
        """Evaluate if the task goal is achieved."""
        score = 0.0
        reward_misc = {}
        is_subagent_task = "textcraft" not in (self._task.id or "")
        if self._skip_subagent_reward_computation and is_subagent_task:
            reward_misc["success"] = False
            reward_misc["reward/success"] = 0.0
            return 0.0, reward_misc

        # Only give reward if agent explicitly called finish()
        if self._state.finished:
            finish_msg = finish_message.get()
            if finish_msg:
                # Check if the task goal is met
                # The task.misc should contain target items
                target_items = self._task.misc.get("target_items", {})

                # Must have target items defined
                if not target_items:
                    reward_misc["success"] = False
                    reward_misc["error"] = "No target items defined in task"
                    return score, reward_misc

                inventory = self._code_executor.inventory

                # Check if all target items were CRAFTED (difference from initial inventory)
                # This prevents giving credit for items that were already in starting inventory
                all_met = True
                missing_items = {}
                for item, required_count in target_items.items():
                    current_count = inventory.get(item, 0)
                    initial_count = self._initial_inventory.get(item, 0)
                    crafted_count = current_count - initial_count
                    if crafted_count < required_count:
                        all_met = False
                        missing_items[item] = required_count - crafted_count

                if all_met:
                    score = 1.0
                    reward_misc["success"] = True
                    reward_misc["target_items"] = target_items
                    reward_misc["initial_inventory"] = dict(self._initial_inventory)
                    reward_misc["final_inventory"] = dict(inventory)
                else:
                    reward_misc["success"] = False
                    reward_misc["missing_items"] = missing_items
                    reward_misc["target_items"] = target_items
                    reward_misc["initial_inventory"] = dict(self._initial_inventory)
                    reward_misc["final_inventory"] = dict(inventory)
            else:
                # Episode finished but agent didn't call finish() - timeout/max steps
                reward_misc["success"] = False
                reward_misc["error"] = "Episode ended without calling finish() - likely timeout or max steps reached"

        reward_misc["reward/success"] = score
        return score, reward_misc

    async def reset(self) -> CodeActObservation:
        """Reset the environment and set action space."""
        obs = await super().reset()
        # Set action space description on both state and observation
        # (obs may be a deepcopy of state, so we need to set both)
        action_space = await self._code_executor.describe_action_space()
        self._state.action_space = action_space
        obs.action_space = action_space
        return obs

    async def fork(self, task: Task) -> "TextCraftEnv":
        """Fork the environment for a subagent."""
        # Parse the goal string to extract targets if it's a crafting task
        targets = self._parse_craft_targets_from_goal(task.goal)

        if targets:
            # Update task.misc with TextCraft-specific data
            task.misc = task.misc.copy() if task.misc else {}
            task.misc.update(
                {
                    "target_items": targets,
                    "initial_inventory": self._code_executor.inventory,  # Current state snapshot for display
                }
            )

        # Create forked environment sharing the same inventory reference
        # This allows subagent changes to automatically propagate to parent
        forked_env = TextCraftEnv(
            task=task,
            recipes_dir=self._recipes_dir,
            recipe_db=self._recipe_db,
            initial_inventory=self._code_executor.inventory,
            _share_inventory=True,  # Share inventory by reference for subagent propagation
            use_synth=self._use_synth,
            skip_subagent_reward_computation=self._skip_subagent_reward_computation,
        )

        return forked_env

    def _parse_craft_targets_from_goal(self, goal: str) -> Optional[Dict[str, int]]:
        """
        Parse crafting targets from a goal string.

        Expected format: "Craft the following items: 2x stick, 1x oak_planks"

        Returns:
            Dictionary of {item: count} or None if not a crafting goal
        """
        import re

        # Check if this is a crafting goal
        if not goal or "Craft the following items:" not in goal:
            return None

        # Extract the items part after "Craft the following items:"
        match = re.search(r"Craft the following items:\s*(.+)", goal)
        if not match:
            return None

        items_str = match.group(1).strip()

        # Parse "2x stick, 1x oak_planks" format
        targets = {}
        # Match patterns like "2x stick" or "1x oak_planks" or "4x c7_i2" (with digits)
        for item_match in re.finditer(r"(\d+)x\s+([a-z0-9_]+)", items_str):
            count = int(item_match.group(1))
            item_name = item_match.group(2)
            targets[item_name] = count

        return targets if targets else None


class TextCraftRecursiveEnv(TextCraftEnv):
    """Environment for TextCraft crafting tasks with recursive agent spawning."""

    def __init__(
        self,
        task: Task,
        recipes_dir: Optional[Path] = None,
        recipe_db: Optional[Any] = None,
        initial_inventory: Optional[Dict[str, int]] = None,
        _share_inventory: bool = False,
        use_synth: bool = False,
        skip_subagent_reward_computation: bool = False,
    ):
        super().__init__(
            task,
            recipes_dir=recipes_dir,
            recipe_db=recipe_db,
            initial_inventory=initial_inventory,
            _share_inventory=_share_inventory,
            use_synth=use_synth,
            skip_subagent_reward_computation=skip_subagent_reward_computation,
        )
        # Use self._recipes_dir and self._initial_inventory which were set by parent class
        # (parent applies defaults: recipes_dir from __file__, inventory from task.misc)
        # Replace executor with Recursive version, sharing the same inventory reference
        self._code_executor = TextCraftRecursiveCodeExecutor(
            task,
            recipes_dir=self._recipes_dir,
            recipe_db=self._recipe_db,
            inventory=self._code_executor.inventory,  # Use parent's executor inventory
            _share_inventory=True,  # Always share since we're using parent's inventory dict
            use_synth=self._use_synth,
        )

    def _get_subagent_stats_and_reset(self) -> tuple[int, float]:
        """Get per-step unique launched children and summed child success score."""
        stats = self._code_executor.get_subagent_stats()
        self._code_executor.reset_subagent_stats()
        return stats

    async def evaluate(self) -> Tuple[float, dict]:
        score, reward_misc = await super().evaluate()

        launched, success_total = self._get_subagent_stats_and_reset()
        reward_misc["reward/subagent_launched"] = launched
        reward_misc["reward/subagent_succeeded"] = success_total

        return score, reward_misc

    async def fork(self, task: Task) -> "TextCraftRecursiveEnv":
        """Fork the environment for a subagent."""
        # Parse the goal string to extract targets if it's a crafting task
        targets = self._parse_craft_targets_from_goal(task.goal)

        if targets:
            # Update task.misc with TextCraft-specific data
            task.misc = task.misc.copy() if task.misc else {}
            task.misc.update(
                {
                    "target_items": targets,
                    "initial_inventory": self._code_executor.inventory,
                }
            )

        # Create forked environment sharing the same inventory reference
        forked_env = TextCraftRecursiveEnv(
            task=task,
            recipes_dir=self._recipes_dir,
            recipe_db=self._recipe_db,
            initial_inventory=self._code_executor.inventory,
            _share_inventory=True,
            use_synth=self._use_synth,
            skip_subagent_reward_computation=self._skip_subagent_reward_computation,
        )

        return forked_env


class TextCraftDepthAwareCodeExecutor(TextCraftRecursiveCodeExecutor):
    """Code executor for depth-aware budget tracking.

    Subagent budget is fixed at construction time — the agent does not
    specify ``num_steps`` when calling ``launch_subagent``.
    """

    def __init__(
        self,
        task: Task,
        subagent_max_steps: int = 25,
        recipes_dir: Optional[Path] = None,
        recipe_db: Optional[Any] = None,
        inventory: Optional[Dict[str, int]] = None,
        _share_inventory: bool = False,
        use_synth: bool = False,
    ):
        self._subagent_max_steps = subagent_max_steps
        super().__init__(
            task,
            recipes_dir=recipes_dir,
            recipe_db=recipe_db,
            inventory=inventory,
            _share_inventory=_share_inventory,
            use_synth=use_synth,
        )

    # ------------------------------------------------------------------
    # Override launch_subagent: drop num_steps from the signature
    # ------------------------------------------------------------------
    async def launch_subagent(self, targets: Dict[str, int], context: str = "") -> str:
        """Launch a subagent to craft the specified targets.

        The subagent budget is fixed and does not need to be specified.

        Args:
            targets: Dictionary mapping item names to target counts.
            context: Optional context string for the subagent.

        Returns:
            Message from the subagent indicating success or failure.
        """
        from platoon.episode.context import current_trajectory, current_trajectory_collection

        target_str = ", ".join([f"{count}x {item}" for item, count in targets.items()])
        goal = f"Craft the following items: {target_str}"
        if context:
            goal += f"\n\nContext provided from parent agent: {context}"

        # Track trajectories before launch to find the new child
        traj_collection = current_trajectory_collection.get()
        current_traj = current_trajectory.get()
        traj_ids_before = set(traj_collection.trajectories.keys())

        result = await _launch_subagent(goal=goal, max_steps=self._subagent_max_steps)

        for traj_id, traj in traj_collection.trajectories.items():
            if traj_id in traj_ids_before or traj_id in self._launched_subagent_ids_this_step:
                continue
            if not traj.parent_info or traj.parent_info.id != current_traj.id:
                continue
            self._launched_subagent_ids_this_step.add(traj_id)
            success_reward = 0.0
            if traj.steps:
                final_step = traj.steps[-1]
                reward_misc = final_step.misc.get("reward_misc", {})
                success_reward = float(reward_misc.get("reward/success", 0.0))
            self._subagent_success_by_child_this_step[traj_id] = success_reward
        return result

    # ------------------------------------------------------------------
    # Action space description — no budget / num_steps references
    # ------------------------------------------------------------------
    def _get_action_space_parts(self, include_subagent: bool = False) -> str:
        if self.use_synth:
            actions = """Available Actions (python functions):

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
            if include_subagent:
                actions += """
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
        else:
            actions = """Available Actions (python functions):

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
            if include_subagent:
                actions += """
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
        return actions

    async def describe_action_space(self) -> str:
        return self._get_action_space_parts(include_subagent=True)

    async def fork(self, task: Task) -> "TextCraftDepthAwareCodeExecutor":
        return TextCraftDepthAwareCodeExecutor(
            task=task,
            subagent_max_steps=self._subagent_max_steps,
            recipes_dir=self.recipes_dir,
            recipe_db=self.recipe_db,
            inventory=self.inventory,
            _share_inventory=True,
            use_synth=self.use_synth,
        )


class TextCraftDepthAwareEnv(TextCraftRecursiveEnv):
    """Environment for depth-aware recursive TextCraft training.

    Uses ``TextCraftDepthAwareCodeExecutor`` so agents do not specify
    ``num_steps`` when delegating.
    """

    def __init__(
        self,
        task: Task,
        subagent_max_steps: int = 25,
        recipes_dir: Optional[Path] = None,
        recipe_db: Optional[Any] = None,
        initial_inventory: Optional[Dict[str, int]] = None,
        _share_inventory: bool = False,
        use_synth: bool = False,
        skip_subagent_reward_computation: bool = False,
    ):
        # Let the parent chain set up defaults (recipes_dir, initial_inventory, etc.)
        super().__init__(
            task,
            recipes_dir=recipes_dir,
            recipe_db=recipe_db,
            initial_inventory=initial_inventory,
            _share_inventory=_share_inventory,
            use_synth=use_synth,
            skip_subagent_reward_computation=skip_subagent_reward_computation,
        )
        self._subagent_max_steps = subagent_max_steps

        # Replace the executor with the depth-aware version
        self._code_executor = TextCraftDepthAwareCodeExecutor(
            task,
            subagent_max_steps=subagent_max_steps,
            recipes_dir=self._recipes_dir,
            recipe_db=self._recipe_db,
            inventory=self._code_executor.inventory,
            _share_inventory=True,
            use_synth=self._use_synth,
        )

    async def fork(self, task: Task) -> "TextCraftDepthAwareEnv":
        targets = self._parse_craft_targets_from_goal(task.goal)
        if targets:
            task.misc = task.misc.copy() if task.misc else {}
            task.misc.update(
                {
                    "target_items": targets,
                    "initial_inventory": self._code_executor.inventory,
                }
            )

        return TextCraftDepthAwareEnv(
            task=task,
            subagent_max_steps=self._subagent_max_steps,
            recipes_dir=self._recipes_dir,
            recipe_db=self._recipe_db,
            initial_inventory=self._code_executor.inventory,
            _share_inventory=True,
            use_synth=self._use_synth,
            skip_subagent_reward_computation=self._skip_subagent_reward_computation,
        )


# Factory functions for synthetic recipes
def create_synth_env(task: Task, items_per_domain_tier: int = 25, **kwargs) -> TextCraftEnv:
    """Create a TextCraftEnv with synthetic recipes.

    Args:
        task: The task to execute
        items_per_domain_tier: Number of items per domain per tier (must match dataset generation)
        **kwargs: Additional arguments passed to TextCraftEnv

    Returns:
        TextCraftEnv configured for synthetic recipes
    """
    from .synth_recipe_loader import SynthRecipeLoader

    recipe_db = SynthRecipeLoader(items_per_domain_tier=items_per_domain_tier)
    return TextCraftEnv(task, recipe_db=recipe_db, use_synth=True, **kwargs)


def create_synth_recursive_env(
    task: Task,
    items_per_domain_tier: int = 25,
    **kwargs,
) -> TextCraftRecursiveEnv:
    """Create a TextCraftRecursiveEnv with synthetic recipes.

    Args:
        task: The task to execute
        items_per_domain_tier: Number of items per domain per tier (must match dataset generation)
        **kwargs: Additional arguments passed to TextCraftRecursiveEnv

    Returns:
        TextCraftRecursiveEnv configured for synthetic recipes
    """
    from .synth_recipe_loader import SynthRecipeLoader

    recipe_db = SynthRecipeLoader(items_per_domain_tier=items_per_domain_tier)
    return TextCraftRecursiveEnv(
        task,
        recipe_db=recipe_db,
        use_synth=True,
        **kwargs,
    )


def create_synth_depth_aware_env(
    task: Task,
    subagent_max_steps: int = 25,
    items_per_domain_tier: int = 25,
    **kwargs,
) -> TextCraftDepthAwareEnv:
    """Create a TextCraftDepthAwareEnv with synthetic recipes.

    Each agent (root and subagents) gets an independent step budget of
    *subagent_max_steps*.  The agent does not need to specify the budget
    when delegating.

    Args:
        task: The task to execute
        subagent_max_steps: Fixed step budget per agent (root and subagents)
        items_per_domain_tier: Number of items per domain per tier (must match dataset generation)
        **kwargs: Additional arguments passed to TextCraftDepthAwareEnv

    Returns:
        TextCraftDepthAwareEnv configured for synthetic recipes
    """
    from .synth_recipe_loader import SynthRecipeLoader

    recipe_db = SynthRecipeLoader(items_per_domain_tier=items_per_domain_tier)
    return TextCraftDepthAwareEnv(
        task,
        subagent_max_steps=subagent_max_steps,
        recipe_db=recipe_db,
        use_synth=True,
        **kwargs,
    )
