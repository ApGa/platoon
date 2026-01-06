"""TextCraft environment for recursive agent spawning in crafting tasks."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from platoon.envs.codeact import CodeActEnv, CodeActObservation, IPythonCodeExecutor, ForkableCodeExecutor
from platoon.envs.base import Task
from platoon.agents.actions.common import finish
from platoon.agents.actions.subagent import launch_subagent as _launch_subagent
from platoon.episode.context import finish_message
from .recipe_loader import RecipeDatabase
from platoon.envs.codeact import safe_asyncio


class TextCraftCodeExecutor(IPythonCodeExecutor, ForkableCodeExecutor):
    """Code executor for TextCraft with crafting actions."""
    
    def __init__(self, task: Task, recipes_dir: Path, inventory: Optional[Dict[str, int]] = None, _share_inventory: bool = False):
        self.recipes_dir = Path(recipes_dir)
        self.recipe_db = RecipeDatabase(self.recipes_dir)
        # When _share_inventory=True, use the inventory dict directly (for subagent propagation)
        # When False (default), make a copy to avoid unintended mutations
        if _share_inventory and inventory is not None:
            self.inventory = inventory
        else:
            self.inventory = inventory.copy() if inventory else {}
        self.subagent_results: Dict[str, Dict[str, int]] = {}  # subagent_id -> {item: count}
        super().__init__(task, actions=(finish, self.craft, self.get_info, self.view_inventory, self.launch_subagent, safe_asyncio))
    
    def craft(self, ingredients: Dict[str, int], target: Tuple[str, int]) -> str:
        """
        Craft an item using ingredients.
        
        Args:
            ingredients: Dictionary mapping item names to counts to consume
            target: Tuple of (item_name, total_count) where total_count is the total number of items to produce
        
        Returns:
            Success message or error message
        """
        import math
        
        # Validate input types and values
        if not isinstance(ingredients, dict):
            return f"Error: ingredients must be a dict, got {type(ingredients).__name__}"
        
        if not isinstance(target, tuple) or len(target) != 2:
            return f"Error: target must be a tuple of (item_name, count)"
        
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
                errors.append(f"Recipe {recipe_idx + 1}: Target count {target_count} is not divisible by recipe result count {recipe.result_count}")
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
                        recipe_error = f"Wrong amount of {recipe_ing}. Need {total_required}, provided {ingredients[recipe_ing]}"
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
                errors.append(f"Recipe {recipe_idx + 1}: Extra ingredients not required: {', '.join(extra_ingredients)}")
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
                "recipes": []
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
                                "use_one_of": concrete_items
                            }
                        else:
                            expanded_ingredients[ing] = count
                    
                    info["recipes"].append({
                        "ingredients": expanded_ingredients,
                        "result_count": recipe.result_count
                    })
            
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
    
    async def describe_action_space(self) -> str:
        return """Available Actions:
1. def craft(ingredients: dict, target: tuple[str, int]) -> str
   - Craft an item using ingredients dictionary and target (item_name, total_count)
   - target_count is the TOTAL number of items to produce (not recipe executions)
   - target_count MUST be evenly divisible by the recipe's result count
   - If multiple recipes exist for an item, all are tried until one succeeds
   - Example: craft({"stick": 2, "oak_planks": 3}, ("wooden_pickaxe", 1))
   - Example: craft({"oak_log": 4}, ("oak_planks", 16))  # 4 logs → 16 planks (4 items per craft)

2. def get_info(items: list) -> list[dict]
   - Get information about items (recipes, whether they can be crafted, etc.)
   - Example: get_info(["yellow_dye", "yellow_terracotta"])
   - IMPORTANT - Tag-based Ingredients:
        Some recipes use "tag:X" ingredients (e.g., "tag:planks", "tag:logs"). These are ingredient categories.
        If an ingredient in a recipe is a tag, concrete items that satisfy the tag will also be listed. 
        When crafting with a tag-based ingredient, you must provide a CONCRETE item from that category, NOT the tag name itself. 

3. def view_inventory() -> dict
   - View your current inventory to see available ingredients.
   - Example: inv = view_inventory()
   
4. def finish(message: str) -> str
   - Complete the task with a message
   - Example: finish("Successfully crafted all required items")
"""
    
    async def reset(self) -> 'TextCraftCodeExecutor':
        # Reset inventory to initial state if needed
        return self
    
    async def fork(self, task: Task) -> 'TextCraftCodeExecutor':
        """Fork the executor for a subagent."""
        return TextCraftCodeExecutor(
            task=task,
            recipes_dir=self.recipes_dir,
            inventory=self.inventory,
            _share_inventory=True  # Share inventory by reference for subagent propagation
        )


class TextCraftRecursiveCodeExecutor(TextCraftCodeExecutor):
    def __init__(self, task: Task, recipes_dir: Path, inventory: Optional[Dict[str, int]] = None, _share_inventory: bool = False):
        super().__init__(task, recipes_dir, inventory, _share_inventory)
        # Track subagent outcomes for current step: (launched_count, success_count)
        self._subagent_stats_this_step: tuple[int, int] = (0, 0)
    
    def reset_subagent_stats(self) -> None:
        """Reset subagent tracking for a new step."""
        self._subagent_stats_this_step = (0, 0)
    
    def get_subagent_stats(self) -> tuple[int, int]:
        """Get (launched_count, success_count) for current step."""
        return self._subagent_stats_this_step
    
    async def launch_subagent(self, targets: Dict[str, int], num_steps: int, context: str = "") -> str:
        """Launch a subagent and track whether it succeeded."""
        from platoon.episode.context import current_trajectory_collection, current_trajectory
        
        # Get current trajectory to identify child
        traj_collection = current_trajectory_collection.get()
        current_traj = current_trajectory.get()
        
        # Track trajectories before launch to find the new one
        traj_ids_before = set(traj_collection.trajectories.keys())
        
        # Call parent's launch_subagent
        result = await super().launch_subagent(targets, num_steps, context)
        
        # Find the new child trajectory and check if it succeeded at its task
        launched, succeeded = self._subagent_stats_this_step
        launched += 1
        
        for traj_id, traj in traj_collection.trajectories.items():
            if traj_id not in traj_ids_before:
                # This is a new trajectory - check if it's our child
                if traj.parent_info and traj.parent_info.id == current_traj.id:
                    # Check the final step's reward_misc for actual task success
                    # (not traj.reward, which includes subagent bonuses)
                    if traj.steps:
                        final_step = traj.steps[-1]
                        reward_misc = final_step.misc.get('reward_misc', {})
                        if reward_misc.get('success', False):
                            succeeded += 1
                    break
        
        self._subagent_stats_this_step = (launched, succeeded)
        return result
    
    async def describe_action_space(self) -> str:
        return """Available Actions:
1. def craft(ingredients: dict, target: tuple[str, int]) -> str
   - Craft an item using ingredients dictionary and target (item_name, total_count)
   - target_count is the TOTAL number of items to produce (not recipe executions)
   - target_count MUST be evenly divisible by the recipe's result count
   - If multiple recipes exist for an item, all are tried until one succeeds
   - Example: craft({"stick": 2, "oak_planks": 3}, ("wooden_pickaxe", 1))
   - Example: craft({"oak_log": 4}, ("oak_planks", 16))  # 4 logs → 16 planks (4 items per craft)

2. def get_info(items: list) -> list[dict]
   - Get information about items (recipes, whether they can be crafted, etc.)
   - Example: get_info(["yellow_dye", "yellow_terracotta"])
   - IMPORTANT - Tag-based Ingredients:
        Some recipes use "tag:X" ingredients (e.g., "tag:planks", "tag:logs"). These are ingredient categories.
        If an ingredient in a recipe is a tag, concrete items that satisfy the tag will also be listed. 
        When crafting with a tag-based ingredient, you must provide a CONCRETE item from that category, NOT the tag name itself. 

3. def view_inventory() -> dict
   - View your current inventory to see available ingredients.
   - Example: inv = view_inventory()
   
4. def finish(message: str) -> str
   - Complete the task with a message
   - Example: finish("Successfully crafted all required items")

5. async def launch_subagent(targets: dict, num_steps: int, context: str = "") -> str
   - Launch a subagent to craft specific targets
   - Example: await launch_subagent({"yellow_dye": 1, "stick": 2}, 20)
   - The subagent will have access to the same inventory and recipes
   - Make sure to provide sufficient num_steps budget to the subagent to complete the task.
   - Returns the subagent's finish message
   - You optionally can provide a context string to the subagent with a summary of any useful context you have gathered for its task,
        to help reduce redundant actions.

Note that asyncio has already been imported for you. You can launch subagents using `await launch_subagent` or `asyncio.create_task` + `await asyncio.gather` to launch multiple subagents concurrently.
"""
    
    async def fork(self, task: Task) -> 'TextCraftRecursiveCodeExecutor':
        """Fork the executor for a subagent."""
        return TextCraftRecursiveCodeExecutor(
            task=task,
            recipes_dir=self.recipes_dir,
            inventory=self.inventory,
            _share_inventory=True
        )


class TextCraftEnv(CodeActEnv):
    """Environment for TextCraft crafting tasks with recursive agent spawning."""
    
    def __init__(self, task: Task, recipes_dir: Optional[Path] = None, initial_inventory: Optional[Dict[str, int]] = None, _share_inventory: bool = False):
        if recipes_dir is None:
            recipes_dir = Path(__file__).parent / "recipes"
        
        # Get initial inventory from task if not provided
        if initial_inventory is None and task.misc.get("initial_inventory"):
            initial_inventory = task.misc["initial_inventory"]
        
        code_executor = TextCraftCodeExecutor(task, recipes_dir, initial_inventory, _share_inventory=_share_inventory)
        super().__init__(task, code_executor)
        self._recipes_dir = recipes_dir
        # Only copy for bookkeeping if not sharing
        self._initial_inventory = initial_inventory if _share_inventory else (initial_inventory.copy() if initial_inventory else {})

    
    async def evaluate(self) -> Tuple[float, dict]:
        """Evaluate if the task goal is achieved."""
        score = 0.0
        reward_misc = {}
        
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
    
    async def fork(self, task: Task) -> 'TextCraftEnv':
        """Fork the environment for a subagent."""
        # Parse the goal string to extract targets if it's a crafting task
        targets = self._parse_craft_targets_from_goal(task.goal)
        
        if targets:
            # Update task.misc with TextCraft-specific data
            task.misc = task.misc.copy() if task.misc else {}
            task.misc.update({
                "target_items": targets,
                "initial_inventory": self._code_executor.inventory,  # Current state snapshot for display
            })
        
        # Create forked environment sharing the same inventory reference
        # This allows subagent changes to automatically propagate to parent
        forked_env = TextCraftEnv(
            task=task,
            recipes_dir=self._recipes_dir,
            initial_inventory=self._code_executor.inventory,
            _share_inventory=True  # Share inventory by reference for subagent propagation
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
        # Match patterns like "2x stick" or "1x oak_planks"
        for item_match in re.finditer(r"(\d+)x\s+([a-z_]+)", items_str):
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
        initial_inventory: Optional[Dict[str, int]] = None, 
        _share_inventory: bool = False,
        per_step_subagent_success_reward: float = 0.0,
        per_step_subagent_reward_ceiling: float = float('inf'),
    ):
        super().__init__(task, recipes_dir, initial_inventory, _share_inventory=_share_inventory)
        # Use self._recipes_dir and self._initial_inventory which were set by parent class
        # (parent applies defaults: recipes_dir from __file__, inventory from task.misc)
        # Replace executor with Recursive version, sharing the same inventory reference
        self._code_executor = TextCraftRecursiveCodeExecutor(
            task, self._recipes_dir, 
            self._code_executor.inventory,  # Use parent's executor inventory (already shared if _share_inventory=True)
            _share_inventory=True  # Always share since we're using parent's inventory dict
        )
        self._per_step_subagent_success_reward = per_step_subagent_success_reward
        self._per_step_subagent_reward_ceiling = per_step_subagent_reward_ceiling

    def _get_subagent_stats_and_reset(self) -> tuple[int, int]:
        """Get (launched_count, success_count) for current step and reset for next step."""
        stats = self._code_executor.get_subagent_stats()
        self._code_executor.reset_subagent_stats()
        return stats

    async def evaluate(self) -> Tuple[float, dict]:
        score, reward_misc = await super().evaluate()
        
        # Get subagent stats for this step (also resets for next step)
        launched, succeeded = self._get_subagent_stats_and_reset()
        
        # Add reward for successful subagents (scaled by count, capped at ceiling)
        subagent_reward = 0.0
        if self._per_step_subagent_success_reward > 0 and succeeded > 0:
            subagent_reward = min(
                self._per_step_subagent_success_reward * succeeded,
                self._per_step_subagent_reward_ceiling
            )
            score += subagent_reward
        
        reward_misc["subagent_launched"] = launched
        reward_misc["subagent_succeeded"] = succeeded
        reward_misc["reward/subagent_success"] = subagent_reward
        
        return score, reward_misc
    
    async def fork(self, task: Task) -> 'TextCraftRecursiveEnv':
        """Fork the environment for a subagent."""
        # Parse the goal string to extract targets if it's a crafting task
        targets = self._parse_craft_targets_from_goal(task.goal)
        
        if targets:
            # Update task.misc with TextCraft-specific data
            task.misc = task.misc.copy() if task.misc else {}
            task.misc.update({
                "target_items": targets,
                "initial_inventory": self._code_executor.inventory,
            })
        
        # Create forked environment sharing the same inventory reference
        # Note: subagent rewards are NOT propagated - each trajectory is evaluated independently
        forked_env = TextCraftRecursiveEnv(
            task=task,
            recipes_dir=self._recipes_dir,
            initial_inventory=self._code_executor.inventory,
            _share_inventory=True,
        )
        
        return forked_env
