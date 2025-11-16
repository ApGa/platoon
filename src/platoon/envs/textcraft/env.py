"""TextCraft environment for recursive agent spawning in crafting tasks."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from platoon.envs.codeact import CodeActEnv, CodeActObservation, CodeActAction, CodeActStep, IPythonCodeExecutor, ForkableCodeExecutor
from platoon.envs.base import Task, SubTask
from platoon.agents.actions.common import finish
from platoon.agents.actions.subagent import launch_subagent as _launch_subagent
from platoon.episode.context import finish_message
from .recipe_loader import RecipeDatabase, Recipe


class TextCraftCodeExecutor(IPythonCodeExecutor, ForkableCodeExecutor):
    """Code executor for TextCraft with crafting actions."""
    
    def __init__(self, task: Task, recipes_dir: Path, inventory: Optional[Dict[str, int]] = None):
        self.recipes_dir = Path(recipes_dir)
        self.recipe_db = RecipeDatabase(self.recipes_dir)
        self.inventory = inventory.copy() if inventory else {}
        self.subagent_results: Dict[str, Dict[str, int]] = {}  # subagent_id -> {item: count}
        super().__init__(task, actions=(finish, self.craft, self.get_info, self.view_inventory, self.launch_subagent, asyncio))
    
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
                    info["recipes"].append({
                        "ingredients": recipe.ingredients,
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
    
    async def launch_subagent(self, targets: Dict[str, int], num_steps: int) -> str:
        """
        Launch a subagent to craft the specified targets.
        
        Args:
            targets: Dictionary mapping item names to target counts
            num_steps: Maximum number of steps for the subagent
        
        Returns:
            Message from the subagent indicating success or failure
        """
        # Convert targets dict to a goal string
        target_str = ", ".join([f"{count}x {item}" for item, count in targets.items()])
        goal = f"Craft the following items: {target_str}"
        
        # Use the general launch_subagent function
        # Inventory is shared by reference, so changes propagate automatically
        result = await _launch_subagent(goal=goal, max_steps=num_steps)
        
        return result
    
    async def describe_action_space(self) -> str:
        return """Available Actions:
1. craft(ingredients: dict, target: tuple[str, int]) -> str
   - Craft an item using ingredients dictionary and target (item_name, total_count)
   - target_count is the TOTAL number of items to produce (not recipe executions)
   - target_count MUST be evenly divisible by the recipe's result count
   - If multiple recipes exist for an item, all are tried until one succeeds
   - Example: craft({"stick": 2, "oak_planks": 3}, ("wooden_pickaxe", 1))
   - Example: craft({"oak_log": 4}, ("oak_planks", 16))  # 4 logs → 16 planks (4 items per craft)

2. get_info(items: list) -> list[dict]
   - Get information about items (recipes, whether they can be crafted, etc.)
   - Example: get_info(["yellow_dye", "yellow_terracotta"])

3. finish(message: str) -> str
   - Complete the task with a message
   - Example: finish("Successfully crafted all required items")

4. await launch_subagent(targets: dict, num_steps: int) -> str
   - Launch a subagent to craft specific targets
   - Example: await launch_subagent({"yellow_dye": 1, "stick": 2}, 10)
   - The subagent will have access to the same inventory and recipes
   - Returns the subagent's finish message

5. view_inventory() -> dict
   - View your current inventory
   - Example: inv = view_inventory()
   
NOTE: A good efficient strategy could be to use get_info to get the recipes for target items, view_inventory to understand missing ingredients, and recursively 
call launch_subagent to craft any missing ingredients from the inventory.

Note that asyncio has already been imported for you. You can launch subagents using `await launch_subagent` or `asyncio.create_task` + `await asyncio.gather` to launch multiple subagents concurrently.
"""
    
    async def reset(self) -> 'TextCraftCodeExecutor':
        # Reset inventory to initial state if needed
        return self
    
    async def fork(self, task: Task) -> 'TextCraftCodeExecutor':
        """Fork the executor for a subagent."""
        return TextCraftCodeExecutor(
            task=task,
            recipes_dir=self.recipes_dir,
            inventory=self.inventory  # Share inventory by reference
        )


class TextCraftEnv(CodeActEnv):
    """Environment for TextCraft crafting tasks with recursive agent spawning."""
    
    def __init__(self, task: Task, recipes_dir: Optional[Path] = None, initial_inventory: Optional[Dict[str, int]] = None):
        if recipes_dir is None:
            recipes_dir = Path(__file__).parent / "recipes"
        
        # Get initial inventory from task if not provided
        if initial_inventory is None and task.misc.get("initial_inventory"):
            initial_inventory = task.misc["initial_inventory"]
        
        code_executor = TextCraftCodeExecutor(task, recipes_dir, initial_inventory)
        super().__init__(task, code_executor)
        self._recipes_dir = recipes_dir
        self._initial_inventory = initial_inventory.copy() if initial_inventory else {}

    
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
                
                # Check if all target items are in inventory with sufficient counts
                all_met = True
                missing_items = {}
                for item, required_count in target_items.items():
                    available = inventory.get(item, 0)
                    if available < required_count:
                        all_met = False
                        missing_items[item] = required_count - available
                
                if all_met:
                    score = 1.0
                    reward_misc["success"] = True
                    reward_misc["target_items"] = target_items
                    reward_misc["final_inventory"] = dict(inventory)
                else:
                    reward_misc["success"] = False
                    reward_misc["missing_items"] = missing_items
                    reward_misc["target_items"] = target_items
                    reward_misc["final_inventory"] = dict(inventory)
            else:
                # Episode finished but agent didn't call finish() - timeout/max steps
                reward_misc["success"] = False
                reward_misc["error"] = "Episode ended without calling finish() - likely timeout or max steps reached"
        
        return score, reward_misc
    
    async def reset(self) -> CodeActObservation:
        """Reset the environment and set action space."""
        obs = await super().reset()
        # Set action space description
        self._state.action_space = await self._code_executor.describe_action_space()
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
            initial_inventory=self._code_executor.inventory  # Share by reference
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


