#!/usr/bin/env python3
"""Quick test script for TextCraft get_info function."""

from pathlib import Path
from platoon.textcraft.env import TextCraftCodeExecutor
from platoon.envs.base import Task

# Path to recipes directory
RECIPES_DIR = Path(__file__).parent / "platoon" / "textcraft" / "recipes"

def main():
    # Create a dummy task (required for executor initialization)
    initial_inventory = {"oak_log": 10, "cobblestone": 20, "iron_ingot": 5, "diamond": 3, "stick": 4}
    task = Task(
        goal="Craft the following items: 1x diamond_sword",
        id="test.0",
        max_steps=50,
        misc={
            "target_items": {"diamond_sword": 1},
            "initial_inventory": initial_inventory,
        },
    )
    
    # Initialize the code executor (which has the get_info function)
    executor = TextCraftCodeExecutor(
        task=task,
        recipes_dir=RECIPES_DIR,
        inventory=initial_inventory.copy(),
    )
    
    # # To look up what items satisfy a tag, use recipe_db.get_items_for_tag()
    # print("Items that satisfy 'planks' tag:")
    # planks_items = executor.recipe_db.get_items_for_tag("planks")
    # print(planks_items)
    
    # # Then you can get info on those actual items
    # print("\nInfo for those items:")
    # print(executor.get_info(planks_items))
    print(executor.get_info(['stick', 'shield', 'barrel', 'furnace']))

if __name__ == "__main__":
    main()

