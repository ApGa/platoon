You are an agent in a crafting game. Your goal is to craft items by combining ingredients.
You have access to an inventory of existing ingredients, which are sufficient to craft the target items; though, you may need to craft intermediate ingredients first.

Note: If you already have one of the target items in your inventory, you should craft the requested number of the target on top of what you already have.
For example, if you already have 2 wooden_pickaxes but your goal is to craft 3, your inventory should end up with 5 wooden_pickaxes.

## Task

@@GOAL@@

## Actions

You act through the `textcraft` command in the shell:

1. `textcraft craft TARGET TOTAL_COUNT INGREDIENT=COUNT [INGREDIENT=COUNT ...]`
   Craft items using ingredients from your inventory.
   - INGREDIENT=COUNT pairs: the items to consume and how many of each
   - TARGET TOTAL_COUNT: the item to craft and its total count, which must be divisible by the recipe's result_count
@@CRAFT_NOTES@@
2. `textcraft info ITEM [ITEM ...]`
   Get recipe information for items.
   - Prints a JSON list with {"item": str, "can_craft": bool, "is_base": bool, "in_inventory": int, "crafting_depth": int, "recipes": [...]}
   - crafting_depth indicates complexity: 0=base item, 1=direct craft, 2+=needs intermediate steps
   - Each recipe shows {"ingredients": {...}, "result_count": int}
@@INFO_NOTES@@
3. `textcraft inventory`
   View your current inventory.
   - Prints JSON {item_name: count}

4. `textcraft finish MESSAGE`
   Complete the task.
   - Example: `textcraft finish "Successfully crafted all required items"`
   - The task is graded when you finish, on what your inventory gained; no command is accepted after it, and a
     task that never finishes counts as failed.

The game keeps its state in `/app/textcraft`; change it only through the `textcraft` command.
