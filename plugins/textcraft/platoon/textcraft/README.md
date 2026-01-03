# TextCraft Environment

A crafting game environment for training and testing LLM agents that can recursively spawn other agents.

## Overview

TextCraft is a Minecraft-inspired crafting environment where agents must craft items by combining ingredients. The environment supports hierarchical crafting tasks that require multiple steps, making it ideal for testing recursive agent spawning capabilities.

## Installation

### Basic Installation

```bash
cd plugins/textcraft
uv sync
```

### With Training Backend

Choose one of the following backends:

**Tinker Backend**:
```bash
uv sync --extra tinker
```

**AReaL Backend** (requires uv):
```bash
uv sync --extra areal
```

**With WandB Logging**:
```bash
uv sync --extra tinker --extra wandb
# or
uv sync --extra areal --extra wandb
```

## Environment Variables

Set the following environment variables before training:

```bash
# Required for Tinker backend
export TINKER_API_KEY=your_tinker_api_key

# Optional: For WandB logging
export WANDB_API_KEY=your_wandb_api_key
```

## Training

### Tinker Backend

```bash
# Basic training
uv run python -m platoon.textcraft.train_tinker \
    --config platoon/textcraft/textcraft_tinker.yaml

# With CLI overrides
uv run python -m platoon.textcraft.train_tinker \
    --config platoon/textcraft/textcraft_tinker.yaml \
    train.num_steps=1000 \
    train.batch_size=8

# With WandB logging
uv run python -m platoon.textcraft.train_tinker \
    --config platoon/textcraft/textcraft_tinker.yaml \
    stats.wandb.enabled=true \
    stats.wandb.project=textcraft
```

### AReaL Backend

```bash
uv run python -m platoon.textcraft.train \
    --config platoon/textcraft/textcraft_reinforce_plus_plus.yaml
```

## Configuration

### Tinker Config (`textcraft_tinker.yaml`)

Key configuration options:
- `train.num_steps`: Number of training steps
- `train.batch_size`: Batch size for training
- `train.rollouts_per_task`: Number of rollouts per task for group advantage
- `train.learning_rate`: Learning rate
- `workflow.timeout`: Timeout for each rollout (seconds)
- `stats.wandb.enabled`: Enable WandB logging

### AReaL Config (`textcraft_reinforce_plus_plus.yaml`)

See the config file for available options.

## Components

### Environment (`env.py`)

- **TextCraftEnv**: Main environment class extending `CodeActEnv`
- **TextCraftCodeExecutor**: Code executor with crafting actions

### Actions

1. **craft(ingredients: dict, target: tuple[str, int])** -> str
   - Craft an item using ingredients dictionary and target (item_name, count)
   - Example: `craft({"stick": 2, "planks": 3}, ("wooden_pickaxe", 1))`

2. **get_info(items: list)** -> list[dict]
   - Get information about items (recipes, whether they can be crafted, etc.)
   - Example: `get_info(["yellow_dye", "yellow_terracotta"])`

3. **finish(message: str)** -> str
   - Complete the task with a message
   - Example: `finish("Successfully crafted all required items")`

4. **launch_subagent(targets: dict, num_steps: int)** -> str
   - Launch a subagent to craft specific targets
   - Example: `launch_subagent({"yellow_dye": 1}, 10)`
   - The subagent will have access to the same inventory and recipes

5. **view_inventory()** -> dict
   - View your current inventory
   - Example: `inv = view_inventory()`

### Recipe Loader (`recipe_loader.py`)

- **RecipeDatabase**: Loads and manages Minecraft crafting recipes
- **Recipe**: Represents a single crafting recipe
- Filters out non-crafting recipes (smelting, stonecutting, etc.)
- Builds dependency graphs for finding hierarchical recipes

### Task Generation (`tasks.py`)

- Generates hierarchical crafting tasks with configurable depth
- Creates train/test splits
- Tasks include:
  - Target items to craft
  - Initial inventory with base materials
  - Maximum steps allowed

### Agent (`agents/textcraft/agent.py`)

- **TextCraftAgent**: Extends `CodeActAgent` with crafting-specific prompts
- **TextCraftPromptBuilder**: Custom prompt builder for crafting tasks
- Supports recursive spawning via `fork()` method

## Usage

### Generating Tasks

```bash
python -m platoon.envs.textcraft.tasks \
    --num_samples 10000 \
    --eval_size 1000 \
    --min_depth 2 \
    --max_depth 5
```

### Loading Tasks

```python
from platoon.textcraft import get_task, get_task_ids

# Get task IDs
train_ids = get_task_ids("train", num_samples_train=10000, num_samples_val=1000)
val_ids = get_task_ids("val", num_samples_train=10000, num_samples_val=1000)

# Load a task
task = get_task("textcraft.train.0")
```

### Using the Environment

```python
from platoon.textcraft import TextCraftEnv
from platoon.textcraft import get_task

task = get_task("textcraft.train.0")
env = TextCraftEnv(task)

obs = await env.reset()
# obs contains task description, action space, and initial state
```

## Task Format

Tasks are stored as JSONL files with the following structure:

```json
{
  "goal": "Craft the following items: 1x yellow_terracotta, 2x wooden_pickaxe",
  "id": "textcraft.train.0",
  "max_steps": 50,
  "misc": {
    "target_items": {
      "yellow_terracotta": 1,
      "wooden_pickaxe": 2
    },
    "initial_inventory": {
      "terracotta": 8,
      "yellow_dye": 1,
      "planks": 9,
      "stick": 2
    }
  }
}
```

## Features

- **Hierarchical Crafting**: Tasks require multiple crafting steps
- **Recursive Spawning**: Agents can spawn subagents to handle intermediate items
- **Inventory Management**: Shared inventory between parent and child agents
- **Filtered Tasks**: Only includes crafting recipes (excludes smelting, stonecutting, etc.)

## Notes

- The environment filters out non-crafting recipes (smelting, stonecutting, campfire cooking, etc.)
- Tags in recipes (e.g., `minecraft:planks`) are handled but may need special logic for validation
- Inventory is shared between parent and child agents when forking
- Task generation focuses on hierarchical recipes with depth 2-5 by default

