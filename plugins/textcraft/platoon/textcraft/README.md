# TextCraft Environment

A crafting game environment for training and testing LLM agents that can recursively spawn other agents.

## Overview

TextCraft provides two crafting environments:

1. **TextCraft** (Original): Minecraft-inspired crafting with recipes derived from actual Minecraft. Max crafting depth of ~4.
2. **TextCraft-Synth** (New): Synthetic procedurally-generated crafting world with much deeper hierarchies (up to depth 12) and difficulty-tagged tasks.

Both environments support hierarchical crafting tasks that require multiple steps, making them ideal for testing recursive agent spawning capabilities.

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
uv run python3 -m areal.launcher.local \
    platoon/textcraft/train_areal.py \
    --config platoon/textcraft/textcraft_areal.yaml \
    experiment_name=textcraft-reinforce \
    trial_name=trial0
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

### AReaL Config (`textcraft_areal.yaml`)

See the config file for available options.

## Components

### Environment (`env.py`)

- **TextCraftEnv**: Unified environment class supporting both original and synthetic recipes
- **TextCraftRecursiveEnv**: Environment with subagent support and tracking
- **TextCraftCodeExecutor**: Code executor with crafting actions
- **create_synth_env()**: Factory function for creating synth environments
- **create_synth_recursive_env()**: Factory function for synth environments with subagent support

The environment can be configured via parameters:
- `recipes_dir`: Path to recipe directory (for original Minecraft recipes)
- `recipe_db`: Pre-initialized recipe database (for synthetic or custom recipes)
- `use_synth`: Use synthetic examples in action space prompts

### Actions

1. **craft(ingredients: dict, target: tuple[str, int])** -> str
   - Craft an item using ingredients dictionary and target (item_name, count)
   - Example: `craft({"stick": 2, "planks": 3}, ("wooden_pickaxe", 1))`

2. **get_info(items: list)** -> list[dict]
   - Get information about items (recipes, whether they can be crafted, etc.)
   - Example: `get_info(["yellow_dye", "yellow_terracotta"])`

3. **view_inventory()** -> dict
   - View your current inventory
   - Example: `inv = view_inventory()`

4. **finish(message: str)** -> str
   - Complete the task with a message
   - Example: `finish("Successfully crafted all required items")`

5. **launch_subagent(targets: dict, num_steps: int)** -> str (Recursive envs only)
   - Launch a subagent to craft specific targets
   - Example: `launch_subagent({"yellow_dye": 1}, 10)`
   - The subagent will have access to the same inventory and recipes

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

---

# TextCraft-Synth: Synthetic Deep Crafting Dataset

TextCraft-Synth is a procedurally-generated crafting dataset with much deeper crafting hierarchies than the original Minecraft-based TextCraft.

## Key Features

- **Deeper Hierarchies**: Crafting depths from 2 to 12 (vs. max 4 in original)
- **Difficulty Tagging**: Tasks tagged by difficulty level
  - `easy`: depth 2-3 (simple multi-step crafting)
  - `medium`: depth 4-6 (moderate complexity)
  - `hard`: depth 7-9 (deep hierarchies)  
  - `extreme`: depth 10-12 (very deep, cross-domain crafting)
- **Generic Item Names**: Items use abstract names (e.g., `m0_i1`, `c2_i3`) to prevent LLM priors from "cheating" - agents must use `get_info()` to discover recipes
- **Material Domains**: 5 distinct material domains (metal, crystal, organic, arcane, tech)
- **Cross-Domain Items**: High-tier items require materials from multiple domains
- **~380 unique items** across 13 tiers of crafting depth
- **No Train/Val Overlap**: Target items are split at the item level to ensure clean generalization testing

## Generating the Synthetic Dataset

### Generate Recipes

```bash
python -m platoon.textcraft.synth_recipe_generator \
    --output-dir platoon/textcraft/synth_recipes \
    --seed 42 \
    --items-per-tier 8
```

### Generate Tasks

```bash
python -m platoon.textcraft.synth_tasks \
    --num_samples 10000 \
    --eval_size 1000 \
    --seed 42
```

To generate tasks of a specific difficulty only:

```bash
python -m platoon.textcraft.synth_tasks \
    --num_samples 1000 \
    --eval_size 100 \
    --difficulty extreme
```

## Using TextCraft-Synth

### Loading Tasks

```python
from platoon.textcraft import (
    get_synth_task, 
    get_synth_task_ids,
    get_synth_task_ids_by_difficulty,
    Difficulty
)

# Get all task IDs
train_ids = get_synth_task_ids("train", num_samples_train=10000, num_samples_val=1000)
val_ids = get_synth_task_ids("val", num_samples_train=10000, num_samples_val=1000)

# Get task IDs filtered by difficulty
hard_train_ids = get_synth_task_ids_by_difficulty("train", Difficulty.HARD)

# Load a task
task = get_synth_task("textcraft_synth.train.0")
```

### Using the Environment

```python
from platoon.textcraft import create_synth_env, get_synth_task

task = get_synth_task("textcraft_synth.train.0")
env = create_synth_env(task)

obs = await env.reset()
# obs contains task description, action space, and initial state
```

### With Subagent Support

```python
from platoon.textcraft import create_synth_recursive_env, get_synth_task

task = get_synth_task("textcraft_synth.train.0")
env = create_synth_recursive_env(
    task,
    per_step_subagent_success_reward=0.1,
    per_step_subagent_reward_ceiling=0.3
)
```

### Using the Unified TextCraftEnv Directly

```python
from platoon.textcraft import TextCraftEnv, get_synth_task
from platoon.textcraft import SynthRecipeLoader

# Load synthetic recipes
recipe_db = SynthRecipeLoader()

task = get_synth_task("textcraft_synth.train.0")
env = TextCraftEnv(task, recipe_db=recipe_db, use_synth=True)
```

## Task Format

Tasks include difficulty and depth metadata:

```json
{
  "goal": "Craft the following items: 1x m5_i3, 2x c3_i2",
  "id": "textcraft_synth.train.0",
  "max_steps": 75,
  "misc": {
    "target_items": {"m5_i3": 1, "c3_i2": 2},
    "initial_inventory": {"raw_m0": 50, "raw_m1": 30, "raw_c0": 20, ...},
    "difficulty": "hard",
    "max_depth": 9,
    "num_craft_steps": 45,
    "gold_trajectory": [...]
  }
}
```

**Note**: All tasks have a fixed budget of 75 steps. The action space prompt encourages agents to batch actions for efficiency.

## Training with TextCraft-Synth

### Tinker Backend

```bash
uv run python -m platoon.textcraft.train_tinker_synth \
    --config platoon/textcraft/textcraft_synth_tinker.yaml
```

The training script uses 10,000 training tasks and evaluates on 100 validation tasks (sampled from the 1,000 validation set for faster iteration).

## Architecture

### Material Domains

Each domain has its own crafting progression with generic naming:

| Domain | Prefix | Base Items | Characteristics |
|--------|--------|------------|-----------------|
| Metal | `m` | `raw_m0`, `raw_m1`, ... | Traditional materials |
| Crystal | `c` | `raw_c0`, `raw_c1`, ... | Gemstone-based |
| Organic | `o` | `raw_o0`, `raw_o1`, ... | Natural materials |
| Arcane | `a` | `raw_a0`, `raw_a1`, ... | Magical elements |
| Tech | `t` | `raw_t0`, `raw_t1`, ... | Electronic components |

### Tier Progression

Items follow a generic naming scheme: `{domain_prefix}{variant}_i{tier}`

| Tier | Depth | Example Items |
|------|-------|---------------|
| 0 | 0 | `raw_m0`, `raw_c1`, `raw_t2` (base materials) |
| 1 | 1 | `m0_i1`, `c1_i1`, `t2_i1` |
| 2 | 2 | `m0_i2`, `c1_i2`, `t2_i2` |
| 3 | 3 | `m0_i3`, `c1_i3`, `t2_i3` |
| ... | ... | ... |
| 10 | 10 | `m0_i10`, cross-domain items |
| 11 | 11 | `x{n}_mc`, `x{n}_mct` (multi-domain) |
| 12 | 12 | `x{n}_mcao`, `x{n}_mcaot` (all domains) |

### Cross-Domain Items (Tier 10-12)

The highest tier items combine materials from multiple domains:

| Pattern | Domains Combined |
|---------|------------------|
| `x{n}_mc` | Metal + Crystal |
| `x{n}_mct` | Metal + Crystal + Tech |
| `x{n}_mcao` | Metal + Crystal + Arcane + Organic |
| `x{n}_mcaot` | All 5 domains |


