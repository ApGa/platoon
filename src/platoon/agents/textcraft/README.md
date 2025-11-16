# TextCraft Training Setup

This directory contains the training setup for the TextCraft environment with recursive agent spawning capabilities.

## Overview

TextCraft is a crafting-based environment inspired by Minecraft recipes where agents learn to:
- Craft items by combining ingredients
- Plan hierarchical crafting sequences
- Recursively spawn subagents to handle subtasks
- Manage shared inventory across parent and subagents

## Files

### Agent & Environment
- `agent.py` - TextCraft agent (extends CodeActAgent with subagent support)
- `rollout.py` - Single rollout execution logic
- `areal_workflow.py` - AReaL workflow integration for distributed training

### Training
- `textcraft_reinforce_plus_plus.py` - Main training script (REINFORCE++ algorithm)
- `textcraft_reinforce_plus_plus.yaml` - Training configuration

### Testing
- `test_rollout.py` - Test script for single rollout execution

## Quick Start

### 1. Test a Single Rollout

```bash
# Make sure you have an LLM endpoint running
# Update the base_url in test_rollout.py if needed

python test_rollout.py textcraft_train_0
```

### 2. Launch Training

```bash
# From the platoon root directory
cd /mnt/efs/platoon

# Launch training with AReaL
python -m areal.launcher \
    --config src/platoon/train/textcraft_reinforce_plus_plus.yaml \
    --launch
```

## Configuration

Key configuration parameters in `textcraft_reinforce_plus_plus.yaml`:

- **Model**: `Qwen/Qwen2.5-7B-Instruct` (configurable via `actor.path`)
- **Max Steps**: 25 steps per episode
- **Batch Size**: 128 tasks per batch
- **Context Length**: 8192 tokens
- **Training Epochs**: 10 epochs
- **Learning Rate**: 5e-6

### Adjusting for Your Setup

1. **Model Selection**: Change `actor.path` to your preferred model
2. **Compute Resources**: Adjust `cluster.n_gpus_per_node` based on available GPUs
3. **Batch Size**: Tune `train_dataset.batch_size` based on memory
4. **Max Steps**: Adjust `workflow_config.max_steps_per_rollout` for task complexity

## Action Space

The TextCraft agent has access to:

1. **`craft(item: str, target_count: int, ingredients: Dict[str, int])`**
   - Craft items using ingredients
   - Supports tag-based ingredients (e.g., "tag:planks")
   
2. **`get_info(items: List[str])`**
   - Query recipe information for items
   - Returns ingredients, result counts, and crafting depth

3. **`view_inventory()`**
   - View current inventory state

4. **`launch_subagent(targets: Dict[str, int], num_steps: int)`**
   - Recursively spawn a subagent to craft specific items
   - Shares inventory by reference (changes propagate automatically)

5. **`finish(message: str)`**
   - Complete the episode with a message

## Dataset

The environment uses pre-generated crafting tasks:
- **Training**: 1000 tasks (`textcraft_train.jsonl`)
- **Validation**: 100 tasks (`textcraft_val.jsonl`)

Each task includes:
- `goal` - Natural language description
- `target_items` - Items to craft with counts
- `initial_inventory` - Starting materials
- `gold_trajectory` - Optimal solution sequence

Tasks are stored in: `/mnt/efs/platoon/src/platoon/envs/textcraft/`

## Training Features

### Recursive Agent Spawning
- Agents can launch subagents to handle subtasks
- Shared inventory semantics (changes propagate automatically)
- Subagents inherit parent's state and action space

### Hierarchical Tasks
- Tasks require multi-step planning
- Intermediate items must be crafted before final targets
- Depths range from simple (depth 1) to complex (depth 3+)

### Gold Trajectories
- Each task includes an optimal solution
- Used for verification and curriculum learning
- Includes concrete crafting steps with exact ingredients

## Monitoring

Training metrics are logged to:
- **WandB**: Project `textcraft-platoon`
- **Local Logs**: `/mnt/efs/tmp/areal/experiments/textcraft-reinforce-plus-plus-trial1/`

Key metrics:
- `task_reward` - Success rate (1.0 for success, 0.0 for failure)
- `actor_loss` - REINFORCE loss
- `entropy` - Policy entropy
- `num_steps` - Steps per episode

## Troubleshooting

### Out of Memory
- Reduce `train_dataset.batch_size`
- Reduce `actor.mb_spec.max_tokens_per_mb`
- Enable gradient checkpointing (already enabled)

### Training Too Slow
- Increase `rollout.max_concurrent_rollouts`
- Enable `async_training: true`
- Increase `rollout.max_head_offpolicyness`

### Low Success Rate
- Increase `workflow_config.max_steps_per_rollout`
- Adjust learning rate in `actor.optimizer.lr`
- Enable reward normalization (already enabled)

## Advanced: Custom Tasks

To generate custom tasks:

```python
from platoon.envs.textcraft.tasks import create_textcraft_datasets

# Generate new datasets
create_textcraft_datasets(
    train_size=1000,
    val_size=100,
    max_target_items=3,
    min_depth=2,
    max_depth=4,
    seed=42
)
```

## Architecture

```
TextCraftAgent (CodeActAgent)
    ↓
TextCraftEnv (CodeActEnv)
    ↓
TextCraftCodeExecutor (IPythonCodeExecutor + ForkableCodeExecutor)
    ↓
RecipeDatabase (Minecraft recipes)
```

## Citation

If you use this environment, please cite:
```
TextCraft: A hierarchical crafting environment for training 
recursive agent systems with shared state semantics.
```

