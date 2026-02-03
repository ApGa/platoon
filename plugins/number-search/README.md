# Number Search Plugin

A simple number guessing environment for training and testing LLM agents with reinforcement learning.

## Overview

Number Search is a binary search-style environment where agents must find a target number through guessing. The environment provides feedback ("Too low", "Too high", or "Correct!") after each guess, making it ideal for testing basic reasoning and search strategies.

## Installation

### Basic Installation

```bash
cd plugins/number-search
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
uv run python -m platoon.number_search.train_tinker \
    --config platoon/number_search/number_search_tinker.yaml

# With CLI overrides
uv run python -m platoon.number_search.train_tinker \
    --config platoon/number_search/number_search_tinker.yaml \
    train.num_steps=1000 \
    train.batch_size=8

# With WandB logging
uv run python -m platoon.number_search.train_tinker \
    --config platoon/number_search/number_search_tinker.yaml \
    stats.wandb.enabled=true \
    stats.wandb.project=number-search
```

### AReaL Backend

```bash
uv run python3 -m areal.launcher.local \
    platoon/number_search/train.py \
    --config platoon/number_search/number_search_areal.yaml \
    experiment_name=number-search-reinforce \
    trial_name=trial0
```

## Configuration

### Tinker Config (`number_search_tinker.yaml`)

Key configuration options:
- `train.num_steps`: Number of training steps
- `train.batch_size`: Batch size for training
- `train.rollouts_per_task`: Number of rollouts per task for group advantage
- `train.learning_rate`: Learning rate
- `workflow.timeout`: Timeout for each rollout (seconds)
- `stats.wandb.enabled`: Enable WandB logging

### AReaL Config (`number_search_areal.yaml`)

See the config file for available options.

## Environment Details

### Actions

- **guess(n: int)**: Make a guess, returns "Too low", "Too high", or "Correct!"

### Rewards

- +1.0 for correct guess
- 0.0 otherwise

## Task Format

Tasks are stored in JSONL format:

```json
{
  "id": "number_search.train.0",
  "goal": "Find the hidden number between 1 and 100.",
  "max_steps": 10,
  "misc": {
    "target": 42,
    "min_val": 1,
    "max_val": 100
  }
}
```

