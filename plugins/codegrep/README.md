# CodeGrep Plugin

A plugin for training agents for code search and localization, inspired by Cognition's swe-grep.

## Overview

CodeGrep trains agents to find relevant files in a codebase given an issue description. This is useful as a first step in automated software engineering workflows where agents need to identify which files to modify.

## Installation

### Basic Installation

```bash
cd plugins/codegrep
uv sync
```

### With Training Backend

Choose one of the following backends:

**Tinker Backend** (recommended):
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
uv run python -m platoon.codegrep.train_tinker \
    --config platoon/codegrep/codegrep_tinker.yaml

# With CLI overrides
uv run python -m platoon.codegrep.train_tinker \
    --config platoon/codegrep/codegrep_tinker.yaml \
    train.num_steps=1000 \
    train.batch_size=4

# With WandB logging
uv run python -m platoon.codegrep.train_tinker \
    --config platoon/codegrep/codegrep_tinker.yaml \
    stats.wandb.enabled=true \
    stats.wandb.project=codegrep
```

### AReaL Backend

```bash
uv run python3 -m areal.launcher.local \
    platoon/codegrep/train.py \
    --config platoon/codegrep/codegrep_areal.yaml \
    experiment_name=codegrep-reinforce \
    trial_name=trial0
```

## Configuration

### Tinker Config (`codegrep_tinker.yaml`)

Key configuration options:
- `train.num_steps`: Number of training steps
- `train.batch_size`: Batch size for training
- `train.rollouts_per_task`: Number of rollouts per task for group advantage
- `train.learning_rate`: Learning rate
- `workflow.timeout`: Timeout for each rollout (seconds)
- `stats.wandb.enabled`: Enable WandB logging

### AReaL Config (`codegrep_areal.yaml`)

See the config file for available options.

## Environment Details

### Task

Given an issue description from a GitHub repository, the agent must identify the relevant files that need to be modified to address the issue.

### Actions

The agent uses OpenHands tools (grep, file navigation, etc.) to explore the codebase and identify relevant files.

### Rewards

Rewards are based on how well the agent's file list matches the ground truth files that were actually modified in the fix.

## Data

Training data is stored in parquet format:
- `train.parquet`: Training set
- `train_shuffled.parquet`: Shuffled training set

## Dependencies

This plugin depends on:
- `platoon`: Core platoon library
- `platoon-openhands`: OpenHands integration for code exploration tools

