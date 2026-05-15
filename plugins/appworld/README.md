# AppWorld Plugin

This plugin adds support for the AppWorld environment which contains 9 day-to-day apps, operable via 457 APIs, populated with digital activities of ~100 people living in a simulated world, and an associated benchmark of natural, diverse, and challenging autonomous agent tasks requiring rich and interactive coding.

## Installation

### Basic Installation

```bash
cd plugins/number-search
uv sync
```

### With Training Backend

Currently, AppWorld training is only supported with the AReaL training backend.

**AReaL Backend** (requires uv):
```bash
uv sync --extra areal --extra wandb
```

### AppWorld-Specific Setup

```bash
export APPWORLD_ROOT="<path where to download appworld data and log internal appworld logging>"
uv run appworld install
uv run appworld download data
```

## Environment Variables

Set the following environment variables before training:

```bash
export APPWORLD_ROOT="<same path used when performing appworld-specific setup>"
# Optional: For WandB logging
export WANDB_API_KEY=your_wandb_api_key
```

## Training

### AReaL Backend

```bash
uv run python3 platoon/appworld/train_areal.py \
    --config platoon/appworld/appworld_areal.yaml \
    scheduler.type=local \
    experiment_name=number-search-reinforce \
    trial_name=trial0
```
