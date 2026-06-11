# AppWorld

This plugin adds AppWorld tasks to Platoon. AppWorld is a benchmark of API-based agent tasks over simulated day-to-day apps and users.

## Install

```bash
cd plugins/appworld
uv sync --extra areal
```

Install AppWorld data once:

```bash
export APPWORLD_ROOT=/path/to/appworld
uv run appworld install
uv run appworld download data
```

Use the same `APPWORLD_ROOT` when running training or inference.

## Train

```bash
uv run python3 platoon/appworld/train_scripts/areal/train_areal.py \
  --config platoon/appworld/configs/areal/appworld_ctx40000_4b-linear.yaml
```

## Inference

```bash
uv run python -m platoon.appworld.run_inference \
  --config platoon/appworld/configs/inference/appworld_inference.yaml
```
