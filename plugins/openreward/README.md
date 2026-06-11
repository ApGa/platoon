# Platoon OpenReward Plugin

Train OpenHands SDK agents against OpenReward environments.

## Setup

Start an OpenReward environment server first:

```bash
docker run --rm \
  -e OPENREWARD_PORT=8080 \
  -p 8080:8080 \
  ghcr.io/apga/openreward-toolathlon-gym:latest
```

Install the plugin with the backend you want:

```bash
cd plugins/openreward
uv sync --extra areal
uv sync --extra tinker
```

## Train With AReaL

```bash
uv run python -m platoon.openreward.train_scripts.areal.train_areal \
  --config platoon/openreward/configs/areal/toolathlon_openhands_areal.yaml \
  openreward.session_url=http://localhost:8080
```

## Train With Tinker

```bash
uv run python -m platoon.openreward.train_scripts.tinker.train_tinker \
  --config platoon/openreward/configs/tinker/toolathlon_openhands_tinker.yaml \
  openreward.session_url=http://localhost:8080
```

## Inference

```bash
uv run python -m platoon.openreward.inference_scripts.run_inference \
  --config platoon/openreward/configs/inference/toolathlon_openhands_inference.yaml \
  openreward.session_url=http://localhost:8080
```

The rollout attaches an OpenReward MCP bridge to OpenHands. The bridge owns the
OpenReward session, so `get_task`, catalog tool calls, `python_execute`, and
`claim_done` all run against the same environment workspace.
