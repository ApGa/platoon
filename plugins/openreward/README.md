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
OpenReward session. At reset time, the env resolves `get_task` itself and uses
the returned task prompt as the agent's first goal, so the root agent does not
spend its first model step bootstrapping the task. Catalog tool calls,
`python_execute`, and `claim_done` still run through the same bridge session and
are exposed through OpenHands tool schemas. Bridge tools declare an empty
OpenHands resource set, so PTC can run independent MCP calls concurrently with
`asyncio.gather(...)`.

## Recursive Programmatic Tool Calling

Set `openreward.enable_programmatic_tool_calling: true` to add OpenHands'
persistent Python tool. Set `openreward.enable_recursive_subagents: true` to add
OpenHands' `task_tracker` plan tool and a Platoon-backed `launch_subagent` tool.
Enable both flags to train recursive agents that orchestrate tool calls and
child agents from PTC:

```python
results = await asyncio.gather(
    atools.launch_subagent(goal="inspect one candidate"),
    atools.launch_subagent(goal="inspect another candidate"),
)
```

Child agents reuse the forked Platoon agent and environment, including whichever
PTC and recursion capabilities are enabled on the parent. OpenReward child
agents reuse the parent's live MCP bridge tools, so parent and child tool calls
operate on the same OpenReward session instead of spawning separate task
sessions. Their trajectories are recorded in the same `TrajectoryCollection`
with parent links, so existing AReaL/Tinker data processing can include
depth-aware samples. Child step budgets are configured through
`openreward.subagent_default_max_steps`, which defaults to 50. Use
`openreward.subagent_max_depth` to cap recursive depth. Successful subagent calls
return the child `finish` message directly, without appending budget metadata;
children that fail before finishing return a short failure status instead of raw
episode-loop diagnostics.
