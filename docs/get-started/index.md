# Get started

Platoon is a framework for building agents, training them with reinforcement learning, and — the
part that makes it different — training *systems* of agents that delegate work to each other.

If you have used an RL-for-agents framework before, the shape will be familiar: you write an
environment and an agent, a rollout function ties them together, and a trainer turns rollouts into
gradient steps. What Platoon adds is that an agent can **fork its environment and hand a sub-task
to a child agent**, and the training pipeline understands the resulting tree.

## Pick your path

<div class="pl-cards" markdown>

<div class="pl-card" markdown>
<span class="pl-card__kicker">15 minutes</span>
### [Installation](installation.md)
Install Platoon with `uv`, choose a training backend, and set up the credentials each backend
needs. Includes the Megatron / Transformer Engine story if you need it.
</div>

<div class="pl-card" markdown>
<span class="pl-card__kicker">30 minutes</span>
### [Quickstart](quickstart.md)
Run `number-search` — the smallest complete Platoon plugin — first as a plain rollout, then as a
real RL job.
</div>

<div class="pl-card" markdown>
<span class="pl-card__kicker">Read once, refer often</span>
### [Core concepts](concepts.md)
`Task`, `Env`, `Agent`, `Trajectory`, rollout, workflow, registry. A handful of ideas, and
everything else in Platoon is built from them.
</div>

<div class="pl-card" markdown>
<span class="pl-card__kicker">A decision to make early</span>
### [Choosing a backend](backends.md)
AReaL on your own GPUs, or the managed Tinker service. What each supports, and how much of your
code changes if you switch.
</div>

<div class="pl-card" markdown>
<span class="pl-card__kicker">The main event</span>
### [Your first custom task](first-task.md)
The short version of "make Platoon do *my* thing": a few files, one config block, one command.
</div>

</div>

## What you need

| | |
| --- | --- |
| **Python** | 3.12 — the project pins `requires-python = "~=3.12.0"` |
| **Package manager** | [`uv`](https://docs.astral.sh/uv/) — the lockfiles, extras and conflict rules assume it |
| **To run rollouts only** | Any OpenAI-compatible endpoint. No GPU needed. |
| **To train with AReaL** | NVIDIA GPUs (reference configs use 8 per node); Linux |
| **To train with Tinker** | A Tinker API key. No local GPU needed. |

!!! tip "You can get a long way without a GPU"

    Environments, agents, rollouts, reward functions, and the whole visualization toolchain run
    against a hosted model endpoint. Write and debug your task locally against an API model, then
    point the same code at a training run. See [Evaluate a model
    endpoint](../tutorials/inference.md).

## The short version

```bash
# 1. Install the core plus a backend
uv sync --extra areal          # or: uv sync --extra tinker

# 2. Install a plugin — each one is its own uv project
cd plugins/number-search
uv sync --extra areal

# 3. Train
uv run python3 platoon/number_search/train.py \
  --config platoon/number_search/nv_number_search_cispo_areal.yaml
```

Then read [Core concepts](concepts.md) to find out what just happened, or jump to [Your first
custom task](first-task.md) to make it yours.
