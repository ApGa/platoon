# Get started

Platoon builds agents and trains them with reinforcement learning, including **multi-agent
workflows** in which one agent delegates sub-tasks to other agents — recursion is one case, a planner
calling specialists is another. The pipeline trains the resulting tree of work, not just a flat
rollout. And it composes services rather than owning everything: training engines behind the Tinker
API, environments behind an environment server, inference endpoints.

## Start here

<div class="pl-cards" markdown>

<div class="pl-card" markdown>
<span class="pl-card__kicker">15 minutes</span>
### [Installation](installation.md)
Install with `uv`, pick a training backend, and set the credentials it needs.
</div>

<div class="pl-card" markdown>
<span class="pl-card__kicker">30 minutes</span>
### [Quickstart](quickstart.md)
Two paths to a first result: evaluate an agent against any endpoint, then start a training run.
</div>

<div class="pl-card" markdown>
<span class="pl-card__kicker">Read once, refer often</span>
### [Core concepts](concepts.md)
`Task`, `Env`, `Agent`, `Trajectory`, rollouts, workflows, and the registry.
</div>

</div>

## What you need

| | |
| --- | --- |
| **Python** | 3.12 (`requires-python = "~=3.12.0"`) |
| **Package manager** | [`uv`](https://docs.astral.sh/uv/) — lockfiles and extras assume it |
| **Rollouts only** | Any OpenAI-compatible endpoint; no GPU |
| **Training with AReaL** | Linux and NVIDIA GPUs |
| **Training with Tinker** | An API key for a Tinker-compatible backend; no local GPU |

Tasks, agents, rewards and the inspection tools all run against a hosted endpoint, so you can build
and debug a plugin before you have a GPU — see [Evaluate a model](../guides/evaluate.md).

## The short version

```bash
# Install the core plus a backend
uv sync --extra areal          # or: uv sync --extra tinker

# Install a plugin — each one is its own uv project
cd plugins/number-search
uv sync --extra areal

# Train
uv run python platoon/number_search/train.py \
  --config platoon/number_search/nv_number_search_cispo_areal.yaml
```

Next: [Core concepts](concepts.md) explains what that run did, and
[Backends](../architecture/backends.md) covers choosing between AReaL and Tinker.
