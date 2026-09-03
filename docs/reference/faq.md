# FAQ

Short answers to the questions that come up first, each pointing at the page that goes deeper.

## Setup

### Do I need a GPU?

Not for everything. The Tinker path trains against a remote Tinker-compatible service and needs no
local accelerator, and evaluation only calls an OpenAI-compatible endpoint. AReaL is the GPU path,
and the configs in the repository assume a full node — `cluster.n_gpus_per_node: 8`. See
[backends](../architecture/backends.md).

### Which backend should I use?

Tinker if you have no cluster and want to start now; AReaL if you own the GPUs and want control
over the loss, the actor and the parallelism. Your environment, agent and rollout code are identical
either way — the config schema and the CLI override syntax are not.
[Backends](../architecture/backends.md) compares them properly.

### Can I point Platoon at my own model endpoint?

For inference and evaluation, yes: set `inference.model_name`, `inference.model_endpoint` and
`inference.model_api_key` and the rollout talks to whatever OpenAI-compatible service you name. For
training, no — RL needs the exact prompt and the sampled tokens back, so both training workflows
point every rollout at the trainer's own proxy. See [evaluate](../guides/evaluate.md).

### Why can I only install one backend extra per environment?

`tinker` and `areal` resolve different torch builds from different indexes, so they are declared a
uv `conflicts` group. One virtual environment, one backend. Since each plugin is its own uv project
with its own `.venv`, you can keep both around in different directories.
[Installation](../get-started/installation.md) has the commands.

## Plugins and wiring

### Does my plugin have to live in this repository?

No. A plugin is an ordinary Python package. Give it a `platoon.plugins` entry point, install it
alongside Platoon, and it is discoverable — no fork, no upstreaming, and your research project stays
in your own repo. The plugins shipped here are examples of the same mechanism, not a privileged
location. See [plugins](../plugins/index.md).

### What is the difference between a task plugin and a capability plugin?

A **task plugin** packages a task or environment plus the rollout program that runs it. A
**capability plugin** adds framework functionality — an agent harness, an environment-server
integration, a new reward or loss. One package can be both. [Plugins](../plugins/index.md).

### Do I have to register my components?

No. `Registry.resolve` looks a spec string up in the registry for that kind and, when it is not
there, falls back to importing it, accepting `pkg.module.attr` and `pkg.module:attr`. An
`environments:` block with zero `@register_*` calls is valid. Registering buys a short stable name
decoupled from your module layout, and a list of the available names when you mistype one.

One constraint on AReaL: the workflow reaches worker processes as import paths, so `rollout` and
`task_loader` must be module-level functions — not lambdas, closures or partials. See
[components](../architecture/components.md).

### Why does `environments:` take only one entry?

The top-level `environments:` block is registry wiring — which dataset loader, task loader, rollout,
reward processor and workflow this run uses — and a run uses one set of those. Mix tasks inside your
dataset loader instead. OpenReward's nested `openreward.environments:` list is a different thing
entirely: a task mixture with sampling weights, and it takes as many entries as you like.

### How do I change the prompt?

For CodeAct agents, subclass `CodeActPromptBuilder` in
<span class="pl-src">platoon/agents/codeact/prompt_builder.py</span> and pass the instance as the
`prompt_builder` argument to `CodeActAgent`; `build_system_prompt`, `build_user_prompt` and
`build_next_action_str` are the hooks. To switch between the built-in formats without subclassing,
pass `prompt_mode`. [Extend](../guides/extend.md) shows a builder in a real plugin.

### How do I pass arguments to my rollout function?

Not directly. A workflow calls `rollout(task, rollout_config)` with exactly those two arguments.
Anything that varies per run belongs on `rollout_config`, including its free-form `extra` dict;
anything that varies per variant is a second registered name bound to a pre-parameterized function.
[Extend](../guides/extend.md) has the pattern.

## Multi-agent runs

### How do sub-agents get credit?

A rollout produces a tree of trajectories, and every trainable trajectory in it becomes training
data with its own reward — a delegate's steps are not folded into its parent's tokens. Two optional
adjustments reshape the tree afterwards, and they are mutually exclusive: a delegation bonus pays a
parent in proportion to how many of its direct children succeeded, while root-success propagation
overwrites every reward in the tree with the root's outcome. See
[multi-agent workflows](../guides/multi-agent.md).

### How do I stop an agent delegating too much?

Three levers, in increasing bluntness. A depth cap, which is also stated in the system prompt so the
model is told rather than silently refused. A step budget shared across the tree, so a delegation
spends the parent's remaining steps. And rewards — cap the delegation bonus, or require a verifier
to agree the child did the work. A refused delegation is a plain string the agent reads, not an
exception. [Multi-agent workflows](../guides/multi-agent.md).

## Training

### What is `group_size`?

The number of rollouts run for the same task. Their rewards form the baseline that advantages are
centered against, which makes it the main variance knob: too small and the baseline is noise, too
large and every step costs proportionally more rollouts. `8` is a common training value; evaluation
runs at `1`.

When rollouts are flaky, pair it with `min_successful_group_size` so a group that loses too many
members is rejected rather than trained on a degenerate baseline — `8` and `4` is a common pairing.
[Configuration](configuration.md) lists both.

### Why is my training data being filtered out?

Several independent filters sit between a rollout and a gradient, and most default to on:
`min_successful_group_size` rejects an under-populated group, `filter_zero_variance_groups` drops a
group whose rewards are all identical, `filter_zero_advantage_datums` drops datums whose centered
reward is exactly zero, `filter_errors` blanks error tokens, and `subagent_datum_keep_probability`
samples non-root datums away when set below `1.0`.

All of these live under `workflow_config` except `filter_errors`, which is set per split through
`environments[].workflow_kwargs` and defaults to on for training and off for evaluation.

An all-solved or never-solved task produces empty batches with nothing erroring — that is the
zero-variance filter working, and it points at task difficulty rather than at the config. Turn the
filters off one at a time to find which stage is eating the batch.
[Execution](../architecture/execution.md) traces the whole funnel.

### How do I resume a crashed run?

=== "AReaL"

    Resumption is AReaL's `recover:` block, set to `mode: auto` in the shipped configs. Re-run
    the same command with the same `experiment_name` and `trial_name` and it continues from the
    last recovery checkpoint.

=== "Tinker"

    Checkpoint records are JSON lines in `<log_path>/checkpoints.jsonl`. On startup the trainer
    takes the last record carrying a `state_path` and resumes from its batch, restoring the
    optimizer state and the W&B run id along with the weights. Re-running with the same `log_path`
    is the whole procedure.

[Scale](../guides/scale.md) covers what a resume does and does not restore.

### How do I run on Slurm?

Allocate the nodes yourself and set `scheduler.type: slurm_prealloc`. `PreallocatedSlurmScheduler`
launches the actor and inference roles as `srun` steps inside the allocation you already hold rather
than submitting jobs of its own, and pins them to distinct nodes. It allocates whole nodes, so each
role's GPU count must be a multiple of `cluster.n_gpus_per_node`. The scripts under `slurm-scripts/`
are a worked example for one site; [scale](../guides/scale.md) says what to change.

## Common problems

**A CLI override has no effect.** The two backends use two different loaders. AReaL takes bare
`key=value` and rejects unknown keys; Tinker and inference take `--dotted.key value` and ignore
anything else, including a copy-pasted AReaL-style override. [CLI](cli.md) has both forms.

**`Unknown dataset_loader: '...'. Available: [...]`** — the module that registers your component
never ran in this process. Point `environments[0].package` at a module whose import registers it, or
set `discover_entry_points: true` and ship a `platoon.plugins` entry point. An empty `Available:`
list means nothing registered at all. [Components](../architecture/components.md).

**`LLM API key is required` against a local server.** The client requires both a key and a base URL
even when the server ignores them. Export `OPENAI_API_KEY` (any value) and `OPENAI_BASE_URL`, or set
`inference.model_api_key` and `inference.model_endpoint` in the config.

**`Skipping optimizer update because advantage computation returned no batch`.** Every rollout in
the step was filtered out; work through the filters above. The workflow logs a line for each group
it rejects, and that line names the stage.

**Rollouts finish but look truncated.** `rollout_config.step_timeout` defaults to 300 seconds and
bounds a single `agent.act` or `env.step`; a trajectory that hits it ends normally, with
`trajectory_timed_out` in its `misc`. Multi-agent runs need it and the whole-rollout `timeout`
raised a long way, because a parent's delegation call does not return until the child episode
finishes.

**A rollout hangs.** Set `PLATOON_DEBUG_HANGS=1`. A watchdog thread then dumps the stack of any
tracked async task outstanding for longer than `PLATOON_DEBUG_HANG_THRESHOLD_SEC` (default 60). It
is wired into the CodeAct agent's model-call path, so it is most useful for rollouts stuck waiting
on inference.

## See also

- [Configuration](configuration.md) — every key Platoon owns, with its default
- [CLI](cli.md) — entrypoints and override syntax
- [Core concepts](../get-started/concepts.md) — the vocabulary the rest of the site assumes
