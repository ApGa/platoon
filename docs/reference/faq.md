# FAQ

Short answers to the questions that come up in the first week, each pointing at the page that
goes deeper. If you have a specific error message rather than a question, start at
[troubleshooting](troubleshooting.md).

## Choosing a setup

### Do I need a GPU?

Not for everything. The Tinker backend trains against a remote service and needs no local
accelerator, and the inference workflows only call an OpenAI-compatible endpoint. AReaL is the GPU
path, and in practice it expects eight of them: every committed AReaL config in the repository
sets `cluster.n_gpus_per_node: 8`, and there is no single-GPU AReaL config anywhere in the tree.
See [backends](../get-started/backends.md).

### Which backend should I use?

Tinker if you have no cluster and want a task training today; AReaL if you own the GPUs and want
control over the loss, the actor, parallelism, and MoE features like router replay. Your
environment, agent, and rollout code are identical either way — the config schema and the CLI
override syntax are not. [Backends](../get-started/backends.md) compares them properly.

### Can I point Platoon at my own model endpoint?

For inference and evaluation, yes: set `inference.model_name`, `inference.model_endpoint` and
`inference.model_api_key`, and the rollout talks to whatever OpenAI-compatible service you name.
For training, no. `_build_rollout_config` in the AReaL workflow and `_get_rollout_config` in the
Tinker one overwrite `model_name`, `model_endpoint` and `model_api_key` on every rollout with the
trainer's own proxy, because RL needs the exact prompt and sampled tokens back. Setting them in a
training YAML is decorative. See [the inference tutorial](../tutorials/inference.md).

### Do I have to write a training script?

Not if your components are reachable by name or by import path. Fill in an `environments:` block
and run `python -m platoon.train.areal.train` or `python -m platoon.train.tinker.train`; both
entrypoints resolve the dataset loader, task loader, rollout, reward processor and workflow class
out of that block and know nothing about your task.

The catch is that this branch is mid-migration. `plugins/textcraft` is the only plugin that
registers components today, and its live `environments:` block is on the Tinker side — the AReaL
twin is commented out. Every other plugin still ships a bespoke `train_*.py`. Both paths work.
[The registry](../architecture/registry.md) explains the difference, and
[plugin anatomy](../walkthroughs/plugin-anatomy.md) walks a real plugin.

## Wiring components

### Must I register my components, or can I use import paths?

Import paths work everywhere. `Registry.resolve` in <span class="pl-src">platoon/registry.py</span>
looks the spec string up in the registry for that kind and, when it is not there, falls through to
`import_from_string`, which accepts `pkg.module.attr` and `pkg.module:attr`. So an `environments:`
block with zero `@register_*` calls is valid. Registering buys you a short stable name decoupled
from your module layout, plus a list of available names in the error message when you typo one.

One constraint on AReaL: the workflow is shipped to worker processes as import paths, so `rollout`
and `task_loader` must be importable module-level functions. A lambda, a closure or a
`functools.partial` makes `to_workflow_kwargs` raise. Details in
[component contracts](components.md).

### Why is only one environment allowed?

Because multi-environment training is not built yet. `AutoEnvironment.from_config` raises
`NotImplementedError("Multiple environments are not yet supported; provide exactly one entry")`,
and both trainer configs raise the same thing in `__post_init__`. The key is a list because the
schema is aspirational, not because you can put more than one entry in it. Mix tasks inside your
dataset loader instead — or use OpenReward's own mixture list, which is a different mechanism
entirely.

### What is the difference between the two `environments:` keys?

They are unrelated, and the collision is a real trap.

| | Top-level `environments:` | `openreward.environments:` |
| --- | --- | --- |
| Type | `list[EnvironmentConfig]` | `list[OpenRewardEnvironmentConfig]` |
| Defined in | <span class="pl-src">platoon/train/components.py</span> | <span class="pl-src">plugins/openreward/platoon/openreward/config_defs.py</span> |
| Fields | `package`, `dataset_loader`, `task_loader`, `rollout`, `reward_processor`, `workflow`, … | `label`, `env_name`, `session_url`, `sampling_weight`, … |
| Purpose | registry wiring: which components this run uses | task-source mixture with sampling weights |
| How many | exactly one | as many as you like |

Nesting is what tells them apart: the first sits at the top level of the trainer config, the
second under the `openreward:` key. See [the registry](../architecture/registry.md) and
[the OpenReward integration](../integrations/openreward.md).

### How do I pass arguments to my rollout function?

You do not, directly. Both workflows call `await self.rollout_fn(task, rollout_config)` with
exactly two positional arguments, and `EnvironmentConfig` has no `rollout_kwargs` field. Extra
parameters are reachable only through their defaults, so to vary them you register a second name
bound to a pre-parameterized function — which is what textcraft does with its `linear`,
`recursive` and `depth_aware` rollouts. Anything genuinely per-run belongs on `rollout_config`,
including its free-form `extra` dict. See [customizing the rollout](../customization/rollout.md).

## Agents, environments and prompts

### Is multi-turn supported?

Multi-turn is the only mode. `run_episode` in <span class="pl-src">platoon/episode/loop.py</span>
loops `while not halt_episode(obs)`, alternating `agent.act` and `env.step` until the environment
halts, the step budget runs out, or a timeout fires. A single-turn task is an environment that
halts after one step. See [agents and environments](../architecture/agents-envs.md).

### How do I change the prompt?

For CodeAct agents, subclass `CodeActPromptBuilder` in
<span class="pl-src">platoon/agents/codeact/prompt_builder.py</span> — `build_system_prompt`,
`build_user_prompt` and `build_next_action_str` are the hooks — and pass the instance as the
`prompt_builder` argument to `CodeActAgent`. If you only want to switch between the built-in
formats, pass `prompt_mode` instead. [Customizing the agent](../customization/agent.md) has the
full picture.

## Training

### What is `group_size` and how do I choose it?

It is the number of rollouts run for the same task, and their rewards form the baseline that
advantages are centered against. That makes it the main variance knob: too small and the baseline
is noise, too large and every training step costs proportionally more rollouts. The defaults
differ by backend — `1` on AReaL, `8` on Tinker — and essentially every real config uses `8`, with
a few at `4`. Evaluation runs at `group_size: 1`; the AReaL entrypoint forces that on the eval
workflow.

Pair a large group with `min_successful_group_size` when rollouts are flaky. The production
recursive configs use `group_size: 8` with `min_successful_group_size: 4`, so a group that loses
half its members is rejected rather than trained on a degenerate baseline. See
[the group rollout workflow](../walkthroughs/group-rollout-workflow.md).

### Why did all my data get filtered out?

Several independent filters sit between a rollout and a gradient, and most default to on:

| Filter | Backend | Default | Drops |
| --- | --- | --- | --- |
| `min_successful_group_size` | AReaL | `1` | the whole group when too few members return data or complete their root |
| `filter_zero_variance_groups` | AReaL | `true` | groups of more than one member whose retained rewards are all identical |
| `filter_errors` | both | `true` for train | error tokens that would otherwise receive positive credit |
| `subagent_datum_keep_probability` | both | `1.0` | non-root datums, by Bernoulli draw, when set below 1 |
| `filter_zero_advantage_datums` | both | `true` | datums whose centered reward is exactly zero |
| `train.max_staleness` | Tinker | `None` | rollouts older than `train_step - max_staleness` |

An all-zero or all-one reward signal produces empty batches even though nothing errored — that is
the zero-variance filter working as designed, and it is usually a task-difficulty problem rather
than a bug. Also note that `filter_zero_advantage_datums` is unsafe when the KL coefficient or
reward bias is nonzero, when reward or advantage normalization is on, or when an MoE auxiliary
loss is present; the AReaL trainer warns about the incompatibilities it can detect but never turns
the filter off for you. [Trajectory to batch](../walkthroughs/trajectory-to-batch.md) traces the
whole funnel.

!!! warning "`workflow_config.filter_errors` in YAML does nothing"

    Both shared entrypoints read `filter_errors` out of `environments[0].workflow_kwargs`, not out
    of `workflow_config`. The `WorkflowConfig` field of the same name exists but neither workflow
    reads it — only OpenReward's own train scripts forward it. To change it on the registry path,
    set `environments[0].workflow_kwargs: {filter_errors: false}`.

### What does a backend string like `fsdp:d4p1t1c2` mean?

It is AReaL's allocation grammar, parsed in its `areal/api/alloc_mode.py`: a backend name, a
colon, then one letter-plus-number per parallelism dimension. `d` is data, `p` pipeline, `t`
tensor, `c` context, `e` expert. So `fsdp:d4p1t1c2` is FSDP with data-parallel 4 and
context-parallel 2. Training backends are `fsdp`, `megatron` and `archon`; inference backends are
`sglang` and `vllm`, and they accept only `d`, `t` and `p`. MoE runs can split attention from FFN:
`megatron:(attn:d10p2t4c2|ffn:d10p2t1e8)`.

Platoon requires both `rollout.backend` and `actor.backend` explicitly — neither has a default,
and `PlatoonArealRLTrainerConfig.__post_init__` raises if either is empty.
[Parallelism](../recipes/parallelism.md) covers how to size them.

### How do I add a loss function?

On AReaL, decorate a function with `register_loss_fn` from
<span class="pl-src">platoon/train/areal/loss_functions.py</span>, then select it with
`loss_fn_config.loss_fn` and feed it through `loss_fn_config.loss_fn_kwargs`. The built-ins
registered there are `grpo`, `ppo` and `cispo`. Unlike the other registries this one registers
with `exist_ok=True`, so you can override a built-in name.

On Tinker you cannot. `train.loss_fn` is a string handed to the remote service in
`forward_backward_async` — the loss runs there, not in your process. See
[custom losses](../customization/loss.md).

### How do I resume a crashed run?

=== "AReaL"

    Resumption is AReaL's `recover:` block. The committed configs set `mode: auto` with
    `experiment_name`, `trial_name` and `fileroot` interpolated from the top level; re-run the
    same command with the same experiment and trial name and it continues from the last recovery
    checkpoint. Wall-time draining depends on this: the trainer raises "Cannot drain before the
    allocation deadline because recovery checkpointing is disabled" when `recover.mode` is
    `disabled` or `off`.

=== "Tinker"

    Checkpoint records are JSON lines in `<log_path>/checkpoints.jsonl`. On startup the trainer
    takes the last record carrying a `state_path` and resumes from its `batch`, restoring the
    optimizer state and the W&B run id along with the weights, so re-running with the same
    `log_path` is the whole procedure. For hangs, wrap the command in
    `python -m platoon.train.tinker.restart_wrapper -- <command>`: it restarts only on the
    watchdog exit code (default `2`, at most 5 times) and passes any other failure straight
    through.

[The training run walkthrough](../walkthroughs/training-run.md) has the step-by-step.

## Sub-agents and recursion

### How do sub-agents get credit?

A rollout produces a tree of trajectories, and every trainable trajectory in it becomes training
data with its own reward — a sub-agent's steps are not folded into the parent's tokens. On top of
that, two post-rollout adjustments in <span class="pl-src">platoon/utils/subagent_rewards.py</span>
reshape the tree's rewards, and they are mutually exclusive:

- `add_direct_subagent_delegation_rewards` gives a parent `coefficient * succeeded / launched` for
  its direct trainable children, using each child's score *before* its own bonus so bonuses do not
  compound up the tree.
- `propagate_root_success` overwrites every trajectory's reward with the root's success, which
  makes each child answerable for the outcome it contributed to and erases its individual score.

Verifier trajectories are excluded from training entirely and never count as a delegation. See
[the sub-agent model](../architecture/subagents.md) and
[recursive recipes](../recipes/recursive.md).

### How do I stop an agent delegating too much?

Three levers, in increasing bluntness. A depth cap: `openreward.subagent_max_depth` installs
`DepthAwareStepBudgetTracker` and also states the limit in the system prompt, so the model is told
rather than silently refused. A budget: the tree shares one step budget, and each child gets a
fixed `subagent_default_max_steps` (`50` by default in OpenReward) that the model does not choose.
And rewards: cap the delegation bonus the way textcraft's `textcraft/synth/delegation_capped`
reward processor does, or turn on outcome and behavior judging so a child only earns credit when a
verifier agrees it did the work.

A refused delegation is not an exception — `launch_subagent` returns a plain string starting
`"Not enough budget to launch subagent for goal ..."`, and the agent has to read it.
[Recursive agents](../tutorials/recursive-agents.md) is the tutorial.

## Running and debugging

### Where do rollout logs go?

=== "AReaL"

    `<workflow_config.rollout_config.output_dir>/<output_subdir>/<engine version>`, where
    `output_subdir` is `train_rollout` for the train workflow and `eval_rollout` for eval unless
    you override it in `workflow_kwargs`.

=== "Tinker"

    `<log_path>/rollouts/<stats_scope>/<checkpoint_version>`. The `output_dir` you set in YAML is
    overwritten whenever `log_path` is set, which it always is on the shared entrypoint.

By convention a rollout also attaches a `JsonlFileSink` and writes an event stream to
`<output_dir>/events/events_<task_id>_<collection_id>.jsonl`. Those are the files
`python -m platoon.visualization.cli` tails, replays and analyzes — see
[the visualization tutorial](../tutorials/visualization.md) and
[the event schema](schemas.md).

### How do I debug a hanging rollout?

Set `PLATOON_DEBUG_HANGS=1`. A watchdog thread then dumps the stack of any tracked async task
outstanding for longer than `PLATOON_DEBUG_HANG_THRESHOLD_SEC` (default 60), repeating every
`PLATOON_DEBUG_HANG_INTERVAL_SEC` (default 15). It is wired into the CodeAct agent's model-call
path, so it is most useful for rollouts stuck waiting on inference. If the problem is "slow"
rather than "stuck", `PLATOON_PROFILE_SPANS=1` writes span timings to `PLATOON_PROFILE_SPANS_PATH`.

Before reaching for either, check your timeouts. `rollout_config.step_timeout` defaults to 300 s
and `timeout` to `None`, and recursive runs need both raised a long way, because a parent's
`await launch_subagent(...)` does not return until the child episode *and* its verifier have
finished. More in [troubleshooting](troubleshooting.md).

### How do I run on Slurm?

Set `scheduler.type: slurm_prealloc` and allocate the nodes yourself. `PreallocatedSlurmScheduler`
in <span class="pl-src">platoon/train/areal/preallocated_slurm.py</span> launches AReaL's actor
and inference worker roles as `srun` steps inside the allocation you already hold, instead of
submitting jobs of its own. It allocates whole nodes only: a role whose total GPU count is not a
multiple of `cluster.n_gpus_per_node` raises.

The launcher scripts under `slurm-scripts/` are site-specific — NVIDIA cluster paths, an Enroot
container image, per-node environment servers — and the directory is gitignored with individual
files force-tracked, so a fresh clone does not get all of them. Read them as a worked example, not
a turnkey script. [Multi-node](../tutorials/multi-node.md) and [scale](../recipes/scale.md) cover
what you actually have to change.

## Installing

### Why are the `tinker` and `areal` extras mutually exclusive?

Because they resolve different torch builds from different indexes — the `areal` fork of the lock
pulls a `+cu129` wheel from the PyTorch index, the non-`areal` fork takes a plain PyPI build. The
root `pyproject.toml` declares them a uv `conflicts` group so a single lockfile can carry both
resolutions without having to satisfy both at once. Practically: one venv, one backend. Since each
plugin is its own uv project with its own `.venv`, you can still keep both around in different
directories.

### Do I need Transformer Engine?

Only for the Megatron actor backend. FSDP training and every inference workflow work straight
after `uv sync --extra areal`. Transformer Engine is deliberately excluded from the lock with a
`transformer-engine; sys_platform == 'never'` override, because its torch bindings are sdist-only
and locking them would force a CUDA compile on every `uv sync`, including for people who never
touch Megatron — Platoon's Megatron actor import is lazy for the same reason. Install it by hand,
somewhere a real CUDA toolkit exists, before selecting `actor.backend: megatron`. Megatron also
wants APEX, for the `gradient_accumulation_fusion` kernel.
[Installation](../get-started/installation.md) has the commands.

## See also

- [Troubleshooting](troubleshooting.md) — real error messages and what to change
- [Configuration reference](configuration.md) — every key, type and default
- [Core concepts](../get-started/concepts.md) — the vocabulary the rest of the site assumes
