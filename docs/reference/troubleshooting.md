# Troubleshooting

Symptom-first index of the failures Platoon actually produces. Every message quoted below was read
out of the source tree; match one against your log, then read the cause and the fix. For what a key
*means* rather than why it is rejected, see the [configuration reference](configuration.md).

---

## Install and dependencies

**`uv sync --extra tinker --extra areal` fails to resolve.**
The two backend extras are declared a uv `conflicts` group in the root
<span class="pl-src">pyproject.toml</span>. They resolve different torch builds from different
indexes, so exactly one of them can be installed into a given venv. Pick one:

```bash
uv sync --extra tinker      # remote training service, no local GPUs
uv sync --extra areal       # local/distributed GPU training
```

**`pip install -e ".[areal]"` cannot find `areal`.**
AReaL is not on PyPI. It comes from a git revision pinned under `[tool.uv.sources]` in
<span class="pl-src">pyproject.toml</span>, and only uv honors that. Use `uv sync`, not pip.

**A plugin venv resolves differently from the root, or your override "does not apply."**
Each plugin under `plugins/` is a standalone uv project with its own lockfile and its own `.venv`.
uv only honors `override-dependencies` declared by the *root project of the resolution*, which is
why every plugin re-declares the same override block. If you add an override to the root
`pyproject.toml`, add it to the plugin you actually build from as well.

**Python version.** `requires-python = "~=3.12.0"` everywhere. 3.11 and 3.13 will not resolve.

**`flash-attn` tries to build and fails.** FA2 is deliberately installed from sdist with
`no-build-isolation-package = ["flash-attn", "causal-conv1d", "mamba-ssm"]` so it builds against the
venv's torch. It needs a working toolchain, and the dependency is guarded by
`sys_platform == 'linux' and platform_machine == 'x86_64'` — on any other platform it is not
selected at all.

See [installation](../get-started/installation.md) for the full matrix.

---

## Config validation errors

These come from `__post_init__` on the config dataclasses, so they fire during config construction,
before any worker starts. Read the message literally.

### AReaL trainer config

Defined on `PlatoonArealRLTrainerConfig` and `WorkflowConfig` in
<span class="pl-src">platoon/train/areal/config_defs.py</span>.

| Message | Cause | Fix |
|---|---|---|
| `rollout.backend must be set explicitly` | Neither backend field has a default. | Set `rollout.backend`. |
| `actor.backend must be set explicitly` | Same. | Set `actor.backend` to `fsdp` or `megatron`. |
| `Multiple environments are not yet supported; provide exactly one entry` | `environments:` has more than one entry. | Keep exactly one `EnvironmentConfig`. |
| `workflow group_size must be positive` | `workflow_config.group_size < 1`. | Group RL needs a group. |
| `min_successful_group_size must be in [1, group_size]` | Quorum larger than the group. | Lower it, or raise `group_size`. |
| `straggler_quorum requires straggler_timeout_seconds` | Quorum set, no tail clock. | Set both, or neither. |
| `straggler_quorum must be in [1, group_size] or null` | Out of range. | — |
| `straggler_timeout_seconds must be positive or null` | Zero or negative. | Use `null` to disable. |
| `subagent_datum_keep_probability must be in [0, 1]` | It is a probability. | — |

`normalize_environment_configs` in <span class="pl-src">platoon/train/components.py</span> rejects a
mapping where a list is required:

```
`environments` must be a list; use `environments: - ...` for a single environment
```

Write it as a YAML sequence, even with one entry.

!!! warning "Two unrelated keys named `environments`"
    The top-level `environments:` is a list of `EnvironmentConfig` — registry wiring that selects
    the dataset loader, task loader, rollout function and workflow. OpenReward configs also carry a
    nested `openreward.environments:` list with `label` / `env_name` / `session_url` /
    `sampling_weight` fields; that one is an environment *mixture* and is unrelated. Validation
    messages that mention labels, sampling weights, or session URL pools are always about the
    OpenReward list.

### Router replay (R3)

`actor.enable_router_replay` is the single public gate, and it drags in a precondition list. Each
precondition is a separate `ValueError`:

- `actor.enable_router_replay requires the Megatron actor backend`
- `actor.enable_router_replay requires the SGLang rollout backend`
- `actor.enable_router_replay requires rollout.return_routed_experts=true`
- `actor.enable_router_replay requires actor.megatron.enable_mtp=false; rollout routes do not include MTP layers`
- `actor.enable_router_replay currently requires proximal log-probability recomputation to be disabled; forward-only replay is not implemented`
- `actor.enable_router_replay with gradient checkpointing requires actor.megatron.recompute_granularity=full and recompute_method=uniform`

Setting `workflow_config.enable_router_replay` (or the two `router_replay_*` dimensions) in YAML does
nothing — `__post_init__` copies them from the actor and overwrites whatever you wrote.

### Token-efficiency reward

`TokenEfficiencyRewardConfig` validates its own numbers. The one that surprises people:

```
enabled token_efficiency_reward requires at least one positive token weight
```

Both `input_token_weight` and `output_token_weight` are zero while `enabled: true`, so the penalty
would be identically zero.

### Deprecated spelling

```
Conflicting rollout propagation settings: propagate_root_success and deprecated propogate_root_success
```

`propogate_root_success` (missing an `a`) is a compatibility key on `RolloutConfig` in
<span class="pl-src">platoon/config_defs.py</span>. Setting both to different values raises; use the
canonical spelling.

### OpenReward plugin config

From `__post_init__` in
<span class="pl-src">plugins/openreward/platoon/openreward/config_defs.py</span>:

| Message | What to change |
|---|---|
| `At least one OpenReward environment must have sampling_start_step=0` | Something has to be sampleable at step 0. |
| `Staged OpenReward environment admission requires balance_accepted_batches=false` | Staged admission and strict balancing are mutually exclusive. |
| `enable_subagent_behavior_judging requires enable_subagent_reward_judging=true` | Behavior judging reads the outcome score. |
| `OpenReward environment labels must be unique` | Labels key the URL pools and the balancing quotas. |
| `OpenReward environment session URL pool env-var names must be unique` | Two environments resolve to the same `OPENREWARD_SESSION_URLS_<LABEL>`. |
| `Configure task_names or task_indices for an environment, not both` | Pick one selection mode. |

And from the trainer and rollout:

- `OpenReward balance_accepted_batches=true is incompatible with dynamic_bs=true`
- `OpenReward delegation rewards require propagate_root_success=false so direct child verifier scores remain intact`

---

## Overrides that silently do nothing

The two backends use two different loaders, and getting the syntax backwards produces *no error*.

=== "AReaL"

    `areal.api.cli_args.load_expr_config` (OmegaConf). Overrides are bare `key=value`, no dashes,
    and `${...}` interpolation works inside the YAML:

    ```bash
    python -m platoon.train.areal.train --config cfg.yaml \
      actor.path=/models/qwen cluster.n_nodes=16 workflow_config.group_size=8
    ```

    An unknown key is an error here.

=== "Tinker"

    `platoon.utils.config.load_config` (argparse). Overrides are `--dotted.key value` or
    `--dotted.key=value`, and there is no interpolation:

    ```bash
    python -m platoon.train.tinker.train --config cfg.yaml \
      --train.batch_size 64 --stats.wandb.mode offline
    ```

Three traps in the Tinker and inference loader, all in
<span class="pl-src">platoon/utils/config.py</span>:

1. **Bare `key=value` is dropped without a word.** `_parse_overrides` skips any token that does not
   start with `--`. A copy-pasted AReaL-style override just vanishes.
2. **`--key 1` becomes `True`, not `1`.** `_parse_value` tests the boolean words before `int()`, and
   `"1"`/`"yes"`/`"true"` are truthy while `"0"`/`"no"`/`"false"` are falsy. So
   `--inference.workflow.num_rollouts_per_task 1` sets it to `True`. Use `2`, or edit the YAML.
3. **Unknown keys are ignored.** `_dataclass_from_dict` iterates the dataclass fields, never the
   incoming dict, so a typo like `batchsize:` is a silent no-op.

A comma also turns a value into a list (`--stats.wandb.notes "a, b"` becomes `["a", "b"]`), and
`--key --otherkey` makes `key` a boolean `True`, swallowing the value you meant to pass.

If a key you set has no effect, check it against
[the config architecture page](../architecture/config.md) before assuming the feature is broken.

---

## Registry and component resolution

`Unknown dataset_loader: 'my_loader'. Available: [...]`
: Raised by `Registry.get` in <span class="pl-src">platoon/registry.py</span>. The registration
  module never ran in this process. Either set `environments[0].package` to a module whose import
  registers the component, or set `discover_entry_points: true` and ship a
  `[project.entry-points."platoon.plugins"]` entry. The `Available:` list tells you what *did*
  register — an empty list means nothing imported at all.

`Expected an import path like 'package.module.object', got 'my_loader'`
: `import_from_string` fell through because the string is neither a registry name nor a dotted path.

`'rollout' registry already has an entry named 'textcraft'`
: The registration module was imported twice under different names, or two plugins claim the name.
  Pass `exist_ok=True` only if you genuinely mean to replace it.

`Config must set environments[0].dataset_loader`
: `AutoDataset` needs a loader for the split. `eval_dataset_loader` falls back to `dataset_loader`;
  `task_loader` and `rollout` have no fallback.

`GroupRolloutWorkflow requires importable rollout_fn/get_task_fn`
: Rollout entry points are shipped to workers as `module` + `qualname` only. `infer_import_path`
  returns `None` for a lambda, a closure, or anything defined in `__main__`. Define them at module
  top level.

!!! note "Most plugins do not use the registry yet"
    Only textcraft declares a `platoon.plugins` entry point today. The other plugins ship their own
    `train_*.py` script and never touch `AutoEnvironment`. See
    [the registry page](../architecture/registry.md) for which path applies to you.

---

## Startup and scheduling

**`ERROR: could not locate executable srun, scontrol, and sbatch clients.`**
followed by `Set PLATOON_SLURM_BIN_DIR to the Slurm client bin directory.`
: `sbatch --export=NONE` leaves the batch shell with a minimal PATH, and on BCM clusters the Slurm
  binaries live outside it. Export `PLATOON_SLURM_BIN_DIR` before submitting.

**`ERROR: '<config>' has no top-level 'openreward:' section.`**
: The OpenReward launcher injects `openreward.*` Hydra overrides and greps the *raw file* for the
  section before Hydra composes anything. A derived config must keep a literal `openreward:` key —
  the same is true of literal `actor.backend` and `actor.path` lines, which the launcher also greps.

**`ERROR: no default config for N nodes; pass an explicit config path.`**
: The launcher only maps 2, 8, and 16 nodes to a bundled config. Pass the path yourself.

**`ERROR: OPENREWARD_ACTOR_PATH requires OPENREWARD_TRIAL_NAME.`**
: Overriding the actor path without a distinct trial name would collide with the previous run's
  experiment directory.

**Jobs start, but the actor and SGLang fight over the same GPUs while the rest of the allocation
idles.**
: This is exactly what `PreallocatedSlurmScheduler` exists to prevent — AReaL leaves `nodelist`
  unset, so every single-node `--overlap` step otherwise lands on node 0. Confirm
  `scheduler.type: slurm_prealloc`; the round-robin node pinning lives in
  <span class="pl-src">platoon/train/areal/preallocated_slurm.py</span>.

**`Preallocated Slurm only supports allocating entire nodes. Requesting N GPUs but each node has M.`**
: A role's GPU count is not a whole multiple of `n_gpus_per_node`. Adjust the parallelism topology,
  not the allocation.

**`PLATOON_AREAL_PREALLOC_CONFIGURE_CONCURRENCY must be a positive integer, got '...'`**
: Non-numeric or non-positive value. The default is `16`.

**`Platoon's updated AReaL integration requires single-controller mode`**
: The trainer only supports AReaL's single-controller scheduler. `scheduler.type` silently defaults
  to `local` when unset.

More on topology: [multi-node](../tutorials/multi-node.md) and [scale](../recipes/scale.md).

---

## Rollout timeouts and stragglers

Three timeouts nest, and they surface very differently.

| Level | Where it is set | What you see |
|---|---|---|
| Per step (`agent.act` / `env.step`) | `rollout_config.step_timeout`, default `300` | Trajectory finishes *normally* with `trajectory_timed_out` in `misc` and `Episode timed out at step N` in `error_message`. No exception reaches the caller. |
| Whole trajectory | `rollout_config.timeout`, default `null` | Cancellation: `trajectory_cancelled` in `misc`, `Episode cancelled at step N` in `error_message`. |
| Subprocess hard kill | derived: `(timeout or 900) + 120 + 60` | `[SubprocessWorker] Hard timeout exceeded — killing subprocess process group` on stderr, then `Subprocess hard timeout (Ns) for task T rollout R` in the workflow log. |

The step timeout is the one people miss. A run whose success rate looks plausible but low is often
full of step-timed-out trajectories that never raised anything — grep the trajectory `misc` for
`trajectory_timed_out`. The predicates are in
<span class="pl-src">platoon/utils/trajectory_status.py</span>.

**`Cutting off N tail rollout(s) for task T after Ss straggler grace`**
: Working as configured. Once `straggler_quorum` members have *settled* — completed, interrupted and
  failed-closed all count, since each has stopped making progress — the remainder get
  `straggler_timeout_seconds` and are then cancelled. `straggler_quorum` defaults to
  `group_size - 1`. Training eligibility is separate, governed by `min_successful_group_size`.

!!! warning "Straggler cutoff only runs in subprocess mode"
    `straggler_timeout_seconds` is read exclusively inside the subprocess execution path. With the
    default `workflow_config.use_subprocesses: false`, the asyncio path has no tail cutoff and every
    member waits out its full absolute rollout timeout.

**`N rollout wrapper task(s) remained pending after process-pool termination`**
: Third-party code below `run_in_executor` swallowed cancellation. The process group has already
  been SIGKILLed; this is telemetry, not a leak you have to chase.

**`Timed out closing AReaL proxy session S after 30s`**
: Session teardown exceeded its fixed grace. Usually a symptom of an env server that stopped
  responding — read the env-server logs before blaming the proxy.

---

## Hangs

Three independent hang detectors. Use the one that matches the scope of the problem.

### Trainer stall watchdog (AReaL, on by default)

Installed with the rest of the AReaL patches. It writes to stderr with a
`[platoon-stall-watchdog pid=…]` prefix:

- `Python threads could not run for Ns (stop-the-world GC pause or GIL-holding native call)` — a
  freeze longer than `PLATOON_STALL_DUMP_SECS` (default `180`) also dumps every thread stack through
  `faulthandler`.
- `engine RPC method 'X' has been running for Ns; all other engine RPCs (e.g. pause) are queued
  behind it` — after `PLATOON_ENGINE_STALL_SECS` (default `600`).
- `high file descriptor usage: N/M open; leaked sockets can wedge this process before any crash`.

It also registers `SIGUSR1`, so `kill -USR1 <pid>` gives an on-demand all-thread dump at any time.
Disable the whole thing with `PLATOON_STALL_WATCHDOG=0`.

### Async task hang debug (opt-in)

<span class="pl-src">platoon/utils/async_hang_debug.py</span> is a no-op unless `PLATOON_DEBUG_HANGS`
is `1`, `true`, `yes`, or `on`. Enabled, a background thread logs stuck agent requests:

```
hang_watchdog kind=... request_id=... age_s=... thread_ident=... task_name=... task_coro=... waiter=...
```

followed by a stack and a small source window around the current frame. Tunables:

| Variable | Default | Meaning |
|---|---|---|
| `PLATOON_DEBUG_HANG_THRESHOLD_SEC` | `60` | Age at which a tracked task counts as stale |
| `PLATOON_DEBUG_HANG_INTERVAL_SEC` | `15` | Watchdog poll period |
| `PLATOON_DEBUG_HANG_MAX_TASKS` | `3` | Stale tasks dumped per pass |
| `PLATOON_DEBUG_HANG_MAX_FRAMES` | `8` | Stack frames per dump |
| `PLATOON_DEBUG_HANG_SOURCE_CONTEXT` | `12` | Source lines rendered around the frame |

The stack printed is the stack of the *thread that registered the task*, not necessarily the
coroutine's own frames. The watchdog thread self-terminates once nothing is tracked and restarts
lazily on the next tracked request. Only the CodeAct agent registers tasks today, in
<span class="pl-src">platoon/agents/codeact/agent.py</span>.

### Tinker watchdog (on by default)

`WatchdogConfig` in <span class="pl-src">platoon/train/tinker/config_defs.py</span>:
`enabled: true`, `timeout_seconds: 600`, `exit_code: 2`. Ten minutes without a trainer heartbeat and
the process exits. A long but healthy sampling phase can trip it — raise `watchdog.timeout_seconds`
rather than disabling the guard.

---

## OOM and memory

**CUDA OOM in a worker, not in the trainer.** In single-controller mode the trainer process does no
GPU work; the actor/ref/critic and the rollout engines run in scheduler-launched workers. That is
why `_ensure_expandable_segments_env` in
<span class="pl-src">platoon/train/areal/config_defs.py</span> injects
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` into every worker launch spec. Setting that
variable in your own shell will not reach the workers.

**OOM inside the loss, at the vocabulary reduction.** AReaL's guarded loss computes diagnostics that
materialize full float32 logits. Platoon rewrites those two reductions to a chunked form at import
time, and refuses to launch rather than silently skipping when the upstream source has moved:

- `AReaL's vocab-reduction implementation drifted: expected exactly one unsafe mean and one unsafe norm, found mean=..., norm=...`
- `Cannot inspect AReaL's guarded vocab reductions; refusing to launch without verifying the logits-memory safeguard.`
- `AReaL's guarded loss method has an unexpected source layout; refusing an unverified vocab-reduction patch.`

All three mean the pinned AReaL revision changed. Do not work around them by disabling the patch;
the run will OOM at the first large-vocabulary backward pass.

**Levers worth pulling before shrinking the batch:** `actor.megatron.recompute_granularity` and
`recompute_method`, `rollout.ensure_batch_divisible_by`, and `workflow_config.group_size`. See
[parallelism](../recipes/parallelism.md).

**`No space left on device` from an env-server container.** Disk, not GPU memory. In the Slurm path
the env-server step redirects the uv cache, `TMPDIR` and the XDG directories onto the host `/tmp`
because the container overlay fills under sustained concurrency. If you write your own env-server
step, do the same.

---

## Sub-agents and budgets

`BudgetExceededError` carries a `reason` and a `guidance` string; the guidance is what the model
sees, so it reads as advice rather than as a stack trace.

| Message | `reason` | Cause |
|---|---|---|
| `Requested step budget N exceeds remaining budget M.` | `step_budget` | `StepBudgetTracker`: a launch reserves `max_steps + 1` steps, and the parent tree does not have that much left. |
| `Launching a subagent from depth D would exceed the maximum allowed depth of N.` | `depth` | `DepthAwareStepBudgetTracker.max_depth` reached. |
| `Verifier helper agents may not launch further subagents.` | `verifier_depth` | The synthetic verifier tree allows exactly one helper level. |

`WARNING: Exhausted budget when running episode. Halting episode; task may be incomplete.`
: Set as the trajectory's `error_message` by `halt_episode` in
  <span class="pl-src">platoon/episode/loop.py</span> when remaining budget reaches zero. The
  subagent action rewrites it for the parent as `Subagent did not finish before its step budget was
  exhausted.` This is a normal termination, not a crash.

`StepBudgetTracker` charges subagent steps against the parent's budget;
`DepthAwareStepBudgetTracker` gives every trajectory its own `task.max_steps` and bounds the tree by
depth instead. Choosing the wrong one is the usual reason a recursive run either starves immediately
or never terminates — see [subagents](../architecture/subagents.md).

---

## "All my batches got filtered out"

Several independent knobs drop data and they compose, so work down this list in order; each stage
logs when it fires. Stages 1, 2 and 6 are AReaL-only — the Tinker workflow has no group-acceptance
gates and no divisibility trim. Stages 3, 4 and 5 apply on both backends, and stage 7 gives the
Tinker wording.

### 1. Group rejected for too few members

```
Rejecting task T group with N returned members; minimum is M
Rejecting task T group with N completed roots; minimum is M
Rejecting task T group with no valid root rewards
```

`min_successful_group_size` is checked twice: once against members that returned anything, and again
against members whose *root reward is valid*. A group of interrupted or timed-out roots is rejected
even when every member returned data.

### 2. Zero-variance group dropped

`filter_zero_variance_groups` defaults to `true`. When every retained reward in a group is identical
there is no within-group signal, so the whole group goes:

```
All retained rewards identical for task T (unprocessed mean=X.XX)
```

The `zero_variance_reward_group` stat counts these. If your task is nearly always solved or nearly
never solved, this is where the batch disappears. Setting it `false` keeps the group (some
production configs do); the better fix is usually the difficulty distribution — see
[curriculum](../recipes/curriculum.md).

### 3. Exact-zero-advantage datums removed

`filter_zero_advantage_datums` also defaults to `true`. After group centering, datums whose scalar
reward is exactly zero are dropped before model-side compute, keeping only the minimum structural
padding.

!!! danger "This filter is unsound in several common modes"
    It uses centered scalar reward as an early proxy for final policy advantage. That proxy breaks
    when KL is nonzero, `reward_bias` is nonzero, reward or advantage normalization is active,
    `overlong_reward_penalty` is enabled, a critic or teacher objective is present, the model has an
    independent MoE/router auxiliary loss, or a custom batch transform adds to rewards. The trainer
    emits a `RuntimeWarning` at construction listing the incompatibilities it detected — any custom
    batch transform triggers it — and then proceeds anyway. Read it; it will not disable the feature
    for you.

### 4. Sub-agent datums sampled away

`subagent_datum_keep_probability < 1.0` retains every root datum and independently samples each
post-merge subagent datum. A value of `1.0` reproduces the historical training batch exactly.
Evaluation forces it back to `1.0`.

### 5. Error tokens suppressed

`filter_errors` defers typed action errors until group centering, then blanks only the error tokens
that would otherwise receive positive credit. On the registry entry point this value comes from
`environments[0].workflow_kwargs`, **not** from `workflow_config.filter_errors` —
`run_areal_training` pops it from `workflow_kwargs` with default `True` for train and `False` for
eval. Explicit plugin train scripts differ from each other here; check yours.

### 6. Divisibility trim, and the whole-step drop

After every filter the batch is trimmed to a multiple of
`lcm(rollout.ensure_batch_divisible_by, dp_size)`. Trimming is skipped when the batch is smaller than
one full multiple, and if the total is below `dp_size` the entire step is dropped. You then see:

```
Skipping optimizer update because advantage computation returned no batch
```

The trimmer draws a random subset and prefers non-root datums, falling back to roots only when there
are not enough non-root candidates.

### 7. Tinker-side equivalents

The Tinker workflow logs the same funnel in different words: `No results found for task T`,
`No completed root rewards available for task T; retaining rollout stats but skipping training`,
`No train data found for task T`.

Related batch-transform errors mean a transform got a batch it cannot interpret, not an empty one:

- `depth_level_weighting requires traj_depth and traj_start in tinker datums` (Tinker)
- `depth_level_weighting produced zero total weight for this microbatch` (Tinker)
- `workflow_config.depth_level_discount_gamma produced zero total weight for this batch` (AReaL —
  `depth_level_discount_gamma` does not exist on the Tinker `WorkflowConfig`)
- `Unable to infer batch size from batch contents` (AReaL)

The full path from trajectory to optimizer batch is in
[trajectory to batch](../walkthroughs/trajectory-to-batch.md); the filtering stages themselves are
walked in [the group rollout workflow](../walkthroughs/group-rollout-workflow.md).

---

## API keys and endpoints

**`LLM API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.`**
and **`LLM base URL is required. Set OPENAI_BASE_URL environment variable or pass base_url parameter.`**
: `LLMClient` in <span class="pl-src">platoon/utils/llm_client.py</span> requires both, even against
  a keyless local server — export a dummy `OPENAI_API_KEY`. Every shipped inference config leaves
  `model_api_key: null`, so this bites benchmarks against local endpoints.

**`LiteLLM API call failed: <sanitized>`**
: `LiteLLMClient` re-raises as `RuntimeError` with the request payload stripped out. The original
  exception is chained, so the traceback still carries the real cause.

!!! warning "There is no retry anywhere"
    `LiteLLMClient` sets `num_retries=0` and `LLMClient` has none at all. A single 429 or transport
    blip fails the whole rollout. Front the endpoint with a retrying LiteLLM proxy, or pass retry
    kwargs from your own plugin's client.

**Rollouts hammer the endpoint until it rate-limits.** `PLATOON_LITELLM_MAX_INFLIGHT` is off by
default; without it nothing caps concurrent litellm requests beyond your worker count, and recursive
agents multiply that by the number of live subagents. Set it to a positive integer — a non-integer
or non-positive value is silently ignored.

**`OpenReward get_task returned no JSON prompt payload`** / **`OpenReward MCP bridge did not expose get_task`**
: The env server answered, but not with a task. Check that the server is the environment you think
  it is: mixed runs route by label, and a mismatched `OPENREWARD_SESSION_URLS_<LABEL>` pool sends
  requests to the wrong backend.

**`Unable to enumerate tasks for OpenReward environment 'X' split='Y': list_tasks failed with ...; num_tasks/get_task fallback failed with ...`**
: Both catalog paths failed. `task_names` forces the legacy `list_tasks` API, which large
  environments reject — use `task_indices`, or drop the selection.

**`No tasks selected for OpenReward environment 'X' split='Y'`**
: The selection filtered everything out. `train_task_limit` and `eval_task_limit` must be
  *positive*; `0` is a validation error, not "unlimited" — use `null`.

**MCP tool listing times out on a cold environment.** `OPENREWARD_MCP_TIMEOUT` defaults to 120
seconds. Cold TMax and SWE-rebench tasks import multi-gigabyte container images; the
mixed-environment launchers raise it to 1800. See [OpenReward](../integrations/openreward.md).

---

## Weights & Biases

**W&B is silently off.** In the Slurm path, if neither `OPENREWARD_WANDB_MODE` nor `WANDB_API_KEY`
is set, the launcher disables W&B with a warning and continues:

```
WARNING: WANDB_API_KEY is not set; disabling W&B logging.
         Export WANDB_API_KEY before sbatch to retain online logging.
```

That is deliberate: AReaL calls `wandb.login()` during trainer construction, *after* costly worker
startup, so a missing key would otherwise waste the allocation. To make it fatal instead, set
`OPENREWARD_WANDB_MODE=online`, which produces
`ERROR: OPENREWARD_WANDB_MODE=online requires WANDB_API_KEY.` and exits before any work. Any value
other than `online`, `offline`, or `disabled` is rejected outright.

**`Failed to initialize WandB: <error>`**
: Logged as a *warning* by `StatsLogger` in <span class="pl-src">platoon/utils/stats_logger.py</span>
  — the run then continues with no telemetry at all. If your dashboard is empty and the job is
  clearly alive, grep the head of the log for this line.

**Steps missing from a chart.** `StatsLogger.log` skips when `step - last_logged < log_interval`, so
repeated or out-of-order steps are dropped. Pass `force=True` for a step you must see.

---

## Megatron and Transformer Engine

**`ModuleNotFoundError: transformer_engine` the moment you set `actor.backend: megatron`.**
TE is deliberately excluded from the lock via `transformer-engine; sys_platform == 'never'` in
<span class="pl-src">pyproject.toml</span>. Its torch bindings are sdist-only and do not build
without a CUDA toolkit, so forcing them into the resolution graph would break `uv sync` for *every*
backend, FSDP included. Platoon's Megatron actor import is lazy precisely so FSDP runs never touch
it.

Install TE by hand, where a real `nvcc` exists — inside the training container, or after
`module load cuda`:

```bash
uv pip install ninja
# --no-config bypasses the `sys_platform == 'never'` override; --no-build-isolation
# builds against the venv's torch instead of pulling a mismatched one.
CUDA_HOME=/usr/local/cuda uv pip install --no-config --no-build-isolation \
  "transformer-engine[pytorch]==2.12.0"
uv pip install --no-config "transformer-engine==2.12.0"
```

**`Could not find transformer-engine`** during that install
: The empty `transformer-engine` meta-package is dropped by the override, so TE's own sanity check
  fails. That is what the third command is for. Verify with
  `python -c "import transformer_engine.pytorch"`.

**`crt/host_defines.h not found`**
: No CUDA toolkit on the machine doing the build. Build inside the container; the resulting wheel is
  reusable on bare nodes.

**Missing `fused_weight_gradient_mlp_cuda`.**
Megatron's `ColumnParallelLinear` defaults to `gradient_accumulation_fusion=True`, which hard-requires
APEX's CUDA kernel (AReaL only disables that fusion for LoRA). APEX, like TE, is not in the lock and
must be source-built with `--cpp_ext --cuda_ext` where `nvcc` exists.

**Your Slurm job did not build TE or APEX.** The launcher auto-enables both by grepping the config
for a Megatron backend line:

```bash
if grep -qiE "^[[:space:]]*backend:[[:space:]]*['\"]?megatron" "${CONFIG}" 2>/dev/null; then
  OPENREWARD_BUILD_TE=${OPENREWARD_BUILD_TE:-1}
  OPENREWARD_BUILD_APEX=${OPENREWARD_BUILD_APEX:-1}
```

A derived config that hides the backend behind interpolation, or spells it differently, defeats the
grep. Keep `backend:` literal, or set `OPENREWARD_BUILD_TE=1` / `OPENREWARD_BUILD_APEX=1` explicitly.

**`FP32 LM-head output was requested, but a post-process Megatron model chunk has no output_layer to adapt.`**
: `megatron.enable_fp32_lm_head` with a model whose post-process chunk exposes no `output_layer`. It
  is a language-model-head option and is deliberately ignored for critics.

**`Skipping optimizer update because gradient norm is non-finite: <value>`**
: The non-finite gradient guard fired. Megatron's BF16 optimizer has no gradient scaler, so clipping
  an infinite norm by a zero coefficient can produce NaN and poison the weights permanently — the
  guard refuses the step instead. The trainer then logs `Skipped optimizer, scheduler, and weight
  broadcast for global step N after a non-finite/unsuccessful actor update; checkpointing the
  unchanged finite state.` One of these is a hiccup; a run of them means the learning rate or the
  advantage scale is wrong.

**`Actor workers disagreed on per-minibatch optimizer update success: [...]`**
: A hard failure by design. All DP workers participate in the same gradient-norm collectives and
  must report the same pattern; a worker that reported success may already have mutated its local
  weights, so reducing the disagreement to `False` would be unsafe. Recover from the previous
  checkpoint.

**`Unsupported Platoon actor backend: 'X' (expected 'fsdp' or 'megatron')`**
: Only those two. Topology suffixes such as `megatron:(attn:...|ffn:...)` are parsed off before this
  check, so a bad value here is a genuine typo.

Background on the integration and its patches: [the AReaL backend](../architecture/areal.md).

---

## Filing a useful bug report

Most items are one command each; without them a report is unactionable.

- **Backend and mode.** Tinker, AReaL single-node, or AReaL preallocated Slurm — plus node and GPU
  count.
- **The exact command line**, including every override, verbatim. The override syntax is itself a
  common bug.
- **The resolved config**, not the YAML you edited. AReaL writes a `config.yaml` into the experiment
  directory; attach that. On the Tinker path, attach the YAML plus the override list.
- **The first error, not the last.** Worker failures cascade, and the interesting traceback is
  usually hundreds of lines above the one that killed the job.
- **Which process produced it** — trainer/controller, actor worker, rollout worker, or env server.
  On Slurm these are separate log files.
- **Version pins.** The Platoon commit, whether the worktree was dirty, the plugin directory you
  installed from, and
  `uv pip list | grep -E 'areal|torch|sglang|transformers|megatron|tinker'`.
- **For a hang:** a stack dump. `kill -USR1 <pid>` on the trainer, plus a rerun with
  `PLATOON_DEBUG_HANGS=1` if the hang is inside a rollout.
- **For an empty-batch problem:** the workflow's rejection lines for several consecutive steps, and
  the values of `min_successful_group_size`, `filter_zero_variance_groups`,
  `filter_zero_advantage_datums`, `subagent_datum_keep_probability` and
  `rollout.ensure_batch_divisible_by`. Which stage dropped the data is the entire question.
- **For a numerical problem:** FSDP or Megatron, LoRA on or off, and the `grad_norm` series around
  the failure.

## See also

- [Configuration reference](configuration.md) — every key, its type and its default
- [FAQ](faq.md) — shorter questions that are not failures
- [Plugins](plugins.md) — which plugin uses which entry point
- [Config architecture](../architecture/config.md) — the two loaders and why they differ
