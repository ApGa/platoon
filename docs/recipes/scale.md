# Long-running and preallocated jobs

A run that holds 256 GPUs for three days fails differently from one that holds 8 for an hour. It
will be killed at a wall-time boundary, it will meet a node whose environment server dies, and it
will have a rollout that never returns. This page is about the settings that decide whether it
survives those things — which to turn on, what to set them to, and what each one costs.

[Scale to multiple nodes](../tutorials/multi-node.md) teaches submission, and
[AReaL backend internals](../architecture/areal.md) explains how the machinery works. Neither tells
you what to choose. Every key here is listed with its default in
[the configuration reference](../reference/configuration.md).

<span class="pl-tag pl-tag--areal">AReaL</span> Everything on this page is AReaL-side. Tinker trains
on a remote service and has no allocation to keep alive.

## Pick a job model first

`scheduler.type` decides which world you are in, and almost everything else on this page follows
from it.

| `scheduler.type` | What runs the workers | Reach for it when |
|---|---|---|
| unset | AReaL's own scheduler — `_init_scheduler` falls through to upstream | Single node, or a cluster where one job per worker role is acceptable. |
| `slurm_prealloc` | `PreallocatedSlurmScheduler` launches each role as an `srun` step inside an allocation you already hold | You want one Slurm job that either starts whole or not at all, and a batch shell you control. |

Every committed multi-node config in the repo uses `slurm_prealloc`; every single-node config leaves
`scheduler` out entirely. That is the honest state of the tree — there is no committed multi-node
config that uses upstream's scheduler.

The reason to care is not scheduling efficiency. It is that the preallocated model gives you a
**batch shell that outlives the trainer**. Signal traps, successor submission, health monitors, the
GPU keepalive and the drain-marker check all live in that shell. With one job per worker role there
is nowhere to put them. Everything below assumes you have one.

The cost is that you write and maintain the launcher.
<span class="pl-src">slurm-scripts/openreward-toolathlon-prealloc-base.sh</span> is over 1,200 lines,
and a good half of it is site-specific — Pyxis, Enroot, Lustre, a content-addressed venv builder. Do
not adopt it wholesale; the tutorial's minimal launcher is the right starting point.

!!! warning "Whole nodes only"
    `PreallocatedSlurmScheduler` raises when a role's total GPU count is not a multiple of
    `cluster.n_gpus_per_node`. Plan the split between actor and rollout in whole nodes before you
    submit, not after 20 minutes of worker startup.

## Deadlines: drain at a step boundary

Being killed mid-update costs you every rollout in that step plus everything since the last recovery
checkpoint. `StepDeadlineGuard` avoids that by refusing to *start* a step it does not think will
finish.

Turn it on with two exports. There is no config key:

```bash
export PLATOON_TRAINING_DEADLINE_EPOCH=$(
  date --date="$(scontrol show job "${SLURM_JOB_ID}" -o | sed 's/.*EndTime=//;s/ .*//')" +%s
)
export PLATOON_TRAINING_DRAIN_FILE=${JOB_STATE_DIR}/deadline-drain-requested.json
```

Setting the epoch without the drain file raises. Leaving the epoch unset disables the guard silently
— `from_environment` returns `None` and `train()` never checks a deadline.

**Enable it whenever the run is longer than the time limit.** There is no reason not to: the guard
costs one comparison per step, and a run that finishes inside its allocation never trips it.

### Choosing the two numbers you will actually tune

`PLATOON_DEADLINE_INITIAL_STEP_SECONDS` (default 1800) is a **permanent floor** on the step-time
estimate, not a starting guess that history replaces. The estimate is
`max(floor, max(recent_durations) * multiplier)`, so a floor above your true step time means the
guard drains earlier than it needs to, forever. Set it to a plausible cold-start step time and let
history take over. Every committed wrapper keeps 1800, calling it "the established 30-minute
bootstrap floor".

`PLATOON_DEADLINE_SAFETY_SECONDS` (default 300) is fixed shutdown headroom added on top. The
committed wrappers disagree, and both positions are defensible:

```bash title="slurm-scripts/openreward-toolathlon-prealloc-32node-ptc-recursive-bs8-behavior-gated.sh"
# The cold first update is excluded from adaptive history. Keep the established
# 30-minute bootstrap floor, but rely on the adaptive 15% step-time headroom
# instead of stacking a second fixed wall-time buffer for this trial.
export OPENREWARD_DEADLINE_INITIAL_STEP_SECONDS=${OPENREWARD_DEADLINE_INITIAL_STEP_SECONDS:-1800}
export OPENREWARD_DEADLINE_SAFETY_SECONDS=0
```

The mixed-environment wrappers use 600 instead, because those runs tear down host-native TMax and
SWE-rebench server pools as well as the trainer. Pick by how long your shutdown actually takes.

The first completed step of each allocation is deliberately excluded from the timing history — it
starts from an empty async rollout buffer. That is why the floor matters: for one step per
allocation, the floor *is* the whole estimate.

### What a drain costs, and when it hurts

A drain wastes the remainder of the allocation. If your step estimate is 30 minutes and the guard
drains with 35 minutes left, you have paid 35 idle GPU-minutes to avoid losing a 30-minute step.
That is the right trade at any batch size worth training, but it means a run whose steps are long
relative to the allocation spends a visible fraction of its time draining. Longer allocations
amortize the guard: four hours with 30-minute steps puts a ceiling of roughly 12% on the waste.

!!! warning "A drain requires recovery checkpointing"
    `_ensure_recover_checkpoint_at` raises when `recover.mode` is `disabled` or `off`. The guard
    cannot advertise a clean exit it has not made durable, so the drain path fails loudly rather
    than exiting quietly with stale state.

## Straggler policy: two keys, and they only work together

Agentic rollouts have heavy tails. One group member stuck pulling a container holds all seven of its
peers until the absolute rollout timeout. Two keys cut the tail:

```yaml title="workflow_config, from toolathlon_openhands_areal_prealloc_32node-...-bs8.yaml"
  straggler_timeout_seconds: 900
  straggler_quorum: 6
  subprocess_shutdown_grace_seconds: 10
  min_successful_group_size: 4
```

The config's own comment states the intent better than a paraphrase would:

> Once six members return usable trajectories, give the remaining tail 15 more minutes, then
> explicitly reap the group's dedicated subprocess pool. A retrospective trial-2 simulation retained
> 7.1/8 members on average. Reject groups with fewer than four usable members rather than
> constructing a degenerate baseline.

**`straggler_quorum` is meaningless without `straggler_timeout_seconds`** — literally, it is a
`ValueError` in `WorkflowConfig.__post_init__`. The quorum only names *when the clock starts*; the
timeout is the clock. A quorum alone would mean "notice that six members finished, then do nothing",
which is what already happens by default. Set the timeout, and leave the quorum at its default
(`group_size - 1`) unless you have a reason to prefer another value.

Choosing values:

- **Timeout.** How long is a plausible slow-but-real completion once the group's median is done?
  900 seconds against a 3600-second absolute timeout is a strong cut — it says a member more than
  15 minutes behind the pack is not worth waiting for.
- **Quorum.** Lower it below `group_size - 1` when your tail is fat rather than thin. At
  `group_size: 8`, a quorum of 6 starts the clock as soon as two members are outstanding.
- **Acceptance is a separate decision.** `min_successful_group_size` rejects a group; the straggler
  keys only decide when to stop waiting. Cutting the tail harder pushes more groups below the
  acceptance floor, and a rejected group is replenished, which costs rollouts. Tune the two
  together.

!!! warning "Both apply only on the subprocess path"
    `straggler_timeout_seconds` is read exclusively inside `_arun_episode_with_subprocesses` in
    <span class="pl-src">platoon/train/areal/workflows/group_rollout_workflow.py</span>. With the
    default `use_subprocesses: false` the asyncio path is a plain `asyncio.gather` with no tail
    cutoff, and setting these keys does nothing at all.

"Settled" means terminal, not useful. An interrupted partial or a failed-closed member counts toward
the quorum, because excluding it would leave the last live member waiting out its full absolute
timeout — exactly the situation the feature exists to prevent.

## Subprocess rollouts

`workflow_config.use_subprocesses: true` runs each group member in its own spawned process from a
`ProcessPoolExecutor` sized to `group_size`. Turn it on when any of these is true:

- **Your environment spawns children.** AppWorld starts REST servers and databases; OpenHands starts
  an agent server. The worker calls `os.setpgrp()`, so a timeout kills the worker *and* its whole
  process tree. Without that, an orphan holding a port hangs every subsequent rollout on the node.
- **The model can install packages.** `run_rollout_subprocess` wraps the rollout in
  `isolated_rollout_python_environment`, a disposable venv overlay with `pip` shimmed to `uv pip`.
  Its docstring is explicit that the caller must be a short-lived subprocess: the context manager
  mutates `sys.executable`, `sys.path` and process environment variables, which is not safe to do
  on the shared asyncio path.
- **You want the straggler cutoff**, per the previous section.

The costs are real. Spawn startup per member per group; `group_size` concurrent processes on the
controller node; and memory. The 32-node config raises the rollout role's request to 128 GiB per
task, with a comment that the previous 512-GiB per-node cgroup reached about 477 GiB RSS and
contributed to rollout-controller stalls. If you turn subprocesses on at a large `group_size`, size
the rollout `scheduling_spec` memory before you submit.

The committed OpenReward and AppWorld AReaL configs all run with `use_subprocesses: true`, set in
their base configs and inherited through Hydra defaults. The textcraft, oolong, deepdive and
email-search inference configs set it `false` — lightweight in-process environments do not need the
isolation and should not pay for it. AppWorld and OpenReward keep it `true` even for inference.

`subprocess_shutdown_grace_seconds` (default 5.0) is the SIGTERM-to-SIGKILL window when the pool is
reaped; the committed large-scale configs use 10. Raise it if your environment needs to flush state
on shutdown, lower it if reaping is holding up the next group.

## Which timeout fires first

Five deadlines nest around a single rollout, and they fail very differently. This is the table worth
keeping open while you choose values.

| Deadline | Where it comes from | What it stops | What survives |
|---|---|---|---|
| Per step | `rollout_config.step_timeout`, default `300` | One `agent.act` or `env.step` | The episode ends normally with `trajectory_timed_out` in `misc`. **Trainable partial data.** No exception reaches the caller. |
| Whole trajectory | `rollout_config.timeout`, default `null` | The whole episode task | `trajectory_cancelled` in `misc`. A coherent partial collection: interrupted policy data is filtered downstream, completed descendants stay usable. |
| Group tail cutoff | `straggler_timeout_seconds`, once `straggler_quorum` members settle | Every still-pending member of that one group | Nothing from the cut members — they yield `None`. Settled peers are unaffected. |
| Subprocess hard kill | derived: `(timeout or 900) + 120 + 60` | The worker's whole process group, via `SIGALRM` then `killpg` | Nothing from that member, though proxy interactions recorded before the kill are still exported. |
| Parent future grace | derived: hard timeout `+ 30` | The executor future, then the whole pool | Nothing; forces a pool shutdown for the group. |

Above all of these sits the step deadline guard, which acts only at a step boundary and never
interrupts work in flight.

Two constants are not configurable and occasionally matter when reading logs: episode resource close
gets 10 seconds per resource in <span class="pl-src">platoon/episode/loop.py</span>, and proxy
session close gets 30 seconds in the group workflow.

The committed 32-node recursive config makes the intended ordering concrete:

```yaml
  rollout_config:
    # Trial-3 data contains legitimate recursive trajectories beyond 45m. A
    # 60m total deadline retains about 93% of observed successful completions;
    # the worker adds three minutes for initialization/cleanup before SIGKILL.
    timeout: 3600
    step_timeout: 2700
```

That gives `step_timeout` (2700) < `timeout` (3600) < hard kill (3780) < parent grace (3810), with
the straggler clock cutting in 900 seconds after the sixth member settles. Keep that ordering. Two
ways to break it:

- **`step_timeout` ≥ `timeout`.** The per-step deadline can never fire, so you lose the graceful path
  that preserves trainable partial data, and every slow trajectory becomes a cancellation instead.
- **Leaving `timeout: null` in subprocess mode.** The hard kill still exists, but it is computed from
  the `900` fallback — 1080 seconds total — which is almost certainly shorter than the trajectory
  budget you had in mind. Set `timeout` explicitly whenever `use_subprocesses` is on.

## Recovery and resume

```yaml
recover:
  mode: auto
  freq_epochs: 1
  freq_steps: 5
  freq_secs: 3600
```

`mode: auto` is what makes a restarted trainer resume at the right step. On startup AReaL looks for
a recover-info file for this `experiment_name`/`trial_name`/`fileroot`, and resumes if it is valid
and every model's recovery checkpoint exists; otherwise the run starts from step 0. `on` behaves the
same way. `disabled` and `off` never recover, and also make a deadline drain a hard error.

There is no mode that fails loudly when a checkpoint is missing — **auto silently starts over**. On a
32-node job you would much rather learn that from `global_step 0` in the first minute of the log
than from a flat reward curve six hours later.

### What a recovery checkpoint is, and is not

| | Recovery checkpoint | `saver` checkpoint |
|---|---|---|
| Format | DCP, sharded, with optimizer state unless `no_save_optim` | Hugging Face weights, no optimizer |
| Retention | **One rotating slot** per model | One directory per saved step |
| Purpose | Resume this trial | Export, evaluate, ship |

It is not a model you can hand to anyone, and it is not a history: the slot is overwritten, and only
the previous generation survives, as `recover_checkpoint_old`. Platoon patches AReaL's writer to
stage into `recover_checkpoint_new` and rotate with two atomic renames, precisely because a Slurm
kill mid-write used to leave a truncated, unreadable archive and no way back. If you want a model
you can load next month, that is `saver.freq_steps` — the committed 16-node config saves every 25
steps while recovery-checkpointing every 5.

### Choosing the frequency

`freq_steps` is the knob that matters, and it is a straight trade: checkpoint cost against work lost
per crash. Committed practice spans the range.

- **`freq_steps: 5`** on the 16- and 32-node Toolathlon configs, alongside `freq_secs: 3600`.
- **`freq_steps: 1`** on the recursive efficiency config, with the reasoning stated in the file:

> Checkpoint every completed update so a four-hour successor loses at most the in-flight group,
> rather than replaying several expensive recursive updates.

That is the rule. When one step is expensive enough that replaying two of them costs more than
writing a checkpoint every step, checkpoint every step. Note that a deadline drain forces a boundary
checkpoint regardless of the frequency gate, so `freq_steps > 1` costs you nothing on the *planned*
path — only on an unplanned kill.

### What recovery does not restore

- **Rollouts in flight.** The whole in-progress step is gone.
- **The OpenReward accepted-batch lookahead cache.** Recovery restores the quota phase, and
  therefore the environment composition, but exact task-record order can change after a restart.
- **The distinction between a fresh run and a resumed one, if you reuse a trial name.** This is the
  expensive mistake. Recovery keys off `experiment_name`/`trial_name`/`fileroot` and nothing else, so
  reusing a trial name after changing batch size or loss scaling silently recovers an incompatible
  optimizer. The committed configs treat a new `trial_name` as part of the change — the batch-size-8
  variant states plainly that it must not "recover the earlier batch-size-4 optimizer or rollout
  queue into this run". The OpenReward launcher makes `OPENREWARD_ACTOR_PATH` without an explicit
  `OPENREWARD_TRIAL_NAME` a hard error for the same reason; adopt that rule in your own launcher.

## Operational guards worth copying

The OpenReward launcher carries four guards that exist because something failed expensively. Read
them for the pattern, not the code — most are wired to this cluster's Enroot and Lustre setup.

| Guard | What it catches | What you pay |
|---|---|---|
| **Env-server supervisor** (<span class="pl-src">plugins/openreward/scripts/openreward-toolathlon-resilient-entrypoint.sh</span>) | One Uvicorn worker dying takes down only its own sessions, not the node. It is restarted at the same port with bounded exponential backoff, so nginx's session hashing stays valid. | A restart budget (5 attempts) and a reset window (300 s) to tune. An nginx exit or an exhausted budget is **deliberately fatal**, so the allocation gets replaced rather than serving a degraded node. |
| **Immutable runtime guard** (`environment_runtime_healthy` in the base launcher) | Anything that mutates the published training venv. The venv is `chmod a-w`'d, `pip`/`pip3`/`pip3.12` are symlinked to a script that exits 2, and a probe asserts every core distribution resolves inside the venv and that `pip` is *not* installed. | Two consecutive probe failures SIGTERM the trainer. If you build environments this way, `uv` must stay on PATH for the rollout overlay or model-authored installs break. |
| **SWE-rebench source guard** (<span class="pl-src">plugins/openreward/swe-rebench-runtime-guard.sh</span>) | A supplemental environment checkout drifting from its pinned commit, an unclean worktree, two denylisted "sandbox-incident" revisions, and a *behavioral* Enroot whiteout probe — file capabilities alone are only diagnostic, because they are dropped inside a nested user namespace. | It refuses to start. That is the point: a continuation cannot silently pick up a different checkout. |
| **Dependency auto-detection** (base launcher) | A Megatron config launched into an environment without Transformer Engine or APEX. The launcher greps the raw config for a `backend:` line starting with `megatron`, flips `OPENREWARD_BUILD_TE` / `OPENREWARD_BUILD_APEX`, and runs an import smoke test on the controller before starting the trainer. | Every derived config must keep a literal `actor.backend` line — the grep runs pre-Hydra and cannot see interpolation. <span class="pl-src">tests/test_openreward_prealloc_dependency_detection.py</span> executes the extracted block against quoted, unquoted and non-Megatron configs. |

A fifth is worth knowing even though it is not a correctness guard.
<span class="pl-src">slurm-scripts/gpu_keepalive.py</span> runs a periodic BF16 matmul burst on every
visible GPU so a cluster with idle-GPU cancellation does not reclaim the allocation during long
startup or long rollouts. Its readiness marker is written only *after* a real burst succeeds, so
readiness proves working GPUs rather than a live Python process. If your site has no idle
reclamation, skip it.

The health monitors that consume these signals run in the batch shell every
`OPENREWARD_SERVER_HEALTH_CHECK_SECS` (20). Three consecutive env-server failures, or two consecutive
runtime-probe failures, write a marker file and SIGTERM the trainer — because a dead env server used
to leave the trainer alive indefinitely, rejecting zero-data rollouts while every GPU stayed
allocated.

## Pre-flight for a long occupancy

This assumes the config checklist in [Scale to multiple nodes](../tutorials/multi-node.md) has
already passed: parallelism arithmetic, shared filesystem, environment reachable from every node.
These are the survival items on top of it.

- [ ] `PLATOON_TRAINING_DEADLINE_EPOCH` and `PLATOON_TRAINING_DRAIN_FILE` are exported, and the drain
      file path is **job-local**. A successor that reads its predecessor's marker drains immediately.
- [ ] Any inherited `PLATOON_TRAINING_DEADLINE_EPOCH` is unset at the top of the launcher.
      `--export=ALL` hands a continuation its predecessor's absolute deadline otherwise.
- [ ] `recover.mode: auto`, with a `freq_steps` you can afford to replay.
- [ ] `trial_name` is either new or intentionally reused. Write down which, before you submit.
- [ ] `rollout_config.timeout` is set explicitly if `use_subprocesses: true`, and
      `step_timeout < timeout`.
- [ ] If `straggler_quorum` is set, so is `straggler_timeout_seconds`, and `use_subprocesses: true`.
- [ ] `min_successful_group_size` is compatible with the tail you are about to cut.
- [ ] The rollout role's `scheduling_spec` memory covers `group_size` concurrent subprocesses.
- [ ] `--signal=B:USR1@300` is in the SBATCH header, and something handles `USR1`.
- [ ] Successor submission is single-writer. The base launcher claims the right to submit with an
      atomic `mkdir`, because four paths can race to request one.
- [ ] The restart budget is bounded. `OPENREWARD_MAX_INFRA_RESTARTS` defaults to 3, and clean drains
      deliberately do not consume it, so a healthy run chains indefinitely while a broken one gives
      up.
- [ ] You know the stop-file path. The launcher prints both the "stop after the current job" and
      "stop immediately" commands at startup — copy them out of the job log rather than
      reconstructing the path.
- [ ] `stats_logger.wandb.mode` matches reality. AReaL calls `wandb.login()` during trainer
      construction, *after* worker startup; the base launcher disables W&B with a warning when no key
      is present, precisely so missing telemetry cannot abort a 128-GPU job late.

## When a run dies at hour nine

### 1. Classify the exit before looking at anything else

The base launcher's own classification is the right decision procedure, whether or not you use that
launcher:

| Evidence | Meaning | Action |
|---|---|---|
| Drain marker exists **and** status 0 | Planned step-boundary drain | Resubmit. Not an infrastructure restart — it does not consume the budget. |
| Health-failure marker exists | An env server or the runtime probe killed the trainer | Fix the node or the environment, then resubmit against the budget. |
| Status 1, no markers | Trainer or controller runtime failure, including an exhausted rollout RPC retry | Resubmit against the budget. AReaL reports these as a plain exit 1. |
| Any other nonzero status | Terminal | Read the logs. Do not resubmit blindly. |

A clean exit 0 with a drain marker is a *continuation*, not a completion. If you see one and no
successor was submitted, check the stop file and the restart budget first.

### 2. Take inventory

| Artifact | Where | Still there? |
|---|---|---|
| Recovery checkpoint | `recover_checkpoint` — and possibly `_old` — under the trial's checkpoint root on `cluster.fileroot` | Yes. If the kill landed between the two renames, loading falls back to `_old`. |
| Exported model checkpoints | Per-step directories under the same root, written at `saver.freq_steps` | Yes. These are the only artifacts you can load outside this trial. |
| Rollout outputs | `rollout_config.output_dir`, with the workflow's `output_subdir` and the engine version appended | Yes, for whatever completed. |
| Drain marker JSON | `PLATOON_TRAINING_DRAIN_FILE` | Present only on a clean drain. Written atomically, so it is never partial. |
| W&B run | Online only if a key was present and the mode was not silently downgraded | Check the W&B mode line in the job log. |
| Trainer log | Slurm job output | Yes. |
| Env-server and keepalive logs | Separate per-node files named by run and job id | Yes, if your launcher writes them separately. The base launcher does. |

### 3. Know what is gone

- The in-flight step, entirely.
- Every completed update newer than the last recovery checkpoint. With `freq_steps: 5` that is up to
  four steps.
- The rollout controller's in-flight buffer — and, for OpenReward, the accepted-batch lookahead
  cache, so task order after the restart will not match what it would have been.
- Nothing else. Optimizer state is in the recovery checkpoint, and a resumed run picks up at
  `last_step_info.next().global_step`.

### 4. Resume

If the launcher chain is intact, a successor is already queued with
`--dependency=afterany:<job-id>` and will recover on its own. Restarting by hand: keep
`experiment_name`, `trial_name` and `fileroot` identical, leave `recover.mode: auto`, and check the
first lines of the trainer log for a recovered start step rather than 0. Starting from 0 under a
reused trial name is how nine hours of work gets overwritten by a fresh run.

## See also

- [Scale to multiple nodes](../tutorials/multi-node.md) — writing the launcher and submitting it.
- [AReaL backend internals](../architecture/areal.md) — how the scheduler, deadline guard and
  subprocess pool work.
- [Troubleshooting](../reference/troubleshooting.md) — the log lines these mechanisms emit.
- [Configuration reference](../reference/configuration.md) — every key with its default.
- [A training run, end to end](../walkthroughs/training-run.md) — what one global step does, and
  therefore what a drain protects.
