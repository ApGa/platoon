# Run at scale

You have a run that works on one node. This page takes it to a Slurm allocation of many nodes and
keeps it alive across a wall-time limit shorter than the run.

<span class="pl-tag pl-tag--areal">AReaL</span> Everything here is AReaL-side: a Tinker-compatible
backend trains on a remote service, so there is no allocation to place or keep alive.

## What changes

Less than you expect. The loss, the workflow, the dataset and the reward are untouched, and every
key below has its default in the [configuration reference](../reference/configuration.md). Comparing
a single-node config with its multi-node sibling, four things move:

```yaml
# Launch AReaL's workers as srun steps inside an allocation you already hold.
scheduler:
  type: slurm_prealloc

cluster:
  n_nodes: 2
  n_gpus_per_node: 8
  fileroot: /shared/platoon/experiments
  name_resolve:
    type: nfs
    nfs_record_root: /shared/platoon/name_resolve

rollout:
  backend: sglang:d8p1t1     # 8 GPUs of inference
actor:
  backend: fsdp:d4p1t1c2     # 8 GPUs of training
```

| Key | Why it changes |
| --- | --- |
| `scheduler.type` | `slurm_prealloc` selects `PreallocatedSlurmScheduler` instead of AReaL's own scheduler. |
| `cluster.n_nodes` | Must match the allocation. Launchers usually override it from `SLURM_NNODES`. |
| `cluster.fileroot`, `cluster.name_resolve.nfs_record_root` | Must live on a filesystem every node can see. |
| `rollout.backend`, `actor.backend` | The parallelism has to add up to the GPUs you allocated. |

### The GPU arithmetic

Both backend strings declare a placement whose world size is the product of its `d`, `t`, `p` and
`c` dimensions, and together they must fill the allocation exactly. Shapes that have run, at 8 GPUs
per node:

| Nodes | `rollout.backend` | `actor.backend` | Split |
| --- | --- | --- | --- |
| 2 | `sglang:d8p1t1` | `fsdp:d4p1t1c2` | 8 + 8 |
| 16 | `sglang:d6p1t8` | `megatron:(attn:d5p2t4c2\|ffn:d5p2t1e8)` | 48 + 80 |
| 32 | `sglang:d12p1t8` | `megatron:(attn:d10p2t4c2\|ffn:d10p2t1e8)` | 96 + 160 |

The preallocated scheduler enforces one constraint the single-node path does not: **whole nodes
only**. A role whose total GPU count is not a multiple of `n_gpus_per_node` is a hard error, so plan
the split before you submit. Quote the hybrid Megatron form in YAML — it is a bare scalar containing
parentheses and a pipe.

### Shared filesystem

This is where multi-node runs hang instead of failing. `cluster.name_resolve` is how workers find
each other: with `type: nfs`, each writes its address under `nfs_record_root` and reads its peers'
from the same place. Point that at a node-local path and every node builds a private,
consistent-looking view of a one-member cluster. Nothing errors; startup never completes.

Before you trust a path, write to it from one allocated node and list it from another —
`touch ${dir}/probe.$(hostname) && ls ${dir}/probe.*`. The same applies to `cluster.fileroot` and
to the Python environment itself. Resolve model weights to a local snapshot directory too, and
export `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` into every worker — a few hundred ranks
resolving one Hub ID at startup gets the cluster rate-limited.

## Choosing parallelism

| | `fsdp` | `megatron` |
| --- | --- | --- |
| Extra install | none beyond `uv sync --extra areal` | Transformer Engine and APEX, source-built |
| Dimensions | `d`, `t`, `c` | `d`, `t`, `p`, `c`, `e`, plus the split `attn`/`ffn` form |
| Expert parallelism | no | yes |
| Typical use | dense models on one or two nodes | MoE checkpoints and large models at 8+ nodes |

**Start on FSDP.** It needs nothing the AReaL extra does not already install, and the Megatron actor
sits behind a lazy import so an FSDP-only environment never pulls in Transformer Engine.

**Move to Megatron when the model forces you** — an MoE checkpoint you want expert-parallel, or a
model too large for data-plus-context parallelism. The cost is a real build: Transformer Engine's
torch bindings are source-only and Megatron's fused gradient-accumulation kernel requires APEX, so
both must be compiled where `nvcc` exists and then cached. Follow the
[installation page](../get-started/installation.md), and prove the build on a single-node Megatron
config before you ask a scheduler for sixteen nodes.

**LoRA** is the whole story on a Tinker-compatible backend: training is always a LoRA adapter, and
`train.lora_rank` (default 32) is the only capacity decision. On AReaL it is off by default and
configured upstream through `actor.use_lora`, `actor.lora_rank`, `actor.lora_alpha` and
`actor.target_modules`. Write `target_modules` as fully qualified patterns so they select the
language model and nothing else.

## The preallocated model

AReaL's upstream Slurm scheduler submits one job per worker role. That is the wrong shape here: the
trainer, the inference engines and the environment servers all have to be alive at once, and a
partially scheduled set of jobs burns queue priority while doing nothing.
`PreallocatedSlurmScheduler` inverts it. You `sbatch` one allocation; it launches each role with
`srun` as a step inside it, leaving AReaL's RPC and name-resolution model unchanged. You keep the
job and a batch shell that outlives the trainer, which is where signal handlers, health monitors and
successor submission live.

So one sbatch job contains overlapping `srun` steps you start — any services your task needs, plus
the controller on the first node — and the actor and inference steps the trainer starts for you. The
scheduler is configured entirely from the environment; there is no config schema for it.

| Variable | What it sets |
| --- | --- |
| `PLATOON_AREAL_PREALLOC_SRUN_BIN`, `..._SRUN_ARGS` | How each worker step is launched. `--overlap` belongs in the args. |
| `PLATOON_AREAL_PREALLOC_GPU_FLAG` | The GPU request template, e.g. `--gpus-per-node={gpus}`. |
| `PLATOON_AREAL_PREALLOC_WORKER_PREAMBLE` | Bash prepended to every worker command: `PATH`, offline flags, library paths. |
| `PLATOON_AREAL_PREALLOC_USE_PYXIS` | `0` drops every container argument for a bare-metal cluster. |
| `PLATOON_AREAL_PREALLOC_CONTAINER_IMAGE`, `..._MOUNTS`, `..._WORKDIR` | Pyxis/Enroot wiring when you do use containers. |

Separated roles are round-robined across allocation nodes rather than stacking on node 0, and worker
configuration runs concurrently across hosts. [Execution](../architecture/execution.md) covers both.

## The repository's Slurm scripts

`slurm-scripts/` holds one large base launcher,
<span class="pl-src">slurm-scripts/openreward-toolathlon-prealloc-base.sh</span>, and a set of thin
wrappers around it. Each wrapper picks a config, exports the run-specific variables, and `exec`s the
base launcher. Copy that shape: durable machinery in one script, per-experiment differences in a
wrapper. The base launcher is written for one cluster, so read it as a list of problems long jobs
have and decide which of them you also have.

**Portable, reuse directly:** `scheduler.type: slurm_prealloc` and the `cluster:` block; the
`PLATOON_AREAL_PREALLOC_*` interface; the deadline and drain contract below; the `workflow_config`
straggler keys; `--signal=B:USR1@300` plus an `sbatch --dependency=afterany` successor; and
<span class="pl-src">slurm-scripts/gpu_keepalive.py</span>, a periodic matmul burst on every visible
GPU so a cluster with idle-GPU reclamation does not take the allocation back during a long startup.

**Site-specific, read for the idea:** account and partition names, all shared-filesystem paths, the
Pyxis and Enroot arguments, the content-addressed venv builder, and everything named `OPENREWARD_*`.

Three details bite everyone. The controller step must run **without** container arguments, because
it spawns nested `srun` steps and cannot do that from inside a Pyxis container. Slurm runs a copy of
your script from its spool directory, so a launcher that resubmits itself has to locate the checkout
through `PLATOON_REPO_ROOT` or `SLURM_SUBMIT_DIR` rather than `BASH_SOURCE`. And AReaL overrides on
the controller command line are bare `key=value` — `cluster.n_nodes=16`, never
`--cluster.n_nodes 16`; `--config` is the one flag that takes a dash.

## Keeping a long run alive

### Drain at a step boundary

Being killed mid-update costs you the whole step plus everything since the last recovery checkpoint.
`StepDeadlineGuard` refuses to *start* a step it does not think will finish: it checkpoints the last
completed step, writes a drain marker, and exits cleanly. Two exports turn it on, and there is no
config key.

```bash
export PLATOON_TRAINING_DEADLINE_EPOCH=$(
  date --date="$(scontrol show job "${SLURM_JOB_ID}" -o | sed 's/.*EndTime=//;s/ .*//')" +%s
)
export PLATOON_TRAINING_DRAIN_FILE=${JOB_STATE_DIR}/deadline-drain-requested.json
```

The estimate is `max(PLATOON_DEADLINE_INITIAL_STEP_SECONDS, max(recent steps) * multiplier)` plus
`PLATOON_DEADLINE_SAFETY_SECONDS`. The first is a permanent floor, not a starting guess, so setting
it above your real step time makes the guard drain early forever; 1800 is the usual value. The
second is shutdown headroom — raise it if you tear down services as well as the trainer. Keep both
values job-local and derive them from the current job: a successor that inherits its predecessor's
absolute deadline drains immediately.

### Do not let one straggler hold the group

Agentic rollouts have heavy tails, and one member stuck pulling a container holds all of its peers
until the absolute rollout timeout. Four `workflow_config` keys cut it:

```yaml
workflow_config:
  use_subprocesses: true
  straggler_timeout_seconds: 900
  straggler_quorum: 6
  min_successful_group_size: 4
```

Once `straggler_quorum` members have settled — completed, interrupted or failed — the rest get
`straggler_timeout_seconds` more and then the group's process pool is reaped. Acceptance is separate:
`min_successful_group_size` rejects a group too small for a meaningful within-task baseline, and a
rejected group is replenished, which costs rollouts. Cut the tail harder and more groups fall below
the floor, so tune the two together.

The cutoff is read only on the subprocess rollout path, hence `use_subprocesses: true`. That path is
worth having anyway when your environment spawns children: each rollout gets its own process group,
so a timeout kills the whole tree instead of leaving an orphan holding a port. Set
`rollout_config.timeout` explicitly when it is on, and keep `step_timeout` below it.

### Recover and resume

```yaml
recover:
  mode: auto
  freq_epochs: 1
  freq_steps: 5
  freq_secs: 3600
```

`mode: auto` is what makes a restarted trainer resume at the right step. On startup AReaL looks for
recovery state matching this `experiment_name`, `trial_name` and `fileroot`; if it is valid the run
continues, otherwise it starts from step 0. `freq_steps` trades checkpoint cost against work lost
per crash — 5 suits steps of tens of minutes, 1 when replaying two steps costs more than writing a
checkpoint every step. A drain forces a checkpoint regardless. A resume restores optimizer state and
the step counter, and loses the in-flight step plus every update newer than the last checkpoint.

A recovery checkpoint is one rotating slot per model, sharded, with optimizer state — not a model
you can hand to anyone. That is `saver.freq_steps`, which writes Hugging Face weights per saved step.

!!! warning "A new run needs a new `trial_name`"
    Recovery keys off `experiment_name`, `trial_name` and `fileroot` and nothing else. Reuse a trial
    name after changing batch size or loss scaling and you silently recover an incompatible
    optimizer. Decide before you submit whether you are resuming or starting over.

### Chain allocations

`--signal=B:USR1@300` in the SBATCH header delivers `SIGUSR1` to the batch shell five minutes before
the limit. The handler submits a successor:

```bash
sbatch --parsable --dependency="afterany:${SLURM_JOB_ID}" --export=ALL "${JOB_SCRIPT}" "${CONFIG}"
```

Several paths can request one — the wall-time signal, `SIGTERM`, a health monitor, a clean drain —
so claim the right to submit with an atomic `mkdir` and let the rest become no-ops. Bound the chain
two ways: a restart budget that only *unplanned* restarts consume, so a healthy run chains
indefinitely while a broken one gives up; and a stop file whose presence makes the next submission
decline. Print the stop path and the `scancel` command at startup so you can copy them out of the
job log.

Classify before you resubmit by hand. A drain marker with exit status 0 is a planned continuation. A
health-monitor marker means a node or a service needs attention first. Any other nonzero status is
terminal — read the logs rather than replaying the failure across allocations.

## Pre-flight checklist

Before you hold many GPUs for hours:

- [ ] The run works end to end on one node with the same task and reward code.
- [ ] `rollout.backend` and `actor.backend` world sizes sum to `n_nodes * n_gpus_per_node`, and each
      role's GPU count is a multiple of `n_gpus_per_node`.
- [ ] `cluster.fileroot` and `cluster.name_resolve.nfs_record_root` are on a filesystem you verified
      is shared, by writing from one allocated node and reading from another.
- [ ] The Python environment resolves identically on every node, and model weights are in a local
      cache directory rather than a Hub ID every rank resolves at once.
- [ ] `trial_name` is new, or you intend to recover the existing trial, and `recover.mode: auto` has
      a `freq_steps` you can afford to replay.
- [ ] `PLATOON_TRAINING_DEADLINE_EPOCH` and `PLATOON_TRAINING_DRAIN_FILE` are exported and job-local
      if the run is longer than the time limit.
- [ ] `stats_logger.wandb.mode` matches reality. W&B login happens during trainer construction,
      after worker startup, so a missing key fails the job late.
- [ ] Smoke-tested at two nodes with a short `--time` and a low `total_train_steps`.

## Next

- [Backends](../architecture/backends.md) — what each training backend expects of your cluster.
- [Execution](../architecture/execution.md) — what one global step does, and what a drain protects.
- [Configuration reference](../reference/configuration.md) — every key with its default.
