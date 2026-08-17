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
  --openreward.session_url=http://localhost:8080
```

## Inference

```bash
uv run python -m platoon.openreward.inference_scripts.run_inference \
  --config platoon/openreward/configs/inference/toolathlon_openhands_inference.yaml \
  --openreward.session_url=http://localhost:8080
```

## Multiple Environments

Set `openreward.environments` to route one training dataset across multiple
OpenReward servers. Equal `sampling_weight` values are the default balanced
mix; change them to request another task-group ratio.

```yaml
openreward:
  balance_accepted_batches: true
  accepted_batch_max_replacement_rounds: 8
  environments:
    - label: toolathlon
      env_name: toolathlongym
      session_url: http://localhost:8082
      session_urls_env_var: OPENREWARD_SESSION_URLS_TOOLATHLON
      sampling_weight: 1
    - label: tmax
      env_name: tmax/TMax-15K-Harbor
      session_url: http://localhost:8083
      session_urls_env_var: OPENREWARD_SESSION_URLS_TMAX
      sampling_weight: 1
    - label: swe_rebench
      env_name: nebius/SWE-rebench-V2
      session_url: http://localhost:8084
      session_urls_env_var: OPENREWARD_SESSION_URLS_SWE_REBENCH
      sampling_weight: 1
```

Both AReaL and Tinker construct deterministic, balanced submitted-task batches.
For mixed AReaL runs, `balance_accepted_batches` defaults to `true`: the training
dispatcher admits exactly the weighted quota for the current optimizer step,
waits for that round, and retries only environments whose groups were rejected.
This prevents a faster environment from displacing a slower one. Repeated
failures stop with a quota/attempt diagnostic after
`accepted_batch_max_replacement_rounds` instead of silently changing the mix.
Strict accepted balance is incompatible with AReaL `dynamic_bs`; set
`balance_accepted_batches: false` to restore completion-order batching and
multi-step prefetch when exact per-step composition is not required.

Replacement routing keeps at most one global batch of lookahead records per
environment. Overflow is skipped before rollout and reported through
`openreward/accepted_batch/input_discards`, so differential rejection rates
cannot grow controller memory without bound. Checkpoint recovery restores the
quota phase and therefore the accepted environment composition, but it does not
restore this small lookahead cache; exact task-record order can change after a
restart.

The quota is over task groups, not post-processing datums or tokens: recursive
trees and filtering can still produce different per-environment token totals.
Each environment may also set `train_task_limit`, `eval_task_limit`,
`task_indices`, distinct splits, and a static `session_urls` pool.

The 16-node Toolathlon + TMax + SWE-rebench example is launched with
`slurm-scripts/openreward-multienv-prealloc.sh`; its companion server helper
runs the TMax and SWE-rebench Python services on the host so they can create
writable Enroot task containers.

The wrapper delegates Toolathlon and trainer lifecycle management to the
tracked, sanitized `slurm-scripts/openreward-toolathlon-prealloc-base.sh`; its
keepalive and immutable-environment helpers are tracked as well. Credential
values and proxy URLs are never embedded in these scripts: provide any needed
`WANDB_API_KEY`, `OPENAI_API_KEY`, `LITELLM_API_KEY`, `HF_TOKEN`,
`OPENAI_BASE_URL`, and `LITELLM_BASE_URL` in the submission environment. Set
`PLATOON_USER_ROOT` only when the checkout is not under the usual
`<user-root>/source/platoon` layout.

The Toolathlon launcher bind-mounts
`plugins/openreward/scripts/openreward-toolathlon-resilient-entrypoint.sh`
into each environment container. Nginx continues to hash sessions to stable
internal ports, while the supervisor restarts only an unexpectedly exited
Uvicorn worker at its original port. Other workers and their live sessions stay
up. Restarts use bounded exponential backoff; tune
`OPENREWARD_WORKER_RESTART_MAX_ATTEMPTS`,
`OPENREWARD_WORKER_RESTART_RESET_SECS`,
`OPENREWARD_WORKER_RESTART_BACKOFF_INITIAL_SECS`, and
`OPENREWARD_WORKER_RESTART_BACKOFF_MAX_SECS` when needed. An nginx exit or an
exhausted worker budget remains fatal so the launcher's environment health
monitor can restart the allocation instead of serving a degraded node.

The wrapper defaults to the non-recursive mixed config. To launch the bounded
recursive variant explicitly, pass its config path to the same wrapper:

```bash
sbatch slurm-scripts/openreward-multienv-prealloc.sh \
  plugins/openreward/platoon/openreward/configs/areal/toolathlon_tmax_swe_openhands_areal_prealloc_16node-cp-ptc-recursive-r3-fp32-lm-head.yaml
```

That variant enables PTC and recursive subagents with a 50-step child budget
and maximum depth 2, plus depth-level weighting, leave-one-out baselines, and
root-success propagation. All three environments inherit the rollout-wide
`read_only` child policy, so no child can mutate or finish the shared root
session.
Use a new `trial_name` for subsequent launches so recursive checkpoints cannot
be recovered into the non-recursive experiment.

The helper verifies the external checkouts before starting: `external/tmax`
must be at the tested TMax commit and
`external/swe-rebench-v2-openrewardenv` at the tested SWE-rebench commit. The
defaults are pinned directly in the helper. Override `TMAX_SOURCE_REVISION` or
`SWE_REBENCH_SOURCE_REVISION` only after validating a newer fork revision and
rebuilding its `.venv-openreward` runtime.

The rollout attaches an OpenReward MCP bridge to OpenHands. The bridge owns the
OpenReward session. At reset time, the env resolves `get_task` itself and uses
the returned task prompt as the agent's first goal, so the root agent does not
spend its first model step bootstrapping the task. Environment tools still run
through the same bridge session and are exposed through OpenHands tool schemas.
Direct tools such as Toolathlon's `python_execute` remain direct; catalog tools
use the environment's advertised dispatcher/meta-tool. Bridge tools declare an
empty OpenHands resource set, so PTC can run independent MCP calls concurrently
with `asyncio.gather(...)`.

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

OpenReward defaults PTC to `programmatic_tool_calling_mode:
orchestration_only`. In this mode its local Python runtime can transform values
and orchestrate `tools`/`atools`, while common direct filesystem, process,
shell, and network operations raise corrective errors because they would act on
the rollout worker rather than the task container. This is a behavioral guard,
not a security boundary. Set the mode to `unrestricted` only when PTC
intentionally owns the execution environment. Environment tools can advertise
task-side capabilities and direct or catalog-dispatch invocation routes; older
OpenReward providers can supply equivalent per-tool declarations through an
environment's `tool_routing_overrides` mapping.

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

Child environment access defaults to `shared`. To make every child
investigative while keeping the parent as sole writer, set the rollout-wide
policy:

```yaml
openreward:
  subagent_environment_access: read_only
```

`read_only` affects ordinary forked children only; the root keeps every
environment tool. Children receive the strict inspection allowlist `get_task`,
`get_status`, `get_tool_details`, and `view`. They do not receive `bash`,
`str_replace`, `create_file`, `submit_answer`, `claim_done`, generic
`call_tool`, `python_execute`, or unknown future environment tools. A child
should return file/line evidence and a proposed replacement or patch through
its local `finish` tool; only the parent edits and submits. Synthetic reward
verifier branches are an intentional exception: they receive shared
non-terminal tools so they can reproduce environment actions and run tests.

This phase-1 mode does **not** fork the OpenReward environment or create a Git
worktree. Child `view` calls inspect the same live workspace/session as the
parent, and concurrent parent activity can change what a child sees. In
`shared` mode, concurrent edits can race. Children receive mutating tools but
not the known terminal tools (`submit_answer` and `claim_done`); they report
through their local `finish`, and the parent submits the shared result. Reward
verifiers also omit these terminal tools and are prompted to inspect with their
direct environment tools before considering delegation. Every subagent receives
the same environment-neutral system guidance explaining that its parent and
siblings may share the live workspace. It can use the shared tree directly or,
when conflict risk warrants it, create an isolated directory or Git worktree
and then integrate the result or return its location, commit, or patch to the
parent. Roots do not receive this child-only guidance.

Set `openreward.enable_subagent_reward_judging: true` to automatically launch a
verifier agent after each normal subagent finishes. The verifier receives the
child goal, final message, and trajectory id, then inspects the shared
environment with tools and returns a JSON verdict via `finish`. The judged child
trajectory stores the normalized result in `misc.subagent_reward_judgment`; the
verifier trajectory stores the judged child id in
`misc.subagent_reward_verifies_trajectory_id`.
`openreward.subagent_reward_judge_max_steps` controls the verifier step budget
and defaults to 20. A verifier receives a fresh, verifier-bound
`launch_subagent` runtime and may use one helper level for independent,
specialized checks. Helper calls use `subagent_default_max_steps` (50 in the
recursive training configs). Helpers inherit the verifier-tree marker, remain
nested under the verifier, are excluded from policy training, and never receive
a second reward verifier; deeper verifier delegation is rejected independently
of the policy recursion-depth limit. Verifier trajectories are marked
`misc.exclude_from_training: true`; judged worker trajectories expose
`reward/subagent_judgment`, and OpenReward's reward processor uses that score as
the worker subtrajectory reward.

## Full-allocation GPU keepalive

Preallocated OpenReward jobs start `slurm-scripts/gpu_keepalive.py` before
environment-server and immutable-environment setup, then keep that same overlapping
`srun` step alive for the complete trainer runtime. The launcher starts one task per
allocation node with all `GPUS_PER_NODE` devices visible to that task. On the
standard 16-node allocation this means 16 keepalive ranks, each touching its
node’s eight GPUs.

Each rank publishes a readiness marker only after completing a real BF16
matrix-multiply burst. The launcher waits for every node and continuously
monitors the step; an unexpected keepalive exit terminates the unprotected job.
Normal launcher cleanup stops the monitor and keepalive after the trainer exits.

Defaults match the established preallocated jobs: a five-second tick,
4096-square matrices, 2000 operations per burst, a 16200-second maximum
runtime, three consecutive errors before failure, and an expected device count
equal to `GPUS_PER_NODE`. The launcher overrides the in-script startup delay to
zero. Tune these with the `KEEPALIVE_*` environment variables or disable the
step with `OPENREWARD_GPU_KEEPALIVE=0`.
