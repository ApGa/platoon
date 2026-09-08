# Toolathlon on the miles-tinker gateway: client-side R3, straggler policy, configs

This documents the platoon-side pieces of the Qwen3.6-35B-A3B recursive
Toolathlon run on the miles-tinker gateway (design sections 8.3-8.7).

## Routing handle in datums (contract C2)

`platoon/utils/tinker_data_processing.py`:

- `SequenceAccumulator.last_routing_id: int | None` holds the gateway
  `routing_id` of the LAST interaction merged into the accumulated sequence
  (`clear()` resets it). `trajectory_to_data` sets it after every appended
  interaction from `getattr(interaction, "routing_id", None)`
  (`TinkerLLMInteraction.routing_id`, captured by the proxy from the gateway's
  JSON sample response).
- `make_datum_from_accumulator` emits
  `loss_fn_inputs["routing_handle"] = TensorData int64 [1]` when set
  (`ROUTING_HANDLE_KEY`). Because merged interactions are prefix extensions,
  the datum's tokens (`model_input + target_tokens[-1:]`) equal that call's
  prompt + generated tokens, so the gateway can materialize the sampled expert
  routing for the whole datum from one handle. A final interaction without an
  id leaves the key absent (coverage is all-or-nothing per datum).
- `platoon/train/tinker/rl.py` strips only
  `mask, checkpoint_version, traj_depth, traj_start, LOSS_NORMALIZATION_TOKENS_KEY`
  before `forward_backward`; `routing_handle` passes through.

## Straggler quorum / timeout / minimum group (contract C11)

`WorkflowConfig` (`platoon/train/tinker/config_defs.py`) gains
`straggler_timeout_seconds: float | None = None`,
`straggler_quorum: int | None = None`, `min_successful_group_size: int = 1`
with the AReaL validation rules (quorum requires the timeout; quorum and
minimum in `[1, group_size]`).

`GroupRolloutWorkflow._run_group_rollouts` (in-process, no subprocesses):
without a timeout it is `asyncio.gather`; with one, once `straggler_quorum`
members (default `group_size - 1`) have settled the rest get
`straggler_timeout_seconds`, then are cancelled (`task.cancel()`; `run_rollout`'s
cleanup marks trajectories interrupted; leaked Toolathlon sessions are reaped
server-side). Cancelled members become `observed=False` outcomes and the
`straggler_cancelled` scalar is emitted (0 when nothing was cancelled). A group
with fewer than `min_successful_group_size` usable members returns an empty
task output (workload still accounted; `group_size_rejected` scalar).

## Trainer robustness

Every `save_checkpoint_async` in `rl.py` (launch, per-step, interrupt, final)
runs under `_await_with_heartbeat(name="save_checkpoint", total_timeout=7200)`;
the per-step save also feeds the watchdog (`shared_state.watchdog`), since a
full-parameter `save_state` takes minutes. The rollout worker loop feeds the
watchdog every 60 s while a task group is in flight
(`_await_rollout_with_heartbeat`, unshielded so an interrupt still cancels the
group); a group can take rollout timeout (3600 s) + straggler grace (900 s) +
the judge/reward tail, so `watchdog.timeout_seconds` is 7200 in the run configs.

The proxy clamps every request's `max_tokens` to the context tokens left after
the rendered prompt (`TinkerLLM._fit_max_tokens_to_context`, AReaL parity), so
the condenser's 26214-token completion budget is a ceiling; a prompt that fills
the window raises `litellm.ContextWindowExceededError`, which OpenHands treats
as a hard condensation request instead of a retried transport error.

`platoon/train/tinker/restart_wrapper.py` (pre-existing) restarts the trainer
on the watchdog exit code: `python -m platoon.train.tinker.restart_wrapper
--max-restarts 20 --restart-delay 700 --watchdog-exit-code 2 -- <trainer cmd>`.

## Single-process rollout hazards (stream isolation)

The Tinker path runs *every* trajectory in one asyncio process on the head node;
the AReaL path ran a subprocess per node, so a rollout owned its interpreter.
Library code that mutates interpreter globals is therefore no longer per-rollout
state. The 15-node run `ta15-rec-q36-r3-v1` lost 968 of 1060 trajectories at step
0 to `MCPError: MCP Connection Failure` (58 % of batch 0, 100 % from batch 1 on)
for exactly that reason, while the 4-concurrent smoke on the same code lost 0 of
34:

- `openhands.tools.programmatic_tool_calling.impl` (enabled by
  `openreward.enable_programmatic_tool_calling`, which is true in **both** run
  yamls) wraps each cell in `contextlib.redirect_stdout` / `redirect_stderr`,
  which rebind the process-global streams to `io.StringIO`. Tool executors run
  on worker threads (`Agent.astep` -> `_aexecute_actions` ->
  `loop.run_in_executor`), so cells from different rollouts overlap.
- `fastmcp.client.transports.stdio` samples `sys.stderr` when it spawns the
  stdio server and hands it to `subprocess.Popen`; a `StringIO` has no
  `fileno()`, so any `mcp_bridge` spawned inside a cell's window dies with
  `io.UnsupportedOperation: fileno` -> `RuntimeError: Client failed to connect:
  fileno` -> `MCPError("MCP Connection Failure")` (the SDK drops the message at
  `openhands/sdk/mcp/client.py:55`, which is what made this hard to see).
- `redirect_*` restores the stream it observed on entry, so interleaved
  enter/exit permanently strands `sys.stderr` on a dead buffer -- hence the
  degradation from 58 % to 100 %.

`plugins/openreward/platoon/openreward/stream_isolation.py`, installed from
`rollout.py` at import, replaces `sys.stdout`/`sys.stderr` with a
`ThreadScopedStream` that keeps a real `fileno()` and routes writes to a
per-thread capture stack, rebinds the PTC module's `redirect_*` names to
push/pop that stack, and pins `StdioMCPServer.to_transport`'s `log_file` to the
console stream so a bridge spawn never depends on the current `sys.stderr`.
`OPENREWARD_STREAM_ISOLATION=0` restores the upstream behaviour (debug only).

Separately, `_patch_mcp_boot_timeout` now retries a *fast* `MCPError` boot
failure (`OPENREWARD_MCP_CONNECT_RETRIES`, default 2, `..._RETRY_DELAY` 2.0 s)
but never a `MCPTimeoutError`. That is a mitigation for transient service
hiccups over a 168 h run, not the fix.

## Runaway PTC cells (cell guard, observability only)

A second consequence of the single-process design, on the same code path.
OpenHands executes each programmatic-tool-calling cell **in-process** on an
`InteractiveShellEmbed` (`ProgrammaticToolCallingExecutor.__call__` ->
`_run_cell` -> `shell.run_cell_async`), and there is **no timeout anywhere on
that path**. The call runs on a worker thread taken from the event loop's
default executor (`Tool.acall` -> `loop.run_in_executor(None, ...)`, reached via
`Agent.astep` -> `_aexecute_actions`), and that pool is shared by every
concurrent rollout in the client process.

The failure mode is that a model-authored cell need not terminate, and nothing
in the stack can make it:

- **A Python thread cannot be killed.** `asyncio` cancellation, `wait_for`, and
  platoon's straggler timeout (`WorkflowConfig.straggler_timeout_seconds`,
  `group_rollout_workflow`) all abandon the *awaiting coroutine* only. The
  thread underneath keeps running the model's code forever, holding a core and
  an executor slot.
- **The leak is silent.** The trajectory is recorded as interrupted, the train
  loop keeps turning, the watchdog keeps getting heartbeats. The only symptom is
  throughput quietly decaying as threads are consumed, until enough have leaked
  that rollouts stop being scheduled at all while the trainer idles on 15 nodes.
- **The pool is capped.** The default asyncio executor is
  `min(32, cpu_count + 4)` threads, shared across
  `train.num_concurrent_rollout_workflow_workers` x workflow `group_size` root
  trajectories plus their depth-2 subagents and judges.

`plugins/openreward/platoon/openreward/cell_guard.py`, installed from
`rollout.py` at import next to the stream isolation, wraps
`ProgrammaticToolCallingExecutor.__call__` (idempotently, degrading to a no-op
when OpenHands is absent) to record each cell's start time, worker thread and a
one-line prefix of the source the model wrote. A daemon monitor thread
(`ptc-cell-guard`) then reports each cell once as it crosses each threshold.

**The guard only measures and reports. It never interrupts, cancels, truncates
or kills a cell, and it never exits the process.** A legitimate long-running
cell is reported and then left alone; the report is a diagnostic, not a verdict.

| env var | default | meaning |
| --- | --- | --- |
| `OPENREWARD_PTC_CELL_GUARD` | `1` | `0`/`false`/`no`/`off` makes the whole module a no-op |
| `OPENREWARD_PTC_CELL_WARN_SECONDS` | `300` | age at which a running cell is reported *stuck* (WARNING) |
| `OPENREWARD_PTC_CELL_LEAK_SECONDS` | `1800` | age at which its thread is *presumed leaked* (ERROR); clamped up to the warn threshold |
| `OPENREWARD_PTC_CELL_GUARD_REPORT_SECONDS` | `60` | cadence of the summary line, emitted only while a count is non-zero |
| `OPENREWARD_PTC_CELL_SOURCE_CHARS` | `240` | how much cell source each report quotes |

What an operator sees. Once per cell, as it crosses the warn threshold:

```
PTC cell guard: cell #417 still running after 305s (warn threshold 300s) on thread
asyncio_0 (id=139742867777344). The cell runs in-process on an IPython shell with no
timeout and cannot be interrupted; this thread is occupied until the cell returns on
its own. Cell source: 'while True: check_status()'
```

then once more when it is written off, and a summary every 60 s while any count
is non-zero:

```
PTC cell guard: cell #417 PRESUMED LEAKED after 1801s (leak threshold 1800s) on thread
asyncio_0 (id=139742867777344). Treat this rollout worker thread as permanently lost:
Python cannot kill a running thread, so it is counted, not recovered. ...
PTC cell guard: 3 cell(s) stuck >300s, 1 thread(s) presumed leaked >1800s (5 cell(s)
in flight; 2104 started / 2099 finished this process). Leaked threads are never
returned to the rollout executor pool.
```

A slow-but-finite cell that eventually returns logs `-- slow, not leaked` and
decrements the counts, so both counts are *current*, not cumulative: a non-zero
steady state means threads that are genuinely gone.

`rl.py` publishes the counters as scalars on every training step
(`_ptc_cell_guard_stats`, folded into `step_stats` just before
`stats_logger.log`): **`ptc_cells_stuck`** and **`ptc_threads_leaked`**, plus
`ptc_cells_running` / `ptc_cells_started` / `ptc_cells_finished` for context.
The import of the plugin is soft, so a run without openreward logs no such
scalars rather than failing. `ptc_threads_leaked` climbing off zero is the
alert: it is monotonic in practice and each unit is one rollout worker thread
that will never come back.

**This is instrumentation, not a fix.** The durable fix is to run each rollout
in its own **process**, as the AReaL path did. A process is the only unit of
execution Python can actually kill, so only process-per-rollout can *bound* a
runaway cell rather than merely counting it. Process isolation is independently
required anyway: some environments cannot share an interpreter at all --
AppWorld patches `time` with `freezegun` process-wide, so two concurrent
AppWorld rollouts in one process are already incorrect regardless of runaway
risk. Until rollouts are subprocesses, the residual exposure is unchanged: a
runaway that allocates can still OOM the client process and take training with
it, and the guard cannot see a thread that is blocked inside a lock it does not
own.

## Process-per-rollout (`rollout_isolation`)

The durable fix for both hazards above, and for a third: a segfault or an OOM in
rollout code currently kills the trainer and costs a ~490 GB checkpoint reload.
`WorkflowConfig.rollout_isolation` selects where a rollout runs.

- `inprocess` (default) -- today's behaviour exactly: rollouts are asyncio tasks
  in the trainer.
- `process` -- one `exec`'d worker per rollout,
  `python -m platoon.train.tinker.workers.entrypoint <request>`, in its own
  session and process group, for exactly one rollout tree and then gone.

Both modes run the **same function** (`platoon/train/tinker/workers/run.py`,
`run_rollout_request`); only the host differs. Turning it on takes no config
edit, because the launcher forwards arbitrary overrides:

```
EXTRA='--train.workflow_config.rollout_isolation process' <launcher>
```

Set it on **both** `train.workflow_config` and `eval.workflow_config` when eval
is enabled: an unisolated eval rollout reintroduces exactly the unkillable-cell
and crash-containment hazards this removes.

### What isolation buys

| hazard | `inprocess` | `process` |
| --- | --- | --- |
| a PTC cell that never returns | counted only; consumes an executor thread and a core for the rest of the run | `killpg(SIGKILL)` at the rollout deadline, with a `.diag` marker proving it |
| a segfault in rollout code | kills the trainer | one member fails; siblings and the trainer are untouched |
| gradual memory growth | unbounded | RSS watchdog exits with a distinct code and is classified as an OOM |
| freezegun / IPython singleton / `sys.stdout` rebinding | shared by every concurrent rollout | one rollout tree per interpreter |
| a rollout's MCP bridges | reaped only when the trainer exits | reaped with their own worker, via `PR_SET_PDEATHSIG` |

Honest limits: depth-2 subagents, judges and the reward verifier stay inside
their root rollout's process by design, so the IPython/stream class of bug is
reduced to a blast radius of one trajectory tree rather than eliminated; and a
single huge allocation can still reach the kernel OOM killer, whose victim
selection is not ours. Run `deploy/abci/openreward-node-memory-guard.py` on the
head node before flipping this in production.

### Config keys

All defaulted, so a config that sets none of them behaves exactly as before.

| key | default | meaning |
| --- | --- | --- |
| `rollout_isolation` | `inprocess` | `inprocess` or `process` |
| `worker_startup_budget_seconds` | `300` | import/registration deadline; sized for a concurrent cold start on shared storage (~26 s of imports each) |
| `worker_finalize_grace_seconds` | `120` | added to `rollout_config.timeout` for datum conversion and writing the response |
| `worker_parent_grace_seconds` | `60` | parent wall deadline above the child's, so the child's own alarm owns its tree first and can leave a marker |
| `worker_termination_grace_seconds` | `10` | SIGTERM -> grace -> SIGKILL; must stay well under the launcher's 30 s budget for the whole client |
| `worker_launch_concurrent_groups` | `2` | groups (not rollouts) allowed to be starting at once; a group is admitted atomically so its members never skew against the straggler quorum clock |
| `worker_rss_soft_limit_bytes` | `24 GiB` | per-worker RSS soft limit; above it the worker writes `phase=rss` and exits with code 87 |
| `worker_mem_available_floor_bytes` | `0` | refuse to launch below this `MemAvailable`; 0 disables |
| `worker_binding_poll_seconds` | `5` | how often a worker re-checks the published sampling-client binding |
| `worker_unbounded_rollout_budget_seconds` | `86400` | child budget when `rollout_config.timeout` is null |
| `worker_python_env_isolation` | `false` | wrap the rollout in a disposable Python overlay; requires `rollout_isolation: process` |

Timeout ladder at production values (`rollout_config.timeout: 3600`):

```
child startup alarm      300 s   (armed before the imports, from the environment)
child rollout alarm     3720 s   = timeout + finalize grace, re-armed at readiness
parent wall deadline    3780 s   = child budget + parent grace
straggler tail grace     900 s   after quorum settles (the dominant path)
SIGTERM -> SIGKILL        10 s
watchdog                7200 s
```

### Model access

Workers never build a `ServiceClient` (that would be a gateway session per
rollout). The trainer pickles its `SamplingClient` -- ~287 bytes -- into
`$TMPDIR/platoon-rollout-ipc/model_binding.pkl` (0600) on every
`update_sampling_client`, and each worker re-`stat`s that file at most once per
`worker_binding_poll_seconds` and reloads on change. Workers **follow**, they do
not pin: pinning is a no-op under the full-finetune gateway (which always serves
the latest pushed weights) and a hard mid-rollout failure under a LoRA gateway
(sampler versions are retired after `sampler_versions_to_keep` steps). Weight
staleness and `max_staleness` accounting are unchanged.

Every worker sets `TINKER_TELEMETRY=0` before any tinker import and reports that
it did. This is not tidiness: an unpickled sampling client builds a *shadow*
holder that reuses the trainer's session id, and with telemetry on it can emit
`SESSION_END`, which the gateway reads as the client exiting -- it then reclaims
the session's models and slots and the training state dies with the lease.
Production sets `TINKER_TELEMETRY=1`, so a worker inherits exactly the dangerous
value unless it fixes it.

### Member dispositions

A worker death is a **member** failure, decided by `min_successful_group_size`
like any other -- never a group failure and never a trainer failure.
Classification is **response-first**: a readable response wins regardless of the
return code or of who sent the signal, because a worker needs ~1.9 s of
interpreter teardown after it writes one, and its group's tail grace can expire
inside that window.

| disposition | condition | scalar |
| --- | --- | --- |
| `completed` / `no_trajectories` / `conversion_empty` / `error` | the worker's own verdict | (as today) |
| `straggler_killed` | cancelled by the group's tail grace | `straggler_cancelled` |
| `hard_timeout` | no response, `.diag` says `phase=startup` or `phase=rollout` | `rollout_worker_hard_timeout` |
| `rollout_worker_oom` | no response, `.diag` says `phase=rss`, or exit code 87 | `rollout_worker_oom` |
| `worker_crashed` | no response, no marker, non-zero or signalled exit | `rollout_worker_crashed` |
| `worker_launch_failed` | exec failed twice, or the response is unreadable | `rollout_worker_launch_failed` |

The `.diag` marker is what separates a hard timeout from an OOM kill from an
external `kill-leftovers.sh` sweep -- three failures with the identical
observable (`rc=-9`, no response) and three opposite operator responses. It also
carries the dying worker's cumulative cell-guard counters, which is the only way
`ptc_cells_stuck_at_exit` can ever be non-zero: the worker with the stuck cell is
exactly the one that gets SIGKILLed and writes no response.

### Where to look when something goes wrong

Each worker's stdout and stderr land in
`<log_path>/rollouts/<scope>/<version>/workers/<request_id>.log`, next to the
rollout artifacts, and survive the worker's own SIGKILL. The first line is
machine-readable:

```
[rollout-worker] request=<id> pid=<pid> pgid=<pgid> extra_allow=1 telemetry=0 torch_threads=1 startup=24.9s
```

`extra_allow=1` is the R3 patch (without it the gateway's `routing_id` is
silently dropped and every datum loses `routing_handle`); `telemetry=0` is the
session foot-gun above; `torch_threads=1` is the thread pin, without which each
worker takes `os.cpu_count()` intra- *and* inter-op threads on the node that
also hosts the gateway for every node.

For a single rollout you can `tail -f`, run
`--train.workflow_config.group_size 1 --train.num_concurrent_rollout_workflow_workers 1`.
To restore today's behaviour exactly, set
`--train.workflow_config.rollout_isolation inprocess`.

## Client-side throughput metrics

A step on this path is rollout-bound while the serving engines sit nearly idle,
and nothing measured the client, so "the client is the bottleneck" could only be
inferred. These scalars are logged every training step through the stats logger
and therefore reach W&B. Each answers one question.

| metric | question it answers |
| --- | --- |
| `rollout_client/rollouts_in_flight_mean` / `_max` / `_configured` | Are we actually running as many rollouts at once as we asked for? A mean far below `_configured` means rollouts are not overlapping. |
| `rollout_client/concurrency_achieved_fraction` | The same in one number: achieved / configured. |
| `rollout_client/groups_in_flight_mean` / `_max` | The same at task granularity, so group-atomic admission is visible. |
| `rollout_client/trainer_queue_wait_seconds_total` / `_mean` / `_max` | How long the training loop blocked on the rollout queue this step. Close to the step wall clock = rollout-bound; near zero = training-bound. |
| `rollout_client/trainer_queue_ready_fraction` | Fraction of dequeues satisfied immediately. 1.0 means the producer is comfortably ahead. |
| `rollout_client/agent_step_seconds_p50` / `_p90` | How long one agent step takes end to end (previous model-call end to this one's end). |
| `rollout_client/agent_step_model_seconds_p50` / `_p90` | Of that, the gateway round trip (`sample_async`). |
| `rollout_client/agent_step_client_seconds_p50` / `_p90` | Of that, time inside the client rendering/tokenizing the prompt and parsing/detokenizing the response -- the work that serialises on the GIL. A large or concurrency-growing value **is** the client bottleneck. |
| `rollout_client/agent_step_tool_seconds_p50` / `_p90` | Of that, tool execution, environment round trips and agent scaffolding. |
| `rollout_client/agent_step_model_fraction` / `_client_fraction` / `_tool_fraction` | The same split as shares of accounted time: one number for where the episode went. |
| `rollout_client/model_calls_per_rollout_mean` | The multiplier on every per-call cost. |
| `rollout_client/cpu_percent` | Trainer-process CPU over the step (100 = one core busy). |
| `rollout_client/worker_cpu_percent` | CPU of reaped children over the step (`RUSAGE_CHILDREN`); with process isolation the rollout CPU appears here. |
| `rollout_client/total_cpu_percent` / `cpu_cores_busy` / `cpu_saturation` | Trainer + workers; cores busy; and that as a fraction of the node's cores. Near 1.0 means the client is out of CPU. |
| `rollout_client/rss_bytes` / `threads` | Trainer RSS and OS thread count (the pool a runaway PTC cell consumes). |
| `rollout_client/worker_rss_bytes` / `worker_processes_live` | Live worker footprint at sample time. |
| `rollout_client/worker_peak_rss_bytes_sum` / `_max` | The workers' own self-sampled peaks for the rollouts consumed this step. |
| `rollout_client/dequeue_to_first_model_call_seconds_p50` / `_p90` | From a task leaving the trainer's queue to its first model call. In `process` mode this includes worker launch, so it is the startup cost as the schedule feels it. |
| `rollout_worker_launch_seconds` | Per member, `exec` to readiness. Watch the intra-group spread against `straggler_timeout_seconds`. |
| `rollout_worker_startup_seconds` | The child's own import and registration time, measured from process start. |
| `rollout_worker_wall_seconds` | Total supervised wall clock for one worker. |
| `rollout_worker_routing_id_coverage` | Fraction of model calls that came back with a gateway routing id. Below 1.0 is logged as a WARNING: a dropped routing id raises nothing anywhere, it just makes the gradient quietly worse. |
| `rollout_worker_hard_timeout` / `_crashed` / `_oom` / `_launch_failed` / `_launch_retried` / `_killed` / `_peak_rss_bytes` | One per member outcome; see the disposition table above. Alert on `rollout_worker_launch_failed` as a first-class signal. |
| `ptc_cells_started` / `ptc_cells_finished` / `ptc_cells_stuck_at_exit` / `ptc_threads_leaked_at_exit` / `ptc_worker_reports` | Cell-guard counters forwarded from workers. **Cumulative only**: summing the instantaneous gauges across short-lived workers would read zero precisely when cells are being killed for hanging. The instantaneous `ptc_cells_stuck` / `ptc_threads_leaked` / `ptc_cells_running` keep their original meaning in `inprocess` mode. |

## Run configs

`plugins/openreward/platoon/openreward/configs/tinker/`:

- `abci_toolathlon_15node_recursive_qwen36_miles_tinker_r3_v1.yaml`
- `abci_toolathlon_15node_nonrecursive_qwen36_miles_tinker_r3_v1.yaml`
  (delta: `openreward.enable_recursive_subagents`,
  `openreward.enable_programmatic_tool_calling` false; `stats.trial_name`,
  `stats.wandb.group`, last tag `nonrecursive`)

`load_config` drops unknown keys silently; `tests/openreward/test_tinker_toolathlon_configs.py`
walks both yamls and asserts every key maps to a dataclass field.

## Tests (openreward venv, CPU)

```
cd /home/aci18912it/source/platoon-tinker && HF_HUB_OFFLINE=1 plugins/openreward/.venv/bin/python -m pytest \
  tests/utils/test_tinker_data_processing_routing.py \
  tests/train/tinker/test_group_rollout_straggler.py \
  tests/openreward/test_tinker_toolathlon_configs.py \
  tests/test_openreward_stream_isolation.py \
  tests/test_openreward_ptc_cell_guard.py \
  tests/test_openreward_mcp_connect_retry.py -q
```
