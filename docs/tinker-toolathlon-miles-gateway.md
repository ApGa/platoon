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
