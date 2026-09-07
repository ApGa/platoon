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
  `openreward.enable_programmatic_tool_calling`, i.e. the **recursive** yaml
  only) wraps each cell in `contextlib.redirect_stdout` / `redirect_stderr`,
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
  tests/test_openreward_mcp_connect_retry.py -q
```
