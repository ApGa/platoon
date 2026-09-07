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
full-parameter `save_state` takes minutes. `watchdog.timeout_seconds` is 3600
in the run configs.

`platoon/train/tinker/restart_wrapper.py` (pre-existing) restarts the trainer
on the watchdog exit code: `python -m platoon.train.tinker.restart_wrapper
--max-restarts 20 --restart-delay 700 --watchdog-exit-code 2 -- <trainer cmd>`.

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
  tests/openreward/test_tinker_toolathlon_configs.py -q
```
