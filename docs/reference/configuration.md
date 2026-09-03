# Configuration

Platoon has three config trees, one per entrypoint family:

| Tree | Root class | Defined in |
|---|---|---|
| AReaL training | `PlatoonArealRLTrainerConfig` | <span class="pl-src">platoon/train/areal/config_defs.py</span> |
| Tinker training | `PlatoonTinkerRLTrainerConfig` | <span class="pl-src">platoon/train/tinker/config_defs.py</span> |
| Inference and evaluation | `InferenceBenchmarkConfig` | <span class="pl-src">platoon/inference/config_defs.py</span> |

All three share one class, `RolloutConfig`, the settings the agent loop reads.

## Override syntax

The two training paths use different loaders, so they take different override syntax.

=== "AReaL"

    OmegaConf, via `load_expr_config`. Overrides are bare `key=value` with **no leading dashes**.

    ```bash
    uv run python platoon/number_search/train.py \
      --config platoon/number_search/number_search_areal.yaml \
      trial_name=debug-run \
      train_dataset.batch_size=16
    ```

    `defaults:` composition and `${...}` interpolation both work, which is why configs write
    `tokenizer_path: ${actor.path}`. Unknown keys are a hard error. The fully resolved config is
    written to `config.yaml` under the run's log path.

=== "Tinker"

    A dataclass loader, `load_config` from `platoon.utils.config`. Overrides are
    `--dotted.key value` or `--dotted.key=value`.

    ```bash
    uv run python -m platoon.train.tinker.train \
      --config platoon/textcraft/configs/tinker/textcraft_synth_depth_aware_tinker.yaml \
      --train.batch_size 8 \
      --stats.trial_name debug-run
    ```

    No `defaults:` composition and no interpolation. A bare `--flag` with no value parses as `true`.
    Values are coerced by shape: booleans, `none`, int, float, comma-separated list, else string.

Inference entrypoints use the same loader as Tinker, so their overrides are `--dotted.key value`
too. See the [CLI reference](cli.md) for which entrypoint runs from where.

!!! note "Spelling matters on the Tinker and inference paths"
    That loader walks the dataclass fields, so a YAML key the dataclass does not declare is ignored
    and the field keeps its default. If a setting appears to have no effect, check the key name
    first.

## Keys you will change first

| Key | Where | What it does |
|---|---|---|
| `experiment_name`, `trial_name` | AReaL | Names the run; feeds checkpoint, recovery and W&B paths. |
| `actor.path` | AReaL | HF checkpoint or local directory for the policy. |
| `actor.backend`, `rollout.backend` | AReaL | Engine placement, e.g. `fsdp:d4p1t1` and `sglang:d4p1t1`. Both must be set. |
| `train.model_name`, `train.renderer_name` | Tinker | Model id and prompt renderer. Both required. |
| `train_dataset.batch_size` / `train.batch_size` | AReaL / Tinker | Tasks per optimizer batch. |
| `workflow_config.group_size` | Both | Rollouts per task in one group. |
| `workflow_config.rollout_config.max_steps` | Both | Agent step cap for the root episode. |
| `…rollout_config.inference_params.max_completion_tokens` | Both | Generation length. The default of 512 is short for agentic tasks. |
| `actor.optimizer.lr` / `train.optimizer.learning_rate` | AReaL / Tinker | Learning rate. |
| `loss_fn_config.loss_fn` / `train.loss_fn` | AReaL / Tinker | Policy loss. |
| `cluster.fileroot` / `log_path` | AReaL / Tinker | Where the run writes. |

## Rollout config

`RolloutConfig` in <span class="pl-src">platoon/config_defs.py</span>. It sits at a different path in
each tree:

| Tree | Path |
|---|---|
| AReaL | `workflow_config.rollout_config` |
| Tinker | `train.workflow_config.rollout_config`, `eval.workflow_config.rollout_config` |
| Inference | `inference.workflow.rollout_config` |

| Key | Type | Default | What it does |
|---|---|---|---|
| `model_name` | `str \| None` | `None` | Model id sent to the OpenAI-compatible endpoint. AReaL configs set `${actor.path}`. |
| `model_endpoint` | `str \| None` | `None` | Base URL of the inference endpoint. |
| `model_api_key` | `str \| None` | `None` | API key for that endpoint. |
| `max_steps` | `int \| None` | `None` | Step cap for the root episode. `None` imposes no cap. |
| `timeout` | `int \| None` | `None` | Whole-trajectory timeout, in seconds. |
| `step_timeout` | `int` | `300` | Per-step timeout covering `agent.act` plus `env.step`. |
| `output_dir` | `str` | `"rollout_results"` | Where the trajectory collection is written. |
| `propagate_root_success` | `bool \| None` | `None`, resolved to `False` | Train every trajectory in the tree toward the root task's outcome. |
| `skip_subagent_reward_computation` | `bool` | `False` | Skip per-subagent reward computation. |
| `verbose` | `bool` | `True` | Per-step logging during the rollout. |
| `inference_params` | `InferenceParams` | see below | Sampling settings. |

On the inference path, `model_name`, `model_endpoint`, `model_api_key` and `output_dir` are filled
in from the `inference:` block and from the task being run, so setting them under
`inference.workflow.rollout_config` has no effect.

### `inference_params:`

| Key | Type | Default | What it does |
|---|---|---|---|
| `temperature` | `float \| None` | `1.0` | Sampling temperature. |
| `top_p` | `float \| None` | `1.0` | Nucleus sampling cutoff. |
| `max_completion_tokens` | `int` | `512` | Per-call generation budget. Raise it for agentic tasks. |

## Workflow config

`WorkflowConfig` in <span class="pl-src">platoon/train/areal/config_defs.py</span> governs how a
group of rollouts is collected and turned into training data. The mechanics are in
[execution](../architecture/execution.md).

| Key | Type | Default | What it does |
|---|---|---|---|
| `group_size` | `int` | `1` | Rollouts per task in one group. Typical values are 4 or 8. |
| `rollout_config` | `RolloutConfig` | see above | Per-rollout settings. |
| `use_subprocesses` | `bool` | `False` | Run each group member in an isolated subprocess. |
| `straggler_timeout_seconds` | `float \| None` | `None` | Once quorum is reached, wait at most this long for the tail before reaping the group's process pool. |
| `straggler_quorum` | `int \| None` | `None` | Settled members that start the tail clock. `None` means `group_size - 1`. Requires `straggler_timeout_seconds`. |
| `subprocess_shutdown_grace_seconds` | `float` | `5.0` | Grace period before the group pool is killed. |
| `min_successful_group_size` | `int` | `1` | Reject and replenish a group returning fewer usable members. Must be in `[1, group_size]`. |
| `leave_one_out_baseline` | `bool` | `False` | Leave-one-out advantage centering instead of group-mean centering. |
| `depth_level_weighting` | `bool` | `False` | Weight trajectories inversely by how often their tree depth appears in the batch. |
| `depth_level_discount_gamma` | `float \| None` | `None` | The alternative: discount rewards by `gamma^depth`. |
| `subagent_datum_keep_probability` | `float` | `1.0` | Keep every root datum, then Bernoulli-sample each subagent datum. Must be in `[0, 1]`. |
| `subagent_datum_sampling_seed` | `int` | `0` | Seed for that sampling. |
| `filter_zero_advantage_datums` | `bool` | `True` | Drop datums whose centered scalar reward is exactly zero. |
| `filter_zero_variance_groups` | `bool` | `True` | Reject groups whose members all received the same reward. |
| `token_efficiency_reward` | `TokenEfficiencyRewardConfig` | see below | Token-cost penalty attributed to a policy subtree. |

!!! warning "`filter_zero_advantage_datums` assumes a reward-only objective"
    It uses the centered scalar reward as a proxy for the final advantage. Turn it off when anything
    else can turn a zero reward into a nonzero objective: a nonzero `actor.kl_ctl`, reward or
    advantage normalization, a reward bias, a critic or distillation term, an auxiliary router loss,
    or a custom transform that adds to rewards. The trainer warns at startup when it detects one.

### `token_efficiency_reward:`

The workflow annotates each policy trajectory with a penalty; a reward processor that reads the
annotation is what applies it, and [OpenReward](../plugins/openreward.md)'s does. The subtraction
happens before group centering, so what survives centering is relative efficiency.

```text
effective = output_token_weight * output_tokens + input_token_weight * logical_input_tokens
penalty   = min(max_penalty, coefficient * log2(1 + effective / reference_tokens))
```

| Key | Type | Default | What it does |
|---|---|---|---|
| `enabled` | `bool` | `False` | Must be a real boolean. |
| `coefficient` | `float` | `0.05` | Penalty slope. |
| `reference_tokens` | `float` | `20000.0` | Token count at which the log term equals 1. |
| `max_penalty` | `float` | `0.20` | Cap on the penalty. |
| `input_token_weight` | `float` | `0.01` | Discount for logical input tokens. |
| `output_token_weight` | `float` | `1.0` | Weight on generated tokens. |

## Loss config

=== "AReaL"

    | Key | Type | Default | What it does |
    |---|---|---|---|
    | `loss_fn_config.loss_fn` | `str` | `"grpo"` | Registered loss name. `grpo`, `ppo` and `cispo` ship in <span class="pl-src">platoon/train/areal/loss_functions.py</span>. |
    | `loss_fn_config.loss_fn_kwargs` | `dict` | `{}` | Loss-specific kwargs, overriding the registered loss's own defaults. CISPO registers `clip_low_threshold=0.0` and `clip_high_threshold=5.0`. |

    Kwargs go inside `loss_fn_kwargs`; putting them at the top of the block is rejected.
    `actor.loss_fn` is overwritten from this block, so the loss has exactly one home. Registering
    your own is covered in [extend](../guides/extend.md).

=== "Tinker"

    | Key | Type | Default | What it does |
    |---|---|---|---|
    | `train.loss_fn` | `str` | `"cispo"` | Passed straight to the Tinker API. These are not Platoon's AReaL loss names. |
    | `train.loss_fn_config` | `dict` | `{"clip_low_threshold": 0.0, "clip_high_threshold": 5.0}` | Passed straight through, and read locally for clip-fraction metrics. A YAML block replaces this dict rather than merging into it. |

## Dataset config (AReaL)

Platoon narrows AReaL's dataset config to dataloader settings only. The dataset itself comes from
`environments[]` or from a plugin train script, never from a path here.

| Key | Type | Default | What it does |
|---|---|---|---|
| `train_dataset.batch_size` | `int` | `1` | Tasks per training batch. Set it explicitly. |
| `train_dataset.shuffle` | `bool` | `True` | Shuffle the train split each epoch. |
| `train_dataset.num_workers` | `int` | `0` | Dataloader workers. |
| `train_dataset.drop_last` | `bool` | `True` | Drop the trailing partial batch of each epoch. |
| `valid_dataset.batch_size` | `int` | `1` | Tasks per validation batch. |
| `valid_dataset.shuffle` | `bool` | `False` | Shuffle the validation split. |
| `valid_dataset.num_workers` | `int` | `0` | Dataloader workers. |
| `valid_dataset.drop_last` | `bool` | `False` | Drop the trailing partial validation batch. |

`valid_dataset: null` skips validation-dataset construction entirely.

!!! warning "Sampling is not configured under `gconfig:`"
    Platoon replaces AReaL's generation config with a one-field class, `gconfig.lora_name` (default
    `"default_lora"`). Temperature and generation length live in
    `workflow_config.rollout_config.inference_params`; rollouts per task live in
    `workflow_config.group_size`.

## Batch composition (AReaL)

Two fields Platoon adds to AReaL's `rollout:` block, in
<span class="pl-src">platoon/utils/train.py</span>.

| Key | Type | Default | What it does |
|---|---|---|---|
| `rollout.shuffle_cross_task` | `bool` | `False` | Shuffle datums across tasks before dispatch. |
| `rollout.ensure_batch_divisible_by` | `int` | `1` | Trim the accepted batch so its size divides by `lcm(dp_size, this)`, preferring non-root datums. |

## `environments:`

The registry wiring list, `EnvironmentConfig` in
<span class="pl-src">platoon/train/components.py</span>, shared by both trainer configs. Each string
is either a registry name or a dotted import path. Provide exactly one entry, as a list.

| Key | Type | Default | What it does |
|---|---|---|---|
| `package` | `str \| None` | `None` | Dotted module imported for its registration side effects. |
| `discover_entry_points` | `bool` | `False` | Also load every `platoon.plugins` entry point. |
| `dataset_loader` | `str \| None` | `None` | Required for the train split. |
| `eval_dataset_loader` | `str \| None` | `None` | Falls back to `dataset_loader`. |
| `task_loader` | `str \| None` | `None` | Required. Builds a task from a dataset row. |
| `rollout` | `str \| None` | `None` | Required for the train split. The rollout program to run. |
| `eval_rollout` | `str \| None` | `None` | Falls back to `rollout`. |
| `reward_processor` | `str \| None` | `None` | Unset means the trajectory's own reward is used as-is. |
| `workflow` | `str` | `"group_rollout"` | `"group_rollout"` means the backend's default workflow class; any other value is looked up in the workflow registry. |
| `dataset_kwargs` | `dict` | `{}` | Extra kwargs for the train dataset loader. |
| `eval_dataset_kwargs` | `dict` | `{}` | Extra kwargs for the eval dataset loader. |
| `workflow_kwargs` | `dict` | `{}` | Extra kwargs for the train workflow constructor, including `filter_errors` (`True` here). |
| `eval_workflow_kwargs` | `dict` | `{}` | Same, for evaluation, where `filter_errors` defaults to `False`. |

These names are what a plugin registers, whether it lives in this repository or in your own package.
See [components](../architecture/components.md) and [your first plugin](../guides/first-plugin.md).

!!! note "Not OpenReward's `environments:`"
    This top-level list selects Platoon components by name. The OpenReward plugin has its own nested
    `openreward.environments:` list with fields like `env_name` and `sampling_weight` — a task
    mixture, a different thing sharing a key name. See [OpenReward](../plugins/openreward.md).

## Tinker trainer config

`train`, `eval` and `log_path` are required; everything else has a default.

| Key | Type | Default | What it does |
|---|---|---|---|
| `log_path` | `str` | **required** | Base log directory. The run path is `log_path/stats.experiment_name/stats.trial_name`. |
| `tinker_base_url` | `str \| None` | `None` | Service URL of the Tinker-compatible backend. |
| `environments` | `list[EnvironmentConfig]` | one default entry | Same class as above. |

### `train:`

| Key | Type | Default | What it does |
|---|---|---|---|
| `model_name` | `str` | **required** | HuggingFace model identifier. |
| `renderer_name` | `str` | **required** | Prompt renderer, e.g. `qwen3`, `qwen3_instruct`. |
| `renderer_kwargs` | `dict` | `{}` | Renderer attribute overrides. |
| `context_window_length` | `int \| None` | `None` | Passed to the renderer. |
| `batch_size` | `int` | `32` | Tasks per optimizer batch, and the default rollout worker count. |
| `num_epochs` | `int \| None` | `None` | Training length in epochs. Set this, `max_training_steps`, or both. |
| `max_training_steps` | `int \| None` | `None` | Training length in optimizer steps. |
| `num_minibatches` | `int` | `1` | Weight updates per batch. Must divide `batch_size`. |
| `num_microbatches` | `int` | `1` | Gradient-accumulation splits per minibatch. Must divide the minibatch. |
| `max_staleness` | `int \| None` | `None` | Maximum off-policy lag, in optimizer steps. |
| `lora_rank` | `int` | `32` | Rank of the LoRA training client. |
| `optimizer.learning_rate` | `float` | `3e-5` | Adam learning rate. |
| `optimizer.beta1`, `optimizer.beta2` | `float` | `0.9`, `0.95` | Adam betas. |
| `optimizer.weight_decay` | `float` | `0.0` | Weight decay. |
| `optimizer.grad_clip_norm` | `float` | `0.0` | When `> 0`, the backend also returns `grad_norm` in its metrics. |
| `num_concurrent_rollout_workflow_workers` | `int \| None` | `None`, filled with `batch_size` | Concurrent rollouts in flight. |

### `train.workflow_config:` / `eval.workflow_config:`

The Tinker tree has its own, smaller `WorkflowConfig`: `group_size` (default **8**),
`rollout_config`, `leave_one_out_baseline` (`False`), `depth_level_weighting` (`False`),
`subagent_datum_keep_probability` (`1.0`), `subagent_datum_sampling_seed` (`0`), `filter_errors`
(`True`) and `filter_zero_advantage_datums` (`True`). The AReaL-only keys above — subprocesses,
straggler handling, token-efficiency reward — do not exist here.

`eval.workflow_config` defaults to `group_size=1`, `filter_errors=False` and
`filter_zero_advantage_datums=False`. Those come from a factory, so if you write an
`eval.workflow_config:` block at all, write all three keys.

### `eval:`, `checkpoint:`, `stats:`, `watchdog:`

| Key | Type | Default | What it does |
|---|---|---|---|
| `eval.strategy` | `"epoch" \| "step" \| "none"` | `"epoch"` | When evaluation runs. |
| `eval.every` | `int` | `1` | Interval, in epochs or steps. |
| `eval.num_concurrent_rollout_workflow_workers` | `int` | `256` | Concurrent eval rollouts. |
| `checkpoint.strategy` | `"epoch" \| "step" \| "none"` | `"epoch"` | When checkpoints are saved. |
| `checkpoint.every` | `int` | `1` | Interval, in epochs or steps. |
| `checkpoint.load_checkpoint_path` | `str \| None` | `None` | Weights to load when auto-resume finds nothing. |
| `stats.experiment_name` | `str` | `"platoon_tinker"` | Run family name. |
| `stats.trial_name` | `str` | `"run"` | Run name within the family. |
| `stats.wandb.mode` | `str` | `"online"` | `online`, `offline` or `disabled`. |
| `stats.wandb.project`, `entity`, `group`, `name`, `notes` | `str \| None` | `None` | Standard W&B fields. |
| `stats.wandb.tags` | `list[str]` | `[]` | W&B tags. |
| `watchdog.enabled` | `bool` | `True` | Hard-exits the process if the backend stops responding. |
| `watchdog.timeout_seconds` | `float` | `600` | Long agentic rollouts want 3600 or more. |
| `watchdog.exit_code` | `int` | `2` | Exit code used when the watchdog fires. |

## Inference config

Used by the evaluation and benchmarking entrypoints, under `inference:`. Plugins add their own
top-level keys alongside it. See [evaluate](../guides/evaluate.md).

| Key | Type | Default | What it does |
|---|---|---|---|
| `inference.model_name` | `str` | **required** | Model id in OpenAI form. |
| `inference.model_endpoint` | `str \| None` | `None` | Base URL of the served model. |
| `inference.model_api_key` | `str \| None` | `None` | API key for that endpoint. |
| `inference.output_dir` | `str` | `"inference_results"` | Rollouts and reports are written underneath. |
| `inference.resume` | `bool` | `True` | Reuse existing per-task artifacts instead of re-running them. |
| `inference.workflow.num_rollouts_per_task` | `int` | `1` | pass@k sampling width. |
| `inference.workflow.num_concurrent_workers` | `int` | `32` | Concurrent rollouts. |
| `inference.workflow.success_threshold` | `float` | `1.0` | Success cutoff, used only when a trajectory carries no explicit success metric. |
| `inference.workflow.fail_fast` | `bool` | `False` | Stop the run on the first failure. |
| `inference.workflow.use_subprocesses` | `bool` | `False` | Run each rollout in a process pool. |
| `inference.workflow.rollout_config` | `RolloutConfig` | see above | Per-rollout settings. |

## Upstream AReaL keys

The rest of an AReaL YAML belongs to upstream AReaL, not to Platoon: `cluster`, `scheduler`,
`sglang`, `saver`, `evaluator`, `recover`, `stats_logger`, and most of `actor:`, `ref:` and
`rollout:` — optimizer, clipping, normalization, LoRA, and the Megatron and FSDP blocks. Their
meanings and defaults track the AReaL revision pinned in `pyproject.toml` — see
[AReaL's `cli_args.py`](https://github.com/inclusionAI/AReaL/blob/main/areal/api/cli_args.py).

Two carry Platoon-specific behavior. `scheduler.type` is set to `"local"` when you leave it unset,
and also accepts `"slurm_prealloc"` for runs launched into a pre-allocated Slurm allocation.
`rollout.agent.tool_call_parser` must match the served model's tool-call format, or the harness sees
no tool calls; [OpenHands](../plugins/openhands.md) covers the parser choice. Backend selection
itself is covered in [backends](../architecture/backends.md).
