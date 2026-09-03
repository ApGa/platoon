# Configuration reference

Every config key Platoon defines, with the default read out of the dataclass that declares it. Keys
Platoon inherits from upstream AReaL are listed separately and marked as passthrough — this page
does not restate upstream defaults it cannot pin. For how the config tree is assembled and why it is
shaped this way, see [config architecture](../architecture/config.md).

Platoon has three independent config trees:

| Tree | Root class | Defined in |
|---|---|---|
| AReaL training | `PlatoonArealRLTrainerConfig` | <span class="pl-src">platoon/train/areal/config_defs.py</span> |
| Tinker training | `PlatoonTinkerRLTrainerConfig` | <span class="pl-src">platoon/train/tinker/config_defs.py</span> |
| Inference benchmarking | `InferenceBenchmarkConfig`, under `inference:` | <span class="pl-src">platoon/inference/config_defs.py</span> |

They share exactly one class: `RolloutConfig` in <span class="pl-src">platoon/config_defs.py</span>.

## Two loaders, two override syntaxes

Getting this backwards is the fastest way to waste an afternoon. The AReaL path uses OmegaConf; the
Tinker and inference paths use a hand-rolled argparse loader.

=== "AReaL"

    `load_expr_config` from `areal.api.cli_args`. Overrides are bare `key=value` with **no leading
    dashes**.

    ```bash
    cd plugins/number-search
    uv run python3 platoon/number_search/train.py \
      --config platoon/number_search/number_search_areal.yaml \
      trial_name=debug-run \
      train_dataset.batch_size=16
    ```

    That is the per-plugin script rather than `python -m platoon.train.areal.train`, because no
    AReaL config in the repository fills in the top-level `environments:` block the shared
    entrypoint reads — it would fail with `Config must set environments[0].dataset_loader`. The
    override syntax is the same either way; both call `load_expr_config`.

    - `defaults:` composition works, pulling sibling YAMLs from the same directory.
    - `${...}` interpolation works, which is why every config writes `tokenizer_path: ${actor.path}`.
    - Unknown keys are a **hard error** — the structured-config merge rejects them.
    - The fully resolved config is written to `config.yaml` under the stats-logger log path.

=== "Tinker"

    `load_config` from `platoon.utils.config`. Overrides are `--dotted.key value` or
    `--dotted.key=value`.

    ```bash
    cd plugins/textcraft
    uv run python -m platoon.train.tinker.train \
      --config platoon/textcraft/configs/tinker/textcraft_synth_depth_aware_tinker.yaml \
      --train.batch_size 8 \
      --stats.trial_name debug-run
    ```

    - No `defaults:` composition and no `${...}` interpolation.
    - Unknown keys are **silently dropped** (see the warning below).
    - A bare `--flag` with no following value parses as `true`.
    - `_parse_value` coerces in order: `true/yes/1` and `false/no/0` to bool, `none/null` to `None`,
      then int, then float, then a comma-containing string to a list, else a plain string.

Inference entrypoints use the same `load_config` as Tinker, so inference overrides are
`--dotted.key value` too.

Both examples start with a `cd` into the plugin. `uv run` resolves the venv of the project it is
invoked in, and only a plugin's own venv has that plugin importable; the root project installs the
`platoon` core alone. The [CLI reference](cli.md) has the full table of what runs from where.

!!! warning "Typos are silent on the Tinker and inference paths"
    `_dataclass_from_dict` in <span class="pl-src">platoon/utils/config.py</span> iterates over the
    dataclass fields, never over the YAML dict. A key the dataclass does not declare — a
    misspelling, a key copied from the wrong config tree, a stale key from an older revision — is
    dropped without a warning and the field keeps its default. When a Tinker setting appears to have
    no effect, check the spelling before you check the code.

## The keys you will actually change first

| Key | Where | What it does |
|---|---|---|
| `experiment_name`, `trial_name` | AReaL | Names the run. Feed checkpoint, recovery and W&B paths; changing `trial_name` starts a fresh recovery lineage. Upstream fields, required in practice. |
| `actor.path` | AReaL | HF checkpoint or local directory for the policy. |
| `actor.backend`, `rollout.backend` | AReaL | Engine placement strings such as `fsdp:d4p1t1` and `sglang:d4p1t1`. Platoon raises if either is empty. |
| `train.model_name`, `train.renderer_name` | Tinker | Model id and prompt renderer. Both required, no defaults. |
| `train_dataset.batch_size` / `train.batch_size` | AReaL / Tinker | Tasks per optimizer batch. |
| `workflow_config.group_size` | Both | Rollouts per task in one GRPO group. |
| `workflow_config.rollout_config.max_steps` | Both | Agent step cap for the root episode. |
| `workflow_config.rollout_config.inference_params.max_completion_tokens` | Both | Generation length. The default of 512 is short for agentic tasks. |
| `actor.optimizer.lr` / `train.optimizer.learning_rate` | AReaL / Tinker | Learning rate. |
| `loss_fn_config.loss_fn` / `train.loss_fn` | AReaL / Tinker | Policy loss. See [algorithm recipes](../recipes/algorithms.md). |
| `cluster.fileroot`, `cluster.n_nodes` / `log_path` | AReaL / Tinker | Where the run writes, and how much hardware it asks for. |

---

## Shared: `RolloutConfig`

`RolloutConfig` in <span class="pl-src">platoon/config_defs.py</span> is the per-rollout knob set the
agent loop actually reads. It sits at a different path in each tree:

| Tree | Path |
|---|---|
| AReaL | `workflow_config.rollout_config` |
| Tinker | `train.workflow_config.rollout_config` and `eval.workflow_config.rollout_config` |
| Inference | `inference.workflow.rollout_config` |

| Key | Type | Default | What it does |
|---|---|---|---|
| `model_name` | `str \| None` | `None` | Model id sent to the OpenAI-compatible endpoint. AReaL configs set `${actor.path}`. Overwritten from `inference.model_name` on the inference path. |
| `model_endpoint` | `str \| None` | `None` | Base URL. Overwritten on the inference path. |
| `model_api_key` | `str \| None` | `None` | Overwritten on the inference path. |
| `train` | `bool` | `False` | Training-mode rollout. Both `GroupRolloutWorkflow` constructors force it to `True`; the inference workflow forces it back to `False`. Setting it in YAML is decorative. |
| `max_steps` | `int \| None` | `None` | Step cap for the **root** episode. `None` imposes no cap from this setting. |
| `output_dir` | `str` | `"rollout_results"` | Where the trajectory collection is written. Replaced per task on the inference path, so the YAML value there is decorative. |
| `timeout` | `int \| None` | `None` | Whole-trajectory timeout, in seconds. |
| `step_timeout` | `int` | `300` | Per-step timeout covering `agent.act` plus `env.step`. Propagated into the episode-loop context vars. |
| `return_dict` | `bool` | `False` | Return the collection instead of writing it to disk. Both training workflows and the inference workflow force `True`. |
| `propagate_root_success` | `bool \| None` | `None`, resolved to `False` | Train every trajectory in the tree toward the root task outcome. |
| `propogate_root_success` | `bool \| None` | `None` | **Deprecated misspelling.** See below. |
| `skip_subagent_reward_computation` | `bool` | `False` | Skip per-subagent reward computation. |
| `inference_params` | `InferenceParams` | see below | Sampling settings. Coerced from a plain dict so subprocess and pickle round-trips work. |
| `extra` | `dict[str, Any]` | `{}` | Free-form bag carried to remote and subprocess rollout workers. The OpenReward plugin ships its whole plugin config through `extra["openreward"]`. Not set directly in any YAML. |

`verbose` (`bool`, default `True`) also exists and does what its name says.

### The `propogate_root_success` misspelling

Both spellings are real fields. `__post_init__` on `RolloutConfig` reconciles them:

- If only the deprecated key is set, its value wins.
- If both are set to **conflicting** values, it raises
  `"Conflicting rollout propagation settings"`.
- The canonical field always ends up a concrete boolean, never `None`.

Write `propagate_root_success`. The misspelled key exists only so pre-correction YAMLs still load,
and nothing in the current configs uses it.

### `InferenceParams`

| Key | Type | Default | What it does |
|---|---|---|---|
| `temperature` | `float \| None` | `1.0` | |
| `top_p` | `float \| None` | `1.0` | |
| `max_completion_tokens` | `int` | `512` | Per-call generation budget. Deliberately conservative; agentic configs raise it. |

---

## AReaL trainer config

`PlatoonArealRLTrainerConfig` subclasses upstream `GRPOConfig` and narrows or replaces ten fields.
Everything else in an AReaL YAML is inherited and covered under
[upstream passthrough](#upstream-areal-passthrough).

```python title="platoon/train/areal/config_defs.py"
@dataclass
class PlatoonArealRLTrainerConfig(GRPOConfig):
    gconfig: PlatoonGenerationConfig = field(default_factory=PlatoonGenerationConfig)
    eval_gconfig: PlatoonGenerationConfig | None = None
    train_dataset: PlatoonTrainDatasetConfig = field(default_factory=PlatoonTrainDatasetConfig)
    valid_dataset: PlatoonValidDatasetConfig | None = field(default_factory=PlatoonValidDatasetConfig)
    workflow_config: WorkflowConfig = field(default_factory=WorkflowConfig)
    rollout: VariableBatchInferenceEngineConfig = field(default_factory=VariableBatchInferenceEngineConfig)
    actor: PlatoonPPOActorConfig = field(default_factory=PlatoonPPOActorConfig)
    ref: PlatoonPPOActorConfig | None = None
    loss_fn_config: LossFnConfig = field(default_factory=LossFnConfig)
    environments: list[EnvironmentConfig] = field(default_factory=lambda: [EnvironmentConfig()])
```

A config that *parses* needs two keys, `rollout.backend` and `actor.backend`; everything else comes
from dataclass defaults. A config that *runs* also needs `experiment_name`, `trial_name`,
`actor.path`, and a `cluster.fileroot` visible from every node.

### `gconfig:` / `eval_gconfig:`

Platoon replaces AReaL's entire `GenerationHyperparameters` with a one-field class.

| Key | Type | Default | What it does |
|---|---|---|---|
| `gconfig.lora_name` | `str` | `"default_lora"` | Minimal generation config still required by upstream AReaL internals. |
| `eval_gconfig` | `PlatoonGenerationConfig \| None` | `None` | Filled from `gconfig.new()` in `__post_init__` when unset. |

!!! warning "Generation is not configured here"
    `gconfig.n_samples`, `gconfig.max_new_tokens` and `gconfig.temperature` do not exist on this
    path, and setting them is a hard config error. Sampling lives in
    `workflow_config.rollout_config.inference_params`; the number of rollouts per task lives in
    `workflow_config.group_size`.

### `train_dataset:` / `valid_dataset:`

Platoon replaces AReaL's `_DatasetConfig` with minimal, dataloader-only classes.
`train_dataset.path`, `train_dataset.type` and `valid_dataset.path` are rejected — the dataset comes
from `environments[]` or from a plugin train script, never from a path in this block.

| Key | Type | Default | What it does |
|---|---|---|---|
| `train_dataset.batch_size` | `int` | `1` | Tasks per training batch. Every real config sets it, and `rollout.consumer_batch_size` interpolates from it. |
| `train_dataset.shuffle` | `bool` | `True` | |
| `train_dataset.num_workers` | `int` | `0` | |
| `train_dataset.drop_last` | `bool` | `True` | Drops the trailing partial batch of each epoch. |
| `valid_dataset.batch_size` | `int` | `1` | |
| `valid_dataset.shuffle` | `bool` | `False` | |
| `valid_dataset.num_workers` | `int` | `0` | |
| `valid_dataset.drop_last` | `bool` | `False` | |

`valid_dataset` is optional. Writing `valid_dataset: null` makes plugin train scripts skip
validation-dataset construction entirely.

### `workflow_config:`

`WorkflowConfig` in <span class="pl-src">platoon/train/areal/config_defs.py</span>, Platoon's own
class with no upstream counterpart. The mechanics it controls are explained in the
[group rollout workflow walkthrough](../walkthroughs/group-rollout-workflow.md).

| Key | Type | Default | What it does |
|---|---|---|---|
| `group_size` | `int` | `1` | Rollouts per task in one group. Must be `>= 1`. Real configs use 4 or 8. |
| `rollout_config` | `RolloutConfig` | see above | Coerced from a plain dict in `__post_init__`. |
| `use_subprocesses` | `bool` | `False` | Run each group member in an isolated subprocess. Every config that mentions it sets `true`. |
| `straggler_timeout_seconds` | `float \| None` | `None` | Once quorum is reached, wait at most this long for the tail before reaping the group's process pool. `None` gives every member the full rollout timeout. Must be positive when set. |
| `straggler_quorum` | `int \| None` | `None` | Number of *settled* members — completed, interrupted, or failed-closed — that starts the tail clock. `None` means `group_size - 1`. Must be in `[1, group_size]`, and **requires** `straggler_timeout_seconds`. |
| `subprocess_shutdown_grace_seconds` | `float` | `5.0` | Grace period before the group pool is killed. Must be non-negative. |
| `min_successful_group_size` | `int` | `1` | Reject and replenish a group that returns fewer usable members. Must be in `[1, group_size]`. Recursive runs use 4 against a group size of 8. |
| `leave_one_out_baseline` | `bool` | `False` | Leave-one-out advantage centering instead of plain group-mean centering. |
| `depth_level_weighting` | `bool` | `False` | Trainer-side full-batch inverse-frequency weighting by tree depth. |
| `depth_level_discount_gamma` | `float \| None` | `None` | The alternative to weighting: discount rewards by `gamma^depth`. |
| `subagent_datum_keep_probability` | `float` | `1.0` | Keep every root datum, then independently Bernoulli-sample each post-merge subagent datum. `1.0` reproduces the historical batch exactly. Must be in `[0, 1]`. |
| `subagent_datum_sampling_seed` | `int` | `0` | Seed for that sampling. Must be a non-bool int. Configs set `${seed}`. |
| `filter_errors` | `bool` | `True` | Defer typed action errors until group centering, then suppress only the error tokens that would otherwise receive positive policy credit. **Neither workflow reads this field**; see the warning below. |
| `token_efficiency_reward` | `TokenEfficiencyRewardConfig` | see below | Token-cost penalty attributed to a policy subtree. Coerced from a dict. |
| `filter_zero_advantage_datums` | `bool` | `True` | Throughput fast path that drops datums whose centered scalar reward is exactly zero. **Read the safety note below.** |
| `filter_zero_variance_groups` | `bool` | `True` | Reject groups whose members all received the same reward. |
| `enable_router_replay` | `bool` | `False` | **Do not set.** Overwritten from `actor.enable_router_replay`. |
| `router_replay_num_layers` | `int \| None` | `None` | **Do not set.** Copied from `actor.router_replay_num_layers`. |
| `router_replay_topk` | `int \| None` | `None` | **Do not set.** Copied from `actor.router_replay_topk`. |

The generic AReaL entrypoint deep-copies `workflow_config` for evaluation and overrides three
fields: `group_size = 1`, `subagent_datum_keep_probability = 1.0`,
`filter_zero_advantage_datums = False`. Evaluation always retains the complete trajectory tree, so
tuning those for eval in YAML does nothing.

!!! warning "`workflow_config.filter_errors` is not the value that takes effect"
    Both `WorkflowConfig` classes declare `filter_errors`, and neither `GroupRolloutWorkflow` reads
    it. The effective value is the workflow's own constructor argument, which defaults to `False`
    and which the shared entrypoints supply from `workflow_kwargs.pop("filter_errors", True)` for
    training and `eval_workflow_kwargs.pop("filter_errors", False)` for evaluation. Set it through
    `environments[0].workflow_kwargs: {filter_errors: false}`, not through `workflow_config`.
    OpenReward's own `train_areal.py` and `train_tinker.py` forward the config field explicitly
    (`config.workflow_config.filter_errors` and `config.train.workflow_config.filter_errors`),
    which is a property of those scripts and not of the field.

### `workflow_config.token_efficiency_reward:`

`TokenEfficiencyRewardConfig`. The workflow computes the penalty and writes it into each policy
trajectory's `misc`; nothing in the pipeline subtracts it. Only a reward processor that reads the
annotation changes the reward, and OpenReward's is the one that does. That subtraction happens
inside the reward processor, so the penalty enters the reward *before* group centering — every
member pays its own, and what survives centering is relative efficiency. See
[custom rewards](../customization/rewards.md).

```
effective = output_token_weight * output_tokens + input_token_weight * logical_input_tokens
penalty   = min(max_penalty, coefficient * log2(1 + effective / reference_tokens))
```

| Key | Type | Default | What it does |
|---|---|---|---|
| `enabled` | `bool` | `False` | Must be an actual bool, not a truthy value. |
| `coefficient` | `float` | `0.05` | Penalty slope. Finite, non-negative. |
| `reference_tokens` | `float` | `20000.0` | Token count at which the log term equals 1. Finite, strictly positive. |
| `max_penalty` | `float` | `0.20` | Cap on the penalty. Finite, non-negative. |
| `input_token_weight` | `float` | `0.01` | Discount for logical input tokens. Exported AReaL prompts resend the full context even when the inference server can reuse a cached prefix, so input tokens are charged separately and cheaply. |
| `output_token_weight` | `float` | `1.0` | Finite, non-negative. When `enabled`, at least one of the two weights must be positive. |
| `attribution` | `str` | `"policy_subtree"` | The only legal value; anything else raises. |

### `rollout:`

`VariableBatchInferenceEngineConfig` in <span class="pl-src">platoon/utils/train.py</span> is AReaL's
`InferenceEngineConfig` plus two Platoon fields.

| Key | Type | Default | What it does |
|---|---|---|---|
| `rollout.shuffle_cross_task` | `bool` | `False` | Shuffle datums across tasks before dispatch. |
| `rollout.ensure_batch_divisible_by` | `int` | `1` | Trim the accepted batch so its size is divisible by `lcm(dp_size, this)`. The trim draws a random subset and prefers non-root datums, falling back to roots only when there are not enough non-root candidates. |

`rollout.backend` is an upstream field that Platoon makes mandatory: an empty value raises
`"rollout.backend must be set explicitly"`. Everything else under `rollout:` is upstream
passthrough; the keys this repo's configs actually use are listed
[below](#upstream-areal-passthrough).

### `actor:` and `ref:`

`PlatoonPPOActorConfig` subclasses upstream `PPOActorConfig` and adds six fields.

| Key | Type | Default | What it does |
|---|---|---|---|
| `actor.loss_fn` | `str` | `"grpo"` | **Runtime-only.** `PlatoonArealRLTrainerConfig.__post_init__` overwrites it from `loss_fn_config.loss_fn`, so setting it in YAML has no effect. |
| `actor.loss_fn_kwargs` | `dict[str, Any]` | `{}` | Same: merged from `loss_fn_config.loss_fn_kwargs`, with `loss_fn_config` winning on conflicts. |
| `actor.enable_router_replay` | `bool` | `False` | The single public gate for router replay (R3). Enabling it triggers the validation chain below. |
| `actor.router_replay_num_layers` | `int \| None` | `None` | Required positive int when R3 is enabled. |
| `actor.router_replay_topk` | `int \| None` | `None` | Required positive int when R3 is enabled. |
| `actor.router_replay_num_experts` | `int \| None` | `None` | Optional; must be positive when set. |

Enabling `actor.enable_router_replay` adds six cross-field checks in
`PlatoonArealRLTrainerConfig.__post_init__`:

1. `actor.backend` must start with `megatron`.
2. `rollout.backend` must start with `sglang`.
3. `rollout.return_routed_experts` must be `true`.
4. `actor.megatron.enable_mtp` must be `false` — rollout routes do not include MTP layers.
5. `actor.should_compute_prox_logp()` must be false, meaning both `recompute_logprob` and
   `use_decoupled_loss` off. Forward-only replay is not implemented.
6. With `actor.gradient_checkpointing` on, `actor.megatron.recompute_granularity` must be `full` and
   `actor.megatron.recompute_method` must be `uniform`.

Everything else under `actor:` — optimizer, clipping, normalization, the Megatron and FSDP blocks —
is upstream `PPOActorConfig` / `TrainEngineConfig`. Platoon requires `actor.backend` to be non-empty
and raises otherwise.

`ref:` is the same class, defaulting to `None`, meaning no reference engine. If you provide a `ref:`
block and omit `ref.backend`, Platoon copies `actor.backend` into it. Repo configs colocate `ref`
with the actor and set `optimizer: null` so it is never trained.

!!! note "Worker allocator environment"
    `__post_init__` injects `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` into the `env_vars`
    of every `scheduling_spec` on `rollout`, `actor`, `ref`, `critic` and `teacher`. In
    single-controller AReaL the trainer object is not the process doing GPU work, so the setting has
    to reach the scheduler-launched workers before CUDA initializes there. An existing value is
    preserved and appended to.

### `loss_fn_config:`

| Key | Type | Default | What it does |
|---|---|---|---|
| `loss_fn_config.loss_fn` | `str` | `"grpo"` | Registered loss name. `grpo`, `ppo` and `cispo` are registered in <span class="pl-src">platoon/train/areal/loss_functions.py</span>. Typed as `str` rather than `Literal` for OmegaConf compatibility, so a typo fails at lookup, not at parse. |
| `loss_fn_config.loss_fn_kwargs` | `dict[str, Any]` | `{}` | Loss-specific kwargs. The registered loss's own defaults apply first and these override them. CISPO registers `clip_low_threshold=0.0` and `clip_high_threshold=5.0`. |

Kwargs must go inside `loss_fn_kwargs`. Writing `loss_fn_config.clip_low_threshold` at the top of
the block is rejected.

### `environments:`

The registry wiring list, shared with the Tinker tree. See
[`EnvironmentConfig`](#environmentconfig) below.

### Post-init side effects worth knowing

| Effect | Consequence |
|---|---|
| `scheduler.type` set to `"local"` when unset | Platoon's AReaL path relies on the single-controller scheduler. |
| `eval_gconfig` filled from `gconfig.new()` | You rarely need an `eval_gconfig:` block at all. |
| `actor.loss_fn` and `actor.loss_fn_kwargs` overwritten | Loss selection has exactly one public home. |
| `workflow_config` router-replay fields copied from `actor` | Remote rollout workers get the dimensions needed to reshape SGLang's flattened routing data, without a second gate to keep in sync. |
| `environments` normalized, then `len > 1` raises | Exactly one entry is supported today. |

---

## Tinker trainer config

`PlatoonTinkerRLTrainerConfig` in <span class="pl-src">platoon/train/tinker/config_defs.py</span>.
Three fields have no default and are required: `train`, `eval`, `log_path`.

| Key | Type | Default | What it does |
|---|---|---|---|
| `train` | `TrainConfig` | **required** | |
| `eval` | `EvalConfig` | **required** | |
| `log_path` | `str` | **required** | Base log directory. The run path is `log_path/stats.experiment_name/stats.trial_name`. |
| `tinker_base_url` | `str \| None` | `None` | Passed to `tinker.ServiceClient(base_url=...)`. |
| `environments` | `list[EnvironmentConfig]` | `[EnvironmentConfig()]` | Same class as the AReaL tree. More than one entry raises. |
| `checkpoint` | `CheckpointConfig` | see below | |
| `stats` | `StatsConfig` | see below | |
| `watchdog` | `WatchdogConfig` | see below | |

### `train:`

| Key | Type | Default | What it does |
|---|---|---|---|
| `train.model_name` | `str` | **required** | HuggingFace model identifier. |
| `train.renderer_name` | `str` | **required** | Prompt renderer, e.g. `qwen3`, `qwen3_instruct`. |
| `train.renderer_kwargs` | `dict` | `{}` | Renderer attribute overrides. |
| `train.context_window_length` | `int \| None` | `None` | Passed to the renderer. |
| `train.batch_size` | `int` | `32` | Tasks per optimizer batch, and the default rollout worker count. |
| `train.num_epochs` | `int \| None` | `None` | Set epochs, steps, or both. With both, `num_train_batches` takes `min(epochs × batches_per_epoch, max_training_steps)` — the shorter run wins, despite the "takes max" comment in the dataclass. Setting neither raises. |
| `train.max_training_steps` | `int \| None` | `None` | |
| `train.num_minibatches` | `int` | `1` | Weight updates per batch. `batch_size % num_minibatches == 0` is asserted at startup. |
| `train.num_microbatches` | `int` | `1` | Gradient-accumulation splits per minibatch. `(batch_size / num_minibatches) % num_microbatches == 0` is asserted. |
| `train.max_staleness` | `int \| None` | `None` | Maximum off-policy lag, in optimizer steps. |
| `train.loss_fn` | `str` | `"cispo"` | Passed **straight to Tinker's** `forward_backward_async`. This is not Platoon's AReaL loss registry and the names are not interchangeable. |
| `train.loss_fn_config` | `dict` | `{"clip_low_threshold": 0.0, "clip_high_threshold": 5.0}` | Passed straight to Tinker; also read locally for clip-fraction metrics. This default is a whole dict, so a YAML block replaces it rather than merging into it. |
| `train.lora_rank` | `int` | `32` | Rank for `create_lora_training_client`. |
| `train.workflow_config` | `WorkflowConfig` (Tinker) | see below | |
| `train.num_concurrent_rollout_workflow_workers` | `int \| None` | `None` | `__post_init__` fills it with `batch_size` when unset. |

`train.optimizer` is `AdamParams`:

| Key | Type | Default | What it does |
|---|---|---|---|
| `train.optimizer.learning_rate` | `float` | `3e-5` | |
| `train.optimizer.beta1` | `float` | `0.9` | |
| `train.optimizer.beta2` | `float` | `0.95` | |
| `train.optimizer.eps` | `float` | `1e-8` | |
| `train.optimizer.weight_decay` | `float` | `0.0` | |
| `train.optimizer.grad_clip_norm` | `float` | `0.0` | When `> 0`, Tinker returns `grad_norm` in `OptimStepResponse.metrics`. Set a large value such as `1e12` to get the logging without actually clipping. |

### `train.workflow_config:` / `eval.workflow_config:`

!!! warning "A different class with the same name"
    The Tinker `WorkflowConfig` is a smaller, separate dataclass from the AReaL one. Its default
    `group_size` is **8**, not 1, and it has no `use_subprocesses`, `straggler_timeout_seconds`,
    `straggler_quorum`, `subprocess_shutdown_grace_seconds`, `min_successful_group_size`,
    `depth_level_discount_gamma`, `token_efficiency_reward`, `filter_zero_variance_groups` or
    router-replay fields. Since the Tinker loader drops unknown keys silently, copying one of those
    over from an AReaL config does nothing and says nothing.

| Key | Type | Default | What it does |
|---|---|---|---|
| `group_size` | `int` | `8` | |
| `rollout_config` | `RolloutConfig` | see above | |
| `leave_one_out_baseline` | `bool` | `False` | |
| `depth_level_weighting` | `bool` | `False` | Weight trajectories inversely by depth-level frequency. |
| `subagent_datum_keep_probability` | `float` | `1.0` | Must be in `[0, 1]`. |
| `subagent_datum_sampling_seed` | `int` | `0` | Must be a non-bool int. |
| `filter_errors` | `bool` | `True` | |
| `filter_zero_advantage_datums` | `bool` | `True` | Baselines and rollout metrics are computed before this filter runs. |

### `eval:`

`EvalConfig` extends `TrainEventTriggerConfig`.

| Key | Type | Default | What it does |
|---|---|---|---|
| `eval.strategy` | `"epoch" \| "step" \| "none"` | `"epoch"` | |
| `eval.every` | `int` | `1` | Also gates whether the eval loop task is created at all. |
| `eval.num_concurrent_rollout_workflow_workers` | `int` | `256` | |
| `eval.workflow_config` | `WorkflowConfig` (Tinker) | `group_size=1`, `filter_errors=False`, `filter_zero_advantage_datums=False` | The eval default differs from the train default on all three of those fields. |

!!! warning "Writing any `eval.workflow_config:` block discards those three eval defaults"
    They come from a `default_factory` on `EvalConfig`, and `_dataclass_from_dict` never consults it
    once the YAML supplies a `workflow_config` mapping — it builds a fresh `WorkflowConfig` from
    that mapping plus the *class* defaults. So an `eval.workflow_config:` block that sets only
    `group_size: 1` silently gets `filter_errors: true` and `filter_zero_advantage_datums: true`
    back. If you write the block at all, write all three keys.

### `checkpoint:`

| Key | Type | Default | What it does |
|---|---|---|---|
| `checkpoint.strategy` | `"epoch" \| "step" \| "none"` | `"epoch"` | |
| `checkpoint.every` | `int` | `1` | |
| `checkpoint.load_checkpoint_path` | `str \| None` | `None` | Explicit weights to load when auto-resume finds nothing. |

### `stats:`

Platoon's own logging config, not AReaL's. `StatsConfig.to_stats_logger_config` builds a
`StatsLoggerConfig` from <span class="pl-src">platoon/utils/stats_logger.py</span>.

| Key | Type | Default | What it does |
|---|---|---|---|
| `stats.experiment_name` | `str` | `"platoon_tinker"` | |
| `stats.trial_name` | `str` | `"run"` | |
| `stats.wandb.mode` | `str` | `"online"` | `"online"`, `"offline"` or `"disabled"`. The default is `online`, so a Tinker run tries to log to W&B unless you set `disabled`. This is Platoon's own `WandBConfig` in <span class="pl-src">platoon/utils/stats_logger.py</span>, not AReaL's identically named class under `stats_logger:`. |
| `stats.wandb.project` | `str \| None` | `None` | |
| `stats.wandb.entity` | `str \| None` | `None` | |
| `stats.wandb.name` | `str \| None` | `None` | |
| `stats.wandb.group` | `str \| None` | `None` | |
| `stats.wandb.tags` | `list[str]` | `[]` | |
| `stats.wandb.notes` | `str \| None` | `None` | |
| `stats.wandb.api_key` | `str \| None` | `None` | |
| `stats.wandb.base_url` | `str \| None` | `None` | |
| `stats.wandb.resume_run_id` | `str \| None` | `None` | W&B run id to resume. |

`StatsLoggerConfig` also carries `print_stats` (`True`) and `log_interval` (`1`), but
`to_stats_logger_config` does not pass them through, so neither is reachable from a Tinker YAML.

### `watchdog:`

| Key | Type | Default | What it does |
|---|---|---|---|
| `watchdog.enabled` | `bool` | `True` | Background thread that hard-exits the process if Tinker hangs. |
| `watchdog.timeout_seconds` | `float` | `600` | Real configs use 3600 or 7200; long agentic rollouts trip the default. |
| `watchdog.exit_code` | `int` | `2` | Exit code used when the watchdog kills the process. |

---

## Inference config

Every plugin's inference root config embeds `InferenceBenchmarkConfig` under `inference:`, from
<span class="pl-src">platoon/inference/config_defs.py</span>. Plugins add their own top-level keys
alongside it — `dataset_split`, `num_tasks`, `stage`, `seed`, and so on — which are per-plugin and
covered in [the inference tutorial](../tutorials/inference.md).

| Key | Type | Default | What it does |
|---|---|---|---|
| `inference.model_name` | `str` | **required** | Model id in LiteLLM/OpenAI form. Omitting it raises a `TypeError` at load. |
| `inference.model_endpoint` | `str \| None` | `None` | Base URL of the served model. |
| `inference.model_api_key` | `str \| None` | `None` | |
| `inference.output_dir` | `str` | `"inference_results"` | The runner writes rollouts and reports underneath this. |
| `inference.resume` | `bool` | `True` | Reuse existing per-task rollout artifacts instead of re-running them. |
| `inference.workflow` | `InferenceWorkflowConfig` | see below | |

| Key | Type | Default | What it does |
|---|---|---|---|
| `inference.workflow.num_rollouts_per_task` | `int` | `1` | pass@k sampling width. |
| `inference.workflow.num_concurrent_workers` | `int` | `32` | asyncio fan-out. |
| `inference.workflow.success_threshold` | `float` | `1.0` | Fallback success cutoff, used only when the trajectory carries no explicit success metric. |
| `inference.workflow.fail_fast` | `bool` | `False` | |
| `inference.workflow.use_subprocesses` | `bool` | `False` | Run each rollout in a process pool. |
| `inference.workflow.subprocess_max_workers` | `int \| None` | `None` | |
| `inference.workflow.rollout_config` | `RolloutConfig` | see above | Several fields are overwritten at runtime; see the note. |

!!! note "Fields the inference workflow overwrites"
    `rollout_config.model_name`, `model_endpoint` and `model_api_key` are copied from the
    `inference.*` values, `output_dir` is replaced per task, `train` is forced to `False` and
    `return_dict` to `True`. Setting any of those under `inference.workflow.rollout_config` has no
    effect.

---

## `EnvironmentConfig`

The top-level `environments:` list, defined in
<span class="pl-src">platoon/train/components.py</span>. This is the registry wiring layer: each
string field is either a registry name or a dotted import path, resolved by the `Auto*` factories in
<span class="pl-src">platoon/train/auto.py</span>. See [the registry](../architecture/registry.md).

!!! warning "This is not OpenReward's `environments:`"
    The top-level `environments:` list described here selects Platoon components by name. The
    OpenReward plugin has its own nested `openreward.environments:` list, with fields like `label`,
    `env_name`, `session_url` and `sampling_weight` — that is an environment *mixture*, an entirely
    different thing that happens to share a key name. See
    [OpenReward](../integrations/openreward.md).

| Key | Type | Default | What it does |
|---|---|---|---|
| `package` | `str \| None` | `None` | Dotted module imported for its registration side effects. |
| `discover_entry_points` | `bool` | `False` | Also load every `platoon.plugins` entry point. |
| `trainer_config` | `str \| None` | `None` | **Dead key.** The field exists and plugins register names for it, but no resolver ever looks one up. |
| `dataset_loader` | `str \| None` | `None` | Required for the train split; `AutoDataset` raises without it. |
| `eval_dataset_loader` | `str \| None` | `None` | Falls back to `dataset_loader`. |
| `task_loader` | `str \| None` | `None` | Required. |
| `rollout` | `str \| None` | `None` | Required for the train split. |
| `eval_rollout` | `str \| None` | `None` | Falls back to `rollout`. |
| `reward_processor` | `str \| None` | `None` | When unset, `AutoRewardProcessor` returns the identity `lambda traj: (traj["reward"], {})`. |
| `workflow` | `str` | `"group_rollout"` | The literal `"group_rollout"` means "use the backend's default workflow class". Any other value is resolved from the `workflow` registry. |
| `dataset_kwargs` | `dict` | `{}` | Extra kwargs for the train `dataset_loader(config, split, **kwargs)`. |
| `eval_dataset_kwargs` | `dict` | `{}` | Extra kwargs for the eval loader. |
| `workflow_kwargs` | `dict` | `{}` | Extra kwargs for the train workflow constructor. The entrypoint pops two with defaults: `output_subdir` (AReaL, `"train_rollout"`) or `stats_scope` (Tinker, `"train"`), and `filter_errors` (`True`). |
| `eval_workflow_kwargs` | `dict` | `{}` | Same, with `output_subdir="eval_rollout"` / `stats_scope="eval"` and `filter_errors=False`. |

`environments` must be a list. Passing a bare dict or a single `EnvironmentConfig` raises with a
targeted message, and more than one entry raises `NotImplementedError` — in both trainer configs and
again in `AutoEnvironment.from_config`.

!!! info "Adoption today"
    Most plugins still ship their own `train_*.py` script that wires components in Python and never
    touches `environments:`. One config in the repo uses the block today,
    <span class="pl-src">plugins/textcraft/platoon/textcraft/configs/tinker/textcraft_synth_depth_aware_tinker.yaml</span>.
    It is the recommended path for new work, not the universal one.

---

## Upstream AReaL passthrough

These blocks appear in this repo's AReaL YAMLs but are defined by upstream AReaL, not by Platoon.
Their defaults belong to the AReaL revision pinned in `pyproject.toml`, so this page names the keys
the configs actually set and points at AReaL for semantics and defaults rather than restating values
it cannot verify against that pin. Upstream source:
[`areal/api/cli_args.py`](https://github.com/inclusionAI/AReaL/blob/main/areal/api/cli_args.py).

| Block | Keys this repo's configs set |
|---|---|
| top level | `experiment_name`, `trial_name`, `seed`, `total_train_epochs`, `tokenizer_path`, `enable_offload` |
| `cluster` | `n_nodes`, `n_gpus_per_node`, `fileroot`, `name_resolve.type`, `name_resolve.nfs_record_root` |
| `scheduler` | `type` |
| `rollout` | `backend`, `experiment_name`, `trial_name`, `fileroot`, `tokenizer_path`, `consumer_batch_size`, `max_concurrent_rollouts`, `queue_size`, `max_head_offpolicyness`, `setup_timeout`, `request_timeout`, `enable_rollout_tracing`, `dump_to_file`, `scheduling_spec`, `use_lora`, `return_routed_experts`, `agent.tool_call_parser`, `agent.reasoning_parser`, `agent.session_timeout_seconds` |
| `actor`, `ref` | `backend`, `path`, `experiment_name`, `trial_name`, `init_from_scratch`, `disable_dropout`, `gradient_checkpointing`, `dtype`, `mb_spec.max_tokens_per_mb`, the whole `optimizer` block, `scheduling_spec`, `scheduling_strategy`, `eps_clip`, `eps_clip_higher`, `kl_ctl`, `kl_estimator`, `ppo_n_minibatches`, `recompute_logprob`, `use_decoupled_loss`, `reward_scaling`, `reward_bias`, `reward_norm.*`, `adv_norm.*`, `use_lora`, `lora_rank`, `lora_alpha`, `target_modules`, `peft_type`, `weight_update_mode`, `megatron.*` |
| `sglang` | `model_path`, `random_seed`, `skip_tokenizer_init`, `dtype`, `max_running_requests`, `context_length`, `mem_fraction_static`, `attention_backend`, `disable_radix_cache`, `enable_lora` |
| `saver`, `evaluator`, `recover` | `experiment_name`, `trial_name`, `fileroot`, `freq_epochs`, `freq_steps`, `freq_secs`, plus `recover.mode` and `evaluator.eval_before_train` |
| `stats_logger` | `experiment_name`, `trial_name`, `fileroot`, `wandb.mode`, `wandb.project`, `wandb.group` |

Three upstream keys carry Platoon-specific behavior worth calling out:

- `scheduler.type` — Platoon sets it to `"local"` when you leave it unset, and adds the extra value
  `"slurm_prealloc"` for runs launched into a pre-allocated Slurm allocation.
- `evaluator.eval_before_train` — Platoon skips building the eval rollout controller entirely when
  this is falsy *and* all three `evaluator.freq_*` are `None`.
- `rollout.agent.tool_call_parser` — the upstream default parser cannot parse Qwen3-Coder's XML tool
  calls; with the wrong parser, `tool_calls` come back `null` and OpenHands ends every episode after
  one step. See [OpenHands](../integrations/openhands.md).

`vllm:`, `critic:`, `teacher:` and `perf_tracer:` exist upstream and are reachable from an AReaL
config, but no config in this repo sets them.

---

## Keys that silently drop training data

Several defaults discard datums or whole groups without failing. They are on for good throughput
reasons, but when a batch is smaller than you expected, start here.

| Key | Default | What it drops |
|---|---|---|
| `workflow_config.filter_zero_advantage_datums` | `True` | Datums whose centered scalar reward is exactly zero. |
| `workflow_config.filter_zero_variance_groups` | `True` | Entire groups where every member got the same reward. |
| `workflow_config.min_successful_group_size` | `1` | Rejects and replenishes a group returning fewer usable members. Raising it discards more groups. |
| `workflow_config.subagent_datum_keep_probability` | `1.0` | Below 1.0, Bernoulli-drops post-merge subagent datums. Roots are always kept. |
| `filter_errors` (workflow constructor arg) | `True` for train, `False` for eval | Masks error tokens that would otherwise receive positive credit. Token-level, not datum-level. Set through `environments[].workflow_kwargs`; the `workflow_config` field of the same name is inert. |
| `rollout.ensure_batch_divisible_by` | `1` | Trims the accepted batch to a divisible size, preferring non-root datums. |
| `train_dataset.drop_last` | `True` | The trailing partial batch of every epoch. |

!!! danger "`filter_zero_advantage_datums` is only sound for reward-only objectives"
    It uses the centered scalar reward as an early proxy for the final policy advantage. That proxy
    breaks whenever something else can turn a zero reward into a nonzero objective. Disable it when
    any of these holds:

    - `actor.kl_ctl` is nonzero
    - reward or advantage normalization is active (`actor.reward_norm`, `actor.adv_norm`)
    - `actor.reward_bias` is nonzero, or `actor.overlong_reward_penalty` is enabled
    - a critic or teacher/distillation objective is configured
    - the model has an independent MoE or router auxiliary loss
    - a custom batch transform adds to rewards

    The trainer emits a `RuntimeWarning` at startup repeating these constraints and listing the
    incompatible settings it detected, because the remote workflow cannot validate the full actor
    and objective configuration by itself. The generic entrypoint forces the filter off for
    evaluation regardless.

---

## See also

- [Config architecture](../architecture/config.md) — how the three trees are composed and loaded.
- [CLI reference](cli.md) — entrypoints and their arguments.
- [Components reference](components.md) — the registry names these config strings resolve against.
- [Plugins reference](plugins.md) — the per-plugin top-level keys each trainer config adds.
- [Troubleshooting](troubleshooting.md) — what the common config errors mean.
