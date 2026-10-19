# Train and evaluate

This guide covers the commands that run a policy: debug one task with `platoon rollout`, train with
`platoon train`, evaluate with `platoon eval` and look at trajectories with `platoon viz`. It also explains the
settings that most often matter: the training service, sampling defaults, renderers, checkpoints, restarts and the
metrics that tell you whether a run is healthy.

**Goal:** go from a config to a trained checkpoint and an evaluation report, checking each stage before spending
compute on the next.

**Prerequisites:**

- platoon installed with the `tinker` extra (`pip install 'platoon[tinker,ors]'`, or
  `uv sync --extra tinker --extra ors` in a checkout).
- Access to a Tinker-compatible training service: an API key in `TINKER_API_KEY` (one that starts with `tml-`),
  and for a self-hosted service, such as [miles-tinker](https://github.com/neulab/miles-tinker), its URL in `TINKER_BASE_URL` (or
  `tinker_base_url` in a training config).
- A config that pairs an environment with a harness. [Write and override configs](configs.md) explains the file
  format; [`examples/config-layers`](../../examples/config-layers/README.md) is a complete set.

## The commands

| Command | What it does |
|---|---|
| `platoon rollout CONFIG --task ID [key=value ...]` | Run one task and print its agent tree, for debugging |
| `platoon train CONFIG [key=value ...]` | Train on the environments the config selects |
| `platoon eval CONFIG [key=value ...]` | Run every task of a split, grade it and write a report |
| `platoon viz tail\|replay\|show-dump ...` | Open the trajectory viewer on event files or saved trajectories |
| `platoon config show\|validate CONFIG` | Print the merged config or check it |
| `platoon harness list\|show NAME` | The harnesses (`platoon-codeact`, `platoon-react`, `platoon-bash`, `mini-swe-agent`, and the installed packages') |
| `platoon program list\|show NAME` | The episode programs (`single`, `gym`, and the installed packages') |
| `platoon env list [CONFIG]\|show NAME` | The environment service schemes, the named environments, and the environments a config selects |
| `platoon extensions list\|show NAME\|check` | The installed packages' components with their versions, one of them in detail, and what is wrong with them |
| `platoon sandbox reap RUN_DIR` | Reap the sandboxes a run left running when its process was killed ([Long runs](#long-runs)) |

Every command takes `--help`.

## Steps

### 1. Run one task first

`platoon rollout` runs a single task through the same path training uses (environment session, harness, model
gateway, grading) and prints the agent tree. It works with a training config (it samples `train.model_name` with
`train.renderer_name`) and with an evaluation config:

```bash
uv run platoon rollout examples/config-layers/smoke.yaml --task train:0 --output-dir runs/debug
```

```text
rollout train-0-5933e9d60286  task train:0
agents 1  steps 10  max depth 0  reward 1.0  outcome scored reward=1.0
└─ policy e4c499d6  steps 10  reward 1.0  outcome scored reward=1.0
```

Task ids depend on the environment: `<split>:<index>` for ORS environments, the task directory name for Harbor
datasets, and `<environment name>::<task id>` in a mixture of several environments. Useful flags:

| Flag | Effect |
|---|---|
| `--split NAME` | Which split's task and rollout settings to use (default: `train` for a training config, the config's `split` for an evaluation config) |
| `--model NAME`, `--renderer NAME` | Sample another base model, or render with another renderer |
| `--model-path tinker://...` | Sample a trained checkpoint instead of the base model |
| `--json-output` | Print the summary as JSON instead of a tree |
| `--prepare` | Run `platoon prepare` on the config and overrides first, on one machine: the whole config, whatever task runs, with the tokenizer of the model and renderer the rollout samples ([Prepare as a run starts](sandboxes-and-harbor.md#prepare-as-a-run-starts-on-one-machine)) |

The run directory gets `config.resolved.yaml`, `lock.json`, `trajectory_collection.json`, which
`platoon viz show-dump` opens, and the rollout's event file under `rollouts/events/`, which `platoon viz replay`
reads.

### 2. Train

```bash
uv run platoon train examples/config-layers/smoke.yaml train.batch_size=4
```

Each step takes `train.batch_size` tasks, runs `train.workflow_config.group_size` rollouts of each, gives every
rollout an advantage relative to its group (`leave_one_out_baseline: true` excludes the rollout itself from its
baseline), and trains on the exact tokens the policy sampled. Rollouts whose outcome was not measured (an
environment or grader failure, a harness crash, a timeout) are kept out of training and out of the group baseline;
they never count as reward 0.

The settings you will change most:

| Setting | Default | Meaning |
|---|---|---|
| `train.batch_size` | 32 | Tasks per step; at most the training tasks of one environment under the default `train.task_sampler: auto` |
| `train.num_epochs`, `train.max_training_steps` | none | Run length; set at least one, or `platoon train` exits 78 (with both, the shorter wins) |
| `train.workflow_config.group_size` | 8 | Rollouts per task |
| `train.lora_rank` | 32 | Rank of the LoRA training client platoon asks the service for |
| `train.optimizer.learning_rate` | 3e-5 | Adam learning rate (`beta1`, `beta2`, `eps`, `weight_decay`, `grad_clip_norm` sit beside it) |
| `train.loss_fn`, `train.loss_fn_config` | `cispo`, clip thresholds 0.0 and 5.0 | The loss the service computes in `forward_backward`; a `loss_fn_config` replaces the default whole, so it names every key its loss takes |
| `train.max_staleness` | none | How many steps rollouts may run ahead of training, and how many policy versions old their data may be before it is dropped; none sets no limit. An environment may raise its own groups' bound ([Settings per environment](#settings-per-environment)) |
| `train.max_concurrent_rollouts` | `batch_size × group_size` | Rollouts in flight at once, started a whole group at a time (`eval.max_concurrent_rollouts`, 256 by default, for an evaluation) |
| `train.workflow_config.rollout_config.max_steps`, `.timeout` | none | Step budget per agent, and time limit of each rollout's root agent in seconds; without one, a harness stops at its own limits (its spec's `timeout_s`, a Harbor task's `[agent] timeout_sec`), which also set a Harbor rollout's deadline ([Harness timeouts](sandboxes-and-harbor.md#harness-timeouts-and-crashes)) |
| `train.workflow_config.rollout_isolation` | `inprocess` | `process` runs each rollout in its own killable worker process, which sees no GPU (its sandboxes get theirs from their provider) |
| `train.workflow_config.worker_start_method` | `forkserver` | How a worker process starts: forked from a process that imported the worker's stack once (seconds), or `exec`, a new interpreter that imports it (17-24 s, about a minute with 64 at once) |
| `eval.every` | 0 | Evaluate during training every N training steps; 0 turns it off, and the eval split is then never listed. Evaluation rolls out as training does ([Evaluate during training](#evaluate-during-training)) |
| `eval.at_start` | false | Also evaluate the starting policy, at step 0, before the first update: the point the held-out curve starts from |

A run writes everything under `<log_path>/<stats.experiment_name>/<stats.trial_name>/`. A config without `log_path`
writes under the site profile's `runs`, with `stats.experiment_name` named after the config file unless it names one
([Run on a cluster](clusters.md#4-write-the-sites-profile-once)); a relative `log_path` is relative to the working
directory. The run directory holds the run record
(`config.resolved.yaml`, `lock.json`, and `process.json`, the process that runs it), `checkpoints.jsonl`,
`evaluations.jsonl` (each evaluation during training that completed), and `rollouts/train/<policy version>/` and
`rollouts/eval/<policy version>/` with each rollout's harness logs (`harness/`) and viewer event files (`events/`).
Metrics go to the console after every step and to W&B unless `stats.wandb.mode` is `disabled` (W&B needs
`WANDB_API_KEY`).

### 3. Evaluate

An evaluation config has an `inference:` section in place of `train:` (see
[`examples/config-layers/eval.yaml`](../../examples/config-layers/eval.yaml)):

```yaml
inference:
  model_name: Qwen/Qwen3-4B-Instruct-2507
  renderer_name: qwen3_instruct
  model_path: null                 # or a tinker:// sampler checkpoint
  workflow:
    num_rollouts_per_task: 4
    num_concurrent_workers: 16
    rollout_config:
      max_steps: 12
      timeout: 300
environments:
  - ...                            # the same entries as training
split: eval
```

```bash
uv run platoon eval eval.yaml                                                       # the base model
uv run platoon eval eval.yaml inference.model_path=tinker://<run>/sampler_weights/final   # a checkpoint
```

Each start writes to a directory of its own, `runs/eval/<policy>/<UTC time>/` under the working directory (under
the site profile's `runs` when it has one: [Run on a cluster](clusters.md#4-write-the-sites-profile-once)), and
names it on stderr (`platoon: inference.output_dir is not set, so this evaluation writes to ...`). `<policy>` is the
checkpoint, without its `tinker://`, else the model, as one directory name: the two commands above write to
`runs/eval/Qwen-Qwen3-4B-Instruct-2507/20261018T091500Z/` and `runs/eval/<run>-sampler_weights-final/<UTC time>/`.
A second start in the same second gets `-2` after the time. The directory, there or in `inference.output_dir`,
holds:

- `config.resolved.yaml`, which names the directory as `inference.output_dir`, and `lock.json`;
- `rollouts/<task>/rollout_<k>/`: `trajectory_collection.json`, `metadata.json`, `harness/` logs and `events/`;
- `reports/final_report.json` (summary, step and wall-time statistics, reward components, per-task results) and
  `reports/task_results.jsonl`.

`platoon eval` prints the report summary as JSON.

**Resuming.** Name the directory to resume an evaluation that stopped, or to report rollouts already on disk:
`platoon eval eval.yaml inference.output_dir=runs/eval/<policy>/<UTC time>`. Its rollouts are reused, and only
those that are missing or errored run. An evaluation never resumes another policy's rollouts: each rollout's
`metadata.json` records the policy that made it (`provenance`: the backend and the model, a Tinker policy's
checkpoint and renderer, or a hosted model's base URL origin, and whether judges ran), and a directory whose rollouts
record another stops the run with exit 78 before anything in it changes, also with `inference.resume: false` or
`stage: report`. `stage: report` needs the directory named: a new one holds no rollouts.

The summary fields to read first:

| Field | Meaning |
|---|---|
| `success_rate` | Successful rollouts over **valid** rollouts: a harness rollout succeeds when the reward its environment scored, before any reward processor, reaches `inference.workflow.success_threshold` (default 1.0); a legacy rollout function's own success flag counts at 1 |
| `success_at_k` | Mean over tasks of the best of each task's `num_rollouts_per_task` rollouts |
| `reward_mean` | Mean reward of the valid rollouts |
| `valid_rollouts`, `errored_rollouts` | Rollouts whose outcome was measured, and those that were not |
| `outcome_status_counts` | How many rollouts ended `scored`, `invalid`, `error`, `timeout`, `cancelled` or raised |
| `reward_key_counts` | How many rollouts trained each reward key, the name their grader reported the reward under (each rollout's `metadata.json` names it as `reward_key`) |

An errored rollout (an infrastructure failure, a harness crash, a timeout) stays out of `success_rate` and the
reward statistics. Benchmarks that time-limit the policy count a timeout as a failure instead:
`inference.workflow.count_timeout_as_failure=true`.

Other evaluation settings:

| Setting | Default | Meaning |
|---|---|---|
| `inference.output_dir` | none | The directory to write to, or to resume or report; none writes each start to a new `runs/eval/<policy>/<UTC time>/` (in the site profile's `runs` when it has one) |
| `inference.resume` | true | Reuse the rollouts already in `output_dir` (only ever this policy's), so a rerun only runs what is missing |
| `inference.retry_errored` | true | With `resume`, run the errored rollouts again |
| `stage` | `full` | `rollouts` only collects rollouts; `report` only rebuilds the report from rollouts on disk, needs `output_dir`, and needs no `TINKER_API_KEY` |
| `inference.hosted` | none | A hosted model to evaluate instead of a Tinker policy: a `base_url` that speaks the harness's dialect ([Evaluate a hosted model](evaluate-hosted-models.md)) |
| `inference.judges` | true | false runs no verifier agents, behavior or parent-usage judges, and refuses judge calls ([Turn judges off](evaluate-hosted-models.md#turn-judges-off)) |
| `inference.workflow.use_subprocesses` | false | Run rollouts in worker processes, so one that hangs or dies costs only itself |
| `inference.workflow.fail_fast` | false | Stop after the rollout stage when any rollout errored |

`inference.renderer_name` is required for environments whose harnesses sample a Tinker policy; without it no policy
is served and `platoon eval` refuses to start.

Two more settings change what an evaluation samples:

- `inference.hosted` evaluates a hosted model instead of a Tinker policy: a base URL that speaks the harness's own
  dialect (OpenAI, Anthropic, a vLLM or SGLang server, OpenRouter, a LiteLLM proxy). The harness, grading and report
  stay the same.
- `inference.judges: false` turns off the model calls that judge agents rather than solve the task: verifier
  agents, behavior and parent-usage judges, and judge calls. Success metrics never depend on them.

[Evaluate a hosted model](evaluate-hosted-models.md) covers both.

### 4. Look at the trajectories

```bash
uv run platoon viz tail --rdir runs/number-search/smoke/rollouts       # the rollouts finished so far
uv run platoon viz replay --dir runs/number-search/smoke/rollouts/train/3/events
uv run platoon viz show-dump runs/debug/trajectory_collection.json      # a saved rollout
```

The viewer shows every agent's steps, tool calls and outcome. `--mode codeact` or `--mode openhands` forces a
rendering style; the default picks one per trajectory.

## Leave tasks out

`exclude_tasks` on an environment entry leaves tasks out of every task list the entry gives a run: its training
tasks, its evaluation tasks during training, and the tasks `platoon eval` runs. Use it for tasks that cannot be
solved or graded as written, so that they cost no rollouts:

```yaml
environments:
  - service: python://my_project.tasks:MyEnvironment
    harness: platoon-react
    exclude_tasks:
      - broken-grader-task     # a task name
      - train:13               # a task id
```

Each item is a task id or a task name:

| Environment | Task id | Task name |
|---|---|---|
| ORS (`python://`, `ors+http://`) | `<split>:<index>` | the task spec's `task_name`, `id`, `task_id` or `name` |
| Harbor dataset (`service: harbor`) | the task directory's name | the same |
| Task table (`service: table`) | the row's `id` | the same |
| A `dataset_loader` | what the loader returns | none |

- **When.** The tasks leave each list as the service lists it: after `shuffle_seed` shuffles it and before `limit`
  takes its share, so `limit: 100` still gives 100 tasks and a seeded sample keeps the order of the tasks that
  remain (a `dataset_loader` applies its own options, and the tasks leave what it returns). A mixture's
  interleaving and the dataloader's shuffle and batches see only the tasks that remain.
- **Ids stay.** No task is renumbered: `train:291` is still `train:291` when `train:13` is left out. The training
  order is drawn from fewer tasks, so it differs from that of a run without the exclusion; compare such runs on the
  tasks they share.
- **Checks.** `platoon train` and `platoon eval` check every item when they list the tasks, before anything runs,
  and exit 78 naming each item that matches no task of the environment. An item may name a task of another split,
  so one environment entry can serve training and evaluation. An exclusion that leaves an entry no tasks is a
  configuration error too. A dataset loader's tasks are known only by the ids it returns, so there an item that
  matches none of them is logged as a warning instead.
- **Named environments.** An entry that selects a named environment (`env: acme-mazes@1`) leaves tasks out the
  same way: its items name that environment's tasks, and they leave each list before the named environment's
  `limit` takes its share. A named environment cannot set `exclude_tasks` itself, so the entry's list is the whole
  list ([Named environments](../extending/index.md#named-environments)).
- **Record.** The run directory's `lock.json` gets `excluded_tasks`, which says for each environment and split how
  many tasks were listed and which ids were left out, so the record shows what each name stood for in that run:

  ```json
  "excluded_tasks": {"default": {"train": {"listed": 200, "excluded": ["train:8", "train:13"]}}}
  ```

`platoon rollout --task` runs any task, an excluded one included. An evaluation report covers every rollout in its
directory, so evaluate into a new one after you change `exclude_tasks`: leave `inference.output_dir` unset, or set
another.

## Train on several environments

`environments` may list several entries, a mixture. Each has a `name`, its own service, harness, program and
options, and a share of the training tasks, its `weight`. A mixture's task ids read `<name>::<task id>`, and its
metrics break down by entry ([Label tasks by a column](#label-tasks-by-a-column)). Without weights, every task of the
mixture is drawn equally often:

```yaml
environments:
  - name: code
    service: harbor
    service_kwargs: {dataset: hf://my-org/code-tasks@3f1c0e9b6d2a4c8e9f0a1b2c3d4e5f6a7b8c9d0e}
    harness: openhands
  - name: math
    service: python://my_project.math:MathEnvironment
    harness: platoon-react
```

| `weight` | An entry's share of the training slots | How often each of its tasks is drawn |
|---|---|---|
| `tasks` (default) | Its training tasks over the mixture's: 475 tasks and 22 share 475 to 22 | As often as every other task of the mixture |
| A positive number | Its number over their sum: `1` and `1` give each half | As often as its share allows: at `1` and `1`, each of 22 tasks about 22 times as often as each of 475 |

- **One or the other.** A mixture's weights are all numbers or all `tasks`: a mixture with both, such as one that
  gives a number to some entries and none to the others, does not load (exit 78).
- **Which tasks count.** `tasks` counts the tasks an entry trains on: those its listing keeps (`split_name`, `limit`,
  a Harbor dataset's `select`), once its `exclude_tasks` and a hold-out have left.
- **Another mix.** Numbers in proportion to the tasks give the same order as `tasks`, and change it from there: `475`
  and `44` draw each of the 22 tasks twice as often as each of the 475.

**The order.** Training draws a mixture's tasks in a weighted fair order: each slot goes to the entry furthest below
its share, so every batch holds about each entry's share, and each entry draws its tasks in shuffled passes, seeded
by its `seed`, every task once per pass. An epoch is as many tasks as the mixture holds, filled to whole batches, and
each epoch goes on with the order where the one before it stopped:

- An entry with fewer slots in an epoch than it has tasks draws the others in the next epochs, and each task of an
  entry is drawn as often as every other, give or take one.
- The order follows from the tasks, the weights, the seeds and `train.batch_size`, so a resumed run draws the
  batches that the run it resumes would have drawn.

A single environment's tasks are drawn in shuffled passes too (seed 42): an epoch is the whole batches they fill,
and the tasks left over start the next epoch. `train.task_sampler: weighted` draws them as a mixture of one instead,
with the entry's `seed` and the last batch filled from the next pass.

`train.task_sampler` chooses how a run draws its training tasks:

| `train.task_sampler` | A mixture | A single environment |
|---|---|---|
| `auto` (default) | The weighted fair order above | Shuffled passes by the dataloader (seed 42), each epoch the whole batches they fill |
| `weighted` | The weighted fair order above | The weighted order, as a mixture of one |

It was called `train.task_order` before platoon 0.3.0. That name still loads, with the same values, until platoon 0.4.0
removes it: the commands name it once on stderr with a deprecation warning, `lock.json` records it, and
`platoon config validate --strict` fails on it. A config may set both names when they agree, or when `task_order` is
`auto`, its default before 0.3.0, as a dump of platoon 0.2.0's `config show --defaults` spells it: `task_sampler` then
decides, and when the two differ, the warning says that `task_order` was ignored. A config that sets
`task_order: weighted` beside another `task_sampler` does not load (exit 78).

**Check the mix before a run.** Once `platoon train` lists the tasks, it logs how it draws them: each entry's tasks,
weight and share of the slots, and how often each of its tasks is drawn over the planned steps (`num_epochs`,
`max_training_steps`). It warns when a mixture's weights:

- leave tasks out of the run: an entry's share of the run's slots is smaller than its tasks, in a run long enough to
  draw every task once with `weight: tasks`;
- draw one entry's tasks more than 4 times as often as another's.

Each warning gives the numbers and the fix. `platoon config validate --tasks CONFIG` lists the tasks as `platoon
train` does and prints the same before a run
([Write and override configs](configs.md#4-check-the-result-before-you-run-it)). For the two entries above with
`weight: 1` each, 475 tasks and 22, 32 tasks a batch and 10 epochs:

```console
$ uv run platoon config validate --tasks train.yaml
ok   train.yaml (PlatoonTinkerRLTrainerConfig)
  training tasks: 497, drawn in epochs of 512 (16 batches of 32), 160 steps planned
  environments[code]: 475 tasks, weight 1, 50.0% of the slots: each task drawn about 5.4 times over the run's 160 steps
  environments[math]: 22 tasks, weight 1, 50.0% of the slots: each task drawn about 116.4 times over the run's 160 steps
  warning: environments[math]'s 22 training tasks are each drawn about 21.6 times as often as environments[code]'s 475 (about 116.4 and 5.4 times each over the run's 160 steps, at weights 1 and 1); weight: tasks on every entry draws every task of the mixture equally often
```

## Settings per environment

The environments of a mixture rarely want the same training settings: a dataset calibrated with a 30-step budget,
another whose rollouts take hours and want fewer of them per group and an older policy than the rest. Each entry may
set the training settings that act on its own groups, laid over the run's:

```yaml
train:
  batch_size: 20
  max_staleness: 3                          # the run's bound: how far rollouts run ahead of training
  rollout_group_limits: {slow-tasks: 1}     # one slow group at a time
  workflow_config:
    group_size: 8
    min_successful_group_size: 4
    rollout_config:
      max_steps: 200
      timeout: 3600
      inference_params: {temperature: 1.0, max_completion_tokens: 49152}
checkpoint: {recovery_every: 10, keep_recovery: 5}   # 5 × 10 = 50 > 48, the largest staleness bound
eval:
  every: 10
  workflow_config: {group_size: 4}
environments:
  - name: quick-tasks
    service: harbor
    service_kwargs: {dataset: hf://my-org/quick-tasks@3f1c0e9b6d2a4c8e9f0a1b2c3d4e5f6a7b8c9d0e}
    harness: openhands
    workflow_config:
      rollout_config: {max_steps: 30}       # its calibrated budget, in training and in its evaluations
  - name: slow-tasks
    service: harbor
    service_kwargs: {dataset: hf://my-org/slow-tasks@0d9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a4e3d2c1b}
    harness: openhands
    max_staleness: 48                       # its groups take hours: they train up to 48 versions old
    workflow_config:
      group_size: 4                         # four long rollouts per task, not eight
      min_successful_group_size: 3          # one may be lost
      leave_one_out_baseline: true
      rollout_config: {max_steps: null, timeout: null}   # the task's own limits, not the run's
    eval_overrides:
      workflow_config: {group_size: 1}      # one evaluation rollout per task
```

| An entry's key | What it sets | Default |
|---|---|---|
| `workflow_config` | Its training groups' settings, laid over `train.workflow_config`: `group_size`, `leave_one_out_baseline`, `min_successful_group_size`, `straggler_timeout_seconds`, `straggler_quorum`, `filter_errors`, `filter_message_actions`, `subagent_datum_keep_probability`, `subagent_datum_sampling_seed`, and in `rollout_config` `max_steps`, `timeout`, `step_timeout`, `inference_params`, `sampling_precedence`, `extra`, `propagate_root_success`, `skip_subagent_reward_computation` and `enable_subagent_verifiers` ([`EntryWorkflowSettings`](../reference/config.md#entryworkflowsettings)) | the run's |
| `eval_overrides.workflow_config` | The same keys, for its evaluation rollouts | none |
| `max_staleness` | How many policy versions old its training groups may be and still train, at least `train.max_staleness` | the run's |

**How they merge.** A key an entry sets replaces the run's, `null` included (`max_steps: null` takes the task's own
step budget, else the harness's), a section (`rollout_config`, its `inference_params`) merges key by key, and a
free-form mapping (`extra`) replaces the run's whole, as an `eval.workflow_config` merges over training's. Lowest
first:

| An entry's training groups | Its evaluation rollouts during training | Its rollouts in `platoon eval` |
|---|---|---|
| The defaults | Evaluation's defaults: one rollout per task, the filters off | The defaults |
| `train.workflow_config` | The entry's training settings, for the keys evaluation takes from training: the rollout and rollout-worker settings | The entry's `workflow_config.rollout_config` |
| A named environment's `workflow_config.rollout_config`, then the entry's `workflow_config` | The keys `eval.workflow_config` sets | The keys `inference.workflow.rollout_config` sets |
| | The entry's `eval_overrides.workflow_config` | The entry's `eval_overrides.workflow_config.rollout_config` |

- **The phase's section wins over an entry.** With `eval.workflow_config.rollout_config.max_steps: 100`, the quick
  entry's evaluations run 100 steps, not 30, and `platoon train` warns of it; its
  `eval_overrides.workflow_config.rollout_config.max_steps: 30` keeps its budget.
- **Group keys stay out of evaluation.** An entry's `group_size` and the other keys that shape groups and their
  baselines are its training's; its evaluation takes them from `eval.workflow_config` and its `eval_overrides`.
- **`platoon eval` applies an entry's rollout settings only,** and warns of the rest of an entry's training
  settings, which it does not apply.
- **The train split takes no `eval_overrides`.** `platoon eval` of `split: train` (and `platoon rollout --split
  train`) applies an entry's `workflow_config.rollout_config` beneath `inference.workflow`, but not its
  `eval_overrides`, as for the entry's other evaluation overrides.
- **A named environment** may ship the rollout settings its tasks were measured with
  (`workflow_config.rollout_config`) as its entry's lowest layer, which its evaluations inherit beneath the phase's
  section. The rest of `workflow_config`, `eval_overrides.workflow_config` and `max_staleness` are the run's to
  choose: a shipped evaluation override would win over the run's evaluation settings.
- **Overrides** reach an entry by its index, as `environments.1.max_staleness=40` or
  `environments.1.workflow_config.group_size=4`.

**What stays the run's.** An entry that sets one of these does not load (exit 78), and the message says where it goes:
depth weighting and the zero-advantage filter, which act on microbatches that mix entries' groups; straggler policies
by label (`straggler_policies`, which win over an entry's `straggler_timeout_seconds` and `straggler_quorum` for the
groups of their label); the rollout-worker settings (`rollout_isolation`, `worker_*`), since one pool of workers runs
every entry's rollouts; and `rollout_config.judges`, `model_name`, `train`, `output_dir`, `return_dict` and
`verbose`. The batch, the loss, the optimizer, `max_concurrent_rollouts`, `label_by`, `rollout_group_limits` and the
workflow class are the run's too.

**Staleness.** `train.max_staleness` stays the bound on how far rollouts run ahead of training, and the bound of each
entry that sets none. An entry's own bound may only raise it, for its own groups, as the stale filter reads it: a
lower one would have its groups drawn as far ahead as every other entry's and then dropped as stale, so loading refuses
it. Recovery pruning keeps the checkpoints within the largest bound of the run and its entries, so
`checkpoint.keep_recovery × checkpoint.recovery_every` must exceed that bound. Give a slow entry a limit on its label
too (`rollout_group_limits`): its groups count toward the rollouts in flight for as long as they run, and enough of
them slow every step to their pace; `platoon train` warns of an entry that raises its bound without one.

**Concurrency.** `train.max_concurrent_rollouts` counts rollouts: a group takes its entry's group size of them at once,
first come first served, so groups of different sizes share it and it is never exceeded. It must hold a group of the
largest entry, in training and in evaluation (`eval.max_concurrent_rollouts`). With `rollout_group_limits`, it should
also hold the capped labels' groups at their limits and a group of the largest other entry: otherwise a larger group
waits for a capped one to end, and every group behind it waits too, which `platoon config validate` and `platoon
train` warn of.

**A capped label never stalls the others.** The rows of a label at its `rollout_group_limits` wait beside the batch's
other rows rather than among them, so they never hold another label's rows back, and they do not count toward the
batches `train.max_staleness` lets the run draw ahead: it draws other rows to keep its workers busy. Each step logs
how many rows wait (`train/batch/parked_rows` and `train/batch/<column>/<label>/parked_rows`). When more than four
batches of them wait, the label's new rows are skipped, counted (`train/batch/skipped_rows` and
`train/batch/<column>/<label>/skipped_rows`) and warned of once, while a row of another label can start: its share of
the order asks for more groups than its limit runs, and the other labels' rows fill the steps. Lower its weight or
raise its limit to train them all. So a capped label trains a smaller share than its weight while its limit cannot
keep up, also before any row is skipped, and the rows still waiting when the run ends never train: the run replaces
them with rows of the batches after its last, so that its last steps do not wait for the label's groups. This applies
to every config that sets `rollout_group_limits`, with or without settings per environment.

**Custom workflows.** A `GroupRolloutWorkflow` subclass applies an entry's settings without changes: inside a group,
`self.config` holds the group's entry's settings, and
`platoon.train.tinker.workflows.group_rollout_workflow.current_entry()` names the entry. Code that reads
`self.config` from a thread of `loop.run_in_executor` gets the run's. A workflow class that is no `GroupRolloutWorkflow`
must apply them itself and declare `applies_entry_settings = True`: otherwise a config whose entries set any does not
load into `platoon train` (exit 78), rather than ignore them.

**See what each entry runs with.** `platoon config show CONFIG` prints, after the config and as comments, each entry
that sets its own settings with the value of every key, where it came from and the value that layer replaced;
`--entry NAME` prints one entry's. `platoon config validate CONFIG` adds one line per such entry with what differs
from the run, and the warnings. `platoon train` logs both, and `lock.json` records the differences
(`entry_settings`). For the config above (abridged):

```console
$ uv run platoon config show train.yaml --entry slow-tasks
environments[slow-tasks], training groups:
  group_size: 4                                                entry                    (train.workflow_config: 8)
  leave_one_out_baseline: true                                 entry                    (default: false)
  min_successful_group_size: 3                                 entry                    (train.workflow_config: 4)
  ...
  rollout_config.max_steps: null                               entry                    (train.workflow_config: 200)
  rollout_config.timeout: null                                 entry                    (train.workflow_config: 3600)
  ...
  rollout_config.inference_params.temperature: 1.0             train.workflow_config    (default: null)
  ...
  max_staleness: 48                                            entry                    (train.max_staleness: 3, which bounds how far rollouts run ahead)
environments[slow-tasks], evaluation during training:
  group_size: 1                                                eval_overrides           (eval.workflow_config: 4)
  ...
  rollout_config.max_steps: null                               entry, inherited
  rollout_config.timeout: null                                 entry, inherited
  ...
```

## Label tasks by a column

Each task of a run is a dataset row: its `task_id` and the columns its environment gives it. A mixture gives each
task the entry it belongs to, in the column `_platoon_environment`, and a Harbor dataset each task's index fields,
such as its `source_dataset` ([Index fields as dataset columns](sandboxes-and-harbor.md#index-fields-as-dataset-columns)).
`train.label_by` names the column that labels the run's tasks, by default the mixture entry. A task's label is its
value of the column as text (`true` or `false` for a boolean), and a task without the column has none. By label:

| What | Per label |
|---|---|
| Metrics | Each training step logs each label's groups, reward, steps and counts, `train/batch/<column>/<label>/{groups, task_reward, num_steps, outcome/<status>, reward_key/<name>, reward_key_mismatch, stale_groups}`, also at zero, and each evaluation during training logs them as `eval/<column>/<label>/...` ([Metrics to watch](#metrics-to-watch)). The default column logs as `environment`, so a mixture's metrics read `train/batch/environment/<name>/...`, and a single environment, whose tasks have no entry column, logs none. A `/` in a column's name or a label is written `--` |
| `train.rollout_group_limits` | The most rollout groups of that label that run at once, for example to cap a slow entry of a mixture; its groups wait for a slot before they take a rollout worker, beside the other labels' rows, which they never hold back, and past four batches of waiting rows its new rows are skipped and counted ([Settings per environment](#settings-per-environment)) |
| `straggler_policies` of `train.workflow_config`, and of `eval.workflow_config` for evaluation during training | The label's own straggler policy, `{timeout_seconds, quorum}`, in place of `straggler_timeout_seconds` and `straggler_quorum`; a null `timeout_seconds` waits for every rollout |

```yaml
train:
  label_by: suite                     # a column of the tasks' rows; default: the mixture entry
  rollout_group_limits: {slow: 4}     # at most 4 groups of the tasks labelled slow at once
  workflow_config:
    straggler_policies: {slow: {timeout_seconds: 900, quorum: 6}}
```

A column with more than 64 labels, such as `task_id`, breaks the metrics down by its 64 most frequent labels, with a
warning; the batch's own metrics count every task, and limits and straggler policies apply to every label.

`platoon train` checks the column when it lists the tasks, before anything runs: it stops with exit 78 when no
training task has the column (the default aside, which a single environment's tasks do not have), and warns when
only some have it, and of a limit or a straggler policy for a label that no task of its split has.

`label_by` was called `rollout_group_key`, and straggler policies once named a column of their own,
`workflow_config.straggler_policy_key`. Both still load until platoon 0.3.0, each with a deprecation warning, and
`platoon config validate --strict` fails on them. A workflow config that sets `straggler_policy_key` keys its
straggler policies on that column, as before; drop it to key them on `label_by`.

## Evaluate during training

A training config's `eval` section evaluates the eval split while training runs, every `eval.every` training steps; 0,
the default, evaluates nothing, and the eval split is then never listed. A config without the section does not
evaluate. The first evaluation is at step `every`. With `eval.at_start: true` the starting policy is evaluated too, at
step 0, and the first update waits until that evaluation ends, so that the held-out curve starts at the base model, or
at the weights `checkpoint.load_checkpoint_path` loads; with evaluation off, `eval.at_start` is ignored, with a
warning. An evaluation runs beside training and is logged when it ends, at the step training has reached by then, with
`eval/step`, the step it evaluated. `eval.max_concurrent_rollouts` (256 by default) counts its rollouts at once, beside
training's.

Its rollouts run as training's do: `eval.workflow_config` is laid over `train.workflow_config`.

- **From training**, unless the section sets them: the rollout settings (`rollout_config`: step and time limits,
  sampling, `extra` and the rest) and the rollout-worker settings (`rollout_isolation`, `worker_start_method` and
  the other `worker_*` keys).
- **Evaluation's own defaults**, unless the section sets them: the keys that shape groups, advantages and filters.
  Each task is rolled out once (`group_size: 1`), with `filter_errors` and `filter_zero_advantage_datums` off, no
  straggler cutoff, `min_successful_group_size: 1`, a mean baseline and no depth weighting.
- **Merging:** a key the section sets replaces training's. `rollout_config` and its `inference_params` merge key by
  key, so an evaluation that changes its temperature keeps training's completion budget; `null` unsets a value
  training sets (an `inference_params` key then takes the gateway's default, a `timeout` none); a free-form `extra`
  replaces training's whole.

```yaml
train:
  workflow_config:
    group_size: 8
    rollout_isolation: process
    rollout_config:
      max_steps: 200
      timeout: 3600
      inference_params: {temperature: 1.0, top_p: 1.0, max_completion_tokens: 49152}

eval:
  every: 25                       # every 25 steps: 200 steps, 3,600 s and 49,152 tokens, in processes
  at_start: true                  # and the starting policy, at step 0, before the first update
  max_concurrent_rollouts: 32
  workflow_config:
    rollout_config:
      inference_params: {temperature: 0.6}   # and training's top_p and max_completion_tokens
```

`platoon config show CONFIG --defaults` prints what each evaluation runs with, and `platoon rollout CONFIG --split
eval --task ID` runs one held-out task with its rollout settings. A config built in Python, rather than loaded, takes
its `eval.workflow_config` as given.

### Evaluation in a resumed run

A start that resumes the run, by hand or after a failure with `watchdog.max_restarts`, evaluates at the multiples of
`eval.every` from the step it resumes at on. Whether it evaluates that step itself depends on what the run directory's
`evaluations.jsonl` says, since a step's checkpoint is saved before the step's evaluation starts. Each evaluation adds a
line to it once its metrics are logged: the step it evaluated (`step`), the step they were logged at (`logged_at`), its
`tasks` and its `seconds`. So:

- an evaluation due at the step a start resumes at runs again only if the file does not list it: one that a crash
  cut short runs, and one that completed does not run twice;
- `eval.at_start` concerns step 0 alone. A run that resumes there, after a crash before its first step's checkpoint,
  evaluates the starting policy again, and holds its first update for it, only if no earlier start completed that
  evaluation;
- a start that resumes a run at its last step has nothing left to train: it runs the evaluation due there if a crash
  cut it short, and ends.

The steps after the one a start resumes at are trained again, so they are evaluated again, whatever the file says
of them. An evaluation that a failure cut short runs again only in a start that resumes at its step, though. Training
goes on while an evaluation runs, so a later step's checkpoint can be saved before the evaluation ends, such as one
of `checkpoint.recovery_every`: a failure after it resumes the run past the evaluation's step, from a later policy, and
that evaluation does not run again. The held-out curve then has no point at its step.

## The training service

platoon does not run the model. It talks to a service that implements the Tinker API: hosted Tinker (the
default), or a self-hosted Tinker-compatible server, for example
[miles-tinker](https://github.com/neulab/miles-tinker). The training process (the platoon client) needs no GPU; it
renders prompts, runs rollouts, builds training data and calls the service.

- **Where:** `tinker_base_url` in a training config, else `TINKER_BASE_URL`. A training config's URL serves every
  Tinker client the config opens, the policy that `platoon rollout` samples for it included. Evaluation configs
  have no such key, so `platoon eval` reads only `TINKER_BASE_URL`.
- **Credentials:** `TINKER_API_KEY`, which must start with `tml-`, as the Tinker client requires. `platoon train`
  exits 78 without a usable key, and so do `platoon eval` and `platoon rollout` when they serve a Tinker policy, each
  with one line and before it prepares, writes or reaps anything. A key that the service refuses (HTTP 401, or 403 for
  what the key may not do) ends them with one line that names the key and the service (exit 78), and `platoon eval`
  starts no rollout after it.
- **What platoon asks for:** a LoRA training client of `train.model_name` with rank `train.lora_rank` (or one
  restored from a checkpoint, below), `forward_backward` with `train.loss_fn`, `optim_step` with the Adam settings,
  and sampling clients for the current weights.
- **SDK version:** the `tinker` extra pins the Tinker SDK; [RFC 0006](../rfcs/0006-backend-profiles-and-sdk-upgrade.md)
  proposes backend profiles and the SDK upgrade.

Two deadlines guard against a stalled service: `train.forward_backward_timeout_seconds` (default 5400 s, for a
whole step's accumulated forward-backward) and `train.optim_step_timeout_seconds` (300 s). A third bounds each
sampling call: [the sampling timeout](#the-sampling-timeout).

### The sampling timeout

A sampling call that takes longer than the sampling timeout fails. A training config sets it beside the model, in
`train.sample_timeout_seconds`, for its rollouts and its evaluations alike; an evaluation config sets
`inference.sample_timeout_seconds`. A config that sets neither takes the variable `TINKER_SAMPLE_TIMEOUT_SECONDS`,
else 600 s:

```yaml
train:
  model_name: Qwen/Qwen3-4B-Instruct-2507
  renderer_name: qwen3_instruct
  sample_timeout_seconds: 1800  # a long completion on a loaded service
```

- **The key wins over the variable.** The config is the run's record (`config.resolved.yaml`), while a gateway job
  may export the variable for every client it runs, as miles-tinker's presets do. So a config that sets 900 samples
  under 900 whatever the job exported, and a launch that sets only the variable samples as it did before the key
  existed. To change the bound for one launch, override the key on the command line
  (`train.sample_timeout_seconds=3600`), which the run records.
- **One bound per run.** The trainer resolves it once, and its rollout workers (`rollout_isolation: process`) sample
  under the trainer's bound, not under their own environment's. Each process of `platoon eval` resolves the same
  config and variable.
- **The log says which.** `platoon train` and `platoon eval` log the bound and where it came from as they start,
  naming a variable that the key overrides:

  ```text
  Sampling timeout: 1800 s (train.sample_timeout_seconds, which wins over TINKER_SAMPLE_TIMEOUT_SECONDS=600)
  ```

  With the key unset, a variable that is not a positive number stops them with exit 78 before anything runs.
- **Harness clients wait longer.** Give a harness's client a request timeout longer than the sampling timeout, so
  that a slow sample ends as the gateway's error rather than as a client that gave up
  ([Harness timeouts and crashes](sandboxes-and-harbor.md#harness-timeouts-and-crashes)).
- **Hosted models** have no such key: `inference.hosted.timeout_s` bounds each of their HTTP attempts
  ([Evaluate a hosted model](evaluate-hosted-models.md)).

## Sampling defaults

Every model call goes through platoon's model gateway, which renders the messages, samples, parses the reply and
records the exact tokens for training. A call's sampling parameters come from, in order:

1. the call itself (`ctx.chat(messages, temperature=0.7)` in a function harness, or the fields of an HTTP request
   from a CLI harness);
2. the rollout's defaults, `rollout_config.inference_params` (`temperature`, `top_p`, `top_k`,
   `max_completion_tokens`), which apply to every agent of the rollout, subagents included;
3. the gateway's defaults: temperature 1.0, top_p 1.0, top_k -1 (off) and 4096 completion tokens.

With `rollout_config.sampling_precedence: config`, a policy call's own `temperature`, `top_p` and `top_k` never
apply: it samples with the rollout's, else the gateway's, and with the rollout's `max_completion_tokens` when the
config sets one ([Harness timeouts and crashes](sandboxes-and-harbor.md#harness-timeouts-and-crashes)).

Only the `inference_params` a config sets apply. A harness that sends no `max_tokens` (platoon-bash and
mini-swe-agent send none) gets `max_completion_tokens` when the config sets it and 4096 otherwise. An evaluation
during training samples with training's `inference_params` unless it sets its own, key by key
([Evaluate during training](#evaluate-during-training)):

```yaml
train:
  workflow_config:
    rollout_config:
      inference_params: {temperature: 1.0, max_completion_tokens: 2048}
```

`train.context_window_length` (or `inference.context_window_length`) turns on the context check. With it set,
the gateway clamps each call's completion budget to the tokens left in the window, and refuses a prompt that leaves
none with the API's native context-length error (HTTP 400, `context_length_exceeded` for Chat Completions), so a
harness can condense its history instead of retrying. What a harness that then crashes counts as is
[`on_context_overflow`](troubleshooting.md#context_overflow).

## Renderers

The renderer turns messages and tools into prompt tokens and parses sampled tokens back into a message with tool
calls. Training and evaluation must use the same renderer for a checkpoint, or the policy sees prompts it was not
trained on. Set it with `train.renderer_name` and `inference.renderer_name`; `renderer_kwargs` sets attributes of
the renderer (an attribute it does not have is an error).

Any [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) renderer name works, for example
`qwen3`, `qwen3_instruct`, `qwen3_disable_thinking`, `qwen3_5`, `llama3`, `deepseekv3`, `kimi_k2`, `nemotron3`
and `gpt_oss_medium_reasoning`. platoon adds `chat_template`:

| Renderer | Prompts | Tool calls parsed |
|---|---|---|
| a tinker-cookbook renderer | The cookbook's implementation of the model's format | The cookbook's parser for that format |
| `chat_template` | The model's own Hugging Face chat template (`tokenizer.apply_chat_template`), with the tools passed to it | The parser of the model's format, chosen by a round trip through the template ([Models and tool calling](models-and-tool-calling.md)) |

`chat_template` is text-only (image and video parts are refused) and stops on the `eos_token_id` of the model's
`generation_config.json`. Its `renderer_kwargs`:

| Key | Default | Effect |
|---|---|---|
| `enable_thinking` | true | Passed to the template; a request may override it through `chat_template_kwargs` |
| `preserve_thinking` | unset | Passed to the template when set; a request may override it |
| `reasoning_effort` | unset | Passed to the template when set |
| `chat_template_kwargs` | `{}` | Any other template variables |
| `tool_call_parser` | `auto` | The [tool-call parser](models-and-tool-calling.md#set-the-parser): `auto`, a parser's name, `package.module:attr` or `none` |
| `reasoning_tags` | `auto` | The reasoning tags, when the model's are not its format's: a pair such as `["<think>", "</think>"]`, or `none` |

Four names are `chat_template` with preset settings, kept so that configs written for the renderers it replaced
resolve to the same prompts:

| Alias | Preset |
|---|---|
| `qwen3_6_preserve_thinking` | `preserve_thinking: true` |
| `qwen3_6_preserve_thinking_disable_thinking` | `preserve_thinking: true`, `enable_thinking: false` |
| `qwen3_8_preserve_thinking` | `preserve_thinking: true`, `reasoning_effort: xhigh` |
| `qwen3_8_preserve_thinking_disable_thinking` | `preserve_thinking: true`, `reasoning_effort: xhigh`, `enable_thinking: false` |

[RFC 0011](../rfcs/0011-checkpoint-export-and-chat-template.md) proposes exporting checkpoints together with the
chat template they were trained with.

## Checkpoints and resume

Training saves checkpoints into the service and records each one as a row of `checkpoints.jsonl` in the run
directory, with its `state_path` (weights and optimizer, for resuming) and `sampler_path` (weights for sampling).

| Name | When |
|---|---|
| `000010` | Every `checkpoint.every` training steps (0, the default, saves none); kept |
| `recovery-000012` | At every launch, and every `checkpoint.recovery_every` steps when that is set; disposable: `checkpoint.keep_recovery` deletes all but the newest few |
| `000007_interrupted` | At the first SIGINT or SIGTERM, when a step completed since launch; weights and optimizer, kept |
| `final` | When training reaches its last step, after the evaluation due at that step |

Between full checkpoints, each step only pushes new sampling weights, which cannot be resumed from.

**Delete old recovery checkpoints.** A full checkpoint of a large model trained in full is hundreds of gigabytes, so
a long run that sets `recovery_every` can fill the service's disk. With `checkpoint.keep_recovery: N`, each time a
recovery save is recorded in `checkpoints.jsonl`, the run deletes its recovery checkpoints beyond the newest N,
weights and sampler weights, through the training service's checkpoint deletion (the Tinker SDK's
`RestClient.delete_checkpoint_from_tinker_path`). It never deletes:

- a permanent checkpoint (`000010`, `000007_interrupted`, `final`);
- the checkpoint a resume would start from;
- a recovery checkpoint this start saved within the run's largest staleness bound of the policy (`train.max_staleness`,
  or an environment's `max_staleness` above it), which rollouts may still sample from (one an earlier start saved
  serves none of this start's rollouts);
- anything the run's own `checkpoints.jsonl` does not list, such as another run's checkpoints or the one
  `load_checkpoint_path` names.

```yaml
train: {max_staleness: 3}
checkpoint: {every: 50, recovery_every: 5, keep_recovery: 2}
```

`keep_recovery` needs `train.max_staleness`, and `keep_recovery × recovery_every` must exceed the largest staleness
bound, of the run and of its entries: otherwise more recovery checkpoints than it keeps are within reach of rollouts.
A config that breaks either rule does not load (exit 78). Without `recovery_every`, the only recovery checkpoints are
the ones each launch saves, and it deletes those of earlier launches. Size the service's disk for `keep_recovery + 1`
recovery checkpoints, since a save completes before the oldest goes, besides the permanent ones; the newest
`keep_recovery` stay when the run ends.

Deletes run beside training: a step never waits for one. They go through a Tinker session of their own, since the
SDK reports a failed delete as the end of the session it ran in, which a self-hosted service may answer by releasing
that session's models. The run directory's `checkpoints_deleted.jsonl` lists what was deleted, so a resumed run goes
on where the last start left off and never asks for one twice. Each step logs `checkpoint/deleted` and
`checkpoint/delete_failures` (Tinker checkpoints, two per recovery save, since the step before),
`checkpoint/recovery_kept` (the run's recovery checkpoints left) and `checkpoint/pruning` (1 while it is on). A
failed delete is tried again after the next recovery save, and a checkpoint the service no longer has (a 404, once
the service has deleted one in this start) is not asked for again; three failures in a row, or a service that cannot
delete checkpoints, stop the deletes until the run starts again, with a warning, and training goes on
([Troubleshooting](troubleshooting.md#the-training-service)).

Pruning goes by the run directory's `checkpoints.jsonl`, so a copy of a run directory deletes the recovery checkpoints
of the run it was copied from, that run's resume point too once the copy has saved newer ones: branch a run in a new
directory with `load_checkpoint_path`, or leave `keep_recovery` unset in the copy. Point `load_checkpoint_path` at a
permanent checkpoint too: the run that saved a recovery checkpoint deletes it in time, whoever starts from it. Keep
`TINKER_API_KEY` the same for every start of a run, since a service answers 404 for a checkpoint another key saved.

**Stopping a run.** The first SIGINT or SIGTERM stops training at the next step boundary: a step that has already
applied an optimizer update finishes, and one that has not is dropped, to be run again on resume. `platoon train`
then saves `<step>_interrupted` when a step completed since launch, and exits 0. A signal during the evaluation
due at the last step ends that evaluation, and the run saves `final` as usual; a later start of the run runs that
evaluation and nothing else ([Evaluation in a resumed run](#evaluation-in-a-resumed-run)). A second signal exits at once
with code 1 and saves nothing. The interrupt checkpoint is a full save: for a large model trained in full it can take
longer than the time a scheduler leaves between SIGTERM and SIGKILL
([Run on a cluster](clusters.md#6-submit-the-client)). A save cut short records nothing in `checkpoints.jsonl`, so the
next launch resumes from the checkpoint before it. A run that is killed loses the steps since its last full checkpoint,
which `checkpoint.recovery_every` bounds.

**Resume** is automatic: start the same command again with the same `log_path`, `stats.experiment_name` and
`stats.trial_name`. platoon reads the last resumable row of `checkpoints.jsonl`, restores the weights and the
optimizer state, continues from that step, and reattaches the W&B run. It evaluates at the multiples of
`eval.every` from that step on, that step itself only if no earlier start completed its evaluation ([Evaluation in a
resumed run](#evaluation-in-a-resumed-run)). To start over, use a new `stats.trial_name`.

**What a resume must not change.** A run directory holds one run. Its first start writes `identity.json` there:
the model, the renderer, and for each environment entry with a hold-out (a Harbor entry's `service_kwargs.holdout`,
[One dataset, one environment](sandboxes-and-harbor.md#one-dataset-one-environment)) the rule, the dataset and every
held-out task. A start that resumes compares its own, once it has listed its tasks and before it opens the training
service, and stops with exit 78 when:

- the model (`train.model_name`) or the renderer (`train.renderer_name`) is another;
- a task the run held out would now train, as a changed `holdout.seed`, `fraction` or `group_by` would make it;
- a task is now held out that was not, a hold-out added to a running run included.

The message names the entry, the counts and a few task ids, and the start's `lock.json` records them under
`resume_refused`. Like any start, the refused one has already replaced the run directory's `config.resolved.yaml` and
`lock.json` with its own; the run's next start writes them again. A changed refusal moves no held-out task: such a
resume goes on, and logs the held-out tasks that left the evaluation or joined it again, and those that only left the
selection. A start without a checkpoint writes `identity.json` afresh. To change any of these, start a new run under
another `stats.trial_name`.

When training evaluates, it also warns once per environment entry whose evaluation lists tasks that training draws
from: without a hold-out or a split manifest, every split lists every task, so those numbers measure tasks training
sees.

**Start from trained weights** with `checkpoint.load_checkpoint_path: tinker://...`: it loads the weights with a
fresh optimizer, and applies only when the run directory has no checkpoint of its own yet.

**Evaluate a checkpoint** by passing its `sampler_path` as `inference.model_path` (or `--model-path` to
`platoon rollout`).

**Hangs:** a watchdog (`watchdog.timeout_seconds`, default 600) exits the trainer with `watchdog.exit_code`
(default 2) when the training loop makes no progress. With `watchdog.max_restarts`, `platoon train` then starts
training again from the last checkpoint ([Long runs](#long-runs)). Set `checkpoint.recovery_every` to bound how many
steps a crash can cost.

## Long runs

A run of hours or days meets failures that starting again fixes: a training service that stalls or restarts, a
crash, a node that runs out of memory. With `watchdog.max_restarts`, `platoon train` starts training again after
them, from the newest checkpoint, without a new job; and every run reaps the sandboxes its dead processes leave.

```yaml
train: {max_staleness: 3}
watchdog: {timeout_seconds: 7200, max_restarts: 20}
checkpoint: {every: 50, recovery_every: 5, keep_recovery: 2}
```

**Restarts.** With `watchdog.max_restarts` above 0 (default 0), `platoon train` trains in a child process, a
*start*, and starts training again after any exit but these:

- 0: training finished, or a signal stopped it at a step boundary;
- 78: a configuration error, which no restart fixes;
- an interrupt: exit code 130 or 143, or a SIGINT or SIGTERM that `platoon train` itself got.

So the watchdog's exit, a crash and a training service that stalls (a deadline error such as `forward_backward
(microbatch 0) timed out`, exit 1) or dies all restart. Each start resumes from the newest checkpoint, as any start of
the run does, and evaluates as any resume does: the evaluation of the step it resumes at runs again if the failure cut
it short, and not if it completed ([Evaluation in a resumed run](#evaluation-in-a-resumed-run)). Set
`checkpoint.recovery_every` to bound the steps a failure costs. Each start also saves a recovery checkpoint at launch,
so set `checkpoint.keep_recovery` too, which deletes the older ones ([Checkpoints and resume](#checkpoints-and-resume)).
`platoon train` stops restarting after `max_restarts` restarts, or after three starts in a row that saved no new
checkpoint, which is what a failure that every start meets looks like (the recovery checkpoint each start saves at
launch does not count), and exits with the last start's exit code.

**No delay between starts.** A start asks the training service for its training client until the service accepts
it, for up to 20 minutes, and logs each refusal (`The training service refused the training client (attempt 2,
...)`). That covers a service that is restarting, and one that holds the session of the start that died until the
session expires, as a miles-tinker server that trains all weights does with the one model it trains at a time. A
refusal that waiting cannot change (401, 403, 404 or 422) ends the start at once.

**Signals.** The first SIGINT or SIGTERM stops the running start at its next step boundary, as it stops `platoon
train` without restarts, and no start follows; a second one ends the start at once. A terminal's Ctrl+C, or a
scheduler's SIGTERM, that reaches every process of the job counts once: `platoon train` passes the signal on to the
start as SIGUSR2, which the trainer never takes for a second signal (and a second one as SIGUSR1, which ends it).

**Sandboxes that a run leaves.** Every sandbox of a `platoon train` or `platoon eval` run carries the label
`platoon.run: <run directory>`, in the run's own process and in its rollout workers. A worker that is killed, or a
trainer that ends at once (the watchdog, a second signal, SIGKILL), leaves its sandboxes running. So:

- `platoon train` reaps the sandboxes labelled with its run when it starts, before each restart and when it exits,
  and first kills the rollout workers whose trainer is gone;
- `platoon eval` reaps those labelled with its output directory when it exits;
- `platoon sandbox reap RUN_DIR` reaps those of a run whose process was killed, such as a job that ran out of time.

Each logs how many it found, for example `Reaped 4 sandbox(es) labelled platoon.run=/runs/exp/trial before restart 1
(singularity: 4)`. They go through the sandbox providers that the run's `config.resolved.yaml` names, each Harbor
entry's `service_kwargs.provider` (`docker` by default) with its `provider_options`, and take only the sandboxes
labelled with that exact run directory, so other runs on the node keep theirs. A provider leaves alone the sandboxes
of a process that still runs, except Docker, which cannot tell whose a container is. So the command that runs a run
records itself in the run directory, in `process.json`: its host, pid and start time, and those of the start it runs.
`platoon train` and `platoon eval` refuse a run directory that another of their processes still runs on this host
(exit 78), and `platoon sandbox reap` refuses (exit 1) a run whose process still runs, or one that last ran on another
host, which holds its sandboxes: run it there.

`python -m platoon.train.tinker.restart_wrapper`, which restarted the command on the watchdog's exit code only, is
deprecated: it still works, with a warning, until platoon 0.3.0 removes it. Set `watchdog.max_restarts` instead.

## Metrics to watch

Each step prints its metrics and logs them to W&B. Start with these:

| Metric | Healthy | When it is not |
|---|---|---|
| `train/batch/task_reward/avg` | Rises over steps | Flat at 0 or 1: groups carry no signal; check the task difficulty and `group_size` |
| `train/batch/task_reward_valid/avg` | Close to 1 | Rollouts are failing to measure: see [Troubleshooting](troubleshooting.md) and the harness logs |
| `train/batch/outcome/<status>` | Nearly all `scored` | The step's rollouts counted by outcome status: `scored`, `invalid`, `error`, `timeout`, `cancelled`, and `none` for a rollout that recorded none: it raised, its worker died, or it was cancelled as a straggler. One whose environment could not open its session, such as a sandbox that would not start, counts as `error`. Many of any other status: see [Troubleshooting](troubleshooting.md) |
| `train/batch/<column>/<label>/task_reward/avg`, `.../outcome/<status>`, `.../groups` | Each label as healthy as the batch | Each label's own groups, reward and outcome counts, logged every step ([Label tasks by a column](#label-tasks-by-a-column)): in a mixture, each entry's (`train/batch/environment/<name>/...`); with `train.label_by: source_dataset`, each source's. A label the step took no group of logs zero groups and no reward. One that lags or fails alone: look at its tasks and its environment |
| `train/batch/reward_key/<name>` | Each name's share steady | The step's rollouts counted by the reward key they trained, the name their grader reported the reward under: `reward` for most environments, one of a Harbor reward contract's listed names ([Rewards](sandboxes-and-harbor.md#rewards-the-reward-contract-and-log-classifiers)). A rollout platoon scored 0 itself, or whose outcome was not scored, trained none |
| `train/batch/reward_key_mismatch` | 0 | Groups whose rollouts trained different reward keys, which did not train: the reward files a task's tests wrote differed between its rollouts |
| `eval/rollout_completion/task_reward/avg` | Rises at each evaluation, from the starting policy's at `eval/step` 0 with `eval.at_start` | Training reward rises but this does not: overfitting, or a train/eval mismatch |
| `eval/<column>/<label>/task_reward/avg` | Each label rises at each evaluation | The pooled evaluation reward moves with the mix of labels: compare each label's own curve, such as each source's of a held-out split |
| `train/batch/num_steps/avg`, `train/batch/num_output_tokens/avg` | Stable or falling | Runaway growth: agents loop or ramble; lower `max_steps` or `max_completion_tokens` |
| `train/optim/kl_sample_train_v1`, `train/optim/importance_weight_mean` | KL near 0, weight near 1 | Sampling and training disagree: renderer mismatch, or rollouts too stale (`max_staleness`) |
| `train/optim/clip_frac_total` | Small | Most tokens clipped: the update is too large for the loss's thresholds |
| `train/batch/skipped_optimizer` | 0 | 1 means the step took no optimizer update and the policy kept its version: its objective was zero (every advantage exactly zero, say every group's rollouts scored alike) or it had no datums. `filter_zero_advantage_datums` does not change this; it only drops zero-advantage datums from microbatches that also hold nonzero ones |
| `train/stale_rollouts`, `train/batch/<column>/<label>/stale_groups` | 0 or rare | Rollouts older than `max_staleness` were dropped; by label, which one's groups grow too old: raise its entry's bound ([Settings per environment](#settings-per-environment)) |
| `train/batch/skipped_rows`, `train/batch/<column>/<label>/skipped_rows` | 0 | Rows of a label at its `rollout_group_limits` were skipped, four batches of them waiting already: lower its weight or raise its limit |
| `train/batch/parked_rows`, `train/batch/<column>/<label>/parked_rows` | Small, steady | Rows of a label at its `rollout_group_limits` that wait for a slot: a count that grows, step after step, is a label whose share of the order its limit cannot serve, so it trains less than its weight; those still waiting when the run ends never train |
| `rollout_client/trainer_queue_wait_seconds_total` | Small next to `timing/batch_total` | The trainer waits on rollouts: raise rollout concurrency or speed up the environment |
| `rollout_client/cpu_saturation` | Below 1 | The client machine is out of CPU: use fewer concurrent rollouts or a bigger node |
| `rollout_client/agent_step_tool_fraction` | Depends on the environment | Most of each step is spent in tools and the environment, not the model |

A step's `batch/*` metrics, each label's `batch/<column>/<label>/*` included, cover the groups the step trained on,
leaving out groups dropped as stale (`max_staleness`), which `stale_groups` counts. A step's rollout and credit metrics
(`rollout_completion/*`, `credit/*`, `workload/*`) cover every group that step consumed. With rollouts running
ahead of training, groups of the next step finish while a step trains; their statistics are logged with the step
that consumes them. The exceptions are
`rollout_completion/failed_rollouts` and a failed group's `rollout_completion/rollout_time`, which are logged
when the group fails.

An evaluation runs beside training, which goes on meanwhile, and its metrics are logged when it ends, at the step
training has reached by then; `eval/step` says which step it started at. Its rollouts sample the policy that is
current when they run, so a long evaluation measures the policy of `eval/step` and the ones training reaches
meanwhile. The starting policy's evaluation (`eval/step` 0, with `eval.at_start`) is the exception: the run applies its
first update only once that evaluation has ended, so the held-out curve starts at the base model.

`platoon eval`'s report has the matching numbers per task (`reports/task_results.jsonl`).

## Next steps

- [Sandboxes and Harbor tasks](sandboxes-and-harbor.md): train and evaluate harnesses inside task containers.
- [Run on a cluster](clusters.md): the training service, environment servers and the client as scheduler jobs.
- [Troubleshooting](troubleshooting.md): what failed rollouts and error messages mean.
- [Architecture](../architecture.md): how the gateway, environments and harnesses fit together.
- [Renderers and tool-call parsers](../extending/renderers.md) and [Hooks, telemetry and
  metrics](../extending/hooks-and-metrics.md): a renderer or parser of your own, and metrics from your own code.
