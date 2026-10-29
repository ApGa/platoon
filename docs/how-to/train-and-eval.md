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
| `platoon tasks show\|export RUN_DIR` | A training run's per-task statistics: each task's groups, passes and expected pass rates ([Per-task statistics](#per-task-statistics)) |

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
| `train.stale_filter` | `group` | What a too-old group loses: `group`, the whole group, judged by its oldest datum; `datum` (experimental), only its datums beyond the bound, each judged by the oldest version its own model calls sampled ([Per-datum stale masks](#per-datum-stale-masks-experimental)) |
| `train.max_concurrent_rollouts` | `batch_size × group_size` | Rollouts in flight at once, the permits the groups share: each rollout holds one until it ends, or under `permit_release: group` a group holds its size of them until it ends (`eval.max_concurrent_rollouts`, 256 by default, for an evaluation) |
| `train.permit_release` | `rollout` | When permits come back: `rollout`, each as its rollout ends, so the next group starts on them; `group`, the default before 0.4.0, a group's all at once when its workflow returns ([Rollout permits](#rollout-permits)) |
| `train.workflow_config.rollout_config.max_steps`, `.timeout` | none | Step budget per agent, and time limit of each rollout's root agent in seconds; without one, a harness stops at its own limits (its spec's `timeout_s`, a Harbor task's `[agent] timeout_sec`), which also set a Harbor rollout's deadline ([Harness timeouts](sandboxes-and-harbor.md#harness-timeouts-and-crashes)) |
| `train.workflow_config.rollout_isolation` | `inprocess` | `process` runs each rollout in its own killable worker process, which sees no GPU (its sandboxes get theirs from their provider) |
| `train.workflow_config.worker_start_method` | `forkserver` | How a worker process starts: forked from a process that imported the worker's stack once (seconds), or `exec`, a new interpreter that imports it (17-24 s, about a minute with 64 at once) |
| `eval.every` | 0 | Evaluate during training every N training steps; 0 turns it off, and the eval split is then never listed. Evaluation rolls out as training does ([Evaluate during training](#evaluate-during-training)) |
| `eval.strategy`, `checkpoint.strategy` | unset: steps | The unit of `every`: `step`, a training step, or `epoch`, a pass over the training tasks; a section that sets it and leaves `every` out triggers every 1 of its units ([Cadences in steps or epochs](#cadences-in-steps-or-epochs)) |
| `eval.at_start` | false | Also evaluate the starting policy, at step 0, before the first update: the point the held-out curve starts from |

A run writes everything under `<log_path>/<stats.experiment_name>/<stats.trial_name>/`. A config without `log_path`
writes under the site profile's `runs`, with `stats.experiment_name` named after the config file unless it names one
([Run on a cluster](clusters.md#4-write-the-sites-profile-once)); a relative `log_path` is relative to the working
directory. The run directory holds the run record
(`config.resolved.yaml`, `lock.json`, and `process.json`, the process that runs it), `checkpoints.jsonl`,
`evaluations.jsonl` (each evaluation during training that completed), `tasks/` (the per-task statistics each full
checkpoint saves, and their event log: [Per-task statistics](#per-task-statistics)), and
`rollouts/train/<policy version>/` and `rollouts/eval/<policy version>/` with each rollout's harness logs
(`harness/`) and viewer event files (`events/`).
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
- The order follows from the tasks, the weights, the seeds and `train.batch_size`, and each full checkpoint saves
  where it is, so a resumed run goes on with it where the checkpoint left it, and rolls out again only the groups that
  had not settled ([Checkpoints and resume](#checkpoints-and-resume)).

A single environment's tasks are drawn in shuffled passes too (seed 42): an epoch is the whole batches they fill,
and the tasks left over start the next epoch. `train.task_sampler: weighted` draws them as a mixture of one instead,
with the entry's `seed` and the last batch filled from the next pass.

`train.task_sampler` chooses how a run draws its training tasks:

| `train.task_sampler` | A mixture | A single environment |
|---|---|---|
| `auto` (default) | The weighted fair order above | Shuffled passes by the dataloader (seed 42), each epoch the whole batches they fill |
| `weighted` | The weighted fair order above | The weighted order, as a mixture of one |
| `uniform` | Shuffled passes over every task of every entry together (seed 42), with no entry slots: an entry's share of the draws is its share of the tasks, and a batch holds whatever the shuffle gives it | The batches `auto` draws, in epochs that hold every task |
| `pass_rate` (experimental) | Each group's task chosen among every task of the run by its pass rate, when the group can start ([below](#choose-tasks-by-their-pass-rates-experimental)), with no entry slots | The same |
| A registered name, or `module:Class` | A task sampler of your own ([Write a task sampler](../extending/task-samplers.md)) | The same |

`uniform` draws every training task of the run equally often, as `weight: tasks` does, but without the weighted
order's balance: the entries' shares hold over the run, not batch by batch. It reads no entry's weight or seed: an
entry that sets a numeric `weight` does not load with it (exit 78), `weight: tasks`, the default, is accepted, and an
entry's `seed` changes nothing. An epoch is as many tasks as the run has, filled to whole batches from the next pass.

**Each entry's own sampler.** Under `weighted` (and `auto` for a mixture), the run's sampler hands each slot to an
entry, and the entry's sampler fills it: its `task_sampler` (`uniform` by default: the entry's tasks in shuffled
passes, seeded by its `seed`), with its `task_sampler_options`. So each entry keeps its share of the slots whatever its
sampler does with them. A single environment with a `task_sampler` of its own is drawn as a mixture of one.

```yaml
train:
  task_sampler: weighted
environments:
  - name: hard
    service: harbor
    service_kwargs: {dataset: hf://my-org/hard-tasks@3f1c0e9b6d2a4c8e9f0a1b2c3d4e5f6a7b8c9d0e}
    harness: openhands
    task_sampler: pass_rate        # experimental: the tasks whose next group is likely to train
    task_sampler_options: {explore: 0.2}
  - name: easy                     # uniform
    service: python://my_project.math:MathEnvironment
    harness: platoon-react
```

| `train.task_sampler` | An entry's `task_sampler` | An entry's numeric `weight` |
|---|---|---|
| `auto`, `weighted` | `uniform` (default), `pass_rate`, a registered name, `module:Class` | its share of the slots |
| `uniform`, `pass_rate` | refused: they hand no slots to the entries | refused: they draw among every task of the run |
| a custom run sampler | allowed when it hands slots (`entry_slots`), else refused | its own `check` decides |

An entry cannot name a run's sampler (`auto`, `weighted`), and options for a sampler that takes none (`uniform`,
`weighted`, `auto`) are refused; each refusal names its key, in the same report as the file's other problems
(`platoon config validate`, exit 78). A named environment (`env:`) cannot set `task_sampler` or
`task_sampler_options`: how a run draws its tasks is the run's choice. `lock.json` records every sampler a run uses,
with its options, defaults included, and the checkpoints' sampling fingerprint holds the samplers platoon 0.3.0 did
not have, so a resume that changes one says so ([Checkpoints and resume](#checkpoints-and-resume)).

It was called `train.task_order` before platoon 0.3.0, and platoon 0.4.0 removed that name: a config that still sets
it does not load (exit 78), with `train.task_order: unknown key; platoon 0.4.0 removed it; use train.task_sampler`, and
neither does a config copied from platoon 0.2.0's `config show --defaults`, which spells `task_order: auto`. Set
`task_sampler` instead: it takes the old name's values, `auto` and `weighted`, which draw as they did.

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

An entry whose sampler is not the default says so in its line: `environments[hard]: 475 tasks, weight tasks, 95.6% of
the slots, each filled by pass_rate (experimental) when its group can start: ...`; a run-level `pass_rate` in the first
line: `training tasks: 497, drawn by pass_rate (experimental) over every entry's tasks, in epochs of 512 ...`.

### Choose tasks by their pass rates (experimental)

`task_sampler: pass_rate` on an entry chooses, for each of its slots, the task whose next group is most likely to mix
passes and failures, since only such a group trains. It reads each task's posterior pass rate from the run's per-task
statistics ([Per-task statistics](#per-task-statistics): decayed by policy version with a half-life of 32, over a
prior of weight 2 at the entry's pass rate), draws a pass rate for each task from it and takes the task whose group of
`G` would most likely mix outcomes (Thompson sampling). A fifth of its draws go to the task it drew least recently, a
task it has never seen waits for its first result before it is drawn again, and no task is drawn more than three times
in a window of as many draws as the entry has tasks. It decides when the entry can start a group, among the tasks
whose label has room, so a decision never waits a group's duration for its permits.

| `task_sampler_options` | Default | What it does |
|---|---|---|
| `explore` | 0.2 | the share of draws that go to the eligible task drawn least recently |
| `max_draws_per_epoch` | 3 | the most draws of a task in a window of as many draws as the pool has tasks (null: no cap); lifted for a draw where every task is at it |
| `max_outstanding_per_task` | null | a task with this many draws not yet trained is not drawn |
| `candidates` | 4096 | at most this many tasks are scored by a draw, a seeded sample beyond |
| `seed` | 0 | added to the entry's `seed` |

The metrics to watch, under `train/sampler/environment/<entry>/` (`train/sampler/` for a run-level `pass_rate`):

- `informative_share`: of the groups a step consumed that it drew, the share that trained. Compare it with the share
  of informative groups (`train/batch/environment/<entry>/informative_groups` over the entry's groups) of a run with
  the entry on `uniform`: that is what it gains. `predicted_informative`, the chance of training it predicted for the
  same groups, runs well above it even when all is well: its prior, of weight 2 at the entry's pass rate, is far
  flatter than the pass rates of real tasks, most of which every group passes or every group fails, so it overrates
  the tasks it has little evidence of (1.4 to 2.6 times the share realized in simulations at fixed pass rates). Read
  how the gap moves, not the gap;
- `top10_share` (the share of the window's draws that went to the most-drawn tenth of the tasks: 0.1 is uniform's, about
  0.3 the cap's), `eligible_share` (near 0: label limits or the rules leave it no choice), `cap_lifted`, `prior_share`
  and `evidence_age` (evidence older than the half-life: consider `uniform` for the entry);
- `draws/explore`, `draws/exploit`, `draws/fallback` (a sampler that failed, which should stay 0), `decision_ms` and
  `decision_ms_max`; and the statistics' `tasks/environment/<entry>/coverage`.

Its limits:

- **It is experimental.** It becomes the recommendation only after a GPU comparison with `uniform` shows more
  informative groups at an unchanged held-out pass@1. Held-out evaluation decides: under `pass_rate`, a within-run
  `pass_at_1` averages the tasks it chose to see.
- **Slow entries keep `uniform`.** An entry whose groups take many steps gives each task too little evidence to learn
  from; keep it on `uniform` beside the adaptive ones.
- **A group size of 1** never mixes outcomes, so `pass_rate` refuses such an entry.
- **A decision costs time in proportion to the entry's tasks**: about 1 ms at 475 tasks, 7 ms at 4,096 and
  15 ms at 19,560, on the event loop that starts the run's groups and with nothing else running meanwhile. A step
  that starts makes its decisions back to back, up to a batch of them (up to `max_staleness + 1` batches when a run
  starts), so at 19,560 tasks a batch of 32 holds the loop about 0.4 s at each step's start. `candidates` bounds
  the tasks a decision scores (lower it to shorten a decision on a large pool), but not the pass over every task that
  finds the eligible ones (`decision_ms` and `decision_ms_max` show the cost).
- **Run-level `pass_rate`** (`train.task_sampler: pass_rate`) chooses among every task of the run, entries competing by
  score, so it takes no numeric weights and no entry samplers; at the same pass rate a task of larger groups scores
  higher, so an entry of larger groups draws more. It holds the permits of the largest group of the entries that can
  start while it decides. A capped label gets no held slot of the horizon under it, so at `max_staleness >= 1` its
  next group may wait for the next step: give a capped environment its own entry sampler under `weighted` instead.
- **Repeated task ids** in an entry's listing: it draws by task id, so only the first row of a repeated id runs, and
  `platoon train` warns.

### Train only on groups that carry signal (experimental)

A group whose rollouts all scored alike has zero advantages: it trains nothing, yet in a batch it takes a place that a
group with signal could have. `train.group_admission: informative` decides, when each training group settles, whether it
counts toward the batch: a group whose workflow's verdict says it trains nothing is rejected, its place is freed (at
once, for a group whose rollouts scored alike), and another group starts in its place while the trainer trains (refill).
A step then trains the first `batch_size` admitted groups it dequeues, and stops waiting once it has counted
`refill.max_groups_per_step` times `batch_size` groups, training what it admitted:

```yaml
train:
  batch_size: 32
  group_admission: informative               # all, the default, admits every group, as before
  group_admission_options: {band: [0.2, 0.8]}   # optional: also reject groups whose pass rate is outside it
  refill: {max_groups_per_step: 3.0}         # the default: a step counts at most 96 groups
  num_minibatches: 1                         # required beside a rule
```

| Key | Default | What it does |
|---|---|---|
| `train.group_admission` | `all` | `all` admits every group; `informative` rejects the groups that train nothing; a name an entry's `package` registers with `@group_admission`, or `module:Class` ([Write a group admission rule](../extending/group-admission.md)) |
| `train.group_admission_options` | `{}` | the rule's options; `informative` takes `band: [lo, hi]`, two numbers `0 <= lo < hi <= 1` |
| `train.refill.max_groups_per_step` | 3.0 | a finite number of at least 1: a step counts at most `ceil(value × batch_size)` groups. There is no null: a pool the policy has solved would never close a step; set a large number instead |
| `environments[].group_admission`, `group_admission_options` | null: the run's | an entry's own rule, with its own options (none set: `{}`); options without a rule do not load |

**What `informative` rejects**, by its reason: a group whose verdict is `uniform` (`uniform_pass` when every valid
rollout passed, `uniform_fail` when none did, `uniform_mixed` otherwise), a group that carries no datums (its verdict:
`reward_key_mismatch`, `no_results`, `below_min`, `no_completed_root`, `no_datums`, or `failed`), and, with a band, a
group whose pass rate is below or above it (`below_band`, `above_band`: a group can train and still pass every rollout
by success, say rewards of 1.0 and 0.98 under a threshold of 0.9). It admits the rest, a group whose output carries no
verdict included (reason `unknown`, with a warning once): a workflow subclass that returns plain lists of datums then
trains as under `all`. Loading refuses `informative` beside a workflow class that gives no verdicts at all (no
`GroupRolloutWorkflow`, and no `gives_verdicts = True`).

**What a step trains.** It trains the first `batch_size` admitted groups it dequeues that are not stale; a stale group,
and a group whose workflow failed, count toward its cap, not toward the batch, so the step takes another in their place.
Each group it dequeues counts toward the cap, and so does the marker of each group rejected while that step was current;
the markers of groups rejected while an earlier step trained pass over without counting, so that the work done while the
trainer trained never closes the next step early. A short step trains what it admitted, and a step that admitted nothing
takes no optimizer update (`train/batch/skipped_optimizer` 1) and the run goes on: the run still ends after its planned
steps. A rejected group that carries datums frees its slot of the horizon at once; a failed group, or one that carried
no datum (an outage can end such groups at once), keeps it until the step that dequeues its marker, as under `all`, so
that such groups never start faster than steps take them. At `max_staleness: 0`, and on the run's last step, no group
starts that the step could not count, so none of those is left running when it closes; but a step can close short of its
batch, so up to `batch_size` groups started before the last step can still be running or waiting when the run ends, and
the final checkpoint keeps them. Once any entry's rule is not `all`, every entry's groups are counted so: a stale or
failed group of an entry at `all` no longer counts toward the batch either. Evaluation never meets a rule.

**What it costs.** Each group is rolled out whether it is admitted or not, so the rollouts per informative group are
one over the informative share, with or without a rule: all mode rolls out as many and trains the rest's zeros. What
changes is the rollouts a step takes. With groups informative a share `p` of the time, `B = 32` and the default cap:

| `p` | `all`: informative groups a step | `informative`, cap 3.0: admitted a step (5-95%) | Short steps | Groups counted a step | Without a cap |
|---|---|---|---|---|---|
| 0.13 | 4.2 | 12.5 (7-18) | 100% | 96 | 246 |
| 0.15 | 4.8 | 14.4 (9-20) | 100% | 96 | 213 |
| 0.18 | 5.8 | 17.3 (11-24) | 100% | 96 | 178 |
| 0.21 | 6.7 | 20.2 (14-27) | 99.7% | 96 | 152 |
| 0.33 | 10.7 | 30.2 (25-32) | 46% | 90 | 96 |
| 0.50 | 16.0 | 32.0 | 0% | 64 | 64 |

A step reaches `batch_size` 95% of the time from a cap near `1.3 / p`; resampling "until the batch is full" is a large
cap set by hand, and a step that keeps waiting logs a warning each time its count passes another 10 batches. What to
expect, from an event-driven model of the loop:

- **Where rollouts bound the run** (the groups at once, not the trainer, set the pace), a step takes about three times
  as long at `p` near 0.13, and the run trains as many informative groups an hour as under `all`, spread over a third
  as many steps.
- **Where the trainer bounds it**, refill fills the rollout capacity the trainer leaves idle: more informative groups an
  hour (25 to 153 at `p = 0.13`, `max_staleness: 1`, in the model, at six times the rollouts an hour), which a run that
  pays for its rollouts pays. The cap bounds what a step counts, not what starts while the trainer trains: during the
  update of a step that admitted `a` groups, the horizon lets about `((max_staleness + 1) × batch_size - a) / p` groups
  start (and the rollout permits as many as they hold), many times the cap where `p` is small. A capped label's backlog
  and an adaptive entry's queue of slots fill sooner there.
- **At `max_staleness: 0`** refill gains where few groups are informative and loses where many are (35 to 25
  informative groups an hour at `p = 0.33` in the model): the end of each step runs only the few groups it still needs.
- **Settings counted in steps stretch**: `max_training_steps`, `num_epochs` (the same number of steps, each drawing more
  positions), `eval.every`, `checkpoint.every` and `checkpoint.recovery_every` each cover more rollouts and hours: a
  step waits for up to the cap's multiple of the groups, and where the trainer bounds the run more groups still start
  while it trains. A crash loses as much more work.
- **Staleness.** A rejected group is never stale: it never trains. Where the trainer is the bottleneck, a short step
  lets up to `batch_size` minus its admitted groups start that only a step further ahead could consume, and the stale
  filter drops those that wait too long: at `max_staleness: 1`, up to `batch_size` minus the admitted groups of each
  short step (0.1 to 0.2 a step in the model at `batch_size: 32`, more at small batches; `train/admission/stale` counts
  them); elsewhere fewer groups go stale than under `all`, since a rejected group no longer holds a slot while it waits.

**The step's gradient.** In all mode a uniform group's tokens stay in each microbatch's normalization, so a step's
gradient is roughly the informative share times the mean over informative tokens. Under `informative` the denominator
counts admitted groups only: the raw gradient grows by about one over the informative share, and it is the mean over
the step's admitted groups. Adam's update does not change with a constant scale, so keep the learning rate; but the
update follows a mean over more groups, gradient clipping (`grad_clip_norm`) engages on the larger raw norm, and a run
that turns a rule on at a resume takes larger first updates while Adam's second moment catches up: turn it on at step
0. A short step's gradient is the mean over the groups it admitted; with `num_microbatches` above 1, its partial last
microbatch is weighed by its share of groups (as if the missing groups were uniform), so a step that admitted `a` of
`B` groups, `a >= B / M`, has the gradient of `a × M / B` full microbatches. `num_minibatches` must be 1 beside a rule:
a short step's last minibatch would take a full optimizer step on a few groups. Under a loss whose objective is not
zero at zero advantages (`loss_fn_config` keys besides cispo's, ppo's or importance_sampling's clip thresholds: a KL
or an entropy term), a uniform group's datums train those terms, and rejecting it removes its share of them: loading
warns. To compare a run with a rule against one without, compare them at matched rollouts or hours, never at matched
steps, and watch `train/optim/grad_norm`, the KL and importance-weight metrics, `train/admission/short`,
`train/admission/stale` and `train/admission/draws_per_admitted`, beside the held-out pass@1.

**Per-entry rules.** An entry may set its own rule, such as `informative` for an environment whose groups are mostly
uniform beside `all` for one whose groups are too rare and costly to discard. Under `weighted` an entry's share is a
share of the draws, and a rejection does not refund it, so the trained mix shifts toward the entries that reject less:
an entry at 87% of the draws under `informative` with 15% of its groups informative, beside one at 13% under `all`,
admits 0.87 × 0.15 = 0.13 of the draws against 0.13, about half the trained groups each. Use the same rule on every
entry, or scale an entry's weight by its admission rate (which drifts as the policy learns), and watch
`train/admission/<column>/<label>/admitted`. `train.refill` is the run's, since a step mixes entries; a named
environment (`env:`) cannot set a rule.

**With an adaptive sampler.** `pass_rate` sees every settled group, admitted or rejected, and a rejection that frees a
slot leads to a decision that reads it at once; they pair, the sampler predicting before the rollout and the rule
catching what it missed. Refill makes the decisions when rejections come, which the order the groups complete in
decides, so a `max_staleness: 0` run with an adaptive sampler and a rule does not replay decision for decision.

**What each step logs**, only in a run with a rule, under `train/admission/` and per label under
`train/admission/<column>/<label>/` (each label at zero when it had no group):

| Metric | What it counts |
|---|---|
| `groups_admitted` (per label: `admitted`) | the groups the step trained: admitted and not stale |
| `admitted/<reason>` | of those, the ones admitted with another reason than `admitted`, such as `unknown` |
| `rejected`, `rejected/<reason>` | the rejected groups' markers the step dequeued, counted or passed over, by reason |
| `stale`, `failed` | admitted groups the stale filter dropped; failed groups |
| `informative_rate` | of the groups with an output the step dequeued (admitted, stale, rejected), the share whose verdict is `trained` |
| `counted` | what the step counted toward its cap |
| `short` | 1 when it admitted fewer than `batch_size` |
| `draws_per_admitted` | the draws since the last step over the groups it admitted (not logged for one that admitted none) |
| `errors` | decisions that failed open (a rule that raised or answered outside the contract): 0 in a healthy run |

`train/sampler/.../predicted_informative` and `informative_share` count, in a run with a rule, the groups a step trained
and the rejected groups' markers it dequeued, so that `informative_share` stays the share that would have trained.

**What a resume does.** A rejection is final when its group settles: checkpoints keep only admitted groups, and a resume
from a checkpoint saved after the rejection never draws its position again. A resume that turns a rule on or off,
changes one or its options, or changes the cap warns, names each change in `lock.json` (`sampling_changed`), and goes
on; the groups a checkpoint kept were decided when they settled and train as they are. A crash between a rejection and
the next full save loses the rejection with it, and so does a stop (Ctrl+C, a SIGTERM) before the start's first step
completes, which saves no checkpoint: the resume draws the position again ([Checkpoints and
resume](#checkpoints-and-resume)).

`platoon config show` prints each entry's rule in its block of effective settings, and the sampling plan a line such as
`group admission (experimental): informative for hard, all for easy; a step trains the first 32 admitted groups of at
most 96 (refill.max_groups_per_step 3.0)`. It becomes the recommendation only after a GPU comparison shows a higher
held-out pass@1 at matched rollouts.

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
| `group_admission`, `group_admission_options` | Which of its settled groups count toward the batch (experimental): its own rule, with its own options ([Train only on groups that carry signal](#train-only-on-groups-that-carry-signal-experimental)) | the run's |

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
too (`rollout_group_limits`): its groups count toward the horizon below for as long as they run, and enough of them
slow every step to their pace; `platoon train` warns of an entry that raises its bound without one.

**The horizon.** A group starts only when it can: within the horizon, with rollout permits (a group of
`GroupRolloutWorkflow` with the first of its group's, any other with its whole group's, as every group under
`train.permit_release: group`), and with room in its label's limit. With `train.max_staleness: k`, the horizon holds
`(k + 1) × train.batch_size` groups: those started and not yet taken by a step, and those the current step has taken; a
step that starts, its policy published, frees one batch more. So rollouts run at most `k` steps ahead of training, as
before, and with `k = 0` each step trains exactly the groups of its batch, so it waits for each of them, a slow entry's
too, unless a label's limit holds one of them back (below). Without a bound the rollout permits alone limit the groups
that run, and in any case the run starts no group past those its last step trains (under a group admission rule, a step
can close short of its batch: up to `batch_size` groups may then outlive the run).

**Concurrency.** `train.max_concurrent_rollouts` counts rollouts: a group reserves its entry's group size of them, first
come first served, and starts once they are granted, so groups of different sizes share it and it is never exceeded (a
group of `GroupRolloutWorkflow` starts on the first and its members as the rest free, unless
`train.permit_release: group` starts every group on all of them: [Rollout permits](#rollout-permits)). It must hold a
group of the largest entry, in training and in evaluation (`eval.max_concurrent_rollouts`). With `rollout_group_limits`,
it should also hold the capped labels' groups at their limits and a group of the largest other entry: otherwise a larger
group waits for a capped one to end, and every group behind it waits too, which `platoon config validate` and
`platoon train` warn of.

**A capped label never stalls the others.** A task the order draws whose label is at its `rollout_group_limits` waits
in that label's backlog, first in, first out, and does not count toward the horizon: the run draws on, and starts the
other labels' groups, so a step may train a draw that comes after one that waits. Each step logs how many draws wait
(`train/batch/parked_rows` and `train/batch/<column>/<label>/parked_rows`). When four batches of them wait, the label's
new draws are skipped, counted (`train/batch/skipped_rows` and `train/batch/<column>/<label>/skipped_rows`) and warned
of once: its share of the order asks for more groups than its limit runs, and the other labels' groups fill the steps.
Lower its weight or raise its limit to train them all. The run draws only while some label has room, so a draw is
never made only to be skipped, and while every label is at its limit it waits for a group to end.

The horizon's slots free when a step starts, and the groups that start then take them. So that a capped label's next
group need not wait for the next step, each of its running groups holds a slot of the horizon while it has draws
waiting, which the other labels' groups leave free: when the group ends, the label's oldest waiting draw starts at
once. Its limit, not the steps, then paces the label, as far as the capped labels' groups and the slots they hold fit
in `k` batches of the horizon, which leaves the other labels at least one. With `k = 0` there is no slot to hold: a
capped label starts at most its limit of groups each step, however short they are. So a capped label trains a smaller
share than its weight while its limit (with `k = 0`, its limit each step) cannot keep up, also before any draw is
skipped; the draws still waiting when the run ends never run, and a resumed run takes them up where they were. This
applies to every config that sets `rollout_group_limits`, with or without settings per environment.

**Custom workflows.** A `GroupRolloutWorkflow` subclass applies an entry's settings without changes: inside a group,
`self.config` holds the group's entry's settings, and
`platoon.train.workflow.current_entry()` names the entry. Code that reads
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

## Credit each environment's groups (experimental)

A group's rewards become its advantages through its credit. Without a setting, every rollout's datums, the root's and
each subagent's alike, are centered on the group's mean completed-root reward (or the leave-one-out mean with
`leave_one_out_baseline`), as [Credit assignment](../tutorials/credit-assignment.md) describes. An environment entry may
choose a credit estimator instead, with its options beside it:

```yaml
environments:
  - name: delegate
    service: python://vault_env:VaultEnvironment
    harness: example-delegator
    credit: keyed                              # null, the default: the workflow's own credit
    credit_options: {key: [task, role], leave_one_out: true}
  - name: solo                                 # no credit key: the workflow's own credit
    service: python://guess:GuessTheNumber
    harness: platoon-react
```

| Estimator | What it does | Options |
|---|---|---|
| `group` | The workflow's own credit, as an estimator: each member's datums centered on its baseline from the group's completed root rewards, value for value | none |
| `keyed` | Each agent's rows centered on the mean reward of the group's agents that share its key: under `[task, role]` the roots on the roots' mean and the subagents on the subagents', where `group` centers every agent on its rollout's root baseline | `key`: one or more of `task`, `policy`, `role` and `depth` (default `[task, policy, role]`); `leave_one_out` (default false): the mean of the other agents of its key, an agent alone in its key getting 0 |
| `reverse_kl` | On-policy distillation: a frozen teacher scores every token the student sampled, and each action token's advantage is the negative reverse KL between them ([Distill a frozen teacher](#distill-a-frozen-teacher-experimental)) | `teacher` (required): a frozen policy of `policies:`; `kl_coef` (default 1); `reward_coef` (default 0): the weight of the row's reward centered on its member's baseline |
| a name an entry's `package` registers with `@credit_estimator`, or `module:Class` | yours ([Write a credit estimator](../extending/credit-estimators.md); [`examples/credit-by-role`](../../examples/credit-by-role/README.md) is one of about 40 lines) | its own |

**Where it runs.** The training workflow awaits the entry's estimator once per group, in the trainer process, after
every member of the group has settled; the error-token filter, the group's verdict and its output come after it, as
without one. Under `train.permit_release: rollout`, the default, the members have returned their rollout permits by
then; under `group` the group holds them, and in both modes its label slot, until the estimator returns. Evaluation never runs an
estimator, so an evaluation's rollouts cost nothing more. An estimator that reads agents (`keyed`, `by_role`) has the
rollout workers tag each row with the agent that produced it (worker protocol 3), and sees each member's agents: each
agent's role (the root's is `policy`, a subagent's the role its harness minted it with, `subagent` for
`ctx.agents.spawn`), its depth and parent, its rows, and its own reward, the processed reward of its trajectory, as
the converter loaded it onto its action tokens. An agent that was not scored, interrupted or a child whose verifier
gave no verdict (a judge error, an unparseable or a pending one), has no reward: it enters no baseline, and it has no
rows. A row that reaches such an estimator without its tags (a workflow subclass that runs its members another way)
fails its group, with an error that says why. An estimator that raises, or whose `assign` does not return within its
`assign_timeout_seconds` (600 by default), stops the run with an error that names it and its entry; but a policy's
scoring that fails (a teacher's service that raised or did not answer within its `sample_timeout_seconds`) fails only
the group it was crediting, whose verdict is then `failed` and which trains nothing, counted under `credit/failures`.

**What loads.** The estimator resolves when the config loads (core's names, then names the entries' packages register,
then `module:Class`), its options are checked against its schema and its `check` runs: an unknown name, an option it
does not take or a refused check stops `platoon train` and `platoon config validate` with exit code 78. So does an
estimator beside a workflow class that does not apply it (one that is no `GroupRolloutWorkflow` and does not declare
`applies_credit = True`, or a subclass that overrides `compute_baselines` or `assign_advantages`, which keeps
crediting with them), and options without an estimator. A named environment (`env:`) cannot choose one. `lock.json`
records each estimator with the entries that use it and their options, defaults included (`kind: credit`); a run whose
entries set none records nothing new, and runs as before, request for request.

**What a resume does.** A group's credit is final when the group settles: a checkpoint that keeps a settled group
keeps its advantages, and the resume trains it with them. An estimator with state (a running baseline, say) changes it
as each group is filed, and each full checkpoint saves it with the groups it keeps (`loop_state.credit`), so a resume
takes up both; a resume whose entry names another estimator than the one that saved the state starts it afresh, with a
warning. Credit metrics go with the group, recorded in the step that trains it: `keyed` logs
`credit/keyed/baselines` (the keys a group's agents fell into) and `credit/agent_advantage/abs_avg`, beside the
`credit/baseline/*` of the workflow's own credit.

## Distill a frozen teacher (experimental)

On-policy distillation trains the policy on its own rollouts with a larger model's judgement of every token it sampled.
Declare the teacher as a frozen policy beside `train:`, whose model is the student, `main`, and credit the entry's
groups with `reverse_kl`:

```yaml
train:
  model_name: Qwen/Qwen3-4B-Instruct-2507      # the student, main
  renderer_name: qwen3_instruct
  loss_fn: importance_sampling
  loss_fn_config: {}
policies:
  teacher:
    model_name: Qwen/Qwen3-32B
    renderer_name: qwen3
    trainable: false                            # frozen: it scores, and is never stepped, saved or pruned
    # tinker_base_url: http://<host>:<port>     # its own service; without it, the run's (the student's)
environments:
  - name: guess
    # ...
    credit: reverse_kl
    credit_options: {teacher: teacher}          # kl_coef: 1 and reward_coef: 0 by default
```

For each training group, once its rollouts are done, the teacher scores the tokens the student sampled
(`compute_logprobs` on its sampling client), and each action token's advantage becomes
`-kl_coef * (student logprob - teacher logprob)`, plus `reward_coef` times the row's reward centered on its member's
baseline when `reward_coef` is above 0. With `reward_coef: 1` this is tinker-cookbook's on-policy distillation
advantage (the cookbook's discounted KL, `kl_discount_factor`, is not built); the default, 0, distills alone. The
scoring runs in the trainer after the group's rollouts returned their permits, so it holds no rollout slot; an
evaluation runs no estimator, so it never calls the teacher. The step's metrics add `credit/reverse_kl/avg`, the mean
reverse KL per action token.

- **The teacher's service.** Without `tinker_base_url` the teacher uses the run's service, the student's, in a Tinker
  session of its own, so that a fatal error on the teacher's session never ends the student's. Give it a URL of its
  own when the backend cannot serve it beside the student (a self-hosted service of one base model); a job script
  passes it as `policies.teacher.tinker_base_url=http://<host>:<port>`. A service that cannot be reached stops
  `platoon train` as the trainer is built, with an error that names the policy and its URL.
- **When scoring fails.** At most 32 of the teacher's calls are in flight at once, over every group, and each is
  bounded by the teacher's `sample_timeout_seconds` (`policies.teacher.sample_timeout_seconds`). A call that fails or
  runs out of time, a response of the wrong length (not one logprob per token scored), or a token the service gives
  no logprob for, fails that group alone: its verdict is `failed`, it trains nothing, the step counts it under
  `credit/failures`, and the run goes on. The first call of a group that fails cancels the group's other calls.
- **The sampler it serves.** A frozen policy samples its base model, or the `tinker://` sampler its
  `load_checkpoint_path` names, such as a teacher trained earlier.
- **What loading refuses** (exit 78, `platoon config validate` too): a `teacher` that names no frozen policy of
  `policies:`, or names `main`; a teacher whose tokenizer differs from the student's (cross-tokenizer distillation is
  not built), compared by model name before any Tinker session opens. A tokenizer that cannot be loaded there (no
  network) leaves that comparison undone, with a warning; `platoon train` then compares the tokenizers it loaded for
  the two policies as it builds the estimator, after their sessions opened, and stops with exit code 78 when they
  differ.

[`examples/on-policy-distillation`](../../examples/on-policy-distillation/README.md) distills Qwen3-32B into the
quickstart's Qwen3-4B, with the teacher on the student's service and on its own.

## Sample several policies (experimental)

A run's agents may sample other policies than `main`: a frozen judge, a frozen helper that a harness mints as a
subagent, or a second policy the run trains ([Train several policies in
lockstep](#train-several-policies-in-lockstep-experimental)). Declare each under `policies:`, with `train:`'s names
for the same settings, and bind agent roles to them under `agents:`:

```yaml
policies:
  judge:
    model_name: Qwen/Qwen3-32B
    renderer_name: qwen3
    trainable: false
    tinker_base_url: http://<host>:<port>       # optional: without it, the run's tinker_base_url
agents:
  roles:
    judge: {policy: judge}                      # agents opened with role judge sample the judge, for life
```

- **Which agent samples which policy.** The root agent samples `agents.policy`, `main` without it (a policy the run
  trains). An agent its harness or program opens with a role (`ctx.agents.spawn(..., role=...)`, `POST
  /v1/platoon/agents {"role": ...}`, an episode program's judge) samples that role's policy for its whole life, and
  `agents.policy` when `agents.roles` names none for its role. A request never
  picks a policy, and only text crosses between policies: each agent renders, samples and parses with its own policy's
  renderer and tokenizer.
- **What trains.** `main`, and each policy of `policies:` that does not say `trainable: false` ([Train several policies
  in lockstep](#train-several-policies-in-lockstep-experimental)). A frozen policy's agents make no rows: their rewards
  still reach the credit (an estimator that reads agents sees them, with their policy), and each row is stamped with
  the version of its own policy, which that policy's staleness bound measures.
- **Rollout workers.** Under `rollout_isolation: process`, each worker follows one binding file per sampled policy
  (`model_binding.pkl` for `agents.policy`'s, `model_binding.<id>.pkl` for each other), which the trainer publishes
  before the first rollout and before each evaluation, and again for each trained policy each time it publishes a new
  sampler.
- **What loading refuses** (exit 78, `platoon config validate` too): an id that is not a lowercase name
  (`[a-z][a-z0-9_-]`, at most 32 characters); `policies.main`, which `train:` declares; a `sample_timeout_seconds`
  that is not a positive number, as `train:`'s; a role bound to a policy the run does not declare; a frozen
  `agents.policy`; roles bound to several policies beside a workflow that cannot bind them (`binds_policies`, which
  `GroupRolloutWorkflow` declares); and either key under `trainer: sft`, which rolls nothing out. A frozen policy that
  no agent samples and no entry's `credit_options` names is only warned of, and so is a policy bound to the role
  `policy`, the root agent's, which the root never samples.
- **What a resume checks.** A run with `policies:` or `agents:` records each policy's model, renderer and whether it
  trains, a frozen one's `load_checkpoint_path`, and each role's policy, in its `identity.json`, never a URL: a resume
  that adds or removes a policy, changes one of those, or binds a role elsewhere stops with exit code 78, naming each
  change; one that reaches the services at new URLs resumes. A run without either key writes and checks what it always
  did.
- **What a program reads.** Each `TinkerLLMInteraction` of an agent tree names the policy that sampled it (`policy`,
  None in a run of one policy), so a program that trains on its agents' interactions keeps the frozen policies' apart.

## Train several policies in lockstep (experimental)

A run may train several policies, one per agent role: a delegator and the workers it spawns, a writer and a critic.
Declare each under `policies:` (a policy trains unless it says `trainable: false`) and bind the roles to them:

```yaml
train:                                          # main: the root agent's policy, and the run's training settings
  model_name: Qwen/Qwen3-4B-Instruct-2507
  renderer_name: qwen3_instruct
  max_staleness: 1
policies:
  worker:
    model_name: Qwen/Qwen3-4B-Instruct-2507
    renderer_name: qwen3_instruct
    lora_rank: 16                               # its own adapter (default: train.lora_rank)
    optimizer: {learning_rate: 1.0e-5}          # its own optimizer (default: train.optimizer)
    loss_fn: importance_sampling                # its own loss (default: train.loss_fn)
    max_staleness: 2                            # its own bound (default: its entry's, else train.max_staleness)
    # tinker_base_url: http://<host>:<port>     # its own service; without it, the run's
agents:
  roles:
    subagent: {policy: worker}                  # the agents ctx.agents.spawn opens train the worker
```

Every step trains every policy, in lockstep (RFC 0021; each policy may step on its own clock instead: [Step each
policy on its own clock](#step-each-policy-on-its-own-clock-experimental)):

- **Each policy's keys.** A trained policy takes `train:`'s names, each unset one `train:`'s: `lora_rank`,
  `optimizer`, `loss_fn`, `loss_fn_config` (when unset, `train.loss_fn_config` if its loss is `train:`'s, else none:
  the loss's own defaults), `max_staleness` (when unset, its entry's bound, else `train.max_staleness`), and
  `load_checkpoint_path` with `load_optimizer_state: true` to start a new run from a checkpoint's weights and its Adam
  state. `batch_size` and `checkpoint` per policy are refused in lockstep, where every policy trains on the step's
  `train.batch_size` groups and saves at `checkpoint.every` and `checkpoint.recovery_every`: they are for policies that
  step on their own clocks.
- **Credit, then rows by policy.** A group is credited once, over every agent of every rollout; then its rows split by
  the policy that sampled them, each to its own policy's learner. An entry without `credit:` credits with `keyed` on
  `[task, policy, role]` in a run of several trained policies: each agent is centered on the mean reward of the
  group's agents of its policy and role. Any estimator works: the rollout workers tag every row with its agent, and
  the group's output keeps the tags.
- **Staleness.** Each policy's rows of a group are judged against that policy's version and its bound (its own
  `max_staleness`, else its entry's, else `train.max_staleness`), per group or per datum as `train.stale_filter` says.
  A policy's stale rows are dropped together, logged with its name, and the group's other rows still train; the group
  counts as stale only when all its rows are. The dispatcher's horizon counts the smallest bound of the trained
  policies that agents sample, so rollouts run no further ahead than the strictest policy allows.
- **Updates.** Each microbatch sends each policy's rows, prepared under that policy's objective, to its own training
  client; at each minibatch's end every policy's optimizer step is submitted before any result is awaited, so that
  their services work at the same time, and a step takes as long as the slowest. A policy with no rows in a step (no
  agent of its role ran) does not step and keeps its sampler and version, and the step's metrics record it
  (`train/policies/<id>/batch/skipped_optimizer`).
- **Publishing.** Each policy that stepped publishes its new sampler and counts one more version of its own; the
  rollout workers get it through that policy's binding file. The root agents' policy (`agents.policy`) may be any
  trained policy; its binding is the workers' default.
- **Step hooks.** A [step hook](../extending/step-hooks.md) may update any trained policy: `end.train(policy, ...)`
  and `end.learner(policy, ...)` go through that policy's own learner, its metrics under its prefix
  (`train/policies/worker/<hook>/...`). Each policy that the step or a hook updated publishes once, and the hooks hear
  every trained policy's publication; `train/policies/<id>/batch/hook_optimizer_updates` counts the hooks' updates of
  each policy besides `main`, and `train/batch/hook_optimizer_updates` those of every policy.
- **Metrics.** `main`'s keep their names. Each other policy's go under `policies/<id>/` in front of the name a run of
  one policy gives them: `policies/worker/optim/lr`, `train/policies/worker/stale_rollouts`,
  `train/policies/worker/optim/...`, `policies/worker/checkpoint/deleted`. Run-level metrics (`workload/*`,
  `batch/*` of the groups, `dispatch/*`, `timing/*`) stay as they are. An evaluation's row of `evaluations.jsonl`
  adds `versions`, each sampled policy's version when it started.
- **Failures.** An error of a policy's service, opening its training client, in `forward_backward`, `optim_step` or a
  save, fails the step, as it does for one policy, with a note that names the policy and its service (`policy worker,
  on http://<host>:<port>: optim_step failed`), in the step's own updates, a step hook's and a trainer's own
  (`run.learner`) alike; supervision restarts the start, which resumes every policy from the last checkpoint.

**Checkpoints.** Each checkpoint saves every trained policy. The policies other than `main` save first, at the same
time, each on its own service into `<run>/policies/<id>/checkpoints.jsonl`; then `main` saves into the run's
`checkpoints.jsonl`, and its row names each other policy's row of the checkpoint by its `state_path`
(`loop_state.policies`). `main`'s row, written last, commits the checkpoint. A resume starts from the newest `main` row
whose named rows every policy still has; a newer one that does not hold (a crash between the saves, a row a pruner
deleted) is skipped with a warning, which `lock.json` records (`policies_restore`); when no `main` row holds, the start
stops with exit code 78, since no restart can mend it. With `checkpoint.keep_recovery` each policy's recovery
checkpoints are pruned on its own service, within its own staleness bound: a pruner never deletes a row that the newest
`main` row, or one of the newest `keep_recovery` `main` recovery rows, names. Each checkpoint saves every trained
policy, one that did not step since the last too; those saves hold weights no rollout samples, and its pruner deletes
them as it does any older checkpoint, so a policy that seldom steps keeps no more recovery checkpoints than one that
steps each time. `keep_recovery × recovery_every` must exceed each policy's own bound too.

**What loading refuses** (exit 78, `platoon config validate` too): a trained policy that no agent samples (no row would
ever train it), `main` included when `agents.policy` names another; a frozen `agents.policy`; a frozen policy with a
trained policy's keys (`lora_rank`, `optimizer`, `loss_fn`, `loss_fn_config`, `max_staleness`,
`load_optimizer_state`); `load_optimizer_state` without `load_checkpoint_path`; a `lora_rank` below 1, a negative
`max_staleness` and an empty `loss_fn`, each named by its policy (`policies.worker: lora_rank must be ...`); a policy's
`batch_size` or `checkpoint` in lockstep; a reverse-KL teacher whose tokenizer differs from any trained policy's; and an entry
whose workflow credits with hooks of its own (`applies_credit` false). A resume that trains a policy that was frozen,
or freezes one that trained, stops with exit code 78, as any change of the policy table does.

[`examples/two-agent-lockstep`](../../examples/two-agent-lockstep/README.md) trains the subagents example's delegator
and its workers as two policies, on one service and on two.

## Step each policy on its own clock (experimental)

`train.stepping: independent` lets each trained policy step, publish and save on its own clock, with its own batch,
staleness bound and checkpoint cadence, and no policy waits for another's training or saves (RFC 0021, section 2.11).
`lockstep` stays the default, and with one trained policy both are the same.

```yaml
train:
  stepping: independent                         # default: lockstep
  batch_size: 8                                 # main's batch, and the plan's groups per step
  max_staleness: 1                              # main's bound: a horizon of (1 + 1) x 8 = 16 groups
policies:
  worker:
    model_name: Qwen/Qwen3-4B-Instruct-2507
    renderer_name: qwen3_instruct
    batch_size: 2                               # its own batch (default: train.batch_size)
    max_staleness: 7                            # its own bound: a horizon of (7 + 1) x 2 = 16
    checkpoint: {recovery_every: 4}             # its own cadence, in its own steps (default: checkpoint.*)
agents:
  roles:
    subagent: {policy: worker}
checkpoint:
  recovery_every: 1                             # main's cadence, in its steps
```

- **Shares.** Each group is credited once, over every agent, as in lockstep. Then it splits into one share per trained
  policy with rows in it: that policy's rows of the group. A policy's steps take its shares in the order they settled,
  `policies.<id>.batch_size` of them (`train.batch_size` for `main`), each trained by its own learner; then the policy
  publishes its new sampler, and only it. Every group is a share of `main`'s too, one that holds none of its rows when
  it has none, so that `main`'s steps take every group, as a lockstep step does; a group that failed is `main`'s share
  alone.
- **Rates stay coupled.** The plan is `train.batch_size` groups per `main` step, as in lockstep: each of `main`'s shares
  is one group of the plan. Each policy has a horizon of its own, `(max_staleness + 1) x batch_size` of its shares, and
  a group starts only when every policy's horizon has room, since which policies a rollout samples is known only once it
  settles. Every group feeds every policy whose agents run in it, so the slowest learner still bounds how many groups
  start; what changes is that no policy waits for another, and a policy with a smaller batch steps more often. A
  policy's rows then age faster in its own versions, so its bound grows by the same factor, `(k + 1) x batch >= main's
  (k + 1) x batch`, or its horizon holds dispatch back. `train/dispatch/blocked_share/horizon:<id>` says which horizon
  did.
- **Staleness.** Each policy judges its own shares, against its version and bound. A stale share is dropped for that
  policy alone; the group's other shares are untouched. `policies/<id>/peer_lag/<q>` reports how many versions each
  other policy moved between the sampling of the shares' rows and their training.
- **Commits.** Every full save of any policy is a commit: a point a resume starts from. It captures, at once, the run's
  draws, every group that has not retired with what each policy did with its share (the step that consumed it and
  its age then), and every other policy's newest saved row and published version; it then writes them and the
  policy's weights, and appends the policy's row to its own `checkpoints.jsonl` (`main`'s the run's, another policy's
  `<run>/policies/<id>/checkpoints.jsonl`), with `loop_state.commit_seq` and `loop_state.policies`, every policy's row
  by its `state_path`. A group stays kept until each policy that consumed its share has saved a row after that step
  (`run/retained_groups`), so keep `recovery_every` small. A pruner never deletes a row that a commit in flight, or
  the newest commit in some policy's file, names.
- **Resume.** A start resumes from the commit with the highest `commit_seq` in any policy's file whose named rows its
  policies still have; a newer one that does not hold is skipped and recorded in `lock.json` (`policies_restore`). Each
  policy goes on from the row the commit names, at its own step. A policy whose samplers published after that row were
  lost resumes at one version more than the commit records it published, so that no new sampler reuses a lost number.
  Each share a policy consumed after its row goes back to the front of its queue and trains again, judged at the age it
  had then (`policies/<id>/retrained_shares`; a trainer of your own judges it at `ledger.judge_version(policy, group)`),
  so no share trains twice in a policy's weights. A share the commit records as not yet consumed is judged afresh when
  it is taken, and may be dropped as stale then: the version rule and the shares trained again before it age it, so keep
  `recovery_every` small. A policy other than `main` names its own commits with its id (`recovery-<id>-<step>`,
  `<id>-<step>`), apart from the rows the start's launch and exit saves write into its file.
- **Evaluation** counts `main`'s steps (`eval.every`); an evaluation's row in `evaluations.jsonl` adds `versions_end`
  when some policy published while it ran.
- **Metrics.** Each step logs a row of its own: `main`'s with the names a run of one policy gives them, the run's
  metrics (`train/dispatch/*`, `train/workload/*`, `timing/*`, `run/retained_groups`) and `run/step`; every other
  policy's under `policies/<id>/`, with `policies/<id>/step`. The rows carry no W&B step of their own: W&B numbers them,
  and each metric plots against its row's axis. Every policy's rows add `policies/<id>/share_wait_s` (how long its step
  waited for shares), `policies/<id>/retrained_shares` and `policies/<id>/peer_lag/<q>`.
- **Step hooks** run at the end of each policy's own step, with `end.policy` naming it, and may update that policy only:
  `end.learner(other, ...)` is refused, and `run.learner` sends nothing from a hook, whichever policy's learner it is.
  The ends of different policies' steps run at the same time, so a hook's methods may too; a commit waits until no
  step's end runs its hooks, and so saves no hook's state part-way. It saves the state as it is then, while every other
  policy resumes from the older row the commit names and runs its step ends since then again: a hook whose state moves
  at a policy's step ends keeps it by that policy's own step, and takes it up at `start.policies[<id>].step` ([Keep its
  state across restarts](../extending/step-hooks.md#keep-its-state-across-restarts)).
- **Interrupts and exit.** A stop request stops each policy's loop at its step's end; a step that applied an update
  finishes first. A loop that stopped takes no more shares, so once its horizon is full no group starts again: a step
  that still waits for shares then closes short with those it has, as at the plan's end (it counts as a step of the
  plan, so the run draws that many fewer groups), rather than wait for groups that never start or give the stopped
  policy shares beyond its bound. The save at exit (`<step>_interrupted`, or `final`) saves every policy and names them
  all. A start that resumes with `main` at the plan's end, after a stop or a crash that came after `main`'s last step,
  trains the shares the other policies have left and saves `final`. A trainer of your own passes the stop to
  `ledger.take(policy, n, stop=run.shutdown)`, which raises `TrainInterrupted` once it is set.
- **Failures.** A policy's service that fails, in an update or a save, stops the start, as in lockstep: the other
  policy's loop is cancelled, nothing saves at exit, and a commit whose save had not appended its row is not one. The
  next start resumes from the newest commit, each policy from the row it names. A policy that is slow, or whose save
  hangs, holds the others back once its horizon fills: rates stay coupled.

**What loading refuses** (exit 78), besides lockstep's: a policy's `batch_size` that `train.num_minibatches x
train.num_microbatches` does not divide, `checkpoint.recovery_every` below 1 or `checkpoint.every` below 0, and a group
admission rule other than `all` (`train.group_admission` or an entry's), whose steps count their groups toward a cap
that this release keeps for steps of every policy at once. With `checkpoint.keep_recovery`, each policy's own pruner
keeps its promise only when `keep_recovery` times the policy's own `checkpoint.recovery_every` (else the run's), counted
in its own steps, exceeds its own `max_staleness` (else the run's largest bound): loading refuses a policy whose cadence
does not. Loading warns of a trained policy that saves no full checkpoint before the run's end (no `recovery_every` and
`checkpoint.every: 0`): every group it consumed stays kept, and every commit writes them all. A resume that changes the
stepping stops with exit code 78: `identity.json` records `stepping: independent`, a start checks that its run's rows
are commits (or are not), and a lockstep resume would train consumed shares again. Loading reads `train.stepping`; a
trainer of your own whose `needs()` choose the stepping is held to the same rules as its start begins, and sets
`train.stepping: independent` to give a policy its own `batch_size` or `checkpoint`. `trainer: sft` trains one policy,
on the run's one clock, and refuses `train.stepping` as it does the other `train:` keys only the rl trainer reads
([Fine-tune on a dataset](#fine-tune-on-a-dataset)).

[`examples/two-agent-independent`](../../examples/two-agent-independent/README.md) trains the subagents example's
delegator and its workers, the worker stepping about four times for each of the delegator's steps.

## Limit the tools each role may call (experimental)

The agents of a role may be limited to some of their environment's tools: subagents that may only read, a reviewer
that may not submit. List the tools under `agents.roles.<role>.tools`:

```yaml
agents:
  roles:
    subagent: {tools: [read]}                   # the children ctx.agents.spawn opens: read, and nothing else
    reviewer: {tools: [view, "read_*"]}         # agents minted with POST /v1/platoon/agents {"role": "reviewer"}
    judge: {policy: judge, tools: []}           # beside a policy (Sample several policies): no tool at all
```

- **What a list names.** Each item is a tool's name or an `fnmatch` pattern (`read_*`), matched case-sensitively
  against the names the agents see: the environment's tools, and the tools of a Harbor task's MCP servers as the
  gateway serves them, `<server>__<tool>` (`mcp_server__get_secret`, or `mcp_server__*` for all of one server's). A
  role without `tools` keeps every tool, as before, and `tools: []` allows none.
- **Which agents it limits.** Every agent opened with the role: a child its harness spawns
  (`ctx.agents.spawn(..., role=...)`, whose role is `subagent` unless it says otherwise), mints (`POST
  /v1/platoon/agents {"role": ...}`) or its program opens (a `judge`, a `verifier`), and the root agent when the list
  is under its role, `policy`. An agent under a limited one keeps that limit, besides its own role's: a child never
  sees more tools than its parent, as with tool access (`agents.child_access`). A verifier (`child_grading:
  verifier`) is the exception, as for its access (`agents.verifier.access`): it runs under the child it judges, but
  is limited by the lists of the agent that spawned that child and its own role's, `verifier`, not by the child's. A
  role may give `tools` alone: its agents sample `agents.policy` (`main` without it), as a role `agents.roles`
  leaves out does.
- **Where it holds.** In every listing of tools an agent gets (`ctx.list_tools()`, MCP `tools/list`), after its tool
  access level, and where the gateway runs a call (`ctx.call_tool()`, MCP `tools/call`), so that a harness that calls
  a tool its listing does not hold is refused too. A call to another tool comes back to the agent as an error result
  that says why (`Tool 'answer' is not allowed: agents of role 'subagent' may call only read. The call was not run.`),
  the environment never sees it, and the run goes on; the agent's trajectory counts the refusals in its misc
  (`tools_refused: {count, first}`). `platoon train` applies the lists to its training and evaluation rollouts, in
  process and in the rollout workers of its group workflow; a workflow class of your own whose rollouts run in
  workers it starts itself passes them on as that workflow does (`WorkflowSetup.role_tools`, the workers' requests'
  `role_tools`).
- **What it does not reach.** A harness's own tools (the shell and editor an OpenHands or Codex harness runs in the
  task's sandbox, the subagents example's `delegate`) and the MCP servers a harness reaches by itself
  (`{task_mcp_servers}` in its spec) never pass through the gateway's tool plane: limit those in the harness. A
  function harness that calls its environment session itself (`ctx.session.call_tool(...)`) passes the tool plane by
  as well, its tool access included. An ORS bridge's `get_task` tool (`bridge_tools: true`), when a role allows it,
  shows the environment's whole tool list, the tools the role does not allow included, though a call to one is still
  refused. A task MCP tool is matched by the name the gateway serves, in which every character of its server's name
  and its own but an ASCII letter, a digit and `_` is replaced by `_`: `mcp_server__*` matches the tools of a server
  `mcp-server`, and `mcp-server__*` matches none, which loading does not warn of, as it does not check patterns
  against a Harbor task's tools.
- **What loading refuses** (exit 78, `platoon config validate` too): `tools` that is not a list (write `tools: [read]`,
  not `tools: read`), an item that is not a non-empty string, a key of a role other than `policy` and `tools`, and a
  role that is no name. A role that looks like a misspelling of one of platoon's own (`subagents` for `subagent`) is
  only warned of, since a harness may mint agents of any role. A pattern that matches none of the tools of the run's
  environments is only warned of, and only when every environment names its tools before a session opens: an ORS
  environment run in process (`python://`) whose class declares them all, a task table (none), or a service class with
  `declared_tools`. Harbor tasks, whose MCP servers list their tools once a session opens, and ORS env servers are not
  checked.

[`examples/subagents/read-only-subagents.yaml`](../../examples/subagents/read-only-subagents.yaml) trains the
subagents example with subagents that may only read, so that only the root answers.

## Label tasks by a column

Each task of a run is a dataset row: its `task_id` and the columns its environment gives it. A mixture gives each
task the entry it belongs to, in the column `_platoon_environment`, and a Harbor dataset each task's index fields,
such as its `source_dataset` ([Index fields as dataset columns](sandboxes-and-harbor.md#index-fields-as-dataset-columns)).
`train.label_by` names the column that labels the run's tasks, by default the mixture entry. A task's label is its
value of the column as text (`true` or `false` for a boolean), and a task without the column has none. By label:

| What | Per label |
|---|---|
| Metrics | Each training step logs each label's groups, reward, steps and counts, `train/batch/<column>/<label>/{groups, task_reward, num_steps, outcome/<status>, reward_key/<name>, reward_key_mismatch, stale_groups}`, also at zero, and each evaluation during training logs them as `eval/<column>/<label>/...` ([Metrics to watch](#metrics-to-watch)). The default column logs as `environment`, so a mixture's metrics read `train/batch/environment/<name>/...`, and a single environment, whose tasks have no entry column, logs none. A `/` in a column's name or a label is written `--` |
| `train.rollout_group_limits` | The most rollout groups of that label that run at once, for example to cap a slow entry of a mixture; its draws wait for a slot in its backlog, which never holds another label's groups back, and past four batches of waiting draws its new draws are skipped and counted ([Settings per environment](#settings-per-environment)) |
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
`workflow_config.straggler_policy_key`. Platoon 0.3.0 removed both: a config that still sets one does not load (exit
78), and the message names `train.label_by`, whose labels straggler policies key on.

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

### Cadences in steps or epochs

`every` counts training steps in `eval` and in `checkpoint`. With `strategy: epoch` it counts epochs instead, as
`train.num_epochs` does: passes over the training tasks, in whole batches ([Train on several
environments](#train-on-several-environments) says what an epoch draws). So `eval: {strategy: epoch, every: 1}`
evaluates at the end of each epoch. `strategy: step` counts steps, as a section that leaves `strategy` out does.

A section that sets `strategy` and leaves `every` out triggers its event every 1 of its units, as `every` did by
default when it counted in `strategy`'s units: `eval: {strategy: step}` evaluates every step, and `checkpoint:
{strategy: epoch}` saves a permanent checkpoint at the end of each epoch. A section that leaves out both, like a
config without the section, never triggers it, and so does `every: 0` with either strategy.

```yaml
eval:
  strategy: epoch     # every counts epochs
  every: 1            # the end of each epoch
checkpoint:
  strategy: epoch     # every left out: 1, a permanent checkpoint at the end of each epoch
  recovery_every: 5   # recovery checkpoints still count steps
```

A start that resumes evaluates at the steps a run that never stopped would, the multiples of the interval
([Evaluation in a resumed run](#evaluation-in-a-resumed-run)), while it counts the interval of permanent checkpoints
from the step it resumes at, as it counts `recovery_every` ([Checkpoints and resume](#checkpoints-and-resume)).
`strategy: none`, which turned the event off whatever `every` said, is gone since platoon 0.3.0; set `every: 0`, which
a config or an override that still says `none` is told to (exit 78).

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

## Fine-tune on a dataset

`trainer: sft` fine-tunes the policy on chat datasets instead of rollouts: supervised fine-tuning, with the loss on the
model's turns. The datasets follow [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)'s conventions
([`data/README.md`](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md)), so a dataset prepared for
LLaMA-Factory trains here as it is, but for the few differences [listed below](#where-it-differs-from-llama-factory):
a `dataset_info.json` names each dataset and says where its rows are and how to read them, in the `alpaca` or the
`sharegpt` format. [`examples/sft`](../../examples/sft/README.md) trains on one of each.

```yaml
trainer: sft
trainer_options:
  dataset: alpaca_demo, sharegpt_demo     # names in dataset_dir/dataset_info.json; a list works too
  dataset_dir: data                        # relative to this file (left out: data, beside it)
  batch_size: 32                           # conversations per step, one optimizer step over them
  num_epochs: 3                            # or max_training_steps; with both, the shorter
  max_length: 4096                         # LLaMA-Factory's cutoff_len: tokens a conversation is cut to
train:                                     # the policy: model, renderer, adapter and optimizer
  model_name: Qwen/Qwen3-8B
  renderer_name: qwen3
  lora_rank: 32
  optimizer: {learning_rate: 1.0e-4}
checkpoint: {every: 200, recovery_every: 20, keep_recovery: 2}
```

`trainer_options` (the [config reference](../reference/config.md#trainers) lists every key):

| Key | Default | What it does |
|---|---|---|
| `dataset` | required | The datasets, by name: one, several separated by commas (as LLaMA-Factory's `dataset`), or a list. Their conversations are put together in this order |
| `dataset_dir` | `data` | The directory of `dataset_info.json` and the files it names (LLaMA-Factory's `dataset_dir`), relative to the config file; left out, `data` beside the config file |
| `batch_size` | 32 | Conversations per step; an epoch's last step takes what is left |
| `num_epochs`, `max_training_steps` | unset | How long the run trains: one of them is required, and with both it trains the shorter |
| `num_microbatches` | 1 | forward_backward calls a step's conversations are sent in, before its one optimizer step |
| `max_length` | unset | Tokens a rendered conversation is cut to, as LLaMA-Factory's `cutoff_len` cuts it: turn by turn from the first (with `mask_history`, from the last), each turn's prompt and reply sharing what is left, and the turns past it left out; unset, no cut |
| `mask_history` | false | Train on the last model turn of each conversation only (LLaMA-Factory's `mask_history`) |
| `seed` | 42 | Each epoch's order of the conversations, and the rows `num_samples` takes |

`train:` declares the policy (`model_name`, `renderer_name`, `renderer_kwargs`, `lora_rank`, `optimizer` and the
deadlines of its updates). Its keys that only the rl trainer reads (its batch, duration, loss, staleness, stepping and
rollout settings) are refused under `sft` when set, each with the `trainer_options` key to set instead when there is
one, as `train.batch_size` names `trainer_options.batch_size`. An sft run rolls nothing out, so it needs no
`environments`, and it refuses `policies:` and `agents:`, which only the rl trainer samples or scores with
([Sample several policies](#sample-several-policies-experimental)).

### The datasets

`dataset_info.json` maps each dataset's name to where its rows are and how to read them, with LLaMA-Factory's keys and
defaults:

| Key | Default | Meaning |
|---|---|---|
| `hf_hub_url` | | A dataset on the Hugging Face Hub; it wins over `file_name` |
| `file_name` | | A file in `dataset_dir`, or a folder of files of one type: `json`, `jsonl`, `csv`, `parquet` or `arrow` |
| `formatting` | `alpaca` | `alpaca` or `sharegpt` |
| `split`, `subset`, `folder` | `train`, none, none | Which split, configuration and folder of the dataset to read |
| `num_samples` | all | Rows to take: each row once, in an order `seed` decides, before any twice (again in that order when it asks for more than there are) |
| `columns` | | Which column holds what: alpaca's `prompt` (`instruction`), `query` (`input`), `response` (`output`) and `history`; sharegpt's `messages` (`conversations`); either format's `system` and `tools` |
| `tags` | | How a sharegpt message names its role and text: `role_tag` (`from`), `content_tag` (`value`), and the roles `user_tag` (`human`), `assistant_tag` (`gpt`), `observation_tag` (`observation`), `function_tag` (`function_call`) and `system_tag` (`system`) |

The defaults in parentheses are those of an entry without `columns` (or `tags`). An entry that gives them replaces
the defaults whole, as LLaMA-Factory does: a key it leaves out names nothing. LLaMA-Factory's own `firefly` entry,
`{"prompt": "input", "response": "target"}`, reads no query; and a `tags` without `system_tag` reads no system message,
so a row that opens with one is out of order, and skipped. A key that LLaMA-Factory does not read either, such as
`file_sha1`, is ignored, with a warning.

How a row becomes a conversation, as LLaMA-Factory's converters do it:

- **alpaca:** the `history`'s prompt and response pairs come first, each a user turn and an assistant turn; then
  `instruction` and `input` joined by a newline (an empty one left out) as the user's turn, and `output` as the
  assistant's. A row without a text output is skipped.
- **sharegpt:** the messages in order. The user's turns and the tools' results (`observation`) go at odd positions,
  the assistant's turns and its tool calls (`function_call`) at even ones, and a conversation ends with one of the
  latter. A first message with the system tag is the system prompt, in place of the `system` column. A row out of that
  order is skipped. The OpenAI format of LLaMA-Factory's README is `sharegpt` with `messages` and the tags `role`,
  `content`, `user`, `assistant` and `system`.
- **Tools and tool calls:** `tools` is a JSON list of functions (`name`, `description`, `parameters`; an OpenAI tool
  `{"type": "function", "function": ...}` too). A `function_call` message is JSON, one call (`name`, `arguments`) or
  a list of them.

A skipped row is counted in one warning per dataset, which names the first and says why it was skipped. Preference
datasets (`ranking`, the `chosen`, `rejected` and `kto_tag` columns), multimodal ones (`images`, `videos`, `audios`),
`formatting: openai` and rows from sources other than the Hub and files (`ms_hub_url`, `om_hub_url`, `script_url`,
`cloud_file_name`, which LLaMA-Factory reads over `file_name` when there is no `hf_hub_url`) are refused by name.
`platoon config validate` reads `dataset_info.json`, and refuses a dataset it does not name; `platoon config validate
--tasks` loads the datasets, and says how many conversations each has.

### What trains

Each conversation is rendered by the policy's renderer, as the policy sees a conversation when it samples: the system
prompt and the tools first, as the renderer puts them, each tool call as the assistant's `tool_calls` and each result as
a `tool` message. The tokens of the model's turns, its replies and its tool calls, train (with `mask_history`, the last
one's only); the system prompt, the tools, the user's turns, the tools' results, the renderer's headers and a system
message the renderer adds of its own (as `kimi_k2` and `nemotron3` do) do not. The renderer must be a tinker-cookbook
one, which renders message by message (`chat_template` renders whole conversations), and, for conversations with tools,
one with tool calling, such as `qwen3`. Before the service opens, the run renders the first conversation, the first with
tools, the first with a tool call and the first with neither a system prompt nor tools, and refuses a renderer that
cannot render one or that renders a tool call without it (as `llama3` does), naming the conversation's dataset and row
(exit 78); so is one that cannot render a later conversation, when a step comes to it. Each step's loss
is the mean negative log-likelihood over its trained tokens (`train/nll`, over all its microbatches), as LLaMA-Factory's
is, so a learning rate means the same whatever the batch size. `max_length` shares its tokens among a conversation's
turns as LLaMA-Factory's `cutoff_len` does, so a reply keeps tokens to train on also behind a long prompt; a conversation
left with none is left out of its step, the first of each dataset named in a warning, and
`train/untrained_conversations` counts them. A run in which no conversation has a token to train within `max_length`
stops (exit 78) once it has walked every conversation.

### Epochs, checkpoints and resume

Each epoch walks every conversation once, in an order that `seed` and the epoch's number decide. Every full
checkpoint saves where the run is (`loop_state.sft`: the epoch and how far into it), so a resume goes on with the next
conversation, and each conversation trains once an epoch, also across restarts. A resume may change `batch_size`: the
run's end is counted on from where the checkpoint left the conversations. A resume with other conversations or another
`seed` would walk another order, and is refused (exit 78): start a new run from the checkpoint's weights with
`checkpoint.load_checkpoint_path` instead (and its optimizer's state, with `checkpoint.load_optimizer_state: true`).

Checkpoints come as for any run ([Checkpoints and resume](#checkpoints-and-resume)). `checkpoint.keep_recovery` needs
no staleness bound here: nothing samples the policy, so no recovery checkpoint is kept for a sampler. Nothing samples
it during training either: the policy's version stays where the run began, and `optimizer_steps` counts the steps. A
step hook that a subclass of the sft trainer registers runs at each step's end ([Write a step
hook](../extending/step-hooks.md)): the policy then publishes once a step, for the hook, and `optimizer_steps` counts
the hook's updates too.

Evaluation during training is not there yet for `sft` (`eval.every` must be 0): evaluate its checkpoints with
`platoon eval` ([Evaluate](#3-evaluate)).

### Where it differs from LLaMA-Factory

- The policy's renderer renders each conversation, as the policy sees it when it samples; LLaMA-Factory's templates
  (its `template` argument) are not used.
- `max_length` has no default: unset, nothing is cut (LLaMA-Factory's `cutoff_len` is 2048).
- `dataset_dir` is relative to the config file, as every input path of a config is; LLaMA-Factory's is relative to the
  working directory.
- `num_samples` beyond a dataset's rows takes them again in the seeded order; LLaMA-Factory draws the extra ones at
  random.
- A `function_call` message that is not JSON skips its row; LLaMA-Factory stops with an error while it preprocesses.
- Not read: `formatting: openai`, the sources besides the Hub and files, preference and multimodal datasets, and
  LLaMA-Factory's training arguments `train_on_prompt`, `packing`, the `mix_strategy` other than `concat`,
  `max_samples` and `streaming`.

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
| `000010` | Every `checkpoint.every` training steps, or epochs with `checkpoint.strategy: epoch` (0, the default, saves none), counted from the step each start of the run begins at; kept |
| `recovery-000012` | At every launch, and every `checkpoint.recovery_every` steps when that is set; disposable: `checkpoint.keep_recovery` deletes all but the newest few |
| `000007_interrupted` | At the first SIGINT or SIGTERM, when a step completed since launch; weights and optimizer, kept |
| `final` | When training reaches its last step, after the evaluation due at that step |

Between full checkpoints, each step only pushes new sampling weights, which cannot be resumed from. Each full
checkpoint also saves the run's per-task statistics, which a resume loads ([Per-task
statistics](#per-task-statistics)), and where the run's draws stand, which it takes up (below).

**Delete old recovery checkpoints.** A full checkpoint of a large model trained in full is hundreds of gigabytes, so
a long run that sets `recovery_every` can fill the service's disk. With `checkpoint.keep_recovery: N`, each time a
recovery save is recorded in `checkpoints.jsonl`, the run deletes its recovery checkpoints beyond the newest N,
weights and sampler weights, through the training service's checkpoint deletion (the Tinker SDK's
`RestClient.delete_checkpoint_from_tinker_path`). It never deletes:

- a permanent checkpoint (`000010`, `000007_interrupted`, `final`);
- the checkpoint a resume would start from;
- a recovery checkpoint this start saved within the run's largest staleness bound of the policy (`train.max_staleness`,
  or an environment's `max_staleness` above it), which rollouts may still sample from (one an earlier start saved
  serves none of this start's rollouts, and under `trainer: sft` none does: [Fine-tune on a
  dataset](#fine-tune-on-a-dataset));
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
applied an optimizer update finishes, and one that has not is dropped, its groups kept for the resume to train.
`platoon train` then saves `<step>_interrupted` when a step completed since launch, and exits 0. A signal during the
evaluation due at the last step ends that evaluation, and the run saves `final` as usual; a later start of the run
runs that evaluation and nothing else ([Evaluation in a resumed run](#evaluation-in-a-resumed-run)). A second signal
exits at once with code 1 and saves nothing. The interrupt checkpoint is a full save: for a large model trained in
full it can take longer than the time a scheduler leaves between SIGTERM and SIGKILL
([Run on a cluster](clusters.md#6-submit-the-client)). A save cut short records nothing in `checkpoints.jsonl`, so the
next launch resumes from the checkpoint before it. A run that is killed loses the steps since its last full
checkpoint, which `checkpoint.recovery_every` bounds; the groups that had settled by that checkpoint and that no step
had trained are not lost with them.

**Resume** is automatic: start the same command again with the same `log_path`, `stats.experiment_name` and
`stats.trial_name`. platoon reads the last resumable row of `checkpoints.jsonl`, restores the weights and the
optimizer state, continues from that step, and reattaches the W&B run. It evaluates at the multiples of
`eval.every` from that step on, that step itself only if no earlier start completed its evaluation ([Evaluation in a
resumed run](#evaluation-in-a-resumed-run)). To start over, use a new `stats.trial_name`.

**What a resume takes up.** With rollouts running ahead of training, a checkpoint falls between groups the run has
drawn and the steps that train them. Each full checkpoint saves where the run's draws stand, in
`tasks/dispatch-s<start>-<checkpoint>.json` (its row names it as `dispatch`), and a resume takes them up, so that
every position of the order is consumed exactly once: trained, dropped as stale, failed, skipped, or rejected by a
group admission rule.

- **Groups that settled and that no step had trained** are kept with the checkpoint, in
  `tasks/kept-s<start>-<checkpoint>.json.gz` (their datums and what the step that trains them logs), and the first
  steps of the resume train them as they are, without rolling them out again. A group of a slow environment that took
  hours is not lost to a restart. Every full checkpoint keeps them: the one each start saves at launch, recovery,
  permanent, `<step>_interrupted` and `final`. A group whose workflow failed is kept as failed, and the resume
  consumes it as failed, as the run would have without the restart. A workflow that returns something other than a
  `GroupRolloutWorkflow` output or a plain list of datums is not kept: its task is rolled out again.
- **A group a rule rejected** ([Train only on groups that carry
  signal](#train-only-on-groups-that-carry-signal-experimental)) has its fate when it settles: no checkpoint keeps it,
  and a resume from a later checkpoint never draws it again (a crash, or a stop before a start's first step completes,
  loses the rejections made since the last full save, whose positions the resume draws again). A resume that turns a
  rule on trains the groups the checkpoint kept as they are, since they were decided when they settled.
- **Draws whose groups were still running, or waited for their label's limit,** are drawn again first, at their
  places in the order, as new draws of the resume, each from the row it drew. When the run's training tasks changed
  so that that row now holds another task, the draw takes the first row of its own task.
- **The order goes on** where the checkpoint left it: the run's task sampler, and each environment's when the run's
  hands it the environment's share of the slots (`weighted`, and `auto` for a mixture), at the count of decisions it
  saved, with the state it saved (an adaptive sampler's, such as `pass_rate`'s record of what it drew when). A
  checkpoint of platoon 0.3.0, whose file (format 1) saved no counts, is resumed exactly too: each count is recomputed
  from the order's position.
- **Slots waiting for an adaptive entry's sampler** (tokens: under `weighted`, a slot owed to an entry whose sampler
  decides when its group can start) are saved with the file, the one waiting for its rollout permits first, and go
  back to their entry's queue. When the entry's sampler is static now, it binds them, in order; when the entry no
  longer gets slots, or left the run, they are dropped, each with a `slot_skipped` event in `tasks/events.jsonl`, and
  the resume's warning counts them.

That file does not depend on the per-task statistics: a resume is exact also from a checkpoint saved after the
statistics stopped, or when its snapshot could not be written, and the statistics then go on from an older snapshot. Its
format is 2 since platoon 0.4.0; platoon 0.3.0 cannot read it, and resumes such a checkpoint from its batch, as one
without the file. A resume whose kept groups cannot be read (their file deleted, changed, or never written because the
disk refused it, each with a warning) rolls those tasks out again instead, and trains each once all the same; the
statistics count both of such a task's groups, since both ran, as one draw. A checkpoint without the file (one saved
before platoon 0.3.0, or one whose write failed, with a warning) is resumed as before: from the checkpoint's batch,
rolling out again what had not trained. One saved under a group admission rule, whose steps draw more positions than
their batch, records in its row where the order stood, and the resume draws on from there, the draws that had not
trained lost.

The kept groups cost disk and time at each full save: about ten bytes a token of their datums, written again at
every full save while they wait for a step, in a thread beside training. With `max_staleness: 3`, batches of 32
groups of 8 and rollouts of 16,000 tokens, a save can keep some 12 million tokens, about 130 MB. Pruning deletes a
recovery checkpoint's files with it (`checkpoint.keep_recovery`); without it, they stay, as the checkpoints do.

**Changed sampling inputs.** The checkpoint also saves a fingerprint of what the order follows from: `task_sampler`,
`batch_size`, `label_by`, the environment entries in the order the config lists them, and each entry's weight, its
seed (when a sampler reads it) and its training tasks; and the samplers platoon 0.3.0 did not have, with their options
as parsed, defaults included: the run's `task_sampler_options` for a sampler other than `auto`, `weighted` and
`uniform`, and an entry's `task_sampler` and `task_sampler_options` when its sampler is not `uniform`; and, in a run
with a group admission rule, each entry's rule and options and `refill.max_groups_per_step`. So a config that writes a
default out reads as no change, and a 0.3.0 checkpoint's fingerprint is the same under 0.4.0. A resume whose
inputs differ warns, naming each change, records it in `lock.json` (`sampling_changed`), and goes on: the order goes
on at the same place, now the order the new inputs give, and the draws of tasks that left the run are lost, as the
per-task statistics count them. A sampler saved under the same name and class loads its state, changed options
included; another one starts afresh (a static one at the count the position implies, an adaptive one reading the
statistics the run has), and so does one whose saved state does not load, with a warning.

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
fresh optimizer, and applies only when the run directory has no checkpoint of its own yet. With
`checkpoint.load_optimizer_state: true` beside it, the new run takes that checkpoint's optimizer state too, Adam's
moments, as a new trial that goes on from it (the path must be a full checkpoint's `state_path`). Either way the new
run's counters start at 0, and nothing else of the other run's state is taken up.

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
- `platoon eval` reaps those labelled with its output directory when it starts and when it exits; a SIGTERM stops
  it as Ctrl+C does, cancelling its rollouts in flight, and it exits 143 once it has reaped them;
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
gone since platoon 0.3.0: it exits 78, naming `watchdog.max_restarts`, which replaces it.

## Rollout permits

A training run's groups share its budget of rollout permits, `train.max_concurrent_rollouts`. Each rollout's permit
comes back as the rollout ends (`train.permit_release: rollout`, the default since platoon 0.4.0), and the next group
starts on the permits that come back. Before 0.4.0 a group held its permits until its workflow returned, so a rollout
that had finished kept its permit until its group's slowest rollout ended: on long agentic tasks that left a quarter to
a third of the budget idle while the trainer waited for data. `train.permit_release: group` keeps that behavior
([below](#the-previous-behavior)).

```yaml
train:
  max_concurrent_rollouts: 256  # the rollouts that run at once
  max_staleness: 4              # one more than before, if the run warns of stale groups (below)
```

- **Rolling starts.** `GroupRolloutWorkflow` runs each member on its group's lease, so a group starts on its first free
  permit and its members start as the rest free, over about `(group_size - 1) × (a rollout's time) /
  max_concurrent_rollouts` (10 to 12 s in long agentic runs). Each member's straggler deadline then carries its own
  start delay, so a late start never shortens its grace. A workflow of your own does the same by running its rollouts
  on the lease ([Run rollouts on the lease](../extending/rollout-leases.md)); any other workflow's groups start once
  their whole size of permits is free, as before.
- **When a permit comes back.** Under process isolation, once the rollout's worker process tree is reaped and the
  sandboxes and env sessions it left are deleted. In process (the default), once the rollout's coroutine, its
  subagents and their grading have ended, and the sandbox and session teardowns that platoon finishes in the
  background too, such as one a straggler cut interrupts or one a rollout leaves running as it ends: the permit waits
  for them at most 60 s, and the run warns of a teardown that runs longer. Under `group`, nothing waits for them, as
  before, and such a teardown can outlive its group's release.
- **What stays.** Groups are drawn, decided by task samplers, admitted, credited, filtered for staleness, kept by
  checkpoints and resumed as before, and evaluation is unchanged. A partly finished group counts as a whole group
  toward the horizon, and label limits count groups, so `max_staleness` keeps its meaning. A run at `max_staleness: 0`
  with the default budget trains the same datums in the same order under either setting.
- **What it costs.** Where the permits bound a run, more groups are open at once and steps get shorter, so trained
  data is older, and more long groups cross the staleness bound: stale drops can rise, and those are the hard, mixed
  groups. When more than 2% of the groups the last 10 steps consumed went stale, the run warns, once per start of
  training and under either setting (from its tenth step on, so a shorter run never does), naming what held the
  dispatcher back and advising `max_staleness + 1`. If the horizon had room (`train/dispatch/blocked_share/horizon`
  near 0), raising `train.max_staleness` by one changes only the stale filter: a group may then be one more policy
  version old and still train, and recovery pruning keeps one more checkpoint. `train.stale_filter: datum` instead keeps
  the part of a long group within the bound ([Per-datum stale masks](#per-datum-stale-masks-experimental)). A restart
  also redoes more rollouts (every open group's), and more finished rollouts wait in the trainer for their siblings.
- **Where it helps.** Where the permits decide when groups start (`blocked_share/permits` near 1). Where the horizon
  does (`blocked_share/horizon` near 1), freed permits have nothing to start: there only a larger `max_staleness` or a
  faster trainer helps, and the run trains as it did before 0.4.0.

**`max_concurrent_rollouts`** counts the rollouts that run at once, no longer the whole groups it holds, and
`rollout_client/concurrency_achieved_fraction` reads as the permits' busy share, close to 1 where the permits bind;
`groups_in_flight` can exceed the budget over the group size. Raise the budget while the inference engines have room
and `blocked_share/permits` stays near 1. The run warns, once per start, when `rollout_client/rollouts_in_flight_max`
exceeds the budget: a workflow returned a permit while its rollout still ran. Only `GroupRolloutWorkflow`'s members
count as rollouts in flight, so a workflow of your own is not checked, and an evaluation's `GroupRolloutWorkflow`
left on the default `stats_scope`, `train`, counts too (a train script's; `platoon train` gives evaluation its own).

**Straggler settings** (`straggler_timeout_seconds`, `straggler_quorum`, `straggler_policies`) stay as they are, off by
default. They are a deadline against runaway rollouts, not a throughput lever: once a finished rollout no longer holds
its permit while its group waits, a cut buys only the straggler's own remaining time and slightly fresher data, and it
censors slow rollouts, which fail more often, so the sample that trains leans optimistic. `train/batch/censoring/*`
shows what a cut selects.

**The startup gate** (`worker_launch_concurrent_groups`, process isolation only) still admits groups, not rollouts: a
group holds its slot until its last member is up, which under rolling starts includes the time its permits take to
free. With the fork server (the default) a worker is up in a fraction of a second, and at most one group is partly
granted at a time, so the gate's second slot serves the next group. With `worker_start_method: exec` (17-24 s a
worker) the gate can hold groups back while they hold their permits, which `rollout_client/startup_gate_waiting_mean`
shows: then raise `worker_launch_concurrent_groups`.

| Metric | What it says |
|---|---|
| `rollout_client/permits_held_mean` | The permits granted to groups, time-weighted. Held minus `rollouts_in_flight_mean` (`GroupRolloutWorkflow`'s members) is permit time no rollout used: under `group`, mostly finished rollouts waiting for their siblings; under `rollout`, close to 0 |
| `train/dispatch/blocked_share/<reason>` | The share of the step in which the dispatcher could not start a group, by reason: `permits` (raise `max_concurrent_rollouts`, or, under `group`, return permits as rollouts end), `horizon` (only `max_staleness` or a faster trainer helps), `labels` (every label it could start was at its `rollout_group_limits`), `plan` (the run's remaining steps need no more groups), `sampler` (an adaptive task sampler found no eligible task) |
| `train/dispatch/blocked_share/horizon:<id>` | When the trained policies step each on its own clock, in `main`'s rows: the share of the step in which policy `<id>`'s horizon held the dispatcher back, `horizon` being any of them. Near 1 for one policy: its horizon, `(max_staleness + 1) x batch_size` of its shares, is smaller than `main`'s (raise its `max_staleness`), or its learner is too slow for the run's rate ([Step each policy on its own clock](#step-each-policy-on-its-own-clock-experimental)) |
| `rollout_client/startup_gate_waiting_mean` | The groups waiting for a worker-startup slot, time-weighted (process isolation) |
| `train/batch/censoring/censored_share`, `.../mixed_groups`, `.../slowest_pass_rate`, `.../others_pass_rate`, and each label's under `train/batch/<column>/<label>/censoring/` | Over every group the step dequeued that a straggler policy applies to (none without one): the share of rollouts the policy cancelled, and over the groups whose valid rollouts both passed and failed, the pass rate of each group's slowest valid rollout against that of the others. A slowest rate well below the others' means a cut drops failures more than successes |

### The previous behavior

To restore what platoon did before 0.4.0, set `train.permit_release: group`:

```yaml
train:
  permit_release: group  # a group's permits come back together, when its workflow returns
```

Every group then takes its whole size of permits before it starts and returns them when its workflow returns, so its
finished rollouts keep their permits until its slowest one ends. The lease's blocks hold nothing, and the run starts
and ends groups as earlier releases did, event for event. The default changed in 0.4.0 ahead of a GPU comparison of
the two settings (RFC 0020); `group` stays as the way back, and as the other arm of such a comparison.

## Per-datum stale masks (experimental)

The stale filter drops a group whose oldest datum is more policy versions old than its bound (`train.max_staleness`,
or its environment's own), so one early model call of one long rollout drops the work of its whole group.
`train.stale_filter: datum` judges each datum instead, and drops only those beyond the bound:

```yaml
train:
  max_staleness: 3
  stale_filter: datum   # default: group
```

- **A datum's version.** Each model call records the policy version it sampled: in process, the version the trainer
  had published when the call was issued; under process isolation, that of the binding the worker followed for it.
  A datum, a run of model calls whose prompts extend one another merged into one sequence, carries the oldest version
  among its own calls, where under `group` every datum of a rollout carries the rollout's oldest (the request's
  version, in process). A rollout that spans a published policy then keeps the datums it sampled since. In a run that
  samples several policies ([Sample several policies](#sample-several-policies-experimental)), each call records the
  version of the policy that sampled it, and only the trained policy's datums are judged: a frozen policy's agents
  make none.
- **What it drops.** Each datum more versions old than the bound. The others train with the advantages their group's
  credit gave them: the credit ran over the whole group as it settled, so each term kept keeps its expectation. A group
  that keeps any datum is consumed: its statistics, its fate in the per-task statistics and its place in the batch are
  those of a group that trained. A group none of whose datums is within the bound is dropped whole, as under `group`,
  and counts as a stale group (`train/batch/stale_groups`, and the warning of [Rollout permits](#rollout-permits)).
  That warning counts only these groups: one that keeps any datum counts as consumed, however many it lost, which
  `train/batch/stale_datums` shows (below). Under depth-level weighting a trajectory whose first datum is dropped
  counts from its first datum kept.
- **What stays.** Credit, verdicts, group admission, the per-task statistics (each rollout's versions there are still
  its rollout's) and the horizon. A group a checkpoint kept carries the versions it was stamped with as it settled,
  so a resume under the other setting judges it by those.
- **A workflow of your own.** `GroupRolloutWorkflow` stamps each datum with its own version under `datum`: it takes the
  setting (its `takes_stale_filter` capability), and the trainer sets its `stale_filter` as training starts. The
  datums of a workflow that does not take it keep the versions they carry, by which the filter judges each of them
  (their rollout's, as platoon's conversion stamps them), and the run warns once per start. A subclass that drops or
  adds a member's datums after the conversion keeps the result's `own_versions` in step with them, or those datums
  keep their rollout's version (warned once per start). One that overrides `arun_episode` or `_run_group` without
  calling `GroupRolloutWorkflow`'s stamps nothing: it sets `takes_stale_filter = False` in its body, so that the run
  warns, or stamps its datums itself (`TrajectoryCollectionResult.stamp_own_versions`).
- **Why it is not the default.** It changes "a stale group is dropped whole" into "per datum, with the group's credit
  unchanged", and `group` stays the default until a GPU comparison on held-out evaluation (RFC 0020, open question 3).
  In replays of long agentic runs, the masks removed 90-94% of the stale drops at the same bound.

A step logs each group it kept with datums dropped (`Stale datums masked in batch <n>: ...`), and under `datum`:

| Metric | What it says |
|---|---|
| `train/batch/stale_datums` | The datums the step dropped as stale: those beyond the bound in the groups it trained, and every datum of the groups it dropped whole |

## Rollout start-ups (experimental)

Before its first model call a rollout sets up its worker, its environment session or sandbox and its harness, and part
of that work is shared by every rollout starting on the node. At a run's start or a resume every permit is free, so the
whole budget starts at once, and an evaluation during training adds its own rollouts. `train.max_starting_rollouts`
lets at most that many rollouts be starting at once, training's and evaluation's together, first come, first served:

```yaml
train:
  max_concurrent_rollouts: 256  # the rollouts that run at once
  max_starting_rollouts: 32     # of which at most this many set up at once (default: null, no limit)
  workflow_config: {group_size: 8}
```

- **A start-up.** A `GroupRolloutWorkflow` rollout takes a start slot once it holds its permit, before its request is
  built and its worker launched, and gives it back at the first of: its first model call is issued (its root's or any
  subagent's, as the model gateway hands the call to sampling, not at its first token), it ends, or
  `workflow_config.worker_startup_budget_seconds` + 300 s pass without a call (600 s by default). Under process
  isolation the worker marks its first call in a node-local file next to its readiness file, which the trainer polls
  as it polls readiness, so the slot can come back up to 0.2 s late.
- **Where it helps.** Where start-ups share a capacity that slows down as more of them use it at once, or run past
  per-start timeouts when too many start together: the singularity provider's `copy` layout and the enroot provider,
  which extract a root filesystem for each sandbox ([The singularity provider's root
  filesystem](sandboxes-and-harbor.md#the-singularity-providers-root-filesystem)); task sets with an image per task;
  several runs on one node. The wait comes before the worker starts, so no start-up or rollout timeout counts it.
- **Where it does not.** It shortens no one-time work: the fork server's import of the worker stack, the copy of a
  harness toolchain to each node, which the node's first rollouts wait for, or the modules each worker imports after
  its fork ([Troubleshooting](troubleshooting.md#the-first-rollouts-of-a-run-take-minutes-to-reach-the-model)). Any
  value makes a run's whole first wave reach the model later, though its first rollouts can reach it sooner: in long
  agentic runs whose first rollouts took about 100 s to their first call, most of it such one-time work, a limit of 32
  would have made the whole first wave reach the model 44-56 s later, and 64 about 10-23 s later (estimates). With
  `worker_start_method: exec`, `worker_launch_concurrent_groups` already bounds the workers' import storms.
- **One rule.** A start-up must not wait for another rollout of the run. A rollout holding a slot waits for nothing, so
  one that waited for its group-mates would hold its slot until the timer while they wait for slots.
- **Sizing.** At least the largest group size: below it a group's members start in waves, and the run warns, though each
  member keeps its full straggler grace, its deadline shifted by its wait as by a permit's. To keep the shared resource
  busy, about twice (the start-ups it completes a second) × (the seconds one start-up takes alone); far below that, the
  resource idles between start-ups. With the limit at that product itself, a model of measured runs of short crafting
  tasks lost 14-18% of their throughput (estimates).
- **With the rest.** A rollout waiting for a slot holds its permit but does not run yet, so while the limit binds the
  gap between `rollout_client/permits_held_mean` and `rollouts_in_flight_mean` widens, by `start_limit/waiting_mean`:
  the lever is then `max_starting_rollouts`, or turning it off, not a larger `max_concurrent_rollouts`, which only adds
  waiters. Under process isolation a group keeps its `worker_launch_concurrent_groups` slot until its members are up, so
  most of a binding limit's queue waits at that gate (`rollout_client/startup_gate_waiting_mean`). An evaluation during
  training takes slots of the same limit, in turn. In process, training's rollouts that come behind a whole evaluation
  wait for its start-ups, about eight start-up times behind 256 evaluation rollouts at a limit of 32. A fresh start with
  `eval.at_start` is such a case: its evaluation reaches the limit before training's first rollouts, so the first step
  comes that much later. Standalone `platoon eval` is not limited (`num_concurrent_workers` bounds its rollouts), and
  the limit is one trainer's: several runs on a node do not share it.

Only `GroupRolloutWorkflow`'s members take start slots ([Run rollouts on the
lease](../extending/rollout-leases.md#grouprolloutworkflow-and-its-subclasses)).

| Metric | What it says |
|---|---|
| `rollout_client/start_limit/starting_mean`, `.../waiting_mean` | The training rollouts holding a start slot, and those waiting for one, time-weighted; an evaluation's under `eval/rollout_client/start_limit/`, logged with it |
| `rollout_client/start_limit/wait_seconds_p50`, `_p90`, `_max` | How long the rollouts that took a slot in the step waited for it (0 for one taken at once) |
| `rollout_client/start_limit/releases/first_call`, `.../end`, `.../timer`, `.../cancelled` | The slots given back in the step, by cause: mostly `first_call` when the harness calls the model through platoon's gateway; `cancelled` for rollouts cancelled while they held one, as by a straggler cut or a stop |

The time from a slot to the first call, except the request's build, is
`rollout_client/dequeue_to_first_model_call_seconds_p50` (and `_p90`), as before. The run warns once per start when a
slot comes back on its timer, and when most come back only at their rollout's end: the harness makes no model call
through platoon's gateway (`platoon-oracle`, `platoon-noop`, a rollout function of your own), or its rollouts fail
before one, and the limit then bounds whole rollouts.

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
| `train/stale_rollouts`, `train/batch/<column>/<label>/stale_groups` | 0 or rare | Rollouts older than `max_staleness` were dropped; by label, which one's groups grow too old: raise its entry's bound ([Settings per environment](#settings-per-environment)). Under `train.stale_filter: datum`, groups none of whose datums was within the bound; `train/batch/stale_datums` counts the datums dropped ([Per-datum stale masks](#per-datum-stale-masks-experimental)) |
| `train/batch/skipped_rows`, `train/batch/<column>/<label>/skipped_rows` | 0 | Draws of a label at its `rollout_group_limits` were skipped, four batches of them waiting already: lower its weight or raise its limit |
| `train/batch/parked_rows`, `train/batch/<column>/<label>/parked_rows` | Small, steady | Draws of a label at its `rollout_group_limits` that wait for a slot: a count that grows, step after step, is a label whose share of the order its limit cannot serve, so it trains less than its weight; those still waiting when the run ends never run |
| `train/batch/informative_groups`, `.../all_pass_groups`, `.../all_fail_groups`, `.../uniform_mixed_groups`, and each label's under `train/batch/<column>/<label>/` | A large share informative | The step's groups that trained, and those whose advantages were all zero: every rollout passed, none did, or both ([Per-task statistics](#per-task-statistics)). Mostly all-pass or all-fail: the tasks are too easy or too hard for the policy, and their groups train nothing |
| `train/tasks/<column>/<label>/pass_at_1`, `.../pass_at_G` | Rise over steps | Each label's mean expected pass@1 and pass@G over its tasks with a settled group, G its entry's group size: the pass@k curve of training, for every run. Flat while the reward rises: a shaping reward processor rewards something other than passing |
| `train/tasks/<column>/<label>/no_evidence` | 0 | The label's tasks whose last three groups had no valid rollout: a broken grader or setup; `platoon tasks export` lists them (`no_evidence_streak`) |
| `train/sampler/draws`, `train/sampler/draws/<reason>`, and each entry's sampler's under `train/sampler/environment/<entry>/` | `draws/fallback` 0 | The draws since the last step, by why their sampler chose them: `order` for a static order, `explore` and `exploit` for `pass_rate`, `reissue` for a draw a resume drew again. `fallback` counts decisions a sampler failed (it raised, or answered outside the contract) that the dispatcher filled itself: fix the sampler ([Write a task sampler](../extending/task-samplers.md)) |
| `train/admission/groups_admitted`, `.../short`, `.../rejected/<reason>`, `.../draws_per_admitted`, `.../errors` | `errors` 0 | In a run with a group admission rule: what each step admitted and rejected and why, whether it was short, and the draws per admitted group ([Train only on groups that carry signal](#train-only-on-groups-that-carry-signal-experimental)). Short steps at every step: raise `train.refill.max_groups_per_step`, or choose tasks with `pass_rate` |
| `train/sampler/decision_ms`, `.../decision_ms_max` | A few ms | The mean and longest time a sampler's decisions took since the last step; long ones hold up the event loop. An adaptive entry's `.../tokens` counts its slots waiting for a group to start, and `train/sampler/skipped_slots` those a full queue skipped. `pass_rate`'s own metrics: [Choose tasks by their pass rates](#choose-tasks-by-their-pass-rates-experimental) |
| `rollout_client/trainer_queue_wait_seconds_total` | Small next to `timing/batch_total` | The trainer waits on rollouts: raise rollout concurrency or speed up the environment |
| `train/dispatch/blocked_share/permits`, `.../horizon` | One of them near 1 says what decides when groups start | `permits`: the rollout budget binds; a larger `max_concurrent_rollouts` while the engines have room (under `train.permit_release: group`, first the default, `rollout`), unless `rollout_client/start_limit/waiting_mean` is above 0: then rollouts hold permits while they wait to start up, and the lever is `max_starting_rollouts`. `horizon`: only `max_staleness` or a faster trainer helps ([Rollout permits](#rollout-permits)) |
| `train/dispatch/blocked_share/horizon:<id>`, `policies/<id>/share_wait_s` | Neither near 1, or long, for one policy alone | When the trained policies step each on its own clock ([Step each policy on its own clock](#step-each-policy-on-its-own-clock-experimental)): the share of `main`'s step in which policy `<id>`'s horizon held the dispatcher back, and how long the policy's step waited for its shares. A horizon near 1 is a policy whose `(max_staleness + 1) x batch_size` is below `main`'s, or whose learner is too slow for the run's rate; the others then wait for shares |
| `policies/<id>/peer_lag/<q>` | Small | How many versions policy `<q>` moved between the sampling of the rows policy `<id>` trained and their training |
| `policies/<id>/retrained_shares`, `run/retained_groups` | 0 but after a resume; steady | The shares a resume put back that the step trained again, and the groups kept until each policy that consumed its share has saved after that step, which every commit writes: they grow with a policy's `recovery_every` |
| `rollout_client/permits_held_mean` against `rollout_client/rollouts_in_flight_mean` | Close under `permit_release: rollout`, the default; under `group` the gap is the straggler idle that `rollout` returns | Permits held while no rollout ran: finished rollouts waiting for their group's slowest (`permit_release: group`), groups waiting at the startup gate, or rollouts waiting for a start slot |
| `rollout_client/start_limit/waiting_mean`, `.../wait_seconds_p90`, and under process isolation `rollout_client/startup_gate_waiting_mean` | Near 0, or briefly at a run's start | With `train.max_starting_rollouts`: rollouts wait to start up, holding their permits. Raise `max_starting_rollouts`, or turn it off, not `max_concurrent_rollouts` ([Rollout start-ups](#rollout-start-ups-experimental)) |
| `rollout_client/start_limit/releases/<cause>` | Mostly `first_call` | Many `end` or `timer`: the rollouts make no model call through platoon's gateway, or fail before one, so the limit bounds whole rollouts: turn it off for this workload, or fix the failures |
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

## Per-task statistics

Training keeps a record of each training task: what each of its groups did, and what they say so far about how often
it passes. Each step logs it by label, each full checkpoint saves it, and `platoon tasks` reads it.

**What a group did.** Each group settles with the workflow's verdict on what it trains: `trained` (an informative
group, with a nonzero advantage), `uniform` (every retained advantage zero), or why it trains nothing:
`reward_key_mismatch`, `no_results`, `below_min` (fewer valid rollouts than `min_successful_group_size`),
`no_completed_root`, `no_datums`, and `failed` for a workflow that raised or returned nothing. Every settled group
counts once, stale ones too: their rollouts ran, and a uniform group says the most about a task's difficulty. A
workflow class of your own that returns a plain list of datums gives no verdict (`unknown`) and says nothing of its
rollouts, so its groups count but give no evidence: its tasks read as not seen, and the step logs no pass@k for them.
The trainer numbers each training row it draws, `_platoon_seq` and `_platoon_position` in the row a workflow gets, to
join the group's record to its draw.

**When a group enters them.** A group's record is made when it settles, and the statistics take it in, with every
other record made since, in the order the groups were drawn: before each step's metrics, each full checkpoint and each
decision of an adaptive task sampler. So their values do not depend on the order in which a step's groups completed,
down to the last bits of their decayed sums, and an adaptive sampler and a checkpoint read every group that has
settled. A group's fate (consumed, stale, failed) does not wait for it, and a group a rule rejected has its fate
(`rejected`) when it settles. A resume is the exception: the groups that had settled when its checkpoint was saved were
taken in then, and those drawn again after it, so a task whose group drawn again came before one that had settled takes
them in another order than an uninterrupted run, its streaks and newest groups included.

**What passes.** A rollout passes when it is valid (its root completed with a scored outcome, so it entered the
baseline) and the reward its environment scored, before any reward processor, reaches its entry's
`success_threshold`, 1.0 by default: `platoon eval` judges success the same way. A shaping reward processor that trains
0.98 for a correct rollout changes what trains, not what passes. Set the threshold of an environment that gives
partial credit on its entry; a named environment may ship its own:

```yaml
environments:
  - name: proofs
    env: acme-proofs@1
    success_threshold: 0.5
```

As in `platoon eval`, a rollout whose trace records no outcome passes on its own success flag at 1, whatever the
threshold, and one without a flag on its trained reward. A rollout without a valid result (an error, a timeout, a
cancelled straggler, a dead worker) counts as neither a pass nor a failure, as the baseline leaves it out. A uniform
group is all-pass when every valid rollout passed, all-fail when none did, and mixed otherwise: no datum kept the
signal, or the trained reward and the scored one disagree.

**What each step logs.**

| Metric | What it is |
|---|---|
| `train/batch/{informative,all_pass,all_fail,uniform_mixed}_groups`, and each label's under `train/batch/<column>/<label>/` | The step's groups that trained, and those whose advantages were all zero, by success: together, the groups it trained on (stale groups are left out, as from every `batch/` metric) |
| `train/tasks/<column>/<label>/pass_at_1`, `pass_at_G` | The mean expected pass@1 and pass@G over the label's seen tasks, those with a group whose rollouts were recorded, G its entry's group size; `train/tasks/pass_at_1` and `pass_at_G` over every task |
| `train/tasks/<column>/<label>/coverage` | The share of the label's tasks seen |
| `train/tasks/<column>/<label>/bands/{zero,low,mid,high,solved}` | The label's seen tasks by expected pass rate: at most 0.05, at most 0.2, below 0.8, below 0.95, and from 0.95 |
| `train/tasks/<column>/<label>/no_evidence` | The label's tasks whose last three groups had no valid rollout |

The expected rates come from each task's passes and failures, weighted by age: a group sampled `n` policy versions ago
weighs `2^(-n/32)`, so a step that takes no optimizer update ages nothing. A task's counts start from a prior of
weight 2 centered on its entry's pass rate, so a task not yet seen reads as its entry does, and its own groups soon
outweigh the prior: one group of 4 to 8 valid rollouts already counts for more. pass@G is the chance that at least
one of G rollouts passes, under that estimate.

**Saved with each checkpoint.** Each full checkpoint writes the statistics to `tasks/state-s<start>-<checkpoint>.json`
in the run directory, where `<start>` numbers the run's starts, and its row of `checkpoints.jsonl` names the file with
its SHA-256 (`tasks`). A start that resumes loads the file its checkpoint names before the checkpoint it saves at
launch, which saves them again under its own number. Groups drawn before the checkpoint that no completed step had
trained on stay outstanding: the resume trains the groups the checkpoint kept and draws the others again ([What a resume
takes up](#checkpoints-and-resume)), each still one draw of its task. A file that cannot be written (a full disk, say)
leaves its checkpoint without one, with a warning, and the next full checkpoint writes the statistics again. A start
whose checkpoint names no file (one saved before platoon 0.3.0, or whose write failed, or after the statistics stopped),
or whose file is missing or changed, goes on from the newest older checkpoint's file that reads back, without the groups
since, else from empty, with a warning; `lock.json` lists each such start, the checkpoint, why, and the checkpoint whose
statistics it loaded (`tasks_state_lost`). The draws that older file holds as outstanding and the checkpoint no longer
does count as lost; the draws themselves resume exactly all the same. A start whose checkpoint saved no state of the
draws (one saved before platoon 0.3.0, or one whose write of it failed) counts all the outstanding draws as lost and
draws from the checkpoint's batch, or, from a checkpoint saved under a group admission rule, from where its row says the
order stood. Training goes on either way. `checkpoint.keep_recovery` deletes a pruned checkpoint's files with it.

`tasks/events.jsonl` logs each draw, settle and fate (`consumed`, `stale`, `failed`, `skipped`, `lost`, and `rejected`
for a group a rule rejected, its event naming the rule and the reason as `rule` and `admission`) and each start,
one JSON line each, by start and draw number. A draw names the task sampler that drew it and why (`sampler`, such as
`uniform@<entry>` for an environment's `uniform` under `weighted` or `pass_rate@<entry>`, and `reason`, `order` for a
draw of the order, `explore` or `exploit` for `pass_rate`'s, `fallback` for one the dispatcher filled after a sampler
failed), with the sampler's `score` when it gave one, as the group's record in the statistics does; a draw a resume
draws again names the draw it replaces (`"reason": "reissue"`, `"reissues": [start, seq]`). A slot owed to an adaptive
entry's sampler whose position was used up without a draw is a `slot_skipped` event (`why`: `backlog`, its entry's
queue was full; `no_slots` or `entry_left`, a resume found that its entry no longer gets slots, or left the run). A
start logs its own events once its launch checkpoint is saved, so that a start whose launch save fails, whose number
the next start takes, logs none; one that dies in the moment between that save and its first events logs none either,
and the draws it drew again appear only as those a later start's draws replace. What a start did after its last
checkpoint stays in the log when it dies, or stops before its first step completes (it saves no checkpoint then): a
group a rule rejected then has a fate in the log, and another when a resume draws its position again. A resume never
reads the log: losing it loses history only.

**Read them.** `platoon tasks show RUN_DIR` prints the statistics of the run's newest full checkpoint, while it trains
too (an older checkpoint's, when the newest one's file cannot be read, and stderr says why): one row per task with a
settled group, the likeliest to give a group that trains first. Each row has the task's groups, its passes over its
valid rollouts, its expected pass rate, pass@G and chance of a group that trains, the newest policy version of its
recent groups (each group's oldest sampled version, which the stale filter reads), its draws not yet trained on, and
what became of the rest:

```console
$ platoon tasks show runs/mix/trial-1
runs/mix/trial-1: checkpoint recovery-000004, policy version 4: 3 of 4 tasks have a settled group
task           groups  pass/att  E[p]  E[pass@G]  E[inf]  newest v  out  consumed/stale/failed/skipped/lost/rejected
quick::hard         2       1/7  0.23       0.59    0.57         3    0  2/0/0/0/0/0
quick::easy         2       7/8  0.80       0.99    0.53         2    0  2/0/0/0/0/0
slow::partial       2       1/2  0.50       0.70    0.40         3    1  0/1/0/0/0/0
```

`--sort pass_rate` puts the hardest first, `--label` keeps one label's tasks and `--top N` the first N. `platoon tasks
export RUN_DIR` writes a row for every task of the run as CSV, or JSON lines with `--format jsonl`, with the counts
behind them, the kinds of groups named as the step metrics name them (`informative_groups`, `all_pass_groups`,
`all_fail_groups`, `uniform_mixed_groups`).

The statistics' types (`platoon.train.task_stats`) are experimental: a later release may add fields to them.

## Next steps

- [Sandboxes and Harbor tasks](sandboxes-and-harbor.md): train and evaluate harnesses inside task containers.
- [Run on a cluster](clusters.md): the training service, environment servers and the client as scheduler jobs.
- [Troubleshooting](troubleshooting.md): what failed rollouts and error messages mean.
- [Architecture](../architecture.md): how the gateway, environments and harnesses fit together.
- [Renderers and tool-call parsers](../extending/renderers.md) and [Hooks, telemetry and
  metrics](../extending/hooks-and-metrics.md): a renderer or parser of your own, and metrics from your own code.
