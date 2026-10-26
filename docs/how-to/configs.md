# Write and override configs

Every `platoon` command that runs something takes one YAML config file plus optional `key=value` overrides.
This guide shows how to split a config into layers with `extends:`, change any value from the command line,
check a config before you spend compute on it, and find the exact config a past run used.

**Goal:** a training config and an evaluation config that share one environment definition, checked with
`platoon config` and overridden from the command line.

**Prerequisites:** platoon installed (`pip install 'platoon[tinker,ors]'`, or `uv sync --extra tinker --extra ors`
in a checkout). The runnable version of this guide is
[`examples/config-layers`](../../examples/config-layers/README.md) in the repository, on number search, the example
environment that ships with platoon. `platoon config` itself never contacts a training service.

## The two kinds of config

`platoon` picks the schema from the file's sections:

| The file has | Schema | Used by |
|---|---|---|
| an `inference:` section | `PlatoonEvalConfig` (`platoon/eval/run.py`) | `platoon eval`, `platoon rollout` |
| a `train:` section | `PlatoonTinkerRLTrainerConfig` (`platoon/train/config_defs.py`) | `platoon train`, `platoon rollout` |
| neither | none: plain YAML, only meant to be extended | `extends:` |

The top-level keys of a training config:

| Key | What it holds |
|---|---|
| `environments` | A list of environment entries: `service`, `harness`, `harness_options`, `agents`, `weight`, `dataset_kwargs`, ... (`EnvironmentConfig` in `platoon/train/components.py`) |
| `train` | Model, renderer, LoRA rank, batch size, duration, loss, `optimizer`, and `workflow_config` (grouping, credit, rollout limits and isolation) |
| `eval` | When to evaluate during training (`every` training steps; 0, the default, never; `at_start` also evaluates the starting policy, at step 0), how many rollouts run at once (`max_concurrent_rollouts`), and how its rollouts differ from training's (`workflow_config`, laid over `train.workflow_config`). A config without the section does not evaluate |
| `checkpoint` | When to save (`every` and `recovery_every`, in training steps), how many recovery checkpoints to keep (`keep_recovery`) and `load_checkpoint_path` |
| `log_path`, `stats` | Where the run writes: `<log_path>/<stats.experiment_name>/<stats.trial_name>/`, and the W&B settings under `stats.wandb`. Without `log_path`, the site profile's `runs` ([Run on a cluster](clusters.md#4-write-the-sites-profile-once)), and then `stats.experiment_name` defaults to the config file's name |
| `tinker_base_url` | The URL of the Tinker-compatible training service (hosted Tinker, or a self-hosted server such as [miles-tinker](https://github.com/neulab/miles-tinker)), for every Tinker client the config opens; `null` uses the SDK's default (`TINKER_BASE_URL`, else hosted Tinker) |
| `watchdog` | Hang detection: `enabled`, `timeout_seconds`, `exit_code` |

An evaluation config has `inference` (the policy, the `workflow` and an optional `output_dir`), `environments`,
`split` and `stage` (`full`, `rollouts` or `report`). [Train and evaluate](train-and-eval.md) covers what the
sections do; this guide covers how files and overrides combine.

## Steps

### 1. Put what is shared in a base file

A base file can hold any part of a config. This one holds only the `environments` list, so both a training config
and an evaluation config can use it:

```yaml
# environments.yaml
environments:
  - service: python://platoon.envs.number_search.env:NumberSearchEnvironment
    harness: platoon-react
    agents:
      max_depth: 0
    dataset_kwargs: {split_name: train, limit: 1000}
    eval_dataset_kwargs: {split_name: val, limit: 20}
```

### 2. Build on it with `extends:`

`extends:` takes one path or a list of paths, relative to the file that names them:

```yaml
# train.yaml
extends: environments.yaml

train:
  model_name: Qwen/Qwen3-4B-Instruct-2507
  renderer_name: qwen3_instruct
  batch_size: 8
  max_training_steps: 50
  optimizer:
    learning_rate: 3.0e-5
eval:
  every: 0                        # no evaluation during training
log_path: ./runs
stats:
  experiment_name: number-search
  trial_name: base
```

```yaml
# smoke.yaml
extends: train.yaml

train:
  batch_size: 2
  max_training_steps: 2
stats:
  trial_name: smoke
```

The merge rules (`platoon/utils/config.py`):

- Bases load first, in the order listed, then the file itself merges over them. A base may extend further files.
- **Mappings merge key by key.** `smoke.yaml` changes `train.batch_size` and keeps `train.model_name`.
- **Everything else replaces, lists included.** A file that sets `environments:` replaces the whole list, so
  restate every entry you want to keep. To change one entry, override it by index instead (step 3).
- **A section you set in part keeps its field's defaults for the keys you leave out.** `eval.workflow_config`
  defaults to one rollout per task with `filter_errors` and `filter_zero_advantage_datums` off, and an
  `eval.workflow_config` that sets only `rollout_config` keeps those values, not the defaults of `WorkflowConfig`
  (8 rollouts per task, both filters on). So a section that sets `min_successful_group_size` or a straggler quorum
  above 1 sets `group_size` too; otherwise loading fails, naming the section and the defaults it kept. A free-form
  mapping such as `train.loss_fn_config` replaces its default whole, so restate every key you want to keep.
  `platoon config show --defaults` (step 4) shows what a run gets.
- **`eval.workflow_config` is laid over `train.workflow_config`.** The keys it leaves out keep training's rollout
  settings (`rollout_config`) and rollout-worker settings (`rollout_isolation`, `worker_*`), so an evaluation rolls
  out as training does; only the keys that shape groups, advantages and filters keep evaluation's defaults
  ([Evaluate during training](train-and-eval.md#evaluate-during-training)).
- The `extends` key itself is consumed and never reaches the run. A cycle, a missing base or a file whose top level
  is not a mapping is a configuration error.
- **Input paths are relative to the file that names them,** as `extends:` is, so a config runs the same from any
  directory: an environment's harness `.toml` spec, and the options its service reads files from (a Harbor entry's
  `dataset`, `registry`, `adapter.repo`, `dataset_dir`, `task_dirs`, `splits`, `lock`, `tasks`, `overlays` and
  `mounts[].source`, a task table's `path`). A relative input that names nothing beside its file is a configuration
  error, never looked up anywhere else; when the path exists relative to the working directory, where such paths
  resolved before, the error names that path too. A split manifest named relative to a `dataset_dir` that the same
  file names stays relative to it, a `lock` that a fetch will write need not exist yet, and paths inside a dataset
  (`index[].path`) stay relative to the dataset.
- **Overrides and outputs are relative to the working directory:** a path given on the command line (step 3), and
  where a run writes, `log_path` and `inference.output_dir`, in which a leading `~` names the home directory. The
  run's record holds these paths absolute (step 6).

### 3. Override values on the command line

Each argument after the config file is one override, applied in order on top of the merged files:

| Form | Example | Effect |
|---|---|---|
| `a.b=value` | `train.batch_size=16` | Set a nested key |
| `a.N.b=value` | `environments.0.harness=platoon-codeact` | Reach into a list by index |
| `a.-1.b=value` | `environments.-1.weight=2` | Negative indices count from the end |
| `a.b+=value` | `stats.wandb.tags+=nightly` | Append one item to a list (an unset list starts empty) |
| `a.b={yaml}` | `'train.loss_fn_config={clip_low_threshold: 0.0, clip_high_threshold: 4.0}'` | Containers are written as YAML; a mapping merges into a mapping the config files set, and over a default it follows the rules of step 2 |
| `a.b=null` | `checkpoint.load_checkpoint_path=null` | `null` (or `none`, `~`) for fields that allow it |

Each value is parsed against the type of the field it sets, not guessed from its text: `stats.trial_name=1` stays
the string `"1"`, `train.batch_size=16` is the integer 16, `train.optimizer.learning_rate=1e-5` is a float, and a
boolean field takes `true`/`false` (also `yes`/`no`, `on`/`off`, `1`/`0`). Fields typed as free-form mappings
(`service_kwargs`, `harness_options`, `dataset_kwargs`, ...) parse their values as YAML.

A harness's options are set the same way, in an entry's `harness_options`, or in its
`eval_overrides.harness_options` for evaluation rollouts only (every split but `train`). They are checked against
the harness's options when the run starts, like the config's own keys: an option a TOML spec does not declare, or a
value of the wrong type (`num_retries=many`, `reasoning_replay=0`), stops the run with exit 78 and its dotted path:

```bash
uv run platoon train train.yaml environments.0.harness_options.reasoning_replay=false
uv run platoon train train.yaml environments.1.eval_overrides.harness_options.num_retries=5
uv run platoon eval eval.yaml environments.0.harness_options.request_timeout_s=300
```

Quote overrides that contain spaces, braces or other characters your shell treats specially:

```bash
uv run platoon train smoke.yaml \
  environments.0.dataset_kwargs.limit=100 \
  'environments.0.harness_options.system_prompt=Guess one integer per turn.' \
  'stats.wandb.tags+=nightly'
```

The same overrides work with `platoon eval`, `platoon rollout` and `platoon config show`.

### 4. Check the result before you run it

`platoon config show` prints the merged config after `extends:` and overrides, which is exactly what a run would
load:

```bash
uv run platoon config show smoke.yaml train.batch_size=4
uv run platoon config show smoke.yaml --defaults        # also print every field's default
uv run platoon config show smoke.yaml --kind train      # force a schema instead of picking it by section
uv run platoon config show mix.yaml --entry slow-tasks  # what one environment entry runs with
```

When environment entries set training settings of their own (`workflow_config`, `max_staleness`), `config show` then
prints, as comments, what each of them runs with: every key's value and the layer it came from
([Settings per environment](train-and-eval.md#settings-per-environment)).

`platoon config validate` checks one or more files and reports every problem at once:

```console
$ uv run platoon config validate train.yaml broken.yaml
ok   train.yaml (PlatoonTinkerRLTrainerConfig)
FAIL broken.yaml
  train.learning_rate: unknown key
  train.task_sampler: 'random' is not one of 'auto', 'weighted', 'uniform', 'pass_rate', a name an environment's package registers, or module:Class
  stats.trail_name: unknown key; did you mean stats.trial_name?
```

It exits with 0 when every file is valid and 78 when any is not, so it fits in CI or a job script's first line.
It checks the files and the named environments their entries select (`env:`), then resolves every harness,
program and other component the config names, as a run does when it starts, which imports them, and checks the
options set for them (a TOML spec's `[options]`, which `platoon harness show NAME` lists): run it where the config's
extensions are installed, or pass `--no-resolve` to check only the files and the named environments. A service's
options (`service_kwargs`) are checked against what the service takes, without building it, so a misspelled one
fails with a did-you-mean (`environments.0.service_kwargs.datset_dir: unknown key; did you mean
environments.0.service_kwargs.dataset_dir?`), and so are the keys of `dataset_kwargs` and `eval_dataset_kwargs`
that a service's task listing takes (`split_name`, `shuffle_seed`, `limit`). The service checks the options'
values when a run builds it to list the tasks, and a value it refuses there is a configuration error that names the
entry. `config validate` neither lists tasks nor contacts a service: a training config without a run length, an
`exclude_tasks` item that names no task and a missing `TINKER_API_KEY` pass it, and the commands that run the config
refuse them with 78 when they start. `--schema module:Class` validates against another dataclass or pydantic model,
for example `--schema platoon.config_defs:InferenceParams`.

`--tasks` lists a training config's tasks as `platoon train` does, which reads the datasets and may contact env
servers: what listing them finds wrong fails the file, as it stops a run (an `exclude_tasks` item that names no
task, a batch larger than the training tasks), and under the file it prints how the run draws the tasks, with the
warnings for a mixture's weights ([Train on several environments](train-and-eval.md#train-on-several-environments)).

`key=value` overrides among the files apply to each of them, as `train` and `eval` apply theirs, so a config whose
launcher passes some keys is checked as it runs (`platoon config validate train.yaml log_path=runs`). A file that
holds only `environments`, which training and evaluation configs extend, is checked against the entry schema, for
both phases (`ok   environments.yaml (EnvironmentsConfig)`).

A key that is going away still loads, for one minor release. `config validate` lists each deprecated key a file
sets under it, with the key that replaces it and the release that removes it, and `--strict` fails on them, so CI
finds them before that release does. Platoon 0.3.0, which renamed `train.task_order` to `train.task_sampler`, said
so of a `train.yaml` that still set the old name:

```console
$ uv run platoon config validate --strict train.yaml
FAIL train.yaml
  train.task_order: deprecated; use train.task_sampler instead (platoon 0.4.0 removes it)
```

`train`, `eval`, `rollout` and `config show` name each one once on stderr and run as before. In Python, loading
such a config warns with `pt.PlatoonDeprecationWarning`, a `FutureWarning`;
`warnings.simplefilter("error", pt.PlatoonDeprecationWarning)`, or pytest's
`-W error::platoon.PlatoonDeprecationWarning`, turns them into errors.

Once the release a deprecation names removes the key, a config that still sets it does not load (exit code 78): the
key is unknown, and its problem says what replaces it and which release removed it. For that `train.yaml`, since
platoon 0.4.0:

```console
$ uv run platoon config validate train.yaml
FAIL train.yaml
  train.task_order: unknown key; platoon 0.4.0 removed it; use train.task_sampler
```

A service option that a release removed fails the same way as the config loads, `--no-resolve` or not
(`environments.0.service_kwargs.cache: unknown key; platoon 0.3.0 removed it; use the site profile's cache, ...`), and
so does a value a release removed, in a file or an override: `eval.strategy=none` is the `none` that platoon 0.3.0
removed, not a null (`eval.strategy=null` leaves `strategy` unset). The [changelog](../../CHANGELOG.md) lists what each
release removed, with what replaces each form.

### 5. Keep keys strict

Unknown keys are errors, never silently ignored: a misspelled key fails with its full dotted path and, when a
field name is close, a suggestion. Values of the wrong type fail too (`'never' is not one of ...`), and YAML that
reads an exponent without a dot as text (`3e-5`) is still read as a float in a float field.

Some fields accept any keys, because their contents belong to something else:

| Field | Keys belong to |
|---|---|
| `environments[].service_kwargs` | The environment service (for example `dataset_dir` and `provider` for Harbor), whose keys are checked against what it takes when the config loads |
| `environments[].harness_options` | The harness: a TOML spec checks them against its `[options]` when the run starts; a function harness reads them as `ctx.options` |
| `environments[].eval_overrides.harness_options` | The same harness, for evaluation rollouts: merged over `harness_options`, key by key |
| `environments[].dataset_kwargs`, `eval_dataset_kwargs` | The dataset loader, or a service's task listing, which takes `split_name`, `limit` and `shuffle_seed` |
| `environments[].workflow_kwargs`, `eval_workflow_kwargs` | The rollout workflow |
| `train.renderer_kwargs`, `inference.renderer_kwargs` | The renderer; unknown attributes fail when the renderer is built |
| `train.loss_fn_config` | The loss function |
| `*.rollout_config.extra` | Your own rollout code |

A plugin that needs its own config sections subclasses the trainer config and names the class on an environment
entry with `trainer_config: my_plugin.config:MyTrainerConfig`; the config then validates against that class. The
class is read from the config files before the overrides, which are checked against it, so set `trainer_config` in a
file: an override that changes it is an error. A `trainer_config` that does not import, or that names something
other than a class, is a configuration error too (exit code 78), since the config would otherwise validate without the
class's sections and checks.

Two keys people often get wrong: the learning rate is `train.optimizer.learning_rate`, and per-call sampling
settings live in `train.workflow_config.rollout_config.inference_params` (see [Train and
evaluate](train-and-eval.md#sampling-defaults)).

### 6. Find what a run used

`train`, `eval` and `rollout` resolve the config once, write it into the run directory, and then load that copy,
so the record is exactly what ran:

| Command | Run directory |
|---|---|
| `platoon train` | `<log_path>/<stats.experiment_name>/<stats.trial_name>/` |
| `platoon eval` | `inference.output_dir`, default a new `<runs>/eval/<policy>/<UTC time>/` for each start, where `<runs>` is the site profile's `runs`, else `runs` |
| `platoon rollout` | `--output-dir`, default `runs/adhoc/<UTC time>-<task>/` |

Two files go there, replacing any older copies:

- `config.resolved.yaml`: the merged config after `extends:` and overrides, with its input and output paths
  absolute, so it loads from any directory, and each Harbor entry's sandbox provider named. Its header lists the
  source files in merge order and every override:

  ```yaml
  # Written by platoon: the merged config after extends and overrides. Loadable as a config file.
  # source: /work/examples/config-layers/environments.yaml
  # source: /work/examples/config-layers/train.yaml
  # source: /work/examples/config-layers/smoke.yaml
  # override: train.workflow_config.rollout_config.timeout=120
  environments:
  - service: python://platoon.envs.number_search.env:NumberSearchEnvironment
  ...
  ```

- `lock.json`: what the code was. `versions` (platoon, tinker, tinker_cookbook, openreward, hpc_sandbox) and
  `commits`, the git commit of each of those installed from a git URL (the commits of hpc-sandbox between two
  releases share a version, so only the commit tells them apart); `platoon_git_sha` and `platoon_git_dirty` (when
  platoon runs from a git checkout), `python`, `hostname`, `started_at` and `argv`; `deprecations`, each deprecated
  key the config sets (`path`, `replacement` and `removed_in`, the release that removes it); `image_pulls`, when
  `PLATOON_IMAGE_PULLS` names the file of lines `python -m hpc_sandbox pull` printed in the job, each image's
  `reference`, `path`, `sha256` and `manifest_digest`
  ([Choose a container runtime](clusters.md#5-choose-a-container-runtime-for-sandboxes));
  `components`, each component the config names (named environment, harnesses, program, reward processor, service
  and the other import strings) with what it resolved to and the distribution and version that provide it, and for
  a harness with options (a TOML spec's `[options]`) the options each entry ran it with, defaults included, per
  phase (`options: {<entry>: {train: {...}, eval: {...}}}`: a training run records `eval` only where
  `eval_overrides` changes it, and an evaluation or a `platoon rollout` records the phase of the split it runs);
  `extensions`, each of those distributions besides platoon with its requirements on platoon and how it was
  installed (`direct_url` shows an editable checkout); and `tool_call_parsing`, how the policy's responses are
  parsed: with the `chat_template` renderer, the tool-call parser chosen for the model's template ([Models and tool
  calling](models-and-tool-calling.md#set-the-parser)). When an environment entry sets `exclude_tasks`, `train` and
  `eval` add `excluded_tasks` once they have listed the tasks: the ids left out of each list
  ([Leave tasks out](train-and-eval.md#leave-tasks-out)). When entries set training settings of their own, `train`
  adds `entry_settings`: what each entry's settings change from the run's
  ([Settings per environment](train-and-eval.md#settings-per-environment)). A start that cannot load the per-task
  statistics its checkpoint names adds itself to `tasks_state_lost`: its number, the checkpoint, why, and the older
  checkpoint whose statistics it loaded instead, if any ([Per-task statistics](train-and-eval.md#per-task-statistics)),
  and one whose sampling inputs (the task sampler, the batch size, the label column, the entries' order, an entry's
  weight, seed or training tasks, a group admission rule and its cap) differ from its checkpoint's adds itself to
  `sampling_changed`: its number, the checkpoint and the changes
  ([Checkpoints and resume](train-and-eval.md#checkpoints-and-resume)).

To repeat a run, pass its record back as the config: `platoon train <run dir>/config.resolved.yaml`. Use a new
`stats.trial_name` for a fresh run; reusing the same run directory resumes it (see [Checkpoints and
resume](train-and-eval.md#checkpoints-and-resume)). An evaluation's record names its directory, so `platoon eval
<run dir>/config.resolved.yaml` resumes that evaluation; an evaluation never resumes another policy's rollouts
([Evaluate](train-and-eval.md#3-evaluate)).

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | A runtime failure (an exception while running) |
| 64 | A usage error: an unknown command or a missing argument |
| 78 | A configuration error: invalid YAML, an unknown key, a bad value or a missing file, or a config a run cannot start with ([Troubleshooting](troubleshooting.md#the-command-exits-with-code-78) lists them). Restarting will not help |

## Next steps

- [Train and evaluate](train-and-eval.md): what the `train`, `eval` and `inference` sections control.
- [Troubleshooting](troubleshooting.md): what the common errors mean.
- [RFC 0008](../rfcs/0008-config-and-cli.md) proposes the next steps for the config and CLI; the [RFC
  index](../rfcs/README.md) explains how proposals are made and accepted.
