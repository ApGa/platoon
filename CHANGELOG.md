# Changelog

What each release of platoon changes. Platoon's version is below 1.0: a minor release (0.3.0) may change the Python
API, the config format or the command line, and a patch release (0.2.1) only fixes bugs. To stay on one minor
release, require a range such as `platoon[tinker,ors]>=0.2,<0.3`.

## 0.3.0 (unreleased)

- **`train.task_sampler` names how a run draws its training tasks.** It takes the values of `train.task_order`,
  which it replaces: `auto`, the default, draws a mixture in its weighted fair order and a single environment in the
  dataloader's shuffled passes, and `weighted` draws a single environment in the weighted order too. The order a run
  draws is unchanged. `train.task_order` still loads, with a deprecation warning, until platoon 0.4.0 removes it;
  `lock.json` records it and `platoon config validate --strict` fails on it. Beside another `task_sampler`,
  `task_order: auto` (its default in 0.2.0, as a dump of that release's `config show --defaults` spells it) is
  ignored, with a warning that says so, and `task_order: weighted` does not load
  ([Train on several environments](docs/how-to/train-and-eval.md#train-on-several-environments)).
- **Each environment of a mixture sets the training settings of its own groups.** An entry's `workflow_config` is laid
  over `train.workflow_config` for its groups: its `group_size`, its baseline (`leave_one_out_baseline`), its
  `min_successful_group_size`, its straggler defaults, its error filters and subagent datum sampling, and its rollout
  settings (`rollout_config`: `max_steps`, `timeout`, sampling and the rest), so a dataset's calibrated step budget and
  a slow environment's fewer, longer rollouts sit beside the run's. Its rollout settings are its evaluations'
  defaults, below `eval.workflow_config` (and in `platoon eval`, below `inference.workflow`), and
  `eval_overrides.workflow_config` sets what its evaluation rollouts change. An entry's `max_staleness` raises the
  staleness bound of its own groups, while `train.max_staleness` stays the bound on how far rollouts run ahead, and
  recovery pruning keeps the checkpoints within the largest bound. Loading refuses a key that acts on microbatches or
  on the run's rollout workers with where to set it, an entry bound below the run's, and an entry's settings beside a
  workflow class that does not apply them (exit 78). Named environments may ship rollout settings
  (`workflow_config.rollout_config`) only. `platoon config show` prints each entry's effective settings with the
  layer each came from (`--entry NAME` for one), `platoon config validate` and `platoon train` say what each entry
  changes, and `lock.json` records it (`entry_settings`). `max_concurrent_rollouts` counts rollouts that groups of
  different sizes share. A label at its `rollout_group_limits` no longer stalls the others, in every config that sets
  limits: its waiting rows neither fill the pending batch nor count toward the batches `train.max_staleness` lets the
  run draw ahead, and past four batches of them its new rows are skipped and counted (`train/batch/skipped_rows`). So
  a capped label whose limit cannot keep up with its share trains less than its weight, and the rows still waiting
  when a run ends never train (`train/batch/parked_rows` counts them each step); `platoon train` and `platoon config
  validate` warn of a `max_concurrent_rollouts` that the capped labels' groups can fill. Each step also logs each
  label's `stale_groups` and `num_steps`
  ([Settings per environment](docs/how-to/train-and-eval.md#settings-per-environment)).

## 0.2.1 (2026-10-19)

### Fixes

- **A busy node no longer makes the trainer kill rollout workers that were already forked.** On a node whose CPUs are
  all busy, the trainer's event loop could be held up for longer than the fork server's one-minute deadline. When it
  ran again, it gave up on workers the fork server had forked meanwhile, killed them and started them again by exec,
  each importing the worker's stack itself, which loaded the node further. The trainer now reads what the fork server
  sent before it judges a deadline, so an answer, or the hello of a server that finished importing, is used however
  late it is read, and a fork fails only once the server has answered no request for a minute (one that answers the
  requests before it is working through them, in order). A fork server still importing the stack after the worker's
  startup budget is no longer killed and replaced by another that imports it from the start while the workers wait for
  that one: they start by exec meanwhile, and the server forks the next ones once it has imported the stack. If it
  then finds that it cannot serve (a module did not import, a thread runs in it), the trainer logs why and starts no
  other, as for a server that says so in time.

## 0.2.0 (2026-10-19)

The first release on PyPI, numbered 0.2.0 because the original platoon gave its version as 0.1.0, so that the
two never share one. It replaces the original platoon, whose AReaL-based trainer, standalone inference
scripts and plugin layout are gone. The original code is at the tag
[`archive/v0.1`](https://github.com/ApGa/platoon/tree/archive/v0.1), and the code that the
[Recursive Agent Optimization (RAO)](https://arxiv.org/abs/2605.06639) paper's experiments ran on stays on the
[`apga/rao-snapshot`](https://github.com/ApGa/platoon/tree/apga/rao-snapshot) branch.

### What it does

- **Trains agents with reinforcement learning through a Tinker-compatible training service**, hosted Tinker or a
  self-hosted server such as [miles-tinker](https://github.com/neulab/miles-tinker). `platoon train` runs groups
  of episodes per task, builds training data from the exact sampled tokens and centers rewards within each group
  (mean or leave-one-out baselines, with depth weighting for recursive agents). Each step logs its reward and its
  rollouts' outcome counts, and a rollout whose environment could not open its session counts as an error.
  `train.label_by` names the dataset column that labels the tasks, by default the mixture entry: each step's metrics
  and each evaluation's break down by its labels (the 64 most frequent), `train.rollout_group_limits` and the
  straggler policies of training and evaluation key on them, and a column that no training task has stops the run
  before it starts ([Label tasks by a column](docs/how-to/train-and-eval.md#label-tasks-by-a-column)).
- **Any harness.** A harness is a Python function (`@pt.harness`) or a command-line agent described by a TOML
  spec. The built-ins are `platoon-codeact`, `platoon-react`, `platoon-bash`, `mini-swe-agent`, `openhands` (an
  OpenHands SDK agent), and `platoon-oracle` and `platoon-noop`, which run a task's reference solution or nothing,
  to check a dataset before training.
- **One harness per agent, configured by options.** A TOML spec declares its options in an `[options]` table (type,
  default, description, allowed values or range), which `platoon harness show` lists and its launch renders; a
  config sets them in an environment's `harness_options`, on the command line like any key
  (`environments.0.harness_options.num_retries=5`), and for evaluation rollouts only in `eval_overrides`. They are
  checked before anything runs, an unknown or badly typed one is a configuration error (exit 78) that names its
  key, and `lock.json` records the options each harness ran with, defaults included. `openhands` takes
  `reasoning_replay`, `request_timeout_s`, `num_retries` and `condenser`: its defaults train, and Harbor's
  `openhands-sdk` adapter's settings are an evaluation's options, from the same toolchain
  ([RFC 0016](docs/rfcs/0016-component-configuration.md)).
- **Harnesses inside task sandboxes.** A command-line harness runs in a task's own sandbox, also one without a
  network: the gateway serves each rollout on a Unix socket of its own, which a route relays into the sandbox, and
  the harness runs from a toolchain built once per site (`platoon toolchain build`), under `toolchains` in the site
  profile's cache, else `~/.cache/platoon/toolchains`, and mounted read-only; a run never builds one, and refuses to
  start, naming the `platoon prepare` command, while it is missing. A spec's `[budget] enforce = "gateway"` caps its
  policy calls at the gateway, a harness in a task sandbox may make only policy calls, `on_harness_exit` says whether a
  harness that ran out of time or crashed is graded or left out, and `rollout_config.sampling_precedence: config` has
  policy calls sample as the config says, never as a harness asks.
- **A recording model gateway.** Every model call, from Python (`ctx.chat`) or from any client over
  OpenAI-compatible HTTP, is rendered, sampled, parsed and recorded with its exact tokens. The OpenAI Responses and
  Anthropic Messages dialects are experimental. An MCP tool plane serves each agent the environment's tools. A
  sampling call fails after the sampling timeout, which a config sets beside its model
  (`train.sample_timeout_seconds`, `inference.sample_timeout_seconds`); without the key, `TINKER_SAMPLE_TIMEOUT_SECONDS`
  sets it, else 600 s. The key wins over the variable, which a gateway job may export for every client it runs, a
  training run's rollout workers sample under the trainer's bound, and `platoon train` and `platoon eval` log the
  bound with where it came from ([The sampling timeout](docs/how-to/train-and-eval.md#the-sampling-timeout)).
- **Evaluation with hosted models.** `platoon eval` and `platoon rollout` can evaluate a model behind any server
  that speaks the harness's own dialect: `inference.hosted` names the base URL and the variable that holds its key,
  and `inference.model_name` the model. The gateway forwards each call unchanged and treats tools as training does,
  so trajectories, step budgets and usage are still recorded, and the key never reaches the agent's sandbox.
  `inference.judges: false` turns off the model calls that judge agents rather than solve the task
  ([Evaluate a hosted model](docs/how-to/evaluate-hosted-models.md)).
- **Tool calls for any chat template.** The `chat_template` renderer renders with the model's own template and
  reads tool calls with a parser chosen by name, by a round trip through the template (`tool_call_parser: auto`),
  or written by you (`module:attr`).
- **Environments as services:** OpenReward (ORS) environment classes, in process or on environment servers;
  Harbor task directories, graded by their own tests in `process`, Docker, Singularity/Apptainer or enroot
  sandboxes (enroot takes the Singularity provider's option names); step environments through `pt.gym`; and task
  tables. An environment entry's `exclude_tasks` leaves out tasks that cannot be solved or graded as written, and
  `lock.json` records which.
- **Every training task trains.** A single environment's tasks are drawn in seeded shuffled passes that go on from one
  epoch to the next: an epoch holds the whole batches its tasks fill, and the tasks left over start the next epoch
  instead of being left out of every one. `environments` may also list several entries, each with its own service,
  harness, program and options. Training interleaves their tasks in a weighted fair order: `weight` is `tasks`, the
  default, each entry's number of training tasks, which draws every task of the mixture equally often, or a number,
  and the entries share the slots in proportion. The epochs hold as many tasks as the entries do, filled to whole
  batches, each going on with the order where the one before it stopped, so an entry that gets fewer slots than it has
  tasks draws the rest in the next epochs, not the same ones every epoch. A resumed run draws the batches the run it
  resumes would have drawn. `platoon train` logs each entry's share and how often each of its tasks is drawn over the
  planned steps, and warns when the weights leave an entry's tasks out of the run or draw one entry's tasks more than
  4 times as often as another's; `platoon config validate --tasks` lists the tasks and prints the same before a run
  ([Train on several environments](docs/how-to/train-and-eval.md#train-on-several-environments)).
- **Harbor datasets as published.** A Harbor environment names a dataset (`hf://org/name@commit`, or a local
  directory) that `platoon harbor fetch` downloads once into a content-addressed cache and pins with a lock, which
  rollout workers read offline; `platoon harbor images` builds the tasks' images once per site from their
  Dockerfiles, sharing one image among tasks whose build steps match, and an entry that names no image map uses
  `images/<provider>.json` in its cache, unless it serves a directory. Tasks may sit at any depth under a registry,
  a separate verifier grades a task's artifacts in a sandbox of its own, and a dataset's reward contract says which
  of its rewards trains. A dataset whose tasks report that reward under different names is one environment: the
  contract lists the names (`reward: [reward, correctness]`), and each reading must report exactly one of them, from
  one reward file; a group whose rollouts trained different names does not train (`reward_key_mismatch`). Each
  rollout's record names the reward key it trained (`misc.outcome.reward_key`), each step counts its rollouts by it
  (`train/batch/reward_key/<name>`), and `platoon eval`'s report by `reward_key_counts`. A fetch records index
  fields from the dataset's files and, with `index: [{metadata: [...]}]`, from each task's `task.toml` `[metadata]`
  table, records what each task's sessions upload and the digest of each of its input files, and refuses tasks that
  upload more than `refuse.uploads_over_mb`. `holdout` holds out whole groups of a dataset's tasks for evaluation, a
  share of each source, and lists them as the `eval` split and the rest
  as `train`, with a report of each split's makeup; tasks that share an input file (a text file compared as text,
  whatever its line endings) are always in one group, so no held-out task holds data that training has seen, and
  `holdout.ignore_inputs` names files that every task holds by design
  ([One dataset, one environment](docs/how-to/sandboxes-and-harbor.md#one-dataset-one-environment)). A run's
  dataset has each task's index fields as columns, as text, which `train.label_by` can name; a field named like
  one of platoon's own columns is left out, and a configuration error when `label_by` names it.
  `platoon harbor fetch`, `verify` and `images plan|build|verify|alias` take a training or evaluation config, with
  overrides, and act on each Harbor entry's dataset as the run would, `images` with the entry's image map and its
  sandbox provider's builder; `fetch` leaves a dataset pinned to a commit alone when its lock is current (`--repair`
  fetches anyway), and fetches a lock written before input files were recorded once more. Fetches of one lock, and
  builds of one image, take turns on file locks, also on several nodes, so each runs once. Fetch config files are
  deprecated: an entry of the run config says the same. A build step that fetches remote content, an `ADD` of a URL
  or of a git repository, also one whose URL an `ARG` or `ENV` value holds, is built as Docker builds it: the `oci`
  builder runs it with BuildKit, which verifies its `--checksum` and does not extract a remote archive, the build key
  covers it as written, and the `def` builder, which cannot fetch, refuses it and names `oci`. The `oci` builder also
  builds a Dockerfile whose `# syntax=` directive names a 1.x release of Docker's Dockerfile frontend, with the one
  built into SingularityCE's BuildKit, which cannot load a frontend image; it refuses any other frontend.
- **One command prepares a config.** `platoon prepare CONFIG` makes, once per site, what the config's runs need and
  never make themselves: each Harbor dataset fetched into its lock, the images its tasks build, with the builder of
  its sandbox provider, into its image map, its harnesses' toolchains, and the policy's tokenizer, into default
  locations under the site profile's cache, else `~/.cache/platoon`. Each step does nothing once current, two prepares
  at once take turns, also on two nodes, and `--dry-run` says what is missing
  ([Prepare a config](docs/how-to/sandboxes-and-harbor.md#prepare-a-config-once-per-site)). It never replaces a
  toolchain built from another recipe, which a run of another config may be running from: the step names the forced
  build that does. Before any rollout, `platoon train`, `eval` and `rollout` refuse (exit 78) a config whose toolchains,
  or whose fetched tasks' images, are not built, naming the `platoon prepare` command for their config, overrides
  included. On a laptop or a single node, `--prepare` on any of them runs that command first, with its lines on stderr,
  and starts the run only once the config is ready, else exits with prepare's status; on a cluster, prepare once on a
  CPU node instead, since in a GPU job the flag spends GPU time building.
- **One run per run directory.** A run's first start writes `identity.json`: its model, renderer and held-out
  tasks. A resume that would change them stops with exit 78 before it opens the training service, and records why in
  `lock.json`; training that evaluates warns when its evaluation lists training tasks
  ([Checkpoints and resume](docs/how-to/train-and-eval.md#checkpoints-and-resume)). `platoon eval` writes each start to
  a new directory, `runs/eval/<policy>/<UTC time>` (the checkpoint, else the model), unless `inference.output_dir` names
  one, and never resumes another policy's rollouts: a directory whose rollouts another model, checkpoint, renderer or
  hosted base URL made, or with judges switched, stops it with exit 78 before anything in the directory changes
  ([Evaluate](docs/how-to/train-and-eval.md#3-evaluate)).
- **Cadences in training steps, concurrency in rollouts.** `eval.every` evaluates the eval split every that many
  training steps, and `checkpoint.every` saves a permanent checkpoint as often; 0, the default of both, does neither,
  while the launch, recovery and final checkpoints are saved either way
  ([Checkpoints and resume](docs/how-to/train-and-eval.md#checkpoints-and-resume)). Each evaluation logs the step it
  started at (`eval/step`), and the first is at step `every`, as in the original platoon: `eval.at_start: true`, false
  by default, also evaluates the starting policy, at step 0, and holds the first update until that evaluation ends, so
  that the held-out curve starts at the base model
  ([Evaluate during training](docs/how-to/train-and-eval.md#evaluate-during-training)). A start that resumes, by hand or
  after a failure, evaluates at the multiples of `every` from the step it resumes at, and that step only if the run
  directory's `evaluations.jsonl`, which records each evaluation that completed, does not list it: that step's
  evaluation runs again if a crash cut it short, and does not run twice
  ([Evaluation in a resumed run](docs/how-to/train-and-eval.md#evaluation-in-a-resumed-run)).
  `train.max_concurrent_rollouts` and `eval.max_concurrent_rollouts` count the rollouts that run at once, a whole group
  at a time: by default every rollout of a batch in training, and 256 in an evaluation. **Breaking:** an `eval` or
  `checkpoint` section that sets neither `every` nor the deprecated `strategy` no longer evaluates, or saves a permanent
  checkpoint, each epoch, and `every` alone no longer counts epochs: set it in steps (an epoch is the training tasks
  over `train.batch_size`). An evaluation's default concurrency counts 256 rollouts, not 256 groups, which differs once
  its `group_size` is above 1.
- **Evaluation that rolls out as training does.** A training config's `eval.workflow_config` is laid over
  `train.workflow_config`: an evaluation during training keeps training's rollout settings (`rollout_config`: step
  and time limits, sampling, `extra`) and rollout-worker settings (`rollout_isolation`, `worker_start_method` and the
  other `worker_*` keys) unless it sets them, while the keys that shape groups, advantages and filters keep
  evaluation's defaults (one rollout per task, no filters, no straggler cutoff). A key it sets replaces training's;
  `rollout_config` and its `inference_params` merge key by key, `null` unsets a value training sets, and `extra`
  replaces training's whole ([Evaluate during training](docs/how-to/train-and-eval.md#evaluate-during-training)).
  **Breaking:** an evaluation that relied on evaluation's defaults for these settings (the harness's own step limit,
  no time limit, the gateway's sampling and 4,096-token completions, in-process rollouts) now takes training's; set a
  key under `eval.workflow_config` to keep the old value.
- **Harbor grading that agents cannot forge.** Grading first kills every process the agents left running
  (`Sandbox.kill_processes`, part of the sandbox provider contract). Where the tests run in the sandbox the agents
  worked in, it makes the tests' and the reward's directories afresh; a separate verifier gets the task's artifacts
  as directories and regular files only. In either, grading kills what the tests left running and reads only
  regular reward files. A reward file of any other kind scores 0, and so does an attempt whose agents removed,
  moved or locked their workdir (`workdir_missing`), without running the tests; a grade whose processes cannot all
  be killed is an untrainable error
  ([How grading keeps the agent out](docs/how-to/harbor-grading.md#how-grading-keeps-the-agent-out)).
- **Nothing a stopped rollout started keeps running.** Rollout workers label the sandboxes they start, and the
  sandboxes a killed or hung worker left are reaped by label. Every sandbox of a run also carries the run's
  directory as its `platoon.run` label, so a job can reap what its own run left and no other run's. A rollout's
  deadline counts its sandbox starts and its grading, so a slow verifier does not kill the worker that grades it.
  Without a `rollout_config.timeout`, the deadline counts the harness's own time limit where the environment says
  what a rollout needs beyond it, as Harbor's does: the earliest of its spec's `timeout_s` and a Harbor task's
  `[agent] timeout_sec` (times `agent_timeout_multiplier`). So a Harbor config needs no `timeout` to bound its
  rollouts, and an evaluation's worker processes (`use_subprocesses`) let a harness run as long as its task allows.
- **Rollout workers that start in seconds.** With `rollout_isolation: process`, a rollout's worker is forked from a
  fork server that imported the worker's stack and built the run's tokenizer and renderer once
  (`worker_start_method: forkserver`, the default), instead of a new interpreter that imports it (`exec`: about 20 s,
  a minute when 64 start at once on one node). A forked worker is the process an exec'd one would be, and one the
  fork server cannot fork starts by exec. On GPU nodes too: neither the fork server nor a worker sees a GPU or starts
  the CUDA driver, whose thread would keep the fork server from forking. The singularity provider's
  `rootfs_layout: fuse-overlay` (hpc-sandbox 0.3) starts a sandbox in about a second over one extraction of its image
  per node, and a run that sets a layout its provider cannot use is refused when its tasks are listed
  ([The singularity provider's root filesystem](docs/how-to/sandboxes-and-harbor.md#the-singularity-providers-root-filesystem)).
- **Runs that start again by themselves.** With `watchdog.max_restarts: N`, `platoon train` trains in a child process
  and starts it again from the newest checkpoint after any failure but a configuration error (exit 78) and an
  interrupt: the watchdog's exit, a crash, a training service that stalls or dies. It stops after `N` restarts, or after
  three starts in a row that saved no new checkpoint. A start asks the training service for its training client until
  the service accepts it, for up to 20 minutes, logging each refusal, so no delay is needed while a service still holds
  a dead start's session. `platoon train` reaps the sandboxes labelled with its run when it starts, before each restart
  and when it exits, `platoon eval` those labelled with its output directory when it exits, and `platoon sandbox reap
  RUN_DIR` those of a run that was killed; each logs how many. A run directory records the process that runs it
  (`process.json`): `platoon sandbox reap` refuses a run that still runs, and `platoon train` and `platoon eval` a
  directory that another of their processes runs. `python -m platoon.train.tinker.restart_wrapper`, which restarted on the watchdog's exit
  only, is deprecated, and 0.3.0 removes it ([Long runs](docs/how-to/train-and-eval.md#long-runs)).
- **Subagents.** An agent starts subagents with `ctx.agents.spawn`, under step budgets and depth limits, with
  forked environment state and optional verifiers. One rollout trains every agent in the tree, and episode
  programs (`@pt.program`) compose episodes and shape rewards. The default program, `single`, gives subagents the
  root's success when `propagate_root_success` is set (after scoring them 0 when `skip_subagent_reward_computation`
  is), and `platoon-delegation` is a reward processor for recursive agents.
- **Outcomes that say nothing about the policy** (a crashed grader, a dead sandbox, a timeout) never train as a
  reward of 0, and do not count toward a group's `min_successful_group_size`.
- **A Tinker key that is missing or refused is one line.** `platoon train`, `eval` and `rollout` check
  `TINKER_API_KEY` before they prepare, write or reap anything, and without a key that starts with `tml-` stop with
  one line and exit 78; a hosted model needs the variable its `api_key_env` names instead, checked as early, and a
  report, or an evaluation or rollout that serves no policy, needs neither. A key that the Tinker service refuses
  (HTTP 401, or 403 for what the key may not do) ends each of them with one line that names the key and the service,
  instead of the client's traceback, and exit 78: `platoon eval` starts no rollout after it, and a supervised
  `platoon train` does not restart.
- **Recovery checkpoints that do not fill the disk.** With `checkpoint.keep_recovery: N`, training deletes its
  recovery checkpoints beyond the newest N through the Tinker API, beside training and on a Tinker session of its
  own: never a permanent checkpoint, the one a resume starts from, one that rollouts may still sample from (within
  `train.max_staleness` policy versions) or another run's. A config whose `keep_recovery × recovery_every` does not
  exceed `max_staleness` does not load (exit 78). A resumed run goes on where it left off
  (`checkpoints_deleted.jsonl`); failed deletes are counted (`checkpoint/delete_failures`), and three in a row, or a
  service that cannot delete checkpoints, stop the deletes with a warning while training goes on
  ([Checkpoints and resume](docs/how-to/train-and-eval.md#checkpoints-and-resume)).
- **The same training however an environment runs.** A tool's failure is graded alike in process and on an
  environment server, a served class named by `service_kwargs.env_class` shows agents the same tools as in process,
  and a test plays both modes and compares every token, outcome and datum.
- **Extensions.** Installed packages provide harnesses, episode programs, named environments and reward processors
  by name through entry points, which platoon imports only when a config names them; `platoon extensions list`,
  `show` and `check` list them, describe one and check them ([Extending platoon](docs/extending/index.md)). Each
  ships packages of its own: `platoon` is a regular package, which no extension installs into. The repository's
  first-party extensions (`extensions/`) are built and tested with core, in one uv workspace.
- **`platoon.testing`** runs environments, harnesses and programs on CPU with a scripted policy and a toy
  tokenizer, with no model download.
- **One command line:** `platoon train`, `eval`, `rollout`, `config`, `harness`, `program`, `env`, `extensions`,
  `harbor`, `toolchain` and `viz`, with strict configs that build on each other through `extends:` and load as the class
  their environments' `trainer_config` names, so that a plugin's config sections validate, and a `lock.json` in every
  run directory with the versions of platoon and its main dependencies and the commit of each one installed from git,
  and the images the job pulled for its sandboxes when it names their record
  (`PLATOON_IMAGE_PULLS`). Before anything trains, a run refuses (exit 78) a config whose rollouts could never
  train, such as a TOML harness spec whose settings never apply or whose tools never reach the task, or environment
  grading that cannot grade a subagent's own goal, and one whose `trainer_config` does not resolve or one of whose
  `package` modules does not import. `platoon config validate` resolves every component a config names, as a run
  does when it starts (`--no-resolve` checks the files and named environments alone), takes the run's `key=value`
  overrides, and checks a file of environments alone against the entry schema. An entry's `service_kwargs` are bound
  to what its service takes when the config loads, so a misspelled option fails with a did-you-mean, and a value the
  service refuses when the run builds it is a configuration error that names the entry.
- **A site profile.** Where a site keeps platoon's files and which runtime starts its sandboxes go in one file per
  site, never in a project: `~/.config/platoon/site.yaml`, or the file `PLATOON_SITE` names, with four keys
  ([Write the site's profile](docs/how-to/clusters.md#4-write-the-sites-profile-once)). `cache` is the root of what
  `platoon prepare` makes (`<cache>/harbor`, `<cache>/toolchains`); `runs` holds the run directory of a training
  config without `log_path` (`<runs>/<experiment>/<trial>`, the experiment named after the config file unless it
  names one, a directory that belongs to that file, so that another project's file of the same name is refused rather
  than take over its run) and the directory of each start of an evaluation without `inference.output_dir`
  (`<runs>/eval/<policy>/<UTC time>`), while a relative `log_path` a config names stays relative to the working
  directory; `provider` and `provider_options` start the sandboxes of every Harbor entry that names no provider, under
  the entry's own `provider_options`, and of `platoon sandbox reap` for a run whose record names none. The profile
  cannot weaken isolation: it names only a provider whose class says it isolates its sandboxes (`docker`, `singularity`,
  `enroot`; never `process`) and only the options that keep that isolation (`allow_network_none`, `rootfs_layout`,
  `scratch_root`, `image_cache`, `stage_images`, `gpu`), and `platoon train`, `eval` and `rollout` refuse a Harbor entry
  on an unisolated provider that the run config does not name itself. `~` and `$VAR` expand in it, and a key it does not
  take, or a `PLATOON_SITE` that names nothing, is a configuration error. `platoon train`, `eval`, `rollout` and
  `prepare` print the profile they used and what it gave the config, and refuse a provider from it that the node cannot
  run, quoting it. `config.resolved.yaml` holds what it gave and names each Harbor entry's provider, so a run directory
  loads the same without it, and `lock.json` records the profile's path, SHA-256 and values (`site`) and each Harbor
  entry's provider and options with where each came from (`sandbox_providers`). A training config without an `eval`
  section does not evaluate. platoon's temporary files (rollout workers' files, gateway sockets, sandbox and harness
  state, artifacts and staged toolchains) go to node-local scratch, the first of `$PLATOON_NODE_SCRATCH`,
  `$PBS_LOCALDIR`, `$SLURM_TMPDIR` and `$TMPDIR` that is set, else `/tmp`, rather than `$TMPDIR` alone.
- **Paths relative to the file that names them.** A config file's input paths are relative to that file, as
  `extends:` is: an environment's harness `.toml` spec and the options its service reads files from (a Harbor entry's
  `dataset`, `dataset_dir`, `task_dirs`, `splits`, `lock`, `tasks`, `overlays` and `mounts[].source`, a task table's
  `path`), so a config runs the same from any directory. A split manifest named relative to a `dataset_dir` that the
  same file names stays relative to it. An override's paths and the outputs (`log_path`, `inference.output_dir`, where a
  leading `~` names the home directory) stay relative to the working directory, and `config.resolved.yaml` holds them
  absolute. **Breaking:** these input paths resolved against the working directory before, and nothing falls back to it:
  a relative input that names nothing beside its file stops the command with exit 78, and names the path relative to the
  working directory too when that one exists ([Write and override
  configs](docs/how-to/configs.md#2-build-on-it-with-extends)).
- **Deprecations that say what replaces them.** A config key that is going away keeps working for one minor release,
  with a `pt.PlatoonDeprecationWarning` (a `FutureWarning`, which Python shows) that names the key's path, its
  replacement and the release that removes it. The commands name each one once on stderr, `lock.json` lists them
  (`deprecations`), and `platoon config validate` lists them under each file and, with `--strict`, fails on them. The
  first is `rollout_config.propogate_root_success`, the misspelling of `propagate_root_success`, which 0.3.0 removes,
  and `platoon harbor` warns the same way of a fetch config file, which 0.3.0 removes too.
  0.3.0 also removes `train.rollout_group_key`, now `train.label_by`, and `workflow_config.straggler_policy_key`,
  whose straggler policies key on `label_by`'s labels now; a workflow config that still sets it keys its own on that
  column until then. It removes the `strategy` of `eval` and `checkpoint` as well: `every` alone counts training
  steps, while a config that still sets `strategy` keeps its meaning until then (`epoch` counts `every` in passes over
  the training tasks, `none` turns the event off, and an `every` left out is 1, as it was). And it removes
  `num_concurrent_rollout_workflow_workers` of `train` and `eval`, which counted rollout groups: until then it reads
  as that many groups' rollouts, so `max_concurrent_rollouts` is the old value times the section's
  `workflow_config.group_size`. It removes a Harbor entry's `cache`, `image_map` and `toolchain_root`, and the
  variables `PLATOON_HARBOR_CACHE` and `PLATOON_TOOLCHAINS`, which the site profile's `cache` replaces: until then they
  work as before, and a variable set beside a profile's `cache` must name the same directory.

### Install

```bash
pip install 'platoon[tinker,ors]'
```

- Python 3.12 or later, on Linux; CI tests 3.12, 3.13 and 3.14. On 3.14, platoon needs datasets 4.4 and dill 0.4
  or later, whose picklers can hash a dataset there (pip upgrades older ones), and importing the Tinker SDK warns that
  pydantic's V1 core is not compatible with 3.14: the SDK takes only V1's date parsers and typing helpers, with which
  it builds its responses as on 3.12.
- The `tinker` extra pins the Tinker SDK to 0.16.1 and takes tinker-cookbook 0.3: later tinker-cookbook releases
  change what a renderer's parse returns or need a later Tinker SDK, which a future release adopts.
- The `hpc` extra installs hpc-sandbox 0.3 from PyPI for the `singularity` and `enroot` sandboxes. It deletes a
  stopped sandbox's files in the background, and the next sweep on the node deletes what the end of a job cut short;
  where the sandboxes' scratch outlives jobs, start each job with hpc-sandbox's janitor
  ([Sandboxes left by stopped rollouts](docs/how-to/sandboxes-and-harbor.md#sandboxes-left-by-stopped-rollouts)).
- Number search, the first run's environment, ships in the package as the built-in named environment
  `platoon-number-search@1` (`platoon.envs.number_search`), which generates its tasks from a seed. The recipes and
  the runnable examples are in the repository, not in the package: clone it to follow the tutorials. TextCraft is
  the first-party extension `platoon-textcraft` (`extensions/textcraft`), a distribution with a version of its own
  (`pip install 'platoon-textcraft[platoon]'`), which also exports TextCraft-Synth as a Harbor dataset
  (`python -m platoon_textcraft.harbor export`), whose verifier replays the game's action log. `recipes/rao` carries
  the RAO paper's TextCraft-Synth arms over to Tinker: its single agent (`linear.yaml`), its recursive agent
  (`recursive.yaml`: 25 steps for every agent, at most 6 levels of delegation, and each subagent rewarded for
  crafting its own targets, the paper's Dense rewards) and its Sparse ablation (`ablation_sparse.yaml`: every
  subagent takes the root's success). A recursive agent that sizes its children's budgets from its own 200 steps is
  not one of the paper's arms, so the recipe has none.
- `platoon.__version__` and `platoon --version` give the installed version, and the package ships type information
  (`py.typed`).

### Documentation

[The documentation](docs/README.md) has a getting-started guide, tutorials whose examples a CPU test runs, how-to
guides, guides to extending platoon, a command-line, config and Python API reference generated from the code, an
architecture overview and the RFCs, among them the drafts on Harbor datasets, images, grading and multi-container
tasks (0010) and on the public surface and conformance (0017), and the accepted design for component configuration
(0016).
[CONTRIBUTING.md](CONTRIBUTING.md) covers development, Conventional Commits, the checks CI runs, among them the
snapshot of the public API and the import boundaries of core, the first-party extensions and the examples, and how
to cut a release.
