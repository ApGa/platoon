# Troubleshooting

Each entry gives the symptom as you see it, the cause, and the fix. Search this page for the message or the reason
you got.

## First: find out what happened to a rollout

1. **Reproduce one task** with `platoon rollout CONFIG --task ID`. The tree shows every agent's steps, reward and
   outcome, for example `outcome error (platoon-bash@1 exited with code 2)`.
2. **Read the outcome.** Every rollout ends with an outcome whose status says whether the measurement is valid. Only
   `scored` outcomes train; everything else is kept out of training and out of the group baseline, and never
   becomes a reward of 0.

   | Status | Meaning | Trains |
   |---|---|---|
   | `scored` | The environment graded the attempt. A failed attempt is scored too, with a low reward and often a reason such as `no_submit` or `context_overflow` | yes |
   | `invalid` | The environment declared the measurement invalid, or the verifier timed out | no |
   | `error` | An infrastructure failure: the environment, the sandbox, the grader or the harness broke. In training, also a rollout whose environment could not open its session, such as a sandbox that did not start: it records no outcome (`platoon eval` counts it as `raised`), and its `EnvSetupError` is in the trainer's log or its rollout worker's | no |
   | `timeout` | The harness ran out of time (`agent_timeout`) | no |
   | `cancelled` | The run was cancelled before it finished | no |

   The outcome is in each trajectory's `misc.outcome` in `trajectory_collection.json`. `platoon eval` counts
   them in `outcome_status_counts`, and training logs each step's counts as `train/batch/outcome/<status>`, and
   each label's as `train/batch/<column>/<label>/outcome/<status>`: by default a mixture's per entry, as
   `train/batch/environment/<name>/outcome/<status>` ([Label tasks by a column](train-and-eval.md#label-tasks-by-a-column)).
3. **Read the harness log** of a CLI harness: `<rollout output dir>/harness/<agent id>.log` (training:
   `<run dir>/rollouts/train/<policy version>/harness/`; evaluation: `<output_dir>/rollouts/<task>/rollout_<k>/harness/`).
4. **Watch the share of valid rollouts** in training: `train/batch/task_reward_valid/avg` well below 1 means
   rollouts are failing to measure, not that the policy got worse.

## The command exits with code 78

The config is invalid, or a run cannot start with it; nothing ran, and restarting will not help. The message
lists every problem:

```text
platoon: invalid config smoke.yaml:
  override train.batch_sise=4: train.batch_sise: unknown key; did you mean train.batch_size?
```

Common ones:

| Message | Fix |
|---|---|
| `...: unknown key; did you mean ...?` | Fix the spelling. The learning rate is `train.optimizer.learning_rate` |
| `... is not one of 'auto', 'weighted'` | Use one of the listed values |
| `environments has 1 entries; index 1 is out of range` | List indices count from 0; negative ones from the end |
| ``a harness needs an environment `service` to open sessions with`` | Add `service:` to the entry |
| `environments.<i>.service_kwargs.<option>: <path> names <file's directory>/<path>, which does not exist: a relative path in a config file is relative to the file's directory, ...` (or `environments.<i>.harness:` for a `.toml` spec) | Relative input paths resolve against the config file that names them, not the working directory. Write the path relative to the file, or absolute; when the message adds that the path exists relative to the working directory, that is the file it named before ([Write and override configs](configs.md#2-build-on-it-with-extends)) |
| `$PLATOON_SITE names <path>, which is not a file: ...` | `PLATOON_SITE` must name a site profile: fix the path, or unset it to use `~/.config/platoon/site.yaml` when that exists ([Write the site's profile](clusters.md#4-write-the-sites-profile-once)) |
| `invalid config <site.yaml>:` (from `config validate`, `<site.yaml>:` before each line) then `<key>: unknown key; a site profile takes cache, runs, provider and provider_options: ...`, `provider: process runs the agents' commands on this host, without isolation, ...`, `provider_options.<option>: a site profile sets only the options that do not weaken a sandbox's isolation, ...` or `<key>: $VAR is not set` | The site profile says where files live and which isolating runtime starts sandboxes, nothing else. Move another key, the `process` provider, or an option outside its list (`binary`, `extra_start_args`, binds) into the run config that needs it, and export a variable a value names |
| `environments.<i> (<name>): the site profile <path> chooses the sandbox provider <provider>, which this node cannot run: ...` | The profile's runtime is missing on this node (no executable on `PATH`, or no `hpc` extra). Run on a node that has it, or point `PLATOON_SITE` at the profile of the site you are on |
| `environments.<i> (<name>): its sandbox provider, process, runs the agents' commands on this host without isolation, and it comes from the named environment the entry selects (env:), not the run config: ...` | `platoon train`, `eval` and `rollout` run agents on the host only when the run config itself says so: set `environments.<i>.service_kwargs.provider: process` in it, or choose an isolating provider |
| `$PLATOON_HARBOR_CACHE names <dir>, but the site profile <path> gives <cache>/harbor ...` (or `$PLATOON_TOOLCHAINS`) | The deprecated variable and the profile's `cache` disagree on where the files are: unset the variable, or point the profile's `cache` at the directory that holds `harbor` and `toolchains` |
| `log_path: required key is missing; set it, or runs in the site profile ...` | Give the config a `log_path`, or the site profile a `runs`, under which a config without one gets its run directory |
| `<run directory> holds the training run of <other config>: the site profile names a training run's directory after its config file's name, and <config> has the same name, ...` | Two training config files of one name, in different projects, would share the run directory the profile names after it, and one would take over the other's run. Name this config's experiment (`stats.experiment_name`) or give it a `log_path` ([Write the site's profile](clusters.md#4-write-the-sites-profile-once)) |
| ``Every entry of a multi-environment `environments` list needs a `name` `` | Name each entry of a mixture; names must be unique and must not contain `::` |
| `Mixture entries [...] weigh by their tasks (weight: tasks, the weight of an entry that sets none) and [...] by numbers: ...` | A mixture's weights are all numbers, which share its slots in proportion, or all `tasks`, the default, which draws every task equally often: give each entry a number (`1` each for equal shares), or no entry one ([Train on several environments](train-and-eval.md#train-on-several-environments)) |
| `... roll out with harness_rollout, whose harnesses sample the policy, but inference.renderer_name is not set` | Set `inference.renderer_name` in the evaluation config |
| `ValueError: checkpoint.every must be a number of training steps, or 0 for never, got -1` | Set `checkpoint.every` to the training steps between permanent checkpoints, or to 0 for none; `eval.every` takes the same values |
| `checkpoint.keep_recovery: 1 × checkpoint.recovery_every: 3 = 3 must exceed train.max_staleness: 3: ...` (or `... must exceed environments[<name>].max_staleness: 48, the run's largest staleness bound ...`), or `checkpoint.keep_recovery needs train.max_staleness: ...` | Recovery checkpoints within `max_staleness` policy versions may still serve rollouts and are never deleted, so `keep_recovery` must reach past them, and past an entry's own bound: raise `keep_recovery` or `recovery_every` as the message says, or lower the bound it names; set `train.max_staleness` when it is missing ([Checkpoints and resume](train-and-eval.md#checkpoints-and-resume)) |
| `ValueError: eval.workflow_config: min_successful_group_size must be in [1, group_size] (the keys it leaves out keep their eval.workflow_config defaults: group_size: 1, ...)` | A partial section keeps its field's defaults, here one rollout per task: set `group_size` in it too ([Write and override configs](configs.md#2-build-on-it-with-extends)). A straggler quorum above 1 fails the same way. The defaults it lists include the rollout and worker settings it keeps from `train.workflow_config` ([Evaluate during training](train-and-eval.md#evaluate-during-training)) |
| `train: set num_epochs or max_training_steps (with both, training runs the shorter)` | Set one of them |
| `environments.<i>.workflow_config.<key>: unknown key; an entry does not set it: ...` (also under `eval_overrides.workflow_config`) | The key acts on microbatches, which mix entries' groups, or on the run's one pool of rollout workers: set it in `train.workflow_config`, or `eval.workflow_config`, as the message says. An entry sets the keys that act on one group or one rollout ([Settings per environment](train-and-eval.md#settings-per-environment)) |
| `environments[<name>].workflow_config: min_successful_group_size must be in [1, group_size] (the keys it leaves out keep train.workflow_config's values: group_size: 8, ...)` | An entry's settings are laid over the run's: the keys it leaves out keep the run's values, here its group size. Set `group_size` in the entry too. The same holds for a straggler quorum |
| `environments[<name>].max_staleness: 2 is below train.max_staleness: 3, ...` | An entry's bound may only raise the run's, which also bounds how far every entry's rollouts run ahead of training: set `train.max_staleness` to the smallest bound and raise the slower entries' own ([Settings per environment](train-and-eval.md#settings-per-environment)) |
| `train.max_concurrent_rollouts: 12 holds no group of environments[<name>], whose 16 rollouts start together ...` | A group's rollouts start together, so the budget must hold the largest entry's group: raise `max_concurrent_rollouts` (`eval.max_concurrent_rollouts` for evaluation) |
| `train.max_concurrent_rollouts: 12 rollouts, of which the groups of the labels with a limit (train.rollout_group_limits) hold up to 8 (slow: 2 × 4): ...` (a warning) | While the capped labels run as many groups as their limits allow, their groups hold so many rollouts that a group of a larger entry waits for one of them to end, and every group after it with it, so the capped labels set the run's pace: raise `max_concurrent_rollouts` to the number the message gives ([Settings per environment](train-and-eval.md#settings-per-environment)) |
| `Label '<label>' has 80 rows waiting for its limit (train.rollout_group_limits: 1 at once): its new rows are skipped, ...` (a warning) | The label's share of the order asks for more groups than its limit runs at once, so it trains less than its weight: lower its entry's weight or raise its limit. `train/batch/<column>/<label>/parked_rows` and `skipped_rows` show how many of its rows wait and how many were skipped ([Settings per environment](train-and-eval.md#settings-per-environment)) |
| `environments[<name>].workflow_config: the run's workflow, <ref>, does not apply an entry's settings ...` | A workflow class of your own that is no `GroupRolloutWorkflow` never reads an entry's settings: apply them in it and declare `applies_entry_settings = True` on the class, or set them in `train.workflow_config` |
| `train.batch_size: 32 is more than the 20 training tasks, so no full batch exists; ...` | Lower `train.batch_size`, or set `train.task_sampler: weighted`, which repeats tasks into whole batches |
| `train.label_by: no training task has a 'sorce_dataset' value to label it by; their columns are ...` | `train.label_by` names a column that none of the training tasks has, often a misspelling: the message lists the columns they have, a Harbor dataset's index fields among them ([Label tasks by a column](train-and-eval.md#label-tasks-by-a-column)) |
| `environments[<name>]: train.label_by names 'task_id', a column this environment's tasks have (an index field) whose name is that of one of platoon's own columns, ...` | The dataset keeps platoon's own column (`task_id`, `_platoon*`), not the index field of that name: label by another column ([Index fields as dataset columns](sandboxes-and-harbor.md#index-fields-as-dataset-columns)) |
| `TINKER_API_KEY is not set: ...` or `TINKER_API_KEY must start with tml-, ...` | Export a key that starts with `tml-`. `platoon train` always needs one, and `platoon eval` and `platoon rollout` need one when they serve a Tinker policy (a hosted model needs the variable its `api_key_env` names instead), so rebuilding a finished evaluation's report without a key takes `stage=report`. They check it first: without one they prepare, write and reap nothing |
| `the Tinker service at <origin> refused TINKER_API_KEY (HTTP 401: ...)`, or `(HTTP 403: ...)`, or `... refused TINKER_API_KEY when asked for a training client (...)` (or `a sampling client of <model>`) | The service at that address does not take the key (401), or takes it but not for what was asked (403, as a self-hosted service refuses a checkpoint that another key saved); the rest is the service's answer, with the key replaced by its name. Set `TINKER_API_KEY` to a key that service accepts, or point `TINKER_BASE_URL` (`tinker_base_url` in a training config) at the service your key is for, and resume a run with the key that started it. `platoon eval` starts no rollout after it, and a supervised `platoon train` does not restart |
| `environments[<name>].harness: ...` | The harness does not resolve or load: an unknown or ambiguous name, a malformed spec, a `.toml` path that does not exist, a `module:function` whose module does not import or has no such function, or a function that is not an `@pt.harness`. `platoon harness list` shows what exists |
| `environments.<i>.harness_options.<key>: unknown key; ...`, or `... is not a bool`, `... is not one of ...`, `... is above the maximum ...` (also under `eval_overrides.harness_options`) | The harness does not take that option, or not that value: `platoon harness show <harness>` lists a TOML spec's options with their types and allowed values. A spec without options takes none |
| `what the run needs is not prepared on this site:`, then lines such as `environments[<name>].harness: harness <spec> runs from the toolchain <name>, which a run never builds, and it is not built there ...` or `environments[<name>].service: ... build their images from Dockerfiles, which the image map ... lacks ...`, and `prepare it with: platoon prepare CONFIG ...` | A run never builds toolchains or images unasked: run the `platoon prepare` command it names, once per site, on a node with network access, then start the run again (on a laptop or a single node, `--prepare` on the run does both). The run looks under the site profile's `cache`, as prepare does: run both with the same profile ([Prepare a config](sandboxes-and-harbor.md#prepare-a-config-once-per-site)) |
| `environments[<name>].harness: harness <spec> runs from the toolchain <name>, but the toolchain at ... was built from another recipe (...): a run of another config or platoon may be running it, so platoon prepare leaves it as it is; rebuild it in place, while no job uses it, with: platoon toolchain build ... --force` | Another config, or another platoon version, built that toolchain from a different recipe, and runs look it up under the root at every rollout, so neither a run nor `platoon prepare` replaces it. Run the command it names once no job uses the toolchain, or run this config with a site profile whose `cache` is another directory ([Build the toolchain](sandboxes-and-harbor.md#1-build-the-toolchain)) |
| `environments[<name>].program: Unknown program ...`, or `.reward_processor:`, `.env:` | The same for the entry's other components: install the package that provides the name, or name it as `module:attr`. A bare name with several versions installed is ambiguous: add `@version`. `platoon extensions list` shows what is installed |
| `environments[<name>].package: '<module>' does not import (...)` | The module the entry imports first, so that it registers the entry's components, does not import where the config loads: run from where it imports (the examples' READMEs set `PYTHONPATH=.`), or install the package that provides it. The entry's other components are checked once it imports |
| `environments.<i>.trainer_config: '<module:Class>' does not resolve (...)` | The config class the entry names does not import where the config loads: install the package that defines it, or fix the reference |
| `environments.<i>.trainer_config: an override changes the config class from ... to ...` | The class is chosen from the config files, before the overrides: set `trainer_config` in a file |
| `<key>: deprecated; use <replacement> instead (platoon <version> removes it)` | A warning on stderr, and the run goes on: the key works until the release it names. `config validate --strict` fails on it with 78, as does an error warning filter (`warnings.simplefilter("error", pt.PlatoonDeprecationWarning)`). Set the replacement instead ([Write and override configs](configs.md#4-check-the-result-before-you-run-it)) |
| `... entry point ... failed to load: <package> <version> requires platoon<range>, but platoon <version> is running` | The package does not support this platoon. Install a version of it that does; `PLATOON_IGNORE_EXTENSION_REQUIREMENTS=1` loads it anyway, for development |
| `environments[<name>]: harness <spec> is a TOML harness spec, which runs no episode program, and a task table has no grader, ...` | Every rollout would be `invalid`. Play a task table with a function harness and a program that sets the outcome (`program: gym`, or your own) ([Gym adapters and task tables](../tutorials/gym-and-task-tables.md)) |
| `environments[<name>]: program '<program>' never runs: harness <spec> is a TOML harness spec, ...` or `... agents settings never apply: ...` | A TOML harness runs as one agent, with no episode program and no subagents. Use a function harness, or drop the setting |
| `environments[<name>].agents: child_grading: env ...` | Env grading could not grade a child's own goal: a task table has no grader, and a `shared` fork (the default) grades a child by its parent's live environment unless an in-process class defines `platoon_fork`. Use `child_grading: verifier` or `none`, or define `platoon_fork`; a child on a `fresh` (or Harbor `copy`) fork of its own is graded on the whole task ([How subagents are graded](../tutorials/subagents.md#how-subagents-are-graded)) |
| `environments[<name>]: harness <spec> takes no env tools ([mcp] is not enabled), and ... its own tools never reach the environment, ...` | Every rollout would score 0. Use a harness that takes the env's tools over MCP (`platoon-bash`), a function harness, or run the spec in a Harbor task's sandbox (`placement: auto`) ([Placement and built-in tools](../tutorials/cli-harness.md#placement-and-built-in-tools)) |
| `environments '<a>' (<EnvA>) and '<b>' (<EnvB>) both reach the env server ... without a variant; ...` | The mixture's entries would all run that server's default environment. Give each entry its own pool, set `legacy_global_pool: false`, or select the environment with `?variant=<Env>` ([Run environments on env servers](../tutorials/remote-environments.md#step-3-give-each-environment-a-url-pool)) |
| `environments[<name>].service: The env server ... serves [...] and answers requests without a variant with '<default>', its default, ...` | The env name is one of the server's other environments: add `?variant=<Env>` to the service URI, or `variant` to `service_kwargs`. `platoon rollout` meets it when the session opens, as an `env_config` setup error |
| `environments[<name>].service: The env server ... serves ORS env '<Env>' with other tools than env_class ...` or `... env_class '...' does not import: ...` | `service_kwargs.env_class` must name, importably, the class the server serves: the hints read from another version could describe tools the server does not have. Serve the version platoon imports, or drop `env_class` and name the hints with `terminal_tools`, `read_only_tools`, `concurrent_tools` and `tool_meta` ([Point a config at the servers](../tutorials/remote-environments.md#step-2-point-a-config-at-the-servers)). `platoon rollout` meets it when the session opens, as an `env_config` setup error |
| `environments[<name>].exclude_tasks: [...] match no task id or name of the environment (its tasks have ids and names like ...)` | Each listed item is misspelled, or is not a task id or name of that environment: ORS task ids are `<split>:<index>`, a Harbor dataset's are directory names, and a mixture entry names its own tasks, without `<name>::` ([Leave tasks out](train-and-eval.md#leave-tasks-out)) |
| `environments[<name>].exclude_tasks leaves none of the <n> tasks of split '<split>'` | The entry excludes every task it lists for that split: drop some items, or the entry |
| `renderer_kwargs.tool_call_parser: No tool-call parser reads the chat template's own tool calls back ...` | Name the model's parser, or write one: [Tool calls](#tool-calls) |
| `inference.renderer_name configure a Tinker policy, but inference.hosted is set, ...` (or `model_path`, `renderer_kwargs`, `sample_timeout_seconds`, `context_window_length`) | A hosted model serves `inference.model_name` itself: leave the Tinker settings out; `inference.hosted.timeout_s` bounds its calls ([Evaluate a hosted model](evaluate-hosted-models.md)) |
| `train.hosted: unknown key; hosted models (inference.hosted) serve evaluation only, ...` (or `eval.hosted`, `train.backend`) or `inference: unknown key; inference is an evaluation config's section ...` | A training config samples through Tinker: hosted calls hold no token ids to train on. Evaluate the hosted model with an evaluation config instead |
| `inference.backend: unknown key; a hosted model is inference.hosted: ...` | The section is `inference.hosted` now, with a `base_url` that speaks your harness's dialect and the variable its key is in ([Evaluate a hosted model](evaluate-hosted-models.md)). A Tinker server's URL goes in the `TINKER_BASE_URL` variable |
| `inference.hosted.kind: unknown key; ...` (or `require_api_key`, `headers`, `headers_env`, `extra_body`, `drop_params`) | Settings of an earlier hosted design, which the message says what replaced: no kind (the dialect is the harness's), no `api_key_env` for a server without auth, headers and body settings in the harness or a proxy ([Hosted models](#hosted-models)) |
| `inference.hosted: set OPENAI_API_KEY in the environment, which api_key_env names` (or another variable) | Export the key in that variable, or leave `api_key_env` out for a server without auth ([Hosted models](#hosted-models)) |
| `inference.hosted.api_key_env must be the name of an environment variable ...` | `api_key_env` names the variable that holds the key (`OPENAI_API_KEY`), never the key itself: move the key into that variable, and revoke it if the config was shared |
| `inference.hosted.base_url must have no query or fragment, ...` (or `holds credentials`) | Put the key in the variable `api_key_env` names, and leave the query out; a proxy can add what the server needs |
| `inference.hosted.max_concurrent_requests is <n>, below the <m> worker processes that share it ...` | With `use_subprocesses`, each worker process that runs rollouts at once makes a request at a time at least: raise the limit to the number given, or lower `subprocess_max_workers` or `num_concurrent_workers` |
| `inference.hosted: <origin> refused the <dialect> probe: <status> <message>` (or `did not answer`) | The base URL does not serve the model, the key or the dialect; the rest of the message is the server's ([Hosted models](#hosted-models)) |
| `environments[<name>]: inference.judges is false, so no verifier grades this entry's children ...` | The entry's reward processor or program may read child verdicts, which no verifier gives with judges off: set `agents.child_grading: none` on the entry for this evaluation, or keep `inference.judges` on ([Turn judges off](evaluate-hosted-models.md#turn-judges-off)) |
| `inference.workflow.rollout_config.judges is set by platoon eval from inference.judges; ...` | Set `inference.judges: false` instead |
| `train.workflow_config.rollout_config.judges: false is an evaluation's setting ...` (or `eval.`) | Training runs its judges. Turn verifiers off with `enable_subagent_verifiers: false` or the environment's `agents.verifier.enabled: false`, or leave `judges` out |
| `<output_dir> holds rollouts made with model 'a' (now 'b'), ...` (or `backend`, `base_url_origin`, `judges`, `model_path`, `renderer`) | An evaluation never resumes another policy's rollouts: one report would mix two policies or two judges settings, or (with `stage: report`) label the rollouts with settings that did not make them. Nothing in the directory changed. Leave `inference.output_dir` unset, which writes each start to a new `runs/eval/<policy>/<UTC time>/`, or set a new one; or report with the settings the rollouts were made with |
| `stage: report builds the report of rollouts already on disk: set inference.output_dir ...` | Without `inference.output_dir` an evaluation writes to a new directory, which holds no rollouts: name the directory of the evaluation to report, such as `inference.output_dir=runs/eval/<policy>/<UTC time>` |
| `the evaluation stopped after <n> rollout(s), <m> not started: judges_off: a judge agent was refused ...` | A program or harness asked for a judge while `inference.judges` is false. Keep judges on for it, or have the program check `EpisodeContext.judges` ([Turn judges off](evaluate-hosted-models.md#turn-judges-off)) |
| `the evaluation stopped after <n> rollout(s), <m> not started: backend_refused: 401 ...` (or 402, 403, 404, or a 3xx redirect) | The provider refused the run's key, payment, permission or model mid-run, or its base URL redirected: fix it and run again with `inference.output_dir` set to the run's directory, which resumes it |

`platoon config validate CONFIG` reports the problems found while loading (keys, values, files, sections and named
environments) and resolves every component the config names, as the commands that run it do, without running
anything ([Write and override configs](configs.md)); `--no-resolve` skips the components, on a machine without the
config's extensions. The commands that run a config find the rest when they start, before anything trains: the
harness settings whose rollouts could never train (before any task is listed), the run length and batch size, the
mixture's env-server pools, the env names, the classes that `env_class` names and the `exclude_tasks` items
(`platoon train` and `platoon eval`, while they list tasks), the tool-call parser and `TINKER_API_KEY`. Each check
reports every environment entry's problems at once; a config refused for its harness settings has its
`exclude_tasks` items checked once they are fixed.

## context_overflow

**Symptom.** A rollout's outcome is `scored` with reason `context_overflow`, usually with reward 0; or, with
`on_context_overflow: exclude`, an `error` outcome. Harnesses that use the HTTP gateway get a 400 response such as:

```text
This model's maximum context length is 32768 tokens. However, your messages resulted in 32801 tokens.
Please reduce the length of the messages.
```

**Cause.** The prompt filled the context window. With `train.context_window_length` (or
`inference.context_window_length`) set, the gateway clamps each call's completion budget to the space left and
refuses a prompt that leaves none, with the API's own context-length error, so a harness can condense its history
and continue. A harness that crashes after such a refusal has failed because of the policy's own
long context, so the episode is graded as a policy failure.

**Fix.**

- Set `context_window_length` to the model's real window; without it the gateway does not check prompt length.
- Keep prompts shorter: fewer steps (`max_steps`), shorter tool output, or a harness that condenses its history.
- To keep such episodes out of training instead of scoring them, set `on_context_overflow: exclude` on the
  environment entry.

## env_busy

**Symptom.** A tool result reads:

```text
Not run: env busy. The call waited 240 s for other calls on this environment to finish (the limit is 240 s);
it was not executed and does not count. Try again.
```

or a rollout's outcome is `error` with a reason like `grading waited 240.0s for the calls in flight`.

**Cause.** Agents that share one live environment (an episode's root and its subagents, or a harness that makes
parallel calls) take turns on it. Tool calls run one at a time unless the environment declares them safe to
overlap, and grading runs alone after every call in flight. A call that does not get its turn within
`queue_timeout_s` (default 240 s) is refused without running; a grade that cannot start in time is an
infrastructure error rather than a guess.

**Fix.**

- Find the call that holds the environment: a slow command, or a tool that blocks. Shorten it, or bound it (Harbor:
  `tool_timeout_s`, default 300 s).
- Declare tools that are safe to run side by side: `concurrent_tools: [bash, view]` in the Harbor service's
  `service_kwargs`; for ORS environments, `concurrent_tools` in `service_kwargs` or, for an environment that runs
  in process, the class attribute `platoon_concurrent_tools`.
- Raise the limit: `queue_timeout_s: 600` in `service_kwargs` (`null` waits forever). Keep it below your harness's
  own per-call deadline, or the harness gives up on calls that will still run.

## Lease and close timeouts

Every rollout holds a lease on its live environment (an ORS session or a Harbor sandbox), shared by the agents of
the episode. Three limits govern it:

| Limit | Default | When it fires |
|---|---|---|
| `queue_timeout_s` | 240 s | A tool call, a grade, a copy or a fork hook waits longer than this for its turn: it is refused without running (`env_busy`). A copy fork fails with `... was still busy after ...s ..., so it could not be copied` |
| `close_timeout_s` | 30 s | The last agent releases the lease: teardown waits this long for calls in flight, then stops the environment in the background |
| `tool_timeout_s` (Harbor) | 300 s | One `bash`, `view`, `write` or `edit` call runs longer |

A call that has started always runs to its end, even when its caller gave up, so a raised limit never leaves a
half-applied call behind.

## no_submit

**Symptom.** The outcome is `scored`, reward 0, reason `no_submit`.

**Cause.** The episode ended without the environment's terminal tool (a `submit`, `answer` or `claim_done` tool)
reporting the episode finished. Typical reasons: the harness stopped with a final message instead of calling the
tool, the step budget ran out first, or the tool raised an error each time (the errors are kept in the outcome's
metadata).

**Fix.**

- Run the task with `platoon rollout` and read the last steps. `platoon-react` hands a final answer to the
  environment's terminal tool when there is one; other harnesses must call it themselves.
- Raise `rollout_config.max_steps` if agents run out of steps while still making progress.
- If the environment can grade its state without a submission, give the environment class a `platoon_grade()`
  method; platoon calls it when nothing was submitted (environments that run in process only).
- A related reason, `max_tool_calls`, means the session hit its `max_tool_calls` cap; the last reward counts.

## Missing env-server pool variable

**Symptom.** A run with an `ors+http:///EnvName` service fails while listing tasks, or every rollout fails to open
its session, with a connection error for `http://localhost:8080`.

**Cause.** The service names no hosts, so platoon reads the pool from `OPENREWARD_SESSION_URLS_<LABEL>`, then
`OPENREWARD_SESSION_URLS`, then `OPENREWARD_SESSION_URL`, and finally falls back to `http://localhost:8080`.
None was set in the client's environment.

**Fix.** Export the variable in the job that runs platoon, as a comma-separated list of `host:port`:

```bash
export OPENREWARD_SESSION_URLS_MYENV=cpu-node-3:8080,cpu-node-4:8080   # for ors+http:///MyEnv
```

The label is the environment name in the URI (or its `label=`), upper-cased, with every other character as `_`. Or
list the hosts in the URI: `ors+http://cpu-node-3:8080,cpu-node-4:8080/MyEnv`. In a mixture, set
`legacy_global_pool: false` on each remote entry, so a missing variable is not filled from a pool that serves
another environment; a mixture whose entries would reach one server that way is refused with exit 78
([above](#the-command-exits-with-code-78)). [Run on a
cluster](clusters.md#2-start-the-environment-servers-if-you-use-them) shows a pattern for collecting server
addresses.

## A CLI harness exits with an error code

**Symptom.** The outcome is `error` with a reason like `platoon-bash@1 exited with code 2`.

**Cause.** A command-line harness (a TOML spec such as `platoon-bash` or `mini-swe-agent`) ended with a non-zero
exit code that its spec does not map to `completed`. Exit 0 means completed, and the episode is graded. Any other
code is a harness crash, an infrastructure error that does not train, unless the spec's `[exit_codes]` table maps it
to `completed`; `platoon-bash` maps 3 (out of turns) that way, because running out of turns is the policy's doing.
A harness that crashes after the gateway refused an overflowing prompt counts as
[`context_overflow`](#context_overflow) instead.

| `platoon-bash` exit code | Meaning |
|---|---|
| 0 | Gave a final answer |
| 2 | A model request failed: the gateway is unreachable or returned an error |
| 3 | No final answer within its turn limit (graded) |
| 4 | Could not reach the MCP tool server, or an MCP tool call failed |

**Fix.** Read the harness log (above). For exit 2 from a harness inside a task sandbox, check how the sandbox
reaches the gateway: on the host's `127.0.0.1` when it shares the host network, else through its route
(`127.0.0.1:18080` in the sandbox, relayed to the rollout's socket; [Sandboxes and Harbor
tasks](sandboxes-and-harbor.md#3-pick-where-the-harness-runs)). The forwarder's log is
`/tmp/platoon-route/gateway.log` in the sandbox. Run `platoon harness show NAME` to see a spec's launch command,
timeout and exit-code table.

Related outcomes:

- `timeout` with reason `agent_timeout`: the harness was stopped at the earliest of the spec's `timeout_s`, the
  rollout's `timeout` and a Harbor task's own `[agent] timeout_sec` (times `agent_timeout_multiplier`). With
  `on_harness_exit: {timeout: grade}` the env is graded instead, with reason `agent_timeout`
  ([Harness timeouts and crashes](sandboxes-and-harbor.md#harness-timeouts-and-crashes)).
- `error` (`harness_crash`), or a crash graded with reason `harness_crash`: the harness exited with a code its spec
  does not map, and `on_harness_exit.crash` decides. `error` (`harness_crash_gateway`): it crashed after the model
  gateway failed one of its calls, under `crash: grade_if_gateway_clean`; read the worker log for the sampling
  error.
- `error` (`harness_setup`): the harness could not be installed or started. An install command failed, its
  named toolchain is missing or does not match its manifest, `placement: task_sandbox` was asked of an environment
  without a sandbox, or the task sandbox has no network and its environment service sets up no route to the
  gateway (use `placement: local`). A sandbox without a network whose image has no Python for the route's
  forwarder fails earlier, when the session opens (a `harness_setup` setup error): name a toolchain in the spec.
- `error` (`harness_config`): the spec cannot run as configured, for example a launch that renders `{max_steps}`
  while neither the rollout (`rollout_config.max_steps`), the task nor the spec sets a step budget.

## Harbor tasks

| Message | Cause and fix |
|---|---|
| `Skipping Harbor task ...: ... multi-container stack (sidecar services), which platoon cannot run yet (RFC 0010)` | The task has a compose file; platoon refuses it rather than score it without its services. See [What platoon refuses](sandboxes-and-harbor.md#what-platoon-refuses) |
| `No runnable Harbor tasks in ...` | No subdirectory has a `task.toml`, or every task was refused (no `instruction.md` or `tests/test.sh`, a compose file, ...); the message lists the first reasons |
| `Two Harbor tasks are named '...'` | Two directories in `task_dirs` have the same name, and a task's name is its id; this counts tasks platoon refuses too. Rename one, or list only one ([Lay out the tasks](sandboxes-and-harbor.md#1-lay-out-the-tasks)) |
| `Harbor task '...' has no environment.docker_image and no default image` | Set `default_image` in `service_kwargs`; platoon does not build `environment/Dockerfile` |
| `... [agent].user = '...', but rootless sandboxes run every process as one user` | Set `agent_user: ignore` in `service_kwargs` to run the task as the sandbox's single uid |
| `Harbor task '...' is refused: it needs 1 GPU(s), and the '...' sandbox provider cannot give a sandbox GPUs` | Run GPU tasks on a provider with the `gpu` capability, or leave them out of the split |
| `harness ... would run in the task sandbox, which does not share this host's network (its provider does not declare host_loopback) and has no route to the model gateway` | The sandbox provider does not declare that its sandboxes share this host's network. Give the harness `placement: local`, or, in a provider of your own whose sandboxes do share it, declare the `host_loopback` capability |
| `N of the M tasks of split '...' run without a network (...), and the singularity provider starts a sandbox without a network only with allow_network_none: true ...` | The tasks set `network_mode = "no-network"`, and setuid Singularity reserves network namespaces for root unless the site permits them. Set `provider_options: {allow_network_none: true}` where the site does, or leave those tasks out of the split or of the run (`exclude_tasks`). platoon stops here, when it lists the tasks, rather than lose every rollout of theirs at sandbox start |
| `N of the M tasks of split '...' run without a network (...), and the enroot provider starts no sandbox without a network while its provider_options set allow_network_none: false ...` | The tasks set `network_mode = "no-network"`, and the enroot provider's options turn network none off, as a block shared with a Singularity site may. Enroot needs no site permission for it: drop `allow_network_none: false` (or set it to `true`), or leave those tasks out (`exclude_tasks`) |
| `Harbor task '...' builds its verifier's image from tests/Dockerfile, and no image is known for it` | The separate verifier's image must be built first; leave such tasks out until the service has a site image map |
| An `error` outcome after the tests ran | The verifier wrote no `/logs/verifier/reward.txt` or `reward.json`, or wrote something that is not a number (`verifier_no_reward`), or not the [reward contract](sandboxes-and-harbor.md#rewards-the-reward-contract-and-log-classifiers)'s `primary` (`verifier_no_primary`: a `reward.txt` reports `reward`). Run `tests/test.sh` by hand in the task's image |
| An `error` outcome, `verifier_start` or `artifact_collect` | A separate verifier's sandbox could not start (the reason and `setup_failure_kind` are in the outcome), or an artifact that exists could not be copied out of the agent's sandbox (`the sandbox has no tar to pack ...`: the task's image needs `tar`) |
| `scored` 0 with reason `reward_file_not_regular` or `reward_file_too_large` | After the tests ran, a reward file was a symlink, a directory or another file that is not a regular one, had something mounted on it or on `/logs/verifier`, could not be read, or was over 1 MiB; the outcome's `reward_files` metadata names it. Grading makes `/logs/verifier` afresh and kills the agents' processes before the tests, so code the tests ran (the agent's) put it there, and the attempt trains as a failure. If your own tests write such a file, make them write a regular file ([How grading keeps the agent out](harbor-grading.md#how-grading-keeps-the-agent-out)) |
| `scored` 0 with reason `workdir_missing` | When grading began, or when the tests' command was to start in the agents' sandbox, the task's workdir, where the tests run and the artifacts are, was not a directory a command could start in, so the tests did not run. The outcome's `workdir_state` metadata says why: `missing` (the agents removed or moved it, or left a symlink to nothing), `not_a_directory` (they left a file in its place) or `not_enterable` (they took its permissions). The agents did it, so the attempt trains as a failure; their own tool calls fail from then on too ([How grading keeps the agent out](harbor-grading.md#how-grading-keeps-the-agent-out)). If the workdir check itself fails, the grade is an `error` (`verifier_setup`) instead |
| `artifacts_skipped` in an outcome's metadata | Artifacts reach a separate verifier as directories and regular files only, of at most `max_artifact_bytes` each: the paths it names were symlinks, devices, FIFOs or hard links to something else (`symlink`, `special`, `link`), or too large (`too_large`), and the verifier saw them missing. If the task's own solution leaves them so, make it write regular files, or raise `max_artifact_bytes` |
| `error` with failure kind `verifier_isolation` | Grading could not kill the processes left in the sandbox, before the tests, before the artifacts were copied, or after the tests: one survived SIGKILL, the sandbox's `/proc` does not list its processes (a command mounted over it), the sandbox could not list its own processes when it started (an image without `/bin/sh`), or it has no PID namespace of its own, so the kill would reach processes outside it (Singularity where the site's `singularity.conf` says `allow pid ns = no`; the log says the sandbox "shares the PID namespace" of platoon). The grade is not trusted, so it does not train |
| `error` with failure kind `verifier_output` | The sandbox's shell could not report the verifier's output: the image lacks `/bin/sh`, `cat`, `head`, `tail` or `wc`, or the command failed |
| A task whose tests check a server or another process the agent started never passes | Grading kills every process the agents left running before the tests run. Have `tests/test.sh` start what it checks |
| An `invalid` outcome | The verifier timed out (raise `[verifier] timeout_sec` or `verifier_timeout_multiplier`), a classifier matched its log (the kind names it), or the reward contract rejected the reading (`verifier_unexpected_keys`, `verifier_out_of_range`, or the `check` hook's kind) |

## Tool calls

**Symptom.** A run stops at start, with exit code 78, saying that no tool-call parser reads the chat template's own
tool calls back. **Cause.** With the `chat_template` renderer and `tool_call_parser: auto`, platoon renders a
probe turn with three tool calls through the model's template and keeps the parser that reads it back; none did.
**Fix.** Set `renderer_kwargs.tool_call_parser` to the parser the model's serving docs name, set `reasoning_tags`
when the model reasons between other tags, write a parser of your own, or set `none` for a model that does not
call tools ([When no parser fits](models-and-tool-calling.md#when-no-parser-fits)).

**Symptom.** The model writes tool calls but the harness runs none, or a call's text shows up in the response's
content. **Cause.** The call did not parse: it was cut off, its JSON does not decode, or it is in another format
than the parser's. Each call's generation record counts it in `unparsed_tool_calls`, and training reports
`rollout_completion/num_unparsed_tool_calls`; the gateway's log shows the first few with the reason. **Fix.**
Check `lock.json`'s `tool_call_parsing` for the parser in use. A model that keeps writing calls its parser cannot
read needs another parser (or its own); a rising count during training means the policy is drifting from its
format ([Malformed and unparsed calls](models-and-tool-calling.md#malformed-and-unparsed-calls)).

## Hosted models

An evaluation of a hosted model (`inference.hosted`, [Evaluate a hosted model](evaluate-hosted-models.md)):

| Symptom | Cause and fix |
|---|---|
| Exit 78 with `refused the <dialect> probe: 401 ...` | The server refused the key: check the variable `api_key_env` names |
| Exit 78 with `refused the <dialect> probe: 404 ...` naming the model | `inference.model_name` is not a model the server serves; a vLLM or SGLang server serves its `--served-model-name` |
| Exit 78 with `refused the <dialect> probe: 404 ...` about the route (`Not Found`) | The base URL does not speak that dialect, which a harness of the run speaks: point it at a server that does (a LiteLLM proxy speaks all three), or check that it ends where `/v1` ends (`https://api.anthropic.com/v1`) |
| Exit 78 with `did not answer the <dialect> probe: ClientConnectorError ...` | Nothing answers at `hosted.base_url`, or a proxy variable (`HTTPS_PROXY`) sends the call where it cannot go: set `NO_PROXY` for a local server |
| Exit 78 with `refused the <dialect> probe: 3xx the base URL redirects to <origin> ...` | The gateway follows no redirect, so the key never goes elsewhere: set `hosted.base_url` to the URL that answers, such as its `https://` form |
| Rollouts errored with `backend_unavailable: <n> model call(s) failed after retries` | The server kept answering 429, 5xx or nothing through `hosted.max_retries` retries. Lower `hosted.max_concurrent_requests` or `num_concurrent_workers`, raise `max_retries`, then run again with `inference.output_dir` set to the run's directory: resuming reruns them |
| `the evaluation stopped after <n> rollout(s), <m> not started: backend_refused: ...` | The server refused the run's key, payment, permission or model mid-run (401, 402, 403, 404), or redirected the call (3xx): fix it and run again with `inference.output_dir` set to the run's directory, which resumes it |
| A harness log with a 400 `tool_choice=... is not supported by the model gateway` | The harness forces or names a tool call, which training cannot enforce, so the hosted model is not asked to either: configure the harness to leave `tool_choice` unset, `auto` or `none` |
| A harness log with a 400 about `temperature`, `top_p` or another parameter | The model refuses a parameter the harness or `inference_params` sends: drop it from the harness's settings or `inference_params`, or let a proxy drop it (LiteLLM's `drop_params`) |
| A harness log with a 400 about `reasoning_content` in a message | A function harness sends back the reasoning `ctx.chat` returned, which the server refuses in requests: put a proxy in front of it that drops the field |
| Answers end mid-thought with `finish_reason: length` (`stop_reason: max_tokens`) | The completion budget is too small for a model that thinks: raise the harness's own budget, or `inference.workflow.rollout_config.inference_params.max_completion_tokens` for calls that set none |
| `calls_without_usage` in the report, and calls left unpriced | A Chat Completions harness streams without `stream_options.include_usage`, so the server reports no usage: ask for it in the harness |

## Sandbox providers

| Message | Fix |
|---|---|
| `The singularity provider needs the hpc-sandbox package: install the hpc extra, e.g. pip install 'platoon[hpc]', ...` (or `enroot`) | Install the `hpc` extra ([The hpc extra](../getting-started/install.md#the-hpc-extra)). In a checkout, name every extra you use with `uv sync`, since it uninstalls the others ([Extras](../getting-started/install.md#extras)) |
| `The docker provider needs the 'docker' CLI on PATH` | Install Docker, or set `provider_options: {binary: podman}` |
| `Could not start a container from ...` | The image cannot be pulled or started; pull it by hand to see why |
| `the enroot provider passes NVIDIA GPUs only, not gpu='rocm'` | Use `provider: singularity` for AMD GPUs |

## Restarts and leftover sandboxes

What `platoon train` with `watchdog.max_restarts`, and the reaping of the sandboxes a run leaves, report
([Long runs](train-and-eval.md#long-runs)):

| Message | Cause and fix |
|---|---|
| `Training exited with code 1; restarting it (restart 1 of at most 20)` | A start failed, and the next resumes from the newest checkpoint. The start's own log, above, says why: a deadline error (`forward_backward ... timed out`) is a training service that stalled, a connection error one that died |
| `The training service refused the training client (attempt 3, 25s after the first): ...` | The service did not accept this start yet: it is restarting, or it still holds the session of a start that died, which a service that trains one model at a time does until the session expires. The start asks again, for up to 20 minutes. A refusal that lasts that long ends the start: check that the service runs, and that no other client trains on it |
| `Training exited with code 1, and 3 starts in a row saved no new checkpoint: not restarting` | Every start failed before it saved a checkpoint at a later step, as a failure that each start meets would. Fix what the starts' logs show, then run `platoon train` again, which resumes. Set `checkpoint.recovery_every`, so that a start that makes progress keeps it |
| `Training exited with code 1 after 20 restart(s), which is watchdog.max_restarts: not restarting` | The run failed `max_restarts` times. Look at why in the starts' logs, then run `platoon train` again, which resumes |
| `<run dir> is in use: platoon train (pid 1234) still runs it on this host; ...` (exit 78) | Another `platoon train` or `platoon eval` runs this directory, and two would write over each other's records and reap each other's sandboxes. Wait for it to exit, or give this run another `stats.trial_name` (an evaluation another `inference.output_dir`). A process that is gone, or one on another host, does not count |
| `Reaped 4 sandbox(es) labelled platoon.run=<run dir> before restart 1 (singularity: 4)` | The sandboxes a dead start, or a killed rollout worker, left running were stopped and removed. At exit a count above 0 means rollouts that were killed (stragglers, hard timeouts) |
| `Cannot reap the sandboxes labelled {...} (<provider>)` | The provider that the run's config names could not reap here, such as `docker` on a host without Docker; the warning's traceback says why. Its sandboxes, if any, are left running |
| `platoon: not reaping the sandboxes of <run dir>: platoon train (pid 1234) still runs it on this host` (exit 1) | `platoon sandbox reap` is for a run whose process was killed: a running one reaps its own when it exits |
| `platoon: not reaping the sandboxes of <run dir>: its last process ... ran on host <host>, ...` (exit 1) | The run's sandboxes are on that host, and only there can it be told whether the run still runs: run `platoon sandbox reap` there |
| `python -m platoon.train.tinker.restart_wrapper (run_with_restart): deprecated; use watchdog.max_restarts ...` | Set `watchdog.max_restarts` in the training config and run `platoon train` alone. Platoon 0.3.0 removes the wrapper |

## A mixture's tasks repeat, or never train

`platoon train` and `platoon config validate --tasks` warn when a mixture's weights draw its tasks unevenly
([Train on several environments](train-and-eval.md#train-on-several-environments)):

| Message | Fix |
|---|---|
| `environments[<name>]'s <n> training tasks are each drawn about <ratio> times as often as environments[<other>]'s <m> ...` | The weights give a small entry as large a share as a big one: set `weight: tasks` on every entry to draw every task equally often, or weights in proportion to the tasks you want each entry to stand for |
| `environments[<name>]: its weight (...) gives it about <k> of the run's <slots> task slots (...), fewer than its <n> training tasks, so about <n - k> of them never train; ...` | The entry's share of the run is smaller than its tasks: set `weight: tasks` on every entry, raise the entry's weight, or train longer |

## The training service

| Symptom | Cause and fix |
|---|---|
| `Tinker sample_async timed out after 600.0s` | One sampling call took longer than the sampling timeout: `train.sample_timeout_seconds` (or `inference.sample_timeout_seconds`), else `TINKER_SAMPLE_TIMEOUT_SECONDS`, else 600 s. The `Sampling timeout:` line the run logged at its start says which. Check the service's load, or raise the key, which wins over the variable ([The sampling timeout](train-and-eval.md#the-sampling-timeout)) |
| `The sample timeout (TINKER_SAMPLE_TIMEOUT_SECONDS) must be finite and positive, ...` (or `must be a number of seconds`) | The config sets no `sample_timeout_seconds`, so the variable bounds each sampling call, and it holds something other than a positive number of seconds: fix or unset it, or set the key |
| `forward_backward ... timed out` or `optim_step ... timed out` | The service did not finish a step in `train.forward_backward_timeout_seconds` (5400) or `train.optim_step_timeout_seconds` (300) |
| The trainer exits with code 2 | The watchdog saw no progress for `watchdog.timeout_seconds` (600). Rerun to resume, or set `watchdog.max_restarts`, and `platoon train` starts training again by itself ([Long runs](train-and-eval.md#long-runs)) |
| `Renderer '...' does not support renderer_kwargs['...']` | The key is not an attribute of that renderer; see [Renderers](train-and-eval.md#renderers) |
| `The api_key client option must be set ...` | Set `TINKER_API_KEY`. The `platoon` commands check it before they start ([exit code 78](#the-command-exits-with-code-78)); code that opens a Tinker client itself meets this error instead |
| `Could not delete recovery checkpoint tinker://...: ...` | A delete that `checkpoint.keep_recovery` asked for failed; the next recovery save tries it again. A 404 (the service does not have the checkpoint) is believed once the service has deleted a checkpoint in this start, and the checkpoint is not asked for again. Before that it may only mean that the service has no delete route, which answers 404 too: the checkpoint is asked for again, and for the state of a recovery checkpoint the running start saved, which nothing but a delete removes, the 404 counts as failed |
| `Recovery checkpoint tinker://.../weights/... is not on the service (...)` | The service answered 404 for a recovery checkpoint's state after deleting others: someone deleted it, or it was saved under another API key, which the service answers alike. If it still exists, delete it yourself, and keep `TINKER_API_KEY` the same for every start of a run |
| `checkpoint.keep_recovery: stopped deleting recovery checkpoints for the rest of this start: 3 deletes failed in a row, ...` | Training goes on, but its recovery checkpoints now pile up. Fix what the message names (the service's disk, its permissions, a different API key) and restart the run, which deletes what is left; or delete the old `recovery-*` checkpoints yourself: the run directory's `checkpoints.jsonl` lists them, and `checkpoints_deleted.jsonl` what was deleted |
| `... stopped deleting recovery checkpoints ...: the Tinker service cannot delete checkpoints: ...` | The service, or the client, has no checkpoint deletion: it answered 405 or 501 to `DELETE /api/v1/training_runs/<id>/checkpoints/<checkpoint>`, or the client has no delete method. (A service without the route at all answers 404, and pruning stops after three of those for this start's own states.) Delete old recovery checkpoints on the service yourself, or leave `checkpoint.keep_recovery` unset |
| `Stopped deleting recovery checkpoints after waiting 30 s at the end of the run` | A delete was still running when training ended; the run's next start finishes it |

## Still stuck

Open the trajectory in the viewer (`platoon viz show-dump .../trajectory_collection.json`) to see each step the
agent took. For how the pieces fit together, read [the architecture](../architecture.md). If platoon is missing
something you need, the [RFC index](../rfcs/README.md) lists the proposals in progress and explains how to propose a
change.
