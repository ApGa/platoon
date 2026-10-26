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
| `... is not one of 'auto', 'weighted', 'uniform'` | Use one of the listed values |
| `train.task_sampler: 'random' is not one of 'auto', 'weighted', 'uniform', 'pass_rate', a name an environment's package registers, or module:Class` (or `environments.<i>.task_sampler: ... 'uniform', 'pass_rate', ...`) | No entry names a `package` that could register the name: fix the spelling, name a sampler of your own as `module:Class`, or add the entry's `package` that registers it ([Write a task sampler](../extending/task-samplers.md)) |
| `environments[<name>].task_sampler: unknown task sampler '<name>'; did you mean ...?` | The packages the entries name register no sampler of that name: fix the spelling, or the `package` ([Register it, or name it](../extending/task-samplers.md#register-it-or-name-it)) |
| `environments.<i>.task_sampler: weighted shares a run's slots among its entries; ...` (or `auto`) | An entry's sampler chooses among the entry's own tasks: use `uniform`, `pass_rate` or a sampler of your own there, and `weighted` as `train.task_sampler` |
| `train.task_sampler_options: weighted takes no options; leave task_sampler_options out` (or `environments.<i>.task_sampler_options: uniform takes no options; ...`) | Only a sampler with options, such as `pass_rate`, takes `task_sampler_options`: drop them, or set the sampler they belong to |
| `environments.<i>.task_sampler_options.<key>: unknown key; did you mean ...?`, or `... explore: must be between 0 and 1, got 1.5` | `pass_rate` takes `explore`, `max_draws_per_epoch`, `max_outstanding_per_task`, `candidates` and `seed` ([Choose tasks by their pass rates](train-and-eval.md#choose-tasks-by-their-pass-rates-experimental)) |
| `environments[<name>].task_sampler: pass_rate: train.task_sampler: uniform draws every training task of the run itself, with no entry slots, so it asks no entry's sampler; ...` (or `pass_rate`, or `<module:Class> hands no slots to entries`) | The run's sampler hands no slots, so no entry's sampler is asked: drop the entry's `task_sampler`, or set `train.task_sampler: weighted` ([Train on several environments](train-and-eval.md#train-on-several-environments)) |
| `environments[<name>].weight: 3: train.task_sampler: pass_rate chooses among every training task of the run by its statistics, ...` | A run-level `pass_rate` takes no weights: drop them, or keep `weighted` and set `task_sampler: pass_rate` on the entries, which keeps each entry's share |
| `environments[<name>].task_sampler: pass_rate chooses the tasks whose next group is likely to mix passes and failures, which a group of one never does ...` | A group of one rollout is all pass or all fail: use `uniform` for the entry, or a `group_size` of 2 or more |
| `...task_sampler: <module:Class> does not import (...)`, `... is not a task sampler: ...`, `... hands slots to entries, which only the run's sampler does`, or `... hands slots to entries and is adaptive; ...` | The class does not import, is not a sampler (`next`, `state_dict`, `load_state_dict`, and `adaptive` a bool), or is used at a level it cannot serve ([Write a task sampler](../extending/task-samplers.md)) |
| `environments[<name>].task_sampler: <name>: <message>` | The sampler's own `check` refused the config, for the reason it gives |
| `environments[<name>]: task ids repeat (t3 2 times, ...): pass_rate draws by task id, ...` (a warning) | The entry's listing repeats task ids: a sampler that draws by task id runs only the first row of each. Give each task one id, or keep the entry on `uniform`, which draws row by row |
| `Task sampler <name> raised at decision <n> (...)` or `task sampler <name>: decision <n> answered ..., which it may not` (logged once, as an error) | The sampler failed a decision, and the run drew a fallback task in its place, as it does for every failed decision of that sampler (`train/sampler/.../draws/fallback` counts them). Fix the sampler; `check_task_sampler` finds most such bugs before a run ([Test it](../extending/task-samplers.md#test-it)). Core's samplers fail only where an order handed to a trainer built in code fails, and that stops the run, as in platoon 0.3.0 |
| `Task sampler <name>: the run's per-task statistics are not available (...)` (a warning) | The statistics stopped (an earlier warning says why) or never started, so `pass_rate` draws by exploration alone, the least recently drawn task first, for the rest of the start |
| `The state that task sampler <name> saved does not load (...): it starts afresh` (a warning) | The sampler's saved state does not fit it (its class changed what it saves, say): it starts afresh on the statistics the run has, and the run goes on |
| `Resuming from checkpoint <name>: cannot read the outstanding draws' rows: ...` or `... cannot read the tokens: ...` (a warning) | The checkpoint's `tasks/dispatch-*.json` holds a part that does not read, though its SHA-256 matches (another tool wrote it): the resume takes up the rest, each draw drawn again from the first row of its task, or without the slots that waited for an entry's sampler |
| `Task sampler <name>: its metric '<key>' is one the dispatcher writes itself under the sampler's prefix ...` (logged once, as an error) | The sampler's `metrics()` named a key the dispatcher logs itself (`draws`, `draws/<reason>`, `decision_ms`, ...), which is left out so that the dispatcher's own count stands: rename the metric ([Metrics](../extending/task-samplers.md#metrics)) |
| `The run's sampling inputs changed since checkpoint <name>: ... environments[<name>].task_sampler: uniform -> pass_rate ...` (a warning) | A resume under another sampler, or other options: the run goes on with them and records the change in `lock.json` (`sampling_changed`); a new sampler starts afresh ([Checkpoints and resume](train-and-eval.md#checkpoints-and-resume)) |
| `train.group_admission: 'informativ' is not one of 'all', 'informative', a name an environment's package registers, or module:Class` (or `environments.<i>.group_admission: ...`) | No entry names a `package` that could register the name: fix the spelling, name a rule of your own as `module:Class`, or add the entry's `package` that registers it ([Write a group admission rule](../extending/group-admission.md)) |
| `environments[<name>].group_admission: unknown group admission rule '<name>'; did you mean ...?` | The packages the entries name register no rule of that name: fix the spelling, or the `package` ([Register it, or name it](../extending/group-admission.md#register-it-or-name-it)) |
| `train.group_admission_options: all takes no options; leave group_admission_options out` (or an entry's) | `all` takes no options: drop them, or set the rule they belong to |
| `...group_admission_options.<key>: unknown key; did you mean band?`, or `...band: must be two numbers lo < hi between 0 and 1, got [0.8, 0.2]` | `informative` takes `band: [lo, hi]` alone ([Train only on groups that carry signal](train-and-eval.md#train-only-on-groups-that-carry-signal-experimental)) |
| `environments.<i>.group_admission_options: an entry's options go with the rule it sets: ...` | An entry's options configure the rule the entry sets: set its `group_admission` beside them, or drop them to inherit the run's rule with the run's options |
| `train.refill.max_groups_per_step: must be a finite number of at least 1 (...), got 0.5`, or `train.refill.max_groups_per_step: null is not allowed (expected float)` | The cap is a number of at least 1, a multiple of `train.batch_size`; there is no null (no cap), since a pool the policy has solved would never close a step: set a large number instead |
| `train.num_minibatches: 2 takes 2 optimizer steps a batch, and a group admission rule (...) makes short steps: ...` | Beside a rule other than `all`, a short step's last minibatch would take a full optimizer step on a few groups: set `num_minibatches: 1`; `num_microbatches` may stay |
| `train.refill.when_short: unknown key; not in this release: ...`, `train.on_stale: unknown key; not in this release: ...` | The design's keys this release leaves out: a short step trains the groups it admitted, and a stale group is dropped (under a rule the step takes another in its place) |
| `...group_admission: informative rejects the groups whose workflow's verdict says they train nothing; the run's workflow, <ref>, gives no verdicts ...` | The run's workflow class is no `GroupRolloutWorkflow` and does not declare `gives_verdicts = True`, so `informative` would admit every group: use `all`, a rule of your own, or a workflow that gives verdicts |
| `...group_admission: <module:Class> does not import (...)`, `... is not a group admission rule: a class with admit(group, stats)`, or `environments[<name>].group_admission: <name>: <message>` | The class does not import, has no `admit` (or a coroutine one), or its own `check` refused the config, for the reason it gives |
| `train.loss_fn: <loss> with loss_fn_config keys ... may train terms that do not depend on the advantage ...` (a warning) | Under that loss a uniform group's datums train its advantage-independent terms (a KL or an entropy term), and the rule rejects such groups, which removes their share of those terms: intended, or use the default `cispo` thresholds |
| `Group admission rule <name> failed on the group of task <id> (...): the group is admitted, and training goes on; ...` (logged once, as an error) | The rule raised or answered outside the contract (anything but an `Admission` or a bool, or a reason that is not a non-empty string of at most 64 characters): the group was admitted, as every such decision is, and `train/admission/errors` counts them. Fix the rule; `check_group_admission` finds most such bugs before a run |
| `Group admission rule informative admits the groups of environments[<name>] whose output carries no verdict (reason unknown): ...` (logged once) | The entry's workflow returns plain lists of datums, or a verdict platoon does not know, so `informative` cannot tell whether a group trains: each counts toward the batch, under `train/admission/admitted/unknown` |
| `Batch <n> trains <a> admitted groups, fewer than train.batch_size (...): it counted <C> groups, its cap (...), and of the groups it dequeued, <r> were rejected, <s> stale and <f> failed; ...` (logged once) | The step closed at its cap. Where rejections dominate, most groups were rejected at this pass rate: raise `train.refill.max_groups_per_step` (about `1.3 / p` fills 95% of steps), or choose tasks with `task_sampler: pass_rate`. Where failed groups do (the message then says `Most of them failed`), fix what makes them fail: the rollout errors in the log, `train/admission/failed` |
| `Batch <n> admitted no group: of the groups it dequeued, <r> were rejected, <s> stale and <f> failed, ...` (logged once) | No group of the step was admitted: the policy may have solved the pool (rejections), or the environment fails every group (failed ones, and the message says so). The step took no update and the run goes on; look at `train/admission/rejected/<reason>` and `failed` |
| `Batch <n> has counted <c> groups and admitted <a> of train.batch_size (...); it waits for more, ...` | A step under a large cap keeps waiting for admitted groups, its metrics logged only when it closes: a pool the policy has solved never fills a step. Lower the cap, or add harder tasks |
| `The run's sampling inputs changed since checkpoint <name>: ... environments[<name>].group_admission: all -> informative ...` (a warning) | A resume that turned a rule on or off, changed one or its options, or changed `train.refill.max_groups_per_step`: the run goes on with it, the groups the checkpoint kept train as they are, and `lock.json` records the change (`sampling_changed`) |
| `entry and adaptive samplers need AutoDataset.training's sampling ...` | A trainer built in code without `train_sampling` draws its dataloader's order, which runs no other sampler: pass `train_sampling=AutoDataset.training(config).sampling`, as `platoon train` does |
| `environments has 1 entries; index 1 is out of range` | List indices count from 0; negative ones from the end |
| ``a harness needs an environment `service` to open sessions with`` | Add `service:` to the entry |
| `environments.<i>.service_kwargs.<option>: <path> names <file's directory>/<path>, which does not exist: a relative path in a config file is relative to the file's directory, ...` (or `environments.<i>.harness:` for a `.toml` spec) | Relative input paths resolve against the config file that names them, not the working directory. Write the path relative to the file, or absolute; when the message adds that the path exists relative to the working directory, that is the file it named before ([Write and override configs](configs.md#2-build-on-it-with-extends)) |
| `$PLATOON_SITE names <path>, which is not a file: ...` | `PLATOON_SITE` must name a site profile: fix the path, or unset it to use `~/.config/platoon/site.yaml` when that exists ([Write the site's profile](clusters.md#4-write-the-sites-profile-once)) |
| `invalid config <site.yaml>:` (from `config validate`, `<site.yaml>:` before each line) then `<key>: unknown key; a site profile takes cache, runs, provider and provider_options: ...`, `provider: process runs the agents' commands on this host, without isolation, ...`, `provider_options.<option>: a site profile sets only the options that do not weaken a sandbox's isolation, ...` or `<key>: $VAR is not set` | The site profile says where files live and which isolating runtime starts sandboxes, nothing else. Move another key, the `process` provider, or an option outside its list (`binary`, `extra_start_args`, binds) into the run config that needs it, and export a variable a value names |
| `environments.<i> (<name>): the site profile <path> chooses the sandbox provider <provider>, which this node cannot run: ...` | The profile's runtime is missing on this node (no executable on `PATH`, or no `hpc` extra). Run on a node that has it, or point `PLATOON_SITE` at the profile of the site you are on |
| `environments.<i> (<name>): its sandbox provider, process, runs the agents' commands on this host without isolation, and it comes from the named environment the entry selects (env:), not the run config: ...` | `platoon train`, `eval` and `rollout` run agents on the host only when the run config itself says so: set `environments.<i>.service_kwargs.provider: process` in it, or choose an isolating provider |
| `$PLATOON_HARBOR_CACHE: platoon 0.3.0 removed it; use the site profile's cache, whose harbor directory it named ...` (or `$PLATOON_TOOLCHAINS`, its `toolchains` directory) | Unset the variable, and point the site profile's `cache` at the directory that holds `harbor` and `toolchains` ([Write the site's profile](clusters.md#4-write-the-sites-profile-once)) |
| `environments.<i>.service_kwargs.cache: unknown key; platoon 0.3.0 removed it; use the site profile's cache, ...` (or `image_map`, `toolchain_root`) | Drop the option: the Harbor cache, a fetched dataset's image map (`images/<provider>.json` in it) and the harness toolchains are in the site profile's `cache` ([Prepare a config](sandboxes-and-harbor.md#prepare-a-config-once-per-site)) |
| `log_path: required key is missing; set it, or runs in the site profile ...` | Give the config a `log_path`, or the site profile a `runs`, under which a config without one gets its run directory |
| `<run directory> holds the training run of <other config>: the site profile names a training run's directory after its config file's name, and <config> has the same name, ...` | Two training config files of one name, in different projects, would share the run directory the profile names after it, and one would take over the other's run. Name this config's experiment (`stats.experiment_name`) or give it a `log_path` ([Write the site's profile](clusters.md#4-write-the-sites-profile-once)) |
| ``Every entry of a multi-environment `environments` list needs a `name` `` | Name each entry of a mixture; names must be unique and must not contain `::` |
| `Mixture entries [...] weigh by their tasks (weight: tasks, the weight of an entry that sets none) and [...] by numbers: ...` | A mixture's weights are all numbers, which share its slots in proportion, or all `tasks`, the default, which draws every task equally often: give each entry a number (`1` each for equal shares), or no entry one ([Train on several environments](train-and-eval.md#train-on-several-environments)) |
| `... roll out with harness_rollout, whose harnesses sample the policy, but inference.renderer_name is not set` | Set `inference.renderer_name` in the evaluation config |
| `ValueError: checkpoint.every must be a number of training steps, or 0 for never, got -1` (or `... of passes over the training tasks (strategy: epoch) ...`) | Set `checkpoint.every` to the training steps between permanent checkpoints (epochs with `strategy: epoch`), or to 0 for none; `eval.every` takes the same values |
| `eval.strategy: 'none' is not one of 'step', 'epoch'; platoon 0.3.0 removed it; use eval.every: 0, which never evaluates` (or `checkpoint.strategy`) | Set `every: 0`, which turns the event off, and drop `strategy` ([Cadences in steps or epochs](train-and-eval.md#cadences-in-steps-or-epochs)) |
| `checkpoint.keep_recovery: 1 × checkpoint.recovery_every: 3 = 3 must exceed train.max_staleness: 3: ...` (or `... must exceed environments[<name>].max_staleness: 48, the run's largest staleness bound ...`), or `checkpoint.keep_recovery needs train.max_staleness: ...` | Recovery checkpoints within `max_staleness` policy versions may still serve rollouts and are never deleted, so `keep_recovery` must reach past them, and past an entry's own bound: raise `keep_recovery` or `recovery_every` as the message says, or lower the bound it names; set `train.max_staleness` when it is missing ([Checkpoints and resume](train-and-eval.md#checkpoints-and-resume)) |
| `ValueError: eval.workflow_config: min_successful_group_size must be in [1, group_size] (the keys it leaves out keep their eval.workflow_config defaults: group_size: 1, ...)` | A partial section keeps its field's defaults, here one rollout per task: set `group_size` in it too ([Write and override configs](configs.md#2-build-on-it-with-extends)). A straggler quorum above 1 fails the same way. The defaults it lists include the rollout and worker settings it keeps from `train.workflow_config` ([Evaluate during training](train-and-eval.md#evaluate-during-training)) |
| `train: set num_epochs or max_training_steps (with both, training runs the shorter)` | Set one of them |
| `environments.<i>.workflow_config.<key>: unknown key; an entry does not set it: ...` (also under `eval_overrides.workflow_config`) | The key acts on microbatches, which mix entries' groups, or on the run's one pool of rollout workers: set it in `train.workflow_config`, or `eval.workflow_config`, as the message says. An entry sets the keys that act on one group or one rollout ([Settings per environment](train-and-eval.md#settings-per-environment)) |
| `environments[<name>].workflow_config: min_successful_group_size must be in [1, group_size] (the keys it leaves out keep train.workflow_config's values: group_size: 8, ...)` | An entry's settings are laid over the run's: the keys it leaves out keep the run's values, here its group size. Set `group_size` in the entry too. The same holds for a straggler quorum |
| `...: success_threshold must be a finite number, got inf` | Set the lowest scored reward that counts as a pass to a number, such as `0.5` for an environment that gives partial credit ([Per-task statistics](train-and-eval.md#per-task-statistics)) |
| `environments[<name>].max_staleness: 2 is below train.max_staleness: 3, ...` | An entry's bound may only raise the run's, which also bounds how far every entry's rollouts run ahead of training: set `train.max_staleness` to the smallest bound and raise the slower entries' own ([Settings per environment](train-and-eval.md#settings-per-environment)) |
| `train.max_concurrent_rollouts: 12 holds no group of environments[<name>], whose 16 rollouts start together ...` | A group reserves its whole size of permits, and its rollouts start together under `train.permit_release: group` and in evaluation, so the budget must hold the largest entry's group: raise `max_concurrent_rollouts` (`eval.max_concurrent_rollouts` for evaluation) |
| `train.max_concurrent_rollouts: 12 rollouts, of which the groups of the labels with a limit (train.rollout_group_limits) hold up to 8 (slow: 2 × 4): ...` (a warning) | While the capped labels run as many groups as their limits allow, their groups hold so many rollouts that a group of a larger entry waits for one of them to end, and every group after it with it, so the capped labels set the run's pace: raise `max_concurrent_rollouts` to the number the message gives ([Settings per environment](train-and-eval.md#settings-per-environment)) |
| `Label '<label>' has 80 rows waiting for its limit (train.rollout_group_limits: 1 at once): its new rows are skipped, ...` (a warning) | The label's share of the order asks for more groups than its limit runs at once (at `max_staleness: 0`, more than its limit each step), so it trains less than its weight: lower its entry's weight or raise its limit. `train/batch/<column>/<label>/parked_rows` and `skipped_rows` show how many of its draws wait and how many were skipped ([Settings per environment](train-and-eval.md#settings-per-environment)) |
| `environments[<name>].weight: 3: train.task_sampler: uniform draws every training task of every entry equally often, ...` | `uniform` gives each entry its share of the tasks and reads no weight: leave the entries' weights out (or `weight: tasks`), or keep the weights with `train.task_sampler: weighted` ([Train on several environments](train-and-eval.md#train-on-several-environments)) |
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
| `<key>: unknown key; platoon <version> removed it; use <replacement>` | That release removed the key, after a minor release of deprecation warnings: set what the message names instead. The [changelog](../../CHANGELOG.md) lists each removed form with what replaces it |
| `<file>: a fetch config file; platoon 0.3.0 removed it; use a training or evaluation config, whose Harbor entry names the dataset and its fetch settings ...` (or `platoon harbor fetch --lock, which applied to fetch config files alone: ...`, or `--cache`) | Move the file's keys into a Harbor entry's `service_kwargs` in the run's config, `--lock`'s path into its `lock`, and run `platoon harbor` on that config; the site profile's `cache` replaces `--cache` ([Fetch a dataset by name](sandboxes-and-harbor.md#fetch-a-dataset-by-name)) |
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
| `Skipping Harbor task ...: ... multi-container stack (sidecar services), and the singularity provider runs no multi-container stacks` | The task's compose file names services besides `main`, and the environment's provider runs no stacks (or the environment sets `stacks: refuse`): platoon refuses the task rather than score it without them. `docker` and `process` run stacks. See [Multi-container tasks](sandboxes-and-harbor.md#multi-container-tasks) |
| `... and the docker CLI (...) has no compose plugin: `docker compose version` fails (...)` | The `docker` provider runs stacks as `docker compose` projects, and this host's CLI has no compose plugin: install it (the `docker-compose-plugin` package), or leave the stacks out ([Stacks on Docker](sandboxes-and-harbor.md#stacks-on-docker)) |
| `all predefined address pools have been fully subnetted`, or `could not find an available, non-overlapping IPv4 address pool among the defaults to assign to the network`, in a `sidecar_start` or `sandbox_start` error | Each stack on Docker takes a network, and one more while it has egress, and Docker's default address pools hold about 30, so about 15 stacks with egress at once: widen them (`default-address-pools` in Docker's `daemon.json`), or run fewer stacks at once ([Stacks on Docker](sandboxes-and-harbor.md#stacks-on-docker)) |
| `... is not a directory of this user's, so no stack's compose project may go there` | The `docker` provider's `stack_root` (by default `platoon-docker-stacks-<uid>` in node scratch) is a symlink, or another user's directory: on a node's shared scratch, whoever made it could change a stack's project file. Remove it, or set `stack_root` to a directory of your own ([Stacks on Docker](sandboxes-and-harbor.md#stacks-on-docker)) |
| `... has a ':', which Docker reads as the end of a bind's source, so no stack's compose project may go there` | The `docker` provider's `stack_root` has a `:` in its path, which would cut short the source of every bind below it (a stack's volumes). Set `stack_root` to a path without one ([Stacks on Docker](sandboxes-and-harbor.md#stacks-on-docker)) |
| `The Docker daemon is ...: before 25.0.5, an internal network passes its containers' DNS queries on to the host's resolvers (CVE-2024-29018)` | A warning: a stack's services could send data out through DNS in a phase without egress. Upgrade Docker; a distribution's build of an older version may carry the fix ([Stacks on Docker](sandboxes-and-harbor.md#stacks-on-docker)) |
| A lock's `refused` is `stack: the compose file adds sidecar services (refuse: {sidecars: false} fetches it as a stack)` | The environment's fetch settings set `refuse: {sidecars: true}`, which refuses stacks. Drop it to lock them as runnable, which a fetch does by default since 0.5.0, where the provider runs stacks ([Multi-container tasks](sandboxes-and-harbor.md#multi-container-tasks)) |
| `N of the M tasks of split ... are multi-container stacks (sidecar services: ...), and ...; run them on a provider that runs stacks, or leave them out (...)` | A fetch locks stacks as runnable, and the environment's provider runs none here (the reason follows `and`), or the environment sets `stacks: refuse`. Run them on a provider that runs stacks, leave them out (`exclude_tasks`), or set `refuse: {sidecars: true}` in the fetch settings, which locks them refused, as a fetch did by default before 0.5.0 ([Multi-container tasks](sandboxes-and-harbor.md#multi-container-tasks)) |
| `The Harbor lock ... does not exist (the lock ..., which platoon wrote for these settings before refuse.sidecars defaulted to false, refused N task(s) as stacks, ...)` | The dataset's lock from before 0.5.0 refused its stacks, which a fetch now locks as runnable, so it no longer answers an entry that sets no `refuse.sidecars`. Run `platoon prepare` on the run's config (or start `platoon train` or `platoon eval`, which fetch a pinned dataset) to lock them, or set `refuse: {sidecars: true}` to keep the old lock ([Multi-container tasks](sandboxes-and-harbor.md#multi-container-tasks)) |
| `... multi-container stack ..., and the singularity provider runs a multi-container stack as an hpc-sandbox pod, and the installed hpc-sandbox (0.4.0) has no pod services that run as your uid or with your subordinate ids, host aliases or ports exposed to the host: install hpc-sandbox 0.5` | Stacks run as pods of hpc-sandbox 0.5. Install the `hpc` extra's, 0.5.1, in the run's environment ([The hpc extra](../getting-started/install.md#the-hpc-extra)): `pip install -U 'platoon[hpc]'`, or in a checkout `uv sync` with `--extra hpc` among your extras; until then, leave the stacks out (`exclude_tasks`) |
| `... this node does not let an unprivileged user nest user namespaces, which pods need (...)` | The node's kernel or its security policy keeps users from making user namespaces, or from making one inside another, as each pod and service does. Run the stacks on nodes that allow it ([Run on a cluster](clusters.md#5-choose-a-container-runtime-for-sandboxes)), or leave them out |
| `N of the M tasks of split ... have sidecars that run registry images the image map ... has not pinned by digest with their configuration` | The run needs what `platoon prepare` makes: run it on the run's config, which pins each image (or `platoon harbor images pin CONFIG`) |
| `The singularity provider runs the sidecars of stacks as your own uid, not as root with your subordinate ids, ...` | Your user has no subordinate ids on this node, or `newuidmap` and `newgidmap` are missing: sidecars run as `caller`, and an image that insists on root fails its start (`sidecar_start`). `hpc-sandbox doctor` says which |
| An `error` outcome, `sidecar_start` | A sidecar of the task's stack did not start: its command failed or exited, its healthcheck never passed (or no longer did by the start's last probe), or an init job did not complete within `sandbox_start_budget_s`. The reason names the service and ends with its log; the rollout is never scored ([Multi-container tasks](sandboxes-and-harbor.md#multi-container-tasks)) |
| `N of the M tasks of split ... are multi-container stacks that no rollout can start (...)` | A stack cannot start anywhere: a sidecar of a registry image that its compose file gives no command, where no image map pins the image (the `process` provider's), or a `service_healthy` dependency on a service without a healthcheck. The message names the first; give the sidecar a command with an overlay's `[[compose]]`, or leave the tasks out (`exclude_tasks`). A sidecar whose build context's image the image map lacks stops the run as not prepared instead: `platoon prepare` builds it |
| `N of the M tasks of split ... are stacks whose services ask for privileges (...)` | The dataset was fetched with `refuse: {privileges: false}`, which keeps compose services that ask for `privileged` or for a capability such as `NET_ADMIN`, and the environment's provider gives none: run them on `docker`, or leave them out ([Stacks on Docker](sandboxes-and-harbor.md#stacks-on-docker)) |
| An `invalid` outcome, `sidecar_unhealthy` | A sidecar was down when the tests ran, before or after them, or while a separate verifier's artifacts were collected (`metadata`: `sidecar`, `sidecar_status`, `sidecar_log`): the attempt is not trained. An agent may have stopped it (a database's `SHUTDOWN`); where the policy should own that, set `on_sidecar_down: grade` |
| A lock's `refused` is `compose: <service>.<key>: ...`, or a task is skipped with `environment/docker-compose.yaml: <service>.<key>: ...` | The compose file has a key or value outside the subset platoon runs, such as `privileged` or a bind of a host path. A dataset's overlay may drop or replace it (`[[compose]]`), or leave the task out ([Compose files](sandboxes-and-harbor.md#compose-files)); for `privileged` and capabilities such as `NET_ADMIN` or `DAC_READ_SEARCH`, a fetch with `refuse: {privileges: false}` keeps the task for the `docker` provider ([Stacks on Docker](sandboxes-and-harbor.md#stacks-on-docker)) |
| `... gives main cap_add, shm_size, ..., which platoon does not apply to the one sandbox of a main-only task yet` | The task's compose file names only `main`, with settings that one sandbox cannot apply yet. An overlay may drop them when the task does not need them ([Compose files](sandboxes-and-harbor.md#compose-files)) |
| An `error` outcome, `main_exited` | The command that the task's compose file gives `main` (its init) exited before grading, which in Docker would have ended the container: the attempt is never scored. Its output is in `/tmp/platoon-init/init.log` in the sandbox; an init that exits by itself is a bug of the task, one an agent killed shows in its trajectory. In a stack, `main` itself may be gone by grading (its container exited); where the whole stack is (a pod whose holder was killed), the kind is `sidecar_probe` |
| An `error` outcome, `env_config`, whose reason is a variable's name (`'OPENAI_API_KEY'`) | The variable is unset where the session opens, and the task needs it: a template of the task's `[environment].env` names it without a default (`${NAME}`), or its compose file names it without one (`${NAME}`, `${NAME:?...}`) and an overlay lists it (`compose_host_env`). Set it in the environment the rollouts run in, or give it a default (`${NAME:-}`; in a compose file, with an overlay's `[[compose]]`) ([Compose files](sandboxes-and-harbor.md#compose-files)) |
| `WARNING ... hold environment/compose.yaml, which Harbor does not read` | Harbor reads only `environment/docker-compose.yaml`: the tasks run as one container without that file, as in Harbor. If the dataset meant it, an overlay's `[[copy]]` can add it as `environment/docker-compose.yaml` ([Compose files](sandboxes-and-harbor.md#compose-files)) |
| `No runnable Harbor tasks in ...` | No subdirectory has a `task.toml`, or every task was refused (no `instruction.md` or `tests/test.sh`, a compose file, ...); the message lists the first reasons |
| `Two Harbor tasks are named '...'` | Two directories in `task_dirs` have the same name, and a task's name is its id; this counts tasks platoon refuses too. Rename one, or list only one ([Lay out the tasks](sandboxes-and-harbor.md#1-lay-out-the-tasks)) |
| `Harbor task '...' has no environment.docker_image and no default image` | Set `default_image` in `service_kwargs`; platoon does not build `environment/Dockerfile` |
| `... [agent].user = '...', and the environment refuses tasks that name users` | Set `agent_user: honor` in `service_kwargs` (since 0.5.0) to run each phase as its user, in sandboxes that give root and other users, or `ignore` to run the task as the sandbox's own user ([Users in task sandboxes](sandboxes-and-harbor.md#users-in-task-sandboxes)) |
| `N of the M tasks of split '...' cannot run as their [agent].user and [verifier].user ask (...)` | With `honor`: the message names a task and why. A sandbox of it must give root and other users and its provider's options cannot: on `singularity`, set `fakeroot: true` (and the copy layout) in `provider_options`, or in `verifier_provider_options` for a separate verifier; `enroot` and `process` cannot. Or it is a multi-container stack, or a root verifier graded in its agents' own sandbox whose agents are not honored: set `agent_user: honor`, or `ignore`, or leave it out with `exclude_tasks` |
| `N of the M tasks of split '...' honor users in sandboxes that lose the owners of their images' files (...)` | On `singularity`, a sandbox that runs a user other than root, or anything as its image's `USER` (the agents' sandbox under `agent_user: honor`), needs the image map's record of its image: its owner manifest and `USER`. Run `platoon prepare` on the config, which records those of the tasks' `docker_image`s; build again an image that platoon before 0.5, the `def` builder or the `oci` builder with a squashfs-tools before 4.4 built, with the command the message names, where `unsquashfs` is 4.4 or later; serve a dataset directory as a `dataset` to fetch; give a task whose image is a local file a `docker_image` or a Dockerfile. With an hpc-sandbox before 0.5, which applies no manifest, install the `hpc` extra's, 0.5.1 ([Users in task sandboxes](sandboxes-and-harbor.md#users-in-task-sandboxes)) |
| An `error` outcome, `env_config`, whose reason says `the image's /etc/passwd has no user '...'` or `the image has no way to switch to the user ...` | The task names a user that its image lacks, or the image has neither util-linux's `setpriv`, nor `su` for that user, nor Python 3 to switch to it with (a harness toolchain's interpreter serves too). Fix the task with an overlay, or leave it out |
| `... the installed hpc-sandbox takes no fakeroot option: it cannot delete what the other users of a --fakeroot sandbox write ...` | Root verifiers on `singularity` need hpc-sandbox 0.4 or later, whose singularity provider takes `fakeroot`: it gives back what the tests leave as other users before it deletes a sandbox, where an older one, such as 0.3.0, would leave each such sandbox's files on the node. Install the `hpc` extra's hpc-sandbox, 0.5.1 ([The hpc extra](../getting-started/install.md#the-hpc-extra)): `pip install -U 'platoon[hpc]'`, or in a checkout `uv sync` with `--extra hpc` among your extras ([Verifiers that run as root](sandboxes-and-harbor.md#verifiers-that-run-as-root)) |
| `N of the M tasks of split '...' honor users (...) on N image(s) that the oci builder of platoon before 0.3.0 built ...`, or `platoon prepare`'s images step fails saying so | Those images' receipts record no `builder_revision`: on a host whose umask is 027, their files keep out the users other than root. Build them again with the command the message names (`platoon harbor images build CONFIG --key KEY ...`; without `--key`, it builds only the keys the map lacks) |
| `WARNING ... honor runs none of the N task(s) of ... that name users to run as (...)` | The fetch's `refuse.agent_user`, on by default, left those tasks out of the lock. Set `refuse: {agent_user: false}` in `service_kwargs` |
| An `error` outcome, `verifier_start`, whose reason says the verifier's sandbox `runs its commands as uid ...` or `maps only the uids ...` | A verifier that runs as root started in a sandbox without root, or without the other users its tests switch to: a runtime that runs its commands as your uid, or maps only your uid. Give the verifiers' sandboxes root that can switch users ([Verifiers that run as root](sandboxes-and-harbor.md#verifiers-that-run-as-root)) |
| An `error` outcome, `verifier_start`, whose reason says the sandbox `cannot run the image's /bin/sh as nobody (65534): ...` (`Permission denied`, or `error while loading shared libraries`) | The image's files keep other users out, so the tests could not run anything as another user. An image of the image map that the `oci` builder of platoon before 0.3.0 built under a umask such as 027 does that: build it again with the command the reason names. Another image needs its programs readable and executable by every user |
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
| `The run's per-task statistics go on from those checkpoint recovery-000110 saved, older than checkpoint recovery-000120 ...: checkpoint recovery-000120 names no snapshot of the per-task statistics` (or `... start from empty: ...`; or `... does not have the SHA-256 its checkpoint recorded`, `... cannot be read: ...`) | The checkpoint the start resumed from was saved by a platoon before 0.3.0, or when its statistics could not be written (a warning said so then), or its file in the run directory's `tasks/` is missing or was changed. The start goes on from the newest older checkpoint's statistics that read back, without the groups since, else from empty, which the next full checkpoint saves; `lock.json` lists the start under `tasks_state_lost`. Training goes on either way ([Per-task statistics](train-and-eval.md#per-task-statistics)) |
| `Could not write the per-task statistics with checkpoint recovery-000120 (...): it names no snapshot, the statistics go on, ...` | The run directory's disk refused the file (full, a quota, an I/O error). The statistics stay in memory and the next full checkpoint writes them; free the space before a resume needs them. A resume from that checkpoint goes on from an older one's statistics; its draws resume exactly all the same, from the checkpoint's own state of them ([Per-task statistics](train-and-eval.md#per-task-statistics)) |
| `Could not write where the run's draws stand with checkpoint recovery-000120 (...): a resume from it draws from its batch, ...` | The disk refused the file of the state of the run's draws (`tasks/dispatch-s<start>-<checkpoint>.json`). A resume from that very checkpoint draws from its batch, as one from a checkpoint saved before platoon 0.3.0 does, and rolls out again what had not trained; the next full checkpoint saves its own ([What a resume takes up](train-and-eval.md#checkpoints-and-resume)) |
| `Could not write the 6 groups that settled and no step has trained yet with checkpoint recovery-000120 (...): a resume from it rolls their tasks out again` | The disk refused the file of the groups the checkpoint keeps (`tasks/kept-s<start>-<checkpoint>.json.gz`). A resume from that checkpoint rolls those tasks out again, and trains each once all the same; the next full checkpoint keeps its own ([What a resume takes up](train-and-eval.md#checkpoints-and-resume)) |
| `Cannot keep the settled group (2, 17) with checkpoint recovery-000120` (with a traceback) | One group's output could not be written down (a value the file cannot hold). A resume from that checkpoint rolls that group's task out again; the other groups are kept. Report the traceback if it repeats |
| `The groups the checkpoint kept, which had settled and no step had trained yet, cannot be read, so their tasks are rolled out again: ...` | The file the checkpoint's state of the run's draws names for its kept groups is missing or was changed. The resume rolls those tasks out again; nothing else is lost |
| `The run's sampling inputs changed since checkpoint recovery-000120: environments[code].weight: 1 -> 2; .... The run goes on with them, ...` | The config the run resumed with draws its tasks from other inputs than the checkpoint's: another `train.task_sampler`, `batch_size` or `label_by`, the environment entries listed in another order (`environments: order a, b -> b, a`), or an entry's weight, seed or training tasks (an `exclude_tasks`, a hold-out, a dataset that changed). The run goes on from where its order stopped, in the order the new inputs give, the draws of tasks that left the run are lost, and `lock.json` lists the change under `sampling_changed`. If it was not meant, resume with the config the run started with ([What a resume takes up](train-and-eval.md#checkpoints-and-resume)) |
| `Could not take the fingerprint of the run's sampling inputs` (with a traceback) | The checkpoints of this start save no fingerprint, so a resume from one of them cannot say whether the sampling inputs changed, and does not warn; training is not affected. Report the traceback |
| `The state of the run's draws that checkpoint recovery-000120 saved cannot be read (...): the draws it left outstanding are lost, ...` | Its file in the run directory's `tasks/` is missing, was changed or does not read: the resume draws from the checkpoint's batch, as a resume from a checkpoint saved before platoon 0.3.0 does, and rolls out again what had not trained; from a checkpoint saved under a group admission rule, it draws from where the checkpoint's row says the order stood (`position <n>, where its row says it stood`) |
| `platoon: python -m platoon.train.tinker.restart_wrapper (run_with_restart): platoon 0.3.0 removed it; use watchdog.max_restarts ...` (exit 78) | Set `watchdog.max_restarts` in the training config and run `platoon train` alone, which restarts training after any failure ([Long runs](train-and-eval.md#long-runs)) |

## A mixture's tasks repeat, or never train

`platoon train` and `platoon config validate --tasks` warn when a mixture's weights draw its tasks unevenly
([Train on several environments](train-and-eval.md#train-on-several-environments)):

| Message | Fix |
|---|---|
| `environments[<name>]'s <n> training tasks are each drawn about <ratio> times as often as environments[<other>]'s <m> ...` | The weights give a small entry as large a share as a big one: set `weight: tasks` on every entry to draw every task equally often, or weights in proportion to the tasks you want each entry to stand for |
| `environments[<name>]: its weight (...) gives it about <k> of the run's <slots> task slots (...), fewer than its <n> training tasks, so about <n - k> of them never train; ...` | The entry's share of the run is smaller than its tasks: set `weight: tasks` on every entry, raise the entry's weight, or train longer |

## Rollout permits

Since platoon 0.4.0 each training rollout returns its permit as it ends (`train.permit_release: rollout`, the
default), so a run whose groups wait for rollout permits starts them sooner than before: `train.max_concurrent_rollouts`
counts the rollouts running, no longer the whole groups it holds, more groups are open at once, and more long groups
can go stale. `train.permit_release: group` restores the behavior of earlier releases
([Rollout permits](train-and-eval.md#rollout-permits)). What a run warns of about rollout permits and staleness, under
either setting, and what the metrics say:

| Message or symptom | Fix |
|---|---|
| `3.1% of the groups the last 10 steps consumed (31 of 1000) were dropped as stale, older than their bound (train.max_staleness: 3, or their entry's own); what held the dispatcher back most was the rollout permits (train.max_concurrent_rollouts) (95%). The horizon had room ..., so train.max_staleness: 4 changes only the stale filter ...` (logged once per start of training, from its tenth step on) | Long groups outlasted more steps than their bound allows, and those are the hard, mixed ones; under `permit_release: rollout`, the default, more do in a run that the permits bound, since more groups are open at once and steps are shorter. Raise `train.max_staleness` by one, as the message advises (`max_staleness + 1`), which with the horizon free changes only the stale filter (the message says when the horizon held the dispatcher back, and a larger bound also lets rollouts run further ahead). `train/batch/<column>/<label>/stale_groups` says whose groups go stale; an entry's own `max_staleness` raises its bound alone |
| After an upgrade to 0.4.0, a run logs `Each rollout returns its permit as it ends (train.permit_release: rollout); a group's rollouts start as permits free` at each start, `rollout_client/groups_in_flight_max` exceeds `max_concurrent_rollouts` over the group size, steps are shorter, or more groups are dropped as stale | The new default, which returns each rollout's permit as the rollout ends, so the budget counts the rollouts running. If more than 2% of the groups go stale the run warns (above): raise `train.max_staleness` by one. To go back to earlier releases' behavior, or to compare with it, set `train.permit_release: group` |
| `<n> rollouts ran at once, more than train.max_concurrent_rollouts (<m>): a workflow returned a rollout's permit while the rollout still ran ...` (logged once per start) | A `GroupRolloutWorkflow` subclass ran a member outside its lease block, or more members than its entry's group size: keep each rollout's whole life in its block ([Run rollouts on the lease](../extending/rollout-leases.md)). Only `GroupRolloutWorkflow`'s members count as rollouts in flight, and an evaluation's `GroupRolloutWorkflow` left on the default `stats_scope`, `train`, counts with them: give it `stats_scope="eval"`, as `platoon train` does |
| `IndexError: rollout <i> of task <id> is outside its draw's lease, of size <n> ...` or `RuntimeError: rollout <i> of task <id> entered a second lease block ...` (the group fails) | A workflow ran more rollouts than its entry's group size, or ran an index twice, such as a retry in a block of its own or a whole group run again: run each index once, from 0 to the lease's size less one, and retry a rollout inside its block ([The rules a workflow keeps](../extending/rollout-leases.md#the-rules-a-workflow-keeps)) |
| `Rollout <i> of task <id> entered its lease block after its draw was released ...` or `... was waiting for its permit when its draw was released ...` (logged once per draw) | A rollout outlived its workflow's `arun_episode`, so it ran on no permit: await every rollout, a failed sibling's included, before the workflow returns |
| `Rollout <i> of task <id> returns its permit with <n> of its teardowns still running after 60s ...` | In process, under `permit_release: rollout`, a sandbox or env session teardown that platoon finishes in the background ran past the 60 s that its rollout's permit waits for it, so until it ends more sandboxes or sessions can be alive than the budget. Look in the log for what hangs: a sandbox's stop, or an ORS environment's teardown that waits for a tool call still running in its thread |
| `train/dispatch/blocked_share/permits` near 1, `rollout_client/permits_held_mean` well above `rollouts_in_flight_mean`, and `rollout_client/startup_gate_waiting_mean` above 0 | Groups wait for a worker-startup slot while they hold their permits, as they can with `worker_start_method: exec`: raise `train.workflow_config.worker_launch_concurrent_groups`, or start workers from the fork server, the default. Or, with `train.max_starting_rollouts` set and `rollout_client/start_limit/waiting_mean` above 0, the start limit binds and most of its queue waits at that gate: raise the limit, or turn it off |
| `train/dispatch/blocked_share/permits` near 1 and `permits_held_mean` close to `rollouts_in_flight_mean` | The budget binds: raise `train.max_concurrent_rollouts` while the inference engines have room |

## The first rollouts of a run take minutes to reach the model

At a run's start, or a resume in a new job, the first rollouts can take a minute or more to their first model call,
where later ones take seconds. In the runs measured that was one-time work the first rollouts on each node pay, in
this order, not their sandboxes starting together:

1. **The fork server's import** of the rollout worker's stack, under process isolation: the run logs `rollout worker
   fork server <pid> imported the worker's stack in 19.5s`. Every worker waits for it once, at the run's start.
2. **A harness toolchain's copy to the node.** The first rollout of a CLI harness on a node copies the harness's
   toolchain to node-local storage, and the node's other rollouts wait for it: about a minute for a large toolchain,
   again in each new job, whose node-local storage starts empty.
3. **What each worker imports after its fork**: the modules of an environment's service or of a harness that the fork
   server did not import, read from shared storage by every worker that starts, again at every step. The fork server
   imports the environments' packages (`environments[].package`), so a package that imports what its rollouts use has
   it imported once.

`train.max_starting_rollouts` shortens none of these: it orders start-ups, so with any value the whole first wave
reaches the model later, though its first rollouts can reach it sooner. It is for start-ups that contend for a shared
capacity ([Rollout start-ups](train-and-eval.md#rollout-start-ups-experimental)), and the run says when it bounds more
than start-ups:

| Message | Cause and fix |
|---|---|
| `A rollout held its start slot for 600s without a model call, so the start limit released it on its timer: ...` (logged once per start) | A rollout made no model call through platoon's gateway in `workflow_config.worker_startup_budget_seconds` + 300 s: its harness calls no model there (`platoon-oracle`, `platoon-noop`, a rollout function of your own), or its start-up takes longer. The limit stays live, letting another rollout start; raise the budget if start-ups are that slow |
| `Most rollouts returned their start slot only at their end or on its timer (40 of 64): ...` (logged once per start) | The rollouts make no model call through platoon's gateway, or fail before one (see `train/batch/outcome/error` and their logs), so `train.max_starting_rollouts` bounds whole rollouts, not start-ups: turn it off for this workload, or fix what fails |
| `train.max_starting_rollouts (4) is below the largest group size (8): a group's members start in waves; ...` | Each member keeps its full straggler grace, so this only costs time: set it to at least the group size unless start-up is what bounds the run |

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
| `rollout worker <id> got no routing ids for its 4 model calls, nor did any train rollout before it in this start of training: the model service sends none. ...` | No rollout of this start of training, in the scope the line names (`train` or `eval`), got a routing id, which a service with rollout routing replay (R3) sends with each sampled sequence. A dense model's service sends none, and there is nothing to fix. A mixture-of-experts model then trains without routing replay, which takes a service that does it. The line comes once for each start of training and scope, with `rollout_isolation: process` only; the metric `train/rollout_worker_routing_id_coverage` (and `eval/...`) is the share of each rollout's model calls that got a routing id |
| `rollout worker <id> returned routing ids for only 2/4 model calls (R3 routing_handle coverage 0.500)` (or `... 0/4 ..., though the model service sent routing ids to earlier train rollouts`) | The service sends routing ids, but this rollout's model calls, or some of them, got none. A datum whose last model call got none has no `routing_handle`, so with a mixture-of-experts model it trains without routing replay, on a worse gradient that nothing else reports |
| `rollout worker <id> got routing ids for 4/4 model calls, the first train rollout in this start of training to get any, after 3 rollout(s) without them (...)` | Either the service began to send routing ids, or the rollouts without them, the first three of which the line names, lost theirs. Their datums have no `routing_handle`; from then on, each rollout without routing ids is logged |

## Rollouts are slow to start

A rollout's start, from its group's start to its first model call, holds one-time work that platoon now does before
the first rollout, and work that each rollout does again:

- **The harness toolchain.** A CLI harness (`openhands`) runs from a toolchain, tens of thousands of files that are
  copied once per node from the shared filesystem to node-local disk
  ([Harness toolchains](sandboxes-and-harbor.md#harness-toolchains)). `platoon train` stages its toolchains while it
  opens the training service and saves at launch, and starts no rollout, training's or an evaluation's, before that
  is done; `platoon eval` does so while it lists its tasks. So the copy overlaps the launch save and the fork
  server's start, and the first rollouts wait only for what it takes beyond those: the `Staged the toolchains` line
  gives how long that was. The trainer waits ten minutes at most, and not past a shutdown request; the rollouts then
  start and wait for the copy themselves, as before. A node that runs rollouts without such a start stages its copy
  at its first rollout, as before.
- **The rollout workers' fork server** (`rollout_isolation: process` with the default `worker_start_method:
  forkserver`) starts as soon as the training client is open. Besides the worker's stack it imports what each
  environment and harness imports when a rollout opens it, derived from the config: an ORS environment's module and
  the openreward SDK (which pulls in anthropic, openai and opentelemetry), each function harness's module (CodeAct's
  pulls in IPython), the programs and what `harness_rollout` imports. It also builds what each worker built: the
  harness and program lookups, and an in-process ORS environment's task lists for the run's splits. A worker then
  starts with all of it, instead of importing it again; with a hundred workers starting at once on a shared
  filesystem that was most of a rollout's start. Workers started by `exec`, in-process rollouts and rollouts under
  `worker_python_env_isolation` (which import their environments inside a Python overlay of their own) preload
  nothing.

| Message | Cause and fix |
|---|---|
| `Staging the toolchains openhands-1.54.0 on this node ahead of the first rollout`, then `toolchain openhands-1.54.0 at <copy>: copied (81.4s)` and `Staged the toolchains in 81.4s, 43.4s of it after the rest of the start-up: the rollouts start now` | The copy, once per node, and how long the trainer waited for it after its own start-up (the launch save, the fork server): the first rollouts started that much later. Each start and each resume in a new job copies again when the copy lives in the job's scratch (`$PBS_LOCALDIR`, `$SLURM_TMPDIR`). To keep it across jobs, set `PLATOON_TOOLCHAIN_CACHE` to a node-local directory that outlives the job: the next job's first staging then checks the copy (`... was staged by an earlier job, and its files still match its manifest (1.3s)`), hashing it again against the manifest's tree hash, instead of copying it. Old versions are not removed: delete them while no job uses them |
| `Starting the rollouts before the toolchains openhands-1.54.0 are staged (after 600s): ...` (or `(a shutdown was asked for)`) | The staging took longer than the trainer waits for it, or the run is stopping: the rollouts start, and each that needs the toolchain waits for the copy under its lock, within its own time limit. A later line says when the staging ended (`Staged the toolchains in ..., after the rollouts started`). A copy that takes minutes points at a slow or busy shared filesystem |
| `toolchain ... was staged by an earlier job, but its files no longer matched its manifest: copied again`, and `the old copy is set aside at <cache>/.<copy>.<hex>.stale, since rollouts of other jobs may still run from it: remove it once none does` | Files of a kept copy were removed or changed (a node's cleanup of old files, say): it was copied again beside it, and the fresh copy took its place. The old copy is kept whole, for the rollouts of other jobs on the node that run from it; delete it once none does. A cache that a cleanup prunes is better left in the job's scratch |
| `the staged toolchain <copy> is used without being checked against its manifest: no stamp can be written beside it (...)` (or `its lock file cannot be opened (...)`) | `PLATOON_TOOLCHAIN_CACHE` is read-only, another user's, or on a full disk, so platoon cannot record that it checked the kept copy, and uses it as it is, as platoon 0.3 did. Make the cache writable to have each job check it once |
| `could not stage the toolchain <name> ahead of the rollouts; each rollout stages it itself: ...` | The toolchain could not be staged (not built, unreadable, not matching its manifest): each rollout tries again, and fails as `harness_setup` with the same reason. Fix what the message says, usually with `platoon prepare` |
| `rollout worker fork server <pid> imported the worker's stack and 9 module(s) the environments open with in 15.0s` | The fork server is ready; the count is of the modules it imported for the environments that the worker's stack had not. It takes a few seconds more than before, once per run, while the trainer saves at launch, and its workers start with all of it |
| `the rollout worker fork server left <module> out of what it imports ahead of its workers, which then import or build it themselves: it started threads: ...` (or `it left file descriptors open: ...`, `it started processes: ...`, `it drew 1 seed(s) from the OS ...`, `it does not import: ...`) | Importing the module, or the build named, is not fork-safe: a worker forked next to a running thread can deadlock, one that inherits a file, socket or event loop shares it with every other worker, and every worker would draw the numbers of a numpy generator seeded at import. The server started over without it, once for all such items, and the workers import it themselves, as before. Nothing to fix in platoon; the module's maker could defer the thread, file, socket or generator to its first use |
| Each rollout still spends seconds before its first model call, on an idle node | An environment or harness imports more only when a rollout opens it, which the config does not show. List those modules in `train.workflow_config.worker_preload` (evaluation inherits it), and the fork server imports them too, each only when it is fork-safe. Leave out a module that makes a value of its own process when it is imported, such as an id or a temporary directory, which every worker would share. It applies to workers forked from the fork server only; elsewhere a warning says it has no effect |

## Still stuck

Open the trajectory in the viewer (`platoon viz show-dump .../trajectory_collection.json`) to see each step the
agent took. For how the pieces fit together, read [the architecture](../architecture.md). If platoon is missing
something you need, the [RFC index](../rfcs/README.md) lists the proposals in progress and explains how to propose a
change.
