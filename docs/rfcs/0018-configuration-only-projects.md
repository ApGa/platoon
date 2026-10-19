# RFC 0018: Configuration-only projects

| | |
|---|---|
| Status | Draft |
| Created | 2026-10-17 |
| Authors | Apurva Gandhi (@ApGa) |
| Builds on | [RFC 0010](0010-harbor-registry-and-multi-container-tasks.md) (Harbor datasets, images and grading), [RFC 0016](0016-component-configuration.md) (component configuration), [RFC 0008](0008-config-and-cli.md) (config and CLI: site profiles, run directories and exit codes, in part) |
| Discussion | The pull request that adds this RFC |
| Implementation | Milestone M1 in the pull request that adds this RFC; M2 in the pull request that adds `train.label_by`; M3 in the pull request that lays `eval.workflow_config` over `train.workflow_config`; M4 in the pull request that adds `platoon prepare`; M5 in the pull request that adds the site profile; M6 in the pull request that restarts training after failures |

## Summary

A project that trains on a Harbor dataset with platoon should write configuration only: an environment entry that
names the dataset, a training config, and a launch line.

**What this RFC changes:**
- One dataset becomes one environment, whatever reward file each of its tasks writes.
- A held-out split and per-source metrics come from the dataset's own fields.
- Training evaluates with its own rollout settings, from step 0 when a config asks.
- Everything a config needs is prepared on a site with one command, into default locations.
- Where things live, and which isolating sandbox runtime runs them, move into a per-site profile.
- A config's input paths resolve against the file that names them.
- `platoon train` restarts after failures, prunes its own recovery checkpoints, reaps its sandboxes, and keeps one
  run per run directory.

**What it adds:**
- keys: `holdout`, `{metadata: [...]}` index entries, `refuse.uploads_over_mb`, `eval.at_start`,
  `watchdog.max_restarts`, `checkpoint.keep_recovery`;
- commands: `platoon prepare`, `platoon sandbox reap`;
- one file, the site profile, and one variable, `PLATOON_SITE`.

Most other changes give an existing key another value, or a default.

## Motivation

The first project to train on a Harbor dataset with platoon was a downstream project, on a private science dataset
in Harbor's format. It needed 20 files and 1,826 lines beside its configs:

- **One dataset, three environments.** SmolData's tasks write `reward.txt` (key `reward`) or `reward.json` (keys
  `correctness` and `submission`). A reward contract trains one named key, and a reading without it is an error.
  So the dataset became three entries, joined by YAML anchors, plus:
  - hand weights to restore its mix, which skipped 255 tasks and repeated 257 every epoch;
  - group limits keyed by the three names;
  - a 316-line script that read each task's `tests/test.sh` to route it;
  - a test to keep it all in sync.
- **Splits by script.** Without a manifest, every split lists every task, so `platoon train` evaluates on its
  training tasks without a word. Holding out whole groups needed the same script, and split files named by path.
  - A group must also be the right unit. SmolData's questions about one Kaggle dataset share its files, so a
    question-level split held out tasks whose data training had seen. Kaggle dataset names are not enough either:
    the same public tables are uploaded under several names.
- **A second file for one number.** Training never evaluates its starting policy, and its evaluation section starts
  from default rollout settings. The base model's held-out score needed an evaluation config that restated seven
  training settings under other names.
- **Preparation outside the config.** `platoon harbor fetch|verify|images` read a separate fetch config that
  repeats the entry's dataset keys, and image maps and toolchain roots have no default location. `platoon harbor
  fetch` re-hashes 30 GB even when the lock is current, and a run that lacks an image fails rollout by rollout. The
  example ran six prep steps from a PBS script, one of them a `python -c` call into `ensure_lock`.
- **Site facts with nowhere to go.** The sandbox provider and its options sit in the shared entry; caches and
  toolchains in environment variables; the run directory in `log_path`; and `TMPDIR` must be moved off a small
  `/tmp` by hand.
- **Paths against the working directory.** Service options resolve relative paths against the process's working
  directory, while `extends:` and fetch configs resolve against their file.
- **Wrappers around a long run.** A run restarts only under `restart_wrapper`, and only after its watchdog fires,
  while a stalled training service ends it. Its recovery checkpoints (352 GB each for a 27B model) are left to a
  pruner, and leftover sandboxes to a Python one-liner.

## Proposal

### 1. A dataset is one environment

#### 1.1 Reward-key alternatives

```yaml
service_kwargs:
  reward: {primary: [reward, correctness], range: [0, 1]}
```

**`primary` names the reward that trains,** as today, or lists the names that may train. With a list:
- **Exactly one listed key.** A reading must report exactly one of the listed names. None is an error
  (`verifier_no_primary`, as today); two or more make the reading invalid (`verifier_ambiguous_primary`).
- **One reward file.** The reading must come from one reward file: a verifier that left both `reward.json` and
  `reward.txt` is invalid (`verifier_ambiguous_reward_files`).
- **Recorded.** The outcome records the name that trained as its `primary`.

So the list's order never matters. A policy whose code the tests run cannot choose the key by writing a second key
or a second file. Milestone M2 adds the group check: members of a training group that trained different names do
not train together (section 1.5).

**The rest of the contract:**
- `reward: [reward, correctness]` is short for `{primary: [reward, correctness]}`, as `reward: correctness` is short
  for `{primary: correctness}`.
- `components`, when set, contains every listed name. `range` bounds the name that trained. A `check` hook sees the
  rewards, as today.
- With one name, a verifier's reading is graded as today; a list of one name follows the list's rules.
- Outcomes platoon scores itself score 0 under the contract's first name; today they score `reward` whatever the
  contract names. They are a reward file that is not a regular file, and an agent that made its working directory
  unusable.

#### 1.2 Index fields from each task's metadata

```yaml
service_kwargs:
  index:
    - {metadata: [kaggle_dataset_name]}
```

- A fetch reads each selected task's `task.toml` `[metadata]` table, and records the named scalar fields among the
  task's index fields in the lock. There `holdout`, metrics and check hooks read them.
- The fields are read after selection, so `select` cannot name them.
- A field that the dataset's own index gives the same task with another value fails the fetch, naming both values.
- Like every index entry, it is one of the lock's inputs: adding it fetches a new lock, and configs without it keep
  theirs.

#### 1.3 A held-out split from the index

```yaml
service_kwargs:
  holdout: {fraction: 0.05, group_by: [kaggle_dataset_name, split_group], stratify_by: source_dataset}  # seed: 0
```

**Splits.** The service lists two: `eval`, the held-out tasks, and `train`, the rest.
- Those are the splits `platoon train` and `platoon eval` ask for, so the entry needs no `split_name`.
- Another split name is a configuration error that names the two, and so is an `eval` split with no runnable task.
- `holdout` and `splits` together are a configuration error.

**The rule** runs over the dataset's selected tasks: a lock's rows, refused ones included, or a `dataset_dir`'s
tasks.
- **Strata.** Tasks fall into strata by their `stratify_by` field (one stratum without it). A runnable task without
  the field is an error; a refused one sits out.
- **Groups.** Tasks fall into groups by the first `group_by` field they have. A runnable task with none of them is an
  error; a refused one sits out.
- **Shared input files.** Groups whose tasks share an input file are joined into one, also through other groups,
  whatever `group_by` says (section 1.3.1). Files compare by their bytes, and text files by their text, line endings
  and a byte-order mark aside. So tasks that hold the same data always land on the same side. A joined group takes
  the name of its member with the most tasks, so a group that joins none keeps its name and its rank.
- **The walk.** Within each stratum, groups are ranked by the SHA-256 of `"<seed>\0<group>"`. Walking that order, a
  group is held out while the stratum's held-out tasks, with the group's, still number at most
  ⌈`fraction` × the stratum's selected tasks⌉. A group that does not fit is passed over, and a group is never split:
  one with tasks in several strata must fit the share of each, and is held out in all of them or in none.
- **The lists.** Each split lists its runnable tasks, in the order of the SHA-256 of `"<seed>\0<task id>"`, so
  `limit` takes a spread-out sample.
  - Refused tasks leave both splits wherever they fell, so changing a refusal never moves a group.
  - `exclude_tasks` apply after, as today.
- **Reproducible.** The same lock and rule give the same split anywhere. Rollout workers receive task ids and
  compute nothing.

**The report.** The trainer logs it when it lists the tasks, and `platoon harbor fetch CONFIG` (later `platoon
prepare`) prints it:
- per stratum: the selected and runnable tasks, the held-out tasks and groups, the largest held-out group and its
  share, and refusals by reason;
- the groups the `group_by` fields gave, how many of them shared input files joined and into how many, the files
  that joined them, and the largest group, with a file that joined it when it is a joined one;
- a digest of the evaluation's task ids, as the service lists them (before the entry's `exclude_tasks`).

**On the science dataset's phase 0** (seeds 0-4), the rule holds out 202-242 runnable SmolData tasks (245 with the refused
ones) and 61 MathConstructOptimize tasks per seed. For seed 0: 220 SmolData tasks in 21 groups (largest, the Titanic
tables, 28%) and 61 MathConstructOptimize tasks in 23 families (largest 28%). Shared input files join 90 of the 563
groups that `kaggle_dataset_name`, else `split_group`, gives into 30, leaving 503; the largest group has 346 tasks.
For every seed, no held-out task shares an input file with a training task, nor a Kaggle dataset. Grouped by Kaggle
name alone, 63 of seed 0's 218 held-out SmolData tasks (29%) did, and 2.5-46% for the other seeds. Compared byte for
byte only, 53 of seed 0's 220 (24%) still did, and 82 of seed 4's 221 (37%): the same tables with other line
endings, or with a byte-order mark (section 1.3.1).

##### 1.3.1 Input files

**What counts.** A task's input files are what its agents' sandbox takes from the task: the files of its build
context, `environment/`, that its Dockerfile's `COPY` and `ADD` steps take, whether the build prefix bakes them into
the image or a session uploads them when it opens (the deferred steps), and a prebuilt-image task's `environment/`,
which Harbor uploads whole. Never inputs:
- the build's own files: `Dockerfile`, `.dockerignore`, a compose file;
- a file of the context that no step takes, which no sandbox holds;
- the tests and the solution, which the agents never see, and a separate verifier's own build context;
- remote content (`ADD <url>`, `COPY --from`);
- an empty file, which holds no data and joins nothing.

**Why these.**
- *Not only what a session uploads* (what `uploads_over_mb` counts). What the agents see must not depend on how
  the image is built. SmolData's own Dockerfile bakes `input/` into its image, with a `COPY` before a `RUN`; the
  downstream project's config moves the `COPY` behind the `RUN` with an overlay, so a session uploads it. Uploads
  alone would protect the overlaid config and silently not the original.
- *Not the whole `environment/`.* A file no step takes never reaches a sandbox, yet can join groups. Every SmolData
  task holds the same `environment/pull_bucket.py`, a helper its Dockerfile never copies: counting it would make the
  4,900 tasks one group, which no share fits.
- *Not the tests.* The agents never see them, and datasets share grading code: SmolData's grader is the same in
  every task.

**Text compares as text.** A public table is uploaded under several names, and not always byte for byte. On phase 0
the Titanic tables come with CRLF line endings and with LF ones, and the IBM attrition table with a byte-order mark
and without. Compared byte for byte, 53 of seed 0's 220 held-out SmolData tasks (24%) and 82 of seed 4's 221 (37%)
held a table that training held too. So each input file has a digest:
- a binary file's (a NUL byte among its first 8000 bytes, as git tells them apart) is the SHA-256 of its bytes;
- a text file's is the SHA-256 of its text with CRLF and CR line endings as LF, every line ended, and a leading
  byte-order mark, the spaces and tabs that end its lines, and its blank lines dropped. Text already in that form has
  its own SHA-256 as its digest, and text that normalizes to nothing joins nothing, as an empty file does.

Other differences escape it: a table re-quoted, with other columns or a subset of its rows, and data that is not in
the task's files (an `ADD` of a URL, a here-document, a `RUN` that writes it). Comparing tables as data is a
dataset's job.

**Files every task holds by design.** Some datasets give each task the same files as part of the environment, such
as a game's code and rules; the TextCraft-Synth export copies its game and its recipes into every task. Shared by
all, they make the dataset one group. Then the `eval` split holds no task, and listing it is a configuration error
that names a file that joined the largest group, and the key that fixes it:

```yaml
holdout: {fraction: 0.05, group_by: level, ignore_inputs: [environment/game.py, environment/rules.json]}
```

`ignore_inputs` takes patterns of task-relative paths (fnmatch, so `*` crosses `/`). A pattern that matches no
input file is an error. A run records the patterns with the rule (`identity.json`), and the report counts the files
they matched.

**Where the digests come from.**
- *The lock.* A fetch records each task's input files in its lock row, by path, with the digest of each
  (`input_files`). A binary file's is the SHA-256 the fetch computes for the task's digest anyway; a text file is read
  once more, to normalize it. The task digest that rollout workers check vouches for the files. On phase 0 they add
  41,503 entries, about 5 MB, to the lock.
- *A dataset directory* (`dataset_dir`, `task_dirs`) is hashed when its tasks are listed, each file once per
  process.
- *A lock written before this change* still answers its config. Input files are an output of the fetch, not one of
  its inputs, so the inputs digest does not change, and a run without a hold-out reads the lock as before. A
  hold-out on it is a configuration error that names `platoon harbor fetch`, never a split without the files.
  `platoon harbor fetch` treats such a pinned lock as out of date and fetches it again, once. `platoon harbor verify`
  of a config with a hold-out exits 78 on it, as it does before any fetch.

**The default, not an option.** The joining is always on, and no key turns it off: a hold-out that leaks is never
what a project wants, and a project that knew where its data is shared would not need the protection.
`ignore_inputs` does not turn it off either: it names the files that are not data. Platoon never decides that by
itself, since a guess would fail silently: a dataset whose questions all ask about one table would then hold out
questions on data that training has seen, with no error.

#### 1.4 Refusing tasks that would upload too much

```yaml
service_kwargs:
  refuse: {uploads_over_mb: 100}
```

- **What a fetch records.** Each task's `flags.upload_bytes`: what a session copies into one of its sandboxes, the
  larger of the agents' and a separate verifier's. That is the sources of the deferred `COPY` steps, or a
  prebuilt-image task's `environment/`.
- **The refusal.** With `uploads_over_mb`, a task above the limit is refused as `uploads_over_mb`, with its size in
  the reason. It is recorded in the lock like any refusal, and counted in the report.
- **The lock's inputs.** Like every refusal, the threshold is one of the lock's inputs. Unset, it leaves the inputs
  digest byte-identical, so existing locks still answer their configs.
- **Strict switches.** `refuse`'s other keys take booleans: a string such as `"false"` is a configuration error,
  where today it reads as true.
- **A stopgap for some datasets.** Once sessions stage a task's inputs once per node, a project that set the
  threshold for speed deletes its line. Since the hold-out walks before refusals, its groups do not move.

#### 1.5 Per-source metrics, and one trained key per group

- **Index fields become columns.** A Harbor listing gives each task's scalar index fields as dataset columns beside
  `task_id`, each value as text (`true` or `false` for a boolean, a number as Python writes it), so that a field
  whose values mix types fits one column. A field named like one of platoon's own columns (`task_id`, `_platoon*`) is
  left out of the dataset. Naming such a field in a key that reads columns (`train.label_by`, and the deprecated
  `straggler_policy_key`) is an error.
- **The label column.** `train.label_by` names the column that labels a run's tasks, by default the mixture entry
  (`_platoon_environment`). It replaces `train.rollout_group_key` and `workflow_config.straggler_policy_key` (open
  question 1): `rollout_group_limits`, the straggler policies of training and evaluation, and the metrics below key
  on its labels, a task's value of the column as text. The old keys load as deprecated aliases until 0.3.0 (RFC 0016,
  section 10), and a workflow config that still sets `straggler_policy_key` keys its own policies on that column. A
  column that no training task has stops the run with exit 78, before anything runs.
- **Metrics by the label column.** The per-step metrics a mixture logged per entry (`train/batch/environment/<name>/...`)
  break down by the label column instead. Its default is the entry, so those names do not change.
  - `label_by: source_dataset` logs `train/batch/source_dataset/<value>/{groups, task_reward, outcome/<status>,
    reward_key/<name>, reward_key_mismatch}` every step, also at zero, with `/` in a value written as `--`.
  - Evaluation logs the same breakdown under `eval/source_dataset/<value>/...`.
  - A column with more than 64 values logs only the 64 most frequent, with a warning. `rollout_group_limits` and
    straggler policies are never capped.
- **One trained key per group.** A group whose members trained different reward names does not train
  (`reward_key_mismatch`, counted per step). Each rollout's record names the key it trained (its root's
  `misc.outcome.reward_key`, the outcome's `primary`), and each step counts its rollouts by trained name
  (`train/batch/reward_key/<name>`); `platoon eval` names it in each rollout's `metadata.json` and counts by it in its
  report (`reward_key_counts`). Platoon's own zeros carry no reported name: their outcome's metadata marks them
  (`platoon_zero`), they record `reward_key: null`, and they take no part in the check.

### 2. Evaluation during training

1. **Step 0.** Training that evaluates can also evaluate its starting policy, at step 0, as tinker-cookbook and SkyRL
   (`eval_before_train`) do: `eval.at_start: true`. It is off by default (decided 2026-10-19; M1 had made it the rule):
   the first update waits for that evaluation, which on a large held-out split with slow sandbox starts takes the first
   hours of a run, and the original platoon never evaluated step 0. A resumed run evaluates at its multiples, as before,
   and the step it resumes at only when the run directory's `evaluations.jsonl`, the record of the evaluations that
   completed, does not list it: a step's checkpoint is saved before its evaluation starts, so the checkpoints cannot
   tell an evaluation that completed from one a crash cut short. An evaluation runs beside training and is logged when
   it ends, at the step training has reached by then, so each one also logs the step it evaluated (`eval/step`). Its
   rollouts sample the policy that is current when they run, so a run that evaluates its starting policy applies its
   first update only once that evaluation has ended.
   - **Cadence in steps.** `eval.every` counts training steps, and so does `checkpoint.every`; 0, the default of
     both, turns the event off. `strategy`, which made `every` count epochs by default, so that `eval: {every: 20}`
     meant 20 epochs, is a deprecated key until 0.3.0: a config that still sets it keeps its meaning.
   - **Concurrency in rollouts.** `train.max_concurrent_rollouts` and `eval.max_concurrent_rollouts` count the
     rollouts that run at once, a whole group at a time (by default every rollout of a training batch, and 256 in an
     evaluation). They replace `num_concurrent_rollout_workflow_workers`, which counted groups, so that one number
     meant 256 rollouts in training (groups of 8) and 32 in evaluation; until 0.3.0 it reads as that many groups'
     rollouts.
2. **Training's settings.** A training config's `eval.workflow_config` is laid over `train.workflow_config`
   instead of the defaults:
   - **inherited:** the rollout settings (`rollout_config`) and the rollout-worker settings (`rollout_isolation`,
     `worker_start_method` and the other `worker_*` keys);
   - **evaluation's own defaults, unless set:** the keys that shape groups, advantages and filters (`group_size: 1`,
     `min_successful_group_size`, the straggler keys, `leave_one_out_baseline`, `filter_errors`,
     `filter_zero_advantage_datums`, the depth and subagent-datum keys);
   - **merging:** a key the section sets replaces training's. Sections merge key by key (`rollout_config`, and its
     `inference_params`); `null` unsets a value training sets (an `inference_params` key then takes the gateway's
     default); a free-form mapping such as `extra` replaces training's whole; and a key and the deprecated name it
     was renamed from count as one key.

   Before, the evaluation started from defaults: one that restated nothing sampled 4,096 tokens over 30 agent steps
   while training sampled 49,152 over 200. The loader lays the section over the built `train.workflow_config` (the
   trainer config class's `LAID_OVER`), so `platoon config show --defaults` and `platoon rollout --split eval` show
   and run what each evaluation runs with, while `config.resolved.yaml` keeps the section as the config gave it.
3. **(Later.)** `platoon eval` takes a training config:
   - the policy comes from `train`;
   - the rollouts from its `eval` section, as point 2 resolves it;
   - the output goes under the run directory (`<run>/eval/<UTC time>`), with a sandbox label of its own;
   - it checks the run's `identity.json` (section 7).

### 3. Preparing a config

```text
platoon prepare CONFIG [key=value ...] [--dry-run]
```

For each entry of a training or evaluation config, on the site it runs on:

1. **The dataset.** A Harbor `dataset` that no lock answers is fetched, only what the cache lacks, as `platoon
   harbor fetch CONFIG` fetches it (a local directory is fetched again); an entry with a hold-out prints its report.
2. **Images.** The images the runnable tasks need, and the site's image map lacks, are built with the builder of the
   entry's sandbox provider: `oci` for `singularity`, `docker` for `docker`. The `process` provider runs no image, so
   its keys are recorded as `local`. For a provider that pulls images from a registry, it prints the build-and-push
   command and exits 1, rather than guess a registry. An entry that serves a directory builds none.
3. **Toolchains.** Each toolchain the entry's harnesses name that is not built is built (RFC 0016 M2's check says
   which). One built from another recipe is never replaced in place: a live run of another config, or of another
   platoon, may be staging its rollouts from it, so the step fails and names the forced build
   (`platoon toolchain build --force`), to run while no job uses it (decided at M4).
4. **The policy's tokenizer.** The model's tokenizer and chat template are loaded once, so the run's processes find
   them in the Hugging Face cache.

**Output.** It prints, per entry, the report of section 1.3 and the images and toolchains it found or built. Each
step does nothing when its output is current.
- **Exit codes:** 0 when the config is ready, 1 when a build failed, 78 on a configuration error.
- **`--dry-run`** reports what is missing and changes nothing, and exits 1 when anything is.
- **Two at once,** also on two nodes of a site, take turns: a fetch on its lock's file lock, each image on one of its
  own beside the image map, each toolchain on its own; the second finds the first's work current.
- **Only the environments** of the config are read, and checked as a run checks them, so a training config that
  leaves `log_path` to its launcher prepares as it is.
- **When to run it:** again after any change to an entry's dataset, selection, refusals, index, overlays or
  harness.

**Default locations:**
- The Harbor cache, with its locks, defaults to `<cache root>/harbor`, and toolchains to `<cache root>/toolchains`.
- A lock-backed entry's image map defaults to `<cache root>/harbor/images/<provider>.json`, with its image files in
  `<cache root>/harbor/images/<provider>/`. A missing default file means no map. Entries with `dataset_dir` or
  `task_dirs` keep today's behavior: they run their `docker_image` or `default_image` unless they name `image_map`.
- The cache root comes from the site profile (section 4), else `~/.cache/platoon`. `$PLATOON_HARBOR_CACHE` and
  `$PLATOON_TOOLCHAINS` still name the Harbor cache and the toolchain root until 0.3.0 removes them (section 4).

**The expert commands.** `platoon harbor fetch|verify|images ...` take a training or evaluation config, and act on
each Harbor entry with a `dataset` (`images` also on one with a `lock`), with the entry's image map and its
provider's builder. They remain for what needs their flags: `--shard i/n`, `--push`, `--key`, `--builder`,
`alias`.
- `platoon harbor fetch` does nothing for a dataset pinned to a commit when a lock answers and the cache holds its
  tasks. `--repair` re-hashes the cache (what it always did), and `--update` resolves moving references. A local
  directory is fetched again each time, since its files may change under the same path.
- Fetch config files are deprecated: a run config's entry says the same. They still work for one release, with a
  deprecation warning, and platoon 0.3.0 removes them (decided at M4: platoon's own commands, docs and the first
  project's old configs read them).

**At run start,** `platoon train` and `eval` still fetch a pinned dataset that no lock answers. Before any rollout,
and before the training service opens, these stop the run with exit 78, with the `platoon prepare` command for the
config and overrides they were given in the message:
- a task whose image, or its separate verifier's, is not in the map, or whose image file is gone or is not the size
  its build recorded (checked only for the tasks of a fetched dataset whose plan has a build key, when they are
  listed, right after a fetch);
- a harness whose toolchain is not built (checked before the tasks are listed); one built from another recipe stops
  the run too, naming the forced build instead.

`platoon rollout`, which runs one task and never fetches, also stops when no lock answers the task's entry.

**A run that prepares first** (decided at M4). `platoon train`, `eval` and `rollout` take `--prepare`. Once the
command has checked its config, it runs `platoon prepare` on the same config and overrides, through the same code and
under the same file locks, and starts the run only when that ends ready. Otherwise the run does not start, and the
command exits with prepare's status: 1, or 78 for a configuration error. Prepare's lines go to stderr, so the run's
stdout holds what it prints without the flag. Two runs with the flag take turns as two prepares do, and once the
config is ready the flag makes nothing (a local directory is fetched again, as prepare fetches it). The tokenizer it
loads is the run's: `platoon rollout`'s `--model` and `--renderer` replace the config's. An evaluation with
`stage=report` runs no rollout and prepares nothing.
- **Where:** a laptop or a single node, where building inline is cheap.
- **Where not:** a cluster's GPU jobs. There prepare stays a step of its own, once, on a CPU node: a GPU job with the
  flag holds its GPUs while it fetches and builds, which can take hours, and a second job that waits on the first
  one's builds holds its GPUs too. Without the flag, a run refuses within seconds and names the prepare command.

**Later.** A fetch reuses the rows of earlier locks of the same resolved commit, for tasks whose files and overlays
did not change. Changing `select` or `refuse` then re-evaluates rows instead of re-staging every task.

### 4. The site profile

```yaml
# ~/.config/platoon/site.yaml, or the file $PLATOON_SITE names
cache: /shared/$USER/platoon-cache
runs: /shared/$USER/platoon-runs
provider: singularity
provider_options: {allow_network_none: true, rootfs_layout: fuse-overlay}
```

**One file per site,** written by whoever runs platoon there, never part of a project.
- It is `$PLATOON_SITE` (the one new variable), else `$XDG_CONFIG_HOME/platoon/site.yaml`, else none.
- It has four keys. Another key, or a `PLATOON_SITE` that does not exist, is a configuration error.
- `~` and `$VAR` expand, so one file can serve a group without shared run directories. A variable that is not set is
  an error, and a relative path is relative to the file (decided at M5).

**The keys:**
- **`cache`:** the root of section 3's default locations. It replaces the entries' `cache`, `image_map` and
  `toolchain_root`, and the variables `PLATOON_HARBOR_CACHE` and `PLATOON_TOOLCHAINS`. These are deprecated (decided
  at M5): since launchers and configs written before the profile set them, they work as before, with a deprecation
  warning, until 0.3.0 removes them. A variable set beside a profile's `cache` must name the directory the profile gives:
  which one a run used would otherwise depend on which part of platoon looked.
- **`runs`:** the default of `log_path`, and of evaluation output, for a config that sets none.
  - An explicit relative `log_path` keeps today's base, the working directory.
  - When `log_path` comes from the profile, `stats.experiment_name` defaults to the config file's name, so two
    configs never share a run directory by default. Two files of one name in different projects would, so a run
    directory named so belongs to its file: the run records the file's path, and a start of another file is refused
    (decided at M5).
  - An evaluation config without `inference.output_dir` writes each start to a new directory,
    `<runs>/eval/<policy>/<UTC time>` (section 8, point 3; M5 had named it `<runs>/eval/<config file name>`).
- **`provider` and `provider_options`:** the sandbox runtime of every Harbor entry that names none, and of
  `platoon sandbox reap` for a record that names none. They are spelled as an entry spells them; an entry that names
  its own provider uses its own options. An entry that names no provider lays its own `provider_options` over the
  profile's (decided at M5), so the run config adds what the profile may not, binds or `extra_start_args`.

**What it may not do.** The profile says where files live, and which isolating runtime starts sandboxes; it never
weakens isolation and never changes what trains.
- **No unisolated provider.** It may name only a provider that isolates: `process` is a configuration error there.
  `platoon train` refuses a Harbor entry on an unisolated provider unless the run config itself names it, and so do
  `eval` and `rollout` (decided at M5). A provider class says that it isolates with `isolates = True`; one that does
  not say so cannot be named in a profile.
- **No isolation options.** It may set only options that do not weaken isolation: `allow_network_none`,
  `rootfs_layout`, `scratch_root`, `image_cache`, `stage_images` and `gpu`. Binds and extra runtime arguments belong
  in the run config.
- **No training settings.** It holds no dataset, model, reward or recipe key.

**Visible and recorded.**
- `platoon train`, `eval`, `rollout` and `prepare` print the profile they used and the values it gave.
- They refuse a provider the node cannot run, with the profile snippet in the message (a provider class says why
  through `unavailable(options)`); `prepare --dry-run` reports, and refuses nothing.
- The run's `lock.json` records the file's path and SHA-256, and the effective provider and options with their
  source. `config.resolved.yaml` records the values, so the run directory reloads without the profile, and names each
  Harbor entry's provider, the service's default too (decided at M5), so a supervised restart keeps it whatever the
  profile says by then.

**Two changes come with it,** because both remove site boilerplate:
- platoon's temporary files go under node scratch (`$PLATOON_NODE_SCRATCH`, `$PBS_LOCALDIR`, `$SLURM_TMPDIR`,
  `$TMPDIR`, `/tmp`), as toolchain copies already do;
- a training config without an `eval` section does not evaluate.

This is the part of RFC 0008's site profiles that platoon needs now; RFC 0008's M5 grows the same file.

### 5. Input paths are relative to the file that names them

- **Inputs.** A relative input path in a config file names a file relative to that file's directory, as `extends:`
  does already. Inputs are what platoon reads:
  - a harness `.toml` path;
  - the options a service declares as paths in a class attribute (`path_options`). For Harbor: a local `dataset`,
    `dataset_dir`, `task_dirs`, `splits`, `lock`, `tasks`, `overlays`, `mounts[].source`. For the table service:
    `path`.

  The loader makes them absolute per file, before files merge. Paths inside a dataset (`index[].path`) stay
  relative to the dataset, and so does a split manifest named relative to a `dataset_dir` that the same file names
  (decided at M5: the service has always read it from the dataset directory).
- **Overrides and outputs.** Command-line overrides resolve against the working directory. Outputs (`log_path`,
  `inference.output_dir`) resolve against the working directory, a leading `~` against the home directory, and
  default to the site's `runs`.
- **The record.** `config.resolved.yaml` records absolute paths, so a run directory reloads from anywhere.
- **No fallback.** A relative input that does not exist against its file is an error. When the path exists against
  the working directory, the error names that path too. A `lock` that a fetch will write need not exist yet: it is
  refused only when the path names a file relative to the working directory (decided at M5).
- **No deprecation window.** The change of meaning comes with the error above rather than a release of warnings
  (section 9).

### 6. A run that looks after itself

```yaml
watchdog: {timeout_seconds: 7200, max_restarts: 20}
checkpoint: {every: 50, recovery_every: 5, keep_recovery: 2}
```

**Restarts.** With `watchdog.max_restarts` above 0 (default 0), `platoon train` runs the trainer as a child process.
- It starts the trainer again after any exit but these:
  - 0;
  - a configuration error (78);
  - an interrupt (130, or 143 from the scheduler's SIGTERM).
- It starts it at most `max_restarts` times, and gives up after three consecutive starts that saved no new
  checkpoint.
- Each start resumes from the newest checkpoint. The run's leftover rollout workers and sandboxes are reaped first.
- So a stalled training service, which ends with a deadline error and exit 1, restarts as the watchdog's exit does.
- `python -m platoon.train.tinker.restart_wrapper` is deprecated: launchers written before this call it, so it keeps
  working, with a warning, until 0.3.0 removes it.

**No delay key.** A start retries opening its training client until the service accepts it, up to 20 minutes, and
logs each refusal. A fixed delay would be sized to one service's session lease.

**Recovery checkpoints.** With `checkpoint.keep_recovery: N` (default: keep every one), platoon deletes the run's
older `recovery-*` checkpoints beyond the newest N, once a recovery save is recorded in `checkpoints.jsonl`. It
deletes them through the Tinker API (`RestClient.delete_checkpoint`).
- **Never deleted:** a permanent checkpoint, the resume point, or a checkpoint within `max_staleness` versions of
  the policy. `keep_recovery × recovery_every` must exceed `max_staleness`, which the config's loading checks.
- **Failed deletes** are counted (`checkpoint/delete_failures`); after three in a row, pruning stops with a warning.

**Sandboxes.** `platoon train` reaps the sandboxes labelled with its run (`platoon.run`, exact match) when it starts,
before a restart and when it exits, and `platoon eval` those labelled with its output directory when it exits; each
logs how many it found. `platoon sandbox reap RUN_DIR` does the same for a run that was killed, with the providers
its `config.resolved.yaml` records. The command that runs a run records itself in its directory (`process.json`:
host, pid, start time, and the start it supervises). `sandbox reap` refuses while a process it names is alive, and
when it names another host; a `platoon train` or `platoon eval` that finds a live one on its host stops with exit 78.

**Rollout deadlines.** A rollout that hangs costs its task's own time, not a day. Without `rollout_config.timeout`,
a rollout worker's alarm and an evaluation slot's hard timeout count the harness's own time limit: the earliest of
its spec's `timeout_s` and the task's `[agent] timeout_sec` times `agent_timeout_multiplier`, which an environment
service gives through an optional `agent_timeout_s(task_id)`. They add what the task needs beyond the harness
(`rollout_overhead_s`), as they do to a `timeout`. A Harbor config then needs no `timeout` to bound its workers, and
keeps one only to stop harnesses sooner than their tasks allow. A rollout that nothing limits, that a rollout
function or an episode program of the environment's own runs, or whose environment does not say what it needs beyond
the harness, keeps the fallback: a day for a worker, 900 s for an evaluation slot.

### 7. One run per run directory

**The record.** The first start of a run writes `identity.json` into its run directory:
- the model and the renderer;
- for each entry with a hold-out: the rule, the dataset, every held-out task (refused ones included, so that a
  changed refusal moves none) and the evaluation's task ids.

**The check.** A start that resumes (its directory holds a checkpoint) compares the record, after the tasks are
listed and before Tinker is opened. It stops with exit 78, naming the entry, the counts and up to three ids, for:
- another model or renderer;
- a held-out task that would now train;
- a task now held out that was not, a hold-out added to a running run included.

Held-out tasks that only left the selection are logged, and so are those a changed refusal took out of the
evaluation or put back into it; such a resume goes on. The start's `lock.json` records a refusal. RFC 0008's
`run.json` can absorb the file later.

**The warning.** When training evaluates, platoon warns once per entry whose evaluation lists training tasks: its
numbers then measure tasks training sees.

### 8. Smaller changes

1. **Entries by name.** An override path segment that is not an integer names the entry with that `name`, as in
   `environments.science.harness=platoon-oracle`. Integers index as today. A name that no entry has is an error
   that lists the names.
2. **Harnesses that make no model call.** A harness declares that it never calls the policy; `platoon-oracle` and
   `platoon-noop` do. `platoon eval` of a config whose harnesses all declare it needs no model, renderer or Tinker
   service, and the gateway refuses any call that arrives anyway.
3. **An evaluation's own directory.** `platoon eval` of a config without `inference.output_dir` writes each start to a
   new directory, `<runs>/eval/<policy>/<UTC time>`: `<policy>` is the checkpoint, else the model, and `<runs>` is the
   site profile's `runs` (section 4), else `runs` under the working directory. A start never resumes another policy's
   rollouts: each rollout records the policy that made it, and a directory whose rollouts record another model,
   checkpoint, renderer, hosted base URL or judges setting stops the start with exit 78 before anything in it changes.
   It stops rather than start afresh elsewhere, because a directory the config names is where the user will look for the
   results, as a training resume that would change the run stops (section 7).
4. **Every task trains.** The motivation's mix skipped 255 tasks and repeated 257 every epoch because a weighted
   order was drawn once and repeated: an entry with fewer slots than tasks trained on the same subset every epoch,
   and a single environment's last partial batch was left out of every epoch. Each epoch now goes on with the order
   where the one before it stopped, and the tasks a single environment's epoch leaves over start the next one, so a
   resumed run still draws what the run it resumes would have. One dataset needs no weights (section 1); a mixture
   of datasets draws every task equally often with `weight: tasks`, which gives each entry the share of its training
   tasks, counted once a hold-out, the selection and `exclude_tasks` have taken theirs. It is the default: an equal
   share per entry, the default before, drew a small dataset's tasks far more often than a large one's, and changing
   a default after the first release would change what configs train. A mixture's weights are all numbers or all
   `tasks`. `platoon train` warns when the weights leave an entry's tasks out of the run, or draw one entry's tasks
   more than 4 times as often as another's, and `platoon config validate --tasks` lists the tasks and shows the same
   before a run.

### 9. What changes for users

No platoon release has shipped; PyPI holds only the 0.0.0 placeholder. So, as RFC 0016 did, no change of behavior gets
a deprecation window, and a renamed key keeps a deprecated alias for one minor release (RFC 0016, section 10). The
CHANGELOG lists each with its migration.

**Changed:**
- Evaluation inherits training's rollout and worker settings.
- `eval.every` and `checkpoint.every` count training steps, and default to 0, which turns the event off.
- `platoon harbor fetch` does nothing when current, and fetches a lock written before input files were recorded
  once more.
- A missing image stops a run at start.
- Without a rollout `timeout`, a rollout's deadline counts its harness's and its task's own time limits, not a day (a
  rollout worker) or 900 s (an evaluation slot).
- A resume with another model, or with moved held-out tasks, stops.
- `platoon eval` without `inference.output_dir` writes each start to a new directory, and an evaluation never resumes
  another policy's rollouts.
- `log_path` defaults to the site's `runs`, and there `stats.experiment_name` defaults to the config's file name.
- Relative inputs resolve against their file.
- With one primary name, platoon's own zeros record that name.
- `refuse`'s switches take booleans only.
- A group whose rollouts trained different reward keys does not train.
- Each epoch of a weighted order goes on where the one before it stopped, instead of repeating the first, and a single
  environment's last partial batch starts the next epoch instead of being left out of every one.
- A mixture whose entries set no weight draws every task equally often (`weight: tasks`), where each entry had an
  equal share; `weight: 1` on every entry keeps the equal shares.

**Renamed,** each old key a deprecated alias until 0.3.0:
- `train.label_by` replaces `train.rollout_group_key` and `workflow_config.straggler_policy_key`;
- `every` alone replaces `strategy` with `every` in `eval` and `checkpoint` (a config that still sets `strategy` keeps
  its meaning);
- `max_concurrent_rollouts` replaces `num_concurrent_rollout_workflow_workers` in `train` and `eval`, counting
  rollouts: the old value times the section's `workflow_config.group_size`.

**Removed:**
- `restart_wrapper`, in 0.3.0 (deprecated until then);
- fetch config files (deprecated at M4, removed in 0.3.0);
- entries' `cache`, `image_map` and `toolchain_root`, and `PLATOON_HARBOR_CACHE` and `PLATOON_TOOLCHAINS`, replaced by
  the site profile (deprecated at M5, removed in 0.3.0).

**New:**
- keys: a list `primary`, `holdout` (with `ignore_inputs`), `{metadata}` index entries, `refuse.uploads_over_mb`,
  `eval.at_start`, `watchdog.max_restarts`, `checkpoint.keep_recovery`, and `weight: tasks`;
- each lock row's `input_files`, by which a hold-out joins the groups that share data;
- Harbor index columns, and metrics by the label column;
- commands: `platoon prepare`, with `--prepare` on `train`, `eval` and `rollout`, `platoon sandbox reap`, and
  `platoon config validate --tasks`;
- the site profile and `PLATOON_SITE`;
- entries by name, and harnesses without model calls;
- `eval/step`, the step an evaluation during training evaluated, and `evaluations.jsonl`, the run directory's record of
  the evaluations that completed, which a resume reads.

**Unchanged:** `splits` manifests, `lock`, `select`, `overlays`, and every other entry key.

## The hard parts

- **A held-out split must not move under a run.** The walk depends on the stratum's other groups, so changing the
  selection can move groups at the boundary. A run's `identity.json` makes a resume with moved tasks refuse; a new
  run gets the new split.
- **A group must be the unit that shares inputs.** On the science dataset that is the Kaggle dataset, not the question or
  its notebook, and even Kaggle names miss the tables uploaded under several names. A wrong unit makes the
  held-out curve measure seen data for the run's whole life. So platoon joins the groups whose tasks share an input
  file itself, and the report shows each split's makeup before a run starts.
- **Shared files that are not data.** A file that every task holds by design, such as a game's rules, joins every
  group. Platoon cannot tell it from a table that every question asks about, so it stops with an error that names
  the file, and the config names it in `ignore_inputs`.
- **Filling a share favors small groups and can concentrate.** A group larger than what remains is passed over.
  Within the share, one large family can still dominate a source's held-out set: 79-82% for two seeds on
  MathConstructOptimize. The report and the seed decide; a cap would bias further against large groups.
- **Reward alternatives and integrity.** Platoon never invents a key, and each reading reports exactly one listed key
  from one file. A group trains one key. The list's order never matters.
- **A site file outside the repository.** It could make runs irreproducible or unsafe. So it holds only where things
  live and which isolating runtime starts sandboxes, and runs record what it gave them.
- **Deleting checkpoints.** It is opt-in. A recovery checkpoint is deleted only after a newer one is recorded, never
  the resume point, never one a sampling client may still use, and only checkpoints the run's own
  `checkpoints.jsonl` lists.
- **Restarting after any failure.** A deterministic crash would loop, so restarts stop after three starts without a
  new checkpoint, and a configuration error never restarts.
- **Path resolution changes meaning.** A path that resolved against the working directory may now resolve against a
  config file's directory. The error names both, so nothing changes silently.
- **Index columns are the dataset's.** Their names and values come from data. A column that collides with platoon's
  own is left out, and is an error when named. High-cardinality columns log only their most frequent values.
- **Remote and hosted providers.** The lock, the cache and the toolchains stay on the trainer's side, as today.
  `platoon prepare` builds images only for providers that run local image files or a local daemon, and names the
  build-and-push command for the others.

## Alternatives

- **The first reported listed key trains** (this RFC's first draft). A reading that reported two listed keys, or a
  second reward file, would let the order and the file decide. That is a choice a policy whose code the tests run
  could make.
- **A per-task primary named by an index field** (`primary_from: reward_key`). Needs a field no dataset has.
- **An independent hash per group for the hold-out.** Stable as a selection grows, but it holds no share when group
  sizes vary by 60×: 1 to 199 MathConstructOptimize tasks over five seeds.
- **Walking after refusals.** It hits the runnable share exactly, but every refusal change moves groups at the
  boundary.
- **Grouping SmolData by notebook.** Notebooks on one Kaggle dataset share its files: 91-97% of the held-out tasks
  would share data with training.
- **Grouping by Kaggle dataset name alone,** this RFC's first draft. 2.5-46% of the held-out SmolData tasks share a
  byte-identical file with training, by the seed (29% for seed 0).
- **Comparing input files byte for byte only.** The same tables with other line endings or a byte-order mark escape
  it: 24% of seed 0's held-out SmolData tasks and 37% of seed 4's.
- **Joining by shared input files as an option.** A hold-out that leaks is never wanted, so it is the default, and
  no key turns it off.
- **Treating a file that every task holds as the environment, automatically.** It guesses: a dataset whose questions
  all ask about one table would hold out questions on seen data, silently. `ignore_inputs` names such files instead.
- **Joining by what a session uploads, or by every file of `environment/`.** The first depends on how the image is
  built, the second joins tasks by files no sandbox holds (section 1.3.1).
- **Input digests computed at every listing, or kept in the content-addressed cache.** Hashing SmolData's inputs
  (106 GiB over its tasks, 27.6 GiB of distinct content) at every start takes minutes; per-task files in the cache
  are a second record to keep in step with the lock, read one by one at every listing.
- **Splits from a dataset column** (`split`). Right for a dataset that publishes one; the science dataset publishes groups.
- **A shared `model:` section** for `train` and `inference`. A new section; step-0 evaluation (`eval.at_start`) and,
  later, `platoon eval TRAIN_CONFIG` remove the second file instead.
- **Train and eval build images and toolchains at start.** A build needs network, fakeroot or a daemon, and minutes,
  and writes shared site directories. So a run never builds unasked; it refuses with the command instead, and
  `--prepare` runs that command first where building inline is cheap (section 3).
- **One environment variable per site setting.** Invisible to configs and records (RFC 0008's motivation).
- **One project directory for every relative path** (Compose's rule). Breaks `extends:` across directories.
- **Gateway-side pruning.** Works for miles-tinker only, and needs the client's `checkpoints.jsonl` to know the resume
  point.
- **Restart only on the watchdog's exit code.** Misses the stalls a long run meets: a training service that stops
  answering ends with a deadline error, not the watchdog.
- **Deprecation windows for changes of behavior.** Nothing has been released. Renamed keys keep the aliases of
  RFC 0016, section 10, which cost nothing to keep for one minor release.

## Rollout plan

| Milestone | Scope | Exit check | Pull request |
|---|---|---|---|
| M1 one environment per dataset | list `primary`; metadata index fields; `uploads_over_mb`; `holdout`, joined by shared input files, and its report; `identity.json`; the evaluation-overlap warning; `platoon harbor fetch\|verify` on run configs; step-0 evaluation | one entry scores a `reward.txt` and a `reward.json {correctness, submission}` task; mutation and forgery checks pass under a list contract; the phase-0 lock gives section 1.3's split, with no held-out task sharing an input file or a Kaggle dataset with training for seeds 0-4; a resume with a changed `holdout.seed` exits 78; a lock written before this milestone still answers its config | the pull request that adds this RFC |
| M2 per-source metrics | index columns; metrics by the label column in training and evaluation; one key per group; key counts | a CPU run of a two-source fixture logs both sources' metrics every step and in evaluation; a mixed-key group does not train | the pull request that adds `train.label_by` |
| M3 evaluation inherits | section 2, point 2; `every` in steps and concurrency in rollouts (point 1) | a training config whose `eval` names only `every` evaluates with training's rollout and worker settings | the pull request that lays `eval.workflow_config` over `train.workflow_config` |
| M4 preparing a config | section 3; service options checked when a config loads (RFC 0016, section 4) | `platoon prepare` of the downstream project's config builds into an empty site, and a second run does nothing; a missing image exits 78 at start; `dataset_dir` entries without `image_map` run as before | the pull request that adds `platoon prepare` |
| M5 sites and paths | sections 4, 5 | the downstream project's configs name no site path and run from any directory with a cluster's profile; the profile refuses `process`; an explicit relative `log_path` keeps its base | the pull request that adds the site profile |
| M6 supervision | section 6 | a stalled fake service restarts the trainer, which resumes; older recovery checkpoints go, never within `max_staleness`; no labelled sandbox outlives a run; `sandbox reap` refuses a live run | the pull request that restarts training after failures |
| M7 the rest | `platoon eval` of training configs; entries by name; harnesses without model calls; lock-row reuse | per item | later |

## Open questions

1. **Renames before the first release:** `label_by`, `every` in steps, concurrency in rollouts. Decided
   (2026-10-17): all three, with deprecated aliases. M2 renamed `label_by`; M3 counts `every` in steps, with
   `strategy` deprecated, and concurrency in rollouts, as `max_concurrent_rollouts`.
2. **Which phase's sampling wins.** Should training default to `sampling_precedence: config`, so that a third-party
   harness that sends its own temperature cannot change the policy's sampling during training? Not needed for
   `openhands`, which sends none. Decide with the next harness that sends sampling parameters.
3. **A cap on one group's share of the held-out set,** if reports on other datasets show the concentration
   MathConstructOptimize shows for some seeds.
