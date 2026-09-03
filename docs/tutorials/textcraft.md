# Train on TextCraft

By the end of this page you will have a training run producing a `task_reward` curve on
TextCraft-Synth, know which metrics say it is working, and know where the rollouts landed so you
can read one.

TextCraft is the natural first real run. It ships its own tasks and recipes as files, needs no
external service, and it is the **only plugin that registers components** — so you can drive it
through the shared trainers instead of a per-plugin script.

## Before you start

| You have | What you can do here |
|---|---|
| An 8-GPU Linux node | Everything, on the AReaL tab. |
| A Tinker account, no GPUs | Everything, on the Tinker tab. Training happens remotely. |
| Neither | Steps 1-3. You get a baseline success rate and real rollouts to inspect, but no training. |

There is no single-GPU AReaL config in this repository — every AReaL config under `plugins/` sets
`cluster.n_gpus_per_node: 8`. If you can satisfy neither backend, stop after step 3; that part needs
nothing but an OpenAI-compatible endpoint.

You should already have `uv` and Python 3.12. See [installation](../get-started/installation.md) if
not, and [quickstart](../get-started/quickstart.md) if you have never run anything in this repo.

## The task

TextCraft-Synth is a procedurally generated crafting world: five material domains, thirteen tiers,
and item names with no meaning — `m2_i1_12`, `raw_a1`, `c5_ore`. The naming is the point. A model
cannot guess that `oak_log` makes `oak_planks` when the items are called `m9_ore` and `m2_i2_12`, so
it has to query the environment. Recipes reach depth 12, deep enough that one agent's step budget
becomes the binding constraint — which is what motivates the recursive variants.

The agent acts by writing Python. Four actions are always available (`craft`, `get_info`,
`view_inventory`, `finish`), plus `launch_subagent` in the recursive environments. All of them live
in <span class="pl-src">plugins/textcraft/platoon/textcraft/env.py</span>.

A shipped task looks like this:

```json title="plugins/textcraft/platoon/textcraft/textcraft_synth_train.jsonl (one row, reformatted)"
{"goal": "Craft the following items: 1x m2_i2_12",
 "id": "textcraft_synth.train.30",
 "max_steps": 75,
 "misc": {"target_items": {"m2_i2_12": 1},
          "initial_inventory": {"m9_ore": 2, "m3_ore": 1, "m8_ore": 3},
          "difficulty": "easy", "max_depth": 2, "num_craft_steps": 2}}
```

A good trajectory on it is short: look up the target, look up what the target needs, craft bottom
up, finish.

```python
# The model has never seen these names, so it must ask.
get_info(["m2_i2_12"])
# -> [{"item": "m2_i2_12", "can_craft": True, "is_base": False, "in_inventory": 0,
#      "crafting_depth": 2,
#      "recipes": [{"ingredients": {"m2_i1_12": 1, "m3_ore": 1}, "result_count": 3}]}]

get_info(["m2_i1_12"])          # recurse one level

craft({"m9_ore": 1}, ("m2_i1_12", 2))                 # 1 craft, yields 2
craft({"m2_i1_12": 1, "m3_ore": 1}, ("m2_i2_12", 3))  # 1 craft, yields 3

finish("Crafted 3x m2_i2_12")
```

Two rules trip up untrained models, and both show up as `craft` errors in the logs:

- **Counts must divide.** `craft` requires `target_count % recipe.result_count == 0`, and the
  ingredient amounts must equal `ingredients_per_craft * num_crafts` **exactly** — extras are
  rejected, not ignored.
- **Reward requires crafting, not owning.** `TextCraftEnv.evaluate` scores `1.0` only if the agent
  called `finish()` *and* `final_inventory[item] - initial_inventory[item] >= required` for every
  target. Items already in the starting inventory earn nothing.

That is the whole reward: binary, one number per episode, emitted as `reward/success`.

`misc["gold_trajectory"]` is a solvability certificate written by the dataset generator, not a list
of actions — its `target` field counts *crafts* while `craft`'s counts *items*. Do not feed it to
the environment.

## 1. Install

```bash
cd plugins/textcraft
uv sync --extra areal        # or: uv sync --extra tinker
```

The two extras are declared mutually exclusive; pick one. Plain `uv sync` is enough for step 3.

No data generation step is needed. `textcraft_synth_train.jsonl` (2 522 tasks) and
`textcraft_synth_val.jsonl` (632 tasks) are committed, and the recipe database is regenerated in
memory from `seed=42, items_per_domain_tier=25` by `SynthRecipeLoader`. The `synth_recipes/`
directory beside them is an inspection dump that nothing loads.

!!! warning "2 522 and 632 are not round numbers, and callers have to know them"
    `get_synth_task_ids` defaults to 10 000 / 1 000, which produces ids past the end of the shipped
    files. Every caller in the repository passes `num_samples_train=2522, num_samples_val=632`
    instead. The registry's dataset loader defaults to exactly those values — one reason to prefer
    it.

## 2. Check the environment loads

Before spending GPU hours, confirm the recipe database and the dataset agree with each other.

```bash
uv run python -c "
from platoon.textcraft.synth_tasks import get_synth_task
from platoon.textcraft.env import create_synth_env
task = get_synth_task('textcraft_synth.train.30')
env = create_synth_env(task)
print(task.goal, '| difficulty', task.misc['difficulty'], '| depth', task.misc['max_depth'])
print(env.code_executor.get_info(list(task.misc['target_items'])))
"
```

If that prints a recipe, you are wired up. The two only agree because both regenerate from the same
seed and `items_per_domain_tier` — if you ever regenerate the dataset with a different value, pass
the same one to `create_synth_env`.

## 3. Get a baseline (no GPU, no training)

Run the inference benchmark against any OpenAI-compatible endpoint. This is the number training has
to beat, and it produces rollouts you can read.

```bash
uv run python platoon/textcraft/inference_scripts/run_synth_inference.py \
  --config platoon/textcraft/configs/inference/textcraft_synth_inference.yaml \
  --inference.model_name openai/Qwen/Qwen3-4B-Instruct-2507 \
  --inference.model_endpoint http://127.0.0.1:30000/v1 \
  --inference.output_dir ./inference_results/textcraft_baseline \
  --difficulty easy,medium \
  --num_tasks 20
```

Overrides on this path are `--dotted.key value`. The script prints the `summary` block of
`final_report.json` — `success_rate`, `success_at_k`, `reward_mean`. The shipped config sets
`use_recursive_agent: false`, so this is the flat agent. Full detail in
[evaluate a model endpoint](inference.md).

!!! warning "Pass two difficulties, or leave the key alone"
    `--difficulty medium` on its own is coerced to the string `"medium"`, and the branch handling a
    scalar difficulty in `get_dataset_task_ids` references an undefined name. A comma makes it a
    list. The shipped `difficulty: ['easy', 'medium', 'hard']` also works. What does not work is
    `difficulty: null`, which returns an empty task list.

## 4. Point the config at the registry

<span class="pl-src">plugins/textcraft/platoon/textcraft/registry.py</span> registers everything the
shared trainers need, under short names:

| Kind | Registered name | What it is |
|---|---|---|
| `dataset_loader` | `textcraft/synth` | Task ids, filtered by `difficulties`, capped by `limit` |
| `task_loader` | `textcraft/synth` | `get_synth_task` |
| `rollout` | `textcraft/synth/linear` | Flat agent, no delegation |
| `rollout` | `textcraft/synth/recursive` | `launch_subagent`; child steps spend the parent's budget |
| `rollout` | `textcraft/synth/depth_aware` | `launch_subagent`; each agent gets its own budget, tree depth capped |
| `reward_processor` | `textcraft/synth/delegation_capped` | Sums `reward/*` across steps, returns `reward/success` |

A config selects them through a top-level `environments:` list of `EnvironmentConfig` — registry
wiring, exactly one entry. (It is unrelated to the nested plugin-local `environments:` mixture list
inside some openreward configs.) See [the registry](../architecture/registry.md).

=== "AReaL"

    Start from `configs/areal/textcraft_synth_ctx8192_linear_medium_areal.yaml` and add this block
    at the top level:

    ```yaml
    environments:
      - package: platoon.textcraft.registry
        dataset_loader: textcraft/synth
        eval_dataset_loader: textcraft/synth
        task_loader: textcraft/synth
        rollout: textcraft/synth/linear
        reward_processor: textcraft/synth/delegation_capped
        workflow: group_rollout
        dataset_kwargs:
          difficulties: ["medium"]
        eval_dataset_kwargs:
          difficulties: null
          limit: 100
    ```

    !!! danger "Then delete `train_difficulties`, `eval_difficulties` and `recursive`"
        Those three keys belong to `TextCraftSynthArealTrainerConfig`, which only the plugin's own
        script uses. The shared entrypoint hard-codes `PlatoonArealRLTrainerConfig`, and AReaL's
        `load_expr_config` merges your YAML onto that dataclass in struct mode — an unknown
        top-level key raises `ConfigKeyError: Key 'train_difficulties' not in
        'PlatoonArealRLTrainerConfig'` before anything starts. The `environments` block replaces
        them: `dataset_kwargs` carries the difficulty filter, `rollout` carries what `recursive`
        used to select.

    `configs/areal/textcraft_synth_ctx40000_depth_aware_medium_areal.yaml` ships the equivalent
    block commented out, for the depth-aware rollout.

=== "Tinker"

    `configs/tinker/textcraft_synth_depth_aware_tinker.yaml` already has the block. It is the one
    config in the repository wired for the registry out of the box:

    ```yaml title="plugins/textcraft/platoon/textcraft/configs/tinker/textcraft_synth_depth_aware_tinker.yaml"
    environments:
      - package: platoon.textcraft.registry
        trainer_config: textcraft/synth/tinker
        dataset_loader: textcraft/synth
        eval_dataset_loader: textcraft/synth
        task_loader: textcraft/synth
        rollout: textcraft/synth/depth_aware
        reward_processor: textcraft/synth/delegation_capped
        workflow: group_rollout
        dataset_kwargs:
          difficulties: ["medium"]
          num_samples_train: 2522
          num_samples_val: 632
        eval_dataset_kwargs:
          difficulties: null
          limit: 100
          num_samples_train: 2522
          num_samples_val: 632
    ```

    For a cheaper first run, change `rollout` to `textcraft/synth/linear` — a flat agent with no
    delegation tree. `trainer_config` is inert: it is registerable and settable, but no code reads
    it.

`package: platoon.textcraft.registry` is what makes the names resolvable. `AutoEnvironment.load`
imports that module, and importing it runs the `register_*` calls. TextCraft also declares a
`platoon.plugins` entry point, so `discover_entry_points: true` finds it without naming the package.

## 5. Change the account-specific values

=== "AReaL"

    | Key | Shipped value | What you need |
    |---|---|---|
    | `cluster.fileroot` | `/mnt/efs/tmp/areal/experiments` | A directory writable from every node. `rollout`, `saver`, `recover`, `evaluator` and `stats_logger` all inherit it. |
    | `cluster.name_resolve.nfs_record_root` | `/mnt/efs/tmp/areal/name_resolve` | A directory visible to every node — file-based service discovery. |
    | `workflow_config.rollout_config.output_dir` | an absolute path containing the old trial name | Yours. This one does **not** follow `trial_name`, so override both or new rollouts land in the previous run's directory. |
    | `actor.path` | `Qwen/Qwen3-4B-Instruct-2507` | Keep it if the nodes can reach Hugging Face. `tokenizer_path` and `sglang.model_path` follow it. |
    | `trial_name` | `...-linear-medium-trial-0` | A fresh name per run. |
    | `stats_logger.wandb.project` | `recursive-agents` | Your project — or set `stats_logger.wandb.mode=disabled`. |

    !!! warning "W&B fails the run late, not early"
        `mode` is `online`, and AReaL calls `wandb.login()` during trainer construction — after
        costly worker startup has already begun. For a first run pass
        `stats_logger.wandb.mode=disabled`.

=== "Tinker"

    | Key | Shipped value | What you need |
    |---|---|---|
    | `train.model_name` | `Qwen/Qwen3-4B-Instruct-2507` | A model your Tinker account can train. |
    | `train.renderer_name` | `qwen3_instruct` | Must match the model family. |
    | `stats.trial_name` | `textcraft-synth-depth-aware-...-trial-0` | A fresh name per run. `log_path/experiment_name/trial_name` is the run directory. |
    | `stats.wandb.project` | `recursive-agents` | Your project, or set `stats.wandb.mode` to `disabled`. |
    | `tinker_base_url` | `null` | Only for a non-default service URL. |

    `log_path` is already relative (`./logs/runs`), so nothing else needs touching. Platoon
    constructs `tinker.ServiceClient(base_url=tinker_base_url)` and passes no credential —
    authentication is whatever the `tinker` SDK reads from your environment.

## 6. Run it

=== "AReaL"

    ```bash
    cd plugins/textcraft
    uv run python3 platoon/textcraft/train_scripts/areal/train_areal_synth.py \
      --config platoon/textcraft/configs/areal/textcraft_synth_ctx8192_linear_medium_areal.yaml \
      cluster.fileroot=/scratch/$USER/areal \
      cluster.name_resolve.nfs_record_root=/scratch/$USER/areal/name_resolve \
      workflow_config.rollout_config.output_dir=/scratch/$USER/areal/rollouts/textcraft \
      stats_logger.wandb.mode=disabled \
      trial_name=textcraft-smoke-1
    ```

    Overrides are bare `key=value` — **no leading dashes**. `scheduler.type` defaults to `local` in
    `PlatoonArealRLTrainerConfig`, so you do not need to pass it.

    This is the per-plugin script, not the shared `python -m platoon.train.areal.train`. None of
    the checked-in TextCraft AReaL configs carry a top-level `environments:` block, so the shared
    entrypoint has nothing to resolve components from and stops at
    `Config must set environments[0].dataset_loader`. The Tinker tab is the one that runs on the
    registry today.

    Startup is minutes, not seconds. The SGLang rollout engine (`rollout.backend: sglang:d6p1t1`)
    and the FSDP actor with its colocated reference model (`actor.backend: fsdp:d2p1t1`) come up
    first, and `rollout.setup_timeout` is `900`. The fully resolved config is written to
    `config.yaml` under the stats-logger log directory — read that file if an override did not land.

=== "Tinker"

    ```bash
    cd plugins/textcraft
    uv run python -m platoon.train.tinker.train \
      --config platoon/textcraft/configs/tinker/textcraft_synth_depth_aware_tinker.yaml \
      --stats.trial_name textcraft-smoke-1 \
      --stats.wandb.mode disabled
    ```

    Overrides are `--dotted.key value` — the opposite of the AReaL tab. Unknown YAML keys are
    ignored rather than fatal here: `load_config` reads only the fields the dataclass declares.

    There are no local workers to start, so the first rollouts begin almost immediately.

Getting the two override syntaxes backwards is the most common first-run mistake. Both loaders are
described in [configuration](../reference/configuration.md).

## 7. What to watch

Both backends run a `GroupRolloutWorkflow` and record the first five rows. The last two are AReaL
only.

| Metric | Why you care |
|---|---|
| `task_reward` | The root trajectory's reward for one rollout: `0.0` or `1.0`. Its mean over a batch is your success rate, and it is the whole signal. |
| `task_reward_at_k_mean` / `_max` / `_min` | Aggregated over a task's `group_size: 8` rollouts. `_max` rising while `_mean` is flat means the policy can solve the task but not reliably — normal early progress. |
| `reward/success` | Per trajectory rather than per rollout, so on the recursive rollouts it includes subagents. Divergence from `task_reward` says the children are doing better or worse than the root. |
| `num_steps` | Should trend down as the model stops flailing on rejected `craft` calls. |
| `num_output_tokens`, `avg_output_tokens_per_step` | Runaway generation is the usual cause of a stalled run. |
| `group_size_effective`, `group_size_rejected` | How many of the 8 requested rollouts survived. Persistent rejection means rollouts are erroring, not merely scoring zero. |
| `zero_variance_reward_group` | Groups where every member got the same reward. They carry no advantage signal. High early is expected; high late is not. |

The recursive and depth-aware rollouts add `reward/subagent_launched` and
`reward/subagent_succeeded`. They are reported but do **not** move the score: the delegation bonus
is multiplied by `_TEXTCRAFT_SYNTH_DELEGATION_REWARD_CAP`, which is `0.0` everywhere in the
repository today. Delegation is rewarded only through whether it helps the root succeed.

Medium difficulty on a 4B instruct model starts low. `task_reward` flat at zero for the first few
steps is the task, not a bug. Flat at zero *with* `group_size_rejected` climbing is a bug — read the
rollout events before touching hyperparameters, and see
[troubleshooting](../reference/troubleshooting.md).

## 8. Where things landed

=== "AReaL"

    | What | Path |
    |---|---|
    | Logs and the resolved `config.yaml` | `{cluster.fileroot}/logs/<user>/{experiment_name}/{trial_name}/` |
    | Checkpoints | `{cluster.fileroot}/checkpoints/<user>/{experiment_name}/{trial_name}/` |
    | Training rollout events | `{rollout_config.output_dir}/train_rollout/{engine_version}/events/` |
    | Eval rollout events | `{rollout_config.output_dir}/eval_rollout/{engine_version}/events/` |

    The first two rows are AReaL's own convention. The numeric engine-version directory is appended
    by the workflow so rollouts generated after a weight update do not overwrite earlier ones.
    `saver.freq_steps: 25` and `evaluator.freq_steps: 5` control how often the checkpoints and eval
    rollouts appear.

=== "Tinker"

    Everything lives under `{log_path}/{experiment_name}/{trial_name}`:

    ```text
    logs/runs/recursive-agents/textcraft-smoke-1/
    ├── checkpoints.jsonl
    └── rollouts/
        ├── train/{checkpoint_version}/events/events_<task-id>_<collection-uuid>.jsonl
        └── eval/{checkpoint_version}/events/...
    ```

    The workflow overwrites `rollout_config.output_dir` with that path, so the `./rollout_results`
    and `./eval_results` values in the YAML never take effect. Weights live on the Tinker service;
    `checkpoints.jsonl` records one line per checkpoint with the paths to fetch them by.

Each event file is one `TrajectoryCollection` — the root trajectory plus every subagent's — written
incrementally by a `JsonlFileSink` as the episode runs.

## 9. Read a rollout

```bash
uv run python -m platoon.visualization.cli tail --rdir ./logs/runs
```

`tail` follows a running job; `replay --dir <events-dir> --delay 0.25` walks a finished one step by
step. Open a failing episode and find the step where `craft` returned an error — for a model that
has not learned the divisibility rule, that is most of them. The [TUI tutorial](visualization.md)
covers navigation, subagent trees and run-vs-run diffs.

## Variation: change the curriculum from config alone

`dataset_kwargs` is forwarded straight into the registered dataset loader, so you can change what a
run trains on without touching Python. Easy tasks only:

```yaml
    dataset_kwargs:
      difficulties: ["easy"]
      limit: 500
```

Difficulty is a depth band fixed when the dataset was generated: easy is depth 2-3, medium 4-6, hard
7-9, extreme 10-12. The shipped train split holds 588 easy, 852 medium, 544 hard and 538 extreme
tasks; the val split 147 / 213 / 136 / 136. Several difficulties concatenate:

```yaml
    dataset_kwargs:
      difficulties: ["easy", "medium"]
```

An unknown name fails loudly with the valid options listed. `limit` is applied *after* filtering and
the lists are concatenated rather than interleaved, so `["easy", "medium"]` with `limit: 500` gives
you 500 easy tasks and no medium ones. `eval_dataset_kwargs` is separate — the shipped configs leave
`difficulties: null` there so evaluation covers the whole spread while training stays narrow.

This is worth doing as your second run. Comparing an easy-only run against the medium run above is
the cheapest way to learn whether your model is limited by reasoning depth or by the action format.
[Curriculum recipes](../recipes/curriculum.md) covers changing the mix during training.

!!! note "Filtering is a linear scan"
    `get_synth_task_ids_by_difficulty` loads every task in the split to read its difficulty tag. On
    2 522 rows that is a one-time cost of a second or two, not a problem — but there is no index.

## The other route: per-plugin train scripts

The registry is newer than the scripts, and every other plugin still uses a script. TextCraft ships
six under `train_scripts/`, doing the same wiring in Python that `environments:` does in YAML:

```bash
cd plugins/textcraft

# AReaL: one script; the rollout is chosen by the config's recursive / depth_aware flags
uv run python3 platoon/textcraft/train_scripts/areal/train_areal_synth.py \
  --config platoon/textcraft/configs/areal/textcraft_synth_ctx8192_linear_medium_areal.yaml

# Tinker: one script per rollout style
uv run python -m platoon.textcraft.train_scripts.tinker.train_tinker_synth_depth_aware \
  --config platoon/textcraft/configs/tinker/textcraft_synth_depth_aware_tinker.yaml
```

Take this route if you want to modify the training loop itself, or if you are handed a config that
still carries `train_difficulties` / `depth_aware`. Two traps:

- **Always pass `--config`.** `train_tinker.py`, `train_tinker_synth.py` and
  `train_tinker_synth_recursive.py` compute a default config path next to the script, but the
  configs live a directory up in `configs/tinker/`, so the default resolves to a file that does not
  exist and the run dies in `load_yaml_config` with `FileNotFoundError`. Only
  `train_tinker_synth_depth_aware.py` gets the relative path right. Their docstrings also name a
  stale module path (`platoon.textcraft.train_tinker` rather than
  `platoon.textcraft.train_scripts.tinker.train_tinker`).
- **The Tinker synth scripts ignore your YAML for difficulty.**
  `train_tinker_synth_depth_aware.py` hard-codes `train_difficulties = ["medium"]` in Python. Only
  the registry route reads `dataset_kwargs.difficulties`.

`configs/areal/` also holds ready-made variants organized by context length, rollout style and
difficulty: `ctx4096` through `ctx40000`, `linear` / `recursive` / `depth_aware`, `medium` / `hard`.

## Rough edges worth knowing

- **`max_steps` is overwritten twice.** The task file says 75; the workflow replaces it with
  `workflow_config.rollout_config.max_steps` before every rollout; then
  `run_synth_depth_aware_rollout` replaces it again with its own `per_agent_max_steps=25`. The
  number in the YAML only matters for the linear and recursive rollouts.
- **The depth cap is a function default, not a config key.** `run_synth_depth_aware_rollout` uses
  `max_depth=6` from `_TEXTCRAFT_SYNTH_MAX_DEPTH`. The comment at the top of
  `textcraft_synth_depth_aware_tinker.yaml` says 12; the code says 6. Change it by editing the
  default in <span class="pl-src">plugins/textcraft/platoon/textcraft/synth_rollout.py</span>, not
  the YAML.
- **`textcraft_synth_multi_target_*.jsonl` are dead files.** Nothing references them; the current
  `DIFFICULTY_CONFIG` is single-target.
- **Subagent detection is a substring check.** `TextCraftEnv.evaluate` decides whether it is scoring
  a subagent with `"textcraft" not in task.id`. Root ids are `textcraft_synth.*`; forked ids are
  uuid4s. Any new id scheme that drops that substring will be misclassified.

## Next

- [Evaluate a model endpoint](inference.md) — the same rollout function with no trainer behind it,
  which is how you should be iterating on a task.
- [Inspect rollouts in the TUI](visualization.md) — the tool for finding the step where reward went wrong.
- [Train a system of agents](recursive-agents.md) — switch `rollout` to `textcraft/synth/depth_aware`
  and follow credit down the delegation tree.
- [Build a task from scratch](build-a-plugin.md) — the same pieces, in an empty directory.
- [The registry](../architecture/registry.md) — what `environments:` resolves, and how to register your own.
- [Training backends](../get-started/backends.md) — how AReaL and Tinker differ once you care.
- [Scale to multiple nodes](multi-node.md) — when one node stops being enough.
