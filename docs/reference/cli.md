# CLI reference

Every command line in the repository: training entrypoints, inference runners, the visualization
CLI, the offline analysis scripts, the dataset generators, and the development commands. Flags
listed here were read out of the `argparse` definitions in the source; if a flag is not on this
page, it does not exist.

Platoon declares **no console scripts**. There is no `platoon` binary on your `PATH` — everything
is `python -m <module>` or `python <path/to/script.py>`.

## Override syntax: the thing that bites first

Two config loaders live in this repository and they take overrides in incompatible forms. Getting
them backwards produces confusing failures rather than a clean error.

| Path | Loader | Config flag | Override form |
| --- | --- | --- | --- |
| AReaL training | `load_expr_config` (`areal.api.cli_args`) | `--config <yaml>` | `key=value`, **no dashes** |
| Tinker training | `load_config` (<span class="pl-src">platoon/utils/config.py</span>) | `--config` / `-c` | `--dotted.key value` or `--dotted.key=value` |
| Inference runners | `load_config` | `--config` / `-c` | `--dotted.key value` |
| Visualization, analysis, generators | plain `argparse` | n/a | ordinary flags |

```bash
cd plugins/number-search

# AReaL — OmegaConf style, bare key=value
uv run python3 platoon/number_search/train.py \
  --config platoon/number_search/nv_number_search_cispo_areal.yaml \
  trial_name=debug-run train_dataset.batch_size=16

# Tinker — argparse style, leading dashes required
uv run python -m platoon.number_search.train_tinker \
  --config platoon/number_search/number_search_tinker.yaml \
  --train.batch_size 64
```

The AReaL side also resolves `${...}` interpolation inside the YAML; the `load_config` side does
not. See [config loading](../architecture/config.md) for why there are two.

!!! warning "`load_config` coerces `0` and `1` to booleans"

    `_parse_value` in <span class="pl-src">platoon/utils/config.py</span> checks booleans before
    numbers: `true`/`yes`/`1` become `True` and `false`/`no`/`0` become `False`. So
    `--num_tasks 1` sets `num_tasks` to `True`, not `1`. Values containing a comma are split into
    a list, and a `--flag` with nothing after it becomes `True`. Use `2` or more where you meant a
    count of one, or set the value in the YAML. This affects Tinker training and every inference
    runner; the AReaL `key=value` path is unaffected.

## Where to run each command from

This is the most common first-run failure, and it comes from the repository layout: the root
project installs only the `platoon` core package, while each plugin under `plugins/<name>/` is its
own `uv` project whose venv contains both the plugin and `platoon` as an editable path dependency
(`[tool.uv.sources] platoon = { path = "../..", editable = true }`).

| Command family | Run from |
| --- | --- |
| Anything importing `platoon.<plugin>` — training, inference, dataset generators | `plugins/<name>/` |
| Shared registry entrypoints (`platoon.train.{areal,tinker}.train`) | `plugins/<name>/` — the plugin package must be importable |
| Visualization CLI, analysis scripts | either; a plugin directory is usually more convenient because output paths in configs are relative |
| `pytest`, `ruff`, `ty`, `pre-commit`, `mkdocs` | repo root |

`uv run` from a plugin directory picks up that plugin's venv, which is why the plugin examples
below all start with a `cd`.

## Training entrypoints

### The shared, registry-driven entrypoints

```bash
python -m platoon.train.tinker.train --config <yaml> [--dotted.key value ...]
python -m platoon.train.areal.train  --config <yaml> [key=value ...]
```

Both read a top-level `environments:` list of `EnvironmentConfig` and resolve every component
through the `Auto*` factories in <span class="pl-src">platoon/train/auto.py</span> — no per-plugin
Python needed. `run_tinker_training` and `run_areal_training`, in
<span class="pl-src">platoon/train/tinker/train.py</span> and
<span class="pl-src">platoon/train/areal/train.py</span>, take no arguments of their own;
everything comes from the config. Neither passes a `default_config_path` when run as `__main__`,
so `--config` is effectively required.

```bash
cd plugins/textcraft
uv run python -m platoon.train.tinker.train \
  --config platoon/textcraft/configs/tinker/textcraft_synth_depth_aware_tinker.yaml
```

!!! note "How far the registry path is actually wired"

    One plugin declares the `[project.entry-points."platoon.plugins"]` hook today —
    <span class="pl-src">plugins/textcraft/pyproject.toml</span>, pointing at
    `platoon.textcraft.registry`. One config in the tree carries a live top-level `environments:`
    block:
    <span class="pl-src">plugins/textcraft/platoon/textcraft/configs/tinker/textcraft_synth_depth_aware_tinker.yaml</span>.
    The AReaL counterpart,
    <span class="pl-src">plugins/textcraft/platoon/textcraft/configs/areal/textcraft_synth_ctx40000_depth_aware_medium_areal.yaml</span>,
    has its `environments:` block commented out, so `platoon.train.areal.train` has no
    ready-to-run config in the repository — you write one. Every other plugin still ships its own
    training script. See [the registry](../architecture/registry.md).

    This top-level `environments:` is component wiring. It is unrelated to the nested
    `openreward.environments:` mixture list in the openreward configs, which selects task sources
    — see [openreward](../integrations/openreward.md).

### Per-plugin training scripts

Each takes `--config <yaml>` plus overrides in its backend's syntax. AReaL scripts use
`load_expr_config`; Tinker scripts use `load_config`.

| Plugin | <span class="pl-tag pl-tag--areal">AReaL</span> module | <span class="pl-tag pl-tag--tinker">Tinker</span> module |
| --- | --- | --- |
| appworld | `platoon.appworld.train_scripts.areal.train_areal` | — |
| codegrep | `platoon.codegrep.train` | `platoon.codegrep.train_tinker` |
| deepdive | `platoon.deepdive.train_scripts.areal.train_areal` | `platoon.deepdive.train_scripts.tinker.train_tinker` |
| email-search | `platoon.email_search.train_scripts.areal.train_areal` | `platoon.email_search.train_scripts.tinker.train_tinker` |
| number-search | `platoon.number_search.train` | `platoon.number_search.train_tinker` |
| oolong | `platoon.oolong.train_scripts.areal.train_areal` | `platoon.oolong.train_scripts.tinker.train_tinker` |
| openreward | `platoon.openreward.train_scripts.areal.train_areal` | `platoon.openreward.train_scripts.tinker.train_tinker` |
| textcraft | `…train_scripts.areal.train_areal`, `…areal.train_areal_synth` | `…tinker.train_tinker`, `…tinker.train_tinker_synth`, `…tinker.train_tinker_synth_recursive`, `…tinker.train_tinker_synth_depth_aware` |

`number-search` and `codegrep` keep their scripts at the package top level; the rest use
`train_scripts/{areal,tinker}/`. The `openhands` plugin has no training script — it is a shared
agent implementation, not a task.

Every Tinker script passes a `default_config_path`; AReaL scripts do not, so on the AReaL side
`--config` is always required.

!!! warning "Three TextCraft Tinker defaults point at files that do not exist"
    `train_tinker.py`, `train_tinker_synth.py` and `train_tinker_synth_recursive.py` build their
    default path as `Path(__file__).parent / "<name>.yaml"` — beside the script, not under
    `configs/tinker/`. Run one without `--config` and `load_yaml_config` raises `FileNotFoundError`
    before anything else happens. Only `train_tinker_synth_depth_aware.py`, which uses
    `../../configs/tinker/...`, resolves to a real file. Every other plugin's Tinker default is
    correct, but passing `--config` explicitly costs nothing and never surprises you.

```bash
# AReaL, TextCraft-Synth — the rollout style is chosen by config flags, not by the script
cd plugins/textcraft
uv run python3 platoon/textcraft/train_scripts/areal/train_areal_synth.py \
  --config platoon/textcraft/configs/areal/textcraft_synth_ctx8192_depth_aware_medium_areal.yaml

# Tinker, one script per rollout style
uv run python -m platoon.textcraft.train_scripts.tinker.train_tinker_synth_recursive \
  --config platoon/textcraft/configs/tinker/textcraft_synth_recursive_tinker.yaml
```

Multi-node AReaL runs are launched from the shell scripts in `slurm-scripts/`, which `srun` the
same module inside an existing allocation — for example
<span class="pl-src">slurm-scripts/openreward-toolathlon-prealloc-base.sh</span> runs
`python -m platoon.openreward.train_scripts.areal.train_areal --config ${CONFIG} cluster.n_nodes=…`.
See [multi-node training](../tutorials/multi-node.md).

### Restart wrapper <span class="pl-tag pl-tag--tinker">Tinker</span>

```bash
python -m platoon.train.tinker.restart_wrapper [options] -- <command ...>
```

Runs a training command and restarts it when it exits with the watchdog exit code. Any other
non-zero exit is not retried.

| Flag | Type | Default | What it does |
| --- | --- | --- | --- |
| `--max-restarts` | int | `5` | Restarts before giving up |
| `--watchdog-exit-code` | int | `2` | The only exit code that triggers a restart |
| `--restart-delay` | float | `10.0` | Seconds to wait before restarting |
| `command` | `REMAINDER` | — | Everything after `--` |

```bash
cd plugins/textcraft
python -m platoon.train.tinker.restart_wrapper --max-restarts 5 \
  -- uv run python -m platoon.textcraft.train_scripts.tinker.train_tinker \
     --config platoon/textcraft/configs/tinker/textcraft_tinker.yaml
```

The `--config` in that inner command is not optional: `train_tinker.py` is one of the three scripts
whose built-in default points at a file that is not there.

`Ctrl-C` is forwarded to the child and the wrapper exits `130`. `run_with_restart` in
<span class="pl-src">platoon/train/tinker/restart_wrapper.py</span> is the same thing as a Python
API.

## Inference runners

Trainer-free benchmark harnesses. Each plugin ships one script; all of them parse their config
with `load_config`, so the flags are `--config` plus `--dotted.key value` overrides. There are no
other flags — everything else is a config key.

| Plugin | Module | Default config |
| --- | --- | --- |
| appworld | `platoon.appworld.inference_scripts.run_inference` | `configs/inference/appworld_inference.yaml` — **default does not resolve** |
| deepdive | `platoon.deepdive.inference_scripts.run_inference` | `configs/inference/deepdive_inference.yaml` |
| email-search | `platoon.email_search.inference_scripts.run_inference` | `configs/inference/email_search_inference.yaml` |
| oolong | `platoon.oolong.inference_scripts.run_inference` | `configs/inference/oolong_inference.yaml` |
| openreward | `platoon.openreward.inference_scripts.run_inference` | `configs/inference/toolathlon_openhands_inference.yaml` |
| textcraft | `platoon.textcraft.inference_scripts.run_inference` | `configs/inference/textcraft_inference.yaml` — **default does not resolve** |
| textcraft (synth) | `platoon.textcraft.inference_scripts.run_synth_inference` | `configs/inference/textcraft_synth_inference.yaml` |

`codegrep`, `number-search` and `openhands` have no inference script.

!!! warning "Two of those defaults point at a path that does not exist"
    The appworld and textcraft runners build their default as
    `Path(__file__).parent / "configs" / "inference" / ...`, which lands in
    `inference_scripts/configs/inference/`. The configs are one level up, in the package's own
    `configs/inference/`. Both scripts therefore raise `FileNotFoundError` when run without
    `--config`. The other four use `.parent.parent` (or `parents[1]`) and resolve correctly.
    The column above lists where each config actually lives, not where the script looks.

!!! warning "The README's AppWorld command is stale"

    The root `README.md` shows `python -m platoon.appworld.run_inference`. That module does not
    exist; the script lives under `inference_scripts/`.

Top-level config keys shared by every runner, all overridable from the command line:

| Key | Type | Default | What it does |
| --- | --- | --- | --- |
| `stage` | `full` \| `rollouts` \| `report` | `full` | `rollouts` collects only; `report` rescores what is already on disk without touching the endpoint |
| `num_tasks` | int | `100` (appworld: `None`) | How many tasks to draw |
| `task_id` | str \| None | `None` | Single-task quick path; overrides the dataset |
| `use_recursive_agent` | bool | `True` (oolong: `False`) | Recursive vs. flat rollout function |
| `dataset_split` | str | per plugin | Split name; the allowed values differ per plugin |
| `shuffle_tasks` | bool | `False` | Shuffle before truncating to `num_tasks` |
| `seed` | int | `42` | Seed for that shuffle |

Plugin-specific extras, read from the config dataclass in each script: email-search adds
`max_messages` (default `1`) and `exclude_known_bad_queries` (`True`); oolong adds
`oolong_dataset` (`synth`), `task_group`, `answer_type`, `min_context_len` and `max_context_len`;
the TextCraft-Synth script adds `difficulty` (a list, default `None`). The nested `inference:`
block is `InferenceBenchmarkConfig`, documented in [configuration](configuration.md).

```bash
cd plugins/textcraft
uv run python -m platoon.textcraft.inference_scripts.run_inference \
  --config platoon/textcraft/configs/inference/textcraft_inference.yaml \
  --inference.model_endpoint http://127.0.0.1:30000/v1 \
  --inference.output_dir ./inference_results/exp2

# Collect now, score later — the report stage is pure disk work and repeatable
… run_inference --config cfg.yaml --stage rollouts
… run_inference --config cfg.yaml --stage report
```

[The inference tutorial](../tutorials/inference.md) covers serving a model and reading the report.

## Visualization CLI

```bash
python -m platoon.visualization.cli {tail,replay,show-dump,analyze-compare,analyze-errors} ...
```

One `argparse` entrypoint with five subcommands, defined in
<span class="pl-src">platoon/visualization/cli.py</span>. A subcommand is required. All three
viewer subcommands accept `--mode` (`auto` | `codeact` | `openhands`, default `auto`) and
`--selectable-text` (off by default; disables mouse capture so terminal drag-selection works).

Rollout event logs land at `{rollout_config.output_dir}/events/events_<task>_<collection>.jsonl`.
For keybindings and how to read the tree, see
[the visualization tutorial](../tutorials/visualization.md).

### `tail`

Follow live event logs. Sources are concatenated in the order `--dir`, `--rdir`, then positionals;
if none resolve, the parser errors out.

| Flag | Type | Default | What it does |
| --- | --- | --- | --- |
| `paths` | positional, `nargs="*"` | — | JSONL files to tail |
| `--dir DIR` | str | `None` | Directory of `*.jsonl`, non-recursive |
| `--rdir DIR` | str | `None` | Directory root, recursive `rglob("*.jsonl")` |
| `--mode` | choice | `auto` | Step rendering mode |
| `--selectable-text` | flag | off | Release the mouse to the terminal |

```bash
uv run python -m platoon.visualization.cli tail --rdir ./rollout_results
```

`tail` does not seek to the end first — it replays whatever is already in the file, then follows.

### `replay`

Replay a finished log from the start. **There is no `--rdir` here**; only `tail` has one.

| Flag | Type | Default | What it does |
| --- | --- | --- | --- |
| `paths` | positional, `nargs="*"` | — | JSONL files to replay |
| `--dir DIR` | str | `None` | Directory of `*.jsonl`, non-recursive |
| `--delay SECONDS` | float | `0.5` | Seconds between events during autoplay; `0` loads everything instantly |
| `--mode` | choice | `auto` | |
| `--selectable-text` | flag | off | |

```bash
uv run python -m platoon.visualization.cli replay --dir ./rollout_results/events --delay 0.25
uv run python -m platoon.visualization.cli replay --mode openhands events.jsonl --delay 0
```

### `show-dump`

View a serialized `TrajectoryCollection` — a `.json` holding one dump, or a `.jsonl` with one dump
per line. Each input is converted to a temporary event JSONL under `$TMPDIR` (default `/tmp`) and
replayed at delay `0`, so **`show-dump` has no `--delay`**; it is always instant. Files with any
other extension are skipped without a message.

| Flag | Type | Default | What it does |
| --- | --- | --- | --- |
| `paths` | positional, `nargs="*"` | — | `.json` or `.jsonl` dump files |
| `--dir DIR` | str | `None` | Directory of `.json`/`.jsonl`, non-recursive |
| `--mode` | choice | `auto` | |
| `--selectable-text` | flag | off | |

```bash
uv run python -m platoon.visualization.cli show-dump \
  ./inference_results/my_run/rollouts/task_x/rollout_0/trajectory_collection.json
```

### `analyze-compare`

Pair two runs by task id, bucket them into A-better / B-better / ties / unmatched, optionally ask
an LLM to explain each pair, then open a table UI. Both sides need inputs or the parser errors.

| Flag | Type | Default | What it does |
| --- | --- | --- | --- |
| `method_a`, `method_b` | positional | — | Labels for the two sides |
| `--a PATH` | repeatable | `[]` | One input file for A |
| `--a-dir DIR` | str | `None` | Directory of `.json`/`.jsonl` inputs for A |
| `--b PATH` | repeatable | `[]` | One input file for B |
| `--b-dir DIR` | str | `None` | Directory of inputs for B |
| `--analysis-model MODEL` | str | `None` | LLM model id for the right-pane explanations; without it everything degrades to keyword heuristics |
| `--analyze-both-failed` | flag | off | Also LLM-explain ties where both sides failed |
| `--analysis-cache DIR` | str | `None` | Override the on-disk analysis cache directory |
| `--no-ui` | flag | off | Print summary JSON to stdout instead of opening the TUI |

```bash
uv run python -m platoon.visualization.cli analyze-compare baseline candidate \
  --a-dir /runs/baseline/events --b-dir /runs/candidate/events

# CI-friendly: counts and analyses as JSON, no terminal UI
uv run python -m platoon.visualization.cli analyze-compare a b \
  --a-dir A --b-dir B --no-ui
```

`--no-ui` prints `{"counts": {"a_better", "b_better", "ties", "unmatched"}, "analyses": {…}}`.

### `analyze-errors`

Extract per-collection failures for one run, cluster them, and open a table UI. Successful
collections are skipped unless you ask for them.

| Flag | Type | Default | What it does |
| --- | --- | --- | --- |
| `label` | positional | — | Label for the run |
| `--paths PATH` | repeatable | `[]` | One input file (plural flag name, singular value) |
| `--dir DIR` | str | `None` | Directory of `.json`/`.jsonl` inputs |
| `--model MODEL` | str | `None` | LLM model id |
| `--analysis-cache DIR` | str | `None` | Analysis cache directory override |
| `--no-cluster` | flag | off | Skip clustering entirely |
| `--sample N` | int | `None` | Randomly sample N failures to analyze |
| `--sample-seed N` | int | `None` | Seed for that sampling |
| `--passes N` | int | `2` | Hierarchical clustering passes |
| `--no-ui` | flag | off | Print issues and clusters as JSON |
| `--include-successes` | flag | off | Do not skip successful collections |
| `--llm-issues` | flag | off | Use the LLM to extract issues per collection instead of the keyword heuristic; slower, and needs `--model` |
| `--precompute-analyses` | flag | off | Run and cache per-issue LLM analyses before opening the UI |

```bash
uv run python -m platoon.visualization.cli analyze-errors candidate --dir /runs/candidate/events

uv run python -m platoon.visualization.cli analyze-errors candidate \
  --dir /runs/candidate/events --model openai/gpt-4o-mini \
  --llm-issues --precompute-analyses --sample 50 --sample-seed 0
```

## Analysis tools

These read trajectory collection dumps or event logs and print JSON or a table. They are plain
`python -m` scripts with no config file.

### Headline metrics

```bash
python -m platoon.analysis.compute_metrics [paths ...] [--dir DIR] [--denom N]
```

| Flag | Type | Default | What it does |
| --- | --- | --- | --- |
| `paths` | positional | — | `.json` (one dump) or `.jsonl` (one dump per line) |
| `--dir DIR` | str | `None` | Directory of dumps, non-recursive |
| `--denom N` | int | `None` | Accuracy denominator override; applied only when `> 0` |

Prints `total_collections`, `successes`, `denominator_used`, `accuracy`, `total_steps` and
`avg_steps_per_collection`. Success is a strict `reward == 1.0` on the first trajectory or its last
step, so a partial-credit reward scheme reports zero successes here.

```bash
uv run python -m platoon.analysis.compute_metrics --dir /runs/candidate/dumps
uv run python -m platoon.analysis.compute_metrics a.json b.jsonl --denom 200
```

### AppWorld metrics by difficulty

```bash
python -m platoon.analysis.appworld_metrics [paths ...] [--dir DIR] \
    [--difficulties 1,2,3] [--denom1 N] [--denom2 N] [--denom3 N]
```

`--difficulties` defaults to `"1,2,3"` (1 = easy, 2 = medium, 3 = hard); `--denom1/2/3` override
the per-difficulty denominators, each applied only when `> 0`. Output is keyed `easy` / `medium` /
`hard` / `overall`, each with the same six fields as `compute_metrics`. It needs
`appworld.load_task_ids` importable and will best-effort set `APPWORLD_ROOT` itself.

### Checkpoint acceptance

```bash
python -m platoon.analysis.checkpoint_acceptance [--dcp-recovery PATH] [--hf-export PATH] [--json]
```

Metadata-only validation — it never loads tensor shards. At least one of the two path flags is
required. `--hf-export` accepts either the export directory or its
`model.safetensors.index.json`. It exits `0` when every check passes and `1` otherwise, which makes
it usable as a gate in a job script.

```bash
uv run python -m platoon.analysis.checkpoint_acceptance \
  --dcp-recovery /ckpt/recover/step_100 --hf-export /ckpt/hf/step_100 --json
```

### Trajectories to SFT data

Two scripts with the same purpose and, annoyingly, different flag spellings. The core one uses
dashes; the AppWorld one uses underscores.

```bash
python -m platoon.train.collections_to_sft_data --input-dir DIR --output-path FILE [options]
```

| Flag | Type | Default | What it does |
| --- | --- | --- | --- |
| `--input-dir` | str, required | — | Scanned recursively for `.json`/`.jsonl` dumps |
| `--output-path` | str, required | — | JSONL output, one `{"messages": [...]}` per line |
| `--builder` | str | `platoon.appworld.agent:AppWorldRecursiveCodeActPromptBuilder` | `module.path:ClassName` implementing `build_messages_from_traj_dump` |
| `--reward-threshold` | float | `1.0` | Minimum trajectory reward passed to the builder |
| `--prompt-mode` | `sequence_extension` \| `no_sequence_extension` | `sequence_extension` | Prompt mode for the builder |
| `--include-reasoning` / `--no-include-reasoning` | flag pair | `True` | Keep reasoning tags in assistant messages |
| `--appworld-root` | str | `None` | Sets `APPWORLD_ROOT`; required by the AppWorld builders |

The AppWorld variant, `platoon.appworld.train_scripts.collections_to_sft_data`, takes
`--input_dir` and `--output_file` (both required), `--reward_threshold` (default `1.0`),
`--recursive` (flag; selects the recursive prompt builder), `--prompt_mode` (same choices, same
default) and `--no_reasoning` (flag). It searches only for files literally named
`trajectory_collection.json`.

### Plugin-specific analyzers

```bash
python -m platoon.appworld.inference_scripts.analyze_task_fanout \
  --events-root PATH --task-id TASK
```

Both flags are required. Prints per-collection fan-out and subagent cancellation rates for one
AppWorld task, plus an aggregate by fan-out bucket.

```bash
python plugins/textcraft/platoon/textcraft/inference_scripts/analyze_synth_results_by_difficulty.py \
  RESULT_DIR [RESULT_DIR ...] [--json-out PATH] [--json]
```

Takes one or more inference result directories (`nargs="+"`), reads
`reports/task_results.jsonl` or `reports/final_report.json` from each, and prints
difficulty-stratified comparison tables. `--json-out PATH` writes the machine-readable summary to
a file; `--json` prints it to stdout instead of the tables.

## Dataset generation

These regenerate committed data files **in place, inside the source tree** — the output paths are
computed from the module's own location, not from a flag. Regenerating with different arguments
rewrites the checked-in JSONL.

| Command | Flags | Writes |
| --- | --- | --- |
| `python -m platoon.number_search.tasks` | `--num_samples` (int, `50000`), `--eval_size` (int, `1000`) | `number_search_train.jsonl`, `number_search_val.jsonl` |
| `python -m platoon.textcraft.tasks` | `--num_samples` (`10000`), `--eval_size` (`1000`), `--min_depth` (`2`), `--max_depth` (`5`) | `textcraft_train.jsonl`, `textcraft_val.jsonl` |
| `python -m platoon.textcraft.synth_recipe_generator` | `--output-dir` (Path, `<pkg>/synth_recipes`), `--seed` (`42`), `--items-per-tier` (`8`), `--semantic-names` (flag) | recipe JSON files |
| `python -m platoon.textcraft.synth_tasks` | `--num_samples` (`10000`), `--eval_size` (`1000`), `--seed` (`42`), `--difficulty` (`easy`\|`medium`\|`hard`\|`extreme`\|`all`, default `all`), `--semantic-names` (flag) | `textcraft_synth_train.jsonl`, `textcraft_synth_val.jsonl` |
| `python -m platoon.oolong.tasks` | `--generate` (flag), `--dataset` (`synth`\|`real`\|`both`, default `synth`), `--max_samples` (int, `None`) | `oolong_<ds>_validation.jsonl`, `oolong_<ds>_test.jsonl` |
| `python -m platoon.email_search.data.local_email_db` | `--db-path` (str, `None`), `--overwrite` (flag) | the local sqlite database |

```bash
cd plugins/textcraft
uv run python -m platoon.textcraft.synth_recipe_generator \
  --output-dir platoon/textcraft/synth_recipes --seed 42
uv run python -m platoon.textcraft.synth_tasks --num_samples 10000 --eval_size 1000 --seed 42
```

`--semantic-names` switches TextCraft-Synth from generic item names (`m0_i1`) to semantic ones
(`iron_refined`). Generic is the default and is what you want for benchmarking, because semantic
names let the model lean on its priors instead of the recipe tree.

Without `--generate`, `platoon.oolong.tasks` prints a usage summary and exits. The email-search
database path defaults to `$PLATOON_EMAIL_SEARCH_DB_PATH` when that variable is set, otherwise the
repo-local data directory.

## Spawned, not run by hand

`python -m platoon.openreward.mcp_bridge` exposes one OpenReward session over stdio MCP. The
openreward rollout builds its argument list and spawns it per episode
(<span class="pl-src">plugins/openreward/platoon/openreward/rollout.py</span>), so you normally
never type it; running it manually is useful only for debugging a session server. Its flags:
`--env-name` (`toolathlongym`), `--split` (`train`), `--task-index` (int, `0`), `--task-name`,
`--session-url` (default `$OPENREWARD_SESSION_URL`), `--api-url` (`$OPENREWARD_API_URL`),
`--api-key` (`$OPENREWARD_API_KEY`, else `local`), `--output-dir` (Path, **required**),
`--max-tool-calls` (int, `0`) and `--tool-routing-overrides-json` (a JSON object, default `"{}"`).

## Development commands

Run these from the repository root.

| Command | What it does |
| --- | --- |
| `uv sync --dev` | Core package plus `pytest`, `pytest-asyncio`, `ruff`, `mypy`, `pre-commit`; no training backend |
| `uv run pytest tests/ -v` | The full suite, exactly as CI runs it |
| `uv run pytest tests/ -k subagent -v` | One theme |
| `uvx pre-commit run --all-files` | Ruff lint, Ruff format and `ty` — what CI's lint job runs |
| `uv run ruff check --fix .` | Lint only; `line-length = 120`, rule sets `E`, `F`, `I` |
| `uv run ruff format .` | Format only |
| `uvx ty check` | Type check; the three Textual TUI modules are excluded in `ty.toml` |

Commit messages are checked by a `conventional-pre-commit` hook at the `commit-msg` stage, which
has to be installed separately from the standard hooks. [Contributing](../contributing.md) has the
full setup, including the separate docs virtualenv and `mkdocs build --strict`.

## See also

- [Configuration reference](configuration.md) — every key these commands read
- [Troubleshooting](troubleshooting.md) — what the common failures mean
- [Installation](../get-started/installation.md) — backend extras and plugin venvs
- [Visualization tutorial](../tutorials/visualization.md) — keybindings and how to read the tree
