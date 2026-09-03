# CLI

Platoon ships no console script. Every command is `python -m <module>`, run with `uv run` so it
picks up the right virtualenv.

## Override syntax

Two config loaders live in the repository, and they take overrides in different forms. Get this
right first and most command lines follow.

| Path | Config flag | Override form |
| --- | --- | --- |
| <span class="pl-tag pl-tag--areal">AReaL</span> training | `--config <yaml>` | `key=value`, **no dashes** |
| <span class="pl-tag pl-tag--tinker">Tinker</span> training, evaluation runners | `--config` / `-c` | `--dotted.key value` |
| Visualization CLI | n/a | ordinary flags |

```bash
# AReaL — OmegaConf style
uv run python platoon/number_search/train.py \
  --config platoon/number_search/nv_number_search_cispo_areal.yaml \
  trial_name=debug-run train_dataset.batch_size=16

# Tinker — argparse style
uv run python -m platoon.number_search.train_tinker \
  --config platoon/number_search/number_search_tinker.yaml \
  --train.batch_size 64
```

The AReaL loader also resolves `${...}` interpolation inside the YAML. The `--dotted.key` loader
parses values by type, so `1` and `0` become booleans — see [configuration](configuration.md) for
the full coercion rules.

## Where to run from

| Command | Directory |
| --- | --- |
| Training, evaluation, anything importing `platoon.<plugin>` | `plugins/<name>/` |
| Visualization CLI | anywhere; a plugin directory is convenient because config paths are relative |
| `pytest`, `ruff`, `pre-commit`, `mkdocs` | repository root |

Each plugin under `plugins/` is its own `uv` project whose environment contains the plugin plus
Platoon as an editable dependency, which is why the examples below start with a `cd`. A plugin kept
in your own repository works the same way — see [extend Platoon](../guides/extend.md).

## Training

### Registry entrypoints

Config-only training. Both entrypoints read a top-level `environments:` list and resolve every
component — agent, environment, rollout, backend — through the registry, with no per-plugin Python.

```bash
python -m platoon.train.tinker.train --config <yaml> [--dotted.key value ...]
python -m platoon.train.areal.train  --config <yaml> [key=value ...]
```

```bash
cd plugins/textcraft
uv run python -m platoon.train.tinker.train \
  --config platoon/textcraft/configs/tinker/textcraft_synth_depth_aware_tinker.yaml
```

That top-level `environments:` list is component wiring. It is unrelated to the nested
`openreward.environments:` mixture list, which selects task sources — see
[OpenReward](../plugins/openreward.md).

### Per-plugin training scripts

A plugin can also ship its own entrypoint. Each takes `--config <yaml>` plus overrides in its
backend's syntax.

| Plugin | <span class="pl-tag pl-tag--areal">AReaL</span> | <span class="pl-tag pl-tag--tinker">Tinker</span> |
| --- | --- | --- |
| appworld | `platoon.appworld.train_scripts.areal.train_areal` | — |
| codegrep | `platoon.codegrep.train` | `platoon.codegrep.train_tinker` |
| deepdive | `platoon.deepdive.train_scripts.areal.train_areal` | `platoon.deepdive.train_scripts.tinker.train_tinker` |
| email-search | `platoon.email_search.train_scripts.areal.train_areal` | `platoon.email_search.train_scripts.tinker.train_tinker` |
| number-search | `platoon.number_search.train` | `platoon.number_search.train_tinker` |
| oolong | `platoon.oolong.train_scripts.areal.train_areal` | `platoon.oolong.train_scripts.tinker.train_tinker` |
| openreward | `platoon.openreward.train_scripts.areal.train_areal` | `platoon.openreward.train_scripts.tinker.train_tinker` |
| textcraft | `…areal.train_areal`, `…areal.train_areal_synth` | `…tinker.train_tinker`, `…tinker.train_tinker_synth`, `…tinker.train_tinker_synth_recursive`, `…tinker.train_tinker_synth_depth_aware` |

The TextCraft modules are under `platoon.textcraft.train_scripts.`; the openhands plugin has no
training script, because it contributes an agent harness rather than a task.

```bash
cd plugins/textcraft
uv run python -m platoon.textcraft.train_scripts.tinker.train_tinker_synth_recursive \
  --config platoon/textcraft/configs/tinker/textcraft_synth_recursive_tinker.yaml
```

Pass `--config` explicitly. Some scripts carry a built-in default, but naming the file keeps the run
reproducible and the command readable.

Multi-node AReaL runs invoke the same modules from the shell scripts in `slurm-scripts/`. See
[scale up](../guides/scale.md).

### Restart wrapper <span class="pl-tag pl-tag--tinker">Tinker</span>

```bash
python -m platoon.train.tinker.restart_wrapper [--max-restarts N] -- <command ...>
```

Runs a training command and restarts it when the watchdog kills a hung step (exit code `2`). Any
other non-zero exit is left alone, and `Ctrl-C` is forwarded to the child. `--max-restarts` defaults
to `5`, `--restart-delay` to `10` seconds.

```bash
cd plugins/textcraft
uv run python -m platoon.train.tinker.restart_wrapper --max-restarts 5 \
  -- uv run python -m platoon.textcraft.train_scripts.tinker.train_tinker \
     --config platoon/textcraft/configs/tinker/textcraft_tinker.yaml
```

## Evaluation runners

Trainer-free benchmark harnesses that point a task plugin at an OpenAI-compatible endpoint. Each
takes `--config` plus `--dotted.key value` overrides; there are no other flags, since everything
else is a config key.

| Plugin | Module |
| --- | --- |
| appworld | `platoon.appworld.inference_scripts.run_inference` |
| deepdive | `platoon.deepdive.inference_scripts.run_inference` |
| email-search | `platoon.email_search.inference_scripts.run_inference` |
| oolong | `platoon.oolong.inference_scripts.run_inference` |
| openreward | `platoon.openreward.inference_scripts.run_inference` |
| textcraft | `platoon.textcraft.inference_scripts.run_inference`, `…run_synth_inference` |

Configs live under each plugin's `configs/inference/`. The keys you override most often:

| Key | Default | What it does |
| --- | --- | --- |
| `stage` | `full` | `rollouts` collects only; `report` rescores what is on disk, with no endpoint calls |
| `num_tasks` | `100` | How many tasks to draw |
| `task_id` | `None` | Run a single task, ignoring the dataset |
| `use_recursive_agent` | per plugin | Multi-agent rollout function rather than the flat one |
| `dataset_split` | per plugin | Which split to draw from |

```bash
cd plugins/textcraft
uv run python -m platoon.textcraft.inference_scripts.run_inference \
  --config platoon/textcraft/configs/inference/textcraft_inference.yaml \
  --inference.model_endpoint http://127.0.0.1:30000/v1 \
  --inference.output_dir ./inference_results/exp2
```

[Evaluate a model](../guides/evaluate.md) walks through serving a model and reading the report.

## Visualization CLI

One entrypoint with five subcommands. A subcommand is required.

```bash
python -m platoon.visualization.cli {tail,replay,show-dump,analyze-compare,analyze-errors} ...
```

`tail`, `replay` and `show-dump` accept `--mode` (`auto` | `codeact` | `openhands`) and
`--selectable-text`, which releases the mouse so terminal drag-selection works.

**`tail`** — follow live event logs. Takes JSONL paths, `--dir` for one directory, or `--rdir` to
recurse. It replays what is already in the file, then follows.

```bash
uv run python -m platoon.visualization.cli tail --rdir ./rollout_results
```

**`replay`** — replay a finished log from the start. Paths or `--dir`; `--delay` sets the seconds
between events (default `0.5`, `0` loads everything at once).

```bash
uv run python -m platoon.visualization.cli replay --dir ./rollout_results/events --delay 0
```

**`show-dump`** — view a serialized `TrajectoryCollection`: a `.json` holding one dump, or a
`.jsonl` with one per line. Always instant, so there is no `--delay`.

```bash
uv run python -m platoon.visualization.cli show-dump \
  ./inference_results/my_run/rollouts/task_x/rollout_0/trajectory_collection.json
```

**`analyze-compare`** — pair two runs by task id, bucket them into A-better, B-better, ties and
unmatched, then open a table UI. Inputs come from `--a` / `--b` (repeatable) or `--a-dir` /
`--b-dir`. Add `--analysis-model` for LLM-written explanations, or `--no-ui` to print the summary as
JSON.

```bash
uv run python -m platoon.visualization.cli analyze-compare baseline candidate \
  --a-dir /runs/baseline/events --b-dir /runs/candidate/events
```

**`analyze-errors`** — extract failures from one run, cluster them, and open a table UI. Inputs come
from `--paths` (repeatable) or `--dir`. `--model` enables LLM analysis, `--sample N` limits how many
failures are analyzed, and `--no-ui` prints JSON.

```bash
uv run python -m platoon.visualization.cli analyze-errors candidate \
  --dir /runs/candidate/events --model openai/gpt-4o-mini
```

[Inspect rollouts](../guides/inspect-rollouts.md) covers keybindings and how to read the tree.

## Development

Run these from the repository root.

| Command | What it does |
| --- | --- |
| `uv sync --dev` | Core package plus the test and lint tools; no training backend |
| `uv run pytest tests/ -v` | The full test suite, as CI runs it |
| `uvx pre-commit run --all-files` | Ruff lint, Ruff format and `ty` — the lint job |
| `uv run ruff check --fix .` | Lint only |
| `uv run ruff format .` | Format only |

[Contributing](../contributing.md) has the rest of the setup, including the docs build.

## See also

- [Configuration](configuration.md) — every key these commands read
- [Installation](../get-started/installation.md) — backend extras and plugin environments
- [FAQ](faq.md) — common failures and what they mean
