# TextCraft

TextCraft is a crafting environment for training tool-using and recursive LLM agents.

It includes:

- Original TextCraft: Minecraft-style recipes with shallow crafting trees.
- TextCraft-Synth: procedurally generated recipes with deeper hierarchies, difficulty tags, and abstract item names.

## Install

```bash
cd plugins/textcraft
uv sync --extra areal --extra wandb
```

Use `--extra tinker` for Tinker experiments.

## Train

Tinker:

```bash
uv run python -m platoon.textcraft.train_scripts.tinker.train_tinker \
  --config platoon/textcraft/configs/tinker/textcraft_synth_recursive_tinker.yaml
```

AReaL:

```bash
uv run python -m areal.launcher.local \
  platoon/textcraft/train_scripts/areal/train_areal_synth.py \
  --config platoon/textcraft/configs/areal/textcraft_synth_ctx40000_recursive_medium_areal.yaml
```

TextCraft-Synth AReaL configs live in:

```text
platoon/textcraft/configs/areal/
```

Common variants are organized by context length, rollout style, difficulty, and loss settings.

## Generate Data

Original TextCraft tasks:

```bash
uv run python -m platoon.textcraft.tasks \
  --num_samples 10000 \
  --eval_size 1000
```

TextCraft-Synth recipes and tasks:

```bash
uv run python -m platoon.textcraft.synth_recipe_generator \
  --output-dir platoon/textcraft/synth_recipes \
  --seed 42

uv run python -m platoon.textcraft.synth_tasks \
  --num_samples 10000 \
  --eval_size 1000 \
  --seed 42
```

## Environment

Agents interact through crafting actions such as `craft`, `get_info`, `view_inventory`, `finish`, and, in recursive environments, `launch_subagent`.

Important modules:

- `env.py`: environment implementations and synth environment factories.
- `tasks.py`: original TextCraft task loading and generation.
- `synth_tasks.py`: TextCraft-Synth task generation and loading.
- `synth_recipe_generator.py`: synthetic recipe generation.
- `registry.py`: plugin component definitions used by `platoon.train.auto`.
