<img src="assets/platoon_icon_cropped_no_background.png" width="320">

Build and train systems of agents.

## Install

Use `uv` for the main development workflow:

```bash
uv sync
```

Install the training backend you need:

```bash
uv sync --extra tinker --extra wandb
# OR
uv sync --extra areal --extra wandb
```

Install a plugin from its directory:

```bash
cd plugins/<plugin-name>
uv sync --extra <backend> --extra wandb
```

AReaL is installed through `uv` extras. Tinker and WandB may also require service credentials in your environment.

## Plugins

- `plugins/textcraft`: crafting tasks, including the synthetic recursive TextCraft benchmark.
- `plugins/appworld`: AppWorld API tasks.
- `plugins/oolong`: long-context aggregation tasks.
- `plugins/codegrep`: code localization tasks.
- `plugins/email-search`: ART-E email-search tasks.
- `plugins/number-search`: compact number-guessing tasks for quick RL smoke tests.

Each plugin README contains task-specific setup and example commands.

## Training

Tinker example:

```bash
cd plugins/textcraft
uv run python -m platoon.textcraft.train_scripts.tinker.train_tinker \
  --config platoon/textcraft/configs/tinker/textcraft_tinker.yaml
```

AReaL example:

```bash
cd plugins/number-search
uv run python3 platoon/number_search/train.py \
  --config platoon/number_search/nv_number_search_cispo_areal.yaml
```

Most config values can be overridden from the CLI:

```bash
uv run python3 platoon/number_search/train.py \
  --config platoon/number_search/nv_number_search_cispo_areal.yaml \
  trial_name=debug-run \
  train_dataset.batch_size=16
```

## Inference

Standalone inference workflows benchmark an OpenAI-compatible endpoint and write rollouts plus aggregate reports under `inference.output_dir`.

```bash
cd plugins/appworld
uv run python -m platoon.appworld.run_inference \
  --config platoon/appworld/configs/inference/appworld_inference.yaml
```

## AReaL Config Surface

Platoon intentionally exposes a smaller AReaL config surface than upstream AReaL:

- `rollout.backend` and `actor.backend` select engine placement.
- `workflow_config` controls rollout grouping and reward processing.
- `workflow_config.rollout_config.inference_params` controls rollout generation.
- `loss_fn_config` selects the policy loss and loss-specific arguments.
- `environments` selects dataset, task, rollout, reward, and workflow components through `platoon.train.auto`.
- `train_dataset.batch_size` and `valid_dataset.batch_size` control dataloader sizing.

## Visualization

Use the trajectory visualization CLI to tail, replay, and analyze rollout event logs:

```bash
uv run -m platoon.visualization.cli --help
```

See [`platoon/visualization/README.md`](platoon/visualization/README.md).

