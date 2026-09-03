<img src="assets/platoon_icon_cropped_no_background.png" width="320">

Build and train systems of agents.

**[Documentation](https://apga.github.io/platoon/)** — installation, quickstart, tutorials, code
walkthroughs, extension guides, and the full configuration reference.

| | |
| --- | --- |
| [Get started](https://apga.github.io/platoon/get-started/) | Install, run your first job, learn the core concepts |
| [Tutorials](https://apga.github.io/platoon/tutorials/) | Train on TextCraft, evaluate an endpoint, build your own task |
| [Code walkthroughs](https://apga.github.io/platoon/walkthroughs/) | The real source, traced end to end |
| [Customization](https://apga.github.io/platoon/customization/) | Add your own environment, agent, reward, loss or workflow |
| [Integrations](https://apga.github.io/platoon/integrations/) | OpenHands and OpenReward |
| [Reference](https://apga.github.io/platoon/reference/) | Every config key, component contract and CLI flag |

To build the docs locally:

```bash
uv venv .docs-venv --python 3.12
uv pip install --python .docs-venv -r docs/requirements.txt
./.docs-venv/bin/mkdocs serve
```

## Install

Use `uv` for the main development workflow:

```bash
uv sync
```

Install the training backend you need:

```bash
uv sync --extra tinker
# OR
uv sync --extra areal
```

Install a plugin from its directory:

```bash
cd plugins/<plugin-name>
uv sync --extra <backend>
```

AReaL is installed through `uv` extras. WandB is a core dependency; Tinker and WandB may require service credentials in your environment.

### Megatron backend (Transformer Engine)

The AReaL **FSDP** backend and all inference workflows work out of the box after
`uv sync --extra areal`. The **Megatron** backend additionally needs NVIDIA
Transformer Engine (TE), which is intentionally **not** part of the locked
dependencies: `transformer-engine-torch` is source-only and has no prebuilt wheel
for the pinned torch, so locking it would force a CUDA compile on every `uv sync`
(which would break FSDP/inference installs too). It is therefore excluded via a
`transformer-engine; sys_platform == 'never'` override and installed separately
only when you actually run Megatron.

Installing TE requires a real CUDA toolkit (`nvcc`) for the one-time compile of
`transformer-engine-torch` — e.g. inside the training container or after
`module load cuda`. Once built, the wheel is reusable on bare nodes without a
toolkit. Into the plugin venv (after `uv sync --extra areal`):

```bash
# Build deps + a real CUDA toolkit must be available (container / module load cuda).
uv pip install ninja
# --no-config bypasses the `sys_platform == 'never'` override; --no-build-isolation
# builds against the venv's torch instead of pulling a mismatched one.
CUDA_HOME=/usr/local/cuda uv pip install --no-config --no-build-isolation \
  "transformer-engine[pytorch]==2.12.0"
# The empty `transformer-engine` meta is dropped by the override, so install it
# explicitly or TE's PyPI sanity check fails ("Could not find transformer-engine"):
uv pip install --no-config "transformer-engine==2.12.0"
```

Verify with `python -c "import transformer_engine.pytorch"`. Select the backend
per run via `actor.backend: megatron` in the config.

> Tip: this can be automated with a local (uncommitted) helper that builds the
> wheel once, caches it under `.te-wheels/`, and fast-reinstalls it into any plugin
> venv — see the manual steps above for what such a helper needs to do.

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

