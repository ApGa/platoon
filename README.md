<img src="assets/platoon_icon_cropped_no_background.png" width="400">


## Setup

### Core Installation

Install the core platoon package:

```bash
# Using uv (recommended)
uv sync

# Using pip
pip install -e .
```

### Training Backend Installation

Platoon supports two training backends: **Tinker** and **AReaL**. Install the one you need:

#### Tinker Backend

```bash
# Using uv
uv sync --extra tinker

# Using pip
pip install -e ".[tinker]"
```

#### AReaL Backend

> **Note**: AReaL requires `uv` for installation as it's not available on PyPI.

```bash
# Using uv only (required)
uv sync --extra areal
```

### Optional Dependencies

#### WandB (Experiment Tracking)

WandB should be installed alongside your chosen training backend:

```bash
# With Tinker backend

# Using uv
uv sync --extra tinker --extra wandb

# Using pip
pip install -e ".[tinker,wandb]"

# With AReaL backend (uv only)
uv sync --extra areal --extra wandb
```

### Plugin Installation

Install a plugin or extension:

```bash
cd plugins/<plugin-name>
uv sync  # or: pip install -e .
```

## Training a Model with Reinforcement Learning

Platoon supports two training backends: Tinker and AReaL.

### Training with Tinker

Tinker uses a service-based architecture. Make sure your Tinker service is running before training.

#### Single Plugin Training Example

```bash
cd plugins/textcraft  # or number-search, codegrep

# Using uv
uv run python -m platoon.textcraft.train_tinker --config textcraft_tinker.yaml

# Using python directly (after pip install)
python -m platoon.textcraft.train_tinker --config textcraft_tinker.yaml
```

#### CLI Overrides

Override config values from the command line:

```bash
uv run python -m platoon.textcraft.train_tinker \
    --config textcraft_tinker.yaml \
    stats.experiment_name=my-experiment \
    stats.trial_name=trial1 \
    train.batch_size=64 \
    train.optimizer.learning_rate=1e-5
```

#### WandB Logging

Enable WandB logging by setting the mode in your config or via CLI:

```bash
# Via CLI override
uv run python -m platoon.textcraft.train_tinker \
    --config textcraft_tinker.yaml \
    stats.wandb.mode=online \
    stats.wandb.project=my-project
```

Or in your YAML config:

```yaml
stats:
  experiment_name: my-experiment
  trial_name: trial1
  wandb:
    mode: online  # Options: online, offline, disabled
    project: my-project
    entity: my-team  # optional
    tags:
      - experiment-tag
```

### Training with AReaL

AReaL uses a distributed training architecture. Refer to [AReaL documentation](https://github.com/inclusionAI/AReaL) for detailed setup instructions.

#### Single Node Training Example

```bash
cd plugins/textcraft  # or number-search, codegrep

uv run python3 -m areal.launcher.local \
    platoon/textcraft/train.py \
    --config platoon/textcraft/textcraft_areal.yaml \
    experiment_name=textcraft-reinforce \
    trial_name=trial0
```

#### Multi-Node Training

See AReaL documentation for distributed training setup.

## Configuration

### Tinker Config Structure

```yaml
# Training configuration
train:
  model_name: Qwen/Qwen3-4B      # HuggingFace model identifier
  renderer_name: qwen3            # Prompt renderer type
  batch_size: 32
  num_epochs: 10
  lora_rank: 32
  optimizer:
    learning_rate: 1e-6
  workflow_config:
    group_size: 8                 # Rollouts per task for GRPO
    rollout_config:
      max_steps: 50
      timeout: 900

# Eval configuration
eval:
  strategy: epoch                 # When to evaluate: epoch, step, none
  every: 1                        # Frequency of evaluation

# Checkpoint configuration
checkpoint:
  strategy: epoch
  every: 5
  load_checkpoint_path: null      # Resume from checkpoint

# Paths
log_path: ./logs
tinker_base_url: null             # Tinker service URL (uses default if null)

# Stats and logging
stats:
  experiment_name: my-experiment
  trial_name: trial1
  wandb:
    mode: online
    project: my-project
```

### AReaL Config Structure

See [AReaL documentation](https://github.com/inclusionAI/AReaL) for config options.

## Visualizing Trajectories

See the dedicated guide: [Trajectory visualization CLI](platoon/visualization/README.md).

## Development

### Setup

```bash
# Install dev dependencies (include your existing extras to preserve them)
uv sync --extra tinker --group dev                 # Tinker backend
uv sync --extra areal --group dev                  # AReaL backend
uv sync --extra tinker --extra wandb --group dev   # Tinker + WandB

# Install pre-commit hooks
uvx pre-commit install
```

### Running Tests

```bash
uv run pytest tests/ -v
```

### Linting and Type Checking

The project uses [ruff](https://docs.astral.sh/ruff/) for linting/formatting and [ty](https://docs.astral.sh/ty/) for type checking. Both run automatically via pre-commit hooks.

```bash
# Run all pre-commit checks manually
uvx pre-commit run --all-files

# Run individual tools
uv run ruff check .           # Lint
uv run ruff format .          # Format
uvx ty check                  # Type check
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`. They include:
- **ruff**: Linting with auto-fix
- **ruff-format**: Code formatting
- **ty**: Type checking
- **conventional-pre-commit**: Validates commit message format

If a hook fails, fix the issues and commit again.

### Commit Messages

This project uses [Conventional Commits](https://www.conventionalcommits.org/). Commit messages must follow the format:

```
type(scope): description
```

Common types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
- `feat: add user authentication`
- `fix(api): handle null response`
- `docs: update README`

### CI

Pull requests and pushes to `main` trigger CI checks (see [.github/workflows/ci.yml](.github/workflows/ci.yml)):
- **pr-title**: Validates PR title follows conventional commit format
- **lint**: Runs pre-commit hooks (ruff + ty)
- **test**: Runs pytest
