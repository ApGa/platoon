# CodeGrep

CodeGrep trains agents to localize files relevant to a software issue. It is intended as a lightweight code-search environment for automated software-engineering workflows.

## Install

```bash
cd plugins/codegrep
uv sync --extra tinker --extra wandb
```

Use `--extra areal` for AReaL experiments.

## Train

Tinker:

```bash
uv run python -m platoon.codegrep.train_tinker \
  --config platoon/codegrep/codegrep_tinker.yaml
```

AReaL:

```bash
uv run python3 platoon/codegrep/train.py \
  --config platoon/codegrep/codegrep_areal.yaml
```

## Data and Rewards

Training data is stored as parquet files in the plugin. Each task gives an issue description; the agent searches the repository and predicts relevant files. Rewards compare the predicted files with ground-truth files from the fix.

This plugin depends on `platoon-openhands` for code exploration tools.
