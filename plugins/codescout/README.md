# CodeScout Training in Platoon using Apptainer Runtime

This README explains how to train LLMs using RL recipe the [CodeScout paper](https://arxiv.org/abs/2603.17829) when using the Apptainer sandbox.

> Assumption: commands are run on Linux, from the repo root unless stated otherwise. ripgrep must be installed on the system.


![CodeScout main figure (verified file-level)](https://raw.githubusercontent.com/OpenHands/codescout/974238b1d22308fd9cf0c79d3544697f4206ec2c/docs/verified_file_main.png)

![CodeScout main figure (verified function-level)](https://raw.githubusercontent.com/OpenHands/codescout/974238b1d22308fd9cf0c79d3544697f4206ec2c/docs/verified_function_main.png)

![CodeScout system diagram](https://raw.githubusercontent.com/OpenHands/codescout/974238b1d22308fd9cf0c79d3544697f4206ec2c/docs/recipe.png)

---

## Instructions to Train CodeScout Models

### Environment setup

Execute the following commands from the repository root:

```bash
mkdir -p /tmp/apptainer_cache
mkdir -p /tmp/apptainer_tmp
mkdir -p /tmp/areal/experiments
mkdir -p /tmp/areal/name_resolve
uv sync --extra areal --extra wandb
source .venv/bin/activate
uv pip install -e plugins/codescout
```

---

### Apptainer and logging environment variables

Set these enviroment variables before launching training:

```bash
export APPTAINER_CACHEDIR=/tmp/apptainer_cache
export APPTAINER_TMPDIR=/tmp/apptainer_tmp
export OPENHANDS_SUPPRESS_BANNER=1
export WANDB_API_KEY="<your_wandb_api_key>"
```

### Train CodeScout Models

We provide an example config in [train_codescout.yaml](./platoon/codescout/train_codescout.yaml) which trains CodeScout-1.7B using RL from an SFT'ed checkpoint (CodeScout-1.7B-RFT) which can be modified to use a different model, GPU setup, and other training hyper-parameters.

```bash
cd plugins/codescout
python3 -m areal.launcher.local \
	platoon/codescout/train.py \
	--config platoon/codescout/train_codescout.yaml
```