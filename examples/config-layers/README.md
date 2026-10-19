# Layered configs

Four small config files that build on each other with `extends:`, so a training run, a smoke run and an
evaluation share one definition of the environment. The how-to guide [Write and override
configs](../../docs/how-to/configs.md) explains each mechanism used here.

| File | Extends | What it adds |
|---|---|---|
| `environments.yaml` | | The `environments` list: number search, played by the `platoon-react` harness |
| `train.yaml` | `environments.yaml` | Model, optimizer, rollout limits, checkpoints and logging |
| `smoke.yaml` | `train.yaml` | A two-step run with small batches, and a W&B tag |
| `eval.yaml` | `environments.yaml` | An `inference:` section, so `platoon eval` evaluates a policy on the `val` split |

## Prerequisites

- platoon installed with the `tinker` and `ors` extras (number search, the environment here, ships with platoon):

  ```bash
  uv sync --extra tinker --extra ors
  ```

- For `platoon train`, `platoon eval` and `platoon rollout`: a Tinker API key in `TINKER_API_KEY` (and
  `TINKER_BASE_URL` for a self-hosted Tinker-compatible service, such as
  [miles-tinker](https://github.com/neulab/miles-tinker)). `platoon config` needs neither.

## Run it

1. Check every file. Each line names the schema the file validated against:

   ```console
   $ uv run platoon config validate examples/config-layers/*.yaml
   ok   examples/config-layers/environments.yaml (EnvironmentsConfig)
   ok   examples/config-layers/eval.yaml (PlatoonEvalConfig)
   ok   examples/config-layers/smoke.yaml (PlatoonTinkerRLTrainerConfig)
   ok   examples/config-layers/train.yaml (PlatoonTinkerRLTrainerConfig)
   ```

2. Print what the smoke run will actually use, with two more overrides on top:

   ```bash
   uv run platoon config show examples/config-layers/smoke.yaml \
     environments.0.dataset_kwargs.limit=10 'stats.wandb.tags+=nightly'
   ```

   The output is one YAML document with no `extends:` key: `environments` comes from `environments.yaml`,
   `train.model_name` from `train.yaml`, `train.batch_size: 2` from `smoke.yaml`, and the tags are
   `[smoke, nightly]`.

3. Run one task through the smoke config before training on it:

   ```bash
   uv run platoon rollout examples/config-layers/smoke.yaml --task train:0 --output-dir runs/debug
   ```

   The tree shows one agent and its outcome. A policy that finds the number prints something like:

   ```text
   rollout train-0-5933e9d60286  task train:0
   agents 1  steps 10  max depth 0  reward 1.0  outcome scored reward=1.0
   └─ policy e4c499d6  steps 10  reward 1.0  outcome scored reward=1.0
   ```

   `runs/debug/` now holds `config.resolved.yaml` (the merged config, with the source files and overrides in
   its header), `lock.json` (package versions, git SHA, host, time and command line),
   `trajectory_collection.json` and the rollout's event file under `rollouts/events/`.

4. Train, or evaluate:

   ```bash
   uv run platoon train examples/config-layers/smoke.yaml
   uv run platoon eval examples/config-layers/eval.yaml inference.workflow.num_rollouts_per_task=1
   ```

   Training writes into `runs/number-search/smoke/` (`log_path`/`experiment_name`/`trial_name`); evaluation
   into a new directory for each evaluation, `runs/eval/<policy>/<UTC time>/`.

## How it is tested

`tests/examples/test_config_layers_example.py` validates and merges these files, checks each override form, and
runs step 3 with a scripted binary-search policy in place of the Tinker sampler, so the example keeps working on a
CPU-only machine.
