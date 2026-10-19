# Your first run

**Goal:** run one number-search episode with the `platoon-codeact` harness and read its trajectory, train a LoRA
on number search for five steps, and watch the training rollouts in the trajectory viewer.

**Time:** about ten minutes, plus the training time on your backend.

Number search is a small environment that ships with platoon: the agent must find a hidden integer in a range,
and after each guess it is told "too low" or "too high". With `platoon-codeact`, the model acts by writing a
Python cell that calls the environment's `guess` tool as a function:

```text
<python>
guess(542)
</python>
```

## Prerequisites

- Python 3.12 or later, and a clone of the platoon repository with platoon installed with the `tinker` and `ors`
  extras ([Install](install.md)).
- Access to a **Tinker-compatible training backend** that can sample and train `Qwen/Qwen3-4B-Instruct-2507`:
  hosted Tinker, or a self-hosted server such as [miles-tinker](https://github.com/neulab/miles-tinker) (see below).
- No local GPU: the backend samples and trains, and platoon runs the environment and the harness on your CPU.

### What a Tinker-compatible training backend is

Platoon does not run the model. It talks to a service that implements the
[Tinker](https://thinkingmachines.ai/tinker/) API, through the Tinker Python SDK (`tinker` 0.16.1, the `tinker`
extra). The service provides two things:

- **sampling**: generate tokens from the base model or from a saved checkpoint (`platoon rollout`,
  `platoon eval`, and the rollouts of `platoon train`);
- **training**: compute the loss and gradients of a batch, take an optimizer step on LoRA (or full) weights, and
  save checkpoints that sampling can load (`platoon train`).

There are two kinds:

| Backend | What you set |
|---|---|
| The hosted Tinker service | `TINKER_API_KEY`, your Tinker API key. |
| A self-hosted server that implements the Tinker API, for example [miles-tinker](https://github.com/neulab/miles-tinker) | `TINKER_BASE_URL`, the server's URL, and `TINKER_API_KEY`, the key your server expects. |

The SDK reads both variables, so they apply to every command. The Tinker client takes only a key that starts with
`tml-`, so a server that checks no key still needs one such as `tml-local`. A training config can also set the URL
with the `tinker_base_url` key, which every command that runs that config uses (`platoon train`, and `platoon
rollout` of the config).

If your backend serves a different model, change `train.model_name` and `train.renderer_name` in the config (and
`inference.model_name` and `inference.renderer_name` in `eval.yaml`) to a model it serves and the renderer that
matches it ([Concepts](concepts.md#the-model-gateway-and-agent-keys)).

## Steps

Run every command from the root of your platoon clone. Number search needs nothing installed beyond platoon:
`platoon.envs.number_search` holds the environment, which generates its 50,000 training and 1,000 validation tasks
from a seed, and the CodeAct prompts the example uses.

### 1. Set your backend credentials

```bash
export TINKER_API_KEY=...                      # your key
export TINKER_BASE_URL=https://tinker.example  # only for a self-hosted server
```

### 2. Read the config

The example config is [`examples/number-search/train.yaml`](../../examples/number-search/train.yaml). Its
`environments` entry pairs the number-search environment with the `platoon-codeact` harness:

```yaml
environments:
  - service: python://platoon.envs.number_search.env:NumberSearchEnvironment
    harness: platoon-codeact
    harness_options:
      prompt_builder: platoon.envs.number_search.prompts:NumberSearchPromptBuilder
      include_reasoning: false
      subagents: false                 # one agent per episode: no launch_subagent()
    agents:
      max_depth: 0
    dataset_kwargs: {split_name: train, limit: 1000}
    eval_dataset_kwargs: {split_name: val, limit: 20}
```

The `train` section names the model (`Qwen/Qwen3-4B-Instruct-2507`, renderer `qwen3_instruct`), batches of 8
tasks with 4 episodes each, and 5 training steps; what it leaves out keeps its default, such as a LoRA of rank 32
(`train.lora_rank`). Episodes get at most 10 model calls of up to 512 tokens each.

Check what platoon makes of it:

```bash
platoon env list examples/number-search/train.yaml
```

Expected output, after the tables of service schemes and named environments:

```text
Environments of examples/number-search/train.yaml:
  name     weight  service                                                          harness          program  rollout
  default  tasks   python://platoon.envs.number_search.env:NumberSearchEnvironment  platoon-codeact  -        harness_rollout
```

The named environments include `platoon-number-search@1`, which stands for the same entry: a config can say
`env: platoon-number-search@1` instead of spelling out the service and the harness options
(`platoon env show platoon-number-search` prints it).

`platoon config show examples/number-search/train.yaml --defaults` prints every setting, defaults included.

### 3. Run one episode

```bash
platoon rollout examples/number-search/train.yaml --task val:0 --output-dir runs/first-rollout
```

`--task val:0` is the first validation task: "Guess the correct number between 119 and 965." The rollout opens a
session of the number-search environment, runs `platoon-codeact` against the base model, grades the episode and
prints the agent tree. Expected output (the ids, the step count and the reward depend on what the model does):

```text
... | INFO | platoon.harness.rollout | harness rollout val-0-18f7cfc3451b (platoon-codeact@1, single): scored
rollout val-0-18f7cfc3451b  task val:0
agents 1  steps 9  max depth 0  reward 1.0  outcome scored reward=1.0
└─ policy 21ee80a1  steps 9  reward 1.0  outcome scored reward=1.0
```

Read it as: one agent (the root, role `policy`, no subagents) made 9 model calls, found the number, and the
environment scored the episode 1.0. An episode in which the model never finds the number is still a valid
measurement, scored 0:

```text
agents 1  steps 10  max depth 0  reward 0.0  outcome scored reward=0.0 (no_submit)
```

`runs/first-rollout/` now holds `config.resolved.yaml` (the config the rollout ran), `lock.json` (package
versions, git commit, command line), `trajectory_collection.json` (the episode) and the rollout's event file
under `rollouts/events/`, which `platoon viz replay` reads.

### 4. Look at the trajectory

```bash
platoon viz show-dump runs/first-rollout/trajectory_collection.json
```

The viewer opens in your terminal with a tree of the episode's agents and their steps. Select a step to see the
cell the model wrote (`code`) and what it printed (`output`, for example `'Too low, try again.'`). Press `q` to
quit; [`platoon/visualization/README.md`](../../platoon/visualization/README.md) lists the other keys.

### 5. Train

```bash
platoon train examples/number-search/train.yaml
```

Each training step runs 8 tasks × 4 episodes, centers each episode's reward on the leave-one-out mean of its
group, and trains on the sampled tokens. Every 5 steps the run saves a checkpoint and evaluates on 20 validation
tasks. Each step logs a block of metrics, grouped by prefix:

```text
... | INFO | platoon.utils.stats_logger | ============================================================
... | INFO | platoon.utils.stats_logger | Step <n>
... | INFO | platoon.utils.stats_logger | ============================================================
```

The run ends with `Training complete. Total time: ...`.

Everything the run writes goes to `runs/number-search/first-run/` (`log_path`, then `stats.experiment_name`, then
`stats.trial_name`):

| Path | Contents |
|---|---|
| `config.resolved.yaml`, `lock.json` | The config the run trained with, and what it ran on. |
| `checkpoints.jsonl` | One record per checkpoint, with its `sampler_path` (`tinker://...`). |
| `evaluations.jsonl` | One line per evaluation that completed, so that a resumed run does not run it again. |
| `rollouts/train/<version>/events/` | One event file per training episode, by policy version. |
| `rollouts/eval/<version>/events/` | The evaluation episodes. |

Running the same command again resumes from the last checkpoint in that directory. To start a separate run,
change the trial name: `platoon train examples/number-search/train.yaml stats.trial_name=second-run`. Any
`key=value` after the config overrides one setting, with dots for nesting.

### 6. Watch the training rollouts

In a second terminal, while training runs or after it ends:

```bash
platoon viz tail --rdir runs/number-search/first-run/rollouts
```

The viewer loads the event files that exist when it starts, one per finished episode, and shows each like the
single episode in step 4.

## Next steps

- Evaluate the trained policy on 100 validation tasks: take a `sampler_path` from `checkpoints.jsonl` and run
  `platoon eval examples/number-search/eval.yaml inference.model_path=tinker://...`. It prints a JSON report
  summary (`success_rate`, `reward_mean`, ...) and writes the rollouts and report to a new directory,
  `runs/eval/<policy>/<UTC time>/` (`<policy>` is the checkpoint), which it names on stderr.
- Debug one task against a checkpoint: `platoon rollout examples/number-search/train.yaml --task val:7
  --model-path tinker://...`.
- Try the tool-calling harness, `platoon-react`, on the same task. The example has a config for it, in which the
  model calls `guess` as a native tool instead of writing cells:
  `platoon rollout examples/number-search/train_react.yaml --task val:0`.
- Write your own agent loop: [Write a function harness](../tutorials/function-harness.md). Bring your own tasks:
  [Build an environment](../tutorials/custom-environment.md).
- Read [Concepts](concepts.md) for harnesses, subagents, programs and outcomes, and
  [Architecture](../architecture.md) for how the pieces fit.
