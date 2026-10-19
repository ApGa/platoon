# Quickstart: train an agent to guess a number

**Goal:** write an environment and a harness, run one task to check the wiring, then train a LoRA with
reinforcement learning and evaluate it.

**Prerequisites:**

- platoon installed with the `tinker` and `ors` extras (see the [top-level README](../../README.md#install));
- a Tinker API key in `TINKER_API_KEY` for steps 3 to 5 (hosted Tinker, or `TINKER_BASE_URL` for a self-hosted
  Tinker-compatible server, such as [miles-tinker](https://github.com/neulab/miles-tinker)).

The example has five files:

| File | What it is |
|---|---|
| [`guess.py`](guess.py) | the environment: an OpenReward (ORS) `Environment` class with one tool, `guess` |
| [`guesser.py`](guesser.py) | the harness: a Python function that calls the model and the env's tools |
| [`train.yaml`](train.yaml) | a training config that pairs the two |
| [`eval.yaml`](eval.yaml) | an evaluation config |
| [`test_guess.py`](test_guess.py) | tests of the environment, the harness and `train.yaml`, which run on CPU |

## 1. Read the environment

`GuessTheNumber` hides a number from 1 to 100. Its `guess` tool answers `higher` or `lower`, and a correct
guess returns `finished=True` with `reward=1.0`, which ends and grades the episode. An episode that never
guesses right is graded 0. Each split has 100 tasks; platoon names them `<split>:<index>`, so `train:41` is
the task whose secret is 42.

## 2. Read the harness

`example-guesser` is a function harness, registered with `@pt.harness("example-guesser")`. It receives an `AgentContext`:

- `ctx.instruction` is the environment's prompt;
- `ctx.list_tools()` and `ctx.call_tool()` reach the environment's tools;
- `ctx.chat()` samples the policy through platoon's model gateway, which records the exact tokens of every
  call for training;
- `ctx.budget.remaining()` is the step budget (`max_steps` in the config).

It returns an `AgentResult`. The environment, not the harness, decides the reward.

## 3. Run one task

From the repository root, run one task with the base model and print the agent tree:

```bash
cd examples/quickstart
PYTHONPATH=. uv run platoon rollout train.yaml --task train:41
```

`PYTHONPATH=.` lets platoon import `guess` and `guesser`. Expected output (the guesses and step count depend
on the model):

```
rollout train-41-e21a6442a5ff  task train:41
agents 1  steps 7  max depth 0  reward 1.0  outcome scored reward=1.0
└─ policy 32f8339e  steps 7  reward 1.0  outcome scored reward=1.0
```

The run directory (`runs/adhoc/<time>-train-41/` unless you pass `--output-dir`) holds the resolved config,
`lock.json`, `trajectory_collection.json` and the rollout's event file under `rollouts/events/`. Open the
trajectory in the viewer:

```bash
uv run platoon viz show-dump runs/adhoc/*/trajectory_collection.json
```

## 4. Train

```bash
PYTHONPATH=. uv run platoon train train.yaml
```

This runs five optimizer steps of 8 tasks with 4 rollouts each (`train.batch_size`,
`train.workflow_config.group_size`), with leave-one-out baselines, then saves a checkpoint and evaluates on 20
test tasks. Logs, checkpoints and the resolved config go to `runs/quickstart/guess-qwen3-4b-lora32/`.

Any config key can be overridden on the command line, with dots for nesting:

```bash
PYTHONPATH=. uv run platoon train train.yaml train.batch_size=16 train.max_training_steps=20
```

## 5. Evaluate

Evaluate the base model, or a checkpoint from step 4:

```bash
PYTHONPATH=. uv run platoon eval eval.yaml inference.model_path=tinker://<run id>/sampler_weights/<name>
```

The command prints a JSON summary (`success_rate`, `success_at_k`, reward statistics) and writes rollouts and
a report to a new directory for each evaluation, `runs/eval/<policy>/<UTC time>/` (the checkpoint, else the model),
which it names on stderr. Name that directory to resume an evaluation that stopped:
`inference.output_dir=runs/eval/<policy>/<UTC time>`.

## How this example is tested

[`test_guess.py`](test_guess.py) tests the environment, the harness and the training config with `platoon.testing`,
on CPU, with a scripted policy in place of Tinker:

```bash
PYTHONPATH=. uv run pytest -p platoon.testing test_guess.py
```

[Test an environment, a harness or a program on CPU](../../docs/how-to/test-extensions.md) walks through it.
[`tests/examples/test_quickstart.py`](../../tests/examples/test_quickstart.py) also runs steps 3 and 5 through the
`platoon` CLI and validates both configs.

## Next steps

- Swap the harness: set `harness: platoon-react` (native tool calls) or `harness: platoon-codeact` (the model
  writes Python that calls the tools), or train on a mixture such as `{example-guesser: 1, platoon-react: 1}`.
- Read [docs/architecture.md](../../docs/architecture.md) for how environments, harnesses, the model gateway
  and the trainer fit together.
