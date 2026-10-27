# Replay supervised conversations during RL with a step hook

**Goal:** add work at the end of each RL step without writing a trainer: a step hook that replays supervised
conversations on the policy that RL trains, attached to the built-in `rl` trainer from the config.

**Prerequisites:** the [quickstart](../quickstart/README.md), which this example extends; platoon installed with the
`tinker` and `ors` extras; a Tinker API key in `TINKER_API_KEY` to train.

| File | What it is |
|---|---|
| [`replay_sft.py`](replay_sft.py) | the hook, `ReplaySFT`, and the cross-entropy objective its updates train under, its own |
| [`conversations.jsonl`](conversations.jsonl) | the conversations it replays, one `{"messages": [...]}` per line |
| [`train.yaml`](train.yaml) | the quickstart's training config, with the hook attached |

## How the hook works

`ReplaySFT` is a `platoon.experimental.StepHookBase`. At the end of every `every`-th step, after the step's own
policy-gradient updates and before the step publishes, its `update(end)`:

1. takes the next `batch_size` conversations and renders them with the policy's renderer, the loss on the
   assistant's turns;
2. trains one optimizer update of the policy on them under its own objective,
   `end.train(policy, datums, objective=..., adam_params=...)`: the same training client and Adam state as the RL
   updates, with a learning rate of its own, its metrics under `replay_sft/`.

Platoon counts the hook's update. The step then publishes once, the RL updates and the replay together, as one new
policy version, so the replay spends none of the staleness budget (`train.max_staleness`). The step's metrics row
gains `batch/hook_optimizer_updates` and `replay_sft/*`.

The hook is also a checkpoint participant (`keys = ("replay_sft",)`): every full checkpoint saves its cursor, and a
resumed run takes it up before the training client opens, so it goes on with the next conversations.

## Attach it

```yaml
trainer_options:
  hooks:
    - hook: replay_sft:ReplaySFT       # or just replay_sft:ReplaySFT, to take its defaults
      options: {every: 2, learning_rate: 1.0e-5}
```

The rl trainer builds each hook as it starts, before the training service opens, with its class's
`from_run(run, options)` (`StepHookBase`'s takes no options; `ReplaySFT`'s reads them). A hook that does not import,
or refuses its options, stops the run with exit code 78.

## Run it

```bash
cd examples/quickstart
PYTHONPATH=.:../replay-sft-hook uv run platoon train ../replay-sft-hook/train.yaml
```

`PYTHONPATH` names the quickstart's directory, for its environment and harness, and this one, for the hook. The run
writes to `runs/quickstart/guess-qwen3-4b-replay-sft/`.

## How this example is tested

[`tests/examples/test_replay_sft_hook_example.py`](../../tests/examples/test_replay_sft_hook_example.py) validates
the config, then trains with it on CPU, on `platoon.testing`'s fake Tinker service, with a workflow whose groups are
one datum each in place of the quickstart's rollouts: the replay updates train under cross entropy with their own
learning rate, every other step publishes its RL and replay updates as one version, and a resumed run goes on with the
next conversations.

[Write a step hook](../../docs/extending/step-hooks.md) is the guide.
