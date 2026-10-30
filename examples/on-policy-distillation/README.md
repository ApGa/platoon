# Distill a frozen teacher into the policy that samples

**Goal:** train a small student on its own rollouts with a large teacher's judgement of every token it sampled
(on-policy distillation), with nothing but config: a frozen teacher policy, and the `reverse_kl` credit estimator.

The student samples each rollout of the [quickstart](../quickstart/README.md)'s GuessTheNumber game, as in RL
training. The teacher never plays: once a group of rollouts is done, it scores the tokens the student sampled, and
each token's advantage becomes the negative reverse KL between the two policies at that token,
`-kl_coef * (student logprob - teacher logprob)`. A token the teacher finds likelier than the student did is
reinforced, and one it finds unlikely is pushed down. This example distills alone; with `reward_coef: 1` the game's
reward joins the KL as in tinker-cookbook's on-policy distillation.

**Prerequisites:** the [quickstart](../quickstart/README.md), whose environment and harness this example trains;
platoon installed with the `tinker` and `ors` extras; a Tinker API key in `TINKER_API_KEY` to train.

| File | What it is |
|---|---|
| [`train.yaml`](train.yaml) | the quickstart's training, with a frozen teacher on the student's service |
| [`train-teacher-service.yaml`](train-teacher-service.yaml) | the same, with the teacher on a service of its own |

## The teacher: a frozen policy

```yaml
policies:
  teacher:
    model_name: Qwen/Qwen3-32B
    renderer_name: qwen3
    trainable: false
```

`train:` declares the run's policy `main`, the student, as it always did. `policies:` declares the run's other
policies under the same names. `trainable: false` makes the teacher frozen: the run opens only the sampling client it
scores with, and never steps, saves or prunes it. Its `load_checkpoint_path`, when set, names the sampler it serves (a
`tinker://` sampler path), say a teacher you trained earlier.

The teacher scores the student's token ids, so it must share the student's tokenizer: Qwen3 models do. A teacher of
another tokenizer stops `platoon train` and `platoon config validate` with exit code 78 as the config loads, before
any Tinker session opens. Where a tokenizer cannot be loaded then (no network), `platoon train` compares the
tokenizers it loaded for the two policies as it builds the estimator, and stops with exit code 78 there.

## Where the teacher runs

`train.yaml` gives the teacher no `tinker_base_url`, so it uses the run's service, the student's, which serves both
models. It opens a Tinker session of its own there, so that a fatal error on the teacher's session never ends the
student's.

`train-teacher-service.yaml` gives the teacher its own service instead, for a backend that cannot serve the teacher
beside the student, such as a self-hosted service of one base model:

```yaml
policies:
  teacher:
    tinker_base_url: http://localhost:8001
```

A job script that starts the teacher's service passes its address on the command line,
`policies.teacher.tinker_base_url=http://<host>:<port>`. The run's `identity.json` records each policy's model and
renderer, never its URL, so a resume may reach the services at new addresses.

## The credit: `reverse_kl`

```yaml
environments:
  - name: guess
    # ...
    credit: reverse_kl
    credit_options:
      teacher: teacher       # the frozen policy that scores
      # kl_coef: 1.0         # the weight of the KL term
      # reward_coef: 0.0     # above 0, adds the game's reward, centered on the group's baseline (1: the cookbook's)
```

The estimator runs once per training group, in the trainer, after the group's rollouts returned their rollout permits,
so the teacher's calls hold no rollout slot. It sets the advantage of every token the student sampled (its action
tokens). An evaluation runs no estimator, so it never calls the teacher. Each scoring call is bounded by the teacher's
`sample_timeout_seconds`: one that fails or runs out of time fails that group alone, which trains nothing and counts
under `credit/failures`, and the run goes on. `train.loss_fn: importance_sampling` trains the student on those
advantages with its own sampled logprobs as the behaviour policy's. A group whose rollouts all scored alike still
trains: its tokens carry the teacher's signal, so its verdict is `trained`, not `uniform`.

The step's metrics add `credit/reverse_kl/avg`, the mean reverse KL per action token: it falls as the student learns
to sample as the teacher would.

## Run it

```bash
export PYTHONPATH=$PWD/examples/quickstart
uv run platoon config validate examples/on-policy-distillation/train.yaml
uv run platoon train examples/on-policy-distillation/train.yaml
```

The run writes to `runs/quickstart/guess-distilled/`.

## How this example is tested

[`tests/examples/test_on_policy_distillation_example.py`](../../tests/examples/test_on_policy_distillation_example.py)
validates both configs and runs `platoon train` with each on CPU, on `platoon.testing`'s fake Tinker service, with
the quickstart's scripted bisection policy as the student and the fake service's sampling client as the teacher. It
checks that the teacher scored exactly the rows the step sent, on a session of its own (on its own URL in the second
config), that every action token's advantage is the negative reverse KL, and that the evaluation ran without calling
the teacher.

The configuration keys are in the [config reference](../../docs/reference/config.md), and
[Train and evaluate](../../docs/how-to/train-and-eval.md#distill-a-frozen-teacher-experimental) puts them in context.
