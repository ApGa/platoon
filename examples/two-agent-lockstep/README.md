# Train two agents' policies in lockstep

**Goal:** train two policies in one run, one per agent role, with nothing but config: the agent that delegates and
the agents it spawns each sample a policy of their own, and every step trains both, each with its own LoRA adapter,
optimizer, loss and staleness bound, on its own Tinker service or on a shared one.

The run trains the [subagents example](../subagents/README.md)'s delegator on its Vault tasks. The root agent reads
the task and hands halves of its keys to subagents, which read their keys and answer; the root adds up their answers.
Here the root samples `main`, the policy `train:` declares, and the subagents sample `worker`. Each policy learns its
own job: the delegator to split and add up, the worker to read and answer.

**Prerequisites:** the [subagents example](../subagents/README.md), whose environment and harness this example trains;
platoon installed with the `tinker` and `ors` extras; a Tinker API key in `TINKER_API_KEY` to train.

| File | What it is |
|---|---|
| [`train.yaml`](train.yaml) | the subagents example's training, with a second trained policy for the subagents, on the run's service |
| [`train-worker-service.yaml`](train-worker-service.yaml) | the same, with the worker on a service of its own |

## Two policies, two roles

```yaml
policies:
  worker:
    model_name: Qwen/Qwen3-4B-Instruct-2507
    renderer_name: qwen3_instruct
    lora_rank: 16
    optimizer: {learning_rate: 1.0e-5}
    max_staleness: 2

agents:
  roles:
    subagent: {policy: worker}
```

`train:` declares `main`, as it always did. `policies:` declares the run's other policies under `train:`'s names, and
a policy is trained unless it says `trainable: false`. A trained policy's unset `optimizer` and `loss_fn` are
`train:`'s; its `loss_fn_config` is `train:`'s when its loss is, else empty; its `lora_rank` is `train.lora_rank`
unless it sets one. It may load weights to start from (`load_checkpoint_path`), and their optimizer state too
(`load_optimizer_state: true`).

`agents.roles.subagent.policy: worker` binds the agents spawned with the role `subagent` (`ctx.agents.spawn`'s default)
to the worker, for their whole lives; the root agent samples `agents.policy`, `main` when it is left out. Every trained
policy must be sampled by some agent, or loading stops with exit code 78: no row would ever train it.

## What a step does

Each step takes `train.batch_size` settled groups, as it always did, and trains both policies on them:

- **Credit** runs once per group, over every agent of every rollout. With two trained policies an entry without
  `credit:` credits with `keyed` on `[task, policy, role]`: each agent is centered on the mean reward of the group's
  agents of its policy and role, so the delegators of a group get one baseline and their workers another. Rows then
  split by the policy that sampled them.
- **Staleness** is judged per policy: the delegator's rows against `main`'s version and `train.max_staleness` (1),
  the worker's against the worker's version and its own `max_staleness` (2). A policy's stale rows of a group are
  dropped together, and the group's other rows still train. The dispatcher's horizon counts the smallest bound of the
  two, so rollouts run at most one step ahead.
- **Updates:** each microbatch sends each policy's rows to its own training client, and at each minibatch's end both
  optimizer steps are submitted before either result is awaited, so the two services work at the same time.
- **Publishing:** each policy that stepped publishes a new sampler and counts one more version of its own. A policy
  that had no rows in a step (no rollout spawned a worker) does not step and keeps its version, and the step's metrics
  say so (`train/policies/worker/batch/skipped_optimizer`).

The step's metrics keep their names for `main`; the worker's go under `policies/worker/`, such as
`policies/worker/optim/lr` and `train/policies/worker/optim/...`. A failure of either service fails the step, as it
does for one policy, with an error that names the policy and its service; the run then resumes from its last
checkpoint.

## Checkpoints

Every checkpoint saves both policies: first the worker, into `runs/subagents/delegator-and-worker/policies/worker/`,
then `main`, into the run directory. `main`'s row names the worker's row by its `state_path` (`loop_state.policies`),
and it is written last, so it commits both. A resume starts from the newest `main` row whose worker row still exists:
a crash between the two saves leaves a worker row that no `main` row names, which the resume ignores. With
`checkpoint.keep_recovery`, each policy's recovery checkpoints are pruned on its own service, and the worker's pruner
keeps the rows that the newest `main` row and the newest `keep_recovery` `main` recovery rows name. A checkpoint saves
the worker also after a step in which no rollout spawned one; those saves hold weights no rollout samples, and its
pruner deletes them as it does any older checkpoint.

## Where the worker trains

`train.yaml` gives the worker no `tinker_base_url`, so it trains on the run's service, beside `main`, in a Tinker
session of its own. Hosted Tinker trains both adapters there. `train-worker-service.yaml` gives the worker a service of
its own, for a backend that serves one model or one adapter per service, such as a self-hosted service started for
each policy on its own nodes:

```yaml
policies:
  worker:
    tinker_base_url: http://localhost:8001
```

A job script that starts the worker's service passes its address on the command line,
`policies.worker.tinker_base_url=http://<host>:<port>`. The run's `identity.json` records each policy's model,
renderer and whether it trains, never its URL, so a resume may reach the services at new addresses; a resume that adds
or removes a policy, or trains one that was frozen, stops with exit code 78.

## Run it

```bash
export PYTHONPATH=$PWD/examples/subagents
uv run platoon config validate examples/two-agent-lockstep/train.yaml
uv run platoon train examples/two-agent-lockstep/train.yaml
```

The run writes to `runs/subagents/delegator-and-worker/`.

## How this example is tested

[`tests/examples/test_two_agent_lockstep_example.py`](../../tests/examples/test_two_agent_lockstep_example.py)
validates both configs and runs `platoon train` with each on CPU, on `platoon.testing`'s fake Tinker service (two of
them for the second config), with the subagents example's scripted policy sampling for both roles at a logprob of each
policy's own. It checks that both policies stepped in each step, each on its own rows only, that each checkpoint saved
both and named the worker's row, that a second `platoon train` resumes both from the last checkpoint, and that a
`platoon train` that freezes the worker of a run that trained it stops with exit code 78 before any Tinker session
opens.

The configuration keys are in the [config reference](../../docs/reference/config.md), and
[Train and evaluate](../../docs/how-to/train-and-eval.md#train-several-policies-in-lockstep-experimental) puts them in
context.
