# Train two agents' policies each on its own clock

**Goal:** train two policies in one run, one per agent role, each stepping, publishing and saving on its own clock: the
agents the delegator spawns train on small batches and step about four times for each of the delegator's steps, and
neither waits for the other's training or saves.

The run trains the [subagents example](../subagents/README.md)'s delegator on its Vault tasks, with the two policies of
the [lockstep example](../two-agent-lockstep/README.md): the root agent samples `main`, the policy `train:` declares,
and the subagents it spawns sample `worker`. Here the two step independently (`train.stepping: independent`).

**Prerequisites:** the [subagents example](../subagents/README.md), whose environment and harness this example trains;
platoon installed with the `tinker` and `ors` extras; a Tinker API key in `TINKER_API_KEY` to train.

| File | What it is |
|---|---|
| [`train.yaml`](train.yaml) | the subagents example's training, with a second trained policy for the subagents, each policy on its own clock |

```bash
export PYTHONPATH=$PWD/examples/subagents
uv run platoon train examples/two-agent-independent/train.yaml
```

## Two clocks

```yaml
train:
  stepping: independent
  batch_size: 8                   # main's batch
  max_staleness: 1                # main's bound: a horizon of (1 + 1) x 8 = 16 groups
policies:
  worker:
    batch_size: 2                 # the worker's batch
    max_staleness: 7              # its bound: a horizon of (7 + 1) x 2 = 16
    checkpoint: {recovery_every: 4}
checkpoint:
  recovery_every: 1
```

Each group is credited once, over every agent, and then splits into a **share** of each policy: its rows of the group.
`main`'s steps take 8 shares each, the worker's 2, each policy on its own clock. Every group holds a share of `main`
(every rollout runs the delegator), and a share of the worker when some rollout of it delegated (the delegator may
answer alone), so the worker receives at most 8 shares for each of `main`'s steps and steps up to four times
meanwhile, publishing each time.

The worker's rows therefore age about four of its versions for each of `main`'s, and its bound grows with that:
`max_staleness: 7` gives it a horizon of 16 shares, `main`'s 16 groups, so its horizon does not hold the dispatcher
back. Each policy judges only its own rows: a worker share that is too old is dropped for the worker alone.

## Saves

Each policy saves on its own cadence, counted in its own steps: `main` at each of its steps (`checkpoint`), the worker
at every fourth of its own (`policies.worker.checkpoint`). Every full save of either is a **commit**, a point a resume
starts from: its row, in its policy's own `checkpoints.jsonl` (`main`'s in the run directory, the worker's in
`policies/worker/`, named with its id: `recovery-worker-<step>`), names the other policy's newest saved row and
records what each policy did with its share of every group that is still kept. A resume takes the commit with the
highest `commit_seq`; each policy goes on from the row it names, and each share a policy trained after that row trains
again, once.

## What to watch

Each step logs a row of its own: `main`'s with the names a run of one policy gives them, on the run's axis
(`run/step`), and the worker's under `policies/worker/`, on its own (`policies/worker/step`). Besides:

- `train/dispatch/blocked_share/horizon:<id>`, in `main`'s rows: the share of the step in which policy `<id>`'s horizon
  held the dispatcher back;
- `policies/<id>/share_wait_s`: how long a policy's step waited for its shares;
- `policies/<id>/peer_lag/<q>`: how far the other policy moved between the sampling of a policy's rows and their
  training;
- `run/retained_groups`: the groups kept until each policy that consumed its share has saved.

[Step each policy on its own clock](../../docs/how-to/train-and-eval.md#step-each-policy-on-its-own-clock-experimental)
describes the rest. A resume that changes `train.stepping` stops with exit code 78: the run keeps the stepping it
began with.
