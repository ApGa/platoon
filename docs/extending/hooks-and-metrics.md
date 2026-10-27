# Hooks, telemetry and metrics

**Goal.** Get numbers and signals from your extension into a run: report counters from your code, tell training
which of your harness's model calls failed or only sent a message, and get the run's metrics into the tools you
use. You will register a hook, check it on CPU, and learn where platoon's metrics go.

**Prerequisites.**

- Platoon installed (`pip install 'platoon[tinker,ors]'`, or `uv sync --extra tinker --extra ors` in a checkout).
- [Train and evaluate](../how-to/train-and-eval.md#metrics-to-watch): the metrics a training step logs.

## Hooks

Core never imports your extension. At four fixed seams it calls the functions registered under a hook name instead,
in the order they were registered:

| Hook | Signature | Called | Effect |
|---|---|---|---|
| `rollout_worker.telemetry` | `fn() -> Mapping[str, float]` | after each training step in the trainer, and when a rollout worker finishes | Counters of the current process, logged with the step's metrics |
| `rollout_worker.diagnostics` | `fn() -> Mapping[str, str]` | while a rollout worker runs (`rollout_isolation: process`) | Short strings the worker writes into the diagnostic marker it leaves if it is killed, for example while hung |
| `trajectory.error_completion_ids` | `fn(steps) -> set[str]` | when a trajectory becomes training data, with `filter_errors` on (the default) | The completion ids of the model calls whose actions failed, besides the ones platoon detects itself; their tokens get no positive advantage |
| `trajectory.message_completion_ids` | `fn(steps) -> set[str]` | when a trajectory becomes training data, with `filter_message_actions` on | The completion ids of the model calls that only sent a message and ran no action; they stay in the context but train no loss |

`steps` are a trajectory's steps as dicts, one per model call; a step's completion id is
`step["misc"]["action_misc"]["completion_id"]`, and its other keys are what the harness annotated with
`ctx.annotate_step(...)`.

### Register a hook

Register at import time, in the module that defines your component:

```python
# my_hooks/counters.py
import platoon as pt
from platoon.hooks import register

COUNTS = {"my_harness_retries": 0.0}


def telemetry() -> dict[str, float]:
    return dict(COUNTS)


def failed_completions(steps: list[dict]) -> set[str]:
    """The model calls that my-retrying marked as failed, with ctx.annotate_step(my_failed=True)."""

    return {step["misc"]["action_misc"]["completion_id"] for step in steps if step.get("misc", {}).get("my_failed")}


register("rollout_worker.telemetry", telemetry)
register("trajectory.error_completion_ids", failed_completions)


@pt.harness("my-retrying")
async def retrying(ctx: pt.AgentContext) -> pt.AgentResult:
    """Ask once more when the first reply is empty."""

    messages = [{"role": "user", "content": ctx.instruction}]
    reply = await ctx.chat(messages)
    if not reply.text.strip():
        ctx.annotate_step(my_failed=True)
        COUNTS["my_harness_retries"] += 1
        reply = await ctx.chat(messages)
    return pt.AgentResult("completed", final_message=reply.text)
```

Registering the same function twice keeps one registration, so a module imported twice does not double its effect.

**Make every process import it.** A hook exists only in processes that imported the module that registers it.
Register hooks in the module that defines your harness, as above: every process that runs the harness imports it.
Hooks that live in a module of their own need the environment entry's `package:`, which the trainer, evaluation
subprocesses and rollout workers import when they start:

```yaml
environments:
  - service: python://guess:GuessTheNumber
    harness: my-retrying
    package: my_hooks.counters
```

### Rules

- **Detect only your own steps.** The trajectory hooks run on every trajectory of every harness in the run. Mark
  your harness's steps with `ctx.annotate_step(...)`, as above, and return ids only for steps that carry your
  marks; a harness mixture would otherwise suppress another harness's tokens.
- **Never raise from a trajectory hook.** It runs while the rollout becomes training data; return an empty set
  when in doubt.
- **Telemetry and diagnostics are best effort.** A telemetry or diagnostics function that raises contributes
  nothing; it never fails a step or a rollout.
- **Telemetry and process isolation.** With `rollout_isolation: inprocess` (the default), every counter is logged
  with each step's metrics under its own name. With `rollout_isolation: process`, rollouts run in worker processes
  and the trainer adds up only a fixed set of counter names across them
  (`CUMULATIVE_KEYS` in [`platoon/train/workers/telemetry.py`](../../platoon/train/workers/telemetry.py));
  other names are not reported from workers.

### Check it on CPU

Hooks are plain functions; call them as platoon does:

```python
# tests/test_my_hooks.py
import my_hooks.counters  # noqa: F401  (registers the hooks)
from platoon.hooks import hooks
from platoon.utils.trajectory_error_filtering import detected_error_completion_ids


def test_the_failed_turn_is_detected():
    steps = [
        {"misc": {"action_misc": {"completion_id": "call-1"}, "my_failed": True}},
        {"misc": {"action_misc": {"completion_id": "call-2"}}},
    ]
    assert detected_error_completion_ids(steps) == {"call-1"}


def test_the_counter_is_reported():
    assert {"my_harness_retries": 0.0} in [snapshot() for snapshot in hooks("rollout_worker.telemetry")]
```

```bash
PYTHONPATH=. uv run pytest tests/test_my_hooks.py -q
```

```text
..                                                                       [100%]
2 passed in ...s
```

## Metrics

A training run logs each step's metrics to the console and, unless `stats.wandb.mode` is `disabled`, to
[Weights & Biases](https://wandb.ai). Evaluation writes its report into `inference.output_dir`, by default a new
`runs/eval/<policy>/<UTC time>/` for each start ([Train and evaluate](../how-to/train-and-eval.md)).

| You want | Do |
|---|---|
| A training reward broken into parts | Return components named `reward/<name>` from a [reward processor](rewards.md); each step logs their averages (`train/batch/reward/<name>/avg`), and evaluation reports them per rollout |
| Counters from your code | The `rollout_worker.telemetry` hook (above) |
| Per-step notes for reading trajectories | `ctx.annotate_step(...)` and `ctx.annotate(...)`: kept in `trajectory_collection.json` and shown by `platoon viz`, never logged as metrics or trained |
| Metrics without network access | `stats.wandb.mode: offline`: W&B writes the run into `<log_path>/<experiment_name>/<trial_name>/wandb/`, and `wandb sync` uploads it later |
| Metrics in another tracker | Not pluggable yet: platoon logs only to the console and W&B. Read W&B's offline files, or propose a sink interface in an [RFC](../rfcs/README.md#how-to-propose-a-change) |

The run directory also keeps what you need to recompute metrics offline: `checkpoints.jsonl`, the rollouts' event
files under `rollouts/train/<policy version>/events/` (which `platoon viz replay` reads), and, for evaluation,
`reports/final_report.json` and `reports/task_results.jsonl`.

## Next steps

- [Reward processors and finalizers](rewards.md): report reward components as metrics.
- [Train and evaluate](../how-to/train-and-eval.md#metrics-to-watch): the metrics that say whether a run is
  healthy.
- [Extending platoon](index.md): every extension point and how platoon finds it.
