# Write a step hook (experimental)

A step hook adds work at the end of each training step, after the step's own updates and before it publishes the new
policy: a supervised update on the policy that RL trains, a producer that writes training data with the live policy,
a receipt of the draws the step consumed. It runs in any trainer, the built-in `rl` trainer among them, which attaches
the hooks its config names. A hook needs no trainer of its own and copies none of platoon's code.

Step hooks live in `platoon.experimental` and are **experimental**: they are documented, and their names are in the
public API's snapshot, but a minor release may change them, and its CHANGELOG then says how. Pin the platoon version
you build against. [`examples/replay-sft-hook`](../../examples/replay-sft-hook/README.md) is a complete one, with a CPU
test.

Below, a *start* is one process of `platoon train` on a run directory, from its launch save to its exit: a resume, or
a supervised restart, is a new start of the same run. *Failing the start* ends that process with the error; under
`watchdog.max_restarts` the next start resumes from the run's newest full checkpoint.

## When a hook runs

At each step's end, in one sequence, the `rl` trainer's step and any other trainer's (`run.end_step()`) alike:

1. **drain**: each hook's `drain(end)`, in registration order. Side work that samples the current policy finishes or
   stops here: under a fully fine-tuned service, a sampler samples the newest weights, so nothing may sample once the
   hooks start updating.
2. **update**: each hook's `update(end)`, which may run optimizer updates of the policy through `end.train(...)` or
   `end.learner(...)`, on the same training client and Adam state as the step's own updates.
3. **publish**: each policy that the step or a hook updated publishes **once**: one save and one new version, however
   many updates the step and its hooks took. The hooks' updates spend no staleness budget: the horizon counts steps.
4. **published**: each hook's `published(end)`, also when nothing published (`end.published` is then empty), and once
   after the start's launch, with `end.kind == "launch"`.
5. The clock advances to the next step.

A run with no hook skips steps 1, 2 and 4: its steps are exactly what they were.

## Write one

```python
import platoon.experimental as px


class ReplaySFT(px.StepHookBase):
    name = "replay_sft"                    # its metrics' prefix (replay_sft/...), its waits' and its log lines' name

    async def update(self, end: px.StepEnd) -> None:
        if (end.step + 1) % 2:
            return                         # every other step
        datums = self.next_batch()         # tinker.Datum with target_tokens and weights
        await end.train("main", datums, objective=CrossEntropy(), adam_params=self.adam_params())  # None: the policy's
```

An objective is any class of `platoon.experimental.Objective` (a Tinker loss and the metrics of each microbatch):
platoon's cross-entropy is the sft trainer's, `platoon.train.objectives.cross_entropy.CrossEntropy`, and the example's
[`CrossEntropy`](../../examples/replay-sft-hook/replay_sft.py) is one to copy.

Every method is optional when the hook subclasses `StepHookBase`, whose methods are empty; a class of the `StepHook`
protocol itself defines all four (`drain`, `update`, `published`, `aclose`). What a hook reads of the step's end
(`StepEnd`):

| Field | What it is |
|---|---|
| `kind`, `step` | `"step"` and the step that ends (before the clock advances), or `"launch"` and the start's first step |
| `step_updates` | the step's own optimizer updates, by policy (the `rl` trainer's: its policy-gradient updates) |
| `hook_updates` | the hooks' updates so far, by policy, as platoon counts them |
| `consumed` | the draws the step committed, by key `(start, seq)`, each with its fate (`consumed`, `stale`, `failed`) |
| `published` | who published at this step's end, at which new version (in `published()`) |
| `tracker` | the step's metrics row: what a hook records here is logged with the step |
| `train(policy, datums, objective=..., adam_params=None)` | one optimizer update of the policy on the datums; during `update()` only |
| `learner(policy, objective=...)` | the policy's learner under the hook's objective, for updates of several microbatches; during `update()` only |
| `wait(awaitable, name=..., timeout=..., interruptible=False)` | the run's wait: a log line and a watchdog heartbeat every minute |

**Updates.** `end.train` sends the datums as given, in one forward_backward, so a hook weights tokens as it likes (zero
on a prompt, a mean over a batch), then steps the optimizer once, after the evaluation of the starting policy when
`eval.at_start` asks for one. `adam_params` takes the Adam parameters of that one update, a `tinker.AdamParams`
(`policy.optimizer_params.model_copy(update={"learning_rate": ...})`); without it the update steps with the policy's
own, `train.optimizer`'s. For an update of several microbatches, `end.learner(...).update()` gives a
`PolicyUpdate` (`platoon.train.policy.learner`): `submit` each microbatch, then `apply(adam_params=...)` once. A
hook updates through its step's end only: `run.learner` is its trainer's, and a learner of it sends nothing while a
step's end runs the hooks. A hook's learner sends and counts during the `update()` that took it only, refusing after,
in a later step too, and so does a learner made from it under another objective (`for_objective`): take one in each
`update()`. A policy takes one update at a time: an update is open from its first `submit` until
`apply` returns, and a second one sent meanwhile is refused (two `end.train` of one policy under `asyncio.gather`), as
is a step's end that finds one open, the step's or a hook's, an `apply` left running in the background among them.
Platoon counts each update that reaches the optimizer: the publish decision, `optim/updates_total` and the
checkpoints' `optimizer_steps` include the hooks', and the step's row gains `batch/hook_optimizer_updates`
(`batch/optimizer_updates` keeps counting the step's own). A hook's learner records its metrics under `<name>/`.

**What a hook works with.** These import Tinker, so `platoon.experimental` does not export them; import them under
`TYPE_CHECKING` to annotate a hook:

| Type | Where | What a hook uses of it |
|---|---|---|
| the run, `from_run`'s `run` | `platoon.train.trainer.run.TrainerRun` | `policies` (by id; `policies.leader` is the id of the policy the run trains), `config`, `run_dir` |
| a policy, `run.policies[id]` | `platoon.train.policy.policy.Policy` | `id`, `version`, `renderer`, `tokenizer`, `optimizer_params` (a `tinker.AdamParams`), `snapshot()` |
| a learner and its update | `platoon.train.policy.learner.Learner`, `PolicyUpdate` | `update()`; `submit(datums, originals, minibatch=, microbatch=)`, `apply(adam_params=)` |
| a refused update | `platoon.train.policy.learner.UpdateRejected` | raised by an update that fails closed; let it end the start |
| a stop | `platoon.experimental.TrainInterrupted` | raised by an interruptible wait; let it end the step |

**Side inference.** A producer samples the policy through a snapshot of its published sampler,
`policy.snapshot()` (`PolicySnapshot`): its version, renderer, tokenizer, model name and context window,
`sample(prompt, params)`, which raises `PolicyMoved` when the policy published before the call or while it ran, and
`gateway(context_window_length=...)`, a model gateway on that sampler with no recorder. Take a new snapshot in
`published()`, pause the producer in `drain()`. The check reads the policy's version, not its weights: under a fully
fine-tuned service the step's own optimizer steps change what the sampler samples before the step publishes, so a
sample is its version's only up to the step's first update.

## Keep its state across restarts

A hook with state to resume is also a checkpoint participant: `keys` (its entries of each full checkpoint's
`loop_state`), `async checkpoint_state(name)` (those entries) and `resume(start)` (take them up from
`start.loop_state`; `start.record` is None on a fresh run). Registering it as a hook registers it with the run's
checkpoints too. The rules:

- **Its keys are its own.** A key of the run's counters (`batch`, `policy_version`, ...), one kept for several
  policies, or one another participant holds is refused when it registers.
- **It resumes before the training client opens** when it registers in the trainer's `check()`, as the `rl` trainer's
  config-attached hooks do.
- **Nothing saved is dropped.** A key of the checkpoint a start resumes from that no part of the run claims stops the
  start (exit 78): resume with the hooks that saved it, or start a new run.
- **Every full checkpoint holds it**, the launch, cadenced, interrupted and final ones; a weights-only save holds
  none.

A hook (and a participant) may close what it opened in `async aclose()`: the run calls each, in reverse order of
registration, after the trainer's `close()`, also when the start failed. A start that failed raises its error as it
was, the close's errors noted on it.

## Attach it

**To the `rl` trainer, from the config:**

```yaml
trainer_options:
  hooks:
    - my_pkg.hooks:ReplaySFT                                  # an import path, or a name a package registered
    - {hook: my_pkg.hooks:Notes, options: {every: 5}}         # with options
```

The `rl` trainer builds each as it starts, before its training service opens, with its class's
`from_run(run, options)` (`StepHookBase`'s takes no options; override it to read some), and registers it. A hook that
does not import, build or register (a name another hook has, say) stops the run with exit code 78, naming its item of
`trainer_options.hooks`. The module must be importable where the trainer runs: install its package, or put it on
`PYTHONPATH`.

**From a trainer of your own**, register it in `check()` (or `prepare()`), and end each step with
`await run.end_step()`, which runs the sequence above, counting the step's own updates from the policies' optimizer
steps:

```python
class MyTrainer(px.TrainerBase):
    def check(self, run):
        run.hooks.register(self.notes)            # a participant too, when it has keys

    async def run_steps(self, run):
        learner = run.learner("main", objective=CrossEntropy())   # once the clock started: ready() starts it
        while run.clock.step < run.plan.end_step and not run.shutdown.is_set():
            update = learner.update()
            await update.submit(datums, datums, minibatch=0, microbatch=0)
            await update.apply()
            await run.end_step()                  # the hooks, one publication, the next step
```

A trainer whose policies nothing but a step hook samples (no rollouts, no sampling of its own, as the `sft` trainer)
ends its steps with `run.end_step(sampled=False)`: a step without a hook, or one that neither it nor a hook updated,
then publishes no sampler and saves only the full checkpoint the cadence asks for, while one with hooks publishes as
above.

A trainer that subclasses the `rl` trainer (`platoon.train.PlatoonTinkerRLTrainer`) registers its hooks in `check()`,
calling `super().check(run)` first, and keeps the `rl` trainer's steps as they are. The `rl` trainer reads the hooks
of the options it was built with: a mapping, read as `trainer_options`, or options of the subclass's own with a
`hooks` field (a sequence of `platoon.train.trainer.hook_config.HookSpec`); built without options, as `platoon train`
builds it, it reads the config's `trainer_options`.

## Failures and interrupts

- **A hook that raises in `drain` or `update` fails the start** before anything publishes, as a failed update does:
  nothing publishes and nothing saves of that step. A critical hook that raises in `published` fails the start after
  the step published, and a checkpoint the cadence saved then stays.
- **A hook that is not critical** (`critical = False`) only hears publications: it defines `published` (and
  `aclose`), not `drain` or `update`, and an error it raises there is logged while training goes on.
- **A stop request never saves half a step.** A wait that may take long passes `interruptible=True` to `end.wait`. When
  the trainer's steps are atomic, that wait raises `TrainInterrupted` once a stop is asked for, and the run saves
  nothing of the step. When they are not, the `rl` trainer's, the wait runs to its end, the step finishes its hooks
  and publishes, and the stop takes effect at the next step's boundary, where the run saves `<step>_interrupted`. The
  hook does not see the stop then: its wait runs to its end or to its `timeout` (`TimeoutError`, which fails the
  start), so bound what may not end by itself. An interrupt that comes as the hooks run all the same (a
  `TrainInterrupted` a hook raises, or an interruptible `run.wait` on the run `from_run` gave it) saves no checkpoint
  of the step either: the resume starts from the last full one, which a line of the log names.
- **Atomic steps.** A trainer's `needs()` (`TrainerNeeds`) says whether its steps reach a checkpoint whole
  (`atomic_steps`): once a step sent training work, an interrupt saves no `<step>_interrupted` checkpoint of it, and
  the resume starts from the last full one, which a line of the log names. A learner of the run's (`run.learner`,
  `end.learner`) says when a step sent work; a step that sends work by hand sets `run.clock.step_open = True`. And
  whether its updates fail closed (`fail_closed_updates`): an optimizer step that the service skipped, or whose metrics
  are not finite, raises `UpdateRejected` before it counts, the hooks' updates too. Both are on for a trainer on
  `TrainerBase`; the `rl` trainer keeps both off, as it always did, and a subclass may turn them on.

## The draws a step consumed

A trainer whose `needs()` set `draw_provenance` gives each training episode its draw, `ctx.draw`
(`platoon.experimental.DrawInfo`: the run directory, the draw's key `(start, seq)`, its position in the order and the
step current when its group started): an episode program may write files of its draw in the run directory, for a hook
to read. Evaluation episodes never carry it, and `ctx.draw` is None there. A step's end names the draws it committed
by the same keys (`end.consumed`); a key is never reused in a run, not even by a draw a resume makes again. Without the
need, the rollouts' requests are as they always were.

## Start a new trial from a checkpoint

`checkpoint.load_optimizer_state: true`, beside `checkpoint.load_checkpoint_path`, starts a new run from that
checkpoint's weights and its optimizer's state, as a new trial that goes on from it; its counters start at 0, and its
participants take up nothing (`start.record` is None).

## Test it

`platoon.testing.check_trainer(MyTrainer)` runs a trainer on a fake Tinker service, with a step hook of its own: it
checks that the hooks hear the launch and each step's end in order, that each step publishes once and counts the
hook's updates, and that a stop as the hooks drain a step keeps the step whole: nothing saved of it when the trainer's
steps are atomic, the step finished first when they are not ([Check a
trainer](../how-to/test-extensions.md#check-a-trainer)). A hook alone is tested on a step's end built for the test,
whose `end.train` and `end.learner` hand out a fake learner that records what the hook sends:

```python
class RecordingLearner:                        # what the hook sends, in place of a Learner
    def __init__(self):
        self.sent = []

    def update(self):
        return self

    async def submit(self, datums, originals, *, minibatch, microbatch):
        self.sent.append(len(datums))

    async def apply(self, tally=None, *, adam_params=None):
        self.sent.append("apply")


async def test_replay_sft_updates_at_every_other_step():
    learner = RecordingLearner()
    await ReplaySFT().update(px.StepEnd.for_test(step=1, step_updates={"main": 1}, learner=learner))
    assert learner.sent == [4, "apply"]        # one update of four conversations
```
