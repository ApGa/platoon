# Write a task sampler (experimental)

A task sampler chooses which training task each rollout group runs. Platoon's own samplers are `weighted`
(each slot of the order to an environment entry by weight), `uniform` (seeded shuffled passes) and `pass_rate`
(the tasks whose next group is most likely to mix passes and failures, from the run's per-task statistics); this page
shows how to write your own, register it and test it.

The sampler API lives in `platoon.experimental` and is **experimental**: it is documented, and its names are in the
public API's snapshot, but a minor release may change it, and its CHANGELOG then says how. Pin the platoon version you
build against. The [experimental API reference](../reference/experimental.md) lists every name with its signature.

## Where a sampler sits

A config chooses samplers at two levels:

```yaml
train:
  task_sampler: weighted        # the run's: auto, weighted, uniform, pass_rate, a registered name, module:Class
environments:
  - name: hard
    package: acme_curricula.samplers   # imported first, so its @task_sampler names resolve
    task_sampler: acme-skip-solved      # this entry's: uniform (default), pass_rate, a registered name, module:Class
    task_sampler_options: {keep: 0.5}
  - name: easy                          # uniform
```

- **The run's sampler** (`train.task_sampler`) decides each position of the run's order. `weighted`, and `auto` for a
  mixture, hand the position to an entry, as a **slot**; the entry's own sampler then chooses which of its tasks fills
  it. So each entry keeps its share of the slots whatever its sampler does with them: the form to use for an adaptive
  sampler.
- **An entry's sampler** (`environments[].task_sampler`) chooses among that entry's tasks, for the slots it is handed.
  A run sampler that hands no slots (`uniform`, `pass_rate`, or a custom one without `entry_slots`) asks no entry's
  sampler, and a config that sets one under it is refused.

A sampler of yours can serve either level; only a static run sampler may hand slots (`entry_slots = True`).

## Static and adaptive

A sampler's class says which it is, with `adaptive`:

- **Static** (`adaptive = False`): a function of its decision index alone. The dispatcher asks it when it draws a new
  position, as the built-ins always were, and its draw may wait for its label (`train.rollout_group_limits`). It sees
  no statistics: its context's `stats` is None.
- **Adaptive** (`adaptive = True`): it reads the run's per-task statistics, so it decides only when a group can start:
  within the horizon, with its group's rollout permits held, and with some label of its pool below its limit. Its
  context's `eligible(task_id)` says which tasks' labels have room, and it must choose among those. Under `weighted`, a
  slot owed to an adaptive entry waits as a token in the entry's queue until the entry can start a group.

## The rules

Each decision is one call of `next(ctx)`, which returns a `Draw(task_id, reason, score)`, a bare task id, a
`Slot(entry)` (a static run sampler with `entry_slots`), or None.

1. **A task of its pool, or nothing.** Answer a task of `ctx.task_ids` (for an adaptive sampler, one that
   `ctx.eligible` accepts), and None only when nothing is eligible. A static sampler never answers None.
2. **Fast and synchronous.** `next` is a plain function, and runs on the event loop: keep a decision to milliseconds,
   since a step that starts makes up to a batch of decisions in a row, with nothing else running. Read
   `ctx.stats.counts(task_id)` (a task's counts and streaks, read without building anything) for every task, take the
   numbers you need from it and let it go (kept for every task of a large pool through a decision, such objects make
   the garbage collector's pauses long), and read `ctx.stats.task(task_id)` (the whole record) for a few.
3. **Deterministic.** Take randomness from `ctx.rng`, a `random.Random` keyed by the sampler's seed (`setup.seed`: the
   entry's `seed` at the entry level, 0 at the run level, plus an integer `seed` of its options when they have one, so
   do not add that again), its name and its decision index, or seed your own from `ctx.key` (a 128-bit key:
   `numpy.random.Generator(numpy.random.Philox(key))` for a sampler that brings numpy). Never iterate a `set` of
   strings: its order changes from process to process.
4. **Static means static.** A static sampler depends on nothing but its decision index and its state: not on
   `ctx.step` or `ctx.versions`, since a resume recomputes where it stands from the position alone.
5. **A small, complete state.** `state_dict()` returns JSON data with string keys, saved with each full checkpoint; a
   fresh instance that loads it answers the next decisions alike. `load_state_dict(state)` drops tasks that left the
   pool, and raises `ValueError` for a state it cannot read (the dispatcher then starts the sampler afresh, with a
   warning). `{}` starts afresh.
6. **Groups in draw order.** An adaptive sampler that defines `observe(group)` is handed each settled group of its pool
   once, as a `GroupRecord`, in the order the groups were drawn, before its next decision and before every save. It
   must also work without them.

The sampler's decision index, `ctx.decision`, advances whenever one of its decisions binds a draw (its own, or the
fallback below), never for a None while nothing is eligible. The dispatcher keeps it, and saves it.

## What the dispatcher refuses

An answer outside the contract never draws: a task outside the sampler's pool, a task whose label is full from an
adaptive sampler, a `Slot` the sampler may not give, a `Draw` whose `reason` is not a non-empty string of at most 64
characters or whose `score` is neither None nor finite, None while something is eligible, and an exception from
`next`. The decision then gets a **fallback draw**: of the sampler's eligible tasks in listing order, the one at offset
`ctx.key` modulo their number (for a sampler that hands slots, a slot to an entry chosen by weight). The fallback uses
the decision up, its reason is `fallback`, each kind of failure is logged once per sampler, and every fallback counts
in the step metrics (`train/sampler/.../draws/fallback`). A run should keep that count at 0.

A draw's `reason` and `score` go into the group's record and its draw event in `tasks/events.jsonl`, and the reasons
into the step metrics (`train/sampler/.../draws/<reason>`): a short lower-case word (`[a-z][a-z0-9_]*`, at most 32
characters) becomes a metric key, the first 16 of them a sampler gives; any other counts under `draws/other`.

## Metrics

`metrics(stats)` returns the sampler's own metrics of the step, a mapping of names to finite numbers, which every
step logs under the sampler's prefix: `train/sampler/` for the run's sampler, `train/sampler/environment/<entry>/` for
an entry's. It is called once a step, with the statistics' view for an adaptive sampler and with None for a static one,
and covers the decisions since the last call. The names the dispatcher writes itself under that prefix are reserved:
`draws` and `draws/<reason>`, `decision_ms`, `decision_ms_max`, `tokens`, `skipped_slots`, `predicted_informative` and
`informative_share`. A sampler's metric of a reserved name is left out, with an error logged once, so that
`draws/fallback` always counts the fallbacks the dispatcher drew; `check_task_sampler` refuses such a name.

## The statistics

An adaptive sampler's `ctx.stats` is a read-only `TaskStatsView` of the run's per-task statistics at the current
policy version (`ctx.stats.now`):

| Read | What it gives |
|---|---|
| `counts(task_id)` | `groups`, `outstanding`, `drawn`, `uniform_streak`, `no_evidence_streak`, `evidence_version`: the cheap read |
| `task(task_id)` | the whole `TaskStats`: attempts, passes, kinds, fates, the newest groups, decayed means |
| `posterior(task_id)` | `(alpha, beta)` of the task's posterior pass rate |
| `entry(name)`, `label(label)`, `labels()` | `GroupTotals` of an entry's or a label's groups |
| `group_size(entry)`, `threshold(entry)` | an entry's training group size and the reward that counts as a pass |
| `live` | False when the statistics stopped, or never started: then it reads what they held, or nothing |

A decision costs time in proportion to the pool whatever else it does, since it tests each task's eligibility: read
`counts` for every task, and `task` only for the few you need. Both raise `KeyError` for a task the statistics do not
hold.

## Options and checks

A sampler's options are `task_sampler_options` in the config. Its class names their schema in `options_schema`, or the
decorator's `options=`: a dataclass (each unknown key refused with a did-you-mean, each wrong type with its path, and a
`ValueError` its `__post_init__` raises reported under the field it names), a pydantic model, or a function of the
options' mapping that returns them parsed and raises `ValueError`. The sampler is built as `Class(setup, options)`,
`options` parsed (None without a schema), once per start of the process. `lock.json` records each sampler with its
options, defaults included.

A static `check(options, ctx)` (or the decorator's `check=`) refuses a config the sampler cannot serve: raise
`ValueError`, and `platoon config validate` and `platoon train` report it, with exit 78, before anything runs. Its
`ctx` (`SamplerCheckContext`) holds the level, the entry, the run's sampler and each entry's weight, seed, group size,
success threshold and sampler.

## Register it, or name it

Register a sampler under a short name with `@task_sampler(name, options=..., check=...)` in a module an entry's
`package` names, so that it is imported before the samplers resolve; or name it in the config as `module:Class`, with
no registration at all. Core's names (`auto`, `weighted`, `uniform`, `pass_rate`) are reserved: registering one is an
error, whatever was imported first, and so is registering a name that reads as one of them whatever its case and its
`-` or `_` (`Pass-Rate`), or a name twice. A config that names a sampler no package
could register is refused when it loads; with a `package`, an unknown name is refused before the run starts, with the
nearest name.

## Test it

`platoon.testing.check_task_sampler` checks a sampler against these rules, on a fixture of 40 tasks in two entries with
synthetic statistics (an adaptive sampler under three eligibility patterns, and under the three in turn; a static one
with every task eligible), and raises `AssertionError` naming each check it fails:

```python
from platoon.testing import check_task_sampler


def test_skip_solved_keeps_the_contract():
    check_task_sampler("acme_curricula.samplers:SkipSolved", options={"keep": 0.5}, levels=("entry",))
```

Its checks are `protocol`, `options`, `pool-task-or-nothing`, `fast` (the median decision on a pool of 4,096 tasks
within `budget_ms`, 20 ms by default), `deterministic`, `static` (also at another training step and policy version),
`state` (also after `observe` handed it groups), `observe` and `metrics`. Given by reference (a
registered name of an importable package, or `module:Class`), the sampler's decisions are also replayed in two
processes with other hash seeds, which catches iteration over a set; a class given directly skips that replay. See
[Test an environment, a harness or a program on CPU](../how-to/test-extensions.md#check-a-task-sampler) for the rest of
the kit. A test that needs statistics of its own builds a `TaskStatsStore` and feeds it `GroupRecord`s.

## A worked example

This sampler draws an entry's tasks in shuffled passes, but skips a task whose last groups all scored alike (all passed
or all failed), with a chance that halves with each such group in a row:

```python
# acme_curricula/samplers.py
import random
from dataclasses import dataclass

import platoon.experimental as pe


@dataclass(frozen=True)
class SkipSolvedOptions:
    # The chance of keeping a task, per group in a row whose rollouts all scored alike.
    keep: float = 0.5
    # The longest such run that lowers it further.
    max_streak: int = 4

    def __post_init__(self) -> None:
        if not 0 < self.keep <= 1:
            raise ValueError(f"keep: must be above 0 and at most 1, got {self.keep}")
        if self.max_streak < 1:
            raise ValueError(f"max_streak: must be at least 1, got {self.max_streak}")


@pe.task_sampler("acme-skip-solved", options=SkipSolvedOptions)
class SkipSolved(pe.Sampler):
    """Shuffled passes over the entry's tasks, skipping a task whose recent groups all scored alike."""

    adaptive = True  # it reads the statistics, so it decides when a group can start

    def __init__(self, setup: pe.SamplerSetup, options: SkipSolvedOptions | None = None) -> None:
        super().__init__(setup, options or SkipSolvedOptions())
        self.passes, self.order = 0, []

    def next(self, ctx: pe.SamplerContext) -> pe.Draw | None:
        first = None  # the first task that can run this call met, drawn if the coin skips every one
        for _ in range(2 * len(ctx.task_ids)):  # the rest of this pass and a whole new one: every task once at least
            if not self.order:
                self.order = list(ctx.task_ids)
                random.Random(f"{self.setup.seed}:{self.passes}").shuffle(self.order)
                self.passes += 1
            task = self.order.pop()
            if not ctx.eligible(task):
                continue  # it cannot run now, and loses its turn in this pass
            first = first or task
            streak = ctx.stats.counts(task).uniform_streak  # the cheap read
            if ctx.rng.random() < self.options.keep ** min(streak, self.options.max_streak):
                return pe.Draw(task, reason="kept" if streak else "order", score=float(streak))
        if first is None:
            return None  # nothing can run now: it is asked again when something changes
        return pe.Draw(first, reason="unkept")  # never None while a task can run (rule 1)

    def state_dict(self) -> dict:
        return {"passes": self.passes, "order": list(self.order)}

    def load_state_dict(self, state) -> None:
        if not state:
            return  # afresh
        passes, order = state.get("passes"), state.get("order")
        if not isinstance(passes, int) or not isinstance(order, list):
            raise ValueError(f"not a SkipSolved state: {dict(state)!r}")  # rule 5: the dispatcher starts it afresh
        pool = dict.fromkeys(self.setup.task_ids)
        self.passes, self.order = passes, [task for task in order if task in pool]
```

An entry selects it with `package: acme_curricula.samplers` and `task_sampler: acme-skip-solved`, or with
`task_sampler: acme_curricula.samplers:SkipSolved` and no package. A sampler for "tasks whose recent mean reward lies in
a band", or one that prefers cheap informative tasks, has the same shape: it reads `ctx.stats.task(task_id).means` or
the newest groups (`history`) of the few tasks it considers.
