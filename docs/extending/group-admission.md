# Write a group admission rule (experimental)

A group admission rule decides, when a training group settles, whether the group counts toward the batch. Platoon's own
rules are `all` (every group, the default) and `informative` (the groups whose workflow's verdict says they train, with
an optional pass-rate band); this page shows how to write your own, register it and test it. What admission does to a
run (refill, the cap, short steps, the metrics) is in
[Train only on groups that carry signal](../how-to/train-and-eval.md#train-only-on-groups-that-carry-signal-experimental).

The admission API lives in `platoon.experimental` and is **experimental**: it is documented, and its names are in the
public API's snapshot, but a minor release may change it, and its CHANGELOG then says how. Pin the platoon version you
build against. The [experimental API reference](../reference/experimental.md) lists every name with its signature.

## Where a rule sits

A config chooses a rule for the run, and an entry may choose its own:

```yaml
train:
  group_admission: informative           # the run's: all, informative, a registered name, module:Class
  refill: {max_groups_per_step: 3.0}
environments:
  - name: hard
    package: acme_rules.quota            # imported first, so its @group_admission names resolve
    group_admission: acme-quota          # this entry's own, with its own options
    group_admission_options: {limit: 4}
  - name: easy                           # the run's: informative
```

An entry that sets no rule takes the run's, with the run's options; an entry's options go with the rule it sets. Once
any entry's rule is not `all`, the run counts its steps as a rule-mode run does: a step trains the first `batch_size`
admitted groups it dequeues that are not stale, within the cap.

## The rule

A rule is a class with `admit(group, stats)`, one synchronous call per settled training group, which answers an
`Admission(admit, reason)` or a bare `bool`:

- `group` is the group's `GroupRecord`: its task, entry, labels, the sampler that drew it and why, its members'
  results (`MemberResult`: the trained and the scored reward, whether each was valid) and its workflow's verdict
  (`trained`, `uniform`, or why it trains nothing; `unknown` for a workflow that gives none). `group.kind(threshold)`
  splits a uniform group into all-pass, all-fail and mixed.
- `stats` is a read-only `TaskStatsView` of the run's per-task statistics, the view task samplers read
  ([Write a task sampler](task-samplers.md#the-statistics)): `counts`, `task`, `posterior`, `entry`, `label`,
  `threshold(entry)`.
- `Admission(True)` admits it (reason `admitted`); `Admission(False, "too_easy")` rejects it, with a reason: a
  non-empty string of at most 64 characters, which goes into the rejection's `fate` event in `tasks/events.jsonl` and
  the step's metrics (`train/admission/rejected/<reason>`; a short lower-case word becomes a metric key, the first 16 a
  rule gives, any other counts under `rejected/other`). `False` reads as `Admission(False)`, reason `rejected`.

It is built once per start of the process, `Class(setup, options)`: `setup` is an `AdmissionSetup` (its qualified
name, such as `acme-quota@hard` at the entry level, its level, its entry, the run's entries with their task ids, the
batch size and the label column), `options` its options parsed. Subclass `AdmissionRule` for the defaults.

## The rules a rule keeps

1. **Pure.** A rule is a function of `(group, stats, options)`: it keeps no state between calls. A resume builds a new
   instance, which must decide the groups that settle after it as the run would have; a count kept in the rule would
   start again from zero. History belongs in the statistics, which checkpoints save.
2. **Fast and synchronous.** `admit` is a plain function, called in the group's own coroutine on the event loop: keep
   a decision to a millisecond or two.
3. **Deterministic.** Never iterate a `set` of strings: its order changes from process to process.
4. **An answer the trainer takes.** Anything but an `Admission` or a `bool`, a reason outside the contract, or an
   exception admits the group (fail open: an admitted group that trains nothing costs one slot, while a wrong rejection
   could empty every step), logged once per rule and counted in `train/admission/errors`, which a run should keep at 0.

## What the view holds, and when

A rule reads the statistics as of their last flush, which comes at each step's metrics, each full checkpoint and each
decision of an adaptive task sampler: the groups that settled since then wait to be taken in, in the order they were
drawn, so a decision never depends on the order in which groups complete. The draws and their fates are current, but
the fates come at different times: `rejected` (and, under a rule, `failed`) when a group settles, `consumed` and
`stale` only when the step that dequeued the group commits. So a count of a task's consumed groups lags by the
admitted groups dequeued or waiting and not yet committed. An entry's or a label's totals (`GroupTotals`) carry no
fates; a task's record (`stats.task(task_id).fates`) does.

## Options and checks

A rule's options are `group_admission_options`. Its class names their schema in `options_schema`, or the decorator's
`options=`: a dataclass (each unknown key refused with a did-you-mean, each wrong type with its path, a `ValueError`
its `__post_init__` raises reported under the field it names), a pydantic model, or a function of the options'
mapping that returns them parsed and raises `ValueError`; without one, any option is refused. `lock.json` records each
rule other than `all` with its options, defaults included, and the checkpoints' sampling fingerprint holds them, so a
resume that changes one says so.

A static `check(options, ctx)` (or the decorator's `check=`) refuses a config the rule cannot serve: raise
`ValueError`, and `platoon config validate` and `platoon train` report it, with exit 78, before anything runs. Its
`ctx` (`AdmissionCheckContext`) holds the level, the entry, each entry's weight, seed, group size, success threshold
and sampler, the batch size, the label column, `max_groups_per_step` and whether the run's workflow gives verdicts.

## Register it, or name it

Register a rule under a short name with `@group_admission(name, options=..., check=...)` in a module an entry's
`package` names, so that it is imported before the rules resolve; or name it in the config as `module:Class`, with no
registration at all. Core's names (`all`, `informative`) are reserved: registering one is an error, whatever was
imported first, and so is registering a name twice. A config that names a rule no package could register is refused
when it loads; with a `package`, an unknown name is refused before the run starts, with the nearest name.

## Test it

`platoon.testing.check_group_admission` checks a rule against these rules, on a fixture of 40 tasks in two entries
with synthetic statistics and 400 settled groups of every verdict and shape (without valid members, without members,
without labels or an entry), at the run level and at the entry level, and raises `AssertionError` naming each check it
fails:

```python
from platoon.testing import check_group_admission


def test_the_quota_keeps_the_contract():
    check_group_admission("acme_rules.quota:Quota", options={"limit": 4})
```

Its checks are `protocol` (it builds at each level, with entries and without, and its `check` returns or raises
`ValueError`), `options`, `answers`, `pure` and `fast` (the median decision on the statistics of 4,096 tasks within
`budget_ms`, 2 ms by default). `pure` looks for state kept between calls: an instance asked about the groups twice, and
another asked in reverse order, answer alike; asked ten times over, the first answers as an instance built for each
group does, and over a second view of other statistics as a new instance does. Given by reference (a registered name of
an importable package, or `module:Class`), the rule's answers are also replayed in two processes with other hash seeds,
which catches iteration over a set; a class given directly skips that replay. `levels=("entry",)` checks a rule that
only an entry may set. Passing the check is no proof: a rule whose state shows only after more calls, or on other
inputs, passes it. See
[Test an environment, a harness or a program on CPU](../how-to/test-extensions.md#check-a-group-admission-rule).

## A worked example

This rule rejects a task's groups once it has `limit` consumed groups, a per-task quota that keeps a few easy tasks of
one entry from filling the batches. It counts committed groups, read from the task's fates, so it can overshoot its
quota by the task's admitted groups not yet committed: at most a step's `batch_size` and the admitted groups waiting.
Its check refuses it as the run's rule of a mixture, where it would hold back every entry's tasks alike.

```python
# acme_rules/quota.py
from dataclasses import dataclass

import platoon.experimental as pe


@dataclass(frozen=True)
class QuotaOptions:
    # The consumed groups of a task after which its groups are rejected.
    limit: int = 4

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError(f"limit: must be at least 1, got {self.limit}")


def _check(options: QuotaOptions, ctx: pe.AdmissionCheckContext) -> None:
    if ctx.level == "run" and len(ctx.entries) > 1:
        raise ValueError("a per-task quota holds back one entry's easy tasks: set it on that entry, not the run")


@pe.group_admission("acme-quota", options=QuotaOptions, check=_check)
class Quota(pe.AdmissionRule):
    """Rejects the groups of a task that has trained ``limit`` groups (its consumed fates)."""

    def admit(self, group: pe.GroupRecord, stats: pe.TaskStatsView) -> pe.Admission:
        try:
            consumed = stats.task(group.task_id).fates.get("consumed", 0)
        except KeyError:
            return pe.Admission(True)  # a task outside the statistics' pool: nothing to count
        if consumed >= self.options.limit:
            return pe.Admission(False, "over_quota")
        return pe.Admission(True)
```

An entry selects it with `package: acme_rules.quota` and `group_admission: acme-quota`, or with
`group_admission: acme_rules.quota:Quota` and no package. A rule that rejects the groups of a task whose posterior pass
rate is outside a band, or that admits a slow entry's groups whatever they say, has the same shape.
