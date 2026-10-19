# RFC 0019: Training mixtures: per-environment settings, task statistics and samplers

| | |
|---|---|
| Status | Draft |
| Created | 2026-10-19; revised 2026-10-20 |
| Authors | Apurva Gandhi (@ApGa) |
| Builds on | [RFC 0018](0018-configuration-only-projects.md), section 8 (the weighted fair order, `weight: tasks` and the sampling plan); [RFC 0016](0016-component-configuration.md) (component options, and the rollout settings of each phase, which section 2 of this RFC amends); [RFC 0013](0013-extension-model.md) (registration by name); [RFC 0017](0017-public-surface-and-conformance.md) (experimental API, conformance kits); [RFC 0003](0003-multi-policy-training.md) and [RFC 0001](0001-native-multi-agent.md), for what must stay compatible; the trainer (`platoon/train/tinker/rl.py`) and the group workflow (`platoon/train/tinker/workflows/group_rollout_workflow.py`) |
| Discussion | The pull request that adds this RFC, which also records the decisions of 2026-10-20 (open questions 1 to 3 and 7 to 11) |
| Implementation | M0b, M1 and M2 in the three pull requests that follow this one; none merged yet |

## Summary

A training run draws its tasks through a **task sampler**, and a mixture's environments may each choose the training
settings of their own groups. Platoon also keeps, for every training run, **per-task statistics**: what each group of
each task did, what passed, and what that says about the task's pass rate. They are the record that adaptive samplers
read, and a measurement of pass@k during training for every run.

**What lands first** (platoon 0.3.0, three pull requests, each mergeable on its own):

- **M0b, a name.** `train.task_order` becomes `train.task_sampler`, with the same values; the old name stays a
  deprecated alias until 0.4.0.
- **M1, per-environment settings.** An `environments` entry may set its own `workflow_config` (group size, baseline,
  minimum group size, straggler defaults, error filters, subagent datum sampling and rollout settings), its evaluation
  overrides of them, and its staleness bound. Group sizes may differ, so `max_concurrent_rollouts` becomes a budget of
  rollouts that groups share. A label at its `rollout_group_limits` no longer stalls the others.
- **M2, per-task statistics.** Every settled group is recorded with its workflow's verdict and what each rollout
  returned; passes are judged as `platoon eval` judges success. Each step logs each label's expected pass@1 and pass@G;
  each full checkpoint saves the statistics and a resume loads them; `platoon tasks show` and `export` read them.

**What comes after** (M3-M5): a dispatcher that draws a task only when its group can start, with exact resume; the
sampler API, with an adaptive `pass_rate` sampler that prefers tasks whose next group is likely to train; and group
admission with refill. Each ships experimental, and the adaptive parts only after an A/B test.

## Motivation

**Most groups can train nothing.** Under platoon's group baselines (the group mean, or leave-one-out), a group whose
valid rewards are all equal has a zero advantage everywhere, so it trains nothing, while it still costs a batch slot
and its rollouts. In a 60-step run of a large model on a few thousand agentic tasks, about four groups in five scored
the same in every rollout, most of them passing every time. Filtering such groups after the fact (as DAPO does) costs
`1/P_inf` groups per kept one, where `P_inf = 1 - p^G - (1-p)^G` is the chance that a group of `G` rollouts mixes
outcomes at pass rate `p`: three groups per kept group at `p = 0.05` or `0.95`, thirteen at `0.01`. With rollouts of
minutes to hours, choosing tasks before their rollouts pays more than discarding groups after them, and choosing needs
a record of what each task did. Today there is none: a group's output carries a label but no task id, and the step
metrics average rewards over the batch.

**A mixture's environments want different settings.** Consider a mix of a quick dataset, calibrated with a 30-step
budget and ten-minute rollouts, and a slow one, whose rollouts take hours. A config cannot say what such a run needs:

1. **One step limit for both.** `rollout_config.max_steps` is run-wide.
2. **One staleness bound for both.** At a bound of 3 every slow group is dropped as stale; at 48 the quick data is far
   off-policy too, and recovery pruning must keep 48 versions of checkpoints.
3. **One group size.** Eight multi-hour rollouts per slow group, where four would do.
4. **A label limit stalls the other label.** `rollout_group_limits: {slow: 1}` parks the slow rows in the one pending
   queue of `batch_size` rows; once they fill it, the producer blocks and the quick rows stop too.

**Nothing learns per task, and a restart loses work.** A resume starts the producer again at the checkpoint's batch:
groups that earlier steps had already prefetched and consumed run again, and the groups in flight at the checkpoint
never train.

## Proposal

### 1. The task sampler

`train.task_sampler` names how a run draws its training tasks. Its values are `train.task_order`'s, unchanged:

| `train.task_sampler` | A mixture | A single environment |
|---|---|---|
| `auto` (default) | the weighted fair order of RFC 0018, section 8 | shuffled passes by the dataloader |
| `weighted` | the weighted fair order | the weighted order, as a mixture of one |

The rename comes first because later milestones extend the choice: a run-level `uniform` (every task of every entry
equally often, with no entry slots), an entry-level `task_sampler` that chooses which of the entry's own tasks comes
next (`uniform`, today's seeded passes, or the adaptive `pass_rate`), and custom samplers by registered name or
`module:Class`, each with typed options (`task_sampler_options`, RFC 0016). Under `weighted`, the run's sampler hands
each slot to an entry, and the entry's sampler fills it.

`train.task_order` is a deprecated alias until 0.4.0: it loads with a warning and is read as `task_sampler`;
`lock.json` records it; `platoon config validate --strict` fails on it. A config may set both names when they agree,
or when `task_order` is `auto`, its default before the rename, as a dump of 0.2.0's `config show --defaults` spells
it: `task_sampler` decides, and when the two differ, the warning says that `task_order` was ignored.
`task_order: weighted` beside another `task_sampler` is a conflict, and the config does not load (decided 2026-10-20,
open question 10). `task_sampler` defaults to `null`, read as `auto`, so that loading can tell a value a config sets
from the default.

### 2. Per-environment training settings

An entry may set the settings that act on its own groups, laid over the run's:

| An entry's key | What it sets | Default |
|---|---|---|
| `workflow_config` | its training groups' settings, laid over `train.workflow_config`: `group_size`, `leave_one_out_baseline`, `min_successful_group_size`, `straggler_timeout_seconds`, `straggler_quorum`, `filter_errors`, `filter_message_actions`, the subagent datum sampling, and `rollout_config`'s `max_steps`, `timeout`, `step_timeout`, `inference_params`, `sampling_precedence`, `extra` and the subagent reward settings | the run's |
| `eval_overrides.workflow_config` | the same keys, for its evaluation rollouts (every split but `train`, as its other `eval_overrides`) | none |
| `max_staleness` | the stale filter's bound for its training groups, at least `train.max_staleness` | the run's |
| `success_threshold` (M2) | the scored reward that counts as a pass in the statistics (section 3.2) | 1.0 |

**What stays the run's.** Keys that act on microbatches, which mix entries' groups (depth weighting, the
zero-advantage filter), the straggler policies by label, the rollout-worker settings (one pool runs every entry's
rollouts), and what platoon sets itself (`judges`, `model_name`, `output_dir` and the like). An entry that sets one
does not load, and the message says where it goes.

**The layers**, lowest first, merged by the loader's rules for a section laid over another (a key replaces, `null`
included; `rollout_config` and `inference_params` merge key by key; a free-form mapping such as `extra` replaces
whole):

| An entry's training groups | Its evaluation during `platoon train` | Its rollouts in `platoon eval` |
|---|---|---|
| defaults | evaluation's defaults | `RolloutConfig`'s defaults |
| `train.workflow_config` | the entry's training settings, for the keys evaluation takes from training | the entry's `workflow_config.rollout_config` |
| a named environment's `workflow_config.rollout_config`, then the entry's `workflow_config` | the keys `eval.workflow_config` sets | the keys `inference.workflow.rollout_config` sets |
| | the entry's `eval_overrides.workflow_config` | the entry's `eval_overrides.workflow_config.rollout_config` |

So the phase's section wins over an entry's settings, which are that phase's defaults for the entry, and the entry's
`eval_overrides` win over both. This amends RFC 0016, section 2, which made the step limit the phase's setting alone:
a dataset's calibrated step budget can now travel with its entry, while a run-wide evaluation setting still decides
for every entry that does not override it. A named environment may ship `workflow_config.rollout_config` only, as its
entry's lowest layer: the rest of `workflow_config`, `eval_overrides.workflow_config` and `max_staleness` are the
run's to choose. It ships no rollout settings for its evaluations alone either, which would need a layer beneath the
phase's section (decided 2026-10-20, open question 9).

**Staleness.** `train.max_staleness` stays the bound on how far rollouts run ahead of training, and the bound of each
entry that sets none. An entry's own bound may only raise it, at the stale filter: a lower bound would have the
entry's groups drawn as far ahead as every other entry's and then dropped. Recovery pruning keeps the checkpoints
within the largest bound of the run and its entries, since a slow entry's group may still sample from one. An entry
that raises its bound without a limit on its label gets a warning: its groups count toward the lookahead for as long
as they run, and enough of them slow every step to their pace.

**Concurrency.** `max_concurrent_rollouts` counts rollouts. A group takes its entry's group size of them at once, first
come first served, so groups of different sizes share the budget and never exceed it. Loading refuses a budget that
holds no group of the largest entry, and warns of one that the capped labels' groups can fill: a larger group would
wait, first in line, for a capped group to end. That one stays a warning, not an error: it slows the run to the
capped labels' pace but does not stop it (decided 2026-10-20, open question 7).

**A capped label never stalls the others** (on today's producer and queue, until M3 replaces them): the rows of a label
at its limit wait beside the batch's rows rather than among them, and do not count toward the staleness lookahead;
past four batches of them, the label's new rows are skipped and counted, while another label can start a group. So a
label whose limit cannot keep up with its share of the order trains less than its weight, and the metrics say so.
These rules apply by default to every config that sets `rollout_group_limits`, those written before them included
(decided 2026-10-20, open question 3). At the end of a run, the rows still waiting never train: the producer replaces
them with rows past the run's planned batches, so that the last steps do not wait for the capped label, and a waiting
row whose label frees before the run ends still runs, as a group no step consumes. Those extra groups are accepted
until M3 (decided 2026-10-20, open question 11).

**Applying them.** Inside a group, `GroupRolloutWorkflow.config` is the settings of the group's entry (a `ContextVar`
set from the row), and `current_entry()` names it, so a subclass's `compute_baselines` reads its entry's settings
without changes. A workflow class that is no `GroupRolloutWorkflow` must declare that it applies an entry's settings,
or a config whose entries set any does not load. `platoon config show` prints each entry's effective settings with
the layer each came from, and `lock.json` records what differs from the run.

### 3. Per-task statistics

#### 3.1 What a group says

`GroupRolloutWorkflow`'s output gains `members`, what each rollout returned (its trained reward, the reward its
environment scored, whether it was valid, its outcome, its disposition, its steps, tokens and seconds, and the policy
versions its datums sampled), and `verdict`, the workflow's own account of what the group trains: `trained` (it carries
signal: an informative group), `uniform` (every retained advantage is zero), or why it trains nothing
(`reward_key_mismatch`, `no_results`, `below_min`, `no_completed_root`, `no_datums`), `failed` when the workflow raised
or returned nothing, and `unknown` for a workflow that gives no verdict. Taking the workflow's verdict, rather than
recomputing reward variance, keeps the definition right for leave-one-out baselines, subagent credit and, later,
role-aware credit (RFC 0001). A uniform group is all-pass, all-fail, or mixed by success; with `trained`, these four
kinds partition the groups a step trains on.

#### 3.2 What passes

A member passes when it is valid (its root completed with a scored outcome, so it entered the baseline) and the reward
its environment scored, before any reward processor, reaches its entry's `success_threshold`: `platoon eval`'s rule.
Judged on the trained reward instead, a shaping processor that trains 0.98 for a correct rollout would make every
group all-fail. Members without a valid result (errors, timeouts, cancelled stragglers, dead workers) are censored:
neither passes nor failures, as the baseline leaves them out. Censoring leans optimistic, since a cut-off straggler is
more likely a slow failure than a slow pass; it stays the rule, so that the statistics count the members the baseline
counts (decided 2026-10-20, open question 1).

#### 3.3 The record

`platoon/train/task_stats.py` keeps one record per training task, and totals per entry and per label: groups,
attempts, passes, reward, the kinds of groups, the current runs of uniform groups and of groups without a valid member
(three in a row flag a task without evidence: a broken grader or setup), decayed means of steps, tokens, seconds and
reward, the newest eight groups' summaries, and what became of each draw (`consumed`, `stale`, `failed`, `skipped`,
`lost`). Every settled group counts once, stale ones included: the work happened, and a uniform group is the most
telling evidence about a task's difficulty. The statistics have no switch: every training run keeps them, at under
1 KB per task in each snapshot and about 270 bytes per group in the event log, since the metrics and every later
milestone read them (decided 2026-10-20, open question 8). Their types (`MemberResult`, `GroupRecord`, `TaskStats`
and the store) are experimental and stay out of `platoon`'s exports until M4's sampler API makes them public, in
`platoon.experimental` (decided 2026-10-20, open question 2).

**Evidence decays by policy version.** A group sampled `n` versions ago weighs `2^(-n/32)`; a step that updates no
weights ages nothing. A task's pass rate is `Beta(a0 + s, b0 + f)` of its decayed passes `s` and failures `f`, with a
prior of weight 2 centered on its entry's decayed pass rate, `(1 + P) / (2 + N)`. The expected pass@k and the chance
that a group of `k` mixes outcomes have closed forms for fractional counts:

    E[pass@k]        = 1 - prod_{i<k} (b + i) / (a + b + i)
    E[informative_k] = E[pass@k] - prod_{i<k} (a + i) / (a + b + i)

The prior lets a task seen once borrow strength from its entry, and the unbiased estimator `1 - C(n-c, k) / C(n, k)`
would need `n >= k` attempts of one task.

**Metrics.** Each step logs, for the whole run and per label: the mean expected pass@1 and pass@G over the tasks with
recorded members (`train/tasks/.../pass_at_1`, `pass_at_G`), the share of tasks with any (`coverage`), the tasks by
pass-rate band, the tasks without evidence, and the kinds of the groups it trained on
(`train/batch/.../informative_groups` and the rest). Under an adaptive sampler, `pass_at_1` is a within-run view: it
averages the tasks the sampler chose to see, so only a held-out evaluation compares runs that sample differently.

#### 3.4 Persistence and resume

At each full save (launch, recovery and permanent, `<step>_interrupted`, `final`) the store is written as
`tasks/state-s<start>-<checkpoint>.json` in the run directory, synced and renamed into place, and the checkpoint's row
names it in `loop_state` with its SHA-256 and format. The start is in the name because every start saves a recovery
checkpoint at launch: a run that resumes from `recovery-000120` saves a second checkpoint of that name. A draw is keyed
by `(start, seq)`, with `start` one more than the largest start the run's rows name.

A resume loads the snapshot its checkpoint names before its launch save, so that the launch save holds the loaded
state. Until M3 reissues them, the snapshot's outstanding draws (drawn, never consumed by a committed step) count as
`lost`. A checkpoint without a readable snapshot (one saved before this release, a failed write, a changed file) goes
on from the newest older checkpoint whose snapshot reads, else from empty; a warning says which, and `lock.json`
records it per start (`tasks_state_lost`). A snapshot that cannot be written does not stop the statistics: the
checkpoint names none, and the next full save writes them again. Recovery pruning deletes a pruned checkpoint's
snapshot with it, the file its own row names. `tasks/events.jsonl` logs each start, draw, settle and fate, for audit;
a resume never reads it. The statistics never fail training: an unexpected error stops them for the rest of the
start, with a warning.

`platoon tasks show RUN_DIR` prints the newest snapshot's tasks, sorted by the chance of an informative group, by pass
rate or by groups, and `platoon tasks export` writes every task's row as CSV or JSON lines; both are read-only and
safe while the run trains.

### 4. Later: the dispatcher and exact resume (M3)

A dispatcher replaces the producer and the pending queue. It asks for a draw only when a group can start: within the
horizon (the groups dispatched and not yet dequeued, plus those the current step counted, stay below
`(max_staleness + 1) × batch_size`), with rollout permits, and with label capacity. A static draw whose label is full
waits in its label's first-in, first-out backlog, which does not count toward the horizon; past `4 × batch_size` of
them, new draws of that label are recorded as `skipped`, and the dispatcher draws only while some label of the run has
room. A checkpoint's snapshot also holds the samplers' state, the positions and backlogs, and the outstanding draws,
which a resume reissues first, as new groups with new `seq` numbers. So no draw is lost and none trains twice, and
every position of a static order is consumed exactly once. A fingerprint of the sampling inputs (weights, samplers and
their options, the pools) tells a resume what changed; it warns and goes on. The run-level `uniform` sampler lands
here.

### 5. Later: the sampler API and `pass_rate` (M4)

A sampler is a small class with a `next(ctx)` method that returns a pool task (or, for a run sampler, an entry's
slot), with `state_dict` and `load_state_dict`, an optional `observe(group)`, typed options, and registration by name
(`@task_sampler`). A static sampler (`adaptive = False`) reads no statistics, and its n-th draw is a pure function of n,
so it can bind early and resume by position; an adaptive one reads the statistics of section 3 through a read-only
view and decides when a group can start. Every decision is synchronous, deterministic (its randomness keyed by the
seed, the sampler's name and the decision index), and returns nothing only when nothing is eligible.
`platoon.testing` gains a conformance check. The types live in `platoon.experimental` once RFC 0017's surface exists,
with the statistics' types of section 3.3, which become public with them (decided 2026-10-20, open question 2).

`pass_rate` maximizes the chance that a task's next group mixes passes and failures, `E[informative_G]` under the
posterior of section 3.3, by Thompson sampling, with a fifth of the draws going to the least recently drawn tasks
and a cap of three draws per task per pool-sized window of draws, which bounds concentration. It is meant per entry,
under `weighted`, so that a mixture keeps its shares. It ships experimental and becomes recommended only after a GPU
A/B test shows a higher informative share at an unchanged held-out pass@1.

### 6. Later: group admission and refill (M5)

A `group_admission` component decides at settle whether a group counts toward the batch (`all`, the default, admits
every group, as today; `informative` admits groups that train, optionally within a pass-rate band). A rejected group
frees its slot at once, so the dispatcher refills while the trainer is busy, and a step trains the first
`batch_size` admitted groups, within a cap of dequeued groups per step. It ships only after its own A/B test.

## The hard parts

1. **Fresh evidence against exact resume.** Evidence enters when a group settles, and the store and every sampler's
   state are saved together at one consistent point, the start of each full save. A restart can lose a group's result
   but never count it twice, with no replay of the event log.
2. **A capped label before the dispatcher.** On today's producer, a capped label's waiting rows must neither fill the
   pending queue nor use up the staleness lookahead, or the label paces the run. M1 parks them beside the queue, gates
   the producer on the rows that can run, skips past a full backlog only while a label of the run has room, and
   replaces the rows still waiting at the end of the run. The price is a smaller share for that label, which the
   metrics show, and groups at the end of a run that never train, both accepted until M3 makes the horizon exact
   (decided 2026-10-20, open questions 3 and 11).
3. **Staleness per entry.** The lookahead must stay the run's, or a slow entry's bound would let every entry run far
   ahead; and no entry's bound may be below it. Pruning must use the largest bound, so a large bound costs storage
   until the pruner knows which checkpoints rollouts still sample (open question 5).
4. **Group sizes that differ by entry.** Concurrency counts rollouts, and a whole group takes its permits at once,
   first come first served, so a large group is not overtaken by small ones; capped labels must leave room for the
   others' groups, which loading checks.
5. **Signal against coverage.** An adaptive sampler earns its gain by drawing informative tasks again. Exploration,
   decay and the per-window cap keep coverage, and the metrics make concentration visible.

## Alternatives

- **Samplers that keep their own statistics** through `observe`, with a trainer-side ledger of positions. Simpler, but
  every sampler re-implements decayed counting, and the metrics, the viewer and admission rules each need a feed of
  their own. Here platoon keeps one record that serves all of them; `observe` remains for a sampler's own model.
- **An event log replayed on resume** as the source of truth. Snapshots at full saves need no replay and no lineage
  rules; the log stays as an audit trail.
- **Keep the producer and add a feedback call.** Decisions would stay up to `(k + 1) × batch_size` rows ahead of their
  rollouts, and only a dispatcher makes the horizon exact for capped labels.
- **Staleness bounds by label**, like `rollout_group_limits`. A column other than the entry is known only after the
  tasks are listed, so the pruning check could not run when the config loads.
- **A `training:` block per entry** instead of `workflow_config`. `workflow_config` mirrors `train.workflow_config` and
  `eval.workflow_config`, so the same key means the same thing in three places.
- **The entry wins over the phase** for its rollout settings in evaluation. More specific, but it reverses the rule
  that the phase decides; the phase keeps the last word, `platoon config show` names the winning layer, and loading
  warns when a phase's setting overrides an entry's.
- **Doing nothing.** A mixture keeps one compromise of settings, or a second run per slow environment, and four groups
  in five can keep training nothing.

## Rollout plan

| Milestone | Scope | Exit check | Pull requests |
|---|---|---|---|
| M0b the sampler's name | `train.task_sampler: auto \| weighted`; `task_order` a deprecated alias until 0.4.0 | a config that sets `task_order: weighted` loads with a warning and draws the same rows; `--strict` fails on it; a dump of 0.2.0's `config show --defaults` loads beside `task_sampler: weighted` with a warning that `task_order` was ignored | |
| M1 per-environment settings | section 2: entry `workflow_config`, `eval_overrides.workflow_config`, `max_staleness`; rollout permits; the capped-label rules on today's queue; `config show` per entry | a CPU run with fakes: two entries of different group sizes, step limits, minimum groups and baselines each run with their own, within the budget; a slow entry's groups train under its own bound; a capped slow label leaves the other's steps running, with and without a staleness bound | |
| M2 per-task statistics | section 3: members and verdicts, the store, snapshots and resume, `events.jsonl`, the `tasks/` metrics, `platoon tasks` | every verdict site has a test; a shaping reward processor changes no pass count; a loaded snapshot saves byte for byte; a resume from a recovery checkpoint keeps the older row's snapshot until pruning deletes it alone; `platoon tasks show` matches golden output | |
| M3 the dispatcher and exact resume | section 4; the run-level `uniform` | at `k = 0` every step trains exactly today's rows, and at `k >= 1` the pipeline never exceeds `(k + 1) × batch_size`; a capped slow label leaves the other label's step time within 10%; a run interrupted twice consumes every position exactly once; no draw while every label is at its limit | |
| M4 the sampler API and `pass_rate` | section 5 | the conformance check passes on every built-in; the closed forms match exact Beta-binomial sums; a `k = 0` run replays decision for decision; a GPU A/B against `uniform` | |
| M5 admission and refill | section 6 | with fakes, steps admit `batch_size` groups within the cap; a GPU A/B | |

## Open questions

Questions 1 to 3 are the draft's, and 7 to 11 came up in the review of M0b, M1 and M2; all of them were decided on
2026-10-20. Questions 4 to 6 stay open for the milestones they concern.

1. **Members without a valid result: censored or failures?** M2 censors them, as the baseline does, which leans
   optimistic when stragglers are slow failures. A `censored: failure` option, or counting a timeout the entry does
   not grade as a failure, would lean the other way. Decide before M4, whose sampler acts on the estimate. *Decided
   2026-10-20:* censored, as M2 does, not failures (section 3.2).
2. **When the statistics types become public.** `MemberResult`, `GroupRecord`, `TaskStats` and the store are
   experimental in `platoon.train.task_stats` and not exported from `platoon`. They could move to
   `platoon.experimental` with the sampler API (M4), or wait until a custom sampler has used them. *Decided
   2026-10-20:* they become public with M4's sampler API, in `platoon.experimental` (sections 3.3 and 5).
3. **The capped-label rules for configs that already set `rollout_group_limits`.** M1 applies them to every such config,
   so a capped label whose limit cannot keep up trains less than its weight instead of pacing the run. Should they be
   opt-in until M3? *Decided 2026-10-20:* they apply by default, to every config that sets limits (section 2).
4. **Settled groups across restarts.** A resume reruns groups that had settled but were not yet consumed, which is
   hours of work for a slow entry. Should full saves persist them? Decide with M3.
5. **Pruning under a large staleness bound.** Keeping every checkpoint within the largest bound costs storage. Should
   the pruner learn which versions rollouts still sample? Decide with M3.
6. **When refill ships.** After its A/B test (M5), or opt-in with M4? Decide with M4.
7. **A rollout budget that the capped labels' groups can fill: an error or a warning?** Such a budget makes a larger
   group wait for a capped one to end, and every group after it with it. *Decided 2026-10-20:* a warning, at load and
   once the tasks are listed: the run slows to the capped labels' pace, but goes on (section 2).
8. **A switch to turn the statistics off.** *Decided 2026-10-20:* none; every training run keeps them, since they cost
   little and the metrics and the later milestones read them (section 3.3).
9. **Rollout settings that a named environment ships for its evaluations alone.** *Decided 2026-10-20:* none; a named
   environment ships `workflow_config.rollout_config` only, as its entry's lowest layer, and the run's config decides
   its evaluations, with the phase's section and the entry's `eval_overrides` (section 2).
10. **A dump of 0.2.0's `config show --defaults` beside `task_sampler`.** The dump spells `task_order: auto`, 0.2.0's
    default, so under the rule M0b first had, which refused both names set to different samplers, a config built on
    it that sets `task_sampler: weighted` did not load. *Decided 2026-10-20:* `task_sampler` wins over
    `task_order: auto`, with a warning that `task_order` was ignored; both set and equal load with the deprecation
    warning; only `task_order: weighted` beside another `task_sampler` is a conflict, exit 78 (section 1).
11. **The groups a capped label leaves at the end of a run.** The rows still waiting when the run ends never train,
    and a waiting row whose label frees before then runs as a group that no step consumes, since rows past the run's
    plan replaced it. *Decided 2026-10-20:* accepted until M3, whose dispatcher draws only rows that can start
    (section 2).
