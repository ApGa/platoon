# RFC 0019: Training mixtures: per-environment settings, task statistics and samplers

| | |
|---|---|
| Status | Draft |
| Created | 2026-10-19; revised 2026-10-20 and 2026-10-21 |
| Authors | Apurva Gandhi (@ApGa) |
| Builds on | [RFC 0018](0018-configuration-only-projects.md), section 8 (the weighted fair order, `weight: tasks` and the sampling plan); [RFC 0016](0016-component-configuration.md) (component options, and the rollout settings of each phase, which section 2 of this RFC amends); [RFC 0013](0013-extension-model.md) (registration by name); [RFC 0017](0017-public-surface-and-conformance.md) (experimental API, conformance kits); [RFC 0003](0003-multi-policy-training.md) and [RFC 0001](0001-native-multi-agent.md), for what must stay compatible; the trainer (`platoon/train/tinker/rl.py`) and the group workflow (`platoon/train/tinker/workflows/group_rollout_workflow.py`) |
| Discussion | The pull request that adds this RFC, which also records the decisions of 2026-10-20 (open questions 1 to 3 and 7 to 11); M3's, which records those on questions 4 to 6; M4's and M5's, which records the revision of question 6 (2026-10-21) |
| Implementation | M0b, M1, M2 and M3 in platoon 0.3.0; M4 (the sampler API and `pass_rate`) and M5 (group admission and refill), experimental, in platoon 0.4.0; their GPU A/Bs pending |

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

**Then** (in platoon 0.3.0 too): **M3**, a dispatcher that draws a task only when its group can start, so
that a resume takes up exactly where the run stood and trains the groups a checkpoint kept, and the run-level
`uniform` sampler. **After it** (M4-M5, together in platoon 0.4.0): the sampler API, with an adaptive `pass_rate`
sampler that prefers tasks whose next group is likely to train, and group admission with refill. Both ship
experimental and opt-in, and each becomes the recommendation only after its A/B test.

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
| `uniform` (M3) | shuffled passes over every task of every entry, with no entry slots | `auto`'s batches |
| `pass_rate` (M4, experimental) | each group's task chosen among every task of the run by its pass rate, with no entry slots | the same |
| a registered name, `module:Class` (M4) | a task sampler of a package's, or of a class | the same |

The rename comes first because later milestones extend the choice: a run-level `uniform` (every task of every entry
equally often, with no entry slots; M3, section 4), an entry-level `task_sampler` that chooses which of the entry's
own tasks comes next (`uniform`, today's seeded passes, or the adaptive `pass_rate`), and custom samplers by
registered name or `module:Class`, each with typed options (`task_sampler_options`, RFC 0016). Under `weighted`, the
run's sampler hands each slot to an entry, and the entry's sampler fills it. As built in M4, each entry's
`task_sampler` (`uniform` by default, `pass_rate`, a registered name, `module:Class`) and `task_sampler_options` are
asked only by a run sampler that hands slots (`weighted`, and `auto` for a mixture); under `uniform` or `pass_rate`
they are refused, and so are numeric weights under `pass_rate`.

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

**A capped label never stalls the others** (M1 on the producer and its queue; M3's dispatcher keeps the rules, section
4): the rows of a label at its limit wait beside the batch's rows rather than among them, and do not count toward the
staleness lookahead; past four batches of them, the label's new rows are skipped and counted, while another label can
start a group. So a label whose limit cannot keep up with its share of the order trains less than its weight, and the
metrics say so. These rules apply by default to every config that sets `rollout_group_limits`, those written before
them included (decided 2026-10-20, open question 3). Under M1, at the end of a run, the rows still waiting never
train: the producer replaces them with rows past the run's planned batches, so that the last steps do not wait for the
capped label, and a waiting row whose label frees before the run ends still runs, as a group no step consumes. Those
extra groups were accepted until M3 (decided 2026-10-20, open question 11), whose dispatcher starts no group past the
plan.

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

`platoon/train/task_stats.py` keeps one record per training task, and totals per entry and per label: groups, attempts,
passes, reward, the kinds of groups, the current runs of uniform groups and of groups without a valid member (three in a
row flag a task without evidence: a broken grader or setup), decayed means of steps, tokens, seconds and reward, the
newest eight groups' summaries, and what became of each draw (`consumed`, `stale`, `failed`, `skipped`, `lost`, and,
since M5, `rejected`). Every settled group counts once, stale ones included: the work happened, and a uniform group is
the most telling evidence about a task's difficulty. The statistics have no switch: every training run keeps them, at
under 1 KB per task in each snapshot and about 270 bytes per group in the event log, since the metrics and every later
milestone read them (decided 2026-10-20, open question 8). Their types (`MemberResult`, `GroupRecord`, `TaskStats` and
the store) are experimental and stay out of `platoon`'s exports; M4 made them public, in `platoon.experimental` (decided
2026-10-20, open question 2). A settled group's record is made when it settles and applied, with every record made
since, in draw order (`(start, seq)`) before the next adaptive decision, full save or step's metrics, so the statistics
do not depend on the order in which groups completed.

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
state. The snapshot's outstanding draws (drawn, never consumed by a committed step) are what M3 takes up from the
dispatcher's own state (section 4): it trains the groups the checkpoint kept and draws the others again; a checkpoint
without that state, as M2 saved it, has them counted as `lost`. A checkpoint without a readable snapshot (one saved
before this release, a failed write, a changed file) goes on from the newest older checkpoint whose snapshot reads,
else from empty; a warning says which, and `lock.json` records it per start (`tasks_state_lost`). A snapshot that
cannot be written does not stop the statistics: the checkpoint names none, and the next full save writes them again.
Recovery pruning deletes a pruned checkpoint's snapshot with it, the file its own row names. `tasks/events.jsonl` logs
each start, draw, settle and fate, for audit; a resume never reads it. The statistics never fail training: an
unexpected error stops them for the rest of the start, with a warning, and its later checkpoints save none, so a
resume from one of them goes on from an older snapshot's statistics; its draws resume exactly all the same (section
4).

`platoon tasks show RUN_DIR` prints the newest snapshot's tasks, sorted by the chance of an informative group, by pass
rate or by groups, and `platoon tasks export` writes every task's row as CSV or JSON lines; both are read-only and
safe while the run trains.

### 4. The dispatcher and exact resume (M3)

A dispatcher (`platoon/train/tinker/dispatcher.py`) replaces the producer, the pending queue and the pool of rollout
workers. It draws a task only when its group can start, and starts each group as a task of its own, when three things
allow it:

- **The horizon.** With `k = train.max_staleness`, the groups dispatched and not yet dequeued (running, or settled and
  waiting for a step), plus those the current step has dequeued, stay below `(k + 1) × batch_size`; the count drops
  back when a step starts, after its policy is published. That is the producer's lookahead, made exact: with `k = 0`
  each step trains exactly the groups drawn for it, as before, unless a label's limit holds one of them back. Without
  a bound there is no horizon. Either way the dispatcher starts no group past those the run's remaining steps
  consume, so the groups that M1 drew past the plan for a capped label are gone (open question 11).
- **Rollout permits** (section 2): a group takes its entry's group size at once, first come first served.
- **Label capacity.** A draw whose label is at its `rollout_group_limits` waits in its label's first-in, first-out
  backlog, which does not count toward the horizon. Past `4 × batch_size` of them, a new draw of that label is recorded
  as `skipped`: its position is used up. The dispatcher draws only while some label of the run (one its training tasks
  carry) has room, so it never draws only to skip, and while every label is at its limit it draws nothing until a group
  ends.

It takes the oldest waiting draw whose label has room (the draws a resume reissued first), else the next position of
the order. It wakes on what it waits for: a group's release of its label slot and permits, a dequeue, a step that
starts; the producer's 20-second poll is gone.

**Slots held for a capped label.** The horizon's slots free when a step starts, and the draws that start then take
them, so a capped label's group that ends in the middle of a step would leave the label's next draw waiting for the
next step: a limit of `n` would become `n` groups a step, however short the label's groups. So while a capped label
has draws waiting, each of its running groups holds one more slot of the horizon, which the other labels' draws leave
free, and the label's oldest waiting draw takes it when the group ends. The capped labels' running groups and the
slots they hold stay within `k × batch_size`, so that the other labels keep at least a batch of the horizon; with
`k = 0` no slot is held, and a capped label starts at most its limit of groups a step.

**Exact resume.** Every draw is numbered `(start, seq)` and stamped on its row, and the dispatcher keeps every draw no
committed step consumed (outstanding). Each full checkpoint saves its state in a file of its own,
`tasks/dispatch-s<start>-<checkpoint>.json`, which the checkpoint's row names (`dispatch`), apart from the per-task
statistics, so that a resume is exact also when they stopped or their snapshot could not be written: the sampler, a
fingerprint of its inputs, the next position of the order, the outstanding draws and each label's waiting draws.
Every full save, the recovery saves and the one at launch included, also writes the groups that settled and that no
committed step consumed to `tasks/kept-s<start>-<checkpoint>.json.gz`: gzip-compressed JSON of their datums, exactly,
and of what the step that consumes a group logs of it, or, for a group whose workflow failed, that it failed. The
state names that file with its SHA-256, and the checkpoint's row names it too, so that recovery pruning deletes both
with their checkpoint (decided 2026-10-20, open question 4). A resume:

1. loads the kept groups, which go to the result queue first: the steps after the resume train each once, without
   rolling it out again, under the draw it settled for, and consume a failed one as failed;
2. draws again first every other outstanding draw (one whose group was running, or could not be kept: a workflow's
   output other than `GroupRolloutWorkflow`'s or a plain list of datums), in `(start, seq)` order, then the draws
   that waited for their label, back in their backlogs, in order: each as a new draw of this start (reason `reissue`)
   at its old position, which replaces the old one among the statistics' outstanding draws, so that it stays one draw
   of its task;
3. goes on with the order at the saved position.

So every position of the order is consumed exactly once: trained, dropped as stale, failed, skipped or rejected (M5,
section 6). Kept groups that cannot be read (their file lost or changed, or never written because the disk refused it,
each with a warning) are rolled out again instead, as before this milestone; the statistics count both of such a task's
groups, which both ran, as one draw. A checkpoint without the dispatcher's state (one saved before platoon 0.3.0, one
whose write failed, or a state that cannot be read) counts the outstanding draws as `lost` and draws from the
checkpoint's batch, as the producer did; from one saved under a group admission rule (section 6), whose steps draw more
positions than their batch, it draws from where the order stood, which the checkpoint's row records. A snapshot of the
statistics older than the checkpoint, loaded in place of its own, counts as `lost` the draws it holds as outstanding
that the checkpoint no longer does.

**Changed inputs.** The fingerprint holds `task_sampler`, `batch_size`, `label_by`, the entries in the order the
config lists them (a weighted order seeds and breaks ties by it, and `uniform` lists the pool in it), and for each
entry its weight, its seed when the sampler reads the entries' seeds, and the number of its training tasks and a
digest of their ids. A resume whose inputs differ warns, naming each change, records it in `lock.json`
(`sampling_changed`), and goes on: the order goes on at the saved position, now in the order the new inputs give, and
the outstanding draws of tasks that left the pool are `lost`.

**`uniform`.** The run-level `uniform` draws every training task of every entry in seeded shuffled passes
(`Dataset.shuffle(42 + 101 p)` of the pool in listing order, the passes a single environment's dataloader draws), with
no entry slots, in epochs of the pool's tasks filled to whole batches: every task equally often, as `weight: tasks`
draws them, with batches as the shuffle mixes them. It reads no entry's weight or seed, so an entry's numeric `weight`
beside it does not load; for one environment it draws `auto`'s batches.

### 5. The sampler API and `pass_rate` (M4)

As built in platoon 0.4.0, experimental, in `platoon.experimental`, with the statistics' types of section 3.3 (decided
2026-10-20, open question 2):

- **A sampler** is a class built as `Class(setup, options)` with `next(ctx)`, `state_dict()` and
  `load_state_dict(state)`, an optional `observe(group)` and `metrics(stats)`, a class flag `adaptive`, typed options
  (`options_schema`: a dataclass, a pydantic model or a parser, RFC 0016's path) and an optional load-time
  `check(options, ctx)`. `next` returns a `Draw(task_id, reason, score)` of its pool (or a bare task id), or, from a
  static run sampler whose class sets `entry_slots`, a `Slot(entry)` that hands the draw to that entry's sampler.
- **Static and adaptive.** A static sampler (`adaptive = False`) reads no statistics and is a pure function of its
  decision index: it is asked when the dispatcher draws a new position, and its draw binds at once and may wait for
  its label, as the built-ins' always did. An adaptive one reads the statistics through a read-only view (cheap
  per-task counts, and whole records for a few tasks) and is asked only when a group can start: within the horizon,
  with the group's rollout permits held first, and choosing among the tasks whose label has room. Under `weighted`, a
  slot owed to an adaptive entry waits as a token in the entry's queue; the token waiting for its permits is saved
  with the dispatcher's state, so no full save loses its position.
- **Determinism.** A decision's randomness is keyed by the sampler's seed, its name and its decision index, which
  advances with each draw it binds. It returns nothing only when nothing is eligible. A sampler that raises, returns
  nothing while something is eligible, or answers outside the contract gets a fallback draw (the eligible task at
  `key` modulo their number, or a slot by weight), which uses the decision up, is logged once per sampler and counted
  in the step metrics; the GPU A/B requires that count to stay 0 (decided 2026-10-21). Core's static samplers, which
  a valid setup never makes fail, get none: what they raise (an order handed to the trainer that fails) stops the run,
  as 0.3.0's orders did.
- **Registration.** `@task_sampler(name, options=..., check=...)` from a module an entry's `package` names, or
  `module:Class`. Core's names (`auto`, `weighted`, `uniform`, `pass_rate`) are reserved whatever the import order. An
  unknown name stays in the file's one report when no entry names a package, and is refused before the run starts
  otherwise. `lock.json` records each sampler with its options, defaults included, and the sampling fingerprint holds
  the samplers platoon 0.3.0 did not have.
- **State.** Each sampler's state and decision count go into the dispatcher's file of each full checkpoint (format 2),
  taken right after the statistics have applied every settled group, so that sampler states, statistics and
  outstanding draws agree. A resume of a 0.3.0 checkpoint recomputes the static samplers' counts from the position.
- **The conformance check**, `platoon.testing.check_task_sampler`, runs named checks (the protocol, options, a pool
  task or nothing, fast, deterministic in other processes, static, state, `observe`, metrics), and every built-in
  passes it.

`pass_rate` maximizes the chance that a task's next group mixes passes and failures, `E[informative_G]` under the
posterior of section 3.3 (the store's half-life of 32 versions, prior of weight 2 at the entry's decayed rate, members
without a valid result censored), by Thompson sampling on `1 - p^G - (1 - p)^G`. A fifth of its draws go to the eligible
task drawn least recently (`explore: 0.2`), an unseen task waits for its first result, a task whose last three groups
had no valid member is drawn by exploration only, and a task is drawn at most three times per pool-sized window of its
decisions (`max_draws_per_epoch: 3`), which bounds concentration; `max_outstanding_per_task` (off), `candidates` (4096)
and `seed` are its other options (decided 2026-10-21: these five only). It is meant per entry, under `weighted`, so that
a mixture keeps its shares; at the run level it chooses among every task. Each step logs, per sampler, its draws by
reason, its decision time, the informative share it predicted and the one it got for the groups the step consumed,
`top10_share`, `eligible_share`, `cap_lifted`, `prior_share` and `evidence_age`. It ships experimental and becomes
recommended only after a GPU A/B test shows a higher informative share at an unchanged held-out pass@1.

**What a `k = 0` run replays.** Since the statistics apply settled groups in draw order (section 3.3), a run at
`max_staleness: 0`, with rollout permits for the whole batch (for a run-level adaptive sampler, the batch size times
the run's largest group) and no label limit holding a draw back, makes the same decisions whatever order its groups
complete in, scores bit for bit. Across a resume the interrupted step's groups enter the statistics in another order:
those that settled before the save at it, the others after the resume. The scores then agree to a relative 1e-12, and
the decisions replay unless a task's group drawn again came before one of its groups that had settled: its streaks and
newest groups then follow another order (`pass_rate` reads the streak of groups without a valid member). With fewer
permits or a binding limit, a decision waits and may see groups of its own step that completed first.

Left for later: further `pass_rate` options (a half-life of its own, other priors and scores), an entry-point group for
samplers, adaptive run samplers that hand slots (entry weights that follow learning), evaluation aggregates in the
view, and a held slot for a capped label under a run-level adaptive sampler.

### 6. Group admission and refill (M5)

A **group admission rule** decides, when a training group settles, whether it counts toward the batch. It is a small
class with `admit(group, stats) -> Admission`, where `group` is the group's `GroupRecord` (section 3.3) and `stats` the
read-only view of the statistics a sampler reads (section 5), and `Admission(admit, reason)`, or a bare `bool`. It is
stateless: a pure function of the group, the view and its options, so that a resume decides as the run would have, and
`platoon.testing.check_group_admission` checks it. Rules register with `@group_admission(name, options=..., check=...)`
from an entry's `package` or are named as `module:Class`; options are typed and checked as a sampler's are (RFC 0016),
and core's names are reserved. The types are in `platoon.experimental`.

- **`all`** (the default) admits every group: a run whose every entry's rule is `all` trains exactly as before M5.
- **`informative`** rejects the groups whose workflow's verdict (section 3.1) says they train nothing (`uniform`, by
  success `uniform_pass`, `uniform_fail` or `uniform_mixed`, and the verdicts of a group without datums), and, with
  `band: [lo, hi]`, those whose pass rate is outside it (`below_band`, `above_band`); it admits the rest, a group
  whose output carries no verdict included (`unknown`), so that a workflow that gives none never empties every step.
  Resolution refuses it beside a workflow class that gives no verdicts.
- **Per entry.** `train.group_admission` and `group_admission_options` are the run's; an entry may set its own
  (`environments[].group_admission`, with its own options), and a named environment may set neither. The cap,
  `train.refill.max_groups_per_step`, and the step's accounting are the run's: once any entry's rule is not `all`,
  every entry's groups are counted as below.

**Where it decides.** In the group's own coroutine, right after the statistics record the group and before any step
dequeues it, so before the stale filter, which needs the consuming policy's version. The view reads the statistics as of
their last flush (it never flushes: section 3.3's draw order holds). A rejected group's fate, `rejected`, is final
there: it leaves the outstanding draws, is never kept, its datums are dropped, and no resume draws it again, unless the
start crashes, or stops before its first step completes, before the next full save. Its slot of the horizon is free at
once, so the dispatcher starts another group while the trainer trains (refill); a group that failed or carried no datum,
which can end at once (an outage), keeps its slot until a step dequeues its marker, as without a rule, so that such
groups never start faster than steps take them. A marker with the group's statistics goes through the result queue, so
the step that dequeues it records them. A group whose workflow failed is decided the same way under a rule (fate
`failed`, without asking the rule). A rule that raises, or answers outside the contract, admits the group, logged once
and counted.

**What a step trains.** The first `batch_size` admitted groups it dequeues that are not stale; it closes once it has
counted `C = ceil(refill.max_groups_per_step × batch_size)` groups (3.0 by default, a finite number of at least 1),
training what it admitted. It counts every group it dequeues (admitted, stale, or a kept failed one) and every marker of
a group rejected or failed **while it was current**; a marker of a group rejected while an earlier step was current
passes over without counting. (The draft's "within a cap of dequeued groups per step" counted every marker: in the
asynchronous loop the markers queued while the trainer trains then fill the next step's cap, which closes short while
admitted groups wait behind them and go stale.) At `max_staleness: 0`, and on a run's last step, the dispatcher also
starts no group that the current step could not count, so none of those is stranded; but since a step can close short of
its batch, up to `batch_size` groups started before the last step can outlive the run, which the final checkpoint keeps.
A short step trains what it admitted, in one minibatch (`num_minibatches` must be 1 beside a rule), its partial last
microbatch weighed by its share of groups; a step that admitted nothing takes no update, and the run ends after its
planned steps.

**What it costs.** Rollouts per informative group are one over the informative share with or without a rule; what
changes is the rollouts a step takes. The cap bounds what a step counts, not what starts while the trainer trains: at
`max_staleness` 1 or more the horizon and the rollout permits bound those, about `((k + 1) × batch_size - a) / p` groups
during the update of a step that admitted `a`, at an admitted share `p`, which can be many times the cap where few
groups are admitted. Where rollouts bound a run, a step takes several times as long and the run trains as many
informative groups an hour; where the trainer bounds it, refill spends the rollout capacity the trainer leaves idle and
trains more of them an hour; at `max_staleness: 0` it gains where few groups are informative and loses where many are.
The gradient becomes the mean over the admitted groups, larger by about one over the informative share, which Adam's
update absorbs but clipping does not; under a loss whose objective is not zero at zero advantages, a rejected group's
advantage-independent terms go too, and loading warns. After a step that closed short, up to `batch_size` minus its
admitted groups may start that only a step `k + 1` ahead could consume, so the lookahead of section 4 holds within that
margin, and the stale filter drops what waits too long (at `max_staleness: 1`, up to that many a short step).

Each step logs what it admitted, rejected (by reason), dropped as stale and counted, whether it was short, and the
draws per admitted group (`train/admission/...`), only under a rule. The dispatcher's state is unchanged (rules keep no
state, markers are not kept); the fingerprint gains the rules and the cap under a rule, so that a resume that turns one
on or off, or changes it, says so. Left out of M5: `refill.when_short: admit_rejected` (it would make a rejection's fate
provisional), `train.on_stale: requeue` (independent of admission), `informative`'s `min_valid` (the entry's
`min_successful_group_size` does it), `weighted`'s `balance: admitted`, a cap of null, and rule state. M5 ships
experimental and opt-in with M4, `all` the default; its GPU A/B decides whether it is recommended (open question 6,
revised 2026-10-21).

## The hard parts

1. **Fresh evidence against exact resume.** A settled group's record is made at settle and applied in draw order
   before the next adaptive decision, full save or step's metrics, and the store and every sampler's state are saved
   together at one consistent point, right after that application at the start of each full save. A restart can lose
   a group's result but never count it twice, with no replay of the event log.
2. **A capped label before the dispatcher.** On M1's producer, a capped label's waiting rows must neither fill the
   pending queue nor use up the staleness lookahead, or the label paces the run. M1 parks them beside the queue, gates
   the producer on the rows that can run, skips past a full backlog only while a label of the run has room, and
   replaces the rows still waiting at the end of the run. The price is a smaller share for that label, which the
   metrics show, and groups at the end of a run that never train, both accepted until M3 made the horizon exact
   (decided 2026-10-20, open questions 3 and 11). M3's dispatcher keeps the backlogs and the skips, and counts in its
   horizon only the groups that run, so it starts no group past the plan; it holds a slot of the horizon for a capped
   label's next group, so that the label's limit, not the steps, paces it.
3. **Staleness per entry.** The lookahead must stay the run's, or a slow entry's bound would let every entry run far
   ahead; and no entry's bound may be below it. Pruning must use the largest bound, so a large bound costs storage
   until the pruner knows which checkpoints rollouts still sample (open question 5, deferred).
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
| M3 the dispatcher and exact resume | section 4: the dispatcher, its state in a file of each full checkpoint and the groups it keeps (open question 4), the fingerprint; the run-level `uniform` | at `k = 0` every step trains exactly today's rows when no label's limit holds one back (CPU runs with fakes, and the same `forward_backward` datums, in order, as before it), and at `k >= 1` the pipeline never exceeds `(k + 1) × batch_size`; a capped slow label leaves the other label's step time within 10%; a run interrupted twice consumes every position exactly once; no draw while every label is at its limit; a resume trains each group its checkpoint kept once, without rolling it out again, one from a checkpoint without them rolls them out again, and pruning deletes them with their checkpoint | |
| M4 the sampler API and `pass_rate` | section 5 | the conformance check passes on every built-in; the closed forms match exact Beta-binomial sums; a `k = 0` run replays decision for decision (with permits for the whole batch and no binding label limit; across a resume, scores to a relative 1e-12, and a task's streaks in another order where its groups straddle the interrupt); a GPU A/B against `uniform` (pending) | |
| M5 admission and refill | section 6 | with fakes, steps admit `batch_size` groups within the cap; with `all`, every config trains exactly as with M4 alone; a GPU A/B (pending) | M4's (one pull request, one commit) |

## Open questions

Questions 1 to 3 are the draft's, and 7 to 11 came up in the review of M0b, M1 and M2; all of them were decided on
2026-10-20, 1 to 3 and 7 to 11 with M0b to M2, and 4 to 6 with M3 (question 5 deferred).

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
   hours of work for a slow entry. Should full saves persist them? *Decided 2026-10-20:* yes. Every full save, the
   recovery saves included, keeps them with its snapshot, a resume trains each once without rolling it out again, and
   pruning deletes them with their checkpoint (section 4).
5. **Pruning under a large staleness bound.** Keeping every checkpoint within the largest bound costs storage. Should
   the pruner learn which versions rollouts still sample? *Deferred 2026-10-20:* not for now. The runs in view keep
   small staleness bounds; recovery pruning keeps every checkpoint within the largest bound, as section 2 says.
6. **When refill ships.** After its A/B test (M5), or opt-in with M4? *Decided 2026-10-20:* after its own A/B test,
   not as an opt-in with M4 (section 6). *Revised 2026-10-21:* opt-in with M4, in the same pull request, experimental,
   `all` the default, so that every config that does not set a rule trains as before; the A/B decides whether it is
   recommended.
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
    (section 2); since M3 it starts no group past the plan (section 4).
