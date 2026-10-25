# Changelog

What each release of platoon changes. Platoon's version is below 1.0: a minor release (0.3.0) may change the Python
API, the config format or the command line, and a patch release (0.2.1) only fixes bugs. To stay on one minor
release, require a range such as `platoon[tinker,ors]>=0.3,<0.4`.

## 0.4.0 (unreleased)

### New features

- **Task samplers you choose per environment, and an adaptive one (experimental).** Each environment entry may set
  `task_sampler` and `task_sampler_options`: under `train.task_sampler: weighted` (a mixture's default), the run hands
  each slot to an entry by weight, as before, and the entry's sampler chooses which of its tasks fills it. `uniform`,
  the default, draws the entry's tasks in the seeded passes it always has; `pass_rate` chooses the task whose next
  group is most likely to mix passes and failures, so that more groups train, from the per-task statistics: Thompson
  sampling on each task's pass rate, with a fifth of the draws going to the tasks drawn least recently and at most
  three draws of a task per pool-sized window. `train.task_sampler: pass_rate` chooses over every task of the run
  instead. Each step logs what it drew and why, the informative share it predicted and the one it got, how
  concentrated its draws are, and how long it took to decide (`train/sampler/...`). It is experimental: it becomes the
  recommendation only after a GPU comparison with `uniform` shows more informative groups at an unchanged held-out
  pass@1
  ([Choose tasks by their pass rates](docs/how-to/train-and-eval.md#choose-tasks-by-their-pass-rates-experimental)).
- **Write your own task sampler (experimental).** A sampler is a small class, registered with
  `@task_sampler(name, options=...)` from an entry's `package`, or named as `module:Class`, at the run level or an
  entry's, with typed options (`train.task_sampler_options`, or the entry's `task_sampler_options`) checked against its
  schema before the run starts (core's samplers' already when the config loads) and recorded in `lock.json`. A sampler
  that reads no statistics draws when the order reaches it, as the built-ins always have; one that reads them decides
  when its group can start, among the tasks whose label has room, reproducibly from its seed and its decision count.
  The API and the per-task statistics types it reads are in the new `platoon.experimental` module, and
  `platoon.testing.check_task_sampler` checks a sampler against the contract
  ([Write a task sampler](docs/extending/task-samplers.md)).
- **Train only on the groups that carry signal, with refill (experimental).** `train.group_admission: informative` (or
  an environment's own `group_admission`) decides when each training group settles whether it counts toward the batch: a
  group whose workflow says it trains nothing (its rollouts scored alike, or it carried no datums) is rejected, and one
  whose rollouts scored alike frees its place at once, so that another group starts while the trainer works (one without
  datums, which can end at once, keeps its place until a step takes it). A step trains the first `batch_size` admitted
  groups that are not stale, and stops waiting once it has counted `refill.max_groups_per_step` (3.0) times `batch_size`
  groups, training what it admitted. `group_admission_options: {band: [lo, hi]}` also rejects groups whose pass rate is
  outside the band. `all`, the default, admits every group, as before. A step then trains up to several times the
  informative groups, and only those, at more rollouts a step: up to the cap's multiple of the groups it counts, and
  where the trainer bounds the run, the groups that start while it trains, which the horizon and the rollout permits
  bound, not the cap; each step logs what it admitted and rejected, and why (`train/admission/...`). Under a loss whose
  objective is not zero at zero advantages, a rejected group's advantage-independent terms go too, and loading warns;
  beside a rule, `num_minibatches` must be 1. Custom rules register with `@group_admission` or are named as
  `module:Class`; the API is in `platoon.experimental`, and `platoon.testing.check_group_admission` checks a rule. It is
  experimental: it becomes the recommendation only after a GPU comparison ([Train only on groups that carry
  signal](docs/how-to/train-and-eval.md#train-only-on-groups-that-carry-signal-experimental), [Write a group admission
  rule](docs/extending/group-admission.md)).
- **Rollout permits that come back as each rollout ends, by default.** A training rollout returns its permit
  (`train.max_concurrent_rollouts`) as soon as it ends rather than when its group's workflow returns
  (`train.permit_release: rollout`, the default), so the next group starts on the permits a group's finished rollouts
  give back instead of waiting for its slowest one; on long agentic tasks a quarter to a third of the budget sat idle
  that way. `GroupRolloutWorkflow` runs each member on its group's lease, so a group starts on its first free permit
  and its members as the rest free (rolling starts), each member's straggler deadline shifted by its own start delay; a
  workflow that does not say it runs its rollouts on the lease gets each group's permits whole, as before. Groups are
  drawn, decided, credited, admitted and resumed as before, a partly finished group counts whole toward the horizon,
  and a run at `max_staleness: 0` with the default budget trains the same datums in the same order. More groups are
  open at once, so more long groups can go stale: the run warns when they do, and raising `max_staleness` by one is the
  documented answer. The default changes ahead of a GPU comparison of the two settings; `train.permit_release: group`
  keeps the behavior of earlier releases, event for event (RFC 0020;
  [Rollout permits](docs/how-to/train-and-eval.md#rollout-permits); what changes for a config that sets nothing is
  under Changes, below).
- **Run a workflow's own rollouts on the lease (experimental).** `platoon.experimental` exports `rollout_lease()` and
  `RolloutLease`: a workflow runs each rollout in `async with rollout_lease().rollout(i):`, whose permit comes back
  when the block exits, and declares `runs_rollouts_on_lease = True` for rolling starts. In a training draw a block for
  an index outside the draw's size raises `IndexError`, and a second block for an index `RuntimeError`, under either
  setting; outside a training draw, as in an evaluation, the lease is a no-op ([Run rollouts on the
  lease](docs/extending/rollout-leases.md)).
- **See what keeps groups from starting.** Every training run logs, each step, the share of the step in which the
  dispatcher could not start a group and why (`train/dispatch/blocked_share/{permits,horizon,labels,plan,sampler}`),
  the permits held (`rollout_client/permits_held_mean`), the groups waiting for a worker-startup slot under process
  isolation (`rollout_client/startup_gate_waiting_mean`), and, where a straggler policy applies, what it censors
  (`train/batch/censoring/*`, and each label's: the share of rollouts it cancelled, and over mixed groups the pass
  rate of each group's slowest valid rollout against the others'). It warns, once per start of training, when more
  than 2% of the groups the last 10 steps consumed went stale (so from the tenth step on), naming what held the
  dispatcher back and the fixes, and when more of `GroupRolloutWorkflow`'s rollouts ran at once than
  `max_concurrent_rollouts`.
- **Rollouts start without the work that each used to do again, or wait for.** Two costs fell on a rollout's start.
  A CLI harness's first rollouts on a node waited while one of them copied the harness's toolchain to node-local disk
  (about 36,000 files and 800 MB for OpenHands), at every start and every resume in a new job: `platoon train` now
  stages its toolchains while it opens the training service and saves at launch, and starts no rollout, training's
  or an evaluation's, before that is done (`platoon eval` while it lists its tasks), so that no rollout waits for the
  copy and the first ones wait only for what of it outlasts the launch save; it waits ten minutes at most, and not
  past a shutdown request. And each rollout worker imported, after its fork, what its environment and harness import
  when they open, about 3,700 modules for an in-process ORS environment played by CodeAct, and parsed the
  environment's task file again: the workers' fork server now does that once, before it forks them, from what the
  config names (the environment's module and the openreward SDK, each function harness's and program's module, the
  registry lookups, an in-process environment's task lists for the run's splits), and starts as soon as the training
  client is open. It keeps only what leaves it fork-safe: a module or a build that starts a thread or a process,
  leaves a file, socket or event loop open, or seeds a numpy generator from the OS is left out with a warning that
  says why, and the workers import it themselves. `train.workflow_config.worker_preload` adds modules that an
  environment imports only when it opens. In CPU benchmarks of TextCraft-Synth played by CodeAct with Qwen3-4B's
  renderer, a forked worker reached its agent in 0.03 s instead of 7-9 s, 25-30 s and 48-56 s with 32, 128 and 256
  workers starting at once, with 149 file system calls on the shared filesystem instead of 17,134 and 30 MB of memory
  of its own instead of 290 MB. In CPU emulations of an OpenHands first wave whose toolchain copy took 77-90 s, the
  first sandbox could open 8-17 s sooner after the training client opened, and 46-53 s sooner behind a 38 s launch
  save ([Rollouts are slow to start](docs/how-to/troubleshooting.md#rollouts-are-slow-to-start)).
- **A limit on rollouts starting at once (experimental).** `train.max_starting_rollouts: N` lets at most `N` rollouts
  set up their workers, sandboxes and harnesses at once, training's and evaluation's together, first come, first
  served: a `GroupRolloutWorkflow` rollout takes a start slot once it holds its permit and gives it back as its first
  model call is issued, as it ends, or after `worker_startup_budget_seconds` + 300 s without a call. It is for
  start-ups that contend for a shared capacity, such as sandboxes that each extract a root filesystem; off by default
  (`null`), it changes nothing. It shortens no one-time start-up work (the fork server's import, a harness toolchain's
  copy to each node, the modules each worker imports), and with any value a run's whole first wave reaches the model
  later. Each step logs the rollouts starting and waiting to start, their waits and why slots came back
  (`rollout_client/start_limit/...`), and the run warns when slots come back on their timer, or mostly only at their
  rollouts' end ([Rollout start-ups](docs/how-to/train-and-eval.md#rollout-start-ups-experimental)).

### Changes

- `train.task_sampler` takes registered names and `module:Class`. An unknown name is still refused when the config
  loads, in the same report as the file's other problems, with a longer message that begins as before, unless an
  entry names a `package` that could register it; it is then refused before the run starts.
- A full checkpoint's `tasks/dispatch-*.json` is format 2: it adds the row each outstanding draw drew, the count and
  state of each task sampler, the run's and each environment's under `weighted` (and `auto` for a mixture), and the
  slots waiting for an environment's adaptive sampler. A resume reads platoon 0.3.0's format 1 as before, and goes on
  exactly where its run stood; platoon 0.3.0 cannot resume exactly from a checkpoint that 0.4.0 wrote (it draws from
  the checkpoint's batch, as from a checkpoint of 0.2)
  ([Checkpoints and resume](docs/how-to/train-and-eval.md#checkpoints-and-resume)).
- The per-task statistics record which task sampler drew each group, and why: `GroupRecord.sampler` names the
  sampler that chose the task (the run's sampler as the config names it, or an entry's as `<sampler>@<entry>`, such as
  `uniform@<entry>` for an environment's `uniform` under `weighted`), with the draw's `reason` and `score`, and so do
  the draw events of `tasks/events.jsonl`, every one of them.
- Every training run's `lock.json` records its task samplers, the run's and each entry's it asks, as components of
  `kind: task_sampler` with their options, defaults included; and every run logs what its samplers drew since the last
  step, by reason, and how long their decisions took (`train/sampler/draws`, `train/sampler/draws/<reason>`,
  `train/sampler/decision_ms`, and an entry's under `train/sampler/environment/<entry>/`).
- The per-task statistics take in settled groups in the order they were drawn, at each step's metrics, each full
  checkpoint and each adaptive decision, not in the order they completed, so their values no longer depend on
  completion order within a step.
- The per-task statistics have a sixth fate, `rejected`, a group admission rule's: `platoon tasks show` prints it
  after `lost`, and `platoon tasks export` has a `fate_rejected` column after `fate_lost`, so the five columns after it
  (`group_size` to `newest_version`) move one place right. A rejection's `fate` event in `tasks/events.jsonl` names the
  rule and its reason (`rule`, `admission`).
- `GroupRolloutWorkflow` declares `gives_verdicts = True`; a workflow class of your own that gives each group a
  verdict declares it too, so that `group_admission: informative` loads beside it.
- A full checkpoint of a run under a group admission rule records in its row of `checkpoints.jsonl` where the order
  stood (`next_position`): a resume that cannot read the checkpoint's state of the draws draws on from there, since
  such a run's steps draw more positions than their batch.
- `GroupRolloutWorkflow` declares `runs_rollouts_on_lease = True` and runs each member in a block of its group's
  lease; a subclass that overrides `arun_episode`, `_run_group_rollouts` or `_run_group_member`, itself or through a
  mixin, reads as `False` unless its own class body sets it. Under `permit_release: group` the blocks hold nothing,
  but a training group of a subclass that runs a member index twice, as a retry of the whole group in
  `arun_episode` does, now fails with `RuntimeError` under either setting: a retry stays inside the member's block.
  A member's rollout sees no lease, so a group that it runs in turn takes no permits of its own and runs as before.
- **Breaking: rollout permits come back as each rollout ends, unless `train.permit_release: group`.** The default of
  `train.permit_release` is `rollout`, so `train.max_concurrent_rollouts` counts the rollouts running, no longer the
  whole groups it holds. Where the permits bound a run, more groups run at once and start sooner, steps get shorter, and
  more long groups can be dropped as stale: when more than 2% of the groups the last 10 steps consumed went stale, the
  run warns and advises `max_staleness + 1`. There its rollouts are requested in another order, some on a newer policy,
  and a restart redoes more of them. In process, a rollout's permit also waits, at most 60 s, for the sandbox and env
  session teardowns that platoon finishes in the background, so that no more sandboxes are alive at once than the budget
  (under `group` such a teardown can still outlive its group, as before). A run whose permits never bind, as at
  `max_staleness: 0` with the default budget, trains as before. Each start of training under `rollout` says so in a line
  of its log (`Each rollout returns its permit as it ends (train.permit_release: rollout) ...`). `train.permit_release:
  group` restores the behavior of earlier releases ([The previous
  behavior](docs/how-to/train-and-eval.md#the-previous-behavior)).
- Under `train.permit_release: rollout`, `rollout_client/rollouts_in_flight_configured` is `max_concurrent_rollouts`
  itself, so `concurrency_achieved_fraction` reads as the permits' busy share, and `groups_in_flight` can exceed the
  budget over the group size. The straggler settings' documentation calls them a deadline against runaway rollouts,
  not a throughput lever.
- Under `train.permit_release: rollout`, an in-process rollout's permit comes back once the sandbox and env session
  teardowns that platoon finishes in the background have ended too, such as one that a straggler cut interrupts or an
  ORS environment's that waits for its code to end, so that no more sandboxes are alive at once than
  `max_concurrent_rollouts`. The permit waits for them at most 60 s, and the run warns of a teardown that runs longer;
  a cancellation ends the wait and no teardown, and a rollout that had ended keeps its outcome. Under `group` nothing
  waits for them, as before ([Rollout permits](docs/how-to/train-and-eval.md#rollout-permits)).
- A staged harness toolchain has a stamp beside it, `.<copy>.staged`, that records the tree hash it was checked
  against and the jobs that checked it. Where the copy outlives its job (`PLATOON_TOOLCHAIN_CACHE` on node-local disk
  that persists), the first staging of each later job hashes the copy again and keeps it when it matches its
  manifest; before, a kept copy was used unchecked. One that no longer matches is copied again beside it, and the old
  copy is set aside whole (`.<copy>.<hex>.stale`), never removed, since rollouts of another job may still run from it.
  A copy staged by an earlier platoon is checked once. A cache that platoon cannot write to (read-only, another
  user's, a full disk) records no check, so its kept copy is used unchecked, as before, with a warning.
  `platoon.harness.toolchains.prestage` stages toolchains ahead of rollouts.
- The rollout workers' fork server takes 4-8 s longer to start and holds 150-230 MB more memory, which the workers it
  forks share (14-20 s against 10-13 s, and 890-930 MB against 670-740 MB, in CPU benchmarks of TextCraft-Synth played
  by CodeAct with Qwen3-4B's renderer), once per run and while the trainer saves at launch; each worker holds far less
  of its own.
  Its first log line counts the modules it imported for the environments beyond the worker's stack (`imported the
  worker's stack and 9 module(s) the environments open with`). An environment variable that one of those imports
  sets is set in each worker too. Under `worker_start_method: exec`, `rollout_isolation: inprocess` and
  `worker_python_env_isolation` nothing is imported ahead, as before, and a `worker_preload` set there is warned
  about.

### Fixes

- A resume under `train.task_sampler: uniform` draws again exactly the row it had drawn; it could take the listing's
  form of the row, whose nested values differ in their empty keys.
- A rollout worker that is cancelled, as by a straggler cut or a stop, no longer leaves its readiness and diagnostics
  files in node-local scratch, where they stayed until a later run found them a day old.

## 0.3.1 (unreleased)

### Fixes

- **A model service that sends no routing ids is warned about once, not for each rollout.** With
  `rollout_isolation: process`, the trainer warned about every rollout whose model calls did not all get a routing
  id, which a server with rollout routing replay (R3) sends with each sampled sequence. With a service that sends
  none, as a dense model's does, the trainer logged a warning for each rollout: 18,125 in one 70-step run. Now it
  warns once for each start of training, about training and about evaluation rollouts each, that the service sends no
  routing ids: expected for a dense model, but no rollout routing replay for a mixture-of-experts model. It still
  warns about each rollout that got routing ids for some of its model calls but not all, or for none after the service
  has sent some, and it reports the first rollout to get any after rollouts without them, naming the first few of
  those. The metrics `train/rollout_worker_routing_id_coverage` and `eval/rollout_worker_routing_id_coverage` are
  logged as before.

## 0.3.0 (2026-10-20)

Each environment of a training mixture can set the training settings of its own groups, training keeps per-task
statistics and logs pass@k, a dispatcher starts each training group when it can start, and a resume takes up exactly
where the run stood. A label at its `rollout_group_limits` no longer paces the run, which changes what such a config
trains. Harbor verifiers that ask for root can run as root (`verifier_user: honor`), and the `openhands` harness can
condense at a token count. `train.task_order` is now `train.task_sampler`, with a new `uniform` sampler. The forms that
platoon 0.2.0 deprecated are gone, but `strategy` stays, as the unit of `every`. The `hpc` extra takes hpc-sandbox 0.4,
and the TextCraft extension's release for platoon 0.3 is `platoon-textcraft` 0.2.0.

### New features

- **Each environment of a mixture sets the training settings of its own groups.** An entry's `workflow_config` is laid
  over `train.workflow_config` for its groups: its `group_size`, its baseline (`leave_one_out_baseline`), its
  `min_successful_group_size`, its straggler defaults, its filters (`filter_errors`, `filter_message_actions`) and
  subagent datum sampling, and its rollout settings (`rollout_config`: `max_steps`, `timeout`, sampling and the rest),
  so a dataset's calibrated step budget and a slow environment's fewer, longer rollouts sit beside the run's. Its
  rollout settings are its evaluations' defaults, below `eval.workflow_config` (and in `platoon eval`, below
  `inference.workflow`), and `eval_overrides.workflow_config` sets what its evaluation rollouts change (every split but
  `train`). An entry's `max_staleness` raises the staleness bound of its own groups, while `train.max_staleness` stays
  the bound on how far rollouts run ahead, and recovery pruning keeps the checkpoints within the largest bound. Loading
  refuses a key that acts on microbatches or on the run's rollout workers with where to set it, an entry bound below the
  run's, and an entry's settings beside a workflow class that does not apply them (exit 78). Named environments may ship
  rollout settings (`workflow_config.rollout_config`) only. `platoon config show` prints each entry's effective settings
  with the layer each came from (`--entry NAME` for one), `platoon config validate` and `platoon train` say what each
  entry changes, and `lock.json` records it (`entry_settings`). `max_concurrent_rollouts` counts rollouts that groups of
  different sizes share, and each step logs each label's `stale_groups` and `num_steps`
  ([Settings per environment](docs/how-to/train-and-eval.md#settings-per-environment)).
- **Training keeps per-task statistics.** Each settled group of a training task is recorded with the workflow's
  verdict on what it trains (`trained`, `uniform`, or why it trains nothing) and what each rollout returned, passes
  judged as `platoon eval` judges success: on the reward the environment scored, before any reward processor, against
  the entry's new `success_threshold` (1.0 by default; a named environment may ship its own), which only these
  statistics read (`platoon eval` keeps `inference.workflow.success_threshold`). Each step logs each label's mean
  expected pass@1 and pass@G under decayed per-task counts (`train/tasks/<column>/<label>/pass_at_1`, `pass_at_G`), the
  share of its tasks seen, its tasks by pass-rate band and those without evidence, and the kinds of the groups it
  trained on (`train/batch/informative_groups`, `all_pass_groups`, `all_fail_groups`, `uniform_mixed_groups`, also by
  label). Each full checkpoint saves the statistics to `tasks/` in the run directory and names the file in its
  `checkpoints.jsonl` row, a resume loads them before its launch save (from a checkpoint without them, it goes on from
  the newest older one that has them, else from empty, and `lock.json` lists the start under `tasks_state_lost`),
  recovery pruning deletes a pruned checkpoint's file, and `tasks/events.jsonl` logs every draw, settle and fate. The
  statistics never fail training. `platoon tasks show` and `platoon tasks export` read them
  ([Per-task statistics](docs/how-to/train-and-eval.md#per-task-statistics)).
- **A resume takes up exactly where the run stood.** Each full checkpoint saves where the run's draws stand in a file
  of its own (`tasks/dispatch-s<start>-<checkpoint>.json`, which its row names as `dispatch`), and the groups that
  settled and that no step had trained yet (`tasks/kept-s<start>-<checkpoint>.json.gz`): a resume trains those as they
  are, a failed one as failed, without rolling them out again, draws again first the tasks whose groups were running
  or waiting, and goes on with the order where it stopped. So every position of the order is consumed exactly once,
  trained, dropped as stale, failed or skipped, and an interrupt or a crash no longer reruns hours of a slow
  environment's groups or loses the groups in flight, also when the per-task statistics stopped or could not be
  written. Every full save writes the groups it keeps again, about ten bytes a token, in a thread beside training;
  `checkpoint.keep_recovery` deletes them with their checkpoint. A resume whose sampling inputs changed (the task
  sampler, the batch size, the label column, the entries' order, an entry's weight, seed or tasks) warns, records it
  in `lock.json` (`sampling_changed`) and goes on; one from a checkpoint of an earlier platoon draws from its batch,
  as before ([Checkpoints and resume](docs/how-to/train-and-eval.md#checkpoints-and-resume)).
- **`train.task_sampler: uniform`** (`train.task_sampler` is the new name of `train.task_order`, below) draws every
  training task of every entry in seeded shuffled passes, with no entry slots: every task equally often, as
  `weight: tasks` does, with batches as the shuffle mixes them rather than balanced by entry. It reads no entry's weight
  or seed: a config that gives an entry a numeric weight does not load (exit 78). For a single environment it draws the
  batches `auto` draws, in epochs that hold every task
  ([Train on several environments](docs/how-to/train-and-eval.md#train-on-several-environments)).
- **Harbor verifiers that run as root.** Some Harbor tasks grade as root (`[verifier] user = "root"`) and run the
  agent's code as another user, nobody as a rule, so that it cannot read the files the grader keeps for itself, such
  as held-out answers. In a rootless sandbox their tests run as your own uid, which cannot switch users: they score
  every attempt 0, the reference solution's included, or run the agent's code where it can read those files.
  `verifier_user: honor` now runs such a task's separate verifier as root, in a sandbox started with
  `verifier_provider_options` laid over `provider_options`: on Singularity, `{rootfs_layout: copy, fakeroot: true}`,
  whose `--fakeroot` maps root to your uid and your subordinate ids to the other users, so nothing gains a privilege
  on the host. On Singularity it needs hpc-sandbox 0.4, whose provider takes `fakeroot` and gives back what the other
  users write before it deletes a sandbox, and which the `hpc` extra now installs (below); with an older hpc-sandbox,
  such as 0.3.0, such a run is refused.
  `verifier_user` (`refuse`, `ignore` or `honor`) decides for `[verifier].user` when it is set, and without it
  `agent_user` decides for both users, as before. With a fetched dataset, set `refuse: {agent_user: false}` too: the
  fetch otherwise leaves out every task that names a user, and a run with `honor` warns of the root verifiers it left
  out. A run refuses, when it lists its tasks (exit 78), a root verifier whose sandboxes cannot give it root and the
  other users (`singularity` without `fakeroot`, or with an hpc-sandbox that does not take it, `enroot`, which maps only
  your uid, and `process`), and, for a fetched dataset, a `[verifier].user` that `honor` cannot honor (another user
  than root, or a root verifier of a task graded in its agents' own sandbox), which a dataset directory skips with a
  warning. Every started root verifier's sandbox is checked to run as root with every user up to nobody mapped, and
  to let nobody run the image's shell, else the grade is an `error` (`verifier_start`). A root verifier grades as
  Docker would: with umask 022, Docker's modes for the directories platoon makes and for a working directory the
  image lacks, the image's own modes for those it has, its artifacts readable by every user, and the outcome records
  `verifier_user: root`. Separate verifiers' sandboxes are reaped with their own options too
  ([Verifiers that run as root](docs/how-to/sandboxes-and-harbor.md#verifiers-that-run-as-root)).
- **The `openhands` harness condenses at a token count.** Its new option `condenser_max_tokens` gives its summarizing
  condenser a token trigger, and runs that condenser whatever `condenser` says: before each model call, it condenses
  once the prompt the agent would send is longer than that many tokens, summarizing the oldest turns after the system
  prompt and the task until the rest fits in half the trigger. `condenser_max_tokens: 104857` condenses at 80% of a
  131072-token context window; a trigger that a prompt passes with nothing left to forget, such as one below the
  agent's first prompt, can never be met, and is set aside with a line in the harness log. The runner asks the
  gateway for each count (`POST /v1/messages/count_tokens`), so it is the length of the prompt the gateway renders
  with the policy's renderer; where a hosted model's provider gives no count, the SDK's estimate counts, and a prompt
  that still overflows the window condenses as before. The summaries are the agent's `aux` calls (below), trained
  with the rollout's advantage. The default, 0, sets no trigger and counts no tokens
  ([`openhands`](docs/tutorials/builtin-harnesses.md#openhands)).

### Changes

- **A dispatcher starts each training group when it can start.** The producer that drew a batch's tasks ahead of the
  rollout workers, the pending queue and the pool of workers are replaced by a dispatcher that draws a task only when
  its group can start: within the horizon of `train.max_staleness` (at most `max_staleness + 1` batches of groups
  started and not yet trained), with rollout permits for the whole group, and with room in its label's
  `rollout_group_limits`. A run without `rollout_group_limits` trains the groups it trained before; at
  `max_staleness: 0` each step trains exactly the groups of its batch, unless a label's limit holds one of them back.
  The run starts no group past its last step ([Settings per
  environment](docs/how-to/train-and-eval.md#settings-per-environment)).
- **Breaking: a label at its `rollout_group_limits` no longer stalls the others, and its groups start as its limit
  allows, not as the steps do.** In every config that sets limits, a draw whose label is at its limit waits in the
  label's backlog, outside the horizon: it neither takes a place in a batch nor counts toward the batches
  `train.max_staleness` lets the run draw ahead. Past four batches of them, the label's new draws are skipped and
  counted, and the run draws only while some label has room. While a capped label has draws waiting, each of its
  running groups holds a slot of the horizon for the next one, which starts as soon as the group ends, as far as the
  capped labels' groups and those slots fit in `max_staleness` batches; at `max_staleness: 0` there is no slot to
  hold, and a capped label starts at most its limit of groups a step. So a capped label whose limit cannot keep up
  with its share trains less than its weight, and its draws still waiting when the run ends never start;
  `train/batch/parked_rows` counts the draws that wait, each step, and `train/batch/skipped_rows` those skipped.
  `platoon train` and `platoon config validate` warn of a `max_concurrent_rollouts` that the capped labels' groups can
  fill. There is no switch: in 0.2, a label at its limit held back the other labels' rows, and so paced the whole
  run. This changes what a config with `rollout_group_limits` trains: its capped labels' draws that wait are drawn
  past at every bound, `max_staleness: 0` included
  ([Settings per environment](docs/how-to/train-and-eval.md#settings-per-environment)).
- **The `openhands` harness's condenser keeps the task and shows its summary call what happened.** The spec is now
  `openhands@1.54.0.post1`, from the same toolchain, so no site builds it again, and a config that names
  `openhands@1.54.0` gets it; with the default options (no condenser and no token trigger), the agent runs as before.
  With `condenser: llm_summarizing` or a token trigger, the summary call reads each forgotten turn whole: the commands
  it ran, their whole output and an earlier summary entire, where the SDK's condenser showed it 500 characters of each
  and no tool call's arguments, so a summary call's prompt is now as long as the turns it forgets. A condensation that
  resets the whole context keeps the system prompt and the task, which the SDK's summarized away with the rest, and
  one for the token trigger forgets only what it takes to get under half the trigger. A summary call answered without
  text (a tool call, or a completion cut short in its reasoning) still condenses, with a notice in the summary's place
  that the earlier turns were dropped, where the SDK's condenser dropped them and said nothing. A context reset whose
  summary call fails, such as one the gateway refuses for overflowing the context window, is made again with the
  turns cut shorter each time, and if every try fails, the turns are dropped with the same notice, where the SDK's
  condenser stopped the agent. No option brings back the SDK's condenser.
- **The `openhands` condenser's summary calls are `aux` calls.** The runner marks each summary call with
  `X-Platoon-Call-Kind: aux`, and the agent's own turns stay `policy` calls. A summary's trajectory step now says
  `kind: aux` in its `action_misc`, and an evaluation report counts the summaries under `usage.aux` rather than
  `usage.policy`. They train as before, with the rollout's advantage, each condensation still takes one of the
  agent's steps (`max_steps`), and the step counts in training and evaluation metrics still count each summary as a
  step. In a task sandbox, the gateway takes them under the rule below.
- **A harness in a task sandbox may make `aux` calls.** In 0.2 the gateway refused every call but a policy call from a
  harness in a task sandbox, where code the policy runs can read the agent's key. It now takes its `aux` calls, unless
  a gateway step budget (`[budget] enforce = "gateway"`) counts its calls: that budget does not count `aux` calls, so
  under it they are still refused (403). Without one the gateway counts none of the harness's calls, and it samples
  the harness's `aux` calls as its policy calls, also under `sampling_precedence: config`, where an `aux` call
  elsewhere keeps its own parameters; so an `aux` call gets that code nothing that a policy call would not. Its
  `judge` calls are still refused in every case
  ([how a run ends](docs/how-to/sandboxes-and-harbor.md#3-how-a-run-ends)).

### Fixes

- **The `oci` image builder keeps the permissions an image gives other users.** Run by anyone but root,
  `unsquashfs` gave the files it unpacked the image's modes through the host's umask, so under a umask such as 027
  every file of an image the `oci` builder made lost the read and execute bits the image gives other users. Rootless
  sandboxes never noticed, since everything there runs as the files' owner, but a verifier that runs as root and
  switches users could not even run the image's shell. The builder now unpacks under umask 0, in a build directory
  only you can enter, and records `builder_revision` in its receipts. Images it built before are not built again by
  themselves, since their build keys do not change: with `verifier_user: honor`, `platoon train`, `eval`, `rollout`
  and `prepare` refuse the root verifiers' images among them, naming the build that makes them again
  (`platoon harbor images build CONFIG --key KEY`)
  ([Verifiers that run as root](docs/how-to/sandboxes-and-harbor.md#verifiers-that-run-as-root)).

### Removed

- **Breaking: the forms that platoon 0.2.0 deprecated are gone.** A config, an override or a command that still uses
  one fails as a configuration error (exit 78), with a message that names what replaces it and says that platoon 0.3.0
  removed it, such as `train.rollout_group_key: unknown key; platoon 0.3.0 removed it; use train.label_by`
  ([Write and override configs](docs/how-to/configs.md#4-check-the-result-before-you-run-it)). Each form, with what
  replaces it:
  - `rollout_config.propogate_root_success`, the misspelling: `propagate_root_success`;
  - `train.rollout_group_key`: `train.label_by`;
  - `workflow_config.straggler_policy_key` of `train` and `eval`: `train.label_by`, whose labels straggler policies
    key on;
  - `num_concurrent_rollout_workflow_workers` of `train` and `eval`, which counted rollout groups:
    `max_concurrent_rollouts`, which counts rollouts (the old value times the section's `workflow_config.group_size`);
  - `strategy: none` of `eval` and `checkpoint`: `every: 0` (`strategy: step` and `epoch` stay, below);
  - a Harbor entry's `cache`, `image_map` and `toolchain_root`, and the variables `PLATOON_HARBOR_CACHE` and
    `PLATOON_TOOLCHAINS`: the site profile's `cache`, whose `harbor` directory holds the Harbor cache with each fetched
    dataset's image map (`images/<provider>.json`), and whose `toolchains` directory holds the harness toolchains.
    A command that loads a config, or takes the cache from the site, refuses to run while one of the variables is
    set. An entry that serves a dataset directory has no image map: its tasks run their `docker_image` or
    `default_image`, and naming the directory as a `dataset` to fetch has `platoon prepare` build their images;
  - fetch config files of `platoon harbor`, and `platoon harbor fetch`'s `--lock` and `--cache`, which applied to them
    alone: a Harbor entry of a training or evaluation config, and `--workers` for their `workers`;
  - `python -m platoon.train.tinker.restart_wrapper` and `run_with_restart`, whose import raises an `ImportError`:
    `watchdog.max_restarts` in the training config, with which `platoon train` restarts training after any failure.

### Deprecations

- **`train.task_order` is now `train.task_sampler`,** which names how a run draws its training tasks and takes the same
  values: `auto`, the default, draws a mixture in its weighted fair order and a single environment in the dataloader's
  shuffled passes, and `weighted` draws a single environment in the weighted order too. The order a run draws is
  unchanged. `train.task_order` still loads, with a deprecation warning, until platoon 0.4.0 removes it; `lock.json`
  records it and `platoon config validate --strict` fails on it. Beside another `task_sampler`, `task_order: auto` (its
  default in 0.2.0, as a dump of that release's `config show --defaults` spells it) is ignored, with a warning that says
  so, and `task_order: weighted` does not load
  ([Train on several environments](docs/how-to/train-and-eval.md#train-on-several-environments)).
- **`strategy` is no longer deprecated.** Platoon 0.2.0 deprecated the `strategy` of `eval` and `checkpoint` for removal
  in 0.3.0; it stays instead, as the unit of `every`, and loads without a deprecation warning. `every` counts in its
  units: `step`, what a section that leaves it out counts, or `epoch`, a pass over the training tasks, so
  `eval: {strategy: epoch, every: 1}` evaluates at the end of each epoch. A config that sets it keeps the meaning it has
  had: a section that sets `strategy` and leaves `every` out triggers its event every 1 of its units, and one that sets
  neither never does. `strategy: none`, which `every: 0` duplicates, is gone (above)
  ([Cadences in steps or epochs](docs/how-to/train-and-eval.md#cadences-in-steps-or-epochs)).

### Install and upgrade

```bash
pip install -U 'platoon[tinker,ors]>=0.3,<0.4'
# with the hpc extra and the TextCraft extension, name them too:
pip install -U 'platoon[tinker,ors,hpc]>=0.3,<0.4' 'platoon-textcraft[platoon]'
```

- **Name every extra and extension you installed.** pip upgrades only what the command names and what that needs, and
  it does not record the extras a package was installed with, so `pip check` passes either way: an upgrade without
  `hpc` keeps hpc-sandbox 0.3, and one without `platoon-textcraft` keeps its 0.1.0 (both below).
- **Replace what 0.3.0 removed** (above): `platoon config validate` reports every removed key or value that a config
  file still sets, each with what replaces it, and refuses to run while a removed variable is set, naming what replaces
  it ([Check the result before you run it](docs/how-to/configs.md#4-check-the-result-before-you-run-it)). A script that
  runs `python -m platoon.train.tinker.restart_wrapper`, `platoon harbor` on a fetch config file, or
  `platoon harbor fetch` with `--lock` or `--cache` stops with exit 78 too, naming what replaces it.
- **Check what trains differently with the same config** (Changes, above): a config with `rollout_group_limits` no
  longer lets a capped label pace the run, so a capped label whose limit cannot keep up trains less than its weight, and
  its new draws are skipped once four batches of them wait; an `openhands` harness with `condenser: llm_summarizing` or
  a token trigger shows its summary call the forgotten turns whole, and an evaluation report counts the summaries under
  `usage.aux`; and a harness in a task sandbox gets its `aux` calls answered and trained, where 0.2 refused them, unless
  a gateway step budget counts its calls. None of these has a switch back.
- **To upgrade a project that pins one minor release**, move its range to 0.3, such as `platoon[tinker,ors]>=0.3,<0.4`.
  Platoon 0.3 refuses the components of an installed extension whose requirement on platoon excludes it, such as
  `platoon>=0.2,<0.3`, with an error that names both versions, until a release of the extension admits 0.3;
  meanwhile, `PLATOON_IGNORE_EXTENSION_REQUIREMENTS=1` loads them anyway
  ([Compatibility and provenance](docs/extending/index.md#compatibility-and-provenance)).
- **The `hpc` extra takes hpc-sandbox 0.4** (`hpc-sandbox>=0.4.0,<0.5`), whose singularity provider has the
  `fakeroot` option that verifiers that run as root need (above). An upgrade that names the extra upgrades an
  hpc-sandbox 0.3 it finds. One that does not keeps it, and platoon 0.3 then refuses `fakeroot`, and with it the root
  verifiers on Singularity, with a message that names hpc-sandbox 0.4. Sandboxes started without `fakeroot` (or
  `--fakeroot` in `extra_start_args`) behave as with hpc-sandbox 0.3.
- **The TextCraft extension's release for platoon 0.3 is `platoon-textcraft` 0.2.0**, whose `platoon` extra requires
  `platoon[ors]>=0.3,<0.4`. Its code and data are those of 0.1.0, but for the dataset card that its Harbor export
  writes, whose example config for platoon no longer sets `image_map`, which platoon 0.3.0 removed. The release for
  platoon 0.2 stays 0.1.0: there, keep the extension within 0.1 (`platoon-textcraft>=0.1,<0.2`), since platoon 0.2
  refuses the components of 0.2.0, also when it is installed without its extra. An upgrade of platoon alone leaves
  0.1.0 installed, which pip does not flag and platoon 0.3 refuses: upgrade the two together, as above
  ([The TextCraft extension](docs/getting-started/install.md#the-textcraft-extension)).
- **Every training run keeps per-task statistics and where its draws stand, with no switch** (above). Its run
  directory gains `tasks/`: with each full checkpoint, a snapshot of the statistics, which its row of
  `checkpoints.jsonl` names (`tasks`), the state of its draws (`dispatch`) and the groups it keeps, about ten bytes a
  token (with `max_staleness: 3`, batches of 32 groups of 8 and rollouts of 16,000 tokens, a save can keep some 130 MB);
  and `tasks/events.jsonl`, a line for each draw, settle and fate, which grows with the run. `checkpoint.keep_recovery`
  deletes a pruned recovery checkpoint's files with it; those of permanent checkpoints stay, as the checkpoints do, and
  without it so do all. Each step logs more series for each label. A run that platoon 0.2 started resumes from its
  checkpoint's batch, as before, with empty statistics and a warning, and its `lock.json` lists the start under
  `tasks_state_lost`.
- **Code of your own.** The rows a workflow gets carry two more keys, `_platoon_seq` and `_platoon_position`, which
  number the training rows the trainer draws, and a loaded `TrainConfig` names its sampler in `task_sampler` alone,
  with `task_order` None. Inside a group, a `GroupRolloutWorkflow` subclass's `self.config` holds the settings of the
  group's entry, which `platoon.train.tinker.workflows.group_rollout_workflow.current_entry()` names; a workflow class
  that is no `GroupRolloutWorkflow` must apply an entry's settings itself and declare `applies_entry_settings = True`,
  or a config whose entries set any does not load into `platoon train` (exit 78). A workflow class of your own that
  returns a plain list of datums gives the per-task statistics no evidence, and one that returns neither that nor a
  `GroupRolloutWorkflow`'s output has its task rolled out again on a resume, rather than its group kept. A sandbox
  provider may say why its sandboxes cannot give a root verifier root and other users, with a `root_refusal(options)`
  class method.

### Documentation

[RFC 0019](docs/rfcs/0019-training-mixtures.md), a draft, designs training mixtures' per-environment settings, per-task
statistics and task samplers. This release ships its first milestones: `train.task_sampler`, settings per environment,
per-task statistics, and the dispatcher with exact resume and the run-level `uniform` sampler.

## 0.2.1 (2026-10-19)

### Fixes

- **A busy node no longer makes the trainer kill rollout workers that were already forked.** On a node whose CPUs are
  all busy, the trainer's event loop could be held up for longer than the fork server's one-minute deadline. When it
  ran again, it gave up on workers the fork server had forked meanwhile, killed them and started them again by exec,
  each importing the worker's stack itself, which loaded the node further. The trainer now reads what the fork server
  sent before it judges a deadline, so an answer, or the hello of a server that finished importing, is used however
  late it is read, and a fork fails only once the server has answered no request for a minute (one that answers the
  requests before it is working through them, in order). A fork server still importing the stack after the worker's
  startup budget is no longer killed and replaced by another that imports it from the start while the workers wait for
  that one: they start by exec meanwhile, and the server forks the next ones once it has imported the stack. If it
  then finds that it cannot serve (a module did not import, a thread runs in it), the trainer logs why and starts no
  other, as for a server that says so in time.

## 0.2.0 (2026-10-19)

The first release on PyPI, numbered 0.2.0 because the original platoon gave its version as 0.1.0, so that the
two never share one. It replaces the original platoon, whose AReaL-based trainer, standalone inference
scripts and plugin layout are gone. The original code is at the tag
[`archive/v0.1`](https://github.com/ApGa/platoon/tree/archive/v0.1), and the code that the
[Recursive Agent Optimization (RAO)](https://arxiv.org/abs/2605.06639) paper's experiments ran on stays on the
[`apga/rao-snapshot`](https://github.com/ApGa/platoon/tree/apga/rao-snapshot) branch.

### What it does

- **Trains agents with reinforcement learning through a Tinker-compatible training service**, hosted Tinker or a
  self-hosted server such as [miles-tinker](https://github.com/neulab/miles-tinker). `platoon train` runs groups
  of episodes per task, builds training data from the exact sampled tokens and centers rewards within each group
  (mean or leave-one-out baselines, with depth weighting for recursive agents). Each step logs its reward and its
  rollouts' outcome counts, and a rollout whose environment could not open its session counts as an error.
  `train.label_by` names the dataset column that labels the tasks, by default the mixture entry: each step's metrics
  and each evaluation's break down by its labels (the 64 most frequent), `train.rollout_group_limits` and the
  straggler policies of training and evaluation key on them, and a column that no training task has stops the run
  before it starts ([Label tasks by a column](docs/how-to/train-and-eval.md#label-tasks-by-a-column)).
- **Any harness.** A harness is a Python function (`@pt.harness`) or a command-line agent described by a TOML
  spec. The built-ins are `platoon-codeact`, `platoon-react`, `platoon-bash`, `mini-swe-agent`, `openhands` (an
  OpenHands SDK agent), and `platoon-oracle` and `platoon-noop`, which run a task's reference solution or nothing,
  to check a dataset before training.
- **One harness per agent, configured by options.** A TOML spec declares its options in an `[options]` table (type,
  default, description, allowed values or range), which `platoon harness show` lists and its launch renders; a
  config sets them in an environment's `harness_options`, on the command line like any key
  (`environments.0.harness_options.num_retries=5`), and for evaluation rollouts only in `eval_overrides`. They are
  checked before anything runs, an unknown or badly typed one is a configuration error (exit 78) that names its
  key, and `lock.json` records the options each harness ran with, defaults included. `openhands` takes
  `reasoning_replay`, `request_timeout_s`, `num_retries` and `condenser`: its defaults train, and Harbor's
  `openhands-sdk` adapter's settings are an evaluation's options, from the same toolchain
  ([RFC 0016](docs/rfcs/0016-component-configuration.md)).
- **Harnesses inside task sandboxes.** A command-line harness runs in a task's own sandbox, also one without a
  network: the gateway serves each rollout on a Unix socket of its own, which a route relays into the sandbox, and
  the harness runs from a toolchain built once per site (`platoon toolchain build`), under `toolchains` in the site
  profile's cache, else `~/.cache/platoon/toolchains`, and mounted read-only; a run never builds one, and refuses to
  start, naming the `platoon prepare` command, while it is missing. A spec's `[budget] enforce = "gateway"` caps its
  policy calls at the gateway, a harness in a task sandbox may make only policy calls, `on_harness_exit` says whether a
  harness that ran out of time or crashed is graded or left out, and `rollout_config.sampling_precedence: config` has
  policy calls sample as the config says, never as a harness asks.
- **A recording model gateway.** Every model call, from Python (`ctx.chat`) or from any client over
  OpenAI-compatible HTTP, is rendered, sampled, parsed and recorded with its exact tokens. The OpenAI Responses and
  Anthropic Messages dialects are experimental. An MCP tool plane serves each agent the environment's tools. A
  sampling call fails after the sampling timeout, which a config sets beside its model
  (`train.sample_timeout_seconds`, `inference.sample_timeout_seconds`); without the key, `TINKER_SAMPLE_TIMEOUT_SECONDS`
  sets it, else 600 s. The key wins over the variable, which a gateway job may export for every client it runs, a
  training run's rollout workers sample under the trainer's bound, and `platoon train` and `platoon eval` log the
  bound with where it came from ([The sampling timeout](docs/how-to/train-and-eval.md#the-sampling-timeout)).
- **Evaluation with hosted models.** `platoon eval` and `platoon rollout` can evaluate a model behind any server
  that speaks the harness's own dialect: `inference.hosted` names the base URL and the variable that holds its key,
  and `inference.model_name` the model. The gateway forwards each call unchanged and treats tools as training does,
  so trajectories, step budgets and usage are still recorded, and the key never reaches the agent's sandbox.
  `inference.judges: false` turns off the model calls that judge agents rather than solve the task
  ([Evaluate a hosted model](docs/how-to/evaluate-hosted-models.md)).
- **Tool calls for any chat template.** The `chat_template` renderer renders with the model's own template and
  reads tool calls with a parser chosen by name, by a round trip through the template (`tool_call_parser: auto`),
  or written by you (`module:attr`).
- **Environments as services:** OpenReward (ORS) environment classes, in process or on environment servers;
  Harbor task directories, graded by their own tests in `process`, Docker, Singularity/Apptainer or enroot
  sandboxes (enroot takes the Singularity provider's option names); step environments through `pt.gym`; and task
  tables. An environment entry's `exclude_tasks` leaves out tasks that cannot be solved or graded as written, and
  `lock.json` records which.
- **Every training task trains.** A single environment's tasks are drawn in seeded shuffled passes that go on from one
  epoch to the next: an epoch holds the whole batches its tasks fill, and the tasks left over start the next epoch
  instead of being left out of every one. `environments` may also list several entries, each with its own service,
  harness, program and options. Training interleaves their tasks in a weighted fair order: `weight` is `tasks`, the
  default, each entry's number of training tasks, which draws every task of the mixture equally often, or a number,
  and the entries share the slots in proportion. The epochs hold as many tasks as the entries do, filled to whole
  batches, each going on with the order where the one before it stopped, so an entry that gets fewer slots than it has
  tasks draws the rest in the next epochs, not the same ones every epoch. A resumed run draws the batches the run it
  resumes would have drawn. `platoon train` logs each entry's share and how often each of its tasks is drawn over the
  planned steps, and warns when the weights leave an entry's tasks out of the run or draw one entry's tasks more than
  4 times as often as another's; `platoon config validate --tasks` lists the tasks and prints the same before a run
  ([Train on several environments](docs/how-to/train-and-eval.md#train-on-several-environments)).
- **Harbor datasets as published.** A Harbor environment names a dataset (`hf://org/name@commit`, or a local
  directory) that `platoon harbor fetch` downloads once into a content-addressed cache and pins with a lock, which
  rollout workers read offline; `platoon harbor images` builds the tasks' images once per site from their
  Dockerfiles, sharing one image among tasks whose build steps match, and an entry that names no image map uses
  `images/<provider>.json` in its cache, unless it serves a directory. Tasks may sit at any depth under a registry,
  a separate verifier grades a task's artifacts in a sandbox of its own, and a dataset's reward contract says which
  of its rewards trains. A dataset whose tasks report that reward under different names is one environment: the
  contract lists the names (`reward: [reward, correctness]`), and each reading must report exactly one of them, from
  one reward file; a group whose rollouts trained different names does not train (`reward_key_mismatch`). Each
  rollout's record names the reward key it trained (`misc.outcome.reward_key`), each step counts its rollouts by it
  (`train/batch/reward_key/<name>`), and `platoon eval`'s report by `reward_key_counts`. A fetch records index
  fields from the dataset's files and, with `index: [{metadata: [...]}]`, from each task's `task.toml` `[metadata]`
  table, records what each task's sessions upload and the digest of each of its input files, and refuses tasks that
  upload more than `refuse.uploads_over_mb`. `holdout` holds out whole groups of a dataset's tasks for evaluation, a
  share of each source, and lists them as the `eval` split and the rest
  as `train`, with a report of each split's makeup; tasks that share an input file (a text file compared as text,
  whatever its line endings) are always in one group, so no held-out task holds data that training has seen, and
  `holdout.ignore_inputs` names files that every task holds by design
  ([One dataset, one environment](docs/how-to/sandboxes-and-harbor.md#one-dataset-one-environment)). A run's
  dataset has each task's index fields as columns, as text, which `train.label_by` can name; a field named like
  one of platoon's own columns is left out, and a configuration error when `label_by` names it.
  `platoon harbor fetch`, `verify` and `images plan|build|verify|alias` take a training or evaluation config, with
  overrides, and act on each Harbor entry's dataset as the run would, `images` with the entry's image map and its
  sandbox provider's builder; `fetch` leaves a dataset pinned to a commit alone when its lock is current (`--repair`
  fetches anyway), and fetches a lock written before input files were recorded once more. Fetches of one lock, and
  builds of one image, take turns on file locks, also on several nodes, so each runs once. Fetch config files are
  deprecated: an entry of the run config says the same. A build step that fetches remote content, an `ADD` of a URL
  or of a git repository, also one whose URL an `ARG` or `ENV` value holds, is built as Docker builds it: the `oci`
  builder runs it with BuildKit, which verifies its `--checksum` and does not extract a remote archive, the build key
  covers it as written, and the `def` builder, which cannot fetch, refuses it and names `oci`. The `oci` builder also
  builds a Dockerfile whose `# syntax=` directive names a 1.x release of Docker's Dockerfile frontend, with the one
  built into SingularityCE's BuildKit, which cannot load a frontend image; it refuses any other frontend.
- **One command prepares a config.** `platoon prepare CONFIG` makes, once per site, what the config's runs need and
  never make themselves: each Harbor dataset fetched into its lock, the images its tasks build, with the builder of
  its sandbox provider, into its image map, its harnesses' toolchains, and the policy's tokenizer, into default
  locations under the site profile's cache, else `~/.cache/platoon`. Each step does nothing once current, two prepares
  at once take turns, also on two nodes, and `--dry-run` says what is missing
  ([Prepare a config](docs/how-to/sandboxes-and-harbor.md#prepare-a-config-once-per-site)). It never replaces a
  toolchain built from another recipe, which a run of another config may be running from: the step names the forced
  build that does. Before any rollout, `platoon train`, `eval` and `rollout` refuse (exit 78) a config whose toolchains,
  or whose fetched tasks' images, are not built, naming the `platoon prepare` command for their config, overrides
  included. On a laptop or a single node, `--prepare` on any of them runs that command first, with its lines on stderr,
  and starts the run only once the config is ready, else exits with prepare's status; on a cluster, prepare once on a
  CPU node instead, since in a GPU job the flag spends GPU time building.
- **One run per run directory.** A run's first start writes `identity.json`: its model, renderer and held-out
  tasks. A resume that would change them stops with exit 78 before it opens the training service, and records why in
  `lock.json`; training that evaluates warns when its evaluation lists training tasks
  ([Checkpoints and resume](docs/how-to/train-and-eval.md#checkpoints-and-resume)). `platoon eval` writes each start to
  a new directory, `runs/eval/<policy>/<UTC time>` (the checkpoint, else the model), unless `inference.output_dir` names
  one, and never resumes another policy's rollouts: a directory whose rollouts another model, checkpoint, renderer or
  hosted base URL made, or with judges switched, stops it with exit 78 before anything in the directory changes
  ([Evaluate](docs/how-to/train-and-eval.md#3-evaluate)).
- **Cadences in training steps, concurrency in rollouts.** `eval.every` evaluates the eval split every that many
  training steps, and `checkpoint.every` saves a permanent checkpoint as often; 0, the default of both, does neither,
  while the launch, recovery and final checkpoints are saved either way
  ([Checkpoints and resume](docs/how-to/train-and-eval.md#checkpoints-and-resume)). Each evaluation logs the step it
  started at (`eval/step`), and the first is at step `every`, as in the original platoon: `eval.at_start: true`, false
  by default, also evaluates the starting policy, at step 0, and holds the first update until that evaluation ends, so
  that the held-out curve starts at the base model
  ([Evaluate during training](docs/how-to/train-and-eval.md#evaluate-during-training)). A start that resumes, by hand or
  after a failure, evaluates at the multiples of `every` from the step it resumes at, and that step only if the run
  directory's `evaluations.jsonl`, which records each evaluation that completed, does not list it: that step's
  evaluation runs again if a crash cut it short, and does not run twice
  ([Evaluation in a resumed run](docs/how-to/train-and-eval.md#evaluation-in-a-resumed-run)).
  `train.max_concurrent_rollouts` and `eval.max_concurrent_rollouts` count the rollouts that run at once, a whole group
  at a time: by default every rollout of a batch in training, and 256 in an evaluation. **Breaking:** an `eval` or
  `checkpoint` section that sets neither `every` nor the deprecated `strategy` no longer evaluates, or saves a permanent
  checkpoint, each epoch, and `every` alone no longer counts epochs: set it in steps (an epoch is the training tasks
  over `train.batch_size`). An evaluation's default concurrency counts 256 rollouts, not 256 groups, which differs once
  its `group_size` is above 1.
- **Evaluation that rolls out as training does.** A training config's `eval.workflow_config` is laid over
  `train.workflow_config`: an evaluation during training keeps training's rollout settings (`rollout_config`: step
  and time limits, sampling, `extra`) and rollout-worker settings (`rollout_isolation`, `worker_start_method` and the
  other `worker_*` keys) unless it sets them, while the keys that shape groups, advantages and filters keep
  evaluation's defaults (one rollout per task, no filters, no straggler cutoff). A key it sets replaces training's;
  `rollout_config` and its `inference_params` merge key by key, `null` unsets a value training sets, and `extra`
  replaces training's whole ([Evaluate during training](docs/how-to/train-and-eval.md#evaluate-during-training)).
  **Breaking:** an evaluation that relied on evaluation's defaults for these settings (the harness's own step limit,
  no time limit, the gateway's sampling and 4,096-token completions, in-process rollouts) now takes training's; set a
  key under `eval.workflow_config` to keep the old value.
- **Harbor grading that agents cannot forge.** Grading first kills every process the agents left running
  (`Sandbox.kill_processes`, part of the sandbox provider contract). Where the tests run in the sandbox the agents
  worked in, it makes the tests' and the reward's directories afresh; a separate verifier gets the task's artifacts
  as directories and regular files only. In either, grading kills what the tests left running and reads only
  regular reward files. A reward file of any other kind scores 0, and so does an attempt whose agents removed,
  moved or locked their workdir (`workdir_missing`), without running the tests; a grade whose processes cannot all
  be killed is an untrainable error
  ([How grading keeps the agent out](docs/how-to/harbor-grading.md#how-grading-keeps-the-agent-out)).
- **Nothing a stopped rollout started keeps running.** Rollout workers label the sandboxes they start, and the
  sandboxes a killed or hung worker left are reaped by label. Every sandbox of a run also carries the run's
  directory as its `platoon.run` label, so a job can reap what its own run left and no other run's. A rollout's
  deadline counts its sandbox starts and its grading, so a slow verifier does not kill the worker that grades it.
  Without a `rollout_config.timeout`, the deadline counts the harness's own time limit where the environment says
  what a rollout needs beyond it, as Harbor's does: the earliest of its spec's `timeout_s` and a Harbor task's
  `[agent] timeout_sec` (times `agent_timeout_multiplier`). So a Harbor config needs no `timeout` to bound its
  rollouts, and an evaluation's worker processes (`use_subprocesses`) let a harness run as long as its task allows.
- **Rollout workers that start in seconds.** With `rollout_isolation: process`, a rollout's worker is forked from a
  fork server that imported the worker's stack and built the run's tokenizer and renderer once
  (`worker_start_method: forkserver`, the default), instead of a new interpreter that imports it (`exec`: about 20 s,
  a minute when 64 start at once on one node). A forked worker is the process an exec'd one would be, and one the
  fork server cannot fork starts by exec. On GPU nodes too: neither the fork server nor a worker sees a GPU or starts
  the CUDA driver, whose thread would keep the fork server from forking. The singularity provider's
  `rootfs_layout: fuse-overlay` (hpc-sandbox 0.3) starts a sandbox in about a second over one extraction of its image
  per node, and a run that sets a layout its provider cannot use is refused when its tasks are listed
  ([The singularity provider's root filesystem](docs/how-to/sandboxes-and-harbor.md#the-singularity-providers-root-filesystem)).
- **Runs that start again by themselves.** With `watchdog.max_restarts: N`, `platoon train` trains in a child process
  and starts it again from the newest checkpoint after any failure but a configuration error (exit 78) and an
  interrupt: the watchdog's exit, a crash, a training service that stalls or dies. It stops after `N` restarts, or after
  three starts in a row that saved no new checkpoint. A start asks the training service for its training client until
  the service accepts it, for up to 20 minutes, logging each refusal, so no delay is needed while a service still holds
  a dead start's session. `platoon train` reaps the sandboxes labelled with its run when it starts, before each restart
  and when it exits, `platoon eval` those labelled with its output directory when it exits, and `platoon sandbox reap
  RUN_DIR` those of a run that was killed; each logs how many. A run directory records the process that runs it
  (`process.json`): `platoon sandbox reap` refuses a run that still runs, and `platoon train` and `platoon eval` a
  directory that another of their processes runs. `python -m platoon.train.tinker.restart_wrapper`, which restarted on the watchdog's exit
  only, is deprecated, and 0.3.0 removes it ([Long runs](docs/how-to/train-and-eval.md#long-runs)).
- **Subagents.** An agent starts subagents with `ctx.agents.spawn`, under step budgets and depth limits, with
  forked environment state and optional verifiers. One rollout trains every agent in the tree, and episode
  programs (`@pt.program`) compose episodes and shape rewards. The default program, `single`, gives subagents the
  root's success when `propagate_root_success` is set (after scoring them 0 when `skip_subagent_reward_computation`
  is), and `platoon-delegation` is a reward processor for recursive agents.
- **Outcomes that say nothing about the policy** (a crashed grader, a dead sandbox, a timeout) never train as a
  reward of 0, and do not count toward a group's `min_successful_group_size`.
- **A Tinker key that is missing or refused is one line.** `platoon train`, `eval` and `rollout` check
  `TINKER_API_KEY` before they prepare, write or reap anything, and without a key that starts with `tml-` stop with
  one line and exit 78; a hosted model needs the variable its `api_key_env` names instead, checked as early, and a
  report, or an evaluation or rollout that serves no policy, needs neither. A key that the Tinker service refuses
  (HTTP 401, or 403 for what the key may not do) ends each of them with one line that names the key and the service,
  instead of the client's traceback, and exit 78: `platoon eval` starts no rollout after it, and a supervised
  `platoon train` does not restart.
- **Recovery checkpoints that do not fill the disk.** With `checkpoint.keep_recovery: N`, training deletes its
  recovery checkpoints beyond the newest N through the Tinker API, beside training and on a Tinker session of its
  own: never a permanent checkpoint, the one a resume starts from, one that rollouts may still sample from (within
  `train.max_staleness` policy versions) or another run's. A config whose `keep_recovery × recovery_every` does not
  exceed `max_staleness` does not load (exit 78). A resumed run goes on where it left off
  (`checkpoints_deleted.jsonl`); failed deletes are counted (`checkpoint/delete_failures`), and three in a row, or a
  service that cannot delete checkpoints, stop the deletes with a warning while training goes on
  ([Checkpoints and resume](docs/how-to/train-and-eval.md#checkpoints-and-resume)).
- **The same training however an environment runs.** A tool's failure is graded alike in process and on an
  environment server, a served class named by `service_kwargs.env_class` shows agents the same tools as in process,
  and a test plays both modes and compares every token, outcome and datum.
- **Extensions.** Installed packages provide harnesses, episode programs, named environments and reward processors
  by name through entry points, which platoon imports only when a config names them; `platoon extensions list`,
  `show` and `check` list them, describe one and check them ([Extending platoon](docs/extending/index.md)). Each
  ships packages of its own: `platoon` is a regular package, which no extension installs into. The repository's
  first-party extensions (`extensions/`) are built and tested with core, in one uv workspace.
- **`platoon.testing`** runs environments, harnesses and programs on CPU with a scripted policy and a toy
  tokenizer, with no model download.
- **One command line:** `platoon train`, `eval`, `rollout`, `config`, `harness`, `program`, `env`, `extensions`,
  `harbor`, `toolchain` and `viz`, with strict configs that build on each other through `extends:` and load as the class
  their environments' `trainer_config` names, so that a plugin's config sections validate, and a `lock.json` in every
  run directory with the versions of platoon and its main dependencies and the commit of each one installed from git,
  and the images the job pulled for its sandboxes when it names their record
  (`PLATOON_IMAGE_PULLS`). Before anything trains, a run refuses (exit 78) a config whose rollouts could never
  train, such as a TOML harness spec whose settings never apply or whose tools never reach the task, or environment
  grading that cannot grade a subagent's own goal, and one whose `trainer_config` does not resolve or one of whose
  `package` modules does not import. `platoon config validate` resolves every component a config names, as a run
  does when it starts (`--no-resolve` checks the files and named environments alone), takes the run's `key=value`
  overrides, and checks a file of environments alone against the entry schema. An entry's `service_kwargs` are bound
  to what its service takes when the config loads, so a misspelled option fails with a did-you-mean, and a value the
  service refuses when the run builds it is a configuration error that names the entry.
- **A site profile.** Where a site keeps platoon's files and which runtime starts its sandboxes go in one file per
  site, never in a project: `~/.config/platoon/site.yaml`, or the file `PLATOON_SITE` names, with four keys
  ([Write the site's profile](docs/how-to/clusters.md#4-write-the-sites-profile-once)). `cache` is the root of what
  `platoon prepare` makes (`<cache>/harbor`, `<cache>/toolchains`); `runs` holds the run directory of a training
  config without `log_path` (`<runs>/<experiment>/<trial>`, the experiment named after the config file unless it
  names one, a directory that belongs to that file, so that another project's file of the same name is refused rather
  than take over its run) and the directory of each start of an evaluation without `inference.output_dir`
  (`<runs>/eval/<policy>/<UTC time>`), while a relative `log_path` a config names stays relative to the working
  directory; `provider` and `provider_options` start the sandboxes of every Harbor entry that names no provider, under
  the entry's own `provider_options`, and of `platoon sandbox reap` for a run whose record names none. The profile
  cannot weaken isolation: it names only a provider whose class says it isolates its sandboxes (`docker`, `singularity`,
  `enroot`; never `process`) and only the options that keep that isolation (`allow_network_none`, `rootfs_layout`,
  `scratch_root`, `image_cache`, `stage_images`, `gpu`), and `platoon train`, `eval` and `rollout` refuse a Harbor entry
  on an unisolated provider that the run config does not name itself. `~` and `$VAR` expand in it, and a key it does not
  take, or a `PLATOON_SITE` that names nothing, is a configuration error. `platoon train`, `eval`, `rollout` and
  `prepare` print the profile they used and what it gave the config, and refuse a provider from it that the node cannot
  run, quoting it. `config.resolved.yaml` holds what it gave and names each Harbor entry's provider, so a run directory
  loads the same without it, and `lock.json` records the profile's path, SHA-256 and values (`site`) and each Harbor
  entry's provider and options with where each came from (`sandbox_providers`). A training config without an `eval`
  section does not evaluate. platoon's temporary files (rollout workers' files, gateway sockets, sandbox and harness
  state, artifacts and staged toolchains) go to node-local scratch, the first of `$PLATOON_NODE_SCRATCH`,
  `$PBS_LOCALDIR`, `$SLURM_TMPDIR` and `$TMPDIR` that is set, else `/tmp`, rather than `$TMPDIR` alone.
- **Paths relative to the file that names them.** A config file's input paths are relative to that file, as
  `extends:` is: an environment's harness `.toml` spec and the options its service reads files from (a Harbor entry's
  `dataset`, `dataset_dir`, `task_dirs`, `splits`, `lock`, `tasks`, `overlays` and `mounts[].source`, a task table's
  `path`), so a config runs the same from any directory. A split manifest named relative to a `dataset_dir` that the
  same file names stays relative to it. An override's paths and the outputs (`log_path`, `inference.output_dir`, where a
  leading `~` names the home directory) stay relative to the working directory, and `config.resolved.yaml` holds them
  absolute. **Breaking:** these input paths resolved against the working directory before, and nothing falls back to it:
  a relative input that names nothing beside its file stops the command with exit 78, and names the path relative to the
  working directory too when that one exists ([Write and override
  configs](docs/how-to/configs.md#2-build-on-it-with-extends)).
- **Deprecations that say what replaces them.** A config key that is going away keeps working for one minor release,
  with a `pt.PlatoonDeprecationWarning` (a `FutureWarning`, which Python shows) that names the key's path, its
  replacement and the release that removes it. The commands name each one once on stderr, `lock.json` lists them
  (`deprecations`), and `platoon config validate` lists them under each file and, with `--strict`, fails on them. The
  first is `rollout_config.propogate_root_success`, the misspelling of `propagate_root_success`, which 0.3.0 removes,
  and `platoon harbor` warns the same way of a fetch config file, which 0.3.0 removes too.
  0.3.0 also removes `train.rollout_group_key`, now `train.label_by`, and `workflow_config.straggler_policy_key`,
  whose straggler policies key on `label_by`'s labels now; a workflow config that still sets it keys its own on that
  column until then. It removes the `strategy` of `eval` and `checkpoint` as well: `every` alone counts training
  steps, while a config that still sets `strategy` keeps its meaning until then (`epoch` counts `every` in passes over
  the training tasks, `none` turns the event off, and an `every` left out is 1, as it was). And it removes
  `num_concurrent_rollout_workflow_workers` of `train` and `eval`, which counted rollout groups: until then it reads
  as that many groups' rollouts, so `max_concurrent_rollouts` is the old value times the section's
  `workflow_config.group_size`. It removes a Harbor entry's `cache`, `image_map` and `toolchain_root`, and the
  variables `PLATOON_HARBOR_CACHE` and `PLATOON_TOOLCHAINS`, which the site profile's `cache` replaces: until then they
  work as before, and a variable set beside a profile's `cache` must name the same directory.

### Install

```bash
pip install 'platoon[tinker,ors]'
```

- Python 3.12 or later, on Linux; CI tests 3.12, 3.13 and 3.14. On 3.14, platoon needs datasets 4.4 and dill 0.4
  or later, whose picklers can hash a dataset there (pip upgrades older ones), and importing the Tinker SDK warns that
  pydantic's V1 core is not compatible with 3.14: the SDK takes only V1's date parsers and typing helpers, with which
  it builds its responses as on 3.12.
- The `tinker` extra pins the Tinker SDK to 0.16.1 and takes tinker-cookbook 0.3: later tinker-cookbook releases
  change what a renderer's parse returns or need a later Tinker SDK, which a future release adopts.
- The `hpc` extra installs hpc-sandbox 0.3 from PyPI for the `singularity` and `enroot` sandboxes. It deletes a
  stopped sandbox's files in the background, and the next sweep on the node deletes what the end of a job cut short;
  where the sandboxes' scratch outlives jobs, start each job with hpc-sandbox's janitor
  ([Sandboxes left by stopped rollouts](docs/how-to/sandboxes-and-harbor.md#sandboxes-left-by-stopped-rollouts)).
- Number search, the first run's environment, ships in the package as the built-in named environment
  `platoon-number-search@1` (`platoon.envs.number_search`), which generates its tasks from a seed. The recipes and
  the runnable examples are in the repository, not in the package: clone it to follow the tutorials. TextCraft is
  the first-party extension `platoon-textcraft` (`extensions/textcraft`), a distribution with a version of its own
  (`pip install 'platoon-textcraft[platoon]'`), which also exports TextCraft-Synth as a Harbor dataset
  (`python -m platoon_textcraft.harbor export`), whose verifier replays the game's action log. `recipes/rao` carries
  the RAO paper's TextCraft-Synth arms over to Tinker: its single agent (`linear.yaml`), its recursive agent
  (`recursive.yaml`: 25 steps for every agent, at most 6 levels of delegation, and each subagent rewarded for
  crafting its own targets, the paper's Dense rewards) and its Sparse ablation (`ablation_sparse.yaml`: every
  subagent takes the root's success). A recursive agent that sizes its children's budgets from its own 200 steps is
  not one of the paper's arms, so the recipe has none.
- `platoon.__version__` and `platoon --version` give the installed version, and the package ships type information
  (`py.typed`).

### Documentation

[The documentation](docs/README.md) has a getting-started guide, tutorials whose examples a CPU test runs, how-to
guides, guides to extending platoon, a command-line, config and Python API reference generated from the code, an
architecture overview and the RFCs, among them the drafts on Harbor datasets, images, grading and multi-container
tasks (0010) and on the public surface and conformance (0017), and the accepted design for component configuration
(0016).
[CONTRIBUTING.md](CONTRIBUTING.md) covers development, Conventional Commits, the checks CI runs, among them the
snapshot of the public API and the import boundaries of core, the first-party extensions and the examples, and how
to cut a release.
