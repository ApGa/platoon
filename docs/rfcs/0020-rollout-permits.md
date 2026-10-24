# RFC 0020: Rollout permits that return as each rollout ends

| | |
|---|---|
| Status | Draft |
| Created | 2026-10-25; revised 2026-10-25 to match the implementation of P1 and P2 |
| Authors | Apurva Gandhi (@ApGa) |
| Builds on | [RFC 0019](0019-training-mixtures.md), sections 4 (the dispatcher: the horizon, rollout permits, exact resume), 5 (samplers) and 6 (admission and refill); [RFC 0017](0017-public-surface-and-conformance.md) (`platoon.experimental`); [RFC 0001](0001-native-multi-agent.md) and [RFC 0003](0003-multi-policy-training.md), which must stay compatible; the dispatcher (`platoon/train/tinker/dispatcher.py`), the trainer's rollout permits and group runner (`platoon/train/tinker/rl.py`) and the group workflow (`platoon/train/tinker/workflows/group_rollout_workflow.py`) |
| Discussion | The pull request that adds this RFC, which also records the decisions of 2026-10-25 (open questions 1 to 11) |
| Implementation | P1 and P2 in one pull request (open) |

## Summary

Today a training group takes its `G` rollout permits at once and gives them back only when its workflow returns, so a
rollout that has finished keeps its permit until its slowest sibling ends. On long agentic tasks that idles a quarter
to a third of the rollout budget while the trainer waits for data. This RFC gives each rollout's permit back the moment
that rollout ends, and starts the next group's rollouts as permits free. The group stays the unit of everything else:
drawing, sampler decisions, credit, admission, the staleness horizon, statistics and resume.

- **Permits per rollout.** Under `train.permit_release: rollout`, the dispatcher reserves a draw's permits when it
  starts the draw, and the pool grants them first come, first served: one at a time to a workflow that runs its
  rollouts on the lease, so that its members start as their permits are granted (rolling starts), and whole to any
  other workflow, as today (co-start).
- **The lease.** A workflow runs each rollout inside `async with rollout_lease().rollout(i):`, which waits for a permit
  and returns it once the rollout has ended; `rollout_lease()` and its type, `RolloutLease`, are experimental, in
  `platoon.experimental`. `GroupRolloutWorkflow` runs every member this way and says so with a new class flag,
  `runs_rollouts_on_lease`, so it and its credit overrides get the gain unchanged.
- **Straggler deadlines** shift per member by its own start delay, which is zero under co-start. The straggler
  settings and the startup gate are otherwise unchanged.
- **Visibility.** New metrics show the permits held and what kept new groups from starting, and a warning says when
  stale drops rise and names the fix.

`permit_release: group` stays the default until a GPU check and, for any workflow that runs each rollout index once
(section 2), keeps today's behavior event for event.

**What it buys**, by trace replays of 12 measured runs (appendix A). The effective rate is trained rollouts per hour
times the credit's efficiency, which does not change here. Ranges span the engines' measured response to more load.

| | 27B model, long agentic science tasks | 30B model, the same tasks | 4B model, short crafting tasks (10 runs) |
|---|---|---|---|
| What keeps new groups from starting today | the permits | the permits | the staleness horizon |
| Rollout slots idle today, waiting for a sibling | 32.9% | 25.5% | 7-26% |
| Wall time for the same steps | −15% to −21% | −20% to −22% | 0 |
| Effective rate | +12% to +20% | +24% to +27% | 0 |
| Rollouts dropped as stale | 0.2% → 5.2-5.6% | 0.2% → 1.4-1.5% | unchanged |
| With `max_staleness + 1`, the documented advice (replayed with co-start) | +15.5% to +23.5%, stale 1.5-1.9% | +24% to +27%, stale 0.2% | n/a |

**What it costs.** More groups are open at once (at most 66 instead of 33 for the 27B workload) and steps get shorter,
so trained data is older and more long groups cross the staleness bound. Those are the hard, mixed groups: at today's
bound, informative groups per hour rise only +4.5% to +11.3% for the 27B workload. One more version of
`max_staleness`, or later per-datum stale masks, takes this back, and the warning says when it is needed. A restart
also redoes more rollouts (about 9-10 minutes of fleet time instead of 7.7 for the 27B workload), and half as many
again finished rollouts wait in the trainer for their siblings. As built, P1 and P2 add about 880 lines net, about 520
of them code and the rest docstrings, comments and blank lines, plus tests and docs.

**What it does not fix.** Where the staleness horizon, not the permits, decides when groups start, as in all ten
crafting runs, nothing changes. There the levers are another version of `max_staleness`, per-datum stale masks and the
wait before a rollout's first model call (open question 10).

**Beyond groups.** The lease counts rollout permits, not group members. A one-rollout-per-task algorithm such as PPO
with a critic gets a permit that comes back before its post-rollout work, and a recursive agent tree or a swarm
episode is one rollout on one permit (section 8).

## Motivation

**Where the idle comes from.** RFC 0019's dispatcher starts a group whenever the horizon, its label's limit and `G`
free permits allow, so batching is continuous at the level of groups, and a group holds its permits until its slowest
member ends:

- A group takes its entry's group size of permits at once, first come, first served (`Dispatcher._start`, and
  `_GroupPermits` in `rl.py`).
- It returns all of them only when its workflow returns: `_run_training_group` calls the dispatcher's `release` callback
  in its `finally`. The workflow is handed only the row, so no workflow can give a permit back early.
- Inside the group, a finished member keeps its permit until the slowest member ends or the straggler policy cancels
  it (`GroupRolloutWorkflow._run_group_rollouts`). A group idles `G × T_group − Σ T_member` permit-seconds.

**How much it costs, measured** on three workloads, every rollout of 12 runs joined to its reward (appendix A.1):

| | 27B model, long agentic science tasks | 30B model, the same tasks | 4B model, short crafting tasks (10 runs) |
|---|---|---|---|
| Setup | 32 tasks × 8 rollouts a step, `max_staleness` 3, 256 permits; rollouts p50 408 s, p90 821 s | the same recipe; p50 321 s, p90 609 s | 16 tasks × 8, `max_staleness` 2, 384 permits (the whole horizon); p50 64-177 s |
| Permit time held by a finished rollout waiting for a sibling | **32.9%** | **25.5%** | 7-26% |
| Slowest rollout over the group's median, p50 / p90 | 1.51 / 2.36 | 1.30 / 1.69 | 1.04-1.27 / 1.18-1.78 |
| What kept new groups from starting | the permits, 99% of the time; 57 of 128 horizon slots used | the permits, 99%; 57 of 128 | the horizon, 84-97% |
| Trainer waiting for rollouts, share of each step | 70% | 71% | 0-81% |
| Inference engines | never queued; KV cache 34% full at the median | never queued; 8% | 3-9% |
| Do slow rollouts fail more? | weakly | yes: a mixed group's slowest rollout passes 0.59, its first 0.79 | yes: the slowest 0.37-0.58, the middle ranks 0.58-0.77 |

Four facts shape the design:

1. **Where permits bind, the idle is the throughput limit.** The long-rollout runs' engines had 2.9× and 11× KV
   headroom and the trainer starved for 70% of each step, yet a third and a quarter of the rollout slots sat idle.
2. **Where the horizon binds, permits are not the limit.** The horizon counts groups started and not yet trained,
   `(k + 1) × batch_size`, which is the staleness contract. In the crafting runs it was full, so freed permits could
   not have started anything.
3. **Cutting tails is a biased way to get the time back.** Slow rollouts fail more often, so a rule that cancels them
   keeps a sample that leans optimistic. Today's straggler policy (quorum 6, 600 s) is nearly inert only because a
   1200 s agent timeout bounds the tails first: it cut 0.38% and 0.12% of the rollouts.
4. **The workflow is an extension point that may stop being a group.** The fix must not assume that a draw is `G`
   rollouts of one task that start and end together.

## Proposal

### 1. Permits per rollout

Under `train.permit_release: rollout`, the rollout permits (the budget `max_concurrent_rollouts`) are reserved per
draw and granted per rollout:

- **Reservation.** When the dispatcher starts a draw, it reserves `size` permits for it, its entry's group size as
  today. The draw counts toward the horizon and its label's limit as one group from that moment, as today.
- **Grants.** The pool grants permits to reservations in the order they were made, as it grants whole groups today,
  mixed group sizes and a large group waiting first included. A draw whose workflow runs its rollouts on the lease
  (sections 2 and 3) gets its permits one at a time as they free: *rolling starts*. Any other draw gets its whole
  reservation once `size` permits are free: *co-start*, today's rule.
- **Starts.** The dispatcher starts a new draw only when the pool can grant it something, which with these grants
  means that no older draw still waits for permits. So at most one draw per pool is partly granted at a time, and a
  group's members start over the time the next `size − 1` permits take to free.
- **Returns.** A rollout returns its permit when it ends. When the workflow returns, by success, failure or
  cancellation, every permit the draw still holds returns and the ungranted rest of its reservation is withdrawn. A
  rollout cancelled while it waits for a grant neither leaks a permit nor counts one twice.

```python
# platoon/train/tinker/permits.py (sketch): replaces today's _GroupPermits in rl.py
class PermitPool:  # train.max_concurrent_rollouts
    # rolling: granted one permit at a time; active (permit_release: rollout): its blocks take and return permits
    def reserve(self, count: int, *, rolling: bool, active: bool) -> "DrawLease": ...


class DrawLease:  # one object: the draw's reservation, its workflow's lease (section 2) and the dispatcher's release
    async def ready(self) -> None: ...  # its first permit (rolling) or all of them granted
    def resize(self, size: int) -> None: ...  # before the draw starts: an adaptive decision's group size
    def withdraw(self) -> None: ...  # every permit it holds back, the ungranted remainder withdrawn
    def __call__(self) -> None: ...  # the workflow returned: withdraw, and free the draw's label slot
```

**Adaptive decisions.** An adaptive sampler still decides only when its draw can start (RFC 0019, section 5): under
co-start with all of the draw's permits held, as today, and under rolling starts once its first permit is granted,
after which the reservation takes the drawn group's size. Either way the decision comes just before the group's first
rollout starts.

Under `permit_release: group` nothing changes: a draw takes its whole reservation before it starts and returns it when
its workflow returns.

**Why rolling starts are safe.** With `P` permits busy and rollouts that last `T` on average, a permit frees every
`T / P` seconds, so a group's members start over about `(G − 1) × T / P`. In the measured runs (`G` = 8, `P` = 256)
that is 12 s for the 27B workload (`T` = 451 s) and 10 s for the 30B one (366 s), against a straggler grace of 600 s.
Skew inside a group matters in two places. The straggler clock would give a late starter less time, and the deadline
offset of section 3 removes that exactly: every member gets the same time from its own start. And members could sample
different policy versions, but a publish falls inside a 10-12 s window for about 2-3% of groups at the measured step
times (EST), and the stale filter already judges a group by its oldest datum. The spread grows as the budget shrinks
toward one group (about `T` at `P = G`); the offset keeps the grace fair at any spread.

**What co-start costs.** A draw that waits for all its permits is granted none of them until the last one frees, and
the ones already free sit idle meanwhile, since no later draw may take them: on average `(G − 1) / 2` permits, a share
`(G − 1) / (2P)` of the budget, 7/512 = 1.4% at `G` = 8 and `P` = 256. The trace replays agree: the permits are 98.6%
busy with co-start and 99.7-100% with rolling starts, which add 0.3 to 1.2 points of effective rate. Co-start stays
only where it is needed for safety: a workflow that does not run its rollouts on the lease may start them all at once,
and only a whole reservation keeps it within the budget.

### 2. The lease

A workflow reaches its draw's permits through a lease. A new module, `platoon/train/leases.py`, holds the type and a
context variable; it imports no `tinker`, so that `platoon.experimental` can export both names:

```python
# platoon/train/leases.py (sketch)
class RolloutLease(Protocol):
    @property
    def size(self) -> int: ...  # rollout permits the draw reserved: its entry's group size today; 0 outside a draw

    # run rollout `index` on a permit; its value: the seconds it waited for its permit, 0.0 when it took one at once
    def rollout(self, index: int) -> AbstractAsyncContextManager[float]: ...


def rollout_lease() -> RolloutLease:
    """The lease of the training draw running in this asyncio context; elsewhere (evaluation, tests, a direct call)
    the no-op lease, of size 0, whose blocks hold nothing."""
```

- **`async with lease.rollout(i) as waited:`** waits until the draw holds a granted permit that none of its rollouts
  has used (at once under co-start), runs the block on it, and returns the permit to the pool when the block exits.
  `waited` is the seconds the block waited for its permit: exactly 0.0 when it took one at once, as it always does
  outside a training draw, under `group` and under co-start.
- **The block covers the rollout's whole life.** The permit returns only after the rollout's worker and sandboxes are
  gone, and a cancelled rollout returns its permit only after its cancellation has been awaited. A rollout's retry
  stays inside its block.
- **Each index once.** In a training draw, under either setting, a block for an index outside `0 .. size − 1` raises
  `IndexError`, and a second block for an index, while the first runs or after it ended, raises `RuntimeError` as it
  enters, so that a misused index fails its group rather than run a rollout on no permit; a block cancelled while it
  waits gives its index back. A `GroupRolloutWorkflow` subclass that retries its whole group in `arun_episode`
  therefore now fails that group, under `group` too. A draw is never granted more than `size` permits.
- **Blocks after the release.** A block that enters after its draw is released, or still waits for a permit then,
  holds nothing, and the trainer warns once for that draw.
- **Under `permit_release: group` a draw's blocks hold nothing:** they enter at once and return nothing, and check each
  index as under `rollout`; the permits return whole when the workflow returns. That reproduces today's event order
  exactly, so `group` is both the rollback path and the A/B arm.
- **The runner sets it.** `_run_training_group` sets the context variable around the workflow call. The workflow's task
  and its members' tasks are created with `asyncio.ensure_future`, which copies the context, so the lease reaches
  every member of the group and no other group, the way the workflow already gets its entry's settings. Evaluation sets
  none and gets the no-op lease, whose blocks hold nothing whatever their index.
- **The dispatcher's runner keeps its shape.** The draw's `DrawLease` is still callable as the `release` callback it
  replaces (`GroupRunner` in `dispatcher.py`): calling it releases the draw and frees its label slot. Evaluation
  reserves whole groups from a pool of its own with `reserve`, `ready` and `withdraw`, and today's `_GroupPermits`,
  with its `acquire` and `release`, is gone.

### 3. `GroupRolloutWorkflow` on the lease

`GroupRolloutWorkflow` runs every member inside its block, in the one method every member passes through, and says so
with a new class flag beside `applies_entry_settings` and `gives_verdicts`:

```python
# platoon/train/tinker/workflows/group_rollout_workflow.py (sketch)
class GroupRolloutWorkflow:
    runs_rollouts_on_lease: ClassVar[bool] = True  # new: every member runs inside lease.rollout(i)

    async def _run_group_member(self, data, rollout_number, admission):
        ...  # today's: the member's readiness callback for the startup gate
        async with rollout_lease().rollout(rollout_number) as waited:  # new: waits for a permit, returns it at the end
            admission.starts.started(rollout_number, waited)  # new: its start delay (below)
            with holding(None):  # new: the rollout sees no lease, so a group it runs takes no permits (section 8)
                return await self.arun_episode_single(data, rollout_number)
```

Why this is the right point:

- **Process isolation.** `supervisor.run` returns only after the worker's process tree is reaped and its orphaned
  sessions and sandboxes are deleted, in a shielded `finally`. The permit stands for exactly those resources.
- **In-process isolation.** The rollout coroutine, its subagents and their grading have ended. A sandbox or env session
  teardown that platoon finishes in the background (`let_finish`) can still outlive the block: one a cancellation
  interrupts, as it outlives a group's release today, and one a rollout leaves running as it ends, which `rollout`
  lets overlap the next rollout. With every teardown deferred, a test with a budget of 4 had 11 sandboxes alive at once
  under `rollout` and 6 under `group`. Waiting for those cleanups inside the block, bounded and shielded, is a
  prerequisite of P4.
- **Every exit passes through it.** A straggler the policy cancels leaves its block after its cancellation is awaited,
  and so do a launch failure and any error.
- **Overrides keep the gain.** `arun_episode_single`, `compute_baselines`, `assign_advantages`, `_run_group` and
  `_make_task_output` overrides all still run members through `_run_group_member`.

**Subclasses, conservatively.** A subclass that overrides `arun_episode`, `_run_group_rollouts` or `_run_group_member`
may start members some other way, or run one outside its block, so the trainer treats it as not running on the lease
unless it sets `runs_rollouts_on_lease` in its own class body. The flag is read along the method resolution order: a
flag set in a class body covers that class's overrides and its bases', so an override that a mixin or another branch
of a diamond brings reads as not running on the lease too. Such a subclass gets co-start, and its permits return per
rollout only where its members still run through `GroupRolloutWorkflow`'s `_run_group_member` (an override that calls
`super()`). A subclass that overrides only credit or the member (`compute_baselines`, `assign_advantages`,
`_run_group`, `_make_task_output`, `arun_episode_single`) keeps rolling starts.

**Straggler deadlines.** Today, once `straggler_quorum` members have settled, the rest get `straggler_timeout_seconds`
more, on one clock the group shares. Under rolling starts, member `i`'s deadline is that shared deadline plus
`start_i − start_first`, its own start delay within the group, where a member starts when its block takes its permit.
`GroupRolloutWorkflow` enters every member's block at once, so each block's value, the seconds it waited, gives that
start; a workflow that enters its blocks at different times offsets by start times it records itself. So every member
gets the same time from its own start, and a late start never shortens its grace. A member that has not started when
the quorum settles gets its offset when it starts, and one that never enters a block (an override of
`_run_group_member` that does not call `super()`) counts as starting with the group. Under co-start every offset is
exactly 0, and the policy runs today's code.

**The startup gate** (process isolation only) stays one permit per group, taken before the first member starts and
released when every member is up or exited. Per-member results, verdicts, credit, statistics records, admission at
settle, kept groups and the stale filter are unchanged.

### 4. The contract for custom workflows

Nobody has to implement anything new. The trainer still calls only `arun_episode(row)`, plus `publish_model_binding`
when the class has it, and the output contract is unchanged. A custom workflow that never touches the lease behaves
exactly as today: its draw is granted whole, and its permits come back when it returns. One that runs its own rollouts
can return each permit as its rollout ends, and with the flag it gets rolling starts:

```python
import asyncio

from platoon.experimental import rollout_lease


class MyWorkflow:  # a custom workflow that runs its own rollouts
    runs_rollouts_on_lease = True  # every rollout below runs inside lease.rollout(i)

    async def arun_episode(self, row):
        lease = rollout_lease()  # read it in the workflow's own task: a no-op lease outside training

        async def one(i: int):
            async with lease.rollout(i):  # waits for a permit; returns it once rollout i and its sandboxes are gone
                return await self.run_rollout(row, i)

        results = await asyncio.gather(*(one(i) for i in range(self.group_size)))  # lease.size in training
        return self.assign_credit(row, results)
```

The rules:

1. **What the permits are.** A draw reserves `lease.size` permits. They are rollout permits, not group members; today
   the count is the entry's `group_size`.
2. **One block per rollout.** Run rollout `i` (`0 ≤ i < size`) inside `lease.rollout(i)`, once, and let the block exit
   only once the rollout has ended and its worker and sandboxes are gone; await a cancelled rollout inside its block,
   and retry a rollout inside its block too. A misused index fails the group (section 2).
3. **The flag.** Set `runs_rollouts_on_lease = True` only when every rollout runs inside a block; the draw then gets
   rolling starts. Without it the draw is granted whole before `arun_episode` starts, and each block still returns its
   permit when its rollout ends.
4. **The rest** of what the draw holds comes back when `arun_episode` returns, raises or is cancelled, as today; a
   block that enters after that, or still waits for its permit then, holds nothing, and the trainer warns once for
   that draw.
5. **Threads.** Context variables do not follow `run_in_executor` unless the context is copied: read the lease in the
   task first.
6. **Reserved.** Asking for permits beyond `size`, for draws whose size varies or for an episode's child agents
   (section 8).

| Who | What changes under `permit_release: rollout` | What they do |
|---|---|---|
| Users who only write configs | permit-bound runs get faster; `max_concurrent_rollouts` now means rollouts running; stale drops can rise, and the warning says so | nothing; if warned, `max_staleness + 1` |
| `GroupRolloutWorkflow` subclasses that override credit or `arun_episode_single` | rolling starts and permits per rollout, automatically | nothing |
| Subclasses that override `_run_group_rollouts`, `_run_group_member` or `arun_episode`, themselves or through a mixin | co-start; permits per rollout where members still run through `GroupRolloutWorkflow`'s `_run_group_member` | optional: run every rollout inside `lease.rollout(i)` and set the flag in the class body |
| Custom workflows with their own `arun_episode(row)` | today's behavior | optional: the same |
| Evaluation, checkpoints, dispatcher state, kept groups, `tasks/` statistics | unchanged; dispatcher formats 1 and 2 resume | nothing |

### 5. What stays group-shaped

| Stage | Under `permit_release: rollout` |
|---|---|
| Drawing, sampler decisions (M4), horizon slot, label slot | unchanged: one per draw, from the moment the draw is reserved |
| Starting a group | when the pool grants its first permit (rolling starts) or all of them (co-start), first come, first served |
| The members of a group | start as their permits are granted, under the startup gate; together under co-start |
| A member ends | **its permit returns** |
| Straggler policy | unchanged, but each member's deadline is offset by its own start delay (zero under co-start) |
| Credit, verdict, statistics record, admission at settle, stale filter, training, kept groups, dispatcher state | unchanged |

**The horizon keeps counting a partly finished group as a whole group.** The horizon bounds groups started and not yet
trained. A partly finished group has trained nothing, and its finished members' data keeps ageing while it waits.
Counting it as less than one slot, or counting rollouts instead of groups, would let more than `(k + 1) × batch_size`
groups be open: a silent increase of `max_staleness` that grows with the weight of the tails. That trade is sometimes
worth making, but it belongs to the user, as one config key. Where permits bind, the horizon does not stop freed
permits: the long-rollout runs used 57 of 128 slots, and 70-74 after the change.

**Label limits keep counting groups.** A running group never has more than `G` rollouts, so a label's rollouts in
flight stay within `limit × G`, as today.

### 6. Staleness: the side effect and its two answers

**Why staleness rises.** Each group takes about as long as before (its rollouts run 4-13% slower at the higher engine
load), but more groups are open at once and steps are shorter, so versions advance faster while a long group runs.
Trained data is older (a mean age of 1.32 → 1.77-1.80 versions for the 27B workload, 1.28 → 1.63-1.64 for the 30B
one), and more long groups cross the bound: stale drops go from 0.2% of rollouts to 5.2-5.6% for the heavy-tailed 27B
workload and to 1.4-1.5% for the 30B one. The groups that go stale are the long, hard, mixed ones (today's stale groups
in the 30B run had a mean reward of 0.53 against 0.93, and 73% mixed outcomes against 19%), so for the 27B workload
informative groups per hour rise only +4.5% to +11.3% while trained rollouts rise +12% to +20%.

A user who flips the switch and reads nothing would lose most of the informative-group gain. So the side effect is
made visible, and it has two explicit answers:

- **A warning, and `max_staleness + 1` as documented advice** (P2). When more than 2% of the groups the last 10 steps
  consumed went stale, the trainer warns, once per start of training and under either setting, from the tenth step on
  (it needs a full window of 10 steps), names what held the dispatcher back most and gives the fixes. If the horizon
  has room (`blocked_share/horizon` near 0), `max_staleness + 1` changes only the stale filter; if the horizon binds, it
  also starts more work and ages data further. In the replays of the long-rollout runs it gives the best effective
  rate of any change at 256 permits, +15.5% to +23.5% (27B) and +24% to +27% (30B), with stale drops of 1.5-1.9% and
  0.2%. Data is about 0.1 versions older on average, 1-4% of trained data is four versions old, recovery pruning keeps
  one more checkpoint, and training stays corrected per token against the sampling log-probability (`cispo`, the
  default loss).
- **Per-datum stale masks** (P5, later and opt-in). A rollout's datums all carry one version, the oldest it sampled,
  and the stale filter drops a whole group by its oldest datum, so one early call of one long member drops eight
  members' work. P5 stamps each datum (a run of prefix-merged calls) with the oldest version among its own calls and,
  under `train.stale_filter: datum`, drops only a group's over-stale datums. The group's credit, computed over all its
  members, does not change, so each kept term keeps its expectation. In the replays the masks remove 90-94% of the
  long-rollout runs' stale drops at today's bound (+13.6% to +21.8% effective rate for the 27B workload) and nearly
  all of the crafting runs', and they are the only measure that shrinks the stale filter's selection against long
  rollouts.

The bound never moves by itself, and there is no cap on groups in flight (open question 2).

**Bias.** Per-rollout permits censor nobody: every member still runs to its end or to the straggler policy, credit
runs over the whole group, and the deadline offset keeps the order of starts out of who is cut. The indirect effects
are older data, which `cispo` corrects per token, and a larger stale-drop selection, which the advice and the masks
bring back to about today's level (appendix A.3).

### 7. Configuration, metrics and warnings

| Key | Values | Default | Meaning |
|---|---|---|---|
| `train.permit_release` | `group` \| `rollout` | `group`; `rollout` from P4, after the GPU check (open question 1) | `group`: today's behavior and event order; the lease's blocks hold nothing (section 2). `rollout`: each rollout's permit returns when its block exits; rolling starts for workflows that run their rollouts on the lease, co-start for the others |
| `train.stale_filter` (P5) | `group` \| `datum` | `group` (open question 3) | section 6 |
| `train.max_concurrent_rollouts` | unchanged | unchanged | under `rollout`, the rollouts that run at once, which the permits then make literally true |

**New metrics and warnings** (P2), which show directly what the measurements behind this RFC needed a log parser for:

- **`rollout_client/permits_held_mean`**: the time-weighted permits granted to draws. Held minus
  `rollouts_in_flight_mean` is the idle: today the straggler idle; under `rollout` about 0, with rolling starts or
  co-start: a draw that waits for its whole reservation holds none of it, so co-start's cost, `(G − 1) / (2P)` of the
  budget, shows in `concurrency_achieved_fraction`.
- **`train/dispatch/blocked_share/{permits,horizon,labels,plan,sampler}`**: the share of each step in which the
  dispatcher could not start a draw, by reason. The dispatcher knows the reason: it is either awaiting permits, or
  waiting for a change after finding nothing that may start, or after an adaptive task sampler found nothing eligible
  (`sampler`). Permits binding means raise `max_concurrent_rollouts`; the horizon binding means only `max_staleness` or
  a faster trainer helps.
- **`rollout_client/startup_gate_waiting_mean`**: the groups waiting for a worker-startup slot, time-weighted (process
  isolation; logged once a group has reached the gate).
- **The staleness warning** of section 6, once per start of training, from its tenth step on. Until P5 lands it names
  the masks as planned, so that its text stays true.
- **An invariant check.** If `rollout_client/rollouts_in_flight_max` exceeds `max_concurrent_rollouts`, a workflow
  returned a permit while its rollout still ran, and the trainer warns once per start. It counts
  `GroupRolloutWorkflow`'s members only (and an evaluation's `GroupRolloutWorkflow` left on the default `stats_scope`,
  `train`), so it cannot see a custom workflow's rollouts, which keep rule 2 of section 4 themselves.
- **Censoring metrics**, `train/batch/censoring/{censored_share,mixed_groups,slowest_pass_rate,others_pass_rate}` and
  each label's under `train/batch/<column>/<label>/censoring/`, from the `MemberResult` fields every group record
  already carries, over the groups a step dequeued that a straggler policy applies to (a run without one logs none):
  the share of members the straggler policy censored, and, over mixed groups, the pass rate of each group's slowest
  valid member against the rest's. Anyone who turns a cut on sees what it selects.
- **`concurrency_achieved_fraction`** reads as the permits' busy share, close to 1 in permit-bound runs, and
  `groups_in_flight` can exceed `max_concurrent_rollouts / G`.

**Straggler settings stay** as they are, off by default (open question 8). With group-held permits a cut was the only
way to free a finished member's idle permit; with per-rollout permits waiting idles nothing, so a cut buys only the
straggler's own remaining time and slightly fresher data, at the price of censoring by duration. The docs will call
these settings a deadline against runaway members, not a throughput lever. They will also say what
`max_concurrent_rollouts` means under `rollout` and give the staleness advice (`docs/how-to/train-and-eval.md`), and a
guide in `docs/extending/` will describe the lease.

### 8. Algorithms without groups

**PPO with a critic**, one rollout per task, works with the dispatcher today and loses nothing here. `group_size: 1`
gives one permit per draw, the horizon and the batch count episodes, and a one-rollout draw has no idle inside a group
(the replays keep 100% of its permits busy), so rolling starts and co-start coincide. On `GroupRolloutWorkflow`, the
member's block returns the permit when the rollout ends, before credit runs, so value calls and GAE would hold no
rollout slot. What PPO still lacks is not scheduling: a critic Tinker can train and serve (a generative critic on a
second LoRA is UNVERIFIED), a second learner with its own optimizer and checkpoints (RFC 0003), an asynchronous credit
hook, workflow state that survives a resume and updates in draw order, outputs a checkpoint can keep, and a sampler
score for group size 1, which `pass_rate` refuses.

**Recursive agents and swarms** (RFC 0001; recursive agents as in [RAO](https://arxiv.org/abs/2605.06639)):

- **One episode, one permit.** A recursive tree (its root, every child and the children's grading) or a swarm episode
  is one rollout on one permit. Under rolling starts its permit is granted when the episode starts and returns when the
  whole episode has ended. Its agents' concurrency stays under the episode's own settings, `agents.max_depth`,
  `agents.budget`, `agents.child_max_steps` and `agents.child_timeout_s`, and the rollout's time limit. A group that a
  `GroupRolloutWorkflow` member's rollout runs in turn is part of the episode too: the member runs its rollout with the
  no-op lease (section 3), so the nested group's members take no permits of their own.
- **Credit is unchanged.** It still runs once per group after every tree has ended, RAO's baseline over the group's
  completed roots and its depth weighting included.
- **Permits count episodes, not live agents.** A tree with many live children holds one permit, so
  `max_concurrent_rollouts` does not bound the model streams of a recursive run. That is true today too, and this
  design does not change it.
- **The way forward, prepared here.** The pool's incremental, first-come, first-served grants are the mechanism an
  episode would use to take permits for its agents. Statically, an episode reserves `weight` permits up front: the
  unit plan's `weight`, below. Dynamically, children request permits as they spawn. That form needs a lending rule: a
  parent blocked on a child lends its permit to that child, or else every permit can end up held by parents waiting on
  children that wait for permits, a deadlock. That is why asking for permits beyond `size` stays reserved, to land
  with RFC 0001's scheduling of child agents.
- **Do not overload `group_size`** to make a swarm's lease a pool of slots: it is also the credit group, the statistics
  record's size and `pass_rate`'s `k`.

**Several policies** (RFC 0003): one permit pool per policy, with one lease per pool held by the draw. Credit once,
then routing, and drops atomic per policy are unaffected.

**The later contract: designed, not built** (open question 6). When the first workflow without groups is built (PPO
with a critic together with RFC 0003, or a swarm), the lease grows into a unit plan that each environment entry
declares before any row is drawn, beside workflow state saved in draw order (`observe`, `state_dict`) and custom
outputs a checkpoint can keep:

```python
# Later, with its first consumer; not part of this proposal
@dataclass(frozen=True)
class UnitPlan:
    rollouts: int = 1  # G for GRPO, 1 for PPO with a critic
    weight: int = 1  # permits one rollout holds: a tree whose children have their own sandboxes weighs more
    co_start: int = 1  # permits granted before the unit starts; the rest as they free
    max_rollouts: int | None = None  # replacements and later waves included (None: rollouts)
```

Nothing in the lease has to be undone to get there: `lease.size` becomes the plan's `rollouts × weight`, and the
reserved request for more permits serves the rollouts beyond the plan.

### 9. How the guarantees generalize

RFC 0019's guarantees (M3 to M5) hold for groups under this proposal and for one rollout per draw; the later unit plan
keeps each of them per unit.

| Guarantee | GRPO groups | One rollout per draw (PPO with a critic) |
|---|---|---|
| **Exactly once**: every position is consumed once and gets one fate | unchanged: one draw, one group, one fate. The lease touches only the permits, never outstanding draws, settlement or positions | unchanged: a draw is one rollout |
| **A restart loses work but never counts it twice** | a group running at a checkpoint, partly granted or not, is an outstanding draw and is reissued whole, finished members included | a running rollout is reissued; settled ones are kept |
| **One consistent save point; dispatcher formats 1 and 2 resume** | unchanged: reservations and leases are not state | unchanged |
| **The staleness horizon**: at most `(k + 1) × batch_size` started and untrained | unchanged, in groups; a partly finished group counts whole | in draws, which are rollouts: set `batch_size` to `B × G` for the same rollouts a step |
| **The stale filter** | per group, by its oldest datum; with P5, per datum with the group's credit unchanged | per rollout; with P5, per datum |
| **Start skew inside a credit group** | rolling starts spread a group over `(G − 1) × T / P` (10-12 s in the measured runs), and each member's deadline is offset by its start delay; co-start for workflows not on the lease | none |
| **A `k = 0` run replays bit for bit** | holds: with the default budget (`B × G`) permits never bind, so every draw is granted whole at once and the run trains the same datums in the same order | holds |
| **Samplers (M4)**: one decision per draw, made when its group can start | unchanged; under rolling starts the decision is made once the draw's first permit is granted. With more groups open, `pass_rate` decides with more of its recent draws outstanding: slightly staler evidence, not a correctness change | unchanged; `pass_rate` refuses group size 1 |
| **Admission and refill (M5)**: decided once per draw at settle, before the stale filter | unchanged; a rejected group's permits are already back | `informative` is moot at group size 1; use `all` |
| **Statistics**: one record per settled draw; censored members neither pass nor fail | unchanged | one member per record |
| **Label limits** | count groups; a label's rollouts stay within `limit × G` | count rollouts |

## The hard parts

1. **Returning a permit at the right moment.** A permit stands for a worker and its sandboxes, so it may come back only
   after they are gone. The `async with` block makes that structural: `GroupRolloutWorkflow` gets it from the
   supervisor's shielded cleanup under process isolation, while in process a teardown left to the background can
   outlive the block until P4's prerequisite lands (section 3). A custom workflow gets the contract's rules: the run
   checks its indices (an error) and blocks that outlive its draw (a warning) for any workflow, but whether a rollout
   ran outside its block only for `GroupRolloutWorkflow`'s members, the only rollouts the invariant check counts. One
   edge, at shutdown only: a second cancellation during the shielded cleanup can return the permit while the cleanup
   still runs, which is transient and harmless.
2. **Cancellation while a rollout waits.** A member can be cancelled while it waits for its grant, by a stop, an
   interrupt or its group's cancellation. Its block must give up its place without leaking a permit granted at that
   instant or counting one twice, as `_GroupPermits.acquire` already does for whole groups, and the draw's ungranted
   remainder must be withdrawn when the workflow returns, so that the draw behind it is served. P1's tests cover each
   path.
3. **Waiting moves rather than disappears.** A group still waits for its slowest member, but it no longer holds idle
   permits while it waits. Waiting now costs staleness and memory, which the warning makes visible and
   `max_staleness + 1` or the masks absorb. And horizon-bound runs get nothing, by design: making freed permits start
   work there needs a larger `max_staleness` or a member that trains before its group completes, a change of algorithm
   whose two measured forms both lose (see Alternatives).
4. **The startup gate under rolling starts** is now held for a group's start spread as well. With the default fork
   server, a worker is up in 0.2 s at the median, and since at most one group is partly granted, the gate's two slots
   (`worker_launch_concurrent_groups`) suffice. With `worker_start_method: exec` (17-24 s a worker), a group still
   starting up can hold the second slot when the next group needs it, and the gate binds, which
   `rollout_client/startup_gate_waiting_mean` shows, beside the gap between permits held and rollouts running. The
   replays do not model the gate.
5. **The budget becomes the knob for the next lever,** since it then means rollouts running, and the engines had 2.9×
   and 11× KV headroom. But the 27B projection extrapolates past the load its run reached, so recipes change only after
   the GPU sweep, guided by `blocked_share/permits` and the engines' KV use (open question 4).

## Alternatives

- **Only raise `max_concurrent_rollouts`, with no code** (368-376 for the 27B workload, 336 for the 30B one). It ties on
  throughput (+14% to +23% and +22% to +25% effective rate) and drops fewer stale groups for the 27B workload
  (2.5-3.0%), because fewer groups are open. But it works by letting up to 31-47% more rollouts than the budget run at
  peaks, and it needs a factor measured per run. A good stopgap; not a substitute, since the budget would stop meaning
  what it says, and per-rollout permits are what make it a clean knob for the larger lever.
- **Co-start for every workflow,** the first version of this design. It costs `(G − 1) / (2P)` of the budget for
  nothing once deadlines are offset, so it stays only where it is needed for safety (open question 11).
- **A public unit contract now.** The right eventual shape (section 8), but about 1,800 lines with no measured gain,
  since the lease delivers all of it, and no consumer yet: PPO is blocked on RFC 0003 and a critic anyway, and an
  experimental API designed before its first user is likely to be designed wrong.
- **Two-phase group credit**: the first members to finish settle as one part with leave-one-out credit among
  themselves, each later member alone against that part's mean. It matches per-rollout throughput and keeps data
  fresher, but its later members get a weaker baseline, which leaves it 1.6-2.6 points of effective rate below
  per-rollout permits with `max_staleness + 1` on the long-rollout runs. Under group admission it fragments the batch
  and loses 11-28%, its fractional step cap can deadlock, and it costs 1,200-1,800 lines in the dispatcher's
  resume-critical core.
- **Group-free credit as a speed lever** (a per-task baseline frozen at dispatch). Untreated it is slower, by 15-17%
  (27B) and 34% (30B) of wall time, because no advantage is exactly zero any more and the trainer does 1.8-4.6× the
  work. It stays the enabler for PPO and swarms, not a way to speed up GRPO.
- **Count a partly finished group as less in the horizon, or count rollouts**: a silent, tail-dependent increase of
  `max_staleness` (section 5). Raise `max_staleness` instead, explicitly.
- **Let the dispatcher own members** ("schedule rollouts, assemble groups", with callbacks into the workflow). It would
  force every workflow into a state machine and split the straggler policy and credit across callbacks; a coroutine
  with a lease keeps today's workflow code.
- **Straggler cuts, hedges ("start 8, keep the first 6"), deadline cuts and cancelling doomed groups.** Each loses
  informative groups or saves under 1%, and each censors slow members, which fail more often (appendix A.4).
- **A cap on groups in flight** (for example 48): dominated by `max_staleness + 1` on both effective rate and stale
  drops.
- **Over-provision and defer** (APRIL, Kimi): these work around a step barrier, which platoon does not have, and
  pausing a straggler frees nothing scarce when the engines have headroom.
- **Doing nothing.** A quarter to a third of the rollout budget stays idle on long-rollout runs while the trainer
  starves.

## Rollout plan

| Milestone | Scope | Exit check | Pull requests |
|---|---|---|---|
| P1 permits per rollout | sections 1 to 5: reservations and first-come, first-served grants with rolling starts and co-start; `RolloutLease` and `rollout_lease()` in `platoon.experimental`; `runs_rollouts_on_lease`; `GroupRolloutWorkflow` on the lease, with straggler deadline offsets; `train.permit_release`, default `group`; docs. With P2, in one pull request | CPU tests with fakes. **Permits:** under co-start one returned permit starts nothing and `size` of them start the waiting draw; under rolling starts a draw starts on one free permit and its members start as permits free; first come, first served with trickling permits and mixed group sizes; at most one partly granted draw; the budget is never exceeded (checked on every grant). **The lease:** each index once, bounded at `size` (errors under either setting), holding nothing after release (a warning per draw); failure and cancellation return everything, a member cancelled while it waits for a grant included; a stop while a draw waits behaves as today. **Replay:** a `k = 0` run trains the same datums in the same order as before under both modes' defaults, and `group` reproduces today's event order. **Stragglers:** with `P = 2G` and one straggling member, a third group starts before the straggler ends; a cancelled straggler returns its permit only after its cancellation is awaited; a late-starting member gets its full grace; a member that never enters a block keeps the group's deadline. **Compatibility:** a `_run_group_rollouts` override that bypasses `_run_group_member` keeps co-start and group-held permits, and an override from a mixin or another branch of a diamond reads as not on the lease; a custom workflow without the flag that uses `lease.rollout(i)` returns permits per rollout under co-start; evaluation gets a no-op lease; RFC 0019's resume, sampler and admission tests pass unchanged | |
| P2 visibility | section 7: `permits_held_mean`, `blocked_share/*`, `startup_gate_waiting_mean`, the staleness warning, the in-flight invariant check, the censoring metrics; docs | the metrics match a hand-computed fake run, the censoring metrics only where a straggler policy applies; the warning fires once on a fake run with 3% stale groups while the horizon has room, names both fixes, and stays silent below 2% and before the tenth step; the invariant warning fires on a `GroupRolloutWorkflow` subclass that says it runs on the lease but runs its members outside their blocks; the blocked shares agree with the dispatcher's own waits | |
| P3 GPU check | the 30B workload's recipe for 30-50 steps, `rollout` against `group`; one crafting run; then a short budget sweep (256, 320, 384) with `max_staleness + 1` | `concurrency_achieved_fraction` ≥ 0.95; step time at least 15% shorter (the replays predict 20-22%); stale groups ≤ 2%, else apply `max_staleness + 1`; permits held back to 0 at the end; the crafting run unchanged within noise; the sweep shows where step time stops falling | |
| P4 the default | first, in process, a member's block waits, bounded and shielded, for the teardowns its rollout left to finish in the background (section 3); then `permit_release: rollout` by default, `group` kept as the rollback; the recipes' `max_concurrent_rollouts` per the sweep (open questions 1 and 4) | with every in-process teardown deferred, no more sandboxes are alive at once than the budget; release notes say that `max_concurrent_rollouts` now means rollouts running and that staleness can rise, naming the warning | |
| P5 per-datum stale masks | section 6: per-call versions, per-datum stamps, `train.stale_filter: group \| datum` (default `group`), `batch/stale_datums` | with fakes, a rollout that spans versions keeps exactly its in-bound datums; `stale_filter: group` trains byte for byte as today; the fates and statistics of whole groups are unchanged; then a GPU A/B (`max_staleness + 1`, masks, both) on held-out evaluation before any default changes (open question 3) | |
| Later, each on its own trigger | the unit plan of section 8 with the first workflow without groups (and RFC 0003); permits beyond `size`, with the lending rule, with RFC 0001's scheduling of child agents; one permit pool per policy with RFC 0003; a journal of finished rollouts if restarts become frequent | each with its own design | |

P1 to P4 deliver the whole measured gain on permit-bound runs. P5 is independent of them and also helps today's
accounting.

## Open questions

All eleven were decided on 2026-10-25, before this RFC was filed: 1 to 10 take the design study's recommendations, and
11 records the switch to rolling starts. Each keeps its options, so that the reasons stay on record.

1. **Per-rollout permits by default?** `rollout` as the default after the P3 GPU check, with `group` kept as the
   rollback, or opt-in for a release. *Decided 2026-10-25:* `rollout` becomes the default only after the GPU check
   (P4); P1 and P2 ship with `group` the default.
2. **How to answer the extra staleness?** (a) The warning plus documented `max_staleness + 1` advice; (b) raise
   `max_staleness` automatically when permits bind and the horizon has room; (c) a cap on groups in flight. *Decided
   2026-10-25:* (a). The bound is the user's staleness contract and never moves by itself; (c) is dominated, so there
   is no cap on groups in flight.
3. **May the stale filter become per datum?** That changes "stale drops are atomic per group" into "per datum, with
   the group's credit unchanged". *Decided 2026-10-25:* yes, opt-in when P5 lands; the default only after its own GPU
   A/B on held-out evaluation.
4. **Raise `max_concurrent_rollouts` in the recipes?** The replays predict +50% to +56% effective rate for the 30B
   workload at 336 permits with `max_staleness + 1`; the 27B workload's +27% to +45% at 368-376 is an extrapolation.
   *Decided 2026-10-25:* only after the P3 sweep, guided afterwards by `blocked_share/permits` and the engines' KV use.
5. **Publish the lease now,** or keep it internal to `GroupRolloutWorkflow` until a custom workflow asks? *Decided
   2026-10-25:* publish `RolloutLease` and `rollout_lease()` in `platoon.experimental`, with permits beyond `size`
   reserved: custom workflows are the extension point the lease is for.
6. **Defer the unit contract, two-phase credit and group-free credit?** *Decided 2026-10-25:* deferred until a workflow
   without groups is built: the unit contract with the first PPO-with-a-critic workflow (with RFC 0003, after a spike
   shows that Tinker can train a critic) or with the first swarm workflow.
7. **Label limits in groups or in rollouts?** *Decided 2026-10-25:* they keep counting groups. Per-label limits in
   rollouts come only when an environment whose capacity is its sandboxes needs them.
8. **The straggler policy?** *Decided 2026-10-25:* the settings stay unchanged and off by default, with the new
   censoring metrics, and there is no hedge, deadline cut, doomed cancel or member replacement (replacement to be
   revisited only if launches start failing). One change: under rolling starts each member's deadline is shifted by
   its own start delay, which is zero under co-start.
9. **Evaluation?** *Decided 2026-10-25:* unchanged. Its default group size is 1, and its worker loops bound its groups
   anyway.
10. **The wait before a rollout's first model call.** The crafting runs spent a median of 25-153 s per run, 34-84% of
    each rollout's time, before its first model call, on nodes about 91% idle; halving it would be worth up to +35%
    there, more than anything in this design. *Decided 2026-10-25:* it gets its own investigation later, out of scope
    here.
11. **Rolling starts or co-start for `GroupRolloutWorkflow`?** The first version of this design kept co-start for every
    workflow, so that start skew could not reach the straggler clock. *Decided 2026-10-25:* rolling starts, in P1, with
    each member's straggler deadline offset by its start delay; co-start stays only for workflows that do not run their
    rollouts on the lease, where it is needed for safety.

## Appendix A: measurements

### A.1 The workloads

| | 27B model, long agentic science tasks | 30B model, the same tasks | 4B model, short crafting tasks |
|---|---|---|---|
| Runs | 1: 70 steps, 13.3 h, 2,266 groups, 18,125 rollouts | 1: 167 steps, 22.9 h, 5,380 groups, 43,063 rollouts | 10, 5 arms × 2 seeds: 11-28 steps, 0.4-0.7 h and about 450 groups each |
| Batch, staleness, budget | 32 groups of 8, `max_staleness` 3, 256 permits (32 groups), horizon 128 | the same | 16 groups of 8, `max_staleness` 2, 384 permits, the whole horizon of 48 groups |
| Straggler policy | quorum 6, 600 s grace; agent timeout 1200 s | the same | none |
| Rollouts, engines | process workers; 3 engines | process workers; 12 engines | process workers; 5 engines |
| Arms | | | the `uniform` and `pass_rate` samplers, each without a rule and with `informative` admission and refill |

The 27B run predates the dispatcher; its worker loops held a group's slots exactly as the dispatcher's permits do.
Each rollout's start and end come from the trainer's worker log lines, joined to the rollout's event record (reward,
outcome, model-call spans): 100% of rollouts joined in every run. Platoon's own `rollouts_in_flight_mean` agrees with
the trace (168.4 against 171.7, and 191.1 against 190.7), and the idle accounting closes within 0.03% of the permit
time held in the long-rollout runs and 0.3% in the crafting runs.

### A.2 The trace replays

A trace-driven, event-driven model of the training loop: the dispatcher (the horizon, first-come, first-served
permits, label limits, admission at settle and refill); the trainer (each step's logged busy time, and the stale
filter on the oldest datum, stamped at a rollout's first model call as process workers stamp it); the engines
(processor sharing, with the decode-throughput exponents measured in the runs: 0.41-0.6 for the 27B workload,
0.77-0.86 for the 30B one, 0.62 for the crafting runs); and the rollouts (the `j`-th group started replays the `j`-th
observed group, with its rollouts' logged durations).

- **Validation.** Replaying today's accounting reproduces all 12 runs: run time within 0.3% for the long-rollout runs
  (13.26 h against 13.27 h, 22.89 h against 22.92 h) and within 3% for every crafting run, the idle share of held
  permits (0.328 against 0.329, 0.254 against 0.255), the binding constraint in every run, stale counts (4/4, 11/11,
  55/56) and rejected-group counts within 4%.
- **Noise.** Eight paired seeds (supply shuffled, durations and trainer time jittered) give a standard deviation of
  0.1-0.4 points on the effective-rate change for the long-rollout runs and at most 1.7 for the crafting runs.
- **What it cannot see.** Durations are exogenous and the tasks drawn are fixed (`pass_rate` would draw others), the
  engine response is observational, and the trainer's time is replayed per step. Evaluation traffic and the startup
  gate are not modelled. Effective rate and informative groups are proxies for learning: only a GPU A/B on held-out
  evaluation decides questions of training quality.

### A.3 What per-rollout permits buy

Changes against today's accounting, replayed. Today the 27B workload runs 13.3 h, trains 1,345 rollouts/h and 35.7
informative groups/h, has a mean data age of 1.32 versions and at most 33 groups open, and loses 461 fleet-seconds per
restart.

| 27B workload | Wall time | Effective rate | Informative groups/h | Stale rollouts | Mean age | Groups open, max | Restart loss, fleet-s |
|---|---|---|---|---|---|---|---|
| Today, budget raised to 368-376 (no code) | −14% to −21% | +13.7% to +22.6% | +10.5% to +18.9% | 2.5-3.0% | 1.74-1.79 | 47-48 | 549-576 |
| **Rolling starts** | −15% to −21% | +11.8% to +19.6% | +4.5% to +11.3% | 5.2-5.6% | 1.77-1.80 | 66 | 558-602 |
| Co-start | −15% to −20% | +11.4% to +19.3% | +4.4% to +12.0% | 5.0-5.1% | 1.75-1.78 | 66 | 552-595 |
| Co-start + `max_staleness + 1` | as co-start | +15.5% to +23.5% | +13.1% to +20.4% | 1.5-1.9% | 1.84-1.88 | 66 | 554-597 |
| Co-start + per-datum masks | as co-start | +13.6% to +21.8% | not comparable | 0.3-0.5% | 1.79-1.82 | 66 | 552-595 |
| Co-start + both | as co-start | +16.1% to +24.4% | not comparable | 0.1% | 1.86-1.90 | 66 | 554-597 |
| Co-start + budget 368-376 + `max_staleness + 1` | −27% to −36% | +27.3% to +45.0% | +17.1% to +31.8% | 6.6-7.8% | 2.41-2.51 | 94-95 | 638-734 |

Today the 30B workload runs 22.9 h, trains 1,860 rollouts/h, has a mean data age of 1.28 versions and at most 33
groups open, and loses 316 fleet-seconds per restart.

| 30B workload | Wall time | Effective rate | Informative groups/h | Stale rollouts | Mean age | Groups open, max | Restart loss, fleet-s |
|---|---|---|---|---|---|---|---|
| Today, budget raised to 336 (no code) | −19% to −21% | +22.4% to +25.2% | +19.3% to +21.9% | 1.1% | 1.61-1.62 | 43 | 342-350 |
| **Rolling starts** | −20% to −22% | +24.0% to +27.0% | +19.8% to +22.5% | 1.4-1.5% | 1.63-1.64 | 58 | 345-353 |
| Co-start | −19% to −21% | +22.9% to +25.8% | +19.1% to +22.0% | 1.2% | 1.61-1.62 | 57 | 343-351 |
| Co-start + `max_staleness + 1` | as co-start | +24.1% to +27.1% | +24.1% to +27.0% | 0.2% | 1.63-1.65 | 57 | 343-351 |
| Co-start + per-datum masks | as co-start | +23.3% to +26.2% | not comparable | 0.1% | 1.62-1.63 | 57 | 343-351 |
| Co-start + both | as co-start | +24.2% to +27.1% | not comparable | 0.0% | 1.64-1.65 | 57 | 343-351 |
| Co-start + budget 336 + `max_staleness + 1` | −35% to −38% | +50.1% to +56.2% | +42.8% to +46.0% | 2.3-2.7% | 2.21-2.26 | 72 | 355-373 |

- The absorbers and the larger budget were replayed with co-start only. In the base case rolling starts add 0.3 to 1.2
  points of effective rate and 0.2-0.5 points of stale drops.
- "Not comparable": under masks a group counts as informative if any datum of a mixed group survives, which overstates
  the count; the masks' effective rate weighs each rollout by the share of its datums that trained.
- Busy permits go 0.67 → 0.985 (27B) and 0.75 → 0.986 (30B) with co-start, and to 0.997-1.0 with rolling starts.
- The 30B workload's 336 permits on 12 engines stay within the load its run reached; the 27B workload's 368-376 on 3
  engines extrapolate past it (at most 256 running were observed).
- With rolling starts, finished rollouts held in the trainer while their siblings run go 82 → 124 (27B) and 65 → 87
  (30B), and a restart redoes 1.8-1.9× (27B) and 1.5× (30B) today's rollout-seconds: every rollout of every unsettled
  group, as today, but more groups are unsettled.
- The stale filter's selection, as the pass rate of trained rollouts minus that of all completed rollouts: for the 27B
  workload +0.000 today, +0.007 with per-rollout permits, +0.003 with `max_staleness + 1` and +0.002 with both
  absorbers; for the 30B one +0.001, +0.003, +0.001 and +0.001.

**The crafting runs** (horizon-bound): per-rollout permits change nothing in any of the ten, under either start rule.
The horizon decided every start: without admission every group started within a second of a publish, and with
admission and refill the horizon, counted in groups, still bound. The levers there are of another kind: per-datum
masks (+0.1% to +7.5% effective rate in the four runs without admission, and half the stale filter's selection in one,
+0.048 → +0.024-0.028), `max_staleness + 1` with permits to match (+5% to +34%, a staleness trade), and the wait
before the first model call (open question 10).

### A.4 Straggler cuts, priced

Censored rollouts' pass rates come from the traces, which saw them finish.

| Policy | What the replays say | Why not |
|---|---|---|
| Shorter quorum cuts (quorum 6 with 43-160 s, quorum 7 with 0 s) | each trains 3-14% fewer informative groups per hour than per-rollout permits alone; cut rollouts pass at 0.83-0.92 against 0.93 for those kept | biased against slow failures, and per-rollout permits alone train more |
| Hedging: start 8, keep the first 6 | loses to plain groups of 6 (27B: −6.9% informative groups against +0.5%; 30B: −7.4% against +13.3%) | the same bias, and two more sandboxes per group |
| Deadline cut: stop a group's stragglers when it would go stale at the next publish | up to +1.6 points of trained rollouts per hour (long-rollout runs), up to −3.5% (crafting runs) | censors 1-12% of the crafting rollouts, where long rollouts fail most |
| Cancel a group already too stale to train | saves at most 0.1% (long-rollout runs) and 0.9% (crafting runs) of wall time | hides 29-72 rollouts per run from the statistics the samplers read |
| Replace rollouts that fail to start | unbiased, since a launch failure says nothing about the policy | no rollout failed to start in any measured run |

### A.5 Sources

- Co-start with per-rollout release is the "sample backfill" that miles kept in production
  (`miles: miles/rollout/submission_scheduler.py`, `SampleBackfillSubmission`), as did DeepSeek-V4.1-Flash
  (arXiv 2609.19969, section 5.2.1).
- Rolling starts follow prime-rl's scheduler, which continues an open group before it opens a new one, one permit at a
  time (`prime-rl: src/prime_rl/orchestrator/dispatcher.py`). PipelineRL (`pipelinerl: pipelinerl/actor.py`), ROLL
  Flash (arXiv 2510.11345) and AReaL's `GroupedRolloutWorkflow` also schedule rollouts and assemble groups.
- Over-provision and defer: APRIL (arXiv 2509.18521) and Kimi K3 (arXiv 2607.24653).
