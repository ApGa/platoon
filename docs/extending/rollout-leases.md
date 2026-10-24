# Run rollouts on the lease (experimental)

A training run's rollouts share a budget of rollout permits, `train.max_concurrent_rollouts`. With
`train.permit_release: rollout`, each rollout's permit comes back as that rollout ends, so the next group starts on
the permits that come back instead of waiting for a group's slowest rollout. A workflow gets this by running each of
its rollouts in a block of its draw's **lease**. `GroupRolloutWorkflow`, the default, already does: this page is for
the authors of a workflow of their own, and of a `GroupRolloutWorkflow` subclass that starts its members itself. What
the setting does to a run (the speed, the staleness it costs, the metrics) is in
[Rollout permits](../how-to/train-and-eval.md#rollout-permits-experimental).

The lease lives in `platoon.experimental` and is **experimental**: it is documented, and its names are in the public
API's snapshot, but a minor release may change it, and its CHANGELOG then says how. Pin the platoon version you build
against. The [experimental API reference](../reference/experimental.md) lists both names with their signatures.

## Where the lease sits

```yaml
train:
  permit_release: rollout            # each rollout's permit comes back as the rollout ends (default: group)
  max_concurrent_rollouts: 256       # the rollouts that run at once
  workflow_config: {group_size: 8}
```

When the dispatcher starts a draw (a training group: one `arun_episode(row)` of your workflow), it reserves the draw's
group size of rollout permits, and the run grants reservations first come, first served:

- **Rolling starts.** A draw whose workflow says it runs every rollout on the lease (`runs_rollouts_on_lease = True`)
  gets its permits one at a time, as they free: it starts on the first, and each of its rollouts starts as its permit
  is granted.
- **Co-start.** Any other draw gets its whole reservation once that many permits are free, so it may start all its
  rollouts together, as every draw does under `permit_release: group`.

Either way a rollout's block returns its permit when it exits, and whatever the draw still holds comes back when
`arun_episode` returns. A workflow that never touches the lease behaves exactly as before: its draw is granted whole,
and its permits come back when it returns. Under `permit_release: group` a block holds nothing, and the permits come
back together.

## The lease

```python
from platoon.experimental import RolloutLease, rollout_lease


async def one_rollout(i: int) -> None:
    lease: RolloutLease = rollout_lease()  # in the workflow's own task
    # lease.size: the permits the draw reserved, its entry's group size (0 outside a training draw)
    async with lease.rollout(i) as waited:
        ...  # rollout i, on a permit; waited: the seconds the block waited for it (0.0 when it entered at once)
```

`rollout_lease()` reads the lease of the training draw whose workflow runs in this asyncio context: the trainer sets it
around each draw's `arun_episode(row)`, and the tasks your workflow creates copy it. Outside a training draw (an
evaluation, a test, a direct call) it is a lease of size 0 whose blocks enter at once and hold nothing, so a workflow
runs its rollouts the same way everywhere.

`async with lease.rollout(i):` waits until the draw holds a granted permit that none of its rollouts uses (at once
under co-start), runs the block on it, and returns the permit to the run when the block exits.

## The rules a workflow keeps

1. **One block per rollout, around the whole rollout.** The permit stands for the rollout's worker and sandboxes: let
   the block exit only once the rollout has ended and they are gone. Cancel and await a rollout inside its block, so
   that its cancellation has unwound before its permit comes back, and retry a rollout inside its block too.
2. **At most `size` rollouts, each index once.** Run rollout `i`, `0 <= i < lease.size`, in `lease.rollout(i)`, once.
   In a training draw, under either setting, `lease.rollout(i)` raises `IndexError` for an index outside that range,
   and a second block for an index, while the first runs or after it ended, raises `RuntimeError` as it enters: a
   misused index fails its group rather than run a rollout on no permit.
3. **The flag says every rollout runs in a block.** Set `runs_rollouts_on_lease = True` on the class only when it does;
   the draw then gets rolling starts. Without it the draw is granted whole before `arun_episode` starts, and each block
   still returns its permit when its rollout ends.
4. **The rest comes back with the draw.** Whatever the draw still holds comes back when `arun_episode` returns, raises
   or is cancelled; a block that enters after that, or still waits for its permit then, holds nothing, and the run
   warns once for that draw.
5. **Read it in the task.** Context variables follow tasks, not threads: `loop.run_in_executor` does not see the lease
   unless the context is copied, so read `rollout_lease()` in the workflow's task before handing work to a thread.
6. **Reserved.** Asking for permits beyond `size`, for a draw whose size varies or for an episode's child agents, is
   left to a later release (RFC 0020).

Rule 1 is yours to keep: if a workflow of your own breaks it, more rollouts run at once than
`train.max_concurrent_rollouts`, and nothing warns. The run's check of `rollout_client/rollouts_in_flight_max` against
the budget counts only `GroupRolloutWorkflow`'s members, and an evaluation's `GroupRolloutWorkflow` left on the default
`stats_scope`, `train`, counts too: a train script's can be, while `platoon train` gives evaluation its own
([Troubleshooting](../how-to/troubleshooting.md)).

## `GroupRolloutWorkflow` and its subclasses

`GroupRolloutWorkflow` runs each member in its block in `_run_group_member`, the one method every member passes
through, around `arun_episode_single`. With `rollout_isolation: process` the block exits only once the member's worker
process tree is reaped and the env sessions and sandboxes it left are deleted. In process, the default, it exits once
the member's rollout coroutine, its subagents and their grading have ended, and a sandbox or env session teardown that
platoon finishes in the background can outlive it: one a cancellation interrupts (a straggler cut), much as it outlives
the group's release under `permit_release: group`, and one a rollout leaves running as it ends (an env torn down once
the code still running in it ends), whose permit `rollout` hands to the next rollout meanwhile. With every teardown
left behind, a test with a budget of 4 had 11 sandboxes alive at once under `rollout` and 6 under `group`.

It declares `runs_rollouts_on_lease = True`. A subclass keeps rolling starts unless it may start members some other
way:

| A subclass that overrides | Rolling starts | Permits per rollout |
|---|---|---|
| `compute_baselines`, `assign_advantages`, `_run_group`, `_make_task_output` or `arun_episode_single` | yes | yes |
| `_run_group_rollouts`, `_run_group_member` or `arun_episode` | no: its groups are granted whole, unless its own class body sets `runs_rollouts_on_lease = True` | where its members still run in `GroupRolloutWorkflow`'s `_run_group_member` (an override that calls `super()._run_group_member`) |

The class flag says so: such a subclass reads as `runs_rollouts_on_lease = False` unless it sets the flag itself. An
override from a mixin or from another branch of a diamond counts too: a flag set in a class body covers the overrides
of that class and its bases only.

**Straggler deadlines.** Under rolling starts a group's members start over the time its permits take to free, so each
member's straggler deadline carries its own start delay: once `straggler_quorum` members have settled, a member gets
`straggler_timeout_seconds` more plus the seconds it started after the group's first member, so a late start never
shortens its grace. `GroupRolloutWorkflow` enters every member's block at once, so the value of each block, the
seconds it waited, is that member's start delay; a workflow with a straggler clock of its own that enters its blocks at
different times records `loop.time()` inside each block instead, and offsets each rollout's deadline by its start less
the first's.

**The startup gate** (`worker_launch_concurrent_groups`, process isolation only) is still one slot per group, held
until every member is up or has exited, which under rolling starts includes the time its last permit takes. With the
fork server a worker is up in a fraction of a second and at most one group is partly granted at a time, so the gate's
second slot serves the next group; with `worker_start_method: exec` it can hold groups back, which
`rollout_client/startup_gate_waiting_mean` shows.

## Recursive agents and swarms

An episode is one rollout on one permit, however many agents it runs: a recursive tree (its root, every child agent and
the children's grading, by their forked env sessions or by verifier agents) or a swarm holds its permit until the whole
episode has ended, and its children take none of their own. Their concurrency stays under the episode's own settings
(`agents.max_depth`, `agents.budget`, `agents.child_max_steps`, `agents.child_timeout_s` and the rollout's time limit);
`max_concurrent_rollouts` does not bound an episode's live agents, as it never did. Letting child agents take permits
of the run's budget is left to a later release (RFC 0020). A group that a member's rollout runs in turn is part of the
episode too: `GroupRolloutWorkflow` runs each member's rollout with the no-op lease, so the nested group's members take
no permits of their own.

## A worked example

A workflow of your own that runs a task's attempts at once, each on its own permit:

```python
# acme_workflows/attempts.py
import asyncio
from typing import Any

from platoon.experimental import rollout_lease


class Attempts:
    """Runs ``config.group_size`` attempts of each task at once, each on its own rollout permit."""

    runs_rollouts_on_lease = True  # every attempt runs inside lease.rollout(i)

    def __init__(self, *, config: Any, **trainer_arguments: Any) -> None:
        self.config = config  # train.workflow_config, as platoon train passes it

    async def run_attempt(self, row: dict, index: int) -> list[Any]:
        """One attempt of the task: its datums, once its worker and sandboxes are gone."""

        raise NotImplementedError

    async def arun_episode(self, row: dict) -> list[Any]:
        lease = rollout_lease()  # read it in the workflow's own task

        async def attempt(index: int) -> list[Any]:
            async with lease.rollout(index):  # waits for a permit, and returns it once the attempt has ended
                return await self.run_attempt(row, index)

        attempts = await asyncio.gather(*(attempt(index) for index in range(self.config.group_size)))
        return [datum for datums in attempts for datum in datums]
```

A real workflow implements `run_attempt` and centers each attempt's advantages before it returns the datums, as
`GroupRolloutWorkflow` does. `platoon train` constructs it as it constructs any workflow class
(`environments[].workflow: acme_workflows.attempts:Attempts`): with the run's rollout function, task loader, workflow
config and model, which the constructor takes by keyword.

**Test it.** Outside a training draw `rollout_lease()` is the no-op lease, so a unit test calls `arun_episode` as it is.
Check rule 1 there, by counting your rollouts as they run and asserting that each runs inside its block: a run cannot
check it for you, since `rollout_client/rollouts_in_flight_*`, and the warning that more rollouts ran at once than
`train.max_concurrent_rollouts`, count only `GroupRolloutWorkflow`'s members. A run does check rule 2 (its errors) and
rule 4 (its warning) for any workflow.
