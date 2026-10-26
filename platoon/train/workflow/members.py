"""A group's members: the startup gate of process isolation, each member's lease block and start slot, the straggler
policy, and one member's rollout, in process or in a rollout worker (``arun_episode_single``).

Each member runs in a block of its draw's rollout lease
(``platoon.train.leases``) in ``_run_group_member``, around its whole rollout,
so under ``train.permit_release: rollout`` its permit comes back as it ends (in
process, once the teardowns it left running in the background have ended too,
at most ``CLEANUP_WAIT_SECONDS`` later) and a group's members start as permits
free, each member's straggler deadline carrying its own start delay;
``runs_rollouts_on_lease`` says which subclasses keep this.  Inside its block a
member takes a start slot of the run's start limit (``start_limit``,
:mod:`platoon.train.workflow.start_limit`).

The functions here that take the workflow first are its methods still: ``GroupRolloutWorkflow`` keeps each under its
name, so to replace one, replace the method, on the class or an instance, not the function.
"""

import asyncio
import contextlib
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from platoon.train.components import ENVIRONMENT_COLUMN
from platoon.train.labels import task_label
from platoon.train.leases import holding, rollout_lease
from platoon.train.workers.run import run_rollout_request
from platoon.train.workflow.context import _group_ready_callback
from platoon.train.workflow.output import _SingleRolloutOutcome
from platoon.train.workflow.start_limit import NO_START_SLOT, START_SLOT_GRACE_SECONDS, StartSlot, held_slot
from platoon.utils.rollout_workload import RolloutWorkload
from platoon.utils.stopping import CleanupScope

if TYPE_CHECKING:
    from platoon.train.workflow.group import GroupRolloutWorkflow

logger = logging.getLogger(__name__)


#: How long, at most, an in-process member's lease block that holds a permit waits for the teardowns its rollout left
#: running in the background before the permit comes back (``_run_group_member``): an env session's teardown at its
#: default ``close_timeout_s`` (30 s) waits that long for its calls in flight and as long again to close its env.
CLEANUP_WAIT_SECONDS = 60.0


async def _finish_cleanups(cleanups: CleanupScope | None, data: dict, rollout_number: int) -> None:
    """Wait, at most :data:`CLEANUP_WAIT_SECONDS`, for the teardowns that a member's rollout left running in the
    background, and warn when some still run then."""

    if cleanups is None or not cleanups.running():
        return
    unfinished = await cleanups.finish(CLEANUP_WAIT_SECONDS)
    if unfinished:
        logger.warning(
            "Rollout %d of task %s returns its permit with %d of its teardowns still running after %gs: until they "
            "end, more sandboxes and env sessions can be alive than train.max_concurrent_rollouts",
            rollout_number,
            data.get("task_id"),
            unfinished,
            CLEANUP_WAIT_SECONDS,
        )


class _GroupAdmission:
    """What a group's members share: one startup permit held for the whole
    group until its members are up, and when each member started (``starts``).

    The permit is released when every member has signalled readiness or exited,
    or after ``timeout_seconds`` -- whichever comes first -- and never twice.
    """

    def __init__(self, semaphore: asyncio.Semaphore | None, expected: int, timeout_seconds: float):
        self.starts = _MemberStarts()
        self._semaphore = semaphore
        self._pending = max(0, int(expected))
        self._released = semaphore is None
        self._timer: asyncio.TimerHandle | None = None
        if semaphore is not None and timeout_seconds > 0:
            # A group whose members never signal readiness (all of them failed
            # to launch, say) must not hold the permit forever.
            self._timer = asyncio.get_running_loop().call_later(timeout_seconds, self.release)

    def member_slot(self):
        """A one-shot callback for a single member."""

        fired = False

        def _slot() -> None:
            nonlocal fired
            if fired:
                return
            fired = True
            self._pending -= 1
            if self._pending <= 0:
                self.release()

        return _slot

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._semaphore is not None:
            self._semaphore.release()


class _MemberStarts:
    """When each member of a group started: the seconds it waited for its permit and then for its start slot, its start
    delay (the members' tasks are created together).  Exactly 0 for every member whose permit came with the group's and
    that took a slot at once (or needs none), whose straggler deadline is then the group's, as before; with rolling
    starts or a binding start limit a late member's deadline carries its delay behind the group's first start.  A member
    that never entered a block (an override of ``_run_group_member`` that does not call it) counts as started with the
    group."""

    def __init__(self) -> None:
        self.created = asyncio.get_running_loop().time()
        self.waiting: set[int] = set()  # members that wait for a permit or a start slot
        self._waited: dict[int, float] = {}

    def started(self, index: int, waited: float) -> None:
        self.waiting.discard(index)
        self._waited[index] = waited

    def offset(self, index: int) -> float | None:
        """Member ``index``'s start delay behind the group's first start; None while it waits."""

        if index in self.waiting:
            return None
        waited = self._waited.get(index)
        return 0.0 if waited is None else waited - min(self._waited.values())

    def together(self) -> bool:
        """Whether every member started with the group: none waits for its permit or slot, and none waited."""

        return not self.waiting and not any(self._waited.values())

    def least_offset(self, now: float) -> float:
        """The least start delay a member that still waits can have, at loop time ``now``."""

        return max(0.0, now - self.created - min(self._waited.values(), default=0.0))


def _admission_semaphore(workflow: "GroupRolloutWorkflow") -> asyncio.Semaphore:
    limit = max(1, int(getattr(workflow.config, "worker_launch_concurrent_groups", 2)))
    semaphore = getattr(workflow, "_admission", None)
    if semaphore is None or getattr(workflow, "_admission_limit", 0) != limit:
        semaphore = asyncio.Semaphore(limit)
        workflow._admission = semaphore
        workflow._admission_limit = limit
    return semaphore


async def _admit_group(workflow: "GroupRolloutWorkflow", group_size: int) -> "_GroupAdmission":
    """Take one startup permit for the WHOLE group, not one per rollout.

    Two things make this the right granularity.  A per-rollout permit
    released at ``exec`` bounds nothing at all: ``create_subprocess_exec``
    returns in well under a millisecond while the child then spends 20-30 s
    importing, so the gate must be readiness, not launch.  But a
    *per-rollout* readiness gate would then inject start-time skew inside a
    group, and the straggler policy starts its tail-grace clock when the
    quorum settles.  With FIFO admission and tasks created in group order,
    the same rollout numbers would deterministically start last and be
    cancelled having had a fraction of their budget, biasing the
    leave-one-out baseline toward whichever members launch early.  Admitting
    whole groups removes intra-group skew from startup; inter-group skew is
    harmless because each group has its own clock.  One acquire per group
    also cannot deadlock.

    Under rolling starts (``train.permit_release: rollout``) a group's
    members start as its permits are granted: the group holds its slot until
    the last of them is up, and each member's straggler deadline carries its
    own start delay (``_MemberStarts``).  With ``worker_start_method: exec``
    the gate can then bind, which ``rollout_client/startup_gate_waiting_mean``
    shows.
    """

    if workflow._isolation != "process":
        return _GroupAdmission(None, group_size, 0.0)
    semaphore = workflow._admission_semaphore()
    with workflow._monitor().startup_gate_wait():  # rollout_client/startup_gate_waiting_mean
        await semaphore.acquire()
    budget = float(getattr(workflow.config, "worker_startup_budget_seconds", 300.0)) + 60.0
    return _GroupAdmission(semaphore, group_size, budget)


async def _run_group_member(
    workflow: "GroupRolloutWorkflow", data: dict, rollout_number: int, admission: "_GroupAdmission"
) -> _SingleRolloutOutcome:
    """Run one member on one of its draw's rollout permits, reporting its readiness to the group's startup permit.

    The lease block covers the member's whole rollout: with ``rollout_isolation: process``, ``arun_episode_single``
    returns only once the member's worker process tree is reaped and the env sessions and sandboxes it left are
    deleted; in process, once its rollout coroutine, its subagents and their grading have ended, and a block that
    holds a permit (``train.permit_release: rollout``) then waits, at most :data:`CLEANUP_WAIT_SECONDS`, for the
    sandbox and env session teardowns that a cancellation or an early close left to finish in the background (the
    rollout's :class:`~platoon.utils.stopping.CleanupScope`).  A cancellation ends that wait, and no teardown; a
    member whose rollout had ended keeps its outcome.  Under ``permit_release: group`` the blocks hold nothing and
    no member waits, as before.  A cancelled member's block exits once its cancellation has unwound.  Inside it the
    member takes its start slot (:meth:`_member_start_slot`), which ``arun_episode_single`` hands to the rollout's
    host to give back at the first model call, and at the latest before that wait.  The seconds it waited for its
    permit and then for its slot are its start delay (:class:`_MemberStarts`).  The rollout runs with no lease
    (``holding(None)``), so a group it runs in turn takes no permits of its own, as its child agents take none, and
    no start slots."""

    ready = admission.member_slot()
    token = _group_ready_callback.set(ready)
    starts = admission.starts
    starts.waiting.add(rollout_number)
    try:
        block = rollout_lease().rollout(rollout_number)
        async with block as waited:
            cleanups = CleanupScope() if getattr(block, "holds", False) else None
            try:
                with cleanups if cleanups is not None else contextlib.nullcontext():
                    async with workflow._member_start_slot() as start:
                        starts.started(rollout_number, waited + start.waited)
                        with holding(None):
                            outcome = await workflow.arun_episode_single(data, rollout_number)
            except (Exception, asyncio.CancelledError):
                await _finish_cleanups(cleanups, data, rollout_number)
                raise
            with contextlib.suppress(asyncio.CancelledError):  # its rollout has ended: its outcome stands
                await _finish_cleanups(cleanups, data, rollout_number)
            return outcome
    finally:
        starts.waiting.discard(rollout_number)
        _group_ready_callback.reset(token)
        ready()


def _member_start_slot(workflow: "GroupRolloutWorkflow") -> StartSlot:
    """A member's start slot of the run's start limit (``start_limit``), whose timer gives it back after the group's
    ``worker_startup_budget_seconds`` plus :data:`~platoon.train.workflow.start_limit.START_SLOT_GRACE_SECONDS`
    without a model call; the null slot without a limit, and for a member of a group that another member's rollout
    runs, which would wait for the slot that member holds."""

    limit = workflow.start_limit
    if limit is None or held_slot() is not None:
        return NO_START_SLOT
    budget = float(getattr(workflow.config, "worker_startup_budget_seconds", 300.0))
    return limit.slot(timeout_seconds=budget + START_SLOT_GRACE_SECONDS, monitor=workflow._monitor())


async def _run_group_rollouts(workflow: "GroupRolloutWorkflow", data: dict) -> list:
    """Run ``group_size`` rollouts concurrently with the straggler policy.

    Without ``straggler_timeout_seconds`` this is equivalent to
    ``asyncio.gather``, except that once a member raises, or the group is
    cancelled, it cancels the members still running and raises only once
    they have ended, so that none runs on after the group's permits have
    come back. With it, once ``straggler_quorum`` members
    (default ``group_size - 1``) have settled, the remaining members get at
    most ``straggler_timeout_seconds`` more; leftovers are cancelled and
    reported as unobserved outcomes plus the ``straggler_cancelled`` scalar.
    A member that started after waiting for its rollout permit (rolling
    starts, ``train.permit_release: rollout``) gets its start delay on top
    of that deadline, and one that has not started when the quorum settles
    gets it when it starts, so a late start never shortens a member's grace.
    """
    group_size = workflow.config.group_size
    straggler_timeout = getattr(workflow.config, "straggler_timeout_seconds", None)
    quorum = getattr(workflow.config, "straggler_quorum", None)
    # The policy of the task's label (train.label_by).
    column = getattr(workflow, "label_by", ENVIRONMENT_COLUMN)
    policy = getattr(workflow.config, "straggler_policies", {}).get(task_label(data, column))
    if policy is not None:
        straggler_timeout, quorum = policy.timeout_seconds, policy.quorum
    admission = await workflow._admit_group(group_size)
    tasks = [asyncio.ensure_future(workflow._run_group_member(data, i, admission)) for i in range(group_size)]
    if straggler_timeout is None:
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            # A member raised, or this task was cancelled (gather then cancelled the members): cancel each member
            # once, since a second cancellation would cut its unwinding short, such as a worker's termination grace.
            for task in tasks:
                if not task.cancelling():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            admission.release()

    if quorum is None:
        quorum = max(group_size - 1, 1)
    pending: set[asyncio.Future] = set(tasks)
    late: list[asyncio.Future] = []  # cancelled at their own deadlines
    try:
        while pending and len(pending) > group_size - quorum:
            _done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        if pending and admission.starts.together():
            # Every member started with the group: one deadline, the quorum's plus the grace, for all of them.
            _done, pending = await asyncio.wait(pending, timeout=straggler_timeout)
        elif pending:
            deadline = asyncio.get_running_loop().time() + straggler_timeout
            late = await workflow._cancel_late_members(data, tasks, pending, admission.starts, deadline, quorum)
            pending = set()
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        admission.release()
        raise

    # A rollout can settle between the deadline and this cancellation;
    # only members that are still running count as cancelled stragglers.
    stragglers = [task for task in tasks if not task.done()]
    if stragglers:
        logger.warning(
            "Cancelling %s straggler rollout(s) for task %s after %ss grace (quorum %s/%s)",
            len(stragglers),
            data["task_id"],
            straggler_timeout,
            quorum,
            group_size,
        )
        for task in stragglers:
            task.cancel()
        # run_rollout's own cancellation cleanup marks trajectories as
        # interrupted; leaked environment sessions are reaped server-side.
        await asyncio.gather(*stragglers, return_exceptions=True)
    workflow.tracker.scalar(straggler_cancelled=float(len(stragglers) + len(late)))
    admission.release()

    outcomes: list = []
    for task in tasks:
        if task.cancelled():
            outcomes.append(
                _SingleRolloutOutcome(
                    result=None,
                    workload=RolloutWorkload(),
                    observed=False,
                    disposition="straggler_killed",
                )
            )
        else:
            # Propagates a legacy override's exception like gather did.
            outcomes.append(task.result())
    return outcomes


async def _cancel_late_members(
    workflow: "GroupRolloutWorkflow",
    data: dict,
    tasks: Sequence[asyncio.Future],
    pending: set[asyncio.Future],
    starts: _MemberStarts,
    deadline: float,
    quorum: int,
) -> list[asyncio.Future]:
    """Once the quorum settled, wait for the members still running, each until ``deadline`` (loop time: the
    quorum's plus the grace) plus its own start delay, and cancel each that runs past its own; a member whose block
    still waits gets its delay when it starts.  Returns once every member has ended, the cancelled ones unwound,
    with those it cancelled."""

    loop = asyncio.get_running_loop()
    number = {task: index for index, task in enumerate(tasks)}
    cut: list[asyncio.Future] = []
    while pending:
        now = loop.time()
        due: list[asyncio.Future] = []
        wake: float | None = None
        for task in pending.difference(cut):
            offset = starts.offset(number[task])
            # One whose block still waits cannot pass its deadline before the least start delay it can have.
            at = deadline + (starts.least_offset(now) if offset is None else offset)
            if offset is not None and at <= now:
                due.append(task)
            else:
                wake = at if wake is None else min(wake, at)
        if due and not cut:
            logger.warning(
                "Cancelling straggler rollout(s) for task %s past their grace from their own starts (quorum %s/%s)",
                data["task_id"],
                quorum,
                len(tasks),
            )
        for task in due:
            task.cancel()
        cut += due
        timeout = None if wake is None else max(0.0, wake - now)
        _done, pending = await asyncio.wait(pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
    return cut


async def arun_episode_single(
    workflow: "GroupRolloutWorkflow", data: dict, rollout_number: int = 0
) -> _SingleRolloutOutcome:
    """Run a single rollout and return training data.

    The blanket guard stays and now also covers request building: that adds
    ``pickle.dumps``, a write to node-local scratch and (in ``process``
    mode) opening a log file, any of which can raise on a full disk.  An
    escape here would propagate through ``_run_group_rollouts``'s
    ``task.result()`` and cost ALL group members, including the ones that
    completed and would have satisfied ``min_successful_group_size``.

    Args:
        data: Dictionary containing 'task_id' and optionally other task data.
        rollout_number: Index of this rollout within the group.

    Returns:
        Internal outcome containing converted data when usable and the raw
        workload even when conversion or rollout execution failed.
    """
    try:
        request = workflow._build_worker_request(data, rollout_number)
        monitor = workflow._monitor()
        start = held_slot()  # its host gives it back at the first model call; none without a limit
        first_call: dict[str, Any] = {} if start is None else {"on_first_call": start.first_call}
        if workflow._isolation != "process":
            with monitor.rollout_in_flight():
                response = await run_rollout_request(
                    request,
                    workflow.model_info.llm,
                    rollout_fn=workflow.rollout_fn,
                    reward_processor=workflow.reward_processor,
                    **first_call,
                )
            monitor.record_rollout_timing(response.timing)
            return workflow._outcome_from_response(response, response.status)

        ready = _group_ready_callback.get()
        with monitor.rollout_in_flight():
            outcome = await workflow.supervisor.run(
                request,
                log_dir=workflow._worker_log_dir(request.model.checkpoint_version),
                on_ready=ready,
                on_launch=monitor.register_worker,
                on_reap=monitor.unregister_worker,
                **first_call,
            )
        workflow._record_worker_telemetry(outcome)
        return workflow._outcome_from_response(outcome.response, outcome.disposition)
    except asyncio.CancelledError:
        # The straggler policy owns cancellation.
        raise
    except Exception as exc:
        logger.exception(
            "Error in tinker workflow for task %s and rollout %s: %s",
            data.get("task_id"),
            rollout_number,
            exc,
        )
        workflow.tracker.scalar(rollout_worker_launch_failed=1.0)
        return _SingleRolloutOutcome(
            result=None,
            workload=RolloutWorkload(),
            observed=False,
            disposition="worker_launch_failed",
        )
