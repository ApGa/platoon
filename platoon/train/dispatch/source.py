"""The run's training draws (:class:`RolloutSource`, RFC 0021 section 2.3): what a trainer that trains on rollouts of
the run's tasks draws them through, and the groups they settle into.

A source holds one start's draws:

- the run's per-task statistics (its task records);
- the order the training tasks are drawn in, and the task samplers that draw it;
- the group admission rules;
- the dispatcher, which starts each draw's group when the horizon, the rollout permits and the labels allow;
- once started, the queue the groups settle into (the groups a resume kept first), where the steps that consume them
  dequeue them.

It runs the training workflow's ``arun_episode`` for each draw, on the draw's row, with the draw's provenance in it
when the trainer asks for it (:meth:`RolloutSource.episode_row`).  When the run's trained policies step independently
(``policies``), each group that settles is split into shares, one per trained policy, which its steps take from the
source's share ledger (``ledger``, :mod:`platoon.train.dispatch.shares`) instead of the queue, and the dispatcher keeps
a horizon for each policy (:mod:`platoon.train.dispatch.horizons`); a commit captures the draws at once
(:meth:`RolloutSource.capture`).  It remembers the draws each step committed, with
their fates (``consumed``), until the next step starts.  It takes part in every full checkpoint of the run
(``loop_state``'s ``tasks``, ``next_position`` and ``dispatch``, and ``credit``, the state of the workflow's credit
estimators that have state, taken with the groups the dispatcher keeps), which a start that resumes takes up.

Each part is a module function that takes the source:

- the per-task statistics' beginning and start: here;
- building and starting the dispatcher: :mod:`platoon.train.dispatch.setup`;
- running a group: :mod:`platoon.train.dispatch.runner`;
- what a checkpoint saves of the draws: :mod:`platoon.train.checkpoints.saving`; taking it up:
  :mod:`platoon.train.checkpoints.resume`.

A trainer gets the start's source from its run (:meth:`~platoon.train.trainer.run.TrainerRun.rollouts`), the rl
trainer over the tasks it holds.
"""

import asyncio
import logging
from collections.abc import Callable, Coroutine, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from platoon.programs import DRAW_KEY, DrawInfo
from platoon.train.checkpoints import resume, saving
from platoon.train.checkpoints.pruning import read_checkpoint_records
from platoon.train.credit.runtime import CREDIT_KEY, load_credit_states
from platoon.train.dataloader import PlatoonTinkerDataloader, _run_steps
from platoon.train.dispatch import setup
from platoon.train.dispatch.horizons import Horizons
from platoon.train.dispatch.settled import SettledGroup, as_settled_group
from platoon.train.dispatch.shares import Shares
from platoon.train.task_records import TaskRecorder, record_lost_state, resume_record
from platoon.train.task_stats import DISPATCH_KEY, MAIN
from platoon.train.trainer.waiting import _next_group
from platoon.utils.stats_tracker import get as get_tracker

if TYPE_CHECKING:
    from platoon.train.admissions import AdmissionSet
    from platoon.train.checkpoints.resume import StartPoint
    from platoon.train.dispatch.dispatcher import BoundDraw, Dispatcher
    from platoon.train.policy.set import PolicySet
    from platoon.train.trainer.api import TaskSources
    from platoon.train.trainer.signals import Watchdog
    from platoon.utils.stats_tracker import StatsTracker

logger = logging.getLogger(__name__)

Key = tuple[int, int]


class RolloutSource:
    """One start's training draws and the groups they settle into, over the run's config (its ``train.*`` settings),
    its run directory (``run_dir``), its training tasks (``tasks``: the dataset, the order of its epochs and the task
    samplers that draw it, as ``platoon train`` lists them), the training workflow, and each policy's version now
    (``versions()``, which the task samplers read), of which the per-task statistics record the sampled policy's
    (``leader``'s).  It opens nothing until :meth:`open`."""

    # Its entries of a full checkpoint row's loop_state, as a participant of the run's checkpoints.
    keys: tuple[str, ...] = ("tasks", saving.ORDER_POSITION_KEY, DISPATCH_KEY, CREDIT_KEY)
    # The run's policies, when its trained policies step independently (None: in lockstep), which the run sets as it
    # opens the source; then, once open, each policy's horizon and the ledger of the groups' shares.
    policies: "PolicySet | None" = None
    horizons: Horizons | None = None
    ledger: Shares | None = None

    def __init__(
        self,
        config: Any,
        *,
        run_dir: str,
        tasks: "TaskSources",
        workflow: Any,
        versions: Callable[[], Mapping[str, int]],
        leader: str = MAIN,
    ) -> None:
        self.config = config
        self.run_dir = run_dir
        self.tasks = tasks
        self.workflow = workflow
        self.versions = versions
        self.leader = leader
        # Whether each draw's row carries its provenance (DrawInfo), which the trainer asks for (TrainerNeeds); the
        # run sets it as it opens the source.
        self.draw_provenance = False
        # The draws the steps committed since the last step started, by key, each with its fate.
        self.consumed: dict[Key, str] = {}
        # Opened with the start (open): the per-task statistics (None when the run keeps none, or they could not start;
        # tasks_lost then says what a resume could not read), the order of the training tasks and the step after the
        # run's last, the group admission rules (None: every entry admits every group) and the dispatcher.
        self.recorder: TaskRecorder | None = None
        self.tasks_lost: Any = None
        self.order: PlatoonTinkerDataloader | None = None
        self.end_step = 0
        self.admissions: AdmissionSet | None = None
        self.dispatcher: Dispatcher | None = None
        # Set as it starts (start): the queue its groups settle into, the watchdog that running groups feed, and the
        # tracker their statistics go to when no group output carries them.
        self.results: asyncio.Queue[SettledGroup] | None = None
        self.watchdog: Watchdog | None = None
        self.tracker: StatsTracker | None = None

    def version(self) -> int:
        """The sampled policy's version now (``leader``'s): the version the per-task statistics record a group at, and
        a group admission rule decides it at."""

        return self.versions()[self.leader]

    def training_tasks(self) -> Any:
        """Every task training draws from, once each, in pool order (:func:`training_tasks`)."""

        return training_tasks(self.tasks)

    # --- what a start opens, before its launch save --------------------------------------------------------------

    def open(self, start: "StartPoint") -> None:
        """Open the start's draws before its launch save, which saves them: the per-task statistics, with those of the
        checkpoint the start resumes from (:meth:`begin_task_records`); the order of the training tasks, and the step
        after the run's last; and the dispatcher of the training groups from the start's step
        (:meth:`new_dispatcher`).  The draws the checkpoint left outstanding are taken up as the start point is
        (:meth:`resume`)."""

        self.begin_task_records(start.record, start.step)
        self.order = PlatoonTinkerDataloader(
            self.tasks.train_dataset, self.config.train.batch_size, order=self.tasks.train_order
        )
        self.end_step = _run_steps(self.config, len(self.order))
        if self.policies is not None:  # its trained policies step independently: a horizon each
            self.horizons = Horizons.of_run(self.config, self.policies, steps=max(0, self.end_step - start.step))
            logger.info(
                "The run's trained policies step independently, each on its own clock; their horizons, in shares: %s",
                ", ".join(
                    f"{policy} {size if size is not None else 'unbounded'}"
                    for policy, size in self.horizons.sizes().items()
                ),
            )
        self.dispatcher = self.new_dispatcher(self.order, start.step, self.end_step)
        if self.policies is not None:
            self.ledger = Shares(
                self.policies,
                dispatcher=self.dispatcher,
                record_fates=lambda keys, fates: self.record_fates(keys, fates=fates),
            )

    def begin_task_records(self, record: Any, step: int) -> None:
        """Begin this start's per-task statistics (:func:`_begin_task_records`)."""

        _begin_task_records(self, record, step)

    def new_dispatcher(self, order: PlatoonTinkerDataloader, start_step: int, end_step: int) -> "Dispatcher":
        """The dispatcher of the start's training groups (:func:`~platoon.train.dispatch.setup._new_dispatcher`)."""

        return setup._new_dispatcher(self, order, start_step, end_step)

    def started(self) -> None:
        """The launch save succeeded: the per-task statistics log the start (:func:`_task_records_started`)."""

        _task_records_started(self)

    def close(self) -> None:
        """Close the per-task statistics' event log."""

        if self.recorder is not None:
            self.recorder.close()

    # --- a participant of the run's full checkpoints ---------------------------------------------------------------

    async def checkpoint_state(self, name: str) -> dict[str, Any]:
        """Its entries of full checkpoint ``name``: the statistics' snapshot, the dispatcher's state with the groups it
        keeps, and under a group admission rule where the order stands
        (:func:`~platoon.train.checkpoints.saving._checkpoint_state`)."""

        return await saving._checkpoint_state(self, name, {})

    def capture(self, name: str) -> Coroutine[Any, Any, dict[str, Any]]:
        """When the run's policies step independently, what commit ``name`` saves of the draws, taken at once under the
        commit lock, with the shares of each kept group; the writing still to do
        (:func:`~platoon.train.checkpoints.saving.capture_state`).  The lockstep checkpoints await
        :meth:`checkpoint_state` instead."""

        return saving.capture_state(self, name)

    def resume(self, start: "StartPoint") -> None:
        """Take up the draws that the checkpoint the start resumes from left outstanding (a fresh run has none):
        :meth:`restore`, into the dispatcher :meth:`open` built; and the state its row saved of the workflow's credit
        estimators that have state (:func:`~platoon.train.credit.runtime.load_credit_states`)."""

        assert self.dispatcher is not None  # open() built it: the source registers once it is open
        if start.record is not None:
            self.restore(self.dispatcher, start.record, start.policy_version)
            load_credit_states(self.workflow, start.loop_state.get(CREDIT_KEY))
        if self.ledger is not None:  # each policy's shares consumed after the row it resumes from go back
            self.ledger.revert(start)

    def restore(self, dispatcher: "Dispatcher", record: Any, policy_version: int) -> None:
        """Take up what checkpoint ``record`` saved of the draws: the groups it kept, the draws to draw again and the
        order's position (:func:`~platoon.train.checkpoints.resume._restore_dispatch`)."""

        resume._restore_dispatch(self, dispatcher, record, policy_version)

    # --- running its groups ----------------------------------------------------------------------------------------

    def episode_row(self, draw: "BoundDraw") -> dict[str, Any]:
        """The row the group of ``draw`` runs on: the draw's own, or, when the trainer asks for draw provenance, a copy
        with the draw's :class:`~platoon.programs.DrawInfo` under ``DRAW_KEY``, which the workflow passes on to each
        episode (``ctx.draw``)."""

        if not self.draw_provenance:
            return draw.row
        assert self.dispatcher is not None  # it started the draw's group
        info = DrawInfo(self.run_dir, draw.start, draw.seq, draw.position, self.dispatcher.step)
        return {**draw.row, DRAW_KEY: info}

    def start(
        self,
        task_group: asyncio.TaskGroup,
        *,
        results: "asyncio.Queue[SettledGroup] | None" = None,
        watchdog: "Watchdog | None" = None,
        tracker: "StatsTracker | None" = None,
        stop: asyncio.Event | None = None,
    ) -> "Dispatcher":
        """Start the dispatcher in ``task_group``, each group a task of its own there, its groups settling into
        ``results`` (default: a queue of its own; the share ledger in its place when the policies step independently),
        the groups a resume kept first; until ``stop`` is set, or the task group ends (the rl trainer's loops end it
        with ``TerminateTrainLoop``).  Running groups feed ``watchdog``, and a group whose output carries no
        statistics of its own records them in ``tracker`` (default: the ``train`` tracker).  The dispatcher
        (:func:`~platoon.train.dispatch.setup._dispatch`)."""

        self.results = results if results is not None else asyncio.Queue()
        self.watchdog = watchdog
        self.tracker = tracker if tracker is not None else get_tracker("train")
        return setup._dispatch(self, task_group, stop=stop)

    async def take(self, n: int, *, stop: asyncio.Event | None = None) -> list[SettledGroup]:
        """The next ``n`` settled groups, in the order they settled (the groups a resume kept first), each counted as
        dequeued: its slot of the horizon goes to the next draw.  A group that failed has no output (``output``
        None).  Once ``stop`` is set (the run's shutdown request, on which the dispatcher starts no more groups),
        ``TrainInterrupted`` instead, as the groups it waits for may never start: the step commits none of its groups,
        which a checkpoint keeps.  Under a group admission rule only the rl trainer's steps consume groups, since they
        count each against the rule's cap."""

        if self.admissions is not None:
            raise RuntimeError("under a group admission rule, only the rl trainer's steps consume the run's groups")
        if self.ledger is not None:
            raise RuntimeError("the run's policies step independently: each takes its shares (source.ledger.take)")
        assert self.results is not None and self.dispatcher is not None  # start() started the dispatcher
        groups = []
        for _ in range(n):
            groups.append(as_settled_group(await _next_group(self.results, stop, interruptible=stop is not None)))
            self.dispatcher.dequeued()
        return groups

    def committed(self, keys: Sequence[Key | None], *, fates: Mapping[Key, str] | None = None) -> None:
        """A step consumed the groups of these draws (each group's ``key``; None for one without a draw): each one's
        fate goes to the per-task statistics (:meth:`record_fates`), then the dispatcher no longer keeps them, so that
        a full checkpoint saves neither them nor their draws.  ``fates`` gives the fate the step decided for each
        group, by draw (``consumed``, ``stale`` or ``failed``); a group it leaves out has its fate already, as a group
        that a group admission rule decided as it settled does.  Without it, each group of these draws that the
        dispatcher holds was consumed, or failed when it has no output.  The groups are those the dispatcher holds
        until then, however the step dequeued them (:meth:`take`, or ``results``)."""

        self.record_fates(keys, fates=fates)
        for key in keys:
            if key is None:
                continue
            group = self.dispatcher.holding(key) if fates is None and self.dispatcher is not None else None
            if (fate := fates.get(key) if fates is not None else _fate_of(group)) is not None:
                self.consumed[key] = fate
        if self.dispatcher is not None:
            self.dispatcher.committed(keys)

    def record_fates(self, keys: Sequence[Key | None], *, fates: Mapping[Key, str] | None = None) -> None:
        """The per-task statistics' part of :meth:`committed`: each group's fate, in the order of ``keys``, with its
        record, which the group the dispatcher holds carries.  A step that a test drives without the dispatcher (its
        ``_dispatch`` replaced) records the fates alone."""

        recorder, dispatcher = self.recorder, self.dispatcher
        if recorder is None:
            return
        for key in keys:
            if key is None:
                continue
            group = dispatcher.holding(key) if dispatcher is not None else None
            fate = fates.get(key) if fates is not None else _fate_of(group)
            if fate is None:
                continue
            if group is not None and group.record is not None:
                recorder.fate(group.record, fate)
            else:  # a group a start kept after its statistics stopped: no record of its own
                recorder.fate_of(key, fate)

    def step_started(self, policy: str | None = None) -> None:
        """The next step started (:meth:`~platoon.train.trainer.run.StepClock.advance`): the horizon moves one batch
        on, and the draws the last step committed are forgotten (``consumed``).  ``policy``: whose step, when the
        policies step independently: its horizon moves on
        (:meth:`~platoon.train.dispatch.shares.Shares.step_started`); in lockstep, None."""

        if policy is not None and self.ledger is not None:
            self.ledger.step_started(policy)
            return
        self.consumed = {}
        if self.dispatcher is not None:
            self.dispatcher.step_started()


def training_tasks(tasks: "TaskSources") -> Any:
    """Every task training draws from, once each, in pool order: a given order's tasks (a weighted order's, those its
    first epoch leaves out included), else the training dataset's rows."""

    items = getattr(tasks.train_order, "items", None)
    return tasks.train_dataset if items is None else items


def _fate_of(group: SettledGroup | None) -> str | None:
    """The fate of a group a step consumed, which the step did not decide: ``consumed``, or ``failed`` for one without
    output; None for a draw whose group the dispatcher does not hold."""

    if group is None:
        return None
    return "consumed" if group.output is not None else "failed"


def _begin_task_records(source: RolloutSource, record: Any, step: int) -> None:
    """Begin the source's records of the run's per-task statistics (:class:`TaskRecorder`) over every task training
    draws from, with those of the checkpoint row ``record`` the start resumes from, at step ``step``, when it names a
    snapshot that reads back, else those of the newest older checkpoint whose snapshot does, else from empty
    (:func:`_task_records_started` records why in ``lock.json``)."""

    source.recorder = None
    source.tasks_lost = None
    try:
        source.recorder, source.tasks_lost = TaskRecorder.begin(
            run_dir=source.run_dir,
            config=source.config,
            tasks=source.training_tasks() or [],
            records=read_checkpoint_records(source.run_dir),
            checkpoint=record.name if record is not None else None,
            resume=resume_record(record) if record is not None else None,
            batch=step,
        )
    except Exception:  # the statistics never fail training
        logger.warning("This start keeps no per-task statistics: they could not start", exc_info=True)


def _task_records_started(source: RolloutSource) -> None:
    """The launch save succeeded: the start's events go to the event log, and a start that could not load the
    statistics its checkpoint named says so in ``lock.json`` (``tasks_state_lost``).  Never raises."""

    if source.recorder is None:
        return
    try:
        source.recorder.started()
    except Exception:  # the statistics never fail training
        logger.warning("Could not log the start of this start's per-task statistics", exc_info=True)
    if source.tasks_lost is not None:
        record_lost_state(source.run_dir, source.tasks_lost)
