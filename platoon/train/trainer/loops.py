"""The training loop of one start (``_train_loop``) and its step (:class:`TrainStep`); and stopping the loops
(``shutdown_loops``).

The loop starts the dispatcher, whose groups settle into the queue the steps dequeue from, publishes the first sampler,
and sets up what its steps share (:class:`StepContext`).  Then it runs one :class:`TrainStep` per step to the end
batch.  A step reads top to bottom (:meth:`TrainStep.run`), in seven stages: it collects settled groups into
microbatches, prepares each microbatch's datums and sends them, updates the policy once per minibatch through its
learner (:mod:`platoon.train.policy.learner`), under its objective (:mod:`platoon.train.objectives`), then commits,
publishes the new policy, moves on to the next step and logs its metrics.  The parts of each stage are in
:mod:`platoon.train.step`.

The step's flow stays in this module, beside the loop, because tests read it here: the source of the batch transforms
before the one zero-advantage filter (:meth:`TrainStep.prepare`) and of the exact non-submitted guard
(:func:`record_workload_metrics`, with the loop's local names); the logger of the line that says why a step took no
update; and the five globals the tests patch here, ``client_monitor``, ``client_step_metrics``,
``_ptc_cell_guard_stats``, ``metric_labels`` and ``compute_training_metrics`` (which the objective computes its
metrics with, through :meth:`TrainStep._training_metrics`, for that reason).

``_train_loop`` and ``shutdown_loops`` take the trainer first, and are its methods still: ``PlatoonTinkerRLTrainer``
keeps each under its name, so to replace one, replace the method, on the class or an instance, not the function.  A
step calls the trainer's other seams through the trainer too.
"""

import asyncio
import dataclasses
import logging
import time
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import tinker

from platoon.train import entry_settings
from platoon.train.admissions import AdmissionSet
from platoon.train.dispatch.permits import _rollouts_at_once
from platoon.train.dispatch.settled import _output_label, _record_group_stats, _Settled, _settled
from platoon.train.labels import label_column, metric_labels, metric_prefix
from platoon.train.objectives.api import objective_is_zero_for_zero_advantages
from platoon.train.objectives.policy_gradient import PolicyGradient, compute_training_metrics
from platoon.train.policy.learner import Learner
from platoon.train.policy.policy import Policy
from platoon.train.policy.spec import MAIN, PolicySpec
from platoon.train.step.batch_transforms import (
    BatchTransformContext,
    depth_credit_metrics,
    filter_zero_advantage_datums,
    get_datum_counts,
    has_zero_objective,
    run_batch_transforms,
    set_loss_normalization_token_counts,
    summarize_depths,
)
from platoon.train.step.collect import StepDispatch, StepTally
from platoon.train.step.counting import GroupCounting
from platoon.train.step.datums import _step_datums, advantage_scale, scale_advantages, trained_rows
from platoon.train.step.metrics import (
    Censoring,
    RunTotals,
    _ptc_cell_guard_stats,
    record_batch_counts,
    record_batch_stats,
    record_label_limits,
    record_sampler_metrics,
    record_task_metrics,
)
from platoon.train.step.microbatch import MicrobatchFill, Prepared, StepGeometry, zero_advantage_filter_metrics
from platoon.train.step.publish import _no_update_reason, publish_policy
from platoon.train.step.staleness import StaleFilter, StaleWatch
from platoon.train.tinker.rollout_metrics import monitor as client_monitor
from platoon.train.tinker.rollout_metrics import step_metrics as client_step_metrics
from platoon.train.trainer.run import StepClock
from platoon.train.trainer.signals import TerminateTrainLoop, Watchdog
from platoon.train.trainer.state import TrainLoopSharedState
from platoon.train.workflow.protocol import RolloutWorkflow
from platoon.utils.stats_tracker import StatsTracker

if TYPE_CHECKING:
    from platoon.train.dispatch.dispatcher import Dispatcher
    from platoon.train.policy.checkpoints import Waiter
    from platoon.train.task_records import TaskRecorder
    from platoon.train.trainer.rl import PlatoonTinkerRLTrainer

logger = logging.getLogger(__name__)


async def _train_loop(
    trainer: "PlatoonTinkerRLTrainer",
    task_group: asyncio.TaskGroup,
    shared_state: TrainLoopSharedState,
    start_batch: int,
    end_batch: int,
    train_workflow: RolloutWorkflow,
):
    """Train this start's steps, ``start_batch`` to ``end_batch``: start the dispatcher of its training groups in
    ``task_group``, publish the first sampler, then run one :class:`TrainStep` per step; then let the evaluation loop
    handle the last step, and stop the loops (``shutdown_loops``, which ends this one with ``TerminateTrainLoop``)."""

    column = label_column(trainer.config)
    # Every task the run draws from, a weighted order's tasks that its first epoch leaves out included.
    tasks = trainer._training_tasks()
    group_limits = getattr(trainer.config.train, "rollout_group_limits", {})
    # The groups settle into the result queue, which the steps dequeue from: the dispatcher starts each group when
    # the horizon, the rollout permits and the label limits allow (platoon.train.dispatch.dispatcher).
    task_rollout_result_queue = asyncio.Queue[_Settled]()
    dispatcher = trainer._dispatch(
        task_group, shared_state, start_batch, end_batch, train_workflow, task_rollout_result_queue
    )

    # Publish the initial sampling-client binding before any group can
    # start, then declare the concurrency ceiling the metrics compare
    # against: max_concurrent_rollouts is a budget of rollouts, of which
    # a group takes its size in permits, and the monitor warns once when
    # more rollouts run at once.
    trainer._publish_worker_binding(
        train_workflow,
        shared_state.sampling_client,
        shared_state.model_info.llm.version,
        required=True,
    )
    budget = trainer.config.train.max_concurrent_rollouts
    client_monitor("train").set_configured_concurrency(
        _rollouts_at_once(trainer.config, "train", budget), budget=budget
    )

    # Start training loop.
    shared_state.train_step = start_batch
    logger.info(f"Starting training from batch {start_batch} to {end_batch}")
    context = StepContext.build(
        trainer,
        shared_state,
        train_workflow,
        start_batch,
        end_batch,
        queue=task_rollout_result_queue,
        dispatcher=dispatcher,
        tasks=tasks,
        column=column,
        group_limits=group_limits,
    )
    totals = RunTotals()
    while shared_state.train_step < end_batch:
        await TrainStep(context, totals).run()

    await trainer._wait_for_last_step_evaluation(shared_state)
    await trainer.shutdown_loops(shared_state)


async def shutdown_loops(trainer: "PlatoonTinkerRLTrainer", shared_state: TrainLoopSharedState):
    shared_state.shutdown_event.set()
    shared_state.sampling_client_updated_event.set()
    raise TerminateTrainLoop()


@dataclass(frozen=True)
class StepContext:
    """What the steps of a start share, set up before the first step:

    - ``trainer``, whose seams the steps call, and ``state``, what the loops share;
    - ``workflow``, the training workflow, and ``queue``, the queue its groups settle into;
    - ``dispatch``: the start's draws and the dispatcher that starts their groups.  The steps tell it what they
      dequeued and committed;
    - ``start_batch`` and ``end_batch``: the start's first batch and its end;
    - ``column``, ``labels`` and ``prefixes``: the labels the metrics break down by (``column``'s values) and each
      one's metric prefix; ``group_limits``: the labels' limits;
    - ``recorder`` and ``admissions``: the run's per-task statistics and its group admission rules, when it keeps
      them;
    - ``stale_filter``: each entry's bound on a group's staleness;
    - ``learner``: what updates the policy the steps train, under its objective;
    - ``clock``: the run's clock, which the end of each step advances.

    Two of them count as the steps go.  ``censoring`` counts what the straggler policy censors in a step, until the
    step's metrics are logged.  ``stale_watch`` counts the stale groups of the last steps, for the staleness warning."""

    trainer: "PlatoonTinkerRLTrainer"
    state: TrainLoopSharedState
    workflow: RolloutWorkflow
    queue: "asyncio.Queue[_Settled]"
    dispatch: StepDispatch
    start_batch: int
    end_batch: int
    column: str
    labels: Sequence[str]
    prefixes: dict[str, str]
    group_limits: Mapping[str, int]
    recorder: "TaskRecorder | None"
    admissions: AdmissionSet | None
    censoring: Censoring
    stale_filter: StaleFilter
    stale_watch: StaleWatch
    learner: Learner
    clock: "StepClock | _StateClock"

    @classmethod
    def build(
        cls,
        trainer: "PlatoonTinkerRLTrainer",
        shared_state: TrainLoopSharedState,
        train_workflow: RolloutWorkflow,
        start_batch: int,
        end_batch: int,
        *,
        queue: "asyncio.Queue[_Settled]",
        dispatcher: "Dispatcher | None",
        tasks: Any,
        column: str,
        group_limits: Mapping[str, int],
    ) -> "StepContext":
        """The context of a start's steps, once its dispatcher started (``dispatcher``; None when a test replaced the
        trainer's ``_dispatch``), over every task the run draws from (``tasks``)."""

        # Each step's metrics break down by these labels of the training tasks (train.label_by), also at zero: those of
        # every task the run draws from, a weighted order's tasks that its first epoch leaves out included.
        labels = metric_labels(tasks, column)
        recorder = trainer._recorder
        run_bound = trainer.config.train.max_staleness
        settings = entry_settings.run_settings(trainer.config)
        # The leader of the loops' policies; in a state a test made as platoon 0.4.0 did, the policy it holds.
        policy = _test_state_policy(shared_state, trainer.config) or _trained_policy(shared_state)
        return cls(
            trainer=trainer,
            state=shared_state,
            workflow=train_workflow,
            queue=queue,
            dispatch=StepDispatch(trainer._rollouts, dispatcher),
            start_batch=start_batch,
            end_batch=end_batch,
            column=column,
            labels=labels,
            prefixes={label: metric_prefix(column, label) for label in labels},
            group_limits=group_limits,
            recorder=recorder,
            # A run under a group admission rule (RFC 0019, section 6): each step trains the first batch_size admitted
            # groups it dequeues that are not stale, and closes once it has counted its cap (platoon.train.admissions).
            admissions=trainer._admissions,
            # What the straggler policy censors of the groups each step dequeues, and the warning of stale groups.
            censoring=Censoring(
                settings,
                trainer.config.train.workflow_config,
                threshold=recorder.threshold if recorder is not None else None,
            ),
            # The stale filter's bound for each entry's groups: its own max_staleness, else the run's.
            stale_filter=StaleFilter(run_bound, {label: own.max_staleness for label, own in settings.items()}),
            stale_watch=StaleWatch(run_bound),
            # What updates the policy once per minibatch, under its objective, each result under its train.* deadline.
            learner=Learner(
                policy,
                objective=_objective(policy.spec),
                timeouts=trainer.config.train,
                wait=_update_waiter(trainer, shared_state.watchdog),
                tracker=shared_state.train_tracker,
            ),
            clock=_step_clock(shared_state),
        )

    def version(self) -> int:
        """The policy's version: the one the steps' groups are sampled from, which each publication counts up."""

        return self.state.model_info.llm.version


class TrainStep:
    """One training step: ``batch_size`` settled groups, in ``num_minibatches`` optimizer updates of
    ``num_microbatches`` microbatches each; then the commit, the new policy and the step's metrics."""

    def __init__(self, context: StepContext, totals: RunTotals) -> None:
        self.context = context
        self.totals = totals
        self.state = context.state
        self.started = time.perf_counter()
        self.geometry = StepGeometry.of(context.trainer.config)
        self.tally = StepTally.empty(context.labels)
        self.counting = GroupCounting(context.admissions, context.dispatch, self.tally, self.state.train_step)

    async def run(self) -> None:
        """The step, top to bottom: collect each microbatch, prepare it and send it, and update the policy once per
        minibatch; then commit, publish the policy, move on to the next step and log."""

        # One optimizer update per minibatch.
        for minibatch_num in range(self.geometry.num_minibatches):
            update = self.context.learner.update()
            # Its microbatches accumulate gradients, each sent as soon as its groups are collected, so that training
            # overlaps the rollouts of the same batch (besides the overlap across batches, train.max_staleness).
            for microbatch_num in range(self.geometry.num_microbatches):
                if self.counting.closed:  # a step under a rule that counted its cap dequeues no more
                    break
                fill = await self.collect_microbatch(minibatch_num, microbatch_num)
                if not fill.datums and self.counting.closed:
                    break  # a step under a rule closed on its cap with nothing more to send
                prepared = self.prepare(fill, minibatch_num, microbatch_num)
                if prepared is not None:
                    await update.submit(
                        prepared.datums, prepared.originals, minibatch=minibatch_num, microbatch=microbatch_num
                    )

            if not update.submitted:
                logger.warning(
                    f"Minibatch {minibatch_num} of batch {self.state.train_step} takes no optimizer update: "
                    "none of its microbatches had a nonzero objective"
                )
                continue

            await self.before_first_update()
            await update.apply(self.tally)

        record_workload_metrics(self.tally, self.state.train_tracker)
        self.counting.step_closed()
        self.commit()
        await self.publish()
        self.advance()
        self.log()

    # --- collect: settled groups into a microbatch -------------------------------------------------------------------

    async def collect_microbatch(self, minibatch_num: int, microbatch_num: int) -> MicrobatchFill:
        """Dequeue settled groups into a microbatch until it has its groups, or, under a rule, the step closed."""

        fill = MicrobatchFill()
        tasks_per_microbatch = self.geometry.tasks_per_microbatch
        while fill.places < tasks_per_microbatch and not self.counting.closed:
            # In all mode each item takes a place of the microbatch, counted first so that nothing in the body can
            # skip it; under a rule only an admitted group that is not stale does (at the end).
            place = fill.places
            if self.counting.counts_every_group:
                fill.places += 1
            queue_wait_start = time.perf_counter()
            logger.debug(
                f"Waiting for rollout {place + 1}/{tasks_per_microbatch} "
                f"(minibatch {minibatch_num}, microbatch {microbatch_num})"
            )
            settled = _settled(
                await self.context.trainer._next_rollout(
                    self.context.queue,
                    getattr(self.state, "shutdown_event", None),  # a state a test made may have none: never interrupted
                    interruptible=self.tally.optimizer_steps == 0,
                    ready=self.context.admissions is not None,
                )
            )
            if self.counting.counts_every_group:
                self.context.dispatch.dequeued()
            taken = self.take_group(settled, fill, queue_wait_start)
            if not self.counting.counts_every_group and taken == "admitted":
                fill.places += 1
        return fill

    def take_group(self, settled: _Settled, fill: MicrobatchFill, queue_wait_start: float) -> str:
        """Take a group the step dequeued into ``fill``: count it, drop it when it is stale, count it under a rule, and
        keep its datums when it trains.  What the step took it as: ``marker`` (of a group a rule decided when it
        settled), ``stale``, ``failed`` or ``admitted``."""

        rollout = settled.rollout
        # The group's label: its row's, else its output's.
        label = settled.label if settled.label is not None else _output_label(rollout)
        # Its rows that the run trains, its one trained policy's: a list of datums is its own rows.  None: it failed.
        rows = None if rollout is None else trained_rows(rollout)
        failed = rows is None
        # The group's own statistics belong to this step, which consumes it, stale or not.
        _record_group_stats(rollout, self.state.train_tracker)
        rollout_stats = self.tally.count_dequeued(settled)
        self._record_queue_wait(queue_wait_start)
        stale = self._filter_stale(settled, rows, label)
        # A marker of a group a rule rejected (or that failed) when it settled has its fate already.
        marker = settled.rejected is not None
        self.context.censoring.add(label, settled.record)
        if not marker:
            self.context.stale_watch.dequeued(stale)
        taken = "marker" if marker else "stale" if stale else "failed" if failed else "admitted"
        self.counting.take(settled, taken, label)
        if not marker:
            self._record_fate(settled, "stale" if stale else "failed" if failed else "consumed")

        if not (stale or failed):  # its statistics count, and its datums train (a marker keeps none)
            if rollout_stats is not None:
                self.tally.merge_stats(rollout_stats, label)
            fill.datums.extend(_step_datums(rows))
            self.tally.total_datums += len(rows)
        return taken

    def _record_queue_wait(self, queue_wait_start: float) -> None:
        """How long the step waited for the group since ``queue_wait_start``, for the rollout client metrics."""

        queue_wait_elapsed = time.perf_counter() - queue_wait_start
        # The trainer's own view of "rollout-bound": time spent
        # blocked on the result queue during this step.
        client_monitor("train").record_queue_wait(queue_wait_elapsed, immediate=queue_wait_elapsed < 0.001)
        if queue_wait_elapsed > 60.0:
            logger.warning(f"Waited {queue_wait_elapsed:.1f}s for rollout from queue (slow)")

    def _filter_stale(self, settled: _Settled, rows: Sequence[tinker.Datum] | None, label: str | None) -> bool:
        """The stale filter: whether the group, of ``label``, whose rows are ``rows`` (None: it failed), is more policy
        versions old than its entry's bound.  A stale group is logged and counted here, and the caller drops its
        datums."""

        entry = settled.entry
        # Filter out stale rollouts, by the bound of the group's entry. TODO: Consider requeuing.
        verdict = self.context.stale_filter.check(entry, rows, self.context.version())
        if not verdict.stale:
            return False
        logger.info(
            f"Stale rollout detected in batch {self.state.train_step}: "
            f"policy version {verdict.now}, "
            f"rollout version {verdict.version}, beyond the bound of "
            f"{verdict.bound} versions"
            + (f" of environments[{entry}]" if entry is not None else "")
            + ". Filtering out."
        )
        self.state.train_tracker.scalar(stale_rollouts=1.0)
        self.tally.count_stale(label)
        return True

    def _record_fate(self, settled: _Settled, fate: str) -> None:
        """The fate of a group the step took (not a marker, whose fate the rule gave it) in the per-task statistics,
        which the step commits; and, of a consumed group, its kind and its task sampler's prediction."""

        recorder, record = self.context.recorder, settled.record
        if recorder is None:
            return
        self.tally.record_fate(settled.key, fate)
        if record is not None and fate == "consumed":
            if (kind := record.kind(recorder.threshold(record.entry))) is not None:
                self.tally.count_kind(kind, settled.label)
            where = self.context.dispatch.prefix(record.sampler)
            if record.score is not None and where is not None:
                self.tally.count_prediction(where, record)

    # --- prepare: a microbatch's datums for forward_backward ---------------------------------------------------------

    def prepare(self, fill: MicrobatchFill, minibatch_num: int, microbatch_num: int) -> Prepared | None:
        """A collected microbatch's datums as they train and as they are sent: through the batch transforms and the
        zero-advantage filter, then without the loss inputs forward_backward does not take and with their advantages
        scaled.  None for a microbatch that sends nothing (counted and logged): it had no datums, the transforms or the
        filter left it none, or its objective is zero."""

        if len(fill.datums) == 0:
            logger.warning(f"No rollouts found for microbatch {microbatch_num} (minibatch {minibatch_num})")
            self.tally.empty_microbatches += 1
            return None
        depths_before = summarize_depths(fill.datums)
        task_rollout_results = run_batch_transforms(
            fill.datums, self.context.trainer.batch_transforms, self._transform_context(minibatch_num, microbatch_num)
        )
        self._record_depth_credit(depths_before, task_rollout_results)
        if task_rollout_results is None or len(task_rollout_results) == 0:
            logger.warning(f"All datums were filtered out for microbatch {microbatch_num} (minibatch {minibatch_num})")
            self.tally.empty_microbatches += 1
            return None

        # Depth weighting is part of the objective, so zero-signal
        # datums participate in that normalization before being
        # removed from model compute. Moving this filter into the
        # task workflow would change per-depth gradient weights.
        config = self.context.trainer.config
        objective = self.context.learner.objective  # the loss the microbatch is sent under
        loss_fn, loss_fn_config = objective.loss_fn, objective.loss_fn_config
        if config.train.workflow_config.filter_zero_advantage_datums and objective_is_zero_for_zero_advantages(
            loss_fn, loss_fn_config
        ):
            eligible = get_datum_counts(task_rollout_results)
            task_rollout_results = filter_zero_advantage_datums(task_rollout_results)
            self._after_zero_advantage_filter(eligible, task_rollout_results)

        if len(task_rollout_results) == 0:
            logger.warning(
                f"All datums had zero advantage for microbatch {microbatch_num} (minibatch "
                f"{minibatch_num}): its objective is zero, so it takes no forward/backward"
            )
            self.tally.zero_objective_microbatches += 1
            return None

        # Kept datums or not (filter_zero_advantage_datums), a microbatch whose every advantage is
        # exactly zero has a zero objective under these losses: it is never sent.
        if has_zero_objective(task_rollout_results, loss_fn, loss_fn_config):
            logger.info(
                "Zero policy-gradient objective for microbatch %s (minibatch %s): every advantage is "
                "exactly zero, so it takes no forward/backward and adds nothing to an optimizer step",
                microbatch_num,
                minibatch_num,
            )
            self.tally.zero_objective_microbatches += 1
            return None

        return Prepared(task_rollout_results, self._to_send(task_rollout_results, fill, microbatch_num))

    def _transform_context(self, minibatch_num: int, microbatch_num: int) -> BatchTransformContext:
        """Where the batch transforms find a microbatch: the run's config, and its step, minibatch and microbatch."""

        return BatchTransformContext(
            config=self.context.trainer.config,
            train_step=self.state.train_step,
            minibatch_num=minibatch_num,
            microbatch_num=microbatch_num,
        )

    def _record_depth_credit(self, depths_before: Any, task_rollout_results: list[tinker.Datum] | None) -> None:
        """The batch transforms' depth-credit metrics, when the microbatch's datums carry trajectory depths
        (``depths_before``, as they were before the transforms)."""

        if depths_before is not None:
            self.state.train_tracker.scalar(
                **depth_credit_metrics(depths_before, summarize_depths(task_rollout_results or []))
            )

    def _after_zero_advantage_filter(
        self, eligible: tuple[int, int, int], task_rollout_results: list[tinker.Datum]
    ) -> None:
        """After the zero-advantage filter dropped the datums whose every action token has a zero advantage: its
        metrics, from the counts of the datums it had (``eligible``) and of those it kept; and the datums it kept
        carry the action tokens of those it dropped."""

        self.state.train_tracker.scalar(
            **zero_advantage_filter_metrics(eligible, get_datum_counts(task_rollout_results))
        )
        if task_rollout_results:
            # Preserve the pre-filter action-token denominator;
            # otherwise removing zero terms would amplify every
            # remaining gradient in this microbatch.
            _, _, eligible_loss_tokens = eligible
            set_loss_normalization_token_counts(
                task_rollout_results,
                represented_loss_tokens=eligible_loss_tokens,
            )

    def _to_send(
        self, task_rollout_results: list[tinker.Datum], fill: MicrobatchFill, microbatch_num: int
    ) -> list[tinker.Datum]:
        """The datums a microbatch sends to forward_backward, counted: copies without the loss inputs that only the
        trainer reads (the objective's side channels), their advantages scaled by the action tokens the microbatch
        represents."""

        self.tally.count_submitted(*get_datum_counts(task_rollout_results))

        # Keep original datums for metrics computation (they have mask and logprobs).
        side_channels = self.context.learner.objective.side_channels
        filtered_datums = [
            tinker.Datum(
                model_input=datum.model_input,
                loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k not in side_channels},
            )
            for datum in task_rollout_results
        ]

        # Normalize by represented action-token mass so Tinker's
        # sum-reduced objective behaves like a mean reduction and
        # grad_norm is not sensitive to batch size.
        weight = self.counting.short_microbatch_weight(microbatch_num, fill.places, self.geometry.tasks_per_microbatch)
        scale_advantages(filtered_datums, advantage_scale(task_rollout_results, weight))
        return filtered_datums

    # --- update, commit, publish, advance ----------------------------------------------------------------------------

    @staticmethod
    def _training_metrics(**inputs: Any) -> dict[str, float]:
        """A microbatch's training metrics, for the objective: ``compute_training_metrics`` as this module names it at
        each call, where tests patch it."""

        return compute_training_metrics(**inputs)

    async def before_first_update(self) -> None:
        """Before the run's first update: an evaluation of its starting policy (eval.at_start) must sample that
        policy throughout."""

        if self.context.start_batch == 0 and self.state.optimizer_steps == 0:
            await self.context.trainer._wait_for_starting_policy_evaluation(self.state)

    def commit(self) -> None:
        """The step commits what became of each of its groups, with the fates it decided, to the start's draws
        (:meth:`~platoon.train.dispatch.source.RolloutSource.committed`), before a full checkpoint saves the statistics
        and keeps the groups that settled and no step consumed."""

        self.context.dispatch.committed(self.tally.keys, self.tally.fates)

    async def publish(self) -> None:
        """Publish the step's policy, or say why the step took no update: consuming an all-zero batch does not change
        the policy, and in particular must not age slower, useful rollouts as stale."""

        state = self.state
        update_start = time.perf_counter()
        if self.tally.optimizer_steps:
            await publish_policy(self.context)
        else:
            logger.info(
                "Batch %s (step %s) takes no optimizer update: %s; the policy stays at version %s",
                state.train_step,
                state.train_step + 1,
                _no_update_reason(
                    self.tally.zero_objective_microbatches,
                    self.tally.empty_microbatches,
                    counted=self.counting.counted_without_admission(),
                ),
                self.context.version(),
            )
        update_elapsed = time.perf_counter() - update_start
        state.train_tracker.scalar(update_weights_time=update_elapsed)

    def advance(self) -> None:
        """The next step starts: the run's clock advances (:meth:`~platoon.train.trainer.run.StepClock.advance`), the
        draws' horizon moves on (:meth:`~platoon.train.dispatch.source.RolloutSource.step_started`), and the evaluation
        loop and the watchdog hear of it.  Evaluation's cadence counts consumed batches, zero-signal ones included,
        whose sampler may be unchanged."""

        self.context.clock.advance(consumed=self.context.dispatch)

    # --- log: the step's metrics, then its line ----------------------------------------------------------------------

    def log(self) -> None:
        """The step's metrics, in the order the tracker takes them; then the logged stats and the step's line."""

        batch_elapsed = time.perf_counter() - self.started
        context, tally, totals, state = self.context, self.tally, self.totals, self.state
        tracker = state.train_tracker
        version = context.version()

        record_batch_stats(tracker, tally, totals, context.column)
        if context.recorder is not None:
            record_task_metrics(tracker, context.recorder, tally, context.prefixes, version)
        dispatcher = context.dispatch.dispatcher
        sampler_metrics = None
        if dispatcher is not None:
            sampler_metrics = record_sampler_metrics(tracker, dispatcher, totals, tally, version)
        self.counting.record_step_metrics(tracker, context.prefixes, sampler_metrics)
        if context.group_limits and dispatcher is not None:
            record_label_limits(tracker, dispatcher, totals, context.labels, context.column)
        record_batch_counts(tracker, tally, state)
        # What held the dispatcher back from starting groups, and what the straggler policy censored.
        blocked = context.dispatch.blocked_shares()
        tracker.scalar(**blocked, **context.censoring.step_metrics(context.prefixes))
        context.stale_watch.step_ended(blocked)
        self._log_stats(batch_elapsed)

    def _log_stats(self, batch_elapsed: float) -> None:
        """Export the step's metrics with the run's progress, the process's counters and the pruner's, log them, and
        say the step's line."""

        context, state = self.context, self.state
        step_stats = state.train_tracker.export(reset=True)
        step_stats["train/step"] = state.train_step
        step_stats["train/epoch"] = state.train_step // state.num_train_batches_per_epoch
        step_stats["progress/done_frac"] = state.train_step / context.end_batch
        optimizer = context.learner.policy.spec.optimizer  # the settings the policy's optimizer steps with
        assert optimizer is not None  # policy_specs gives each policy its own
        step_stats["optim/lr"] = optimizer.learning_rate
        step_stats["timing/batch_total"] = batch_elapsed
        # Non-zero ``ptc_threads_leaked`` means rollout worker threads are gone
        # for good; see ``_ptc_cell_guard_stats``.
        step_stats.update(_ptc_cell_guard_stats("train"))
        # Client-side throughput: concurrency actually achieved, the queue
        # wait the trainer experienced, where an agent step's wall clock
        # went, and whether this process (plus its rollout workers) is CPU
        # saturated.  See ``platoon.train.tinker.rollout_metrics``.
        step_stats.update(client_step_metrics("train"))
        if (pruner := context.trainer._recovery_pruner) is not None:
            # Recovery checkpoint pruning: paths deleted and deletes failed since the last step, and what is kept.
            step_stats.update(pruner.step_metrics())
        state.stats_logger.log(step=state.train_step, stats=step_stats)

        logger.info(
            f"Step {state.train_step}/{context.end_batch} | "
            f"Epoch {state.train_step // state.num_train_batches_per_epoch} | "
            f"Batch time: {batch_elapsed:.1f}s | "
            f"Datums: {self.tally.total_datums}"
        )


def _trained_policy(shared_state: TrainLoopSharedState) -> Policy:
    """The policy the steps train: the leader of the loops' policies."""

    policies = shared_state.policies
    assert policies is not None  # a shared state given none makes its own
    return policies[policies.leader]


def _objective(spec: PolicySpec) -> PolicyGradient:
    """The objective the steps update a policy under: the policy-gradient loss its spec names, with its config, which
    the trainer took from ``train.loss_fn`` and ``train.loss_fn_config`` when it was built
    (:func:`~platoon.train.policy.spec.policy_specs`).  Its training metrics are this module's
    ``compute_training_metrics``, which tests patch here (:meth:`TrainStep._training_metrics`)."""

    assert spec.loss_fn is not None and spec.loss_fn_config is not None  # policy_specs names each policy's
    return PolicyGradient(spec.loss_fn, spec.loss_fn_config, training_metrics=TrainStep._training_metrics)


def _test_state_policy(shared_state: Any, config: Any) -> Any:
    """For a state that a test made as platoon 0.4.0 did: the policy the steps train, given the settings a run's policy
    has in its spec; None for any other state.  It goes with the shared state's 0.4.0 fields, once the tests make their
    states with the run's policies.

    A state made without the run's policies (a ``SimpleNamespace`` with the policy's training client, optimizer
    parameters and optimizer steps) is trained in the policy's place; the policy that a shared state makes from those
    fields has a spec without the settings.  Each setting the spec lacks is the run's as the config has it now:
    ``train.optimizer``, ``train.loss_fn`` and ``train.loss_fn_config``, the loss ``cispo`` with no config where a
    test's config names none (as the step read them before the specs held them)."""

    policies = getattr(shared_state, "policies", None)
    policy = shared_state if policies is None else policies[policies.leader]
    spec: PolicySpec = getattr(policy, "spec", None) or PolicySpec(MAIN, model_name="", renderer_name="")
    settings = (spec.optimizer, spec.loss_fn, spec.loss_fn_config)
    if policies is not None and all(setting is not None for setting in settings):
        return None
    train = config.train
    policy.spec = dataclasses.replace(
        spec,
        optimizer=getattr(train, "optimizer", None) if spec.optimizer is None else spec.optimizer,
        loss_fn=getattr(train, "loss_fn", "cispo") if spec.loss_fn is None else spec.loss_fn,
        loss_fn_config=getattr(train, "loss_fn_config", {}) if spec.loss_fn_config is None else spec.loss_fn_config,
    )
    return policy


def _step_clock(shared_state: Any) -> "StepClock | _StateClock":
    """The clock each step's end advances: the loops' (the run's); for a state that a test made as platoon 0.4.0 did,
    without one, a clock over that state's own fields (:class:`_StateClock`)."""

    clock = getattr(shared_state, "clock", None)
    return clock if clock is not None else _StateClock(shared_state)


class _StateClock:
    """The clock of a state that a test made as platoon 0.4.0 did, without the run's clock (a ``SimpleNamespace`` with
    ``train_step``, ``sampling_client_updated_event`` and ``watchdog``): it advances those fields, in the order
    :meth:`~platoon.train.trainer.run.StepClock.advance` does.  It goes with the shared state's 0.4.0 fields, once the
    tests make their states with a clock."""

    def __init__(self, state: Any) -> None:
        self.state = state

    def advance(self, *, consumed: StepDispatch | None = None) -> None:
        state = self.state
        state.train_step += 1
        if consumed is not None:
            consumed.step_started()
        state.sampling_client_updated_event.set()
        if state.watchdog:
            state.watchdog.heartbeat()


def _update_waiter(trainer: "PlatoonTinkerRLTrainer", watchdog: Watchdog | None) -> "Waiter":
    """How an update's results are awaited: under the trainer's heartbeat, looked up on the trainer at each wait, which
    logs every 60 s while it waits and feeds ``watchdog``.  It passes ``heartbeat_interval=60.0`` explicitly, unlike a
    save's waiter (``checkpoints.saving._feeding``), which takes the default: a test that replaces the heartbeat
    checks the arguments of each call."""

    async def wait(awaitable: Awaitable[Any], *, name: str, total_timeout: float | None = None) -> Any:
        return await trainer._await_with_heartbeat(
            awaitable, name=name, heartbeat_interval=60.0, watchdog=watchdog, total_timeout=total_timeout
        )

    return wait


def record_workload_metrics(tally: StepTally, tracker: StatsTracker) -> None:
    """A step's ``workload/*`` metrics: what generating the groups it dequeued took, stale ones included, and what it
    submitted to forward_backward of their datums."""

    # The guard below keeps the loop's local names, which a test reads in this module's source.
    batch_workload = tally.workload
    batch_total_tasks = tally.total_tasks
    batch_total_tasks_with_workload_metadata = tally.total_tasks_with_workload_metadata
    batch_submitted_training_datums = tally.submitted_training_datums
    exact_training_batch_metrics: dict[str, float] = {}
    # A custom workflow's plain list of datums carries no workload: the step still reports what it submitted, but
    # the non-submitted count is exact only when every group it dequeued carried its workload.
    if batch_total_tasks_with_workload_metadata == batch_total_tasks:
        batch_non_submitted_datums = batch_workload.postmerge_datums - batch_submitted_training_datums
        if batch_non_submitted_datums < 0:
            raise ValueError(
                "Submitted training datums exceed the consumed post-merge datum count: "
                f"submitted={batch_submitted_training_datums}, "
                f"postmerge={batch_workload.postmerge_datums}"
            )
        exact_training_batch_metrics["workload/training_batch/total_non_submitted_datums"] = float(
            batch_non_submitted_datums
        )
    tracker.scalar(
        total_datums_per_batch=tally.total_datums,
        **batch_workload.to_metrics("workload/batch"),
        **{
            "workload/batch/total_tasks": float(batch_total_tasks),
            "workload/batch/total_tasks_with_workload_metadata": float(batch_total_tasks_with_workload_metadata),
            "workload/batch/total_requested_rollouts": float(tally.total_requested_rollouts),
            "workload/batch/total_observed_rollouts": float(tally.total_observed_rollouts),
            "workload/batch/total_workflow_trainable_rollouts": float(tally.total_workflow_trainable_rollouts),
            "workload/batch/total_task_retained_datums": float(tally.total_task_retained_datums),
            "workload/batch/total_task_workflow_trainable_datums": float(tally.total_task_retained_datums),
            "workload/batch/total_task_workflow_non_trainable_datums": float(
                batch_workload.postmerge_datums - tally.total_task_retained_datums
            ),
            "workload/batch/total_task_filter_dropped_datums": float(
                batch_workload.post_sampling_datums - tally.total_task_retained_datums
            ),
            "workload/training_batch/total_submitted_datums": float(batch_submitted_training_datums),
            "workload/training_batch/total_attention_tokens": float(tally.submitted_training_attention_tokens),
            "workload/training_batch/total_action_tokens": float(tally.submitted_training_loss_tokens),
            **exact_training_batch_metrics,
        },
    )
