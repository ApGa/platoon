"""One training step of the rl trainer (:class:`TrainStep`), and what the steps of a start share (:class:`StepContext`).

The training loop (:mod:`platoon.train.trainer.loops`) builds the context once and runs one ``TrainStep`` per step.  A
step reads top to bottom (:meth:`TrainStep.run`), in seven stages, whose parts are this package's other modules:

1. collect settled groups into microbatches (``collect``, ``counting``, ``staleness``, ``microbatch``, ``datums``);
2. prepare each microbatch's datums and send them (``batch_transforms``, ``microbatch``, ``datums``);
3. update each trained policy once per minibatch, through its learner (:mod:`platoon.train.policy.learner`), under its
   objective (:mod:`platoon.train.objectives`);
4. commit what became of its groups to the start's draws;
5. end the step (:func:`~platoon.train.trainer.hooks.finish_step`, as every trainer's step ends): the run's step hooks
   drain and update the policies, the step publishes each policy that it or a hook updated once (``publish``), and the
   hooks hear of it; with no hook, the publication alone;
6. advance the run's clock to the next step (the end's last part);
7. log the step's metrics (``metrics``).

In a run that trains several policies in lockstep, each policy's rows of each group go their own lane through those
stages (``lanes``); when they step independently, each policy has a loop of its own, whose steps are this step with the
policy's one lane, taking its shares of the groups (``independent``).
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import tinker

from platoon.train import entry_settings
from platoon.train.admissions import AdmissionSet
from platoon.train.dispatch.settled import SettledGroup, _output_label, _record_group_stats, as_settled_group
from platoon.train.evaluation.loop import _wait_for_starting_policy_evaluation
from platoon.train.labels import metric_labels, metric_prefix
from platoon.train.objectives.api import objective_is_zero_for_zero_advantages
from platoon.train.objectives.policy_gradient import PolicyGradient
from platoon.train.policy.learner import PolicyUpdate
from platoon.train.policy.policy import Policy
from platoon.train.policy.spec import PolicyId, PolicySpec
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
from platoon.train.step.datums import _kept_datums, _step_datums, advantage_scale, policy_rows, scale_advantages
from platoon.train.step.lanes import LaneTally, PolicyLane, policy_lanes
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
from platoon.train.step.publish import _no_update_reason, publish_policies
from platoon.train.step.staleness import StaleFilter, StaleVerdict, StaleWatch
from platoon.train.tinker.rollout_metrics import monitor as client_monitor
from platoon.train.tinker.rollout_metrics import step_metrics as client_step_metrics
from platoon.train.trainer.hooks import StepEnd, finish_step
from platoon.train.trainer.run import StepClock, TrainerRun
from platoon.train.trainer.signals import Watchdog
from platoon.train.trainer.state import TrainLoopSharedState
from platoon.train.trainer.waiting import _next_group
from platoon.train.workflow.protocol import RolloutWorkflow
from platoon.utils.stats_tracker import StatsTracker

if TYPE_CHECKING:
    from platoon.train.dispatch.dispatcher import Dispatcher
    from platoon.train.dispatch.shares import ShareQueue
    from platoon.train.policy.checkpoints import Waiter
    from platoon.train.task_records import TaskRecorder
    from platoon.train.trainer.rl import PlatoonTinkerRLTrainer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepContext:
    """What the steps of a start share, set up before the first step:

    - ``trainer``, whose config, needs, batch transforms and pruners the steps read, and ``state``, what the loops
      share;
    - ``workflow``, the training workflow, and ``queue``, the queue its groups settle into (a policy's share queue,
      when the policies step independently);
    - ``dispatch``: the start's draws and the dispatcher that starts their groups.  The steps tell it what they
      dequeued and committed;
    - ``start_batch`` and ``end_batch``: the start's first batch and its end;
    - ``column``, ``labels`` and ``prefixes``: the labels the metrics break down by (``column``'s values) and each
      one's metric prefix; ``group_limits``: the labels' limits;
    - ``recorder`` and ``admissions``: the run's per-task statistics and its group admission rules, when it keeps
      them;
    - ``lanes``: each trained policy's lane through a step, the leader's first
      (:class:`~platoon.train.step.lanes.PolicyLane`): its learner, under its objective, and its stale filter, its
      entries' bounds on a group's staleness, judged per group or per datum (``train.stale_filter``);
    - ``clock``: the run's clock, which the end of each step advances;
    - ``run``: the start the trainer runs, whose step hooks run at each step's end (``run.hooks``).

    Two of them count as the steps go.  ``censoring`` counts what the straggler policy censors in a step, until the
    step's metrics are logged.  ``stale_watch`` counts the stale groups of the last steps, for the staleness warning."""

    trainer: "PlatoonTinkerRLTrainer"
    state: TrainLoopSharedState
    workflow: RolloutWorkflow
    queue: "asyncio.Queue[SettledGroup] | ShareQueue"
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
    stale_watch: StaleWatch
    lanes: tuple[PolicyLane, ...]
    clock: StepClock
    run: TrainerRun

    @classmethod
    def build(
        cls,
        trainer: "PlatoonTinkerRLTrainer",
        shared_state: TrainLoopSharedState,
        train_workflow: RolloutWorkflow,
        start_batch: int,
        end_batch: int,
        *,
        queue: "asyncio.Queue[SettledGroup]",
        dispatcher: "Dispatcher | None",
        tasks: Any,
        column: str,
        group_limits: Mapping[str, int],
    ) -> "StepContext":
        """The context of a start's steps, once its dispatcher started (``dispatcher``; None when a test replaced the
        trainer's ``_dispatch``), over every task the run draws from (``tasks``)."""

        # Each step's metrics break down by these labels (train.label_by) of every task the run draws from, also at 0.
        labels = metric_labels(tasks, column)
        # The start's draws, with the run's per-task statistics and its group admission rules (None in a test's loop).
        draws = trainer._rollouts
        recorder = draws.recorder if draws is not None else None
        settings = entry_settings.run_settings(trainer.config)
        clock, run = shared_state.clock, trainer._run
        assert run is not None  # train() and run() record the start they run, whose steps these are
        return cls(
            trainer=trainer,
            state=shared_state,
            workflow=train_workflow,
            queue=queue,
            dispatch=StepDispatch(draws, dispatcher),
            start_batch=start_batch,
            end_batch=end_batch,
            column=column,
            labels=labels,
            prefixes={label: metric_prefix(column, label) for label in labels},
            group_limits=group_limits,
            recorder=recorder,
            # A run under a group admission rule (RFC 0019, section 6): each step trains the first batch_size admitted
            # groups it dequeues that are not stale, and closes once it has counted its cap (platoon.train.admissions).
            admissions=draws.admissions if draws is not None else None,
            # What the straggler policy censors of the groups each step dequeues, and the warning of stale groups.
            censoring=Censoring(
                settings,
                trainer.config.train.workflow_config,
                threshold=recorder.threshold if recorder is not None else None,
            ),
            stale_watch=StaleWatch(trainer.config.train.max_staleness),
            lanes=_step_lanes(trainer, shared_state, settings, clock, start_batch),
            clock=clock,
            run=run,
        )

    def version(self) -> int:
        """The leader's version: the one the steps' groups are sampled from, which each publication counts up."""

        return self.state.model_info.llm.version


class TrainStep:
    """One training step: ``batch_size`` settled groups, in ``num_minibatches`` optimizer updates of each trained
    policy, of ``num_microbatches`` microbatches each; then the commit, the policies it updated and the step's
    metrics."""

    def __init__(self, context: StepContext, totals: RunTotals) -> None:
        self.context = context
        self.totals = totals
        self.state = context.state
        self.started = time.perf_counter()
        self.geometry = StepGeometry.of(context.trainer.config)
        self.tally = StepTally.empty(context.labels)
        self.counting = GroupCounting(context.admissions, context.dispatch, self.tally, self.state.train_step)
        # What the step counts of each lane: the leader's counts are the step's own, each other lane's its own.
        self.tallies = {lane.policy_id: self.tally if lane.leads else LaneTally() for lane in context.lanes}
        # The step hooks' updates of each policy at the step's end, as platoon counted them (none without hooks).
        self.hook_updates: dict[PolicyId, int] = {}

    def tally_of(self, lane: PolicyLane) -> "StepTally | LaneTally":
        """What the step counts of ``lane``: its optimizer steps, and its microbatches that sent nothing."""

        return self.tallies[lane.policy_id]

    async def run(self) -> None:
        """The step, top to bottom: collect each microbatch, prepare each lane's part of it and send it, and update each
        trained policy once per minibatch; then commit, end the step (the step hooks, the publication of each policy
        that it or a hook updated, the next step) and log.  The end advances the run's clock, which moves the draws'
        horizon on and wakes the evaluation loop, also after a step that published nothing: evaluation's cadence counts
        consumed batches, zero-signal ones included."""

        # One optimizer update per minibatch.
        for minibatch_num in range(self.geometry.num_minibatches):
            updates = {lane.policy_id: lane.learner.update() for lane in self.context.lanes}
            # Its microbatches accumulate gradients, each sent as soon as its groups are collected, so that training
            # overlaps the rollouts of the same batch (besides the overlap across batches, train.max_staleness).
            for microbatch_num in range(self.geometry.num_microbatches):
                if self.counting.closed:  # a step under a rule that counted its cap dequeues no more
                    break
                fill = await self.collect_microbatch(minibatch_num, microbatch_num)
                if fill.empty() and self.counting.closed:
                    break  # a step under a rule closed on its cap with nothing more to send
                for lane in self.context.lanes:  # each trained policy's datums go to its own service
                    prepared = self.prepare(fill, minibatch_num, microbatch_num, lane)
                    if prepared is not None:
                        await updates[lane.policy_id].submit(
                            prepared.datums, prepared.originals, minibatch=minibatch_num, microbatch=microbatch_num
                        )
            await self.update(updates, minibatch_num)

        record_workload_metrics(self.tally, self.state.train_tracker)
        self.counting.step_closed()
        self.commit()
        context = self.context
        hooks, clock = context.run.hooks, context.clock
        await finish_step(hooks, self.step_end, clock=clock, source=context.dispatch, publish=self.publish)
        self.log()

    # --- collect: settled groups into a microbatch -------------------------------------------------------------------

    async def collect_microbatch(self, minibatch_num: int, microbatch_num: int) -> MicrobatchFill:
        """Dequeue settled groups into a microbatch until it has its groups, or, under a rule, the step closed."""

        fill = MicrobatchFill()
        groups_per_microbatch = self.geometry.groups_per_microbatch
        while fill.places < groups_per_microbatch and not self.counting.closed:
            # In all mode each item takes a place of the microbatch, counted first so that nothing in the body can
            # skip it; under a rule only an admitted group that is not stale does (at the end).
            place = fill.places
            if self.counting.counts_every_group:
                fill.places += 1
            queue_wait_start = time.perf_counter()
            logger.debug(
                f"Waiting for rollout {place + 1}/{groups_per_microbatch} "
                f"(minibatch {minibatch_num}, microbatch {microbatch_num})"
            )
            settled = as_settled_group(
                await _next_group(
                    self.context.queue,
                    self.state.shutdown_event,
                    interruptible=not any(tally.optimizer_steps for tally in self.tallies.values()),
                    ready=self.context.admissions is not None,
                )
            )
            if self.counting.counts_every_group:
                self.context.dispatch.dequeued()
            taken = self.take_group(settled, fill, queue_wait_start)
            if not self.counting.counts_every_group and taken == "admitted":
                fill.places += 1
        return fill

    def take_group(self, settled: SettledGroup, fill: MicrobatchFill, queue_wait_start: float) -> str:
        """Take a group the step dequeued into ``fill``: count it, drop each trained policy's rows of it that are stale
        (the whole group when all its rows are), count it under a rule, and keep each policy's rows that train (under
        the per-datum stale filter, those within the bound).  What the step took it as: ``marker`` (of a group a rule
        decided when it settled), ``stale``, ``failed`` or ``admitted``."""

        output = settled.output
        # The group's label: its row's, else its output's.
        label = settled.label if settled.label is not None else _output_label(output)
        # Its rows that the run trains, by trained policy: a list of datums is the leader's rows.  None: it failed.
        rows = None if output is None else policy_rows(output, [lane.policy_id for lane in self.context.lanes])
        failed = rows is None
        # The group's own statistics belong to this step, which consumes it, stale or not.
        _record_group_stats(output, self.state.train_tracker)
        rollout_stats = self.tally.count_dequeued(settled)
        self._record_queue_wait(queue_wait_start)
        verdicts = self._filter_stale(settled, rows)
        # Stale when every policy's rows it has are: a policy's stale rows alone are dropped, and the others train.
        judged = [verdict for policy, verdict in verdicts.items() if rows is not None and rows[policy]]
        stale = bool(judged) and all(verdict.stale for verdict in judged)
        if stale:
            self.tally.count_stale(label)
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
            assert rows is not None  # not failed
            for lane in self.context.lanes:
                if not verdicts[lane.policy_id].stale:
                    self._keep_rows(lane, rows[lane.policy_id], verdicts[lane.policy_id], fill)
        return taken

    def _keep_rows(
        self, lane: PolicyLane, rows: Sequence[tinker.Datum], verdict: StaleVerdict, fill: MicrobatchFill
    ) -> None:
        """Add a lane's rows of a group that trains to the microbatch, as the step trains them (each its own copy),
        without the datums the per-datum stale filter masked."""

        datums = _step_datums(rows)
        if verdict.masked:  # under the per-datum filter, without those beyond the bound
            datums = _kept_datums(datums, verdict.masked)
        lane.add_rows(fill, datums)
        self.tally.total_datums += len(datums)

    def _record_queue_wait(self, queue_wait_start: float) -> None:
        """How long the step waited for the group since ``queue_wait_start``, for the rollout client metrics."""

        queue_wait_elapsed = time.perf_counter() - queue_wait_start
        # The trainer's own view of "rollout-bound": time spent
        # blocked on the result queue during this step.
        client_monitor("train").record_queue_wait(queue_wait_elapsed, immediate=queue_wait_elapsed < 0.001)
        if queue_wait_elapsed > 60.0:
            logger.warning(f"Waited {queue_wait_elapsed:.1f}s for rollout from queue (slow)")

    def _filter_stale(
        self, settled: SettledGroup, rows: Mapping[str, Sequence[tinker.Datum]] | None
    ) -> dict[str, StaleVerdict]:
        """The stale filter's verdict on each trained policy's rows of the group (``rows``, by policy; None: it failed,
        which only the leader's filter is asked of): whether they are more of their policy's versions old than its
        bound, and under the per-datum filter, which of its datums are.  Stale rows, and the datums masked of rows that
        are not, are logged and counted here, and the caller drops them."""

        lanes = self.context.lanes if rows is not None else self.context.lanes[:1]
        return {
            lane.policy_id: self._judge(settled, lane, rows[lane.policy_id] if rows is not None else None)
            for lane in lanes
        }

    def _judge(self, settled: SettledGroup, lane: PolicyLane, rows: Sequence[tinker.Datum] | None) -> StaleVerdict:
        """The stale filter's verdict on a lane's rows of the group (:meth:`_filter_stale`), at its policy's version."""

        entry = settled.entry
        # Filter out stale rollouts, by the bound of the group's entry. TODO: Consider requeuing.
        verdict = lane.stale_filter.check(entry, rows, lane.version())
        if verdict.masked:
            self._count_masked(verdict, rows, entry, lane)
        if not verdict.stale:
            return verdict
        # Under the per-datum filter, every datum is beyond the bound, and the version is the oldest datum's.
        version = "oldest datum version" if lane.stale_filter.per_datum else "rollout version"
        logger.info(
            f"Stale rollout detected in batch {self.state.train_step}{lane.policy_suffix}: "
            f"policy version {verdict.now}, "
            f"{version} {verdict.version}, beyond the bound of "
            f"{verdict.bound} versions"
            + (f" of environments[{entry}]" if entry is not None else "")
            + ". Filtering out."
        )
        lane.tracker.scalar(stale_rollouts=1.0)
        if lane.stale_filter.per_datum and rows is not None:
            self.tally.stale_datums += len(rows)
        return verdict

    def _count_masked(
        self, verdict: StaleVerdict, rows: Sequence[tinker.Datum] | None, entry: str | None, lane: PolicyLane
    ) -> None:
        """Log and count the datums the per-datum stale filter masked of a lane's rows that still train."""

        logger.info(
            f"Stale datums masked in batch {self.state.train_step}{lane.policy_suffix}: {len(verdict.masked)} of the "
            f"group's {len(rows or ())}, the oldest at policy version {verdict.version}, beyond the bound of "
            f"{verdict.bound} versions" + (f" of environments[{entry}]" if entry is not None else "") + f" at version "
            f"{verdict.now}"
        )
        self.tally.stale_datums += len(verdict.masked)

    def _record_fate(self, settled: SettledGroup, fate: str) -> None:
        """The fate of a group the step took (not a marker, whose fate the rule gave it), which the step commits to
        the per-task statistics and its end names; and, of a consumed group, its kind and its task sampler's
        prediction."""

        self.tally.record_fate(settled.key, fate)  # the step's end names it (StepEnd.consumed)
        recorder, record = self.context.recorder, settled.record
        if recorder is None:
            return
        if record is not None and fate == "consumed":
            if (kind := record.kind(recorder.threshold(record.entry))) is not None:
                self.tally.count_kind(kind, settled.label)
            where = self.context.dispatch.prefix(record.sampler)
            if record.score is not None and where is not None:
                self.tally.count_prediction(where, record)

    # --- prepare: a microbatch's datums for forward_backward ---------------------------------------------------------

    def prepare(
        self, fill: MicrobatchFill, minibatch_num: int, microbatch_num: int, lane: PolicyLane
    ) -> Prepared | None:
        """A collected microbatch's datums of a lane as they train and as they are sent: through the batch transforms
        and the zero-advantage filter, then without the loss inputs forward_backward does not take and with their
        advantages scaled.  None for a microbatch that sends nothing of the lane (counted and logged): it had no datums,
        the transforms or the filter left it none, or its objective is zero."""

        tally, datums = self.tally_of(lane), lane.rows_in(fill)
        if len(datums) == 0:
            _nothing_sent(lane)(
                f"No rollouts found for microbatch {microbatch_num} (minibatch {minibatch_num}){lane.policy_suffix}"
            )
            tally.empty_microbatches += 1
            return None
        depths_before = summarize_depths(datums)
        microbatch_datums = run_batch_transforms(
            datums, self.context.trainer.batch_transforms, self._transform_context(minibatch_num, microbatch_num)
        )
        self._record_depth_credit(depths_before, microbatch_datums, lane.tracker)
        if microbatch_datums is None or len(microbatch_datums) == 0:
            logger.warning(
                f"All datums were filtered out for microbatch {microbatch_num} (minibatch {minibatch_num})"
                f"{lane.policy_suffix}"
            )
            tally.empty_microbatches += 1
            return None

        microbatch_datums = self._with_nonzero_objective(microbatch_datums, lane, minibatch_num, microbatch_num)
        if microbatch_datums is None:
            return None
        return Prepared(microbatch_datums, self._to_send(microbatch_datums, fill, microbatch_num, lane))

    def _with_nonzero_objective(
        self, microbatch_datums: list[tinker.Datum], lane: PolicyLane, minibatch_num: int, microbatch_num: int
    ) -> list[tinker.Datum] | None:
        """A lane's transformed datums of a microbatch under its objective: without those whose every advantage is zero,
        when the workflow config filters them and the objective is zero for them; None for a microbatch whose objective
        is zero (counted and logged), which takes no forward/backward."""

        # Depth weighting is part of the objective, so zero-signal
        # datums participate in that normalization before being
        # removed from model compute. Moving this filter into the
        # task workflow would change per-depth gradient weights.
        config = self.context.trainer.config
        objective = lane.learner.objective  # the loss the microbatch is sent under
        loss_fn, loss_fn_config = objective.loss_fn, objective.loss_fn_config
        if config.train.workflow_config.filter_zero_advantage_datums and objective_is_zero_for_zero_advantages(
            loss_fn, loss_fn_config
        ):
            eligible = get_datum_counts(microbatch_datums)
            microbatch_datums = filter_zero_advantage_datums(microbatch_datums)
            self._after_zero_advantage_filter(eligible, microbatch_datums, lane.tracker)

        if len(microbatch_datums) == 0:
            logger.warning(
                f"All datums had zero advantage for microbatch {microbatch_num} (minibatch "
                f"{minibatch_num}){lane.policy_suffix}: its objective is zero, so it takes no forward/backward"
            )
            self.tally_of(lane).zero_objective_microbatches += 1
            return None

        # Kept datums or not (filter_zero_advantage_datums), a microbatch whose every advantage is
        # exactly zero has a zero objective under these losses: it is never sent.
        if has_zero_objective(microbatch_datums, loss_fn, loss_fn_config):
            logger.info(
                "Zero policy-gradient objective for microbatch %s (minibatch %s)%s: every advantage is "
                "exactly zero, so it takes no forward/backward and adds nothing to an optimizer step",
                microbatch_num,
                minibatch_num,
                lane.policy_suffix,
            )
            self.tally_of(lane).zero_objective_microbatches += 1
            return None
        return microbatch_datums

    def _transform_context(self, minibatch_num: int, microbatch_num: int) -> BatchTransformContext:
        """Where the batch transforms find a microbatch: the run's config, and its step, minibatch and microbatch."""

        return BatchTransformContext(
            config=self.context.trainer.config,
            train_step=self.state.train_step,
            minibatch_num=minibatch_num,
            microbatch_num=microbatch_num,
        )

    @staticmethod
    def _record_depth_credit(depths_before: Any, microbatch_datums: list[tinker.Datum] | None, tracker: Any) -> None:
        """The batch transforms' depth-credit metrics, when the microbatch's datums carry trajectory depths
        (``depths_before``, as they were before the transforms), in its lane's ``tracker``."""

        if depths_before is not None:
            tracker.scalar(**depth_credit_metrics(depths_before, summarize_depths(microbatch_datums or [])))

    @staticmethod
    def _after_zero_advantage_filter(eligible: tuple[int, int, int], kept: list[tinker.Datum], tracker: Any) -> None:
        """After the zero-advantage filter dropped the datums whose every action token has a zero advantage: its
        metrics, in the lane's ``tracker``, from the counts of the datums it had (``eligible``) and of those it kept
        (``kept``); and the datums it kept carry the action tokens of those it dropped."""

        tracker.scalar(**zero_advantage_filter_metrics(eligible, get_datum_counts(kept)))
        if kept:
            # Preserve the pre-filter action-token denominator;
            # otherwise removing zero terms would amplify every
            # remaining gradient in this microbatch.
            _, _, eligible_loss_tokens = eligible
            set_loss_normalization_token_counts(
                kept,
                represented_loss_tokens=eligible_loss_tokens,
            )

    def _to_send(
        self, microbatch_datums: list[tinker.Datum], fill: MicrobatchFill, microbatch_num: int, lane: PolicyLane
    ) -> list[tinker.Datum]:
        """The datums of a lane's microbatch it sends to forward_backward, counted among the step's: copies without the
        loss inputs that only the trainer reads (its objective's side channels), their advantages scaled by the action
        tokens the lane's microbatch represents."""

        self.tally.count_submitted(*get_datum_counts(microbatch_datums))

        # Keep original datums for metrics computation (they have mask and logprobs).
        side_channels = lane.learner.objective.side_channels
        filtered_datums = [
            tinker.Datum(
                model_input=datum.model_input,
                loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k not in side_channels},
            )
            for datum in microbatch_datums
        ]

        # Normalize by represented action-token mass so Tinker's
        # sum-reduced objective behaves like a mean reduction and
        # grad_norm is not sensitive to batch size.
        weight = self.counting.short_microbatch_weight(microbatch_num, fill.places, self.geometry.groups_per_microbatch)
        scale_advantages(filtered_datums, advantage_scale(microbatch_datums, weight))
        return filtered_datums

    # --- update, commit, then the step's end: the step hooks, the publication, the advance ---------------------------

    async def update(self, updates: Mapping[str, PolicyUpdate], minibatch_num: int) -> None:
        """Update each trained policy the minibatch sent microbatches of (``updates``, by policy), after the starting
        policy's evaluation when one is due (the learners' barrier, awaited once): every one's optimizer step is
        submitted first, then each one's results awaited and counted, so that their services work at the same time;
        with one policy, its update's ``apply``, call for call.  A policy that sent none takes no update, logged.  A
        service that fails fails the step (lockstep), naming its policy in a run of several."""

        sent = []
        for lane in self.context.lanes:
            if updates[lane.policy_id].submitted:
                sent.append(lane)
            else:
                _nothing_sent(lane)(
                    f"Minibatch {minibatch_num} of batch {self.state.train_step} takes no optimizer update"
                    f"{lane.policy_suffix}: none of its microbatches had a nonzero objective"
                )
        if not sent:
            return
        if not sent[0].named:  # the run's one trained policy
            await updates[sent[0].policy_id].apply(self.tally)
            return
        if (barrier := sent[0].learner.before_update) is not None:  # every lane's learner shares it
            await barrier()
        for lane in sent:
            await updates[lane.policy_id].submit_optim_step()
        for lane in sent:
            await updates[lane.policy_id].finish(self.tally_of(lane))

    def commit(self) -> None:
        """The step commits what became of each of its groups, with the fates it decided, to the start's draws
        (:meth:`~platoon.train.dispatch.source.RolloutSource.committed`), before a full checkpoint saves the statistics
        and keeps the groups that settled and no step consumed."""

        self.context.dispatch.committed(self.tally.keys, self.tally.fates)

    def step_end(self) -> StepEnd:
        """What the step's end tells the step hooks, when it has any (they run in the step's run): the step, the own
        optimizer updates of each policy it trains, the draws it committed with their fates, and each trained policy's
        learner, its lane's, which a hook's learner of that policy shares the training client, the metrics' prefix and
        the barrier of the starting policy's evaluation with."""

        lanes = self.context.lanes
        learners = {lane.policy_id: lane.learner for lane in lanes}
        return self.context.run._step_end(
            "step",
            step=self.state.train_step,
            step_updates={lane.policy_id: self.tally_of(lane).optimizer_steps for lane in lanes},
            consumed=dict(self.tally.fates),
            learner_of=learners.__getitem__,
            policy=self.context.clock.policy,  # whose step, when the policies step independently
            tracker=self.state.train_tracker,
        )

    async def publish(self, end: StepEnd | None) -> dict[PolicyId, int]:
        """Publish each policy that the step or its hooks (``end``'s, None without hooks) updated, once
        (:func:`~platoon.train.step.publish.publish_policies`), or say why one took no update: consuming an all-zero
        batch does not change the policy, and in particular must not age slower, useful rollouts as stale.  Who
        published, at which version."""

        state = self.state
        update_start = time.perf_counter()
        self.hook_updates = dict(end.hook_updates) if end is not None else {}
        updated = [
            lane
            for lane in self.context.lanes
            if self.tally_of(lane).optimizer_steps or self.hook_updates.get(lane.policy_id)
        ]
        published: dict[PolicyId, int] = {}
        if updated:
            await self._publish(updated)
            if end is not None:  # who published, which the hooks hear
                published = {lane.policy_id: lane.version() for lane in updated}
        for lane in self.context.lanes:
            if all(lane.policy_id != other.policy_id for other in updated):
                tally = self.tally_of(lane)
                logger.info(
                    "Batch %s (step %s) takes no optimizer update%s: %s; the policy stays at version %s",
                    state.train_step,
                    state.train_step + 1,
                    lane.policy_suffix,
                    _no_update_reason(
                        tally.zero_objective_microbatches,
                        tally.empty_microbatches,
                        counted=self.counting.counted_without_admission(),
                    ),
                    lane.version(),
                )
        update_elapsed = time.perf_counter() - update_start
        state.train_tracker.scalar(update_weights_time=update_elapsed)
        return published

    async def _publish(self, updated: Sequence[PolicyLane]) -> None:
        """Save and publish the policies of the ``updated`` lanes
        (:func:`~platoon.train.step.publish.publish_policies`): the cadence's checkpoint of every trained policy when
        one is due, else their weights alone."""

        await publish_policies(self.context, updated)

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
        for lane in context.lanes:
            if not lane.leads:  # the leader's counts are among the step's
                hooked = self.hook_updates.get(lane.policy_id, 0) if context.run.hooks else None
                lane.record_counts(self.tally_of(lane), hook_updates=hooked)
        if context.lanes[0].stale_filter.per_datum:  # the datums dropped as stale, of the groups it masked or dropped
            tracker.scalar(**{"batch/stale_datums": float(tally.stale_datums)})
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
        for lane in context.lanes:
            optimizer = lane.policy.spec.optimizer  # the settings the policy's optimizer steps with
            assert optimizer is not None  # policy_specs gives each trained policy its own
            step_stats[f"{lane.prefix}optim/lr"] = optimizer.learning_rate
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
        for policy, pruner in context.trainer._policy_pruners.items():  # each other trained policy's
            step_stats.update({f"policies/{policy}/{name}": value for name, value in pruner.step_metrics().items()})
        state.stats_logger.log(step=state.train_step, stats=step_stats)

        logger.info(
            f"Step {state.train_step}/{context.end_batch} | "
            f"Epoch {state.train_step // state.num_train_batches_per_epoch} | "
            f"Batch time: {batch_elapsed:.1f}s | "
            f"Datums: {self.tally.total_datums}"
        )


def _nothing_sent(lane: PolicyLane) -> Callable[[str], None]:
    """How the step logs that a lane sent nothing: a warning for a run's one trained policy, DEBUG for each of several,
    which often have no rows in a microbatch (an agent role its rollouts did not open)."""

    return logger.debug if lane.named else logger.warning


def _trained_policy(shared_state: TrainLoopSharedState) -> Policy:
    """The policy the steps train: the leader of the loops' policies."""

    policies = shared_state.policies
    return policies[policies.leader]


def _step_lanes(
    trainer: "PlatoonTinkerRLTrainer",
    shared_state: TrainLoopSharedState,
    settings: Any,
    clock: StepClock,
    start_batch: int,
) -> tuple[PolicyLane, ...]:
    """Each trained policy's lane through the start's steps, the leader's first: its learner, which updates its policy
    once per minibatch under the loss its spec names, each result under its ``train.*`` deadline and the trainer's
    heartbeat, the starting policy's evaluation awaited before the run's first update
    (:func:`_starting_policy_barrier`), the step opened on ``clock`` by its first submission, failing closed when the
    trainer's needs say so (a subclass's: the rl trainer's do not); and its stale filter, each entry's groups' bound
    (its own ``max_staleness``, else the run's: ``settings``), per datum or group."""

    return policy_lanes(
        _trained_policy(shared_state),
        _other_trained(shared_state),
        objective=_objective,
        timeouts=trainer.config.train,
        wait=_update_waiter(trainer, shared_state.watchdog),
        tracker=shared_state.train_tracker,
        stale_filter=StaleFilter.of_run(trainer.config.train, settings),
        leader_version=lambda: shared_state.model_info.llm.version,
        fail_closed=trainer.needs(trainer.config, trainer.options).fail_closed_updates,
        before_update=_starting_policy_barrier(shared_state, start_batch),
        clock=clock,
    )


def _starting_policy_barrier(shared_state: TrainLoopSharedState, start_batch: int) -> Callable[[], Awaitable[None]]:
    """What the run's first update of any kind awaits, of any trained policy, the step hooks' too: an evaluation of its
    starting policy (``eval.at_start``) must sample that policy throughout
    (:func:`~platoon.train.evaluation.loop._wait_for_starting_policy_evaluation`)."""

    async def before_update() -> None:
        if start_batch == 0 and shared_state.optimizer_steps == 0:
            await _wait_for_starting_policy_evaluation(shared_state)

    return before_update


def _other_trained(shared_state: TrainLoopSharedState) -> list[Policy]:
    """The policies the steps train besides the leader, in the run's order."""

    policies = shared_state.policies
    return [policy for policy in policies.trainable() if policy.id != policies.leader]


def _objective(spec: PolicySpec) -> PolicyGradient:
    """The objective the steps update a policy under: the policy-gradient loss its spec names, with its config, which
    the trainer took from ``train.loss_fn`` and ``train.loss_fn_config`` when it was built
    (:func:`~platoon.train.policy.spec.policy_specs`)."""

    assert spec.loss_fn is not None and spec.loss_fn_config is not None  # policy_specs names each policy's
    return PolicyGradient(spec.loss_fn, spec.loss_fn_config)


def _update_waiter(trainer: "PlatoonTinkerRLTrainer", watchdog: Watchdog | None) -> "Waiter":
    """How an update's results are awaited: under the trainer's heartbeat, looked up on the trainer at each wait, which
    logs every 60 s while it waits and feeds ``watchdog``.  It passes ``heartbeat_interval=60.0`` explicitly, unlike a
    save's waiter (``checkpoints.saving._feeding``), which takes the default."""

    async def wait(awaitable: Awaitable[Any], *, name: str, total_timeout: float | None = None) -> Any:
        return await trainer._await_with_heartbeat(
            awaitable, name=name, heartbeat_interval=60.0, watchdog=watchdog, total_timeout=total_timeout
        )

    return wait


def record_workload_metrics(tally: StepTally, tracker: StatsTracker) -> None:
    """A step's ``workload/*`` metrics: what generating the groups it dequeued took, stale ones included, and what it
    submitted to forward_backward of their datums."""

    workload = tally.workload
    submitted = tally.submitted_training_datums
    exact_training_batch_metrics: dict[str, float] = {}
    # A custom workflow's plain list of datums carries no workload: the step still reports what it submitted, but
    # the non-submitted count is exact only when every group it dequeued carried its workload.
    if tally.groups_with_workload == tally.groups_dequeued:
        non_submitted = workload.postmerge_datums - submitted
        if non_submitted < 0:
            raise ValueError(
                "Submitted training datums exceed the consumed post-merge datum count: "
                f"submitted={submitted}, "
                f"postmerge={workload.postmerge_datums}"
            )
        exact_training_batch_metrics["workload/training_batch/total_non_submitted_datums"] = float(non_submitted)
    tracker.scalar(
        total_datums_per_batch=tally.total_datums,
        **workload.to_metrics("workload/batch"),
        **{
            "workload/batch/total_tasks": float(tally.groups_dequeued),
            "workload/batch/total_tasks_with_workload_metadata": float(tally.groups_with_workload),
            "workload/batch/total_requested_rollouts": float(tally.total_requested_rollouts),
            "workload/batch/total_observed_rollouts": float(tally.total_observed_rollouts),
            "workload/batch/total_workflow_trainable_rollouts": float(tally.total_workflow_trainable_rollouts),
            "workload/batch/total_task_retained_datums": float(tally.total_task_retained_datums),
            "workload/batch/total_task_workflow_trainable_datums": float(tally.total_task_retained_datums),
            "workload/batch/total_task_workflow_non_trainable_datums": float(
                workload.postmerge_datums - tally.total_task_retained_datums
            ),
            "workload/batch/total_task_filter_dropped_datums": float(
                workload.post_sampling_datums - tally.total_task_retained_datums
            ),
            "workload/training_batch/total_submitted_datums": float(submitted),
            "workload/training_batch/total_attention_tokens": float(tally.submitted_training_attention_tokens),
            "workload/training_batch/total_action_tokens": float(tally.submitted_training_loss_tokens),
            **exact_training_batch_metrics,
        },
    )
