"""The rl trainer's steps when its trained policies step independently (RFC 0021, section 2.11; ``train.stepping:
independent``): one loop per trained policy, each on its own clock (:func:`train_policies`).

Each policy's loop (:class:`PolicyLoop`) runs one :class:`PolicyStep` after the other.  A policy step is the lockstep
step (``TrainStep``, :mod:`platoon.train.trainer.loops`) with the policy's one lane, which takes the policy's shares of
the groups from the share ledger (:mod:`platoon.train.dispatch.shares`) in place of the result queue: its own batch of
them (``policies.<id>.batch_size``; the leader's ``train.batch_size``), on its own clock.  A share a resume put back is
judged by the age it had when the policy first consumed it.  At its end the step records what it consumed, under the
commit lock; its step hooks run with ``end.policy`` set; and it publishes its policy alone: from a commit when the
policy's own cadence says a full checkpoint is due (:mod:`platoon.train.checkpoints.commits`), else from a
weights-only save.  No policy waits for another's training or saves.  The leader's loop runs to the plan's end; each
other policy's until no share will come to it, its last step perhaps short.

Each step logs one row (:meth:`PolicyStep._log_stats`): the leader's with today's names, the run's metrics
(``train/dispatch/*``, ``train/workload/*``, ``timing/*``) and ``run/retained_groups``, on the run's axis
(``run/step``); each other policy's under ``policies/<id>/``, on its own axis (``policies/<id>/step``).  Every policy's
row adds ``policies/<id>/share_wait_s`` (how long its step waited for shares), ``policies/<id>/retrained_shares`` and
``policies/<id>/peer_lag/<q>`` (how many versions each other policy moved between the sampling of the shares' rows and
their training).

An interrupt stops each loop at its next step boundary: a step that applied an update finishes, as a lockstep step
does, with the shares it gets: a loop the stop ended tells the share ledger
(:meth:`~platoon.train.dispatch.shares.Shares.stopped`), its policy takes no more shares, and once its horizon is full
no group starts again, so a step that waits for more closes short, as at the plan's end.  Then the start's exit save
names every policy's row.
"""

import asyncio
import copy
import dataclasses
import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from platoon.train import entry_settings
from platoon.train.checkpoints.saving import _feeding
from platoon.train.dispatch.permits import _rollouts_at_once
from platoon.train.dispatch.settled import _Settled
from platoon.train.dispatch.shares import PLAN_END, Share
from platoon.train.labels import label_column, metric_labels, metric_prefix
from platoon.train.policy.learner import Learner
from platoon.train.policy.policy import Policy
from platoon.train.policy.spec import PolicyId
from platoon.train.step import publish
from platoon.train.step.collect import StepDispatch
from platoon.train.step.lanes import PolicyLane
from platoon.train.step.metrics import Censoring, RunTotals
from platoon.train.step.microbatch import MicrobatchFill, StepGeometry
from platoon.train.step.staleness import StaleFilter, StaleVerdict, StaleWatch
from platoon.train.tinker.rollout_metrics import monitor as client_monitor
from platoon.train.trainer import loops
from platoon.train.trainer.signals import TrainInterrupted
from platoon.utils.stats_tracker import StatsTracker

if TYPE_CHECKING:
    import tinker

    from platoon.train.trainer.hooks import StepEnd
    from platoon.train.trainer.rl import PlatoonTinkerRLTrainer
    from platoon.train.trainer.state import TrainLoopSharedState

logger = logging.getLogger(__name__)


async def train_policies(
    trainer: "PlatoonTinkerRLTrainer",
    task_group: asyncio.TaskGroup,
    shared_state: "TrainLoopSharedState",
    end_batch: int,
    train_workflow: Any,
) -> None:
    """The train loop of a start whose policies step independently (``_train_loop``'s): start the dispatcher, whose
    groups settle into the share ledger, publish each sampled policy's first sampler, then run each trained policy's
    loop (:class:`PolicyLoop`) at once, each policy handing the samplers it publishes to the rollout workers; once every
    loop ended, let the evaluation loop handle the leader's last step and stop the loops.  ``TrainInterrupted`` when a
    stop request ended some loop."""

    run, source = trainer._run, trainer._rollouts
    assert run is not None and source is not None and source.ledger is not None  # the run opened its draws
    # The groups settle into the share ledger, in place of the result queue a lockstep step dequeues from.
    trainer._dispatch(task_group, shared_state, run.start.step, end_batch, train_workflow, asyncio.Queue())
    loops._publish_first_bindings(trainer, shared_state, train_workflow)
    budget = trainer.config.train.max_concurrent_rollouts
    client_monitor("train").set_configured_concurrency(
        _rollouts_at_once(trainer.config, "train", budget), budget=budget
    )
    policies = shared_state.policies
    assert policies is not None and shared_state.clocks is not None  # the trainer made each policy's clock
    logger.info(
        "Starting training from batch %s to %s, each trained policy on its own clock (%s)",
        run.start.step,
        end_batch,
        ", ".join(f"{policy}: step {clock.step}" for policy, clock in shared_state.clocks.items()),
    )
    column = label_column(trainer.config)
    labels = metric_labels(trainer._training_tasks(), column)
    policy_loops = [
        PolicyLoop(trainer, shared_state, policy, end_batch=end_batch, workflow=train_workflow, labels=labels)
        for policy in policies.trainable()
    ]
    with publish.subscribed_bindings(trainer, policies, train_workflow):
        async with asyncio.TaskGroup() as group:
            for loop in policy_loops:
                group.create_task(loop.run_steps(), name=f"train_loop:{loop.policy.id}")
        if any(loop.interrupted for loop in policy_loops):
            raise TrainInterrupted()
        await trainer._wait_for_last_step_evaluation(shared_state)
        await trainer.shutdown_loops(shared_state)


class _PlanEnded(Exception):
    """A policy's step found the plan's end before it took any share: its loop ends."""


class PolicyLoop:
    """One trained policy's loop (``policy``): its steps' context (:meth:`_context`), its batch, its totals over the
    start, and whether a stop request ended it (``interrupted``).  The leader's loop runs to ``end_batch``; another
    policy's until no share will come to it."""

    def __init__(
        self,
        trainer: "PlatoonTinkerRLTrainer",
        shared_state: "TrainLoopSharedState",
        policy: Policy,
        *,
        end_batch: int,
        workflow: Any,
        labels: Sequence[str],
    ) -> None:
        run, source, policies, clocks = trainer._run, trainer._rollouts, shared_state.policies, shared_state.clocks
        assert run is not None and source is not None and source.ledger is not None  # the run opened its draws
        assert policies is not None and clocks is not None  # the trainer made each policy's clock
        self.trainer, self.run, self.policy = trainer, run, policy
        self.ledger = source.ledger
        self.clock = clocks[policy.id]
        self.leads = policy.id == policies.leader
        self.end_batch = end_batch
        train = trainer.config.train
        self.batch_size = train.batch_size if self.leads or policy.spec.batch_size is None else policy.spec.batch_size
        self.context = self._context(shared_state, workflow, labels)
        self.totals = RunTotals()
        self.interrupted = False

    async def run_steps(self) -> None:
        """The policy's steps, one after the other, until the leader's plan ends (the leader) or no share will come (any
        other policy); a stop request ends the loop at the next step boundary (``interrupted``)."""

        clock = self.clock
        try:
            while not self.leads or clock.step < self.end_batch:
                await PolicyStep(self.context, self.totals, self).run()
        except _PlanEnded:
            logger.info(f"Policy {self.policy.id} took its last shares at step {clock.step}")
        except TrainInterrupted:
            self.interrupted = True
            self.ledger.stopped(self.policy.id)  # an open step of another policy waits for no share it would hold back

    def _context(self, shared_state: "TrainLoopSharedState", workflow: Any, labels: Sequence[str]) -> loops.StepContext:
        """What the policy's steps share: the leader's are the loops' shared state, its tracker and the run's dispatcher
        and statistics; another policy's a view of the state (:func:`_view`), a tracker of its own and neither; each
        its share queue, its clock, its one lane and its own stale watch."""

        trainer, run, policy = self.trainer, self.run, self.policy
        config, source = trainer.config, trainer._rollouts
        assert source is not None and run.start is not None
        state = shared_state if self.leads else _view(shared_state, policy.id)
        settings = entry_settings.run_settings(config)
        recorder = source.recorder if self.leads else None
        column = label_column(config)
        own = policy.spec.max_staleness
        bound = config.train.max_staleness if self.leads or own is None else own
        begins = run.start.policies[policy.id].step
        return loops.StepContext(
            trainer=trainer,
            state=state,
            workflow=workflow,
            queue=self.ledger.queue(policy.id),
            dispatch=_PolicyDispatch(source, source.dispatcher if self.leads else None),
            start_batch=begins if begins is not None else run.start.step,
            end_batch=self.end_batch,
            column=column,
            labels=labels,
            prefixes={label: metric_prefix(column, label) for label in labels},
            group_limits=(config.train.rollout_group_limits or {}) if self.leads else {},
            recorder=recorder,
            admissions=None,
            censoring=Censoring(
                settings, config.train.workflow_config, threshold=recorder.threshold if recorder is not None else None
            ),
            stale_watch=StaleWatch(bound),
            lanes=(self._lane(state, settings),),
            clock=self.clock,
            run=run,
            hooks=run.hooks,
        )

    def _lane(self, state: "TrainLoopSharedState", settings: Any) -> PolicyLane:
        """The policy's one lane: its learner, under the loss its spec names, on its clock, which names it in its
        errors; its stale filter, its own bound if it has one; its version; and the step's tracker, whose metrics keep
        today's names (another policy's row puts ``policies/<id>/`` in front as it is logged)."""

        trainer, policy = self.trainer, self.policy
        config = trainer.config
        stale_filter = StaleFilter.of_run(config.train, settings)
        learner = Learner(
            policy,
            objective=loops._objective(policy.spec),
            timeouts=config.train,
            wait=loops._update_waiter(trainer, state.watchdog),
            tracker=state.train_tracker,
            fail_closed=trainer.needs(config, trainer.options).fail_closed_updates,
            before_update=self._starting_evaluation(),
            clock=self.clock,
            named=True,
        )
        own = stale_filter if self.leads else stale_filter.for_policy(policy.spec.max_staleness)
        return PolicyLane(learner, own, lambda: policy.version, state.train_tracker, named=True)

    def _starting_evaluation(self) -> Any:
        """What the policy's first update of a run launched at step 0 awaits: the evaluation of the starting policies
        (``eval.at_start``), as the lockstep step's barrier does, through the loops' shared state."""

        trainer, policy, shared_state = self.trainer, self.policy, self.trainer._shared_state
        begins = self.run.start.policies[policy.id].step

        async def before_update() -> None:
            if not begins and policy.optimizer_steps == 0:
                await trainer._wait_for_starting_policy_evaluation(shared_state)

        return before_update

    async def sampler_after_step(self, step: int) -> "tinker.SamplingClient":
        """The sampling client of the policy's weights after its step ``step``: of a commit when its own cadence says
        one is due (:meth:`~platoon.train.trainer.run.TrainerRun.own_step_sampler`, under the trainer's heartbeat),
        whose recovery save goes to the policy's pruner; else of a weights-only save."""

        trainer, policy = self.trainer, self.policy
        logger.info(f"Saving policy {policy.id}'s weights after its step {step}")
        watchdog = self.context.state.watchdog
        sampling_client, kind, saved = await self.run.own_step_sampler(policy.id, wait=_feeding(trainer, watchdog))
        if kind == "recovery" and saved is not None:
            trainer._recovery_saved(saved, policy.version + 1, None if self.leads else policy.id)
        return sampling_client

    def pruner(self) -> Any:
        """The pruner of the policy's recovery checkpoints, if any."""

        return self.trainer._recovery_pruner if self.leads else self.trainer._policy_pruners.get(self.policy.id)


class PolicyStep(loops.TrainStep):
    """One step of one policy on its own clock: the lockstep step with the policy's one lane (see the module
    docstring), whose ``loop`` gives its share queue, its batch and its publication."""

    def __init__(self, context: loops.StepContext, totals: RunTotals, loop: PolicyLoop) -> None:
        super().__init__(context, totals)
        self.loop = loop
        self.geometry = StepGeometry.of(context.trainer.config, batch_size=loop.batch_size)
        # The shares a resume put back that the step took, by draw: the version each is judged at.
        self.judged_at: dict[tuple[int, int], int] = {}
        self.share_wait = 0.0
        # Each other policy's lag on the shares the step took (peer_lag).
        self.lags: dict[PolicyId, list[int]] = {}

    async def collect_microbatch(self, minibatch_num: int, microbatch_num: int) -> MicrobatchFill:
        """Take the policy's shares into a microbatch until it has its groups.  When no share will come, at the plan's
        end, or after a stop request once no group can start (:meth:`~platoon.train.dispatch.shares.Shares.done`), the
        step closes short (its partial last microbatch weighs as a full one's would), and a step that took nothing ends
        the loop."""

        fill, context = MicrobatchFill(), self.context
        while fill.places < self.geometry.tasks_per_microbatch and not self.counting.closed:
            waited = time.perf_counter()
            item = await context.trainer._next_rollout(
                context.queue,
                self.state.shutdown_event,
                interruptible=not any(tally.optimizer_steps for tally in self.tallies.values()),
            )
            self.share_wait += time.perf_counter() - waited
            if item is PLAN_END:
                if not self.tally.total_tasks:
                    raise _PlanEnded()
                self.counting.closed = True
                break
            fill.places += 1
            self._took(item)
            self.take_group(item.group, fill, waited)
        return fill

    def _took(self, share: Share) -> None:
        """A share the step took: the version a share a resume put back is judged at, and each peer's lag on it."""

        if (judged_at := share.judged_at()) is not None and share.group.key is not None:
            self.judged_at[share.group.key] = judged_at
        versions = self.loop.run.policies
        for peer, version in share.peers.items():
            self.lags.setdefault(peer, []).append(versions[peer].version - version)

    def _judge(self, settled: _Settled, lane: PolicyLane, rows: "Sequence[tinker.Datum] | None") -> StaleVerdict:
        """The stale filter's verdict on the share's rows; a share a resume put back is judged at the version its
        recorded age gives, as it was when its policy first consumed it."""

        if settled.key is not None and (now := self.judged_at.get(settled.key)) is not None:
            lane = dataclasses.replace(lane, version=lambda: now)
        return super()._judge(settled, lane, rows)

    def commit(self) -> None:
        """Nothing yet: what the step consumed is recorded under the commit lock, which takes an await, as the step
        publishes (:meth:`publish`)."""

    async def publish(self, end: "StepEnd | None") -> dict[PolicyId, int]:
        """Record what the step consumed (:meth:`_record_consumed`), then publish the policy when the step or a hook
        updated it."""

        await self._record_consumed()
        return await super().publish(end)

    async def _record_consumed(self) -> None:
        """The step's commit: its shares' fates and ages (:meth:`~platoon.train.dispatch.shares.Shares.consumed`),
        recorded under the commit lock, so that a commit captures all of them or none."""

        loop = self.loop
        async with loop.ledger.lock:
            loop.ledger.consumed(loop.policy.id, self.tally.keys, step=self.state.train_step, fates=self.tally.fates)

    async def _publish(self, updated: Sequence[PolicyLane]) -> None:
        """Publish the policy's weights after the step, from a commit or a weights-only save
        (:meth:`PolicyLoop.sampler_after_step`)."""

        (lane,) = updated
        sampling_client = await self.loop.sampler_after_step(self.state.train_step)
        publish._publish(self.context, lane, sampling_client)

    def _log_stats(self, batch_elapsed: float) -> None:
        """The step's row (see the module docstring), logged without a W&B step of its own, and its line."""

        context, state, loop = self.context, self.state, self.loop
        policy, step = loop.policy, state.train_step
        stats = state.train_tracker.export(reset=True)
        stats["train/step"] = step
        assert policy.spec.optimizer is not None  # policy_specs gives each trained policy its own
        stats["optim/lr"] = policy.spec.optimizer.learning_rate
        stats["timing/batch_total"] = batch_elapsed
        if (pruner := loop.pruner()) is not None:
            stats.update(pruner.step_metrics())
        own = {f"policies/{policy.id}/{name}": value for name, value in self._own_metrics().items()}
        if loop.leads:
            stats["train/epoch"] = step // state.num_train_batches_per_epoch
            stats["progress/done_frac"] = step / context.end_batch
            stats.update(loops._ptc_cell_guard_stats("train"))
            stats.update(loops.client_step_metrics("train"))
            row = {**stats, **own, "run/retained_groups": float(loop.ledger.retained()), "run/step": step}
            axis = "run/step"
        else:
            row = {**{f"policies/{policy.id}/{name}": value for name, value in stats.items()}, **own}
            axis = f"policies/{policy.id}/step"
            row[axis] = step
        state.stats_logger.log_row(row, axis=axis)
        logger.info(
            f"Step {step}{f'/{context.end_batch}' if loop.leads else ''} of policy {policy.id} | "
            f"Batch time: {batch_elapsed:.1f}s | Datums: {self.tally.total_datums}"
        )

    def _own_metrics(self) -> dict[str, float]:
        """The step's metrics that only a policy on its own clock has: how long it waited for its shares, those a resume
        put back that it trained again, and each peer's mean lag."""

        metrics = {"share_wait_s": self.share_wait, "retrained_shares": float(len(self.judged_at))}
        metrics.update({f"peer_lag/{peer}": sum(lags) / len(lags) for peer, lags in self.lags.items()})
        return metrics


class _PolicyDispatch(StepDispatch):
    """The start's draws as one policy's steps tell them: the ledger counts each share it hands out, and the next step
    of the policy moves its horizon on (:meth:`~platoon.train.dispatch.source.RolloutSource.step_started`).  The
    leader's holds the dispatcher, whose task samplers' and blocked shares' metrics its rows log; another policy's
    none."""

    def dequeued(self) -> None:
        """Nothing: the share ledger counts each share as it hands it out."""

    def step_started(self, policy: PolicyId | None = None) -> None:
        if self.draws is not None:
            self.draws.step_started(policy)


def _view(shared_state: "TrainLoopSharedState", policy: PolicyId) -> "TrainLoopSharedState":
    """A policy's view of the loops' shared state, for its own steps: the same state, whose clock is the policy's own,
    whose tracker is its own, and whose policies are led by the policy
    (:meth:`~platoon.train.policy.set.PolicySet.led_by`).  The lockstep step reads the policy it trains first, its
    model info, clients and counters, as the leader's (``state.policies.leader``), so in this view "the leader" is the
    policy whose steps run; the run's own leader, which carries the plan, stays ``trainer.policies.leader``."""

    assert shared_state.policies is not None and shared_state.clocks is not None  # the trainer made each policy's clock
    view = copy.copy(shared_state)
    view.policies = shared_state.policies.led_by(policy)
    view.clock = shared_state.clocks[policy]
    view.train_tracker = StatsTracker("train")
    return view
