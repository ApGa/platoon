"""One start of one trainer (:class:`TrainerRun`): the lifecycle every trainer goes through, and what the trainer's
hooks use; and the run's step clock (:class:`StepClock`).

:meth:`TrainerRun.execute` reads as the start's phases, in the order they run.  Each phase that is the trainer's own
calls its hook (:class:`~platoon.train.trainer.api.TrainerBase`):

0. the start's workflows, which the trainer's setup builds on its policies, when the trainer rolls out and was given
   none (``platoon train`` builds the rl trainer's itself, and hands them to its ``train()``);
1. the trainer's checks (``check``), where it may register participants of the run's checkpoints and step hooks;
2. the signal handlers: a first signal asks the steps to stop at the next step boundary (``run.shutdown``);
3. each policy's service client, and where the start begins (:class:`~platoon.train.checkpoints.resume.StartPoint`),
   which the participants of the run's checkpoints registered so far take up, before any training client opens;
4. each policy's training client at the start point; a start stopped before the service accepted it ends here;
5. the trainer prepares its steps (``prepare``), and says where they end; a trainer that rolls out opens the start's
   draws there (:meth:`TrainerRun.rollouts`).  Then every key of the checkpoint the start resumes from must be claimed:
   by the run's counters or by a participant (exit 78 otherwise);
6. the launch save, a full checkpoint every start writes, then the draws and the trainer start from it (``started``),
   each trained policy's first sampling client, of the launch save's weights, which the rollout workers of draws the
   run opened are given, with the binding of each frozen policy they sample, once, and the step hooks hear of it;
7. the trainer readies its steps (``ready``), and the run's clock starts;
8. the steps (``run_steps``), each ending with the step hooks (:meth:`TrainerRun.end_step`; the rl trainer's step runs
   the same sequence), after which the clock's watchdog stops, if the steps left it running;
9. the save at exit: ``<step>_interrupted`` after an interrupt (the trainer's, or a stop request its steps stopped
   on short of their end), unless it came inside a step that no checkpoint may save (rule 10): one that sent training
   work, when the trainer's steps are atomic, else one whose end ran its step hooks; ``final`` at the run's end;
10. in ``finally``, also after a failure: the trainer closes what it opened (``close``), then each hook and participant
    (``aclose``), in reverse order of registration; then, after steps that ended and closed cleanly, the draws'
    statistics.  A start that failed raises its error, the close's errors noted on it; else the close's errors.

When the trainer's needs say its trained policies step independently (``TrainerNeeds.stepping``, the rl trainer's
``train.stepping``) and it trains two or more (:attr:`TrainerRun.independent`), each has a clock of its own
(:attr:`TrainerRun.clocks`, :mod:`platoon.train.trainer.clocks`; the leader's is ``clock``): a step's end names whose
step it is (:meth:`TrainerRun.end_step`), only that policy publishes, on its own checkpoint cadence, and every full
save is a commit (:mod:`platoon.train.checkpoints.commits`); the start's launch and exit saves stay barriers of every
policy.  A start keeps the stepping its run's checkpoints were saved with (exit 78 otherwise).
"""

import asyncio
import contextlib
import contextvars
import functools
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol

from platoon.train.checkpoints.barrier import Counters
from platoon.train.checkpoints.commits import COMMIT_SEQ_KEY, Commits
from platoon.train.checkpoints.participants import RunCheckpoints
from platoon.train.checkpoints.pruning import RECOVERY_CHECKPOINT_PREFIX
from platoon.train.checkpoints.resume import PolicyStart, StartPoint
from platoon.train.checkpoints.saving import checkpoint_kind_for_step
from platoon.train.config_defs import event_interval
from platoon.train.dispatch.source import RolloutSource
from platoon.train.objectives.policy_gradient import PolicyGradient
from platoon.train.policy.checkpoints import sampler_from
from platoon.train.policy.learner import Learner, PrefixedTracker
from platoon.train.policy.policy import Policy, adam_params
from platoon.train.policy.spec import PolicyId, stepping_problems
from platoon.train.trainer.api import StepPlan, Workflows
from platoon.train.trainer.clocks import OwnClocks
from platoon.train.trainer.hooks import RunSide, StepEnd, StepHooks, finish_step, hear_launch, is_critical
from platoon.train.trainer.signals import TrainInterrupted, Watchdog
from platoon.utils.config import ConfigError

if TYPE_CHECKING:
    from platoon.train.objectives.api import Objective
    from platoon.train.trainer.api import TrainerBase

logger = logging.getLogger(__name__)

# Whether the task that reads it runs a step's end's hooks, when the run's policies step independently: the run's
# learners send nothing there, whichever policy's clock they step on (TrainerRun._outside_hooks).
_IN_STEP_HOOKS: contextvars.ContextVar[bool] = contextvars.ContextVar("platoon_in_step_hooks", default=False)


class _Consumer(Protocol):
    """What a step consumed its groups from, whose horizon moves on as the next step starts (``policy``: whose, when the
    policies step independently)."""

    def step_started(self, policy: PolicyId | None = None) -> None: ...


class StepClock:
    """The run's step clock: the step training is at (``step``), the step after the run's last (``end``: the steps run
    while ``step < end``; None for a clock outside a run), the event each step's end sets (``updated``: the evaluation
    loop wakes on it), and the run's watchdog, which each step's end feeds (None when the config turns it off).
    ``step_open`` says that the step has sent training work (a learner's first submission sets it, and so do the step
    hooks' updates; a step that sends work by hand sets it itself), until it ends: rule 10 saves no checkpoint of a
    step that is open, when the trainer's steps are atomic.  ``ending`` says that the step's end runs its step hooks
    (:func:`~platoon.train.trainer.hooks.finish_step`), until the clock advances: when the trainer's steps are not
    atomic, rule 10 saves no checkpoint of a step that is ending.  ``open_updates`` holds each policy's open update,
    sent and not counted yet, which a step's end refuses to start its hooks or to publish with.  ``policy``: whose clock
    it is, when the run's policies step independently (None: the run's one clock)."""

    def __init__(
        self,
        *,
        step: int,
        updated: asyncio.Event,
        end: int | None = None,
        watchdog: Watchdog | None = None,
        policy: PolicyId | None = None,
    ) -> None:
        self.step = step
        self.end = end
        self.updated = updated
        self.watchdog = watchdog
        self.policy = policy
        self.step_open = False
        self.ending = False
        self.open_updates: dict[PolicyId, Any] = {}

    @classmethod
    def started(cls, config: Any, *, step: int, end: int | None = None) -> "StepClock":
        """A clock at ``step``, as a start's steps begin: its watchdog started when ``watchdog.enabled`` says so (it
        ends a process that stops making progress), and its event set, so that the evaluation loop's first wait passes
        at once."""

        watchdog = None
        if config.watchdog.enabled:
            watchdog = Watchdog(
                timeout_seconds=config.watchdog.timeout_seconds,
                exit_code=config.watchdog.exit_code,
            )
            watchdog.start()
        updated = asyncio.Event()
        updated.set()
        return cls(step=step, updated=updated, end=end, watchdog=watchdog)

    def advance(self, *, consumed: _Consumer | None = None) -> None:
        """The next step starts: the step counts one more, and is neither open nor ending yet; what the step consumed
        its groups from moves its horizon on (``consumed``; this policy's, of a policy's own clock); the evaluation loop
        hears of it, also after a step that published no new policy; and the watchdog is fed."""

        self.step += 1
        self.step_open = False
        self.ending = False
        if consumed is not None:
            if self.policy is None:
                consumed.step_started()
            else:
                consumed.step_started(self.policy)
        self.updated.set()
        if self.watchdog:
            self.watchdog.heartbeat()


class TrainerRun:
    """One start of ``trainer``, through the lifecycle every trainer shares (:meth:`execute`), with the given
    workflows (None: those the trainer's setup builds, when it rolls out).  What its hooks use: the config, the run
    directory, the policies (by id; ``policies.leader`` is the first one the run trains), the workflows, what the
    trainer needs of the run (``needs``), the run's checkpoints
    (:class:`~platoon.train.checkpoints.participants.RunCheckpoints`) and step hooks
    (:class:`~platoon.train.trainer.hooks.StepHooks`), the shutdown request, where the start begins and where its
    steps end, the start's draws (:meth:`rollouts`), the launch save, the step clock with its watchdog, which every
    trained policy steps on in lockstep, or each its own (:attr:`clocks`, when they step independently), and
    what a step uses: a policy's learner (:meth:`learner`), the waits (:meth:`wait`), and the step's end
    (:meth:`end_step`, which publishes through :meth:`publish`)."""

    # Set as the phases run: the shutdown request (2), the start point (3), the plan (5), the paths of the launch save
    # by policy (6), the clock (7: start_clock, or the trainer's own).
    shutdown: asyncio.Event
    start: StartPoint
    plan: StepPlan
    launch_save: dict[PolicyId, dict[str, str]]
    clock: StepClock

    def __init__(self, trainer: "TrainerBase", *, workflows: Workflows | None = None) -> None:
        self.trainer = trainer
        self.config = trainer.config
        self.run_dir = trainer.run_log_path
        self.policies = trainer.policies
        self.workflows = workflows
        self.needs = trainer.needs(self.config, trainer.options)
        # Whether its trained policies step each on its own clock: the trainer asks for it, and it trains two or more.
        self.independent = self.needs.stepping == "independent" and len(self.policies.trainable()) > 1
        # Every step hook and checkpoint participant, in the order they joined: phase 10 closes them in reverse.
        self._parts: list[Any] = []
        self.checkpoints = RunCheckpoints(
            self.run_dir, self.policies, wandb_run_id=self._wandb_run_id, joined=self._joined
        )
        self.hooks = StepHooks(self.checkpoints, joined=self._joined)
        if self.independent:  # every full save of any policy is a commit, which captures no hook part-way
            self.checkpoints.commits = Commits(
                self.run_dir,
                self.policies,
                participants=self.checkpoints.participants,
                wandb_run_id=self._wandb_run_id,
            )
            self.hooks.running = self._hook_phase
        # Each trained policy's clock when they step independently (clocks_from); None: the run's one clock.
        self._own: OwnClocks | None = None
        # Whether an interrupt stopped the steps (8), and whether the steps ended and the save at exit ran (9).
        self.interrupted = False
        self.ended = False
        # The start's draws, once the trainer opened them (rollouts); None before, and for a trainer that rolls out
        # none.  Draws the run opened (rollouts) have their rollout workers given each sampler the run publishes.
        self.source: RolloutSource | None = None
        self._binds_workers = False
        # Each policy's optimizer steps as the clock's step began (start_clock, then each end_step).
        self._step_began: dict[PolicyId, int] = {}

    def _joined(self, part: Any) -> None:
        """A step hook or a checkpoint participant joined the run: phase 10 closes it (``aclose``), in reverse order of
        joining, once however it joined."""

        if all(other is not part for other in self._parts):
            self._parts.append(part)

    def _wandb_run_id(self) -> str | None:
        stats_logger = self.trainer._stats_logger
        return stats_logger.wandb_run_id if stats_logger is not None else None

    def rollouts(self) -> RolloutSource:
        """The start's training draws (:class:`~platoon.train.dispatch.source.RolloutSource`): opened at the start's
        point over the run's training tasks, ``train.*`` and the training workflow, and registered with the run's
        checkpoints, which hands it the start point, so that it takes up the draws the checkpoint left outstanding.
        Call it in ``prepare()``, once, and start it (``start``) as the steps begin; the run starts its per-task
        statistics after the launch save and closes them after the trainer's ``close``, and gives the training
        workflow's rollout workers each sampler it publishes, the launch's first.  Only a trainer whose ``needs()``
        say it rolls out may call it."""

        trainer, workflows = self.trainer, self.workflows
        if not self.needs.rollouts:
            raise RuntimeError(f"{type(trainer).__name__} rolls out nothing (its needs() say so): it has no draws")
        assert trainer.setup is not None and trainer.setup.tasks is not None and workflows is not None
        source = RolloutSource(
            self.config,
            run_dir=self.run_dir,
            tasks=trainer.setup.tasks,
            workflow=workflows.train,
            versions=self.policies.versions,
            leader=self.policies.leader,
        )
        self._binds_workers = True
        return self._open_draws(source)

    def _open_draws(self, source: RolloutSource) -> RolloutSource:
        """Open ``source`` as the start's draws, at the start's point, and register it with the run's checkpoints, which
        hands it the start point: it takes up the draws the checkpoint left outstanding.  The run starts its per-task
        statistics after the launch save and closes them after the trainer's ``close``.  :meth:`rollouts` opens a
        trainer's draws through it, and the rl trainer its own (a source whose steps go through its methods)."""

        assert self.source is None, "a start opens its draws once"
        # Each training episode sees its draw (ctx.draw) when the trainer asks for it.
        source.draw_provenance = self.needs.draw_provenance
        if self.independent:  # each settled group is split into shares, one per trained policy
            source.policies = self.policies
        source.open(self.start)
        if (commits := self.checkpoints.commits) is not None and source.ledger is not None:
            commits.attach(source.ledger)  # which takes the commit lock
        self.checkpoints.register(source)
        self.source = source
        return source

    def start_clock(self) -> StepClock:
        """Start the run's clock at the start's step, towards the plan's end (:meth:`StepClock.started`); the first
        step's updates count from here (:meth:`end_step`)."""

        self.clock = StepClock.started(self.config, step=self.start.step, end=self.plan.end_step)
        if self.independent:
            self.clocks_from(self.clock)
        self._step_began = self._optimizer_steps()
        return self.clock

    @property
    def clocks(self) -> dict[PolicyId, StepClock]:
        """Each trained policy's step clock, by policy: in lockstep, the run's one clock (``clock``) for every one;
        stepping independently, each its own, the leader's ``clock``."""

        if self._own is not None:
            return dict(self._own)
        return {policy.id: self.clock for policy in self.policies.trainable()}

    def clocks_from(self, clock: StepClock) -> dict[PolicyId, StepClock]:
        """When the trained policies step independently: ``clock``, the run's (the leader's), and a clock of each other
        trained policy at the step its start point gives it (:class:`~platoon.train.trainer.clocks.OwnClocks`).  The
        run's clocks from now on (:attr:`clocks`)."""

        self._own = OwnClocks(
            clock,
            config=self.config,
            policies=self.policies,
            start=self.start,
            steps_per_epoch=self.plan.steps_per_epoch,
        )
        return dict(self._own)

    def _clock_of(self, policy: PolicyId) -> StepClock:
        """The clock ``policy`` steps on: its own when the policies step independently, else the run's."""

        return self._own[policy] if self._own is not None else self.clock

    # --- what a step uses ------------------------------------------------------------------------------------------

    def learner(self, policy: PolicyId, *, objective: "Objective | None" = None, metric_prefix: str = "") -> Learner:
        """The learner of ``policy`` (:class:`~platoon.train.policy.learner.Learner`): its training client and its
        optimizer, which steps with its spec's Adam settings (``train.optimizer``), under ``objective`` (None: the
        policy-gradient loss its spec names, ``train.loss_fn``).  Its results are awaited under the run's waits within
        ``train.*``'s deadlines, its metrics go to the train tracker's step row under ``metric_prefix`` (after
        ``policies/<id>/`` for a trained policy other than the leader, as in the rl trainer's steps), it fails
        closed when the trainer's needs say so (rule 10), and its first submission of a step opens the run clock's
        step: build it once the clock started (``ready()``, ``run_steps()``).  In a run of several trained policies,
        an error of the policy's service in an update names the policy and its service (its ``named``).  It is the
        trainer's: in a run with step hooks it sends and counts nothing as a step's end runs them (``clock.ending``),
        and a hook updates through its step's end instead (``end.train``, ``end.learner``, whose learner of a policy
        names it as this one does).  A frozen policy has none (ValueError)."""

        if getattr(self, "clock", None) is None:
            raise RuntimeError("run.learner() opens the run clock's steps: build it once the clock started (ready())")
        trained = self.policies[policy]
        if not trained.trainable:
            trains = [other.id for other in self.policies.trainable()]
            raise ValueError(f"policy {policy!r} is frozen: it has no learner (the run trains {trains})")
        spec = trained.spec
        if getattr(trained, "optimizer_params", None) is None:
            assert spec.optimizer is not None  # policy_specs gives each trained policy its own, else train.optimizer
            trained.optimizer_params = adam_params(spec.optimizer)
        if objective is None:
            assert spec.loss_fn is not None and spec.loss_fn_config is not None  # policy_specs names each policy's
            objective = PolicyGradient(spec.loss_fn, spec.loss_fn_config)
        tracker = self.trainer._train_tracker
        assert tracker is not None  # the trainer was entered (async with)
        records = tracker if policy == self.policies.leader else PrefixedTracker(tracker, f"policies/{policy}/")
        clock = self._clock_of(policy)  # stepping independently, its own
        guard = None
        if self.hooks:  # the learner sends nothing as its clock's step end runs the hooks
            guard = functools.partial(self._outside_hooks, clock) if self.independent else self._outside_hooks
        return Learner(
            trained,
            objective=objective,
            timeouts=self.config.train,
            wait=self._learner_wait,
            tracker=records,
            metric_prefix=metric_prefix,
            fail_closed=self.needs.fail_closed_updates,
            clock=clock,
            guard=guard,
            named=len(self.policies.trainable()) > 1,
        )

    def _outside_hooks(self, clock: StepClock | None = None) -> None:
        """The guard of the run's learners in a run with step hooks: they send and count nothing as a step's end runs
        the hooks (``clock.ending``: the run's, or the policy's own when the policies step independently), whose updates
        go through their step's end.  Stepping independently (``clock`` given), they also send nothing from the hooks of
        any policy's step end (:meth:`_hook_phase`): a hook updates only the policy whose step ends, through its end."""

        if (clock or self.clock).ending or (clock is not None and _IN_STEP_HOOKS.get()):
            raise RuntimeError(
                "run.learner() is the trainer's, which sends nothing as a step's end runs its step hooks: a hook "
                "updates the policy through end.train(...) or end.learner(...), during its update()"
            )

    @contextlib.asynccontextmanager
    async def _hook_phase(self) -> AsyncGenerator[None, None]:
        """Stepping independently, the run's ``hooks.running``, held while a step's end runs its hooks: the commits'
        hook phase (:meth:`~platoon.train.checkpoints.commits.Commits.hook_phase`), and, in the task that runs them,
        no learner of the run's sends (:meth:`_outside_hooks`), the other policies' learners included, so that a hook
        reaches no other policy through ``run.learner``."""

        assert self.checkpoints.commits is not None  # the run's policies step independently
        token = _IN_STEP_HOOKS.set(True)
        try:
            async with self.checkpoints.commits.hook_phase():
                yield
        finally:
            _IN_STEP_HOOKS.reset(token)

    async def _learner_wait(self, awaitable: Awaitable[Any], *, name: str, total_timeout: float | None = None) -> Any:
        """:meth:`wait`, as a learner awaits its results (the waiter of a save or an update: ``total_timeout``)."""

        return await self.wait(awaitable, name=name, timeout=total_timeout)

    async def wait(
        self, awaitable: Awaitable[Any], *, name: str, timeout: float | None = None, interruptible: bool = False
    ) -> Any:
        """Await ``awaitable`` with a log line and a watchdog heartbeat every minute (the trainer's
        ``_await_with_heartbeat``), for at most ``timeout`` seconds (``TimeoutError`` after); with ``interruptible``,
        ``TrainInterrupted`` once a stop is asked for instead, the awaitable cancelled.  Without, a stop request lets
        it finish: the step stops at its next interruptible wait."""

        if interruptible:
            awaitable = _unless_stopped(awaitable, self.shutdown)
        watchdog = getattr(getattr(self, "clock", None), "watchdog", None)
        return await self.trainer._await_with_heartbeat(awaitable, name=name, watchdog=watchdog, total_timeout=timeout)

    async def end_step(self, *, sampled: bool = True, policy: PolicyId | None = None) -> dict[PolicyId, int]:
        """The end of a step of a trainer other than rl, after the step's own optimizer updates and its commit
        (``source.committed``; stepping independently, ``source.ledger.consumed``): the step hooks drain and update,
        each policy that the step or a hook updated publishes once (:meth:`publish`), the hooks hear it, and the clock
        advances, moving the horizon of the start's draws on (:func:`~platoon.train.trainer.hooks.finish_step`, which
        the rl trainer's step runs too).  With no hook, :meth:`publish` then the advance; when nothing but a step hook
        samples the policies (``sampled=False``: the sft trainer's), a step without hooks, or one that neither it nor a
        hook updated, publishes nothing, and saves the full checkpoint the cadence asks for after it, if any
        (:meth:`_save_due`).  The run counts the updates from each policy's optimizer steps since the step began.
        Stepping independently, ``policy`` names whose step ends: its clock advances, and only it publishes and may
        commit.  Who published, at which version."""

        if self.independent and policy is None:
            raise ValueError("the run's policies step independently: end_step(policy=...) names whose step ends")
        clock, source = self._clock_of(policy) if policy is not None else self.clock, self.source
        step, began = clock.step, self._step_began
        ledger = source.ledger if source is not None else None

        def updates() -> dict[PolicyId, int]:
            counted = self._updates_since(began)
            return counted if policy is None else {policy: counted.get(policy, 0)}

        def end_of() -> StepEnd:
            if ledger is not None and policy is not None:
                consumed = dict(ledger.last_consumed.get(policy, {}))
            else:
                consumed = dict(source.consumed) if source is not None else {}
            return self._step_end("step", step=step, step_updates=updates(), consumed=consumed, policy=policy)

        async def publish(end: StepEnd | None) -> dict[PolicyId, int]:
            updated = [each for each, count in updates().items() if count > 0]
            if not sampled and (end is None or not updated):
                # Nothing but a hook samples the policies, and there is none, or no update to publish: no sampler,
                # but the checkpoint the cadence asks for all the same.
                await self._save_due({}, policy=policy)  # the counters as they are
                return {}
            return await self.publish(updated)

        published = await finish_step(self.hooks, end_of, clock=clock, source=source, publish=publish)
        if policy is None:
            self._step_began = self._optimizer_steps()
        else:
            self._step_began = {**self._step_began, policy: self.policies[policy].optimizer_steps}
        if ledger is not None and policy is not None:
            ledger.last_consumed[policy] = {}
        return dict(published)

    def _optimizer_steps(self) -> dict[PolicyId, int]:
        """Each policy's optimizer steps so far."""

        return {policy.id: policy.optimizer_steps for policy in self.policies.values()}

    def _updates_since(self, began: Mapping[PolicyId, int]) -> dict[PolicyId, int]:
        """Each policy's optimizer steps since ``began`` (:meth:`_optimizer_steps` then)."""

        return {policy: count - began.get(policy, 0) for policy, count in self._optimizer_steps().items()}

    async def publish(self, updated: Sequence[PolicyId]) -> dict[PolicyId, int]:
        """After a step's updates, each policy of ``updated`` publishes a sampler of its new weights, once, as one new
        version (:meth:`~platoon.train.policy.policy.Policy.publish`): from a full checkpoint of the run, which saves
        every trained policy, when the checkpoint cadence says one is due after this step (``checkpoint.every``,
        ``checkpoint.recovery_every``), else from a weights-only save of each; the rollout workers of the draws the run
        opened are given it.  Each one's new version.  A frozen policy publishes none (ValueError): it keeps the
        sampler it serves."""

        policies = [self.policies[policy_id] for policy_id in updated]
        if frozen := [policy.id for policy in policies if not policy.trainable]:
            raise ValueError(f"policy {frozen[0]!r} is frozen: it keeps the sampler it serves, and publishes none")
        if not policies:
            return {}
        if self.independent and len(policies) > 1:
            raise ValueError("the run's policies step independently: each step's end publishes its own policy only")
        sampling_clients = await self._samplers_after_step(policies)
        published = {}
        for policy in policies:
            policy.publish(sampling_clients[policy.id])
            self._bind_workers(policy)
            published[policy.id] = policy.version
        return published

    def _bind_workers(self, policy: Policy, *, required: bool = False) -> None:
        """Give the sampling client of ``policy``, a policy the run trains, to the training workflow's rollout workers
        (its ``publish_model_binding``), as the rl trainer's steps do, when the run opened the start's draws
        (:meth:`rollouts`) and their agents sample it: as their default binding for the agents' default policy
        (``agents.policy``), else as its own.  The launch's is ``required``: under ``rollout_isolation: process`` a
        worker cannot start without it; a later one's failure is logged, each worker keeping the client it holds."""

        agents = self.policies.agents
        if not self._binds_workers or policy.id not in agents.sampled():
            return
        assert self.workflows is not None  # a run that opened draws has its workflows
        other = None if policy.id == agents.default else policy.id
        _publish_binding(self.workflows.train, policy.sampling_client, policy.version, required=required, policy=other)

    def _bind_frozen_workers(self) -> None:
        """Give the training workflow's rollout workers the binding of each frozen policy their agents sample, once, at
        the launch of a run that opened the start's draws (:meth:`rollouts`), after each trained one's
        (:meth:`_bind_workers`): a frozen policy's sampler never changes, so no step publishes it again (the rl
        trainer's loop gives its workers every policy's first binding itself, before its first group:
        :func:`~platoon.train.policy.binding.publish_other_bindings`).  Required, as the launch's bindings of the
        trained policies are."""

        assert self.workflows is not None  # a run that opened draws has its workflows
        for policy in self.policies.sampled():
            if not policy.trainable:
                _publish_binding(
                    self.workflows.train, policy.sampling_client, policy.version, required=True, policy=policy.id
                )

    async def _samplers_after_step(self, policies: Sequence[Policy]) -> dict[PolicyId, Any]:
        """The sampling client of each of ``policies``' weights after the clock's step: of the full checkpoint that the
        cadence saves after it (:meth:`_save_due`), whose rows record one version more for each of ``policies`` and
        their optimizer steps; else of a weights-only save of each, all at once
        (:meth:`~platoon.train.policy.set.PolicySet.each`).  Stepping independently, ``policies`` is the one policy
        whose step ends, after its own clock's step: of its commit, else of a weights-only save of it."""

        counters = {policy.id: Counters(policy.version + 1, policy.optimizer_steps) for policy in policies}
        if self.independent:  # publish refuses more than the one policy whose step ends
            (policy,) = policies
            saved = await self._save_due(counters, policy=policy.id)
            if saved is None:
                return {policy.id: await policy.save_for_sampler()}
            return {policy.id: sampler_from(policy.training_client, saved[policy.id])}
        saved = await self._save_due(counters)
        if saved is None:
            saving = {policy.id: policy.save_for_sampler() for policy in policies}
            return await self.policies.each(saving, doing="saving its weights")
        return {policy.id: sampler_from(policy.training_client, saved[policy.id]) for policy in policies}

    async def _save_due(
        self, counters: Mapping[PolicyId, Counters], *, policy: PolicyId | None = None
    ) -> dict[PolicyId, dict[str, str]] | None:
        """The full checkpoint of the run that the cadence asks for after the clock's step (``checkpoint.every``,
        ``checkpoint.recovery_every``), named for the next step: a barrier of every trained policy's saves, whose rows
        record ``counters`` of the policies there and the others' counters as they are; None when none is due.
        Stepping independently, that of ``policy`` alone, whose step ends: the commit its own cadence asks for after
        its own clock's step (:meth:`_commit_due`), its row recording its ``counters``, else its counters as they are.
        A recovery checkpoint goes to the trainer's pruning once ``checkpoints.jsonl`` records it
        (``_recovery_saved``): the leader's save, at the version its row records."""

        leader = self.policies[self.policies.leader]
        if self.independent:
            assert policy is not None  # end_step names whose step ends
            trained = self.policies[policy]
            counted = counters.get(policy, Counters(trained.version, trained.optimizer_steps))
            kind, paths = await self._commit_due(policy, counted)
            if paths is None:
                return None
            if kind == "recovery" and policy == leader.id:
                self.trainer._recovery_saved(paths, counted.version)
            return {policy: paths}
        step, checkpoint = self.clock.step, self.config.checkpoint
        save_every = event_interval(checkpoint, self.plan.steps_per_epoch or 1)
        kind = checkpoint_kind_for_step(step - self.start.step + 1, save_every, checkpoint.recovery_every)
        if kind is None:
            return None
        name = f"{step + 1:06d}" if kind == "permanent" else f"{RECOVERY_CHECKPOINT_PREFIX}{step + 1:06d}"
        logger.info(f"Saving {kind} checkpoint {name} after step {step}")
        own = counters.get(leader.id, Counters(leader.version, leader.optimizer_steps))
        others = {policy: counted for policy, counted in counters.items() if policy != leader.id}
        saved = await self._save(
            name, step=step + 1, policy_version=own.version, optimizer_steps=own.optimizer_steps, others=others
        )
        if kind == "recovery":
            self.trainer._recovery_saved(saved[leader.id], own.version)
        return saved

    async def own_step_sampler(
        self, policy_id: PolicyId, *, wait: Any = None
    ) -> tuple[Any, str | None, dict[str, str] | None]:
        """Stepping independently, the sampling client of ``policy_id``'s weights after its own step: of a commit when
        its own cadence says a full checkpoint is due after it (:meth:`_commit_due`), recording one version more and
        awaited under ``wait`` (default: the trainer's heartbeat); else of a weights-only save.  With the kind of the
        checkpoint (None: weights only) and the paths it wrote, which the caller hands to its pruning (the rl trainer's
        steps)."""

        policy = self.policies[policy_id]
        counters = Counters(policy.version + 1, policy.optimizer_steps)
        kind, saved = await self._commit_due(policy_id, counters, wait=wait)
        if saved is None:
            return await policy.save_for_sampler(), None, None
        return sampler_from(policy.training_client, saved), kind, saved

    async def _commit_due(
        self, policy_id: PolicyId, counters: Counters, *, wait: Any = None
    ) -> tuple[str | None, dict[str, str] | None]:
        """Stepping independently, the commit that ``policy_id``'s own cadence asks for after its clock's step
        (:meth:`~platoon.train.trainer.clocks.OwnClocks.checkpoint_after`), whose row of it records ``counters``,
        awaited under ``wait`` (default: the trainer's heartbeat): its kind and the paths it wrote; (None, None) when
        none is due."""

        assert self._own is not None  # the trained policies step independently
        kind, name = self._own.checkpoint_after(policy_id)
        if kind is None:
            return None, None
        step = self._own[policy_id].step
        logger.info(f"Saving {kind} checkpoint {name} of policy {policy_id} after its step {step}")
        saved = await self.checkpoints.commit(
            policy_id, name, step=step + 1, counters=counters, wait=wait or self.trainer._await_with_heartbeat
        )
        return kind, saved

    def _step_end(
        self,
        kind: Literal["launch", "step"],
        *,
        step: int,
        step_updates: Mapping[PolicyId, int] | None = None,
        consumed: Mapping[Any, str] | None = None,
        published: Mapping[PolicyId, int] | None = None,
        learner_of: "Callable[[PolicyId], Learner] | None" = None,
        policy: PolicyId | None = None,
        tracker: Any = None,
    ) -> StepEnd:
        """What a step's end, or the launch, tells the hooks, where every one is built: the run's policies, the waits,
        whether a hook's wait may stop the step (the trainer's steps are atomic), and the learner of each policy, which
        a hook's learner shares the training client and checks of: ``learner_of``'s (the rl step's own, whose barrier
        the hooks' updates wait at too), else the run's (:meth:`learner`).  ``policy``: whose step ends, when the
        policies step independently: the only one the hooks may update, and whose updates the end counts.  ``tracker``:
        the step's metrics row (default: the trainer's train tracker; a policy's own, stepping independently)."""

        tracker = tracker if tracker is not None else self.trainer._train_tracker
        assert tracker is not None  # the trainer was entered (async with)
        trained = {trained.id: trained for trained in self.policies.trainable()}
        run = RunSide(
            policies=trained if policy is None else {policy: trained[policy]},
            learner_of=learner_of or self.learner,
            wait=self.wait,
            interruptible=self.needs.atomic_steps,
        )
        return StepEnd(
            kind=kind,
            step=step,
            step_updates=step_updates or {},
            consumed=consumed or {},
            tracker=tracker,
            published=published or {},
            policy=policy,
            _run=run,
        )

    async def execute(self) -> None:
        """The start's phases, in order (the module's docstring numbers them); phase 10 also after a failure."""

        try:
            await self._phases()
        except BaseException as error:
            await self._close(error)
            raise
        await self._close(None)

    async def _phases(self) -> None:
        """Phases 0 to 9."""

        trainer = self.trainer
        if self.workflows is None:  # 0
            self.workflows = self._built_workflows()
        trainer.check(self)  # 1
        self._check_stepping()
        self.shutdown = asyncio.Event()  # 2: graceful Ctrl+C, and what platoon train passes on
        trainer._setup_signal_handlers(self.shutdown)
        # 3. A key the service refuses (401, 403) ends training with one line that names the key and the service.
        self.policies.connect()
        leader = self.policies.leader
        others = [policy.id for policy in self.policies.trainable() if policy.id != leader]
        self.start = StartPoint.read(
            self.run_dir, self.config.train.num_minibatches, leader=leader, others=others, independent=self.independent
        )
        self._check_resumed_stepping()
        self.checkpoints.resume(self.start)
        if not await trainer._open_policies(self, self.start):  # 4
            return  # stopped before training began: nothing to save
        self.plan = await trainer.prepare(self, self.start)  # 5
        self._check_claimed()
        await self._launch()  # 6
        await trainer.ready(self)  # 7
        await self._run_steps()  # 8
        await self._save_at_exit()  # 9
        self.ended = True

    def _check_stepping(self) -> None:
        """1. What loading refuses of how the trained policies step (:func:`~platoon.train.policy.spec.
        stepping_problems`), against the stepping the trainer's needs choose, which loading reads from
        ``train.stepping``: a ``ConfigError`` (exit 78) for a policy's own batch or checkpoint cadence in a start whose
        policies step together, and for a group admission rule or a batch the step does not divide in one whose
        policies step independently."""

        if problems := stepping_problems(self.config, independent=self.independent):
            raise ConfigError(problems)

    def _check_resumed_stepping(self) -> None:
        """3. A start keeps the stepping its run's checkpoints were saved with: one whose trained policies step
        together that would resume a commit (a row with ``commit_seq``) stops with a ``ConfigError`` (exit 78), since
        it would train again the shares each policy consumed on its own clock.  (A start that steps independently finds
        no commit in a run saved in lockstep, which :func:`~platoon.train.checkpoints.commits.newest_commit`
        refuses.)"""

        record = self.start.record
        if record is not None and not self.independent and COMMIT_SEQ_KEY in self.start.loop_state:
            raise ConfigError(
                f"checkpoint {record.name} is a commit of trained policies that step independently, each on its own "
                "clock, and this start steps them together: a resume keeps the stepping its run began with "
                "(train.stepping: independent, with every policy the run trained), else a new trial in a new run "
                "directory"
            )

    def _built_workflows(self) -> Workflows | None:
        """0. The start's workflows, which the trainer's setup builds on its policies
        (:meth:`~platoon.train.trainer.api.TrainerSetup.build_workflows`), for a trainer that rolls out; None for one
        that rolls out none, or that was built outside ``platoon train``, without a setup."""

        trainer = self.trainer
        if trainer.setup is None or not self.needs.rollouts:
            return None
        return trainer.setup.build_workflows(self.policies)

    async def _launch(self) -> None:
        """6. Save the launch checkpoint, then start from it: the trainer's ``started``, and each trained policy's first
        sampling client, of the launch save's weights, which it samples at the checkpoint's policy version, with the
        checkpoint's optimizer steps (a frozen policy keeps the sampler it was built with), and which the rollout
        workers of the draws the run opened are given before any group starts, with the binding of each frozen policy
        their agents sample, once; then the step hooks hear of it.  The launch save is a *recovery* checkpoint: every
        (re)launch writes one, so naming it permanently would keep a ~490 GB checkpoint per restart."""

        start = self.start
        name = f"{RECOVERY_CHECKPOINT_PREFIX}{start.step:06d}"
        logger.info(f"Saving recovery checkpoint {name} at launch (batch {start.step})")
        self.launch_save = await self._save(
            name,
            step=start.step,
            policy_version=start.policy_version,
            optimizer_steps=start.optimizer_steps,
            # Every other trained policy's row records where it starts: its sampler is not published yet.
            others={
                policy: Counters(begins.version, begins.optimizer_steps, begins.step)
                for policy, begins in start.policies.items()
                if policy != self.policies.leader
            },
        )
        if self.source is not None:
            self.source.started()
        await self.trainer.started(self)
        for policy in self.policies.trainable():
            _first_sampler(policy, self.launch_save[policy.id], start.policies[policy.id])
            self._bind_workers(policy, required=True)
        if self._binds_workers:
            self._bind_frozen_workers()
        if self.hooks:
            published = {policy.id: policy.version for policy in self.policies.trainable()}
            await hear_launch(self.hooks, self._step_end("launch", step=start.step, published=published))

    def _check_claimed(self) -> None:
        """5. Every key of the checkpoint the start resumes from is claimed, by the run's counters or by a
        participant of its checkpoints; else the start stops (``ConfigError``, exit 78): what the checkpoint saved
        would be dropped silently, the run's parts differing from the checkpoint's."""

        if unclaimed := self.checkpoints.unclaimed(self.start):
            record = self.start.record
            raise ConfigError(
                [
                    f"checkpoint {record.name if record else '?'} saved loop_state key {key!r}, which no part of this "
                    "run takes up: resume with the trainer, step hooks and participants that saved it, or start a new "
                    "run"
                    for key in unclaimed
                ]
            )

    async def _run_steps(self) -> None:
        """8. The trainer's steps.  An interrupt they raise (``TrainInterrupted``: a wait that a stop request ended)
        stops them, as one they report does (``run.interrupted``).  Then the clock's watchdog stops."""

        try:
            await self.trainer.run_steps(self)
        except TrainInterrupted:
            self.interrupted = True
            logger.info("Training interrupted")
        finally:
            self._stop_watchdog()

    def _stop_watchdog(self) -> None:
        """8. Once the steps stop: the clock's watchdog stops, unless the steps stopped it (the rl trainer's do)."""

        clock = getattr(self, "clock", None)
        if clock is not None and clock.watchdog is not None and clock.watchdog.running:
            clock.watchdog.stop()

    async def _save_at_exit(self) -> None:
        """9. Once the steps stop: an interrupted start that trained a step saves a permanent checkpoint to resume from
        (``<step>_interrupted``; one that trained none saves nothing), and a start that trained to the plan's end the
        final one; a start that had no step left to train says why (the trainer's ``_nothing_to_train``).  Stepping
        independently, a step of any policy counts: a start that resumes with the leader at the plan's end (it died, or
        stopped, after the leader's last commit) still trains the other policies' last shares, and saves them.  Steps
        that a stop request stopped short of the plan's end were interrupted, also when the trainer did not say so
        (``run.interrupted``): a final row would claim steps they never trained.  An interrupt inside a step that no
        checkpoint may save (:meth:`_unsaved_step`) saves nothing: the weights would be the step's, or half its, the
        counters and the participants' state those before it."""

        step, first, end = self.clock.step, self.start.step, self.plan.end_step
        leader = self.policies[self.policies.leader]
        own = self._own
        others = own.exit_counters() if own is not None else None  # stepping independently, each at its own step
        others_stepped = own is not None and own.others_stepped()
        if self.interrupted or (step < end and self.shutdown.is_set()):
            if (inside := self._unsaved_step()) is not None:
                logger.info(
                    f"Training was interrupted inside step {step}, {inside}: no checkpoint saves it, and the run "
                    f"resumes from checkpoint {self.checkpoints.last_saved}"
                )
            elif step > first or others_stepped:
                logger.info(f"Saving permanent checkpoint {step:06d}_interrupted at step {step}...")
                await self._save(
                    f"{step:06d}_interrupted",
                    step=step,
                    policy_version=leader.version,
                    optimizer_steps=leader.optimizer_steps,
                    others=others,
                )
                logger.info(f"Checkpoint saved. Resume training to continue from step {step}.")
        elif first < end or others_stepped:
            logger.info(f"Saving permanent checkpoint final at step {end}...")
            await self._save(
                "final", step=end, policy_version=leader.version, optimizer_steps=leader.optimizer_steps, others=others
            )
            logger.info("Training complete successfully!")
        else:
            self.trainer._nothing_to_train(self)

    def _unsaved_step(self) -> str | None:
        """Why no checkpoint may save the step an interrupt came inside (rule 10), or None: when the trainer's steps
        are atomic (``needs().atomic_steps``), the step had sent training work (``clock.step_open``); when they are not,
        its end ran its step hooks (``clock.ending``), after its own updates and its commit, the hooks' updates perhaps
        half done."""

        clocks = list(self._own.values()) if self._own is not None else [self.clock]
        if self.needs.atomic_steps:
            return "which had sent training work" if any(clock.step_open for clock in clocks) else None
        return "as its step hooks ran" if any(clock.ending for clock in clocks) else None

    async def _save(
        self,
        name: str,
        *,
        step: int,
        policy_version: int,
        optimizer_steps: int,
        others: Mapping[PolicyId, Counters] | None = None,
    ) -> dict[PolicyId, dict[str, str]]:
        """Full checkpoint ``name`` of the run (:meth:`RunCheckpoints.save`), awaited under the trainer's heartbeat;
        ``others``: the counters each other trained policy's row records (default: its own now)."""

        return await self.checkpoints.save(
            name,
            step=step,
            policy_version=policy_version,
            optimizer_steps=optimizer_steps,
            wait=self.trainer._await_with_heartbeat,
            **({"others": others} if others else {}),
        )

    async def _close(self, error: BaseException | None) -> None:
        """10. In ``finally``, also after a failure (``error``): the trainer's ``close``, then each hook's and
        participant's ``aclose``, in reverse order of joining, every error kept (one that is not critical is logged
        instead); then, after steps that ended and closed cleanly, the draws' statistics, as they always closed.  After
        a failure, its error is raised as it was (``execute``), each error of the close logged and noted on it, so
        that a ``ConfigError`` still exits 78; else the close's error, or its errors together."""

        errors = await self._close_parts()
        if error is None and self.ended and not errors and self.source is not None:
            self.source.close()
        if error is not None:
            for exc in errors:
                logger.error(f"Closing the start after its failure failed too: {exc!r}", exc_info=exc)
                error.add_note(f"Closing the start failed too: {exc!r}")
        elif len(errors) == 1:
            raise errors[0]
        elif errors:
            raise ExceptionGroup("closing the start failed", errors)

    async def _close_parts(self) -> list[Exception]:
        """The trainer's ``close``, then each hook's and participant's ``aclose``, in reverse order of joining; the
        errors of each, but of a part that is not critical, whose error is logged."""

        errors: list[Exception] = []
        try:
            await self.trainer.close(self)
        except Exception as exc:
            errors.append(exc)
        for part in reversed(self._parts):
            if (aclose := getattr(part, "aclose", None)) is None:
                continue
            try:
                await aclose()
            except Exception as exc:
                if is_critical(part):
                    errors.append(exc)
                else:
                    logger.exception(f"Closing {getattr(part, 'name', part)!r} failed; it is not critical")
        return errors


async def _unless_stopped(awaitable: Awaitable[Any], stop: asyncio.Event) -> Any:
    """``awaitable``'s result, or ``TrainInterrupted`` once ``stop`` is set first (the awaitable is cancelled)."""

    task = asyncio.ensure_future(awaitable)
    stopped = asyncio.ensure_future(stop.wait())
    try:
        await asyncio.wait({task, stopped}, return_when=asyncio.FIRST_COMPLETED)
    except BaseException:
        task.cancel()
        raise
    finally:
        stopped.cancel()
    if task.done():
        return task.result()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    raise TrainInterrupted()


def _first_sampler(policy: Policy, saved: dict[str, str], start: PolicyStart) -> None:
    """The policy's first sampling client, of the sampler weights the launch save wrote (``saved``), which the policy
    publishes at the version its start point says, without counting a new one (nothing subscribes yet: the rollout
    workers get their first bindings as the steps begin), and its optimizer steps so far."""

    # Not policy.checkpoints.sampler_from, whose log lines the launch never wrote.
    sampling_client = policy.training_client.create_sampling_client(saved["sampler_path"])
    policy.publish(sampling_client, version=start.version)
    policy.optimizer_steps = start.optimizer_steps


def _publish_binding(
    workflow: Any, sampling_client: Any, version: int, *, required: bool = False, policy: PolicyId | None = None
) -> None:
    """Give a sampling client to ``workflow``'s rollout workers (its ``publish_model_binding``): the agents' default
    policy's, or ``policy``'s, another policy they sample.  A workflow without workers takes none: its rollouts run in
    this process, on the policies' model gateways.  A failure is logged, each worker keeping the client it holds; a
    ``required`` one, under ``rollout_isolation: process``, raises: a worker cannot start without it."""

    if (publish := getattr(workflow, "publish_model_binding", None)) is None:
        return
    try:
        if policy is None:
            publish(sampling_client, version)
        else:
            publish(sampling_client, version, policy=policy)
    except Exception:
        logger.exception("could not publish the rollout worker sampling-client binding")
        if required and getattr(workflow, "rollout_isolation", "inprocess") == "process":
            raise
