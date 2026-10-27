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
   each policy's first sampling client, of the launch save's weights, which the rollout workers of draws the run
   opened are given, and the step hooks hear of it;
7. the trainer readies its steps (``ready``), and the run's clock starts;
8. the steps (``run_steps``), each ending with the step hooks (:meth:`TrainerRun.end_step`; the rl trainer's step runs
   the same sequence);
9. the save at exit: ``<step>_interrupted`` after an interrupt (the trainer's, or a stop request its steps stopped
   on short of their end), unless it came inside a step that no checkpoint may save (rule 10): one that sent training
   work, when the trainer's steps are atomic, else one whose end ran its step hooks; ``final`` at the run's end;
10. in ``finally``, also after a failure: the trainer closes what it opened (``close``), then each hook and participant
    (``aclose``), in reverse order of registration; then, after steps that ended and closed cleanly, the draws'
    statistics.  A start that failed raises its error, the close's errors noted on it; else the close's errors.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol

from platoon.train.checkpoints.participants import RunCheckpoints
from platoon.train.checkpoints.pruning import RECOVERY_CHECKPOINT_PREFIX
from platoon.train.checkpoints.resume import PolicyStart, StartPoint
from platoon.train.checkpoints.saving import checkpoint_kind_for_step
from platoon.train.config_defs import event_interval
from platoon.train.dispatch.source import RolloutSource
from platoon.train.objectives.policy_gradient import PolicyGradient
from platoon.train.policy.checkpoints import sampler_from
from platoon.train.policy.learner import Learner
from platoon.train.policy.policy import Policy, adam_params
from platoon.train.policy.spec import PolicyId
from platoon.train.trainer.api import StepPlan, Workflows
from platoon.train.trainer.hooks import RunSide, StepEnd, StepHooks, finish_step, hear_launch, is_critical
from platoon.train.trainer.signals import TrainInterrupted, Watchdog
from platoon.utils.config import ConfigError

if TYPE_CHECKING:
    from platoon.train.objectives.api import Objective
    from platoon.train.trainer.api import TrainerBase

logger = logging.getLogger(__name__)


class _Consumer(Protocol):
    """What a step consumed its groups from, whose horizon moves on as the next step starts."""

    def step_started(self) -> None: ...


class StepClock:
    """The run's step clock: the step training is at (``step``), the step after the run's last (``end``: the steps run
    while ``step < end``; None for a clock outside a run), the event each step's end sets (``updated``: the evaluation
    loop wakes on it), and the run's watchdog, which each step's end feeds (None when the config turns it off).
    ``step_open`` says that the step has sent training work (a learner's first submission sets it, and so do the step
    hooks' updates; a step that sends work by hand sets it itself), until it ends: rule 10 saves no checkpoint of a
    step that is open, when the trainer's steps are atomic.  ``ending`` says that the step's end runs its step hooks
    (:func:`~platoon.train.trainer.hooks.finish_step`), until the clock advances: when the trainer's steps are not
    atomic, rule 10 saves no checkpoint of a step that is ending.  ``open_updates`` holds each policy's open update,
    sent and not counted yet, which a step's end refuses to start its hooks or to publish with."""

    def __init__(
        self, *, step: int, updated: asyncio.Event, end: int | None = None, watchdog: Watchdog | None = None
    ) -> None:
        self.step = step
        self.end = end
        self.updated = updated
        self.watchdog = watchdog
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
        its groups from moves its horizon on (``consumed``); the evaluation loop hears of it, also after a step that
        published no new policy; and the watchdog is fed."""

        self.step += 1
        self.step_open = False
        self.ending = False
        if consumed is not None:
            consumed.step_started()
        self.updated.set()
        if self.watchdog:
            self.watchdog.heartbeat()


class TrainerRun:
    """One start of ``trainer``, through the lifecycle every trainer shares (:meth:`execute`), with the given
    workflows (None: those the trainer's setup builds, when it rolls out).  What its hooks use: the config, the run
    directory, the policies (by id; ``policies.leader`` is the one the run trains), the workflows, what the trainer
    needs of the run (``needs``), the run's checkpoints
    (:class:`~platoon.train.checkpoints.participants.RunCheckpoints`) and step hooks
    (:class:`~platoon.train.trainer.hooks.StepHooks`), the shutdown request, where the start begins and where its
    steps end, the start's draws (:meth:`rollouts`), the launch save, the step clock with its watchdog, and what a
    step uses: a policy's learner (:meth:`learner`), the waits (:meth:`wait`), and the step's end (:meth:`end_step`,
    which publishes through :meth:`publish`)."""

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
        # Every step hook and checkpoint participant, in the order they joined: phase 10 closes them in reverse.
        self._parts: list[Any] = []
        self.checkpoints = RunCheckpoints(
            self.run_dir, self.policies, wandb_run_id=self._wandb_run_id, joined=self._joined
        )
        self.hooks = StepHooks(self.checkpoints, joined=self._joined)
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
        source.open(self.start)
        self.checkpoints.register(source)
        self.source = source
        return source

    def start_clock(self) -> StepClock:
        """Start the run's clock at the start's step, towards the plan's end (:meth:`StepClock.started`); the first
        step's updates count from here (:meth:`end_step`)."""

        self.clock = StepClock.started(self.config, step=self.start.step, end=self.plan.end_step)
        self._step_began = self._optimizer_steps()
        return self.clock

    # --- what a step uses ------------------------------------------------------------------------------------------

    def learner(self, policy: PolicyId, *, objective: "Objective | None" = None, metric_prefix: str = "") -> Learner:
        """The learner of ``policy`` (:class:`~platoon.train.policy.learner.Learner`): its training client and its
        optimizer, which steps with its spec's Adam settings (``train.optimizer``), under ``objective`` (None: the
        policy-gradient loss its spec names, ``train.loss_fn``).  Its results are awaited under the run's waits within
        ``train.*``'s deadlines, its metrics go to the train tracker's step row under ``metric_prefix``, it fails
        closed when the trainer's needs say so (rule 10), and its first submission of a step opens the run clock's
        step: build it once the clock started (``ready()``, ``run_steps()``).  It is the trainer's: in a run with
        step hooks it sends and counts nothing as a step's end runs them (``clock.ending``), and a hook updates
        through its step's end instead (``end.train``, ``end.learner``)."""

        if getattr(self, "clock", None) is None:
            raise RuntimeError("run.learner() opens the run clock's steps: build it once the clock started (ready())")
        trained = self.policies[policy]
        spec = trained.spec
        if getattr(trained, "optimizer_params", None) is None:
            assert spec.optimizer is not None  # policy_specs gives each policy train.optimizer
            trained.optimizer_params = adam_params(spec.optimizer)
        if objective is None:
            assert spec.loss_fn is not None and spec.loss_fn_config is not None  # policy_specs names each policy's
            objective = PolicyGradient(spec.loss_fn, spec.loss_fn_config)
        tracker = self.trainer._train_tracker
        assert tracker is not None  # the trainer was entered (async with)
        return Learner(
            trained,
            objective=objective,
            timeouts=self.config.train,
            wait=self._learner_wait,
            tracker=tracker,
            metric_prefix=metric_prefix,
            fail_closed=self.needs.fail_closed_updates,
            clock=self.clock,
            guard=self._outside_hooks if self.hooks else None,
        )

    def _outside_hooks(self) -> None:
        """The guard of the run's learners in a run with step hooks: they send and count nothing as a step's end runs
        the hooks (``clock.ending``), whose updates go through their step's end."""

        if self.clock.ending:
            raise RuntimeError(
                "run.learner() is the trainer's, which sends nothing as a step's end runs its step hooks: a hook "
                "updates the policy through end.train(...) or end.learner(...), during its update()"
            )

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

    async def end_step(self) -> dict[PolicyId, int]:
        """The end of a step of a trainer other than rl, after the step's own optimizer updates and its commit
        (``source.committed``): the step hooks drain and update, each policy that the step or a hook updated publishes
        once (:meth:`publish`), the hooks hear it, and the clock advances, moving the horizon of the start's draws on
        (:func:`~platoon.train.trainer.hooks.finish_step`, which the rl trainer's step runs too).  With no hook,
        :meth:`publish` then the advance.  The run counts the updates from each policy's optimizer steps since the
        step began.  Who published, at which version."""

        step, began, source = self.clock.step, self._step_began, self.source

        def end_of() -> StepEnd:
            consumed = dict(source.consumed) if source is not None else {}
            return self._step_end("step", step=step, step_updates=self._updates_since(began), consumed=consumed)

        async def publish(end: StepEnd | None) -> dict[PolicyId, int]:
            return await self.publish([policy for policy, count in self._updates_since(began).items() if count > 0])

        published = await finish_step(self.hooks, end_of, clock=self.clock, source=source, publish=publish)
        self._step_began = self._optimizer_steps()
        return dict(published)

    def _optimizer_steps(self) -> dict[PolicyId, int]:
        """Each policy's optimizer steps so far."""

        return {policy.id: policy.optimizer_steps for policy in self.policies.values()}

    def _updates_since(self, began: Mapping[PolicyId, int]) -> dict[PolicyId, int]:
        """Each policy's optimizer steps since ``began`` (:meth:`_optimizer_steps` then)."""

        return {policy: count - began.get(policy, 0) for policy, count in self._optimizer_steps().items()}

    async def publish(self, updated: Sequence[PolicyId]) -> dict[PolicyId, int]:
        """After a step's updates, each policy of ``updated`` publishes a sampler of its new weights, as one new
        version: from a full checkpoint of the run when the checkpoint cadence says one is due after this step
        (``checkpoint.every``, ``checkpoint.recovery_every``), else from a weights-only save; the rollout workers of
        the draws the run opened are given it.  Each one's new version."""

        published = {}
        for policy_id in updated:
            policy = self.policies[policy_id]
            sampling_client = await self._sampler_after_step(policy)
            policy.info.llm.update_sampling_client(sampling_client)
            policy.sampling_client = sampling_client
            self._bind_workers(policy)
            published[policy_id] = policy.version
        return published

    def _bind_workers(self, policy: Policy, *, required: bool = False) -> None:
        """Give the sampling client of ``policy``, the one the run trains, to the training workflow's rollout workers
        (its ``publish_model_binding``), as the rl trainer's step does, when the run opened the start's draws
        (:meth:`rollouts`).  The launch's is ``required``: under ``rollout_isolation: process`` a worker cannot start
        without it; a later one's failure is logged, each worker keeping the client it holds."""

        if not self._binds_workers or policy.id != self.policies.leader:
            return
        assert self.workflows is not None  # a run that opened draws has its workflows
        workflow = self.workflows.train
        if (publish := getattr(workflow, "publish_model_binding", None)) is None:
            return  # its rollouts run in this process, on the policy's model gateway
        try:
            publish(policy.sampling_client, policy.version)
        except Exception:
            logger.exception("could not publish the rollout worker sampling-client binding")
            if required and getattr(workflow, "rollout_isolation", "inprocess") == "process":
                raise

    async def _sampler_after_step(self, policy: Policy) -> Any:
        """The sampling client of ``policy``'s weights after the clock's step: of the full checkpoint that the cadence
        saves after it, named for the next step, else of a weights-only save."""

        step, checkpoint = self.clock.step, self.config.checkpoint
        save_every = event_interval(checkpoint, self.plan.steps_per_epoch or 1)
        kind = checkpoint_kind_for_step(step - self.start.step + 1, save_every, checkpoint.recovery_every)
        if kind is None:
            return await policy.save_for_sampler()
        name = f"{step + 1:06d}" if kind == "permanent" else f"{RECOVERY_CHECKPOINT_PREFIX}{step + 1:06d}"
        logger.info(f"Saving {kind} checkpoint {name} after step {step}")
        saved = await self._save(
            name, step=step + 1, policy_version=policy.version + 1, optimizer_steps=policy.optimizer_steps
        )
        return sampler_from(policy.training_client, saved[policy.id])

    def _step_end(
        self,
        kind: Literal["launch", "step"],
        *,
        step: int,
        step_updates: Mapping[PolicyId, int] | None = None,
        consumed: Mapping[Any, str] | None = None,
        published: Mapping[PolicyId, int] | None = None,
        learner_of: "Callable[[PolicyId], Learner] | None" = None,
    ) -> StepEnd:
        """What a step's end, or the launch, tells the hooks, where every one is built: the run's policies, the waits,
        whether a hook's wait may stop the step (the trainer's steps are atomic), and the learner of each policy, which
        a hook's learner shares the training client and checks of: ``learner_of``'s (the rl step's own, whose barrier
        the hooks' updates wait at too), else the run's (:meth:`learner`)."""

        tracker = self.trainer._train_tracker
        assert tracker is not None  # the trainer was entered (async with)
        run = RunSide(
            policies=dict(self.policies),
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
        self.shutdown = asyncio.Event()  # 2: graceful Ctrl+C, and what platoon train passes on
        trainer._setup_signal_handlers(self.shutdown)
        # 3. A key the service refuses (401, 403) ends training with one line that names the key and the service.
        self.policies.connect()
        self.start = StartPoint.read(self.run_dir, self.config.train.num_minibatches, leader=self.policies.leader)
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

    def _built_workflows(self) -> Workflows | None:
        """0. The start's workflows, which the trainer's setup builds on its policies
        (:meth:`~platoon.train.trainer.api.TrainerSetup.build_workflows`), for a trainer that rolls out; None for one
        that rolls out none, or that was built outside ``platoon train``, without a setup."""

        trainer = self.trainer
        if trainer.setup is None or not self.needs.rollouts:
            return None
        return trainer.setup.build_workflows(self.policies)

    async def _launch(self) -> None:
        """6. Save the launch checkpoint, then start from it: the trainer's ``started``, and each policy's first
        sampling client, of the launch save's weights, which it samples at the checkpoint's policy version, with the
        checkpoint's optimizer steps, and which the rollout workers of the draws the run opened are given before any
        group starts; then the step hooks hear of it.  The launch save is a *recovery* checkpoint: every (re)launch
        writes one, so naming it permanently would keep a ~490 GB checkpoint per restart."""

        start = self.start
        name = f"{RECOVERY_CHECKPOINT_PREFIX}{start.step:06d}"
        logger.info(f"Saving recovery checkpoint {name} at launch (batch {start.step})")
        self.launch_save = await self._save(
            name, step=start.step, policy_version=start.policy_version, optimizer_steps=start.optimizer_steps
        )
        if self.source is not None:
            self.source.started()
        await self.trainer.started(self)
        for policy in self.policies.values():
            _first_sampler(policy, self.launch_save[policy.id], start.policies[policy.id])
            self._bind_workers(policy, required=True)
        if self.hooks:
            published = {policy.id: policy.version for policy in self.policies.values()}
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
        stops them, as one they report does (``run.interrupted``)."""

        try:
            await self.trainer.run_steps(self)
        except TrainInterrupted:
            self.interrupted = True
            logger.info("Training interrupted")

    async def _save_at_exit(self) -> None:
        """9. Once the steps stop: an interrupted start that trained a step saves a permanent checkpoint to resume from
        (``<step>_interrupted``; one that trained none saves nothing), and a start that trained to the plan's end the
        final one; a start that had no step left to train says why (the trainer's ``_nothing_to_train``).  Steps that a
        stop request stopped short of the plan's end were interrupted, also when the trainer did not say so
        (``run.interrupted``): a final row would claim steps they never trained.  An interrupt inside a step that no
        checkpoint may save (:meth:`_unsaved_step`) saves nothing: the weights would be the step's, or half its, the
        counters and the participants' state those before it."""

        step, first, end = self.clock.step, self.start.step, self.plan.end_step
        leader = self.policies[self.policies.leader]
        if self.interrupted or (step < end and self.shutdown.is_set()):
            if (inside := self._unsaved_step()) is not None:
                logger.info(
                    f"Training was interrupted inside step {step}, {inside}: no checkpoint saves it, and the run "
                    f"resumes from checkpoint {self.checkpoints.last_saved}"
                )
            elif step > first:
                logger.info(f"Saving permanent checkpoint {step:06d}_interrupted at step {step}...")
                await self._save(
                    f"{step:06d}_interrupted",
                    step=step,
                    policy_version=leader.version,
                    optimizer_steps=leader.optimizer_steps,
                )
                logger.info(f"Checkpoint saved. Resume training to continue from step {step}.")
        elif first < end:
            logger.info(f"Saving permanent checkpoint final at step {end}...")
            await self._save("final", step=end, policy_version=leader.version, optimizer_steps=leader.optimizer_steps)
            logger.info("Training complete successfully!")
        else:
            self.trainer._nothing_to_train(self)

    def _unsaved_step(self) -> str | None:
        """Why no checkpoint may save the step an interrupt came inside (rule 10), or None: when the trainer's steps
        are atomic (``needs().atomic_steps``), the step had sent training work (``clock.step_open``); when they are not,
        its end ran its step hooks (``clock.ending``), after its own updates and its commit, the hooks' updates perhaps
        half done."""

        clock = self.clock
        if self.needs.atomic_steps:
            return "which had sent training work" if clock.step_open else None
        return "as its step hooks ran" if clock.ending else None

    async def _save(
        self, name: str, *, step: int, policy_version: int, optimizer_steps: int
    ) -> dict[PolicyId, dict[str, str]]:
        """Full checkpoint ``name`` of the run (:meth:`RunCheckpoints.save`), awaited under the trainer's heartbeat."""

        return await self.checkpoints.save(
            name,
            step=step,
            policy_version=policy_version,
            optimizer_steps=optimizer_steps,
            wait=self.trainer._await_with_heartbeat,
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
    samples at the version its start point says, without counting a new one, and its optimizer steps so far."""

    # Not policy.checkpoints.sampler_from, whose log lines the launch never wrote.
    sampling_client = policy.training_client.create_sampling_client(saved["sampler_path"])
    policy.info.llm.set_version(start.version)
    policy.info.llm.update_sampling_client(sampling_client, increment_version=False)
    policy.sampling_client = sampling_client
    policy.optimizer_steps = start.optimizer_steps
