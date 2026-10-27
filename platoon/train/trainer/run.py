"""One start of one trainer (:class:`TrainerRun`): the lifecycle every trainer goes through, and what the trainer's
hooks use; and the run's step clock (:class:`StepClock`).

:meth:`TrainerRun.execute` reads as the start's phases, in the order they run.  Each phase that is the trainer's own
calls its hook (:class:`~platoon.train.trainer.api.TrainerBase`):

0. the start's workflows, which the trainer's setup builds on its policies, when the trainer rolls out and was given
   none (``platoon train`` builds the rl trainer's itself, and hands them to its ``train()``);
1. the trainer's checks (``check``);
2. the signal handlers: a first signal asks the steps to stop at the next step boundary (``run.shutdown``);
3. each policy's service client, and where the start begins (:class:`~platoon.train.checkpoints.resume.StartPoint`),
   which the participants of the run's checkpoints registered so far take up;
4. each policy's training client at the start point; a start stopped before the service accepted it ends here;
5. the trainer prepares its steps (``prepare``), and says where they end; a trainer that rolls out opens the start's
   draws there (:meth:`TrainerRun.rollouts`);
6. the launch save, a full checkpoint every start writes, then the draws and the trainer start from it (``started``),
   and each policy's first sampling client, of the launch save's weights;
7. the trainer readies its steps (``ready``), and the run's clock starts;
8. the steps (``run_steps``);
9. the save at exit: ``<step>_interrupted`` after an interrupt (the trainer's, or a stop request its steps stopped
   on short of their end), ``final`` at the run's end;
10. the trainer closes what it opened (``close``), then the run its draws.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Protocol

from platoon.train.checkpoints.participants import RunCheckpoints
from platoon.train.checkpoints.pruning import RECOVERY_CHECKPOINT_PREFIX
from platoon.train.checkpoints.resume import PolicyStart, StartPoint
from platoon.train.dispatch.source import RolloutSource
from platoon.train.policy.policy import Policy
from platoon.train.policy.spec import PolicyId
from platoon.train.trainer.api import StepPlan, Workflows
from platoon.train.trainer.signals import Watchdog

if TYPE_CHECKING:
    from platoon.train.trainer.api import TrainerBase

logger = logging.getLogger(__name__)


class _Consumer(Protocol):
    """What a step consumed its groups from, whose horizon moves on as the next step starts."""

    def step_started(self) -> None: ...


class StepClock:
    """The run's step clock: the step training is at (``step``), the step after the run's last (``end``: the steps run
    while ``step < end``; None for a clock outside a run), the event each step's end sets (``updated``: the evaluation
    loop wakes on it), and the run's watchdog, which each step's end feeds (None when the config turns it off)."""

    def __init__(
        self, *, step: int, updated: asyncio.Event, end: int | None = None, watchdog: Watchdog | None = None
    ) -> None:
        self.step = step
        self.end = end
        self.updated = updated
        self.watchdog = watchdog

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
        """The next step starts: the step counts one more; what the step consumed its groups from moves its horizon
        on (``consumed``); the evaluation loop hears of it, also after a step that published no new policy; and the
        watchdog is fed."""

        self.step += 1
        if consumed is not None:
            consumed.step_started()
        self.updated.set()
        if self.watchdog:
            self.watchdog.heartbeat()


class TrainerRun:
    """One start of ``trainer``, through the lifecycle every trainer shares (:meth:`execute`), with the given
    workflows (None: those the trainer's setup builds, when it rolls out).  What its hooks use: the config, the run
    directory, the policies, the workflows, the run's checkpoints
    (:class:`~platoon.train.checkpoints.participants.RunCheckpoints`), the shutdown request, where the start begins and
    where its steps end, the start's draws (:meth:`rollouts`), the launch save, and the step clock with its
    watchdog."""

    # Set as the phases run: the shutdown request (2), the start point (3), the plan (5), the paths of the launch save
    # by policy (6), the clock (7: start_clock, or the trainer's own), and whether an interrupt stopped the steps (8).
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
        self.checkpoints = RunCheckpoints(self.run_dir, self.policies, wandb_run_id=self._wandb_run_id)
        self.interrupted = False
        # The start's draws, once the trainer opened them (rollouts); None before, and for a trainer that rolls out
        # none.
        self.source: RolloutSource | None = None

    def _wandb_run_id(self) -> str | None:
        stats_logger = self.trainer._stats_logger
        return stats_logger.wandb_run_id if stats_logger is not None else None

    def rollouts(self) -> RolloutSource:
        """The start's training draws (:class:`~platoon.train.dispatch.source.RolloutSource`): opened at the start's
        point over the run's training tasks, ``train.*`` and the training workflow, and registered with the run's
        checkpoints, which hands it the start point, so that it takes up the draws the checkpoint left outstanding.
        Call it in ``prepare()``, once, and start it (``start``) as the steps begin; the run starts its per-task
        statistics after the launch save and closes them after the trainer's ``close``.  Only a trainer whose
        ``needs()`` say it rolls out may call it."""

        trainer, workflows = self.trainer, self.workflows
        if not trainer.needs(self.config, trainer.options).rollouts:
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
        return self._open_draws(source)

    def _open_draws(self, source: RolloutSource) -> RolloutSource:
        """Open ``source`` as the start's draws, at the start's point, and register it with the run's checkpoints, which
        hands it the start point: it takes up the draws the checkpoint left outstanding.  The run starts its per-task
        statistics after the launch save and closes them after the trainer's ``close``.  :meth:`rollouts` opens a
        trainer's draws through it, and the rl trainer its own (a source whose steps go through its methods)."""

        assert self.source is None, "a start opens its draws once"
        source.open(self.start)
        self.checkpoints.register(source)
        self.source = source
        return source

    def start_clock(self) -> StepClock:
        """Start the run's clock at the start's step, towards the plan's end (:meth:`StepClock.started`)."""

        self.clock = StepClock.started(self.config, step=self.start.step, end=self.plan.end_step)
        return self.clock

    async def execute(self) -> None:
        """The start's phases, in order (the module's docstring numbers them)."""

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
        await self._launch()  # 6
        await trainer.ready(self)  # 7
        await trainer.run_steps(self)  # 8
        await self._save_at_exit()  # 9
        # 10.  Only after steps that ended, not after a failure (RFC 0021, section 5, puts it in ``finally``).
        await trainer.close(self)
        if self.source is not None:
            self.source.close()

    def _built_workflows(self) -> Workflows | None:
        """0. The start's workflows, which the trainer's setup builds on its policies
        (:meth:`~platoon.train.trainer.api.TrainerSetup.build_workflows`), for a trainer that rolls out; None for one
        that rolls out none, or that was built outside ``platoon train``, without a setup."""

        trainer = self.trainer
        if trainer.setup is None or not trainer.needs(self.config, trainer.options).rollouts:
            return None
        return trainer.setup.build_workflows(self.policies)

    async def _launch(self) -> None:
        """6. Save the launch checkpoint, then start from it: the trainer's ``started``, and each policy's first
        sampling client, of the launch save's weights, which it samples at the checkpoint's policy version, with the
        checkpoint's optimizer steps.  The launch save is a *recovery* checkpoint: every (re)launch writes one, so
        naming it permanently would keep a ~490 GB checkpoint per restart."""

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

    async def _save_at_exit(self) -> None:
        """9. Once the steps stop: an interrupted start that trained a step saves a permanent checkpoint to resume from
        (``<step>_interrupted``; one that trained none saves nothing), and a start that trained to the plan's end the
        final one; a start that had no step left to train says why (the trainer's ``_nothing_to_train``).  Steps that a
        stop request stopped short of the plan's end were interrupted, also when the trainer did not say so
        (``run.interrupted``): a final row would claim steps they never trained."""

        step, first, end = self.clock.step, self.start.step, self.plan.end_step
        leader = self.policies[self.policies.leader]
        if self.interrupted or (step < end and self.shutdown.is_set()):
            if step > first:
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


def _first_sampler(policy: Policy, saved: dict[str, str], start: PolicyStart) -> None:
    """The policy's first sampling client, of the sampler weights the launch save wrote (``saved``), which the policy
    samples at the version its start point says, without counting a new one, and its optimizer steps so far."""

    # Not policy.checkpoints.sampler_from, whose log lines the launch never wrote.
    sampling_client = policy.training_client.create_sampling_client(saved["sampler_path"])
    policy.info.llm.set_version(start.version)
    policy.info.llm.update_sampling_client(sampling_client, increment_version=False)
    policy.sampling_client = sampling_client
    policy.optimizer_steps = start.optimizer_steps
