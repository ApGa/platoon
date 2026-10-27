"""The sft trainer (``trainer: sft``, :class:`SupervisedTrainer`): supervised fine-tuning of the run's policy on chat
datasets in LLaMA-Factory's conventions (:mod:`platoon.train.supervised`).

A start goes through the lifecycle every trainer shares (:class:`~platoon.train.trainer.run.TrainerRun`), which calls
the trainer's hooks in this order:

- ``check``: load the conversations ``trainer_options`` names, and render those that show what they ask of the
  policy's renderer (the first, the first with tools, the first with a tool call and the first with neither a system
  prompt nor tools), so that a dataset or a renderer it cannot train with stops the run before any service opens;
  then the cursor that walks the conversations takes part in the run's checkpoints, and takes up where the checkpoint
  the start resumes from left them;
- ``prepare``: the step the run ends at, counted on from the cursor, and the steps of an epoch, by which
  ``checkpoint.strategy: epoch`` counts;
- ``started``: the pruning of the run's recovery checkpoints starts with the launch save;
- ``ready``: the run's clock starts, and the policy's learner, under the cross-entropy objective;
- ``run_steps``: the steps (:meth:`SupervisedTrainer.step`), until the plan's end or a stop request, which a step
  never cuts short; each ends with ``run.end_step``;
- ``close``: the pruning stops.

It rolls nothing out and samples nothing: no workflow, dispatcher or sampler, and no evaluation during training.  Its
steps are atomic and its updates fail closed (:meth:`SupervisedTrainer.needs`).  The policy publishes a sampler only
for a step hook (one a subclass registers in ``check``), once a step, since nothing else samples it: without one, the
policy's version, which counts the samplers a run publishes, stays where the start began.  Its optimizer steps count
the updates.
"""

import logging
import math
import time
from typing import Any

import tinker
from tinker_cookbook.renderers import Renderer

from platoon.train.checkpoints import pruning
from platoon.train.checkpoints.resume import StartPoint
from platoon.train.objectives.cross_entropy import CrossEntropy, mean_per_token, step_nll, trained_tokens
from platoon.train.policy.learner import Learner
from platoon.train.policy.policy import Policy
from platoon.train.supervised.cursor import DatasetCursor, last_step, steps_per_epoch
from platoon.train.supervised.formats import Conversation
from platoon.train.supervised.loading import SupervisedDataset, load_conversations
from platoon.train.supervised.options import SupervisedOptions
from platoon.train.supervised.rendering import CannotRender, check_renders, probes, supervised_datum
from platoon.train.trainer.api import StepPlan, TrainerBase, TrainerNeeds
from platoon.train.trainer.run import TrainerRun
from platoon.utils.config import ConfigError

logger = logging.getLogger(__name__)


class SupervisedTrainer(TrainerBase):
    """Fine-tunes the run's policy on the conversations of the datasets its options name (``trainer_options``,
    :class:`~platoon.train.supervised.options.SupervisedOptions`): each step is one optimizer step over the next
    ``batch_size`` of them, whose loss is the mean negative log-likelihood of the model's turns."""

    options: SupervisedOptions
    # Set as the hooks run: the conversations, the cursor, the datasets a warning named for a conversation with no
    # token to train, and, until a conversation of the start trains, those the start walked with none (check; None
    # once one trained); the pruner of recovery checkpoints (started; None without checkpoint.keep_recovery); and the
    # learner (ready).
    _dataset: SupervisedDataset
    _cursor: DatasetCursor
    _untrained: set[str]
    _walked_untrained: set[int] | None
    # Quoted, and the class read through its module where it is used: tests/test_moved_modules.py imports this module
    # while it patches the class through its old path (platoon.train.tinker.checkpoint_pruning).
    _pruner: "pruning.RecoveryCheckpointPruner | None" = None
    _learner: Learner

    @classmethod
    def needs(cls, config: Any, options: Any) -> TrainerNeeds:
        """It rolls nothing out.  Its steps are atomic and its updates fail closed (RFC 0021, rule 10): an interrupt
        inside a step that sent its update saves no checkpoint of the step, and an optimizer step that the service
        skipped, or whose metrics are not finite, stops the run before it counts."""

        return TrainerNeeds(rollouts=False, atomic_steps=True, fail_closed_updates=True)

    # --- the lifecycle's hooks, in the order they run ----------------------------------------------------------------

    def check(self, run: TrainerRun) -> None:
        """Load the conversations, and render those that show what they ask of the renderer; the cursor takes part in
        the run's checkpoints."""

        self._dataset = load_conversations(self.options)
        self._untrained, self._walked_untrained = set(), set()
        for conversation in probes(self._dataset.conversations):
            try:
                check_renders(self._renderer(), conversation)
            except CannotRender as exc:
                raise self._cannot_render(conversation, exc) from exc
        self._cursor = DatasetCursor(
            len(self._dataset.conversations), seed=self.options.seed, fingerprint=self._dataset.fingerprint
        )
        run.checkpoints.register(self._cursor)

    async def prepare(self, run: TrainerRun, start: StartPoint) -> StepPlan:
        """The run ends after ``num_epochs`` passes over the conversations, counted on from where the cursor took up,
        or at ``max_training_steps``; a permanent checkpoint comes every ``checkpoint.every`` steps (or epochs, under
        ``strategy: epoch``), which the steps' ends save (``run.end_step``)."""

        conversations = len(self._dataset.conversations)
        epoch = steps_per_epoch(conversations, self.options.batch_size)
        end = last_step(self.options, self._cursor, start.step)
        logger.info(f"Training on {conversations} conversations, {epoch} steps an epoch, to step {end}")
        return StepPlan(end_step=end, steps_per_epoch=epoch)

    async def started(self, run: TrainerRun) -> None:
        """The launch save is a recovery checkpoint: the pruning of the run's recovery checkpoints starts with it."""

        self._pruner = self._start_pruning()
        self._recovery_saved(run.launch_save[self._policy().id], run.start.policy_version)

    async def ready(self, run: TrainerRun) -> None:
        """The run's clock starts; the policy's learner (``run.learner``) updates it under cross-entropy, each result
        awaited under its ``train.*`` deadline (``run.wait``)."""

        await super().ready(run)
        self._learner = run.learner(self.policies.leader, objective=CrossEntropy())

    async def run_steps(self, run: TrainerRun) -> None:
        """The steps, to the plan's end, or to the step boundary after a stop request."""

        while run.clock.step < run.plan.end_step and not run.shutdown.is_set():
            await self.step(run)

    async def close(self, run: TrainerRun) -> None:
        """The pruning stops, after waiting a little for the deletes in flight; also after a failure."""

        if self._pruner is not None:
            await self._pruner.aclose()

    # --- a step ------------------------------------------------------------------------------------------------------

    async def step(self, run: TrainerRun) -> None:
        """One step: the next ``batch_size`` conversations, rendered; one optimizer step over those with a token to
        train, under the mean over their trained tokens; then the step's end (``run.end_step``): its step hooks, if
        any, the checkpoint the cadence asks for, and the clock's advance, the policy publishing a sampler only for a
        step hook; then the step's metrics are logged.  A start that walked every conversation without one to train
        stops (:meth:`_refuse_if_none_trains`)."""

        began = time.perf_counter()
        indices = self._cursor.take(self.options.batch_size)
        conversations = [self._dataset.conversations[index] for index in indices]
        datums = [self._render(conversation) for conversation in conversations]
        trained = self._with_tokens_to_train(conversations, datums)
        if trained:
            self._walked_untrained = None
            await self._update(mean_per_token(trained))
        else:
            self._refuse_if_none_trains(indices)
            logger.warning(
                "Step %d takes no optimizer step: none of its %d conversations has a token to train within "
                "trainer_options.max_length",
                run.clock.step,
                len(datums),
            )
        await run.end_step(sampled=False)
        self._log(run, datums, trained, time.perf_counter() - began)

    def _render(self, conversation: Conversation) -> tinker.Datum:
        """The datum ``conversation`` trains on, rendered with the policy's renderer."""

        try:
            return supervised_datum(
                self._renderer(),
                conversation,
                mask_history=self.options.mask_history,
                max_length=self.options.max_length,
            )
        except CannotRender as exc:
            raise self._cannot_render(conversation, exc) from exc

    def _with_tokens_to_train(
        self, conversations: list[Conversation], datums: list[tinker.Datum]
    ) -> list[tinker.Datum]:
        """The datums with a token to train.  A conversation with none left within ``max_length`` is left out; the
        first of each dataset is named in a warning, and ``train/untrained_conversations`` counts each step's."""

        trained = []
        for conversation, datum in zip(conversations, datums, strict=True):
            if trained_tokens(datum) > 0:
                trained.append(datum)
            elif conversation.dataset not in self._untrained:
                self._untrained.add(conversation.dataset)
                logger.warning(
                    "Dataset %s: %s has no token to train within trainer_options.max_length: %s, and is left out of "
                    "its step, as each such conversation is (train/untrained_conversations counts them)",
                    conversation.dataset,
                    conversation.origin,
                    self.options.max_length,
                )
        return trained

    def _refuse_if_none_trains(self, indices: list[int]) -> None:
        """Raise :class:`ConfigError` once the start walked every conversation, and none had a token to train within
        ``max_length``: the run would end having trained nothing."""

        if self._walked_untrained is None:
            return
        self._walked_untrained.update(indices)
        if len(self._walked_untrained) == len(self._dataset.conversations):
            raise ConfigError(
                f"trainer_options.max_length: {self.options.max_length} leaves none of the "
                f"{len(self._dataset.conversations)} conversations a token to train, so the run would train nothing: "
                "raise it, or leave it out"
            )

    async def _update(self, datums: list[tinker.Datum]) -> None:
        """One optimizer step over ``datums``, sent in ``num_microbatches`` forward_backward calls."""

        update = self._learner.update()
        size = math.ceil(len(datums) / self.options.num_microbatches)
        for number, start in enumerate(range(0, len(datums), size)):
            microbatch = datums[start : start + size]
            # Cross-entropy reads no side channel: the datums it trains on are those its metrics read.
            await update.submit(microbatch, originals=microbatch, minibatch=0, microbatch=number)
        await update.apply()

    def _log(self, run: TrainerRun, datums: list[tinker.Datum], trained: list[tinker.Datum], seconds: float) -> None:
        """The step's metrics: the learner's (``train/nll``, ``optim/*``), the run's progress, the step's
        conversations and tokens, its time and the pruner's; then the step's line."""

        step, end = run.clock.step, run.plan.end_step
        stats = self.train_tracker.export(reset=True)
        if (nll := step_nll(stats, scope=self.train_tracker.name)) is not None:
            stats["train/nll"] = nll
        stats["train/step"] = step
        stats["train/epoch"] = self._cursor.epoch
        stats["progress/done_frac"] = step / end
        stats["optim/lr"] = self._policy().optimizer_params.learning_rate
        stats["train/conversations"] = len(datums)
        stats["train/untrained_conversations"] = len(datums) - len(trained)
        stats["train/tokens"] = sum(datum.model_input.length for datum in datums)
        stats["train/trained_tokens"] = sum(trained_tokens(datum) for datum in trained)
        stats["timing/batch_total"] = seconds
        if self._pruner is not None:
            stats.update(self._pruner.step_metrics())
        self.stats_logger.log(step=step, stats=stats)
        logger.info(f"Step {step}/{end} | Epoch {self._cursor.epoch} | Batch time: {seconds:.1f}s")

    # --- what the hooks share ----------------------------------------------------------------------------------------

    def _policy(self) -> Policy:
        """The policy the trainer trains: the run's one, its leader."""

        return self.policies[self.policies.leader]

    def _renderer(self) -> Renderer:
        """The policy's renderer, which renders each conversation as the policy sees it when it samples."""

        return self._policy().info.llm.renderer

    def _cannot_render(self, conversation: Conversation, exc: CannotRender) -> ConfigError:
        """The error of a renderer that cannot render ``conversation`` for training, naming the conversation and the
        kind of renderer it needs."""

        tool_calling = bool(conversation.tools) or any(turn.calls for turn in conversation.turns)
        needed = "with tool calling, such as qwen3" if tool_calling else "which renders message by message"
        return ConfigError(
            f"train.renderer_name: {self.config.train.renderer_name} cannot render {conversation.origin} for training "
            f"({exc}); trainer: sft needs a tinker-cookbook renderer {needed}"
        )

    def _start_pruning(self) -> "pruning.RecoveryCheckpointPruner | None":
        """The pruner of the run's recovery checkpoints (``checkpoint.keep_recovery``), on a Tinker session of its own;
        None without ``keep_recovery``.  No rollout samples the policy, so it keeps no checkpoint for one."""

        keep = self.config.checkpoint.keep_recovery
        if keep is None:
            return None
        base_url = self._policy().spec.tinker_base_url
        return pruning.RecoveryCheckpointPruner(
            run_dir=self.run_log_path,
            keep=keep,
            max_staleness=None,
            service_client_factory=lambda: tinker.ServiceClient(base_url=base_url),
        )

    def _recovery_saved(self, paths: dict[str, str], policy_version: int) -> None:
        """A recovery checkpoint that ``checkpoints.jsonl`` now records goes to the pruner, if any."""

        if self._pruner is not None:
            self._pruner.recovery_saved(paths, policy_version)
