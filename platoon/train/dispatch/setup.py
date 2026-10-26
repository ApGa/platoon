"""A start's dispatcher: built over the run's task samplers and group admission rules, and started in the start's
task group.

The functions here that take the trainer first are its methods still: ``PlatoonTinkerRLTrainer`` keeps each under its
name, so to replace one, replace the method, on the class or an instance, not the function.
"""

import asyncio
import logging
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any

from platoon.train import entry_settings
from platoon.train.admissions import AdmissionSet, choose_admissions, rule_mode
from platoon.train.checkpoints.pruning import read_checkpoint_records
from platoon.train.dataset_order import PRESERVE_DATASET_ORDER_COLUMN
from platoon.train.dispatch.dispatcher import BoundDraw, Dispatcher
from platoon.train.dispatch.permits import DrawLease, _rows
from platoon.train.labels import dataset_labels, label_column
from platoon.train.sampler_runtime import SamplerSet, TaskPool
from platoon.train.sampling import sampling_fingerprint
from platoon.train.task_stats import next_start
from platoon.train.tinker.rollout_metrics import monitor as client_monitor
from platoon.train.trainer.state import TrainLoopSharedState
from platoon.train.workflow.protocol import RolloutWorkflow, capabilities_of

if TYPE_CHECKING:
    from platoon.train.trainer.rl import PlatoonTinkerRLTrainer

logger = logging.getLogger(__name__)


def _named_samplers(config: Any) -> list[str]:
    """The task samplers ``config`` names other than the ones a dataloader's order draws as (``auto``, ``weighted``,
    ``uniform``, and an entry's ``uniform``), one line each."""

    from platoon.train.samplers import STATIC_RUN_SAMPLERS

    named = []
    run = str(getattr(getattr(config, "train", None), "task_sampler", None) or "auto")
    if run not in STATIC_RUN_SAMPLERS:
        named.append(f"train.task_sampler: {run}")
    for environment in getattr(config, "environments", None) or []:
        own = getattr(environment, "task_sampler", None)
        if (own is not None and own != "uniform") or getattr(environment, "task_sampler_options", None):
            named.append(f"environments[{environment.label}].task_sampler: {own or 'uniform'}")
    return named


def _dispatch(
    trainer: "PlatoonTinkerRLTrainer",
    task_group: asyncio.TaskGroup,
    shared_state: TrainLoopSharedState,
    start_batch: int,
    end_batch: int,
    train_workflow: RolloutWorkflow,
    results: asyncio.Queue,
) -> Dispatcher | None:
    """Start the dispatcher of this start's training groups (:mod:`platoon.train.dispatch.dispatcher`) in
    ``task_group``: the one built at launch (:meth:`_new_dispatcher`), with what the checkpoint it resumes from
    saved, else a new one.  Its groups settle into ``results``, the groups a resume kept first.  The train loop's
    tests put their groups in ``results`` in its place, and return None: the loop then tells no dispatcher of its
    steps."""

    dispatcher = getattr(trainer, "_dispatcher", None)
    if dispatcher is None:
        dispatcher = trainer._dispatcher = trainer._new_dispatcher(
            shared_state.train_dataloader, start_batch, end_batch
        )
    dispatcher.bind(results)
    dispatcher.rolling = capabilities_of(train_workflow).runs_rollouts_on_lease  # rolling starts
    if dispatcher.permit_release == "rollout":
        logger.info(
            "Each rollout returns its permit as it ends (train.permit_release: rollout); a group's rollouts start %s",
            "as permits free" if dispatcher.rolling else "together (its workflow sets no runs_rollouts_on_lease)",
        )

    async def run_group(draw: BoundDraw, lease: DrawLease) -> None:
        await trainer._run_training_group(shared_state, train_workflow, dispatcher, draw, lease)

    def spawn(coroutine: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        return task_group.create_task(coroutine)

    task_group.create_task(
        dispatcher.run(spawn=spawn, run_group=run_group, stop=shared_state.shutdown_event),
        name="train_dispatcher",
    )
    return dispatcher


def _new_dispatcher(
    trainer: "PlatoonTinkerRLTrainer", train_dataloader: Any, start_batch: int, end_batch: int
) -> Dispatcher:
    """The dispatcher of this start's training groups, from ``start_batch`` to ``end_batch``, over the run's order
    (:meth:`_samplers`), starting at the checkpoint's batch; and, for a run under a group admission rule, its rules
    beside it (``_admissions``, :class:`~platoon.train.admissions.AdmissionSet`), whose cap the dispatcher counts
    its steps by."""

    config = trainer.config
    batch_size = config.train.batch_size
    tasks = trainer._training_tasks()
    column = label_column(config)
    recorder = getattr(trainer, "_recorder", None)
    rows, _ = _rows(config, "train", config.train.max_concurrent_rollouts)
    rows.permits.on_held = client_monitor("train").set_permits_held  # rollout_client/permits_held_mean
    sampler = str(getattr(config.train, "task_sampler", None) or "auto")
    samplers = trainer._samplers(train_dataloader, sampler)
    if recorder is not None:
        samplers.attach(recorder)  # adaptive samplers read the statistics, and observe settled groups
    # The group admission rules (RFC 0019, section 6), resolved as platoon train resolved them, whose resolution
    # ran each rule's check with the run's workflow (run_components): none in a run whose every rule is all, which
    # trains as before.
    choices = choose_admissions(config, check=False) if rule_mode(config) else None
    trainer._admissions = (
        AdmissionSet(choices, samplers=samplers, batch_size=batch_size, label_column=column)
        if choices is not None and choices.active
        else None
    )
    try:
        fingerprint = sampling_fingerprint(
            config, tasks or [], getattr(trainer, "train_sampling", None), admission=choices
        )
    except Exception:  # what a resume compares never stops training
        logger.warning("Could not take the fingerprint of the run's sampling inputs", exc_info=True)
        fingerprint = {}
    if recorder is not None:
        start = recorder.start
    else:  # the statistics did not start: number this start's draws as they would have
        start = next_start(read_checkpoint_records(trainer.run_log_path))
    return Dispatcher(
        rows=rows,
        samplers=samplers,
        batch_size=batch_size,
        max_staleness=config.train.max_staleness,
        steps=max(0, end_batch - start_batch),
        limits=getattr(config.train, "rollout_group_limits", None) or {},
        labels=dataset_labels(tasks, column) if tasks is not None else (),
        start=start,
        next_position=start_batch * batch_size,
        recorder=recorder,
        version=lambda: trainer.model_info.llm.version,
        finish_batches=getattr(config.train, "num_minibatches", 1) > 1,
        sampler=sampler,
        fingerprint=fingerprint,
        step=start_batch,
        groups_per_step=trainer._admissions.cap if trainer._admissions is not None else None,
        permit_release=config.train.permit_release,
    )


def _samplers(trainer: "PlatoonTinkerRLTrainer", train_dataloader: Any, sampler: str) -> SamplerSet:
    """The task samplers the run draws through: those of ``train_sampling``, as platoon train hands them the
    trainer; else one that draws each position of the dataloader's order (its batch ``b`` is positions ``b ×
    batch_size`` on) as it is, over the training tasks (:meth:`_training_tasks`), whose first row of a task a
    resume draws again."""

    config = trainer.config
    batch_size = config.train.batch_size
    column = label_column(config)
    sampling = getattr(trainer, "train_sampling", None)
    if sampling is not None:
        return SamplerSet.build(sampling, batch_size=batch_size, label_column=column)
    if named := _named_samplers(config):
        raise ValueError(
            f"{'; '.join(named)}: a trainer given no train_sampling draws its dataloader's order, so it runs no "
            "other task sampler: entry and adaptive samplers need AutoDataset.training's sampling, which platoon "
            "train hands the trainer (train_sampling=AutoDataset.training(config).sampling)"
        )
    batches: dict[int, list[dict]] = {}

    def order(position: int) -> dict:
        batch, index = divmod(position, batch_size)
        if batch not in batches:
            batches.clear()  # the dispatcher reads the order forward: one batch at a time
            batches[batch] = train_dataloader.get_batch(batch)
        return batches[batch][index]

    tasks = trainer._training_tasks()
    entries = tuple(entry_settings.run_settings(config))
    drop = (PRESERVE_DATASET_ORDER_COLUMN,)
    pool = None
    try:
        if hasattr(tasks, "column_names"):  # a dataset: its rows are converted only when a resume reads them
            pool = TaskPool.from_dataset(tasks, entries=entries, column=column, drop_columns=drop)
        elif tasks is not None:
            pool = TaskPool.from_rows(tasks, entries=entries, column=column, drop_columns=drop)
    except Exception:  # a resume then cannot draw the outstanding draws again, as with no training tasks
        logger.warning("Could not list the training tasks that a resume draws again", exc_info=True)
    return SamplerSet.of_order(order, pool=pool, name=sampler, batch_size=batch_size, label_column=column)
