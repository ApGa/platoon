"""A start's dispatcher: built over the run's task samplers and group admission rules, and started in the start's
task group, its groups run by :mod:`platoon.train.dispatch.runner`.

Each function here takes the start's rollout source (:class:`~platoon.train.dispatch.source.RolloutSource`), whose
parts it reads and sets: the source calls them (its ``new_dispatcher`` and ``start``), and so do the rl trainer's
methods of the same names, with its source.
"""

import asyncio
import logging
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any

from platoon.train import entry_settings
from platoon.train.admissions import AdmissionSet, choose_admissions, rule_mode
from platoon.train.checkpoints.pruning import read_checkpoint_records
from platoon.train.dataset_order import PRESERVE_DATASET_ORDER_COLUMN
from platoon.train.dispatch import runner
from platoon.train.dispatch.dispatcher import BoundDraw, Dispatcher
from platoon.train.dispatch.permits import DrawLease, _rows
from platoon.train.labels import dataset_labels, label_column
from platoon.train.sampler_runtime import SamplerSet, TaskPool
from platoon.train.sampling import sampling_fingerprint
from platoon.train.task_stats import next_start
from platoon.train.tinker.rollout_metrics import monitor as client_monitor
from platoon.train.workflow.protocol import capabilities_of

if TYPE_CHECKING:
    from platoon.train.dispatch.source import RolloutSource

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


def _dispatch(source: "RolloutSource", task_group: asyncio.TaskGroup, *, stop: asyncio.Event | None) -> Dispatcher:
    """Start the source's dispatcher (:mod:`platoon.train.dispatch.dispatcher`) in ``task_group``: the one built at
    launch, with what the checkpoint it resumes from saved.  Its groups settle into the source's ``results``, the
    groups a resume kept first, each run by :func:`~platoon.train.dispatch.runner._run_training_group`; it starts none
    once ``stop`` is set."""

    dispatcher = source.dispatcher
    assert dispatcher is not None and source.results is not None  # the start opened it; start() made the queue
    dispatcher.bind(source.results)
    dispatcher.rolling = capabilities_of(source.workflow).runs_rollouts_on_lease  # rolling starts
    if dispatcher.permit_release == "rollout":
        logger.info(
            "Each rollout returns its permit as it ends (train.permit_release: rollout); a group's rollouts start %s",
            "as permits free" if dispatcher.rolling else "together (its workflow sets no runs_rollouts_on_lease)",
        )

    async def run_group(draw: BoundDraw, lease: DrawLease) -> None:
        await runner._run_training_group(source, draw, lease)

    def spawn(coroutine: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        return task_group.create_task(coroutine)

    task_group.create_task(dispatcher.run(spawn=spawn, run_group=run_group, stop=stop), name="train_dispatcher")
    return dispatcher


def _new_dispatcher(source: "RolloutSource", train_dataloader: Any, start_batch: int, end_batch: int) -> Dispatcher:
    """The dispatcher of this start's training groups, from ``start_batch`` to ``end_batch``, over the run's order
    (:func:`_samplers`), starting at the checkpoint's batch; and, for a run under a group admission rule, its rules
    beside it (the source's ``admissions``, :class:`~platoon.train.admissions.AdmissionSet`), whose cap the dispatcher
    counts its steps by."""

    config = source.config
    batch_size = config.train.batch_size
    tasks = source.training_tasks()
    column = label_column(config)
    recorder = source.recorder
    rows, _ = _rows(config, "train", config.train.max_concurrent_rollouts)
    rows.permits.on_held = client_monitor("train").set_permits_held  # rollout_client/permits_held_mean
    sampler = str(getattr(config.train, "task_sampler", None) or "auto")
    samplers = _samplers(source, train_dataloader, sampler)
    if recorder is not None:
        samplers.attach(recorder)  # adaptive samplers read the statistics, and observe settled groups
    # The group admission rules (RFC 0019, section 6), resolved as platoon train resolved them, whose resolution
    # ran each rule's check with the run's workflow (run_components): none in a run whose every rule is all, which
    # trains as before.
    choices = choose_admissions(config, check=False) if rule_mode(config) else None
    source.admissions = (
        AdmissionSet(choices, samplers=samplers, batch_size=batch_size, label_column=column)
        if choices is not None and choices.active
        else None
    )
    try:
        fingerprint = sampling_fingerprint(config, tasks or [], source.tasks.train_sampling, admission=choices)
    except Exception:  # what a resume compares never stops training
        logger.warning("Could not take the fingerprint of the run's sampling inputs", exc_info=True)
        fingerprint = {}
    if recorder is not None:
        start = recorder.start
    else:  # the statistics did not start: number this start's draws as they would have
        start = next_start(read_checkpoint_records(source.run_dir))
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
        versions=source.versions,
        leader=source.leader,
        finish_batches=getattr(config.train, "num_minibatches", 1) > 1,
        sampler=sampler,
        fingerprint=fingerprint,
        step=start_batch,
        groups_per_step=source.admissions.cap if source.admissions is not None else None,
        permit_release=config.train.permit_release,
    )


def _samplers(source: "RolloutSource", train_dataloader: Any, sampler: str) -> SamplerSet:
    """The task samplers the run draws through: those of the source's ``train_sampling``, as platoon train hands
    them the trainer; else one that draws each position of the dataloader's order (its batch ``b`` is positions ``b ×
    batch_size`` on) as it is, over the training tasks (``training_tasks()``), whose first row of a task a resume
    draws again."""

    config = source.config
    batch_size = config.train.batch_size
    column = label_column(config)
    sampling = source.tasks.train_sampling
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

    tasks = source.training_tasks()
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
