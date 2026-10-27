"""``platoon train``'s start of training: :func:`arun_tinker_training` loads and checks the config, lists the tasks,
builds the run's components from its environments (the Auto factories of :mod:`platoon.train.auto`) and trains on the
tasks with the trainer (:mod:`platoon.train.trainer`), in the steps that follow it, whose checks stop a config that
cannot train with a :class:`ConfigError` before Tinker opens.  :mod:`platoon.train.supervise` runs it, in its own
process or in each supervised start; so does ``python -m platoon.train.launch --config <file>``."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from platoon.options import Phase
from platoon.train import entry_settings
from platoon.train.auto import (
    AutoDataset,
    AutoEnvironment,
    AutoRewardProcessor,
    AutoRollout,
    AutoTaskLoader,
    AutoWorkflow,
    TrainingTasks,
    check_environment_routes,
    check_harness_toolchains,
    check_mixture_pools,
)
from platoon.train.components import EnvironmentConfig
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig, evaluates, training_problems
from platoon.train.exclusion import TaskExclusion, record_exclusions
from platoon.train.identity import check_training_run
from platoon.train.labels import check_labels, label_column
from platoon.train.policy.spec import MAIN
from platoon.train.sampling import log_sampling_plan
from platoon.train.startup import worker_preload
from platoon.train.tinker.proxy import ModelInfo
from platoon.train.trainer.rl import PlatoonTinkerRLTrainer
from platoon.train.workflow.group import GroupRolloutWorkflow
from platoon.train.workflow.protocol import WorkflowSetup, _built_by_keywords, capabilities_of
from platoon.utils.config import ConfigError, load_config, update_run_lock

logger = logging.getLogger(__name__)


async def arun_tinker_training(args: list[str] | None = None, default_config_path: str | None = None) -> None:
    """Run Tinker training from an environment-backed Platoon config.

    A config that loads but cannot train (:func:`training_problems`), a missing ``TINKER_API_KEY``, or, with
    ``train.sample_timeout_seconds`` unset, a ``TINKER_SAMPLE_TIMEOUT_SECONDS`` that is not a positive number raises
    :class:`ConfigError` before any environment package is imported, task listed or Tinker session opened; an
    environment whose harness settings cannot run or could never train (``check_environment_routes``), before any
    task is listed; an ``exclude_tasks`` item that names no task, while the tasks are listed, before Tinker is
    opened; a label column (``train.label_by``) that no training task has, a label's straggler quorum above the group
    size of an entry whose tasks carry it, and a resume that would change the run's model or held-out tasks
    (:mod:`platoon.train.identity`), once they are listed; and an entry's own training settings beside a workflow
    class that does not apply them, before the trainer is built.  What the exclusions left out, and what each entry's
    own settings change from the run's (``entry_settings``), go into the run directory's ``lock.json``.

    The steps, in that order: :func:`_load_and_check`, :func:`_list_tasks`, :func:`_check_listed`,
    :func:`_record_run` and :func:`_build_components` before the trainer is built, then :func:`_build_workflows` in
    the trainer's context, on the trainer's policy, and the trainer trains.
    """

    config, settings, environment = _load_and_check(args, default_config_path)
    training, eval_dataset, exclusions = _list_tasks(config)
    _check_listed(config, settings, training, eval_dataset)
    _record_run(config, settings, exclusions)
    components = _build_components(config, environment)

    trainer = PlatoonTinkerRLTrainer(
        config=config,
        train_dataset=training.dataset,  # a weighted order's first epoch, or the listing the dataloader shuffles
        eval_dataset=eval_dataset,
        train_order=training.order,  # the epochs the trainer counts
        train_sampling=training.sampling,  # the task samplers it draws through, which draw the order's rows
    )
    async with trainer:
        train_workflow, eval_workflow = _build_workflows(
            config, components, trainer.model_info, settings, training, eval_dataset
        )
        await trainer.train(train_workflow=train_workflow, eval_workflow=eval_workflow)


def _load_and_check(
    args: list[str] | None, default_config_path: str | None
) -> tuple[PlatoonTinkerRLTrainerConfig, dict[str, entry_settings.EntrySettings], EnvironmentConfig]:
    """The run's config, loaded and checked before any environment package is imported, task listed or Tinker session
    opened; each entry's own training settings, laid over the run's; and the run's environment, once the environments'
    packages are imported and their mixture pools, routes and harness toolchains checked, before any task is listed."""

    config, _ = load_config(
        args=args or sys.argv[1:],
        config_class=PlatoonTinkerRLTrainerConfig,
        default_config_path=default_config_path,
    )
    config: PlatoonTinkerRLTrainerConfig = config
    if problems := training_problems(config):
        raise ConfigError(problems)
    from platoon.eval.config_defs import require_tinker_api_key, sample_timeout_note  # the served policy's checks

    require_tinker_api_key()
    # The bound on each sampling call and where it came from, in the log; a variable it falls back to that is not a
    # positive number stops the run here.
    logger.info(sample_timeout_note("train.sample_timeout_seconds", config.train.sample_timeout_seconds))
    # Each entry's own training settings, laid over the run's (checked when the config loaded).
    settings = entry_settings.entry_settings(config)
    entry_settings.log_warnings(config, settings)
    AutoEnvironment.load(config)
    environment = AutoEnvironment.from_config(config)
    check_mixture_pools(AutoEnvironment.environments(config))  # before any env server is contacted
    check_environment_routes(AutoEnvironment.environments(config))  # before any task is listed
    check_harness_toolchains(AutoEnvironment.environments(config))  # a run never builds one
    return config, settings, environment


def _list_tasks(config: PlatoonTinkerRLTrainerConfig) -> tuple[TrainingTasks, Any, list[TaskExclusion]]:
    """The run's training tasks, as it draws them (:meth:`AutoDataset.training`), checked against the config, which
    may not train on that many, before Tinker opens; then its evaluation tasks, when it evaluates.  Each split leaves
    out what its environments' ``exclude_tasks`` name, which the exclusions record (an item that names no task stops
    the run while the tasks are listed)."""

    exclusions: list[TaskExclusion] = []
    training = AutoDataset.training(config, exclusions=exclusions)
    train_dataset = training.dataset  # a weighted order's first epoch, or the listing the dataloader shuffles
    log_sampling_plan(training.plan)  # each entry's share, and a warning for weights that leave tasks out or repeat
    num_train_tasks = len(train_dataset) if hasattr(train_dataset, "__len__") else None
    if problems := training_problems(config, num_train_tasks=num_train_tasks):
        raise ConfigError(problems)  # before the trainer opens Tinker
    # With evaluation off the eval split is never read, so it is not listed and need not exist.
    eval_dataset = AutoDataset.from_config(config, "eval", exclusions=exclusions) if evaluates(config) else None
    return training, eval_dataset, exclusions


def _check_listed(
    config: PlatoonTinkerRLTrainerConfig,
    settings: dict[str, entry_settings.EntrySettings],
    training: TrainingTasks,
    eval_dataset: Any,
) -> None:
    """The checks that need the listed tasks, before Tinker opens: a label column no training task has, a label's
    straggler quorum above the group size of an entry whose tasks carry it, and a resume that would change the run's
    model or held-out tasks stop the run; rollout permits that the groups of the labels with a limit can fill are
    warned of."""

    # A label column no training task has stops here (exit 78); every task training draws from counts, those a
    # weighted order's first epoch leaves out included.
    check_labels(config, training.tasks, eval_dataset)
    if problems := entry_settings.listed_straggler_problems(config, settings, training.tasks, eval_dataset):
        raise ConfigError(problems)  # a label's quorum above the group size of an entry whose tasks carry it
    for line in entry_settings.listed_permit_warnings(config, settings, training.tasks):
        logger.warning("%s", line)  # rollout permits that the groups of the labels with a limit can fill
    # One run per run directory: a resume with another model or other held-out tasks stops here (exit 78).
    check_training_run(
        config,
        Path(config.log_path, config.stats.experiment_name, config.stats.trial_name),
        training.tasks,  # every task training draws from, those a weighted order's first epoch leaves out included
        eval_dataset,
    )


def _record_run(
    config: PlatoonTinkerRLTrainerConfig,
    settings: dict[str, entry_settings.EntrySettings],
    exclusions: list[TaskExclusion],
) -> None:
    """What the run directory's ``lock.json`` records of this start: the tasks that the exclusions left out, and what
    each entry's own settings change from the run's."""

    # The run directory platoon train wrote lock.json into (PlatoonTinkerRLTrainer.run_log_path).
    run_dir = Path(config.log_path, config.stats.experiment_name, config.stats.trial_name)
    record_exclusions(run_dir, exclusions)
    if differences := entry_settings.lock_record(config, settings):
        update_run_lock(run_dir, {"entry_settings": differences})


@dataclass(frozen=True)
class _Components:
    """What the run's environments give training and evaluation to roll out with (the Auto factories of
    :mod:`platoon.train.auto`): each split's rollout function and task loader, the reward processor, the workflow
    class, and the arguments the environment gives each split's workflow."""

    rollout_fn: Any
    eval_rollout_fn: Any
    get_task_fn: Any
    eval_get_task_fn: Any  # its routes carry the entries' eval_overrides
    reward_processor: Any
    workflow_cls: Any
    workflow_kwargs: dict[str, Any]
    eval_workflow_kwargs: dict[str, Any]


def _build_components(config: PlatoonTinkerRLTrainerConfig, environment: EnvironmentConfig) -> _Components:
    """The run's components (:class:`_Components`); a workflow class that does not apply an entry's own settings stops
    the run, before the trainer is built."""

    components = _Components(
        rollout_fn=AutoRollout.from_config(config, "train"),
        eval_rollout_fn=AutoRollout.from_config(config, "eval"),
        get_task_fn=AutoTaskLoader.from_config(config, "train"),
        eval_get_task_fn=AutoTaskLoader.from_config(config, "eval"),
        reward_processor=AutoRewardProcessor.from_config(config),
        workflow_cls=AutoWorkflow.from_config(config, default=GroupRolloutWorkflow),
        workflow_kwargs=environment.workflow_kwargs,
        eval_workflow_kwargs=environment.eval_workflow_kwargs,
    )
    # A workflow class that does not apply an entry's own settings would ignore them without a word (exit 78).
    if problems := entry_settings.workflow_problems(config.environments, components.workflow_cls, environment.workflow):
        raise ConfigError(problems)
    return components


def _build_workflows(
    config: PlatoonTinkerRLTrainerConfig,
    components: _Components,
    model_info: ModelInfo,
    settings: dict[str, entry_settings.EntrySettings],
    training: TrainingTasks,
    eval_dataset: Any,
) -> tuple[Any, Any]:
    """The training and the evaluation workflow, of the run's workflow class, each built from its phase's
    :class:`~platoon.train.workflow.protocol.WorkflowSetup` (:func:`_workflow_setup`, :func:`_built`): each rolls out
    its split's tasks on the policy (``model_info``), into the run directory; under entries with settings of their
    own, each group runs with its entry's; and a workflow whose rollout workers fork from a fork server gets what that
    server imports and builds ahead of them (:func:`set_worker_preload`, over the training tasks and
    ``eval_dataset``)."""

    workflow_cls = components.workflow_cls
    group_kwargs = {**worker_package_kwargs(workflow_cls, config), **label_kwargs(workflow_cls, config)}
    setups = [
        _workflow_setup(phase, config, components, model_info, settings, group_kwargs) for phase in ("train", "eval")
    ]
    train_workflow = _built(workflow_cls, setups[0])
    eval_workflow = _built(workflow_cls, setups[1])
    # What the fork server of each workflow's rollout workers imports and builds ahead of them.
    environments = AutoEnvironment.environments(config)
    set_worker_preload(train_workflow, environments, phase="train", tasks=training.tasks)
    set_worker_preload(
        eval_workflow, environments, phase="eval", tasks=eval_dataset if eval_dataset is not None else ()
    )
    return train_workflow, eval_workflow


def _workflow_setup(
    phase: Phase,
    config: PlatoonTinkerRLTrainerConfig,
    components: _Components,
    model_info: ModelInfo,
    settings: dict[str, entry_settings.EntrySettings],
    group_kwargs: dict[str, Any],
) -> WorkflowSetup:
    """What ``phase``'s workflow is built with: the phase's rollout function, task loader and settings (``train`` or
    ``eval``), each entry's own when an entry sets any, the run's one policy (``model_info``), the run directory, the
    phase's stats scope and ``filter_errors`` (an environment's workflow arguments may set either), the reward
    processor, the environments' packages and the run's label column for a workflow that runs rollout workers
    (``group_kwargs``), and the environment's other workflow arguments."""

    training = phase == "train"
    section = config.train if training else config.eval
    options = {**group_kwargs, **(components.workflow_kwargs if training else components.eval_workflow_kwargs)}
    entry_configs = None
    if entry_settings.sets_settings(config.environments):  # each group runs with its entry's settings
        if training:
            entry_configs = {label: own.train for label, own in settings.items()}
        else:
            entry_configs = {label: own.eval for label, own in settings.items() if own.eval is not None}
    return WorkflowSetup(
        phase=phase,
        rollout_fn=components.rollout_fn if training else components.eval_rollout_fn,
        get_task_fn=components.get_task_fn if training else components.eval_get_task_fn,
        config=section.workflow_config,
        entry_configs=entry_configs,
        policies={MAIN: model_info},
        # The run directory, the trainer's run_log_path.
        log_path=os.path.join(config.log_path, config.stats.experiment_name, config.stats.trial_name),
        stats_scope=options.pop("stats_scope", phase),
        filter_errors=options.pop("filter_errors", section.workflow_config.filter_errors),
        reward_processor=components.reward_processor,
        packages=options.pop("packages", None),
        label_by=options.pop("label_by", None),
        options=options,
    )


def _built(workflow_cls: Any, setup: WorkflowSetup) -> Any:
    """The workflow of ``workflow_cls`` for ``setup``: its ``from_setup`` when it has one, as ``GroupRolloutWorkflow``
    does; else built as platoon train always built it, with ``GroupRolloutWorkflow``'s keyword arguments."""

    from_setup = getattr(workflow_cls, "from_setup", None)
    return from_setup(setup) if from_setup is not None else _built_by_keywords(workflow_cls, setup)


def worker_package_kwargs(workflow_cls: type, config: Any) -> dict[str, Any]:
    """The environments' packages (``AutoEnvironment.packages``) as workflow arguments, for a workflow whose
    rollout workers import them (``runs_rollout_workers``, as a ``GroupRolloutWorkflow``); nothing for any other
    workflow class."""

    if not capabilities_of(workflow_cls).runs_rollout_workers:
        return {}
    return {"packages": AutoEnvironment.packages(config)}


def set_worker_preload(workflow: Any, environments: Any, *, phase: Phase, tasks: Any) -> None:
    """Give a workflow whose rollouts run in workers forked from a fork server (one that runs rollout workers, as a
    ``GroupRolloutWorkflow`` does, with ``rollout_isolation: process`` and ``worker_start_method: forkserver``) what
    that server imports and builds ahead of them: what ``phase``'s rollouts of ``environments`` import and build when
    they open, and its settings' ``worker_preload`` (:func:`~platoon.train.startup.worker_preload`).  ``tasks`` are
    the phase's task rows.  Nothing for any other workflow, nor under ``worker_python_env_isolation``, whose rollouts
    import their environments and harnesses inside a Python overlay of their own (its interpreter, ``PATH`` and
    packages), which an import in the server would not see; a ``worker_preload`` that has no effect is warned about."""

    if not capabilities_of(workflow).runs_rollout_workers:
        return
    settings = workflow.config
    extra = tuple(getattr(settings, "worker_preload", ()) or ())
    rollouts = "training" if phase == "train" else "evaluation"
    if (
        getattr(settings, "rollout_isolation", "inprocess") != "process"
        or getattr(settings, "worker_start_method", "forkserver") != "forkserver"
    ):
        if extra:
            logger.warning(
                "workflow_config.worker_preload has no effect on the %s rollouts: only rollout workers forked from a "
                "fork server import it ahead (rollout_isolation: process, worker_start_method: forkserver)",
                rollouts,
            )
        return
    if getattr(settings, "worker_python_env_isolation", False):
        if extra:
            logger.warning(
                "workflow_config.worker_preload has no effect on the %s rollouts: under worker_python_env_isolation "
                "each rollout imports its environment and harness inside its own Python overlay",
                rollouts,
            )
        return
    workflow.worker_preload = worker_preload(environments, phase=phase, tasks=tasks, extra=extra)


def label_kwargs(workflow_cls: type, config: Any) -> dict[str, Any]:
    """The run's label column (``train.label_by``) as a workflow argument, for a workflow whose straggler policies key
    on it (``runs_rollout_workers``, as a ``GroupRolloutWorkflow``); nothing for any other workflow class."""

    if not capabilities_of(workflow_cls).runs_rollout_workers:
        return {}
    return {"label_by": label_column(config)}


def run_tinker_training(args: list[str] | None = None, default_config_path: str | None = None) -> None:
    asyncio.run(arun_tinker_training(args=args, default_config_path=default_config_path))


if __name__ == "__main__":
    run_tinker_training()
