"""Tinker training entrypoint for Auto-selected Platoon components."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

from platoon.train import entry_settings
from platoon.train.auto import (
    AutoDataset,
    AutoEnvironment,
    AutoRewardProcessor,
    AutoRollout,
    AutoTaskLoader,
    AutoWorkflow,
    check_environment_routes,
    check_harness_toolchains,
    check_mixture_pools,
)
from platoon.train.exclusion import TaskExclusion, record_exclusions
from platoon.train.identity import check_training_run
from platoon.train.labels import check_labels, label_column
from platoon.train.sampling import log_sampling_plan
from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig, evaluates, training_problems
from platoon.train.tinker.rl import PlatoonTinkerRLTrainer
from platoon.train.tinker.workflows import GroupRolloutWorkflow
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
    """

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

    exclusions: list[TaskExclusion] = []
    training = AutoDataset.training(config, exclusions=exclusions)
    train_dataset = training.dataset  # a weighted order's first epoch, or the listing the dataloader shuffles
    log_sampling_plan(training.plan)  # each entry's share, and a warning for weights that leave tasks out or repeat
    num_train_tasks = len(train_dataset) if hasattr(train_dataset, "__len__") else None
    if problems := training_problems(config, num_train_tasks=num_train_tasks):
        raise ConfigError(problems)  # before the trainer opens Tinker
    # With evaluation off the eval split is never read, so it is not listed and need not exist.
    eval_dataset = AutoDataset.from_config(config, "eval", exclusions=exclusions) if evaluates(config) else None
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
    # The run directory platoon train wrote lock.json into (PlatoonTinkerRLTrainer.run_log_path).
    run_dir = Path(config.log_path, config.stats.experiment_name, config.stats.trial_name)
    record_exclusions(run_dir, exclusions)
    if differences := entry_settings.lock_record(config, settings):
        update_run_lock(run_dir, {"entry_settings": differences})
    rollout_fn = AutoRollout.from_config(config, "train")
    eval_rollout_fn = AutoRollout.from_config(config, "eval")
    get_task_fn = AutoTaskLoader.from_config(config, "train")
    eval_get_task_fn = AutoTaskLoader.from_config(config, "eval")  # its routes carry the entries' eval_overrides
    reward_processor = AutoRewardProcessor.from_config(config)
    workflow_cls = AutoWorkflow.from_config(config, default=GroupRolloutWorkflow)
    # A workflow class that does not apply an entry's own settings would ignore them without a word (exit 78).
    if problems := entry_settings.workflow_problems(config.environments, workflow_cls, environment.workflow):
        raise ConfigError(problems)

    trainer = PlatoonTinkerRLTrainer(
        config=config, train_dataset=train_dataset, eval_dataset=eval_dataset, train_order=training.order
    )
    async with trainer:
        group_kwargs = {**worker_package_kwargs(workflow_cls, config), **label_kwargs(workflow_cls, config)}
        workflow_kwargs = {**group_kwargs, **environment.workflow_kwargs}
        train_workflow = workflow_cls(
            rollout_fn=rollout_fn,
            get_task_fn=get_task_fn,
            config=config.train.workflow_config,
            model_info=trainer.model_info,
            log_path=trainer.run_log_path,
            stats_scope=workflow_kwargs.pop("stats_scope", "train"),
            filter_errors=workflow_kwargs.pop("filter_errors", config.train.workflow_config.filter_errors),
            reward_processor=reward_processor,
            **workflow_kwargs,
        )

        eval_workflow_kwargs = {**group_kwargs, **environment.eval_workflow_kwargs}
        eval_workflow = workflow_cls(
            rollout_fn=eval_rollout_fn,
            get_task_fn=eval_get_task_fn,
            config=config.eval.workflow_config,
            model_info=trainer.model_info,
            log_path=trainer.run_log_path,
            stats_scope=eval_workflow_kwargs.pop("stats_scope", "eval"),
            filter_errors=eval_workflow_kwargs.pop("filter_errors", config.eval.workflow_config.filter_errors),
            reward_processor=reward_processor,
            **eval_workflow_kwargs,
        )
        if entry_settings.sets_settings(config.environments):
            # Each group runs with its entry's settings (GroupRolloutWorkflow.config inside the group).
            train_workflow.entry_configs = {label: own.train for label, own in settings.items()}
            eval_workflow.entry_configs = {label: own.eval for label, own in settings.items() if own.eval is not None}

        await trainer.train(train_workflow=train_workflow, eval_workflow=eval_workflow)


def worker_package_kwargs(workflow_cls: type, config: Any) -> dict[str, Any]:
    """The environments' packages (``AutoEnvironment.packages``) as workflow arguments, for a workflow whose
    rollout workers import them (a ``GroupRolloutWorkflow``); nothing for any other workflow class."""

    if not (isinstance(workflow_cls, type) and issubclass(workflow_cls, GroupRolloutWorkflow)):
        return {}
    return {"packages": AutoEnvironment.packages(config)}


def label_kwargs(workflow_cls: type, config: Any) -> dict[str, Any]:
    """The run's label column (``train.label_by``) as a workflow argument, for a workflow whose straggler policies key
    on it (a ``GroupRolloutWorkflow``); nothing for any other workflow class."""

    if not (isinstance(workflow_cls, type) and issubclass(workflow_cls, GroupRolloutWorkflow)):
        return {}
    return {"label_by": label_column(config)}


def run_tinker_training(args: list[str] | None = None, default_config_path: str | None = None) -> None:
    asyncio.run(arun_tinker_training(args=args, default_config_path=default_config_path))


if __name__ == "__main__":
    run_tinker_training()
