"""Auto factories for training components selected by environment config.

A single-entry ``environments`` list resolves exactly the components it names.
A mixture interleaves training tasks across entries by weight, labels each task
id with its environment (``<name>::<task id>``), and dispatches task loading,
rollout and reward processing per environment. Dispatch rides on the task: the
task loader attaches an :class:`EnvironmentRoute` to each task's misc, so the
rollout and reward dispatchers below, which rollout workers import by name,
need no trainer config. A single environment with a service or a harness gets a
route too, which is how ``harness_rollout`` finds its service and spec.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from platoon.options import PHASES, Phase, check_harness_options, harness_options, phase_of
from platoon.registry import import_from_string, import_modules, resolve_component
from platoon.task import Task
from platoon.train.components import (
    ENVIRONMENT_COLUMN,
    HARNESS_ROLLOUT,
    TASK_SAMPLERS,
    EnvironmentConfig,
    EnvironmentRoute,
    WeightedOrder,
    choose_harness,
    labeled_task_id,
    merge_run_level_settings,
    mixture_weights,
    normalize_environment_configs,
    platoon_column,
    split_task_id,
)
from platoon.train.labels import column_keys, label_text

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Sequence

    from platoon.train.exclusion import TaskExclusion
    from platoon.train.sampling import SamplingPlan

logger = logging.getLogger(__name__)

_ORS_REMOTE_SCHEMES = ("ors+http://", "ors+https://")


class AutoEnvironment:
    """Resolve the environments selected by config."""

    @classmethod
    def environments(cls, config: Any) -> list[EnvironmentConfig]:
        environments = normalize_environment_configs(getattr(config, "environments", None))
        if not environments:
            raise ValueError("Config must define at least one environment in `environments`")
        return environments

    @classmethod
    def from_config(cls, config: Any) -> EnvironmentConfig:
        """The environment whose run-level settings (workflow, workflow kwargs) apply to the run."""

        return merge_run_level_settings(cls.environments(config))

    @classmethod
    def packages(cls, config: Any) -> tuple[str, ...]:
        """The environments' ``package`` modules, which register the components they name locally (the argument
        of :func:`import_environment_packages`).  Components of installed extensions need none: they load by name
        from their entry points (:mod:`platoon.extensions`)."""

        return tuple(environment.package for environment in cls.environments(config) if environment.package)

    @classmethod
    def load(cls, config: Any) -> None:
        import_environment_packages(cls.packages(config))


def import_environment_packages(packages: Sequence[str]) -> None:
    """Import the ``package`` modules that register the environments' harnesses, programs and loaders.  Also run
    first in the processes that run rollouts away from the process that loaded the config, which imports none of
    them: evaluation subprocess slots (their initializer) and training rollout workers (``rollout_isolation:
    process``), given :meth:`AutoEnvironment.packages`."""

    import_modules(packages)


def _resolve_required_component(kind: str, spec: str | None, environment: EnvironmentConfig) -> Any:
    if spec is None:
        raise ValueError(f"Config must set environments[{environment.label}].{kind}")
    return resolve_component(kind, spec)


def _needs_route(environment: EnvironmentConfig) -> bool:
    return environment.service is not None or environment.harness is not None


def check_mixture_pools(environments: Sequence[EnvironmentConfig]) -> None:
    """Refuse mixture entries that would run another entry's environment on ORS env servers.

    An env server answers a request that names no ``variant`` with its default environment, whatever env name the
    ``ors+http://.../<EnvName>`` URI gives (the name is the deployment's), so entries with different env names and
    no variant that reach one server would all run the same environment.  That happens silently when an entry's
    own pool variable is missing and it falls back to a process-wide pool that another entry uses: the legacy
    ``OPENREWARD_SESSION_URLS`` (``legacy_global_pool``), ``OPENREWARD_SESSION_URL`` or ``localhost:8080``.

    Raises :class:`~platoon.utils.config.ConfigError` for such entries, naming the variable each one lacks, and
    logs a warning for an entry that takes a process-wide pool no other entry shares.  Entries with a variant, in
    process, or with an ``owner/name`` deployment (which the hosted service routes by that name) are not checked,
    nor is a single environment.  Pools are read now, as listing the entries' tasks reads them; the train, eval and
    rollout commands check before they contact any env server.
    """

    if len(environments) < 2:
        return
    from platoon.utils.config import ConfigError

    claims: dict[str, list[tuple[str, str, str | None, str]]] = {}  # url -> [(entry, env name, fallback, variable)]
    for environment in environments:
        service = _remote_ors_service(environment)
        if service is None or service.variant is not None or "/" in service.env_name:
            continue
        urls, fallback = service.session_pool()
        claim = (environment.label, service.env_name, fallback, service.own_pool_env_var())
        for url in dict.fromkeys(urls):
            claims.setdefault(url, []).append(claim)
    problems = []
    for url, claimants in claims.items():
        if len({env_name.lower() for _, env_name, _, _ in claimants}) < 2:
            continue
        entries = [f"{label!r} ({env_name})" for label, env_name, _, _ in claimants]
        fallbacks = [
            f"{label!r} took its pool from {fallback}, since {variable} is unset or empty"
            for label, _, fallback, variable in claimants
            if fallback is not None
        ]
        problems.append(
            f"environments {', '.join(entries[:-1])} and {entries[-1]} {'both' if len(entries) == 2 else 'all'} "
            f"reach the env server {url} without a variant; an ORS env server answers those requests with its "
            "default environment, so they would run the same one"
            + (f" ({'; '.join(fallbacks)})" if fallbacks else "")
            + ". Give each entry its own pool (service_kwargs.urls or its OPENREWARD_SESSION_URLS_<LABEL>; "
            "legacy_global_pool: false stops the fallback to OPENREWARD_SESSION_URLS), or select its environment "
            "with ?variant=<Env>"
        )
    if problems:
        raise ConfigError(problems)
    warned = set()
    for claimants in claims.values():
        for label, env_name, fallback, variable in claimants:
            if fallback is not None and label not in warned:
                warned.add(label)
                logger.warning(
                    "Mixture entry %r (ORS %s) takes its env servers from %s, a process-wide pool, since %s is unset "
                    "or empty; give it its own pool (or set legacy_global_pool: false), so that it cannot reach "
                    "another entry's servers",
                    label,
                    env_name,
                    fallback,
                    variable,
                )


def _remote_ors_service(environment: EnvironmentConfig) -> Any | None:
    """The service of an ``ors+http(s)://`` entry, built without contacting it; None for any other entry, and for
    one that cannot be built (listing its tasks or its rollouts report why)."""

    if not isinstance(environment.service, str) or not environment.service.startswith(_ORS_REMOTE_SCHEMES):
        return None
    from platoon.envs.service import get_service

    try:
        return get_service(environment.service, **environment.service_kwargs)
    except Exception:
        logger.debug("Cannot build the service of environment %r", environment.label, exc_info=True)
        return None


@dataclass(frozen=True)
class TrainingTasks:
    """A run's training tasks, as ``platoon train`` draws them (:meth:`AutoDataset.training`)."""

    # What AutoDataset.from_config gives for the train split: the first epoch of a weighted order, its rows marked
    # with PRESERVE_DATASET_ORDER_COLUMN, or a single environment's listing, which the dataloader shuffles.
    dataset: Any
    # Every task the run draws from, once each: a weighted order's tasks as rows, those its first epoch leaves out
    # included, or a single environment's listing (``dataset``).
    tasks: Any
    # The weighted order, which the dataloader draws epoch after epoch; None for a listing that it shuffles.
    order: WeightedOrder[dict[str, Any]] | None = None
    # How the run draws the tasks: each environment's share and how often each task is drawn over the planned steps
    # (platoon.train.sampling); None when the listing's size is not known.
    plan: SamplingPlan | None = None


class AutoDataset:
    """Build datasets from environment dataset loaders; a mixture becomes one labeled, weighted dataset.

    A mixture's training tasks are drawn in the weighted fair order of its environments (a :class:`WeightedOrder`,
    each environment's tasks shuffled with its ``seed``), in epochs of as many tasks as there are, filled to whole
    batches; each epoch takes up the order where the one before it stopped (:meth:`training`).  A single
    environment's tasks go to the dataloader as listed, which draws them in seeded shuffled passes (seed 42), each
    epoch the whole batches they fill, its last partial batch starting the next epoch, unless ``train.task_sampler``
    is ``weighted``: then they take the mixture's path, as a mixture of one (task ids stay unlabeled).  Evaluation, and
    any split of an evaluation config, visits every selected task once, in config order.

    Each environment's ``exclude_tasks`` leave its list first (:func:`_load_split`), so the order and the batches
    are drawn from the tasks that remain.
    """

    @classmethod
    def from_config(cls, config: Any, split: str, *, exclusions: list[TaskExclusion] | None = None) -> Any:
        """The tasks of ``split``: for a training config's train split, the first epoch of the order training draws
        them in (:meth:`training`), and otherwise every selected task once, in config order.  What each environment's
        ``exclude_tasks`` left out is appended to ``exclusions`` when given.  Every environment's configuration
        errors are raised together."""

        if split == "train" and _trains(config):
            return cls.training(config, exclusions=exclusions).dataset
        environments = AutoEnvironment.environments(config)
        if len(environments) == 1:
            dataset = _load_split(config, environments[0], split, exclusions)
            return _dataset_from_rows(_rows(dataset)) if isinstance(dataset, list) else dataset
        listed = _listed_rows(config, environments, split, exclusions)
        return _dataset_from_rows([row for environment_rows in listed.values() for row in environment_rows])

    @classmethod
    def training(cls, config: Any, *, exclusions: list[TaskExclusion] | None = None) -> TrainingTasks:
        """The training tasks of a training config, listed once, and the order its dataloader draws them in.

        A mixture's tasks, or a single environment's under ``train.task_sampler: weighted``, take a
        :class:`WeightedOrder` over the environments, whose epochs hold as many tasks as there are, filled to whole
        batches of ``train.batch_size``: each epoch takes up the credits and each environment's pass where the one
        before it stopped, so an environment that gets fewer slots than it has tasks draws the rest in the next
        epochs, and the whole run follows from the environments' seeds.  An environment's share is its ``weight``
        over the mixture's, ``weight: tasks`` counting the tasks it lists here (:func:`mixture_weights`).  What each
        environment's ``exclude_tasks`` left out is appended to ``exclusions`` when given."""

        from platoon.train.sampling import sampling_plan

        environments = AutoEnvironment.environments(config)
        train = getattr(config, "train", None)
        batch_size = getattr(train, "batch_size", None)
        if len(environments) == 1 and _task_sampler(config) != "weighted":
            dataset = _load_split(config, environments[0], "train", exclusions)
            dataset = _dataset_from_rows(_rows(dataset)) if isinstance(dataset, list) else dataset
            count = len(dataset) if hasattr(dataset, "__len__") else 0
            epoch_size = count // batch_size * batch_size if batch_size else count  # the whole batches they fill
            plan = None
            if epoch_size:
                label = environments[0].label
                plan = sampling_plan(environments, {label: count}, {label: 1.0}, train=train, epoch_size=epoch_size)
            return TrainingTasks(dataset=dataset, tasks=dataset, plan=plan)

        from platoon.train.tinker.dataset_order import PRESERVE_DATASET_ORDER_COLUMN

        listed = _listed_rows(config, environments, "train", exclusions)
        # Every row has every column, as a dataset's rows do, whichever epoch draws it.
        columns = list(dict.fromkeys(column for rows in listed.values() for row in rows for column in row))
        pools = {
            label: [{column: row.get(column) for column in columns} for row in rows] for label, rows in listed.items()
        }
        counts = {label: len(rows) for label, rows in pools.items()}
        size = sum(counts.values())
        if batch_size:
            size = math.ceil(size / batch_size) * batch_size  # each epoch is whole batches, its last one filled
        weights = mixture_weights(environments, counts)
        order = WeightedOrder(
            pools,
            weights,
            seeds={environment.label: environment.seed for environment in environments},
            epoch_size=size,
        )
        first = _dataset_from_rows([{**row, PRESERVE_DATASET_ORDER_COLUMN: True} for row in order.epoch(0)])
        return TrainingTasks(
            dataset=first,
            tasks=[row for rows in pools.values() for row in rows],
            order=order,
            plan=sampling_plan(environments, counts, weights, train=train, epoch_size=size),
        )


def _trains(config: Any) -> bool:
    """Whether a config trains: one with a ``train`` section, whose train split is drawn in batches.  An evaluation
    config visits every task of any split once."""

    return getattr(config, "train", None) is not None


def _listed_rows(
    config: Any, environments: Sequence[EnvironmentConfig], split: str, exclusions: list[TaskExclusion] | None
) -> dict[str, list[dict[str, Any]]]:
    """Each environment's tasks of ``split`` as rows, by label, in config order: in a mixture each task id reads
    ``<name>::<task id>`` and each row names its environment in ENVIRONMENT_COLUMN.  Every environment's
    configuration errors are raised together."""

    from platoon.utils.config import ConfigError

    mixture = len(environments) > 1
    listed: dict[str, list[dict[str, Any]]] = {}
    problems: list[str] = []
    for environment in environments:
        label = environment.label
        try:
            rows = _rows(_load_split(config, environment, split, exclusions))
        except ConfigError as exc:
            problems.extend(exc.problems)
            continue
        listed[label] = [
            {**row, "task_id": labeled_task_id(label, str(row["task_id"])), ENVIRONMENT_COLUMN: label}
            if mixture
            else row
            for row in rows
        ]
    if problems:
        raise ConfigError(problems)
    return listed


def _task_sampler(config: Any) -> str:
    """``train.task_sampler``: ``weighted`` orders a single environment's training tasks as a mixture's."""

    sampler = getattr(getattr(config, "train", None), "task_sampler", None) or "auto"
    if sampler not in TASK_SAMPLERS:
        raise ValueError(f"train.task_sampler must be one of {TASK_SAMPLERS}, got {sampler!r}")
    return sampler


def _dataset_from_rows(rows: list[dict[str, Any]]) -> Any:
    """A dataset with every column any environment's rows have (``Dataset.from_list`` keeps the first row's)."""

    from datasets import Dataset

    columns = list(dict.fromkeys(column for row in rows for column in row))
    return Dataset.from_dict({column: [row.get(column) for row in rows] for column in columns})


def _load_split(
    config: Any, environment: EnvironmentConfig, split: str, exclusions: list[TaskExclusion] | None = None
) -> Any:
    """The tasks of ``split`` that ``environment`` lists, without its ``exclude_tasks``; what they left out is
    appended to ``exclusions`` when given."""

    from platoon.utils.config import ConfigError

    loader_spec = (
        environment.dataset_loader
        if split == "train"
        else environment.eval_dataset_loader or environment.dataset_loader
    )
    kwargs = environment.dataset_kwargs if split == "train" else environment.eval_dataset_kwargs
    where = f"environments[{environment.label}]"
    if loader_spec is None and environment.service is not None:
        from platoon.envs.service import EnvNotPrepared, EnvSetupError
        from platoon.train.exclusion import UnknownTasks

        try:
            listing = _service_task_ids(
                environment.service, environment.service_kwargs, split, environment.exclude_tasks, **kwargs
            )
        except EnvSetupError as exc:
            if exc.failure_kind != "env_config":
                raise
            if isinstance(exc, EnvNotPrepared):  # what platoon prepare makes is missing, such as a task's image
                from platoon.prepare import NotPrepared

                raise NotPrepared(f"{where}.service: {exc}") from exc
            # The service refuses its own settings (an ORS env name that its env server would answer with another
            # environment): a configuration error, which the CLI reports in one line with exit 78.
            raise ConfigError(f"{where}.service: {exc}") from exc
        except UnknownTasks as exc:
            raise ConfigError(f"{where}.exclude_tasks: {exc}") from exc
        tasks = _service_rows(listing, column_keys(config), where)
        if not environment.exclude_tasks:
            return tasks
        result, listed, excluded = tasks, listing.listed, listing.excluded
    else:
        loader = _resolve_required_component("dataset_loader", loader_spec, environment)
        result = loader(config, split, **kwargs)
        if not environment.exclude_tasks:
            return result
        from platoon.train.exclusion import exclude_loaded_tasks

        result, listed, excluded = exclude_loaded_tasks(result, environment.exclude_tasks, where=where)
    if listed and len(excluded) == listed:
        raise ConfigError(f"{where}.exclude_tasks leaves none of the {listed} tasks of split {split!r}")
    logger.info("%s: exclude_tasks left out %d of the %d tasks of split %r", where, len(excluded), listed, split)
    if exclusions is not None:
        from platoon.train.exclusion import TaskExclusion

        exclusions.append(TaskExclusion(environment.label, split, listed, tuple(excluded)))
    return result


@dataclass(frozen=True)
class _ServiceListing:
    """What :func:`_service_task_ids` listed: the task ids the run gets, how many the splits held, the ids
    ``exclude_tasks`` left out (in listed order), and the columns the service gives each task it kept (None for a
    service without ``task_columns``)."""

    task_ids: list[str]
    listed: int
    excluded: list[str]
    columns: dict[str, dict[str, Any]] | None = None


def _service_rows(listing: _ServiceListing, keys: dict[str, str], where: str) -> list[str] | list[dict[str, Any]]:
    """The listed tasks as the run's dataset rows: their ids alone, or, for a service that gives its tasks columns of
    their own (``task_columns``, a Harbor task's index fields), each task's ``task_id`` and those columns, each value
    as text (:func:`~platoon.train.labels.label_text`; one that is not a scalar, or empty, is left out).

    A column named like one of platoon's own (``task_id``, ``_platoon*``) is left out of the rows, and a
    :class:`~platoon.utils.config.ConfigError` when one of the run's keys that read columns names it (``keys``,
    :func:`~platoon.train.labels.column_keys`): the key would read platoon's column, not the task's.
    """

    from platoon.utils.config import ConfigError

    if listing.columns is None:
        return listing.task_ids
    rows: list[dict[str, Any]] = []
    left_out: set[str] = set()
    for task_id in listing.task_ids:
        row: dict[str, Any] = {"task_id": task_id}
        for name, value in (listing.columns.get(task_id) or {}).items():
            if not isinstance(name, str) or not name:
                continue
            if platoon_column(name):
                left_out.add(name)
            elif (text := label_text(value)) is not None:
                row[name] = text
        rows.append(row)
    if named := sorted(name for name in left_out if name in keys):
        raise ConfigError(
            [
                f"{where}: {keys[name]} names {name!r}, a column this environment's tasks have (an index field) whose "
                "name is that of one of platoon's own columns, which the dataset keeps instead, so the key would not "
                "read the task's: name another column"
                for name in named
            ]
        )
    return rows


def _service_task_ids(
    service_name: str,
    service_kwargs: dict[str, Any],
    split: str,
    exclude_tasks: Sequence[str] = (),
    /,
    *,
    split_name: str | Sequence[str] | None = None,
    shuffle_seed: int | None = None,
    limit: int | None = None,
) -> _ServiceListing:
    """Task ids of ``split`` (or ``split_name``, e.g. eval -> test) listed by an environment service.

    ``split_name`` may be a list of splits, whose ids are concatenated in order (TextCraft-Synth's
    ``[val-easy, val-medium, val-hard]``). ``shuffle_seed`` shuffles the ids with ``random.Random(seed)`` before
    ``limit`` keeps the first ones: a seeded sample, drawn the way the original inference scripts drew theirs.
    ``exclude_tasks`` leave between the two, so a seeded sample keeps the order of the tasks that remain and
    ``limit`` counts only those; an item that names no task of the service raises
    :class:`~platoon.train.exclusion.UnknownTasks`.  A service with ``check_tasks(split, task_ids)`` then refuses,
    with an ``env_config`` ``EnvSetupError``, the tasks of each split that remain and that it would not run.
    """

    from platoon.envs.service import get_service
    from platoon.train.exclusion import excluded_service_tasks

    if isinstance(shuffle_seed, bool) or not isinstance(shuffle_seed, int | None):
        raise ValueError(f"shuffle_seed must be an integer, got {shuffle_seed!r}")
    splits = [split_name or split] if split_name is None or isinstance(split_name, str) else list(split_name)
    if not splits or not all(isinstance(name, str) and name for name in splits):
        raise ValueError(f"split_name must be a split name or a non-empty list of them, got {split_name!r}")
    from platoon.envs.service import EnvSetupError

    try:
        service = get_service(service_name, **service_kwargs)
    except (TypeError, ValueError) as exc:  # the service refuses its options: a configuration error naming the entry
        raise EnvSetupError("env_config", f"service_kwargs: {exc}") from exc

    async def list_splits() -> tuple[list[str], set[str], dict[str, dict[str, Any]] | None]:
        listed = {name: await service.list_tasks(name) for name in dict.fromkeys(splits)}
        task_ids = [task_id for name in splits for task_id in listed[name]]
        excluded = await excluded_service_tasks(service, splits, task_ids, exclude_tasks) if exclude_tasks else set()
        check_tasks = getattr(service, "check_tasks", None)
        if callable(check_tasks):  # what the service would refuse to run, of the tasks the run keeps
            for name, ids in listed.items():
                await check_tasks(name, [task_id for task_id in ids if task_id not in excluded])
        task_columns = getattr(service, "task_columns", None)
        columns = None
        if callable(task_columns):  # the columns the service gives the tasks the run keeps (Harbor: index fields)
            columns = await task_columns([task_id for task_id in task_ids if task_id not in excluded])
        return task_ids, excluded, columns

    task_ids, excluded, columns = _run_to_completion(list_splits)
    listed, left_out = len(task_ids), [task_id for task_id in task_ids if task_id in excluded]
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(task_ids)
    task_ids = [task_id for task_id in task_ids if task_id not in excluded]
    return _ServiceListing(task_ids if limit is None else task_ids[:limit], listed, left_out, columns)


def _run_to_completion(make: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
    """Run a coroutine from synchronous setup code, also when an event loop is already running here."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make())
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(make())).result()


def _rows(dataset: Any) -> list[dict[str, Any]]:
    if isinstance(dataset, list):
        return [row if isinstance(row, dict) else {"task_id": row} for row in dataset]
    to_list = getattr(dataset, "to_list", None)
    if callable(to_list):
        return to_list()
    raise TypeError(f"A mixture dataset loader must return task ids, rows or a datasets.Dataset, got {dataset!r}")


class AutoTaskLoader:
    """Resolve a task loader from environment config."""

    @classmethod
    def from_config(cls, config: Any, split: str = "train") -> Any:
        """The task loader of ``split``'s rollouts: the routes it attaches carry the options of the split's phase
        (``platoon.options.phase_of``: the entries' ``eval_overrides`` apply to every split but ``train``)."""

        environments = AutoEnvironment.environments(config)
        if len(environments) == 1 and not _needs_route(environments[0]):
            return _resolve_required_component("task_loader", environments[0].task_loader, environments[0])
        return TaskRouter(environments, phase=phase_of(split))


class TaskRouter:
    """Load a task through its environment and attach the environment's route.

    Mixture task ids carry their environment name; a single environment's ids are
    its own. An environment with a service and no task loader yields reference
    tasks that ``harness_rollout`` resolves through the service in the worker.
    The harness for each task is drawn here, so every member of a group gets it.
    The routes carry the options of ``phase``: an evaluation's have the entries'
    ``eval_overrides`` applied.
    """

    def __init__(self, environments: list[EnvironmentConfig], *, phase: Phase = "train") -> None:
        self._mixture = len(environments) > 1
        self._environments = {environment.label: environment for environment in environments}
        self._routes = environment_routes(environments, phase=phase)
        self._loaders = {
            environment.label: resolve_component("task_loader", environment.task_loader)
            for environment in environments
            if environment.task_loader is not None
        }

    def __call__(self, task_id: str) -> Task:
        label, raw_id = split_task_id(task_id) if self._mixture else (next(iter(self._environments)), task_id)
        environment = self._environments.get(label)
        if environment is None:
            raise ValueError(
                f"Task {task_id!r} names unknown environment {label!r}; known: {sorted(self._environments)}"
            )
        loader = self._loaders.get(label)
        task = loader(raw_id) if loader is not None else Task(id=raw_id)
        route = replace(
            self._routes[label], harness=choose_harness(environment.harness, task_id=raw_id, seed=environment.seed)
        )
        return route.attach(task)


def check_environment_routes(environments: Sequence[EnvironmentConfig]) -> None:
    """Build the route of every entry that has one (:func:`environment_routes`), for each phase, before any task is
    listed or env server contacted, so that a harness that does not resolve, cannot run as configured, takes no such
    options or could never train is refused first, with exit 78.  The train and eval commands call it once the
    entries' packages are imported; the rollout command builds the same routes when it loads its task
    (:class:`TaskRouter`)."""

    for phase in PHASES:
        environment_routes(environments, phase=phase, only_routed=True)


def environment_routes(
    environments: Sequence[EnvironmentConfig], *, phase: Phase = "train", only_routed: bool = False
) -> dict[str, EnvironmentRoute]:
    """:func:`environment_route` of each entry (each that needs one, with ``only_routed``), by label, with the options
    of ``phase``.  Every entry's configuration errors are raised together, in one ``ConfigError``, so that a
    mixture's problems all show at once, as those of its task lists do (``exclude_tasks``)."""

    from platoon.utils.config import ConfigError

    routes: dict[str, EnvironmentRoute] = {}
    errors: list[ConfigError] = []
    for index, environment in enumerate(environments):
        if only_routed and not _needs_route(environment):
            continue
        try:
            routes[environment.label] = environment_route(environment, phase=phase, index=index)
        except ConfigError as exc:
            errors.append(exc)
    if errors:
        problems = [problem for error in errors for problem in error.problems]
        raise ConfigError(list(dict.fromkeys(problems))) from errors[0]
    return routes


def environment_route(
    environment: EnvironmentConfig, *, phase: Phase = "train", index: int | None = None
) -> EnvironmentRoute:
    """Resolve an environment's components to worker-importable references, and check its harness settings.

    A harness that does not resolve (an unknown or ambiguous name, a malformed spec, a spec file that does not
    exist, a ``module:function`` that does not import or is not an ``@harness``), that cannot run as configured, or
    whose rollouts could never train (:func:`harness_setting_problems`), and options its harness does not take
    (``platoon.options``), are a ``ConfigError``, which the CLI reports with one line per problem and exit 78.  The
    route carries the harness options of ``phase`` (``eval``: with ``eval_overrides`` applied); ``index`` is the
    entry's position in ``environments``, for the dotted paths of option problems.
    """

    from platoon.harness.registry import get_harness
    from platoon.utils.config import ConfigError

    harnesses = []
    for ref in environment.harness_refs():
        try:
            harnesses.append(get_harness(ref))
        except (ValueError, OSError, ImportError, AttributeError) as exc:  # the cause stays chained
            raise ConfigError(f"environments[{environment.label}].harness: {exc}") from exc
    rollout = _component_ref("rollout", environment.rollout) if environment.rollout else None
    if rollout is None and environment.harness is not None:
        rollout = HARNESS_ROLLOUT
    eval_rollout = _component_ref("rollout", environment.eval_rollout) if environment.eval_rollout else rollout
    harness_rollout = HARNESS_ROLLOUT in (rollout, eval_rollout)
    problems = harness_setting_problems(environment, harnesses, harness_rollout=harness_rollout)
    where = f"environments.{index}" if index is not None else f"environments[{environment.label}]"
    for harness in harnesses:
        problems += check_harness_options(environment, harness, where=where, phases=(phase,)).problems
    if problems:
        raise ConfigError(list(dict.fromkeys(problems)))
    return EnvironmentRoute(
        name=environment.label,
        rollout=rollout,
        eval_rollout=eval_rollout,
        reward_processor=(
            _component_ref("reward_processor", environment.reward_processor) if environment.reward_processor else None
        ),
        service=environment.service,
        service_kwargs=dict(environment.service_kwargs),
        placement=environment.placement,
        builtins=environment.builtins,
        on_context_overflow=environment.on_context_overflow,
        on_harness_exit=environment.on_harness_exit,
        resolve_task=environment.service is not None and environment.task_loader is None,
        program=environment.program,
        harness_options=harness_options(environment, phase),
        agents=environment.agents,
    )


def check_harness_toolchains(environments: Sequence[EnvironmentConfig]) -> None:
    """Refuse a run, before any rollout, whose TOML harness specs run from a named toolchain that is not built where
    the run finds it: under the toolchain root, ``toolchains`` in the site profile's cache, else
    ``~/.cache/platoon/toolchains`` (:func:`platoon.harness.toolchains.toolchain_root`; the deprecated
    ``service_kwargs.toolchain_root`` and ``$PLATOON_TOOLCHAINS`` name another until platoon 0.3.0).

    A run never builds a toolchain.  Building one needs uv and network access to PyPI, takes minutes, and writes a
    directory that every sandbox of a site mounts read-only, so it is a step of its own, once per site (``platoon
    prepare``); options never change a toolchain, so changing them never needs a rebuild.  A toolchain that is not
    built raises :class:`~platoon.prepare.NotPrepared`, which the command line reports with the ``platoon prepare``
    command for its config.  What prepare does not make is a plain ``ConfigError`` that says what does: a toolchain
    that a spec without a recipe names, which only its user can put there, and one built from another recipe (or
    whose manifest cannot be read), which a live run of another config or platoon may be staging its rollouts from,
    so that only a forced build replaces it (``platoon toolchain build --force``).  Exit 78 either way: a restart would
    not build it either.
    """

    from platoon.harness.registry import get_harness
    from platoon.harness.spec import HarnessSpec
    from platoon.harness.toolchains import toolchain_problem, toolchain_state
    from platoon.prepare import NotPrepared
    from platoon.utils.config import ConfigError

    unbuilt: list[str] = []
    elsewhere: list[str] = []  # what platoon prepare does not make
    for environment in environments:
        for ref in environment.harness_refs():
            try:
                harness = get_harness(ref)
            except (ValueError, OSError, ImportError, AttributeError):
                continue  # refused where the route is built, with its own error
            if not isinstance(harness, HarnessSpec):
                continue
            root = environment.service_kwargs.get("toolchain_root")
            root = None if root is None else str(root)
            state = toolchain_state(harness, root=root)
            if state is None:
                continue
            where = f"environments[{environment.label}].harness"
            if state.buildable and not state.needs_force:
                unbuilt.append(f"{where}: {state.describe()}")
            else:
                elsewhere.append(f"{where}: {toolchain_problem(harness, ref=ref, root=root)}")
    if elsewhere:
        raise ConfigError(list(dict.fromkeys([*elsewhere, *unbuilt])))
    if unbuilt:
        raise NotPrepared(list(dict.fromkeys(unbuilt)))


# Services whose sessions have no task sandbox, so a harness spec acts on their envs only through MCP tools.
_SANDBOXLESS_SERVICES = ("table", "python://", "ors+http://", "ors+https://")


def harness_setting_problems(
    environment: EnvironmentConfig, harnesses: Sequence[Any], *, harness_rollout: bool = True
) -> list[str]:
    """What in an environment entry's harness settings cannot run as configured, or means that its rollouts (or
    its subagents) could never train, one line each.

    ``harnesses`` are the entry's resolved harnesses (``EnvironmentConfig.harness_refs``).  A harness spec that
    cannot disable its built-in tools cannot run with ``builtins: disable``.  The rest applies when the entry's
    training or evaluation rollouts run through ``harness_rollout`` (``harness_rollout``; a rollout of the entry's
    own runs its harness as it likes).  A TOML harness spec runs as one agent with no episode program, so with one:

    * on ``service: table`` (tasks with no tools and no grader) every rollout is ``invalid`` (``self_grade``);
    * a ``program`` other than ``single``, and ``agents`` settings, never apply (``agents.max_depth: 0``, which says
      that no agent spawns, is consistent with a spec, and its other ``agents`` settings are moot);
    * without ``[mcp]`` it reaches the env's tools only from inside a task sandbox, so on a service without one
      (``table``, ``python://``, ``ors+http(s)://``), or placed ``local``, it cannot act on the environment: every
      rollout scores 0 (``no_submit``), or on Harbor the task's tests grade a sandbox it never acted in.

    With a function harness and ``agents.child_grading: env`` (:func:`_env_child_grading_problem`), a child graded
    by a session that cannot grade its own goal is refused, unless ``agents.max_depth: 0`` allows no children.
    """

    from platoon.harness.spec import HarnessSpec
    from platoon.train.components import AgentTreeConfig

    where = f"environments[{environment.label}]"
    specs = [harness for harness in harnesses if isinstance(harness, HarnessSpec)]
    problems = [
        f"{where} disables built-in tools, which harness {spec.ref} cannot do"
        for spec in specs
        if environment.builtins == "disable" and spec.builtins.disable is None
    ]
    if not harness_rollout:
        return problems
    service = environment.service or ""
    for spec in specs:
        if service == "table":
            problems.append(
                f"{where}: harness {spec.ref} is a TOML harness spec, which runs no episode program, and a task table "
                "has no grader, so every rollout would be invalid (self_grade) and none would train; play a table "
                "with a function harness and a program that sets the outcome (program: gym, or your own)"
            )
        elif not spec.mcp.enabled and (
            service.startswith(_SANDBOXLESS_SERVICES) or (service == "harbor" and environment.placement == "local")
        ):
            if service == "harbor":  # the task's tests grade a sandbox the agent never acted in
                reached, scored = "placed local", "the task's tests would grade a sandbox it never acted in"
            else:
                reached, scored = f"on {service}, which has no task sandbox,", "every rollout would score 0 (no_submit)"
            problems.append(
                f"{where}: harness {spec.ref} takes no env tools ([mcp] is not enabled), and {reached} its own "
                f"tools never reach the environment, so {scored} and nothing would train; use a harness that takes "
                "the env's tools over MCP (platoon-bash) or a function harness"
            )
    if specs:
        names = ", ".join(spec.ref for spec in specs)
        if environment.program != "single":
            problems.append(
                f"{where}: program {environment.program!r} never runs: harness {names} is a TOML harness spec, which "
                "runs as one agent with no episode program; use a function harness, or the default program (single)"
            )
        if environment.agents != AgentTreeConfig() and environment.agents.max_depth != 0:  # 0: no subagents, as here
            problems.append(
                f"{where}: agents settings never apply: harness {names} is a TOML harness spec, which cannot spawn "
                "subagents; use a function harness, or leave agents out"
            )
    if len(specs) < len(harnesses):  # a function harness, which may spawn subagents
        problem = _env_child_grading_problem(environment)
        if problem is not None:
            problems.append(f"{where}.agents: {problem}")
    return problems


def _env_child_grading_problem(environment: EnvironmentConfig) -> str | None:
    """Why ``agents.child_grading: env`` cannot grade the children of ``environment`` by their own goal, or None.

    ``env`` grades a child by its forked session.  A task table has no grader, so its children would be
    ``invalid``.  On a ``shared`` fork (``agents.fork: shared``, or unset: every built-in service forks shared by
    default) the child's session is its parent's live env, which grades a child's own goal only through an
    in-process class's ``platoon_fork`` hook.  Without one a child is graded by the episode's first submission, or
    scores 0 (``no_submit``), whatever its goal: an env server cannot run the hook, and a Harbor sandbox runs the
    task's whole ``test.sh`` on the shared state.  A child on a fork of its own (``fresh``, or ``copy``) is graded on
    the whole task, which is left to the config.
    """

    if environment.agents.child_grading != "env" or environment.agents.max_depth == 0:  # 0: no children to grade
        return None
    service = environment.service or ""
    if service == "table":
        return (
            "child_grading: env grades a child by its forked session, and a task table has no grader, so every "
            "child would be invalid and none would train; use child_grading: verifier or none"
        )
    if environment.agents.fork not in (None, "shared"):
        return None
    if service.startswith(_ORS_REMOTE_SCHEMES):
        return (
            "child_grading: env on shared forks grades a child by its parent's session on the env server, which "
            "gives it the episode's first submission (or no_submit 0) instead of a grade of its own goal: an env "
            "server cannot run a platoon_fork hook. Use child_grading: verifier or none (agents.fork: fresh grades "
            "each child on the whole task, in a session of its own)"
        )
    if service == "harbor":
        return (
            "child_grading: env on shared forks grades a child with the task's whole test.sh on the sandbox it shares "
            "with its parent, not by its own goal, and holds that sandbox while the tests run. Use child_grading: "
            "verifier or none (agents.fork: copy or fresh grades each child on the whole task, in a sandbox of its "
            "own)"
        )
    if service.startswith("python://"):
        environment_class = _ors_environment_class(environment)
        if environment_class is not None and not callable(getattr(environment_class, "platoon_fork", None)):
            return (
                f"child_grading: env on shared forks needs {environment_class.__name__}.platoon_fork, which gives a "
                "child its own goal on the shared state; without it a child is graded by the episode's first "
                "submission (or no_submit 0), whatever its goal. Define platoon_fork, or use child_grading: verifier "
                "or none"
            )
    return None


def _ors_environment_class(environment: EnvironmentConfig) -> type | None:
    """The in-process ORS class of a ``python://`` entry; None when it does not import (its rollouts report why)."""

    from platoon.envs.service import get_service

    try:
        return get_service(environment.service or "", **environment.service_kwargs).environment_class
    except Exception:
        logger.debug("Cannot import the environment class of %r", environment.label, exc_info=True)
        return None


def _component_ref(kind: str, spec: str) -> str:
    from platoon.train.tinker.workers.protocol import validated_ref

    return validated_ref(resolve_component(kind, spec), what=f"{kind} {spec!r}")


class AutoRollout:
    """Resolve train/eval rollout functions from environment config."""

    @classmethod
    def from_config(cls, config: Any, split: str = "train") -> Any:
        environments = AutoEnvironment.environments(config)
        if len(environments) > 1:
            return mixture_rollout if split == "train" else mixture_eval_rollout
        environment = environments[0]
        rollout_spec = environment.rollout if split == "train" else environment.eval_rollout or environment.rollout
        if rollout_spec is None and environment.harness is not None:
            return import_from_string(HARNESS_ROLLOUT)
        return _resolve_required_component("rollout", rollout_spec, environment)


async def mixture_rollout(task: Task, config: Any) -> Any:
    """Run a mixture task with its environment's training rollout."""

    return await _resolve_ref(_route(task).rollout, task)(task, config)


async def mixture_eval_rollout(task: Task, config: Any) -> Any:
    """Run a mixture task with its environment's evaluation rollout."""

    return await _resolve_ref(_route(task).eval_rollout, task)(task, config)


def _route(task: Task) -> EnvironmentRoute:
    route = EnvironmentRoute.of_task(task)
    if route is None:
        raise ValueError(f"Task {task.id!r} carries no environment route; load mixture tasks with AutoTaskLoader")
    return route


def _resolve_ref(ref: str | None, task: Task) -> Any:
    if ref is None:
        raise ValueError(f"Environment of task {task.id!r} has no rollout configured")
    from platoon.train.tinker.workers.protocol import resolve_ref

    return resolve_ref(ref)


class AutoRewardProcessor:
    """Resolve a reward processor from environment config."""

    @classmethod
    def from_config(cls, config: Any) -> Any:
        environments = AutoEnvironment.environments(config)
        if len(environments) > 1 and any(environment.reward_processor for environment in environments):
            return mixture_reward_processor
        if environments[0].reward_processor is not None:
            return resolve_component("reward_processor", environments[0].reward_processor)
        # The workflow recognizes this object and names no reference for rollout workers.
        from platoon.train.tinker.workers.run import DEFAULT_REWARD_PROCESSOR

        return DEFAULT_REWARD_PROCESSOR


def mixture_reward_processor(trajectory: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Process a trajectory with its environment's reward processor, or take its own reward."""

    route = EnvironmentRoute.of_trajectory(trajectory)
    if route is None:
        raise ValueError("Trajectory carries no environment route; its task was not loaded with AutoTaskLoader")
    if route.reward_processor is None:
        return trajectory["reward"], {}
    from platoon.train.tinker.workers.protocol import resolve_ref

    return resolve_ref(route.reward_processor)(trajectory)


class AutoWorkflow:
    """Resolve a workflow class from environment config."""

    @classmethod
    def from_config(cls, config: Any, default: type) -> type:
        environment = AutoEnvironment.from_config(config)
        if environment.workflow == "group_rollout":
            return default
        return resolve_component("workflow", environment.workflow)
