"""``platoon prepare``: make, once per site, what a config's runs need and never make themselves.

    platoon prepare CONFIG [key=value ...] [--dry-run]

``CONFIG`` is a training or evaluation config, or a file of ``environments`` that they extend, with overrides as
``platoon train`` takes them.  Only its environments are read, and checked as a run checks them, so a training
config that leaves ``log_path`` to its launcher prepares as it is.  For each entry, in order:

1. **The dataset.** A Harbor entry's ``dataset`` is fetched into its lock, as ``platoon harbor fetch`` fetches it: a
   dataset pinned to a commit whose lock answers the entry, and whose tasks the cache holds, is left alone, and a
   local directory is fetched again, since its files may have changed.  An entry with a hold-out prints its report.
   An entry that names a lock alone reads it; one that serves a directory has nothing to fetch.
2. **The images.** The images that the lock's runnable tasks, and their separate verifiers, build from Dockerfiles
   and that the entry's image map lacks are built with the builder of the entry's sandbox provider and recorded in
   the map: ``oci`` for ``singularity``, ``docker`` for ``docker``.  The ``process`` provider runs no image, so their
   keys are recorded as ``local``.  Any other provider pulls its images from a registry, which prepare does not
   guess: the step fails and names the command that builds and pushes them.  The map is the entry's default image
   map, ``images/<provider>.json`` in the Harbor cache, with the image files in ``images/<provider>/``
   (:mod:`platoon.envs.harbor.images`).  An entry that serves a directory builds nothing: its tasks run their
   ``docker_image`` or ``default_image``.  With ``verifier_user: honor``, an image that a verifier
   which runs as root runs on, and that the ``oci`` builder of platoon before 0.3.0 built, fails the step, naming the
   build that makes it again: prepare never replaces an image.
3. **The toolchains.** Each toolchain the entry's harness specs run from that is not built under the toolchain root
   is built there (:func:`platoon.harness.toolchains.ensure_toolchain`).  One built from another recipe is never
   replaced: a live run of another config, or of another platoon, may be staging its rollouts from it, so the step
   fails and names the forced build that replaces it (``platoon toolchain build ... --force``), to run while no job
   uses it.
4. **The tokenizer.** The policy's tokenizer is loaded once through the renderer the config names (``train`` or
   ``inference``: ``model_name``, ``renderer_name``, ``renderer_kwargs``), so the run's processes find it in the
   Hugging Face cache; skipped for a config without a renderer, a hosted model, or without the ``tinker`` extra.

Each step does nothing when what it makes is current, so prepare can run again at any time, and two at once, also
on two nodes that share the site's filesystem, take turns: a fetch on its lock's file lock, each image on one of its
own, each toolchain on its own, and the second finds the first's work current.

Each step prints a line on stdout, ``<entry> <step>: <what happened>``, and the last line says whether the config is
ready.  The exit status is 0 when it is, 1 when a step failed, and 78 on a configuration error.  With ``dry_run``
nothing changes: each step says what it would do, and the exit status is 1 when anything is missing.

``platoon train``, ``eval`` and ``rollout`` check what this makes before any rollout, and refuse with exit 78
(:class:`NotPrepared`), naming the ``platoon prepare`` command for their config.  With ``--prepare`` they run that
command first, and start only when it ends ready.
"""

from __future__ import annotations

import os
import shlex
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from platoon.utils.config import ConfigError, ResolvedConfig

if TYPE_CHECKING:
    from platoon.envs.harbor.lock import DatasetLock
    from platoon.envs.harbor.service import HarborEnvironmentService
    from platoon.harness.spec import HarnessSpec
    from platoon.train.components import EnvironmentConfig

__all__ = [
    "NotPrepared",
    "PrepareStep",
    "check_task_prepared",
    "load_environments",
    "prepare",
    "prepare_command",
    "summary",
]

Say = Callable[[str], None]
# current: nothing to do; done: made now; missing: a dry run would make it; failed: could not make it; none: nothing to
# make (a dataset directory, a task set that builds no image, ...).
Status = Literal["current", "done", "missing", "failed", "none"]


class NotPrepared(ConfigError):
    """A run needs what ``platoon prepare`` makes, and it is missing: a configuration error (exit 78), since starting
    again would not make it.  ``problems`` says what is missing, one line each; the command line adds the ``platoon
    prepare`` command for the config it ran."""

    def __init__(self, problems: str | Sequence[str], *, command: str | None = None) -> None:
        super().__init__(problems)
        self.command = command
        lines = ["what the run needs is not prepared on this site:", *self.problems]
        lines.append(f"prepare it with: {command}" if command else "prepare it with: platoon prepare CONFIG")
        self.args = ("\n  ".join(lines),)


def prepare_command(config: str | os.PathLike[str], overrides: Sequence[str] = ()) -> str:
    """The ``platoon prepare`` command line for ``config`` with ``overrides``, quoted for a shell."""

    return shlex.join(["platoon", "prepare", os.fspath(config), *overrides])


@dataclass(frozen=True)
class PrepareStep:
    """What one step of :func:`prepare` found or did."""

    where: str  # the entry, ``environments[<name>]``, or ``policy`` for the tokenizer
    step: str  # ``dataset``, ``images``, ``toolchain <name>`` or ``tokenizer``
    status: Status
    detail: str

    def line(self) -> str:
        return f"{self.where} {self.step}: {self.detail}"


def load_environments(config: str | os.PathLike[str], overrides: Sequence[str] = ()) -> ResolvedConfig:
    """The ``environments`` of a training or evaluation config, or of a file of environments, after ``extends`` and
    the overrides, checked as either kind of config checks them (:class:`~platoon.train.components.EnvironmentsConfig`),
    without the config's other sections.  ``ResolvedConfig.data`` keeps the whole config."""

    from platoon.train.components import EnvironmentsConfig
    from platoon.utils.config import build_config, find_deprecations, normalize_config, resolve_config, validate_config

    if not Path(config).is_file():
        raise ConfigError(f"{config}: no such config file")
    plain = resolve_config(config, overrides, None, warn=False)
    environments = plain.data.get("environments")
    if not environments:
        raise ConfigError(f"{config} lists no environments", source=config)
    data = normalize_config({"environments": environments}, EnvironmentsConfig)
    if problems := validate_config(data, EnvironmentsConfig):
        raise ConfigError(problems, source=config)
    try:
        loaded = build_config(EnvironmentsConfig, data)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{type(exc).__name__}: {exc}", source=config) from exc
    return ResolvedConfig(
        data=plain.data,
        config=loaded,
        schema=EnvironmentsConfig,
        sources=plain.sources,
        overrides=plain.overrides,
        deprecations=(*plain.deprecations, *find_deprecations(data, EnvironmentsConfig)),
        site=plain.site,
    )


def prepare(
    resolved: ResolvedConfig,
    *,
    dry_run: bool = False,
    command: str = "platoon prepare CONFIG",
    out: Callable[[str], None] = print,
    say: Say | None = None,
) -> list[PrepareStep]:
    """Prepare what the runs of ``resolved`` (:func:`load_environments`) need, step by step (see the module
    docstring); each step's line goes to ``out`` as it ends, and what the fetches and builds log to ``say`` (default:
    stderr).  ``command`` is the ``platoon prepare`` command line, for messages.  Raises :class:`ConfigError` for a
    configuration problem; a step that fails is returned as ``failed``."""

    from platoon.harness.registry import get_harness
    from platoon.harness.spec import HarnessSpec
    from platoon.harness.toolchains import is_named, toolchain_root

    log = say or _stderr
    steps: list[PrepareStep] = []

    def record(step: PrepareStep) -> None:
        steps.append(step)
        out(step.line())

    toolchains: set[tuple[Path, str]] = set()
    for environment in resolved.config.environments:
        where = f"environments[{environment.label}]"
        service = harbor_service(environment)
        if service is not None:
            dataset_step, lock, report = _dataset(where, service, dry_run=dry_run, say=log)
            record(dataset_step)
            if report:
                out("\n".join(report))
            if lock is not None:
                record(_images(where, service, lock, dry_run=dry_run, command=command, say=log))
            elif service.locked is not None:  # a dry run, before the dataset is fetched
                record(PrepareStep(where, "images", "missing", "not planned: the dataset is not fetched yet"))
        for ref in environment.harness_refs():
            try:
                spec = get_harness(ref)
            except (ValueError, OSError, ImportError, AttributeError) as exc:  # as a run's start refuses it
                raise ConfigError(f"{where}.harness: {exc}") from exc
            if not isinstance(spec, HarnessSpec):  # a function harness runs from this process's packages
                continue
            name = spec.install.toolchain
            if name is None or not is_named(name):
                continue
            key = (toolchain_root(), name)
            if key not in toolchains:
                toolchains.add(key)
                record(_toolchain(where, spec, ref, dry_run=dry_run, say=log))
    if (tokenizer := _tokenizer(resolved.data, dry_run=dry_run)) is not None:
        record(tokenizer)
    return steps


def summary(steps: Sequence[PrepareStep], *, dry_run: bool) -> tuple[str, int]:
    """The last line :func:`prepare`'s command prints, and its exit status."""

    counts = {status: sum(step.status == status for step in steps) for status in ("current", "done", "missing")}
    failed = [step for step in steps if step.status == "failed"]
    if failed:
        return f"not ready: {len(failed)} step(s) failed ({', '.join(f'{s.where} {s.step}' for s in failed)})", 1
    if dry_run and counts["missing"]:
        return f"not ready: {counts['missing']} step(s) would make something (a dry run changes nothing)", 1
    made = f"{counts['done']} made now, " if not dry_run else ""
    return f"ready: {made}{counts['current']} already current", 0


def harbor_service(environment: EnvironmentConfig) -> HarborEnvironmentService | None:
    """The Harbor service an entry runs, built from its ``service_kwargs`` as the run builds it (relative paths against
    the working directory); None for an entry of another service."""

    from platoon.envs.harbor.service import HarborEnvironmentService
    from platoon.envs.service import service_target
    from platoon.registry import import_from_string

    if not isinstance(environment.service, str):
        return None
    target = service_target(environment.service)
    if target is None:
        return None
    try:
        factory = import_from_string(target)
    except (ImportError, AttributeError, ValueError):
        return None  # loading the config reported it
    if not (isinstance(factory, type) and issubclass(factory, HarborEnvironmentService)):
        return None
    try:
        return factory(**environment.service_kwargs)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"environments[{environment.label}].service_kwargs: {exc}") from exc


async def check_task_prepared(environments: Sequence[EnvironmentConfig], task_id: str, split: str) -> None:
    """What a rollout of one task needs that only ``platoon prepare`` makes, for ``platoon rollout``, which never
    fetches: a Harbor entry's lock that answers it, the task's files in the cache, and the task's images.  Raises
    :class:`NotPrepared` naming what is missing, or :class:`ConfigError` for a task its service refuses to run."""

    from platoon.envs.harbor.lock import LockError
    from platoon.envs.service import EnvNotPrepared, EnvSetupError
    from platoon.train.components import split_task_id

    try:
        label, name = split_task_id(task_id) if len(environments) > 1 else (environments[0].label, task_id)
    except ValueError:
        return  # the task loader refuses it, with its own error
    environment = next((entry for entry in environments if entry.label == label), None)
    service = harbor_service(environment) if environment is not None else None
    if service is None or service.locked is None:
        return
    where = f"environments[{label}]"
    try:
        dataset = service.locked.dataset(fetch_allowed=False)
        row = dataset.row(name)
    except LockError as exc:
        raise NotPrepared(f"{where}: {exc}") from exc
    except (KeyError, ValueError) as exc:
        raise ConfigError(f"{where}: {exc}") from exc
    if not (dataset.cache / row.path).is_dir():
        raise NotPrepared(f"{where}: the cache {dataset.cache} lacks the files of task {name!r} ({row.path})")
    try:
        await service.check_tasks(split, [name])
    except EnvNotPrepared as exc:
        raise NotPrepared(f"{where}.service: {exc}") from exc
    except EnvSetupError as exc:
        raise ConfigError(f"{where}.service: {exc}") from exc


# --- the steps -------------------------------------------------------------------------------------------------


def _dataset(
    where: str, service: HarborEnvironmentService, *, dry_run: bool, say: Say
) -> tuple[PrepareStep, DatasetLock | None, list[str]]:
    """Fetch the entry's dataset into its lock (``platoon harbor fetch``'s rules), or read the lock it names; with
    the lock, the lines of its hold-out's report, if it has one."""

    from platoon.envs.harbor.fetch import FetchError, answering_lock, fetch_if_needed
    from platoon.envs.harbor.lock import LockError, read_lock

    locked = service.locked
    if locked is None:
        return PrepareStep(where, "dataset", "none", "serves a directory, which is not fetched"), None, []
    if locked.fetch is None:
        try:
            lock = read_lock(locked.lock_path)
        except LockError as exc:
            raise ConfigError(f"{where}.service_kwargs.lock: {exc}") from exc
        detail = f"current: the lock {locked.lock_path}, {_tasks(lock)} (it names no dataset to fetch)"
        return PrepareStep(where, "dataset", "current", detail), lock, _report(where, service)
    settings = locked.fetch
    local = settings.local
    moving = not local and not settings.pinned  # resolved by platoon harbor fetch --update, never here
    if dry_run or moving:
        lock, gap = _current(settings)
        if lock is None and moving:
            message = (
                f"{settings.dataset} is a moving reference, which no current lock answers: pin a commit, or resolve "
                "it with `platoon harbor fetch CONFIG --update`"
            )
            if not dry_run:
                raise ConfigError(f"{where}: {message}")
            return PrepareStep(where, "dataset", "missing", message), None, []
        if lock is None:
            return PrepareStep(where, "dataset", "missing", f"would fetch {settings.dataset}: {gap}"), None, []
        again = " (a local directory, which prepare fetches again)" if local else ""
        detail = f"current: the lock {settings.lock_path()}, {_tasks(lock)}{again}"
        step = PrepareStep(where, "dataset", "current", detail)
    else:
        fetched_before = local and answering_lock(settings) is not None  # a directory is fetched again every time
        try:
            lock, fetched = fetch_if_needed(settings, log=say)
        except FetchError as exc:
            raise ConfigError(f"{where}: {exc}") from exc
        if fetched:
            again = " again (a local directory's files may have changed)" if fetched_before else ""
            detail = f"fetched {settings.dataset}{again} into the lock {settings.lock_path()}, {_tasks(lock)}"
        else:
            detail = f"current: the lock {settings.lock_path()} answers it and the cache holds its {_tasks(lock)}"
        step = PrepareStep(where, "dataset", "done" if fetched else "current", detail)
    return step, lock, _report(where, service)


def _report(where: str, service: HarborEnvironmentService) -> list[str]:
    """The lines of the entry's hold-out report, computed from its lock (which answers it); none without one."""

    from platoon.envs.service import EnvSetupError

    if service.holdout is None:
        return []
    try:
        split = service.split_report()
    except EnvSetupError as exc:
        raise ConfigError(f"{where}: {exc}") from exc
    if split is None:
        return []
    first, *rest = split.report()
    return [f"{where} {first}", *rest]


def _current(settings: Any) -> tuple[DatasetLock | None, str]:
    """The lock that answers ``settings`` when it is current (:func:`~platoon.envs.harbor.fetch.current_lock`), else
    None and why not."""

    from platoon.envs.harbor.fetch import answering_lock, current_lock

    lock = current_lock(settings)
    if lock is not None:
        return lock, ""
    path = settings.lock_path()
    answering = answering_lock(settings)
    if answering is None:
        why = "it does not exist" if not path.exists() else "it was written for other fetch settings"
        return None, f"no lock answers it ({path}: {why})"
    if not answering.records_input_files:
        return None, f"its lock {path} records no input files (written before platoon recorded them)"
    missing = sum(1 for task in answering.tasks if not (settings.cache / task.path).is_dir())
    return (
        None,
        f"the cache {settings.cache} lacks {missing:,} of the {len(answering.tasks):,} tasks of its lock {path}",
    )


def _images(
    where: str, service: HarborEnvironmentService, lock: DatasetLock, *, dry_run: bool, command: str, say: Say
) -> PrepareStep:
    """Build the images the lock's runnable tasks need and the entry's image map lacks, with its provider's builder.
    With ``verifier_user: honor``, the step then fails for images that verifiers which run as root run on and that
    the ``oci`` builder of platoon before 0.3.0 built, naming the build that makes them again
    (:meth:`~platoon.envs.harbor.service.HarborEnvironmentService.root_images_closed`): prepare never replaces an
    image, which a run may be using."""

    step = _missing_images(where, service, lock, dry_run=dry_run, command=command, say=say)
    if step.status not in ("current", "done") or not (closed := service.root_images_closed()):
        return step
    tasks = len({name for needing in closed.values() for name in needing})
    build = f"platoon harbor images build {_config_part(command)} " + " ".join(f"--key {key}" for key in closed)
    return PrepareStep(
        where,
        "images",
        "failed",
        f"{step.detail}; but {len(closed)} of them, which {tasks} task(s) run their verifier as root on, were built by "
        "the oci builder of platoon before 0.3.0, which unpacked them under the host's umask (under one such as 027, "
        f"their files keep out the users the tests switch to): build them again with `{build}`, then run `{command}` "
        "again",
    )


def _missing_images(
    where: str, service: HarborEnvironmentService, lock: DatasetLock, *, dry_run: bool, command: str, say: Say
) -> PrepareStep:
    """Build the images the lock's runnable tasks need and the entry's image map lacks, with its provider's builder."""

    from platoon.envs.harbor.builders import PROVIDER_BUILDERS, BuildError, build_images, plan_images
    from platoon.envs.harbor.images import ImageEntry, ImageMap, ImageMapError, default_image_dir, update_image_map

    images = service.task_images
    path = images.path
    assert path is not None  # images are planned from a lock, whose entry has a default image map
    shown = f"the image map {path}"
    try:
        current = images.load()
    except ImageMapError as exc:
        raise ConfigError(f"{where}: {exc}") from exc  # the map's file names itself
    plan = plan_images(lock, current)
    if not plan.images:
        return PrepareStep(where, "images", "none", "none to build: no runnable task builds its image")
    missing = plan.missing
    if not missing:
        return PrepareStep(where, "images", "current", f"current: {shown} has the {_images_count(plan.images)}")
    provider = service.provider_name
    builder = PROVIDER_BUILDERS.get(provider)
    lacking = f"{len(missing)} of the {_images_count(plan.images)}"
    if dry_run:
        how = (
            "record as local" if provider == "process" else f"build with the {builder} builder" if builder else "build"
        )
        return PrepareStep(where, "images", "missing", f"would {how} {lacking} into {shown}")
    if provider == "process":
        entries = {
            image.key: ImageEntry(image="local", builder="alias", receipt={"aliased": True}) for image in missing
        }
        update_image_map(path, entries)
        detail = f"recorded {lacking} as local in {shown}: the process provider runs no image"
        return PrepareStep(where, "images", "done", detail)
    if builder is None:
        build = f"platoon harbor images build {_config_part(command)} --builder docker --push REGISTRY/NAME"
        return PrepareStep(
            where,
            "images",
            "failed",
            f"{lacking} are missing from {shown}, and the {provider} provider pulls its images from a registry, which "
            f"prepare does not guess: build and push them with `{build}`, then run `{command}` again",
        )
    locked = service.locked
    assert locked is not None  # images are planned from a lock
    out_dir = default_image_dir(locked.cache, provider)
    try:
        built = build_images(lock, cache=locked.cache, image_map=path, out_dir=out_dir, builder=builder, log=say)
    except BuildError as exc:
        return PrepareStep(where, "images", "failed", f"{lacking} could not all be built into {shown}: {exc}")
    left = plan_images(lock, ImageMap.load(path)).missing
    if left:
        return PrepareStep(where, "images", "failed", f"{len(left)} image(s) are still missing from {shown}")
    if not built:  # another prepare built every one while this one waited for its turn
        detail = f"current: {shown} has the {_images_count(plan.images)} (built meanwhile by another prepare)"
        return PrepareStep(where, "images", "current", detail)
    meanwhile = len(missing) - len(built)
    others = f" ({meanwhile} built meanwhile by another prepare)" if meanwhile else ""
    return PrepareStep(where, "images", "done", f"built {len(built)} with the {builder} builder into {shown}{others}")


def _toolchain(where: str, spec: HarnessSpec, ref: str, *, dry_run: bool, say: Say) -> PrepareStep:
    """Build the spec's named toolchain under the toolchain root unless one built from its recipe is there.  A
    finished one built from another recipe (or whose manifest cannot be read) is never replaced, since a live run of
    another config or platoon may be staging its rollouts from it: the step fails, naming the forced build that
    replaces it.  ``ref`` is the harness as the config names it, for that command."""

    from platoon.harness.toolchains import (
        ToolchainBuildError,
        ensure_toolchain,
        toolchain_problem,
        toolchain_root,
        toolchain_state,
    )

    name = spec.install.toolchain
    step = f"toolchain {name}"
    path = toolchain_root() / str(name)
    state = toolchain_state(spec)
    if state is None:
        return PrepareStep(where, step, "current", f"current: built at {path} for {spec.ref}")
    if not state.buildable:
        raise ConfigError(f"{where}.harness: {state.describe()}")
    if state.needs_force:
        return PrepareStep(where, step, "failed", toolchain_problem(spec, ref=ref) or state.describe())
    if dry_run:
        return PrepareStep(where, step, "missing", f"would build it at {path} for {spec.ref}: {state.describe()}")
    try:
        manifest, built = ensure_toolchain(spec, log=say)
    except ToolchainBuildError as exc:
        return PrepareStep(where, step, "failed", f"building it at {path} for {spec.ref} failed: {exc}")
    if not built:
        return PrepareStep(where, step, "current", f"current: built at {path} for {spec.ref} (meanwhile, by another)")
    return PrepareStep(where, step, "done", f"built at {path} for {spec.ref} ({manifest.tree_hash})")


def _tokenizer(data: Mapping[str, Any], *, dry_run: bool) -> PrepareStep | None:
    """Load the policy's tokenizer once through the renderer the config names, into the Hugging Face cache: current
    when the load found everything there, made when it downloaded something.  A dry run says whether the model's
    tokenizer files are cached.  A renderer setting the renderer refuses (a ``renderer_kwargs`` key, its tool-call
    parser) raises :class:`ConfigError`, as when a run starts (exit 78)."""

    name = "inference" if isinstance(data.get("inference"), Mapping) else "train"
    section = data.get(name)
    if not isinstance(section, Mapping) or section.get("hosted") is not None:
        return None
    model, renderer = section.get("model_name"), section.get("renderer_name")
    if not isinstance(model, str) or not isinstance(renderer, str) or not model or not renderer:
        return None
    step, what = "tokenizer", f"{model} through the {renderer} renderer"
    if Path(model).expanduser().is_dir():
        return PrepareStep("policy", step, "current", f"current: {model} is a local directory")
    if dry_run:
        if _cached_tokenizer(model):
            return PrepareStep("policy", step, "current", f"current: {model}'s tokenizer is in the Hugging Face cache")
        return PrepareStep("policy", step, "missing", f"would load {what} into the Hugging Face cache")
    try:
        from platoon.train.tinker.proxy import build_renderer
        from platoon.train.tinker.renderers import ToolCallParserError
    except ImportError as exc:
        return PrepareStep("policy", step, "none", f"not loaded: the tinker extra is not installed ({exc})")
    before = _cached_files(model)
    try:
        build_renderer(model, renderer, dict(section.get("renderer_kwargs") or {}))
    except ToolCallParserError as exc:  # the config's, which the run's start refuses too (platoon.cli)
        raise ConfigError(f"{name}.renderer_kwargs.tool_call_parser: {exc}") from exc
    except ValueError as exc:
        if "renderer_kwargs" in str(exc):  # a key the renderer does not take
            raise ConfigError(f"{name}.renderer_kwargs: {exc}") from exc
        return PrepareStep("policy", step, "failed", f"{what}: {type(exc).__name__}: {exc}")
    except Exception as exc:  # no network, no access, an unknown model: the run would fail on it too
        return PrepareStep("policy", step, "failed", f"{what}: {type(exc).__name__}: {exc}")
    if before and _cached_files(model) == before:
        return PrepareStep("policy", step, "current", f"current: loaded {what} from the Hugging Face cache")
    return PrepareStep("policy", step, "done", f"loaded {what}, into the Hugging Face cache")


def _cached_files(model: str) -> set[str]:
    """The files of ``model``'s repository in the Hugging Face cache (its blobs), to tell whether a load added any."""

    from huggingface_hub import constants

    folder = Path(constants.HF_HUB_CACHE) / f"models--{model.replace('/', '--')}" / "blobs"
    return {entry.name for entry in folder.iterdir()} if folder.is_dir() else set()


def _cached_tokenizer(model: str) -> bool:
    """Whether the Hugging Face cache holds ``model``'s tokenizer configuration."""

    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(model, "tokenizer_config.json"), str)


# --- helpers ---------------------------------------------------------------------------------------------------


def _tasks(lock: DatasetLock) -> str:
    return f"{len(lock.tasks):,} task(s), {len(lock.runnable()):,} runnable"


def _images_count(images: Sequence[Any]) -> str:
    return f"{len(images)} image(s) the runnable tasks build"


def _config_part(command: str) -> str:
    """The config and overrides of a ``platoon prepare`` command line, for another command."""

    words = shlex.split(command)
    return shlex.join(words[2:]) if words[:2] == ["platoon", "prepare"] else "CONFIG"


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
