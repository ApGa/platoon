"""``platoon harbor``: fetch Harbor datasets into a locked cache, verify them, and plan and build their images.

    platoon harbor fetch CONFIG [key=value ...] [--update | --repair]   fetch and lock the datasets a config names
    platoon harbor verify CONFIG [key=value ...]          check the cache entries of the locks against their digests
    platoon harbor images plan CONFIG [key=value ...]     the images the tasks need, and which the image map lacks
    platoon harbor images build CONFIG [key=value ...]    build the missing images and record them in the image map
    platoon harbor images verify CONFIG [key=value ...]   check the map's image files against their SHA-256
    platoon harbor images alias CONFIG [key=value ...] --image IMAGE   point missing keys at an existing image

``CONFIG`` is a training or evaluation config, whose Harbor environment entries name their datasets (each entry's
paths resolve as ``platoon train`` resolves them, overrides included); ``verify`` and the images commands also take
a lock file, the images commands with ``--image-map``.  The cache defaults to the entry's, else
``harbor`` in the site profile's cache (:mod:`platoon.site`), else ``~/.cache/platoon/harbor``, and a lock-backed
entry's image map to
``images/<provider>.json`` in its cache, whose images go to ``images/<provider>/``.

A fetch config file, which names one dataset and resolves its paths against itself, still works where a config or a
lock does, with a deprecation warning: a training or evaluation config's entry names the same, and platoon 0.3.0
removes fetch config files.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from cyclopts import App

from platoon.utils.config import ConfigError

if TYPE_CHECKING:
    from platoon.envs.harbor.fetch import FetchConfig
    from platoon.envs.harbor.images import ImageMap
    from platoon.envs.harbor.lock import DatasetLock
    from platoon.envs.harbor.service import HarborEnvironmentService

# The release that removes fetch config files (deprecated: a training or evaluation config's entry says the same).
FETCH_CONFIG_REMOVED_IN = "0.3.0"

harbor_app = App(name="harbor", help="Fetch Harbor datasets into a locked cache, verify them, and build their images.")
images_app = App(name="images", help="Plan, build and verify the images of a lock's tasks, once per site.")
harbor_app.command(images_app)


def _say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


@harbor_app.command
def fetch(
    config: Path,
    /,
    *overrides: str,
    update: bool = False,
    repair: bool = False,
    lock: Path | None = None,
    cache: Path | None = None,
    workers: int | None = None,
) -> int:
    """Fetch the Harbor datasets a config names into the cache, and lock them.

    CONFIG is a training or evaluation config, whose Harbor entries with a dataset are fetched (a fetch config file,
    deprecated, is fetched too). A dataset pinned to a commit whose lock answers the config, with a cache that holds
    its tasks, is left as it is: nothing is downloaded or hashed, unless the lock was written before platoon recorded
    the tasks' input files. A local directory is fetched again, since its files may have changed. Fetches of one lock
    take turns, also on several nodes, so a second one finds the first's lock current. Each entry with a hold-out
    prints the hold-out's report. Prints one JSON line per lock.

    Parameters
    ----------
    config
        A training or evaluation config (or a fetch config file, deprecated).
    overrides
        key=value overrides of a training or evaluation config, as platoon train takes them.
    update
        Resolve a moving reference (a branch, a tag, no revision) to its current commit, and fetch it.
    repair
        Fetch a pinned dataset even when its lock answers: hash every cache entry again, and replace a damaged one.
    lock
        A fetch config file's lock: write it here instead.
    cache
        A fetch config file's cache: use this one instead.
    workers
        Parallel downloads (default 8).
    """

    from dataclasses import replace

    from platoon.envs.harbor.fetch import FetchError, answering_lock, fetch_if_needed, is_pinned

    if workers is not None and workers < 1:
        raise ConfigError(f"--workers must be a positive integer, got {workers}")
    for label, settings, service in _fetch_targets(config, overrides, lock=lock, cache=cache):
        if workers is not None:
            settings = replace(settings, workers=workers)
        where = f"environments[{label}]: " if label is not None else ""
        remote = settings.dataset.startswith("hf://") and is_pinned(settings.dataset)  # its files cannot change
        if remote and not (update or repair) and (old := answering_lock(settings)) and not old.records_input_files:
            _say(
                f"{where}the lock {settings.lock_path()} records no input files, as locks written before platoon "
                "recorded them do: fetching again to record them"
            )
        try:
            result, fetched = fetch_if_needed(settings, update=update, repair=repair, log=_say)
        except FetchError as exc:
            raise ConfigError(f"{where}{exc}") from exc
        if not fetched:
            _say(
                f"{where}the lock {settings.lock_path()} answers the config and the cache holds its "
                f"{len(result.tasks)} task(s): nothing to fetch (--repair hashes them again)"
            )
        line: dict[str, Any] = {} if label is None else {"environment": label}
        line.update(
            lock=str(settings.lock_path()), tasks=len(result.tasks), runnable=len(result.runnable()), fetched=fetched
        )
        if service is not None and (split := _split_report(service, label)) is not None:
            first, *rest = split.report()
            _say("\n".join([f"{where}{first}", *rest]))
            line["holdout"] = {"eval": len(split.eval), "train": len(split.train), "eval_digest": split.eval_digest}
        print(json.dumps(line))
    return 0


@harbor_app.command
def verify(config: Path, /, *overrides: str, cache: Path | None = None, task: list[str] | None = None) -> int:
    """Check the cache entries of a lock: every task's files must hash to its digest. Exits 1 on a mismatch.

    CONFIG is a training or evaluation config: the lock of each Harbor entry that names a dataset (the lock that
    answers it) or a lock; or a lock file (or a fetch config file, deprecated). An entry with a hold-out whose lock
    records no input files (one written before platoon recorded them) exits 78, as before a fetch.

    Parameters
    ----------
    config
        A training or evaluation config, or a lock file.
    overrides
        key=value overrides of a training or evaluation config.
    cache
        The cache a lock file's or fetch config's entries are in.
    task
        Check only these task ids.
    """

    from platoon.envs.harbor.lock import verify_lock

    if _is_run_config(config):
        if cache is not None:
            raise ConfigError(_RUN_CONFIG_PATHS)
        targets = _verify_targets(config, overrides)
    else:
        if overrides:
            raise ConfigError(_NO_OVERRIDES)
        targets = [(None, *_open_lock(config, cache))]
    wanted = list(task or ())
    if unknown := [name for name in wanted if not any(name in dataset_lock for _, dataset_lock, _ in targets)]:
        raise ConfigError(f"--task {unknown}: no lock lists them")
    failed = False
    for label, dataset_lock, cache_dir in targets:
        names = [name for name in wanted if name in dataset_lock] if wanted else None
        if names == []:
            continue
        problems = list(verify_lock(dataset_lock, cache_dir, names=names))
        for problem in problems:
            print(problem)
        checked = len(names) if names else len(dataset_lock.tasks)
        where = f"environments[{label}]: " if label is not None else ""
        print(f"{where}{checked - len(problems)} of {checked} task(s) match their digests", file=sys.stderr)
        failed = failed or bool(problems)
    return 1 if failed else 0


_RUN_CONFIG_PATHS = (
    "--lock and --cache apply to fetch config files; a training or evaluation config names its own, in an entry's "
    "service_kwargs (override environments.N.service_kwargs.cache=...), else the site profile's"
)
_RUN_CONFIG_IMAGE_PATHS = (
    "--image-map and --cache apply to lock files; a training or evaluation config's entries name their own in "
    "service_kwargs (image_map, cache), and a lock-backed entry's image map defaults to images/<provider>.json in its "
    "cache"
)
_NO_OVERRIDES = "key=value overrides apply to training and evaluation configs, not to a fetch config or a lock file"


def _is_run_config(path: Path) -> bool:
    """Whether ``path`` is a training or evaluation config (it lists environments or extends another file) rather
    than a fetch config file or a lock."""

    if path.suffix not in (".yaml", ".yml"):
        return False
    import yaml

    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    return isinstance(data, Mapping) and ("environments" in data or "extends" in data)


def _fetch_targets(
    config: Path, overrides: Sequence[str], *, lock: Path | None, cache: Path | None
) -> list[tuple[str | None, FetchConfig, HarborEnvironmentService | None]]:
    """What ``fetch`` fetches: each Harbor entry of a run config that names a dataset, with its service, or a fetch
    config file's one dataset."""

    from platoon.envs.harbor.fetch import FetchError, load_fetch_config

    if not _is_run_config(config):
        if overrides:
            raise ConfigError(_NO_OVERRIDES)
        _warn_fetch_config(config)
        try:
            return [(None, load_fetch_config(config, lock=lock, cache=cache), None)]
        except FetchError as exc:
            raise ConfigError(str(exc)) from exc
    if lock is not None or cache is not None:
        raise ConfigError(_RUN_CONFIG_PATHS)
    targets: list[tuple[str | None, FetchConfig, HarborEnvironmentService | None]] = []
    for label, service in _harbor_entries(config, overrides):
        if service.locked is None or service.locked.fetch is None:
            _say(f"environments[{label}]: names no dataset to fetch (dataset_dir, task_dirs or a lock alone)")
            continue
        targets.append((label, service.locked.fetch, service))
    if not targets:
        raise ConfigError(f"{config}: no Harbor environment entry names a dataset to fetch")
    return targets


def _verify_targets(config: Path, overrides: Sequence[str]) -> list[tuple[str | None, Any, Path]]:
    """Each Harbor entry of a run config with a lock: the lock that answers its dataset, or the lock it names."""

    from platoon.envs.harbor.fetch import answering_lock
    from platoon.envs.harbor.lock import LockError, read_lock

    targets: list[tuple[str | None, Any, Path]] = []
    for label, service in _harbor_entries(config, overrides):
        locked = service.locked
        if locked is None:
            _say(f"environments[{label}]: serves a directory, which has no lock to verify")
            continue
        if locked.fetch is not None:
            dataset_lock = answering_lock(locked.fetch)
            if dataset_lock is None:
                raise ConfigError(
                    f"environments[{label}]: no lock answers its dataset settings ({locked.lock_path} does not exist "
                    f"or was written for others): run `platoon harbor fetch {config}` first"
                )
        else:
            try:
                dataset_lock = read_lock(locked.lock_path)
            except LockError as exc:
                raise ConfigError(f"environments[{label}]: {exc}") from exc
        if service.holdout is not None and not dataset_lock.records_input_files:
            fix = f"run `platoon harbor fetch {config}` first" if locked.fetch is not None else "fetch it again"
            raise ConfigError(
                f"environments[{label}]: the lock {locked.lock_path} records no input files, which its hold-out "
                f"needs to keep the tasks that share one on one side: {fix}"
            )
        targets.append((label, dataset_lock, locked.cache))
    if not targets:
        raise ConfigError(f"{config}: no Harbor environment entry has a lock to verify")
    return targets


def _harbor_entries(config: Path, overrides: Sequence[str]) -> list[tuple[str, HarborEnvironmentService]]:
    """The Harbor entries of a training or evaluation config, each with the service ``platoon train`` builds from its
    ``service_kwargs``: paths resolve as the run resolves them, the overrides apply, and the site profile gives an
    entry without a provider its own.  The deprecated keys and variables it uses are named on stderr."""

    from platoon.cli import _warn_of_deprecations
    from platoon.envs.harbor.service import HarborEnvironmentService
    from platoon.envs.service import service_target
    from platoon.registry import import_from_string
    from platoon.train.components import EnvironmentsConfig
    from platoon.utils.config import find_deprecations, resolve_config

    resolved = resolve_config(config, list(overrides), warn=False)
    data = resolved.data
    environments = data.get("environments")
    _warn_of_deprecations(
        [*resolved.deprecations, *find_deprecations({"environments": environments or []}, EnvironmentsConfig)]
    )
    if not isinstance(environments, list) or not environments:
        raise ConfigError(f"{config} lists no environments")
    entries: list[tuple[str, HarborEnvironmentService]] = []
    for entry in environments:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("service"), str):
            continue
        target = service_target(entry["service"])
        try:
            factory = import_from_string(target) if target else None
        except (ImportError, AttributeError, ValueError):
            continue  # not a service this command knows; platoon train reports what is wrong with it
        if not (isinstance(factory, type) and issubclass(factory, HarborEnvironmentService)):
            continue
        label = str(entry.get("name") or "default")  # as the entry's label in task ids and messages
        try:
            entries.append((label, factory(**dict(entry.get("service_kwargs") or {}))))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"environments[{label}].service_kwargs: {exc}") from exc
    if not entries:
        raise ConfigError(f"{config} has no Harbor environment entry")
    return entries


def _split_report(service: HarborEnvironmentService, label: str | None):
    """The entry's hold-out, computed from the lock just fetched; None without one."""

    from platoon.envs.service import EnvSetupError

    try:
        return service.split_report()
    except EnvSetupError as exc:
        raise ConfigError(f"environments[{label}]: {exc}") from exc


@dataclass(frozen=True)
class _ImageTarget:
    """What an images command acts on: a lock, the cache its tasks are in, and the image map of its images."""

    label: str | None  # the entry of a training or evaluation config; None for a lock or fetch config file
    lock: DatasetLock
    cache: Path
    image_map: Path | None  # the map's file: None for an entry's inline map, or a lock file given no --image-map
    inline: Mapping[str, Any] | None = None  # an entry's inline image map
    provider: str | None = None  # an entry's sandbox provider
    default: bool = False  # an entry's default image map (images/<provider>.json in its cache)

    @property
    def where(self) -> str:
        return f"environments[{self.label}]: " if self.label is not None else ""

    def current_map(self) -> ImageMap | None:
        """The image map as it is now; None before its file exists."""

        from platoon.envs.harbor.images import ImageMap, ImageMapError

        if self.inline is not None:
            try:
                return ImageMap.load(self.inline)
            except ImageMapError as exc:
                raise ConfigError(f"{self.where}service_kwargs.image_map: {exc}") from exc
        return _open_map(self.image_map)

    def map_file(self, command: str) -> Path:
        """The map's file, which ``command`` writes or reads; a configuration error for an inline map, or a lock file
        given no ``--image-map``."""

        if self.image_map is not None:
            return self.image_map
        if self.inline is not None:
            raise ConfigError(
                f"{self.where}its image map is inline in service_kwargs.image_map, which platoon harbor images "
                f"{command} cannot write: name a JSON file there instead"
            )
        raise ConfigError(f"platoon harbor images {command} of a lock file needs --image-map")


def _image_targets(
    config: Path, overrides: Sequence[str], *, image_map: Path | None, cache: Path | None
) -> list[_ImageTarget]:
    """The locks an images command acts on: each Harbor entry of a training or evaluation config that has one (with
    its image map: the entry's ``image_map``, else its default one), or a lock file (or a fetch config file's lock)
    with ``--image-map``."""

    from platoon.envs.harbor.fetch import answering_lock
    from platoon.envs.harbor.lock import LockError, read_lock

    if not _is_run_config(config):
        if overrides:
            raise ConfigError(_NO_OVERRIDES)
        dataset_lock, cache_dir = _open_lock(config, cache)
        return [_ImageTarget(None, dataset_lock, cache_dir, image_map)]
    if image_map is not None or cache is not None:
        raise ConfigError(_RUN_CONFIG_IMAGE_PATHS)
    targets = []
    for label, service in _harbor_entries(config, overrides):
        locked = service.locked
        if locked is None:
            _say(f"environments[{label}]: serves a directory, whose tasks run their own images")
            continue
        if locked.fetch is not None:
            dataset_lock = answering_lock(locked.fetch)
            if dataset_lock is None:
                raise ConfigError(
                    f"environments[{label}]: no lock answers its dataset settings ({locked.lock_path} does not exist "
                    f"or was written for others): run `platoon harbor fetch {config}` first"
                )
        else:
            try:
                dataset_lock = read_lock(locked.lock_path)
            except LockError as exc:
                raise ConfigError(f"environments[{label}]: {exc}") from exc
        images = service.task_images
        targets.append(
            _ImageTarget(
                label,
                dataset_lock,
                locked.cache,
                images.path,
                inline=images.inline,
                provider=service.provider_name,
                default=images.is_default,
            )
        )
    if not targets:
        raise ConfigError(f"{config}: no Harbor environment entry has a lock")
    return targets


@images_app.command(name="plan")
def images_plan(config: Path, /, *overrides: str, image_map: Path | None = None, json_output: bool = False) -> int:
    """List the images the tasks need: one per build key, with its task count, base images and whether the image map
    has it.

    CONFIG is a training or evaluation config: each Harbor entry's lock, with the entry's image map (its image_map,
    else `images/<provider>.json` in its cache); or a lock file, with --image-map.

    Parameters
    ----------
    config
        A training or evaluation config, or a lock file.
    overrides
        key=value overrides of a training or evaluation config.
    image_map
        A lock file's image map (JSON); without it, or before it exists, every image is missing.
    json_output
        Print the plan as JSON (for a training or evaluation config, keyed by entry).
    """

    from platoon.envs.harbor.builders import plan_images

    targets = _image_targets(config, overrides, image_map=image_map, cache=None)
    plans = [(target, plan_images(target.lock, target.current_map())) for target in targets]
    if json_output:
        if len(plans) == 1 and plans[0][0].label is None:
            print(json.dumps(plans[0][1].to_json(), indent=1))
        else:
            maps = {target.label: str(target.image_map) if target.image_map else "inline" for target, _ in plans}
            data = {target.label: {"image_map": maps[target.label], **plan.to_json()} for target, plan in plans}
            print(json.dumps(data, indent=1))
        return 0
    for target, plan in plans:
        if target.label is not None:
            print(f"{target.where}image map {target.image_map or 'inline'}")
        rows = [
            (
                image.key[:19],
                image.role,
                image.kind,
                len(image.tasks),
                ", ".join(image.base) or "-",
                "yes" if image.present else "no",
                image.example.name if image.example else "-",
            )
            for image in plan.images
        ]
        print(_table(("key", "role", "kind", "tasks", "base", "in map", "example"), rows))
        summary = f"{len(plan.images)} image(s), {len(plan.missing)} missing from the image map"
        if plan.prebuilt:
            summary += f"; prebuilt references: {plan.prebuilt}"
        if plan.default_image:
            summary += f"; {plan.default_image} task(s) without an image (the service's default_image)"
        if plan.refused:
            summary += f"; refused tasks: {plan.refused}"
        print(summary)
    return 0


@images_app.command(name="build")
def images_build(
    config: Path,
    /,
    *overrides: str,
    image_map: Path | None = None,
    cache: Path | None = None,
    builder: Literal["oci", "def", "docker"] | None = None,
    shard: str = "0/1",
    out: Path | None = None,
    key: list[str] | None = None,
    scratch: Path | None = None,
    timeout: float = 3600.0,
    push: str | None = None,
) -> int:
    """Build the images the image map lacks and add each to the map, with a receipt (builder, version, time,
    log, size, SHA-256). Exits 1 if any build failed.

    CONFIG is a training or evaluation config: each Harbor entry's lock, into the entry's image map (its image_map,
    else `images/<provider>.json` in its cache, with the images in `images/<provider>/`), with the builder of the
    entry's sandbox provider; or a lock file, with --image-map.  Builds of one map that run at once, also on several
    nodes, build each image once.

    Parameters
    ----------
    config
        A training or evaluation config, or a lock file.
    overrides
        key=value overrides of a training or evaluation config.
    image_map
        A lock file's image map (JSON), created if missing.
    cache
        The cache a lock file's tasks are in.
    builder
        oci (BuildKit through singularity build --oci, then a native SIF), def (a fakeroot definition file), or
        docker (docker build; with push, pushed and recorded by digest). Default: the entry's provider's (oci for
        singularity, docker for docker); oci for a lock file.
    shard
        i/n: build every n-th missing key starting at i, for job arrays.
    out
        Where image files go (default: images/ next to the image map; `images/<provider>/` next to a default one).
    key
        Build these keys (rebuilding them if present) instead of the missing ones.
    scratch
        Node-local directory for build trees (default $PLATOON_NODE_SCRATCH, $PBS_LOCALDIR, $SLURM_TMPDIR, $TMPDIR,
        /tmp).
    timeout
        Seconds each build command may take.
    push
        docker builder: a repository to push to (registry/name).
    """

    from platoon.envs.harbor.builders import PROVIDER_BUILDERS, BuildError, build_images, plan_images
    from platoon.envs.harbor.images import default_image_dir

    try:
        index, count = (int(part) for part in shard.split("/"))
    except ValueError:
        raise ConfigError(f"--shard {shard!r}: expected i/n") from None
    targets = _image_targets(config, overrides, image_map=image_map, cache=cache)
    wanted = list(key or ())
    plans = {id(target): {image.key for image in plan_images(target.lock).images} for target in targets}
    if unknown := [name for name in wanted if not any(name in keys for keys in plans.values())]:
        raise ConfigError(f"--key {unknown}: the runnable tasks use no such build key")
    failed = False
    for target in targets:
        keys = [name for name in wanted if name in plans[id(target)]] if wanted else None
        if keys == []:
            continue
        chosen = builder or ("oci" if target.label is None else PROVIDER_BUILDERS.get(target.provider or ""))
        if chosen is None:
            raise ConfigError(
                f"{target.where}the {target.provider} provider has no builder of its own: for a provider that pulls "
                "its images from a registry, build and push them with --builder docker --push REGISTRY/NAME; for one "
                "that runs no image, point the keys at one with platoon harbor images alias"
            )
        map_file = target.map_file("build")
        assert target.provider is not None or not target.default
        out_dir = out or (default_image_dir(target.cache, target.provider or "") if target.default else None)
        try:
            built = build_images(
                target.lock,
                cache=target.cache,
                image_map=map_file,
                out_dir=out_dir,
                builder=chosen,
                shard=(index, count),
                keys=keys,
                scratch=scratch,
                timeout_s=timeout,
                push=push,
                log=_say,
            )
        except BuildError as exc:
            _say(f"platoon harbor images build: {target.where}{exc}")
            failed = True
            continue
        except ValueError as exc:
            raise ConfigError(f"{target.where}{exc}") from exc
        images = {name: entry.image for name, entry in built.items()}
        if target.label is None:
            print(json.dumps(images, indent=1))
        else:
            print(json.dumps({"environment": target.label, "image_map": str(map_file), "built": images}))
    return 1 if failed else 0


@images_app.command(name="verify")
def images_verify(config: Path, /, *overrides: str, image_map: Path | None = None) -> int:
    """Check the image map against the images the runnable tasks use: each must be in the map, and a local image
    file must exist and hash to the SHA-256 its entry records. Exits 1 if any is missing or mismatched.

    CONFIG is a training or evaluation config (each Harbor entry's lock, with its image map), or a lock file, with
    --image-map.  Each problem is a line on stdout. A registry reference cannot be hashed here, nor can a file whose
    entry records no SHA-256: they are listed on stderr as not checked, and do not fail the check.

    Parameters
    ----------
    config
        A training or evaluation config, or a lock file.
    overrides
        key=value overrides of a training or evaluation config.
    image_map
        A lock file's image map (JSON).
    """

    from platoon.envs.harbor.builders import verify_images
    from platoon.envs.harbor.images import ImageMap

    failed = False
    for target in _image_targets(config, overrides, image_map=image_map, cache=None):
        if target.inline is None and not target.default:  # a default map that does not exist yet has no images
            path = target.map_file("verify")
            if not path.exists():
                raise ConfigError(f"{target.where}image map {path} does not exist")
        site_map = target.current_map() or ImageMap()
        checks = verify_images(target.lock, site_map)
        counts = {status: 0 for status in ("ok", "missing", "unchecked")}
        for check in checks:
            counts[check.status] += 1
            if check.status != "ok":
                line = f"{target.where}{check.image.key} ({check.image.role}, {len(check.image.tasks)} task(s)): "
                print(line + check.detail, file=sys.stdout if check.status == "missing" else sys.stderr)
        print(
            f"{target.where}{len(checks)} image(s): {counts['ok']} match their SHA-256, {counts['missing']} missing or "
            f"mismatched, {counts['unchecked']} not checked",
            file=sys.stderr,
        )
        failed = failed or bool(counts["missing"])
    return 1 if failed else 0


@images_app.command(name="alias")
def images_alias(
    config: Path, /, *overrides: str, image: str, image_map: Path | None = None, key: list[str] | None = None
) -> int:
    """Point the image map's missing keys (or KEY) at an existing image: a prebuilt SIF that matches the build,
    or ``local`` for the process provider, which runs no image.

    CONFIG is a training or evaluation config (each Harbor entry's lock, into its image map), or a lock file, with
    --image-map.

    Parameters
    ----------
    config
        A training or evaluation config, or a lock file.
    overrides
        key=value overrides of a training or evaluation config.
    image
        The image reference or file to record.
    image_map
        A lock file's image map (JSON), created if missing.
    key
        Alias these keys instead of the missing ones.
    """

    from platoon.envs.harbor.builders import plan_images
    from platoon.envs.harbor.images import ImageEntry, update_image_map

    aliased: list[str] = []
    for target in _image_targets(config, overrides, image_map=image_map, cache=None):
        map_file = target.map_file("alias")
        plan = plan_images(target.lock, target.current_map())
        keys = [name for name in key if name in {i.key for i in plan.images}] if key else [i.key for i in plan.missing]
        entries = {name: ImageEntry(image=image, builder="alias", receipt={"aliased": True}) for name in keys}
        update_image_map(map_file, entries)
        aliased += sorted(entries)
    print(json.dumps(aliased, indent=1))
    return 0


def _open_lock(path: Path, cache: Path | None):
    """The lock at ``path`` (or named by the fetch config file at ``path``, which is deprecated) and its cache
    directory."""

    from platoon.envs.harbor.fetch import FetchError, load_fetch_config
    from platoon.envs.harbor.lock import LockError, default_cache, read_lock

    try:
        if path.suffix in (".yaml", ".yml"):
            _warn_fetch_config(path)
            settings = load_fetch_config(path, cache=cache)
            return read_lock(settings.lock_path()), settings.cache
        if cache is None:
            _check_site_variables()
        return read_lock(path), cache if cache is not None else default_cache()
    except (FetchError, LockError) as exc:
        raise ConfigError(str(exc)) from exc


def _check_site_variables() -> None:
    """For a command that takes the Harbor cache from the site: refuse a deprecated cache variable that names another
    directory than the site profile's cache gives it, and name each one that is set on stderr."""

    from platoon.cli import _warn_of_deprecations
    from platoon.site import variable_deprecations, variable_problems

    if problems := variable_problems():
        raise ConfigError(problems)
    _warn_of_deprecations(variable_deprecations())


def _warn_fetch_config(path: Path) -> None:
    """Say, once per command, that a fetch config file is deprecated: a run config's entry says the same."""

    from platoon.cli import _warn_of_deprecations
    from platoon.deprecations import Deprecation

    deprecation = Deprecation(
        path=f"{path} (a fetch config file)",
        replacement="a training or evaluation config, whose Harbor entry names the dataset and its fetch settings",
        removed_in=FETCH_CONFIG_REMOVED_IN,
    )
    _warn_of_deprecations([deprecation])
    _check_site_variables()  # its cache defaults to the site's


def _open_map(path: Path | None):
    from platoon.envs.harbor.images import ImageMap, ImageMapError

    if path is None or not path.exists():
        return None  # no map yet: every image is missing
    try:
        return ImageMap.load(path)
    except ImageMapError as exc:
        raise ConfigError(str(exc)) from exc


def _table(header: tuple[str, ...], rows: list[tuple]) -> str:
    cells = [[str(cell) for cell in header], *[[str(cell) for cell in row] for row in rows]]
    widths = [max(len(row[column]) for row in cells) for column in range(len(header))]
    return "\n".join("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in cells)
