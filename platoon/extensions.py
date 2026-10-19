"""Extensions: installed distributions that add components to platoon through entry points.

An extension declares each component it provides as one entry point in the group of the component's kind, named
exactly as configs name the component::

    [project.entry-points."platoon.harnesses"]
    acme-agent = "acme_platoon.agent:acme_agent"                 # an @pt.harness function, or a TOML spec path
    [project.entry-points."platoon.programs"]
    acme-best-of-n = "acme_platoon.programs:best_of_n"           # an @pt.program function
    [project.entry-points."platoon.environments"]
    acme-mazes = "acme_platoon.entries:MAZES"                    # a pt.EnvironmentEntry (a named environment)
    [project.entry-points."platoon.reward_processors"]
    acme-length-penalty = "acme_platoon.rewards:length_penalty"  # fn(trajectory) -> (score, components)

An entry point may also load to a list of components of its kind that share its name (several versions), and a
harness or program entry point to a module whose import registers them.

**Discovery.**  A reference that holds ``:`` is an import string (``module:attr``) and skips the lookup.  A name,
``name[@version]``, resolves in this order:

1. core's built-ins, whose names are reserved: the ``platoon`` owner (``platoon-*``) and :data:`CORE_NAMES`;
2. the entry point of the kind's group named exactly ``name``, from installed metadata (scanned once per
   process), imported the first time a config or a program names it.  Nothing else of an extension is imported,
   so installing one never changes what a run that does not name its components does;
3. names that modules a config imports with ``package:`` registered.

A bare name resolves only when exactly one version is registered; ``name@version`` picks one, and a version
prefix matches at dots (``@2`` matches ``2.4.6``).  :func:`list_components` and :func:`show_components` read
metadata only (``show_components`` imports an entry point when asked to load it).

**Names.**  :data:`NAME_PATTERN`: lower-case words of letters and digits joined by ``.``, ``_`` or ``-``, never
ending in ``.toml``.  The first ``-``-separated word is the name's owner, one per distribution
(``platoon extensions check`` warns otherwise).  The ``platoon`` owner and :data:`CORE_NAMES` belong to core: only
code in the ``platoon`` package, and the project's own ``platoon-*`` distributions, may register them.

**Compatibility.**  Before it imports an entry point, platoon reads the distribution's requirement on ``platoon``
from its installed metadata and refuses an extension whose version range excludes the running platoon (an
:class:`ExtensionError` that names both).  :data:`IGNORE_REQUIREMENTS_ENV` set to ``1`` skips the check, for
development against a checkout.

**Provenance.**  Every component a run names is resolved when the run starts, and ``lock.json`` records the
distribution and version behind each one (:func:`run_components`).
"""

from __future__ import annotations

import dataclasses
import difflib
import functools
import importlib
import importlib.metadata
import json
import logging
import os
import re
import sys
import threading
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import Distribution, EntryPoint
from pathlib import Path
from typing import Any, Literal, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Component kind -> the entry-point group extensions declare it in.
GROUPS: dict[str, str] = {
    "harness": "platoon.harnesses",
    "program": "platoon.programs",
    "environment": "platoon.environments",
    "reward_processor": "platoon.reward_processors",
}
KINDS: tuple[str, ...] = tuple(GROUPS)
# Groups platoon no longer reads, with what replaced them; ``platoon extensions check`` flags their entries.
RETIRED_GROUPS: dict[str, str] = {
    "platoon.plugins": "declare each component in the group of its kind (platoon.harnesses, platoon.programs, "
    "platoon.environments, platoon.reward_processors), named as configs name it",
}
# The owner that belongs to core: ``platoon`` and every ``platoon-*`` name.
RESERVED_OWNER = "platoon"
# Core's names without the ``platoon-`` prefix: the short names it had before the naming rule, and the product
# names of the third-party agents it ships specs for.  Reserved for core like the prefix.
CORE_NAMES: dict[str, frozenset[str]] = {
    "harness": frozenset({"mini-swe-agent", "openhands"}),
    "program": frozenset({"single", "gym"}),
    "environment": frozenset(),
    "reward_processor": frozenset(),
}
NAME_PATTERN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
VERSION_PATTERN = re.compile(r"[0-9A-Za-z]+(?:[._+-][0-9A-Za-z]+)*")
NAME_RULE = "lower-case letters and digits in words joined by '.', '_' or '-' (not ending in .toml)"
# Set to 1 to load extensions whose platoon requirement excludes the running platoon (development only).
IGNORE_REQUIREMENTS_ENV = "PLATOON_IGNORE_EXTENSION_REQUIREMENTS"

_PLATOON_DIR = Path(__file__).resolve().parent


class ExtensionError(ValueError):
    """A component that cannot be resolved: unknown, ambiguous, refused by its compatibility check, or broken."""


# --- names -----------------------------------------------------------------------------------------------------


def check_name(name: object, *, what: str = "component") -> str:
    """``name`` if it follows :data:`NAME_PATTERN`; else a ``ValueError`` that says the rule."""

    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name) or name.endswith(".toml"):
        raise ValueError(f"{what} name {name!r} must be {NAME_RULE}, e.g. 'acme-agent'")
    return name


def check_version(version: object, *, what: str = "component") -> str:
    """``version`` as a string if it follows :data:`VERSION_PATTERN`; else a ``ValueError``."""

    text = str(version) if isinstance(version, (int, str)) and not isinstance(version, bool) else None
    if text is None or not VERSION_PATTERN.fullmatch(text):
        raise ValueError(f"{what} version {version!r} must be letters and digits joined by '.', '_', '+' or '-'")
    return text


def split_ref(ref: str) -> tuple[str, str | None]:
    """``(name, version)`` of ``name[@version]``; the version is None when the reference gives none."""

    name, separator, version = ref.partition("@")
    return name, (version if separator else None)


def owner(name: str) -> str:
    """The owner of a component name: its first ``-``-separated word."""

    return name.split("-", 1)[0]


def is_reserved(kind: str, name: str) -> bool:
    """Whether ``name`` belongs to core: the ``platoon`` owner, or one of :data:`CORE_NAMES` of ``kind``."""

    return owner(name) == RESERVED_OWNER or name in CORE_NAMES.get(kind, frozenset())


def may_use_reserved_names(*, module: str | None = None, distribution: str | None = None) -> bool:
    """Whether code in ``module``, or an entry point of ``distribution``, may register core's names: code in the
    ``platoon`` package, and the project's own distributions (``platoon`` and ``platoon-*``)."""

    if module is not None and (module == RESERVED_OWNER or module.startswith(RESERVED_OWNER + ".")):
        return True
    if distribution is not None:
        canonical = canonical_name(distribution)
        return canonical == RESERVED_OWNER or canonical.startswith(RESERVED_OWNER + "-")
    return False


def check_registration(
    kind: str, name: str, *, module: str | None = None, distribution: str | None = None, what: str | None = None
) -> None:
    """Refuse a ``kind`` named ``name`` that breaks the naming rule, or that takes a name of core's from outside
    core (registered by code in ``module``, or by an entry point of ``distribution``)."""

    label = what or kind.replace("_", " ")
    check_name(name, what=label)
    if is_reserved(kind, name) and not may_use_reserved_names(module=module, distribution=distribution):
        if distribution is not None:
            source = f"distribution {distribution!r}"
        else:
            source = f"module {module!r}" if module is not None else "outside platoon"
        raise ValueError(
            f"{label} name {name!r} ({source}) is reserved for platoon's own components: the 'platoon' owner "
            f"(platoon-*) and core's names {sorted(CORE_NAMES.get(kind, ()))}; prefix it with your own owner, "
            f"e.g. 'acme-{name.removeprefix(RESERVED_OWNER + '-')}'"
        )


def canonical_name(distribution: str) -> str:
    """A distribution name normalized as PEP 503 does (``Platoon_TextCraft`` -> ``platoon-textcraft``)."""

    return re.sub(r"[-_.]+", "-", distribution).lower()


def matching(ref: str, candidates: Iterable[T]) -> list[T]:
    """The candidates (with ``name`` and ``version``) that ``name[@version]`` selects: the name exactly, and the
    version exactly or as a dotted prefix (``2`` matches ``2.4.6``); every version when the reference gives none."""

    name, version = split_ref(ref)
    return [
        candidate
        for candidate in candidates
        if getattr(candidate, "name", None) == name
        and (
            version is None
            or (candidate_version := str(getattr(candidate, "version", ""))) == version
            or candidate_version.startswith(version + ".")
        )
    ]


def nearest(name: str, names: Iterable[str]) -> list[str]:
    """Up to three registered names closest to ``name``, for a did-you-mean."""

    return difflib.get_close_matches(name, sorted({split_ref(item)[0] for item in names}), n=3, cutoff=0.6)


# --- metadata ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Origin:
    """Where a component came from: its distribution and that distribution's version, the entry point that named
    it, and the module that defines it.  Every field is None when unknown (code that is not installed)."""

    distribution: str | None = None
    version: str | None = None
    entry_point: str | None = None
    module: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def label(self) -> str:
        """``distribution version``, or ``-`` for code that no installed distribution holds."""

        if self.distribution is None:
            return "-"
        return f"{self.distribution} {self.version}" if self.version else self.distribution


def distribution_name(distribution: Distribution | None) -> str | None:
    if distribution is None:
        return None
    try:
        return distribution.metadata["Name"] or distribution.name
    except Exception:
        return getattr(distribution, "name", None)


def entry_point_origin(entry_point: EntryPoint) -> Origin:
    """The origin of what ``entry_point`` names: its distribution (when it has metadata) and its module."""

    distribution = entry_point.dist
    try:
        module = entry_point.module
    except AttributeError:  # a value that is not module[:attr]
        module = None
    return Origin(
        distribution=distribution_name(distribution),
        version=None if distribution is None else distribution.version,
        entry_point=f"{entry_point.group}:{entry_point.name}",
        module=module,
    )


@functools.cache
def _all_entry_points() -> importlib.metadata.EntryPoints:
    return importlib.metadata.entry_points()  # every group in one pass over the installed metadata


@functools.cache
def _scan(group: str) -> tuple[EntryPoint, ...]:
    return tuple(_all_entry_points().select(group=group))


def installed_entry_points(kind: str) -> tuple[EntryPoint, ...]:
    """Every installed entry point of ``kind``'s group (metadata only; scanned once per process)."""

    return _scan(GROUPS[kind])


def entry_points_named(kind: str, name: str) -> list[EntryPoint]:
    """The installed entry points of ``kind``'s group named exactly ``name``.

    A name of core's (:func:`is_reserved`) only counts from the project's own distributions, so no extension can
    take it over.
    """

    found = [entry_point for entry_point in installed_entry_points(kind) if entry_point.name == name]
    if is_reserved(kind, name):
        found = [
            entry_point
            for entry_point in found
            if may_use_reserved_names(distribution=distribution_name(entry_point.dist))
        ]
    return found


def refresh() -> None:
    """Scan installed metadata again (after installing an extension in a running process) and forget recorded
    load failures, so that a failed entry point is tried again.  Components already loaded stay registered."""

    _all_entry_points.cache_clear()
    _scan.cache_clear()
    _distributions.cache_clear()
    _packages.cache_clear()
    with _lock:
        _failures.clear()


# --- compatibility ----------------------------------------------------------------------------------------------


@functools.cache
def platoon_version() -> str | None:
    """The version of the running platoon: the imported checkout's ``pyproject.toml`` when platoon runs from a
    source checkout (which may shadow an installed copy), else the installed metadata."""

    pyproject = _PLATOON_DIR.parent / "pyproject.toml"
    try:
        project = tomllib.loads(pyproject.read_text()).get("project", {})
    except (OSError, tomllib.TOMLDecodeError):
        project = {}
    if project.get("name") == RESERVED_OWNER and project.get("version"):
        return str(project["version"])
    try:
        return importlib.metadata.version(RESERVED_OWNER)
    except importlib.metadata.PackageNotFoundError:
        return None


def platoon_requirements(distribution: Distribution) -> list[str]:
    """The distribution's requirements on ``platoon`` that apply here: every one whose environment marker holds
    for this Python with no extra or with one of the extras the distribution provides."""

    from packaging.requirements import InvalidRequirement, Requirement

    extras = [""] + list(distribution.metadata.get_all("Provides-Extra") or [])
    found = []
    for line in distribution.requires or ():
        try:
            requirement = Requirement(line)
        except InvalidRequirement:
            continue
        if canonical_name(requirement.name) != RESERVED_OWNER:
            continue
        if requirement.marker is not None and not any(
            requirement.marker.evaluate({"extra": extra}) for extra in extras
        ):
            continue
        found.append(line)
    return found


def incompatibility(distribution: Distribution | None) -> str | None:
    """Why the running platoon cannot load ``distribution``'s components (a requirement on platoon whose range
    excludes it), or None when it can, when the distribution states no range, or when either version is unknown."""

    if distribution is None:
        return None
    running = platoon_version()
    if running is None:
        return None
    from packaging.requirements import Requirement
    from packaging.version import InvalidVersion, Version

    try:
        version = Version(running)
    except InvalidVersion:
        return None
    for line in platoon_requirements(distribution):
        specifier = Requirement(line).specifier
        if str(specifier) and not specifier.contains(version, prereleases=True):
            requirement = line.split(";")[0].strip()
            name = distribution_name(distribution)
            return f"{name} {distribution.version} requires {requirement}, but platoon {running} is running"
    return None


def suggested_requirement() -> str:
    """The requirement on platoon to suggest to an extension: from the running release up to the next release that
    may break it, which below 1.0 is the next minor release (``platoon>=0.2,<0.3`` on 0.2.0) and from 1.0 the next
    major one (``platoon>=1.1,<2`` on 1.1.0).  It always admits the running platoon, so following it never makes an
    extension incompatible with the platoon that suggested it."""

    from packaging.version import InvalidVersion, Version

    running = platoon_version()
    try:
        version = Version(running) if running is not None else None
    except InvalidVersion:
        version = None
    if version is None:
        return "platoon>=X.Y,<N"
    lower = str(version) if version.is_prerelease else f"{version.major}.{version.minor}"
    upper = f"0.{version.minor + 1}" if version.major == 0 else str(version.major + 1)
    return f"platoon>={lower},<{upper}"


def requirements_ignored() -> bool:
    return os.environ.get(IGNORE_REQUIREMENTS_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def check_compatible(entry_point: EntryPoint) -> None:
    """Refuse ``entry_point`` before it is imported when its distribution's platoon requirement excludes the
    running platoon (unless :data:`IGNORE_REQUIREMENTS_ENV` is set)."""

    why = incompatibility(entry_point.dist)
    if why is None:
        return
    if requirements_ignored():
        logger.warning("%s; loading it anyway, since %s is set", why, IGNORE_REQUIREMENTS_ENV)
        return
    raise ExtensionError(
        f"{why}: install a version of {distribution_name(entry_point.dist)} that supports this platoon, or set "
        f"{IGNORE_REQUIREMENTS_ENV}=1 to load it anyway"
    )


# --- loading ----------------------------------------------------------------------------------------------------

# accept(value, origin, entry point): register what an entry point loaded to, raising when it is not a component
# of the kind or not named as the entry point is.
Accept = Callable[[Any, Origin, EntryPoint], None]
# Kind -> the function (``module:attr``) that registers what one of its entry points loads to.
ACCEPTORS: dict[str, str] = {
    "harness": "platoon.harness.registry:accept_entry_point",
    "program": "platoon.programs:accept_entry_point",
    "environment": "platoon.envs.named:accept_entry_point",
    "reward_processor": "platoon.registry:accept_reward_processor",
}


def acceptor(kind: str) -> Accept:
    module, _, attr = ACCEPTORS[kind].partition(":")
    return getattr(importlib.import_module(module), attr)


_Key = tuple[str, str, str, str | None]
# Guards the bookkeeping below; never held while an extension imports.
_lock = threading.RLock()
# (group, name, value, distribution) -> the origin of a loaded entry point.
_loaded: dict[_Key, Origin] = {}
# The same key -> (the thread importing it, set when it is done), while it loads.
_loading: dict[_Key, tuple[int, threading.Event]] = {}
# group -> {"name = value": why it failed}
_failures: dict[str, dict[str, str]] = {}
# (kind, ref) -> the origin of a component an entry point registered.
_origins: dict[tuple[str, str], Origin] = {}


def _key(entry_point: EntryPoint) -> _Key:
    return (entry_point.group, entry_point.name, entry_point.value, distribution_name(entry_point.dist))


def entry_point_label(entry_point: EntryPoint) -> str:
    """``name = value``, and the distribution with its version when the entry point has metadata."""

    origin = entry_point_origin(entry_point)
    label = f"{entry_point.name} = {entry_point.value}"
    return label if origin.distribution is None else f"{label} ({origin.label()})"


def is_loaded(entry_point: EntryPoint) -> bool:
    with _lock:
        return _key(entry_point) in _loaded


def forget_loaded(kind: str) -> None:
    """Forget which entry points of ``kind``'s group have loaded, so that each registers again the next time it is
    named (their modules stay imported): for a registry that was rebuilt empty, such as the harness specs'."""

    group = GROUPS[kind]
    with _lock:
        for key in [key for key in _loaded if key[0] == group]:
            del _loaded[key]


def load_entry_point(kind: str, entry_point: EntryPoint, accept: Accept | None = None) -> Origin:
    """Load ``entry_point`` of ``kind``'s group once and register what it names (through the kind's
    :data:`ACCEPTORS` function, or ``accept``).

    In order: the entry point's name must follow the naming rule and may take a name of core's only from the
    project's own distributions; its distribution's platoon requirement must admit the running platoon; then it
    is imported and ``accept`` checks and registers the value.  A failure raises :class:`ExtensionError`, naming
    the entry point and its distribution, and is recorded (:func:`load_failures`); it touches no other entry
    point.  Loading it again returns the recorded origin, or raises the recorded failure again.

    No lock is held while the extension imports: lookups of other components go on meanwhile, and a thread that
    names the same entry point waits for the import to finish.
    """

    origin = entry_point_origin(entry_point)
    if accept is not None:  # into a registry of the caller's: no process bookkeeping beyond failures
        return _load(kind, entry_point, accept, origin)
    key = _key(entry_point)
    label = f"{entry_point.name} = {entry_point.value}"
    while True:
        with _lock:
            loaded = _loaded.get(key)
            if loaded is not None:
                return loaded
            failed = _failures.get(GROUPS[kind], {}).get(label)
            if failed is not None:
                raise ExtensionError(
                    f"{GROUPS[kind]} entry point {entry_point_label(entry_point)} failed to load: {failed}"
                )
            loading = _loading.get(key)
            if loading is None:
                done = threading.Event()
                _loading[key] = (threading.get_ident(), done)
                break
            if loading[0] == threading.get_ident():
                return origin  # this thread is importing it, and the import names it again
        loading[1].wait()  # another thread is importing it; then its origin or its failure is recorded
    try:
        _load(kind, entry_point, acceptor(kind), origin)
        with _lock:
            _loaded[key] = origin
    finally:
        with _lock:
            del _loading[key]
        done.set()
    return origin


def _load(kind: str, entry_point: EntryPoint, accept: Accept, origin: Origin) -> Origin:
    group = GROUPS[kind]
    try:
        check_registration(
            kind,
            entry_point.name,
            module=origin.module if origin.distribution is None else None,
            distribution=origin.distribution,
            what=f"{group} entry point",
        )
        check_compatible(entry_point)
        value = entry_point.load()
        accept(value, origin, entry_point)
    except Exception as exc:
        why = str(exc) if isinstance(exc, ExtensionError) else f"{type(exc).__name__}: {exc}"
        record_failure(kind, entry_point, why)
        logger.warning("%s entry point %s failed to load: %s", group, entry_point_label(entry_point), why)
        raise ExtensionError(f"{group} entry point {entry_point_label(entry_point)} failed to load: {why}") from exc
    return origin


def load_named(kind: str, name: str) -> list[Origin]:
    """Load every installed entry point of ``kind``'s group named exactly ``name`` (see :func:`entry_points_named`);
    the first failure raises after the others are tried."""

    origins: list[Origin] = []
    errors: list[ExtensionError] = []
    for entry_point in entry_points_named(kind, name):
        try:
            origins.append(load_entry_point(kind, entry_point))
        except ExtensionError as exc:
            errors.append(exc)
    if errors:
        raise errors[0]
    return origins


def load_failures(kind: str) -> dict[str, str]:
    """The entry points of ``kind``'s group that failed to load in this process: ``name = value`` -> why."""

    with _lock:
        return dict(_failures.get(GROUPS[kind], {}))


def record_failure(kind: str, entry_point: EntryPoint, why: str) -> None:
    with _lock:
        _failures.setdefault(GROUPS[kind], {})[f"{entry_point.name} = {entry_point.value}"] = why


def record_origin(kind: str, ref: str, origin: Origin) -> None:
    """Remember that the component ``kind`` ``ref`` came from ``origin`` (an entry point's distribution).  The
    first origin stays, unless it names no distribution and this one does (an installed copy of a checkout whose
    entry points a test registered first)."""

    with _lock:
        existing = _origins.get((kind, ref))
        if existing is None or (existing.distribution is None and origin.distribution is not None):
            _origins[(kind, ref)] = origin


def recorded_origin(kind: str, ref: str) -> Origin | None:
    with _lock:
        return _origins.get((kind, ref))


def installed_hint(kind: str) -> str:
    """The installed entry points of ``kind``'s group, for an unknown-name error (metadata only)."""

    entries = sorted(
        f"{entry_point.name} ({entry_point_origin(entry_point).label()})"
        for entry_point in installed_entry_points(kind)
    )
    return f"; installed {GROUPS[kind]} entry points: {entries}" if entries else ""


def unknown(kind: str, ref: str, registered: Iterable[str], *, extra: str = "") -> ExtensionError:
    """The error for a ``kind`` reference that resolves to nothing: what is registered, what is installed, the
    nearest names, what failed to load, and how to provide one."""

    registered = sorted(set(registered))
    name, _ = split_ref(ref)
    installed = [entry_point.name for entry_point in installed_entry_points(kind)]
    close = nearest(name, [*registered, *installed])
    label = kind.replace("_", " ")
    message = f"Unknown {label} {ref!r}. Available: {registered}{installed_hint(kind)}"
    if close:
        message += f"; did you mean {' or '.join(repr(item) for item in close)}?"
    message += (
        f". Install the distribution that provides it (an entry point named {name!r} in {GROUPS[kind]}), "
        f"or name it as module:attr{extra}"
    )
    failures = load_failures(kind)
    if failures:
        message += f". {GROUPS[kind]} entry points that failed to load: {failures}"
    return ExtensionError(message)


def ambiguous(kind: str, ref: str, refs: Iterable[str]) -> ExtensionError:
    label = kind.replace("_", " ")
    return ExtensionError(
        f"{label.capitalize()} reference {ref!r} is ambiguous: {sorted(set(refs))}; name one version, e.g. "
        f"'{sorted(set(refs))[0]}'"
    )


# --- where code comes from ----------------------------------------------------------------------------------------


@functools.cache
def _distributions() -> tuple[Distribution, ...]:
    return tuple(importlib.metadata.distributions())


def _direct_dir(distribution: Distribution) -> Path | None:
    """The local directory a distribution was installed from (PEP 610 ``direct_url.json``: editable and local
    installs), or None."""

    try:
        text = distribution.read_text("direct_url.json")
    except Exception:
        return None
    if not text:
        return None
    try:
        url = json.loads(text).get("url", "")
    except (ValueError, AttributeError):
        return None
    if not isinstance(url, str) or not url.startswith("file://"):
        return None
    from urllib.parse import unquote, urlsplit

    return Path(unquote(urlsplit(url).path)).resolve()


def direct_url(distribution: Distribution) -> dict[str, Any] | None:
    """The distribution's PEP 610 ``direct_url.json`` (how an editable or local install was made), or None."""

    try:
        text = distribution.read_text("direct_url.json")
        return json.loads(text) if text else None
    except Exception:
        return None


def _module_root(module: str) -> str:
    """The package that identifies a module's distribution: its top-level package; below a namespace package
    (PEP 420, which several distributions share, such as an organization's ``acme.envs`` and ``acme.agents``), the
    first package that is not one; and inside ``platoon``, which is core's alone, its first two levels (the package of
    a module that another distribution placed there, which platoon never imports)."""

    parts = module.split(".")
    if parts[0] == RESERVED_OWNER:
        return ".".join(parts[:2])
    depth = 1
    while depth < len(parts) and _is_namespace_package(".".join(parts[:depth])):
        depth += 1
    return ".".join(parts[:depth])


def _is_namespace_package(name: str) -> bool:
    """Whether the imported module ``name`` is a namespace package (a package with no ``__init__`` file)."""

    module = sys.modules.get(name)
    return getattr(module, "__path__", None) is not None and getattr(module, "__file__", None) is None


def is_core_module(module: str) -> bool:
    """Whether ``module`` is part of core: in the ``platoon`` package and inside core's own directory (not a
    ``platoon.<name>`` package that another distribution placed elsewhere on the path).  Imports nothing."""

    if module == RESERVED_OWNER:
        return True
    if not module.startswith(RESERVED_OWNER + "."):
        return False
    path = getattr(sys.modules.get(module), "__file__", None)
    if path is not None:
        return Path(path).resolve().is_relative_to(_PLATOON_DIR)
    relative = Path(*module.split(".")[1:])
    return (_PLATOON_DIR / relative).with_suffix(".py").is_file() or (_PLATOON_DIR / relative / "__init__.py").is_file()


@functools.cache
def _packages() -> Mapping[str, list[str]]:
    return importlib.metadata.packages_distributions()


def origin_of_module(module: str | None) -> Origin:
    """The distribution that holds ``module``, best effort and from metadata: core for core's modules; the
    distribution of a loaded entry point from the same package (below any namespace package, which several
    distributions share); the one that installs the package; or the editable install whose directory holds the
    module's file.  None for code that is not installed."""

    if not module:
        return Origin()
    if is_core_module(module):
        return Origin(distribution=RESERVED_OWNER, version=platoon_version(), module=module)
    root = _module_root(module)
    with _lock:
        loaded = [origin for origin in _loaded.values() if origin.module]
    for origin in loaded:
        if origin.distribution and _module_root(origin.module or "") == root:
            return dataclasses.replace(origin, entry_point=None, module=module)
    # The installer's record of top-level names; it cannot tell apart the packages below a namespace package.
    by_package = _packages().get(root, []) if "." not in root else []
    if len(set(by_package)) == 1 and root != RESERVED_OWNER:
        name = by_package[0]
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        return Origin(distribution=name, version=version, module=module)
    path = getattr(sys.modules.get(module), "__file__", None)
    if path is not None:
        target = Path(path).resolve()
        best: tuple[int, Distribution] | None = None
        for distribution in _distributions():
            directory = _direct_dir(distribution)
            if directory is not None and target.is_relative_to(directory):
                depth = len(directory.parts)
                if best is None or depth > best[0]:
                    best = (depth, distribution)
        if best is not None:
            return Origin(distribution=distribution_name(best[1]), version=best[1].version, module=module)
    return Origin(module=module)


def origin_of(kind: str, ref: str, obj: Any) -> Origin:
    """The origin of the component ``kind`` ``ref``: the entry point it was loaded through, else the distribution
    of the module that defines ``obj``."""

    recorded = recorded_origin(kind, ref)
    if recorded is not None:
        return recorded
    return origin_of_module(getattr(obj, "__module__", None))


# --- listing and checking -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentInfo:
    """One component as metadata describes it, without importing it."""

    kind: str
    name: str
    # The distribution that provides it and that distribution's version (core's for a built-in).
    distribution: str | None
    version: str | None
    # ``module:attr`` the entry point names, or where a built-in lives.
    value: str
    # ``ok``; ``incompatible: <why>``; ``ignored: <why>`` for an entry point the lookup never uses.
    status: str = "ok"
    builtin: bool = False
    # The distribution's requirements on platoon, as its metadata states them.
    requires: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), "requires": list(self.requires)}


def builtin_components(kind: str | None = None) -> list[ComponentInfo]:
    """Core's built-in components: the harnesses, programs, named environments and reward processors the
    ``platoon`` distribution registers itself (imports core's built-in modules, never an extension)."""

    version = platoon_version()
    found: list[ComponentInfo] = []
    if kind in (None, "harness"):
        from platoon.harness.registry import builtin_harnesses

        found += [
            ComponentInfo("harness", ref, RESERVED_OWNER, version, source, builtin=True)
            for ref, source in builtin_harnesses()
        ]
    if kind in (None, "program"):
        from platoon.programs import builtin_programs

        found += [
            ComponentInfo("program", record.ref, RESERVED_OWNER, version, record.source, builtin=True)
            for record in builtin_programs()
        ]
    if kind in (None, "environment"):
        from platoon.envs.named import builtin_environments

        found += [
            ComponentInfo("environment", ref, RESERVED_OWNER, version, source, builtin=True)
            for ref, source in builtin_environments()
        ]
    if kind in (None, "reward_processor"):
        from platoon.registry import builtin_reward_processors

        found += [
            ComponentInfo("reward_processor", name, RESERVED_OWNER, version, source, builtin=True)
            for name, source in builtin_reward_processors()
        ]
    return found


def builtin_names(kind: str) -> frozenset[str]:
    """The names of core's built-in components of ``kind`` that resolve before any entry point of the same name
    (named environments and reward processors), so that an installed entry point with one of these names is never
    loaded."""

    if kind == "environment":
        from platoon.envs.named import BUILTIN_ENVIRONMENTS

        return frozenset(BUILTIN_ENVIRONMENTS)
    if kind == "reward_processor":
        from platoon.registry import BUILTIN_REWARD_PROCESSORS

        return frozenset(BUILTIN_REWARD_PROCESSORS)
    return frozenset()


def list_components(kind: str | None = None) -> list[ComponentInfo]:
    """Every component the installation provides, built-ins first, then the extensions' entry points from their
    metadata (nothing of an extension is imported), each with its distribution, version and status."""

    kinds = KINDS if kind is None else (kind,)
    rows = [row for each in kinds for row in builtin_components(each)]
    for each in kinds:
        builtins = builtin_names(each)
        for entry_point in installed_entry_points(each):
            origin = entry_point_origin(entry_point)
            requires: tuple[str, ...] = ()
            status = "ok"
            try:
                check_registration(
                    each, entry_point.name, distribution=origin.distribution, what=f"{GROUPS[each]} entry point"
                )
            except ValueError as exc:
                status = f"ignored: {exc}"
            if status == "ok" and entry_point.name in builtins:
                status = f"ignored: {entry_point.name} is built into platoon, which uses its own"
            if entry_point.dist is not None:
                requires = tuple(platoon_requirements(entry_point.dist))
                why = incompatibility(entry_point.dist)
                if why is not None and status == "ok":
                    status = f"incompatible: {why}"
            rows.append(
                ComponentInfo(
                    kind=each,
                    name=entry_point.name,
                    distribution=origin.distribution,
                    version=origin.version,
                    value=entry_point.value,
                    status=status,
                    requires=requires,
                )
            )
    return rows


# Kind -> where configs and code use a component of that kind.
USED_IN: dict[str, tuple[str, ...]] = {
    "harness": ("environments[].harness", "ep.run(harness=...)", "ctx.agents.spawn(harness=...)"),
    "program": ("environments[].program",),
    "environment": ("environments[].env",),
    "reward_processor": ("environments[].reward_processor",),
}


@dataclass(frozen=True)
class ComponentDetails:
    """One component as ``platoon extensions show`` describes it: its row of :func:`list_components`, the entry
    point that provides it, where configs and code use it, and what it registers."""

    info: ComponentInfo
    # ``group:name`` of the entry point that provides it; None for a built-in.
    entry_point: str | None
    # Where configs and code use a component of this kind (:data:`USED_IN`).
    used_in: tuple[str, ...]
    # ``name@version`` of each component it registers (a reward processor's has no version): a built-in's own, an
    # extension's once its entry point is loaded; None while it is not.
    registers: tuple[str, ...] | None
    # Why its entry point failed to load, when it was loaded and failed.
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The fields of :meth:`ComponentInfo.to_dict`, then this class's own."""

        registers = None if self.registers is None else list(self.registers)
        extra = {"entry_point": self.entry_point, "used_in": list(self.used_in), "registers": registers}
        return {**self.info.to_dict(), **extra, "error": self.error}


def show_components(ref: str, kind: str | None = None, *, load: bool = False) -> list[ComponentDetails]:
    """The installed components named ``ref`` (``name[@version]``), each as :func:`list_components` lists it,
    with its entry point and where configs use it; a name that several kinds use gives one of each.

    Reads metadata only, so an extension's versions are unknown.  With ``load``, imports each extension's entry
    point as a config naming it would (after its compatibility check), and records what it registers, or why it
    failed; a version in ``ref`` then selects what it registers.  Raises :class:`ExtensionError` when nothing
    installed has the name, or none of it has the version.
    """

    name, version = split_ref(ref)
    if ":" in ref or not name:
        raise ExtensionError(
            f"{ref!r} is not a component name: name an installed component as name[@version], as `platoon extensions "
            "list` shows it"
        )
    rows = list_components(kind)
    found: list[ComponentDetails] = []
    versions: list[str] = []  # what the name's components register, when none has the version asked for
    for info in rows:
        if info.builtin:
            if split_ref(info.name)[0] == name:
                if _has_version(ref, info.name):
                    found.append(ComponentDetails(info, None, USED_IN[info.kind], (info.name,)))
                else:
                    versions.append(info.name)
            continue
        if info.name != name:
            continue
        details = ComponentDetails(info, f"{GROUPS[info.kind]}:{info.name}", USED_IN[info.kind], None)
        if load:
            details = _load_details(details)
            if version is not None and details.error is None:
                selected = tuple(item for item in details.registers or () if _has_version(ref, item))
                if not selected:
                    versions += details.registers or ()
                    continue
                details = dataclasses.replace(details, registers=selected)
        found.append(details)
    if found:
        return found
    label = "component" if kind is None else kind.replace("_", " ")
    if versions:
        raise ExtensionError(f"No {label} {ref!r} is installed; {name!r} has {sorted(set(versions))}")
    close = nearest(name, [info.name for info in rows])
    hint = f"; did you mean {' or '.join(repr(item) for item in close)}?" if close else ""
    raise ExtensionError(f"No {label} named {name!r} is installed{hint} (`platoon extensions list` lists them)")


def _has_version(ref: str, registered: str) -> bool:
    """Whether the registered ``name[@version]`` has the version ``ref`` asks for (any, when ``ref`` names none)."""

    return bool(matching(ref, [_Named(*split_ref(registered))]))


@dataclass(frozen=True)
class _Named:
    name: str
    version: str | None


def _load_details(details: ComponentDetails) -> ComponentDetails:
    """``details`` after loading its entry point: what it registers, or why it failed."""

    info = details.info
    entry_points = [
        entry_point
        for entry_point in installed_entry_points(info.kind)
        if entry_point.name == info.name
        and entry_point.value == info.value
        and distribution_name(entry_point.dist) == info.distribution
    ]
    if not entry_points:
        return dataclasses.replace(details, error="its entry point is no longer installed")
    try:
        origin = load_entry_point(info.kind, entry_points[0])
    except ExtensionError as exc:
        return dataclasses.replace(details, error=str(exc))
    with _lock:
        registers = sorted(
            ref
            for (kind, ref), recorded in _origins.items()
            if kind == info.kind
            and recorded.entry_point == origin.entry_point
            and recorded.distribution == origin.distribution
        )
    return dataclasses.replace(details, registers=tuple(registers))


@dataclass(frozen=True)
class Problem:
    """One finding of :func:`check_extensions`: an ``error`` stops the component from loading, a ``warning`` does
    not."""

    severity: Literal["error", "warning"]
    distribution: str | None
    message: str

    def __str__(self) -> str:
        where = self.distribution or "(no distribution)"
        return f"{self.severity}: {where}: {self.message}"


def check_extensions(distributions: Sequence[str] = (), *, load: bool = False) -> list[Problem]:
    """Check the installed extensions (or only ``distributions``) from their metadata: component names, one owner
    per distribution, the platoon requirement (present, bounded above, admitting the running platoon), entries in
    retired groups; with ``load``, also import each entry point and check what it registers."""

    wanted = {canonical_name(name) for name in distributions}
    problems: list[Problem] = []
    owners: dict[str, set[str]] = {}
    seen: dict[str, Distribution] = {}

    def selected(distribution: str | None) -> bool:
        return not wanted or (distribution is not None and canonical_name(distribution) in wanted)

    for kind in KINDS:
        builtins = builtin_names(kind)
        for entry_point in installed_entry_points(kind):
            origin = entry_point_origin(entry_point)
            if not selected(origin.distribution):
                continue
            if origin.distribution is not None and entry_point.dist is not None:
                seen.setdefault(origin.distribution, entry_point.dist)
            label = f"{GROUPS[kind]} entry point {entry_point.name} = {entry_point.value}"
            try:
                check_registration(kind, entry_point.name, distribution=origin.distribution)
            except ValueError as exc:
                problems.append(Problem("error", origin.distribution, f"{label}: {exc}"))
                continue
            if entry_point.name in builtins:
                problems.append(
                    Problem(
                        "warning",
                        origin.distribution,
                        f"{label}: {entry_point.name} is built into platoon, which never loads this entry point; "
                        "uninstall the distribution (it is a leftover of an older layout)",
                    )
                )
                continue
            if origin.distribution is not None and not may_use_reserved_names(distribution=origin.distribution):
                owners.setdefault(origin.distribution, set()).add(owner(entry_point.name))
            if load:
                problems += _check_load(kind, entry_point, label, origin)
    for distribution, names in sorted(owners.items()):
        if len(names) > 1:
            problems.append(
                Problem(
                    "warning",
                    distribution,
                    f"components with more than one owner {sorted(names)}: name every component of one "
                    "distribution <owner>-<name> with the same owner",
                )
            )
    for distribution_label, distribution in sorted(seen.items()):
        requirements = platoon_requirements(distribution)
        why = incompatibility(distribution)
        if why is not None:
            problems.append(Problem("error", distribution_label, f"incompatible: {why}"))
        elif not requirements:
            problems.append(
                Problem(
                    "warning",
                    distribution_label,
                    "declares no requirement on platoon; add one with an upper bound, e.g. "
                    f"'{suggested_requirement()}', so that platoon can refuse a version it does not support",
                )
            )
        elif not any(_has_upper_bound(line) for line in requirements):
            problems.append(
                Problem(
                    "warning",
                    distribution_label,
                    f"its platoon requirement {requirements} has no upper bound; bound it below the next release "
                    f"that may break it (the next minor release below 1.0), e.g. '{suggested_requirement()}'",
                )
            )
    for group, replacement in RETIRED_GROUPS.items():
        for entry_point in _scan(group):
            distribution = distribution_name(entry_point.dist)
            if selected(distribution):
                problems.append(
                    Problem(
                        "warning",
                        distribution,
                        f"{group} entry point {entry_point.name} = {entry_point.value}: platoon no longer reads "
                        f"{group}; {replacement}",
                    )
                )
    return problems


def _has_upper_bound(requirement: str) -> bool:
    from packaging.requirements import Requirement

    return any(spec.operator in ("<", "<=", "==", "~=", "===") for spec in Requirement(requirement).specifier)


def _check_load(kind: str, entry_point: EntryPoint, label: str, origin: Origin) -> list[Problem]:
    """Load one entry point as a config naming it would, reporting what goes wrong."""

    try:
        load_entry_point(kind, entry_point)
    except ExtensionError as exc:
        return [Problem("error", origin.distribution, str(exc))]
    return []


# --- provenance --------------------------------------------------------------------------------------------------

# The environment fields that name a component, and the kind each resolves as.
_COMPONENT_FIELDS: tuple[tuple[str, str], ...] = (
    ("rollout", "rollout"),
    ("eval_rollout", "rollout"),
    ("dataset_loader", "dataset_loader"),
    ("eval_dataset_loader", "dataset_loader"),
    ("task_loader", "task_loader"),
    ("reward_processor", "reward_processor"),
)


def run_components(config: Any, *, split: str | None = None) -> list[dict[str, Any]]:
    """Resolve every component the config's ``environments`` name, check their options, and describe each for
    ``lock.json``.

    Imports the environments' ``package`` modules first, as a run does.  For each named environment, harness,
    program, reward processor, service class and import-string component: the kind, the reference as the config
    gives it, what it resolved to (``name@version``, or ``module:attr``), the distribution and version that
    provide it, the entry point that named it, and the environments that use it.  A harness with an option schema
    (a TOML spec's ``[options]``) also records the options each entry runs it with, defaults included, per phase
    (``options``: ``{entry: {train: ..., eval: ...}}``, :mod:`platoon.options`): the phases the config runs
    (:func:`platoon.options.config_phases`), or with ``split`` that split's phase only, for a command that runs one
    split whatever the config says (``platoon rollout``).  A reference that does not resolve, and options its
    component does not take, raise :class:`~platoon.utils.config.ConfigError` with every problem, which the CLI
    reports with exit 78 before the run starts.
    """

    from platoon.options import config_phases
    from platoon.train.auto import AutoEnvironment, import_environment_packages
    from platoon.utils.config import ConfigError

    try:
        environments = AutoEnvironment.environments(config)
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:  # none at all: a run would have nothing to run
        raise ConfigError(str(exc)) from None
    records: dict[tuple[str, str], dict[str, Any]] = {}
    problems: list[str] = []
    # The packages first, as a run imports them: they register components the entries name.  One that does not
    # import is its entry's problem, and that entry's components wait for it instead of being reported unknown.
    unimported: set[str] = set()
    for environment in environments:
        try:
            import_environment_packages([environment.package] if environment.package else [])
        except Exception as exc:
            problems.append(
                f"environments[{environment.label}].package: {environment.package!r} does not import "
                f"({type(exc).__name__}: {exc})"
            )
            unimported.add(environment.label)

    def add(kind: str, ref: str, label: str, field: str, describe: Callable[[str], dict[str, Any]]) -> None:
        key = (kind, ref)
        if key not in records:
            try:
                records[key] = {"kind": kind, "ref": ref, **describe(ref), "environments": []}
            except Exception as exc:
                problems.append(f"environments[{label}].{field}: {exc}")
                return
        records[key]["environments"].append(label)

    for environment in environments:
        label = environment.label
        if label in unimported:
            continue
        if environment.env:
            add("environment", environment.env, label, "env", _describe_environment)
        for ref in environment.harness_refs():
            add("harness", ref, label, "harness", _describe_harness)
        if environment.harness is not None:
            add("program", environment.program, label, "program", _describe_program)
        if environment.service is not None:
            add("service", environment.service, label, "service", _describe_service)
        for field, kind in _COMPONENT_FIELDS:
            ref = getattr(environment, field)
            if ref:
                add(kind, ref, label, field, functools.partial(_describe_registry_component, kind))
        if environment.workflow != "group_rollout":
            describe = functools.partial(_describe_registry_component, "workflow")
            add("workflow", environment.workflow, label, "workflow", describe)
    problems += _check_options(environments, records, phases=config_phases(config, split))
    problems += _check_service_options(environments, records)
    if getattr(config, "train", None) is not None:
        problems += _check_entry_settings_workflow(environments, records)
    if problems:
        raise ConfigError(problems)
    return list(records.values())


def _check_entry_settings_workflow(
    environments: Sequence[Any], records: Mapping[tuple[str, str], dict[str, Any]]
) -> list[str]:
    """An entry's own training settings beside a training workflow class that does not apply them
    (:func:`platoon.train.entry_settings.workflow_problems`), which ``platoon train`` refuses too.  The default
    workflow applies them; one that does not resolve is reported already."""

    from platoon.registry import resolve_component
    from platoon.train import entry_settings
    from platoon.train.components import merge_run_level_settings

    workflow = merge_run_level_settings(environments).workflow
    if workflow == "group_rollout" or ("workflow", workflow) not in records:
        return []
    return entry_settings.workflow_problems(environments, resolve_component("workflow", workflow), workflow)


def _check_service_options(environments: Sequence[Any], records: Mapping[tuple[str, str], dict[str, Any]]) -> list[str]:
    """Bind each entry's ``service_kwargs`` to the factory its service builds from, without building it
    (:func:`platoon.envs.service.service_option_problems`), and check its ``dataset_kwargs`` and
    ``eval_dataset_kwargs`` against what a service's task listing takes, where it lists the entry's tasks (no dataset
    loader of its own).  Returns the problems, one line each, with dotted paths."""

    from platoon.envs.service import listing_option_problems, service_option_problems

    problems: list[str] = []
    for index, environment in enumerate(environments):
        if environment.service is None or ("service", environment.service) not in records:
            continue  # no service, or one that does not resolve, which is reported already
        where = f"environments.{index}"
        problems += service_option_problems(
            environment.service, environment.service_kwargs, where=f"{where}.service_kwargs"
        )
        if environment.dataset_loader is None:
            problems += listing_option_problems(environment.dataset_kwargs, where=f"{where}.dataset_kwargs")
        if (environment.eval_dataset_loader or environment.dataset_loader) is None:
            problems += listing_option_problems(environment.eval_dataset_kwargs, where=f"{where}.eval_dataset_kwargs")
    return problems


def _check_options(
    environments: Sequence[Any], records: Mapping[tuple[str, str], dict[str, Any]], *, phases: Sequence[Any]
) -> list[str]:
    """Check each entry's options against the schema of each resolved harness it runs (``platoon.options``), and
    record in the harness's lock record what each entry's rollouts of each phase run with, defaults included
    (``options``: ``{entry label: {phase: options}}``).  Returns the problems, one line each, with dotted paths."""

    from platoon.harness.registry import get_harness
    from platoon.options import check_harness_options, lock_record

    problems: list[str] = []
    for index, environment in enumerate(environments):
        for ref in environment.harness_refs():
            record = records.get(("harness", ref))
            if record is None:  # it did not resolve, which is reported already
                continue
            checked = check_harness_options(environment, get_harness(ref), where=f"environments.{index}", phases=phases)
            problems += [problem for problem in checked.problems if problem not in problems]
            if checked.values:
                record.setdefault("options", {})[environment.label] = lock_record(checked.values)
    return problems


def _record(resolved: str | None, origin: Origin, *, source: str | None = None) -> dict[str, Any]:
    return {
        "resolved": resolved,
        "source": source,
        "distribution": origin.distribution,
        "distribution_version": origin.version,
        "entry_point": origin.entry_point,
    }


def _describe_environment(ref: str) -> dict[str, Any]:
    from platoon.envs.named import environment_source, get_environment

    entry = get_environment(ref)
    return _record(entry.ref, origin_of("environment", entry.ref, None), source=environment_source(entry.ref))


def _describe_harness(ref: str) -> dict[str, Any]:
    from platoon.harness.registry import get_harness, harness_origin

    harness = get_harness(ref)
    return _record(harness.ref, harness_origin(harness), source=harness.source)


def _describe_program(ref: str) -> dict[str, Any]:
    from platoon.programs import resolve_program

    program = resolve_program(ref)
    return _record(program.ref, origin_of("program", program.ref, program.fn), source=program.source)


def _describe_service(ref: str) -> dict[str, Any]:
    from platoon.envs.service import service_target
    from platoon.registry import import_from_string

    target = service_target(ref)
    if target is None:  # a remote ORS deployment: the env servers' own versions are theirs to report
        return _record(ref, Origin(distribution=RESERVED_OWNER, version=platoon_version()), source=None)
    if ref.startswith("python://"):  # an ORS environment: the SDK first, set up as the service sets it up
        from platoon.envs.ors.sdk import import_sdk

        import_sdk()
    value = import_from_string(target)
    source = f"{getattr(value, '__module__', '?')}:{getattr(value, '__qualname__', '?')}"
    return _record(source, origin_of_module(getattr(value, "__module__", None)), source=source)


def _describe_registry_component(kind: str, ref: str) -> dict[str, Any]:
    from platoon.registry import resolve_component

    value = resolve_component(kind, ref)
    source = f"{getattr(value, '__module__', '?')}:{getattr(value, '__qualname__', '?')}"
    recorded = recorded_origin(kind, ref)
    if recorded is not None:  # an extension's component, by name
        return _record(ref, recorded, source=source)
    return _record(source, origin_of_module(getattr(value, "__module__", None)), source=source)


def extension_records(components: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """The distributions other than platoon behind ``components``, each with its version, its requirements on
    platoon and how it was installed (PEP 610 ``direct_url``, which shows editable checkouts), for ``lock.json``."""

    found: dict[str, dict[str, Any]] = {}
    for component in components:
        name = component.get("distribution")
        if not name or canonical_name(name) == RESERVED_OWNER or name in found:
            continue
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = {"version": component.get("distribution_version"), "requires": [], "direct_url": None}
            continue
        found[name] = {
            "version": distribution.version,
            "requires": platoon_requirements(distribution),
            "direct_url": direct_url(distribution),
        }
    return found


__all__ = [
    "CORE_NAMES",
    "ComponentDetails",
    "ComponentInfo",
    "ExtensionError",
    "GROUPS",
    "IGNORE_REQUIREMENTS_ENV",
    "KINDS",
    "NAME_PATTERN",
    "Origin",
    "Problem",
    "RESERVED_OWNER",
    "RETIRED_GROUPS",
    "USED_IN",
    "check_compatible",
    "check_extensions",
    "check_name",
    "check_registration",
    "check_version",
    "entry_points_named",
    "extension_records",
    "forget_loaded",
    "incompatibility",
    "installed_entry_points",
    "is_reserved",
    "list_components",
    "load_entry_point",
    "load_named",
    "matching",
    "origin_of",
    "origin_of_module",
    "owner",
    "platoon_requirements",
    "platoon_version",
    "refresh",
    "run_components",
    "show_components",
    "split_ref",
]
