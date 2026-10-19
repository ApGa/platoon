"""Where harnesses come from: TOML specs and ``@harness`` functions, looked up by ``name[@version]``.

* **Built-ins** (core): the TOML specs in ``platoon/harness/builtin/specs/`` and the function harnesses that the
  modules in :data:`BUILTIN_FUNCTION_MODULES` register, imported on the first lookup.  Their names are reserved
  for core (:func:`platoon.extensions.is_reserved`), so no extension can take one over.
* **Extensions**: the installed ``platoon.harnesses`` entry point named exactly as the harness, imported the first
  time a config or code names that harness (:mod:`platoon.extensions`, which checks its distribution's platoon
  requirement first).  It loads to a decorated function, a ``FunctionHarness``, a ``HarnessSpec``, a spec file or
  directory of ``*.toml`` files (a path), a module whose import registers ``@harness`` functions, or a list of
  those.  Nothing is imported to list them.
* **Local code**: harnesses that a ``package:`` module registered, ``module:function`` references and ``.toml``
  spec paths.

References are ``name@version``; the version may be omitted when exactly one is registered, or shortened to a
dotted prefix (``mini-swe-agent@2`` matches ``2.4.6``).  A reference that matches several versions is ambiguous,
whether they are TOML specs, function harnesses or both; when a TOML spec and a function harness share one
``name@version`` the spec wins, with a warning.

An entry point that fails to load is recorded with why (:func:`harness_load_failures`) and leaves every other
harness available.  An entry point named after its module instead of its harness (``my-harnesses =
"my_harnesses.one_shot"`` registering ``my-one-shot``, the form from before names were exact) is still found: a
name that matches nothing loads every harness entry point not loaded yet, and a warning names the ones to rename.
"""

from __future__ import annotations

import functools
import importlib
import logging
import os
import types
from collections.abc import Iterable
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

from platoon import extensions
from platoon.extensions import ExtensionError, Origin
from platoon.harness.function import FunctionHarness, function_harnesses, harness_of, register_function_harness
from platoon.harness.spec import HarnessSpec, load_spec

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = extensions.GROUPS["harness"]
BUILTIN_SPECS_DIR = Path(__file__).resolve().parent.parent / "builtin" / "specs"
# Modules whose import registers the built-in function harnesses; a module that does not exist is skipped.
BUILTIN_FUNCTION_MODULES = (
    "platoon.harness.builtin.codeact",
    "platoon.harness.builtin.react",
    "platoon.harness.builtin.oracle",
)

Harness = HarnessSpec | FunctionHarness


def _is_builtin_spec(spec: HarnessSpec) -> bool:
    source = spec.source
    if not source:
        return False
    return Path(source).resolve().is_relative_to(BUILTIN_SPECS_DIR)


class HarnessRegistry:
    """Specs by ``name@version``, where each came from, and the entry points that failed to load into it."""

    def __init__(self) -> None:
        self._specs: dict[str, HarnessSpec] = {}
        self._origins: dict[str, Origin] = {}
        # ``name = value`` of each entry point that failed to load or register, with why; unknown-harness errors
        # name them, because the missing harness may be one of theirs.
        self.entry_point_failures: dict[str, str] = {}

    def add(self, spec: HarnessSpec, *, origin: Origin | None = None) -> None:
        """Register ``spec``; a name of core's is refused unless the spec is core's or a project distribution's."""

        if not _is_builtin_spec(spec):
            extensions.check_registration(
                "harness",
                spec.name,
                distribution=None if origin is None else origin.distribution,
                what=f"harness spec {spec.source}",
            )
        existing = self._specs.get(spec.ref)
        if existing is not None:
            raise ValueError(f"harness {spec.ref} is defined twice: {existing.source} and {spec.source}")
        self._specs[spec.ref] = spec
        if origin is not None:
            self._origins[spec.ref] = origin

    def add_path(self, path: str | os.PathLike[str], *, origin: Origin | None = None) -> list[str]:
        """Add one spec file, or every ``*.toml`` file in a directory; returns the references added."""

        path = Path(path)
        refs = []
        for spec_path in sorted(path.glob("*.toml")) if path.is_dir() else [path]:
            spec = load_spec(spec_path)
            self.add(spec, origin=origin)
            refs.append(spec.ref)
        return refs

    def add_entry_point_value(self, value: Any, *, origin: Origin | None = None) -> list[str]:
        """Register what a ``platoon.harnesses`` entry point loaded to; returns the references it holds."""

        if isinstance(value, HarnessSpec):
            self.add(value, origin=origin)
            return [value.ref]
        if isinstance(value, types.ModuleType):  # importing the module registered its @harness functions
            return [
                record.ref
                for record in function_harnesses()
                if record.fn.__module__ == value.__name__ or record.fn.__module__.startswith(value.__name__ + ".")
            ]
        if isinstance(value, FunctionHarness):
            extensions.check_registration(
                "harness", value.name, module=value.fn.__module__, distribution=_distribution(origin)
            )
            register_function_harness(value)
            return [value.ref]
        record = harness_of(value)
        if record is not None:  # a decorated function, registered by its decorator
            return [record.ref]
        if isinstance(value, (str, os.PathLike)):
            return self.add_path(value, origin=origin)
        if isinstance(value, Iterable):
            return [ref for item in value for ref in self.add_entry_point_value(item, origin=origin)]
        raise TypeError(
            f"a {ENTRY_POINT_GROUP} entry point must name a spec path or HarnessSpec, a module of @harness "
            f"functions or one of them, got {value!r}"
        )

    def add_entry_point(self, entry_point: EntryPoint) -> None:
        """Load one ``platoon.harnesses`` entry point and register what it names.

        An entry point that fails (its distribution's platoon requirement excludes this platoon, its package does
        not import, or it names something that is not a harness) is logged and recorded in
        :attr:`entry_point_failures` instead of raising, so one broken package does not take down every other
        harness.
        """

        try:
            extensions.load_entry_point(
                "harness", entry_point, accept=lambda value, origin, point: _accept(self, value, origin, point)
            )
        except ExtensionError:
            key = f"{entry_point.name} = {entry_point.value}"
            self.entry_point_failures[key] = extensions.load_failures("harness").get(key, "failed to load")

    def origin(self, ref: str) -> Origin | None:
        """Where the spec ``ref`` came from, when an entry point registered it."""

        return self._origins.get(ref)

    def get(self, ref: str) -> HarnessSpec:
        matches = extensions.matching(ref, self._specs.values())
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(f"Unknown harness {ref!r}. Available: {self.refs()}{_failures(self.entry_point_failures)}")
        raise ValueError(f"Harness reference {ref!r} is ambiguous: {sorted(spec.ref for spec in matches)}")

    def find(self, ref: str) -> list[HarnessSpec]:
        """Every spec ``ref`` matches."""

        return extensions.matching(ref, self._specs.values())

    def refs(self) -> list[str]:
        return sorted(self._specs)


def _distribution(origin: Origin | None) -> str | None:
    return None if origin is None else origin.distribution


def _failures(entry_point_failures: dict[str, str]) -> str:
    if not entry_point_failures:
        return ""
    return f". {ENTRY_POINT_GROUP} entry points that failed to load: {entry_point_failures}"


def _accept(registry: HarnessRegistry, value: Any, origin: Origin, entry_point: EntryPoint) -> None:
    refs = registry.add_entry_point_value(value, origin=origin)
    if not refs:
        raise ValueError("it registers no harness")
    for ref in refs:
        extensions.record_origin("harness", ref, origin)
    if not any(ref.partition("@")[0] == entry_point.name for ref in refs):
        logger.warning(
            "%s entry point %s registers %s, none named %r: name each entry point after the harness it registers "
            '(e.g. %s = "module:function"), so that platoon imports only the harnesses a config names; until then '
            "it is found only by loading every harness entry point",
            ENTRY_POINT_GROUP,
            extensions.entry_point_label(entry_point),
            sorted(refs),
            entry_point.name,
            refs[0].partition("@")[0],
        )


def accept_entry_point(value: Any, origin: Origin, entry_point: EntryPoint) -> None:
    """Register what a ``platoon.harnesses`` entry point loads to into the default registry."""

    _accept(default_registry(), value, origin, entry_point)


@functools.cache
def default_registry() -> HarnessRegistry:
    """The process's spec registry: the built-in specs, and the specs extensions' entry points add as they load.

    A new registry (after ``default_registry.cache_clear()``) holds no extension's specs, so the harness entry
    points load into it again when next named.
    """

    registry = HarnessRegistry()
    registry.add_path(BUILTIN_SPECS_DIR)
    extensions.forget_loaded("harness")
    return registry


@functools.cache
def _import_builtin_function_harnesses() -> dict[str, str]:
    """Import :data:`BUILTIN_FUNCTION_MODULES` once; returns the modules that failed to import, with why."""

    failures: dict[str, str] = {}
    for module in BUILTIN_FUNCTION_MODULES:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name is not None and (exc.name == module or module.startswith(exc.name + ".")):
                continue  # not shipped (yet)
            failures[module] = f"{type(exc).__name__}: {exc}"
            logger.warning("built-in harness module %s failed to import: %s", module, exc)
        except Exception as exc:
            failures[module] = f"{type(exc).__name__}: {exc}"
            logger.warning("built-in harness module %s failed to import", module, exc_info=True)
    return failures


def _load_unloaded_entry_points() -> None:
    """Load every installed harness entry point not loaded yet, each failure isolated (recorded and logged)."""

    for entry_point in extensions.installed_entry_points("harness"):
        if extensions.is_loaded(entry_point):
            continue
        try:
            extensions.load_entry_point("harness", entry_point)
        except ExtensionError:
            pass


def _candidates(ref: str) -> list[Harness]:
    specs = default_registry().find(ref)
    functions = extensions.matching(ref, function_harnesses())
    if specs and functions and len({harness.ref for harness in [*specs, *functions]}) > 1:
        return [*specs, *functions]  # several versions, specs and functions among them: ambiguous, as for one kind
    if specs and functions:
        logger.warning(
            "harness %r names both the TOML spec(s) %s and the function harness(es) %s; using the spec",
            ref,
            sorted(spec.ref for spec in specs),
            sorted(function.ref for function in functions),
        )
    return list(specs or functions)


def _resolve(ref: str) -> Harness:
    _import_builtin_function_harnesses()
    name, _ = extensions.split_ref(ref)
    extensions.load_named("harness", name)
    matches = _candidates(ref)
    if not matches and not extensions.is_reserved("harness", name):
        _load_unloaded_entry_points()  # an entry point named after its module, not its harness
        matches = _candidates(ref)
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise extensions.ambiguous("harness", ref, [match.ref for match in matches])
    error = extensions.unknown("harness", ref, available_harnesses(), extra=", or name a .toml spec path")
    builtin_failures = _import_builtin_function_harnesses()
    if builtin_failures:
        raise ExtensionError(f"{error}. Built-in harness modules that failed to import: {builtin_failures}")
    raise error


def get_harness_spec(ref: str) -> HarnessSpec:
    """Resolve a TOML spec ``name[@version]`` (see :func:`get_harness`), or load a ``.toml`` path."""

    if ref.endswith(".toml"):
        return load_spec(ref)
    name, _ = extensions.split_ref(ref)
    extensions.load_named("harness", name)
    registry = default_registry()
    if not registry.find(ref) and not extensions.is_reserved("harness", name):
        _load_unloaded_entry_points()
    matches = registry.find(ref)
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise extensions.ambiguous("harness", ref, [spec.ref for spec in matches])
    raise extensions.unknown("harness", ref, registry.refs(), extra=", or name a .toml spec path")


def get_harness(ref: str) -> Harness:
    """Resolve ``name[@version]``, a ``.toml`` path or a ``module:function`` reference to a ``HarnessSpec`` or a
    ``FunctionHarness``.

    A name is a built-in's, an installed ``platoon.harnesses`` entry point's (imported the first time it is
    named), or one a ``package`` module registered.  Raises :class:`~platoon.extensions.ExtensionError` (a
    ``ValueError``) when nothing, or more than one harness, matches.
    """

    if ref.endswith(".toml"):
        return load_spec(ref)
    if ":" in ref:
        module, _, attr = ref.removeprefix("python://").partition(":")
        value = functools.reduce(getattr, attr.split("."), importlib.import_module(module))
        record = value if isinstance(value, FunctionHarness) else harness_of(value)
        if record is None:
            raise ValueError(f"{ref!r} is not an @harness function")
        return record
    return _resolve(ref)


def spec_origin(spec: HarnessSpec) -> Origin:
    """Where a spec came from: core for a built-in, the distribution of the entry point that added it, or nothing
    for a spec named by its path."""

    origin = default_registry().origin(spec.ref)
    if origin is not None and default_registry().find(spec.ref) == [spec]:
        return origin
    if _is_builtin_spec(spec):
        return Origin(distribution=extensions.RESERVED_OWNER, version=extensions.platoon_version())
    return Origin()


def harness_origin(harness: Harness) -> Origin:
    """Where a harness came from (:func:`spec_origin`, or the distribution of a function harness's module)."""

    if isinstance(harness, HarnessSpec):
        return spec_origin(harness)
    return extensions.origin_of("harness", harness.ref, harness.fn)


def available_harnesses() -> list[str]:
    """Every registered harness reference, TOML specs and functions: the built-ins, and the harnesses loaded or
    imported so far (an installed extension's harness appears once it is named; see
    :func:`platoon.extensions.list_components` for what is installed)."""

    _import_builtin_function_harnesses()
    return sorted({*default_registry().refs(), *(function.ref for function in function_harnesses())})


def builtin_harnesses() -> list[tuple[str, str]]:
    """Core's harnesses, ``(name@version, source)``: the built-in specs and function harnesses."""

    _import_builtin_function_harnesses()
    registry = default_registry()
    specs = [(spec.ref, spec.source or "-") for spec in map(registry.get, registry.refs()) if _is_builtin_spec(spec)]
    functions = [
        (record.ref, record.source)
        for record in function_harnesses()
        if extensions.is_core_module(record.fn.__module__)
    ]
    return sorted(specs + functions)


def harness_load_failures() -> dict[str, str]:
    """What failed to load, with why: built-in harness modules and ``platoon.harnesses`` entry points
    (``name = value``).  Their harnesses are missing from :func:`available_harnesses`."""

    return {
        **_import_builtin_function_harnesses(),
        **extensions.load_failures("harness"),
        **default_registry().entry_point_failures,
    }


__all__ = [
    "BUILTIN_FUNCTION_MODULES",
    "BUILTIN_SPECS_DIR",
    "ENTRY_POINT_GROUP",
    "Harness",
    "HarnessRegistry",
    "accept_entry_point",
    "available_harnesses",
    "builtin_harnesses",
    "default_registry",
    "get_harness",
    "get_harness_spec",
    "harness_load_failures",
    "harness_origin",
    "spec_origin",
]
