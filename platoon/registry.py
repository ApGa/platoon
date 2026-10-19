"""Small typed registries of components that configs name: loaders, rollouts, reward processors, workflows.

:func:`resolve_component` resolves, for the kinds that have an entry-point group (reward processors:
``platoon.reward_processors``), core's built-in of that name (:data:`BUILTIN_REWARD_PROCESSORS`, such as
``platoon-delegation``), then the installed entry point of that exact name, then a name a ``package`` module
registered (a name that both provide is an error that names both), then an import string (``module:attr`` or
``module.attr``); the other kinds, a registered name, then an import string.  A
dotted ``module.attr`` that imports keeps that meaning even when an installed entry point has the same name, so
installing a package never changes it.
Harnesses, programs and named environments have registries of their own (:mod:`platoon.harness.registry`,
:mod:`platoon.programs`, :mod:`platoon.envs.named`).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from importlib.metadata import EntryPoint

    from platoon.extensions import Origin

T = TypeVar("T")


@dataclass(frozen=True)
class RegistryItem(Generic[T]):
    """A registered component plus metadata useful for serialization/docs."""

    name: str
    value: T
    import_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Registry(Generic[T]):
    """Name-to-component registry with decorator-friendly registration."""

    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, RegistryItem[T]] = {}

    def register(
        self,
        name: str,
        value: T | None = None,
        *,
        import_path: str | None = None,
        exist_ok: bool = False,
        **metadata: Any,
    ) -> Callable[[T], T] | T:
        """Register a value immediately or return a decorator.

        A kind that extensions provide by name (``reward_processor``) takes names that follow platoon's naming rule
        (:mod:`platoon.extensions`).
        """

        def decorator(component: T) -> T:
            from platoon import extensions

            if self.kind in extensions.GROUPS:
                extensions.check_registration(self.kind, name, module=getattr(component, "__module__", None))
            if name in self._items and not exist_ok:
                raise ValueError(f"{self.kind!r} registry already has an entry named {name!r}")
            self._items[name] = RegistryItem(
                name=name,
                value=component,
                import_path=import_path or infer_import_path(component),
                metadata=dict(metadata),
            )
            return component

        if value is None:
            return decorator
        return decorator(value)

    def get(self, name: str) -> T:
        if name not in self._items:
            available = sorted(self._items)
            raise ValueError(f"Unknown {self.kind}: {name!r}. Available: {available}")
        return self._items[name].value

    def get_item(self, name: str) -> RegistryItem[T]:
        if name not in self._items:
            available = sorted(self._items)
            raise ValueError(f"Unknown {self.kind}: {name!r}. Available: {available}")
        return self._items[name]

    def resolve(self, spec: str | T) -> T:
        """Resolve a registry name, import path, or already-materialized value."""

        if isinstance(spec, str):
            if spec in self._items:
                return self.get(spec)
            return import_from_string(spec)
        return spec

    def names(self) -> list[str]:
        return list(self._items.keys())

    def items(self) -> list[RegistryItem[T]]:
        return list(self._items.values())


_REGISTRIES: dict[str, Registry[Any]] = {}
# Core's reward processors: name -> where the function is defined.  A name here resolves to core's function before
# any entry point is looked at, so it needs nothing installed and no installed distribution can shadow it.
BUILTIN_REWARD_PROCESSORS: dict[str, str] = {"platoon-delegation": "platoon.rewards:delegation_capped"}


def get_registry(kind: str) -> Registry[Any]:
    """Return a process-local registry by kind, creating it on first use."""

    if kind not in _REGISTRIES:
        _REGISTRIES[kind] = Registry(kind)
    return _REGISTRIES[kind]


def register_component(kind: str, name: str, value: T | None = None, **kwargs: Any) -> Callable[[T], T] | T:
    """Register a component in a named registry."""

    return get_registry(kind).register(name, value, **kwargs)


def resolve_component(kind: str, spec: str | T) -> T:
    """Resolve a component: a registered name, an installed extension's entry point of that exact name (for the
    kinds that have an entry-point group), or a ``module:attr`` / ``module.attr`` import path."""

    registry = get_registry(kind)
    if not isinstance(spec, str):
        return registry.resolve(spec)
    from platoon import extensions

    if kind not in extensions.GROUPS or ":" in spec:
        return registry.resolve(spec)
    if kind == "reward_processor" and spec in BUILTIN_REWARD_PROCESSORS:
        return builtin_reward_processor(spec)
    local = spec in registry.names() and not registry.get_item(spec).metadata.get("entry_point")
    if "." in spec and not local and extensions.entry_points_named(kind, spec):
        # ``module.attr``, the dotted form from before ``module:attr``, keeps meaning what it imports: an entry point
        # that an installed package names the same way is used only when no such module or attribute exists.
        imported = _import_dotted(spec)
        if imported is not _MISSING:
            return imported
    # The entry point named exactly ``spec`` registers here; one that a ``package`` module registered under the same
    # name already is an error that names both, as for harnesses and programs.
    extensions.load_named(kind, spec)
    if spec in registry.names():
        return registry.get(spec)
    try:
        return import_from_string(spec)
    except (ImportError, AttributeError, ValueError) as exc:
        raise extensions.unknown(
            kind, spec, registry.names(), extra=f" (as an import path it fails: {type(exc).__name__}: {exc})"
        ) from exc


def builtin_reward_processor(name: str) -> Any:
    """Core's reward processor ``name`` (:data:`BUILTIN_REWARD_PROCESSORS`), registered under its name (again, after
    a test restored the registry)."""

    from platoon import extensions

    registry = get_registry("reward_processor")
    item = registry._items.get(name)
    if item is not None and item.metadata.get("builtin"):
        return item.value
    source = BUILTIN_REWARD_PROCESSORS[name]
    value = import_from_string(source)
    # A name of core's, registered by core: the naming rule holds by construction.
    registry._items[name] = RegistryItem(
        name=name, value=value, import_path=infer_import_path(value), metadata={"builtin": True}
    )
    module = source.partition(":")[0]
    origin = extensions.Origin(
        distribution=extensions.RESERVED_OWNER, version=extensions.platoon_version(), module=module
    )
    extensions.record_origin("reward_processor", name, origin)
    return value


def builtin_reward_processors() -> list[tuple[str, str]]:
    """Core's reward processors, registered: ``(name, module:attr)`` each."""

    for name in BUILTIN_REWARD_PROCESSORS:
        builtin_reward_processor(name)
    return sorted(BUILTIN_REWARD_PROCESSORS.items())


def accept_reward_processor(value: Any, origin: Origin, entry_point: EntryPoint) -> None:
    """Register what a ``platoon.reward_processors`` entry point loads to (a callable ``fn(trajectory) -> (score,
    components)``) under the entry point's name."""

    from platoon import extensions

    if not callable(value):
        raise TypeError(f"a platoon.reward_processors entry point must load to a callable, got {value!r}")
    registry = get_registry("reward_processor")
    if entry_point.name in registry.names() and registry.get(entry_point.name) is not value:
        raise ValueError(
            f"reward processor {entry_point.name!r} is defined twice: "
            f"{registry.get_item(entry_point.name).import_path} and {entry_point.value}"
        )
    # The entry point's name passed the naming rule for its distribution when it loaded.
    registry._items[entry_point.name] = RegistryItem(
        name=entry_point.name, value=value, import_path=infer_import_path(value), metadata={"entry_point": True}
    )
    extensions.record_origin("reward_processor", entry_point.name, origin)


_MISSING: Any = object()


def _import_dotted(path: str) -> Any:
    """What the dotted import path ``module.attr`` names, or :data:`_MISSING` when no such module or attribute
    exists (or ``path`` is no import path); an error raised inside an existing module propagates."""

    module_path, _, attr = path.rpartition(".")
    if not module_path or not all(part.isidentifier() for part in path.split(".")):
        return _MISSING
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        if exc.name is not None and (module_path == exc.name or module_path.startswith(exc.name + ".")):
            return _MISSING
        raise
    return getattr(module, attr, _MISSING)


def import_from_string(path: str) -> Any:
    """Import ``module.attr`` or ``module:attr[.attr...]`` references (``pkg.mod:Outer.inner`` too)."""

    module_path, colon, attr = path.partition(":")
    if colon and module_path and attr and ":" not in attr:
        value: Any = importlib.import_module(module_path)
        try:
            for part in attr.split("."):
                value = getattr(value, part)
            return value
        except AttributeError:
            if "." not in attr:
                raise
            # ``pkg:mod.attr``, which once read as the dotted ``pkg.mod.attr``: read it that way below.
    module_path, separator, attr = path.replace(":", ".").rpartition(".")
    if not separator or not module_path or not attr:
        raise ValueError(f"Expected an import path like 'package.module.object', got {path!r}")
    value = importlib.import_module(module_path)
    for part in attr.split("."):
        value = getattr(value, part)
    return value


def infer_import_path(value: Any) -> str | None:
    """Best-effort import path for functions/classes used in remote workers."""

    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not module or not qualname or "<locals>" in qualname or module == "__main__":
        return None
    return f"{module}.{qualname}"


def import_modules(module_paths: Iterable[str]) -> None:
    """Import modules for registration side effects."""

    for module_path in module_paths:
        importlib.import_module(module_path)


def register_dataset_loader(name: str, value: T | None = None, **kwargs: Any) -> Callable[[T], T] | T:
    return register_component("dataset_loader", name, value, **kwargs)


def register_task_loader(name: str, value: T | None = None, **kwargs: Any) -> Callable[[T], T] | T:
    return register_component("task_loader", name, value, **kwargs)


def register_rollout(name: str, value: T | None = None, **kwargs: Any) -> Callable[[T], T] | T:
    return register_component("rollout", name, value, **kwargs)


def register_reward_processor(name: str, value: T | None = None, **kwargs: Any) -> Callable[[T], T] | T:
    return register_component("reward_processor", name, value, **kwargs)


def register_workflow(name: str, value: T | None = None, **kwargs: Any) -> Callable[[T], T] | T:
    return register_component("workflow", name, value, **kwargs)


def register_trainer_config(name: str, value: T | None = None, **kwargs: Any) -> Callable[[T], T] | T:
    return register_component("trainer_config", name, value, **kwargs)
