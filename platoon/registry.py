"""Small typed registries for plugin-provided Platoon components."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Generic, TypeVar

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
        """Register a value immediately or return a decorator."""

        def decorator(component: T) -> T:
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


def get_registry(kind: str) -> Registry[Any]:
    """Return a process-local registry by kind, creating it on first use."""

    if kind not in _REGISTRIES:
        _REGISTRIES[kind] = Registry(kind)
    return _REGISTRIES[kind]


def register_component(kind: str, name: str, value: T | None = None, **kwargs: Any) -> Callable[[T], T] | T:
    """Register a component in a named registry."""

    return get_registry(kind).register(name, value, **kwargs)


def resolve_component(kind: str, spec: str | T) -> T:
    """Resolve a component from a named registry or dotted import path."""

    return get_registry(kind).resolve(spec)


def import_from_string(path: str) -> Any:
    """Import ``module.attr`` or ``module:attr`` references."""

    module_path, separator, attr = path.replace(":", ".").rpartition(".")
    if not separator or not module_path or not attr:
        raise ValueError(f"Expected an import path like 'package.module.object', got {path!r}")
    module = importlib.import_module(module_path)
    value: Any = module
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


def discover_entry_points(group: str = "platoon.plugins") -> list[str]:
    """Import plugin registration modules advertised through package entry points."""

    loaded: list[str] = []
    for entry_point in entry_points(group=group):
        entry_point.load()
        loaded.append(entry_point.name)
    return loaded


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
