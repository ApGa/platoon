"""Write the snapshot of platoon's public API: every name its public modules export, with its signature.

    python scripts/public_api.py            # rewrite tests/public_api.txt from the code
    python scripts/public_api.py --check    # print how the code differs from the snapshot and exit 1, writing nothing

The public modules (:data:`PUBLIC_MODULES`) are the ones the documentation makes public: ``platoon`` itself
(``import platoon as pt``) and ``pt.gym``, which ``docs/reference/api.md`` documents, ``platoon.testing``, whose
guide (``docs/how-to/test-extensions.md``) makes its exports a stable API, and ``platoon.experimental``, whose names
are documented (``docs/reference/experimental.md``, which covers both of its APIs, and the guides
``docs/extending/task-samplers.md`` and ``docs/extending/group-admission.md``) but may change in a minor release: the
snapshot shows every such change too.  A module's public names are its ``__all__``.

The snapshot has one line per item, sorted: a function's signature; a class's bases and its public members (fields
with their types and defaults, attributes, properties, and methods with their signatures); a type alias's value; a
constant's type, and its value when that is a plain literal.  Docstrings are left out, so documentation never
changes it.  ``tests/test_public_api.py`` fails while the snapshot differs from the code, so that every change to the
public API is made on purpose: run this script, commit the snapshot, and say in the pull request what changed
(removing or renaming a public name starts as an RFC: CONTRIBUTING.md, "When a change needs an RFC").
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import enum
import functools
import importlib
import inspect
import json
import sys
import types
import typing
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path, PurePath
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO / "tests" / "public_api.txt"

# Public module -> the page that makes its exports public.
PUBLIC_MODULES: dict[str, str] = {
    "platoon": "docs/reference/api.md",
    "platoon.gym": "docs/reference/api.md",
    "platoon.testing": "docs/how-to/test-extensions.md",
    "platoon.experimental": "docs/reference/experimental.md",
}
# Exported modules that are public only as a callable, with what a call runs: ``pt.harness`` is the decorator.
CALLABLE_MODULES: dict[str, str] = {"platoon.harness": "platoon.harness.function:harness"}
# Dunder methods that belong to a class's interface; the others are implementation.
INTERFACE_DUNDERS = frozenset(
    {
        "__init__",
        "__call__",
        "__enter__",
        "__exit__",
        "__aenter__",
        "__aexit__",
        "__iter__",
        "__aiter__",
        "__len__",
        "__getitem__",
        "__contains__",
    }
)

HEADER = """\
# platoon's public API: every name the public modules export, with its signature (scripts/public_api.py).
# tests/test_public_api.py fails while this file differs from the code. After a change to the public API on
# purpose, run `python scripts/public_api.py` and commit this file with the change.
"""

_DATACLASS_DEFAULTS = {
    "init": True,
    "repr": True,
    "eq": True,
    "order": False,
    "unsafe_hash": False,
    "frozen": False,
    "match_args": True,
    "kw_only": False,
    "slots": False,
    "weakref_slot": False,
}


def public_names(module: str) -> list[str]:
    """The public names of the public module ``module``: its ``__all__``."""

    return list(importlib.import_module(module).__all__)


def entries(modules: Sequence[str] | None = None) -> list[tuple[str, str]]:
    """``(dotted path, description)`` of every item of the public API, sorted by path."""

    found: dict[str, str] = {}
    for module_name in PUBLIC_MODULES if modules is None else modules:
        module = importlib.import_module(module_name)
        for name in module.__all__:
            for path, text in describe(f"{module_name}.{name}", getattr(module, name)):
                found[path] = text
    return sorted(found.items())


def render(modules: Sequence[str] | None = None) -> str:
    """The snapshot: :data:`HEADER`, then one ``path: description`` line per item of the public API."""

    return HEADER + "".join(f"{path}: {text}\n" for path, text in entries(modules))


# --- one item -------------------------------------------------------------------------------------------------


def describe(path: str, value: Any) -> Iterator[tuple[str, str]]:
    """The snapshot lines of the public name at ``path``: one, or a class's line and one per public member."""

    if isinstance(value, types.ModuleType):
        target = CALLABLE_MODULES.get(value.__name__)
        if target is not None:
            module, _, attribute = target.partition(":")
            yield path, function_text(getattr(importlib.import_module(module), attribute))
        else:
            yield path, f"module {value.__name__}"
    elif _is_alias(value):
        yield path, f"alias {annotation_text(getattr(value, '__value__', value))}"
    elif isinstance(value, type):
        yield path, class_text(value)
        for name, text in class_members(value).items():
            yield f"{path}.{name}", text
    elif callable(value):
        yield path, function_text(value)
    else:
        yield path, constant_text(value)


def class_text(cls: type) -> str:
    """``class Name(Bases)``, after the non-default ``@dataclass`` parameters of a dataclass."""

    bases = [base for base in cls.__dict__.get("__orig_bases__", cls.__bases__) if base is not object]
    text = f"class {cls.__name__}"
    if bases:
        # A class base by its name (Protocol, AssertionError); a generic one (Generic[T]) as an annotation.
        names = [base.__qualname__ if isinstance(base, type) else annotation_text(base) for base in bases]
        text += f"({', '.join(names)})"
    if dataclasses.is_dataclass(cls):
        params = getattr(cls, "__dataclass_params__", None)
        changed = [
            f"{name}={getattr(params, name)!r}"
            for name, default in _DATACLASS_DEFAULTS.items()
            if hasattr(params, name) and getattr(params, name) != default
        ]
        text = f"@dataclass({', '.join(changed)}) {text}" if changed else f"@dataclass {text}"
    if issubclass(cls, enum.Enum):
        text += " [" + ", ".join(f"{member.name}={member.value!r}" for member in cls) + "]"
    return text


def class_members(cls: type) -> dict[str, str]:
    """``{name: description}`` of a class's public members, from the class and its bases that platoon defines
    (the subclass wins): dataclass fields, annotated attributes, properties, methods and class constants."""

    members: dict[str, str] = {}
    fields = {item.name: item for item in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else {}
    generated_init = dataclasses.is_dataclass(cls) and getattr(getattr(cls, "__dataclass_params__", None), "init", True)
    for base in reversed(cls.__mro__):
        if not _is_platoon(base):
            continue
        for name, annotation in inspect.get_annotations(base).items():
            if not name.startswith("_") and name not in fields:
                members[name] = f"attribute {annotation_text(annotation)}"
        for name, value in vars(base).items():
            if name.startswith("_") and name not in INTERFACE_DUNDERS:
                continue
            if name == "__init__" and generated_init:
                continue  # the fields say what the generated __init__ takes
            text = _member_text(value)
            if text is not None:
                members[name] = text
    for name, item in fields.items():
        if not name.startswith("_"):
            members[name] = field_text(item)
    return members


def _member_text(value: Any) -> str | None:
    if isinstance(value, (property, functools.cached_property)):
        getter = value.fget if isinstance(value, property) else value.func
        returns = inspect.signature(getter).return_annotation if getter else inspect.Signature.empty
        text = "property" if returns is inspect.Signature.empty else f"property {annotation_text(returns)}"
        return text + (" (settable)" if getattr(value, "fset", None) is not None else "")
    if isinstance(value, classmethod):
        return "classmethod " + function_text(value.__func__, drop_first=True)
    if isinstance(value, staticmethod):
        return "staticmethod " + function_text(value.__func__)
    if inspect.isfunction(value):
        if not _is_platoon(value):
            return None  # set by a library on the class (typing.Protocol's __init__, say)
        return function_text(value, drop_first=True)
    if isinstance(value, type):
        return class_text(value)
    if callable(value) or isinstance(value, (types.MemberDescriptorType, types.GetSetDescriptorType)):
        return None
    return constant_text(value)


def field_text(item: dataclasses.Field[Any]) -> str:
    """``field <type> = <default>``, with the field's non-default ``init`` and ``kw_only``."""

    text = f"field {annotation_text(item.type)}"
    if item.default is not dataclasses.MISSING:
        text += f" = {value_text(item.default)}"
    elif item.default_factory is not dataclasses.MISSING:
        text += f" = <factory {_qualified_name(item.default_factory)}>"
    flags = [flag for flag, on in (("init=False", not item.init), ("kw_only", item.kw_only is True)) if on]
    return text + (f" ({', '.join(flags)})" if flags else "")


def function_text(fn: Callable[..., Any], *, drop_first: bool = False) -> str:
    """``def name(params) -> returns`` (``async def`` for a coroutine or async generator function)."""

    unwrapped = inspect.unwrap(fn)
    asynchronous = inspect.iscoroutinefunction(unwrapped) or inspect.isasyncgenfunction(unwrapped)
    name = getattr(fn, "__name__", None) or getattr(unwrapped, "__name__", "?")
    return f"{'async def' if asynchronous else 'def'} {name}{signature_text(fn, drop_first=drop_first)}"


def signature_text(fn: Callable[..., Any], *, drop_first: bool = False) -> str:
    """``(params) -> returns`` on one line, with ``/`` and ``*`` where the parameter kinds change."""

    signature = inspect.signature(fn)
    params = list(signature.parameters.values())
    if drop_first and params and params[0].kind in (params[0].POSITIONAL_ONLY, params[0].POSITIONAL_OR_KEYWORD):
        params = params[1:]
    parts: list[str] = []
    star = False
    for index, param in enumerate(params):
        if param.kind is param.KEYWORD_ONLY and not star:
            parts.append("*")
            star = True
        if param.kind is param.VAR_POSITIONAL:
            star = True
        text = {param.VAR_POSITIONAL: "*", param.VAR_KEYWORD: "**"}.get(param.kind, "") + param.name
        if param.annotation is not param.empty:
            text += ": " + annotation_text(param.annotation)
        if param.default is not param.empty:
            text += (" = " if param.annotation is not param.empty else "=") + value_text(param.default)
        parts.append(text)
        following = params[index + 1] if index + 1 < len(params) else None
        if param.kind is param.POSITIONAL_ONLY and (following is None or following.kind is not param.POSITIONAL_ONLY):
            parts.append("/")
    returns = ""
    if signature.return_annotation is not signature.empty:
        returns = " -> " + annotation_text(signature.return_annotation)
    return f"({', '.join(parts)}){returns}"


def annotation_text(annotation: Any) -> str:
    """An annotation as the source writes it (a string annotation as it is), without the ``typing.`` and
    ``collections.abc.`` prefixes."""

    if isinstance(annotation, str):
        return annotation
    text = inspect.formatannotation(annotation)
    return text.replace("typing.", "").replace("collections.abc.", "")


def value_text(value: Any) -> str:
    """A default or constant value that reads the same in every process: strings in double quotes, literals by
    their ``repr``, enum members by name, classes and functions by their qualified name, anything else by its
    type."""

    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, PurePath):
        return "<Path>"  # its value differs between checkouts
    if value is None or isinstance(value, (bool, int, float, complex, bytes)):
        return repr(value)
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, (type, types.FunctionType, types.BuiltinFunctionType)):
        return _qualified_name(value)
    if isinstance(value, (tuple, list)) and all(_is_literal(item) for item in value):
        return repr(value)
    if isinstance(value, (frozenset, set)) and all(_is_literal(item) for item in value):
        return f"{type(value).__name__}({sorted(value, key=repr)!r})"
    if isinstance(value, dict) and all(_is_literal(key) and _is_literal(item) for key, item in value.items()):
        return repr(dict(sorted(value.items(), key=lambda pair: repr(pair[0]))))
    return f"<{type(value).__name__}>"


def constant_text(value: Any) -> str:
    """``constant <type> = <value>`` for a plain literal; ``constant <type>`` for anything else (a path, say,
    which differs between checkouts)."""

    kind = "Path" if isinstance(value, PurePath) else type(value).__name__
    text = value_text(value)
    return f"constant {kind}" if text.startswith("<") else f"constant {kind} = {text}"


def _is_literal(value: Any) -> bool:
    return value is None or isinstance(value, (str, bool, int, float, complex, bytes))


def _is_alias(value: Any) -> bool:
    return typing.get_origin(value) is not None or isinstance(value, typing.TypeAliasType)


def _is_platoon(value: Any) -> bool:
    module = getattr(value, "__module__", None) or ""
    return module == "platoon" or module.startswith("platoon.")


def _qualified_name(value: Any) -> str:
    module = getattr(value, "__module__", None)
    name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None) or type(value).__name__
    return name if module in (None, "builtins") else f"{module}.{name}"


# --- main -------------------------------------------------------------------------------------------------------


def diff(committed: str, current: str) -> str:
    """A unified diff from the committed snapshot to the code's."""

    return "".join(
        difflib.unified_diff(
            committed.splitlines(keepends=True),
            current.splitlines(keepends=True),
            fromfile="tests/public_api.txt (committed)",
            tofile="tests/public_api.txt (from the code)",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the snapshot of platoon's public API.")
    parser.add_argument("--check", action="store_true", help="print the differences and exit 1, writing nothing")
    args = parser.parse_args(argv)
    current = render()
    committed = SNAPSHOT.read_text(encoding="utf-8") if SNAPSHOT.exists() else ""
    if current == committed:
        print(f"{SNAPSHOT.relative_to(REPO)} is up to date")
        return 0
    if args.check:
        print(diff(committed, current), end="")
        return 1
    SNAPSHOT.write_text(current, encoding="utf-8")
    print(f"wrote {SNAPSHOT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    sys.exit(main())
