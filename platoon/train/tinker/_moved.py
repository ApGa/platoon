"""The shims of the module paths that platoon 0.5.0 moved out of ``platoon.train.tinker``.

Each old module still imports, and code written against it keeps working:

- A name read through it is the moved object itself, so ``isinstance`` checks, pickles and class identity hold, and
  the first read of each name warns with a :class:`~platoon.deprecations.PlatoonDeprecationWarning` that names the
  module to import it from.  A read is live: a module global that the moved code rebinds reads as it is now.
- A module that platoon 0.5.0 split still has the names it only imported: such a name reads from the first of the
  module's new modules that has it (``homes``, the one that holds its main class first).
- ``from <old module> import *`` binds what the new module exports, for a module that moved whole, and the public names
  of the definitions that moved out of it, for one that was split.
- A name set or deleted through an old module, as a test's ``monkeypatch.setattr`` or ``mock.patch`` does, is set or
  deleted in each of its new modules that holds the name's object (in each that has one, for a module's own
  ``logger``; in each of them, for a name that none has, such as a builtin's), so the moved code sees the patch as the
  code saw it in the old module; a name deleted through it reads as gone until it is set again; and setting it back to
  what it was before, as undoing a patch does, puts back what each of those modules held.  Undoing a ``mock.patch``
  through it, with ``create=True`` or without and nested in other patches or not, does what it did in the old module:
  it sets back what the patch found there, or deletes the name if the patch found none (:func:`_found`).  A moved
  method's module function is not its class's method, though: the class keeps it under its name, and a patch of the
  class or of an instance is what reaches the code.
- An old package's names are the package's own, as they were: setting or deleting one changes the package only.  The
  submodules that its init imported before 0.5.0 are its attributes, each imported when it is first read.
- Importing an old module path imports the trainer, as it did before 0.5.0 through the init of
  ``platoon.train.tinker``, and its start, and binds the names that ``platoon.train`` and ``platoon.train.workflow``
  import lazily, so that no module of the trainer is first imported, and none of those names first read, while a
  patch made through an old path is active (:func:`_import_trainer`).  ``platoon.train.tinker`` itself, whose policy
  client the new code imports, does not.

Platoon 0.6.0 removes the shims (CONTRIBUTING.md, "Deprecations").  Platoon's own code never imports one
(``tests/test_moved_modules.py``).
"""

from __future__ import annotations

import importlib
import os
import sys
import types
import warnings
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from platoon.deprecations import PlatoonDeprecationWarning

#: The release that removes the shims.
REMOVED_IN = "0.6.0"
#: The release that moved the modules.
MOVED_IN = "0.5.0"
# A warning names the frame that read the name: the first one outside platoon's package; or, when platoon itself read it
# for someone (a pickle or a config that names an old path) under none but the interpreter's own frames (python -m),
# platoon's frame that read it.
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + os.sep
_MISSING = object()
# What a delete through an old module passes for the value.
_DELETE = object()
# Where an old module's namespace keeps the function that sets a name through it (:class:`_OldModule`).
_WRITE = "__moved_write__"
# The names that each module defines for itself, as a split module's new modules each define their own logger: a name
# set through the old module is set in each of them that has one.
_OWN = frozenset({"logger"})


class _OldModule(types.ModuleType):
    """The module of an old module path: a name set or deleted through it is set or deleted where the moved code reads
    it, or, for an old package, kept as the package's own (:func:`moved`); and a delete that undoes a ``mock.patch``
    sets back what the patch found (:func:`_found`)."""

    def __setattr__(self, attribute: str, value: Any) -> None:
        write = self.__dict__.get(_WRITE)
        if write is None or not write(attribute, value):
            super().__setattr__(attribute, value)

    def __delattr__(self, attribute: str) -> None:
        found = _found(sys._getframe(1), self, attribute)
        write = self.__dict__.get(_WRITE)
        try:
            if write is None or not write(attribute, _DELETE):
                super().__delattr__(attribute)
        except AttributeError:
            if found is _MISSING:  # an undoing sets back what its patch found, even if the name was deleted meanwhile
                raise
        if found is not _MISSING and not hasattr(self, attribute):
            setattr(self, attribute, found)


def _found(frame: types.FrameType, module: types.ModuleType, attribute: str) -> Any:
    """What the ``mock.patch`` of ``module.attribute`` that ``frame`` undoes found there: _MISSING if ``frame`` undoes
    no such patch, or if the patch found none.

    Mock undoes a patch of a name that its target does not hold in its own ``__dict__``, as a shim does not hold the
    names it re-exports, by deleting the name; then, if the name reads as gone and the patch was made without
    ``create=True`` (which mock forces for a builtin's name), it sets back what it found.  It set back a name of the
    old module, which held its names, in any case: the shim does that after the delete.  ``frame`` is mock's undoing
    if it runs ``_patch.__exit__`` in a module that has mock's ``DEFAULT``: unittest.mock's, or the same code that the
    ``mock`` package (which pytest-mock can use) runs as ``mock.mock``."""

    if frame.f_code.co_qualname != "_patch.__exit__" or "DEFAULT" not in frame.f_globals:
        return _MISSING
    patch, default = frame.f_locals["self"], frame.f_globals["DEFAULT"]
    if getattr(patch, "target", None) is not module or patch.attribute != attribute or patch.temp_original is default:
        return _MISSING
    return patch.temp_original


def _module(name: str) -> types.ModuleType:
    return sys.modules.get(name) or importlib.import_module(name)


def _import_trainer() -> None:
    """Import the trainer, as importing an old module path did before platoon 0.5.0 (through the init of
    ``platoon.train.tinker``, which imported it then): read each name that ``platoon.train`` and
    ``platoon.train.workflow`` import lazily, which imports the trainer's modules and binds those names; and import
    ``platoon.train.launch``, which imports ``GroupRolloutWorkflow`` from its module, where the trainer's start before
    0.5.0 imported it from the old package.  A patch made through an old path, after its import, then finds them all
    imported and bound: none is first imported or read while it is active, which would keep the patch's object."""

    importlib.import_module("platoon.train.launch")
    for name in ("platoon.train", "platoon.train.workflow"):
        package = importlib.import_module(name)
        for attribute in package.__all__:
            getattr(package, attribute)


def _reader() -> types.FrameType | None:
    inner = None
    frame = sys._getframe(1)
    while frame is not None:
        module = str(frame.f_globals.get("__name__") or "")
        filename = frame.f_code.co_filename
        if module == __name__ or module == "importlib" or module.startswith("importlib._bootstrap"):
            pass
        elif filename.startswith(_PACKAGE_DIR):
            inner = inner or frame
        elif filename.startswith("<") and inner is not None:
            return inner
        else:
            return frame
        frame = frame.f_back
    return inner


def _warn(message: str) -> None:
    frame = _reader()
    if frame is None:
        warnings.warn(message, PlatoonDeprecationWarning, stacklevel=2)
        return
    module_globals = frame.f_globals  # as warnings.warn has it: the registry that "once per location" filters keep
    warnings.warn_explicit(
        message,
        PlatoonDeprecationWarning,
        frame.f_code.co_filename,
        frame.f_lineno,
        module=module_globals.get("__name__", "<string>"),
        registry=module_globals.setdefault("__warningregistry__", {}),
    )


def moved(
    namespace: dict[str, Any],
    targets: str | Mapping[str, str],
    *,
    name: str | None = None,
    homes: Iterable[str] = (),
    public: Mapping[str, str] | None = None,
    submodules: Iterable[str] = (),
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """The ``__getattr__`` and ``__dir__`` of a shim module whose globals are ``namespace``: ``targets`` is the module
    the whole old module moved to, or, for a module that was split or a package, each name's new module (for a split
    module, the module that defines the name).  ``homes`` are a split module's new modules, the one that holds its main
    class first: where a name the old module only imported is read, and where a name set through it is set.
    ``public`` names the package that re-exports a split module's name, which the warning says to import it from.
    ``submodules`` are an old package's modules that its init imported.  ``name`` is the old module's name, for a shim
    that also runs as ``__main__``.  The old-to-new map is kept in the shim's ``__moved__``."""

    old = name or namespace["__name__"]
    if old != __package__:  # platoon.train.tinker itself stays light: the new code imports its policy client
        _import_trainer()
    namespace["__moved__"] = targets
    whole = isinstance(targets, str)
    package = "__path__" in namespace
    places: tuple[str, ...] = (targets,) if isinstance(targets, str) else tuple(homes)
    mapped: Mapping[str, str] = {} if isinstance(targets, str) else targets  # a split module's or a package's names
    advised = dict(public or {})
    offered = frozenset(submodules)
    own = frozenset(namespace)  # the names the shim defines itself, which stay its own
    warned: set[str] = set()
    # A name set or deleted through a module shim, until it is set back to what it was: the module it reads from, what
    # it read before the first change, and what each of the new modules the changes go to held then (_MISSING: none).
    written: dict[str, tuple[str, Any, dict[str, Any]]] = {}
    # An old package's names deleted through it, which read as gone until they are set again.
    deleted: set[str] = set()

    def find(attribute: str) -> tuple[str, Any]:
        """The module ``attribute`` reads from now, and its value (AttributeError: none has it)."""

        if attribute in written:
            main = written[attribute][0]
            value = vars(_module(main)).get(attribute, _MISSING)
            if value is _MISSING:
                raise AttributeError(attribute)
            return main, value
        if isinstance(targets, str):
            return targets, getattr(_module(targets), attribute)
        new = targets.get(attribute)
        if new is not None:
            return new, getattr(_module(new), attribute)
        for home in places:
            value = vars(_module(home)).get(attribute, _MISSING)
            if value is not _MISSING:
                return home, value
        raise AttributeError(attribute)

    def note(attribute: str, new: str, value: Any) -> None:
        """Warn of ``attribute``, the first time it is used through the old path."""

        if attribute in warned:
            return
        warned.add(attribute)
        new = advised.get(attribute, new)
        if whole or attribute in targets:
            advice = f"import it from {new}"
        elif isinstance(value, types.ModuleType):  # a module the old module imported
            advice = f"import {value.__name__}"
        elif hasattr(value, "__qualname__") and isinstance(getattr(value, "__module__", None), str):
            advice = f"import it from {value.__module__}"  # a class or a function it imported: from its own module
        else:
            advice = f"import it from {new}"
        _warn(
            f"{old}.{attribute} moved to {new}.{attribute} in platoon {MOVED_IN}: {advice} "
            f"(platoon {REMOVED_IN} removes {old}.{attribute})"
        )

    # What ``from <old> import *`` binds when the new module has no ``__all__`` of its own: its public names, the
    # first time they are asked for (a split module's: the public names that moved).
    exports: list[list[str]] = [] if isinstance(targets, str) else [[n for n in targets if not n.startswith("_")]]

    def exported() -> list[str]:
        """The old module's ``__all__``: the new module's own (the very list), else its public names, or the public
        names that moved out of a split module; the same list at each read."""

        if isinstance(targets, str):
            target = _module(targets)
            names = getattr(target, "__all__", None)
            if names is not None:
                return names
            if not exports:
                exports.append([n for n in vars(target) if not n.startswith("_")])
        return exports[0]

    def __getattr__(attribute: str) -> Any:
        if attribute == "__all__":  # not a package's: it has its own
            return exported()
        if (attribute.startswith("__") and attribute.endswith("__")) or attribute in deleted or attribute in own:
            raise AttributeError(f"module {old!r} has no attribute {attribute!r}")
        try:
            new, value = find(attribute)
        except AttributeError:
            if package and attribute in offered:
                return importlib.import_module(f"{old}.{attribute}")  # which sets it in the package, as its init did
            raise AttributeError(f"module {old!r} has no attribute {attribute!r}") from None
        note(attribute, new, value)
        return value

    def __dir__() -> list[str]:
        names = set(namespace)
        if isinstance(targets, str):
            names.update(dir(_module(targets)))
        else:
            names.update(targets)
            for home in places:
                names.update(vars(_module(home)))
        names.update(offered)
        gone = {attribute for attribute, (main, _, _) in written.items() if attribute not in vars(_module(main))}
        return sorted(names - deleted - gone)

    def reach(attribute: str, current: Any) -> list[str]:
        """The new modules that a change of ``attribute`` through the old module goes to: each that holds its object
        ``current`` (each that has one, for a module's own name), or each of them, for a name that none has."""

        if current is _MISSING:
            return list(places)
        if attribute in _OWN:
            return [home for home in places if attribute in vars(_module(home))]
        return [home for home in places if vars(_module(home)).get(attribute, _MISSING) is current]

    def write(attribute: str, value: Any) -> bool:
        """Set ``attribute`` to ``value`` (or delete it, for ``_DELETE``) where the moved code reads it; False for a
        name the shim keeps itself (its own names), or one that it cannot delete (none has it)."""

        if (attribute.startswith("__") and attribute.endswith("__")) or attribute in namespace or attribute in own:
            return False
        if attribute not in written:
            try:
                main, current = find(attribute)
            except AttributeError:
                main, current = (places[0] if places else ""), _MISSING
            if current is _MISSING and value is _DELETE:
                return False
            holders = reach(attribute, current)  # importing each of them first, as importing the old module did
            if value is current:
                return True
            if not holders:
                return False
            if current is not _MISSING:
                note(attribute, main, current)
            held = {home: vars(_module(home)).get(attribute, _MISSING) for home in holders}
            written[attribute] = (main, current, held)
        main, before, held = written[attribute]
        if value is _DELETE:
            if attribute not in vars(_module(main)):
                return False  # deleted already, as a second delete of a module's name raises AttributeError
            for home in held:
                vars(_module(home)).pop(attribute, None)
            if all(previous is _MISSING for previous in held.values()):
                del written[attribute]  # none of them had it before: as it was
            return True
        if value is before:  # set back (a patch undone): what each held before the first change
            for home, previous in held.items():
                if previous is _MISSING:
                    vars(_module(home)).pop(attribute, None)
                else:
                    setattr(_module(home), attribute, previous)
            del written[attribute]
            return True
        for home in held:
            setattr(_module(home), attribute, value)
        return True

    def keep(attribute: str, value: Any) -> bool:
        """Set ``attribute`` to ``value`` (or delete it, for ``_DELETE``) in the old package, whose names are its own;
        False for a name set or deleted as in any module."""

        if attribute.startswith("__") and attribute.endswith("__"):
            return False
        mine = attribute in mapped or attribute in offered
        if value is _DELETE:
            if attribute in namespace:
                del namespace[attribute]
            elif not mine or attribute in deleted:
                return False
            if mine:
                deleted.add(attribute)
            return True
        deleted.discard(attribute)
        if attribute in mapped and value is getattr(_module(mapped[attribute]), attribute, _MISSING):
            namespace.pop(attribute, None)  # set back to the name it re-exports: read from there again
            return True
        return False

    if namespace["__name__"] == old:
        module = sys.modules.get(old)
        if module is not None and module.__dict__ is namespace:
            namespace[_WRITE] = keep if package else write
            if type(module) is types.ModuleType:
                module.__class__ = _OldModule
    return __getattr__, __dir__


def warn_run(old: str, new: str) -> None:
    """Warn that ``python -m old``, which runs ``new``'s ``main``, is going away."""

    warnings.warn(
        f"python -m {old} is deprecated: platoon {MOVED_IN} moved it to {new}; run python -m {new} "
        f"(platoon {REMOVED_IN} removes {old})",
        PlatoonDeprecationWarning,
        stacklevel=2,
    )
