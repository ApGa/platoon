"""The module paths that platoon 0.5.0 moved out of ``platoon.train.tinker`` (CHANGELOG, 0.5.0).

Each old path is a shim (``platoon/train/tinker/_moved.py``) that re-exports the moved objects, warning once per name,
until platoon 0.6.0 removes it: the names a split module only imported too, its star import, a patch made through
it and its undoing, and an old package's submodules.  Platoon's own code, its tests, docs and examples use the new
paths only.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TINKER = REPO / "platoon" / "train" / "tinker"


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(REPO).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _shims() -> tuple[dict[str, set[str] | None], dict[str, list[str]], dict[str, dict[str, str]]]:
    """Each shim module, with the names it maps (None: a whole module, every name); each split module's shim, with its
    new modules (``homes``), where the names it only imported read from; and the packages that re-export a split
    module's names, which its warnings name (``public``)."""

    found: dict[str, set[str] | None] = {}
    homes: dict[str, list[str]] = {}
    public: dict[str, dict[str, str]] = {}
    for path in sorted(TINKER.rglob("*.py")):
        if path.name == "_moved.py" or "_moved(" not in path.read_text():
            continue
        names: set[str] | None = None
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_moved" and len(node.args) == 2:
                if isinstance(node.args[1], ast.Dict):
                    names = {str(key.value) for key in node.args[1].keys if isinstance(key, ast.Constant)}
                for keyword in node.keywords:
                    if keyword.arg == "homes" and isinstance(keyword.value, ast.Tuple):
                        items = keyword.value.elts
                        homes[_module_name(path)] = [str(i.value) for i in items if isinstance(i, ast.Constant)]
                    elif keyword.arg == "public" and isinstance(keyword.value, ast.Dict):
                        pairs = zip(keyword.value.keys, keyword.value.values, strict=True)
                        public[_module_name(path)] = {
                            str(key.value): str(value.value)
                            for key, value in pairs
                            if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
                        }
        found[_module_name(path)] = names
    return found, homes, public


SHIMS, HOMES, PUBLIC = _shims()
# What may still name an old path: the shims themselves, the entrypoint's old name that the worker invariants still
# recognize and the supervisor's prefix for it, the release notes and the RFCs (history), and this test.
_ALLOWED_FILES = {
    "CHANGELOG.md",
    "tests/test_moved_modules.py",
}
_ALLOWED_STRINGS = {
    ("platoon/train/workers/env.py", "platoon.train.tinker.workers.entrypoint"),
    ("platoon/train/workers/supervisor.py", "platoon.train.tinker."),
}


def _shim_reference(dotted: str) -> str | None:
    """The shim that ``dotted`` (a module, or a module's attribute) reaches, if any."""

    parts = dotted.split(".")
    for cut in range(len(parts), 3, -1):
        module = ".".join(parts[:cut])
        if module in SHIMS:
            names = SHIMS[module]
            if cut == len(parts) or names is None or parts[cut] in names:
                return module
            sub = f"{module}.{parts[cut]}"
            return sub if sub in SHIMS else None
    if len(parts) >= 4 and parts[:3] == ["platoon", "train", "tinker"]:
        names = SHIMS.get("platoon.train.tinker") or set()
        if parts[3] in names:
            return "platoon.train.tinker"
    return None


def _python_files() -> list[str]:
    listed = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-co", "--exclude-standard", "*.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [path for path in listed if (REPO / path).is_file()]


def _shim_uses(relative: str) -> list[str]:
    """The imports of a shim in a Python file, and the strings that name one as a module (``python -m``, a
    ``module:attr`` reference, a patch target); docstrings aside."""

    tree = ast.parse((REPO / relative).read_text())
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                reached = _shim_reference(f"{node.module}.{alias.name}") or _shim_reference(node.module)
                if node.module == "platoon.train.tinker" and _shim_reference(f"{node.module}.{alias.name}") is None:
                    reached = None  # a module that stays in the package (proxy, renderers, ...)
                if reached:
                    found.append(f"{relative}:{node.lineno}: from {node.module} import {alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _shim_reference(alias.name):
                    found.append(f"{relative}:{node.lineno}: import {alias.name}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            for dotted in re.findall(r"platoon\.train\.tinker(?:\.[A-Za-z_]\w*)+", node.value):
                if (relative, dotted) in _ALLOWED_STRINGS or (relative, node.value) in _ALLOWED_STRINGS:
                    continue
                if _shim_reference(dotted):
                    found.append(f"{relative}:{node.lineno}: {node.value!r}")
    return found


def test_no_code_imports_a_moved_module_path():
    """Platoon imports none of its shims, and nor do its tests, scripts, examples, recipes or extensions: the codemod
    that moved the code rewrote them all."""

    files = [
        path for path in _python_files() if not path.startswith("platoon/train/tinker/") and path not in _ALLOWED_FILES
    ]
    uses = [use for path in files for use in _shim_uses(path)]
    assert not uses, "\n".join(uses)


def test_no_doc_names_a_moved_module_path():
    """The docs, the README and CONTRIBUTING name the new paths; the RFCs and the release notes keep their history."""

    shim_files = {
        name.removeprefix("platoon.train.tinker.").replace(".", "/") for name in SHIMS if name != "platoon.train.tinker"
    }
    pattern = re.compile(r"platoon[./]train[./]tinker[./]([A-Za-z_][\w./]*)")
    found = []
    docs = [*sorted((REPO / "docs").rglob("*.md")), REPO / "README.md", REPO / "CONTRIBUTING.md"]
    for path in docs:
        relative = path.relative_to(REPO).as_posix()
        if relative.startswith("docs/rfcs/"):
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            for match in pattern.finditer(line):
                rest = match.group(1).replace(".", "/")
                stem = rest[: -len("/py")] if rest.endswith("/py") else rest
                if any(stem == shim or stem.startswith(shim + "/") for shim in shim_files) or (
                    _shim_reference(match.group(0).replace("/", ".").removesuffix(".py"))
                ):
                    found.append(f"{relative}:{number}: {match.group(0)}")
    assert not found, "\n".join(found)


_SHIM_CHECK = """
import importlib, inspect, json, sys, warnings

from platoon.deprecations import PlatoonDeprecationWarning

shims, homes, public = (json.loads(argument) for argument in sys.argv[1:4])
MISSING = object()
failures = []
checked = 0


def read(name, module, attribute, expected, new):
    global checked
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = getattr(module, attribute)
        again = getattr(module, attribute)
    checked += 1
    warned = [str(item.message) for item in caught if issubclass(item.category, PlatoonDeprecationWarning)]
    if value is not expected or again is not expected:
        failures.append(f"{name}.{attribute} is not {new}.{attribute}")
    if len(warned) != 1:
        failures.append(f"{name}.{attribute} warned {len(warned)} times")
    elif f"{new}.{attribute}" not in warned[0] or f"0.6.0 removes {name}.{attribute}" not in warned[0]:
        failures.append(f"{name}.{attribute}: {warned[0]}")


for name in shims:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module(name)
    if any(issubclass(item.category, PlatoonDeprecationWarning) for item in caught):
        failures.append(f"importing {name} warns")
    targets = module.__moved__
    own = set(vars(module))  # what the shim defines itself, before any name is read through it
    if isinstance(targets, str):
        target = importlib.import_module(targets)
        names = {attribute: targets for attribute in vars(target) if not attribute.startswith("__")}
    else:
        names = dict(targets)
    names = {attribute: new for attribute, new in names.items() if attribute not in own}
    for attribute, new in names.items():
        shown = public.get(name, {}).get(attribute, new)  # the module its warning names
        read(name, module, attribute, getattr(importlib.import_module(new), attribute), shown)
    # A split module's names map to the modules that define them, among its new modules (where a name set through it
    # is set); a package that re-exports one, which its warning names, exports the same object.
    for attribute, new in (names.items() if name in homes else ()):
        value = vars(importlib.import_module(new)).get(attribute, MISSING)
        if new not in homes[name] or value is MISSING:
            failures.append(f"{name}.{attribute}: {new} is not one of its new modules that holds it")
        elif inspect.isclass(value) or inspect.isfunction(value):
            if value.__name__ == attribute and value.__module__ != new:  # a definition (not an alias of one)
                failures.append(f"{name}.{attribute} maps to {new}, not to {value.__module__}, which defines it")
        shown = public.get(name, {}).get(attribute)
        if shown is not None and getattr(importlib.import_module(shown), attribute) is not value:
            failures.append(f"{name}.{attribute}: {shown} does not export it")
    # A split module's names that it only imported: each from the first of its new modules that has it.
    seen = set(names) | own
    for home in homes.get(name, []):
        for attribute, expected in list(vars(importlib.import_module(home)).items()):
            if not attribute.startswith("__") and attribute not in seen:
                seen.add(attribute)
                read(name, module, attribute, expected, home)
    if hasattr(module, "__path__"):
        continue
    # from <old module> import *: the new module's exports, or a split module's public names that moved.
    if isinstance(targets, str):
        target = importlib.import_module(targets)
        exported = getattr(target, "__all__", None)
        exported = [n for n in vars(target) if not n.startswith("_")] if exported is None else exported
        expected = {attribute: getattr(target, attribute) for attribute in exported}
    else:
        expected = {a: getattr(importlib.import_module(new), a) for a, new in targets.items() if not a.startswith("_")}
    star = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(f"from {name} import *", star)
    star.pop("__builtins__", None)
    if star.keys() != expected.keys() or any(star[a] is not value for a, value in expected.items()):
        failures.append(f"from {name} import * binds {sorted(star)}, not {sorted(expected)}")
    if module.__all__ is not module.__all__:
        failures.append(f"{name}.__all__ is a new list at each read")
print(json.dumps({"checked": checked, "failures": failures}))
"""


def test_each_moved_module_re_exports_the_same_objects_and_warns_once_per_name():
    """Every name of every shim is the moved object itself (class identity holds for pickles and ``isinstance``), and
    so is every name a split module only imported, read from the first of its new modules that has it; reading one
    warns once, naming its new module and the release that removes the old path; importing a shim warns nothing; and
    ``from <old module> import *`` binds what the new module exports (a split module's public names that moved)."""

    completed = subprocess.run(
        [sys.executable, "-c", _SHIM_CHECK, json.dumps(sorted(SHIMS)), json.dumps(HOMES), json.dumps(PUBLIC)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["checked"] > 100 and not report["failures"], "\n".join(report["failures"])


_PATCH_CHECK = """
import importlib, json, pickle, sys, warnings
from unittest import mock

import pytest

warnings.simplefilter("ignore")
import tinker

import platoon.train
import platoon.train.tinker.rl as rl
import platoon.train.tinker.workers.entrypoint as old_entrypoint
import platoon.train.tinker.workflows.group_rollout_workflow as old_group
from platoon.train.checkpoints import saving
from platoon.train.credit import group as credit
from platoon.train.trainer import rl as trainer
from platoon.train.trainer import service, signals, waiting
from platoon.train.workers import entrypoint, supervisor
from platoon.train.workflow import context, group

homes = json.loads(sys.argv[1])
failures = []


def check(what, ok):
    if not ok:
        failures.append(what)


original = waiting.ROLLOUT_HEARTBEAT_INTERVAL_S
with mock.patch("platoon.train.tinker.rl.ROLLOUT_HEARTBEAT_INTERVAL_S", 0.5):
    check("mock.patch through the old path reaches the moved code", waiting.ROLLOUT_HEARTBEAT_INTERVAL_S == 0.5)
    check("and reads back through the old path", rl.ROLLOUT_HEARTBEAT_INTERVAL_S == 0.5)
check("its undoing puts the original back", waiting.ROLLOUT_HEARTBEAT_INTERVAL_S is original)
check("and leaves nothing in the shim", "ROLLOUT_HEARTBEAT_INTERVAL_S" not in vars(rl))

users = [home for home in homes if vars(importlib.import_module(home)).get("tinker") is tinker]
fake = object()
with mock.patch.object(rl, "tinker", fake):
    check("a name the old module imported is patched in each new module of it that uses it",
          all(vars(sys.modules[home])["tinker"] is fake for home in users))
    check("and in no other module", sys.modules["platoon.train.step.batch_transforms"].tinker is tinker)
check("its undoing puts it back in each", all(vars(sys.modules[home])["tinker"] is tinker for home in users))

original = service.TRAINING_CLIENT_OPEN_TIMEOUT_S
rl.TRAINING_CLIENT_OPEN_TIMEOUT_S = 0.25
check("a name set through the old path is set where the code reads it", service.TRAINING_CLIENT_OPEN_TIMEOUT_S == 0.25)
del rl.TRAINING_CLIENT_OPEN_TIMEOUT_S
check("deleted through it, it is gone", not hasattr(service, "TRAINING_CLIENT_OPEN_TIMEOUT_S"))
rl.TRAINING_CLIENT_OPEN_TIMEOUT_S = original
check("set again, it is back", service.TRAINING_CLIENT_OPEN_TIMEOUT_S is original)

preload = supervisor.WORKER_PRELOAD
with mock.patch("platoon.train.tinker.workers.supervisor.WORKER_PRELOAD", ("x",)):
    check("a patch of a module that moved whole reaches it", supervisor.WORKER_PRELOAD == ("x",))
check("and its undoing", supervisor.WORKER_PRELOAD is preload)

flag = signals._SHUTDOWN_REQUESTED
signals._SHUTDOWN_REQUESTED = not flag
check("a global the moved code rebinds reads as it is now", rl._SHUTDOWN_REQUESTED is (not flag))
signals._SHUTDOWN_REQUESTED = flag
phase = entrypoint._PHASE
entrypoint._PHASE = "rollout"
check("so does one of a module that moved whole", old_entrypoint._PHASE == "rollout")
entrypoint._PHASE = phase

# A name that platoon.train re-exports is patched in the module that defines it and in those that import it from
# there, and its undoing puts the object back in each; platoon.train's own name stays the object.
watchdog = signals.Watchdog
with mock.patch("platoon.train.tinker.rl.Watchdog", fake):
    check("a patch of rl.Watchdog reaches the module that defines it, and those that use it",
          signals.Watchdog is fake and saving.Watchdog is fake and trainer.Watchdog is fake)
check("and its undoing puts it back in each", signals.Watchdog is watchdog and saving.Watchdog is watchdog)
check("and leaves platoon.train's own", platoon.train.Watchdog is watchdog and "Watchdog" not in vars(rl))
trainer_class = trainer.PlatoonTinkerRLTrainer
patcher = pytest.MonkeyPatch()
patcher.setattr(rl, "PlatoonTinkerRLTrainer", fake)
check("monkeypatch.setattr of rl.PlatoonTinkerRLTrainer reaches its module", trainer.PlatoonTinkerRLTrainer is fake)
patcher.undo()
check("and its undo puts the class back there, not in the shim",
      trainer.PlatoonTinkerRLTrainer is trainer_class and "PlatoonTinkerRLTrainer" not in vars(rl))
check("so that the class pickles by reference again", pickle.loads(pickle.dumps(trainer_class)) is trainer_class)
for name, module in (("GroupRolloutWorkflow", group), ("current_entry", context), ("group_baselines", credit)):
    original = vars(module)[name]
    with mock.patch(f"platoon.train.tinker.workflows.group_rollout_workflow.{name}", fake):
        check(f"a patch of group_rollout_workflow.{name} reaches {module.__name__}", vars(module)[name] is fake)
    check(f"and its undoing puts {name} back", vars(module)[name] is original and name not in vars(old_group))
    patcher = pytest.MonkeyPatch()
    patcher.delattr(old_group, name)
    check(f"group_rollout_workflow.{name} deleted is gone", not hasattr(old_group, name) and name not in vars(module))
    patcher.undo()
    check(f"and its undo puts {name} back", vars(module)[name] is original)

# The old module's logger is each new module's own: a patch of it reaches each, and its undoing puts back each one's.
MISSING = object()


def now(name, modules):
    return [vars(sys.modules[module]).get(name, MISSING) for module in modules]


loggers = {home: vars(sys.modules[home])["logger"] for home in homes if "logger" in vars(sys.modules[home])}
own = list(loggers.values())
with mock.patch("platoon.train.tinker.rl.logger", fake):
    check("a patch of rl.logger reaches each new module's logger", all(v is fake for v in now("logger", loggers)))
check("and its undoing puts back each module's own", all(a is b for a, b in zip(now("logger", loggers), own)))
patcher = pytest.MonkeyPatch()
patcher.delattr(rl, "logger")
check("rl.logger deleted is gone from each", not hasattr(rl, "logger") and set(now("logger", loggers)) == {MISSING})
patcher.undo()
check("and the undo puts back each module's own", all(a is b for a, b in zip(now("logger", loggers), own)))

# A builtin's name set on a split module shadows the builtin for its code, in each of its new modules, until undone.
with mock.patch("platoon.train.tinker.rl.print") as fake_print:
    check("a builtin patched through rl reaches each new module", all(v is fake_print for v in now("print", homes)))
check("and its undoing leaves it in none", set(now("print", homes)) == {MISSING} and not hasattr(rl, "print"))

# mock undoes a patch of a name that its target does not hold itself, as a shim holds none, by deleting the name, and
# without create=True (which it forces for a builtin's name) by setting back what it found: either way, the undoing
# puts back what the patch found, as in the old module, which held its names.  So does the ``mock`` package's (which
# pytest-mock can use): unittest.mock's code, loaded here as the module that package names it, ``mock.mock``.
def standalone_mock():
    import importlib.util, types

    sys.modules["mock"] = types.ModuleType("mock")
    sys.modules["mock"].__path__ = []
    spec = importlib.util.spec_from_file_location("mock.mock", mock.__file__)
    sys.modules["mock.mock"] = copy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(copy)
    return copy


old_package = sys.modules["platoon.train.tinker"]
for patching in (mock, standalone_mock()):
    by = f" ({patching.__name__})"
    original = service.TRAINING_CLIENT_OPEN_TIMEOUT_S
    with patching.patch("platoon.train.tinker.rl.TRAINING_CLIENT_OPEN_TIMEOUT_S", 0.5, create=True):
        check("mock.patch(create=True) through rl reaches it" + by, service.TRAINING_CLIENT_OPEN_TIMEOUT_S == 0.5)
    check("and its undoing puts the original back" + by, service.TRAINING_CLIENT_OPEN_TIMEOUT_S is original)
    with patching.patch("platoon.train.tinker.rl.TRAINING_CLIENT_OPEN_TIMEOUT_S", 0.5):
        del rl.TRAINING_CLIENT_OPEN_TIMEOUT_S
    check("so does one whose name was deleted meanwhile" + by, service.TRAINING_CLIENT_OPEN_TIMEOUT_S is original)
    with patching.patch("platoon.train.tinker.workers.supervisor.WORKER_PRELOAD", ("x",)):
        with patching.patch("platoon.train.tinker.workers.supervisor.WORKER_PRELOAD", ("y",), create=True):
            check("a patch with create=True inside another reaches the code" + by, supervisor.WORKER_PRELOAD == ("y",))
        check("its undoing puts back the outer patch" + by, supervisor.WORKER_PRELOAD == ("x",))
    check("and the outer one's undoing the original" + by, supervisor.WORKER_PRELOAD is preload)
    with patching.patch("platoon.train.tinker.rl.print") as outer:
        with patching.patch("platoon.train.tinker.rl.print") as inner:
            check("a builtin patched twice reaches each module" + by, all(v is inner for v in now("print", homes)))
        check("undoing the inner patch puts back the outer one" + by, all(v is outer for v in now("print", homes)))
    check("and undoing the outer one leaves it in none" + by, set(now("print", homes)) == {MISSING})
    with patching.patch("platoon.train.tinker.TrainConfig", "patched", create=True):
        pass
    check("and so does one of an old package's name" + by, old_package.TrainConfig is platoon.train.TrainConfig)
print(json.dumps(failures))
"""


def test_a_patch_through_a_moved_module_path_reaches_the_moved_code():
    """A name set or deleted through an old module path, as ``monkeypatch.setattr`` and ``mock.patch`` do, is set or
    deleted in each new module that holds it (in each that has one, for a module's own ``logger``; in each, for a
    builtin's name), and undone there: a name that ``platoon.train`` re-exports too, a patch made with
    ``create=True``, and one nested in another; a global the moved code rebinds reads live."""

    completed = subprocess.run(
        [sys.executable, "-c", _PATCH_CHECK, json.dumps(HOMES["platoon.train.tinker.rl"])],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    failures = json.loads(completed.stdout.strip().splitlines()[-1])
    assert not failures, "\n".join(failures)


_PACKAGE_CHECK = """
import inspect, json, pydoc, sys, types, warnings
from unittest import mock

import pytest

warnings.simplefilter("ignore")
import platoon.train.tinker as old

failures = []
for name in ("rl", "config_defs", "dispatcher", "workers", "workflows", "proxy", "key_refusal", "renderers"):
    if not isinstance(getattr(old, name, None), types.ModuleType):
        failures.append(f"platoon.train.tinker.{name} is not a module")
if not isinstance(getattr(old.workers, "protocol", None), types.ModuleType):
    failures.append("platoon.train.tinker.workers.protocol is not a module")
if not isinstance(getattr(old.workflows, "group_rollout_workflow", None), types.ModuleType):
    failures.append("platoon.train.tinker.workflows.group_rollout_workflow is not a module")
# Not its attributes, as 0.4.0's init did not import them: a module that does not exist, the removed restart wrapper
# (importing it raises ImportError), and the launch module that moved.
for name in ("no_such_module", "restart_wrapper", "train"):
    if hasattr(old, name) or name in dir(old):
        failures.append(f"platoon.train.tinker.{name} is an attribute")
try:
    inspect.getmembers(old)
    pydoc.render_doc(old)
except Exception as error:
    failures.append(f"introspecting platoon.train.tinker raised {error!r}")
import platoon.train

old.TrainConfig = "set"
if platoon.train.TrainConfig == "set":
    failures.append("a name set on the old package changed platoon.train's")
del old.TrainConfig
if hasattr(old, "TrainConfig"):
    failures.append("a name deleted from the old package is still there")
old.TrainConfig = platoon.train.TrainConfig
if "TrainConfig" in vars(old) or old.TrainConfig is not platoon.train.TrainConfig:
    failures.append("set back to the object it re-exports, the old package's name does not read from platoon.train")
patcher = pytest.MonkeyPatch()
patcher.delattr(old, "Watchdog")
if hasattr(old, "Watchdog"):
    failures.append("monkeypatch.delattr leaves the old package's name")
patcher.undo()
if old.Watchdog is not platoon.train.Watchdog or "Watchdog" in vars(old):
    failures.append("and its undo does not put the name back as it was")
with mock.patch("platoon.train.tinker.TrainConfig", "patched", create=True):
    if old.TrainConfig != "patched":
        failures.append("mock.patch(create=True) of the old package's name does not read as made")
if old.TrainConfig is not platoon.train.TrainConfig or "TrainConfig" in vars(old):
    failures.append("undoing mock.patch(create=True) does not put the old package's name back as it was")
shim = "platoon.train.tinker.workers.supervisor"
if shim in sys.modules:
    failures.append(f"{shim} was imported before the check, which needs a submodule not read yet")
with mock.patch(shim, "patched", create=True):  # which imports it, as reading it does
    pass
if old.workers.supervisor is not sys.modules[shim]:
    failures.append(f"undoing mock.patch(create=True) of {shim}, which the patch imported, does not leave it there")
print(json.dumps(failures))
"""


def test_an_old_packages_submodules_are_its_attributes_and_its_names_its_own():
    """After ``import platoon.train.tinker`` alone, the submodules that its init imported (and those of ``.workers``
    and ``.workflows``) are its attributes, as before platoon 0.5.0, and no other is, the removed restart wrapper
    among them, so the package's ``dir()``, ``inspect.getmembers()`` and ``help()`` work; a name set on the package
    changes the package only, and one deleted from it is gone until it is set again; and undoing a ``mock.patch``
    with ``create=True`` of a name or of a submodule, which the patch imports, puts it back as it was."""

    completed = subprocess.run([sys.executable, "-c", _PACKAGE_CHECK], capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr[-4000:]
    failures = json.loads(completed.stdout.strip().splitlines()[-1])
    assert not failures, "\n".join(failures)


_SWEEP_CHECK = """
import builtins, contextlib, importlib, json, sys, types, warnings
from unittest import mock

import pytest

warnings.simplefilter("ignore")
shims, homes = json.loads(sys.argv[1]), json.loads(sys.argv[2])
MISSING, DELETE = object(), object()
# The ways a test patches a module's name, alone and nested (the outer patch first); and those of a builtin's name,
# which mock.patch makes with create=True.
PLANS = (
    ("mock.patch",),
    ("mock.patch(create=True)",),
    ("mock.patch.object",),
    ("monkeypatch.setattr",),
    ("monkeypatch.delattr",),
    ("mock.patch", "mock.patch(create=True)"),
    ("mock.patch(create=True)", "mock.patch"),
    ("monkeypatch.setattr", "mock.patch(create=True)"),
    ("mock.patch(create=True)", "monkeypatch.delattr"),
)
BUILTIN_PLANS = (("mock.patch",), ("mock.patch", "mock.patch"))
BUILTINS = {name for name in dir(builtins) if not name.startswith("_")}
failures = []
checked = 0
listed = {}


def modules():
    # Every platoon module, listed again once a module is imported.
    if listed.get("count") != len(sys.modules):
        found = list(sys.modules.items())
        platoon = [(n, m) for n, m in found if n.split(".")[0] == "platoon" and isinstance(m, types.ModuleType)]
        listed.update(count=len(found), modules=platoon)
    return listed["modules"]


def bindings(attribute):
    return {n: vars(m).get(attribute, MISSING) for n, m in modules()}


def namespaces():
    return {n: dict(vars(m)) for n, m in modules()}


def read_by(places):
    # The names that the code of these modules reads (and those of the attributes it reads).
    names = set()
    for place in places:
        path = importlib.import_module(place).__file__
        codes = [compile(open(path).read(), path, "exec")]
        while codes:
            code = codes.pop()
            names.update(code.co_names)
            codes.extend(c for c in code.co_consts if isinstance(c, types.CodeType))
    return names


class Fake:
    pass


@contextlib.contextmanager
def monkeypatched(module, attribute, value):
    patcher = pytest.MonkeyPatch()
    try:
        if value is DELETE:
            patcher.delattr(module, attribute)
        else:
            patcher.setattr(module, attribute, value)
        yield
    finally:
        patcher.undo()


def patching(way, name, module, attribute, value):
    if way == "mock.patch":
        return mock.patch(f"{name}.{attribute}", value)
    if way == "mock.patch(create=True)":
        return mock.patch(f"{name}.{attribute}", value, create=True)
    if way == "mock.patch.object":
        return mock.patch.object(module, attribute, value)
    return monkeypatched(module, attribute, value)


def sweep(name, module, attribute, original, plan, expected):
    # Patch the name in each way of the plan, nested; through each patch, and through each outer one again once the
    # inner one is undone, the name reads as patched through the old path and the patch is in the modules expected
    # (a delete: gone from those that had it); once all are undone, every platoon module holds what it held.
    global checked
    checked += 1
    where = f"{name}.{attribute} [{' > '.join(plan)}]"
    before = bindings(attribute)
    had = [n for n, v in before.items() if v is not MISSING]
    values = [DELETE if way == "monkeypatch.delattr" else Fake() for way in plan]

    def observe(value, point):
        now = bindings(attribute)
        if value is DELETE:
            read = not hasattr(module, attribute)
            reached, want = sorted(n for n in had if now.get(n, MISSING) is MISSING), sorted(set(expected) & set(had))
        else:
            read = getattr(module, attribute, None) is value
            reached, want = sorted(n for n, v in now.items() if v is value), sorted(expected)
        if not read:
            failures.append(f"{where}, {point}: through the old path, the patch does not read as made")
        if reached != want:
            failures.append(f"{where}, {point}: reached {reached}, not {want}")

    def nest(depth):
        with patching(plan[depth], name, module, attribute, values[depth]):
            observe(values[depth], f"in {plan[depth]}")
            if depth + 1 < len(plan):
                nest(depth + 1)
                observe(values[depth], f"in {plan[depth]} after the inner undoing")

    try:
        nest(0)
    except Exception as error:
        failures.append(f"{where}: raised {error!r}")
    after = bindings(attribute)
    changed = sorted(n for n, v in before.items() if after.get(n, MISSING) is not v)
    if changed:
        failures.append(f"{where}: after its undoing, {changed} differ")
        for n in changed:  # as it was, for the checks that follow
            if before[n] is MISSING:
                vars(sys.modules[n]).pop(attribute, None)
            else:
                vars(sys.modules[n])[attribute] = before[n]
    if getattr(module, attribute, MISSING) is not original:
        failures.append(f"{where}: after its undoing, the old path does not read the original")


for name in shims:
    module = importlib.import_module(name)
    targets = module.__moved__
    places = [targets] if isinstance(targets, str) else homes.get(name, [])
    package = hasattr(module, "__path__")
    originals = {}
    for attribute in dir(module):
        if not (attribute.startswith("__") and attribute.endswith("__")):
            originals[attribute] = getattr(module, attribute)  # importing what reading it imports, first
    snapshot = namespaces()
    for attribute, original in originals.items():
        before = bindings(attribute)
        if package or attribute in vars(module):  # an old package's names are its own, and so are a shim's own
            expected = [name]
        elif attribute == "logger":  # each new module's own
            expected = [m for m in places if before.get(m, MISSING) is not MISSING]
        else:  # where the old module's code that read it is now: each new module that holds its object
            expected = [m for m in places if before.get(m, MISSING) is original]
        for plan in PLANS:
            sweep(name, module, attribute, original, plan, expected)
    # The builtins' names that the moved code reads (two for a package, whose own code reads none): a patch of one is
    # in each new module, as it was in the old module, or kept by the package.
    names = sorted((read_by(places) & BUILTINS) - set(originals)) if places else ["len", "print"]
    for attribute in names:
        for plan in BUILTIN_PLANS:
            sweep(name, module, attribute, MISSING, plan, [name] if package else places)
    for n, old in snapshot.items():
        new = vars(sys.modules[n]) if n in sys.modules else {}
        for key in set(old) | set(new):
            if key != "__warningregistry__" and old.get(key, MISSING) is not new.get(key, MISSING):
                failures.append(f"after the patches of {name}'s names, {n}.{key} differs")
print(json.dumps({"checked": checked, "failures": failures[:200]}))
"""


def test_a_patch_through_a_moved_module_path_and_its_undoing_leave_every_module_as_platoon_0_4_0_did():
    """Every name of every shim, patched through the old path as tests patch a module's name (``mock.patch``, also
    with ``create=True``, ``mock.patch.object``, ``monkeypatch.setattr`` and ``monkeypatch.delattr``, alone and nested
    in one another) and undone, behaves as the old module's did in platoon 0.4.0, where the old modules were the
    modules themselves: the patch raises nothing, reads as made through the old path, and reaches where the old
    module's code that read the name is now (each new module that holds its object; each that has one, for a module's
    own ``logger``; an old package's names, and a shim's own, stay theirs), and so does an outer patch again once the
    inner one is undone; and its undoing leaves every module of platoon as it was, name by name and object by object.
    So does a patch of a builtin's name that the moved code reads, alone and nested in another (``mock.patch`` makes
    it with ``create=True``): it is in each new module until undone."""

    completed = subprocess.run(
        [sys.executable, "-c", _SWEEP_CHECK, json.dumps(sorted(SHIMS)), json.dumps(HOMES)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["checked"] > 10000 and not report["failures"], "\n".join(report["failures"])


_COLD_CHECK = """
import importlib, json, os, pkgutil, shutil, signal, sys, tempfile, types, warnings
from unittest import mock

import pytest

warnings.simplefilter("ignore")
cases = json.loads(sys.argv[1])
# Each case runs in a fork of this process, which imports here what platoon imports from elsewhere, and nothing of
# platoon: in each, as in a test that starts by importing an old path, the old path is the first of platoon imported.
for name in ("torch", "tinker", "tinker_cookbook", "datasets", "transformers"):
    importlib.import_module(name)
assert not any(name.split(".")[0] == "platoon" for name in sys.modules)


class Fake:
    pass


def patch(path, name, way, value):
    if way == "monkeypatch.setattr":
        patcher = pytest.MonkeyPatch()
        patcher.setattr(importlib.import_module(path), name, value)
        return patcher.undo
    patcher = mock.patch(f"{path}.{name}", value, create=way == "mock.patch(create=True)")
    patcher.start()
    return patcher.stop


def import_everything():
    import platoon.train

    failed = []
    for info in pkgutil.walk_packages(platoon.train.__path__, "platoon.train.", onerror=lambda name: None):
        try:
            importlib.import_module(info.name)
        except Exception as error:
            if not info.name.endswith(".restart_wrapper"):  # which raises ImportError, as platoon 0.3.0 removed it
                failed.append(f"importing {info.name} raised {error!r}")
    return failed


def holding(value):
    return sorted(
        f"{name}.{key}"
        for name, module in list(sys.modules.items())
        if name.split(".")[0] == "platoon" and isinstance(module, types.ModuleType)
        for key, held in list(vars(module).items())
        if held is value
    )


def imported(path):
    # Importing an old path imports the trainer and platoon train's start, as the old package's init did, before any
    # of its names is read; platoon.train.tinker itself, whose policy client the new code imports, no longer does.
    module = importlib.import_module(path)
    if path == "platoon.train.tinker":
        return module, []
    trainer = ("platoon.train.trainer.rl", "platoon.train.launch")
    return module, [f"importing {path} did not import {name}" for name in trainer if name not in sys.modules]


def patched(path, name, way):
    module, problems = imported(path)
    original, fake = getattr(module, name), Fake()
    return module, original, fake, patch(path, name, way, fake), problems


def importing(path, name, way):
    # The name patched right after the old path's import, the rest of platoon imported while the patch is active.
    module, original, fake, undo, problems = patched(path, name, way)
    problems += import_everything()
    undo()
    problems += [f"{held} keeps the patch after its undoing" for held in holding(fake)]
    if getattr(module, name) is not original:
        problems.append("the old path does not read the original after the undoing")
    return [f"{path}.{name} [{way}]: {problem}" for problem in problems]


def reading(package, new, names):
    # Each name of an old package, read through it for the first time while the old module that defines it is patched.
    problems = []
    for path, name, way in names:
        module, original, fake, undo, found = patched(path, name, way)
        problems += found
        if getattr(importlib.import_module(package), name) is not original:
            problems.append(f"{package}.{name} [{way}], first read while {path}.{name} is patched, reads the patch")
        undo()
        for where in (package, new, path):
            if getattr(importlib.import_module(where), name) is not original:
                problems.append(f"{where}.{name} [{way}] keeps the patch after its undoing")
    return problems


results = tempfile.mkdtemp()
running = set()
for index, case in enumerate(cases):
    if len(running) >= min(8, max(2, os.cpu_count() or 2)):
        running.discard(os.wait()[0])
    pid = os.fork()
    if pid == 0:
        signal.alarm(300)
        try:
            found = {"importing": importing, "reading": reading}[case[0]](*case[1:])
        except BaseException as error:
            found = [f"{case}: raised {error!r}"]
        with open(os.path.join(results, f"{index}.json"), "w") as out:
            json.dump(found, out)
        os._exit(0)
    running.add(pid)
while running:
    running.discard(os.wait()[0])
failures = []
for index, case in enumerate(cases):
    try:
        with open(os.path.join(results, f"{index}.json")) as found:
            failures += json.load(found)
    except OSError:
        failures.append(f"{case}: no result")
shutil.rmtree(results)
print(json.dumps({"checked": len(cases), "failures": failures}))
"""
# For each old module path, a name that a module of the trainer imports from where a patch through it puts it.
_IMPORTED_WHILE_PATCHED = {
    "platoon.train.tinker": "PlatoonTinkerRLTrainer",
    "platoon.train.tinker.batch_transforms": "BatchTransformContext",
    "platoon.train.tinker.censoring": "Censoring",
    "platoon.train.tinker.checkpoint_pruning": "RecoveryCheckpointPruner",
    "platoon.train.tinker.config_defs": "WorkflowConfig",
    "platoon.train.tinker.dataset_order": "EpochOrder",
    "platoon.train.tinker.dispatcher": "Dispatcher",
    "platoon.train.tinker.evaluation_record": "record_evaluation",
    "platoon.train.tinker.kept_groups": "read_kept",
    "platoon.train.tinker.permits": "DrawLease",
    "platoon.train.tinker.rl": "TrainLoopSharedState",
    "platoon.train.tinker.staleness": "StaleWatch",
    "platoon.train.tinker.start_limit": "StartLimit",
    "platoon.train.tinker.task_records": "TaskRecorder",
    "platoon.train.tinker.train": "PlatoonTinkerRLTrainer",
    "platoon.train.tinker.worker_env": "NO_CUDA_DRIVER",
    "platoon.train.tinker.workers": "RolloutWorkerRequest",
    "platoon.train.tinker.workers.binding": "publish_binding",
    "platoon.train.tinker.workers.entrypoint": "main",
    "platoon.train.tinker.workers.protocol": "RolloutWorkerResponse",
    "platoon.train.tinker.workers.registry": "WorkerRecord",
    "platoon.train.tinker.workers.run": "run_rollout_request",
    "platoon.train.tinker.workers.supervisor": "RolloutWorkerSupervisor",
    "platoon.train.tinker.workers.telemetry": "record_worker_cell_guard",
    "platoon.train.tinker.workers.workload": "add_datum_funnel",
    "platoon.train.tinker.workflows": "GroupRolloutWorkflow",
    "platoon.train.tinker.workflows.base": "RolloutWorkflow",
    "platoon.train.tinker.workflows.group_rollout_workflow": "GroupRolloutWorkflow",
}
# Each name that an old package exports, with the old module that defines it; and the new package that exports it.
_EXPORTED = {
    ("platoon.train.tinker", "platoon.train"): {
        "PlatoonTinkerRLTrainer": "rl",
        "Watchdog": "rl",
        "BatchTransform": "batch_transforms",
        "BatchTransformContext": "batch_transforms",
        "DepthLevelWeightingTransform": "batch_transforms",
        "build_default_batch_transforms": "batch_transforms",
        "PlatoonTinkerRLTrainerConfig": "config_defs",
        "TrainConfig": "config_defs",
        "EvalConfig": "config_defs",
        "CheckpointConfig": "config_defs",
        "StatsConfig": "config_defs",
        "WatchdogConfig": "config_defs",
        "AdamParams": "config_defs",
        "WorkflowConfig": "config_defs",
    },
    ("platoon.train.tinker.workflows", "platoon.train.workflow"): {
        "RolloutWorkflow": "base",
        "GroupRolloutWorkflow": "group_rollout_workflow",
    },
    ("platoon.train.tinker.workers", "platoon.train.workers"): {
        "PROTOCOL_VERSION": "protocol",
        "ConversionPolicy": "protocol",
        "ModelBinding": "protocol",
        "RolloutWorkerRequest": "protocol",
        "RolloutWorkerResponse": "protocol",
        "callable_ref": "protocol",
        "resolve_ref": "protocol",
        "validated_ref": "protocol",
        "add_datum_funnel": "workload",
        "workload_from_raw_rollout": "workload",
    },
}


def test_a_patch_made_through_an_old_path_right_after_its_import_is_what_platoon_0_4_0_kept():
    """Importing an old module path imports the trainer, as it did in platoon 0.4.0: its modules and
    ``platoon.train.launch``, before any of its names is read (``platoon.train.tinker`` itself no longer does).  So a
    patch made through it right after its import, with nothing else of platoon imported, reaches no module that is
    first imported while it is active: importing the rest of platoon meanwhile raises nothing, and the patch's undoing
    leaves its object nowhere.
    A name of an old package, read through it for the first time while the old module that defines it is patched,
    reads the original, as it did then, and so do the old package and the new one that exports the name once the
    patch is undone.  Each old path, and each old package, in a process of its own; ``mock.patch``, also with
    ``create=True``, and ``monkeypatch.setattr``."""

    assert set(_IMPORTED_WHILE_PATCHED) == set(SHIMS)
    for (package, _), names in _EXPORTED.items():
        assert set(names) == SHIMS[package], package
    ways = ("mock.patch", "mock.patch(create=True)", "monkeypatch.setattr")
    cases: list[tuple] = [
        ("importing", path, name, ways[index % len(ways)])
        for index, (path, name) in enumerate(sorted(_IMPORTED_WHILE_PATCHED.items()))
    ]
    for (package, new), names in _EXPORTED.items():  # one process per old package, its names one after the other
        named = [(f"{package}.{old}", name, ways[index % len(ways)]) for index, (name, old) in enumerate(names.items())]
        cases.append(("reading", package, new, named))
    completed = subprocess.run(
        [sys.executable, "-c", _COLD_CHECK, json.dumps(cases)], capture_output=True, text=True, timeout=600
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["checked"] == len(cases) and not report["failures"], "\n".join(report["failures"])


def test_a_moved_name_warns_where_it_is_imported(tmp_path):
    """The warning names the importing file, not platoon's own frames, and says what to import instead."""

    script = tmp_path / "old_import.py"
    script.write_text("from platoon.train.tinker.rl import PlatoonTinkerRLTrainer  # noqa: F401\n")
    completed = subprocess.run(
        [sys.executable, "-W", "always::FutureWarning", str(script)], capture_output=True, text=True, timeout=600
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert f"{script}:1: PlatoonDeprecationWarning" in completed.stderr
    assert (
        "import it from platoon.train (platoon 0.6.0 removes platoon.train.tinker.rl.PlatoonTinkerRLTrainer)"
        in completed.stderr
    )


@pytest.mark.parametrize(
    ("orig_argv", "expected"),
    [
        (["python", "-m", "platoon.train.workers.entrypoint", "request"], True),
        (["python", "-m", "platoon.train.tinker.workers.entrypoint", "request"], True),
        (["python", "-m", "platoon.train.supervise", "config.yaml"], False),
    ],
)
def test_a_worker_is_recognized_by_either_entrypoint_name(monkeypatch, orig_argv, expected):
    from platoon.train.workers import env

    monkeypatch.delenv(env.WORKER_ENV_MARKER, raising=False)
    monkeypatch.setattr(sys, "orig_argv", orig_argv)
    monkeypatch.setattr(sys, "argv", ["-m"])
    assert env.in_rollout_worker_process() is expected


def test_the_old_entrypoint_name_still_runs_a_worker():
    """``python -m platoon.train.tinker.workers.entrypoint`` runs the new ``main`` (here: its usage line, for a call
    without a request), with a warning that names the new module."""

    environment = {key: value for key, value in os.environ.items() if key != "PLATOON_TINKER_ROLLOUT_WORKER"}
    completed = subprocess.run(
        [sys.executable, "-m", "platoon.train.tinker.workers.entrypoint"],
        capture_output=True,
        text=True,
        timeout=600,
        env=environment,
    )
    assert completed.returncode == 2, completed.stderr[-4000:]
    assert "usage: python -m platoon.train.workers.entrypoint <request_path>" in completed.stderr
    assert "run python -m platoon.train.workers.entrypoint (platoon 0.6.0 removes" in completed.stderr


def test_a_supervisor_given_the_old_entrypoint_name_runs_the_entrypoint(tmp_path):
    """A supervisor built with the entrypoint's module before platoon 0.5.0 runs the entrypoint, as one built with the
    default does, so that its fork server preloads the worker's stack for it as before."""

    from platoon.train.workers.supervisor import WORKER_MODULE, RolloutWorkerSupervisor

    supervisor = RolloutWorkerSupervisor(
        module="platoon.train.tinker.workers.entrypoint", scratch_dir=str(tmp_path), start_method="forkserver"
    )
    assert supervisor.module == WORKER_MODULE
    assert RolloutWorkerSupervisor(module="my.worker", scratch_dir=str(tmp_path)).module == "my.worker"
