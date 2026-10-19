"""Import boundaries, checked on the source with ``ast``.

* Core (the ``platoon`` package) never imports the extensions that plug into it: the first-party extensions in
  ``extensions/``, and the packages of every installed extension (a distribution that declares platoon entry points).
  Neither an import statement, nor a constant passed to ``importlib.import_module`` or ``__import__``, nor any other
  string constant that reads as an import string (``pkg.mod:attr``, ``python://pkg.mod:Class``, or a dotted module
  name, as a table of modules that code imports later holds) may name one; docstrings are prose and exempt.
  ``test_core_boundary.py`` checks at run time that importing core's entry points loads no extension.
* The first-party extensions' code (``extensions/*/src``) and the examples import only platoon's public names: those
  the public modules export (``scripts/public_api.py``: ``PUBLIC_MODULES``), and an extension's own package.  An
  attribute of a name bound to a public module counts as well, however the module was bound (``import platoon as
  pt``, ``import platoon.gym as g``, ``from platoon import testing``).  :data:`PRIVATE_IMPORTS` lists the private
  names they import today, each with why.  A private import it does not list fails the test, and so does a listed
  one that no longer happens, so the list stays what the public-surface work has to resolve.
"""

from __future__ import annotations

import ast
import importlib.metadata
import importlib.util
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
CORE = REPO / "platoon"
EXTENSIONS = REPO / "extensions"
EXAMPLES = REPO / "examples"

TEXTCRAFT = "extensions/textcraft/src/platoon_textcraft/"
EXTENSION_PACKAGE = "examples/extension-package/"

# Private names the first-party extensions and the examples import, ``module:name`` (or a module, for ``import
# module``), with the files that import each.  Making them public, or replacing them, is the public-surface work of
# RFC 0013 (section 12) and RFC 0017; until then a new one needs a reason here.
PRIVATE_IMPORTS: dict[str, tuple[str, ...]] = {
    # The CodeAct prompt-builder API, which the built-in harness guide teaches (CodeAct prompts).
    "platoon.harness.builtin.codeact:CellFunction": (TEXTCRAFT + "prompts.py",),
    "platoon.harness.builtin.codeact:CodeActPromptBuilder": (TEXTCRAFT + "prompts.py",),
    "platoon.harness.builtin.codeact:CodeActState": (TEXTCRAFT + "prompts.py",),
    "platoon.harness.builtin.codeact:PromptMode": (TEXTCRAFT + "prompts.py",),
    # Helpers of function harnesses beside the agent contract: the step-limit message, the out-of-steps check, a
    # terminal tool's arguments, and ReAct's tool schema.
    "platoon.harness.agent:LIMIT_MESSAGE": (
        EXTENSION_PACKAGE + "src/example_extension/harness.py",
        "examples/function-harness/tool_loop.py",
        "examples/subagents/delegator.py",
    ),
    "platoon.harness.agent:out_of_steps": ("examples/subagents/delegator.py",),
    "platoon.harness.agent:terminal_arguments": ("examples/custom-environment/packing_team.py",),
    "platoon.harness.builtin.react:openai_tool": ("examples/subagents/delegator.py",),
    # Environment services: the service an environments entry names, and the task-table service.
    "platoon.envs:TaskTableService": (EXTENSION_PACKAGE + "src/example_extension/tasks.py",),
    "platoon.envs:get_service": ("examples/custom-environment/check_env.py",),
    # Whether a trajectory was interrupted, for reward shaping in an episode program.
    "platoon.utils.trajectory_status:trajectory_was_interrupted": (
        "examples/program-reward-shaping/efficient_search.py",
    ),
    # The command line, which the examples' own tests run in process.
    "platoon.cli:main": (EXTENSION_PACKAGE + "tests/test_example_extension.py", "examples/quickstart/test_guess.py"),
}


@pytest.fixture(scope="module")
def api() -> ModuleType:
    """``scripts/public_api.py``, which says what is public."""

    spec = importlib.util.spec_from_file_location("public_api", REPO / "scripts" / "public_api.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("public_api", module)
    spec.loader.exec_module(module)
    return module


# --- reading imports --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Import:
    """One import in a file: ``import module`` (``name`` None) or ``from module import name``, made absolute."""

    line: int
    module: str
    name: str | None


def python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def module_name(path: Path, root: Path) -> str:
    """The dotted name of the module at ``path`` under the import root ``root``."""

    parts = path.relative_to(root).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imports(path: Path, root: Path) -> Iterator[Import]:
    """The imports in ``path``, which imports from ``root``: statements anywhere in the file, relative ones made
    absolute, and ``importlib.import_module`` or ``__import__`` calls with a constant name."""

    module = module_name(path, root)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield Import(node.lineno, alias.name, None)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parents = package.split(".")[: len(package.split(".")) - node.level + 1] if package else []
                base = ".".join([*parents, *filter(None, [node.module])])
            for alias in node.names:
                yield Import(node.lineno, base, alias.name)
        elif isinstance(node, ast.Call) and _imports_by_name(node.func) and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                yield Import(node.lineno, first.value, None)


def _imports_by_name(func: ast.expr) -> bool:
    if isinstance(func, ast.Attribute):
        return func.attr == "import_module"
    return isinstance(func, ast.Name) and func.id in ("import_module", "__import__")


# A string that reads as an import string: an optional scheme (``python://``), a module, and an optional ``:attr``.
IMPORT_STRING = re.compile(r"(?:[a-z][a-z0-9+.-]*://)?(?P<module>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)(?::[A-Za-z_][\w.]*)?")


def import_strings(path: Path) -> Iterator[Import]:
    """The string constants in ``path`` that read as import strings, as an import of their module: ``pkg.mod:attr``,
    ``python://pkg.mod:Class`` and dotted module names (``pkg.mod``), which code hands to ``import_from_string``, a
    service or a table of modules that it imports later.  A single word without ``:`` is not one, and docstrings
    are prose."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            match = IMPORT_STRING.fullmatch(node.value.strip())
            if match and ("." in match["module"] or ":" in node.value):
                yield Import(node.lineno, match["module"], None)


def within(module: str, package: str) -> bool:
    return module == package or module.startswith(package + ".")


def first_party_packages() -> dict[str, Path]:
    """The packages the first-party extensions in ``extensions/`` provide, by module name, with the extension's import
    root: the top-level packages in its ``src/``."""

    found: dict[str, Path] = {}
    for root in sorted(EXTENSIONS.glob("*/src")):
        for init in sorted(root.glob("*/__init__.py")):
            found[init.parent.name] = root
    return found


# --- core ---------------------------------------------------------------------------------------------------------


def installed_extension_packages() -> dict[str, str]:
    """The packages of the installed extensions, by module name, with the distribution of each: the package of the
    module that each platoon entry point of a distribution other than platoon names (its top level, or
    ``platoon.<name>`` for a package that an old-style plugin placed inside platoon), core's own modules left out."""

    from platoon import extensions

    found: dict[str, str] = {}
    for group in (*extensions.GROUPS.values(), *extensions.RETIRED_GROUPS):
        for entry_point in importlib.metadata.entry_points(group=group):
            distribution = extensions.distribution_name(entry_point.dist) or "(no distribution)"
            if extensions.canonical_name(distribution) == extensions.RESERVED_OWNER:
                continue
            parts = entry_point.value.partition(":")[0].strip().split(".")
            package = ".".join(parts[:2] if parts[0] == extensions.RESERVED_OWNER else parts[:1])
            if package != extensions.RESERVED_OWNER and not extensions.is_core_module(package):
                found[package] = distribution
    return found


def test_core_never_imports_the_extensions_that_plug_into_it() -> None:
    first_party = first_party_packages()
    assert first_party, "no packages found under extensions/*/src: point first_party_packages at the extensions"
    never = {package: str(root.relative_to(REPO)) for package, root in first_party.items()}
    never |= installed_extension_packages()

    found = []
    for path in python_files(CORE):
        for how, items in (("imports", imports(path, REPO)), ("names in an import string", import_strings(path))):
            for item in items:
                targets = [item.module, *([f"{item.module}.{item.name}"] if item.name else [])]
                hits = sorted({package for package in never for target in targets if within(target, package)})
                found += [f"{path.relative_to(REPO)}:{item.line}: {how} {hit} ({never[hit]})" for hit in hits]

    assert not found, "core imports what plugs into it:\n" + "\n".join(found)


def test_the_core_check_sees_every_kind_of_import(tmp_path: Path) -> None:
    package = tmp_path / "platoon" / "envs"
    package.mkdir(parents=True)
    probe = package / "probe.py"
    probe.write_text(
        "import importlib\n"
        "import platoon.textcraft.env\n"
        "from platoon import number_search\n"
        "from .. import textcraft\n"
        "def later():\n"
        "    from acme_platoon.agents import agent\n"
        "    return importlib.import_module('platoon.number_search.prompts')\n"
    )

    found = [(item.module, item.name) for item in imports(probe, tmp_path)]

    assert found == [
        ("importlib", None),
        ("platoon.textcraft.env", None),
        ("platoon", "number_search"),
        ("platoon", "textcraft"),
        ("acme_platoon.agents", "agent"),
        ("platoon.number_search.prompts", None),
    ]


def test_the_core_check_sees_import_strings_but_not_prose(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(
        '"""A module docstring may name platoon.textcraft.env."""\n'
        "from importlib import import_module as load\n"
        "from platoon.registry import import_from_string\n"
        "SERVICE = 'python://platoon.number_search.env:NumberSearchEnvironment'\n"
        "MODULES = ('platoon.harness.builtin.react', 'platoon.textcraft.prompts')\n"
        "def later():\n"
        '    """Nor may a function docstring: platoon.textcraft.rewards:delegation_capped."""\n'
        "    load('platoon.textcraft')\n"
        "    return import_from_string('acme_platoon:agent')\n"
        "NOT_ONE = ('textcraft', 'see platoon.textcraft', 'platoon.textcraft.', 'platoon/textcraft/env.py')\n"
    )

    found = sorted(item.module for item in import_strings(probe))

    assert found == [
        "acme_platoon",
        "platoon.harness.builtin.react",
        "platoon.number_search.env",
        "platoon.textcraft",
        "platoon.textcraft.prompts",
    ]


# --- extensions and examples ------------------------------------------------------------------------------------


def extension_files() -> Iterator[tuple[Path, Path, tuple[str, ...]]]:
    """``(file, its import root, the packages it may import freely)`` of every file of a first-party extension's code
    and of every example file: an extension's own packages, and nothing for an example (which imports its own
    modules under names outside platoon)."""

    packages = first_party_packages()
    for root in sorted(EXTENSIONS.glob("*/src")):
        own = tuple(package for package, where in packages.items() if where == root)
        for path in python_files(root):
            yield path, root, own
    for path in python_files(EXAMPLES):
        example = EXAMPLES / path.relative_to(EXAMPLES).parts[0]
        root = example / "src" if path.is_relative_to(example / "src") else example
        yield path, root, ()


def private_imports(path: Path, root: Path, own: tuple[str, ...], api: ModuleType) -> Iterator[str]:
    """The private platoon names ``path`` uses: imports of names the public modules do not export, imports of
    modules that are not public, and attributes of ``platoon`` (``pt.x``) that are not public names."""

    public = {module: set(api.public_names(module)) for module in api.PUBLIC_MODULES}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for item in imports(path, root):
        if not within(item.module, "platoon") or any(within(item.module, package) for package in own):
            continue
        if item.name is None:
            if item.module not in public:
                yield item.module
        elif item.module not in public or (
            item.name not in public[item.module] and f"{item.module}.{item.name}" not in public
        ):
            yield f"{item.module}:{item.name}"
    yield from _private_attributes(tree, public, api)


def module_bindings(tree: ast.Module, modules: set[str]) -> dict[str, str]:
    """The names a file binds to the ``platoon`` package or to one of ``modules`` (the public ones), with the module
    each is bound to: ``import platoon [as pt]`` and ``import platoon.x`` (which binds ``platoon``),
    ``import platoon.x as y``, and ``from platoon[.x] import y [as z]`` where ``y`` is such a module."""

    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname is None and within(alias.name, "platoon"):
                    bound["platoon"] = "platoon"
                elif alias.asname is not None and (alias.name == "platoon" or alias.name in modules):
                    bound[alias.asname] = alias.name
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module and within(node.module, "platoon"):
            for alias in node.names:
                if f"{node.module}.{alias.name}" in modules:
                    bound[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return bound


def _private_attributes(tree: ast.Module, public: dict[str, set[str]], api: ModuleType) -> Iterator[str]:
    """``pt.x.y`` chains on a name bound to the ``platoon`` package or to a public module (``g.x`` after
    ``from platoon import gym as g``) that leave the public names."""

    bindings = module_bindings(tree, set(public) | set(api.CALLABLE_MODULES))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        chain: list[str] = []
        current: ast.expr = node
        while isinstance(current, ast.Attribute):
            chain.insert(0, current.attr)
            current = current.value
        if not chain or not (isinstance(current, ast.Name) and current.id in bindings):
            continue
        module = bindings[current.id]
        if module in api.CALLABLE_MODULES:  # bound to pt.harness, which is public only as the decorator
            yield f"{module}:{chain[0]}"
            continue
        for index, attribute in enumerate(chain):
            target = f"{module}.{attribute}"
            if target in public:  # a public module: its own names follow
                module = target
                continue
            if attribute not in public[module]:
                yield f"{module}:{attribute}"
            elif target in api.CALLABLE_MODULES and index + 1 < len(chain):  # pt.harness is only the decorator
                yield f"{target}:{chain[index + 1]}"
            break


def test_extensions_and_examples_import_only_platoons_public_names(api: ModuleType) -> None:
    found: dict[str, set[str]] = {}
    for path, root, own in extension_files():
        for name in private_imports(path, root, own, api):
            found.setdefault(name, set()).add(str(path.relative_to(REPO)))
    listed = {name: set(files) for name, files in PRIVATE_IMPORTS.items()}

    new = {name: sorted(files - listed.get(name, set())) for name, files in found.items()}
    gone = {name: sorted(files - found.get(name, set())) for name, files in listed.items()}
    new = {name: files for name, files in sorted(new.items()) if files}
    gone = {name: files for name, files in sorted(gone.items()) if files}

    assert not new, (
        f"extensions or examples import private platoon names: {new}. Import public names instead (import platoon as "
        "pt; platoon.testing); a name that is not public yet goes in PRIVATE_IMPORTS, with why"
    )
    assert not gone, f"PRIVATE_IMPORTS lists imports that no longer happen; remove them: {gone}"


def test_the_public_name_check_flags_private_modules_names_and_attributes(api: ModuleType, tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import platoon as pt\n"
        "import platoon.gym\n"
        "from platoon import Task, gym, testing\n"
        "from platoon.testing import run_episode\n"
        "from platoon.gym import loop, nothing_public\n"
        "from platoon.harness.agent import LIMIT_MESSAGE\n"
        "import platoon.utils.trajectory_status\n"
        "pt.Outcome.infra, pt.gym.loop, platoon.gym.as_environment\n"
        "pt.utils, pt.harness.builtin, pt.gym.not_exported\n"
    )

    assert sorted(set(private_imports(probe, tmp_path, (), api))) == [
        "platoon.gym:not_exported",
        "platoon.gym:nothing_public",
        "platoon.harness.agent:LIMIT_MESSAGE",
        "platoon.harness:builtin",
        "platoon.utils.trajectory_status",
        "platoon:utils",
    ]


def test_the_public_name_check_follows_every_name_bound_to_a_public_module(api: ModuleType, tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from platoon import gym as g, harness, testing\n"
        "import platoon.testing as t\n"
        "import platoon.gym as gym2\n"
        "g.loop, gym2.as_environment, gym2.StepEnv.reset, testing.run_episode, t.make_gateway, harness\n"
        "g.not_exported, testing.policy.ScriptedPolicy, t.episode, harness.builtin.codeact\n"
    )

    assert sorted(set(private_imports(probe, tmp_path, (), api))) == [
        "platoon.gym:not_exported",
        "platoon.harness:builtin",
        "platoon.testing:episode",
        "platoon.testing:policy",
    ]
