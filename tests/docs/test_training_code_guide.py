"""The reading guide of the training code (``docs/training-code.md``) names only what the code has: each name it links
is defined in the file the link points at (a function, a class or a module-level name, or ``Class.member``), each file
it names is the one the link points at, and each module it names (``platoon.train`` and the like) exists.  The links
themselves resolve: ``test_docs_links.py``."""

from __future__ import annotations

import ast
import functools
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GUIDE = REPO / "docs" / "training-code.md"

# A link whose text is a code span: [`Name`](../platoon/...py) or [`Class.member`](...).
_LINK = re.compile(r"\[`(?P<text>[^`]+)`\]\((?P<target>[^)#\s]+)(?:#[^)\s]*)?\)")
_NAME = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?")
_MODULE = re.compile(r"platoon(?:\.\w+)*")


@functools.cache
def _definitions(path: Path) -> dict[str, frozenset[str]]:
    """The names ``path`` defines at module level, each with the members its class body defines (none for a function
    or a variable)."""

    found: dict[str, frozenset[str]] = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.ClassDef):
            found[node.name] = frozenset(_assigned(node.body))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found[node.name] = frozenset()
        else:
            found.update(dict.fromkeys(_assigned([node]), frozenset()))
    return found


def _assigned(body: list[ast.stmt]) -> set[str]:
    """The names a block defines: its functions, classes and assignments."""

    names: set[str] = set()
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _citations() -> list[tuple[str, Path]]:
    """Each code-span link of the guide to a Python file: its text and the file."""

    found = []
    for match in _LINK.finditer(GUIDE.read_text(encoding="utf-8")):
        target = (GUIDE.parent / match["target"]).resolve()
        if target.suffix == ".py":
            found.append((match["text"], target))
    return found


def _module_file(module: str) -> Path | None:
    """The file of a dotted module name of this repository, or None for one that has none."""

    path = REPO.joinpath(*module.split("."))
    for candidate in (path.with_suffix(".py"), path / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def test_each_name_the_guide_links_is_defined_where_the_link_points():
    citations = _citations()
    problems = []
    for text, path in citations:
        relative = path.relative_to(REPO)
        if not path.is_file():
            problems.append(f"{text}: {relative} does not exist")
        elif text.endswith(".py"):  # a file, named by its path
            if not relative.as_posix().endswith(text):
                problems.append(f"{text}: the link points at {relative}")
        elif _MODULE.fullmatch(text) and text.startswith("platoon."):
            if _module_file(text) != path:
                problems.append(f"{text}: the link points at {relative}, not at the module")
        elif _NAME.fullmatch(text):
            name, _, member = text.partition(".")
            defined = _definitions(path)
            if name not in defined:
                problems.append(f"{text}: {relative} defines no {name}")
            elif member and member not in defined[name]:
                problems.append(f"{text}: {name} in {relative} has no {member}")
    assert not problems, "\n".join(problems)
    assert len([text for text, _ in citations if _NAME.fullmatch(text)]) > 80  # the guide names the code it maps


def test_each_module_the_guide_names_exists():
    named = {match for match in re.findall(r"`(platoon(?:\.\w+)+)`", GUIDE.read_text(encoding="utf-8"))}
    missing = sorted(module for module in named if _module_file(module) is None)
    assert not missing, missing
