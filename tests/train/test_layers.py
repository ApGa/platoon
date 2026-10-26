"""The dependency rule of the training code (RFC 0021, section 3.2): generic code never imports a trainer or a plugin,
and a plugin never imports a trainer.

=========  ============================================================================================================
generic    ``policy/``, ``objectives/``, ``trainer/{api,run,registry,hooks,signals,waiting}.py``, ``dispatch/``,
           ``checkpoints/``, ``evaluation/``, ``workers/``, and the flat modules but ``launch.py``
trainers   ``trainer/{rl,loops,state,sft,interleaved}.py``, ``step/``
plugins    ``workflow/``, ``credit/``
=========  ============================================================================================================

Every import counts, a function's own, those under ``TYPE_CHECKING`` and relative ones (as the absolute import each
is) included.  ``launch.py`` (which names the trainers), the package's exports (``platoon/train/__init__.py``) and the
old paths' shims (``tinker/``) belong to no group.  The workflow's contract (``workflow/protocol.py``), which generic
code calls the workflow through, is everyone's.  No import breaks the rule.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "platoon" / "train"
GENERIC_TRAINER_MODULES = {"api", "run", "registry", "hooks", "signals", "waiting"}
TRAINER_MODULES = {"rl", "loops", "state", "sft", "interleaved"}
GENERIC_PACKAGES = {"policy", "objectives", "dispatch", "checkpoints", "evaluation", "workers"}
PLUGIN_PACKAGES = {"workflow", "credit"}
UNGROUPED = {"launch", "__init__", "tinker"}
# The contract modules that every group may import.
CONTRACTS = {"platoon.train.workflow.protocol"}
# What each group may not import.
FORBIDDEN = {"generic": {"trainers", "plugins"}, "plugins": {"trainers"}, "trainers": set()}


def group(module: str) -> str | None:
    """The group of ``module`` (``platoon.train...``), or None for one that belongs to none."""

    parts = module.split(".")[2:]
    if not parts or parts[0] in UNGROUPED:
        return None
    head = parts[0]
    if head == "trainer":
        name = parts[1] if len(parts) > 1 else "__init__"
        return "generic" if name in GENERIC_TRAINER_MODULES else "trainers" if name in TRAINER_MODULES else None
    if head == "step":
        return "trainers"
    if head in PLUGIN_PACKAGES:
        return "plugins"
    if head in GENERIC_PACKAGES or len(parts) == 1:
        return "generic"
    return None


def module_name(path: Path) -> str:
    relative = path.relative_to(PACKAGE.parents[1]).with_suffix("")
    parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
    return ".".join(parts)


def is_module(name: str) -> bool:
    path = PACKAGE.parents[1].joinpath(*name.split("."))
    return path.with_suffix(".py").exists() or (path / "__init__.py").exists()


def imports(path: Path) -> set[str]:
    """Every platoon.train module that ``path`` imports, anywhere in it, a relative import as the absolute one it is."""

    own = module_name(path)
    package = own if path.name == "__init__.py" else own.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
            for alias in node.names:
                submodule = f"{base}.{alias.name}"
                found.add(submodule if is_module(submodule) else base)
    return {name for name in found if name.startswith("platoon.train.")}


@functools.cache
def breaks() -> dict[tuple[str, str], str]:
    """Each import that breaks the rule, with the groups it crosses."""

    found = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        importer = module_name(path)
        if (own := group(importer)) is None:
            continue
        for imported in imports(path) - CONTRACTS:
            if (theirs := group(imported)) in FORBIDDEN[own]:
                found[(importer, imported)] = f"{own} imports {theirs}"
    return found


def test_the_groups_are_as_rfc_0021_names_them():
    assert group("platoon.train.trainer.run") == group("platoon.train.policy.learner") == "generic"
    assert group("platoon.train.trainer.loops") == group("platoon.train.step.publish") == "trainers"
    assert group("platoon.train.workflow.group") == group("platoon.train.credit.group") == "plugins"
    assert group("platoon.train.task_stats") == "generic"
    assert group("platoon.train.launch") is group("platoon.train.tinker.proxy") is None


def test_generic_code_imports_no_trainer_or_plugin_and_plugins_no_trainer():
    found = breaks()
    assert not found, "imports that break the dependency rule (RFC 0021, section 3.2):\n" + "\n".join(
        f"  {importer} imports {imported} ({crossing})" for (importer, imported), crossing in sorted(found.items())
    )
