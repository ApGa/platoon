"""The public API matches its committed snapshot, ``tests/public_api.txt`` (``scripts/public_api.py``), so that every
change to it is made on purpose; and the snapshot names every public name and changes with any signature."""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
UPDATE = (
    "If the change is on purpose, run `python scripts/public_api.py`, commit tests/public_api.txt with it, and say in "
    "the pull request what changed; removing or renaming a public name starts as an RFC (CONTRIBUTING.md, "
    '"When a change needs an RFC").'
)


@pytest.fixture(scope="module")
def api() -> ModuleType:
    """``scripts/public_api.py``, imported from its file."""

    spec = importlib.util.spec_from_file_location("public_api", REPO / "scripts" / "public_api.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("public_api", module)
    spec.loader.exec_module(module)
    return module


def test_the_public_api_matches_its_snapshot(api: ModuleType) -> None:
    committed = api.SNAPSHOT.read_text(encoding="utf-8")
    current = api.render()

    if current != committed:
        pytest.fail(f"The public API differs from tests/public_api.txt:\n{api.diff(committed, current)}\n{UPDATE}")


def test_the_snapshot_names_every_public_name_of_every_public_module(api: ModuleType) -> None:
    paths = {line.partition(": ")[0] for line in api.SNAPSHOT.read_text(encoding="utf-8").splitlines()}

    for module in api.PUBLIC_MODULES:
        missing = [name for name in api.public_names(module) if f"{module}.{name}" not in paths]
        assert not missing, f"tests/public_api.txt does not name {module}'s {missing}"
    assert {"platoon.harness", "platoon.Outcome.infra", "platoon.gym.loop", "platoon.testing.run_episode"} <= paths


PROBE = '''
import dataclasses


@dataclasses.dataclass(frozen=True)
class Probe:
    """A class of the public API."""

    size: int = 1
    tags: list = dataclasses.field(default_factory=list)

    def grow(self, by: int = 1) -> "Probe":
        """Grow."""

    @property
    def area(self) -> int:
        return self.size


async def probe(name: str, /, *, limit: int | None = None) -> str:
    """A function of the public API."""
'''

# Edits to PROBE, each a change to its API, and whether the snapshot changes with it.
EDITS = {
    "a docstring": ('"""Grow."""', '"""Grow by some amount."""', False),
    "a function body": ("return self.size", "return self.size * self.size", False),
    "a default": ("limit: int | None = None", "limit: int | None = 10", True),
    "a parameter name": ("by: int = 1", "step: int = 1", True),
    "a parameter kind": ("name: str, /, *,", "name: str, *,", True),
    "a return type": ('-> "Probe"', "-> None", True),
    "a field type": ("size: int = 1", "size: float = 1", True),
    "a new field": ("    size: int = 1\n", "    size: int = 1\n    label: str = ''\n", True),
    "the dataclass parameters": ("dataclass(frozen=True)", "dataclass", True),
    "async": ("async def probe", "def probe", True),
    "a property's type": ("def area(self) -> int", "def area(self) -> float", True),
}


@pytest.mark.parametrize("edit", EDITS, ids=list(EDITS))
def test_the_snapshot_changes_with_the_signatures_and_not_with_the_docs(api: ModuleType, edit: str) -> None:
    old, new, changes = EDITS[edit]
    assert old in PROBE

    def lines(source: str) -> list[tuple[str, str]]:
        namespace: dict = {"__name__": "platoon._public_api_probe"}  # a platoon module, as the API's are
        # Compiled without this file's ``from __future__ import annotations``, so annotations are objects.
        exec(compile(textwrap.dedent(source), "<probe>", "exec", dont_inherit=True), namespace)
        return [line for name in ("Probe", "probe") for line in api.describe(f"probe.{name}", namespace[name])]

    before, after = lines(PROBE), lines(PROBE.replace(old, new))

    assert (before != after) is changes, f"{edit}: {before} -> {after}"


def test_the_script_rewrites_a_stale_snapshot_and_check_reports_it(api, tmp_path, monkeypatch, capsys) -> None:
    snapshot = tmp_path / "public_api.txt"
    snapshot.write_text(api.render().replace("platoon.program: def program(", "platoon.program: def old_program("))
    monkeypatch.setattr(api, "SNAPSHOT", snapshot)
    monkeypatch.setattr(api, "REPO", tmp_path)

    assert api.main(["--check"]) == 1
    shown = capsys.readouterr().out
    assert "-platoon.program: def old_program(" in shown and "+platoon.program: def program(" in shown

    assert api.main([]) == 0
    assert snapshot.read_text() == api.render()
    assert api.main(["--check"]) == 0
    assert capsys.readouterr().out.endswith("public_api.txt is up to date\n")
