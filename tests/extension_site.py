"""A scratch site directory where a test installs fake distributions that declare platoon entry points.

``sandbox(tmp_path, monkeypatch)`` puts the directory first on ``sys.path`` and, when the test ends, forgets the
modules it holds and restores every registry the extensions can add to (harnesses, programs, named environments,
reward processors) and the extension loader's own state, so that one test's extensions never leak into another::

    @pytest.fixture
    def site(tmp_path, monkeypatch):
        with sandbox(tmp_path, monkeypatch) as site:
            yield site

    def test_something(site):
        site.module("acme_platoon/agent.py", AGENT_CODE)
        site.install("acme-platoon", entry_points={"platoon.harnesses": {"acme-agent": "acme_platoon.agent:agent"}})
"""

from __future__ import annotations

import contextlib
import importlib
import sys
import textwrap
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from platoon import extensions, programs, registry
from platoon.envs import named
from platoon.harness import function
from platoon.harness import registry as harness_registry


@dataclass
class Site:
    """The scratch site directory: ``module`` writes code into it, ``install`` writes a distribution's metadata."""

    root: Path

    def module(self, path: str, code: str) -> Path:
        """Write ``code`` to ``path`` (relative, e.g. ``acme_platoon/agent.py``), with ``__init__.py`` files for the
        packages on the way."""

        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        package = target.parent
        while package != self.root:
            (package / "__init__.py").touch()
            package = package.parent
        target.write_text(textwrap.dedent(code))
        importlib.invalidate_caches()
        return target

    def install(
        self,
        name: str,
        version: str = "0.1.0",
        *,
        entry_points: Mapping[str, Mapping[str, str]] = {},
        requires: Sequence[str] = ("platoon>=0.3,<0.4",),
        extras: Sequence[str] = (),
        direct_url: str | None = None,
    ) -> Path:
        """Install distribution ``name`` (metadata only: its code goes in with :meth:`module`)."""

        dist_info = self.root / f"{name.replace('-', '_')}-{version}.dist-info"
        dist_info.mkdir()
        metadata = ["Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}"]
        metadata += [f"Requires-Dist: {requirement}" for requirement in requires]
        metadata += [f"Provides-Extra: {extra}" for extra in extras]
        (dist_info / "METADATA").write_text("\n".join(metadata) + "\n")
        lines = []
        for group, entries in entry_points.items():
            lines.append(f"[{group}]")
            lines += [f"{entry} = {value}" for entry, value in entries.items()]
        (dist_info / "entry_points.txt").write_text("\n".join(lines) + "\n")
        if direct_url is not None:
            (dist_info / "direct_url.json").write_text(direct_url)
        importlib.invalidate_caches()
        extensions.refresh()
        return dist_info

    def install_pyproject(self, pyproject: Path) -> Path:
        """Install a checkout's distribution as its ``pyproject.toml`` declares it: name, version, requirements and
        entry points (the code stays where it is, importable from ``sys.path``)."""

        project = tomllib.loads(pyproject.read_text())["project"]
        return self.install(
            project["name"],
            project["version"],
            entry_points=project.get("entry-points", {}),
            requires=project.get("dependencies", ()),
            extras=tuple(project.get("optional-dependencies", {})),
        )


def _snapshot() -> dict:
    with function._lock:
        harnesses = dict(function._function_harnesses)
    with programs._lock:
        program_records = dict(programs._programs)
    with named._lock:
        entries = dict(named._entries)
    with extensions._lock:
        loaded = dict(extensions._loaded)
        failures = {group: dict(items) for group, items in extensions._failures.items()}
        origins = dict(extensions._origins)
    components = {kind: dict(item._items) for kind, item in registry._REGISTRIES.items()}
    return {
        "harnesses": harnesses,
        "programs": program_records,
        "entries": entries,
        "loaded": loaded,
        "failures": failures,
        "origins": origins,
        "components": components,
    }


def _restore(state: dict) -> None:
    with function._lock:
        function._function_harnesses.clear()
        function._function_harnesses.update(state["harnesses"])
    with programs._lock:
        programs._programs.clear()
        programs._programs.update(state["programs"])
    with named._lock:
        named._entries.clear()
        named._entries.update(state["entries"])
    with extensions._lock:
        extensions._loaded.clear()
        extensions._loaded.update(state["loaded"])
        extensions._failures.clear()
        extensions._failures.update(state["failures"])
        extensions._origins.clear()
        extensions._origins.update(state["origins"])
    for kind in list(registry._REGISTRIES):
        if kind in state["components"]:
            registry._REGISTRIES[kind]._items = dict(state["components"][kind])
        else:
            del registry._REGISTRIES[kind]
    harness_registry.default_registry.cache_clear()


def _inside(module: object, root: Path) -> bool:
    path = getattr(module, "__file__", None)
    if path is not None:
        return Path(path).resolve().is_relative_to(root.resolve())
    locations = getattr(module, "__path__", None)  # a namespace package has a path but no file
    return locations is not None and any(Path(where).resolve().is_relative_to(root.resolve()) for where in locations)


@contextlib.contextmanager
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Site]:
    """A :class:`Site` on ``sys.path`` for one test, with every registry restored when it ends."""

    root = tmp_path / "site"
    root.mkdir()
    # Core's built-ins register on the first lookup; register them before the snapshot so it keeps them.
    harness_registry.available_harnesses()
    programs.available_programs()
    state = _snapshot()
    monkeypatch.syspath_prepend(str(root))
    extensions.refresh()
    try:
        yield Site(root)
    finally:
        for name in [name for name, module in list(sys.modules.items()) if _inside(module, root)]:
            del sys.modules[name]
        _restore(state)
        while str(root) in sys.path:  # monkeypatch restores sys.path too, after this fixture's teardown
            sys.path.remove(str(root))
        importlib.invalidate_caches()
        extensions.refresh()
