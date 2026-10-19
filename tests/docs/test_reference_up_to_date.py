"""The generated reference pages in ``docs/reference`` match the code (``scripts/gen_reference.py``)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import platoon

REPO = Path(__file__).resolve().parents[2]
REGENERATE = "run `python scripts/gen_reference.py` and commit the result"


def _generator():
    spec = importlib.util.spec_from_file_location("gen_reference", REPO / "scripts" / "gen_reference.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("gen_reference", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pages() -> dict[str, str]:
    return _generator().render_all()


@pytest.mark.parametrize("page", ["cli.md", "config.md", "api.md"])
def test_reference_page_is_up_to_date(pages: dict[str, str], page: str) -> None:
    path = REPO / "docs" / "reference" / page
    assert path.is_file(), f"docs/reference/{page} is missing; {REGENERATE}"
    assert path.read_text(encoding="utf-8") == pages[page], f"docs/reference/{page} is stale; {REGENERATE}"


def test_api_reference_covers_every_export(pages: dict[str, str]) -> None:
    missing = [name for name in platoon.__all__ if f"### `pt.{name}`" not in pages["api.md"]]
    assert not missing, f"api.md does not document {missing}"


def test_config_reference_covers_every_config_class(pages: dict[str, str]) -> None:
    generator = _generator()
    classes, _ = generator.config_classes()
    names = {cls.__name__ for cls in classes}
    assert {"PlatoonTinkerRLTrainerConfig", "PlatoonEvalConfig", "EnvironmentConfig", "RolloutConfig"} <= names
    for cls in classes:
        for field in generator.config_fields(cls):
            assert f"| `{field.name}` |" in pages["config.md"], (
                f"config.md does not document {cls.__name__}.{field.name}"
            )


def test_config_reference_marks_every_deprecated_key(pages: dict[str, str]) -> None:
    from platoon.deprecations import deprecated_keys

    generator = _generator()
    classes, _ = generator.config_classes()
    for cls in classes:
        names = {item.name for item in generator.config_fields(cls)}
        for name, deprecated in deprecated_keys(cls).items():
            assert name in names, f"{cls.__name__}.DEPRECATED_KEYS names {name}, which is not one of its keys"
            rows = [line for line in pages["config.md"].splitlines() if line.startswith(f"| `{name}` |")]
            replacement = f"`{deprecated.renamed_to}`" if deprecated.renamed_to is not None else deprecated.instead
            marked = f"**Deprecated:** use {replacement} instead; platoon {deprecated.removed_in} removes it."
            assert any(marked in row for row in rows), f"config.md does not mark {cls.__name__}.{name}"


def test_cli_reference_covers_every_command(pages: dict[str, str]) -> None:
    for command in ("train", "eval", "rollout", "config show", "config validate", "harness list", "harness show"):
        assert f"platoon {command}" in pages["cli.md"]
    for command in ("toolchain build", "prepare"):
        assert f"platoon {command}" in pages["cli.md"]
    for command in ("env list", "viz tail", "viz replay", "viz show-dump"):
        assert f"platoon {command}" in pages["cli.md"]
