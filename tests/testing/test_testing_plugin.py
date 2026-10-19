"""platoon.testing's pytest plugin is opt-in: ``-p platoon.testing`` or ``pytest_plugins`` loads its fixtures, and
installing platoon alone does not."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
USES_THE_FIXTURES = """
from platoon.gateway import core


def test_the_fixtures(toy_tokenizer, toy_renderer, serve_policy):
    policy = serve_policy(["hello"])

    assert core.default_gateway().sampler is policy
    assert toy_tokenizer.encode("A", add_special_tokens=False) == [65]
    assert toy_renderer.get_stop_sequences() == [258]


def test_the_policy_is_no_longer_served_after_its_test():
    assert core._default_gateway is None
"""


def run_pytest(directory: Path, *args: str) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")])}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args],
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.mark.parametrize("enable", ["-p", "pytest_plugins"])
def test_opting_in_provides_the_fixtures(tmp_path, enable):
    (tmp_path / "test_extension.py").write_text(USES_THE_FIXTURES)
    if enable == "pytest_plugins":
        (tmp_path / "conftest.py").write_text('pytest_plugins = ["platoon.testing"]\n')
        completed = run_pytest(tmp_path)
    else:
        completed = run_pytest(tmp_path, "-p", "platoon.testing")

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "2 passed" in completed.stdout


def test_without_opting_in_there_are_no_fixtures(tmp_path):
    (tmp_path / "test_extension.py").write_text(USES_THE_FIXTURES)

    completed = run_pytest(tmp_path)

    assert completed.returncode != 0 and "fixture 'toy_tokenizer' not found" in completed.stdout


def test_platoon_declares_no_pytest11_entry_point():
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]

    assert "pytest11" not in project.get("entry-points", {})
