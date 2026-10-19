"""docs/how-to/test-extensions.md stays true: it quotes the quickstart's tests verbatim, those tests pass when run as
the guide runs them, its gateway example runs, and its reference lists every name ``platoon.testing`` exports."""

from __future__ import annotations

import os
import re
import subprocess
import sys

from harness_example_fakes import EXAMPLES, REPO

import platoon.testing

GUIDE = (REPO / "docs" / "how-to" / "test-extensions.md").read_text()
TESTS = EXAMPLES / "quickstart" / "test_guess.py"


def python_blocks() -> list[str]:
    return re.findall(r"^```python\n(.*?)^```$", GUIDE, re.S | re.M)


def section(heading: str) -> str:
    start = GUIDE.index(f"\n## {heading}\n")
    end = GUIDE.find("\n## ", start + 1)
    return GUIDE[start : end if end != -1 else None]


def test_the_steps_quote_the_example_tests():
    steps = section("Steps")
    quoted = [block for block in python_blocks() if block in steps]
    text = TESTS.read_text()

    assert len(quoted) == 6 and [block for block in quoted if block not in text] == []
    assert re.findall(r"^def (test_\w+)", text, re.M) == re.findall(r"^def (test_\w+)", "".join(quoted), re.M)


def test_the_example_tests_pass_as_the_guide_runs_them():
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join([".", str(REPO), os.environ.get("PYTHONPATH", "")])}

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "platoon.testing", "-p", "no:cacheprovider", "test_guess.py"],
        cwd=TESTS.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "5 passed" in completed.stdout


def test_the_gateway_example_runs():
    (block,) = [block for block in python_blocks() if block in section("Lower level: the gateway")]

    exec(compile(block, "test-extensions.md", "exec"), {})


def test_the_reference_lists_every_export():
    reference = section("Reference")

    assert [name for name in platoon.testing.__all__ if f"`{name}`" not in reference] == []
