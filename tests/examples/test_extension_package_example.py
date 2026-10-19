"""examples/extension-package: once installed, the package's components are found by the names its entry points give;
its configs name them; a rollout runs them end to end with a scripted policy; its own tests pass; and the tutorial
(docs/extending/publish-a-package.md) quotes its files and platoon's output as they are.

A test "installs" the package by putting its sources, and a ``.dist-info`` directory written from its
``pyproject.toml``, on ``sys.path``: ``importlib.metadata`` then finds its entry points as it finds those of any
installed distribution.  CPU only; the rollouts use the examples' scripted policy (see conftest.py).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import pytest
import yaml

from platoon import cli
from platoon.extensions import GROUPS, ExtensionError

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "tests") not in sys.path:
    sys.path.insert(0, str(REPO / "tests"))
# Fake installs, which restore every registry and the extension loader's state after a test.
from extension_site import sandbox  # noqa: E402

EXAMPLE = REPO / "examples" / "extension-package"
SOURCES = EXAMPLE / "src"
PACKAGE = SOURCES / "example_extension"
TUTORIAL = REPO / "docs" / "extending" / "publish-a-package.md"
PROJECT = tomllib.loads((EXAMPLE / "pyproject.toml").read_text())["project"]
_FENCE = re.compile(r"^```(?P<lang>[\w-]*)\n(?P<body>.*?)^```$", re.S | re.M)


def tutorial_blocks(language: str | None = None) -> list[str]:
    text = TUTORIAL.read_text(encoding="utf-8")
    return [match["body"] for match in _FENCE.finditer(text) if language is None or match["lang"] == language]


def shown_block(first_line: str) -> str:
    """The tutorial's one ``text`` block that starts with ``first_line``: what it shows a command prints."""

    found = [code for code in tutorial_blocks("text") if code.startswith(first_line)]
    assert len(found) == 1, f"publish-a-package.md has {len(found)} text blocks starting with {first_line!r}"
    return found[0]


def write_dist_info(site: Path) -> Path:
    """The example package's installed metadata, in ``site``: name, version, requirements and entry points, for a
    process of its own (in this one, ``install`` installs the package)."""

    dist_info = site / f"{PROJECT['name'].replace('-', '_')}-{PROJECT['version']}.dist-info"
    dist_info.mkdir(parents=True)
    metadata = ["Metadata-Version: 2.1", f"Name: {PROJECT['name']}", f"Version: {PROJECT['version']}"]
    metadata += [f"Requires-Dist: {requirement}" for requirement in PROJECT.get("dependencies", [])]
    (dist_info / "METADATA").write_text("\n".join(metadata) + "\n")
    groups = [
        "\n".join([f"[{group}]", *(f"{name} = {value}" for name, value in entries.items())])
        for group, entries in PROJECT.get("entry-points", {}).items()
    ]
    (dist_info / "entry_points.txt").write_text("\n\n".join(groups) + "\n")
    return dist_info


def _drop_example_modules() -> None:
    for name in [name for name in sys.modules if name.split(".")[0] == "example_extension"]:
        del sys.modules[name]


@contextlib.contextmanager
def install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, requires: list[str] | None = None) -> Iterator[Path]:
    """The example package, installed until the block ends: its sources and its metadata on ``sys.path``, with
    every registry restored afterwards."""

    monkeypatch.syspath_prepend(str(SOURCES))
    with sandbox(tmp_path, monkeypatch) as site:
        site.install(
            PROJECT["name"],
            PROJECT["version"],
            entry_points=PROJECT["entry-points"],
            requires=requires or PROJECT["dependencies"],
        )
        try:
            yield site.root
        finally:
            _drop_example_modules()


@pytest.fixture
def installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """The example package, installed for one test."""

    with install(tmp_path, monkeypatch) as site:
        yield site


# --------------------------------------------------------------------------- the tutorial and the package


@pytest.mark.parametrize(
    "path",
    [
        PACKAGE / "harness.py",
        PACKAGE / "program.py",
        PACKAGE / "rewards.py",
        PACKAGE / "tasks.py",
        PACKAGE / "entries.py",
        EXAMPLE / "pyproject.toml",
        EXAMPLE / "named.yaml",
        EXAMPLE / "tests" / "test_example_extension.py",
    ],
    ids=lambda path: str(path.relative_to(EXAMPLE)),
)
def test_the_tutorial_quotes_the_file_as_it_is(path: Path) -> None:
    assert path.read_text() in tutorial_blocks(), f"publish-a-package.md does not quote {path.name} as it is"


def test_the_tutorial_quotes_the_eval_config_s_environment_and_a_task_row() -> None:
    quoted = [yaml.safe_load(code) for code in tutorial_blocks("yaml")]
    environments = yaml.safe_load((EXAMPLE / "eval.yaml").read_text())["environments"]
    assert {"environments": environments} in quoted
    [row] = tutorial_blocks("json")
    assert row in (PACKAGE / "data" / "arithmetic.jsonl").read_text().splitlines(keepends=True)


def test_every_component_is_named_by_its_owner_and_every_entry_point_by_its_component(installed) -> None:
    from platoon.harness.function import harness_of

    entry_points = PROJECT["entry-points"]
    assert set(entry_points) == set(GROUPS.values())  # one component of every kind
    for group, entries in entry_points.items():
        for name, value in entries.items():
            assert name.startswith("example-"), f"{group} entry {name} does not start with the owner example-"
            loaded = EntryPoint(name=name, value=value, group=group).load()
            if group == "platoon.harnesses":
                assert harness_of(loaded) is not None and harness_of(loaded).name == name
            elif group == "platoon.environments":
                assert loaded.name == name
            else:
                assert callable(loaded), f"{group} entry {name} loads {loaded!r}"


def test_the_configs_validate(installed: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Validating resolves the components the configs name, which the installed package provides.
    assert cli.config_validate(EXAMPLE / "eval.yaml", EXAMPLE / "train.yaml") == 0

    out = capsys.readouterr().out
    assert "(PlatoonEvalConfig)" in out and "(PlatoonTinkerRLTrainerConfig)" in out


# --------------------------------------------------------------------------- installed


def test_installed_the_harness_is_found_by_its_name(installed: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["harness", "show", "example-self-check"])

    assert capsys.readouterr().out == shown_block("example-self-check@1  (function)")

    cli.main(["harness", "list"])
    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert any(row[:2] == ["example-self-check@1", "function"] for row in rows)


def answer_then_correct(prompt: str) -> str:
    """The scripted model: 100 at first, then the right sum once it is asked to check (task test-0: 22 + 80)."""

    return "102" if "Check your answer step by step" in prompt else "100"


def _rollout(config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], *overrides: str) -> dict[str, Any]:
    run_dir = tmp_path / "run"
    cli.main(["rollout", str(config), *overrides, "--task", "test-0", "--output-dir", str(run_dir)])
    out = capsys.readouterr().out
    assert "task test-0\nagents 1  steps 2  max depth 0  reward 1.0  outcome scored reward=1.0, close=1.0\n" in out
    collection = json.loads((run_dir / "trajectory_collection.json").read_text())
    (root,) = collection["trajectories"].values()
    assert root["misc"]["outcome"]["rewards"] == {"reward": 1.0, "close": 1.0}
    assert [step["misc"]["turn"] for step in root["steps"]] == ["answer", "check"]
    return root


def test_a_rollout_runs_the_package_s_components(installed, serve_policy, tmp_path, capsys) -> None:
    """The harness by the name its entry point gives; the program and the reward processor by import string, as a
    config can name them on any platoon."""

    serve_policy(answer_then_correct)

    root = _rollout(
        EXAMPLE / "eval.yaml",
        tmp_path,
        capsys,
        "environments.0.program=example_extension.program:exact_answer",
        "environments.0.reward_processor=example_extension.rewards:partial_credit",
    )

    assert root["misc"]["harness"]["spec"] == "example-self-check@1"


def test_a_rollout_names_every_component_by_its_entry_point(installed, serve_policy, tmp_path, capsys) -> None:
    serve_policy(answer_then_correct)

    _rollout(EXAMPLE / "eval.yaml", tmp_path, capsys)

    lock = json.loads((tmp_path / "run" / "lock.json").read_text())
    providers = {(record["kind"], record["ref"]): record for record in lock["components"]}
    for kind, ref in (
        ("harness", "example-self-check"),
        ("program", "example-exact-answer"),
        ("reward_processor", "example-partial-credit"),
    ):
        assert providers[(kind, ref)]["distribution"] == PROJECT["name"], (kind, ref)
        assert providers[(kind, ref)]["distribution_version"] == PROJECT["version"], (kind, ref)
    assert PROJECT["name"] in lock["extensions"]


def test_a_rollout_through_the_named_environment(installed, serve_policy, tmp_path, capsys) -> None:
    serve_policy(answer_then_correct)

    _rollout(EXAMPLE / "named.yaml", tmp_path, capsys)

    resolved = yaml.safe_load((tmp_path / "run" / "config.resolved.yaml").read_text())
    (environment,) = resolved["environments"]
    assert environment["env"] == "example-arithmetic@1"  # pinned to the version the name resolved to
    assert (environment["harness"], environment["harness_options"]) == ("example-self-check", {"checks": 1})
    lock = json.loads((tmp_path / "run" / "lock.json").read_text())
    [named] = [record for record in lock["components"] if record["kind"] == "environment"]
    assert (named["resolved"], named["distribution"]) == ("example-arithmetic@1", PROJECT["name"])


def test_installed_the_extension_lists_and_checks_clean(installed, capsys) -> None:
    def platoon_and_example(table: str) -> list[list[str]]:
        """The header and the rows of platoon and of the example (other installed extensions may list programs too),
        as columns, without platoon's own version: the docs show the release they were written with."""
        header, *rows = (line.split() for line in table.splitlines())
        kept = [row for row in rows if row[2] in ("platoon", PROJECT["name"])]
        return [header] + [row[:3] + row[4:] if row[2] == "platoon" else row for row in kept]

    cli.main(["extensions", "list", "--kind", "program"])
    assert platoon_and_example(capsys.readouterr().out) == platoon_and_example(shown_block("kind "))

    assert cli.extensions_check(PROJECT["name"], load=True) == 0
    assert capsys.readouterr().out == shown_block("ok: no problems found")

    cli.main(["env", "show", "example-arithmetic"])
    assert capsys.readouterr().out == shown_block("example-arithmetic@1")


def test_a_platoon_outside_the_package_s_range_refuses_it(tmp_path, monkeypatch) -> None:
    from platoon.programs import resolve_program

    with install(tmp_path, monkeypatch, requires=["platoon >= 99"]):
        with pytest.raises(ExtensionError, match=r"example-extension 0\.1\.0 requires platoon *>= *99"):
            resolve_program("example-exact-answer")


def test_the_package_s_own_tests_pass(tmp_path: Path) -> None:
    """All of them, the named environment's included: the package is installed for the run, as its README says."""

    site = tmp_path / "site-packages"
    write_dist_info(site)
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(SOURCES), str(site), str(REPO), os.environ.get("PYTHONPATH", "")]),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
        cwd=EXAMPLE,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    count = len(re.findall(r"^def test_", (EXAMPLE / "tests" / "test_example_extension.py").read_text(), re.M))
    assert f"{count} passed" in result.stdout
    [shown] = [code for code in tutorial_blocks("text") if " passed in " in code]  # what the tutorial says it prints
    assert f"\n{count} passed in " in f"\n{shown}"
