"""Installing an extension never changes a run that does not name it, checked with a real install.

The other extension tests write a distribution's metadata onto ``sys.path`` (``tests/extension_site.py``).  Here an
installer (uv, else pip) installs wheels into a scratch venv layered over the test environment, and platoon runs in
that venv's own interpreter:

* ``example-extension`` (``examples/extension-package``), whose components the configs name; and
* ``rival-extension``, which declares a component in every group under names of its own, entry points that claim
  core's names, a harness entry point named after its module (the form from before names were exact), a reward
  processor entry point named like the dotted import path of the example's, and an entry in the retired
  ``platoon.plugins`` group.  Importing any module of it is recorded in a file, and registers components under the
  example's names, so a run that imported one would also fail with "defined twice".

``platoon rollout`` of the example's named environment, and of a config that names the example's reward processor
by its dotted import path, gives the same outcome and the same ``lock.json`` components with the rival installed as
without it, and imports nothing of the rival.  Listing shows the rival's components without importing them, and a
command that names one imports it, so the rival is really there to be found.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import shutil
import site
import subprocess
import sys
import textwrap
import tomllib
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("tinker_cookbook", reason="platoon.testing's scripted policy needs the tinker extra")
if sys.platform == "win32":
    pytest.skip("the scratch venv's layout is POSIX's", allow_module_level=True)
if shutil.which("uv") is None and importlib.util.find_spec("pip") is None:
    pytest.skip("installing into a scratch venv needs an installer: uv or pip", allow_module_level=True)

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "extension-package"

RIVAL_FILES = {
    "rival_extension/__init__.py": """
        import os

        if os.environ.get("RIVAL_IMPORTS"):  # every import of this package, recorded for the test
            with open(os.environ["RIVAL_IMPORTS"], "a") as record:
                record.write(__name__ + "\\n")
    """,
    "rival_extension/components.py": """
        import platoon as pt


        @pt.harness("rival-agent")
        async def agent(ctx):
            return pt.AgentResult("completed", final_message="rival")


        @pt.harness("example-self-check")  # the example's name: a run that imported this module fails
        async def impostor(ctx):
            return pt.AgentResult("completed", final_message="impostor")


        @pt.program("rival-program")
        async def program(ep):
            return await ep.grade()


        @pt.program("example-exact-answer")  # the example's name
        async def impostor_program(ep):
            return await ep.grade()


        RIVAL_ARITHMETIC = pt.EnvironmentEntry(
            "rival-arithmetic", service="example_extension.tasks:arithmetic", harness="rival-agent"
        )


        def reward(trajectory):
            return -1.0, {"reward/rival": 1.0}
    """,
    "rival_extension/legacy.py": """
        import rival_extension.components  # noqa: F401  (registers harnesses on import, the old form)
    """,
}
RIVAL_ENTRY_POINTS = {
    "platoon.harnesses": {
        "rival-agent": "rival_extension.components:agent",
        "rival-harnesses": "rival_extension.legacy",  # named after its module, from before names were exact
        "platoon-react": "rival_extension.components:impostor",  # core's names: never used
        "mini-swe-agent": "rival_extension.components:impostor",
    },
    "platoon.programs": {
        "rival-program": "rival_extension.components:program",
        "single": "rival_extension.components:impostor_program",
    },
    "platoon.environments": {"rival-arithmetic": "rival_extension.components:RIVAL_ARITHMETIC"},
    "platoon.reward_processors": {
        "rival-reward": "rival_extension.components:reward",
        "example_extension.rewards.partial_credit": "rival_extension.components:reward",  # the example's import path
    },
    "platoon.plugins": {"rival": "rival_extension"},  # retired: platoon never reads it
}

# Runs each config given after the output directory through `platoon rollout` with a scripted policy, in the scratch
# venv's interpreter, and writes what each run produced and which modules of the rival were imported.
DRIVER = """
import contextlib, io, json, sys
from pathlib import Path

from platoon.cli import main
from platoon.testing import serve_policy

out, runs = Path(sys.argv[1]), {}
for config in sys.argv[2:]:
    run_dir, stdout, code = out / Path(config).stem, io.StringIO(), 0
    with serve_policy(["100", "102"]) as policy, contextlib.redirect_stdout(stdout):
        try:
            main(["rollout", config, "--task", "test-0", "--output-dir", str(run_dir), "--json-output"])
        except SystemExit as exc:
            code = exc.code
    lock = json.loads((run_dir / "lock.json").read_text()) if (run_dir / "lock.json").is_file() else {}
    summary = json.loads(stdout.getvalue()) if code == 0 else {}
    runs[Path(config).stem] = {
        "exit": code,
        "calls": len(policy.calls),
        "reward": summary.get("reward"),
        "steps": summary.get("total_steps"),
        "components": lock.get("components"),
        "extensions": lock.get("extensions"),
    }
rival = sorted(name for name in sys.modules if name.split(".")[0] == "rival_extension")
(out / "report.json").write_text(json.dumps({"runs": runs, "rival_modules": rival}, indent=2))
"""


def build_wheel(
    directory: Path,
    name: str,
    version: str,
    files: Mapping[str, str],
    *,
    requires: Sequence[str],
    entry_points: Mapping[str, Mapping[str, str]],
) -> Path:
    """A pure-Python wheel of ``files`` (path in the wheel -> text) with the metadata an installer reads."""

    stem = f"{name.replace('-', '_')}-{version}"
    info = f"{stem}.dist-info"
    metadata = ["Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}"]
    metadata += [f"Requires-Dist: {requirement}" for requirement in requires]
    groups = [
        "\n".join([f"[{group}]", *(f"{entry} = {value}" for entry, value in entries.items())])
        for group, entries in entry_points.items()
    ]
    contents = {
        **files,
        f"{info}/METADATA": "\n".join(metadata) + "\n",
        f"{info}/WHEEL": "Wheel-Version: 1.0\nGenerator: platoon-tests\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        f"{info}/entry_points.txt": "\n\n".join(groups) + "\n",
    }
    record = []
    for path, text in contents.items():
        data = text.encode()
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        record.append(f"{path},sha256={digest},{len(data)}")
    record.append(f"{info}/RECORD,,")
    wheel = directory / f"{stem}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, text in contents.items():
            archive.writestr(path, text)
        archive.writestr(f"{info}/RECORD", "\n".join(record) + "\n")
    return wheel


def example_wheel(directory: Path) -> Path:
    """examples/extension-package as a wheel: its package, data included, and its pyproject's metadata."""

    project = tomllib.loads((EXAMPLE / "pyproject.toml").read_text())["project"]
    sources = EXAMPLE / "src"
    files = {
        path.relative_to(sources).as_posix(): path.read_text()
        for path in sorted(sources.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }
    return build_wheel(
        directory,
        project["name"],
        project["version"],
        files,
        requires=project["dependencies"],
        entry_points=project["entry-points"],
    )


def rival_wheel(directory: Path) -> Path:
    files = {path: textwrap.dedent(code).lstrip() for path, code in RIVAL_FILES.items()}
    return build_wheel(
        directory, "rival-extension", "6.6.6", files, requires=["platoon>=0.2,<0.3"], entry_points=RIVAL_ENTRY_POINTS
    )


def make_scratch_venv(root: Path) -> Path:
    """A venv of its own that also sees the test environment's packages (through a ``.pth`` file); returns its
    interpreter.  What gets installed into it stays in it."""

    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(root)], check=True, timeout=300)
    python = root / "bin" / "python"
    purelib = subprocess.run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout.strip()
    parents = [path for path in site.getsitepackages() if Path(path).is_dir()]
    lines = [f"import site; site.addsitedir({path!r})\n" for path in parents]
    (Path(purelib) / "_platoon_test_parent_environment.pth").write_text("".join(lines))
    return python


def install(python: Path, wheel: Path, cwd: Path) -> None:
    uv = shutil.which("uv")
    if uv is not None:
        command = [uv, "pip", "install", "--python", str(python), "--no-deps", "--offline", "--no-cache"]
        command += ["--no-config", str(wheel)]
    else:
        command = [str(python), "-m", "pip", "install", "--no-deps", "--no-index", "--disable-pip-version-check"]
        command += [str(wheel)]
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, f"{command} failed:\n{result.stdout}\n{result.stderr}"


def run_python(python: Path, *arguments: str, env: Mapping[str, str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(python), *arguments], env=dict(env), cwd=cwd, capture_output=True, text=True, timeout=900
    )


def rollouts(python: Path, out: Path, configs: Sequence[Path], env: Mapping[str, str]) -> dict[str, Any]:
    out.mkdir(parents=True)
    driver = out / "driver.py"
    driver.write_text(DRIVER)
    result = run_python(python, str(driver), str(out), *map(str, configs), env=env, cwd=out)
    assert result.returncode == 0, f"the driver failed:\n{result.stdout}\n{result.stderr}"
    return json.loads((out / "report.json").read_text())


@pytest.fixture(scope="module")
def scratch(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the configs before and after the rival is installed, then list and name its components."""

    root = tmp_path_factory.mktemp("installed-extensions")
    wheels = root / "wheels"
    wheels.mkdir()
    python = make_scratch_venv(root / "venv")
    install(python, example_wheel(wheels), cwd=root)

    configs = root / "configs"
    configs.mkdir()
    for name in ("eval.yaml", "named.yaml"):
        shutil.copy(EXAMPLE / name, configs / name)
    dotted = configs / "dotted.yaml"  # the example's reward processor by its dotted import path
    dotted.write_text(
        "extends: named.yaml\n\nenvironments:\n  - name: arithmetic\n    env: example-arithmetic@1\n"
        "    reward_processor: example_extension.rewards.partial_credit\n"
    )
    imports = root / "rival-imports.txt"
    env = {key: value for key, value in os.environ.items() if key not in ("PYTHONHOME", "PYTHONSTARTUP")}
    env.update(
        PYTHONPATH=str(REPO),  # the platoon under test
        RIVAL_IMPORTS=str(imports),
        HF_HUB_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1",
    )
    named = [configs / "named.yaml", dotted]

    def imported_since(start: str) -> str:
        """The rival's imports recorded since the record read ``start``."""
        return (imports.read_text() if imports.exists() else "").removeprefix(start)

    before = rollouts(python, root / "before", named, env)
    install(python, rival_wheel(wheels), cwd=root)
    after = rollouts(python, root / "after", named, env)
    imported_by_runs = imported_since("")
    listed = run_python(python, "-m", "platoon.cli", "extensions", "list", "--json-output", env=env, cwd=root)
    imported_by_listing = imported_since(imported_by_runs)
    shown = run_python(python, "-m", "platoon.cli", "harness", "show", "rival-agent", env=env, cwd=root)
    return {
        "before": before,
        "after": after,
        "imported_by_runs": imported_by_runs,
        "listed": listed,
        "imported_by_listing": imported_by_listing,
        "shown": shown,
        "imported_by_show": imported_since(imported_by_runs + imported_by_listing),
    }


def test_the_runs_find_the_examples_components_by_name(scratch):
    for name, run in scratch["before"]["runs"].items():
        assert (run["exit"], run["calls"], run["reward"], run["steps"]) == (0, 2, 1.0, 2), name
        resolved = {component["kind"]: component["resolved"] for component in run["components"]}
        assert resolved["environment"] == "example-arithmetic@1" and resolved["harness"] == "example-self-check@1"
        assert resolved["program"] == "example-exact-answer@1"
        assert set(run["extensions"]) == {"example-extension"}
    (processor,) = [
        item for item in scratch["before"]["runs"]["dotted"]["components"] if item["kind"] == "reward_processor"
    ]
    assert (processor["resolved"], processor["distribution"]) == (
        "example_extension.rewards:partial_credit",
        "example-extension",
    )


def test_installing_a_rival_extension_changes_nothing_in_a_run_that_does_not_name_it(scratch):
    assert scratch["after"]["runs"] == scratch["before"]["runs"]  # outcome, steps and every lock.json component
    assert scratch["after"]["rival_modules"] == [] and scratch["imported_by_runs"] == ""


def test_the_rival_is_installed_and_listed_without_being_imported(scratch):
    listed = scratch["listed"]
    assert listed.returncode == 0, listed.stderr
    rows = {
        (row["kind"], row["name"]): row for row in json.loads(listed.stdout) if row["distribution"] == "rival-extension"
    }
    for key in [
        ("harness", "rival-agent"),
        ("harness", "rival-harnesses"),
        ("program", "rival-program"),
        ("environment", "rival-arithmetic"),
        ("reward_processor", "rival-reward"),
    ]:
        assert rows[key]["status"] == "ok", key
    for key in [("harness", "platoon-react"), ("harness", "mini-swe-agent"), ("program", "single")]:
        assert rows[key]["status"].startswith("ignored: "), key  # core's names stay core's
    assert scratch["imported_by_listing"] == ""


def test_a_command_that_names_a_rival_component_imports_it(scratch):
    shown = scratch["shown"]
    assert shown.returncode == 0, shown.stderr
    assert shown.stdout.startswith("rival-agent@1")
    assert "rival_extension" in scratch["imported_by_show"].split()
