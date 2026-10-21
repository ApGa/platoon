"""Spec keys for harnesses that run from a toolchain inside a task sandbox: named toolchains, ``[install.recipe]``,
``[resources]`` and ``[budget]``, and the two built-in OpenHands specs that use them."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest
from harness_fakes import FakeAgent, FakeSession, ProcessSandbox
from site_cache import use_site_cache
from test_harness_spec import CONTEXT, minimal

from platoon.harness.registry import get_harness_spec
from platoon.harness.runner import HarnessRunner, HarnessSetupError
from platoon.harness.spec import (
    BudgetSpec,
    HarnessSpecError,
    ResourcesSpec,
    load_spec,
    parse_spec,
    resolve_toolchain,
)
from platoon.task import Task

SPECS = Path(__file__).resolve().parents[2] / "platoon" / "harness" / "builtin" / "specs"


def recipe_spec(tmp_path: Path, recipe: str, *, toolchain: str = "my-agent-1.0") -> Path:
    (tmp_path / "my-agent.txt").write_text("my-agent==1.0 --hash=sha256:00\n")
    (tmp_path / "make.py").write_text("print('generated')\n")
    path = tmp_path / "my-agent.toml"
    path.write_text(
        textwrap.dedent(
            f"""
            name = "my-agent"
            version = "1.0"
            [install]
            toolchain = "{toolchain}"
            [install.recipe]
            {recipe}
            [launch]
            argv = ["{{toolchain}}/bin/python3", "-m", "my_agent"]
            """
        )
    )
    return path


def test_a_toolchain_is_an_absolute_path_or_a_name():
    assert parse_spec(minimal(install={"toolchain": "/opt/agent"})).install.toolchain == "/opt/agent"
    assert parse_spec(minimal(install={"toolchain": "my-agent-1.0"})).install.toolchain == "my-agent-1.0"
    for bad in ("relative/path", "../up", "-dash", ""):
        with pytest.raises(HarnessSpecError, match="absolute path or a name"):
            parse_spec(minimal(install={"toolchain": bad}))


def test_a_named_toolchain_resolves_under_its_root(monkeypatch, tmp_path):
    assert resolve_toolchain("/opt/agent", "/ignored") == "/opt/agent"
    assert resolve_toolchain("my-agent-1.0", "/sites/toolchains") == "/sites/toolchains/my-agent-1.0"
    monkeypatch.setenv("HOME", str(tmp_path))
    assert resolve_toolchain("my-agent-1.0") == f"{tmp_path}/.cache/platoon/toolchains/my-agent-1.0"
    use_site_cache(monkeypatch, tmp_path / "site")
    assert resolve_toolchain("my-agent-1.0") == f"{tmp_path}/site/toolchains/my-agent-1.0"


def test_a_recipe_resolves_its_files_next_to_the_spec(tmp_path):
    path = recipe_spec(
        tmp_path,
        'python = "3.12"\nlock = "my-agent.txt"\ncache = ["tiktoken"]\ngenerate = { "share/x.json" = "make.py" }',
    )
    recipe = load_spec(path).install.recipe

    assert recipe is not None and recipe.python == "3.12" and recipe.cache == ("tiktoken",)
    assert recipe.lock == (tmp_path / "my-agent.txt").resolve()
    assert dict(recipe.generate) == {"share/x.json": (tmp_path / "make.py").resolve()}
    hash(load_spec(path).install)  # the runner keys installs by InstallSpec


def test_a_recipe_may_pin_the_patch_version(tmp_path):
    recipe = load_spec(recipe_spec(tmp_path, 'python = "3.12.14"\nlock = "my-agent.txt"')).install.recipe

    assert recipe is not None and recipe.python == "3.12.14"


@pytest.mark.parametrize(
    ("recipe", "error"),
    [
        ('python = "3"\nlock = "my-agent.txt"', "must be a CPython version"),
        ('python = "pypy3.10"\nlock = "my-agent.txt"', "must be a CPython version"),
        ('python = "3.12.14.1"\nlock = "my-agent.txt"', "must be a CPython version"),
        ('python = "3.12."\nlock = "my-agent.txt"', "must be a CPython version"),
        ('python = "3.12"\nlock = "missing.txt"', "'missing.txt' not found"),
        ('python = "3.12"\nlock = "my-agent.txt"\ncache = ["pip"]', r"unknown cache\(s\) \['pip'\]"),
        ('python = "3.12"\nlock = "my-agent.txt"\ngenerate = { "/abs" = "make.py" }', "relative path inside"),
        ('python = "3.12"\nlock = "my-agent.txt"\ngenerate = { "a/../../b" = "make.py" }', "relative path inside"),
        ('python = "3.12"\nlock = "my-agent.txt"\ngenerate = { "x" = "nope.py" }', "'nope.py' not found"),
        ('python = "3.12"\nlock = "my-agent.txt"\nvenv = true', r"unknown key\(s\) \['venv'\]"),
        ('lock = "my-agent.txt"', "missing required key 'python'"),
    ],
)
def test_a_bad_recipe_fails_when_the_spec_loads(tmp_path, recipe, error):
    with pytest.raises(HarnessSpecError, match=error):
        load_spec(recipe_spec(tmp_path, recipe))


def test_a_recipe_needs_a_toolchain(tmp_path):
    (tmp_path / "lock.txt").write_text("")
    with pytest.raises(HarnessSpecError, match="needs \\[install\\] toolchain"):
        parse_spec(minimal(install={"recipe": {"python": "3.12", "lock": "lock.txt"}}), base_dir=tmp_path)


def test_resources_parse_with_defaults():
    assert parse_spec(minimal()).resources == ResourcesSpec(memory_mb=None)
    assert parse_spec(minimal(resources={"memory_mb": 2048})).resources.memory_mb == 2048
    for bad, error in (
        ({"resources": {"memory_mb": 1.5}}, "whole number"),
        ({"resources": {"memory_mb": 0}}, "positive number"),
        ({"resources": {"cpus": 2}}, r"unknown key\(s\) \['cpus'\]"),
    ):
        with pytest.raises(HarnessSpecError, match=error):
            parse_spec(minimal(**bad))


def test_budget_enforcement_defaults_to_the_harness():
    assert parse_spec(minimal()).budget == BudgetSpec(enforce="harness")
    assert parse_spec(minimal(budget={"enforce": "gateway"})).budget.enforce == "gateway"
    for bad, error in (
        ({"budget": {"enforce": "trainer"}}, "is not one of"),
        ({"budget": {"max_calls": 9}}, r"unknown key\(s\) \['max_calls'\]"),
    ):
        with pytest.raises(HarnessSpecError, match=error):
            parse_spec(minimal(**bad))


def test_openhands_runs_its_runner_from_its_toolchain():
    spec = get_harness_spec("openhands")
    assert spec.ref == "openhands@1.54.0.post2" and get_harness_spec("openhands@1.54").ref == spec.ref
    assert spec.install.toolchain == "openhands-1.54.0" and spec.install.commands == ()
    recipe = spec.install.recipe
    assert recipe is not None and (recipe.python, recipe.cache, dict(recipe.generate)) == ("3.12.14", ("tiktoken",), {})
    assert "openhands-sdk==1.54.0" in recipe.lock.read_text() and "openhands-tools==1.54.0" in recipe.lock.read_text()
    assert (spec.resources.memory_mb, spec.budget.enforce, spec.max_steps) == (2048, "harness", 30)
    assert spec.builtins.tools == ("terminal", "file_editor", "task_tracker", "finish", "think")

    rendered = spec.render(_with(toolchain="/tc/openhands-1.54.0", max_steps=12))

    assert rendered.argv[:3] == ("/tc/openhands-1.54.0/bin/python3", "-I", "/tmp/agent/run_openhands.py")
    assert rendered.argv[rendered.argv.index("--max-iterations") + 1] == "12"
    assert rendered.env["LLM_BASE_URL"] == "http://127.0.0.1:8000/v1" and rendered.env["LLM_MODEL"] == "openai/policy"
    assert rendered.env["LLM_API_KEY"] == "secret-key" and all("secret-key" not in arg for arg in rendered.argv)
    assert rendered.env["TIKTOKEN_CACHE_DIR"] == "/tc/openhands-1.54.0/share/tiktoken"
    assert (
        rendered.assets["run_openhands.py"].name == "run_openhands.py" and rendered.assets["run_openhands.py"].exists()
    )
    assert rendered.final_message_file == "/tmp/agent/final_message.txt"
    assert [spec.exit_status(code) for code in (0, 1, 3, 4, None)] == [
        "completed",
        "error",
        "completed",
        "completed",
        "error",
    ]


def test_the_built_in_lock_ships_next_to_the_spec():
    assert sorted(path.name for path in SPECS.glob("*.txt")) == ["openhands-1.54.0.txt"]
    for name in ("openhands-1.54.0.txt",):
        lines = (SPECS / name).read_text().splitlines()
        requirements = [line for line in lines if line and not line.startswith((" ", "#"))]
        assert requirements and all("==" in line for line in requirements), name
        assert not any(line.startswith("--index-url") for line in lines), "the index is the site's choice"


@pytest.mark.asyncio
async def test_the_runner_renders_a_named_toolchain_under_its_root(tmp_path):
    toolchains = tmp_path / "toolchains"
    (toolchains / "probe-1").mkdir(parents=True)
    spec = parse_spec(
        minimal(install={"toolchain": "probe-1"}, launch={"argv": ["sh", "-c", "echo tc={toolchain}"]}),
    )
    session = FakeSession(task=Task(id="t", goal="g"), instruction="g", sandbox=ProcessSandbox(tmp_path / "sandbox"))
    runner = HarnessRunner(logs_dir=tmp_path / "logs", sandbox_state_root=str(tmp_path / "state"), kill_grace_s=0.2)
    agent = FakeAgent(agent_id="a", api_key="k", rollout_id="r", trajectory_id="t")

    result = await runner.run(
        spec,
        session=session,
        agent=agent,
        base_url="http://127.0.0.1:1",
        placement="task_sandbox",
        toolchain_root=str(toolchains),
    )

    assert result.exit_code == 0
    assert f"tc={toolchains}/probe-1" in Path(result.logs_path).read_text()
    missing = HarnessRunner(logs_dir=tmp_path / "logs2", sandbox_state_root=str(tmp_path / "state"))
    with pytest.raises(
        HarnessSetupError, match=re.escape(f"toolchain {tmp_path}/elsewhere/probe-1 for toy@1.0 is not")
    ):
        await missing.run(
            spec,
            session=session,
            agent=agent,
            base_url="http://127.0.0.1:1",
            placement="task_sandbox",
            toolchain_root=str(tmp_path / "elsewhere"),
        )


def _with(**changes):
    import dataclasses

    return dataclasses.replace(CONTEXT, **changes)
