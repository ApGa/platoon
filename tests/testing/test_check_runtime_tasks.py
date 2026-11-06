"""``platoon.testing.check_runtime_tasks``, the red-team check of a Harbor environment's runtime tasks: its refusal
cases pass on CPU, a ``materialize`` that lets a sabotage through fails the cases it should have refused, by name, and
its grading cases run each task end to end, here on sandboxes whose launcher cannot grade (they give it no root), which
every grading case then names."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import platoon.sandbox
from platoon.envs.harbor import HarborEnvironmentService, runtime
from platoon.sandbox import ExecResult, SandboxSpec
from platoon.sandbox.process import ProcessProvider, ProcessSandbox
from platoon.testing import check_runtime_tasks
from platoon.testing.runtime_tasks import GRADES, REFUSALS, to_toml

TEMPLATE: dict[str, Any] = {
    "environment": {"docker_image": "science", "network_mode": "no-network", "workdir": "/app"},
    "verifier": {"environment_mode": "separate", "user": "root", "timeout_sec": 60},
}


class RootAnsweringProvider(ProcessProvider):
    """The process provider, whose sandboxes say they run as root with other users mapped, as a --fakeroot one does;
    their commands still run as this user."""

    def __init__(self, root: str, refusal: str | None = None) -> None:
        super().__init__(root)
        self.options = {"root": root, "refusal": refusal}

    @classmethod
    def root_refusal(cls, options: Mapping[str, Any]) -> str | None:
        return options.get("refusal")

    async def start(self, spec: SandboxSpec) -> ProcessSandbox:
        sandbox = await super().start(spec)
        real = sandbox.exec

        async def exec(cmd: str | list[str], **options: Any) -> ExecResult:
            if isinstance(cmd, list) and "platoon-root-check" in cmd:
                return ExecResult(0, "uid 0\n" + "".join(f"{k}_map 0 1\n{k}_map 1 65536\n" for k in ("uid", "gid")), "")
            if isinstance(cmd, list) and "platoon-nobody-check" in cmd:
                return ExecResult(0, "ran\n", "")
            return await real(cmd, **options)

        sandbox.exec = exec  # type: ignore[method-assign]
        return sandbox


@pytest.fixture
def options(tmp_path: Path, monkeypatch) -> dict[str, Any]:
    """The service_kwargs of an environment that takes runtime tasks, on the root-answering process provider."""

    monkeypatch.setitem(platoon.sandbox._BUILTIN_PROVIDERS, "root-answering", f"{__name__}:RootAnsweringProvider")
    (tmp_path / "tasks").mkdir()
    return {
        "dataset_dir": str(tmp_path / "tasks"),
        "provider": "root-answering",
        "provider_options": {"root": str(tmp_path / "sandboxes")},
        "agent_user": "ignore",
        "runtime_tasks": {"images": {"science": "local"}, "template": TEMPLATE, "store": str(tmp_path / "store")},
    }


def test_the_refusal_cases_pass_where_materialize_keeps_the_rules(options):
    check_runtime_tasks(options, grade=False)
    check_runtime_tasks(HarborEnvironmentService(**options), grade=False)


def test_a_materialize_that_lets_a_sabotage_through_fails_the_cases_it_should_have_refused(options, monkeypatch):
    monkeypatch.setattr(runtime, "template_differences", lambda config, template: [])  # task.toml goes unchecked

    with pytest.raises(AssertionError) as failed:
        check_runtime_tasks(options, grade=False)

    message = str(failed.value)
    assert "fail 4 red-team case(s)" in message
    for case in ("network", "gpus", "image"):
        assert f"\n  {case}: materialize took it in, as " in message
    # Loading it refuses a shared root verifier whose agents are not honored, but not as the case expects.
    assert "\n  shared_grading: materialize refused it, but no problem matches 'environment_mode'" in message
    assert "accepts:" not in message


def test_a_materialize_that_takes_artifacts_outside_the_working_directory_fails_that_case(options, monkeypatch):
    monkeypatch.setattr(runtime, "artifact_problems", lambda config: [])

    with pytest.raises(AssertionError) as failed:
        check_runtime_tasks(options, grade=False)

    message = str(failed.value)
    assert "fail 1 red-team case(s)" in message and "\n  outside_artifact: materialize took it in, as " in message


def test_the_grading_cases_run_each_task_and_name_every_grade_that_comes_out_otherwise(options):
    """These sandboxes give the launcher no root, so it grades nothing: every grading case says so."""

    with pytest.raises(AssertionError) as failed:
        check_runtime_tasks(options)

    message = str(failed.value)
    assert f"fail {len(GRADES)} red-team case(s)" in message
    for case in GRADES:
        assert f"\n  {case}: graded ('invalid', None, 'verifier_trusted_launcher', None)" in message
    assert "not root" in message


def test_the_grading_cases_need_sandboxes_that_can_run_runtime_tasks(options):
    options["provider_options"]["refusal"] = "no root on this provider"

    with pytest.raises(
        AssertionError, match="sandboxes: the environment's sandboxes cannot run runtime tasks: .*no root"
    ):
        check_runtime_tasks(options)


def test_an_environment_without_runtime_tasks_is_refused(options):
    del options["runtime_tasks"]

    with pytest.raises(AssertionError, match="takes no runtime tasks"):
        check_runtime_tasks(options, grade=False)


def test_the_cases_are_named_once():
    assert len(set(REFUSALS + GRADES)) == len(REFUSALS) + len(GRADES)


def test_the_cases_scripts_are_python_that_compiles():
    """The grading cases' graders and solutions run only on sandboxes that give the launcher root: on CPU, they must at
    least be Python."""

    from platoon.testing import runtime_tasks

    scripts = {f"grader {name}": source for name, source in runtime_tasks._GRADERS.items()}
    for name in ("_HONEST_GRADER", "_FRACTION_GRADER", "_SOLVE", "_PRYING_SOLVE", "_FAILING_SOLVE", "_SHADOW"):
        scripts[name] = getattr(runtime_tasks, name)
    for name, source in scripts.items():
        compile(source, name, "exec")


def test_to_toml_writes_what_tomllib_reads_back():
    table = {
        "version": "1.0",
        "environment": {"cpus": 2.5, "allow": True, "env": {"A B": 'x\n"y"'}, "mcp_servers": [{"name": "s", "n": 1}]},
        "verifier": {"environment": {}, "timeout_sec": 60},
        "artifacts": ["a", "b"],
    }

    assert tomllib.loads(to_toml(table)) == table
