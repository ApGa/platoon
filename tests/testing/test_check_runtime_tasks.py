"""``platoon.testing.check_runtime_tasks``, the red-team check of a Harbor environment's runtime tasks: its refusal
cases pass on CPU, a ``materialize`` that lets a sabotage through fails the cases it should have refused, by name, and
its grading cases run each task end to end, here on sandboxes whose launcher cannot grade (they give it no root), which
every grading case then names."""

from __future__ import annotations

import asyncio
import tomllib
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import platoon.sandbox
from platoon.envs.harbor import HarborEnvironmentService, runtime
from platoon.envs.harbor.runtime import MaterializedTask
from platoon.envs.service import Outcome
from platoon.sandbox import ExecResult, SandboxSpec
from platoon.sandbox.process import ProcessProvider, ProcessSandbox
from platoon.task import Task
from platoon.testing import check_runtime_tasks
from platoon.testing import runtime_tasks as red_team
from platoon.testing.runtime_tasks import FLAKY_GRADES, GRADES, REFUSALS, to_toml

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


def test_a_materialize_that_lets_a_variant_replace_the_submission_fails_that_case(options, monkeypatch):
    monkeypatch.setattr(runtime, "_overlay_problem", lambda task_dir, variant, protected: None)

    with pytest.raises(AssertionError) as failed:
        check_runtime_tasks(options, grade=False)

    message = str(failed.value)
    assert "fail 1 red-team case(s)" in message and "\n  variant_overlay: materialize took it in, as " in message


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


class _GradedInTurn:
    """A service and its one session whose grades are ``grades`` in turn, the last from then on: what the
    ``flaky_grader`` case sees of an environment."""

    def __init__(self, grades: list[Outcome], regrade: int = 1) -> None:
        self.grades, self.graded = grades, 0
        self.runtime_tasks = SimpleNamespace(regrade=regrade, template=TEMPLATE)

    async def materialize(self, source: Path, *, grader_owner: str) -> MaterializedTask:
        assert grader_owner == "generated" and (source / "tests" / "gen" / "grade.py").is_file()
        return MaterializedTask(task=Task(id="0" * 64), digest="0" * 64)

    async def open(self, task: Task, *, rollout_id: str) -> Any:
        return self

    async def run_solution(self) -> ExecResult:
        return ExecResult(0, "", "")

    async def grade(self) -> Outcome:
        self.graded += 1
        return self.grades[min(self.graded, len(self.grades)) - 1]

    async def close(self) -> None:
        pass


def _agreed(reward: float) -> Outcome:
    """A score that every regrade gave again."""

    return Outcome(status="scored", rewards={"reward": reward}, metadata={"regrades": [{"reward": reward}]})


def _flaky_case(tmp_path: Path, service: _GradedInTurn) -> str | None:
    """What the ``flaky_grader`` case says of ``service``: None where it passes, else why not."""

    try:
        asyncio.run(red_team._graded(service, "flaky_grader", red_team._Sources(tmp_path, service.runtime_tasks)))
    except red_team._Failed as failed:
        return str(failed)
    return None


def test_the_flaky_case_grades_one_submission_again_until_a_regrade_disagrees(tmp_path):
    flaky = Outcome.infra("verifier_flaky", status="invalid", reason="1, then 0")
    service = _GradedInTurn([_agreed(1.0), _agreed(0.0), flaky])

    assert _flaky_case(tmp_path, service) is None and service.graded == 3


def test_the_flaky_case_fails_where_no_regrade_ever_disagrees(tmp_path):
    service = _GradedInTurn([_agreed(1.0)])

    failed = _flaky_case(tmp_path, service)

    assert service.graded == FLAKY_GRADES
    assert failed == (
        f"never ('invalid', None, 'verifier_flaky', None) in {FLAKY_GRADES} grades of one submission, each a score "
        "that every regrade gave again (the last ('scored', 1.0, None, None))"
    )


def test_the_flaky_case_fails_where_no_generated_graders_score_is_graded_again(tmp_path):
    service = _GradedInTurn([Outcome(status="scored", rewards={"reward": 1.0})], regrade=0)

    assert _flaky_case(tmp_path, service) == (
        "graded ('scored', 1.0, None, None): runtime_tasks.regrade is 0, so no generated grader's score is graded again"
    )
    assert service.graded == 1


@pytest.mark.parametrize(
    ("grade", "got"),
    [
        (Outcome.infra("verifier_start", reason="no sandbox"), "('error', None, 'verifier_start', None)"),
        (
            # A regrade whose replay wrote other outputs and got another score: a score, but no agreed one.
            Outcome(
                status="scored",
                rewards={"reward": 0.0},
                reason="solver_nonreproducible",
                metadata={"regrades": [{"reward": 1.0}]},
            ),
            "('scored', 0.0, None, 'solver_nonreproducible')",
        ),
    ],
    ids=["infrastructure-fault", "solvers-zero-of-a-regrade"],
)
def test_the_flaky_case_stops_at_a_grade_that_is_no_agreed_score(tmp_path, grade, got):
    service = _GradedInTurn([_agreed(1.0), grade])

    assert _flaky_case(tmp_path, service) == f"graded {got}, not ('invalid', None, 'verifier_flaky', None): "
    assert service.graded == 2


def test_the_cases_are_named_once():
    assert len(set(REFUSALS + GRADES)) == len(REFUSALS) + len(GRADES)


def test_the_cases_scripts_are_python_that_compiles():
    """The grading cases' graders and solutions run only on sandboxes that give the launcher root: on CPU, they must at
    least be Python."""

    from platoon.testing import runtime_tasks

    scripts = {f"grader {name}": source for name, source in runtime_tasks._GRADERS.items()}
    names = ("_HONEST_GRADER", "_FRACTION_GRADER", "_SOLVE", "_PRYING_SOLVE", "_FAILING_SOLVE", "_SHADOW")
    for name in (*names, "_IGNORING_SOLVE", "_HARDCODED_SOLVE", "_SEALED_SOLVE"):
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
