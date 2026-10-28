"""The FrontierSmith recipe (``recipes/frontiersmith``): its configs load, resolve every component they name and
validate, and run the same environment entry (the training check on its 16 problems); its overlay applied to a task
in the adapter's form serves the judge from the node service with the task's problem as the scope and leaves no host
path in the task; the judge front's guard of the engine's go-judge client and its health checks; and the verifier's
readings.  CPU only."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from platoon.envs.harbor.fetch import fetch, fetch_config
from platoon.envs.harbor.node_services import declared_node_services
from platoon.eval.run import PlatoonEvalConfig
from platoon.train.auto import AutoEnvironment
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.utils.config import load_config

RECIPE = Path(__file__).resolve().parent
JUDGE = "frontier-cs-judge"
FETCH_PATH = "/tmp/fetch/repo/Frontier-CS/algorithmic"  # where an adapter run left its checkout


def _entry(config: Any) -> dict[str, Any]:
    [environment] = config.environments
    return {
        "service": environment.service,
        "service_kwargs": environment.service_kwargs,
        "harness": environment.harness,
    }


def test_the_configs_load_validate_and_run_one_environment():
    evaluation, _ = load_config(["--config", str(RECIPE / "eval.yaml")], PlatoonEvalConfig)
    training, _ = load_config(["--config", str(RECIPE / "train-check.yaml")], PlatoonTinkerRLTrainerConfig)
    AutoEnvironment.load(training)
    evaluated, trained = _entry(evaluation), _entry(training)
    assert trained["service_kwargs"].pop("tasks") == [str(RECIPE / "tasks" / "check.json")]
    assert evaluated == trained
    kwargs = trained["service_kwargs"]
    assert kwargs["adapter"]["commit"] == "ac3046cce7a270e3177cbb41376f36e03de1d518"
    assert "--include-non-numeric" in kwargs["adapter"]["args"]
    assert kwargs["overlays"] == [str(RECIPE / "overlays" / "frontiersmith")]
    assert kwargs["network_policy"] == "none"
    [spec] = declared_node_services(kwargs["node_services"], provider="singularity").values()
    assert (spec.name, spec.port, spec.cpus, spec.network) == (JUDGE, 8081, 16, "none")
    assert spec.front.routes == ("POST /evaluate", "GET /health")
    scripts = next(bind for bind in spec.binds if bind.target == "/node")
    assert scripts.source == str(RECIPE / "node-service")  # relative to the config file
    assert all(bind.stage for bind in spec.binds if bind.target == "/app/problems")
    assert len(json.loads((RECIPE / "tasks" / "check.json").read_text())) == 16


def _adapter_task(root: Path, problem: str) -> Path:
    """A task as FrontierSmith's adapter writes it (its files' form, not its content)."""

    task = root / f"frontier-cs-algorithm-{problem}"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "solution").mkdir()
    (task / "instruction.md").write_text("Write /app/solution.cpp.\n")
    (task / "task.toml").write_text(
        f'schema_version = "1.1"\n\n[task]\nname = "frontier-cs/frontier-cs-algorithm-{problem}"\n\n'
        f'[verifier]\ntimeout_sec = 160.0\n\n[verifier.env]\nPROBLEM_ID = "{problem}"\n'
        'JUDGE_URL = "http://judge:8081"\nMAX_POLL_TIME = "130"\n\n[agent]\ntimeout_sec = 10800.0\n\n'
        '[environment]\nnetwork_mode = "public"\ncpus = 2\nmemory_mb = 4096\n\n[environment.env]\n'
        f'FRONTIER_CS_ALGORITHMIC_PATH = "${{FRONTIER_CS_ALGORITHMIC_PATH:-{FETCH_PATH}}}"\n'
    )
    (task / "environment" / "docker-compose.yaml").write_text(
        "services:\n  main:\n    depends_on:\n      judge:\n        condition: service_started\n"
        "    volumes:\n      - ./statement.txt:/app/statement.txt:ro\n\n  judge:\n    build:\n"
        f"      context: ${{FRONTIER_CS_ALGORITHMIC_PATH:-{FETCH_PATH}}}\n    privileged: true\n"
        '    shm_size: "4g"\n    ulimits:\n      stack:\n        soft: -1\n        hard: -1\n'
        '    environment:\n      PORT: "8081"\n    volumes:\n'
        f"      - ${{FRONTIER_CS_ALGORITHMIC_PATH:-{FETCH_PATH}}}/problems/{problem}:/app/problems/{problem}:ro\n"
        "      - ${HOST_ARTIFACTS_PATH}/judge/submissions:/app/submissions\n      - /app/data\n"
    )
    (task / "environment" / "statement.txt").write_text("Print a number.\n")
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\nRUN curl -fsSL https://example.com | bash\n")
    (task / "tests" / "test.sh").write_text("#!/bin/bash\necho 0.0 > /logs/verifier/reward.txt\n")
    (task / "tests" / "evaluate.py").write_text("print('the adapter verifier')\n")
    (task / "solution" / "solve.sh").write_text("#!/bin/bash\nexit 1\n")
    return task


def test_the_overlay_serves_the_judge_from_the_node_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cache = tmp_path / "cache"
    profile = tmp_path / "site.yaml"
    profile.write_text(json.dumps({"cache": str(cache)}))
    monkeypatch.setenv("PLATOON_SITE", str(profile))
    dataset = tmp_path / "dataset"
    _adapter_task(dataset, "frontiersmith_81")
    lock = fetch(
        fetch_config(str(dataset), lock=tmp_path / "lock.json", overlays=[str(RECIPE / "overlays" / "frontiersmith")])
    )
    [task] = lock.tasks
    assert task.refused is None and task.compose is not None
    assert task.compose["services"]["judge"] == {"external": JUDGE, "scope": {"problem_id": "${PROBLEM_ID}"}}
    assert "depends_on" in task.compose["services"]["main"]  # kept: the stack drops it, the node service is up first
    [cached] = (cache / "harbor" / "tasks").iterdir()
    for path in cached.rglob("*"):
        if path.is_file():
            assert FETCH_PATH not in path.read_text(errors="replace"), path
    assert "FRONTIER_CS_ALGORITHMIC_PATH" not in (cached / "task.toml").read_text()
    dockerfile = (cached / "environment" / "Dockerfile").read_text()
    assert "curl -fsSL" not in dockerfile and "g++" in dockerfile
    assert "echo 0.0" not in (cached / "tests" / "test.sh").read_text()
    assert (cached / "tests" / "evaluate.py").read_text() == (
        RECIPE / "overlays" / "frontiersmith" / "files" / "evaluate.py"
    ).read_text()


def _module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_front_guards_the_engine_s_go_judge_client_once(tmp_path: Path):
    front = _module(RECIPE / "node-service" / "judge_front.py", "judge_front")
    client = tmp_path / "gojudge.js"
    client.write_text(
        "export class GoJudgeClient {\n    async runOne(cmd) {\n        const { data } = await post(cmd);\n"
        "        return data[0];\n    }\n\n    async run(cmds) {\n        const { data } = await post(cmds);\n"
        "        return data;\n    }\n\n    // Delete file\n}\n"
    )
    front.guard(str(client))
    guarded = client.read_text()
    assert "return platoonCheckResults(data)[0];" in guarded and "return platoonCheckResults(data);" in guarded
    assert "'Internal Error'" in guarded.replace('"', "'")
    front.guard(str(client))  # once: a second start finds it guarded
    assert client.read_text() == guarded
    unknown = tmp_path / "other.js"
    unknown.write_text("export class GoJudgeClient {}\n")
    with pytest.raises(SystemExit, match="not the go-judge client this front knows"):
        front.guard(str(unknown))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"status": "done", "score": 70.0, "passed": False, "cases": [{"scoreRatio": 1.0}, {"scoreRatio": 0.4}]}, 70.0),
        ({"status": "error", "error": "Error: compile failed: main.cpp:1: error"}, 0.0),
        ({"status": "error", "error": "Error: judge infrastructure: go-judge could not run a program"}, None),
        ({"status": "done", "score": 500.0}, None),  # the unbounded score of a forged Ratio is no grade
    ],
)
def test_the_front_tells_a_grade_from_a_judge_failure(raw: dict, expected: float | None):
    front = _module(RECIPE / "node-service" / "judge_front.py", "judge_front")
    if expected is None:
        with pytest.raises(front.JudgeError):
            front._sanitized(raw)
    else:
        sanitized = front._sanitized(raw)
        assert sanitized["score"] == expected
        if "cases" in raw:  # what a rollout sees of them: how many, and how many it solved; no checker message
            assert (sanitized["cases"], sanitized["perfect_cases"]) == (2, 1)


def test_the_front_s_health_answers_a_pass_for_a_while_and_a_failure_never(monkeypatch: pytest.MonkeyPatch):
    # Every rollout's grading asks for the judge's health: a pass answers the next few seconds' checks, a failure none,
    # so one bad moment counts as one failed check and never as the supervisor's retries in a row.
    front = _module(RECIPE / "node-service" / "judge_front.py", "judge_front")
    found = ["the engine's /problems answered 500", "", "go-judge cannot be reached"]
    checks: list[str] = []

    def check() -> str:
        checks.append(found[len(checks)])
        return checks[-1]

    monkeypatch.setattr(front, "_check", check)
    assert front.health() == "the engine's /problems answered 500"
    assert front.health() == ""  # looked again
    assert front.health() == "" and len(checks) == 2  # the pass answers
    monkeypatch.setattr(front, "_health_passed", front.time.monotonic() - front.HEALTH_CACHE_S)
    assert front.health() == "go-judge cannot be reached" and len(checks) == 3


def test_a_judge_that_cannot_be_reached_makes_the_reading_invalid(tmp_path: Path):
    evaluate = _module(RECIPE / "overlays" / "frontiersmith" / "files" / "evaluate.py", "frontiersmith_evaluate")
    evaluate.SOLUTION = tmp_path / "solution.cpp"
    evaluate.REWARD = tmp_path / "verifier" / "reward.json"
    evaluate.JUDGE_URL = "http://127.0.0.1:9"  # nothing listens there
    evaluate.WAIT_S = 1.0
    evaluate.main()
    assert json.loads(evaluate.REWARD.read_text()) == {"reward": 0.0}  # no solution: the policy's 0
    evaluate.SOLUTION.write_text("int main() {}\n")
    evaluate.main()
    assert json.loads(evaluate.REWARD.read_text()) == {"invalid": 1, "failure_kind": "judge_unreachable"}


def test_the_start_script_is_valid_shell():
    script = RECIPE / "node-service" / "start-judge.sh"
    assert os.access(script, os.X_OK)
    assert subprocess.run(["sh", "-n", str(script)], capture_output=True).returncode == 0
