"""Configs whose rollouts could never train are refused when a run loads them: exit 78, one line per problem.

A TOML harness spec runs as one agent with no episode program, so on a task table (no grader) every rollout is
invalid, and a program or agents settings never apply; a spec without ``[mcp]`` reaches the env's tools only from
inside a task sandbox, so on a service without one (or placed ``local``) every rollout scores 0.
"""

from __future__ import annotations

import pytest
import yaml

from platoon import cli
from platoon.harness.registry import get_harness
from platoon.train.components import EnvironmentConfig

ROWS = {"rows": [{"id": "t1", "goal": "say hi"}], "splits": {"train": ["t1"], "eval": ["t1"]}}
NUMBER_SEARCH = "python://platoon.envs.number_search.env:NumberSearchEnvironment"


@pytest.fixture(autouse=True)
def tinker_key(monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")  # eval and train serve a policy: past their key check


def _run(tmp_path, command: str, *environments: dict) -> int:
    """Run ``command`` on a config with ``environments`` (named, when more than one) and return its exit code."""

    path = tmp_path / f"{command}.yaml"
    if command == "train":
        config = {
            "train": {
                "model_name": "Qwen/Qwen3-4B",
                "renderer_name": "qwen3",
                "batch_size": 1,
                "max_training_steps": 1,
            },
            "eval": {"every": 0},
            "log_path": str(tmp_path / "logs"),
            "stats": {"experiment_name": "exp", "trial_name": "trial"},
        }
    else:
        config = {
            "inference": {
                "model_name": "fake-model",
                "renderer_name": "qwen3_instruct",
                "output_dir": str(tmp_path / "results"),
            }
        }
    entries = [
        {**environment, "eval_dataset_kwargs": {"split_name": "val"}}  # its splits: train, val
        if environment["service"] == NUMBER_SEARCH
        else environment
        for environment in environments
    ]
    path.write_text(yaml.safe_dump({**config, "environments": entries}))
    argv = [command, str(path)]
    if command == "rollout":
        task = "t1" if entries[0]["service"] == "table" else "val:0"
        if len(entries) > 1:
            task = f"{entries[0]['name']}::{task}"
        argv += ["--task", task, "--output-dir", str(tmp_path / "run")]
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    return exit_info.value.code


REFUSED = [
    (
        {"service": "table", "service_kwargs": ROWS, "harness": "platoon-bash"},
        "harness platoon-bash@1 is a TOML harness spec, which runs no episode program, and a task table has no grader",
    ),
    (
        {"service": NUMBER_SEARCH, "harness": "platoon-bash", "program": "gym"},
        "program 'gym' never runs: harness platoon-bash@1 is a TOML harness spec",
    ),
    (
        {"service": NUMBER_SEARCH, "harness": "platoon-bash", "agents": {"max_depth": 1}},
        "agents settings never apply: harness platoon-bash@1 is a TOML harness spec",
    ),
    (
        {"service": NUMBER_SEARCH, "harness": "mini-swe-agent"},
        f"harness mini-swe-agent@2.4.6 takes no env tools ([mcp] is not enabled), and on {NUMBER_SEARCH}, which has "
        "no task sandbox",
    ),
    (
        {"service": "ors+http://127.0.0.1:9/NumberSearchEnvironment", "harness": "mini-swe-agent"},
        "harness mini-swe-agent@2.4.6 takes no env tools ([mcp] is not enabled)",
    ),
]
REFUSED_IDS = ["toml-on-table", "toml-with-program", "toml-with-agents", "no-mcp-in-process", "no-mcp-remote"]


@pytest.mark.parametrize("command", ["rollout", "eval", "train"])
@pytest.mark.parametrize(("environment", "expected"), REFUSED, ids=REFUSED_IDS)
def test_a_config_that_could_never_train_exits_78(tmp_path, capsys, command, environment, expected):
    assert _run(tmp_path, command, environment) == cli.EXIT_CONFIG

    err = capsys.readouterr().err
    assert expected in err and "environments[default]" in err
    assert "Traceback" not in err


@pytest.mark.parametrize("command", ["rollout", "eval", "train"])
def test_every_entrys_settings_are_refused_together_before_any_task_is_listed(tmp_path, capsys, command):
    """A mixture's harness-setting problems all show at once, as its exclude_tasks problems do, and first: they are
    checked before any task is listed, the exclude_tasks items only while the tasks are listed."""

    table = {"name": "table", "service": "table", "service_kwargs": ROWS, "harness": "platoon-bash"}
    search = {"name": "search", "service": NUMBER_SEARCH, "harness": "platoon-bash", "program": "gym"}

    assert _run(tmp_path, command, {**table, "exclude_tasks": ["nope"]}, search) == cli.EXIT_CONFIG

    err = capsys.readouterr().err
    assert "environments[table]: harness platoon-bash@1 is a TOML harness spec, which runs no episode program" in err
    assert "environments[search]: program 'gym' never runs: harness platoon-bash@1 is a TOML harness spec" in err
    assert "exclude_tasks" not in err and "Traceback" not in err


@pytest.mark.parametrize("command", ["eval", "train"])
def test_exclude_tasks_items_are_checked_once_the_settings_can_train(tmp_path, capsys, monkeypatch, command):
    def reached(*args, **kwargs):
        raise AssertionError("the config was not refused before the run started")

    monkeypatch.setattr("platoon.train.tinker.train.PlatoonTinkerRLTrainer", reached)
    monkeypatch.setattr("platoon.eval.run.EvalRunner", reached)
    table = {"name": "table", "service": "table", "service_kwargs": ROWS, "harness": "platoon-react", "program": "gym"}
    search = {"name": "search", "service": NUMBER_SEARCH, "harness": "platoon-bash"}

    assert _run(tmp_path, command, {**table, "exclude_tasks": ["nope"]}, search) == cli.EXIT_CONFIG

    err = capsys.readouterr().err
    assert "environments[table].exclude_tasks: ['nope'] match no task id or name of the environment" in err
    assert "environments[search]" not in err and "Traceback" not in err


def _problems(**entry) -> list[str]:
    from platoon.train.auto import harness_setting_problems

    environment = EnvironmentConfig(**entry)
    return harness_setting_problems(environment, [get_harness(ref) for ref in environment.harness_refs()])


def test_every_problem_of_an_entry_is_reported():
    problems = _problems(
        service="table", service_kwargs=ROWS, harness="mini-swe-agent", program="gym", agents={"max_depth": 1}
    )

    assert len(problems) == 3  # the table, the program, the agents; the table already explains why it cannot act


def test_a_spec_without_mcp_placed_local_on_harbor_is_refused_and_in_its_sandbox_is_not():
    (local,) = _problems(service="harbor", harness="mini-swe-agent", placement="local")
    assert "placed local its own tools never reach the environment" in local
    # Harbor has no no_submit: the task's tests grade the sandbox as the agent left it, untouched.
    assert "the task's tests would grade a sandbox it never acted in" in local and "no_submit" not in local

    assert _problems(service="harbor", harness="mini-swe-agent") == []  # placement auto: in the task sandbox


@pytest.mark.parametrize(
    "entry",
    [
        {"service": NUMBER_SEARCH, "harness": "platoon-bash"},  # [mcp]: the env's tools over MCP
        {"service": NUMBER_SEARCH, "harness": "platoon-codeact", "program": "gym", "agents": {"max_depth": 1}},
        {"service": "table", "service_kwargs": ROWS, "harness": "platoon-react", "program": "gym"},
        {"service": "harbor", "harness": "platoon-bash", "placement": "local"},
        {"service": "pkg.module:Service", "harness": "mini-swe-agent"},  # a custom service may have a sandbox
        # max_depth 0 says that no agent spawns, as a spec never does: its other settings are moot
        {"service": NUMBER_SEARCH, "harness": "platoon-bash", "agents": {"max_depth": 0, "child_max_steps": 6}},
    ],
    ids=[
        "mcp-spec",
        "function-harness",
        "function-harness-on-table",
        "mcp-spec-local-on-harbor",
        "custom-service",
        "mcp-spec-max-depth-0",
    ],
)
def test_configs_that_can_train_are_not_refused(entry):
    assert _problems(**entry) == []


def test_a_mixture_of_a_function_harness_and_a_spec_takes_no_program_for_the_spec():
    (problem,) = _problems(service=NUMBER_SEARCH, harness={"platoon-codeact": 1, "platoon-bash": 1}, program="gym")
    assert "program 'gym' never runs: harness platoon-bash@1 is a TOML harness spec" in problem


def test_an_entry_with_a_rollout_of_its_own_is_not_checked():
    """A rollout function of the entry's own runs its harness as it likes: only harness_rollout's rules apply here."""

    from platoon.train.auto import harness_setting_problems

    environment = EnvironmentConfig(service="table", service_kwargs=ROWS, harness="mini-swe-agent", program="gym")
    harnesses = [get_harness(ref) for ref in environment.harness_refs()]

    assert len(harness_setting_problems(environment, harnesses)) == 2  # the table and the program
    assert harness_setting_problems(environment, harnesses, harness_rollout=False) == []
