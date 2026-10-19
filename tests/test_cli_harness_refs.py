"""A run config whose harness cannot be resolved, or cannot run as configured, is a configuration error: the run
commands exit 78 with one line that names the environment, not 1 with a traceback (a restart wrapper retries
exit 1, and no restart fixes a config), and so does ``config validate``, which resolves the harness as they do.  So
is a reference that does not load: a spec file that does not exist, a ``module:function`` whose module does not
import or has no such function, and an entry's ``package`` that does not import."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from platoon import cli

ROWS = {"rows": [{"id": "t1", "goal": "say hi"}], "splits": {"train": ["t1"], "eval": ["t1"]}}


@pytest.fixture(autouse=True)
def tinker_key(monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")  # eval and train serve a policy: past their key check


def _environment(harness: str, **extra) -> dict:
    return {"service": "table", "service_kwargs": ROWS, "harness": harness, **extra}


def _argv(tmp_path, command: str, environment: dict) -> list[str]:
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
    path.write_text(yaml.safe_dump({**config, "environments": [environment]}))
    if command == "rollout":
        return [command, str(path), "--task", "t1", "--output-dir", str(tmp_path / "run")]
    return [command, str(path)]


def _run(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    return exit_info.value.code


def _bad_spec(tmp_path) -> str:
    path = tmp_path / "bad.toml"
    path.write_text('name = "bad"\nversion = "1"\n[launch]\nargv = ["agent", "{no_such_placeholder}"]\n')
    return str(path)


def _missing_spec(tmp_path) -> str:
    return str(tmp_path / "missing.toml")


# References that do not load, with what the one line says about each.
UNLOADABLE = [
    (_missing_spec, "[Errno 2] No such file or directory"),
    ("no_such_package.agents:run", "No module named 'no_such_package'"),
    ("platoon.task:no_such_harness", "module 'platoon.task' has no attribute 'no_such_harness'"),
]
UNLOADABLE_IDS = ["missing-spec-file", "unimportable-module", "missing-function"]


@pytest.mark.parametrize("command", ["rollout", "eval", "train"])
@pytest.mark.parametrize(
    ("harness", "expected"),
    [
        ("platoon-codeactt", "environments[default].harness: Unknown harness 'platoon-codeactt'"),
        ("platoon.task:Task", "environments[default].harness: 'platoon.task:Task' is not an @harness function"),
        (_bad_spec, "environments[default].harness: "),
        *((ref, f"environments[default].harness: {message}") for ref, message in UNLOADABLE),
    ],
    ids=["unknown-name", "not-a-harness", "malformed-spec", *UNLOADABLE_IDS],
)
def test_a_harness_that_does_not_resolve_exits_78_naming_the_environment(tmp_path, capsys, command, harness, expected):
    ref = harness(tmp_path) if callable(harness) else harness

    assert _run(_argv(tmp_path, command, _environment(ref))) == cli.EXIT_CONFIG

    err = capsys.readouterr().err
    assert expected in err
    assert "Traceback" not in err


@pytest.mark.parametrize("command", ["eval", "train"])
def test_config_validate_resolves_the_harness_unless_told_not_to(tmp_path, capsys, command):
    path = _argv(tmp_path, command, _environment("platoon-codeactt"))[1]

    assert _run(["config", "validate", path]) == cli.EXIT_CONFIG
    out = capsys.readouterr().out
    assert out.startswith(f"FAIL {path}\n  environments[default].harness: Unknown harness 'platoon-codeactt'")
    cli.main(["config", "validate", "--no-resolve", path])  # the files and named environments only
    assert capsys.readouterr().out.startswith(f"ok   {path} (")
    assert _run(["config", "validate", "--resolve", path]) == cli.EXIT_CONFIG  # the default, spelled out


def test_config_validate_resolves_nothing_for_a_report_only_evaluation(tmp_path, capsys):
    """platoon eval builds a report from finished rollouts without importing a component (stage: report), so
    config validate does not ask for them either."""

    path = Path(_argv(tmp_path, "eval", _environment("platoon-codeactt"))[1])
    path.write_text(yaml.safe_dump({**yaml.safe_load(path.read_text()), "stage": "report"}))

    cli.main(["config", "validate", str(path)])

    assert capsys.readouterr().out == f"ok   {path} (PlatoonEvalConfig)\n"


NO_PACKAGE = (
    "environments[default].package: 'no_such_package_xyz' does not import "
    "(ModuleNotFoundError: No module named 'no_such_package_xyz')"
)


@pytest.mark.parametrize("command", ["rollout", "eval", "train"])
def test_a_package_that_does_not_import_exits_78_naming_the_environment(tmp_path, capsys, command):
    """A run imports each entry's package before it resolves the entry's components, which the package registers;
    one that does not import is the entry's problem, and its components are not reported unknown besides."""

    environment = _environment("acme-agent", package="no_such_package_xyz")

    assert _run(_argv(tmp_path, command, environment)) == cli.EXIT_CONFIG

    err = capsys.readouterr().err
    assert NO_PACKAGE in err
    assert "Unknown harness" not in err and "Traceback" not in err


def test_config_validate_reports_a_package_that_does_not_import_and_goes_on(tmp_path, capsys):
    bad = _argv(tmp_path, "train", _environment("acme-agent", package="no_such_package_xyz"))[1]
    (tmp_path / "good").mkdir()
    good = _argv(tmp_path / "good", "eval", _environment("platoon-codeact"))[1]

    assert _run(["config", "validate", bad, good]) == cli.EXIT_CONFIG

    assert capsys.readouterr().out == f"FAIL {bad}\n  {NO_PACKAGE}\nok   {good} (PlatoonEvalConfig)\n"


def test_config_validate_fails_an_evaluation_config_with_no_environments(tmp_path, capsys):
    """An evaluation config may leave environments out, as a layer for others to extend, but a run of it alone
    would have nothing to evaluate."""

    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump({"inference": {"model_name": "fake-model"}}))

    assert _run(["config", "validate", str(path)]) == cli.EXIT_CONFIG
    out = capsys.readouterr().out
    assert out == f"FAIL {path}\n  Config must define at least one environment in `environments`\n"
    cli.main(["config", "validate", "--no-resolve", str(path)])  # the file alone is valid
    assert capsys.readouterr().out == f"ok   {path} (PlatoonEvalConfig)\n"


@pytest.mark.parametrize("command", ["rollout", "eval"])
def test_disabling_built_ins_a_harness_cannot_disable_exits_78(tmp_path, capsys, command):
    environment = _environment("platoon-bash", builtins="disable")

    assert _run(_argv(tmp_path, command, environment)) == cli.EXIT_CONFIG

    err = capsys.readouterr().err
    assert "environments[default] disables built-in tools, which harness platoon-bash@1 cannot do" in err
    assert "Traceback" not in err


@pytest.mark.parametrize(("harness", "expected"), UNLOADABLE, ids=UNLOADABLE_IDS)
def test_harness_show_of_a_reference_that_does_not_load_exits_78(tmp_path, capsys, harness, expected):
    ref = harness(tmp_path) if callable(harness) else harness

    assert _run(["harness", "show", ref]) == cli.EXIT_CONFIG

    err = capsys.readouterr().err
    assert expected in err
    assert "Traceback" not in err
