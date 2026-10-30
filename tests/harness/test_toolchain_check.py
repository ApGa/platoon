"""A run never builds a harness toolchain: it is built once per site (``platoon prepare``, or ``platoon toolchain
build``), and every setting of the harness, every option, runs from it.  Before any rollout, ``platoon train``,
``eval`` and ``rollout`` refuse a config whose spec's toolchain is not built where the run finds it, with the
``platoon prepare`` command for their config."""

from __future__ import annotations

import dataclasses
import json
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml
from site_cache import use_site_cache
from test_harness_spec import CONTEXT
from test_toolchain_build import PYTHON, fake_uv, uv_calls  # noqa: F401 - fake_uv is a fixture

from platoon import cli
from platoon.harness.registry import get_harness_spec
from platoon.harness.spec import load_spec
from platoon.harness.toolchains import (
    MANIFEST,
    build_command,
    build_toolchain,
    read_manifest,
    toolchain_problem,
    toolchain_state,
)
from platoon.prepare import NotPrepared
from platoon.train.auto import check_harness_toolchains
from platoon.train.components import normalize_environment_configs
from platoon.utils.config import ConfigError

OPENHANDS = "platoon toolchain build openhands --root {root}"


def agent_spec(tmp_path: Path, *, lock: str = "pkg==1.0 --hash=sha256:00\n") -> Path:
    """A spec with a recipe and an option that its launch renders."""

    specs = tmp_path / "specs"
    specs.mkdir(exist_ok=True)
    (specs / "agent.txt").write_text(lock)
    path = specs / "agent.toml"
    path.write_text(
        textwrap.dedent(
            f"""\
            name = "agent"
            version = "1.0"
            [install]
            toolchain = "agent-1.0"
            [install.recipe]
            python = "{PYTHON}"
            lock = "agent.txt"
            [options.replay]
            type = "bool"
            default = true
            [options.retries]
            type = "int"
            default = 0
            [launch]
            argv = [
                "{{toolchain}}/bin/python3", "-m", "agent",
                "--replay={{options.replay}}", "--retries={{options.retries}}",
            ]
            """
        )
    )
    return path


def entries(harness: str, **entry) -> list:
    return normalize_environment_configs([{"service": "harbor", "harness": harness, **entry}])


def test_a_toolchain_that_is_not_built_is_refused_with_what_builds_it(tmp_path, monkeypatch):
    use_site_cache(monkeypatch, tmp_path / "site")
    root = tmp_path / "site" / "toolchains"
    spec = get_harness_spec("openhands")
    command = OPENHANDS.format(root=root)

    problem = toolchain_problem(spec, ref="openhands")

    described = (
        "harness openhands@1.54.0.post2 runs from the toolchain openhands-1.54.0, which a run never builds, and it is "
        "not "
        f"built there (toolchain root {root})"
    )
    assert (
        problem == f"{described}: build it once per site with `platoon prepare` on the run's config, or with: {command}"
    )
    with pytest.raises(NotPrepared) as error:  # the command line adds the platoon prepare command for its config
        check_harness_toolchains(entries("openhands"))
    assert error.value.problems == [f"environments[default].harness: {described}"]
    assert str(error.value).endswith("prepare it with: platoon prepare CONFIG")


def test_the_run_looks_under_the_site_profiles_cache_and_the_command_quotes_it(tmp_path, monkeypatch):
    use_site_cache(monkeypatch, tmp_path / "shared site")
    root = tmp_path / "shared site" / "toolchains"

    with pytest.raises(NotPrepared, match=f"it is not built there \\(toolchain root {root}\\)"):
        check_harness_toolchains(entries("openhands@1.54.0"))
    assert build_command("openhands@1.54.0", root) == f"platoon toolchain build openhands@1.54.0 --root '{root}'"


def test_without_a_site_profile_the_run_looks_under_the_cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".cache" / "platoon" / "toolchains"

    problem = toolchain_problem(get_harness_spec("openhands"), ref="openhands")

    assert problem is not None and f"it is not built there (toolchain root {root})" in problem
    assert problem.endswith(OPENHANDS.format(root=root))


def test_a_missing_toolchain_without_a_recipe_is_a_configuration_error(tmp_path, monkeypatch):
    """platoon prepare cannot build a toolchain that its spec gives no recipe for, so a run that misses one is told to
    put it there, as a plain configuration error, not to prepare."""

    use_site_cache(monkeypatch, tmp_path / "site")
    spec = tmp_path / "plain.toml"
    spec.write_text('name = "plain"\nversion = "1"\n[install]\ntoolchain = "plain-1"\n[launch]\nargv = ["true"]\n')
    with pytest.raises(ConfigError) as error:
        check_harness_toolchains(entries(str(spec)))
    assert not isinstance(error.value, NotPrepared)
    assert error.value.problems[0].endswith(f"put it at {tmp_path / 'site' / 'toolchains' / 'plain-1'}")


def test_an_unfinished_build_is_not_a_toolchain(tmp_path):
    (tmp_path / "openhands-1.54.0" / "bin").mkdir(parents=True)  # a killed build: no manifest

    problem = toolchain_problem(get_harness_spec("openhands"), ref="openhands", root=tmp_path)

    assert problem is not None and "an unfinished build is there" in problem
    assert problem.endswith(OPENHANDS.format(root=tmp_path))


def test_a_spec_without_a_toolchain_or_with_a_path_is_never_refused(tmp_path):
    assert toolchain_problem(get_harness_spec("platoon-bash")) is None
    check_harness_toolchains(entries("platoon-bash"))


def test_the_toolchain_is_built_once_and_serves_every_setting(tmp_path, fake_uv):  # noqa: F811
    spec = load_spec(agent_spec(tmp_path))
    root = tmp_path / "toolchains"

    manifest = build_toolchain(spec, root=root, uv=str(fake_uv), log=lambda line: None)
    builds = len(uv_calls(tmp_path))

    for options in ({}, {"replay": False}, {"retries": 5}, {"replay": False, "retries": 3}):
        context = dataclasses.replace(CONTEXT, toolchain=manifest.build_path, options=options)
        rendered = spec.render(context)
        assert rendered.argv[0] == f"{root / 'agent-1.0'}/bin/python3"
        assert rendered.argv[3:] == (
            f"--replay={'false' if options.get('replay') is False else 'true'}",
            f"--retries={options.get('retries', 0)}",
        )
        assert toolchain_problem(spec, ref=str(tmp_path / "specs" / "agent.toml"), root=root) is None
        assert build_toolchain(spec, root=root, uv=str(fake_uv), log=lambda line: None) == manifest  # a no-op
    assert len(uv_calls(tmp_path)) == builds  # nothing built again
    assert read_manifest(root / "agent-1.0") == manifest


def test_a_toolchain_built_from_another_recipe_must_be_rebuilt_by_force(tmp_path, fake_uv, monkeypatch):  # noqa: F811
    """A toolchain built from another recipe may be what a live run of another config, or of another platoon, stages
    its rollouts from, so neither platoon prepare nor a run replaces it: the run stops with a plain configuration error
    that names the forced build, not with the platoon prepare command."""

    use_site_cache(monkeypatch, tmp_path / "site")
    root = tmp_path / "site" / "toolchains"
    build_toolchain(load_spec(agent_spec(tmp_path)), root=root, uv=str(fake_uv), log=lambda line: None)
    changed_path = agent_spec(tmp_path, lock="pkg==2.0 --hash=sha256:11\n")
    changed = load_spec(changed_path)

    problem = toolchain_problem(changed, ref="agent", root=root)

    assert problem is not None and "was built from another recipe (lock sha256" in problem
    assert "a run of another config or platoon may be running it, so platoon prepare leaves it as it is" in problem
    force = f"platoon toolchain build agent --root {root} --force"
    assert problem.endswith(f"rebuild it in place, while no job uses it, with: {force}")
    state = toolchain_state(changed, root=root)
    assert state is not None and state.reason == "stale" and state.buildable and state.needs_force
    with pytest.raises(ConfigError) as error:
        check_harness_toolchains(entries(str(changed_path)))
    assert not isinstance(error.value, NotPrepared)
    assert error.value.problems == [
        f"environments[default].harness: {toolchain_problem(changed, ref=str(changed_path), root=root)}"
    ]
    (root / "agent-1.0" / MANIFEST).write_text("{not json")
    assert "cannot be read" in (toolchain_problem(changed, ref="agent", root=root) or "")
    state = toolchain_state(changed, root=root)
    assert state is not None and state.reason == "unreadable" and state.needs_force


@dataclass(frozen=True)
class FakePolicy:
    """Stands in for the evaluation package's ``EvalPolicy``, whose ``install`` would serve a Tinker sampler."""

    model_name: str
    renderer_name: str
    model_path: str | None = None
    renderer_kwargs: dict = field(default_factory=dict)
    context_window_length: int | None = None

    def install(self) -> None:
        INSTALLED.append(self)


INSTALLED: list[FakePolicy] = []


def test_platoon_rollout_refuses_before_any_rollout(tmp_path, monkeypatch, capsys):
    use_site_cache(monkeypatch, tmp_path / "site")
    monkeypatch.setattr(sys.modules[cli._evaluation().__package__], "EvalPolicy", FakePolicy, raising=False)
    INSTALLED.clear()
    config = tmp_path / "train.yaml"
    entry = {"service": "harbor", "service_kwargs": {"dataset_dir": "/x"}, "harness": "openhands"}
    config.write_text(
        yaml.safe_dump(
            {
                "train": {"model_name": "Qwen/Qwen3-4B", "renderer_name": "qwen3", "batch_size": 1},
                "eval": {"every": 0},
                "log_path": str(tmp_path / "logs"),
                "stats": {"experiment_name": "exp", "trial_name": "trial"},
                "environments": [entry],
            }
        )
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["rollout", str(config), "--task", "t1", "--output-dir", str(tmp_path / "run")])

    assert exit_info.value.code == cli.EXIT_CONFIG
    err = capsys.readouterr().err
    assert f"it is not built there (toolchain root {tmp_path / 'site' / 'toolchains'})" in err
    assert err.rstrip().endswith(f"prepare it with: platoon prepare {config}")  # the config the command ran
    assert INSTALLED == []  # refused before the policy was served, so before any rollout
    lock = json.loads((tmp_path / "run" / "lock.json").read_text())
    (harness,) = [component for component in lock["components"] if component["kind"] == "harness"]
    assert harness["options"] == {
        "default": {
            "train": {
                "reasoning_replay": True,
                "request_timeout_s": 3600,
                "num_retries": 0,
                "condenser": "none",
                "condenser_max_tokens": 0,
                "allow_token_count_estimate": False,
            }
        }
    }


def test_platoon_rollout_records_the_options_of_the_split_it_runs(tmp_path, monkeypatch, capsys):
    """``--split test`` runs an evaluation rollout, with the entry's ``eval_overrides``, though the training config
    itself never evaluates: lock.json records those options, not training's."""

    use_site_cache(monkeypatch, tmp_path / "site")
    monkeypatch.setattr(sys.modules[cli._evaluation().__package__], "EvalPolicy", FakePolicy, raising=False)
    config = tmp_path / "train.yaml"
    entry = {
        "service": "harbor",
        "service_kwargs": {"dataset_dir": "/x"},
        "harness": "openhands",
        "eval_overrides": {"harness_options": {"reasoning_replay": False, "num_retries": 5}},
    }
    config.write_text(
        yaml.safe_dump(
            {
                "train": {"model_name": "Qwen/Qwen3-4B", "renderer_name": "qwen3", "batch_size": 1},
                "eval": {"every": 0},
                "log_path": str(tmp_path / "logs"),
                "stats": {"experiment_name": "exp", "trial_name": "trial"},
                "environments": [entry],
            }
        )
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["rollout", str(config), "--task", "t1", "--split", "test", "--output-dir", str(tmp_path / "run")])

    assert exit_info.value.code == cli.EXIT_CONFIG  # the toolchain is not built, as above
    lock = json.loads((tmp_path / "run" / "lock.json").read_text())
    (harness,) = [component for component in lock["components"] if component["kind"] == "harness"]
    assert harness["options"] == {
        "default": {
            "eval": {
                "reasoning_replay": False,
                "request_timeout_s": 3600,
                "num_retries": 5,
                "condenser": "none",
                "condenser_max_tokens": 0,
                "allow_token_count_estimate": False,
            }
        }
    }
