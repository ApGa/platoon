"""Input paths resolve against the config file that names them (RFC 0018, section 5): a harness ``.toml`` spec and
the options a service declares as naming files, made absolute per file before the files merge, with no fallback to
the working directory; overrides and outputs stay relative to the working directory, and the loaded data holds every
path absolute."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from platoon.envs.service import PathOption, service_path_options
from platoon.utils.config import ConfigError, load_yaml_config, resolve_config, write_run_record

COMMIT = "f216b86ca1a18d44433d6cb45f70bec81f47c66e"


def _write(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _kwargs(data: dict[str, Any], index: int = 0) -> dict[str, Any]:
    return data["environments"][index]["service_kwargs"]


def test_a_files_input_paths_resolve_against_its_directory_from_any_working_directory(tmp_path, monkeypatch):
    configs = tmp_path / "project" / "configs"
    _dirs(configs / "tasks", configs / "more" / "one", tmp_path / "data" / "shared")
    config = _write(
        configs / "env.yaml",
        {
            "environments": [
                {
                    "service": "harbor",
                    "service_kwargs": {
                        "dataset_dir": "tasks",
                        "mounts": [
                            {"source": "../../data/shared", "target": "/data"},
                            {"source": "/abs/elsewhere", "target": "/other"},
                        ],
                    },
                },
                {"service": "harbor", "service_kwargs": {"task_dirs": ["more/one"]}},
            ]
        },
    )
    monkeypatch.chdir(tmp_path)  # neither the project nor the configs directory

    data = resolve_config(config).data

    assert _kwargs(data)["dataset_dir"] == str(configs / "tasks")
    assert _kwargs(data)["mounts"] == [
        {"source": str(tmp_path / "data" / "shared"), "target": "/data"},
        {"source": "/abs/elsewhere", "target": "/other"},
    ]
    assert _kwargs(data, 1)["task_dirs"] == [str(configs / "more" / "one")]


def test_each_file_of_an_extends_chain_resolves_its_own_paths(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    _dirs(tmp_path / "datasets" / "ds", shared / "overlays" / "fix")
    (shared / "lists").mkdir()
    (shared / "lists" / "a.json").write_text("[]")
    _write(
        shared / "env.yaml",
        {
            "environments": [
                {
                    "service": "harbor",
                    "service_kwargs": {
                        "dataset": "../datasets/ds",
                        "overlays": ["overlays/fix"],
                        "tasks": "lists/a.json",
                        "lock": "locks/ds.lock.json",  # a fetch writes it: it need not exist yet
                    },
                }
            ]
        },
    )
    run = _write(tmp_path / "runs" / "train.yaml", {"extends": "../shared/env.yaml", "log_path": "out"})
    monkeypatch.chdir(tmp_path / "runs")

    data = resolve_config(run).data

    assert _kwargs(data) == {
        "dataset": str(tmp_path / "datasets" / "ds"),
        "overlays": [str(shared / "overlays" / "fix")],
        "tasks": str(shared / "lists" / "a.json"),
        "lock": str(shared / "locks" / "ds.lock.json"),
    }
    assert data["log_path"] == "out"  # an output: as written, relative to the working directory


def test_an_overrides_paths_and_the_outputs_resolve_against_the_working_directory(tmp_path, monkeypatch):
    """I-R3: an explicit relative log_path keeps its base, the working directory, and so does an override; the run's
    record holds the outputs absolute."""

    configs = tmp_path / "configs"
    _dirs(configs, tmp_path / "work" / "tasks")
    config = _write(
        configs / "eval.yaml",
        {"inference": {"model_name": "m", "output_dir": "results"}, "log_path": "./runs", "environments": []},
    )
    monkeypatch.chdir(tmp_path / "work")

    resolved = resolve_config(
        config,
        ["environments+={service: harbor, service_kwargs: {dataset_dir: tasks}}", "inference.output_dir=out/eval"],
    )
    record = yaml.safe_load(write_run_record(tmp_path / "record", resolved).config_path.read_text())

    assert _kwargs(resolved.data)["dataset_dir"] == str(tmp_path / "work" / "tasks")
    assert (resolved.data["inference"]["output_dir"], resolved.data["log_path"]) == ("out/eval", "./runs")
    assert record["inference"]["output_dir"] == str(tmp_path / "work" / "out" / "eval")
    assert record["log_path"] == str(tmp_path / "work" / "runs")


def test_a_relative_input_that_names_nothing_beside_its_file_is_an_error_naming_the_working_directorys(
    tmp_path, monkeypatch
):
    configs = tmp_path / "configs"
    _dirs(configs, tmp_path / "tasks")  # where the old rule found it: the working directory
    config = _write(
        configs / "env.yaml", {"environments": [{"service": "harbor", "service_kwargs": {"dataset_dir": "tasks"}}]}
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigError) as caught:
        resolve_config(config)

    message = str(caught.value)
    assert str(config) in message
    assert (
        f"environments.0.service_kwargs.dataset_dir: tasks names {configs / 'tasks'}, which does not exist: a relative "
        f"path in a config file is relative to the file's directory, {configs}. {tmp_path / 'tasks'} exists, relative "
        "to the working directory, where relative paths resolved before: write the path relative to env.yaml, or "
        "absolute"
    ) in message


def test_a_relative_input_that_names_nothing_anywhere_is_an_error(tmp_path, monkeypatch):
    config = _write(
        tmp_path / "configs" / "env.yaml",
        {"environments": [{"service": "table", "service_kwargs": {"path": "tasks.jsonl"}, "harness": "spec.toml"}]},
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigError) as caught:
        load_yaml_config(config)

    problems = caught.value.problems
    assert len(problems) == 2
    assert problems[0].startswith(f"environments.0.harness: spec.toml names {tmp_path / 'configs' / 'spec.toml'}")
    assert problems[1].startswith(f"environments.0.service_kwargs.path: tasks.jsonl names {tmp_path / 'configs'}")
    assert all("relative to the working directory" not in problem for problem in problems)


def test_a_lock_a_fetch_writes_may_not_exist_unless_the_working_directory_holds_it(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    _write(tmp_path / "locks" / "x.lock.json", {})
    entry = {"service": "harbor", "service_kwargs": {"dataset": f"hf://org/name@{COMMIT}", "lock": "locks/x.lock.json"}}
    config = _write(configs / "env.yaml", {"environments": [entry]})
    monkeypatch.chdir(configs)  # the lock is written beside the file: nothing there yet, nothing in the cwd

    assert _kwargs(resolve_config(config).data)["lock"] == str(configs / "locks" / "x.lock.json")

    monkeypatch.chdir(tmp_path)  # the old rule's lock exists here: the path changed meaning
    with pytest.raises(ConfigError, match=r"service_kwargs\.lock: locks/x\.lock\.json names .* exists, relative to"):
        resolve_config(config)


def test_a_split_manifest_stays_relative_to_the_dataset_directory_that_holds_it(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    _dirs(configs / "tasks", tmp_path / "datasets" / "ds")
    _write(configs / "splits.json", {"train": ["a"]})
    config = _write(
        configs / "env.yaml",
        {
            "environments": [
                {"service": "harbor", "service_kwargs": {"dataset_dir": "tasks", "splits": "splits.json"}},
                {"service": "harbor", "service_kwargs": {"dataset": "../datasets/ds", "splits": "splits.json"}},
                {"service": "harbor", "service_kwargs": {"dataset_dir": "tasks", "splits": {"eval": ["a"]}}},
            ]
        },
    )
    monkeypatch.chdir(tmp_path)

    data = resolve_config(config).data

    assert _kwargs(data, 0)["splits"] == "splits.json"  # inside dataset_dir, as before
    assert _kwargs(data, 1)["splits"] == str(configs / "splits.json")
    assert _kwargs(data, 2)["splits"] == {"eval": ["a"]}  # an inline manifest is no path


def test_a_split_manifest_is_relative_to_a_dataset_directory_only_one_the_same_file_names(tmp_path, monkeypatch):
    """The loader resolves a file's paths before the overrides apply, so a manifest relative to a dataset_dir that only
    an override gives cannot be told from one relative to the file: the error says so."""

    _dirs(tmp_path / "data" / "tasks")
    _write(tmp_path / "data" / "tasks" / "splits.json", {"train": ["a"]})
    config = _write(
        tmp_path / "configs" / "env.yaml",
        {"environments": [{"service": "harbor", "service_kwargs": {"splits": "splits.json"}}]},
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigError) as caught:
        resolve_config(config, [f"environments.0.service_kwargs.dataset_dir={tmp_path / 'data' / 'tasks'}"])

    assert caught.value.problems == [
        f"environments.0.service_kwargs.splits: splits.json names {tmp_path / 'configs' / 'splits.json'}, which does "
        f"not exist: a relative path in a config file is relative to the file's directory, {tmp_path / 'configs'} (or "
        "to dataset_dir, when the same file names one)"
    ]


def test_values_that_are_no_local_paths_stay_as_they_are(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "registry.json").write_text("[]")
    registry_url = "https://example.com/registry.json"
    config = _write(
        configs / "env.yaml",
        {
            "environments": [
                {"service": "harbor", "service_kwargs": {"dataset": f"hf://org/name@{COMMIT}"}},
                {"service": "harbor", "service_kwargs": {"dataset": "harbor://org/name@1.0"}},
                {"service": "no.such.module:Service", "service_kwargs": {"dataset_dir": "anything"}},
                {"service": "harbor", "service_kwargs": {"dataset_dir": "~/datasets/ds"}, "harness": "platoon-bash"},
                {"service": "harbor", "service_kwargs": {"dataset": "harbor://kumo@1.0", "registry": registry_url}},
                {"service": "harbor", "service_kwargs": {"dataset": "harbor://kumo@1.0", "registry": "registry.json"}},
            ]
        },
    )
    monkeypatch.chdir(tmp_path)

    data = resolve_config(config).data

    assert _kwargs(data, 0)["dataset"] == f"hf://org/name@{COMMIT}"
    assert _kwargs(data, 1)["dataset"] == "harbor://org/name@1.0"
    assert _kwargs(data, 2)["dataset_dir"] == "anything"  # a service that does not import: its own error later
    assert _kwargs(data, 3)["dataset_dir"] == os.path.expanduser("~/datasets/ds")
    assert data["environments"][3]["harness"] == "platoon-bash"
    assert _kwargs(data, 4)["registry"] == registry_url
    assert _kwargs(data, 5)["registry"] == str(configs / "registry.json")  # a legacy registry file, like any input


def test_an_adapters_local_repository_is_relative_to_the_config_file(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    (configs / "adapters" / "repo").mkdir(parents=True)
    repos = ["adapters/repo", "https://example.com/org/repo.git", "git@example.com:org/repo.git", "~/repo"]
    config = _write(
        configs / "env.yaml",
        {
            "environments": [
                {"service": "harbor", "service_kwargs": {"adapter": {"repo": repo, "commit": COMMIT}}} for repo in repos
            ]
        },
    )
    monkeypatch.chdir(tmp_path)

    data = resolve_config(config).data

    assert _kwargs(data, 0)["adapter"] == {"repo": str(configs / "adapters" / "repo"), "commit": COMMIT}
    assert _kwargs(data, 1)["adapter"]["repo"] == "https://example.com/org/repo.git"
    assert _kwargs(data, 2)["adapter"]["repo"] == "git@example.com:org/repo.git"  # git's address, no path
    assert _kwargs(data, 3)["adapter"]["repo"] == os.path.expanduser("~/repo")
    missing = _write(
        configs / "missing.yaml",
        {"environments": [{"service": "harbor", "service_kwargs": {"adapter": {"repo": "nowhere", "commit": COMMIT}}}]},
    )
    with pytest.raises(ConfigError, match=r"environments\.0\.service_kwargs\.adapter\.repo: nowhere names"):
        resolve_config(missing)


def test_harness_spec_paths_resolve_in_a_reference_and_in_a_weighted_mapping(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "a.toml").write_text("")
    (configs / "b.toml").write_text("")
    config = _write(
        configs / "env.yaml",
        {
            "environments": [
                {"service": "table", "service_kwargs": {"rows": ["hi"]}, "harness": "a.toml"},
                {"service": "table", "service_kwargs": {"rows": ["hi"]}, "harness": {"b.toml": 1, "platoon-react": 2}},
            ]
        },
    )
    monkeypatch.chdir(tmp_path)

    data = resolve_config(config).data

    assert data["environments"][0]["harness"] == str(configs / "a.toml")
    assert data["environments"][1]["harness"] == {str(configs / "b.toml"): 1, "platoon-react": 2}


def test_a_named_environments_service_decides_which_options_are_paths(tmp_path, monkeypatch):
    """An entry that selects a named environment (``env:``) names no service of its own: the named environment's
    service declares the paths the entry's own service_kwargs give."""

    class Named:
        def entry(self) -> dict[str, Any]:
            return {"service": "harbor"}

    monkeypatch.setattr("platoon.envs.named.get_environment", lambda reference: Named())
    configs = tmp_path / "configs"
    _dirs(configs / "tasks")
    config = _write(
        configs / "env.yaml", {"environments": [{"env": "x@1", "service_kwargs": {"dataset_dir": "tasks"}}]}
    )
    monkeypatch.chdir(tmp_path)

    assert _kwargs(load_yaml_config(config))["dataset_dir"] == str(configs / "tasks")


def test_a_run_record_holds_absolute_paths_and_reloads_from_anywhere(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    _dirs(configs / "tasks")
    config = _write(
        configs / "env.yaml",
        {"environments": [{"service": "harbor", "service_kwargs": {"dataset_dir": "tasks"}}], "log_path": "runs"},
    )
    monkeypatch.chdir(tmp_path)
    resolved = resolve_config(config)
    record = write_run_record(tmp_path / "runs" / "one", resolved)

    monkeypatch.chdir(configs / "tasks")
    reloaded = resolve_config(record.config_path).data

    environment = resolved.data["environments"][0]
    named = {**environment, "service_kwargs": {**environment["service_kwargs"], "provider": "docker"}}
    # The record names the provider the run started with, the service's default here.
    assert reloaded == {**resolved.data, "log_path": str(tmp_path / "runs"), "environments": [named]}
    assert _kwargs(reloaded)["dataset_dir"] == str(configs / "tasks")


def test_services_declare_their_path_options():
    harbor = service_path_options("harbor")
    assert set(harbor) == {
        "dataset",
        "registry",
        "adapter.repo",
        "dataset_dir",
        "task_dirs",
        "splits",
        "lock",
        "tasks",
        "overlays",
        "mounts[].source",
    }
    assert harbor["splits"] == PathOption(relative_to="dataset_dir")
    assert harbor["lock"] == PathOption(written=True)
    assert service_path_options("table") == {"path": PathOption()}
    assert service_path_options("ors+http:///Env") == {}
    assert service_path_options("python://platoon.envs.number_search.env:NumberSearchEnvironment") == {}
    assert service_path_options("no.such.module:Service") == {}
