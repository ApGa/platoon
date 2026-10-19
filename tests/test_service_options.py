"""A service's options (an entry's ``service_kwargs``) are bound to what its factory takes when the config loads,
without building the service: an unknown key fails with its path and a did-you-mean, in ``platoon config validate``
and before a run lists any task; the values stay the service's to check, and a value it refuses when the run builds
it is a configuration error that names the entry (RFC 0016, section 4)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from platoon import cli
from platoon.envs.service import listing_option_problems, service_option_problems
from platoon.train.auto import AutoDataset
from platoon.train.components import EnvironmentConfig
from platoon.utils.config import ConfigError

WHERE = "environments.0.service_kwargs"


class AnyOptions:
    """A service factory that takes any option."""

    def __init__(self, **options) -> None:
        self.options = options


def needs_a_root(root: str, *, depth: int = 1) -> AnyOptions:
    """A service factory with a required argument."""

    return AnyOptions(root=root, depth=depth)


def test_an_unknown_option_of_a_built_in_service_fails_with_a_did_you_mean():
    assert service_option_problems("harbor", {"datset_dir": "tasks", "provider": "docker"}, where=WHERE) == [
        f"{WHERE}.datset_dir: unknown key; did you mean {WHERE}.dataset_dir?"
    ]
    assert (
        service_option_problems("harbor", {"dataset": "hf://org/name@abc", "holdout": {"fraction": 2}}, where=WHERE)
        == []
    )
    (problem,) = service_option_problems("table", {"path": "tasks.jsonl", "colour": 1}, where=WHERE)
    assert problem.startswith(f"{WHERE}.colour: unknown key; the table service takes rows, splits, path, name")


def test_what_a_service_reference_gives_is_not_an_option():
    (problem,) = service_option_problems("python://pkg.mod:Env", {"environment": "x"}, where=WHERE)
    assert (
        problem
        == f"{WHERE}.environment: the service reference 'python://pkg.mod:Env' gives it; leave it out of service_kwargs"
    )
    assert service_option_problems("ors+http://host:8080/Env", {"urls": ["http://other:1"]}, where=WHERE) == []
    assert service_option_problems("ors+http://host:8080/Env", {"env_nmae": "x"}, where=WHERE)[0].startswith(
        f"{WHERE}.env_nmae: unknown key"
    )


def test_a_factory_that_takes_any_option_or_does_not_import_is_not_checked():
    assert service_option_problems(f"{__name__}:AnyOptions", {"anything": 1}, where=WHERE) == []
    assert service_option_problems("no.such.module:Service", {"anything": 1}, where=WHERE) == []
    assert service_option_problems(f"{__name__}:needs_a_root", {"depth": 2}, where=WHERE) == [
        f"{WHERE}: missing a required argument: 'root'"
    ]


def test_a_service_listing_takes_three_keys():
    assert listing_option_problems({"split_name": "test", "limit": 8, "shuffle_seed": 1}, where="x") == []
    (problem,) = listing_option_problems({"split": "test"}, where="environments.0.dataset_kwargs")
    assert (
        problem
        == "environments.0.dataset_kwargs.split: unknown key; did you mean environments.0.dataset_kwargs.split_name?"
    )


def _config(tmp_path: Path, **entry) -> Path:
    path = tmp_path / "train.yaml"
    data = {
        "train": {"model_name": "Qwen/Qwen3-4B", "renderer_name": "qwen3", "batch_size": 1},
        "eval": {"every": 0},
        "log_path": str(tmp_path / "logs"),
        "environments": [{"service": "harbor", "harness": "platoon-bash", **entry}],
    }
    path.write_text(yaml.safe_dump(data))
    return path


def test_validate_and_train_refuse_a_misspelled_option_before_any_task_is_listed(tmp_path: Path, capsys, monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")  # train checks its key before anything else: past it
    config = _config(tmp_path, service_kwargs={"datset_dir": str(tmp_path / "tasks")}, dataset_kwargs={"limt": 2})

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["config", "validate", str(config)])
    assert exit_info.value.code == 78
    out = capsys.readouterr().out
    assert f"  {WHERE}.datset_dir: unknown key; did you mean {WHERE}.dataset_dir?" in out
    assert "  environments.0.dataset_kwargs.limt: unknown key; did you mean environments.0.dataset_kwargs.limit?" in out
    cli.main(["config", "validate", str(config), "--no-resolve"])  # options are checked with the components
    assert capsys.readouterr().out.startswith("ok   ")

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["train", str(config)])
    assert exit_info.value.code == 78
    assert f"{WHERE}.datset_dir: unknown key" in capsys.readouterr().err


def test_a_value_the_service_refuses_when_it_is_built_is_a_configuration_error(tmp_path: Path):
    entry = EnvironmentConfig(
        service="harbor", service_kwargs={"dataset_dir": str(tmp_path), "concurrent_tools": ["x"]}
    )
    config = SimpleNamespace(environments=[entry], train=SimpleNamespace(batch_size=None, task_sampler="auto"))

    with pytest.raises(ConfigError, match=r"environments\[default\]\.service: service_kwargs: concurrent_tools names"):
        AutoDataset.from_config(config, "train")
