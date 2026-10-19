"""The site profile (``platoon.site``, RFC 0018 section 4): where a site keeps platoon's files and which isolating
runtime starts its sandboxes, read once per site, laid under what configs leave out, never able to weaken isolation
(I-R2) or move a run directory a config names (I-R3), and recorded with each run."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

import pytest
import yaml

from platoon import cli
from platoon.deprecations import PlatoonDeprecationWarning
from platoon.site import (
    PROVIDER_OPTIONS,
    SiteProfile,
    SiteProfileError,
    SiteUse,
    cache_root,
    check_run_directory,
    load_profile,
    profile_path,
    site_profile,
)
from platoon.train import reaping
from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig, evaluates
from platoon.utils.config import ConfigError, resolve_config, write_run_record

TRAIN = {"model_name": "Qwen/Qwen3-4B", "renderer_name": "qwen3", "batch_size": 4}


def _write(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _profile(tmp_path: Path, monkeypatch, **values: Any) -> Path:
    """A site profile with ``values``, named in ``PLATOON_SITE``."""

    path = _write(tmp_path / "site" / "site.yaml", values)
    monkeypatch.setenv("PLATOON_SITE", str(path))
    return path


def _harbor(**kwargs: Any) -> dict[str, Any]:
    return {"service": "harbor", "service_kwargs": {"task_dirs": [], **kwargs}}


def _exit_code(argv: list[str]) -> int | str | None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    return exit_info.value.code


def _use(resolved: Any) -> SiteUse:
    """What the site profile gave a resolved config."""

    assert resolved.site is not None
    return resolved.site


# --- which file ----------------------------------------------------------------------------------------------------


def test_the_profile_is_platoon_site_else_the_xdg_config_file_else_none(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert profile_path() is None
    assert site_profile() == SiteProfile()

    default = _write(tmp_path / "xdg" / "platoon" / "site.yaml", {"cache": "/c"})
    assert profile_path() == default
    assert site_profile().cache == Path("/c")

    named = _write(tmp_path / "elsewhere.yaml", {"cache": "/d"})
    monkeypatch.setenv("PLATOON_SITE", str(named))
    assert profile_path() == named and site_profile().cache == Path("/d")
    monkeypatch.setenv("PLATOON_SITE", "")  # the empty string counts as unset
    assert profile_path() == default


def test_a_platoon_site_that_names_nothing_is_a_configuration_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PLATOON_SITE", str(tmp_path / "missing.yaml"))

    with pytest.raises(SiteProfileError, match=r"\$PLATOON_SITE names .*missing\.yaml, which is not a file"):
        site_profile()
    assert issubclass(SiteProfileError, ConfigError)  # exit 78


def test_the_profile_is_read_again_when_it_changes(tmp_path, monkeypatch):
    path = _profile(tmp_path, monkeypatch, cache="/one")
    assert cache_root() == Path("/one")
    path.write_text("cache: /second-root\n")  # another size, so another key even within one mtime tick
    assert cache_root() == Path("/second-root")


# --- what it holds -------------------------------------------------------------------------------------------------


def test_values_expand_home_and_variables_and_paths_are_relative_to_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("GROUP_ROOT", "/shared/g1")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    path = _write(
        tmp_path / "site" / "site.yaml",
        {
            "cache": "${GROUP_ROOT}/$USER-cache",
            "runs": "../runs",
            "provider": "singularity",
            "provider_options": {
                "allow_network_none": True,
                "image_cache": "~/images",
                "rootfs_layout": "fuse-overlay",
            },
        },
    )
    monkeypatch.setenv("USER", "alice")

    profile = load_profile(path)

    assert profile.cache == Path("/shared/g1/alice-cache")
    assert profile.runs == tmp_path / "runs"
    assert profile.provider == "singularity"
    assert profile.provider_options == {
        "allow_network_none": True,
        "image_cache": str(tmp_path / "home" / "images"),
        "rootfs_layout": "fuse-overlay",
    }
    assert profile.sha256 and len(profile.sha256) == 64
    assert profile.record()["path"] == str(path)


def test_a_profile_takes_four_keys_and_says_what_it_is_for(tmp_path):
    path = _write(tmp_path / "site.yaml", {"cahce": "/c", "model_name": "m", "runs": "/r"})

    with pytest.raises(SiteProfileError) as caught:
        load_profile(path)

    problems = caught.value.problems
    assert problems[0] == "cahce: unknown key; did you mean cache?"
    assert problems[1].startswith("model_name: unknown key; a site profile takes cache, runs, provider and")
    assert "never what trains" in problems[1]
    assert str(path) in str(caught.value)


def test_an_unset_variable_is_an_error_naming_it(tmp_path, monkeypatch):
    monkeypatch.delenv("NO_SUCH_VARIABLE", raising=False)
    path = _write(tmp_path / "site.yaml", {"runs": "$NO_SUCH_VARIABLE/runs"})

    with pytest.raises(SiteProfileError, match=r"runs: \$NO_SUCH_VARIABLE is not set"):
        load_profile(path)


def test_the_profile_refuses_the_process_provider_and_options_outside_the_allowlist(tmp_path):
    """I-R2: a site file may name only a provider that isolates its sandboxes, and only options that do not weaken
    the isolation; binds and extra runtime arguments belong in a run config."""

    path = _write(
        tmp_path / "site.yaml",
        {"provider": "process", "provider_options": {"extra_start_args": ["--bind", "/data"], "binary": "/x"}},
    )

    with pytest.raises(SiteProfileError) as caught:
        load_profile(path)

    problems = caught.value.problems
    assert problems[0].startswith("provider: process runs the agents' commands on this host, without isolation")
    assert problems[1].startswith("provider_options.extra_start_args: a site profile sets only the options that")
    assert "binds, extra runtime arguments and the runtime's executable belong in a run config" in problems[1]
    assert problems[2].startswith("provider_options.binary:")
    assert PROVIDER_OPTIONS == (
        "allow_network_none",
        "rootfs_layout",
        "scratch_root",
        "image_cache",
        "stage_images",
        "gpu",
    )


class UndeclaredProvider:
    """A provider class that does not say whether it isolates its sandboxes."""


class IsolatingProvider:
    isolates = True


def test_the_profile_names_only_a_provider_class_that_says_it_isolates(tmp_path):
    for name in ("docker", "singularity", "enroot", f"{__name__}:IsolatingProvider"):
        assert load_profile(_write(tmp_path / "ok.yaml", {"provider": name})).provider == name
    with pytest.raises(SiteProfileError, match="does not say that it isolates its sandboxes"):
        load_profile(_write(tmp_path / "undeclared.yaml", {"provider": f"{__name__}:UndeclaredProvider"}))
    with pytest.raises(SiteProfileError, match=r"'no\.such:Provider' does not resolve"):
        load_profile(_write(tmp_path / "missing.yaml", {"provider": "no.such:Provider"}))


def test_provider_options_need_a_provider_and_the_right_kinds_of_values(tmp_path):
    with pytest.raises(SiteProfileError, match="options for no provider"):
        load_profile(_write(tmp_path / "a.yaml", {"provider_options": {"allow_network_none": True}}))
    with pytest.raises(SiteProfileError, match=r"allow_network_none: 'yes' is not true or false"):
        load_profile(
            _write(tmp_path / "b.yaml", {"provider": "singularity", "provider_options": {"allow_network_none": "yes"}})
        )


def test_an_empty_profile_sets_nothing(tmp_path, monkeypatch):
    path = tmp_path / "site.yaml"
    path.write_text("# nothing here yet\n")
    monkeypatch.setenv("PLATOON_SITE", str(path))

    assert site_profile().values == {}
    assert site_profile().describe() == f"site profile {path}: sets nothing"
    assert cache_root() == Path("~/.cache/platoon").expanduser()


# --- the cache root ------------------------------------------------------------------------------------------------


def test_the_cache_root_is_the_profiles_and_holds_the_harbor_cache_and_the_toolchains(tmp_path, monkeypatch):
    from platoon.envs.harbor.lock import default_cache
    from platoon.harness.toolchains import toolchain_root

    assert cache_root() == Path("~/.cache/platoon").expanduser()
    _profile(tmp_path, monkeypatch, cache=str(tmp_path / "cache"))

    assert cache_root() == tmp_path / "cache"
    assert default_cache() == tmp_path / "cache" / "harbor"
    assert toolchain_root() == tmp_path / "cache" / "toolchains"


def test_the_deprecated_cache_variables_still_work_and_warn(tmp_path, monkeypatch):
    from platoon.envs.harbor.lock import default_cache

    config = _write(tmp_path / "train.yaml", {"train": TRAIN, "log_path": str(tmp_path / "runs")})
    monkeypatch.setenv("PLATOON_HARBOR_CACHE", str(tmp_path / "old-cache"))

    with pytest.warns(PlatoonDeprecationWarning) as caught:
        resolved = resolve_config(config, (), PlatoonTinkerRLTrainerConfig)

    assert default_cache() == tmp_path / "old-cache"
    assert [deprecation.path for deprecation in resolved.deprecations] == ["$PLATOON_HARBOR_CACHE"]
    assert "use the site profile's cache, whose harbor directory it names" in str(caught[0].message)
    assert "platoon 0.3.0 removes it" in str(caught[0].message)


def test_a_cache_variable_that_disagrees_with_the_profile_is_an_error(tmp_path, monkeypatch):
    config = _write(tmp_path / "train.yaml", {"train": TRAIN, "log_path": str(tmp_path / "runs")})
    _profile(tmp_path, monkeypatch, cache=str(tmp_path / "cache"))
    (tmp_path / "cache" / "harbor").mkdir(parents=True)
    monkeypatch.setenv("PLATOON_HARBOR_CACHE", str(tmp_path / "elsewhere"))

    with pytest.raises(ConfigError, match=r"\$PLATOON_HARBOR_CACHE names .*elsewhere, but the site profile"):
        resolve_config(config, (), PlatoonTinkerRLTrainerConfig, warn=False)

    (tmp_path / "link").symlink_to(tmp_path / "cache" / "harbor")  # the same directory by another path: no conflict
    monkeypatch.setenv("PLATOON_HARBOR_CACHE", str(tmp_path / "link"))
    resolve_config(config, (), PlatoonTinkerRLTrainerConfig, warn=False)


# --- what it gives a config ----------------------------------------------------------------------------------------


def test_runs_gives_a_config_without_log_path_its_run_directory_named_after_the_file(tmp_path, monkeypatch):
    _profile(tmp_path, monkeypatch, runs=str(tmp_path / "platoon-runs"))
    config = _write(tmp_path / "configs" / "train-qwen.yaml", {"train": TRAIN})

    resolved = resolve_config(config, (), PlatoonTinkerRLTrainerConfig)

    assert resolved.config.log_path == str(tmp_path / "platoon-runs")
    assert resolved.config.stats.experiment_name == "train-qwen"
    assert dict(_use(resolved).gave) == {
        "log_path": str(tmp_path / "platoon-runs"),
        "stats.experiment_name": "train-qwen",
    }
    named = resolve_config(config, ["stats.experiment_name=mine"], PlatoonTinkerRLTrainerConfig)
    assert (named.config.log_path, named.config.stats.experiment_name) == (str(tmp_path / "platoon-runs"), "mine")


def test_an_explicit_relative_log_path_keeps_its_base_the_working_directory(tmp_path, monkeypatch):
    """I-R3: the profile's runs never moves a run directory that a config names: an explicit relative log_path stays
    relative to the working directory, and its experiment name stays the default."""

    _profile(tmp_path, monkeypatch, runs=str(tmp_path / "platoon-runs"))
    config = _write(tmp_path / "configs" / "train.yaml", {"train": TRAIN, "log_path": "./runs"})
    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path / "work")

    for resolved in (
        resolve_config(config, (), PlatoonTinkerRLTrainerConfig),
        resolve_config(
            _write(tmp_path / "configs" / "other.yaml", {"train": TRAIN}),
            ["log_path=runs"],
            PlatoonTinkerRLTrainerConfig,
        ),
    ):
        assert Path(resolved.config.log_path).resolve() == tmp_path / "work" / "runs"  # as written: ./runs or runs
        assert resolved.config.stats.experiment_name == "platoon_tinker"
        assert "log_path" not in _use(resolved).gave
        record = write_run_record(tmp_path / "record", resolved).config_path
        assert yaml.safe_load(record.read_text())["log_path"] == str(tmp_path / "work" / "runs")


def test_without_runs_a_config_still_needs_its_log_path(tmp_path):
    config = _write(tmp_path / "train.yaml", {"train": TRAIN})

    with pytest.raises(ConfigError) as caught:
        resolve_config(config, (), PlatoonTinkerRLTrainerConfig, site=SiteProfile())

    assert caught.value.problems == [
        "log_path: required key is missing; set it, or runs in the site profile (~/.config/platoon/site.yaml, or the "
        "file $PLATOON_SITE names), which gives a config without log_path its run directory"
    ]


def test_runs_holds_a_new_directory_for_each_start_of_an_evaluation_without_an_output_directory(tmp_path, monkeypatch):
    """An evaluation's directory of its own for each start goes under the profile's runs: <runs>/eval/<policy>/<UTC
    time>.  The config keeps no output_dir when it loads, and the directory is named after the policy, not the file."""

    from datetime import datetime, timezone

    from platoon.eval.run import PlatoonEvalConfig, prepare_output_dir

    _profile(tmp_path, monkeypatch, runs=str(tmp_path / "platoon-runs"))
    config = _write(tmp_path / "eval-base.yaml", {"inference": {"model_name": "Qwen/Qwen3-4B"}})
    now = datetime(2026, 10, 19, 1, 2, 3, tzinfo=timezone.utc)

    resolved = resolve_config(config, (), PlatoonEvalConfig)
    assert resolved.config.inference.output_dir is None
    assert dict(_use(resolved).gave) == {} and _use(resolved).named_after is None
    first = prepare_output_dir(resolved.config, runs=_use(resolved).profile.runs, now=now)
    assert first == resolved.config.inference.output_dir
    assert Path(first) == tmp_path / "platoon-runs" / "eval" / "Qwen-Qwen3-4B" / "20261019T010203Z"
    again = resolve_config(config, (), PlatoonEvalConfig).config
    assert prepare_output_dir(again, runs=_use(resolved).profile.runs, now=now) == first + "-2"
    explicit = resolve_config(config, ["inference.output_dir=here"], PlatoonEvalConfig)
    assert explicit.config.inference.output_dir == "here"  # relative to the working directory, as written


def test_a_run_directory_the_profile_names_after_a_file_name_belongs_to_that_file(tmp_path, monkeypatch, capsys):
    """Two projects' train.yaml would share <runs>/train/run, and the second would resume the first's run as its own:
    the run records the file its directory is named after, and a start of another file of that name is refused."""

    _profile(tmp_path, monkeypatch, runs=str(tmp_path / "platoon-runs"))
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")  # train checks its key first, before it writes anything
    monkeypatch.setattr("platoon.train.supervise.run_training", lambda *args: 78)  # each start stops after its record
    first, second = (
        _write(tmp_path / project / "train.yaml", {"train": {**TRAIN, "max_training_steps": 1}})
        for project in ("project-a", "project-b")
    )
    run = tmp_path / "platoon-runs" / "train" / "run"

    assert _exit_code(["train", str(first)]) == 78
    capsys.readouterr()
    assert json.loads((run / "lock.json").read_text())["site"]["named_after"] == str(first.resolve())

    assert _exit_code(["train", str(second)]) == 78
    assert (
        f"{run} holds the training run of {first.resolve()}: the site profile names a training run's directory after "
        f"its config file's name, and {second.resolve()} has the same name, so it would take that run for its own; "
        "name this config's experiment (stats.experiment_name=...), or give it a log_path"
    ) in capsys.readouterr().err
    assert json.loads((run / "lock.json").read_text())["site"]["named_after"] == str(first.resolve())  # as it was

    resolved = resolve_config(second, ["stats.experiment_name=train"], PlatoonTinkerRLTrainerConfig, warn=False)
    check_run_directory(run, _use(resolved))  # a name the config gives itself is its own choice
    check_run_directory(run, _use(resolve_config(first, (), PlatoonTinkerRLTrainerConfig, warn=False)))  # a resume


def test_an_evaluations_directory_is_never_named_after_its_file(tmp_path, monkeypatch):
    """Two projects' eval.yaml under one profile: each start has a directory of its own, named after the policy, so
    the record names no file and neither is refused, as another project's train.yaml of one name would be."""

    from platoon.eval.run import PlatoonEvalConfig, prepare_output_dir

    _profile(tmp_path, monkeypatch, runs=str(tmp_path / "platoon-runs"))
    first, second = (
        _write(tmp_path / project / "eval.yaml", {"inference": {"model_name": "m"}})
        for project in ("project-a", "project-b")
    )
    outputs = []
    for config in (first, second):
        resolved = resolve_config(config, (), PlatoonEvalConfig, warn=False)
        assert _use(resolved).named_after is None
        output = Path(prepare_output_dir(resolved.config, runs=_use(resolved).profile.runs))
        check_run_directory(output, _use(resolved))  # nothing to refuse
        record = write_run_record(output, resolved)
        assert "named_after" not in json.loads(record.lock_path.read_text())["site"]
        outputs.append(output)
    assert outputs[0] != outputs[1]
    assert {output.parent for output in outputs} == {tmp_path / "platoon-runs" / "eval" / "m"}


def test_an_output_under_home_is_where_the_run_reads_its_record(tmp_path, monkeypatch):
    """A leading ~ in log_path or inference.output_dir names the home directory where the command writes the run's
    record and where the run it starts reads it, never a directory named ~."""

    from platoon.eval.run import PlatoonEvalConfig

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    train = _write(tmp_path / "train.yaml", {"train": TRAIN, "log_path": "~/runs"})
    resolved = resolve_config(train, (), PlatoonTinkerRLTrainerConfig, warn=False)
    assert resolved.config.log_path == str(tmp_path / "home" / "runs")
    record = write_run_record(Path(resolved.config.log_path) / "e" / "t", resolved)
    assert resolve_config(record.config_path, (), PlatoonTinkerRLTrainerConfig).config.log_path == str(
        tmp_path / "home" / "runs"
    )

    evaluation = _write(tmp_path / "eval.yaml", {"inference": {"model_name": "m", "output_dir": "~/evals/x"}})
    loaded = resolve_config(evaluation, (), PlatoonEvalConfig, warn=False).config
    assert loaded.inference.output_dir == str(tmp_path / "home" / "evals" / "x")
    assert not (tmp_path / "~").exists()


def test_a_harbor_entry_without_a_provider_takes_the_profiles_and_lays_its_options_over(tmp_path, monkeypatch):
    profile = _profile(
        tmp_path,
        monkeypatch,
        provider="singularity",
        provider_options={"allow_network_none": True, "rootfs_layout": "fuse-overlay"},
    )
    config = _write(
        tmp_path / "train.yaml",
        {
            "train": TRAIN,
            "log_path": str(tmp_path / "runs"),
            "environments": [
                {"name": "plain", **_harbor()},
                {
                    "name": "binds",
                    **_harbor(provider_options={"extra_start_args": ["--bind", "/d"], "rootfs_layout": "copy"}),
                },
                {"name": "own", **_harbor(provider="docker", provider_options={"binary": "podman"})},
                {"name": "table", "service": "table", "service_kwargs": {"rows": ["hi"]}},
            ],
        },
    )

    resolved = resolve_config(config, (), PlatoonTinkerRLTrainerConfig)
    kwargs = [entry["service_kwargs"] for entry in resolved.data["environments"]]

    assert kwargs[0]["provider"] == "singularity"
    assert kwargs[0]["provider_options"] == {"allow_network_none": True, "rootfs_layout": "fuse-overlay"}
    assert kwargs[1]["provider"] == "singularity"
    assert kwargs[1]["provider_options"] == {
        "allow_network_none": True,
        "rootfs_layout": "copy",  # the run config's own value wins
        "extra_start_args": ["--bind", "/d"],
    }
    assert kwargs[2] == {"task_dirs": [], "provider": "docker", "provider_options": {"binary": "podman"}}
    assert kwargs[3] == {"rows": ["hi"]}
    choices = {choice.entry: choice for choice in _use(resolved).providers}
    assert [choice.source for choice in _use(resolved).providers] == ["site", "site", "config"]
    assert choices["environments.1 (binds)"].option_sources == {
        "allow_network_none": "site",
        "rootfs_layout": "config",
        "extra_start_args": "config",
    }

    record = write_run_record(tmp_path / "run", resolved)
    lock = json.loads(record.lock_path.read_text())
    assert lock["site"]["path"] == str(profile) and len(lock["site"]["sha256"]) == 64
    assert lock["site"]["provider"] == "singularity"
    assert lock["site"]["cache_root_source"] == "default"
    assert [(item["environment"], item["provider"], item["source"]) for item in lock["sandbox_providers"]] == [
        ("environments.0 (plain)", "singularity", "site"),
        ("environments.1 (binds)", "singularity", "site"),
        ("environments.2 (own)", "docker", "config"),
    ]
    header = record.config_path.read_text().splitlines()
    assert f"# site profile: {profile} (sha256 {lock['site']['sha256']})" in header
    assert "# from the site profile: environments.0.service_kwargs.provider" in header

    monkeypatch.delenv("PLATOON_SITE")  # the record loads the same without the profile
    assert resolve_config(record.config_path, (), PlatoonTinkerRLTrainerConfig).data == resolved.data


def test_without_a_profile_a_harbor_entry_keeps_the_services_default(tmp_path):
    config = _write(
        tmp_path / "train.yaml",
        {
            "train": TRAIN,
            "log_path": str(tmp_path),
            "environments": [{"name": "a", **_harbor()}, {"name": "b", "service": "harbor"}],
        },
    )

    resolved = resolve_config(config, (), PlatoonTinkerRLTrainerConfig)

    assert "provider" not in resolved.data["environments"][0]["service_kwargs"]
    assert "service_kwargs" not in resolved.data["environments"][1]  # nothing given, nothing added
    assert [(choice.provider, choice.source) for choice in _use(resolved).providers] == [("docker", "default")] * 2


def test_sandbox_reap_takes_the_profiles_provider_for_a_record_that_names_none(tmp_path, monkeypatch):
    environments = [_harbor(), _harbor(provider="enroot")]
    assert reaping.sandbox_providers(environments) == [("docker", {}), ("enroot", {})]

    _profile(tmp_path, monkeypatch, provider="singularity", provider_options={"scratch_root": "/local"})
    assert reaping.sandbox_providers(environments) == [("singularity", {"scratch_root": "/local"}), ("enroot", {})]


def test_the_record_names_the_provider_a_run_started_with_so_a_later_profile_cannot_change_it(tmp_path, monkeypatch):
    config = _write(
        tmp_path / "train.yaml",
        {"train": TRAIN, "log_path": str(tmp_path / "runs"), "environments": [{"name": "h", **_harbor()}]},
    )
    resolved = resolve_config(config, (), PlatoonTinkerRLTrainerConfig)
    assert "provider" not in resolved.data["environments"][0]["service_kwargs"]  # the loaded config gets nothing
    record = write_run_record(tmp_path / "run", resolved)
    snapshot = yaml.safe_load(record.config_path.read_text())
    assert snapshot["environments"][0]["service_kwargs"]["provider"] == "docker"  # the record names what ran

    _profile(tmp_path, monkeypatch, provider="singularity")  # written while the run goes on
    reloaded = resolve_config(record.config_path, (), PlatoonTinkerRLTrainerConfig)  # a supervised restart
    assert reloaded.data["environments"][0]["service_kwargs"]["provider"] == "docker"
    assert reaping.sandbox_providers(snapshot["environments"]) == [("docker", {})]


def test_a_null_provider_names_none(tmp_path, monkeypatch):
    config = _write(
        tmp_path / "train.yaml",
        {"train": TRAIN, "log_path": str(tmp_path), "environments": [{"name": "h", **_harbor(provider=None)}]},
    )

    plain = resolve_config(config, (), PlatoonTinkerRLTrainerConfig)
    assert "provider" not in plain.data["environments"][0]["service_kwargs"]  # the service's default, not None
    assert [(choice.provider, choice.source) for choice in _use(plain).providers] == [("docker", "default")]

    _profile(tmp_path, monkeypatch, provider="singularity")
    given = resolve_config(config, (), PlatoonTinkerRLTrainerConfig)
    assert given.data["environments"][0]["service_kwargs"]["provider"] == "singularity"
    assert [(choice.provider, choice.source) for choice in _use(given).providers] == [("singularity", "site")]


def test_a_harbor_service_classs_own_default_provider_is_recorded_and_checked(tmp_path, monkeypatch):
    """A subclass of the Harbor service that starts its sandboxes on the host by default is a provider the run config
    does not name: recorded as such, and refused as platoon train refuses a named environment's."""

    import sys
    import types

    from platoon.envs.harbor.service import HarborEnvironmentService

    class HostHarbor(HarborEnvironmentService):
        def __init__(self, *args: Any, provider: str = "process", **kwargs: Any) -> None:
            super().__init__(*args, provider=provider, **kwargs)

    module = types.ModuleType("host_harbor_service")
    setattr(module, "HostHarbor", HostHarbor)  # an attribute that ModuleType does not declare
    monkeypatch.setitem(sys.modules, "host_harbor_service", module)
    config = _write(
        tmp_path / "train.yaml",
        {
            "train": TRAIN,
            "log_path": str(tmp_path),
            "environments": [{"service": "host_harbor_service:HostHarbor", "service_kwargs": {"task_dirs": []}}],
        },
    )

    resolved = resolve_config(config, (), PlatoonTinkerRLTrainerConfig)
    assert [(choice.provider, choice.source) for choice in _use(resolved).providers] == [("process", "default")]
    assert [choice.provider for choice in _use(resolved).unisolated()] == ["process"]


def test_config_validate_names_the_site_profile_a_problem_is_in(tmp_path, monkeypatch, capsys):
    profile = _profile(tmp_path, monkeypatch, provider="process")
    config = _write(tmp_path / "train.yaml", {"train": TRAIN, "log_path": str(tmp_path)})

    assert _exit_code(["config", "validate", str(config)]) == 78

    assert (
        f"FAIL {config}\n  {profile}: provider: process runs the agents' commands on this host, without isolation"
    ) in capsys.readouterr().out


# --- the deprecated entry keys -------------------------------------------------------------------------------------


def test_a_harbor_entrys_cache_image_map_and_toolchain_root_are_deprecated(tmp_path, capsys):
    config = _write(
        tmp_path / "train.yaml",
        {
            "train": TRAIN,
            "log_path": str(tmp_path),
            "environments": [_harbor(cache="c", image_map="m.json", toolchain_root="t")],
        },
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PlatoonDeprecationWarning)
        resolved = resolve_config(config, (), PlatoonTinkerRLTrainerConfig)

    assert [deprecation.path for deprecation in resolved.deprecations] == [
        "environments.0.service_kwargs.cache",
        "environments.0.service_kwargs.image_map",
        "environments.0.service_kwargs.toolchain_root",
    ]
    assert resolved.deprecations[0].message == (
        "environments.0.service_kwargs.cache: deprecated; use the site profile's cache (its harbor directory) instead "
        "(platoon 0.3.0 removes it)"
    )
    assert len([item for item in caught if issubclass(item.category, PlatoonDeprecationWarning)]) == 3
    assert resolved.data["environments"][0]["service_kwargs"]["cache"] == "c"  # still the working directory's

    assert _exit_code(["config", "validate", "--strict", str(config)]) == 78
    assert "environments.0.service_kwargs.image_map: deprecated" in capsys.readouterr().out


# --- the commands --------------------------------------------------------------------------------------------------


def _train_config(tmp_path: Path, **extra: Any) -> Path:
    return _write(
        tmp_path / "configs" / "train-site.yaml",
        {"train": {**TRAIN, "max_training_steps": 1}, "environments": [{"name": "h", **_harbor()}], **extra},
    )


def test_platoon_train_names_the_profile_and_records_it_in_the_run(tmp_path, monkeypatch, capsys):
    profile = _profile(tmp_path, monkeypatch, runs=str(tmp_path / "platoon-runs"), provider="singularity")
    monkeypatch.setattr("platoon.sandbox.singularity.SingularityProvider.unavailable", classmethod(lambda cls, o: None))
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")  # train checks its key first, before it writes anything
    monkeypatch.setattr("platoon.train.supervise.run_training", lambda *args: 78)  # each start stops after its record
    config = _train_config(tmp_path)

    assert _exit_code(["train", str(config)]) == 78

    err = capsys.readouterr().err
    assert f"platoon: site profile {profile}: runs {tmp_path / 'platoon-runs'}, provider singularity\n" in err
    assert (
        f"platoon: from the site profile: log_path={tmp_path / 'platoon-runs'}, stats.experiment_name=train-site, "
        "environments.0.service_kwargs.provider=singularity\n"
    ) in err
    run = tmp_path / "platoon-runs" / "train-site" / "run"
    lock = json.loads((run / "lock.json").read_text())
    assert lock["site"]["gave"]["stats.experiment_name"] == "train-site"
    assert lock["sandbox_providers"][0]["provider"] == "singularity"


def test_platoon_train_refuses_a_profile_provider_this_node_cannot_run(tmp_path, monkeypatch, capsys):
    profile = _profile(tmp_path, monkeypatch, provider="singularity", provider_options={"allow_network_none": True})
    monkeypatch.setattr(
        "platoon.sandbox.singularity.SingularityProvider.unavailable",
        classmethod(lambda cls, options: "no singularity or apptainer executable is on PATH"),
    )
    config = _train_config(tmp_path, log_path=str(tmp_path / "runs"))

    assert _exit_code(["train", str(config)]) == 78

    err = capsys.readouterr().err
    assert (
        f"environments.0 (h): the site profile {profile} chooses the sandbox provider singularity, which this node "
        "cannot run: no singularity or apptainer executable is on PATH. The profile says:\n"
        "      provider: singularity\n"
        "      provider_options: {allow_network_none: true}\n"
    ) in err
    assert not (tmp_path / "runs").exists()  # refused before anything is written


def test_platoon_train_refuses_an_unisolated_provider_the_run_config_does_not_name(tmp_path, monkeypatch, capsys):
    """I-R2: a named environment's entry cannot put a run on the process provider; the run config itself can."""

    from platoon.envs import named

    class Named:
        ref = "unsafe@1"

        def entry(self) -> dict[str, Any]:
            return {"service": "harbor", "service_kwargs": {"task_dirs": [], "provider": "process"}}

    monkeypatch.setattr(
        named,
        "expand_environment",
        lambda entry: {
            **Named().entry(),
            **{key: value for key, value in entry.items() if key != "env"},
            "env": "unsafe@1",
        },
    )
    monkeypatch.setattr(named, "get_environment", lambda reference: Named())
    config = _write(
        tmp_path / "train.yaml",
        {
            "train": {**TRAIN, "max_training_steps": 1},
            "log_path": str(tmp_path / "runs"),
            "environments": [{"name": "u", "env": "unsafe@1"}],
        },
    )

    assert _exit_code(["train", str(config)]) == 78
    err = capsys.readouterr().err
    assert (
        "environments.0 (u): its sandbox provider, process, runs the agents' commands on this host without isolation, "
        "and it comes from the named environment the entry selects (env:), not the run config"
    ) in err

    resolved = resolve_config(
        config, ["environments.0.service_kwargs.provider=process"], PlatoonTinkerRLTrainerConfig, warn=False
    )
    assert _use(resolved).unisolated() == []  # named by the run config: allowed


def test_platoon_prepare_names_the_profile_and_a_dry_run_never_refuses_its_provider(tmp_path, monkeypatch, capsys):
    profile = _profile(tmp_path, monkeypatch, provider="singularity")
    monkeypatch.setattr(
        "platoon.sandbox.singularity.SingularityProvider.unavailable",
        classmethod(lambda cls, options: "not here"),
    )
    config = _write(tmp_path / "envs.yaml", {"environments": [{"name": "h", **_harbor(task_dirs=[])}]})

    cli.main(["prepare", str(config), "--dry-run"])
    err = capsys.readouterr().err
    assert f"platoon: site profile {profile}: provider singularity\n" in err

    assert _exit_code(["prepare", str(config)]) == 78
    assert "chooses the sandbox provider singularity, which this node cannot run: not here" in capsys.readouterr().err


def test_a_report_only_evaluation_needs_no_sandbox_runtime(tmp_path, monkeypatch, capsys):
    """stage=report reads finished rollouts and starts no sandbox, so a profile provider this node cannot run is no
    reason to refuse it."""

    _profile(tmp_path, monkeypatch, provider="singularity")
    monkeypatch.setattr(
        "platoon.sandbox.singularity.SingularityProvider.unavailable", classmethod(lambda cls, options: "not here")
    )
    config = _write(
        tmp_path / "eval.yaml",
        {
            "inference": {"model_name": "m", "output_dir": str(tmp_path / "results")},
            "environments": [{"name": "h", **_harbor(), "harness": "platoon-bash"}],
            "stage": "report",
        },
    )

    try:
        cli.main(["eval", str(config)])
    except SystemExit:
        pass  # whatever the report of no rollouts ends with
    assert "which this node cannot run" not in capsys.readouterr().err
    assert _exit_code(["eval", str(config), "stage=full"]) == 78
    assert "which this node cannot run: not here" in capsys.readouterr().err


def test_without_a_profile_the_commands_say_where_they_looked(tmp_path, capsys):
    config = _write(tmp_path / "envs.yaml", {"environments": [{"name": "h", **_harbor(task_dirs=[])}]})

    cli.main(["prepare", str(config), "--dry-run"])

    xdg = os.environ["XDG_CONFIG_HOME"]
    assert f"platoon: no site profile ({xdg}/platoon/site.yaml does not exist): cache root " in capsys.readouterr().err


# --- the eval section ----------------------------------------------------------------------------------------------


def test_a_training_config_without_an_eval_section_does_not_evaluate_and_rolls_out_as_training(tmp_path):
    config = _write(
        tmp_path / "train.yaml",
        {
            "train": {**TRAIN, "workflow_config": {"rollout_config": {"max_steps": 200, "timeout": 3600}}},
            "log_path": str(tmp_path),
        },
    )

    loaded = resolve_config(config, (), PlatoonTinkerRLTrainerConfig).config

    assert not evaluates(loaded)
    assert loaded.eval.workflow_config.rollout_config.max_steps == 200  # laid over training's, as a given section is
    assert loaded.eval.workflow_config.group_size == 1
