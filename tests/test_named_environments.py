"""Named environments (``platoon.envs.named``): ``pt.EnvironmentEntry``, ``env:`` in an ``environments`` entry, the
merge under the config's own keys, versions, core's built-in entries, and the components a run records in
``lock.json``.  The TextCraft extension's tests resolve its named environments from its installed metadata.

Fake distributions go into a scratch site directory (``extension_site``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from extension_site import sandbox

import platoon as pt
from platoon import cli, extensions
from platoon.envs.named import expand_environment, get_environment
from platoon.extensions import ExtensionError
from platoon.utils.config import ConfigError, resolve_config

ENTRIES = """
import platoon as pt

MAZES = pt.EnvironmentEntry(
    "acme-mazes",
    version="1",
    description="Walk to the exit of a grid maze.",
    service="python://acme_platoon.env:Mazes",
    service_kwargs={"max_tool_calls": 8, "terminal_tools": ["exit"]},
    harness_defaults={
        "platoon-react": {"system_prompt": "Find the exit.", "sampling": {"temperature": 0.7}},
        "platoon-codeact": {"prompt_builder": "acme_platoon.prompts:MazePrompts"},
    },
    agents={"max_depth": 0},
    dataset_kwargs={"split_name": "train"},
    eval_dataset_kwargs={"split_name": "val", "limit": 50},
)

MAZES_2 = pt.EnvironmentEntry(
    "acme-mazes",
    version="2",
    service="python://acme_platoon.env:Mazes",
    service_kwargs={"max_tool_calls": 16},
    eval_dataset_kwargs={"split_name": "test"},
)

BOTH = [MAZES, MAZES_2]
"""

AGENT = '''
import platoon as pt


@pt.harness("acme-agent")
async def agent(ctx):
    """Acme's agent."""
'''

PROGRAMS = '''
import platoon as pt


@pt.program("acme-best-of-n")
async def best_of_n(ep):
    """Run the root agent and keep its best attempt."""
'''

REWARDS = """
def length_penalty(trajectory):
    return float(trajectory.get("reward", 0.0)), {}
"""


@pytest.fixture
def site(tmp_path, monkeypatch):
    with sandbox(tmp_path, monkeypatch) as site:
        yield site


@pytest.fixture
def acme(site):
    """``acme-platoon``: ``acme-mazes`` at version 1 only, a harness, a program and a reward processor."""

    site.module("acme_platoon/entries.py", ENTRIES)
    site.module("acme_platoon/env.py", "class Mazes:\n    pass\n")
    site.module("acme_platoon/agent.py", AGENT)
    site.module("acme_platoon/programs.py", PROGRAMS)
    site.module("acme_platoon/rewards.py", REWARDS)
    site.install(
        "acme-platoon",
        "1.4.0",
        entry_points={
            "platoon.environments": {"acme-mazes": "acme_platoon.entries:MAZES"},
            "platoon.harnesses": {"acme-agent": "acme_platoon.agent:agent"},
            "platoon.programs": {"acme-best-of-n": "acme_platoon.programs:best_of_n"},
            "platoon.reward_processors": {"acme-length-penalty": "acme_platoon.rewards:length_penalty"},
        },
        direct_url='{"url": "file:///src/acme-platoon", "dir_info": {"editable": true}}',
    )
    return site


# --- the entry -----------------------------------------------------------------------------------------------------


def test_an_entry_is_a_checked_partial_environments_entry():
    entry = pt.EnvironmentEntry("acme-mazes", service="python://acme.env:Mazes", seed=3)

    assert (entry.ref, entry.entry()) == ("acme-mazes@1", {"service": "python://acme.env:Mazes", "seed": 3})
    entry.entry()["seed"] = 4
    assert entry.entry()["seed"] == 3  # each call returns a copy

    with pytest.raises(ValueError, match=r"sets \['harness_options', 'weight'\], which a named environment may not"):
        pt.EnvironmentEntry("acme-mazes", harness_options={"x": 1}, weight=2)
    with pytest.raises(ValueError, match="servce: unknown key; did you mean service"):
        pt.EnvironmentEntry("acme-mazes", servce="python://acme.env:Mazes")
    with pytest.raises(ValueError, match="seed: 'three' is not a int"):
        pt.EnvironmentEntry("acme-mazes", seed="three")
    with pytest.raises(ValueError, match="environment name 'Acme' must be"):
        pt.EnvironmentEntry("Acme")
    with pytest.raises(ValueError, match="harness_defaults key \\(a harness name\\) name 'platoon-react@1' must be"):
        pt.EnvironmentEntry("acme-mazes", harness_defaults={"platoon-react@1": {}})


# --- expanding an environments entry -------------------------------------------------------------------------------


def test_the_named_entry_is_the_lowest_layer_of_the_config_entry(acme):
    expanded = expand_environment(
        {
            "env": "acme-mazes",
            "harness": "platoon-react",
            "harness_options": {"sampling": {"max_completion_tokens": 64}},
            "service_kwargs": {"terminal_tools": ["give_up"]},
            "eval_dataset_kwargs": {"limit": None},
        }
    )

    assert expanded == {
        "env": "acme-mazes@1",  # pinned to the version the bare name resolved to
        "service": "python://acme_platoon.env:Mazes",
        "service_kwargs": {"max_tool_calls": 8, "terminal_tools": ["give_up"]},  # mappings merge, lists replace
        "agents": {"max_depth": 0},
        "dataset_kwargs": {"split_name": "train"},
        "eval_dataset_kwargs": {"split_name": "val", "limit": None},  # an explicit null overrides
        "harness": "platoon-react",
        "harness_options": {  # the react defaults, under the config's own options
            "system_prompt": "Find the exit.",
            "sampling": {"temperature": 0.7, "max_completion_tokens": 64},
        },
    }
    assert expand_environment(expanded) == expanded  # loading the resolved config again changes nothing


@pytest.mark.parametrize(
    ("harness", "options"),
    [
        ("platoon-codeact", {"prompt_builder": "acme_platoon.prompts:MazePrompts"}),
        ("platoon-codeact@1", {"prompt_builder": "acme_platoon.prompts:MazePrompts"}),
        ("acme-agent", None),  # no defaults for this harness: it inherits none of the others'
        ("my/agent.toml", None),  # a spec path or module:function names no harness to match
        ({"platoon-bash": 1, "mini-swe-agent": 1}, None),  # a mixture without defaults
        ({"platoon-react": 1}, {"system_prompt": "Find the exit.", "sampling": {"temperature": 0.7}}),
    ],
)
def test_harness_defaults_follow_the_entrys_harness(acme, harness, options):
    assert expand_environment({"env": "acme-mazes@1", "harness": harness}).get("harness_options") == options


def test_a_harness_mixture_that_includes_a_harness_with_defaults_is_refused(acme):
    mixture = {"platoon-react": 1, "platoon-codeact": 1}
    with pytest.raises(ValueError, match="this entry's harness is a mixture of"):
        expand_environment({"env": "acme-mazes@1", "harness": mixture})

    # Unless the entry sets the mixture's shared options itself: then no harness's defaults apply.
    expanded = expand_environment({"env": "acme-mazes@1", "harness": mixture, "harness_options": {"max_steps": 5}})
    assert expanded["harness_options"] == {"max_steps": 5}
    assert expand_environment(expanded) == expanded


def test_an_entry_by_import_string_needs_no_installed_distribution(site):
    site.module("local_envs.py", ENTRIES)

    expanded = expand_environment({"env": "local_envs:MAZES_2", "harness": "platoon-react"})

    assert expanded["env"] == "local_envs:MAZES_2" and expanded["service_kwargs"] == {"max_tool_calls": 16}
    with pytest.raises(ExtensionError, match="env 'local_envs:BOTH' is not a pt.EnvironmentEntry"):
        get_environment("local_envs:BOTH")
    # Named by import string or not, a name of core's stays core's.
    site.module("local_core_names.py", ENTRIES.replace('"acme-mazes"', '"platoon-mazes"'))
    with pytest.raises(ExtensionError, match="'platoon-mazes' .* is reserved for platoon's own components"):
        get_environment("local_core_names:MAZES")


def test_versions_live_side_by_side_and_a_bare_name_needs_just_one(site):
    site.module("acme_platoon/entries.py", ENTRIES)
    site.install("acme-platoon", entry_points={"platoon.environments": {"acme-mazes": "acme_platoon.entries:BOTH"}})

    assert get_environment("acme-mazes@1").entry()["eval_dataset_kwargs"]["split_name"] == "val"
    assert get_environment("acme-mazes@2").entry()["eval_dataset_kwargs"]["split_name"] == "test"
    with pytest.raises(ExtensionError, match=r"'acme-mazes' is ambiguous: \['acme-mazes@1', 'acme-mazes@2'\]"):
        get_environment("acme-mazes")


def test_an_unknown_environment_says_what_is_installed(acme):
    with pytest.raises(ExtensionError) as error:
        get_environment("acme-maze")

    message = str(error.value)
    assert message.startswith("Unknown environment 'acme-maze'.")
    assert "installed platoon.environments entry points: ['acme-mazes (acme-platoon 1.4.0)']" in message
    assert "did you mean 'acme-mazes'" in message


# --- loading configs -----------------------------------------------------------------------------------------------


def _eval_config(tmp_path: Path, *environments: dict) -> Path:
    path = tmp_path / "eval.yaml"
    path.write_text(
        yaml.safe_dump(
            {"inference": {"model_name": "m", "output_dir": str(tmp_path / "out")}, "environments": list(environments)}
        )
    )
    return path


def test_a_config_loads_its_named_environment_after_extends_and_overrides(acme, tmp_path):
    from platoon.eval.run import PlatoonEvalConfig

    path = _eval_config(tmp_path, {"env": "acme-mazes", "harness": "platoon-codeact"})

    resolved = resolve_config(
        path, ["environments.0.harness=platoon-react", "environments.0.seed=5"], PlatoonEvalConfig
    )

    (environment,) = resolved.config.environments
    assert (environment.env, environment.service, environment.seed) == (
        "acme-mazes@1",
        "python://acme_platoon.env:Mazes",
        5,
    )
    assert environment.harness_options == {"system_prompt": "Find the exit.", "sampling": {"temperature": 0.7}}
    assert environment.agents.max_depth == 0
    assert resolved.data["environments"][0]["env"] == "acme-mazes@1"  # what config.resolved.yaml holds


def test_a_bad_named_environment_is_a_config_error_at_its_entry(acme, tmp_path):
    from platoon.eval.run import PlatoonEvalConfig

    path = _eval_config(tmp_path, {"name": "a", "env": "acme-mazes@3"}, {"name": "b", "env": "acme-mazes@1"})

    with pytest.raises(ConfigError) as error:
        resolve_config(path, [], PlatoonEvalConfig)

    assert error.value.problems[0].startswith("environments.0.env: Unknown environment 'acme-mazes@3'")
    assert len(error.value.problems) == 1


def test_a_named_environment_leaves_exclude_tasks_to_the_entries_that_select_it(site, tmp_path, capsys):
    # An entry's own exclude_tasks would replace the environment's list instead of adding to it, and bring back the
    # tasks the environment left out.
    with pytest.raises(ValueError, match=r"sets \['exclude_tasks'\], which a named environment may not set"):
        pt.EnvironmentEntry("acme-mazes", service="python://acme.env:Mazes", exclude_tasks=["train:0"])

    site.module(
        "acme_platoon/entries.py",
        'import platoon as pt\n\nMAZES = pt.EnvironmentEntry("acme-mazes", service="python://acme.env:Mazes", '
        'exclude_tasks=["train:0"])\n',
    )
    site.install("acme-platoon", entry_points={"platoon.environments": {"acme-mazes": "acme_platoon.entries:MAZES"}})

    assert cli.config_validate(_eval_config(tmp_path, {"env": "acme-mazes@1"})) == cli.EXIT_CONFIG
    out = capsys.readouterr().out
    assert "environments.0.env: platoon.environments entry point acme-mazes = acme_platoon.entries:MAZES" in out
    assert "failed to load: ValueError: environment acme-mazes@1 sets ['exclude_tasks'], which a named" in out


def test_environments_built_from_plain_data_expand_too(acme):
    from platoon.train.components import normalize_environment_configs

    (environment,) = normalize_environment_configs([{"env": "acme-mazes@1", "harness": "acme-agent"}])

    assert environment.service == "python://acme_platoon.env:Mazes" and environment.env == "acme-mazes@1"


# --- what a run records --------------------------------------------------------------------------------------------


def _train_config(tmp_path: Path, environment: dict) -> Path:
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "train": {"model_name": "Qwen/Qwen3-4B", "renderer_name": "qwen3", "batch_size": 16},
                "eval": {"every": 0},
                "log_path": str(tmp_path / "logs"),
                "stats": {"experiment_name": "exp", "trial_name": "trial"},
                "environments": [environment],
            }
        )
    )
    return path


def test_lock_json_records_every_component_with_its_distribution(acme, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("platoon.train.tinker.train.run_tinker_training", calls.append)
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")  # train checks its key before anything else: past it
    sdk_imports = []
    monkeypatch.setattr("platoon.envs.ors.sdk.import_sdk", lambda: sdk_imports.append(True))
    config = _train_config(
        tmp_path,
        {
            "env": "acme-mazes",
            "harness": {"acme-agent": 1, "platoon-bash": 1},
            "program": "acme-best-of-n",
            "reward_processor": "acme-length-penalty",
        },
    )

    cli.main(["train", str(config)])

    lock = json.loads((tmp_path / "logs/exp/trial/lock.json").read_text())
    components = {(item["kind"], item["ref"]): item for item in lock["components"]}
    acme_version = ("acme-platoon", "1.4.0")
    for key, resolved, entry_point in [
        (("environment", "acme-mazes@1"), "acme-mazes@1", "platoon.environments:acme-mazes"),
        (("harness", "acme-agent"), "acme-agent@1", "platoon.harnesses:acme-agent"),
        (("program", "acme-best-of-n"), "acme-best-of-n@1", "platoon.programs:acme-best-of-n"),
        (
            ("reward_processor", "acme-length-penalty"),
            "acme-length-penalty",
            "platoon.reward_processors:acme-length-penalty",
        ),
    ]:
        item = components[key]
        assert (item["resolved"], item["entry_point"]) == (resolved, entry_point)
        assert (item["distribution"], item["distribution_version"]) == acme_version
        assert item["environments"] == ["default"]
    bash = components[("harness", "platoon-bash")]
    assert (bash["resolved"], bash["distribution"], bash["entry_point"]) == ("platoon-bash@1", "platoon", None)
    assert bash["source"].endswith("platoon/harness/builtin/specs/platoon-bash.toml")
    service = components[("service", "python://acme_platoon.env:Mazes")]
    assert (service["source"], service["distribution"]) == ("acme_platoon.env:Mazes", "acme-platoon")
    assert sdk_imports  # an ORS class is imported as the service imports it: after the SDK, set up
    assert lock["extensions"] == {
        "acme-platoon": {
            "version": "1.4.0",
            "requires": ["platoon>=0.2,<0.3"],
            "direct_url": {"url": "file:///src/acme-platoon", "dir_info": {"editable": True}},
        }
    }
    snapshot = yaml.safe_load((tmp_path / "logs/exp/trial/config.resolved.yaml").read_text())
    assert snapshot["environments"][0]["env"] == "acme-mazes@1"
    assert calls  # the trainer ran from the snapshot


def test_a_run_that_names_an_unknown_component_exits_78_before_it_starts(acme, tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr("platoon.train.tinker.train.run_tinker_training", calls.append)
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")  # train checks its key before anything else: past it
    config = _train_config(tmp_path, {"env": "acme-mazes@1", "harness": "acme-agent", "program": "acme-best-of-m"})

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["train", str(config)])

    assert exit_info.value.code == cli.EXIT_CONFIG
    err = capsys.readouterr().err
    assert "environments[default].program: Unknown program 'acme-best-of-m'" in err
    assert "did you mean 'acme-best-of-n'" in err
    assert calls == [] and not (tmp_path / "logs/exp/trial/lock.json").exists()
    # The same check, without running anything; --no-resolve checks the files and the named environment only.
    assert cli.config_validate(config) == cli.EXIT_CONFIG
    assert cli.config_validate(config, resolve=False) == 0


# --- the env commands ------------------------------------------------------------------------------------------------


def test_env_list_and_show(acme, tmp_path, capsys):
    path = _eval_config(tmp_path, {"env": "acme-mazes", "harness": "platoon-react"})

    cli.main(["env", "list", str(path)])
    out = capsys.readouterr().out

    named = out.split("Named environments (an environments entry's `env`):")[1].split("Environments of")[0]
    assert [line.split() for line in named.strip().splitlines()] == [
        ["env", "distribution", "source"],
        [
            "platoon-number-search@1",
            "platoon",
            extensions.platoon_version(),
            "platoon.envs.number_search.entries:NUMBER_SEARCH",
        ],
        ["acme-mazes", "acme-platoon", "1.4.0", "acme_platoon.entries:MAZES"],
    ]
    rows = [line.split() for line in out.split(f"Environments of {path}:")[1].strip().splitlines()]
    assert rows[1] == ["default", "tasks", "python://acme_platoon.env:Mazes", "platoon-react", "-", "harness_rollout"]

    cli.main(["env", "show", "acme-mazes"])
    shown = capsys.readouterr().out
    assert shown.splitlines()[:4] == [
        "acme-mazes@1",
        "source: acme_platoon.entries:MAZES",
        "distribution: acme-platoon 1.4.0",
        "description: Walk to the exit of a grid maze.",
    ]
    assert yaml.safe_load(shown.split("description: Walk to the exit of a grid maze.\n")[1])["harness_defaults"][
        "platoon-react"
    ] == {"system_prompt": "Find the exit.", "sampling": {"temperature": 0.7}}


# --- core's built-in named environments ---------------------------------------------------------------------------


def test_a_built_in_resolves_to_cores_entry_whatever_is_installed(site, capsys):
    # A leftover of the old layout: number search's distribution, whose entry point names a module that is gone.
    site.install(
        "platoon-number-search",
        entry_points={"platoon.environments": {"platoon-number-search": "platoon.number_search.entries:NUMBER_SEARCH"}},
    )

    entry = get_environment("platoon-number-search")

    assert entry.ref == "platoon-number-search@1"
    assert extensions.recorded_origin("environment", entry.ref).label() == f"platoon {extensions.platoon_version()}"
    assert expand_environment({"env": "platoon-number-search@1"})["service"] == (
        "python://platoon.envs.number_search.env:NumberSearchEnvironment"
    )
    rows = {(info.name, info.distribution): info for info in extensions.list_components("environment")}
    assert rows[("platoon-number-search@1", "platoon")].builtin
    leftover = rows[("platoon-number-search", "platoon-number-search")]
    assert leftover.status == "ignored: platoon-number-search is built into platoon, which uses its own"
    (problem,) = extensions.check_extensions(["platoon-number-search"], load=True)
    assert problem.severity == "warning" and "built into platoon, which never loads this entry point" in problem.message

    cli.main(["env", "show", "platoon-number-search"])
    assert capsys.readouterr().out.splitlines()[:3] == [
        "platoon-number-search@1",
        "source: platoon.envs.number_search.entries:NUMBER_SEARCH",
        f"distribution: platoon {extensions.platoon_version()}",
    ]
