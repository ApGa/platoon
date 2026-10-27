"""Extensions (``platoon.extensions``): components found by exact name in their entry-point group and imported only
when named, the naming rule, the compatibility check, listing and checking, and the retired ``platoon.plugins``.

Each test installs fake distributions into a scratch site directory (``extension_site``), so nothing depends on
what the test environment has installed.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import logging
import sys
import threading
import typing

import pytest
from extension_site import sandbox

import platoon as pt
from platoon import cli, extensions
from platoon.extensions import ExtensionError
from platoon.harness.registry import (
    BUILTIN_SPECS_DIR,
    HarnessRegistry,
    default_registry,
    get_harness,
    harness_load_failures,
    harness_origin,
)
from platoon.harness.spec import HarnessSpecError, parse_spec
from platoon.programs import resolve_program
from platoon.registry import register_component, resolve_component

AGENT = '''
import platoon as pt


@pt.harness("acme-agent", version="2")
async def agent(ctx):
    """Acme's agent."""
'''

OTHER_AGENT = '''
import platoon as pt


@pt.harness("acme-other")
async def other(ctx):
    """Another of Acme's agents."""
'''

PROGRAMS = '''
import platoon as pt


@pt.program("acme-best-of-n")
async def best_of_n(ep):
    """Run the root agent and keep its best attempt."""


@pt.program("acme-twice", version="1")
async def twice_v1(ep):
    """Run the root agent twice."""


@pt.program("acme-twice", version="2")
async def twice_v2(ep):
    """Run the root agent twice, the second time with the first's answer."""


TWICE = [twice_v1, twice_v2]
'''

REWARDS = """
def length_penalty(trajectory):
    return float(trajectory.get("reward", 0.0)) - 0.1, {"penalty": 0.1}
"""

ENTRIES = """
import platoon as pt

MAZES = pt.EnvironmentEntry(
    "acme-mazes",
    version="1",
    service="python://acme_platoon.env:Mazes",
    harness_defaults={"platoon-react": {"system_prompt": "Find the exit."}},
    eval_dataset_kwargs={"split_name": "val"},
)
"""

ACME = {
    "platoon.harnesses": {"acme-agent": "acme_platoon.agent:agent", "acme-other": "acme_platoon.other:other"},
    "platoon.programs": {
        "acme-best-of-n": "acme_platoon.programs:best_of_n",
        "acme-twice": "acme_platoon.programs:TWICE",
    },
    "platoon.environments": {"acme-mazes": "acme_platoon.entries:MAZES"},
    "platoon.reward_processors": {"acme-length-penalty": "acme_platoon.rewards:length_penalty"},
}


@pytest.fixture
def site(tmp_path, monkeypatch):
    with sandbox(tmp_path, monkeypatch) as site:
        yield site


@pytest.fixture
def acme(site):
    """The ``acme-platoon`` distribution: two harnesses, two programs (one in two versions), a named environment
    and a reward processor."""

    site.module("acme_platoon/agent.py", AGENT)
    site.module("acme_platoon/other.py", OTHER_AGENT)
    site.module("acme_platoon/programs.py", PROGRAMS)
    site.module("acme_platoon/rewards.py", REWARDS)
    site.module("acme_platoon/entries.py", ENTRIES)
    site.module("acme_platoon/env.py", "class Mazes:\n    pass\n")
    site.install("acme-platoon", "1.4.0", entry_points=ACME)
    return site


def _imported(module: str) -> bool:
    return module in sys.modules


# --- discovery -----------------------------------------------------------------------------------------------------


def test_a_name_imports_only_the_entry_point_of_that_name(acme):
    harness = get_harness("acme-agent")

    assert harness.ref == "acme-agent@2" and harness.source == "acme_platoon.agent:agent"
    assert _imported("acme_platoon.agent")
    assert not any(_imported(f"acme_platoon.{name}") for name in ("other", "programs", "rewards", "entries"))
    assert harness_origin(harness) == extensions.Origin(
        distribution="acme-platoon",
        version="1.4.0",
        entry_point="platoon.harnesses:acme-agent",
        module="acme_platoon.agent",
    )


def test_listing_reads_metadata_and_imports_nothing(acme):
    rows = {(info.kind, info.name): info for info in extensions.list_components()}

    assert not any(_imported(f"acme_platoon.{name}") for name in ("agent", "other", "programs", "rewards", "entries"))
    for kind, name in [
        ("harness", "acme-agent"),
        ("harness", "acme-other"),
        ("program", "acme-best-of-n"),
        ("program", "acme-twice"),
        ("environment", "acme-mazes"),
        ("reward_processor", "acme-length-penalty"),
    ]:
        info = rows[(kind, name)]
        assert (info.distribution, info.version, info.status, info.builtin) == ("acme-platoon", "1.4.0", "ok", False)
        assert info.requires == ("platoon>=0.4,<0.5",)
    assert rows[("harness", "platoon-codeact@1")].builtin
    assert rows[("program", "single@1")].distribution == "platoon"


def test_programs_resolve_by_name_and_version(acme):
    assert resolve_program("acme-best-of-n").source == "acme_platoon.programs:best_of_n"
    with pytest.raises(ExtensionError, match=r"'acme-twice' is ambiguous: \['acme-twice@1', 'acme-twice@2'\]"):
        resolve_program("acme-twice")
    assert resolve_program("acme-twice@2").fn.__name__ == "twice_v2"
    assert extensions.recorded_origin("program", "acme-twice@1").distribution == "acme-platoon"
    assert resolve_program("single").ref == "single@1"  # core's, untouched by the extension


def test_a_spec_and_a_function_harness_of_other_versions_make_a_bare_name_ambiguous(site, caplog):
    site.module("acme_agent_fn.py", AGENT)  # acme-agent@2, a function harness
    (site.root / "acme_agent.toml").write_text('name = "acme-agent"\nversion = "1"\n[launch]\nargv = ["acme"]\n')
    site.module(
        "acme_agent_specs.py", "from pathlib import Path\n\nSPEC = str(Path(__file__).with_name('acme_agent.toml'))\n"
    )
    site.install("acme-new", entry_points={"platoon.harnesses": {"acme-agent": "acme_agent_fn:agent"}})
    site.install("acme-old", entry_points={"platoon.harnesses": {"acme-agent": "acme_agent_specs:SPEC"}})

    with pytest.raises(ExtensionError, match=r"'acme-agent' is ambiguous: \['acme-agent@1', 'acme-agent@2'\]"):
        get_harness("acme-agent")  # two versions: the spec's is not preferred over the function's
    assert get_harness("acme-agent@1").source.endswith("acme_agent.toml")
    assert get_harness("acme-agent@2").source == "acme_agent_fn:agent"

    # One name@version as both a spec and a function harness: the spec wins, with a warning.
    (site.root / "acme_agent_v2.toml").write_text('name = "acme-agent"\nversion = "2"\n[launch]\nargv = ["acme"]\n')
    default_registry().add_path(site.root / "acme_agent_v2.toml")
    with caplog.at_level(logging.WARNING, logger="platoon.harness.registry"):
        assert get_harness("acme-agent@2").source.endswith("acme_agent_v2.toml")
    assert "names both the TOML spec(s) ['acme-agent@2'] and the function harness(es) ['acme-agent@2']" in caplog.text


def test_a_reward_processor_resolves_by_name_and_names_its_distribution(acme):
    from platoon.train.auto import AutoRewardProcessor
    from platoon.train.components import EnvironmentConfig

    processor = resolve_component("reward_processor", "acme-length-penalty")

    assert processor({"reward": 1.0}) == (0.9, {"penalty": 0.1})
    config = type("Config", (), {"environments": [EnvironmentConfig(reward_processor="acme-length-penalty")]})()
    assert AutoRewardProcessor.from_config(config) is processor
    assert extensions.recorded_origin("reward_processor", "acme-length-penalty").label() == "acme-platoon 1.4.0"
    # An import path keeps working, and a name that is neither says what is installed.
    assert resolve_component("reward_processor", "acme_platoon.rewards:length_penalty") is processor
    with pytest.raises(
        ExtensionError, match=r"Unknown reward processor 'acme-length'.*did you mean 'acme-length-penalty'"
    ):
        resolve_component("reward_processor", "acme-length")


def test_a_dotted_import_path_is_never_taken_over_by_an_entry_point_of_that_name(site):
    site.module("mine/rewards.py", "def score(trajectory):\n    return 1.0, {}\n")
    site.module("rival_rewards.py", "def evil(trajectory):\n    return -1.0, {}\n")
    assert resolve_component("reward_processor", "mine.rewards.score")({}) == (1.0, {})
    rival = {"mine.rewards.score": "rival_rewards:evil", "rival.v2": "rival_rewards:evil"}
    site.install("acme-rival", entry_points={"platoon.reward_processors": rival})

    # Installed later, an entry point named like the import path changes nothing: it is never even imported.
    assert resolve_component("reward_processor", "mine.rewards.score")({}) == (1.0, {})
    assert not _imported("rival_rewards")
    # A dotted name that imports nothing is a name, and the entry point's.
    assert resolve_component("reward_processor", "rival.v2")({}) == (-1.0, {})


def test_a_reward_processor_name_from_a_package_module_and_an_entry_point_is_an_error_naming_both(site):
    """As for harnesses and programs: the installed entry point is found by its name first, and a ``package:``
    module that registered another function under that name makes the name ambiguous, not silently local."""

    site.module("local_rewards.py", "def score(trajectory):\n    return 1.0, {}\n")
    site.module("rival_rewards.py", "def evil(trajectory):\n    return -1.0, {}\n")
    import local_rewards

    register_component("reward_processor", "acme-score", local_rewards.score)
    assert resolve_component("reward_processor", "acme-score") is local_rewards.score
    site.install("acme-rival", entry_points={"platoon.reward_processors": {"acme-score": "rival_rewards:evil"}})

    with pytest.raises(
        ExtensionError, match="'acme-score' is defined twice: local_rewards.score and rival_rewards:evil"
    ):
        resolve_component("reward_processor", "acme-score")


def test_an_unknown_name_lists_what_is_installed_and_suggests_the_nearest(acme):
    with pytest.raises(ExtensionError) as error:
        get_harness("acme-agnet")

    message = str(error.value)
    assert message.startswith("Unknown harness 'acme-agnet'. Available: [")
    assert "installed platoon.harnesses entry points: ['acme-agent (acme-platoon 1.4.0)'" in message
    assert "did you mean 'acme-agent'" in message
    assert "an entry point named 'acme-agnet' in platoon.harnesses" in message


def test_one_name_and_version_from_two_distributions_is_an_error_naming_both(site):
    for owner in ("one", "two"):
        site.module(f"acme_{owner}/entries.py", ENTRIES.replace("acme_platoon.env", f"acme_{owner}.env"))
        site.install(
            f"acme-{owner}", entry_points={"platoon.environments": {"acme-mazes": f"acme_{owner}.entries:MAZES"}}
        )
    from platoon.envs.named import get_environment

    defined_twice = r"acme-mazes@1 is defined twice: acme_(one|two)\.entries:MAZES and acme_(one|two)\.entries:MAZES"
    with pytest.raises(ExtensionError, match=defined_twice):
        get_environment("acme-mazes")


def test_a_component_named_unlike_its_entry_point_is_refused(site):
    site.module("acme_platoon/programs.py", PROGRAMS)
    site.install("acme-platoon", entry_points={"platoon.programs": {"acme-wrong": "acme_platoon.programs:best_of_n"}})

    with pytest.raises(ExtensionError, match="loads to program acme-best-of-n@1, whose name is not the entry point's"):
        resolve_program("acme-wrong")


def test_a_harness_entry_point_named_for_its_module_is_found_on_a_miss_with_a_warning(site, caplog):
    site.module("acme_harnesses/one_shot.py", AGENT.replace('"acme-agent", version="2"', '"acme-one-shot"'))
    site.install("acme-harnesses", entry_points={"platoon.harnesses": {"acme-harnesses": "acme_harnesses.one_shot"}})

    assert get_harness("platoon-react").ref == "platoon-react@1"
    assert not _imported("acme_harnesses.one_shot")  # a hit never loads the others

    with caplog.at_level(logging.WARNING, logger="platoon.harness.registry"):
        harness = get_harness("acme-one-shot")

    assert harness.ref == "acme-one-shot@1"
    assert "registers ['acme-one-shot@1'], none named 'acme-harnesses'" in caplog.text
    assert "name each entry point after the harness it registers" in caplog.text


def test_a_broken_entry_point_fails_its_own_name_only(site):
    site.module("acme_broken.py", "raise ImportError('cannot import name NodeContext')\n")
    site.install("acme-broken", entry_points={"platoon.harnesses": {"acme-broken": "acme_broken:agent"}})

    assert get_harness("platoon-react").ref == "platoon-react@1"
    assert harness_load_failures() == {}
    with pytest.raises(ExtensionError) as error:
        get_harness("acme-broken")

    assert "acme-broken = acme_broken:agent (acme-broken 0.1.0) failed to load" in str(error.value)
    assert "ImportError: cannot import name NodeContext" in str(error.value)
    assert harness_load_failures() == {"acme-broken = acme_broken:agent": "ImportError: cannot import name NodeContext"}


def test_a_registry_of_ones_own_loads_an_entry_point_into_itself(site):
    """What a test suite does for a checkout whose metadata is not installed (the CONTRIBUTING pattern)."""

    from importlib.metadata import EntryPoint

    site.module("acme_platoon/agent.py", AGENT)
    registry = HarnessRegistry()
    registry.add_entry_point(EntryPoint(name="acme-agent", value="acme_platoon.agent:agent", group="platoon.harnesses"))
    registry.add_entry_point(EntryPoint(name="acme-gone", value="acme_gone:agent", group="platoon.harnesses"))

    assert get_harness("acme-agent").ref == "acme-agent@2"
    assert list(registry.entry_point_failures) == ["acme-gone = acme_gone:agent"]


def test_a_rebuilt_spec_registry_gets_the_extensions_specs_again(site):
    site.module(
        "acme_specs/__init__.py", "from pathlib import Path\n\nSPEC = str(Path(__file__).with_name('cli.toml'))\n"
    )
    (site.root / "acme_specs" / "cli.toml").write_text('name = "acme-cli"\nversion = "3"\n[launch]\nargv = ["acme"]\n')
    site.install("acme-specs", entry_points={"platoon.harnesses": {"acme-cli": "acme_specs:SPEC"}})

    assert get_harness("acme-cli").ref == "acme-cli@3"
    default_registry.cache_clear()  # a fresh registry holds core's specs only

    assert get_harness("acme-cli").ref == "acme-cli@3"
    assert harness_origin(get_harness("acme-cli")).label() == "acme-specs 0.1.0"


GATE = """
import threading

ENTERED = threading.Event()
RELEASE = threading.Event()
"""

SLOW_AGENT = '''
import acme_gate
import platoon as pt

acme_gate.ENTERED.set()
assert acme_gate.RELEASE.wait(30), "the test never released the import"


@pt.harness("acme-slow")
async def slow(ctx):
    """An agent whose module takes a while to import."""
'''


class _Call(threading.Thread):
    """``fn(*args)`` on a thread of its own, keeping what it returned or raised."""

    def __init__(self, fn, *args):
        super().__init__(daemon=True)
        self.fn, self.args, self.result, self.error = fn, args, None, None
        self.start()

    def run(self):
        try:
            self.result = self.fn(*self.args)
        except BaseException as exc:  # reported by the test
            self.error = exc


def test_an_import_in_progress_holds_up_only_the_lookups_of_its_own_name(acme, site):
    site.module("acme_gate.py", GATE)
    site.module("acme_slow.py", SLOW_AGENT)
    site.install("acme-slow", entry_points={"platoon.harnesses": {"acme-slow": "acme_slow:slow"}})
    import acme_gate

    try:
        first = _Call(get_harness, "acme-slow")
        assert acme_gate.ENTERED.wait(30)
        second = _Call(get_harness, "acme-slow")
        other = _Call(resolve_program, "acme-best-of-n")  # another extension's entry point, meanwhile

        other.join(30)
        assert not other.is_alive() and other.error is None and other.result.ref == "acme-best-of-n@1"
        second.join(0.2)
        assert second.is_alive()  # it waits for the import the first thread started
    finally:
        acme_gate.RELEASE.set()
    first.join(30)
    second.join(30)

    assert (first.error, second.error) == (None, None)
    assert first.result.ref == second.result.ref == "acme-slow@1"
    assert extensions.recorded_origin("harness", "acme-slow@1").label() == "acme-slow 0.1.0"


# --- names ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["acme-agent", "mini-swe-agent", "acme2", "acme.v2_beta-3", "a"])
def test_names_that_follow_the_rule(name):
    assert extensions.check_name(name) == name


@pytest.mark.parametrize(
    "name", ["Acme", "acme:agent", "acme/agent", "acme@1", "acme-agent.toml", "-acme", "acme-", "acme--agent", "", 3]
)
def test_names_that_break_the_rule(name):
    with pytest.raises(ValueError, match="must be lower-case letters and digits"):
        extensions.check_name(name)


def test_the_decorators_refuse_a_name_that_could_never_resolve():
    async def agent(ctx): ...

    with pytest.raises(ValueError, match="harness name 'acme:agent' must be"):
        pt.harness("acme:agent")  # registered, it would read as an import string and never resolve
    with pytest.raises(ValueError, match="program name 'Acme' must be"):
        pt.program("Acme")
    with pytest.raises(ValueError, match="harness 'acme-agent' version 'v 2' must be"):
        pt.harness("acme-agent", version="v 2")
    with pytest.raises(HarnessSpecError, match="name 'Acme-Agent' must match the naming rule"):
        parse_spec({"name": "Acme-Agent", "version": "1", "launch": {"argv": ["acme"]}})


def test_core_names_are_reserved_for_core(tmp_path):
    async def agent(ctx): ...

    for register, name in [(pt.harness, "platoon-mine"), (pt.program, "single"), (pt.program, "gym")]:
        with pytest.raises(ValueError, match=f"name '{name}' .* is reserved for platoon's own components"):
            register(name)(agent)
    with pytest.raises(ValueError, match="reserved for platoon's own components"):
        register_component("reward_processor", "platoon-score", lambda trajectory: (0.0, {}))
    for name in ("mini-swe-agent", "openhands"):  # core's specs that keep a third-party agent's product name
        spec = tmp_path / f"{name}.toml"
        spec.write_text(f'name = "{name}"\nversion = "9"\n[launch]\nargv = ["{name}"]\n')
        with pytest.raises(ValueError, match=f"'{name}' .* is reserved"):
            HarnessRegistry().add_path(spec)
        HarnessRegistry().add_path(BUILTIN_SPECS_DIR / f"{name}.toml")  # core's own


def test_an_extension_cannot_take_a_core_spec_s_product_name(site):
    """An extension's entry point named like a spec core ships under a third-party agent's name (openhands) is never
    loaded: the name resolves to core's spec."""

    site.module("acme_fake.py", AGENT.replace('"acme-agent", version="2"', '"openhands"'))
    site.install("acme-fake", entry_points={"platoon.harnesses": {"openhands": "acme_fake:agent"}})

    assert get_harness("openhands").source.endswith("platoon/harness/builtin/specs/openhands.toml")
    assert "acme_fake" not in sys.modules


def test_lock_json_credits_the_harbor_harnesses_to_platoon():
    """The run record names the distribution behind each component it resolves: the OpenHands spec and the dataset
    check harnesses that ship for Harbor tasks are core's, named by no entry point."""

    from types import SimpleNamespace

    names = ("openhands", "platoon-oracle", "platoon-noop")
    config = SimpleNamespace(environments=[{"service": "harbor", "harness": dict.fromkeys(names, 1)}])

    harnesses = {item["ref"]: item for item in extensions.run_components(config) if item["kind"] == "harness"}

    assert {ref: (item["resolved"], item["distribution"], item["entry_point"]) for ref, item in harnesses.items()} == {
        "openhands": ("openhands@1.54.0.post2", "platoon", None),
        "platoon-oracle": ("platoon-oracle@1", "platoon", None),
        "platoon-noop": ("platoon-noop@1", "platoon", None),
    }
    assert harnesses["openhands"]["source"].endswith("platoon/harness/builtin/specs/openhands.toml")
    assert harnesses["platoon-oracle"]["source"] == "platoon.harness.builtin.oracle:oracle"


def test_an_extension_cannot_take_a_core_name(site):
    site.module("acme_fake.py", AGENT.replace('"acme-agent", version="2"', '"acme-codeact"'))
    site.install("acme-fake", entry_points={"platoon.harnesses": {"platoon-codeact": "acme_fake:agent"}})

    assert get_harness("platoon-codeact").source == "platoon.harness.builtin.codeact.harness:codeact"
    assert not _imported("acme_fake")
    assert extensions.entry_points_named("harness", "platoon-codeact") == []
    (info,) = [info for info in extensions.list_components("harness") if info.distribution == "acme-fake"]
    assert info.status.startswith("ignored: platoon.harnesses entry point name 'platoon-codeact'")
    problems = extensions.check_extensions(["acme-fake"])
    assert [problem.severity for problem in problems] == ["error"]
    assert "is reserved for platoon's own components" in problems[0].message


def test_the_projects_own_distributions_may_use_the_platoon_owner(site):
    site.module("platoon_acme_entries.py", ENTRIES.replace('"acme-mazes"', '"platoon-mazes"'))
    site.install("platoon-acme", entry_points={"platoon.environments": {"platoon-mazes": "platoon_acme_entries:MAZES"}})
    from platoon.envs.named import get_environment

    assert get_environment("platoon-mazes").ref == "platoon-mazes@1"


# --- compatibility -------------------------------------------------------------------------------------------------


def test_an_extension_whose_platoon_range_excludes_this_platoon_is_refused_before_import(site, monkeypatch, caplog):
    site.module("acme_platoon/agent.py", AGENT)
    site.install("acme-platoon", "3.0", entry_points=ACME, requires=["platoon>=99,<100"])
    running = extensions.platoon_version()

    with pytest.raises(ExtensionError) as error:
        get_harness("acme-agent")

    assert f"acme-platoon 3.0 requires platoon>=99,<100, but platoon {running} is running" in str(error.value)
    assert "PLATOON_IGNORE_EXTENSION_REQUIREMENTS=1" in str(error.value)
    assert not _imported("acme_platoon.agent")
    (info,) = [info for info in extensions.list_components("harness") if info.name == "acme-agent"]
    assert info.status.startswith("incompatible: acme-platoon 3.0 requires")

    extensions.refresh()  # forget the failure, as after a fix
    monkeypatch.setenv(extensions.IGNORE_REQUIREMENTS_ENV, "1")
    with caplog.at_level(logging.WARNING, logger="platoon.extensions"):
        assert get_harness("acme-agent").ref == "acme-agent@2"
    assert "loading it anyway" in caplog.text


@pytest.mark.parametrize(
    ("requirement", "extras", "refused"),
    [
        ('platoon[ors]>=99; extra == "platoon"', ("platoon",), True),  # the platoon glue behind an extra
        ('platoon>=99; python_version < "3"', (), False),  # a marker that does not hold here
        ("platoon", (), False),  # no range
        ("openreward>=99", (), False),  # not a requirement on platoon
    ],
)
def test_the_requirements_that_count(site, requirement, extras, refused):
    dist_info = site.install("acme-platoon", requires=[requirement], extras=extras)
    from importlib.metadata import PathDistribution

    assert (extensions.incompatibility(PathDistribution(dist_info)) is not None) is refused


# --- checking ------------------------------------------------------------------------------------------------------


def test_check_reports_names_owners_requirements_and_retired_groups(site, capsys):
    site.install(
        "acme-platoon",
        entry_points={
            "platoon.harnesses": {"acme-agent": "a:b", "Bad_Name": "a:c"},
            "platoon.programs": {"zeta-x": "a:d"},
        },
    )
    site.install("acme-loose", entry_points={"platoon.harnesses": {"loose-agent": "a:e"}}, requires=["platoon>=0.1"])
    site.install("acme-free", entry_points={"platoon.harnesses": {"free-agent": "a:f"}}, requires=[])
    site.install("acme-old", entry_points={"platoon.plugins": {"acme": "acme_old.registry"}})

    problems = {
        (problem.distribution, problem.severity, problem.message.split(":")[0])
        for problem in extensions.check_extensions(["acme-platoon", "acme-loose", "acme-free", "acme-old"])
    }

    suggested = extensions.suggested_requirement()
    assert problems == {
        ("acme-platoon", "error", "platoon.harnesses entry point Bad_Name = a"),
        ("acme-platoon", "warning", "components with more than one owner ['acme', 'zeta']"),
        (
            "acme-loose",
            "warning",
            "its platoon requirement ['platoon>=0.1'] has no upper bound; bound it below the next release that "
            f"may break it (the next minor release below 1.0), e.g. '{suggested}'",
        ),
        (
            "acme-free",
            "warning",
            f"declares no requirement on platoon; add one with an upper bound, e.g. '{suggested}', "
            "so that platoon can refuse a version it does not support",
        ),
        ("acme-old", "warning", "platoon.plugins entry point acme = acme_old.registry"),
    }
    assert cli.extensions_check("acme-platoon") == cli.EXIT_CONFIG
    assert cli.extensions_check("acme-loose") == 0
    out = capsys.readouterr().out
    assert "error: acme-platoon: platoon.harnesses entry point Bad_Name = a:c" in out
    assert out.rstrip().endswith("0 error(s), 1 warning(s)")


@pytest.mark.parametrize(
    ("running", "suggested"),
    [
        ("0.1.0", "platoon>=0.1,<0.2"),  # below 1.0, a minor release may break extensions
        ("0.9.3", "platoon>=0.9,<0.10"),
        ("0.2.0.dev1", "platoon>=0.2.0.dev1,<0.3"),
        ("1.1.0", "platoon>=1.1,<2"),
        ("2.3.4", "platoon>=2.3,<3"),
        ("1.2.0.dev1", "platoon>=1.2.0.dev1,<2"),  # a pre-release sorts below 1.2, so it is its own lower bound
    ],
)
def test_the_suggested_requirement_admits_the_platoon_that_suggests_it(site, monkeypatch, running, suggested):
    """Following ``extensions check``'s advice never makes an extension incompatible with the running platoon."""

    monkeypatch.setattr(extensions, "platoon_version", lambda: running)
    site.install("acme-free", entry_points={"platoon.harnesses": {"free-agent": "a:f"}}, requires=[])

    (problem,) = extensions.check_extensions(["acme-free"])

    assert extensions.suggested_requirement() == suggested and f"e.g. '{suggested}'" in problem.message
    advised = site.install("acme-advised", entry_points={"platoon.harnesses": {"advised": "a:g"}}, requires=[suggested])
    from importlib.metadata import PathDistribution

    assert extensions.incompatibility(PathDistribution(advised)) is None
    assert extensions.check_extensions(["acme-advised"]) == []


def test_check_with_load_imports_each_component_as_a_config_would(acme, site):
    site.module("acme_broken.py", "raise ImportError('no such SDK')\n")
    site.install("acme-broken", entry_points={"platoon.harnesses": {"acme-broken": "acme_broken:agent"}})

    assert extensions.check_extensions(["acme-platoon"], load=True) == []
    (problem,) = extensions.check_extensions(["acme-broken"], load=True)
    assert problem.severity == "error" and "ImportError: no such SDK" in problem.message


# --- provenance ----------------------------------------------------------------------------------------------------


def test_code_under_a_shared_namespace_package_is_not_credited_to_another_of_its_distributions(site):
    """``acmens`` is a namespace package that two distributions share: ``acme-agents`` provides a harness by entry
    point, and ``acmens.envs`` holds an environment class a config names by import string."""

    site.module("acmens/agents/harness.py", AGENT)
    site.module("acmens/envs/mazes.py", "class Mazes:\n    pass\n")
    (site.root / "acmens" / "__init__.py").unlink()  # PEP 420: no __init__, so acmens.* may come from anywhere
    importlib.invalidate_caches()
    site.install("acme-agents", entry_points={"platoon.harnesses": {"acme-agent": "acmens.agents.harness:agent"}})
    get_harness("acme-agent")
    importlib.import_module("acmens.envs.mazes")

    assert extensions.origin_of_module("acmens.agents.harness").distribution == "acme-agents"
    assert extensions.origin_of_module("acmens.envs.mazes").distribution is None  # installed by nothing

    envs = (site.root / "acmens" / "envs").as_uri()
    site.install("acme-envs", direct_url=json.dumps({"url": envs, "dir_info": {"editable": True}}))
    assert extensions.origin_of_module("acmens.envs.mazes").label() == "acme-envs 0.1.0"


# --- listing commands ----------------------------------------------------------------------------------------------


def test_extensions_list_prints_every_component_with_its_distribution(acme, capsys):
    cli.main(["extensions", "list"])
    rows = [line.split() for line in capsys.readouterr().out.splitlines()]

    assert rows[0] == ["kind", "name", "distribution", "version", "status", "source"]
    assert next(row for row in rows if row[1] == "platoon-codeact@1")[:3] == ["harness", "platoon-codeact@1", "platoon"]
    assert [
        "reward_processor",
        "acme-length-penalty",
        "acme-platoon",
        "1.4.0",
        "ok",
        "acme_platoon.rewards:length_penalty",
    ] in rows
    assert not _imported("acme_platoon.rewards")

    cli.main(["extensions", "list", "--kind", "environment", "--json-output"])
    listed = json.loads(capsys.readouterr().out)
    assert [(item["kind"], item["name"], item["distribution"], item["builtin"]) for item in listed] == [
        ("environment", "platoon-number-search@1", "platoon", True),
        ("environment", "acme-mazes", "acme-platoon", False),
    ]

    assert typing.get_args(cli.ComponentKind) == extensions.KINDS
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["extensions", "list", "--kind", "harnesses"])
    assert exit_info.value.code == cli.EXIT_USAGE


def test_harness_and_program_lists_show_installed_components_without_importing_them(acme, capsys):
    cli.main(["harness", "list"])
    harness_rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    cli.main(["program", "list"])
    program_rows = [line.split() for line in capsys.readouterr().out.splitlines()]

    assert harness_rows[0] == ["harness", "kind", "distribution", "source"]
    assert ["platoon-react@1", "function", "platoon"] == [row for row in harness_rows if row[0] == "platoon-react@1"][
        0
    ][:3]
    assert ["acme-agent", "-", "acme-platoon", "1.4.0", "acme_platoon.agent:agent"] in harness_rows
    assert program_rows[0] == ["program", "distribution", "source"]
    assert ["acme-twice", "acme-platoon", "1.4.0", "acme_platoon.programs:TWICE"] in program_rows
    assert ["single@1", "platoon"] == [row for row in program_rows if row[0] == "single@1"][0][:2]
    assert not _imported("acme_platoon.agent") and not _imported("acme_platoon.programs")

    cli.main(["program", "show", "acme-twice@1"])
    assert capsys.readouterr().out.splitlines()[:3] == [
        "acme-twice@1",
        "source: acme_platoon.programs:twice_v1",
        "distribution: acme-platoon 1.4.0",
    ]


HARNESS_USES = "used in: environments[].harness, ep.run(harness=...), ctx.agents.spawn(harness=...)"
RANK = '''
import platoon as pt


@pt.program("acme-rank")
async def rank(ep):
    """Rank the attempts."""


def score(trajectory):
    return 1.0, {}
'''


def _row(kind: str, name: str) -> extensions.ComponentInfo:
    (info,) = [info for info in extensions.list_components(kind) if info.name == name]
    return info


def test_extensions_show_describes_a_component_from_its_metadata_as_the_list_does(acme, capsys):
    assert cli.main(["extensions", "show", "acme-agent"]) is None

    assert capsys.readouterr().out.splitlines() == [
        "acme-agent  (harness)",
        "version: unknown until it is loaded (--load imports it)",
        "distribution: acme-platoon 1.4.0",
        "entry point: platoon.harnesses:acme-agent",
        "source: acme_platoon.agent:agent",
        f"requires: {', '.join(_row('harness', 'acme-agent').requires)}",
        "status: ok",
        HARNESS_USES,
    ]
    assert not _imported("acme_platoon.agent")

    cli.main(["extensions", "show", "acme-mazes", "--json-output"])
    (shown,) = json.loads(capsys.readouterr().out)
    cli.main(["extensions", "list", "--kind", "environment", "--json-output"])
    (listed,) = [item for item in json.loads(capsys.readouterr().out) if item["name"] == "acme-mazes"]
    assert shown == {
        **listed,
        "entry_point": "platoon.environments:acme-mazes",
        "used_in": ["environments[].env"],
        "registers": None,
        "error": None,
    }
    assert not _imported("acme_platoon.entries")


def test_extensions_show_with_load_imports_the_component_to_show_its_versions(acme, site, capsys):
    cli.main(["extensions", "show", "acme-twice", "--load"])
    assert "version: 1, 2" in capsys.readouterr().out.splitlines()
    assert _imported("acme_platoon.programs") and not _imported("acme_platoon.agent")

    cli.main(["extensions", "show", "acme-twice@2", "--load", "--json-output"])
    assert [item["registers"] for item in json.loads(capsys.readouterr().out)] == [["acme-twice@2"]]
    cli.main(["extensions", "show", "acme-length-penalty", "--load"])
    assert "version: -" in capsys.readouterr().out.splitlines()

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["extensions", "show", "acme-twice@3", "--load"])
    assert exit_info.value.code == cli.EXIT_CONFIG
    assert "No component 'acme-twice@3' is installed; 'acme-twice' has ['acme-twice@1', 'acme-twice@2']" in (
        capsys.readouterr().err
    )

    # An entry point named after its module, which registers a harness of another name.
    site.module("acme_harnesses/one_shot.py", AGENT.replace('"acme-agent", version="2"', '"acme-one-shot"'))
    site.install("acme-harnesses", entry_points={"platoon.harnesses": {"acme-harnesses": "acme_harnesses.one_shot"}})
    cli.main(["extensions", "show", "acme-harnesses", "--load"])
    lines = capsys.readouterr().out.splitlines()
    assert "version: 1" in lines and "registers: acme-one-shot@1" in lines


def test_extensions_show_a_built_in_component(capsys):
    cli.main(["extensions", "show", "platoon-codeact"])
    lines = capsys.readouterr().out.splitlines()

    assert lines[:5] == [
        "platoon-codeact@1  (harness)",
        "version: 1",
        f"distribution: platoon {extensions.platoon_version()}",
        "entry point: - (built into platoon)",
        f"source: {_row('harness', 'platoon-codeact@1').value}",
    ]
    assert lines[5:] == ["requires: - (part of platoon)", "status: ok", HARNESS_USES]

    cli.main(["extensions", "show", "single@1", "--json-output"])
    (single,) = json.loads(capsys.readouterr().out)
    assert (single["kind"], single["builtin"], single["registers"]) == ("program", True, ["single@1"])
    assert single["used_in"] == ["environments[].program"]


def test_extensions_show_gives_each_kind_that_uses_a_name(site, capsys):
    site.module("acme_rank/parts.py", RANK)
    site.install(
        "acme-rank",
        "2.0.0",
        entry_points={
            "platoon.programs": {"acme-rank": "acme_rank.parts:rank"},
            "platoon.reward_processors": {"acme-rank": "acme_rank.parts:score"},
        },
        requires=["platoon>=0.1"],
    )

    cli.main(["extensions", "show", "acme-rank", "--json-output"])
    shown = json.loads(capsys.readouterr().out)
    assert [(item["kind"], item["requires"]) for item in shown] == [
        ("program", ["platoon>=0.1"]),
        ("reward_processor", ["platoon>=0.1"]),
    ]

    cli.main(["extensions", "show", "acme-rank", "--kind", "reward_processor"])
    out = capsys.readouterr().out
    assert out.startswith("acme-rank  (reward_processor)\n") and "used in: environments[].reward_processor" in out


def test_extensions_show_reports_what_keeps_a_component_from_loading(site, capsys):
    site.install("acme-broken", entry_points={"platoon.programs": {"acme-broken": "acme_missing.programs:run"}})
    site.module("acme_future/agent.py", AGENT.replace("acme-agent", "acme-future"))
    site.install(
        "acme-future",
        entry_points={"platoon.harnesses": {"acme-future": "acme_future.agent:agent"}},
        requires=["platoon>=99"],
    )

    cli.main(["extensions", "show", "acme-future"])  # from metadata: the requirement already excludes it
    status = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("status: "))
    assert status == f"status: {_row('harness', 'acme-future').status}" and "requires platoon>=99" in status
    assert not _imported("acme_future.agent")

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["extensions", "show", "acme-broken", "--load"])
    assert exit_info.value.code == cli.EXIT_CONFIG
    status = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("status: "))
    assert status.startswith("status: failed to load: platoon.programs entry point acme-broken = ") and (
        "ModuleNotFoundError: No module named 'acme_missing'" in status
    )


def test_extensions_show_an_unknown_name_is_a_config_error_with_the_nearest_names(acme, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["extensions", "show", "acme-agnet"])
    assert exit_info.value.code == cli.EXIT_CONFIG
    assert "No component named 'acme-agnet' is installed; did you mean 'acme-agent' or " in capsys.readouterr().err

    with pytest.raises(SystemExit):
        cli.main(["extensions", "show", "acme_platoon.agent:agent"])
    assert "is not a component name" in capsys.readouterr().err


def test_an_unknown_program_is_a_config_error(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["program", "show", "acme-missing"])

    assert exit_info.value.code == cli.EXIT_CONFIG
    assert "Unknown program 'acme-missing'" in capsys.readouterr().err


# --- the retired plugin group --------------------------------------------------------------------------------------


def test_the_platoon_plugins_group_and_discover_entry_points_are_gone(tmp_path):
    import platoon.registry
    from platoon.train.components import EnvironmentConfig
    from platoon.utils.config import ConfigError, resolve_config

    assert not hasattr(platoon.registry, "discover_entry_points")
    assert "discover_entry_points" not in {field.name for field in dataclasses.fields(EnvironmentConfig)}
    config = tmp_path / "train.yaml"
    config.write_text("environments:\n  - service: harbor\n    discover_entry_points: true\n")
    from platoon.eval.run import PlatoonEvalConfig

    with pytest.raises(ConfigError, match="environments.0.discover_entry_points: unknown key"):
        resolve_config(config, [], PlatoonEvalConfig)
