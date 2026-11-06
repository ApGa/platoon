"""Root roles in a run of several policies (``environments[].agents.root_role``, ``@pt.program(root_role=...)``): the
role each entry's root takes, by entry; the config binds that role to its policy as any other (``agents.roles``), so a
root may sample a frozen policy through it while a frozen ``agents.policy`` stays refused, and so is a run whose roots
alone sample, none of them the agents' default; no root takes a role of the agents that judge others; a root role that
looks like a misspelling is warned of, reading the programs' roles only when resolving them is allowed; and
``identity.json`` records every entry's root role with the bindings, which a resume may not change for an entry the run
has, while entries may join the run or leave it."""

from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import platoon as pt
from platoon import cli
from platoon.train import identity
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.policy.spec import MAIN, agent_bindings, policy_problems, policy_warnings, root_roles
from platoon.utils.config import ConfigError, build_config

SETTINGS: dict[str, Any] = {
    "train": {"model_name": "Qwen/Qwen3-4B-Instruct-2507", "renderer_name": "qwen3_instruct"},
    "eval": {"every": 0},
    "log_path": "logs",
}
NUMBER_SEARCH = "python://platoon.envs.number_search.env:NumberSearchEnvironment"
FROZEN = {"model_name": "Qwen/Qwen3-4B-Instruct-2507", "renderer_name": "qwen3_instruct", "trainable": False}
GENERATOR = {"policies": {"generator": FROZEN}, "agents": {"roles": {"generator": {"policy": "generator"}}}}
# A frozen judge that a canary entry's roots sample, through a role of their own.
CANARY = {
    "policies": {"generator": FROZEN, "judge": FROZEN},
    "agents": {"roles": {"generator": {"policy": "generator"}, "canary": {"policy": "judge"}}},
}


@pt.program("roottest-authoring", root_role="generator")
async def authoring(ep: pt.EpisodeContext) -> pt.Outcome:
    await ep.run()
    return await ep.grade()


@pt.program("roottest-misspelt", root_role="generater")
async def misspelt(ep: pt.EpisodeContext) -> pt.Outcome:
    await ep.run()
    return await ep.grade()


def entry(name: str, **settings: Any) -> dict[str, Any]:
    return {"name": name, "service": NUMBER_SEARCH, "harness": "platoon-react", **settings}


def config(*entries: dict[str, Any], **sections: Any) -> PlatoonTinkerRLTrainerConfig:
    settings = {**copy.deepcopy(SETTINGS), **copy.deepcopy(sections)}
    if entries:
        settings["environments"] = list(entries)
    return build_config(PlatoonTinkerRLTrainerConfig, settings)


ENTRIES = (
    entry("solve", agents={"root_role": "solver"}),
    entry("author", program="roottest-authoring"),
    entry("plain", program="roottest-authoring", agents={"root_role": "policy"}),
    entry("floor"),
)


def test_each_entry_s_root_takes_its_own_role_else_its_program_s_else_policy():
    roles = {"solve": "solver", "author": "generator", "plain": "policy", "floor": "policy"}
    assert root_roles(config(*ENTRIES)) == roles
    # Without the programs (no import): only the roles the entries name.
    assert root_roles(config(*ENTRIES), programs=False) == {**roles, "author": "policy"}
    assert root_roles(config(entry("floor"))) == {"floor": "policy"}
    assert root_roles(config()) == {"default": "policy"}  # the config's one entry
    assert root_roles(SimpleNamespace()) == {}  # a config built without environments


def test_a_root_role_bound_to_a_frozen_policy_loads_and_the_root_s_policy_is_the_config_s():
    loaded = config(entry("author", program="roottest-authoring"), entry("floor"), **GENERATOR)

    assert policy_problems(loaded) == [] and policy_warnings(loaded) == []
    bindings = agent_bindings(loaded)
    assert (bindings.default, dict(bindings.roles), bindings.sampled()) == (
        MAIN,
        {"generator": "generator"},
        (MAIN, "generator"),
    )


def test_a_frozen_agents_policy_is_still_refused_and_says_how_a_root_samples_a_frozen_policy():
    with pytest.raises(ConfigError) as raised:
        config(
            entry("author", program="roottest-authoring"),
            policies={"generator": FROZEN},
            agents={"policy": "generator"},
        )
    assert raised.value.problems == [
        "agents.policy: 'generator' is frozen, and the agents' default, which the root agents sample, is a policy the "
        "run trains: name main or a policy the run trains, and bind a role to the frozen one (agents.roles.<role>."
        "policy) instead, which an entry's root agent may take (environments[].agents.root_role)"
    ]


def test_a_run_whose_roots_alone_sample_and_none_samples_the_default_is_refused():
    with pytest.raises(ConfigError) as raised:
        config(entry("author", agents={"root_role": "generator", "max_depth": 0}), **GENERATOR)
    assert raised.value.problems == [
        "agents.policy: no agent samples main, the agents' default, which the run trains: every entry's root agent "
        "takes a role bound to another policy (environments[].agents.root_role) and spawns no agent (agents.max_depth: "
        "0), so no row would ever train it: give an entry's root a role that samples main, or let its agents spawn "
        "subagents"
    ]
    # Its subagents sample main; another entry's roots do; an unbound role's roots do.
    config(entry("author", agents={"root_role": "generator"}), **GENERATOR)
    config(entry("author", agents={"root_role": "generator", "max_depth": 0}), entry("floor"), **GENERATOR)
    config(entry("author", agents={"root_role": "solver", "max_depth": 0}), **GENERATOR)


@pytest.mark.parametrize("role", ["verifier", "judge"])
def test_no_root_takes_a_role_of_the_agents_that_judge_others(role, tmp_path, capsys):
    """Refused in a training config and in an evaluation's, with its judges off too: such a root would train on
    agents.policy, run with judges off and be booked as a verifier."""

    environments = [entry("canaries", agents={"root_role": role})]
    training = tmp_path / "train.yaml"
    training.write_text(yaml.safe_dump({**SETTINGS, "environments": environments}))
    evaluation = tmp_path / "eval.yaml"
    inference = {**SETTINGS["train"], "judges": False}
    evaluation.write_text(yaml.safe_dump({"inference": inference, "environments": environments}))

    with pytest.raises(SystemExit) as exited:
        cli.main(["config", "validate", "--no-resolve", str(training), str(evaluation)])

    refused = (
        f"agents.root_role: {role!r} is the role of agents platoon opens to judge others, which never train, so no "
        "root takes it: give the root a role of its own, and bind that role to the judge's policy "
        "(agents.roles.<role>.policy)"
    )
    out = capsys.readouterr().out
    assert exited.value.code == cli.EXIT_CONFIG and f"FAIL {training}" in out and f"FAIL {evaluation}" in out
    assert out.count(refused) == 2
    # A root that judges takes a role of its own, bound to the frozen judge.
    loaded = config(entry("canaries", agents={"root_role": "canary"}), entry("floor"), **CANARY)
    assert policy_problems(loaded) == [] and policy_warnings(loaded) == []


def test_a_root_role_that_looks_like_a_misspelling_is_warned_of():
    assert policy_warnings(config(entry("author", agents={"root_role": "generater"}), **GENERATOR)) == [
        "environments[author].agents.root_role: agents.roles names no role 'generater', so this entry's root agents "
        "sample agents.policy (main) and may call every tool; did you mean 'generator'?"
    ]
    assert policy_warnings(config(entry("helpers", agents={"root_role": "subagents"}))) == [
        "environments[helpers].agents.root_role: agents.roles names no role 'subagents', so this entry's root agents "
        "sample agents.policy (main) and may call every tool; did you mean 'subagent'?"
    ]
    # A free name, a role agents.roles names and platoon's own are no misspellings, nor is a name close to verifier or
    # judge, which no root takes; nor is a role of agents.roles close to one of platoon's that an entry's root takes,
    # since platoon then opens agents of it.
    assert policy_warnings(config(entry("solve", agents={"root_role": "solver"}), **GENERATOR)) == []
    assert policy_warnings(config(entry("author", agents={"root_role": "generator"}), **GENERATOR)) == []
    assert policy_warnings(config(entry("helpers", agents={"root_role": "subagent"}))) == []
    assert policy_warnings(config(entry("canaries", agents={"root_role": "judges"}))) == []
    judges = {"policies": {"judges": FROZEN}, "agents": {"roles": {"judges": {"policy": "judges"}}}}
    assert policy_warnings(config(entry("canaries", agents={"root_role": "judges"}), **judges)) == []
    assert policy_warnings(config(entry("canaries"), **judges)) == [
        "agents.roles.judges: platoon opens no agent of this role, so its settings reach only the agents a harness "
        "mints with it; did you mean 'judge'?"
    ]
    # A program's root role is its program's to misspell, and is read only where resolving programs is allowed.
    misspelt = config(entry("author", program="roottest-misspelt"), **GENERATOR)
    assert policy_warnings(misspelt) == [
        "environments[author].program: agents.roles names no role 'generater', the root role program "
        "'roottest-misspelt' declares, so this entry's root agents sample agents.policy (main) and may call every "
        "tool; did you mean 'generator'?"
    ]
    assert policy_warnings(misspelt, resolve=False) == []


@pytest.fixture
def program_module(tmp_path: Path, monkeypatch):
    """Write the module ``name`` into a directory on ``sys.path``, which a config's ``program: name:program`` imports,
    and forget it again."""

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    written: list[str] = []

    def write(name: str, source: str) -> None:
        (tmp_path / f"{name}.py").write_text(source)
        sys.modules.pop(name, None)
        importlib.invalidate_caches()
        written.append(name)

    yield write
    for name in written:
        sys.modules.pop(name, None)


# A module whose program declares the root role generator, registered under the module's name.
PROGRAM = """
import platoon as pt


@pt.program("roottest-{name}", root_role="generator")
async def program(ep):
    await ep.run()
    return await ep.grade()
"""


def test_config_validate_no_resolve_reads_no_program_s_root_role(program_module, tmp_path, capsys):
    program_module("roottest_unread", PROGRAM.format(name="unread"))
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump({**SETTINGS, "environments": [entry("author", program="roottest_unread:program")]}))

    cli.main(["config", "validate", "--no-resolve", str(path)])

    assert capsys.readouterr().out.splitlines()[0] == f"ok   {path} (PlatoonTinkerRLTrainerConfig)"
    assert "roottest_unread" not in sys.modules  # --no-resolve imports none of the config's programs
    loaded = config(entry("author", program="roottest_unread:program"))
    assert policy_warnings(loaded) == [] and "roottest_unread" in sys.modules  # resolving them imports them


# --- identity ---------------------------------------------------------------------------------------------------------


def written_identity(run_dir: Path, loaded: Any, monkeypatch) -> dict[str, Any]:
    """The ``identity.json`` that a first start of ``loaded`` writes into ``run_dir``."""

    monkeypatch.setattr(identity, "resumes", lambda run_dir: False)
    identity.check_run_identity(run_dir, identity.run_identity(loaded, []), {})
    return json.loads((run_dir / identity.IDENTITY_NAME).read_text())


def resume(run_dir: Path, loaded: Any, monkeypatch) -> None:
    """A start of ``loaded`` that resumes the run in ``run_dir``."""

    monkeypatch.setattr(identity, "resumes", lambda run_dir: True)
    identity.check_run_identity(run_dir, identity.run_identity(loaded, []), {})


def test_the_identity_records_every_entry_s_root_role_with_the_bindings(tmp_path: Path, monkeypatch):
    loaded = config(*ENTRIES, **GENERATOR)
    roles = {"solve": "solver", "author": "generator", "plain": "policy", "floor": "policy"}

    assert identity.policy_table(loaded)["agents"] == {
        "policy": MAIN,
        "roles": {"generator": "generator"},
        "root_roles": roles,
    }
    assert written_identity(tmp_path, loaded, monkeypatch)["agents"]["root_roles"] == roles
    resume(tmp_path, loaded, monkeypatch)  # the same root roles resume


def test_an_identity_without_root_roles_is_as_it_was(tmp_path: Path, monkeypatch):
    loaded = config(entry("floor"), entry("plain", agents={"root_role": "policy"}), **GENERATOR)

    # Every root takes the role policy: the record is the one written before root roles.
    assert written_identity(tmp_path, loaded, monkeypatch)["agents"] == {
        "policy": MAIN,
        "roles": {"generator": "generator"},
    }
    # A run of one policy records no table, root roles or not: every root samples main, whatever its role.
    assert identity.policy_table(config(*ENTRIES)) == {}


@pytest.mark.parametrize(
    ("entries", "problems"),
    [
        (ENTRIES, []),
        (
            (entry("solve"), *ENTRIES[1:]),
            [
                "environments[solve].agents.root_role: the run's root agents take the role 'solver', and this start's "
                "'policy'"
            ],
        ),
        (
            (
                entry("solve", agents={"root_role": "solver"}),
                entry("author", program="roottest-authoring", agents={"root_role": "critic"}),
                *ENTRIES[2:],
            ),
            [
                "environments[author].agents.root_role: the run's root agents take the role 'generator', and this "
                "start's 'critic'"
            ],
        ),
        (
            (*ENTRIES[:2], entry("plain", program="roottest-authoring"), ENTRIES[3]),
            [
                "environments[plain].agents.root_role: the run's root agents take the role 'policy', and this start's "
                "'generator'"
            ],
        ),
        # The run's last entries of other roles than policy keep theirs, or are dropped, or take policy.
        ((ENTRIES[2], ENTRIES[3]), []),
        (
            (entry("solve"), entry("author", program="roottest-authoring", agents={"root_role": "policy"}), ENTRIES[3]),
            [
                "environments[solve].agents.root_role: the run's root agents take the role 'solver', and this start's "
                "'policy'",
                "environments[author].agents.root_role: the run's root agents take the role 'generator', and this "
                "start's 'policy'",
            ],
        ),
    ],
)
def test_a_resume_that_changes_the_root_role_of_an_entry_the_run_has_names_it(
    entries, problems, tmp_path: Path, monkeypatch
):
    previous = written_identity(tmp_path, config(*ENTRIES, **GENERATOR), monkeypatch)
    assert identity.policy_changes(previous, identity.run_identity(config(*entries, **GENERATOR), [])) == problems


def test_a_resume_may_drop_or_add_an_entry_whatever_its_root_role(tmp_path: Path, monkeypatch):
    run = config(*ENTRIES, **CANARY)
    written = written_identity(tmp_path, run, monkeypatch)

    resume(tmp_path, config(*ENTRIES[1:], **CANARY), monkeypatch)  # solve left the run
    resume(tmp_path, config(ENTRIES[2], ENTRIES[3], **CANARY), monkeypatch)  # every root of another role left
    assert json.loads((tmp_path / identity.IDENTITY_NAME).read_text()) == written  # the run's record, as it was
    canaries = entry("canaries", agents={"root_role": "canary"})
    resume(tmp_path, config(*ENTRIES, canaries, **CANARY), monkeypatch)
    resume(tmp_path, config(*ENTRIES[1:], entry("more"), **CANARY), monkeypatch)

    # The run's record takes up the entries that joined it, so that a later resume checks their roots too.
    recorded = json.loads((tmp_path / identity.IDENTITY_NAME).read_text())
    assert recorded["agents"]["root_roles"] == {
        **written["agents"]["root_roles"],
        "canaries": "canary",
        "more": "policy",
    }
    assert {key: value for key, value in recorded.items() if key not in ("agents", "written_at")} == {
        key: value for key, value in written.items() if key not in ("agents", "written_at")
    }
    with pytest.raises(ConfigError) as raised:
        resume(tmp_path, config(*ENTRIES, entry("canaries", agents={"root_role": "generator"}), **CANARY), monkeypatch)
    assert raised.value.problems[1:-1] == [
        "environments[canaries].agents.root_role: the run's root agents take the role 'canary', and this start's "
        "'generator'"
    ]


def test_a_root_of_another_role_in_a_run_that_recorded_no_root_role_is_refused(tmp_path: Path, monkeypatch):
    written_identity(tmp_path, config(entry("floor"), **CANARY), monkeypatch)

    resume(tmp_path, config(entry("floor"), entry("more"), **CANARY), monkeypatch)  # a root of the role policy joins
    for entries in [
        (entry("floor", agents={"root_role": "canary"}),),
        (entry("floor"), entry("canaries", agents={"root_role": "canary"})),
    ]:
        label = entries[-1]["name"]
        with pytest.raises(ConfigError) as raised:
            resume(tmp_path, config(*entries, **CANARY), monkeypatch)
        assert raised.value.problems[1:-1] == [
            f"environments[{label}].agents.root_role: this start's root agents take the role 'canary', and the run "
            "recorded no root role, so its root agents took the role 'policy': its record cannot tell an entry that "
            "joins the run from one whose roots change role"
        ]


def test_a_program_that_does_not_resolve_stops_a_start_that_records_root_roles(program_module, tmp_path, monkeypatch):
    program_module("roottest_flaky", "raise ImportError('half written')\n")
    loaded = config(entry("author", program="roottest_flaky:program"), entry("floor"), **GENERATOR)

    with pytest.raises(ConfigError) as raised:
        written_identity(tmp_path, loaded, monkeypatch)
    assert raised.value.problems == [
        "environments[author].program: 'roottest_flaky:program' does not resolve (ImportError: half written), so the "
        "run cannot record the role it declares for the entry's root agents (identity.json), and its rollouts could "
        "not run it"
    ]
    assert not (tmp_path / identity.IDENTITY_NAME).exists()  # no guess recorded

    program_module("roottest_flaky", PROGRAM.format(name="flaky"))  # fixed: its first start records its role
    assert written_identity(tmp_path, loaded, monkeypatch)["agents"]["root_roles"] == {
        "author": "generator",
        "floor": "policy",
    }
    resume(tmp_path, loaded, monkeypatch)
