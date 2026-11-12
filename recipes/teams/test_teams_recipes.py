"""The teams recipes (``textcraft-synth/``): RAO's recursive arm run as a team (``tree``) and a mesh of peers that train
on ``propagate`` rows centred by ``credit: keyed`` (``mesh``).  Each config loads from any directory, resolves every
component it names and validates, and the ``platoon-harness`` program's load check passes its team; the tree differs
from the RAO recipe's recursive arm only in its harness, program and names.  CPU only; the TextCraft extension
(``platoon-textcraft``) and the ``platoon-harness`` extra must be installed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from platoon_harness.credit import PROPAGATE

from platoon import cli
from platoon.harness.builtin.platoon_harness.bridge import entry_of, load
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.utils.config import load_config

HERE = Path(__file__).resolve().parent
ARMS_DIR = HERE / "textcraft-synth"
RAO_RECURSIVE = HERE.parent / "rao" / "textcraft-synth" / "recursive.yaml"
TEAMS_ARMS = ("tree", "mesh")


def _load(path: Path) -> tuple[Any, dict[str, Any]]:
    return load_config(["--config", str(path)], PlatoonTinkerRLTrainerConfig)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        value = {str(index): item for index, item in enumerate(value)}
    if isinstance(value, dict):
        return {key: item for name, child in value.items() for key, item in _flatten(child, f"{prefix}{name}.").items()}
    return {prefix.rstrip("."): value}


def test_the_directory_holds_exactly_the_arms_and_their_teams():
    assert sorted(path.name for path in ARMS_DIR.iterdir()) == [
        "common.yaml",
        "mesh.toml",
        "mesh.yaml",
        "tree.toml",
        "tree.yaml",
    ]


@pytest.mark.parametrize("arm", TEAMS_ARMS)
def test_every_teams_recipe_resolves(arm, tmp_path, monkeypatch, capsys):
    """As a run does when it starts, from another directory: the named environment, the harness, the program, the
    reward processor and the service resolve, no key is deprecated, and the program's load check passes the team
    the config names beside it."""

    monkeypatch.chdir(tmp_path)

    assert cli.config_validate(ARMS_DIR / f"{arm}.yaml", resolve=True, strict=True) == 0, capsys.readouterr().out


def test_the_tree_is_raos_recursive_arm_through_the_bridge():
    """The tree differs from the RAO recipe's recursive arm in its harness, its program and their options, and its run's
    names alone; its team starts one agent, which runs on the root's own endpoint, and spawns agents of its role."""

    rao, tree = (_flatten(_load(path)[1]) for path in (RAO_RECURSIVE, ARMS_DIR / "tree.yaml"))
    differing = {key for key in rao.keys() | tree.keys() if rao.get(key) != tree.get(key)}
    config, _ = _load(ARMS_DIR / "tree.yaml")
    (environment,) = config.environments
    team = load(environment.harness_options["team"])

    assert not [
        key for key in differing if not key.startswith(("environments.0.harness", "environments.0.program", "stats."))
    ]
    assert (environment.harness, environment.program) == ("platoon-harness", "platoon-harness")
    assert entry_of(team) == "crafter" and team.roles["crafter"].may_spawn == ("crafter",)


def test_the_mesh_trains_its_peers_on_propagate_centred_by_keyed():
    config, _ = _load(ARMS_DIR / "mesh.yaml")
    (environment,) = config.environments
    team = load(environment.harness_options["team"])

    assert (environment.harness, environment.program, environment.credit) == (
        "platoon-harness",
        "platoon-harness",
        "keyed",
    )
    assert environment.harness_options["rewards"] == {"peer": {"producer": PROPAGATE, "retained": 0.5}}
    assert entry_of(team) is None and team.recipients() == ("@peer-1", "@peer-2", "@peer-3")
    assert config.train.workflow_config.group_size == 8
    assert config.train.workflow_config.rollout_config.max_steps == 75  # shared by every peer, which is minted
