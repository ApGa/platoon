"""The ``platoon-harness`` program's load check, ``check_entry``: it refuses what an entry and its team decide, naming
each problem, when the config loads, and runs nothing to do so."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from platoon_harness import Team
from platoon_harness.credit import PROPAGATE
from teams_toys import ASKING, SOLO, named, team
from testing_toys import LOCKBOX

import platoon as pt
from platoon import cli
from platoon.harness.builtin.platoon_harness.bridge import check_entry
from platoon.options import Phase
from platoon.train.components import EnvironmentConfig

PAIR = named(team(alpha={"harness": ASKING}, beta={"harness": ASKING}))
TREE = named(team(lead={"harness": ASKING, "may_spawn": ["worker"]}, worker={"harness": ASKING}))
# A spawned agent's `adopt` child: platoon's spawn of `mid` returns only once `leaf` has ended.
ADOPTING = {
    "lead": {"harness": ASKING, "may_spawn": ["mid"]},
    "mid": {"harness": ASKING, "may_spawn": ["leaf"]},
    "leaf": {"harness": ASKING, "on_parent_exit": "adopt"},
}
UNENDING = named(
    Team.model_validate(
        {"roles": {"solver": {"harness": ASKING, "approvals": "allow"}}, "topology": {"until": "program"}}
    )
)
# An `until` only a person reaches: the role it names stays until told to finish.
STAYING = named(
    Team.model_validate(
        {
            "roles": {"solver": {"harness": ASKING, "approvals": "allow", "stays": True}},
            "topology": {"until": "role:solver"},
        }
    )
)
PROPAGATING = {"producer": PROPAGATE, "retained": 0.5}


def refusal(phase: Phase = "train", **entry: Any) -> str | None:
    """What ``check_entry`` refuses in ``phase`` of an entry that runs the program, or None."""

    fields: dict[str, Any] = {"service": LOCKBOX, "harness": "platoon-harness", "program": "platoon-harness", **entry}
    environment = EnvironmentConfig(**fields)
    try:
        check_entry({}, pt.CheckContext(environment=environment, phase=phase, rollout=None))
    except ValueError as refused:
        return str(refused)
    return None


def solo(**role: Any) -> dict[str, Any]:
    """The ``harness_options`` of a team of one role, ``solver``, with ``role``'s settings."""

    return {"team": named(team(solver={"harness": ASKING, **role}))}


@pytest.mark.parametrize(
    ("entry", "refused"),
    [
        pytest.param({"harness": "platoon-react"}, "harness: program platoon-harness runs harness", id="harness"),
        pytest.param(
            {"harness_options": {"team": SOLO, "tries": 2}}, "harness_options.tries: harness platoon-harness", id="key"
        ),
        pytest.param({"harness_options": {"team": "crew"}}, "harness_options.team: name a team file", id="team"),
        pytest.param({"harness_options": solo(model="m")}, "team roles.solver.model: a trainable", id="model"),
        pytest.param({"harness_options": solo(auth="key")}, "team roles.solver.auth: a trainable", id="auth"),
        pytest.param({"harness_options": solo(approvals="ask")}, "team roles.solver.approvals: ", id="approvals"),
        pytest.param(
            {"harness_options": solo(allow_native_children=True)},
            "team roles.solver.allow_native_children: ",
            id="native",
        ),
        pytest.param({"harness_options": solo(workspace="worktree")}, "team roles.solver.workspace: ", id="workspace"),
        pytest.param(
            {"harness_options": {"team": STAYING}}, "team topology.until: `solver`'s agents stay until", id="until"
        ),
        pytest.param(
            {"harness_options": {"team": UNENDING}},
            "team topology.until: `program` is left to the program that owns the session, and this program",
            id="program",
        ),
        pytest.param(
            {"harness_options": solo(harness={"command": ["bin/agent"], "mode": "wire"})},
            "team roles.solver.harness: `bin/agent` is a relative path",
            id="relative",
        ),
        pytest.param({"harness_options": solo(count=0)}, "the team starts no agent", id="nobody"),
        *(
            pytest.param(
                {"harness_options": {"team": named(team(**{name: {"harness": ASKING}}))}},
                f"team roles.{name}: platoon gives the role `{name}` a meaning of its own",
                id=name,
            )
            for name in ("policy", "verifier", "judge")
        ),
        *(
            pytest.param(
                {"harness_options": {"team": PAIR}, "agents": {setting: value}},
                f"agents.{setting}: a team that starts several agents mints them",
                id=setting,
            )
            for setting, value in (
                ("fork", "copy"),
                ("child_grading", "env"),
                ("child_access", "read_only"),
                ("max_depth", 2),
                ("child_max_steps", 4),
                ("child_timeout_s", 30),
                ("budget", "depth_aware"),
            )
        ),
        pytest.param(
            {"harness_options": {"team": TREE}, "agents": {"child_grading": "verifier"}},
            "agents.child_grading: platoon runs a child's verifier on the child's harness and options",
            id="child-verifier",
        ),
        pytest.param(
            {"harness_options": {"team": named(team(**ADOPTING))}},
            "team roles.leaf.on_parent_exit: platoon's spawn of an agent of `mid` returns only once the agent's "
            "children have ended",
            id="adopt",
        ),
        pytest.param(
            {"harness_options": {"team": SOLO, "rewards": {"solver": PROPAGATING}}},
            "harness_options.rewards.solver: only the root plays `solver`",
            id="root-only",
        ),
        pytest.param(
            {"harness_options": {"team": TREE, "rewards": {"scribe": PROPAGATING}}},
            "harness_options.rewards.scribe: the team has no role `scribe`",
            id="no-role",
        ),
        pytest.param(
            {
                "harness_options": {"team": TREE, "rewards": {"worker": PROPAGATING}},
                "reward_processor": "acme:score",
                "credit": "keyed",
            },
            "reward_processor: a producer's row lands where",
            id="processor",
        ),
        pytest.param(
            {"harness_options": {"team": TREE, "rewards": {"worker": {"producer": "acme_nowhere:produce"}}}},
            "harness_options.rewards.worker.producer: `acme_nowhere:produce` does not resolve",
            id="producer",
        ),
        pytest.param(
            {"harness_options": {"team": TREE, "rewards": {"worker": "propagate"}}},
            "harness_options.rewards.worker is {producer: <module:attr>",
            id="rewards",
        ),
        pytest.param(
            {
                "harness_options": {
                    "team": PAIR,
                    "rewards": {"alpha": PROPAGATING, "beta": {**PROPAGATING, "retained": 1}},
                }
            },
            f"harness_options.rewards.beta: `{PROPAGATE}` runs once a rollout, and rewards.alpha runs it with other",
            id="settings",
        ),
        pytest.param(
            {"harness_options": {"team": TREE, "rewards": {"worker": PROPAGATING}}, "credit": "group"},
            "credit: each span of a role `rewards` names trains on its own row only under a credit that reads spans, "
            "and 'group' reads none: write `credit: keyed`",
            id="credit",
        ),
    ],
)
def test_check_entry_refuses_what_the_config_decides(entry: dict[str, Any], refused: str):
    assert refused in (refusal("train", **{"harness_options": {"team": SOLO}, **entry}) or "")


def test_check_entry_passes_what_runs_and_runs_no_harness():
    """A team's harness is described when its session starts, so one that names no driver passes the load check."""

    nowhere = named(team(lead={"harness": "acme_nowhere:driver", "may_spawn": ["worker"]}, worker={"harness": ASKING}))
    spawning = {"budget": "depth_aware", "child_grading": "env"}

    assert refusal(harness_options={"team": SOLO}) is None
    assert refusal(harness_options={"team": TREE}, agents={"budget": "linear"}) is None
    # `adopt` under the entry, which no spawn started, and in a team that starts several, which mints every agent.
    entry_child = named(team(lead={"harness": ASKING, "may_spawn": ["mid"]}, mid=ADOPTING["leaf"]))
    assert refusal(harness_options={"team": entry_child}) is None
    assert refusal(harness_options={"team": named(team(**ADOPTING, beta={"harness": ASKING}))}) is None
    assert refusal(harness_options={"team": nowhere}, agents=spawning) is None
    rewards = {"team": TREE, "rewards": {"worker": PROPAGATING}}
    assert refusal(harness_options=rewards, agents=spawning, credit="keyed") is None
    assert refusal(harness_options=rewards, agents=spawning) is None  # credited as the run chooses
    assert refusal("eval", harness_options=rewards, agents=spawning, credit="group") is None  # it trains nothing
    both = {"team": PAIR, "rewards": {"alpha": PROPAGATING, "beta": PROPAGATING}}
    assert refusal(harness_options=both, credit="keyed") is None


def validate(tmp_path: Path, team: str, harness_options: dict[str, Any], **sections: Any) -> None:
    """Run ``platoon config validate`` from ``tmp_path`` on a training config in ``tmp_path / "recipe"``, beside its
    team file, whose one entry runs the program with ``harness_options``."""

    (tmp_path / "recipe").mkdir()
    (tmp_path / "recipe" / "crew.toml").write_text(team)
    config = tmp_path / "recipe" / "train.yaml"
    entry = {
        "service": LOCKBOX,
        "harness": "platoon-harness",
        "program": "platoon-harness",
        "harness_options": {"team": "crew.toml", **harness_options},
    }
    train = {"model_name": "Qwen/Qwen3-4B", "renderer_name": "qwen3", "batch_size": 1, "max_training_steps": 1}
    stats = {"experiment_name": "e", "trial_name": "t"}
    settings = {"train": train, "log_path": str(tmp_path / "logs"), "stats": stats, **sections}
    config.write_text(yaml.safe_dump({**settings, "environments": [entry]}))
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(tmp_path)
        cli.main(["config", "validate", str(config)])


def test_a_refused_entry_stops_the_config_naming_its_team_file_beside_it(tmp_path, capsys):
    """``platoon config validate`` exits 78 on a team file the load check refuses, named beside the config and read
    from another working directory."""

    with pytest.raises(SystemExit) as exited:
        validate(tmp_path, f'[roles.solver]\nharness = "{ASKING}"\nmodel = "m"\napprovals = "allow"\n', {})

    captured = capsys.readouterr()
    assert exited.value.code == cli.EXIT_CONFIG
    assert "environments.0.program (train): team roles.solver.model: a trainable team names no model" in (
        captured.out + captured.err
    )
    assert not Path(tmp_path / "logs").exists()


def test_an_entry_without_credit_takes_keyed_for_its_rewards_in_a_run_that_trains_several_policies(tmp_path, capsys):
    """Such a run credits an entry that names no credit with keyed, which reads spans, so the load check lets its
    rewards through."""

    crew = (
        f'[roles.lead]\nharness = "{ASKING}"\napprovals = "allow"\nmay_spawn = ["worker"]\n'
        f'[roles.worker]\nharness = "{ASKING}"\napprovals = "allow"\n'
    )
    worker = {"model_name": "Qwen/Qwen3-4B", "renderer_name": "qwen3"}

    validate(
        tmp_path,
        crew,
        {"rewards": {"worker": PROPAGATING}},
        policies={"worker": worker},
        agents={"roles": {"worker": {"policy": "worker"}}},
    )

    assert capsys.readouterr().out.startswith("ok ")
