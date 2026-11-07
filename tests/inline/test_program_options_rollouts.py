"""A program's options in its rollouts (``ep.options``): the route's ``program_options``, parsed in the rollout against
the program's schema with its defaults (an evaluation route's with the entry's ``eval_overrides`` merged in), None for
a program that declares none; options the program does not take end the rollout as an error, never a score, before
the program runs; ``platoon.testing.run_episode`` takes them and refuses bad ones before anything runs; and what
``platoon program show`` prints of a program's options and check."""

from __future__ import annotations

import asyncio
import enum
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

import pydantic
import pytest
from inline_fakes import SERVICE, ToyService, routed_task

import platoon as pt
from platoon import cli
from platoon.config_defs import RolloutConfig
from platoon.harness.rollout import EmbeddedGateway, run_harness_rollout
from platoon.testing import ScriptedPolicy, run_episode
from platoon.train.auto import environment_route
from platoon.train.components import EnvironmentConfig
from platoon.utils.config import ConfigError


@dataclass(frozen=True)
class Budget:
    goal_prefix: str = "goal:"
    rounds: int = field(default=1, metadata={"description": "Rounds the root runs"})


def refuse_many(options: Budget, ctx: pt.CheckContext) -> None:
    if options.rounds > 3:
        raise ValueError("rounds: at most 3")


@pt.program("optsrun-budgeted", options=Budget, check=refuse_many)
async def budgeted(ep: pt.EpisodeContext[Budget]) -> pt.Outcome:
    """Records the options it got on the root's misc, then runs and grades the root."""

    ep.root.misc["program_options_seen"] = asdict(ep.options)
    await ep.run()
    return await ep.grade()


class Keyed(pydantic.BaseModel, extra="forbid"):
    key: str = pydantic.Field("plain", description="The key the root signs with")


@pt.program("optsrun-keyed", options=Keyed)
async def keyed(ep: pt.EpisodeContext[Keyed]) -> pt.Outcome:
    ep.root.misc["program_options_seen"] = ep.options.model_dump()
    await ep.run()
    return await ep.grade()


@pt.program("optsrun-parsed", options=lambda values: {"parsed": True, **values})
async def parsed_program(ep: pt.EpisodeContext[dict[str, Any]]) -> pt.Outcome:
    ep.root.misc["program_options_seen"] = ep.options
    await ep.run()
    return await ep.grade()


@pt.program("optsrun-plain")
async def plain(ep: pt.EpisodeContext) -> pt.Outcome:
    ep.root.misc["program_options_seen"] = ep.options
    await ep.run()
    return await ep.grade()


@dataclass(frozen=True)
class Needed:
    rounds: int  # no default: an entry must set it


@pt.program("optsrun-needed", options=Needed)
async def needed(ep: pt.EpisodeContext[Needed]) -> pt.Outcome:
    await ep.run()
    return await ep.grade()


class Pace(enum.Enum):
    SLOW = "slow"
    BRISK = "brisk"


class Tuned(pydantic.BaseModel, extra="forbid"):
    share: float = pydantic.Field(0.5, ge=0, le=1, description="Share of the reward kept")
    count: int = pydantic.Field(2, gt=0)
    mode: Literal["fast", "careful"] | None = None
    pace: Pace = Pace.SLOW
    enabled: bool = True
    label: str = ""


@pt.program("optsrun-tuned", options=Tuned)
async def tuned(ep: pt.EpisodeContext[Tuned]) -> pt.Outcome:
    await ep.run()
    return await ep.grade()


@pytest.fixture(autouse=True)
def _reset_service():
    ToyService.sessions.clear()
    yield
    ToyService.sessions.clear()


def reply(message: str) -> str:
    return "FINISH done" if message.startswith("obs:") else "SUBMIT ok"


def run(gateway, task) -> dict[str, Any]:
    config = RolloutConfig(verbose=False, return_dict=True)
    collection = asyncio.run(run_harness_rollout(task, config, gateway=EmbeddedGateway(gateway)))
    assert isinstance(collection, dict)  # config.return_dict
    return collection


def root_of(collection: dict[str, Any]) -> dict[str, Any]:
    (root,) = collection["trajectories"].values()
    return root


@pytest.mark.parametrize(
    ("program", "options", "seen"),
    [
        ("optsrun-budgeted", {"rounds": 2}, {"goal_prefix": "goal:", "rounds": 2}),
        ("optsrun-budgeted", None, {"goal_prefix": "goal:", "rounds": 1}),
        ("optsrun-keyed", {"key": "secret"}, {"key": "secret"}),
        ("optsrun-parsed", {"a": 1}, {"parsed": True, "a": 1}),
        ("optsrun-plain", None, None),
    ],
)
def test_a_program_reads_its_options_parsed_with_their_defaults(make_gateway, program, options, seen):
    gateway, sampler = make_gateway(reply)

    root = root_of(run(gateway, routed_task(goal="goal:direct", program=program, program_options=options or {})))

    assert root["misc"]["program_options_seen"] == seen
    assert root["misc"]["outcome"]["status"] == "scored" and root["reward"] == 1.0
    assert sampler.calls == ["goal:direct", "obs: submitted"]


def test_an_evaluation_route_runs_the_entry_s_eval_overrides(make_gateway):
    entry = EnvironmentConfig(
        service=SERVICE,
        harness="toy-agent",
        program="optsrun-budgeted",
        program_options={"rounds": 2},
        eval_overrides={"program_options": {"goal_prefix": "check:"}},
    )
    seen = {}
    for phase in ("train", "eval"):
        gateway, _ = make_gateway(reply)
        route = replace(environment_route(entry, phase=phase), harness="toy-agent")  # as TaskRouter draws it
        task = route.attach(pt.Task(id="t0", goal="goal:direct", max_steps=50))
        seen[phase] = root_of(run(gateway, task))["misc"]["program_options_seen"]

    assert seen == {
        "train": {"goal_prefix": "goal:", "rounds": 2},
        "eval": {"goal_prefix": "check:", "rounds": 2},
    }


def test_options_the_program_does_not_take_end_the_rollout_as_an_error_before_it_runs(make_gateway):
    """A route built past the load-time checks (by hand, here) is parsed again in the rollout: the program never runs,
    and the rollout is an error, never a score."""

    gateway, sampler = make_gateway(reply)

    root = root_of(
        run(gateway, routed_task(goal="goal:direct", program="optsrun-budgeted", program_options={"round": 2}))
    )

    outcome = root["misc"]["outcome"]
    assert outcome["status"] == "error" and outcome["failure_kind"] == "program_options"
    assert outcome["reason"] == (
        "program optsrun-budgeted: program_options.round: unknown key; did you mean program_options.rounds?"
    )
    assert "program_options_seen" not in root["misc"] and sampler.calls == []


def test_run_episode_takes_program_options_and_refuses_bad_ones_before_anything_runs():
    policy = ScriptedPolicy([])

    with pytest.raises(ConfigError) as error:
        asyncio.run(
            run_episode(
                pt.Task(id="t0", goal="goal:direct", max_steps=50),
                service=SERVICE,
                harness="toy-agent",
                program="optsrun-budgeted",
                program_options={"rounds": "two"},
                policy=policy,
            )
        )

    assert error.value.problems == ["program_options.rounds: 'two' is not a int"]
    assert policy.calls == [] and ToyService.sessions == []  # no session opened, no model called


def test_run_episode_refuses_a_required_option_left_out_and_a_program_that_does_not_resolve_before_anything_runs():
    policy = ScriptedPolicy([])
    task = pt.Task(id="t0", goal="goal:direct", max_steps=50)

    with pytest.raises(ConfigError) as error:  # as a config that sets no options for it is refused when it loads
        asyncio.run(run_episode(task, service=SERVICE, harness="toy-agent", program="optsrun-needed", policy=policy))
    assert error.value.problems == ["program_options.rounds: required key is missing"]
    with pytest.raises(ValueError, match="Unknown program 'optsrun-nowhere'"):
        asyncio.run(run_episode(task, service=SERVICE, harness="toy-agent", program="optsrun-nowhere", policy=policy))
    assert policy.calls == [] and ToyService.sessions == []  # no session opened, no model called


def test_run_episode_runs_the_program_with_its_options():
    policy = ScriptedPolicy(["SUBMIT ok", "FINISH done"])

    run_ = asyncio.run(
        run_episode(
            pt.Task(id="t0", goal="goal:direct", max_steps=50),
            service=SERVICE,
            harness="toy-agent",
            program="optsrun-budgeted",
            program_options={"rounds": 3},
            policy=policy,
        )
    )

    assert run_.root.misc["program_options_seen"] == {"goal_prefix": "goal:", "rounds": 3}
    assert run_.reward == 1.0


def test_program_show_lists_the_options_and_the_check(capsys):
    """Each option's name, type, default, the values it takes and what it does, as harness show lists a spec's."""

    cli.main(["program", "show", "optsrun-budgeted"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[3:8] == [
        f"options ({__name__}:Budget, set in an environments entry's program_options):",
        "  option       type  default  values  description",
        "  goal_prefix  str   goal:    -       -",
        "  rounds       int   1        -       Rounds the root runs",
        f"check: {__name__}:refuse_many",
    ]
    cli.main(["program", "show", "optsrun-tuned"])
    assert capsys.readouterr().out.splitlines()[3:11] == [
        f"options ({__name__}:Tuned, set in an environments entry's program_options):",
        "  option   type        default  values                description",
        "  share    float       0.5      0 to 1                Share of the reward kept",
        "  count    int         2        > 0                   -",
        "  mode     str | null  null     one of fast, careful  -",
        "  pace     Pace        slow     one of slow, brisk    -",
        "  enabled  bool        true     -                     -",
        '  label    str         ""       -                     -',
    ]
    cli.main(["program", "show", "optsrun-keyed"])
    assert capsys.readouterr().out.splitlines()[3:6] == [
        f"options ({__name__}:Keyed, set in an environments entry's program_options):",
        "  option  type  default  values  description",
        "  key     str   plain    -       The key the root signs with",
    ]
    cli.main(["program", "show", "optsrun-parsed"])
    shown = capsys.readouterr().out.splitlines()[3]
    assert shown.startswith(f"options: parsed by {__name__}:<lambda>")
    cli.main(["program", "show", "single"])
    shown = capsys.readouterr().out.splitlines()
    assert shown[3] == "options: - (program_options must be empty)"
    assert not any(line.startswith("check:") for line in shown)
