"""Options of episode programs in configs (RFC 0016): ``@pt.program(name, options=..., check=...)`` and an entry's
``program_options``, checked when a run resolves its config, before anything runs, with full dotted paths and a
did-you-mean (exit 78); refused for a program that declares none and for an entry that runs no program; changed for
evaluation rollouts by ``eval_overrides``; the program's check, run for each phase the command runs with a
``pt.CheckContext``; carried to rollouts by per-phase routes that leave out options no entry sets; layered over a
named environment's while the entry keeps its program; and recorded in ``lock.json`` with their defaults."""

from __future__ import annotations

import enum
import gc
import json
import pickle
import re
import textwrap
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import pydantic
import pytest
import yaml

import platoon as pt
from platoon import cli
from platoon.envs.named import expand_environment
from platoon.extensions import run_components
from platoon.options import schema_options
from platoon.programs import resolve_program
from platoon.task import Task
from platoon.train.auto import AutoTaskLoader, check_environment_routes, environment_route
from platoon.train.components import (
    ROUTE_MISC_KEY,
    EnvironmentConfig,
    EnvironmentRoute,
    normalize_environment_configs,
)
from platoon.utils.config import ConfigError

NUMBER_SEARCH = "python://platoon.envs.number_search.env:NumberSearchEnvironment"
WHERE = "environments.0.program_options"
# The keys of a route before program options, in order: a route whose entry sets none travels with exactly these.
ROUTE_KEYS = [
    "name",
    "rollout",
    "eval_rollout",
    "reward_processor",
    "service",
    "service_kwargs",
    "harness",
    "placement",
    "builtins",
    "on_context_overflow",
    "on_harness_exit",
    "resolve_task",
    "program",
    "harness_options",
    "agents",
]


@dataclass(frozen=True)
class Attempts:
    n: int = 4  # root attempts
    keep: Literal["best", "first_success"] = "best"

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"n: must be at least 1, got {self.n}")


# What each call of the check saw, (options, ctx), and the step budget its rollout config gave it, by phase.
SEEN: list[tuple[Any, pt.CheckContext]] = []
BUDGETS: dict[str, int | None] = {}


def budget_check(options: Attempts, ctx: pt.CheckContext) -> None:
    """Refuse more attempts than the root has steps; then change what it was given, which is a copy of its own."""

    SEEN.append((options, ctx))
    budget = BUDGETS[ctx.phase] = ctx.rollout.max_steps
    ctx.environment.program_options["n"] = 99
    ctx.rollout.max_steps = 0
    if budget is not None and budget < options.n:
        raise ValueError(f"n: {options.n} attempts need at least {options.n} steps, and max_steps is {budget}")


@pt.program("optstest-attempts", options=Attempts, check=budget_check)
async def attempts(ep: pt.EpisodeContext[Attempts]) -> pt.Outcome:
    await ep.run()
    return await ep.grade()


class Rated(pydantic.BaseModel, extra="forbid"):
    rate: float = pydantic.Field(0.5, ge=0)


@pt.program("optstest-rated", options=Rated)
async def rated(ep: pt.EpisodeContext[Rated]) -> pt.Outcome:
    await ep.run()
    return await ep.grade()


def parse_limits(values: dict[str, Any]) -> dict[str, Any]:
    if unknown := sorted(set(values) - {"most"}):
        raise ValueError(f"takes most only, not {unknown}")
    return {"most": 3, **values}


@pt.program("optstest-parsed", options=parse_limits)
async def parsed(ep: pt.EpisodeContext[dict[str, Any]]) -> pt.Outcome:
    await ep.run()
    return await ep.grade()


def parse_pattern(values: dict[str, Any]) -> dict[str, Any]:
    return {"pattern": re.compile(values.get("pattern", ".*"))}  # a value that JSON cannot hold


@pt.program("optstest-compiled", options=parse_pattern)
async def compiled(ep: pt.EpisodeContext[dict[str, Any]]) -> pt.Outcome:
    await ep.run()
    return await ep.grade()


@pt.program("optstest-plain")
async def plain(ep: pt.EpisodeContext) -> pt.Outcome:
    await ep.run()
    return await ep.grade()


def _broken_check(options: Any, ctx: pt.CheckContext) -> None:
    raise KeyError("rollout_config")


@pt.program("optstest-broken-check", options=Attempts, check=_broken_check)
async def broken_check(ep: pt.EpisodeContext[Attempts]) -> pt.Outcome:
    await ep.run()
    return await ep.grade()


class Loose(pydantic.BaseModel):  # pydantic's default for a key it does not declare: ignore it
    rate: float = 0.5
    speed: int = pydantic.Field(1, alias="pace")


class Open(pydantic.BaseModel, extra="allow"):  # keeps the keys it does not declare
    rate: float = 0.5


@pt.program("optstest-loose", options=Loose)
async def loose(ep: pt.EpisodeContext[Loose]) -> pt.Outcome:
    await ep.run()
    return await ep.grade()


class Mode(enum.Enum):
    FAST = "fast"
    SLOW = "slow"


@dataclass(frozen=True)
class Placed:
    workdir: Path = Path("work")
    mode: Mode = Mode.FAST
    scratch: Path | None = None


@pt.program("optstest-placed", options=Placed)
async def placed(ep: pt.EpisodeContext[Placed]) -> pt.Outcome:
    await ep.run()
    return await ep.grade()


@dataclass(frozen=True)
class Staged:
    stage_seconds: int = 5400  # what the program's stages take, which the rollout's timeout must leave them


def timeout_check(options: Staged, ctx: pt.CheckContext) -> None:
    """Refuse a rollout timeout that leaves the stages less than they take: a check that reads the run's settings."""

    SEEN.append((options, ctx))
    if ctx.rollout.timeout is None or ctx.rollout.timeout < options.stage_seconds:
        raise ValueError(f"rollout_config.timeout {ctx.rollout.timeout} is below the stages' {options.stage_seconds} s")


@pt.program("optstest-staged", options=Staged, check=timeout_check)
async def staged(ep: pt.EpisodeContext[Staged]) -> pt.Outcome:
    await ep.run()
    return await ep.grade()


async def _refuse_later(options: Any, ctx: pt.CheckContext) -> None:
    raise ValueError("never acceptable")


def awaitable_check(options: Any, ctx: pt.CheckContext) -> Any:
    """A plain function that returns a coroutine, which the decorator cannot tell from a check."""

    return _refuse_later(options, ctx)


@pt.program("optstest-awaitable-check", options=Attempts, check=awaitable_check)
async def awaitable_checked(ep: pt.EpisodeContext[Attempts]) -> pt.Outcome:
    await ep.run()
    return await ep.grade()


@pytest.fixture(autouse=True)
def _fresh_checks():
    SEEN.clear()
    BUDGETS.clear()
    yield
    SEEN.clear()
    BUDGETS.clear()


def entry(**keys: Any) -> dict[str, Any]:
    return {"service": NUMBER_SEARCH, "harness": "platoon-react", "program": "optstest-attempts", **keys}


def train_config(tmp_path: Path, *entries: dict, evaluates: bool = True, **sections: Any) -> Path:
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "train": {"model_name": "Qwen/Qwen3-4B", "renderer_name": "qwen3", "batch_size": 16},
                "eval": {"every": 1} if evaluates else {"every": 0},
                "log_path": str(tmp_path / "logs"),
                "stats": {"experiment_name": "exp", "trial_name": "trial"},
                "environments": list(entries) or [entry()],
                **sections,
            },
            sort_keys=False,
        )
    )
    return path


def eval_config(tmp_path: Path, *entries: dict, **inference: Any) -> Path:
    path = tmp_path / "eval.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "inference": {"model_name": "Qwen/Qwen3-4B", "output_dir": str(tmp_path / "results"), **inference},
                "environments": list(entries),
            },
            sort_keys=False,
        )
    )
    return path


def load(path: Path, *overrides: str, kind: Literal["train", "eval", "auto"] = "train"):
    return cli._resolve(path, overrides, kind).config  # noqa: SLF001 - the commands' own loading


def exit_code(argv: list[str]) -> int | str | None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    return exit_info.value.code


def problems_of(config, **kwargs: Any) -> list[str]:
    with pytest.raises(ConfigError) as error:
        run_components(config, **kwargs)
    return error.value.problems


def program_record(config, **kwargs: Any) -> dict:
    (record,) = [record for record in run_components(config, **kwargs) if record["kind"] == "program"]
    return record


def route_of(config, split: str) -> EnvironmentRoute:
    route = EnvironmentRoute.of_task(AutoTaskLoader.from_config(config, split)("train:0"))
    assert route is not None
    return route


# --- the decorator --------------------------------------------------------------------------------------------------


def test_the_decorator_records_the_schema_and_the_check_and_refuses_what_is_neither():
    record = resolve_program("optstest-attempts")
    assert record.options_schema is Attempts and record.check is budget_check
    assert record.parse_options({"n": 2}) == Attempts(n=2) and record.parse_options(None) == Attempts()
    assert resolve_program("optstest-plain").parse_options({}) is None  # a program without options gets None
    with pytest.raises(TypeError, match=r"^@program\('optstest-bad'\): options must be a schema"):
        pt.program("optstest-bad", options=Attempts(n=2))  # an instance, not a schema
    with pytest.raises(TypeError, match=r"^@program\('optstest-bad'\): check must be a function check\(options, ctx\)"):
        pt.program("optstest-bad", check="refuse")
    with pytest.raises(
        TypeError, match=r"^@program\('optstest-bad'\): check must be a plain function .*not an async def"
    ):
        pt.program("optstest-bad", check=_refuse_later)  # it runs when the config loads, where nothing awaits it


def test_a_check_that_returns_an_awaitable_refuses_the_entry_without_leaving_it_unawaited(tmp_path):
    config = load(train_config(tmp_path, entry(program="optstest-awaitable-check"), evaluates=False))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert problems_of(config) == [
            "environments.0.program: optstest-awaitable-check@1, for the entry's train rollouts: its check is async, "
            "so it never ran: a check is a plain function check(options, ctx)"
        ]
        gc.collect()
    assert not [warning for warning in caught if "never awaited" in str(warning.message)]  # closed, not dropped


# --- checked before anything runs -----------------------------------------------------------------------------------


def test_a_misspelled_option_fails_validate_and_train_with_exit_78_before_anything_runs(tmp_path, monkeypatch, capsys):
    trained = []
    monkeypatch.setattr("platoon.train.launch.run_tinker_training", trained.append)
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")  # train checks its key before anything else: past it
    config = train_config(tmp_path, entry(program_options={"kep": "best"}))
    problem = f"{WHERE}.kep: unknown key; did you mean {WHERE}.keep?"

    assert exit_code(["config", "validate", str(config)]) == cli.EXIT_CONFIG
    assert f"FAIL {config}\n  {problem}" in capsys.readouterr().out
    assert exit_code(["train", str(config)]) == cli.EXIT_CONFIG
    assert problem in capsys.readouterr().err
    assert trained == [] and not (tmp_path / "logs").exists()  # nothing ran, and no run directory was made


def test_every_problem_is_reported_at_once_where_it_is_written(tmp_path):
    config = load(
        train_config(
            tmp_path,
            entry(
                name="a", weight=1, program_options={"keep": "worst"}, eval_overrides={"program_options": {"kep": 1}}
            ),
            entry(name="b", weight=1, program_options={"n": 2}, eval_overrides={"program_options": {"n": 0}}),
        )
    )

    assert problems_of(config) == [
        "environments.0.program_options.keep: 'worst' is not one of 'best', 'first_success'",
        # The entry's own problem is not told again where its evaluation options merge over it.
        "environments.0.eval_overrides.program_options.kep: unknown key; did you mean "
        "environments.0.eval_overrides.program_options.keep?",
        "environments.1.eval_overrides.program_options.n: must be at least 1, got 0",
    ]


@pytest.mark.parametrize(
    ("program", "options", "problem"),
    [
        ("optstest-attempts", {"n": "two"}, f"{WHERE}.n: 'two' is not a int"),
        ("optstest-attempts", {"n": 0}, f"{WHERE}.n: must be at least 1, got 0"),
        ("optstest-rated", {"rat": 1}, f"{WHERE}.rat: unknown key; did you mean {WHERE}.rate?"),
        ("optstest-rated", {"rate": -1}, f"{WHERE}.rate: Input should be greater than or equal to 0"),
        ("optstest-loose", {"rat": 1}, f"{WHERE}.rat: unknown key; did you mean {WHERE}.rate?"),
        ("optstest-parsed", {"least": 1}, f"{WHERE}: takes most only, not ['least']"),
        ("optstest-plain", {"n": 1}, f"{WHERE}: program optstest-plain@1 takes no options; leave program_options out"),
        ("single", {"n": 1}, f"{WHERE}: program single@1 takes no options; leave program_options out"),
    ],
)
def test_each_kind_of_schema_refuses_what_it_does_not_take(tmp_path, program, options, problem):
    config = load(train_config(tmp_path, entry(program=program, program_options=options)))

    assert problems_of(config) == [problem]


def test_a_pydantic_schema_refuses_an_unknown_key_with_a_did_you_mean_unless_it_keeps_such_keys():
    """The path every component's Python schema takes (task samplers, admission rules, credit estimators, programs):
    a key the model does not take is refused as a config section's is, whatever the model says of extra keys, unless
    it keeps them; by pydantic's default (ignore) the key would vanish, and the default would run."""

    for model in (Rated, Loose):  # extra="forbid"; pydantic's default
        assert schema_options(model, name="acme").option_problems({"rat": 1, "colour": 2}, "opts") == [
            "opts.rat: unknown key; did you mean opts.rate?",
            "opts.colour: unknown key",
        ]
    loose = schema_options(Loose, name="acme")
    # Every problem at once: the unknown key, and what the model says of the keys it takes.
    assert loose.option_problems({"rat": 1, "rate": "fast"}, "opts") == [
        "opts.rat: unknown key; did you mean opts.rate?",
        "opts.rate: Input should be a valid number, unable to parse string as a number",
    ]
    assert loose.parse({"rate": 0.25, "pace": 3}, "opts") == Loose(rate=0.25, pace=3)  # an alias is a key it takes
    with pytest.raises(ConfigError) as error:
        resolve_program("optstest-loose").parse_options({"rat": 2})  # what ep.options would have been: rate=0.5
    assert error.value.problems == ["program_options.rat: unknown key; did you mean program_options.rate?"]
    kept = schema_options(Open, name="acme").parse({"rate": 0.25, "colour": "red"}, "opts")
    assert kept.model_extra == {"colour": "red"}  # extra="allow" keeps them, as a config's free-form model does


def test_an_entry_without_a_harness_runs_no_program_and_takes_no_program_options(tmp_path):
    config = load(
        train_config(
            tmp_path,
            {
                "service": NUMBER_SEARCH,
                "program_options": {"n": 2},
                "eval_overrides": {"program_options": {"n": 3}},
            },
        )
    )

    assert problems_of(config) == [
        "environments.0.program_options: the entry has no harness, so no episode program runs its rollouts to take "
        "them; leave program_options out",
        "environments.0.eval_overrides.program_options: the entry has no harness, so no episode program runs its "
        "rollouts to take them; leave program_options out",
    ]


def test_command_line_overrides_set_options_and_are_checked_like_the_configs(tmp_path):
    path = train_config(tmp_path, entry(), evaluates=False)

    config = load(path, "environments.0.program_options.n=2", "environments.0.program_options.keep=first_success")

    assert config.environments[0].program_options == {"n": 2, "keep": "first_success"}
    assert program_record(config)["options"] == {"default": {"train": {"n": 2, "keep": "first_success"}}}
    assert problems_of(load(path, "environments.0.program_options.n=many")) == [f"{WHERE}.n: 'many' is not a int"]


# --- the check ------------------------------------------------------------------------------------------------------


def test_a_check_that_refuses_names_the_entry_the_program_and_the_phases(tmp_path):
    budget = {"workflow_config": {"rollout_config": {"max_steps": 1}}}
    eval_budget = {"eval_overrides": {"workflow_config": {"rollout_config": {"max_steps": 1}}}}
    config = load(
        train_config(
            tmp_path,
            entry(name="tight", weight=1, program_options={"n": 2}, **budget),
            entry(name="eval-tight", weight=1, program_options={"n": 2}, **eval_budget),
            entry(name="roomy", weight=1, program_options={"n": 1}, **budget),
        )
    )

    assert problems_of(config) == [
        "environments.0.program: optstest-attempts@1, for the entry's train and eval rollouts: n: 2 attempts need at "
        "least 2 steps, and max_steps is 1",
        "environments.1.program: optstest-attempts@1, for the entry's eval rollouts: n: 2 attempts need at least 2 "
        "steps, and max_steps is 1",
    ]


def test_the_check_sees_the_entry_the_phase_and_its_rollout_config_as_copies(tmp_path):
    config = load(
        train_config(
            tmp_path,
            entry(
                name="search",
                program_options={"n": 2},
                eval_overrides={"program_options": {"keep": "first_success"}},
                workflow_config={"rollout_config": {"max_steps": 12}},
            ),
            train={
                "model_name": "Qwen/Qwen3-4B",
                "renderer_name": "qwen3",
                "batch_size": 16,
                "workflow_config": {"rollout_config": {"max_steps": 5, "timeout": 60}},
            },
        )
    )

    run_components(config)

    assert [(options, ctx.phase) for options, ctx in SEEN] == [
        (Attempts(n=2), "train"),
        (Attempts(n=2, keep="first_success"), "eval"),
    ]
    train, evaluation = (ctx for _, ctx in SEEN)
    assert isinstance(train, pt.CheckContext) and train.environment.label == "search"
    assert BUDGETS == {"train": 12, "eval": 12}  # the entry's own rollout settings over the run's
    assert train.rollout.timeout == 60  # the run's, which the entry leaves alone
    # What the check changed was its own copy: the config, and the next phase's context, keep theirs.
    assert config.environments[0].program_options == {"n": 2}
    assert config.train.workflow_config.rollout_config.max_steps == 5
    with pytest.raises(AttributeError):
        train.phase = "eval"  # type: ignore[misc]


def test_the_check_runs_for_the_phases_the_command_runs(tmp_path):
    entries = (entry(program_options={"n": 2}, eval_overrides={"program_options": {"n": 3}}),)

    run_components(load(train_config(tmp_path, *entries, evaluates=False)))
    assert [(options.n, ctx.phase) for options, ctx in SEEN] == [(2, "train")]  # no evaluation: training's only
    SEEN.clear()

    run_components(load(train_config(tmp_path, *entries, evaluates=False)), split="test")  # platoon rollout --split
    assert [(options.n, ctx.phase) for options, ctx in SEEN] == [(3, "eval")]
    SEEN.clear()
    BUDGETS.clear()

    evaluation = load(
        eval_config(tmp_path, *entries, workflow={"rollout_config": {"max_steps": 7}}),
        kind="eval",
    )
    run_components(evaluation)
    assert [(options.n, ctx.phase) for options, ctx in SEEN] == [(3, "eval")]
    assert BUDGETS == {"eval": 7}  # platoon eval's rollout config


def test_a_check_that_fails_another_way_is_reported_with_its_type(tmp_path):
    config = load(train_config(tmp_path, entry(program="optstest-broken-check"), evaluates=False))

    assert problems_of(config) == [
        "environments.0.program: optstest-broken-check@1, for the entry's train rollouts: its check raised KeyError: "
        "'rollout_config'"
    ]


def test_a_file_of_environments_alone_has_its_options_checked_and_its_checks_wait_for_the_runs_that_extend_it(tmp_path):
    """A file of environments alone gives no run's rollout settings, which a check reads: it has the options of both
    phases checked, and runs no check, which a training config that extends it runs with its own settings."""

    path = tmp_path / "environments.yaml"
    budget = {"workflow_config": {"rollout_config": {"max_steps": 1}}}
    path.write_text(yaml.safe_dump({"environments": [entry(eval_overrides={"program_options": {"n": 0}}, **budget)]}))
    assert problems_of(load(path, kind="auto")) == [
        "environments.0.eval_overrides.program_options.n: must be at least 1, got 0"
    ]

    path.write_text(yaml.safe_dump({"environments": [entry(program_options={"n": 2}, **budget)]}))
    assert program_record(load(path, kind="auto"))["options"] == {"default": {"train": {"n": 2, "keep": "best"}}}
    assert SEEN == []  # with the defaults' step budget (none) and the entry's 1 step, it would have refused

    sections = yaml.safe_load(train_config(tmp_path, evaluates=False).read_text())
    del sections["environments"]  # the file's entries
    run = tmp_path / "run.yaml"
    run.write_text(yaml.safe_dump({"extends": path.name, **sections}))
    assert problems_of(load(run)) == [
        "environments.0.program: optstest-attempts@1, for the entry's train rollouts: n: 2 attempts need at least 2 "
        "steps, and max_steps is 1"
    ]


def test_prepare_runs_no_check_and_a_run_with_prepare_runs_them_with_its_settings_before_it_prepares(
    tmp_path, monkeypatch, capsys
):
    """platoon prepare loads a config's environments alone, without the run's rollout settings that a check reads: it
    checks their options and runs no check, which would refuse what the run's settings allow.  train --prepare runs the
    checks with its own settings, and before the prepare makes anything."""

    monkeypatch.setenv("TINKER_API_KEY", "tml-test")  # train checks its key before anything else: past it
    started: list[Any] = []
    monkeypatch.setattr("platoon.train.supervise.run_training", lambda *args, **kwargs: started.append(args) or 0)
    model = tmp_path / "model"  # a local model: the prepare's tokenizer step is current, with no download
    model.mkdir()
    train = {
        "model_name": str(model),
        "renderer_name": "qwen3",
        "batch_size": 16,
        "workflow_config": {"rollout_config": {"timeout": 7200}},  # what the stages need: the check passes
    }
    path = train_config(tmp_path, entry(program="optstest-staged"), evaluates=False, train=train)

    cli.main(["prepare", "--dry-run", str(path)])  # exit 0: it returns
    assert capsys.readouterr().out.splitlines()[-1] == "ready: 1 already current"
    assert SEEN == []  # with the defaults' timeout (none), the check would have refused it
    cli.main(["train", str(path), "--prepare"])
    assert len(started) == 1
    assert [(ctx.phase, ctx.rollout.timeout) for _, ctx in SEEN] == [("train", 7200)]  # the run's settings, once

    def never(*args: Any, **kwargs: Any) -> int:
        raise AssertionError("prepared before the program's check refused the run")

    monkeypatch.setattr(cli, "_prepare_first", never)
    short = "train.workflow_config.rollout_config.timeout=60"
    assert exit_code(["train", str(path), "--prepare", short]) == cli.EXIT_CONFIG
    assert (
        "environments.0.program: optstest-staged@1, for the entry's train rollouts: rollout_config.timeout 60 is below "
        "the stages' 5400 s"
    ) in capsys.readouterr().err
    assert len(started) == 1


# --- recorded -------------------------------------------------------------------------------------------------------


def test_lock_json_records_the_options_with_their_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr("platoon.train.launch.run_tinker_training", lambda args: None)
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")  # train checks its key before anything else: past it
    path = train_config(
        tmp_path,
        entry(name="a", weight=1, program_options={"n": 2}),
        entry(
            name="b", weight=1, program_options={"n": 2}, eval_overrides={"program_options": {"keep": "first_success"}}
        ),
        entry(name="c", weight=1, program="optstest-plain"),
        entry(name="d", weight=1, program="optstest-parsed", program_options={}),
        entry(name="e", weight=1, program="optstest-placed", eval_overrides={"program_options": {"mode": "slow"}}),
    )

    cli.main(["train", str(path)])

    lock = json.loads((tmp_path / "logs/exp/trial/lock.json").read_text())
    records = {record["ref"]: record for record in lock["components"] if record["kind"] == "program"}
    assert records["optstest-attempts"]["options"] == {
        "a": {"train": {"n": 2, "keep": "best"}},  # evaluation runs the same options: training's only
        "b": {"train": {"n": 2, "keep": "best"}, "eval": {"n": 2, "keep": "first_success"}},
    }
    assert "options" not in records["optstest-plain"]  # a program without options records none
    assert records["optstest-parsed"]["options"] == {"d": {"train": {"most": 3}}}  # what the parser filled in
    assert records["optstest-placed"]["options"] == {  # an enum's value and a path's text, as a config writes them
        "e": {
            "train": {"workdir": "work", "mode": "fast", "scratch": None},
            "eval": {"workdir": "work", "mode": "slow", "scratch": None},
        }
    }


def test_options_parsed_into_values_json_cannot_hold_are_recorded_as_written(tmp_path, caplog):
    config = load(
        train_config(tmp_path, entry(program="optstest-compiled", program_options={"pattern": "a+"}), evaluates=False)
    )

    assert program_record(config)["options"] == {"default": {"train": {"pattern": "a+"}}}
    assert "program optstest-compiled@1's options parse into values that lock.json cannot hold" in caplog.text


def test_enum_and_path_options_are_built_as_their_classes():
    record = resolve_program("optstest-placed")

    options = record.parse_options({"workdir": "/srv/run", "mode": "slow", "scratch": "/tmp/s"})

    assert options == Placed(workdir=Path("/srv/run"), mode=Mode.SLOW, scratch=Path("/tmp/s"))
    assert options.mode is Mode.SLOW and isinstance(options.workdir, Path)
    assert record.parse_options({"mode": Mode.SLOW}).mode is Mode.SLOW  # a member given in Python stays itself
    assert record.parse_options({}) == Placed()
    with pytest.raises(ConfigError) as error:
        record.parse_options({"mode": "medium", "workdir": 5})
    assert error.value.problems == [
        "program_options.mode: 'medium' is not a Mode",
        "program_options.workdir: 5 is not a Path",
    ]


# --- routes ---------------------------------------------------------------------------------------------------------


def test_routes_carry_each_phases_options_as_the_entry_writes_them(tmp_path):
    config = load(
        train_config(
            tmp_path, entry(program_options={"n": 2}, eval_overrides={"program_options": {"keep": "first_success"}})
        )
    )

    assert route_of(config, "train").program_options == {"n": 2}
    for split in ("eval", "test", "val"):  # every split but train is an evaluation
        assert route_of(config, split).program_options == {"n": 2, "keep": "first_success"}
    task = AutoTaskLoader.from_config(config, "train")("train:0")
    trajectory = {"task": {"misc": pickle.loads(pickle.dumps(task.misc))}}
    assert task.misc[ROUTE_MISC_KEY]["program_options"] == {"n": 2}
    routed = EnvironmentRoute.of_trajectory(trajectory)
    assert routed is not None and routed.program_options == {"n": 2}


def test_a_route_whose_entry_sets_no_program_options_travels_as_it_did_before():
    route = environment_route(EnvironmentConfig(service=NUMBER_SEARCH, harness="platoon-react"))
    payload = route.attach(Task(id="val:0")).misc[ROUTE_MISC_KEY]

    # The same keys, values and order as before: a pickled request, and a golden's JSON, keep their bytes.
    assert list(payload) == ROUTE_KEYS
    expected = asdict(route)
    del expected["program_options"]
    del expected["agents"]["root_role"]
    assert pickle.dumps(payload) == pickle.dumps(expected)
    unrouted = EnvironmentRoute.of_task(EnvironmentRoute(name="e").attach(Task(id="t")))
    assert unrouted is not None and unrouted.program_options == {}


def test_the_trainers_route_check_refuses_bad_options_for_library_callers():
    environments = normalize_environment_configs([entry(program_options={"keep": "worst"})])

    with pytest.raises(ConfigError) as error:
        check_environment_routes(environments)
    assert error.value.problems == [f"{WHERE}.keep: 'worst' is not one of 'best', 'first_success'"]
    # A route resolves its program only for the options its entry sets: one that sets none is built as before, whatever
    # its program, and a rollout of a program that does not resolve fails where it runs it.
    nowhere = normalize_environment_configs([entry(program="tests.nowhere:program")])
    check_environment_routes(nowhere)
    with pytest.raises(ConfigError) as error:
        environment_route(
            normalize_environment_configs([entry(program="tests.nowhere:program", program_options={"n": 1})])[0]
        )
    assert error.value.problems == ["environments[default].program: No module named 'tests.nowhere'"]


# --- named environments ---------------------------------------------------------------------------------------------


def test_a_named_environments_options_apply_while_the_entry_keeps_its_program(tmp_path, monkeypatch):
    (tmp_path / "optstest_tasks.py").write_text(
        textwrap.dedent(
            """
            import platoon as pt

            TASKS = pt.EnvironmentEntry(
                "optstest-tasks",
                service="python://platoon.envs.number_search.env:NumberSearchEnvironment",
                harness="platoon-react",
                program="optstest-attempts",
                program_options={"n": 2, "keep": "first_success"},
                eval_overrides={"program_options": {"n": 1}},
            )
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    kept = load(train_config(tmp_path, {"env": "optstest_tasks:TASKS", "program_options": {"n": 3}}, evaluates=False))
    assert kept.environments[0].program_options == {"n": 3, "keep": "first_success"}  # the entry's over the named's
    assert program_record(kept)["options"] == {"default": {"train": {"n": 3, "keep": "first_success"}}}
    evaluated = load(
        train_config(tmp_path, {"env": "optstest_tasks:TASKS", "eval_overrides": {"program_options": {"keep": "best"}}})
    )
    # The named environment's evaluation overrides lie beneath the entry's own, and over its options, as any do.
    assert evaluated.environments[0].eval_overrides.program_options == {"n": 1, "keep": "best"}
    assert program_record(evaluated)["options"] == {
        "default": {"train": {"n": 2, "keep": "first_success"}, "eval": {"n": 1, "keep": "best"}}
    }
    pinned = load(train_config(tmp_path, {"env": "optstest_tasks:TASKS", "program": "optstest-attempts@1"}))
    assert pinned.environments[0].program_options == {"n": 2, "keep": "first_success"}  # the same program, pinned
    assert pinned.environments[0].eval_overrides.program_options == {"n": 1}
    other = load(train_config(tmp_path, {"env": "optstest_tasks:TASKS", "program": "optstest-plain"}))
    # Another program: none of the named environment's options, for either phase.
    assert other.environments[0].program_options == {} and other.environments[0].eval_overrides.program_options == {}
    assert "eval_overrides" not in expand_environment({"env": "optstest_tasks:TASKS", "program": "optstest-plain"})
    run_components(other)


def test_a_named_environment_sets_program_options_with_its_program():
    for keys in ({"program_options": {"n": 2}}, {"eval_overrides": {"program_options": {"n": 2}}}):
        written = "program_options" if "program_options" in keys else "eval_overrides.program_options"
        with pytest.raises(ValueError, match=f"sets {written} without program: options go with the program"):
            pt.EnvironmentEntry("optstest-tasks", service="harbor", **keys)
    named = pt.EnvironmentEntry(
        "optstest-tasks", service="harbor", program="optstest-attempts", eval_overrides={"program_options": {"n": 2}}
    )
    assert named.entry()["eval_overrides"] == {"program_options": {"n": 2}}
