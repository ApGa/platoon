"""The keys of credit (RFC 0021, section 6): each environment entry's ``credit`` and ``credit_options``.  What loads and
what is refused, with its message, when the config loads or where the estimators resolve (``run_components``, as
``platoon train`` and ``platoon config validate`` run it), ``running``'s priors file relative to the config file
included, also as a role's estimator under ``roles``; ``lock.json``'s records, only for entries that set one, and for
each role's estimator under ``roles``, a package's included; and the workflows ``platoon train`` builds: the training
workflow credits each such entry's groups with its estimator, the evaluation's never.  An entry of groups of one under
``roles`` is not warned of, and a share that its role's policy cannot hold is refused."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from fakes.launch import built_workflows

from platoon import registry as registry_module
from platoon.envs.named import FORBIDDEN_KEYS
from platoon.extensions import origin_of_module, run_components
from platoon.task import Task
from platoon.train import entry_settings, launch
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig, WorkflowConfig
from platoon.train.credit.api import PolicyView
from platoon.train.credit.keyed import Keyed, KeyedOptions
from platoon.train.credit.registry import REGISTRY_KIND, CreditChoice, build_entry_credit
from platoon.train.credit.roles import Roles
from platoon.train.credit.running import Running, RunningOptions, RunningPriors, StratumPrior
from platoon.train.policy.spec import MAIN, policy_warnings
from platoon.utils.config import ConfigError, resolve_config

HERE = __name__


def tasks(config: Any, split: str, *, prefix: str = "t", count: int = 6) -> list[dict[str, Any]]:
    return [{"task_id": f"{prefix}{index}"} for index in range(count)]


async def rollout(task: Any, rollout_config: Any) -> dict:
    raise AssertionError("not run")


def get_task(task_id: str) -> Task:
    return Task(id=task_id)


class Refusing(Keyed):
    """``keyed``, whose load-time check refuses a depth key."""

    @staticmethod
    def check(options: KeyedOptions, ctx: Any) -> None:
        if "depth" in options.key:
            raise ValueError(f"environments[{ctx.entry}] trains no subagents to key on depth")


def _entry(name: str, **fields: Any) -> dict[str, Any]:
    return {
        "name": name,
        "dataset_loader": f"{HERE}:tasks",
        "dataset_kwargs": {"prefix": f"{name}-"},
        "rollout": f"{HERE}:rollout",
        "task_loader": f"{HERE}:get_task",
        **fields,
    }


def _load(tmp_path: Path, *environments: dict[str, Any], sections: dict[str, Any] | None = None, **train: Any) -> Any:
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "train": {"model_name": "m", "renderer_name": "r", "batch_size": 4, "max_training_steps": 3, **train},
                "eval": {"every": 0},
                "watchdog": {"enabled": False},
                "log_path": str(tmp_path / "logs"),
                "environments": list(environments),
                **(sections or {}),
            }
        )
    )
    return resolve_config(path, [], PlatoonTinkerRLTrainerConfig).config


def _refused(tmp_path: Path, *environments: dict[str, Any]) -> list[str]:
    with pytest.raises(ConfigError) as error:
        _load(tmp_path, *environments)
    return error.value.problems


def _resolution(tmp_path: Path, *environments: dict[str, Any], **train: Any) -> list[str]:
    config = _load(tmp_path, *environments, **train)
    with pytest.raises(ConfigError) as error:
        run_components(config)
    return error.value.problems


def test_options_without_an_estimator_and_values_of_the_wrong_kind_are_refused_as_the_config_loads(tmp_path):
    [problem] = _refused(tmp_path, _entry("a", credit_options={"key": ["role"]}))
    assert "credit_options go with the estimator they configure: set credit beside them" in problem
    assert any("credit must name a credit estimator" in line for line in _refused(tmp_path, _entry("a", credit="")))
    assert any("credit_options" in line for line in _refused(tmp_path, _entry("a", credit="keyed", credit_options=[1])))


def test_a_named_environment_cannot_choose_the_credit():
    assert {"credit", "credit_options"} <= FORBIDDEN_KEYS


@pytest.mark.parametrize(
    ("fields", "says"),
    [
        ({"credit": "keyd"}, "environments.0.credit: unknown credit estimator 'keyd'; did you mean keyed?"),
        ({"credit": "gae"}, "environments.0.credit: credit estimator 'gae' is reserved"),
        ({"credit": "reverse_kl"}, "environments.0.credit_options.teacher: required key is missing"),
        ({"credit": "keyed", "credit_options": {"keys": ["role"]}}, "environments.0.credit_options"),
        ({"credit": "keyed", "credit_options": {"key": ["member"]}}, "environments.0.credit_options"),
        ({"credit": "group", "credit_options": {"leave_one_out": True}}, "takes no options"),
        ({"credit": f"{HERE}:Refusing", "credit_options": {"key": ["depth"]}}, "trains no subagents to key on depth"),
        ({"credit": f"{HERE}:tasks"}, "is not a credit estimator: a class with async assign(group)"),
        (
            {"credit": "running", "credit_options": {"strata": ["task_id"]}},
            "environments.0.credit_options.strata: task_id is platoon's own column, not one of the task's",
        ),
        (
            {"credit": "running", "credit_options": {"priors": {"mean": 0, "scale": -1}}},
            "environments.0.credit_options.priors: scale: a finite number above 0, got -1",
        ),
    ],
)
def test_what_does_not_resolve_or_take_its_options_stops_the_run_with_its_path(tmp_path, fields, says):
    problems = _resolution(tmp_path, _entry("a", **fields))
    assert any(says in problem for problem in problems), problems


def test_running_s_priors_file_is_relative_to_the_config_file_and_lock_json_records_what_it_holds(
    tmp_path, monkeypatch
):
    (tmp_path / "priors").mkdir()
    (tmp_path / "priors" / "base.yaml").write_text(
        "mean: -0.25\nscale: 0.5\nstrata: [{stratum: [masked], mean: 0.1}]\n"
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # the working directory has no priors/base.yaml
    options = {"strata": ["kind"], "priors": "priors/base.yaml"}
    config = _load(tmp_path, _entry("solo", credit="running", credit_options=options), max_staleness=2)

    assert config.environments[0].credit_options["priors"] == str(tmp_path / "priors" / "base.yaml")
    [record] = [record for record in run_components(config) if record["kind"] == "credit"]
    assert record["resolved"] == "platoon.train.credit.running:Running"
    assert record["options"]["solo"]["priors"] == {
        "mean": -0.25,
        "scale": 0.5,
        "weight": 16.0,
        "strata": [{"stratum": ["masked"], "mean": 0.1}],
    }
    train, _ = built_workflows(config)
    estimator = train.entry_credit["solo"]
    assert isinstance(estimator, Running) and estimator.options.priors == RunningPriors(
        mean=-0.25, scale=0.5, strata=[StratumPrior(["masked"], 0.1)]
    )
    assert estimator.setup.max_staleness == {MAIN: 2}  # how old its rows may be: how far back its statistics reach


def test_running_needs_the_entry_s_rows_to_have_a_staleness_bound(tmp_path):
    [problem] = _resolution(tmp_path, _entry("solo", credit="running"))
    assert problem.startswith("environments.0.credit: running: it reads each agent's statistics as they were when")
    assert problem.endswith("set train.max_staleness (how many policy versions old a rollout may be and still train)")
    config = _load(tmp_path, _entry("solo", credit="running"), max_staleness=0)
    assert [record["ref"] for record in run_components(config) if record["kind"] == "credit"] == ["running"]


_BY_ROLE = {
    "generator": {"credit": "running", "options": {"strata": ["kind"], "priors": "priors/base.yaml"}, "share": 0.1},
    "solver": {"credit": "keyed", "options": {"key": ["role"], "leave_one_out": True}},
}


def test_roles_s_options_name_a_role_s_priors_file_relative_to_the_config_file_and_lock_json_records_each_role(
    tmp_path, monkeypatch
):
    (tmp_path / "priors").mkdir()
    (tmp_path / "priors" / "base.yaml").write_text("mean: -0.25\nscale: 0.5\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # the working directory has no priors/base.yaml
    entry = _entry("envgen", credit="roles", credit_options={"by_role": _BY_ROLE}, workflow_config={"group_size": 1})
    config = _load(tmp_path, entry, max_staleness=2)

    priors = config.environments[0].credit_options["by_role"]["generator"]["options"]["priors"]
    assert priors == str(tmp_path / "priors" / "base.yaml")
    record, *roles = [record for record in run_components(config) if record["kind"] == "credit"]
    assert record["resolved"] == "platoon.train.credit.roles:Roles" and "level" not in record
    by_role = record["options"]["envgen"]["by_role"]
    assert by_role["generator"]["share"] == 0.1 and by_role["generator"]["kl"] is None
    assert by_role["generator"]["options"]["priors"] == {"mean": -0.25, "scale": 0.5, "weight": 16.0, "strata": []}
    assert by_role["generator"]["options"]["half_life"] == 4.0  # each role's options, defaults included
    assert by_role["solver"]["options"] == {"key": ["role"], "leave_one_out": True}
    # Each role's estimator has a record of its own: what it resolved to, and what provides it.
    assert [(role["ref"], role["resolved"], role["level"], role["distribution"]) for role in roles] == [
        ("running", "platoon.train.credit.running:Running", "role", "platoon"),
        ("keyed", "platoon.train.credit.keyed:Keyed", "role", "platoon"),
    ]
    assert roles[1]["environments"] == ["envgen"]
    assert roles[1]["options"] == {"envgen": {"solver": {"key": ["role"], "leave_one_out": True}}}

    train, _ = built_workflows(config)
    estimator = train.entry_credit["envgen"]
    assert isinstance(estimator, Roles)
    generator, solver = estimator._estimators["generator"], estimator._estimators["solver"]
    assert isinstance(generator, Running) and isinstance(solver, Keyed)
    assert generator.setup.max_staleness == {MAIN: 2} and generator.options.priors == RunningPriors(-0.25, 0.5)
    assert (generator.setup.name, generator.setup.entry, solver.setup.name) == ("running", "envgen", "keyed")
    # Its groups of one train: no warning of them, as there is of the same entry under keyed.
    assert not [
        line for line in entry_settings.warnings(config, entry_settings.entry_settings(config)) if "is 1" in line
    ]
    keyed = _load(tmp_path / "priors", {**entry, "credit": "keyed", "credit_options": {}}, max_staleness=2)
    assert [line for line in entry_settings.warnings(keyed, entry_settings.entry_settings(keyed)) if "is 1" in line]


_FROZEN = {"model_name": "m", "renderer_name": "r", "trainable": False}


@pytest.mark.parametrize(
    ("by_role", "sections", "says"),
    [
        (
            {"generator": {"credit": "running"}},
            None,
            "environments.0.credit: roles: by_role.generator: running: it reads each agent's statistics",
        ),
        (
            {"generator": {"credit": "keyed", "kl": {"to": "reference", "coef": 0.1}}},
            None,
            "by_role.generator.kl.to: 'reference' is not a policy of policies: (declared: none)",
        ),
        (
            {"generator": {"credit": "keyed", "kl": {"to": "main", "coef": 0.1}}},
            None,
            "by_role.generator.kl.to: main is the student, the policy the run trains; name a frozen policy",
        ),
        (
            {"generator": {"credit": "keyed", "kl": {"to": "reference", "coef": 0.1}}},
            {
                "policies": {"reference": {"model_name": "m", "renderer_name": "r"}},
                "agents": {"roles": {"helper": {"policy": "reference"}}},
            },
            "by_role.generator.kl.to: policies.reference trains; the policy a kl term scores with is frozen",
        ),
        (
            {"solver": {"credit": "keyed", "options": {"key": ["member"]}}},
            None,
            "environments.0.credit_options.by_role.solver.options.key: one or more of task, policy, role, depth",
        ),
    ],
)
def test_roles_runs_each_role_s_check_and_checks_a_kl_term_s_policy_as_the_config_loads(
    tmp_path, by_role, sections, says
):
    config = _load(tmp_path, _entry("envgen", credit="roles", credit_options={"by_role": by_role}), sections=sections)
    with pytest.raises(ConfigError) as error:
        run_components(config)
    assert any(says in problem for problem in error.value.problems), error.value.problems


def test_a_kl_term_s_frozen_policy_loads_and_is_no_policy_the_run_opens_and_never_uses(tmp_path):
    by_role = {"generator": {"credit": "keyed", "kl": {"to": "reference", "coef": 0.1}}}
    entry = _entry("envgen", credit="roles", credit_options={"by_role": by_role})
    config = _load(tmp_path, entry, sections={"policies": {"reference": _FROZEN, "unused": _FROZEN}})
    record, _ = [record for record in run_components(config) if record["kind"] == "credit"]
    assert record["options"]["envgen"]["by_role"]["generator"]["kl"] == {"to": "reference", "coef": 0.1}
    [line] = policy_warnings(config)  # the policy a role's KL term names inside credit_options is used
    assert line.startswith("policies.unused: no agent samples it")


_KEYED = {"key": ["task", "policy", "role"], "leave_one_out": False}  # keyed's options, all defaults
_ACME_ROLE = """
import platoon.experimental as px


@px.credit_estimator("acme-solver")
class Solver(px.CreditEstimatorBase):
    needs_agents = True

    async def assign(self, group):
        pass
"""


def test_lock_json_records_a_role_s_estimator_that_a_package_registers_with_what_provides_it(tmp_path, monkeypatch):
    monkeypatch.setattr(registry_module, "_REGISTRIES", {**registry_module._REGISTRIES})
    registry_module._REGISTRIES.pop(REGISTRY_KIND, None)
    package = f"acme_role_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{package}.py").write_text(_ACME_ROLE)
    monkeypatch.syspath_prepend(str(tmp_path))
    by_role = {"policy": {"credit": "keyed"}, "solver": {"credit": "acme-solver"}, "judge": {"credit": "acme-solver"}}
    entries = [
        _entry(name, package=package, credit="roles", credit_options={"by_role": by_role}) for name in ("a", "b")
    ]
    try:
        records = [record for record in run_components(_load(tmp_path, *entries)) if record["kind"] == "credit"]
    finally:
        sys.modules.pop(package, None)

    roles, acme, keyed = records  # by_role's roles in the order the config file holds them: judge, policy, solver
    assert (roles["ref"], roles["environments"]) == ("roles", ["a", "b"])
    origin = origin_of_module(package)
    assert acme == {
        "kind": "credit",
        "ref": "acme-solver",
        "resolved": f"{package}:Solver",
        "source": f"{package}:Solver",
        "distribution": origin.distribution,
        "distribution_version": origin.version,
        "entry_point": None,
        "level": "role",
        "environments": ["a", "b"],
        "options": {"a": {"judge": {}, "solver": {}}, "b": {"judge": {}, "solver": {}}},
    }
    assert (keyed["ref"], keyed["level"], keyed["options"]) == (
        "keyed",
        "role",
        {"a": {"policy": _KEYED}, "b": {"policy": _KEYED}},
    )


_TRAINED = {"model_name": "m", "renderer_name": "r"}


@pytest.mark.parametrize(
    ("shared", "agents", "says"),
    [
        (  # two adapters: the generator's rows are a lane of their own, with no role without a share beside them
            "generator",
            {"roles": {"generator": {"policy": "generator"}}},
            "environments.0.credit: roles: by_role.generator.share: the agents of role generator sample the policy "
            "generator, and no role by_role names without a share samples it",
        ),
        (  # the root's role samples the agents' default, whatever agents.roles says
            "policy",
            {"policy": "generator", "roles": {"solver": {"policy": "main"}}},
            "by_role.policy.share: the agents of role policy sample the policy generator",
        ),
    ],
)
def test_a_share_whose_role_s_policy_has_no_role_without_a_share_is_refused_as_the_config_loads(
    tmp_path, shared, agents, says
):
    by_role = {shared: {"credit": "keyed", "share": 0.1}, "solver": {"credit": "keyed"}}
    entry = _entry("envgen", credit="roles", credit_options={"by_role": by_role})
    config = _load(tmp_path, entry, sections={"policies": {"generator": _TRAINED}, "agents": agents})
    with pytest.raises(ConfigError) as error:
        run_components(config)
    assert any(says in problem for problem in error.value.problems), error.value.problems

    # One policy for both roles: the solvers' mass holds the share.
    records = [record for record in run_components(_load(tmp_path, entry)) if record["kind"] == "credit"]
    assert records[0]["ref"] == "roles"


def test_each_estimator_is_built_with_the_staleness_bound_of_each_policy_s_rows_of_its_entry():
    # Main's rows take the entry's bound; another policy's its own, when it has one (policies.<id>.max_staleness).
    choices = {label: CreditChoice(label, "running", Running, RunningOptions()) for label in ("quick", "slow")}
    settings = {
        "quick": SimpleNamespace(train=WorkflowConfig(group_size=1), max_staleness=2),
        "slow": SimpleNamespace(train=WorkflowConfig(group_size=1), max_staleness=6),
    }
    views = {policy: PolicyView(policy) for policy in (MAIN, "worker", "helper")}
    built = build_entry_credit(choices, settings, views, {"worker": 4})
    assert {label: estimator.setup.max_staleness for label, estimator in built.items()} == {
        "quick": {MAIN: 2, "worker": 4, "helper": 2},
        "slow": {MAIN: 6, "worker": 4, "helper": 6},
    }


_OF_ONE = """
import platoon.experimental as px


@px.credit_estimator("acme-of-one")
class OfOne(px.CreditEstimatorBase):
    credits_single_members = True

    async def assign(self, group):
        pass
"""


def test_a_start_reads_the_flag_of_an_estimator_its_entry_s_package_registers_before_it_warns(
    tmp_path, monkeypatch, caplog
):
    # A supervised start trains in a fresh process, where nothing imported the entries' packages before platoon
    # train's checks: the group-of-one warning reads the flag of the estimator that the entry's package registers.
    monkeypatch.setattr(registry_module, "_REGISTRIES", {**registry_module._REGISTRIES})
    registry_module._REGISTRIES.pop(REGISTRY_KIND, None)
    package = f"acme_of_one_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{package}.py").write_text(_OF_ONE)
    monkeypatch.syspath_prepend(str(tmp_path))
    entry = _entry("solo", package=package, credit="acme-of-one", workflow_config={"group_size": 1})
    config = _load(tmp_path, entry)
    assert package not in sys.modules
    try:
        with caplog.at_level(logging.WARNING, logger="platoon.train.entry_settings"):
            launch._check_environments(config)
    finally:
        sys.modules.pop(package, None)
    assert not [record for record in caplog.records if "group_size is 1" in record.getMessage()]


def test_a_priors_file_that_is_not_there_stops_the_run_as_the_config_loads(tmp_path):
    [problem] = _refused(tmp_path, _entry("solo", credit="running", credit_options={"priors": "missing.yaml"}))
    assert problem.startswith(f"environments.0.credit_options.priors: missing.yaml names {tmp_path / 'missing.yaml'}")


def test_lock_json_records_each_estimator_with_its_entries_and_options_and_none_without_one(tmp_path):
    config = _load(
        tmp_path,
        _entry("hard", credit="keyed", credit_options={"key": ["role"], "leave_one_out": True}),
        _entry("easy"),
        _entry("also", credit="keyed", credit_options={"key": ["role"], "leave_one_out": True}),
        _entry("mean", credit="group"),
    )
    records = [record for record in run_components(config) if record["kind"] == "credit"]
    assert [(record["ref"], record["environments"], record["options"]) for record in records] == [
        (
            "keyed",
            ["hard", "also"],
            {"hard": {"key": ["role"], "leave_one_out": True}, "also": {"key": ["role"], "leave_one_out": True}},
        ),
        ("group", ["mean"], {"mean": {}}),
    ]
    assert records[0]["resolved"] == "platoon.train.credit.keyed:Keyed" and records[0]["distribution"] == "platoon"
    plain = _load(tmp_path, _entry("a"))
    assert not [record for record in run_components(plain) if record["kind"] == "credit"]


class _OwnWorkflow:
    """A workflow that credits its own way."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    async def arun_episode(self, data: dict) -> list:
        return []


def test_an_estimator_beside_a_workflow_that_does_not_apply_it_stops_the_run(tmp_path):
    problems = _resolution(tmp_path, _entry("a", credit="keyed", workflow=f"{HERE}:_OwnWorkflow"))
    assert any(problem.startswith("environments[a].credit: the run's workflow") for problem in problems), problems


def test_the_training_workflow_credits_with_each_entry_s_estimator_and_the_evaluation_s_with_none(tmp_path):
    config = _load(tmp_path, _entry("hard", credit="keyed", credit_options={"key": ["role"]}), _entry("easy"))
    train, evaluation = built_workflows(config)

    assert list(train.entry_credit) == ["hard"]
    estimator = train.entry_credit["hard"]
    assert isinstance(estimator, Keyed) and estimator.options == KeyedOptions(key=["role"])
    assert (estimator.setup.name, estimator.setup.entry, estimator.setup.policies) == ("keyed", "hard", ("main",))
    # Its groups run with their entry's settings, which name their entry.
    assert set(train.entry_configs) == {"hard", "easy"}
    assert dict(evaluation.entry_credit) == {}


def test_without_credit_the_workflows_are_built_as_before(tmp_path):
    train, evaluation = built_workflows(_load(tmp_path, _entry("a")))
    assert dict(train.entry_credit) == {} == dict(evaluation.entry_credit)
    assert dict(train.entry_configs) == {}
