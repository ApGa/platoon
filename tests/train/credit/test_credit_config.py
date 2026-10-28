"""The keys of credit (RFC 0021, section 6): each environment entry's ``credit`` and ``credit_options``.  What loads and
what is refused, with its message, when the config loads or where the estimators resolve (``run_components``, as
``platoon train`` and ``platoon config validate`` run it); ``lock.json``'s records, only for entries that set one; and
the workflows ``platoon train`` builds: the training workflow credits each such entry's groups with its estimator, the
evaluation's never."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from platoon.envs.named import FORBIDDEN_KEYS
from platoon.extensions import run_components
from platoon.task import Task
from platoon.train import entry_settings, launch
from platoon.train.auto import AutoEnvironment
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.credit.keyed import Keyed, KeyedOptions
from platoon.train.policy.policy import Policy
from platoon.train.policy.set import PolicySet
from platoon.train.policy.spec import MAIN, PolicySpec
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


def _load(tmp_path: Path, *environments: dict[str, Any]) -> Any:
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "train": {"model_name": "m", "renderer_name": "r", "batch_size": 4, "max_training_steps": 3},
                "eval": {"every": 0},
                "watchdog": {"enabled": False},
                "log_path": str(tmp_path / "logs"),
                "environments": list(environments),
            }
        )
    )
    return resolve_config(path, [], PlatoonTinkerRLTrainerConfig).config


def _refused(tmp_path: Path, *environments: dict[str, Any]) -> list[str]:
    with pytest.raises(ConfigError) as error:
        _load(tmp_path, *environments)
    return error.value.problems


def _resolution(tmp_path: Path, *environments: dict[str, Any]) -> list[str]:
    config = _load(tmp_path, *environments)
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
    ],
)
def test_what_does_not_resolve_or_take_its_options_stops_the_run_with_its_path(tmp_path, fields, says):
    problems = _resolution(tmp_path, _entry("a", **fields))
    assert any(says in problem for problem in problems), problems


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


def _workflows(config: Any) -> tuple[Any, Any]:
    """The training and evaluation workflows ``platoon train`` builds for ``config``."""

    AutoEnvironment.load(config)
    settings = entry_settings.entry_settings(config)
    components = launch._build_components(config, AutoEnvironment.from_config(config))
    model_info: Any = SimpleNamespace(
        model_name="m", renderer_name="r", renderer_kwargs={}, llm=SimpleNamespace(version=0)
    )
    policies = PolicySet([Policy(PolicySpec(MAIN, model_name="m", renderer_name="r"), model_info)])
    training: Any = SimpleNamespace(tasks=[])
    return launch._build_workflows(config, components, policies, settings, training, None)


def test_the_training_workflow_credits_with_each_entry_s_estimator_and_the_evaluation_s_with_none(tmp_path):
    config = _load(tmp_path, _entry("hard", credit="keyed", credit_options={"key": ["role"]}), _entry("easy"))
    train, evaluation = _workflows(config)

    assert list(train.entry_credit) == ["hard"]
    estimator = train.entry_credit["hard"]
    assert isinstance(estimator, Keyed) and estimator.options == KeyedOptions(key=["role"])
    assert (estimator.setup.name, estimator.setup.entry, estimator.setup.policies) == ("keyed", "hard", ("main",))
    # Its groups run with their entry's settings, which name their entry.
    assert set(train.entry_configs) == {"hard", "easy"}
    assert dict(evaluation.entry_credit) == {}


def test_without_credit_the_workflows_are_built_as_before(tmp_path):
    train, evaluation = _workflows(_load(tmp_path, _entry("a")))
    assert dict(train.entry_credit) == {} == dict(evaluation.entry_credit)
    assert dict(train.entry_configs) == {}
