"""Options of TOML harness specs in configs (RFC 0016): checked when a run resolves its config, before anything runs,
with full dotted paths and a did-you-mean (exit 78); layered over a named environment's ``harness_defaults``, under
the command line; changed for evaluation rollouts by ``eval_overrides``; carried to rollouts by per-phase routes; and
recorded in ``lock.json`` with their defaults."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Literal

import pytest
import yaml

from platoon import cli
from platoon.extensions import run_components
from platoon.train.auto import AutoTaskLoader, check_environment_routes
from platoon.train.components import EnvironmentRoute, normalize_environment_configs
from platoon.utils.config import ConfigError

DEFAULTS = {"reasoning_replay": True, "request_timeout_s": 3600, "num_retries": 0, "condenser": "none"}
ADAPTER = {"reasoning_replay": False, "request_timeout_s": 300, "num_retries": 5}  # Harbor's openhands-sdk adapter
WHERE = "environments.0.harness_options"


def openhands(**entry) -> dict:
    return {"service": "harbor", "service_kwargs": {"dataset_dir": "/tasks"}, "harness": "openhands", **entry}


def train_config(tmp_path: Path, *entries: dict, evaluates: bool = True) -> Path:
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "train": {"model_name": "Qwen/Qwen3-4B", "renderer_name": "qwen3", "batch_size": 16},
                "eval": {"every": 1} if evaluates else {"every": 0},
                "log_path": str(tmp_path / "logs"),
                "stats": {"experiment_name": "exp", "trial_name": "trial"},
                "environments": list(entries) or [openhands()],
            },
            sort_keys=False,
        )
    )
    return path


def eval_config(tmp_path: Path, *entries: dict) -> Path:
    path = tmp_path / "eval.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "inference": {"model_name": "Qwen/Qwen3-4B", "output_dir": str(tmp_path / "results")},
                "environments": list(entries),
            },
            sort_keys=False,
        )
    )
    return path


def load(path: Path, *overrides: str, kind: Literal["train", "eval"] = "train"):
    return cli._resolve(path, overrides, kind).config  # noqa: SLF001 - the commands' own loading


def exit_code(argv: list[str]) -> int | str | None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    return exit_info.value.code


def harness_record(config) -> dict:
    (record,) = [record for record in run_components(config) if record["kind"] == "harness"]
    return record


def route_options(config, split: str) -> dict:
    task = AutoTaskLoader.from_config(config, split)("task-1")
    route = EnvironmentRoute.of_task(task)
    assert route is not None
    return route.harness_options


# --- checked before anything runs ----------------------------------------------------------------------------------


def test_a_misspelled_option_fails_validate_and_train_with_exit_78_before_anything_runs(tmp_path, monkeypatch, capsys):
    trained = []
    monkeypatch.setattr("platoon.train.tinker.train.run_tinker_training", trained.append)
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")  # train checks its key before anything else: past it
    config = train_config(tmp_path, openhands(harness_options={"num_retires": 2}))
    problem = f"{WHERE}.num_retires: unknown key; did you mean {WHERE}.num_retries?"

    assert exit_code(["config", "validate", str(config)]) == cli.EXIT_CONFIG
    assert f"FAIL {config}\n  {problem}" in capsys.readouterr().out
    assert exit_code(["train", str(config)]) == cli.EXIT_CONFIG
    assert problem in capsys.readouterr().err
    assert trained == [] and not (tmp_path / "logs").exists()  # nothing ran, and no run directory was made


def test_every_problem_is_reported_at_once_where_it_is_written(tmp_path):
    config = load(
        train_config(
            tmp_path,
            openhands(
                harness_options={"reasoning_replay": "no", "num_retries": 11},
                eval_overrides={"harness_options": {"request_timeout_s": 0, "condensor": "llm_summarizing"}},
            ),
        )
    )

    with pytest.raises(ConfigError) as error:
        run_components(config)

    assert error.value.problems == [
        f"{WHERE}.reasoning_replay: 'no' is not a bool",
        f"{WHERE}.num_retries: 11 is above the maximum 10",
        "environments.0.eval_overrides.harness_options.request_timeout_s: 0 is below the minimum 1",
        "environments.0.eval_overrides.harness_options.condensor: unknown key; did you mean "
        "environments.0.eval_overrides.harness_options.condenser?",
    ]


def test_a_spec_without_options_takes_none(tmp_path):
    config = load(train_config(tmp_path, openhands(harness="platoon-bash", harness_options={"max_turns": 5})))

    with pytest.raises(ConfigError, match=f"{WHERE}.max_turns: unknown key; harness platoon-bash@1 takes no options"):
        run_components(config)


def test_a_function_harness_still_takes_its_options_unchecked(tmp_path):
    entry = {"service": "table", "harness": "platoon-react", "harness_options": {"system_prompt": "Be brief."}}

    record = harness_record(load(train_config(tmp_path, entry)))

    assert record["ref"] == "platoon-react" and "options" not in record


def test_a_mixtures_shared_mapping_must_suit_each_of_its_harnesses(tmp_path):
    entry = openhands(harness={"openhands": 1, "platoon-react": 1}, harness_options={"system_prompt": "Be brief."})

    with pytest.raises(ConfigError) as error:
        run_components(load(train_config(tmp_path, entry)))

    assert error.value.problems == [
        f"{WHERE}.system_prompt: unknown key; harness openhands@1.54.0 takes reasoning_replay, request_timeout_s, "
        "num_retries, condenser",
        f"{WHERE}: the harnesses of the mixture ['openhands', 'platoon-react'] share one mapping, which must suit each "
        "of them; give a harness whose options the others do not take an environments entry of its own",
    ]


def test_eval_overrides_take_options_and_say_where_the_rest_goes(tmp_path):
    path = train_config(tmp_path, openhands(eval_overrides={"max_steps": 500}))

    with pytest.raises(ConfigError, match=r"environments.0.eval_overrides.max_steps: unknown key; the step limit is"):
        load(path)


def test_the_trainers_route_check_refuses_bad_options_for_library_callers():
    environments = normalize_environment_configs([openhands(harness_options={"condenser": "summarize"})])

    with pytest.raises(ConfigError, match=f"{WHERE}.condenser: 'summarize' is not one of 'none', 'llm_summarizing'"):
        check_environment_routes(environments)


# --- layering ------------------------------------------------------------------------------------------------------


def test_command_line_overrides_reach_a_list_entrys_options_and_are_checked_like_the_configs(tmp_path):
    path = train_config(tmp_path, openhands(name="a", weight=1), openhands(name="b", weight=2))

    config = load(
        path,
        "environments.1.harness_options.reasoning_replay=false",
        "environments.1.harness_options.condenser=llm_summarizing",
        "environments.1.eval_overrides.harness_options.num_retries=5",
        "environments.0.harness_options.request_timeout_s=120",
    )

    assert config.environments[0].harness_options == {"request_timeout_s": 120}
    assert config.environments[1].harness_options == {"reasoning_replay": False, "condenser": "llm_summarizing"}
    assert config.environments[1].eval_overrides.harness_options == {"num_retries": 5}
    options = harness_record(config)["options"]
    assert options["a"] == {"train": {**DEFAULTS, "request_timeout_s": 120}}
    assert options["b"] == {
        "train": {**DEFAULTS, "reasoning_replay": False, "condenser": "llm_summarizing"},
        "eval": {**DEFAULTS, "reasoning_replay": False, "condenser": "llm_summarizing", "num_retries": 5},
    }
    with pytest.raises(ConfigError, match=r"environments.1.harness_options.num_retries: 'many' is not an int"):
        run_components(load(path, "environments.1.harness_options.num_retries=many"))


def test_a_named_environments_defaults_lie_under_the_config_and_the_command_line(tmp_path, monkeypatch):
    (tmp_path / "acme_tasks.py").write_text(
        textwrap.dedent(
            """
            import platoon as pt

            TASKS = pt.EnvironmentEntry(
                "acme-tasks",
                service="harbor",
                service_kwargs={"dataset_dir": "/tasks"},
                harness="openhands",
                harness_defaults={"openhands": {"num_retries": 2, "request_timeout_s": 600}},
            )
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    entry = {"env": "acme_tasks:TASKS", "harness_options": {"request_timeout_s": 900}}

    config = load(train_config(tmp_path, entry), "environments.0.harness_options.condenser=llm_summarizing")

    expected = {**DEFAULTS, "num_retries": 2, "request_timeout_s": 900, "condenser": "llm_summarizing"}
    assert harness_record(config)["options"] == {"default": {"train": expected}}
    assert route_options(config, "train") == {
        "num_retries": 2,
        "request_timeout_s": 900,
        "condenser": "llm_summarizing",
    }


def test_a_named_environment_cannot_set_eval_overrides_of_harness_options():
    import platoon as pt

    with pytest.raises(ValueError, match="sets eval_overrides.harness_options, which a named environment may not set"):
        pt.EnvironmentEntry("acme-tasks", service="harbor", eval_overrides={"harness_options": {"num_retries": 5}})


# --- phases --------------------------------------------------------------------------------------------------------


def test_routes_carry_each_phases_options(tmp_path):
    config = load(
        train_config(
            tmp_path,
            openhands(harness_options={"condenser": "llm_summarizing"}, eval_overrides={"harness_options": ADAPTER}),
        )
    )

    assert route_options(config, "train") == {"condenser": "llm_summarizing"}
    for split in ("eval", "test", "val"):  # every split but train is an evaluation
        assert route_options(config, split) == {"condenser": "llm_summarizing", **ADAPTER}
    assert harness_record(config)["options"] == {
        "default": {
            "train": {**DEFAULTS, "condenser": "llm_summarizing"},
            "eval": {**DEFAULTS, "condenser": "llm_summarizing", **ADAPTER},
        }
    }


def test_lock_json_records_the_options_with_their_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr("platoon.train.tinker.train.run_tinker_training", lambda args: None)
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")  # train checks its key before anything else: past it

    cli.main(["train", str(train_config(tmp_path, evaluates=False))])

    lock = json.loads((tmp_path / "logs/exp/trial/lock.json").read_text())
    (record,) = [component for component in lock["components"] if component["kind"] == "harness"]
    assert (record["ref"], record["resolved"]) == ("openhands", "openhands@1.54.0")
    assert record["options"] == {"default": {"train": DEFAULTS}}  # no evaluation: only training's


def test_an_evaluation_config_runs_and_records_its_eval_overrides(tmp_path):
    entry = openhands(harness_options={"num_retries": 1}, eval_overrides={"harness_options": ADAPTER})
    config = load(eval_config(tmp_path, entry), kind="eval")

    assert route_options(config, config.split) == ADAPTER  # its num_retries over the entry's own
    assert route_options(config, "train") == {"num_retries": 1}  # platoon eval on the train split: no overrides
    assert harness_record(config)["options"] == {"default": {"eval": {**DEFAULTS, **ADAPTER}}}


def test_lock_json_records_the_phase_of_the_split_a_command_runs(tmp_path):
    """An evaluation of the train split runs the entry's own options, and ``platoon rollout --split`` runs its split's,
    whatever phases the config itself runs: lock.json records what ran."""

    entry = openhands(harness_options={"num_retries": 1}, eval_overrides={"harness_options": ADAPTER})
    on_train = load(eval_config(tmp_path, entry), "split=train", kind="eval")
    assert route_options(on_train, on_train.split) == {"num_retries": 1}
    assert harness_record(on_train)["options"] == {"default": {"train": {**DEFAULTS, "num_retries": 1}}}

    training = load(train_config(tmp_path, entry, evaluates=False))
    (record,) = [record for record in run_components(training, split="test") if record["kind"] == "harness"]
    assert record["options"] == {"default": {"eval": {**DEFAULTS, **ADAPTER}}}
    (record,) = [record for record in run_components(training, split="train") if record["kind"] == "harness"]
    assert record["options"] == {"default": {"train": {**DEFAULTS, "num_retries": 1}}}


def test_the_harbor_adapters_settings_are_an_evaluation_configs_options(tmp_path):
    """RFC 0016, section 2: an evaluation that scores as Harbor's adapter does is a config, not another harness."""

    entry = openhands(name="harbor-openhands", harness_options=ADAPTER)
    path = eval_config(tmp_path, entry)

    config = load(path, "inference.workflow.rollout_config.max_steps=500", kind="eval")

    assert route_options(config, config.split) == ADAPTER
    assert config.inference.workflow.rollout_config.max_steps == 500
    assert harness_record(config)["options"] == {"harbor-openhands": {"eval": {**DEFAULTS, **ADAPTER}}}
