"""Config loading: ``extends`` chains, typed overrides, strict keys with suggestions, deprecated keys, and run
records."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import pydantic
import pytest
import yaml

import platoon as pt
from platoon.deprecations import DeprecatedKey, Deprecation, PlatoonDeprecationWarning
from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.utils.config import (
    IMAGE_PULLS_ENV,
    LOCK_NAME,
    RESOLVED_CONFIG_NAME,
    ConfigError,
    apply_overrides,
    config_sources,
    distribution_commit,
    find_deprecations,
    image_pulls,
    load_config,
    load_yaml_config,
    resolve_config,
    run_lock,
    write_run_record,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
# A layered config (extends:), frozen from the RAO recipe's linear arm when it spelled out its environment.
RAO_LINEAR = REPO_ROOT / "tests/fixtures/layered/arm.yaml"


@dataclass
class Inner:
    length: int = 1
    ratio: float = 0.5


@dataclass
class Item:
    name: str
    weight: float = 1.0
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class Schema:
    name: str = "run"
    count: int = 0
    ratio: float = 1.0
    flag: bool = False
    mode: Literal["fast", "careful"] = "fast"
    limit: int | None = None
    label: str | None = None
    tags: list[str] = field(default_factory=list)
    harness: str | dict[str, float] | None = None
    inner: Inner = field(default_factory=Inner)
    items: list[Item] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def _write(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


# --- extends -----------------------------------------------------------------------------------------------------


def test_extends_chains_relative_to_each_file_merging_mappings_and_replacing_lists(tmp_path):
    _write(tmp_path / "presets/base.yaml", {"inner": {"length": 3, "ratio": 0.1}, "tags": ["a", "b"], "count": 1})
    _write(tmp_path / "presets/mid.yaml", {"extends": "base.yaml", "inner": {"ratio": 0.2}, "tags": ["c"]})
    run = _write(tmp_path / "runs/run.yaml", {"extends": "../presets/mid.yaml", "count": 2})

    assert load_yaml_config(run) == {"inner": {"length": 3, "ratio": 0.2}, "tags": ["c"], "count": 2}
    assert config_sources(run) == [
        (tmp_path / "presets/base.yaml").resolve(),
        (tmp_path / "presets/mid.yaml").resolve(),
        run.resolve(),
    ]


def test_a_list_of_bases_merges_in_order_under_the_file(tmp_path):
    _write(tmp_path / "a.yaml", {"name": "a", "inner": {"length": 1}, "extra": {"from_a": True}})
    _write(tmp_path / "b.yaml", {"name": "b", "inner": {"ratio": 0.9}})
    run = _write(tmp_path / "run.yaml", {"extends": ["a.yaml", "b.yaml"], "count": 5})

    config = resolve_config(run, [], Schema).config

    assert (config.name, config.count, config.inner, config.extra) == ("b", 5, Inner(1, 0.9), {"from_a": True})


def test_an_extends_cycle_is_an_error_naming_the_chain(tmp_path):
    _write(tmp_path / "a.yaml", {"extends": "b.yaml"})
    _write(tmp_path / "b.yaml", {"extends": "sub/../a.yaml"})

    with pytest.raises(ConfigError, match=r"extends cycle: .*a\.yaml -> .*b\.yaml -> .*a\.yaml"):
        load_yaml_config(tmp_path / "a.yaml")


def test_a_missing_base_names_the_file_that_extends_it(tmp_path):
    run = _write(tmp_path / "run.yaml", {"extends": "nowhere.yaml"})

    with pytest.raises(ConfigError, match=r"run\.yaml: extends .*nowhere\.yaml, which does not exist"):
        load_yaml_config(run)


def test_extends_must_name_paths(tmp_path):
    with pytest.raises(ConfigError, match="extends must be a path or a list of paths"):
        load_yaml_config(_write(tmp_path / "run.yaml", {"extends": {"file": "a.yaml"}}))


# --- overrides ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("override", "key", "expected"),
    [
        ("name=1", "name", "1"),  # a string field keeps the text
        ("count=1", "count", 1),
        ("count=1_000", "count", 1000),
        ("ratio=1e-5", "ratio", 1e-5),
        ("ratio=3", "ratio", 3.0),
        ("flag=yes", "flag", True),
        ("flag=False", "flag", False),
        ("mode=careful", "mode", "careful"),
        ("limit=null", "limit", None),
        ("limit=7", "limit", 7),
        ("label=None", "label", None),
        ("tags=[a, b]", "tags", ["a", "b"]),
        ("harness=platoon-codeact", "harness", "platoon-codeact"),
        ("harness={platoon-codeact: 2, platoon-react: 1}", "harness", {"platoon-codeact": 2, "platoon-react": 1}),
        ("extra.seed=7", "extra", {"seed": 7}),
        ("extra.lr=1e-4", "extra", {"lr": 1e-4}),
        ("extra.when=never", "extra", {"when": "never"}),
    ],
)
def test_override_values_parse_against_the_field_type(override, key, expected):
    data = apply_overrides({}, [override], Schema)

    assert data[key] == expected
    assert type(data[key]) is type(expected)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("count=1.5", "count is an int, got '1.5'"),
        ("flag=maybe", "flag is a bool; use true or false"),
        ("mode=slow", "mode must be one of 'fast', 'careful', got 'slow'"),
        ("name=null", None),  # a str field takes the text "null"
        ("tags=a", "tags: expected a list"),
        ("inner={length: x}", "inner.length: 'x' is not a int"),
        ("count", "is not of the form key=value"),
        ("=3", "is not of the form key=value"),
    ],
)
def test_badly_typed_overrides_fail(override, message):
    if message is None:
        assert apply_overrides({}, [override], Schema) == {"name": "null"}
        return
    with pytest.raises(ConfigError, match=re.escape(message)):
        apply_overrides({}, [override], Schema)


def test_overrides_reach_into_lists_by_index_and_append_with_plus_equals():
    data = {"items": [{"name": "a"}, {"name": "b"}], "tags": ["x"]}

    data = apply_overrides(
        data,
        [
            "items.0.weight=2",
            "items.-1.options.depth=3",
            "items+={name: c, weight: 0.5}",
            "tags+=y",
            "extra.seen+=1",
        ],
        Schema,
    )

    assert data["items"] == [
        {"name": "a", "weight": 2.0},
        {"name": "b", "options": {"depth": 3}},
        {"name": "c", "weight": 0.5},
    ]
    assert data["tags"] == ["x", "y"]
    assert data["extra"] == {"seen": [1]}


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("items.2.name=x", "items has 2 entries; index 2 is out of range"),
        ("items.first.name=x", "items is a list; 'first' is not an index"),
        ("items+={weight: 1}", "items.2.name: required key is missing"),
        ("count+=1", "count is a int, not a list"),
        ("count.value=1", "count is a int, which has no key 'value'"),
    ],
)
def test_list_paths_and_appends_are_checked(override, message):
    with pytest.raises(ConfigError, match=re.escape(message)):
        apply_overrides({"items": [{"name": "a"}, {"name": "b"}]}, [override], Schema)


def test_an_unknown_override_key_fails_with_its_full_path_and_a_suggestion():
    with pytest.raises(ConfigError) as error:
        apply_overrides({}, ["inner.lenght=3", "nme=x", "zzz=1"], Schema)

    assert error.value.problems == [
        "override inner.lenght=3: inner.lenght: unknown key; did you mean inner.length?",
        "override nme=x: nme: unknown key; did you mean name?",
        "override zzz=1: zzz: unknown key",
    ]


def test_a_mapping_override_merges_into_the_mapping_it_replaces():
    data = apply_overrides({"inner": {"length": 4, "ratio": 0.1}}, ["inner={ratio: 0.3}"], Schema)

    assert data["inner"] == {"length": 4, "ratio": 0.3}


def test_overrides_without_a_schema_parse_as_yaml():
    data = apply_overrides({"a": {"b": 1}}, ["a.c=0", "a.d=yes", "a.e=[1, 2]", "f=text"])

    assert data == {"a": {"b": 1, "c": 0, "d": True, "e": [1, 2]}, "f": "text"}


# --- validation --------------------------------------------------------------------------------------------------


def test_unknown_file_keys_fail_with_dotted_paths_and_suggestions(tmp_path):
    run = _write(
        tmp_path / "run.yaml",
        {"cont": 1, "inner": {"rato": 0.2}, "items": [{"name": "a", "wieght": 2}], "extra": {"anything": {"goes": 1}}},
    )

    with pytest.raises(ConfigError) as error:
        resolve_config(run, [], Schema)

    assert error.value.problems == [
        "cont: unknown key; did you mean count?",
        "inner.rato: unknown key; did you mean inner.ratio?",
        "items.0.wieght: unknown key; did you mean items.0.weight?",
    ]
    assert str(run) in str(error.value)


def test_wrong_types_and_missing_required_keys_fail(tmp_path):
    run = _write(tmp_path / "run.yaml", {"count": "many", "flag": 1, "mode": "slow", "items": [{"weight": 2}]})

    with pytest.raises(ConfigError) as error:
        resolve_config(run, [], Schema)

    assert error.value.problems == [
        "count: 'many' is not a int",
        "flag: 1 is not a bool",
        "mode: 'slow' is not one of 'fast', 'careful'",
        "items.0.name: required key is missing",
    ]


def test_yaml_exponents_without_a_dot_read_as_floats_in_float_fields(tmp_path):
    run = tmp_path / "run.yaml"
    run.write_text("ratio: 3e-5\nname: 1e3\ninner:\n  ratio: 1e-8\n")

    resolved = resolve_config(run, [], Schema)

    assert (resolved.config.ratio, resolved.config.inner.ratio, resolved.config.name) == (3e-5, 1e-8, "1e3")
    assert resolved.data["ratio"] == 3e-5


def test_post_init_errors_surface_as_config_errors(tmp_path):
    run = _write(
        tmp_path / "run.yaml", {"train": {"model_name": "m", "renderer_name": "r"}, "eval": {}, "log_path": "x"}
    )

    with pytest.raises(ConfigError, match="group_size must be positive"):
        resolve_config(run, ["train.workflow_config.group_size=0"], PlatoonTinkerRLTrainerConfig)


def test_a_layered_config_resolves_with_typed_overrides_into_the_trainer_config():
    resolved = resolve_config(
        RAO_LINEAR,
        [
            "stats.trial_name=1",
            "train.batch_size=8",
            "train.optimizer.learning_rate=1e-5",
            "environments.0.dataset_kwargs.split_name=train-hard",
            "stats.wandb.tags+=smoke",
        ],
        PlatoonTinkerRLTrainerConfig,
    )
    config = resolved.config

    assert config.stats.trial_name == "1"
    assert (config.train.batch_size, config.train.optimizer.learning_rate) == (8, 1e-5)
    assert config.environments[0].dataset_kwargs == {"split_name": "train-hard"}
    assert config.stats.wandb.tags[-1] == "smoke"
    assert config.train.workflow_config.leave_one_out_baseline is True


def test_a_typo_in_a_layered_config_override_suggests_the_field():
    with pytest.raises(ConfigError, match=r"did you mean train\.workflow_config\.group_size\?"):
        resolve_config(RAO_LINEAR, ["train.workflow_config.group_sise=4"], PlatoonTinkerRLTrainerConfig)


# --- nested defaults ---------------------------------------------------------------------------------------


@dataclass
class Tuned:
    """A section whose field default is not its class's own (``Inner(length=3)``), beside one that is."""

    inner: Inner = field(default_factory=lambda: Inner(length=3))
    plain: Inner = field(default_factory=Inner)


def test_a_partial_section_is_laid_over_its_field_default(tmp_path):
    run = _write(tmp_path / "run.yaml", {"inner": {"ratio": 0.25}, "plain": {"ratio": 0.25}})

    config = resolve_config(run, [], Tuned).config

    assert config.inner == Inner(length=3, ratio=0.25)
    assert config.plain == Inner(length=1, ratio=0.25)


def _trainer_config(tmp_path: Path, eval_section: dict[str, Any], *overrides: str, **train: Any) -> Any:
    run = _write(
        tmp_path / "train.yaml",
        {
            "train": {"model_name": "m", "renderer_name": "r", **train},
            "eval": eval_section,
            "log_path": str(tmp_path),
        },
    )
    return resolve_config(run, list(overrides), PlatoonTinkerRLTrainerConfig).config


def test_a_partial_eval_workflow_config_keeps_the_eval_defaults(tmp_path):
    workflow = _trainer_config(
        tmp_path, {"workflow_config": {"rollout_config": {"max_steps": 10}}}
    ).eval.workflow_config

    assert workflow.rollout_config.max_steps == 10
    assert (workflow.group_size, workflow.filter_errors, workflow.filter_zero_advantage_datums) == (1, False, False)


def test_an_override_into_the_eval_section_keeps_the_eval_defaults(tmp_path):
    config = _trainer_config(tmp_path, {"every": 0}, "eval.workflow_config.rollout_config.timeout=600")

    assert (config.eval.workflow_config.group_size, config.eval.workflow_config.rollout_config.timeout) == (1, 600)


def test_keys_a_partial_section_sets_still_win_and_train_keeps_the_class_defaults(tmp_path):
    config = _trainer_config(
        tmp_path,
        {"workflow_config": {"group_size": 4, "filter_errors": True}},
        "train.workflow_config.leave_one_out_baseline=true",
    )

    assert (config.eval.workflow_config.group_size, config.eval.workflow_config.filter_errors) == (4, True)
    assert config.eval.workflow_config.filter_zero_advantage_datums is False
    train = config.train.workflow_config
    assert (train.group_size, train.filter_errors, train.leave_one_out_baseline) == (8, True, True)


def test_the_deprecated_propagation_key_still_loads_in_partial_sections(tmp_path):
    legacy = {"rollout_config": {"propogate_root_success": True}}

    with pytest.warns(PlatoonDeprecationWarning, match="propogate_root_success: deprecated") as warned:
        config = _trainer_config(tmp_path, {"workflow_config": legacy}, workflow_config=legacy)

    deprecations = [item for item in warned if item.category is PlatoonDeprecationWarning]
    assert len(deprecations) == 2  # one for each section that sets it

    assert config.train.workflow_config.rollout_config.propagate_root_success is True
    assert config.eval.workflow_config.rollout_config.propagate_root_success is True
    assert config.eval.workflow_config.group_size == 1


def test_a_loss_fn_config_replaces_the_default_mapping_whole(tmp_path):
    # Free-form: a loss's own keys, so nothing is merged in (another loss may not take the cispo thresholds).
    config = _trainer_config(tmp_path, {}, "train.loss_fn_config={clip_high_threshold: 4.0}")

    assert config.train.loss_fn_config == {"clip_high_threshold": 4.0}


EVAL_DEFAULTS = "group_size: 1, filter_errors: false, filter_zero_advantage_datums: false"


@pytest.mark.parametrize(
    ("workflow", "refusal"),
    [
        ({"min_successful_group_size": 4}, r"min_successful_group_size must be in [1, group_size]"),
        ({"straggler_timeout_seconds": 600, "straggler_quorum": 6}, "straggler quorum must be in [1, group_size]"),
        (
            {"straggler_policies": {"slow": {"timeout_seconds": 60, "quorum": 4}}},
            "straggler quorum must be in [1, group_size]",
        ),
    ],
    ids=["min-successful-group-size", "straggler-quorum", "straggler-policies"],
)
def test_a_partial_section_its_defaults_refuse_names_the_section_and_the_defaults_it_kept(tmp_path, workflow, refusal):
    # Group settings copied from train into eval, without a group_size: eval's own default, 1, refuses them.
    with pytest.raises(ConfigError) as raised:
        _trainer_config(tmp_path, {"workflow_config": workflow})

    assert f"eval.workflow_config: {refusal}" in str(raised.value)
    assert f"(the keys it leaves out keep their eval.workflow_config defaults: {EVAL_DEFAULTS})" in str(raised.value)


def test_the_defaults_a_refused_section_kept_leave_out_the_keys_it_sets(tmp_path):
    with pytest.raises(ConfigError) as raised:
        _trainer_config(tmp_path, {"workflow_config": {"group_size": 2, "min_successful_group_size": 4}})

    assert str(raised.value).endswith(
        "eval.workflow_config: min_successful_group_size must be in [1, group_size] (the keys it leaves out keep "
        "their eval.workflow_config defaults: filter_errors: false, filter_zero_advantage_datums: false)"
    )


def test_a_refused_section_that_kept_no_defaults_reads_as_before(tmp_path):
    # train.workflow_config defaults to its class's defaults, so nothing it leaves out differs from them.
    with pytest.raises(ConfigError) as raised:
        _trainer_config(tmp_path, {}, workflow_config={"min_successful_group_size": 16})

    assert str(raised.value).endswith("\n  ValueError: min_successful_group_size must be in [1, group_size]")


@dataclass
class PluginTrainerConfig(PlatoonTinkerRLTrainerConfig):
    plugin: dict[str, Any] = field(default_factory=dict)


def _rao_linear_data() -> dict[str, Any]:
    """The RAO linear arm's YAML, with its ``extends`` made absolute so a copy elsewhere still resolves."""
    data = yaml.safe_load(RAO_LINEAR.read_text())
    data["extends"] = str(RAO_LINEAR.parent / data["extends"])
    return data


def test_an_environment_names_a_trainer_config_subclass_that_validates_its_sections(tmp_path):
    data = _rao_linear_data()
    data["environments"][0]["trainer_config"] = f"{__name__}:PluginTrainerConfig"
    data["plugin"] = {"judge": "on"}
    run = _write(tmp_path / "run.yaml", data)

    config = resolve_config(run, [], PlatoonTinkerRLTrainerConfig).config

    assert type(config) is PluginTrainerConfig
    assert config.plugin == {"judge": "on"}


@pytest.mark.parametrize(
    ("reference", "why"),
    [
        ("no.such.module:Config", "ModuleNotFoundError: No module named 'no'"),
        (f"{__name__}:MissingTrainerConfig", f"AttributeError: module '{__name__}' has no attribute"),
        ("unregistered-trainer-config", "ValueError: Expected an import path"),  # a name no package registers
    ],
    ids=["module", "attribute", "name"],
)
def test_a_trainer_config_that_does_not_resolve_is_a_config_error(tmp_path, reference, why):
    # It used to be logged, and the config then validated against the base schema, without the class's checks.
    data = _rao_linear_data()
    data["environments"][0]["trainer_config"] = reference
    run = _write(tmp_path / "run.yaml", data)

    with pytest.raises(ConfigError) as raised:
        resolve_config(run, [], PlatoonTinkerRLTrainerConfig)

    assert raised.value.source == str(run)
    [problem] = raised.value.problems
    assert problem.startswith(f"environments.0.trainer_config: {reference!r} does not resolve ({why}")


def test_a_trainer_config_that_names_something_other_than_a_class_is_a_config_error(tmp_path):
    data = _rao_linear_data()
    data["environments"][0]["trainer_config"] = f"{__name__}:_rao_linear_data"
    run = _write(tmp_path / "run.yaml", data)

    with pytest.raises(ConfigError, match=rf"environments.0.trainer_config: '{__name__}:_rao_linear_data' is not a"):
        resolve_config(run, [], PlatoonTinkerRLTrainerConfig)


def test_an_evaluation_config_loads_as_itself_when_its_entries_name_a_training_config_class(tmp_path):
    from platoon.eval.run import PlatoonEvalConfig

    run = _write(
        tmp_path / "eval.yaml",
        {
            "inference": {"model_name": "m", "output_dir": str(tmp_path / "out")},
            "environments": [{"service": "harbor", "trainer_config": f"{__name__}:PluginTrainerConfig"}],
        },
    )

    assert type(resolve_config(run, [], PlatoonEvalConfig).config) is PlatoonEvalConfig


def test_an_override_keeps_the_trainer_config_class_the_files_chose(tmp_path):
    data = _rao_linear_data()
    data["environments"][0]["trainer_config"] = f"{__name__}:PluginTrainerConfig"
    run = _write(tmp_path / "run.yaml", data)

    overrides = ["plugin.judge=strict", "environments.0.dataset_kwargs.limit=4"]
    config = resolve_config(run, overrides, PlatoonTinkerRLTrainerConfig).config
    assert type(config) is PluginTrainerConfig and config.plugin == {"judge": "strict"}

    with pytest.raises(ConfigError) as raised:
        resolve_config(run, ["environments.0.trainer_config=null"], PlatoonTinkerRLTrainerConfig)
    assert raised.value.problems == [
        "environments[].trainer_config: an override changes the config class from PluginTrainerConfig to "
        "PlatoonTinkerRLTrainerConfig, but the class is chosen from the config files, before the overrides; set "
        "trainer_config in a file"
    ]


def test_an_override_that_names_a_trainer_config_that_does_not_resolve_is_a_config_error(tmp_path):
    run = _write(tmp_path / "run.yaml", _rao_linear_data())

    with pytest.raises(ConfigError, match=r"environments.0.trainer_config: 'no.such.module:Config' does not resolve"):
        resolve_config(run, ["environments.0.trainer_config=no.such.module:Config"], PlatoonTinkerRLTrainerConfig)


class PydanticInner(pydantic.BaseModel):
    depth: int = 1


class PydanticSchema(pydantic.BaseModel):
    name: str = "x"
    inner: PydanticInner = PydanticInner()
    tags: list[str] = []


def test_pydantic_models_are_schemas_too(tmp_path):
    run = _write(tmp_path / "run.yaml", {"inner": {"depth": 2}})

    resolved = resolve_config(run, ["name=7", "tags+=a"], PydanticSchema)

    assert resolved.config == PydanticSchema(name="7", inner=PydanticInner(depth=2), tags=["a"])
    with pytest.raises(ConfigError, match=r"inner\.dept: unknown key; did you mean inner\.depth\?"):
        resolve_config(run, ["inner.dept=3"], PydanticSchema)


def test_load_config_keeps_its_command_line_forms(tmp_path):
    run = _write(tmp_path / "run.yaml", {"count": 1})

    config, data = load_config(
        ["--config", str(run), "--inner.length", "4", "--flag", "--ratio=0.25", "name=7", "tags+=a"], Schema
    )

    assert config == Schema(name="7", count=1, ratio=0.25, flag=True, inner=Inner(length=4), tags=["a"])
    assert data == {"count": 1, "inner": {"length": 4}, "flag": True, "ratio": 0.25, "name": "7", "tags": ["a"]}
    with pytest.raises(ConfigError, match="unexpected argument 'stray'"):
        load_config(["--config", str(run), "stray"], Schema)


# --- deprecations ------------------------------------------------------------------------------------------------

LEGACY_PROPAGATION = "train.workflow_config.rollout_config.propogate_root_success"
PROPAGATION = "train.workflow_config.rollout_config.propagate_root_success"
LEGACY_MESSAGE = f"{LEGACY_PROPAGATION}: deprecated; use {PROPAGATION} instead (platoon 0.3.0 removes it)"


def _legacy_training_config(tmp_path: Path, rollout_config: dict[str, Any]) -> Path:
    return _write(
        tmp_path / "train.yaml",
        {
            "train": {"model_name": "m", "renderer_name": "r", "workflow_config": {"rollout_config": rollout_config}},
            "eval": {"every": 0},
            "log_path": str(tmp_path),
        },
    )


def test_a_deprecated_key_loads_with_a_warning_that_names_its_path_and_replacement(tmp_path):
    run = _legacy_training_config(tmp_path, {"propogate_root_success": True})

    with pytest.warns(PlatoonDeprecationWarning) as warned:
        resolved = resolve_config(run, [], PlatoonTinkerRLTrainerConfig)

    deprecations = [warning for warning in warned if warning.category is PlatoonDeprecationWarning]
    assert [str(warning.message) for warning in deprecations] == [LEGACY_MESSAGE]
    assert deprecations[0].filename == __file__  # named where the caller outside platoon loaded the config
    assert resolved.deprecations == (Deprecation(LEGACY_PROPAGATION, PROPAGATION, "0.3.0"),)
    assert resolved.config.train.workflow_config.rollout_config.propagate_root_success is True  # still read


def test_a_deprecated_key_counts_wherever_the_config_gives_it(tmp_path):
    base = _legacy_training_config(tmp_path / "base", {"propogate_root_success": None})
    run = _write(tmp_path / "run.yaml", {"extends": str(base)})
    overrides = ["eval.workflow_config.rollout_config.propogate_root_success=false"]

    resolved = resolve_config(run, overrides, PlatoonTinkerRLTrainerConfig, warn=False)

    # A null in a base counts too: the release that removes the key refuses it whatever its value.
    assert [deprecation.path for deprecation in resolved.deprecations] == [
        LEGACY_PROPAGATION,
        "eval.workflow_config.rollout_config.propogate_root_success",
    ]


def test_an_evaluation_config_reports_the_key_under_its_own_section(tmp_path):
    from platoon.eval.run import PlatoonEvalConfig

    run = _write(
        tmp_path / "eval.yaml",
        {"inference": {"model_name": "m", "workflow": {"rollout_config": {"propogate_root_success": True}}}},
    )

    with pytest.warns(PlatoonDeprecationWarning, match=r"^inference\.workflow\.rollout_config\.propogate_root_success"):
        resolved = resolve_config(run, [], PlatoonEvalConfig)

    assert resolved.deprecations[0].replacement == "inference.workflow.rollout_config.propagate_root_success"


def test_a_config_without_deprecated_keys_loads_without_a_warning(tmp_path):
    run = _legacy_training_config(tmp_path, {"propagate_root_success": True})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resolved = resolve_config(run, [], PlatoonTinkerRLTrainerConfig)

    assert resolved.deprecations == ()


def test_an_error_filter_makes_a_deprecated_key_raise(tmp_path):
    run = _legacy_training_config(tmp_path, {"propogate_root_success": True})

    with warnings.catch_warnings():
        warnings.simplefilter("error", pt.PlatoonDeprecationWarning)
        with pytest.raises(PlatoonDeprecationWarning, match=re.escape(LEGACY_MESSAGE)):
            load_config(["--config", str(run)], PlatoonTinkerRLTrainerConfig)
        # The command line reports them itself: warn=False only records them.
        assert resolve_config(run, [], PlatoonTinkerRLTrainerConfig, warn=False).deprecations


def test_the_deprecation_warning_is_a_future_warning_that_python_shows_by_default(tmp_path):
    """Python hides a DeprecationWarning outside __main__; a FutureWarning shows in any program that loads a config
    with a deprecated key."""

    run = _legacy_training_config(tmp_path, {"propogate_root_success": True})
    load = (
        "from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig\n"
        "from platoon.utils.config import resolve_config\n"
        f"resolve_config({str(run)!r}, [], PlatoonTinkerRLTrainerConfig)\n"
    )

    completed = subprocess.run([sys.executable, "-c", load], capture_output=True, text=True, timeout=300)

    assert completed.returncode == 0, completed.stderr
    assert f"<string>:3: PlatoonDeprecationWarning: {LEGACY_MESSAGE}" in completed.stderr
    assert issubclass(pt.PlatoonDeprecationWarning, FutureWarning)


@dataclass
class Retiring:
    DEPRECATED_KEYS: ClassVar[dict[str, DeprecatedKey]] = {
        "old_size": DeprecatedKey(removed_in="0.4.0", renamed_to="size"),
        "hook": DeprecatedKey(removed_in="0.5.0", instead="a hook option"),
    }

    size: int = 1
    old_size: int | None = None
    hook: str | None = None


class RetiringModel(pydantic.BaseModel, extra="forbid"):
    DEPRECATED_KEYS: ClassVar[dict[str, DeprecatedKey]] = {"old": DeprecatedKey(removed_in="0.4.0", renamed_to="new")}

    new: int = 0
    old: int | None = None


@dataclass
class Holder:
    one: Retiring | None = None
    many: list[Retiring] = field(default_factory=list)
    named: dict[str, Retiring] = field(default_factory=dict)
    free: dict[str, Any] = field(default_factory=dict)


def test_deprecated_keys_are_found_in_lists_mappings_and_unions_with_their_paths():
    data = {
        "one": {"old_size": 2},
        "many": [{"size": 1}, {"hook": "x"}],
        "named": {"a": {"old_size": None, "hook": "y"}},
        "free": {"old_size": 3},  # a free-form mapping's keys belong to someone else
    }

    found = find_deprecations(data, Holder)

    assert [(item.path, item.replacement, item.removed_in) for item in found] == [
        ("one.old_size", "one.size", "0.4.0"),
        ("many.1.hook", "a hook option", "0.5.0"),
        ("named.a.old_size", "named.a.size", "0.4.0"),
        ("named.a.hook", "a hook option", "0.5.0"),
    ]
    assert found[1].message == "many.1.hook: deprecated; use a hook option instead (platoon 0.5.0 removes it)"
    assert find_deprecations({"old_size": 1}, Retiring)[0].replacement == "size"  # a top-level key
    assert [item.path for item in find_deprecations({"old": 1, "new": 2}, RetiringModel)] == ["old"]  # pydantic too


def test_a_deprecated_key_names_one_replacement():
    with pytest.raises(ValueError, match="renamed_to and instead"):
        DeprecatedKey(removed_in="0.3.0")
    with pytest.raises(ValueError, match="renamed_to and instead"):
        DeprecatedKey(removed_in="0.3.0", renamed_to="a", instead="b")


def test_the_lock_records_the_deprecated_keys_a_config_sets(tmp_path):
    run = _legacy_training_config(tmp_path, {"propogate_root_success": True})
    resolved = resolve_config(run, [], PlatoonTinkerRLTrainerConfig, warn=False)

    record = write_run_record(tmp_path / "out", resolved, argv=["platoon", "train", str(run)])

    lock = json.loads(record.lock_path.read_text())
    assert lock["deprecations"] == [{"path": LEGACY_PROPAGATION, "replacement": PROPAGATION, "removed_in": "0.3.0"}]
    assert "deprecations" not in run_lock([])  # a lock written without a loaded config cannot know them


# --- run records -------------------------------------------------------------------------------------------------


def test_a_run_record_holds_the_resolved_config_and_the_lock(tmp_path):
    _write(tmp_path / "base.yaml", {"inner": {"length": 2}})
    run = _write(tmp_path / "run.yaml", {"extends": "base.yaml", "count": 1})
    resolved = resolve_config(run, ["count=3"], Schema)

    record = write_run_record(tmp_path / "out/run-1", resolved, argv=["platoon", "train", str(run), "count=3"])

    assert record.config_path == tmp_path / "out/run-1" / RESOLVED_CONFIG_NAME
    snapshot = record.config_path.read_text()
    assert f"# source: {(tmp_path / 'base.yaml').resolve()}" in snapshot
    assert "# override: count=3" in snapshot
    assert yaml.safe_load(snapshot) == {"inner": {"length": 2}, "count": 3}
    assert resolve_config(record.config_path, [], Schema).config == resolved.config

    lock = json.loads((tmp_path / "out/run-1" / LOCK_NAME).read_text())
    assert set(lock["versions"]) == {"platoon", "tinker", "tinker_cookbook", "openreward", "hpc_sandbox"}
    assert lock["versions"]["platoon"]  # from pyproject when platoon is not installed
    assert lock["versions"]["tinker"] == "0.16.1"
    assert set(lock["commits"]) <= set(lock["versions"])
    head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    assert lock["platoon_git_sha"] == (head.stdout.strip() if head.returncode == 0 else None)
    assert lock["argv"] == ["platoon", "train", str(run), "count=3"]
    assert lock["hostname"] and lock["started_at"].endswith("+00:00")
    assert lock["deprecations"] == []


HPC_SANDBOX_COMMIT = "02f88b989855bfdeec95785787e22cc097c32c88"
# PEP 610 direct_url.json as pip and uv write it: for a git URL, a local directory (editable) and a broken record.
DIRECT_URLS = {
    "hpc_sandbox": {
        "url": "https://github.com/ApGa/hpc-sandbox.git",
        "vcs_info": {"vcs": "git", "commit_id": HPC_SANDBOX_COMMIT, "requested_revision": HPC_SANDBOX_COMMIT},
    },
    "tinker_cookbook": {"url": "file:///src/tinker-cookbook", "dir_info": {"editable": True}},
    "tinker": "not a JSON object",
}


class _Installed:
    """An installed distribution as importlib.metadata finds it; ``direct_url.json`` from ``DIRECT_URLS``."""

    def __init__(self, name: str) -> None:
        self.name, self.version = name, "0.1.0"

    def read_text(self, filename: str) -> str | None:
        record = DIRECT_URLS.get(self.name)
        if filename != "direct_url.json" or record is None:
            return None
        return record if isinstance(record, str) else json.dumps(record)


def test_the_lock_records_the_commit_of_each_distribution_installed_from_git(monkeypatch):
    """The commits of hpc-sandbox between two releases share a version, so lock.json also records the commit that a
    git install's direct_url.json names; editable checkouts, index installs and broken records give none."""

    import importlib.metadata

    def distribution(name: str) -> _Installed:
        if name == "openreward":
            raise importlib.metadata.PackageNotFoundError(name)
        return _Installed(name)

    monkeypatch.setattr(importlib.metadata, "distribution", distribution)

    assert distribution_commit("hpc_sandbox") == HPC_SANDBOX_COMMIT
    assert [distribution_commit(name) for name in ("tinker_cookbook", "tinker", "platoon", "openreward")] == [None] * 4
    lock = run_lock(["platoon", "train"])
    assert lock["commits"] == {"hpc_sandbox": HPC_SANDBOX_COMMIT}
    assert lock["versions"]["hpc_sandbox"] == "0.1.0"


PLATOON_COMMIT = "ca0d3adea65a731660c609e969e9a7900c1b2a4f"  # archive/v0.1, the original platoon


class _InstalledPlatoon(_Installed):
    """platoon installed from a git URL into ``root`` (its site-packages), where a checkout may shadow it."""

    def __init__(self, root: Path) -> None:
        super().__init__("platoon")
        self.root = root

    def read_text(self, filename: str) -> str | None:
        record = {"url": "https://github.com/ApGa/platoon.git", "vcs_info": {"vcs": "git", "commit_id": PLATOON_COMMIT}}
        return json.dumps(record) if filename == "direct_url.json" else None

    def locate_file(self, path: str) -> Path:
        return self.root / path


def test_the_lock_records_platoons_commit_only_when_its_installed_copy_runs(monkeypatch, tmp_path: Path):
    """A checkout on the path shadows an installed platoon, and lock.json gives the checkout's version and git SHA
    (tests/test_packaging.py), so it records the installed copy's commit only when that copy is the platoon that
    runs."""

    import importlib.metadata

    import platoon

    def installed_in(root: Path):
        def distribution(name: str) -> _Installed:
            if name != "platoon":
                raise importlib.metadata.PackageNotFoundError(name)
            return _InstalledPlatoon(root)

        return distribution

    monkeypatch.setattr(importlib.metadata, "distribution", installed_in(tmp_path / "site-packages"))
    assert distribution_commit("platoon") is None
    assert run_lock(["platoon", "train"])["commits"] == {}

    running = Path(platoon.__file__).resolve().parent.parent  # where the imported platoon package lives
    monkeypatch.setattr(importlib.metadata, "distribution", installed_in(running))
    assert distribution_commit("platoon") == PLATOON_COMMIT
    assert run_lock(["platoon", "train"])["commits"] == {"platoon": PLATOON_COMMIT}


# --------------------------------------------------------------------------- the images a job pulled (hpc-sandbox 0.2)

# Lines as `python -m hpc_sandbox pull` prints them: a tag (no manifest digest) and a digest pin.
PULLED_TAG = {
    "reference": "docker://python:3.12-slim",
    "path": "/shared/cache/sha256-" + "a1" * 32 + ".sif",
    "sha256": "a1" * 32,
    "manifest_digest": None,
}
PULLED_PIN = {
    "reference": "docker://alpine@sha256:" + "b2" * 32,
    "path": "/shared/cache/sha256-" + "c3" * 32 + ".sif",
    "sha256": "c3" * 32,
    "manifest_digest": "sha256:" + "b2" * 32,
}


def test_the_lock_records_the_images_the_job_pulled(tmp_path, monkeypatch):
    """A job that ran `python -m hpc_sandbox pull ... > pulls.jsonl` names that file in $PLATOON_IMAGE_PULLS, and
    lock.json records its lines as they are, blank lines left out and keys hpc-sandbox may add kept."""

    pulls = tmp_path / "pulls.jsonl"
    later = {**PULLED_PIN, "added_by_a_later_release": 1}
    pulls.write_text(json.dumps(PULLED_TAG) + "\n\n" + json.dumps(later) + "\n")
    monkeypatch.setenv(IMAGE_PULLS_ENV, str(pulls))

    record = write_run_record(tmp_path / "run", {"count": 1}, argv=["platoon", "train"])

    assert json.loads(record.lock_path.read_text())["image_pulls"] == [PULLED_TAG, later]
    assert image_pulls(pulls) == [PULLED_TAG, later]


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
def test_a_job_that_names_no_pulls_records_none(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(IMAGE_PULLS_ENV, raising=False)
    else:
        monkeypatch.setenv(IMAGE_PULLS_ENV, value)

    assert image_pulls() is None
    assert "image_pulls" not in run_lock(["platoon", "train"])


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "holds no image pulls"),
        ("\n\n", "holds no image pulls"),
        ("pulling docker://python:3.12-slim\n", "line 1: expected a line of `python -m hpc_sandbox pull`"),
        (json.dumps([PULLED_TAG]), "line 1"),
        (json.dumps(PULLED_TAG) + "\n" + json.dumps({**PULLED_TAG, "sha256": "not-a-digest"}), "line 2"),
        (json.dumps({key: PULLED_TAG[key] for key in ("reference", "path", "sha256")}), "line 1"),
        (json.dumps({**PULLED_PIN, "manifest_digest": "b2" * 32}), "line 1"),
        (json.dumps({**PULLED_TAG, "reference": ""}), "line 1"),
    ],
    ids=["empty", "blank", "a-log-line", "a-list", "bad-sha256", "no-manifest-key", "bad-manifest", "no-reference"],
)
def test_a_bad_record_of_pulls_stops_the_run_before_anything_is_written(tmp_path, monkeypatch, content, message):
    pulls = tmp_path / "pulls.jsonl"
    pulls.write_text(content)
    monkeypatch.setenv(IMAGE_PULLS_ENV, str(pulls))

    with pytest.raises(ConfigError, match=re.escape(message)):
        write_run_record(tmp_path / "run", {"count": 1}, argv=["platoon", "train"])
    assert not (tmp_path / "run").exists()


def test_a_record_of_pulls_that_cannot_be_read_stops_the_run(tmp_path, monkeypatch):
    monkeypatch.setenv(IMAGE_PULLS_ENV, str(tmp_path / "missing.jsonl"))

    with pytest.raises(ConfigError, match=f"Cannot read the image pulls in {IMAGE_PULLS_ENV}="):
        run_lock(["platoon", "train"])
