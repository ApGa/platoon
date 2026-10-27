"""The keys of group admission (RFC 0019, section 6): ``train.group_admission`` and ``train.group_admission_options``,
``train.refill.max_groups_per_step``, and each environment entry's ``group_admission`` and ``group_admission_options``.
Which combinations load and which are refused, each with its message, when the config loads (in the file's one report)
or where the rules resolve (``run_components``, as ``platoon train`` and ``platoon config validate`` run it); the hints
for the design's keys this release leaves out; ``lock.json``'s records; the sampling fingerprint's ``admission`` key
and what a resume names of its changes; the plan's line; ``platoon config show``'s line; and the warning for a loss
whose objective is not zero at zero advantages."""

from __future__ import annotations

import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from platoon import cli
from platoon import registry as registry_module
from platoon.extensions import run_components
from platoon.train import entry_settings
from platoon.train.admission_api import REGISTRY_KIND
from platoon.train.admissions import AdmitAll, Informative, choose_admissions, plan_line, rule_mode
from platoon.train.auto import AutoDataset
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.sampling import fingerprint_changes, sampling_fingerprint
from platoon.utils.config import ConfigError, resolve_config

HERE = __name__


def tasks(config: Any, split: str, *, prefix: str = "t", count: int = 6) -> list[dict[str, Any]]:
    return [{"task_id": f"{prefix}{index}"} for index in range(count)]


def _entry(name: str, **fields: Any) -> dict[str, Any]:
    return {
        "name": name,
        "dataset_loader": f"{HERE}:tasks",
        "dataset_kwargs": {"prefix": f"{name}-"},
        "rollout": f"{HERE}:tasks",
        **fields,
    }


def _write(tmp_path: Path, train: dict[str, Any], environments: list[dict[str, Any]]) -> Path:
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "train": {"model_name": "m", "renderer_name": "r", "batch_size": 4, "max_training_steps": 3, **train},
                "eval": {"every": 0},
                "watchdog": {"enabled": False},
                "log_path": str(tmp_path / "logs"),
                "environments": environments,
            }
        )
    )
    return path


def _load(tmp_path: Path, train: dict[str, Any], environments: list[dict[str, Any]]) -> Any:
    return resolve_config(_write(tmp_path, train, environments), [], PlatoonTinkerRLTrainerConfig).config


def _refused(tmp_path: Path, train: dict[str, Any], environments: list[dict[str, Any]]) -> list[str]:
    with pytest.raises(ConfigError) as error:
        _load(tmp_path, train, environments)
    return error.value.problems


def _resolution(tmp_path: Path, train: dict[str, Any], environments: list[dict[str, Any]]) -> list[str]:
    config = _load(tmp_path, train, environments)
    with pytest.raises(ConfigError) as error:
        run_components(config)
    return error.value.problems


@pytest.fixture(autouse=True)
def _registry_left_as_found() -> Iterator[None]:
    registry = registry_module.get_registry(REGISTRY_KIND)
    saved = dict(registry._items)
    modules = set(sys.modules)
    yield
    registry._items = saved
    for name in set(sys.modules) - modules:
        if name.startswith("acme_"):
            del sys.modules[name]


# --- what loads ----------------------------------------------------------------------------------------------------


def test_the_keys_default_to_all_mode_and_load_at_both_levels(tmp_path):
    plain = _load(tmp_path, {}, [_entry("a")])
    assert plain.train.group_admission == "all" and plain.train.group_admission_options == {}
    assert plain.train.refill.max_groups_per_step == 3.0
    assert plain.environments[0].group_admission is None and not rule_mode(plain)
    mixed = _load(
        tmp_path,
        {
            "group_admission": "informative",
            "group_admission_options": {"band": [0.2, 0.8]},
            "refill": {"max_groups_per_step": 8},
        },
        [_entry("hard"), _entry("easy", group_admission="all")],
    )
    choices = choose_admissions(mixed)
    assert choices.entries["hard"].cls is Informative and choices.entries["hard"] is choices.run
    assert choices.entries["easy"].cls is AdmitAll and choices.own == {"easy"}
    assert choices.run.options.band == (0.2, 0.8) and choices.max_groups_per_step == 8.0
    assert choices.active and choices.qualified("easy") == "all@easy" and choices.qualified("hard") == "informative"
    every_all = choose_admissions(_load(tmp_path, {"group_admission": "all"}, [_entry("a", group_admission="all")]))
    assert not every_all.active


@pytest.mark.parametrize(
    ("train", "entries", "line"),
    [
        (
            {"group_admission": "informativ"},
            [{}],
            "train.group_admission: 'informativ' is not one of 'all', 'informative', a name an environment's package "
            "registers, or module:Class",
        ),
        (
            {},
            [{"group_admission": "informatve"}],
            "environments.0.group_admission: 'informatve' is not one of 'all', 'informative', a name an environment's "
            "package registers, or module:Class",
        ),
        (
            {"group_admission_options": {"band": [0.2, 0.8]}},
            [{}],
            "train.group_admission_options: all takes no options; leave group_admission_options out",
        ),
        (
            {"group_admission": "informative", "group_admission_options": {"bands": [0.2, 0.8]}},
            [{}],
            "train.group_admission_options.bands: unknown key; did you mean",
        ),
        (
            {},
            [{"group_admission": "informative", "group_admission_options": {"band": [0.8, 0.2]}}],
            "environments.0.group_admission_options.band: must be two numbers lo < hi between 0 and 1, got [0.8, 0.2]",
        ),
        (
            {},
            [{"group_admission_options": {"band": [0.2, 0.8]}}],
            "environments.0.group_admission_options: an entry's options go with the rule it sets: set "
            "environments.0.group_admission beside them (the entry inherits train.group_admission and its options "
            "otherwise)",
        ),
        (
            {"refill": {"max_groups_per_step": 0.5}},
            [{}],
            "train.refill.max_groups_per_step: must be a finite number of at least 1 (a step under a group admission "
            "rule counts at most this many times train.batch_size groups), got 0.5",
        ),
        ({"refill": {"max_groups_per_step": None}}, [{}], "train.refill.max_groups_per_step: null is not allowed"),
        (
            {"refill": {"when_short": "admit_rejected"}},
            [{}],
            "train.refill.when_short: unknown key; not in this release: a short step trains the groups it admitted",
        ),
        (
            {"on_stale": "requeue"},
            [{}],
            "train.on_stale: unknown key; not in this release: a stale group is dropped, and under a group admission "
            "rule the step takes another in its place",
        ),
        (
            {"num_minibatches": 2},
            [{"group_admission": "informative"}],
            "train.num_minibatches: 2 takes 2 optimizer steps a batch, and a group admission rule "
            "(environments[a].group_admission: informative) makes short steps: the last minibatch of a short step "
            "would take a full optimizer step on a few groups, and after its first update a step cannot be interrupted "
            "until it has counted its groups; set num_minibatches: 1 (num_microbatches may stay)",
        ),
    ],
)
def test_what_loading_refuses(tmp_path, train, entries, line):
    problems = _refused(tmp_path, train, [_entry("a", **entry) for entry in entries])
    assert any(problem.startswith(line) for problem in problems), problems


def test_num_minibatches_loads_beside_all_and_num_microbatches_beside_a_rule(tmp_path):
    _load(tmp_path, {"num_minibatches": 2}, [_entry("a")])
    _load(tmp_path, {"num_minibatches": 2, "group_admission": "all"}, [_entry("a", group_admission="all")])
    _load(tmp_path, {"num_microbatches": 2, "group_admission": "informative"}, [_entry("a")])


def test_a_config_built_in_code_is_refused_as_a_loaded_one_is():
    from platoon.train.components import EnvironmentConfig
    from platoon.train.config_defs import TrainConfig

    with pytest.raises(ConfigError, match="train.num_minibatches: 2 takes 2 optimizer steps a batch"):
        PlatoonTinkerRLTrainerConfig(
            train=TrainConfig(model_name="m", renderer_name="r", num_minibatches=2, group_admission="informative"),
            log_path="logs",
        )
    with pytest.raises(ConfigError, match="'nonesuch' is not one of"):
        PlatoonTinkerRLTrainerConfig(
            train=TrainConfig(model_name="m", renderer_name="r"),
            environments=[EnvironmentConfig(group_admission="nonesuch")],
            log_path="logs",
        )
    with pytest.raises(ValueError, match="max_groups_per_step: must be a finite number of at least 1"):
        TrainConfig(model_name="m", renderer_name="r", refill={"max_groups_per_step": float("inf")})


@pytest.mark.parametrize("refill", [None, 3, "3.0"])
def test_a_train_section_built_in_code_takes_a_refill_section_or_its_mapping_only(refill):
    from platoon.train.config_defs import RefillConfig, TrainConfig

    with pytest.raises(ValueError, match="train.refill must be a mapping"):
        TrainConfig(model_name="m", renderer_name="r", refill=refill)
    built = TrainConfig(model_name="m", renderer_name="r", refill=RefillConfig(max_groups_per_step=4.0))
    assert built.refill.max_groups_per_step == 4.0


@pytest.mark.parametrize(
    ("train", "entry", "where"),
    [
        (
            {"group_admission": ["informative"], "group_admission_options": {"band": [0.2, 0.8]}},
            {},
            "train.group_admission",
        ),
        (
            {},
            {"group_admission": {"informative": None}, "group_admission_options": {"band": [0.2, 0.8]}},
            "environments",
        ),
    ],
)
def test_a_rule_named_by_something_other_than_a_string_is_refused_beside_options_and_nothing_raises(
    tmp_path, train, entry, where
):
    """The options' checks read the rule's name raw: a list or a mapping there (a YAML slip) is the name's own problem,
    reported with the rest, and ``platoon config validate`` says so and exits 78, without a traceback."""

    problems = _refused(tmp_path, train, [_entry("a", **entry)])
    assert any(problem.startswith(where) and "group_admission" in problem for problem in problems), problems
    with pytest.raises(SystemExit) as exited:
        cli.main(["config", "validate", str(_write(tmp_path, train, [_entry("a", **entry)]))])
    assert exited.value.code == 78


def test_a_rule_named_by_reference_is_a_rule_even_when_it_resolves_to_all(tmp_path):
    """Rule mode is decided by the names a config gives, at load and at the start alike: ``module:Class`` naming
    ``AdmitAll`` is a rule (the cap counts, lock.json and the fingerprint record it), so num_minibatches above 1 is
    refused beside it, as the sampling plan's line says."""

    reference = "platoon.train.admissions:AdmitAll"
    config = _load(tmp_path, {"group_admission": reference}, [_entry("a")])
    choices = choose_admissions(config)
    assert rule_mode(config) and choices.active and choices.run.cls is AdmitAll
    environments = config.environments
    assert plan_line(config.train, environments) is not None
    training = AutoDataset.training(config)
    assert "admission" in sampling_fingerprint(config, training.tasks, training.sampling, admission=choices)
    assert [record["ref"] for record in run_components(config) if record["kind"] == "group_admission"] == [reference]
    problems = _refused(tmp_path, {"group_admission": reference, "num_minibatches": 2}, [_entry("a")])
    assert any(problem.startswith("train.num_minibatches: 2 takes 2 optimizer steps") for problem in problems)


def test_a_named_environment_cannot_set_a_group_admission_rule():
    from platoon.envs.named import EnvironmentEntry

    for key, value in (("group_admission", "informative"), ("group_admission_options", {"band": [0.1, 0.9]})):
        with pytest.raises(ValueError, match="which a named environment may not set") as error:
            EnvironmentEntry("acme-thing", **{"service": "harbor", key: value})
        assert key in str(error.value) and "group admission rule" in str(error.value)


# --- resolution ----------------------------------------------------------------------------------------------------

PACKAGE = """
from platoon.experimental import Admission, AdmissionRule, group_admission


def refuse(options, ctx):
    if ctx.batch_size > 64:
        raise ValueError("batches larger than 64 are not supported")


@group_admission("acme-quota", check=refuse)
class Quota(AdmissionRule):
    def admit(self, group, stats):
        return Admission(True)


class NotARule:
    pass


class Workflow:
    pass
"""


def _package(tmp_path: Path, monkeypatch, name: str, source: str) -> str:
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(source))
    monkeypatch.syspath_prepend(str(tmp_path))
    return name


@pytest.mark.parametrize(
    ("train", "entry", "line"),
    [
        (
            {},
            {"group_admission": "acme-quote"},
            "environments[a].group_admission: unknown group admission rule 'acme-quote'; did you mean acme-quota? "
            "Core's: all and informative; a package an entry names registers others with @group_admission, or name "
            "one as module:Class",
        ),
        ({"group_admission": "acme_nowhere:Rule"}, {}, "train.group_admission: acme_nowhere:Rule does not import"),
        (
            {"group_admission": "PACKAGE:NotARule"},
            {},
            "train.group_admission: PACKAGE:NotARule is not a group admission rule: a class with admit(group, stats)",
        ),
        (
            {"batch_size": 128},
            {"group_admission": "acme-quota"},
            "environments[a].group_admission: acme-quota: batches larger than 64 are not supported",
        ),
        (
            {},
            {"group_admission": "informative", "workflow": "PACKAGE:Workflow"},
            "environments[a].group_admission: informative rejects the groups whose workflow's verdict says they train "
            "nothing; the run's workflow, PACKAGE:Workflow, gives no verdicts (it is no GroupRolloutWorkflow, and does "
            "not declare gives_verdicts = True), so informative would admit every group and change nothing but the "
            "step's accounting",
        ),
    ],
)
def test_what_resolution_refuses(tmp_path, monkeypatch, train, entry, line):
    package = _package(tmp_path, monkeypatch, "acme_rules_one", PACKAGE)
    train = {
        key: value.replace("PACKAGE", package) if isinstance(value, str) else value for key, value in train.items()
    }
    entry = {
        key: value.replace("PACKAGE", package) if isinstance(value, str) else value for key, value in entry.items()
    }
    problems = _resolution(tmp_path, train, [_entry("a", package=package, **entry)])
    assert any(problem.startswith(line.replace("PACKAGE", package)) for problem in problems), problems


def test_resolution_reads_whether_a_workflow_class_gives_verdicts_of_its_gives_verdicts_alone(tmp_path, monkeypatch):
    """``informative`` beside a workflow class that declares only ``gives_verdicts`` resolves; beside one that declares
    only another capability, it is refused."""

    flags = "\n".join(
        f"class Only_{flag}:\n    {flag} = True\n"
        for flag in ("gives_verdicts", "applies_entry_settings", "runs_rollouts_on_lease", "takes_start_limit")
    )
    package = _package(tmp_path, monkeypatch, "acme_rules_flags", PACKAGE + flags)

    def refused(workflow: str) -> bool:
        entry = _entry("a", package=package, group_admission="informative", workflow=f"{package}:{workflow}")
        config = _load(tmp_path, {}, [entry])
        try:
            run_components(config)
        except ConfigError as error:
            return any(" gives no verdicts " in problem for problem in error.problems)
        return False

    assert not refused("Only_gives_verdicts")
    assert refused("Only_applies_entry_settings") and refused("Only_runs_rollouts_on_lease")
    assert refused("Only_takes_start_limit")


def test_lock_json_records_each_rule_other_than_all_with_its_options_and_none_in_all_mode(tmp_path, monkeypatch):
    package = _package(tmp_path, monkeypatch, "acme_rules_two", PACKAGE)
    config = _load(
        tmp_path,
        {"group_admission": "informative"},
        [
            _entry("hard", package=package),
            _entry("easy", group_admission="all"),
            _entry("q", group_admission="acme-quota"),
        ],
    )
    records = [record for record in run_components(config) if record["kind"] == "group_admission"]
    assert [(record["ref"], record["level"], record["environments"], record["options"]) for record in records] == [
        ("informative", "run", ["hard"], {"train": {"band": None}}),
        ("acme-quota", "entry", ["q"], {"q": {}}),
    ]
    assert records[0]["resolved"] == "platoon.train.admissions:Informative" and records[0]["distribution"] == "platoon"
    plain = _load(tmp_path, {"group_admission": "all"}, [_entry("a", group_admission="all")])
    assert not [record for record in run_components(plain) if record["kind"] == "group_admission"]


# --- the fingerprint, the plan, config show, the loss warning ---------------------------------------------------------


def test_all_mode_keeps_its_fingerprint_and_each_change_of_admission_is_named(tmp_path):
    environments = [_entry("hard"), _entry("easy")]
    plain = _load(tmp_path, {}, environments)
    training = AutoDataset.training(plain)
    before = sampling_fingerprint(plain, training.tasks, training.sampling)
    written = _load(tmp_path, {"group_admission": "all"}, [_entry("hard", group_admission="all"), _entry("easy")])
    assert (
        sampling_fingerprint(written, training.tasks, training.sampling, admission=choose_admissions(written)) == before
    )
    assert "admission" not in before
    ruled = _load(
        tmp_path,
        {"refill": {"max_groups_per_step": 4.0}},
        [_entry("hard", group_admission="informative", group_admission_options={"band": [0.2, 0.8]}), _entry("easy")],
    )
    after = sampling_fingerprint(ruled, training.tasks, training.sampling, admission=choose_admissions(ruled))
    assert after["admission"] == {
        "max_groups_per_step": 4.0,
        "entries": {
            "hard": {"rule": "informative", "options": {"band": [0.2, 0.8]}},
            "easy": {"rule": "all", "options": {}},
        },
    }
    # Turned on or off: the rule's change alone (a run without a rule has no cap and its entries' options are all's).
    assert fingerprint_changes(before, after) == ["environments[hard].group_admission: all -> informative"]
    assert fingerprint_changes(after, before) == ["environments[hard].group_admission: informative -> all"]
    # Under a rule on both sides: the cap, and the options of a rule that stayed.
    moved = _load(
        tmp_path,
        {"refill": {"max_groups_per_step": 5.0}},
        [_entry("hard", group_admission="informative", group_admission_options={"band": [0.3, 0.7]}), _entry("easy")],
    )
    assert fingerprint_changes(
        after, sampling_fingerprint(moved, training.tasks, training.sampling, admission=choose_admissions(moved))
    ) == [
        "train.refill.max_groups_per_step: 4.0 -> 5.0",
        'environments[hard].group_admission_options: {"band": [0.2, 0.8]} -> {"band": [0.3, 0.7]}',
    ]
    other = _load(
        tmp_path,
        {"refill": {"max_groups_per_step": 4.0}},
        [_entry("hard", group_admission="platoon.train.admissions:AdmitAll"), _entry("easy")],
    )
    assert fingerprint_changes(
        after, sampling_fingerprint(other, training.tasks, training.sampling, admission=choose_admissions(other))
    ) == ["environments[hard].group_admission: informative -> platoon.train.admissions:AdmitAll"]
    defaults = _load(tmp_path, {}, [_entry("hard", group_admission="informative"), _entry("easy")])
    no_band = _load(
        tmp_path,
        {},
        [_entry("hard", group_admission="informative", group_admission_options={"band": None}), _entry("easy")],
    )
    assert (
        fingerprint_changes(
            sampling_fingerprint(defaults, training.tasks, training.sampling, admission=choose_admissions(defaults)),
            sampling_fingerprint(no_band, training.tasks, training.sampling, admission=choose_admissions(no_band)),
        )
        == []
    )  # a default written out is no change


def test_the_plan_names_the_rules_and_what_a_step_trains(tmp_path):
    config = _load(tmp_path, {}, [_entry("hard", group_admission="informative"), _entry("nuclear")])
    lines = AutoDataset.training(config).plan.describe()
    assert lines[-1] == (
        "group admission (experimental): informative for hard, all for nuclear; a step trains the first 4 admitted "
        "groups of at most 12 (refill.max_groups_per_step 3.0)"
    )
    plain = _load(tmp_path, {}, [_entry("a")])
    assert not any("group admission" in line for line in AutoDataset.training(plain).plan.describe())
    single = _load(tmp_path, {"group_admission": "informative"}, [_entry("a")])
    assert plan_line(single.train, single.environments).startswith("group admission (experimental): informative for a;")


def test_config_show_prints_each_entry_s_rule_and_its_layer(tmp_path):
    config = _load(
        tmp_path,
        {"group_admission": "informative"},
        [
            _entry("hard"),
            _entry("easy", group_admission="all"),
            _entry("band", group_admission="informative", group_admission_options={"band": [0.2, 0.8]}),
        ],
    )
    settings = entry_settings.run_settings(config)
    assert [
        line.split()[1:] for line in entry_settings.describe(config, "hard", settings) if "group_admission" in line
    ] == [["informative", "train.group_admission"]]
    assert any(line.split()[1:] == ["all", "entry"] for line in entry_settings.describe(config, "easy", settings))
    assert any(
        "group_admission: informative {band: [0.2, 0.8]}" in line and line.rstrip().endswith("entry")
        for line in entry_settings.describe(config, "band", settings)
    )
    assert entry_settings.sets_anything(config.environments[1]) and not entry_settings.sets_anything(
        config.environments[0]
    )
    plain = _load(tmp_path, {}, [_entry("a")])
    assert not any(
        "group_admission" in line for line in entry_settings.describe(plain, "a", entry_settings.run_settings(plain))
    )


def test_a_rule_under_a_loss_with_advantage_independent_terms_warns_when_the_config_is_validated(tmp_path, capsys):
    loss = {
        "loss_fn": "cispo",
        "loss_fn_config": {"clip_low_threshold": 0.0, "clip_high_threshold": 5.0, "kl_coef": 0.1},
    }
    warned = _load(tmp_path, {**loss, "group_admission": "informative"}, [_entry("a")])
    [line] = [
        line for line in entry_settings.warnings(warned, entry_settings.run_settings(warned)) if "train.loss_fn" in line
    ]
    assert line.startswith(
        "train.loss_fn: cispo with loss_fn_config keys clip_high_threshold, clip_low_threshold, kl_coef may train "
        "terms that do not depend on the advantage"
    )
    assert "environments[a] (informative)" in line
    for train in ({"group_admission": "informative"}, loss):
        quiet = _load(tmp_path, train, [_entry("a")])
        assert not [
            line
            for line in entry_settings.warnings(quiet, entry_settings.run_settings(quiet))
            if "train.loss_fn" in line
        ]
    path = _write(tmp_path, {**loss, "group_admission": "informative"}, [_entry("a")])
    cli.main(["config", "validate", str(path)])  # it exits only when a config is invalid
    assert "warning: train.loss_fn: cispo with loss_fn_config keys" in capsys.readouterr().out


def test_config_show_with_defaults_says_what_a_run_does(tmp_path, capsys):
    path = _write(tmp_path, {}, [_entry("a")])
    cli.main(["config", "show", str(path), "--defaults"])
    shown = yaml.safe_load(capsys.readouterr().out)
    assert shown["train"]["group_admission"] == "all" and shown["train"]["group_admission_options"] == {}
    assert shown["train"]["refill"] == {"max_groups_per_step": 3.0}


def test_config_validate_names_an_entry_s_own_rule_when_it_is_not_the_run_s(tmp_path, capsys):
    path = _write(
        tmp_path,
        {"group_admission": "informative"},
        [
            _entry("hard"),
            _entry("same", group_admission="informative"),
            _entry("band", group_admission="informative", group_admission_options={"band": [0.2, 0.8]}),
            _entry("easy", group_admission="all"),
        ],
    )
    cli.main(["config", "validate", str(path)])
    shown = capsys.readouterr().out
    assert "environments[band]: group_admission informative {band: [0.2, 0.8]}" in shown
    assert "environments[easy]: group_admission all" in shown
    assert "environments[same]" not in shown and "environments[hard]" not in shown
