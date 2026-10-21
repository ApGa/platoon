"""The keys that choose a run's task samplers (RFC 0019, sections 1 and 5): ``train.task_sampler`` and
``train.task_sampler_options``, each environment entry's ``task_sampler`` and ``task_sampler_options``.  Which
combinations load and which are refused, each with its message, when the config loads (in the file's one report) or
where the samplers resolve (``run_components``, as ``platoon train`` and ``platoon config validate`` run it);
registering a sampler with ``@task_sampler`` from an entry's package, or naming one as ``module:Class``; core's names
reserved; typed options; ``lock.json``'s records; the plan's lines; and the sampling fingerprint's new keys."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from platoon import cli
from platoon import registry as registry_module
from platoon.experimental import Draw, Sampler, task_sampler
from platoon.extensions import ExtensionError, run_components
from platoon.train.auto import AutoDataset
from platoon.train.sampler_api import REGISTRY_KIND, resolve_task_sampler
from platoon.train.samplers import PassRateOptions, PassRateSampler, UniformSampler, WeightedSampler, choose_samplers
from platoon.train.sampling import fingerprint_changes, sampling_fingerprint
from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.utils.config import ConfigError, resolve_config

HERE = __name__
RUN_LINE = (
    "train.task_sampler: 'random' is not one of 'auto', 'weighted', 'uniform', 'pass_rate', a name an environment's "
    "package registers, or module:Class"
)


def tasks(config: Any, split: str, *, prefix: str = "t", count: int = 6, repeat: bool = False) -> list[dict[str, Any]]:
    rows = [{"task_id": f"{prefix}{index}"} for index in range(count)]
    return rows + rows[:2] if repeat else rows


def _entry(name: str, **fields: Any) -> dict[str, Any]:
    kwargs = {"prefix": f"{name}-", **fields.pop("dataset_kwargs", {})}
    return {
        "name": name,
        "dataset_loader": f"{HERE}:tasks",
        "dataset_kwargs": kwargs,
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
    """The task samplers a test registers are its own: put the registry back, and forget its packages."""

    registry = registry_module.get_registry(REGISTRY_KIND)
    saved = dict(registry._items)
    modules = set(sys.modules)
    yield
    registry._items = saved
    for name in set(sys.modules) - modules:
        if name.startswith("acme_"):
            del sys.modules[name]


# --- what loads ----------------------------------------------------------------------------------------------------


def test_an_entry_s_uniform_draws_what_leaving_it_out_draws(tmp_path):
    environments = [_entry("a", weight=2, seed=3), _entry("b", weight=1, seed=5)]
    unset = AutoDataset.training(_load(tmp_path, {}, environments))
    explicit = AutoDataset.training(
        _load(tmp_path, {}, [{**environments[0], "task_sampler": "uniform"}, environments[1]])
    )
    assert [row["task_id"] for row in unset.dataset] == [row["task_id"] for row in explicit.dataset]
    assert {label: choice.cls for label, choice in explicit.sampling.entries.items()} == {
        "a": UniformSampler,
        "b": UniformSampler,
    }


def test_an_entry_s_pass_rate_under_weighted_fills_that_entry_s_slots(tmp_path):
    config = _load(
        tmp_path, {}, [_entry("a", task_sampler="pass_rate", task_sampler_options={"explore": 0.5}), _entry("b")]
    )
    training = AutoDataset.training(config)
    assert training.sampling.run.cls is WeightedSampler
    choice = training.sampling.entries["a"]
    assert choice.cls is PassRateSampler and choice.options == PassRateOptions(explore=0.5)
    assert training.sampling.entries["b"].cls is UniformSampler


def test_a_single_environment_with_its_own_sampler_is_a_mixture_of_one(tmp_path):
    config = _load(tmp_path, {}, [_entry("a", task_sampler="pass_rate")])
    training = AutoDataset.training(config)
    assert training.sampling.run.cls is WeightedSampler and training.sampling.entries["a"].cls is PassRateSampler
    assert training.sampling.epoch_size == 8  # six tasks filled to whole batches of four


def test_a_pool_wide_pass_rate_decides_over_every_task(tmp_path):
    config = _load(tmp_path, {"task_sampler": "pass_rate"}, [_entry("a"), _entry("b")])
    training = AutoDataset.training(config)
    assert training.sampling.run.cls is PassRateSampler and training.sampling.entries == {}
    assert len(training.sampling.pool) == 12 and training.plan.describe()[0].startswith(
        "training tasks: 12, drawn by pass_rate (experimental) over every entry's tasks, in epochs of 12"
    )


# --- refused when the config loads -------------------------------------------------------------------------------


def test_an_unknown_name_without_a_package_is_refused_in_the_file_s_one_report(tmp_path):
    problems = _refused(tmp_path, {"task_sampler": "random", "learning_rate": 1.0}, [_entry("a", task_sampler="rand")])
    assert RUN_LINE in problems and "train.learning_rate: unknown key" in problems
    assert (
        "environments.0.task_sampler: 'rand' is not one of 'uniform', 'pass_rate', a name an environment's package "
        "registers, or module:Class"
    ) in problems


@pytest.mark.parametrize("name", ["auto", "weighted"])
def test_an_entry_cannot_name_a_run_s_sampler(tmp_path, name):
    assert _refused(tmp_path, {}, [_entry("a", task_sampler=name)]) == [
        f"environments.0.task_sampler: {name} shares a run's slots among its entries; an entry's sampler chooses "
        "among the entry's own tasks: uniform, pass_rate, a name a package registers, or module:Class"
    ]


@pytest.mark.parametrize(
    ("train", "entry", "line"),
    [
        ({"task_sampler": "weighted", "task_sampler_options": {"x": 1}}, {}, "train.task_sampler_options: weighted"),
        ({"task_sampler_options": {"x": 1}}, {}, "train.task_sampler_options: auto"),
        ({}, {"task_sampler_options": {"x": 1}}, "environments.0.task_sampler_options: uniform"),
    ],
)
def test_options_for_a_sampler_that_takes_none_are_refused(tmp_path, train, entry, line):
    leave = line.split(".")[-1].split(":")[0]
    assert _refused(tmp_path, train, [_entry("a", **entry)]) == [f"{line} takes no options; leave {leave} out"]


def test_pass_rate_s_options_are_checked_when_the_config_loads(tmp_path):
    problems = _refused(
        tmp_path,
        {"task_sampler": "pass_rate", "task_sampler_options": {"explor": 0.1, "candidates": 0}},
        [_entry("a", task_sampler_options={})],
    )
    assert problems == [
        "train.task_sampler_options.explor: unknown key; did you mean train.task_sampler_options.explore?"
    ]
    problems = _refused(tmp_path, {}, [_entry("a", task_sampler="pass_rate", task_sampler_options={"explore": 1.5})])
    assert problems == ["environments.0.task_sampler_options.explore: must be between 0 and 1, got 1.5"]
    problems = _refused(tmp_path, {}, [_entry("a", task_sampler="pass_rate", task_sampler_options={"candidates": 0})])
    assert problems == ["environments.0.task_sampler_options.candidates: must be an integer from 1, got 0"]
    loaded = _load(
        tmp_path,
        {},
        [_entry("a", task_sampler="pass_rate", task_sampler_options={"max_draws_per_epoch": None, "seed": 4})],
    )
    choice = choose_samplers(loaded).entries["a"]
    assert choice.options == PassRateOptions(max_draws_per_epoch=None, seed=4)


@pytest.mark.parametrize("run", ["uniform", "pass_rate"])
def test_a_run_sampler_that_hands_no_slots_refuses_entry_samplers_and_numeric_weights(tmp_path, run):
    hands_none = (
        f"train.task_sampler: {run} draws every training task of the run itself, with no entry slots, so it asks no "
        "entry's sampler; leave it out, or set train.task_sampler: weighted to give each entry its share of the slots "
        "and let its own sampler fill them"
    )
    problems = _refused(
        tmp_path,
        {"task_sampler": run},
        [
            _entry("a", task_sampler="pass_rate", task_sampler_options={"explore": 0.5}),
            _entry("b", task_sampler="uniform"),
        ],
    )
    assert problems == [
        f"environments[a].task_sampler: pass_rate: {hands_none}",
        f"environments[a].task_sampler_options: {hands_none}",
        f"environments[b].task_sampler: uniform: {hands_none}",
    ]
    weighed = _refused(tmp_path, {"task_sampler": run}, [_entry("a", weight=3), _entry("b", weight=1)])
    assert [problem.split(": train.task_sampler")[0] for problem in weighed] == [
        "environments[a].weight: 3",
        "environments[b].weight: 1",
    ]


def test_pass_rate_s_numeric_weight_message(tmp_path):
    assert _refused(tmp_path, {"task_sampler": "pass_rate"}, [_entry("a", weight=3)]) == [
        "environments[a].weight: 3: train.task_sampler: pass_rate chooses among every training task of the run by its "
        "statistics, with no entry's share, so it takes no weights; leave weight out, or keep train.task_sampler: "
        "weighted and set task_sampler: pass_rate on the entries, which keeps each entry's share"
    ]


def test_an_override_names_a_sampler_as_a_file_does(tmp_path):
    path = _write(tmp_path, {}, [_entry("a")])
    with pytest.raises(ConfigError) as error:
        resolve_config(path, ["train.task_sampler=random"], PlatoonTinkerRLTrainerConfig)
    assert RUN_LINE in error.value.problems


# --- refused where the samplers resolve ------------------------------------------------------------------------


def _package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, source: str) -> str:
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(source))
    monkeypatch.syspath_prepend(str(tmp_path))
    return name


PACKAGE = """
from dataclasses import dataclass

from platoon.experimental import Draw, Sampler, task_sampler


@dataclass(frozen=True)
class SkipOptions:
    keep: int = 2


def refuse_big_batches(options, ctx):
    if ctx.batch_size > 64:
        raise ValueError("batches larger than 64 are not supported")


@task_sampler("acme-skip", options=SkipOptions, check=refuse_big_batches)
class Skip(Sampler):
    def next(self, ctx):
        return Draw(ctx.task_ids[ctx.decision % len(ctx.task_ids)], "skip")


@task_sampler("acme-pool")
class Pool(Sampler):
    def next(self, ctx):
        return ctx.task_ids[0]


class Slots(Sampler):
    entry_slots = True

    def next(self, ctx):
        from platoon.experimental import Slot

        return Slot(next(iter(ctx.task_ids)))


class Learned(Slots):
    adaptive = True


def parse(values):
    if set(values) - {"level"}:
        raise ValueError(f"takes level only, got {sorted(values)}")
    return dict(values)


class Parsed(Skip):
    options_schema = staticmethod(parse)


class NotASampler:
    def next(self, ctx):
        return None
"""


def test_a_package_s_sampler_resolves_by_its_name_and_by_module_class_with_its_options_and_check(tmp_path, monkeypatch):
    package = _package(tmp_path, monkeypatch, "acme_samplers_one", PACKAGE)
    config = _load(
        tmp_path,
        {"task_sampler": "weighted"},
        [
            _entry("a", package=package, task_sampler="acme-skip", task_sampler_options={"keep": 3}),
            _entry("b", task_sampler=f"{package}:Skip"),
        ],
    )
    records = [record for record in run_components(config) if record["kind"] == "task_sampler"]  # imports the package
    choices = choose_samplers(config)
    assert choices.entries["a"].cls is choices.entries["b"].cls
    assert choices.entries["a"].options_json() == {"keep": 3} and choices.entries["b"].options_json() == {"keep": 2}
    run, *entries = records
    assert (run["ref"], run["level"], run["resolved"]) == ("weighted", "run", "platoon.train.samplers:WeightedSampler")
    assert run["options"] == {"train": {}} and run["distribution"] == "platoon"
    assert {record["ref"]: (record["level"], record["environments"], record["options"]) for record in entries} == {
        "acme-skip": ("entry", ["a"], {"a": {"keep": 3}}),
        f"{package}:Skip": ("entry", ["b"], {"b": {"keep": 2}}),
    }
    big = _load(tmp_path, {"batch_size": 128}, [_entry("a", package=package, task_sampler="acme-skip")])
    with pytest.raises(ConfigError) as error:
        run_components(big)
    assert error.value.problems == ["environments[a].task_sampler: acme-skip: batches larger than 64 are not supported"]


def test_pass_rate_s_record_holds_its_options_with_their_defaults(tmp_path):
    config = _load(tmp_path, {}, [_entry("hard", task_sampler="pass_rate", task_sampler_options={"explore": 0.1})])
    record = next(record for record in run_components(config) if record["ref"] == "pass_rate")
    assert record["level"] == "entry" and record["environments"] == ["hard"]
    assert record["options"] == {
        "hard": {
            "explore": 0.1,
            "max_draws_per_epoch": 3,
            "max_outstanding_per_task": None,
            "candidates": 4096,
            "seed": 0,
        }
    }
    assert record["resolved"] == "platoon.train.samplers:PassRateSampler"


@pytest.mark.parametrize(
    ("train", "entry", "line"),
    [
        (
            {},
            {"task_sampler": "pass-rate"},
            "environments[a].task_sampler: unknown task sampler 'pass-rate'; did you mean pass_rate? Core's: auto and "
            "weighted (run level), uniform and pass_rate (either level); a package an entry names registers others "
            "with @task_sampler, or name one as module:Class",
        ),
        ({"task_sampler": "acme_nowhere:Skip"}, {}, "train.task_sampler: acme_nowhere:Skip does not import"),
        (
            {"task_sampler": "PACKAGE:NotASampler"},
            {},
            "train.task_sampler: PACKAGE:NotASampler is not a task sampler: a class with next(ctx), state_dict() and "
            "load_state_dict(state), and adaptive set to True or False",
        ),
        (
            {"task_sampler": "PACKAGE:Pool"},
            {"task_sampler": "pass_rate"},
            "environments[a].task_sampler: pass_rate: train.task_sampler: PACKAGE:Pool hands no slots to entries (its "
            "entry_slots is False), so it asks no entry's sampler; leave it out",
        ),
        (
            {},
            {"task_sampler": "PACKAGE:Slots"},
            "environments[a].task_sampler: PACKAGE:Slots hands slots to entries, which only the run's sampler does",
        ),
        (
            {"task_sampler": "PACKAGE:Learned"},
            {},
            "train.task_sampler: PACKAGE:Learned hands slots to entries and is adaptive; in this release only a static "
            "run sampler (adaptive = False) hands slots, so set one of them to False",
        ),
        (
            {},
            {"task_sampler": "PACKAGE:Parsed", "task_sampler_options": {"depth": 2}},
            "environments.0.task_sampler_options: takes level only, got ['depth']",
        ),
    ],
)
def test_what_resolution_refuses(tmp_path, monkeypatch, train, entry, line):
    package = _package(tmp_path, monkeypatch, "acme_samplers_two", PACKAGE)
    train = {
        key: value.replace("PACKAGE", package) if isinstance(value, str) else value for key, value in train.items()
    }
    entry = {
        key: value.replace("PACKAGE", package) if isinstance(value, str) else value for key, value in entry.items()
    }
    line = line.replace("PACKAGE", package)
    problems = _resolution(tmp_path, train, [_entry("a", package=package, **entry)])
    assert any(problem.startswith(line) for problem in problems), problems


def test_pass_rate_refuses_a_group_of_one(tmp_path):
    single = {"workflow_config": {"group_size": 1}}
    assert _resolution(tmp_path, {}, [_entry("a", task_sampler="pass_rate", **single)]) == [
        "environments[a].task_sampler: pass_rate chooses the tasks whose next group is likely to mix passes and "
        "failures, which a group of one never does (its group size is 1); use uniform, or a group size of 2 or more"
    ]
    problems = _resolution(tmp_path, {"task_sampler": "pass_rate"}, [_entry("a"), _entry("b", **single)])
    assert problems == [
        "train.task_sampler: pass_rate chooses the tasks whose next group is likely to mix passes and failures, which "
        "a group of one never does (environments[b]'s group size is 1); use uniform, or a group size of 2 or more"
    ]


def test_a_named_environment_cannot_set_a_task_sampler():
    from platoon.envs.named import EnvironmentEntry

    for key, value in (("task_sampler", "pass_rate"), ("task_sampler_options", {"explore": 0.1})):
        with pytest.raises(ValueError, match="which a named environment may not set") as error:
            EnvironmentEntry("acme-thing", **{"service": "harbor", key: value})
        assert key in str(error.value) and "task sampler" in str(error.value)


def test_config_validate_exits_78_for_each_kind_of_refusal(tmp_path, monkeypatch, capsys):
    """An unknown entry sampler in the same report as an unknown key, when the config loads; options a built-in does
    not take; and a name a package does not register, where the samplers resolve."""

    loading = _write(tmp_path, {"learning_rate": 1.0}, [_entry("a", task_sampler="rand")])
    assert _exit_code(["config", "validate", str(loading)]) == 78
    out = capsys.readouterr().out
    assert "  train.learning_rate: unknown key" in out and "  environments.0.task_sampler: 'rand' is not one of" in out
    (tmp_path / "options").mkdir()
    options = _write(tmp_path / "options", {"task_sampler_options": {"explore": 0.1}}, [_entry("a")])
    assert _exit_code(["config", "validate", str(options)]) == 78
    assert "train.task_sampler_options: auto takes no options" in capsys.readouterr().out
    package = _package(tmp_path, monkeypatch, "acme_samplers_three", PACKAGE)
    (tmp_path / "resolving").mkdir()
    resolving = _write(tmp_path / "resolving", {}, [_entry("a", package=package, task_sampler="acme-skipp")])
    assert _exit_code(["config", "validate", str(resolving)]) == 78
    assert "unknown task sampler 'acme-skipp'; did you mean acme-skip?" in capsys.readouterr().out


def _exit_code(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    return exit_info.value.code


# --- registration ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["auto", "weighted", "uniform", "pass_rate"])
def test_core_s_names_are_reserved(name):
    with pytest.raises(ValueError, match=f"task sampler name '{name}' is core's"):
        task_sampler(name)


@pytest.mark.parametrize(
    ("name", "reads"), [("pass-rate", "pass_rate"), ("Uniform", "uniform"), ("PASS_RATE", "pass_rate")]
)
def test_a_name_that_reads_as_one_of_core_s_is_reserved_too(name, reads):
    """``pass-rate`` registered by a package would be what a config that misspells core's ``pass_rate`` resolves to."""

    with pytest.raises(ValueError, match=f"task sampler name '{name}' reads as '{reads}', which is core's"):
        task_sampler(name)


def test_core_s_names_are_reserved_whatever_was_imported_first():
    """A process that imports the sampler API alone, before core's samplers' module, cannot register core's names, and
    core's names resolve to core's classes."""

    code = textwrap.dedent(
        """
        import sys
        from platoon.train.sampler_api import task_sampler, resolve_task_sampler
        assert "platoon.train.samplers" not in sys.modules
        try:
            task_sampler("uniform")
        except ValueError as exc:
            print("refused:", exc)
        print(resolve_task_sampler("uniform").__module__)
        """
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "refused: task sampler name 'uniform' is core's" in done.stdout
    assert done.stdout.strip().endswith("platoon.train.samplers")


def test_a_name_registered_twice_is_an_error_and_core_s_resolve_to_core_s_classes():
    @task_sampler("acme-twice")
    class First(Sampler):
        def next(self, ctx):
            return Draw(ctx.task_ids[0])

    with pytest.raises(ValueError, match="already has an entry named 'acme-twice'"):
        task_sampler("acme-twice")(First)
    assert resolve_task_sampler("acme-twice") is First
    assert resolve_task_sampler("pass_rate") is PassRateSampler and resolve_task_sampler("weighted") is WeightedSampler
    with pytest.raises(ExtensionError, match="auto names no class"):
        resolve_task_sampler("auto")


@pytest.mark.parametrize("name", ["", "two words", "acme:x", "uniform@a"])
def test_a_name_must_be_a_word_without_a_colon_or_an_at(name):
    with pytest.raises(ValueError, match="a task sampler's name is a word"):
        task_sampler(name)


# --- options schemas --------------------------------------------------------------------------------------------


def test_options_schemas_parse_as_dataclasses_pydantic_models_and_parsers():
    from platoon.options import schema_options

    @dataclass(frozen=True)
    class Limits:
        most: int = 3

        def __post_init__(self) -> None:
            if self.most < 1:
                raise ValueError(f"most: must be at least 1, got {self.most}")

    dataclass_options = schema_options(Limits, name="acme")
    assert dataclass_options.parse({"most": 5}, "opts") == Limits(5)
    assert dataclass_options.option_problems({"mots": 5}, "opts") == ["opts.mots: unknown key; did you mean opts.most?"]
    assert dataclass_options.option_problems({"most": 0}, "opts") == ["opts.most: must be at least 1, got 0"]
    assert dataclass_options.option_problems({"most": "x"}, "opts") == ["opts.most: 'x' is not a int"]
    assert dataclass_options.as_json(Limits(4)) == {"most": 4}
    pydantic = pytest.importorskip("pydantic")

    class Model(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="forbid")
        rate: float = 0.5

    model_options = schema_options(Model, name="acme")
    assert model_options.parse({"rate": 0.25}, "opts") == Model(rate=0.25)
    assert model_options.option_problems({"rat": 1}, "opts")[0].startswith("opts.rat: ")
    assert model_options.as_json(Model()) == {"rate": 0.5}
    parser_options = schema_options(lambda values: dict(values), name="acme")
    assert parser_options.parse({"a": 1}, "opts") == {"a": 1} and parser_options.as_json({"a": 1}, {"a": 1}) == {"a": 1}
    assert schema_options(None, name="acme").option_problems({"a": 1}, "x.task_sampler_options") == [
        "x.task_sampler_options: acme takes no options; leave task_sampler_options out"
    ]


# --- the plan's lines, the repeated-ids warning, the fingerprint ---------------------------------------------


def test_the_plan_names_an_entry_s_sampler(tmp_path):
    config = _load(tmp_path, {}, [_entry("hard", task_sampler="pass_rate"), _entry("easy")])
    lines = AutoDataset.training(config).plan.describe()
    assert lines[1].startswith(
        "environments[hard]: 6 tasks, weight tasks, 50.0% of the slots, each filled by pass_rate (experimental) when "
        "its group can start: its tasks drawn about "
    )
    assert lines[2].startswith("environments[easy]: 6 tasks, weight tasks, 50.0% of the slots: each task drawn about")


def test_repeated_task_ids_under_a_sampler_that_draws_by_id_are_warned_of(tmp_path, caplog):
    config = _load(tmp_path, {}, [_entry("a", task_sampler="pass_rate", dataset_kwargs={"repeat": True})])
    AutoDataset.training(config)
    assert "environments[a]: task ids repeat (a-0 2 times, a-1 2 times): pass_rate draws by task id" in caplog.text


def test_a_0_3_0_config_keeps_its_fingerprint_and_each_new_key_s_change_is_named(tmp_path):
    environments = [_entry("a", weight=2, seed=3), _entry("b", weight=1, seed=5)]
    plain = _load(tmp_path, {}, environments)
    training = AutoDataset.training(plain)
    before = sampling_fingerprint(plain, training.tasks)
    assert sampling_fingerprint(plain, training.tasks, training.sampling) == before
    assert "task_sampler_options" not in before and all(
        "task_sampler" not in entry for entry in before["entries"].values()
    )
    changed = _load(
        tmp_path,
        {},
        [{**environments[0], "task_sampler": "pass_rate", "task_sampler_options": {"explore": 0.3}}, environments[1]],
    )
    after_training = AutoDataset.training(changed)
    after = sampling_fingerprint(changed, after_training.tasks, after_training.sampling)
    assert after["entries"]["a"]["task_sampler"] == "pass_rate"
    assert after["entries"]["a"]["task_sampler_options"]["explore"] == 0.3
    changes = fingerprint_changes(before, after)
    assert "environments[a].task_sampler: uniform -> pass_rate" in changes
    assert any(
        change.startswith('environments[a].task_sampler_options: {} -> {"candidates": 4096') for change in changes
    )
    pool = _load(tmp_path, {"task_sampler": "pass_rate"}, [_entry("a", seed=3), _entry("b", seed=5)])
    pool_training = AutoDataset.training(pool)
    pool_print = sampling_fingerprint(pool, pool_training.tasks, pool_training.sampling)
    assert pool_print["task_sampler_options"]["max_draws_per_epoch"] == 3
    assert any(
        change.startswith("train.task_sampler_options: {} -> ") for change in fingerprint_changes(before, pool_print)
    )
    # The same options written out as their defaults read as no change.
    defaults = _load(
        tmp_path,
        {"task_sampler": "pass_rate", "task_sampler_options": {"explore": 0.2}},
        [_entry("a", seed=3), _entry("b", seed=5)],
    )
    default_training = AutoDataset.training(defaults)
    assert (
        fingerprint_changes(
            pool_print, sampling_fingerprint(defaults, default_training.tasks, default_training.sampling)
        )
        == []
    )
