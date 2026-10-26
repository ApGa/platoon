"""The number-search example (``examples/number-search``) on CPU, as ``docs/getting-started/first-run.md`` runs it.

Each test runs a tutorial command through ``platoon.cli.main``, or loads the example's configs.  A scripted policy
that bisects stands in for the Tinker sampler, behind the real gateway, renderer and tokenizer: ``platoon rollout``
runs the number-search ORS environment in process with the ``platoon-codeact`` harness, and its trajectory dump is
converted the way ``platoon viz show-dump`` converts it.  ``platoon train`` runs up to the trainer, which is stubbed.

The tokenizer is ``platoon.testing``'s toy tokenizer, which speaks the Qwen3 chat format the config's renderer
renders, so no model download is needed.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OPENREWARD_DISABLE_UPDATE_CHECK", "1")

REPO = Path(__file__).resolve().parents[2]
# inline_fakes (PolicySampler) and the gateway_fakes it imports.
for fakes in ("gateway", "inline"):
    if str(REPO / "tests" / fakes) not in sys.path:
        sys.path.insert(0, str(REPO / "tests" / fakes))

from inline_fakes import PolicySampler  # noqa: E402

from platoon import cli  # noqa: E402
from platoon.gateway import ModelGateway, set_default_gateway  # noqa: E402
from platoon.testing import toy_tokenizer  # noqa: E402

EXAMPLE = REPO / "examples" / "number-search"
CONFIG = EXAMPLE / "train.yaml"
EVAL_CONFIG = EXAMPLE / "eval.yaml"
REACT_CONFIG = EXAMPLE / "train_react.yaml"
NEMOTRON_CONFIG = EXAMPLE / "train_nemotron.yaml"
TRAIN_CONFIGS = (CONFIG, REACT_CONFIG, NEMOTRON_CONFIG)
SERVICE = "python://platoon.envs.number_search.env:NumberSearchEnvironment"


class Bisection:
    """A scripted number-search policy: bisect the range the task names, narrowed by the env's replies."""

    def __init__(self) -> None:
        self.low = self.high = self.guess = 0

    def __call__(self, message: str) -> str:
        task = re.search(r"between (\d+) and (\d+)", message)
        if task:
            self.low, self.high = map(int, task.groups())
        elif "Too low" in message:
            self.low = self.guess + 1
        elif "Too high" in message:
            self.high = self.guess - 1
        self.guess = (self.low + self.high) // 2
        return f"<python>guess({self.guess})</python>"


SAMPLERS: list[PolicySampler] = []


@dataclass(frozen=True)
class ScriptedPolicy:
    """Stands in for ``platoon.eval.EvalPolicy``: ``install`` serves the bisection policy instead of Tinker."""

    model_name: str
    renderer_name: str
    model_path: str | None = None
    renderer_kwargs: dict = field(default_factory=dict)
    context_window_length: int | None = None

    def install(self) -> None:
        from tinker_cookbook.renderers import get_renderer

        tokenizer = toy_tokenizer()
        sampler = PolicySampler(tokenizer, Bisection())
        SAMPLERS.append(sampler)
        renderer = get_renderer(self.renderer_name, tokenizer)
        set_default_gateway(
            ModelGateway(sampler=sampler, renderer=renderer, tokenizer=tokenizer, model_name=self.model_name)
        )


@pytest.fixture
def scripted_policy(monkeypatch):
    SAMPLERS.clear()
    # platoon rollout builds platoon.eval.EvalPolicy; platoon eval builds it from its config (EvalConfig.policy).
    monkeypatch.setattr(sys.modules[cli._evaluation().__package__], "EvalPolicy", ScriptedPolicy, raising=False)
    monkeypatch.setattr("platoon.eval.config_defs.EvalPolicy", ScriptedPolicy)
    yield SAMPLERS
    set_default_gateway(None)


def test_the_example_configs_are_valid(capsys):
    cli.main(["config", "validate", str(CONFIG), str(EVAL_CONFIG), str(REACT_CONFIG), str(NEMOTRON_CONFIG)])

    assert capsys.readouterr().out == (
        f"ok   {CONFIG} (PlatoonTinkerRLTrainerConfig)\nok   {EVAL_CONFIG} (PlatoonEvalConfig)\n"
        f"ok   {REACT_CONFIG} (PlatoonTinkerRLTrainerConfig)\nok   {NEMOTRON_CONFIG} (PlatoonTinkerRLTrainerConfig)\n"
    )


def _load(path: Path, config_class: type | None = None):
    from platoon.train.auto import AutoEnvironment
    from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
    from platoon.utils.config import load_config

    config, _ = load_config(["--config", str(path)], config_class or PlatoonTinkerRLTrainerConfig)
    AutoEnvironment.load(config)
    return config


@pytest.mark.parametrize("path", TRAIN_CONFIGS, ids=lambda path: path.name)
def test_every_training_config_runs_a_function_harness_on_the_environment(path):
    from platoon.harness.builtin.codeact import CodeActOptions
    from platoon.registry import import_from_string
    from platoon.train.auto import AutoRollout
    from platoon.train.components import HARNESS_ROLLOUT

    config = _load(path)

    (environment,) = config.environments
    assert environment.service == SERVICE and environment.harness in {"platoon-codeact", "platoon-react"}
    assert AutoRollout.from_config(config) is import_from_string(HARNESS_ROLLOUT)
    # 512-token completions, set explicitly: an unset budget would leave the gateway's default (4096).
    for workflow in (config.train.workflow_config, config.eval.workflow_config):
        assert workflow.rollout_config.inference_params.max_completion_tokens == 512
    if environment.harness == "platoon-codeact":
        # Valid platoon-codeact options, which name an importable prompt builder.
        CodeActOptions.from_mapping(environment.harness_options).make_prompt_builder()


def test_the_react_config_trains_as_the_codeact_one_does():
    from platoon.eval.run import PlatoonEvalConfig

    config, react = _load(CONFIG), _load(REACT_CONFIG)

    assert (config.train.model_name, config.train.renderer_name, config.train.lora_rank) == (
        "Qwen/Qwen3-4B-Instruct-2507",
        "qwen3_instruct",
        32,
    )
    assert react.environments[0].harness == "platoon-react" and react.train == config.train
    assert react.environments[0].workflow_kwargs == config.environments[0].workflow_kwargs == {"filter_errors": False}
    assert (react.stats.experiment_name, react.stats.trial_name) == ("number-search", "first-run-react")
    nemotron = _load(NEMOTRON_CONFIG)
    assert (nemotron.train.model_name, nemotron.train.renderer_name) == (
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "nemotron3_disable_thinking",
    )
    evaluation = _load(EVAL_CONFIG, PlatoonEvalConfig)
    assert evaluation.inference.policy().renderer_name == "qwen3_instruct"
    assert evaluation.inference.workflow.rollout_config.inference_params.max_completion_tokens == 512


def test_rollout_solves_one_task_and_viz_reads_its_dump(tmp_path, capsys, monkeypatch, scripted_policy):
    run_dir = tmp_path / "first-rollout"

    cli.main(["rollout", str(CONFIG), "--task", "val:0", "--output-dir", str(run_dir)])

    out = capsys.readouterr().out
    (sampler,) = scripted_policy
    steps = len(sampler.calls)
    assert out.startswith("rollout val-0-")
    assert f"agents 1  steps {steps}  max depth 0  reward 1.0  outcome scored reward=1.0" in out
    assert re.search(rf"└─ policy [0-9a-f]{{8}}  steps {steps}  reward 1.0  outcome scored reward=1.0", out)
    assert 1 < steps <= 10  # val:0 is "between 119 and 965", target 447: bisection needs a few guesses
    for name in ("config.resolved.yaml", "lock.json", "trajectory_collection.json"):
        assert (run_dir / name).is_file()

    # platoon viz show-dump converts the dump into viewer events before it opens the viewer.
    from platoon.visualization.event_sinks import trajectory_collection_dump_to_events

    dump = json.loads((run_dir / "trajectory_collection.json").read_text())
    (trajectory,) = dump["trajectories"].values()
    assert trajectory["steps"][0]["misc"]["code"].startswith("guess(")
    assert trajectory["misc"]["harness"]["spec"] in ("platoon-codeact@1", "codeact@1")
    assert len(list(trajectory_collection_dump_to_events(dump))) > steps


def _named(config: Path, path: Path) -> Path:
    """The example config with its environment named (``env: platoon-number-search@1``) instead of spelled out."""

    data = yaml.safe_load(config.read_text())
    data["environments"] = [
        {
            "env": "platoon-number-search@1",
            "harness": "platoon-codeact",
            "dataset_kwargs": {"split_name": "train", "limit": 1000},
            "eval_dataset_kwargs": {"limit": 20},
            "workflow_kwargs": {"filter_errors": False},
        }
    ]
    path.write_text(yaml.safe_dump(data))
    return path


def test_the_named_environment_is_the_examples_entry(tmp_path):
    """The built-in ``platoon-number-search@1`` gives the entry the example spells out."""

    from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
    from platoon.utils.config import resolve_config

    named = resolve_config(_named(CONFIG, tmp_path / "named.yaml"), [], PlatoonTinkerRLTrainerConfig)

    (spelled_out,) = yaml.safe_load(CONFIG.read_text())["environments"]
    (entry,) = named.data["environments"]
    assert entry.pop("env") == "platoon-number-search@1"
    assert entry == spelled_out


def test_a_rollout_of_the_named_environment_records_the_built_in(tmp_path, capsys, scripted_policy):
    run_dir = tmp_path / "named-rollout"

    config = _named(CONFIG, tmp_path / "named.yaml")
    cli.main(["rollout", str(config), "--task", "val:0", "--output-dir", str(run_dir)])

    assert "reward 1.0  outcome scored reward=1.0" in capsys.readouterr().out
    lock = json.loads((run_dir / "lock.json").read_text())
    components = {component["kind"]: component for component in lock["components"]}
    environment = components["environment"]
    assert (environment["resolved"], environment["distribution"], environment["entry_point"]) == (
        "platoon-number-search@1",
        "platoon",
        None,
    )
    assert environment["source"] == "platoon.envs.number_search.entries:NUMBER_SEARCH"
    assert (components["service"]["distribution"], components["harness"]["distribution"]) == ("platoon", "platoon")
    assert "platoon-number-search" not in lock["extensions"]


def test_train_snapshots_the_config_and_hands_it_to_the_trainer(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr("platoon.train.launch.run_tinker_training", calls.append)
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")  # train checks its key before anything else: past it

    cli.main(["train", str(CONFIG), f"log_path={tmp_path}", "train.max_training_steps=1"])

    run_dir = tmp_path / "number-search" / "first-run"
    assert calls == [["--config", str(run_dir / "config.resolved.yaml")]]
    snapshot = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    assert snapshot["train"]["max_training_steps"] == 1
    assert snapshot["environments"][0]["service"] == SERVICE


def test_train_lists_the_configured_tasks():
    from platoon.train.auto import AutoDataset
    from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
    from platoon.utils.config import load_config

    config, _ = load_config(args=["--config", str(CONFIG)], config_class=PlatoonTinkerRLTrainerConfig)

    train, evaluation = AutoDataset.from_config(config, "train"), AutoDataset.from_config(config, "eval")
    assert (len(train), len(evaluation)) == (1000, 20)
    assert evaluation[0]["task_id"] == "val:0"


def test_eval_reports_the_success_rate(tmp_path, capsys, monkeypatch, scripted_policy):
    overrides = [
        f"inference.output_dir={tmp_path}",
        "inference.workflow.num_concurrent_workers=1",  # the scripted policy plays one episode at a time
        "environments.0.eval_dataset_kwargs={split_name: val, limit: 2}",
    ]

    cli.main(["eval", str(EVAL_CONFIG), *overrides])

    summary = json.loads(capsys.readouterr().out)
    assert (summary["total_tasks"], summary["total_rollouts"], summary["success_rate"]) == (2, 2, 1.0)
    assert (tmp_path / "reports").is_dir()
