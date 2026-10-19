"""examples/config-layers: the layered configs validate, merge and take overrides as its README and
docs/how-to/configs.md say, and ``platoon rollout`` runs one task from them and writes the run record.

CPU only: the policy is a scripted binary search behind the real model gateway (``platoon.testing``'s toy
tokenizer under the config's renderer), installed where ``EvalPolicy`` would start a Tinker sampler.  The
environment is number search, which ships with platoon.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OPENREWARD_DISABLE_UPDATE_CHECK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gateway"))

from gateway_fakes import sample_response  # noqa: E402

from platoon import cli  # noqa: E402
from platoon.testing import toy_tokenizer  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "config-layers"
CONFIGS = ("environments.yaml", "train.yaml", "smoke.yaml", "eval.yaml")


def _exit_code(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    return exit_info.value.code


def _show(capsys, *argv: str) -> dict[str, Any]:
    cli.main(["config", "show", *argv])
    return yaml.safe_load(capsys.readouterr().out)


def test_every_config_validates(capsys):
    cli.main(["config", "validate", *(str(EXAMPLE / name) for name in CONFIGS)])

    out = capsys.readouterr().out
    assert f"ok   {EXAMPLE / 'environments.yaml'} (EnvironmentsConfig)" in out  # the entry schema
    assert f"ok   {EXAMPLE / 'train.yaml'} (PlatoonTinkerRLTrainerConfig)" in out
    assert f"ok   {EXAMPLE / 'smoke.yaml'} (PlatoonTinkerRLTrainerConfig)" in out
    assert f"ok   {EXAMPLE / 'eval.yaml'} (PlatoonEvalConfig)" in out


def test_extends_merges_mappings_and_keeps_the_shared_environments(capsys):
    smoke = _show(capsys, str(EXAMPLE / "smoke.yaml"))

    assert "extends" not in smoke
    assert smoke["environments"][0]["harness"] == "platoon-react"  # from environments.yaml, two files down
    assert smoke["train"]["batch_size"] == 2  # smoke.yaml wins
    assert smoke["train"]["model_name"] == "Qwen/Qwen3-4B-Instruct-2507"  # train.yaml's, merged key by key
    assert smoke["train"]["workflow_config"]["group_size"] == 2
    assert smoke["train"]["workflow_config"]["rollout_config"]["max_steps"] == 12
    assert smoke["stats"] == {
        "experiment_name": "number-search",
        "trial_name": "smoke",
        "wandb": {"mode": "disabled", "tags": ["smoke"]},
    }


def test_overrides_reach_into_lists_append_and_parse_by_field_type(capsys):
    shown = _show(
        capsys,
        str(EXAMPLE / "smoke.yaml"),
        "environments.0.dataset_kwargs.limit=10",
        "environments.-1.harness=platoon-bash",
        "stats.wandb.tags+=nightly",
        "train.optimizer.learning_rate=1e-5",
        "stats.trial_name=1",
        "train.loss_fn_config={clip_high_threshold: 4.0}",
    )

    environment = shown["environments"][0]
    assert environment["dataset_kwargs"] == {"split_name": "train", "limit": 10}  # a mapping key, merged
    assert environment["harness"] == "platoon-bash"  # negative indices count from the end
    assert shown["stats"]["wandb"]["tags"] == ["smoke", "nightly"]
    assert shown["train"]["optimizer"]["learning_rate"] == 1e-5
    assert shown["stats"]["trial_name"] == "1"  # a string field keeps the text
    assert shown["train"]["loss_fn_config"] == {"clip_high_threshold": 4.0}


def test_defaults_fill_in_every_schema_field(capsys):
    shown = _show(capsys, str(EXAMPLE / "smoke.yaml"), "--defaults")

    assert shown["train"]["workflow_config"]["rollout_isolation"] == "inprocess"
    assert shown["checkpoint"]["recovery_every"] is None


def test_an_unknown_key_is_a_configuration_error_with_a_suggestion(capsys):
    assert _exit_code(["config", "show", str(EXAMPLE / "smoke.yaml"), "train.batch_sise=4"]) == cli.EXIT_CONFIG

    err = capsys.readouterr().err
    assert "override train.batch_sise=4: train.batch_sise: unknown key; did you mean train.batch_size?" in err


def test_a_list_index_out_of_range_is_a_configuration_error(capsys):
    assert _exit_code(["config", "show", str(EXAMPLE / "smoke.yaml"), "environments.1.harness=x"]) == 78

    assert "environments has 1 entries; index 1 is out of range" in capsys.readouterr().err


def test_validate_reports_every_problem_of_a_broken_layer(tmp_path, capsys):
    broken = tmp_path / "broken.yaml"
    broken.write_text(f"extends: {EXAMPLE / 'smoke.yaml'}\ntrain:\n  learning_rate: 1.0e-5\n  task_order: random\n")

    assert _exit_code(["config", "validate", str(broken)]) == cli.EXIT_CONFIG

    out = capsys.readouterr().out
    assert f"FAIL {broken}" in out
    assert "  train.learning_rate: unknown key" in out
    assert "  train.task_order: 'random' is not one of 'auto', 'weighted'" in out


# --- platoon rollout with a scripted policy ----------------------------------------------------------------------

_RANGE = re.compile(r"between (\d+) and (\d+)")
_GUESS = re.compile(r'"number":\s*(\d+)')
_FEEDBACK = re.compile(r"Too (low|high)")


@dataclass
class BinarySearchSampler:
    """Plays number search by bisection, reading the range and its earlier guesses from the prompt."""

    tokenizer: Any
    replies: list[str] = field(default_factory=list)

    async def sample_async(self, *, prompt, sampling_params, num_samples):
        text = self.tokenizer.decode(prompt.to_ints())
        low, high = (int(value) for value in _RANGE.findall(text)[-1])
        for guess, feedback in zip(_GUESS.findall(text), _FEEDBACK.findall(text)):
            if feedback == "low":
                low = int(guess) + 1
            else:
                high = int(guess) - 1
        reply = f'<tool_call>\n{{"name": "guess", "arguments": {{"number": {(low + high) // 2}}}}}\n</tool_call>'
        self.replies.append(reply)
        tokens = self.tokenizer.encode(reply + "<|im_end|>", add_special_tokens=False)
        return sample_response(tokens)

    def sample(self, **kwargs):
        raise AssertionError("the gateway samples asynchronously")


INSTALLED: list[Any] = []


@dataclass(frozen=True)
class ScriptedPolicy:
    """Stands in for ``platoon.eval.EvalPolicy``: serves the binary search instead of a Tinker sampler."""

    model_name: str
    renderer_name: str
    model_path: str | None = None
    renderer_kwargs: dict = field(default_factory=dict)
    context_window_length: int | None = None

    def install(self) -> None:
        from tinker_cookbook.renderers import get_renderer

        from platoon.gateway import ModelGateway, set_default_gateway

        tokenizer = toy_tokenizer()
        sampler = BinarySearchSampler(tokenizer)
        renderer = get_renderer(self.renderer_name, tokenizer)
        set_default_gateway(
            ModelGateway(sampler=sampler, renderer=renderer, tokenizer=tokenizer, model_name=self.model_name)
        )
        INSTALLED.append((self, sampler))


@pytest.fixture
def scripted_policy(monkeypatch):
    from platoon.gateway import set_default_gateway

    INSTALLED.clear()
    monkeypatch.setattr(sys.modules[cli._evaluation().__package__], "EvalPolicy", ScriptedPolicy, raising=False)
    yield INSTALLED
    set_default_gateway(None)


def test_rollout_runs_one_task_from_the_layered_config_and_writes_the_run_record(tmp_path, capsys, scripted_policy):
    run_dir = tmp_path / "debug"

    cli.main(
        [
            "rollout",
            str(EXAMPLE / "smoke.yaml"),
            "--task",
            "train:0",
            "--output-dir",
            str(run_dir),
            "train.workflow_config.rollout_config.timeout=120",
        ]
    )

    out = capsys.readouterr().out
    assert "outcome scored reward=1.0" in out
    ((policy, sampler),) = scripted_policy
    assert (policy.model_name, policy.renderer_name) == ("Qwen/Qwen3-4B-Instruct-2507", "qwen3_instruct")
    assert 1 <= len(sampler.replies) <= 12

    record = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    assert record["train"]["batch_size"] == 2
    assert record["train"]["workflow_config"]["rollout_config"]["timeout"] == 120
    header = (run_dir / "config.resolved.yaml").read_text().splitlines()[:5]
    assert header[0].startswith("# Written by platoon")
    assert any(line.startswith("# source: ") and line.endswith("environments.yaml") for line in header)
    assert "# override: train.workflow_config.rollout_config.timeout=120" in header

    lock = json.loads((run_dir / "lock.json").read_text())
    assert set(lock) >= {"versions", "platoon_git_sha", "platoon_git_dirty", "python", "hostname", "started_at", "argv"}
    assert lock["argv"][:2] == ["platoon", "rollout"]

    collection = json.loads((run_dir / "trajectory_collection.json").read_text())
    (root,) = collection["trajectories"].values()
    assert root["reward"] == 1.0
