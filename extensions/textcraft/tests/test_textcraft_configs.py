"""The TextCraft extension's configs (``configs/``): the training config runs a function harness on the ORS
environment, the evaluation configs load, and the TextCraft-Synth one samples the easy, medium and hard val tasks."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest
from platoon.harness.builtin.codeact import CodeActOptions
from platoon.registry import import_from_string
from platoon.train.auto import AutoDataset, AutoEnvironment, AutoRollout
from platoon.train.components import HARNESS_ROLLOUT
from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.utils.config import load_config

from platoon_textcraft import env
from platoon_textcraft.prompts import TextCraftPromptBuilder

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EVAL_CONFIGS = (CONFIGS / "textcraft_eval.yaml", CONFIGS / "textcraft_synth_eval.yaml")


def _load(path: Path, config_class: type = PlatoonTinkerRLTrainerConfig) -> tuple[Any, dict[str, Any]]:
    config, data = load_config(["--config", str(path)], config_class)
    AutoEnvironment.load(config)
    return config, data


def test_the_training_config_runs_a_function_harness_on_the_environment():
    config, data = _load(CONFIGS / "textcraft_train.yaml")
    (environment,) = config.environments

    assert environment.env == "textcraft@1"
    assert environment.service == "python://platoon_textcraft.env:TextCraftEnvironment"
    assert environment.harness == "platoon-codeact" and environment.program == "single"
    assert AutoRollout.from_config(config) is import_from_string(HARNESS_ROLLOUT)
    # 512-token completions, set explicitly: an unset budget would leave the gateway's default (4096).
    for workflow in (config.train.workflow_config, config.eval.workflow_config):
        assert workflow.rollout_config.inference_params.max_completion_tokens == 512
    builder = CodeActOptions.from_mapping(environment.harness_options).make_prompt_builder()
    assert isinstance(builder, TextCraftPromptBuilder)
    assert (builder.variant, builder.synth) == ("recursive", False)
    assert AutoDataset.from_config(config, "eval")["task_id"] == [f"val:{index}" for index in range(100)]


@pytest.mark.parametrize("path", EVAL_CONFIGS, ids=lambda path: path.name)
def test_the_evaluation_configs_load(path):
    from platoon.eval.run import PlatoonEvalConfig

    config, _ = _load(path, PlatoonEvalConfig)

    assert config.inference.policy().renderer_name == "qwen3_instruct"
    assert config.inference.workflow.rollout_config.inference_params.max_completion_tokens == 512
    (environment,) = config.environments
    CodeActOptions.from_mapping(environment.harness_options).make_prompt_builder()


def test_the_synth_eval_samples_the_easy_medium_and_hard_val_tasks():
    """``textcraft_synth_eval.yaml`` evaluates the easy, medium and hard val tasks in file order, shuffled with
    ``random.Random(42)``, first 100 (no extreme tasks): the sample of the RAO paper's inference runs."""

    from platoon.eval.run import PlatoonEvalConfig

    config, _ = _load(CONFIGS / "textcraft_synth_eval.yaml", PlatoonEvalConfig)
    task_ids = AutoDataset.from_config(config, config.split)["task_id"]
    lines = (env.DATA_DIR / "textcraft_synth_val.jsonl").read_text().splitlines()
    levels = ("easy", "medium", "hard")
    by_level = {
        level: [i for i, line in enumerate(lines) if json.loads(line)["misc"]["difficulty"] == level]
        for level in levels
    }
    expected = [index for level in levels for index in by_level[level]]
    random.Random(42).shuffle(expected)

    drawn = [by_level[split.removeprefix("val-")][int(index)] for split, index in (i.rsplit(":", 1) for i in task_ids)]
    assert drawn == expected[:100]
    assert drawn[:5] == [533, 592, 159, 521, 208]  # the first lines that sample drew
