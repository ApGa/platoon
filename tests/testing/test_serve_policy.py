"""platoon.testing.serve_policy: a scripted policy served to harness_rollout, ``platoon rollout`` and ``platoon eval``
in this process, and everything restored afterwards."""

from __future__ import annotations

import json
import os

import yaml
from testing_toys import LOCKBOX

from platoon import cli
from platoon.eval.config_defs import EvalPolicy
from platoon.gateway import core
from platoon.testing import TOY_MODEL_NAME, ScriptedPolicy, serve_policy, tool_call
from platoon.testing.gateway import PLACEHOLDER_TINKER_API_KEY

SOLVE = [tool_call("unlock", code="1")]


def write_config(path, **inference) -> str:
    config = {
        "environments": [
            {
                "name": "lockbox",
                "service": LOCKBOX,
                "harness": "platoon-react",
                "eval_dataset_kwargs": {"split_name": "test"},
            }
        ],
        "inference": {
            "model_name": "my-org/my-model",
            "renderer_name": "qwen3_instruct",
            "output_dir": str(path.parent / "eval"),
            "workflow": {"num_rollouts_per_task": 1},
            **inference,
        },
    }
    path.write_text(yaml.safe_dump(config))
    return str(path)


def test_the_block_serves_the_policy_and_restores_what_it_replaced(monkeypatch):
    sentinel = object()
    install = EvalPolicy.__dict__["install"]
    monkeypatch.setattr(core, "_default_gateway", sentinel)
    monkeypatch.delenv("TINKER_API_KEY", raising=False)

    with serve_policy(SOLVE) as policy:
        assert isinstance(policy, ScriptedPolicy) and core.default_gateway().sampler is policy
        assert core.default_gateway().model_name == TOY_MODEL_NAME
        assert EvalPolicy.__dict__["install"] is not install
        assert os.environ["TINKER_API_KEY"] == PLACEHOLDER_TINKER_API_KEY

    assert core._default_gateway is sentinel and EvalPolicy.__dict__["install"] is install
    assert "TINKER_API_KEY" not in os.environ


def test_a_key_that_is_set_is_kept(monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "tml-mine")

    with serve_policy(SOLVE):
        assert os.environ["TINKER_API_KEY"] == "tml-mine"


def test_platoon_rollout_samples_the_policy_through_the_configs_renderer(tmp_path, capsys):
    config = write_config(tmp_path / "eval.yaml", renderer_kwargs={"strip_thinking_from_history": False})

    with serve_policy([tool_call("unlock", code="5"), *SOLVE]) as policy:
        cli.main(["rollout", config, "--task", "test:1", "--output-dir", str(tmp_path / "run"), "--json-output"])
        gateway = core.default_gateway()

    summary = json.loads(capsys.readouterr().out)
    assert summary["reward"] == 1.0 and summary["total_steps"] == 2 and len(policy.calls) == 2
    assert gateway.model_name == "my-org/my-model" and gateway.renderer.strip_thinking_from_history is False
    assert gateway.context_window_length is None
    assert (tmp_path / "run" / "trajectory_collection.json").is_file()


def test_platoon_eval_samples_the_policy(tmp_path, capsys):
    # The toy tokenizer's prompts are longer than a real model's, so the config's context window is left out.
    config = write_config(tmp_path / "eval.yaml", context_window_length=64)
    before = core._default_gateway

    with serve_policy(lambda prompt: tool_call("unlock", code=prompt.split("Its code is ")[1][0])) as policy:
        cli.main(["eval", config])

    summary = json.loads(capsys.readouterr().out)
    assert summary["success_rate"] == 1.0 and len(policy.calls) == 3  # one call per task
    assert core._default_gateway is before


def test_settings_and_a_renderer_override_the_config(tmp_path, capsys):
    config = write_config(tmp_path / "eval.yaml")

    with serve_policy(["<think>the code is 1</think>" + SOLVE[0]], renderer="qwen3", max_tokens=99) as policy:
        cli.main(["rollout", config, "--task", "test:1", "--output-dir", str(tmp_path / "run"), "--json-output"])
        renderer = core.default_gateway().renderer

    assert json.loads(capsys.readouterr().out)["reward"] == 1.0
    assert type(renderer).__name__ == "Qwen3Renderer" and policy.calls[0].sampling_params.max_tokens == 99


def test_the_configs_sample_timeout_applies_unless_settings_give_one(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("TINKER_SAMPLE_TIMEOUT_SECONDS", "300")
    config = write_config(tmp_path / "eval.yaml", sample_timeout_seconds=1800)
    bounds = []

    for settings in ({}, {"sample_timeout_seconds": 5}):
        with serve_policy(SOLVE, **settings):
            run_dir = tmp_path / f"run-{len(bounds)}"
            cli.main(["rollout", config, "--task", "test:1", "--output-dir", str(run_dir), "--json-output"])
            bounds.append(core.default_gateway().sample_timeout_seconds)

    assert bounds == [1800.0, 5.0]


def test_serving_twice_nests():
    with serve_policy(["outer"]) as outer:
        with serve_policy(["inner"]) as inner:
            assert core.default_gateway().sampler is inner
        assert core.default_gateway().sampler is outer
