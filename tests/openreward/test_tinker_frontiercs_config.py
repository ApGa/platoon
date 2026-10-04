"""Frontier's algorithmic cutoff must not shorten native CPU grading or evaluation."""

from pathlib import Path

import yaml

from platoon.openreward.tinker_config import OpenRewardTinkerTrainerConfig
from platoon.train.tinker.config_defs import StragglerPolicy
from platoon.utils.config import load_config


CONFIG_DIR = Path(__file__).resolve().parents[2] / "plugins/openreward/platoon/openreward/configs/tinker"


def test_v25_frontier_cutoff_and_compute_filter_preserve_the_training_recipe():
    path = CONFIG_DIR / "abci_frontiercs_qwen38_10node_v25.yaml"
    previous = yaml.safe_load((CONFIG_DIR / "abci_frontiercs_qwen38_10node_v24.yaml").read_text())
    config, raw = load_config(["--config", str(path)], OpenRewardTinkerTrainerConfig)
    workflow = config.train.workflow_config
    assert workflow.straggler_policy_key == "_openreward_environment"
    assert workflow.straggler_policies == {"fcs_algorithmic": StragglerPolicy(timeout_seconds=900, quorum=6)}
    assert workflow.straggler_timeout_seconds is None
    assert workflow.straggler_quorum is None
    assert workflow.filter_zero_advantage_datums is True
    assert workflow.group_size == 8
    assert workflow.min_successful_group_size == 4
    assert config.eval.workflow_config.straggler_policies == {}
    assert config.eval.workflow_config.filter_zero_advantage_datums is False
    assert raw["eval"] == previous["eval"]
    assert config.train.renderer_kwargs["reasoning_effort"] == "medium"
    assert config.train.context_window_length == 131072
    assert workflow.rollout_config.inference_params.max_completion_tokens == 49152
    assert workflow.rollout_config.timeout == 50400
    assert workflow.rollout_config.step_timeout == 36000
    assert config.train.forward_backward_timeout_seconds == 43200
    assert config.train.rollout_group_limits == {"fcs_2_0": 2}
    # Only the requested compute filter, environment-specific cutoff, and run
    # identity differ from v24; allocation remains a launch-time choice.
    previous["train"]["workflow_config"].update(
        filter_zero_advantage_datums=True,
        straggler_policy_key="_openreward_environment",
        straggler_policies={"fcs_algorithmic": {"timeout_seconds": 900, "quorum": 6}},
    )
    previous["stats"]["trial_name"] = "frontiercs-q38-v25"
    previous["stats"]["wandb"]["group"] = "frontiercs-q38-base-v25"
    assert raw == previous
