from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "plugins/openreward/platoon/openreward/configs/areal"


def _compose(config_name: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name=config_name)


def test_mixed_ptc_tracker_full_config_is_untruncated_nonrecursive_and_no_eval():
    config = _compose(
        "toolathlon_tmax_swe_openhands_areal_prealloc_16node-cp-ptc-task-"
        "tracker-full-r3-fp32-lm-head"
    )

    environments = {
        environment.label: environment
        for environment in config.openreward.environments
    }
    assert list(environments) == ["toolathlon", "tmax", "swe_rebench"]
    for environment in environments.values():
        assert environment.train_task_limit is None
        assert environment.eval_task_limit is None
        assert environment.get("task_names") is None
        assert environment.get("task_indices") is None
        assert environment.sampling_weight == 1.0

    model_path = str(
        REPO_ROOT / ".cache/platoon-models/Qwen3.6-35B-A3B-preserve-thinking"
    )
    assert config.actor.path == model_path
    assert config.openreward.enable_programmatic_tool_calling is True
    assert config.openreward.enable_task_tracker is True
    assert config.openreward.enable_recursive_subagents is False
    assert config.openreward.balance_accepted_batches is False
    assert config.train_dataset.batch_size == 8
    assert config.rollout.consumer_batch_size == 8
    assert config.workflow_config.group_size == 8
    assert config.workflow_config.straggler_quorum == 6
    assert config.workflow_config.straggler_timeout_seconds == 900
    assert config.workflow_config.min_successful_group_size == 4
    assert config.valid_dataset is None
    assert config.evaluator.eval_before_train is False
    assert config.evaluator.freq_epochs is None
    assert config.evaluator.freq_steps is None
    assert config.evaluator.freq_secs is None
    assert config.trial_name.endswith(
        "ptc-task-tracker-full-r3-fp32-lm-head-trial0"
    )


def test_recursive_efficiency_toolathlon_catalog_is_untruncated():
    config = _compose(
        "toolathlon_openhands_areal_prealloc_32node-cp-ptc-recursive-judged-"
        "r3-fp32-lm-head-bs8-efficiency"
    )

    assert config.openreward.env_name == "toolathlongym"
    assert config.openreward.train_task_limit is None
    assert config.openreward.task_names is None
    assert OmegaConf.select(config, "openreward.task_indices") is None
