from __future__ import annotations

import subprocess
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "plugins/openreward/platoon/openreward/configs/areal"
MIXED_16_LAUNCHER = (
    REPO_ROOT
    / "slurm-scripts/openreward-multienv-prealloc-16node-ptc-task-tracker-full.sh"
)
MIXED_32_LAUNCHER = (
    REPO_ROOT
    / "slurm-scripts/openreward-multienv-prealloc-32node-ptc-recursive-bs8-efficiency.sh"
)
MIXED_SERVER = REPO_ROOT / "slurm-scripts/openreward-multienv-server.sh"


def _compose(config_name: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name=config_name)


def _assert_wandb_identifiers_fit(config) -> None:
    experiment_name = config.experiment_name
    trial_name = config.trial_name
    name = OmegaConf.select(config, "stats_logger.wandb.name") or trial_name
    group = OmegaConf.select(config, "stats_logger.wandb.group") or f"{experiment_name}_{trial_name}"
    id_suffix = OmegaConf.select(config, "stats_logger.wandb.id_suffix") or "train"
    identifiers = {
        "name": name,
        "group": group,
        "id": f"{experiment_name}_{trial_name}_{id_suffix}",
    }
    assert {key: len(value) for key, value in identifiers.items() if len(value) > 128} == {}


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


def test_hardened_nonrecursive_config_starts_a_fresh_full_mixed_trial():
    config = _compose(
        "toolathlon_tmax_swe_openhands_areal_prealloc_16node-cp-ptc-task-"
        "tracker-full-r3-fp32-lm-head-hardened-v1"
    )

    assert config.trial_name == "mix16-ptc-full-hard-v5-trial0"
    assert config.stats_logger.wandb.group == "mix16-ptc-full-hard-v5"
    _assert_wandb_identifiers_fit(config)
    assert config.openreward.enable_recursive_subagents is False
    assert config.openreward.enable_programmatic_tool_calling is True
    assert config.openreward.enable_task_tracker is True
    assert config.train_dataset.batch_size == 8
    assert config.workflow_config.group_size == 8
    assert config.valid_dataset is None
    assert config.actor.path.endswith(
        ".cache/platoon-models/Qwen3.6-35B-A3B-preserve-thinking"
    )


def test_mixed_recursive_32node_config_combines_latest_proven_settings():
    config = _compose(
        "toolathlon_tmax_swe_openhands_areal_prealloc_32node-cp-ptc-recursive-"
        "judged-r3-fp32-lm-head-bs8-efficiency"
    )

    environments = {
        environment.label: environment
        for environment in config.openreward.environments
    }
    assert list(environments) == ["toolathlon", "tmax", "swe_rebench"]
    for environment in environments.values():
        assert environment.train_task_limit is None
        assert environment.eval_task_limit is None
        assert environment.sampling_weight == 1.0

    assert config.trial_name == "mix32-rec-eff-v5-trial0"
    assert config.cluster.n_nodes == 32
    assert config.actor.backend == "megatron:(attn:d10p2t4c2|ffn:d10p2t1e8)"
    assert config.rollout.backend == "sglang:d12p1t8"
    assert config.rollout.max_concurrent_rollouts == 12
    assert config.rollout.scheduling_spec[0].mem == 128
    assert config.sglang.mem_fraction_static == 0.70
    assert config.train_dataset.batch_size == 8
    assert config.workflow_config.group_size == 8

    assert config.openreward.balance_accepted_batches is False
    assert config.openreward.enable_programmatic_tool_calling is True
    assert config.openreward.enable_task_tracker is True
    assert config.openreward.enable_recursive_subagents is True
    assert config.openreward.subagent_environment_access == "read_only"
    assert config.openreward.enable_subagent_reward_judging is True
    assert config.openreward.subagent_delegation_reward_coefficient == 0.0

    workflow = config.workflow_config
    assert workflow.depth_level_weighting is True
    assert workflow.leave_one_out_baseline is True
    assert workflow.subagent_datum_keep_probability == 0.25
    assert workflow.straggler_quorum == 6
    assert workflow.straggler_timeout_seconds == 900
    assert workflow.min_successful_group_size == 4
    assert workflow.rollout_config.timeout == 3600
    assert workflow.rollout_config.step_timeout == 2700
    assert workflow.rollout_config.propogate_root_success is False
    assert workflow.token_efficiency_reward.enabled is True
    assert workflow.token_efficiency_reward.attribution == "policy_subtree"
    assert workflow.token_efficiency_reward.coefficient == 0.05

    assert config.rollout.request_timeout == 12600
    assert config.rollout.agent.session_timeout_seconds == 7200
    assert config.recover.freq_steps == 1
    assert config.valid_dataset is None
    assert config.evaluator.eval_before_train is False
    assert config.stats_logger.wandb.project == "openreward-multienv-openhands"
    assert config.stats_logger.wandb.group == "mix32-rec-eff-v5"
    _assert_wandb_identifiers_fit(config)


def test_mixed_launchers_are_valid_and_pin_verified_swe_catalog():
    subprocess.run(
        [
            "bash",
            "-n",
            str(MIXED_16_LAUNCHER),
            str(MIXED_32_LAUNCHER),
            str(MIXED_SERVER),
        ],
        check=True,
    )
    mixed_16_text = MIXED_16_LAUNCHER.read_text()
    mixed_32_text = MIXED_32_LAUNCHER.read_text()
    assert MIXED_32_LAUNCHER.stat().st_mode & 0o111
    assert "OPENREWARD_MIXED_EXPECTED_NNODES=32" in mixed_32_text
    assert "OPENREWARD_SUBAGENT_DELEGATION_REWARD_COEFFICIENT=0.0" in mixed_32_text
    assert "fp32-lm-head-hardened-v1.yaml" in mixed_16_text
    verified_catalog = ".cache/swe-rebench-v2-filtered-verified"
    assert verified_catalog in mixed_16_text
    assert verified_catalog in mixed_32_text
    assert verified_catalog in MIXED_SERVER.read_text()
