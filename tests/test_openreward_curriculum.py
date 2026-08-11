from __future__ import annotations

import os
import subprocess
from pathlib import Path

from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "plugins/openreward/platoon/openreward/configs/areal"
CONFIG_16 = (
    "toolathlon_swe_openhands_areal_prealloc_16node-cp-ptc-task-tracker-"
    "full-r3-fp32-lm-head-ta20-curriculum"
)
CONFIG_32 = (
    "toolathlon_swe_openhands_areal_prealloc_32node-cp-ptc-recursive-judged-"
    "r3-fp32-lm-head-bs8-efficiency-ta20-curriculum"
)
LAUNCHER_16 = (
    REPO_ROOT
    / "slurm-scripts/openreward-toolathlon-swe-prealloc-16node-ptc-task-tracker-ta20-curriculum.sh"
)
LAUNCHER_32 = (
    REPO_ROOT
    / "slurm-scripts/openreward-toolathlon-swe-prealloc-32node-ptc-recursive-ta20-curriculum.sh"
)
COMMON_LAUNCHER = (
    REPO_ROOT
    / "slurm-scripts/openreward-multienv-prealloc-16node-ptc-task-tracker-full.sh"
)


def _compose(config_name: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name=config_name)


def _assert_two_phase_catalog(config) -> None:
    environments = {
        environment.label: environment
        for environment in config.openreward.environments
    }
    assert list(environments) == ["toolathlon", "swe_rebench"]
    assert "tmax" not in environments
    assert environments["toolathlon"].sampling_start_step == 0
    assert environments["swe_rebench"].sampling_start_step == 20
    for environment in environments.values():
        assert environment.sampling_weight == 1.0
        assert environment.train_task_limit is None
        assert environment.eval_task_limit is None
    assert config.openreward.balance_accepted_batches is False
    assert config.train_dataset.batch_size == 8
    assert config.workflow_config.group_size == 8
    assert config.valid_dataset is None


def test_nonrecursive_toolathlon_swe_curriculum_composes_from_fresh_trial():
    config = _compose(CONFIG_16)

    _assert_two_phase_catalog(config)
    assert config.trial_name == "ta-swe16-ptc-ta20-v2-trial0"
    assert config.stats_logger.wandb.group == "ta-swe16-ptc-ta20-v2"
    assert config.openreward.enable_recursive_subagents is False
    assert config.openreward.enable_programmatic_tool_calling is True
    assert config.openreward.enable_task_tracker is True


def test_recursive_toolathlon_swe_curriculum_preserves_efficiency_settings():
    config = _compose(CONFIG_32)

    _assert_two_phase_catalog(config)
    assert config.trial_name == "ta-swe32-rec-ta20-v2-trial0"
    assert config.stats_logger.wandb.group == "ta-swe32-rec-ta20-v2"
    assert config.cluster.n_nodes == 32
    assert config.openreward.enable_recursive_subagents is True
    assert config.openreward.subagent_environment_access == "shared"
    assert config.openreward.subagent_delegation_reward_coefficient == 0.0
    assert config.workflow_config.token_efficiency_reward.enabled is True
    assert config.workflow_config.token_efficiency_reward.attribution == "policy_subtree"


def test_curriculum_launchers_disable_tmax_and_are_shell_valid():
    subprocess.run(
        [
            "bash",
            "-n",
            str(LAUNCHER_16),
            str(LAUNCHER_32),
            str(COMMON_LAUNCHER),
        ],
        check=True,
    )
    assert LAUNCHER_16.stat().st_mode & 0o111
    assert LAUNCHER_32.stat().st_mode & 0o111
    assert "OPENREWARD_ENABLE_TMAX=0" in LAUNCHER_16.read_text()
    assert "OPENREWARD_ENABLE_TMAX=0" in LAUNCHER_32.read_text()
    common_text = COMMON_LAUNCHER.read_text()
    assert "ENABLE_TMAX=${OPENREWARD_ENABLE_TMAX:-1}" in common_text
    assert '[[ "${ENABLE_TMAX}" -eq 1 ]]' in common_text


def test_curriculum_launchers_resolve_checkout_from_slurm_spool(tmp_path):
    missing_config = tmp_path / "stop-after-repo-resolution.yaml"
    for index, launcher in enumerate((LAUNCHER_16, LAUNCHER_32)):
        spool_dir = tmp_path / str(index) / "slurm" / "spool"
        spool_dir.mkdir(parents=True)
        spooled_launcher = spool_dir / "slurm_script"
        spooled_launcher.write_bytes(launcher.read_bytes())

        result = subprocess.run(
            ["bash", str(spooled_launcher), str(missing_config)],
            cwd=tmp_path,
            env={**os.environ, "SLURM_SUBMIT_DIR": str(REPO_ROOT)},
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 2
        assert f"ERROR: config not found: {missing_config}" in result.stderr
        assert "could not locate the Platoon checkout" not in result.stderr
