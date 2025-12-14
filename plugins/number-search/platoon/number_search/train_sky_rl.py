"""
Number Search training with Platoon SkyRL.

Usage:
    # From the number-search plugin directory:
    python -m platoon.number_search.train_sky_rl --config-name=number_search_skyrl
    
    # With overrides:
    python -m platoon.number_search.train_sky_rl --config-name=number_search_skyrl \
        trainer.policy.model.path=Qwen/Qwen2.5-1.5B-Instruct \
        trainer.seed=123
        
    # With a different config:
    python -m platoon.number_search.train_sky_rl --config-name=my_config
"""

import os
import hydra
import ray
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from skyrl_train.entrypoints.main_base import config_dir as skyrl_config_dir, validate_cfg
from skyrl_train.utils.utils import prepare_runtime_environment
from skyrl_train.utils.ppo_utils import sync_registries
from platoon.train.sky_rl.trainer import entrypoint
from platoon.number_search.rollout import run_rollout
from platoon.number_search.tasks import get_task
from pathlib import Path


# Get the local config directory
LOCAL_CONFIG_DIR = str(Path(__file__).parent / "config")

# Get the platoon root directory (parent of plugins/number-search)
PLATOON_ROOT = Path(__file__).parent.parent.parent.parent.parent.resolve()


def initialize_ray_with_platoon(cfg: DictConfig):
    """
    Initialize Ray with custom runtime environment that includes the platoon package.
    
    This is needed because the number-search plugin depends on platoon via an editable
    install with a relative path. When Ray packages local modules, the relative path
    breaks. We solve this by adding both package directories to PYTHONPATH.
    """
    env_vars = prepare_runtime_environment(cfg)
    
    # Add both platoon root and number-search plugin to PYTHONPATH so workers can import them
    plugin_dir = str(PLATOON_ROOT / "plugins" / "number-search")
    existing_pythonpath = env_vars.get("PYTHONPATH", "")
    new_pythonpath = f"{PLATOON_ROOT}:{plugin_dir}"
    if existing_pythonpath:
        new_pythonpath = f"{new_pythonpath}:{existing_pythonpath}"
    env_vars["PYTHONPATH"] = new_pythonpath
    
    # Use working_dir to include the entire platoon project
    runtime_env = {
        "env_vars": env_vars,
        "working_dir": str(PLATOON_ROOT),
        "excludes": [
            # Exclude large/unnecessary directories to speed up packaging
            ".git",
            ".venv",
            "plugins/*/.venv",
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
            ".ruff_cache",
            "*.egg-info",
        ],
    }
    
    ray.init(runtime_env=runtime_env)
    sync_registries()


@hydra.main(
    config_path="config",  # Local config directory (relative to this file)
    config_name="number_search_skyrl",
    version_base=None
)
def main(cfg: DictConfig):
    """Run training with the provided config."""
    # Set PLATOON_ROOT env var for config resolution if not set
    if "PLATOON_ROOT" not in os.environ:
        os.environ["PLATOON_ROOT"] = str(PLATOON_ROOT)
        print(f"Set PLATOON_ROOT={PLATOON_ROOT}")
    
    validate_cfg(cfg)
    initialize_ray_with_platoon(cfg)
    ray.get(entrypoint.remote(cfg, run_rollout, get_task))


if __name__ == "__main__":
    main()
