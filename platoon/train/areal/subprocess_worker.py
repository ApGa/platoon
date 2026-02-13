"""Lightweight subprocess worker for AReaL rollouts.

This module provides a simple worker function that can be executed in a subprocess
to run rollouts in isolation. The worker:
1. Dynamically imports the rollout and task functions
2. Creates the task and config objects
3. Runs the async rollout function in a new event loop
4. Returns trajectory data directly (no file I/O)
"""

import asyncio
import importlib
from typing import Any


def run_rollout_subprocess(
    rollout_fn_module: str,
    rollout_fn_name: str,
    get_task_fn_module: str,
    get_task_fn_name: str,
    task_id: str,
    config_dict: dict[str, Any],
) -> dict | None:
    """Execute a single rollout in a subprocess.

    This function is designed to be called via ProcessPoolExecutor. It handles:
    - Dynamic imports of plugin-specific functions
    - Reconstruction of config and task objects
    - Running async rollout in subprocess's event loop
    - Exception handling and logging

    Args:
        rollout_fn_module: Module path for the rollout function (e.g., "platoon.textcraft.rollout")
        rollout_fn_name: Name of the rollout function (e.g., "run_rollout")
        get_task_fn_module: Module path for the task getter (e.g., "platoon.textcraft.tasks")
        get_task_fn_name: Name of the task getter function (e.g., "get_task")
        task_id: Task identifier to retrieve
        config_dict: Serialized RolloutConfig as dictionary

    Returns:
        Trajectory data dict with "trajectories" key, or None if rollout failed.
        The rollout function's internal timeouts are respected.
    """
    try:
        # Dynamic imports - allows subprocess to work with any plugin
        rollout_module = importlib.import_module(rollout_fn_module)
        rollout_fn = getattr(rollout_module, rollout_fn_name)

        task_module = importlib.import_module(get_task_fn_module)
        get_task_fn = getattr(task_module, get_task_fn_name)

        # Reconstruct config from serialized dict
        from platoon.config_defs import RolloutConfig

        config = RolloutConfig(**config_dict)

        # Get task and apply max_steps if specified
        task = get_task_fn(task_id)
        if config.max_steps is not None:
            task.max_steps = config.max_steps

        # Run async rollout in subprocess's event loop
        # The rollout function's internal timeouts (config.timeout, config.step_timeout)
        # will be respected during execution
        results = asyncio.run(rollout_fn(task, config))

        return results

    except Exception as e:
        # Log error to stderr (visible in parent process logs)
        print(f"[SubprocessWorker] Rollout failed for task {task_id}: {e}")
        import traceback

        traceback.print_exc()
        return None
