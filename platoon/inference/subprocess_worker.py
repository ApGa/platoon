"""Subprocess worker for inference rollouts."""

from __future__ import annotations

import asyncio
import importlib
import signal
from typing import Any


def _timeout_handler(_signum: int, _frame: Any) -> None:
    raise TimeoutError("Inference rollout subprocess hard timeout exceeded")


def run_rollout_subprocess(
    rollout_fn_module: str,
    rollout_fn_name: str,
    get_task_fn_module: str,
    get_task_fn_name: str,
    task_id: str,
    rollout_config_dict: dict[str, Any],
) -> dict[str, Any] | None:
    """Execute a single rollout in a subprocess."""
    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        from platoon.config_defs import RolloutConfig

        rollout_config = RolloutConfig(**rollout_config_dict)
        hard_timeout = int((rollout_config.timeout or 900) + 120)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(hard_timeout)

        rollout_module = importlib.import_module(rollout_fn_module)
        rollout_fn = getattr(rollout_module, rollout_fn_name)

        task_module = importlib.import_module(get_task_fn_module)
        get_task_fn = getattr(task_module, get_task_fn_name)

        task = get_task_fn(task_id)
        if rollout_config.max_steps is not None:
            task.max_steps = rollout_config.max_steps

        result = asyncio.run(rollout_fn(task, rollout_config))
        if hasattr(result, "to_dict"):
            result = result.to_dict()
        if isinstance(result, dict):
            return result
        return None
    except Exception as e:
        print(f"[InferenceSubprocessWorker] Rollout failed for task {task_id}: {e}")
        import traceback

        traceback.print_exc()
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
