from __future__ import annotations

import os
from typing import Any

from platoon.envs.base import Task

from platoon.openreward.config_defs import OpenRewardConfig


def _configure_openreward_urls(config: OpenRewardConfig) -> None:
    os.environ.setdefault("OPENREWARD_DISABLE_UPDATE_CHECK", "1")
    if config.session_url:
        os.environ["OPENREWARD_SESSION_URL"] = config.session_url
        os.environ["OPENREWARD_API_URL"] = config.api_url or config.session_url
    elif config.api_url:
        os.environ["OPENREWARD_API_URL"] = config.api_url


def _task_name(task: Any, fallback: int) -> str:
    spec = getattr(task, "task_spec", None)
    if isinstance(spec, dict):
        for key in ("task_name", "id", "task_id", "name"):
            if spec.get(key):
                return str(spec[key])
    for key in ("task_name", "id", "task_id", "name"):
        value = getattr(task, key, None)
        if value:
            return str(value)
    return f"task-{fallback}"


def get_task_ids(config: OpenRewardConfig, *, split: str | None = None, limit: int | None = None) -> list[str]:
    _configure_openreward_urls(config)

    from openreward import OpenReward

    client = OpenReward(api_key=config.api_key)
    try:
        environment = client.environments.get(name=config.env_name)
        task_names = [
            _task_name(task, index)
            for index, task in enumerate(environment.list_tasks(split=split or config.split))
        ]
    finally:
        client.close()

    if config.task_names is not None:
        allowed = set(config.task_names)
        task_names = [task_name for task_name in task_names if task_name in allowed]
    if limit is not None:
        task_names = task_names[:limit]
    return task_names


def get_task(task_id: str) -> Task:
    return Task(
        id=task_id,
        goal=(
            "Call `get_task()` first and use its returned prompt as the task instructions. "
            "Use the returned environment tools directly by name. If the environment itself "
            "exposes a catalog/meta tool such as `call_tool`, follow the task prompt for when "
            "to use it. Call `claim_done` when complete."
        ),
    )
