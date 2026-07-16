from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any

from platoon.envs.base import Task

from platoon.openreward.config_defs import OpenRewardConfig, OpenRewardEnvironmentConfig
from platoon.openreward.constants import (
    OPENREWARD_ENVIRONMENT_LABEL_KEY,
    OPENREWARD_RESOLVE_GOAL_KEY,
    OPENREWARD_TASK_INDEX_KEY,
    OPENREWARD_TASK_NAME_KEY,
    OPENREWARD_TASK_SPLIT_KEY,
)

OPENREWARD_TASK_REFERENCE_PREFIX = "openreward:v1:"
OPENREWARD_ENVIRONMENT_COLUMN = "_openreward_environment"
OPENREWARD_SAMPLING_WEIGHT_COLUMN = "_openreward_sampling_weight"
OPENREWARD_MIXTURE_COLUMN = "_openreward_mixture"


def _configure_openreward_urls(config: OpenRewardEnvironmentConfig) -> None:
    os.environ.setdefault("OPENREWARD_DISABLE_UPDATE_CHECK", "1")
    if config.session_url:
        os.environ["OPENREWARD_SESSION_URL"] = config.session_url
        os.environ["OPENREWARD_API_URL"] = config.api_url or config.session_url
    elif config.api_url:
        os.environ["OPENREWARD_API_URL"] = config.api_url


@dataclass(frozen=True)
class OpenRewardTaskReference:
    environment_label: str
    split: str
    task_index: int
    task_name: str | None = None

    def __post_init__(self) -> None:
        if not self.environment_label:
            raise ValueError("OpenReward task reference requires an environment label")
        if not self.split:
            raise ValueError("OpenReward task reference requires a split")
        if isinstance(self.task_index, bool) or not isinstance(self.task_index, int) or self.task_index < 0:
            raise ValueError("OpenReward task reference index must be a non-negative integer")

    def encode(self) -> str:
        payload: dict[str, Any] = {
            "environment": self.environment_label,
            "index": self.task_index,
            "split": self.split,
        }
        if self.task_name is not None:
            payload["name"] = self.task_name
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        token = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
        return OPENREWARD_TASK_REFERENCE_PREFIX + token

    @classmethod
    def decode(cls, value: str) -> OpenRewardTaskReference | None:
        if not value.startswith(OPENREWARD_TASK_REFERENCE_PREFIX):
            return None
        token = value.removeprefix(OPENREWARD_TASK_REFERENCE_PREFIX)
        if not token:
            raise ValueError("OpenReward task reference payload is empty")
        try:
            raw = base64.b64decode(
                token + "=" * (-len(token) % 4),
                altchars=b"-_",
                validate=True,
            )
            payload = json.loads(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Malformed OpenReward task reference") from exc
        if not isinstance(payload, dict):
            raise ValueError("Malformed OpenReward task reference payload")
        required = {"environment", "index", "split"}
        if not required.issubset(payload) or not set(payload).issubset(required | {"name"}):
            raise ValueError("Malformed OpenReward task reference fields")
        environment_label = payload["environment"]
        split = payload["split"]
        task_index = payload["index"]
        task_name = payload.get("name")
        if not isinstance(environment_label, str) or not isinstance(split, str):
            raise ValueError("Malformed OpenReward task reference strings")
        if task_name is not None and not isinstance(task_name, str):
            raise ValueError("Malformed OpenReward task reference name")
        return cls(
            environment_label=environment_label,
            split=split,
            task_index=task_index,
            task_name=task_name,
        )


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


def _listed_tasks(environment: Any, split: str) -> list[Any]:
    return list(environment.list_tasks(split=split))


def _indexed_task_ids(
    environment: Any,
    config: OpenRewardEnvironmentConfig,
    *,
    split: str,
    limit: int | None,
) -> list[str]:
    count = int(environment.num_tasks(split=split))
    if config.task_indices is not None:
        invalid = [index for index in config.task_indices if index >= count]
        if invalid:
            raise ValueError(
                f"Task indices {invalid[:5]} are outside {config.env_name} split={split!r} "
                f"range 0..{max(count - 1, 0)}"
            )
        indices = config.task_indices if limit is None else config.task_indices[:limit]
    else:
        # Do not materialize every numeric index before applying a small task
        # limit. Large OpenReward catalogs can contain millions of tasks.
        stop = count if limit is None else min(count, limit)
        indices = range(stop)
    return [
        OpenRewardTaskReference(
            environment_label=config.resolved_label,
            split=split,
            task_index=index,
        ).encode()
        for index in indices
    ]


def _named_task_ids(
    environment: Any,
    config: OpenRewardEnvironmentConfig,
    *,
    split: str,
    limit: int | None,
    namespace: bool,
) -> list[str]:
    tasks = _listed_tasks(environment, split)
    indexed_names = [(index, _task_name(task, index)) for index, task in enumerate(tasks)]
    if config.task_names is not None:
        allowed = set(config.task_names)
        indexed_names = [item for item in indexed_names if item[1] in allowed]
    if limit is not None:
        indexed_names = indexed_names[:limit]
    if not namespace:
        return [name for _, name in indexed_names]
    return [
        OpenRewardTaskReference(
            environment_label=config.resolved_label,
            split=split,
            task_index=index,
            task_name=name,
        ).encode()
        for index, name in indexed_names
    ]


def _environment_task_ids(
    config: OpenRewardEnvironmentConfig,
    *,
    split: str,
    limit: int | None,
    namespace: bool,
) -> list[str]:
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
    ):
        raise ValueError("OpenReward task limit must be a positive integer when set")

    _configure_openreward_urls(config)

    from openreward import OpenReward

    client = OpenReward(api_key=config.api_key)
    try:
        environment = client.environments.get(name=config.env_name)
        # Explicit names require the legacy catalog API. Numeric selection and
        # every true mixture use OpenReward's scalable num_tasks/get_task API.
        if config.task_names is not None:
            return _named_task_ids(
                environment,
                config,
                split=split,
                limit=limit,
                namespace=namespace,
            )
        if namespace or config.task_indices is not None or split != config.split:
            return _indexed_task_ids(environment, config, split=split, limit=limit)

        # Preserve display names and resume keys for legacy small environments.
        # Large environments intentionally reject list_tasks, so fall back to
        # compact indexed references without materializing their task specs.
        try:
            return _named_task_ids(
                environment,
                config,
                split=split,
                limit=limit,
                namespace=False,
            )
        except Exception as list_error:
            try:
                return _indexed_task_ids(environment, config, split=split, limit=limit)
            except Exception as indexed_error:
                raise RuntimeError(
                    f"Unable to enumerate tasks for OpenReward environment "
                    f"{config.env_name!r} split={split!r}: "
                    f"list_tasks failed with {type(list_error).__name__}: {list_error}; "
                    f"num_tasks/get_task fallback failed with "
                    f"{type(indexed_error).__name__}: {indexed_error}"
                ) from indexed_error
    finally:
        client.close()


def get_task_records(config: OpenRewardConfig, *, evaluation: bool = False) -> list[dict[str, Any]]:
    """Build routed task rows for training or evaluation.

    Training rows carry a private weight column consumed by the OpenReward
    AReaL sampler. Evaluation rows intentionally enumerate every selected task
    once and omit that column.
    """

    records: list[dict[str, Any]] = []
    for environment in config.resolved_environments():
        split = (environment.eval_split or environment.split) if evaluation else environment.split
        limit = environment.eval_task_limit if evaluation else environment.train_task_limit
        task_ids = _environment_task_ids(
            environment,
            split=split,
            limit=limit,
            namespace=config.is_mixture or split != environment.split,
        )
        if not task_ids:
            raise RuntimeError(
                f"No tasks selected for OpenReward environment {environment.resolved_label!r} "
                f"split={split!r}"
            )
        for task_id in task_ids:
            record: dict[str, Any] = {
                "task_id": task_id,
                OPENREWARD_ENVIRONMENT_COLUMN: environment.resolved_label,
            }
            if not evaluation and config.is_mixture:
                record[OPENREWARD_SAMPLING_WEIGHT_COLUMN] = environment.sampling_weight
                record[OPENREWARD_MIXTURE_COLUMN] = True
            records.append(record)
    return records


def get_task_ids(config: OpenRewardConfig, *, split: str | None = None, limit: int | None = None) -> list[str]:
    """Compatibility task-id API; mixed callers should prefer get_task_records."""

    if config.is_mixture:
        return [record["task_id"] for record in get_task_records(config)]
    environment = config.environment()
    selected_split = split or environment.split
    return _environment_task_ids(
        environment,
        split=selected_split,
        limit=limit,
        namespace=selected_split != environment.split,
    )


def get_task(task_id: str) -> Task:
    reference = OpenRewardTaskReference.decode(task_id)
    misc: dict[str, Any] = {OPENREWARD_RESOLVE_GOAL_KEY: True}
    display_id = task_id
    if reference is not None:
        misc.update(
            {
                OPENREWARD_ENVIRONMENT_LABEL_KEY: reference.environment_label,
                OPENREWARD_TASK_INDEX_KEY: reference.task_index,
                OPENREWARD_TASK_SPLIT_KEY: reference.split,
            }
        )
        if reference.task_name is not None:
            misc[OPENREWARD_TASK_NAME_KEY] = reference.task_name
            display_id = reference.task_name
        else:
            display_id = f"{reference.environment_label}:{reference.split}:{reference.task_index}"
    return Task(
        id=task_id,
        goal=f"Task {display_id}",
        misc=misc,
    )
