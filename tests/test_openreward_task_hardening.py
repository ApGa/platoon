from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENREWARD_ROOT = REPO_ROOT / "plugins/openreward/platoon/openreward"


def _load_tasks(monkeypatch):
    package = types.ModuleType("platoon.openreward")
    package.__path__ = [str(OPENREWARD_ROOT)]
    monkeypatch.setitem(sys.modules, "platoon.openreward", package)
    monkeypatch.delitem(sys.modules, "platoon.openreward.config_defs", raising=False)
    monkeypatch.delitem(sys.modules, "platoon.openreward.tasks", raising=False)
    return __import__("platoon.openreward.tasks", fromlist=["get_task_records"])


def _install_client(monkeypatch, environment):
    class Client:
        def __init__(self, api_key):
            self.environments = types.SimpleNamespace(get=lambda name: environment)

        def close(self):
            pass

    module = types.ModuleType("openreward")
    module.OpenReward = Client
    monkeypatch.setitem(sys.modules, "openreward", module)


@pytest.mark.parametrize("field_name", ["train_task_limit", "eval_task_limit"])
def test_task_limits_must_be_positive(monkeypatch, field_name):
    tasks = _load_tasks(monkeypatch)

    with pytest.raises(ValueError, match=rf"{field_name} must be a positive integer"):
        tasks.OpenRewardEnvironmentConfig(**{field_name: 0})


def test_get_task_ids_rejects_an_explicit_zero_limit(monkeypatch):
    tasks = _load_tasks(monkeypatch)

    with pytest.raises(ValueError, match="task limit must be a positive integer"):
        tasks.get_task_ids(tasks.OpenRewardConfig(), limit=0)


def test_indexed_task_limit_does_not_materialize_the_full_catalog(monkeypatch):
    tasks = _load_tasks(monkeypatch)

    class Environment:
        def num_tasks(self, split):
            assert split == "train"
            return 10**12

        def list_tasks(self, split):
            raise AssertionError("mixed environments must not call list_tasks")

    _install_client(monkeypatch, Environment())
    config = tasks.OpenRewardConfig.from_mapping(
        {
            "environments": [
                {
                    "env_name": "huge-catalog",
                    "train_task_limit": 2,
                }
            ]
        }
    )

    records = tasks.get_task_records(config)

    assert len(records) == 2
    references = [
        tasks.OpenRewardTaskReference.decode(record["task_id"])
        for record in records
    ]
    assert [reference.task_index for reference in references] == [0, 1]


def test_legacy_catalog_failure_falls_back_to_indexed_tasks(monkeypatch):
    tasks = _load_tasks(monkeypatch)

    class Environment:
        def list_tasks(self, split):
            raise NotImplementedError("catalog is intentionally disabled")

        def num_tasks(self, split):
            return 2

    _install_client(monkeypatch, Environment())
    config = tasks.OpenRewardConfig(env_name="large", train_task_limit=1)

    records = tasks.get_task_records(config)

    assert len(records) == 1
    reference = tasks.OpenRewardTaskReference.decode(records[0]["task_id"])
    assert reference is not None
    assert reference.task_index == 0


def test_legacy_task_enumeration_reports_both_failures(monkeypatch):
    tasks = _load_tasks(monkeypatch)

    class Environment:
        def list_tasks(self, split):
            raise ConnectionError("catalog endpoint unavailable")

        def num_tasks(self, split):
            raise AttributeError("indexed endpoint unavailable")

    _install_client(monkeypatch, Environment())
    config = tasks.OpenRewardConfig(env_name="broken")

    with pytest.raises(RuntimeError) as exc_info:
        tasks.get_task_records(config)

    message = str(exc_info.value)
    assert "broken" in message
    assert "split='train'" in message
    assert "list_tasks failed with ConnectionError: catalog endpoint unavailable" in message
    assert "num_tasks/get_task fallback failed with AttributeError: indexed endpoint unavailable" in message
    assert isinstance(exc_info.value.__cause__, AttributeError)
