"""A fixed Tinker mixture must not permanently omit the final partial batch."""

from collections import Counter
from types import SimpleNamespace

from datasets import Dataset

import platoon.train.tinker  # noqa: F401
from platoon.openreward.train_scripts.tinker import train_tinker
from platoon.train.tinker.rl import PlatoonTinkerDataloader


def test_frontier_materialization_covers_all_200_tasks_in_each_complete_epoch(monkeypatch):
    records = [
        {
            "task_id": f"{env}/{index}",
            "_openreward_environment": env,
            "_openreward_sampling_weight": float(count),
        }
        for env, count in [("algorithmic", 188), ("cpu", 12)]
        for index in range(count)
    ]
    monkeypatch.setattr(train_tinker, "get_task_records", lambda *args, **kwargs: records)
    config = SimpleNamespace(
        openreward=SimpleNamespace(is_mixture=True),
        train=SimpleNamespace(batch_size=32),
    )
    ordered = train_tinker._select_task_records(config, evaluation=False, seed=343)
    dataloader = PlatoonTinkerDataloader(Dataset.from_list(ordered), batch_size=32)
    assert len(dataloader) == 7
    for epoch in range(2):
        batches = [dataloader.get_batch(epoch * 7 + index) for index in range(7)]
        assert all(len(batch) == 32 for batch in batches)
        rows = [row for batch in batches for row in batch]
        assert {row["task_id"] for row in rows} == {row["task_id"] for row in records}
        # The existing weighted fair scheduler still decides every slot. Its
        # 224-row rounding gives 211:13, with 24 repeated rows after covering
        # the full pools; the configured 188:12 weights are not overwritten.
        assert Counter(row["_openreward_environment"] for row in rows) == {
            "algorithmic": 211,
            "cpu": 13,
        }
        assert all(
            row["_openreward_sampling_weight"] == (188.0 if row["_openreward_environment"] == "algorithmic" else 12.0)
            for row in rows
        )
        for env in ("algorithmic", "cpu"):
            env_ids = [row["task_id"] for row in rows if row["_openreward_environment"] == env]
            pool_size = 188 if env == "algorithmic" else 12
            assert len(set(env_ids[:pool_size])) == pool_size
