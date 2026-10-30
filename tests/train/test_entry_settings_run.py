"""Each environment entry's groups run with the entry's own settings, in the trainer's own loop (Tinker faked,
``fakes.trainer``): its group size, its rollout settings, its minimum group size, its baseline and its staleness bound,
and a capped label never stalls the others."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from datasets import Dataset
from fakes.config import mixture_config
from fakes.trainer import TrainingClient, datum, make_trainer

from platoon.task import Task
from platoon.train import entry_settings
from platoon.train import launch as train_module
from platoon.train.components import ENVIRONMENT_COLUMN
from platoon.train.dispatch.permits import DrawLease, PermitPool
from platoon.train.policy.policy import Policy
from platoon.train.policy.set import PolicySet
from platoon.train.policy.spec import MAIN, PolicySpec
from platoon.train.workers.protocol import PROTOCOL_VERSION, RolloutWorkerResponse
from platoon.train.workflow import context as context_module
from platoon.train.workflow import members as members_module
from platoon.train.workflow.group import GroupRolloutWorkflow
from platoon.utils.config import ConfigError
from platoon.utils.rollout_workload import RolloutWorkload
from platoon.utils.tinker_data_processing import TrajectoryCollectionResult, TrajectoryStats


def _rows(*names_and_counts: tuple[str, int]) -> Dataset:
    """A mixture's training rows: ``count`` tasks of each entry, its name in the entry column."""

    return Dataset.from_list(
        [
            {"task_id": f"{name}::{name}{index}", ENVIRONMENT_COLUMN: name}
            for name, count in names_and_counts
            for index in range(count)
        ]
    )


# --- the exit check: two entries of different settings ----------------------------------------------------------


class _Workflow(GroupRolloutWorkflow):
    """The real group workflow, which records each group's baselines with the settings it saw."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.baselines: list[tuple[str | None, bool, list[tuple[float, bool]], list[float]]] = []

    def compute_baselines(self, results):
        baselines = super().compute_baselines(results)
        rewards = [(result.task_reward, result.task_reward_valid) for result in results]
        self.baselines.append((context_module.current_entry(), self.config.leave_one_out_baseline, rewards, baselines))
        return baselines


def test_each_entry_s_groups_run_with_its_own_settings_within_the_concurrency_budget(tmp_path, monkeypatch):
    """Two entries: groups of 8 and 4, a step limit of 30 and none, a minimum group of 6 and 3, and a leave-one-out
    baseline on one.  Each entry's rollout settings reach its worker requests, each group's size, minimum and baseline
    are its entry's, and the rollouts running at once never exceed max_concurrent_rollouts."""

    config = mixture_config(
        tmp_path,
        {
            "batch_size": 2,
            "max_training_steps": 4,
            "max_concurrent_rollouts": 12,  # a group of 8 and one of 4 at once, never two of 8
            "workflow_config": {"group_size": 8, "min_successful_group_size": 6, "rollout_config": {"max_steps": 30}},
        },
        [
            {"name": "quick"},
            {
                "name": "slow",
                "workflow_config": {
                    "group_size": 4,
                    "min_successful_group_size": 3,
                    "leave_one_out_baseline": True,
                    "rollout_config": {"max_steps": None},
                },
            },
        ],
    )
    settings = entry_settings.entry_settings(config)
    requests: list[Any] = []
    running = {"now": 0, "most": 0}

    async def run_rollout_request(request, llm, *, rollout_fn=None, reward_processor=None):
        requests.append(request)
        running["now"] += 1
        running["most"] = max(running["most"], running["now"])
        try:
            await asyncio.sleep(0.005)
        finally:
            running["now"] -= 1
        # quick0 returns 5 trainable rollouts of 8 (below its minimum of 6); slow0 3 of 4 (its minimum).
        number = request.rollout_number
        valid = not (request.task_id.endswith("quick0") and number >= 5) and not (
            request.task_id.endswith("slow0") and number == 3
        )
        reward = float(number % 2)
        stats = TrajectoryStats(
            trajectory_id="root",
            reward=reward,
            num_steps=2,
            num_input_tokens=3,
            num_output_tokens=2,
            num_datums=1,
            is_root=True,
        )
        result = TrajectoryCollectionResult(
            datums=[datum(advantage=reward, version=0)] if valid else [],
            task_reward=reward,
            trajectory_stats=[stats],
            root_rewards_dict={},
            task_reward_valid=valid,
        )
        workload = RolloutWorkload(
            environment_steps=2,
            model_calls=2,
            trajectories=1,
            postmerge_datums=1,
            policy_eligible_datums=int(valid),
            post_sampling_datums=int(valid),
        )
        return RolloutWorkerResponse(
            protocol_version=PROTOCOL_VERSION,
            request_id=request.request_id,
            status="completed",
            result=result,
            workload=workload,
            observed=True,
            outcome_status="scored" if valid else "error",
        )

    monkeypatch.setattr(members_module, "run_rollout_request", run_rollout_request)
    trainer, _saves, logged = make_trainer(
        config, training_client=TrainingClient(), train_dataset=_rows(("quick", 4), ("slow", 4))
    )
    llm = SimpleNamespace(version=0, model_name="m")
    workflow = _Workflow(
        rollout_fn=lambda task, rollout_config: None,
        get_task_fn=lambda task_id: Task(id=task_id),
        config=config.train.workflow_config,
        model_info=SimpleNamespace(model_name="m", llm=llm),
    )
    workflow.entry_configs = {label: own.train for label, own in settings.items()}

    asyncio.run(trainer.train(train_workflow=workflow))

    assert logged == [1, 2, 3, 4]
    by_task: dict[str, list[Any]] = {}
    for request in requests:
        by_task.setdefault(request.task_id, []).append(request)
    for task_id, task_requests in by_task.items():
        quick = task_id.startswith("quick::")
        group = 8 if quick else 4
        assert len(task_requests) % group == 0 and {r.rollout_number for r in task_requests} == set(range(group))
        assert {r.rollout_config["max_steps"] for r in task_requests} == {30 if quick else None}
        assert {r.task.max_steps for r in task_requests} == {30 if quick else None}
    assert running["most"] <= 12
    # Baselines: the mean of the group for quick, leave-one-out for slow; quick0's group (5 trainable of 8) did not
    # train, and slow0's (3 of 4) did.
    seen = {entry for entry, *_ in workflow.baselines}
    assert seen <= {"quick", "slow"} and "slow" in seen
    for entry, leave_one_out, rewards, baselines in workflow.baselines:
        assert leave_one_out is (entry == "slow")
        completed = [reward for reward, valid in rewards if valid]
        mean = sum(completed) / len(completed)
        if entry == "quick":
            assert baselines == [mean] * len(rewards)
        else:  # each completed root on the mean of the others; an untrainable one never enters a baseline
            total = sum(completed)
            assert baselines == [
                (total - reward) / (len(completed) - 1) if valid else mean for reward, valid in rewards
            ]
    trainable = [(entry, sum(valid for _, valid in rewards)) for entry, _, rewards, _ in workflow.baselines]
    assert ("quick", 5) not in trainable and ("slow", 3) in trainable


# --- staleness by entry ---------------------------------------------------------------------------------------------


class _AgedWorkflow:
    """Each group returns one datum: sampled at the current policy version for a quick task, at version 0 for a slow
    one, whose groups so grow a version older each step."""

    def __init__(self, llm) -> None:
        self.llm = llm

    async def arun_episode(self, data):
        await asyncio.sleep(0)
        slow = data[ENVIRONMENT_COLUMN] == "slow"
        return [datum(version=0 if slow else self.llm.version)]


def _stale_run(tmp_path: Path, slow_bound: int | None) -> list[dict[str, float]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    slow: dict[str, Any] = {"name": "slow"}
    if slow_bound is not None:
        slow["max_staleness"] = slow_bound
    config = mixture_config(
        tmp_path,
        {"batch_size": 2, "max_training_steps": 12, "max_staleness": 3, "max_concurrent_rollouts": 16},
        [{"name": "quick"}, slow],
    )
    trainer, _saves, logged = make_trainer(config, train_dataset=_rows(("quick", 1), ("slow", 1)))
    asyncio.run(trainer.train(train_workflow=_AgedWorkflow(trainer.model_info.llm)))
    return logged.stats


def test_a_slow_entry_s_groups_train_under_its_own_bound_while_the_run_s_stays(tmp_path):
    """With the run's bound at 3, a slow entry's groups ten versions old train under its own bound of 48; without
    it, they are dropped as stale, and each step counts its stale groups by label."""

    own = _stale_run(tmp_path / "own", 48)
    assert len(own) == 12
    assert sum(step["train/batch/stale_groups"] for step in own) == 0
    assert all(step["train/batch/environment/slow/stale_groups"] == 0 for step in own)

    run = _stale_run(tmp_path / "run", None)
    assert len(run) == 12
    stale = [step["train/batch/environment/slow/stale_groups"] for step in run]
    # No group is older than 3 versions before the fifth step; from then on, about half the groups are slow ones.
    assert stale[:4] == [0, 0, 0, 0] and sum(stale) >= 6
    assert all(step["train/batch/environment/quick/stale_groups"] == 0 for step in run)
    assert all(step["train/batch/stale_groups"] == step["train/batch/environment/slow/stale_groups"] for step in run)


# --- a capped label never stalls the others -------------------------------------------------------------------------


class _SlowWorkflow:
    """A quick task's group returns at once; a slow task's takes two seconds."""

    async def arun_episode(self, data):
        await asyncio.sleep(2.0 if data[ENVIRONMENT_COLUMN] == "slow" else 0)
        return [datum(version=0)]


def test_a_capped_slow_label_skips_its_excess_rows_and_never_stalls_the_other_label(tmp_path):
    """With rollout_group_limits {slow: 1} and the slow entry at 13% of the order, the quick entry's steps keep running
    for 100 steps while slow groups run one at a time, and the slow draws beyond its backlog are skipped and counted.
    Draws waiting for the limit once filled a pending queue and stopped every entry's at the slow groups' pace: over a
    hundred slow groups of two seconds each."""

    config = mixture_config(
        tmp_path,
        {
            "batch_size": 8,
            "max_training_steps": 100,
            "max_concurrent_rollouts": 16,  # two groups of 8: one is stuck on the slow label
            "rollout_group_limits": {"slow": 1},
        },
        [{"name": "quick"}, {"name": "slow"}],
    )
    trainer, _saves, logged = make_trainer(config, train_dataset=_rows(("quick", 87), ("slow", 13)))

    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=_SlowWorkflow()), timeout=60))

    assert len(logged) == 100
    skipped = sum(step["train/batch/skipped_rows"] for step in logged.stats)
    assert skipped > 0
    assert skipped == sum(step["train/batch/environment/slow/skipped_rows"] for step in logged.stats)
    assert all(step["train/batch/environment/quick/skipped_rows"] == 0 for step in logged.stats)


@pytest.mark.parametrize(
    ("train", "environments", "steps"),
    [
        pytest.param(
            {"batch_size": 8, "max_concurrent_rollouts": 16, "max_staleness": 3},
            [{"name": "quick"}, {"name": "slow"}],
            100,
            id="run-bound-3",
        ),
        pytest.param(
            {"batch_size": 8, "max_concurrent_rollouts": 16, "max_staleness": 8},
            [{"name": "quick"}, {"name": "slow"}],
            100,
            id="run-bound-8",
        ),
        pytest.param(  # the shape of the guide's example: groups of 8 and 4, the slow entry's bound 48
            {
                "batch_size": 20,
                "max_staleness": 3,
                "workflow_config": {"group_size": 8, "min_successful_group_size": 4},
            },
            [
                {"name": "quick", "workflow_config": {"rollout_config": {"max_steps": 30}}},
                {
                    "name": "slow",
                    "max_staleness": 48,
                    "workflow_config": {
                        "group_size": 4,
                        "min_successful_group_size": 3,
                        "leave_one_out_baseline": True,
                    },
                },
            ],
            60,
            id="guide-example",
        ),
    ],
)
def test_a_capped_slow_label_never_stalls_the_other_label_under_a_staleness_bound(tmp_path, train, environments, steps):
    """train.max_staleness bounds the groups started ahead of training (the horizon), and the draws of a capped label
    that wait (parked) or are skipped do not count toward it: the quick entry's steps keep running while slow groups
    run one at a time.  Counted in batch indices, the parked and skipped slow rows once filled the lookahead, and every
    step then waited for a slow group (48 of 100 steps in a minute)."""

    config = mixture_config(
        tmp_path,
        {"max_training_steps": steps, "rollout_group_limits": {"slow": 1}, **train},
        environments,
    )
    trainer, _saves, logged = make_trainer(config, train_dataset=_rows(("quick", 87), ("slow", 13)))

    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=_SlowWorkflow()), timeout=60))

    assert len(logged) == steps
    assert sum(step["train/batch/skipped_rows"] for step in logged.stats) > 0
    parked = [step["train/batch/parked_rows"] for step in logged.stats]
    assert max(parked) > 0 and parked == [step["train/batch/environment/slow/parked_rows"] for step in logged.stats]
    assert all(step["train/batch/environment/quick/parked_rows"] == 0 for step in logged.stats)


# --- rollout permits ------------------------------------------------------------------------------------------------


def test_rollout_permits_admit_whole_groups_first_come_first_served():
    async def run():
        permits = PermitPool(12)
        order: list[str] = []
        first = permits.reserve(8)

        async def take(name: str, count: int) -> DrawLease:
            reservation = permits.reserve(count)
            try:
                await reservation.ready()
            except BaseException:
                reservation.withdraw()
                raise
            order.append(name)
            return reservation

        large = asyncio.create_task(take("large", 8))
        await asyncio.sleep(0)
        small = asyncio.create_task(take("small", 4))  # 4 are free, but the large group waits first
        await asyncio.sleep(0)
        assert order == []
        cancelled = asyncio.create_task(take("cancelled", 4))
        await asyncio.sleep(0)
        cancelled.cancel()
        await asyncio.gather(cancelled, return_exceptions=True)
        first.withdraw()
        granted = await asyncio.gather(large, small)
        assert order == ["large", "small"]
        for reservation in granted:
            reservation.withdraw()
        assert permits.reserve(12).startable  # nothing was lost to the cancelled waiter

    asyncio.run(run())


# --- platoon train hands each workflow its entries' settings ----------------------------------------------------


TASKS = {"quick": ["a", "b"], "slow": ["c", "d"]}


def load_tasks(config, split, *, entry):
    return TASKS[entry] if split == "train" else []


def load_task(task_id: str) -> Task:
    return Task(id=task_id)


async def rollout(task: Task, config) -> dict:
    return {}


class NoEntrySettings:
    """A workflow class of a project's own that does not apply an entry's settings."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    async def arun_episode(self, data):
        return []


class _Trainer:
    """Stands in for the trainer: records the workflows platoon train builds."""

    built: list[_Trainer] = []

    def __init__(self, *, config, train_dataset, eval_dataset, train_order, train_sampling=None) -> None:
        self.config = config
        self.model_info: Any = SimpleNamespace(model_name="m", llm=SimpleNamespace(version=0))
        self.policies = PolicySet([Policy(PolicySpec(MAIN, model_name="m", renderer_name="r"), self.model_info)])
        self.run_log_path = str(Path(config.log_path, config.stats.experiment_name, config.stats.trial_name))
        _Trainer.built.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def train(self, *, train_workflow, eval_workflow):
        self.workflows = (train_workflow, eval_workflow)


def _train_config(tmp_path: Path, **top) -> str:
    entry = {"dataset_loader": f"{__name__}:load_tasks", "task_loader": f"{__name__}:load_task"}
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "train": {"model_name": "m", "renderer_name": "r", "batch_size": 2, "max_training_steps": 1},
                "eval": {"every": 0},
                "log_path": str(tmp_path / "logs"),
                "environments": [
                    {"name": "quick", **entry, "rollout": f"{__name__}:rollout", "dataset_kwargs": {"entry": "quick"}},
                    {
                        "name": "slow",
                        **entry,
                        "rollout": f"{__name__}:rollout",
                        "dataset_kwargs": {"entry": "slow"},
                        "workflow_config": {"group_size": 4, "rollout_config": {"max_steps": 12}},
                    },
                ],
                **top,
            }
        )
    )
    return str(path)


def test_platoon_train_hands_each_workflow_its_entries_settings_and_records_them(tmp_path, monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")
    monkeypatch.setattr(train_module, "PlatoonTinkerRLTrainer", _Trainer)
    monkeypatch.setattr(_Trainer, "built", [])
    run_dir = tmp_path / "logs" / "platoon_tinker" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "lock.json").write_text("{}")

    asyncio.run(train_module.arun_tinker_training(["--config", _train_config(tmp_path)]))

    train_workflow, eval_workflow = _Trainer.built[0].workflows
    assert train_workflow.entry_configs["slow"].group_size == 4
    assert train_workflow.entry_configs["slow"].rollout_config.max_steps == 12
    assert train_workflow.entry_configs["quick"] is train_workflow.config  # the run's own
    assert eval_workflow.entry_configs["slow"].rollout_config.max_steps == 12  # evaluation inherits it
    assert eval_workflow.entry_configs["slow"].group_size == 1
    lock = json.loads((run_dir / "lock.json").read_text())
    assert lock["entry_settings"]["slow"]["train"] == {"group_size": 4, "rollout_config.max_steps": 12}


def test_platoon_train_refuses_entry_settings_a_workflow_class_does_not_apply(tmp_path, monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")
    monkeypatch.setattr(train_module, "PlatoonTinkerRLTrainer", _Trainer)
    monkeypatch.setattr(_Trainer, "built", [])
    path = _train_config(tmp_path)
    data = yaml.safe_load(Path(path).read_text())
    data["environments"][0]["workflow"] = f"{__name__}:NoEntrySettings"
    Path(path).write_text(yaml.safe_dump(data))

    with pytest.raises(
        ConfigError, match=r"environments\[slow\]\.workflow_config: the run's workflow, .*NoEntrySettings"
    ):
        asyncio.run(train_module.arun_tinker_training(["--config", path]))
    assert _Trainer.built == []

    NoEntrySettings.applies_entry_settings = True  # a class that applies them declares so
    try:
        asyncio.run(train_module.arun_tinker_training(["--config", path]))
    finally:
        del NoEntrySettings.applies_entry_settings
    assert _Trainer.built


def test_a_start_whose_listed_tasks_fail_their_checks_records_nothing_in_lock_json(tmp_path, monkeypatch):
    """The checks that need the listed tasks run before the start records itself in lock.json: a label column that no
    task has stops the run with the run's lock as it was, without the entries' own settings the start would record."""

    monkeypatch.setenv("TINKER_API_KEY", "tml-test")
    monkeypatch.setattr(train_module, "PlatoonTinkerRLTrainer", _Trainer)
    monkeypatch.setattr(_Trainer, "built", [])
    run_dir = tmp_path / "logs" / "platoon_tinker" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "lock.json").write_text("{}")
    path = _train_config(tmp_path)
    data = yaml.safe_load(Path(path).read_text())
    data["train"]["label_by"] = "suite"  # no task has the column
    Path(path).write_text(yaml.safe_dump(data))

    with pytest.raises(ConfigError, match="suite"):
        asyncio.run(train_module.arun_tinker_training(["--config", path]))

    assert json.loads((run_dir / "lock.json").read_text()) == {}
    assert _Trainer.built == []
