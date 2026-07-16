from __future__ import annotations

import importlib.util
import sys
import types
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENREWARD_ROOT = REPO_ROOT / "plugins/openreward/platoon/openreward"


def _module(monkeypatch, name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _install_openreward_package(monkeypatch) -> None:
    package = types.ModuleType("platoon.openreward")
    package.__path__ = [str(OPENREWARD_ROOT)]
    monkeypatch.setitem(sys.modules, "platoon.openreward", package)


def _load_tasks(monkeypatch):
    _install_openreward_package(monkeypatch)
    monkeypatch.delitem(sys.modules, "platoon.openreward.config_defs", raising=False)
    monkeypatch.delitem(sys.modules, "platoon.openreward.tasks", raising=False)
    return __import__("platoon.openreward.tasks", fromlist=["get_task_records"])


def _load_bridge(monkeypatch):
    class FastMCP:
        pass

    _module(monkeypatch, "mcp")
    _module(monkeypatch, "mcp.server")
    _module(monkeypatch, "mcp.server.fastmcp", FastMCP=FastMCP)
    spec = importlib.util.spec_from_file_location(
        "openreward_mixture_bridge",
        OPENREWARD_ROOT / "mcp_bridge.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "openreward_mixture_bridge", module)
    spec.loader.exec_module(module)
    return module


def _load_mixture_module():
    spec = importlib.util.spec_from_file_location(
        "openreward_mixture_sampler",
        OPENREWARD_ROOT / "mixture.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["openreward_mixture_sampler"] = module
    spec.loader.exec_module(module)
    return module


def _load_areal_trainer(monkeypatch):
    _install_openreward_package(monkeypatch)

    class BaseTrainer:
        pass

    class StatefulDataLoader:
        def __init__(self, dataset, **kwargs):
            self.dataset = dataset
            self.batch_size = kwargs["batch_size"]
            self.sampler = kwargs["sampler"]

    _module(monkeypatch, "areal")
    _module(monkeypatch, "areal.utils", stats_tracker=types.SimpleNamespace())
    _module(monkeypatch, "platoon.train.areal", PlatoonArealRLTrainer=BaseTrainer)
    _module(monkeypatch, "torchdata")
    _module(
        monkeypatch,
        "torchdata.stateful_dataloader",
        StatefulDataLoader=StatefulDataLoader,
    )
    _module(
        monkeypatch,
        "platoon.openreward.tasks",
        OPENREWARD_ENVIRONMENT_COLUMN="environment",
        OPENREWARD_MIXTURE_COLUMN="mixture",
        OPENREWARD_SAMPLING_WEIGHT_COLUMN="weight",
    )
    monkeypatch.delitem(sys.modules, "platoon.openreward.mixture", raising=False)

    spec = importlib.util.spec_from_file_location(
        "openreward_areal_trainer_test",
        OPENREWARD_ROOT / "areal_trainer.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "openreward_areal_trainer_test", module)
    spec.loader.exec_module(module)
    return module


def test_areal_trainer_keeps_training_sampler_when_validation_is_also_mixed(monkeypatch):
    trainer_module = _load_areal_trainer(monkeypatch)

    class Dataset:
        column_names = ["environment", "mixture", "weight"]

        def __init__(self):
            self.columns = {
                "environment": ["a", "b", "c"] * 8,
                "weight": [1.0] * 24,
            }

        def __getitem__(self, key):
            return self.columns[key]

    train_config = types.SimpleNamespace(
        batch_size=8,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )
    valid_config = types.SimpleNamespace(
        batch_size=12,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )
    trainer = trainer_module.OpenRewardArealRLTrainer.__new__(trainer_module.OpenRewardArealRLTrainer)
    trainer.config = types.SimpleNamespace(seed=7, train_dataset=train_config)
    trainer._openreward_train_sampler = None

    train_loader = trainer._create_dataloader(Dataset(), train_config, rank=0, world_size=1)
    valid_loader = trainer._create_dataloader(Dataset(), valid_config, rank=0, world_size=1)

    assert train_loader.batch_size == 8
    assert valid_loader.batch_size == 12
    assert trainer._openreward_train_sampler is train_loader.sampler
    assert valid_loader.sampler is not train_loader.sampler


def test_mixed_config_defaults_to_equal_weights_and_validates_labels(monkeypatch):
    tasks = _load_tasks(monkeypatch)
    config = tasks.OpenRewardConfig.from_mapping(
        {
            "environments": [
                {"label": "toolathlon", "env_name": "toolathlongym"},
                {"label": "tmax", "env_name": "tmax/TMax-15K-Harbor"},
            ]
        }
    )

    assert config.is_mixture is True
    assert [environment.sampling_weight for environment in config.resolved_environments()] == [1.0, 1.0]
    assert config.balance_accepted_batches is True
    assert config.accepted_batch_max_replacement_rounds == 8

    with pytest.raises(ValueError, match="labels must be unique"):
        tasks.OpenRewardConfig.from_mapping(
            {
                "environments": [
                    {"label": "same", "env_name": "one"},
                    {"label": "same", "env_name": "two"},
                ]
            }
        )
    with pytest.raises(ValueError, match="finite and positive"):
        tasks.OpenRewardConfig.from_mapping({"environments": [{"env_name": "one", "sampling_weight": 0.0}]})
    with pytest.raises(ValueError, match="pool env-var names must be unique"):
        tasks.OpenRewardConfig.from_mapping(
            {
                "environments": [
                    {"label": "swe-rebench", "env_name": "one"},
                    {"label": "swe_rebench", "env_name": "two"},
                ]
            }
        )
    with pytest.raises(ValueError, match="balance_accepted_batches"):
        tasks.OpenRewardConfig.from_mapping({"balance_accepted_batches": 1})
    with pytest.raises(ValueError, match="max_replacement_rounds"):
        tasks.OpenRewardConfig.from_mapping({"accepted_batch_max_replacement_rounds": -1})


def test_task_reference_round_trip_and_malformed_payload(monkeypatch):
    tasks = _load_tasks(monkeypatch)
    reference = tasks.OpenRewardTaskReference(
        environment_label="swe",
        split="validation/heldout",
        task_index=17,
        task_name="owner/repo#17",
    )

    encoded = reference.encode()
    assert tasks.OpenRewardTaskReference.decode(encoded) == reference
    task = tasks.get_task(encoded)
    assert task.id == encoded
    assert task.misc[tasks.OPENREWARD_ENVIRONMENT_LABEL_KEY] == "swe"
    assert task.misc[tasks.OPENREWARD_TASK_SPLIT_KEY] == "validation/heldout"
    assert task.misc[tasks.OPENREWARD_TASK_INDEX_KEY] == 17
    assert task.misc[tasks.OPENREWARD_TASK_NAME_KEY] == "owner/repo#17"

    assert tasks.OpenRewardTaskReference.decode("legacy-task") is None
    with pytest.raises(ValueError, match="Malformed|empty"):
        tasks.OpenRewardTaskReference.decode(tasks.OPENREWARD_TASK_REFERENCE_PREFIX + "not-base64!")


def test_mixed_task_records_use_num_tasks_without_listing(monkeypatch):
    tasks = _load_tasks(monkeypatch)

    class Environment:
        def __init__(self, count):
            self.count = count
            self.num_calls = []
            self.list_calls = 0

        def num_tasks(self, split):
            self.num_calls.append(split)
            return self.count

        def list_tasks(self, split):
            self.list_calls += 1
            raise AssertionError(f"list_tasks must not be called for {split}")

    environments = {"one": Environment(4), "two": Environment(3)}

    class Client:
        def __init__(self, api_key):
            self.api_key = api_key
            self.environments = types.SimpleNamespace(get=lambda name: environments[name])

        def close(self):
            pass

    _module(monkeypatch, "openreward", OpenReward=Client)
    config = tasks.OpenRewardConfig.from_mapping(
        {
            "environments": [
                {
                    "label": "first",
                    "env_name": "one",
                    "split": "train-a",
                    "eval_split": "eval-a",
                    "train_task_limit": 2,
                    "eval_task_limit": 1,
                    "sampling_weight": 2.0,
                },
                {
                    "label": "second",
                    "env_name": "two",
                    "split": "train-b",
                    "train_task_limit": 3,
                    "eval_task_limit": 2,
                    "sampling_weight": 1.0,
                },
            ]
        }
    )

    train = tasks.get_task_records(config)
    assert len(train) == 5
    assert [record[tasks.OPENREWARD_SAMPLING_WEIGHT_COLUMN] for record in train] == [
        2.0,
        2.0,
        1.0,
        1.0,
        1.0,
    ]
    assert all(record[tasks.OPENREWARD_MIXTURE_COLUMN] is True for record in train)
    decoded = [tasks.OpenRewardTaskReference.decode(record["task_id"]) for record in train]
    assert [(item.environment_label, item.split, item.task_index) for item in decoded] == [
        ("first", "train-a", 0),
        ("first", "train-a", 1),
        ("second", "train-b", 0),
        ("second", "train-b", 1),
        ("second", "train-b", 2),
    ]

    evaluation = tasks.get_task_records(config, evaluation=True)
    assert len(evaluation) == 3
    assert all(tasks.OPENREWARD_SAMPLING_WEIGHT_COLUMN not in record for record in evaluation)
    eval_refs = [tasks.OpenRewardTaskReference.decode(record["task_id"]) for record in evaluation]
    assert [(item.environment_label, item.split, item.task_index) for item in eval_refs] == [
        ("first", "eval-a", 0),
        ("second", "train-b", 0),
        ("second", "train-b", 1),
    ]
    assert all(environment.list_calls == 0 for environment in environments.values())


def test_legacy_single_environment_records_do_not_enable_mixture_sampler(monkeypatch):
    tasks = _load_tasks(monkeypatch)

    class Environment:
        def num_tasks(self, split):
            assert split == "train"
            return 2

        def list_tasks(self, split):
            raise NotImplementedError

    class Client:
        def __init__(self, api_key):
            self.environments = types.SimpleNamespace(get=lambda name: Environment())

        def close(self):
            pass

    _module(monkeypatch, "openreward", OpenReward=Client)
    config = tasks.OpenRewardConfig.from_mapping({"env_name": "legacy", "task_indices": [0, 1]})

    records = tasks.get_task_records(config)

    assert len(records) == 2
    assert all(tasks.OPENREWARD_MIXTURE_COLUMN not in record for record in records)
    assert all(tasks.OPENREWARD_SAMPLING_WEIGHT_COLUMN not in record for record in records)


def test_bridge_index_selection_never_lists_catalog(monkeypatch, tmp_path):
    bridge = _load_bridge(monkeypatch)

    class TaskRecord:
        task_spec = {"task_id": "selected-7"}

    class Session:
        def get_prompt(self):
            return "Fix it"

        def list_tools(self, format):
            assert format == "openai"
            return []

    class SessionContext:
        def __enter__(self):
            return Session()

        def __exit__(self, exc_type, exc, traceback):
            return False

    class Environment:
        def __init__(self):
            self.get_calls = []

        def list_tasks(self, split):
            raise AssertionError("indexed bridge must not call list_tasks")

        def num_tasks(self, split):
            assert split == "train"
            return 12

        def get_task(self, split, index):
            self.get_calls.append((split, index))
            return TaskRecord()

        def session(self, task):
            assert isinstance(task, TaskRecord)
            return SessionContext()

    environment = Environment()

    class Client:
        def __init__(self, api_key):
            self.environments = types.SimpleNamespace(get=lambda name: environment)

        def close(self):
            pass

    _module(monkeypatch, "openreward", OpenReward=Client)
    runtime = bridge.OpenRewardMCPBridge(
        bridge.BridgeConfig(
            env_name="large-env",
            split="train",
            task_index=7,
            task_name=None,
            session_url="http://localhost:8080",
            api_url=None,
            api_key="local",
            output_dir=tmp_path,
            max_tool_calls=0,
        )
    )
    try:
        assert environment.get_calls == [("train", 7)]
        assert runtime.task_name == "selected-7"
    finally:
        runtime.close()


def test_balanced_sampler_rotates_equal_remainders_and_shards_global_batches():
    sampler_module = _load_mixture_module()
    ids = ["a"] * 9 + ["b"] * 9 + ["c"] * 9
    weights = [1.0] * len(ids)
    sampler = sampler_module.BalancedEnvironmentSampler(
        ids,
        weights,
        global_batch_size=8,
        seed=11,
        shuffle=True,
    )

    assert [Counter(batch) for batch in sampler.environment_batches()] == [
        Counter({"a": 3, "b": 3, "c": 2}),
        Counter({"a": 3, "c": 3, "b": 2}),
        Counter({"b": 3, "c": 3, "a": 2}),
    ]

    rank_samplers = [
        sampler_module.BalancedEnvironmentSampler(
            ids,
            weights,
            global_batch_size=8,
            num_replicas=2,
            rank=rank,
            seed=11,
        )
        for rank in range(2)
    ]
    rank_indices = [list(value) for value in rank_samplers]
    rebuilt = [item for pair in zip(*rank_indices) for item in pair]
    rebuilt_batches = [
        Counter(ids[index] for index in rebuilt[start : start + 8]) for start in range(0, len(rebuilt), 8)
    ]
    assert rebuilt_batches == [Counter(batch) for batch in sampler.environment_batches()]


def test_balanced_sampler_explicit_ratio_and_epoch_determinism():
    sampler_module = _load_mixture_module()
    ids = ["a"] * 8 + ["b"] * 8 + ["c"] * 8
    weights = [2.0] * 8 + [1.0] * 8 + [1.0] * 8
    sampler = sampler_module.BalancedEnvironmentSampler(
        ids,
        weights,
        global_batch_size=8,
        seed=19,
    )

    assert all(Counter(batch) == Counter({"a": 4, "b": 2, "c": 2}) for batch in sampler.environment_batches())
    epoch_zero = list(sampler)
    assert epoch_zero == list(sampler)
    sampler.set_epoch(1)
    assert list(sampler) != epoch_zero

    # No environment repeats an underlying task before its local pool is
    # exhausted. Environment a has exactly eight slots in the first two batches.
    first_two_batches = list(sampler)[:16]
    a_indices = [index for index in first_two_batches if ids[index] == "a"]
    assert len(a_indices) == len(set(a_indices)) == 8


def test_materialized_record_order_preserves_exact_tinker_batch_ratios():
    sampler_module = _load_mixture_module()
    records = (
        [{"env": "a", "weight": 2.0, "task_id": f"a-{index}"} for index in range(8)]
        + [{"env": "b", "weight": 1.0, "task_id": f"b-{index}"} for index in range(8)]
        + [{"env": "c", "weight": 1.0, "task_id": f"c-{index}"} for index in range(8)]
    )

    ordered = sampler_module.materialize_balanced_record_order(
        records,
        environment_key="env",
        sampling_weight_key="weight",
        global_batch_size=8,
        seed=23,
        preserve_order_key="_preserve_order",
    )

    assert all(record["_preserve_order"] is True for record in ordered)
    batches = [ordered[start : start + 8] for start in range(0, len(ordered), 8)]
    assert len(batches) == 3
    assert all(Counter(record["env"] for record in batch) == Counter({"a": 4, "b": 2, "c": 2}) for batch in batches)


def _strict_batch_input_stream(environment_batches):
    task_id = 0
    while True:
        for environment_batch in environment_batches:
            for environment in environment_batch:
                yield types.SimpleNamespace(
                    task_id=task_id,
                    data={"environment": environment},
                )
                task_id += 1


class _StrictBatchDispatcher:
    def __init__(self, reject_counts=None):
        self.runner = types.SimpleNamespace(max_queue_size=128)
        self.logger = None
        self.staleness_manager = types.SimpleNamespace(get_stats=lambda: "fake-stats")
        self.reject_counts = Counter(reject_counts or {})
        self.current_inputs = []
        self.round_environments = []
        self.environment_by_task_id = {}

    def submit_task_input(self, task_input):
        self.current_inputs.append(task_input)
        self.environment_by_task_id[task_input.task_id] = task_input.data["environment"]

    def wait_results(self, count, timeout):
        assert timeout == 1
        assert len(self.current_inputs) == count
        inputs, self.current_inputs = self.current_inputs, []
        self.round_environments.append([task_input.data["environment"] for task_input in inputs])
        results = []
        for task_input in inputs:
            environment = task_input.data["environment"]
            if self.reject_counts[environment] > 0:
                self.reject_counts[environment] -= 1
                results.append(None)
            else:
                results.append(types.SimpleNamespace(task_id=task_input.task_id))
        # Completion order must not affect the accepted environment quota.
        return list(reversed(results))


class _RejectOneLabelPerInitialRoundDispatcher(_StrictBatchDispatcher):
    def __init__(self, label, batch_size):
        super().__init__()
        self.label = label
        self.batch_size = batch_size

    def wait_results(self, count, timeout):
        if count == self.batch_size:
            self.reject_counts[self.label] += 1
        return super().wait_results(count, timeout)


def test_strict_accepted_batches_retry_only_deficits_and_rotate_targets():
    sampler_module = _load_mixture_module()
    target_batches = [
        ["a", "b", "c", "a", "b", "c", "a", "b"],
        ["c", "a", "b", "c", "a", "b", "c", "a"],
        ["b", "c", "a", "b", "c", "a", "b", "c"],
    ]
    dispatcher = _StrictBatchDispatcher(reject_counts={"b": 1})
    coordinator = sampler_module.StrictEnvironmentBatchCoordinator(
        target_batches,
        input_environment=lambda task_input: task_input.data["environment"],
        max_replacement_rounds=2,
    )
    coordinator.install(dispatcher)
    input_stream = _strict_batch_input_stream(target_batches)

    first = dispatcher.active_submit_and_wait(input_stream, batch_size=8)
    first_environments = [dispatcher.environment_by_task_id[result.task_id] for result in first]

    assert Counter(first_environments) == Counter({"a": 3, "b": 3, "c": 2})
    assert dispatcher.round_environments[:2] == [target_batches[0], ["b"]]
    assert coordinator.last_target_counts == Counter({"a": 3, "b": 3, "c": 2})
    assert coordinator.last_accepted_counts == coordinator.last_target_counts
    assert coordinator.last_attempt_counts == Counter({"a": 3, "b": 4, "c": 2})
    assert coordinator.last_retry_rounds == 1
    assert not dispatcher.current_inputs

    second = dispatcher.active_submit_and_wait(input_stream, batch_size=8)
    second_environments = [dispatcher.environment_by_task_id[result.task_id] for result in second]
    assert Counter(second_environments) == Counter({"a": 3, "b": 2, "c": 3})
    assert dispatcher.round_environments[2] == target_batches[1]
    assert coordinator.last_retry_rounds == 0


def test_strict_accepted_batches_resume_offset_and_fail_closed():
    sampler_module = _load_mixture_module()
    targets = [["a", "b", "c"], ["c", "a", "c"]]
    input_stream = _strict_batch_input_stream(targets)
    dispatcher = _StrictBatchDispatcher(reject_counts={"c": 1})
    coordinator = sampler_module.StrictEnvironmentBatchCoordinator(
        targets,
        input_environment=lambda task_input: task_input.data["environment"],
        start_batch_index=1,
        max_replacement_rounds=0,
    )
    coordinator.install(dispatcher)

    with pytest.raises(RuntimeError, match="exhausted replacement rounds"):
        dispatcher.active_submit_and_wait(input_stream, batch_size=3)

    assert dispatcher.round_environments == [targets[1]]
    assert coordinator.last_target_counts == Counter({"c": 2, "a": 1})
    assert coordinator.last_accepted_counts == Counter({"a": 1, "c": 1})
    assert coordinator.last_attempt_counts == coordinator.last_target_counts
    assert not dispatcher.current_inputs

    dynamic_dispatcher = _StrictBatchDispatcher()
    dynamic_coordinator = sampler_module.StrictEnvironmentBatchCoordinator(
        [["a", "b", "c"]],
        input_environment=lambda task_input: task_input.data["environment"],
    )
    dynamic_coordinator.install(dynamic_dispatcher)
    with pytest.raises(ValueError, match="dynamic_bs=true"):
        dynamic_dispatcher.active_submit_and_wait(
            _strict_batch_input_stream([["a", "b", "c"]]),
            batch_size=3,
            dynamic_bs=True,
        )
    assert dynamic_dispatcher.round_environments == []


def test_strict_accepted_batches_bound_retry_buffers_and_validate_round_limit():
    sampler_module = _load_mixture_module()
    targets = [["a", "b", "c"]]

    with pytest.raises(ValueError, match="non-negative integer"):
        sampler_module.StrictEnvironmentBatchCoordinator(
            targets,
            input_environment=lambda task_input: task_input.data["environment"],
            max_replacement_rounds=1.5,
        )

    dispatcher = _RejectOneLabelPerInitialRoundDispatcher("b", batch_size=3)
    coordinator = sampler_module.StrictEnvironmentBatchCoordinator(
        targets,
        input_environment=lambda task_input: task_input.data["environment"],
        max_replacement_rounds=1,
    )
    coordinator.install(dispatcher)
    input_stream = _strict_batch_input_stream(targets)

    for _ in range(100):
        results = dispatcher.active_submit_and_wait(input_stream, batch_size=3)
        environments = [dispatcher.environment_by_task_id[result.task_id] for result in results]
        assert Counter(environments) == Counter(targets[0])

    assert all(len(values) <= 3 for values in coordinator._buffered_inputs.values())
    assert sum(coordinator.total_discarded_input_counts.values()) > 0
