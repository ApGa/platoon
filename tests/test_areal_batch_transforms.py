"""Focused regression tests for Platoon's trainer-side AReaL batch transforms."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_batch_transforms_module():
    return _load_module(
        "platoon_batch_transforms_test",
        REPO_ROOT / "platoon/train/areal/batch_transforms.py",
    )


def _load_trainer_module(batch_transforms_module):
    areal_pkg = types.ModuleType("platoon.train.areal")
    areal_pkg.__path__ = []

    sys.modules["platoon.train.areal"] = areal_pkg
    sys.modules["platoon.train.areal.batch_transforms"] = batch_transforms_module

    actor_mod = types.ModuleType("platoon.train.areal.actor")
    actor_mod.PlatoonPPOActor = type(
        "PlatoonPPOActor",
        (),
        {"as_controller": classmethod(lambda cls, config, scheduler: cls())},
    )
    actor_mod.PlatoonMegatronPPOActor = type(
        "PlatoonMegatronPPOActor",
        (),
        {"as_controller": classmethod(lambda cls, config, scheduler: cls())},
    )
    sys.modules["platoon.train.areal.actor"] = actor_mod

    config_mod = types.ModuleType("platoon.train.areal.config_defs")
    config_mod.PlatoonArealRLTrainerConfig = object
    config_mod.PlatoonPPOActorConfig = object
    sys.modules["platoon.train.areal.config_defs"] = config_mod

    deadline_mod = types.ModuleType("platoon.train.areal.deadline")
    deadline_mod.StepDeadlineGuard = SimpleNamespace(
        from_environment=lambda: None
    )
    sys.modules["platoon.train.areal.deadline"] = deadline_mod

    preallocated_mod = types.ModuleType("platoon.train.areal.preallocated_slurm")
    preallocated_mod.PreallocatedSlurmScheduler = type("PreallocatedSlurmScheduler", (), {})
    sys.modules["platoon.train.areal.preallocated_slurm"] = preallocated_mod

    workflow_serialization_mod = types.ModuleType("platoon.train.areal.workflow_serialization")
    workflow_serialization_mod.normalize_remote_workflow = lambda workflow: workflow
    sys.modules["platoon.train.areal.workflow_serialization"] = workflow_serialization_mod

    api_mod = types.ModuleType("areal.api")
    api_mod.WorkflowLike = object
    sys.modules["areal.api"] = api_mod

    cli_mod = types.ModuleType("areal.api.cli_args")
    cli_mod.OpenAIProxyConfig = type("OpenAIProxyConfig", (), {"admin_api_key": "test-key"})
    sys.modules["areal.api.cli_args"] = cli_mod

    infra_mod = types.ModuleType("areal.infra")
    infra_mod.RolloutController = type("RolloutController", (), {})
    infra_mod.current_platform = SimpleNamespace(synchronize=lambda: None)
    sys.modules["areal.infra"] = infra_mod

    trainer_mod = types.ModuleType("areal.trainer.rl_trainer")
    trainer_mod.PPOTrainer = type("PPOTrainer", (), {})
    sys.modules["areal.trainer.rl_trainer"] = trainer_mod

    @contextmanager
    def _null_context(*args, **kwargs):
        yield None

    utils_mod = types.ModuleType("areal.utils")
    utils_mod.logging = SimpleNamespace(getLogger=lambda name: SimpleNamespace(info=lambda *a, **k: None))
    utils_mod.perf_tracer = SimpleNamespace(trace_scope=_null_context)
    utils_mod.stats_tracker = SimpleNamespace(
        record_timing=_null_context,
        scalar=lambda **kwargs: None,
    )
    sys.modules["areal.utils"] = utils_mod

    environ_mod = types.ModuleType("areal.utils.environ")
    environ_mod.is_single_controller = lambda: True
    sys.modules["areal.utils.environ"] = environ_mod

    perf_mod = types.ModuleType("areal.utils.perf_tracer")
    perf_mod.Category = SimpleNamespace(COMPUTE="compute", COMM="comm", IO="io", INSTR="instr")
    sys.modules["areal.utils.perf_tracer"] = perf_mod

    data_mod = types.ModuleType("areal.utils.data")

    def concat_padded_tensors(items):
        out = {}
        for key in items[0]:
            values = [item[key] for item in items]
            if torch.is_tensor(values[0]):
                out[key] = torch.cat(values, dim=0)
            else:
                out[key] = sum(values, [])
        return out

    data_mod.concat_padded_tensors = concat_padded_tensors
    sys.modules["areal.utils.data"] = data_mod

    datasets_mod = types.ModuleType("datasets")
    datasets_mod.Dataset = object
    sys.modules["datasets"] = datasets_mod

    return _load_module(
        "platoon_rl_test",
        REPO_ROOT / "platoon/train/areal/rl.py",
    )


def _evaluator_config(**overrides):
    values = {
        "eval_before_train": False,
        "freq_epochs": None,
        "freq_steps": None,
        "freq_secs": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "override",
    [
        {"eval_before_train": True},
        {"freq_epochs": 1},
        {"freq_steps": 0},
        {"freq_secs": 60},
    ],
)
def test_evaluation_enabled_recognizes_every_schedule(override):
    batch_transforms = _load_batch_transforms_module()
    rl_module = _load_trainer_module(batch_transforms)

    config = SimpleNamespace(evaluator=_evaluator_config(**override))

    assert rl_module._evaluation_enabled(config)


def test_init_rollout_skips_unused_eval_controller(monkeypatch):
    batch_transforms = _load_batch_transforms_module()
    rl_module = _load_trainer_module(batch_transforms)
    trainer = rl_module.PlatoonArealRLTrainer.__new__(rl_module.PlatoonArealRLTrainer)
    trainer.config = SimpleNamespace(evaluator=_evaluator_config())
    delegated: list[tuple[object, bool, str | None]] = []

    def init_rollout(_self, rollout_config, is_eval=False, lora_path=None):
        delegated.append((rollout_config, is_eval, lora_path))
        return "upstream-rollout"

    monkeypatch.setattr(rl_module.PPOTrainer, "_init_rollout", init_rollout, raising=False)

    assert trainer._init_rollout("eval-config", is_eval=True) is None
    assert delegated == []
    assert trainer._init_rollout("train-config", is_eval=False, lora_path="adapter") == ("upstream-rollout")
    assert delegated == [("train-config", False, "adapter")]


def test_init_rollout_preserves_enabled_evaluation(monkeypatch):
    batch_transforms = _load_batch_transforms_module()
    rl_module = _load_trainer_module(batch_transforms)
    trainer = rl_module.PlatoonArealRLTrainer.__new__(rl_module.PlatoonArealRLTrainer)
    # Explicit zero is still an AReaL setting, not the disabled sentinel.
    trainer.config = SimpleNamespace(evaluator=_evaluator_config(freq_steps=0))

    monkeypatch.setattr(
        rl_module.PPOTrainer,
        "_init_rollout",
        lambda _self, rollout_config, is_eval=False, lora_path=None: (
            rollout_config,
            is_eval,
            lora_path,
        ),
        raising=False,
    )

    assert trainer._init_rollout("eval-config", is_eval=True) == (
        "eval-config",
        True,
        None,
    )


def test_disabled_evaluation_reuses_training_proxy_url():
    batch_transforms = _load_batch_transforms_module()
    rl_module = _load_trainer_module(batch_transforms)

    class Controller(rl_module.RolloutController):
        def __init__(self):
            self.proxy_starts = 0

        def start_proxy(self):
            self.proxy_starts += 1

    trainer = rl_module.PlatoonArealRLTrainer.__new__(rl_module.PlatoonArealRLTrainer)
    trainer.config = SimpleNamespace(rollout=SimpleNamespace(agent=SimpleNamespace(mode="inline")))
    trainer.rollout = Controller()
    trainer.eval_rollout = None

    trainer._start_platoon_proxies()

    assert trainer.rollout.proxy_starts == 1
    assert trainer.proxy_base_url is None
    assert trainer.eval_proxy_base_url is trainer.proxy_base_url


def test_depth_level_weighting_transform_matches_inverse_frequency_formula():
    batch_transforms = _load_batch_transforms_module()
    transform = batch_transforms.DepthLevelWeightingTransform()
    context = batch_transforms.BatchTransformContext(
        config=SimpleNamespace(
            workflow_config=SimpleNamespace(depth_level_discount_gamma=None),
        ),
        actor_dp_world_size=1,
    )

    batch = {
        "rewards": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        "traj_depth": torch.tensor([0, 0, 1, 0, 1, 1]),
        "traj_start": torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0, 0.0]),
    }

    transformed = transform(batch, context)

    # Full-batch counts:
    # depth 0 -> datum_count=3, traj_count=2
    # depth 1 -> datum_count=3, traj_count=2
    # So both depths receive weight 1 after normalization.
    assert transformed is not None
    assert torch.allclose(transformed["rewards"], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    assert "traj_depth" not in transformed
    assert "traj_start" not in transformed


def test_depth_level_weighting_transform_matches_gamma_discount_formula():
    batch_transforms = _load_batch_transforms_module()
    transform = batch_transforms.DepthLevelWeightingTransform()
    context = batch_transforms.BatchTransformContext(
        config=SimpleNamespace(
            workflow_config=SimpleNamespace(depth_level_discount_gamma=0.5),
        ),
        actor_dp_world_size=1,
    )

    batch = {
        "rewards": torch.tensor([2.0, 2.0, 2.0, 2.0]),
        "traj_depth": torch.tensor([0, 1, 2, 2]),
    }

    transformed = transform(batch, context)
    raw = torch.tensor([1.0, 0.5, 0.25, 0.25])
    expected = 2.0 * raw * (raw.numel() / raw.sum())

    assert transformed is not None
    assert torch.allclose(transformed["rewards"], expected)
    assert "traj_depth" not in transformed


def test_trainer_batch_transforms_run_after_trainable_datum_filtering():
    batch_transforms = _load_batch_transforms_module()
    rl_module = _load_trainer_module(batch_transforms)

    seen = {}

    def custom_transform(batch, context):
        seen["batch_rewards"] = batch["rewards"].clone()
        seen["has_trainable_datums"] = "trainable_datums" in batch
        seen["global_step"] = context.global_step
        return batch

    trainer = rl_module.PlatoonArealRLTrainer.__new__(rl_module.PlatoonArealRLTrainer)
    trainer.actor = SimpleNamespace(data_parallel_world_size=1)
    trainer.config = SimpleNamespace(
        rollout=SimpleNamespace(shuffle_cross_task=False, ensure_batch_divisible_by=1),
        workflow_config=SimpleNamespace(depth_level_weighting=False, depth_level_discount_gamma=None),
    )
    trainer.batch_transforms = [custom_transform]

    processed = trainer._postprocess_rollout_batch(
        [
            {
                "rewards": torch.tensor([[1.0], [2.0], [3.0]]),
                "trainable_datums": torch.tensor([True, False, True]),
                "attention_mask": torch.tensor([[1], [1], [1]], dtype=torch.bool),
            }
        ],
        global_step=7,
        epoch=1,
        epoch_step=2,
    )

    assert processed is not None
    assert len(processed) == 2
    assert torch.equal(processed[0]["rewards"], torch.tensor([[1.0]]))
    assert torch.equal(processed[1]["rewards"], torch.tensor([[3.0]]))
    assert torch.equal(seen["batch_rewards"].squeeze(-1), torch.tensor([1.0, 3.0]))
    assert seen["has_trainable_datums"] is False
    assert seen["global_step"] == 7


def test_depth_weighting_normalizes_after_final_dp_trim():
    batch_transforms = _load_batch_transforms_module()
    rl_module = _load_trainer_module(batch_transforms)

    trainer = rl_module.PlatoonArealRLTrainer.__new__(rl_module.PlatoonArealRLTrainer)
    trainer.actor = SimpleNamespace(data_parallel_world_size=2)
    trainer.config = SimpleNamespace(
        rollout=SimpleNamespace(shuffle_cross_task=False, ensure_batch_divisible_by=2),
        workflow_config=SimpleNamespace(depth_level_weighting=True, depth_level_discount_gamma=None),
    )
    trainer.batch_transforms = [batch_transforms.DepthLevelWeightingTransform()]

    processed = trainer._postprocess_rollout_batch(
        [
            {
                # The fifth datum is trimmed.  The final batch has two
                # one-datum trajectories at each depth, so every weight is 1.
                # Weighting before trimming would give depth-dependent values.
                "rewards": torch.ones(5, 1),
                "attention_mask": torch.ones(5, 1, dtype=torch.bool),
                "traj_depth": torch.tensor([0, 0, 1, 1, 1]),
                "traj_start": torch.ones(5),
            }
        ],
        global_step=0,
        epoch=0,
        epoch_step=0,
    )

    assert processed is not None
    assert len(processed) == 4
    assert torch.equal(
        torch.cat([datum["rewards"].reshape(-1) for datum in processed]),
        torch.ones(4),
    )


def test_random_trim_repairs_start_for_every_surviving_segment(monkeypatch):
    batch_transforms = _load_batch_transforms_module()
    rl_module = _load_trainer_module(batch_transforms)
    trainer = rl_module.PlatoonArealRLTrainer.__new__(rl_module.PlatoonArealRLTrainer)
    trainer.actor = SimpleNamespace(data_parallel_world_size=2)
    trainer.config = SimpleNamespace(
        rollout=SimpleNamespace(shuffle_cross_task=True, ensure_batch_divisible_by=2),
    )

    # Drop datum 3 (the start of segment 3) while retaining datum 4.  The
    # segment side channel lets the trainer promote datum 4 to the repaired
    # start rather than biasing selection toward a contiguous prefix/tail.
    monkeypatch.setattr(
        torch,
        "randperm",
        lambda size, device=None: torch.tensor([3, 4, 0, 2, 1], device=device),
    )
    processed = trainer._maybe_shuffle_and_trim_batch(
        {
            "datum_id": torch.arange(5),
            "traj_start": torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0]),
            "traj_depth": torch.tensor([0, 1, 1, 1, 1]),
            rl_module._TRAJECTORY_SEGMENT_ID_FIELD: torch.tensor([1, 1, 2, 3, 3]),
        }
    )

    assert processed is not None
    assert processed["datum_id"].tolist() == [4, 0, 2, 1]
    assert torch.equal(processed["traj_start"], torch.tensor([1.0, 1.0, 1.0, 0.0]))
    assert int(processed["traj_start"].sum().item()) == 3
    assert rl_module._TRAJECTORY_SEGMENT_ID_FIELD not in processed


def test_trim_preserves_later_roots_when_nonroot_datums_are_available(monkeypatch):
    batch_transforms = _load_batch_transforms_module()
    rl_module = _load_trainer_module(batch_transforms)
    trainer = rl_module.PlatoonArealRLTrainer.__new__(rl_module.PlatoonArealRLTrainer)
    trainer.actor = SimpleNamespace(data_parallel_world_size=4)
    trainer.config = SimpleNamespace(
        rollout=SimpleNamespace(shuffle_cross_task=True, ensure_batch_divisible_by=4),
    )
    monkeypatch.setattr(
        torch,
        "randperm",
        lambda size, device=None: torch.tensor([5, 1, 0, 6, 2, 3, 4], device=device),
    )

    processed = trainer._maybe_shuffle_and_trim_batch(
        {
            "datum_id": torch.arange(7),
            # Group 2's root is late in the concatenated batch (datum 5).
            "traj_depth": torch.tensor([0, 1, 1, 1, 1, 0, 1]),
            "traj_start": torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]),
            rl_module._TRAJECTORY_SEGMENT_ID_FIELD: torch.tensor([1, 2, 2, 2, 2, 3, 4]),
        }
    )

    assert processed is not None
    assert len(processed["datum_id"]) == 4
    assert {0, 5}.issubset(set(processed["datum_id"].tolist()))
    assert int(processed["traj_start"].sum().item()) == 3


def test_segment_ids_are_created_before_trainable_filter_and_repair_lost_start():
    batch_transforms = _load_batch_transforms_module()
    rl_module = _load_trainer_module(batch_transforms)
    trainer = rl_module.PlatoonArealRLTrainer.__new__(rl_module.PlatoonArealRLTrainer)
    trainer.actor = SimpleNamespace(data_parallel_world_size=1)
    trainer.config = SimpleNamespace(
        rollout=SimpleNamespace(shuffle_cross_task=False, ensure_batch_divisible_by=1),
    )

    reduced = trainer._reduce_rollout_batch(
        [
            {
                "datum_id": torch.arange(4),
                "attention_mask": torch.ones(4, 1, dtype=torch.bool),
                "traj_depth": torch.tensor([0, 0, 1, 1]),
                "traj_start": torch.tensor([1.0, 0.0, 1.0, 0.0]),
                # Remove segment 2's original start but retain its second datum.
                "trainable_datums": torch.tensor([True, True, False, True]),
            }
        ]
    )
    assert reduced is not None
    assert torch.equal(
        reduced[rl_module._TRAJECTORY_SEGMENT_ID_FIELD],
        torch.tensor([1, 1, 2]),
    )
    assert torch.equal(reduced["traj_start"], torch.tensor([1.0, 0.0, 0.0]))

    repaired = trainer._maybe_shuffle_and_trim_batch(reduced)
    assert repaired is not None
    assert torch.equal(repaired["traj_start"], torch.tensor([1.0, 0.0, 1.0]))
    assert rl_module._TRAJECTORY_SEGMENT_ID_FIELD not in repaired


def test_trainer_strips_heterogeneous_workflow_stats_before_cross_task_concat():
    batch_transforms = _load_batch_transforms_module()
    rl_module = _load_trainer_module(batch_transforms)
    trainer = rl_module.PlatoonArealRLTrainer.__new__(rl_module.PlatoonArealRLTrainer)

    batch = trainer._reduce_rollout_batch(
        [
            {
                "rewards": torch.tensor([[1.0]]),
                "attention_mask": torch.tensor([[True]]),
                "task_reward": torch.tensor([1.0]),
                "task_reward_valid": torch.tensor([True]),
                "reward/subagent_judgment": torch.tensor([0.75]),
                "_platoon_reward_metric_present/reward/subagent_judgment": torch.tensor([True]),
            },
            {
                "rewards": torch.tensor([[0.0]]),
                "attention_mask": torch.tensor([[True]]),
                "task_reward": torch.tensor([0.0]),
                "task_reward_valid": torch.tensor([False]),
            },
        ]
    )

    assert batch is not None
    assert torch.equal(batch["rewards"], torch.tensor([[1.0], [0.0]]))
    assert "task_reward" not in batch
    assert "task_reward_valid" not in batch
    assert "reward/subagent_judgment" not in batch
    assert not any(key.startswith("_platoon_reward_metric_present/") for key in batch)


def test_trainer_sums_and_strips_exact_accepted_task_workload_sidecars():
    batch_transforms = _load_batch_transforms_module()
    rl_module = _load_trainer_module(batch_transforms)
    trainer = rl_module.PlatoonArealRLTrainer.__new__(rl_module.PlatoonArealRLTrainer)
    trainer.actor = SimpleNamespace(data_parallel_world_size=1)

    def task_item(
        reward,
        *,
        steps,
        calls,
        inputs,
        outputs,
        trajectories,
        requested,
        observed,
        trainable,
        postmerge=0,
        policy_eligible=0,
        post_sampling=0,
        task_retained=0,
    ):
        item = {
            "rewards": torch.tensor([[reward]]),
            "attention_mask": torch.tensor([[True]]),
            rl_module._WORKLOAD_SIDECAR_FIELDS["environment_steps"]: torch.tensor([steps]),
            rl_module._WORKLOAD_SIDECAR_FIELDS["model_calls"]: torch.tensor([calls]),
            rl_module._WORKLOAD_SIDECAR_FIELDS["input_tokens"]: torch.tensor([inputs]),
            rl_module._WORKLOAD_SIDECAR_FIELDS["output_tokens"]: torch.tensor([outputs]),
            rl_module._WORKLOAD_SIDECAR_FIELDS["trajectories"]: torch.tensor([trajectories]),
            rl_module._WORKLOAD_REQUESTED_ROLLOUTS_KEY: torch.tensor([requested]),
            rl_module._WORKLOAD_OBSERVED_ROLLOUTS_KEY: torch.tensor([observed]),
            rl_module._WORKLOAD_TRAINABLE_ROLLOUTS_KEY: torch.tensor([trainable]),
            rl_module._WORKLOAD_DATUM_SIDECAR_FIELDS["postmerge_datums"]: torch.tensor(
                [postmerge]
            ),
            rl_module._WORKLOAD_DATUM_SIDECAR_FIELDS["policy_eligible_datums"]: torch.tensor(
                [policy_eligible]
            ),
            rl_module._WORKLOAD_DATUM_SIDECAR_FIELDS["post_sampling_datums"]: torch.tensor(
                [post_sampling]
            ),
            rl_module._WORKLOAD_TASK_RETAINED_DATUMS_KEY: torch.tensor([task_retained]),
        }
        return item

    raw_batch = [
        task_item(
            1.0,
            steps=10,
            calls=5,
            inputs=100,
            outputs=20,
            trajectories=3,
            requested=8,
            observed=7,
            trainable=6,
            postmerge=9,
            policy_eligible=8,
            post_sampling=6,
            task_retained=6,
        ),
        task_item(
            0.0,
            steps=14,
            calls=9,
            inputs=200,
            outputs=30,
            trajectories=4,
            requested=8,
            observed=8,
            trainable=5,
            postmerge=8,
            policy_eligible=7,
            post_sampling=5,
            task_retained=4,
        ),
    ]

    summary = rl_module._extract_accepted_batch_workload(raw_batch)
    assert summary is not None
    assert summary.tasks == 2
    assert summary.requested_rollouts == 16
    assert summary.observed_rollouts == 15
    assert summary.trainable_rollouts == 11
    assert summary.task_retained_datums == 10
    assert summary.workload == rl_module.RolloutWorkload(
        environment_steps=24,
        model_calls=14,
        input_tokens=300,
        output_tokens=50,
        trajectories=7,
        postmerge_datums=17,
        policy_eligible_datums=15,
        post_sampling_datums=11,
    )

    reduced = trainer._reduce_rollout_batch(raw_batch)
    assert reduced is not None
    assert torch.equal(reduced["rewards"], torch.tensor([[1.0], [0.0]]))
    assert not any(key.startswith(rl_module._WORKLOAD_SIDECAR_PREFIX) for key in reduced)


def test_submitted_training_batch_metrics_count_post_filter_masks():
    batch_transforms = _load_batch_transforms_module()
    rl_module = _load_trainer_module(batch_transforms)

    metrics = rl_module._training_batch_workload_metrics(
        [
            {
                "attention_mask": torch.tensor([[True, True, True, False]]),
                "loss_mask": torch.tensor([[False, True, True, True]]),
            },
            {
                "attention_mask": torch.tensor([[True, True]]),
                "loss_mask": torch.tensor([[False, True]]),
            },
        ],
        total_postmerge_datums=7,
    )

    assert metrics == {
        "workload/training_batch/total_submitted_datums": 2.0,
        "workload/training_batch/total_attention_tokens": 5.0,
        "workload/training_batch/total_action_tokens": 3.0,
        "workload/training_batch/total_non_submitted_datums": 5.0,
    }


def test_submitted_training_batch_metrics_report_an_explicit_empty_batch():
    batch_transforms = _load_batch_transforms_module()
    rl_module = _load_trainer_module(batch_transforms)

    assert rl_module._training_batch_workload_metrics(
        [],
        total_postmerge_datums=7,
    ) == {
        "workload/training_batch/total_submitted_datums": 0.0,
        "workload/training_batch/total_attention_tokens": 0.0,
        "workload/training_batch/total_action_tokens": 0.0,
        "workload/training_batch/total_non_submitted_datums": 7.0,
    }


def test_maybe_shuffle_and_trim_localizes_before_inferring_batch_size():
    batch_transforms = _load_batch_transforms_module()
    rl_module = _load_trainer_module(batch_transforms)

    class LocalTensor:
        def __init__(self, tensor):
            self.tensor = tensor

        @property
        def shape(self):
            return self.tensor.shape

        @property
        def ndim(self):
            return self.tensor.ndim

        def to_local(self):
            return self.tensor

    trainer = rl_module.PlatoonArealRLTrainer.__new__(rl_module.PlatoonArealRLTrainer)
    trainer.actor = SimpleNamespace(data_parallel_world_size=1)
    trainer.config = SimpleNamespace(
        rollout=SimpleNamespace(shuffle_cross_task=False, ensure_batch_divisible_by=1),
    )

    processed = trainer._maybe_shuffle_and_trim_batch(
        {
            "attention_mask": LocalTensor(torch.ones(2, 3, dtype=torch.bool)),
            "rewards": LocalTensor(torch.tensor([[1.0], [2.0]])),
        }
    )

    assert processed is not None
    assert torch.equal(processed["attention_mask"], torch.ones(2, 3, dtype=torch.bool))
    assert torch.equal(processed["rewards"], torch.tensor([[1.0], [2.0]]))


def test_split_batch_to_trajectories_restores_dp_dispatch_shape():
    batch_transforms = _load_batch_transforms_module()

    batch = {
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 0],
                [1, 1, 0, 0],
            ],
            dtype=torch.bool,
        ),
        "rewards": torch.tensor([[1.0], [2.0]]),
        "logprobs": torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.0],
                [0.4, 0.5, 0.0, 0.0],
            ]
        ),
        "meta": ["a", "b"],
    }

    split = batch_transforms.split_batch_to_trajectories(batch)

    assert len(split) == 2
    assert torch.equal(split[0]["attention_mask"], torch.tensor([[1, 1, 1]], dtype=torch.bool))
    assert torch.equal(split[1]["attention_mask"], torch.tensor([[1, 1]], dtype=torch.bool))
    assert torch.equal(split[0]["logprobs"], torch.tensor([[0.1, 0.2, 0.3]]))
    assert torch.equal(split[1]["logprobs"], torch.tensor([[0.4, 0.5]]))
    assert split[0]["meta"] == "a"
    assert split[1]["meta"] == "b"


def test_split_batch_to_trajectories_trims_routed_experts_on_sequence_dim():
    batch_transforms = _load_batch_transforms_module()
    routes = torch.arange(2 * 5 * 3 * 2, dtype=torch.uint8).reshape(2, 5, 3, 2)
    valid = torch.tensor(
        [
            [True, True, True, False, False],
            [True, False, False, False, False],
        ]
    )
    batch = {
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 0],
                [1, 1, 0, 0, 0],
            ],
            dtype=torch.bool,
        ),
        "routed_experts": routes,
        "routed_experts_valid": valid,
    }

    split = batch_transforms.split_batch_to_trajectories(batch)

    assert split[0]["routed_experts"].shape == (1, 4, 3, 2)
    assert split[1]["routed_experts"].shape == (1, 2, 3, 2)
    torch.testing.assert_close(split[0]["routed_experts"], routes[0:1, :4])
    torch.testing.assert_close(split[1]["routed_experts"], routes[1:2, :2])
    torch.testing.assert_close(split[0]["routed_experts_valid"], valid[0:1, :4])
    torch.testing.assert_close(split[1]["routed_experts_valid"], valid[1:2, :2])


def test_router_replay_sidechannels_skip_compute_and_reattach_exact_tensors():
    batch_transforms = _load_batch_transforms_module()
    rl_module = _load_trainer_module(batch_transforms)
    routes = [
        torch.arange(1 * 3 * 2 * 1, dtype=torch.uint8).reshape(1, 3, 2, 1),
        torch.arange(1 * 2 * 2 * 1, dtype=torch.uint8).reshape(1, 2, 2, 1),
    ]
    valid = [
        torch.tensor([[True, True, False]]),
        torch.tensor([[True, False]]),
    ]
    trajectories = [
        {
            "input_ids": torch.tensor([[10, 11, 12]]),
            "routed_experts": routes[0],
            "routed_experts_valid": valid[0],
        },
        {
            "input_ids": torch.tensor([[20, 21]]),
            "routed_experts": routes[1],
            "routed_experts_valid": valid[1],
        },
    ]

    compute_batch, sidechannels = rl_module._detach_router_replay_sidechannels(trajectories)

    assert sidechannels is not None
    assert all("routed_experts" not in item for item in compute_batch)
    assert compute_batch[0]["input_ids"] is trajectories[0]["input_ids"]
    marked = rl_module._add_router_replay_order_markers(compute_batch, sidechannels)
    advantages = [dict(item, advantages=torch.ones_like(item["input_ids"])) for item in marked]
    restored, marker_cleanup = rl_module._reattach_router_replay_sidechannels(advantages, sidechannels)

    assert marker_cleanup is not None
    assert len(marker_cleanup) == 2
    assert restored[0]["routed_experts"] is routes[0]
    assert restored[0]["routed_experts_valid"] is valid[0]
    assert restored[1]["routed_experts"] is routes[1]
    assert restored[1]["routed_experts_valid"] is valid[1]
    assert rl_module._ROUTER_REPLAY_ORDER_FIELD not in restored[0]


def test_router_replay_sidechannels_fail_closed_on_partial_or_reordered_data():
    batch_transforms = _load_batch_transforms_module()
    rl_module = _load_trainer_module(batch_transforms)
    route = torch.zeros(1, 2, 1, 1, dtype=torch.uint8)
    valid = torch.tensor([[True, False]])

    with pytest.raises(RuntimeError, match="both routed_experts"):
        rl_module._detach_router_replay_sidechannels([{"routed_experts": route}])

    trajectories = [
        {"input_ids": torch.tensor([[1, 2]]), "routed_experts": route, "routed_experts_valid": valid},
        {"input_ids": torch.tensor([[3, 4]]), "routed_experts": route.clone(), "routed_experts_valid": valid.clone()},
    ]
    compute_batch, sidechannels = rl_module._detach_router_replay_sidechannels(trajectories)
    marked = rl_module._add_router_replay_order_markers(compute_batch, sidechannels)

    with pytest.raises(RuntimeError, match="order changed"):
        rl_module._reattach_router_replay_sidechannels(list(reversed(marked)), sidechannels)


def test_router_replay_order_marker_handle_remains_visible_for_cleanup():
    batch_transforms = _load_batch_transforms_module()
    rl_module = _load_trainer_module(batch_transforms)
    routes = torch.zeros(1, 2, 1, 1, dtype=torch.uint8)
    valid = torch.tensor([[True, False]])
    sidechannels = [(routes, valid)]

    class FakeRTensor:
        def __init__(self, value):
            self.value = value

        def to_local(self):
            return self.value

    raw_marker = FakeRTensor(torch.tensor([0]))
    advantages = [
        {
            "input_ids": torch.tensor([[1, 2]]),
            rl_module._ROUTER_REPLAY_ORDER_FIELD: raw_marker,
        }
    ]

    restored, marker_cleanup = rl_module._reattach_router_replay_sidechannels(advantages, sidechannels)

    assert marker_cleanup == [raw_marker]
    assert marker_cleanup[0] is raw_marker
    assert rl_module._ROUTER_REPLAY_ORDER_FIELD not in restored[0]
    assert restored[0]["routed_experts"] is routes


def test_router_replay_reattach_baseline_returns_no_cleanup_target():
    batch_transforms = _load_batch_transforms_module()
    rl_module = _load_trainer_module(batch_transforms)
    trajectories = [{"input_ids": torch.tensor([[1, 2]])}]

    restored, marker_cleanup = rl_module._reattach_router_replay_sidechannels(trajectories, None)

    assert restored is trajectories
    assert marker_cleanup is None


def test_batch_cleanup_keeps_raw_rollout_and_marker_handles_visible():
    batch_transforms = _load_batch_transforms_module()
    rl_module = _load_trainer_module(batch_transforms)
    raw_handles = object()
    processed = object()
    advantages = object()
    marker_handles = object()
    observed = []
    actor = SimpleNamespace(clear_batches=lambda *targets: observed.extend(targets))

    targets = rl_module._batch_cleanup_targets(
        raw_handles,
        processed,
        advantages,
        marker_handles,
    )
    actor.clear_batches(*targets)

    assert observed == [raw_handles, processed, advantages, marker_handles]

    # Filtering the processed batch must not hide the original RTensor shards.
    filtered_targets = rl_module._batch_cleanup_targets(raw_handles, None, None, None)
    assert filtered_targets == (raw_handles,)


def _zero_filter_batch(rewards, loss_masks):
    width = len(loss_masks[0])
    return {
        "datum_id": torch.arange(len(rewards)),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "loss_mask": torch.tensor(loss_masks, dtype=torch.bool),
        "attention_mask": torch.ones(len(rewards), width, dtype=torch.bool),
    }


def test_early_global_zero_reward_filter_is_exact_and_preserves_minimum_dp_padding():
    batch_transforms = _load_batch_transforms_module()
    rl_module = _load_trainer_module(batch_transforms)
    batch = _zero_filter_batch(
        [1.0, 2.0, 3.0, 4.0, 5.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [[True]] * 12,
    )
    # Tiny nonzero signal must never be classified by tolerance.
    batch["rewards"][0] = 1e-12
    torch.manual_seed(11)

    retained, metrics = rl_module._filter_zero_centered_reward_batch(
        batch,
        dispatch_dp_size=4,
    )

    assert retained is not None
    assert retained["datum_id"].shape[0] == 8
    assert set(range(5)).issubset(set(retained["datum_id"].tolist()))
    assert metrics["zero_padding_datums"] == 3.0
    assert metrics["filtered_zero_advantage_datums"] == 4.0
    assert metrics["policy_gradient_denominator_scale"] == pytest.approx(2 / 3)
    tiny_index = retained["datum_id"].tolist().index(0)
    assert retained["rewards"][tiny_index].item() == pytest.approx(1e-12 * 2 / 3)


def test_early_global_zero_reward_filter_all_zero_returns_no_batch():
    batch_transforms = _load_batch_transforms_module()
    rl_module = _load_trainer_module(batch_transforms)

    retained, metrics = rl_module._filter_zero_centered_reward_batch(
        _zero_filter_batch([0.0] * 4, [[True, True]] * 4),
        dispatch_dp_size=2,
    )

    assert retained is None
    assert metrics["filtered_zero_advantage_datums"] == 4.0
    assert metrics["filtered_zero_advantage_loss_tokens"] == 8.0
    assert metrics["retained_datums"] == 0.0


def test_postprocess_scales_denominator_after_dp_trim_and_before_dispatch(monkeypatch):
    batch_transforms = _load_batch_transforms_module()
    rl_module = _load_trainer_module(batch_transforms)
    trainer = rl_module.PlatoonArealRLTrainer.__new__(rl_module.PlatoonArealRLTrainer)
    trainer.actor = SimpleNamespace(data_parallel_world_size=2)
    trainer.config = SimpleNamespace(
        rollout=SimpleNamespace(shuffle_cross_task=False, ensure_batch_divisible_by=2),
        workflow_config=SimpleNamespace(
            filter_zero_advantage_datums=True,
            depth_level_weighting=False,
            depth_level_discount_gamma=None,
        ),
    )
    seen_before_filter = {}

    def multiplicative_transform(batch, context):
        _ = context
        seen_before_filter["datum_ids"] = batch["datum_id"].clone()
        seen_before_filter["rewards"] = batch["rewards"].clone()
        batch["rewards"] = batch["rewards"] * 1.0
        return batch

    trainer.batch_transforms = [multiplicative_transform]
    # Global divisibility trim removes signal datum 4 first. The two zero
    # candidates remain visible to the transform, then are removed. Their four
    # action tokens are represented alongside two retained action tokens, so
    # rewards are scaled by 2/(2+4), not 2/(2+4+trimmed_signal_token).
    monkeypatch.setattr(
        torch,
        "randperm",
        lambda size, device=None: torch.tensor([4, 0, 1, 2, 3], device=device)[:size],
    )
    batch = _zero_filter_batch(
        [0.0, 0.0, 1.0, 2.0, 3.0],
        [[True, True], [True, True], [True, False], [True, False], [True, False]],
    )

    processed = trainer._postprocess_rollout_batch(
        [batch],
        global_step=0,
        epoch=0,
        epoch_step=0,
    )

    assert seen_before_filter["datum_ids"].tolist() == [0, 1, 2, 3]
    assert seen_before_filter["rewards"].tolist() == [0.0, 0.0, 1.0, 2.0]
    assert processed is not None
    assert [item["datum_id"].item() for item in processed] == [2, 3]
    torch.testing.assert_close(
        torch.tensor([item["rewards"].item() for item in processed]),
        torch.tensor([1 / 3, 2 / 3]),
    )


def test_postprocess_disabled_keeps_exact_zero_rewards():
    batch_transforms = _load_batch_transforms_module()
    rl_module = _load_trainer_module(batch_transforms)
    trainer = rl_module.PlatoonArealRLTrainer.__new__(rl_module.PlatoonArealRLTrainer)
    trainer.actor = SimpleNamespace(data_parallel_world_size=1)
    trainer.config = SimpleNamespace(
        rollout=SimpleNamespace(shuffle_cross_task=False, ensure_batch_divisible_by=1),
        workflow_config=SimpleNamespace(
            filter_zero_advantage_datums=False,
            depth_level_weighting=False,
            depth_level_discount_gamma=None,
        ),
    )
    trainer.batch_transforms = []

    processed = trainer._postprocess_rollout_batch(
        [_zero_filter_batch([0.0, 1.0], [[True], [True]])],
        global_step=0,
        epoch=0,
        epoch_step=0,
    )

    assert processed is not None
    assert [item["rewards"].item() for item in processed] == [0.0, 1.0]


def test_zero_reward_filter_warning_lists_incompatible_objectives():
    batch_transforms = _load_batch_transforms_module()
    rl_module = _load_trainer_module(batch_transforms)
    config = SimpleNamespace(
        workflow_config=SimpleNamespace(filter_zero_advantage_datums=True),
        actor=SimpleNamespace(
            path="Qwen/Qwen3.6-35B-A3B",
            kl_ctl=0.1,
            reward_bias=0.2,
            reward_norm=SimpleNamespace(mean_level="batch", std_level=None),
            adv_norm=None,
            overlong_reward_penalty=True,
            megatron=SimpleNamespace(bridge_type="megatron-bridge"),
        ),
        critic=object(),
        teacher=object(),
    )

    with pytest.warns(RuntimeWarning, match="Disable it when KL is nonzero") as caught:
        rl_module._warn_for_zero_reward_filter_assumptions(
            config,
            custom_batch_transforms=[lambda batch, context: batch],
        )

    message = str(caught[0].message)
    assert "actor.kl_ctl != 0" in message
    assert "critic objective is present" in message
    assert "teacher/distillation objective is present" in message
    assert "independent global router auxiliary loss" in message
    assert "custom batch transforms" in message


def test_trainer_has_no_post_advantage_zero_scan():
    batch_transforms = _load_batch_transforms_module()
    rl_module = _load_trainer_module(batch_transforms)

    assert not hasattr(rl_module, "_filter_zero_advantage_datums")
    source = inspect.getsource(rl_module._filter_zero_centered_reward_batch)
    assert "advantages" not in source


def test_advance_logical_versions_updates_every_engine_without_weight_operations():
    batch_transforms = _load_batch_transforms_module()
    rl_module = _load_trainer_module(batch_transforms)
    trainer = rl_module.PlatoonArealRLTrainer.__new__(rl_module.PlatoonArealRLTrainer)
    observed: dict[str, list[int]] = {}

    def endpoint(name):
        return SimpleNamespace(set_version=lambda version: observed.setdefault(name, []).append(version))

    trainer.actor = endpoint("actor")
    trainer.critic = endpoint("critic")
    trainer.ref = endpoint("ref")
    trainer.teacher = endpoint("teacher")
    trainer.rollout = endpoint("rollout")
    trainer.eval_rollout = endpoint("eval_rollout")

    trainer._advance_logical_versions(17)

    assert observed == {
        "actor": [17],
        "critic": [17],
        "ref": [17],
        "teacher": [17],
        "rollout": [17],
        "eval_rollout": [17],
    }


def test_deadline_drain_forces_latest_completed_step_recoverable(monkeypatch, tmp_path):
    batch_transforms = _load_batch_transforms_module()
    rl_module = _load_trainer_module(batch_transforms)

    class FakeStepInfo:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    dumped = {}

    class FakeRecoverInfo:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def dump(self, path):
            dumped["path"] = path
            dumped["info"] = self

    sys.modules["areal.api"].StepInfo = FakeStepInfo
    recover_module = types.ModuleType("areal.utils.recover")
    recover_module.RecoverInfo = FakeRecoverInfo
    monkeypatch.setitem(sys.modules, "areal.utils.recover", recover_module)

    saved = []
    handler = SimpleNamespace(
        config=SimpleNamespace(
            mode="auto",
            experiment_name="experiment",
            trial_name="trial",
            fileroot=str(tmp_path),
        ),
        last_step_info=FakeStepInfo(global_step=3),
        freq_ctl=SimpleNamespace(state_dict=lambda: {"frequency": "preserved"}),
        _ensure_recover_supported=lambda engines: None,
        _normalize_recover_engines=lambda engines: engines,
        _save_checkpoint=lambda engine, **kwargs: saved.append((engine, kwargs)),
        recover_info_path=lambda *args: str(tmp_path / "recover_info"),
    )

    trainer = rl_module.PlatoonArealRLTrainer.__new__(rl_module.PlatoonArealRLTrainer)
    trainer.recover_handler = handler
    trainer.actor = object()
    trainer.critic = None
    trainer.tokenizer = object()
    trainer.processor = object()
    trainer.saver = SimpleNamespace(state_dict=lambda: {"saver": 1})
    trainer.evaluator = SimpleNamespace(state_dict=lambda: {"evaluator": 2})
    trainer.stats_logger = SimpleNamespace(state_dict=lambda: {"logger": 3})

    # Special methods are resolved on the type, so use a tiny concrete fake
    # rather than SimpleNamespace for len().
    class FakeDataloader:
        def __len__(self):
            return 10

        def state_dict(self):
            return {"dataloader": 4}

    trainer.train_dataloader = FakeDataloader()

    assert trainer._ensure_recover_checkpoint_at(
        epoch=0,
        epoch_step=4,
        global_step=4,
    )
    assert len(saved) == 1
    assert saved[0][0] is trainer.actor
    assert saved[0][1]["name"] == "default"
    assert handler.last_step_info.global_step == 4
    assert dumped["path"] == str(tmp_path / "recover_info")
    assert dumped["info"].checkpoint_info == {"frequency": "preserved"}
    assert dumped["info"].dataloader_info == {"dataloader": 4}

    # Rechecking the same boundary is idempotent and does not rewrite DCP.
    assert not trainer._ensure_recover_checkpoint_at(
        epoch=0,
        epoch_step=4,
        global_step=4,
    )
    assert len(saved) == 1
