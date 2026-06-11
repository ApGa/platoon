"""Focused regression tests for Platoon's trainer-side AReaL batch transforms."""

from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

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
    utils_mod.stats_tracker = SimpleNamespace(record_timing=_null_context)
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
