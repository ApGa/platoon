"""Depth strategies for subagent datums: inverse-frequency weighting and gamma**depth discounting."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import tinker
import torch
from tinker import TensorData

from platoon.train.config_defs import EvalConfig, PlatoonTinkerRLTrainerConfig, TrainConfig, WorkflowConfig
from platoon.train.step.batch_transforms import (
    BatchTransformContext,
    DepthLevelDiscountTransform,
    DepthLevelWeightingTransform,
    build_default_batch_transforms,
)
from platoon.train.workflow.group import GroupRolloutWorkflow

CONTEXT = BatchTransformContext(config=None, train_step=0, minibatch_num=0, microbatch_num=0)


def _datum(*, depth: int | None, mask: list[float], start: bool = True) -> tinker.Datum:
    loss_fn_inputs = {
        "mask": TensorData.from_torch(torch.tensor(mask)),
        "advantages": TensorData.from_torch(torch.ones(len(mask))),
    }
    if depth is not None:
        loss_fn_inputs["traj_depth"] = TensorData.from_torch(torch.tensor([depth]))
        loss_fn_inputs["traj_start"] = TensorData.from_torch(torch.tensor([float(start)]))
    return tinker.Datum(model_input=tinker.ModelInput.from_ints(list(range(len(mask)))), loss_fn_inputs=loss_fn_inputs)


def _weights(datums: list[tinker.Datum]) -> list[float]:
    """Each datum's advantage multiplier on its first action token (advantages start at one)."""

    return [float(datum.loss_fn_inputs["advantages"].to_torch()[0]) for datum in datums]


def _weighted_action_tokens(datums: list[tinker.Datum]) -> float:
    return sum(
        float((datum.loss_fn_inputs["advantages"].to_torch() * datum.loss_fn_inputs["mask"].to_torch()).sum())
        for datum in datums
    )


def _trainer_config(**workflow: object) -> PlatoonTinkerRLTrainerConfig:
    return PlatoonTinkerRLTrainerConfig(
        train=TrainConfig(model_name="m", renderer_name="r", workflow_config=WorkflowConfig(**workflow)),
        eval=EvalConfig(),
        log_path="logs",
    )


def test_inverse_frequency_weighting_balances_depth_levels_and_keeps_token_mass():
    # One root trajectory (2 action tokens) and two child trajectories (2 + 1 + 1 tokens).
    datums = [
        _datum(depth=0, mask=[1.0, 1.0]),
        _datum(depth=1, mask=[1.0, 0.0, 1.0]),
        _datum(depth=1, mask=[1.0], start=False),
        _datum(depth=1, mask=[1.0]),
    ]

    DepthLevelWeightingTransform()(datums, CONTEXT)

    # raw weights 1/1 (depth 0) and 1/2 (depth 1); 6 tokens / (2 * 1 + 4 * 0.5) = 1.5.
    assert _weights(datums) == pytest.approx([1.5, 0.75, 0.75, 0.75])
    assert _weighted_action_tokens(datums) == pytest.approx(6.0)
    # Non-action positions keep their advantage scaled like the rest of the datum.
    assert torch.allclose(datums[1].loss_fn_inputs["advantages"].to_torch(), torch.full((3,), 0.75))


def test_discount_scales_by_gamma_power_and_keeps_token_mass():
    datums = [
        _datum(depth=0, mask=[1.0, 1.0]),
        _datum(depth=1, mask=[1.0, 0.0]),
        _datum(depth=2, mask=[1.0]),
    ]

    assert DepthLevelDiscountTransform(0.5)(datums, CONTEXT) is datums

    # raw 1, 0.5, 0.25 over 2, 1, 1 action tokens: scale = 4 / 2.75.
    scale = 4 / 2.75
    assert _weights(datums) == pytest.approx([scale, 0.5 * scale, 0.25 * scale])
    assert _weighted_action_tokens(datums) == pytest.approx(4.0)


def test_discount_of_one_is_the_identity():
    datums = [_datum(depth=0, mask=[1.0]), _datum(depth=3, mask=[1.0, 1.0])]

    DepthLevelDiscountTransform(1.0)(datums, CONTEXT)

    assert _weights(datums) == [1.0, 1.0]


def test_discount_requires_depth_labels():
    with pytest.raises(ValueError, match="traj_depth"):
        DepthLevelDiscountTransform(0.5)([_datum(depth=None, mask=[1.0])], CONTEXT)


@pytest.mark.parametrize("gamma", [0.0, -0.5, math.inf, math.nan])
def test_discount_transform_rejects_non_positive_or_non_finite_gamma(gamma):
    with pytest.raises(ValueError, match="positive finite"):
        DepthLevelDiscountTransform(gamma)


@pytest.mark.parametrize(
    ("workflow", "expected"),
    [
        ({}, []),
        ({"depth_level_weighting": True}, [DepthLevelWeightingTransform]),
        ({"depth_level_discount_gamma": 0.5}, [DepthLevelDiscountTransform]),
    ],
)
def test_default_transforms_follow_the_configured_depth_strategy(workflow, expected):
    transforms = build_default_batch_transforms(_trainer_config(**workflow))

    assert [type(transform) for transform in transforms] == expected


def test_default_discount_transform_uses_the_configured_gamma():
    (transform,) = build_default_batch_transforms(_trainer_config(depth_level_discount_gamma=0.25))

    assert transform.gamma == 0.25


@pytest.mark.parametrize("gamma", [0, -1.0, math.inf, math.nan, True, "0.5"])
def test_workflow_config_rejects_invalid_gamma(gamma):
    with pytest.raises(ValueError, match="depth_level_discount_gamma"):
        WorkflowConfig(depth_level_discount_gamma=gamma)


def test_workflow_config_rejects_both_depth_strategies():
    with pytest.raises(ValueError, match="alternatives"):
        WorkflowConfig(depth_level_weighting=True, depth_level_discount_gamma=0.5)


@pytest.mark.parametrize(
    ("workflow", "depth", "start"),
    [
        ({}, False, False),
        ({"depth_level_weighting": True}, True, True),
        ({"depth_level_discount_gamma": 0.5}, True, False),
    ],
)
def test_worker_request_asks_for_the_labels_its_depth_strategy_needs(workflow, depth, start):
    async def rollout(task, config):
        raise AssertionError("not run")

    group = GroupRolloutWorkflow(
        rollout_fn=rollout,
        get_task_fn=lambda task_id: SimpleNamespace(id=task_id, max_steps=None),
        config=WorkflowConfig(**workflow),
        model_info=SimpleNamespace(
            model_name="m",
            renderer_name="qwen3_instruct",
            renderer_kwargs={},
            llm=SimpleNamespace(version=0),
        ),
    )

    conversion = group._build_worker_request({"task_id": "task"}, 0).conversion

    assert (conversion.include_traj_depth, conversion.include_traj_start) == (depth, start)
