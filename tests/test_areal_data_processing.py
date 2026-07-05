from __future__ import annotations

import pytest
import torch

from platoon.utils.areal_data_processing import get_train_data_for_trajectory


class _Completion:
    def __init__(self, prompt_tokens: list[int], output_tokens: list[int], version: int):
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.version = version

    def to_tensor_dict(self) -> dict[str, torch.Tensor]:
        input_ids = self.prompt_tokens + self.output_tokens
        prompt_len = len(self.prompt_tokens)
        output_len = len(self.output_tokens)
        return {
            "input_ids": torch.tensor([input_ids]),
            "loss_mask": torch.tensor([[0] * prompt_len + [1] * output_len]),
            "logprobs": torch.tensor([[0.0] * prompt_len + [-0.1] * output_len]),
            "versions": torch.tensor([[-1] * prompt_len + [self.version] * output_len]),
        }


def _step(completion_id: str, *, error: str | None = None) -> dict:
    step = {"misc": {"action_misc": {"completion_id": completion_id}}}
    if error is not None:
        step["error"] = error
    return step


def _concat_same_length(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Minimal concat helper for fixed-length completion fixtures."""

    return {key: torch.cat([item[key] for item in items], dim=0) for key in items[0]}


@pytest.mark.parametrize("merge_prefixes", [True, False])
def test_repeated_completion_ids_contribute_training_tokens_once(merge_prefixes: bool):
    completions = {
        "completion-a": _Completion([10, 11], [12, 13], version=3),
        "completion-b": _Completion([20, 21], [22, 23], version=4),
    }
    trajectory = {
        "reward": 0.75,
        # Cover both adjacent and non-adjacent repeats while retaining the raw
        # environment-step count for rollout metrics.
        "steps": [
            _step("completion-a"),
            _step("completion-a"),
            _step("completion-b"),
            _step("completion-a"),
        ],
    }

    train_data = get_train_data_for_trajectory(
        trajectory,
        completions,
        task_id="task",
        trajectory_id="trajectory",
        merge_prefixes=merge_prefixes,
        concat_fn=_concat_same_length,
    )

    assert train_data is not None
    assert torch.equal(
        train_data["input_ids"],
        torch.tensor(
            [
                [10, 11, 12, 13],
                [20, 21, 22, 23],
            ]
        ),
    )
    assert train_data["loss_mask"].sum().item() == 4
    assert torch.equal(train_data["num_input_tokens"], torch.tensor([4.0]))
    assert torch.equal(train_data["num_output_tokens"], torch.tensor([4.0]))
    assert torch.equal(train_data["num_steps"], torch.tensor([4.0]))


@pytest.mark.parametrize("merge_prefixes", [True, False])
def test_distinct_completion_ids_with_identical_tokens_are_not_deduplicated(merge_prefixes: bool):
    shared_tokens = _Completion([10, 11], [12, 13], version=3)
    trajectory = {
        "reward": 0.5,
        "steps": [_step("completion-a"), _step("completion-b")],
    }

    train_data = get_train_data_for_trajectory(
        trajectory,
        {"completion-a": shared_tokens, "completion-b": shared_tokens},
        task_id="task",
        trajectory_id="trajectory",
        merge_prefixes=merge_prefixes,
        concat_fn=_concat_same_length,
    )

    assert train_data is not None
    assert train_data["input_ids"].shape[0] == 2
    assert train_data["loss_mask"].sum().item() == 4
    assert torch.equal(train_data["num_output_tokens"], torch.tensor([4.0]))
    assert torch.equal(train_data["num_steps"], torch.tensor([2.0]))


@pytest.mark.parametrize("merge_prefixes", [True, False])
def test_filtered_occurrence_does_not_suppress_later_eligible_duplicate(merge_prefixes: bool):
    trajectory = {
        "reward": 1.0,
        "steps": [
            _step("completion-a", error="parallel action failed"),
            _step("completion-a"),
        ],
    }

    train_data = get_train_data_for_trajectory(
        trajectory,
        {"completion-a": _Completion([10, 11], [12, 13], version=3)},
        task_id="task",
        trajectory_id="trajectory",
        filter_errors=True,
        merge_prefixes=merge_prefixes,
        concat_fn=_concat_same_length,
    )

    assert train_data is not None
    assert train_data["input_ids"].shape[0] == 1
    assert train_data["loss_mask"].sum().item() == 2
    assert torch.equal(train_data["num_output_tokens"], torch.tensor([2.0]))
    assert torch.equal(train_data["num_steps"], torch.tensor([2.0]))
