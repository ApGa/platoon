from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from platoon.agents.actions.subagent import (
    EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY,
    SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY,
)
from platoon.utils.areal_data_processing import (
    POLICY_TRAINING_ELIGIBILITY_MASK_KEY,
    SUBAGENT_DATUM_DEPTH_KEY,
    SUBAGENT_DATUM_KEEP_MASK_KEY,
    RouterReplayConfig,
    get_train_data_for_step,
    get_train_data_for_trajectory,
    get_train_data_for_trajectory_collection,
    harmonize_optional_reward_metrics,
    reward_metric_presence_key,
)
from platoon.utils.trajectory_status import TRAJECTORY_CANCELLED_MISC_KEY


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


class _RoutedCompletion(_Completion):
    def __init__(
        self,
        prompt_tokens: list[int],
        output_tokens: list[int],
        version: int,
        routed_experts: torch.Tensor | None,
        routed_experts_valid: torch.Tensor | None = None,
        *,
        include_validity: bool = True,
    ):
        super().__init__(prompt_tokens, output_tokens, version)
        self.routed_experts = routed_experts
        self.routed_experts_valid = routed_experts_valid
        self.include_validity = include_validity

    def to_tensor_dict(self) -> dict[str, torch.Tensor]:
        result = super().to_tensor_dict()
        if self.routed_experts is not None:
            result["routed_experts"] = self.routed_experts.unsqueeze(0)
            if self.include_validity:
                validity = self.routed_experts_valid
                if validity is None:
                    validity = torch.ones(self.routed_experts.shape[0], dtype=torch.bool)
                result["routed_experts_valid"] = validity.unsqueeze(0)
        return result


def _step(completion_id: str, *, error: str | None = None) -> dict:
    step = {"misc": {"action_misc": {"completion_id": completion_id}}}
    if error is not None:
        step["error"] = error
    return step


def _concat_same_length(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Minimal concat helper for fixed-length completion fixtures."""

    return {key: torch.cat([item[key] for item in items], dim=0) for key in items[0]}


def _concat_padded(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Small CPU-only equivalent of AReaL's concat helper for these fixtures."""

    result = {}
    for key in items[0]:
        values = [item[key] for item in items]
        max_shape = [max(value.shape[dim] for value in values) for dim in range(1, values[0].ndim)]
        padded = []
        for value in values:
            pad = []
            for dim in reversed(range(1, value.ndim)):
                pad.extend([0, max_shape[dim - 1] - value.shape[dim]])
            padded.append(F.pad(value, tuple(pad)))
        result[key] = torch.cat(padded, dim=0)
    return result


def _concat_strict(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Mirror AReaL's requirement that every input has identical keys."""

    expected = set(items[0])
    for item in items[1:]:
        if set(item) != expected:
            raise ValueError("different keys")
    return _concat_padded(items)


def _routes(rows: list[list[int]], *, num_layers: int = 2, topk: int = 2) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.int32).reshape(-1, num_layers, topk)


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


def test_router_replay_aligns_s_minus_one_rows_and_keeps_expert_zero_valid():
    config = RouterReplayConfig(num_layers=2, topk=2)
    raw_routes = _routes(
        [
            [0, 0, 0, 0],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]
    )
    completion = _RoutedCompletion([10, 11], [12, 13], 3, raw_routes)

    result = get_train_data_for_step(
        _step("completion-a"),
        {"completion-a": completion},
        "task",
        router_replay_config=config,
    )

    assert result is not None
    assert result["routed_experts"].shape == (1, 4, 2, 2)
    assert result["routed_experts"].dtype == torch.uint8
    torch.testing.assert_close(result["routed_experts"][0, :3], raw_routes.to(torch.uint8))
    # A row containing expert ID zero is still valid; only the causal terminal
    # row is invalid and zero-filled.
    assert torch.equal(
        result["routed_experts_valid"],
        torch.tensor([[True, True, True, False]]),
    )
    assert torch.count_nonzero(result["routed_experts"][0, 3]) == 0


def test_router_replay_prefix_merge_fills_old_terminal_and_aligns_tool_tokens():
    config = RouterReplayConfig(num_layers=2, topk=2)
    first_routes = _routes(
        [
            [1, 1, 1, 1],
            [2, 2, 2, 2],
        ]
    )
    second_routes = _routes(
        [
            [101, 101, 101, 101],
            [102, 102, 102, 102],
            [3, 3, 3, 3],
            [4, 4, 4, 4],
            [5, 5, 5, 5],
            [6, 6, 6, 6],
        ]
    )
    completions = {
        "first": _RoutedCompletion([10, 11], [12], 3, first_routes),
        # Tokens 50/51 model a tool result inserted between assistant turns.
        "second": _RoutedCompletion([10, 11, 12, 50, 51], [60, 61], 3, second_routes),
    }
    trajectory = {
        "reward": 1.0,
        "steps": [_step("first"), _step("second")],
    }

    result = get_train_data_for_trajectory(
        trajectory,
        completions,
        task_id="task",
        trajectory_id="trajectory",
        concat_fn=_concat_padded,
        router_replay_config=config,
    )

    assert result is not None
    assert torch.equal(result["input_ids"], torch.tensor([[10, 11, 12, 50, 51, 60, 61]]))
    # Existing valid rows remain attached to the first behavior completion;
    # only its formerly-terminal token is filled from the second request.
    expected_values = torch.tensor([1, 2, 3, 4, 5, 6, 0], dtype=torch.uint8)
    assert torch.equal(result["routed_experts"][0, :, 0, 0], expected_values)
    assert torch.equal(
        result["routed_experts_valid"],
        torch.tensor([[True, True, True, True, True, True, False]]),
    )


@pytest.mark.parametrize("merge_prefixes", [True, False])
def test_router_replay_duplicate_completion_id_is_not_duplicated(merge_prefixes: bool):
    config = RouterReplayConfig(num_layers=2, topk=2)
    completion = _RoutedCompletion(
        [10, 11],
        [12, 13],
        3,
        _routes([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]),
    )
    trajectory = {
        "reward": 1.0,
        "steps": [_step("completion-a"), _step("completion-a"), _step("completion-a")],
    }

    result = get_train_data_for_trajectory(
        trajectory,
        {"completion-a": completion},
        task_id="task",
        trajectory_id="trajectory",
        merge_prefixes=merge_prefixes,
        concat_fn=_concat_padded,
        router_replay_config=config,
    )

    assert result is not None
    assert result["input_ids"].shape == (1, 4)
    assert result["routed_experts"].shape == (1, 4, 2, 2)
    assert result["routed_experts_valid"].sum().item() == 3


def test_router_replay_non_prefix_flush_keeps_routes_with_each_datum():
    config = RouterReplayConfig(num_layers=2, topk=2)
    completions = {
        "a": _RoutedCompletion(
            [10, 11],
            [12, 13],
            3,
            _routes([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]),
        ),
        "b": _RoutedCompletion(
            [20, 21],
            [22, 23],
            3,
            _routes([[4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]]),
        ),
    }

    result = get_train_data_for_trajectory(
        {"reward": 1.0, "steps": [_step("a"), _step("b")]},
        completions,
        task_id="task",
        trajectory_id="trajectory",
        concat_fn=_concat_padded,
        router_replay_config=config,
    )

    assert result is not None
    assert result["routed_experts"].shape == (2, 4, 2, 2)
    assert torch.equal(result["routed_experts"][:, 0, 0, 0], torch.tensor([1, 4], dtype=torch.uint8))
    assert torch.equal(
        result["routed_experts_valid"],
        torch.tensor([[True, True, True, False], [True, True, True, False]]),
    )


def test_router_replay_excess_retry_rows_fail_before_training_not_truncate():
    config = RouterReplayConfig(num_layers=2, topk=2)
    # Four rows are ambiguous for a four-token sequence; SGLang's causal
    # contract requires exactly three.
    completion = _RoutedCompletion(
        [10, 11],
        [12, 13],
        3,
        _routes(
            [
                [1, 1, 1, 1],
                [2, 2, 2, 2],
                [3, 3, 3, 3],
                [4, 4, 4, 4],
            ]
        ),
    )

    with pytest.raises(ValueError, match="extra rows"):
        get_train_data_for_step(
            _step("completion-a"),
            {"completion-a": completion},
            "task",
            router_replay_config=config,
        )


def test_router_replay_deduplication_is_per_recursive_trajectory_not_global():
    config = RouterReplayConfig(num_layers=2, topk=2)
    completion = _RoutedCompletion(
        [10, 11],
        [12, 13],
        3,
        _routes([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]),
    )
    collection = {
        "trajectories": {
            "root": {"reward": 1.0, "steps": [_step("shared")]},
            "child": {
                "reward": 0.5,
                "parent_info": {"id": "root"},
                "steps": [_step("shared"), _step("shared")],
            },
        }
    }

    result = get_train_data_for_trajectory_collection(
        collection,
        {"shared": completion},
        task_id="task",
        concat_fn=_concat_padded,
        include_traj_depth=True,
        include_traj_start=True,
        router_replay_config=config,
    )

    assert result is not None
    # The repeated ID is once in each branch, while the duplicate occurrence
    # inside the child remains suppressed.
    assert result["routed_experts"].shape[0] == 2
    assert torch.equal(result["traj_depth"], torch.tensor([0, 1]))
    assert torch.equal(result["routed_experts_valid"].sum(dim=1), torch.tensor([3, 3]))


def test_router_replay_skips_reward_verifier_trajectories_before_route_processing():
    config = RouterReplayConfig(num_layers=2, topk=2)
    collection = {
        "trajectories": {
            "root": {"reward": 1.0, "steps": [_step("root-completion")]},
            "verifier": {
                "reward": 0.0,
                "parent_info": {"id": "root"},
                # Simulate a hard kill before the post-hoc trajectory marker
                # is written: the forked verifier task is tagged pre-launch.
                "task": {"misc": {SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY: True}},
                # Deliberately absent from completions: exclusion must happen
                # before token lookup and R3 route validation.
                "steps": [_step("verifier-completion")],
            },
        }
    }
    completion = _RoutedCompletion(
        [10, 11],
        [12, 13],
        3,
        _routes([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]),
    )

    result = get_train_data_for_trajectory_collection(
        collection,
        {"root-completion": completion},
        task_id="task",
        concat_fn=_concat_padded,
        include_traj_depth=True,
        include_traj_start=True,
        router_replay_config=config,
    )

    assert result is not None
    assert result["routed_experts"].shape[0] == 1
    assert torch.equal(result["traj_depth"], torch.tensor([0]))
    assert torch.equal(result["traj_start"], torch.tensor([1.0]))


def test_postmerge_subagent_sampling_repairs_start_and_keeps_r3_aligned():
    class FixedSampler:
        def sample_mask(self, *, task_id, trajectory_id, depth, num_datums):
            assert task_id == "task"
            if trajectory_id == "root":
                assert depth == 0 and num_datums == 1
                return [True]
            assert trajectory_id == "child"
            assert depth == 1 and num_datums == 2
            # There is deliberately no per-trajectory minimum on the original
            # first datum; the earliest retained datum becomes traj_start.
            return [False, True]

    config = RouterReplayConfig(num_layers=2, topk=2)
    completions = {
        "root-completion": _RoutedCompletion(
            [1, 2],
            [3, 4],
            3,
            _routes([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]),
        ),
        "child-a": _RoutedCompletion(
            [10, 11],
            [12, 13],
            3,
            _routes([[4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]]),
        ),
        # A non-prefix observation forces a second post-merge datum.
        "child-b": _RoutedCompletion(
            [20, 21],
            [22, 23],
            3,
            _routes([[7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]]),
        ),
    }
    collection = {
        "id": "collection",
        "trajectories": {
            "root": {"reward": 1.0, "steps": [_step("root-completion")]},
            "child": {
                "reward": 0.5,
                "parent_info": {"id": "root"},
                "steps": [_step("child-a"), _step("child-b")],
            },
        },
    }

    result = get_train_data_for_trajectory_collection(
        collection,
        completions,
        task_id="task",
        concat_fn=_concat_padded,
        include_traj_depth=True,
        include_traj_start=True,
        router_replay_config=config,
        subagent_datum_sampler=FixedSampler(),
    )

    assert result is not None
    assert torch.equal(result[SUBAGENT_DATUM_KEEP_MASK_KEY], torch.tensor([True, False, True]))
    assert torch.equal(result[SUBAGENT_DATUM_DEPTH_KEY], torch.tensor([0, 1, 1]))
    assert torch.equal(result[POLICY_TRAINING_ELIGIBILITY_MASK_KEY], torch.tensor([True, True, True]))
    assert torch.equal(result["traj_depth"], torch.tensor([0, 1, 1]))
    assert torch.equal(result["traj_start"], torch.tensor([1.0, 0.0, 1.0]))
    assert result["rewards"].shape[0] == 3
    assert result["routed_experts"].shape[0] == 3
    assert result["routed_experts_valid"].shape[0] == 3


def test_policy_excluded_child_stays_in_rewards_but_skips_sampler_and_policy_batch():
    class RecordingSampler:
        def __init__(self):
            self.calls = []

        def sample_mask(self, *, task_id, trajectory_id, depth, num_datums):
            self.calls.append((task_id, trajectory_id, depth, num_datums))
            return [True] * num_datums

    sampler = RecordingSampler()
    collection = {
        "trajectories": {
            # The child-only marker cannot make a malformed root ineligible.
            "root": {
                "reward": 1.0,
                "misc": {EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY: True},
                "steps": [_step("root-completion")],
            },
            "verifier-child": {
                "reward": 0.5,
                "misc": {EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY: True},
                "parent_info": {"id": "root"},
                "steps": [_step("child-completion")],
            },
        }
    }

    def reward_processor(trajectory):
        reward = float(trajectory["reward"])
        return reward, {"reward/raw": reward}

    result = get_train_data_for_trajectory_collection(
        collection,
        {
            "root-completion": _Completion([1, 2], [3, 4], version=3),
            "child-completion": _Completion([5, 6], [7, 8], version=3),
        },
        task_id="task",
        concat_fn=_concat_padded,
        reward_processor=reward_processor,
        include_traj_depth=True,
        include_traj_start=True,
        subagent_datum_sampler=sampler,
    )

    assert result is not None
    assert torch.equal(result[POLICY_TRAINING_ELIGIBILITY_MASK_KEY], torch.tensor([True, False]))
    assert torch.equal(result["task_reward_valid"], torch.tensor([True]))
    assert torch.equal(result[SUBAGENT_DATUM_KEEP_MASK_KEY], torch.tensor([True, True]))
    assert sampler.calls == [("task", "root", 0, 1)]
    # Policy exclusion is deliberately later than reward/delegation processing.
    assert torch.equal(result["reward/raw"], torch.tensor([1.0, 0.5]))
    assert torch.equal(result["root_reward/raw"], torch.tensor([1.0]))


def test_cancelled_root_is_ineligible_but_completed_child_remains_trainable():
    class RecordingSampler:
        def __init__(self):
            self.trajectory_ids = []

        def sample_mask(self, *, task_id, trajectory_id, depth, num_datums):
            _ = task_id, depth
            self.trajectory_ids.append(trajectory_id)
            return [True] * num_datums

    sampler = RecordingSampler()
    collection = {
        "trajectories": {
            "root": {
                "reward": 0.0,
                "error_message": "Episode cancelled at step 4\nCancelledError",
                "misc": {TRAJECTORY_CANCELLED_MISC_KEY: True},
                "steps": [_step("root-completion")],
            },
            "completed-child": {
                "reward": 0.75,
                "parent_info": {"id": "root"},
                "steps": [_step("child-completion")],
            },
        }
    }

    result = get_train_data_for_trajectory_collection(
        collection,
        {
            "root-completion": _Completion([1, 2], [3, 4], version=3),
            "child-completion": _Completion([5, 6], [7, 8], version=3),
        },
        task_id="task",
        concat_fn=_concat_padded,
        include_traj_depth=True,
        include_traj_start=True,
        subagent_datum_sampler=sampler,
    )

    assert result is not None
    assert torch.equal(result[POLICY_TRAINING_ELIGIBILITY_MASK_KEY], torch.tensor([False, True]))
    assert torch.equal(result["task_reward_valid"], torch.tensor([False]))
    assert sampler.trajectory_ids == ["completed-child"]
    # Cancellation is a policy filter, not a loss of rollout diagnostics.
    assert torch.equal(result["rewards"], torch.tensor([0.0, 0.75]))


def test_collection_zero_fills_optional_reward_metrics_for_unjudged_root():
    collection = {
        "trajectories": {
            "root": {"reward": 1.0, "steps": [_step("root-completion")]},
            "child": {
                "reward": 0.75,
                "judgment": 0.75,
                "parent_info": {"id": "root"},
                "steps": [_step("child-completion")],
            },
        }
    }

    def reward_processor(trajectory: dict) -> tuple[float, dict]:
        reward = float(trajectory["reward"])
        components = {"reward/success": reward}
        if "judgment" in trajectory:
            components["reward/subagent_judgment"] = float(trajectory["judgment"])
        return reward, components

    result = get_train_data_for_trajectory_collection(
        collection,
        {
            "root-completion": _Completion([10, 11], [12, 13], version=3),
            "child-completion": _Completion([20, 21], [22, 23], version=3),
        },
        task_id="task",
        reward_processor=reward_processor,
        concat_fn=_concat_strict,
    )

    assert result is not None
    assert torch.equal(result["reward/success"], torch.tensor([1.0, 0.75]))
    assert torch.equal(result["reward/subagent_judgment"], torch.tensor([0.0, 0.75]))
    assert torch.equal(
        result[reward_metric_presence_key("reward/subagent_judgment")],
        torch.tensor([False, True]),
    )


def test_reward_metric_harmonization_uses_each_group_results_own_length():
    results = [
        {
            "reward/success": torch.tensor([1.0, 0.75, 0.5]),
            "reward/subagent_judgment": torch.tensor([0.0, 0.75, 0.5]),
            reward_metric_presence_key("reward/subagent_judgment"): torch.tensor([False, True, True]),
            "root_reward/success": torch.tensor([1.0]),
            "root_reward/diagnostic": torch.tensor([2.0]),
            "structural": torch.ones(3),
        },
        {
            "reward/success": torch.tensor([0.25]),
            "root_reward/success": torch.tensor([0.25]),
            "structural": torch.ones(1),
        },
    ]

    harmonized = harmonize_optional_reward_metrics(results)

    assert torch.equal(
        harmonized[1]["reward/subagent_judgment"],
        torch.zeros(1),
    )
    judgment_mask_key = reward_metric_presence_key("reward/subagent_judgment")
    assert torch.equal(harmonized[0][judgment_mask_key], torch.tensor([False, True, True]))
    assert torch.equal(harmonized[1][judgment_mask_key], torch.tensor([False]))
    assert torch.equal(
        harmonized[1]["root_reward/diagnostic"],
        torch.zeros(1),
    )
    diagnostic_mask_key = reward_metric_presence_key("root_reward/diagnostic")
    assert torch.equal(harmonized[0][diagnostic_mask_key], torch.tensor([True]))
    assert torch.equal(harmonized[1][diagnostic_mask_key], torch.tensor([False]))
    assert harmonized[0]["reward/subagent_judgment"].shape == (3,)
    assert set(harmonized[0]) == set(harmonized[1])


def test_reward_metric_harmonization_preserves_a_real_zero_as_present():
    results = [
        {
            "reward/success": torch.tensor([0.0]),
            "reward/subagent_judgment": torch.tensor([0.0]),
        },
        {"reward/success": torch.tensor([1.0])},
    ]

    harmonized = harmonize_optional_reward_metrics(results)
    mask_key = reward_metric_presence_key("reward/subagent_judgment")

    assert torch.equal(harmonized[0][mask_key], torch.tensor([True]))
    assert torch.equal(harmonized[1][mask_key], torch.tensor([False]))


def test_collection_does_not_zero_fill_non_reward_key_mismatches():
    collection = {
        "trajectories": {
            "root": {"reward": 1.0, "steps": [_step("root-completion")]},
            "child": {
                "reward": 0.75,
                "has_diagnostic": True,
                "parent_info": {"id": "root"},
                "steps": [_step("child-completion")],
            },
        }
    }

    def reward_processor(trajectory: dict) -> tuple[float, dict]:
        components = {"reward/success": float(trajectory["reward"])}
        if trajectory.get("has_diagnostic"):
            components["diagnostic"] = 1.0
        return float(trajectory["reward"]), components

    with pytest.raises(ValueError, match="different keys"):
        get_train_data_for_trajectory_collection(
            collection,
            {
                "root-completion": _Completion([10, 11], [12, 13], version=3),
                "child-completion": _Completion([20, 21], [22, 23], version=3),
            },
            task_id="task",
            reward_processor=reward_processor,
            concat_fn=_concat_strict,
        )


def test_router_replay_requires_explicit_validity_and_qwen_width_and_id_range():
    config = RouterReplayConfig(num_layers=40, topk=8)
    base = torch.zeros((3, 40, 8), dtype=torch.int32)

    missing_validity = _RoutedCompletion(
        [10, 11],
        [12, 13],
        3,
        base,
        include_validity=False,
    )
    with pytest.raises(ValueError, match="explicit routed_experts_valid"):
        get_train_data_for_step(
            _step("c"),
            {"c": missing_validity},
            "task",
            router_replay_config=config,
        )

    bad_width = _RoutedCompletion(
        [10, 11],
        [12, 13],
        3,
        torch.zeros((3, 319), dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="width mismatch"):
        get_train_data_for_step(
            _step("c"),
            {"c": bad_width},
            "task",
            router_replay_config=config,
        )

    bad_id = base.clone()
    bad_id[0, 0, 0] = 256
    with pytest.raises(ValueError, match=r"must be in \[0, 255\]"):
        get_train_data_for_step(
            _step("c"),
            {"c": _RoutedCompletion([10, 11], [12, 13], 3, bad_id)},
            "task",
            router_replay_config=config,
        )

    with pytest.raises(ValueError, match="no routed-expert data"):
        get_train_data_for_step(
            _step("c"),
            {"c": _Completion([10, 11], [12, 13], version=3)},
            "task",
            router_replay_config=config,
        )
