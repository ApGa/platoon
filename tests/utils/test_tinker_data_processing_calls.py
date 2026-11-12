"""Each training row's model calls (RFC 0021's per-call token ranges): the conversion records, for every datum it makes,
the calls whose sampled tokens it trains, each with the target tokens those tokens fill, and the credit workflow hands
them to the estimator on each row (``TrainingSample.calls``)."""

from __future__ import annotations

from types import SimpleNamespace

import tinker

import platoon.train.tinker  # noqa: F401  (resolves the package's import cycle first)
from platoon.train.policy.spec import MAIN
from platoon.train.workflow.crediting import member_rollout
from platoon.utils import tinker_data_processing as tdp
from platoon.utils.trajectory_status import EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY


def _interaction(prompt: list[int], completion: list[int]):
    """Stand-in for ``TinkerLLMInteraction``."""

    return SimpleNamespace(
        obs=tinker.ModelInput.from_ints(prompt),
        action=SimpleNamespace(tokens=list(completion), logprobs=[-0.5] * len(completion)),
        version=None,
    )


def _step(completion_id: str) -> dict:
    return {"misc": {"action_misc": {"completion_id": completion_id}}}


# Two rows: two calls merged (the second's prompt extends the first's prompt and completion), and a call of its own.
CALLS = {
    "c1": _interaction([1, 2, 3], [10, 11]),
    "c2": _interaction([1, 2, 3, 10, 11, 4, 5], [12, 13, 14]),
    "c3": _interaction([9, 9], [15]),
}


def _ranges_cover(datum: tinker.Datum, calls: tuple[tuple[str, int, int], ...]) -> bool:
    covered = {position for _, start, end in calls for position in range(start, end)}
    trained = {position for position, mask in enumerate(datum.loss_fn_inputs["mask"].tolist()) if mask > 0}
    return covered == trained


def test_call_ranges_cover_a_rows_action_tokens():
    result = tdp.trajectory_to_data(
        trajectory={"steps": [_step(call) for call in CALLS]},
        interactions=CALLS,
        task_id="task",
        trajectory_id="root",
        trajectory_advantage=1.0,
        checkpoint_version=0,
    )

    assert result.datum_calls == [(("c1", 2, 4), ("c2", 6, 9)), (("c3", 1, 2),)]
    assert all(_ranges_cover(datum, calls) for datum, calls in zip(result.datums, result.datum_calls, strict=True))
    first = result.datums[0].loss_fn_inputs["target_tokens"].tolist()
    assert (first[2:4], first[6:9]) == ([10, 11], [12, 13, 14])  # each range holds its call's sampled tokens


def test_a_call_a_row_keeps_only_as_context_is_no_call_of_the_row(monkeypatch):
    """A message ``filter_message_actions`` masks stays in its row's context and trains nothing, so the row names only
    the calls whose tokens it trains."""

    monkeypatch.setattr(tdp, "detected_message_completion_ids", lambda steps: {"c2"})

    result = tdp.trajectory_to_data(
        trajectory={"steps": [_step(call) for call in CALLS]},
        interactions=CALLS,
        task_id="task",
        trajectory_id="root",
        trajectory_advantage=1.0,
        checkpoint_version=0,
        filter_message_actions=True,
    )

    assert result.datum_calls == [(("c1", 2, 4),), (("c3", 1, 2),)]
    assert all(_ranges_cover(datum, calls) for datum, calls in zip(result.datums, result.datum_calls, strict=True))


def test_each_row_a_collection_keeps_carries_its_calls_to_its_training_sample():
    child_calls = {"k1": _interaction([30], [31])}
    collection = {
        "trajectories": {
            "root": {"steps": [_step(call) for call in CALLS], "reward": 1.0, "misc": {}},
            "child": {
                "steps": [_step("k1")],
                "reward": 0.0,
                "misc": {EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY: True},  # its row trains nothing
                "parent_info": {"id": "root"},
            },
        }
    }

    result = tdp.get_train_data_for_trajectory_collection(
        collection, {**CALLS, **child_calls}, task_id="task", checkpoint_version=0
    )
    samples = member_rollout(0, result, MAIN, tagged=False).samples

    assert [sample.calls for sample in samples] == [(("c1", 2, 4), ("c2", 6, 9)), (("c3", 1, 2),)]
    assert all(_ranges_cover(sample.datum, sample.calls) for sample in samples)


def test_a_row_whose_conversion_recorded_no_calls_says_none():
    result = tdp.TrajectoryCollectionResult(
        datums=tdp.trajectory_to_data(
            trajectory={"steps": [_step("c3")]},
            interactions=CALLS,
            task_id="task",
            trajectory_id="root",
            trajectory_advantage=1.0,
            checkpoint_version=0,
        ).datums,
        task_reward=1.0,
        trajectory_stats=[],
        root_rewards_dict={},
    )

    assert [sample.calls for sample in member_rollout(0, result, MAIN, tagged=False).samples] == [()]
