"""Each datum's own policy version (RFC 0020, P5): the oldest version its merged calls sampled, beside the rollout's
version that every datum still carries (``checkpoint_version``), and stamped in its place only on request
(``TrajectoryCollectionResult.stamp_own_versions``, which a group workflow calls under the per-datum stale filter)."""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import tinker

import platoon.train.tinker  # noqa: F401  (resolves the package's import cycle first)
from platoon.train.step.staleness import datum_version
from platoon.utils import tinker_data_processing as tdp
from platoon.utils.trajectory_status import EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY


def _interaction(prompt: list[int], completion: list[int], version: int | None):
    """Stand-in for ``TinkerLLMInteraction``, with the version its call sampled."""

    return SimpleNamespace(
        obs=tinker.ModelInput.from_ints(prompt),
        action=SimpleNamespace(tokens=list(completion), logprobs=[-0.5] * len(completion)),
        version=version,
    )


def _step(completion_id: str) -> dict:
    return {"misc": {"action_misc": {"completion_id": completion_id}}}


# Three datums: two calls merged (versions 4 and 6), a call that does not extend them (7) merged with one that records
# no version, and a last call of its own (9).
CALLS = {
    "c1": _interaction([1, 2, 3], [10, 11], 4),
    "c2": _interaction([1, 2, 3, 10, 11, 4, 5], [12, 13], 6),
    "c3": _interaction([9, 9], [14], 7),
    "c4": _interaction([9, 9, 14, 8], [15], None),
    "c5": _interaction([20], [21], 9),
}


def test_each_datum_keeps_its_rollouts_version_and_gets_the_oldest_its_own_calls_sampled():
    result = tdp.trajectory_to_data(
        trajectory={"steps": [_step(call) for call in CALLS]},
        interactions=CALLS,
        task_id="task",
        trajectory_id="root",
        trajectory_advantage=1.0,
        checkpoint_version=2,
    )
    assert [datum.model_input.to_ints()[0] for datum in result.datums] == [1, 9, 20]
    assert [datum_version(datum) for datum in result.datums] == [2, 2, 2]  # the rollout's, as before
    assert result.own_versions == [4, 2, 9]  # the second datum's last call recorded none: the rollout's


def test_a_collection_gives_the_versions_of_the_datums_it_keeps_and_stamps_them_on_request():
    child_calls = {"k1": _interaction([30], [31], 5)}
    collection = {
        "trajectories": {
            "root": {"steps": [_step(call) for call in CALLS], "reward": 1.0},
            # A child whose datums never train: it keeps its stats, and none of its datums or versions.
            "child": {
                "steps": [_step("k1")],
                "reward": 0.0,
                "parent_info": {"id": "root"},
                "misc": {EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY: True},
            },
        }
    }
    result = tdp.get_train_data_for_trajectory_collection(
        trajectory_collection=collection,
        interactions={**CALLS, **child_calls},
        task_id="task",
        checkpoint_version=2,
    )
    assert result.num_policy_excluded_datums == 1 and len(result.datums) == 3
    assert result.own_versions == (4, 2, 9)
    assert [datum_version(datum) for datum in result.datums] == [2, 2, 2]

    assert result.stamp_own_versions()
    assert [datum_version(datum) for datum in result.datums] == [4, 2, 9]


def _one_datum_result(**fields) -> tdp.TrajectoryCollectionResult:
    datums = tdp.trajectory_to_data(
        trajectory={"steps": [_step("c5")]},
        interactions=CALLS,
        task_id="task",
        trajectory_id="root",
        trajectory_advantage=1.0,
        checkpoint_version=2,
    ).datums
    return tdp.TrajectoryCollectionResult(
        datums=datums, task_reward=1.0, trajectory_stats=[], root_rewards_dict={}, **fields
    )


def test_a_result_whose_versions_do_not_match_its_datums_keeps_its_rollouts():
    """None given (a conversion of its own), or datums dropped or added since the conversion (a workflow's own
    filter): the datums keep the version they carry, and the stamping says it did not stamp."""

    for versions in ((), (9, 9)):
        result = _one_datum_result(own_versions=versions)
        assert not result.stamp_own_versions()
        assert [datum_version(datum) for datum in result.datums] == [2]
    result = _one_datum_result(own_versions=(9,))
    assert result.stamp_own_versions() and [datum_version(datum) for datum in result.datums] == [9]


def test_a_result_pickled_before_its_own_versions_existed_reads_them_as_none():
    result = _one_datum_result(own_versions=(9,))
    del result.__dict__["own_versions"]  # the state of a result pickled before the field
    loaded = pickle.loads(pickle.dumps(result))
    assert loaded.own_versions == () and not loaded.stamp_own_versions()
    assert [datum_version(datum) for datum in loaded.datums] == [2]
