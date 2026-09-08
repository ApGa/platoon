"""Raw rollout accounting after the move to ``workers.workload``.

The two helpers used to live in ``group_rollout_workflow``.  They must be
byte-identical after the move (the trainer's funnel validation is unforgiving)
and they must be importable without the workflow module, because the workflow
imports ``workers.run`` and a cycle would break the worker at startup.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import platoon.train.tinker  # noqa: F401  (import order: package before data processing)

from platoon.train.tinker.workers.workload import add_datum_funnel, workload_from_raw_rollout
from platoon.utils.rollout_workload import RolloutWorkload, trajectory_collection_shape
from platoon.utils.tinker_data_processing import TrajectoryCollectionResult, TrajectoryStats


def _legacy_workload_from_raw_rollout(trajectory_collection, interactions) -> RolloutWorkload:
    """Verbatim copy of the pre-move body, kept as a drift detector."""

    trajectories, environment_steps = trajectory_collection_shape(trajectory_collection)
    input_tokens = sum(int(interaction.obs.length) for interaction in interactions.values())
    output_tokens = sum(len(interaction.action.tokens) for interaction in interactions.values())
    return RolloutWorkload(
        environment_steps=environment_steps,
        model_calls=len(interactions),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        trajectories=trajectories,
    )


def _legacy_add_datum_funnel(workload, result) -> RolloutWorkload:
    """Verbatim copy of the pre-move body, kept as a drift detector."""

    if result is None:
        return workload
    postmerge = sum(stats.num_datums for stats in result.trajectory_stats)
    policy_eligible = postmerge - result.num_policy_excluded_datums
    post_sampling = len(result.datums)
    return replace(
        workload,
        postmerge_datums=postmerge,
        policy_eligible_datums=policy_eligible,
        post_sampling_datums=post_sampling,
    )


def _interaction(input_tokens: int, output_tokens: int):
    return SimpleNamespace(
        obs=SimpleNamespace(length=input_tokens),
        action=SimpleNamespace(tokens=list(range(output_tokens))),
    )


def _recursive_tree() -> dict:
    # A root, a subagent that converted, and a subagent that did not: the raw
    # workload must count all three trees and every unique proxy call.
    return {
        "trajectories": {
            "root": {"steps": [{}, {}, {}]},
            "child-ok": {"steps": [{}]},
            "child-unconvertible": {"steps": [{}, {}]},
        }
    }


def _interactions() -> dict:
    return {
        "c1": _interaction(1200, 40),
        "c2": _interaction(2400, 55),
        "c3": _interaction(900, 12),
    }


def test_workload_from_raw_rollout_matches_the_pre_move_function():
    collection = _recursive_tree()
    interactions = _interactions()
    moved = workload_from_raw_rollout(collection, interactions)
    assert moved == _legacy_workload_from_raw_rollout(collection, interactions)
    assert moved == RolloutWorkload(
        environment_steps=6,
        model_calls=3,
        input_tokens=4500,
        output_tokens=107,
        trajectories=3,
    )
    # A rollout that produced nothing at all is still well-formed.
    assert workload_from_raw_rollout(None, {}) == RolloutWorkload()
    assert workload_from_raw_rollout({"trajectories": {}}, {}) == RolloutWorkload()


def test_add_datum_funnel_orders_the_three_counts():
    result = TrajectoryCollectionResult(
        datums=[object(), object()],  # only the count matters here
        task_reward=1.0,
        trajectory_stats=[
            TrajectoryStats(
                trajectory_id=name,
                reward=1.0,
                num_steps=1,
                num_input_tokens=10,
                num_output_tokens=5,
                num_datums=num_datums,
                is_root=name == "root",
            )
            for name, num_datums in (("root", 2), ("child-a", 1), ("child-b", 1))
        ],
        root_rewards_dict={},
        num_policy_excluded_datums=1,
    )
    raw = workload_from_raw_rollout(_recursive_tree(), _interactions())
    funneled = add_datum_funnel(raw, result)

    assert funneled == _legacy_add_datum_funnel(raw, result)
    assert funneled.postmerge_datums == 4
    assert funneled.policy_eligible_datums == 3
    assert funneled.post_sampling_datums == 2
    # RolloutWorkload.__post_init__ requires the ordering; constructing it at
    # all proves the funnel is monotone.
    assert (
        funneled.post_sampling_datums
        <= funneled.policy_eligible_datums
        <= funneled.postmerge_datums
    )
    # The raw fields are carried through untouched.
    assert funneled.model_calls == raw.model_calls
    assert funneled.input_tokens == raw.input_tokens


def test_add_datum_funnel_of_none_is_the_identity():
    raw = workload_from_raw_rollout(_recursive_tree(), _interactions())
    assert add_datum_funnel(raw, None) is raw


def test_workload_module_imports_without_the_workflow_package():
    """Import-cycle guard.

    ``group_rollout_workflow`` imports ``workers.run``, so nothing under
    ``workers/`` may import the workflow.  A cycle would surface as a worker
    that cannot start, long after the code review that introduced it.
    """

    package_dir = pathlib.Path(__file__).resolve().parents[3] / "platoon" / "train" / "tinker"
    script = f"""
import sys, types

import platoon.train  # real parent package, so the stub below has a parent

# Stand in for the tinker package so its eager __init__ (which imports rl.py
# and therefore the workflows package) never runs; this isolates the question
# to "do workers/* themselves reach for the workflow?".
package = types.ModuleType("platoon.train.tinker")
package.__path__ = [{str(package_dir)!r}]
sys.modules["platoon.train.tinker"] = package


class BlockWorkflows:
    def find_spec(self, name, path=None, target=None):
        if name.startswith("platoon.train.tinker.workflows"):
            raise AssertionError("workers/* must not import the workflow: " + name)
        return None


sys.meta_path.insert(0, BlockWorkflows())

import platoon.train.tinker.workers.protocol as protocol
import platoon.train.tinker.workers.run as run
import platoon.train.tinker.workers.workload as workload

assert workload.add_datum_funnel is not None
assert run.run_rollout_request is not None
assert protocol.PROTOCOL_VERSION == 1
print("clean")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert "clean" in completed.stdout
