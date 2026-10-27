"""Raw rollout accounting shared by the in-process and worker rollout paths.

Both helpers live here, not in the group rollout workflow's module
(:mod:`platoon.train.workflow.group`), because ``run.run_rollout_request`` must call
:func:`add_datum_funnel` immediately after conversion, in the process that did
the conversion, and ``workers/*`` must never import the workflow module (the
workflow imports ``workers.run``, so the dependency runs the other way).

Why the funnel cannot be left to the parent
-------------------------------------------
``GroupRolloutWorkflow._make_task_output`` hard-validates

    0 <= task_retained <= post_sampling <= policy_eligible <= postmerge

and raises ``ValueError`` otherwise.  ``RolloutWorkload`` defaults all three
funnel fields to ``0``, so a rollout that returned datums *without* the funnel
would fail that check for every task that produced training data.  The trainer
catches the failure, records ``failed_rollouts`` and enqueues ``None``: total,
silent data loss.  :func:`platoon.train.workers.protocol.dump_response`
therefore asserts the invariant in the child, where the traceback is
attributable.
"""

from __future__ import annotations

from dataclasses import replace

from platoon.utils.rollout_workload import RolloutWorkload, trajectory_collection_shape
from platoon.utils.tinker_data_processing import TrajectoryCollectionResult

__all__ = ["add_datum_funnel", "workload_from_raw_rollout"]


def workload_from_raw_rollout(
    trajectory_collection: dict | None,
    interactions: dict,
) -> RolloutWorkload:
    """Count one raw recursive tree and its unique proxy interactions."""

    trajectories, environment_steps = trajectory_collection_shape(trajectory_collection)
    # ``interactions`` is keyed by completion ID, so walking its values counts
    # every logical model request exactly once even when a recursive tree later
    # excludes, interrupts, or fails to convert one of its trajectories.
    input_tokens = sum(int(interaction.obs.length) for interaction in interactions.values())
    output_tokens = sum(len(interaction.action.tokens) for interaction in interactions.values())
    return RolloutWorkload(
        environment_steps=environment_steps,
        model_calls=len(interactions),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        trajectories=trajectories,
    )


def add_datum_funnel(
    workload: RolloutWorkload,
    result: TrajectoryCollectionResult | None,
) -> RolloutWorkload:
    """Attach post-merge, policy, and sampling counts to raw workload."""

    if result is None:
        return workload
    postmerge = sum(stats.num_datums for stats in result.trajectory_stats)
    policy_eligible = postmerge - result.num_policy_excluded_datums
    post_sampling = len(result.datums)
    # RolloutWorkload validates that the funnel is ordered and non-negative.
    return replace(
        workload,
        postmerge_datums=postmerge,
        policy_eligible_datums=policy_eligible,
        post_sampling_datums=post_sampling,
    )
