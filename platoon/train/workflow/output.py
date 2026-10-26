"""A group's output and its members': a member's outcome (``_SingleRolloutOutcome``), the group's datum list with its
workload and statistics (``_TaskRolloutOutput``, the contract's ``GroupOutput``), what each member returned, for the
per-task statistics, and how a full checkpoint keeps the group's output (its codec, registered with
:func:`~platoon.train.checkpoints.kept_groups.register_codec`).

The functions here that take the workflow first are its methods still: ``GroupRolloutWorkflow`` keeps each under its
name, so to replace one, replace the method, on the class or an instance, not the function.
"""

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import tinker
import torch

from platoon.train.checkpoints.kept_groups import decode_datum, encode_datum, register_codec
from platoon.train.task_stats import MAIN, MemberResult
from platoon.train.workers.protocol import RolloutWorkerResponse
from platoon.train.workflow.protocol import OutputReport, carried_report
from platoon.utils.rollout_stats import (
    REWARD_KEY_MISMATCH,
    TOOL_CALL_STATS,
    RolloutStats,
    outcome_counts,
    reward_key_counts,
    summarize_rollout_group,
)
from platoon.utils.rollout_workload import RolloutWorkload, record_workload_distribution, sum_rollout_workloads
from platoon.utils.stats_tracker import DeferredStats
from platoon.utils.tinker_data_processing import TrajectoryCollectionResult

if TYPE_CHECKING:
    from platoon.train.workflow.group import GroupRolloutWorkflow


@dataclass(frozen=True)
class _SingleRolloutOutcome:
    """Internal result that keeps raw generation work even if conversion fails.

    ``disposition`` records *how* the member ended (see the taxonomy in
    ``workers.supervisor``).  It is defaulted so existing callers -- including
    tests that construct this positionally -- are unaffected, and it is never
    load-bearing for training: a worker death is a member failure, decided by
    ``min_successful_group_size`` like any other, never a group or trainer
    failure.  ``outcome_status`` is the status of the outcome the member's root
    recorded (``RolloutWorkerResponse.outcome_status``), also when it left no
    result, and ``error`` when its environment could not open its session; it
    is counted (``outcome/<status>``), never trained on.  ``reward_key`` is the
    reward key the member's root trained (``RolloutWorkerResponse.reward_key``),
    counted (``reward_key/<key>``), and on which the group's members must agree.
    """

    result: TrajectoryCollectionResult | None
    workload: RolloutWorkload
    observed: bool
    disposition: str = "completed"
    outcome_status: str | None = None
    reward_key: str | None = None
    # The reward the member's environment scored, before any reward processor
    # (``RolloutWorkerResponse.success_reward``), and its rollout's seconds: for
    # the per-task statistics, never trained on.
    success_reward: float | None = None
    seconds: float | None = None


class _TaskRolloutOutput(list[tinker.Datum]):
    """List-compatible task result carrying workload and rollout statistics.

    The trainer and legacy consumers can continue treating this as an ordinary
    datum list.  In particular, an empty instance behaves like an empty list,
    while still carrying the generation work for a task which produced no
    trainable policy data.  ``group_stats`` holds what the workflow measured
    about the group, for the trainer to record in the step that consumes it.
    ``label`` is the task's label (its value of the run's label column,
    ``train.label_by``; by default the mixture entry it belongs to, so None
    outside a mixture), by which the trainer's metrics break down, and
    ``entry`` the label of the environment entry whose settings the group ran
    with (``current_entry``, :mod:`platoon.train.workflow.context`).  ``members``
    say what each member returned and
    ``verdict`` what the group trains, or why it trains nothing, for the per-task
    statistics (``platoon.train.task_stats``).

    It is the workflow contract's ``GroupOutput`` (:mod:`platoon.train.workflow.protocol`): its datums are all the
    run's one trained policy's (:meth:`rows`), and :attr:`report` reads its attributes.
    """

    def __init__(
        self,
        datums: list[tinker.Datum],
        *,
        workload: RolloutWorkload,
        requested_rollouts: int,
        observed_rollouts: int,
        trainable_rollouts: int,
        task_retained_datums: int,
        rollout_stats: RolloutStats | None = None,
        group_stats: DeferredStats | None = None,
        label: str | None = None,
        entry: str | None = None,
        members: tuple[MemberResult, ...] = (),
        verdict: str = "unknown",
    ) -> None:
        super().__init__(datums)
        self.workload = workload
        self.requested_rollouts = requested_rollouts
        self.observed_rollouts = observed_rollouts
        self.trainable_rollouts = trainable_rollouts
        self.task_retained_datums = task_retained_datums
        self.rollout_stats = rollout_stats
        self.group_stats = group_stats if group_stats is not None else DeferredStats()
        self.label = label
        self.entry = entry
        self.members = members
        self.verdict = verdict

    @property
    def report(self) -> OutputReport:
        """What the trainer reads of the group besides its datums: its attributes as they are when it is read, each
        checked as any output's that carries them (:func:`~platoon.train.workflow.protocol.carried_report`)."""

        return carried_report(self)

    def rows(self) -> Mapping[str, Sequence[tinker.Datum]]:
        """Its datums, all one policy's: ``{"main": self}``.  The group does not know which policy the run trains, so
        it names it ``main``, as a run of one policy does; the step does not ask it, but reads it as a list, whose
        datums are all the trained policy's (:func:`~platoon.train.workflow.protocol.rows_by_policy`)."""

        return {MAIN: self}


# The contract's name of the class: the same class object, by whose exact type a full checkpoint keeps the group.
GroupOutput = _TaskRolloutOutput


def _summarize_outcomes(outcomes: list[_SingleRolloutOutcome]) -> RolloutStats:
    results = [outcome.result for outcome in outcomes if outcome.result is not None]
    trajectories = [stats for result in results for stats in result.trajectory_stats]
    data = {
        "task_reward": [result.task_reward for result in results],
        "task_reward_valid": [result.task_reward_valid for result in results],
        **{
            key: [getattr(stats, key) for stats in trajectories]
            for key in ("num_steps", "num_input_tokens", "num_output_tokens", *TOOL_CALL_STATS)
        },
    }
    root_keys = {key for result in results for key in result.root_rewards_dict}
    for key in root_keys:
        data[f"root_{key}"] = [result.root_rewards_dict[key] for result in results if key in result.root_rewards_dict]
    reward_keys = {key for stats in trajectories for key in stats.rewards_dict if key.startswith("reward/")}
    for key in reward_keys:
        data[key] = [stats.rewards_dict[key] for stats in trajectories if key in stats.rewards_dict]
    stats = summarize_rollout_group(data)
    # Every member, also one that left no result (a setup failure, a dead worker, a cancelled straggler).
    stats.counts.update(outcome_counts(outcome.outcome_status for outcome in outcomes))
    stats.counts.update(reward_key_counts(getattr(outcome, "reward_key", None) for outcome in outcomes))
    stats.counts[REWARD_KEY_MISMATCH] = int(len(_trained_keys(outcomes)) > 1)
    return stats


def _member(outcome: _SingleRolloutOutcome) -> MemberResult:
    """What one member returned, for the per-task statistics: its trained reward and the reward its environment
    scored, as ``platoon eval`` judges success (None for an outcome that was not scored; the reward its environment
    reported before any reward processor, else, for a rollout that records none, its trained reward; a rollout whose
    trace records no outcome reports its own success flag, which counts as a pass at 1), its validity, outcome and
    disposition, its root's steps and output tokens, and the policy versions its datums sampled."""

    result = outcome.result
    status = getattr(outcome, "outcome_status", None)
    success = getattr(outcome, "success_reward", None)
    valid = bool(result is not None and result.task_reward_valid)
    if status is not None and status != "scored":
        scored = None
    elif success is not None:
        scored = success
    else:
        scored = result.task_reward if valid else None
    root = next((stats for stats in result.trajectory_stats if stats.is_root), None) if result is not None else None
    versions = [
        int(version.to_torch().reshape(-1)[0].item())
        for datum in (result.datums if result is not None else ())
        if (version := datum.loss_fn_inputs.get("checkpoint_version")) is not None
    ]
    return MemberResult(
        reward=result.task_reward if result is not None else None,
        scored=scored,
        valid=valid,
        outcome=status or "none",
        disposition=getattr(outcome, "disposition", "completed"),
        reward_key=getattr(outcome, "reward_key", None),
        steps=root.num_steps if root is not None else None,
        output_tokens=root.num_output_tokens if root is not None else None,
        seconds=getattr(outcome, "seconds", None),
        versions={MAIN: (min(versions), max(versions))} if versions else {},
        flag=status is None and success is not None,
    )


def _trained_keys(outcomes: Sequence[_SingleRolloutOutcome]) -> set[str]:
    """The reward keys a group's members trained (a member that trained none, such as one platoon scored 0 itself,
    adds none): more than one, and the group does not train."""

    return {key for outcome in outcomes if (key := getattr(outcome, "reward_key", None))}


def _make_task_output(
    workflow: "GroupRolloutWorkflow",
    datums: list[tinker.Datum],
    *,
    outcomes: list[_SingleRolloutOutcome],
    trainable_rollouts: int,
    verdict: str = "unknown",
    members: tuple[MemberResult, ...] | None = None,
) -> _TaskRolloutOutput:
    """Record unit distributions and attach exact totals to the queue item, with what each member returned
    (``members``, read from ``outcomes`` before any datum was filtered when not given) and the group's ``verdict``
    (``platoon.train.task_stats.VERDICTS``)."""

    rollout_workloads = [outcome.workload for outcome in outcomes]
    task_workload = sum_rollout_workloads(rollout_workloads)
    postmerge_datums = task_workload.postmerge_datums
    policy_eligible_datums = task_workload.policy_eligible_datums
    post_sampling_datums = task_workload.post_sampling_datums
    task_retained_datums = len(datums)
    if not 0 <= task_retained_datums <= post_sampling_datums <= policy_eligible_datums <= postmerge_datums:
        raise ValueError(
            "Invalid task datum funnel: "
            f"postmerge={postmerge_datums}, policy_eligible={policy_eligible_datums}, "
            f"post_sampling={post_sampling_datums}, task_retained={task_retained_datums}"
        )
    group_stats = workflow._group_stats()
    record_workload_distribution(
        group_stats,
        prefix="workload/rollout",
        workloads=rollout_workloads,
    )
    record_workload_distribution(
        group_stats,
        prefix="workload/task",
        workloads=[task_workload],
    )

    # The workload/task/count denominator was registered immediately above,
    # so these are task-level distributions with the same sample boundary.
    group_stats.stat(
        denominator="workload/task/count",
        **{
            "workload/task/requested_rollouts": torch.tensor([float(workflow.config.group_size)], dtype=torch.float32),
            "workload/task/observed_rollouts": torch.tensor(
                [float(sum(outcome.observed for outcome in outcomes))], dtype=torch.float32
            ),
            "workload/task/workflow_trainable_rollouts": torch.tensor([float(trainable_rollouts)], dtype=torch.float32),
            "workload/task/total_task_retained_datums": torch.tensor(
                [float(task_retained_datums)], dtype=torch.float32
            ),
            "workload/task/total_task_workflow_trainable_datums": torch.tensor(
                [float(task_retained_datums)], dtype=torch.float32
            ),
            "workload/task/total_task_workflow_non_trainable_datums": torch.tensor(
                [float(postmerge_datums - task_retained_datums)], dtype=torch.float32
            ),
            "workload/task/total_task_filter_dropped_datums": torch.tensor(
                [float(post_sampling_datums - task_retained_datums)], dtype=torch.float32
            ),
        },
    )
    rollout_stats = _summarize_outcomes(outcomes)
    rollout_stats.record(group_stats, prefix="rollout_completion")
    return _TaskRolloutOutput(
        datums,
        workload=task_workload,
        requested_rollouts=workflow.config.group_size,
        observed_rollouts=sum(outcome.observed for outcome in outcomes),
        trainable_rollouts=trainable_rollouts,
        task_retained_datums=task_retained_datums,
        rollout_stats=rollout_stats,
        group_stats=group_stats if isinstance(group_stats, DeferredStats) else None,
        members=members if members is not None else tuple(_member(outcome) for outcome in outcomes),
        verdict=verdict,
    )


def _outcome_from_response(
    workflow: "GroupRolloutWorkflow",
    response: RolloutWorkerResponse | None,
    disposition: str,
) -> _SingleRolloutOutcome:
    """Map a worker response (or its absence) onto a group member outcome."""

    if response is None:
        return _SingleRolloutOutcome(
            result=None,
            workload=RolloutWorkload(),
            observed=False,
            disposition=disposition,
        )
    return _SingleRolloutOutcome(
        result=response.result,
        workload=response.workload,
        observed=response.observed,
        disposition=disposition,
        outcome_status=response.outcome_status,
        reward_key=response.reward_key,
        success_reward=getattr(response, "success_reward", None),
        seconds=response.rollout_seconds,
    )


def _encode_group_output(output: GroupOutput) -> dict[str, Any]:
    """A group's output as a full checkpoint keeps it: its datums exactly, and every attribute that the step that
    consumes it reads."""

    stats = output.rollout_stats
    return {
        "datums": [encode_datum(datum) for datum in output],
        "workload": dataclasses.asdict(output.workload),
        "requested_rollouts": output.requested_rollouts,
        "observed_rollouts": output.observed_rollouts,
        "trainable_rollouts": output.trainable_rollouts,
        "task_retained_datums": output.task_retained_datums,
        "rollout_stats": dataclasses.asdict(stats) if stats is not None else None,
        "group_stats": output.group_stats.to_json(),
        "label": output.label,
        "entry": output.entry,
        "members": [member.to_json() for member in output.members],
        "verdict": output.verdict,
    }


def _decode_group_output(data: Mapping[str, Any]) -> GroupOutput:
    """A kept group's output as it settled (:func:`_encode_group_output`'s data)."""

    stats = data["rollout_stats"]
    return _TaskRolloutOutput(
        [decode_datum(datum) for datum in data["datums"]],
        workload=RolloutWorkload(**data["workload"]),
        requested_rollouts=int(data["requested_rollouts"]),
        observed_rollouts=int(data["observed_rollouts"]),
        trainable_rollouts=int(data["trainable_rollouts"]),
        task_retained_datums=int(data["task_retained_datums"]),
        rollout_stats=RolloutStats(**stats) if stats is not None else None,
        group_stats=DeferredStats.from_json(data["group_stats"]),
        label=data["label"],
        entry=data["entry"],
        members=tuple(MemberResult.from_json(member) for member in data["members"]),
        verdict=str(data["verdict"]),
    )


register_codec(GroupOutput, "group", _encode_group_output, _decode_group_output)
