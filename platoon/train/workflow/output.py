"""A group's output and its members': a member's outcome (:class:`MemberOutcome`), the group's datum list with its
workload and statistics (:class:`GroupOutput`), the group's rows tagged by who produced them
(:class:`PolicyGroupOutput`, under a credit estimator that reads agents), what each member returned, for the per-task
statistics, and how a full checkpoint keeps the group's output (a codec of each, registered with
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
from platoon.train.credit.api import TrainingSample
from platoon.train.credit.runtime import Credit
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
class MemberOutcome:
    """What one member of a group came to: its result (None when it left none), and the generation work it took, kept
    also when converting its rollout into datums fails.

    - ``disposition`` records *how* the member ended (see the taxonomy in ``workers.supervisor``).  It defaults to
      ``completed``, so a caller that builds the outcome positionally (tests do) need not pass it.  It is never
      load-bearing for training: a worker death is a member failure, decided by ``min_successful_group_size`` like
      any other, never a group or trainer failure.
    - ``outcome_status`` is the status of the outcome the member's root recorded
      (``RolloutWorkerResponse.outcome_status``), also when it left no result, and ``error`` when its environment
      could not open its session.  It is counted (``outcome/<status>``), never trained on.
    - ``reward_key`` is the reward key the member's root trained (``RolloutWorkerResponse.reward_key``), counted
      (``reward_key/<key>``).  The group's members must agree on it.
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


class GroupOutput(list[tinker.Datum]):
    """A group's output: its datums, as a list, carrying the group's workload and rollout statistics.

    The trainer, and any code that expects a workflow's plain list, can treat
    it as an ordinary datum list.  In particular, an empty instance behaves like an empty list,
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

    It is a group's output written to the workflow contract (:class:`~platoon.train.workflow.protocol.RolloutOutput`):
    its datums are all the run's one trained policy's (:meth:`rows`), and :attr:`report` reads its attributes.  A full
    checkpoint keeps it by its exact type (its codec, below).
    """

    # The credit of the group by its entry's estimator, when the estimator has a ``settle``, which the run calls with it
    # as it files the group (platoon.train.credit.runtime.settle_credit); None after that, and without one.  An
    # arun_episode that returns another output in its place carries it over.
    credit: Credit | None = None

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


@dataclass(eq=False)
class PolicyGroupOutput:
    """A group's rows tagged by who produced them (:class:`~platoon.train.credit.api.TrainingSample`: the policy, the
    member, and the agent, role, trajectory and depth), with its report: what ``GroupRolloutWorkflow`` returns, in place
    of a ``GroupOutput``, for a group that an estimator which reads agents (``needs_agents``) credited (a group that
    trains nothing before any credit stays an empty ``GroupOutput``).  It is no list: the step reads its rows by policy
    (:meth:`rows`), each policy's in the group's order, and the rest of it through its report.  The tags stay in the
    trainer process: the rows a step sends are the datums."""

    samples: tuple[TrainingSample, ...]
    report: OutputReport
    # The group's credit until the run settles it, as a GroupOutput's (``GroupOutput.credit``); a checkpoint keeps none.
    credit: Credit | None = dataclasses.field(default=None, repr=False)

    def rows(self) -> Mapping[str, Sequence[tinker.Datum]]:
        """Its rows by the policy each trains, in the group's order."""

        rows: dict[str, list[tinker.Datum]] = {}
        for sample in self.samples:
            rows.setdefault(sample.policy, []).append(sample.datum)
        return rows

    def __len__(self) -> int:
        """Its rows, of every policy."""

        return len(self.samples)


def _summarize_outcomes(outcomes: list[MemberOutcome]) -> RolloutStats:
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
    stats.counts.update(reward_key_counts(outcome.reward_key for outcome in outcomes))
    stats.counts[REWARD_KEY_MISMATCH] = int(len(_trained_keys(outcomes)) > 1)
    return stats


def _member(outcome: MemberOutcome) -> MemberResult:
    """What one member returned, for the per-task statistics:

    - ``reward``: its trained reward;
    - ``scored``: the reward its environment scored, as ``platoon eval`` judges success.  None for an outcome that was
      not scored.  Else it is the reward the environment reported before any reward processor, or, for a rollout
      that records none, its trained reward.  A rollout whose trace records no outcome reports its own success flag
      (``flag``), which counts as a pass at 1;
    - its validity, outcome, disposition and reward key;
    - its root's steps and output tokens, and how long its rollout took;
    - ``versions``: the policy versions its datums sampled."""

    result = outcome.result
    status = outcome.outcome_status
    success = outcome.success_reward
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
        disposition=outcome.disposition,
        reward_key=outcome.reward_key,
        steps=root.num_steps if root is not None else None,
        output_tokens=root.num_output_tokens if root is not None else None,
        seconds=outcome.seconds,
        versions={MAIN: (min(versions), max(versions))} if versions else {},
        flag=status is None and success is not None,
    )


def _trained_keys(outcomes: Sequence[MemberOutcome]) -> set[str]:
    """The reward keys a group's members trained (a member that trained none, such as one platoon scored 0 itself,
    adds none): more than one, and the group does not train."""

    return {key for outcome in outcomes if (key := outcome.reward_key)}


def _make_task_output(
    workflow: "GroupRolloutWorkflow",
    datums: list[tinker.Datum],
    *,
    outcomes: list[MemberOutcome],
    trainable_rollouts: int,
    verdict: str = "unknown",
    members: tuple[MemberResult, ...] | None = None,
) -> GroupOutput:
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
    return GroupOutput(
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
) -> MemberOutcome:
    """Map a worker response (or its absence) onto a group member outcome."""

    if response is None:
        return MemberOutcome(
            result=None,
            workload=RolloutWorkload(),
            observed=False,
            disposition=disposition,
        )
    return MemberOutcome(
        result=response.result,
        workload=response.workload,
        observed=response.observed,
        disposition=disposition,
        outcome_status=response.outcome_status,
        reward_key=response.reward_key,
        success_reward=response.success_reward,
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
    return GroupOutput(
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


def _encode_policy_group_output(output: PolicyGroupOutput) -> dict[str, Any]:
    """A tagged group's output as a full checkpoint keeps it: each row exactly, with its tags, and its report."""

    report = output.report
    return {
        "samples": [
            {
                "datum": encode_datum(sample.datum),
                "policy": sample.policy,
                "member": sample.member,
                "agent_id": sample.agent_id,
                "role": sample.role,
                "trajectory_id": sample.trajectory_id,
                "depth": sample.depth,
            }
            for sample in output.samples
        ],
        "report": {
            "label": report.label,
            "entry": report.entry,
            "workload": dataclasses.asdict(report.workload) if report.workload is not None else None,
            "requested_rollouts": report.requested_rollouts,
            "observed_rollouts": report.observed_rollouts,
            "trainable_rollouts": report.trainable_rollouts,
            "task_retained_datums": report.task_retained_datums,
            "rollout_stats": dataclasses.asdict(report.rollout_stats) if report.rollout_stats is not None else None,
            "group_stats": report.group_stats.to_json() if report.group_stats is not None else None,
            "members": [member.to_json() for member in report.members],
            "verdict": report.verdict,
        },
    }


def _decode_policy_group_output(data: Mapping[str, Any]) -> PolicyGroupOutput:
    """A kept tagged group's output as it settled (:func:`_encode_policy_group_output`'s data)."""

    report = data["report"]
    workload, stats, group_stats = report["workload"], report["rollout_stats"], report["group_stats"]
    return PolicyGroupOutput(
        samples=tuple(
            TrainingSample(
                decode_datum(sample["datum"]),
                policy=str(sample["policy"]),
                member=int(sample["member"]),
                agent_id=sample["agent_id"],
                role=sample["role"],
                trajectory_id=sample["trajectory_id"],
                depth=sample["depth"],
            )
            for sample in data["samples"]
        ),
        report=OutputReport(
            label=report["label"],
            entry=report["entry"],
            workload=RolloutWorkload(**workload) if workload is not None else None,
            requested_rollouts=report["requested_rollouts"],
            observed_rollouts=report["observed_rollouts"],
            trainable_rollouts=report["trainable_rollouts"],
            task_retained_datums=report["task_retained_datums"],
            rollout_stats=RolloutStats(**stats) if stats is not None else None,
            group_stats=DeferredStats.from_json(group_stats) if group_stats is not None else None,
            members=tuple(MemberResult.from_json(member) for member in report["members"]),
            verdict=str(report["verdict"]),
        ),
    )


register_codec(PolicyGroupOutput, "policy_group", _encode_policy_group_output, _decode_policy_group_output)
