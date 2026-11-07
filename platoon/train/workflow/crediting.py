"""A group's credit under its entry's credit estimator (``environments[].credit``, :mod:`platoon.train.credit.api`), as
``GroupRolloutWorkflow`` runs it: the estimator of the group being run (``_entry_estimator``), the group the estimator
sees (``_credit_group``: the members that returned a result, their rows with the versions they were sampled at, for an
estimator that reads agents their agents from the workers' tags, and the task's own columns), and the group's output
once it is credited (``_credited_output``).  The estimator's ``assign`` itself is awaited through
:func:`platoon.train.credit.runtime.assign_credit`; a group whose credit failed because a policy's scoring did
(``_credit_failed``) trains nothing.  Without an estimator the workflow centers each member's rows itself
(``assign_advantages``), as it always did, and none of this runs.

In a run that trains several policies, each row must say which policy it trains, so that the step hands it to that
policy's learner: the workers tag every row with the agent that produced it, the estimator sees the group's agents, and
the output holds the rows with their tags (:func:`tags_rows`), whatever the estimator reads.  Every entry then has an
estimator: ``keyed``, on ``[task, policy, role]``, for one that names none (RFC 0021, D5).

The functions here that take the workflow first are its methods still: ``GroupRolloutWorkflow`` keeps each under its
name, so to replace one, replace the method, on the class or an instance, not the function.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from platoon.train.components import platoon_column
from platoon.train.credit.api import AgentRun, CreditGroup, MemberRollout, PolicyView, ScoringFailed, TrainingSample
from platoon.train.credit.runtime import Credit
from platoon.train.labels import label_text, task_label
from platoon.train.policy.spec import MAIN, PolicyId
from platoon.train.workflow.context import current_entry
from platoon.train.workflow.output import GroupOutput, PolicyGroupOutput
from platoon.train.workflow.protocol import capabilities_of
from platoon.utils.tinker_data_processing import TrajectoryCollectionResult

if TYPE_CHECKING:
    from platoon.train.workflow.group import GroupRolloutWorkflow

logger = logging.getLogger(__name__)


def _entry_estimator(workflow: "GroupRolloutWorkflow") -> Any | None:
    """The credit estimator of the group being run: its entry's (``entry_credit``, which ``platoon train`` gives a
    workflow for the entries that set one), unless the workflow credits its groups with hooks of its own: its
    capabilities say it applies no estimator (``applies_credit``, as loading reads them: a subclass that overrides
    ``compute_baselines`` or ``assign_advantages`` and declares nothing); None without one."""

    credit = workflow.entry_credit
    return credit.get(current_entry()) if credit and capabilities_of(workflow).applies_credit else None


def tags_rows(workflow: "GroupRolloutWorkflow", estimator: Any) -> bool:
    """Whether the group's rows carry who produced them, the agent and its policy: the group's ``estimator`` reads
    agents (``needs_agents``), or the run trains several policies (``trained_policies``), whose rows go each to its own
    policy's learner."""

    return bool(getattr(estimator, "needs_agents", False)) or len(workflow.trained_policies) > 1


def _credit_group(
    workflow: "GroupRolloutWorkflow", data: dict, results: Sequence[TrajectoryCollectionResult], *, tagged: bool
) -> CreditGroup:
    """The group an estimator credits: the members that returned a result (``results``), in the group's order, each
    with its rows and, when ``tagged`` (the estimator reads agents), its agents, each row of the policy its agent
    samples (``main`` untagged); the task of row ``data``, its label, its entry and its own columns
    (:func:`task_columns`), the entry's settings, the run's policies as estimators read them (``policy_views``, which
    ``platoon train`` gives the training workflow; without them, ``main``'s from the workflow's model info) with each
    one's version now, and where the group's statistics go."""

    views = workflow.policy_views
    if not views:
        llm = workflow.model_info.llm
        renderer, tokenizer = getattr(llm, "renderer", None), getattr(llm, "tokenizer", None)
        views = {MAIN: PolicyView(MAIN, renderer=renderer, tokenizer=tokenizer, current_version=lambda: llm.version)}
    return CreditGroup(
        task_id=data["task_id"],
        label=task_label(data, workflow.label_by),
        entry=current_entry(),
        settings=workflow.config,
        members=[member_rollout(index, result, MAIN, tagged=tagged) for index, result in enumerate(results)],
        policies=dict(views),
        versions={policy: view.version for policy, view in views.items()},
        stats=workflow._group_stats(),
        columns=task_columns(data),
    )


def task_columns(data: Any) -> dict[str, str]:
    """The task's own columns of its dataset row ``data``, as an estimator reads them (``CreditGroup.columns``): each
    column but platoon's own (``task_id``, ``_platoon*``), as text, as a label reads it
    (:func:`~platoon.train.labels.label_text`), leaving out a column whose value is empty or not a scalar."""

    if not isinstance(data, Mapping):
        return {}
    return {
        name: text
        for name, value in data.items()
        if isinstance(name, str) and not platoon_column(name) and (text := label_text(value)) is not None
    }


def _credit_failed(workflow: "GroupRolloutWorkflow", data: dict, error: ScoringFailed) -> str:
    """The verdict of the group of task row ``data`` whose credit failed because a policy's scoring did (``error``,
    which ``assign_credit`` lets through): ``failed``, warned of and counted under ``credit/failures``.  The group
    trains nothing, and the run goes on."""

    logger.warning("Task %s group trains nothing: its credit failed, %s", data["task_id"], error)
    workflow._group_stats().scalar(**{"credit/failures": 1.0})
    return "failed"


def member_rollout(index: int, result: TrajectoryCollectionResult, policy: PolicyId, *, tagged: bool) -> MemberRollout:
    """Member ``index``'s rollout as an estimator sees it: each of its rows, of ``policy``, with the version it was
    sampled at (:func:`sampled_versions`); when ``tagged``, each row with the agent that produced it, of the policy
    that agent samples, and every agent of its tree (:func:`agent_runs`)."""

    task_reward, valid = result.task_reward, result.task_reward_valid
    versions = sampled_versions(result)
    if not tagged:
        samples = [
            TrainingSample(datum, policy, index, version=version)
            for datum, version in zip(result.datums, versions, strict=True)
        ]
        return MemberRollout(index, samples, [], task_reward, valid, result)
    agents = agent_runs(result, policy)
    samples = []
    for datum, trajectory, version in zip(result.datums, result.datum_trajectories or (), versions, strict=True):
        agent = agents[trajectory]
        sample = TrainingSample(
            datum, agent.policy, index, agent.agent_id, agent.role, trajectory, agent.depth, version=version
        )
        agent.samples.append(sample)
        samples.append(sample)
    return MemberRollout(index, samples, list(agents.values()), task_reward, valid, result)


def sampled_versions(result: TrajectoryCollectionResult) -> list[int | None]:
    """The version each of a member's rows was sampled at, in the order of its datums: its own, the oldest its model
    calls sampled, or its rollout's when one of them recorded none (``own_versions``, which the conversion gives
    whatever ``train.stale_filter`` is); for a result whose own versions do not match its datums in number (a
    workflow that changed them after the conversion), None for each, so that each row reads the version its datum
    carries."""

    if len(result.own_versions) == len(result.datums):
        return list(result.own_versions)
    return [None] * len(result.datums)


def agent_runs(result: TrajectoryCollectionResult, policy: PolicyId) -> dict[str, AgentRun]:
    """Every agent of a member's tree, by trajectory, from the tags its rollout's conversion gave it (worker protocol
    3: ``agents``, and each row's trajectory, ``datum_trajectories``), each of the policy its tag names (``policy``
    for a tag that names none: a run of one policy).  A ``ValueError`` for rows without tags, or with tags of no
    agent of the tree: an estimator that reads agents never credits a row by guesswork."""

    trajectories = result.datum_trajectories
    if result.agents is None or trajectories is None or len(trajectories) != len(result.datums):
        raise ValueError(
            "a member's rows carry no tags of the agents that produced them, which the entry's credit estimator reads "
            "(needs_agents): its rollout was not converted from the workflow's worker request (a workflow that "
            "overrides how a member runs), or in a rollout worker of another platoon"
        )
    agents = {
        tag.trajectory_id: AgentRun(
            tag.agent_id, tag.trajectory_id, tag.parent_id, tag.role, tag.policy or policy, tag.depth, tag.reward
        )
        for tag in result.agents
    }
    if unknown := sorted(set(trajectories) - set(agents)):
        raise ValueError(f"a member's rows are tagged with trajectories of no agent of its tree: {unknown}")
    return agents


def _credited_output(
    workflow: "GroupRolloutWorkflow", output: GroupOutput, credit: Credit, kept: Sequence[int]
) -> GroupOutput | PolicyGroupOutput:
    """The group's output once its estimator credited it: when its rows carry tags (:func:`tags_rows`), the rows the
    error-token filter kept (``kept``, their places among the group's rows as they were handed to the estimator,
    ``credit.samples``) with their tags (:class:`~platoon.train.workflow.output.PolicyGroupOutput`), else ``output`` as
    it is; carrying the credit (its ``credit``) when the estimator has a ``settle``, which the run calls as it files the
    group (:func:`~platoon.train.credit.runtime.settle_credit`)."""

    credited: GroupOutput | PolicyGroupOutput = output
    if tags_rows(workflow, credit.estimator):
        credited = PolicyGroupOutput(tuple(credit.samples[index] for index in kept), output.report)
    if callable(getattr(credit.estimator, "settle", None)):
        credited.credit = credit
    return credited
