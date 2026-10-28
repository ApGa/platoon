"""A group's credit under its entry's credit estimator (``environments[].credit``, :mod:`platoon.train.credit.api`), as
``GroupRolloutWorkflow`` runs it: the estimator of the group being run (``_entry_estimator``), the group the estimator
sees (``_credit_group``: the members that returned a result, their rows, and, for an estimator that reads agents,
their agents from the workers' tags), and the group's output once it is credited (``_credited_output``).  The
estimator's ``assign`` itself is awaited through :func:`platoon.train.credit.runtime.assign_credit`; a group whose
credit failed because a policy's scoring did (``_credit_failed``) trains nothing.  Without an estimator the workflow
centers each member's rows itself (``assign_advantages``), as it always did, and none of this runs.

The functions here that take the workflow first are its methods still: ``GroupRolloutWorkflow`` keeps each under its
name, so to replace one, replace the method, on the class or an instance, not the function.
"""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from platoon.train.credit.api import AgentRun, CreditGroup, MemberRollout, PolicyView, ScoringFailed, TrainingSample
from platoon.train.credit.runtime import Credit
from platoon.train.labels import task_label
from platoon.train.policy.spec import MAIN, PolicyId
from platoon.train.workflow.context import current_entry
from platoon.train.workflow.output import PolicyGroupOutput, _TaskRolloutOutput
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


def _credit_group(
    workflow: "GroupRolloutWorkflow", data: dict, results: Sequence[TrajectoryCollectionResult], *, tagged: bool
) -> CreditGroup:
    """The group an estimator credits: the members that returned a result (``results``), in the group's order, each
    with its rows and, when ``tagged`` (the estimator reads agents), its agents, each row of the policy its agent
    samples (``main`` untagged); the task of row ``data``, its label and its entry, the entry's settings, the run's
    policies as estimators read them (``policy_views``, which ``platoon train`` gives the training workflow; without
    them, ``main``'s from the workflow's model info) with each one's version now, and where the group's statistics
    go."""

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
    )


def _credit_failed(workflow: "GroupRolloutWorkflow", data: dict, error: ScoringFailed) -> str:
    """The verdict of the group of task row ``data`` whose credit failed because a policy's scoring did (``error``,
    which ``assign_credit`` lets through): ``failed``, warned of and counted under ``credit/failures``.  The group
    trains nothing, and the run goes on."""

    logger.warning("Task %s group trains nothing: its credit failed, %s", data["task_id"], error)
    workflow._group_stats().scalar(**{"credit/failures": 1.0})
    return "failed"


def member_rollout(index: int, result: TrajectoryCollectionResult, policy: PolicyId, *, tagged: bool) -> MemberRollout:
    """Member ``index``'s rollout as an estimator sees it: each of its rows, of ``policy``; when ``tagged``, each row
    with the agent that produced it, of the policy that agent samples, and every agent of its tree
    (:func:`agent_runs`)."""

    task_reward, valid = result.task_reward, result.task_reward_valid
    if not tagged:
        samples = [TrainingSample(datum, policy, index) for datum in result.datums]
        return MemberRollout(index, samples, [], task_reward, valid, result)
    agents = agent_runs(result, policy)
    samples = []
    for datum, trajectory in zip(result.datums, result.datum_trajectories or (), strict=True):
        agent = agents[trajectory]
        sample = TrainingSample(datum, agent.policy, index, agent.agent_id, agent.role, trajectory, agent.depth)
        agent.samples.append(sample)
        samples.append(sample)
    return MemberRollout(index, samples, list(agents.values()), task_reward, valid, result)


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
    workflow: "GroupRolloutWorkflow", output: _TaskRolloutOutput, credit: Credit, kept: Sequence[int]
) -> _TaskRolloutOutput | PolicyGroupOutput:
    """The group's output once its estimator credited it: under an estimator that reads agents, the rows the error-token
    filter kept (``kept``, their places among the group's rows as they were handed to the estimator,
    ``credit.samples``) with their tags (:class:`~platoon.train.workflow.output.PolicyGroupOutput`), else ``output`` as
    it is; carrying the credit (its ``credit``) when the estimator has a ``settle``, which the run calls as it files the
    group (:func:`~platoon.train.credit.runtime.settle_credit`)."""

    credited: _TaskRolloutOutput | PolicyGroupOutput = output
    if credit.estimator.needs_agents:
        credited = PolicyGroupOutput(tuple(credit.samples[index] for index in kept), output.report)
    if callable(getattr(credit.estimator, "settle", None)):
        credited.credit = credit
    return credited
