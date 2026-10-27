"""platoon.experimental: documented API that a minor release may still change (RFC 0017, section 2).

For now, task samplers and the per-task statistics they read (RFC 0019, section 5): a sampler class
(:class:`Sampler`, or any class of the :class:`TaskSampler` protocol) answers each decision with a task of its pool
(:class:`Draw`) or, from a run's sampler that hands its slots to the entries, a :class:`Slot`; it is registered with
:func:`task_sampler` from a module an environment entry's ``package`` names, or named as ``module:Class``, and
``platoon.testing.check_task_sampler`` checks it against the contract.  An adaptive sampler reads the statistics
through a :class:`TaskStatsView`; :class:`TaskStatsStore` and the record types let a test build one.
``docs/extending/task-samplers.md`` is the guide.

And group admission rules (RFC 0019, section 6): a rule class (:class:`AdmissionRule`, or any class of the
:class:`GroupAdmission` protocol) decides, for each settled training group and the statistics' view, whether it counts
toward the batch (:class:`Admission`, or a bool); it is built with an :class:`AdmissionSetup`, may refuse a config
in its load-time ``check`` (:class:`AdmissionCheckContext`), is registered with :func:`group_admission` or named as
``module:Class``, and ``platoon.testing.check_group_admission`` checks it.  ``docs/extending/group-admission.md`` is the
guide.

And rollout leases (RFC 0020): a training workflow runs each of its rollouts on one of its draw's rollout permits, in
a block of the draw's lease (:class:`RolloutLease`, which :func:`rollout_lease` reads in the workflow's task), so that
under ``train.permit_release: rollout`` each permit comes back as its rollout ends.
``docs/extending/rollout-leases.md`` is the guide.

And trainers (RFC 0021): what ``platoon train`` runs (:class:`Trainer`), built from what it prepared
(:class:`TrainerSetup`: the config, each policy's :class:`PolicySpec` by its id, :data:`MAIN` for ``train:``'s, the
run's :class:`TaskSources` and how its :class:`Workflows` are built), saying what it asks of the run
(:class:`TrainerNeeds`) and where its steps end (:class:`StepPlan`).  :class:`TrainerBase` takes a trainer through the
lifecycle every trainer shares, calling its hooks in order, and ``platoon.testing.check_trainer`` checks one.  What
every full checkpoint of the run holds besides its counters comes from its participants
(:class:`CheckpointParticipant`), and a policy's rows train under an :class:`Objective`.  ``trainer:`` selects the
trainer; this release has two, ``rl`` and ``sft``.
And step hooks (RFC 0021): work at the end of each step of any trainer (:class:`StepHook`, :class:`StepHookBase`),
which hears what the step did (:class:`StepEnd`): it drains side work, adds optimizer updates of the step's policy
(``end.train``, ``end.learner``), and hears what the step published; the rl trainer attaches those its config names
(``trainer_options.hooks``).  A hook's interruptible wait raises :class:`TrainInterrupted` on a stop request when
the trainer's steps are atomic (``TrainerNeeds.atomic_steps``); when they are not, it runs to its end.  Side inference
samples a policy's published sampler through a :class:`PolicySnapshot`, which refuses a sample once the policy has
published again (:class:`PolicyMoved`); and a training episode sees its draw (``ctx.draw``, a :class:`DrawInfo`) when
its trainer asks for it.  ``docs/extending/step-hooks.md`` is the guide.
And the workflow contract (RFC 0021): what the trainer needs of a rollout workflow (:class:`RolloutWorkflow`), what a
workflow class can do beyond it (:class:`WorkflowCapabilities`, read through :func:`capabilities_of`), what
``platoon train`` builds a workflow class with (:class:`WorkflowSetup`, which its ``from_setup`` takes), and how a
group's output is read: its report (:class:`OutputReport`, :func:`report_of`) and its rows by policy
(:class:`RolloutOutput`, :func:`rows_by_policy`).  A trainer that rolls out draws the run's groups through
``run.rollouts()``.
And credit estimators (RFC 0021): how an environment entry's training groups turn rewards into advantages
(:class:`CreditEstimator`, or a subclass of :class:`CreditEstimatorBase`), built per entry with a :class:`CreditSetup`
and checked at load (:class:`CreditCheckContext`); its ``assign`` sees a :class:`CreditGroup` of
:class:`MemberRollout` members, each agent an :class:`AgentRun` (role, policy, own reward, rows), each row a
:class:`TrainingSample`, and the run's policies as :class:`PolicyView` views.  It is registered with
:func:`credit_estimator` or named as ``module:Class``, and ``platoon.testing.check_credit_estimator`` checks it.
``docs/extending/credit-estimators.md`` is the guide.

This module imports neither ``tinker`` nor ``datasets``.
"""

from platoon.programs import DrawInfo
from platoon.train.admission_api import (
    Admission,
    AdmissionCheckContext,
    AdmissionRule,
    AdmissionSetup,
    GroupAdmission,
    group_admission,
)
from platoon.train.checkpoints.participants import CheckpointParticipant
from platoon.train.credit.api import (
    AgentRun,
    CreditCheckContext,
    CreditEstimator,
    CreditEstimatorBase,
    CreditGroup,
    CreditSetup,
    MemberRollout,
    PolicyView,
    TrainingSample,
)
from platoon.train.credit.registry import credit_estimator
from platoon.train.leases import RolloutLease, rollout_lease
from platoon.train.objectives.api import Objective
from platoon.train.policy.snapshot import PolicyMoved, PolicySnapshot
from platoon.train.policy.spec import MAIN, PolicySpec
from platoon.train.sampler_api import (
    Draw,
    Sampler,
    SamplerCheckContext,
    SamplerContext,
    SamplerSetup,
    Slot,
    TaskSampler,
    TaskStatsView,
    task_sampler,
)
from platoon.train.task_stats import (
    EntryInfo,
    GroupRecord,
    GroupTotals,
    MemberResult,
    TaskStats,
    TaskStatsStore,
    informative_at,
    pass_at,
)
from platoon.train.trainer.api import (
    StepPlan,
    TaskSources,
    Trainer,
    TrainerBase,
    TrainerNeeds,
    TrainerSetup,
    Workflows,
)
from platoon.train.trainer.hooks import StepEnd, StepHook, StepHookBase
from platoon.train.trainer.signals import TrainInterrupted
from platoon.train.workflow.protocol import (
    OutputReport,
    RolloutOutput,
    RolloutWorkflow,
    WorkflowCapabilities,
    WorkflowSetup,
    capabilities_of,
    report_of,
    rows_by_policy,
)

__all__ = [
    # task samplers
    "Draw",
    "Sampler",
    "SamplerCheckContext",
    "SamplerContext",
    "SamplerSetup",
    "Slot",
    "TaskSampler",
    "TaskStatsView",
    "task_sampler",
    # the per-task statistics (RFC 0019, decision Q2)
    "EntryInfo",
    "GroupRecord",
    "GroupTotals",
    "MemberResult",
    "TaskStats",
    "TaskStatsStore",
    "informative_at",
    "pass_at",
    # group admission (RFC 0019, section 6)
    "Admission",
    "AdmissionCheckContext",
    "AdmissionRule",
    "AdmissionSetup",
    "GroupAdmission",
    "group_admission",
    # rollout leases (RFC 0020)
    "RolloutLease",
    "rollout_lease",
    # trainers (RFC 0021)
    "Trainer",
    "TrainerBase",
    "TrainerNeeds",
    "TrainerSetup",
    "TaskSources",
    "Workflows",
    "StepPlan",
    "PolicySpec",
    "MAIN",
    "CheckpointParticipant",
    "Objective",
    # step hooks (RFC 0021)
    "StepHook",
    "StepHookBase",
    "StepEnd",
    "TrainInterrupted",
    "PolicySnapshot",
    "PolicyMoved",
    "DrawInfo",
    # the workflow contract (RFC 0021)
    "RolloutWorkflow",
    "WorkflowCapabilities",
    "capabilities_of",
    "WorkflowSetup",
    "RolloutOutput",
    "OutputReport",
    "report_of",
    "rows_by_policy",
    # credit estimators (RFC 0021)
    "CreditEstimator",
    "CreditEstimatorBase",
    "CreditSetup",
    "CreditCheckContext",
    "CreditGroup",
    "MemberRollout",
    "AgentRun",
    "TrainingSample",
    "PolicyView",
    "credit_estimator",
]
