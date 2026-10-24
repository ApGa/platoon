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

This module imports neither ``tinker`` nor ``datasets``.
"""

from platoon.train.admission_api import (
    Admission,
    AdmissionCheckContext,
    AdmissionRule,
    AdmissionSetup,
    GroupAdmission,
    group_admission,
)
from platoon.train.leases import RolloutLease, rollout_lease
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
]
