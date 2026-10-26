"""Moved: platoon 0.5.0 split this module into :mod:`platoon.train.workflow.group`,
:mod:`platoon.train.workflow.members`, :mod:`platoon.train.workflow.requests`, :mod:`platoon.train.workflow.output`,
:mod:`platoon.train.credit.group`, :mod:`platoon.train.workflow.telemetry`, :mod:`platoon.train.workflow.context`.
Its names still import from here, and so do the names it imported, each with a ``PlatoonDeprecationWarning`` that
names its new module, until platoon 0.6.0 removes this module."""

from platoon.train.tinker._moved import moved as _moved

__getattr__, __dir__ = _moved(
    globals(),
    {
        "_group_ready_callback": "platoon.train.workflow.context",
        "_group_stats": "platoon.train.workflow.context",
        "_group_config": "platoon.train.workflow.context",
        "_group_entry": "platoon.train.workflow.context",
        "_LEASE_METHODS": "platoon.train.workflow.group",
        "_routing_id_lock": "platoon.train.workflow.telemetry",
        "_NAMED_ROLLOUTS": "platoon.train.workflow.telemetry",
        "CLEANUP_WAIT_SECONDS": "platoon.train.workflow.members",
        "current_entry": "platoon.train.workflow.context",
        "_RoutingIdState": "platoon.train.workflow.telemetry",
        "_SingleRolloutOutcome": "platoon.train.workflow.output",
        "_TaskRolloutOutput": "platoon.train.workflow.output",
        "_summarize_outcomes": "platoon.train.workflow.output",
        "_member": "platoon.train.workflow.output",
        "_trained_keys": "platoon.train.workflow.output",
        "_workload_from_raw_rollout": "platoon.train.workflow.group",
        "_add_datum_funnel": "platoon.train.workflow.group",
        "group_baselines": "platoon.train.credit.group",
        "_filter_positive_centered_error_tokens": "platoon.train.credit.group",
        "_finish_cleanups": "platoon.train.workflow.members",
        "_GroupAdmission": "platoon.train.workflow.members",
        "_MemberStarts": "platoon.train.workflow.members",
        "GroupRolloutWorkflow": "platoon.train.workflow.group",
    },
    homes=(
        "platoon.train.workflow.group",
        "platoon.train.workflow.members",
        "platoon.train.workflow.requests",
        "platoon.train.workflow.output",
        "platoon.train.credit.group",
        "platoon.train.workflow.telemetry",
        "platoon.train.workflow.context",
    ),
    public={
        "current_entry": "platoon.train.workflow",
        "group_baselines": "platoon.train.workflow",
        "GroupRolloutWorkflow": "platoon.train.workflow",
    },
)
