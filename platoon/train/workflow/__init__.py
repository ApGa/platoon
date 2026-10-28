"""One group of rollouts: ``GroupRolloutWorkflow`` (``group``) and its parts.

- ``group``: the workflow, ``GroupRolloutWorkflow``;
- ``context``: what runs in a group's context (``current_entry()``);
- ``members``: runs a group's members (the startup gate, the lease blocks, the straggler policy);
- ``start_limit``: limits the rollouts starting at once;
- ``requests``: builds a member's worker request;
- ``output``: the group's output and its members';
- ``crediting``: a group's credit under its entry's credit estimator;
- ``telemetry``: the rollout workers' telemetry;
- ``settings``: typed reads of a group's settings (``GroupSettings``);
- ``protocol``: what the trainer needs of a workflow (``RolloutWorkflow``).

The group's baselines, error-token filter and verdict are :mod:`platoon.train.credit.group`.  The names below import
from here.
"""

from typing import TYPE_CHECKING, Any

# Public name -> the module that defines it.
_EXPORTS = {
    "GroupRolloutWorkflow": "platoon.train.workflow.group",
    "RolloutWorkflow": "platoon.train.workflow.protocol",
    "current_entry": "platoon.train.workflow.context",
    "group_baselines": "platoon.train.credit.group",
}

__all__ = [
    "GroupRolloutWorkflow",
    "RolloutWorkflow",
    "current_entry",
    "group_baselines",
]


def __getattr__(name: str) -> Any:
    import importlib

    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


if TYPE_CHECKING:
    from platoon.train.credit.group import group_baselines
    from platoon.train.workflow.context import current_entry
    from platoon.train.workflow.group import GroupRolloutWorkflow
    from platoon.train.workflow.protocol import RolloutWorkflow
