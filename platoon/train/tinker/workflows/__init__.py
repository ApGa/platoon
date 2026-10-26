"""Moved: platoon 0.5.0 moved the rollout workflows to :mod:`platoon.train.workflow`.  The names this package exported
still import from here, each with a ``PlatoonDeprecationWarning``, until platoon 0.6.0 removes it."""

from platoon.train.tinker._moved import moved as _moved

__getattr__, __dir__ = _moved(
    globals(),
    {
        "RolloutWorkflow": "platoon.train.workflow",
        "GroupRolloutWorkflow": "platoon.train.workflow",
    },
    submodules=("base", "group_rollout_workflow"),
)

__all__ = [
    "RolloutWorkflow",
    "GroupRolloutWorkflow",
]
