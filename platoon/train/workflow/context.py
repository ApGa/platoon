"""What runs in a group's context: the context variables ``GroupRolloutWorkflow.arun_episode`` sets for the group it
runs (its settings, its entry, where its statistics go) and its members' readiness callback; ``current_entry()``.
"""

from contextvars import ContextVar

from platoon.train.config_defs import WorkflowConfig
from platoon.utils.stats_tracker import DeferredStats

# Set by ``_run_group_rollouts`` before it creates its member tasks, so each
# member can report readiness back to its group's admission permit without
# changing ``arun_episode_single``'s signature (custom workflows override it).
_group_ready_callback: ContextVar = ContextVar("platoon_group_ready_callback", default=None)
# Set by ``arun_episode`` for the group it runs: where the group's own statistics go (see platoon.train.workflow.group's
# module docstring).
_group_stats: ContextVar[DeferredStats | None] = ContextVar("platoon_group_stats", default=None)
# Set by ``arun_episode`` for the group it runs: the settings of the group's environment entry, which
# ``GroupRolloutWorkflow.config`` (platoon.train.workflow.group) reads inside the group, and the entry's label
# (``current_entry``).
_group_config: ContextVar[WorkflowConfig | None] = ContextVar("platoon_group_config", default=None)
_group_entry: ContextVar[str | None] = ContextVar("platoon_group_entry", default=None)


def current_entry() -> str | None:
    """The label of the environment entry whose group runs in this context (inside ``arun_episode``), such as a
    mixture entry's name; None outside a group, and for a row that names no entry of the run."""

    return _group_entry.get()
