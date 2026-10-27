"""Typed reads of a group's workflow settings (:class:`GroupSettings`).

Inside a group, ``GroupRolloutWorkflow.config`` is the ``WorkflowConfig`` of the group's environment entry
(:mod:`platoon.train.workflow.context`).  Code that runs in a group reads some of its settings through this module, for
example ``GroupSettings(workflow.config).worker_startup_budget_seconds``.

The settings here are those a partial config may lack: in this repository, tests put a ``SimpleNamespace`` with only
some settings in ``config``.  A missing setting reads as ``WorkflowConfig``'s default, so each setting has one default.
The settings that every config sets, such as ``group_size``, ``rollout_config`` or ``depth_level_weighting``, are read
from ``config`` directly.
"""

from collections.abc import Mapping
from typing import Any

from platoon.train.config_defs import StragglerPolicy, WorkflowConfig

# The default of each setting: WorkflowConfig's own.
_DEFAULTS = WorkflowConfig()


class GroupSettings:
    """A workflow config's settings by name and type.  Each property reads ``config`` when it is read, and converts the
    value as its readers did (``float``, ``int``, ``str``, ``bool``); a setting ``config`` lacks reads as
    ``WorkflowConfig``'s default."""

    def __init__(self, config: WorkflowConfig) -> None:
        self._config = config

    def _get(self, name: str) -> Any:
        return getattr(self._config, name, getattr(_DEFAULTS, name))

    # --- the group: how many members must train, and the straggler policy ----------------------------------------

    @property
    def min_successful_group_size(self) -> int:
        return self._get("min_successful_group_size")

    @property
    def straggler_timeout_seconds(self) -> float | None:
        return self._get("straggler_timeout_seconds")

    @property
    def straggler_quorum(self) -> int | None:
        return self._get("straggler_quorum")

    @property
    def straggler_policies(self) -> Mapping[str, StragglerPolicy]:
        return self._get("straggler_policies")

    def straggler_policy(self, label: str | None) -> tuple[float | None, int | None]:
        """The straggler timeout and quorum of a group whose task has ``label``: the label's own policy in
        ``straggler_policies``, else ``straggler_timeout_seconds`` and ``straggler_quorum``."""

        policy = self.straggler_policies.get(label)
        if policy is not None:
            return policy.timeout_seconds, policy.quorum
        return self.straggler_timeout_seconds, self.straggler_quorum

    # --- converting a member's rollout into datums ------------------------------------------------------------------

    @property
    def filter_message_actions(self) -> bool:
        return self._get("filter_message_actions")

    @property
    def depth_level_discount_gamma(self) -> float | None:
        return self._get("depth_level_discount_gamma")

    @property
    def subagent_sampling(self) -> tuple[float, int]:
        """The subagent datum sampling's keep probability and seed."""

        return self._get("subagent_datum_keep_probability"), self._get("subagent_datum_sampling_seed")

    # --- the rollout workers (``rollout_isolation: process``) -------------------------------------------------------

    @property
    def worker_start_method(self) -> str:
        return str(self._get("worker_start_method"))

    @property
    def worker_launch_concurrent_groups(self) -> int:
        return int(self._get("worker_launch_concurrent_groups"))

    @property
    def worker_startup_budget_seconds(self) -> float:
        return float(self._get("worker_startup_budget_seconds"))

    @property
    def worker_finalize_grace_seconds(self) -> float:
        return float(self._get("worker_finalize_grace_seconds"))

    @property
    def worker_unbounded_rollout_budget_seconds(self) -> float:
        return float(self._get("worker_unbounded_rollout_budget_seconds"))

    @property
    def worker_parent_grace_seconds(self) -> float:
        return float(self._get("worker_parent_grace_seconds"))

    @property
    def worker_termination_grace_seconds(self) -> float:
        return float(self._get("worker_termination_grace_seconds"))

    @property
    def worker_rss_soft_limit_bytes(self) -> int:
        return int(self._get("worker_rss_soft_limit_bytes"))

    @property
    def worker_mem_available_floor_bytes(self) -> int:
        return int(self._get("worker_mem_available_floor_bytes"))

    @property
    def worker_binding_poll_seconds(self) -> float:
        return float(self._get("worker_binding_poll_seconds"))

    @property
    def worker_python_env_isolation(self) -> bool:
        return bool(self._get("worker_python_env_isolation"))
