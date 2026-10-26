"""What the trainer needs of a rollout workflow (:class:`RolloutWorkflow`): ``arun_episode`` turns one task's row
into the datums of its group.  :class:`~platoon.train.workflow.group.GroupRolloutWorkflow` is platoon's.
"""

from typing import Protocol

import tinker


class RolloutWorkflow(Protocol):
    """Protocol for rollout workflows used in tinker RL training.

    Implementations should receive model_info and other dependencies via constructor.
    """

    async def arun_episode(self, data: dict) -> list[tinker.Datum] | None: ...
