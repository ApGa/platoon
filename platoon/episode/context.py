from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from platoon.agents.base import Agent
    from platoon.envs.base import Env
    from platoon.episode.trajectory import BudgetTracker, Trajectory, TrajectoryCollection

current_agent: ContextVar["Agent"] = ContextVar("current_agent")
current_env: ContextVar["Env"] = ContextVar("current_env")
current_trajectory: ContextVar["Trajectory"] = ContextVar("current_trajectory")
current_trajectory_collection: ContextVar["TrajectoryCollection"] = ContextVar("current_trajectory_collection")
error_message: ContextVar[str | None] = ContextVar("error_message", default=None)
budget_tracker: ContextVar["BudgetTracker"] = ContextVar("budget_tracker")
finish_message: ContextVar[str | None] = ContextVar("finish_message", default=None)
episode_step_timeout: ContextVar[int] = ContextVar("episode_step_timeout", default=300)