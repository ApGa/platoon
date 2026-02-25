from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Protocol, runtime_checkable

from platoon.envs.base import Task
from platoon.episode.context import (
    current_trajectory,
    current_trajectory_collection,
    finish_message,
)


@dataclass
class TrajectoryStep:
    misc: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParentInfo:
    id: str
    fork_step: int


@dataclass
class Trajectory:
    id: str
    task: Task | None = None
    parent_info: ParentInfo | None = None
    steps: list[TrajectoryStep] = field(default_factory=list)
    reward: float = 0.0
    finish_message: str | None = None
    error_message: str | None = None
    misc: dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: TrajectoryStep) -> None:
        self.steps.append(step)
        if finish_message.get(None) is not None:
            self.finish_message = finish_message.get()
        reward = getattr(step, "reward", None)
        if reward is not None:
            self.reward += reward


@runtime_checkable
class TrajectoryEventHandler(Protocol):
    def on_trajectory_created(self, trajectory: Trajectory) -> None:
        pass

    def on_trajectory_step_added(self, trajectory: Trajectory, step: TrajectoryStep) -> None:
        pass

    def on_trajectory_task_set(self, trajectory: Trajectory, task: Task | None) -> None:
        pass

    def on_trajectory_finished(self, trajectory: Trajectory) -> None:
        """Called when a trajectory is finalized.

        Consumers can use this to capture any final updates such as total reward,
        finish message, or error message that may have been set outside of a step event.
        """
        pass


@dataclass
class TrajectoryCollection:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trajectories: dict[str, Trajectory] = field(default_factory=dict)
    event_handlers: list[TrajectoryEventHandler] = field(default_factory=list)

    def __post_init__(self):
        event_handlers = self.event_handlers
        self.event_handlers = []
        self.register_event_handlers(event_handlers)
        self._step_count = sum(len(traj.steps) for traj in self.trajectories.values())

    @property
    def step_count(self) -> int:
        return self._step_count

    def register_event_handlers(
        self, event_handlers: TrajectoryEventHandler | Iterable[TrajectoryEventHandler]
    ) -> None:
        if isinstance(event_handlers, Iterable) and not isinstance(event_handlers, (str, bytes)):
            for h in event_handlers:
                if isinstance(h, TrajectoryEventHandler):
                    self.event_handlers.append(h)
                else:
                    raise ValueError(f"Handler {h} is not a TrajectoryEventHandler")
        elif isinstance(event_handlers, TrajectoryEventHandler):
            self.event_handlers.append(event_handlers)
        else:
            raise ValueError(f"Handler {event_handlers} is not a TrajectoryEventHandler")

    def create_trajectory(self, parent_traj: Trajectory | None = None) -> Trajectory:
        parent_info = (
            ParentInfo(
                id=parent_traj.id,
                fork_step=len(parent_traj.steps),
            )
            if parent_traj is not None
            else None
        )

        trajectory = Trajectory(id=str(uuid.uuid4()), parent_info=parent_info)
        self.trajectories[trajectory.id] = trajectory
        for handler in self.event_handlers:
            try:
                handler.on_trajectory_created(trajectory)
            except Exception:
                # Best-effort: do not let handlers break rollout
                pass
        return trajectory

    def add_trajectory_step(self, trajectory_id: str, step: TrajectoryStep) -> None:
        self._step_count += 1
        self.trajectories[trajectory_id].add_step(step)
        for handler in self.event_handlers:
            try:
                handler.on_trajectory_step_added(self.trajectories[trajectory_id], step)
            except Exception as e:
                print(f"Error in on_trajectory_step_added for trajectory {trajectory_id}: {str(e)}")
                pass

    def set_trajectory_task(self, trajectory_id: str, task: Task | None) -> None:
        trajectory = self.trajectories[trajectory_id]
        trajectory.task = task
        for handler in self.event_handlers:
            try:
                handler.on_trajectory_task_set(trajectory, task)
            except Exception as e:
                print(f"Error in on_trajectory_task_set for trajectory {trajectory_id}: {str(e)}")
                pass

    def finish_trajectory(self, trajectory_id: str) -> None:
        """Mark a trajectory as finished and notify handlers."""
        trajectory = self.trajectories[trajectory_id]
        for handler in self.event_handlers:
            try:
                handler.on_trajectory_finished(trajectory)
            except Exception as e:
                # Best-effort: do not let handlers break rollout
                print(f"Error in on_trajectory_finished for trajectory {trajectory_id}: {str(e)}")
                pass

    def to_dict(self) -> dict[str, Any]:
        """Serialize without handlers or file pointers for portability.

        Avoids dataclasses.asdict to skip non-serializable fields (handlers).
        """
        return {"id": self.id, "trajectories": {traj_id: asdict(traj) for traj_id, traj in self.trajectories.items()}}


# TODO: This is minimal for now on purpose, to better understand needs.
# We can consider expanding the protocol later to add more informative methods for the user to query about the budget.


class BudgetExceededError(ValueError):
    """Raised when a budget limit would be exceeded.

    Attributes:
        reason: A short tag identifying the kind of budget that was exceeded
                (e.g. ``"step_budget"`` or ``"depth"``).
        guidance: An actionable suggestion for the caller describing how to
                  work around the limitation.
    """

    def __init__(self, message: str, *, reason: str = "step_budget", guidance: str = ""):
        super().__init__(message)
        self.reason = reason
        self.guidance = guidance


@runtime_checkable
class BudgetTracker(Protocol):
    def reserve_budget(self, requested_budget: float, raise_on_failure: bool = False) -> bool: ...

    def release_budget(self, amount_to_release: float) -> None: ...

    def remaining_budget(self) -> float:
        return self.remaining_budget_for(current_trajectory.get().id)

    def remaining_budget_for(self, trajectory_id: str) -> float: ...

    def used_budget(self) -> float:
        return self.used_budget_for(current_trajectory.get().id)

    def used_budget_for(self, trajectory_id: str) -> float: ...


@dataclass
class StepBudgetTracker(BudgetTracker):
    reserved_trajectory_budgets: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))

    def _allocated_budget(self, trajectory_id: str) -> float:
        traj = current_trajectory_collection.get().trajectories[trajectory_id]
        return traj.task.max_steps or float("inf")

    # TODO: This is inefficient, we might just want to store the child map as we build the tree.
    def _iter_descendant_trajectory_ids(self, trajectory_id: str) -> Iterable[str]:
        collection = current_trajectory_collection.get()
        stack: list[str] = [trajectory_id]
        seen: set[str] = set()
        while stack:
            current_id = stack.pop()
            for child_id, child_traj in collection.trajectories.items():
                parent_info = child_traj.parent_info
                if parent_info is not None and parent_info.id == current_id and child_id not in seen:
                    seen.add(child_id)
                    yield child_id
                    stack.append(child_id)

    def _used_budget(self, trajectory_id: str, recursive: bool = False) -> float:
        collection = current_trajectory_collection.get()
        traj = collection.trajectories[trajectory_id]
        used_steps = len(traj.steps)
        if recursive:
            for desc_id in self._iter_descendant_trajectory_ids(trajectory_id):
                used_steps += len(collection.trajectories[desc_id].steps)
        return used_steps

    def _reserved_budget(self, trajectory_id: str) -> float:
        return self.reserved_trajectory_budgets[trajectory_id]

    def used_budget_for(self, trajectory_id: str) -> float:
        return self._used_budget(trajectory_id, recursive=True)

    def remaining_budget_for(self, trajectory_id: str) -> float:
        remaining = (
            self._allocated_budget(trajectory_id)
            - self.used_budget_for(trajectory_id)
            - self._reserved_budget(trajectory_id)
        )
        return remaining

    def reserve_budget(self, requested_budget: float, raise_on_failure: bool = False) -> bool:
        curr_traj_id = current_trajectory.get().id
        if self.remaining_budget() < requested_budget:
            if raise_on_failure:
                raise BudgetExceededError(
                    f"Requested step budget {requested_budget} exceeds remaining budget {self.remaining_budget()}.",
                    reason="step_budget",
                    guidance=(
                        "Note: launch_subagent will automatically reserve max_steps + 1 steps "
                        "since you will need one or more steps to process the result of the "
                        "subagent and complete the task. "
                        "You could try requesting a smaller budget or perform the task yourself."
                    ),
                )
            return False
        self.reserved_trajectory_budgets[curr_traj_id] += requested_budget
        return True

    def release_budget(self, amount_to_release: float) -> None:
        curr_traj_id = current_trajectory.get().id
        self.reserved_trajectory_budgets[curr_traj_id] -= amount_to_release
        if self.reserved_trajectory_budgets[curr_traj_id] < 0:
            self.reserved_trajectory_budgets[curr_traj_id] = 0


@dataclass
class DepthAwareStepBudgetTracker(BudgetTracker):
    """A budget tracker where subagent steps do NOT consume parent budget.

    Each trajectory is independently bounded by its own ``task.max_steps``.
    Optionally, the maximum depth of the hierarchical trajectory tree can be
    capped via *max_depth* (root trajectory is at depth 0).
    """

    max_depth: int | None = None

    # ---- internal helpers ----

    def _allocated_budget(self, trajectory_id: str) -> float:
        traj = current_trajectory_collection.get().trajectories[trajectory_id]
        return traj.task.max_steps or float("inf")

    def _trajectory_depth(self, trajectory_id: str) -> int:
        """Return the depth of *trajectory_id* in the tree (root = 0)."""
        collection = current_trajectory_collection.get()
        depth = 0
        traj = collection.trajectories[trajectory_id]
        while traj.parent_info is not None:
            depth += 1
            traj = collection.trajectories[traj.parent_info.id]
        return depth

    # ---- BudgetTracker interface ----

    def used_budget_for(self, trajectory_id: str) -> float:
        """Only the trajectory's own steps count towards its budget."""
        collection = current_trajectory_collection.get()
        return len(collection.trajectories[trajectory_id].steps)

    def remaining_budget_for(self, trajectory_id: str) -> float:
        return self._allocated_budget(trajectory_id) - self.used_budget_for(trajectory_id)

    def reserve_budget(self, requested_budget: float, raise_on_failure: bool = False) -> bool:
        """Check depth constraint only; subagent steps are not reserved from the parent."""
        if self.max_depth is not None:
            curr_traj_id = current_trajectory.get().id
            current_depth = self._trajectory_depth(curr_traj_id)
            if current_depth + 1 > self.max_depth:
                if raise_on_failure:
                    raise BudgetExceededError(
                        f"Launching a subagent from depth {current_depth} would exceed "
                        f"the maximum allowed depth of {self.max_depth}.",
                        reason="depth",
                        guidance=(
                            "The maximum depth for hierarchical delegation has been reached. "
                            "You should perform the task yourself instead of delegating."
                        ),
                    )
                return False
        return True

    def release_budget(self, amount_to_release: float) -> None:
        """No-op: subagent steps do not consume parent budget."""
        pass
