"""The task model: what one rollout works on (``Task``), and a subagent's task with its parent stack (``SubTask``)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Task:
    """What one rollout works on: a goal, an id, a step budget and environment-specific data."""

    # The instruction the agent works on.
    goal: str | None = None
    # The task's id within its environment.
    id: str | None = None
    # The agent's step budget (None: the rollout config's).
    max_steps: int | None = None
    # Environment-specific task data.
    misc: dict[str, Any] = field(default_factory=dict)
    # What ``fork`` makes: a plain ``Task``, or a ``SubTask`` that keeps its parents' goals.
    fork_strategy: Literal["task", "subtask"] = "subtask"

    def __str__(self) -> str:
        if self.max_steps:
            return f"Your Goal: {self.goal}\nBudget: You have a total budget of {self.max_steps} steps to complete this task."  # noqa: E501
        else:
            return f"Your Goal: {self.goal}"

    def fork(self, goal: str, max_steps: int | None = None, task_misc: dict | None = None) -> Task | SubTask:
        """A child task with a new id and ``goal``, sharing this task's ``misc`` unless ``task_misc`` is given."""
        if task_misc is None:
            task_misc = self.misc

        if self.fork_strategy == "task":
            return Task(
                goal=goal,
                max_steps=max_steps,
                id=str(uuid.uuid4()),
                misc=task_misc,
                fork_strategy=self.fork_strategy,
            )
        else:
            return SubTask(
                goal=goal,
                max_steps=max_steps,
                id=str(uuid.uuid4()),
                parent_tasks=[self],
                misc=task_misc,
            )

    @classmethod
    def from_dict(cls, task_dict: dict) -> Task:
        """A task from its fields as a dict."""
        return Task(**task_dict)


@dataclass
class SubTask(Task):
    parent_tasks: list[Task] = field(default_factory=list)

    def __str__(self) -> str:
        task_str = super().__str__()
        # Parent Tasks
        parent_tasks_str = (
            "For additional context, here are the parent tasks in the stack so far (most recent first):\n"
        )
        depth = len(self.parent_tasks)
        if depth > 0:
            # Add parent goals in reverse order (most recent first)
            for i, parent_task in enumerate(reversed(self.parent_tasks)):
                level = depth - i
                parent_tasks_str += f"Level {level}: {parent_task.goal}\n\n"
            parent_tasks_str = parent_tasks_str.rstrip()
        else:
            parent_tasks_str += "No parent tasks. This is the root task."

        return f"{task_str}\n\n{parent_tasks_str}"

    def fork(self, goal: str, max_steps: int | None = None, task_misc: dict | None = None) -> SubTask:
        if task_misc is None:
            task_misc = self.misc
        return SubTask(
            goal=goal,
            max_steps=max_steps,
            id=str(uuid.uuid4()),
            parent_tasks=self.parent_tasks + [self],
            misc=task_misc,
        )

    @classmethod
    def from_task(cls, task: Task) -> SubTask:
        if isinstance(task, SubTask):
            return task
        return SubTask(
            goal=task.goal,
            max_steps=task.max_steps,
            id=task.id,
            misc=task.misc,
        )

    @classmethod
    def from_dict(cls, task_dict: dict) -> SubTask:
        if "parent_tasks" not in task_dict:
            parent_tasks = []
        else:
            parent_tasks = [SubTask.from_dict(parent_task_dict) for parent_task_dict in task_dict["parent_tasks"]]
        return SubTask(
            goal=task_dict["goal"],
            max_steps=task_dict["max_steps"],
            id=task_dict["id"],
            parent_tasks=parent_tasks,
            misc=task_dict["misc"],
        )
