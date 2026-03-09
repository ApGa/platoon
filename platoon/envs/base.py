from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable


@runtime_checkable
class Env(Protocol):
    async def reset(self) -> Observation: ...

    async def step(self, action: Action) -> Observation: ...

    async def close(self) -> None: ...

    async def observe(self) -> Observation: ...

    @property
    def task(self) -> Task: ...


@runtime_checkable
class ForkableEnv(Env, Protocol):
    async def fork(self, task: Task) -> ForkableEnv: ...


@dataclass
class Task:
    goal: str | None = None
    id: str | None = None
    max_steps: int | None = None
    misc: dict[str, Any] = field(default_factory=dict)
    fork_strategy: Literal["task", "subtask"] = "subtask"

    def __str__(self) -> str:
        if self.max_steps:
            return f"Your Goal: {self.goal}\nBudget: You have a total budget of {self.max_steps} steps to complete this task."  # noqa: E501
        else:
            return f"Your Goal: {self.goal}"

    def fork(self, goal: str, max_steps: int | None = None, task_misc: dict | None = None) -> Task | SubTask:
        
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

    def fork(self, goal: str, max_steps: int | None = None, **kwargs) -> SubTask:
        return SubTask(
            goal=goal,
            max_steps=max_steps,
            id=str(uuid.uuid4()),
            parent_tasks=self.parent_tasks + [self],
            misc=self.misc,
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


@dataclass
class Observation:
    task: Task | None = None
    finished: bool = False
    reward: float = 0.0
    misc: dict = field(default_factory=dict)


Action: TypeAlias = Any
ResetAction: Action = "RESET"
