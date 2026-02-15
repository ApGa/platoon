from platoon.envs.base import Task


def get_task(task_id: str) -> Task:
    return Task(id=task_id)
