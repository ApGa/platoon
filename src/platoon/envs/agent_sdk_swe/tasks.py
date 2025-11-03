from platoon.envs.base import Task
from datasets import load_dataset
# Logic that creates a Task object from instance id
"""
Keep the Huggingface dataset cached here.
Load all data from the huggingface dataset for the given instance id
"""
# TODO: does global variables work properly here??
swe_gym_dataset = None
# TODO: make it flexible to load datasets other than SWE-Gym
def get_task(instance_id: str) -> Task:
    global swe_gym_dataset
    if not swe_gym_dataset:
        dataset = load_dataset("SWE-Gym/SWE-Gym", split="train")
        swe_gym_dataset = {item["instance_id"]: item for item in dataset}
    assert instance_id in swe_gym_dataset, f"Instance id {instance_id} not found in SWE-Gym dataset"
    instance = swe_gym_dataset[instance_id]
    task = Task(
        id=instance["instance_id"],
        misc=instance
    )
    return task

