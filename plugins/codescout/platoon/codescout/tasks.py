from platoon.envs.base import Task
from typing import Dict, Optional
import numpy as np
from datasets import load_dataset

EVAL_AGENT_SERVER_IMAGE = "docker.io/adityasoni8/eval-agent-server"
SDK_SHORT_SHA = "b498a69"
ENV_SETUP_COMMANDS = ["export PIP_CACHE_DIR=~/.cache/pip"]
SYSTEM_PROMPT_FILENAME = "system_prompt.j2"
USER_PROMPT_FILENAME = "user_prompt.j2"
APPTAINER_CACHE_DIR = "/scratch/apptainer_cache"

data_loaded: bool = False
train_data_map: Optional[Dict[str, Task]] = {}
val_data_map: Optional[Dict[str, Task]] = {}

def create_task_from_instance(x: dict) -> Task:
    task = Task(
        id=x['instance_id'],
        misc=x,
    )
    return task

def load_data():
    global data_loaded, train_data_map, val_data_map
    if data_loaded:
        return train_data_map, val_data_map

    dataset = load_dataset("adityasoni17/SWE-smith-py-code-search", split='train').to_pandas()
    np.random.seed(42)
    split_indices = np.random.rand(len(dataset)) < 0.9
    train_df = dataset.iloc[split_indices]
    val_df = dataset.iloc[~split_indices]
    for _, row in train_df.iterrows():
        if len(row["problem_statement"]) > 0: #NOTE: optionally filter training instances by repo or other criteria here if needed
            train_data_map[row['instance_id']] = create_task_from_instance(row.to_dict())
    for _, row in val_df.iterrows():
        if len(row["problem_statement"]) > 0: #NOTE: optionally filter validation instances by repo or other criteria here if needed
            val_data_map[row['instance_id']] = create_task_from_instance(row.to_dict())
    data_loaded = True
    print(f"Loaded {len(train_data_map)} training instances and {len(val_data_map)} validation instances.", flush=True)
    return train_data_map, val_data_map

def get_task(task_id: str) -> Task:
    load_data()
    global train_data_map, val_data_map
    if task_id in train_data_map:
        return train_data_map[task_id]
    elif task_id in val_data_map:
        return val_data_map[task_id]
    else:
        raise ValueError(f"Task ID {task_id} not found in training or validation data.")