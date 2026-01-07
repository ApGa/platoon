from platoon.envs.base import Task
import pandas as pd
import os
from typing import Dict, Literal, Optional, List
import numpy as np

EVAL_AGENT_SERVER_IMAGE = "ghcr.io/openhands/eval-agent-server"
SDK_SHORT_SHA = "main"
ENV_SETUP_COMMANDS = ["export PIP_CACHE_DIR=~/.cache/pip"]
PROMPT_FILENAME = "default.j2"
data_loaded: bool = False
train_data_map: Optional[Dict[str, Task]] = {}
val_data_map: Optional[Dict[str, Task]] = {}

def create_task_from_instance(x: dict) -> Task:
    task = Task(
        id=x['instance_id'],
        misc=x,
        # NOTE: optionally add new parameters to instance dicts here if needed
        # misc={
        #     "instance_id": x['instance_id'],
        #     "repo": x['repo'],
        #     "base_commit": x['base_commit'],
        #     "problem_statement": x['problem_statement'],
        #     "target": x['target'],
        #     "workspace_type": x.get("workspace_type", "docker"), # default to docker
        #     "docker_image_prefix": x.get("docker_image_prefix", "docker.io/xingyaoww/"),
        #     "dataset_type": x.get("dataset_type", "swe-bench"),
        #     "prompt_filename": x.get("prompt_filename", PROMPT_FILENAME),
        # }
    )
    return task

def load_data():
    global data_loaded, train_data_map, val_data
    if data_loaded:
        return
    data_path = os.path.join(os.path.dirname(__file__), "train.parquet") #NOTE: make it huggingface dataset if possible
    dataset = pd.read_parquet(data_path)
    np.random.seed(42)
    split_indices = np.random.rand(len(dataset)) < 0.8
    train_df = dataset.iloc[split_indices]
    val_df = dataset.iloc[~split_indices]
    for _, row in train_df.iterrows():
        train_data_map[row['instance_id']] = create_task_from_instance(row.to_dict())
    for _, row in val_df.iterrows():
        val_data_map[row['instance_id']] = create_task_from_instance(row.to_dict())
    data_loaded = True
    return train_data_map, val_data_map


# NOTE: we should have enough RAM to hold all the training instances in RAM since they are only <200MB in size, so no need to lazy load from disk for now
def get_task(task_id: str) -> Task:
    load_data()
    global train_data_map, val_data_map
    if task_id in train_data_map:
        return train_data_map[task_id]
    elif task_id in val_data_map:
        return val_data_map[task_id]
    else:
        raise ValueError(f"Task ID {task_id} not found in training or validation data.")