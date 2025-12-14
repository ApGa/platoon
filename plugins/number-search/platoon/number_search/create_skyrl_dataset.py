"""
Convert number_search JSONL data to SkyRL-compatible parquet format.

The SkyRL format expects:
- uid: unique identifier for the sample
- data_source: name/identifier of the data source
- prompt: conversation format (list of message dicts)
- env_class: environment class identifier  
- env_extras: additional metadata (we put task_id here)
- reward_spec: reward configuration (optional, we use env-based rewards)
"""

import json
import pathlib
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Any


def load_jsonl(file_path: pathlib.Path) -> list[dict[str, Any]]:
    """Load tasks from JSONL file."""
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def convert_task_to_skyrl_format(task: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a number_search task to SkyRL dataset format.
    
    Since Platoon uses task_id to retrieve the full task from the task registry,
    we use an empty prompt and pass the task_id in env_extras.
    """
    task_id = task["id"]
    
    # Empty prompt - the actual prompt is built by the agent/env using task_id
    # We include a minimal placeholder to satisfy the format
    prompt = [
        {
            "role": "user",
            "content": "",  # Empty - task details come from task registry via task_id
        }
    ]
    
    return {
        "uid": task_id,  # Unique identifier for this sample
        "data_source": "number_search",
        "prompt": prompt,
        "env_class": "number_search",  # Environment class identifier
        "env_extras": {
            "task_id": task_id,  # Used by PlatoonSkyRLGenerator to retrieve the task
        },
        "reward_spec": {
            "method": "rule",
            "ground_truth": task["misc"]["target"],  # The target number
        },
        "extra_info": {
            "goal": task["goal"],
            "low": task["misc"]["low"],
            "high": task["misc"]["high"],
            "target": task["misc"]["target"],
            "max_steps": task["max_steps"],
        },
    }


def save_as_parquet(data: list[dict[str, Any]], output_path: pathlib.Path) -> None:
    """
    Save data as parquet file.
    
    We need to handle nested dicts (prompt, env_extras, reward_spec, extra_info)
    by serializing them as JSON strings, as parquet doesn't handle arbitrary nested structures well.
    """
    # Serialize nested structures as JSON strings for parquet compatibility
    rows = []
    for item in data:
        row = {
            "uid": item["uid"],
            "data_source": item["data_source"],
            "prompt": json.dumps(item["prompt"]),  # Serialize as JSON string
            "env_class": item["env_class"],
            **item["env_extras"],
            # "env_extras": json.dumps(item["env_extras"]),  # Serialize as JSON string
            # "reward_spec": json.dumps(item["reward_spec"]),  # Serialize as JSON string
            # "extra_info": json.dumps(item["extra_info"]),  # Serialize as JSON string
        }
        rows.append(row)
    
    # Create PyArrow table
    table = pa.Table.from_pylist(rows)
    
    # Write to parquet
    pq.write_table(table, output_path)
    print(f"Saved {len(rows)} samples to {output_path}")


def main():
    parent_dir = pathlib.Path(__file__).parent
    
    # Load existing JSONL files
    train_jsonl = parent_dir / "number_search_train.jsonl"
    val_jsonl = parent_dir / "number_search_val.jsonl"
    
    print(f"Loading training data from {train_jsonl}...")
    train_tasks = load_jsonl(train_jsonl)
    print(f"Loaded {len(train_tasks)} training tasks")
    
    print(f"Loading validation data from {val_jsonl}...")
    val_tasks = load_jsonl(val_jsonl)
    print(f"Loaded {len(val_tasks)} validation tasks")
    
    # Convert to SkyRL format
    print("Converting to SkyRL format...")
    train_data = [convert_task_to_skyrl_format(task) for task in train_tasks]
    val_data = [convert_task_to_skyrl_format(task) for task in val_tasks]
    
    # Save as parquet
    train_parquet = parent_dir / "number_search_train.parquet"
    val_parquet = parent_dir / "number_search_val.parquet"
    
    save_as_parquet(train_data, train_parquet)
    save_as_parquet(val_data, val_parquet)
    
    print("\nDone! Created:")
    print(f"  - {train_parquet}")
    print(f"  - {val_parquet}")
    
    # Print a sample for verification
    print("\nSample training record:")
    sample = train_data[0]
    print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()

