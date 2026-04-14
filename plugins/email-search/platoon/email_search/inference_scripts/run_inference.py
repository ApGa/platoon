"""Email-search inference benchmark script using OpenAI-compatible endpoints."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import random
import sys
from typing import Literal

from platoon.email_search.rollout import run_recursive_rollout, run_rollout
from platoon.email_search.tasks import get_task, get_task_ids
from platoon.inference import (
    DefaultInferenceGroupWorkflow,
    InferenceBenchmarkConfig,
    InferenceBenchmarkRunner,
)
from platoon.utils.config import load_config

logger = logging.getLogger(__name__)
_EMAIL_SEARCH_DELEGATION_REWARD_CAP = 0.1
_DEFAULT_REWARD_KEYS = {
    "reward/success": 0.0,
    "reward/subagent_launched": 0.0,
    "reward/subagent_succeeded": 0.0,
}


@dataclass
class EmailSearchInferenceConfig:
    inference: InferenceBenchmarkConfig
    dataset_split: Literal["train", "test"] = "test"
    num_tasks: int = 100
    max_messages: int | None = 1
    exclude_known_bad_queries: bool = True
    use_recursive_agent: bool = True
    task_id: str | None = None
    stage: Literal["full", "rollouts", "report"] = "full"
    shuffle_tasks: bool = False
    seed: int = 42


def reward_processor(traj: dict) -> tuple[float, dict[str, float]]:
    rewards_dict: dict[str, float] = dict(_DEFAULT_REWARD_KEYS)
    found_reward_metric = False
    for step in traj.get("steps", []):
        reward_misc = step.get("misc", {}).get("reward_misc", {})
        for reward_key, reward_value in reward_misc.items():
            if reward_key.startswith("reward/"):
                found_reward_metric = True
                rewards_dict[reward_key] = rewards_dict.get(reward_key, 0.0) + float(reward_value)

    score = rewards_dict["reward/success"]
    launched = rewards_dict["reward/subagent_launched"]
    if launched > 0:
        subagent_success_rate = rewards_dict["reward/subagent_succeeded"] / launched
        score += _EMAIL_SEARCH_DELEGATION_REWARD_CAP * subagent_success_rate

    if not found_reward_metric:
        score = float(traj.get("reward", 0.0))
    return score, rewards_dict


def get_dataset_task_ids(config: EmailSearchInferenceConfig) -> list[str]:
    if config.task_id is not None:
        return [config.task_id]

    task_ids = list(
        get_task_ids(
            split=config.dataset_split,
            max_messages=config.max_messages,
            exclude_known_bad_queries=config.exclude_known_bad_queries,
        )
    )
    if config.shuffle_tasks:
        rng = random.Random(config.seed)
        rng.shuffle(task_ids)
    if config.num_tasks is not None and config.num_tasks > 0:
        task_ids = task_ids[: config.num_tasks]
    return task_ids


async def main(args: list[str]) -> None:
    default_config = (
        Path(__file__).parent.parent / "configs" / "inference" / "email_search_inference.yaml"
    )
    config, _ = load_config(
        args=args,
        config_class=EmailSearchInferenceConfig,
        default_config_path=str(default_config),
    )

    rollout_fn = run_recursive_rollout if config.use_recursive_agent else run_rollout
    dataset = [] if config.stage == "report" else [{"task_id": task_id} for task_id in get_dataset_task_ids(config)]

    workflow = DefaultInferenceGroupWorkflow(
        rollout_fn=rollout_fn,
        get_task_fn=get_task,
        config=config.inference.workflow,
        model_name=config.inference.model_name,
        model_endpoint=config.inference.model_endpoint,
        model_api_key=config.inference.model_api_key,
        reward_processor=reward_processor,
    )
    runner = InferenceBenchmarkRunner(
        workflow=workflow,
        output_dir=config.inference.output_dir,
    )
    result = await runner.arun(
        dataset=dataset,
        resume=config.inference.resume,
        run_rollouts=config.stage in {"full", "rollouts"},
        generate_report=config.stage in {"full", "report"},
    )

    print(json.dumps(result.get("summary", result), indent=2))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    asyncio.run(main(sys.argv[1:]))
