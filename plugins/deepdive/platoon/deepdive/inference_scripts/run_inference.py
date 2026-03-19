"""DeepDive inference benchmark script using OpenAI-compatible endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from platoon.inference import (
    DefaultInferenceGroupWorkflow,
    InferenceBenchmarkConfig,
    InferenceBenchmarkRunner,
)
from platoon.deepdive.rollout import run_recursive_rollout, run_rollout
from platoon.deepdive.tasks import get_task, get_task_ids
from platoon.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)
_DEEPDIVE_DELEGATION_REWARD_CAP = 0.4


@dataclass
class DeepDiveInferenceConfig:
    inference: InferenceBenchmarkConfig
    dataset_split: Literal["qa_rl", "qa_sft"] = "qa_rl"
    num_tasks: int = 100
    use_recursive_agent: bool = True
    task_id: str | None = None
    stage: Literal["full", "rollouts", "report"] = "full"
    shuffle_tasks: bool = False
    seed: int = 42


def reward_processor(traj: dict) -> tuple[float, dict[str, float]]:
    rewards_dict: dict[str, float] = {}
    for step in traj.get("steps", []):
        reward_misc = step.get("misc", {}).get("reward_misc", {})
        for reward_key, reward_value in reward_misc.items():
            if not reward_key.startswith("reward/"):
                continue
            rewards_dict[reward_key] = rewards_dict.get(reward_key, 0.0) + float(reward_value)

    score = rewards_dict.get("reward/success", 0.0)
    launched = rewards_dict.get("reward/subagent_launched", 0.0)
    if launched > 0:
        subagent_success_rate = rewards_dict.get("reward/subagent_succeeded", 0.0) / launched
        score += _DEEPDIVE_DELEGATION_REWARD_CAP * subagent_success_rate

    if not rewards_dict:
        score = float(traj.get("reward", 0.0))
    return score, rewards_dict


def get_dataset_task_ids(config: DeepDiveInferenceConfig) -> list[str]:
    if config.task_id is not None:
        return [config.task_id]

    task_ids = get_task_ids(config.dataset_split)
    selected_task_ids = list(task_ids)
    if config.shuffle_tasks:
        import random

        rng = random.Random(config.seed)
        rng.shuffle(selected_task_ids)
    if config.num_tasks is not None and config.num_tasks > 0:
        selected_task_ids = selected_task_ids[: config.num_tasks]
    return selected_task_ids


async def main(args: list[str]) -> None:
    default_config = Path(__file__).parent.parent / "configs" / "inference" / "deepdive_inference.yaml"
    config, _ = load_config(
        args=args,
        config_class=DeepDiveInferenceConfig,
        default_config_path=str(default_config),
    )

    rollout_fn = run_recursive_rollout if config.use_recursive_agent else run_rollout
    if config.stage == "report":
        dataset = []
    else:
        task_ids = get_dataset_task_ids(config)
        dataset = [{"task_id": task_id} for task_id in task_ids]

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

    if "summary" in result:
        logger.info(
            "Inference benchmark complete. Final report saved under: %s",
            config.inference.output_dir,
        )
        print(json.dumps(result["summary"], indent=2))
    else:
        logger.info("Inference rollout stage complete. Output dir: %s", config.inference.output_dir)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
