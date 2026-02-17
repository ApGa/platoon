"""AppWorld inference benchmark script using OpenAI-compatible endpoints.

Usage:
    python -m platoon.appworld.run_inference \
        --config platoon/appworld/configs/inference/appworld_inference.yaml
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from appworld import load_task_ids

from platoon.appworld.rollout import run_recursive_rollout, run_rollout
from platoon.appworld.tasks import get_task
from platoon.inference import (
    DefaultInferenceGroupWorkflow,
    InferenceBenchmarkConfig,
    InferenceBenchmarkRunner,
)
from platoon.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@dataclass
class AppWorldInferenceConfig:
    inference: InferenceBenchmarkConfig
    dataset_split: Literal["train", "dev", "test_normal", "test_challenge"] = "dev"
    num_tasks: int | None = None
    use_recursive_agent: bool = True
    # Optional one-task quick path.
    task_id: str | None = None
    # full: run rollouts + report
    # rollouts: only collect rollouts
    # report: only build report from existing rollouts
    stage: Literal["full", "rollouts", "report"] = "full"
    shuffle_tasks: bool = False
    seed: int = 42


def reward_processor(traj: dict) -> tuple[float, dict[str, float]]:
    """Extract AppWorld reward components from trajectory steps."""
    rewards_dict: dict[str, float] = {}
    for step in traj.get("steps", []):
        reward_misc = step.get("misc", {}).get("reward_misc", {})
        for reward_key, reward_value in reward_misc.items():
            if not reward_key.startswith("reward/"):
                continue
            rewards_dict[reward_key] = rewards_dict.get(reward_key, 0.0) + float(reward_value)

    # Match AppWorld training behavior: keep auxiliary reward bounded.
    success_reward = rewards_dict.get("reward/success", 0.0)
    other_rewards = min(sum(rewards_dict.values()) - success_reward, 0.4)
    score = success_reward + other_rewards
    if not rewards_dict:
        score = float(traj.get("reward", 0.0))
    return float(score), rewards_dict


def get_dataset_task_ids(config: AppWorldInferenceConfig) -> list[str]:
    if config.task_id is not None:
        return [config.task_id]

    task_ids = list(load_task_ids(dataset_name=config.dataset_split))
    if config.shuffle_tasks:
        import random

        rng = random.Random(config.seed)
        rng.shuffle(task_ids)
    if config.num_tasks is not None and config.num_tasks > 0:
        return task_ids[: config.num_tasks]
    return task_ids


async def main(args: list[str]) -> None:
    default_config = Path(__file__).parent / "configs" / "inference" / "appworld_inference.yaml"
    config, _ = load_config(
        args=args,
        config_class=AppWorldInferenceConfig,
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
    run_rollouts = config.stage in {"full", "rollouts"}
    generate_report = config.stage in {"full", "report"}
    result = await runner.arun(
        dataset=dataset,
        resume=config.inference.resume,
        run_rollouts=run_rollouts,
        generate_report=generate_report,
    )

    if "summary" in result:
        logger.info("Inference benchmark complete. Final report saved under: %s", config.inference.output_dir)
        print(json.dumps(result["summary"], indent=2))
    else:
        logger.info("Inference rollout stage complete. Output dir: %s", config.inference.output_dir)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
