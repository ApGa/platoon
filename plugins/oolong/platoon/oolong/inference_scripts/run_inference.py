"""Oolong inference benchmark script using OpenAI-compatible endpoints.

Usage:
    python platoon/oolong/inference_scripts/run_inference.py \
        --config platoon/oolong/configs/inference/oolong_inference.yaml
"""

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
from platoon.oolong.rollout import run_rollout, run_recursive_rollout
from platoon.oolong.tasks import (
    AnswerType,
    TaskGroup,
    get_task,
    get_task_ids,
)
from platoon.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)
_OOLONG_DELEGATION_REWARD_CAP = 0.4


@dataclass
class OolongInferenceConfig:
    inference: InferenceBenchmarkConfig
    oolong_dataset: Literal["synth", "real", "both"] = "synth"
    dataset_split: Literal["validation", "test"] = "validation"
    num_tasks: int = 100
    use_recursive_agent: bool = False
    task_id: str | None = None
    # Optional filters
    task_group: str | None = None  # counting, user, timeline
    answer_type: str | None = None  # NUMERIC, LABEL, COMPARISON, USER, MONTH_YEAR
    min_context_len: int | None = None
    max_context_len: int | None = None
    # full: run rollouts + report
    # rollouts: only collect rollouts
    # report: only build report from existing rollouts
    stage: Literal["full", "rollouts", "report"] = "full"
    shuffle_tasks: bool = False
    seed: int = 42


def reward_processor(traj: dict) -> tuple[float, dict[str, float]]:
    """Extract Oolong reward components from trajectory steps."""
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
        score += _OOLONG_DELEGATION_REWARD_CAP * subagent_success_rate

    if not rewards_dict:
        score = float(traj.get("reward", 0.0))
    return score, rewards_dict


def get_dataset_task_ids(config: OolongInferenceConfig) -> list[str]:
    """Build the list of task IDs to evaluate based on config."""
    if config.task_id is not None:
        return [config.task_id]

    # Build filter kwargs
    filter_kwargs = {}
    if config.task_group is not None:
        filter_kwargs["task_group"] = TaskGroup(config.task_group)
    if config.answer_type is not None:
        filter_kwargs["answer_type"] = AnswerType(config.answer_type)
    if config.min_context_len is not None:
        filter_kwargs["min_context_len"] = config.min_context_len
    if config.max_context_len is not None:
        filter_kwargs["max_context_len"] = config.max_context_len

    task_ids = get_task_ids(
        dataset=config.oolong_dataset,
        split=config.dataset_split,
        **filter_kwargs,
    )

    if config.shuffle_tasks:
        import random

        rng = random.Random(config.seed)
        rng.shuffle(task_ids)

    if config.num_tasks is not None and config.num_tasks > 0:
        task_ids = task_ids[: config.num_tasks]

    if not task_ids:
        raise ValueError(
            "No Oolong tasks matched the requested filters: "
            f"dataset={config.oolong_dataset}, split={config.dataset_split}, "
            f"task_group={config.task_group}, answer_type={config.answer_type}, "
            f"min_context_len={config.min_context_len}, max_context_len={config.max_context_len}."
        )

    return task_ids


async def main(args: list[str]) -> None:
    default_config = Path(__file__).parent.parent / "configs" / "inference" / "oolong_inference.yaml"
    config, _ = load_config(
        args=args,
        config_class=OolongInferenceConfig,
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
