"""OpenReward inference benchmark script using OpenAI-compatible endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from platoon.inference import DefaultInferenceGroupWorkflow, InferenceBenchmarkRunner
from platoon.utils.config import load_config

from platoon.openreward.config_defs import OpenRewardInferenceConfig
from platoon.openreward.rollout import reward_processor, run_rollout
from platoon.openreward.tasks import get_task, get_task_ids

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def _attach_openreward_config(config: OpenRewardInferenceConfig) -> None:
    rollout_extra = dict(config.inference.workflow.rollout_config.extra or {})
    rollout_extra["openreward"] = asdict(config.openreward)
    config.inference.workflow.rollout_config.extra = rollout_extra


def _dataset(config: OpenRewardInferenceConfig) -> list[dict[str, str]]:
    if config.task_id is not None:
        return [{"task_id": config.task_id}]

    task_ids = get_task_ids(
        config.openreward,
        split=config.openreward.split,
        limit=config.openreward.train_task_limit,
    )
    if config.shuffle_tasks:
        rng = random.Random(config.seed)
        rng.shuffle(task_ids)
    return [{"task_id": task_id} for task_id in task_ids]


def _normalize_stage(stage: str) -> Literal["full", "rollouts", "report"]:
    if stage not in {"full", "rollouts", "report"}:
        raise ValueError(f"Invalid stage {stage!r}; expected full, rollouts, or report")
    return stage  # type: ignore[return-value]


async def main(args: list[str]) -> None:
    default_config = Path(__file__).parents[1] / "configs" / "inference" / "toolathlon_openhands_inference.yaml"
    config, _ = load_config(
        args=args,
        config_class=OpenRewardInferenceConfig,
        default_config_path=str(default_config),
    )
    config: OpenRewardInferenceConfig = config
    _attach_openreward_config(config)

    stage = _normalize_stage(config.stage)
    dataset = [] if stage == "report" else _dataset(config)

    workflow = DefaultInferenceGroupWorkflow(
        rollout_fn=run_rollout,
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
        run_rollouts=stage in {"full", "rollouts"},
        generate_report=stage in {"full", "report"},
    )

    if "summary" in result:
        logger.info("Inference benchmark complete. Final report saved under: %s", config.inference.output_dir)
        print(json.dumps(result["summary"], indent=2))
    else:
        logger.info("Inference rollout stage complete. Output dir: %s", config.inference.output_dir)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
