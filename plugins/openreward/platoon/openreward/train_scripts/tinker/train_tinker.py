"""OpenReward training script using the Tinker backend."""

from __future__ import annotations

import asyncio
import random
import sys
from dataclasses import asdict, replace
from pathlib import Path

from datasets import Dataset
from platoon.train.tinker.dataset_order import PRESERVE_DATASET_ORDER_COLUMN
from platoon.train.tinker.rl import PlatoonTinkerRLTrainer
from platoon.train.tinker.workflows import GroupRolloutWorkflow
from platoon.utils.config import load_config

from platoon.openreward.mixture import materialize_balanced_record_order
from platoon.openreward.rewards import reward_processor
from platoon.openreward.rollout import run_rollout
from platoon.openreward.tasks import (
    OPENREWARD_ENVIRONMENT_COLUMN,
    OPENREWARD_SAMPLING_WEIGHT_COLUMN,
    get_task,
    get_task_records,
)
from platoon.openreward.tinker_config import OpenRewardTinkerTrainerConfig


def _attach_openreward_config(config: OpenRewardTinkerTrainerConfig) -> None:
    payload = asdict(config.openreward)
    for workflow_config in (config.train.workflow_config, config.eval.workflow_config):
        rollout_extra = dict(workflow_config.rollout_config.extra or {})
        rollout_extra["openreward"] = payload
        workflow_config.rollout_config.extra = rollout_extra


def _select_task_records(
    config: OpenRewardTinkerTrainerConfig,
    *,
    evaluation: bool,
    seed: int,
) -> list[dict]:
    records = get_task_records(config.openreward, evaluation=evaluation)
    if not evaluation and config.openreward.is_mixture:
        ordered = materialize_balanced_record_order(
            records,
            environment_key=OPENREWARD_ENVIRONMENT_COLUMN,
            sampling_weight_key=OPENREWARD_SAMPLING_WEIGHT_COLUMN,
            global_batch_size=config.train.batch_size,
            seed=seed,
            preserve_order_key=PRESERVE_DATASET_ORDER_COLUMN,
        )
        return ordered

    rng = random.Random(seed)
    rng.shuffle(records)
    return records


async def main(args: list[str]) -> None:
    default_config = Path(__file__).parents[2] / "configs" / "tinker" / "toolathlon_openhands_tinker.yaml"
    config, _ = load_config(
        args=args,
        config_class=OpenRewardTinkerTrainerConfig,
        default_config_path=str(default_config),
    )
    config: OpenRewardTinkerTrainerConfig = config
    _attach_openreward_config(config)

    train_records = _select_task_records(
        config,
        evaluation=False,
        seed=config.seed,
    )
    eval_records = _select_task_records(
        config,
        evaluation=True,
        seed=config.seed + 1,
    )

    train_dataset = Dataset.from_list(train_records)
    eval_dataset = Dataset.from_list(eval_records)

    trainer = PlatoonTinkerRLTrainer(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    async with trainer:
        train_workflow = GroupRolloutWorkflow(
            rollout_fn=run_rollout,
            get_task_fn=get_task,
            config=config.train.workflow_config,
            model_info=trainer.model_info,
            log_path=trainer.run_log_path,
            stats_scope="train",
            filter_errors=config.train.workflow_config.filter_errors,
            reward_processor=reward_processor,
        )

        eval_workflow_config = config.eval.workflow_config
        # Evaluation reports rollout metrics but never trains, so datum-level
        # compute filtering is deliberately a train-only policy.
        eval_workflow_config.filter_zero_advantage_datums = False
        if eval_workflow_config.group_size != 1:
            # ``replace`` rather than re-listing fields: the old field-subset
            # construction silently dropped every key it did not name, so a
            # setting present in the yaml -- including the rollout isolation
            # keys -- would be quietly discarded for eval only.  An unisolated
            # eval rollout reintroduces exactly the unkillable-cell and
            # crash-containment hazards process isolation removes, so the keys
            # must survive.  Eval forces exactly three fields and nothing else.
            eval_workflow_config = replace(
                eval_workflow_config,
                group_size=1,
                filter_errors=False,
                filter_zero_advantage_datums=False,
                # Both of these are validated against group_size, which eval
                # forces to 1, so any configured value above 1 would make
                # WorkflowConfig.__post_init__ raise at startup.  A one-member
                # group has nothing to be a straggler against and cannot
                # require more than one successful member.  The old
                # field-subset construction hid this by dropping the keys along
                # with everything else it did not name.
                straggler_quorum=None,
                min_successful_group_size=1,
            )

        eval_workflow = GroupRolloutWorkflow(
            rollout_fn=run_rollout,
            get_task_fn=get_task,
            config=eval_workflow_config,
            model_info=trainer.model_info,
            log_path=trainer.run_log_path,
            stats_scope="eval",
            filter_errors=False,
            reward_processor=reward_processor,
        )

        await trainer.train(
            train_workflow=train_workflow,
            eval_workflow=eval_workflow,
        )


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
