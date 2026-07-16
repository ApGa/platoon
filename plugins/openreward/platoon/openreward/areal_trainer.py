"""AReaL trainer specialization for balanced OpenReward mixtures."""

from __future__ import annotations

from typing import Any

from platoon.train.areal import PlatoonArealRLTrainer
from torchdata.stateful_dataloader import StatefulDataLoader

from platoon.openreward.mixture import BalancedEnvironmentSampler
from platoon.openreward.tasks import (
    OPENREWARD_ENVIRONMENT_COLUMN,
    OPENREWARD_MIXTURE_COLUMN,
    OPENREWARD_SAMPLING_WEIGHT_COLUMN,
)


class OpenRewardArealRLTrainer(PlatoonArealRLTrainer):
    """Use exact global submitted-task quotas for mixed training datasets."""

    def _create_dataloader(
        self,
        dataset: Any,
        dataset_config: Any,
        rank: int,
        world_size: int,
    ) -> StatefulDataLoader:
        column_names = set(getattr(dataset, "column_names", []))
        mixture_columns = {
            OPENREWARD_ENVIRONMENT_COLUMN,
            OPENREWARD_MIXTURE_COLUMN,
            OPENREWARD_SAMPLING_WEIGHT_COLUMN,
        }
        if not mixture_columns.issubset(column_names):
            return super()._create_dataloader(
                dataset,
                dataset_config=dataset_config,
                rank=rank,
                world_size=world_size,
            )

        global_batch_size = int(dataset_config.batch_size)
        sampler = BalancedEnvironmentSampler(
            dataset[OPENREWARD_ENVIRONMENT_COLUMN],
            dataset[OPENREWARD_SAMPLING_WEIGHT_COLUMN],
            global_batch_size=global_batch_size,
            num_replicas=world_size,
            rank=rank,
            seed=int(self.config.seed),
            shuffle=bool(dataset_config.shuffle),
            drop_last=bool(dataset_config.drop_last),
        )
        return StatefulDataLoader(
            dataset,
            batch_size=global_batch_size // world_size,
            sampler=sampler,
            drop_last=bool(dataset_config.drop_last),
            num_workers=int(dataset_config.num_workers),
            collate_fn=lambda values: values,
        )
