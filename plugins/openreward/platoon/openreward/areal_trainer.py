"""AReaL trainer specialization for balanced OpenReward mixtures."""

from __future__ import annotations

from typing import Any

from areal.utils import stats_tracker
from platoon.train.areal import PlatoonArealRLTrainer
from torchdata.stateful_dataloader import StatefulDataLoader

from platoon.openreward.mixture import (
    AcceptedEnvironmentBatchObserver,
    BalancedEnvironmentSampler,
    EnvironmentSamplingStartGate,
    StrictEnvironmentBatchCoordinator,
)
from platoon.openreward.tasks import (
    OPENREWARD_ENVIRONMENT_COLUMN,
    OPENREWARD_MIXTURE_COLUMN,
    OPENREWARD_SAMPLING_WEIGHT_COLUMN,
)


class OpenRewardArealRLTrainer(PlatoonArealRLTrainer):
    """Use exact global submitted-task quotas for mixed training datasets."""

    def __init__(
        self,
        config: Any,
        train_dataset: Any,
        val_dataset: Any,
        batch_transforms: list[Any] | None = None,
    ) -> None:
        self._openreward_train_sampler: BalancedEnvironmentSampler | None = None
        self._strict_environment_batches: StrictEnvironmentBatchCoordinator | None = None
        self._accepted_environment_observer: AcceptedEnvironmentBatchObserver | None = None
        self._environment_sampling_start_gate: EnvironmentSamplingStartGate | None = None
        if config.openreward.is_mixture and config.openreward.balance_accepted_batches and config.dynamic_bs:
            raise ValueError("OpenReward balance_accepted_batches=true is incompatible with dynamic_bs=true")

        super().__init__(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_transforms=batch_transforms,
        )

        if config.openreward.is_mixture and config.openreward.balance_accepted_batches:
            sampler = self._openreward_train_sampler
            if sampler is None:
                raise RuntimeError("Strict OpenReward balance expected a mixed training sampler")
            start_step = self.recover_info.last_step_info.next().global_step if self.recover_info is not None else 0
            coordinator = StrictEnvironmentBatchCoordinator(
                sampler.environment_batches(),
                input_environment=self._task_input_environment,
                start_batch_index=start_step,
                max_replacement_rounds=(config.openreward.accepted_batch_max_replacement_rounds),
            )
            coordinator.install(self.rollout.dispatcher)
            self._strict_environment_batches = coordinator
        elif config.openreward.is_mixture:
            sampler = self._openreward_train_sampler
            if sampler is None:
                raise RuntimeError("OpenReward mixture expected a mixed training sampler")
            observer = AcceptedEnvironmentBatchObserver(
                sampler.environment_order,
                input_environment=self._task_input_environment,
            )
            self._accepted_environment_observer = observer
            observer.install(self.rollout.dispatcher)
            environment_start_steps = {
                environment.resolved_label: environment.sampling_start_step
                for environment in config.openreward.resolved_environments()
            }
            if any(start_step > 0 for start_step in environment_start_steps.values()):
                gate = EnvironmentSamplingStartGate(
                    environment_start_steps,
                    input_environment=self._task_input_environment,
                    current_step=self.rollout.get_version,
                )
                # Keep this outermost: the observer should see only inputs
                # admitted by the curriculum, not candidates which are skipped.
                gate.install(self.rollout.dispatcher)
                self._environment_sampling_start_gate = gate

    @staticmethod
    def _task_input_environment(task_input: Any) -> str:
        data = getattr(task_input, "data", None)
        if not isinstance(data, dict):
            raise TypeError("Strict OpenReward task input requires dictionary data")
        environment = data.get(OPENREWARD_ENVIRONMENT_COLUMN)
        if not isinstance(environment, str) or not environment:
            raise ValueError("Strict OpenReward task input is missing its environment label")
        return environment

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

        if (
            dataset_config is self.config.train_dataset
            and world_size != 1
            and any(
                environment.sampling_start_step > 0
                for environment in self.config.openreward.resolved_environments()
            )
        ):
            raise ValueError(
                "Staged OpenReward environment admission currently requires the "
                "single-controller AReaL dataloader (world_size=1)"
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
        # PPOTrainer constructs the validation loader after the training loader.
        # Keep only the training sampler: validation may use a different batch
        # size and must not replace the strict optimizer-step quota schedule.
        if dataset_config is self.config.train_dataset:
            self._openreward_train_sampler = sampler
        return StatefulDataLoader(
            dataset,
            batch_size=global_batch_size // world_size,
            sampler=sampler,
            drop_last=bool(dataset_config.drop_last),
            num_workers=int(dataset_config.num_workers),
            collate_fn=lambda values: values,
        )

    def _postprocess_rollout_batch(
        self,
        rollout_batch: list[dict[str, Any]],
        global_step: int,
        epoch: int,
        epoch_step: int,
    ) -> list[dict[str, Any]] | None:
        coordinator = self._strict_environment_batches
        if coordinator is not None:
            total = sum(coordinator.last_accepted_counts.values())
            metrics: dict[str, float] = {
                "openreward/accepted_batch/strict": 1.0,
                "openreward/accepted_batch/retry_rounds": float(coordinator.last_retry_rounds),
                "openreward/accepted_batch/retry_groups": float(
                    sum(coordinator.last_attempt_counts.values()) - sum(coordinator.last_target_counts.values())
                ),
                "openreward/accepted_batch/input_discards": float(
                    sum(coordinator.last_discarded_input_counts.values())
                ),
            }
            for environment in coordinator.environment_order:
                label = "".join(char if char.isalnum() or char in "._-" else "_" for char in environment)
                accepted = coordinator.last_accepted_counts[environment]
                metrics[f"openreward/accepted_batch/{label}/groups"] = float(accepted)
                metrics[f"openreward/accepted_batch/{label}/input_discards"] = float(
                    coordinator.last_discarded_input_counts[environment]
                )
                metrics[f"openreward/accepted_batch/{label}/fraction"] = float(accepted) / total if total else 0.0
            stats_tracker.scalar(**metrics)
        elif self._accepted_environment_observer is not None:
            observer = self._accepted_environment_observer
            total = observer.last_accepted_total
            metrics = {
                "openreward/accepted_batch/strict": 0.0,
                "openreward/accepted_batch/unknown_groups": float(observer.last_unknown_results),
                "openreward/accepted_batch/unknown_fraction": (
                    float(observer.last_unknown_results) / total if total else 0.0
                ),
            }
            for environment in observer.environment_order:
                label = "".join(char if char.isalnum() or char in "._-" else "_" for char in environment)
                accepted = observer.last_accepted_counts[environment]
                metrics[f"openreward/accepted_batch/{label}/groups"] = float(accepted)
                metrics[f"openreward/accepted_batch/{label}/fraction"] = (
                    float(accepted) / total if total else 0.0
                )
            stats_tracker.scalar(**metrics)
        gate = getattr(self, "_environment_sampling_start_gate", None)
        if gate is not None:
            curriculum_metrics: dict[str, float] = {
                "openreward/curriculum/logical_step": float(gate.last_step),
                "openreward/curriculum/admitted_environments": float(
                    len(gate.last_admitted_environments)
                ),
                "openreward/curriculum/skipped_inputs": float(
                    sum(gate.last_skipped_input_counts.values())
                ),
            }
            for environment in gate.environment_order:
                label = "".join(
                    char if char.isalnum() or char in "._-" else "_"
                    for char in environment
                )
                curriculum_metrics[f"openreward/curriculum/{label}/start_step"] = float(
                    gate.environment_start_steps[environment]
                )
                curriculum_metrics[f"openreward/curriculum/{label}/active"] = float(
                    environment in gate.last_admitted_environments
                )
                curriculum_metrics[f"openreward/curriculum/{label}/admitted_inputs"] = float(
                    gate.last_admitted_input_counts[environment]
                )
                curriculum_metrics[f"openreward/curriculum/{label}/skipped_inputs"] = float(
                    gate.last_skipped_input_counts[environment]
                )
            stats_tracker.scalar(**curriculum_metrics)
        return super()._postprocess_rollout_batch(
            rollout_batch,
            global_step=global_step,
            epoch=epoch,
            epoch_step=epoch_step,
        )
