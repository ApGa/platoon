"""Tinker RL Trainer with stats tracking and WandB logging."""

import asyncio
import logging
import os
import signal
import time
import tinker
from dataclasses import dataclass
from datasets import Dataset
from tinker_cookbook import checkpoint_utils
from platoon.train.tinker.proxy import register_tinker_llm, ModelInfo
from platoon.train.tinker.workflows.base import RolloutWorkflow
from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig, TrainEventTriggerConfig
from platoon.utils.stats_tracker import StatsTracker, get as get_tracker
from platoon.utils.stats_logger import StatsLogger

logger = logging.getLogger(__name__)

# Global flag to track if we've already received a shutdown signal
_SHUTDOWN_REQUESTED = False

class TerminateTrainLoop(Exception):
    """Exception raised to terminate the train loop."""


class TerminateEvalLoop(Exception):
    """Exception raised to terminate the eval loop for a single evaluation run."""


class PlatoonTinkerDataloader:

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle_seed: int | None = 42,
        drop_last: bool = True,
    ):
        if shuffle_seed is not None:
            dataset = dataset.shuffle(seed=shuffle_seed)

        self.dataset = dataset
        self.batched_dataset = self.dataset.batch(
            batch_size=batch_size,
            drop_last_batch=drop_last,
        )
    
    def get_batch(self, batch_num: int) -> list[dict]:
        batch_num = batch_num % len(self.batched_dataset)
        batch = self.batched_dataset[batch_num]
        # Convert dict of lists to list of dicts for training.
        keys = list(batch.keys())
        length = len(next(iter(batch.values()))) if keys else 0
        return [{k: batch[k][i] for k in keys} for i in range(length)]

    def __len__(self) -> int:
        return len(self.batched_dataset)



@dataclass
class TrainLoopSharedState:
    config: PlatoonTinkerRLTrainerConfig
    shutdown_event: asyncio.Event
    train_step: int
    train_dataloader: PlatoonTinkerDataloader
    eval_dataloader: PlatoonTinkerDataloader | None
    optimizer_params: tinker.AdamParams
    service_client: tinker.ServiceClient
    training_client: tinker.TrainingClient
    sampling_client: tinker.SamplingClient
    sampling_client_step: int
    sampling_client_updated_event: asyncio.Event
    model_info: ModelInfo
    stats_logger: StatsLogger
    train_tracker: StatsTracker
    
    def _get_event_frequency(self, config: TrainEventTriggerConfig) -> int:
        """Normalize the event frequency to the number of training steps.
        Args:
            config: The event trigger configuration.
        Returns:
            The event frequency normalized to the number of training steps.
        """
        if config.strategy == 'epoch':
            return self.num_train_batches_per_epoch * config.every
        elif config.strategy == 'step':
            return config.every
        
        raise ValueError(f"Cannot compute event frequency for strategy: {config.strategy}")

    @property
    def eval_every(self) -> int:
        return self._get_event_frequency(self.config.eval)

    @property
    def save_every(self) -> int:
        return self._get_event_frequency(self.config.checkpoint)


    @property
    def num_train_batches_per_epoch(self) -> int:
        return len(self.train_dataloader)
    
    @property
    def num_train_batches(self) -> int:
        return self.num_train_batches_per_epoch * self.config.train.num_epochs


class PlatoonTinkerRLTrainer:
    def __init__(
        self,
        config: PlatoonTinkerRLTrainerConfig,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
    ):
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self._model_info = register_tinker_llm(
            self.config.train.model_name,
            self.config.train.renderer_name
        )
        self._stats_logger: StatsLogger | None = None
        self._train_tracker: StatsTracker | None = None

    @property
    def model_info(self) -> ModelInfo:
        """The model info containing the LLM proxy and renderer."""
        return self._model_info

    @property
    def run_log_path(self) -> str:
        """Get the full log path for this run: log_path/experiment_name/trial_name.
        
        This path is used for all run-specific artifacts including:
        - Checkpoints
        - Rollout results
        - WandB logs
        """
        return os.path.join(
            self.config.log_path,
            self.config.stats.experiment_name,
            self.config.stats.trial_name,
        )

    async def __aenter__(self) -> "PlatoonTinkerRLTrainer":
        """Initialize resources when entering the async context."""
        # Check for resume info to get wandb run ID
        # Use run_log_path which includes experiment_name/trial_name
        resume_info = checkpoint_utils.get_last_checkpoint(self.run_log_path)
        
        # Pass base log_path to StatsLogger - it will add experiment_name/trial_name internally
        stats_logger_config = self.config.stats.to_stats_logger_config(self.config.log_path)
        
        # If resuming, use the saved wandb run ID
        if resume_info and "wandb_run_id" in resume_info:
            stats_logger_config.wandb.resume_run_id = resume_info["wandb_run_id"]
            logger.info(f"Resuming WandB run: {resume_info['wandb_run_id']}")
        
        self._stats_logger = StatsLogger(stats_logger_config, exp_config=self.config)
        self._train_tracker = get_tracker("train")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Clean up resources when exiting the async context."""
        if self._stats_logger is not None:
            self._stats_logger.close()
        # Don't suppress exceptions
        return False

    async def _train_dataloader_loop(
        self, 
        shared_state: TrainLoopSharedState, 
        start_batch: int,
        end_batch: int,
        data_queue: asyncio.Queue[dict],
    ):
        i_batch = start_batch
        max_staleness = self.config.train.max_staleness
        while not shared_state.shutdown_event.is_set() and i_batch < end_batch:
            
            # Control staleness by waiting for training to catch up.
            if max_staleness is not None and i_batch - shared_state.train_step > max_staleness:
                await asyncio.sleep(20.0) 
                continue 

            batch = shared_state.train_dataloader.get_batch(i_batch)
            for data in batch:
                await data_queue.put(data)
            i_batch += 1

    async def _await_with_heartbeat(
        self,
        coro,
        name: str,
        heartbeat_interval: float = 60.0,
    ):
        """Await a coroutine with periodic heartbeat logging.
        
        Useful for long-running async operations where we want visibility
        that we're still waiting.
        """
        start_time = time.perf_counter()
        task = asyncio.ensure_future(coro)
        
        while not task.done():
            try:
                # Wait for either the task to complete or the heartbeat interval
                await asyncio.wait_for(asyncio.shield(task), timeout=heartbeat_interval)
            except asyncio.TimeoutError:
                elapsed = time.perf_counter() - start_time
                logger.info(f"Still waiting for {name}... ({elapsed:.0f}s elapsed)")
        
        return task.result()

    async def _rollout_workflow_worker_loop( 
        self, 
        shared_state: TrainLoopSharedState, 
        workflow: RolloutWorkflow,
        task_data_queue: asyncio.Queue[dict],
        rollout_result_queue: asyncio.Queue[list[tinker.Datum | None]],
        tracker: StatsTracker | None = None,
    ):
        # Use provided tracker or default to train_tracker
        stats_tracker = tracker if tracker is not None else shared_state.train_tracker
        rollout_count = 0
        
        while not shared_state.shutdown_event.is_set():
            data = await task_data_queue.get()
            task_id = data.get('task_id', 'unknown')
            start_time = time.perf_counter()
            rollout_count += 1
            
            try:
                rollout = await workflow.arun_episode(data)
            except Exception as e:
                logger.exception(f"Exception in rollout worker for task {task_id}: {e}")
                stats_tracker.scalar(failed_rollouts=1.0)
                rollout = None
            
            elapsed = time.perf_counter() - start_time
            stats_tracker.scalar(rollout_time=elapsed)
            
            if elapsed > 120.0:
                logger.warning(f"Rollout for task {task_id} took {elapsed:.1f}s (very slow)")
            elif elapsed > 60.0:
                logger.info(f"Rollout for task {task_id} took {elapsed:.1f}s (slow)")
                
            rollout_result_queue.put_nowait(rollout)

    async def _train_loop(
        self, 
        task_group: asyncio.TaskGroup,
        shared_state: TrainLoopSharedState, 
        start_batch: int, 
        end_batch: int,
        train_workflow: RolloutWorkflow,
    ):

        # Initialize queues to pipeline data loading, rollout sampling and training.
        task_data_queue = asyncio.Queue[dict](maxsize=self.config.train.batch_size)
        task_rollout_result_queue = asyncio.Queue[list[tinker.Datum | None]]()

        # Launch background tasks to start streaming in rollouts for training.
        background_streaming_tasks = [
            task_group.create_task(self._train_dataloader_loop(shared_state, start_batch, end_batch, task_data_queue), name="train_data_loader_loop"),
            # Concurrency for task processing (rollout workflow sampling) is controlled by num_concurrent_rollout_workflow_workers.
            *[
                task_group.create_task(
                    self._rollout_workflow_worker_loop(
                        shared_state, train_workflow, task_data_queue, task_rollout_result_queue
                    ), 
                    name=f"train_rollout_workflow_worker_{i}"
                )
                for i in range(self.config.train.num_concurrent_rollout_workflow_workers)
            ],
        ]

        # Start training loop.
        shared_state.train_step = start_batch
        logger.info(f"Starting training from batch {start_batch} to {end_batch}")

        while shared_state.train_step < end_batch:
            batch_start_time = time.perf_counter()
            
            assert self.config.train.batch_size % self.config.train.num_minibatches == 0, (
                f"batch_size {self.config.train.batch_size} must be divisible by num_minibatches {self.config.train.num_minibatches}"
            )
            tasks_per_minibatch = self.config.train.batch_size // self.config.train.num_minibatches
            
            assert tasks_per_minibatch % self.config.train.num_microbatches == 0, (
                f"tasks_per_minibatch {tasks_per_minibatch} must be divisible by num_microbatches {self.config.train.num_microbatches}"
            )

            tasks_per_microbatch = tasks_per_minibatch // self.config.train.num_microbatches

            # Track forward_backward timing
            fwd_bwd_start = time.perf_counter()
            total_datums = 0

            # We perform num_minibatches weight updates per batch.
            for minibatch_num in range(self.config.train.num_minibatches):

                forward_backward_futures: list[tinker.APIFuture[tinker.ForwardBackwardOutput]] = []
                
                # Microbatches are used for gradient accumulation. While gradient accumulation may be less important with tinker, this can help performance.
                # Microbatches are processed as they become available, allowing us to overlap rollout sampling with training even within the same batch.
                # This is a second-level of pipelining orthogonal to the async/off-policy pipelining at the batch level.
                for microbatch_num in range(self.config.train.num_microbatches):
                    
                    task_rollout_results: list[tinker.Datum] = []
                    
                    for task_num in range(tasks_per_microbatch):
                        queue_wait_start = time.perf_counter()
                        logger.debug(f"Waiting for rollout {task_num+1}/{tasks_per_microbatch} (minibatch {minibatch_num}, microbatch {microbatch_num})")
                        rollout = await task_rollout_result_queue.get()
                        queue_wait_elapsed = time.perf_counter() - queue_wait_start
                        if queue_wait_elapsed > 60.0:
                            logger.warning(f"Waited {queue_wait_elapsed:.1f}s for rollout from queue (slow)")

                        # Filter out stale rollouts. TODO: Consider requeuing.
                        if self.config.train.max_staleness is not None \
                            and rollout is not None \
                            and len(rollout) > 0:
                            # Extract checkpoint_version from loss_fn_inputs (stored as a 1-element tensor)
                            checkpoint_version_tensor = rollout[0].loss_fn_inputs.get("checkpoint_version")
                            if checkpoint_version_tensor is not None:
                                rollout_checkpoint_version = int(checkpoint_version_tensor.to_torch().item())
                            else:
                                rollout_checkpoint_version = shared_state.train_step  # Assume current if not present
                            
                            if shared_state.train_step - rollout_checkpoint_version > self.config.train.max_staleness:
                                logger.info(f"Stale rollout detected in batch {shared_state.train_step}. Filtering out.")
                                shared_state.train_tracker.scalar(stale_rollouts=1.0)
                                rollout = None

                        if rollout is not None:
                            task_rollout_results.extend(rollout)
                            total_datums += len(rollout)

                    # Filter out mask and checkpoint_version from loss_fn_inputs before forward_backward
                    # Neither is needed in forward_backward computation.
                    # mask is redundant since advantages are 0 for masked tokens.
                    filtered_datums = [
                        tinker.Datum(
                            model_input=datum.model_input,
                            loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k not in ('mask', 'checkpoint_version')},
                        )
                        for datum in task_rollout_results
                    ]
                    logger.debug(f"Submitting forward_backward_async with {len(filtered_datums)} datums (minibatch {minibatch_num}, microbatch {microbatch_num})")
                    fwd_bwd_submit_start = time.perf_counter()
                    try:
                        forward_backward_futures.append(
                            await shared_state.training_client.forward_backward_async(
                                filtered_datums,
                                loss_fn=self.config.train.loss_fn,
                                loss_fn_config=self.config.train.loss_fn_config,
                            )
                        )
                    except Exception as e:
                        logger.exception(f"forward_backward_async submission failed after {time.perf_counter() - fwd_bwd_submit_start:.1f}s: {e}")
                        raise
                    logger.debug(f"forward_backward_async submitted in {time.perf_counter() - fwd_bwd_submit_start:.1f}s")

                optim_start = time.perf_counter()
                logger.debug("Submitting optim_step_async")
                try:
                    optim_future = await shared_state.training_client.optim_step_async(shared_state.optimizer_params)
                except Exception as e:
                    logger.exception(f"optim_step_async submission failed: {e}")
                    raise
                logger.debug(f"optim_step_async submitted in {time.perf_counter() - optim_start:.1f}s")

                # Consume all forward backward results.
                for microbatch_num, forward_backward_future in enumerate(forward_backward_futures):
                    result_start = time.perf_counter()
                    logger.debug(f"Waiting for forward_backward result (microbatch {microbatch_num})")
                    try:
                        # Use a wrapper to log periodic "still waiting" messages
                        forward_backward_result = await self._await_with_heartbeat(
                            forward_backward_future.result_async(),
                            name=f"forward_backward (microbatch {microbatch_num})",
                            heartbeat_interval=60.0,
                        )
                    except Exception as e:
                        logger.exception(f"forward_backward result_async failed after {time.perf_counter() - result_start:.1f}s: {e}")
                        raise
                    result_elapsed = time.perf_counter() - result_start
                    if result_elapsed > 60.0:
                        logger.warning(f"forward_backward result took {result_elapsed:.1f}s (slow)")
                    else:
                        logger.debug(f"forward_backward result received in {result_elapsed:.1f}s")
                    # TODO: Gather logprob and other metadata from training results for stats logging

                # Wait for optimizer step to complete.
                logger.debug("Waiting for optim_step result")
                optim_result_start = time.perf_counter()
                try:
                    await self._await_with_heartbeat(
                        optim_future.result_async(),
                        name="optim_step",
                        heartbeat_interval=60.0,
                    )
                except Exception as e:
                    logger.exception(f"optim_step result_async failed after {time.perf_counter() - optim_result_start:.1f}s: {e}")
                    raise
                optim_elapsed = time.perf_counter() - optim_result_start
                if optim_elapsed > 60.0:
                    logger.warning(f"optim_step result took {optim_elapsed:.1f}s (slow)")
                else:
                    logger.debug(f"optim_step result received in {optim_elapsed:.1f}s")
            
            fwd_bwd_elapsed = time.perf_counter() - fwd_bwd_start
            shared_state.train_tracker.scalar(fwd_bwd_time=fwd_bwd_elapsed)
            shared_state.train_tracker.scalar(total_datums_per_batch=total_datums)
            
            # Update sampling client with timing
            update_start = time.perf_counter()
            sampling_client = await self._save_checkpoint_and_get_sampling_client(
                training_client=shared_state.training_client,
                i_batch=shared_state.train_step,
                save_every=shared_state.save_every,
                start_batch=start_batch,
            )
            shared_state.model_info.llm.update_sampling_client(sampling_client)
            shared_state.sampling_client = sampling_client
            update_elapsed = time.perf_counter() - update_start
            shared_state.train_tracker.scalar(update_weights_time=update_elapsed)
            
            shared_state.train_step += 1
            shared_state.sampling_client_step += 1
            
            # Signal eval loop that sampling client has been updated
            shared_state.sampling_client_updated_event.set()
            
            # Track total batch time
            batch_elapsed = time.perf_counter() - batch_start_time
            
            # Log stats for this step
            step_stats = shared_state.train_tracker.export(reset=True)
            step_stats["train/step"] = shared_state.train_step
            step_stats["train/epoch"] = shared_state.train_step // shared_state.num_train_batches_per_epoch
            step_stats["progress/done_frac"] = shared_state.train_step / end_batch
            step_stats["optim/lr"] = self.config.train.optimizer.learning_rate
            step_stats["timing/batch_total"] = batch_elapsed
            shared_state.stats_logger.log(step=shared_state.train_step, stats=step_stats)
            
            logger.info(
                f"Step {shared_state.train_step}/{end_batch} | "
                f"Epoch {shared_state.train_step // shared_state.num_train_batches_per_epoch} | "
                f"Batch time: {batch_elapsed:.1f}s | "
                f"Datums: {total_datums}"
            )
        
        await self.shutdown_loops(shared_state)
    
    async def _eval_completion_monitor(
        self,
        total_eval_tasks: int,
        eval_rollout_result_queue: asyncio.Queue[list[tinker.Datum | None]],
    ):
        """Monitor task that waits for all eval results and then terminates the eval TaskGroup."""
        for _ in range(total_eval_tasks):
            await eval_rollout_result_queue.get()
        raise TerminateEvalLoop()

    async def _eval_loop(self, shared_state: TrainLoopSharedState, workflow: RolloutWorkflow):
        eval_tracker = get_tracker("eval")

        while not shared_state.shutdown_event.is_set():

            await shared_state.sampling_client_updated_event.wait()
            
            # Check shutdown again after waking up
            if shared_state.shutdown_event.is_set():
                break
            
            # Clear the event so we wait for the next update
            shared_state.sampling_client_updated_event.clear()

            if shared_state.train_step % shared_state.eval_every == 0 and shared_state.train_step > 0:
                eval_start = time.perf_counter()
                logger.info(f"Starting evaluation at step {shared_state.train_step}")

                eval_data_queue = asyncio.Queue[dict]()
                eval_rollout_result_queue = asyncio.Queue[list[tinker.Datum | None]]()

                # Populate eval queue with all eval data upfront
                total_eval_tasks = 0
                for i_batch in range(len(shared_state.eval_dataloader)):
                    batch = shared_state.eval_dataloader.get_batch(i_batch)
                    for data in batch:
                        eval_data_queue.put_nowait(data)
                        total_eval_tasks += 1
                
                num_workers = self.config.eval.num_concurrent_rollout_workflow_workers
                
                # Run eval with TaskGroup - completion monitor will raise TerminateEvalLoop when done
                try:
                    async with asyncio.TaskGroup() as tg:
                        # Spawn workers with eval tracker
                        for i in range(num_workers):
                            tg.create_task(
                                self._rollout_workflow_worker_loop(
                                    shared_state, 
                                    workflow, 
                                    eval_data_queue, 
                                    eval_rollout_result_queue,
                                    tracker=eval_tracker,
                                ), 
                                name=f"eval_rollout_workflow_worker_{i}",
                            )
                        # Spawn completion monitor
                        tg.create_task(
                            self._eval_completion_monitor(total_eval_tasks, eval_rollout_result_queue),
                            name="eval_completion_monitor",
                        )
                except* TerminateEvalLoop:
                    pass  # Expected - eval completed successfully
                
                eval_elapsed = time.perf_counter() - eval_start
                logger.info(f"Evaluation completed in {eval_elapsed:.1f}s for {total_eval_tasks} tasks")
                
                # Export and log eval stats
                eval_stats = eval_tracker.export(reset=True)
                eval_stats["eval/time"] = eval_elapsed
                eval_stats["eval/num_tasks"] = total_eval_tasks
                # Use force=True since train already logged at this step
                shared_state.stats_logger.log(step=shared_state.train_step, stats=eval_stats, force=True)


    async def shutdown_loops(self, shared_state: TrainLoopSharedState):
        shared_state.shutdown_event.set()
        shared_state.sampling_client_updated_event.set()
        raise TerminateTrainLoop()

    async def _save_checkpoint_and_get_sampling_client(
        self,
        training_client: tinker.TrainingClient,
        i_batch: int,
        save_every: int,
        start_batch: int = 0,
    ) -> tinker.SamplingClient:
        """Save checkpoint with wandb_run_id and return new sampling client.
        
        Unlike tinker_cookbook's version, this includes the wandb_run_id in the
        checkpoint loop_state to enable proper WandB run resumption.
        """
        if i_batch > start_batch and i_batch % save_every == 0:
            # Save a full checkpoint with loop state
            path_dict = await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"{i_batch:06d}",
                log_path=self.run_log_path,
                loop_state={
                    "batch": i_batch,
                    "wandb_run_id": self._stats_logger.wandb_run_id,
                },
                kind="both",
            )
            return training_client.create_sampling_client(path_dict["sampler_path"])
        else:
            return await training_client.save_weights_and_get_sampling_client_async()

    async def _create_training_client(self, service_client: tinker.ServiceClient, resume_info: dict | None) -> tinker.TrainingClient:
        if resume_info:
            # Resuming interrupted training - load optimizer state for proper continuation
            training_client = (
                await service_client.create_training_client_from_state_with_optimizer_async(
                    resume_info["state_path"]
                )
            )
            logger.info(f"Resumed training from {resume_info['state_path']}")
        elif self.config.checkpoint.load_checkpoint_path:
            # Starting fresh from a checkpoint - load weights only (fresh optimizer)
            training_client = await service_client.create_training_client_from_state_async(
                self.config.checkpoint.load_checkpoint_path
            )
            logger.info(f"Loaded weights from {self.config.checkpoint.load_checkpoint_path}")
        else:
            training_client = await service_client.create_lora_training_client_async(
                self.config.train.model_name, rank=self.config.train.lora_rank
            )
            logger.info(f"Created new training client for {self.config.train.model_name} with rank {self.config.train.lora_rank}")

        return training_client

    def _setup_signal_handlers(self, shutdown_event: asyncio.Event) -> None:
        """Set up signal handlers for graceful shutdown.
        
        First Ctrl+C: Sets shutdown_event, allowing graceful termination.
        Second Ctrl+C: Forces immediate exit.
        """
        global _SHUTDOWN_REQUESTED
        _SHUTDOWN_REQUESTED = False
        
        def signal_handler(signum, frame):
            global _SHUTDOWN_REQUESTED
            sig_name = signal.Signals(signum).name
            
            if _SHUTDOWN_REQUESTED:
                # Second signal - force exit
                logger.warning(f"Received {sig_name} again. Forcing immediate exit.")
                # Force cleanup of wandb
                try:
                    import wandb
                    wandb.finish(quiet=True)
                except Exception:
                    pass
                os._exit(1)
            else:
                # First signal - request graceful shutdown
                _SHUTDOWN_REQUESTED = True
                logger.info(f"Received {sig_name}. Requesting graceful shutdown... (press Ctrl+C again to force exit)")
                shutdown_event.set()
        
        # Register handlers for SIGINT (Ctrl+C) and SIGTERM
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def train(
        self, 
        train_workflow: RolloutWorkflow, 
        eval_workflow: RolloutWorkflow | None = None
    ):
        if self._stats_logger is None or self._train_tracker is None:
            raise RuntimeError(
                "Trainer must be used as an async context manager. "
                "Use 'async with trainer:' before calling train()."
            )
        
        shutdown_event = asyncio.Event()
        
        # Set up signal handlers for graceful Ctrl+C shutdown
        self._setup_signal_handlers(shutdown_event)

        service_client = tinker.ServiceClient(base_url=self.config.tinker_base_url)

        resume_info = checkpoint_utils.get_last_checkpoint(self.run_log_path)
        start_batch = resume_info.get("batch", 0) if resume_info else 0

        training_client = await self._create_training_client(service_client, resume_info)

        # Initial sampling client to use
        path_dict = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name=f"{start_batch:06d}",
            log_path=self.run_log_path,
            loop_state={
                "batch": start_batch,
                "wandb_run_id": self._stats_logger.wandb_run_id,
            },
            kind="both",
        )
        sampling_client = training_client.create_sampling_client(path_dict["sampler_path"])
        sampling_client_step = start_batch
        sampling_client_updated_event = asyncio.Event()
        sampling_client_updated_event.set()

        # Set initial version and update sampling client (without auto-incrementing)
        self.model_info.llm.set_version(start_batch)
        self.model_info.llm.update_sampling_client(sampling_client, increment_version=False)

        optimizer_params = tinker.AdamParams(
            learning_rate=self.config.train.optimizer.learning_rate,
            beta1=self.config.train.optimizer.beta1,
            beta2=self.config.train.optimizer.beta2,
            eps=self.config.train.optimizer.eps,
        )

        train_dataloader = PlatoonTinkerDataloader(self.train_dataset, self.config.train.batch_size)
        eval_dataloader = None
        if self.eval_dataset is not None:
            eval_dataloader = PlatoonTinkerDataloader(self.eval_dataset, batch_size=1, shuffle_seed=None, drop_last=False)

        shared_state = TrainLoopSharedState(
            config=self.config,
            shutdown_event=shutdown_event,
            train_step=start_batch,
            optimizer_params=optimizer_params,
            service_client=service_client,
            training_client=training_client,
            sampling_client=sampling_client,
            sampling_client_step=sampling_client_step,
            sampling_client_updated_event=sampling_client_updated_event,
            model_info=self._model_info,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            stats_logger=self._stats_logger,
            train_tracker=self._train_tracker,
        )

        end_batch = shared_state.num_train_batches
  

        interrupted = False
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._train_loop(tg, shared_state, start_batch, end_batch, train_workflow), name="train_loop")
                if self.eval_dataset is not None and self.config.eval.every > 0:
                    tg.create_task(self._eval_loop(shared_state, eval_workflow), name="eval_loop")
        except* TerminateTrainLoop:
            pass
        except* (KeyboardInterrupt, asyncio.CancelledError):
            interrupted = True
            logger.info("Training interrupted. Saving checkpoint before exit...")

        # Save checkpoint
        current_step = shared_state.train_step
        if interrupted:
            # Save interrupt checkpoint so we can resume
            if current_step > start_batch:
                logger.info(f"Saving interrupt checkpoint at step {current_step}...")
                await checkpoint_utils.save_checkpoint_async(
                    training_client=shared_state.training_client,
                    name=f"{current_step:06d}_interrupted",
                    log_path=self.run_log_path,
                    kind="both",
                    loop_state={
                        "batch": current_step,
                        "wandb_run_id": self._stats_logger.wandb_run_id,
                    },
                )
                logger.info(f"Checkpoint saved. Resume training to continue from step {current_step}.")
        elif start_batch < end_batch:
            # Training completed normally - save final checkpoint
            await checkpoint_utils.save_checkpoint_async(
                training_client=shared_state.training_client,
                name="final",
                log_path=self.run_log_path,
                kind="both",
                loop_state={
                    "batch": end_batch,
                    "wandb_run_id": self._stats_logger.wandb_run_id,
                },
            )
            logger.info("Training complete successfully!")
        else:
            logger.info("Training was already complete; nothing to do")
