from dataclasses import dataclass, field
from datasets import Dataset
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.api.alloc_mode import AllocationMode
from areal.utils import seeding, stats_tracker  
from areal.api.workflow_api import RolloutWorkflow
from platoon.utils.train import VariableBatchInferenceEngineConfig, set_expandable_segments, post_process_and_redistribute_tensor_container
from platoon.train.aent.aent_args import AEntPPOActorConfig
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.api.cli_args import GRPOConfig
from areal.platforms import current_platform
from areal.utils.network import find_free_ports
from areal.experimental.openai.proxy import ProxyServer
from areal.utils import logging
from areal.experimental.openai.client import ArealOpenAI
import os
import datetime
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
import torch
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.ppo.actor import PPOActorConfig
from platoon.train.aent.actor import FSDPAEntPPOActor
from copy import deepcopy

logger = logging.getLogger("Platoon RL Trainer")


# TODO: We should make this customizable with a factory.
@dataclass
class RolloutConfig:
    model_name: str | None = None
    model_endpoint: str | None = None
    model_api_key: str | None = None
    train: bool = False
    max_steps: int | None = None
    output_dir: str = 'rollout_results'
    verbose: bool = True
    timeout: int | None = None
    return_dict: bool = False
    
@dataclass
class WorkflowConfig:
    rollout_config: RolloutConfig = field(default_factory=RolloutConfig)

@dataclass
class PlatoonStepWiseRLTrainerConfig(GRPOConfig):
    workflow_config: WorkflowConfig = field(default_factory=WorkflowConfig)
    rollout: VariableBatchInferenceEngineConfig = field(default_factory=VariableBatchInferenceEngineConfig)
    #actor: AEntPPOActorConfig = field(default_factory=AEntPPOActorConfig)
    actor: PPOActorConfig = field(default_factory=PPOActorConfig)


# TODO: Add support for megatron training backend and VLLM inference backend.
class PlatoonStepWiseRLTrainer:
    
    def __init__(
        self,
        config: PlatoonStepWiseRLTrainerConfig,
        train_dataset: Dataset,
        val_dataset: Dataset,
        tokenizer: PreTrainedTokenizerFast | None = None
    ):
        set_expandable_segments(True)
        
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        
        rank = int(os.environ.get("RANK", 0))
        
        if self.tokenizer is None:
            self.tokenizer = load_hf_tokenizer(config.tokenizer_path)
            
        if self.tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
            config.gconfig.stop_token_ids.append(self.tokenizer.pad_token_id)
        if self.tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
            config.gconfig.stop_token_ids.append(self.tokenizer.eos_token_id)
        
        seeding.set_random_seed(config.seed, key=f"trainer{rank}")
        allocation_mode = AllocationMode.from_str(config.allocation_mode)
        parallel_strategy = allocation_mode.train
        assert parallel_strategy is not None
        
        
        #self.actor = FSDPAEntPPOActor(config=config.actor)
        self.actor = FSDPPPOActor(config=config.actor)
        self.actor.create_process_group(parallel_strategy=parallel_strategy)
        
        
        # TODO: Use cleaner create_dataloader function.
        self.train_dataloader = StatefulDataLoader(
            train_dataset,
            batch_size=config.train_dataset.batch_size // self.actor.data_parallel_world_size,
            sampler=DistributedSampler(
                train_dataset, 
                num_replicas=self.actor.data_parallel_world_size, 
                rank=self.actor.data_parallel_rank,
                shuffle=config.train_dataset.shuffle,
                drop_last=config.train_dataset.drop_last,
            ),
            num_workers=config.train_dataset.num_workers,
            collate_fn=lambda x: x,
        )
        
        self.val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=config.valid_dataset.batch_size // self.actor.data_parallel_world_size,
            sampler=DistributedSampler(
                val_dataset, 
                num_replicas=self.actor.data_parallel_world_size, 
                rank=self.actor.data_parallel_rank,
                shuffle=config.valid_dataset.shuffle,
                drop_last=config.valid_dataset.drop_last,
            ),
            num_workers=config.valid_dataset.num_workers,
            collate_fn=lambda x: x,
        )
        
        self.ft_spec = FinetuneSpec(
            total_train_epochs=config.total_train_epochs,
            dataset_size=len(self.train_dataloader) * config.train_dataset.batch_size,
            train_batch_size=config.train_dataset.batch_size,
        )
        
        self.rollout = RemoteSGLangEngine(config.rollout)
        self.rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)
        self.eval_rollout = RemoteSGLangEngine(deepcopy(config.rollout))
        # NOTE: eval does not have any offpolicyness control
        self.eval_rollout.config.max_head_offpolicyness = int(1e12)
        self.eval_rollout.initialize()
        
        self.weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(allocation_mode)

        self.actor.initialize(None, self.ft_spec)
        self.actor.connect_engine(self.rollout, self.weight_update_meta)
    
        
        self.ref = None
        if config.actor.kl_ctl > 0 and config.ref is not None:
            self.ref = FSDPPPOActor(config=config.ref)
            self.ref.create_process_group(parallel_strategy=parallel_strategy)
            self.ref.initialize(None, self.ft_spec)
        
        
        self.llm_client = ArealOpenAI(engine=self.rollout, tokenizer=self.tokenizer)
        free_port = find_free_ports(1)[0]
        self.proxy_server = ProxyServer(free_port, client=self.llm_client)
        self.proxy_server.start(wait_until_ready=True)
        
        all_addresses = [None for _ in range(self.actor.data_parallel_world_size)]
        dist.all_gather_object(
            all_addresses, self.proxy_server.public_addr, group=self.actor.data_parallel_group
        )
        logger.info(f"Found {len(all_addresses)} proxy servers: {all_addresses}")
        #dist.barrier(group=self.actor.cpu_group)
        dist.barrier(device_ids=[self.actor.device.index])
        
        
        self.saver = Saver(config.saver, self.ft_spec)
        self.stats_logger = StatsLogger(config, self.ft_spec)
        self.evaluator = Evaluator(config.evaluator, self.ft_spec)

        self.recover_handler = RecoverHandler(config.recover, self.ft_spec)
        self.recover_info = self.recover_handler.load(
            self.actor,
            self.saver,
            self.evaluator,
            self.stats_logger,
            self.train_dataloader,
            inference_engine=self.rollout,
            weight_update_meta=self.weight_update_meta,
        )
        
        # TODO: should do this for other groups as well? Make configurable
        #dist.distributed_c10d._set_pg_timeout(datetime.timedelta(seconds=7200), self.actor.data_parallel_group)
            
    def train(
        self,
        workflow: RolloutWorkflow,
        eval_workflow: RolloutWorkflow
    ):
        config = self.config
        start_step = (
            self.recover_info.last_step_info.next().global_step
            if self.recover_info is not None
            else 0
        )
        
        total_epochs = config.total_train_epochs
        steps_per_epoch = len(self.train_dataloader)
        max_steps = total_epochs * steps_per_epoch
        
        for global_step in range(start_step, max_steps):
            epoch = global_step // steps_per_epoch
            step = global_step % steps_per_epoch
            step_info = StepInfo(
                global_step=global_step,
                epoch=epoch,
                epoch_step=step,
                steps_per_epoch=steps_per_epoch,
            )
            
            with stats_tracker.record_timing("rollout"):
                batch = self.actor.prepare_batch(
                    self.train_dataloader,
                    workflow=workflow,
                    should_accept_fn=lambda sample: True,
                )

                # NOTE: Temporary adv calculation experiment
                #batch['rewards'] = batch['rewards'] - torch.mean(batch['task_reward'])

 
            if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
                with stats_tracker.record_timing("recompute_logp"):
                    logp = self.actor.compute_logp(batch)
                    batch["prox_logp"] = logp
                    log_gpu_stats("recompute logp")
                    
                # current_platform.synchronize()
                # dist.barrier(group=self.actor.cpu_group)
                
                dist.barrier(device_ids=[self.actor.device.index])
                current_platform.synchronize()
                
            
            if self.ref is not None:
                with stats_tracker.record_timing("ref_logp"):
                    ref_logp = self.ref.compute_logp(batch)
                    batch["ref_logp"] = ref_logp
                    log_gpu_stats("ref logp")

                # current_platform.synchronize()
                # dist.barrier(group=self.actor.cpu_group)
                
                dist.barrier(device_ids=[self.actor.device.index])
                current_platform.synchronize()
            
            with stats_tracker.record_timing("compute_advantage"):
                self.actor.compute_advantages(batch)
                log_gpu_stats("compute advantages")
                
            # current_platform.synchronize()
            # dist.barrier(group=self.actor.cpu_group)

            torch.cuda.empty_cache()
            dist.barrier(device_ids=[self.actor.device.index])
            current_platform.synchronize()

            
            with (stats_tracker.record_timing("train_step")):
                #stats = self.actor.aent_ppo_update(batch, global_step)
                stats = self.actor.ppo_update(batch)
                self.actor.step_lr_scheduler()
                log_gpu_stats("ppo update")
                
            # current_platform.synchronize()
            # dist.barrier(group=self.actor.cpu_group)
            
            torch.cuda.empty_cache()
            dist.barrier(device_ids=[self.actor.device.index])
            current_platform.synchronize()

            # pause inference for updating weights, save, and evaluation
            self.rollout.pause()

            with stats_tracker.record_timing("update_weights"):
                self.actor.update_weights(self.weight_update_meta)
                self.actor.set_version(global_step + 1)
                self.rollout.set_version(global_step + 1)
                self.eval_rollout.set_version(global_step + 1)

            with stats_tracker.record_timing("save"):
                self.saver.save(self.actor, epoch, step, global_step, tokenizer=self.tokenizer)

            with stats_tracker.record_timing("checkpoint_for_recover"):
                self.recover_handler.dump(
                    self.actor,
                    step_info,
                    self.saver,
                    self.evaluator,
                    self.stats_logger,
                    self.train_dataloader,
                    tokenizer=self.tokenizer,
                )

            torch.cuda.empty_cache()
            dist.barrier(device_ids=[self.actor.device.index])
            current_platform.synchronize()
            
            # current_platform.synchronize()
            # dist.barrier(group=self.actor.cpu_group)

            with stats_tracker.record_timing("eval"):

                def evaluate_fn():
                    if self.actor.is_data_parallel_head():
                        print(f"Evaluating...")
                        cnt = 0
                        for data in self.val_dataloader:
                            for item in data:
                                self.eval_rollout.submit(item, eval_workflow)
                                cnt += 1
                        try:
                            results = self.eval_rollout.wait(cnt, timeout=1800) # TODO: Make the eval timeout configurable.
                            print(f"Evaluated {cnt} tasks")
                            print(f"Task Rewards: {results['task_reward']}")
                            
                        except TimeoutError as e:
                            print(f"Evaluation timed out after 1800 seconds: {e}")
                        
                    # current_platform.synchronize() 
                    # dist.barrier(group=self.actor.cpu_group)
                    
                    torch.cuda.empty_cache()
                    dist.barrier(device_ids=[self.actor.device.index])
                    current_platform.synchronize()

                self.evaluator.evaluate(
                    evaluate_fn,
                    epoch,
                    step,
                    global_step,
                )

            # current_platform.synchronize()
            # dist.barrier(group=self.actor.cpu_group)
            
            dist.barrier(device_ids=[self.actor.device.index])
            current_platform.synchronize()
            
            # Upload statistics to the logger (e.g., wandb)
            stats = stats_tracker.export_all(reduce_group=self.actor.data_parallel_group)
            self.stats_logger.commit(epoch, step, global_step, stats)

            # current_platform.synchronize()
            # dist.barrier(group=self.actor.cpu_group)
            
            torch.cuda.empty_cache()
            dist.barrier(device_ids=[self.actor.device.index])
            current_platform.synchronize()

            # Resume rollout
            self.rollout.resume()
    
    def close(self):
        self.stats_logger.close()
        self.eval_rollout.destroy()
        self.rollout.destroy()
        if self.ref is not None:
            self.ref.destroy()
        self.actor.destroy()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        if exc_type is not None:
            raise exc_value
