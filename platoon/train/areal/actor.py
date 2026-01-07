"""Custom PPO Actor with configurable loss functions for Platoon AReaL backend.

This module provides a wrapper around AReaL's FSDPPPOActor that supports
configurable loss functions (GRPO, CISPO, SAPO, etc.) while maintaining
full compatibility with the AReaL training infrastructure.
"""

import functools
from typing import Any

import torch

from areal.api.cli_args import MicroBatchSpec
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.ppo.actor import PPOActorConfig
from areal.utils import stats_tracker
from areal.utils.data import split_padded_tensor_dict_into_mb_list
from areal.utils.perf_tracer import trace_perf

from platoon.train.areal.config_defs import LossFnConfig
from platoon.train.areal.loss_functions import (
    cispo_loss_fn,
    grpo_loss_fn,
    get_loss_fn,
)


def create_loss_fn_for_areal(
    loss_fn_name: str,
    loss_fn_config: LossFnConfig,
    actor_config: PPOActorConfig,
    current_version: int | None = None,
) -> callable:
    """Create a loss function compatible with AReaL's train_batch interface.
    
    AReaL's train_batch expects a loss function with signature:
        loss_fn(logits, input_data) -> (loss, stat)
    
    This function creates a partial that binds all configuration parameters.
    
    Args:
        loss_fn_name: Name of the loss function ("grpo", "cispo", "sapo")
        loss_fn_config: LossFnConfig with loss function parameters
        actor_config: PPOActorConfig with actor-level parameters
        current_version: Current training version for off-policy handling
    
    Returns:
        A callable compatible with AReaL's train_batch loss_fn interface
    """
    temperature = getattr(actor_config, 'temperature', 1.0)
    
    if loss_fn_name == "cispo":
        return functools.partial(
            cispo_loss_fn,
            temperature=temperature,
            clip_low_threshold=loss_fn_config.clip_low_threshold,
            clip_high_threshold=loss_fn_config.clip_high_threshold,
            importance_sampling_level=getattr(actor_config, 'importance_sampling_level', 'token'),
        )
    elif loss_fn_name in ("grpo", "ppo"):
        return functools.partial(
            grpo_loss_fn,
            temperature=temperature,
            eps_clip=actor_config.eps_clip,
            eps_clip_higher=getattr(actor_config, 'eps_clip_higher', None),
            c_clip=getattr(actor_config, 'c_clip', None),
            behav_imp_weight_cap=getattr(actor_config, 'behav_imp_weight_cap', None),
            m2_threshold=getattr(actor_config, 'm2_threshold', None),
            importance_sampling_level=getattr(actor_config, 'importance_sampling_level', 'token'),
        )
    else:
        raise ValueError(f"Unknown loss function: {loss_fn_name}")


class PlatoonPPOActor(FSDPPPOActor):
    """Extended PPO Actor with configurable loss functions.
    
    This class extends FSDPPPOActor to support multiple loss functions
    (GRPO, CISPO, SAPO) while maintaining full compatibility with the
    AReaL training infrastructure.
    
    Example usage:
        actor = PlatoonPPOActor(
            config=actor_config,
            loss_fn_config=LossFnConfig(loss_fn="cispo", clip_high_threshold=5.0)
        )
        # Train loop
        actor.ppo_update(batch)  # Uses CISPO loss
    """
    
    def __init__(
        self,
        config: PPOActorConfig,
        loss_fn_config: LossFnConfig | None = None,
    ):
        """Initialize the actor.
        
        Args:
            config: PPOActorConfig for the underlying PPO actor
            loss_fn_config: Configuration for the loss function. If None,
                defaults to GRPO loss.
        """
        super().__init__(config)
        self.loss_fn_config = loss_fn_config or LossFnConfig()
        self._loss_fn_name = self.loss_fn_config.loss_fn
        
    @trace_perf("platoon_ppo_actor.ppo_update", category="compute")
    @stats_tracker.scope_func_wrapper("ppo_actor")
    def ppo_update(self, data: dict[str, Any]) -> None:
        """Run PPO update step with configurable loss function.
        
        This method overrides the parent to use the configured loss function
        instead of the hardcoded GRPO loss.
        """
        attn_mask = data["attention_mask"]
        loss_mask = data["loss_mask"]
        reward_score = data["rewards"]
        seqlens = attn_mask.sum(-1)
        
        # ===== Logging code (same as parent) =====
        result_denominators = {
            "correct_n_seqs": (reward_score > 0).bool(),
            "incorrect_n_seqs": (reward_score <= 0).bool(),
        }
        if self.config.log_agent_stats:
            if "begin_of_trajectory" not in data:
                raise RuntimeError(
                    "'begin_of_trajectory' is expected to log agent statistics"
                )
            if len(self.config.log_agent_stats_keys) == 0:
                raise RuntimeError(
                    "`log_agent_stats_keys` should not be empty when log_agent_stats=True"
                )
            agent_denominator = (data["begin_of_trajectory"] > 0).bool()
            result_denominators["agent"] = agent_denominator
            
        global_denominators = dict(
            n_seqs=torch.ones_like(reward_score, dtype=torch.bool),
            n_tokens=torch.ones_like(loss_mask, dtype=torch.bool),
            n_valid_tokens=loss_mask.bool(),
            **result_denominators,
        )
        stats_tracker.denominator(**global_denominators)
        stats_tracker.stat(
            correct_seq_len=seqlens.float(), denominator="correct_n_seqs"
        )
        stats_tracker.stat(
            incorrect_seq_len=seqlens.float(), denominator="incorrect_n_seqs"
        )

        stats = dict(
            advantages=data["advantages"],
            kl_rewards=data["kl_rewards"],
            final_reward=data["tot_rewards"],
        )
        stats_tracker.stat(**stats, denominator="n_valid_tokens")

        prompt_lens = data["attention_mask"].sum(-1) - data["loss_mask"].sum(-1)
        seq_stats = dict(
            no_eos_ratios=(seqlens == attn_mask.shape[-1]).float(),
            task_reward=reward_score.float(),
            prompt_len=prompt_lens.float(),
            seq_len=seqlens.float(),
        )
        stats_tracker.stat(**seq_stats, denominator="n_seqs")
        
        scalars = dict(
            mask_no_eos_with_zero=self.config.mask_no_eos_with_zero,
            eps_clip=self.config.eps_clip,
            # Note: loss_fn is a string, can't be logged as scalar
        )
        if self.config.c_clip is not None:
            scalars["c_clip"] = self.config.c_clip
            scalars["use_dual_clip"] = 1
        else:
            scalars["use_dual_clip"] = 0
        if self.config.behav_imp_weight_cap is not None:
            scalars["behav_imp_weight_cap"] = self.config.behav_imp_weight_cap
        stats_tracker.scalar(**scalars)

        if self.config.log_agent_stats:
            stats_tracker.stat(
                **{k: data[k].float() for k in self.config.log_agent_stats_keys},
                denominator="agent",
            )
        # ===== End logging code =====

        # Pop keys that are no longer needed (matching areal's ppo_update)
        for key in ["rewards", "tot_rewards", "kl_rewards", "versions"]:
            data.pop(key, None)
            
        # Enable gradient checkpointing
        # Access the engine via self (we are the engine)
        self.train()
        
        mb_inputs = split_padded_tensor_dict_into_mb_list(
            data,
            mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
        )

        with stats_tracker.scope("update"):
            current_version = self.get_version()
            
            # Create loss function with configuration
            loss_fn = create_loss_fn_for_areal(
                loss_fn_name=self._loss_fn_name,
                loss_fn_config=self.loss_fn_config,
                actor_config=self.config,
                current_version=current_version,
            )

            for mb in mb_inputs.mbs:
                train_stat = self.train_batch(
                    mb,
                    loss_fn=loss_fn,
                    loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
                )
                stats_tracker.scalar(**train_stat)


def create_actor(
    config: PPOActorConfig,
    loss_fn_config: LossFnConfig | None = None,
) -> FSDPPPOActor:
    """Factory function to create the appropriate actor based on loss function config.
    
    Args:
        config: PPOActorConfig for the actor
        loss_fn_config: Optional LossFnConfig. If None or loss_fn is "grpo"/"ppo",
            returns the standard FSDPPPOActor for maximum compatibility.
            Otherwise returns PlatoonPPOActor with custom loss function.
    
    Returns:
        An actor instance (either FSDPPPOActor or PlatoonPPOActor)
    """
    if loss_fn_config is None or loss_fn_config.loss_fn in ("grpo", "ppo"):
        # Use standard AReaL actor for GRPO/PPO
        return FSDPPPOActor(config)
    else:
        # Use our custom actor for other loss functions
        return PlatoonPPOActor(config, loss_fn_config)

