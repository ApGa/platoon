"""Custom PPO actor support for Platoon's AReaL backend."""

import gc
from collections.abc import Callable
from typing import Any

import torch
from areal.api import Scheduler
from areal.api.cli_args import MicroBatchSpec
from areal.engine import FSDPPPOActor
from areal.trainer.ppo.actor import PPOActor, PPOActorController
from areal.trainer.ppo.stats import infer_token_denominator
from areal.utils import stats_tracker
from areal.utils.data import (
    broadcast_tensor_container,
    concat_batch,
    split_padded_tensor_dict_into_mb_list,
    tensor_container_to,
)
from areal.utils.perf_tracer import trace_perf

from platoon.train.areal.config_defs import PlatoonPPOActorConfig
from platoon.train.areal.loss_functions import build_loss_fn


class PlatoonActorImpl(PPOActor):
    """PPO actor implementation with registry-driven Platoon losses."""

    def __init__(self, config: PlatoonPPOActorConfig, engine: Any):
        super().__init__(config, engine)
        self.config = config

    def _make_loss_fn(self, current_version: int | None) -> Callable[..., torch.Tensor]:
        common_kwargs = dict(
            importance_sampling_level=self.config.importance_sampling_level,
            eps_clip=self.config.eps_clip,
            eps_clip_higher=self.config.eps_clip_higher,
            c_clip=self.config.c_clip,
            behave_imp_weight_cap=self.config.behave_imp_weight_cap,
            m2_threshold=self.m2_threshold,
            current_version=current_version,
            prox_logp_method=self.config.prox_logp_method,
            use_sapo_loss=self.config.use_sapo_loss,
            sapo_tau_pos=self.config.sapo_tau_pos,
            sapo_tau_neg=self.config.sapo_tau_neg,
            use_decoupled_loss=self.config.use_decoupled_loss,
            behave_imp_weight_mode=self.config.behave_imp_weight_mode,
        )
        return build_loss_fn(
            self.config.loss_fn,
            loss_fn_kwargs=self.config.loss_fn_kwargs,
            common_kwargs=common_kwargs,
        )

    @trace_perf("platoon_ppo_actor.ppo_update", category="compute")
    @stats_tracker.scope_func_wrapper("ppo_actor")
    def ppo_update(self, data: list[dict[str, Any]]) -> None:
        return super().ppo_update(data)

    @trace_perf("platoon_ppo_actor.ppo_update_batched", category="compute")
    @stats_tracker.scope_func_wrapper("ppo_actor")
    def ppo_update_batched(self, data: list[dict[str, Any]] | dict[str, Any] | None = None) -> None:
        """Concat on the DP head before GPU broadcast to reduce peak memory."""

        batch = None
        if isinstance(data, list):
            batch, _ = concat_batch(data)
        elif data is not None:
            batch = data

        if batch is not None:
            batch = tensor_container_to(batch, self.engine.device)
        batch = broadcast_tensor_container(
            batch,
            src_rank=self.engine.current_data_parallel_head(),
            group=self.engine.context_and_model_parallel_group,
        )
        if batch is None:
            return
        self._ppo_update(batch)

    def _ppo_update(self, data: dict[str, Any]) -> None:
        attn_mask = data["attention_mask"]
        loss_mask = data["loss_mask"]
        reward_score = data["rewards"]
        seqlens = attn_mask.sum(-1)

        result_denominators = {
            "correct_n_seqs": (reward_score > 0).bool(),
            "incorrect_n_seqs": (reward_score <= 0).bool(),
        }
        if self.config.log_agent_stats:
            if "begin_of_trajectory" not in data:
                raise RuntimeError("'begin_of_trajectory' is expected to log agent statistics")
            if len(self.config.log_agent_stats_keys) == 0:
                raise RuntimeError("`log_agent_stats_keys` should not be empty when log_agent_stats=True")
            agent_denominator = (data["begin_of_trajectory"] > 0).bool()
            result_denominators["agent"] = agent_denominator

        global_denominators = dict(
            n_seqs=torch.ones_like(reward_score, dtype=torch.bool),
            n_tokens=infer_token_denominator(data, loss_mask),
            n_valid_tokens=loss_mask.bool(),
            **result_denominators,
        )
        stats_tracker.denominator(**global_denominators)
        stats_tracker.stat(correct_seq_len=seqlens.float(), denominator="correct_n_seqs")
        stats_tracker.stat(incorrect_seq_len=seqlens.float(), denominator="incorrect_n_seqs")

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
        )
        if self.config.c_clip is not None:
            scalars["c_clip"] = self.config.c_clip
            scalars["use_dual_clip"] = 1
        else:
            scalars["use_dual_clip"] = 0
        if self.config.behave_imp_weight_cap is not None:
            scalars["behave_imp_weight_cap"] = self.config.behave_imp_weight_cap
        stats_tracker.scalar(**scalars)

        if self.config.log_agent_stats:
            stats_tracker.stat(
                **{k: data[k].float() for k in self.config.log_agent_stats_keys},
                denominator="agent",
            )

        for key in ["rewards", "tot_rewards", "kl_rewards"]:
            data.pop(key, None)

        self.engine.train()
        mb_inputs = split_padded_tensor_dict_into_mb_list(
            data,
            mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
        )

        with stats_tracker.scope("update"):
            current_version = self.engine.get_version()
            loss_fn = self._make_loss_fn(current_version)
            for mb in mb_inputs.mbs:
                train_stat = self.engine.train_batch(
                    mb,
                    loss_fn=loss_fn,
                    loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
                )
                stats_tracker.scalar(**train_stat)


class PlatoonPPOActorController(PPOActorController):
    """Controller extensions for Platoon's actor worker methods."""

    def ppo_update(self, *args, **kwargs) -> None:
        self._custom_function_call(
            "ppo_update_batched",
            *args,
            rpc_meta={"broadcast": False},
            **kwargs,
        )

    def clear_cuda_cache(self) -> None:
        self._custom_function_call("clear_cuda_cache")


class PlatoonPPOActor(FSDPPPOActor):
    """FSDP PPO actor with registry-driven Platoon loss selection."""

    def __init__(self, config: PlatoonPPOActorConfig):
        super().__init__(config)
        self.actor = PlatoonActorImpl(config, self)

    def ppo_update_batched(self, *args, **kwargs) -> None:
        self.actor.ppo_update_batched(*args, **kwargs)

    def clear_cuda_cache(self) -> None:
        gc.collect()
        if torch.cuda.is_available() and getattr(self, "device", None) is not None:
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

    @classmethod
    def as_controller(cls, config: PlatoonPPOActorConfig, scheduler: Scheduler):
        return PlatoonPPOActorController(train_engine=cls, config=config, scheduler=scheduler)


def create_actor(config: PlatoonPPOActorConfig) -> FSDPPPOActor:
    """Create the Platoon actor implementation for the configured loss."""

    return PlatoonPPOActor(config)
