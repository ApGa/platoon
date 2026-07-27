"""Custom PPO actor support for Platoon's AReaL backend."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from areal.api import Scheduler
from areal.api.cli_args import MicroBatchSpec
from areal.engine import FSDPPPOActor
from areal.infra import current_platform
from areal.trainer.ppo.actor import PPOActor, PPOActorController
from areal.trainer.ppo.stats import infer_token_denominator
from areal.utils import logging, stats_tracker
from areal.utils.data import batched_call, split_padded_tensor_dict_into_mb_list
from areal.utils.perf_tracer import trace_perf

from platoon.train.areal.config_defs import PlatoonPPOActorConfig
from platoon.train.areal.fp32_lm_head import install_fp32_lm_head_output_hooks
from platoon.train.areal.loss_functions import build_loss_fn
from platoon.train.areal.numerical_stability import (
    aggregate_optimizer_update_results,
    install_nonfinite_gradient_guard,
    make_optimizer_update_result,
    optimizer_update_succeeded,
)
from platoon.train.areal.router_replay import (
    ROUTED_EXPERTS_FIELD,
    ROUTED_EXPERTS_VALID_FIELD,
    assert_engine_router_replay_batch_consumed,
    configure_router_replay_engine,
    discard_staged_engine_router_replay_batch,
    pop_and_split_actor_router_replay,
    router_replay_initialization,
    run_router_replay_forward_backward,
    stage_engine_router_replay_batch,
)

if TYPE_CHECKING:
    # PlatoonMegatronPPOActor is built dynamically by _get_platoon_megatron_actor_cls
    # (to avoid importing Megatron / Transformer Engine eagerly). Alias the base
    # here so type checkers can resolve the name used in annotations below.
    from areal.engine import MegatronPPOActor as PlatoonMegatronPPOActor

logger = logging.getLogger("PlatoonActor")


class PlatoonActorImpl(PPOActor):
    """PPO actor implementation with Platoon loss selection."""

    def __init__(self, config: PlatoonPPOActorConfig, engine: Any):
        super().__init__(config, engine)
        self.config = config

    def _make_loss_fn(self, current_version: int | None) -> Callable[..., torch.Tensor]:
        common_kwargs = dict(
            importance_sampling_level=self.config.importance_sampling_level,
            eps_clip=self.config.eps_clip,
            eps_clip_higher=self.config.eps_clip_higher,
            c_clip=self.config.c_clip,
            # AReaL HEAD replaced behave_imp_weight_{cap,mode} with the
            # rejection_sampling sub-config (see PPOActorConfig).
            rejection_sampling=self.config.rejection_sampling,
            m2_threshold=self.m2_threshold,
            current_version=current_version,
            prox_logp_method=self.config.prox_logp_method,
            use_sapo_loss=self.config.use_sapo_loss,
            sapo_tau_pos=self.config.sapo_tau_pos,
            sapo_tau_neg=self.config.sapo_tau_neg,
            use_decoupled_loss=self.config.use_decoupled_loss,
        )
        logger.info(
            "Using Platoon loss_fn=%s loss_fn_kwargs=%s current_version=%s",
            self.config.loss_fn,
            self.config.loss_fn_kwargs,
            current_version,
        )
        return build_loss_fn(
            self.config.loss_fn,
            loss_fn_kwargs=self.config.loss_fn_kwargs,
            common_kwargs=common_kwargs,
        )

    @trace_perf("platoon_ppo_actor.ppo_update", category="compute")
    @stats_tracker.scope_func_wrapper("ppo_actor")
    def ppo_update(
        self,
        data: list[dict[str, Any]],
    ) -> dict[str, list[bool]]:
        return batched_call(self._ppo_update, data, unpack=False)

    def _ppo_update(self, data: dict[str, Any]) -> dict[str, list[bool]]:
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
        if self.config.rejection_sampling is not None:
            rs = self.config.rejection_sampling
            scalars["rs_upper"] = rs.upper
            if rs.lower is not None:
                scalars["rs_lower"] = rs.lower
        stats_tracker.scalar(**scalars)

        if self.config.log_agent_stats:
            stats_tracker.stat(
                **{k: data[k].float() for k in self.config.log_agent_stats_keys},
                denominator="agent",
            )

        for key in ["rewards", "tot_rewards", "kl_rewards"]:
            data.pop(key, None)

        self.engine.train()
        replay_data = {"attention_mask": data["attention_mask"]}
        for field_name in (ROUTED_EXPERTS_FIELD, ROUTED_EXPERTS_VALID_FIELD):
            if field_name in data:
                replay_data[field_name] = data.pop(field_name)
        mb_inputs = split_padded_tensor_dict_into_mb_list(
            data,
            mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
        )
        replay_batches = pop_and_split_actor_router_replay(replay_data, mb_inputs, self.config)

        with stats_tracker.scope("update"):
            current_version = self.engine.get_version()
            loss_fn = self._make_loss_fn(current_version)
            minibatch_update_successes: list[bool] = []
            for mb, replay_batch in zip(mb_inputs.mbs, replay_batches, strict=True):
                stage_engine_router_replay_batch(self.engine, replay_batch)
                try:
                    train_stat = self.engine.train_batch(
                        mb,
                        loss_fn=loss_fn,
                        loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
                    )
                    assert_engine_router_replay_batch_consumed(self.engine)
                finally:
                    discard_staged_engine_router_replay_batch(self.engine)
                stats_tracker.scalar(**train_stat)
                minibatch_update_successes.append(
                    optimizer_update_succeeded(
                        train_stat,
                        require_finite_grad_norm=bool(
                            self.config.optimizer is not None and self.config.optimizer.gradient_clipping > 0
                        ),
                    )
                )
            applied_minibatches = sum(minibatch_update_successes)
            attempted_minibatches = len(minibatch_update_successes)
            stats_tracker.scalar(
                optimizer_minibatches_attempted=float(attempted_minibatches),
                optimizer_minibatches_applied=float(applied_minibatches),
                optimizer_minibatches_skipped=float(
                    attempted_minibatches - applied_minibatches
                ),
                optimizer_partial_update=float(
                    0 < applied_minibatches < attempted_minibatches
                ),
            )
            return make_optimizer_update_result(minibatch_update_successes)


class PlatoonPPOActorController(PPOActorController):
    """Actor controller exposing Platoon's worker-side memory hygiene RPC."""

    def ppo_update(self, *args: Any, **kwargs: Any) -> bool:
        results = self._custom_function_call(
            "ppo_update",
            *args,
            rpc_meta={"broadcast": True},
            **kwargs,
        )
        return aggregate_optimizer_update_results(results)

    def clear_device_cache(self) -> None:
        self._custom_function_call("clear_device_cache")


class PlatoonPPOActor(FSDPPPOActor):
    """FSDP PPO actor with Platoon loss selection."""

    def __init__(self, config: PlatoonPPOActorConfig):
        super().__init__(config)
        self.actor = PlatoonActorImpl(config, self)

    def clear_device_cache(self) -> None:
        """Release cached allocator blocks on this worker's GPU.

        The pre-migration SPMD trainer called ``torch.cuda.empty_cache()`` on
        every rank between training phases. In single-controller mode the
        trainer process owns no GPU, so the cleanup must run on the workers.
        Freeing the cache before NCCL-heavy phases (weight-update broadcast,
        DCP checkpoint save) matters because NCCL allocates its buffers outside
        PyTorch's caching allocator.
        """
        current_platform.clear_memory()

    @classmethod
    def as_controller(cls, config: PlatoonPPOActorConfig, scheduler: Scheduler):
        return PlatoonPPOActorController(train_engine=cls, config=config, scheduler=scheduler)


# Cache for the lazily-built Megatron actor subclass (see below).
_platoon_megatron_actor_cls: type | None = None


def _get_platoon_megatron_actor_cls() -> type:
    """Lazily build and return ``PlatoonMegatronPPOActor``.

    ``MegatronPPOActor`` pulls in ``megatron.bridge``, which *unconditionally*
    imports ``transformer_engine`` (e.g. ``megatron.bridge.peft.lora_layers``).
    Importing it at module load would break FSDP-only environments that don't
    install Transformer Engine, so we defer both the import and the subclass
    definition until the Megatron backend is actually requested.

    Mirrors ``PlatoonPPOActor`` for the Megatron training backend: the base
    builds ``self.actor`` in ``__init__`` as the upstream ``PPOActor``; we swap
    in ``PlatoonActorImpl`` so Platoon's loss selection and stats apply to the
    Megatron path too.
    """
    global _platoon_megatron_actor_cls
    if _platoon_megatron_actor_cls is not None:
        return _platoon_megatron_actor_cls

    from areal.engine import MegatronPPOActor

    class PlatoonMegatronPPOActor(MegatronPPOActor):
        """Megatron PPO actor with Platoon loss selection."""

        def __init__(self, config: PlatoonPPOActorConfig):
            super().__init__(config)
            self.actor = PlatoonActorImpl(config, self)

        def initialize(self, *args: Any, **kwargs: Any) -> Any:
            with router_replay_initialization(self):
                result = super().initialize(*args, **kwargs)
            configure_router_replay_engine(self)
            install_nonfinite_gradient_guard(self.optimizer, logger=logger)
            enabled = bool(self.config.megatron.enable_fp32_lm_head)
            hook_count = install_fp32_lm_head_output_hooks(
                self.model,
                enabled=enabled,
                is_critic=self.config.is_critic,
            )
            if enabled and not self.config.is_critic:
                logger.info(
                    "FP32 LM-head output enabled; installed %d hook(s) on this pipeline rank",
                    hook_count,
                )
            return result

        def forward_backward_batch(
            self,
            mb_list,
            process_output_fn,
            forward_only: bool = False,
            gather_cp_output: bool = False,
        ):
            return run_router_replay_forward_backward(
                self,
                super().forward_backward_batch,
                mb_list,
                process_output_fn,
                forward_only=forward_only,
                gather_cp_output=gather_cp_output,
            )

        def clear_device_cache(self) -> None:
            """Release cached allocator blocks on this worker's GPU.

            See ``PlatoonPPOActor.clear_device_cache``; the Megatron engine has
            the same single-controller memory-hygiene requirement before
            NCCL-heavy weight-update and checkpoint phases.
            ``current_platform.clear_memory()`` is backend-agnostic.
            """
            current_platform.clear_memory()

        @classmethod
        def as_controller(cls, config: PlatoonPPOActorConfig, scheduler: Scheduler):
            return PlatoonPPOActorController(train_engine=cls, config=config, scheduler=scheduler)

    _platoon_megatron_actor_cls = PlatoonMegatronPPOActor
    return _platoon_megatron_actor_cls


def __getattr__(name: str):
    # PEP 562 module-level lazy attribute. Lets both
    # ``actor.PlatoonMegatronPPOActor`` and
    # ``from platoon.train.areal.actor import PlatoonMegatronPPOActor`` resolve
    # without importing Megatron / Transformer Engine unless the Megatron actor
    # is actually used.
    if name == "PlatoonMegatronPPOActor":
        return _get_platoon_megatron_actor_cls()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def create_actor(config: PlatoonPPOActorConfig, backend: str = "fsdp") -> "PlatoonPPOActor | PlatoonMegatronPPOActor":
    """Create the Platoon actor implementation for the configured loss/backend."""

    if backend == "megatron":
        return _get_platoon_megatron_actor_cls()(config)
    if backend == "fsdp":
        return PlatoonPPOActor(config)
    raise ValueError(f"Unsupported Platoon actor backend: {backend!r} (expected 'fsdp' or 'megatron')")
