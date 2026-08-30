"""Narrow SGLang scheduler compatibility fixes used by AReaL rollouts.

This module intentionally lives outside :mod:`platoon.train.areal`: SGLang
imports the scheduler process target in a fresh multiprocessing ``spawn``
child, where importing the full training backend would be both unnecessary and
fragile.
"""

from __future__ import annotations

import inspect
import logging
from functools import wraps
from typing import Any

from areal.v2.inference_service.sglang.scheduler import (
    areal_run_scheduler_process as _ORIGINAL_AREAL_RUN_SCHEDULER_PROCESS,
)

logger = logging.getLogger(__name__)


def install_awex_sglang_radix_cache_flush_patch(adapter_cls: Any | None = None) -> bool:
    """Invalidate prefix KV entries after AWEX changes the policy weights.

    AReaL's AWEX SGLang adapter updates model parameters in place without using
    SGLang's native update-weight request, so SGLang does not get its usual
    opportunity to flush the radix cache. Reusing a prefix computed by an old
    policy after the update would mix KV state from two policy versions.

    This runs inside every spawned scheduler process. It is deliberately a
    no-op while AReaL's ``disable_radix_cache`` default remains enabled.
    """

    if adapter_cls is None:
        from areal.v2.weight_update.awex.sglang_adapter import AwexSGLangAdapter

        adapter_cls = AwexSGLangAdapter

    method_names = (
        "execute_weight_update",
        "execute_colocate_weight_update",
    )
    originals = {name: getattr(adapter_cls, name) for name in method_names}
    patched = {
        name: getattr(method, "__platoon_radix_cache_flush_patch__", False) for name, method in originals.items()
    }
    if all(patched.values()):
        return False
    if any(patched.values()):
        raise RuntimeError("AReaL's AWEX SGLang adapter is only partially patched for radix-cache flushing")

    expected_parameters = ("self", "version")
    for name, original in originals.items():
        actual_parameters = tuple(inspect.signature(original).parameters)
        if actual_parameters != expected_parameters:
            raise RuntimeError(
                f"Unsupported AwexSGLangAdapter.{name} signature: "
                f"expected {expected_parameters}, got {actual_parameters}"
            )

    def _wrap_weight_update(name: str, original: Any):
        @wraps(original)
        def _update_then_flush(self, version: int):
            result = original(self, version)
            scheduler = self._scheduler
            if not scheduler.server_args.disable_radix_cache:
                if not scheduler.flush_cache():
                    raise RuntimeError(
                        "SGLang radix-cache flush failed after "
                        f"AwexSGLangAdapter.{name}; refusing to serve stale KV cache entries"
                    )
            return result

        _update_then_flush.__platoon_radix_cache_flush_patch__ = True
        return _update_then_flush

    for name, original in originals.items():
        setattr(adapter_cls, name, _wrap_weight_update(name, original))
    return True


def install_routed_experts_capture_capacity_patch(
    capturer_module: Any | None = None,
    server_args_getter: Any | None = None,
) -> bool:
    """Size SGLang's route cache by tokens when chunked prefill is disabled.

    SGLang 0.5.10 sizes ``_RoutedExpertsDeviceCache`` with
    ``max(chunked_prefill_size * dp_size, max_running_requests)``.  With
    AReaL's deliberate ``chunked_prefill_size=-1`` this degenerates to a
    request-count capacity, even though the cache is indexed by token.  Reuse
    the ``max_running_requests`` argument *locally* as the minimum cache row
    count passed to the original factory; the scheduler's actual request limit
    and its request/KV pools remain unchanged.

    The patch is active only when routed-expert capture itself is enabled.
    """

    if capturer_module is None:
        from sglang.srt.layers.moe import routed_experts_capturer as capturer_module
    if server_args_getter is None:
        from sglang.srt.server_args import get_global_server_args as server_args_getter

    capturer_cls = capturer_module.RoutedExpertsCapturer
    original = capturer_cls.create
    if getattr(original, "__platoon_disabled_chunked_prefill_capacity_patch__", False):
        return False

    expected_parameters = (
        "enable",
        "model_config",
        "num_fused_shared_experts",
        "num_tokens",
        "max_running_requests",
        "device",
    )
    actual_parameters = tuple(inspect.signature(original).parameters)
    if actual_parameters != expected_parameters:
        raise RuntimeError(
            "Unsupported SGLang RoutedExpertsCapturer.create signature: "
            f"expected {expected_parameters}, got {actual_parameters}"
        )

    def _create_with_token_capacity(
        enable: bool,
        model_config: Any,
        num_fused_shared_experts: int,
        num_tokens: int,
        max_running_requests: int,
        device: str,
    ):
        cache_rows = max_running_requests
        if enable:
            server_args = server_args_getter()
            chunked_prefill_size = getattr(server_args, "chunked_prefill_size", None)
            if chunked_prefill_size is not None and chunked_prefill_size <= 0:
                max_prefill_tokens = getattr(server_args, "max_prefill_tokens", None)
                dp_size = getattr(server_args, "dp_size", 1)
                if (
                    isinstance(max_prefill_tokens, bool)
                    or not isinstance(max_prefill_tokens, int)
                    or max_prefill_tokens <= 0
                ):
                    raise RuntimeError(
                        "Routed-expert capture with disabled chunked prefill requires "
                        f"a positive integer max_prefill_tokens, got {max_prefill_tokens!r}"
                    )
                if isinstance(dp_size, bool) or not isinstance(dp_size, int) or dp_size <= 0:
                    raise RuntimeError(f"Routed-expert capture requires a positive integer dp_size, got {dp_size!r}")
                token_capacity = max_prefill_tokens * dp_size
                cache_rows = max(max_running_requests, token_capacity)
                logger.warning(
                    "Expanded SGLang routed-expert capture capacity from %d request rows "
                    "to %d token rows because chunked prefill is disabled",
                    max_running_requests,
                    cache_rows,
                )

        return original(
            enable=enable,
            model_config=model_config,
            num_fused_shared_experts=num_fused_shared_experts,
            num_tokens=num_tokens,
            max_running_requests=cache_rows,
            device=device,
        )

    _create_with_token_capacity.__platoon_disabled_chunked_prefill_capacity_patch__ = True
    capturer_cls.create = staticmethod(_create_with_token_capacity)
    return True


def areal_run_scheduler_process_with_routed_experts_fix(*args, **kwargs) -> None:
    """Spawn-safe AReaL scheduler target that installs compatibility fixes."""

    install_routed_experts_capture_capacity_patch()
    install_awex_sglang_radix_cache_flush_patch()
    _ORIGINAL_AREAL_RUN_SCHEDULER_PROCESS(*args, **kwargs)


areal_run_scheduler_process_with_routed_experts_fix.__platoon_routed_experts_scheduler_target__ = True


def install_areal_scheduler_process_target() -> bool:
    """Make AReaL pass the canonical Platoon target to SGLang ``spawn`` children."""

    from areal.v2.inference_service.sglang import scheduler as scheduler_module

    current = scheduler_module.areal_run_scheduler_process
    if getattr(current, "__platoon_routed_experts_scheduler_target__", False):
        return False
    if current is not _ORIGINAL_AREAL_RUN_SCHEDULER_PROCESS:
        raise RuntimeError("AReaL's SGLang scheduler target was unexpectedly replaced before Platoon startup")
    scheduler_module.areal_run_scheduler_process = areal_run_scheduler_process_with_routed_experts_fix
    return True
