from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import time
from functools import lru_cache
from functools import wraps
from typing import Any


def _patch_hf_tokenizer_download_race() -> None:
    """Avoid corrupt Hugging Face tokenizer JSON during proxy worker startup.

    AReaL's helper force-downloads tokenizers. When multiple proxy workers start
    together, they can concurrently rewrite the same HF cache entry, leaving JSON
    files with duplicated content. Load from cache by default and serialize the
    one forced refresh path used to repair a bad cache entry.
    """

    import transformers  # pyright: ignore[reportMissingImports]
    import areal.utils.hf_utils as hf_utils  # pyright: ignore[reportMissingImports]

    original = hf_utils.load_hf_tokenizer
    if getattr(original, "__platoon_hf_tokenizer_patch__", False):
        return

    @lru_cache(maxsize=8)
    def _load_hf_tokenizer_without_racy_force_download(
        model_name_or_path: str,
        fast_tokenizer=True,
        padding_side: str | None = None,
    ) -> transformers.PreTrainedTokenizerFast:
        kwargs = {}
        if padding_side is not None:
            kwargs["padding_side"] = padding_side

        lock_name = hashlib.sha256(model_name_or_path.encode("utf-8")).hexdigest()
        lock_path = os.path.join(tempfile.gettempdir(), f"platoon-hf-tokenizer-{lock_name}.lock")
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_name_or_path,
                    fast_tokenizer=fast_tokenizer,
                    trust_remote_code=True,
                    force_download=False,
                    **kwargs,
                )
            except json.JSONDecodeError:
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_name_or_path,
                    fast_tokenizer=fast_tokenizer,
                    trust_remote_code=True,
                    force_download=True,
                    **kwargs,
                )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            return tokenizer

    _load_hf_tokenizer_without_racy_force_download.__platoon_hf_tokenizer_patch__ = True
    hf_utils.load_hf_tokenizer = _load_hf_tokenizer_without_racy_force_download

    try:
        import areal.experimental.openai.proxy.proxy_rollout_server as proxy_server  # pyright: ignore[reportMissingImports]

        proxy_server.load_hf_tokenizer = _load_hf_tokenizer_without_racy_force_download
    except Exception:
        pass


def _patch_model_response_custom_stop_sequences() -> None:
    """Allow AReaL proxy responses that stop on custom text stop sequences.

    Platoon agents use OpenAI ``stop`` sequences such as ``</python>``. SGLang
    reports these as ``stop_reason="stop"`` without necessarily appending the
    tokenizer EOS/PAD token, while this AReaL release requires EOS/PAD for every
    non-length stop. In that case the generated tokens are already the training
    target, so return them unchanged instead of failing the proxy request.
    """

    from areal.api.io_struct import ModelResponse  # pyright: ignore[reportMissingImports]

    original = ModelResponse.output_tokens_without_stop
    if getattr(original.fget, "__platoon_custom_stop_patch__", False):
        return

    def _output_tokens_without_custom_stop_error(self) -> list[int]:
        if self.tokenizer is None:
            raise ValueError("tokenizer is None, cannot get output_tokens_without_stop")
        if self.stop_reason not in ["length", "abort"] and self.output_tokens:
            if not self.end_with_stop:
                return self.output_tokens
            pad_or_eos_len = 0
            eos_id = self.tokenizer.eos_token_id
            pad_id = self.tokenizer.pad_token_id
            stop_tokens = {eos_id, pad_id}
            stop_tokens.discard(None)
            for tok in reversed(self.output_tokens):
                if tok in stop_tokens:
                    pad_or_eos_len += 1
                else:
                    break
            if pad_or_eos_len == len(self.output_tokens):
                raise ValueError(
                    "All output_tokens are EOS or PAD tokens; cannot strip stop tokens without removing entire output."
                )
            return self.output_tokens[:-pad_or_eos_len]
        return self.output_tokens

    _output_tokens_without_custom_stop_error.__platoon_custom_stop_patch__ = True
    ModelResponse.output_tokens_without_stop = property(_output_tokens_without_custom_stop_error)


def _patch_batch_task_dispatcher_idle_submit() -> None:
    """Avoid silent dispatcher stalls when no rollout tasks are in flight.

    AReaL's dispatcher reserves ``batch_size`` slots in the async runner queue
    before submitting more work. If that reservation leaves zero apparent
    capacity, ``active_submit_and_wait`` can wait for results even though no
    tasks were ever submitted. Platoon's long rollouts make that look like a
    generation hang, so patch the capacity calculation to use actual free queue
    slots and emit periodic state when dispatch is blocked.
    """

    from areal.infra.workflow_executor import BatchTaskDispatcher  # pyright: ignore[reportMissingImports]

    original = BatchTaskDispatcher.active_submit_and_wait
    if getattr(original, "__platoon_idle_submit_patch__", False):
        return

    @wraps(original)
    def _active_submit_and_wait_with_idle_guard(
        self,
        input_generator,
        batch_size: int,
        dynamic_bs: bool = False,
    ) -> list[Any]:
        accepted_cnt = 0
        total_attempts = 0
        results = []
        last_blocked_log = 0.0

        while True:
            self._check_thread_exception()

            with self._input_cv:
                pending_inputs = len(self._pending_inputs)
            runner_input_size = self.runner.get_input_queue_size()
            cap_staleness = self.staleness_manager.get_pending_limit() - pending_inputs

            if self.runner.max_queue_size < batch_size:
                raise ValueError(
                    "Inference engine config's queue size is too small: "
                    f"{self.runner.max_queue_size} < batch size {batch_size}."
                )

            free_runner_slots = self.runner.max_queue_size - runner_input_size
            capacity = min(cap_staleness, free_runner_slots)

            if capacity > 0 and not self.runner.paused.is_set():
                for _ in range(min(batch_size, capacity)):
                    try:
                        self.submit_task_input(next(input_generator))
                    except StopIteration:
                        raise RuntimeError(
                            "Input generator exhausted before batch completion. "
                            "Use cycle_dataloader() or provide an infinite generator."
                        ) from None
            else:
                now = time.monotonic()
                if now - last_blocked_log >= 30.0:
                    last_blocked_log = now
                    stats = self.staleness_manager.get_stats()
                    self.logger.warning(
                        "Rollout dispatch is waiting without submit capacity: "
                        "batch_size=%s accepted=%s total_attempts=%s "
                        "pending_inputs=%s runner_input_queue=%s runner_output_queue=%s "
                        "max_queue_size=%s cap_staleness=%s free_runner_slots=%s "
                        "paused=%s stats=%s",
                        batch_size,
                        accepted_cnt,
                        total_attempts,
                        pending_inputs,
                        runner_input_size,
                        self.runner.get_output_queue_size(),
                        self.runner.max_queue_size,
                        cap_staleness,
                        free_runner_slots,
                        self.runner.paused.is_set(),
                        stats,
                    )

            try:
                arrived = self.wait_results(count=batch_size - accepted_cnt, timeout=1)
            except TimeoutError:
                arrived = []

            for res in arrived:
                is_accepted = res is not None
                if not is_accepted:
                    if dynamic_bs:
                        total_attempts += 1
                        if total_attempts >= batch_size:
                            break
                    continue

                accepted_cnt += 1
                total_attempts += 1
                results.append(res)

                if dynamic_bs:
                    if total_attempts >= batch_size:
                        break
                elif accepted_cnt >= batch_size:
                    break
            else:
                continue
            break

        return results

    _active_submit_and_wait_with_idle_guard.__platoon_idle_submit_patch__ = True
    BatchTaskDispatcher.active_submit_and_wait = _active_submit_and_wait_with_idle_guard


def _patch_local_scheduler_fork_ready_timeout() -> None:
    """Give forked proxy workers enough time to import and start serving.

    AReaL hardcodes fork readiness checks to 60 seconds. Platoon's proxy
    workers import the workflow stack and can take longer on cold starts, even
    though the workers are healthy once the server finishes booting.
    """

    from areal.infra.scheduler.local import LocalScheduler  # pyright: ignore[reportMissingImports]

    original = LocalScheduler._wait_for_fork_ready
    if getattr(original, "__platoon_fork_ready_timeout_patch__", False):
        return

    @wraps(original)
    async def _wait_for_fork_ready_with_platoon_timeout(
        session,
        host: str,
        port: int,
        timeout: float = 60,
    ) -> bool:
        if timeout == 60:
            timeout = float(os.environ.get("PLATOON_AREAL_FORK_READY_TIMEOUT", "900"))
        return await original(session, host, port, timeout=timeout)

    _wait_for_fork_ready_with_platoon_timeout.__platoon_fork_ready_timeout_patch__ = True
    LocalScheduler._wait_for_fork_ready = staticmethod(_wait_for_fork_ready_with_platoon_timeout)


def _patch_remote_inf_engine_proxy_resolution() -> None:
    """Let custom RolloutWorkflow instances receive worker-local proxy URLs.

    Upstream AReaL already threads a per-worker ``proxy_addr`` through
    ``RemoteInfEngine.submit()``, but only injects it when wrapping agent-like
    workflows in ``OpenAIProxyWorkflow``. Platoon's custom workflows are already
    ``RolloutWorkflow`` instances, so inline mode needs a small patch to bind the
    same worker-local proxy URL onto those workflow objects before execution.
    """

    from areal.api import RolloutWorkflow  # pyright: ignore[reportMissingImports]
    from areal.infra.remote_inf_engine import RemoteInfEngine  # pyright: ignore[reportMissingImports]

    original = RemoteInfEngine._resolve_workflow
    if getattr(original, "__platoon_proxy_patch__", False):
        return

    def _inject_proxy_addr(workflow: RolloutWorkflow, proxy_addr: str) -> None:
        setter = getattr(workflow, "set_proxy_base_url", None)
        if callable(setter):
            setter(proxy_addr)
            return
        if hasattr(workflow, "proxy_base_url"):
            setattr(workflow, "proxy_base_url", proxy_addr)

    @wraps(original)
    def _resolve_workflow_with_proxy_addr(
        self,
        workflow: Any,
        workflow_kwargs: dict[str, Any] | None,
        group_size: int = 1,
        proxy_addr: str | None = None,
    ) -> RolloutWorkflow:
        resolved = original(
            self,
            workflow,
            workflow_kwargs,
            group_size=group_size,
            proxy_addr=proxy_addr,
        )
        if proxy_addr is not None and isinstance(resolved, RolloutWorkflow):
            _inject_proxy_addr(resolved, proxy_addr)
        return resolved

    _resolve_workflow_with_proxy_addr.__platoon_proxy_patch__ = True
    RemoteInfEngine._resolve_workflow = _resolve_workflow_with_proxy_addr


def _patch_fsdp_recover_lr_scheduler_state() -> None:
    """Persist FSDP LR scheduler state in AReaL recover checkpoints.

    This AReaL release saves model and optimizer state in DCP recover
    checkpoints, but not ``lr_scheduler.state_dict()``. On resume, the optimizer
    LR is restored briefly, then the fresh scheduler advances from the beginning
    of the schedule. Save the scheduler state as a small sidecar file so resume
    preserves the schedule.
    """

    try:
        import torch  # pyright: ignore[reportMissingImports]
        import torch.distributed as dist  # pyright: ignore[reportMissingImports]
        from areal.engine.fsdp_engine import FSDPEngine  # pyright: ignore[reportMissingImports]
    except Exception:
        return

    original_save_to_dcp = FSDPEngine._save_to_dcp
    original_load_from_dcp = FSDPEngine._load_from_dcp
    if getattr(original_save_to_dcp, "__platoon_lr_scheduler_patch__", False):
        return

    scheduler_state_name = "lr_scheduler.pt"

    def _state_path(path: str) -> str:
        return os.path.join(path, scheduler_state_name)

    def _dist_is_initialized() -> bool:
        return dist.is_available() and dist.is_initialized()

    def _is_rank_zero() -> bool:
        return not _dist_is_initialized() or dist.get_rank() == 0

    def _barrier(engine: Any) -> None:
        if _dist_is_initialized():
            dist.barrier(group=getattr(engine, "cpu_group", None))

    def _warn(engine: Any, message: str) -> None:
        logger = getattr(engine, "logger", None)
        if logger is not None:
            logger.warning(message)

    @wraps(original_save_to_dcp)
    def _save_to_dcp_with_lr_scheduler(self, path: str, with_optim: bool):
        result = original_save_to_dcp(self, path, with_optim)
        lr_scheduler = getattr(self, "lr_scheduler", None)
        if with_optim and lr_scheduler is not None:
            if _is_rank_zero():
                torch.save(lr_scheduler.state_dict(), _state_path(path))
            _barrier(self)
        return result

    @wraps(original_load_from_dcp)
    def _load_from_dcp_with_lr_scheduler(self, path: str, with_optim: bool):
        result = original_load_from_dcp(self, path, with_optim)
        lr_scheduler = getattr(self, "lr_scheduler", None)
        if with_optim and lr_scheduler is not None:
            scheduler_path = _state_path(path)
            if os.path.exists(scheduler_path):
                state_dict = torch.load(scheduler_path, map_location="cpu", weights_only=False)
                lr_scheduler.load_state_dict(state_dict)
            else:
                _warn(
                    self,
                    f"LR scheduler state not found in recover checkpoint: {scheduler_path}. "
                    "The scheduler will resume from its freshly initialized state.",
                )
        return result

    _save_to_dcp_with_lr_scheduler.__platoon_lr_scheduler_patch__ = True
    _load_from_dcp_with_lr_scheduler.__platoon_lr_scheduler_patch__ = True
    FSDPEngine._save_to_dcp = _save_to_dcp_with_lr_scheduler
    FSDPEngine._load_from_dcp = _load_from_dcp_with_lr_scheduler


def apply_all_patches() -> None:
    """Apply Platoon compatibility patches for the current AReaL release."""

    _patch_hf_tokenizer_download_race()
    _patch_model_response_custom_stop_sequences()
    _patch_batch_task_dispatcher_idle_submit()
    _patch_local_scheduler_fork_ready_timeout()
    _patch_remote_inf_engine_proxy_resolution()
    _patch_fsdp_recover_lr_scheduler_state()
