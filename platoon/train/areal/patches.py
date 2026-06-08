from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import os
import sysconfig
import tempfile
import time
from functools import lru_cache
from functools import wraps
from pathlib import Path
from typing import Any


def _prepend_nvidia_cuda_library_paths() -> None:
    """Expose NVIDIA wheel libs before peft imports transformer_engine.

    Transformer Engine's cu13 build looks for ``libcublas.so.13`` under
    ``site-packages/nvidia/``. Those paths are not on the default loader path
    inside enroot containers, which triggers import-time failures even when the
    platoon image and venv are otherwise correct.
    """

    if getattr(_prepend_nvidia_cuda_library_paths, "__platoon_cuda_lib_patch__", False):
        return

    lib_dirs: list[str] = []
    nvidia_dir = Path(sysconfig.get_path("purelib")) / "nvidia"
    if nvidia_dir.is_dir():
        for lib_dir in sorted(nvidia_dir.glob("**/lib")):
            if lib_dir.is_dir() and any(lib_dir.glob("*.so*")):
                lib_dirs.append(str(lib_dir))
        for lib_dir in sorted(nvidia_dir.glob("**/lib64")):
            if lib_dir.is_dir() and any(lib_dir.glob("*.so*")):
                lib_dirs.append(str(lib_dir))

    for module_name in ("nvidia.cublas", "nvidia.cuda_runtime", "nvidia.cudnn"):
        try:
            module = __import__(module_name, fromlist=["lib"])
            lib_module = getattr(module, "lib", None)
            if lib_module is not None and hasattr(lib_module, "__file__"):
                lib_dirs.append(str(Path(lib_module.__file__).resolve().parent))
        except Exception:
            pass

    for system_dir in (
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-13/lib64",
        "/usr/local/cuda-12/lib64",
    ):
        if os.path.isdir(system_dir):
            lib_dirs.append(system_dir)

    if not lib_dirs:
        _prepend_nvidia_cuda_library_paths.__platoon_cuda_lib_patch__ = True
        return

    deduped = list(dict.fromkeys(lib_dirs))
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        ":".join(deduped) + (":" + existing if existing else "")
    )

    # LD_LIBRARY_PATH is read by the dynamic loader at process startup, so
    # changing it here mainly helps forked subprocesses. Preload the CUDA 13
    # libs by absolute path as well so the current trainer process can import
    # transformer_engine_cu13 immediately after this patch runs.
    for lib_name in ("libcudart.so.13", "libcublasLt.so.13", "libcublas.so.13"):
        for lib_dir in deduped:
            lib_path = Path(lib_dir) / lib_name
            if lib_path.exists():
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                break

    _prepend_nvidia_cuda_library_paths.__platoon_cuda_lib_patch__ = True


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


def _normalize_fsdp_wrap_classes(classes: Any) -> list[str]:
    if classes is None:
        return []
    if isinstance(classes, str):
        return [classes]
    if isinstance(classes, (set, tuple)):
        return list(classes)
    return list(classes)


def _flatten_message_list_content(messages: list[dict[str, Any]]) -> None:
    """Convert OpenAI list-shaped text content blocks to plain strings.

    OpenHands (and other clients) may send ``content`` as
    ``[{"type": "text", "text": "..."}]``. Hugging Face ``apply_chat_template``
    and AReaL's interaction cache expect string ``content`` for text-only turns.
    """

    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if any(
            isinstance(item, dict)
            and item.get("type") in ("image_url", "image", "input_image")
            for item in content
        ):
            continue
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        message["content"] = "".join(text_parts)


def _patch_areal_openai_message_content_flatten() -> None:
    """Flatten OpenHands-style list content before HF chat templates and proxy cache."""

    import areal.experimental.openai.client as client_module  # pyright: ignore[reportMissingImports]

    original_ensure = client_module._ensure_message_dict_list
    if getattr(original_ensure, "__platoon_message_content_patch__", False):
        return

    @wraps(original_ensure)
    def _ensure_message_dict_list_with_flatten(
        name: str,
        value: list[Any],
    ) -> list[dict[str, Any]]:
        normalized = original_ensure(name, value)
        _flatten_message_list_content(normalized)
        return normalized

    _ensure_message_dict_list_with_flatten.__platoon_message_content_patch__ = True
    client_module._ensure_message_dict_list = _ensure_message_dict_list_with_flatten


def _coerce_apply_chat_template_token_ids(token_ids: Any) -> Any:
    """Return a plain ``list[int]`` from Transformers chat-template output.

    Transformers 5 defaults ``return_dict=True`` for ``apply_chat_template``, which
    returns a ``BatchEncoding``. AReaL expects token-id lists for SGLang payloads.
    """

    if token_ids is None:
        return []
    if hasattr(token_ids, "input_ids"):
        ids = token_ids["input_ids"]
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            return list(ids[0])
        return list(ids)
    if isinstance(token_ids, list):
        return token_ids
    return list(token_ids)


def _patch_transformers_apply_chat_template_return_type() -> None:
    """Coerce ``apply_chat_template`` token output for AReaL/SGLang compatibility."""

    import transformers  # pyright: ignore[reportMissingImports]

    base = transformers.PreTrainedTokenizerBase
    original = base.apply_chat_template
    if getattr(original, "__platoon_apply_chat_template_patch__", False):
        return

    @wraps(original)
    def apply_chat_template_with_list_token_ids(self, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        if kwargs.get("tokenize", True):
            return _coerce_apply_chat_template_token_ids(result)
        return result

    apply_chat_template_with_list_token_ids.__platoon_apply_chat_template_patch__ = True
    base.apply_chat_template = apply_chat_template_with_list_token_ids


def _patch_areal_proxy_rollout_fork_command() -> None:
    """Run forked proxy workers through Platoon's patched entry module."""

    from areal.infra.scheduler.local import LocalScheduler  # pyright: ignore[reportMissingImports]

    areal_proxy_module = "areal.experimental.openai.proxy.proxy_rollout_server"
    platoon_proxy_module = "platoon.areal_proxy_rollout"

    original = LocalScheduler._fork_single_worker
    if getattr(original, "__platoon_proxy_fork_patch__", False):
        return

    @wraps(original)
    async def _fork_single_worker_with_platoon_proxy(
        self,
        session,
        role: str,
        idx: int,
        target_wi,
        target_role: str,
        command: str | None = None,
    ):
        if command == areal_proxy_module:
            command = platoon_proxy_module
        return await original(
            self,
            session,
            role,
            idx,
            target_wi,
            target_role,
            command,
        )

    _fork_single_worker_with_platoon_proxy.__platoon_proxy_fork_patch__ = True
    LocalScheduler._fork_single_worker = _fork_single_worker_with_platoon_proxy


def _patch_areal_fsdp_wrap_classes_set_compat() -> None:
    """Normalize FSDP wrap class names before AReaL indexes them by position.

    Newer Transformers releases can expose ``_no_split_modules`` as a ``set``.
    AReaL's ``apply_fsdp2`` still does ``fsdp_transformer_layer_cls_to_wrap[0]``,
    which crashes with ``TypeError: 'set' object is not subscriptable``.
    """

    import areal.engine.fsdp_utils as fsdp_utils  # pyright: ignore[reportMissingImports]

    original = fsdp_utils.apply_fsdp2
    if getattr(original, "__platoon_fsdp_set_patch__", False):
        return

    @wraps(original)
    def apply_fsdp2(model, fsdp_kwargs, wrap_policy):
        if wrap_policy is not None:
            wrap_policy.transformer_layer_cls_to_wrap = _normalize_fsdp_wrap_classes(
                wrap_policy.transformer_layer_cls_to_wrap
            )

        no_split_modules = getattr(model, "_no_split_modules", None)
        if isinstance(no_split_modules, (set, tuple)):
            model._no_split_modules = list(no_split_modules)

        return original(model, fsdp_kwargs, wrap_policy)

    apply_fsdp2.__platoon_fsdp_set_patch__ = True
    fsdp_utils.apply_fsdp2 = apply_fsdp2


def apply_all_patches() -> None:
    """Apply Platoon compatibility patches for the current AReaL release."""

    _prepend_nvidia_cuda_library_paths()
    _patch_hf_tokenizer_download_race()
    _patch_model_response_custom_stop_sequences()
    _patch_batch_task_dispatcher_idle_submit()
    _patch_local_scheduler_fork_ready_timeout()
    _patch_remote_inf_engine_proxy_resolution()
    _patch_transformers_apply_chat_template_return_type()
    _patch_areal_openai_message_content_flatten()
    _patch_areal_proxy_rollout_fork_command()
    _patch_areal_fsdp_wrap_classes_set_compat()
