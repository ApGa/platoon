from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import traceback
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
                        "pending_inputs=%s runner_input_queue=%s "
                        "max_queue_size=%s cap_staleness=%s free_runner_slots=%s "
                        "paused=%s stats=%s",
                        batch_size,
                        accepted_cnt,
                        total_attempts,
                        pending_inputs,
                        runner_input_size,
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


def _patch_remote_inf_engine_asyncio_teardown_race() -> None:
    """Run inference-server fan-out coroutines without asyncio's racy teardown.

    ``areal.infra.remote_inf_engine`` issues control-plane requests (pause/
    resume/offload/onload and weight-update fan-outs) through ``uvloop.run``,
    whose ``asyncio.Runner`` teardown calls ``asyncio.all_tasks()``. That
    snapshots a process-global WeakSet shared by every event loop in the
    process; with the workflow executor churning thousands of rollout tasks in
    another thread, the snapshot fails with ``RuntimeError: Set changed size
    during iteration`` even after CPython's 1000 internal retries (bpo-36607;
    only structurally fixed by per-thread task lists in Python 3.14). At
    recursive-workflow concurrency the churn is continuous, so retrying the
    fan-out does not help either - observed 5 consecutive failures over 2.5
    minutes. Instead, replace the module's ``uvloop.run`` with a runner that
    drives a private event loop directly and skips the cancel-all sweep. The
    fan-out coroutines await everything they spawn before returning, so the
    sweep (the only ``all_tasks()`` caller on this path) is dead weight.
    """

    import uvloop  # pyright: ignore[reportMissingImports]
    import areal.infra.remote_inf_engine as remote_inf_engine  # pyright: ignore[reportMissingImports]

    if getattr(remote_inf_engine.uvloop, "__platoon_asyncio_teardown_race_patch__", False):
        return

    class _RaceFreeUvloop:
        """Module-local ``uvloop`` stand-in whose ``run`` skips Runner teardown."""

        __platoon_asyncio_teardown_race_patch__ = True

        @staticmethod
        def run(coro):
            loop = uvloop.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                finally:
                    loop.close()

        def __getattr__(self, name):
            return getattr(uvloop, name)

    remote_inf_engine.uvloop = _RaceFreeUvloop()


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


def _decode_tool_call_arguments(messages: list[dict[str, Any]]) -> None:
    """Decode assistant tool-call ``arguments`` from JSON strings into dicts.

    OpenAI-format messages (e.g. from OpenHands native tool calling) carry
    ``tool_calls[].function.arguments`` as a JSON string per spec. Chat
    templates such as Qwen3 render them with the Jinja ``items`` filter, which
    requires a mapping and otherwise raises ``TypeError: Can only get item
    pairs from a mapping`` once the conversation history contains a tool call.
    Decode the string into a dict in place so the template sees a mapping;
    leave non-JSON or non-object payloads untouched.
    """

    for message in messages:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                continue
            try:
                decoded = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(decoded, dict):
                function["arguments"] = decoded


def _patch_areal_openai_message_content_flatten() -> None:
    """Normalize OpenHands-style messages before HF chat templates and proxy cache.

    Two adjustments are applied to the proxy's incoming messages:

    - Flatten list-shaped text ``content`` blocks to plain strings.
    - Decode tool-call ``arguments`` from JSON strings into dicts so chat
      templates that iterate them as mappings (e.g. Qwen3) do not crash.
    """

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
        _decode_tool_call_arguments(normalized)
        return normalized

    _ensure_message_dict_list_with_flatten.__platoon_message_content_patch__ = True
    client_module._ensure_message_dict_list = _ensure_message_dict_list_with_flatten


# ---------------------------------------------------------------------------
# Process stall instrumentation
# ---------------------------------------------------------------------------
# Workers have wedged in ways that left no post-mortem evidence (e.g. a
# rollout worker stopped answering `pause` RPCs entirely and the run died
# without a single stack trace). The watchdog below makes the next wedge
# self-diagnosing from the worker's own log.

_STALL_WATCHDOG_STARTED = False
_ENGINE_CALL_LOCK = threading.Lock()
_ENGINE_CALL_STATE: dict[str, Any] = {}


def _watchdog_log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[platoon-stall-watchdog pid={os.getpid()}] {timestamp} {message}",
        file=sys.stderr,
        flush=True,
    )


def _dump_all_thread_stacks(reason: str) -> None:
    name_by_ident = {t.ident: t.name for t in threading.enumerate()}
    lines = [f"all thread stacks ({reason}):"]
    for ident, frame in sys._current_frames().items():
        name = name_by_ident.get(ident, "unknown")
        stack = "".join(traceback.format_stack(frame))
        lines.append(f"--- thread {name!r} (ident={ident}) ---\n{stack}")
    _watchdog_log("\n".join(lines))


def _patch_engine_rpc_call_tracking() -> None:
    """Record which engine RPC method is running on the worker engine thread.

    All engine RPCs (including trivial ones like ``pause``) are serialized
    through a single engine thread, so one stuck method makes the whole worker
    unresponsive to the trainer. Tracking the current method lets the stall
    watchdog name the offender and dump its stack instead of the trainer only
    seeing opaque RPC timeouts.
    """

    try:
        import areal.infra.rpc.guard.engine_blueprint as engine_blueprint  # pyright: ignore[reportMissingImports]
    except Exception:
        return

    original = engine_blueprint._submit_to_engine_thread
    if getattr(original, "__platoon_engine_call_tracking__", False):
        return

    @wraps(original)
    def _submit_with_tracking(func_name: str, func, *args: Any, **kwargs: Any) -> Any:
        @wraps(func)
        def _tracked(*func_args: Any, **func_kwargs: Any) -> Any:
            with _ENGINE_CALL_LOCK:
                _ENGINE_CALL_STATE.update(
                    name=func_name,
                    started=time.monotonic(),
                    active=True,
                )
            try:
                return func(*func_args, **func_kwargs)
            finally:
                with _ENGINE_CALL_LOCK:
                    _ENGINE_CALL_STATE["active"] = False

        return original(func_name, _tracked, *args, **kwargs)

    _submit_with_tracking.__platoon_engine_call_tracking__ = True
    engine_blueprint._submit_to_engine_thread = _submit_with_tracking


def _install_process_stall_watchdog() -> None:
    """Start a watchdog that makes process wedges self-diagnosing.

    Installs, in every Platoon AReaL process (trainer, train workers, rollout
    workers, proxy workers):

    - ``SIGUSR1`` -> faulthandler dump of all thread stacks to stderr, for
      on-demand inspection of a live process (``kill -USR1 <pid>``).
    - A dead-man timer re-armed every few seconds by a heartbeat thread. If
      Python threads cannot run for ``PLATOON_STALL_DUMP_SECS`` (default 180s;
      e.g. a stop-the-world GC pause or a GIL-holding native call),
      faulthandler's C watchdog thread dumps all thread stacks to stderr
      without needing the GIL.
    - A post-hoc warning when the heartbeat thread itself was frozen, which
      timestamps GC/GIL stalls even when they end before the dump fires.
    - A warning plus all-thread stack dump when one engine RPC method has been
      running for over ``PLATOON_ENGINE_STALL_SECS`` (default 600s).
    - A warning when open file descriptors exceed 80% of the soft limit
      (leaked sockets exhaust FDs long before the process dies).

    Disable with ``PLATOON_STALL_WATCHDOG=0``.
    """

    global _STALL_WATCHDOG_STARTED

    if os.environ.get("PLATOON_STALL_WATCHDOG", "1") != "1":
        return
    if _STALL_WATCHDOG_STARTED:
        return
    _STALL_WATCHDOG_STARTED = True

    import faulthandler
    import signal

    try:
        # chain=False: with no prior Python handler installed, chaining would
        # fall through to the default action and terminate the process.
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    except Exception:
        pass

    _patch_engine_rpc_call_tracking()

    heartbeat_interval = 5.0
    freeze_warn_slack = 30.0
    freeze_dump_secs = float(os.environ.get("PLATOON_STALL_DUMP_SECS", "180"))
    engine_stall_secs = float(os.environ.get("PLATOON_ENGINE_STALL_SECS", "600"))
    fd_check_period = 60.0
    fd_warn_fraction = 0.8
    fd_warn_cooldown = 300.0

    def _maybe_warn_fd_usage(last_warn: float) -> float:
        try:
            import resource

            soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
            open_fds = len(os.listdir("/proc/self/fd"))
        except Exception:
            return last_warn
        now = time.monotonic()
        if soft_limit > 0 and open_fds > fd_warn_fraction * soft_limit and now - last_warn > fd_warn_cooldown:
            _watchdog_log(
                f"high file descriptor usage: {open_fds}/{soft_limit} open; "
                "leaked sockets can wedge this process before any crash"
            )
            return now
        return last_warn

    def _watchdog_loop() -> None:
        last_fd_check = 0.0
        last_fd_warn = float("-inf")
        last_engine_report = float("-inf")
        while True:
            try:
                faulthandler.dump_traceback_later(freeze_dump_secs, exit=False, file=sys.stderr)
            except Exception:
                pass

            before_sleep = time.monotonic()
            time.sleep(heartbeat_interval)
            now = time.monotonic()

            gap = now - before_sleep
            if gap > heartbeat_interval + freeze_warn_slack:
                _watchdog_log(
                    f"Python threads could not run for {gap:.0f}s "
                    "(stop-the-world GC pause or GIL-holding native call); "
                    f"stalls over {freeze_dump_secs:.0f}s dump all thread stacks via faulthandler"
                )

            with _ENGINE_CALL_LOCK:
                engine_call = dict(_ENGINE_CALL_STATE)
            if engine_call.get("active"):
                elapsed = now - engine_call["started"]
                if elapsed > engine_stall_secs and now - last_engine_report > engine_stall_secs:
                    last_engine_report = now
                    _dump_all_thread_stacks(
                        f"engine RPC method {engine_call['name']!r} has been running for {elapsed:.0f}s; "
                        "all other engine RPCs (e.g. pause) are queued behind it"
                    )

            if now - last_fd_check > fd_check_period:
                last_fd_check = now
                last_fd_warn = _maybe_warn_fd_usage(last_fd_warn)

    threading.Thread(target=_watchdog_loop, daemon=True, name="platoon-stall-watchdog").start()
    _watchdog_log(
        f"started (freeze dump after {freeze_dump_secs:.0f}s, engine RPC stall report after "
        f"{engine_stall_secs:.0f}s); run `kill -USR1 {os.getpid()}` for an on-demand stack dump"
    )


def apply_all_patches() -> None:
    """Apply Platoon compatibility patches for the current AReaL release.

    Two historical patches were dropped when upgrading to AReaL HEAD
    (``a0f3dca``) because upstream now handles those cases natively:

    - ``apply_chat_template`` return-type coercion: ``areal.utils.hf_utils``
      now provides an ``apply_chat_template`` wrapper that normalizes the
      Transformers 5 dict return to ``list[int]``. Re-patching the tokenizer
      method globally would make that wrapper's ``result["input_ids"]`` fail.
    - FSDP wrap-class set/tuple compatibility: ``areal.engine.fsdp_utils``'s
      ``apply_fsdp2`` now normalizes ``_no_split_modules`` and
      ``transformer_layer_cls_to_wrap`` internally.
    """

    _patch_hf_tokenizer_download_race()
    _patch_model_response_custom_stop_sequences()
    _patch_batch_task_dispatcher_idle_submit()
    _patch_local_scheduler_fork_ready_timeout()
    _patch_remote_inf_engine_asyncio_teardown_race()
    _patch_remote_inf_engine_proxy_resolution()
    _patch_areal_openai_message_content_flatten()
    _install_process_stall_watchdog()
