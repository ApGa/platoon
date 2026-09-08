# Adapted and modified from: https://github.com/microsoft/agent-lightning/blob/main/examples/tinker/agl_tinker/llm.py

from __future__ import annotations

import inspect
import logging
import os
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Sequence,
    Type,
    TypeGuard,
    TypeVar,
    cast,
    get_origin,
)

import litellm
import tinker
import tinker.types
import tinker.types.sample_response
import tinker.types.sampled_sequence
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    ChatCompletionTokenLogprob,
    Choices,
    ModelResponse,
    Usage,
)
from litellm.types.utils import ChoiceLogprobs as LitellmChoiceLogprobs
from litellm.types.utils import Message as LitellmMessage
from litellm.types.utils import TopLogprob as LitellmTopLogprob
from litellm.utils import custom_llm_setup
from pydantic import ConfigDict
from tinker.types import ModelInput, SamplingParams
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.renderers import Message as TinkerMessage
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.renderers import ToolCall as TinkerToolCall
from tinker_cookbook.tokenizer_utils import get_tokenizer
from transformers import AutoProcessor, PreTrainedTokenizer

from platoon.train.tinker.renderers import register as register_platoon_renderers
from platoon.train.tinker.rollout_metrics import RolloutTimeline, record_model_call, timeline_scope

logger = logging.getLogger(__name__)

# A prompt that leaves fewer completion tokens than this in the context window is
# reported as a context-window error (hard condensation) instead of being sampled.
MIN_CONTEXT_HEADROOM_TOKENS = 1

T = TypeVar("T")

# Render kwargs the proxy forwards from a request's ``chat_template_kwargs`` (litellm
# ``optional_params["extra_body"]["chat_template_kwargs"]`` or top-level
# ``optional_params["chat_template_kwargs"]``) into ``renderer.build_generation_prompt``.
CHAT_TEMPLATE_RENDER_KWARGS = ("enable_thinking", "preserve_thinking")

# The condenser is the only caller with this completion budget (``condenser_max_completion_tokens``);
# its optional_params keys are logged once so the chat_template_kwargs plumbing is visible in client.log.
CONDENSER_MAX_COMPLETION_TOKENS = 26214


def _patch_tinker_sample_response_extra_allow() -> None:
    """Rebind ``tinker.types.SampleResponse``/``SampledSequence`` to ``extra="allow"`` subclasses.

    tinker 0.16.1 response models use ``extra="ignore"`` (``tinker/_models.py``), which drops the
    gateway's per-sequence ``routing_id`` (contract C1).  ``SamplingClient._sample_async_impl`` resolves
    ``types.SampleResponse`` at call time, so rebinding the module attributes before any sampling client
    exists is enough: ``getattr(seq, "routing_id", None)`` then works on every sampled sequence.
    Idempotent.  Subprocess sampling (``TINKER_SUBPROCESS_SAMPLING``) would run the sidecar without this
    patch, so it must be unset.
    """
    subprocess_sampling = os.environ.get("TINKER_SUBPROCESS_SAMPLING", "").strip().lower()
    assert subprocess_sampling in ("", "0", "false", "no"), (
        "TINKER_SUBPROCESS_SAMPLING must be unset: the sampling sidecar would not carry the extra='allow' "
        "SampleResponse patch and routing_id (R3) would be dropped"
    )
    if getattr(tinker.types.SampleResponse, "_platoon_extra_allow", False):
        return

    base_sequence = tinker.types.sampled_sequence.SampledSequence
    base_response = tinker.types.sample_response.SampleResponse

    class SampledSequence(base_sequence):  # type: ignore[misc,valid-type]
        model_config = ConfigDict(frozen=True, extra="allow")
        _platoon_extra_allow = True

    class SampleResponse(base_response):  # type: ignore[misc,valid-type]
        model_config = ConfigDict(frozen=True, extra="allow")
        _platoon_extra_allow = True
        sequences: Sequence[SampledSequence]  # type: ignore[assignment]

    SampledSequence.__name__ = SampledSequence.__qualname__ = base_sequence.__name__
    SampleResponse.__name__ = SampleResponse.__qualname__ = base_response.__name__
    for module, name, cls in (
        (tinker.types.sampled_sequence, "SampledSequence", SampledSequence),
        (tinker.types, "SampledSequence", SampledSequence),
        (tinker, "SampledSequence", SampledSequence),
        (tinker.types.sample_response, "SampleResponse", SampleResponse),
        (tinker.types, "SampleResponse", SampleResponse),
        (tinker, "SampleResponse", SampleResponse),
    ):
        if hasattr(module, name):
            setattr(module, name, cls)


_patch_tinker_sample_response_extra_allow()
SampleResponse = tinker.types.SampleResponse

# ``train.renderer_name: qwen3_6_preserve_thinking`` must resolve in every process that calls
# ``get_renderer`` (the trainer, and anything that unpickles a renderer); registration is idempotent.
register_platoon_renderers()


@dataclass
class TinkerLLMInteraction:
    obs: tinker.ModelInput
    action: TokensWithLogprobs
    # Gateway routing handle of the sampled sequence (contract C1); None when the gateway did not send one.
    routing_id: int | None = None


@dataclass(frozen=True)
class RenderRequest:
    """Per-request rendering inputs extracted from litellm's ``optional_params``."""

    tools: list[dict[str, Any]] | None = None
    render_kwargs: dict[str, bool] | None = None


def to_cookbook_tool_specs(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI ``{"type": "function", "function": {name, description, parameters}}`` -> cookbook ``ToolSpec``."""
    specs: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function", tool) if isinstance(tool, Mapping) else tool
        if not isinstance(function, Mapping) or "name" not in function:
            raise ValueError(f"Invalid tool definition: {tool!r}")
        specs.append(
            {
                "name": function["name"],
                "description": function.get("description", "") or "",
                "parameters": function.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return specs


proxy_interactions: ContextVar[dict[str, TinkerLLMInteraction]] = ContextVar("proxy_interactions")


class TinkerLLMProxySession:
    _token: object | None = None
    _timeline_scope: object | None = None

    def __enter__(self) -> TinkerLLMProxySession:
        self._token = proxy_interactions.set({})
        # A rollout tree also gets a timeline, bound the same way and with the
        # same context propagation, so subagents and judges record into their
        # root rollout's timing.
        self._timeline_scope = timeline_scope(RolloutTimeline())
        self.timeline = self._timeline_scope.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._timeline_scope is not None:
            self._timeline_scope.__exit__(exc_type, exc_value, traceback)
            self._timeline_scope = None
        if self._token is not None:
            proxy_interactions.reset(self._token)
            self._token = None

    async def __aenter__(self) -> TinkerLLMProxySession:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        return self.__exit__(exc_type, exc_value, traceback)

    @property
    def interactions(self) -> dict[str, TinkerLLMInteraction]:
        return proxy_interactions.get()


def generate_id(prefix: str) -> str:
    """Generate a unique ID with the given prefix.

    Args:
        prefix: String prefix for the generated ID.

    Returns:
        A unique identifier string.
    """
    return prefix + str(uuid.uuid4())


class TinkerLLM(CustomLLM):
    """LiteLLM provider that proxies Tinker's sampling client.

    Attributes:
        model_name: The HuggingFace model identifier.
        renderer: Prompt renderer for formatting messages.
        tokenizer: Tokenizer for the model.
        sampling_client: Tinker sampling client for generation.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
        top_p: Nucleus sampling parameter.
        seed: Sampling seed forwarded to the gateway (SGLang ``sampling_seed``) when the request
            carries none.  ``None`` (default) leaves the gateway free to sample independently;
            a fixed seed makes all group members identical under deterministic inference.
    """

    def __init__(
        self,
        *,
        model_name: str,
        renderer: Renderer,
        tokenizer: PreTrainedTokenizer,
        sampling_client: tinker.SamplingClient,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = 1.0,
        seed: int | None = None,
        context_window_length: int | None = None,
    ) -> None:
        """Initialize the TinkerLLM."""
        self.model_name = model_name
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.sampling_client = sampling_client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.seed = seed
        self.context_window_length = context_window_length
        self._version: int = 0
        self._build_prompt_params = frozenset(inspect.signature(renderer.build_generation_prompt).parameters)
        self._parse_response_params = frozenset(inspect.signature(renderer.parse_response).parameters)
        self._logged_condenser_params = False

    @property
    def version(self) -> int:
        """Get the current checkpoint version."""
        return self._version

    def set_version(self, version: int) -> None:
        """Set the checkpoint version.

        Args:
            version: The version number to set.
        """
        self._version = version

    def increment_version(self) -> int:
        """Increment and return the checkpoint version.

        Returns:
            The new version number after incrementing.
        """
        self._version += 1
        return self._version

    def update_sampling_client(self, sampling_client: tinker.SamplingClient, increment_version: bool = True) -> None:
        """Update the sampling client used for generation.

        Args:
            sampling_client: New Tinker sampling client to use.
            increment_version: Whether to increment the version after update.
        """
        self.sampling_client = sampling_client
        if increment_version:
            self.increment_version()

    def _canonicalize_messages(self, messages: Any) -> List[TinkerMessage]:
        # Note: We avoid using TypeAdapter for strict validation because TinkerMessage
        # contains ImagePart which has PIL.Image.Image that Pydantic can't handle.
        # Instead, we cast directly since we expect the messages to already be in
        # the correct format (coming from LiteLLM or manually constructed).
        if not isinstance(messages, list):
            raise ValueError(f"Expected list of messages, got {type(messages)}")
        return cast(List[TinkerMessage], messages)

    def _validate_role(self, role: str) -> TypeGuard[Literal["assistant", "user", "system", "tool", "function"]]:
        if role not in ["assistant", "user", "system", "tool", "function"]:
            raise ValueError(f"Invalid role: {role}")
        return True

    def _parse_tool_call(self, tool_call: TinkerToolCall) -> ChatCompletionMessageToolCall:
        return ChatCompletionMessageToolCall(
            id=tool_call.id or generate_id("tinker-tool-call-"),
            function={
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            },
            type="function",
        )

    @staticmethod
    def _split_message_content(content: Any) -> tuple[str, str | None]:
        """tinker-cookbook message content -> (visible text, thinking or None) for LiteLLM."""
        if content is None:
            return "", None
        if isinstance(content, str):
            return content, None
        if isinstance(content, list):
            texts: list[str] = []
            thinks: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type")
                    if part_type == "text":
                        texts.append(str(part.get("text", "")))
                    elif part_type == "thinking":
                        thinks.append(str(part.get("thinking", "")))
                    elif part_type == "image":
                        # LiteLLM's Message model expects plain string content on this path.
                        # Drop image payloads but keep a marker so content is not silently empty.
                        texts.append("[image]")
                    else:
                        texts.append(str(part))
                else:
                    texts.append(str(part))
            return "".join(texts), ("".join(thinks) if thinks else None)
        return str(content), None

    def _get_optional_params(
        self,
        kwargs: Dict[str, Any],
        keys: List[str],
        expected_type: Type[T],
        validate_fn: Callable[[T], bool],
        default_value: T,
    ) -> T:
        optional_params = cast(Dict[str, Any], kwargs.get("optional_params", {}))
        if not isinstance(optional_params, dict):  # type: ignore
            raise ValueError(f"Invalid optional params type: {type(optional_params)}")
        for key in keys:
            if key in optional_params:
                value = optional_params[key]
                # Handle parameterized generics like list[str] by extracting the origin type
                origin = get_origin(expected_type)
                check_type = origin if origin is not None else expected_type
                # Allow int for float params (e.g., top_p=1 instead of top_p=1.0)
                if check_type is float:
                    if not isinstance(value, (int, float)):
                        raise ValueError(f"Invalid {key} type: {type(value)}")
                    value = float(value)  # Convert int to float
                elif not isinstance(value, check_type):
                    raise ValueError(f"Invalid {key} type: {type(value)}")
                if not validate_fn(value):
                    raise ValueError(f"Invalid {key}. Did not pass validation: {value}")
                return value
        return default_value

    def _extract_render_request(self, kwargs: Mapping[str, Any]) -> RenderRequest:
        """Tools and chat-template kwargs from litellm's ``optional_params``.

        litellm delivers OpenHands' native tool calling as ``optional_params["tools"]`` (OpenAI
        ``{"type": "function", "function": {...}}`` dicts) with ``optional_params["tool_choice"]``;
        ``extra_body`` is passed through verbatim for custom providers, and a top-level
        ``chat_template_kwargs`` kwarg lands next to it (verified against the openreward venv's litellm).
        """
        optional_params = cast(Dict[str, Any], kwargs.get("optional_params", {}) or {})
        if not isinstance(optional_params, dict):  # type: ignore[redundant-expr]
            raise ValueError(f"Invalid optional params type: {type(optional_params)}")

        tools = optional_params.get("tools") or None
        if tools is not None and not isinstance(tools, list):
            raise ValueError(f"Invalid tools type: {type(tools)}")
        tool_choice = optional_params.get("tool_choice")
        if tool_choice is not None and tool_choice not in ("auto", "none"):
            raise ValueError(
                f"tool_choice={tool_choice!r} is not supported by the Tinker proxy (only 'auto'/'none'): "
                "forced or named tool calls cannot be enforced by the renderer"
            )

        extra_body = optional_params.get("extra_body") or {}
        if not isinstance(extra_body, dict):
            extra_body = {}
        chat_template_kwargs: dict[str, Any] = {}
        for source in (extra_body.get("chat_template_kwargs"), optional_params.get("chat_template_kwargs")):
            if isinstance(source, dict):
                chat_template_kwargs.update(source)
        render_kwargs: dict[str, bool] = {
            key: bool(chat_template_kwargs[key])
            for key in CHAT_TEMPLATE_RENDER_KWARGS
            if chat_template_kwargs.get(key) is not None
        }
        reasoning_effort = optional_params.get("reasoning_effort", extra_body.get("reasoning_effort"))
        if reasoning_effort == "none":
            render_kwargs["enable_thinking"] = False

        max_tokens = optional_params.get("max_tokens", optional_params.get("max_completion_tokens"))
        if max_tokens == CONDENSER_MAX_COMPLETION_TOKENS and not self._logged_condenser_params:
            self._logged_condenser_params = True
            logger.info(
                "First condenser-sized request (max_tokens=%s): optional_params keys=%s extra_body keys=%s "
                "chat_template_kwargs=%s -> render_kwargs=%s",
                max_tokens,
                sorted(optional_params),
                sorted(extra_body),
                chat_template_kwargs,
                render_kwargs,
            )
        return RenderRequest(tools=tools, render_kwargs=render_kwargs or None)

    def _prepare_model_input(self, **kwargs: Any) -> ModelInput:
        """LiteLLM messages (+ tools / chat_template_kwargs) -> Tinker ModelInput."""
        messages = kwargs.pop("messages", None)
        canonical_messages = self._canonicalize_messages(messages)
        request = self._extract_render_request(kwargs)
        return self._render_prompt(canonical_messages, request)

    def _render_prompt(self, messages: List[TinkerMessage], request: RenderRequest) -> ModelInput:
        render_kwargs: dict[str, Any] = {}
        if request.render_kwargs:
            unsupported = [k for k in request.render_kwargs if k not in self._build_prompt_params]
            if unsupported:
                logger.warning("Renderer %s ignores chat_template_kwargs %s", type(self.renderer).__name__, unsupported)
            render_kwargs.update({k: v for k, v in request.render_kwargs.items() if k in self._build_prompt_params})
        if request.tools:
            if "tools" in self._build_prompt_params:
                # platoon renderers (qwen3_6_preserve_thinking) take the OpenAI tool dicts verbatim.
                render_kwargs["tools"] = request.tools
            else:
                # tinker_cookbook renderers: fold the tool specs into the leading system message.
                messages = list(messages)
                system_prompt = ""
                if messages and messages[0].get("role") == "system":
                    system_prompt, _ = self._split_message_content(messages[0].get("content"))
                    messages = messages[1:]
                prefix = self.renderer.create_conversation_prefix_with_tools(
                    cast(Any, to_cookbook_tool_specs(request.tools)), system_prompt=system_prompt
                )
                messages = list(prefix) + messages
        return self.renderer.build_generation_prompt(messages, **render_kwargs)

    @staticmethod
    def _finish_reason(stop_reason: str, has_tool_calls: bool) -> str:
        if has_tool_calls:
            return "tool_calls"
        return "length" if stop_reason == "length" else "stop"

    def _parse_response(
        self, model_input: ModelInput, response: SampleResponse, *, tools: list[dict[str, Any]] | None = None
    ) -> ModelResponse:
        """Tinker Response -> LiteLLM Response.

        Splits the renderer's parsed message into ``content`` (visible text), ``reasoning_content``
        (thinking) and OpenAI ``tool_calls``; ``finish_reason`` is ``tool_calls`` when a tool call was
        parsed, else ``length``/``stop`` from the sequence's stop reason.  Extracts log probabilities too.
        """
        choices: List[Choices] = []
        completion_token_count = 0
        parse_kwargs: dict[str, Any] = {}
        if tools and "tools" in self._parse_response_params:
            parse_kwargs["tools"] = tools
        for seq in response.sequences:
            completion_token_count += len(seq.tokens)
            if seq.logprobs is not None:
                token_strings: List[str] = self.tokenizer.batch_decode([[token] for token in seq.tokens])  # type: ignore
                # FIXME: This might not be accurate for some corner cases.
                # But it's not actually used in most cases.
                bytes_list: List[List[int]] = [list(token.encode("utf-8")) for token in token_strings]
                logprobs = LitellmChoiceLogprobs(
                    content=[
                        ChatCompletionTokenLogprob(
                            token=token,
                            bytes=bytes,
                            logprob=logprob,
                            # NOTE: This top logprob is fake. It's just used to fool the LiteLLM type validator.
                            top_logprobs=[LitellmTopLogprob(token=token, bytes=bytes, logprob=logprob)],
                        )
                        for token, bytes, logprob in zip(token_strings, bytes_list, seq.logprobs)
                    ]
                )
            else:
                logprobs = None

            parsed_response, parse_success = self.renderer.parse_response(seq.tokens, **parse_kwargs)
            content, thinking = self._split_message_content(parsed_response.get("content"))
            reasoning_content = parsed_response.get("reasoning_content")  # type: ignore[typeddict-item]
            if not isinstance(reasoning_content, str):
                reasoning_content = thinking
            if reasoning_content == "":
                reasoning_content = None
            role = "assistant"
            tool_calls = None
            if parse_success:
                role = parsed_response["role"]
                if not self._validate_role(role):
                    assert False, "This should never happen"
                raw_tool_calls = parsed_response.get("tool_calls", None)
                if raw_tool_calls:
                    tool_calls = [self._parse_tool_call(tool_call) for tool_call in raw_tool_calls]
            # NOTE(yuge): I thought about adding this to make it more robust to empty responses,
            # but later I found it's a configuration error in my renderer. So I think it's better
            # to just log a warning and go with the default path.
            if not content and not tool_calls and not reasoning_content:
                logger.warning("Parsed content is empty. Original response: " + str(response))
            choices.append(
                Choices(
                    message=LitellmMessage(
                        role=role, content=content, reasoning_content=reasoning_content, tool_calls=tool_calls
                    ),
                    finish_reason=self._finish_reason(str(seq.stop_reason), bool(tool_calls)),
                    logprobs=logprobs,
                    token_ids=seq.tokens,
                )
            )
        prompt_token_count = model_input.length
        return ModelResponse(
            id=generate_id("tinker-sampling-"),
            model=self.model_name,
            choices=choices,
            prompt_token_ids=model_input.to_ints(),
            usage=Usage(
                prompt_tokens=prompt_token_count,
                completion_tokens=completion_token_count,
                total_tokens=prompt_token_count + completion_token_count,
            ),
        )

    @staticmethod
    def _routing_id_of(response: SampleResponse) -> int | None:
        """Gateway routing handle of the (single) sampled sequence, or None (contract C1)."""
        if not response.sequences:
            return None
        routing_id = getattr(response.sequences[0], "routing_id", None)
        if routing_id is None:
            return None
        if isinstance(routing_id, bool) or not isinstance(routing_id, int):
            try:
                routing_id = int(routing_id)
            except (TypeError, ValueError):
                logger.warning("Ignoring non-integer routing_id %r", routing_id)
                return None
        return routing_id

    def _record_interaction(
        self, model_input: ModelInput, model_response: ModelResponse, routing_id: int | None = None
    ) -> None:
        assert len(model_response.choices) == 1

        logprobs_content = model_response.choices[0].logprobs.content
        interaction = TinkerLLMInteraction(
            obs=model_input,
            action=TokensWithLogprobs(
                tokens=model_response.choices[0].token_ids,
                maybe_logprobs=[c.logprob for c in logprobs_content] if logprobs_content else [],
            ),
            routing_id=routing_id,
        )
        proxy_interactions.get()[model_response.id] = interaction

    def _fit_max_tokens_to_context(self, model_input: ModelInput, max_completion_tokens: int) -> int:
        """Return the completion budget that fits the context window (AReaL parity).

        The request's ``max_tokens`` is a ceiling, not a target: it is clamped to
        the tokens remaining after the rendered prompt, so the condenser's large
        completion budget no longer rejects a prompt that fits by itself.  When
        the prompt alone leaves less than :data:`MIN_CONTEXT_HEADROOM_TOKENS`,
        this raises ``litellm.ContextWindowExceededError``.  litellm passes that
        class through unchanged; OpenHands classifies it as a context-window
        error (``LLMContextWindowExceedError``), does not retry it, and answers
        with a hard ``CondensationRequest`` instead of five blind retries.
        """
        if self.context_window_length is None:
            return max_completion_tokens
        prompt_length = model_input.length
        headroom = self.context_window_length - prompt_length
        if headroom < MIN_CONTEXT_HEADROOM_TOKENS:
            raise litellm.ContextWindowExceededError(
                message=(
                    f"Context length exceeded: {prompt_length} prompt tokens leave {headroom} of the "
                    f"{self.context_window_length}-token context window (input exceeds the context window)."
                ),
                model=self.model_name,
                llm_provider="platoon-tinker",
            )
        if max_completion_tokens > headroom:
            logger.info(
                f"Clamping max_tokens {max_completion_tokens} -> {headroom}: {prompt_length} prompt tokens "
                f"in a {self.context_window_length}-token context window."
            )
            return headroom
        return max_completion_tokens

    async def acompletion(self, **kwargs: Any) -> ModelResponse:  # type: ignore
        """Main entrypoint for LiteLLM to call."""
        import asyncio
        import time

        # Phase timings for the client-throughput metrics.  The split matters:
        # ``render`` and ``parse`` are pure client CPU (tokenizing and
        # detokenizing a 30k-token conversation), which is exactly the work
        # that serialises on the GIL when many rollouts share an interpreter,
        # while ``sample`` is the gateway round trip.  Seeing which one
        # dominates is the difference between "the engines are slow" and "the
        # client is the bottleneck".
        call_started = time.perf_counter()
        max_tokens = self._get_optional_params(
            kwargs, ["max_completion_tokens", "max_tokens"], int, lambda x: x >= 0, self.max_tokens
        )
        temperature = self._get_optional_params(
            kwargs, ["temperature"], float, lambda x: 0.0 <= x <= 2.0, self.temperature
        )
        top_k = self._get_optional_params(kwargs, ["top_k"], int, lambda x: True, self.top_k)
        top_p = self._get_optional_params(kwargs, ["top_p"], float, lambda x: 0.0 <= x <= 1.0, self.top_p)
        seed = self._get_optional_params(kwargs, ["seed"], int, lambda _: True, self.seed)
        stop_sequences = self._get_optional_params(
            kwargs, ["stop"], list, lambda x: True, self.renderer.get_stop_sequences()
        )
        messages = self._canonicalize_messages(kwargs.pop("messages", None))
        render_request = self._extract_render_request(kwargs)
        model_input = self._render_prompt(messages, render_request)
        max_tokens = self._fit_max_tokens_to_context(model_input, max_tokens)
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            stop=stop_sequences,
        )
        start_time = time.perf_counter()
        render_seconds = start_time - call_started

        # Timeout for sample_async to prevent infinite hangs (10 minutes)
        SAMPLE_TIMEOUT_SECONDS = 600
        try:
            result = await asyncio.wait_for(
                self.sampling_client.sample_async(prompt=model_input, sampling_params=params, num_samples=1),
                timeout=SAMPLE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - start_time
            logger.error(f"sample_async timed out after {elapsed:.1f}s (timeout={SAMPLE_TIMEOUT_SECONDS}s)")
            raise TimeoutError(f"Tinker sample_async timed out after {SAMPLE_TIMEOUT_SECONDS}s")
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.exception(f"sample_async failed after {elapsed:.1f}s: {e}")
            raise
        elapsed = time.perf_counter() - start_time
        if elapsed > 30.0:
            logger.warning(f"sample_async took {elapsed:.1f}s (slow)")
        parse_started = time.perf_counter()
        final_response = self._parse_response(model_input, result, tools=render_request.tools)
        self._record_interaction(model_input, final_response, routing_id=self._routing_id_of(result))
        ended = time.perf_counter()
        record_model_call(
            started_at=call_started,
            render_seconds=render_seconds,
            sample_seconds=elapsed,
            parse_seconds=ended - parse_started,
            ended_at=ended,
        )
        return final_response

    def completion(self, **kwargs: Any) -> ModelResponse:  # type: ignore
        """Main entrypoint for LiteLLM to call."""
        max_tokens = self._get_optional_params(
            kwargs, ["max_completion_tokens", "max_tokens"], int, lambda x: x >= 0, self.max_tokens
        )
        temperature = self._get_optional_params(
            kwargs, ["temperature"], float, lambda x: 0.0 <= x <= 2.0, self.temperature
        )
        top_k = self._get_optional_params(kwargs, ["top_k"], int, lambda x: True, self.top_k)
        top_p = self._get_optional_params(kwargs, ["top_p"], float, lambda x: 0.0 <= x <= 1.0, self.top_p)
        seed = self._get_optional_params(kwargs, ["seed"], int, lambda _: True, self.seed)
        stop_sequences = self._get_optional_params(
            kwargs, ["stop"], list, lambda x: True, self.renderer.get_stop_sequences()
        )
        messages = self._canonicalize_messages(kwargs.pop("messages", None))
        render_request = self._extract_render_request(kwargs)
        model_input = self._render_prompt(messages, render_request)
        max_tokens = self._fit_max_tokens_to_context(model_input, max_tokens)
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            stop=stop_sequences,
        )
        result = self.sampling_client.sample(prompt=model_input, sampling_params=params, num_samples=1)
        final_response = self._parse_response(model_input, result, tools=render_request.tools)
        self._record_interaction(model_input, final_response, routing_id=self._routing_id_of(result))
        return final_response

    def as_model_list(self) -> list[dict]:
        """Generate model configuration for LiteLLM proxy.

        Returns:
            List containing model configuration dict for LiteLLM.
        """
        return [
            {
                "model_name": self.model_name,
                "litellm_params": {
                    "model": f"platoon-tinker/{self.model_name}",
                },
            }
        ]

    def rewrite_litellm_custom_providers(self) -> TinkerLLM:
        """Register this TinkerLLM as a custom provider in LiteLLM.

        !!! warning
            This method modifies the global LiteLLM state, which could interfere with other tests in the
            same process.

        Returns:
            Self for method chaining.
        """
        litellm.custom_provider_map = [{"provider": "platoon-tinker", "custom_handler": self}]
        custom_llm_setup()
        return self


@dataclass
class ModelInfo:
    llm: TinkerLLM
    model_name: str
    base_url: str
    api_key: str
    # Carried so a rollout host that is not the trainer can rebuild an
    # equivalent renderer without reaching into trainer config.  Defaulted so
    # every existing construction keeps working.
    renderer_name: str = ""
    renderer_kwargs: dict[str, Any] = field(default_factory=dict)


def build_tinker_llm(
    *,
    model_name: str,
    renderer_name: str,
    sampling_client: tinker.SamplingClient,
    context_window_length: int | None = None,
    renderer_kwargs: dict[str, Any] | None = None,
    **sampling_defaults: Any,
) -> TinkerLLM:
    """Tokenizer + AutoProcessor + renderer + TinkerLLM, and nothing else.

    Deliberately constructs **no** ``ServiceClient`` and mutates **no** litellm
    state, so a rollout worker can build its own provider without opening a
    second gateway session (a ``ServiceClient`` per worker would mean a session
    per rollout).  ``register_tinker_llm`` is this plus those two side effects.

    The renderer is rebuilt here rather than shipped: ``Renderer.__reduce__``
    carries only ``(renderer_name, model_name, has_image_processor)`` and drops
    the ``renderer_kwargs`` applied below by ``setattr``, so pickling one is a
    latent silent-drift bug.
    """

    tokenizer = get_tokenizer(model_name)
    image_processor = None
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        logger.debug("Could not load AutoProcessor for %s: %s", model_name, e)
    else:
        processor_tokenizer = getattr(processor, "tokenizer", None)
        if processor_tokenizer is not None:
            tokenizer = processor_tokenizer
        image_processor = getattr(processor, "image_processor", None)

    register_platoon_renderers()  # idempotent; makes qwen3_6_preserve_thinking resolvable by get_renderer
    renderer = get_renderer(renderer_name, tokenizer, image_processor=image_processor, model_name=model_name)
    for key, value in (renderer_kwargs or {}).items():
        if not hasattr(renderer, key):
            raise ValueError(f"Renderer '{renderer_name}' does not support renderer_kwargs['{key}']")
        setattr(renderer, key, value)

    return TinkerLLM(
        model_name=model_name,
        sampling_client=sampling_client,
        renderer=renderer,
        tokenizer=tokenizer,
        context_window_length=context_window_length,
        **sampling_defaults,
    )


def register_tinker_llm(
    model_name: str,
    renderer_name: str,
    context_window_length: int | None = None,
    renderer_kwargs: dict[str, Any] | None = None,
) -> ModelInfo:
    """
    Register the TinkerLLMProxy as a custom provider in LiteLLM.

    Args:
        model_name: HuggingFace model identifier (e.g., "Qwen/Qwen3-30B-A3B-Instruct-2507").
        renderer_name: Renderer type for prompt formatting (e.g., "qwen3", "qwen3_instruct").
        context_window_length: Context window length for the model. Defaults to None.
        renderer_kwargs: Optional renderer attribute overrides applied after construction.
    """
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model_name)

    tinker_llm = build_tinker_llm(
        model_name=model_name,
        renderer_name=renderer_name,
        sampling_client=sampling_client,
        context_window_length=context_window_length,
        renderer_kwargs=renderer_kwargs,
    )
    tinker_llm.rewrite_litellm_custom_providers()
    base_url = "None"
    api_key = "None"
    model_name = "platoon-tinker/" + model_name
    return ModelInfo(
        llm=tinker_llm,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        renderer_name=renderer_name,
        renderer_kwargs=dict(renderer_kwargs or {}),
    )
