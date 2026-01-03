# Adapted and modified from: https://github.com/microsoft/agent-lightning/blob/main/examples/tinker/agl_tinker/llm.py

from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Dict, List, Literal, Optional, Type, TypeGuard, TypeVar, cast, get_origin

import litellm
import tinker
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import ChatCompletionMessageToolCall, ChatCompletionTokenLogprob
from litellm.types.utils import ChoiceLogprobs as LitellmChoiceLogprobs
from litellm.types.utils import Choices
from litellm.types.utils import Message as LitellmMessage
from litellm.types.utils import ModelResponse
from litellm.types.utils import TopLogprob as LitellmTopLogprob
from litellm.utils import custom_llm_setup
from pydantic import TypeAdapter
from tinker.types import ModelInput, SampleResponse, SamplingParams
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.renderers import Message as TinkerMessage
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.renderers import ToolCall as TinkerToolCall
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from transformers import PreTrainedTokenizer
from dataclasses import dataclass
from contextvars import ContextVar


logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class TinkerLLMInteraction:
    obs: tinker.ModelInput
    action: TokensWithLogprobs

proxy_interactions: ContextVar[dict[str, TinkerLLMInteraction]] = ContextVar("proxy_interactions")


class TinkerLLMProxySession:
    _token: object | None = None
    
    def __enter__(self) -> TinkerLLMProxySession:
        self._token = proxy_interactions.set({})
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
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
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        *,
        model_name: str,
        renderer: Renderer,
        tokenizer: PreTrainedTokenizer,
        sampling_client: tinker.SamplingClient,
        max_tokens: int = 128000,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = 1.0,
        seed: int = 42,
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
        self._version: int = 0

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

    def _prepare_model_input(self, **kwargs: Any) -> ModelInput:
        """LiteLLM messages -> Tinker ModelInput."""
        messages = kwargs.pop("messages", None)
        canonical_messages = self._canonicalize_messages(messages)
        # TODO: Needs to be updated for latest tinker cookbook version.
        return self.renderer.build_generation_prompt(canonical_messages)

    def _parse_response(self, model_input: ModelInput, response: SampleResponse) -> ModelResponse:
        """Tinker Response -> LiteLLM Response.

        Extract log probabilities as well.
        """
        choices: List[Choices] = []
        for seq in response.sequences:
            if seq.logprobs is not None:
                token_strings: List[str] = self.tokenizer.batch_decode([token] for token in seq.tokens)  # type: ignore
                # FIXME: This might not be accurate for some corner cases.
                # But it's not actually used in most cases.
                bytes_list: List[List[int]] = [list(token.encode("utf-8")) for token in token_strings]
                logprobs = LitellmChoiceLogprobs(
                    content=[
                        ChatCompletionTokenLogprob(
                            token=token,
                            bytes=bytes,
                            logprob=logprob,
                            # NOTE: This top logprob is not the real top logprob. It's just used to fool the LiteLLM type validator.
                            top_logprobs=[LitellmTopLogprob(token=token, bytes=bytes, logprob=logprob)],
                        )
                        for token, bytes, logprob in zip(token_strings, bytes_list, seq.logprobs)
                    ]
                )
            else:
                logprobs = None

            parsed_response, parse_success = self.renderer.parse_response(seq.tokens)
            if parse_success:
                role = parsed_response["role"]
                if not self._validate_role(role):
                    assert False, "This should never happen"
                # FIXME: The content should not be still there if tool call has been parsed.
                content = parsed_response["content"]
                # NOTE(yuge): I thought about adding this to make it more robust to empty responses,
                # but later I found it's a configuration error in my renderer. So I think it's better
                # to just log a warning and go with the default path.
                # if not content:
                #     raise ValueError("Parsed content is empty. Original response: " + str(response))
                if not content:
                    logger.warning("Parsed content is empty. Original response: " + str(response))
                tool_calls = parsed_response.get("tool_calls", None)
                if tool_calls:
                    tool_calls = [self._parse_tool_call(tool_call) for tool_call in tool_calls]
                choices.append(
                    Choices(
                        message=LitellmMessage(role=role, content=content, tool_calls=tool_calls),
                        finish_reason=seq.stop_reason,
                        logprobs=logprobs,
                        token_ids=seq.tokens,
                    )
                )
            else:
                #logger.warning(f"Failed to parse response: {parsed_response}")
                # Go with the default path
                choices.append(
                    Choices(
                        message=LitellmMessage(role="assistant", content=parsed_response["content"]),
                        finish_reason=seq.stop_reason,
                        logprobs=logprobs,
                        token_ids=seq.tokens,
                    )
                )
        return ModelResponse(
            id=generate_id("tinker-sampling-"), choices=choices, prompt_token_ids=model_input.to_ints()
        )

    def _record_interaction(self, model_input: ModelInput, model_response: ModelResponse) -> None:
        assert len(model_response.choices) == 1

        interaction = TinkerLLMInteraction(
            obs=model_input,
            action=TokensWithLogprobs(
                tokens=model_response.choices[0].token_ids,
                maybe_logprobs=[c.logprob for c in model_response.choices[0].logprobs.content],
            ),
        )
        proxy_interactions.get()[model_response.id] = interaction

    async def acompletion(self, **kwargs: Any) -> ModelResponse:  # type: ignore
        """Main entrypoint for LiteLLM to call."""
        import time
        import asyncio
        max_tokens = self._get_optional_params(
            kwargs, ["max_completion_tokens", "max_tokens"], int, lambda x: x >= 0, self.max_tokens
        )
        temperature = self._get_optional_params(
            kwargs, ["temperature"], float, lambda x: 0.0 <= x <= 2.0, self.temperature
        )
        top_k = self._get_optional_params(kwargs, ["top_k"], int, lambda x: True, self.top_k)
        top_p = self._get_optional_params(kwargs, ["top_p"], float, lambda x: 0.0 <= x <= 1.0, self.top_p)
        seed = self._get_optional_params(kwargs, ["seed"], int, lambda _: True, self.seed)
        stop_sequences = self._get_optional_params(kwargs, ["stop"], list, lambda x: True, self.renderer.get_stop_sequences())
        model_input = self._prepare_model_input(**kwargs)
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            stop=stop_sequences,
        )
        start_time = time.perf_counter()
        
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
        final_response = self._parse_response(model_input, result)
        self._record_interaction(model_input, final_response)
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
        stop_sequences = self._get_optional_params(kwargs, ["stop"], list, lambda x: True, self.renderer.get_stop_sequences())
        model_input = self._prepare_model_input(**kwargs)
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            stop=stop_sequences,
        )
        result = self.sampling_client.sample(prompt=model_input, sampling_params=params, num_samples=1)
        final_response = self._parse_response(model_input, result)
        self._record_interaction(model_input, final_response)
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

def register_tinker_llm(
    model_name: str,
    renderer_name: str,
) -> ModelInfo:
    """
    Register the TinkerLLMProxy as a custom provider in LiteLLM.

    Args:
        model_name: HuggingFace model identifier (e.g., "Qwen/Qwen3-30B-A3B-Instruct-2507").
        renderer_name: Renderer type for prompt formatting (e.g., "qwen3", "qwen3_instruct").
        port: Port to expose the LiteLLM proxy. Defaults to 1899.
        store: Optional Lightning store for tracking usage. Defaults to None.
        add_return_token_ids: Whether to add return token ids to the response. Defaults to True.
    """
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model_name)

    tokenizer = get_tokenizer(model_name)
    tinker_llm = TinkerLLM(
        model_name=model_name,
        sampling_client=sampling_client,
        renderer=get_renderer(renderer_name, tokenizer),
        tokenizer=tokenizer,
    )
    tinker_llm.rewrite_litellm_custom_providers()
    base_url = 'None'
    api_key = 'None'
    model_name = 'platoon-tinker/' + model_name
    return ModelInfo(llm=tinker_llm, model_name=model_name, base_url=base_url, api_key=api_key)

