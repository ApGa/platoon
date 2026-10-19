# Adapted and modified from: https://github.com/microsoft/agent-lightning/blob/main/examples/tinker/agl_tinker/llm.py
"""The policy handle shared by the trainer, rollout workers and inference: a model gateway and its version.

``TinkerLLM`` owns this process's :class:`platoon.gateway.core.ModelGateway` (renderer, tokenizer, sampling
client and sampling defaults) and the checkpoint version the trainer advances. ``set_as_default`` makes that
gateway the one harness rollouts serve (``platoon.gateway.default_gateway``), and ``TinkerLLMProxySession``
binds the interaction map and timeline that one rollout's model calls are recorded into, which the rollout
worker converts into training data.
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import tinker
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from transformers import AutoProcessor, PreTrainedTokenizer

from platoon.gateway.core import (
    ModelGateway,
    Sampler,
    TinkerLLMInteraction,
    set_default_gateway,
    to_cookbook_tool_specs,  # noqa: F401  (re-exported: callers import it from here)
)
from platoon.gateway.core import (  # noqa: F401  (re-exported: callers import it from here)
    patch_tinker_sample_response_extra_allow as _patch_tinker_sample_response_extra_allow,
)
from platoon.train.tinker.key_refusal import key_refusals
from platoon.train.tinker.renderers import RENDERER_FACTORIES, tool_call_parsing
from platoon.train.tinker.renderers import register as register_platoon_renderers
from platoon.train.tinker.rollout_metrics import RolloutTimeline, timeline_scope

logger = logging.getLogger(__name__)

# ``train.renderer_name: qwen3_6_preserve_thinking`` must resolve in every process that calls
# ``get_renderer`` (the trainer, and anything that unpickles a renderer); registration is idempotent.
register_platoon_renderers()


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


class TinkerLLM:
    """The policy: a model gateway over a Tinker sampling client, and the checkpoint version it samples.

    Attributes:
        gateway: Renders, samples and parses every call.  The model name, renderer, tokenizer, sampling client,
            sampling defaults and sample timeout below read through to it; see :class:`ModelGateway` for the
            semantics of ``seed`` and of the sample timeout (``sample_timeout_seconds``, else
            ``TINKER_SAMPLE_TIMEOUT_SECONDS``).
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
        sample_timeout_seconds: float | None = None,
    ) -> None:
        """Initialize the TinkerLLM."""
        self.gateway = ModelGateway(
            sampler=sampling_client,
            renderer=renderer,
            tokenizer=tokenizer,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            context_window_length=context_window_length,
            sample_timeout_seconds=sample_timeout_seconds,
        )
        self._version: int = 0

    @property
    def model_name(self) -> str:
        return self.gateway.model_name

    @property
    def renderer(self) -> Renderer:
        return self.gateway.renderer

    @property
    def tokenizer(self) -> Tokenizer:
        return self.gateway.tokenizer

    @property
    def sampling_client(self) -> Sampler:
        return self.gateway.sampler

    @property
    def max_tokens(self) -> int:
        return self.gateway.max_tokens

    @property
    def temperature(self) -> float:
        return self.gateway.temperature

    @property
    def top_k(self) -> int:
        return self.gateway.top_k

    @property
    def top_p(self) -> float:
        return self.gateway.top_p

    @property
    def seed(self) -> int | None:
        return self.gateway.seed

    @property
    def context_window_length(self) -> int | None:
        return self.gateway.context_window_length

    @property
    def sample_timeout_seconds(self) -> float:
        return self.gateway.sample_timeout_seconds

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
        self.gateway.sampler = sampling_client
        if increment_version:
            self.increment_version()

    def set_as_default(self) -> TinkerLLM:
        """Make this policy's gateway the process default, which ``harness_rollout`` serves to harnesses.

        This sets process-global state (``platoon.gateway.default_gateway``); tests that call it should
        reset it with ``platoon.gateway.set_default_gateway(None)``.

        Returns:
            Self for method chaining.
        """
        set_default_gateway(self.gateway)
        return self


@dataclass
class ModelInfo:
    llm: TinkerLLM
    model_name: str
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
    sample_timeout_seconds: float | None = None,
    **sampling_defaults: Any,
) -> TinkerLLM:
    """Tokenizer + AutoProcessor + renderer + TinkerLLM, and nothing else.

    Deliberately constructs **no** ``ServiceClient`` and sets **no** default
    gateway, so a rollout worker can build its own policy without opening a
    second gateway session (a ``ServiceClient`` per worker would mean a session
    per rollout).  ``register_tinker_llm`` is this plus those two side effects.

    The renderer is rebuilt here rather than shipped: ``Renderer.__reduce__``
    carries only ``(renderer_name, model_name, has_image_processor)`` and drops
    the ``renderer_kwargs`` applied below by ``setattr``, so pickling one is a
    latent silent-drift bug.  A chat-template renderer chooses its tool-call
    parser here, so a template that no parser reads fails at start
    (``ToolCallParserError``) rather than at the first tool call.
    """

    tokenizer, renderer = build_renderer(model_name, renderer_name, renderer_kwargs)
    return TinkerLLM(
        model_name=model_name,
        sampling_client=sampling_client,
        renderer=renderer,
        tokenizer=tokenizer,
        context_window_length=context_window_length,
        sample_timeout_seconds=sample_timeout_seconds,
        **sampling_defaults,
    )


#: Tokenizers and renderers built ahead by :func:`prepare_renderer`, by what they were built from; each is taken
#: once, by the first :func:`build_renderer` that asks for the same.
_prepared: dict[tuple[str, str, str], tuple[Tokenizer, Renderer]] = {}


def prepare_renderer(model_name: str, renderer_name: str, renderer_kwargs: dict[str, Any] | None = None) -> None:
    """Build a tokenizer and renderer ahead, for the next :func:`build_renderer` asking for the same to take.

    A rollout worker fork server (``worker_start_method: forkserver``, :mod:`platoon.utils.forkserver`) calls it
    once, so that each worker it forks takes its own copy of them (copy-on-write) instead of spending seconds of CPU
    building the very same objects from the same files; the server itself never uses them."""

    key = _renderer_key(model_name, renderer_name, renderer_kwargs)
    _prepared[key] = _build_renderer(model_name, renderer_name, renderer_kwargs)


def build_renderer(
    model_name: str, renderer_name: str, renderer_kwargs: dict[str, Any] | None = None
) -> tuple[Tokenizer, Renderer]:
    """The tokenizer and the renderer :func:`build_tinker_llm` uses: AutoProcessor's tokenizer and image processor
    when it loads, the renderer built by name with ``renderer_kwargs`` set as attributes, and a chat-template
    renderer's tool-call parser chosen.  The ones :func:`prepare_renderer` built ahead for the same arguments, in
    this process or the one it was forked from, are taken instead, once."""

    prepared = _prepared.pop(_renderer_key(model_name, renderer_name, renderer_kwargs), None)
    if prepared is not None:
        return prepared
    return _build_renderer(model_name, renderer_name, renderer_kwargs)


def _renderer_key(model_name: str, renderer_name: str, renderer_kwargs: dict[str, Any] | None) -> tuple[str, str, str]:
    return model_name, renderer_name, json.dumps(renderer_kwargs or {}, sort_keys=True, default=repr)


def _build_renderer(
    model_name: str, renderer_name: str, renderer_kwargs: dict[str, Any] | None
) -> tuple[Tokenizer, Renderer]:
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
    tool_call_parsing(renderer)
    return tokenizer, renderer


def tool_call_parsing_record(
    model_name: str, renderer_name: str, renderer_kwargs: dict[str, Any] | None = None
) -> dict[str, Any]:
    """How the policy's responses are parsed, as ``lock.json`` records it: the model and renderer, and for a
    chat-template renderer the tool-call parser it chose and why (``ParserChoice.record()``).  Builds the renderer
    as :func:`build_tinker_llm` does, so it raises the same errors."""

    record: dict[str, Any] = {"model_name": model_name, "renderer_name": renderer_name}
    if renderer_name not in RENDERER_FACTORIES:
        return {**record, "parser": None, "reason": "the renderer parses tool calls itself"}
    _, renderer = build_renderer(model_name, renderer_name, renderer_kwargs)
    choice = tool_call_parsing(renderer)
    return {**record, **(choice.record() if choice is not None else {})}


def register_tinker_llm(
    model_name: str,
    renderer_name: str,
    context_window_length: int | None = None,
    renderer_kwargs: dict[str, Any] | None = None,
    *,
    model_path: str | None = None,
    base_url: str | None = None,
    sample_timeout_seconds: float | None = None,
) -> ModelInfo:
    """Sample ``model_name`` (or the Tinker checkpoint ``model_path``) and make it this process's default gateway.

    Args:
        model_name: HuggingFace model identifier (e.g., "Qwen/Qwen3-30B-A3B-Instruct-2507").
        renderer_name: Renderer type for prompt formatting (e.g., "qwen3", "qwen3_instruct").
        context_window_length: Context window length for the model. Defaults to None.
        renderer_kwargs: Optional renderer attribute overrides applied after construction.
        model_path: Optional ``tinker://`` sampler checkpoint to sample instead of the base model.
        base_url: The Tinker server to sample from (a training config's ``tinker_base_url``). None leaves it to
            ``tinker.ServiceClient``: ``TINKER_BASE_URL``, else the hosted service.
        sample_timeout_seconds: The bound on one sampling call (a config's ``sample_timeout_seconds``). None leaves
            it to the gateway: ``TINKER_SAMPLE_TIMEOUT_SECONDS``, else 600.

    A ``TINKER_API_KEY`` that the server refuses raises :class:`~platoon.eval.config_defs.TinkerKeyRefusedError`, a
    configuration error that names the key and the server (:mod:`platoon.train.tinker.key_refusal`).
    """
    # The client opens its session as it is built, so the server must be known here.
    with key_refusals():
        service_client = tinker.ServiceClient(base_url=base_url) if base_url is not None else tinker.ServiceClient()
    with key_refusals(asked=f"a sampling client of {model_path or model_name}"):
        sampling_client = service_client.create_sampling_client(base_model=model_name, model_path=model_path)

    tinker_llm = build_tinker_llm(
        model_name=model_name,
        renderer_name=renderer_name,
        sampling_client=sampling_client,
        context_window_length=context_window_length,
        renderer_kwargs=renderer_kwargs,
        sample_timeout_seconds=sample_timeout_seconds,
    )
    tinker_llm.set_as_default()
    return ModelInfo(
        llm=tinker_llm,
        model_name=model_name,
        renderer_name=renderer_name,
        renderer_kwargs=dict(renderer_kwargs or {}),
    )
