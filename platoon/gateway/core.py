"""Model gateway core: the one render -> sample -> parse -> record implementation.

``ModelGateway`` owns one sampler (a tinker ``SamplingClient``, the rollout worker's
``FollowingSamplingClient``, or a fake with the same ``sample_async``), the renderer and tokenizer, and the
sampling defaults.  Each transport only translates its wire format into a :class:`ChatRequest` and the
:class:`ChatResponse` back:

* in-process function harnesses (``platoon.harness.inline``), which call :meth:`ModelGateway.chat` on their
  agent directly;
* the HTTP server :mod:`platoon.gateway.server`, which authenticates every call by its **agent key** and
  records into that agent.

``open_agent`` mints an agent with a 256-bit key.  ``chat(agent, request)`` stores the call's exact prompt and
sampled tokens as a :class:`TinkerLLMInteraction` under ``agent.interactions[call_id]`` and appends the call to
the agent's ordered call list, from which :mod:`platoon.gateway.trace` builds the rollout's trajectories.

A run that samples several policies serves them through one gateway, the agents' default policy's
(:meth:`ModelGateway.serve_policies`): each other policy has a gateway of its own, its runtime (its sampler,
renderer, tokenizer and sampling defaults), and every agent samples one policy for its whole life, the one its role is
bound to (``AgentSession.policy``), whose runtime renders, samples and parses its calls.

:meth:`ModelGateway.shut_down` stops a gateway for good, for a process that is exiting (a rollout worker told to
stop): it closes every agent, so their model calls and tool-plane calls fail with :class:`AgentClosedError` (HTTP
410), and refuses to open new ones, so no subagent or verifier can start.  It is how a worker stops harnesses that
ignore cancellation, or run in threads of their own, from calling on.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
import secrets
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import tinker
import tinker.types
import tinker.types.sample_response
import tinker.types.sampled_sequence
from pydantic import ConfigDict
from tinker.types import ModelInput, SamplingParams
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.renderers import Message, Renderer, ToolCall
from tinker_cookbook.tokenizer_utils import Tokenizer

from platoon.harness.agent import CALL_KINDS, ToolAccess
from platoon.utils.judges import JUDGE_ROLE, JUDGES_REFUSED_MISC_KEY

if TYPE_CHECKING:
    from platoon.gateway.mcp import AgentTools, ToolCallRecord

logger = logging.getLogger(__name__)

FinishReason = Literal["stop", "length", "tool_calls"]
# How an agent's default sampling (the rollout's inference_params) meets a call's own parameters: ``request`` lets
# the call's win, ``config`` has policy calls sample as the defaults say (RolloutConfig.sampling_precedence,
# SamplingOverrides.for_call).
SamplingPrecedence = Literal["request", "config"]
SAMPLING_PRECEDENCES: tuple[str, ...] = ("request", "config")
MessageRole = Literal["assistant", "user", "system", "tool", "function"]

# A prompt that leaves fewer completion tokens than this in the context window is
# reported as a context-window error (hard condensation) instead of being sampled.
MIN_CONTEXT_HEADROOM_TOKENS = 1

# Render kwargs a request may forward from its ``chat_template_kwargs`` (top level, or inside
# ``extra_body``) into ``renderer.build_generation_prompt``.
CHAT_TEMPLATE_RENDER_KWARGS = ("enable_thinking", "preserve_thinking")

# ``sample_async`` round trips slower than this are logged.
SLOW_SAMPLE_SECONDS = 30.0

# The bound on one ``sample_async`` round trip when a policy's config sets none (``train.sample_timeout_seconds``,
# ``inference.sample_timeout_seconds``): the variable SAMPLE_TIMEOUT_ENV names, else DEFAULT_SAMPLE_TIMEOUT_SECONDS
# (resolve_sample_timeout).
SAMPLE_TIMEOUT_ENV = "TINKER_SAMPLE_TIMEOUT_SECONDS"
DEFAULT_SAMPLE_TIMEOUT_SECONDS = 600.0

# Unparsed tool calls a gateway warns about one by one before it warns only every thousandth.
_UNPARSED_WARNINGS = 3

# Keys of closed agents are remembered, so late calls get 410 rather than 401, up to this many.
MAX_CLOSED_SECRETS = 65536

# The gateway step budget (``AgentSession.call_cap``): the answer to a policy call past it, an assistant message
# without tool calls that ends a harness's loop; and how many of those an agent gets before further calls are
# refused, so a harness that ignores the stop fails fast instead of looping until the rollout's timeout.
BUDGET_STOP_MESSAGE = (
    "[platoon] The step budget of {max_steps} model calls is used up: no turns are left. "
    "Stop here; the task is graded as it stands."
)
MAX_BUDGET_STOPS = 8
# ``AgentSession.misc`` key (and so trajectory misc key) of an agent's gateway step budget.
BUDGET_MISC_KEY = "budget"
# Agents one rollout may mint over HTTP (``POST /v1/platoon/agents``): model code in a sandbox can read an agent key,
# so minting is capped.  ``PLATOON_MAX_MINTED_AGENTS`` sets another default for a process.
DEFAULT_MAX_MINTED_AGENTS = 64
MAX_MINTED_AGENTS_ENV = "PLATOON_MAX_MINTED_AGENTS"

_MESSAGE_ROLES: tuple[MessageRole, ...] = ("assistant", "user", "system", "tool", "function")


def patch_tinker_sample_response_extra_allow() -> None:
    """Rebind ``tinker.types.SampleResponse``/``SampledSequence`` to ``extra="allow"`` subclasses.

    tinker 0.16.1 response models use ``extra="ignore"`` (``tinker/_models.py``), which drops the per-sequence
    ``routing_id`` that a server with rollout routing replay (R3) sends.  ``SamplingClient._sample_async_impl``
    resolves ``types.SampleResponse`` at call time, so rebinding the module attributes before any sampling client
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


patch_tinker_sample_response_extra_allow()


# --------------------------------------------------------------------------- records and errors


@dataclass
class TinkerLLMInteraction:
    obs: tinker.ModelInput
    action: TokensWithLogprobs
    # The server's routing handle for the sampled sequence (R3); None when the server did not send one.
    routing_id: int | None = None
    # The policy that sampled it, in a gateway that serves several (ModelGateway.serve_policies), so that a program
    # that trains on an agent tree's interactions keeps each policy's apart; None in a gateway of one policy.
    policy: str | None = None
    # The policy version the call sampled (``ChatResponse.version``); None when its gateway does not know it.
    version: int | None = None


class GatewayError(Exception):
    """A model call the gateway refuses; transports map each subclass to their native error."""


class ContextWindowExceededError(GatewayError):
    """The rendered prompt leaves less than ``MIN_CONTEXT_HEADROOM_TOKENS`` of the context window."""

    def __init__(self, *, prompt_tokens: int, context_window: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.context_window = context_window
        super().__init__(
            f"Context length exceeded: {prompt_tokens} prompt tokens leave {context_window - prompt_tokens} of the "
            f"{context_window}-token context window (input exceeds the context window)."
        )


class JudgesOffError(GatewayError):
    """A judge refused because the run's judges are off (``inference.judges: false``): a ``judge`` call, or a
    ``judge``-role agent.  The refusal is counted in the refusing agent's misc (:data:`JUDGES_REFUSED_MISC_KEY`),
    so an evaluation can stop at the first one; transports answer it as a permission error (HTTP 403)."""


class HostedCallError(GatewayError):
    """A hosted model's call that failed after its retries (:mod:`platoon.gateway.hosted`): no answer in time
    (``kind`` ``timeout``) or a server error (``server``).  Transports answer it as their timeout or server error, and
    an evaluation errors the rollout that made it (``backend_unavailable``) rather than score it."""

    def __init__(
        self,
        message: str,
        *,
        kind: Literal["timeout", "server"] = "server",
        status: int | None = None,
        request_id: str | None = None,
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.request_id = request_id
        self.attempts = attempts


class HostedRefusedError(HostedCallError):
    """The provider refused a hosted call's key (401), payment (402), permission (403), or model or route (404):
    every call of the run would fail the same way, so an evaluation stops."""


class HostedRequestError(GatewayError, ValueError):
    """The provider refused a hosted call's request as invalid (a 4xx reply other than a context-length refusal):
    the harness's request, not an outage."""

    def __init__(self, message: str, *, status: int | None = None, request_id: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.request_id = request_id


class ProviderContextWindowError(ContextWindowExceededError):
    """The provider refused a hosted call's prompt as longer than the model's context window, which the agent counts
    as a context overflow, as with Tinker.  The token counts are the provider's, or 0 when its message gives none;
    ``provider_message`` is its wording."""

    def __init__(
        self,
        provider_message: str,
        *,
        prompt_tokens: int = 0,
        context_window: int = 0,
        status: int | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(prompt_tokens=prompt_tokens, context_window=context_window)
        self.provider_message = provider_message
        self.status = status
        self.request_id = request_id
        self.args = (f"Context length exceeded: {provider_message}",)


class UnknownAgentError(GatewayError):
    """No open agent has this key or id."""


class AgentClosedError(GatewayError):
    """The agent was closed: its key is revoked and nothing more is recorded into it."""


class GatewayShutDownError(AgentClosedError):
    """The gateway was shut down (:meth:`ModelGateway.shut_down`): every agent is closed and none can open."""


class AgentLimitError(GatewayError, ValueError):
    """A request refused at a limit: a rollout that minted its limit of agents, or an agent that kept calling after
    its step budget was spent.  Transports answer it as an invalid request (HTTP 400)."""


class CallKindRefusedError(GatewayError, ValueError):
    """A call of a kind its agent may not make (``AgentSession.check_kind``): a ``judge`` call of an agent whose key
    code the policy runs can read (``AgentSession.policy_only``), or an ``aux`` call of such an agent while a call cap
    counts its ``policy`` calls.  The HTTP server answers it as a permission error (403)."""


# --------------------------------------------------------------------------- request side


def check_call_kind(kind: str) -> str:
    """``kind`` if it is a call kind (``policy``, ``aux`` or ``judge``); a ValueError otherwise."""
    if kind not in CALL_KINDS:
        raise ValueError(f"Unknown call kind {kind!r}; expected one of {CALL_KINDS}")
    return kind


def generate_id(prefix: str) -> str:
    """A unique id: ``prefix`` followed by a random UUID4."""
    return prefix + str(uuid.uuid4())


def as_messages(messages: Any) -> list[Message]:
    """Accept a request's message list as is.

    Strict validation (pydantic ``TypeAdapter``) is impossible because cookbook messages may hold PIL images,
    and every caller already sends OpenAI-shaped dicts, so only the container is checked.
    """
    if not isinstance(messages, list):
        raise ValueError(f"Expected list of messages, got {type(messages)}")
    return cast(list[Message], messages)


# The fields of a tool that a policy is shown: all a model needs to call it.
TOOL_FUNCTION_FIELDS = ("name", "description", "parameters")


def empty_tool_schema() -> dict[str, Any]:
    """The parameters of a tool that gives none: an object without properties, as the cookbook renderers show it."""

    return {"type": "object", "properties": {}}


def tool_function(definition: Any) -> dict[str, Any] | None:
    """The function a policy is shown of one tool definition: its ``name``, its ``description`` unless it gives none
    (null counts as none), and its ``parameters``, or :func:`empty_tool_schema` when it gives none.  None for a
    definition that is not an object or names no function (no ``name``, or one that is empty or not a string).

    Every other field is left out (``type``, ``strict``, ``defer_loading``, ``cache_control``, a provider's own), since
    a policy is only shown tools to call: nothing constrains, runs or defers them.  The fields kept stay in the
    definition's order, since a chat template may write a tool out as JSON.  Training (:func:`chat_tools`, and the
    Messages and Responses dialects) and the hosted pass-through (:mod:`platoon.gateway.hosted`) both build tools with
    this, so a trained policy and a hosted model are shown the same tools.
    """

    if not isinstance(definition, Mapping):
        return None
    name = definition.get("name")
    if not isinstance(name, str) or not name:
        return None
    function = {
        key: value
        for key, value in definition.items()
        if key in TOOL_FUNCTION_FIELDS and not (key == "description" and value is None)
    }
    if not function.get("parameters"):
        function["parameters"] = empty_tool_schema()
    return function


def chat_tools(tools: Any) -> list[dict[str, Any]] | None:
    """The tools a policy is shown of a Chat Completions request's ``tools``, in order; None for none.

    Every entry, whatever its ``type``, is shown as a plain function, ``{"type": "function", "function": {...}}``, of
    the name, description and parameters it gives in its ``function``, or else at its top level
    (:func:`tool_function`).  So a provider's built-in tool or an Anthropic server tool is a function of its name that
    nothing runs, and no ``strict`` schema constrains a call.  An entry that already is a function tool keeps the
    order of its ``type`` and ``function`` (:func:`_function_tool`), so a plain function tool is shown as the same JSON
    it came as.  Every renderer is handed these: the chat-template renderer as they are, the cookbook renderers as
    :func:`to_cookbook_tool_specs` reads them.  ``tools`` that is not a list, and an entry that is not an object or
    names no function (``{"type": "browser_search"}``, or a ``custom`` tool, whose name is under ``custom``), are a
    ValueError: no model can call such a tool.
    """

    if not tools:
        return None
    if not isinstance(tools, list):
        raise ValueError(f"Invalid tools type: {type(tools)}")
    shown: list[dict[str, Any]] = []
    for tool in tools:
        function = tool_function(tool.get("function", tool)) if isinstance(tool, Mapping) else None
        if function is None:
            raise ValueError(f"Invalid tool definition: {tool!r}")
        shown.append(_function_tool(tool, function))
    return shown


def _function_tool(tool: Mapping[str, Any], function: dict[str, Any]) -> dict[str, Any]:
    """``{"type": "function", "function": function}`` for the Chat Completions entry ``tool``, with ``function`` first
    when ``tool`` already is a function tool written that way: the OpenAI Go SDK writes its tools so, and so does any
    client whose JSON encoder sorts keys.  A chat template that writes each tool out as JSON (``tool | tojson``, as
    Qwen's do) then renders a plain function tool as it came."""

    if tool.get("type") == "function" and "function" in tool:
        keys = list(tool)
        if keys.index("function") < keys.index("type"):
            return {"function": function, "type": "function"}
    return {"type": "function", "function": function}


def to_cookbook_tool_specs(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chat Completions tools -> cookbook ``ToolSpec``s: each tool's function (:func:`chat_tools`) as its name, its
    description (``""`` when it gives none) and its parameters."""

    specs: list[dict[str, Any]] = []
    for tool in chat_tools(tools) or []:
        function = tool["function"]
        specs.append(
            {
                "name": function["name"],
                "description": function.get("description") or "",
                "parameters": function["parameters"],
            }
        )
    return specs


def to_cookbook_messages(messages: Sequence[Mapping[str, Any]]) -> list[Message]:
    """OpenAI-shaped messages -> the shape tinker_cookbook renderers index directly.

    Cookbook renderers concatenate ``message["content"]``, read tool calls as ``tool_call.function.name``
    attributes, and render a ``tool_calls`` key even when it is empty.  So ``None`` content becomes ``""``,
    tool-call dicts become cookbook ``ToolCall`` objects, and an empty ``tool_calls`` is dropped.
    """
    converted: list[Message] = []
    for message in messages:
        normalized = dict(message)
        if normalized.get("content") is None:
            normalized["content"] = ""
        tool_calls = normalized.pop("tool_calls", None)
        if tool_calls:
            normalized["tool_calls"] = [_to_cookbook_tool_call(tool_call) for tool_call in tool_calls]
        converted.append(cast(Message, normalized))
    return converted


def _to_cookbook_tool_call(tool_call: Any) -> ToolCall:
    if isinstance(tool_call, ToolCall):
        return tool_call
    if isinstance(tool_call, Mapping):
        function = tool_call.get("function")
        if not isinstance(function, Mapping):
            function = {}
        name, arguments, call_id = function.get("name"), function.get("arguments"), tool_call.get("id")
    else:  # litellm / OpenAI SDK objects
        function = getattr(tool_call, "function", None)
        name, arguments = getattr(function, "name", None), getattr(function, "arguments", None)
        call_id = getattr(tool_call, "id", None)
    if not isinstance(name, str) or not name:
        raise ValueError(f"Tool call is missing a function name: {tool_call!r}")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments if arguments is not None else {})
    return ToolCall(id=call_id, function=ToolCall.FunctionBody(name=name, arguments=arguments))


def split_message_content(content: Any) -> tuple[str, str | None]:
    """tinker-cookbook message content -> (visible text, thinking or None)."""
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
                    # Responses carry plain string content: drop image payloads but keep a
                    # marker so content is not silently empty.
                    texts.append("[image]")
                else:
                    texts.append(str(part))
            else:
                texts.append(str(part))
        return "".join(texts), ("".join(thinks) if thinks else None)
    return str(content), None


def _extra_body(params: Mapping[str, Any]) -> dict[str, Any]:
    extra_body = params.get("extra_body") or {}
    return extra_body if isinstance(extra_body, dict) else {}


def chat_template_kwargs(params: Mapping[str, Any]) -> dict[str, Any]:
    """``extra_body.chat_template_kwargs`` overlaid with a top-level ``chat_template_kwargs``."""
    merged: dict[str, Any] = {}
    for source in (_extra_body(params).get("chat_template_kwargs"), params.get("chat_template_kwargs")):
        if isinstance(source, dict):
            merged.update(source)
    return merged


def check_tool_choice(tool_choice: Any) -> None:
    """Refuse a ``tool_choice`` other than ``auto`` and ``none`` (None: unset), as a ValueError.

    The renderer can neither force nor name a tool call, so the policy may always call any tool it is given.  The
    training dialects and the hosted pass-through (:mod:`platoon.gateway.hosted`) both refuse with this check.
    """
    if tool_choice is not None and tool_choice not in ("auto", "none"):
        raise ValueError(
            f"tool_choice={tool_choice!r} is not supported by the model gateway (only 'auto'/'none'): "
            "forced or named tool calls cannot be enforced by the renderer"
        )


@dataclass(frozen=True)
class RenderRequest:
    """Per-request rendering inputs: the tools the policy is shown, as OpenAI function tools, and chat-template
    kwargs."""

    tools: list[dict[str, Any]] | None = None
    render_kwargs: dict[str, bool] | None = None

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> RenderRequest:
        """Read tools, ``tool_choice``, ``chat_template_kwargs`` and ``reasoning_effort`` from OpenAI params.

        ``params`` is a Chat Completions body, or the keyword parameters of an inline ``ctx.chat`` call.  The tools
        become the plain functions a policy is shown (:func:`chat_tools`), the same for every renderer and for a
        hosted model; a tool no model can call is refused.  Clients that keep ``extra_body`` as its own key and those
        that merge it into the body (the OpenAI SDK) are both read, and a top-level ``chat_template_kwargs`` overrides
        the one in ``extra_body``.  Only ``reasoning_effort == "none"`` is honoured, as ``enable_thinking=False``.  A
        ``tool_choice`` other than ``auto`` and ``none`` is refused (:func:`check_tool_choice`); those two are
        accepted and not used.
        """
        tools = chat_tools(params.get("tools"))
        check_tool_choice(params.get("tool_choice"))
        template_kwargs = chat_template_kwargs(params)
        render_kwargs: dict[str, bool] = {
            key: bool(template_kwargs[key])
            for key in CHAT_TEMPLATE_RENDER_KWARGS
            if template_kwargs.get(key) is not None
        }
        if params.get("reasoning_effort", _extra_body(params).get("reasoning_effort")) == "none":
            render_kwargs["enable_thinking"] = False
        return cls(tools=tools, render_kwargs=render_kwargs or None)


def _sampling_param(
    params: Mapping[str, Any], keys: Sequence[str], expected_type: type, validate: Callable[[Any], bool]
) -> Any:
    """The first of ``keys`` set in ``params``, type-checked and validated; None when none is set."""
    for key in keys:
        value = params.get(key)
        if value is None:
            continue
        if expected_type is float:
            # Allow int for float params (e.g., top_p=1 instead of top_p=1.0).
            if not isinstance(value, (int, float)):
                raise ValueError(f"Invalid {key} type: {type(value)}")
            value = float(value)
        elif not isinstance(value, expected_type):
            raise ValueError(f"Invalid {key} type: {type(value)}")
        if not validate(value):
            raise ValueError(f"Invalid {key}. Did not pass validation: {value}")
        return value
    return None


@dataclass(frozen=True)
class SamplingOverrides:
    """A request's sampling parameters; None keeps the default (the agent's, then the gateway's).

    A caller's ``stop`` list replaces the renderer's stop sequences rather than extending them.  The same type
    carries an agent's defaults (``AgentSession.sampling_defaults``), which fill what a request leaves unset, or,
    with ``precedence="config"``, decide a policy call's sampling (:meth:`for_call`).
    """

    max_tokens: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    seed: int | None = None
    stop: list[Any] | None = None
    # As an agent's defaults: whether a policy call's own parameters win ("request") or these do ("config").
    precedence: SamplingPrecedence = "request"

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> SamplingOverrides:
        return cls(
            max_tokens=_sampling_param(params, ("max_completion_tokens", "max_tokens"), int, lambda x: x >= 0),
            temperature=_sampling_param(params, ("temperature",), float, lambda x: 0.0 <= x <= 2.0),
            top_k=_sampling_param(params, ("top_k",), int, lambda _: True),
            top_p=_sampling_param(params, ("top_p",), float, lambda x: 0.0 <= x <= 1.0),
            seed=_sampling_param(params, ("seed",), int, lambda _: True),
            stop=_sampling_param(params, ("stop",), list, lambda _: True),
        )

    @classmethod
    def from_inference_params(cls, params: Any, *, precedence: SamplingPrecedence = "request") -> SamplingOverrides:
        """A rollout's default sampling from its ``RolloutConfig.inference_params`` (temperature, top_p, top_k and
        the completion budget): only the parameters the config sets; None, every parameter's default, leaves the
        gateway's.  ``precedence`` is the config's ``sampling_precedence``."""
        if precedence not in SAMPLING_PRECEDENCES:
            raise ValueError(f"sampling_precedence must be one of {SAMPLING_PRECEDENCES}, got {precedence!r}")
        if params is None:
            return cls(precedence=precedence)
        return cls(
            max_tokens=getattr(params, "max_completion_tokens", None),
            temperature=getattr(params, "temperature", None),
            top_k=getattr(params, "top_k", None),
            top_p=getattr(params, "top_p", None),
            precedence=precedence,
        )

    def over(self, defaults: SamplingOverrides | None) -> SamplingOverrides:
        """These parameters, with the ones left unset taken from ``defaults``."""
        if defaults is None:
            return self
        return replace(
            self,
            **{
                f.name: getattr(defaults, f.name) if getattr(self, f.name) is None else getattr(self, f.name)
                for f in fields(self)
                if f.name != "precedence"
            },
        )

    def for_call(self, request: SamplingOverrides, *, kind: str = "policy") -> SamplingOverrides:
        """A call's sampling under these defaults: the call's own parameters over them.

        When these defaults take precedence (``precedence="config"``) and the call is a ``policy`` call, it samples
        with their temperature, top_p and top_k, never its own: one these defaults leave unset stays None, so the
        gateway's default applies.  Its completion budget is theirs when they set one, else its own; its stop
        sequences and seed stay its own, since its harness parses the reply with those stop sequences.  ``aux`` and
        ``judge`` calls keep their own parameters either way, but the gateway samples the ``aux`` calls of an agent
        whose key code the policy runs can read as ``policy`` calls (``AgentSession.sampling_kind``)."""
        merged = request.over(self)
        if self.precedence != "config" or kind != "policy":
            return merged
        return replace(
            merged,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=request.max_tokens if self.max_tokens is None else self.max_tokens,
        )


@dataclass(frozen=True)
class ChatRequest:
    """One chat call: OpenAI-shaped messages plus the request's rendering and sampling overrides.

    ``params`` are the Chat Completions parameters the request was made from (:meth:`from_params`), which a hosted
    gateway sends on with the messages (:mod:`platoon.gateway.hosted`); a renderer reads ``render`` and ``sampling``.
    """

    messages: list[Message]
    render: RenderRequest = field(default_factory=RenderRequest)
    sampling: SamplingOverrides = field(default_factory=SamplingOverrides)
    params: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_params(cls, messages: Any, params: Mapping[str, Any]) -> ChatRequest:
        """A request from Chat Completions parameters (a request body, or ``ctx.chat``'s keyword parameters)."""
        sampling = SamplingOverrides.from_params(params)
        return cls(
            messages=as_messages(messages),
            render=RenderRequest.from_params(params),
            sampling=sampling,
            params=dict(params),
        )


@dataclass(frozen=True)
class PreparedCall:
    """A rendered request whose prompt fits the context window, ready to sample."""

    request: ChatRequest
    prompt: ModelInput
    sampling_params: SamplingParams
    started_at: float
    render_seconds: float


# --------------------------------------------------------------------------- response side


@dataclass(frozen=True)
class CallTiming:
    """``time.perf_counter`` phases of one model call."""

    started_at: float
    render_seconds: float
    sample_seconds: float
    parse_started_at: float
    ended_at: float

    @property
    def parse_seconds(self) -> float:
        return self.ended_at - self.parse_started_at


def routing_id_of(response: tinker.types.SampleResponse) -> int | None:
    """The server's routing handle (R3) for the (single) sampled sequence, or None."""
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


@dataclass(frozen=True)
class ProviderRecord:
    """What a hosted model's call (:mod:`platoon.gateway.hosted`) records in place of token ids.

    ``dialect`` is the API the call was made in (``openai_chat``, ``anthropic_messages`` or ``openai_responses``).
    ``usage`` is what the provider reported, normalized: ``prompt_tokens`` (every input token, cached or not),
    ``completion_tokens`` and ``total_tokens``, plus ``cached_prompt_tokens``, ``cache_write_tokens`` and
    ``reasoning_tokens`` when it reports them; ``usage_reported`` is false when it reported none, as a Chat
    Completions stream does unless it asks for usage.
    """

    dialect: str
    usage: Mapping[str, int]
    usage_reported: bool = True  # false: the provider sent no usage, so ``usage`` is zeros and the call unpriced
    served_model: str | None = None  # the reply's model: a dated snapshot, or a router's choice
    response_id: str | None = None
    request_id: str | None = None
    raw_stop_reason: str | None = None  # the provider's own finish or stop reason
    refusal: bool = False
    attempts: int = 1  # HTTP attempts, retries included
    rate_limited: int = 0  # 429 replies among them
    cost_usd: float | None = None  # ``inference.hosted.pricing`` applied to the usage; None without prices

    def action_misc(self) -> dict[str, Any]:
        """The fields a trajectory step's ``action_misc`` records (``platoon.gateway.trace``)."""

        misc: dict[str, Any] = {
            "dialect": self.dialect,
            "served_model": self.served_model,
            "response_id": self.response_id,
            "request_id": self.request_id,
            "raw_stop_reason": self.raw_stop_reason,
            "attempts": self.attempts,
        }
        if self.cost_usd is not None:
            misc["cost_usd"] = self.cost_usd
        for name in ("refusal", "rate_limited"):
            if getattr(self, name):
                misc[name] = getattr(self, name)
        if not self.usage_reported:
            misc["usage_reported"] = False
        return misc


@dataclass(frozen=True)
class ChatResponse:
    """One sampled and parsed model call.

    ``content`` is the visible text with tool blocks removed and ``reasoning_content`` the thinking span.
    ``tool_calls`` are OpenAI dicts with JSON-string arguments, emitted only when the response parsed and was not
    cut off by the generation limit, so a truncated response never executes tools.  A tool-call block the renderer
    could not parse stays in ``content`` as it was sampled, as inference engines return it, so the harness sees
    what the model wrote; ``parsed_tool_calls``, ``recovered_tool_calls`` (parsed only by a lenient rule, such as a
    missing close tag) and ``unparsed_tool_calls`` count the call blocks of the response.

    A call a hosted model answered (:mod:`platoon.gateway.hosted`) has no prompt or sampled token ids: ``provider``
    holds the provider's record instead, its usage is the provider's, and it never trains (:attr:`trainable`).
    """

    call_id: str
    model: str
    prompt: ModelInput | None
    tokens: list[int]
    logprobs: list[float] | None
    stop_reason: str
    parse_success: bool
    role: MessageRole
    content: str
    reasoning_content: str | None
    tool_calls: list[dict[str, Any]] | None
    parsed_tool_calls: int
    routing_id: int | None
    timing: CallTiming | None = None
    recovered_tool_calls: int = 0
    unparsed_tool_calls: int = 0
    provider: ProviderRecord | None = None
    # The policy that sampled it, in a gateway that serves several (ModelGateway.serve_policies); None in one of one.
    policy: str | None = None
    # The policy version the call sampled (``ModelGateway.version``, or its sampler's own); None when the gateway does
    # not know it, or the call sampled nothing.
    version: int | None = None

    @property
    def finish_reason(self) -> FinishReason:
        if self.tool_calls:
            return "tool_calls"
        return "length" if self.stop_reason == "length" else "stop"

    @property
    def trainable(self) -> bool:
        """Whether the call carries the exact prompt and sampled tokens training needs: a Tinker call does, a
        hosted model's does not."""
        return self.provider is None

    @property
    def prompt_tokens(self) -> int:
        if self.provider is not None:
            return int(self.provider.usage.get("prompt_tokens", 0))
        assert self.prompt is not None
        return self.prompt.length

    @property
    def completion_tokens(self) -> int:
        if self.provider is not None:
            return int(self.provider.usage.get("completion_tokens", 0))
        return len(self.tokens)

    def usage(self) -> dict[str, int]:
        if self.provider is not None:
            return {
                **self.provider.usage,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
            }
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }

    @property
    def generation(self) -> dict[str, Any]:
        """Raw stop/parse diagnostics (``platoon.utils.generation`` adds the finish reason)."""
        return {
            "stop_reason": self.stop_reason,
            "generated_tokens": self.completion_tokens,
            "parse_success": self.parse_success,
            "parsed_tool_calls": self.parsed_tool_calls,
            "emitted_tool_calls": len(self.tool_calls or []),
            "recovered_tool_calls": self.recovered_tool_calls,
            "unparsed_tool_calls": self.unparsed_tool_calls,
        }

    def message(self) -> dict[str, Any]:
        """The assistant message, with ``reasoning_content`` as vLLM and DeepSeek return it."""
        message: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.reasoning_content is not None:
            message["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        return message

    def interaction(self) -> TinkerLLMInteraction:
        """The training record: the exact prompt and the sampled tokens with their logprobs."""
        if self.prompt is None:
            raise TypeError("A hosted model's call holds no prompt or sampled tokens to train on")
        return TinkerLLMInteraction(
            obs=self.prompt,
            action=TokensWithLogprobs(tokens=self.tokens, maybe_logprobs=list(self.logprobs or [])),
            routing_id=self.routing_id,
            policy=self.policy,
            version=self.version,
        )


def _with_unparsed_calls(content: str, unparsed: Sequence[Any]) -> str:
    """``content`` followed by the raw text of the tool-call blocks that did not parse, as engines return it."""
    raw = "\n".join(str(getattr(call, "raw_text", call)) for call in unparsed)
    return f"{content.rstrip()}\n\n{raw}" if content.strip() else raw


def _openai_tool_call(tool_call: ToolCall) -> dict[str, Any]:
    return {
        "id": tool_call.id or generate_id("tinker-tool-call-"),
        "type": "function",
        "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments},
    }


# --------------------------------------------------------------------------- agents


class CallTimeline(Protocol):
    """Receives per-call phase timings, e.g. ``platoon.train.tinker.rollout_metrics.RolloutTimeline``.

    A timeline may also have ``call_started()``, which :meth:`ModelGateway.chat` calls, without arguments, as it issues
    each call, before it samples; it must not raise.
    """

    def record(
        self, *, started_at: float, render_seconds: float, sample_seconds: float, parse_seconds: float, ended_at: float
    ) -> None: ...


@dataclass(eq=False)
class CallTally:
    """The ``policy`` calls charged to one call cap (``AgentSession.call_cap``): those recorded and those sampling.

    An agent shares its tally with every agent minted under it over HTTP (:meth:`ModelGateway.mint_subagent`), so
    its cap bounds the calls of the whole minted tree, however many agents the tree has.
    """

    recorded: int = 0
    in_flight: int = 0


@dataclass(eq=False)
class AgentSession:
    """One agent: the identity its harness calls with, and the ordered record of its model calls.

    ``fork_step`` is the number of model calls the parent agent had made when this agent was opened.  ``tools``,
    ``tool_access`` and ``tool_calls`` are the agent's tool plane (:mod:`platoon.gateway.mcp`): the env tools it
    sees, which children inherit until they are rebound, and the record of its tool calls.  The harness layer
    (``platoon.harness``) keeps the agent's tree position and step budget here, the kind of each call
    (``policy``, ``aux`` or ``judge``), the harness's annotations, how its run ended and the agent's own outcome;
    :mod:`platoon.gateway.trace` reads them all.
    """

    agent_id: str
    rollout_id: str
    trajectory_id: str
    secret: str = field(repr=False)
    interactions: dict[str, TinkerLLMInteraction] = field(repr=False)
    parent_agent_id: str | None = None
    role: str = "policy"
    # The policy the agent samples for its whole life, in a gateway that serves several (ModelGateway.serve_policies):
    # its role's; None in a gateway that serves one.
    policy: str | None = None
    goal: str | None = None
    fork_step: int = 0
    calls: list[ChatResponse] = field(default_factory=list, repr=False)
    timeline: CallTimeline | None = field(default=None, repr=False)
    closed: bool = False
    context_overflows: int = 0  # prompts refused for exceeding the context window
    # Calls the gateway failed for its own reasons: sampling failed or timed out, the sample could not be parsed,
    # or rendering failed for a reason other than the request (on_harness_exit's grade_if_gateway_clean).
    gateway_errors: int = 0
    # Tool plane, set by platoon.gateway.mcp.bind_agent_tools.
    tools: AgentTools | None = field(default=None, repr=False)
    tool_access: ToolAccess = "full"
    tool_calls: list[ToolCallRecord] = field(default_factory=list, repr=False)
    # The rollout's default sampling (RolloutConfig.inference_params), for calls that leave it unset; inherited.
    sampling_defaults: SamplingOverrides | None = field(default=None, repr=False)
    # --- harness layer (platoon.harness)
    depth: int = 0
    max_steps: int | None = None
    call_kinds: dict[str, str] = field(default_factory=dict, repr=False)  # call_id -> kind
    policy_calls: int = 0  # recorded ``policy`` calls: the agent's budget steps
    # The gateway step budget (a spec's ``[budget] enforce = "gateway"``): the most ``policy`` calls the gateway
    # samples for this agent and the agents minted under it over HTTP together, counting those in flight
    # (``call_tally``); HTTP calls past it get ``ModelGateway.budget_stop``.  None: no cap.  Agents it mints over HTTP
    # get the same cap and share its tally.
    call_cap: int | None = None
    # The policy calls charged to ``call_cap``: this agent's and those of every agent minted under it, recorded and in
    # flight.  ``policy_calls`` and ``policy_calls_in_flight`` stay this agent's own.
    call_tally: CallTally = field(default_factory=CallTally, repr=False)
    budget_stops: int = 0  # policy calls answered with the budget stop instead of a sample
    policy_calls_in_flight: int = field(default=0, repr=False)
    # Whether code the policy runs can read the agent's key: set for a harness in a task sandbox.  Its ``judge`` calls,
    # which never train, are refused, and so are its ``aux`` calls while a call cap counts its ``policy`` calls
    # (``call_cap``), since the cap does not count them (CallKindRefusedError, HTTP 403; nothing is sampled for them).
    # Its other ``aux`` calls sample as its ``policy`` calls do (``sampling_kind``).  Agents it mints over HTTP inherit
    # it.
    policy_only: bool = False
    misc: dict[str, Any] = field(default_factory=dict, repr=False)  # -> trajectory misc
    step_misc: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)  # call_id -> step misc
    task: Any = None  # platoon.task.Task the agent works on (children: the forked task)
    result: Any = None  # platoon.harness.agent.AgentResult of the harness run
    outcome: Any = None  # platoon.envs.service.Outcome: this agent's own grade, None to share the episode's
    # Whether judges run (the rollout's ``judges``, inherited): False refuses this agent's ``judge`` calls and
    # ``judge``-role children with JudgesOffError, and turns off the verifiers of its children.
    judges: bool = True

    @property
    def api_key(self) -> str:
        """The agent key, as a harness presents it: ``Authorization: Bearer``, ``x-api-key`` or ``/s/<secret>/``."""
        return self.secret

    @property
    def call_ids(self) -> list[str]:
        return [call.call_id for call in self.calls]

    @property
    def budget_exhausted(self) -> bool:
        """Whether the gateway answered a call of this agent with the budget stop."""
        return self.budget_stops > 0

    def over_call_cap(self) -> bool:
        """Whether the next ``policy`` call would pass :attr:`call_cap`: the calls charged to it (:attr:`call_tally`,
        recorded plus in flight, of this agent and the agents minted under it) have reached it."""
        tally = self.call_tally
        return self.call_cap is not None and tally.recorded + tally.in_flight >= self.call_cap

    def check_kind(self, kind: str) -> None:
        """Raise :class:`CallKindRefusedError` for a call of ``kind`` this agent may not make.  An agent with
        :attr:`policy_only` may make no ``judge`` call, and no ``aux`` call while a call cap counts its ``policy``
        calls (:attr:`call_cap`, which agents minted under a capped one share), since the cap would not count it.
        Without a cap the gateway counts none of the agent's calls, so an ``aux`` call gets it nothing a ``policy``
        call would not: both train alike, and sample alike (:meth:`sampling_kind`).  Any other agent may make every
        kind."""
        if not self.policy_only or kind == "policy":
            return
        if self.call_cap is not None:
            raise CallKindRefusedError(
                f"This agent may make only policy calls, which its step budget counts; its {kind} calls are refused"
            )
        if kind == "judge":
            raise CallKindRefusedError(
                "This agent may make only policy and aux calls, the calls that train; its judge calls are refused"
            )

    def sampling_kind(self, kind: str) -> str:
        """The kind whose sampling a ``kind`` call of this agent gets (:meth:`SamplingOverrides.for_call`): ``policy``
        for an ``aux`` call of an agent with :attr:`policy_only`, since under the rollout's ``sampling_precedence:
        config`` an ``aux`` call keeps its own parameters, and code the policy runs would choose how calls that train
        sample; ``kind`` otherwise."""
        return "policy" if self.policy_only and kind == "aux" else kind

    def call_kind(self, call_id: str) -> str:
        return self.call_kinds.get(call_id, "policy")

    def refuse_judge(self, what: str) -> JudgesOffError:
        """Count a judge this agent asked for while the run's judges are off (``what``: a judge call or a judge
        agent), in its misc under :data:`JUDGES_REFUSED_MISC_KEY`, and return the error that refuses it."""

        refused = self.misc.setdefault(JUDGES_REFUSED_MISC_KEY, {"count": 0, "first": what})
        refused["count"] += 1
        return JudgesOffError(f"{what} was refused: this run's judges are off (inference.judges: false)")

    def record(self, response: ChatResponse, *, kind: str = "policy") -> None:
        if response.trainable:  # a hosted model's call is recorded, but holds nothing to train on
            self.interactions[response.call_id] = response.interaction()
        self.calls.append(response)
        self.call_kinds[response.call_id] = kind
        if kind == "policy":
            self.policy_calls += 1
            self.call_tally.recorded += 1
        timing = response.timing
        if self.timeline is not None and timing is not None:
            self.timeline.record(
                started_at=timing.started_at,
                render_seconds=timing.render_seconds,
                sample_seconds=timing.sample_seconds,
                parse_seconds=timing.parse_seconds,
                ended_at=timing.ended_at,
            )


# --------------------------------------------------------------------------- the gateway


def resolve_sample_timeout(seconds: float | None = None) -> float:
    """The bound in seconds on one sampling call: ``seconds`` (a config's ``sample_timeout_seconds``) when it is set,
    else the variable ``TINKER_SAMPLE_TIMEOUT_SECONDS``, else :data:`DEFAULT_SAMPLE_TIMEOUT_SECONDS` (600).

    The config's key wins over the variable.  The config is the run's record (``config.resolved.yaml``), and a gateway
    job may export the variable for every client it runs, as miles-tinker's presets do; the variable stays the
    fallback of a config that sets no key, so a launch that sets only the variable samples as it did.

    Raises:
        ValueError: The value is not a finite positive number of seconds; the message names where it came from.
    """

    source = "sample_timeout_seconds"
    if seconds is None:
        raw = os.environ.get(SAMPLE_TIMEOUT_ENV)
        if raw is None:
            return DEFAULT_SAMPLE_TIMEOUT_SECONDS
        source = SAMPLE_TIMEOUT_ENV
        try:
            seconds = float(raw)
        except ValueError:
            raise ValueError(f"The sample timeout ({source}) must be a number of seconds, got {raw!r}") from None
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"The sample timeout ({source}) must be finite and positive, got {seconds!r}")
    return float(seconds)


class Sampler(Protocol):
    """What the gateway uses of a tinker ``SamplingClient`` (or ``FollowingSamplingClient``, or a fake)."""

    async def sample_async(
        self, *, prompt: ModelInput, sampling_params: SamplingParams, num_samples: int
    ) -> tinker.types.SampleResponse: ...

    def sample(self, *, prompt: ModelInput, sampling_params: SamplingParams, num_samples: int) -> Any: ...


class PinningSampler(Protocol):
    """A sampler that follows a republished binding, such as a rollout worker's ``FollowingSamplingClient``:
    ``pinned()`` returns the client the next call samples and the policy version that client serves, picked together,
    so that a reload of the binding between the two cannot mislabel the call."""

    def pinned(self) -> tuple[Sampler, int]: ...


class ModelGateway:
    """Renders, samples, parses and records the model calls of one policy.

    Attributes:
        sampler: Read at every call, so a sampler swapped in by the trainer, or one that follows a republished
            binding, takes effect immediately.
        version: The policy version ``sampler`` serves, which each call records (``ChatResponse.version``); the
            policy that owns the gateway keeps it (``TinkerLLM``).  A sampler that follows a republished binding
            reports the version of each client it hands out (:class:`PinningSampler`), which its calls record
            instead.  None (the default): the calls record none.
        seed: Sampling seed forwarded to the backend (SGLang ``sampling_seed``) when the request carries none.
            ``None`` (default) leaves the backend free to sample independently; a fixed seed makes all group
            members identical under deterministic inference.
        sample_timeout_seconds: Bound on one ``sample_async`` round trip: the argument (a config's
            ``sample_timeout_seconds``), else ``TINKER_SAMPLE_TIMEOUT_SECONDS``, else 600
            (:func:`resolve_sample_timeout`).
        max_minted_agents: Agents one rollout may mint over HTTP (:meth:`mint_subagent`); defaults to
            ``PLATOON_MAX_MINTED_AGENTS``, else :data:`DEFAULT_MAX_MINTED_AGENTS` (64).
    """

    def __init__(
        self,
        *,
        sampler: Sampler,
        renderer: Renderer,
        tokenizer: Tokenizer,
        model_name: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = 1.0,
        seed: int | None = None,
        context_window_length: int | None = None,
        sample_timeout_seconds: float | None = None,
        max_minted_agents: int | None = None,
        version: int | None = None,
    ) -> None:
        self.sampler = sampler
        self.version = version
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.seed = seed
        self.context_window_length = context_window_length
        self.sample_timeout_seconds = resolve_sample_timeout(sample_timeout_seconds)
        if max_minted_agents is None:
            max_minted_agents = int(os.environ.get(MAX_MINTED_AGENTS_ENV, DEFAULT_MAX_MINTED_AGENTS))
        if isinstance(max_minted_agents, bool) or max_minted_agents < 0:
            raise ValueError("The limit on minted agents (PLATOON_MAX_MINTED_AGENTS) must be a non-negative integer")
        self.max_minted_agents = max_minted_agents
        self._build_prompt_params = frozenset(inspect.signature(renderer.build_generation_prompt).parameters)
        self._parse_response_params = frozenset(inspect.signature(renderer.parse_response).parameters)
        self._agents: dict[str, AgentSession] = {}
        self._agents_by_id: dict[str, AgentSession] = {}
        self._closed_secrets: dict[str, None] = {}
        self._minted: dict[str, int] = {}  # rollout id -> agents minted over HTTP
        self._shut_down = False
        self._unparsed_calls = 0  # tool-call blocks that did not parse, for the warnings
        # A gateway that serves several policies (serve_policies): the id of the policy it samples, the other
        # policies' gateways by id, and the policy of each role bound to one.  None and empty in a gateway of one.
        self.policy: str | None = None
        self.runtimes: dict[str, ModelGateway] = {}
        self.role_policies: dict[str, str] = {}

    # ------------------------------------------------------------------ policies

    def serve_policies(self, policy: str, *, runtimes: Mapping[str, ModelGateway], roles: Mapping[str, str]) -> None:
        """Serve several policies: this gateway samples ``policy``, the agents' default, which every root agent and
        every agent of a role ``roles`` leaves out samples; each other policy is sampled by its own gateway
        (``runtimes``, by policy id), which renders, samples and parses the calls of the agents whose role ``roles``
        binds to it.  The agents stay this gateway's: it opens, authenticates and records them all."""

        self.policy = policy
        self.runtimes = dict(runtimes)
        self.role_policies = dict(roles)
        for other, runtime in self.runtimes.items():
            runtime.policy = other

    def runtime(self, policy: str | None) -> ModelGateway:
        """The gateway that samples ``policy`` (an agent's, ``AgentSession.policy``): this one for its own policy, and
        for None, the policy of every agent of a gateway that serves one."""

        return self if policy is None or policy == self.policy else self.runtimes[policy]

    def _policy_of(self, role: str, *, root: bool) -> str | None:
        """The policy an agent of ``role`` samples (None in a gateway that serves one): the default for a root
        agent, else its role's."""

        if self.policy is None or root:
            return self.policy
        return self.role_policies.get(role, self.policy)

    # ------------------------------------------------------------------ render

    def render(self, messages: list[Message], request: RenderRequest | None = None) -> ModelInput:
        """Messages (+ tools / chat-template kwargs) -> the prompt tokens.

        Renderers whose ``build_generation_prompt`` takes ``tools`` (platoon's chat-template renderer and its
        Qwen3.6/3.8 aliases) get the OpenAI messages verbatim, and the request's tools as the plain function tools
        ``RenderRequest.from_params`` made of them (:func:`chat_tools`).  tinker_cookbook renderers get cookbook
        messages, with the tool specs folded into the leading system message.  Unsupported chat-template kwargs
        are logged and ignored; the shared renderer is never mutated.
        """
        request = request or RenderRequest()
        render_kwargs: dict[str, Any] = {}
        if request.render_kwargs:
            unsupported = [k for k in request.render_kwargs if k not in self._build_prompt_params]
            if unsupported:
                logger.warning("Renderer %s ignores chat_template_kwargs %s", type(self.renderer).__name__, unsupported)
            render_kwargs.update({k: v for k, v in request.render_kwargs.items() if k in self._build_prompt_params})
        if "tools" in self._build_prompt_params:
            if request.tools:
                render_kwargs["tools"] = request.tools
            return self.renderer.build_generation_prompt(messages, **render_kwargs)
        messages = to_cookbook_messages(messages)
        if request.tools:
            system_prompt = ""
            if messages and messages[0].get("role") == "system":
                system_prompt, _ = split_message_content(messages[0].get("content"))
                messages = messages[1:]
            prefix = self.renderer.create_conversation_prefix_with_tools(
                cast(Any, to_cookbook_tool_specs(request.tools)), system_prompt=system_prompt
            )
            messages = list(prefix) + messages
        return self.renderer.build_generation_prompt(messages, **render_kwargs)

    def fit_max_tokens(self, prompt: ModelInput, max_tokens: int) -> int:
        """Return the completion budget that fits the context window (AReaL parity).

        The request's ``max_tokens`` is a ceiling, not a target: it is clamped to the tokens remaining after the
        rendered prompt, so a large completion budget (the condenser's) never rejects a prompt that fits by
        itself.  A prompt that alone leaves less than :data:`MIN_CONTEXT_HEADROOM_TOKENS` raises
        :class:`ContextWindowExceededError`, which each transport reports as its native context-overflow error
        so the harness condenses instead of retrying.
        """
        if self.context_window_length is None:
            return max_tokens
        prompt_length = prompt.length
        headroom = self.context_window_length - prompt_length
        if headroom < MIN_CONTEXT_HEADROOM_TOKENS:
            raise ContextWindowExceededError(prompt_tokens=prompt_length, context_window=self.context_window_length)
        if max_tokens > headroom:
            logger.info(
                f"Clamping max_tokens {max_tokens} -> {headroom}: {prompt_length} prompt tokens "
                f"in a {self.context_window_length}-token context window."
            )
            return headroom
        return max_tokens

    def prepare(
        self, request: ChatRequest, *, defaults: SamplingOverrides | None = None, kind: str = "policy"
    ) -> PreparedCall:
        """Render the request, fit its completion budget and resolve its sampling parameters: the request's own,
        then ``defaults`` (the calling agent's ``sampling_defaults``), then the gateway's; defaults with
        ``precedence="config"`` decide the sampling of a ``policy`` call (``kind``, the agent's
        ``AgentSession.sampling_kind`` of the call's) instead (:meth:`SamplingOverrides.for_call`)."""
        started_at = time.perf_counter()
        sampling = request.sampling if defaults is None else defaults.for_call(request.sampling, kind=kind)
        prompt = self.render(request.messages, request.render)
        max_tokens = self.fit_max_tokens(
            prompt, self.max_tokens if sampling.max_tokens is None else sampling.max_tokens
        )
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=self.temperature if sampling.temperature is None else sampling.temperature,
            top_k=self.top_k if sampling.top_k is None else sampling.top_k,
            top_p=self.top_p if sampling.top_p is None else sampling.top_p,
            seed=self.seed if sampling.seed is None else sampling.seed,
            stop=self.renderer.get_stop_sequences() if sampling.stop is None else sampling.stop,
        )
        return PreparedCall(request, prompt, params, started_at, time.perf_counter() - started_at)

    # ------------------------------------------------------------------ sample and parse

    async def sample(self, call: PreparedCall) -> ChatResponse:
        """Sample one completion within ``sample_timeout_seconds`` and parse it, with the policy version it sampled
        (:meth:`_pinned`, whose reload of a followed binding counts as sampling, in its time and its failures);
        records nothing."""
        sample_started = time.perf_counter()
        try:
            sampler, version = self._pinned()
            result = await asyncio.wait_for(
                sampler.sample_async(prompt=call.prompt, sampling_params=call.sampling_params, num_samples=1),
                timeout=self.sample_timeout_seconds,
            )
        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - sample_started
            logger.error(f"sample_async timed out after {elapsed:.1f}s (timeout={self.sample_timeout_seconds}s)")
            raise TimeoutError(f"Tinker sample_async timed out after {self.sample_timeout_seconds}s")
        except Exception as e:
            elapsed = time.perf_counter() - sample_started
            logger.exception(f"sample_async failed after {elapsed:.1f}s: {e}")
            raise
        sample_seconds = time.perf_counter() - sample_started
        if sample_seconds > SLOW_SAMPLE_SECONDS:
            logger.warning(f"sample_async took {sample_seconds:.1f}s (slow)")
        return self._parse_timed(call, result, sample_seconds, version)

    def sample_sync(self, call: PreparedCall) -> ChatResponse:
        """The blocking twin of :meth:`sample`, for synchronous callers."""
        sample_started = time.perf_counter()
        sampler, version = self._pinned()
        result = sampler.sample(prompt=call.prompt, sampling_params=call.sampling_params, num_samples=1)
        return self._parse_timed(call, result, time.perf_counter() - sample_started, version)

    def _pinned(self) -> tuple[Sampler, int | None]:
        """The sampler the next call samples and the policy version it serves: those a sampler that follows a
        republished binding picks together (:class:`PinningSampler`), else :attr:`sampler` at :attr:`version`.  The
        version is read first: a publication writes the sampler, then its version (``TinkerLLM``), so one between the
        two reads labels the call older than it is, which the stale filter judges more strictly, never newer."""
        version = self.version
        sampler = self.sampler
        if callable(getattr(type(sampler), "pinned", None)):  # the class's: a delegating __getattr__ never answers
            return cast(PinningSampler, sampler).pinned()
        return sampler, version

    def _parse_timed(
        self, call: PreparedCall, result: tinker.types.SampleResponse, sample_seconds: float, version: int | None
    ) -> ChatResponse:
        parse_started = time.perf_counter()
        response = self.parse(call.prompt, result, call.request.render)
        timing = CallTiming(call.started_at, call.render_seconds, sample_seconds, parse_started, time.perf_counter())
        return replace(response, timing=timing, version=version)

    def parse(
        self, prompt: ModelInput, result: tinker.types.SampleResponse, request: RenderRequest | None = None
    ) -> ChatResponse:
        """Split the sampled sequence into content, reasoning and OpenAI tool calls.

        The parse sees the request's tools (schema-typed arguments), its chat-template kwargs and the prompt, so
        prompt and response agree about the thinking prefix; only what the renderer's ``parse_response`` takes is
        passed.  The raw text of tool-call blocks the renderer reports unparsed (``unparsed_tool_calls``) is
        appended to ``content``.
        """
        if len(result.sequences) != 1:
            raise ValueError(f"Expected one sampled sequence, got {len(result.sequences)}")
        sequence = result.sequences[0]
        request = request or RenderRequest()
        parse_kwargs: dict[str, Any] = {
            key: value for key, value in (request.render_kwargs or {}).items() if key in self._parse_response_params
        }
        if request.tools and "tools" in self._parse_response_params:
            parse_kwargs["tools"] = request.tools
        if "prompt" in self._parse_response_params:
            parse_kwargs["prompt"] = prompt
        parsed, parse_success = self.renderer.parse_response(sequence.tokens, **parse_kwargs)
        content, thinking = split_message_content(parsed.get("content"))
        unparsed = list(parsed.get("unparsed_tool_calls") or [])
        if unparsed:
            content = _with_unparsed_calls(content, unparsed)
            self._log_unparsed(unparsed)
        reasoning_content = parsed.get("reasoning_content")  # type: ignore[typeddict-item]
        if not isinstance(reasoning_content, str):
            reasoning_content = thinking
        if reasoning_content == "":
            reasoning_content = None
        stop_reason = str(sequence.stop_reason)
        role: MessageRole = "assistant"
        tool_calls = None
        if parse_success and stop_reason != "length":
            if parsed["role"] not in _MESSAGE_ROLES:
                raise ValueError(f"Invalid role: {parsed['role']}")
            role = parsed["role"]
            raw_tool_calls = parsed.get("tool_calls")
            if raw_tool_calls:
                tool_calls = [_openai_tool_call(tool_call) for tool_call in raw_tool_calls]
        if not content and not tool_calls and not reasoning_content:
            logger.warning("Parsed content is empty. Original response: " + str(result))
        return ChatResponse(
            call_id=generate_id("tinker-sampling-"),
            model=self.model_name,
            prompt=prompt,
            tokens=sequence.tokens,
            logprobs=sequence.logprobs,
            stop_reason=stop_reason,
            parse_success=bool(parse_success),
            role=role,
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            parsed_tool_calls=len(parsed.get("tool_calls") or []),
            routing_id=routing_id_of(result),
            recovered_tool_calls=int(cast(dict, parsed).get("recovered_tool_calls") or 0),
            unparsed_tool_calls=len(unparsed),
            policy=self.policy,
        )

    def _log_unparsed(self, unparsed: Sequence[Any]) -> None:
        """Warn about the first unparsed tool calls of the process, then about every thousandth."""
        self._unparsed_calls += len(unparsed)
        if self._unparsed_calls <= _UNPARSED_WARNINGS or self._unparsed_calls % 1000 < len(unparsed):
            first = unparsed[0]
            logger.warning(
                "%d tool-call block(s) did not parse and stay in the content (%d so far): %s; raw text %r",
                len(unparsed),
                self._unparsed_calls,
                getattr(first, "error", "?"),
                str(getattr(first, "raw_text", ""))[:200],
            )

    def decode_tokens(self, tokens: Sequence[int]) -> list[str]:
        """Each token decoded on its own (per-token logprob entries)."""
        return self.tokenizer.batch_decode([[token] for token in tokens])

    # ------------------------------------------------------------------ agents

    async def chat(
        self, agent: AgentSession, request: ChatRequest | PreparedCall, *, kind: str = "policy"
    ) -> ChatResponse:
        """Sample a call for ``agent`` and record it there as a ``kind`` call (``policy``, ``aux`` or ``judge``), in the
        runtime of the agent's policy (:meth:`runtime`).

        ``request`` may already be prepared (by that runtime), e.g. by a server that must report rendering errors
        before it starts a stream.  An agent closed while its call was sampling records nothing.  A kind the agent may
        not make (``AgentSession.check_kind``) raises :class:`CallKindRefusedError`, and a ``judge`` call of an agent
        whose judges are off (``AgentSession.judges``) :class:`JudgesOffError`, before anything is sampled.
        """
        check_call_kind(kind)
        agent.check_kind(kind)  # a kind the agent may never make is refused as such, not counted by the judges switch
        if kind == "judge" and not agent.judges:
            raise agent.refuse_judge("A judge call")
        self._ensure_running()
        self._ensure_open(agent)
        runtime = self.runtime(agent.policy)
        if isinstance(request, PreparedCall):
            call = request
        else:
            try:
                call = runtime.prepare(request, defaults=agent.sampling_defaults, kind=agent.sampling_kind(kind))
            except ContextWindowExceededError:
                agent.context_overflows += 1
                raise
            except (GatewayError, ValueError):
                raise  # the request's own fault
            except Exception:
                agent.gateway_errors += 1
                raise
        if (call_started := getattr(agent.timeline, "call_started", None)) is not None:
            call_started()  # the call is issued (CallTimeline)
        counted = 1 if kind == "policy" else 0  # a policy call counts against the agent's call cap while it samples
        agent.policy_calls_in_flight += counted
        agent.call_tally.in_flight += counted
        try:
            response = await runtime.sample(call)
        except Exception:
            agent.gateway_errors += 1
            raise
        finally:
            agent.policy_calls_in_flight -= counted
            agent.call_tally.in_flight -= counted
        self._ensure_open(agent)
        agent.record(response, kind=kind)
        return response

    def budget_stop(self, agent: AgentSession) -> ChatResponse:
        """The answer to a ``policy`` call past the agent's call cap (``AgentSession.over_call_cap``): an assistant
        message without tool calls (:data:`BUDGET_STOP_MESSAGE`), which ends a harness's loop.

        Nothing is sampled or recorded, so it never trains.  The agent counts it in ``budget_stops`` and its misc
        records the budget under ``budget``.  After :data:`MAX_BUDGET_STOPS` stops, the agent's next call raises
        :class:`AgentLimitError` instead: a harness that keeps calling after it was told to stop.
        """
        self._ensure_running()
        self._ensure_open(agent)
        if agent.budget_stops >= MAX_BUDGET_STOPS:
            raise AgentLimitError(
                f"Agent {agent.agent_id} has used its step budget of {agent.call_cap} calls and was told to stop "
                f"{agent.budget_stops} times; no more calls are answered"
            )
        agent.budget_stops += 1
        agent.misc[BUDGET_MISC_KEY] = {"enforce": "gateway", "max_steps": agent.call_cap, "stops": agent.budget_stops}
        return ChatResponse(
            call_id=generate_id("platoon-budget-"),
            model=self.runtime(agent.policy).model_name,
            prompt=ModelInput(chunks=[]),
            tokens=[],
            logprobs=None,
            stop_reason="stop",
            parse_success=True,
            role="assistant",
            content=BUDGET_STOP_MESSAGE.format(max_steps=agent.call_cap),
            reasoning_content=None,
            tool_calls=None,
            parsed_tool_calls=0,
            routing_id=None,
            policy=agent.policy,
        )

    def open_agent(
        self,
        *,
        rollout_id: str,
        trajectory_id: str,
        parent_agent_id: str | None = None,
        role: str = "policy",
        interactions: dict[str, TinkerLLMInteraction],
        goal: str | None = None,
        timeline: CallTimeline | None = None,
        max_steps: int | None = None,
        access: ToolAccess | None = None,
        sampling_defaults: SamplingOverrides | None = None,
    ) -> AgentSession:
        """Mint an agent with a fresh 256-bit key (``AgentSession.secret``).

        ``interactions`` is the map the agent's calls are recorded into, normally the rollout's
        (``TinkerLLMProxySession.interactions``); ``timeline`` optionally receives each call's phase timings.
        ``sampling_defaults`` (the rollout's ``inference_params``) fill the sampling parameters its calls leave
        unset.  A child is one level deeper than its parent and inherits its tool access and sampling defaults
        unless they are given, and its parent's ``judges``: a ``judge``-role child of a parent whose judges are off
        is refused (:class:`JudgesOffError`).  A gateway that is shut down refuses (:class:`GatewayShutDownError`).
        In a gateway that serves several policies the agent samples, for its whole life, the default policy if it is
        a root agent, else its role's (:meth:`serve_policies`).
        """
        self._ensure_running()
        fork_step, depth, inherited_access, inherited_sampling = 0, 0, "full", None
        judges = True
        if parent_agent_id is not None:
            parent = self._agents_by_id.get(parent_agent_id)
            if parent is None:
                raise UnknownAgentError(f"Unknown parent agent {parent_agent_id!r}")
            if parent.rollout_id != rollout_id:
                raise ValueError(f"Parent agent {parent_agent_id!r} belongs to rollout {parent.rollout_id!r}")
            fork_step, depth, inherited_access = len(parent.calls), parent.depth + 1, parent.tool_access
            inherited_sampling = parent.sampling_defaults
            judges = parent.judges
            if role == JUDGE_ROLE and not judges:
                raise parent.refuse_judge("A judge agent")
        agent = AgentSession(
            agent_id=uuid.uuid4().hex,
            rollout_id=rollout_id,
            trajectory_id=trajectory_id,
            secret=secrets.token_urlsafe(32),
            interactions=interactions,
            parent_agent_id=parent_agent_id,
            role=role,
            policy=self._policy_of(role, root=parent_agent_id is None),
            goal=goal,
            fork_step=fork_step,
            timeline=timeline,
            depth=depth,
            max_steps=max_steps,
            tool_access=access or inherited_access,
            sampling_defaults=inherited_sampling if sampling_defaults is None else sampling_defaults,
            judges=judges,
        )
        self._agents[agent.secret] = agent
        self._agents_by_id[agent.agent_id] = agent
        return agent

    def open_subagent(
        self,
        parent: AgentSession,
        *,
        role: str,
        goal: str | None = None,
        max_steps: int | None = None,
        access: ToolAccess | None = None,
    ) -> AgentSession:
        """Mint a subagent of ``parent``: the same rollout, interaction map and timeline.

        The child sees the parent's tools until it is rebound (``platoon.gateway.mcp.bind_agent_tools``), e.g. to a
        forked session.  Its access is ``access``, or the parent's when ``access`` is None.
        """
        self._ensure_running()
        self._ensure_open(parent)
        child = self.open_agent(
            rollout_id=parent.rollout_id,
            trajectory_id=str(uuid.uuid4()),
            parent_agent_id=parent.agent_id,
            role=role,
            interactions=parent.interactions,
            goal=goal,
            timeline=parent.timeline,
            max_steps=max_steps,
            access=access,
        )
        child.tools = parent.tools
        return child

    def mint_subagent(self, parent: AgentSession, *, role: str, goal: str | None = None) -> AgentSession:
        """A subagent a harness mints over HTTP (``POST /v1/platoon/agents``): :meth:`open_subagent`, within the
        rollout's limit on minted agents (:attr:`max_minted_agents`, else :class:`AgentLimitError`).

        The child gets its parent's call cap and shares its call tally, so its policy calls count against the
        parent's step budget and the cap bounds the parent and every agent minted under it together: a capped
        harness cannot escape its budget through children.  It also inherits ``policy_only``.
        """
        self._ensure_running()
        self._ensure_open(parent)
        minted = self._minted.get(parent.rollout_id, 0)
        if minted >= self.max_minted_agents:
            raise AgentLimitError(
                f"Rollout {parent.rollout_id} has minted its limit of {self.max_minted_agents} agents "
                f"({MAX_MINTED_AGENTS_ENV})"
            )
        child = self.open_subagent(parent, role=role, goal=goal)
        child.call_cap, child.call_tally, child.policy_only = parent.call_cap, parent.call_tally, parent.policy_only
        self._minted[parent.rollout_id] = minted + 1
        return child

    def authenticate(self, secret: str) -> AgentSession:
        """The open agent holding ``secret``."""
        agent = self._agents.get(secret)
        if agent is not None:
            return agent
        if secret in self._closed_secrets:
            raise AgentClosedError("This agent is closed")
        raise UnknownAgentError("Unknown agent key")

    def agents(self, rollout_id: str) -> list[AgentSession]:
        """The rollout's open agents in the order they were opened, so each parent precedes its children."""
        return [agent for agent in self._agents_by_id.values() if agent.rollout_id == rollout_id]

    def close_agent(self, agent: AgentSession) -> None:
        """Revoke the agent's key and stop recording into it.  Idempotent."""
        if agent.closed:
            return
        agent.closed = True
        self._agents.pop(agent.secret, None)
        self._agents_by_id.pop(agent.agent_id, None)
        self._closed_secrets[agent.secret] = None
        while len(self._closed_secrets) > MAX_CLOSED_SECRETS:
            del self._closed_secrets[next(iter(self._closed_secrets))]

    def close_rollout(self, rollout_id: str) -> list[AgentSession]:
        """Close every open agent of the rollout and return them, the input of ``build_trajectory_collection``."""
        agents = self.agents(rollout_id)
        for agent in agents:
            self.close_agent(agent)
        self._minted.pop(rollout_id, None)
        return agents

    @property
    def is_shut_down(self) -> bool:
        """Whether :meth:`shut_down` was called: no agent is open and none can open."""
        return self._shut_down

    def shut_down(self) -> None:
        """Stop for good: close every open agent and refuse to open new ones.  Idempotent.

        For a process that is exiting, such as a rollout worker told to stop.  Every later model call, through
        :meth:`chat` or HTTP, and every tool-plane call of the closed agents fails with :class:`AgentClosedError`
        (HTTP 410), so a harness that ignores cancellation, or calls from a thread of its own, can neither sample
        nor touch its env again; :meth:`open_agent` and :meth:`open_subagent` raise :class:`GatewayShutDownError`,
        so no subagent or verifier starts.  A call already sampling records nothing when it returns.
        """
        self._shut_down = True
        for agent in list(self._agents_by_id.values()):
            self.close_agent(agent)

    def _ensure_running(self) -> None:
        if self._shut_down:
            raise GatewayShutDownError("The model gateway is shut down: its process is stopping")

    @staticmethod
    def _ensure_open(agent: AgentSession) -> None:
        if agent.closed:
            raise AgentClosedError(f"Agent {agent.agent_id} is closed")


_default_gateway: ModelGateway | None = None


def set_default_gateway(gateway: ModelGateway | None) -> None:
    """Make ``gateway`` this process's policy gateway (``TinkerLLM.set_as_default``)."""
    global _default_gateway
    _default_gateway = gateway


def default_gateway() -> ModelGateway:
    """This process's policy gateway, the one ``harness_rollout`` serves (``TinkerLLM.set_as_default``)."""
    if _default_gateway is None:
        raise RuntimeError("No model gateway is registered in this process")
    return _default_gateway
