"""The model gateway: every policy call is rendered, sampled, parsed and recorded here.

* :mod:`platoon.gateway.core`: :class:`ModelGateway` and agent :class:`AgentSession` records.
* :mod:`platoon.gateway.server`: :class:`GatewayServer`, the HTTP front and the minimal agent API.
* :mod:`platoon.gateway.dialects`: wire formats, OpenAI Chat Completions plus experimental OpenAI Responses
  (Codex) and Anthropic Messages (Claude Code).
* :mod:`platoon.gateway.trace`: :func:`build_trajectory_collection`, agent records -> trainable trajectories.
* :mod:`platoon.gateway.hosted`: :class:`HostedGateway`, the same gateway over a hosted model, for evaluation: each
  call goes to one base URL in the harness's own dialect.

``platoon.train.tinker.proxy.TinkerLLM`` owns the process's gateway (``set_as_default``), which ``harness_rollout``
serves to harnesses over HTTP and to function harnesses in process.
"""

from platoon.gateway.core import (
    AgentClosedError,
    AgentLimitError,
    AgentSession,
    ChatRequest,
    ChatResponse,
    ContextWindowExceededError,
    GatewayError,
    GatewayShutDownError,
    HostedCallError,
    JudgesOffError,
    ModelGateway,
    ProviderRecord,
    RenderRequest,
    SamplingOverrides,
    TinkerLLMInteraction,
    UnknownAgentError,
    default_gateway,
    set_default_gateway,
)
from platoon.gateway.hosted import HostedGateway
from platoon.gateway.server import GatewayServer
from platoon.gateway.trace import build_trajectory_collection

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ContextWindowExceededError",
    "GatewayError",
    "GatewayServer",
    "GatewayShutDownError",
    "HostedCallError",
    "HostedGateway",
    "JudgesOffError",
    "ModelGateway",
    "AgentClosedError",
    "AgentLimitError",
    "AgentSession",
    "ProviderRecord",
    "RenderRequest",
    "SamplingOverrides",
    "TinkerLLMInteraction",
    "UnknownAgentError",
    "build_trajectory_collection",
    "default_gateway",
    "set_default_gateway",
]
