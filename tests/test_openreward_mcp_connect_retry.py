"""The MCP boot wrapper retries a fast failure but never a timeout.

``create_mcp_tools`` is what OpenHands calls to spawn the openreward mcp_bridge
at step 0 of every episode; when it raises, the trajectory is lost before the
agent takes a single action.  ``rollout._patch_mcp_boot_timeout`` therefore both
raises the SDK's hardcoded 30 s tool-listing timeout and gives a *fast* failure a
bounded second chance.  A ``MCPTimeoutError`` already consumed the whole timeout
budget, so retrying it would only eat the rollout deadline.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENREWARD_ROOT = REPO_ROOT / "plugins/openreward/platoon/openreward"


class _MCPError(Exception):
    pass


class _MCPTimeoutError(_MCPError):
    pass


class _AcceptsKeywords:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _module(monkeypatch, name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    parent_name, _, child = name.rpartition(".")
    if parent_name and parent_name in sys.modules:
        setattr(sys.modules[parent_name], child, module)
    return module


def _identity(agent, *args, **kwargs):
    return agent


def _load_rollout_module(monkeypatch, *, create_mcp_tools):
    """Load ``rollout.py`` with just enough of OpenHands stubbed out."""

    async def _apply_parent_usage_gates(*args, **kwargs):
        return None

    _module(monkeypatch, "openhands")
    _module(monkeypatch, "openhands.sdk", LLM=_AcceptsKeywords, Agent=_AcceptsKeywords)
    _module(monkeypatch, "openhands.sdk.agent")
    agent_base = _module(monkeypatch, "openhands.sdk.agent.base", create_mcp_tools=create_mcp_tools)
    _module(monkeypatch, "openhands.sdk.mcp")
    _module(
        monkeypatch,
        "openhands.sdk.mcp.exceptions",
        MCPError=_MCPError,
        MCPTimeoutError=_MCPTimeoutError,
    )
    _module(monkeypatch, "openhands.sdk.context")
    _module(
        monkeypatch,
        "openhands.sdk.context.condenser",
        LLMSummarizingCondenser=_AcceptsKeywords,
    )
    _module(
        monkeypatch,
        "platoon.agents.actions.subagent",
        SubagentRewardJudgeConfig=_AcceptsKeywords,
        apply_subagent_parent_usage_gates=_apply_parent_usage_gates,
    )
    _module(monkeypatch, "platoon.openhands")
    _module(monkeypatch, "platoon.openhands.agent", OpenHandsAgent=_AcceptsKeywords)
    _module(
        monkeypatch,
        "platoon.openhands.condenser",
        SafeLLMSummarizingCondenser=_AcceptsKeywords,
    )
    _module(
        monkeypatch,
        "platoon.openhands.recursive",
        PROGRAMMATIC_TOOL_CALLING_ORCHESTRATION_ONLY_SYSTEM_PROMPT_SUFFIX="",
        PROGRAMMATIC_TOOL_CALLING_SYSTEM_PROMPT_SUFFIX="",
        RECURSIVE_SUBAGENT_INITIAL_TASK_SUFFIX="",
        RECURSIVE_SUBAGENT_SYSTEM_PROMPT_SUFFIX="",
        RECURSIVE_SUBAGENT_USER_MESSAGE_SUFFIX="",
        TASK_TRACKER_INITIAL_TASK_SUFFIX="",
        TASK_TRACKER_SYSTEM_PROMPT_SUFFIX="",
        append_system_message_suffix=_identity,
        append_user_message_suffix=_identity,
        with_programmatic_tool_calling=_identity,
        with_task_tracker_tool=_identity,
    )
    _module(monkeypatch, "platoon.openreward.env", OpenRewardOpenHandsEnv=_AcceptsKeywords)

    package = types.ModuleType("platoon.openreward")
    package.__path__ = [str(OPENREWARD_ROOT)]
    monkeypatch.setitem(sys.modules, "platoon.openreward", package)
    monkeypatch.delitem(sys.modules, "platoon.openreward.config_defs", raising=False)
    monkeypatch.delitem(sys.modules, "platoon.openreward.rollout", raising=False)

    spec = importlib.util.spec_from_file_location("platoon.openreward.rollout", OPENREWARD_ROOT / "rollout.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "platoon.openreward.rollout", module)
    spec.loader.exec_module(module)
    return module, agent_base


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    monkeypatch.setenv("OPENREWARD_MCP_CONNECT_RETRY_DELAY", "0")
    monkeypatch.setenv("OPENREWARD_MCP_TIMEOUT", "1800")


def test_boot_timeout_override_is_forwarded(monkeypatch):
    seen: list[float] = []

    def create_mcp_tools(config, timeout=30.0):
        seen.append(timeout)
        return "client"

    module, agent_base = _load_rollout_module(monkeypatch, create_mcp_tools=create_mcp_tools)
    assert agent_base.create_mcp_tools({"mcpServers": {}}) == "client"
    # The SDK's hardcoded 30 s must be replaced by OPENREWARD_MCP_TIMEOUT.
    assert seen == [1800.0]
    assert module is not None


def test_a_fast_mcp_failure_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setenv("OPENREWARD_MCP_CONNECT_RETRIES", "2")
    attempts = {"n": 0}

    def create_mcp_tools(config, timeout=30.0):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _MCPError("MCP Connection Failure")
        return "client"

    _, agent_base = _load_rollout_module(monkeypatch, create_mcp_tools=create_mcp_tools)
    assert agent_base.create_mcp_tools({"mcpServers": {}}) == "client"
    assert attempts["n"] == 3


def test_retries_are_bounded(monkeypatch):
    monkeypatch.setenv("OPENREWARD_MCP_CONNECT_RETRIES", "2")
    attempts = {"n": 0}

    def create_mcp_tools(config, timeout=30.0):
        attempts["n"] += 1
        raise _MCPError("MCP Connection Failure")

    _, agent_base = _load_rollout_module(monkeypatch, create_mcp_tools=create_mcp_tools)
    with pytest.raises(_MCPError):
        agent_base.create_mcp_tools({"mcpServers": {}})
    assert attempts["n"] == 3  # the first attempt plus two retries


def test_a_timeout_is_never_retried(monkeypatch):
    monkeypatch.setenv("OPENREWARD_MCP_CONNECT_RETRIES", "5")
    attempts = {"n": 0}

    def create_mcp_tools(config, timeout=30.0):
        attempts["n"] += 1
        raise _MCPTimeoutError("MCP tool listing timed out")

    _, agent_base = _load_rollout_module(monkeypatch, create_mcp_tools=create_mcp_tools)
    with pytest.raises(_MCPTimeoutError):
        agent_base.create_mcp_tools({"mcpServers": {}})
    assert attempts["n"] == 1


def test_retries_can_be_disabled(monkeypatch):
    monkeypatch.setenv("OPENREWARD_MCP_CONNECT_RETRIES", "0")
    attempts = {"n": 0}

    def create_mcp_tools(config, timeout=30.0):
        attempts["n"] += 1
        raise _MCPError("MCP Connection Failure")

    _, agent_base = _load_rollout_module(monkeypatch, create_mcp_tools=create_mcp_tools)
    with pytest.raises(_MCPError):
        agent_base.create_mcp_tools({"mcpServers": {}})
    assert attempts["n"] == 1
