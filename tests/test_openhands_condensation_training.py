from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENHANDS_PLUGIN_ROOT = REPO_ROOT / "plugins/openhands"


def _module(monkeypatch, name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _install_openhands_stubs(monkeypatch) -> None:
    class ConversationExecutionStatus:
        FINISHED = "FINISHED"
        STUCK = "STUCK"
        ERROR = "ERROR"

    _module(monkeypatch, "openhands")
    _module(monkeypatch, "openhands.sdk")
    _module(monkeypatch, "openhands.sdk.agent")
    _module(monkeypatch, "openhands.sdk.agent.base", AgentBase=object)
    _module(monkeypatch, "openhands.sdk.conversation", get_agent_final_response=lambda events: "")
    _module(monkeypatch, "openhands.sdk.conversation.base", BaseConversation=object, ConversationStateProtocol=object)
    _module(monkeypatch, "openhands.sdk.conversation.conversation", Conversation=object)
    _module(monkeypatch, "openhands.sdk.conversation.state", ConversationExecutionStatus=ConversationExecutionStatus)
    action_event = type("ActionEvent", (), {})
    agent_error_event = type("AgentErrorEvent", (), {})
    conversation_error_event = type("ConversationErrorEvent", (), {})
    event = type("Event", (), {})
    llm_convertible_event = type("LLMConvertibleEvent", (), {})
    message_event = type("MessageEvent", (), {})
    observation_base_event = type("ObservationBaseEvent", (), {})
    event_pkg = _module(
        monkeypatch,
        "openhands.sdk.event",
        ActionEvent=action_event,
        AgentErrorEvent=agent_error_event,
        Event=event,
        EventID=str,
        LLMConvertibleEvent=llm_convertible_event,
        MessageEvent=message_event,
        ObservationBaseEvent=observation_base_event,
    )
    event_pkg.__path__ = []
    _module(
        monkeypatch,
        "openhands.sdk.event.base",
        Event=object,
        LLMConvertibleEvent=llm_convertible_event,
    )
    _module(monkeypatch, "openhands.sdk.event.conversation_error", ConversationErrorEvent=conversation_error_event)
    _module(monkeypatch, "openhands.sdk.event.llm_convertible")
    _module(monkeypatch, "openhands.sdk.event.llm_convertible.action", ActionEvent=action_event)
    _module(monkeypatch, "openhands.sdk.tool")
    _module(monkeypatch, "openhands.sdk.tool.builtins")
    _module(monkeypatch, "openhands.sdk.tool.builtins.finish", FinishAction=type("FinishAction", (), {}))
    _module(monkeypatch, "openhands.sdk.workspace")
    _module(monkeypatch, "openhands.sdk.workspace.base", BaseWorkspace=type("BaseWorkspace", (), {}))


def _load_openhands_env_module(monkeypatch):
    _install_openhands_stubs(monkeypatch)
    monkeypatch.syspath_prepend(str(REPO_ROOT))

    package = types.ModuleType("platoon.openhands")
    package.__path__ = [str(OPENHANDS_PLUGIN_ROOT / "platoon/openhands")]
    monkeypatch.setitem(sys.modules, "platoon.openhands", package)
    _module(
        monkeypatch,
        "platoon.openhands.recursive",
        DEFAULT_SUBAGENT_MAX_STEPS=50,
        copy_agent_config_for_fork=lambda agent: agent,
    )

    spec = importlib.util.spec_from_file_location(
        "platoon.openhands.env",
        OPENHANDS_PLUGIN_ROOT / "platoon/openhands/env.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "platoon.openhands.env", module)
    spec.loader.exec_module(module)
    return module


class _TrajectoryCollection:
    def __init__(self):
        self.steps = []

    def add_trajectory_step(self, trajectory_id, step):
        self.steps.append((trajectory_id, step))


class _Completion:
    def to_tensor_dict(self):
        return {
            "input_ids": torch.tensor([[10, 11, 12, 13]]),
            "loss_mask": torch.tensor([[0, 0, 1, 1]]),
            "logprobs": torch.tensor([[0.0, 0.0, -0.1, -0.2]]),
            "versions": torch.tensor([[-1, -1, 3, 3]]),
        }


def test_sanitized_model_condensation_emits_one_synthetic_trainable_step(monkeypatch):
    env_mod = _load_openhands_env_module(monkeypatch)
    env = env_mod.OpenHandsEnv.__new__(env_mod.OpenHandsEnv)
    env._synthetic_condensation_step_event_ids = set()

    collection = _TrajectoryCollection()
    condensation = SimpleNamespace(
        kind="Condensation",
        id="condensation-1",
        # A reasoning-enabled condenser retains this real completion ID after
        # removing private reasoning from the public summary below.
        llm_response_id="chatcmpl-reasoning-summary",
        summary=(
            "USER_CONTEXT: Fix the parser.\n"
            "COMPLETED: Located the implementation.\n"
            "PENDING: Apply and test the patch.\n"
            "CURRENT_STATE: No files changed yet."
        ),
    )
    sys.modules["platoon.openhands.condensation_safety"].remember_condensation_reasoning(
        condensation.id,
        "Inspect forgotten events and identify the durable parser state.",
    )

    env._add_trainable_condensation_steps(collection, "trajectory-1", [condensation])
    env._add_trainable_condensation_steps(collection, "trajectory-1", [condensation])

    assert len(collection.steps) == 1
    trajectory_id, step = collection.steps[0]
    assert trajectory_id == "trajectory-1"
    assert step.action_events is None
    assert step.observation_events == [condensation]
    assert step.misc["action_misc"] == {"completion_id": "chatcmpl-reasoning-summary"}
    assert step.misc["reward_misc"] == {}
    assert step.misc["synthetic_step_type"] == "openhands_condensation"
    assert step.misc["condensation_reasoning"] == ("Inspect forgotten events and identify the durable parser state.")


def test_unsafe_condensation_does_not_emit_trainable_step(monkeypatch):
    env_mod = _load_openhands_env_module(monkeypatch)
    env = env_mod.OpenHandsEnv.__new__(env_mod.OpenHandsEnv)
    env._synthetic_condensation_step_event_ids = set()
    collection = _TrajectoryCollection()

    condensation = SimpleNamespace(
        kind="Condensation",
        id="condensation-unsafe",
        llm_response_id="chatcmpl-unsafe-summary",
        summary=(
            "Here's a thinking process:\n1. Analyze the user's instructions.\n</think>\n\nUSER_CONTEXT: Fix the parser."
        ),
    )

    env._add_trainable_condensation_steps(collection, "trajectory-1", [condensation])

    assert collection.steps == []


def test_deterministic_fallback_condensation_is_not_linked_as_model_completion(
    monkeypatch,
):
    env_mod = _load_openhands_env_module(monkeypatch)
    env = env_mod.OpenHandsEnv.__new__(env_mod.OpenHandsEnv)
    env._synthetic_condensation_step_event_ids = set()
    collection = _TrajectoryCollection()

    condensation = SimpleNamespace(
        kind="Condensation",
        id="condensation-fallback",
        llm_response_id="platoon-nontrainable-condensation-abc123",
        summary=(
            "USER_CONTEXT: Fix the parser.\n"
            "COMPLETED: Earlier history was compacted.\n"
            "PENDING: Continue and test.\n"
            "CURRENT_STATE: Inspect parser.py."
        ),
    )

    env._add_trainable_condensation_steps(collection, "trajectory-1", [condensation])

    assert collection.steps == []


@pytest.mark.asyncio
async def test_reward_verifier_gets_fresh_recursive_launcher_runtime(monkeypatch):
    env_mod = _load_openhands_env_module(monkeypatch)
    recursive = sys.modules["platoon.openhands.recursive"]
    from platoon.agents.actions.subagent import (
        SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY,
    )
    from platoon.envs.base import SubTask, Task

    events: list[object] = []

    class OldRuntime:
        def close(self):
            events.append("old-closed")

    class Runtime:
        def __init__(self):
            events.append("new-created")

        def bind(self, loop, context):
            assert loop is asyncio.get_running_loop()
            assert context is not None
            events.append("new-bound")

    recursive.with_shared_workspace_subagent_prompt = lambda agent: (
        "workspace",
        agent,
    )
    recursive.with_finish_tool = lambda agent: ("finish", agent)
    recursive.LaunchSubagentRuntime = Runtime

    def with_launch_subagent_tool(agent, *, runtime, default_max_steps):
        events.append(("launcher-installed", runtime, default_max_steps))
        return ("launcher", agent)

    recursive.with_launch_subagent_tool = with_launch_subagent_tool

    env = env_mod.OpenHandsEnv.__new__(env_mod.OpenHandsEnv)
    env._task = SubTask(
        id="verifier",
        goal="verify child",
        parent_tasks=[Task(id="child", goal="do work")],
        misc={SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY: True},
    )
    env._agent = "copied-agent-with-inherited-launcher"
    env._enable_recursive_subagents = True
    # The recursive OpenReward jobs give verifier helpers the same 50-step
    # default as policy subagents.
    env._subagent_default_max_steps = 50
    env._launch_subagent_runtime = OldRuntime()

    configured = env._prepare_agent_for_conversation()

    assert configured[0] == "launcher"
    assert events[:3] == ["old-closed", "new-created", "new-bound"]
    assert events[3][0] == "launcher-installed"
    assert events[3][1] is env._launch_subagent_runtime
    assert events[3][2] == 50


@pytest.mark.asyncio
async def test_close_interrupts_async_conversation_before_recursive_children(monkeypatch):
    env_mod = _load_openhands_env_module(monkeypatch)
    order: list[str] = []

    async def conversation_run():
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            order.append("conversation-cancelled")
            return

    class Conversation:
        def interrupt(self):
            order.append("conversation-interrupted")

        def close(self):
            order.append("conversation-closed")

    class Runtime:
        async def aclose(self):
            order.append("children-cancelled")

    env = env_mod.OpenHandsEnv.__new__(env_mod.OpenHandsEnv)
    env._conversation_id = "test"
    env._conversation = Conversation()
    env._conversation_task = asyncio.create_task(conversation_run())
    env._launch_subagent_runtime = Runtime()
    await asyncio.sleep(0)

    await env.close()

    assert order[:3] == [
        "conversation-interrupted",
        "conversation-cancelled",
        "children-cancelled",
    ]
    assert order[-1] == "conversation-closed"


@pytest.mark.asyncio
async def test_close_does_not_join_hung_sdk_close_via_default_executor(monkeypatch):
    env_mod = _load_openhands_env_module(monkeypatch)
    monkeypatch.setattr(env_mod, "_CONVERSATION_CLOSE_TIMEOUT_SECONDS", 0.02)
    close_started = threading.Event()
    release_close = threading.Event()

    class Conversation:
        def interrupt(self):
            return None

        def close(self):
            close_started.set()
            release_close.wait()

    env = env_mod.OpenHandsEnv.__new__(env_mod.OpenHandsEnv)
    env._conversation_id = "test"
    env._conversation = Conversation()
    env._conversation_task = None
    env._launch_subagent_runtime = None

    started_at = asyncio.get_running_loop().time()
    await env.close()
    elapsed = asyncio.get_running_loop().time() - started_at

    assert close_started.is_set()
    assert elapsed < 0.2
    release_close.set()


def test_synthetic_step_uses_existing_areal_completion_extraction_path(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    from platoon.utils.areal_data_processing import get_train_data_for_step

    step = {
        "misc": {
            "action_misc": {"completion_id": "chatcmpl-summary"},
            "synthetic_step_type": "openhands_condensation",
        },
        "action_events": None,
        "observation_events": [{"kind": "Condensation", "llm_response_id": "chatcmpl-summary"}],
    }

    train_data = get_train_data_for_step(
        step,
        {"chatcmpl-summary": _Completion()},
        task_id="task",
    )

    assert train_data is not None
    assert torch.equal(train_data["input_ids"], torch.tensor([[10, 11, 12, 13]]))
    assert torch.equal(train_data["loss_mask"], torch.tensor([[0, 0, 1, 1]]))
    assert torch.equal(train_data["logprobs"], torch.tensor([[0.0, 0.0, -0.1, -0.2]]))
    assert torch.equal(train_data["versions"], torch.tensor([[-1, -1, 3, 3]]))
