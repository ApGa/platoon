from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from platoon.agents.actions.subagent import launch_subagent
from platoon.config_defs import RolloutConfig
from platoon.envs.base import Observation, SubTask, Task
from platoon.episode.context import current_trajectory, current_trajectory_collection
from platoon.episode.trajectory import TrajectoryStep
from platoon.utils.trajectory_status import TRAJECTORY_CANCELLED_MISC_KEY

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENREWARD_ROOT = REPO_ROOT / "plugins/openreward/platoon/openreward"


def _module(monkeypatch, name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_rollout_module(
    monkeypatch,
    *,
    agent_type=None,
    env_type=None,
    recursive_overrides=None,
):
    class _AcceptsKeywords:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _Agent:
        async def reset(self) -> None:
            return None

        async def act(self, observation):
            raise AssertionError("the environment reset should remain blocked")

        async def close(self) -> None:
            return None

    class _HangingEnv:
        def __init__(self, *, task, **kwargs):
            self._task = task

        @property
        def task(self):
            return self._task

        async def reset(self):
            collection = current_trajectory_collection.get()
            root = current_trajectory.get()
            collection.set_trajectory_task(root.id, self._task)
            # Model a child which finished before the root became a straggler.
            # The timeout should finalize only the active root and preserve
            # this completed sibling/descendant in the partial collection.
            child = collection.create_trajectory(parent_traj=root)
            collection.set_trajectory_task(
                child.id,
                Task(id="completed-child", goal="already done", max_steps=1),
            )
            child.reward = 1.0
            child.finish_message = "child done"
            collection.finish_trajectory(child.id)
            await asyncio.Future()
            raise AssertionError("unreachable")

        async def step(self, action):
            raise AssertionError("reset should remain blocked")

        async def observe(self):
            raise AssertionError("observe should not be called")

        async def close(self) -> None:
            return None

    def _identity(agent, *args, **kwargs):
        return agent

    agent_type = agent_type or _Agent
    env_type = env_type or _HangingEnv

    _module(monkeypatch, "openhands")
    _module(monkeypatch, "openhands.sdk", LLM=_AcceptsKeywords, Agent=_AcceptsKeywords)
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
    )
    _module(monkeypatch, "platoon.openhands")
    _module(monkeypatch, "platoon.openhands.agent", OpenHandsAgent=agent_type)
    _module(
        monkeypatch,
        "platoon.openhands.condenser",
        SafeLLMSummarizingCondenser=_AcceptsKeywords,
    )
    recursive_attrs = {
        "PROGRAMMATIC_TOOL_CALLING_ORCHESTRATION_ONLY_SYSTEM_PROMPT_SUFFIX": "",
        "PROGRAMMATIC_TOOL_CALLING_SYSTEM_PROMPT_SUFFIX": "",
        "RECURSIVE_SUBAGENT_INITIAL_TASK_SUFFIX": "",
        "RECURSIVE_SUBAGENT_SYSTEM_PROMPT_SUFFIX": "",
        "RECURSIVE_SUBAGENT_USER_MESSAGE_SUFFIX": "",
        "TASK_TRACKER_INITIAL_TASK_SUFFIX": "",
        "TASK_TRACKER_SYSTEM_PROMPT_SUFFIX": "",
        "append_system_message_suffix": _identity,
        "append_user_message_suffix": _identity,
        "with_programmatic_tool_calling": _identity,
        "with_task_tracker_tool": _identity,
    }
    recursive_attrs.update(recursive_overrides or {})
    _module(
        monkeypatch,
        "platoon.openhands.recursive",
        **recursive_attrs,
    )
    _module(monkeypatch, "platoon.openreward.env", OpenRewardOpenHandsEnv=env_type)

    package = types.ModuleType("platoon.openreward")
    package.__path__ = [str(OPENREWARD_ROOT)]
    monkeypatch.setitem(sys.modules, "platoon.openreward", package)
    monkeypatch.delitem(sys.modules, "platoon.openreward.config_defs", raising=False)
    monkeypatch.delitem(sys.modules, "platoon.openreward.rollout", raising=False)

    spec = importlib.util.spec_from_file_location(
        "platoon.openreward.rollout",
        OPENREWARD_ROOT / "rollout.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "platoon.openreward.rollout", module)
    spec.loader.exec_module(module)
    return module


def _agent_configuration_spies():
    prompts = {"system": [], "user": []}

    def _with_tool(name):
        def _configure(agent, **_kwargs):
            agent.tools.append(name)
            return agent

        return _configure

    def _append_prompt(kind):
        def _append(agent, suffix):
            prompts[kind].append(suffix)
            return agent

        return _append

    return prompts, {
        "PROGRAMMATIC_TOOL_CALLING_ORCHESTRATION_ONLY_SYSTEM_PROMPT_SUFFIX": "ptc-orchestration-guidance",
        "PROGRAMMATIC_TOOL_CALLING_SYSTEM_PROMPT_SUFFIX": "ptc-guidance",
        "RECURSIVE_SUBAGENT_INITIAL_TASK_SUFFIX": "recursive-initial-guidance",
        "RECURSIVE_SUBAGENT_SYSTEM_PROMPT_SUFFIX": "recursive-system-guidance",
        "RECURSIVE_SUBAGENT_USER_MESSAGE_SUFFIX": "recursive-user-guidance",
        "TASK_TRACKER_INITIAL_TASK_SUFFIX": "task-tracker-initial-guidance",
        "TASK_TRACKER_SYSTEM_PROMPT_SUFFIX": "task-tracker-guidance",
        "append_system_message_suffix": _append_prompt("system"),
        "append_user_message_suffix": _append_prompt("user"),
        "with_programmatic_tool_calling": _with_tool("programmatic_tool_calling"),
        "with_task_tracker_tool": _with_tool("task_tracker"),
    }


def test_openreward_agent_uses_openhands_sdk_prompt_directory(monkeypatch, tmp_path) -> None:
    rollout = _load_rollout_module(monkeypatch)
    sdk_agent_dir = tmp_path / "openhands" / "sdk" / "agent"
    prompt_dir = sdk_agent_dir / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "system_prompt.j2").write_text("OpenHands system prompt")

    sdk_agent_module = _module(monkeypatch, "fake_openhands_sdk_agent")
    sdk_agent_module.__file__ = str(sdk_agent_dir / "agent.py")
    monkeypatch.setattr(
        rollout.OpenHandsSDKAgent,
        "__module__",
        sdk_agent_module.__name__,
    )

    agent = rollout.OpenRewardAgent()

    assert Path(agent.prompt_dir) == prompt_dir
    assert (Path(agent.prompt_dir) / "system_prompt.j2").is_file()


def test_nonrecursive_ptc_and_task_tracker_do_not_add_delegation(monkeypatch) -> None:
    prompts, recursive_overrides = _agent_configuration_spies()
    rollout = _load_rollout_module(
        monkeypatch,
        recursive_overrides=recursive_overrides,
    )
    agent = types.SimpleNamespace(tools=[])
    config = rollout.OpenRewardConfig.from_mapping(
        {
            "enable_programmatic_tool_calling": True,
            "enable_task_tracker": True,
            "enable_recursive_subagents": False,
        }
    )

    configured = rollout._configure_openhands_agent(agent, config)

    assert configured.tools == ["programmatic_tool_calling", "task_tracker"]
    assert "launch_subagent" not in configured.tools
    assert prompts["system"] == [
        "ptc-guidance\n\nptc-orchestration-guidance\n\ntask-tracker-guidance"
    ]
    assert prompts["user"] == [""]
    assert all("recursive" not in prompt for values in prompts.values() for prompt in values)


def test_unrestricted_ptc_omits_orchestration_policy_and_passes_mode(monkeypatch):
    calls = []

    def with_ptc(agent, **kwargs):
        calls.append(kwargs)
        return agent

    prompts, recursive_overrides = _agent_configuration_spies()
    recursive_overrides["with_programmatic_tool_calling"] = with_ptc
    rollout = _load_rollout_module(
        monkeypatch,
        recursive_overrides=recursive_overrides,
    )
    config = rollout.OpenRewardConfig.from_mapping(
        {
            "enable_programmatic_tool_calling": True,
            "programmatic_tool_calling_mode": "unrestricted",
        }
    )

    rollout._configure_openhands_agent(types.SimpleNamespace(tools=[]), config)

    assert calls == [
        {
            "mode": "unrestricted",
            "max_tool_calls_per_execution": 1024,
        }
    ]
    assert prompts["system"] == ["ptc-guidance"]


def test_mcp_bridge_receives_environment_routing_overrides(monkeypatch, tmp_path):
    rollout = _load_rollout_module(monkeypatch)
    routing = {
        "version": 1,
        "execution_domain": "task",
        "capabilities": ["tool.dispatch"],
        "invocation": {
            "kind": "dispatcher",
            "name_argument": "name",
            "arguments_argument": "arguments",
            "targets": [],
        },
    }
    environment = rollout.OpenRewardEnvironmentConfig(
        env_name="legacy",
        tool_routing_overrides={"call_tool": routing},
    )
    task = Task(
        id="task-7",
        goal="test",
        misc={rollout.OPENREWARD_TASK_INDEX_KEY: 7},
    )

    config = rollout._build_mcp_config(
        task,
        rollout.OpenRewardConfig(),
        environment,
        str(tmp_path),
    )
    args = config["mcpServers"]["openreward"]["args"]
    option_index = args.index("--tool-routing-overrides-json")

    assert json.loads(args[option_index + 1]) == {"call_tool": routing}


def test_task_tracker_initial_guidance_is_shared_by_recursive_and_nonrecursive(
    monkeypatch,
) -> None:
    _, recursive_overrides = _agent_configuration_spies()
    rollout = _load_rollout_module(
        monkeypatch,
        recursive_overrides=recursive_overrides,
    )
    nonrecursive = rollout.OpenRewardConfig.from_mapping(
        {
            "enable_task_tracker": True,
            "enable_recursive_subagents": False,
        }
    )
    recursive = rollout.OpenRewardConfig.from_mapping(
        {
            "enable_task_tracker": True,
            "enable_recursive_subagents": True,
        }
    )

    assert (
        rollout._initial_task_prompt_suffix(nonrecursive)
        == "task-tracker-initial-guidance"
    )
    assert rollout._initial_task_prompt_suffix(recursive) == (
        "task-tracker-initial-guidance\n\nrecursive-initial-guidance"
    )


def test_openreward_agent_rejects_plain_message_completion(monkeypatch) -> None:
    rollout = _load_rollout_module(monkeypatch)

    class TextContent:
        def __init__(self, *, text):
            self.text = text

    class Message:
        def __init__(self, *, role, content):
            self.role = role
            self.content = content

    class MessageEvent:
        def __init__(self, *, source, llm_message):
            self.source = source
            self.llm_message = llm_message

    _module(monkeypatch, "openhands.sdk.event", MessageEvent=MessageEvent)
    _module(
        monkeypatch,
        "openhands.sdk.llm",
        Message=Message,
        TextContent=TextContent,
    )

    emitted = []
    agent = rollout.OpenRewardAgent(tools_map={"submit_answer": object()})
    agent._emit_message_event = lambda *args: emitted.append("assistant")
    agent._maybe_emit_vllm_tokens = lambda *args: None
    state = types.SimpleNamespace(execution_status="running")

    agent._handle_content_response(
        message=object(),
        llm_response=object(),
        conversation=object(),
        state=state,
        on_event=emitted.append,
    )

    assert state.execution_status == "running"
    assert emitted[0] == "assistant"
    nudge = emitted[1]
    assert nudge.source == "user"
    assert "did not submit" in nudge.llm_message.content[0].text
    assert "call `submit_answer`" in nudge.llm_message.content[0].text


def test_local_condenser_defaults_to_bounded_reasoning_completion(monkeypatch) -> None:
    rollout = _load_rollout_module(monkeypatch)
    config = RolloutConfig(
        model_name="openai/Qwen/Qwen3.6-35B-A3B",
        model_endpoint="http://127.0.0.1:8000/v1",
        model_api_key="local",
        inference_params={"max_completion_tokens": 4096},
    )

    llm = rollout._build_condenser_llm(config, rollout.OpenRewardConfig())

    assert llm.usage_id == "platoon-openreward-openhands-condenser"
    assert llm.max_output_tokens == 26_214
    assert llm.custom_tokenizer is None
    assert llm.litellm_extra_body == {
        "reasoning_effort": "high",
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": False,
        },
    }


def test_condenser_thinking_can_be_explicitly_disabled(monkeypatch) -> None:
    rollout = _load_rollout_module(monkeypatch)
    config = RolloutConfig(
        model_name="openai/hosted-model",
        model_endpoint="http://127.0.0.1:8000/v1",
        model_api_key="local",
        inference_params={"max_completion_tokens": 4096},
    )

    llm = rollout._build_condenser_llm(
        config,
        rollout.OpenRewardConfig(condenser_disable_thinking=True),
    )

    assert llm.litellm_extra_body == {
        "reasoning_effort": "none",
        "chat_template_kwargs": {
            "enable_thinking": False,
            "preserve_thinking": False,
        }
    }


def test_recursive_mode_still_installs_task_tracker_and_delegation_prompts(monkeypatch) -> None:
    prompts, recursive_overrides = _agent_configuration_spies()
    rollout = _load_rollout_module(
        monkeypatch,
        recursive_overrides=recursive_overrides,
    )
    agent = types.SimpleNamespace(tools=[])
    config = rollout.OpenRewardConfig.from_mapping(
        {
            "enable_recursive_subagents": True,
            "subagent_max_depth": 2,
        }
    )

    configured = rollout._configure_openhands_agent(agent, config)

    assert configured.tools == ["task_tracker"]
    assert prompts["system"] == [
        "task-tracker-guidance\n\nrecursive-system-guidance\n\n"
        "Recursive subagents are limited to maximum depth 2; the root agent is depth 0."
    ]
    assert prompts["user"] == ["recursive-user-guidance"]


def _counting_rollout_types(*, child_max_steps: int | None = None):
    class CountingAgent:
        async def reset(self) -> None:
            return None

        async def act(self, observation):
            return "continue"

        async def close(self) -> None:
            return None

        async def fork(self, task: Task):
            return type(self)()

    class CountingEnv:
        instances = []

        def __init__(self, *, task, **kwargs):
            self._task = task
            self._launched_child = False
            self.step_calls = 0
            self.instances.append(self)

        @property
        def task(self):
            return self._task

        async def reset(self):
            collection = current_trajectory_collection.get()
            trajectory = current_trajectory.get()
            collection.set_trajectory_task(trajectory.id, self._task)
            # OpenHands records its initial observation as a trajectory step.
            collection.add_trajectory_step(trajectory.id, TrajectoryStep())
            return Observation(task=self._task)

        async def step(self, action):
            if (
                child_max_steps is not None
                and not isinstance(self._task, SubTask)
                and not self._launched_child
            ):
                self._launched_child = True
                await launch_subagent(
                    goal="bounded child",
                    max_steps=child_max_steps,
                )
            self.step_calls += 1
            collection = current_trajectory_collection.get()
            trajectory = current_trajectory.get()
            collection.add_trajectory_step(trajectory.id, TrajectoryStep())
            return Observation(task=self._task)

        async def observe(self):
            return Observation(task=self._task)

        async def close(self) -> None:
            return None

        async def fork(self, task: Task):
            return type(self)(task=task)

    return CountingAgent, CountingEnv


@pytest.mark.asyncio
async def test_openreward_rollout_timeout_returns_marked_partial_collection(monkeypatch, tmp_path) -> None:
    rollout = _load_rollout_module(monkeypatch)
    config = RolloutConfig(
        output_dir=str(tmp_path),
        timeout=0.01,
        step_timeout=60,
        return_dict=True,
        propogate_root_success=False,
    )

    result = await rollout.run_rollout(
        Task(id="partial-timeout", goal="wait forever", max_steps=1),
        config,
    )

    assert result["misc"]["rollout_timed_out"] is True
    assert len(result["trajectories"]) == 2
    root, child = result["trajectories"].values()
    assert root["misc"][TRAJECTORY_CANCELLED_MISC_KEY] is True
    assert "Episode cancelled" in root["error_message"]
    assert child["finish_message"] == "child done"
    assert TRAJECTORY_CANCELLED_MISC_KEY not in child["misc"]


@pytest.mark.asyncio
async def test_openreward_rollout_applies_configured_root_step_cap(monkeypatch, tmp_path) -> None:
    agent_type, env_type = _counting_rollout_types()
    rollout = _load_rollout_module(
        monkeypatch,
        agent_type=agent_type,
        env_type=env_type,
    )
    source_task = Task(id="bounded-root", goal="keep going")
    config = RolloutConfig(
        output_dir=str(tmp_path),
        max_steps=3,
        timeout=5,
        step_timeout=1,
        return_dict=True,
    )

    result = await rollout.run_rollout(source_task, config)

    root = next(
        trajectory
        for trajectory in result["trajectories"].values()
        if trajectory["parent_info"] is None
    )
    assert len(root["steps"]) == 3
    assert env_type.instances[0]._task.max_steps == 3
    assert env_type.instances[0].step_calls == 2
    assert result["misc"]["rollout_timed_out"] is False
    # Applying a rollout-local override should not mutate reusable task data.
    assert source_task.max_steps is None


@pytest.mark.asyncio
async def test_openreward_root_cap_preserves_recursive_child_budget(monkeypatch, tmp_path) -> None:
    agent_type, env_type = _counting_rollout_types(child_max_steps=2)
    rollout = _load_rollout_module(
        monkeypatch,
        agent_type=agent_type,
        env_type=env_type,
    )
    config = RolloutConfig(
        output_dir=str(tmp_path),
        max_steps=4,
        timeout=5,
        step_timeout=1,
        return_dict=True,
        extra={
            "openreward": {
                "enable_recursive_subagents": True,
            }
        },
    )

    result = await rollout.run_rollout(
        Task(id="recursive-root", goal="delegate then continue"),
        config,
    )

    root = next(
        trajectory
        for trajectory in result["trajectories"].values()
        if trajectory["parent_info"] is None
    )
    child = next(
        trajectory
        for trajectory in result["trajectories"].values()
        if trajectory["parent_info"] is not None
    )
    assert root["task"]["max_steps"] == 4
    assert len(root["steps"]) == 4
    assert child["task"]["max_steps"] == 2
    assert len(child["steps"]) == 2
    assert result["misc"]["rollout_timed_out"] is False
