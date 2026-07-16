import ast
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_openreward_config_module():
    spec = importlib.util.spec_from_file_location(
        "openreward_config_defs",
        REPO_ROOT / "plugins/openreward/platoon/openreward/config_defs.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["openreward_config_defs"] = module
    spec.loader.exec_module(module)
    return module


def _module(monkeypatch, name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _install_openreward_package(monkeypatch) -> None:
    package = types.ModuleType("platoon.openreward")
    package.__path__ = [str(REPO_ROOT / "plugins/openreward/platoon/openreward")]
    monkeypatch.setitem(sys.modules, "platoon.openreward", package)


def _load_openreward_mcp_bridge_module(monkeypatch):
    class FastMCP:
        pass

    _module(monkeypatch, "mcp")
    _module(monkeypatch, "mcp.server")
    _module(monkeypatch, "mcp.server.fastmcp", FastMCP=FastMCP)
    spec = importlib.util.spec_from_file_location(
        "openreward_mcp_bridge",
        REPO_ROOT / "plugins/openreward/platoon/openreward/mcp_bridge.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["openreward_mcp_bridge"] = module
    spec.loader.exec_module(module)
    return module


def _load_openreward_tasks_module(monkeypatch):
    _install_openreward_package(monkeypatch)
    monkeypatch.delitem(sys.modules, "platoon.openreward.tasks", raising=False)
    return __import__("platoon.openreward.tasks", fromlist=["get_task"])


def _load_openreward_env_module(monkeypatch):
    class ConversationExecutionStatus:
        FINISHED = "finished"

    class Event:
        pass

    class OpenHandsEnv:
        def __init__(
            self,
            task,
            agent,
            workspace,
            callbacks=None,
            persistence_dir=None,
            conversation_id=None,
            enable_recursive_subagents=False,
            subagent_default_max_steps=50,
            **kwargs,
        ):
            _ = kwargs
            self._task = task
            self._agent = agent
            self._workspace = workspace
            self._callbacks = callbacks or []
            self._persistence_dir = persistence_dir
            self._conversation_id = conversation_id
            self._enable_recursive_subagents = enable_recursive_subagents
            self._subagent_default_max_steps = subagent_default_max_steps
            self._conversation = None

    _module(monkeypatch, "openhands")
    _module(monkeypatch, "openhands.sdk")
    _module(monkeypatch, "openhands.sdk.conversation")
    _module(
        monkeypatch,
        "openhands.sdk.conversation.state",
        ConversationExecutionStatus=ConversationExecutionStatus,
    )
    _module(monkeypatch, "openhands.sdk.event")
    _module(monkeypatch, "openhands.sdk.event.base", Event=Event)
    _module(monkeypatch, "platoon.openhands")
    _module(monkeypatch, "platoon.openhands.env", OpenHandsEnv=OpenHandsEnv)

    _install_openreward_package(monkeypatch)
    monkeypatch.delitem(sys.modules, "platoon.openreward.env", raising=False)
    return __import__("platoon.openreward.env", fromlist=["OpenRewardOpenHandsEnv"])


def test_openreward_config_splits_ptc_and_recursion_flags():
    config_mod = _load_openreward_config_module()
    config = config_mod.OpenRewardConfig.from_mapping(
        {
            "enable_programmatic_tool_calling": True,
            "enable_recursive_subagents": False,
            "subagent_max_depth": 2,
        }
    )

    assert config.enable_programmatic_tool_calling is True
    assert config.enable_recursive_subagents is False
    assert config.subagent_max_depth == 2


def test_openreward_config_defaults_subagent_budget_to_50():
    config_mod = _load_openreward_config_module()

    config = config_mod.OpenRewardConfig.from_mapping({})

    assert config.subagent_default_max_steps == 50


def test_openreward_config_defaults_subagent_judging_off():
    config_mod = _load_openreward_config_module()

    config = config_mod.OpenRewardConfig.from_mapping({})

    assert config.enable_subagent_reward_judging is False
    assert config.subagent_reward_judge_max_steps == 20
    assert config.subagent_delegation_reward_coefficient == 0.0


def test_openreward_config_rejects_negative_delegation_reward_coefficient():
    config_mod = _load_openreward_config_module()

    with pytest.raises(ValueError, match="must be non-negative"):
        config_mod.OpenRewardConfig.from_mapping({"subagent_delegation_reward_coefficient": -0.1})


def test_openreward_config_accepts_legacy_subagent_judging_keys():
    config_mod = _load_openreward_config_module()

    config = config_mod.OpenRewardConfig.from_mapping(
        {
            "enable_subagent_judging": True,
            "subagent_judge_max_steps": 7,
        }
    )

    assert config.enable_subagent_reward_judging is True
    assert config.subagent_reward_judge_max_steps == 7


def test_openreward_reward_processor_uses_subagent_judgment(monkeypatch):
    _install_openreward_package(monkeypatch)
    monkeypatch.delitem(sys.modules, "platoon.openreward.rewards", raising=False)
    rewards_mod = __import__("platoon.openreward.rewards", fromlist=["reward_processor"])

    reward, components = rewards_mod.reward_processor(
        {
            "reward": 0.0,
            "misc": {"subagent_reward_judgment": {"score": 0.75}},
            "steps": [],
        }
    )

    assert reward == 0.75
    assert components["reward/success"] == 0.75
    assert components["reward/openreward"] == 0.0
    assert components["reward/subagent_judgment"] == 0.75
    assert components["reward/subagent_launched"] == 0.0
    assert components["reward/subagent_succeeded"] == 0.0
    assert "reward/subagent_success_rate" not in components
    assert components["reward/delegation_bonus"] == 0.0
    assert components["reward/total"] == 0.75


def test_openreward_recursive_delegation_bonus_is_direct_and_non_compounding(monkeypatch):
    from platoon.utils.subagent_rewards import (
        SUBAGENT_DELEGATION_REWARD_MISC_KEY,
        add_direct_subagent_delegation_rewards,
    )

    def trajectory(
        reward: float,
        *,
        parent_id: str | None = None,
        excluded: bool = False,
        policy_excluded: bool = False,
    ) -> dict:
        misc = {}
        if excluded:
            misc["exclude_from_training"] = True
        if policy_excluded:
            misc["exclude_from_policy_training"] = True
        result = {
            "reward": reward,
            "misc": misc,
            "steps": [{"misc": {"reward_misc": {"reward/success": reward}}}],
        }
        if parent_id is not None:
            result["parent_info"] = {"id": parent_id}
        return result

    collection = {
        "trajectories": {
            "root": trajectory(1.0),
            "child": trajectory(0.6, parent_id="root"),
            "grandchild-success": trajectory(1.0, parent_id="child"),
            # An invalid verifier suppresses this child's own policy datums but
            # must not erase the delegation from full-tree accounting.
            "grandchild-failure": trajectory(
                0.0,
                parent_id="child",
                policy_excluded=True,
            ),
            # Verifiers have a real parent edge but must never count as a
            # successful delegation by the trajectory they judge.
            "verifier": trajectory(1.0, parent_id="child", excluded=True),
        }
    }

    add_direct_subagent_delegation_rewards(collection, coefficient=0.4)

    root_meta = collection["trajectories"]["root"]["misc"][SUBAGENT_DELEGATION_REWARD_MISC_KEY]
    child_meta = collection["trajectories"]["child"]["misc"][SUBAGENT_DELEGATION_REWARD_MISC_KEY]
    assert root_meta == {
        "coefficient": 0.4,
        "launched": 1.0,
        "succeeded": 0.6,
        "success_rate": 0.6,
        "bonus": pytest.approx(0.24),
    }
    assert child_meta == {
        "coefficient": 0.4,
        "launched": 2.0,
        "succeeded": 1.0,
        "success_rate": 0.5,
        "bonus": 0.2,
    }

    _install_openreward_package(monkeypatch)
    monkeypatch.delitem(sys.modules, "platoon.openreward.rewards", raising=False)
    rewards_mod = __import__("platoon.openreward.rewards", fromlist=["reward_processor"])
    root_reward, root_components = rewards_mod.reward_processor(collection["trajectories"]["root"])
    child_reward, child_components = rewards_mod.reward_processor(collection["trajectories"]["child"])

    assert root_reward == pytest.approx(1.24)
    assert root_components["reward/subagent_succeeded"] == 0.6
    assert root_components["reward/delegation_bonus"] == pytest.approx(0.24)
    # The child gets its own grandchild bonus, but the root's numerator above
    # remains the child's base score (0.6), not this processed total (0.8).
    assert child_reward == pytest.approx(0.8)
    assert child_components["reward/subagent_launched"] == 2.0
    assert child_components["reward/delegation_bonus"] == 0.2


def test_openreward_rollout_condenser_keeps_system_prompt_and_goal_only():
    source = REPO_ROOT.joinpath("plugins/openreward/platoon/openreward/rollout.py").read_text()
    module = ast.parse(source)

    keep_first = None
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "OPENREWARD_CONDENSER_KEEP_FIRST" for target in node.targets
        ):
            keep_first = ast.literal_eval(node.value)
            break

    assert keep_first == 2


def test_openreward_rollout_honors_root_success_propagation_flag():
    source = REPO_ROOT.joinpath("plugins/openreward/platoon/openreward/rollout.py").read_text()
    module = ast.parse(source)
    run_rollout = next(
        node for node in module.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_rollout"
    )

    guarded_calls = [
        node
        for node in ast.walk(run_rollout)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and isinstance(node.test.value, ast.Name)
        and node.test.value.id == "config"
        and node.test.attr == "propogate_root_success"
    ]

    propagation_guards = [
        guard
        for guard in guarded_calls
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "propogate_root_success"
            for node in ast.walk(guard)
        )
    ]
    # The flag may also guard timeout and delegation-reward compatibility, but
    # exactly one branch must own the actual root-success mutation.
    assert len(propagation_guards) == 1


def test_root_success_propagation_relabels_helpful_child_trajectory():
    from platoon.utils.subagent_rewards import propogate_root_success

    collection = {
        "trajectories": {
            "root": {
                "reward": 0.75,
                "steps": [{"misc": {"reward_misc": {"reward/success": 0.75}}}],
            },
            "child": {
                "reward": 0.0,
                "steps": [
                    {
                        "misc": {
                            "reward_misc": {
                                "reward/success": 0.0,
                                "reward/subagent_launched": 1.0,
                            }
                        }
                    }
                ],
            },
        }
    }

    result = propogate_root_success(collection)

    assert result is collection
    assert result["trajectories"]["root"]["reward"] == 0.75
    assert result["trajectories"]["child"]["reward"] == 0.75
    child_reward_misc = result["trajectories"]["child"]["steps"][-1]["misc"]["reward_misc"]
    assert child_reward_misc["reward/success"] == 0.75
    assert child_reward_misc["reward/subagent_succeeded"] == 0.75


def test_openreward_mcp_bridge_declares_tools_lockfree(monkeypatch):
    bridge_mod = _load_openreward_mcp_bridge_module(monkeypatch)

    assert bridge_mod._lockfree_tool_meta() == {bridge_mod.DECLARED_RESOURCES_META_KEY: []}


def test_openreward_task_loader_marks_goal_for_env_resolution(monkeypatch):
    tasks_mod = _load_openreward_tasks_module(monkeypatch)

    task = tasks_mod.get_task("fetch-dashboard")

    assert task.id == "fetch-dashboard"
    assert task.goal == "Task fetch-dashboard"
    assert task.misc[tasks_mod.OPENREWARD_RESOLVE_GOAL_KEY] is True


def test_openreward_goal_format_uses_resolved_prompt_directly(monkeypatch):
    env_mod = _load_openreward_env_module(monkeypatch)
    from platoon.envs.base import Task

    task = Task(
        id="fetch-dashboard",
        goal="Task fetch-dashboard",
    )
    payload = {
        "task_name": "fetch-dashboard",
        "prompt": "Build the KPI report.",
        "environment_tools": [
            {"name": "fetch_rows", "description": "Fetch rows."},
        ],
        "policy": "Call claim_done when complete.",
    }

    goal = env_mod._format_openreward_task_goal(
        task=task,
        payload=payload,
        initial_goal_suffix="Use subagents when helpful.",
    )

    assert goal == "Build the KPI report.\n\nUse subagents when helpful."
    assert "fetch_rows" not in goal
    assert "OpenReward" not in goal
    assert "get_task" not in goal


def test_openreward_goal_format_adds_child_tree_context(monkeypatch):
    env_mod = _load_openreward_env_module(monkeypatch)
    from platoon.envs.base import SubTask, Task

    root = Task(
        id="fetch-dashboard",
        goal="Build the KPI report.",
        misc={
            env_mod._CURRENT_AGENT_TASK_GOAL_KEY: "Build the KPI report.",
            env_mod._ROOT_AGENT_TASK_GOAL_KEY: "Build the KPI report.",
        },
    )
    child = SubTask(
        id="child",
        goal="Inspect the warehouse tables.",
        misc=root.misc,
        parent_tasks=[root],
    )
    payload = {
        "task_name": "fetch-dashboard",
        "prompt": "Build the KPI report.",
        "environment_tools": [{"name": "python_execute"}],
    }

    goal = env_mod._format_openreward_task_goal(task=child, payload=payload)

    assert "You are a sub-agent provided a task by a parent" in goal
    assert "Your Task:\nInspect the warehouse tables." in goal
    assert "call `finish`" in goal
    assert "returned verbatim to the parent" in goal
    assert "return exactly that format" in goal
    assert "Parent Agent Task:\nBuild the KPI report." in goal
    assert "Root Agent Task:" not in goal
    assert "python_execute" not in goal
    assert "OpenReward" not in goal


def test_openreward_goal_format_adds_root_context_for_nested_child(monkeypatch):
    env_mod = _load_openreward_env_module(monkeypatch)
    from platoon.envs.base import SubTask, Task

    root = Task(
        id="root",
        goal="Build the KPI report.",
        misc={
            env_mod._CURRENT_AGENT_TASK_GOAL_KEY: "Build the KPI report.",
            env_mod._ROOT_AGENT_TASK_GOAL_KEY: "Build the KPI report.",
        },
    )
    parent = SubTask(
        id="parent",
        goal="Inspect warehouse tables.",
        misc={
            env_mod._CURRENT_AGENT_TASK_GOAL_KEY: "Inspect warehouse tables.",
            env_mod._ROOT_AGENT_TASK_GOAL_KEY: "Build the KPI report.",
        },
        parent_tasks=[root],
    )
    child = SubTask(
        id="child",
        goal="Summarize revenue tables.",
        misc=parent.misc,
        parent_tasks=[root, parent],
    )

    goal = env_mod._format_openreward_task_goal(
        task=child,
        payload={"prompt": "Build the KPI report."},
    )

    assert "Your Task:\nSummarize revenue tables." in goal
    assert "Parent Agent Task:\nInspect warehouse tables." in goal
    assert "Root Agent Task:\nBuild the KPI report." in goal


def test_openreward_shared_tools_are_non_owning(monkeypatch):
    env_mod = _load_openreward_env_module(monkeypatch)

    class Executor:
        def __init__(self):
            self.called = False
            self.closed = False
            self.interrupted = False

        def __call__(self, action, conversation):
            self.called = True
            return {"action": action, "conversation": conversation}

        def interrupt(self):
            self.interrupted = True

        def close(self):
            self.closed = True

    class Tool:
        def __init__(self, name, executor, server_name):
            self.name = name
            self.executor = executor
            self.mcp_server_name = server_name

        def as_executable(self):
            return self

        def set_executor(self, executor):
            return Tool(self.name, executor, self.mcp_server_name)

    class Agent:
        def __init__(self, tools):
            self.tools_map = {tool.name: tool for tool in tools}

    executor = Executor()
    other_executor = Executor()
    agent = Agent(
        [
            Tool("get_task", executor, "openreward"),
            Tool("claim_done", Executor(), "openreward"),
            Tool("other_tool", other_executor, "other"),
        ]
    )

    shared_tools = env_mod._openreward_mcp_tools(agent)
    shared_executor = shared_tools["get_task"].executor
    result = shared_executor({"ok": True}, "conversation")
    shared_executor.interrupt()
    shared_executor.close()

    assert set(shared_tools) == {"get_task"}
    assert result == {"action": {"ok": True}, "conversation": "conversation"}
    assert executor.called is True
    assert executor.interrupted is True
    assert executor.closed is False
    assert other_executor.called is False


@pytest.mark.asyncio
async def test_openreward_env_resolves_get_task_before_first_message(monkeypatch):
    env_mod = _load_openreward_env_module(monkeypatch)
    from platoon.envs.base import Task

    class TextBlock:
        def __init__(self, text):
            self.text = text

    class FakeTool:
        def action_from_arguments(self, arguments):
            assert arguments == {}
            return {"data": arguments}

        def __call__(self, action, conversation):
            assert action == {"data": {}}
            assert conversation.ready is True
            payload = {
                "task_name": "task",
                "prompt": "Use the data tools.",
                "environment_tools": [{"name": "call_tool"}],
            }
            return types.SimpleNamespace(
                content=[
                    TextBlock("[Tool 'get_task' executed.]"),
                    TextBlock(json.dumps(payload)),
                ]
            )

    class FakeAgent:
        tools_map = {"get_task": FakeTool()}

    class FakeConversation:
        def __init__(self):
            self.ready = False
            self.agent = FakeAgent()

        def _ensure_agent_ready(self):
            self.ready = True

    task = Task(
        id="task",
        goal="Task task",
    )
    env = env_mod.OpenRewardOpenHandsEnv(task=task, agent=object(), workspace=".")
    env._conversation = FakeConversation()

    goal = await env._initial_user_message()

    assert env._conversation.ready is True
    assert goal == "Use the data tools."
    assert "call_tool" not in goal
    assert env._task.goal == goal
    assert env._task.misc[env_mod._CURRENT_AGENT_TASK_GOAL_KEY] == "Use the data tools."
    assert env._task.misc[env_mod._ROOT_AGENT_TASK_GOAL_KEY] == "Use the data tools."


@pytest.mark.asyncio
async def test_openreward_child_uses_shared_get_task_tool(monkeypatch):
    env_mod = _load_openreward_env_module(monkeypatch)
    from platoon.envs.base import SubTask, Task

    class TextBlock:
        def __init__(self, text):
            self.text = text

    class SharedGetTaskTool:
        def __init__(self):
            self.called = False

        def action_from_arguments(self, arguments):
            assert arguments == {}
            return {"data": arguments}

        def __call__(self, action, conversation):
            assert action == {"data": {}}
            assert conversation.ready is True
            self.called = True
            payload = {
                "task_name": "task",
                "prompt": "Use the shared session.",
                "environment_tools": [{"name": "call_tool"}],
            }
            return types.SimpleNamespace(
                content=[
                    TextBlock("[Tool 'get_task' executed.]"),
                    TextBlock(json.dumps(payload)),
                ]
            )

    class FakeAgent:
        def __init__(self):
            self._tools = {}

        @property
        def tools_map(self):
            return self._tools

    class FakeConversation:
        def __init__(self):
            self.ready = False
            self.agent = FakeAgent()

        def _ensure_agent_ready(self):
            self.ready = True

    shared_tool = SharedGetTaskTool()
    root_task = Task(
        id="root",
        goal="Use the data tools.",
        misc={
            env_mod._CURRENT_AGENT_TASK_GOAL_KEY: "Use the data tools.",
            env_mod._ROOT_AGENT_TASK_GOAL_KEY: "Use the data tools.",
        },
    )
    child_task = SubTask(
        id="child",
        goal="Inspect the data.",
        misc=root_task.misc,
        parent_tasks=[root_task],
    )
    env = env_mod.OpenRewardOpenHandsEnv(
        task=child_task,
        agent=object(),
        workspace=".",
        shared_openreward_tools={"get_task": shared_tool},
    )
    env._conversation = FakeConversation()

    goal = await env._initial_user_message()

    assert shared_tool.called is True
    assert "Your Task:\nInspect the data." in goal
    assert "Parent Agent Task:\nUse the data tools." in goal
    assert "Use the shared session." not in goal
    assert "OpenReward" not in goal
    assert env._conversation.agent.tools_map["get_task"] is shared_tool
    assert env._task.misc[env_mod._CURRENT_AGENT_TASK_GOAL_KEY] == "Inspect the data."
    assert env._task.misc[env_mod._ROOT_AGENT_TASK_GOAL_KEY] == "Use the data tools."


@pytest.mark.asyncio
async def test_openreward_fork_reuses_shared_tools_without_mcp_config(monkeypatch):
    env_mod = _load_openreward_env_module(monkeypatch)
    from platoon.envs.base import Task

    _module(
        monkeypatch,
        "platoon.openhands.recursive",
        copy_agent_config_for_fork=lambda agent: agent,
    )

    class FakeAgent:
        def __init__(self, mcp_config):
            self.mcp_config = mcp_config

        def model_copy(self, update):
            return FakeAgent(update.get("mcp_config", self.mcp_config))

    shared_tools = {"get_task": object()}
    parent = env_mod.OpenRewardOpenHandsEnv(
        task=Task(id="root", goal="root"),
        agent=FakeAgent({"mcpServers": {"openreward": {}}}),
        workspace="workspace",
        persistence_dir="/tmp/openreward-persistence",
        enable_recursive_subagents=True,
        subagent_default_max_steps=50,
        initial_goal_suffix="use subagents",
        shared_openreward_tools=shared_tools,
    )

    child = await parent.fork(Task(id="child", goal="child"))

    assert child._agent.mcp_config == {}
    assert child._shared_openreward_tools is shared_tools
    assert child._workspace == "workspace"
    assert child._initial_goal_suffix == "use subagents"
    assert child._enable_recursive_subagents is True
