"""A tool allowlist per agent role (``agents.roles.<role>.tools``): what loading reads, what it refuses (exit 78) and
what it warns of (a pattern no declared tool matches, a role that looks like a misspelling of platoon's own), which
environments declare their tools up front for that warning (``platoon.envs.service.service_declared_tools``), and that
the workflows ``platoon train`` builds carry the lists to their rollout workers' requests."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml
from fakes.launch import built_workflows
from openreward.environments import Environment, ListToolsOutput, TextBlock, ToolOutput, terminal, tool
from pydantic import BaseModel

from platoon import cli
from platoon.envs.service import service_declared_tools
from platoon.train import identity
from platoon.train.config_defs import AgentRoleConfig, PlatoonTinkerRLTrainerConfig
from platoon.train.policy.spec import AgentBindings, agent_bindings, policy_problems, policy_warnings, role_tools
from platoon.train.workers.protocol import PolicyRolloutWorkerRequest
from platoon.utils.config import ConfigError, build_config, validate_config

SETTINGS: dict[str, Any] = {
    "train": {"model_name": "Qwen/Qwen3-4B-Instruct-2507", "renderer_name": "qwen3_instruct"},
    "eval": {"every": 0},
    "log_path": "logs",
}
NUMBER_SEARCH = "python://platoon.envs.number_search.env:NumberSearchEnvironment"  # one tool: guess
ROWS = {"rows": [{"id": "t1", "goal": "say hi"}], "splits": {"train": ["t1"], "eval": ["t1"]}}


def config(**sections: Any) -> PlatoonTinkerRLTrainerConfig:
    return build_config(PlatoonTinkerRLTrainerConfig, {**copy.deepcopy(SETTINGS), **sections})


def entry(service: str, **extra: Any) -> dict[str, Any]:
    return {"service": service, "harness": "platoon-react", **extra}


class Note(BaseModel):
    text: str


class Notes(Environment):
    """An environment whose class declares its tools: read_note, write_note, and the terminal submit."""

    @tool
    async def read_note(self, args: Note) -> ToolOutput:
        return ToolOutput(blocks=[TextBlock(text=args.text)])

    @tool
    async def write_note(self, args: Note) -> ToolOutput:
        return ToolOutput(blocks=[TextBlock(text=args.text)])

    @terminal
    @tool
    async def submit(self, args: Note) -> ToolOutput:
        return ToolOutput(blocks=[TextBlock(text=args.text)], finished=True)


class TaskNotes(Notes):
    """Notes with tools of its own for each task: what its sessions serve is known once one opens."""

    def list_task_tools(self) -> ListToolsOutput:
        return ListToolsOutput(tools=[])


class DeclaredService:
    """A service class that names the tools its sessions serve."""

    declared_tools = ("lookup", "note")


# --- loading ----------------------------------------------------------------------------------------------------------


def test_without_tools_every_role_keeps_every_tool():
    assert role_tools(config()) == {}
    assert role_tools(config(agents={"roles": {"judge": {"policy": "main"}}})) == {}


def test_a_role_may_list_tools_without_a_policy_and_samples_the_agents_default():
    loaded = config(agents={"roles": {"reviewer": {"tools": ["read_*", "view"]}, "judge": {"tools": []}}})

    assert role_tools(loaded) == {"reviewer": ("read_*", "view"), "judge": ()}
    assert agent_bindings(loaded) == AgentBindings()  # every agent samples main, as without agents:
    assert identity.policy_table(loaded) == {}  # identity.json as it was: tools are no part of a run's identity


@pytest.mark.parametrize(
    ("role", "start"),
    [
        ({"tools": "read"}, "agents.roles.reviewer.tools: 'read' is not a list[str] | null"),
        ({"tools": [1]}, "agents.roles.reviewer.tools.0: 1 is not a str"),
        ({"tool": ["read"]}, "agents.roles.reviewer.tool: unknown key"),
    ],
)
def test_loading_refuses_tools_that_are_no_list_of_names_and_unknown_keys(role, start):
    data = {**copy.deepcopy(SETTINGS), "agents": {"roles": {"reviewer": role}}}
    found = validate_config(data, PlatoonTinkerRLTrainerConfig)
    assert any(problem.startswith(start) for problem in found), found


def test_loading_refuses_an_empty_pattern_and_a_role_that_is_no_name():
    with pytest.raises(ConfigError) as raised:
        config(agents={"roles": {"reviewer": {"tools": ["read", ""]}, "": {"tools": ["read"]}}})
    assert raised.value.problems == [
        "agents.roles.reviewer.tools: a tool name or pattern is a non-empty string, got ''",
        "agents.roles: a role is a non-empty name, got ''",
    ]


def test_platoon_train_s_workflows_carry_the_lists_to_their_workers_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("PLATOON_NODE_SCRATCH", str(tmp_path / "scratch"))
    loaded = config(
        log_path=str(tmp_path / "logs"),
        environments=[entry(NUMBER_SEARCH)],
        agents={"roles": {"reviewer": {"tools": ["guess"]}}},
    )

    for workflow in built_workflows(loaded):  # training's and evaluation's
        assert dict(workflow.role_tools) == {"reviewer": ("guess",)}
        request = workflow._build_worker_request({"task_id": "t"}, 0)
        assert isinstance(request, PolicyRolloutWorkerRequest) and request.role_tools == {"reviewer": ("guess",)}


def test_a_config_built_in_code_whose_tools_are_no_list_is_refused():
    agents = SimpleNamespace(policy=None, roles={"r": AgentRoleConfig(tools=cast(Any, "x"))})
    [problem] = policy_problems(SimpleNamespace(policies={}, agents=agents))
    assert problem == "agents.roles.r.tools: a list of tool names or patterns (read_*), got 'x'; [] allows no tool"


def test_config_validate_exits_78_on_tools_that_are_no_list(tmp_path, capsys):
    data = {**copy.deepcopy(SETTINGS), "environments": [entry(NUMBER_SEARCH)]}
    data["agents"] = {"roles": {"reviewer": {"tools": "guess"}}}
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(data))

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["config", "validate", str(path)])

    assert exit_info.value.code == cli.EXIT_CONFIG
    captured = capsys.readouterr()
    assert "agents.roles.reviewer.tools: 'guess' is not a list[str] | null" in captured.out + captured.err


# --- warnings ---------------------------------------------------------------------------------------------------------


def test_a_pattern_that_matches_no_declared_tool_is_warned_of():
    loaded = config(environments=[entry(NUMBER_SEARCH)], agents={"roles": {"reviewer": {"tools": ["guess", "read_*"]}}})
    assert policy_warnings(loaded) == [
        "agents.roles.reviewer.tools: 'read_*' matches no tool of the run's environments (guess), so it allows nothing"
    ]
    matched = config(environments=[entry(NUMBER_SEARCH)], agents={"roles": {"r": {"tools": ["g*"]}}})
    assert policy_warnings(matched) == []


def test_a_role_that_looks_like_a_misspelling_of_platoon_s_own_is_warned_of():
    roles = {"subagents": {"tools": ["read"]}, "Judge": {"policy": "main"}, "verifier": {"tools": []}}
    roles |= {"reviewer": {"tools": ["read"]}, "agent": {"tools": ["read"]}}  # free names, close to none
    assert policy_warnings(config(agents={"roles": roles})) == [
        "agents.roles.subagents: platoon opens no agent of this role, so its settings reach only the agents a harness "
        "mints with it; did you mean 'subagent'?",
        "agents.roles.Judge: platoon opens no agent of this role, so its settings reach only the agents a harness "
        "mints with it; did you mean 'judge'?",
    ]


def test_a_task_table_declares_no_tool():
    loaded = config(environments=[entry("table", service_kwargs=ROWS)], agents={"roles": {"r": {"tools": ["x"]}}})
    assert policy_warnings(loaded) == [
        "agents.roles.r.tools: 'x' matches no tool of the run's environments (none), so it allows nothing"
    ]


def test_nothing_is_warned_of_when_an_environment_does_not_declare_its_tools():
    environments = [
        {**entry(NUMBER_SEARCH), "name": "search"},
        {**entry("ors+http://127.0.0.1:9/NumberSearchEnvironment"), "name": "remote"},
    ]
    loaded = config(environments=environments, agents={"roles": {"reviewer": {"tools": ["read_*"]}}})
    assert policy_warnings(loaded) == []


def test_which_services_declare_their_tools_up_front():
    assert service_declared_tools(NUMBER_SEARCH) == {"guess"}
    assert service_declared_tools(f"python://{__name__}:Notes") == {"read_note", "write_note", "submit"}
    assert service_declared_tools(f"python://{__name__}:TaskNotes") is None  # tools of its own for each task
    assert service_declared_tools(f"{__name__}:DeclaredService") == {"lookup", "note"}
    assert service_declared_tools("table") == frozenset()
    assert service_declared_tools("harbor") is None  # its tasks' MCP servers list their tools once a session opens
    assert service_declared_tools("ors+http://127.0.0.1:9/NumberSearchEnvironment") is None
    assert service_declared_tools(f"python://{__name__}:DeclaredService") is None  # no SDK environment
    assert service_declared_tools("no_such_module:Service") is None  # reported where the service resolves
