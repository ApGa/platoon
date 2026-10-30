"""``agents.child_grading: env`` is refused where the child's forked session could not grade the child's own goal,
a verifier that sees no tools is reported, and a verifier's tools are limited by its spawner's and its own role's
allowlists (``agents.roles.<role>.tools``), not by those of the child it judges.

``env`` grades a child by its forked session.  A task table has no grader.  On a ``shared`` fork (the default of
every built-in service) the child's session is its parent's live env, which grades the child's goal only through
an in-process class's ``platoon_fork`` hook: without it the child gets the episode's first submission, or 0.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import pytest
import yaml
from tinker import ModelInput

from platoon import cli
from platoon.envs.number_search.env import NumberSearchEnvironment
from platoon.envs.service import Outcome, ToolSpec
from platoon.gateway.core import ModelGateway
from platoon.harness import agents as agents_module
from platoon.harness.agent import AgentResult
from platoon.harness.function import FunctionHarness
from platoon.harness.inline import InlineAgentContext, InlineRuntime
from platoon.harness.registry import get_harness
from platoon.harness.verify import VERIFIER_ROLE
from platoon.task import Task
from platoon.train.components import AgentTreeConfig, EnvironmentConfig

NUMBER_SEARCH = "python://platoon.envs.number_search.env:NumberSearchEnvironment"  # no platoon_fork
FORKING = "python://test_child_grading_checks:ForkingEnvironment"  # platoon_fork: a child's own goal
ROWS = {"rows": [{"id": "t1", "goal": "say hi"}]}


class ForkingEnvironment(NumberSearchEnvironment):
    """An in-process class with a ``platoon_fork`` hook, as TextCraft's: a child gets its own goal on the shared
    state."""

    def platoon_fork(self, task_spec: dict[str, Any]) -> ForkingEnvironment:
        return self


def _problems(service: str, **agents: Any) -> list[str]:
    from platoon.train.auto import harness_setting_problems

    entry = EnvironmentConfig(
        service=service,
        service_kwargs=ROWS if service == "table" else {},
        harness="platoon-codeact",
        agents=agents,
    )
    return harness_setting_problems(entry, [get_harness(ref) for ref in entry.harness_refs()])


@pytest.mark.parametrize(
    ("service", "agents", "expected"),
    [
        ("table", {"fork": "fresh"}, "a task table has no grader"),
        ("ors+http://127.0.0.1:9/Env", {}, "an env server cannot run a platoon_fork hook"),
        ("ors+http://127.0.0.1:9/Env", {"fork": "shared"}, "an env server cannot run a platoon_fork hook"),
        ("harbor", {}, "the task's whole test.sh on the sandbox it shares with its parent"),
        (NUMBER_SEARCH, {}, "needs NumberSearchEnvironment.platoon_fork"),
    ],
    ids=["table", "remote-default-fork", "remote-shared", "harbor", "in-process-without-hook"],
)
def test_env_child_grading_that_cannot_grade_a_childs_own_goal_is_refused(service, agents, expected):
    (problem,) = _problems(service, child_grading="env", **agents)
    assert problem.startswith("environments[default].agents: child_grading: env") and expected in problem


@pytest.mark.parametrize(
    ("service", "agents"),
    [
        (FORKING, {"child_grading": "env"}),  # platoon_fork grades each child's own goal
        ("ors+http://127.0.0.1:9/Env", {"child_grading": "env", "fork": "fresh"}),
        ("harbor", {"child_grading": "env", "fork": "copy"}),
        (NUMBER_SEARCH, {"child_grading": "verifier"}),
        ("ors+http://127.0.0.1:9/Env", {"child_grading": "none"}),
        # max_depth 0: no child is ever spawned, so none is graded
        (NUMBER_SEARCH, {"child_grading": "env", "max_depth": 0}),
        ("ors+http://127.0.0.1:9/Env", {"child_grading": "env", "max_depth": 0}),
        ("table", {"child_grading": "env", "max_depth": 0}),
    ],
    ids=[
        "in-process-with-hook",
        "remote-fresh",
        "harbor-copy",
        "verifier",
        "none",
        "in-process-max-depth-0",
        "remote-max-depth-0",
        "table-max-depth-0",
    ],
)
def test_child_grading_that_grades_the_child_is_not_refused(service, agents):
    assert _problems(service, **agents) == []


def test_platoon_rollout_exits_78_for_env_grading_on_a_class_without_platoon_fork(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")
    path = tmp_path / "config.yaml"
    config = {
        "inference": {"model_name": "fake-model", "renderer_name": "qwen3_instruct", "output_dir": str(tmp_path / "r")},
        "environments": [{"service": NUMBER_SEARCH, "harness": "platoon-codeact", "agents": {"child_grading": "env"}}],
    }
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["rollout", str(path), "--task", "val:0", "--output-dir", str(tmp_path / "run")])

    assert exit_info.value.code == cli.EXIT_CONFIG
    err = capsys.readouterr().err
    assert "environments[default].agents: child_grading: env on shared forks needs" in err and "Traceback" not in err


# --------------------------------------------------------------------------- verifiers that see no tools


class _Renderer:
    """Enough of a renderer for a gateway that never samples."""

    def build_generation_prompt(self, messages: Any) -> ModelInput:
        return ModelInput.from_ints([1, 2, 3])

    def parse_response(self, tokens: Any) -> Any:
        return {"role": "assistant", "content": ""}, True

    def get_stop_sequences(self) -> list[Any]:
        return []


@dataclass
class _Session:
    """A session whose tools are ``specs``, forked to sessions with the same tools; every grade is 1."""

    task: Task
    specs: list[ToolSpec] = field(default_factory=list)
    sandbox: Any = None
    instruction: str = ""

    async def tools(self) -> list[ToolSpec]:
        return list(self.specs)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        raise AssertionError("no calls")

    async def grade(self) -> Outcome:
        return Outcome(status="scored", rewards={"reward": 1.0})

    async def fork(self, mode: str = "shared", *, task: Task | None = None) -> _Session:
        return _Session(task=task or self.task, specs=self.specs)

    async def close(self) -> None:
        return None


async def _spawn_one_verified(ctx: Any) -> AgentResult:
    if ctx.depth == 0:
        await ctx.agents.spawn("child", max_steps=3)
        return AgentResult("completed", final_message="root done")
    if ctx.role == VERIFIER_ROLE:
        return AgentResult("completed", final_message=json.dumps({"status": "verified", "score": 1.0}))
    return AgentResult("completed", final_message="child done")


def _listing(listed: dict[str, list[str]]) -> Any:
    """``_spawn_one_verified``, recording the names of the tools each agent lists, by its role."""

    async def run(ctx: Any) -> AgentResult:
        listed[ctx.role] = [tool.name for tool in await ctx.list_tools()]
        return await _spawn_one_verified(ctx)

    return run


def _run_verified_child(
    specs: list[ToolSpec], *, role_tools: dict[str, list[str]] | None = None, fn: Any = _spawn_one_verified
) -> None:
    gateway = ModelGateway(sampler=None, renderer=_Renderer(), tokenizer=None, model_name="toy")  # type: ignore[arg-type]
    gateway.limit_tools(role_tools or {})
    runtime = InlineRuntime(gateway=gateway, base_url=None, agents=AgentTreeConfig(child_grading="verifier"))
    root = gateway.open_agent(rollout_id="r", trajectory_id="root", interactions={}, max_steps=50)
    session = _Session(task=Task(goal="root"), specs=specs)
    harness = FunctionHarness(name="verified", version="1", runtime="inline", executes_model_code=False, fn=fn)
    ctx = InlineAgentContext(runtime, root, task=session.task, session=session, harness=harness, instruction="root")
    asyncio.run(ctx.agents.spawn("child", max_steps=3))


def test_a_verifier_that_sees_no_tools_is_reported_once(monkeypatch, caplog):
    monkeypatch.setattr(agents_module, "_WARNED_TOOLLESS_VERIFIERS", set())
    act = ToolSpec("act", "Change the state.", {"type": "object", "properties": {}})  # not read-only

    with caplog.at_level(logging.WARNING, logger="platoon.harness.agents"):
        _run_verified_child([act])
        _run_verified_child([act])

    warnings = [record.getMessage() for record in caplog.records if "see no tools" in record.getMessage()]
    assert len(warnings) == 1
    assert "Verifier agents on _Session see no tools with read_only access" in warnings[0]


def test_a_verifier_with_a_read_only_tool_is_not_reported(monkeypatch, caplog):
    monkeypatch.setattr(agents_module, "_WARNED_TOOLLESS_VERIFIERS", set())
    look = ToolSpec("look", "Look at the state.", {"type": "object", "properties": {}}, read_only=True)

    with caplog.at_level(logging.WARNING, logger="platoon.harness.agents"):
        _run_verified_child([look])

    assert not [record for record in caplog.records if "see no tools" in record.getMessage()]


NOTE = ToolSpec("note", "Write a note.", {"type": "object", "properties": {}})  # not read-only
LOOK = ToolSpec("look", "Look at the state.", {"type": "object", "properties": {}}, read_only=True)


def test_a_verifier_s_tools_are_not_limited_by_the_role_of_the_child_it_judges(monkeypatch):
    monkeypatch.setattr(agents_module, "_WARNED_TOOLLESS_VERIFIERS", set())
    listed: dict[str, list[str]] = {}

    _run_verified_child([NOTE, LOOK], role_tools={"subagent": ["note"], "verifier": ["look"]}, fn=_listing(listed))

    # The child may only note; its verifier, opened under it with read-only access, looks all the same.
    assert listed == {"subagent": ["note"], "verifier": ["look"]}


def test_a_verifier_whose_roles_leave_it_no_tool_is_reported_with_what_they_allow(monkeypatch, caplog):
    monkeypatch.setattr(agents_module, "_WARNED_TOOLLESS_VERIFIERS", set())
    listed: dict[str, list[str]] = {}

    with caplog.at_level(logging.WARNING, logger="platoon.harness.agents"):
        _run_verified_child([NOTE, LOOK], role_tools={"policy": ["note"], "verifier": ["look"]}, fn=_listing(listed))

    assert listed == {"subagent": ["note"], "verifier": []}  # the root's list (its role, policy) holds below it
    [warning] = [record.getMessage() for record in caplog.records if "see no tools" in record.getMessage()]
    assert (
        "see no tools with read_only access and the tools their roles allow (policy: note; verifier: look)" in warning
    )
    assert warning.endswith("or other tools (agents.roles.<role>.tools)")
