"""ORS environments in process (``python://``): tasks, tools, results, grading, forks and teardown."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Literal

import ors_toys
import pytest
from pydantic import BaseModel, ConfigDict, Field

from platoon.envs.ors import OrsEnvironmentService, OrsSession, TaskRef, pool_env_var, select_session_url
from platoon.envs.ors.pool import resolve_session_urls
from platoon.envs.ors.sdk import normalize_schema
from platoon.envs.ors.session import EnvBusyError
from platoon.envs.service import (
    EnvironmentService,
    EnvSession,
    EnvSetupError,
    Outcome,
    ToolResult,
    ToolSpec,
    get_service,
)
from platoon.task import Task
from platoon.utils.stopping import CleanupScope, cleanups

pytestmark = pytest.mark.asyncio

COUNTER = "python://ors_toys:CounterEnv"


@pytest.fixture(autouse=True)
def _clear_events():
    ors_toys.EVENTS.clear()


async def _open(uri: str = COUNTER, task_id: str = "train:0", **options) -> OrsSession:
    service = get_service(uri, **options)
    return await service.open(await service.get_task(task_id), rollout_id="rollout-1")


def _text(result: ToolResult) -> str:
    return result.as_text()


async def test_python_uri_builds_an_in_process_service_with_index_task_ids():
    service = get_service(COUNTER)

    assert isinstance(service, OrsEnvironmentService) and isinstance(service, EnvironmentService)
    assert not service.is_remote and service.name == "CounterEnv"
    assert await service.list_splits() == ["train", "test"]
    assert await service.list_tasks("test") == ["test:0", "test:1", "test:2"]
    task = await service.get_task("test:2")
    assert task.id == "test:2" and task.goal is None
    assert task.misc["ors"] == {"split": "test", "index": 2, "name": "test-2"}
    with pytest.raises(ValueError, match="no split 'dev'"):
        await service.list_tasks("dev")


async def test_tasks_resolve_by_name_and_task_reference_and_selection_narrows_listing():
    service = OrsEnvironmentService(ors_toys.CounterEnv, split="train")

    by_name = await service.get_task("train-1")
    assert by_name.misc["ors"] == {"split": "train", "index": 1, "name": "train-1"}
    raw = json.dumps({"environment": "counter", "index": 2, "split": "test", "name": "test-2"}).encode()
    reference = "openreward:v1:" + base64.urlsafe_b64encode(raw).decode().rstrip("=")
    assert (await service.get_task(reference)).misc["ors"]["index"] == 2
    with pytest.raises(ValueError, match="No task named 'nope'"):
        await service.get_task("nope")
    assert await OrsEnvironmentService(ors_toys.CounterEnv, task_indices=[2, 0]).list_tasks("train") == [
        "train:2",
        "train:0",
    ]
    assert await OrsEnvironmentService(ors_toys.CounterEnv, task_names=["train-1"]).list_tasks("train") == ["train:1"]
    with pytest.raises(ValueError, match="outside"):
        await OrsEnvironmentService(ors_toys.CounterEnv, task_indices=[7]).list_tasks("train")
    assert TaskRef.parse("train:3") == TaskRef("train", 3) and TaskRef.parse("plain-name") is None


@pytest.mark.parametrize("parent_first", [True, False], ids=["parent-first", "subclass-first"])
async def test_a_subclass_reads_its_own_tasks_whichever_class_lists_first(parent_first):
    shelf, tall = ors_toys.shelf_envs()
    expected = {shelf: ["book-0", "book-1", "book-2"], tall: [f"tall-{index}" for index in range(5)]}
    for cls in (shelf, tall) if parent_first else (tall, shelf):
        service = OrsEnvironmentService(cls)
        names = expected[cls]
        assert await service.list_tasks("train") == [f"train:{index}" for index in range(len(names))]
        task = await service.get_task(f"train:{len(names) - 1}")
        assert task.misc["ors"]["name"] == names[-1]
        session = await service.open(task, rollout_id="r")
        try:
            assert session.instruction == f"Read {names[-1]}."
        finally:
            await session.close()


def _reference(payload: object) -> str:
    raw = json.dumps(payload).encode()
    return "openreward:v1:" + base64.urlsafe_b64encode(raw).decode().rstrip("=")


async def test_a_labeled_service_resolves_only_its_own_task_references():
    labeled = OrsEnvironmentService(ors_toys.CounterEnv, label="counter")
    own = _reference({"environment": "counter", "index": 1, "split": "test"})
    assert (await labeled.get_task(own)).misc["ors"] == {"split": "test", "index": 1, "name": "test-1"}

    other = _reference({"environment": "toolathlon", "index": 1, "split": "test"})
    with pytest.raises(ValueError, match="for environment 'toolathlon', not 'counter'"):
        await labeled.get_task(other)
    with pytest.raises(EnvSetupError, match="toolathlon") as refused:
        await labeled.open(Task(id=other), rollout_id="r")
    assert refused.value.failure_kind == "env_config"
    # Without a label (an in-process class nobody named) a reference resolves by split and index alone.
    assert (await OrsEnvironmentService(ors_toys.CounterEnv).get_task(other)).misc["ors"]["index"] == 1


@pytest.mark.parametrize(
    "reference",
    [
        _reference({"environment": "counter", "index": 1}),
        _reference({"environment": "counter", "index": 1, "split": "test", "extra": 1}),
        _reference({"environment": "", "index": 1, "split": "test"}),
        _reference({"environment": "counter", "index": 1, "split": "test", "name": ""}),
        _reference({"environment": "counter", "index": 1, "split": "test", "name": 3}),
        _reference({"environment": "counter", "index": -1, "split": "test"}),
        _reference({"environment": "counter", "index": True, "split": "test"}),
        _reference(["counter", 1, "test"]),
        "openreward:v1:",
        "openreward:v1:not+base64!",
    ],
)
async def test_malformed_task_references_are_refused(reference):
    with pytest.raises(ValueError, match="Malformed OpenReward task reference"):
        await OrsEnvironmentService(ors_toys.CounterEnv, label="counter").get_task(reference)


async def test_open_reads_the_instruction_from_get_prompt():
    session = await _open(task_id="train:1")
    try:
        assert isinstance(session, EnvSession) and session.sandbox is None
        assert session.instruction == session.prompt == "Reach 3." == session.task.goal
        assert session.task.misc["ors"]["index"] == 1
        assert ors_toys.EVENTS == ["setup:CounterEnv"]
    finally:
        await session.close()
    await session.close()  # idempotent
    assert ors_toys.EVENTS == ["setup:CounterEnv", "teardown:CounterEnv"]


async def test_tools_carry_schemas_hints_and_meta():
    session = await _open()
    try:
        tools = {spec.name: spec for spec in await session.tools()}
    finally:
        await session.close()

    assert set(tools) == {"add", "peek", "search", "submit", "give_up", "flag_error", "explode"}
    add = tools["add"]
    assert add.description == "Add to the counter."
    assert add.input_schema == {
        "type": "object",
        "properties": {"amount": {"type": "integer", "description": "How much to add to the counter."}},
        "required": ["amount"],
    }
    assert add.meta == {"example.dev/kind": "mutation"} and not add.terminal and not add.read_only
    assert tools["peek"].input_schema == {"type": "object", "properties": {}} and tools["peek"].read_only
    assert tools["submit"].terminal and not tools["submit"].read_only
    search = tools["search"]
    assert set(search.input_schema["properties"]) == {"query", "title"}  # "title" is a property, not a keyword
    assert search.input_schema["x-openhands-tool-routing"] == {"lane": "search"}
    assert search.meta == {"provider.dev/source": "docs", "openhands.dev/tool-routing": {"lane": "search"}}


class _Inner(BaseModel):
    title: str = "untitled"


class _Cat(BaseModel):
    title: Literal["cat"]
    meows: int = 1


class _Dog(BaseModel):
    title: Literal["dog"]
    barks: int = 1


class _TitledParams(BaseModel):
    model_config = ConfigDict(json_schema_extra={"x-openhands-tool-routing": {"title": "docs", "lane": "search"}})

    opts: dict = {"title": 1}
    inner: _Inner = _Inner(title="kept")
    pet: _Cat | _Dog = Field(discriminator="title")
    title: str = ""
    tags: list[_Inner] = []


async def test_title_keywords_are_dropped_only_at_schema_positions():
    schema = normalize_schema(_TitledParams.model_json_schema())

    assert "title" not in schema and "title" not in schema["$defs"]["_Cat"]
    properties = schema["properties"]
    assert set(properties) == {"opts", "inner", "pet", "title", "tags"}  # a property named "title" stays
    assert properties["opts"] == {"additionalProperties": True, "default": {"title": 1}, "type": "object"}
    assert properties["inner"] == {"$ref": "#/$defs/_Inner", "default": {"title": "kept"}}
    assert properties["pet"]["discriminator"] == {
        "mapping": {"cat": "#/$defs/_Cat", "dog": "#/$defs/_Dog"},
        "propertyName": "title",
    }
    assert properties["tags"] == {"default": [], "items": {"$ref": "#/$defs/_Inner"}, "type": "array"}
    assert schema["$defs"]["_Cat"]["properties"]["title"] == {"const": "cat", "type": "string"}
    assert schema["$defs"]["_Inner"] == {
        "properties": {"title": {"default": "untitled", "type": "string"}},
        "type": "object",
    }
    assert schema["x-openhands-tool-routing"] == {"title": "docs", "lane": "search"}  # extension data is verbatim
    assert normalize_schema(None) == {"type": "object", "properties": {}}


async def test_service_hints_and_routing_overrides_add_to_the_class_hints():
    session = await _open(
        terminal_tools=["give_up"],
        read_only_tools=["flag_error"],
        tool_meta={"peek": {"readOnlyHint": True}},
        tool_routing_overrides={"search": {"lane": "override"}},
    )
    try:
        tools = {spec.name: spec for spec in await session.tools()}
    finally:
        await session.close()
    assert tools["give_up"].terminal and tools["flag_error"].read_only
    assert tools["peek"].meta == {"readOnlyHint": True}
    assert tools["search"].meta["openhands.dev/tool-routing"] == {"lane": "override"}


async def test_tool_spec_and_result_defaults_keep_existing_callers_working():
    spec = ToolSpec("bash", "Run a command.", {"type": "object"})
    assert (spec.terminal, spec.read_only, spec.meta) == (False, False, None)
    result = ToolResult(content=[{"type": "text", "text": "a"}, {"type": "image", "mimeType": "image/png"}])
    assert result.metadata == {} and result.as_text() == "a\n[image/png]"


async def test_call_tool_maps_tool_outputs_and_errors():
    session = await _open()
    try:
        added = await session.call_tool("add", {"amount": 2})
        assert (added.content, added.metadata, added.is_error, added.finished) == (
            [{"type": "text", "text": "total=2"}],
            {"total": 2},
            False,
            False,
        )
        peeked = await session.call_tool("peek", {})
        assert peeked.content[1] == {"type": "image", "data": "aGk=", "mimeType": "image/png"}

        invalid = await session.call_tool("add", {"amount": "many"})
        assert invalid.is_error and "validation" in _text(invalid)
        unknown = await session.call_tool("nope", {})
        assert unknown.is_error and "not a valid tool" in _text(unknown)
        flagged = await session.call_tool("flag_error", {})
        assert flagged.is_error and _text(flagged) == "bad state"
        assert (await session.call_tool("add", ["not", "a", "mapping"])).is_error  # type: ignore[arg-type]
        raised = await session.call_tool("explode", {})
        assert raised.is_error and _text(raised) == "RuntimeError: boom"
        assert raised.metadata == {"error_reason": "tool_raised", "error_type": "RuntimeError"}
    finally:
        await session.close()


async def test_bridge_json_results_render_as_bridge_json():
    session = await _open(result_format="bridge_json")
    try:
        added = await session.call_tool("add", {"amount": 2})
        assert json.loads(_text(added)) == {
            "finished": False,
            "reward": None,
            "text": "total=2",
            "blocks": ["total=2"],
            "metadata": {"total": 2},
        }
        flagged = await session.call_tool("flag_error", {})
        assert flagged.is_error and _text(flagged) == "Error executing tool flag_error: try again"
    finally:
        await session.close()


@pytest.mark.parametrize(
    ("calls", "expected"),
    [
        ([("add", {"amount": 2}), ("submit", {"answer": 2})], ("scored", 1.0, None)),
        ([("submit", {"answer": 5})], ("scored", 0.0, None)),
        ([], ("scored", 0.0, "no_submit")),
    ],
)
async def test_grade_scores_the_terminal_reward_or_no_submit(calls, expected):
    session = await _open()
    try:
        for name, arguments in calls:
            await session.call_tool(name, arguments)
        outcome = await session.grade()
    finally:
        await session.close()
    assert (outcome.status, outcome.reward, outcome.reason) == expected
    assert outcome.metadata["tool_calls"] == len(calls)


async def test_grade_fails_closed():
    session = await _open()
    try:
        await session.call_tool("submit", {"answer": -1})
        invalid = await session.grade()
    finally:
        await session.close()
    assert (invalid.status, invalid.failure_kind) == ("invalid", "env_invalid")

    # A tool that raised on the live env is the policy's failed call: no submit still scores 0.
    session = await _open()
    try:
        await session.call_tool("explode", {})
        no_submit = await session.grade()
    finally:
        await session.close()
    assert (no_submit.status, no_submit.reward, no_submit.reason) == ("scored", 0.0, "no_submit")
    assert no_submit.metadata["errors"] == ["explode: RuntimeError: boom"]

    # A terminal tool that raised never scores 0: the submit itself crashed.
    session = await _open(terminal_tools=["explode"])
    try:
        await session.call_tool("explode", {})
        error = await session.grade()
    finally:
        await session.close()
    assert (error.status, error.failure_kind, error.trainable) == ("error", "env_error", False)
    assert "boom" in error.reason

    session = await _open()
    try:
        await session.call_tool("give_up", {})  # finished, but no reward
        no_reward = await session.grade()
    finally:
        await session.close()
    assert (no_reward.status, no_reward.failure_kind) == ("invalid", "no_reward")
    closed = await get_service(COUNTER).open(Task(id="train:0"), rollout_id="r")  # resolved from the id alone
    assert closed.instruction == "Reach 2."
    await closed.close()
    assert (await closed.grade()).failure_kind == "session_closed"
    assert (await closed.call_tool("add", {"amount": 1})).is_error


async def test_a_tool_that_ended_an_episode_counts_as_terminal_when_the_env_declares_none():
    # CraftEnv declares no terminal tool, so `done` is terminal once it has ended an episode.
    service = OrsEnvironmentService(ors_toys.CraftEnv)
    session = await service.open(await service.get_task("train:0"), rollout_id="r")
    try:
        assert not {spec.name: spec for spec in await session.tools()}["done"].terminal
        assert (await session.call_tool("done", {})).finished
        assert {spec.name: spec for spec in await session.tools()}["done"].terminal
    finally:
        await session.close()


@pytest.mark.parametrize(
    ("uri", "options"),
    [(COUNTER, {}), ("python://ors_toys:CraftEnv", {"terminal_tools": ["craft"]})],
    ids=["class-attribute", "option"],
)
async def test_an_env_that_declares_its_terminal_tools_keeps_its_declaration(uri, options):
    # CounterEnv declares `submit`; give_up ending an episode does not make it terminal for later listings.
    # CraftEnv with the terminal_tools option likewise keeps `done` listed for no_terminal agents.
    tool = "give_up" if uri == COUNTER else "done"
    service = get_service(uri, **options)
    session = await service.open(await service.get_task("train:0"), rollout_id="r")
    try:
        assert (await session.call_tool(tool, {})).finished
        listed = {spec.name: spec for spec in await session.tools()}
        assert not listed[tool].terminal
        assert listed["submit" if uri == COUNTER else "craft"].terminal
    finally:
        await session.close()


async def test_platoon_grade_grades_state_when_nothing_was_submitted():
    session = await _open("python://ors_toys:GradedCounterEnv")
    try:
        await session.call_tool("add", {"amount": 3})
        outcome = await session.grade()
    finally:
        await session.close()
    assert (outcome.status, outcome.reward) == ("scored", 0.3)
    assert outcome.metadata["graded_by"] == "platoon_grade"


async def test_max_tool_calls_returns_the_bridge_payload_and_latches():
    session = await _open(max_tool_calls=2)
    try:
        await session.call_tool("add", {"amount": 1})
        await session.call_tool("add", {"amount": 1})
        capped = await session.call_tool("add", {"amount": 1})
        assert json.loads(_text(capped)) == {
            "bridge_error": "max_tool_calls_exceeded",
            "finished": True,
            "reward": None,
        }
        assert capped.finished and capped.is_error
        after = await session.call_tool("submit", {"answer": 2})
        assert json.loads(_text(after))["bridge_error"] == "environment_already_finished"
        outcome = await session.grade()
    finally:
        await session.close()
    assert (outcome.status, outcome.reward, outcome.reason) == ("scored", 0.0, "max_tool_calls")


async def test_bridge_tools_and_jsonl_log(tmp_path):
    session = await _open(bridge_tools=True, log_dir=tmp_path, result_format="bridge_json")
    try:
        names = [spec.name for spec in await session.tools()]
        assert names[:2] == ["get_task", "get_status"]
        task = json.loads(_text(await session.call_tool("get_task", {})))
        assert task["prompt"] == "Reach 2." and task["task_index"] == 0 and task["task_name"] == "train-0"
        assert "call the terminal environment tool" in task["policy"]
        await session.call_tool("add", {"amount": 2})
        status = json.loads(_text(await session.call_tool("get_status", {})))
        assert (status["turn"], status["finished"], status["last_reward"]) == (1, False, None)
    finally:
        await session.close()
    log_dir = tmp_path / "rollout-1"
    events = [json.loads(line) for line in (log_dir / "bridge_events.jsonl").read_text().splitlines()]
    assert [event["type"] for event in events] == [
        "session_started",
        "task_requested",
        "tool_call",
        "tool_result",
        "session_closing",
    ]
    assert events[2]["arguments"] == {"amount": 2} and events[3]["result"]["text"] == "total=2"
    assert json.loads((log_dir / "bridge_state.json").read_text())["turn"] == 1


async def test_a_fresh_fork_records_its_own_state_beside_the_parents(tmp_path):
    log_dir = tmp_path / "rollout-1"

    def states() -> dict[str, int]:
        return {path.name: json.loads(path.read_text())["turn"] for path in log_dir.glob("bridge_state*.json")}

    parent = await _open(log_dir=tmp_path)
    try:
        for _ in range(3):
            await parent.call_tool("add", {"amount": 1})
        child = await parent.fork("fresh", task=Task(id="fresh", goal="Again."))
        child_id = child._lease.id  # type: ignore[union-attr]
        try:
            await child.call_tool("add", {"amount": 1})
            assert states() == {"bridge_state.json": 3, f"bridge_state.{child_id}.json": 1}
        finally:
            await child.close()
    finally:
        await parent.close()
    assert states() == {"bridge_state.json": 3, f"bridge_state.{child_id}.json": 1}
    sessions = [json.loads(line)["session"] for line in (log_dir / "bridge_events.jsonl").read_text().splitlines()]
    assert sessions.count(child_id) == 4  # started, call, result, closing


async def test_shared_forks_share_the_instance_and_its_accounting():
    parent = await _open(max_tool_calls=3)
    child = await parent.fork("shared", task=Task(id="child", goal="Add one."))
    try:
        assert child.instruction == "Add one." and parent.instruction == "Reach 2."
        await child.call_tool("add", {"amount": 1})
        assert _text(await parent.call_tool("add", {"amount": 1})) == "total=2"
        await child.close()
        assert ors_toys.EVENTS == ["setup:CounterEnv"]  # the parent still holds the env
        await parent.call_tool("submit", {"answer": 2})
        assert (await parent.grade()).reward == 1.0
    finally:
        await child.close()
        await parent.close()
    assert ors_toys.EVENTS == ["setup:CounterEnv", "teardown:CounterEnv"]


async def test_a_shared_child_whose_tool_raised_leaves_the_root_ungraded_by_it():
    parent = await _open()
    child = await parent.fork("shared", task=Task(id="child", goal="Help."))
    try:
        assert (await child.call_tool("explode", {})).is_error
        outcome = await parent.grade()
    finally:
        await child.close()
        await parent.close()
    assert (outcome.status, outcome.reason) == ("scored", "no_submit")
    assert outcome.metadata["errors"] == ["explode: RuntimeError: boom"]


async def test_bridge_json_errors_show_the_exceptions_plain_text():
    session = await _open(result_format="bridge_json")
    try:
        raised = await session.call_tool("explode", {})
    finally:
        await session.close()
    assert raised.is_error and _text(raised) == "Error executing tool explode: boom"
    assert raised.metadata == {"error_reason": "tool_raised", "error_type": "RuntimeError"}


async def test_platoon_fork_gives_a_child_its_own_goal_on_shared_state():
    parent = await _open("python://ors_toys:CraftEnv")
    child = await parent.fork("shared", task=Task(id="child", goal="craft 2 stick"))
    try:
        assert child.instruction == "craft 2 stick"
        await child.call_tool("craft", {"item": "stick"})
        await child.call_tool("craft", {"item": "stick"})
        await child.call_tool("done", {})
        assert (await child.grade()).reward == 1.0  # the child's own goal
        assert _text(await parent.call_tool("craft", {"item": "stick"})) == "stick: 3"  # shared inventory
        assert (await parent.grade()).reason == "no_submit"  # the child's done() is not the parent's
        await parent.call_tool("craft", {"item": "plank"})
        assert (await child.call_tool("craft", {"item": "x"})).finished  # the child's episode is over
    finally:
        await child.close()
        await parent.close()


class _CopyableCounterEnv(ors_toys.CounterEnv):
    """A counter that copies itself for a copy fork: CounterEnv has a teardown, so it is not deep-copied."""

    def platoon_copy(self) -> _CopyableCounterEnv:
        clone = _CopyableCounterEnv(self.task_spec, {"token": self.secret} if self.secret else {})
        clone.total = self.total
        return clone


async def _cancel_soon(close) -> None:
    """Start ``close()`` and cancel it after one loop iteration, while it is releasing its env."""

    closing = asyncio.create_task(close())
    await asyncio.sleep(0)
    closing.cancel()
    started = time.monotonic()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert time.monotonic() - started < 1.0  # the cancellation propagates promptly


def _teardowns() -> int:
    return ors_toys.EVENTS.count("teardown:TrackedCraftEnv")


async def test_a_cancelled_close_of_a_platoon_fork_child_still_releases_its_parent():
    parent = await _open("python://ors_toys:TrackedCraftEnv")
    child = await parent.fork("shared", task=Task(id="child", goal="craft 1 stick"))
    await _cancel_soon(child.close)
    await parent.close()
    await _until(lambda: _teardowns() == 1)
    await asyncio.sleep(0.1)
    assert _teardowns() == 1


async def test_cancelled_closes_down_a_chain_of_platoon_fork_children_release_the_root_once():
    root = await _open("python://ors_toys:TrackedCraftEnv")
    child = await root.fork("shared", task=Task(id="child", goal="craft 1 stick"))
    grandchild = await child.fork("shared", task=Task(id="grandchild", goal="craft 1 plank"))
    await _cancel_soon(grandchild.close)
    await _cancel_soon(child.close)
    assert _teardowns() == 0
    await root.close()
    await _until(lambda: _teardowns() == 1)
    await asyncio.sleep(0.1)
    assert _teardowns() == 1


async def test_a_cancelled_last_close_still_tears_the_root_down():
    parent = await _open("python://ors_toys:TrackedCraftEnv")
    child = await parent.fork("shared", task=Task(id="child", goal="craft 1 stick"))
    await parent.close()
    assert _teardowns() == 0  # the child still holds it
    await _cancel_soon(child.close)
    await _until(lambda: _teardowns() == 1)


async def test_closing_the_parent_while_its_platoon_fork_hook_runs_defers_its_teardown():
    parent = await _open("python://ors_toys:SlowForkCraftEnv")
    forking = asyncio.create_task(parent.fork("shared", task=Task(id="child", goal="craft 1 stick")))
    await asyncio.sleep(0.1)  # the sync hook runs in its thread
    await parent.close()
    child = await forking
    try:
        await asyncio.sleep(0.2)
        assert _teardowns() == 0  # the child holds the env it forked from
        assert _text(await child.call_tool("craft", {"item": "stick"})) == "stick: 1"
    finally:
        await child.close()
    assert _teardowns() == 1


async def test_closing_a_chain_while_a_grandchild_forks_tears_the_root_down_once():
    root = await _open("python://ors_toys:SlowForkCraftEnv")
    child = await root.fork("shared", task=Task(id="child", goal="craft 1 stick"))
    forking = asyncio.create_task(child.fork("shared", task=Task(id="grandchild", goal="craft 1 plank")))
    await asyncio.sleep(0.1)
    await child.close()
    await root.close()
    grandchild = await forking
    await asyncio.sleep(0.2)
    assert _teardowns() == 0  # the grandchild still holds the chain down to the root
    assert _text(await grandchild.call_tool("craft", {"item": "plank"})) == "plank: 1"
    await grandchild.close()
    await asyncio.sleep(0.1)
    assert _teardowns() == 1


async def test_a_platoon_fork_hook_that_returns_no_env_leaves_the_parent_as_it_was():
    parent = await _open("python://ors_toys:SlowForkCraftEnv")
    try:
        with pytest.raises(TypeError, match="must return an Environment"):
            await parent.fork("shared", task=Task(id="child", goal="nothing"))
        assert parent._lease._holders == 1  # type: ignore[union-attr]
    finally:
        await parent.close()
    assert _teardowns() == 1


async def test_a_cancelled_platoon_fork_lets_go_of_the_parent_whose_teardown_waits_for_the_hook():
    parent = await _open("python://ors_toys:SlowForkCraftEnv")
    forking = asyncio.create_task(parent.fork("shared", task=Task(id="child", goal="craft 1 stick")))
    await asyncio.sleep(0.1)
    forking.cancel()
    with pytest.raises(asyncio.CancelledError):
        await forking
    # No child will hold the parent; the hook still runs on its env, an operation in flight the teardown waits for.
    assert parent._lease._holders == 1  # type: ignore[union-attr]
    await parent.close()
    assert ors_toys.EVENTS == ["fork:done", "teardown:TrackedCraftEnv"]


async def test_fresh_and_copy_forks_get_their_own_env():
    service = OrsEnvironmentService(_CopyableCounterEnv)
    parent = await service.open(await service.get_task("train:0"), rollout_id="rollout-1")
    try:
        await parent.call_tool("add", {"amount": 2})
        fresh = await parent.fork("fresh", task=Task(id="fresh", goal="Again."))
        copied = await parent.fork("copy")
        try:
            assert fresh.instruction == "Again." and copied.instruction == "Reach 2."
            assert _text(await fresh.call_tool("add", {"amount": 1})) == "total=1"  # a new instance
            assert _text(await copied.call_tool("add", {"amount": 1})) == "total=3"  # a snapshot of the parent
            assert _text(await parent.call_tool("add", {"amount": 0})) == "total=2"  # untouched by either
            await copied.call_tool("submit", {"answer": 3})
            assert (await copied.grade()).reward == 0.0 and (await parent.grade()).reason == "no_submit"
        finally:
            await fresh.close()
            await copied.close()
    finally:
        await parent.close()
    assert ors_toys.EVENTS.count("teardown:_CopyableCounterEnv") == 3
    with pytest.raises(ValueError, match="Unknown fork mode"):
        await (await _open()).fork("sideways")  # type: ignore[arg-type]


async def test_an_env_with_a_teardown_is_not_deep_copied():
    service = get_service("python://ors_toys:WorkdirEnv")
    parent = await service.open(await service.get_task("train:0"), rollout_id="r")
    try:
        with pytest.raises(NotImplementedError, match="define platoon_copy"):
            await parent.fork("copy")
        await parent.call_tool("write", {"answer": "42"})
        await parent.call_tool("submit", {})
        assert (await parent.grade()).reward == 1.0  # the refused copy cost the parent nothing
    finally:
        await parent.close()
    assert len(ors_toys.EVENTS) == 1  # torn down once


@pytest.mark.parametrize("first", ["parent", "copy"])
async def test_a_platoon_copy_owns_its_resources(first):
    service = get_service("python://ors_toys:CopyableWorkdirEnv")
    parent = await service.open(await service.get_task("train:0"), rollout_id="r")
    copied = None
    try:
        await parent.call_tool("write", {"answer": "41"})
        copied = await parent.fork("copy")
        await copied.call_tool("write", {"answer": "42"})  # the copy writes to its own workdir
        closing, other = (parent, copied) if first == "parent" else (copied, parent)
        await closing.close()
        assert len(ors_toys.EVENTS) == 1
        await other.call_tool("submit", {})
        assert (await other.grade()).reward == (1.0 if other is copied else 0.0)
    finally:
        await parent.close()
        if copied is not None:
            await copied.close()
    assert len(ors_toys.EVENTS) == len(set(ors_toys.EVENTS)) == 2  # each torn down once


async def test_a_copy_of_a_platoon_fork_view_is_never_torn_down():
    parent = await _open("python://ors_toys:CraftEnv")
    child = await parent.fork("shared", task=Task(id="child", goal="craft 1 stick"))
    copied = await child.fork("copy")
    try:
        assert copied._lease.backend.owned is False  # type: ignore[union-attr]
        copied._lease.backend.env.teardown = lambda: ors_toys.EVENTS.append("copy-teardown")  # type: ignore[union-attr]
        await copied.call_tool("craft", {"item": "stick"})
        assert _text(await child.call_tool("craft", {"item": "stick"})) == "stick: 1"  # the copy is independent
    finally:
        await copied.close()
        await child.close()
        await parent.close()
    assert ors_toys.EVENTS == []


async def test_a_copy_fork_waits_for_the_calls_in_flight():
    parent = await _open("python://ors_toys:TallyEnv")
    sibling = await parent.fork("shared", task=Task(id="sibling", goal="Help."))
    copied = None
    try:
        # One call holds the env's own lock across an await while the other waits for it, which also binds the
        # lock to this loop; the copy is asked for in the middle.
        calls = [asyncio.create_task(view.call_tool("slow_add", {"amount": 1})) for view in (parent, sibling)]
        await asyncio.sleep(0.1)
        copied = await asyncio.wait_for(parent.fork("copy"), 10)
        # The fork waited for both: their results are on the lease, and their callers get them now.
        assert sorted(_text(reply) for reply in await asyncio.gather(*calls)) == ["total=1", "total=2"]
        env = copied._lease.backend.env  # type: ignore[union-attr]
        assert (env.lock.locked(), env.total, env.added) == (False, 2, [1, 1])  # whole calls, a free lock
        assert copied._lease.turn == 2  # type: ignore[union-attr]
        reply = await asyncio.wait_for(copied.call_tool("slow_add", {"amount": 1}), 10)
        assert _text(reply) == "total=3" and parent._lease.backend.env.total == 2  # type: ignore[union-attr]
    finally:
        for session in (copied, sibling, parent):
            if session is not None:
                await session.close()


async def test_a_copy_fork_waits_for_a_siblings_sync_tool():
    parent = await _open("python://ors_toys:TallyEnv")
    sibling = await parent.fork("shared", task=Task(id="sibling", goal="Help."))
    try:
        filling = asyncio.create_task(sibling.call_tool("fill", {"amount": 20_000}))
        await asyncio.sleep(0.05)
        copied = await asyncio.wait_for(parent.fork("copy"), 10)  # never "dictionary changed size"
        try:
            assert _text(await filling) == "rows=20000"  # the fork waited for it: the copy has every row
            assert len(copied._lease.backend.env.table) == 20_000  # type: ignore[union-attr]
        finally:
            await copied.close()
    finally:
        await sibling.close()
        await parent.close()


async def test_a_copy_fork_gives_up_on_an_env_that_stays_busy():
    parent = await _open("python://ors_toys:TallyEnv", queue_timeout_s=0.2)  # how long it waits for its turn
    try:
        filling = asyncio.create_task(parent.call_tool("fill", {"amount": 200_000}))
        await asyncio.sleep(0.05)
        with pytest.raises(EnvBusyError, match="could not be copied"):
            await parent.fork("copy")
        await filling
    finally:
        await parent.close()


async def test_setup_failures_raise_env_setup_errors_and_tear_down():
    service = get_service("python://ors_toys:BrokenSetupEnv")
    with pytest.raises(EnvSetupError) as raised:
        await service.open(await service.get_task("train:0"), rollout_id="r")
    assert raised.value.failure_kind == "env_setup" and "no capacity" in str(raised.value)
    assert ors_toys.EVENTS == ["teardown:BrokenSetupEnv"]
    with pytest.raises(EnvSetupError) as bad_task:
        await get_service(COUNTER).open(Task(id="train:9"), rollout_id="r")
    assert bad_task.value.failure_kind == "env_config"


_HUNG_ENV_CALLS = """
import asyncio, threading, time
from openreward.environments import Environment, TextBlock, ToolOutput, tool
from platoon.envs.ors import OrsEnvironmentService
from platoon.envs.ors.sdk import abandoned_calls
from platoon.task import Task

NEVER = threading.Event()


class HangEnv(Environment):
    @classmethod
    def list_splits(cls):
        return ["train"]

    @classmethod
    def list_tasks(cls, split):
        return [{"hang_teardown": False}, {"hang_teardown": True}]

    def get_prompt(self):
        return [TextBlock(text="Wait.")]

    def teardown(self):
        if self.task_spec["hang_teardown"]:
            NEVER.wait()

    @tool
    def hang(self) -> ToolOutput:
        NEVER.wait()

    @tool
    def ping(self) -> ToolOutput:
        return ToolOutput(blocks=[TextBlock(text="pong")])


async def main():
    service = OrsEnvironmentService(HangEnv, close_timeout_s=0.2)
    loop = asyncio.get_running_loop()

    async def hung_tool():
        session = await service.open(Task(id="train:0"), rollout_id="tool")
        try:
            await asyncio.wait_for(session.call_tool("hang", {}), 0.2)
        except TimeoutError:
            pass
        await session.close()

    async def hung_teardown():
        session = await service.open(Task(id="train:1"), rollout_id="teardown")
        await session.close()

    await asyncio.gather(*(hung_tool() for _ in range(40)), *(hung_teardown() for _ in range(40)))
    assert abandoned_calls() >= 80, abandoned_calls()
    started = time.monotonic()
    await asyncio.wait_for(loop.getaddrinfo("localhost", 80), 10)
    assert await asyncio.wait_for(asyncio.to_thread(lambda: 7), 10) == 7
    session = await asyncio.wait_for(service.open(Task(id="train:0"), rollout_id="fresh"), 10)
    assert (await asyncio.wait_for(session.call_tool("ping", {}), 10)).as_text() == "pong"
    print(f"ok {time.monotonic() - started:.2f}", flush=True)


asyncio.run(main())
"""


async def test_abandoned_sync_env_calls_hold_only_their_own_threads(tmp_path):
    """80 hung sync tools and teardowns, all abandoned: DNS, to_thread and new sessions still work, and the
    process still exits (abandoned calls run on daemon threads, not the loop's default executor)."""

    import os
    import subprocess
    import sys

    script = tmp_path / "hung_env_calls.py"
    script.write_text(_HUNG_ENV_CALLS)
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=120, env=environment
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert any(line.startswith("ok ") for line in completed.stdout.splitlines()), completed.stdout[-3000:]
    assert time.monotonic() - started < 60


async def _until(condition, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting"
        await asyncio.sleep(0.05)


async def test_a_cancelled_sync_setup_is_torn_down_once_it_ends():
    service = get_service("python://ors_toys:SlowSetupEnv")
    opening = asyncio.create_task(service.open(await service.get_task("train:0"), rollout_id="r"))
    await asyncio.sleep(0.3)
    opening.cancel()
    with pytest.raises(asyncio.CancelledError):
        await opening
    assert ors_toys.EVENTS == []  # setup still runs in its thread: nothing is torn down under it
    await _until(lambda: len(ors_toys.EVENTS) == 2)
    assert ors_toys.EVENTS == ["setup:done", "teardown:acquired"]


async def test_a_sync_tool_whose_caller_gave_up_holds_off_teardown_until_it_returns():
    # The last close waits close_timeout_s for the call in flight, then leaves the teardown to the env's backend.
    session = await _open("python://ors_toys:SlowToolEnv", close_timeout_s=0.2)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(session.call_tool("slow", {}), 0.2)
    await session.close()
    assert ors_toys.EVENTS == ["setup:SlowToolEnv"]
    await _until(lambda: len(ors_toys.EVENTS) == 3)
    assert ors_toys.EVENTS == ["setup:SlowToolEnv", "slow:done", "teardown:SlowToolEnv"]


async def test_a_teardown_that_waits_for_a_sync_tool_joins_the_rollouts_cleanup_scope():
    """So an in-process rollout's permit waits for it; it is no cleanup that a stopping rollout worker lets finish,
    since its wait has no bound (``platoon.utils.stopping.in_cleanup_scope``)."""

    scope = CleanupScope()
    with scope:
        session = await _open("python://ors_toys:SlowToolEnv", close_timeout_s=0.2)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(session.call_tool("slow", {}), 0.2)
        await session.close()
    assert ors_toys.EVENTS == ["setup:SlowToolEnv"]
    assert [task for task in scope.running() if task not in cleanups()]  # the teardown waiting for the tool
    assert await scope.finish(10) == 0
    assert ors_toys.EVENTS == ["setup:SlowToolEnv", "slow:done", "teardown:SlowToolEnv"]


async def test_an_env_waiting_on_an_abandoned_sync_tool_is_torn_down_when_the_loop_ends():
    """A rollout worker's or an evaluation slot's loop ends with its rollout, and asyncio.run then cancels the
    background teardown: it still tears the env down, while the tool runs on.

    No step waits a short time, which a loaded runner's stall can outlast: the rollout gives up on the call once the
    tool runs, and the env's teardown keeps the default bound (``close_timeout_s``, which the env takes as it opens).
    Only the close's wait for the call is cut short, since the call never returns here."""

    async def rollout() -> None:
        service = get_service("python://ors_toys:HeldToolEnv")
        session = await service.open(await service.get_task("train:0"), rollout_id="rollout-1")
        call = asyncio.create_task(session.call_tool("held", {}))
        await _until(ors_toys.HELD.is_set)  # the tool runs in its thread
        call.cancel()  # and the rollout gives up on it
        with pytest.raises(asyncio.CancelledError):
            await call
        service.close_timeout_s = 0.01  # the close stops waiting for the call at once
        await session.close()
        assert ors_toys.EVENTS == ["setup:HeldToolEnv"]  # the teardown waits for the tool

    ors_toys.HELD.clear()
    ors_toys.RELEASE.clear()
    try:
        await asyncio.to_thread(asyncio.run, rollout())  # a loop of its own, which ends with the rollout
        assert ors_toys.EVENTS == ["setup:HeldToolEnv", "teardown:HeldToolEnv"]
    finally:
        ors_toys.RELEASE.set()
    await _until(lambda: "held:done" in ors_toys.EVENTS)  # the abandoned tool still returns, to a closed loop


async def test_a_cancelled_async_setup_is_cancelled_and_torn_down_promptly():
    service = get_service("python://ors_toys:AsyncSlowSetupEnv")
    opening = asyncio.create_task(service.open(await service.get_task("train:0"), rollout_id="r"))
    await asyncio.sleep(0.3)
    started = time.monotonic()
    opening.cancel()
    with pytest.raises(asyncio.CancelledError):
        await opening
    assert time.monotonic() - started < 1.0
    assert ors_toys.EVENTS == ["teardown:async"]  # the setup never finished


async def test_secrets_reach_the_env_constructor():
    session = await _open(secrets={"token": "s3cret"})
    try:
        assert session._lease.backend.env.secret == "s3cret"  # type: ignore[union-attr]
    finally:
        await session.close()


@pytest.mark.parametrize(("sdk_level", "leaks"), [(None, False), ("INFO", True)], ids=["quieted", "explicit-level"])
async def test_the_sdks_info_lines_stay_out_of_a_codeact_cell(monkeypatch, sdk_level, leaks):
    # The SDK logs tool_call_completed at INFO on stdout after every tool call. When structlog is first imported
    # while a CodeAct cell captures stdout (a concurrent rollout's cell), its loggers print through the capture
    # proxy, and a tool call made from a cell put the line in the cell's output.
    import logging
    import sys

    import structlog
    from openreward.environments import environment as sdk_environment

    from platoon.harness.builtin.codeact.ipython_shell import ShellCapture

    with ShellCapture():
        proxy = sys.stdout  # what structlog binds as stdout when first imported during a capture
    sdk_logger = structlog.make_filtering_bound_logger(logging.INFO)(
        structlog.PrintLogger(proxy), [structlog.processors.KeyValueRenderer()], {}
    )
    monkeypatch.setattr(sdk_environment, "logger", sdk_logger)  # the SDK's default level
    if sdk_level is None:
        monkeypatch.delenv("OPENREWARD_LOG_LEVEL", raising=False)
    else:
        monkeypatch.setenv("OPENREWARD_LOG_LEVEL", sdk_level)

    session = await _open()  # loading the env in process quiets the SDK unless its level is set explicitly
    try:
        with ShellCapture() as cell:
            result = await session.call_tool("add", {"amount": 2})
            sdk_environment.logger.warning("still_reported")
        output = cell.pop_stdout()
    finally:
        await session.close()

    assert _text(result) == "total=2"
    assert ("tool_call_completed" in output) is leaks
    assert "still_reported" in output  # warnings still print


async def test_sdk_terminal_tools_are_listed_and_marked():
    environments = pytest.importorskip("openreward.environments")
    if not hasattr(environments, "terminal"):
        pytest.skip("this openreward SDK has no @terminal")

    class QuizEnv(ors_toys.CounterEnv):
        platoon_terminal_tools = ()

        # Annotations name module globals: the SDK resolves them with get_type_hints.
        @environments.terminal
        @environments.tool
        async def respond(self, params: ors_toys.QuizAnswer) -> ors_toys.ToolOutput:
            """Answer the quiz."""
            reward = float(params.answer == "4")
            return environments.ToolOutput(blocks=[environments.TextBlock(text="ok")], reward=reward, finished=True)

    session = await OrsEnvironmentService(QuizEnv).open(Task(id="train:0"), rollout_id="r")
    try:
        respond = {spec.name: spec for spec in await session.tools()}["respond"]
        assert respond.terminal and respond.meta == {"openreward.dev/terminal-tool": {"arg": "answer"}}
        assert respond.input_schema["required"] == ["answer"]
        await session.call_tool("respond", {"answer": "4"})
        assert (await session.grade()).reward == 1.0
    finally:
        await session.close()


async def test_only_sdk_terminal_tools_carry_the_message_routing_meta():
    environments = pytest.importorskip("openreward.environments")
    if not hasattr(environments, "terminal"):
        pytest.skip("this openreward SDK has no @terminal or exclusive tools")

    class ExamEnv(ors_toys.CounterEnv):
        platoon_terminal_tools = ()

        @environments.tool(exclusive=True)
        async def hand_in(self, params: ors_toys.HandIn) -> ors_toys.ToolOutput:
            """Hand in an answer with a note; this ends the episode."""
            reward = float(params.answer == self.total == self.task_spec["target"])
            return environments.ToolOutput(blocks=[environments.TextBlock(text="ok")], reward=reward, finished=True)

        @environments.terminal
        @environments.tool
        async def respond(self, params: ors_toys.QuizAnswer) -> ors_toys.ToolOutput:
            """Answer the quiz."""
            reward = float(params.answer == "4")
            return environments.ToolOutput(blocks=[environments.TextBlock(text="ok")], reward=reward, finished=True)

    session = await OrsEnvironmentService(ExamEnv).open(Task(id="train:0"), rollout_id="r")
    try:
        tools = {spec.name: spec for spec in await session.tools()}
        # An exclusive submit tool is terminal for access filtering but takes no final message.
        assert tools["hand_in"].terminal and tools["hand_in"].meta is None
        assert set(tools["hand_in"].input_schema["properties"]) == {"answer", "note"}
        assert tools["respond"].terminal and tools["respond"].meta == {
            "openreward.dev/terminal-tool": {"arg": "answer"}
        }
        await session.call_tool("add", {"amount": 2})
        await session.call_tool("hand_in", {"answer": 2, "note": "done"})
        assert (await session.grade()).reward == 1.0
    finally:
        await session.close()


async def test_uri_parsing_and_validation():
    remote = OrsEnvironmentService.from_uri("ors+http://a:1,b:2/owner/Env?variant=v&split=test")
    assert (remote.env_name, remote.urls, remote.variant, remote.split) == (
        "owner/Env",
        ["http://a:1", "http://b:2"],
        "v",
        "test",
    )
    assert remote.is_remote and remote.label == "owner/Env"
    assert OrsEnvironmentService.from_uri("ors+https://h:3/E").urls == ["https://h:3"]
    assert OrsEnvironmentService.from_uri("ors+http:///E", label="x").urls == []
    with pytest.raises(ValueError, match="names no environment"):
        OrsEnvironmentService.from_uri("ors+http://h:1/")
    with pytest.raises(ValueError, match="Unknown query"):
        OrsEnvironmentService.from_uri("ors+http://h:1/E?bogus=1")
    with pytest.raises(ValueError, match="needs an Environment class"):
        OrsEnvironmentService(ors_toys.CounterEnv, env_name="E")
    with pytest.raises(ValueError, match="max_tool_calls"):
        OrsEnvironmentService(ors_toys.CounterEnv, max_tool_calls=-1)
    harbor = get_service("harbor", dataset_dir="/nonexistent")  # module:Class and built-in names still work
    assert type(harbor).__name__ == "HarborEnvironmentService"


async def test_pools_resolve_in_precedence_order_and_shard_by_rollout_id():
    environ = {
        "OPENREWARD_SESSION_URLS_FRONTIER_CS": "h1:1, http://h2:2",
        "OPENREWARD_SESSION_URLS": "g:1",
        "OPENREWARD_SESSION_URL": "http://single:9",
    }
    assert pool_env_var("frontier-cs") == "OPENREWARD_SESSION_URLS_FRONTIER_CS"
    assert resolve_session_urls(urls=["http://static:1"], label="frontier-cs", environ=environ) == ["http://static:1"]
    assert resolve_session_urls(urls=None, label="frontier-cs", environ=environ) == ["http://h1:1", "http://h2:2"]
    assert resolve_session_urls(urls=None, label="other", environ=environ) == ["http://g:1"]
    assert resolve_session_urls(urls=None, label="other", legacy_global_pool=False, environ=environ) == [
        "http://single:9"
    ]
    assert resolve_session_urls(urls=None, label="x", env_var="MY_POOL", environ={"MY_POOL": "m:1"}) == ["http://m:1"]
    assert resolve_session_urls(urls=None, label="x", environ={}) == ["http://localhost:8080"]

    pool = [f"http://h{index}:1" for index in range(4)]
    picks = [select_session_url(pool, f"rollout-{index}") for index in range(200)]
    assert picks == [select_session_url(pool, f"rollout-{index}") for index in range(200)]  # stable per rollout
    assert set(picks) == set(pool)  # and spread over the pool
    import hashlib

    digest = int(hashlib.sha1(b"rollout-7").hexdigest(), 16)
    assert select_session_url(pool, "rollout-7") == pool[digest % 4]  # the SHA-1 rule


async def test_outcome_shapes_match_the_taxonomy():
    assert Outcome(status="scored", rewards={"reward": 0.0}, reason="no_submit").trainable
