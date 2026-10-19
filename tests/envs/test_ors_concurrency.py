"""Agents that share a live ORS env take turns on it (``platoon.envs.gate``).

Reproductions of the hazards of a shared env, each run through platoon's real paths (sessions and their forks, the
inline runtime's ``spawn``, the gateway's MCP endpoint, an ORS server) and, where it shows the hazard, also
straight into the env as nothing ordered the calls before:

* read-modify-write tools lose updates when calls overlap (sync tools on 0.1.125's semantics: no SDK lock, or
  different tool names, which 0.1.163's per-tool locks do not order);
* Toolathlon's shared MCP pipe fails a call that reads while another waits;
* a terminal call grades a solution that a sibling changes under it (Frontier-CS's finalization race);
* grading and copy snapshots read a half-applied call; a grade, a fork or a listing that cannot start in time;
* parent/child, sibling and HTTP-minted-child calls, and PTC-style gathers in one agent and from thread pools
  (TextCraft's ``platoon_fork`` children on one inventory are the TextCraft extension's test);
* a call that queues past its client's deadline still runs when its turn comes (the zombie call), unless the
  queue deadline refuses it first.

``concurrency_toys`` holds the envs and ``scripted_trees`` the scripted agent trees.  The tests pass on openreward
0.1.163 and 0.1.125.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import aiohttp
import concurrency_toys as toys
import httpx
import pytest
import pytest_asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import McpError
from scripted_trees import call, env_of, make_gateway, run_tree

from platoon.envs.gate import CallClass, EnvBusy
from platoon.envs.ors import OrsEnvironmentService, OrsSession, sdk
from platoon.envs.ors.session import Lease, LocalBackend
from platoon.gateway.core import ModelGateway
from platoon.gateway.mcp import bind_agent_tools, call_agent_tool, mcp_url, tool_call_records
from platoon.gateway.server import GatewayServer
from platoon.harness import agents as agents_module
from platoon.harness.agent import AgentResult
from platoon.harness.function import FunctionHarness
from platoon.harness.inline import InlineAgentContext
from platoon.task import Task

pytestmark = pytest.mark.asyncio


def _client_venv_python() -> Path | None:
    """A Python with the fastmcp and OpenHands SDK clients: ``$PLATOON_CLIENT_VENV/bin/python``, else this
    interpreter when it has both, else None."""

    if venv := os.environ.get("PLATOON_CLIENT_VENV"):
        return Path(venv) / "bin" / "python"
    try:
        found = all(importlib.util.find_spec(name) is not None for name in ("fastmcp", "openhands.sdk"))
    except ImportError:  # find_spec imports a dotted name's parent package
        found = False
    return Path(sys.executable) if found else None


CLIENT_VENV_PYTHON = _client_venv_python()
NO_CLIENT_VENV = (
    "needs the fastmcp and OpenHands SDK clients: run in an environment that has them, or set PLATOON_CLIENT_VENV "
    "to one"
)


@pytest.fixture(autouse=True)
def _reset_toys() -> Iterator[None]:
    toys.reset()
    yield


def service(env: str = "TrackEnv", **options: Any) -> OrsEnvironmentService:
    return OrsEnvironmentService.from_uri(f"python://concurrency_toys:{env}", **options)


async def open_session(env: str = "TrackEnv", **options: Any) -> OrsSession:
    svc = service(env, **options)
    return await svc.open(await svc.get_task("train:0"), rollout_id=f"r-{uuid.uuid4().hex[:6]}")


async def views(session: OrsSession, count: int) -> list[OrsSession]:
    """The session and ``count - 1`` shared forks of it: one view per agent."""
    return [session, *[await session.fork("shared") for _ in range(count - 1)]]


async def close(*sessions: OrsSession) -> None:
    for session in reversed(sessions):
        await session.close()


async def until(condition, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting for the condition"
        await asyncio.sleep(0.005)


def lease_of(session: OrsSession) -> Lease:
    lease = session._lease
    assert lease is not None
    return lease


async def until_running(session: OrsSession, call_class: CallClass = "serial") -> None:
    """Wait until a tool call of ``call_class`` runs on the session's lease (admitted, not just queued)."""
    await until(lambda: lease_of(session).running[call_class] > 0)


def events(log_dir: Path) -> list[dict[str, Any]]:
    (path,) = log_dir.glob("*/bridge_events.jsonl")
    return [json.loads(line) for line in path.read_text().splitlines()]


# ------------------------------------------------------------------------------------------------ sessions


async def test_calls_from_every_view_of_a_live_env_run_one_at_a_time():
    root = await open_session()
    agents = await views(root, 4)
    try:
        tools = ["slow", "slow_b", "slow_sync", "slow"]  # different names: 0.1.163's per-tool locks do not order them
        await asyncio.gather(
            *(
                agent.call_tool(tool, {"seconds": 0.03, "tag": f"v{i}"})
                for i, (agent, tool) in enumerate(zip(agents, tools))
            )
        )
        assert sorted(toys.tags()) == ["v0", "v1", "v2", "v3"] and toys.peak() == 1
        assert root._lease.turn == 4
    finally:
        await close(*agents)


async def test_sync_read_modify_write_tools_lose_no_updates():
    root = await open_session()
    agents = await views(root, 4)
    lease = root._lease
    assert lease is not None
    tools = ["incr_sync", "incr_sync_b", "incr_sync", "incr_sync_b"]
    increment = {"n": 20, "pause": 0.002}
    try:
        # Straight into the env, with nothing ordering the calls: the worker threads interleave and lose updates.
        await asyncio.gather(*(lease.backend.call(tool, dict(increment)) for tool in tools))
        assert env_of(root).state["count"] < 80
        env_of(root).state["count"] = 0
        # Through the agents' sessions: every increment lands.
        results = await asyncio.gather(*(agent.call_tool(tool, dict(increment)) for agent, tool in zip(agents, tools)))
        assert not any(result.is_error for result in results)
        assert env_of(root).state["count"] == 80
        assert (await root.call_tool("incr_async", {"n": 5, "pause": 0.001})).as_text() == "85"
    finally:
        await close(*agents)


async def test_platoon_fork_families_share_one_gate_and_copies_and_fresh_forks_get_their_own():
    root = await open_session("TrackForkEnv")
    child = await root.fork("shared", task=Task(id="child", goal="child goal"))
    grandchild = await child.fork("shared", task=Task(id="grandchild", goal="grandchild goal"))
    plain = await root.fork("shared")
    copy = await root.fork("copy")
    fresh = await root.fork("fresh")
    sessions = [root, child, grandchild, plain, copy, fresh]
    try:
        gate = root._lease.gate  # type: ignore[union-attr]
        assert child._lease.gate is gate and grandchild._lease.gate is gate and plain._lease.gate is gate  # type: ignore[union-attr]
        assert copy._lease.gate is not gate and fresh._lease.gate is not gate  # type: ignore[union-attr]
        assert copy._lease.gate is not fresh._lease.gate  # type: ignore[union-attr]
        assert env_of(child) is not env_of(root) and env_of(child).state is env_of(root).state  # shared state

        await asyncio.gather(
            root.call_tool("slow", {"seconds": 0.05, "tag": "root"}),
            child.call_tool("slow_b", {"seconds": 0.05, "tag": "child"}),
            grandchild.call_tool("slow", {"seconds": 0.05, "tag": "grandchild"}),
            copy.call_tool("slow", {"seconds": 0.05, "tag": "copy"}),
            fresh.call_tool("slow", {"seconds": 0.05, "tag": "fresh"}),
        )
        family = [run for run in toys.runs() if run[1] in ("root", "child", "grandchild")]
        assert len(family) == 3 and not toys.overlapping(family)  # one gate for the state they share
        assert toys.overlapping(toys.runs())  # the copy and the fresh env ran beside the family
    finally:
        await close(*sessions)


async def test_call_classes_follow_declarations_only():
    svc = service(terminal_tools=["slow_sync"], concurrent_tools=["slow_b"], read_only_tools=["read_count"])
    session = await svc.open(await svc.get_task("train:0"), rollout_id="classes")
    lease = session._lease
    assert lease is not None
    try:
        svc.learned_terminal.add("slow")  # a tool seen ending an episode is not declared terminal
        expected = {
            "submit": "exclusive",  # platoon_terminal_tools
            "slow_sync": "exclusive",  # the terminal_tools option
            "scan_declared": "concurrent",  # platoon_concurrent_tools
            "slow_b": "concurrent",  # the concurrent_tools option
            "scan": "concurrent" if toys.HAS_CONCURRENT else "serial",  # @tool(concurrent=True) where the SDK has it
            "read_count": "serial",  # read-only is about what an agent sees, not about overlapping
            "slow": "serial",
            "no_such_tool": "serial",
        }
        assert {name: await svc.call_class(name, lease) for name in expected} == expected
    finally:
        await session.close()
    with pytest.raises(ValueError, match="queue_timeout_s"):
        service(queue_timeout_s=0)
    assert service(queue_timeout_s=None).queue_timeout_s is None and service().queue_timeout_s == 240.0


async def test_a_call_still_runs_when_the_tool_listing_fails(caplog):
    session = await open_session(concurrent_tools=["scan_declared"])
    lease = session._lease
    assert lease is not None

    async def broken_listing() -> list:
        raise ConnectionError("the env server dropped the listing")

    lease.backend.tools = broken_listing  # type: ignore[method-assign]
    try:
        with caplog.at_level(logging.WARNING, logger="platoon.envs.ors.service"):
            assert (await session.call_tool("slow", {"tag": "a"})).as_text() == "slow:a"  # run as a serial call
        assert "Cannot list the tools" in caplog.text
        svc = session._service
        assert await svc.call_class("submit", lease) == "exclusive"  # declared terminal: no listing needed
        assert await svc.call_class("scan_declared", lease) == "serial"  # the safe class without the listing
    finally:
        await session.close()


async def test_declared_concurrent_tools_run_beside_serial_ones():
    root = await open_session(concurrent_tools=["slow_b"])
    agents = await views(root, 4)
    try:
        await asyncio.gather(
            agents[0].call_tool("slow", {"seconds": 0.1, "tag": "serial"}),
            agents[1].call_tool("scan_declared", {"seconds": 0.1, "tag": "declared"}),
            agents[2].call_tool("slow_b", {"seconds": 0.1, "tag": "option"}),
            agents[3].call_tool("scan", {"seconds": 0.1, "tag": "sdk"}),
        )
        assert toys.peak() == (4 if toys.HAS_CONCURRENT else 3)
    finally:
        await close(*agents)


async def test_a_terminal_call_waits_for_calls_in_flight_and_calls_queued_behind_it_never_run():
    """Frontier-CS's finalization race: a sibling's edit must not land while the terminal call grades."""

    # Straight into the env, as before: a write lands while submit judges, and the grade mixes two solutions.
    racy = await open_session()
    try:
        backend = racy._lease.backend  # type: ignore[union-attr]
        await backend.call("write", {"content": "v1"})

        async def late_write() -> None:
            await asyncio.sleep(0.02)
            await backend.call("write", {"content": "late"})

        graded, _ = await asyncio.gather(backend.call("submit", {"seconds": 0.1}), late_write())
        assert graded.metadata == {"graded": "v1", "consistent": False}
    finally:
        await racy.close()

    toys.reset()
    root = await open_session()
    child, sibling = await root.fork("shared"), await root.fork("shared")
    try:
        await root.call_tool("write", {"content": "v1"})
        in_flight = asyncio.ensure_future(child.call_tool("write", {"content": "v2", "pause": 0.1}))
        await asyncio.sleep(0.03)
        submit = asyncio.ensure_future(root.call_tool("submit", {"seconds": 0.1, "tag": "root"}))
        await asyncio.sleep(0.03)
        late = await sibling.call_tool("write", {"content": "late"})  # queued behind the terminal call
        result = await submit
        await in_flight

        assert result.finished and result.reward == 1.0
        assert result.metadata == {"graded": "v2", "consistent": True}  # it waited for the write in flight
        assert json.loads(late.as_text())["bridge_error"] == "environment_already_finished"
        assert toys.tags("write") == ["v1", "v2"] and env_of(root).state["solution"] == "v2"  # "late" never ran
        assert root._lease.turn == 3  # type: ignore[union-attr]
        assert (await root.grade()).reward == 1.0
    finally:
        await close(root, child, sibling)


async def test_a_call_not_admitted_within_the_queue_deadline_is_not_run(tmp_path, caplog):
    for result_format in ("blocks", "bridge_json"):
        toys.reset()
        root = await open_session(queue_timeout_s=0.1, log_dir=tmp_path / result_format, result_format=result_format)
        child = await root.fork("shared")
        try:
            holder = asyncio.ensure_future(child.call_tool("slow", {"seconds": 0.4, "tag": "holder"}))
            await asyncio.sleep(0.03)
            started = time.monotonic()
            with caplog.at_level(logging.WARNING, logger="platoon.envs.ors.session"):
                busy = await root.call_tool("slow_b", {"tag": "late"})
            assert time.monotonic() - started >= 0.09
            await holder

            text = busy.as_text()
            assert busy.is_error and busy.metadata["error_reason"] == "env_busy"
            assert text.startswith("Not run: env busy." if result_format == "blocks" else "Error executing tool slow_b")
            assert "not executed and does not count" in text
            assert toys.tags() == ["holder"] and root._lease.turn == 1  # type: ignore[union-attr]
            assert "slow_b of session" in caplog.text and "not run: env busy" in caplog.text
            (not_run,) = [event for event in events(tmp_path / result_format) if event["type"] == "tool_not_run"]
            assert not_run["tool_name"] == "slow_b" and not_run["reason"] == "env_busy"
            assert not_run["queue_timeout_s"] == 0.1 and not_run["waited_s"] >= 0.09
        finally:
            await close(root, child)


async def test_cancelled_callers_queued_calls_never_run_and_running_ones_finish(tmp_path):
    root = await open_session(log_dir=tmp_path)
    other = await root.fork("shared")
    lease = root._lease
    assert lease is not None
    try:
        # Cancelled while queued: never runs, not counted.
        holder = asyncio.ensure_future(root.call_tool("slow", {"seconds": 0.3, "tag": "holder"}))
        await asyncio.sleep(0.03)
        queued = asyncio.ensure_future(other.call_tool("slow_b", {"tag": "queued"}))
        await asyncio.sleep(0.03)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        await holder
        assert toys.tags() == ["holder"] and lease.turn == 1

        # Cancelled while running: it runs to the end, holds the env until then, and is recorded on the lease.
        abandoned = asyncio.ensure_future(root.call_tool("slow", {"seconds": 0.3, "tag": "abandoned"}))
        await asyncio.sleep(0.03)
        abandoned.cancel()
        with pytest.raises(asyncio.CancelledError):
            await abandoned
        await other.call_tool("slow_b", {"tag": "next"})
        (_, _, _, abandoned_end), (_, _, next_start, _) = toys.runs()[1:]
        assert next_start >= abandoned_end and lease.turn == 3
        results = [event for event in events(tmp_path) if event["type"] == "tool_result"]
        assert [event["turn"] for event in results] == [1, 2, 3]

        # A session closed while its call waits: the call is refused and never runs.
        holder = asyncio.ensure_future(root.call_tool("slow", {"seconds": 0.3, "tag": "holder 2"}))
        await asyncio.sleep(0.03)
        orphan = asyncio.ensure_future(other.call_tool("slow_b", {"tag": "orphan"}))
        await asyncio.sleep(0.01)
        await other.close()
        assert "is closed" in (await orphan).as_text()
        await holder
        assert "orphan" not in toys.tags() and lease.turn == 4
    finally:
        await close(root, other)


async def test_the_last_release_drains_the_calls_in_flight_before_teardown():
    session = await open_session()
    env = env_of(session)
    call = asyncio.ensure_future(session.call_tool("slow", {"seconds": 0.15, "tag": "last"}))
    await asyncio.sleep(0.03)
    call.cancel()
    await session.close()
    ((_, _, _, ended),) = toys.runs()
    assert toys.torn_down(env) and toys.TEARDOWNS[0][1] >= ended
    assert session._lease is None and (await session.call_tool("slow", {})).is_error


async def test_concurrent_calls_record_their_own_turns(tmp_path):
    root = await open_session(log_dir=tmp_path)
    other = await root.fork("shared")
    try:
        await asyncio.gather(
            root.call_tool("scan_declared", {"seconds": 0.05, "tag": "a"}),
            other.call_tool("scan_declared", {"seconds": 0.01, "tag": "b"}),
        )
    finally:
        await close(root, other)
    log = events(tmp_path)
    calls = {event["call_id"]: event["turn"] for event in log if event["type"] == "tool_call"}
    results = {event["call_id"]: event["turn"] for event in log if event["type"] == "tool_result"}
    assert calls == results and sorted(calls.values()) == [1, 2]
    assert all(event["lock_wait_ms"] >= 0 for event in log if event["type"] == "tool_call")


async def test_max_tool_calls_counts_calls_as_they_start():
    root = await open_session(max_tool_calls=2)
    agents = await views(root, 3)
    try:
        results = await asyncio.gather(*(agent.call_tool("slow", {"seconds": 0.02, "tag": "x"}) for agent in agents))
        refused = [json.loads(result.as_text())["bridge_error"] for result in results if result.is_error]
        assert refused == ["max_tool_calls_exceeded"] and len(toys.runs()) == 2
        assert root._lease.turn == 2 and root._lease.cap_hit  # type: ignore[union-attr]
    finally:
        await close(*agents)


async def test_a_tool_waiting_for_anothers_effect_stalls_only_until_its_own_timeout():
    for concurrent, saw_flag in (((), False), (("wait_flag", "set_flag"), True)):
        toys.reset()
        root = await open_session(concurrent_tools=concurrent)
        other = await root.fork("shared")
        try:
            wait = asyncio.ensure_future(root.call_tool("wait_flag", {"timeout": 0.3}))
            await asyncio.sleep(0.03)
            await other.call_tool("set_flag", {})
            # Serial: the set waited for the wait to give up.  Concurrent: the set ran while the wait waited.
            assert (await wait).as_text() == f"saw_flag={saw_flag}"
        finally:
            await close(root, other)


async def test_grading_waits_for_the_calls_in_flight():
    root = await open_session("GradedTrackEnv")
    child = await root.fork("shared")
    try:
        await child.call_tool("fill", {"n": 100})
        # Straight into the env, as before: the sync grader walks the dict while a sync tool grows it.
        fill = asyncio.ensure_future(root._lease.backend.call("fill", {"n": 100, "pause": 0.001}))  # type: ignore[union-attr]
        await asyncio.sleep(0.02)
        with pytest.raises(RuntimeError, match="changed size during iteration"):
            await sdk.run_user_callable(env_of(root).platoon_grade)
        await fill

        fill = asyncio.ensure_future(child.call_tool("fill", {"n": 100, "pause": 0.001}))
        await asyncio.sleep(0.02)
        outcome = await root.grade()
        await fill
        assert (outcome.status, outcome.reward) == ("scored", 1.0)
        assert outcome.metadata["env_metadata"] == {"keys": 300}  # after the fill in flight, all of it
    finally:
        await close(root, child)


async def test_a_grade_from_the_lease_waits_for_calls_in_flight_but_not_for_one_that_cannot_end_the_episode():
    """Without platoon_grade a grade reads only the lease, which a call updates when it ends: it waits for the
    lease's calls in flight (at most queue_timeout_s), and one still running then matters only if it may end the
    episode."""

    # The agent gave up on a call that runs past the deadline.  It is no terminal call: no_submit, at the deadline.
    root = await open_session(queue_timeout_s=0.2)
    try:
        zombie = asyncio.ensure_future(root.call_tool("slow", {"seconds": 1.0, "tag": "zombie"}))
        await until_running(root)
        zombie.cancel()
        started = time.monotonic()
        outcome = await root.grade()
        assert time.monotonic() - started >= 0.19 and "zombie" not in toys.tags()
        assert (outcome.status, outcome.reason, outcome.reward) == ("scored", "no_submit", 0.0)
    finally:
        await root.close()

    # A terminal call that ends within the wait decides the grade, also when its agent gave up on it.
    root = await open_session()
    try:
        await root.call_tool("write", {"content": "v1"})
        submit = asyncio.ensure_future(root.call_tool("submit", {"seconds": 0.2, "tag": "late"}))
        await until_running(root, "exclusive")
        submit.cancel()
        outcome = await root.grade()
        assert (outcome.status, outcome.reward, outcome.metadata["terminal_tool"]) == ("scored", 1.0, "submit")
    finally:
        await root.close()


@pytest.mark.parametrize(
    ("env", "tool", "regraded"),
    [("TrackEnv", "submit", None), ("UndeclaredTrackEnv", "slow", "no_submit")],
    ids=["a-terminal-call", "any-call-of-an-env-without-terminal-tools"],
)
async def test_a_grade_from_the_lease_is_env_busy_while_a_call_that_may_end_the_episode_runs(env, tool, regraded):
    root = await open_session(env, queue_timeout_s=0.1)
    try:
        running = asyncio.ensure_future(root.call_tool(tool, {"seconds": 0.6, "tag": tool}))
        await until(lambda: any(lease_of(root).running.values()))
        outcome = await root.grade()
        assert (outcome.status, outcome.failure_kind) == ("error", "env_busy")
        assert re.fullmatch(
            r"grading waited \d+\.\ds for the calls in flight on the env, and one that may end the episode still runs",
            outcome.reason or "",
        )
        await running
        outcome = await root.grade()  # the call ended: the lease decides
        assert outcome.status == "scored" and outcome.reason == regraded
    finally:
        await root.close()


async def test_platoon_grade_that_cannot_start_in_time_is_env_busy():
    root = await open_session("GradedTrackEnv", queue_timeout_s=0.1)
    try:
        holder = asyncio.ensure_future(root.call_tool("slow", {"seconds": 0.6, "tag": "holder"}))
        await until_running(root)
        outcome = await root.grade()
        assert (outcome.status, outcome.failure_kind) == ("error", "env_busy")
        assert re.fullmatch(r"grading waited \d+\.\ds for its turn on the env", outcome.reason or "")
        await holder
        assert (await root.grade()).reward == 1.0
    finally:
        await root.close()


@pytest.mark.parametrize("env", ["TrackEnv", "GradedTrackEnv"], ids=["from-the-lease", "platoon_grade"])
async def test_a_grade_of_a_session_closed_before_or_while_it_waits_is_session_closed(env):
    root = await open_session(env, queue_timeout_s=None)
    holder = asyncio.ensure_future(root.call_tool("slow", {"seconds": 0.3, "tag": "holder"}))
    await until_running(root)
    grading = asyncio.ensure_future(root.grade())  # it waits for the holder: a call in flight, or its turn
    await asyncio.sleep(0.05)
    await root.close()  # the last release: the holder drains, then the env closes
    outcome = await grading
    assert (outcome.status, outcome.failure_kind) == ("error", "session_closed")
    assert (await holder).as_text() == "slow:holder"
    assert (await root.grade()).failure_kind == "session_closed"


async def test_a_copy_fork_never_snapshots_a_call_half_applied():
    root = await open_session()
    child = await root.fork("shared")
    try:
        fill = asyncio.ensure_future(child.call_tool("fill", {"n": 100, "pause": 0.001}))
        await asyncio.sleep(0.02)
        copy = await root.fork("copy")
        await fill
        try:
            assert len(env_of(copy).state["data"]) == len(env_of(root).state["data"]) == 100
            assert copy._lease.turn == 1  # type: ignore[union-attr]
        finally:
            await copy.close()
    finally:
        await close(root, child)


async def test_a_copy_fork_that_cannot_start_or_is_cancelled_leaks_no_copy(monkeypatch):
    clones: list[Any] = []
    cancel_during_copy: list[asyncio.Task[Any]] = []
    original = LocalBackend.copy

    def recording_copy(self: LocalBackend) -> LocalBackend:
        clone = original(self)
        clones.append(clone.env)
        for task in cancel_during_copy:  # the spawning agent is cancelled while the snapshot runs
            asyncio.get_running_loop().call_soon(task.cancel)
        return clone

    monkeypatch.setattr(LocalBackend, "copy", recording_copy)
    root = await open_session(queue_timeout_s=0.1)
    root_env = env_of(root)
    try:
        holder = asyncio.ensure_future(root.call_tool("slow", {"seconds": 0.6, "tag": "holder"}))
        await until_running(root)
        with pytest.raises(EnvBusy):
            await root.fork("copy")
        waiting = asyncio.ensure_future(root.fork("copy"))
        await asyncio.sleep(0.02)
        waiting.cancel()  # cancelled while it waits for its turn
        with pytest.raises(asyncio.CancelledError):
            await waiting
        await holder
        assert clones == []  # neither fork copied the env

        # The copy and the view that owns it are made in one step: no cancellation lands between them.
        fork = asyncio.ensure_future(root.fork("copy"))
        cancel_during_copy.append(fork)
        copy = await fork
        (clone,) = clones
        assert env_of(copy) is clone
        await copy.close()
        assert toys.torn_down(clone)
    finally:
        await root.close()
    assert toys.torn_down(root_env)


async def test_a_platoon_fork_that_cannot_start_or_is_cancelled_holds_no_part_of_its_parent(monkeypatch):
    loop = asyncio.get_running_loop()
    hooked: list[Any] = []
    cancel_during_hook: list[asyncio.Task[Any]] = []
    original = toys.TrackForkEnv.platoon_fork

    def recording_hook(self: Any, task_spec: dict) -> Any:
        child = original(self, task_spec)
        hooked.append(child)
        for task in cancel_during_hook:  # a sync hook runs in a worker thread
            loop.call_soon_threadsafe(task.cancel)
        return child

    monkeypatch.setattr(toys.TrackForkEnv, "platoon_fork", recording_hook)
    root = await open_session("TrackForkEnv", queue_timeout_s=0.1)
    root_env = env_of(root)
    child_task = Task(id="child", goal="child goal")
    try:
        holder = asyncio.ensure_future(root.call_tool("slow", {"seconds": 0.6, "tag": "holder"}))
        await until_running(root)
        with pytest.raises(EnvBusy):
            await root.fork("shared", task=child_task)
        await holder
        assert hooked == []  # the hook never ran

        fork = asyncio.ensure_future(root.fork("shared", task=child_task))
        cancel_during_hook.append(fork)
        with pytest.raises(asyncio.CancelledError):
            await fork  # the hook ran to its end, and its env was dropped before a child lease held the parent
        assert len(hooked) == 1
    finally:
        await root.close()
    assert toys.torn_down(root_env)  # nothing holds the parent's env open
    assert not any(toys.torn_down(child) for child in hooked)  # platoon never tears down a hook's env


async def test_a_platoon_fork_childs_first_listing_takes_its_turn_and_is_retried_after_env_busy():
    root = await open_session("TrackForkEnv", queue_timeout_s=0.1)
    child = await root.fork("shared", task=Task(id="child", goal="child goal"))
    try:
        holder = asyncio.ensure_future(root.call_tool("slow", {"seconds": 0.8, "tag": "holder"}))
        await until_running(root)
        with pytest.raises(EnvBusy):
            await child.tools()  # the child's own listing runs alone on the shared env, so it waits for the holder
        busy = await child.call_tool("slow_b", {"tag": "child"})  # classifying the call needs that listing
        assert busy.is_error and busy.metadata["error_reason"] == "env_busy"
        await holder
        assert "slow_b" in {spec.name for spec in await child.tools()}  # nothing was cached: it lists now
        assert (await child.call_tool("slow_b", {"tag": "child"})).as_text() == "slow_b:child"
        assert toys.tags() == ["holder", "child"]
    finally:
        await close(root, child)


# ------------------------------------------------------------------------------------------------ ORS servers


@pytest.fixture(scope="module")
def ors_server() -> Iterator[SimpleNamespace]:
    uvicorn = pytest.importorskip("uvicorn")
    from openreward.environments import Server

    ors = Server([toys.TrackEnv, toys.PipeEnv])
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    uv = uvicorn.Server(uvicorn.Config(ors.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=uv.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not uv.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("The ORS test server did not start")
        time.sleep(0.05)
    try:
        yield SimpleNamespace(hostport=f"127.0.0.1:{port}", ors=ors)
    finally:
        uv.should_exit = True
        thread.join(timeout=10)
        sock.close()


async def open_remote(server: SimpleNamespace, env: str, **options: Any) -> OrsSession:
    svc = OrsEnvironmentService.from_uri(f"ors+http://{server.hostport}/{env}?variant={env.lower()}", **options)
    return await svc.open(await svc.get_task("train:0"), rollout_id=f"remote-{uuid.uuid4().hex[:6]}")


async def test_remote_read_modify_write_calls_lose_no_updates(ors_server):
    root = await open_remote(ors_server, "TrackEnv")
    agents = await views(root, 4)
    lease = root._lease
    assert lease is not None
    tools = ["incr_sync", "incr_sync_b", "incr_sync", "incr_sync_b"]
    increment = {"n": 15, "pause": 0.002}
    try:
        await asyncio.gather(*(lease.backend.call(tool, dict(increment)) for tool in tools))  # unordered
        lost = 60 - int((await root.call_tool("read_count", {})).as_text())
        assert lost > 0
        await asyncio.gather(*(agent.call_tool(tool, dict(increment)) for agent, tool in zip(agents, tools)))
        assert int((await root.call_tool("read_count", {})).as_text()) == 120 - lost
    finally:
        await close(*agents)


async def test_toolathlons_shared_pipe_serves_every_agent_without_failures(ors_server):
    """Toolathlon's per-session MCP pipe (verbatim ``send_recv``) fails a read that overlaps another one, even
    between two calls of ``mcp_health``, which changes nothing: read-only is not safe to overlap."""

    root = await open_remote(ors_server, "PipeEnv", read_only_tools=["mcp_health"])
    agents = await views(root, 4)
    lease = root._lease
    assert lease is not None
    call = {"name": "echo", "arguments": {"delay": 0.03, "tag": "x"}}
    try:
        unordered = await asyncio.gather(lease.backend.call("call_tool", call), lease.backend.call("mcp_health", {}))
        failures = [output for output in unordered if output.metadata]
        assert failures and "readuntil() called while another coroutine is already waiting" in str(
            failures[0].blocks[0].text
        )

        unordered = await asyncio.gather(*(lease.backend.call("mcp_health", {}) for _ in range(2)))
        collided = any(output.metadata for output in unordered)
        # Two read-only calls collide too, on 0.1.125; 0.1.163 added per-tool locks (with the concurrent flag), which
        # order calls of one name, but not call_tool against mcp_health.
        assert collided is not toys.HAS_CONCURRENT

        def steps(index: int) -> list[tuple[str, dict[str, Any]]]:
            return [("call_tool", call), ("mcp_health", {}), ("call_tool", call)][index % 2 :]

        async def agent_calls(agent: OrsSession, index: int):
            return await asyncio.gather(*(agent.call_tool(tool, args) for tool, args in steps(index)))

        results = [
            result
            for batch in await asyncio.gather(*(agent_calls(a, i) for i, a in enumerate(agents)))
            for result in batch
        ]
        assert len(results) == 10 and not any(result.is_error for result in results)
    finally:
        await close(*agents)


async def test_a_remote_call_whose_caller_gave_up_finishes_on_the_server_before_the_next_starts(ors_server):
    root = await open_remote(ors_server, "TrackEnv")
    other = await root.fork("shared")
    try:
        abandoned = asyncio.ensure_future(root.call_tool("slow", {"seconds": 0.2, "tag": "abandoned"}))
        await until(lambda: toys.peak() > 0)  # it runs on the server
        abandoned.cancel()
        with pytest.raises(asyncio.CancelledError):
            await abandoned
        assert (await other.call_tool("slow_b", {"tag": "next"})).as_text() == "slow_b:next"
        (_, _, _, abandoned_end), (_, _, next_start, _) = toys.runs()  # the server runs in this process
        assert next_start >= abandoned_end and root._lease.turn == 2  # type: ignore[union-attr]
    finally:
        await close(root, other)


# ------------------------------------------------------------------------------------------------ agent trees


@pytest.mark.parametrize(
    ("env", "tool"),
    [
        ("TrackEnv", "incr_sync"),
        ("TrackEnv", "incr_async"),
        ("TrackForkEnv", "incr_sync"),
        ("TrackForkEnv", "incr_async"),
    ],
)
async def test_children_and_their_parent_lose_no_updates(env, tool):
    session = await open_session(env)
    try:
        increment = call(tool, n=10, pause=0.002)
        plan = [{"gather": [increment, {"spawn": [[increment] for _ in range(4)]}, increment]}]
        tree = await run_tree(session, plan)
        assert tree.result.status == "completed" and [run.result.status for run in tree.spawned] == ["completed"] * 4
        assert env_of(session).state["count"] == 60
    finally:
        await session.close()


async def test_a_parent_its_children_and_their_gathers_take_turns():
    session = await open_session()
    try:
        child = [{"gather": [call("slow", seconds=0.02, tag="c-a"), call("slow_b", seconds=0.02, tag="c-b")]}]
        plan = [{"gather": [call("slow_sync", seconds=0.02, tag="root"), {"spawn": [child, child, child]}]}]
        tree = await run_tree(session, plan)
        assert tree.result.status == "completed"
        assert len(toys.runs()) == 7 and toys.peak() == 1
    finally:
        await session.close()


async def test_a_child_left_running_by_its_timeout_is_drained_before_its_env_is_torn_down(monkeypatch):
    """A child's session close gives up after SESSION_CLOSE_TIMEOUT_SECONDS; its teardown goes on regardless."""

    monkeypatch.setattr(agents_module, "SESSION_CLOSE_TIMEOUT_SECONDS", 0.05)
    # A platoon_fork child (its own lease on the root's env), a fresh child (its own env) and a plain shared child.
    for env, fork in (("TrackForkEnv", None), ("TrackEnv", "fresh"), ("TrackEnv", "shared")):
        toys.reset()
        session = await open_session(env)
        tree = await run_tree(
            session, [{"spawn": [[call("slow", seconds=0.3, tag="child")]], "fork": fork}], child_timeout_s=0.05
        )
        assert [run.result.status for run in tree.spawned] == ["timeout"], (env, fork)
        await session.close()
        owned = toys.INSTANCES if fork == "fresh" else toys.INSTANCES[:1]  # a platoon_fork child owns no env
        await until(lambda: len(toys.TEARDOWNS) == len(owned))
        assert {instance for instance, _ in toys.TEARDOWNS} == {id(instance) for instance in owned}, (env, fork)
        ((_, _, _, ended),) = toys.runs()
        env_with_the_call = toys.INSTANCES[1] if fork == "fresh" else toys.INSTANCES[0]
        (torn_at,) = [at for instance, at in toys.TEARDOWNS if instance == id(env_with_the_call)]
        assert torn_at >= ended, (env, fork)  # drained: torn down after the abandoned call ended


async def test_the_root_grade_waits_for_a_timed_out_childs_call():
    session = await open_session("GradedTrackEnv")
    try:
        plan = [call("fill", n=100), {"spawn": [[call("fill", n=200, pause=0.001)]]}]
        tree = await run_tree(session, plan, child_timeout_s=0.05)
        assert [run.result.status for run in tree.spawned] == ["timeout"]
        outcome = await session.grade()
        assert (outcome.status, outcome.reward) == ("scored", 1.0)
        assert outcome.metadata["env_metadata"] == {"keys": 300}
    finally:
        await session.close()


# ------------------------------------------------------------------------------------------------ the MCP plane


@pytest_asyncio.fixture
async def gateway_server():
    gateway = make_gateway()
    server = GatewayServer(gateway, keepalive_seconds=0.05)
    await server.start(host="127.0.0.1")
    try:
        yield SimpleNamespace(gateway=gateway, base_url=server.base_url)
    finally:
        await server.stop()


def open_root(gateway: ModelGateway, session: OrsSession):
    root = gateway.open_agent(rollout_id=session.rollout_id, trajectory_id="root", interactions={}, max_steps=100)
    bind_agent_tools(gateway, root, session)
    return root


async def open_child(gateway: ModelGateway, root, session: OrsSession):
    child = gateway.open_subagent(root, role="subagent", goal="child", max_steps=10)
    bind_agent_tools(gateway, child, await session.fork("shared"))
    return child


async def mint_children(base_url: str, parent, *goals: str) -> list[str]:
    """The keys of children the parent's harness mints over HTTP; they share the parent's tool provider."""
    keys = []
    async with aiohttp.ClientSession() as http:
        for goal in goals:
            headers = {"Authorization": f"Bearer {parent.secret}"}
            body = {"role": "subagent", "goal": goal}
            async with http.post(f"{base_url}/v1/platoon/agents", json=body, headers=headers) as response:
                keys.append((await response.json())["api_key"])
    return keys


async def tools_call(url: str, key: str, name: str, arguments: dict[str, Any], *, request_id: Any = None, sse=False):
    body = {"jsonrpc": "2.0", "id": request_id or uuid.uuid4().hex, "method": "tools/call"}
    body["params"] = {"name": name, "arguments": arguments}
    accept = "application/json, text/event-stream" if sse else "application/json"
    headers = {"Authorization": f"Bearer {key}", "Accept": accept}
    async with aiohttp.ClientSession() as http:
        async with http.post(url, json=body, headers=headers) as response:
            text = await response.text()
            return response.status, json.loads(text) if not sse and text else text


@pytest.mark.parametrize("queue_timeout_s", [None, 0.3])
async def test_a_call_queued_past_its_clients_deadline(gateway_server, queue_timeout_s):
    """The official MCP client gives up on a call without telling the server, so the gateway cannot drop it.
    Without a queue deadline the call runs anyway when its turn comes (a zombie the model was told had failed);
    with one shorter than the client's, the model hears "not run: env busy" and the env never runs it."""

    gateway = gateway_server.gateway
    session = await open_session(queue_timeout_s=queue_timeout_s)
    root = open_root(gateway, session)
    child = await open_child(gateway, root, session)
    try:
        hold = asyncio.ensure_future(call_agent_tool(child, "slow", {"seconds": 1.0, "tag": "child"}))
        await asyncio.sleep(0.05)
        async with streamable_http_client(mcp_url(gateway_server.base_url, root.secret)) as (read, write, _):
            async with ClientSession(read, write) as client:
                await client.initialize()
                deadline = timedelta(seconds=0.6)
                if queue_timeout_s is None:
                    with pytest.raises(McpError, match="Timed out"):
                        await client.call_tool("slow_b", {"tag": "root"}, read_timeout_seconds=deadline)
                    await hold
                    await until(lambda: "root" in toys.tags())  # the zombie runs once the child's call ends
                else:
                    result = await client.call_tool("slow_b", {"tag": "root"}, read_timeout_seconds=deadline)
                    assert result.isError and result.content[0].text.startswith("Not run: env busy")
                    await hold
        await asyncio.sleep(0.1)
        assert toys.tags() == (["child", "root"] if queue_timeout_s is None else ["child"])
        assert session._lease.turn == (2 if queue_timeout_s is None else 1)  # type: ignore[union-attr]
    finally:
        gateway.close_rollout(session.rollout_id)
        await session.close()


async def test_notifications_cancelled_drops_a_queued_call_and_lets_a_running_one_finish(gateway_server, tmp_path):
    gateway = gateway_server.gateway
    session = await open_session(log_dir=tmp_path)
    root = open_root(gateway, session)
    child = await open_child(gateway, root, session)
    url = mcp_url(gateway_server.base_url)

    async def cancel(request_id: Any) -> None:
        notification = {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": request_id}}
        async with aiohttp.ClientSession() as http:
            headers = {"Authorization": f"Bearer {root.secret}"}
            async with http.post(url, json=notification, headers=headers) as response:
                assert response.status == 202

    try:
        hold = asyncio.ensure_future(call_agent_tool(child, "slow", {"seconds": 0.3, "tag": "child"}))
        await asyncio.sleep(0.05)
        queued = asyncio.ensure_future(
            tools_call(url, root.secret, "slow_b", {"tag": "queued"}, request_id=5, sse=True)
        )
        await asyncio.sleep(0.05)
        await cancel(5)
        status, text = await asyncio.wait_for(queued, 5)
        assert status == 200 and "event: message" not in text  # no response for a cancelled request
        await hold
        assert toys.tags() == ["child"]  # the queued call never ran

        running = asyncio.ensure_future(
            tools_call(url, root.secret, "slow", {"seconds": 0.2, "tag": "running"}, request_id=6, sse=True)
        )
        await asyncio.sleep(0.05)
        await cancel(6)
        status, text = await asyncio.wait_for(running, 5)
        assert "event: message" not in text and "running" not in toys.tags()
        await until(lambda: "running" in toys.tags())  # ... it finishes in the env and is recorded on the lease
        await asyncio.sleep(0.05)
        assert [event["turn"] for event in events(tmp_path) if event["type"] == "tool_result"] == [1, 2]
        assert [(record["name"], record["text"]) for record in tool_call_records(root)] == [
            ("slow_b", "cancelled"),
            ("slow", "cancelled"),
        ]
    finally:
        gateway.close_rollout(session.rollout_id)
        await session.close()


async def test_children_minted_over_http_take_turns_with_their_parent(gateway_server):
    gateway = gateway_server.gateway
    for concurrent in ((), ("wait_flag", "set_flag")):
        toys.reset()
        session = await open_session(concurrent_tools=concurrent)
        root = open_root(gateway, session)
        url = mcp_url(gateway_server.base_url)
        try:
            keys = [root.secret, *await mint_children(gateway_server.base_url, root, "child-1", "child-2")]
            minted = [agent for agent in gateway.agents(session.rollout_id) if agent.parent_agent_id == root.agent_id]
            assert len(minted) == 2 and all(agent.tools is root.tools for agent in minted)

            calls = [("slow", "root"), ("slow_b", "child-1"), ("slow_sync", "child-2")]
            replies = await asyncio.gather(
                *(tools_call(url, key, tool, {"seconds": 0.05, "tag": tag}) for key, (tool, tag) in zip(keys, calls))
            )
            assert all(status == 200 and not body["result"]["isError"] for status, body in replies)
            assert sorted(toys.tags()) == ["child-1", "child-2", "root"] and toys.peak() == 1

            wait = asyncio.ensure_future(tools_call(url, keys[0], "wait_flag", {"timeout": 0.3}))
            await asyncio.sleep(0.05)
            await tools_call(url, keys[1], "set_flag", {})
            _, body = await wait
            assert body["result"]["content"][0]["text"] == f"saw_flag={bool(concurrent)}"
        finally:
            gateway.close_rollout(session.rollout_id)
            await session.close()


async def test_children_minted_over_http_take_turns_with_each_other(gateway_server):
    """Calls take turns per agent, also among children minted over HTTP, which share their parent's session: one
    minted child's gather does not hold off a sibling's single call."""

    gateway = gateway_server.gateway
    session = await open_session()
    root = open_root(gateway, session)
    url = mcp_url(gateway_server.base_url)
    gate = lease_of(session).gate
    try:
        gathering, single = await mint_children(gateway_server.base_url, root, "gathering", "single")
        hold = asyncio.ensure_future(call_agent_tool(root, "wait_flag", {"timeout": 30}))  # until the flag is set
        await until_running(session)
        gathered = [
            asyncio.ensure_future(tools_call(url, gathering, "slow", {"seconds": 0.01, "tag": f"gathered-{index}"}))
            for index in range(5)
        ]
        await until(lambda: gate.waiting == 5)
        alone = asyncio.ensure_future(tools_call(url, single, "slow_b", {"seconds": 0.01, "tag": "single"}))
        await until(lambda: gate.waiting == 6)
        env_of(session).state["flag"] = True
        assert (await hold).as_text() == "saw_flag=True"
        replies = await asyncio.gather(*gathered, alone)
        assert all(status == 200 and not body["result"]["isError"] for status, body in replies)
        assert toys.tags("slow", "slow_b").index("single") == 1  # after one gathered call, not after all five
        assert toys.peak() == 1
    finally:
        gateway.close_rollout(session.rollout_id)
        await session.close()


# ------------------------------------------------------------------------------------------------ PTC thread pools


async def _ptc(ctx: InlineAgentContext) -> AgentResult:
    """A PTC-style agent, as an OpenHands harness with programmatic tool calling runs: a conversation thread off
    the loop whose pool runs one cell's calls at once.  Env tools go over the agent's MCP endpoint; ``launch``
    blocks its pool thread while ``ctx.agents.spawn`` runs the child on the rollout loop."""

    plan = json.loads(ctx.instruction.removeprefix("PTC "))
    loop = asyncio.get_running_loop()
    assert ctx.mcp is not None

    def launch(child_plan: dict[str, Any]) -> str:
        goal = "PTC " + json.dumps(child_plan)
        return asyncio.run_coroutine_threadsafe(ctx.agents.spawn(goal), loop).result(timeout=60).message

    def tool(name: str, arguments: dict[str, Any]) -> str:
        body = {"jsonrpc": "2.0", "id": uuid.uuid4().hex, "method": "tools/call"}
        body["params"] = {"name": name, "arguments": arguments}
        headers = {**ctx.mcp.headers, "Accept": "application/json"}  # type: ignore[union-attr]
        reply = httpx.post(ctx.mcp.url, json=body, headers=headers, timeout=60).json()  # type: ignore[union-attr]
        return reply["result"]["content"][0]["text"]

    def cell() -> list[str]:
        with ThreadPoolExecutor(max_workers=plan.get("pool", 32), thread_name_prefix="ptc-tool") as pool:
            futures = [
                pool.submit(launch, item["launch"])
                if "launch" in item
                else pool.submit(tool, item["tool"], item["args"])
                for item in plan["items"]
            ]
            return [future.result() for future in futures]

    done: asyncio.Future[list[str]] = loop.create_future()

    def conversation() -> None:
        try:
            outputs, error = cell(), None
        except BaseException as exc:  # reported to the loop
            outputs, error = [], exc
        loop.call_soon_threadsafe(_settle, done, outputs, error)

    threading.Thread(target=conversation, daemon=True, name="conversation").start()
    ctx.runtime.results.extend(await done)  # type: ignore[attr-defined]
    return AgentResult("completed", final_message="done")


def _settle(future: asyncio.Future[Any], result: Any, error: BaseException | None) -> None:
    if future.done():
        return
    if error is not None:
        future.set_exception(error)
    else:
        future.set_result(result)


PTC = FunctionHarness(name="ptc", version="1", runtime="inline", executes_model_code=False, fn=_ptc)


def tool(name: str, **args: Any) -> dict[str, Any]:
    return {"tool": name, "args": args}


def launch(*items: dict[str, Any], pool: int = 32) -> dict[str, Any]:
    return {"launch": {"items": list(items), "pool": pool}}


@pytest.mark.parametrize(
    ("scenario", "plan", "serial_peak"),
    [
        (
            "P1: two launches and the root's own call in one gather",
            {
                "items": [
                    launch(tool("slow", seconds=0.05)),
                    launch(tool("slow_b", seconds=0.05)),
                    tool("slow", seconds=0.05),
                ]
            },
            True,
        ),
        (
            "P3: a wait and a launch whose child sets the flag",
            {"items": [tool("wait_flag", timeout=0.3), launch(tool("set_flag"))]},
            True,
        ),
        (
            "P5: 40 launches through a 32-thread pool, children with a concurrent tool",
            {"pool": 32, "items": [launch(tool("scan_declared", seconds=0.02)) for _ in range(40)]},
            None,
        ),
        (
            "P6: children that gather grandchildren and their own call",
            {
                "items": [
                    launch(
                        launch(tool("slow", seconds=0.02)),
                        launch(tool("slow_b", seconds=0.02)),
                        tool("slow", seconds=0.02),
                    )
                    for _ in range(2)
                ]
            },
            True,
        ),
    ],
)
async def test_ptc_gathers_with_launches_never_deadlock(gateway_server, scenario, plan, serial_peak):
    session = await open_session()
    try:
        tree = await run_tree(
            session, plan, harness=PTC, prefix="PTC ", base_url=gateway_server.base_url, gateway=gateway_server.gateway
        )
        assert tree.result.status == "completed", scenario
        if serial_peak:  # nothing but serial tools: one at a time over every agent of the tree
            assert toys.peak() == 1, scenario
    finally:
        gateway_server.gateway.close_rollout(session.rollout_id)
        await session.close()


# ------------------------------------------------------------------------------------------------ the OpenHands client

_OPENHANDS_SCRIPT = r"""
import json, sys, threading, time
from openhands.sdk.mcp.definition import MCPToolAction
from openhands.sdk.mcp.utils import create_mcp_tools

url, secret = sys.argv[1:3]
config = {"mcpServers": {"env": {"url": url, "headers": {"Authorization": f"Bearer {secret}"}}}}
client = create_mcp_tools(config, timeout=60)
tools = {tool.name: tool for tool in client.tools}


def call(step, results, index):
    tool = tools[step["name"]]
    tool.executor.timeout = step["timeout"]  # OpenHands' per-call MCP deadline (300 s by default)
    started = time.monotonic()
    observation = tool.executor(MCPToolAction(data=step["args"]))
    text = " ".join(getattr(block, "text", "") for block in observation.content)
    secs = round(time.monotonic() - started, 2)
    results[index] = {"secs": secs, "is_error": bool(observation.is_error), "text": text}


print("READY", flush=True)
for line in sys.stdin:
    steps = json.loads(line)
    if steps is None:
        break
    results = [None] * len(steps)
    threads = [threading.Thread(target=call, args=(step, results, i)) for i, step in enumerate(steps)]
    for thread in threads:
        thread.start()
        time.sleep(0.1)
    for thread in threads:
        thread.join()
    print("RESULT " + json.dumps(results), flush=True)
client.sync_close()
"""


@pytest.mark.skipif(CLIENT_VENV_PYTHON is None or not CLIENT_VENV_PYTHON.exists(), reason=NO_CLIENT_VENV)
async def test_the_openhands_client_hears_env_busy_before_its_deadline_and_the_env_never_runs_the_call(
    gateway_server, tmp_path
):
    """Three busy-env cases (Z1-Z3 below) with the real OpenHands MCP client, which times out a call without
    telling the server.  With the queue deadline under the client's (240 s under 300 s by default; 0.5 s under
    1.5 s here) the model reads "not run: env busy" and the call never runs, also behind a sibling's call or its
    own."""

    gateway = gateway_server.gateway
    session = await open_session(queue_timeout_s=0.5)
    root = open_root(gateway, session)
    child = await open_child(gateway, root, session)
    script = tmp_path / "openhands_client.py"
    script.write_text(_OPENHANDS_SCRIPT)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENHANDS_SUPPRESS_BANNER": "1"}
    env.pop("PYTHONPATH", None)
    process = await asyncio.create_subprocess_exec(
        str(CLIENT_VENV_PYTHON), "-I", str(script), mcp_url(gateway_server.base_url), root.secret,
        cwd=tmp_path, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )  # fmt: skip
    assert process.stdin is not None and process.stdout is not None
    seen: list[str] = []

    async def read(prefix: str) -> str:
        while True:
            line = (await asyncio.wait_for(process.stdout.readline(), 120)).decode()  # type: ignore[union-attr]
            assert line, "the client exited: " + "".join(seen[-20:])
            seen.append(line)
            if line.startswith(prefix):
                return line.removeprefix(prefix)

    async def run(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        process.stdin.write((json.dumps(steps) + "\n").encode())  # type: ignore[union-attr]
        await process.stdin.drain()  # type: ignore[union-attr]
        return json.loads(await read("RESULT "))

    def step(name: str, timeout: float, **args: Any) -> dict[str, Any]:
        return {"name": name, "timeout": timeout, "args": args}

    try:
        await read("READY")
        # Z1: a sibling holds the env past the client's deadline.
        hold = asyncio.ensure_future(call_agent_tool(child, "slow", {"seconds": 1.0, "tag": "child"}))
        await asyncio.sleep(0.05)
        (z1,) = await run([step("slow_b", 1.5, tag="z1")])
        await hold
        # Z2: the root's terminal call queues behind it.
        hold = asyncio.ensure_future(call_agent_tool(child, "slow", {"seconds": 1.0, "tag": "child 2"}))
        await asyncio.sleep(0.05)
        (z2,) = await run([step("submit", 1.5, seconds=0.0)])
        await hold
        # Z3: the root's own long call holds the env.
        first, second = await run([step("slow", 10.0, seconds=1.0, tag="first"), step("slow_b", 1.5, tag="z3")])

        for busy in (z1, z2, second):
            assert busy["is_error"] and "Not run: env busy" in busy["text"] and busy["secs"] < 1.4, busy
        assert not first["is_error"] and "slow:first" in first["text"]
        await asyncio.sleep(0.1)
        assert toys.tags() == ["child", "child 2", "first"]  # none of the refused calls ever ran
        lease = session._lease
        assert lease is not None and lease.turn == 3 and not lease.finished
    finally:
        if process.returncode is None:
            process.stdin.write(b"null\n")
            try:
                await asyncio.wait_for(process.wait(), 30)
            except TimeoutError:
                process.kill()
        gateway.close_rollout(session.rollout_id)
        await session.close()
