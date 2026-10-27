from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import time
import uuid
from pathlib import Path
from typing import Any, cast

import pytest
from harness_fakes import (
    CommandDidNotStart,
    ExecResult,
    FakeAgent,
    FakeChatServer,
    FakeSession,
    ProcessSandbox,
    answer,
    bash_call,
)

from platoon.harness.registry import get_harness_spec
from platoon.harness.runner import HarnessRunner, HarnessSetupError, resolve_placement
from platoon.harness.spec import parse_spec
from platoon.task import Task

API_KEY = "agent-secret"


def make_runner(tmp_path: Path) -> HarnessRunner:
    return HarnessRunner(
        logs_dir=tmp_path / "logs",
        state_root=tmp_path / "state",
        sandbox_state_root=str(tmp_path / "sandbox-state"),
        kill_grace_s=0.2,
    )


def make_session(tmp_path: Path, *, sandbox: bool, instruction: str = "Write 42 to answer.txt") -> FakeSession:
    return FakeSession(
        task=Task(id="task-1", goal=instruction),
        instruction=instruction,
        sandbox=ProcessSandbox(tmp_path / "sandbox") if sandbox else None,
    )


def agent(agent_id: str | None = None) -> FakeAgent:
    """An agent with a unique id unless one is given, as the gateway's agents have."""

    agent_id = agent_id or f"agent-{uuid.uuid4().hex[:8]}"
    return FakeAgent(agent_id=agent_id, api_key=API_KEY, rollout_id="rollout-1", trajectory_id="traj-1")


def shell_spec(script: str, **sections) -> object:
    return parse_spec({"name": "shell", "version": "1", "launch": {"argv": ["sh", "-c", script]}, **sections})


def wait_dead(pid: int, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


async def wait_for_file(path: Path, timeout_s: float = 10.0) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file() and path.read_text().strip():
            return path.read_text().strip()
        await asyncio.sleep(0.05)
    raise AssertionError(f"{path} never appeared")


@pytest.mark.asyncio
async def test_platoon_bash_in_task_sandbox_acts_on_task_state(tmp_path):
    session = make_session(tmp_path, sandbox=True)
    async with FakeChatServer([bash_call("echo 42 > answer.txt && pwd"), answer("wrote it")]) as server:
        result = await make_runner(tmp_path).run(
            get_harness_spec("platoon-bash"),
            session=session,
            agent=agent(),
            base_url=server.base_url,
            placement="task_sandbox",
        )

    assert result.exit_code == 0 and not result.timed_out
    assert result.final_message == "wrote it"
    assert (session.sandbox.workdir / "answer.txt").read_text().strip() == "42"
    first, second = server.requests
    assert first["messages"][1] == {"role": "user", "content": "Write 42 to answer.txt"}
    assert first["tools"][0]["function"]["name"] == "bash"
    tool_message = second["messages"][-1]
    assert tool_message["role"] == "tool" and tool_message["tool_call_id"] == "call-1"
    assert tool_message["content"].startswith("exit code 0") and str(session.sandbox.workdir) in tool_message["content"]
    assert "wrote it" in Path(result.logs_path).read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("placement", ["task_sandbox", "local"])
async def test_the_tasks_own_mcp_servers_render_for_a_harness_in_its_sandbox(tmp_path, placement):
    """``{task_mcp_servers}`` is the session's list, as JSON, for a harness in the task's sandbox, and empty for one
    that runs locally, which reaches the task's servers through the env's tools instead."""

    session = make_session(tmp_path, sandbox=True)
    servers = [{"name": "mcp-server", "transport": "sse", "url": "http://mcp-server:8000/sse"}]
    setattr(session, "task_mcp_servers", servers)  # an optional attribute of a session
    out = tmp_path / "servers.json"
    launch = {"argv": ["sh", "-c", f'printf %s "$SERVERS" > {out}'], "env": {"SERVERS": "{task_mcp_servers}"}}
    spec = parse_spec({"name": "shell", "version": "1", "launch": launch})
    result = await make_runner(tmp_path).run(
        spec, session=cast(Any, session), agent=agent(), base_url="http://127.0.0.1:9", placement=placement
    )

    assert result.exit_code == 0
    assert json.loads(out.read_text()) == (servers if placement == "task_sandbox" else [])


@pytest.mark.asyncio
async def test_platoon_bash_runs_locally_in_a_private_state_dir_without_worker_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "worker-secret")
    session = make_session(tmp_path, sandbox=False)
    script = 'pwd; echo "[$TINKER_API_KEY]"; echo "[$PLATOON_API_KEY]"; echo "[$NO_PROXY]"'
    async with FakeChatServer([bash_call(script), answer("done")]) as server:
        result = await make_runner(tmp_path).run(
            get_harness_spec("platoon-bash"),
            session=session,
            agent=agent("local-agent"),
            base_url=server.base_url,
            placement="local",
        )

    assert result.exit_code == 0 and result.final_message == "done"
    tool_output = server.requests[1]["messages"][-1]["content"]
    assert str(tmp_path / "state" / "local-agent" / "work") in tool_output
    assert "[]" in tool_output  # the worker's TINKER_API_KEY is not inherited
    assert f"[{API_KEY}]" in tool_output
    assert "127.0.0.1" in tool_output
    assert not (tmp_path / "state" / "local-agent").exists()


@pytest.mark.asyncio
async def test_wrong_secret_is_a_harness_error_exit(tmp_path):
    session = make_session(tmp_path, sandbox=False)
    async with FakeChatServer([answer("never")], api_key="other-secret") as server:
        result = await make_runner(tmp_path).run(
            get_harness_spec("platoon-bash"),
            session=session,
            agent=agent(),
            base_url=server.base_url,
            placement="local",
        )

    assert result.exit_code == 2
    assert get_harness_spec("platoon-bash").exit_status(result.exit_code) == "error"
    assert result.final_message is None


@pytest.mark.asyncio
@pytest.mark.parametrize("placement", ["local", "task_sandbox"])
async def test_timeout_kills_the_whole_harness_tree(tmp_path, placement):
    if placement == "task_sandbox" and shutil.which("setsid") is None:
        pytest.skip("setsid is needed to detach the grandchild")
    pid_file = tmp_path / "child.pid"
    # In the sandbox the grandchild leaves the harness's session, so only the env
    # marker can find it; locally it stays in the harness's process group.
    detach = "setsid " if placement == "task_sandbox" else ""
    spec = shell_spec(f"{detach}sleep 60 & echo $! > {pid_file}; sleep 60", timeout_s=30)
    session = make_session(tmp_path, sandbox=placement == "task_sandbox")

    started = time.monotonic()
    result = await make_runner(tmp_path).run(
        spec, session=session, agent=agent(), base_url="http://127.0.0.1:9", placement=placement, timeout_s=1.0
    )

    assert result.timed_out
    assert time.monotonic() - started < 15
    assert wait_dead(int(pid_file.read_text()))


@pytest.mark.asyncio
async def test_ending_a_launch_never_ends_another_of_an_agent_with_the_same_id(tmp_path):
    """Where a sandbox's processes are this host's, the kill of a launch's processes scans the host: each launch
    carries a marker of its own, so a harness that ends never takes another one down with it."""

    agent_id = f"agent-{uuid.uuid4().hex[:8]}"  # the same for both launches, and no other test's
    pid_file = tmp_path / "long.pid"
    running = asyncio.ensure_future(
        make_runner(tmp_path / "a").run(
            shell_spec(f"echo $$ > {pid_file}; sleep 60"),
            session=make_session(tmp_path / "a", sandbox=True),
            agent=agent(agent_id),
            base_url="http://127.0.0.1:9",
            placement="task_sandbox",
            timeout_s=4.0,
        )
    )
    await wait_for_file(pid_file)

    short = await make_runner(tmp_path / "b").run(
        shell_spec("exit 0"),
        session=make_session(tmp_path / "b", sandbox=True),
        agent=agent(agent_id),
        base_url="http://127.0.0.1:9",
        placement="task_sandbox",
    )

    assert short.exit_code == 0
    result = await running
    assert result.timed_out, result  # ran on until its own timeout, not ended by the other launch's kill


@pytest.mark.asyncio
@pytest.mark.parametrize("placement", ["local", "task_sandbox"])
async def test_cancellation_kills_the_harness_and_propagates(tmp_path, placement):
    if placement == "task_sandbox" and shutil.which("setsid") is None:
        pytest.skip("setsid is needed to detach the grandchild")
    pid_file = tmp_path / "child.pid"
    detach = "setsid " if placement == "task_sandbox" else ""
    spec = shell_spec(f"{detach}sleep 60 & echo $! > {pid_file}; sleep 60")
    session = make_session(tmp_path, sandbox=placement == "task_sandbox")
    run = asyncio.ensure_future(
        make_runner(tmp_path).run(
            spec, session=session, agent=agent(), base_url="http://127.0.0.1:9", placement=placement
        )
    )
    child_pid = int(await wait_for_file(pid_file))

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert wait_dead(child_pid)


@pytest.mark.asyncio
@pytest.mark.parametrize("unenterable_cwd", [None, "raise", "exit-126"])
@pytest.mark.parametrize("away", ["removed", "moved"])
async def test_a_harness_whose_agent_removed_the_workdir_is_still_killed_and_read(tmp_path, unenterable_cwd, away):
    """An agent may remove or move the working directory it runs in, where no command starts after that: the
    process provider raises an OSError, hpc-sandbox raised its SandboxError before 0.2.0, and docker and
    hpc-sandbox 0.2 exit 126. The runner's own commands run in "/", so it still kills the process the harness left
    running, which only the agent's marker finds, and reads the harness's final message."""

    if shutil.which("setsid") is None:
        pytest.skip("setsid is needed to detach the grandchild")
    pid_file = tmp_path / "child.pid"
    command = {"removed": "rm -rf {workdir}", "moved": f"mv {{workdir}} {tmp_path / 'moved'}"}[away]
    spec = shell_spec(
        f"setsid sleep 60 </dev/null >/dev/null 2>&1 & echo $! > {pid_file}; "
        f"printf done > {{state_dir}}/final.txt; cd / && {command}",
        final_message={"file": "{state_dir}/final.txt"},
    )
    session = make_session(tmp_path, sandbox=True)
    session.sandbox.unenterable_cwd = unenterable_cwd

    result = await make_runner(tmp_path).run(
        spec, session=session, agent=agent(), base_url="http://127.0.0.1:9", placement="task_sandbox"
    )

    assert not session.sandbox.workdir.exists()
    assert wait_dead(int(pid_file.read_text()))
    assert (result.exit_code, result.final_message, result.timed_out) == (0, "done", False)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exit", "timeout", "raise"])
async def test_a_kill_that_fails_is_logged_and_the_harness_result_stands(tmp_path, caplog, failure):
    """The kill runs in the launch's ``finally``: when its command fails, times out or the sandbox raises for it, the
    runner logs it and returns what the harness did."""

    session = make_session(tmp_path, sandbox=True)
    exec_ = session.sandbox.exec

    async def kill_fails(cmd, **kwargs):
        if "platoon-kill" not in cmd:
            return await exec_(cmd, **kwargs)
        if failure == "raise":
            raise CommandDidNotStart(f"sandbox {session.sandbox.id} lost its container: killed")
        return ExecResult(-9, "", "", timed_out=True) if failure == "timeout" else ExecResult(2, "", "no /proc here")

    session.sandbox.exec = kill_fails  # type: ignore[method-assign]
    spec = shell_spec("printf done > {state_dir}/final.txt", final_message={"file": "{state_dir}/final.txt"})
    the_agent = agent()

    with caplog.at_level("WARNING", logger="platoon.harness.runner"):
        result = await make_runner(tmp_path).run(
            spec, session=session, agent=the_agent, base_url="http://127.0.0.1:9", placement="task_sandbox"
        )

    assert (result.exit_code, result.final_message) == (0, "done")
    [warning] = [record.getMessage() for record in caplog.records if record.name == "platoon.harness.runner"]
    assert the_agent.agent_id in warning and session.sandbox.id in warning
    expected = {"exit": "exited 2: no /proc here", "timeout": "timed out", "raise": "lost its container"}[failure]
    assert expected in warning


async def dead_soon(pid: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        await asyncio.sleep(0.05)
    return False


@pytest.mark.asyncio
async def test_a_local_harness_cancelled_again_during_its_grace_is_still_killed(tmp_path):
    # A stopping rollout worker cancels its rollout again a second later: that must neither cut the harness's grace
    # short nor skip its SIGKILL, which run on after the cancelled call has returned.
    pid_file, asked = tmp_path / "harness.pid", tmp_path / "asked"
    spec = shell_spec(f"trap 'echo >> {asked}' TERM; echo $$ > {pid_file}; while :; do sleep 0.05; done")
    runner = HarnessRunner(logs_dir=tmp_path / "logs", state_root=tmp_path / "state", kill_grace_s=1.0)
    run = asyncio.ensure_future(
        runner.run(
            spec, session=make_session(tmp_path, sandbox=False), agent=agent(), base_url="http://x", placement="local"
        )
    )
    pid = int(await wait_for_file(pid_file))
    try:
        run.cancel()
        await asyncio.sleep(0.3)  # it got SIGTERM, ignored it, and its grace runs
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

        assert await dead_soon(pid, 10.0), "the harness outlived its grace"
        assert asked.exists()
        from platoon.utils.stopping import registered_process_groups

        assert pid not in registered_process_groups()
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)  # what a failed run left


@pytest.mark.asyncio
async def test_install_runs_once_per_sandbox_and_failures_are_setup_errors(tmp_path):
    marker = tmp_path / "installs.log"
    spec = shell_spec("exit 0", install={"commands": [f"echo installed >> {marker}"]})
    runner = make_runner(tmp_path)
    session = make_session(tmp_path, sandbox=True)
    for _ in range(2):  # two agents
        result = await runner.run(
            spec, session=session, agent=agent(), base_url="http://127.0.0.1:9", placement="task_sandbox"
        )
        assert result.exit_code == 0
    assert marker.read_text().splitlines() == ["installed"]

    broken = shell_spec("exit 0", install={"commands": ["echo cannot install >&2; exit 7"]})
    with pytest.raises(HarnessSetupError, match="exit code 7: cannot install"):
        await runner.run(broken, session=session, agent=agent(), base_url="x", placement="task_sandbox")
    missing = shell_spec("exit 0", install={"toolchain": "/nonexistent/toolchain"})
    with pytest.raises(HarnessSetupError, match="toolchain /nonexistent/toolchain"):
        await runner.run(missing, session=session, agent=agent(), base_url="x", placement="task_sandbox")


@pytest.mark.asyncio
async def test_missing_local_executable_is_a_setup_error(tmp_path):
    spec = parse_spec({"name": "ghost", "version": "1", "launch": {"argv": ["/nonexistent/harness"]}})
    with pytest.raises(HarnessSetupError, match="cannot start '/nonexistent/harness'"):
        await make_runner(tmp_path).run(
            spec, session=make_session(tmp_path, sandbox=False), agent=agent(), base_url="x", placement="local"
        )


@pytest.mark.asyncio
async def test_builtins_are_disabled_locally_by_default_and_kept_in_task_sandboxes(tmp_path):
    spec = parse_spec(
        {
            "name": "echoer",
            "version": "1",
            "launch": {"argv": ["sh", "-c", 'echo "ARGS:$*:$TOOLS"', "echoer", "--model", "{model}"]},
            "builtins": {"tools": ["shell"], "disable": {"args": ["--no-tools"], "env": {"TOOLS": "off"}}},
            "final_message": {"stdout_regex": "ARGS:(.*)"},
        }
    )
    runner = make_runner(tmp_path)

    async def run(placement: str, builtins: str) -> str | None:
        session = make_session(tmp_path, sandbox=placement == "task_sandbox")
        result = await runner.run(
            spec, session=session, agent=agent(), base_url="x", placement=placement, builtins=builtins
        )
        return result.final_message

    assert await run("local", "default") == "--model policy --no-tools:off"
    assert await run("local", "keep") == "--model policy:"
    assert await run("task_sandbox", "default") == "--model policy:"
    assert await run("task_sandbox", "disable") == "--model policy --no-tools:off"


@pytest.mark.asyncio
async def test_final_message_from_json_file_and_last_stdout_match(tmp_path):
    payload = json.dumps({"info": {"submission": "patched", "exit_status": "Submitted"}})
    from_json = shell_spec(
        f"printf '%s' '{payload}' > {{state_dir}}/out.json",
        final_message={"file": "{state_dir}/out.json", "json_key": "info.submission"},
    )
    missing_key = shell_spec(
        f"printf '%s' '{payload}' > {{state_dir}}/out.json",
        final_message={"file": "out.json", "json_key": "info.missing"},
    )
    from_stdout = shell_spec("echo 'FINAL: one'; echo 'FINAL: two'", final_message={"stdout_regex": r"FINAL: (\w+)"})
    runner = make_runner(tmp_path)
    session = make_session(tmp_path, sandbox=True)

    async def final(spec) -> str | None:
        result = await runner.run(spec, session=session, agent=agent(), base_url="x", placement="task_sandbox")
        return result.final_message

    assert await final(from_json) == "patched"
    assert await final(missing_key) is None
    assert await final(from_stdout) == "two"


def test_resolve_placement_prefers_the_task_sandbox(tmp_path):
    with_sandbox = make_session(tmp_path, sandbox=True)
    without = make_session(tmp_path, sandbox=False)

    assert resolve_placement("auto", with_sandbox) == "task_sandbox"
    assert resolve_placement("auto", without) == "local"
    assert resolve_placement("local", with_sandbox) == "local"
    with pytest.raises(HarnessSetupError, match="needs an env session with a sandbox"):
        resolve_placement("task_sandbox", without)
