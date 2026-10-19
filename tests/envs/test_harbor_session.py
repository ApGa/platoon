from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from platoon.envs.harbor import HarborEnvironmentService, HarborSession, load_task
from platoon.envs.service import PLATOON_ZERO_KEY, EnvironmentService, EnvSession, EnvSetupError, get_service
from platoon.sandbox import CWD_ERROR_EXIT_CODE, CWD_ERROR_PREFIX, ExecResult, SandboxSpec
from platoon.sandbox.process import ProcessProvider, ProcessSandbox
from platoon.task import Task

pytestmark = pytest.mark.asyncio

VERIFIER_DIR = '"${PLATOON_SANDBOX_ROOT:-}/logs/verifier"'


async def _open(service: HarborEnvironmentService, task_id: str, rollout_id: str = "rollout-1") -> HarborSession:
    return await service.open(await service.get_task(task_id), rollout_id=rollout_id)


def _host(session: HarborSession, path: str) -> Path:
    assert session.sandbox is not None
    return session.sandbox.host_path(path)


def _text(result) -> str:
    return result.content[0]["text"]


async def test_open_prepares_the_task_sandbox(service: HarborEnvironmentService, provider: ProcessProvider):
    session = await _open(service, "hello-file")
    try:
        assert isinstance(service, EnvironmentService) and isinstance(session, EnvSession)
        assert session.instruction == "Create the file `/app/hello.txt` containing the single line `hello`.\n"
        assert session.agent_timeout_s == 120.0
        for directory in ["/app", "/logs/agent", "/logs/verifier", "/logs/artifacts"]:
            assert _host(session, directory).is_dir()
        assert session.sandbox.spec.image == "docker://ubuntu:24.04"
        assert session.sandbox.spec.labels == {
            "platoon.env": "harbor",
            "platoon.rollout_id": "rollout-1",
            "platoon.task": "platoon/hello-file",
        }
        assert [tool.name for tool in await session.tools()] == ["bash", "view", "write", "edit"]
    finally:
        await session.close()
    assert list(provider.root.iterdir()) == []


async def test_grading_scores_the_sandbox_state(service: HarborEnvironmentService):
    session = await _open(service, "hello-file")
    try:
        unsolved = await session.grade()
        assert (unsolved.status, unsolved.reward) == ("scored", 0.0)
        assert "missing or wrong" in unsolved.metadata["test_output"]

        solved = await session.run_solution()
        assert solved.exit_code == 0
        outcome = await session.grade()
        assert (outcome.status, outcome.reward, outcome.metadata["verifier_exit_code"]) == ("scored", 1.0, 0)
        assert not _host(session, "/tests").exists()  # tests never linger where agents can read them
    finally:
        await session.close()


async def test_failing_task_scores_zero_with_named_components(service: HarborEnvironmentService):
    session = await _open(service, "failing-check")
    try:
        outcome = await session.grade()
        assert outcome.trainable and outcome.rewards == {"reward": 0.0, "checks_passed": 0.0}
        assert [server.name for server in session.mcp_servers] == ["calculator", "docs"]
    finally:
        await session.close()


async def test_broken_verifier_is_an_error_not_zero(service: HarborEnvironmentService):
    session = await _open(service, "broken-verifier")
    try:
        # A reward the agent forged before grading must not count.
        await session.call_tool("bash", {"command": f"echo 1 > {VERIFIER_DIR}/reward.txt"})
        outcome = await session.grade()
        assert (outcome.status, outcome.failure_kind, outcome.reason) == (
            "error",
            "verifier_no_reward",
            "reward_file_missing",
        )
        assert outcome.reward is None and not outcome.trainable
        assert outcome.metadata["verifier_exit_code"] == 1
        assert "evaluator crashed" in outcome.metadata["test_output"]
    finally:
        await session.close()


async def test_verifier_declared_invalid_grades_are_excluded(service: HarborEnvironmentService):
    session = await _open(service, "invalid-grade")
    try:
        outcome = await session.grade()
        assert (outcome.status, outcome.failure_kind, outcome.reward) == ("invalid", "judge_unavailable", None)
    finally:
        await session.close()


async def test_verifier_timeouts_are_invalid(provider: ProcessProvider, make_task):
    task_dir = make_task("slow", task_toml="[verifier]\ntimeout_sec = 0.5\n", test_sh="sleep 30")
    service = HarborEnvironmentService(task_dirs=[task_dir], provider=provider, default_image="local")
    session = await _open(service, "slow")
    try:
        outcome = await session.grade()
        assert (outcome.status, outcome.failure_kind, outcome.metadata["timeout_s"]) == (
            "invalid",
            "verifier_timeout",
            0.5,
        )
    finally:
        await session.close()


async def test_verifier_env_must_resolve_on_the_host(provider: ProcessProvider, make_task, monkeypatch):
    monkeypatch.delenv("PLATOON_TEST_JUDGE_KEY", raising=False)
    task_dir = make_task("needs-key", task_toml='[verifier.env]\nKEY = "${PLATOON_TEST_JUDGE_KEY}"\n')
    service = HarborEnvironmentService(task_dirs=[task_dir], provider=provider, default_image="local")
    session = await _open(service, "needs-key")
    try:
        outcome = await session.grade()
        assert (outcome.status, outcome.failure_kind) == ("error", "verifier_setup")
        assert "PLATOON_TEST_JUDGE_KEY" in outcome.reason
    finally:
        await session.close()


async def test_generic_tools_act_on_the_sandbox(service: HarborEnvironmentService):
    session = await _open(service, "hello-file")
    try:
        written = await session.call_tool("write", {"path": "notes/plan.md", "content": "one\ntwo\nthree\n"})
        assert not written.is_error and _host(session, "/app/notes/plan.md").read_text() == "one\ntwo\nthree\n"

        viewed = await session.call_tool("view", {"path": "/app/notes/plan.md", "view_range": [2, -1]})
        assert _text(viewed) == "     2\ttwo\n     3\tthree"
        listing = await session.call_tool("view", {"path": "/app"})
        assert _text(listing) == "/app:\n.\n./notes\n./notes/plan.md\n"

        edited = await session.call_tool("edit", {"path": "notes/plan.md", "old_str": "two", "new_str": "2"})
        assert not edited.is_error and "     2\t2" in _text(edited)
        assert _host(session, "/app/notes/plan.md").read_text() == "one\n2\nthree\n"

        ran = await session.call_tool("bash", {"command": "cat notes/plan.md | wc -l; ls missing"})
        assert not ran.is_error
        assert _text(ran).startswith("3\n\n[stderr]\nls: cannot access 'missing'")
        assert _text(ran).endswith("[exit code 2]")

        await session.call_tool("bash", {"command": "echo hello > hello.txt"})
        assert (await session.grade()).reward == 1.0
    finally:
        await session.close()


@pytest.mark.parametrize(
    ("name", "arguments", "message"),
    [
        ("browse", {}, "Unknown tool 'browse'"),
        ("bash", {}, "'command' must be a string"),
        ("view", {"path": "/app/missing.txt"}, "No such file or directory"),
        ("view", {"path": "/app/a.txt", "view_range": [3, 9]}, "outside the file's 2 lines"),
        ("edit", {"path": "/app/a.txt", "old_str": "x", "new_str": "y"}, "found 2 occurrences"),
        ("edit", {"path": "/app/missing.txt", "old_str": "x", "new_str": "y"}, "No such file"),
        ("edit", {"path": "/app/a.txt", "old_str": "", "new_str": "y"}, "must not be empty"),
    ],
)
async def test_tool_misuse_is_reported_as_an_error_result(service, name, arguments, message):
    session = await _open(service, "hello-file")
    try:
        await session.call_tool("write", {"path": "/app/a.txt", "content": "x\nx\n"})
        result = await session.call_tool(name, arguments)
        assert result.is_error and message in _text(result)
    finally:
        await session.close()


async def test_shared_forks_share_state_and_the_last_close_stops_the_sandbox(service: HarborEnvironmentService):
    parent = await _open(service, "hello-file")
    child = await parent.fork(task=Task(goal="Write hello.txt for your parent.", misc=parent.task.misc))

    assert child.sandbox is parent.sandbox
    assert child.instruction == "Write hello.txt for your parent."
    await child.call_tool("bash", {"command": "echo hello > hello.txt"})
    assert (await parent.grade()).reward == 1.0

    root = parent.sandbox.root
    await parent.close()
    assert parent.sandbox is None and root.exists()
    assert (await child.grade()).reward == 1.0
    await child.close()
    assert not root.exists()


async def test_copy_forks_restore_an_independent_snapshot(service: HarborEnvironmentService):
    parent = await _open(service, "hello-file")
    try:
        await parent.call_tool("bash", {"command": "echo hello > hello.txt"})
        copy = await parent.fork("copy")
        try:
            assert copy.sandbox is not parent.sandbox
            await parent.call_tool("bash", {"command": "rm hello.txt"})
            assert (await copy.grade()).reward == 1.0
            assert (await parent.grade()).reward == 0.0
        finally:
            await copy.close()
    finally:
        await parent.close()


async def test_fresh_forks_start_a_new_sandbox(service: HarborEnvironmentService):
    parent = await _open(service, "hello-file")
    try:
        await parent.call_tool("bash", {"command": "echo hello > hello.txt"})
        fresh = await parent.fork("fresh")
        try:
            assert fresh.rollout_id == parent.rollout_id
            assert not _host(fresh, "/app/hello.txt").exists()
            assert (await fresh.grade()).reward == 0.0
        finally:
            await fresh.close()
    finally:
        await parent.close()


async def test_copy_forks_need_a_snapshot_capable_provider(provider: ProcessProvider, harbor_fixtures: Path):
    class NoSnapshots(ProcessProvider):
        capabilities = frozenset()

    service = HarborEnvironmentService(harbor_fixtures, provider=NoSnapshots(provider.root))
    session = await _open(service, "hello-file")
    try:
        with pytest.raises(NotImplementedError, match="cannot snapshot"):
            await session.fork("copy")
        with pytest.raises(ValueError, match="Unknown fork mode"):
            await session.fork("clone")  # type: ignore[arg-type]
    finally:
        await session.close()


async def test_closed_sessions_fail_closed(service: HarborEnvironmentService):
    session = await _open(service, "hello-file")
    await session.close()
    await session.close()

    assert (await session.grade()).failure_kind == "session_closed"
    assert (await session.call_tool("bash", {"command": "true"})).is_error
    with pytest.raises(RuntimeError, match="closed"):
        await session.fork()


async def test_cancelled_grading_propagates_and_removes_the_tests(provider: ProcessProvider, make_task):
    service = HarborEnvironmentService(
        task_dirs=[make_task("slow", test_sh="sleep 30")], provider=provider, default_image="local"
    )
    session = await _open(service, "slow")
    try:
        grading = asyncio.create_task(session.grade())
        await asyncio.sleep(0.5)
        assert _host(session, "/tests/test.sh").exists()
        grading.cancel()
        with pytest.raises(asyncio.CancelledError):
            await grading
        assert not _host(session, "/tests").exists()
    finally:
        await session.close()


async def test_tasks_resolve_by_name_path_or_loader(
    service: HarborEnvironmentService, provider: ProcessProvider, harbor_fixtures: Path
):
    lister = HarborEnvironmentService(harbor_fixtures, splits="splits.json", provider="no.such:Provider")
    assert await lister.list_tasks("eval") == ["invalid-grade"]  # listing never builds the provider
    by_path = await service.get_task(str((harbor_fixtures / "hello-file").resolve()))
    assert by_path.misc["harbor"]["name"] == "platoon/hello-file"

    # Workers build a service per rollout and open loader-built or named tasks without indexing the dataset.
    worker_service = get_service(
        "harbor", dataset_dir=str(harbor_fixtures), provider="process", provider_options={"root": str(provider.root)}
    )
    loaded = await worker_service.open(load_task(str(harbor_fixtures / "invalid-grade")), rollout_id="r1")
    named = await worker_service.open(await worker_service.get_task("hello-file"), rollout_id="r2")
    try:
        assert (await loaded.grade()).status == "invalid"
        assert named.harbor_task.name == "platoon/hello-file"
        assert "dataset" not in vars(worker_service)
    finally:
        await loaded.close()
        await named.close()
    with pytest.raises(KeyError, match="Unknown Harbor task '../hello-file'"):
        await worker_service.get_task("../hello-file")


async def test_open_failures_are_classified_and_clean_up(provider: ProcessProvider, make_task, monkeypatch):
    unhealthy = make_task("unhealthy", task_toml='[environment.healthcheck]\ncommand = "false"\ninterval_sec = 0.01\n')
    no_image = make_task("no-image")
    monkeypatch.delenv("PLATOON_TEST_UNSET", raising=False)
    needs_env = make_task(
        "needs-env", task_toml='[environment]\ndocker_image = "x"\nenv = {A = "${PLATOON_TEST_UNSET}"}\n'
    )
    service = HarborEnvironmentService(task_dirs=[unhealthy, no_image, needs_env], provider=provider)

    with pytest.raises(EnvSetupError, match="no default image") as no_image_error:
        await service.open(load_task(str(no_image)), rollout_id="r")
    assert no_image_error.value.failure_kind == "env_config"
    with pytest.raises(EnvSetupError) as env_error:
        await service.open(await service.get_task("needs-env"), rollout_id="r")
    assert env_error.value.failure_kind == "env_config"

    service.default_image = "local"
    with pytest.raises(EnvSetupError, match="failed 3 times") as health_error:
        await service.open(await service.get_task("unhealthy"), rollout_id="r")
    assert health_error.value.failure_kind == "env_healthcheck"
    assert list(provider.root.iterdir()) == []

    class Broken(ProcessProvider):
        async def start(self, spec: SandboxSpec):
            raise OSError("no capacity")

    broken = HarborEnvironmentService(task_dirs=[no_image], provider=Broken(), default_image="local")
    with pytest.raises(EnvSetupError, match="no capacity") as start_error:
        await broken.open(await broken.get_task("no-image"), rollout_id="r")
    assert start_error.value.failure_kind == "sandbox_start"


@pytest.mark.parametrize("image", [None, "ubuntu:24.04"])
async def test_a_multi_container_task_is_never_started_or_scored(provider: ProcessProvider, make_task, image):
    """RFC 0010 P0: a compose task's main container alone would run without its judge, and its verifier would
    write 0.  Whether its image comes from the task or the service's default, the session never opens."""

    started: list[SandboxSpec] = []

    class Recording(ProcessProvider):
        async def start(self, spec: SandboxSpec):
            started.append(spec)
            return await super().start(spec)

    task_toml = "" if image is None else f'[environment]\ndocker_image = "{image}"\n'
    stack = make_task("stack", task_toml=task_toml)
    (stack / "environment" / "docker-compose.yaml").write_text("services:\n  judge:\n    image: judge\n")
    service = HarborEnvironmentService(
        task_dirs=[make_task("single")], provider=Recording(root=provider.root), default_image="local"
    )

    with pytest.raises(ValueError, match="multi-container"):
        await service.get_task(str(stack))  # loading the task refuses it
    task = Task(id="stack", goal="Do the task.", misc={"harbor": {"task_dir": str(stack)}})  # one made elsewhere
    with pytest.raises(EnvSetupError, match="multi-container") as refused:
        await service.open(task, rollout_id="r")  # and so does opening a session for it
    assert refused.value.failure_kind == "env_config"
    assert started == [] and not any(provider.root.glob("*"))  # no sandbox was ever started


# ------------------------------------------------------------------------------------------------ shared sandboxes


async def test_calls_from_shared_forks_take_turns_unless_the_tool_is_declared_concurrent(provider, harbor_fixtures):
    waiter = "for i in $(seq 40); do [ -f flag ] && { echo saw; exit 0; }; sleep 0.02; done; echo timeout"
    for concurrent, heard in (((), "timeout"), (("bash",), "saw")):
        service = HarborEnvironmentService(harbor_fixtures, provider=provider, concurrent_tools=concurrent)
        parent = await _open(service, "hello-file")
        child = await parent.fork()
        try:
            wait = asyncio.ensure_future(parent.call_tool("bash", {"command": waiter}))
            await asyncio.sleep(0.1)
            await child.call_tool("bash", {"command": "touch flag"})
            assert _text(await wait).strip() == heard  # serial: the touch waited for the wait to give up
        finally:
            await child.close()
            await parent.close()


async def test_concurrent_edits_of_one_file_all_apply(service: HarborEnvironmentService):
    parent = await _open(service, "hello-file")
    forks = [await parent.fork() for _ in range(2)]
    try:
        await parent.call_tool("write", {"path": "list.txt", "content": "a\nb\nc\n"})
        edits = [("a", "A"), ("b", "B"), ("c", "C")]
        results = await asyncio.gather(
            *(
                session.call_tool("edit", {"path": "list.txt", "old_str": old, "new_str": new})
                for session, (old, new) in zip([parent, *forks], edits)
            )
        )
        assert not any(result.is_error for result in results)
        assert _host(parent, "/app/list.txt").read_text() == "A\nB\nC\n"  # each edit read the one before it
    finally:
        for fork in forks:
            await fork.close()
        await parent.close()


async def test_grading_waits_for_the_calls_in_flight_and_no_call_sees_the_tests(provider, make_task):
    service = HarborEnvironmentService(
        task_dirs=[
            make_task("slow-check", test_sh=f"sleep 0.3; [ -f hello.txt ] && echo 1 > {VERIFIER_DIR}/reward.txt")
        ],
        provider=provider,
        default_image="local",
    )
    session = await _open(service, "slow-check")
    child = await session.fork()
    try:
        writing = asyncio.ensure_future(child.call_tool("bash", {"command": "sleep 0.2; echo hello > hello.txt"}))
        await asyncio.sleep(0.05)
        grading = asyncio.ensure_future(session.grade())
        await asyncio.sleep(0.05)
        peek = await child.call_tool("bash", {"command": "ls /tests 2>&1; true"})  # queued behind the grade
        assert grading.done() and (await grading).reward == 1.0  # it graded after the write in flight
        assert "No such file or directory" in _text(peek)
        await writing
    finally:
        await child.close()
        await session.close()


async def test_a_call_not_admitted_within_the_queue_deadline_is_not_run(provider, harbor_fixtures, caplog):
    service = HarborEnvironmentService(harbor_fixtures, provider=provider, queue_timeout_s=0.1)
    session = await _open(service, "hello-file")
    child = await session.fork()
    try:
        holder = asyncio.ensure_future(session.call_tool("bash", {"command": "sleep 0.5"}))
        await asyncio.sleep(0.05)
        with caplog.at_level("WARNING", logger="platoon.envs.harbor.session"):
            busy = await child.call_tool("bash", {"command": "echo late > late.txt"})
        assert busy.is_error and _text(busy).startswith("Not run: env busy.")
        assert busy.metadata == {"error_reason": "env_busy"} and "not run: env busy" in caplog.text
        await holder
        assert not _host(session, "/app/late.txt").exists()
    finally:
        await child.close()
        await session.close()


async def test_a_grade_that_cannot_start_in_time_is_env_busy(provider, harbor_fixtures):
    service = HarborEnvironmentService(harbor_fixtures, provider=provider, queue_timeout_s=0.1)
    session = await _open(service, "hello-file")
    child = await session.fork()
    try:
        holder = asyncio.ensure_future(child.call_tool("bash", {"command": "sleep 0.6"}))
        await asyncio.sleep(0.05)
        outcome = await session.grade()
        assert (outcome.status, outcome.failure_kind) == ("error", "env_busy")
        assert re.fullmatch(r"grading waited \d+\.\ds for the calls in flight", outcome.reason or "")
        await holder
        assert (await session.grade()).status == "scored"  # its turn comes once the call ends
    finally:
        await child.close()
        await session.close()


async def test_the_last_close_stops_the_sandbox_after_the_calls_in_flight(service: HarborEnvironmentService):
    session = await _open(service, "hello-file")
    assert session.sandbox is not None
    events: list[str] = []
    sandbox, tools = session.sandbox, session._tools
    stop, call = sandbox.stop, tools.call

    async def recorded_stop() -> None:
        events.append("stop")
        await stop()

    async def recorded_call(name, arguments):
        try:
            return await call(name, arguments)
        finally:
            events.append(f"{name} ended")

    sandbox.stop = recorded_stop  # type: ignore[method-assign]
    tools.call = recorded_call  # type: ignore[method-assign]
    abandoned = asyncio.ensure_future(session.call_tool("bash", {"command": "sleep 0.3"}))
    await asyncio.sleep(0.05)
    abandoned.cancel()
    await session.close()
    assert events == ["bash ended", "stop"]


async def test_a_copy_fork_snapshots_after_the_calls_in_flight(service: HarborEnvironmentService):
    parent = await _open(service, "hello-file")
    child = await parent.fork()
    try:
        writing = asyncio.ensure_future(child.call_tool("bash", {"command": "sleep 0.2; echo hello > hello.txt"}))
        await asyncio.sleep(0.05)
        copy = await parent.fork("copy")
        try:
            assert writing.done()
            assert (await copy.grade()).reward == 1.0
        finally:
            await copy.close()
    finally:
        await child.close()
        await parent.close()


async def test_concurrency_options_are_checked(harbor_fixtures):
    with pytest.raises(ValueError, match="unknown tools"):
        HarborEnvironmentService(harbor_fixtures, concurrent_tools=["browse"])
    with pytest.raises(ValueError, match="queue_timeout_s"):
        HarborEnvironmentService(harbor_fixtures, queue_timeout_s=0)
    with pytest.raises(ValueError, match="close_timeout_s"):
        HarborEnvironmentService(harbor_fixtures, close_timeout_s=0)
    assert HarborEnvironmentService(harbor_fixtures, concurrent_tools="bash").concurrent_tools == {"bash"}


# ------------------------------------------------------------------------------------------- tamper-proof grading


def _running(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except FileNotFoundError:
        return False


async def test_a_reward_file_symlinked_by_code_the_tests_ran_scores_zero(provider, make_task, tmp_path: Path):
    """Tests that run the agent's code let it replace the reward file; a symlink would read a host file ("1"). The
    tests' output it replaces with a symlink is left out of the outcome rather than read from a host file."""

    one, secret = tmp_path / "host-file", tmp_path / "host-secret"
    one.write_text("1\n")
    secret.write_text("a host secret\n")
    test_sh = f"echo 0 > {VERIFIER_DIR}/reward.txt; [ -f hook.sh ] && bash hook.sh; true"
    service = HarborEnvironmentService(
        task_dirs=[make_task("runs-agent-code", test_sh=test_sh)], provider=provider, default_image="local"
    )
    session = await _open(service, "runs-agent-code")
    try:
        hook = (
            f'ln -sfn {one} "$PLATOON_SANDBOX_ROOT/logs/verifier/reward.txt"\n'
            f'ln -sfn {secret} "$PLATOON_SANDBOX_ROOT/logs/verifier/test-stdout.txt"\n'
        )
        assert not (await session.call_tool("write", {"path": "hook.sh", "content": hook})).is_error
        outcome = await session.grade()
        assert (outcome.status, outcome.reward, outcome.reason) == ("scored", 0.0, "reward_file_not_regular")
        assert outcome.trainable and outcome.metadata["reward_files"] == {"reward.txt": "symlink"}
        assert "test_output" not in outcome.metadata
    finally:
        await session.close()


async def test_reward_files_the_agent_left_before_grading_are_ignored(service: HarborEnvironmentService, tmp_path):
    one = tmp_path / "reward.json"
    one.write_text('{"reward": 1}')
    session = await _open(service, "hello-file")
    try:
        planted = await session.call_tool(
            "bash", {"command": f"echo 1 > /logs/verifier/reward.txt; ln -s {one} /logs/verifier/reward.json"}
        )
        assert not planted.is_error and _host(session, "/logs/verifier/reward.json").is_symlink()
        outcome = await session.grade()
        assert (outcome.status, outcome.rewards, outcome.reason) == ("scored", {"reward": 0.0}, None)
        assert "reward_files" not in outcome.metadata  # the tests wrote the only reward file read
    finally:
        await session.close()


async def test_processes_the_agent_left_running_cannot_race_the_tests(service: HarborEnvironmentService, tmp_path):
    """A no-op agent leaves loops that keep the reward files saying 1 (one moved to a session of its own); grading
    kills them first, so the tests' 0 stands, and they are gone afterwards."""

    one = tmp_path / "one"
    one.write_text("1\n")
    symlink_loop = f"while :; do mkdir -p /logs/verifier; ln -sfn {one} /logs/verifier/reward.txt; sleep 0.005; done"
    copy_loop = "while :; do mkdir -p /logs/verifier; cp /tmp/r.json /logs/verifier/reward.json; sleep 0.005; done"
    session = await _open(service, "hello-file")
    try:
        await session.call_tool("write", {"path": "/tmp/r.json", "content": '{"reward": 1}'})
        started = await session.call_tool(
            "bash",
            {
                "command": f"nohup sh -c '{symlink_loop}' >/dev/null 2>&1 & echo $! > /tmp/loop.pid; "
                f"setsid sh -c 'echo $$ > /tmp/detached.pid; {copy_loop}' </dev/null >/dev/null 2>&1 & sleep 0.2"
            },
        )
        assert not started.is_error, _text(started)
        pids = [int(_host(session, f"/tmp/{name}.pid").read_text()) for name in ("loop", "detached")]
        assert all(_running(pid) for pid in pids)

        outcome = await session.grade()

        assert (outcome.status, outcome.rewards, outcome.reason) == ("scored", {"reward": 0.0}, None), outcome
        assert "missing or wrong" in outcome.metadata["test_output"]
        assert not any(_running(pid) for pid in pids)
        assert (await session.call_tool("bash", {"command": "echo hello > hello.txt"})).is_error is False
        assert (await session.grade()).reward == 1.0  # the sandbox works on
    finally:
        await session.close()


@pytest.mark.parametrize("launch", ["", "setsid"], ids=["in-its-group", "own-session"])
async def test_a_process_left_rewriting_reward_txt_during_grading_earns_nothing(
    service: HarborEnvironmentService, launch: str
):
    """A no-op agent leaves a process that waits for grading to start (the tests appear in /tests), then writes 1
    into reward.txt, a regular file, again and again, so it would overwrite the tests' 0; grading kills it first."""

    writer = (
        "echo $$ > /tmp/writer.pid; while [ ! -e /tests/test.sh ]; do sleep 0.005; done; "
        "while :; do echo 1 > /logs/verifier/reward.txt; done"
    )
    session = await _open(service, "hello-file")
    try:
        started = await session.call_tool(
            "bash", {"command": f"{launch} sh -c '{writer}' </dev/null >/dev/null 2>&1 & sleep 0.2"}
        )
        assert not started.is_error, _text(started)
        writer_pid = int(_host(session, "/tmp/writer.pid").read_text())
        assert _running(writer_pid)

        outcome = await session.grade()

        assert (outcome.status, outcome.rewards, outcome.reason) == ("scored", {"reward": 0.0}, None), outcome
        assert outcome.trainable and "missing or wrong" in outcome.metadata["test_output"]
        assert not _running(writer_pid)
    finally:
        await session.close()


async def test_processes_the_tests_left_running_are_killed_before_their_output_is_read(
    provider, make_task, monkeypatch
):
    """Tests that run the agent's code can leave its processes running after test.sh exits, where they could still
    change the reward files while grading reads them: grading kills what the tests left, then reads their output."""

    test_sh = f"echo 0 > {VERIFIER_DIR}/reward.txt; [ -f hook.sh ] && bash hook.sh; true"
    service = HarborEnvironmentService(
        task_dirs=[make_task("runs-agent-code", test_sh=test_sh)], provider=provider, default_image="local"
    )
    session = await _open(service, "runs-agent-code")
    try:
        pid_file = '"$PLATOON_SANDBOX_ROOT/tmp/left.pid"'
        hook = (  # a process in a session of its own, which outlives the tests
            f"setsid sh -c 'echo $$ > {pid_file}; exec sleep 1000' </dev/null >/dev/null 2>&1 &\n"
            f"while [ ! -s {pid_file} ]; do sleep 0.01; done\n"
        )
        assert not (await session.call_tool("write", {"path": "hook.sh", "content": hook})).is_error
        sandbox = session.sandbox
        assert sandbox is not None
        steps: list[str] = []
        kill, run = sandbox.kill_processes, sandbox.exec

        async def recorded_kill() -> None:
            steps.append("kill")
            await kill()

        async def recorded_exec(cmd, **options):
            argv = cmd.split() if isinstance(cmd, str) else cmd
            steps.append(argv[3] if argv[:2] == ["/bin/sh", "-c"] else argv[0])  # a grading script's name, or a program
            return await run(cmd, **options)

        monkeypatch.setattr(sandbox, "kill_processes", recorded_kill)
        monkeypatch.setattr(sandbox, "exec", recorded_exec)

        outcome = await session.grade()

        assert (outcome.status, outcome.rewards, outcome.reason) == ("scored", {"reward": 0.0}, None), outcome
        between = steps[steps.index("bash") + 1 : steps.index("platoon-collect")]  # after the tests, before the read
        assert "kill" in between, steps
        assert not _running(int(_host(session, "/tmp/left.pid").read_text()))
    finally:
        await session.close()


@pytest.mark.parametrize("launch", ["", "setsid"], ids=["in-its-group", "own-session"])
async def test_a_process_the_tests_left_running_cannot_change_their_reward(provider, make_task, launch: str):
    """Tests that run the agent's code may leave a process of it running, which waits for the tests to end and then
    writes 1 into reward.txt, a regular file: grading kills what the tests left before it reads their output, so the
    tests' 0 stands."""

    test_sh = f"echo 0 > {VERIFIER_DIR}/reward.txt; [ -f hook.sh ] && bash hook.sh; true"
    service = HarborEnvironmentService(
        task_dirs=[make_task("runs-agent-code", test_sh=test_sh)], provider=provider, default_image="local"
    )
    session = await _open(service, "runs-agent-code")
    try:
        rewriter = (
            'echo $$ > "$2"; while [ ! -e "$1/go" ]; do sleep 0.01; done; echo 1 > "$1/reward.txt"; touch "$1/done"'
        )
        pid_file = '"$PLATOON_SANDBOX_ROOT/tmp/rewriter.pid"'
        hook = (
            f"{launch} sh -c '{rewriter}' rewriter \"$PLATOON_SANDBOX_ROOT/logs/verifier\" {pid_file} "
            "</dev/null >/dev/null 2>&1 &\n"
            f"while [ ! -s {pid_file} ]; do sleep 0.01; done\n"
        )
        assert not (await session.call_tool("write", {"path": "hook.sh", "content": hook})).is_error
        sandbox = session.sandbox
        verifier, exec_ = sandbox.host_path("/logs/verifier"), sandbox.exec

        async def let_it_rewrite_before_the_output_is_read(cmd, **kwargs):
            if "platoon-collect" in cmd:
                (verifier / "go").touch()
                for _ in range(50):  # a live rewriter answers within milliseconds
                    if (verifier / "done").exists():
                        break
                    await asyncio.sleep(0.02)
            return await exec_(cmd, **kwargs)

        sandbox.exec = let_it_rewrite_before_the_output_is_read  # type: ignore[method-assign]
        outcome = await session.grade()

        assert (outcome.status, outcome.rewards, outcome.reason) == ("scored", {"reward": 0.0}, None), outcome
        assert not (verifier / "done").exists()
        assert not _running(int(_host(session, "/tmp/rewriter.pid").read_text()))
    finally:
        await session.close()


@pytest.mark.skipif(os.geteuid() == 0, reason="root removes what it locked without its permissions")
async def test_directories_the_agent_locked_are_made_afresh(service: HarborEnvironmentService):
    """Directories an agent leaves in /tests and /logs/verifier without permissions, which rm alone cannot empty,
    would make grading an untrainable error; grading gives them their permissions back and removes them, so the
    attempt scores 0 like any other."""

    session = await _open(service, "hello-file")
    try:
        locked = await session.call_tool(
            "bash",
            {
                "command": "mkdir -p /logs/verifier/a/b /tests/c && echo 1 > /logs/verifier/a/b/reward.txt && "
                "chmod 000 /logs/verifier/a/b /logs/verifier/a /logs/verifier /tests/c /tests"
            },
        )
        assert not locked.is_error, _text(locked)
        assert not os.access(_host(session, "/logs/verifier"), os.R_OK)

        outcome = await session.grade()

        assert (outcome.status, outcome.rewards, outcome.reason) == ("scored", {"reward": 0.0}, None), outcome
        assert outcome.trainable and "missing or wrong" in outcome.metadata["test_output"]
    finally:
        # Should grading leave them locked, unlock them, or neither the sandbox nor pytest could remove them.
        subprocess.run(["chmod", "-R", "u+rwX", _host(session, "/logs"), _host(session, "/tests")], check=False)
        await session.close()


@pytest.mark.parametrize("replacement", ["file", "symlink"])
async def test_a_logs_directory_the_agent_replaced_is_made_afresh(
    service: HarborEnvironmentService, tmp_path: Path, replacement: str
):
    """An agent that leaves /logs a file would make grading fail, and one that leaves it a symlink would have grading
    remove what it points to; grading removes such a /logs, makes it afresh, and the attempt scores 0."""

    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "verifier").mkdir(parents=True)
    (elsewhere / "verifier" / "reward.txt").write_text("1\n")
    session = await _open(service, "hello-file")
    try:
        logs = _host(session, "/logs")
        shutil.rmtree(logs)
        if replacement == "file":
            logs.write_text("1\n")
        else:
            logs.symlink_to(elsewhere)

        outcome = await session.grade()

        assert (outcome.status, outcome.rewards, outcome.reason) == ("scored", {"reward": 0.0}, None), outcome
        assert outcome.trainable and "missing or wrong" in outcome.metadata["test_output"]
        assert logs.is_dir() and not logs.is_symlink()
        assert (elsewhere / "verifier" / "reward.txt").read_text() == "1\n"  # what the symlink pointed to stays
    finally:
        await session.close()


async def test_files_the_agent_left_in_tests_are_gone_when_the_tests_run(provider: ProcessProvider, make_task):
    """Tests that take up whatever their directory holds, as pytest takes up a conftest.py, never get a file an agent
    left in /tests before grading: grading removes /tests and uploads only the task's own tests."""

    tests_dir = '"${PLATOON_SANDBOX_ROOT:-}/tests"'
    task_dir = make_task(
        "takes-up-its-directory",
        test_sh=f"if [ -e {tests_dir}/conftest.py ]; then echo 1; else echo 0; fi > {VERIFIER_DIR}/reward.txt",
    )
    service = HarborEnvironmentService(task_dirs=[task_dir], provider=provider, default_image="local")
    session = await _open(service, "takes-up-its-directory")
    try:
        planted = await session.call_tool("write", {"path": "/tests/conftest.py", "content": "# the agent's\n"})
        assert not planted.is_error, _text(planted)
        assert _host(session, "/tests/conftest.py").is_file()

        outcome = await session.grade()

        assert (outcome.status, outcome.rewards, outcome.reason) == ("scored", {"reward": 0.0}, None), outcome
        assert not _host(session, "/tests").exists()  # and the task's tests are gone again
    finally:
        await session.close()


async def test_a_grade_that_cannot_kill_the_agents_processes_is_an_error(
    service: HarborEnvironmentService, monkeypatch
):
    session = await _open(service, "hello-file")
    try:
        await session.call_tool("bash", {"command": "echo hello > hello.txt"})

        async def cannot_kill() -> None:
            raise RuntimeError("platoon: processes still alive after SIGKILL: 42")

        monkeypatch.setattr(session.sandbox, "kill_processes", cannot_kill)
        outcome = await session.grade()
        assert (outcome.status, outcome.failure_kind, outcome.reward) == ("error", "verifier_isolation", None)
        assert outcome.reason is not None and "still alive after SIGKILL: 42" in outcome.reason
        assert not outcome.trainable
        assert not _host(session, "/tests").exists()  # the tests never ran
    finally:
        await session.close()


async def test_sandboxes_without_kill_processes_are_still_graded(provider, harbor_fixtures, caplog):
    from platoon.sandbox.process import ProcessSandbox

    class OlderSandbox(ProcessSandbox):
        kill_processes = None  # a provider written before kill_processes existed

    class OlderProvider(ProcessProvider):
        async def start(self, spec: SandboxSpec):
            sandbox = await super().start(spec)
            sandbox.__class__ = OlderSandbox
            return sandbox

    service = HarborEnvironmentService(harbor_fixtures, provider=OlderProvider(root=provider.root))
    session = await _open(service, "hello-file")
    try:
        with caplog.at_level("WARNING", logger="platoon.envs.harbor.session"):
            assert (await session.grade()).reward == 0.0
            assert (await session.grade()).reward == 0.0
        assert caplog.text.count("OlderSandbox has no kill_processes") == 1
    finally:
        await session.close()


# ------------------------------------------------------------------------------------------- an unusable workdir


class _CommandDidNotStart(RuntimeError):
    """What hpc-sandbox raises (``SandboxError``) for a command that never started for a reason of the sandbox's
    own."""


class _RaisingSandbox(ProcessSandbox):
    """A process sandbox that raises where its working directory cannot be entered, as hpc-sandbox did before 0.2.0:
    the command never starts, and neither do transfers, which copy through commands run in the workdir, even for
    absolute paths."""

    def _enter(self, path: str) -> None:
        directory = self.host_path(path)
        if not (directory.is_dir() and os.access(directory, os.X_OK)):
            raise _CommandDidNotStart(f"sandbox {self.id}: the command did not start (exit 2): can't cd to {path}")

    async def exec(self, cmd, *, timeout_s=None, env=None, cwd=None, stdin=None):
        self._enter(cwd or self.spec.workdir)
        return await super().exec(cmd, timeout_s=timeout_s, env=env, cwd=cwd, stdin=stdin)

    async def upload(self, src, dst):
        self._enter(self.spec.workdir)
        await super().upload(src, dst)

    async def download(self, src, dst):
        self._enter(self.spec.workdir)
        await super().download(src, dst)


class _RaisingProvider(ProcessProvider):
    async def start(self, spec: SandboxSpec) -> _RaisingSandbox:
        sandbox = await super().start(spec)
        sandbox.__class__ = _RaisingSandbox
        return sandbox


def _cannot_enter(directory: Path) -> str | None:
    """Why hpc-sandbox 0.2's prologue cannot enter ``directory`` (one of ``CWD_ERROR_REASONS``), or None."""

    if directory.is_dir() and os.access(directory, os.X_OK):
        return None
    if directory.is_dir():
        return "permission denied"
    return "not a directory" if directory.exists() else "it does not exist or cannot be reached"


def _cwd_line(path: str, reason: str = "it does not exist or cannot be reached") -> str:
    return f"{CWD_ERROR_PREFIX}{path}: {reason}"


class _Exit126Sandbox(ProcessSandbox):
    """A process sandbox that answers as hpc-sandbox 0.2 does where its working directory cannot be entered: the
    command does not start and exits 126, as on docker, with only hpc-sandbox's line on stderr, while transfers,
    which run from "/", still work."""

    async def exec(self, cmd, *, timeout_s=None, env=None, cwd=None, stdin=None):
        self._check_running()
        path = cwd or self.spec.workdir
        if (reason := _cannot_enter(self.host_path(path))) is not None:
            return ExecResult(CWD_ERROR_EXIT_CODE, "", _cwd_line(path, reason) + "\n")
        return await super().exec(cmd, timeout_s=timeout_s, env=env, cwd=cwd, stdin=stdin)


class _Exit126Provider(ProcessProvider):
    async def start(self, spec: SandboxSpec) -> _Exit126Sandbox:
        sandbox = await super().start(spec)
        sandbox.__class__ = _Exit126Sandbox
        return sandbox


@pytest.fixture(params=["process", "raises", "exit-126"])
def any_service(request, provider: ProcessProvider, harbor_fixtures: Path) -> HarborEnvironmentService:
    """The toy tasks on the process provider, whose commands fail to start in a missing workdir with an OSError, on
    one that raises as hpc-sandbox did before 0.2.0, and on one that exits 126 as docker and hpc-sandbox 0.2 do."""

    if request.param == "raises":
        provider = _RaisingProvider(root=provider.root)
    elif request.param == "exit-126":
        provider = _Exit126Provider(root=provider.root)
    return HarborEnvironmentService(harbor_fixtures, provider=provider, splits="splits.json")


def _exits_126(service: HarborEnvironmentService) -> bool:
    return isinstance(service.provider, _Exit126Provider)


SHARED = {"verifier_mode": "shared"}  # what grading adds to every outcome of a shared-mode task
OWN_ZERO = {PLATOON_ZERO_KEY: True}  # what marks a zero platoon gave itself, which trained no reward key

# What an agent does to the workdir (/app), from inside it, and what grading finds there.
UNUSABLE_WORKDIRS = {
    "removed": ("cd / && rm -rf /app", "missing"),
    "solved-then-removed": ("echo hello > hello.txt && cd / && rm -rf /app", "missing"),
    "moved": ("cd / && mv /app /tmp/app", "missing"),
    "a-symlink-to-nothing": ("cd / && rm -rf /app && ln -s /tmp/nowhere /app", "missing"),
    "a-file": ("cd / && rm -rf /app && echo hello > /app", "not_a_directory"),
}


@pytest.mark.parametrize(("command", "state"), UNUSABLE_WORKDIRS.values(), ids=UNUSABLE_WORKDIRS)
async def test_an_agent_that_left_no_workdir_scores_zero(any_service: HarborEnvironmentService, command, state):
    """The tests run in the workdir, where no command starts once it is gone (exit 126 with hpc-sandbox's line, or an
    error): the attempt scores 0 without them, a trainable outcome that says why, and grading leaves the sandbox as
    it is.  Only grading treats it so: the agents' commands fail on as before."""

    session = await _open(any_service, "hello-file")
    try:
        done = await session.call_tool("bash", {"command": command})
        assert not done.is_error, _text(done)
        workdir = _host(session, "/app")
        assert not workdir.is_dir()

        outcome = await session.grade()

        assert (outcome.status, outcome.rewards, outcome.reason) == ("scored", {"reward": 0.0}, "workdir_missing")
        assert outcome.trainable and outcome.metadata == {
            "workdir": "/app",
            "workdir_state": state,
            **SHARED,
            **OWN_ZERO,
        }
        assert not _host(session, "/tests").exists()  # the tests never ran
        assert not workdir.is_dir()  # grading made no workdir to run them in
        later = await session.call_tool("bash", {"command": "true"})
        if _exits_126(any_service):  # as hpc-sandbox 0.2 says it, for the model to read
            reason = {"missing": "it does not exist or cannot be reached", "not_a_directory": "not a directory"}[state]
            assert not later.is_error
            assert _text(later) == f"\n[stderr]\n{_cwd_line('/app', reason)}\n\n[exit code 126]"
        else:
            assert later.is_error
        assert (await session.grade()).reason == "workdir_missing"
    finally:
        await session.close()


@pytest.mark.skipif(os.geteuid() == 0, reason="root enters a directory without its permissions")
async def test_an_agent_that_locked_its_workdir_scores_zero(any_service: HarborEnvironmentService):
    session = await _open(any_service, "hello-file")
    workdir = _host(session, "/app")
    try:
        done = await session.call_tool("bash", {"command": "echo hello > hello.txt && chmod 000 /app"})
        assert not done.is_error, _text(done)

        outcome = await session.grade()

        assert (outcome.status, outcome.rewards, outcome.reason) == ("scored", {"reward": 0.0}, "workdir_missing")
        assert outcome.metadata == {"workdir": "/app", "workdir_state": "not_enterable", **SHARED, **OWN_ZERO}
    finally:
        workdir.chmod(0o755)
        await session.close()


async def test_a_workdir_removed_after_it_was_checked_still_scores_zero(any_service: HarborEnvironmentService):
    """A process the kill missed removes the workdir once the tests are uploaded: their command cannot start, which
    is still the agents' doing."""

    session = await _open(any_service, "hello-file")
    try:
        sandbox = session.sandbox
        upload = sandbox.upload

        async def upload_then_lose_the_workdir(src, dst):
            await upload(src, dst)
            shutil.rmtree(sandbox.host_path("/app"))

        sandbox.upload = upload_then_lose_the_workdir  # type: ignore[method-assign]
        outcome = await session.grade()

        assert (outcome.status, outcome.rewards, outcome.reason) == ("scored", {"reward": 0.0}, "workdir_missing")
        expected = {"workdir": "/app", "workdir_state": "missing", **SHARED, **OWN_ZERO}
        if _exits_126(any_service):  # the tests' command exited 126, after which grading checked the workdir again
            assert outcome.metadata.pop("verifier_seconds") >= 0
            expected["verifier_exit_code"] = 126
            expected["verifier_cwd_error"] = _cwd_line("/app")  # what hpc-sandbox said
        assert outcome.metadata == expected
        assert not _host(session, "/tests").exists()
    finally:
        await session.close()


async def test_a_reward_left_before_the_tests_could_not_start_never_counts(any_service: HarborEnvironmentService):
    """A process the kill missed writes a reward of 1 and removes the workdir once the tests are uploaded: their
    command cannot start (it raises, or exits 126), and grading checks the workdir before it reads any reward file,
    so the attempt scores 0."""

    session = await _open(any_service, "hello-file")
    try:
        sandbox = session.sandbox
        upload = sandbox.upload

        async def upload_then_forge_and_lose_the_workdir(src, dst):
            await upload(src, dst)
            _host(session, "/logs/verifier/reward.txt").write_text("1\n")
            shutil.rmtree(sandbox.host_path("/app"))

        sandbox.upload = upload_then_forge_and_lose_the_workdir  # type: ignore[method-assign]
        outcome = await session.grade()

        assert (outcome.status, outcome.rewards, outcome.reason) == ("scored", {"reward": 0.0}, "workdir_missing")
        assert outcome.trainable and outcome.metadata["workdir_state"] == "missing"
        assert outcome.metadata.get("verifier_exit_code") == (126 if _exits_126(any_service) else None)
        assert outcome.metadata.get("verifier_cwd_error") == (_cwd_line("/app") if _exits_126(any_service) else None)
        assert _host(session, "/logs/verifier/reward.txt").read_text() == "1\n"  # there, and never read
    finally:
        await session.close()


@pytest.mark.parametrize("exit_126", [False, True], ids=["process", "exit-126"])
async def test_tests_that_exit_126_themselves_are_graded_as_before(provider: ProcessProvider, make_task, exit_126):
    """A 126 from the tests themselves, with the workdir in place, is their own exit status: their reward counts."""

    task_dir = make_task("exits-126", test_sh=f"echo 1 > {VERIFIER_DIR}/reward.txt; exit 126")
    if exit_126:
        provider = _Exit126Provider(root=provider.root)
    service = HarborEnvironmentService(task_dirs=[task_dir], provider=provider, default_image="local")
    session = await _open(service, "exits-126")
    try:
        outcome = await session.grade()
        assert (outcome.status, outcome.rewards, outcome.reason) == ("scored", {"reward": 1.0}, None), outcome
        assert outcome.metadata["verifier_exit_code"] == 126 and "verifier_cwd_error" not in outcome.metadata
    finally:
        await session.close()


@pytest.mark.parametrize("reward", ["1", "0"])
async def test_a_126_that_says_the_workdir_could_not_be_entered_is_confirmed_by_the_workdir_check(
    provider: ProcessProvider, make_task, reward: str
):
    """A command may exit 126 by itself and print anything: hpc-sandbox's line alone never decides a grade.  With the
    workdir in place (what grading's own check, run in "/", finds), the tests' reward counts, whatever it is, and the
    outcome records the line."""

    task_dir = make_task("says-126", test_sh=f"echo {reward} > {VERIFIER_DIR}/reward.txt")
    service = HarborEnvironmentService(
        task_dirs=[task_dir], provider=_Exit126Provider(root=provider.root), default_image="local"
    )
    session = await _open(service, "says-126")
    try:
        sandbox = session.sandbox
        exec_ = sandbox.exec

        async def tests_then_say_126(cmd, **kwargs):
            result = await exec_(cmd, **kwargs)
            if "test.sh" in str(cmd):
                return ExecResult(CWD_ERROR_EXIT_CODE, "", _cwd_line("/app") + "\n")
            return result

        sandbox.exec = tests_then_say_126  # type: ignore[method-assign]
        outcome = await session.grade()

        assert (outcome.status, outcome.rewards, outcome.reason) == ("scored", {"reward": float(reward)}, None)
        assert outcome.metadata["verifier_cwd_error"] == _cwd_line("/app")
        assert outcome.metadata["verifier_exit_code"] == 126
    finally:
        await session.close()


async def test_a_sandbox_failure_with_the_workdir_in_place_is_still_an_error(any_service: HarborEnvironmentService):
    """The tests' command fails to start for a reason of the sandbox's own: the workdir is there, so the failure is
    the infrastructure's, an untrainable error as before."""

    session = await _open(any_service, "hello-file")
    try:
        sandbox = session.sandbox
        exec_ = sandbox.exec

        async def tests_cannot_start(cmd, **kwargs):
            if "test.sh" in str(cmd):
                raise _CommandDidNotStart(f"sandbox {sandbox.id}: the command did not start (exit 1): no memory")
            return await exec_(cmd, **kwargs)

        sandbox.exec = tests_cannot_start  # type: ignore[method-assign]
        outcome = await session.grade()

        assert (outcome.status, outcome.failure_kind, outcome.reward) == (
            "error",
            "unexpected:_CommandDidNotStart",
            None,
        )
        assert "no memory" in outcome.metadata["error"] and not outcome.trainable
        assert not _host(session, "/tests").exists()
    finally:
        await session.close()


@pytest.mark.parametrize("failure", ["lost", "no-shell"])
async def test_a_sandbox_that_cannot_check_the_workdir_is_an_error_not_zero(
    any_service: HarborEnvironmentService, failure
):
    """With the workdir gone and the sandbox unable to say so, nothing tells the agents' doing from the sandbox's
    failure: the grade is an untrainable error."""

    session = await _open(any_service, "hello-file")
    try:
        await session.call_tool("bash", {"command": "cd / && rm -rf /app"})
        sandbox = session.sandbox

        async def broken(cmd, **kwargs):
            if failure == "lost":
                raise _CommandDidNotStart(f"sandbox {sandbox.id} lost its container: killed")
            return ExecResult(exit_code=127, stdout="", stderr="/bin/sh: not found")

        sandbox.exec = broken  # type: ignore[method-assign]
        outcome = await session.grade()

        assert outcome.status == "error" and outcome.reward is None and not outcome.trainable
        if failure == "lost":
            assert outcome.failure_kind == "unexpected:_CommandDidNotStart" and "lost" in outcome.metadata["error"]
        else:
            assert (outcome.failure_kind, outcome.reason) == (
                "verifier_setup",
                "cannot check the workdir /app: /bin/sh: not found",
            )
    finally:
        await session.close()


@pytest.mark.parametrize("tests", ["raise", "exit-126"])
async def test_a_sandbox_lost_once_the_tests_are_in_place_is_an_error_not_zero(
    any_service: HarborEnvironmentService, tests
):
    """A process the kill missed plants a reward of 1 and removes the workdir once the tests are uploaded, and then
    the sandbox dies: the tests' command fails (it raises, or exits 126 as one that could not start in its workdir
    does), and so does every command after it, the workdir check included. Nothing tells the agents' doing from the
    sandbox's failure, so the grade is an untrainable error, never a 0 (workdir_missing) or the planted 1."""

    session = await _open(any_service, "hello-file")
    try:
        sandbox = session.sandbox
        upload, exec_ = sandbox.upload, sandbox.exec
        lost = False

        async def upload_then_forge_and_lose_the_workdir(src, dst):
            await upload(src, dst)
            _host(session, "/logs/verifier/reward.txt").write_text("1\n")
            shutil.rmtree(sandbox.host_path("/app"))

        async def tests_then_lose_the_sandbox(cmd, **kwargs):
            nonlocal lost
            if lost:
                raise _CommandDidNotStart(f"sandbox {sandbox.id} lost its container: killed")
            if "test.sh" in str(cmd):
                lost = True
                if tests == "raise":
                    raise _CommandDidNotStart(f"sandbox {sandbox.id} lost its container: killed")
                return ExecResult(CWD_ERROR_EXIT_CODE, "", _cwd_line("/app") + "\n")
            return await exec_(cmd, **kwargs)

        sandbox.upload = upload_then_forge_and_lose_the_workdir  # type: ignore[method-assign]
        sandbox.exec = tests_then_lose_the_sandbox  # type: ignore[method-assign]
        outcome = await session.grade()

        assert outcome.status == "error" and outcome.reward is None and not outcome.trainable, outcome
        assert outcome.failure_kind == "unexpected:_CommandDidNotStart" and "lost" in outcome.metadata["error"]
    finally:
        await session.close()
