from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from harness_fakes import FakeChatServer, FakeGateway, FakeService, answer, bash_call

from platoon.config_defs import RolloutConfig
from platoon.harness.rollout import HARNESS_MISC_KEY, run_harness_rollout
from platoon.harness.runner import HarnessRunner
from platoon.task import Task
from platoon.train.components import EnvironmentRoute

SERVICE = "harness_fakes:FakeService"


@pytest.fixture(autouse=True)
def service_root(tmp_path):
    FakeService.root = tmp_path / "sandboxes"
    FakeService.sessions.clear()
    yield
    FakeService.sessions.clear()


def routed_task(
    harness: str = "platoon-bash", task_id: str = "train-0", *, on_context_overflow: str = "grade", **service_kwargs
) -> Task:
    route = EnvironmentRoute(
        name="fake",
        service=SERVICE,
        service_kwargs=service_kwargs,
        harness=harness,
        on_context_overflow=on_context_overflow,
        resolve_task=True,
    )
    return route.attach(Task(id=task_id, max_steps=7))


def rollout_config(tmp_path: Path, **overrides) -> RolloutConfig:
    return RolloutConfig(output_dir=str(tmp_path / "out"), return_dict=True, verbose=False, **overrides)


def runner(tmp_path: Path) -> HarnessRunner:
    return HarnessRunner(
        logs_dir=tmp_path / "logs",
        state_root=tmp_path / "state",
        sandbox_state_root=str(tmp_path / "sandbox-state"),
        kill_grace_s=0.2,
    )


def spec_file(tmp_path: Path, body: str) -> str:
    path = tmp_path / "spec.toml"
    path.write_text(f'name = "custom"\nversion = "1"\n{body}')
    return str(path)


async def run(tmp_path: Path, task: Task, gateway: FakeGateway, **config) -> dict[str, Any]:
    collection = await run_harness_rollout(
        task, rollout_config(tmp_path, **config), gateway=gateway, runner=runner(tmp_path)
    )
    assert isinstance(collection, dict)
    return collection


@pytest.mark.asyncio
async def test_scored_rollout_resolves_the_task_runs_the_harness_and_grades(tmp_path):
    gateway = FakeGateway(FakeChatServer([bash_call("echo 42 > answer.txt"), answer("done")]))

    collection = await run(tmp_path, routed_task(), gateway)

    (session,) = FakeService.sessions
    task, outcome = gateway.collected[0]
    assert outcome.status == "scored" and outcome.reward == 1.0
    # The service resolved the reference task; the route and the workflow's max_steps carry over.
    assert task.goal == "Write 42 to answer.txt" and task.misc["service_task"] == "train-0"
    assert EnvironmentRoute.of_task(task).harness == "platoon-bash" and task.max_steps == 7
    assert session.task is task and session.graded == 1
    (root,) = collection["trajectories"].values()
    assert root["reward"] == 1.0
    assert [step["misc"]["action_misc"]["completion_id"] for step in root["steps"]] == gateway.server.call_ids
    assert root["finish_message"] == "done"
    assert root["misc"][HARNESS_MISC_KEY]["spec"] == "platoon-bash@1"
    assert root["misc"][HARNESS_MISC_KEY]["placement"] == "task_sandbox"
    assert root["misc"][HARNESS_MISC_KEY]["exit_code"] == 0
    assert gateway.started and gateway.stopped and session.closed and session.sandbox.stopped


@pytest.mark.asyncio
async def test_unsolved_task_is_scored_zero_not_excluded(tmp_path):
    gateway = FakeGateway(FakeChatServer([bash_call("echo 41 > answer.txt"), answer("done")]))

    await run(tmp_path, routed_task(), gateway)

    ((_, outcome),) = gateway.collected
    assert outcome.status == "scored" and outcome.reward == 0.0


@pytest.mark.asyncio
async def test_invalid_grade_passes_through_as_an_untrainable_outcome(tmp_path):
    gateway = FakeGateway(FakeChatServer([answer("done")]))

    await run(tmp_path, routed_task(grade_status="invalid"), gateway)

    ((_, outcome),) = gateway.collected
    assert outcome.status == "invalid" and not outcome.trainable and outcome.reward is None


@pytest.mark.asyncio
async def test_harness_crash_is_an_error_and_skips_grading(tmp_path):
    gateway = FakeGateway(FakeChatServer([answer("never")], api_key="another-agent"))
    gateway.open_agent = _with_key(gateway.open_agent, "wrong-secret")

    collection = await run(tmp_path, routed_task(), gateway)

    ((_, outcome),) = gateway.collected
    assert outcome.status == "error" and outcome.failure_kind == "harness_crash"
    assert "exited with code 2" in outcome.reason
    (session,) = FakeService.sessions
    assert session.graded == 0 and session.closed
    (root,) = collection["trajectories"].values()
    assert root["misc"][HARNESS_MISC_KEY]["exit_code"] == 2


@pytest.mark.asyncio
async def test_harness_timeout_is_a_timeout_outcome(tmp_path):
    harness = spec_file(tmp_path, '[launch]\nargv = ["sleep", "60"]\n')
    gateway = FakeGateway(FakeChatServer([]))

    started = time.monotonic()
    collection = await run(tmp_path, routed_task(harness), gateway, timeout=1)

    assert time.monotonic() - started < 15
    ((_, outcome),) = gateway.collected
    assert outcome.status == "timeout" and outcome.reason == "agent_timeout"
    assert FakeService.sessions[0].graded == 0
    (root,) = collection["trajectories"].values()
    assert root["misc"][HARNESS_MISC_KEY]["timed_out"] is True


@pytest.mark.asyncio
async def test_install_failure_is_a_harness_setup_error(tmp_path):
    harness = spec_file(tmp_path, '[install]\ncommands = ["exit 3"]\n[launch]\nargv = ["true"]\n')
    gateway = FakeGateway(FakeChatServer([]))

    await run(tmp_path, routed_task(harness), gateway)

    ((_, outcome),) = gateway.collected
    assert outcome.status == "error" and outcome.failure_kind == "harness_setup"
    assert FakeService.sessions[0].closed and gateway.stopped


@pytest.mark.asyncio
async def test_grader_exception_fails_closed(tmp_path):
    gateway = FakeGateway(FakeChatServer([answer("done")]))

    await run(tmp_path, routed_task(grade_error=True), gateway)

    ((_, outcome),) = gateway.collected
    assert outcome.status == "error" and outcome.failure_kind == "unexpected:RuntimeError"


@pytest.mark.asyncio
async def test_session_open_failure_propagates_without_starting_the_gateway(tmp_path):
    gateway = FakeGateway(FakeChatServer([]))

    with pytest.raises(RuntimeError, match="sandbox start failed"):
        await run(tmp_path, routed_task(fail_open=True), gateway)

    assert not gateway.started and not gateway.collected


@pytest.mark.asyncio
async def test_cleanup_failures_do_not_discard_a_finished_rollout(tmp_path):
    gateway = FakeGateway(FakeChatServer([answer("done")]))

    async def broken_stop() -> None:
        raise OSError("port already released")

    gateway.stop = broken_stop

    collection = await run(tmp_path, routed_task(), gateway)

    assert collection["trajectories"] and FakeService.sessions[0].closed


@pytest.mark.asyncio
async def test_cancellation_kills_the_harness_and_releases_everything(tmp_path):
    pid_file = tmp_path / "harness.pid"
    harness = spec_file(tmp_path, f'[launch]\nargv = ["sh", "-c", "echo $$ > {pid_file}; sleep 60"]\n')
    gateway = FakeGateway(FakeChatServer([]))
    rollout = asyncio.ensure_future(run(tmp_path, routed_task(harness), gateway))
    deadline = time.monotonic() + 10
    while not (pid_file.is_file() and pid_file.read_text().strip()):
        assert time.monotonic() < deadline, "the harness never started"
        await asyncio.sleep(0.05)

    rollout.cancel()
    with pytest.raises(asyncio.CancelledError):
        await rollout

    (session,) = FakeService.sessions
    assert session.closed and gateway.stopped and not gateway.collected
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while _alive(pid):
        assert time.monotonic() < deadline, "the harness outlived its rollout"
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_local_placement_without_a_task_sandbox(tmp_path):
    gateway = FakeGateway(FakeChatServer([bash_call("pwd"), answer("done")]))

    collection = await run(tmp_path, routed_task(sandbox=False), gateway)

    (root,) = collection["trajectories"].values()
    assert root["misc"][HARNESS_MISC_KEY]["placement"] == "local"
    assert str(tmp_path / "state") in gateway.server.requests[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_tasks_without_a_harness_route_are_rejected(tmp_path):
    gateway = FakeGateway(FakeChatServer([]))

    with pytest.raises(ValueError, match="no environment route with a service and a harness"):
        await run(tmp_path, Task(id="plain"), gateway)


def _with_key(open_agent, api_key):
    def open_with_key(**kwargs):
        agent = open_agent(**kwargs)
        agent.api_key = api_key
        return agent

    return open_with_key


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _overflowed(open_agent):
    """The gateway refused one of the agent's prompts for exceeding the context window."""

    def open_overflowed(**kwargs):
        agent = open_agent(**kwargs)
        agent.context_overflows = 1
        return agent

    return open_overflowed


@pytest.mark.asyncio
@pytest.mark.parametrize("grade_reason", [None, "no_submit"])
async def test_a_crash_after_a_context_overflow_is_graded_as_a_policy_failure(tmp_path, grade_reason):
    gateway = FakeGateway(FakeChatServer([answer("never")], api_key="another-agent"))
    gateway.open_agent = _overflowed(_with_key(gateway.open_agent, "wrong-secret"))

    await run(tmp_path, routed_task(grade_reason=grade_reason), gateway)

    ((_, outcome),) = gateway.collected
    assert outcome.status == "scored" and outcome.trainable and outcome.reward == 0.0
    # context_overflow is the reason even when the grader gives its own, which stays in the metadata.
    assert outcome.reason == "context_overflow" and outcome.metadata.get("grade_reason") == grade_reason
    (session,) = FakeService.sessions
    assert session.graded == 1


@pytest.mark.asyncio
async def test_context_overflow_can_be_excluded_per_environment(tmp_path):
    gateway = FakeGateway(FakeChatServer([answer("never")], api_key="another-agent"))
    gateway.open_agent = _overflowed(_with_key(gateway.open_agent, "wrong-secret"))

    await run(tmp_path, routed_task(on_context_overflow="exclude"), gateway)

    ((_, outcome),) = gateway.collected
    assert outcome.status == "error" and outcome.failure_kind == "context_overflow" and not outcome.trainable
    (session,) = FakeService.sessions
    assert session.graded == 0


@pytest.mark.asyncio
async def test_the_harness_gets_the_tasks_step_budget(tmp_path):
    spec = spec_file(
        tmp_path,
        """
[launch]
argv = ["sh", "-c", "echo {max_steps} > {state_dir}/final.txt"]

[final_message]
file = "{state_dir}/final.txt"
""",
    )
    gateway = FakeGateway(FakeChatServer([]))

    collection = await run(tmp_path, routed_task(harness=spec, expect_text="never"), gateway)

    (root,) = collection["trajectories"].values()
    assert root["finish_message"].strip() == "7"  # routed_task gives the task max_steps=7


class NoRenderer:
    """A renderer for a model gateway that is never asked to sample."""

    def build_generation_prompt(self, messages):
        raise AssertionError("no model call expected")

    def parse_response(self, tokens):
        raise AssertionError("no model call expected")

    def get_stop_sequences(self):
        return []


async def models_status(base_url: str, api_key: str, *, socket_path: str | None = None) -> tuple[int, str]:
    connector = aiohttp.UnixConnector(path=socket_path) if socket_path else None
    async with aiohttp.ClientSession(connector=connector) as http:
        async with http.get(f"{base_url}/v1/models", headers={"Authorization": f"Bearer {api_key}"}) as response:
            return response.status, await response.text()


@pytest.mark.asyncio
async def test_a_rollouts_gateway_takes_only_its_own_rollouts_keys_on_its_tcp_port_and_its_socket():
    from platoon.gateway import ModelGateway
    from platoon.gateway.server import SOCKET_NAME, socket_directory
    from platoon.harness.rollout import EmbeddedGateway

    shared = ModelGateway(sampler=None, renderer=NoRenderer(), tokenizer=None, model_name="none")
    ours, theirs = EmbeddedGateway(shared), EmbeddedGateway(shared)
    short_base = tempfile.mkdtemp(prefix="pgwt-", dir="/tmp")  # short enough for a socket path
    try:
        our_url, their_url = await ours.start(), await theirs.start()
        other = theirs.open_agent(rollout_id="rollout-b", trajectory_id="b")
        assert (await models_status(our_url, other.api_key))[0] == 401  # no key before this rollout opens its own
        root = ours.open_agent(rollout_id="rollout-a", trajectory_id="a")
        socket_path = str(socket_directory(short_base) / SOCKET_NAME)
        await ours.serve_unix(socket_path, rollout_id="rollout-a")

        assert (await models_status(our_url, root.api_key))[0] == 200
        status, body = await models_status(our_url, other.api_key)
        assert status == 401 and "another rollout" in body
        on_socket = [
            await models_status("http://localhost", key, socket_path=socket_path)
            for key in (root.api_key, other.api_key)
        ]
        assert [status for status, _ in on_socket] == [200, 401]
        # Each rollout's key works on its own rollout's gateway only.
        assert [(await models_status(their_url, key))[0] for key in (other.api_key, root.api_key)] == [200, 401]
    finally:
        await ours.stop()
        await theirs.stop()
        shutil.rmtree(short_base, ignore_errors=True)
