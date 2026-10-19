"""The fork server (:mod:`platoon.utils.forkserver`), and rollout workers started from it (``worker_start_method:
forkserver``).

A forked worker must be the process an exec'd one would be: the supervisor's tests (``test_worker_supervisor.py``)
run every case with workers started both ways.  These cover what only a fork server can get wrong: what a child
inherits and how it exits, a server that must not fork, a server that is closed or dies, a caller that gives up while
its worker is being forked, the supervisor's fallback to exec, and the real worker entrypoint started both ways.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pytest

import platoon.train.tinker  # noqa: F401  (import order: package before data processing)
from platoon.config_defs import RolloutConfig
from platoon.task import Task
from platoon.train.tinker.workers import supervisor as supervisor_module
from platoon.train.tinker.workers.protocol import (
    PROTOCOL_VERSION,
    ConversionPolicy,
    ModelBinding,
    RolloutWorkerRequest,
)
from platoon.train.tinker.workers.supervisor import RolloutWorkerSupervisor
from platoon.utils.forkserver import ForkServer, ForkServerError

STUB_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE = "forkserver_stub_worker"


def _env(**extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([STUB_DIR, os.environ.get("PYTHONPATH", "")])
    env.update(extra)
    return env


async def _server(tmp_path: Path, *, module: str = MODULE, preload=(), env=None, prepare=()) -> ForkServer:
    return await ForkServer.start(
        python=sys.executable,
        module=module,
        preload=preload,
        prepare=prepare,
        env=env if env is not None else _env(),
        cwd=str(tmp_path),
        log_path=str(tmp_path / "server.log"),
        timeout_s=120.0,
    )


async def _until(condition, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {condition}")
        await asyncio.sleep(0.02)


def _running(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            raw = handle.read()
    except OSError:
        return False
    return raw[raw.rindex(b")") + 2 : raw.rindex(b")") + 3] not in (b"Z", b"X", b"x")


def _children(pid: int) -> list[int]:
    """The live processes whose parent is ``pid``."""

    found = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as handle:
                raw = handle.read()
        except OSError:
            continue
        fields = raw[raw.rindex(b")") + 2 :].split()
        if int(fields[1]) == pid and fields[0] not in (b"Z", b"X", b"x"):
            found.append(int(name))
    return found


# ------------------------------------------------------------------------------------------- what a child is like


def test_a_forked_worker_is_the_process_an_exec_d_one_would_be(tmp_path):
    env = _env(FORKSERVER_TEST="worker", MULTI="a b\nc")
    workdir = tmp_path / "work"
    workdir.mkdir()
    with open(tmp_path / "exec.log", "ab", buffering=0) as log:  # as the supervisor execs a worker
        completed = subprocess.run(
            [sys.executable, "-m", MODULE, "describe", str(tmp_path / "exec.json")],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env=env,
            cwd=str(workdir),
        )
    assert completed.returncode == 0

    async def scenario() -> int:
        server = await _server(tmp_path, env=_env(FORKSERVER_TEST="server"))  # the server's own differs
        try:
            worker = await server.spawn(
                ["describe", str(tmp_path / "fork.json")],
                env=env,
                cwd=str(workdir),
                log_path=str(tmp_path / "fork.log"),
            )
            assert await worker.wait() == 0
            assert worker.returncode == 0
            return worker.pid
        finally:
            server.close()

    pid = asyncio.run(scenario())
    exec_facts = json.loads((tmp_path / "exec.json").read_text())
    fork_facts = json.loads((tmp_path / "fork.json").read_text())
    assert fork_facts["pid"] == pid
    for facts, log in ((exec_facts, "exec.log"), (fork_facts, "fork.log")):
        assert facts["pgid"] == facts["sid"] == facts["pid"]  # it leads a session of its own
        assert facts["stdin"] == "/dev/null"
        assert facts["stdout"] == facts["stderr"] == str(tmp_path / log)
    for key in ("descriptors", "cwd", "argv0", "name", "env", "signals", "blocked", "threads", "umask", "random_seed"):
        assert fork_facts[key] == exec_facts[key], key
    assert fork_facts["env"]["FORKSERVER_TEST"] == "worker" and fork_facts["env"]["MULTI"] == "a b\nc"
    assert fork_facts["argv"][0] == "describe" and fork_facts["name"] == "__main__"


def test_forked_workers_draw_fresh_random_numbers(tmp_path):
    pytest.importorskip("numpy")

    async def scenario() -> list[dict]:
        # numpy's generator is seeded in the server (with BLAS pinned to one thread, as in a rollout worker)
        server = await _server(tmp_path, preload=("numpy",), env=_env(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"))
        try:
            paths = [tmp_path / f"facts-{index}.json" for index in range(2)]
            for path in paths:
                worker = await server.spawn(["describe", str(path)], env=_env(), cwd=str(tmp_path), log_path=None)
                assert await worker.wait() == 0
            return [json.loads(path.read_text()) for path in paths]
        finally:
            server.close()

    first, second = asyncio.run(scenario())
    assert first["random"] != second["random"]
    assert first["numpy"] != second["numpy"]


def test_generators_the_imports_seeded_from_the_os_draw_fresh_numbers_and_seeded_ones_keep_their_seed(tmp_path):
    """Libraries make generators of their own at import (sympy's, datasets' fingerprint generator): one seeded from the
    OS differs in every exec'd worker, so it must in every forked one; one seeded with a value, or given a state, is
    the same in every worker either way."""

    async def scenario() -> list[dict]:
        server = await _server(tmp_path)
        try:
            paths = [tmp_path / f"facts-{index}.json" for index in range(2)]
            for path in paths:
                worker = await server.spawn(["describe", str(path)], env=_env(), cwd=str(tmp_path), log_path=None)
                assert await worker.wait() == 0
            return [json.loads(path.read_text()) for path in paths]
        finally:
            server.close()

    first, second = asyncio.run(scenario())
    assert first["generators"]["os_seeded"] != second["generators"]["os_seeded"]
    for facts in (first, second):
        assert facts["generators"]["seeded"] == random.Random(1234).random()
        assert facts["generators"]["restored"] == random.Random(99).random()
        assert facts["random_seed"] == "Random.seed"  # the server put random.Random back as it found it


def test_workers_take_what_the_server_built_ahead_and_a_failed_build_is_only_reported(tmp_path):
    prepare = [
        {"ref": f"{MODULE}:prepare", "kwargs": {"value": "built once"}},
        {"ref": f"{MODULE}:fail_to_prepare"},
    ]

    async def scenario() -> tuple[list[str], dict]:
        server = await _server(tmp_path, prepare=prepare)
        try:
            worker = await server.spawn(
                ["describe", str(tmp_path / "facts.json")], env=_env(), cwd=str(tmp_path), log_path=None
            )
            assert await worker.wait() == 0
            return server.prepare_errors, json.loads((tmp_path / "facts.json").read_text())
        finally:
            server.close()

    errors, facts = asyncio.run(scenario())
    assert facts["prepared"] == "built once"
    assert errors == [f"{MODULE}:fail_to_prepare: RuntimeError: the stub cannot prepare"]


@pytest.mark.parametrize(("argv", "code"), [(["exit", "0"], 0), (["exit", "3"], 3), (["raise"], 1)])
def test_a_forked_worker_exits_with_the_code_python_would(tmp_path, argv, code):
    async def scenario() -> int | None:
        server = await _server(tmp_path)
        try:
            worker = await server.spawn(argv, env=_env(), cwd=str(tmp_path), log_path=str(tmp_path / "worker.log"))
            return await worker.wait()
        finally:
            server.close()

    assert asyncio.run(scenario()) == code
    if argv == ["raise"]:
        assert "RuntimeError: the stub worker raised" in (tmp_path / "worker.log").read_text()


@pytest.mark.parametrize("mode", ["thread", "atexit"])
def test_a_forked_worker_leaves_as_the_interpreter_does(tmp_path, mode):
    """It waits for its non-daemon threads and runs its atexit handlers before it exits."""

    marker = tmp_path / "marker"

    async def scenario() -> int | None:
        server = await _server(tmp_path)
        try:
            worker = await server.spawn([mode, str(marker)], env=_env(), cwd=str(tmp_path), log_path=None)
            return await worker.wait()
        finally:
            server.close()

    assert asyncio.run(scenario()) == 0
    assert marker.read_text() == mode


def test_a_forked_worker_killed_by_a_signal_reports_it(tmp_path):
    async def scenario() -> int | None:
        server = await _server(tmp_path)
        try:
            pid_path = tmp_path / "pid"
            worker = await server.spawn(["sleep", str(pid_path)], env=_env(), cwd=str(tmp_path), log_path=None)
            await _until(pid_path.exists)
            os.killpg(worker.pid, signal.SIGKILL)  # it leads its own group
            return await worker.wait()
        finally:
            server.close()

    assert asyncio.run(scenario()) == -signal.SIGKILL


def test_many_workers_asked_for_at_once_all_start(tmp_path):
    """Requests wait in the control socket: none is lost however many are asked for at once, also when they hold
    more than the socket buffers (a large environment)."""

    env = _env(LARGE="x" * 20000)

    async def scenario() -> list[int | None]:
        server = await _server(tmp_path)
        try:
            workers = await asyncio.gather(
                *(server.spawn(["exit", str(i % 7)], env=env, cwd=str(tmp_path), log_path=None) for i in range(100))
            )
            codes = await asyncio.wait_for(asyncio.gather(*(worker.wait() for worker in workers)), 120)
            for worker in workers:
                worker.close()
            return list(codes)
        finally:
            server.close()

    assert asyncio.run(scenario()) == [i % 7 for i in range(100)]


# ------------------------------------------------------------------------------------- servers that cannot serve


def test_a_server_that_runs_a_thread_after_its_imports_does_not_serve(tmp_path):
    async def scenario() -> None:
        with pytest.raises(ForkServerError) as raised:
            await _server(tmp_path, preload=("forkserver_threaded_module",))
        assert raised.value.permanent
        # It says which thread runs and what runs in it.
        assert "the Python thread 'stray-import-thread' running time.sleep" in str(raised.value)

    asyncio.run(scenario())


# ----------------------------------------------------------------------------------- a node with an NVIDIA driver


@pytest.mark.parametrize(
    ("at_import", "gpus", "why"),
    [
        # transformers asks torch.cuda.is_available() at import, which starts the CUDA driver and its thread wherever a
        # driver is installed (forkserver_fake_cuda stands in for it), unless the environment hides the GPUs and has
        # torch ask NVML: a server given an environment that does neither cannot fork.
        (
            "is_available",
            {"CUDA_VISIBLE_DEVICES": "0"},
            "as torch.cuda.is_available() does wherever a driver is installed, unless the environment hides the GPUs "
            "and has torch ask NVML: CUDA_VISIBLE_DEVICES= PYTORCH_NVML_BASED_CUDA_CHECK=1)",
        ),
        # Any other CUDA call starts it whatever the environment says.
        (
            "initialize",
            {"CUDA_VISIBLE_DEVICES": "", "PYTORCH_NVML_BASED_CUDA_CHECK": "1"},
            "with a CUDA call other than torch.cuda.is_available(), which this environment answers without it)",
        ),
    ],
)
def test_a_server_whose_imports_start_the_cuda_driver_says_so(tmp_path, at_import, gpus, why):
    env = _env(FAKE_CUDA_AT_IMPORT=at_import, **gpus)
    if "PYTORCH_NVML_BASED_CUDA_CHECK" not in gpus:
        env.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)

    async def scenario() -> None:
        with pytest.raises(ForkServerError) as raised:
            await _server(tmp_path, preload=("forkserver_fake_cuda",), env=env)
        assert raised.value.permanent
        assert f"'cuda00001400006' (the CUDA driver's: the imports or a prepare step started it, {why}" in str(
            raised.value
        )

    asyncio.run(scenario())


def test_a_server_whose_modules_do_not_import_does_not_serve(tmp_path):
    async def scenario() -> None:
        with pytest.raises(ForkServerError) as raised:
            await _server(tmp_path, module="definitely_not_a_module_xyz")
        assert raised.value.permanent and "importing" in str(raised.value)

    asyncio.run(scenario())


def test_closing_the_server_lets_its_workers_finish(tmp_path):
    marker = tmp_path / "marker"

    async def scenario() -> None:
        server = await _server(tmp_path)
        worker = await server.spawn(["thread", str(marker)], env=_env(), cwd=str(tmp_path), log_path=None)
        server.close()  # no more workers; this one runs to its end
        with pytest.raises(ForkServerError):
            await server.spawn(["exit", "0"], env=_env(), cwd=str(tmp_path), log_path=None)
        await asyncio.wait_for(worker.wait(), 20)
        worker.close()
        await asyncio.to_thread(server.process.wait, 20)  # and the server leaves after its last worker

    asyncio.run(scenario())
    assert marker.read_text() == "thread"


def test_a_dead_server_is_noticed_and_its_workers_still_end(tmp_path):
    async def scenario() -> int | None:
        server = await _server(tmp_path)
        pid_path = tmp_path / "pid"
        worker = await server.spawn(["sleep", str(pid_path)], env=_env(), cwd=str(tmp_path), log_path=None)
        await _until(pid_path.exists)
        server.process.kill()
        await _until(lambda: server.dead)
        with pytest.raises(ForkServerError):
            await server.spawn(["exit", "0"], env=_env(), cwd=str(tmp_path), log_path=None)
        os.killpg(worker.pid, signal.SIGKILL)
        code = await asyncio.wait_for(worker.wait(), 20)  # read from its pidfd: no status without its server
        worker.close()
        return code

    assert asyncio.run(scenario()) is None


def test_a_worker_forked_for_a_caller_that_gave_up_is_killed(tmp_path):
    async def scenario() -> list[int]:
        server = await _server(tmp_path)
        try:
            spawning = asyncio.ensure_future(
                server.spawn(["sleep", str(tmp_path / "pid")], env=_env(), cwd=str(tmp_path), log_path=None)
            )
            await asyncio.sleep(0)  # the request is sent; the answer has not come
            spawning.cancel()
            with pytest.raises(asyncio.CancelledError):
                await spawning
            await _until(lambda: not server._pending, timeout=20)  # the answer came, and found no caller
            await _until(lambda: not _children(server.pid), timeout=20)
            return _children(server.pid)
        finally:
            server.close()

    assert asyncio.run(scenario()) == []


# ------------------------------------------------------------------------------------------------ the supervisor


def _stub_supervisor(tmp_path: Path, mode: str, **kwargs) -> tuple[RolloutWorkerSupervisor, Path]:
    scratch = tmp_path / "ipc"
    scratch.mkdir(parents=True, exist_ok=True)
    markers = tmp_path / "markers"
    markers.mkdir(parents=True, exist_ok=True)
    env = {
        "STUB_MODE": mode,
        "STUB_MARKER_DIR": str(markers),
        "PYTHONPATH": os.pathsep.join([STUB_DIR, os.environ.get("PYTHONPATH", "")]),
    }
    supervisor = RolloutWorkerSupervisor(
        module="worker_stub_entrypoint",
        scratch_dir=str(scratch),
        log_dir=str(tmp_path / "logs"),
        env_overrides=env,
        termination_grace_seconds=1.0,
        parent_grace_seconds=2.0,
        ready_poll_seconds=0.05,
        start_method="forkserver",
        **kwargs,
    )
    return supervisor, markers


def _stub_request(scratch: Path, request_id: str) -> RolloutWorkerRequest:
    return RolloutWorkerRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id=request_id,
        task_id="task",
        rollout_number=0,
        task=Task(goal="g", id="task"),
        rollout_config=asdict(RolloutConfig(output_dir=str(scratch))),
        rollout_fn_ref="worker_stub_entrypoint:main",
        model=ModelBinding(
            binding_path=str(scratch / "model_binding.pkl"),
            bootstrap_blob=b"",
            follow_poll_seconds=5.0,
            model_name="",
            renderer_name="",
        ),
        conversion=ConversionPolicy(),
        response_path=str(scratch / f"{request_id}.res"),
        ready_path=str(scratch / f"{request_id}.ready"),
        startup_budget_seconds=60.0,
        rollout_budget_seconds=20.0,
    )


def _parents(markers: Path) -> dict[int, int]:
    return {
        int(path.name.split("-")[1]): int(path.read_text())
        for path in markers.iterdir()
        if path.name.startswith("parent-")
    }


def test_a_changed_environment_gets_a_fork_server_of_its_own(tmp_path, monkeypatch, caplog):
    """A server forks workers with the environment it started with, which its imports may have read: once the
    trainer's changes, the next worker comes from a new server with the new environment."""

    supervisor, markers = _stub_supervisor(tmp_path, "parent")
    scratch = tmp_path / "ipc"

    async def scenario() -> tuple[list[int], list]:
        first = await supervisor.run(_stub_request(scratch, "first-0"))
        old = supervisor._fork_server
        assert old is not None
        monkeypatch.setenv("FORKSERVER_TEST_CHANGED", "1")  # the trainer's environment changed
        second = await supervisor.run(_stub_request(scratch, "second-0"))
        new = supervisor._fork_server
        assert new is not None and new is not old and not old.alive
        assert new.env["FORKSERVER_TEST_CHANGED"] == "1"
        return [old.pid, new.pid], [first, second]

    try:
        with caplog.at_level(logging.WARNING):
            servers, outcomes = asyncio.run(scenario())
    finally:
        supervisor.close()
    parents = _parents(markers)
    assert [parents[outcome.worker_pid] for outcome in outcomes] == servers
    assert "started with another environment (FORKSERVER_TEST_CHANGED)" in caplog.text


def test_a_fork_server_that_cannot_serve_leaves_workers_to_exec(tmp_path, monkeypatch, caplog):
    supervisor, markers = _stub_supervisor(tmp_path, "parent")
    monkeypatch.setattr(
        supervisor_module.RolloutWorkerSupervisor,
        "_fork_server_preload",
        lambda self, request: ["forkserver_threaded_module"],
    )
    scratch = tmp_path / "ipc"

    async def scenario() -> list:
        return [await supervisor.run(_stub_request(scratch, f"r{index}-0")) for index in range(2)]

    try:
        with caplog.at_level(logging.WARNING):
            outcomes = asyncio.run(scenario())
    finally:
        supervisor.close()
    parents = _parents(markers)
    assert [parents[outcome.worker_pid] for outcome in outcomes] == [os.getpid(), os.getpid()]
    assert supervisor._fork_server_starts == supervisor_module.MAX_FORK_SERVER_STARTS  # not tried again
    assert "stray-import-thread" in caplog.text


def test_workers_and_their_fork_server_see_no_gpu_and_ask_nvml(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")  # whatever the trainer's environment says
    monkeypatch.setenv("PYTORCH_NVML_BASED_CUDA_CHECK", "0")
    supervisor = RolloutWorkerSupervisor(scratch_dir=str(tmp_path), start_method="forkserver")
    for env in (supervisor._child_env(), supervisor._child_env(_stub_request(tmp_path, "r-0"))):  # server, worker
        assert env["CUDA_VISIBLE_DEVICES"] == "" and env["PYTORCH_NVML_BASED_CUDA_CHECK"] == "1"


def test_a_fork_server_whose_imports_ask_whether_cuda_is_available_forks_its_workers(tmp_path, monkeypatch):
    """A run on a node with an NVIDIA driver: transformers asks torch whether CUDA is available while the fork server
    imports the worker's stack.  In the environment the supervisor gives it, no driver thread starts, and the server
    forks every worker."""

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")  # the trainer's GPUs
    monkeypatch.delenv("PYTORCH_NVML_BASED_CUDA_CHECK", raising=False)
    monkeypatch.setenv("FAKE_CUDA_AT_IMPORT", "is_available")
    supervisor, markers = _stub_supervisor(tmp_path, "parent")
    monkeypatch.setattr(
        supervisor_module.RolloutWorkerSupervisor,
        "_fork_server_preload",
        lambda self, request: ["forkserver_fake_cuda"],
    )
    scratch = tmp_path / "ipc"

    async def scenario() -> tuple[int, list]:
        outcomes = [await supervisor.run(_stub_request(scratch, f"r{index}-0")) for index in range(2)]
        assert supervisor._fork_server is not None
        return supervisor._fork_server.pid, outcomes

    try:
        server_pid, outcomes = asyncio.run(scenario())
    finally:
        supervisor.close()
    parents = _parents(markers)
    assert [parents[outcome.worker_pid] for outcome in outcomes] == [server_pid, server_pid]


def test_the_worker_s_stack_never_calls_the_cuda_driver_in_its_fork_server_s_environment(tmp_path, monkeypatch):
    """Wherever an NVIDIA driver is installed, a call into it starts its thread, which keeps a fork server from forking,
    and transformers makes one at import (``torch.cuda.is_available()``).  Checked where no driver is installed too:
    spies on torch's calls into the driver (``cuda_driver_spy``) see that call in a trainer's environment, and none in
    the environment the supervisor gives its fork server and workers, through the whole stack the server imports and
    torch's answers after it."""

    import torch

    if not torch.cuda._is_compiled():
        pytest.skip("a torch built without CUDA never calls the driver")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)  # a trainer's: every GPU, as far as torch knows
    monkeypatch.delenv("PYTORCH_NVML_BASED_CUDA_CHECK", raising=False)

    def spied(env: dict[str, str], modules: list[str]) -> dict:
        completed = subprocess.run(
            [sys.executable, os.path.join(STUB_DIR, "cuda_driver_spy.py"), *modules],
            env=env,
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert completed.returncode == 0, completed.stderr[-4000:]
        return json.loads(completed.stdout.strip().splitlines()[-1])

    assert spied(dict(os.environ), [])["calls"][:1] == ["_cuda_getDeviceCount"]  # the spies see the driver's call
    supervisor = RolloutWorkerSupervisor(scratch_dir=str(tmp_path), start_method="forkserver")
    server_env = supervisor._child_env()
    assert spied(server_env, list(supervisor_module.WORKER_PRELOAD)) == {"calls": [], "answers": [False, 0]}


def test_the_supervisor_forks_many_workers_at_once(tmp_path):
    supervisor, markers = _stub_supervisor(tmp_path, "parent")
    scratch = tmp_path / "ipc"

    async def scenario() -> tuple[int, list]:
        outcomes = await asyncio.gather(*(supervisor.run(_stub_request(scratch, f"r{index}-0")) for index in range(24)))
        assert supervisor._fork_server is not None
        return supervisor._fork_server.pid, outcomes

    try:
        server_pid, outcomes = asyncio.run(scenario())
    finally:
        supervisor.close()
    parents = _parents(markers)
    assert [parents[outcome.worker_pid] for outcome in outcomes] == [server_pid] * 24


def test_the_fork_server_builds_the_run_s_tokenizer_and_renderer_ahead(tmp_path):
    from platoon.train.tinker.workers.protocol import ModelBinding as Binding

    supervisor = RolloutWorkerSupervisor(scratch_dir=str(tmp_path), start_method="forkserver")
    bound = dataclasses.replace(
        _stub_request(tmp_path, "r-0"),
        model=Binding(
            binding_path=str(tmp_path / "binding.pkl"),
            bootstrap_blob=b"",
            follow_poll_seconds=5.0,
            model_name="org/model",
            renderer_name="chat_template",
            renderer_kwargs={"preserve_thinking": False},
        ),
    )
    assert supervisor._fork_server_prepare(bound) == [
        {
            "ref": "platoon.train.tinker.proxy:prepare_renderer",
            "kwargs": {
                "model_name": "org/model",
                "renderer_name": "chat_template",
                "renderer_kwargs": {"preserve_thinking": False},
            },
        }
    ]
    assert supervisor._fork_server_prepare(_stub_request(tmp_path, "r-1")) == []  # no model bound
    stub, _ = _stub_supervisor(tmp_path / "stub", "ok")
    assert stub._fork_server_prepare(bound) == []  # not the rollout worker


def test_a_dead_fork_server_is_replaced(tmp_path):
    supervisor, markers = _stub_supervisor(tmp_path, "parent")
    scratch = tmp_path / "ipc"

    async def scenario() -> tuple[list[int], list]:
        first = await supervisor.run(_stub_request(scratch, "first-0"))
        dead = supervisor._fork_server
        assert dead is not None
        dead.process.kill()
        await _until(lambda: dead.dead)
        second = await supervisor.run(_stub_request(scratch, "second-0"))
        assert supervisor._fork_server is not None and supervisor._fork_server is not dead
        return [dead.pid, supervisor._fork_server.pid], [first, second]

    try:
        servers, outcomes = asyncio.run(scenario())
    finally:
        supervisor.close()
    parents = _parents(markers)
    assert [parents[outcome.worker_pid] for outcome in outcomes] == servers
    assert supervisor._fork_server_starts == 2


# ------------------------------------------------------------------------------- the real worker, started both ways


@pytest.fixture(scope="module")
def real_workers(tmp_path_factory):
    """The real entrypoint run by the supervisor, exec'd and forked, with a rollout that reports its process, in the
    worker's own Python overlay (``python_env_isolation``)."""

    results = {}
    for method in ("exec", "forkserver"):
        tmp_path = tmp_path_factory.mktemp(f"real-{method}")
        scratch = tmp_path / "ipc"
        scratch.mkdir()
        request_id = f"real-{method}-0"
        facts_path = tmp_path / "facts.json"
        supervisor = RolloutWorkerSupervisor(
            scratch_dir=str(scratch),
            log_dir=str(tmp_path / "logs"),
            env_overrides={
                "PYTHONPATH": os.pathsep.join([STUB_DIR, os.environ.get("PYTHONPATH", "")]),
                "STUB_FACTS_PATH": str(facts_path),
                "STUB_REQUEST_PATH": str(scratch / f"{request_id}.req"),
                "HF_HUB_OFFLINE": "1",
            },
            start_method=method,
        )
        request = RolloutWorkerRequest(
            protocol_version=PROTOCOL_VERSION,
            request_id=request_id,
            task_id="task",
            rollout_number=0,
            task=Task(goal="g", id="task"),
            rollout_config=asdict(RolloutConfig(output_dir=str(tmp_path))),
            rollout_fn_ref="worker_stub_rollouts:report_and_finish",
            model=ModelBinding(
                binding_path=str(scratch / "model_binding.pkl"),
                bootstrap_blob=b"",
                follow_poll_seconds=5.0,
                model_name="",
                renderer_name="",
            ),
            conversion=ConversionPolicy(),
            response_path=str(scratch / f"{request_id}.res"),
            ready_path=str(scratch / f"{request_id}.ready"),
            startup_budget_seconds=600.0,
            rollout_budget_seconds=600.0,
            python_env_isolation=True,
        )
        try:
            outcome = asyncio.run(supervisor.run(request))
            server_pid = supervisor._fork_server.pid if supervisor._fork_server is not None else None
        finally:
            supervisor.close()
        assert outcome.log_path is not None
        results[method] = {
            "outcome": outcome,
            "facts": json.loads(facts_path.read_text()),
            "log": Path(outcome.log_path).read_text(),
            "server_pid": server_pid,
        }
    return results


@pytest.mark.parametrize("method", ["exec", "forkserver"])
def test_the_real_worker_runs_its_rollout_either_way(real_workers, method):
    result = real_workers[method]
    outcome = result["outcome"]
    assert outcome.disposition == "no_trajectories", result["log"][-4000:]
    assert outcome.response is not None and outcome.response.request_id == f"real-{method}-0"
    assert outcome.returncode == 0
    facts = result["facts"]
    assert facts["is_session_leader"] is True and facts["pgid"] == facts["pid"] == outcome.worker_pid
    assert facts["TINKER_TELEMETRY"] == "0" and facts["OMP_NUM_THREADS"] == "1"
    assert facts["torch_threads"] == 1 and facts["torch_interop_threads"] == 1
    assert facts["PLATOON_TINKER_ROLLOUT_WORKER"] == "1" and facts["subprocess_sampling_present"] is False
    # It sees no GPU and never started the CUDA driver, nor did its fork server: where a driver is installed, the
    # transformers import would otherwise start the driver's thread in both.
    assert facts["CUDA_VISIBLE_DEVICES"] == "" and facts["PYTORCH_NVML_BASED_CUDA_CHECK"] == "1"
    assert facts["cuda_driver_threads"] == []
    assert facts["extra_allow"] is True
    assert facts["request_file_exists"] is False
    assert facts["python_env"] == "overlay"  # the rollout ran in an overlay of its worker's own, exec'd or forked
    assert "[rollout-worker] request=" in result["log"] and "torch_threads=1" in result["log"]
    if method == "forkserver":
        assert result["server_pid"] is not None


def test_the_real_worker_is_the_same_process_forked_as_exec_d(real_workers):
    exec_facts, fork_facts = real_workers["exec"]["facts"], real_workers["forkserver"]["facts"]
    for key in exec_facts:
        if key not in ("pid", "pgid"):
            assert fork_facts[key] == exec_facts[key], key
    # Its startup no longer includes the imports.
    fork_startup = _startup(real_workers["forkserver"]["log"])
    exec_startup = _startup(real_workers["exec"]["log"])
    assert fork_startup < exec_startup


def _startup(log: str) -> float:
    for line in log.splitlines():
        if line.startswith("[rollout-worker] request="):
            return float(line.rsplit("startup=", 1)[1].rstrip("s"))
    raise AssertionError(f"no header line in {log[-2000:]}")
