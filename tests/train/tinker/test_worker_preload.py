"""What a rollout workers' fork server imports and builds ahead of its workers, beyond the worker's stack
(:mod:`platoon.train.startup`, :mod:`platoon.utils.forkserver`): only what leaves it fork-safe.

Fork safety is the point of these tests: a worker forked next to a running thread can deadlock, and one that inherits
an open file, socket or event loop shares it with every sibling, and one forked after an import seeded a numpy
generator from the OS draws what its siblings draw.  So an optional module or a build that starts a thread or a
process, leaves a descriptor open or draws a seed from the OS is left out (the server starts over without it, once,
having tried every item; the processes it started are ended, and no descriptor passes to the new server) and reported
with what it left; the rest is imported or built once, and an environment variable it sets is set in each worker too,
as the import would have set it there.  A forked worker then holds one thread, no descriptor but stdio (and libffi's,
which it maps and never reads), and no child; the real preload a run derives for an ORS environment played by the
CodeAct harness leaves it so too, and fills the environment class's task cache the worker reads.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

import platoon.train.tinker  # noqa: F401  (import order: package before data processing)
from platoon.config_defs import RolloutConfig
from platoon.task import Task
from platoon.train.components import normalize_environment_configs
from platoon.train.config_defs import WorkflowConfig
from platoon.train.startup import WorkerPreload, worker_preload
from platoon.train.workers import supervisor as supervisor_module
from platoon.train.workers.env import NO_CUDA_DRIVER
from platoon.train.workers.protocol import PROTOCOL_VERSION, ConversionPolicy, ModelBinding, RolloutWorkerRequest
from platoon.train.workers.supervisor import WORKER_PRELOAD, RolloutWorkerSupervisor
from platoon.train.workflow.group import GroupRolloutWorkflow
from platoon.utils.forkserver import ForkServer

STUB_DIR = os.path.dirname(os.path.abspath(__file__))
ENVS_DIR = os.path.join(os.path.dirname(os.path.dirname(STUB_DIR)), "envs")  # ors_toys
MODULE = "forkserver_stub_worker"
UNSAFE = {
    "preload_stubs.thread": "the Python thread 'preload-stub-thread' running time.sleep",
    "preload_stubs.descriptor": "it left file descriptors open:",
    "preload_stubs.inheritable": "preload_stubs/inheritable.py)",
    "preload_stubs.data_file": "tasks.solutions.jsonl)",
    "preload_stubs.listener": "socket:[",
    "preload_stubs.loop": "anon_inode:[eventpoll]",
    "preload_stubs.process": "it started processes:",
    "preload_stubs.seeded": "it drew 1 seed(s) from the OS (a numpy generator made without a seed, or "
    "secrets.randbits), which every worker would share",
}


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    """The environment of a server and its workers: the stubs and the toy environments importable, and no GPU and no
    CUDA driver, as the supervisor gives its servers (torch, in the worker's stack, would otherwise start the driver
    wherever one is installed, and the server would not serve)."""

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([STUB_DIR, ENVS_DIR, os.environ.get("PYTHONPATH", "")])
    env["PRELOAD_STUB_PIDS"] = str(tmp_path / "stub-pids")
    env.update(supervisor_module.WORKER_ENV_OVERRIDES)  # one BLAS and OpenMP thread, as in a rollout worker
    env.update(NO_CUDA_DRIVER)
    env.update(extra)
    return env


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


async def _state_of_a_worker(
    tmp_path: Path, *, preload=(), optional=(), prepare=(), class_ref: str | None = None
) -> tuple[ForkServer, dict]:
    """Start a server, fork one stub worker that reports what the server left in it (``state``), and close it."""

    server = await ForkServer.start(
        python=sys.executable,
        module=MODULE,
        preload=preload,
        optional=optional,
        prepare=prepare,
        env=_env(tmp_path),
        cwd=str(tmp_path),
        log_path=str(tmp_path / "server.log"),
        timeout_s=600.0,
    )
    try:
        argv = ["state", str(tmp_path / "state.json"), *([class_ref] if class_ref else [])]
        worker = await server.spawn(argv, env=_env(tmp_path), cwd=str(tmp_path), log_path=str(tmp_path / "worker.log"))
        assert await worker.wait() == 0, (tmp_path / "worker.log").read_text()[-4000:]
        worker.close()
        return server, json.loads((tmp_path / "state.json").read_text())
    finally:
        server.close()


def _end_stub_processes(tmp_path: Path) -> None:
    pids = tmp_path / "stub-pids"
    for line in pids.read_text().split() if pids.exists() else ():
        try:
            os.kill(int(line), signal.SIGKILL)
        except ProcessLookupError:
            pass


# --------------------------------------------------------------------------------- the server's optional preload


def test_what_is_not_fork_safe_is_left_out_with_the_reason_and_the_rest_is_imported_once(tmp_path):
    """Each optional module and build is checked when it has run, against the state the one before it left: one that
    started a thread (that runs on) or a process, left a file (an inheritable one, or a data file whose name holds
    ".so"), a socket or an event loop open, or seeded a numpy generator from the OS, makes the server start over
    without it, once it has tried them all, and only once; one that does not import is left out; the rest is imported,
    or built, and every forked worker has it, with the environment variables it set, in a process as safe to fork as
    the worker's stack alone left it: no descriptor of what was left out passed to it through the server's new start,
    and the processes those started are gone."""

    pytest.importorskip("numpy")
    optional = [
        "preload_stubs.safe",
        *UNSAFE,
        "preload_stubs.broken",
        "preload_stubs.short_thread",
        "preload_stubs.environment",
        "json",  # imported already, with the stack
    ]
    prepare = [
        {"ref": "preload_stubs.builds:remember", "kwargs": {"value": "built once"}},
        {"ref": "preload_stubs.builds:start_a_thread"},
    ]
    try:
        server, state = asyncio.run(_state_of_a_worker(tmp_path, optional=optional, prepare=prepare))
    finally:
        _end_stub_processes(tmp_path)

    skipped = {entry["item"]: entry["reason"] for entry in server.skipped}
    assert sorted(skipped) == sorted([*UNSAFE, "preload_stubs.broken", "preload_stubs.builds:start_a_thread()"]), (
        server.skipped
    )
    for module, reason in UNSAFE.items():
        assert reason in skipped[module], (module, skipped[module])
    assert skipped["preload_stubs.descriptor"].endswith("preload_stubs/descriptor.py)")
    assert skipped["preload_stubs.broken"] == "it does not import: ImportError: the preload stub does not import"
    assert "the Python thread 'preload-stub-build'" in skipped["preload_stubs.builds:start_a_thread()"]
    assert skipped["preload_stubs.thread"].startswith("it started threads: ")
    assert server.preloaded == 3  # safe, short_thread and environment; json was imported already
    assert server.prepare_errors == []
    # The worker: what was safe is there, built or imported once, and nothing of what was not.
    assert {"preload_stubs.safe", "preload_stubs.short_thread", "preload_stubs.builds"} <= set(state["modules"])
    assert "preload_stubs.environment" in state["modules"]
    assert not set(UNSAFE) & set(state["modules"]) and "preload_stubs.broken" not in state["modules"]
    assert state["built"] == {"value": "built once"}
    assert (state["threads"], state["tasks"], state["descriptors"], state["children"]) == (1, 1, {}, [])
    assert state["environ"]["PRELOAD_STUB_FLAG"] == "set at import"  # as the import would have set it
    started = [int(pid) for pid in (tmp_path / "stub-pids").read_text().split()]
    assert len(started) == 1 and not Path(f"/proc/{started[0]}").exists()  # ended when the server started over
    # The server's log says what it left out, and that it started over once.
    log = (tmp_path / "server.log").read_text()
    assert log.count("is not fork-safe (") == len(UNSAFE) + 1
    assert log.count("fork server: starting over without") == 1


def test_the_preload_a_run_derives_is_fork_safe_and_fills_the_task_cache_the_workers_read(tmp_path):
    """The real thing: the worker's stack, then what a run derives for an in-process ORS environment played by the
    CodeAct harness (the SDK, the environment's module, the harness, the program, harness_rollout's modules, the
    registry lookups and the environment's task lists for the run's splits).  None of it is left out, and the forked
    worker holds one thread, no descriptor but stdio and libffi's, and no child, with the task cache filled."""

    environments = normalize_environment_configs(
        [{"service": "python://ors_toys:CounterEnv", "harness": "platoon-codeact", "agents": {"max_depth": 0}}]
    )
    preload = worker_preload(environments, phase="train", tasks=[{"task_id": "train:0"}, {"task_id": "train:2"}])
    assert "ors_toys" in preload.modules and "platoon.harness.builtin.codeact.harness" in preload.modules

    server, state = asyncio.run(
        _state_of_a_worker(
            tmp_path,
            preload=WORKER_PRELOAD,
            optional=preload.modules,
            prepare=preload.prepare,
            class_ref="ors_toys:CounterEnv",
        )
    )

    assert server.skipped == [] and server.prepare_errors == []
    assert set(preload.modules) <= set(state["modules"])
    assert {"anthropic", "openai", "IPython", "openreward.environments"} <= set(state["modules"])
    assert state["task_cache"] == ["train"]
    assert (state["threads"], state["tasks"], state["descriptors"], state["children"]) == (1, 1, {}, [])
    assert state["environ"].get("OPENREWARD_DISABLE_UPDATE_CHECK") == "1"  # as opening the environment sets it


# ------------------------------------------------------------------------------------------------- the supervisor


def _stub_supervisor(tmp_path: Path, **kwargs) -> RolloutWorkerSupervisor:
    scratch = tmp_path / "ipc"
    scratch.mkdir(parents=True, exist_ok=True)
    markers = tmp_path / "markers"
    markers.mkdir(parents=True, exist_ok=True)
    env = {
        "STUB_MODE": "parent",
        "STUB_MARKER_DIR": str(markers),
        "PYTHONPATH": os.pathsep.join([STUB_DIR, os.environ.get("PYTHONPATH", "")]),
    }
    return RolloutWorkerSupervisor(
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


def test_the_supervisor_logs_what_its_fork_server_left_out_and_forks_its_workers(tmp_path, monkeypatch, caplog):
    supervisor = _stub_supervisor(tmp_path)
    monkeypatch.setattr(
        supervisor_module.RolloutWorkerSupervisor,
        "_fork_server_optional",
        lambda self, request: ["preload_stubs.safe", "preload_stubs.thread"],
    )
    scratch = tmp_path / "ipc"

    async def scenario() -> tuple[int, list]:
        outcomes = [await supervisor.run(_stub_request(scratch, f"r{index}-0")) for index in range(2)]
        assert supervisor._fork_server is not None
        return supervisor._fork_server.pid, outcomes

    try:
        with caplog.at_level(logging.INFO):
            server_pid, outcomes = asyncio.run(scenario())
    finally:
        supervisor.close()
    parents = _parents(tmp_path / "markers")
    assert [parents[outcome.worker_pid] for outcome in outcomes] == [server_pid, server_pid]  # forked, not exec'd
    assert f"fork server {server_pid} imported the worker's stack and 1 module(s) the environments open with" in (
        caplog.text
    )
    assert (
        "the rollout worker fork server left preload_stubs.thread out of what it imports ahead of its workers, which "
        "then import or build it themselves: it started threads: " in caplog.text
    )
    assert "(the Python thread 'preload-stub-thread' running time.sleep)" in caplog.text
    assert "the rollout worker fork server left preload_stubs.safe" not in caplog.text


def test_the_supervisor_gives_its_fork_server_the_run_s_preload_and_warms_it_ahead(tmp_path):
    """A supervisor of rollout workers hands its fork server the preload it was given: the modules as optional, the
    builds after the renderer's.  ``warm`` starts the server without waiting for it; the first worker finds it."""

    preload = WorkerPreload(modules=("a.b",), prepare=({"ref": "x:y", "kwargs": {"k": 1}},))
    supervisor = RolloutWorkerSupervisor(scratch_dir=str(tmp_path), start_method="forkserver", preload=preload)
    request = _stub_request(tmp_path, "r-0")
    assert supervisor._fork_server_optional(request) == ["a.b"]
    assert supervisor._fork_server_prepare(request) == [{"ref": "x:y", "kwargs": {"k": 1}}]  # no model: no renderer
    stub = _stub_supervisor(tmp_path / "stub", preload=preload)
    assert stub._fork_server_optional(request) == [] and stub._fork_server_prepare(request) == []  # not the worker

    async def scenario() -> tuple[ForkServer | None, ForkServer | None, int, int | None]:
        await stub.warm(request)
        warmed = stub._fork_server
        await stub.warm(request)  # a server runs: nothing more
        outcome = await stub.run(_stub_request(tmp_path / "stub" / "ipc", "first-0"))
        return warmed, stub._fork_server, stub._fork_server_starts, outcome.worker_pid

    try:
        warmed, used, starts, worker_pid = asyncio.run(scenario())
    finally:
        stub.close()
    assert warmed is not None and used is warmed and starts == 1
    assert _parents(tmp_path / "stub" / "markers")[worker_pid] == warmed.pid  # forked from the server it warmed


def test_exec_d_workers_and_in_process_rollouts_start_nothing_ahead(tmp_path):
    """Under ``worker_start_method: exec`` no fork server starts ahead, and under ``rollout_isolation: inprocess``
    there is no supervisor at all: nothing changes for them but the toolchains staged ahead."""

    exec_supervisor = RolloutWorkerSupervisor(scratch_dir=str(tmp_path), start_method="exec")
    model_info = SimpleNamespace(
        llm=SimpleNamespace(version=0, model_name=""), model_name="", renderer_name="", renderer_kwargs={}
    )

    def workflow(**settings) -> GroupRolloutWorkflow:
        return GroupRolloutWorkflow(
            rollout_fn=_rollout,
            get_task_fn=lambda task_id: Task(id=task_id),
            config=WorkflowConfig(rollout_config=RolloutConfig(output_dir=str(tmp_path / "rollouts")), **settings),
            model_info=model_info,
        )

    in_process = workflow(rollout_isolation="inprocess")
    exec_workers = workflow(rollout_isolation="process", worker_start_method="exec")

    async def scenario() -> None:
        await exec_supervisor.warm(_stub_request(tmp_path, "r-0"))
        await in_process.warm_up()
        await exec_workers.warm_up()

    asyncio.run(scenario())
    assert exec_supervisor._fork_server is None and exec_supervisor._fork_server_starts == 0
    assert in_process._supervisor is None
    assert exec_workers._supervisor is None or exec_workers._supervisor._fork_server is None


async def _rollout(task, config) -> dict:
    return {"trajectories": {}}


def test_a_shared_object_s_descriptor_is_fork_safe_and_a_data_file_s_is_not(tmp_path):
    """Only a shared object's descriptor (the interpreter's, a ``*.so`` or ``*.so.<n>`` file, or a file mapped
    executable), read only, may pass to the workers, as libffi's does: a data file whose name merely holds ".so" would
    share its offset among them."""

    from platoon.utils.forkserver import fork_safe_descriptor, open_descriptors

    with open("/proc/self/maps") as maps:  # a shared object this process maps (libc's, at least)
        shared_object = next(
            path for path in (line.split(maxsplit=5)[-1].strip() for line in maps) if re.search(r"\.so(\.\d+)*$", path)
        )
    data = tmp_path / "tasks.solutions.jsonl"
    data.write_text("{}\n")
    writable = tmp_path / "libwritable.so"
    writable.write_bytes(b"")
    opened = {
        "interpreter": os.open(os.path.realpath(sys.executable), os.O_RDONLY),
        "shared object": os.open(shared_object, os.O_RDONLY),
        "data": os.open(data, os.O_RDONLY),
        "writable": os.open(writable, os.O_RDWR),
    }
    try:
        targets = open_descriptors()
        safe = {what: fork_safe_descriptor(fd, targets[fd]) for what, fd in opened.items()}
    finally:
        for fd in opened.values():
            os.close(fd)
    assert safe == {"interpreter": True, "shared object": True, "data": False, "writable": False}


def test_a_later_fork_server_leaves_out_at_once_what_an_earlier_one_left_out(tmp_path, caplog):
    """A supervisor that starts another fork server (the first died, or its environment changed) hands it what the
    first left out: the new one leaves those out at once, without trying them and starting over, and the supervisor
    warns about each only once."""

    supervisor = _stub_supervisor(tmp_path)
    supervisor._fork_server_optional = lambda request: ["preload_stubs.safe", "preload_stubs.thread"]
    scratch = tmp_path / "ipc"

    async def scenario() -> tuple[ForkServer, ForkServer]:
        await supervisor.run(_stub_request(scratch, "r0-0"))
        first = supervisor._fork_server
        assert first is not None
        first.close()
        supervisor._fork_server = None
        second = supervisor._launch_fork_server(_stub_request(scratch, "r1-0"), supervisor._child_env(), os.getcwd())
        assert second is not None
        await second.ready(60.0)
        supervisor._use(second)
        return first, second

    try:
        with caplog.at_level(logging.INFO):
            first, second = asyncio.run(scenario())
    finally:
        supervisor.close()
    assert [entry["item"] for entry in first.skipped] == ["preload_stubs.thread"]
    assert second.skipped == first.skipped and second.preloaded == 1
    assert "is not fork-safe" not in Path(supervisor._fork_server_log).read_text()  # not tried again
    assert caplog.text.count("left preload_stubs.thread out of what it imports ahead") == 1


def test_the_preload_is_derived_only_for_forked_workers_outside_a_python_overlay(tmp_path, monkeypatch, caplog):
    """``platoon train`` derives a preload for a workflow whose workers fork from a fork server; for in-process
    rollouts, exec'd workers and rollouts that import their environments inside a Python overlay of their own
    (``worker_python_env_isolation``) it derives none, and warns that a ``worker_preload`` set there has no effect."""

    from platoon.train.launch import set_worker_preload

    monkeypatch.syspath_prepend(ENVS_DIR)
    environments = normalize_environment_configs(
        [{"service": "python://ors_toys:CounterEnv", "harness": "platoon-codeact", "agents": {"max_depth": 0}}]
    )
    model_info = SimpleNamespace(
        llm=SimpleNamespace(version=0, model_name=""), model_name="", renderer_name="", renderer_kwargs={}
    )

    def workflow(**settings) -> GroupRolloutWorkflow:
        return GroupRolloutWorkflow(
            rollout_fn=_rollout,
            get_task_fn=lambda task_id: Task(id=task_id),
            config=WorkflowConfig(
                rollout_config=RolloutConfig(output_dir=str(tmp_path / "rollouts")), worker_preload=["json"], **settings
            ),
            model_info=model_info,
        )

    forked = workflow(rollout_isolation="process")
    set_worker_preload(forked, environments, phase="train", tasks=[{"task_id": "train:0"}])
    assert {"openreward", "ors_toys"} <= set(forked.worker_preload.modules)
    assert forked.worker_preload.modules[-1] == "json" and forked.worker_preload.prepare

    others = [
        workflow(rollout_isolation="inprocess"),
        workflow(rollout_isolation="process", worker_start_method="exec"),
        workflow(rollout_isolation="process", worker_python_env_isolation=True),
    ]
    with caplog.at_level(logging.WARNING):
        for other in others:
            set_worker_preload(other, environments, phase="eval", tasks=[{"task_id": "train:0"}])
    assert [other.worker_preload for other in others] == [WorkerPreload()] * 3
    assert caplog.text.count("workflow_config.worker_preload has no effect on the evaluation rollouts") == 3
    assert "under worker_python_env_isolation each rollout imports its environment" in caplog.text
