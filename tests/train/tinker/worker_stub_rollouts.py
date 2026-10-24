"""Rollout functions used by the real worker entrypoint's tests.

Importable by name from inside a worker process (the test puts this directory
on the child's ``PYTHONPATH``), so the worker resolves them exactly the way it
resolves a production ``rollout_fn``.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _write_facts() -> None:
    """Record what the worker's environment actually looks like from inside."""

    path = os.environ.get("STUB_FACTS_PATH")
    if not path:
        return
    import torch

    facts = {
        "pid": os.getpid(),
        "pgid": os.getpgrp(),
        "is_session_leader": os.getpgid(os.getpid()) == os.getpid(),
        "TINKER_TELEMETRY": os.environ.get("TINKER_TELEMETRY"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
        "PLATOON_TINKER_ROLLOUT_WORKER": os.environ.get("PLATOON_TINKER_ROLLOUT_WORKER"),
        "subprocess_sampling_present": "TINKER_SUBPROCESS_SAMPLING" in os.environ,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "PYTORCH_NVML_BASED_CUDA_CHECK": os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK"),
        "cuda_driver_threads": _cuda_driver_threads(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "request_file_exists": os.path.exists(os.environ.get("STUB_REQUEST_PATH", "")),
        "extra_allow": _extra_allow(),
        "cwd": os.getcwd(),
        "python_env": _python_env(),
    }
    with open(path, "w") as handle:
        json.dump(facts, handle)


def _python_env() -> str:
    """``overlay`` when the rollout runs in its worker's own Python overlay (``python_env_isolation``), else
    ``base``."""

    root = Path(sys.executable).parent.parent
    own = root.name.startswith(f"platoon-rollout-{os.getpid()}-") and os.environ.get("VIRTUAL_ENV") == str(root)
    return "overlay" if own else "base"


def _cuda_driver_threads() -> list[str]:
    """The CUDA driver's threads in this process (``cuda00001400006``, ...), which a worker that started it runs."""

    names = []
    for tid in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{tid}/comm") as handle:
                name = handle.read().strip()
        except OSError:
            continue
        if name.startswith("cuda"):
            names.append(name)
    return sorted(names)


def _extra_allow() -> bool:
    try:
        import tinker

        return tinker.types.SampleResponse.model_config.get("extra") == "allow"
    except Exception:
        return False


async def report_and_finish(_task, _config) -> dict:
    _write_facts()
    return {"trajectories": {}}


def _stuck_cell_counters() -> dict[str, float]:
    """Stands in for an extension's cell guard: one cell started and never returned."""

    return {"ptc_cells_started": 1.0, "ptc_cells_finished": 0.0, "ptc_cells_stuck": 1.0}


def _stuck_cell_quote() -> dict[str, str]:
    return {"stuck_cells": "while True: pass"}


async def spin_forever(_task, _config) -> dict:
    from platoon.hooks import register

    _write_facts()
    # Registered here, inside the worker, like an extension's import-time
    # registration -- without leaking into the test process.
    register("rollout_worker.telemetry", _stuck_cell_counters)
    register("rollout_worker.diagnostics", _stuck_cell_quote)
    # A pure-Python spin: the interpreter still runs signal handlers at
    # bytecode boundaries, so this is killed by the worker's own SIGALRM.
    while True:
        pass


async def sleep_forever(_task, _config) -> dict:
    _write_facts()
    await asyncio.sleep(1e9)
    return {"trajectories": {}}


async def slow_but_finite(_task, _config) -> dict:
    _write_facts()
    time.sleep(float(os.environ.get("STUB_SLEEP_SECONDS", "1")))
    return {"trajectories": {}}


async def call_then_meet(task, _config) -> dict:
    """Tell the rollout's timeline that its first model call is issued, as the model gateway does as it issues one (the
    trainer's start limit then takes the rollout's start slot back), then wait, 60 s at most, until ``task.misc``'s
    ``count`` rollouts have done so, each leaving a file in its ``meet`` directory."""

    from platoon.train.tinker.rollout_metrics import current_timeline

    meeting, count = task.misc["meet"], int(task.misc["count"])
    timeline = current_timeline()
    assert timeline is not None  # bound by the rollout's session
    timeline.call_started()
    Path(meeting, f"called-{os.getpid()}").touch()
    deadline = time.monotonic() + 60.0
    while len(os.listdir(meeting)) < count and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    return {"trajectories": {}}


def _claim_slot(output_dir: str, slots: int = 8) -> int:
    """Take a distinct index for this rollout, whichever process runs it.

    Group members are handed identical tasks and configs, so a differing
    reward has to come from somewhere shared.  An exclusive-create race over a
    fixed set of files gives every member of a group a different index in both
    isolation modes, which is what makes the two modes comparable.
    """

    os.makedirs(output_dir, exist_ok=True)
    for index in range(slots):
        try:
            descriptor = os.open(os.path.join(output_dir, f"slot-{index}"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.close(descriptor)
        return index
    return slots


async def varied_reward_rollout(task, config) -> dict:
    """A real, convertible trajectory with a per-member reward.

    Registers one proxy interaction and one step that references it, so the
    production converter produces a genuine ``tinker.Datum`` -- including its
    routing handle -- in whichever process runs this.
    """

    import tinker
    from tinker_cookbook.completers import TokensWithLogprobs

    from platoon.train.tinker.proxy import TinkerLLMInteraction, proxy_interactions

    index = _claim_slot(os.environ.get("STUB_SLOT_DIR") or config.output_dir)
    completion_id = f"completion-{index}"
    proxy_interactions.get()[completion_id] = TinkerLLMInteraction(
        obs=tinker.ModelInput.from_ints([11, 12, 13, 14]),
        action=TokensWithLogprobs(tokens=[15, 16], maybe_logprobs=[-0.25, -0.5]),
        routing_id=1000 + index,
    )
    return {
        "trajectories": {
            "root": {
                "reward": float(index),
                "steps": [{"misc": {"action_misc": {"completion_id": completion_id}}}],
            }
        }
    }


def doubled_reward(trajectory: dict) -> tuple[float, dict]:
    """A module-level reward processor that a worker resolves by reference."""

    return 2 * trajectory["reward"], {"reward/doubled": 2 * trajectory["reward"]}


async def segfault_rollout(_task, _config) -> dict:
    import ctypes

    ctypes.string_at(0)
    return {"trajectories": {}}


async def spin_rollout(_task, _config) -> dict:
    while True:
        pass


async def hold_a_remote_session(_task, _config) -> dict:
    """Open a session on the test's ORS env server and never close it, as a hung rollout holds one."""

    from platoon.envs.ors import OrsEnvironmentService
    from platoon.task import Task

    service = OrsEnvironmentService(env_name=os.environ["STUB_ORS_ENV"], urls=[os.environ["STUB_ORS_URL"]])
    await service.open(Task(id="train:0"), rollout_id="held")
    await asyncio.sleep(1e9)
    return {"trajectories": {}}


async def hold_a_sandbox(_task, _config) -> dict:
    """Start a sandbox with a background command and never stop it, as a hung rollout holds its task sandbox."""

    from platoon.sandbox import SandboxSpec
    from platoon.sandbox.process import ProcessProvider

    sandbox = await ProcessProvider(root=os.environ["STUB_SANDBOX_ROOT"]).start(SandboxSpec(image="local"))
    started = await sandbox.exec("sleep 1000 > /dev/null 2>&1 & echo $!")
    facts = {"sleep_pid": int(started.stdout), "sandbox_dir": str(sandbox.root.parent), "labels": sandbox.spec.labels}
    Path(os.environ["STUB_FACTS_PATH"]).write_text(json.dumps(facts))
    await asyncio.sleep(1e9)
    return {"trajectories": {}}


async def report_registrations(_task, _config) -> dict:
    """Record whether this worker knows the harness and the program that importing ``eval_pkg`` registers."""

    from platoon.harness.registry import get_harness
    from platoon.programs import get_program

    known = {}
    for kind, lookup, name in (("harness", get_harness, "eval-submitter"), ("program", get_program, "eval-program")):
        try:
            lookup(name)
        except Exception:  # an unknown name
            known[kind] = False
        else:
            known[kind] = True
    with open(os.environ["STUB_FACTS_PATH"], "w") as handle:
        json.dump(known, handle)
    return {"trajectories": {}}


async def report_policy_gateway(_task, _config) -> dict:
    """Record the model and the sampling timeout of the policy gateway this worker serves its rollout, and the
    sampling timeout variable of the worker's own environment."""

    from platoon.gateway import default_gateway

    gateway = default_gateway()
    facts = {
        "model_name": gateway.model_name,
        "sample_timeout_seconds": gateway.sample_timeout_seconds,
        "variable": os.environ.get("TINKER_SAMPLE_TIMEOUT_SECONDS"),
    }
    with open(os.environ["STUB_FACTS_PATH"], "w") as handle:
        json.dump(facts, handle)
    return {"trajectories": {}}


# ----------------------------------------------------------------------------- recursive rollouts that are stopped
#
# A whole recursive tree through platoon's own harness path (harness_rollout, the single program, spawn, the gateway
# and its tool plane) against a fake model and a toy env.  Every model call, tool call, env session and fork is
# appended to STUB_CALL_LOG as ``<time.time()> <kind> <goal>``, so a test can tell what the worker did after it
# began terminating.  ``stubborn-recursive`` swallows cancellation and retries whatever fails, as an OpenHands
# conversation does (interrupt(), a synthetic error, then the next model call); ``polite-recursive`` lets
# cancellation through.


def _log_call(kind: str, goal: str = "") -> None:
    path = os.environ.get("STUB_CALL_LOG")
    if not path:
        return
    line = f"{time.time():.6f} {kind} {goal}\n".encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(descriptor, line)  # one write: lines from concurrent agents never interleave
    finally:
        os.close(descriptor)


class _LoggingSampler:
    async def sample_async(self, *, prompt, sampling_params, num_samples):
        import tinker

        _log_call("model")
        await asyncio.sleep(0.02)
        sequence = {"stop_reason": "stop", "tokens": [7, 8], "logprobs": [-0.1, -0.1]}
        return tinker.types.SampleResponse.model_validate({"sequences": [sequence], "type": "sample"})

    def sample(self, **kwargs):
        raise AssertionError("the gateway samples asynchronously")


class _Renderer:
    def build_generation_prompt(self, messages):
        import tinker

        return tinker.ModelInput.from_ints([1, 2, 3])

    def parse_response(self, tokens):
        return {"role": "assistant", "content": "next"}, True

    def get_stop_sequences(self):
        return []


class _LoggingSession:
    sandbox = None

    def __init__(self, task) -> None:
        self.task = task

    @property
    def instruction(self) -> str:
        return self.task.goal or ""

    async def tools(self):
        from platoon.envs.service import ToolSpec

        return [ToolSpec("look", "Look around.", {"type": "object", "properties": {}}, read_only=True)]

    async def call_tool(self, name, arguments):
        from platoon.envs.service import ToolResult

        _log_call("tool", self.task.goal or "")
        await asyncio.sleep(0.02)
        return ToolResult.text("seen")

    async def grade(self):
        from platoon.envs.service import Outcome

        return Outcome(status="scored", rewards={"reward": 0.0})

    async def fork(self, mode="shared", *, task=None):
        child = task or self.task
        _log_call("fork", child.goal or "")
        return _LoggingSession(child)

    async def close(self) -> None:
        return None


class LoggingService:
    """A toy env service: one look tool, forks for subagents, every call logged."""

    name = "logging"

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    async def list_tasks(self, split):
        return ["t0"]

    async def get_task(self, task_id):
        from platoon.task import Task

        return Task(id=task_id, goal="root")

    async def open(self, task, *, rollout_id):
        _log_call("open", task.goal or "")
        return _LoggingSession(task)


def _parent_death_signal() -> int:
    """The signal this thread gets when its parent dies (``prctl(PR_GET_PDEATHSIG)``), 0 for none."""

    import ctypes

    number = ctypes.c_int(0)
    ctypes.CDLL(None, use_errno=True).prctl(2, ctypes.byref(number), 0, 0, 0)  # PR_GET_PDEATHSIG
    return number.value


async def _keep_spawning(ctx, depth: int, *, stubborn: bool) -> None:
    """Short-lived children, one after another, for as long as the agent runs; a stubborn agent also swallows
    cancellation and retries refused launches."""

    number = 0
    while True:
        number += 1
        if not stubborn:
            run = await ctx.agents.spawn(f"short-d{depth}-{number}", max_steps=1000)
            if not run.launched:
                return
            await asyncio.sleep(0.05)
            continue
        try:
            await ctx.agents.spawn(f"short-d{depth}-{number}", max_steps=1000)
        except BaseException:
            pass
        try:
            await asyncio.sleep(0.05)
        except BaseException:
            pass


async def _recursive_agent(ctx, *, stubborn: bool):
    """Depth 0 and 1 keep a long-running child and a stream of short ones; depth 2 calls on until stopped."""

    from platoon.harness.agent import AgentResult

    if ctx.depth == 0:
        # A subprocess in the worker's process group, which must not outlive the worker.
        sleeper = subprocess.Popen(["sleep", "1000"])
        facts = os.environ.get("STUB_FACTS_PATH")
        if facts:
            noted = {
                "sleeper_pid": sleeper.pid,
                "worker_pid": os.getpid(),
                "parent_death_signal": _parent_death_signal(),
            }
            with open(facts, "w") as handle:
                json.dump(noted, handle)
    if ctx.instruction.startswith("short"):
        await ctx.chat([{"role": "user", "content": "one turn"}])
        await ctx.call_tool("look", {})
        return AgentResult("completed", final_message="short done")
    background = []
    if ctx.depth < 2:
        background.append(asyncio.ensure_future(ctx.agents.spawn(f"long-d{ctx.depth + 1}", max_steps=1000)))
        background.append(asyncio.ensure_future(_keep_spawning(ctx, ctx.depth + 1, stubborn=stubborn)))
    if not stubborn:
        try:
            while True:
                await ctx.chat([{"role": "user", "content": "go on"}])
                await ctx.call_tool("look", {})
                await asyncio.sleep(0.05)
        finally:
            for task in background:
                task.cancel()
    while True:
        try:
            await ctx.chat([{"role": "user", "content": "go on"}])
            await ctx.call_tool("look", {})
            await asyncio.sleep(0.05)
        except BaseException:  # cancellation included: "interrupted", then the next model call
            try:
                await asyncio.sleep(0.05)
            except BaseException:
                pass


try:
    from platoon.harness.function import harness as _harness
except Exception:  # pragma: no cover - the stub module is also imported where platoon's harness layer is absent
    _harness = None

if _harness is not None:

    @_harness("stubborn-recursive")
    async def stubborn_recursive(ctx):
        return await _recursive_agent(ctx, stubborn=True)

    @_harness("polite-recursive")
    async def polite_recursive(ctx):
        return await _recursive_agent(ctx, stubborn=False)


async def _recursive_rollout(task, config, harness_name: str) -> dict:
    from platoon.gateway.core import ModelGateway, set_default_gateway
    from platoon.harness.rollout import harness_rollout
    from platoon.task import Task
    from platoon.train.components import EnvironmentRoute

    gateway = ModelGateway(sampler=_LoggingSampler(), renderer=_Renderer(), tokenizer=None, model_name="stub")
    set_default_gateway(gateway)
    route = EnvironmentRoute(
        name="logging",
        service="worker_stub_rollouts:LoggingService",
        harness=harness_name,
        agents={"max_depth": 2, "budget": "depth_aware", "child_grading": "none"},
    )
    return await harness_rollout(route.attach(Task(id="t0", goal="root")), config)


async def stubborn_recursive_rollout(task, config) -> dict:
    return await _recursive_rollout(task, config, "stubborn-recursive")


async def polite_recursive_rollout(task, config) -> dict:
    return await _recursive_rollout(task, config, "polite-recursive")


# ----------------------------------------------------------------------------- rollouts that are stopped
#
# Each notes its facts in STUB_FACTS_PATH (written whole, once it is ready) and keeps its files in STUB_STATE_DIR.


def _note(**facts) -> None:
    path = os.environ["STUB_FACTS_PATH"]
    with open(path + ".tmp", "w") as handle:
        json.dump({"worker_pid": os.getpid(), "parent_death_signal": _parent_death_signal(), **facts}, handle)
    os.replace(path + ".tmp", path)


async def host_command_rollout(_task, _config) -> dict:
    """A harness CLI on the rollout host (``HarnessRunner``'s ``_LocalHost``: a session of its own) that ignores
    SIGTERM, noting each one; with ``STUB_HOLD_LOOP`` the rollout then holds its event loop in a blocking call."""

    from platoon.harness.runner import _LocalHost

    state = Path(os.environ["STUB_STATE_DIR"])
    host = _LocalHost.open(state / "host", 5.0)  # HarnessRunner's default grace
    pid_path, term_path = state / "command.pid", state / "command.term"
    command = f"trap 'echo term >> {term_path}' TERM; echo $$ > {pid_path}.tmp; mv {pid_path}.tmp {pid_path}; "
    command += "while :; do sleep 0.1; done"
    running = asyncio.ensure_future(host.shell(command, timeout_s=None))
    while not pid_path.exists():
        await asyncio.sleep(0.05)
    _note(command_pid=int(pid_path.read_text()))
    if os.environ.get("STUB_HOLD_LOOP"):
        time.sleep(1000)  # no cancellation reaches this: the loop never runs again
    await running
    return {"trajectories": {}}


async def sandbox_command_rollout(_task, _config) -> dict:
    """A sandbox command that leaves a process running in the background, one that ignores SIGTERM.

    The sandbox is held by a Harbor ``SandboxLease``: the rollout's release stops it in a background task, after the
    calls in flight, as a Harbor session's close does, and stopping it takes a while, as ``docker rm`` does.  With
    ``STUB_STUBBORN`` the rollout swallows every cancellation and never releases it; with ``STUB_LEAVE_IT`` it ends
    at once without releasing it.
    """

    from platoon.envs.harbor.session import SandboxLease
    from platoon.sandbox.base import SandboxSpec
    from platoon.sandbox.process import ProcessSandbox

    class SlowToStop(ProcessSandbox):
        async def stop(self) -> None:
            await asyncio.sleep(2.0)
            await super().stop()

    state = Path(os.environ["STUB_STATE_DIR"])
    spec = SandboxSpec(image="local")
    sandbox = await asyncio.to_thread(SlowToStop.create, state / "sandbox", spec)
    lease = SandboxLease(sandbox, spec)
    await sandbox.exec(
        "(trap '' TERM; echo $BASHPID > /tmp/bg.tmp; mv /tmp/bg.tmp /tmp/bg.pid; exec sleep 1000) > /dev/null 2>&1 &"
    )
    pid_path = sandbox.host_path("/tmp/bg.pid")
    while not pid_path.exists():
        await asyncio.sleep(0.05)
    _note(command_pid=int(pid_path.read_text()), sandbox_dir=str(state / "sandbox"))
    if os.environ.get("STUB_LEAVE_IT"):
        return {"trajectories": {}}
    if os.environ.get("STUB_STUBBORN"):
        while True:
            try:
                await asyncio.sleep(1e9)
            except BaseException:
                pass
    try:
        await asyncio.sleep(1e9)
    finally:
        await lease.release()
    return {"trajectories": {}}


async def swallow_the_first_cancellation(_task, _config) -> dict:
    """Catches the first cancellation and goes on, as an OpenHands conversation goes on after an interrupt; when
    cancelled again it unwinds, running its own cleanup."""

    state = Path(os.environ["STUB_STATE_DIR"])
    _note()
    try:
        await asyncio.sleep(1e9)
    except asyncio.CancelledError:
        (state / "cancelled once").touch()
    try:
        await asyncio.sleep(1e9)
    finally:
        (state / "unwound").touch()
    return {"trajectories": {}}


async def hold_the_loop(_task, _config) -> dict:
    """Holds the event loop for good in a call that no cancellation reaches."""

    _note()
    time.sleep(1000)
    return {"trajectories": {}}
