"""The parent side of process-per-rollout: launch, deadline, kill, classify.

One ``asyncio`` subprocess per rollout, no pool.  Measured with
four workers, ``max_tasks_per_child=1`` and spawn, a single child that
``SIGKILL``s itself fails **every** concurrent future in a
``ProcessPoolExecutor`` with ``BrokenProcessPool``.  Here a hard-timeout kill
is meant to be routine at 24-96 concurrent rollouts, so pooling would make
every straggler kill collaterally destroy its siblings.  With
``create_subprocess_exec(..., start_new_session=True)`` the same three cases
measure ``rc=0`` (normal), ``rc=-11`` with siblings unaffected (segfault
contained) and ``rc=-9`` (spinner killed by its own group) -- exactly the
semantics this design needs.

Classification is **response-first**
====================================

A worker needs ~1.9 s of interpreter teardown *after* it ``os.replace``s its
response, and during that window its parent task is still not done and can be
selected as a straggler.  Classifying on ``(returncode, we_killed_it)`` would
therefore discard a complete rollout's datums because its group's tail grace
expired two seconds after it finished.  So the supervisor always tries to read
the response first; a readable response with a matching protocol wins
regardless of the return code or who sent the signal, and the kill is recorded
separately as ``rollout_worker_killed``.

Stopping workers whose trainer is gone
======================================

Workers lead their own sessions, so a signal to the trainer's process group never reaches them.  A worker stops on
its own within a bounded time when it gets SIGTERM, from this supervisor or through PDEATHSIG when the trainer
dies, or sees its parent gone (:mod:`platoon.train.tinker.workers.entrypoint`).  Three things back that up:

* :func:`signal_live_workers` sends SIGTERM to the groups of this process's live workers; the trainer calls it
  on the exits that run no cleanup (the watchdog's and the second signal's ``os._exit``), in case PDEATHSIG does
  not deliver it;
* every live worker is recorded on disk (:mod:`platoon.train.tinker.workers.registry`), so ``platoon train`` kills
  what a dead trainer left before it starts training again (:mod:`platoon.train.reaping`);
* a new supervisor, before its first launch, kills the recorded workers of every trainer that is gone.

How a worker starts
===================

``start_method="exec"`` launches ``python -m <module> <request>``.  ``"forkserver"`` (the training config's
``worker_start_method`` default) forks it from a fork server (:mod:`platoon.utils.forkserver`) that has imported the
worker's stack once (:data:`WORKER_PRELOAD`, and the run's environment packages) and built the run's tokenizer and
renderer: an exec'd worker spends most of its start importing that stack, 17-24 s alone and about a minute with 64
starting at once on one node, then a few seconds of CPU building those.  A forked worker is
the process an exec'd one would be (its own session, ``/dev/null`` as stdin, its log file, this process's working
directory and the environment :meth:`RolloutWorkerSupervisor._child_env` gives it, the worker invariants and the
startup alarm), and is signalled, waited for and classified the same way.  The fork server dies with the trainer
and stops the workers it forked when it dies; another starts when one died, or when this process's environment or
working directory changed since it started (which its imports may have read).  A worker starts by exec whenever no
fork server can fork it: one did not start (a module did not import, a thread runs in it), the supervisor started
:data:`MAX_FORK_SERVER_STARTS` already, or a fork failed.

Neither the fork server nor a worker sees a GPU or starts the CUDA driver: their environment hides the GPUs and has
torch ask NVML (``CUDA_VISIBLE_DEVICES=``, ``PYTORCH_NVML_BASED_CUDA_CHECK=1``, :data:`WORKER_ENV_OVERRIDES`), so
the ``torch.cuda.is_available()`` that transformers calls at import answers from the environment alone.  Where an
NVIDIA driver is installed it would otherwise start the driver, whose thread leaves the fork server unable to fork
(every worker would start by exec), and whose start every exec'd worker would pay
(:mod:`platoon.train.tinker.worker_env` says more).  A worker needs no GPU: it samples through the training service and
runs its environment in sandboxes, which get their GPUs from their provider and none of a worker's environment.

Signalling is pidfd-based
=========================

Python 3.12's default child watcher reaps each child on its own daemon thread,
so ``proc.returncode is None`` can be stale by the time a signal is sent.
Every rollout worker and every MCP bridge on a host is a session leader, so
a recycled pid would satisfy a naive ``getpgid(pid) == pid`` guard and the
supervisor would ``SIGKILL`` an unrelated rollout's whole group.  A ``pidfd``
opened at launch cannot follow a recycled pid; the group signal uses the
``pgid`` captured at launch and is still guarded.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import logging
import os
import signal
import sys
import threading
import time
import uuid
from dataclasses import dataclass, replace
from typing import Callable

from platoon.envs.ors.cleanup import delete_orphaned_sessions, read_mirror
from platoon.sandbox.cleanup import read_mirror as read_sandbox_mirror
from platoon.sandbox.cleanup import reap_records
from platoon.train.tinker.worker_env import DIAG_PATH_ENV, NO_CUDA_DRIVER, STARTUP_BUDGET_ENV
from platoon.train.tinker.workers.binding import publish_binding
from platoon.train.tinker.workers.protocol import (
    EXIT_RSS_LIMIT,
    IPC_SWEEP_AGE_SECONDS,
    RolloutWorkerRequest,
    RolloutWorkerResponse,
    dump_request,
    ipc_dir,
    load_response,
    read_diag,
    unlink_quietly,
)
from platoon.train.tinker.workers.registry import (
    REGISTRY_DIRNAME,
    WorkerRecord,
    forget_worker,
    process_start_time,
    record_worker,
    registry_dir,
    sweep_orphaned_workers,
)
from platoon.utils.forkserver import ForkedWorker, ForkServer, ForkServerError

logger = logging.getLogger(__name__)

__all__ = ["RolloutWorkerSupervisor", "WorkerOutcome", "signal_live_workers"]

WORKER_MODULE = "platoon.train.tinker.workers.entrypoint"

#: How a worker starts: a new interpreter, or a fork of a fork server that imported the worker's stack once.
START_METHODS = ("exec", "forkserver")
#: What a rollout worker imports before its rollout, which a fork server imports once: the package init (tinker and
#: torch, through the trainer), what the entrypoint's ``main`` imports, and what those pull in (the gateway,
#: tinker_cookbook's renderers, transformers).  The run's environment packages (``request.packages``) are added.
WORKER_PRELOAD: tuple[str, ...] = (
    "platoon.train.tinker",
    "tinker",
    "torch",
    "platoon.train.tinker.proxy",
    "platoon.train.tinker.workers.protocol",
    "platoon.train.tinker.workers.binding",
    "platoon.train.tinker.workers.run",
    "platoon.envs.ors.cleanup",
    "platoon.sandbox.cleanup",
    "platoon.train.auto",
)
#: In an exec'd worker of a module under ``platoon.train.tinker`` the package init applies the worker invariants and
#: arms the startup alarm (:mod:`platoon.train.tinker.worker_env`); a forked worker, whose server ran that init
#: already, does it before its module runs.
WORKER_PACKAGE_INIT = "platoon.train.tinker.worker_env:apply_rollout_worker_env_invariants"
#: How many fork servers a supervisor starts, the first and those that replace one that died.
MAX_FORK_SERVER_STARTS = 4

#: Environment forced on every worker.  Deliberately inherit-plus-override
#: rather than an allowlist: the trainer's environment is already curated
#: (HF_HOME, offline flags, session URLs, TMPDIR, gateway URL and key), and an
#: allowlist would silently drop future keys.
WORKER_ENV_OVERRIDES = {
    "TINKER_TELEMETRY": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    **NO_CUDA_DRIVER,  # no GPU and no CUDA driver in a worker or its fork server (worker_env says why)
    "PLATOON_TINKER_ROLLOUT_WORKER": "1",
}

WORKER_ENV_REMOVED = ("TINKER_SUBPROCESS_SAMPLING",)

# This process's live workers, pid -> (pidfd, pgid), for signal_live_workers.  An RLock: a signal handler on the
# thread that holds it may call signal_live_workers.
_live_workers: dict[int, tuple[int | None, int | None]] = {}
_live_workers_lock = threading.RLock()


def signal_live_workers(sig: int = signal.SIGTERM) -> int:
    """Send ``sig`` to the process group of every rollout worker this process runs; returns how many.

    For exits that run no cleanup (the trainer's watchdog and its second-signal handler end with ``os._exit``).
    On SIGTERM a worker deletes its env sessions and exits within its termination deadline, as it does on the
    PDEATHSIG SIGTERM that the exit itself delivers; this covers a worker for which that signal is lost.  Safe
    from any thread and from a signal handler; never raises.
    """

    with _live_workers_lock:
        workers = list(_live_workers.items())
        for pid, (pidfd, pgid) in workers:
            RolloutWorkerSupervisor._signal_group(pgid, sig)
            RolloutWorkerSupervisor._signal_process(pidfd, pid, sig)
    return len(workers)


@dataclass(frozen=True)
class WorkerOutcome:
    """Everything the workflow needs to turn one worker into a group member."""

    response: RolloutWorkerResponse | None
    returncode: int | None
    disposition: str
    killed_by_parent: bool
    diag: dict | None
    launch_seconds: float
    wall_seconds: float
    retried: bool
    worker_pid: int | None
    log_path: str | None = None
    request_id: str = ""


class RolloutWorkerSupervisor:
    """Runs one rollout in one exec'd process, and reaps its whole tree."""

    def __init__(
        self,
        *,
        python_executable: str | None = None,
        module: str = WORKER_MODULE,
        scratch_dir: str | None = None,
        log_dir: str | None = None,
        termination_grace_seconds: float = 10.0,
        parent_grace_seconds: float = 60.0,
        mem_available_floor_bytes: int = 0,
        env_overrides: dict[str, str] | None = None,
        launch_retries: int = 1,
        ready_poll_seconds: float = 0.2,
        orphaned_session_timeout_seconds: float = 10.0,
        orphaned_sandbox_timeout_seconds: float = 120.0,
        start_method: str = "exec",
    ) -> None:
        if start_method not in START_METHODS:
            raise ValueError(f"start_method must be one of {START_METHODS}, not {start_method!r}")
        self.python_executable = python_executable or sys.executable
        self.module = module
        self.start_method = start_method
        self._fork_server: ForkServer | None = None
        self._fork_server_lock: tuple[asyncio.AbstractEventLoop, asyncio.Lock] | None = None
        self._fork_server_starts = 0
        self.scratch_dir = scratch_dir or ipc_dir()
        self.log_dir = log_dir
        # A job launcher may give the whole client 30 s between SIGTERM and SIGKILL,
        # so a per-worker grace anywhere near that could not complete inside it
        # and the trainer would die mid-reap.  Workers are always reaped
        # concurrently, never serially.
        self.termination_grace_seconds = float(termination_grace_seconds)
        self.parent_grace_seconds = float(parent_grace_seconds)
        self.mem_available_floor_bytes = int(mem_available_floor_bytes)
        self.env_overrides = dict(env_overrides or {})
        self.launch_retries = int(launch_retries)
        self.ready_poll_seconds = float(ready_poll_seconds)
        self.orphaned_session_timeout_seconds = float(orphaned_session_timeout_seconds)
        self.orphaned_sandbox_timeout_seconds = float(orphaned_sandbox_timeout_seconds)
        self.binding_path = os.path.join(self.scratch_dir, "model_binding.pkl")
        self.registry_dir = registry_dir(self.scratch_dir)
        self._orphans_swept: asyncio.Lock | None = None
        self._orphans_sweep_done = False
        self._sweep_stale_files()

    # -- setup -----------------------------------------------------------
    def _sweep_stale_files(self) -> None:
        cutoff = time.time() - IPC_SWEEP_AGE_SECONDS
        try:
            names = os.listdir(self.scratch_dir)
        except OSError:
            return
        for name in names:
            if name in (os.path.basename(self.binding_path), REGISTRY_DIRNAME):
                continue
            path = os.path.join(self.scratch_dir, name)
            try:
                if os.stat(path).st_mtime < cutoff:
                    unlink_quietly(path)
            except OSError:
                continue

    # -- the model binding ----------------------------------------------
    def publish_binding(self, sampling_client, version: int) -> None:
        """Republish the trainer's sampling client so workers can follow it."""

        publish_binding(sampling_client, version, self.binding_path)

    # -- a dead trainer's workers ------------------------------------------
    async def sweep_orphaned_workers(self) -> list[int]:
        """Kill the recorded workers of every trainer that is gone, once, before this supervisor's first launch.

        They belong to a trainer that died without reaping them (see the module docstring) and would otherwise run
        next to this trainer's.  Workers of a trainer that still runs are never touched.  Returns their pids.
        """

        if self._orphans_sweep_done:
            return []
        if self._orphans_swept is None:
            self._orphans_swept = asyncio.Lock()
        async with self._orphans_swept:
            if self._orphans_sweep_done:
                return []
            try:
                stopped = await asyncio.to_thread(
                    sweep_orphaned_workers, self.registry_dir, grace_seconds=self.termination_grace_seconds
                )
            except Exception:
                logger.warning("could not sweep the rollout workers of dead trainers", exc_info=True)
                stopped = []
            self._orphans_sweep_done = True
        if stopped:
            logger.warning("stopped %d rollout worker(s) left by a dead trainer: %s", len(stopped), stopped)
        return stopped

    # -- signalling ------------------------------------------------------
    @staticmethod
    def _signal_process(pidfd: int | None, pid: int, sig: int) -> None:
        try:
            if pidfd is not None and hasattr(signal, "pidfd_send_signal"):
                signal.pidfd_send_signal(pidfd, sig)
            else:  # pragma: no cover - pidfd is available on Linux 5.3+
                os.kill(pid, sig)
        except (ProcessLookupError, OSError) as exc:
            if getattr(exc, "errno", None) not in (errno.ESRCH, errno.EBADF):
                logger.debug("could not signal worker pid %s: %s", pid, exc)

    @staticmethod
    def _signal_group(pgid: int | None, sig: int) -> None:
        if not pgid:
            return
        try:
            # Only signal a group that is still led by the pid we launched.
            if os.getpgid(pgid) != pgid:
                return
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            return

    def _signal_tree(self, pidfd: int | None, pid: int, pgid: int | None, sig: int) -> None:
        self._signal_group(pgid, sig)
        self._signal_process(pidfd, pid, sig)

    # -- memory admission -------------------------------------------------
    def _mem_available_bytes(self) -> int:
        try:
            with open("/proc/meminfo", "rb") as handle:
                for line in handle:
                    if line.startswith(b"MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass
        return 0

    def _memory_floor_blocks_launch(self) -> bool:
        if self.mem_available_floor_bytes <= 0:
            return False
        available = self._mem_available_bytes()
        if available and available < self.mem_available_floor_bytes:
            logger.warning(
                "refusing to launch a rollout worker: MemAvailable %.1f GB is below the floor %.1f GB",
                available / 1024**3,
                self.mem_available_floor_bytes / 1024**3,
            )
            return True
        return False

    # -- the fork server --------------------------------------------------------
    def _fork_server_preload(self, request: RolloutWorkerRequest) -> list[str]:
        if self.module != WORKER_MODULE:
            return []
        return [*WORKER_PRELOAD, *request.packages]

    def _fork_server_child_init(self) -> list[str]:
        return [WORKER_PACKAGE_INIT] if self.module.startswith("platoon.train.tinker.") else []

    def _fork_server_prepare(self, request: RolloutWorkerRequest) -> list[dict]:
        """The run's tokenizer and renderer, built once in the server: every worker it forks takes its own copy
        instead of building them (a few seconds of CPU each, :func:`platoon.train.tinker.proxy.prepare_renderer`)."""

        binding = request.model
        if self.module != WORKER_MODULE or not binding.model_name:
            return []
        kwargs = {
            "model_name": binding.model_name,
            "renderer_name": binding.renderer_name,
            "renderer_kwargs": dict(binding.renderer_kwargs),
        }
        return [{"ref": "platoon.train.tinker.proxy:prepare_renderer", "kwargs": kwargs}]

    async def _fork_server_for(self, request: RolloutWorkerRequest) -> ForkServer | None:
        """The fork server to start ``request``'s worker, started on first use; None to start it by exec.

        A server forks workers with the environment and working directory it started with, which its imports may
        have read: when this process's have changed since, another server starts, and the old one serves the workers
        it forked to their end."""

        if self.start_method != "forkserver":
            return None
        env, cwd = self._child_env(), os.getcwd()
        loop = asyncio.get_running_loop()
        server = self._fork_server
        if server is not None and server.alive and server.loop is loop and server.env == env and server.cwd == cwd:
            return server
        if self._fork_server_starts >= MAX_FORK_SERVER_STARTS:
            return None
        if self._fork_server_lock is None or self._fork_server_lock[0] is not loop:
            self._fork_server_lock = (loop, asyncio.Lock())
        async with self._fork_server_lock[1]:
            server = self._fork_server
            if server is not None and server.alive and server.loop is loop:
                if server.env == env and server.cwd == cwd:
                    return server
                changed = sorted(
                    name for name in env.keys() | server.env.keys() if env.get(name) != server.env.get(name)
                )
                logger.warning(
                    "rollout worker fork server %s started with another %s; starting another",
                    server.pid,
                    f"environment ({', '.join(changed[:8])})" if changed else "working directory",
                )
            if server is not None:
                server.close()  # dead, of another event loop or of another environment; it serves its workers out
                self._fork_server = None
            if self._fork_server_starts >= MAX_FORK_SERVER_STARTS:
                return None
            self._fork_server_starts += 1
            log_path = os.path.join(self.scratch_dir, f"forkserver-{os.getpid()}-{self._fork_server_starts}.log")
            try:
                server = await ForkServer.start(
                    python=self.python_executable,
                    module=self.module,
                    preload=self._fork_server_preload(request),
                    child_init=self._fork_server_child_init(),
                    prepare=self._fork_server_prepare(request),
                    env=env,
                    cwd=cwd,
                    log_path=log_path,
                    timeout_s=request.startup_budget_seconds,
                )
            except Exception as exc:  # noqa: BLE001 - workers then start by exec
                if isinstance(exc, ForkServerError) and exc.permanent:
                    self._fork_server_starts = MAX_FORK_SERVER_STARTS  # starting another would not help
                logger.warning(
                    "rollout workers start by exec, each importing the worker's stack itself: the fork server did not "
                    "start: %s (log %s)",
                    exc,
                    log_path,
                )
                return None
            self._fork_server = server
            logger.info(
                "rollout worker fork server %s imported the worker's stack in %.1fs", server.pid, server.seconds
            )
            for problem in server.prepare_errors:
                logger.warning("the rollout worker fork server could not build ahead (workers build it): %s", problem)
            return server

    def close(self) -> None:
        """Stop the fork server taking workers; it exits once the workers it forked have exited."""

        if self._fork_server is not None:
            self._fork_server.close()
            self._fork_server = None

    # -- the run ----------------------------------------------------------
    def _child_env(self, request: RolloutWorkerRequest | None = None) -> dict[str, str]:
        env = dict(os.environ)
        env.update(WORKER_ENV_OVERRIDES)
        if request is not None:
            # The child's startup deadline must exist before it can read the
            # request (reading it needs the very imports the deadline bounds),
            # so it travels in the environment.
            env[STARTUP_BUDGET_ENV] = repr(float(request.startup_budget_seconds))
            env[DIAG_PATH_ENV] = request.diag_path
        env.update(self.env_overrides)
        for name in WORKER_ENV_REMOVED:
            env.pop(name, None)
        return env

    def _log_path(self, request: RolloutWorkerRequest, log_dir: str | None = None) -> str | None:
        """Where this worker's stdout+stderr land.

        Each worker owns a file that survives its own SIGKILL, which is what
        keeps a rollout traceback readable: otherwise it would land in the
        trainer's log, and a killed worker would leave nothing at all.
        """

        directory = log_dir or self.log_dir
        if not directory:
            return None
        return os.path.join(directory, f"{request.request_id}.log")

    def _fresh_request(self, request: RolloutWorkerRequest) -> RolloutWorkerRequest:
        """A retry gets a new id so it can never collide with the first attempt."""

        stem = request.request_id.rsplit("-", 1)[0]
        request_id = f"{stem}-{uuid.uuid4().hex[:8]}"
        return replace(
            request,
            request_id=request_id,
            response_path=os.path.join(self.scratch_dir, f"{request_id}.res"),
            ready_path=os.path.join(self.scratch_dir, f"{request_id}.ready"),
        )

    async def run(
        self,
        request: RolloutWorkerRequest,
        *,
        log_dir: str | None = None,
        on_ready: Callable[[], None] | None = None,
        on_launch: Callable[[int], None] | None = None,
        on_reap: Callable[[int], None] | None = None,
    ) -> WorkerOutcome:
        """Run one rollout in one process.  Never raises except ``CancelledError``.

        A pre-exec failure (request write, log open, exec) is retried once with
        a fresh request id -- safe because nothing user-visible has run and the
        child unlinks the request only after loading it -- and then reported as
        ``worker_launch_failed``.  This matters: with the venv and the model
        cache on a shared filesystem, a one-minute metadata stall during a
        group's imports would otherwise drop several members below
        ``min_successful_group_size`` and discard the whole task's generation
        cost.
        """

        await self.sweep_orphaned_workers()
        attempt_request = request
        retried = False
        last_error: BaseException | None = None
        for attempt in range(self.launch_retries + 1):
            if attempt:
                retried = True
                attempt_request = self._fresh_request(attempt_request)
            try:
                return await self._run_once(
                    attempt_request,
                    retried=retried,
                    log_dir=log_dir,
                    on_ready=on_ready,
                    on_launch=on_launch,
                    on_reap=on_reap,
                )
            except asyncio.CancelledError:
                raise
            except _LaunchFailed as exc:
                last_error = exc.__cause__ or exc
                logger.warning(
                    "rollout worker launch failed for %s (attempt %s/%s): %s",
                    attempt_request.request_id,
                    attempt + 1,
                    self.launch_retries + 1,
                    last_error,
                )
            except Exception as exc:  # pragma: no cover - run_once is defensive
                last_error = exc
                logger.exception("rollout worker supervision failed for %s", attempt_request.request_id)
                break
        return WorkerOutcome(
            response=None,
            returncode=None,
            disposition="worker_launch_failed",
            killed_by_parent=False,
            diag=None,
            launch_seconds=0.0,
            wall_seconds=0.0,
            retried=retried,
            worker_pid=None,
            log_path=self._log_path(attempt_request, log_dir),
            request_id=attempt_request.request_id,
        )

    async def _run_once(
        self,
        request: RolloutWorkerRequest,
        *,
        retried: bool,
        log_dir: str | None,
        on_ready: Callable[[], None] | None,
        on_launch: Callable[[int], None] | None,
        on_reap: Callable[[int], None] | None,
    ) -> WorkerOutcome:
        started = time.monotonic()
        request_path = os.path.join(self.scratch_dir, f"{request.request_id}.req")
        log_path = self._log_path(request, log_dir)
        log_handle = None
        proc: asyncio.subprocess.Process | ForkedWorker | None = None
        pidfd: int | None = None
        pgid: int | None = None
        killed = False
        ready_watcher: asyncio.Task | None = None
        launch_seconds = 0.0

        if self._memory_floor_blocks_launch():
            raise _LaunchFailed("MemAvailable below the configured floor")

        try:
            dump_request(request, request_path)
            if log_path:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
            server = await self._fork_server_for(request)
            if server is not None:
                try:
                    proc = await server.spawn(
                        [request_path], env=self._child_env(request), cwd=server.cwd, log_path=log_path
                    )
                except ForkServerError as exc:
                    logger.warning(
                        "the fork server did not start rollout worker %s (%s); starting it by exec",
                        request.request_id,
                        exc,
                    )
            if proc is None:
                if log_path:
                    log_handle = open(log_path, "ab", buffering=0)
                proc = await asyncio.create_subprocess_exec(
                    self.python_executable,
                    "-m",
                    self.module,
                    request_path,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=log_handle if log_handle else asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.STDOUT if log_handle else asyncio.subprocess.DEVNULL,
                    start_new_session=True,
                    close_fds=True,
                    env=self._child_env(request),
                )
        except asyncio.CancelledError:
            if log_handle:
                log_handle.close()
            unlink_quietly(request_path)
            raise
        except Exception as exc:
            if log_handle:
                log_handle.close()
            unlink_quietly(request_path)
            raise _LaunchFailed(str(exc)) from exc

        if isinstance(proc, ForkedWorker):
            pidfd = proc.pidfd  # the fork server opened it before the pid could be reused
        else:
            try:
                pidfd = os.pidfd_open(proc.pid) if hasattr(os, "pidfd_open") else None
            except OSError:
                pidfd = None
        try:
            pgid = os.getpgid(proc.pid)
        except OSError:
            pgid = proc.pid
        self._remember(proc.pid, pidfd, pgid, request)
        if on_launch is not None:
            with contextlib.suppress(Exception):
                on_launch(proc.pid)

        waiter = asyncio.ensure_future(proc.wait())
        ready_state = {"seconds": 0.0}
        ready_watcher = asyncio.ensure_future(self._watch_ready(request, proc, started, ready_state, on_ready))

        deadline = request.rollout_budget_seconds + self.parent_grace_seconds
        try:
            try:
                await asyncio.wait_for(asyncio.shield(waiter), timeout=deadline)
                timed_out = False
            except asyncio.TimeoutError:
                timed_out = True
                logger.warning(
                    "rollout worker %s exceeded the parent deadline of %.0fs; reaping its process group",
                    request.request_id,
                    deadline,
                )
        finally:
            # Reap the process TREE on every exit path: normal, timeout, and
            # CancelledError (a straggler cancellation kills the worker).
            # Running until its wait ends (a forked worker whose fork server died exits with no return code).
            if not waiter.done():
                killed = True
                self._signal_tree(pidfd, proc.pid, pgid, signal.SIGTERM)
                if self.termination_grace_seconds > 0:
                    with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                        await asyncio.wait_for(asyncio.shield(waiter), self.termination_grace_seconds)
            if not waiter.done():
                self._signal_tree(pidfd, proc.pid, pgid, signal.SIGKILL)
                with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.shield(waiter), 30.0)
            # Forgotten before its pidfd closes, so signal_live_workers never signals a closed (or reused) fd.
            self._forget(proc.pid, reaped=waiter.done())
            if ready_watcher is not None:
                # Let it observe the exit first: readiness is a real number we
                # want even for a worker that finished quickly.
                if not ready_watcher.done():
                    with contextlib.suppress(BaseException):
                        await asyncio.wait_for(asyncio.shield(ready_watcher), 5.0)
                ready_watcher.cancel()
                with contextlib.suppress(BaseException):
                    await ready_watcher
            if isinstance(proc, ForkedWorker):
                proc.close()
            elif pidfd is not None:
                with contextlib.suppress(OSError):
                    os.close(pidfd)
            if log_handle:
                log_handle.close()
            unlink_quietly(request_path)
            # A worker that died before closing its remote env sessions left them listed: delete them now
            # rather than leave them to the env server's 15-minute reaper.
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(self._delete_orphaned_sessions(request))
            # And the sandboxes it did not stop (Harbor task sandboxes, verifier sandboxes): reaped by its label.
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(self._reap_orphaned_sandboxes(request))
            if on_reap is not None:
                with contextlib.suppress(Exception):
                    on_reap(proc.pid)

        launch_seconds = ready_state["seconds"]
        outcome = self._classify(
            request,
            returncode=proc.returncode,
            killed_by_parent=killed,
            timed_out=timed_out,
            launch_seconds=launch_seconds,
            wall_seconds=time.monotonic() - started,
            retried=retried,
            worker_pid=proc.pid,
            log_path=log_path,
        )
        unlink_quietly(request.response_path)
        unlink_quietly(request.diag_path)
        unlink_quietly(request.ready_path)
        if outcome.response is None:
            _remove_event_temporaries(request, proc.pid)
        return outcome

    def _remember(self, pid: int, pidfd: int | None, pgid: int | None, request: RolloutWorkerRequest) -> None:
        with _live_workers_lock:
            _live_workers[pid] = (pidfd, pgid)
        start_time = process_start_time(pid)
        trainer_start_time = process_start_time(os.getpid())
        if start_time is None or trainer_start_time is None:
            return  # gone already, or no /proc: nothing a sweep could check
        record_worker(
            self.registry_dir,
            WorkerRecord(
                pid=pid,
                pgid=pgid or pid,
                start_time=start_time,
                trainer_pid=os.getpid(),
                trainer_start_time=trainer_start_time,
                request_id=request.request_id,
            ),
        )

    def _forget(self, pid: int, *, reaped: bool) -> None:
        with _live_workers_lock:
            _live_workers.pop(pid, None)
        if reaped:  # an unreaped worker stays recorded, for a later sweep
            forget_worker(self.registry_dir, pid)

    async def _delete_orphaned_sessions(self, request: RolloutWorkerRequest) -> None:
        records = read_mirror(request.sessions_path)
        unlink_quietly(request.sessions_path)
        if records:
            logger.warning(
                "rollout worker %s left %d remote env sessions open; deleting them", request.request_id, len(records)
            )
            await delete_orphaned_sessions(records, self.orphaned_session_timeout_seconds)

    async def _reap_orphaned_sandboxes(self, request: RolloutWorkerRequest) -> None:
        """Reap, by the worker's label, the sandboxes its file still lists: the worker is gone, so nothing else
        will stop them (:mod:`platoon.sandbox.cleanup`)."""

        records = read_sandbox_mirror(request.sandboxes_path)
        unlink_quietly(request.sandboxes_path)
        if records:
            logger.warning(
                "rollout worker %s left sandboxes running; reaping them (%s)",
                request.request_id,
                ", ".join(f"{record.provider} {record.labels}" for record in records),
            )
            reaped = await reap_records(records, self.orphaned_sandbox_timeout_seconds)
            logger.warning("reaped %d sandbox(es) of rollout worker %s", reaped, request.request_id)

    async def _watch_ready(
        self,
        request: RolloutWorkerRequest,
        proc,
        started: float,
        state: dict,
        on_ready: Callable[[], None] | None,
    ) -> None:
        """Fire once the child has finished importing and registering.

        ``create_subprocess_exec`` returns in well under a millisecond, so
        gating group admission on exec would bound nothing; the child then
        spends 20-30 s importing.  Readiness is the only useful gate.
        """

        deadline = started + request.startup_budget_seconds + 30.0
        while True:
            if os.path.exists(request.ready_path):
                state["seconds"] = time.monotonic() - started
                break
            if proc.returncode is not None or time.monotonic() > deadline:
                state["seconds"] = time.monotonic() - started
                break
            await asyncio.sleep(self.ready_poll_seconds)
        if on_ready is not None:
            with contextlib.suppress(Exception):
                on_ready()

    def _classify(
        self,
        request: RolloutWorkerRequest,
        *,
        returncode: int | None,
        killed_by_parent: bool,
        timed_out: bool,
        launch_seconds: float,
        wall_seconds: float,
        retried: bool,
        worker_pid: int,
        log_path: str | None,
    ) -> WorkerOutcome:
        response: RolloutWorkerResponse | None = None
        try:
            if os.path.exists(request.response_path):
                candidate = load_response(request.response_path)
                if candidate.request_id == request.request_id:
                    response = candidate
                else:
                    logger.warning(
                        "rollout worker response at %s is for request %s, not %s; ignoring",
                        request.response_path,
                        candidate.request_id,
                        request.request_id,
                    )
        except Exception:
            logger.warning(
                "rollout worker %s left an unreadable response at %s",
                request.request_id,
                request.response_path,
                exc_info=True,
            )

        diag = read_diag(request.diag_path)

        if response is not None:
            # Response-first: a complete rollout that was signalled during its
            # ~1.9 s of interpreter teardown still counts, and the kill is
            # recorded separately.
            return WorkerOutcome(
                response=response,
                returncode=returncode,
                disposition=response.status,
                killed_by_parent=killed_by_parent,
                diag=diag,
                launch_seconds=launch_seconds,
                wall_seconds=wall_seconds,
                retried=retried,
                worker_pid=worker_pid,
                log_path=log_path,
                request_id=request.request_id,
            )

        if returncode == EXIT_RSS_LIMIT or (diag and diag.get("phase") == "rss"):
            disposition = "rollout_worker_oom"
        elif diag and diag.get("phase") in ("startup", "rollout"):
            disposition = "hard_timeout"
        elif diag and diag.get("phase") == "terminated":
            # It stopped on a SIGTERM (this supervisor's, or its trainer's death through PDEATHSIG) and SIGKILLed
            # its own process group at the end: a crash as far as the group is concerned, unless it was this
            # supervisor's deadline that stopped it.
            disposition = "hard_timeout" if timed_out else "worker_crashed"
        elif timed_out:
            # The child's own alarm should have fired first; if it did not, the
            # timeout ladder is misconfigured.
            logger.warning(
                "rollout worker %s hit the parent deadline without leaving a diagnostic marker; "
                "check that the child budget is below the parent budget",
                request.request_id,
            )
            disposition = "hard_timeout"
        elif returncode is None:
            disposition = "worker_crashed"
        elif returncode != 0:
            disposition = "worker_crashed"
        else:
            # Exited cleanly but wrote nothing readable: treat as a launch
            # failure, never as a partial unpickle.
            disposition = "worker_launch_failed"

        return WorkerOutcome(
            response=None,
            returncode=returncode,
            disposition=disposition,
            killed_by_parent=killed_by_parent,
            diag=diag,
            launch_seconds=launch_seconds,
            wall_seconds=wall_seconds,
            retried=retried,
            worker_pid=worker_pid,
            log_path=log_path,
            request_id=request.request_id,
        )


def _remove_event_temporaries(request: RolloutWorkerRequest, pid: int) -> None:
    """Delete the hidden event files of a worker that was killed while writing its rollout's events."""

    output_dir = (request.rollout_config or {}).get("output_dir")
    if not output_dir:
        return
    from platoon.visualization.event_sinks import remove_temporary_events

    try:
        removed = remove_temporary_events(output_dir, pid)
    except OSError:
        return
    if removed:
        logger.info("removed %d temporary event files rollout worker %s left behind", removed, request.request_id)


class _LaunchFailed(RuntimeError):
    """Nothing user-visible ran, so the attempt is safe to repeat."""
