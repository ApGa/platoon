"""The parent side of process-per-rollout: launch, deadline, kill, classify.

:meth:`RolloutWorkerSupervisor.run` runs one rollout in a worker process of its own
(:mod:`platoon.train.workers.entrypoint`): it launches the worker, forked from its fork server
(:mod:`platoon.train.workers.fork_servers`) or by exec, awaits its exit under the parent's deadline, reaps its process
tree whatever ends the wait, and classifies what it left, response-first (:mod:`platoon.train.workers.outcomes` says
why).

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

Stopping workers whose trainer is gone
======================================

Workers lead their own sessions, so a signal to the trainer's process group never reaches them.  A worker stops on
its own within a bounded time when it gets SIGTERM, from this supervisor or through PDEATHSIG when the trainer
dies, or sees its parent gone (:mod:`platoon.train.workers.entrypoint`).  Three things back that up:

* :func:`signal_live_workers` sends SIGTERM to the groups of this process's live workers; the trainer calls it
  on the exits that run no cleanup (the watchdog's and the second signal's ``os._exit``), in case PDEATHSIG does
  not deliver it;
* every live worker is recorded on disk (:mod:`platoon.train.workers.registry`), so ``platoon train`` kills
  what a dead trainer left before it starts training again (:mod:`platoon.train.reaping`);
* a new supervisor, before its first launch, kills the recorded workers of every trainer that is gone.

How a worker starts
===================

``start_method="exec"`` launches ``python -m <module> <request>``.  ``"forkserver"`` (the training config's
``worker_start_method`` default) forks it from a fork server (:mod:`platoon.utils.forkserver`) that has imported the
worker's stack once (:data:`WORKER_PRELOAD`, and the run's environment packages) and built the run's tokenizer and
renderer: an exec'd worker spends most of its start importing that stack, 17-24 s alone and about a minute with 64
starting at once on one node, then a few seconds of CPU building those.  The server also imports and builds, when
that is fork-safe, what the run's environments and harnesses import and build when a rollout opens them (``preload``,
:mod:`platoon.train.startup`), which each worker would otherwise do again after its fork; an item that
is not fork-safe is left out, and logged with the reason.  :meth:`RolloutWorkerSupervisor.warm` starts the server
ahead of the first worker, as ``platoon train`` does while it gets ready to train.  A forked worker is
the process an exec'd one would be (its own session, ``/dev/null`` as stdin, its log file, this process's working
directory and the environment :meth:`RolloutWorkerSupervisor._child_env` gives it, the worker invariants and the
startup alarm), and is signalled, waited for and classified the same way.  The fork server dies with the trainer
and stops the workers it forked when it dies; another starts when one died (unless it said it cannot serve), or when
this process's environment or working directory changed since it started (which its imports may have read).  A worker
starts by exec whenever no fork server can fork it: one cannot serve (a module did not import, a thread runs in it),
the supervisor started :data:`MAX_FORK_SERVER_STARTS` already, a fork failed, or the server is still importing after
the worker's startup budget (it is not replaced: it forks the workers asked for after it has imported the stack).

Neither the fork server nor a worker sees a GPU or starts the CUDA driver: their environment hides the GPUs and has
torch ask NVML (``CUDA_VISIBLE_DEVICES=``, ``PYTORCH_NVML_BASED_CUDA_CHECK=1``, :data:`WORKER_ENV_OVERRIDES`), so
the ``torch.cuda.is_available()`` that transformers calls at import answers from the environment alone.  Where an
NVIDIA driver is installed it would otherwise start the driver, whose thread leaves the fork server unable to fork
(every worker would start by exec), and whose start every exec'd worker would pay
(:mod:`platoon.train.workers.env` says more).  A worker needs no GPU: it samples through the training service and
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
from dataclasses import dataclass, field, replace
from typing import BinaryIO, Callable

from platoon.envs.ors.cleanup import delete_orphaned_sessions, read_mirror
from platoon.sandbox.cleanup import read_mirror as read_sandbox_mirror
from platoon.sandbox.cleanup import reap_records
from platoon.train.startup import WorkerPreload
from platoon.train.workers import fork_servers
from platoon.train.workers.binding import publish_binding
from platoon.train.workers.env import _ENTRYPOINT_MODULES, DIAG_PATH_ENV, NO_CUDA_DRIVER, STARTUP_BUDGET_ENV

# The two removers are private to platoon.train.workers: the supervisor runs them itself, as an attempt ends.
from platoon.train.workers.outcomes import WorkerOutcome, _remove_event_temporaries, _remove_worker_files, classify
from platoon.train.workers.protocol import (
    IPC_SWEEP_AGE_SECONDS,
    RolloutWorkerRequest,
    binding_path,
    dump_request,
    ipc_dir,
    is_binding_file,
    unlink_quietly,
)
from platoon.train.workers.registry import (
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

WORKER_MODULE = "platoon.train.workers.entrypoint"

#: How a worker starts: a new interpreter, or a fork of a fork server that imported the worker's stack once.
START_METHODS = ("exec", "forkserver")
#: What a fork server imports once, for each worker it forks: the entrypoint's package, whose init applies the worker
#: invariants first; the trainer and its configuration, which the package init imported before platoon 0.5.0 (tinker,
#: torch and datasets, through the trainer), so that a forked worker starts with what it started with then; what the
#: entrypoint's ``main`` imports, and what those pull in (the gateway, tinker_cookbook's renderers, transformers).
#: The run's environment packages (``request.packages``) are added.
WORKER_PRELOAD: tuple[str, ...] = (
    "platoon.train.workers",
    "platoon.train.config_defs",
    "platoon.train.step.batch_transforms",
    "platoon.train.trainer.rl",
    "tinker",
    "torch",
    "platoon.train.tinker.proxy",
    "platoon.train.workers.protocol",
    "platoon.train.workers.binding",
    "platoon.train.workers.run",
    "platoon.envs.ors.cleanup",
    "platoon.sandbox.cleanup",
    "platoon.train.auto",
)
#: In an exec'd worker of a module under ``platoon.train.workers`` (or ``platoon.train.tinker``, the entrypoint's
#: package before platoon 0.5.0) the package init applies the worker invariants and arms the startup alarm
#: (:mod:`platoon.train.workers.env`); a forked worker, whose server ran that init already, does it before its module
#: runs.
WORKER_PACKAGE_INIT = "platoon.train.workers.env:apply_rollout_worker_env_invariants"
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
    **NO_CUDA_DRIVER,  # no GPU and no CUDA driver in a worker or its fork server (workers.env says why)
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


@dataclass(kw_only=True)
class _LaunchedWorker:
    """A worker that :meth:`RolloutWorkerSupervisor._launch` started, and what supervising it to its end takes; the
    comment above each group of fields names the step that sets them."""

    # _launch: the request and when its attempt started, its request file and log, its process.
    request: RolloutWorkerRequest
    started: float
    request_path: str
    log_path: str | None
    log_handle: BinaryIO | None
    proc: asyncio.subprocess.Process | ForkedWorker
    # _track: the pidfd and process group that signal it, the wait for its exit and the watch for its readiness.
    pidfd: int | None = None
    pgid: int | None = None
    waiter: asyncio.Future = field(init=False, repr=False)
    ready_watcher: asyncio.Future | None = None
    # _watch_ready: the seconds it took to be ready, the outcome's launch seconds.
    ready_state: dict = field(default_factory=lambda: {"seconds": 0.0})
    # _reap: whether this supervisor signalled it.
    killed: bool = False


class RolloutWorkerSupervisor:
    """Runs one rollout in one worker process, exec'd or forked from a fork server, and reaps its whole tree."""

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
        preload: WorkerPreload | None = None,
    ) -> None:
        if start_method not in START_METHODS:
            raise ValueError(f"start_method must be one of {START_METHODS}, not {start_method!r}")
        self.python_executable = python_executable or sys.executable
        # Given the entrypoint's module before platoon 0.5.0 (a shim until 0.6.0), run the entrypoint, as by default.
        self.module = WORKER_MODULE if module in _ENTRYPOINT_MODULES else module
        self.start_method = start_method
        # What the fork server imports and builds beyond the worker's stack, each when it is fork-safe: what the run's
        # environments import and build when a rollout opens them (platoon.train.startup).
        self.preload = preload or WorkerPreload()
        # What an earlier fork server left out of that (each {"item": ..., "reason": ...}): a later one leaves it out
        # at once, instead of trying it and starting over; and the items already warned about.
        self._fork_server_skipped: list[dict[str, str]] = []
        self._fork_server_warned: set[str] = set()
        self._fork_server: ForkServer | None = None
        self._fork_server_lock: tuple[asyncio.AbstractEventLoop, asyncio.Lock] | None = None
        self._fork_server_starts = 0
        self._fork_server_log = ""
        # What this process logged of the current fork server: its imports outlasted their deadline; it serves.
        self._fork_server_late = False
        self._fork_server_announced = False
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
        self.binding_path = binding_path(self.scratch_dir)
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
            if is_binding_file(name) or name == REGISTRY_DIRNAME:  # every policy's binding, which workers read
                continue
            path = os.path.join(self.scratch_dir, name)
            try:
                if os.stat(path).st_mtime < cutoff:
                    unlink_quietly(path)
            except OSError:
                continue

    # -- the model binding ----------------------------------------------
    def publish_binding(self, sampling_client, version: int, *, policy: str | None = None) -> None:
        """Republish the trainer's sampling client so workers can follow it: the agents' default policy's, or the one
        of ``policy``, another policy the rollouts sample, into its own binding file."""

        path = self.binding_path if policy is None else binding_path(self.scratch_dir, policy)
        publish_binding(sampling_client, version, path)

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
        """What the server imports before anything else, which it needs to serve: the worker's stack and the
        environments' packages."""

        if self.module != WORKER_MODULE:
            return []
        return [*WORKER_PRELOAD, *request.packages]

    def _fork_server_optional(self, request: RolloutWorkerRequest) -> list[str]:
        """What the server imports next, each module only when it is fork-safe: what the run's environments and
        harnesses import when a rollout opens them (:attr:`preload`)."""

        if self.module != WORKER_MODULE:
            return []
        return list(self.preload.modules)

    def _fork_server_child_init(self) -> list[str]:
        return (
            [WORKER_PACKAGE_INIT] if self.module.startswith(("platoon.train.workers.", "platoon.train.tinker.")) else []
        )

    def _fork_server_prepare(self, request: RolloutWorkerRequest) -> list[dict]:
        """What the server builds once, each only when it is fork-safe, and each worker it forks takes its own copy of
        instead of building it: the run's tokenizer and renderer (a few seconds of CPU each,
        :func:`platoon.train.tinker.proxy.prepare_renderer`), then what the run's environments build when a rollout
        opens them (:attr:`preload`)."""

        if self.module != WORKER_MODULE:
            return []
        steps = [dict(step) for step in self.preload.prepare]
        if not request.model.model_name:
            return steps
        # Each sampled policy's: the agents' default's (model), and the others' of a rollout of several policies.
        bindings = [request.model, *getattr(request, "policies", {}).values()]
        renderers = [
            {
                "ref": "platoon.train.tinker.proxy:prepare_renderer",
                "kwargs": {
                    "model_name": binding.model_name,
                    "renderer_name": binding.renderer_name,
                    "renderer_kwargs": dict(binding.renderer_kwargs),
                },
            }
            for binding in bindings
        ]
        return [*renderers, *steps]

    @property
    def _max_fork_server_starts(self) -> int:
        """The most fork servers it starts: :data:`MAX_FORK_SERVER_STARTS`, read each time the fork server's functions
        compare ``_fork_server_starts`` with it."""

        return MAX_FORK_SERVER_STARTS

    # Which fork server forks a worker; starting, replacing, warming, closing it (platoon.train.workers.fork_servers).
    _fork_server_for = fork_servers._fork_server_for
    _launch_fork_server = fork_servers._launch_fork_server
    warm = fork_servers.warm
    _use = fork_servers._use
    _warn_exec = fork_servers._warn_exec
    close = fork_servers.close

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
        on_first_call: Callable[[], None] | None = None,
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

        ``on_first_call`` is called once the worker has marked its rollout's first model call, for a request that asks
        it to (``signal_first_call``).
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
                    on_first_call=on_first_call,
                )
            except asyncio.CancelledError:
                _remove_worker_files(attempt_request)  # its worker is reaped: a cancelled rollout leaves no file either
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
        on_first_call: Callable[[], None] | None = None,
    ) -> WorkerOutcome:
        """One attempt at running ``request`` in a worker: launch it, track it, await its exit, reap it whatever ends
        the wait, then classify what it left.  Raises :class:`_LaunchFailed` when the worker did not start."""

        worker = await self._launch(request, log_dir)  # if no worker starts, it cleans up and raises
        self._track(worker, on_launch=on_launch, on_ready=on_ready, on_first_call=on_first_call)  # never raises
        try:
            timed_out = await self._await_exit(worker)
        finally:
            # Reap the process TREE on every exit path: normal, timeout, and
            # CancelledError (a straggler cancellation kills the worker).
            await self._reap(worker, on_reap)
        return self._finish(worker, timed_out=timed_out, retried=retried)

    async def _launch(self, request: RolloutWorkerRequest, log_dir: str | None) -> _LaunchedWorker:
        """Write the request, then start its worker: forked from the fork server when one can fork it
        (:meth:`_fork_server_for`), else by exec.  Raises :class:`_LaunchFailed`, having closed the log and removed the
        request file, when the memory floor blocks the launch or the worker did not start."""

        started = time.monotonic()
        request_path = os.path.join(self.scratch_dir, f"{request.request_id}.req")
        log_path = self._log_path(request, log_dir)
        log_handle = None
        proc: asyncio.subprocess.Process | ForkedWorker | None = None

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
        return _LaunchedWorker(
            request=request,
            started=started,
            request_path=request_path,
            log_path=log_path,
            log_handle=log_handle,
            proc=proc,
        )

    def _track(
        self,
        worker: _LaunchedWorker,
        *,
        on_launch: Callable[[int], None] | None,
        on_ready: Callable[[], None] | None,
        on_first_call: Callable[[], None] | None,
    ) -> None:
        """Track the launched worker: its pidfd and process group, which signal it; its record among this process's
        live workers (:meth:`_remember`) and ``on_launch``; then the wait for its exit and the watch for its readiness
        and first call (:meth:`_watch_ready`), which :meth:`_reap` ends."""

        proc = worker.proc
        if isinstance(proc, ForkedWorker):
            worker.pidfd = proc.pidfd  # the fork server opened it before the pid could be reused
        else:
            try:
                worker.pidfd = os.pidfd_open(proc.pid) if hasattr(os, "pidfd_open") else None
            except OSError:
                worker.pidfd = None
        try:
            worker.pgid = os.getpgid(proc.pid)
        except OSError:
            worker.pgid = proc.pid
        self._remember(proc.pid, worker.pidfd, worker.pgid, worker.request)
        if on_launch is not None:
            with contextlib.suppress(Exception):
                on_launch(proc.pid)

        worker.waiter = asyncio.ensure_future(proc.wait())
        worker.ready_watcher = asyncio.ensure_future(
            self._watch_ready(
                worker.request,
                proc,
                worker.started,
                worker.ready_state,
                on_ready=on_ready,
                on_first_call=on_first_call,
                waiter=worker.waiter,
            )
        )

    async def _await_exit(self, worker: _LaunchedWorker) -> bool:
        """Wait for the worker's exit, for its rollout budget and the parent's grace at most; whether that deadline
        passed first."""

        deadline = worker.request.rollout_budget_seconds + self.parent_grace_seconds
        try:
            await asyncio.wait_for(asyncio.shield(worker.waiter), timeout=deadline)
        except asyncio.TimeoutError:
            logger.warning(
                "rollout worker %s exceeded the parent deadline of %.0fs; reaping its process group",
                worker.request.request_id,
                deadline,
            )
            return True
        return False

    async def _reap(self, worker: _LaunchedWorker, on_reap: Callable[[int], None] | None) -> None:
        """Reap the worker's process tree and what it left: a worker still running gets SIGTERM, its termination grace,
        then SIGKILL; it is forgotten, its readiness watch ends, its pidfd, log and request file are closed and
        removed, the env sessions and sandboxes it left open are deleted; then ``on_reap``.  :meth:`_run_once` runs it
        in a ``finally``, so it runs on every exit path, cancellation included."""

        proc, waiter, pidfd = worker.proc, worker.waiter, worker.pidfd
        # Running until its wait ends (a forked worker whose fork server died exits with no return code).
        if not waiter.done():
            worker.killed = True
            self._signal_tree(pidfd, proc.pid, worker.pgid, signal.SIGTERM)
            if self.termination_grace_seconds > 0:
                with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.shield(waiter), self.termination_grace_seconds)
        if not waiter.done():
            self._signal_tree(pidfd, proc.pid, worker.pgid, signal.SIGKILL)
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(waiter), 30.0)
        # Forgotten before its pidfd closes, so signal_live_workers never signals a closed (or reused) fd.
        self._forget(proc.pid, reaped=waiter.done())
        ready_watcher = worker.ready_watcher
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
        if worker.log_handle:
            worker.log_handle.close()
        unlink_quietly(worker.request_path)
        # A worker that died before closing its remote env sessions left them listed: delete them now
        # rather than leave them to the env server's 15-minute reaper.
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.shield(self._delete_orphaned_sessions(worker.request))
        # And the sandboxes it did not stop (Harbor task sandboxes, verifier sandboxes): reaped by its label.
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.shield(self._reap_orphaned_sandboxes(worker.request))
        if on_reap is not None:
            with contextlib.suppress(Exception):
                on_reap(proc.pid)

    def _finish(self, worker: _LaunchedWorker, *, timed_out: bool, retried: bool) -> WorkerOutcome:
        """Classify the reaped worker (:func:`~platoon.train.workers.outcomes.classify`, which reads its response),
        then remove the files it wrote beside its request, and the hidden event files of one that left no response;
        returns the outcome."""

        launch_seconds = worker.ready_state["seconds"]
        outcome = classify(
            worker.request,
            returncode=worker.proc.returncode,
            killed_by_parent=worker.killed,
            timed_out=timed_out,
            launch_seconds=launch_seconds,
            wall_seconds=time.monotonic() - worker.started,
            retried=retried,
            worker_pid=worker.proc.pid,
            log_path=worker.log_path,
        )
        _remove_worker_files(worker.request)
        if outcome.response is None:
            _remove_event_temporaries(worker.request, worker.proc.pid)
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
        on_first_call: Callable[[], None] | None = None,
        waiter: asyncio.Future | None = None,
    ) -> None:
        """Fire once the child has finished importing and registering.

        ``create_subprocess_exec`` returns in well under a millisecond, so
        gating group admission on exec would bound nothing; the child then
        spends 20-30 s importing.  Readiness is the only useful gate.

        With ``on_first_call``, then watch for the marker of the rollout's first
        model call (``request.first_call_path``) and fire it once the marker is
        there, until ``waiter``, the worker's wait, ends (not its return code,
        which a forked worker whose fork server died never gets), looking once
        more then: a call made in the last poll interval still counts.  Nothing
        else ends the watch, so a rollout that calls no model through the
        gateway, or whose start slot came back on its timer, is watched to its
        end, at one ``stat`` a poll.
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
        if on_first_call is None or waiter is None:
            return
        while not os.path.exists(request.first_call_path) and not waiter.done():
            await asyncio.sleep(self.ready_poll_seconds)
        if os.path.exists(request.first_call_path):
            with contextlib.suppress(Exception):
                on_first_call()


class _LaunchFailed(RuntimeError):
    """Nothing user-visible ran, so the attempt is safe to repeat."""
