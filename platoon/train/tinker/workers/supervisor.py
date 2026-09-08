"""The parent side of process-per-rollout: launch, deadline, kill, classify.

One ``asyncio`` subprocess per rollout, no pool.  Measured on this node with
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

Signalling is pidfd-based
=========================

Python 3.12's default child watcher reaps each child on its own daemon thread,
so ``proc.returncode is None`` can be stale by the time a signal is sent.
Every rollout worker and every MCP bridge on this node is a session leader, so
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
import time
import uuid
from dataclasses import dataclass, replace
from typing import Callable

from platoon.train.tinker.worker_env import DIAG_PATH_ENV, STARTUP_BUDGET_ENV
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

logger = logging.getLogger(__name__)

__all__ = ["RolloutWorkerSupervisor", "WorkerOutcome"]

WORKER_MODULE = "platoon.train.tinker.workers.entrypoint"

#: Environment forced on every worker.  Deliberately inherit-plus-override
#: rather than an allowlist: the trainer's environment is already the curated
#: production environment (HF_HOME, offline flags, session URLs, TMPDIR,
#: gateway URL and key), and an allowlist would silently drop future keys.
WORKER_ENV_OVERRIDES = {
    "TINKER_TELEMETRY": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PLATOON_TINKER_ROLLOUT_WORKER": "1",
}

WORKER_ENV_REMOVED = ("TINKER_SUBPROCESS_SAMPLING",)


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
    ) -> None:
        self.python_executable = python_executable or sys.executable
        self.module = module
        self.scratch_dir = scratch_dir or ipc_dir()
        self.log_dir = log_dir
        # The launcher gives the whole client 30 s between SIGTERM and SIGKILL,
        # so a per-worker grace anywhere near that could not complete inside it
        # and the trainer would die mid-reap.  Workers are always reaped
        # concurrently, never serially.
        self.termination_grace_seconds = float(termination_grace_seconds)
        self.parent_grace_seconds = float(parent_grace_seconds)
        self.mem_available_floor_bytes = int(mem_available_floor_bytes)
        self.env_overrides = dict(env_overrides or {})
        self.launch_retries = int(launch_retries)
        self.ready_poll_seconds = float(ready_poll_seconds)
        self.binding_path = os.path.join(self.scratch_dir, "model_binding.pkl")
        self._warn_if_scratch_is_shared()
        self._sweep_stale_files()

    # -- setup -----------------------------------------------------------
    def _warn_if_scratch_is_shared(self) -> None:
        """Response files are 1-6 MB each; on Lustre this design degrades badly."""

        local_root = os.environ.get("PBS_LOCALDIR")
        if not local_root:
            logger.warning(
                "rollout worker scratch %s is not under $PBS_LOCALDIR (unset); rollout IPC files "
                "must live on node-local storage",
                self.scratch_dir,
            )
            return
        if not os.path.realpath(self.scratch_dir).startswith(os.path.realpath(local_root)):
            logger.warning(
                "rollout worker scratch %s is outside $PBS_LOCALDIR (%s); rollout IPC files should "
                "be node-local",
                self.scratch_dir,
                local_root,
            )

    def _sweep_stale_files(self) -> None:
        cutoff = time.time() - IPC_SWEEP_AGE_SECONDS
        try:
            names = os.listdir(self.scratch_dir)
        except OSError:
            return
        for name in names:
            if name == os.path.basename(self.binding_path):
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

    # -- signalling ------------------------------------------------------
    @staticmethod
    def _signal_process(pidfd: int | None, pid: int, sig: int) -> None:
        try:
            if pidfd is not None and hasattr(signal, "pidfd_send_signal"):
                signal.pidfd_send_signal(pidfd, sig)
            else:  # pragma: no cover - pidfd is available on this kernel
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
        keeps a rollout traceback readable after the split: today it would land
        in the trainer's log, and a killed worker would leave nothing at all.
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
        proc = None
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

        try:
            pidfd = os.pidfd_open(proc.pid) if hasattr(os, "pidfd_open") else None
        except OSError:
            pidfd = None
        try:
            pgid = os.getpgid(proc.pid)
        except OSError:
            pgid = proc.pid
        if on_launch is not None:
            with contextlib.suppress(Exception):
                on_launch(proc.pid)

        waiter = asyncio.ensure_future(proc.wait())
        ready_state = {"seconds": 0.0}
        ready_watcher = asyncio.ensure_future(
            self._watch_ready(request, proc, started, ready_state, on_ready)
        )

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
            # CancelledError (a straggler cancellation now becomes a real kill
            # for the first time).
            if proc.returncode is None:
                killed = True
                self._signal_tree(pidfd, proc.pid, pgid, signal.SIGTERM)
                if self.termination_grace_seconds > 0:
                    with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                        await asyncio.wait_for(
                            asyncio.shield(waiter), self.termination_grace_seconds
                        )
            if proc.returncode is None:
                self._signal_tree(pidfd, proc.pid, pgid, signal.SIGKILL)
                with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.shield(waiter), 30.0)
            if ready_watcher is not None:
                # Let it observe the exit first: readiness is a real number we
                # want even for a worker that finished quickly.
                if not ready_watcher.done():
                    with contextlib.suppress(BaseException):
                        await asyncio.wait_for(asyncio.shield(ready_watcher), 5.0)
                ready_watcher.cancel()
                with contextlib.suppress(BaseException):
                    await ready_watcher
            if pidfd is not None:
                with contextlib.suppress(OSError):
                    os.close(pidfd)
            if log_handle:
                log_handle.close()
            unlink_quietly(request_path)
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
        return outcome

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


class _LaunchFailed(RuntimeError):
    """Nothing user-visible ran, so the attempt is safe to repeat."""
