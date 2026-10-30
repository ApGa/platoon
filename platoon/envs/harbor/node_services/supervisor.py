"""The supervisor of one node service on this node, ``python -m platoon.envs.harbor.node_services --spec FILE
--parent PID``: ``platoon train`` and ``platoon eval`` start one per declared node service before any rollout
(:mod:`platoon.envs.harbor.node_services.runner`), in a session of its own, and it dies with them.

**One per node.**  It owns the service once it holds the service's lock on the node (``lock`` in the service's
directory, :func:`~platoon.envs.harbor.node_services.spec.node_dir`): another supervisor of the same service (another
run's, on the same node) waits for the lock while the owner's service serves both runs, and takes over when the owner
ends, starting the service again.  Every supervisor, waiting or owning, holds the service's presence lock
(:data:`~platoon.envs.harbor.node_services.protocol.PRESENCE`) shared for its life, so that a scope that cannot open
while the owner changes waits for the next one rather than fail (:func:`~platoon.envs.harbor.node_services.client
.open_scope`).  The owner:

1. stages the staged binds (:mod:`platoon.envs.harbor.node_services.staging`); its CPUs were pinned as it started
   (:func:`pin_cpus`), so the sandbox and what runs in it are too;
2. starts the service (:class:`~platoon.envs.harbor.node_services.instance.ServiceInstance`) in a sandbox of the
   declared provider, with this process's cleanup labels, the run's (``platoon sandbox reap`` of the run reaps it),
   and a mirror of what it holds (``sandboxes-<pid>.json``), which the run reaps when the supervisor did not stop it;
   then waits for its health check to pass, within its start period.  Each start is a new **generation**, counted on
   the node (``generation`` in the service's directory), across owners;
3. serves its control socket (:mod:`platoon.envs.harbor.node_services.protocol`) and its front
   (:mod:`platoon.envs.harbor.node_services.front`), a scope per rollout;
4. checks the service every ``interval``: ``retries`` failures in a row end that start, which it stops and starts
   again.  A start whose command exited ends at once, also when a rollout's probe finds it so; a probe never ends a
   start on a failed health check, which is the interval's to judge.  A start that ends makes down the scopes that may
   have had a wrong answer of it (:meth:`~platoon.envs.harbor.node_services.front.Front.start_failed`), whose rollouts
   are ``invalid``; the others go on with the next start, their requests waiting for it.  A start that served less
   than :data:`STABLE_S` before it failed is a failed start too, and each start after failed ones waits longer (5 s
   more each time, at most a minute); after ``restarts`` failed starts in a row it gives up (``failed``), and every
   scope opened afterwards is refused;
5. stops when its run tells it to (:data:`STOP_SIGNAL`, which :meth:`~platoon.envs.harbor.node_services.runner
   .NodeServiceRunner.stop` sends) or its run is gone: the kernel sends it SIGTERM then, and it looks for its parent
   every second too.  A SIGTERM while its run lives (a scheduler's to every process of the job, as its run gets one
   to stop at the next step boundary) does not stop it, nor does SIGINT: the run stops it once its last rollouts are
   graded.  Stopping, it answers opens with "try again" (the next owner's), lets the scopes' requests in flight
   finish, stops the service, and only then closes its control socket.
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import fcntl
import logging
import os
import secrets
import signal
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from platoon.envs.harbor.node_services.front import SCOPES_DIR, Front, Scope
from platoon.envs.harbor.node_services.instance import NODE_SERVICE_LABEL, ServiceInstance
from platoon.envs.harbor.node_services.protocol import (
    CONTROL_SOCKET,
    MAX_LINE_BYTES,
    PRESENCE,
    ProtocolError,
    receive,
    send,
)
from platoon.envs.harbor.node_services.spec import SERVICE_SOCKET, NodeServiceError, NodeServiceSpec, node_dir
from platoon.envs.harbor.node_services.staging import StagedCopy, stage
from platoon.sandbox import cleanup, get_provider
from platoon.sandbox.base import Mount, SandboxProvider
from platoon.utils.scratch import private_dir

logger = logging.getLogger(__name__)

State = Literal["starting", "up", "down", "failed", "stopping"]
#: What a supervisor keeps of what it starts, for a reap when it could not stop it (``{pid}`` is its own).
MIRROR = "sandboxes-{pid}.json"
#: The directory of the service's own socket, bound into its sandbox.
RUN_DIR = "run"
#: The file that counts the service's starts on the node, across its owners.
GENERATION = "generation"
#: The signal that stops a supervisor: its run's own stop.
STOP_SIGNAL = signal.SIGUSR1
#: A start that served at least this long before it failed is not a failed start: the failures in a row count again.
STABLE_S = 300.0
_LOCK_POLL_S = 0.2
_PARENT_POLL_S = 1.0
_PR_SET_PDEATHSIG = 1


class _TryAgain(Exception):
    """An open that the next owner of the service may answer: this supervisor is stopping."""


class Supervisor:
    """The supervisor of ``spec``'s service (the module docstring says what it does), whose parent is ``parent_pid``
    and whose sandboxes carry ``labels``, pinned to ``cpus`` CPUs."""

    def __init__(self, spec: NodeServiceSpec, *, parent_pid: int | None, labels: Mapping[str, str], cpus: int) -> None:
        self.spec = spec
        self.parent_pid = parent_pid
        self.labels = dict(labels)
        self.cpus = cpus
        self.directory = node_dir(spec)
        self.state: State = "starting"
        self.detail = ""
        self.generation = 0
        self.instance: ServiceInstance | None = None
        self._token = secrets.token_urlsafe(32)
        self._stopping = asyncio.Event()
        self._changed = asyncio.Condition()
        self._ended = asyncio.Event()  # the current start ended before its watch saw it (a probe found it so)
        self._passed = float("-inf")  # when the current start last passed its health check (monotonic)
        self._front: Front | None = None
        self._connections: set[asyncio.StreamWriter] = set()
        self._staged: list[StagedCopy] = []

    # The life of the supervisor ---------------------------------------------------------------------------------

    async def run(self) -> int:
        """Wait for the service's lock, own the service until told to stop; the exit status."""

        loop = asyncio.get_running_loop()
        loop.add_signal_handler(STOP_SIGNAL, self._stop)
        loop.add_signal_handler(signal.SIGTERM, self._terminated)
        loop.add_signal_handler(signal.SIGINT, self._interrupted)
        watcher = asyncio.ensure_future(self._watch_parent())
        private_dir(self.directory.parent)
        private_dir(self.directory)
        try:
            with open(self.directory / PRESENCE, "a") as present, open(self.directory / "lock", "a") as lock:
                fcntl.flock(present, fcntl.LOCK_SH)  # for its life: a supervisor of the service is on the node
                if not await self._take(lock):
                    return 0
                await self._own()
            return 0 if self.state != "failed" else 1
        finally:
            watcher.cancel()

    def _stop(self) -> None:
        logger.info("The run stops the node service %s", self.spec.name)
        self._stopping.set()

    def _terminated(self) -> None:
        if self._parent_gone():
            logger.info("SIGTERM, and the run of the node service %s is gone: stopping", self.spec.name)
            self._stopping.set()
        else:
            logger.info(
                "SIGTERM while the run %s lives: the node service %s serves on until the run stops it",
                self.parent_pid,
                self.spec.name,
            )

    def _interrupted(self) -> None:
        logger.info("SIGINT: the node service %s serves on until its run stops it", self.spec.name)

    def _parent_gone(self) -> bool:
        return self.parent_pid is None or os.getppid() != self.parent_pid

    async def _take(self, lock: Any) -> bool:
        """Take the service's lock on the node, waiting while another supervisor holds it; False when told to stop
        first."""

        waiting = False
        while not self._stopping.is_set():
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                if not waiting:
                    logger.info(
                        "Another supervisor serves the node service %s here: waiting to take over", self.spec.name
                    )
                    waiting = True
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), timeout=_LOCK_POLL_S)
        return False

    async def _own(self) -> None:
        """Serve the service until told to stop (the module docstring's steps 1-5)."""

        self._clear()
        self.generation = _read_generation(self.directory)
        cleanup.set_labels({**self.labels, NODE_SERVICE_LABEL: self.spec.name})
        cleanup.mirror_to(str(self.directory / MIRROR.format(pid=os.getpid())))
        server = None
        keeper = None
        try:
            mounts = await asyncio.to_thread(self._mounts)
            provider = get_provider(self.spec.provider, **self.spec.provider_options)
            self._front = Front(
                self.spec,
                self.directory,
                backend=self.directory / RUN_DIR / SERVICE_SOCKET,
                token=self._token,
                serving=self.serving,
            )
            await self._front.start()
            server = await asyncio.start_unix_server(
                self._client, path=str(self.directory / CONTROL_SOCKET), limit=MAX_LINE_BYTES
            )
            keeper = asyncio.ensure_future(self._keep(provider, mounts))
            await self._stopping.wait()
        except Exception as exc:
            logger.exception("The node service %s cannot be served", self.spec.name)
            await self._set("failed", f"its supervisor failed: {type(exc).__name__}: {exc}")
        finally:
            if self.state != "failed":  # opens wait for the next owner; the requests in flight finish
                await self._set("stopping", "its supervisor is stopping")
            if keeper is not None:
                keeper.cancel()
                await asyncio.wait([keeper])
            if self._front is not None:
                await self._front.close()
            if self.instance is not None:
                await self.instance.stop()
            if server is not None:
                server.close()
            for writer in list(self._connections):
                writer.close()
            for staged in self._staged:
                staged.release()
            self._clear()
            logger.info("The node service %s stopped", self.spec.name)

    def _mounts(self) -> list[Mount]:
        """The declared binds as the sandbox mounts them, a staged one from its copy on the node (made now, and kept
        while this supervisor lives)."""

        mounts = []
        for bind in self.spec.binds:
            source = bind.source
            if bind.stage:
                staged = stage(bind.source)
                self._staged.append(staged)
                source = str(staged.path)
            mounts.append(Mount(source=source, target=bind.target, read_only=bind.read_only))
        return mounts

    def _clear(self) -> None:
        """Remove the sockets an owner before left (one that died left them)."""

        (self.directory / RUN_DIR).mkdir(mode=0o700, exist_ok=True)
        (self.directory / SCOPES_DIR).mkdir(mode=0o700, exist_ok=True)
        for path in (self.directory / CONTROL_SOCKET, *(self.directory / SCOPES_DIR).glob("*.sock")):
            path.unlink(missing_ok=True)

    async def _watch_parent(self) -> None:
        if self.parent_pid is None:
            return
        while os.getppid() == self.parent_pid:
            await asyncio.sleep(_PARENT_POLL_S)
        logger.info("The parent of the node service %s is gone: stopping", self.spec.name)
        self._stopping.set()

    # The service's starts ----------------------------------------------------------------------------------------

    async def _keep(self, provider: SandboxProvider, mounts: list[Mount]) -> None:
        """Start the service, watch it, and start it again when it fails, until ``restarts`` starts in a row fail
        (the module docstring's step 4)."""

        failures = 0
        while True:
            if failures > 1:
                await asyncio.sleep(min(60.0, 5.0 * (failures - 1)))
            self.generation += 1
            _write_generation(self.directory, self.generation)
            await self._set("starting", f"start {self.generation}")
            instance = ServiceInstance(
                self.spec, run_dir=self.directory / RUN_DIR, token=self._token, mounts=mounts, cpus=self.cpus
            )
            self.instance = instance
            try:
                await instance.start(provider)
                await self._wait_healthy(instance)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failures += 1
                detail = f"{exc}{_log_tail(await instance.logs())}"
                await instance.stop()
                self.instance = None
                if failures >= self.spec.restarts:
                    await self._set("failed", f"it failed to start {failures} times in a row: {detail}")
                    return
                await self._set("down", detail)
                logger.warning("The node service %s did not start: %s; starting it again", self.spec.name, detail)
                continue
            self._ended.clear()
            await self._set("up", "")
            began = time.monotonic()
            logger.info("The node service %s is up (start %d)", self.spec.name, self.generation)
            detail = await self._watch(instance)
            await self._end_start(self.generation, detail)
            failures = 1 if time.monotonic() - began >= STABLE_S else failures + 1
            await instance.stop()
            self.instance = None
            if failures >= self.spec.restarts:
                message = f"it failed {failures} times in a row, each within {STABLE_S:g} s of its start: {detail}"
                await self._set("failed", message)
                logger.error("The node service %s gives up: %s", self.spec.name, message)
                return
            logger.warning("The node service %s is down: %s; starting it again", self.spec.name, detail)

    async def _wait_healthy(self, instance: ServiceInstance) -> None:
        """Wait for the start's health check to pass, by Docker's rules (failures in the start period do not count);
        ``NodeServiceError`` when its command exits, or ``retries`` checks fail in a row after the start period."""

        health = self.spec.health
        began = time.monotonic()
        failures = 0
        while True:
            checked = time.monotonic()
            passed, detail = await instance.check()
            if passed:
                self._passed = checked
                return
            if not await instance.running():
                raise NodeServiceError(f"its command exited before its health check passed: {detail}")
            starting = time.monotonic() - began < health.start_period_s
            if not starting:
                failures += 1
                if failures >= health.retries:
                    raise NodeServiceError(f"it never turned healthy: {detail}")
            await asyncio.sleep(min(health.interval_s, 1.0) if starting else health.interval_s)

    async def _watch(self, instance: ServiceInstance) -> str:
        """Check the current start every ``interval`` until ``retries`` checks fail in a row, or it ended otherwise
        (a probe found its command gone); what failed."""

        health = self.spec.health
        failures = 0
        detail = ""
        while True:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._ended.wait(), timeout=health.interval_s)
            if self._ended.is_set():
                return self.detail
            checked = time.monotonic()
            passed, detail = await instance.check()
            if passed:
                failures = 0
                self._passed = checked
                continue
            if not await instance.running():
                return detail
            failures += 1
            if failures >= health.retries:
                return f"its health check failed {failures} times in a row: {detail}"

    async def _end_start(self, generation: int, detail: str) -> None:
        """End the start ``generation`` of the service, once: it serves no more, and the scopes that may have had a
        wrong answer of it are down (:meth:`Front.start_failed`)."""

        if generation != self.generation or self.state != "up":
            return
        if self._front is not None:
            down = self._front.start_failed(
                f"the start {generation} of the node service {self.spec.name} failed while this rollout used it: "
                f"{detail}",
                since=self._passed,
            )
            if down:
                logger.warning("%d scopes of the node service %s used the start that failed", down, self.spec.name)
        await self._set("down", detail)
        self._ended.set()

    async def serving(self, wait_s: float) -> int | None:
        """The start of the service that serves, once one does (waiting up to ``wait_s`` while it starts again); None
        when none will: it failed for good, its supervisor is stopping, or no start served in time.  The front asks
        before each request."""

        if self.state != "up":
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._settled(), timeout=max(0.0, wait_s))
        return self.generation if self.state == "up" else None

    async def _settled(self) -> None:
        async with self._changed:
            await self._changed.wait_for(lambda: self.state in ("up", "failed", "stopping"))

    async def probe(self, scope: Scope) -> tuple[bool, str]:
        """Whether ``scope``'s rollout may trust what the service answered it: not once the scope is down.  A probe
        also checks that the command of the start that serves still runs, and ends the start when it does not (its
        scopes that used it are down then); a failed health check is the watch's to judge, never a probe's."""

        if scope.down is None and self.state == "up" and (instance := self.instance) is not None:
            generation = self.generation
            if not await instance.running():
                await self._end_start(generation, "its command exited")
        if scope.down is not None:
            return False, scope.down
        return True, ""

    async def _set(self, state: State, detail: str) -> None:
        async with self._changed:
            self.state, self.detail = state, detail
            self._changed.notify_all()

    async def _wait_up(self, wait_s: float) -> None:
        """Wait until the service is up, at most ``wait_s``; ``NodeServiceError`` once it failed for good, or still
        is not up then, and :class:`_TryAgain` once this supervisor is stopping."""

        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._settled(), timeout=max(0.0, wait_s))
        if self.state == "stopping":
            raise _TryAgain(f"the supervisor of the node service {self.spec.name} is stopping")
        if self.state != "up":
            raise NodeServiceError(f"the node service {self.spec.name} is {self.state}: {self.detail}")

    # The control socket ------------------------------------------------------------------------------------------

    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Serve one connection of the control protocol; the scope it opened closes with it."""

        scope: Scope | None = None
        self._connections.add(writer)
        try:
            while True:
                try:
                    message = await receive(reader)
                except ProtocolError:
                    return
                op = message.get("op")
                if op == "close":  # the scope goes before the answer: a closed scope's socket is gone
                    if scope is not None and self._front is not None:
                        await self._front.close_scope(scope)
                    await send(writer, {"ok": True})
                    return
                answer: dict[str, Any]
                try:
                    if op == "ping":
                        answer = {
                            "ok": True,
                            "state": self.state,
                            "generation": self.generation,
                            "detail": self.detail,
                        }
                    elif op == "open" and scope is None:
                        scope = await self._open(message)
                        answer = {"ok": True, "socket": str(scope.socket), "generation": scope.generation}
                    elif op == "status" and scope is not None:
                        up, detail = await self.probe(scope)
                        answer = {"ok": True, "up": up, "detail": detail, "served": sorted(scope.served)}
                    else:
                        answer = {"ok": False, "error": f"cannot answer {op!r} here"}
                except _TryAgain as exc:
                    answer = {"ok": False, "retry": True, "error": str(exc)}
                except NodeServiceError as exc:
                    answer = {"ok": False, "error": str(exc)}
                await send(writer, answer)
        except (ConnectionError, OSError):
            return
        finally:
            self._connections.discard(writer)
            if scope is not None and self._front is not None:
                await self._front.close_scope(scope)
            writer.close()

    async def _open(self, message: Mapping[str, Any]) -> Scope:
        values = message.get("scope") or {}
        if not isinstance(values, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in values.items()
        ):
            raise NodeServiceError("a scope's values must map names to strings")
        wait_s = message.get("wait_s", 0.0)
        await self._wait_up(float(wait_s) if isinstance(wait_s, int | float) else 0.0)
        assert self._front is not None
        return await self._front.open(values, generation=self.generation)


def pin_cpus(wanted: int | str | list[int] | None) -> int:
    """Pin this process, and what it starts from now on, to the CPUs ``wanted``: a list, those (as its run reserved
    them, :mod:`platoon.envs.harbor.node_services.runner`); a number, the first ones it may use; a list as text such as
    ``0-15,32``, those.  The number it runs on.  ``NodeServiceError`` for CPUs it may not use."""

    allowed = sorted(os.sched_getaffinity(0))
    if wanted is None:
        return len(allowed)
    if isinstance(wanted, list):
        chosen = sorted(set(wanted))
    elif isinstance(wanted, int):
        if wanted > len(allowed):
            logger.warning(
                "A node service wants %d CPUs, and this process may use %d: it gets them all", wanted, len(allowed)
            )
        chosen = allowed[:wanted]
    else:
        chosen = cpu_list(wanted)
        if missing := sorted(set(chosen) - set(allowed)):
            raise NodeServiceError(f"the CPUs {missing} are not among those this process may use ({len(allowed)})")
    try:
        os.sched_setaffinity(0, chosen)
    except OSError as exc:
        raise NodeServiceError(f"this process cannot run on the CPUs {chosen}: {exc}") from exc
    return len(chosen)


def cpu_list(text: str) -> list[int]:
    """The CPUs a list such as ``0-15,32`` names, sorted."""

    cpus: list[int] = []
    for part in text.split(","):
        first, _, last = part.partition("-")
        cpus += range(int(first), int(last or first) + 1)
    return sorted(set(cpus))


def die_with_parent(parent_pid: int | None) -> None:
    """Ask the kernel for SIGTERM when this process's parent ends (Linux), and exit now if it ended already."""

    if parent_pid is None:
        return
    with contextlib.suppress(OSError, AttributeError):
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    if os.getppid() != parent_pid:
        raise SystemExit(0)


def _read_generation(directory: Path) -> int:
    """The last start of the service on this node (0 for none)."""

    try:
        return int((directory / GENERATION).read_text().strip() or 0)
    except (OSError, ValueError):
        return 0


def _write_generation(directory: Path, generation: int) -> None:
    with contextlib.suppress(OSError):
        (directory / GENERATION).write_text(f"{generation}\n")


def _log_tail(log: str) -> str:
    log = log.strip()
    return f"; its log ends: {log[-1000:]}" if log else ""


__all__ = [
    "GENERATION",
    "MIRROR",
    "RUN_DIR",
    "STABLE_S",
    "STOP_SIGNAL",
    "Supervisor",
    "cpu_list",
    "die_with_parent",
    "pin_cpus",
]
