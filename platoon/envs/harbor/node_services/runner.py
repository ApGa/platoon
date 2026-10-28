"""The node services of a run (:class:`NodeServiceRunner`): ``platoon train`` and ``platoon eval`` start a supervisor
for each node service their environments declare (:func:`node_services_of`) before any rollout, wait until each one
serves, watch them while the run lasts, and stop them when the run ends; a supervisor also dies with the run's process
however it ends.

**Declarations** (:func:`declared_node_services`): an environment's ``service_kwargs.node_services``, by name, each a
declaration (:class:`~platoon.envs.harbor.node_services.spec.NodeServiceSpec`) or ``site``, the site profile's
declaration of that name; each runs on the environment's sandbox provider, with its ``provider_options`` under the
declaration's own.  Two environments, or two runs on a node, that declare the same service share one supervisor's
service (:mod:`platoon.envs.harbor.node_services.supervisor`).

**CPUs.**  A service that declares ``cpus`` gets CPUs of the run's own, which the run's rollouts then do not use
(:func:`plan_cpus`): a number takes whole cores (a CPU with its hyperthread siblings) from the first CPUs the run may
use, each service its own; a list (``"0-15"``) takes those CPUs.  Its CPUs and their siblings leave every thread of the
run's process, so the rollout workers it starts afterwards, and their sandboxes, run on the others; :meth:`~
NodeServiceRunner.stop` gives them back.  Services that would take every CPU of the run share them with its rollouts
(a warning says so).

**Starting.**  :meth:`NodeServiceRunner.start` starts the supervisors (``python -m
platoon.envs.harbor.node_services``, in sessions of their own, with this process's cleanup labels, the run's), each
logging to ``supervisor-<pid>.log`` in its service's directory on the node; :meth:`~NodeServiceRunner.ready` waits
until each service is up, and raises :class:`~platoon.envs.harbor.node_services.spec.NodeServiceError` with the end of
its log when one fails to start, so that a run whose rollouts could not be graded stops before its first one.  The run
holds each service's presence lock (:data:`~platoon.envs.harbor.node_services.protocol.PRESENCE`) from then on, until
it stops or its node services failed, so that its rollouts' scopes wait for a supervisor that is not listening yet.

**Watching.**  Once they are up, the runner looks at its supervisors every :data:`WATCH_S`: one that died has what it
left reaped (its mirror, ``sandboxes-<pid>.json``) and another started in its place; a service that failed for good
(``failed``: it could not be started again), or a supervisor that died :data:`SUPERVISOR_DEATHS` times, fails the
run, which hears of it through :meth:`~NodeServiceRunner.on_failure` (``platoon train`` stops at the next step
boundary and fails, ``platoon eval`` starts no further rollout and fails).

**Stopping.**  :meth:`NodeServiceRunner.stop`, which ``platoon train`` and ``platoon eval`` call as they end, before
they reap the run's sandboxes, sends each supervisor its stop signal, waits for it to stop its service, then reaps by
label what a supervisor that did not stop it in time left (its mirror); this process's exit does the same, without the
reap, for a run that ended otherwise.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import fcntl
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import IO, Any

from platoon.envs.harbor.node_services.client import NodeServiceUnavailable, ping
from platoon.envs.harbor.node_services.protocol import PRESENCE
from platoon.envs.harbor.node_services.spec import FROM_SITE, NodeServiceError, NodeServiceSpec, node_dir
from platoon.envs.harbor.node_services.supervisor import MIRROR, STOP_SIGNAL, cpu_list
from platoon.sandbox import cleanup
from platoon.utils.scratch import private_dir

logger = logging.getLogger(__name__)

#: How long :meth:`NodeServiceRunner.ready` waits for the services, at most, beyond their start periods (staging a
#: bind copies it once per node first).
READY_WAIT_S = 1800.0
#: How long a supervisor gets to stop its service before it is killed, and its sandbox reaped.
STOP_WAIT_S = 60.0
#: How often the runner looks at its supervisors once the services are up.
WATCH_S = 1.0
#: How many times a run's supervisor of a service may die before the run fails.
SUPERVISOR_DEATHS = 3
_POLL_S = 0.5
_DEFAULT_PROVIDER = "docker"  # the Harbor service's own default


def declared_node_services(
    raw: Any, *, provider: str, provider_options: Mapping[str, Any] | None = None
) -> dict[str, NodeServiceSpec]:
    """The node services an environment declares (``service_kwargs.node_services``), by name, run on ``provider``
    with ``provider_options``; ``site`` takes the site profile's declaration of a name.  :class:`NodeServiceError`
    says what is wrong."""

    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise NodeServiceError(f"node_services must map names to declarations, got {raw!r}")
    declared = {}
    for name, declaration in raw.items():
        if declaration == FROM_SITE:
            declaration = _site_declaration(str(name))
        declared[str(name)] = NodeServiceSpec.parse(
            str(name), declaration, provider=provider, provider_options=provider_options
        )
    return declared


def _site_declaration(name: str) -> Any:
    from platoon.site import site_profile

    profile = site_profile()
    declared = profile.node_services.get(name)
    if declared is None:
        where = profile.path or "no site profile"
        raise NodeServiceError(f"node_services.{name} is {FROM_SITE!r}, and the site profile ({where}) declares none")
    return declared


def node_services_of(environments: Iterable[Any]) -> list[NodeServiceSpec]:
    """The node services that ``environments`` (a run's entries, each with ``service_kwargs``) declare, each once."""

    found: dict[str, NodeServiceSpec] = {}
    for environment in environments:
        kwargs = getattr(environment, "service_kwargs", None) or {}
        if not kwargs.get("node_services"):
            continue
        provider = kwargs.get("provider") or _DEFAULT_PROVIDER
        if not isinstance(provider, str):
            raise NodeServiceError("node services need the environment's provider named (singularity, process, ...)")
        options = kwargs.get("provider_options") or {}
        declared = declared_node_services(kwargs["node_services"], provider=provider, provider_options=options)
        for spec in declared.values():
            found.setdefault(spec.key, spec)
    return list(found.values())


class NodeServiceRunner:
    """The supervisors of a run's node services (the module docstring says how they start, are watched and stop)."""

    def __init__(self, specs: Iterable[NodeServiceSpec]) -> None:
        self.specs = list(specs)
        #: Why the run's node services failed, once they did (the module docstring's watching); None while they serve.
        self.failure: str | None = None
        self._processes: dict[str, subprocess.Popen[bytes]] = {}  # the current supervisor of each, by service key
        self._started: list[tuple[NodeServiceSpec, subprocess.Popen[bytes]]] = []  # every supervisor it started
        self._logs: dict[str, Path] = {}
        self._configs: dict[str, Path] = {}
        self._deaths: dict[str, int] = {}
        self._cpus: dict[str, list[int]] = {}
        self._affinity: set[int] | None = None  # the run's CPUs before the services took theirs
        self._on_failure: list[Callable[[str], Any]] = []
        self._watcher: asyncio.Future[None] | None = None
        self._presence: list[IO[Any]] = []
        self._stopped = False

    async def start(self) -> None:
        """Take the services' CPUs from the run's, and start a supervisor for each service, with this process's
        cleanup labels; returns at once."""

        labels = cleanup.labels()
        self._cpus = self._reserve_cpus()
        for spec in self.specs:
            directory = node_dir(spec)
            private_dir(directory.parent)
            private_dir(directory)
            # The run is present on the node until it stops, or its node services failed: a scope that cannot open
            # while its supervisor starts, or is started again, waits for it (client.open_scope).
            presence = open(directory / PRESENCE, "a")  # noqa: SIM115 - held until the run stops
            fcntl.flock(presence, fcntl.LOCK_SH)
            self._presence.append(presence)
            config = directory / f"supervisor-{os.getpid()}.json"
            config.write_text(json.dumps({"spec": spec.to_json(), "labels": labels, "cpus": self._cpus.get(spec.key)}))
            self._configs[spec.key] = config
            self._spawn(spec)
        atexit.register(self._stop_now)

    def _spawn(self, spec: NodeServiceSpec) -> None:
        """Start a supervisor of ``spec``'s service, in a session of its own: a signal to the run's process group (a
        terminal's Ctrl+C) never reaches it, and the run stops it itself.  On this thread, the event loop's: the kernel
        stops the supervisor once the thread that started it ends."""

        log_path = node_dir(spec) / f"supervisor-{os.getpid()}.log"
        with open(log_path, "ab") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "platoon.envs.harbor.node_services", "--spec", str(self._configs[spec.key])]
                + ["--parent", str(os.getpid())],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self._processes[spec.key], self._logs[spec.key] = process, log_path
        self._started.append((spec, process))
        logger.info("Starting the node service %s (its supervisor logs to %s)", spec.name, log_path)

    def _reserve_cpus(self) -> dict[str, list[int]]:
        """The CPUs of each service that declares some, by key, taken from this process's threads (the module
        docstring's CPUs)."""

        wanted = [(spec.key, spec.name, spec.cpus) for spec in self.specs if spec.cpus is not None]
        if not wanted:
            return {}
        allowed = sorted(os.sched_getaffinity(0))
        plan, rest = plan_cpus(wanted, allowed, siblings=cpu_siblings)
        if not rest:
            logger.warning(
                "The node services take every CPU this run may use (%d): its rollouts share them", len(allowed)
            )
            return plan
        self._affinity = set(allowed)
        _set_affinity(set(rest))
        for spec in self.specs:
            if spec.key in plan:
                logger.info("The node service %s runs on the CPUs %s", spec.name, _cpu_text(plan[spec.key]))
        logger.info("This run's rollouts run on the other %d CPUs: %s", len(rest), _cpu_text(rest))
        return plan

    def on_failure(self, callback: Callable[[str], Any]) -> None:
        """Call ``callback(why)`` once the run's node services fail (the module docstring's watching); at once when
        they did."""

        self._on_failure.append(callback)
        if self.failure is not None:
            callback(self.failure)

    async def ready(self, *, stop: asyncio.Event | None = None, timeout_s: float | None = None) -> None:
        """Wait until every service is up; :class:`NodeServiceError` with the end of its supervisor's log when one
        failed to start, or is not up in time.  Returns early once ``stop`` is set.  Then the runner watches them."""

        for spec in self.specs:
            wait = timeout_s if timeout_s is not None else spec.health.start_period_s + READY_WAIT_S
            await self._ready(spec, wait, stop)
        if self._watcher is None and not self._stopped and (stop is None or not stop.is_set()):
            self._watcher = asyncio.ensure_future(self._watch())

    async def _ready(self, spec: NodeServiceSpec, wait_s: float, stop: asyncio.Event | None) -> None:
        deadline = time.monotonic() + wait_s
        while stop is None or not stop.is_set():
            process = self._processes[spec.key]
            try:
                state = await ping(spec)
            except NodeServiceUnavailable:
                state = {}
            if state.get("state") == "up":
                logger.info("The node service %s is up", spec.name)
                return
            if state.get("state") == "failed":
                raise NodeServiceError(f"the node service {spec.name} failed: {state.get('detail')}")
            if self.failure is not None:
                raise NodeServiceError(self.failure)
            if process.poll() is not None and not state and self._watcher is None:  # once watched, it is replaced
                raise NodeServiceError(
                    f"the supervisor of the node service {spec.name} exited with status {process.returncode}"
                    f"{self._log_tail(spec)}"
                )
            if time.monotonic() >= deadline:
                detail = state.get("detail") or "its supervisor does not answer"
                raise NodeServiceError(
                    f"the node service {spec.name} is not up after {wait_s:g} s ({detail}){self._log_tail(spec)}"
                )
            await asyncio.sleep(_POLL_S)

    async def _watch(self) -> None:
        """Look at the supervisors every :data:`WATCH_S` until the run stops them, or they failed."""

        while not self._stopped and self.failure is None:
            await asyncio.sleep(WATCH_S)
            for spec in self.specs:
                if self._stopped:
                    return
                if (problem := await self._looked_at(spec)) is not None:
                    self._fail(f"the node service {spec.name} failed: {problem}")
                    return

    async def _looked_at(self, spec: NodeServiceSpec) -> str | None:
        """Look at ``spec``'s supervisor: what fails the run, or None (one that died is replaced)."""

        process = self._processes[spec.key]
        if process.poll() is not None:
            deaths = self._deaths[spec.key] = self._deaths.get(spec.key, 0) + 1
            logger.warning(
                "The supervisor of the node service %s exited (status %s): reaping what it left%s",
                spec.name,
                process.returncode,
                self._log_tail(spec),
            )
            await self._reap(spec, process)
            if deaths >= SUPERVISOR_DEATHS:
                return f"its supervisor died {deaths} times"
            self._spawn(spec)
            return None
        try:
            state = await ping(spec)
        except NodeServiceUnavailable:  # starting, or another run's supervisor takes over
            return None
        if state.get("state") == "failed":
            return str(state.get("detail") or "it could not be started again")
        return None

    def _fail(self, why: str) -> None:
        if self.failure is not None:
            return
        self.failure = why
        self._leave()  # a scope that cannot open fails at once
        logger.error("The run cannot grade its rollouts any more: %s", why)
        for callback in self._on_failure:
            try:
                callback(why)
            except Exception:
                logger.exception("A callback of the node services' failure failed")

    async def _reap(self, spec: NodeServiceSpec, process: subprocess.Popen[bytes]) -> None:
        """Reap what ``process``, a supervisor of ``spec``'s that is gone, left (its mirror)."""

        records = cleanup.read_mirror(str(node_dir(spec) / MIRROR.format(pid=process.pid)))
        if records:
            await cleanup.reap_records(records, timeout_s=STOP_WAIT_S)

    async def stop(self) -> None:
        """Stop every supervisor, and reap what one left; give the run its CPUs back; idempotent."""

        if self._stopped:
            return
        self._stopped = True
        self._leave()
        if self._watcher is not None:
            self._watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watcher
        for process in self._processes.values():
            if process.poll() is None:
                process.send_signal(STOP_SIGNAL)
        deadline = time.monotonic() + STOP_WAIT_S
        while any(process.poll() is None for process in self._processes.values()) and time.monotonic() < deadline:
            await asyncio.sleep(_POLL_S)
        for spec, process in self._started:  # what each left, also one that died while the run lasted
            if process.poll() is None:
                logger.warning("The supervisor of the node service %s did not stop in time: killing it", spec.name)
                process.kill()
                await asyncio.to_thread(process.wait)
            await self._reap(spec, process)
        if self._affinity is not None:
            _set_affinity(self._affinity)
            self._affinity = None

    def _leave(self) -> None:
        """The run is no longer present on the node (it held the services' presence locks since it started them)."""

        presence, self._presence = self._presence, []
        for handle in presence:
            handle.close()

    def _stop_now(self) -> None:
        """At this process's exit: tell each supervisor to stop, and give it a little time to."""

        if self._stopped:
            return
        self._stopped = True
        for process in self._processes.values():
            with contextlib.suppress(OSError):
                if process.poll() is None:
                    process.send_signal(STOP_SIGNAL)
        deadline = time.monotonic() + 10.0
        for process in self._processes.values():
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=max(0.0, deadline - time.monotonic()))

    def _log_tail(self, spec: NodeServiceSpec) -> str:
        try:
            text = self._logs[spec.key].read_text(errors="replace").strip()
        except (KeyError, OSError):
            return ""
        return f"; its log ({self._logs[spec.key]}) ends: {text[-1500:]}" if text else ""


def plan_cpus(
    wanted: Sequence[tuple[str, str, int | str]],
    allowed: Sequence[int],
    *,
    siblings: Callable[[int], Iterable[int]],
) -> tuple[dict[str, list[int]], list[int]]:
    """The CPUs of each node service in ``wanted`` (its key, its name and its ``cpus``), by key, and the CPUs of
    ``allowed`` (the run's) that are left for the run, the module docstring's CPUs; ``siblings(cpu)`` gives a CPU's
    hyperthread siblings, itself among them.  :class:`NodeServiceError` for a list that names a CPU the run may not
    use, or one another service takes, and for a service that finds no CPU left."""

    allowed_set = set(allowed)
    taken: set[int] = set()
    plan: dict[str, list[int]] = {}

    def core(cpu: int) -> set[int]:
        return ({cpu} | set(siblings(cpu))) & allowed_set

    for key, name, cpus in wanted:
        if isinstance(cpus, int):
            chosen: set[int] = set()
            for cpu in allowed:
                if len(chosen) >= cpus:
                    break
                if cpu not in taken and cpu not in chosen:
                    chosen |= core(cpu) - taken
            if not chosen:
                raise NodeServiceError(f"the node service {name} wants {cpus} CPUs, and no CPU of the run is left")
            if len(chosen) < cpus:
                logger.warning(
                    "The node service %s wants %d CPUs, and %d of the run's are left: it gets them",
                    name,
                    cpus,
                    len(chosen),
                )
        else:
            chosen = set(cpu_list(cpus))
            if missing := sorted(chosen - allowed_set):
                raise NodeServiceError(
                    f"node_services.{name}.cpus names the CPUs {_cpu_text(missing)}, which this run may not use "
                    f"({_cpu_text(allowed)})"
                )
            if clash := sorted(chosen & taken):
                raise NodeServiceError(
                    f"node_services.{name}.cpus names the CPUs {_cpu_text(clash)}, which another node service takes"
                )
        plan[key] = sorted(chosen)
        for cpu in chosen:
            taken |= core(cpu)
    return plan, [cpu for cpu in allowed if cpu not in taken]


def cpu_siblings(cpu: int) -> list[int]:
    """The hyperthread siblings of ``cpu`` on this host, itself among them (itself alone where the host does not
    say)."""

    try:
        text = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text().strip()
        return cpu_list(text) or [cpu]
    except (OSError, ValueError):
        return [cpu]


def _set_affinity(cpus: set[int]) -> None:
    """Run every thread of this process on ``cpus``: what each starts afterwards runs there too."""

    for task in Path("/proc/self/task").iterdir():
        with contextlib.suppress(OSError, ValueError):
            os.sched_setaffinity(int(task.name), cpus)


def _cpu_text(cpus: Iterable[int]) -> str:
    """CPUs as a list such as ``0-15,32``."""

    ranges: list[list[int]] = []
    for cpu in sorted(set(cpus)):
        if ranges and cpu == ranges[-1][1] + 1:
            ranges[-1][1] = cpu
        else:
            ranges.append([cpu, cpu])
    return ",".join(f"{first}-{last}" if first != last else f"{first}" for first, last in ranges)


__all__ = [
    "READY_WAIT_S",
    "STOP_WAIT_S",
    "SUPERVISOR_DEATHS",
    "WATCH_S",
    "NodeServiceRunner",
    "cpu_siblings",
    "declared_node_services",
    "node_services_of",
    "plan_cpus",
]
