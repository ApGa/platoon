"""What a run readies before its first rollout: the toolchains its environments' harnesses run from, staged on this
node; its rollout workers' fork servers, started; and what those servers import and build ahead of the workers.

**Toolchains and fork servers.**  Both used to fall on a run's first rollouts.  A CLI harness's first rollout on a node
copied the harness's toolchain (tens of thousands of files) from the shared filesystem to node-local disk while every
other rollout of the node waited for that copy, at every start and every resume in a new job; and the first rollouts
of a run with rollout workers waited for the fork server to import the worker's stack, which it began only when they
asked for it.  ``platoon train`` now begins both as soon as its training client is open (:meth:`RolloutStartup.begin`),
so that they run while it saves its launch checkpoint, and waits for the staging before it starts any rollout,
training's or an evaluation's (:meth:`RolloutStartup.ready`): no rollout waits for the copy, and the first ones wait
only for what of it outlasts the launch save (its log says how much: "``<n>s of it after the rest of the start-up``").
It waits :data:`STAGING_WAIT_S` at most, and not past a shutdown request: the rollouts then start and wait for the copy
themselves, as before.  ``platoon eval`` stages the same way while it lists its tasks (it has no fork server).  A
rollout's own staging then finds the copy current (:func:`platoon.harness.toolchains.stage`).

**What a fork server imports and builds ahead.**  A fork server (:mod:`platoon.utils.forkserver`) imports the worker's
stack once (``WORKER_PRELOAD`` and the environments' packages, :mod:`platoon.train.workers.supervisor`), and each
worker it forks starts with those modules imported.  A worker then opened its environment and its harness, which
imported far more: an ORS environment's ``openreward`` SDK pulls in ``anthropic``, ``openai`` and ``opentelemetry``
(about 3,300 modules), the CodeAct harness IPython, prompt_toolkit and jedi (about 400), in each worker again, each
import a search of the interpreter's path on whatever filesystem the venv lives on.  With a hundred workers starting
together, that was the largest part of a rollout's start.  :func:`worker_preload` derives, from the run's environment
entries, what those rollouts import and build when they open, so that the fork server does it once
(:class:`WorkerPreload`):

* the **modules** the rollout imports: its rollout function's and reward processor's, the environment service's (an
  ORS environment class's module and the SDK, a Harbor service and its sandbox provider, a ``module:Class`` service's
  module), each function harness's module (what it imports at import time comes with it), each program's, and those
  that ``harness_rollout`` imports as it runs; then the run's own (``workflow_config.worker_preload``), for what an
  environment imports only when it opens;
* the **builds** a worker would otherwise make: each harness and program looked up in its registry (which scans the
  installed packages' entry points and parses the built-in specs), and each in-process ORS environment's task lists
  for the splits the run's tasks come from, which the SDK caches on the class (``list_tasks`` parses a whole task
  file; TextCraft-Synth's is 20 MB).

The fork server imports and builds each of these only when that is fork-safe: one that starts a thread, leaves a file
or socket open (an event loop holds both), starts a process or seeds a numpy generator from the OS is left out, with
the reason in the trainer's log, and the workers import or build it themselves, as before
(:mod:`platoon.utils.forkserver`); an environment variable one sets (the ORS SDK's update check off) is set in each
worker too.  A worker forked from the server is then the process it was, with those modules imported and those caches
filled: the task lists it reads are those it would have built.  What the checks cannot see is a value an import makes
once per process (an id, a temporary directory), which the workers share; the derived modules keep none, and a module
of ``worker_preload`` must not either.  Under ``worker_python_env_isolation`` nothing is derived: those rollouts import
their environments inside a Python overlay of their own.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from platoon.options import Phase
    from platoon.train.components import EnvironmentConfig

logger = logging.getLogger(__name__)

__all__ = [
    "HARNESS_ROLLOUT_MODULES",
    "STAGING_WAIT_S",
    "RolloutStartup",
    "WorkerPreload",
    "warm_harness",
    "warm_program",
    "warm_task_lists",
    "worker_preload",
]


#: How long a run waits, at most, before its first rollout, for the staging of its toolchains (:meth:`RolloutStartup
#: .ready`); its rollouts then start, and those that need a toolchain wait for the copy themselves.
STAGING_WAIT_S = 600.0


class RolloutStartup:
    """The toolchains a run stages ahead of its rollouts, and the workflows whose rollout workers it starts ahead."""

    def __init__(self, toolchains: Sequence[str], workflows: Sequence[Any] = ()) -> None:
        self.toolchains = list(dict.fromkeys(toolchains))
        self.workflows = [workflow for workflow in workflows if workflow is not None]
        self._staging: asyncio.Future[Any] | None = None
        self._began = 0.0

    @classmethod
    async def begin(cls, environments: Sequence[Any], workflows: Sequence[Any] = ()) -> RolloutStartup:
        """Start the fork servers of ``workflows`` (``warm_up``, for a workflow that has one) and the staging of the
        named toolchains of ``environments``' TOML harness specs (:func:`platoon.train.auto.harness_toolchains`), on a
        thread of its own; returns at once.  A workflow whose start fails is logged, and its first rollouts start it,
        as before."""

        from platoon.train.auto import harness_toolchains

        startup = cls(harness_toolchains(environments), workflows)
        for workflow in startup.workflows:
            warm_up = getattr(workflow, "warm_up", None)
            if not callable(warm_up):
                continue
            try:
                await warm_up()
            except Exception:  # noqa: BLE001 - the first rollouts start it, as before
                logger.warning(
                    "could not start the rollout workers of %s ahead of its rollouts", workflow, exc_info=True
                )
        startup._stage()
        return startup

    def _stage(self) -> None:
        if not self.toolchains:
            return
        from platoon.harness.toolchains import prestage

        self._began = time.monotonic()
        logger.info("Staging the toolchains %s on this node ahead of the first rollout", ", ".join(self.toolchains))
        self._staging = _in_thread(prestage, self.toolchains, log=logger.info)

    async def ready(self, *, timeout: float | None = None, stop: asyncio.Event | None = None) -> None:
        """Wait until the toolchains are staged; at once when there are none.  It waits at most ``timeout`` seconds
        (default :data:`STAGING_WAIT_S`), and no longer once ``stop`` is set (a shutdown was asked for): the staging
        goes on, on its thread, and the rollouts that start meanwhile stage the toolchains themselves, each waiting for
        the copy under its lock, as they did before platoon staged ahead, within their own time limits.  A toolchain
        that could not be staged is logged (:func:`platoon.harness.toolchains.prestage`), and its rollouts stage it
        themselves, as before.  Raises only when cancelled."""

        if self._staging is None:
            return
        waited = time.monotonic()
        deadline = waited + (STAGING_WAIT_S if timeout is None else timeout)
        # A second at a time: a signal handler that sets ``stop`` does not wake the event loop by itself.
        while not self._staging.done() and not (stop is not None and stop.is_set()):
            if (remaining := deadline - time.monotonic()) <= 0:
                break
            await asyncio.wait([self._staging], timeout=min(remaining, 1.0))
        now = time.monotonic()
        if not self._staging.done():
            why = "a shutdown was asked for" if stop is not None and stop.is_set() else f"after {now - waited:.0f}s"
            logger.warning(
                "Starting the rollouts before the toolchains %s are staged (%s): the staging goes on, and each rollout "
                "that needs one meanwhile waits for it",
                ", ".join(self.toolchains),
                why,
            )
            self._staging.add_done_callback(self._staged_late)
            return
        if (error := self._staging.exception()) is not None:  # each rollout stages it itself, as before
            logger.warning("could not stage the toolchains %s ahead of the rollouts", self.toolchains, exc_info=error)
            return
        logger.info(
            "Staged the toolchains in %.1fs, %.1fs of it after the rest of the start-up: the rollouts start now",
            now - self._began,
            now - waited,
        )

    def _staged_late(self, staging: asyncio.Future[Any]) -> None:
        """Say how the staging that the rollouts did not wait for ended."""

        if staging.cancelled():
            return
        if (error := staging.exception()) is not None:
            logger.warning("could not stage the toolchains %s ahead of the rollouts", self.toolchains, exc_info=error)
        else:
            logger.info("Staged the toolchains in %.1fs, after the rollouts started", time.monotonic() - self._began)


def _in_thread(fn: Any, /, *args: Any, **kwargs: Any) -> asyncio.Future[Any]:
    """``fn(*args, **kwargs)`` on a daemon thread of its own, as a future of the running loop: a run that ends while
    it copies is not held up by it (a staging cut short leaves a partial copy, which the next one removes)."""

    loop = asyncio.get_running_loop()
    future: asyncio.Future[Any] = loop.create_future()
    context = contextvars.copy_context()

    def deliver(value: Any, error: BaseException | None) -> None:
        if future.done():
            return
        if error is None:
            future.set_result(value)
        else:
            future.set_exception(error)

    def work() -> None:
        value, error = None, None
        try:
            value = context.run(fn, *args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - delivered to the waiter
            error = exc
        try:
            loop.call_soon_threadsafe(deliver, value, error)
        except RuntimeError:  # the loop closed meanwhile
            pass

    threading.Thread(target=work, name="platoon-toolchain-staging", daemon=True).start()
    return future


# ------------------------------------------------------------------------- what a fork server imports and builds


#: What ``harness_rollout`` (``platoon.harness.rollout``) imports as it runs a rollout, beyond its own imports: the
#: gateway's server, tool plane and trajectories, and for a function harness the inline runtime, its subagents and
#: verifiers, and the episode programs.
HARNESS_ROLLOUT_MODULES: tuple[str, ...] = (
    "platoon.harness.rollout",
    "platoon.gateway",
    "platoon.gateway.server",
    "platoon.gateway.mcp",
    "platoon.gateway.trace",
    "platoon.harness.inline",
    "platoon.harness.agents",
    "platoon.harness.verify",
    "platoon.programs",
    "platoon.utils.subagent_rewards",
)
# The modules of the services and sandbox providers a rollout's environment opens through, beyond the class's own.
_ORS_MODULES = ("platoon.envs.ors", "platoon.envs.ors.service", "openreward")
_PROVIDER_MODULES = {"singularity": ("hpc_sandbox",), "enroot": ("hpc_sandbox",)}
_HARBOR_DEFAULT_PROVIDER = "docker"
_THIS = __name__


@dataclass(frozen=True)
class WorkerPreload:
    """What a fork server imports (``modules``) and builds (``prepare``: ``{"ref": "module:function", "kwargs":
    {...}}`` steps, called once in the server) ahead of its workers, beyond the worker's stack and the environments'
    packages: each only when it is fork-safe (see the module docstring)."""

    modules: tuple[str, ...] = ()
    prepare: tuple[Mapping[str, Any], ...] = ()


def worker_preload(
    environments: Sequence[EnvironmentConfig],
    *,
    phase: Phase = "train",
    tasks: Iterable[Any] = (),
    extra: Sequence[str] = (),
) -> WorkerPreload:
    """What the fork server of ``phase``'s rollout workers imports and builds ahead of them, for ``environments``
    (see the module docstring).  ``tasks`` are the phase's task rows (or ids), whose ORS splits get their task lists
    built ahead; ``extra`` is the run's own list of modules (``workflow_config.worker_preload``), imported after the
    derived ones."""

    from platoon.train.auto import environment_routes
    from platoon.train.components import HARNESS_ROLLOUT

    routes = environment_routes(environments, phase=phase, only_routed=True)
    splits = _ors_splits(environments, tasks)
    modules: list[str] = []
    prepare: list[dict[str, Any]] = []
    for environment in environments:
        route = routes.get(environment.label)
        if route is None:  # a rollout of the entry's own, which names no service or harness
            own = environment.rollout if phase == "train" else environment.eval_rollout or environment.rollout
            modules += _module_of_component("rollout", own)
            continue
        rollout = route.rollout if phase == "train" else route.eval_rollout
        modules += _module_of_ref(rollout)
        if rollout == HARNESS_ROLLOUT:
            modules += HARNESS_ROLLOUT_MODULES
        modules += _module_of_ref(route.reward_processor)
        modules += _service_modules(route.service, route.service_kwargs)
        functions = False
        for ref in environment.harness_refs():
            harness_modules, function = _harness_modules(ref)
            modules += harness_modules
            functions = functions or function
            prepare.append({"ref": f"{_THIS}:warm_harness", "kwargs": {"ref": ref}})
        if functions and rollout == HARNESS_ROLLOUT:  # a TOML spec runs no episode program
            modules += _program_modules(route.program)
            prepare.append({"ref": f"{_THIS}:warm_program", "kwargs": {"name": route.program}})
        service = route.service or ""
        if service.startswith("python://") and splits.get(environment.label):
            kwargs = {"service": service, "splits": sorted(splits[environment.label])}
            prepare.append({"ref": f"{_THIS}:warm_task_lists", "kwargs": kwargs})
    modules += [str(name) for name in extra]
    steps = {repr(sorted(step["kwargs"].items())) + step["ref"]: step for step in prepare}
    return WorkerPreload(modules=tuple(dict.fromkeys(name for name in modules if name)), prepare=tuple(steps.values()))


def _module_of_ref(ref: str | None) -> list[str]:
    """The module of a ``module:attr`` reference."""

    if not ref or ":" not in ref:
        return []
    return [ref.partition(":")[0]]


def _module_of_component(kind: str, spec: str | None) -> list[str]:
    """The module of the component ``spec`` names, by its worker-importable reference; none when it does not
    resolve (its rollouts report why)."""

    if spec is None:
        return []
    from platoon.registry import resolve_component
    from platoon.train.workers.protocol import validated_ref

    try:
        return _module_of_ref(validated_ref(resolve_component(kind, spec), what=f"{kind} {spec!r}"))
    except Exception:  # noqa: BLE001 - only a preload; the rollout resolves it again and says why it cannot
        logger.debug("cannot resolve %s %r for the fork server's preload", kind, spec, exc_info=True)
        return []


def _service_modules(service: str | None, options: Mapping[str, Any]) -> list[str]:
    """What a session of the service ``service`` imports when it opens: the service's own modules, an ORS
    environment class's module and the SDK, a Harbor service's sandbox provider."""

    from platoon.envs.service import service_target

    if not service:
        return []
    modules: list[str] = []
    if service.startswith(("python://", "ors+http://", "ors+https://")):
        modules += _ORS_MODULES
    target = service_target(service)
    if target is not None:
        modules += _module_of_ref(target)
    if service == "harbor":
        provider = options.get("provider", _HARBOR_DEFAULT_PROVIDER)
        if isinstance(provider, str):
            from platoon.sandbox import provider_class

            try:
                modules.append(provider_class(provider).__module__)
            except Exception:  # noqa: BLE001 - building the provider reports it
                logger.debug("cannot import the sandbox provider %r for the fork server's preload", provider)
            modules += _PROVIDER_MODULES.get(provider, ())
    return modules


def _harness_modules(ref: str) -> tuple[list[str], bool]:
    """A harness's module (a function harness's: what it imports at import time comes with it; a TOML spec runs
    through ``harness_rollout``'s own) and whether it is a function harness."""

    from platoon.harness.function import FunctionHarness
    from platoon.harness.registry import get_harness

    try:
        harness = get_harness(ref)
    except Exception:  # noqa: BLE001 - refused where the route is built, with its own error
        return [], False
    if isinstance(harness, FunctionHarness):
        return [harness.fn.__module__], True
    return ["platoon.harness.runner"], False


def _program_modules(name: str) -> list[str]:
    from platoon.programs import resolve_program

    try:
        return [resolve_program(name).fn.__module__]
    except Exception:  # noqa: BLE001 - its rollouts report why
        return []


def _ors_splits(environments: Sequence[EnvironmentConfig], tasks: Iterable[Any]) -> dict[str, set[str]]:
    """The splits the tasks of each in-process ORS entry come from (``<split>:<index>`` ids), by entry label."""

    from platoon.envs.ors.service import TaskRef
    from platoon.train.components import split_task_id

    in_process = {
        environment.label for environment in environments if (environment.service or "").startswith("python://")
    }
    if not in_process:
        return {}
    mixture = len(environments) > 1
    only = environments[0].label
    found: dict[str, set[str]] = {}
    for row in tasks:
        task_id = row.get("task_id") if isinstance(row, Mapping) else row
        if not isinstance(task_id, str):
            continue
        try:
            label, raw_id = split_task_id(task_id) if mixture else (only, task_id)
            ref = TaskRef.parse(raw_id)
        except ValueError:
            continue
        if label in in_process and ref is not None:
            found.setdefault(label, set()).add(ref.split)
    return found


# --------------------------------------------------------------------------------------- the fork server's builds


def warm_harness(ref: str) -> None:
    """Look ``ref`` up in the harness registry, as each rollout does (its entry points scanned, its spec parsed)."""

    from platoon.harness.registry import get_harness

    get_harness(ref)


def warm_program(name: str) -> None:
    """Look the episode program ``name`` up, as each rollout of a function harness does."""

    from platoon.programs import resolve_program

    resolve_program(name)


def warm_task_lists(service: str, splits: Sequence[str]) -> None:
    """Build the task lists of ``splits`` of the in-process ORS environment ``service`` (``python://module:Class``)
    into the class's task cache, which each rollout's task lookup reads (:meth:`OrsEnvironmentService.warm_task_lists`
    says how)."""

    from platoon.envs.service import get_service

    get_service(service).warm_task_lists(splits)
