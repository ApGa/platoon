"""A rollout worker supervisor's fork server: which one forks a worker; starting, replacing, warming and closing it.

:class:`~platoon.train.workers.supervisor.RolloutWorkerSupervisor` starts each worker it launches by forking it from a
fork server (:mod:`platoon.utils.forkserver`) when one can fork it, and by exec otherwise; its module docstring, *How
a worker starts*, says when.  What a server imports and builds ahead of its workers is the supervisor's
(``_fork_server_preload``, ``_fork_server_optional``, ``_fork_server_child_init``, ``_fork_server_prepare``), and so is
the state this module keeps: the server (``_fork_server``), how many it started (``_fork_server_starts``, at most
``_max_fork_server_starts``), its log, and what this process said of it.

``_fork_server_for``, ``_launch_fork_server``, ``warm``, ``_use``, ``_warn_exec`` and ``close`` are the supervisor's
methods still: ``RolloutWorkerSupervisor`` keeps each under its name, so to replace one, replace the method, on the
class or an instance, not the function.  ``_lock``, ``_retire`` and ``_await_imports`` are not methods: they are
steps of ``_fork_server_for`` (and ``_lock`` of ``warm`` too), which call them directly.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

from platoon.utils.forkserver import ForkServer, ForkServerError

if TYPE_CHECKING:
    from platoon.train.workers.protocol import RolloutWorkerRequest
    from platoon.train.workers.supervisor import RolloutWorkerSupervisor

logger = logging.getLogger(__name__)


async def _fork_server_for(supervisor: RolloutWorkerSupervisor, request: RolloutWorkerRequest) -> ForkServer | None:
    """The fork server to start ``request``'s worker, started on first use; None to start it by exec.

    A server forks workers with the environment and working directory it started with, which its imports may
    have read: when this process's have changed since, another server starts, and the old one serves the workers
    it forked to their end.  Requests wait for a new server's imports until the worker's startup budget has
    passed since it started.  A server still importing then is neither waited for nor killed: workers start by
    exec meanwhile, and once it has imported the stack it forks the next ones.  Another server would import the
    stack from the beginning, on a node busy enough to have made this one slow.  A server that says it cannot
    serve (a module did not import, a thread runs in it), however late, turns forking off: so would another."""

    if supervisor.start_method != "forkserver":
        return None
    env, cwd = supervisor._child_env(), os.getcwd()
    loop = asyncio.get_running_loop()
    server = supervisor._fork_server
    if server is not None and server.serving and server.loop is loop and server.env == env and server.cwd == cwd:
        return supervisor._use(server)
    if server is None and supervisor._fork_server_starts >= supervisor._max_fork_server_starts:
        return None
    async with _lock(supervisor, loop):
        server = supervisor._fork_server
        if server is not None and not (
            server.alive and server.loop is loop and server.env == env and server.cwd == cwd
        ):
            _retire(supervisor, server, loop, env, cwd)
            if server.error is not None:
                # It said it cannot serve after a request stopped waiting for it: another would import the same
                # stack, and fail the same way.
                supervisor._fork_server_starts = supervisor._max_fork_server_starts
                supervisor._warn_exec(ForkServerError(server.error, permanent=True))
                return None
            server = None
        if server is None:
            if supervisor._fork_server_starts >= supervisor._max_fork_server_starts:
                return None
            server = supervisor._launch_fork_server(request, env, cwd)
            if server is None:
                return None
        if not server.serving and not await _await_imports(supervisor, server, request):
            return None
        return supervisor._use(server)


def _lock(supervisor: RolloutWorkerSupervisor, loop: asyncio.AbstractEventLoop) -> asyncio.Lock:
    """The lock under which the supervisor starts or replaces its fork server, made for ``loop`` (a lock serves one
    event loop)."""

    if supervisor._fork_server_lock is None or supervisor._fork_server_lock[0] is not loop:
        supervisor._fork_server_lock = (loop, asyncio.Lock())
    return supervisor._fork_server_lock[1]


def _retire(
    supervisor: RolloutWorkerSupervisor,
    server: ForkServer,
    loop: asyncio.AbstractEventLoop,
    env: dict[str, str],
    cwd: str,
) -> None:
    """Close ``server``, which cannot fork the worker asked for (it is dead, of another event loop, or of another
    environment or working directory), and forget it; one that serves on, of this loop, is replaced, and that is
    logged with what comes next."""

    if server.error is None and server.alive and server.loop is loop:
        changed = sorted(name for name in env.keys() | server.env.keys() if env.get(name) != server.env.get(name))
        if supervisor._fork_server_starts < supervisor._max_fork_server_starts:
            then = "starting another"
        else:
            then = (
                "rollout workers start by exec from now on: this supervisor has started "
                f"{supervisor._fork_server_starts} fork server(s), the most it starts"
            )
        logger.warning(
            "rollout worker fork server %s started with another %s; %s",
            server.pid,
            f"environment ({', '.join(changed[:8])})" if changed else "working directory",
            then,
        )
    # Dead, of another event loop or of another environment: it serves its workers out (one still
    # importing has none, and is killed).
    server.close()
    supervisor._fork_server = None


async def _await_imports(
    supervisor: RolloutWorkerSupervisor, server: ForkServer, request: RolloutWorkerRequest
) -> bool:
    """Wait for ``server`` to import the worker's stack, until ``request``'s startup budget has passed since it
    started; whether it serves.  One still importing then is neither waited for longer nor closed (said once); one
    that cannot serve is closed and forgotten, and turns forking off when that is permanent."""

    try:
        await server.ready(request.startup_budget_seconds)
    except ForkServerError as exc:
        if not exc.permanent and server.alive:  # still importing
            if not supervisor._fork_server_late:
                supervisor._fork_server_late = True
                logger.warning(
                    "rollout worker fork server %s has not imported the worker's stack in %.0fs: rollout "
                    "workers start by exec, each importing it itself, until it has (log %s)",
                    server.pid,
                    request.startup_budget_seconds,
                    supervisor._fork_server_log,
                )
            return False
        if exc.permanent:
            supervisor._fork_server_starts = supervisor._max_fork_server_starts  # starting another would not help
        server.close()
        supervisor._fork_server = None
        supervisor._warn_exec(exc)
        return False
    return True


def _launch_fork_server(
    supervisor: RolloutWorkerSupervisor, request: RolloutWorkerRequest, env: dict[str, str], cwd: str
) -> ForkServer | None:
    """Start a fork server for workers like ``request``'s, counted against
    :data:`~platoon.train.workers.supervisor.MAX_FORK_SERVER_STARTS`; None when it could not start (workers then start
    by exec).  It imports the worker's stack meanwhile."""

    supervisor._fork_server_starts += 1
    supervisor._fork_server_log = os.path.join(
        supervisor.scratch_dir, f"forkserver-{os.getpid()}-{supervisor._fork_server_starts}.log"
    )
    supervisor._fork_server_late = supervisor._fork_server_announced = False
    try:
        server = ForkServer.launch(
            python=supervisor.python_executable,
            module=supervisor.module,
            preload=supervisor._fork_server_preload(request),
            child_init=supervisor._fork_server_child_init(),
            prepare=supervisor._fork_server_prepare(request),
            optional=supervisor._fork_server_optional(request),
            skipped=supervisor._fork_server_skipped,
            env=env,
            cwd=cwd,
            log_path=supervisor._fork_server_log,
        )
    except Exception as exc:  # noqa: BLE001 - workers then start by exec
        supervisor._warn_exec(exc)
        return None
    supervisor._fork_server = server
    return server


async def warm(supervisor: RolloutWorkerSupervisor, request: RolloutWorkerRequest) -> None:
    """Start the fork server for workers like ``request``'s now, ahead of the first one, without waiting for its
    imports: the first worker asked for then finds it importing, or serving.  Nothing when workers start by exec,
    a server runs already, or the supervisor started as many as it starts."""

    if supervisor.start_method != "forkserver":
        return
    loop = asyncio.get_running_loop()
    async with _lock(supervisor, loop):
        if supervisor._fork_server is not None or supervisor._fork_server_starts >= supervisor._max_fork_server_starts:
            return
        supervisor._launch_fork_server(request, supervisor._child_env(), os.getcwd())


def _use(supervisor: RolloutWorkerSupervisor, server: ForkServer) -> ForkServer:
    """``server``, which serves; the first time, say so, what it imported beyond the worker's stack, and what it
    left out or could not build ahead."""

    if not supervisor._fork_server_announced:
        supervisor._fork_server_announced = True
        preloaded = getattr(server, "preloaded", 0)
        logger.info(
            "rollout worker fork server %s imported the worker's stack%s in %.1fs",
            server.pid,
            f" and {preloaded} module(s) the environments open with" if preloaded else "",
            server.seconds,
        )
        supervisor._fork_server_skipped = [dict(entry) for entry in getattr(server, "skipped", ())]
        for entry in supervisor._fork_server_skipped:
            if entry.get("item") in supervisor._fork_server_warned:
                continue
            supervisor._fork_server_warned.add(str(entry.get("item")))
            logger.warning(
                "the rollout worker fork server left %s out of what it imports ahead of its workers, which then "
                "import or build it themselves: %s",
                entry.get("item"),
                entry.get("reason"),
            )
        for problem in server.prepare_errors:
            logger.warning("the rollout worker fork server could not build ahead (workers build it): %s", problem)
    return server


def _warn_exec(supervisor: RolloutWorkerSupervisor, exc: BaseException) -> None:
    """Warn that workers start by exec, because the fork server did not start."""

    logger.warning(
        "rollout workers start by exec, each importing the worker's stack itself: the fork server did not start: "
        "%s (log %s)",
        exc,
        supervisor._fork_server_log,
    )


def close(supervisor: RolloutWorkerSupervisor) -> None:
    """Stop the fork server taking workers; it exits once the workers it forked have exited (at once when it is
    still importing the worker's stack)."""

    if supervisor._fork_server is not None:
        supervisor._fork_server.close()
        supervisor._fork_server = None
