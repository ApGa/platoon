"""One evaluation rollout in a worker process, so a harness that hangs or crashes costs only its own rollout.

Each of the workflow's slots has a process of its own, which runs one rollout at
a time and is reused until a rollout in it hangs or dies (see
:mod:`platoon.eval.workflow`). The parent loads the task (through the
environment's task loader and route) and ships it with a ``module:qualname``
reference to the rollout function; the child installs its own policy gateway when
one is configured (once per process), runs the rollout under a hard ``SIGALRM``
deadline in its own process group, and returns the collection's dump.

At the hard deadline the process starts no sandbox any more, stops the sandboxes
its rollout still holds (within :data:`SANDBOX_STOP_SECONDS`), then SIGKILLs the
process groups the rollout started and its own; the rollout is not cancelled
meanwhile. A backstop alarm kills them :data:`KILL_BACKSTOP_SECONDS` after the
deadline anyway, should the rollout hold its event loop. What is left the parent
reaps by the process's label (:mod:`platoon.sandbox.cleanup`).
"""

from __future__ import annotations

import asyncio
import os
import signal
import traceback
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any

from platoon.sandbox import cleanup as sandbox_cleanup
from platoon.utils.stopping import kill_process_groups, register_descendant_groups

if TYPE_CHECKING:
    from platoon.eval.config_defs import EvalPolicy
    from platoon.task import Task

#: How long a rollout past its hard timeout has to stop the sandboxes it still holds before its process group is
#: SIGKILLed.
SANDBOX_STOP_SECONDS = 30.0
#: How long after the hard timeout the process group is SIGKILLed at the latest, also when the rollout's event loop
#: never runs the stop (a blocking call holds it).
KILL_BACKSTOP_SECONDS = SANDBOX_STOP_SECONDS + 5.0
#: How long the parent waits past the hard timeout for the process to end itself before it kills it: the backstop,
#: and a margin for the process's start, since the parent's clock starts before the process arms its alarm.
PARENT_GRACE_SECONDS = KILL_BACKSTOP_SECONDS + 25.0

# The policy this pooled process serves; a process reused for the next rollout keeps its gateway.
_installed_policy: EvalPolicy | None = None
# The rollout in progress: its event loop, and the future that the hard-timeout alarm completes there.
_rollout: tuple[asyncio.AbstractEventLoop, asyncio.Future[None]] | None = None
# Whether the hard timeout of the rollout in progress fired, so the next alarm is the backstop.
_hard_timeout_fired = False


#: The time limit a slot's hard deadline counts for a rollout without a timeout whose harness's own limit it does not
#: count (:func:`_hard_timeout_seconds`).
UNLIMITED_ROLLOUT_SECONDS = 900


def _hard_timeout_seconds(config_dict: dict[str, Any], task: Task | None = None, rollout: str | None = None) -> int:
    """The slot's hard deadline: the rollout's time limit, what the task's environment needs beyond it (a Harbor
    task's sandbox starts and verifier, :func:`platoon.train.components.rollout_overhead_s`), and three minutes of
    slack.  The time limit is the rollout's ``timeout``, else the own limit of the harness that the rollout function
    ``rollout`` runs the task with (:func:`platoon.train.components.harness_time_limit_s`), else 900 s.  Where the
    task's environment does not say what it needs beyond the harness
    (:func:`platoon.train.components.known_rollout_overhead_s`), the harness's own limit counts only when it is the
    longer: a deadline that counted the harness alone could end the rollout while it is graded."""

    from platoon.train.components import harness_time_limit_s, known_rollout_overhead_s

    overhead = known_rollout_overhead_s(task) if task is not None else None
    limit: float | None = config_dict.get("timeout") or None
    if limit is None and task is not None:
        harness = harness_time_limit_s(task, rollout)
        if harness is not None:
            limit = harness if overhead is not None else max(harness, UNLIMITED_ROLLOUT_SECONDS)
    return int((UNLIMITED_ROLLOUT_SECONDS if limit is None else limit) + (overhead or 0.0) + 120 + 60)


def _hard_timeout_handler(_signum, _frame):
    """SIGALRM: the rollout is past its hard timeout, or the backstop came.

    Only signal-safe steps here: the hard timeout arms the backstop and hands the stop to the rollout's event loop
    (:func:`_run_until_hard_timeout`); the backstop, or a hard timeout while no rollout loop runs, kills at once.
    """

    global _hard_timeout_fired
    rollout = _rollout
    if rollout is not None and not _hard_timeout_fired:
        _hard_timeout_fired = True
        signal.setitimer(signal.ITIMER_REAL, KILL_BACKSTOP_SECONDS)
        loop, fired = rollout
        try:
            loop.call_soon_threadsafe(_complete, fired)
        except RuntimeError:  # the loop is closed: the rollout is over
            pass
        else:
            _say("Hard timeout exceeded; stopping the rollout's sandboxes, then killing the process group")
            return
    _say("Hard timeout exceeded; killing subprocess process group")
    _kill_process_group()


def _complete(future: asyncio.Future[None]) -> None:
    if not future.done():
        future.set_result(None)


def _say(message: str) -> None:
    """Print from a signal handler: one write to the standard output descriptor, which takes no lock."""

    try:
        os.write(1, f"[EvalSubprocessWorker] {message}\n".encode())
    except OSError:
        pass


def _kill_process_group() -> None:
    """SIGKILL the process groups the rollout started in sessions of their own (a harness CLI, sandbox commands),
    out of reach of its own group's SIGKILL, then this process's group, this process included.  Signal-safe."""

    try:
        register_descendant_groups()
        kill_process_groups()
    except Exception:
        pass
    try:
        os.killpg(os.getpgrp(), signal.SIGKILL)
    except Exception:
        os._exit(1)


async def _run_until_hard_timeout(rollout: Coroutine[Any, Any, Any]) -> Any:
    """Run the rollout and return its result; once its hard timeout fired, start no sandbox, stop those it still
    holds and end the process instead (the rollout runs on meanwhile, and dies with the process group)."""

    global _rollout
    loop = asyncio.get_running_loop()
    main = asyncio.ensure_future(rollout)
    fired: asyncio.Future[None] = loop.create_future()
    _rollout = (loop, fired)
    try:
        await asyncio.wait([main, fired], return_when=asyncio.FIRST_COMPLETED)
    finally:
        _rollout = None
    if main.done():  # also when it finished as the hard timeout fired: nothing was stopped yet
        return main.result()
    try:
        sandbox_cleanup.shut_down()
        await sandbox_cleanup.stop_live(SANDBOX_STOP_SECONDS)
    finally:
        _kill_process_group()


def run_rollout_subprocess(
    rollout_fn_ref: str,
    task: Task,
    rollout_config_dict: dict[str, Any],
    policy: EvalPolicy | None = None,
) -> dict[str, Any]:
    """Run ``task`` with the rollout function ``rollout_fn_ref`` and return the collection's dump.

    A failure comes back as a ``RuntimeError`` whose message is the original's type and text: only text crosses
    the process boundary, since the original (an ``EnvSetupError``, an ``httpx`` error) may not unpickle in the
    parent, and an exception that fails to unpickle breaks the pool.  The Tinker service's refusal of the key
    (:class:`~platoon.eval.config_defs.TinkerKeyRefusedError`) comes back as itself, which the evaluation ends with.
    """
    global _hard_timeout_fired
    try:
        os.setpgrp()
    except Exception:
        pass

    _hard_timeout_fired = False
    old_handler = signal.signal(signal.SIGALRM, _hard_timeout_handler)
    signal.alarm(_hard_timeout_seconds(rollout_config_dict, task, rollout_fn_ref))
    try:
        from platoon.config_defs import RolloutConfig
        from platoon.train.workers.protocol import resolve_ref

        rollout_fn = resolve_ref(rollout_fn_ref)
        global _installed_policy
        if policy is not None and policy != _installed_policy:
            policy.install()
            _installed_policy = policy
        result = asyncio.run(_run_until_hard_timeout(rollout_fn(task, RolloutConfig(**rollout_config_dict))))
        if hasattr(result, "to_dict"):
            result = result.to_dict()
        if not isinstance(result, dict):
            raise TypeError(f"Expected the rollout to return a trajectory collection dict, got {type(result).__name__}")
        return result
    except Exception as e:
        from platoon.eval.config_defs import TinkerKeyRefusedError  # loaded already: the policy's module

        if isinstance(e, TinkerKeyRefusedError):
            raise  # platoon's own, which unpickles in the parent: the evaluation ends with its line, not a traceback
        print(f"[EvalSubprocessWorker] Rollout failed for task {task.id}: {e}")
        traceback.print_exc()
        raise RuntimeError(f"{type(e).__name__}: {e}") from e
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
