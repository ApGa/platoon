"""``python -m platoon.train.workers.entrypoint <request_path>``.

One rollout tree per interpreter, for its whole life.  The ordering below is
load-bearing and each step says why.

Deliberately **not** ``multiprocessing``:

* ``mp.Process(spawn)`` re-imports ``__main__`` in the child, which for a
  ``python -m ...train_tinker`` trainer re-executes the entire train script.
* ``mp.Process(args=...)`` unpickles its arguments *before* the target runs, so
  a ``SamplingClient`` in ``args`` would build its shadow holder -- and its
  telemetry -- before this process could set ``TINKER_TELEMETRY=0`` or apply
  the R3 ``extra='allow'`` patch.  Reading a *file* after the environment is
  prepared removes that ordering hazard entirely.
* ``ProcessPoolExecutor`` fails every concurrent future in the pool when one
  child is killed, and killing a child is routine here.

**Where the environment is actually fixed.**  ``-m`` imports the parent
packages before this module's body runs, so ``platoon/train/workers/__init__.py``
-- and therefore tinker and torch, which the protocol it exports imports -- has
already executed by the time ``main`` starts.  The invariants that must precede those imports (telemetry off, thread
pins, no sampling sidecar) and the pre-import startup deadline therefore live in
:mod:`platoon.train.workers.env`, which that package init calls first.
``main`` re-applies them so the requirement is also stated where it is used.

**Stopping.**  A worker stops on SIGTERM -- the supervisor's straggler kill, or
the trainer's death through PDEATHSIG -- and when its parent is gone (the
monitor thread checks ``getppid()``, in case that signal was lost), and it
leaves within :data:`TERMINATION_SECONDS` whatever its rollout does
(:func:`_run_until_terminated`):

1. the hard deadline is armed at once, from the signal handler (an alarm that
   SIGKILLs the process group and the groups the rollout started, backed by the
   monitor thread);
2. the rollout's model gateway is shut down, and remote env sessions and new
   sandboxes refused (``ModelGateway.shut_down``,
   ``platoon.envs.ors.cleanup.shut_down``, ``platoon.sandbox.cleanup.shut_down``):
   from then on no model call, tool call, subagent, env session or sandbox
   starts, also from a harness that ignores cancellation or calls from a thread
   of its own;
3. its remote env sessions are deleted, on a thread of their own;
4. every task of the rollout -- the root, its subagents, verifiers and judges --
   is cancelled, and again and again from a second on, for tasks that swallow
   the cancellation; its cleanups -- env teardowns, the SIGTERM, grace and
   SIGKILL of a command it ran (``platoon.utils.stopping.let_finish``) -- are
   never cancelled and run to their end;
5. the sandboxes still registered -- task sandboxes whose teardown never began
   or did not end -- are stopped (``platoon.sandbox.cleanup.stop_live``);
6. the process groups the rollout started in sessions of their own -- a harness
   CLI on the rollout host, sandbox commands, and the sessions its descendants
   started (``platoon.utils.stopping``) -- get SIGTERM, then SIGKILL
   :data:`PROCESS_GROUP_GRACE_SECONDS` later;
7. the worker then SIGKILLs its process group, itself included.

Every sandbox the worker starts carries its ``platoon.worker`` label (its request
id), beside the trainer's labels (its run's ``platoon.run``, from the request),
and how to reap it is listed in the request's sandboxes file, so a sandbox
that outlives the worker -- after a SIGKILL, or a stop whose deadline ran out --
is reaped by its supervisor (``platoon.sandbox.cleanup``).

So nothing it started outlives it, but a process that left its process tree
before the worker found it: a daemon whose parent had exited before the worker
began stopping, or a command that a stopping harness starts in a session of its
own and whose parent is killed before the worker looks again (the end of
:func:`_unwind`).  A worker that exits without being stopped ends the same
groups as it exits.

A rollout that had already finished when the signal came keeps its response.
Nothing a terminated rollout produced is reported: its env sessions are gone.
"""

from __future__ import annotations

import asyncio
import ctypes
import os
import signal
import sys
import threading
import time
from collections.abc import Coroutine
from functools import partial
from typing import TYPE_CHECKING, Any

from platoon.utils.stopping import (
    cleanups,
    end_process_groups,
    kill_process_groups,
    register_descendant_groups,
)

if TYPE_CHECKING:
    from platoon.train.workers.protocol import ModelBinding, RolloutWorkerRequest

_PR_SET_PDEATHSIG = 1

#: Refreshed by the monitor thread, written verbatim by the alarm handler.
_DIAG_PAYLOAD: bytes = b""
_DIAG_FD: int = -1
_PHASE = "startup"
_PHASE_STARTED = 0.0
#: The process that launched this worker (its trainer), recorded with PDEATHSIG.
_PARENT_PID = 0
#: Set once the worker starts stopping: the stop in progress, its deadline (``time.monotonic()``) and what the
#: diagnostic marker says about it from then on.
_TERMINATION: _Termination | None = None
_TERMINATE_BY = 0.0
_TERMINATION_DIAG: dict[str, object] = {}

#: How long a worker takes, at most, from SIGTERM or its parent's death to its exit: deleting its env sessions and
#: unwinding its rollout, then SIGKILLing its process group.  Below the supervisor's default termination grace
#: (``worker_termination_grace_seconds``, 10 s), so a live supervisor never has to kill a stopping worker; once the
#: trainer is gone nothing else bounds it.
TERMINATION_SECONDS = 8.0
#: Of that, how long deleting the worker's remote env sessions may take.
SIGTERM_CLEANUP_SECONDS = 5.0
#: How long a terminating worker lets its cancelled rollout unwind (run its own cleanups) undisturbed, and then
#: how often it cancels again the tasks of a rollout that swallowed the cancellation.
RECANCEL_AFTER_SECONDS = 1.0
RECANCEL_SECONDS = 0.25
#: How long the process groups a terminating worker's rollout started get between their SIGTERM and SIGKILL.
PROCESS_GROUP_GRACE_SECONDS = 0.5
#: What a terminating worker keeps of its deadline for killing its own process group, after the groups its
#: rollout started are gone.
EXIT_MARGIN_SECONDS = 0.5
#: The monitor thread's own kill, should the alarm not fire: this long after the termination deadline.
TERMINATION_BACKSTOP_SECONDS = 2.0


def _install_parent_death_signal() -> None:
    """Die with the trainer.

    The trainer's watchdog exits with ``os._exit`` and so does its
    double-signal handler; neither runs any cleanup.  Without PDEATHSIG a
    watchdog kill would orphan every live worker *and* every MCP bridge those
    workers own.  The launcher's own SIGTERM does not reach workers either --
    they are in a different session by construction -- so this is the mechanism
    that actually stops them.

    Same shape as the MCP bridge's own installer, including the post-``prctl``
    ``getppid()`` recheck that closes the fork/exec race.
    """

    global _PARENT_PID
    if os.name != "posix":
        return
    parent_pid = os.getppid()
    _PARENT_PID = parent_pid
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
            return
    except (AttributeError, OSError):
        return
    if os.getppid() != parent_pid:
        # The parent exited between getppid() and prctl(); the kernel cannot
        # deliver the signal retroactively, so deliver it here.
        os.kill(os.getpid(), signal.SIGTERM)


def _apply_environment_invariants() -> None:
    """Re-assert the invariants ``workers.env`` already applied.

    ``platoon/train/workers/__init__.py`` calls the same function before this
    package's imports happen, which is the only point early enough to matter.
    Repeating it here is free and keeps a direct ``main()`` call correct.

    ``TINKER_TELEMETRY=0``
        An unpickled ``SamplingClient`` builds a shadow holder that reuses the
        trainer's session id and, with telemetry on (the default), can emit
        ``SESSION_END`` from any fatal path.  The gateway reads that as the
        client exiting and reclaims the session's models and slots: the run
        dies.  Leaving it out goes unnoticed until it fires.

    thread pins
        Torch takes ``os.cpu_count()`` intra- and inter-op threads per process
        and nothing in the tree pins it.  At tens of workers on a many-core host
        that may also run the trainer and the gateway, that is thousands of
        threads and one full-width pool per worker.  Pinned in both the parent's
        ``env=`` and here, because some libraries read these at import.

    ``TINKER_SUBPROCESS_SAMPLING`` removed
        The sampling sidecar spawns an interpreter that imports tinker but
        never platoon, so it cannot carry the ``extra='allow'`` patch and would
        silently drop ``routing_id`` (R3).
    """

    from platoon.train.workers.env import apply_rollout_worker_env_invariants

    # Already applied by ``platoon/train/workers/__init__.py`` -- which
    # `python -m ...workers.entrypoint` executes before this module's body, and
    # therefore before tinker or torch are imported.  Re-applied here so the
    # invariant is stated where it is depended on, and so a direct call to
    # ``main()`` in a test still gets it.
    apply_rollout_worker_env_invariants(force=True)


def _open_diag(path: str) -> None:
    global _DIAG_FD
    try:
        _DIAG_FD = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    except OSError:
        _DIAG_FD = -1


def _refresh_diag_payload(**extra) -> None:
    """Pre-format the marker so the signal handler only has to ``os.write``."""

    global _DIAG_PAYLOAD
    from platoon.train.workers.protocol import encode_diag
    from platoon.train.workers.run import cell_guard_cumulative

    fields: dict[str, object] = {
        "phase": _PHASE,
        "elapsed": round(time.monotonic() - _PHASE_STARTED, 1),
        "pid": os.getpid(),
    }
    fields.update(cell_guard_cumulative(at_exit=True))
    fields.update(_hook_diagnostics())
    fields.update(_TERMINATION_DIAG)
    fields.update(extra)
    _DIAG_PAYLOAD = encode_diag(fields)


def _hook_diagnostics() -> dict[str, str]:
    """Fields from every ``rollout_worker.diagnostics`` hook, best effort.

    An extension uses them to say why its rollout hung, e.g. a quote of the
    model-authored source of each overdue programmatic-tool-calling cell.
    """

    from platoon.hooks import hooks

    fields: dict[str, str] = {}
    for diagnostics in hooks("rollout_worker.diagnostics"):
        try:
            fields.update({str(key): str(value) for key, value in diagnostics().items()})
        except Exception:
            continue
    return fields


def _alarm_handler(_signum, _frame) -> None:
    """Leave evidence, then kill the whole tree.

    Safe in a signal handler by construction: one ``os.write`` of a payload
    the monitor thread already formatted, then ``killpg`` -- of the process
    groups the rollout started in sessions of their own (found with raw
    ``/proc`` reads and no lock: :func:`_kill_started_process_groups`), then of
    its own.  Without the marker a hard timeout, an OOM kill and an external
    sweep are indistinguishable to the parent (``rc=-9``, no response) and the
    cell-guard counters of the very rollout that hung are lost -- so
    ``ptc_cells_stuck`` would read zero precisely when a cell was killed for
    hanging.
    """

    from platoon.train.workers.protocol import EXIT_HARD_TIMEOUT

    if _DIAG_FD >= 0 and _DIAG_PAYLOAD:
        try:
            os.write(_DIAG_FD, _DIAG_PAYLOAD)
        except OSError:
            pass
    _kill_started_process_groups()
    try:
        os.killpg(os.getpgrp(), signal.SIGKILL)
    except Exception:
        os._exit(EXIT_HARD_TIMEOUT)


def _kill_started_process_groups() -> None:
    """SIGKILL the process groups the rollout started in sessions of their own: out of reach of this worker's own
    group, they would outlive it (:mod:`platoon.utils.stopping`)."""

    try:
        register_descendant_groups()
        kill_process_groups()
    except Exception:
        pass


def _end_started_process_groups(grace_s: float) -> list[int]:
    """SIGTERM the process groups the rollout started in sessions of their own, give them ``grace_s``, then SIGKILL
    what is left of them; returns the groups that were still there."""

    try:
        register_descendant_groups()
        return end_process_groups(grace_s)
    except Exception:
        return []


def _kill_process_group() -> None:
    """Write the diagnostic marker, then SIGKILL the process groups the rollout started and this worker's own, this
    process included."""

    _alarm_handler(signal.SIGALRM, None)


def _arm_alarm(seconds: float, *, phase: str) -> None:
    global _PHASE, _PHASE_STARTED  # noqa: PLW0603
    _PHASE = phase
    _PHASE_STARTED = time.monotonic()
    _refresh_diag_payload()
    if seconds and seconds > 0:
        signal.setitimer(signal.ITIMER_REAL, float(seconds))
    else:
        signal.setitimer(signal.ITIMER_REAL, 0.0)


def _read_vm_rss_bytes() -> int:
    try:
        with open("/proc/self/status", "rb") as handle:
            for line in handle:
                if line.startswith(b"VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


#: How often the monitor thread re-formats the diagnostic payload and samples
#: RSS.  The payload has to be pre-formatted because the alarm handler may only
#: ``os.write`` it, so every field in a marker is as of the last refresh -- a
#: worker killed less than one interval after arming reports ``elapsed=0.0``.
MONITOR_INTERVAL_SECONDS = 2.0


def _start_monitor_thread(rss_soft_limit_bytes: int, interval: float = MONITOR_INTERVAL_SECONDS) -> threading.Thread:
    """Keep the diagnostic payload fresh, and bound gradual memory growth.

    Not ``RLIMIT_AS``: the bare import stack measures ~12.7 GB of address space
    against ~0.7 GB resident, so an address-space limit loose enough to import
    torch bounds nothing real and a tight one kills every worker at import.
    This bounds a *gradually* growing worker deterministically -- and says so
    in the marker, so the parent classifies ``rollout_worker_oom`` instead of
    inferring it from an anonymous ``rc=-9``.  A single huge allocation can
    still reach the kernel OOM killer; that is a stated limit, not a claim.
    """

    from platoon.train.workers.protocol import EXIT_RSS_LIMIT

    def _loop() -> None:
        while True:
            time.sleep(interval)
            rss = _read_vm_rss_bytes()
            try:
                _refresh_diag_payload(rss_bytes=rss)
            except Exception:
                pass
            _watch_parent_and_deadline()
            if rss_soft_limit_bytes > 0 and rss > rss_soft_limit_bytes:
                try:
                    global _PHASE
                    # The marker's own phase is what lets the parent tell an
                    # over-allocating rollout apart from a hard timeout and
                    # from an external sweep -- three failures that otherwise
                    # look identical and call for opposite responses.
                    _PHASE = "rss"
                    _refresh_diag_payload(rss_bytes=rss, limit_bytes=rss_soft_limit_bytes)
                    if _DIAG_FD >= 0:
                        os.write(_DIAG_FD, _DIAG_PAYLOAD)
                except Exception:
                    pass
                _kill_started_process_groups()
                os._exit(EXIT_RSS_LIMIT)

    thread = threading.Thread(target=_loop, name="rollout-worker-monitor", daemon=True)
    thread.start()
    return thread


def _watch_parent_and_deadline() -> None:
    """The monitor thread's part in stopping (see the module docstring).

    A worker whose parent is gone stops even when the PDEATHSIG SIGTERM never came (``prctl`` refused it, or the
    rollout replaced the SIGTERM handler).  And a stopping worker whose alarm did not fire by the deadline (its
    main thread stuck in a C call) is killed from here.
    """

    if _PARENT_PID and os.getppid() != _PARENT_PID:
        _request_termination("its parent is gone")
    if _TERMINATE_BY and time.monotonic() > _TERMINATE_BY + TERMINATION_BACKSTOP_SECONDS:
        _kill_process_group()


def _request_termination(reason: str) -> None:
    """Start stopping from any thread: the rollout's termination when it runs, else an immediate kill."""

    termination = _TERMINATION
    if termination is not None:
        termination.trigger(reason)
        return
    global _PHASE
    _PHASE = "terminated"
    _TERMINATION_DIAG.update(reason=reason, unwound=0)
    try:
        _refresh_diag_payload()
    except Exception:
        pass
    _kill_process_group()


def _signal_ready(ready_path: str) -> None:
    """Release the parent's admission slot.

    Written *after* imports and provider registration, because
    ``create_subprocess_exec`` returns in well under a millisecond while the
    child then spends 20-30 s importing: gating admission on exec would bound
    nothing at all.
    """

    try:
        descriptor = os.open(ready_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.write(descriptor, b"ready\n")
        os.close(descriptor)
    except OSError:
        pass


HEADER_TEMPLATE = (
    "[rollout-worker] request={request} pid={pid} pgid={pgid} extra_allow={extra_allow} "
    "telemetry={telemetry} torch_threads={torch_threads} startup={startup:.1f}s"
)


def main(argv: list[str] | None = None) -> int:
    """Run the rollout of the request at the one path in ``argv``, and write its response where the request says;
    returns the exit code.  Its steps run in this order, and each says why it comes where it does."""

    from platoon.train.workers.env import process_start_monotonic

    # Not time.monotonic(): by the time main() runs, `python -m
    # platoon.train.workers.entrypoint` has already spent 20-30 s
    # importing the stack while executing the package __init__.  Startup cost
    # must include that or the metric is a lie.
    started = process_start_monotonic()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        print("usage: python -m platoon.train.workers.entrypoint <request_path>", file=sys.stderr)
        return 2
    request_path = arguments[0]

    _become_worker_process()  # 1. its own process group, dying with the trainer, the worker invariants
    request = _load_request(request_path)  # 2. read, unlinked at once (the API key), its diag marker and cleanup files
    _arm_startup_deadline(request, started)  # 3. the rest of the startup budget, under the full alarm handler
    extra_allow = _import_stack()  # 4. tinker, torch and the policy client, which R3 needs
    _import_environment_packages(request)  # 5. what registers the environments' harnesses, programs and loaders
    llm = _build_llm(request.model)  # 6. the policy, following the trainer's sampling client
    startup_seconds = time.monotonic() - started
    _print_header(request, startup_seconds=startup_seconds, extra_allow=extra_allow)  # 7. the header line
    return _run(request, llm, startup_seconds=startup_seconds, extra_allow=extra_allow)  # 8. ready, then the rollout


def _become_worker_process() -> None:
    """Make this process a rollout worker: the leader of its own process group, dying with the trainer, with the
    worker invariants in its environment."""

    # Own process group.  ``start_new_session=True`` already did setsid;
    # this is defensive so a directly-invoked worker is still killable as a
    # group (that group is what bounds a runaway PTC cell).
    try:
        os.setpgrp()
    except OSError:
        pass

    # Die with the trainer.
    _install_parent_death_signal()

    # Environment invariants.  Already applied by ``workers.env`` from the
    # package init, which is the only point ahead of the tinker/torch
    # imports; repeated here so the requirement is stated where it is used.
    _apply_environment_invariants()


def _load_request(path: str) -> RolloutWorkerRequest:
    """Load the request at ``path`` and unlink it immediately: it carries the gateway API key inside the pickled
    sampling client.  Then open its diagnostic marker, and say where the supervisor finds what the rollout opens, in
    case this worker is killed before it can close it: its remote env sessions and its sandboxes, labelled as this
    worker's."""

    from platoon.train.workers.protocol import load_request

    request = load_request(path)
    try:
        os.unlink(path)
    except OSError:
        pass

    _open_diag(request.diag_path)
    # List the remote env sessions this worker opens where the supervisor finds them if the worker is killed
    # before it can close or delete them (its own alarm and RSS watchdog, the supervisor's SIGKILL).
    from platoon.envs.ors.cleanup import mirror_to
    from platoon.sandbox import cleanup as sandbox_cleanup

    mirror_to(request.sessions_path)
    # Label the sandboxes this worker starts, with the trainer's labels (its run's) and its own, and list how to reap
    # them, so the supervisor can when this worker cannot stop them itself.
    sandbox_cleanup.set_labels({**request.sandbox_labels, sandbox_cleanup.WORKER_LABEL: request.request_id})
    sandbox_cleanup.mirror_to(request.sandboxes_path)
    return request


def _arm_startup_deadline(request: RolloutWorkerRequest, started: float) -> None:
    """Arm what is left of the request's startup budget, under the full alarm handler (:func:`_alarm_handler`).

    The alarm is deliberately split into a startup phase and a rollout phase: a single alarm covering both would let
    time spent queueing or fighting a shared-filesystem stall silently eat the rollout's budget and turn a slow start
    into a hard timeout with no data."""

    global _PHASE, _PHASE_STARTED
    signal.signal(signal.SIGALRM, _alarm_handler)
    # The pre-import deadline armed by ``workers.env`` (from the environment)
    # already covered the imports; re-arm the remainder under the richer
    # handler.  A worker launched without that environment gets its whole
    # deadline here, and one that already blew it dies immediately with an
    # attributable marker instead of silently eating the rollout's budget.
    _PHASE, _PHASE_STARTED = "startup", started
    remaining = request.startup_budget_seconds - (time.monotonic() - started)
    _refresh_diag_payload()
    if remaining <= 0:
        _alarm_handler(signal.SIGALRM, None)
    signal.setitimer(signal.ITIMER_REAL, max(0.05, remaining))


def _import_stack() -> bool:
    """Import the stack, pin torch to one thread, and check that tinker's ``SampleResponse`` is ``extra='allow'``,
    which the policy client's import makes it (R3); returns that it is."""

    import tinker  # noqa: E402
    import torch  # noqa: E402

    # This module import applies the extra='allow' SampleResponse patch and
    # registers the platoon renderers.
    from platoon.train.tinker import proxy  # noqa: F401  (for the patch it applies, checked below)

    # interop must be set before any parallel work is scheduled.
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.set_num_threads(1)

    extra_allow = tinker.types.SampleResponse.model_config.get("extra") == "allow"
    if not extra_allow:
        raise RuntimeError(
            "R3: tinker.types.SampleResponse is not extra='allow' in this rollout worker, so the "
            "gateway's routing_id would be dropped and every datum would lose its routing_handle"
        )
    return extra_allow


def _import_environment_packages(request: RolloutWorkerRequest) -> None:
    """Import what registers the environments' harnesses, programs and loaders (environments[].package), as the
    trainer did when it loaded the config: the rollout looks them up by name.  Inside the startup budget, before the
    parent's admission slot is released.  Installed extensions' components load by name from their entry points when
    the rollout names them."""

    if request.packages:
        from platoon.train.auto import import_environment_packages

        import_environment_packages(request.packages)


def _build_llm(binding: ModelBinding) -> Any:
    """The policy the rollout samples, following the trainer's sampling client, set as the process default; None when
    the binding names no model.  NEVER register_tinker_llm(): it constructs a ServiceClient, i.e. a second gateway
    session per rollout."""

    from platoon.train.tinker import proxy  # imported with the stack (_import_stack)
    from platoon.train.workers.binding import FollowingSamplingClient

    llm = None
    if binding.model_name:
        sampling_client = FollowingSamplingClient(
            binding.binding_path,
            binding.bootstrap_blob,
            binding.follow_poll_seconds,
            bootstrap_version=binding.checkpoint_version,
        )
        llm = proxy.build_tinker_llm(
            model_name=binding.model_name,
            renderer_name=binding.renderer_name,
            sampling_client=sampling_client,
            context_window_length=binding.context_window_length,
            renderer_kwargs=binding.renderer_kwargs,
            sample_timeout_seconds=binding.sample_timeout_seconds,
            max_tokens=binding.max_tokens,
            temperature=binding.temperature,
            top_k=binding.top_k,
            top_p=binding.top_p,
            seed=binding.seed,
        )
        llm.set_version(binding.checkpoint_version)
        llm.set_as_default()
    else:
        # A binding with no model name deliberately sets no default gateway.
        # Used by CPU tests and by fault injection; a rollout that then makes
        # a model call fails loudly (``default_gateway()`` raises) rather than
        # silently reaching some other policy.
        logger_message = "[rollout-worker] no model bound; running without a sampling provider"
        print(logger_message, flush=True)
    return llm


def _print_header(request: RolloutWorkerRequest, *, startup_seconds: float, extra_allow: bool) -> None:
    """Print the worker's header line (:data:`HEADER_TEMPLATE`): its request, its process, the invariants it checked
    and its startup seconds."""

    import torch  # imported with the stack (_import_stack)

    print(
        HEADER_TEMPLATE.format(
            request=request.request_id,
            pid=os.getpid(),
            pgid=os.getpgrp(),
            extra_allow=int(extra_allow),
            telemetry=os.environ.get("TINKER_TELEMETRY", "?"),
            torch_threads=torch.get_num_threads(),
            startup=startup_seconds,
        ),
        flush=True,
    )


def _run(request: RolloutWorkerRequest, llm: Any, *, startup_seconds: float, extra_allow: bool) -> int:
    """Release the parent's admission slot, run the rollout under its own deadline until it ends or the worker stops
    (:func:`_run_until_terminated`), write its response, and end the process groups it left running; returns the exit
    code."""

    # Ready: release the parent's admission slot, then swap to the rollout
    # deadline so the rollout gets its full budget from the moment it begins.
    _signal_ready(request.ready_path)
    _arm_alarm(request.rollout_budget_seconds, phase="rollout")
    _start_monitor_thread(request.rss_soft_limit_bytes)

    from platoon.train.workers.protocol import dump_response
    from platoon.train.workers.run import run_rollout_request

    try:
        response = asyncio.run(
            _run_until_terminated(
                run_rollout_request(
                    request,
                    llm,
                    startup_seconds=startup_seconds,
                    extra_allow=extra_allow,
                    # The first call's marker, a file like readiness's, which the supervisor polls for.
                    on_first_call=partial(_signal_ready, request.first_call_path)
                    if request.signal_first_call
                    else None,
                )
            )
        )
        if _TERMINATION is None or _TERMINATION.reason is None:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        # Else the termination deadline stays armed: a rollout that had finished when the signal came writes its
        # response and exits within it.
        if response is not None:
            dump_response(response, request.response_path)
    finally:
        # What the rollout started in sessions of its own and left running would outlive the worker.
        ended = _end_started_process_groups(PROCESS_GROUP_GRACE_SECONDS)
        if ended:
            print(f"[rollout-worker] ended {len(ended)} process group(s) the rollout left running: {ended}", flush=True)
    return EXIT_TERMINATED if response is None else 0


#: Exit code of a worker that stopped without its rollout's response: in practice the stop ends with the process
#: group's SIGKILL (``-9``), and the diagnostic marker says ``phase=terminated``.
EXIT_TERMINATED = 128 + signal.SIGTERM


class _Termination:
    """A worker's one-way switch from running its rollout to exiting (the module docstring's *Stopping*)."""

    def __init__(self, loop: asyncio.AbstractEventLoop, main: asyncio.Future[Any], outer: asyncio.Task[Any]) -> None:
        self.loop = loop
        self.main = main
        self.outer = outer
        self.reason: str | None = None
        self.started_at: float | None = None  # time.time() at which calls were refused
        self.keeps_result = False  # the rollout had finished before the stop began
        self.begun = asyncio.Event()
        self.task: asyncio.Task[None] | None = None
        self._lock = threading.Lock()

    def trigger(self, reason: str) -> None:
        """Start stopping; from the SIGTERM handler or the monitor thread, once.

        Only what is safe in a signal handler happens here: the deadline (an alarm, and the monitor thread's
        backstop) is armed at once, then the rest is handed to the event loop.
        """

        global _PHASE, _TERMINATE_BY
        if not self._lock.acquire(blocking=False):
            return
        try:
            if self.reason is not None:
                return
            self.reason = reason
            _PHASE = "terminated"
            _TERMINATION_DIAG.update(reason=reason, unwound=0)
            _TERMINATE_BY = time.monotonic() + TERMINATION_SECONDS
            signal.setitimer(signal.ITIMER_REAL, TERMINATION_SECONDS)
        finally:
            self._lock.release()
        try:
            self.loop.call_soon_threadsafe(self._begin)
        except RuntimeError:  # the loop is closed: the rollout is over, and the deadline bounds what is left
            pass

    def _begin(self) -> None:
        """On the loop: refuse every call, then delete the env sessions and unwind the rollout."""

        if self.begun.is_set():
            return
        self.started_at = time.time()
        self.keeps_result = self.main.done() and not self.main.cancelled()
        try:
            sessions = _refuse_calls()
        except Exception as exc:  # the deadline still ends the process
            print(f"[rollout-worker] could not refuse further calls: {type(exc).__name__}: {exc}", flush=True)
            sessions = []
        try:
            _refresh_diag_payload()
        except Exception:
            pass
        if self.keeps_result:
            action = "the rollout had finished, so its response is kept"
        else:
            action = f"deleting {len(sessions)} env session(s) and cancelling the rollout"
        print(
            f"[rollout-worker] terminating at {self.started_at:.6f} ({self.reason}): model, tool and env calls "
            f"refused; {action}",
            flush=True,
        )
        deletes = _delete_sessions_in_thread(sessions)
        # The sessions its descendants started, while they are still attached to it: a harness's own commands,
        # which would be orphaned when the rollout's cleanups end that harness.
        try:
            register_descendant_groups()
        except Exception:
            pass
        if not self.keeps_result:
            self.task = self.loop.create_task(self._unwind(deletes))
        self.begun.set()

    async def _unwind(self, deletes: threading.Thread) -> None:
        """Cancel every task of the rollout until it has unwound, let its cleanups finish, and end the process groups
        it started (each within the deadline), then end the process."""

        unwound, unwound_in, unfinished = False, 0.0, 0
        loop = asyncio.get_running_loop()
        try:
            give_up = _TERMINATE_BY - PROCESS_GROUP_GRACE_SECONDS - EXIT_MARGIN_SECONDS
            wait = RECANCEL_AFTER_SECONDS  # first, let the rollout's own cleanups run undisturbed
            while not self.main.done() and time.monotonic() < give_up:
                spared = {asyncio.current_task(), self.outer, *cleanups(loop)}
                for task in asyncio.all_tasks():
                    if task not in spared and not task.done():
                        task.cancel()
                await asyncio.wait([self.main], timeout=max(0.0, min(wait, give_up - time.monotonic())))
                wait = RECANCEL_SECONDS
            unwound = self.main.done()
            unwound_in = time.time() - (self.started_at or time.time())
            # Its cleanups (env teardowns, the SIGTERM, grace and SIGKILL of a command it ran) were never cancelled:
            # they run to their end.
            while (running := cleanups(loop)) and time.monotonic() < give_up:
                await asyncio.wait(running, timeout=max(0.0, give_up - time.monotonic()))
            unfinished = len(cleanups(loop))
            # Sandboxes still registered: their teardown never began (the rollout dropped them) or did not end.
            # What is left after the deadline, the supervisor reaps by this worker's label.
            stopping = _stop_sandboxes(give_up - time.monotonic())
            if stopping is not None:
                await asyncio.wait([stopping], timeout=max(0.0, give_up - time.monotonic()))
            while deletes.is_alive() and time.monotonic() < give_up:
                await asyncio.sleep(0.05)
        finally:
            outcome = f"the rollout unwound in {unwound_in:.1f}s" if unwound else "the rollout did not unwind in time"
            if unfinished:
                print(f"[rollout-worker] {unfinished} of its cleanups did not finish in time", flush=True)
            # Synchronous, so no task of the rollout runs between the SIGTERM and the SIGKILL.
            grace = min(PROCESS_GROUP_GRACE_SECONDS, max(0.0, _TERMINATE_BY - EXIT_MARGIN_SECONDS - time.monotonic()))
            ended = _end_started_process_groups(grace)
            if ended:
                print(f"[rollout-worker] ended {len(ended)} process group(s) the rollout started: {ended}", flush=True)
            print(f"[rollout-worker] terminated: {outcome}; killing the process group", flush=True)
            _TERMINATION_DIAG["unwound"] = int(unwound)
            try:
                _refresh_diag_payload()
            except Exception:
                pass
            for stream in (sys.stdout, sys.stderr):
                try:
                    stream.flush()
                except Exception:
                    pass
            _kill_process_group()


def _refuse_calls() -> list[Any]:
    """Stop the rollout from reaching the model or an env server, or starting a sandbox, again; return the remote
    sessions it holds."""

    from platoon.envs.ors.cleanup import shut_down
    from platoon.gateway.core import default_gateway
    from platoon.sandbox import cleanup as sandbox_cleanup

    sandbox_cleanup.shut_down()
    try:
        gateway = default_gateway()
    except RuntimeError:  # no model bound
        gateway = None
    if gateway is not None:
        gateway.shut_down()
    return shut_down()


def _stop_sandboxes(timeout_s: float) -> asyncio.Future[int] | None:
    """Start stopping the sandboxes the rollout still holds, within ``timeout_s``; None when it holds none."""

    from platoon.sandbox import cleanup as sandbox_cleanup

    if timeout_s <= 0 or not sandbox_cleanup.live_sandboxes():
        return None
    print(f"[rollout-worker] stopping {len(sandbox_cleanup.live_sandboxes())} sandbox(es) left running", flush=True)
    return asyncio.ensure_future(sandbox_cleanup.stop_live(timeout_s))


def _delete_sessions_in_thread(sessions: list[Any]) -> threading.Thread:
    """Delete the worker's remote env sessions on a thread and loop of their own, so neither a rollout that holds
    its loop nor the cancellation of every task on it can stop the deletes."""

    from platoon.envs.ors.cleanup import delete_orphaned_sessions

    def run() -> None:
        try:
            asyncio.run(delete_orphaned_sessions(sessions, SIGTERM_CLEANUP_SECONDS))
        except Exception:
            pass

    thread = threading.Thread(target=run, name="rollout-worker-session-deletes", daemon=True)
    thread.start()
    return thread


async def _run_until_terminated(rollout: Coroutine[Any, Any, Any]) -> Any | None:
    """Run the rollout, stopping it on SIGTERM or its parent's death (the module docstring's *Stopping*).

    Handled here, not in library code: SIGTERM is the supervisor's straggler kill and, through PDEATHSIG, the
    trainer's death.  Returns the rollout's response; a stop ends the process unless the rollout had finished
    before it began, whose response is returned.
    """

    global _TERMINATION
    loop = asyncio.get_running_loop()
    main = asyncio.ensure_future(rollout)
    outer = asyncio.current_task()
    assert outer is not None
    termination = _Termination(loop, main, outer)
    _TERMINATION = termination
    # A plain handler, not loop.add_signal_handler: it runs between bytecodes even while the rollout holds the
    # loop, so the deadline is armed at once.
    previous = signal.signal(signal.SIGTERM, lambda _signum, _frame: termination.trigger("SIGTERM"))
    if previous is None:  # installed outside Python: put the default back
        previous = signal.SIG_DFL
    try:
        try:
            result = await main
        except asyncio.CancelledError:
            if termination.reason is None:
                raise
            result = None
        if termination.reason is None:
            return result
        await termination.begun.wait()
        if termination.keeps_result:
            return result
        if termination.task is not None:
            await termination.task  # ends the process
        return None
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    sys.exit(main())
