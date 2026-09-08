"""``python -m platoon.train.tinker.workers.entrypoint <request_path>``.

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
packages before this module's body runs, so ``platoon/train/tinker/__init__.py``
-- and therefore tinker and torch -- has already executed by the time ``main``
starts.  The invariants that must precede those imports (telemetry off, thread
pins, no sampling sidecar) and the pre-import startup deadline therefore live in
:mod:`platoon.train.tinker.worker_env`, which that package init calls first.
``main`` re-applies them so the requirement is also stated where it is used.
"""

from __future__ import annotations

import asyncio
import ctypes
import os
import signal
import sys
import threading
import time

_PR_SET_PDEATHSIG = 1

#: Refreshed by the monitor thread, written verbatim by the alarm handler.
_DIAG_PAYLOAD: bytes = b""
_DIAG_FD: int = -1
_PHASE = "startup"
_PHASE_STARTED = 0.0


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

    if os.name != "posix":
        return
    parent_pid = os.getppid()
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
    """Re-assert the invariants ``worker_env`` already applied.

    ``platoon/train/tinker/__init__.py`` calls the same function before this
    package's imports happen, which is the only point early enough to matter.
    Repeating it here is free and keeps a direct ``main()`` call correct.

    ``TINKER_TELEMETRY=0``
        An unpickled ``SamplingClient`` builds a shadow holder that reuses the
        trainer's session id and, with telemetry on (the default, and what
        production sets), can emit ``SESSION_END`` from any fatal path.  The
        gateway reads that as the client exiting and reclaims the session's
        models and slots: the run dies.  This single line is the most dangerous
        omission in the whole change, and it is invisible until it fires.

    thread pins
        Torch takes ``os.cpu_count()`` intra- and inter-op threads per process
        and nothing in the tree pins it.  At ~96 concurrent workers on the node
        that also hosts the ray head, the miles driver and the gateway for
        every node, that is thousands of threads and ~96 independent 96-way
        pools.  Pinned in both the parent's ``env=`` and here, because some
        libraries read these at import.

    ``TINKER_SUBPROCESS_SAMPLING`` removed
        The sampling sidecar spawns an interpreter that imports tinker but
        never platoon, so it cannot carry the ``extra='allow'`` patch and would
        silently drop ``routing_id`` (R3).
    """

    from platoon.train.tinker.worker_env import apply_rollout_worker_env_invariants

    # Already applied by ``platoon/train/tinker/__init__.py`` -- which
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
    from platoon.train.tinker.workers.protocol import encode_diag
    from platoon.train.tinker.workers.run import cell_guard_cumulative

    fields: dict[str, object] = {
        "phase": _PHASE,
        "elapsed": round(time.monotonic() - _PHASE_STARTED, 1),
        "pid": os.getpid(),
    }
    fields.update(cell_guard_cumulative(at_exit=True))
    fields["stuck_cells"] = _stuck_cell_prefixes()
    fields.update(extra)
    _DIAG_PAYLOAD = encode_diag(fields)


def _stuck_cell_prefixes(limit: int = 3) -> str:
    """A short quote of the model-authored source of each overdue PTC cell."""

    try:
        from platoon.openreward.cell_guard import registry, warn_seconds

        guard = registry()
        threshold = warn_seconds()
        now = time.monotonic()
        prefixes = []
        with guard._lock:  # noqa: SLF001 - reading the guard's own book, best effort
            records = list(guard._live.values())  # noqa: SLF001
        for record in records:
            if record.elapsed(now) >= threshold:
                prefixes.append(f"{record.source_prefix[:80]}@{record.elapsed(now):.0f}s")
        return " | ".join(prefixes[:limit]) or "-"
    except Exception:
        return "-"


def _alarm_handler(_signum, _frame) -> None:
    """Leave evidence, then kill the whole tree.

    Async-signal-safe by construction: one ``os.write`` of a payload the
    monitor thread already formatted, then ``killpg``.  Without the marker a
    hard timeout, an OOM kill and an external sweep are indistinguishable to
    the parent (``rc=-9``, no response) and the cell-guard counters of the very
    rollout that hung are lost -- so ``ptc_cells_stuck`` would read zero
    precisely when a cell was killed for hanging.
    """

    from platoon.train.tinker.workers.protocol import EXIT_HARD_TIMEOUT

    if _DIAG_FD >= 0 and _DIAG_PAYLOAD:
        try:
            os.write(_DIAG_FD, _DIAG_PAYLOAD)
        except OSError:
            pass
    try:
        os.killpg(os.getpgrp(), signal.SIGKILL)
    except Exception:
        os._exit(EXIT_HARD_TIMEOUT)


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


def _start_monitor_thread(
    rss_soft_limit_bytes: int, interval: float = MONITOR_INTERVAL_SECONDS
) -> threading.Thread:
    """Keep the diagnostic payload fresh, and bound gradual memory growth.

    Not ``RLIMIT_AS``: the bare import stack measures ~12.7 GB of address space
    against ~0.7 GB resident, so an address-space limit loose enough to import
    torch bounds nothing real and a tight one kills every worker at import.
    This bounds a *gradually* growing worker deterministically -- and says so
    in the marker, so the parent classifies ``rollout_worker_oom`` instead of
    inferring it from an anonymous ``rc=-9``.  A single huge allocation can
    still reach the kernel OOM killer; that is a stated limit, not a claim.
    """

    from platoon.train.tinker.workers.protocol import EXIT_RSS_LIMIT

    def _loop() -> None:
        while True:
            time.sleep(interval)
            rss = _read_vm_rss_bytes()
            try:
                _refresh_diag_payload(rss_bytes=rss)
            except Exception:
                pass
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
                os._exit(EXIT_RSS_LIMIT)

    thread = threading.Thread(target=_loop, name="rollout-worker-monitor", daemon=True)
    thread.start()
    return thread


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
    from platoon.train.tinker.worker_env import process_start_monotonic

    # Not time.monotonic(): by the time main() runs, `python -m
    # platoon.train.tinker.workers.entrypoint` has already spent 20-30 s
    # importing the stack while executing the package __init__.  Startup cost
    # must include that or the metric is a lie.
    started = process_start_monotonic()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        print("usage: python -m platoon.train.tinker.workers.entrypoint <request_path>", file=sys.stderr)
        return 2
    request_path = arguments[0]

    # 1. Own process group.  ``start_new_session=True`` already did setsid;
    #    this is defensive so a directly-invoked worker is still killable as a
    #    group (that group is what bounds a runaway PTC cell).
    try:
        os.setpgrp()
    except OSError:
        pass

    # 2. Die with the trainer.
    _install_parent_death_signal()

    # 3. Environment invariants.  Already applied by ``worker_env`` from the
    #    package init, which is the only point ahead of the tinker/torch
    #    imports; repeated here so the requirement is stated where it is used.
    _apply_environment_invariants()

    # 4. Load the request and unlink it immediately -- it carries the gateway
    #    API key inside the pickled sampling client.  The alarm is deliberately
    #    split into a startup phase and a rollout phase: a single alarm
    #    covering both would let time spent queueing or fighting a
    #    shared-filesystem stall silently eat the rollout's budget and turn a
    #    slow start into a hard timeout with no data.
    from platoon.train.tinker.workers.protocol import load_request

    request = load_request(request_path)
    try:
        os.unlink(request_path)
    except OSError:
        pass

    _open_diag(request.diag_path)
    signal.signal(signal.SIGALRM, _alarm_handler)
    # The pre-import deadline armed by ``worker_env`` (from the environment)
    # already covered the imports; re-arm the remainder under the richer
    # handler.  A worker launched without that environment gets its whole
    # deadline here, and one that already blew it dies immediately with an
    # attributable marker instead of silently eating the rollout's budget.
    global _PHASE, _PHASE_STARTED
    _PHASE, _PHASE_STARTED = "startup", started
    remaining = request.startup_budget_seconds - (time.monotonic() - started)
    _refresh_diag_payload()
    if remaining <= 0:
        _alarm_handler(signal.SIGALRM, None)
    signal.setitimer(signal.ITIMER_REAL, max(0.05, remaining))

    # 5. Now import the stack.  This module import applies the extra='allow'
    #    SampleResponse patch and registers the platoon renderers.
    import tinker  # noqa: E402  (must follow the proxy import)
    import torch  # noqa: E402

    from platoon.train.tinker import proxy

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

    # 6. Build the LLM.  NEVER register_tinker_llm(): it constructs a
    #    ServiceClient, i.e. a second gateway session per rollout.
    from platoon.train.tinker.workers.binding import FollowingSamplingClient

    binding = request.model
    llm = None
    if binding.model_name:
        sampling_client = FollowingSamplingClient(
            binding.binding_path,
            binding.bootstrap_blob,
            binding.follow_poll_seconds,
        )
        llm = proxy.build_tinker_llm(
            model_name=binding.model_name,
            renderer_name=binding.renderer_name,
            sampling_client=sampling_client,
            context_window_length=binding.context_window_length,
            renderer_kwargs=binding.renderer_kwargs,
            max_tokens=binding.max_tokens,
            temperature=binding.temperature,
            top_k=binding.top_k,
            top_p=binding.top_p,
            seed=binding.seed,
        )
        llm.set_version(binding.checkpoint_version)
        llm.rewrite_litellm_custom_providers()
    else:
        # A binding with no model name deliberately registers no sampling
        # provider.  Used by CPU tests and by fault injection; a rollout that
        # then makes a model call fails loudly at the litellm boundary rather
        # than silently reaching some other provider.
        logger_message = "[rollout-worker] no model bound; running without a sampling provider"
        print(logger_message, flush=True)

    startup_seconds = time.monotonic() - started
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

    # 7. Ready: release the parent's admission slot, then swap to the rollout
    #    deadline so the rollout gets its full budget from the moment it begins.
    _signal_ready(request.ready_path)
    _arm_alarm(request.rollout_budget_seconds, phase="rollout")
    _start_monitor_thread(request.rss_soft_limit_bytes)

    from platoon.train.tinker.workers.protocol import dump_response
    from platoon.train.tinker.workers.run import run_rollout_request

    response = asyncio.run(
        run_rollout_request(
            request,
            llm,
            startup_seconds=startup_seconds,
            extra_allow=extra_allow,
        )
    )
    signal.setitimer(signal.ITIMER_REAL, 0.0)
    dump_response(response, request.response_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
