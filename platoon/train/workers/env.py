"""Environment invariants a rollout worker must set before anything loads.

Ordering is the whole point, and it is genuinely hard to get right:
``python -m platoon.train.workers.entrypoint`` necessarily executes
``platoon/train/workers/__init__.py`` *first*, and that module imports the
protocol, which imports tinker and torch.  By the time the entrypoint's own
``main()`` runs, much of the stack is already loaded.

So this module is stdlib-only, imports nothing from platoon, and is called from
the top of ``platoon/train/workers/__init__.py``, and of
``platoon/train/tinker/__init__.py``, which the entrypoint's module name before
platoon 0.5.0 (``platoon.train.tinker.workers.entrypoint``, still runnable until
0.6.0) executes first.  It is a **no-op in the
trainer**: it applies only when the process is recognisably a rollout worker,
either because the supervisor set ``PLATOON_TINKER_ROLLOUT_WORKER=1`` in the
child environment or because ``__main__`` is the worker entrypoint itself (a
hand-run worker, or a supervisor whose environment handling regressed).
Importing this module runs ``platoon/train/workers/__init__.py`` first all the
same, so it is an early import, not a cheap one: it loads the protocol, and
with it tinker and torch.

What it sets, and why each one matters
======================================

``TINKER_TELEMETRY=0``
    Unpickling a ``SamplingClient`` builds a *shadow* client holder that reuses
    the trainer's session id and, with telemetry at its default of on, can emit
    ``SESSION_END`` from any fatal path.  The gateway reads that as the client
    exiting: it reclaims the session's models and slots, and the training state
    dies with the lease.  A trainer's environment may set
    ``TINKER_TELEMETRY=1``, so without this a worker would inherit exactly the
    dangerous value.

``OMP_NUM_THREADS`` and friends ``=1``
    Torch takes ``os.cpu_count()`` intra- *and* inter-op threads per process
    and nothing in the tree pins it.  At tens of workers on a many-core host
    that may also run the trainer and the gateway, that is thousands of threads
    and one full-width pool per worker.  These are read by OpenMP/MKL at first
    use, which is why they must be set before the import rather than after
    it.

``TINKER_SUBPROCESS_SAMPLING`` removed
    The sampling sidecar spawns an interpreter that imports tinker but never
    platoon, so it cannot carry the ``extra='allow'`` patch and would silently
    drop the gateway's ``routing_id`` -- a quietly worse gradient, with no
    error anywhere.  The proxy asserts this variable is unset **at import**,
    which is precisely why clearing it has to happen this early.

``CUDA_VISIBLE_DEVICES=`` and ``PYTORCH_NVML_BASED_CUDA_CHECK=1``
    transformers calls ``torch.cuda.is_available()`` at import
    (``transformers.integrations.flash_attention``, which the proxy's
    ``AutoProcessor`` pulls in).  Wherever an NVIDIA driver is installed,
    that call starts the CUDA driver, and the driver a thread of its own,
    also with the GPUs hidden; asking NVML instead starts it too on recent
    drivers (595: ``nvmlInit``).  With both variables torch finds no GPU in
    the environment and touches neither.  The driver's thread would leave the
    fork server unable to fork, so every worker would start by exec, and
    every exec'd worker would pay the driver's start for a GPU it never uses.
    A worker needs no GPU: it samples through the training service and runs
    its environment in sandboxes, which get their GPUs from their provider,
    not from the worker's environment.  Both are read when ``is_available()``
    is called, so setting them before the imports suffices.
"""

from __future__ import annotations

import os
import signal
import sys
import time

__all__ = [
    "STARTUP_BUDGET_ENV",
    "DIAG_PATH_ENV",
    "apply_rollout_worker_env_invariants",
    "arm_startup_alarm_from_env",
    "in_rollout_worker_process",
    "process_start_monotonic",
]

WORKER_ENV_MARKER = "PLATOON_TINKER_ROLLOUT_WORKER"
#: The supervisor passes the startup budget and the diagnostic path in the
#: environment because the deadline has to exist *before* the request can be
#: read: reading it requires unpickling, which requires the very imports the
#: deadline is meant to bound.
STARTUP_BUDGET_ENV = "PLATOON_ROLLOUT_WORKER_STARTUP_BUDGET"
DIAG_PATH_ENV = "PLATOON_ROLLOUT_WORKER_DIAG"
ENTRYPOINT_MODULE = "platoon.train.workers.entrypoint"
#: The entrypoint's module, and its module before platoon 0.5.0, a shim that runs it until platoon 0.6.0.
_ENTRYPOINT_MODULES = (ENTRYPOINT_MODULE, "platoon.train.tinker.workers.entrypoint")
_ENTRYPOINT_SUFFIX = tuple(os.path.join(*module.split(".")) + ".py" for module in _ENTRYPOINT_MODULES)

THREAD_PIN_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
#: No GPU, and torch asks NVML: ``torch.cuda.is_available()`` then answers from the environment, without the driver.
NO_CUDA_DRIVER = {"CUDA_VISIBLE_DEVICES": "", "PYTORCH_NVML_BASED_CUDA_CHECK": "1"}


def in_rollout_worker_process() -> bool:
    """Whether this interpreter exists to run exactly one rollout."""

    if os.environ.get(WORKER_ENV_MARKER) == "1":
        return True
    # ``runpy`` imports the parent packages *before* it rewrites ``sys.argv``,
    # so during ``platoon/train/workers/__init__.py`` -- the moment that matters
    # -- ``sys.argv[0]`` is still ``"-m"``.  ``sys.orig_argv`` is the command
    # line as the interpreter received it and is already correct here.
    original = list(getattr(sys, "orig_argv", ()) or ())
    if any(module in original for module in _ENTRYPOINT_MODULES):
        for index, argument in enumerate(original[:-1]):
            if argument == "-m" and original[index + 1] in _ENTRYPOINT_MODULES:
                return True
    argv0 = sys.argv[0] if sys.argv else ""
    return bool(argv0) and os.path.abspath(argv0).endswith(_ENTRYPOINT_SUFFIX)


def apply_rollout_worker_env_invariants(force: bool = False) -> bool:
    """Set the worker invariants if this is a worker.  Idempotent.

    Returns whether they were applied, so the entrypoint can report it.
    """

    if not force and not in_rollout_worker_process():
        return False
    os.environ["TINKER_TELEMETRY"] = "0"
    for variable in THREAD_PIN_VARIABLES:
        os.environ[variable] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.pop("TINKER_SUBPROCESS_SAMPLING", None)
    os.environ.update(NO_CUDA_DRIVER)
    os.environ[WORKER_ENV_MARKER] = "1"
    arm_startup_alarm_from_env()
    return True


def process_start_monotonic() -> float:
    """When this interpreter actually started, on the monotonic clock.

    ``time.monotonic()`` taken inside ``main()`` is far too late to measure
    startup: ``python -m platoon.train.workers.entrypoint`` spends
    20-30 s importing the tinker/torch stack while executing the *package*
    __init__, long before ``main`` runs.  Reading the kernel's own start time
    makes ``rollout_worker_startup_seconds`` mean what it says.
    """

    try:
        with open("/proc/self/stat", "rb") as handle:
            raw = handle.read()
        tail = raw[raw.rindex(b")") + 2 :].split()
        ticks = int(tail[19])  # field 22: starttime, in clock ticks since boot
        clock = os.sysconf("SC_CLK_TCK")
        with open("/proc/uptime", "rb") as handle:
            uptime = float(handle.read().split()[0])
        age = uptime - (ticks / float(clock))
        if age >= 0:
            return time.monotonic() - age
    except Exception:
        pass
    return _IMPORTED_AT


_IMPORTED_AT = time.monotonic()


def _startup_alarm_handler(_signum, _frame) -> None:
    """Kill a worker that hung before it could install the richer handler.

    Writes the same ``phase=startup`` marker the full handler would, so the
    parent classifies this as a startup overrun rather than as an anonymous
    crash indistinguishable from an OOM or an external sweep.
    """

    path = os.environ.get(DIAG_PATH_ENV)
    if path:
        elapsed = time.monotonic() - process_start_monotonic()
        payload = f"phase=startup\telapsed={elapsed:.1f}\tpid={os.getpid()}\tpre_import=1\n"
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            os.write(descriptor, payload.encode())
            os.close(descriptor)
        except OSError:
            pass
    try:
        os.killpg(os.getpgrp(), signal.SIGKILL)
    except Exception:
        os._exit(88)


def arm_startup_alarm_from_env() -> float:
    """Arm the pre-import startup deadline, if the supervisor set one.

    Returns the budget it armed (0.0 when there was none).  Best effort: a
    worker without the environment falls back to the deadline ``main()``
    computes from the request once it can read it.
    """

    raw = os.environ.get(STARTUP_BUDGET_ENV)
    if not raw:
        return 0.0
    try:
        budget = float(raw)
    except ValueError:
        return 0.0
    if budget <= 0:
        return 0.0
    try:
        signal.signal(signal.SIGALRM, _startup_alarm_handler)
        remaining = max(0.05, budget - (time.monotonic() - process_start_monotonic()))
        signal.setitimer(signal.ITIMER_REAL, remaining)
    except (ValueError, OSError):
        return 0.0
    # Consume the budget so it is not inherited.  The worker's whole descendant
    # tree -- MCP bridges, and anything a programmatic-tool-calling cell spawns
    # -- inherits this environment, and any descendant that imports
    # ``platoon.train.workers`` would run this function too, arm its own deadline,
    # and on expiry ``killpg`` the group: the entire rollout tree, mid-rollout,
    # with a ``phase=startup`` marker blaming the worker's imports.  The
    # entrypoint does not need the variable again -- it re-arms from
    # ``request.startup_budget_seconds`` once it can read the request -- and the
    # itimer armed above keeps running regardless.  ``DIAG_PATH_ENV`` stays: the
    # handler reads it when it fires.
    os.environ.pop(STARTUP_BUDGET_ENV, None)
    return budget
