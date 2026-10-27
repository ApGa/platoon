"""A worker module for the fork server's tests (``tests/train/workers/test_worker_forkserver.py``).

Run as ``python -m forkserver_stub_worker <mode> [args]``, or forked by a fork server that runs it the same way.

Modes:
    describe <path>       write what this process looks like (ids, descriptors, signals, environment, generators)
    exit <code>           exit with ``code``
    raise                 raise an uncaught exception
    sleep <path>          write its pid to ``path``, then sleep
    thread <path>         start a non-daemon thread that writes ``path`` after 0.5 s, and return
    atexit <path>         register an ``atexit`` handler that writes ``path``, and return
    state <path> [class]  write what the fork server left in this process before it ran anything: its threads, the
                          descriptors it holds besides stdio and libffi's (``fork_safe_descriptor``), its children, the
                          modules imported, what ``preload_stubs.builds`` built and its environment's PRELOAD_STUB* and
                          OPENREWARD* variables; with a ``module:Class``, the splits in that class's task cache

It imports ctypes, as every worker's stack does. On Python 3.14 that import makes a libffi closure, and libffi then
holds the file with its trampolines open (the interpreter itself, in uv's builds of Python): a forked worker inherits
that descriptor from its server, which imports ctypes too, and an exec'd worker opens its own. On 3.12 and 3.13 a
process's first ctypes callback opens it.
"""

from __future__ import annotations

import atexit
import ctypes  # noqa: F401  (the module docstring says why)
import json
import os
import random
import signal
import sys
import threading
import time

#: What a fork server built ahead (``prepare``), which the workers it forks inherit.
PREPARED: dict[str, str] = {}
#: Generators made at import, as libraries make their own: one seeded from the OS (a fresh interpreter seeds it anew),
#: one seeded with a value and one given a state (each the same in every worker).
OS_SEEDED = random.Random()
SEEDED = random.Random(1234)
RESTORED = random.Random()
RESTORED.setstate(random.Random(99).getstate())


def prepare(value: str) -> None:
    PREPARED["value"] = value


def fail_to_prepare() -> None:
    raise RuntimeError("the stub cannot prepare")


def _write(path: str, text: str) -> None:
    with open(path, "w") as handle:
        handle.write(text)


def _describe() -> dict:
    listing = os.listdir("/proc/self/fd")  # holds one descriptor of its own while it lists
    facts: dict = {
        "pid": os.getpid(),
        "pgid": os.getpgrp(),
        "sid": os.getsid(0),
        "descriptors": len(listing),
        "stdin": os.readlink("/proc/self/fd/0"),
        "stdout": os.readlink("/proc/self/fd/1"),
        "stderr": os.readlink("/proc/self/fd/2"),
        "cwd": os.getcwd(),
        "argv0": os.path.basename(sys.argv[0]),
        "argv": sys.argv[1:],
        "name": __name__,
        "env": dict(os.environ),
        "signals": {
            name: repr(signal.getsignal(getattr(signal, name)))
            for name in ("SIGINT", "SIGTERM", "SIGCHLD", "SIGPIPE", "SIGALRM", "SIGHUP")
        },
        "blocked": sorted(int(s) for s in signal.pthread_sigmask(signal.SIG_BLOCK, [])),
        "threads": len(os.listdir("/proc/self/task")),
        "random": random.random(),
    }
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        facts["numpy"] = float(numpy.random.random())
    imported = sys.modules.get("forkserver_stub_worker")  # the module the server imported (this one runs as __main__)
    facts["prepared"] = imported.PREPARED.get("value") if imported is not None else None
    if imported is not None:
        facts["generators"] = {
            "os_seeded": imported.OS_SEEDED.random(),
            "seeded": imported.SEEDED.random(),
            "restored": imported.RESTORED.random(),
        }
    facts["random_seed"] = random.Random.seed.__qualname__
    umask = os.umask(0)
    os.umask(umask)
    facts["umask"] = umask
    return facts


def _state(class_ref: str | None) -> dict:
    from platoon.utils.forkserver import child_processes, fork_safe_descriptor, open_descriptors

    descriptors = {fd: what for fd, what in open_descriptors().items() if fd > 2 and not fork_safe_descriptor(fd, what)}
    builds = sys.modules.get("preload_stubs.builds")
    facts: dict = {
        "threads": threading.active_count(),
        "tasks": len(os.listdir("/proc/self/task")),
        "descriptors": {str(fd): what for fd, what in descriptors.items()},
        "children": sorted(child_processes()),
        "modules": sorted(sys.modules),
        "built": dict(builds.BUILT) if builds is not None else None,
        "environ": {name: os.environ[name] for name in os.environ if name.startswith(("PRELOAD_STUB", "OPENREWARD"))},
    }
    if class_ref:
        module, _, name = class_ref.partition(":")
        cls = getattr(sys.modules[module], name) if module in sys.modules else None
        facts["task_cache"] = sorted((cls.__dict__.get("_tasks_cache") or {}) if cls is not None else ())
    return facts


def main() -> int:
    mode = sys.argv[1]
    if mode == "describe":
        _write(sys.argv[2], json.dumps(_describe()))
        return 0
    if mode == "state":
        _write(sys.argv[2], json.dumps(_state(sys.argv[3] if len(sys.argv) > 3 else None)))
        return 0
    if mode == "exit":
        return int(sys.argv[2])
    if mode == "raise":
        raise RuntimeError("the stub worker raised")
    if mode == "sleep":
        _write(sys.argv[2], str(os.getpid()))
        time.sleep(600)
        return 0
    if mode == "thread":
        path = sys.argv[2]
        threading.Thread(target=lambda: (time.sleep(0.5), _write(path, "thread"))).start()
        return 0
    if mode == "atexit":
        path = sys.argv[2]
        atexit.register(_write, path, "atexit")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
