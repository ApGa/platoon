"""A fork server: a process that imports a worker's modules once, then forks one worker process per request.

The rollout worker supervisor uses it (``worker_start_method: forkserver``, :mod:`platoon.train.tinker.workers`): an
exec'd rollout worker spends most of its start importing the tinker / torch / transformers stack, 17-24 s alone and
about a minute when 64 start at once on one node, which they contend for, CPUs and the shared filesystem the venv
lives on alike.  Forked from a server that has imported that stack, a worker starts in a few seconds.

**A forked worker is the process an exec'd one would be.**  ``python -m <module> <argv>`` with the environment, the
working directory and the log file the caller gives: right after the fork the child leads a session of its own,
takes ``/dev/null`` as stdin and the log file as stdout and stderr, closes the server's descriptors, enters the
working directory and replaces its environment with the one given.  It reseeds numpy's and torch's global
generators and every ``random.Random`` the server's imports seeded from the OS (libraries make their own at import:
sympy's, datasets' fingerprint generator, aiohttp's websocket masks), as a fresh interpreter would seed them
(Python's global ``random`` reseeds itself at every fork; a generator seeded with a value keeps it), calls the
server's ``child_init`` functions (what a package init that the server already ran would have done in a fresh
interpreter, with this environment), then runs the module as ``__main__``, as ``python -m`` does, and exits as the
interpreter does: it waits for its non-daemon threads, runs the ``atexit`` handlers and flushes.  What it inherits
from the server is what the imports left behind, and the server's hash seed.

The server may also build, once, what each worker would otherwise build for itself (``prepare``): each worker it
forks then holds its own copy, copy-on-write.  A build that fails is reported, and the workers build it themselves.

**The server is safe to fork from because it does nothing else.**  It runs no thread but its own (it refuses to
fork while any other runs, but jemalloc's background purger, whose own fork handlers make it safe), no event loop,
and only waits on its control socket and on its children.  It leads a session of its own and dies with the process
that started it (``PR_SET_PDEATHSIG``, and a check of its parent every second); the workers it forked, whose parent
it is, then get whatever death signal they set for themselves.  It reaps its children and reports each exit status.

**Nothing it imports may start the CUDA driver**, whose threads (``cuda00001400006``, ...) run from then on, and
which a forked child could not use anyway.  Where an NVIDIA driver is installed, ``torch.cuda.is_available()``
starts it, GPU visible or not, and transformers asks at import (``transformers.integrations.flash_attention``, which
``AutoProcessor`` pulls in).  With ``PYTORCH_NVML_BASED_CUDA_CHECK=1`` torch asks NVML instead, which on recent
drivers (595) starts a driver thread too.  Only an environment that both hides the GPUs (``CUDA_VISIBLE_DEVICES=``)
and asks NVML lets torch answer from the environment alone: the rollout worker supervisor gives its server, and its
workers, that environment.  A server whose imports start the driver does not serve, and says so.

**The caller's side** (:class:`ForkServer`) works in an asyncio loop.  The server opens a pidfd for each child
before the child can be reaped (so the pid it names cannot have been recycled) and sends it along; a
:class:`ForkedWorker` stands in for an ``asyncio.subprocess.Process`` (``pid``, ``returncode``, ``wait()``).  When
the server is gone, its workers' exits are read from their pidfds, with no exit status.  The server is a plain
``subprocess.Popen``, not an asyncio one, whose transport would kill it when collected while workers still run.

**The caller gives up on the server when the server is silent, not when its own event loop was late.**  On a node
whose CPUs are all busy, the caller's loop can be held up for minutes, and a timer that fires when it runs again says
nothing about the server, which may have answered meanwhile.  So before a deadline is judged, the caller reads what
the server sent: an answer that came is used, however late, and the worker it names runs.  The server answers the
requests one at a time, in the order they were sent, so a fork request fails only once the server has answered no
request at all for :data:`SPAWN_TIMEOUT_S` (one that answers the requests before it is working through the queue), or
at once when the server answers a later request (its answer was lost).  A server whose imports outlast their deadline
(:meth:`ForkServer.ready`) is not killed for it: it serves once it has imported them, unless its caller closes it
(:meth:`ForkServer.start` does).

The control socket is a ``SOCK_SEQPACKET`` pair: one JSON object per message, a pidfd riding along with ``forked``.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import contextlib
import ctypes
import gc
import importlib
import itertools
import json
import logging
import os
import runpy
import select
import selectors
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NoReturn

logger = logging.getLogger(__name__)

__all__ = ["FORK_SAFE_THREADS", "ForkServer", "ForkServerError", "ForkedWorker", "unsafe_threads"]

SERVER_MODULE = "platoon.utils.forkserver"
#: Threads the server may run next to its own when it forks (by name): jemalloc's background purging thread, which
#: libraries built with jemalloc start; jemalloc's fork handlers take its locks around a fork.
FORK_SAFE_THREADS = frozenset({"jemalloc_bg_thd"})
MAX_MESSAGE_BYTES = 4 << 20
#: How long a fork request waits while the server answers no request.  The server answers the requests one at a time,
#: in the order they were sent, so one that keeps answering is working through the queue; the reports of its workers'
#: exits do not count.
SPAWN_TIMEOUT_S = 60.0
#: How long the server waits for a forked child to lead its own session; one that has not is killed, and its
#: request refused, so the server never waits on a child.
CHILD_START_TIMEOUT_S = 20.0
_PR_SET_PDEATHSIG = 1
#: The generators the server's imports and builds seeded from the OS (``random.Random()`` seeds itself so), which a
#: fresh interpreter would seed differently in every worker: each forked worker reseeds them (:func:`_reseed`).
_OS_SEEDED: list[Any] = []


class ForkServerError(RuntimeError):
    """The fork server did not start this worker; nothing of the worker ran.  ``permanent``: the server cannot
    serve at all (a module did not import, a thread runs in it), so starting another would not help."""

    def __init__(self, message: str, *, permanent: bool = False) -> None:
        super().__init__(message)
        self.permanent = permanent


# ------------------------------------------------------------------------------------------------- the server side


def unsafe_threads() -> list[str]:
    """The names of the threads that run in this process besides the calling one and :data:`FORK_SAFE_THREADS`."""

    return sorted(name for _, name in _unsafe_tasks())


def _unsafe_tasks() -> list[tuple[int, str]]:
    """``(native id, name)`` of each thread :func:`unsafe_threads` names."""

    own = threading.get_native_id()
    tasks = []
    for tid in os.listdir("/proc/self/task"):
        if int(tid) == own:
            continue
        try:
            with open(f"/proc/self/task/{tid}/comm") as handle:
                name = handle.read().strip()
        except OSError:
            continue  # it ended
        if name not in FORK_SAFE_THREADS:
            tasks.append((int(tid), name))
    return sorted(tasks, key=lambda task: (task[1], task[0]))


def _describe_threads(tasks: Sequence[tuple[int, str]]) -> str:
    """What each thread of ``tasks`` (:func:`_unsafe_tasks`) is, as far as this process can tell."""

    python = {thread.native_id: thread for thread in threading.enumerate()}
    described = []
    for tid, name in tasks:
        thread = python.get(tid)
        if thread is not None:
            target = getattr(thread, "_target", None)
            runs = ".".join(filter(None, (getattr(target, "__module__", None), getattr(target, "__qualname__", None))))
            described.append(f"{name!r} (the Python thread {thread.name!r}{f' running {runs}' if runs else ''})")
        elif name.startswith("cuda"):
            if os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK") == "1":
                how = "with a CUDA call other than torch.cuda.is_available(), which this environment answers without it"
            else:
                how = (
                    "as torch.cuda.is_available() does wherever a driver is installed, unless the environment hides "
                    "the GPUs and has torch ask NVML: CUDA_VISIBLE_DEVICES= PYTORCH_NVML_BASED_CUDA_CHECK=1"
                )
            described.append(f"{name!r} (the CUDA driver's: the imports or a prepare step started it, {how})")
        else:
            described.append(f"{name!r} (a thread a native library started)")
    return ", ".join(described)


def _resolve(ref: str) -> Callable[[], Any]:
    module, _, name = ref.partition(":")
    target: Any = importlib.import_module(module)
    for part in name.split("."):
        target = getattr(target, part)
    return target


def _die_with(parent: int) -> None:
    """SIGKILL this server when ``parent`` dies."""

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    except (AttributeError, OSError):
        pass
    if os.getppid() != parent:  # it died before the prctl
        os._exit(1)


def _send(sock: socket.socket, message: Mapping[str, Any], fds: Sequence[int] = ()) -> None:
    payload = json.dumps(message).encode()
    if fds:
        socket.send_fds(sock, [payload], list(fds))
    else:
        sock.sendall(payload)


def serve(
    control_fd: int,
    module: str,
    preload: Sequence[str],
    child_init: Sequence[str],
    prepare: Sequence[Mapping[str, Any]] = (),
) -> int:
    """Import ``preload`` and ``module``, call each ``prepare`` (``{"ref": "module:function", "kwargs": {...}}``,
    what the workers would otherwise each build: a failure is reported and their workers build it), say hello on
    the control socket, then fork one worker per request until the socket closes, and exit once the workers have
    exited.  It answers each request before it reads the next, which the caller relies on (:class:`ForkServer`)."""

    parent = os.getppid()
    _die_with(parent)
    sock = socket.socket(fileno=control_fd)
    began = time.monotonic()
    error = None
    initializers: list[Callable[[], Any]] = []
    prepare_errors = []
    with _tracking_os_seeded_generators():
        try:
            for name in dict.fromkeys([*preload, module]):
                importlib.import_module(name)
            initializers = [_resolve(ref) for ref in child_init]
        except BaseException as exc:  # noqa: BLE001 - reported; the caller starts its workers otherwise
            error = f"importing the worker's modules failed: {type(exc).__name__}: {exc}"
            traceback.print_exc()
        for item in prepare if error is None else ():
            try:
                _resolve(item["ref"])(**item.get("kwargs", {}))
            except Exception as exc:  # noqa: BLE001 - the workers build it themselves
                prepare_errors.append(f"{item['ref']}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
    gc.collect()
    gc.freeze()  # imported objects are never collected: the children's collections leave their pages shared
    if error is None and (unsafe := _unsafe_tasks()):
        error = (
            "the fork server runs other threads after its imports, and a child forked next to a running thread can "
            f"deadlock: {_describe_threads(unsafe)}"
        )
    if error is not None:
        print(f"fork server: {error}", file=sys.stderr)  # its log says why too, should nobody read its hello
    hello = {"op": "hello", "pid": os.getpid(), "seconds": round(time.monotonic() - began, 3), "error": error}
    _send(sock, {**hello, "prepare_errors": prepare_errors})
    if error is not None:
        return 1
    # A thread in FORK_SAFE_THREADS is allowed; the interpreter's warning about forking next to it is not news.
    warnings.filterwarnings("ignore", message=r".*use of fork\(\) may lead to deadlocks", category=DeprecationWarning)
    wake_r, wake_w = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
    signal.set_wakeup_fd(wake_w)
    signal.signal(signal.SIGCHLD, lambda _signum, _frame: None)
    selector = selectors.DefaultSelector()
    selector.register(sock, selectors.EVENT_READ)
    selector.register(wake_r, selectors.EVENT_READ)
    children: set[int] = set()
    accepting = True
    while accepting or children:
        for key, _ in selector.select(timeout=1.0):
            if key.fileobj is sock:
                try:
                    payload = sock.recv(MAX_MESSAGE_BYTES)
                except OSError:
                    payload = b""
                if not payload:  # the caller is gone: no new workers; the running ones finish
                    accepting = False
                    selector.unregister(sock)
                    continue
                try:
                    message = json.loads(payload)
                except ValueError:
                    print(f"fork server: an unreadable request: {payload[:200]!r}", file=sys.stderr)
                    continue
                if message.get("op") == "fork":
                    own = (sock.fileno(), wake_r, wake_w, selector.fileno())
                    pid = _fork(message, sock, module, initializers, close_in_child=own)
                    if pid is not None:
                        children.add(pid)
            else:
                with contextlib.suppress(BlockingIOError):
                    while os.read(wake_r, 4096):
                        pass
        _reap(sock, children, report=accepting)
        if os.getppid() != parent:
            os._exit(1)  # the caller died and PDEATHSIG did not end this process
    return 0


def _reap(sock: socket.socket, children: set[int], *, report: bool) -> None:
    while children:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            children.clear()
            return
        if pid == 0:
            return
        children.discard(pid)
        if report:
            with contextlib.suppress(OSError):
                _send(sock, {"op": "exited", "pid": pid, "status": status})


def _fork(
    message: dict[str, Any],
    sock: socket.socket,
    module: str,
    initializers: Sequence[Callable[[], Any]],
    *,
    close_in_child: Sequence[int],
) -> int | None:
    token = message.get("token")
    if unsafe := _unsafe_tasks():
        _refuse(sock, token, f"the fork server runs other threads: {_describe_threads(unsafe)}")
        return None
    sync_r, sync_w = os.pipe()
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    try:
        pid = os.fork()
    except OSError as exc:
        os.close(sync_r)
        os.close(sync_w)
        _refuse(sock, token, f"fork failed: {exc}")
        return None
    if pid == 0:
        os.close(sync_r)
        _become_worker(message, module, initializers, sync_w, close_in_child)
    os.close(sync_w)
    poller = select.poll()
    poller.register(sync_r, select.POLLIN)
    started = bool(poller.poll(CHILD_START_TIMEOUT_S * 1000))  # once the child leads its own session, or is gone
    os.close(sync_r)
    if not started:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)
        print(f"fork server: child {pid} did not start in {CHILD_START_TIMEOUT_S:.0f}s; killed it", file=sys.stderr)
        _refuse(sock, token, f"the forked child did not start in {CHILD_START_TIMEOUT_S:.0f}s")
        return pid  # still reaped as a child, and its exit reported (no one waits for it)
    pidfd = os.pidfd_open(pid)  # before this server can reap it, so the pidfd names exactly this child
    try:
        _send(sock, {"op": "forked", "token": token, "pid": pid}, [pidfd])
    except OSError:  # the caller is gone: nothing would supervise this worker
        with contextlib.suppress(OSError):
            os.killpg(pid, signal.SIGKILL)
    finally:
        os.close(pidfd)
    return pid


def _refuse(sock: socket.socket, token: Any, reason: str) -> None:
    with contextlib.suppress(OSError):
        _send(sock, {"op": "refused", "token": token, "reason": reason})


def _become_worker(
    message: dict[str, Any],
    module: str,
    initializers: Sequence[Callable[[], Any]],
    sync_w: int,
    close: Sequence[int],
) -> NoReturn:
    """In the forked child: become the process ``python -m module argv`` would be, then run ``module``."""

    code: int = 1
    try:
        os.setsid()
        os.write(sync_w, b"1")
        os.close(sync_w)
        signal.set_wakeup_fd(-1)
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        signal.pthread_sigmask(signal.SIG_SETMASK, ())
        for fd in close:
            with contextlib.suppress(OSError):
                os.close(fd)
        _redirect(message.get("log_path"))
        os.chdir(message["cwd"])
        os.environ.clear()
        os.environ.update(message["env"])
        _reseed()
        sys.argv = [module, *message["argv"]]
        for initialize in initializers:
            initialize()
        try:
            runpy.run_module(module, run_name="__main__", alter_sys=True)
            code = 0
        except SystemExit as exc:
            code = _exit_code(exc.code)
    except BaseException:  # noqa: BLE001 - what the interpreter prints for an uncaught exception
        with contextlib.suppress(Exception):
            traceback.print_exc()
        code = 1
    finally:
        _exit_as_interpreter(code)


def _redirect(log_path: str | None) -> None:
    stdin = os.open(os.devnull, os.O_RDONLY)
    if log_path:
        out = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o666)
    else:
        out = os.open(os.devnull, os.O_WRONLY)
    os.dup2(stdin, 0)
    os.dup2(out, 1)
    os.dup2(out, 2)
    for fd in {stdin, out} - {0, 1, 2}:
        os.close(fd)


@contextlib.contextmanager
def _tracking_os_seeded_generators():
    """Keep, in :data:`_OS_SEEDED`, every ``random.Random`` that was last seeded from the OS (``seed()`` without a
    value, which ``random.Random()`` calls) while the block runs: the server's imports and builds.  A generator seeded
    with a value, or given a state, keeps it in every worker, as it would in an exec'd one.  Python's global
    generator is left out: ``random`` reseeds it after every fork itself."""

    import random
    import weakref

    seeded: weakref.WeakSet[Any] = weakref.WeakSet()
    original_seed, original_setstate = random.Random.seed, random.Random.setstate

    def seed(self, a=None, version=2):
        original_seed(self, a, version)
        if a is None:
            seeded.add(self)
        else:
            seeded.discard(self)

    def setstate(self, state):
        original_setstate(self, state)
        seeded.discard(self)

    random.Random.seed, random.Random.setstate = seed, setstate  # type: ignore[method-assign]
    try:
        yield
    finally:
        random.Random.seed, random.Random.setstate = original_seed, original_setstate  # type: ignore[method-assign]
        global_generator = getattr(random, "_inst", None)
        _OS_SEEDED[:] = [generator for generator in seeded if generator is not global_generator]


def _reseed() -> None:
    """New seeds for the generators a fresh interpreter seeds from the OS: numpy's and torch's global ones, and those
    the server's imports seeded so (:data:`_OS_SEEDED`)."""

    for generator in _OS_SEEDED:
        with contextlib.suppress(Exception):
            generator.seed()
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        with contextlib.suppress(Exception):
            numpy.random.seed()
    torch = sys.modules.get("torch")
    if torch is not None:
        with contextlib.suppress(Exception):
            torch.seed()


def _exit_code(code: Any) -> int:
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    with contextlib.suppress(Exception):
        print(code, file=sys.stderr)
    return 1


def _exit_as_interpreter(code: int) -> NoReturn:
    """Leave as the interpreter does: wait for the non-daemon threads, run the ``atexit`` handlers, flush."""

    shutdown = getattr(threading, "_shutdown", None)  # what interpreter finalization runs first
    if shutdown is not None:
        with contextlib.suppress(BaseException):
            shutdown()
    with contextlib.suppress(BaseException):
        atexit._run_exitfuncs()
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(BaseException):
            stream.flush()
    try:
        os._exit(code)
    except (OverflowError, TypeError):
        os._exit(1)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=f"python -m {SERVER_MODULE}")
    parser.add_argument("--fd", type=int, required=True, help="the control socket")
    parser.add_argument("--module", required=True, help="what each worker runs as __main__")
    parser.add_argument("--preload", action="append", default=[], help="a module to import once, in the server")
    parser.add_argument(
        "--child-init", action="append", default=[], help="module:function each worker calls before its module"
    )
    parser.add_argument(
        "--prepare", action="append", default=[], help='JSON {"ref": "module:function", "kwargs": {...}}, called once'
    )
    args = parser.parse_args(argv)
    return serve(args.fd, args.module, args.preload, args.child_init, [json.loads(item) for item in args.prepare])


# ------------------------------------------------------------------------------------------------- the caller side


class ForkedWorker:
    """A worker the server forked, in place of an ``asyncio.subprocess.Process``: its ``pid``, the ``pidfd`` the
    server opened for it, and its ``returncode`` once it has exited (negative for a signal, as asyncio's)."""

    def __init__(self, pid: int, pidfd: int, loop: asyncio.AbstractEventLoop) -> None:
        self.pid = pid
        self.pidfd = pidfd
        self.returncode: int | None = None
        self._loop = loop
        self._exited: asyncio.Future[int | None] = loop.create_future()
        self._watching = False

    def _finish(self, status: int | None) -> None:
        if self._exited.done():
            return
        if status is not None:
            self.returncode = os.waitstatus_to_exitcode(status)
        self._exited.set_result(self.returncode)

    def _watch_pidfd(self) -> None:
        """Its server is gone: learn of its exit from its pidfd (without a status)."""

        if self._exited.done() or self.pidfd < 0:
            return

        def exited() -> None:
            self._unwatch()
            self._finish(None)

        try:
            self._loop.add_reader(self.pidfd, exited)
            self._watching = True
        except (OSError, ValueError, RuntimeError):
            self._finish(None)

    def _unwatch(self) -> None:
        if self._watching:
            self._watching = False
            with contextlib.suppress(Exception):
                self._loop.remove_reader(self.pidfd)

    async def wait(self) -> int | None:
        """The exit code, once the worker has exited; None when its server died before it could say."""

        return await asyncio.shield(self._exited)

    def close(self) -> None:
        """Close its pidfd (the caller's, as one it opened for an exec'd process)."""

        self._unwatch()
        if self.pidfd >= 0:
            with contextlib.suppress(OSError):
                os.close(self.pidfd)
            self.pidfd = -1


class ForkServer:
    """The caller's handle on one fork server (see the module docstring); built by :meth:`start`, or by
    :meth:`launch` and :meth:`ready`."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        sock: socket.socket,
        *,
        env: Mapping[str, str],
        cwd: str,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.process = process
        self.pid = process.pid
        self.env = dict(env)
        self.cwd = cwd
        self.loop = loop
        self.seconds = 0.0  # what its imports took, once it said hello
        self.prepare_errors: list[str] = []  # what it could not build ahead (its workers build it)
        self.error: str | None = None  # why it cannot serve, once it said so, whether or not someone still waited
        self.dead = False
        self._serving = False  # it said hello without an error
        self._overdue = False  # a wait for its imports gave up, and nobody waits to hear that it died
        self._sock = sock
        self._closed = False
        self._tokens = itertools.count(1)
        self._send_lock = asyncio.Lock()
        self._pending: dict[int, asyncio.Future[ForkedWorker]] = {}
        self._workers: dict[int, ForkedWorker] = {}
        self._launched = time.monotonic()
        self._answered = self._launched  # when it last answered a fork request
        self._hello: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._hello.add_done_callback(_retrieve)  # whoever waited for a slow start may have given up
        loop.add_reader(sock.fileno(), self._on_readable)
        self._pidfd = os.pidfd_open(process.pid)
        loop.add_reader(self._pidfd, self._on_exit)

    @classmethod
    async def start(
        cls,
        *,
        python: str,
        module: str,
        preload: Sequence[str] = (),
        child_init: Sequence[str] = (),
        prepare: Sequence[Mapping[str, Any]] = (),
        env: Mapping[str, str],
        cwd: str | None = None,
        log_path: str | None = None,
        timeout_s: float,
    ) -> ForkServer:
        """Start a server and wait for its imports (:meth:`ready`); :class:`ForkServerError`, and the server killed,
        when it cannot serve (a module did not import, it runs a thread, it exited) or has not imported them in
        ``timeout_s``."""

        server = cls.launch(
            python=python,
            module=module,
            preload=preload,
            child_init=child_init,
            prepare=prepare,
            env=env,
            cwd=cwd,
            log_path=log_path,
        )
        try:
            await server.ready(timeout_s)
        except BaseException:
            await server.kill()
            raise
        return server

    @classmethod
    def launch(
        cls,
        *,
        python: str,
        module: str,
        preload: Sequence[str] = (),
        child_init: Sequence[str] = (),
        prepare: Sequence[Mapping[str, Any]] = (),
        env: Mapping[str, str],
        cwd: str | None = None,
        log_path: str | None = None,
    ) -> ForkServer:
        """Start a server, for the running event loop; it imports its modules meanwhile (:meth:`ready`)."""

        loop = asyncio.get_running_loop()
        cwd = cwd or os.getcwd()
        ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        ours.setblocking(False)
        log = open(log_path, "ab", buffering=0) if log_path else None  # noqa: SIM115 - handed to the server
        try:
            process = subprocess.Popen(
                [
                    python,
                    "-m",
                    SERVER_MODULE,
                    "--fd",
                    str(theirs.fileno()),
                    "--module",
                    module,
                    *itertools.chain.from_iterable(("--preload", name) for name in preload),
                    *itertools.chain.from_iterable(("--child-init", ref) for ref in child_init),
                    *itertools.chain.from_iterable(("--prepare", json.dumps(item)) for item in prepare),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log if log is not None else subprocess.DEVNULL,
                stderr=subprocess.STDOUT if log is not None else subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
                pass_fds=(theirs.fileno(),),
                env=dict(env),
                cwd=cwd,
            )
        except BaseException:
            ours.close()
            raise
        finally:
            theirs.close()
            if log is not None:
                log.close()
        return cls(process, ours, env=env, cwd=cwd, loop=loop)

    async def ready(self, timeout_s: float) -> None:
        """Wait until the server has imported its modules, ``timeout_s`` after its launch at most.
        :class:`ForkServerError` when it cannot serve (``permanent``: a module did not import, a thread runs in it),
        it is gone, or it has not imported them by then: it then goes on importing, and serves once it has
        (:attr:`serving`), or says why it cannot (:attr:`error`), unless it is closed.  A hello that came while this
        loop was held up is used, however late."""

        try:
            await self._answer(self._hello, self._launched, timeout_s)
        except TimeoutError as exc:
            self._overdue = True
            raise ForkServerError(f"the fork server did not import its modules in {timeout_s:.0f}s") from exc
        if self.error is not None:
            raise ForkServerError(self.error, permanent=True)

    @property
    def alive(self) -> bool:
        return not self.dead and not self._closed

    @property
    def serving(self) -> bool:
        """It has imported its modules and forks workers."""

        return self._serving and self.alive

    async def spawn(
        self, argv: Sequence[str], *, env: Mapping[str, str], cwd: str, log_path: str | None
    ) -> ForkedWorker:
        """Fork a worker that runs the server's module with ``argv``; :class:`ForkServerError` when the server did
        not fork it, its answer was lost, or it has answered no request for :data:`SPAWN_TIMEOUT_S`.  Its answer is
        used whenever it came: the worker it names is killed only when the caller was cancelled, or gave up on a
        silent server, before it came."""

        if not self.alive:
            raise ForkServerError("the fork server is gone")
        if not self._serving:
            raise ForkServerError("the fork server has not imported its modules")
        token = next(self._tokens)
        reply: asyncio.Future[ForkedWorker] = self.loop.create_future()
        self._pending[token] = reply
        message = {"op": "fork", "token": token, "argv": list(argv), "env": dict(env), "cwd": cwd, "log_path": log_path}
        asked = time.monotonic()
        sent = False
        try:
            # One send at a time: the loop keeps one writer per socket, so a second one waiting for room would
            # replace the first one's, which would then never complete.  The lock lets the requests through in the
            # order of their tokens, which is the order the server answers them in (_answered_up_to).
            async with self._send_lock:
                await self._send(json.dumps(message).encode(), asked)
            sent = True  # nothing between the send and here yields to the loop, so no cancellation falls between
            worker = await self._answer(reply, time.monotonic(), SPAWN_TIMEOUT_S)
        except BaseException as exc:
            if sent:
                self._abandon(token)
            else:  # the server never got it
                self._pending.pop(token, None)
            if isinstance(exc, TimeoutError):
                raise ForkServerError(f"the fork server has answered no request for {SPAWN_TIMEOUT_S:.0f}s") from exc
            if isinstance(exc, OSError):
                raise ForkServerError(f"cannot reach the fork server: {exc}") from exc
            raise
        del self._pending[token]
        return worker

    async def _send(self, payload: bytes, since: float) -> None:
        """Hand a request to the server's socket, all of it (one message) or nothing.  While the socket has no room,
        wait for the server to read; :class:`TimeoutError` once it has answered no request for :data:`SPAWN_TIMEOUT_S`
        since ``since`` and the socket still has no room (the server never got the request)."""

        fd = self._sock.fileno()
        while True:
            try:
                self._sock.send(payload)
                return
            except (BlockingIOError, InterruptedError):
                pass
            silent = self._silence(since)
            if silent >= SPAWN_TIMEOUT_S:
                raise TimeoutError
            room: asyncio.Future[None] = self.loop.create_future()
            self.loop.add_writer(fd, _settle, room)
            try:
                await asyncio.wait([room], timeout=SPAWN_TIMEOUT_S - silent)
            finally:
                self.loop.remove_writer(fd)

    async def _answer(self, future: asyncio.Future[Any], since: float, timeout_s: float) -> Any:
        """``future``'s result, which a message from the server settles; :class:`TimeoutError` once the server has
        answered no request for ``timeout_s`` since ``since`` (:meth:`_silence`)."""

        while not future.done():
            silent = self._silence(since)
            if future.done():
                break
            if silent >= timeout_s:
                raise TimeoutError
            await asyncio.wait([future], timeout=timeout_s - silent)
        return future.result()

    def _silence(self, since: float) -> float:
        """How long the server has answered no request, since ``since``.  What it sent is read first: this loop may
        have been held up while the server answered, and its own lateness is not the server's silence.  The reports of
        its workers' exits do not count: a server that sends them but answers no request is not working through the
        queue."""

        if not self.dead and not self._closed:
            self._on_readable()
        return time.monotonic() - max(since, self._answered)

    def _abandon(self, token: int) -> None:
        """Its caller gave up on a fork request: a worker forked for it is killed, now or when the answer comes
        (:meth:`_dispatch` finds the request cancelled), since nothing would supervise it."""

        future = self._pending.get(token)
        if future is None:
            return
        if not future.done():
            future.cancel()
            return
        del self._pending[token]
        if not future.cancelled() and future.exception() is None:
            worker = future.result()
            self._workers.pop(worker.pid, None)
            _kill_group(worker.pid, worker.pidfd)
            worker.close()

    def _on_readable(self) -> None:
        while True:
            try:
                payload, fds, _flags, _address = socket.recv_fds(self._sock, MAX_MESSAGE_BYTES, 1)
            except (BlockingIOError, InterruptedError):
                return
            except OSError:
                payload, fds = b"", []
            if not payload:
                self._lost("it closed its socket")
                return
            try:
                self._dispatch(json.loads(payload), fds)
            except Exception:  # noqa: BLE001 - one bad message must not stop the reader
                logger.warning("Unreadable message from fork server %s", self.pid, exc_info=True)
                for fd in fds:
                    with contextlib.suppress(OSError):
                        os.close(fd)

    def _dispatch(self, message: dict[str, Any], fds: list[int]) -> None:
        op = message.get("op")
        if op == "hello":
            if not self._hello.done():
                if message.get("error"):
                    self.error = str(message["error"])
                else:  # it serves from now on, whether or not someone still waits
                    self.seconds = float(message.get("seconds") or 0.0)
                    self.prepare_errors = list(message.get("prepare_errors") or [])
                    self._serving = True
                self._hello.set_result(message)
        elif op == "forked":
            token, pid = int(message["token"]), int(message["pid"])
            self._answered_up_to(token)
            future = self._pending.get(token)
            if not fds:  # the kernel dropped its pidfd: this process had no descriptor free
                _kill_group(pid, None)  # nothing could supervise this worker, so it ends now
                if future is not None and not future.done():
                    future.set_exception(
                        ForkServerError(
                            f"the fork server forked pid {pid}, but its pidfd did not reach this process, which may "
                            "have no file descriptor free; that worker was killed"
                        )
                    )
                else:
                    self._pending.pop(token, None)
                return
            (pidfd,) = fds
            if future is not None and not future.done():
                # Known from now on: its exit may be the very next message, before spawn() resumes.
                worker = ForkedWorker(pid, pidfd, self.loop)
                self._workers[pid] = worker
                future.set_result(worker)
            else:  # its caller gave up: nothing supervises this worker, so it ends now
                self._pending.pop(token, None)
                _kill_group(pid, pidfd)
                os.close(pidfd)
        elif op == "refused":
            token = int(message["token"])
            self._answered_up_to(token)
            future = self._pending.get(token)
            if future is not None and not future.done():
                future.set_exception(ForkServerError(str(message.get("reason"))))
            elif future is not None:
                self._pending.pop(token, None)
        elif op == "exited":
            worker = self._workers.pop(int(message["pid"]), None)
            if worker is not None:
                worker._finish(int(message["status"]))

    def _answered_up_to(self, token: int) -> None:
        """The server answered request ``token``.  It answers the requests one at a time, in the order they were sent,
        which is the order of their tokens: an earlier one still waiting lost its answer (the server could not send
        it, or this process could not read it), and fails now rather than waiting while the server goes on."""

        self._answered = time.monotonic()
        for earlier, future in self._pending.items():
            if earlier < token and not future.done():
                future.set_exception(ForkServerError("the fork server answered a later request, not this one"))

    def _on_exit(self) -> None:
        if not self.dead and not self._closed:
            self._on_readable()  # what it said before it exited (a hello that says why it cannot serve)
        with contextlib.suppress(Exception):
            self.loop.remove_reader(self._pidfd)
        with contextlib.suppress(OSError):
            os.close(self._pidfd)
        self.process.poll()  # reaps it
        self._lost(f"it exited ({self.process.returncode})")

    def _lost(self, reason: str) -> None:
        """The server is gone: fail what waits on it, and read its workers' exits from their pidfds."""

        if self.dead:
            return
        self.dead = True
        with contextlib.suppress(Exception):
            self.loop.remove_reader(self._sock.fileno())
        if (self._serving or self._overdue) and not self._closed:
            logger.warning("Fork server %s is gone: %s", self.pid, reason)
        if not self._hello.done():
            self._hello.set_exception(ForkServerError(f"the fork server is gone: {reason}"))
        for token, future in list(self._pending.items()):
            if future.cancelled():
                del self._pending[token]  # abandoned: nothing will be forked for it now
            elif not future.done():
                future.set_exception(ForkServerError(f"the fork server is gone: {reason}"))
        for worker in self._workers.values():
            worker._watch_pidfd()
        self._workers.clear()

    def close(self) -> None:
        """Take no more workers; the server exits once the workers it forked have exited.  One that has not said
        hello yet forked none, and is killed: it would import its modules for nothing."""

        if self._closed:
            return
        self._closed = True
        if not self.dead:
            with contextlib.suppress(Exception):
                self.loop.remove_reader(self._sock.fileno())
        with contextlib.suppress(OSError):
            self._sock.close()
        if not self._hello.done():
            self.process.kill()  # reaped when its pidfd says it exited (_on_exit), or by kill()
        # Their exits are no longer reported: watch their pidfds instead.
        for worker in self._workers.values():
            worker._watch_pidfd()
        self._workers.clear()

    async def kill(self) -> None:
        """Stop the server now; the workers it forked get their own death signals."""

        self.close()
        if self.process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                self.process.kill()
            await asyncio.to_thread(self.process.wait)


def _kill_group(pid: int, pidfd: int | None) -> None:
    with contextlib.suppress(OSError):
        os.killpg(pid, signal.SIGKILL)  # it leads its own group before the server answers
    if pidfd is not None:
        with contextlib.suppress(OSError):
            signal.pidfd_send_signal(pidfd, signal.SIGKILL)


def _settle(future: asyncio.Future[None]) -> None:
    if not future.done():
        future.set_result(None)


def _retrieve(future: asyncio.Future[Any]) -> None:
    """Mark ``future``'s exception retrieved, so that asyncio does not report it when nobody waits for it any more."""

    if not future.cancelled():
        future.exception()


if __name__ == "__main__":
    sys.exit(main())
