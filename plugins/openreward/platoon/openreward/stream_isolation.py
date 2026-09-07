"""Keep ``sys.stdout`` / ``sys.stderr`` private to one rollout thread.

Why this module exists
======================

The Tinker rollout path runs *every* trajectory in one asyncio process on the
head node (the AReaL path used one subprocess per node, so each rollout owned
its interpreter).  Two upstream behaviours are only safe when a process hosts a
single agent, and both bite hard once ``num_concurrent_rollout_workflow_workers
x group_size`` root trajectories plus their recursive subagents share an
interpreter:

1. ``openhands.tools.programmatic_tool_calling.impl`` runs each PTC cell under
   ``contextlib.redirect_stdout`` / ``redirect_stderr`` so it can show the cell's
   output back to the model.  Those helpers rebind the *process-global*
   ``sys.stdout`` / ``sys.stderr`` to an ``io.StringIO``.  Tool executors run on
   worker threads (``Agent.astep`` -> ``_aexecute_actions`` ->
   ``loop.run_in_executor``), so N concurrent rollouts hold N overlapping
   redirects.  ``redirect_*`` saves whatever stream it observed on ``__enter__``
   and restores exactly that on ``__exit__``, so interleaved enter/exit orders
   restore a *stale* StringIO:

       A.__enter__()  # saves the console
       B.__enter__()  # saves A's StringIO
       A.__exit__()   # restores the console
       B.__exit__()   # restores A's StringIO  <- leaked, permanently

   From that moment ``sys.stderr`` is a dead buffer for the rest of the run.

2. ``fastmcp.client.transports.stdio._stdio_transport_connect_task`` reads
   ``sys.stderr`` *at connect time* and passes it to ``subprocess.Popen`` as the
   MCP server's stderr.  ``io.StringIO`` has no ``fileno()``, so spawning the
   openreward ``mcp_bridge`` raises ``io.UnsupportedOperation: fileno``.  fastmcp
   turns that into ``RuntimeError("Client failed to connect: fileno")`` and the
   OpenHands SDK masks it as ``MCPError("MCP Connection Failure")``
   (``openhands/sdk/mcp/client.py:55``), which is why the real cause never
   reached the trajectory records.

What this module installs
=========================

``install()`` applies two independent hardenings:

* ``ThreadScopedStream`` replaces ``sys.stdout`` / ``sys.stderr`` with a proxy
  that keeps a real ``fileno()`` and routes ``write``/``flush`` to a per-thread
  capture stack, and the PTC module's ``redirect_stdout`` / ``redirect_stderr``
  names are rebound to :func:`redirect_stdout` / :func:`redirect_stderr`, which
  push/pop that per-thread stack instead of mutating global state.  A cell then
  captures its own thread's output and nothing else -- which is also what the
  observation was always meant to contain, instead of whatever unrelated
  rollouts logged during the same window.
* ``StdioMCPServer.to_transport`` is patched to pin the MCP bridge's stderr to
  the console stream captured at install time, so spawning a bridge can never
  again depend on the current value of ``sys.stderr``.

Both are idempotent and degrade to a no-op when the upstream modules are absent.
"""

from __future__ import annotations

import io
import os
import sys
import threading
from typing import Any, TextIO

__all__ = [
    "ThreadScopedStream",
    "console_stderr",
    "enabled",
    "install",
    "install_mcp_stdio_stderr_pin",
    "install_programmatic_tool_calling_isolation",
    "install_stream_proxies",
    "redirect_stderr",
    "redirect_stdout",
]

_STDOUT = "stdout"
_STDERR = "stderr"

_install_lock = threading.RLock()
_thread_state = threading.local()

_console_stdout: TextIO | None = None
_console_stderr: TextIO | None = None


def _capture_stack(which: str) -> list[Any]:
    """The calling thread's capture stack for ``which`` (innermost last)."""

    stacks = getattr(_thread_state, "stacks", None)
    if stacks is None:
        stacks = {_STDOUT: [], _STDERR: []}
        _thread_state.stacks = stacks
    return stacks[which]


class ThreadScopedStream(io.TextIOBase):
    """A ``sys.stdout``/``sys.stderr`` stand-in with per-thread capture.

    Writes go to the innermost capture target pushed by *this* thread, or to the
    console stream when the thread has none.  Everything else -- ``fileno()``
    above all, which is what ``subprocess`` needs to hand a child its stderr --
    is answered by the console stream, so the proxy is always a valid target for
    process spawning even while some thread is capturing.
    """

    def __init__(self, which: str, console: TextIO) -> None:
        super().__init__()
        self._which = which
        self._console = console

    @property
    def console(self) -> TextIO:
        """The real stream this proxy stands in front of."""

        return self._console

    def _target(self) -> Any:
        stack = _capture_stack(self._which)
        return stack[-1] if stack else self._console

    # -- writing ---------------------------------------------------------
    def write(self, s: str) -> int:  # type: ignore[override]
        return self._target().write(s)

    def writelines(self, lines) -> None:  # type: ignore[override]
        self._target().writelines(lines)

    def flush(self) -> None:
        target = self._target()
        try:
            target.flush()
        except ValueError:
            # A capture buffer closed by its owner must not break logging.
            pass

    # -- capabilities the console owns -----------------------------------
    def fileno(self) -> int:
        return self._console.fileno()

    def isatty(self) -> bool:
        return self._console.isatty()

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return getattr(self._console, "encoding", "utf-8")

    @property
    def errors(self) -> str | None:  # type: ignore[override]
        return getattr(self._console, "errors", None)

    @property
    def buffer(self) -> Any:
        return self._console.buffer  # type: ignore[attr-defined]

    @property
    def closed(self) -> bool:  # type: ignore[override]
        return False

    def close(self) -> None:
        # The proxy owns nothing; closing it would silence the whole process.
        return None

    def __getattr__(self, name: str) -> Any:
        # ``reconfigure``, ``detach``, ``name``, ``mode`` ... : anything the
        # proxy does not define is answered by the real stream, so the swap
        # stays invisible to libraries that poke at ``sys.stderr``.  Private
        # names are refused so a half-initialised proxy cannot recurse through
        # ``self._console``.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._console, name)


class _ThreadScopedRedirect:
    """Drop-in replacement for ``contextlib.redirect_stdout``/``redirect_stderr``.

    Same surface (a reusable, reentrant context manager returning the target),
    but it pushes onto the calling thread's capture stack instead of rebinding
    ``sys.stdout``/``sys.stderr``, so concurrent users cannot restore each
    other's targets.
    """

    def __init__(self, which: str, new_target: Any) -> None:
        self._which = which
        self._new_target = new_target

    def __enter__(self) -> Any:
        install_stream_proxies()
        _capture_stack(self._which).append(self._new_target)
        return self._new_target

    def __exit__(self, *exc_info: object) -> bool:
        stack = _capture_stack(self._which)
        # Pop the innermost entry for this target rather than the top of the
        # stack: a mis-nested caller must not drop somebody else's capture.
        for index in range(len(stack) - 1, -1, -1):
            if stack[index] is self._new_target:
                del stack[index]
                break
        return False


def redirect_stdout(new_target: Any) -> _ThreadScopedRedirect:
    """Thread-scoped ``contextlib.redirect_stdout``."""

    return _ThreadScopedRedirect(_STDOUT, new_target)


def redirect_stderr(new_target: Any) -> _ThreadScopedRedirect:
    """Thread-scoped ``contextlib.redirect_stderr``."""

    return _ThreadScopedRedirect(_STDERR, new_target)


def console_stderr() -> TextIO:
    """The process stderr as it was before any capture was installed.

    Only a stream that can hand a child process a real descriptor qualifies, so
    a caller can pass the result straight to ``subprocess``.
    """

    for candidate in (_console_stderr, sys.__stderr__, sys.stderr):
        if candidate is None or isinstance(candidate, ThreadScopedStream):
            continue
        try:
            candidate.fileno()
        except Exception:
            continue
        return candidate
    return sys.stderr


def install_stream_proxies() -> bool:
    """Put :class:`ThreadScopedStream` on ``sys.stdout``/``sys.stderr`` once."""

    global _console_stdout, _console_stderr
    with _install_lock:
        installed = False
        if not isinstance(sys.stdout, ThreadScopedStream):
            # Remember the *first* console only: something that wraps the proxy
            # later (wandb patches ``write`` rather than rebinding, but other
            # libraries do rebind) must not become the fallback descriptor.
            if _console_stdout is None:
                _console_stdout = sys.stdout
            sys.stdout = ThreadScopedStream(_STDOUT, sys.stdout)  # type: ignore[assignment]
            installed = True
        if not isinstance(sys.stderr, ThreadScopedStream):
            if _console_stderr is None:
                _console_stderr = sys.stderr
            sys.stderr = ThreadScopedStream(_STDERR, sys.stderr)  # type: ignore[assignment]
            installed = True
        return installed


def install_programmatic_tool_calling_isolation() -> bool:
    """Make OpenHands' PTC cell capture thread-scoped instead of process-wide."""

    try:
        from openhands.tools.programmatic_tool_calling import impl as ptc_impl
    except Exception:
        return False
    if getattr(ptc_impl, "_platoon_thread_scoped_redirects", False):
        return False
    install_stream_proxies()
    ptc_impl.redirect_stdout = redirect_stdout
    ptc_impl.redirect_stderr = redirect_stderr
    ptc_impl._platoon_thread_scoped_redirects = True
    return True


def install_mcp_stdio_stderr_pin() -> bool:
    """Pin the MCP bridge subprocess's stderr to the console stream.

    fastmcp otherwise samples ``sys.stderr`` when it spawns the stdio server, so
    any library that swaps the global stream -- PTC before this module, browser
    tooling, an ipython capture -- breaks every concurrent bridge spawn.
    """

    try:
        from fastmcp.client.transports import StdioTransport
        from fastmcp.mcp_config import StdioMCPServer
    except Exception:
        return False
    if getattr(StdioMCPServer.to_transport, "_platoon_pinned_stderr", False):
        return False

    def to_transport(self) -> "StdioTransport":
        return StdioTransport(
            command=self.command,
            args=self.args,
            env=self.env,
            cwd=self.cwd,
            keep_alive=self.keep_alive,
            log_file=console_stderr(),
        )

    to_transport._platoon_pinned_stderr = True  # type: ignore[attr-defined]
    StdioMCPServer.to_transport = to_transport  # type: ignore[method-assign]
    return True


def enabled() -> bool:
    """Whether ``install`` should do anything.

    ``OPENREWARD_STREAM_ISOLATION=0`` restores the upstream behaviour.  It exists
    because this module monkeypatches third-party globals and an operator needs a
    documented way to take that out of the loop -- and because it is how the
    before/after MCP failure rate was measured on one binary.  Turning it off
    reinstates the ``MCP Connection Failure`` storm whenever programmatic tool
    calling is enabled and rollouts overlap, so it is not a production setting.
    """

    return os.environ.get("OPENREWARD_STREAM_ISOLATION", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def install() -> dict[str, bool]:
    """Apply every isolation this module provides; safe to call repeatedly."""

    if not enabled():
        return {"programmatic_tool_calling": False, "mcp_stdio_stderr": False}
    return {
        "programmatic_tool_calling": install_programmatic_tool_calling_isolation(),
        "mcp_stdio_stderr": install_mcp_stdio_stderr_pin(),
    }
