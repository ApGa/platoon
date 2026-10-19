"""How cell output is captured: what IPython cells print, and the IPython-version-safe ``run_cell_async``.

``ShellCapture`` routes ``sys.stdout``/``sys.stderr`` writes to per-capture buffers
through context variables, so concurrently running cells (one per CodeAct agent, each
on its own cell-loop thread) never mix their output. The proxies that do the routing replace
the streams that are current when the first capture opens (whatever pytest or
``contextlib.redirect_stdout`` installed), send writes from outside any capture there, and
put them back when the last capture closes. It also points the shell's display
publisher and ``sys.displayhook`` at a buffer that nothing reads, so, as in the original
``CodeActAgent``, what a cell passes to ``display()`` is dropped; an expression's value
still reaches stdout as ``Out[n]: ...`` (IPython swaps its own display hook in while a cell
runs), and ``drop_out_prompts`` then drops the prompt. ``shorten_install_paths`` names the
installed files a traceback shows by their module paths.
``run_cell_async`` transforms the cell itself, which IPython >= 9.16 requires and older
versions accept.
"""

import contextvars
import functools
import io
import os
import re
import sys
import sysconfig
import threading
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from IPython.core.interactiveshell import ExecutionResult, InteractiveShell

# -------------------------------------------------------------------------
# 1. Global proxy streams (installed while any capture is open)
# -------------------------------------------------------------------------
_current_capture_out: contextvars.ContextVar[Optional[io.TextIOBase]] = contextvars.ContextVar(
    "current_capture_out", default=None
)
_current_capture_err: contextvars.ContextVar[Optional[io.TextIOBase]] = contextvars.ContextVar(
    "current_capture_err", default=None
)


class _Proxy(io.TextIOBase):
    def __init__(self, which: str, default_stream: io.TextIOBase) -> None:
        super().__init__()
        self._which = which
        self._default = default_stream

    def write(self, s: str) -> int:
        target = _current_capture_out.get() if self._which == "out" else _current_capture_err.get()
        (target or self._default).write(s)
        return len(s)

    def flush(self) -> None:
        target = (_current_capture_out.get() if self._which == "out" else _current_capture_err.get()) or self._default
        target.flush()

    def fileno(self) -> int:  # lets faulthandler work
        return self._default.fileno()

    def isatty(self) -> bool:  # nice for colorised logs
        return self._default.isatty()


# -------------------------------------------------------------------------
# Helper functions to (un)install proxy streams
# -------------------------------------------------------------------------
# We only want the proxy streams to be active while a ShellCapture context is
# running.  Installing them at import-time causes side-effects that can make a
# program appear to hang when an exception is raised in user code.  Instead we
# do a reference-counted, on-demand installation so that stdout/stderr revert
# back to their previous values as soon as all ShellCapture scopes are closed.
# The streams to go back to are the ones current when the first capture opens,
# not the ones at import: whatever redirected sys.stdout since (pytest's capture,
# an application's contextlib.redirect_stdout) keeps its stream.

_proxy_install_count = 0  # how many active ShellCapture contexts there are
_proxy_lock = threading.Lock()  # captures open and close on several cell threads
# While the proxies are installed: per sys attribute, the stream they replaced and the proxy installed over it.
_replaced: dict[str, tuple[Any, _Proxy]] = {}


def _install_proxies() -> None:
    """Install proxy streams over the current ones if this is the first request."""
    global _proxy_install_count
    with _proxy_lock:
        if _proxy_install_count == 0:
            for name, which in (("stdout", "out"), ("stderr", "err")):
                stream = getattr(sys, name)
                proxy = _Proxy(which, stream)
                _replaced[name] = (stream, proxy)
                setattr(sys, name, proxy)
        _proxy_install_count += 1


def _uninstall_proxies() -> None:
    """Once the last capture has finished, put back the streams the proxies replaced; a stream something else
    installed in the meantime stays."""
    global _proxy_install_count
    with _proxy_lock:
        if _proxy_install_count == 0:
            # This should never happen, but guard against negative counts.
            return
        _proxy_install_count -= 1
        if _proxy_install_count == 0:
            for name, (stream, proxy) in _replaced.items():
                if getattr(sys, name) is proxy:
                    setattr(sys, name, stream)
            _replaced.clear()


# -------------------------------------------------------------------------
# 2. A minimal buffer stream for each shell
# -------------------------------------------------------------------------
class _Buffer(io.StringIO):
    """Just like StringIO but with a pop() helper."""

    def pop(self) -> str:
        """Read all data and clear the buffer (destructive)."""
        self.seek(0)
        data = self.read()
        self.truncate(0)
        self.seek(0)
        return data


# -------------------------------------------------------------------------
# 3. Async context‑manager that sets the contextvars
# -------------------------------------------------------------------------
class ShellCapture:
    """
    Async-safe stdout/stderr capture using context variables; display output is dropped.

    Usage:
        with ShellCapture() as cap:
            await run_cell_async(shell, code)
        print(cap.pop_stdout())  # captured text

    Async Safety:
    - ✅ Safe for typical sequential usage within async contexts
    - ✅ Uses contextvars for proper async task isolation
    - ⚠️  Don't share the same instance across concurrent tasks
    """

    def __init__(self, capture_display: bool = True) -> None:
        self._out_buf = _Buffer()
        self._err_buf = _Buffer()
        self._displays: List[Any] = []  # what display() publishes during the capture; never read
        self._capture_display = capture_display

        # Store original hooks for restoration
        self._original_display_pub = None
        self._original_display_hook = None

    def pop_stdout(self) -> str:
        """Get captured stdout and clear the buffer."""
        return self._out_buf.pop()

    def pop_stderr(self) -> str:
        """Get captured stderr and clear the buffer."""
        return self._err_buf.pop()

    # CM ------------------------------------------------------------
    def __enter__(self):
        # Swap context‑local targets; keep the tokens to restore later
        self._tok_out = _current_capture_out.set(self._out_buf)
        self._tok_err = _current_capture_err.set(self._err_buf)

        # Set up display capture if requested
        if self._capture_display:
            self._setup_display_capture()

        # Install proxies if this is the first capture context
        _install_proxies()

        return self

    def __exit__(self, exc_type, exc, tb):
        # Restore whatever was there before
        _current_capture_out.reset(self._tok_out)
        _current_capture_err.reset(self._tok_err)

        # Restore display hooks
        if self._capture_display:
            self._restore_display_capture()

        # Restore original streams
        _uninstall_proxies()

        # let exceptions propagate
        return False

    def _setup_display_capture(self) -> None:
        """Set up display publisher and display hook capture."""
        try:
            from IPython.core.displayhook import CapturingDisplayHook
            from IPython.core.displaypub import CapturingDisplayPublisher
            from IPython.core.getipython import get_ipython
        except ImportError:
            # IPython not available, skip display capture
            return

        shell = get_ipython()
        if shell is None:
            # Not running in IPython, skip display capture
            return

        # Capture display publisher
        self._original_display_pub = shell.display_pub
        shell.display_pub = CapturingDisplayPublisher()
        shell.display_pub.outputs = self._displays

        # Capture display hook (for last expression output)
        self._original_display_hook = sys.displayhook
        sys.displayhook = CapturingDisplayHook(shell=shell, outputs=self._displays)

    def _restore_display_capture(self) -> None:
        """Restore original display publisher and display hook."""
        try:
            from IPython.core.getipython import get_ipython
        except ImportError:
            return

        shell = get_ipython()
        if shell is None:
            return

        # Restore display publisher
        if self._original_display_pub is not None:
            shell.display_pub = self._original_display_pub

        # Restore display hook
        if self._original_display_hook is not None:
            sys.displayhook = self._original_display_hook


def strip_ansi_escape_sequences(text):
    """
    Remove ANSI escape sequences (color codes, formatting) from text.
    """
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_escape.sub("", text)


# IPython's prompt before a cell's value, at the start of a line; a terminal may put "\x1b[?7h" in front of it.
_OUT_PROMPT = re.compile(r"^(?:\x1b?\[\?7h)?Out\[\d+\]:", re.MULTILINE)


def drop_out_prompts(text: str) -> str:
    """``text`` without IPython's ``Out[n]:`` prompt at the start of any line; the rest of each line stays.

    That is what the original ``CodeActAgent`` meant to do. It cut an output that starts with the prompt at its first
    colon and joined the rest without colons, so ``Out[1]: {'a': 1}`` became `` {'a' 1}``, and it kept a prompt that
    followed printed lines. A value's line keeps the space after the prompt, as in the original agent:
    ``Out[1]: 'Too low'`` becomes `` 'Too low'``.
    """
    return _OUT_PROMPT.sub("", text)


# The absolute path of a Python source file, or IPython's form of one below the home directory ("~/..."), where a
# path can start. Bounded, so text without a ".py" in it is scanned in linear time.
_SOURCE_FILE = re.compile(r"(?<![\w.~/-])(?:~(?=/)|/)[^\s\"'()<>\[\]{},;:|*?]{0,4096}?\.py(?!\w)")
# Where the standard library and site-packages are, also when sys.path leaves them out.
_INSTALL_DIRS = tuple(sysconfig.get_paths().get(name) for name in ("stdlib", "platstdlib", "purelib", "platlib"))


def shorten_install_paths(text: str) -> str:
    """``text`` with the path of every Python source file below an import root (``sys.path``) shortened to the
    module's path below that root: ``platoon/harness/builtin/codeact/executor.py``, ``json/decoder.py``.

    Tracebacks, warnings and module reprs name the files of the code they show by absolute path -- where platoon,
    its plugins and Python are installed -- in Python's form (``File "/abs/path/json/decoder.py", line 338``) and in
    IPython's (``File ~/path/json/decoder.py:338``). A cell's output reaches the model's prompt, which should not
    carry them; the module path says as much about the code. Other paths, the ones a cell works with, are left alone.
    """

    roots = _import_roots(tuple(sys.path))

    def shorten(match: "re.Match[str]") -> str:
        path = match.group(0)
        absolute = os.path.expanduser(path) if path.startswith("~") else path
        for root in roots:
            if absolute.startswith(root):
                return absolute[len(root) :]
        return path

    return _SOURCE_FILE.sub(shorten, text)


@functools.lru_cache(maxsize=8)
def _import_roots(path: tuple) -> tuple:
    """The absolute directories code is imported from (``sys.path``, the standard library's and site-packages),
    with their resolved forms, each ending in ``/``, longest first."""

    roots = set()
    for entry in (*path, *_INSTALL_DIRS):
        if entry and os.path.isabs(entry):
            for directory in (os.path.normpath(entry), os.path.realpath(entry)):
                if directory != "/":
                    roots.add(directory + "/")
    return tuple(sorted(roots, key=len, reverse=True))


async def run_cell_async(shell: "InteractiveShell", code: str) -> "ExecutionResult":
    """Run ``code`` in ``shell`` from async code, the way ``shell.run_cell`` does.

    Since IPython 9.16 ``run_cell_async`` no longer transforms the cell itself,
    so the caller transforms it and reports a failed transformation.
    """

    try:
        transformed_cell = shell.transform_cell(code)
    except Exception:
        return await shell.run_cell_async(code, transformed_cell=code, preprocessing_exc_tuple=sys.exc_info())
    return await shell.run_cell_async(code, transformed_cell=transformed_cell)
