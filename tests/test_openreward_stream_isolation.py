"""Regression tests for the single-process rollout stream isolation.

The 15-node Tinker run ta15-rec-q36-r3-v1 lost 968 of 1060 trajectories to
``MCPError: MCP Connection Failure`` because OpenHands' programmatic tool
calling rebinds the process-global ``sys.stderr`` to an ``io.StringIO`` while a
cell runs, and fastmcp hands whatever ``sys.stderr`` is at connect time to
``subprocess.Popen`` as the MCP bridge's stderr (``io.StringIO`` has no
``fileno()``).  Overlapping cells additionally leak a stale buffer, which is why
the failure rate went from ~58 % in batch 0 to 100 % from batch 1 on.
"""

from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
import threading
import types
from contextlib import contextmanager, redirect_stderr
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "plugins/openreward/platoon/openreward/stream_isolation.py"


def _load_module():
    """Import the module standalone: it must not need the openreward package."""

    spec = importlib.util.spec_from_file_location("platoon_openreward_stream_isolation_under_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def isolation(tmp_path):
    """A fresh module wired to real console files, not pytest's capture.

    pytest swaps ``sys.stdout``/``sys.stderr`` for fd-less buffers around every
    test phase, so the console whose ``fileno()`` the proxy delegates to has to
    be supplied here -- and the proxy has to be installed inside the test body
    (``with isolation.activate()``), after pytest's own capture is in place.
    """

    module = _load_module()
    module.console_out_path = tmp_path / "console.out"
    module.console_err_path = tmp_path / "console.err"

    @contextmanager
    def activate():
        saved_out, saved_err = sys.stdout, sys.stderr
        with module.console_out_path.open("w") as out, module.console_err_path.open("w") as err:
            sys.stdout, sys.stderr = out, err
            try:
                module.install_stream_proxies()
                yield module
            finally:
                sys.stdout, sys.stderr = saved_out, saved_err

    module.activate = activate
    return module


def test_upstream_redirect_stderr_breaks_subprocess_spawn(tmp_path):
    """The upstream failure mode, pinned so the fix has something to beat."""

    saved = sys.stderr
    with (tmp_path / "console.err").open("w") as console:
        sys.stderr = console
        try:
            with redirect_stderr(io.StringIO()):
                with pytest.raises(io.UnsupportedOperation):
                    subprocess.Popen(
                        [sys.executable, "-c", "pass"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=sys.stderr,
                    )
        finally:
            sys.stderr = saved


def test_upstream_redirect_stderr_leaks_a_stale_buffer_when_interleaved():
    """Interleaved enter/exit permanently strands ``sys.stderr`` on a buffer."""

    console = sys.stderr
    first, second = io.StringIO(), io.StringIO()
    outer, inner = redirect_stderr(first), redirect_stderr(second)
    try:
        outer.__enter__()  # saves the console
        inner.__enter__()  # saves `first`
        outer.__exit__(None, None, None)  # restores the console
        inner.__exit__(None, None, None)  # restores `first` -- the leak
        assert sys.stderr is first
        assert sys.stderr is not console
    finally:
        sys.stderr = console


def test_proxy_keeps_a_real_fileno_while_a_thread_captures(isolation):
    with isolation.activate():
        assert isinstance(sys.stderr, isolation.ThreadScopedStream)
        buffer = io.StringIO()
        with isolation.redirect_stderr(buffer):
            # The whole point: a live capture must not cost the stream its fd.
            assert isinstance(sys.stderr.fileno(), int)
            print("captured", file=sys.stderr)
        assert buffer.getvalue() == "captured\n"
        print("console", file=sys.stderr)
        sys.stderr.flush()
    text = isolation.console_err_path.read_text()
    assert "console" in text
    assert "captured" not in text


def test_spawning_a_subprocess_works_during_a_capture(isolation):
    with isolation.activate():
        with isolation.redirect_stderr(io.StringIO()):
            proc = subprocess.Popen(
                [sys.executable, "-c", "pass"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
            )
            assert proc.wait(timeout=120) == 0


def test_capture_is_private_to_the_capturing_thread(isolation):
    with isolation.activate():
        capturing_started = threading.Event()
        release = threading.Event()
        captured = io.StringIO()
        other = io.StringIO()

        def capture_thread():
            with isolation.redirect_stdout(captured):
                capturing_started.set()
                release.wait(60)

        def other_thread():
            with isolation.redirect_stdout(other):
                print("mine")

        worker = threading.Thread(target=capture_thread)
        worker.start()
        try:
            assert capturing_started.wait(60)
            second = threading.Thread(target=other_thread)
            second.start()
            second.join(60)
            # A concurrent capture must not land in the first thread's buffer,
            # and the first thread's buffer must not be restored out from under
            # it either.
            assert other.getvalue() == "mine\n"
            assert captured.getvalue() == ""
            assert isolation._capture_stack("stdout") == []
        finally:
            release.set()
            worker.join(60)


def test_interleaved_captures_never_strand_the_console(isolation):
    with isolation.activate():
        proxy = sys.stderr
        first, second = io.StringIO(), io.StringIO()
        outer = isolation.redirect_stderr(first)
        inner = isolation.redirect_stderr(second)
        outer.__enter__()
        inner.__enter__()
        outer.__exit__(None, None, None)
        inner.__exit__(None, None, None)
        assert sys.stderr is proxy
        assert isolation._capture_stack("stderr") == []
        assert isinstance(sys.stderr.fileno(), int)


def test_console_stderr_survives_an_upstream_global_redirect(isolation):
    with isolation.activate():
        console = sys.stderr.console
        with redirect_stderr(io.StringIO()):  # a library that still swaps globally
            assert isolation.console_stderr() is console
            assert isinstance(isolation.console_stderr().fileno(), int)


def test_install_stream_proxies_is_idempotent(isolation):
    with isolation.activate():
        proxy = sys.stderr
        assert isolation.install_stream_proxies() is False
        assert sys.stderr is proxy


def test_programmatic_tool_calling_patch_rebinds_the_module_names(isolation, monkeypatch):
    ptc_impl = types.ModuleType("impl")
    ptc_impl.redirect_stdout = redirect_stderr  # placeholders, any callable
    ptc_impl.redirect_stderr = redirect_stderr
    package = types.ModuleType("openhands.tools.programmatic_tool_calling")
    package.impl = ptc_impl
    monkeypatch.setitem(sys.modules, "openhands.tools.programmatic_tool_calling", package)
    monkeypatch.setitem(sys.modules, "openhands.tools.programmatic_tool_calling.impl", ptc_impl)
    with isolation.activate():
        assert isolation.install_programmatic_tool_calling_isolation() is True
        assert ptc_impl.redirect_stderr is isolation.redirect_stderr
        assert ptc_impl.redirect_stdout is isolation.redirect_stdout
        # Idempotent: a second install must not re-wrap.
        assert isolation.install_programmatic_tool_calling_isolation() is False


def test_mcp_stdio_transport_pins_the_console_stderr(isolation):
    """A bridge spawn must get a real descriptor whatever ``sys.stderr`` currently is.

    Order-independent on purpose: importing ``platoon.openreward.rollout`` anywhere in the
    session installs this same pin on the (process-global) pydantic class, so the test asserts
    the resulting behaviour rather than which module instance won the race.
    """

    fastmcp_config = pytest.importorskip("fastmcp.mcp_config")
    from fastmcp.client.transports import StdioTransport

    previous = fastmcp_config.StdioMCPServer.to_transport
    try:
        with isolation.activate():
            isolation.install_mcp_stdio_stderr_pin()
            assert isolation.install_mcp_stdio_stderr_pin() is False  # idempotent
            server = fastmcp_config.StdioMCPServer(command=sys.executable, args=["-c", "pass"])
            buffer = io.StringIO()
            with redirect_stderr(buffer):  # a library that still swaps globally
                transport = server.to_transport()
                assert transport.log_file is not sys.stderr
            assert isinstance(transport, StdioTransport)
            assert transport.log_file is not buffer
            # The whole point: Popen can take this as the child's stderr.
            assert isinstance(transport.log_file.fileno(), int)
    finally:
        fastmcp_config.StdioMCPServer.to_transport = previous


def test_isolation_is_on_by_default(isolation, monkeypatch):
    monkeypatch.delenv("OPENREWARD_STREAM_ISOLATION", raising=False)
    assert isolation.enabled() is True


def test_a_later_rebinding_is_re_wrapped_and_the_first_console_is_kept(isolation):
    """Some libraries rebind ``sys.stderr`` after import; the next capture re-wraps it."""

    with isolation.activate():
        first_console = sys.stderr.console
        later = io.StringIO()  # stands in for a library that rebinds the stream
        sys.stderr = later
        assert isolation.install_stream_proxies() is True
        assert isinstance(sys.stderr, isolation.ThreadScopedStream)
        assert sys.stderr.console is later
        # The descriptor handed to MCP bridges stays the real console, not the
        # fd-less thing that wrapped it.
        assert isolation.console_stderr() is first_console
        assert isinstance(isolation.console_stderr().fileno(), int)


@pytest.mark.parametrize("value", ["0", "false", "off", "NO"])
def test_the_kill_switch_makes_install_a_no_op(isolation, monkeypatch, value):
    monkeypatch.setenv("OPENREWARD_STREAM_ISOLATION", value)
    assert isolation.enabled() is False
    saved_out, saved_err = sys.stdout, sys.stderr
    assert isolation.install() == {"programmatic_tool_calling": False, "mcp_stdio_stderr": False}
    assert sys.stdout is saved_out and sys.stderr is saved_err
