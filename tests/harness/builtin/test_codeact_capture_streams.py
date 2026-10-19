"""``ShellCapture`` puts back the streams it replaced, not the ones that were current when platoon was imported.

Whatever redirects ``sys.stdout`` after the import (pytest's capture, ``contextlib.redirect_stdout`` in an
application that embeds platoon) keeps its stream across a CodeAct cell.
"""

from __future__ import annotations

import contextlib
import contextvars
import io
import sys
import threading

from platoon.harness.builtin.codeact.ipython_shell import ShellCapture


def test_a_capture_restores_the_streams_current_when_it_opened(monkeypatch):
    redirected, redirected_err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", redirected)
    monkeypatch.setattr(sys, "stderr", redirected_err)

    with ShellCapture(capture_display=False) as capture:
        print("in the cell")
        print("cell error", file=sys.stderr)
    print("after the cell")
    print("error after the cell", file=sys.stderr)

    assert sys.stdout is redirected and sys.stderr is redirected_err
    assert (capture.pop_stdout(), capture.pop_stderr()) == ("in the cell\n", "cell error\n")
    assert (redirected.getvalue(), redirected_err.getvalue()) == ("after the cell\n", "error after the cell\n")


def test_a_capture_inside_redirect_stdout_leaves_the_redirect_in_place():
    redirected = io.StringIO()
    with contextlib.redirect_stdout(redirected):
        with ShellCapture(capture_display=False) as capture:
            print("in the cell")
        assert sys.stdout is redirected
        print("after the cell")

    assert capture.pop_stdout() == "in the cell\n"
    assert redirected.getvalue() == "after the cell\n"


def test_output_outside_any_capture_reaches_the_stream_current_when_the_capture_opened(monkeypatch):
    redirected = io.StringIO()
    monkeypatch.setattr(sys, "stdout", redirected)

    with ShellCapture(capture_display=False) as capture:
        # A write from a context without a capture (another task or thread) is not the cell's.
        contextvars.Context().run(lambda: print("not in a cell"))

    assert capture.pop_stdout() == ""
    assert redirected.getvalue() == "not in a cell\n"


def test_a_stream_installed_while_a_capture_is_open_stays(monkeypatch):
    first, second = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", first)

    with ShellCapture(capture_display=False):
        sys.stdout = second  # the application redirects while a cell runs

    assert sys.stdout is second


def test_overlapping_captures_restore_the_stream_once_the_last_one_closes(monkeypatch):
    redirected = io.StringIO()
    monkeypatch.setattr(sys, "stdout", redirected)
    opened, release = threading.Event(), threading.Event()
    outputs: dict[str, str] = {}

    def other_cell() -> None:  # another agent's cell on its own cell-loop thread
        with ShellCapture(capture_display=False) as capture:
            opened.set()
            release.wait(timeout=10)
            print("other cell")
        outputs["other"] = capture.pop_stdout()

    thread = threading.Thread(target=other_cell)
    thread.start()
    assert opened.wait(timeout=10)
    with ShellCapture(capture_display=False) as capture:
        print("this cell")
    outputs["this"] = capture.pop_stdout()
    assert sys.stdout is not redirected  # the other cell is still capturing
    release.set()
    thread.join(timeout=10)

    assert outputs == {"this": "this cell\n", "other": "other cell\n"}
    assert sys.stdout is redirected
    print("after both cells")
    assert redirected.getvalue() == "after both cells\n"
