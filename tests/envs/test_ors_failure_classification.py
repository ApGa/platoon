"""How an ORS session classifies a tool that raised, by what each mode sees of it.

In process the exception itself reaches platoon; an env server sends only its text, wrapped by the openreward
client: 0.1.163 raises ``ToolFailed`` and ends the session, 0.1.125 raises a ``RuntimeError`` and keeps it.  The
SDK's own session and transport errors are infrastructure in both modes; any other raise is the policy's, a class
of another library that shares an SDK error's name included.
"""

from __future__ import annotations

import importlib

import httpx
import pytest

from platoon.envs.ors import sdk
from platoon.envs.ors.session import _failure


class SessionTerminatedError(RuntimeError):
    """Named as the SDK names it, but not the SDK's: what the client raises remotely is classified by its name."""


class TransportError(Exception):
    """Named as the SDK names it, but not the SDK's."""


class ToolFailed(Exception):
    """0.1.163's client error for a tool that raised on the server: it also ends the session."""


SANDBOX_GONE = "sandbox session terminated (sid='sb-1'): server returned 410: Gone"
RETRIES = "Exceeded 5 retries for /exec\n  Encountered 1 error(s):\n    [1] ConnectionResetError: reset by peer"
RETRIES_SENT = (
    "Exceeded 5 retries for /exec\nEncountered 1 error(s):\n[1] ConnectionResetError: reset by peer"  # as SSE
)
PREFIX = "(environment session sid=0123) "  # the client's annotation of a session call's error


def _sdk_class(name: str) -> type | None:
    """The openreward SDK's own class ``name``: 0.1.163 keeps its errors in ``openreward.api.errors``, 0.1.125 in its
    session and HTTP modules; None when the installed SDK has none (0.1.125 has no ``TransportError``)."""

    for module in ("openreward.api.errors", "openreward.api._session.session", "openreward.api._session.http"):
        try:
            found = getattr(importlib.import_module(module), name, None)
        except ImportError:
            continue
        if isinstance(found, type):
            return found
    return None


def _sdk_error(name: str, *args, **kwargs) -> BaseException:
    cls = _sdk_class(name)
    if cls is None:
        pytest.skip(f"the installed openreward SDK has no {name}")
    return cls(*args, **kwargs)


SDK_ERRORS = {
    "SessionTerminatedError": lambda: _sdk_error(
        "SessionTerminatedError", "server returned 410: Gone", sid="sb-1", kind="sandbox"
    ),
    "TransportError": lambda: _sdk_error("TransportError", "judge unavailable"),
    "HeartbeatTimeoutError": lambda: _sdk_error("HeartbeatTimeoutError"),
    "MaxRetriesError": lambda: _sdk_error(
        "MaxRetriesError", "Exceeded 5 retries for /exec", errors=[ConnectionResetError("reset by peer")]
    ),
}


@pytest.mark.parametrize("name", sorted(SDK_ERRORS))
def test_in_process_the_sdks_own_session_and_transport_errors_are_infrastructure(name):
    exc = SDK_ERRORS[name]()

    assert sdk.infra_error(exc, remote=False)
    assert _failure(exc, remote=False) == ("env_infra", True, False)


def test_in_process_a_subclass_of_an_sdk_error_is_infrastructure():
    class BackendRetriesExhausted(_sdk_class("MaxRetriesError")):
        pass

    assert _failure(BackendRetriesExhausted("Exceeded 3 retries for /run"), remote=False) == ("env_infra", True, False)


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("no such item"),
        RuntimeError(SANDBOX_GONE),  # in process the class decides, not the text
        TimeoutError(),
        SessionTerminatedError(SANDBOX_GONE),  # the SDK's name, another module's class
        TransportError("judge unavailable"),
        httpx.ConnectError("All connection attempts failed"),  # httpx's TransportError, not the SDK's
        httpx.ReadTimeout("timed out"),
    ],
    ids=[
        "ValueError",
        "RuntimeError",
        "TimeoutError",
        "named-SessionTerminatedError",
        "named-TransportError",
        "httpx-ConnectError",
        "httpx-ReadTimeout",
    ],
)
def test_in_process_any_other_exception_is_the_policys_failed_call(exc):
    """An env server would send these as text that no SDK error has: they must not be infrastructure here either."""

    assert not sdk.infra_error(exc, remote=False)
    assert _failure(exc, remote=False) == ("tool_raised", False, False)


def test_in_process_an_exception_raised_from_an_infrastructure_error_is_the_tools_own():
    cause = SDK_ERRORS["SessionTerminatedError"]()
    try:
        try:
            raise cause
        except type(cause) as caught:
            raise RuntimeError("could not run the tests") from caught
    except RuntimeError as exc:
        raised = exc
    assert not sdk.infra_error(raised, remote=False)  # an env server would send "could not run the tests" only


@pytest.mark.parametrize("client_error", [ToolFailed, RuntimeError], ids=["0.1.163", "0.1.125"])
@pytest.mark.parametrize(
    ("text", "infra"),
    [
        (SANDBOX_GONE, True),
        ("environment session terminated (sid=None): gone", True),
        (RETRIES, True),
        (RETRIES_SENT, True),
        ("boom", False),
        ("Unknown SSE error", False),  # an empty text: a bare assert, a TimeoutError() or a HeartbeatTimeoutError()
        ("the sandbox session terminated early", False),
    ],
)
def test_remotely_the_text_the_env_server_sent_decides(client_error, text, infra):
    exc = client_error(PREFIX + text)
    kills = client_error is ToolFailed  # 0.1.163 ends the session whatever the tool raised

    assert sdk.infra_error(exc, remote=True) is infra
    assert _failure(exc, remote=True) == (("env_infra", True, kills) if infra else ("tool_raised", False, kills))


def test_remote_session_and_transport_failures_stay_infrastructure():
    dead = SessionTerminatedError("environment session terminated (sid='s'): server returned 410")
    assert _failure(dead, remote=True) == ("dead", True, True)  # the client's own: the session is gone
    assert _failure(ConnectionResetError("reset"), remote=True) == ("transport", True, False)
    assert _failure(RuntimeError(PREFIX + "unknown task_id"), remote=True) == ("transport", True, False)
