"""Settings for the whole test suite: ``tests/conftest.py`` star-imports them, and so do the test suites of the
first-party extensions (``extensions/*/tests/conftest.py``), which run in the same way."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator

import pytest

#: The environment variables a test must leave as it found them, beside ``sys.executable`` and ``sys.path``: the
#: rollout worker marker, with which an in-process rollout enters the Python overlay of a worker process
#: (``platoon.train.tinker.workers.run``), and the variables that overlay sets (``platoon.utils.rollout_python_env``).
PROCESS_ENVIRONMENT = (
    "PLATOON_TINKER_ROLLOUT_WORKER",
    "PATH",
    "PIP_REQUIRE_VIRTUALENV",
    "PYTHONNOUSERSITE",
    "UV_CACHE_DIR",
    "UV_PROJECT_ENVIRONMENT",
    "VIRTUAL_ENV",
)


@pytest.fixture(autouse=True, scope="session")
def _ipython_dir_outside_home(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Keep IPython's profile out of HOME: every IPython shell (CodeAct runs its cells in them) creates a profile
    directory under ``IPYTHONDIR``, by default ``~/.ipython``, so the suite points it at a temporary directory."""

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("IPYTHONDIR", str(tmp_path_factory.mktemp("ipython")))
        yield


@pytest.fixture(autouse=True, scope="session")
def _no_site_profile(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Keep the machine's site profile (``~/.config/platoon/site.yaml``, ``$PLATOON_SITE``) and the deprecated cache
    variables out of the suite, which runs as if the site had none: a test that wants a profile writes one and names it
    in ``PLATOON_SITE`` (``platoon.site``)."""

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg-config")))
        for name in ("PLATOON_SITE", "PLATOON_HARBOR_CACHE", "PLATOON_TOOLCHAINS"):
            patch.delenv(name, raising=False)
        yield


@pytest.fixture(autouse=True)
def _sse_starlette_exit_state_per_test() -> Iterator[None]:
    """Give every test sse_starlette's exit state afresh. sse_starlette keeps it in class attributes of
    ``AppStatus``: ``should_exit_event`` stays bound to the event loop of the first server that streamed in the
    process, and ``should_exit`` stays set once a server shut down. A later test's environment server (the ORS parity,
    remote-environment and concurrency tests run one each) then ends its SSE streams at once, depending on which tests
    ran before it in the same process."""

    try:
        from sse_starlette.sse import AppStatus
    except ImportError:  # only the environment-server tests need sse_starlette
        yield
        return
    AppStatus.should_exit = False
    AppStatus.should_exit_event = None
    yield
    AppStatus.should_exit = False
    AppStatus.should_exit_event = None


@pytest.fixture(autouse=True)
def _interpreter_left_as_found() -> Iterator[None]:
    """Fail the test that leaves ``sys.executable``, ``sys.path`` or a variable of :data:`PROCESS_ENVIRONMENT`
    changed, and put them back.

    A process runs many tests (each pytest-xdist worker a few thousand), so what one test leaves changed breaks a
    later one that did nothing wrong, in whichever process happens to run both: a worker marker left in
    ``os.environ`` made the in-process rollouts of a later test enter the per-process Python overlay concurrently,
    which left ``sys.executable`` naming an overlay already removed, and the tests after that which started a rollout
    worker failed.  This fails the test that changed it, after its fixtures have put back what they patched.

    ``sys.path`` counts as changed when an entry is put in among its entries, taken away or moved.  Entries appended
    after them are taken away without failing the test: they shadow nothing, and importing litellm appends the working
    directory, in whichever test imports it first."""

    executable, path = sys.executable, list(sys.path)
    environment = {name: os.environ.get(name) for name in PROCESS_ENVIRONMENT}
    yield
    changed = []
    if sys.executable != executable:
        changed.append(f"sys.executable {executable!r} -> {sys.executable!r}")
        sys.executable = executable
    if sys.path != path:
        if sys.path[: len(path)] != path:
            added = [entry for entry in sys.path if entry not in path]
            removed = [entry for entry in path if entry not in sys.path]
            changed.append(f"sys.path (added {added}, removed {removed})" if added or removed else "sys.path's order")
        sys.path[:] = path
    for name, value in environment.items():
        now = os.environ.get(name)
        if now != value:
            changed.append(f"{name} {value!r} -> {now!r}")
            if value is None:
                del os.environ[name]
            else:
                os.environ[name] = value
    if changed:
        pytest.fail(f"the test left {'; '.join(changed)} (now put back)", pytrace=False)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "serial: cannot share the machine with other tests, so it runs in a pass of its own, after the parallel one "
        "(CONTRIBUTING.md, Run the tests)",
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """The serial pass (``-m serial``) passes when no test is marked ``serial``: pytest fails a run that selects no
    test."""

    if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED and session.config.option.markexpr == "serial":
        session.exitstatus = pytest.ExitCode.OK


__all__ = [
    "_interpreter_left_as_found",
    "_ipython_dir_outside_home",
    "_no_site_profile",
    "_sse_starlette_exit_state_per_test",
    "pytest_configure",
    "pytest_sessionfinish",
]
