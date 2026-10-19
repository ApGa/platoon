"""Context variables reach IPython cells run with ``run_cell_async``, also through nested shells; installed files in
what cells print are named by their module paths."""

import sys
from contextvars import ContextVar

import pytest
from IPython.terminal.embed import InteractiveShellEmbed
from traitlets.config.loader import Config

from platoon.harness.builtin.codeact.ipython_shell import run_cell_async, shorten_install_paths

executor_context: ContextVar[dict] = ContextVar("executor_context")


def _create_shell():
    """Utility to create a quiet InteractiveShellEmbed instance for testing."""
    # Creating a new shell each time avoids reusing global singleton state that
    # could be modified by previous tests.  History is off, as in the codeact
    # executor: with it on, a shell writes ~/.ipython/profile_default/history.sqlite.
    config = Config()
    config.HistoryManager.enabled = False
    shell = InteractiveShellEmbed(config=config)
    shell.user_ns["executor_context"] = executor_context
    # Make sure the banner and exit messages do not pollute the test output
    shell.banner1 = ""
    shell.banner2 = ""
    shell.exit_msg = ""
    return shell


def test_executor_context_visible_in_shell():
    """Executor context set in Python should be readable inside an IPython shell."""

    test_value = {"value": "simple"}
    token = executor_context.set(test_value)
    try:
        shell = _create_shell()
        # Inside the shell, read the value from the context var
        shell.run_cell("context_value = executor_context.get()")
        assert shell.user_ns["context_value"] == test_value
    finally:
        # Restore previous context to avoid leaking state between tests
        executor_context.reset(token)


def test_executor_context_nested_shells():
    """Context changes should propagate through nested IPython shells."""

    outer_value = {"level": "outer"}
    token = executor_context.set(outer_value)
    try:
        # ---- First (outer) shell ----
        shell1 = _create_shell()
        shell1.run_cell("outer_read = executor_context.get()")
        assert shell1.user_ns["outer_read"] == outer_value

        # Update the context inside the first shell
        shell1.run_cell("executor_context.set({'level': 'shell1'})")
        shell1.run_cell("shell1_read = executor_context.get()")
        assert shell1.user_ns["shell1_read"] == {"level": "shell1"}

        # ---- Second (nested) shell ----
        shell2 = _create_shell()
        shell2.run_cell("nested_initial = executor_context.get()")
        # The nested shell should see the value set by the first shell
        assert shell2.user_ns["nested_initial"] == {"level": "shell1"}

        # Update the context again inside the nested shell
        shell2.run_cell("executor_context.set({'level': 'shell2'})")
        shell2.run_cell("nested_updated = executor_context.get()")
        assert shell2.user_ns["nested_updated"] == {"level": "shell2"}

        # After the nested shell finishes, the first shell should now see the latest value
        shell1.run_cell("after_nested = executor_context.get()")
        assert shell1.user_ns["after_nested"] == {"level": "shell2"}
    finally:
        # Always restore the original context to avoid leaking state
        executor_context.reset(token)


# ----- Asynchronous tests -----


@pytest.mark.asyncio
async def test_async_executor_context_visible_in_shell():
    """executor_context must be readable inside a shell via run_cell_async."""

    test_value = {"value": "async"}
    token = executor_context.set(test_value)
    try:
        shell = _create_shell()
        # Run cell asynchronously that writes the context value into user_ns
        await run_cell_async(shell, "async_value = executor_context.get()")
        assert shell.user_ns["async_value"] == test_value
    finally:
        executor_context.reset(token)


@pytest.mark.asyncio
async def test_async_executor_context_recursive_shell():
    """An IPython shell that spawns another shell should propagate ContextVar changes."""

    outer_value = {"level": "outer_async"}
    token = executor_context.set(outer_value)
    try:
        shell1 = _create_shell()

        # Confirm outer shell sees the initial value
        await run_cell_async(shell1, "outer_initial = executor_context.get()")
        assert shell1.user_ns["outer_initial"] == outer_value

        # Spawn a nested shell *inside* the outer shell and update the context
        await run_cell_async(
            shell1,
            "import asyncio\n"
            "from IPython.terminal.embed import InteractiveShellEmbed\n"
            "from traitlets.config.loader import Config\n"
            ""
            "from platoon.harness.builtin.codeact.ipython_shell import run_cell_async\n"
            "inner_config = Config()\n"
            "inner_config.HistoryManager.enabled = False\n"
            "inner_shell = InteractiveShellEmbed(config=inner_config)\n"
            "inner_shell.banner1 = inner_shell.banner2 = inner_shell.exit_msg = ''\n"
            "inner_shell.user_ns['executor_context'] = executor_context\n"
            # inner reads current value
            "await run_cell_async(inner_shell, 'inner_read = executor_context.get()')\n"  # noqa: E501
            # inner updates contextvar
            "await asyncio.create_task(run_cell_async(inner_shell, 'executor_context.set({\\'level\\': \\'inner_async\\'})\\ncontext_value=executor_context.get()'))\n"  # noqa: E501
            # inner verifies and outer captures after update
            "outer_during_nested1 = inner_shell.user_ns['inner_read']\n"
            "outer_during_nested2 = inner_shell.user_ns['context_value']\n"
            "outer_after_nested = executor_context.get()\n",
        )

        assert shell1.user_ns["outer_during_nested1"] == {"level": "outer_async"}
        assert shell1.user_ns["outer_during_nested2"] == {"level": "inner_async"}
        assert shell1.user_ns["outer_after_nested"] == {"level": "outer_async"}

    finally:
        executor_context.reset(token)


def test_installed_files_are_named_by_their_module_paths(monkeypatch, tmp_path):
    site, home = tmp_path / "site", tmp_path / "home"
    monkeypatch.setattr(sys, "path", [str(site), str(home / "py" / "lib"), "", "relative"])
    monkeypatch.setenv("HOME", str(home))
    text = "\n".join(
        [
            f'  File "{site}/platoon/harness/x.py", line 3, in call',  # Python's traceback
            "File ~/py/lib/json/decoder.py:338, in JSONDecoder.decode(self, s)",  # IPython's, home-relative
            f"<module 'json' from '{home}/py/lib/json/__init__.py'>",
            f"{site}/pkg/mod.py:7: UserWarning: careful",
            f"ImportError: cannot import name 'x' from 'pkg' ({site}/pkg/__init__.py)",
            # Not installed code: other directories, files that are not Python source, a prefix that is not a root
            f"cat /app/solution.py {site}/data.txt {site}x/other.py /elsewhere/tool.py ~/notes.py {site}/m.pyc",
        ]
    )

    assert shorten_install_paths(text).splitlines() == [
        '  File "platoon/harness/x.py", line 3, in call',
        "File json/decoder.py:338, in JSONDecoder.decode(self, s)",
        "<module 'json' from 'json/__init__.py'>",
        "pkg/mod.py:7: UserWarning: careful",
        "ImportError: cannot import name 'x' from 'pkg' (pkg/__init__.py)",
        f"cat /app/solution.py {site}/data.txt {site}x/other.py /elsewhere/tool.py ~/notes.py {site}/m.pyc",
    ]
