"""platoon's pytest plugin: fixtures for testing harnesses, environments and programs on CPU.

Opt in with ``pytest -p platoon.testing``, or with ``pytest_plugins = ["platoon.testing"]`` in the root
``conftest.py``; installing platoon never enables it.  The fixtures:

* ``toy_tokenizer`` and ``toy_renderer``: the toy tokenizer and ``qwen3_instruct`` over it (session-scoped);
* ``serve_policy``: ``serve_policy(policy, **options)`` serves a scripted policy as the process's policy until the
  test ends (:func:`platoon.testing.serve_policy`), and returns the ``ScriptedPolicy``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from platoon import testing


@pytest.fixture(scope="session")
def toy_tokenizer() -> Any:
    """The toy tokenizer (:func:`platoon.testing.toy_tokenizer`)."""

    return testing.toy_tokenizer()


@pytest.fixture(scope="session")
def toy_renderer(toy_tokenizer: Any) -> Any:
    """``qwen3_instruct`` over the toy tokenizer (:func:`platoon.testing.toy_renderer`)."""

    return testing.toy_renderer()


@pytest.fixture
def serve_policy() -> Iterator[Callable[..., testing.ScriptedPolicy]]:
    """``serve_policy(policy, **options)``: serve ``policy`` as this process's policy until the test ends."""

    with contextlib.ExitStack() as stack:

        def serve(policy: Any = (), **options: Any) -> testing.ScriptedPolicy:
            return stack.enter_context(testing.serve_policy(policy, **options))

        yield serve
