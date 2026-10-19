"""The named-hook registry through which extensions plug into core."""

from __future__ import annotations

import pytest

import platoon.hooks
from platoon.hooks import hooks, register


@pytest.fixture(autouse=True)
def _empty_registry(monkeypatch):
    monkeypatch.setattr(platoon.hooks, "_HOOKS", {})


def first():
    return "first"


def second():
    return "second"


def test_an_unknown_hook_has_no_functions():
    assert hooks("nothing.registered") == []


def test_functions_are_returned_in_registration_order():
    register("seam", first)
    register("seam", second)
    register("other.seam", second)

    assert hooks("seam") == [first, second]
    assert hooks("other.seam") == [second]


def test_registering_a_function_again_keeps_one_registration():
    """Importing a module twice must not double what its hooks do."""

    register("seam", first)
    register("seam", first)
    assert hooks("seam") == [first]


def test_the_returned_list_is_a_copy():
    register("seam", first)
    hooks("seam").append(second)
    assert hooks("seam") == [first]
